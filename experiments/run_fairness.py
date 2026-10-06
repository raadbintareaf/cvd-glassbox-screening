"""Fairness audit + mitigation arms for one (tier, model, seed).

Arms:
  baseline    — single validation-selected threshold (screening rule)
  group_thr   — per-group thresholds (validation-selected)      [any model]
  reweigh     — refit with Kamiran-Calders reweighing            [any model]
  repair_zero — EBM only: neutralize Sex terms (+ interactions)
  repair_half — EBM only: attenuate Sex terms (gamma=0.5)
  repair_eq   — EBM only: neutralize Sex + per-group intercept re-centering

Each arm reports the subgroup audit table and appends frontier rows
(overall sensitivity/specificity + max subgroup TPR gap) to raw_results.

Usage:
  python -m experiments.run_fairness --config configs/default.yaml \
      --tier T1 --model ebm --seed 0 --group Sex
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.variable_map import OUTCOME_LABEL, SUBGROUP_LABELS, tier_labels
from src.evaluation.fairness import (apply_group_thresholds,
                                     equalized_odds_gaps, group_thresholds,
                                     reweighing_weights, subgroup_audit)
from src.evaluation.metrics import confusion_at, threshold_for
from src.models.registry import make_model
from src.utils.common import (append_rows, load_config, seed_everything,
                              stratified_splits)
from experiments.run_benchmark import load_analytic, model_params


def frontier_row(y, yhat, groups, arm):
    tp = ((y == 1) & (yhat == 1)).sum()
    fn = ((y == 1) & (yhat == 0)).sum()
    tn = ((y == 0) & (yhat == 0)).sum()
    fp = ((y == 0) & (yhat == 1)).sum()
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    gaps = []
    for lvl in groups.dropna().unique():
        m = (groups == lvl).values
        tpl = ((y == 1) & (yhat == 1) & m).sum()
        fnl = ((y == 1) & (yhat == 0) & m).sum()
        if tpl + fnl >= 5:
            gaps.append(tpl / (tpl + fnl))
    gap = (max(gaps) - min(gaps)) if len(gaps) >= 2 else np.nan
    return dict(arm=arm, sensitivity=float(sens), specificity=float(spec),
                tpr_gap=float(gap))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--tier", required=True)
    ap.add_argument("--model", default="ebm")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--group", default="Sex")
    a = ap.parse_args()
    cfg = load_config(a.config)
    dataset = cfg.get("dataset", "brfss2022")
    seed_everything(a.seed)

    df = load_analytic(cfg, dataset)
    feats = tier_labels(a.tier, drop_stroke=cfg.get("drop_stroke", False))
    X, y = df[feats], df[OUTCOME_LABEL].values.astype(int)
    G = df[SUBGROUP_LABELS]
    idx = stratified_splits(y, a.seed, tuple(cfg["split_fracs"]))
    Xtr, ytr = X.iloc[idx["train"]], y[idx["train"]]
    Xva, yva = X.iloc[idx["val"]], y[idx["val"]]
    Xte, yte = X.iloc[idx["test"]], y[idx["test"]]
    gva = G[a.group].iloc[idx["val"]].reset_index(drop=True)
    gte = G[a.group].iloc[idx["test"]].reset_index(drop=True)
    Gte = G.iloc[idx["test"]].reset_index(drop=True)

    rule = next(r for r in cfg["operating_rules"]
                if r["name"] == cfg.get("fairness_rule", "screening"))
    params = model_params(cfg, a.model, a.tier)

    base = make_model(a.model, params, a.seed).fit(Xtr, ytr, Xva, yva)
    p_va, p_te = (base.predict_proba_pos(Xva), base.predict_proba_pos(Xte))
    thr = threshold_for(yva, p_va, rule["rule"], rule["target"])

    out_dir = Path(cfg["results_dir"]) / "fairness" / dataset / a.tier / \
        a.model / f"seed{a.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    frontier, rows = [], []

    # --- baseline: single validation-selected threshold
    aud = subgroup_audit(yte, p_te, Gte, thr)
    aud.to_csv(out_dir / "audit_baseline.csv", index=False)
    equalized_odds_gaps(aud).to_csv(out_dir / "gaps_baseline.csv",
                                    index=False)
    frontier.append(frontier_row(yte, (p_te >= thr).astype(int), gte,
                                 "baseline"))

    # --- group thresholds (validation-selected per group)
    thrs = group_thresholds(yva, p_va, gva, rule["rule"], rule["target"])
    yhat_gt = apply_group_thresholds(p_te, gte, thrs, thr)
    frontier.append(frontier_row(yte, yhat_gt, gte, "group_thr"))
    pd.DataFrame({"group": list(thrs), "threshold": list(thrs.values())}
                 ).to_csv(out_dir / "group_thresholds.csv", index=False)

    # --- reweighing refit
    w = reweighing_weights(ytr, G[a.group].iloc[idx["train"]]
                           .reset_index(drop=True))
    rw = make_model(a.model, params, a.seed)
    rw.extra_weight_ = w  # consumed by adapters with native sample-weight support
    try:
        # adapters use balanced weights internally; reweighing multiplies in
        from src.models import registry as R
        orig = R.balanced_sample_weight
        R.balanced_sample_weight = lambda yy, _w=w, _o=orig: _o(yy) * _w
        rw.fit(Xtr, ytr, Xva, yva)
    finally:
        R.balanced_sample_weight = orig
    p_va_rw, p_te_rw = rw.predict_proba_pos(Xva), rw.predict_proba_pos(Xte)
    thr_rw = threshold_for(yva, p_va_rw, rule["rule"], rule["target"])
    aud_rw = subgroup_audit(yte, p_te_rw, Gte, thr_rw)
    aud_rw.to_csv(out_dir / "audit_reweigh.csv", index=False)
    frontier.append(frontier_row(yte, (p_te_rw >= thr_rw).astype(int), gte,
                                 "reweigh"))

    # --- EBM-only glass-box repairs
    if a.model == "ebm":
        from src.repair.ebm_repair import (equalize_group_intercept,
                                           neutralize_terms, scale_terms)
        from src.utils.common import save_json
        for arm, maker in (
                ("repair_zero", lambda: neutralize_terms(base.model_,
                                                         [a.group])),
                ("repair_half", lambda: scale_terms(base.model_, [a.group],
                                                    0.5))):
            edited, log = maker()
            save_json(log, str(out_dir / f"editlog_{arm}.json"))
            p_va_e = edited.predict_proba(Xva)[:, 1]
            p_te_e = edited.predict_proba(Xte)[:, 1]
            thr_e = threshold_for(yva, p_va_e, rule["rule"], rule["target"])
            subgroup_audit(yte, p_te_e, Gte, thr_e).to_csv(
                out_dir / f"audit_{arm}.csv", index=False)
            frontier.append(frontier_row(yte, (p_te_e >= thr_e).astype(int),
                                         gte, arm))
        edited, log0 = neutralize_terms(base.model_, [a.group])
        wrapped, log1 = equalize_group_intercept(
            edited, Xva, yva, gva)
        save_json({"stage1": log0, "stage2": log1},
                  str(out_dir / "editlog_repair_eq.json"))
        p_va_q = wrapped.predict_proba_pos(Xva, gva)
        p_te_q = wrapped.predict_proba_pos(Xte, gte)
        thr_q = threshold_for(yva, p_va_q, rule["rule"], rule["target"])
        subgroup_audit(yte, p_te_q, Gte, thr_q).to_csv(
            out_dir / "audit_repair_eq.csv", index=False)
        frontier.append(frontier_row(yte, (p_te_q >= thr_q).astype(int), gte,
                                     "repair_eq"))

    fr = pd.DataFrame(frontier)
    fr.to_csv(out_dir / "frontier.csv", index=False)
    results_csv = str(Path(cfg["results_dir"]) / "raw_results.csv")
    append_rows(results_csv, [
        dict(run_id=f"fair-{dataset}-{a.tier}-{a.model}-s{a.seed}",
             stage="fairness", dataset=dataset, tier=a.tier, model=a.model,
             seed=a.seed, split="test", subgroup=a.group,
             metric=f"{r['arm']}_{k}", value=v, extra="")
        for r in frontier for k, v in r.items() if k != "arm"])
    print(fr.to_string(index=False))


if __name__ == "__main__":
    main()
