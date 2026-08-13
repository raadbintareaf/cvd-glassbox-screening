"""Main benchmark: one (tier, model, seed) per invocation; resumable.

Usage:
  python -m experiments.run_benchmark --config configs/default.yaml \
      --tier T1 --model ebm --seed 0
  python -m experiments.run_benchmark --config configs/smoke.yaml \
      --tier T1 --model xgboost --seed 0          # demo-scale

Outputs:
  results/raw_results.csv                      (append-only, long format)
  results/predictions/<ds>/<tier>/<model>/seed<k>.parquet
  results/models/<ds>/<tier>/<model>/seed<k>/  (CPU models only)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.variable_map import OUTCOME_LABEL, SUBGROUP_LABELS, tier_labels
from src.evaluation.metrics import (all_probability_metrics, confusion_at,
                                    threshold_for)
from src.models.registry import make_model
from src.utils.common import (append_rows, completed_keys, load_config,
                              seed_everything, stratified_splits)


def load_analytic(cfg, dataset: str) -> pd.DataFrame:
    path = Path(cfg["data_dir"]) / f"brfss_{dataset[-4:]}_analytic.parquet"
    if not path.exists():
        raise SystemExit(f"{path} missing — run src.data.build_dataset first "
                         f"(see README step 2).")
    df = pd.read_parquet(path)
    if cfg.get("limit"):
        df = df.sample(n=min(cfg["limit"], len(df)),
                       random_state=0).reset_index(drop=True)
    return df


def model_params(cfg, model: str, tier: str) -> dict:
    tuned = Path(cfg["results_dir"]) / "tuned" / f"{model}__{tier}.yaml"
    if tuned.exists():
        with open(tuned) as f:
            return yaml.safe_load(f) or {}
    default = Path("configs/models") / f"{model}.yaml"
    if default.exists():
        with open(default) as f:
            return (yaml.safe_load(f) or {}).get("defaults", {})
    return {}


def maybe_smotenc(X: pd.DataFrame, y: np.ndarray, seed: int):
    """Explicit ablation arm only (config imbalance: smotenc). Applied to the
    TRAINING split alone — the leakage-free way (cf. Eltawil et al. 2026)."""
    from imblearn.over_sampling import SMOTENC

    from src.models.registry import to_codes_matrix
    M, cat_idx = to_codes_matrix(X)
    col_medians = np.nanmedian(M, axis=0)
    inds = np.where(np.isnan(M))
    M[inds] = np.take(col_medians, inds[1])  # SMOTE cannot handle NaN
    sm = SMOTENC(categorical_features=cat_idx, random_state=seed)
    Mr, yr = sm.fit_resample(M, y)
    Xr = pd.DataFrame(Mr, columns=X.columns)
    for j, c in enumerate(X.columns):
        if j in cat_idx:
            codes = Xr[c].round().astype(int).clip(
                0, len(X[c].cat.categories) - 1)
            Xr[c] = pd.Categorical.from_codes(codes,
                                              categories=X[c].cat.categories)
    return Xr, yr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--tier", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--imbalance", choices=["class_weight", "smotenc"],
                    default="class_weight")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    cfg = load_config(a.config)
    dataset = cfg.get("dataset", "brfss2022")
    results_csv = str(Path(cfg["results_dir"]) / "raw_results.csv")
    stage = "benchmark" if a.imbalance == "class_weight" else "smotenc_abl"
    model_tag = a.model if a.imbalance == "class_weight" \
        else f"{a.model}+smotenc"
    key = (stage, dataset, a.tier, model_tag, str(a.seed))
    if not a.force and key in completed_keys(results_csv):
        print(f"[skip] {key} already in results")
        return

    seed_everything(a.seed)
    df = load_analytic(cfg, dataset)
    feats = tier_labels(a.tier, drop_stroke=cfg.get("drop_stroke", False))
    X_all, y_all = df[feats], df[OUTCOME_LABEL].values.astype(int)
    groups_all = df[SUBGROUP_LABELS]

    idx = stratified_splits(y_all, a.seed, tuple(cfg["split_fracs"]))
    Xtr, ytr = X_all.iloc[idx["train"]], y_all[idx["train"]]
    Xva, yva = X_all.iloc[idx["val"]], y_all[idx["val"]]
    Xte, yte = X_all.iloc[idx["test"]], y_all[idx["test"]]

    if a.imbalance == "smotenc":
        Xtr, ytr = maybe_smotenc(Xtr, ytr, a.seed)

    params = model_params(cfg, a.model, a.tier)
    t0 = time.time()
    m = make_model(a.model, params, a.seed)
    m.fit(Xtr, ytr, Xva, yva)
    fit_s = time.time() - t0

    t0 = time.time()
    p_va = m.predict_proba_pos(Xva)
    p_te = m.predict_proba_pos(Xte)
    pred_s = time.time() - t0

    # thresholds selected on VALIDATION only (pre-specified rules)
    thr = {r["name"]: threshold_for(yva, p_va, r["rule"], r["target"])
           for r in cfg["operating_rules"]}

    # ---- persist predictions (the substrate for all downstream analyses)
    pred_dir = (Path(cfg["results_dir"]) / "predictions" / dataset / a.tier /
                model_tag)
    pred_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for split, ii, ys, ps in (("val", idx["val"], yva, p_va),
                              ("test", idx["test"], yte, p_te)):
        out = pd.DataFrame({"row_id": ii, "split": split, "y": ys, "p": ps})
        out = pd.concat([out.reset_index(drop=True),
                         groups_all.iloc[ii].reset_index(drop=True)], axis=1)
        rows.append(out)
    pd.concat(rows).to_parquet(pred_dir / f"seed{a.seed}.parquet",
                               index=False)

    if not getattr(m, "requires_gpu", False):
        m.save(str(Path(cfg["results_dir"]) / "models" / dataset / a.tier /
                   model_tag / f"seed{a.seed}"))

    # ---- metrics rows
    out_rows = []
    base = dict(run_id=f"{dataset}-{a.tier}-{model_tag}-s{a.seed}",
                stage=stage, dataset=dataset, tier=a.tier, model=model_tag,
                seed=a.seed, subgroup="ALL")
    for split, ys, ps in (("val", yva, p_va), ("test", yte, p_te)):
        for k, v in all_probability_metrics(ys, ps).items():
            out_rows.append({**base, "split": split, "metric": k, "value": v})
        for rule_name, t in thr.items():
            for k, v in confusion_at(ys, ps, t).items():
                out_rows.append({**base, "split": split,
                                 "metric": f"{rule_name}_{k}", "value": v,
                                 "extra": f"thr={t:.6f}"})
    out_rows.append({**base, "split": "train", "metric": "fit_seconds",
                     "value": fit_s})
    out_rows.append({**base, "split": "test", "metric": "predict_seconds",
                     "value": pred_s})
    out_rows.append({**base, "split": "train", "metric": "n_train",
                     "value": len(ytr)})
    append_rows(results_csv, out_rows)
    print(f"[done] {key} auroc_test="
          f"{[r['value'] for r in out_rows if r['metric']=='auroc' and r['split']=='test'][0]:.4f} "
          f"fit={fit_s:.1f}s")


if __name__ == "__main__":
    main()
