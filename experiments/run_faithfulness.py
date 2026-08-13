"""Explanation-faithfulness sweep (contribution C3).

Reference: exact global importances of the trained EBM (loaded from the
benchmark run). Comparators at each explanation-sample size n in the sweep,
repeated n_resamples times on random test subsamples:

  xgb_treeshap   — TreeSHAP on the trained XGBoost      (fast, all sizes)
  ebm_kernelshap — KernelSHAP treating the EBM as a black box (sanity check:
                   how well does a model-agnostic explainer recover the exact
                   importances?)                         (sizes <= 2000)
  xgb_lime       — LIME on XGBoost                       (sizes <= 500)
  fm_kernelshap  — KernelSHAP on TabPFN/TabICL           (--include-fm, GPU)

Usage:
  python -m experiments.run_faithfulness --config configs/default.yaml \
      --tier T1 --seed 0 [--include-fm]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.variable_map import OUTCOME_LABEL, tier_labels
from src.evaluation.faithfulness import (agreement, ebm_global_importance,
                                         lime_importance,
                                         shap_kernel_importance,
                                         shap_tree_importance)
from src.models.registry import EBMAdapter, XGBAdapter, make_model
from src.utils.common import (append_rows, load_config, seed_everything,
                              stratified_splits)
from experiments.run_benchmark import load_analytic, model_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--tier", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--include-fm", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    dataset = cfg.get("dataset", "brfss2022")
    seed_everything(a.seed)

    df = load_analytic(cfg, dataset)
    feats = tier_labels(a.tier, drop_stroke=cfg.get("drop_stroke", False))
    X, y = df[feats], df[OUTCOME_LABEL].values.astype(int)
    idx = stratified_splits(y, a.seed, tuple(cfg["split_fracs"]))
    Xte = X.iloc[idx["test"]].reset_index(drop=True)

    mdir = Path(cfg["results_dir"]) / "models" / dataset / a.tier
    ebm = EBMAdapter.load(str(mdir / "ebm" / f"seed{a.seed}"))
    xgb = XGBAdapter.load(str(mdir / "xgboost" / f"seed{a.seed}"))
    ref = ebm_global_importance(ebm.model_, feats)

    fm = None
    if a.include_fm:
        Xtr = X.iloc[idx["train"]]
        ytr = y[idx["train"]]
        fm_name = cfg.get("faithfulness_fm", "tabicl")
        fm = make_model(fm_name, model_params(cfg, fm_name, a.tier),
                        a.seed).fit(Xtr, ytr)

    sweep = cfg["faithfulness"]["sizes"]
    n_res = cfg["faithfulness"]["n_resamples"]
    lime_cap = cfg["faithfulness"].get("lime_max_size", 500)
    lime_res = cfg["faithfulness"].get("lime_resamples", 5)
    kshap_cap = cfg["faithfulness"].get("kernelshap_max_size", 2000)

    rng = np.random.default_rng(a.seed)
    rows, raw = [], []
    out_dir = (Path(cfg["results_dir"]) / "faithfulness" / dataset / a.tier /
               f"seed{a.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)

    def add(comparator, n, r, imp):
        ag = agreement(ref, imp, k=cfg["faithfulness"].get("top_k", 10))
        raw.append({"comparator": comparator, "n_explain": n, "resample": r,
                    **ag})

    for n in sweep:
        for r in range(n_res):
            sub = Xte.iloc[rng.choice(len(Xte), min(n, len(Xte)),
                                      replace=False)]
            add("xgb_treeshap", n, r, shap_tree_importance(xgb, sub))
            if n <= kshap_cap:
                add("ebm_kernelshap", n, r, shap_kernel_importance(
                    ebm.predict_proba_pos, sub.iloc[:min(n, 200)],
                    seed=a.seed + r))
                if fm is not None:
                    add("fm_kernelshap", n, r, shap_kernel_importance(
                        fm.predict_proba_pos, sub.iloc[:min(n, 200)],
                        seed=a.seed + r))
            if n <= lime_cap and r < lime_res:
                add("xgb_lime", n, r, lime_importance(
                    xgb.predict_proba_pos, sub.iloc[:min(n, 50)], Xte,
                    seed=a.seed + r))

    raw_df = pd.DataFrame(raw)
    raw_df.to_csv(out_dir / "agreement_raw.csv", index=False)
    ref.sort_values(ascending=False).to_csv(out_dir / "ebm_reference.csv")

    results_csv = str(Path(cfg["results_dir"]) / "raw_results.csv")
    agg = raw_df.groupby(["comparator", "n_explain"]).agg(
        tau_mean=("kendall_tau", "mean"), tau_std=("kendall_tau", "std"),
        jac_mean=(f"jaccard_at_{cfg['faithfulness'].get('top_k', 10)}",
                  "mean")).reset_index()
    append_rows(results_csv, [
        dict(run_id=f"faith-{dataset}-{a.tier}-s{a.seed}",
             stage="faithfulness", dataset=dataset, tier=a.tier,
             model=r.comparator, seed=a.seed, split="test",
             subgroup=f"n={int(r.n_explain)}", metric=m, value=getattr(r, m),
             extra="")
        for r in agg.itertuples() for m in ("tau_mean", "tau_std",
                                            "jac_mean")])
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
