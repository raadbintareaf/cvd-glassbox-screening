"""Aggregate raw_results.csv -> summary.csv (+ delong.csv).

Pure function of the results directory: re-run any time more seeds arrive.

  summary.csv : mean +/- std over seeds per (stage,dataset,tier,model,split,
                subgroup,metric) + n_seeds
  wilcoxon.csv: across-seed paired Wilcoxon of every model vs EBM on test
                AUROC/AUPRC per tier (effect = mean difference)
  delong.csv  : seed-0 paired DeLong AUC tests EBM vs each model per tier,
                Holm-adjusted (single canonical split; predictions parquet)

Usage: python -m analysis.aggregate_results --config configs/default.yaml
"""
from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.evaluation.metrics import delong_test, holm
from src.utils.common import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    a = ap.parse_args()
    cfg = load_config(a.config)
    rdir = Path(cfg["results_dir"])
    raw = pd.read_csv(rdir / "raw_results.csv")

    key = ["stage", "dataset", "tier", "model", "split", "subgroup", "metric"]
    summ = (raw.groupby(key, dropna=False)["value"]
            .agg(mean="mean", std="std", n_seeds="count").reset_index())
    summ.to_csv(rdir / "summary.csv", index=False)
    print(f"[aggregate] summary.csv: {len(summ)} rows")

    # ---- across-seed Wilcoxon vs EBM (test AUROC / AUPRC)
    rows = []
    bench = raw[(raw.stage == "benchmark") & (raw.split == "test")
                & (raw.subgroup == "ALL")]
    for (ds, tier, metric) in product(bench.dataset.unique(),
                                      bench.tier.unique(),
                                      ["auroc", "auprc"]):
        sub = bench[(bench.dataset == ds) & (bench.tier == tier)
                    & (bench.metric == metric)]
        piv = sub.pivot_table(index="seed", columns="model", values="value")
        if "ebm" not in piv.columns:
            continue
        for m in piv.columns:
            if m == "ebm" or piv[m].isna().all():
                continue
            paired = piv[["ebm", m]].dropna()
            if len(paired) < 3:
                continue
            try:
                w = stats.wilcoxon(paired["ebm"], paired[m])
                pval = float(w.pvalue)
            except ValueError:
                pval = 1.0
            rows.append({"dataset": ds, "tier": tier, "metric": metric,
                         "model": m, "n_seeds": len(paired),
                         "ebm_mean": paired["ebm"].mean(),
                         "model_mean": paired[m].mean(),
                         "mean_diff_ebm_minus_model":
                             float((paired["ebm"] - paired[m]).mean()),
                         "wilcoxon_p": pval})
    wil = pd.DataFrame(rows, columns=["dataset", "tier", "metric",
        "model", "n_seeds", "ebm_mean", "model_mean",
        "mean_diff_ebm_minus_model", "wilcoxon_p"])
    if len(wil):
        wil["wilcoxon_p_holm"] = np.nan
        for (ds, tier, metric), g in wil.groupby(["dataset", "tier",
                                                  "metric"]):
            wil.loc[g.index, "wilcoxon_p_holm"] = holm(
                g["wilcoxon_p"].tolist())
    wil.to_csv(rdir / "wilcoxon.csv", index=False)
    print(f"[aggregate] wilcoxon.csv: {len(wil)} comparisons")

    # ---- seed-0 DeLong from prediction parquets
    drows = []
    for ds_dir in (rdir / "predictions").glob("*"):
        for tier_dir in ds_dir.glob("*"):
            ebm_pq = tier_dir / "ebm" / "seed0.parquet"
            if not ebm_pq.exists():
                continue
            ebm = pd.read_parquet(ebm_pq)
            ebm = ebm[ebm.split == "test"].sort_values("row_id")
            for model_dir in tier_dir.glob("*"):
                m = model_dir.name
                if m == "ebm" or "+" in m:
                    continue
                pq = model_dir / "seed0.parquet"
                if not pq.exists():
                    continue
                other = pd.read_parquet(pq)
                other = other[other.split == "test"].sort_values("row_id")
                if not np.array_equal(ebm.row_id.values,
                                      other.row_id.values):
                    continue
                a1, a2, p = delong_test(ebm.y.values, ebm.p.values,
                                        other.p.values)
                drows.append({"dataset": ds_dir.name, "tier": tier_dir.name,
                              "model": m, "auc_ebm": a1, "auc_model": a2,
                              "delong_p": p})
    dl = pd.DataFrame(drows, columns=["dataset", "tier", "model",
        "auc_ebm", "auc_model", "delong_p"])
    if len(dl):
        dl["delong_p_holm"] = np.nan
        for (ds, tier), g in dl.groupby(["dataset", "tier"]):
            dl.loc[g.index, "delong_p_holm"] = holm(g["delong_p"].tolist())
    dl.to_csv(rdir / "delong.csv", index=False)
    print(f"[aggregate] delong.csv: {len(dl)} comparisons")


if __name__ == "__main__":
    main()
