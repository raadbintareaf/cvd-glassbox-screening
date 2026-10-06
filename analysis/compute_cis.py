"""Seed-0 stratified bootstrap 95% CIs for test AUROC/AUPRC per (tier, model)
-> results/bootstrap_cis.csv"""
from pathlib import Path
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from src.evaluation.metrics import bootstrap_ci

rows = []
for pq in sorted(Path("results/predictions").glob("*/*/*/seed0.parquet")):
    ds, tier, model = pq.parts[-4], pq.parts[-3], pq.parts[-2]
    if "+" in model:
        continue
    df = pd.read_parquet(pq); t = df[df.split == "test"]
    y, p = t.y.values, t.p.values
    lo, hi = bootstrap_ci(y, p, lambda a, b: roc_auc_score(a, b), n_boot=1000)
    rows.append(dict(dataset=ds, tier=tier, model=model, metric="auroc",
                     point=roc_auc_score(y, p), ci_lo=lo, ci_hi=hi))
    lo, hi = bootstrap_ci(y, p, lambda a, b: average_precision_score(a, b),
                          n_boot=1000)
    rows.append(dict(dataset=ds, tier=tier, model=model, metric="auprc",
                     point=average_precision_score(y, p), ci_lo=lo, ci_hi=hi))
    print(f"[ci] {ds}/{tier}/{model}")
pd.DataFrame(rows).to_csv("results/bootstrap_cis.csv", index=False)
print("wrote results/bootstrap_cis.csv")
