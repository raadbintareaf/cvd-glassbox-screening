"""Survey-weighted sensitivity (stage=weighted): BRFSS design-weight
(_LLCPWT) versions of prevalence, AUROC, and Brier from saved predictions.
Point-estimate reweighting only; full design-based variance is out of scope
and stated in Limitations."""
import argparse
import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.metrics import roc_auc_score
from src.utils.common import append_rows, completed_keys, load_config

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
a = ap.parse_args()
cfg = load_config(a.config)
results_csv = str(Path(cfg["results_dir"]) / "raw_results.csv")
done = completed_keys(results_csv)
analytic = {}
for pq in sorted(Path(cfg["results_dir"], "predictions").glob("*/*/*/seed*.parquet")):
    ds, tier, model = pq.parts[-4], pq.parts[-3], pq.parts[-2]
    seed = pq.stem.replace("seed", "")
    if ("weighted", ds, tier, model, seed) in done or "+" in model:
        continue
    if ds not in analytic:
        analytic[ds] = pd.read_parquet(
            Path(cfg["data_dir"]) / f"brfss_{ds[-4:]}_analytic.parquet",
            columns=["SurveyWeight"])
    df = pd.read_parquet(pq)
    tst = df[df.split == "test"]
    w = analytic[ds].SurveyWeight.values[tst.row_id.values]
    y, p = tst.y.values, tst.p.values
    rows = [dict(run_id=f"wt-{ds}-{tier}-{model}-s{seed}", stage="weighted",
                 dataset=ds, tier=tier, model=model, seed=seed, split="test",
                 subgroup="ALL", metric=m, value=v) for m, v in [
        ("prevalence_weighted", float(np.average(y, weights=w))),
        ("auroc_weighted", float(roc_auc_score(y, p, sample_weight=w))),
        ("brier_weighted", float(np.average((p - y) ** 2, weights=w)))]]
    append_rows(results_csv, rows)
    print(f"[done] weighted {ds}/{tier}/{model}/s{seed}")
