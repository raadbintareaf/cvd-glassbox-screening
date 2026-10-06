"""Isotonic-recalibration sensitivity (stage=recalibration): fit isotonic on
validation predictions, apply to test; zero retraining. Reports calibration
metrics before/after for every saved (tier, model, seed)."""
import argparse
from pathlib import Path
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from src.evaluation.metrics import all_probability_metrics
from src.utils.common import append_rows, completed_keys, load_config

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
a = ap.parse_args()
cfg = load_config(a.config)
results_csv = str(Path(cfg["results_dir"]) / "raw_results.csv")
done = completed_keys(results_csv)
for pq in sorted(Path(cfg["results_dir"], "predictions").glob("*/*/*/seed*.parquet")):
    ds, tier, model = pq.parts[-4], pq.parts[-3], pq.parts[-2]
    seed = pq.stem.replace("seed", "")
    if ("recalibration", ds, tier, model, seed) in done or "+" in model:
        continue
    df = pd.read_parquet(pq)
    cal, tst = df[df.split == "val"], df[df.split == "test"]
    iso = IsotonicRegression(out_of_bounds="clip").fit(cal.p, cal.y)
    p_iso = iso.predict(tst.p)
    rows = []
    base = dict(run_id=f"recal-{ds}-{tier}-{model}-s{seed}",
                stage="recalibration", dataset=ds, tier=tier, model=model,
                seed=seed, split="test", subgroup="ALL")
    for tag, p in (("raw", tst.p.values), ("iso", p_iso)):
        for k, v in all_probability_metrics(tst.y.values, p).items():
            rows.append({**base, "metric": f"{k}_{tag}", "value": v})
    append_rows(results_csv, rows)
    print(f"[done] recal {ds}/{tier}/{model}/s{seed}")
