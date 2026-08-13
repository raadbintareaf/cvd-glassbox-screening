"""EBM + CatBoost WITHOUT class weighting, T1: raw & isotonic calibration."""
import argparse
import numpy as np
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
import src.models.registry as R
from src.data.variable_map import OUTCOME_LABEL, tier_labels
from src.evaluation.metrics import all_probability_metrics
from src.utils.common import (append_rows, completed_keys, load_config,
                              seed_everything, stratified_splits)
from experiments.run_benchmark import load_analytic, model_params

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--model", required=True)
ap.add_argument("--seed", type=int, required=True)
a = ap.parse_args()
cfg = load_config(a.config)
csv = str(Path(cfg["results_dir"])/"raw_results.csv")
key = ("unweighted", "brfss2022", "T1", a.model+"_unw", str(a.seed))
if key in completed_keys(csv):
    print("[skip]", key); raise SystemExit
seed_everything(a.seed)
df = load_analytic(cfg, "brfss2022")
X, y = df[tier_labels("T1")], df[OUTCOME_LABEL].values.astype(int)
idx = stratified_splits(y, a.seed, tuple(cfg["split_fracs"]))
params = dict(model_params(cfg, a.model, "T1"))
if a.model == "catboost":
    params["auto_class_weights"] = None
else:
    R.balanced_sample_weight = lambda yy: np.ones(len(yy))
m = R.make_model(a.model, params, a.seed)
m.fit(X.iloc[idx["train"]], y[idx["train"]], X.iloc[idx["val"]], y[idx["val"]])
p_va = m.predict_proba_pos(X.iloc[idx["val"]])
p_te = m.predict_proba_pos(X.iloc[idx["test"]])
iso = IsotonicRegression(out_of_bounds="clip").fit(p_va, y[idx["val"]])
rows = []
base = dict(run_id=f"unw-{a.model}-s{a.seed}", stage="unweighted",
            dataset="brfss2022", tier="T1", model=a.model+"_unw",
            seed=a.seed, split="test", subgroup="ALL")
for tag, p in (("raw", p_te), ("iso", iso.predict(p_te))):
    for k, v in all_probability_metrics(y[idx["test"]], p).items():
        rows.append({**base, "metric": f"{k}_{tag}", "value": v})
append_rows(csv, rows)
au = [r["value"] for r in rows if r["metric"] == "auroc_raw"][0]
ec = [r["value"] for r in rows if r["metric"] == "ece_raw"][0]
print(f"[done] unweighted {a.model} s{a.seed}: auroc={au:.4f} ece_raw={ec:.4f}")
