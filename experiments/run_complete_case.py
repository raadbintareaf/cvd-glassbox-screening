"""Complete-case ablation (stage=complete_case): the headline models refit on
the T1 complete-case subset, quantifying the cost of the discard-44% design
used by the prior literature. Same protocol, tuned T1 params, 5 seeds."""
import argparse
from pathlib import Path
from src.data.variable_map import OUTCOME_LABEL, tier_labels
from src.evaluation.metrics import all_probability_metrics, confusion_at, threshold_for
from src.models.registry import make_model
from src.utils.common import (append_rows, completed_keys, load_config,
                              seed_everything, stratified_splits)
from experiments.run_benchmark import load_analytic, model_params

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--model", required=True)
ap.add_argument("--seed", type=int, required=True)
a = ap.parse_args()
cfg = load_config(a.config)
results_csv = str(Path(cfg["results_dir"]) / "raw_results.csv")
key = ("complete_case", "brfss2022", "T1", a.model, str(a.seed))
if key in completed_keys(results_csv):
    print(f"[skip] {key}"); raise SystemExit
seed_everything(a.seed)
df = load_analytic(cfg, "brfss2022")
df = df[df.complete_case].reset_index(drop=True)
feats = tier_labels("T1")
X, y = df[feats], df[OUTCOME_LABEL].values.astype(int)
idx = stratified_splits(y, a.seed, tuple(cfg["split_fracs"]))
m = make_model(a.model, model_params(cfg, a.model, "T1"), a.seed)
m.fit(X.iloc[idx["train"]], y[idx["train"]], X.iloc[idx["val"]], y[idx["val"]])
p_va = m.predict_proba_pos(X.iloc[idx["val"]])
p_te = m.predict_proba_pos(X.iloc[idx["test"]])
rows, base = [], dict(run_id=f"cc-{a.model}-s{a.seed}", stage="complete_case",
                      dataset="brfss2022", tier="T1", model=a.model,
                      seed=a.seed, split="test", subgroup="ALL",
                      extra=f"n={len(df)}")
for k, v in all_probability_metrics(y[idx["test"]], p_te).items():
    rows.append({**base, "metric": k, "value": v})
for r in cfg["operating_rules"]:
    t = threshold_for(y[idx["val"]], p_va, r["rule"], r["target"])
    for k, v in confusion_at(y[idx["test"]], p_te, t).items():
        rows.append({**base, "metric": f"{r['name']}_{k}", "value": v})
append_rows(results_csv, rows)
auroc = [r["value"] for r in rows if r["metric"] == "auroc"][0]
print(f"[done] complete_case {a.model} seed{a.seed}: n={len(df):,} auroc={auroc:.4f}")
