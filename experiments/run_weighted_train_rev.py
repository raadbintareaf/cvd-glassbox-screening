"""Design-weight-informed TRAINING (LLCPWT x class balance) for EBM/XGB/CatB."""
import argparse
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
import src.models.registry as R
from src.data.variable_map import OUTCOME_LABEL, tier_labels
from src.utils.common import (append_rows, completed_keys, load_config,
                              seed_everything, stratified_splits)
from experiments.run_benchmark import load_analytic, model_params

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--seed", type=int, required=True)
a = ap.parse_args()
cfg = load_config("configs/default.yaml")
csv = str(Path(cfg["results_dir"])/"raw_results.csv")
key = ("weighted_train","brfss2022","T1",a.model+"_wt",str(a.seed))
if key in completed_keys(csv): print("[skip]",key); raise SystemExit
seed_everything(a.seed)
df = load_analytic(cfg,"brfss2022")
X, y = df[tier_labels("T1")], df[OUTCOME_LABEL].values.astype(int)
w_all = df["SurveyWeight"].values
idx = stratified_splits(y, a.seed, tuple(cfg["split_fracs"]))
wtr = w_all[idx["train"]]; wtr = wtr*len(wtr)/wtr.sum()
params = dict(model_params(cfg, a.model, "T1"))
m = R.make_model(a.model, params, a.seed)
if a.model == "ebm":
    _o = R.balanced_sample_weight
    R.balanced_sample_weight = lambda yy,_w=wtr,_o=_o: _o(yy)*_w
else:
    m.extra_weight_ = wtr
m.fit(X.iloc[idx["train"]], y[idx["train"]],
      X.iloc[idx["val"]], y[idx["val"]])
if a.model == "ebm": R.balanced_sample_weight = _o
p = m.predict_proba_pos(X.iloc[idx["test"]])
wte = w_all[idx["test"]]
rows=[dict(run_id=f"wt-train-{a.model}-s{a.seed}",stage="weighted_train",
    dataset="brfss2022",tier="T1",model=a.model+"_wt",seed=a.seed,
    split="test",subgroup="ALL",metric=k,value=v) for k,v in [
    ("auroc",roc_auc_score(y[idx["test"]],p)),
    ("auroc_weighted",roc_auc_score(y[idx["test"]],p,sample_weight=wte))]]
append_rows(csv, rows)
print(f"[done] wt-train {a.model} s{a.seed} "
      f"auroc={rows[0]['value']:.4f} w={rows[1]['value']:.4f}")
