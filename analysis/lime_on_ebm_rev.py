"""LIME applied to the EBM (integer-encoded categoricals), sizes <=500,
agreement vs the EBM's exact importances."""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import kendalltau
import joblib
from lime.lime_tabular import LimeTabularExplainer
from src.data.variable_map import OUTCOME_LABEL, tier_labels
from src.utils.common import load_config, seed_everything, stratified_splits
from experiments.run_benchmark import load_analytic

OUT = Path("results/revision"); OUT.mkdir(exist_ok=True)
if (OUT/"faithfulness_ebm_lime.csv").exists():
    print("[skip] ebm_lime"); raise SystemExit
cfg = load_config("configs/default.yaml")
seed_everything(0)
df = load_analytic(cfg, "brfss2022")
feats = tier_labels("T1")
y = df[OUTCOME_LABEL].values.astype(int)
idx = stratified_splits(y, 0, tuple(cfg["split_fracs"]))
obj = joblib.load("results/models/brfss2022/T1/ebm/seed0/model.joblib")
ebm = getattr(obj, "model_", obj)
X = df[feats]
cats = [i for i,c in enumerate(feats)
        if str(X[c].dtype) == "category" or X[c].dtype == object]
enc = {}
Xn = np.zeros((len(X), len(feats)))
names_map = {}
for j,c in enumerate(feats):
    if j in cats:
        cc = X[c].astype("category")
        enc[j] = list(cc.cat.categories)
        Xn[:,j] = cc.cat.codes.values.astype(float)
        names_map[j] = [str(v) for v in enc[j]]
    else:
        Xn[:,j] = pd.to_numeric(X[c], errors="coerce").values
med = np.nanmedian(Xn, axis=0)
inds = np.where(np.isnan(Xn)); Xn[inds] = np.take(med, inds[1])
def predict_fn(A):
    d = {}
    for j,c in enumerate(feats):
        if j in enc:
            codes = np.clip(np.round(A[:,j]).astype(int), 0, len(enc[j])-1)
            d[c] = pd.Categorical.from_codes(codes, categories=enc[j])
        else:
            d[c] = A[:,j]
    return ebm.predict_proba(pd.DataFrame(d))
# exact truth
imp = dict(zip(ebm.term_names_, ebm.term_importances()))
truth = {}
for nm,v in imp.items():
    parts = [p.strip() for p in nm.replace(" x "," & ").split(" & ")]
    for p in parts: truth[p] = truth.get(p,0)+v/len(parts)
tv = np.array([truth[f] for f in feats])
Xtr = Xn[idx["train"]]
expl = LimeTabularExplainer(Xtr, feature_names=feats,
    categorical_features=cats, categorical_names=names_map,
    discretize_continuous=True, random_state=0, mode="classification")
Xte = Xn[idx["test"]]
rng = np.random.default_rng(0)
rows = []
for n_explain in (25,100,500):
    for rep in range(10 if n_explain<500 else 5):
        pick = rng.choice(len(Xte), n_explain, replace=False)
        agg = np.zeros(len(feats))
        for i in pick:
            e = expl.explain_instance(Xte[i], predict_fn,
                                      num_features=len(feats),
                                      num_samples=1000)
            for fid,w in e.as_map()[1]:
                agg[fid] += abs(w)
        tau = kendalltau(agg, tv).statistic
        top = set(np.argsort(agg)[-10:]); topt = set(np.argsort(tv)[-10:])
        rows.append(dict(comparator="ebm_lime", n_explain=n_explain,
            rep=rep, kendall_tau=tau,
            jaccard10=len(top&topt)/len(top|topt)))
        print(f"[ebm_lime] n={n_explain} rep={rep} tau={tau:.3f}")
pd.DataFrame(rows).to_csv(OUT/"faithfulness_ebm_lime.csv", index=False)
