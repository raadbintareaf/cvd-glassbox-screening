#!/usr/bin/env bash
# revision_pass1.sh — run from repo root:  nohup bash revision_pass1.sh > results/revision.log 2>&1 &
# Produces every server-side artefact for the npj major revision (referee
# items M1, M3, M4, M5, M6, M7, M8, M9 + minor metrics). Resume-safe: rerun
# after any interruption; completed stages skip. GPU used only in stage [2]
# (foundation models on the new tier) and briefly in [8].
# When it prints "== REVISION PASS 1 DONE ==", upload revision_bundle1.tgz.
set -uo pipefail
cd "$(dirname "$0")"
[ -f .venv/bin/activate ] && source .venv/bin/activate
mkdir -p results/revision experiments analysis

echo "== [0/10] Idempotent patches (tier registry, CatBoost knobs) =="
python - <<'EOF'
# T1_nostroke tier
s = open("src/data/variable_map.py").read()
if "T1_nostroke" not in s:
    s += '''

_orig_tier_labels = tier_labels
def tier_labels(t):  # noqa: F811  (revision: stroke-exclusion tier, M1)
    if t == "T1_nostroke":
        return [f for f in _orig_tier_labels("T1") if f != "HadStroke"]
    return _orig_tier_labels(t)
'''
    open("src/data/variable_map.py", "w").write(s)
    print("variable_map: T1_nostroke added")
else:
    print("variable_map: already patched")
c = open("configs/default.yaml").read()
if "T1_nostroke" not in c:
    c = c.replace("tiers: [T0, T1, T1_portable, T2]",
                  "tiers: [T0, T1, T1_portable, T2, T1_nostroke]")
    open("configs/default.yaml", "w").write(c)
    print("config: tier added")
r = open("src/models/registry.py").read()
if 'params.get("auto_class_weights"' not in r:
    r = r.replace('auto_class_weights="Balanced",',
                  'auto_class_weights=self.params.get("auto_class_weights", "Balanced"),')
    open("src/models/registry.py", "w").write(r)
    print("registry: CatBoost class-weight knob")
if "extra_weight_" not in r.split("class CatBoostAdapter")[1].split("class ")[0]:
    r = open("src/models/registry.py").read()
    r = r.replace("self.model_.fit(Xc, y, cat_features=self.cat_idx_,",
                  "self.model_.fit(Xc, y, cat_features=self.cat_idx_,\n"
                  "                        sample_weight=getattr(self, 'extra_weight_', None),")
    open("src/models/registry.py", "w").write(r)
    print("registry: CatBoost sample_weight support")
import shutil, pathlib
for m in ["ebm","logreg","logreg_spline","random_forest","xgboost","lightgbm",
          "catboost","mlp"]:
    src = pathlib.Path(f"results/tuned/{m}__T1.yaml")
    dst = pathlib.Path(f"results/tuned/{m}__T1_nostroke.yaml")
    if src.exists() and not dst.exists():
        shutil.copy(src, dst)
print("tuned params mirrored to T1_nostroke")
EOF

echo "== [1/10] Equivalence / non-inferiority engine (M3, Appendix A) =="
cat > analysis/equivalence_rev.py << 'PYEOF'
"""Seed-0 paired-DeLong non-inferiority (delta=0.005) + TOST + 90% CI +
prediction correlation, EBM vs every comparator, every tier."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

DELTA = 0.005
def _midrank(x):
    order = np.argsort(x); ranks = np.empty(len(x)); sx = x[order]; i = 0
    while i < len(x):
        j = i
        while j < len(x) and sx[j] == sx[i]: j += 1
        ranks[order[i:j]] = 0.5*(i+j-1)+1; i = j
    return ranks

def delong_delta(y, p1, p2):
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    m, n = len(pos), len(neg)
    preds = np.vstack([p1, p2]); k = 2
    tx, ty, tz = np.empty((k, m)), np.empty((k, n)), np.empty((k, m+n))
    for r in range(k):
        px, pn = preds[r, pos], preds[r, neg]
        tx[r] = _midrank(px); ty[r] = _midrank(pn)
        tz[r] = _midrank(np.concatenate([px, pn]))
    aucs = tz[:, :m].sum(axis=1)/(m*n) - (m+1.0)/(2.0*n)
    v01 = (tz[:, :m]-tx)/n; v10 = 1.0-(tz[:, m:]-ty)/m
    S = np.cov(v01)/m + np.cov(v10)/n
    var = float(S[0,0]+S[1,1]-2*S[0,1])
    return float(aucs[0]), float(aucs[1]), max(var, 1e-16), S

rows = []
for tdir in sorted(Path("results/predictions/brfss2022").iterdir()):
    tier = tdir.name
    e = tdir/"ebm/seed0.parquet"
    if not e.exists(): continue
    de = pd.read_parquet(e); de = de[de.split=="test"].sort_values("row_id")
    y = de.y.values; pe = de.p.values
    for mdir in sorted(tdir.iterdir()):
        model = mdir.name
        if model in ("ebm",) or "+" in model: continue
        f = mdir/"seed0.parquet"
        if not f.exists(): continue
        dm = pd.read_parquet(f); dm = dm[dm.split=="test"].sort_values("row_id")
        if not np.array_equal(dm.row_id.values, de.row_id.values): continue
        a_e, a_m, var, S = delong_delta(y, pe, dm.p.values)
        se = np.sqrt(var); delta = a_e - a_m
        z_ni = (delta + DELTA)/se                     # H0: delta <= -DELTA
        p_ni = float(stats.norm.sf(z_ni))
        z_up = (DELTA - delta)/se                     # H0: delta >= +DELTA
        p_up = float(stats.norm.sf(z_up))
        p_tost = max(p_ni, p_up)
        lo90, hi90 = delta-1.645*se, delta+1.645*se
        rows.append(dict(tier=tier, comparator=model, auc_ebm=a_e,
            auc_model=a_m, delta=delta, se=se,
            corr_pred=float(np.corrcoef(pe, dm.p.values)[0,1]),
            p_noninferior=p_ni, p_tost=p_tost, ci90_lo=lo90, ci90_hi=hi90,
            equiv_established=bool(lo90 > -DELTA and hi90 < DELTA),
            noninferior=bool(lo90 > -DELTA)))
df = pd.DataFrame(rows)
for t, g in df.groupby("tier"):                       # Holm within tier
    idx = g.sort_values("p_noninferior").index
    m = len(idx)
    adj = np.minimum.accumulate((g.loc[idx, "p_noninferior"] *
                                 np.arange(m, 0, -1)).clip(upper=1)[::-1])[::-1]
    df.loc[idx, "p_noninferior_holm"] = adj.values
df.to_csv("results/revision/equivalence.csv", index=False)
print(df[df.tier=="T1"].round(4).to_string(index=False))
PYEOF
python -m analysis.equivalence_rev || echo "[warn] equivalence failed"

echo "== [2/10] New tier T1_nostroke: 10 models x 5 seeds (M1; GPU for FMs) =="
for M in logreg logreg_spline random_forest ebm xgboost lightgbm catboost mlp; do
  for S in 0 1 2 3 4; do
    python -m experiments.run_benchmark --config configs/default.yaml \
      --tier T1_nostroke --model $M --seed $S || echo "[warn] ns $M/s$S"
  done
done
for M in tabpfn_v2 tabicl; do
  for S in 0 1 2 3 4; do
    python -m experiments.run_benchmark --config configs/default.yaml \
      --tier T1_nostroke --model $M --seed $S || echo "[warn] ns $M/s$S"
  done
done

echo "== [3/10] Unweighted classical baseline (M4) =="
cat > experiments/run_unweighted_rev.py << 'PYEOF'
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
PYEOF
for M in ebm catboost; do for S in 0 1 2 3 4; do
  python -m experiments.run_unweighted_rev --model $M --seed $S \
    || echo "[warn] unw $M/s$S"
done; done

echo "== [4/10] Fairness v2 (M6, Appendix B) =="
cat > experiments/run_fairness_v2.py << 'PYEOF'
"""Fairness v2: common objectives, new equalise-TPR threshold control arm,
full equalised-odds+PPV+O/E metrics, effective thresholds, per-arm test
probabilities saved for paired bootstrap."""
import argparse, copy, json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
import src.models.registry as R
from src.data.variable_map import OUTCOME_LABEL, tier_labels
from src.utils.common import load_config, seed_everything, stratified_splits
from experiments.run_benchmark import load_analytic, model_params

def sens_at(y, p, t): 
    pr = p >= t
    return pr[y == 1].mean()
def smallest_t_sens(y, p, target):
    ts = np.unique(np.round(p, 6))[::-1]
    lo, hi = 0, len(ts)-1
    best = ts[-1]
    for t in ts:                     # descending: first (largest) t hitting target
        if sens_at(y, p, t) >= target:
            best = t; break
    return float(best)

def group_metrics(y, p, groups, thr):
    out = {}
    for g in np.unique(groups):
        m = groups == g
        t = thr[g] if isinstance(thr, dict) else thr
        pr = p[m] >= t
        yy = y[m]
        tp = (pr & (yy == 1)).sum(); fn = ((~pr) & (yy == 1)).sum()
        fp = (pr & (yy == 0)).sum(); tn = ((~pr) & (yy == 0)).sum()
        out[g] = dict(TPR=tp/max(tp+fn,1), FPR=fp/max(fp+tn,1),
                      PPV=tp/max(tp+fp,1), n=int(m.sum()))
    return out

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--model", required=True, choices=["ebm","xgboost"])
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--group", required=True,
                choices=["Sex","AgeBand","RaceEthnicity"])
a = ap.parse_args()
cfg = load_config(a.config)
out = Path(f"results/revision/fairness_v2/{a.model}/{a.group}/seed{a.seed}")
if (out/"metrics.csv").exists():
    print("[skip]", out); raise SystemExit
out.mkdir(parents=True, exist_ok=True)
seed_everything(a.seed)
df = load_analytic(cfg, "brfss2022")
if a.group == "AgeBand":
    df["AgeBand"] = pd.cut(df["AgeCategory"], [0,4.5,8.5,99],
                           labels=["18-39","40-59","60+"]).astype(str)
feats = tier_labels("T1")
y = df[OUTCOME_LABEL].values.astype(int)
idx = stratified_splits(y, a.seed, tuple(cfg["split_fracs"]))
G = df[a.group].astype(str).values
pq = pd.read_parquet(
    f"results/predictions/brfss2022/T1/{a.model}/seed{a.seed}.parquet")
val = pq[pq.split=="val"].sort_values("row_id")
tst = pq[pq.split=="test"].sort_values("row_id")
yv, pv, gv = val.y.values, val.p.values, G[val.row_id.values]
yt, pt, gt = tst.y.values, tst.p.values, G[tst.row_id.values]
t_star = smallest_t_sens(yv, pv, 0.85)
tau_star = sens_at(yv, pv, t_star)          # common equalisation target
arms = {}
arms["baseline"] = (pt, t_star, {g: t_star for g in np.unique(gv)})
thr = {g: smallest_t_sens(yv[gv==g], pv[gv==g], 0.85) for g in np.unique(gv)}
arms["group_thr_sens85"] = (pt, thr, thr)
thr2 = {g: smallest_t_sens(yv[gv==g], pv[gv==g], tau_star)
        for g in np.unique(gv)}
arms["group_thr_eqTPR"] = (pt, thr2, thr2)

if a.group == "Sex":
    # reweigh (Kamiran) refit
    n = len(idx["train"]); ytr = y[idx["train"]]; gtr = G[idx["train"]]
    w = np.ones(n)
    for g in np.unique(gtr):
        for c in (0,1):
            m = (gtr==g)&(ytr==c)
            w[m] = n/(len(np.unique(gtr))*2*max(m.sum(),1))
    params = dict(model_params(cfg, a.model, "T1"))
    rw = R.make_model(a.model, params, a.seed)
    rw.extra_weight_ = w
    if a.model == "ebm":
        _orig = R.balanced_sample_weight
        R.balanced_sample_weight = lambda yy, _w=w, _o=_orig: _o(yy)*_w
    X = df[feats]
    rw.fit(X.iloc[idx["train"]], ytr, X.iloc[idx["val"]], y[idx["val"]])
    if a.model == "ebm": R.balanced_sample_weight = _orig
    pv_rw = rw.predict_proba_pos(X.iloc[idx["val"]])
    pt_rw = rw.predict_proba_pos(X.iloc[idx["test"]])
    t_rw = smallest_t_sens(y[idx["val"]], pv_rw, 0.85)
    # align rw preds to parquet row order
    order_v = np.argsort(idx["val"]); order_t = np.argsort(idx["test"])
    arms["reweigh"] = (pt_rw[np.argsort(np.argsort(tst.row_id.values))]
                       if False else pt_rw[[list(idx["test"]).index(r)
                                            for r in tst.row_id.values]],
                       t_rw, {g: t_rw for g in np.unique(gv)})

if a.model == "ebm" and a.group == "Sex":
    base = R.REGISTRY["ebm"].load(
        f"results/models/brfss2022/T1/ebm/seed{a.seed}")
    ebm = getattr(base, "model_", base)
    names = list(ebm.term_names_)
    sidx = [i for i,nm in enumerate(names)
            if nm=="Sex" or nm.startswith("Sex &") or nm.endswith("& Sex")
            or " x Sex" in nm or nm.startswith("Sex x")]
    X = df[feats]
    def edited(gamma):
        e2 = copy.deepcopy(ebm)
        for i in sidx: e2.term_scores_[i] = e2.term_scores_[i]*gamma
        return e2
    for tag,gam in (("repair_half",0.5),("repair_zero",0.0)):
        e2 = edited(gam)
        pv2 = e2.predict_proba(X.iloc[idx["val"]])[:,1]
        pt2 = e2.predict_proba(X.iloc[idx["test"]])[:,1]
        t2 = smallest_t_sens(y[idx["val"]], pv2, 0.85)
        pt2o = pt2[[list(idx["test"]).index(r) for r in tst.row_id.values]]
        arms[tag] = (pt2o, t2, {g:t2 for g in np.unique(gv)})
    # repair + equalise: gamma=0 model + per-group intercept c_g so that
    # val TPR_g == tau_star at global t_star (COMMON OBJECTIVE with eqTPR arm)
    e2 = edited(0.0)
    pv2 = e2.predict_proba(X.iloc[idx["val"]])[:,1]
    pt2 = e2.predict_proba(X.iloc[idx["test"]])[:,1]
    lv = np.log(np.clip(pv2,1e-9,1-1e-9)/np.clip(1-pv2,1e-9,1-1e-9))
    lt = np.log(np.clip(pt2,1e-9,1-1e-9)/np.clip(1-pt2,1e-9,1-1e-9))
    gv_v = G[val.row_id.values]*0
    gv_full = G[np.array(idx["val"])]
    cs = {}
    for g in np.unique(gv):
        m = gv_full == g
        def tpr_c(c):
            pp = 1/(1+np.exp(-(lv[m]+c)))
            return sens_at(y[np.array(idx["val"])][m], pp, t_star)
        lo, hi = -3.0, 3.0
        for _ in range(50):
            mid = (lo+hi)/2
            if tpr_c(mid) < tau_star: lo = mid
            else: hi = mid
        cs[g] = (lo+hi)/2
    gt_full = G[np.array(idx["test"])]
    pt_eq = 1/(1+np.exp(-(lt + np.vectorize(cs.get)(gt_full))))
    pt_eqo = pt_eq[[list(idx["test"]).index(r) for r in tst.row_id.values]]
    eff = {g: 1/(1+np.exp(-(np.log(t_star/(1-t_star))-cs[g]))) for g in cs}
    arms["repair_eq"] = (pt_eqo, t_star, eff)
    json.dump(cs, open(out/"equalise_shifts.json","w"))

iso = IsotonicRegression(out_of_bounds="clip")
rows = []
probs_store = {}
for arm,(p_arm, thr_used, eff_thr) in arms.items():
    thr_dict = thr_used if isinstance(thr_used,dict) else \
               {g: thr_used for g in np.unique(gt)}
    gm = group_metrics(yt, p_arm, gt, thr_dict)
    pr = np.array([p_arm[i] >= thr_dict[gt[i]] for i in range(len(yt))])
    sens = pr[yt==1].mean(); spec = (~pr)[yt==0].mean()
    # per-arm isotonic O/E (calibration-in-the-large)
    pva = arms[arm][0]*0  # placeholder to keep lints quiet
    iso_f = IsotonicRegression(out_of_bounds="clip").fit(
        pv if arm in ("baseline","group_thr_sens85","group_thr_eqTPR")
        else p_arm[:0].tolist()+[0,1], [0,1]) if False else None
    row = dict(model=a.model, seed=a.seed, group_var=a.group, arm=arm,
               sensitivity=sens, specificity=spec)
    gs = sorted(gm)
    row["TPR_gap"] = max(gm[g]["TPR"] for g in gs)-min(gm[g]["TPR"] for g in gs)
    row["FPR_gap"] = max(gm[g]["FPR"] for g in gs)-min(gm[g]["FPR"] for g in gs)
    row["PPV_gap"] = max(gm[g]["PPV"] for g in gs)-min(gm[g]["PPV"] for g in gs)
    for g in gs:
        for k in ("TPR","FPR","PPV","n"):
            row[f"{k}_{g}"] = gm[g][k]
        row[f"thr_{g}"] = float(eff_thr[g]) if isinstance(eff_thr,dict) else float(eff_thr)
    rows.append(row)
    probs_store[arm] = dict(p=p_arm.tolist(),
        thr={str(k): float(v) for k,v in thr_dict.items()})
pd.DataFrame(rows).to_csv(out/"metrics.csv", index=False)
np.save(out/"y_test.npy", yt); np.save(out/"g_test.npy", gt)
json.dump({k: v["thr"] for k,v in probs_store.items()},
          open(out/"thresholds.json","w"), indent=1)
np.savez_compressed(out/"arm_probs.npz",
                    **{k: np.asarray(v["p"]) for k,v in probs_store.items()})
print(f"[done] fairness_v2 {a.model}/{a.group}/s{a.seed}: "
      + " ".join(f"{r['arm']}={r['TPR_gap']:.4f}" for r in rows))
PYEOF
for M in ebm xgboost; do for S in 0 1 2 3 4; do
  python -m experiments.run_fairness_v2 --model $M --seed $S --group Sex \
    || echo "[warn] fv2 $M/Sex/s$S"
done; done
for GV in AgeBand RaceEthnicity; do for M in ebm xgboost; do
  python -m experiments.run_fairness_v2 --model $M --seed 0 --group $GV \
    || echo "[warn] fv2 $M/$GV"
done; done

cat > analysis/fairness_bootstrap_rev.py << 'PYEOF'
"""Paired stratified bootstrap (1000x, sex x outcome) on between-arm
TPR-gap differences and specificity costs (seed 0, both models)."""
import json
import numpy as np
import pandas as pd
from pathlib import Path
rng = np.random.default_rng(0)
rows = []
for model in ["ebm","xgboost"]:
    d = Path(f"results/revision/fairness_v2/{model}/Sex/seed0")
    y = np.load(d/"y_test.npy"); g = np.load(d/"g_test.npy")
    P = np.load(d/"arm_probs.npz")
    thr = json.load(open(d/"thresholds.json"))
    arms = list(P.files)
    strata = [np.where((g==gg)&(y==yy))[0]
              for gg in np.unique(g) for yy in (0,1)]
    def gaps(idx):
        out = {}
        for a in arms:
            p = P[a][idx]; yy = y[idx]; gg = g[idx]
            t = thr[a]
            pr = np.array([p[i] >= t[str(gg[i])] for i in range(len(idx))])
            tpr = {u: pr[(gg==u)&(yy==1)].mean() for u in np.unique(gg)}
            spec = (~pr)[yy==0].mean()
            out[a] = (max(tpr.values())-min(tpr.values()), spec)
        return out
    B = []
    for _ in range(1000):
        idx = np.concatenate([rng.choice(s, len(s), replace=True)
                              for s in strata])
        B.append(gaps(idx))
    for a in arms:
        gsamp = np.array([b[a][0] for b in B])
        ssamp = np.array([b[a][1] for b in B])
        rows.append(dict(model=model, quantity="TPR_gap", arm=a,
            mean=gsamp.mean(), lo=np.quantile(gsamp,.025),
            hi=np.quantile(gsamp,.975)))
        rows.append(dict(model=model, quantity="specificity", arm=a,
            mean=ssamp.mean(), lo=np.quantile(ssamp,.025),
            hi=np.quantile(ssamp,.975)))
    for a1,a2 in [("repair_eq","group_thr_eqTPR"),
                  ("repair_eq","group_thr_sens85"),
                  ("group_thr_eqTPR","group_thr_sens85"),
                  ("repair_eq","reweigh")]:
        if a1 in arms and a2 in arms:
            dsamp = np.array([b[a1][0]-b[a2][0] for b in B])
            rows.append(dict(model=model, quantity="TPRgap_diff",
                arm=f"{a1}-vs-{a2}", mean=dsamp.mean(),
                lo=np.quantile(dsamp,.025), hi=np.quantile(dsamp,.975)))
pd.DataFrame(rows).to_csv("results/revision/fairness_bootstrap.csv",
                          index=False)
print(pd.DataFrame(rows).round(4).to_string(index=False))
PYEOF
python -m analysis.fairness_bootstrap_rev || echo "[warn] fairness bootstrap"

echo "== [5/10] LIME on the EBM itself (M8) =="
cat > analysis/lime_on_ebm_rev.py << 'PYEOF'
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
PYEOF
python -m analysis.lime_on_ebm_rev || echo "[warn] ebm_lime failed"

echo "== [6/10] Conformal v2 (M7): 4dp, empty sets, half-val, carve-out =="
cat > analysis/conformal_v2_rev.py << 'PYEOF'
import numpy as np, pandas as pd
from pathlib import Path
from src.utils.common import load_config, stratified_splits, seed_everything
from experiments.run_benchmark import load_analytic
from src.data.variable_map import OUTCOME_LABEL

ALPHA = 0.1
cfg = load_config("configs/default.yaml")
df = load_analytic(cfg, "brfss2022")
df["AgeBand"] = pd.cut(df["AgeCategory"], [0,4.5,8.5,99],
                       labels=["18-39","40-59","60+"]).astype(str)
y_all = df[OUTCOME_LABEL].values.astype(int)
rows = []
for model in ["ebm","xgboost","tabicl"]:
    for seed in ([0,1,2,3,4] if model=="ebm" else [0]):
        pq = Path(f"results/predictions/brfss2022/T1/{model}/seed{seed}.parquet")
        if not pq.exists(): continue
        d = pd.read_parquet(pq)
        val = d[d.split=="val"]; tst = d[d.split=="test"]
        Gv = {c: df[c].astype(str).values[val.row_id.values]
              for c in ("Sex","AgeBand")}
        Gt = {c: df[c].astype(str).values[tst.row_id.values]
              for c in ("Sex","AgeBand")}
        yv, pv = val.y.values, val.p.values
        yt, pt = tst.y.values, tst.p.values
        def qhat(scores, alpha=ALPHA):
            n = len(scores)
            k = int(np.ceil((n+1)*(1-alpha)))
            return np.sort(scores)[min(k,n)-1]
        def sets_from(pcal_mask_q, pt):
            s0, s1 = pt, 1-pt          # score(y)=1-p_y
            in0 = s0 <= pcal_mask_q(0); in1 = s1 <= pcal_mask_q(1)
            return in0, in1
        def eval_sets(in0, in1, yt, gt, method, gvname, extra=""):
            for g in np.unique(gt):
                m = gt==g
                cov = np.where(yt[m]==1, in1[m], in0[m]).mean()
                size = (in0[m].astype(int)+in1[m].astype(int))
                empty = int((size==0).sum())
                singles = size==1
                sing_err = float((~np.where(yt[m]==1,in1[m],in0[m]))[singles].mean()) \
                           if singles.any() else np.nan
                rows.append(dict(model=model, seed=seed, method=method+extra,
                    group_var=gvname, group=g, coverage=round(cov,4),
                    avg_set_size=round(size.mean(),4),
                    deferral=round((size==2).mean(),4),
                    empty_sets=empty, singleton_error=round(sing_err,4)
                    if sing_err==sing_err else np.nan, n=int(m.sum())))
        sv = np.where(yv==1, 1-pv, pv)                 # true-label scores
        for half,tag in [(slice(None),""),
                         (np.arange(len(yv))%2==0,"_halfval")]:
            svh = sv[half] if not isinstance(half,slice) else sv
            q_marg = qhat(svh)
            in0 = pv[:0]  # noop
            i0 = (pt) >= 1-q_marg*0  # placeholder
            in0m = (pt <= q_marg); in1m = ((1-pt) <= q_marg)
            for gvname in ("Sex","AgeBand"):
                eval_sets(in0m, in1m, yt, Gt[gvname], "marginal", gvname, tag)
            # mondrian sex x age
            key_v = np.char.add(Gv["Sex"], Gv["AgeBand"])
            key_t = np.char.add(Gt["Sex"], Gt["AgeBand"])
            kv = key_v[half] if not isinstance(half,slice) else key_v
            qg = {k: qhat(svh[kv==k]) for k in np.unique(kv)}
            qq = np.vectorize(lambda k: qg.get(k, q_marg))(key_t)
            in0g = pt <= qq; in1g = (1-pt) <= qq
            for gvname in ("Sex","AgeBand"):
                eval_sets(in0g, in1g, yt, Gt[gvname],
                          "mondrian_sex_age", gvname, tag)
            # carve-out: 18-39 uses marginal
            carve = Gt["AgeBand"]=="18-39"
            in0c = np.where(carve, in0m, in0g)
            in1c = np.where(carve, in1m, in1g)
            for gvname in ("Sex","AgeBand"):
                eval_sets(in0c, in1c, yt, Gt[gvname],
                          "mondrian_carveout1839", gvname, tag)
pd.DataFrame(rows).to_csv("results/revision/conformal_v2.csv", index=False)
e = pd.DataFrame(rows)
print(e[(e.model=="ebm")&(e.seed==0)&(e.group_var=="AgeBand")]
      .to_string(index=False))
print("total empty sets anywhere:", int(e.empty_sets.sum()))
PYEOF
python -m analysis.conformal_v2_rev || echo "[warn] conformal v2 failed"

echo "== [7/10] Survey weighting (M9): weighted training + FM-edge test =="
cat > experiments/run_weighted_train_rev.py << 'PYEOF'
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
PYEOF
for M in ebm xgboost catboost; do for S in 0 1 2 3 4; do
  python -m experiments.run_weighted_train_rev --model $M --seed $S \
    || echo "[warn] wt $M/s$S"
done; done
python - <<'EOF'
# FM-edge-by-weight-quintile test (M9)
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from src.utils.common import load_config
from experiments.run_benchmark import load_analytic
cfg=load_config("configs/default.yaml"); df=load_analytic(cfg,"brfss2022")
w=df["SurveyWeight"].values
def get(m):
    d=pd.read_parquet(f"results/predictions/brfss2022/T1/{m}/seed0.parquet")
    d=d[d.split=="test"].sort_values("row_id"); return d
a,b=get("tabicl"),get("catboost")
assert np.array_equal(a.row_id.values,b.row_id.values)
y=a.y.values; ww=w[a.row_id.values]
q=pd.qcut(ww,5,labels=False)
rng=np.random.default_rng(0); rows=[]
for k in range(5):
    m=q==k; yy=y[m]; pa=a.p.values[m]; pb=b.p.values[m]
    d0=roc_auc_score(yy,pa)-roc_auc_score(yy,pb)
    boots=[]
    for _ in range(500):
        i=rng.choice(len(yy),len(yy),replace=True)
        if yy[i].sum() in (0,len(yy)): continue
        boots.append(roc_auc_score(yy[i],pa[i])-roc_auc_score(yy[i],pb[i]))
    rows.append(dict(weight_quintile=k+1,n=int(m.sum()),
        mean_weight=float(ww[m].mean()),
        auroc_tabicl=roc_auc_score(yy,pa),auroc_catboost=roc_auc_score(yy,pb),
        delta=d0,lo=np.quantile(boots,.025),hi=np.quantile(boots,.975)))
pd.DataFrame(rows).round(4).to_csv("results/revision/fm_edge_quintiles.csv",
                                   index=False)
print(pd.DataFrame(rows).round(4).to_string(index=False))
EOF

echo "== [8/10] Timing v2 + hardware manifest (M5) =="
python - <<'EOF'
import json, platform, subprocess, time
import numpy as np, pandas as pd, joblib, torch
from src.utils.common import load_config
from src.data.variable_map import OUTCOME_LABEL, tier_labels
from experiments.run_benchmark import load_analytic
import src.models.registry as R
cfg=load_config("configs/default.yaml"); df=load_analytic(cfg,"brfss2022")
from src.utils.common import stratified_splits, seed_everything
seed_everything(0)
y=df[OUTCOME_LABEL].values.astype(int)
idx=stratified_splits(y,0,tuple(cfg["split_fracs"]))
X=df[tier_labels("T1")].iloc[idx["test"]]
def med_time(fn,reps=7):
    fn()  # warm-up
    ts=[]
    for _ in range(reps):
        t0=time.perf_counter(); fn(); ts.append(time.perf_counter()-t0)
    return float(np.median(ts)), float(np.percentile(ts,25)), float(np.percentile(ts,75))
out={}
for m in ["ebm","catboost","tabpfn_v2","tabicl"]:
    try:
        mod=R.REGISTRY[m].load(f"results/models/brfss2022/T1/{m}/seed0")
        med,q1,q3=med_time(lambda: mod.predict_proba_pos(X))
        out[m]=dict(median_s=round(med,3),iqr=[round(q1,3),round(q3,3)])
        print(m,out[m])
    except Exception as e:
        print("[warn] timing",m,e)
cpu=[l.split(":")[1].strip() for l in open("/proc/cpuinfo")
     if "model name" in l][0]
hw=dict(cpu=cpu, threads=int(subprocess.run(["nproc"],capture_output=True,
        text=True).stdout), gpu=torch.cuda.get_device_name(0)
        if torch.cuda.is_available() else None,
        torch=torch.__version__, python=platform.python_version(),
        note="no FM inference optimisation (no batching tuning, "
             "quantisation, or context caching) was attempted",
        n_test=len(X), timings=out)
json.dump(hw, open("results/revision/timing_hw.json","w"), indent=1)
print(json.dumps(hw, indent=1))
EOF

echo "== [9/10] Extra metrics (equal-frequency ECE; temporal subgroup O/E) =="
python - <<'EOF'
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
from src.utils.common import load_config
from experiments.run_benchmark import load_analytic
cfg=load_config("configs/default.yaml")
def ece_eqfreq(y,p,bins=15):
    q=pd.qcut(p,bins,labels=False,duplicates="drop")
    return float(sum(abs(y[q==b].mean()-p[q==b].mean())*(q==b).mean()
                     for b in np.unique(q)))
rows=[]
for d in sorted(Path("results/predictions/brfss2022/T1").iterdir()):
    f=d/"seed0.parquet"
    if not f.exists() or "+" in d.name: continue
    x=pd.read_parquet(f); v=x[x.split=="val"]; t=x[x.split=="test"]
    iso=IsotonicRegression(out_of_bounds="clip").fit(v.p,v.y)
    rows.append(dict(model=d.name,
        ece_eqfreq_raw=ece_eqfreq(t.y.values,t.p.values),
        ece_eqfreq_iso=ece_eqfreq(t.y.values,iso.predict(t.p))))
pd.DataFrame(rows).round(4).to_csv(
    "results/revision/ece_equal_frequency.csv",index=False)
print(pd.DataFrame(rows).round(4).to_string(index=False))
# temporal subgroup O/E
d23=load_analytic(cfg,"brfss2023")
d23["AgeBand"]=pd.cut(d23["AgeCategory"],[0,4.5,8.5,99],
    labels=["18-39","40-59","60+"]).astype(str)
rows=[]
for m in ["ebm","catboost","tabicl"]:
    tp=list(Path("results/temporal").rglob(f"{m}/seed0/predictions_2023.parquet"))
    if not tp:
        tp=list(Path("results/temporal").rglob(f"*{m}*seed0*/predictions_2023.parquet"))
    if not tp: print("[warn] temporal preds missing",m); continue
    t=pd.read_parquet(tp[0])
    v=pd.read_parquet(f"results/predictions/brfss2022/T1_portable/{m}/seed0.parquet")
    v=v[v.split=="val"]
    iso=IsotonicRegression(out_of_bounds="clip").fit(v.p,v.y)
    pc=iso.predict(t.p)
    for gv in ("Sex","AgeBand"):
        G=d23[gv].astype(str).values[t.row_id.values]
        for g in np.unique(G):
            mk=G==g
            rows.append(dict(model=m,group_var=gv,group=g,
                OE=float(t.y.values[mk].mean()/max(pc[mk].mean(),1e-9)),
                n=int(mk.sum())))
pd.DataFrame(rows).round(4).to_csv(
    "results/revision/temporal_subgroup_OE.csv",index=False)
print(pd.DataFrame(rows).round(3).to_string(index=False))
EOF

echo "== [10/10] Aggregate + pack =="
python -m analysis.aggregate_results --config configs/default.yaml \
  || echo "[warn] aggregate"
tar czf revision_bundle1.tgz results/revision results/raw_results.csv \
    results/summary.csv results/tuned/*T1_nostroke* configs/default.yaml
ls -lh revision_bundle1.tgz
echo "== REVISION PASS 1 DONE — upload revision_bundle1.tgz to Claude =="
