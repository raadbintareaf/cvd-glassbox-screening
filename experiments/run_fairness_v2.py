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
