#!/usr/bin/env bash
# revision_hotfix1.sh — run from repo root AFTER killing the current job:
#   kill %1   (or: pkill -f revision_pass1.sh)
#   bash revision_hotfix1.sh
#   nohup bash revision_pass1.sh > results/revision.log 2>&1 &
# Fixes: (1) tier_labels wrapper now passes **kwargs through and implements
# T1_nostroke via the repo's own drop_stroke switch; (2) run_fairness_v2
# rewritten with O(n) row alignment and a robust model loader; (3) stage-8
# timing gets the same robust loader. All completed work is preserved —
# the rerun skips finished cells.
set -euo pipefail
cd "$(dirname "$0")"

echo "== hotfix 1/3: variable_map wrapper =="
python - <<'EOF'
s = open("src/data/variable_map.py").read()
marker = "# revision: stroke-exclusion tier"
if marker in s:
    s = s[:s.index("_orig_tier_labels = tier_labels")]
s = s.rstrip() + '''

_orig_tier_labels = tier_labels
def tier_labels(t, **kw):  # noqa: F811  # revision: stroke-exclusion tier (M1)
    if t == "T1_nostroke":
        kw = dict(kw); kw["drop_stroke"] = True
        return _orig_tier_labels("T1", **kw)
    return _orig_tier_labels(t, **kw)
'''
open("src/data/variable_map.py", "w").write(s)
import importlib, sys
sys.path.insert(0, ".")
from src.data.variable_map import tier_labels
a = tier_labels("T1"); b = tier_labels("T1_nostroke")
assert "HadStroke" in a and "HadStroke" not in b and len(b) == len(a) - 1
print(f"wrapper OK: T1={len(a)} feats, T1_nostroke={len(b)} feats")
EOF

echo "== hotfix 2/3: run_fairness_v2 rewritten (O(n) alignment) =="
cat > experiments/run_fairness_v2.py << 'PYEOF'
"""Fairness v2 (corrected): common objectives, equalise-TPR control arm,
full EO+PPV metrics, effective thresholds, per-arm probs saved."""
import argparse, copy, json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import src.models.registry as R
from src.data.variable_map import OUTCOME_LABEL, tier_labels
from src.utils.common import load_config, seed_everything, stratified_splits
from experiments.run_benchmark import load_analytic, model_params

def load_any(name, path):
    for attr in ("REGISTRY", "MODELS", "ADAPTERS"):
        reg = getattr(R, attr, None)
        if reg and name in reg and hasattr(reg[name], "load"):
            try: return reg[name].load(path)
            except Exception: pass
    return joblib.load(Path(path) / "model.joblib")

def sens_at(y, p, t):
    pr = p >= t
    return pr[y == 1].mean()

def smallest_t_sens(y, p, target):
    ts = np.unique(np.round(p, 6))[::-1]
    for t in ts:
        if sens_at(y, p, t) >= target:
            return float(t)
    return float(ts[-1])

def group_metrics(y, p, groups, thr):
    out = {}
    for g in np.unique(groups):
        m = groups == g
        t = thr[g]
        pr = p[m] >= t
        yy = y[m]
        tp = (pr & (yy == 1)).sum(); fn = ((~pr) & (yy == 1)).sum()
        fp = (pr & (yy == 0)).sum(); tn = ((~pr) & (yy == 0)).sum()
        out[g] = dict(TPR=tp / max(tp + fn, 1), FPR=fp / max(fp + tn, 1),
                      PPV=tp / max(tp + fp, 1), n=int(m.sum()))
    return out

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--model", required=True, choices=["ebm", "xgboost"])
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--group", required=True,
                choices=["Sex", "AgeBand", "RaceEthnicity"])
a = ap.parse_args()
cfg = load_config(a.config)
out = Path(f"results/revision/fairness_v2/{a.model}/{a.group}/seed{a.seed}")
if (out / "metrics.csv").exists():
    print("[skip]", out); raise SystemExit
out.mkdir(parents=True, exist_ok=True)
seed_everything(a.seed)
df = load_analytic(cfg, "brfss2022")
if a.group == "AgeBand":
    df["AgeBand"] = pd.cut(df["AgeCategory"], [0, 4.5, 8.5, 99],
                           labels=["18-39", "40-59", "60+"]).astype(str)
feats = tier_labels("T1")
y = df[OUTCOME_LABEL].values.astype(int)
idx = stratified_splits(y, a.seed, tuple(cfg["split_fracs"]))
G = df[a.group].astype(str).values
pq = pd.read_parquet(
    f"results/predictions/brfss2022/T1/{a.model}/seed{a.seed}.parquet")
val = pq[pq.split == "val"].sort_values("row_id")
tst = pq[pq.split == "test"].sort_values("row_id")
yv, pv, gv = val.y.values, val.p.values, G[val.row_id.values]
yt, pt, gt = tst.y.values, tst.p.values, G[tst.row_id.values]
# O(n) position maps: raw-row-id -> position in split arrays
pos_te = {int(r): i for i, r in enumerate(np.asarray(idx["test"]))}
order_te = np.array([pos_te[int(r)] for r in tst.row_id.values])
t_star = smallest_t_sens(yv, pv, 0.85)
tau_star = sens_at(yv, pv, t_star)
groups_u = np.unique(gv)
arms = {}
arms["baseline"] = (pt, {g: t_star for g in groups_u})
arms["group_thr_sens85"] = (pt, {g: smallest_t_sens(
    yv[gv == g], pv[gv == g], 0.85) for g in groups_u})
arms["group_thr_eqTPR"] = (pt, {g: smallest_t_sens(
    yv[gv == g], pv[gv == g], tau_star) for g in groups_u})

X = df[feats]
if a.group == "Sex":
    n = len(idx["train"]); ytr = y[idx["train"]]; gtr = G[idx["train"]]
    w = np.ones(n)
    for g in np.unique(gtr):
        for c in (0, 1):
            m = (gtr == g) & (ytr == c)
            w[m] = n / (len(np.unique(gtr)) * 2 * max(m.sum(), 1))
    params = dict(model_params(cfg, a.model, "T1"))
    rw = R.make_model(a.model, params, a.seed)
    rw.extra_weight_ = w
    if a.model == "ebm":
        _orig = R.balanced_sample_weight
        R.balanced_sample_weight = lambda yy, _w=w, _o=_orig: _o(yy) * _w
    rw.fit(X.iloc[idx["train"]], ytr, X.iloc[idx["val"]], y[idx["val"]])
    if a.model == "ebm":
        R.balanced_sample_weight = _orig
    pv_rw = rw.predict_proba_pos(X.iloc[idx["val"]])
    pt_rw = rw.predict_proba_pos(X.iloc[idx["test"]])[order_te]
    t_rw = smallest_t_sens(y[np.asarray(idx["val"])], pv_rw, 0.85)
    arms["reweigh"] = (pt_rw, {g: t_rw for g in groups_u})

if a.model == "ebm" and a.group == "Sex":
    base = load_any("ebm", f"results/models/brfss2022/T1/ebm/seed{a.seed}")
    ebm = getattr(base, "model_", base)
    names = list(ebm.term_names_)
    sidx = [i for i, nm in enumerate(names)
            if nm == "Sex" or nm.startswith("Sex &") or nm.endswith("& Sex")
            or nm.startswith("Sex x") or nm.endswith("x Sex")]
    print(f"[info] sex terms edited: {[names[i] for i in sidx]}")
    def edited(gamma):
        e2 = copy.deepcopy(ebm)
        for i in sidx:
            e2.term_scores_[i] = e2.term_scores_[i] * gamma
        return e2
    for tag, gam in (("repair_half", 0.5), ("repair_zero", 0.0)):
        e2 = edited(gam)
        pv2 = e2.predict_proba(X.iloc[idx["val"]])[:, 1]
        pt2 = e2.predict_proba(X.iloc[idx["test"]])[:, 1][order_te]
        t2 = smallest_t_sens(y[np.asarray(idx["val"])], pv2, 0.85)
        arms[tag] = (pt2, {g: t2 for g in groups_u})
    e2 = edited(0.0)
    pv2 = e2.predict_proba(X.iloc[idx["val"]])[:, 1]
    pt2 = e2.predict_proba(X.iloc[idx["test"]])[:, 1][order_te]
    lv = np.log(np.clip(pv2, 1e-9, 1 - 1e-9) /
                np.clip(1 - pv2, 1e-9, 1 - 1e-9))
    lt = np.log(np.clip(pt2, 1e-9, 1 - 1e-9) /
                np.clip(1 - pt2, 1e-9, 1 - 1e-9))
    yv_full = y[np.asarray(idx["val"])]
    gv_full = G[np.asarray(idx["val"])]
    cs = {}
    for g in groups_u:
        m = gv_full == g
        def tpr_c(c):
            pp = 1 / (1 + np.exp(-(lv[m] + c)))
            return sens_at(yv_full[m], pp, t_star)
        lo, hi = -3.0, 3.0
        for _ in range(50):
            mid = (lo + hi) / 2
            if tpr_c(mid) < tau_star:
                lo = mid
            else:
                hi = mid
        cs[g] = (lo + hi) / 2
    shift = np.vectorize(cs.get)(gt)
    pt_eq = 1 / (1 + np.exp(-(lt + shift)))
    eff = {g: float(1 / (1 + np.exp(-(np.log(t_star / (1 - t_star)) - cs[g]))))
           for g in cs}
    arms["repair_eq"] = (pt_eq, {g: t_star for g in groups_u})
    json.dump({"shifts": cs, "effective_thresholds": eff,
               "global_t": t_star, "tau_star": float(tau_star)},
              open(out / "equalise_shifts.json", "w"), indent=1)

rows = []
store_p, store_t = {}, {}
for arm, (p_arm, thr) in arms.items():
    gm = group_metrics(yt, p_arm, gt, thr)
    tvec = np.vectorize(thr.get)(gt)
    pr = p_arm >= tvec
    row = dict(model=a.model, seed=a.seed, group_var=a.group, arm=arm,
               sensitivity=float(pr[yt == 1].mean()),
               specificity=float((~pr)[yt == 0].mean()))
    gs = sorted(gm)
    for met in ("TPR", "FPR", "PPV"):
        row[f"{met}_gap"] = max(gm[g][met] for g in gs) - \
                            min(gm[g][met] for g in gs)
        for g in gs:
            row[f"{met}_{g}"] = gm[g][met]
    for g in gs:
        row[f"n_{g}"] = gm[g]["n"]
        row[f"thr_{g}"] = float(thr[g])
    if arm == "repair_eq":
        for g in gs:
            row[f"thr_{g}"] = eff[g]
    rows.append(row)
    store_p[arm] = p_arm
    store_t[arm] = {str(k): float(v) for k, v in thr.items()}
    if arm == "repair_eq":
        store_t[arm] = {str(k): eff[k] for k in eff}
pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
np.save(out / "y_test.npy", yt)
np.save(out / "g_test.npy", gt)
json.dump(store_t, open(out / "thresholds.json", "w"), indent=1)
np.savez_compressed(out / "arm_probs.npz", **store_p)
print(f"[done] fairness_v2 {a.model}/{a.group}/s{a.seed}: " +
      "  ".join(f"{r['arm']}:gap={r['TPR_gap']:.4f}" for r in rows))
PYEOF
python - <<'EOF'
import ast
ast.parse(open("experiments/run_fairness_v2.py").read())
print("run_fairness_v2 syntax OK")
EOF

echo "== hotfix 3/3: robust loader in pass-1 timing stage =="
python - <<'EOF'
s = open("revision_pass1.sh").read()
if "def load_any(" not in s:
    s = s.replace(
        "import src.models.registry as R\ncfg=load_config(",
        '''import src.models.registry as R
import joblib as _jl
def load_any(name, path):
    for attr in ("REGISTRY","MODELS","ADAPTERS"):
        reg = getattr(R, attr, None)
        if reg and name in reg and hasattr(reg[name], "load"):
            try: return reg[name].load(path)
            except Exception: pass
    return _jl.load(path + "/model.joblib")
cfg=load_config(''')
    s = s.replace('mod=R.REGISTRY[m].load(f"results/models/brfss2022/T1/{m}/seed0")',
                  'mod=load_any(m, f"results/models/brfss2022/T1/{m}/seed0")')
    open("revision_pass1.sh", "w").write(s)
    print("pass-1 timing stage patched")
else:
    print("pass-1 already patched")
EOF
echo "== HOTFIX DONE — now:  nohup bash revision_pass1.sh > results/revision.log 2>&1 & =="
