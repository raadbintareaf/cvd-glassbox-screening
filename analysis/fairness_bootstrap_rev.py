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
