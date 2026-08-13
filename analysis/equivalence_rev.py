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
