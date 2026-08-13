"""Temporal per-subgroup calibration-in-the-large (O/E) on 2023.
Handles prediction parquets WITHOUT row_id (assumed full-cohort order)."""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
from src.utils.common import load_config
from experiments.run_benchmark import load_analytic

cfg = load_config("configs/default.yaml")
d23 = load_analytic(cfg, "brfss2023")
d23["AgeBand"] = pd.cut(d23["AgeCategory"], [0, 4.5, 8.5, 99],
                        labels=["18-39", "40-59", "60+"]).astype(str)
rows = []
for m in ["ebm", "catboost", "tabicl"]:
    f = Path(f"results/temporal/{m}/seed0/predictions_2023.parquet")
    if not f.exists():
        print("[warn] missing", f)
        continue
    t = pd.read_parquet(f)
    print(m, "columns:", list(t.columns), "n:", len(t))
    pcol = "p" if "p" in t.columns else \
           [c for c in t.columns if c.startswith("p")][0]
    ycol = "y" if "y" in t.columns else \
           [c for c in t.columns if c in ("label", "y_true", "target")][0]
    if "row_id" in t.columns:
        pos = t["row_id"].values
    else:
        assert len(t) == len(d23), \
            f"no row_id and length mismatch: {len(t)} vs {len(d23)}"
        pos = np.arange(len(d23))
    v = pd.read_parquet(
        f"results/predictions/brfss2022/T1_portable/{m}/seed0.parquet")
    v = v[v.split == "val"]
    iso = IsotonicRegression(out_of_bounds="clip").fit(v.p, v.y)
    pc = iso.predict(t[pcol].values)
    yv = t[ycol].values
    for gv in ("Sex", "AgeBand"):
        G = d23[gv].astype(str).values[pos]
        for g in np.unique(G):
            mk = G == g
            rows.append(dict(model=m, group_var=gv, group=g,
                             OE=float(yv[mk].mean() /
                                      max(pc[mk].mean(), 1e-9)),
                             n=int(mk.sum())))
out = pd.DataFrame(rows).round(4)
out.to_csv("results/revision/temporal_subgroup_OE.csv", index=False)
print(out.to_string(index=False))
