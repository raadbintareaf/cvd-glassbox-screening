#!/usr/bin/env bash
# revision_hotfix2.sh — run from repo root, then rerun pass 1b:
#   bash revision_hotfix2.sh
#   nohup bash revision_pass1b.sh > results/revision1b.log 2>&1 &
# Fix A: variable_map.py had TWO stacked tier_labels wrappers (my hotfix-1
#   marker mismatch) whose global rebinding caused infinite recursion on
#   every call and a kwarg TypeError on the nostroke path. This strips ALL
#   wrapper blocks by sentinel search and appends exactly one, then
#   FUNCTIONALLY asserts the two observed failure modes are gone.
# Fix B: temporal predictions carry no row_id column; a standalone
#   temporal_oe_rev.py handles that and pass 1b is rewired to call it.
set -euo pipefail
cd "$(dirname "$0")"
[ -f .venv/bin/activate ] && source .venv/bin/activate

echo "== hotfix2 A: reconstruct variable_map wrapper =="
python - <<'EOF'
import re, sys
p = "src/data/variable_map.py"
s = open(p).read()
cuts = [s.find(x) for x in ("_orig_tier_labels = tier_labels",
                            "_base_tier_labels_rev = tier_labels")]
cuts = [c for c in cuts if c >= 0]
if cuts:
    s = s[:min(cuts)].rstrip() + "\n"
    print(f"stripped wrapper block(s) from offset {min(cuts)}")
else:
    print("no prior wrapper found (clean file)")
s += '''

# --- revision tier T1_nostroke (M1): via the existing drop_stroke switch ---
_base_tier_labels_rev = tier_labels
def tier_labels(t, **kw):  # noqa: F811
    if t == "T1_nostroke":
        kw = dict(kw)
        kw["drop_stroke"] = True
        return _base_tier_labels_rev("T1", **kw)
    return _base_tier_labels_rev(t, **kw)
'''
open(p, "w").write(s)

# functional asserts against BOTH observed failure modes
sys.setrecursionlimit(200)          # recursion would now fail fast & loud
import importlib
sys.path.insert(0, ".")
import src.data.variable_map as V
importlib.reload(V)
t1 = V.tier_labels("T1")                          # plain call: no recursion
t1k = V.tier_labels("T1", drop_stroke=False)      # kwarg pass-through
ns = V.tier_labels("T1_nostroke")                 # new tier
ds = V.tier_labels("T1", drop_stroke=True)        # repo's own switch
t2 = V.tier_labels("T2")
assert "HadStroke" in t1 and "HadStroke" not in ns
assert ns == ds, "nostroke must equal drop_stroke=True"
assert len(ns) == len(t1) - 1 and t1 == t1k
print(f"ASSERTS PASS: T1={len(t1)}  T1_nostroke={len(ns)}  T2={len(t2)}  "
      f"nostroke==drop_stroke: True")
EOF

echo "== hotfix2 B: robust temporal O/E script + rewire pass 1b =="
cat > analysis/temporal_oe_rev.py << 'PYEOF'
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
PYEOF
python - <<'EOF'
s = open("revision_pass1b.sh").read()
a = s.index('echo "== [1b-9]')
b = s.index('echo "== [1b-10]')
s = s[:a] + ('echo "== [1b-9] Temporal subgroup O/E =="\n'
             'python -m analysis.temporal_oe_rev '
             '|| echo "[warn] temporal OE (traceback above)"\n\n') + s[b:]
open("revision_pass1b.sh", "w").write(s)
print("pass 1b stage 1b-9 rewired to temporal_oe_rev")
EOF
echo "== HOTFIX 2 DONE — now:  nohup bash revision_pass1b.sh > results/revision1b.log 2>&1 & =="
