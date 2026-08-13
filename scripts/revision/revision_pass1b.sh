#!/usr/bin/env bash
# revision_pass1b.sh — closes the gaps from pass 1. Run from repo root:
#   nohup bash revision_pass1b.sh > results/revision1b.log 2>&1 &
#   tail -f results/revision1b.log
# NOTHING is suppressed here: failures print full tracebacks and the run
# continues, so one log diagnoses everything. Upload revision_bundle2.tgz
# when "== PASS 1B DONE ==" prints; if any stage shows a Traceback, also
# paste me:  grep -B2 -A18 Traceback results/revision1b.log | head -160
set -uo pipefail
cd "$(dirname "$0")"
[ -f .venv/bin/activate ] && source .venv/bin/activate
mkdir -p results/revision

echo "== [P] Preflight diagnostics =="
pip install -q lime 2>&1 | tail -1
python - <<'EOF'
import importlib, subprocess, sys
sys.path.insert(0, ".")
import src.models.registry as R
print("registry attrs:", [a for a in ("REGISTRY","MODELS","ADAPTERS",
      "balanced_sample_weight","make_model") if hasattr(R, a)])
from src.data.variable_map import tier_labels
a, b = tier_labels("T1"), tier_labels("T1_nostroke")
print(f"tiers: T1={len(a)}  T1_nostroke={len(b)}  "
      f"stroke_removed={'HadStroke' not in b}")
try:
    import lime; print("lime OK", lime.__file__.split('site-packages')[-1])
except Exception as e:
    print("lime IMPORT FAILED:", e)
import yaml
cfg = yaml.safe_load(open("configs/default.yaml"))
print("config tiers:", cfg.get("tiers"))
import pandas as pd
df = pd.read_parquet("data/analytic/brfss_2022_analytic.parquet",
                     columns=None).head(1)
print("has SurveyWeight:", "SurveyWeight" in df.columns,
      "| has RaceEthnicity:", "RaceEthnicity" in df.columns)
EOF

echo "== [1b-2] T1_nostroke: PROBE then full grid (M1) =="
python -m experiments.run_benchmark --config configs/default.yaml \
  --tier T1_nostroke --model logreg --seed 0
if [ $? -ne 0 ]; then
  echo "[FATAL] nostroke probe failed — see traceback above; grid skipped"
else
  for M in logreg logreg_spline random_forest ebm xgboost lightgbm catboost mlp tabpfn_v2 tabicl; do
    for S in 0 1 2 3 4; do
      python -m experiments.run_benchmark --config configs/default.yaml \
        --tier T1_nostroke --model $M --seed $S \
        || echo "[warn] ns $M/s$S (traceback above)"
    done
  done
fi

echo "== [1b-3] CatBoost unweighted seeds 1-4 (M4) =="
for S in 1 2 3 4; do
  python -m experiments.run_unweighted_rev --model catboost --seed $S \
    || echo "[warn] unw catboost/s$S"
done

echo "== [1b-4] Fairness v2 + bootstrap (M6) — probe first =="
python -m experiments.run_fairness_v2 --model ebm --seed 0 --group Sex
if [ $? -ne 0 ]; then
  echo "[FATAL] fairness_v2 probe failed — see traceback above"
else
  for M in ebm xgboost; do for S in 0 1 2 3 4; do
    python -m experiments.run_fairness_v2 --model $M --seed $S --group Sex \
      || echo "[warn] fv2 $M/Sex/s$S"
  done; done
  for GV in AgeBand RaceEthnicity; do for M in ebm xgboost; do
    python -m experiments.run_fairness_v2 --model $M --seed 0 --group $GV \
      || echo "[warn] fv2 $M/$GV"
  done; done
  python -m analysis.fairness_bootstrap_rev || echo "[warn] bootstrap"
fi

echo "== [1b-5] LIME on EBM (M8) =="
python -m analysis.lime_on_ebm_rev || echo "[warn] ebm_lime (traceback above)"

echo "== [1b-7] Weight-informed training (M9) — probe first =="
python -m experiments.run_weighted_train_rev --model ebm --seed 0
if [ $? -ne 0 ]; then
  echo "[FATAL] weighted_train probe failed — see traceback above"
else
  for M in ebm xgboost catboost; do for S in 0 1 2 3 4; do
    python -m experiments.run_weighted_train_rev --model $M --seed $S \
      || echo "[warn] wt $M/s$S"
  done; done
fi

echo "== [1b-8] Timing v2 + hardware (M5) =="
python - <<'EOF'
import json, platform, subprocess, time, traceback
import numpy as np, joblib, torch
from src.utils.common import load_config, stratified_splits, seed_everything
from src.data.variable_map import OUTCOME_LABEL, tier_labels
from experiments.run_benchmark import load_analytic
import src.models.registry as R
def load_any(name, path):
    for attr in ("REGISTRY", "MODELS", "ADAPTERS"):
        reg = getattr(R, attr, None)
        if reg and name in reg and hasattr(reg[name], "load"):
            try: return reg[name].load(path)
            except Exception: pass
    return joblib.load(path + "/model.joblib")
cfg = load_config("configs/default.yaml")
df = load_analytic(cfg, "brfss2022")
seed_everything(0)
y = df[OUTCOME_LABEL].values.astype(int)
idx = stratified_splits(y, 0, tuple(cfg["split_fracs"]))
X = df[tier_labels("T1")].iloc[idx["test"]]
def med(fn, reps=7):
    fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return (float(np.median(ts)), float(np.percentile(ts, 25)),
            float(np.percentile(ts, 75)))
out = {}
for m in ["ebm", "catboost", "tabpfn_v2", "tabicl"]:
    try:
        mod = load_any(m, f"results/models/brfss2022/T1/{m}/seed0")
        fn = (lambda mod=mod: mod.predict_proba_pos(X)) \
            if hasattr(mod, "predict_proba_pos") \
            else (lambda mod=mod: mod.predict_proba(X)[:, 1])
        mm, q1, q3 = med(fn)
        out[m] = dict(median_s=round(mm, 3), iqr=[round(q1, 3), round(q3, 3)])
        print(m, out[m], flush=True)
    except Exception:
        traceback.print_exc()
cpu = [l.split(":")[1].strip() for l in open("/proc/cpuinfo")
       if "model name" in l][0]
hw = dict(cpu=cpu,
          threads=int(subprocess.run(["nproc"], capture_output=True,
                                     text=True).stdout),
          gpu=torch.cuda.get_device_name(0)
          if torch.cuda.is_available() else None,
          torch=torch.__version__, python=platform.python_version(),
          note=("no FM inference optimisation (batching tuning, "
                "quantisation, or context caching) was attempted"),
          n_test=int(len(X)), timings=out)
json.dump(hw, open("results/revision/timing_hw.json", "w"), indent=1)
print(json.dumps(hw, indent=1))
EOF

echo "== [1b-9] Temporal subgroup O/E (broadened paths) =="
python - <<'EOF'
import traceback
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
from src.utils.common import load_config
from experiments.run_benchmark import load_analytic
try:
    cfg = load_config("configs/default.yaml")
    d23 = load_analytic(cfg, "brfss2023")
    d23["AgeBand"] = pd.cut(d23["AgeCategory"], [0, 4.5, 8.5, 99],
                            labels=["18-39", "40-59", "60+"]).astype(str)
    cands = sorted(Path("results/temporal").rglob("*.parquet"))
    print("temporal parquet files found:",
          *[str(c) for c in cands[:12]], sep="\n  ")
    rows = []
    for m in ["ebm", "catboost", "tabicl"]:
        tp = [c for c in cands if f"/{m}/" in str(c) or f"{m}" in c.parts[-2]
              or m in c.stem]
        tp = [c for c in tp if "2023" in str(c)]
        if not tp:
            print("[warn] no 2023 predictions found for", m); continue
        t = pd.read_parquet(tp[0])
        v = pd.read_parquet(
            f"results/predictions/brfss2022/T1_portable/{m}/seed0.parquet")
        v = v[v.split == "val"]
        iso = IsotonicRegression(out_of_bounds="clip").fit(v.p, v.y)
        pc = iso.predict(t.p)
        for gv in ("Sex", "AgeBand"):
            G = d23[gv].astype(str).values[t.row_id.values]
            for g in np.unique(G):
                mk = G == g
                rows.append(dict(model=m, group_var=gv, group=g,
                    OE=float(t.y.values[mk].mean() /
                             max(pc[mk].mean(), 1e-9)), n=int(mk.sum())))
    pd.DataFrame(rows).round(4).to_csv(
        "results/revision/temporal_subgroup_OE.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
except Exception:
    traceback.print_exc()
EOF

echo "== [1b-10] Aggregate + pack =="
python -m analysis.aggregate_results --config configs/default.yaml \
  || echo "[warn] aggregate"
tar czf revision_bundle2.tgz results/revision results/raw_results.csv \
    results/summary.csv
ls -lh revision_bundle2.tgz
echo "== PASS 1B DONE — upload revision_bundle2.tgz =="
