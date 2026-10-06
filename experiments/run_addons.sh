#!/usr/bin/env bash
# Add-on grid: CatBoost + Random Forest (tuning, benchmark, temporal) and the
# three sensitivity analyses. Resume-safe; CPU-only. Does NOT rerun run_all.
set -uo pipefail
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate
CFG=${1:-configs/default.yaml}
TIERS=$(python - "$CFG" <<'PY'
import sys,yaml;print(" ".join(yaml.safe_load(open(sys.argv[1]))["tiers"]))
PY
)
SEEDS=$(python - "$CFG" <<'PY'
import sys,yaml;print(" ".join(map(str,yaml.safe_load(open(sys.argv[1]))["seeds"])))
PY
)
echo "== [A] tuning: catboost + random_forest =="
for T in $TIERS; do for M in catboost random_forest; do
  [ -f results/tuned/${M}__${T}.yaml ] || python -m experiments.run_tuning --config "$CFG" --tier "$T" --model "$M" || echo "[warn] tuning $M/$T failed"
done; done
echo "== [B] benchmark: catboost + random_forest =="
for T in $TIERS; do for M in catboost random_forest; do for S in $SEEDS; do
  python -m experiments.run_benchmark --config "$CFG" --tier "$T" --model "$M" --seed "$S" || echo "[warn] $T/$M/s$S failed"
done; done; done
echo "== [C] temporal 2023: catboost + random_forest =="
for M in catboost random_forest; do for S in $SEEDS; do
  python -m experiments.run_temporal --config "$CFG" --model "$M" --seed "$S" || echo "[warn] temporal $M/s$S failed"
done; done
echo "== [D] complete-case ablation =="
for M in ebm xgboost; do for S in $SEEDS; do
  python -m experiments.run_complete_case --config "$CFG" --model "$M" --seed "$S" || echo "[warn] cc $M/s$S failed"
done; done
echo "== [E] recalibration + weighted sensitivity =="
python -m experiments.run_recalibration --config "$CFG" || echo "[warn] recalibration failed"
python -m experiments.run_weighted --config "$CFG" || echo "[warn] weighted failed"
echo "== ADDONS DONE =="
