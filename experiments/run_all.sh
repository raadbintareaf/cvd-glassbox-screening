#!/usr/bin/env bash
# Full paper grid. Safe to re-run: every step resumes/skips completed work.
# Expected wall time on 24 GB GPU + 128 GB RAM: see README section "Compute".
set -uo pipefail
CFG=${1:-configs/default.yaml}
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate

echo "== [0/8] environment freeze =="
mkdir -p results
pip freeze > results/environment_freeze.txt
python -c "import platform,sys;print(platform.platform());print(sys.version)" \
  >> results/environment_freeze.txt

echo "== [1/8] build analytic datasets (skips if present) =="
[ -f data/analytic/brfss_2022_analytic.parquet ] || \
  python -m src.data.build_dataset --year 2022 --xpt data/raw/LLCP2022.XPT --out data/analytic
[ -f data/analytic/brfss_2023_analytic.parquet ] || \
  python -m src.data.build_dataset --year 2023 --xpt data/raw/LLCP2023.XPT --out data/analytic

TIERS=$(python - "$CFG" <<'PY'
import sys,yaml;print(" ".join(yaml.safe_load(open(sys.argv[1]))["tiers"]))
PY
)
MODELS=$(python - "$CFG" <<'PY'
import sys,yaml;print(" ".join(yaml.safe_load(open(sys.argv[1]))["models"]))
PY
)
SEEDS=$(python - "$CFG" <<'PY'
import sys,yaml;print(" ".join(map(str,yaml.safe_load(open(sys.argv[1]))["seeds"])))
PY
)

echo "== [2/8] hyperparameter tuning (val only, seed-0 split) =="
for T in $TIERS; do for M in $MODELS; do
  F=results/tuned/${M}__${T}.yaml
  [ -f "$F" ] || python -m experiments.run_tuning --config "$CFG" --tier "$T" --model "$M" || echo "[warn] tuning $M/$T failed — defaults will be used"
done; done

echo "== [3/8] main benchmark grid =="
for T in $TIERS; do for M in $MODELS; do for S in $SEEDS; do
  python -m experiments.run_benchmark --config "$CFG" --tier "$T" --model "$M" --seed "$S" \
    || echo "[warn] benchmark $T/$M/seed$S failed — continuing"
done; done; done

echo "== [4/8] SMOTE-NC ablation (T1: ebm + xgboost) =="
for M in ebm xgboost; do for S in $SEEDS; do
  python -m experiments.run_benchmark --config "$CFG" --tier T1 --model "$M" --seed "$S" --imbalance smotenc \
    || echo "[warn] smotenc $M/seed$S failed — continuing"
done; done

echo "== [5/8] fairness audit + mitigation arms (T1: ebm + xgboost) =="
for M in ebm xgboost; do for S in $SEEDS; do
  python -m experiments.run_fairness --config "$CFG" --tier T1 --model "$M" --seed "$S" --group Sex \
    || echo "[warn] fairness $M/seed$S failed — continuing"
done; done

echo "== [6/8] conformal analysis (T1: ebm, xgboost, tabicl) =="
for M in ebm xgboost tabicl; do for S in $SEEDS; do
  python -m experiments.run_conformal --config "$CFG" --tier T1 --model "$M" --seed "$S" \
    || echo "[warn] conformal $M/seed$S failed — continuing"
done; done

echo "== [7/8] faithfulness sweep (T1, seed 0; add --include-fm on GPU) =="
python -m experiments.run_faithfulness --config "$CFG" --tier T1 --seed 0 --include-fm \
  || python -m experiments.run_faithfulness --config "$CFG" --tier T1 --seed 0 \
  || echo "[warn] faithfulness failed"

echo "== [8/8] temporal external validation on BRFSS 2023 (T1_portable) =="
for M in $MODELS; do for S in $SEEDS; do
  python -m experiments.run_temporal --config "$CFG" --model "$M" --seed "$S" \
    || echo "[warn] temporal $M/seed$S failed — continuing"
done; done
python -m experiments.run_temporal --config "$CFG" --model ebm --seed 0 --refit-ebm-2023 || true

echo "== DONE. Now: python -m analysis.aggregate_results --config $CFG =="
