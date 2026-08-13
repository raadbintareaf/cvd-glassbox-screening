#!/usr/bin/env bash
# revision_pass1c.sh — LAST server touch. Run from repo root:
#   nohup bash revision_pass1c.sh > results/revision1c.log 2>&1 &
#   tail -f results/revision1c.log
# Closes the two remaining gaps: (1) formal 1000x paired fairness bootstrap;
# (2) T1-nostroke x {tabpfn_v2, tabicl} x 5 seeds (GPU, ~2-3 h total).
# Tracebacks are NOT suppressed. Upload revision_bundle3.tgz at the end.
set -uo pipefail
cd "$(dirname "$0")"
[ -f .venv/bin/activate ] && source .venv/bin/activate

echo "== [1c-1] GPU headroom =="
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv || true

echo "== [1c-2] Fairness paired bootstrap (M6) =="
python -m analysis.fairness_bootstrap_rev
if [ $? -ne 0 ]; then
  echo "[FATAL] bootstrap failed — traceback above (check arm_probs.npz exist):"
  ls -la results/revision/fairness_v2/ebm/Sex/seed0/ || true
fi

echo "== [1c-3] T1_nostroke foundation models (M1) =="
for M in tabpfn_v2 tabicl; do
  for S in 0 1 2 3 4; do
    python -m experiments.run_benchmark --config configs/default.yaml \
      --tier T1_nostroke --model $M --seed $S \
      || echo "[warn] ns $M/s$S (traceback above)"
  done
done

echo "== [1c-4] Aggregate + pack =="
python -m analysis.aggregate_results --config configs/default.yaml \
  || echo "[warn] aggregate"
tar czf revision_bundle3.tgz \
    results/revision/fairness_bootstrap.csv \
    results/raw_results.csv results/summary.csv 2>/dev/null || \
tar czf revision_bundle3.tgz results/raw_results.csv results/summary.csv
ls -lh revision_bundle3.tgz
echo "== PASS 1C DONE — upload revision_bundle3.tgz =="
