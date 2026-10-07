#!/usr/bin/env bash
# One-command runner for the additional BMC analyses.
#
#   cd ~/Papers/cvd-glassbox-screening
#   bash revision_bmc/run.sh            # everything (about 1-3 hours, see below)
#   bash revision_bmc/run.sh --quick    # skip retraining (about 10 minutes)
#
# Most analyses read saved predictions (minutes). Parts B6/B7 retrain the
# eight classical models, 5 seeds x 3 variants; this is the slow part and is
# resumable: if it stops, run the same command again and it continues.
#
# It runs in the background, so you can close the terminal. To watch it:
#   tail -f results/bmc_extra/run.log
# When it finishes, the file to send back is:  bmc_extra_bundle.tgz
set -uo pipefail
cd "$(dirname "$0")/.."
CONFIG="${CONFIG:-configs/default.yaml}"
EXTRA=""
[ "${1:-}" = "--quick" ] && EXTRA="--skip-retrain"

python -c "import pandas, sklearn, scipy, pyarrow, yaml, interpret" 2>/dev/null || {
  echo "STOPPED: activate the environment used for the paper first (conda activate ...)"; exit 1; }
[ -d results/predictions/brfss2022/T0 ] || { echo "STOPPED: results/predictions/brfss2022/T0 not found. Run from the project folder."; exit 1; }
[ -f data/analytic/brfss_2022_analytic.parquet ] || { echo "STOPPED: data/analytic/brfss_2022_analytic.parquet not found."; exit 1; }

mkdir -p results/bmc_extra
LOG=results/bmc_extra/run.log
nohup bash -c "
  python -u -m revision_bmc.bmc_extra_analyses --config $CONFIG $EXTRA
  cd results && tar --exclude='bmc_extra/predictions' -czf ../bmc_extra_bundle.tgz bmc_extra
  echo
  echo '=================================================================='
  echo 'FINISHED. Send this file to Claude:'
  echo \"  \$(cd .. && pwd)/bmc_extra_bundle.tgz\"
  echo '=================================================================='
" > "$LOG" 2>&1 &

echo "Started in the background (process $!)."
echo "Watch progress:   tail -f $LOG      (Ctrl+C stops watching, not the run)"
echo "When the log says FINISHED, download bmc_extra_bundle.tgz and upload it to Claude."
