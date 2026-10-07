#!/usr/bin/env bash
# Second, short round of checks (about 10-15 minutes, no retraining).
#   cd ~/Papers/cvd-glassbox-screening
#   bash revision_bmc/run2.sh
# Then upload bmc_extra2_bundle.tgz to Claude.
set -uo pipefail
cd "$(dirname "$0")/.."
CONFIG="${CONFIG:-configs/default.yaml}"
python -c "import pandas, sklearn, scipy, pyarrow, yaml" 2>/dev/null || {
  echo "STOPPED: activate the environment used for the paper first (conda activate ...)"; exit 1; }
[ -d results/predictions/brfss2022/T0 ] || { echo "STOPPED: run from the project folder."; exit 1; }
mkdir -p results/bmc_extra2
python -u -m revision_bmc.bmc_extra2 --config "$CONFIG" 2>&1 | tee results/bmc_extra2/run.log
(cd results && tar -czf ../bmc_extra2_bundle.tgz bmc_extra2)
echo
echo "FINISHED. Upload this file to Claude:  $(pwd)/bmc_extra2_bundle.tgz"
