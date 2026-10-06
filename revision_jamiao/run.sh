#!/usr/bin/env bash
# One-command runner for the JAMIA Open revision analyses.
#
#   cd ~/path/to/cvd-glassbox-screening
#   bash revision_jamiao/run.sh
#
# CPU only. Reads saved predictions; retrains nothing. A few minutes.
# When it finishes it prints the path of ONE file to send back:
#   jamiao_rev_bundle.tgz
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
CONFIG="${1:-configs/default.yaml}"

say()  { printf '\n\033[1m%s\033[0m\n' "$*"; }
fail() { printf '\n\033[31mSTOPPED: %s\033[0m\n' "$*"; exit 1; }

say "1/4  Checking the environment"
if   [ -f .venv/bin/activate ]; then source .venv/bin/activate; echo "  using .venv"
elif [ -n "${VIRTUAL_ENV:-}" ];  then echo "  using active venv $VIRTUAL_ENV"
else echo "  no .venv found; using system python ($(command -v python))"; fi
python -c "import pandas, sklearn, scipy, pyarrow, matplotlib" 2>/dev/null \
  || fail "missing Python packages. Run: pip install -r requirements.txt"

say "2/4  Checking the inputs this needs"
[ -f "$CONFIG" ]                        || fail "config $CONFIG not found (are you in the repository folder?)"
[ -f src/data/variable_map.py ]         || fail "src/data/variable_map.py not found. Run this on the server copy of the code, where the original runs were made."
RES=$(python -c "import yaml;print(yaml.safe_load(open('$CONFIG'))['results_dir'])")
DAT=$(python -c "import yaml;print(yaml.safe_load(open('$CONFIG'))['data_dir'])")
[ -d "$RES/predictions/brfss2022/T0" ]  || fail "$RES/predictions/brfss2022/T0 not found. The benchmark predictions are needed."
[ -d "$RES/predictions/brfss2022/T1" ]  || fail "$RES/predictions/brfss2022/T1 not found."
[ -f "$RES/raw_results.csv" ]           || fail "$RES/raw_results.csv not found."
[ -f "$DAT/brfss_2022_analytic.parquet" ] || fail "$DAT/brfss_2022_analytic.parquet not found."
echo "  predictions : $(find "$RES/predictions/brfss2022" -name 'seed*.parquet' | wc -l) files"
echo "  results csv : $(wc -l < "$RES/raw_results.csv") rows"
echo "  2022 cohort : found"
[ -f "$DAT/brfss_2023_analytic.parquet" ] && echo "  2023 cohort : found" || echo "  2023 cohort : not found (missingness will be 2022 only)"

echo
echo "  Fixing src/evaluation/metrics.py:delong_test (order bug; a backup is kept)"
python - <<'PY'
from pathlib import Path
p = Path("src/evaluation/metrics.py")
s = p.read_text()
if "order = np.concatenate([np.where(y == 1)[0], np.where(y == 0)[0]])" in s:
    print("    already fixed")
else:
    old = ('    """Paired DeLong test for AUC(p1) vs AUC(p2). Returns (auc1, auc2, p)."""\n'
           '    y = np.asarray(y)\n')
    new = ('    """Paired DeLong test for AUC(p1) vs AUC(p2). Returns (auc1, auc2, p).\n'
           '    Positives are moved first: the fast algorithm requires it, and the\n'
           '    saved prediction files are sorted by row_id, not by class."""\n'
           '    y = np.asarray(y)\n'
           '    order = np.concatenate([np.where(y == 1)[0], np.where(y == 0)[0]])\n'
           '    y = y[order]\n'
           '    p1, p2 = np.asarray(p1)[order], np.asarray(p2)[order]\n')
    if old in s:
        Path(str(p) + ".bak_before_delong_fix").write_text(s)
        p.write_text(s.replace(old, new, 1))
        print("    fixed (backup: src/evaluation/metrics.py.bak_before_delong_fix)")
    else:
        print("    WARNING: could not find the expected code; left unchanged.")
        print("    (The analyses below use their own correct DeLong, so they are unaffected.)")
PY

say "3/4  Running the analyses (a few minutes)"
mkdir -p "$RES/jamiao_rev"
python -m revision_jamiao.jamiao_analyses --config "$CONFIG" 2>&1 | tee "$RES/jamiao_rev/run.log"
STATUS=${PIPESTATUS[0]}
pip freeze > "$RES/jamiao_rev/environment.txt" 2>/dev/null || true
python --version > "$RES/jamiao_rev/python_version.txt" 2>&1 || true

say "4/4  Packing the results"
tar czf jamiao_rev_bundle.tgz -C "$RES" jamiao_rev
echo
if [ "$STATUS" -eq 0 ]; then
  printf '\033[32mDONE. Send back this one file:\033[0m\n  %s/jamiao_rev_bundle.tgz\n' "$ROOT"
else
  printf '\033[33mFINISHED WITH SOME FAILURES (see run.log inside the bundle).\033[0m\n'
  printf 'Send back this file anyway; the failed parts are listed in run_status.json:\n  %s/jamiao_rev_bundle.tgz\n' "$ROOT"
fi
echo "Quick look at the key numbers:  less $RES/jamiao_rev/SUMMARY.md"
