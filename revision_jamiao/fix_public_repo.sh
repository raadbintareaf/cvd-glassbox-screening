#!/usr/bin/env bash
# Repairs the public GitHub repository so a reviewer can actually run it.
#
# THE PROBLEM
#   .gitignore contains "data/", which git applies at EVERY level, so it also
#   ignored src/data/. As a result src/data/build_dataset.py and
#   src/data/variable_map.py were never pushed, and the public repository
#   cannot build the dataset or run any experiment (ten scripts import
#   src.data). The public src/models/registry.py is also an older copy that
#   lacks the Random forest and CatBoost models reported in the paper.
#
# THE FIX (this script)
#   1. anchors the ignore rules to the repository root (/data/ instead of data/)
#   2. shows you exactly which files would be added
#   3. only if you type "yes": commits and pushes them
#
# Run it on the SERVER copy of the code (the one the paper's runs used):
#   cd ~/path/to/cvd-glassbox-screening
#   bash revision_jamiao/fix_public_repo.sh
set -uo pipefail
cd "$(dirname "$0")/.."

echo "== 1. Checking that the missing files exist here =="
missing=0
for f in src/data/variable_map.py src/data/build_dataset.py; do
  if [ -f "$f" ]; then echo "  found   $f"; else echo "  MISSING $f"; missing=1; fi
done
[ -f src/data/__init__.py ] || { echo "  creating src/data/__init__.py"; : > src/data/__init__.py; }
if [ "$missing" = 1 ]; then
  echo
  echo "This folder does not have the data module either, so it is not the copy"
  echo "the paper was run from. Run this script in the folder where you ran"
  echo "experiments/run_all.sh. Nothing was changed."
  exit 1
fi
grep -q "class CatBoostAdapter" src/models/registry.py \
  && echo "  found   CatBoost in src/models/registry.py" \
  || echo "  WARNING src/models/registry.py has no CatBoost: is this the copy used for the paper?"

echo
echo "== 2. Anchoring .gitignore rules to the repository root =="
python - <<'PY'
from pathlib import Path
p = Path(".gitignore")
lines = p.read_text().splitlines()
fixed = []
for ln in lines:
    s = ln.strip()
    if s in ("data/", "results/", "paper/"):
        fixed.append("/" + s); print(f"  {s:10s} -> /{s}")
    else:
        fixed.append(ln)
p.write_text("\n".join(fixed) + "\n")
PY

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  cat <<'TXT'

This folder is not a git checkout, so nothing can be pushed from here.
Do this instead (about two minutes):

  git clone https://github.com/raadbintareaf/cvd-glassbox-screening.git ~/cvd-public
  cp -r src/data            ~/cvd-public/src/
  cp src/models/registry.py ~/cvd-public/src/models/
  cp -r configs revision_jamiao ~/cvd-public/
  cp .gitignore             ~/cvd-public/
  cd ~/cvd-public
  git add -A && git status        # check the list, then:
  git commit -m "Add missing src/data module and current model registry"
  git push

TXT
  exit 0
fi

echo
echo "== 3. Files that would be committed =="
git add .gitignore src/data src/models/registry.py configs revision_jamiao \
        experiments analysis src/evaluation src/repair 2>/dev/null
git status --short
echo
read -r -p "Commit and push these files? Type yes to continue: " ans
if [ "$ans" != "yes" ]; then
  git reset -q
  echo "Nothing committed. The .gitignore change is kept locally."
  exit 0
fi
git commit -m "Add missing src/data module and current model registry

src/data/ was excluded by an unanchored 'data/' rule in .gitignore, so the
public repository could not build the dataset or run any experiment.
Also adds the Random forest and CatBoost adapters used in the paper and
the analyses for the JAMIA Open revision (revision_jamiao/)."
git push && echo && echo "Pushed. Then make a new Zenodo version so the archived DOI matches:" \
  && echo "  bash release_to_github_zenodo.sh"
