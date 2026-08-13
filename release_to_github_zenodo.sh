#!/usr/bin/env bash
# release_to_github_zenodo.sh — run from the repo root ON YOUR SERVER:
#   bash release_to_github_zenodo.sh
# Interactively: (1) pushes this code to a GitHub repo under your account,
# (2) tags release v2.0.0, (3) deposits a snapshot on Zenodo and mints a
# DOI. You will be asked to PASTE two tokens when prompted (input hidden):
#   GitHub : https://github.com/settings/tokens  -> "Generate new token
#            (classic)" -> scope: [repo] -> copy the ghp_... string
#   Zenodo : https://zenodo.org/account/settings/applications/tokens/new/
#            -> scopes: [deposit:write] [deposit:actions] -> copy token
# Nothing is stored; tokens live only in this shell session.
set -euo pipefail
command -v git >/dev/null || { echo "git required"; exit 1; }
command -v curl >/dev/null || { echo "curl required"; exit 1; }
command -v python3 >/dev/null || { echo "python3 required"; exit 1; }

echo "== [1/6] GitHub account details =="
read -rp "GitHub username: " GHUSER
read -rp "Repository name [cvd-glassbox-screening]: " GHREPO
GHREPO=${GHREPO:-cvd-glassbox-screening}
read -rp "Visibility (public/private) [public]: " VIS
VIS=${VIS:-public}
read -rsp "Paste your GitHub token (hidden): " GHTOKEN; echo
read -rp "Commit author name [Raad Bin Tareaf]: " GNAME
GNAME=${GNAME:-Raad Bin Tareaf}
read -rp "Commit author email: " GEMAIL

echo "== [2/6] Creating GitHub repository (idempotent) =="
PRIV=false; [ "$VIS" = "private" ] && PRIV=true
CODE=$(curl -s -o /tmp/ghcreate.json -w "%{http_code}" \
  -H "Authorization: token $GHTOKEN" -H "Accept: application/vnd.github+json" \
  https://api.github.com/user/repos \
  -d "{\"name\":\"$GHREPO\",\"private\":$PRIV,\"description\":\"Leakage-tiered, fairness-audited benchmark of glass-box and tabular foundation models on BRFSS (npj Digital Medicine submission code)\"}")
if [ "$CODE" = "201" ]; then echo "  repo created";
elif [ "$CODE" = "422" ]; then echo "  repo already exists — continuing";
else echo "  GitHub API returned $CODE:"; cat /tmp/ghcreate.json; exit 1; fi

echo "== [3/6] Committing and pushing =="
if [ ! -d .git ]; then git init -b main >/dev/null; fi
git config user.name  "$GNAME"
git config user.email "$GEMAIL"
git add -A
git commit -m "Release v2.0.0 — npj Digital Medicine revision code" \
  >/dev/null || echo "  (nothing new to commit)"
git remote remove origin 2>/dev/null || true
git remote add origin "https://$GHUSER:$GHTOKEN@github.com/$GHUSER/$GHREPO.git"
git push -u origin main
git tag -f v2.0.0
git push -f origin v2.0.0
curl -s -H "Authorization: token $GHTOKEN" \
  -H "Accept: application/vnd.github+json" \
  https://api.github.com/repos/$GHUSER/$GHREPO/releases \
  -d '{"tag_name":"v2.0.0","name":"v2.0.0 — npj revision code release","body":"Complete pipeline for the leakage-tiered, fairness-audited BRFSS benchmark, including all major-revision analyses (equivalence/TOST, T1-ns tier, fairness v2 with paired bootstrap, conformal v2 with empty-set accounting, survey-weighting analyses, LIME-on-EBM, timing manifest)."}' \
  > /tmp/ghrel.json
echo "  GitHub release: https://github.com/$GHUSER/$GHREPO/releases/tag/v2.0.0"

echo "== [4/6] Zenodo deposit =="
read -rsp "Paste your Zenodo token (hidden): " ZTOKEN; echo
ZBASE=${ZENODO_BASE:-https://zenodo.org}
git archive --format=zip -o /tmp/${GHREPO}-v2.0.0.zip v2.0.0
DEP=$(python3 - "$ZTOKEN" "$ZBASE" <<'PY'
import json, sys, urllib.request
tok, base = sys.argv[1], sys.argv[2]
meta = json.load(open(".zenodo.json"))
req = urllib.request.Request(
    f"{base}/api/deposit/depositions",
    data=json.dumps({"metadata": meta}).encode(),
    headers={"Content-Type": "application/json",
             "Authorization": f"Bearer {tok}"})
r = json.load(urllib.request.urlopen(req))
print(r["id"], r["links"]["bucket"])
PY
)
DEPID=$(echo "$DEP" | awk '{print $1}')
BUCKET=$(echo "$DEP" | awk '{print $2}')
echo "  deposition $DEPID created — uploading snapshot"
curl -s -H "Authorization: Bearer $ZTOKEN" \
  --upload-file /tmp/${GHREPO}-v2.0.0.zip \
  "$BUCKET/${GHREPO}-v2.0.0.zip" > /dev/null

echo "== [5/6] Publishing on Zenodo =="
PUB=$(curl -s -X POST -H "Authorization: Bearer $ZTOKEN" \
  "$ZBASE/api/deposit/depositions/$DEPID/actions/publish")
python3 - "$PUB" <<'PY'
import json, sys
r = json.loads(sys.argv[1])
doi = r.get("doi") or r.get("metadata", {}).get("doi", "UNKNOWN")
cid = r.get("conceptdoi", "")
print(f"\n  DOI (this version) : https://doi.org/{doi}")
if cid: print(f"  Concept DOI (all)  : https://doi.org/{cid}")
print(f"  Record             : {r.get('links',{}).get('record_html','')}")
open("/tmp/zenodo_doi.txt","w").write(doi)
PY

echo "== [6/6] Paste-ready lines for the manuscript =="
DOI=$(cat /tmp/zenodo_doi.txt)
cat <<TXT

  Code availability (paper):
    All code is available at https://github.com/$GHUSER/$GHREPO
    (release v2.0.0) and archived at https://doi.org/$DOI .

  README badge:
    [![DOI](https://zenodo.org/badge/DOI/$DOI.svg)](https://doi.org/$DOI)

  -> Send the GitHub URL and the DOI back to Claude to finish the paper.
TXT
echo "== DONE =="
