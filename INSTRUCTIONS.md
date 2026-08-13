# Code release: step-by-step (GitHub + Zenodo DOI)

## 0. What you need (5 minutes, one time)
1. **GitHub token** — https://github.com/settings/tokens ->
   "Generate new token (classic)" -> tick the **repo** scope ->
   Generate -> copy the `ghp_...` string.
2. **Zenodo token** — log in at https://zenodo.org ->
   https://zenodo.org/account/settings/applications/tokens/new/ ->
   name it, tick **deposit:write** and **deposit:actions** ->
   Create -> copy the token.
3. (Optional) Fill your real ORCID iDs in `CITATION.cff` and check the
   author list in `.zenodo.json` before publishing.

## 1. Upload and unpack on your server
    scp cvd-glassbox-release_v2.zip raad@YOUR_SERVER:~/
    ssh raad@YOUR_SERVER
    unzip -q cvd-glassbox-release_v2.zip -d ~/release
    cd ~/release/cvd-glassbox-screening

## 2. Run the publisher (it will PROMPT you to paste both tokens)
    bash release_to_github_zenodo.sh

   The script: creates the GitHub repo (or reuses it), pushes the code,
   tags **v2.0.0**, creates the GitHub release, deposits a snapshot on
   Zenodo, publishes it, and prints your **DOI** plus paste-ready lines
   for the paper's Code availability section and a README badge.

## 3. Send back to Claude
   - the GitHub URL (https://github.com/YOUR_USER/cvd-glassbox-screening)
   - the Zenodo DOI (10.5281/zenodo.XXXXXXX)
   These replace the two placeholders in the manuscript.

## Notes
- Tokens are never written to disk; they exist only in the shell session.
- Re-running is safe: repo creation is idempotent; a re-publish on
  Zenodo creates a new *version* under the same concept DOI.
- To dry-run Zenodo first: `export ZENODO_BASE=https://sandbox.zenodo.org`
  (needs a sandbox account/token), then run the script.
