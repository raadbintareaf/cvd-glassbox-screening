# JAMIA Open revision: how to run the new analyses

**Time needed:** about 10 minutes of your time, a few minutes of computer time.
**GPU:** not needed. **Retraining:** none. Everything is computed from the
predictions your server already saved during the original runs.

---

## Step 1 — Put this folder on your server

On your own computer, in the folder where you downloaded
`jamiao_revision_code.zip`:

```bash
scp jamiao_revision_code.zip YOUR_USER@YOUR_SERVER:~/
```

Then log in and unzip it **inside the project folder you ran the paper from**
(the one that contains `results/`, `data/` and `src/`):

```bash
ssh YOUR_USER@YOUR_SERVER
cd ~/PATH/TO/cvd-glassbox-screening      # the folder with results/ in it
unzip -o ~/jamiao_revision_code.zip
```

You should now have a `revision_jamiao/` folder next to `results/`.

## Step 2 — Run it (one command)

```bash
bash revision_jamiao/run.sh
```

It checks that everything it needs is there, fixes one bug (below), runs the
analyses, and ends with a green line:

```
DONE. Send back this one file:
  /home/.../cvd-glassbox-screening/jamiao_rev_bundle.tgz
```

If it stops with a red `STOPPED:` line, the message says exactly what is
missing. The usual cause is running it in the wrong folder.

## Step 3 — Send back one file

Download it to your computer:

```bash
scp YOUR_USER@YOUR_SERVER:~/PATH/TO/cvd-glassbox-screening/jamiao_rev_bundle.tgz .
```

and upload `jamiao_rev_bundle.tgz` in the chat. That is all I need to rewrite
the paper. To glance at the key numbers first:
`less results/jamiao_rev/SUMMARY.md`

---

## Step 4 (separate, important) — Repair the public GitHub repository

The public repository cannot currently run. `.gitignore` contains `data/`,
which git applies at every level, so it also hid `src/data/`. The files
`src/data/build_dataset.py` and `src/data/variable_map.py` were never
pushed, and ten scripts import them. The public copy of
`src/models/registry.py` is also older than your server's and lacks the
Random forest and CatBoost models reported in the paper. A reviewer who
tries the code would hit an import error on the first command.

On the server, in the same folder:

```bash
bash revision_jamiao/fix_public_repo.sh
```

It shows you the list of files first and only commits and pushes if you
type `yes`. If the folder is not a git checkout, it prints the five
commands to do it by hand instead. Afterwards, run
`bash release_to_github_zenodo.sh` once so Zenodo archives the repaired
version under the same concept DOI.

---

## What gets computed, and which review comment it answers

| File in the bundle | Answers |
|---|---|
| `leakage_significance.csv` | R1: no statistical test of the leakage drop. Paired DeLong test on the same test respondents for every model, Holm-corrected, plus bootstrap CIs for the AUROC and AUPRC drop. |
| `decision_impact.csv`, `decision_impact_summary.csv` | R1 + Associate Editor: "the effect looks small". Screening decisions per 100,000 people at the pre-specified operating points: people flagged, false positives, missed cases, flags per case found, PPV, for the leaky (T0) and honest (T1) feature sets. |
| `reclassification.csv` | R1: "how many people would receive an incorrect screening decision because of the leakage?" Person-by-person, how many decisions change and how many go from right to wrong. |
| `leakage_mechanism.csv` | What the leaky model actually "detects": the share of its detections that had already reported a coronary diagnosis, and whether its advantage survives among people with no such history. |
| `subgroup_decisions.csv` | R1: the decision impact by sex, age band and race/ethnicity. |
| `sex_audit_plain.csv` | R1: the sex result in plain counts (cases, detected, missed) with confidence intervals, instead of the gap notation the reviewer could not follow. |
| `coverage_explained.csv` | R2: why conformal coverage is higher in females while sensitivity is lower. Coverage is split by sex and by true class. |
| `missingness.csv`, `missingness_handling.txt` | R2: how often each feature is missing, in 2022 and 2023, and exactly how each model handles it. |
| `compute_cost.csv` | R2: training time as well as scoring time, for every model. |
| `temporal_delta.csv` | R2: 2022 vs 2023 with a delta column. |
| `calibration_points.csv`, `prediction_range.csv` | R2: calibration for every model including CatBoost + isotonic, and why calibrated predictions rarely exceed 0.7. |
| `delong_recheck.csv` | Correctness check of an older output; see below. |

Every figure for the revised paper will be drawn from these CSVs, so nothing
needs to be re-run to change a figure.

## A bug this fixes, and what it does and does not affect

`src/evaluation/metrics.py` → `delong_test` ranked the rows in their stored
order. The fast DeLong algorithm needs the positive cases first, and the
saved prediction files are sorted by row number, not by class, so this
function returned wrong AUCs and p-values. `run.sh` fixes it (keeping a
backup as `metrics.py.bak_before_delong_fix`).

**What it affected:** only `results/delong.csv`, written by
`analysis/aggregate_results.py`. **What it did not affect:** the paper's
non-inferiority and equivalence results came from
`analysis/equivalence_rev.py`, which has its own DeLong implementation that
was already correct. `delong_recheck.csv` recomputes every comparison in
`delong.csv` correctly and reports the largest error, so we can confirm
nothing in the manuscript relied on the faulty file.

## Answers to two review questions that come straight from the code

* **Class weights (R2):** computed inside each model's `fit()` from the
  training labels only (`src/models/registry.py`,
  `balanced_sample_weight(y)` and `scale_pos_weight`). Validation and test
  labels never enter them. No leakage.
* **Which data Table 2 uses (R2):** the held-out 2022 test partition
  (20% of the 2022 cohort, 88,413 respondents). The 2023 cohort is used only
  for the temporal validation in Table 4.
