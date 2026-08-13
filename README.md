# CVD-GlassBox-Screening

Leakage-aware, fairness-audited benchmark of **glass-box EBMs vs. tabular
foundation models (TabPFN v2, TabICL)** for population-level cardiovascular
screening on **CDC BRFSS 2022** (development) with **strict temporal external
validation on BRFSS 2023**. Companion code for the manuscript (in prep.).

**What this produces:** every number, table, and figure in the paper, from
raw CDC files to `paper/tables/*.tex` and `paper/figures/*.pdf`, via one
append-only `results/raw_results.csv`. No number is ever typed by hand.

---

## 1. Setup (user machine: 24 GB GPU, 128 GB RAM assumed)

```bash
git clone <this repo> && cd cvd-glassbox-screening
python -m venv .venv && source .venv/bin/activate
# GPU torch first (pick your CUDA):
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## 2. Data (raw CDC files, ~2.4 GB unzipped)

```bash
mkdir -p data/raw && cd data/raw
curl -LO https://www.cdc.gov/brfss/annual_data/2022/files/LLCP2022XPT.zip
curl -LO https://www.cdc.gov/brfss/annual_data/2023/files/LLCP2023XPT.zip
unzip LLCP2022XPT.zip && unzip LLCP2023XPT.zip
# CDC zips contain a trailing space in the inner filename:
mv "LLCP2022.XPT " LLCP2022.XPT 2>/dev/null; mv "LLCP2023.XPT " LLCP2023.XPT 2>/dev/null
cd ../..
```

Provenance note (verified against the files themselves, 2026-08-09):
`SLEPTIM1` (sleep) and `RMVTETH4` (teeth) exist only in the 2022 core;
`COVIDPOS` is renamed `COVIDPO1` in 2023 but remains core (7.8% missing);
`TETANUS1` and `LCSCTSC1` became state-optional modules in 2023 (97.3% /
94.7% missing), which collapses any 2023 complete-case cohort to n≈5.4k of
430.8k. The `T1_portable` tier therefore excludes SleepHours, RemovedTeeth,
and TetanusLast10; ChestScan is already excluded from T1 as leakage. See
`src/data/variable_map.py` and `data/analytic/*_cohort_log.json`.

## 3. Smoke test first (~2 min, CPU only)

```bash
python -m tests.test_smoke
# then a demo-scale real-data pass (8k subsample, CPU models, 1 seed):
python -m src.data.build_dataset --year 2022 --xpt data/raw/LLCP2022.XPT --out data/analytic
python -m experiments.run_benchmark --config configs/smoke.yaml --tier T1 --model ebm --seed 0
```

Demo-scale runs are for integration checking **only** — never paper numbers.

## 4. Full paper grid (resume-safe; re-run after any interruption)

```bash
bash experiments/run_all.sh configs/default.yaml
```

Approximate compute (24 GB GPU): CPU model grid ≈ 6–10 h across cores;
TabPFN v2 (8×10k context ensemble) ≈ 1–2 min fit + ~5–15 min predict per
run; TabICL full-context ≈ minutes–tens of minutes per run depending on
batch size; tuning ≈ 2–4 h; faithfulness sweep ≈ 2–6 h (LIME dominates).
Everything resumes: completed (tier × model × seed) cells are skipped.

## 5. Aggregate → tables → figures

```bash
python -m analysis.aggregate_results --config configs/default.yaml
python -m analysis.make_tables       --config configs/default.yaml
python -m analysis.make_figures      --config configs/default.yaml
```

## 6. Send back for manuscript assembly

Return (or commit) these — they are the only inputs the manuscript needs:

```
results/raw_results.csv
results/summary.csv  results/wilcoxon.csv  results/delong.csv
results/fairness/    results/conformal/    results/faithfulness/
results/temporal/    results/environment_freeze.txt
data/analytic/*_cohort_log.json
paper/tables/  paper/figures/
```

---

## Design registry (pre-registered before any full run)

* **Outcome**: self-reported ever-diagnosed myocardial infarction
  (`CVDINFR4`); **prevalent** disease, screening framing — no causal claims.
* **Tiers**: `T0` full · `T1` screening (drops angina/CHD + chest CT — direct
  post-diagnostic markers) · `T1_portable` (2022∩2023 variables) · `T2`
  self-report minimal. One-command ablation: `drop_stroke: true`.
* **Split**: stratified 60/20/20 × seeds {0..4}; all tuning and threshold
  selection on validation; test touched once per final model.
* **Operating points**: screening (sens ≥ 0.85 on val) and confirmatory
  (spec ≥ 0.90 on val); all confusion metrics reported at these explicit
  thresholds.
* **Statistics**: across-seed Wilcoxon (Holm) + seed-0 paired DeLong (Holm);
  95% stratified bootstrap CIs; calibration slope/intercept/ECE; decision
  curves.
* **Imbalance**: class weighting primary; SMOTE-NC as explicit ablation
  (train-split only).
* **Fairness**: subgroup audit at explicit thresholds (Wilson CIs); arms =
  group thresholds · reweighing · EBM shape repair (zero / attenuate /
  zero+intercept-equalize) with machine-readable edit logs.
* **Uncertainty**: split + Mondrian conformal (Sex; Sex×AgeBand), α = 0.1;
  coverage, set size, deferral.
* **Faithfulness**: TreeSHAP / KernelSHAP / LIME agreement vs. exact EBM
  importances across explanation sample sizes, with resample CIs.
* **Temporal validation**: frozen 2022 models + thresholds applied to the
  entire 2023 cohort (`T1_portable`); EBM-2023 refit only for the
  shape-replication figure.

## Repository map

```
configs/            YAML for everything; CLI takes only coordinates + seed
src/data/           verified variable map + XPT→parquet builder + cohort log
src/models/         unified adapters (EBM, LR±splines, XGB, LGBM, MLP,
                    TabPFN v2 ensemble, TabICL)
src/evaluation/     metrics (single source), fairness, conformal, faithfulness
src/repair/         EBM shape-editing arms with edit logs
experiments/        run_benchmark / tuning / fairness / conformal /
                    faithfulness / temporal / run_all.sh (resume-safe)
analysis/           aggregate → booktabs tables → themed vector figures
tests/              synthetic end-to-end smoke test
```

## License

MIT (code). BRFSS data are public-domain CDC releases; respondents are
de-identified. This repository never redistributes raw CDC files.


## Release v2.0.0 — npj Digital Medicine revision

This version adds every analysis introduced in the major revision:

- `analysis/equivalence_rev.py` — paired-DeLong non-inferiority + TOST
  (pre-specified delta = 0.005, Holm), prediction correlations.
- Tier **T1-ns** (T1 minus prior stroke) registered through the
  pre-existing `drop_stroke` switch; benchmark runs via
  `--tier T1_nostroke`.
- `experiments/run_unweighted_rev.py` — class-weight-free EBM/CatBoost
  baseline (calibration attribution).
- `experiments/run_fairness_v2.py` + `analysis/fairness_bootstrap_rev.py`
  — common selection objective, equalise-TPR control arm, effective
  per-group thresholds, per-arm probabilities, 1,000-draw stratified
  paired bootstrap.
- `analysis/conformal_v2_rev.py` — four-decimal coverage, empty-set
  accounting, split-validation optimism, 18–39 carve-out hybrid.
- `experiments/run_weighted_train_rev.py` and the design-weight
  quintile test — survey-weighting analyses.
- `analysis/lime_on_ebm_rev.py` — same-model LIME faithfulness.
- `analysis/temporal_oe_rev.py` — 2023 subgroup calibration-in-the-large.
- `scripts/revision/` — the exact orchestration scripts used for the
  revision runs (provenance).

Publish this release with `bash release_to_github_zenodo.sh`
(see `INSTRUCTIONS.md`). DOI badge: _added after minting_.
