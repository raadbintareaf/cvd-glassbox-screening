"""Analyses requested by the JAMIA Open reviewers (JAMIO-2026-0501).

Everything here is computed from artefacts the pipeline already saved:

    results/predictions/brfss2022/<tier>/<model>/seed<k>.parquet
    results/raw_results.csv
    data/analytic/brfss_2022_analytic.parquet   (and 2023, if present)

Nothing is retrained. Runs on CPU in a few minutes. Every output table is a
plain CSV in results/jamiao_rev/, and every figure ships with the CSV of the
numbers behind it, so the figures can be redrawn without re-running this.

Analyses, keyed to the review:

  A1  leakage_significance.csv   R1: is the T0 -> T1 drop statistically
                                  significant? Paired DeLong on the same test
                                  respondents, Holm-corrected; paired
                                  bootstrap CI for the AUROC and AUPRC drop.
  A2  decision_impact.csv        R1 + AE: what the drop means for screening
                                  decisions. Counts per 100,000 people at the
                                  pre-specified operating points.
      reclassification.csv       R1: how many people get a different
                                  (and wrong) decision because of leakage.
  A3  leakage_mechanism.csv      Who the leaky model "detects": the share of
                                  its detections that already reported a
                                  coronary diagnosis, and whether its edge
                                  survives among people without one.
  A4  subgroup_decisions.csv     R1: decision impact by sex, age band and
                                  race/ethnicity.
  A5  sex_audit_plain.csv        R1: the sex fairness result as plain counts
                                  (cases, detected, missed) with Wilson CIs.
  A6  coverage_explained.csv     R2: why conformal coverage is higher in
                                  females while sensitivity is lower.
  A7  missingness.csv            R2: per-feature missingness, 2022 and 2023.
  A8  compute_cost.csv           R2: training AND scoring time per model.
  A9  temporal_delta.csv         R2: 2022 internal vs 2023 external, with
                                  a delta column.
  A10 calibration_points.csv     R2: reliability curves for every model,
      prediction_range.csv       raw and isotonic (CatBoost included), and
                                  why calibrated predictions rarely pass 0.7.
  A11 delong_recheck.csv         Correctness check: the shared
                                  metrics.delong_test ranked rows in stored
                                  order, which is wrong unless positives come
                                  first. This re-derives every EBM-vs-model
                                  DeLong test correctly and compares it with
                                  the old results/delong.csv.

Usage (from the repository root):
    python -m revision_jamiao.jamiao_analyses --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from src.evaluation.fairness import wilson_ci
from src.evaluation.metrics import confusion_at, holm, threshold_for
from src.utils.common import load_config

PER = 100_000                     # report counts per 100,000 people
SCREEN = ("sens_at", 0.85)        # pre-specified screening rule (validation)
CONFIRM = ("spec_at", 0.90)       # pre-specified confirmatory rule
N_BOOT = 500                      # paired bootstrap draws (seed 0)
DATASET = "brfss2022"

MODEL_NAMES = {
    "logreg": "Logistic regression", "logreg_spline": "Logistic regression (splines)",
    "random_forest": "Random forest", "xgboost": "XGBoost", "lightgbm": "LightGBM",
    "catboost": "CatBoost", "ebm": "EBM (glass-box)", "mlp": "MLP",
    "tabpfn_v2": "TabPFN v2", "tabicl": "TabICL",
}
MODEL_ORDER = list(MODEL_NAMES)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def is_yes(s: pd.Series) -> np.ndarray:
    v = s.astype(str).str.strip().str.lower()
    return v.isin(["yes", "1", "1.0", "true", "y"]).values


def age_band(df: pd.DataFrame) -> pd.Series:
    """Same banding as analysis/conformal_v2_rev.py: 18-39, 40-59, 60+."""
    a = df["AgeCategory"]
    if not pd.api.types.is_numeric_dtype(a):
        a = a.cat.codes.replace(-1, np.nan) + 1 if isinstance(
            a.dtype, pd.CategoricalDtype) else pd.to_numeric(a, errors="coerce")
    return pd.cut(a, [0, 4.5, 8.5, 99],
                  labels=["18-39", "40-59", "60+"]).astype(str)


def race_column(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if "race" in c.lower():
            return c
    return None


def models_with(pred_root: Path, *tiers: str) -> list[str]:
    sets = []
    for t in tiers:
        d = pred_root / t
        sets.append({p.name for p in d.iterdir() if p.is_dir()} if d.exists() else set())
    common = set.intersection(*sets) if sets else set()
    # keep only the main benchmark arms (skip ablations such as "+smotenc")
    common = {m for m in common if "+" not in m and not m.endswith(("_unw", "_wt"))}
    return sorted(common, key=lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else 99)


def seeds_for(pred_root: Path, tier: str, model: str) -> list[int]:
    d = pred_root / tier / model
    return sorted(int(p.stem.replace("seed", "")) for p in d.glob("seed*.parquet"))


def load_pred(pred_root: Path, tier: str, model: str, seed: int):
    d = pd.read_parquet(pred_root / tier / model / f"seed{seed}.parquet")
    va = d[d.split == "val"].sort_values("row_id").reset_index(drop=True)
    te = d[d.split == "test"].sort_values("row_id").reset_index(drop=True)
    return va, te


def delong_test(y, p1, p2):
    """Paired DeLong test, self-contained and order-safe (positives are moved
    first, as the fast algorithm requires). Returns (auc1, auc2, p)."""
    from scipy import stats

    def midrank(x):
        o = np.argsort(x, kind="mergesort")
        sx = x[o]
        r = np.empty(len(x))
        i = 0
        while i < len(x):
            j = i
            while j < len(x) and sx[j] == sx[i]:
                j += 1
            r[o[i:j]] = 0.5 * (i + j - 1) + 1
            i = j
        return r

    y = np.asarray(y)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    m, n = len(pos), len(neg)
    preds = np.vstack([np.asarray(p1), np.asarray(p2)])
    tx, ty, tz = np.empty((2, m)), np.empty((2, n)), np.empty((2, m + n))
    for r in range(2):
        tx[r] = midrank(preds[r, pos])
        ty[r] = midrank(preds[r, neg])
        tz[r] = midrank(np.concatenate([preds[r, pos], preds[r, neg]]))
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    S = np.cov(v01) / m + np.cov(v10) / n
    var = float(S[0, 0] + S[1, 1] - 2 * S[0, 1])
    if var <= 0:
        return float(aucs[0]), float(aucs[1]), 1.0
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return float(aucs[0]), float(aucs[1]), float(2 * stats.norm.sf(abs(z)))


def per100k(count: float, n: int) -> float:
    return float(count) / n * PER


def paired_boot(y, p0, p1, n_boot=N_BOOT, seed=0):
    """Stratified paired bootstrap of AUROC and AUPRC differences (p0 - p1)."""
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    d_roc, d_prc = [], []
    for _ in range(n_boot):
        ii = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        yy = y[ii]
        d_roc.append(roc_auc_score(yy, p0[ii]) - roc_auc_score(yy, p1[ii]))
        d_prc.append(average_precision_score(yy, p0[ii]) -
                     average_precision_score(yy, p1[ii]))
    q = lambda a: tuple(np.percentile(a, [2.5, 97.5]))
    return q(d_roc), q(d_prc)


# --------------------------------------------------------------------------
# context shared by the analyses
# --------------------------------------------------------------------------

class Ctx:
    def __init__(self, cfg):
        from experiments.run_benchmark import load_analytic
        from src.data.variable_map import OUTCOME_LABEL, tier_labels

        self.cfg = cfg
        self.res_dir = Path(cfg["results_dir"])
        self.out = self.res_dir / "jamiao_rev"
        self.out.mkdir(parents=True, exist_ok=True)
        self.pred = self.res_dir / "predictions" / DATASET
        self.outcome = OUTCOME_LABEL
        self.tier_labels = tier_labels

        log("loading the 2022 analytic cohort")
        self.df = load_analytic(cfg, DATASET)
        self.df["AgeBand"] = age_band(self.df)
        self.race = race_column(self.df)
        self.t0 = list(tier_labels("T0"))
        self.t1 = list(tier_labels("T1"))
        self.leak = [f for f in self.t0 if f not in self.t1]
        log(f"features removed between T0 and T1: {self.leak}")
        self.models = models_with(self.pred, "T0", "T1")
        log(f"models with saved T0 and T1 predictions: {self.models}")
        if not self.models:
            raise SystemExit("No models have both T0 and T1 predictions under "
                             f"{self.pred}. Is results_dir correct?")

    def pair(self, model: str, seed: int):
        """Validation and test predictions for T0 and T1, row-aligned."""
        v0, t0 = load_pred(self.pred, "T0", model, seed)
        v1, t1 = load_pred(self.pred, "T1", model, seed)
        for a, b, name in ((v0, v1, "val"), (t0, t1, "test")):
            if not (np.array_equal(a.row_id.values, b.row_id.values) and
                    np.array_equal(a.y.values, b.y.values)):
                raise RuntimeError(f"{model} seed{seed}: T0 and T1 {name} rows "
                                   "differ; the paired analysis is not valid.")
        return v0, t0, v1, t1


# --------------------------------------------------------------------------
# A1: is the leakage drop statistically significant?
# --------------------------------------------------------------------------

def a1_significance(c: Ctx) -> pd.DataFrame:
    rows = []
    for m in c.models:
        for s in seeds_for(c.pred, "T1", m):
            if s not in seeds_for(c.pred, "T0", m):
                continue
            _, t0, _, t1 = c.pair(m, s)
            y = t0.y.values.astype(int)
            a0, a1, p = delong_test(y, t0.p.values, t1.p.values)
            r = dict(model=m, seed=s, n_test=len(y), n_cases=int(y.sum()),
                     auroc_T0=a0, auroc_T1=a1, auroc_drop=a0 - a1,
                     auprc_T0=average_precision_score(y, t0.p.values),
                     auprc_T1=average_precision_score(y, t1.p.values),
                     delong_p=p)
            r["auprc_drop"] = r["auprc_T0"] - r["auprc_T1"]
            r["auprc_drop_relative"] = r["auprc_drop"] / r["auprc_T0"]
            if s == 0:
                (lo, hi), (plo, phi) = paired_boot(y, t0.p.values, t1.p.values)
                r.update(auroc_drop_lo=lo, auroc_drop_hi=hi,
                         auprc_drop_lo=plo, auprc_drop_hi=phi)
            rows.append(r)
        log(f"A1 {m}: done")
    df = pd.DataFrame(rows)
    df["delong_p_holm"] = np.nan
    for s, g in df.groupby("seed"):
        df.loc[g.index, "delong_p_holm"] = holm(g.delong_p.tolist())
    df.to_csv(c.out / "leakage_significance.csv", index=False)
    return df


# --------------------------------------------------------------------------
# A2: what the drop means for screening decisions
# --------------------------------------------------------------------------

def a2_decisions(c: Ctx):
    imp, recl = [], []
    for m in c.models:
        for s in seeds_for(c.pred, "T1", m):
            if s not in seeds_for(c.pred, "T0", m):
                continue
            v0, t0, v1, t1 = c.pair(m, s)
            y = t0.y.values.astype(int)
            n = len(y)
            for rule_name, (rule, target) in (("screening", SCREEN),
                                              ("confirmatory", CONFIRM)):
                thr0 = threshold_for(v0.y.values, v0.p.values, rule, target)
                thr1 = threshold_for(v1.y.values, v1.p.values, rule, target)
                for tier, tt, thr in (("T0", t0, thr0), ("T1", t1, thr1)):
                    cm = confusion_at(y, tt.p.values, thr)
                    flagged = cm["tp"] + cm["fp"]
                    imp.append(dict(
                        model=m, seed=s, rule=rule_name, tier=tier,
                        threshold=thr, n_test=n, prevalence=y.mean(),
                        sensitivity=cm["sensitivity"],
                        specificity=cm["specificity"], ppv=cm["ppv"],
                        tp=cm["tp"], fp=cm["fp"], fn=cm["fn"], tn=cm["tn"],
                        flagged_per_100k=per100k(flagged, n),
                        false_pos_per_100k=per100k(cm["fp"], n),
                        missed_cases_per_100k=per100k(cm["fn"], n),
                        found_cases_per_100k=per100k(cm["tp"], n),
                        flagged_per_case_found=flagged / max(cm["tp"], 1)))
                # person-level reclassification between the two models
                d0 = t0.p.values >= thr0
                d1 = t1.p.values >= thr1
                right0 = d0 == (y == 1)
                right1 = d1 == (y == 1)
                recl.append(dict(
                    model=m, seed=s, rule=rule_name, n_test=n,
                    cases_found_by_both=int(((y == 1) & d0 & d1).sum()),
                    cases_found_only_T0=int(((y == 1) & d0 & ~d1).sum()),
                    cases_found_only_T1=int(((y == 1) & ~d0 & d1).sum()),
                    noncases_flagged_both=int(((y == 0) & d0 & d1).sum()),
                    noncases_flagged_only_T0=int(((y == 0) & d0 & ~d1).sum()),
                    noncases_flagged_only_T1=int(((y == 0) & ~d0 & d1).sum()),
                    decision_differs=int((d0 != d1).sum()),
                    right_T0_wrong_T1=int((right0 & ~right1).sum()),
                    wrong_T0_right_T1=int((~right0 & right1).sum())))
        log(f"A2 {m}: done")
    imp, recl = pd.DataFrame(imp), pd.DataFrame(recl)
    for col in ["decision_differs", "right_T0_wrong_T1", "wrong_T0_right_T1"]:
        recl[col + "_per_100k"] = recl[col] / recl.n_test * PER
    recl["net_extra_wrong_decisions_per_100k"] = (
        recl.right_T0_wrong_T1_per_100k - recl.wrong_T0_right_T1_per_100k)
    imp.to_csv(c.out / "decision_impact.csv", index=False)
    recl.to_csv(c.out / "reclassification.csv", index=False)

    # one-row-per-model summary (mean over seeds), the table the paper needs
    keys = ["flagged_per_100k", "false_pos_per_100k", "missed_cases_per_100k",
            "found_cases_per_100k", "sensitivity", "specificity", "ppv",
            "flagged_per_case_found"]
    wide = imp.pivot_table(index=["model", "rule"], columns="tier",
                           values=keys, aggfunc="mean")
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    for k in keys:
        wide[f"{k}_change"] = wide[f"{k}_T1"] - wide[f"{k}_T0"]
    wide = wide.reset_index()
    wide.to_csv(c.out / "decision_impact_summary.csv", index=False)
    return imp, recl, wide


# --------------------------------------------------------------------------
# A3: who the leaky model is "detecting"
# --------------------------------------------------------------------------

def a3_mechanism(c: Ctx) -> pd.DataFrame:
    leak_flags = {}
    for f in c.leak:
        try:
            leak_flags[f] = is_yes(c.df[f])
        except Exception:                                   # noqa: BLE001
            log(f"A3: could not read {f} as yes/no; skipped")
    if not leak_flags:
        log("A3: no yes/no leakage feature found; skipped")
        return pd.DataFrame()
    any_leak = np.zeros(len(c.df), bool)
    for v in leak_flags.values():
        any_leak |= v

    rows = []
    for m in c.models:
        for s in seeds_for(c.pred, "T1", m):
            if s not in seeds_for(c.pred, "T0", m):
                continue
            v0, t0, v1, t1 = c.pair(m, s)
            rid = t0.row_id.values
            y = t0.y.values.astype(int)
            thr0 = threshold_for(v0.y.values, v0.p.values, *SCREEN)
            thr1 = threshold_for(v1.y.values, v1.p.values, *SCREEN)
            d0, d1 = t0.p.values >= thr0, t1.p.values >= thr1
            known = any_leak[rid]
            r = dict(model=m, seed=s,
                     cases=int(y.sum()),
                     cases_with_known_marker=int((known & (y == 1)).sum()),
                     noncases_with_known_marker=int((known & (y == 0)).sum()),
                     tp_T0=int((d0 & (y == 1)).sum()),
                     tp_T0_with_known_marker=int((d0 & (y == 1) & known).sum()),
                     tp_T1=int((d1 & (y == 1)).sum()),
                     tp_T1_with_known_marker=int((d1 & (y == 1) & known).sum()))
            r["share_of_cases_with_marker"] = r["cases_with_known_marker"] / max(r["cases"], 1)
            r["share_of_T0_detections_with_marker"] = (
                r["tp_T0_with_known_marker"] / max(r["tp_T0"], 1))
            for f, flag in leak_flags.items():
                k = flag[rid]
                r[f"cases_with_{f}"] = int((k & (y == 1)).sum())
                r[f"T0_detections_with_{f}"] = int((d0 & (y == 1) & k).sum())
            # does the T0 advantage survive among people with NO marker?
            sub = ~known
            if sub.sum() and y[sub].min() == 0 and y[sub].max() == 1:
                r["auroc_T0_no_marker"] = roc_auc_score(y[sub], t0.p.values[sub])
                r["auroc_T1_no_marker"] = roc_auc_score(y[sub], t1.p.values[sub])
                r["n_no_marker"] = int(sub.sum())
                r["cases_no_marker"] = int(y[sub].sum())
                cm0 = confusion_at(y[sub], t0.p.values[sub], thr0)
                cm1 = confusion_at(y[sub], t1.p.values[sub], thr1)
                r["sens_T0_no_marker"] = cm0["sensitivity"]
                r["sens_T1_no_marker"] = cm1["sensitivity"]
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(c.out / "leakage_mechanism.csv", index=False)
    log("A3: done")
    return df


# --------------------------------------------------------------------------
# A4: decision impact by subgroup
# --------------------------------------------------------------------------

def a4_subgroups(c: Ctx) -> pd.DataFrame:
    gcols = ["Sex", "AgeBand"] + ([c.race] if c.race else [])
    rows = []
    for m in c.models:
        for s in seeds_for(c.pred, "T1", m):
            if s not in seeds_for(c.pred, "T0", m):
                continue
            v0, t0, v1, t1 = c.pair(m, s)
            rid = t0.row_id.values
            y = t0.y.values.astype(int)
            thr0 = threshold_for(v0.y.values, v0.p.values, *SCREEN)
            thr1 = threshold_for(v1.y.values, v1.p.values, *SCREEN)
            for g in gcols:
                lv = c.df[g].astype(str).values[rid]
                for level in pd.unique(lv):
                    mk = lv == level
                    if mk.sum() < 200 or y[mk].sum() < 10:
                        continue
                    for tier, tt, thr in (("T0", t0, thr0), ("T1", t1, thr1)):
                        cm = confusion_at(y[mk], tt.p.values[mk], thr)
                        lo, hi = wilson_ci(cm["tp"], cm["tp"] + cm["fn"])
                        rows.append(dict(
                            model=m, seed=s, group_var=g, group=level, tier=tier,
                            n=int(mk.sum()), cases=int(y[mk].sum()),
                            prevalence=float(y[mk].mean()),
                            sensitivity=cm["sensitivity"], sens_lo=lo, sens_hi=hi,
                            specificity=cm["specificity"], ppv=cm["ppv"],
                            missed_cases=cm["fn"],
                            false_pos_per_100k=per100k(cm["fp"], int(mk.sum())),
                            missed_cases_per_100k=per100k(cm["fn"], int(mk.sum()))))
        log(f"A4 {m}: done")
    df = pd.DataFrame(rows)
    df.to_csv(c.out / "subgroup_decisions.csv", index=False)
    return df


# --------------------------------------------------------------------------
# A5: the sex audit as plain counts
# --------------------------------------------------------------------------

def a5_sex_plain(c: Ctx) -> pd.DataFrame:
    rows = []
    for m in [x for x in ("ebm", "xgboost") if x in c.models] or c.models[:1]:
        for s in seeds_for(c.pred, "T1", m):
            v, t = load_pred(c.pred, "T1", m, s)
            thr = threshold_for(v.y.values, v.p.values, *SCREEN)
            sex = c.df["Sex"].astype(str).values[t.row_id.values]
            y = t.y.values.astype(int)
            for level in sorted(pd.unique(sex)):
                mk = sex == level
                cm = confusion_at(y[mk], t.p.values[mk], thr)
                lo, hi = wilson_ci(cm["tp"], cm["tp"] + cm["fn"])
                slo, shi = wilson_ci(cm["tn"], cm["tn"] + cm["fp"])
                rows.append(dict(model=m, seed=s, sex=level, n=int(mk.sum()),
                                 cases=int(y[mk].sum()),
                                 prevalence=float(y[mk].mean()),
                                 detected=cm["tp"], missed=cm["fn"],
                                 sensitivity=cm["sensitivity"], sens_lo=lo, sens_hi=hi,
                                 specificity=cm["specificity"], spec_lo=slo, spec_hi=shi,
                                 false_positives=cm["fp"], ppv=cm["ppv"],
                                 missed_one_in=(1 / cm["fnr"]) if cm["fnr"] > 0 else np.nan))
    df = pd.DataFrame(rows)
    df.to_csv(c.out / "sex_audit_plain.csv", index=False)
    log("A5: done")
    return df


# --------------------------------------------------------------------------
# A6: why coverage is high in females while sensitivity is low
# --------------------------------------------------------------------------

def a6_coverage(c: Ctx, alpha: float = 0.1) -> pd.DataFrame:
    """Reproduces the marginal split-conformal procedure of
    analysis/conformal_v2_rev.py exactly (one quantile on true-label scores,
    validation as calibration) and splits coverage by sex AND true class."""
    rows = []
    m = "ebm" if "ebm" in c.models else c.models[0]
    for s in seeds_for(c.pred, "T1", m):
        v, t = load_pred(c.pred, "T1", m, s)
        yv, pv = v.y.values.astype(int), v.p.values
        yt, pt = t.y.values.astype(int), t.p.values
        sv = np.where(yv == 1, 1 - pv, pv)
        n = len(sv)
        q = np.sort(sv)[min(int(np.ceil((n + 1) * (1 - alpha))), n) - 1]
        in0, in1 = pt <= q, (1 - pt) <= q
        covered = np.where(yt == 1, in1, in0)
        size = in0.astype(int) + in1.astype(int)
        sex = c.df["Sex"].astype(str).values[t.row_id.values]
        for level in sorted(pd.unique(sex)):
            mk = sex == level
            for cls, cname in ((1, "cases"), (0, "non-cases"), (None, "all")):
                mm = mk if cls is None else mk & (yt == cls)
                if not mm.any():
                    continue
                rows.append(dict(model=m, seed=s, sex=level, people=cname,
                                 n=int(mm.sum()),
                                 share_of_sex_group=float(mm.sum() / mk.sum()),
                                 coverage=float(covered[mm].mean()),
                                 set_0=float(((in0) & (~in1))[mm].mean()),
                                 set_1=float(((~in0) & (in1))[mm].mean()),
                                 set_both=float((size == 2)[mm].mean()),
                                 set_empty=float((size == 0)[mm].mean()),
                                 qhat=float(q)))
    df = pd.DataFrame(rows)
    df.to_csv(c.out / "coverage_explained.csv", index=False)
    log("A6: done")
    return df


# --------------------------------------------------------------------------
# A7: missingness per feature, and how each model family handles it
# --------------------------------------------------------------------------

HANDLING = (
    "EBM: missing values form their own bin. XGBoost, LightGBM: native "
    "missing-value branches. CatBoost: native handling. Random forest: see "
    "registry (server copy). Logistic regression and MLP: numeric features "
    "imputed with the TRAINING-split median plus a missing-indicator column; "
    "categorical features get an explicit 'Missing' level. TabPFN v2: NaN "
    "passed through. TabICL: native handling. No imputation model was fit on "
    "validation or test data.")


def a7_missingness(c: Ctx) -> pd.DataFrame:
    from experiments.run_benchmark import load_analytic
    rows = []
    frames = {"2022": c.df}
    try:
        frames["2023"] = load_analytic(c.cfg, "brfss2023")
    except SystemExit:
        log("A7: 2023 analytic file not found; reporting 2022 only")
    tiers = {}
    for t in ("T0", "T1", "T1_portable", "T2"):
        try:
            tiers[t] = set(c.tier_labels(t))
        except Exception:                                   # noqa: BLE001
            pass
    for f in c.t0:
        r = dict(feature=f, dtype=str(c.df[f].dtype),
                 tiers=",".join(t for t, s in tiers.items() if f in s))
        for yr, d in frames.items():
            r[f"missing_pct_{yr}"] = (100 * d[f].isna().mean()
                                      if f in d.columns else np.nan)
        rows.append(r)
    df = pd.DataFrame(rows).sort_values("missing_pct_2022", ascending=False)
    df.to_csv(c.out / "missingness.csv", index=False)
    (c.out / "missingness_handling.txt").write_text(HANDLING + "\n")
    cc = c.df[c.t1].notna().all(axis=1).mean()
    log(f"A7: done (complete-case share on T1 features, 2022: {cc:.3f})")
    return df


# --------------------------------------------------------------------------
# A8 + A9: compute cost and temporal deltas, from raw_results.csv
# --------------------------------------------------------------------------

def _raw(c: Ctx) -> pd.DataFrame:
    rr = pd.read_csv(c.res_dir / "raw_results.csv", low_memory=False)
    rr["value"] = pd.to_numeric(rr["value"], errors="coerce")
    return rr


def a8_cost(c: Ctx) -> pd.DataFrame:
    rr = _raw(c)
    b = rr[(rr.stage == "benchmark") & (rr.dataset == DATASET) & (rr.tier == "T1")]
    keep = b[b.metric.isin(["fit_seconds", "predict_seconds", "n_train"])]
    out = (keep.groupby(["model", "metric"]).value.agg(["mean", "std", "count"])
           .unstack("metric"))
    out.columns = [f"{m}_{a}" for a, m in out.columns]
    out = out.reset_index()
    out["note"] = ("fit = training on 265k rows (for foundation models: context "
                   "ingestion, no gradient training); predict = scoring the "
                   "validation + test partitions")
    out.to_csv(c.out / "compute_cost.csv", index=False)
    log("A8: done")
    return out


def a9_temporal(c: Ctx) -> pd.DataFrame:
    rr = _raw(c)
    metrics = ["auroc", "auprc", "brier", "cal_slope", "ece",
               "screening_sensitivity", "screening_specificity", "screening_ppv"]
    a = rr[(rr.stage == "benchmark") & (rr.dataset == DATASET) &
           (rr.tier == "T1_portable") & (rr.split == "test") &
           (rr.subgroup.fillna("ALL") == "ALL") & rr.metric.isin(metrics)]
    b = rr[(rr.stage == "temporal") & (rr.dataset == "brfss2023") &
           (rr.split == "external") & rr.metric.isin(metrics)]
    if a.empty or b.empty:
        log("A9: temporal rows not found in raw_results.csv; skipped")
        return pd.DataFrame()
    ia = a.groupby(["model", "metric"]).value.mean().unstack()
    ib = b.groupby(["model", "metric"]).value.mean().unstack()
    rows = []
    for mdl in ib.index.intersection(ia.index):
        for k in metrics:
            if k in ia.columns and k in ib.columns:
                rows.append(dict(model=mdl, metric=k, internal_2022=ia.loc[mdl, k],
                                 external_2023=ib.loc[mdl, k],
                                 delta=ib.loc[mdl, k] - ia.loc[mdl, k]))
    df = pd.DataFrame(rows)
    df.to_csv(c.out / "temporal_delta.csv", index=False)
    log("A9: done")
    return df


# --------------------------------------------------------------------------
# A10: calibration for every model, raw and isotonic
# --------------------------------------------------------------------------

def a10_calibration(c: Ctx, n_bins: int = 15):
    pts, rng_rows = [], []
    models = [m for m in MODEL_ORDER if (c.pred / "T1" / m / "seed0.parquet").exists()]
    for m in models:
        v, t = load_pred(c.pred, "T1", m, 0)
        iso = IsotonicRegression(out_of_bounds="clip").fit(v.p.values, v.y.values)
        for kind, p in (("raw", t.p.values), ("isotonic", iso.predict(t.p.values))):
            y = t.y.values
            # quantile bins: equal numbers of people per point
            edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
            idx = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
            for b in range(len(edges) - 1):
                mk = idx == b
                if mk.any():
                    pts.append(dict(model=m, kind=kind, bin=b, n=int(mk.sum()),
                                    mean_predicted=float(p[mk].mean()),
                                    observed=float(y[mk].mean())))
            rng_rows.append(dict(
                model=m, kind=kind, max_predicted=float(p.max()),
                p99_predicted=float(np.quantile(p, 0.99)),
                share_above_0_5=float((p > 0.5).mean()),
                share_above_0_7=float((p > 0.7).mean()),
                observed_rate_in_top_1pct=float(y[p >= np.quantile(p, 0.99)].mean()),
                prevalence=float(y.mean())))
    pd.DataFrame(pts).to_csv(c.out / "calibration_points.csv", index=False)
    pd.DataFrame(rng_rows).to_csv(c.out / "prediction_range.csv", index=False)
    log("A10: done")


# --------------------------------------------------------------------------
# A11: re-check results/delong.csv (written with the old, order-sensitive
# metrics.delong_test) against a correct DeLong on the same predictions
# --------------------------------------------------------------------------

def a11_delong_recheck(c: Ctx) -> pd.DataFrame:
    old_path = c.res_dir / "delong.csv"
    old = pd.read_csv(old_path) if old_path.exists() else pd.DataFrame()
    rows = []
    for tdir in sorted(p for p in c.pred.iterdir() if p.is_dir()):
        tier = tdir.name
        e = tdir / "ebm" / "seed0.parquet"
        if not e.exists():
            continue
        _, te = load_pred(c.pred, tier, "ebm", 0)
        for mdir in sorted(p for p in tdir.iterdir() if p.is_dir()):
            mdl = mdir.name
            if mdl == "ebm" or "+" in mdl or not (mdir / "seed0.parquet").exists():
                continue
            _, tm = load_pred(c.pred, tier, mdl, 0)
            if not np.array_equal(te.row_id.values, tm.row_id.values):
                continue
            y = te.y.values.astype(int)
            a_e, a_m, p = delong_test(y, te.p.values, tm.p.values)
            r = dict(tier=tier, model=mdl, auc_ebm=a_e, auc_model=a_m,
                     auc_ebm_sklearn=roc_auc_score(y, te.p.values),
                     delong_p_correct=p)
            if not old.empty:
                hit = old[(old.tier == tier) & (old.model == mdl)]
                if len(hit):
                    r["auc_ebm_old_file"] = float(hit.iloc[0].get("auc_ebm", np.nan))
                    r["delong_p_old_file"] = float(hit.iloc[0].get("delong_p", np.nan))
            rows.append(r)
    df = pd.DataFrame(rows)
    if not df.empty:
        df["delong_p_correct_holm"] = np.nan
        for t, g in df.groupby("tier"):
            df.loc[g.index, "delong_p_correct_holm"] = holm(g.delong_p_correct.tolist())
    df.to_csv(c.out / "delong_recheck.csv", index=False)
    if "auc_ebm_old_file" in df:
        gap = (df.auc_ebm_old_file - df.auc_ebm_sklearn).abs().max()
        log(f"A11: largest AUROC error in the old delong.csv = {gap:.4f}")
    log("A11: done")
    return df


# --------------------------------------------------------------------------
# summary for the manuscript
# --------------------------------------------------------------------------

def write_summary(c: Ctx, results: dict) -> None:
    L = ["# JAMIA Open revision: computed results", "",
         f"Generated {time.strftime('%Y-%m-%d %H:%M')} from saved predictions. "
         "All numbers below are copied from the CSV files in this folder.", "",
         f"Features removed between T0 and T1: {', '.join(c.leak)}", ""]

    sig = results.get("A1")
    if sig is not None and not sig.empty:
        s0 = sig[sig.seed == 0]
        L += ["## A1. Is the leakage drop significant? (seed 0, paired DeLong, Holm)", "",
              "| model | AUROC T0 | AUROC T1 | drop [95% CI] | AUPRC T0 | AUPRC T1 | AUPRC drop (relative) | Holm p |",
              "|---|---|---|---|---|---|---|---|"]
        for _, r in s0.iterrows():
            L.append(f"| {r.model} | {r.auroc_T0:.4f} | {r.auroc_T1:.4f} | "
                     f"{r.auroc_drop:.4f} [{r.get('auroc_drop_lo', np.nan):.4f}, "
                     f"{r.get('auroc_drop_hi', np.nan):.4f}] | {r.auprc_T0:.3f} | "
                     f"{r.auprc_T1:.3f} | {r.auprc_drop:.3f} ({100*r.auprc_drop_relative:.0f}%) | "
                     f"{r.delong_p_holm:.2e} |")
        L.append("")

    wide = results.get("A2")
    if wide is not None and not wide.empty:
        w = wide[wide.rule == "screening"]
        L += ["## A2. Screening decisions per 100,000 people (sensitivity target 0.85, mean over seeds)", "",
              "| model | flagged T0 | flagged T1 | false positives T0 | false positives T1 | change | flagged per case found T0 | T1 | PPV T0 | PPV T1 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in w.iterrows():
            L.append(f"| {r.model} | {r.flagged_per_100k_T0:,.0f} | {r.flagged_per_100k_T1:,.0f} | "
                     f"{r.false_pos_per_100k_T0:,.0f} | {r.false_pos_per_100k_T1:,.0f} | "
                     f"{r.false_pos_per_100k_change:+,.0f} | {r.flagged_per_case_found_T0:.1f} | "
                     f"{r.flagged_per_case_found_T1:.1f} | {r.ppv_T0:.3f} | {r.ppv_T1:.3f} |")
        w2 = wide[wide.rule == "confirmatory"]
        L += ["", "## A2b. Missed cases per 100,000 at specificity 0.90 (mean over seeds)", "",
              "| model | sensitivity T0 | sensitivity T1 | missed T0 | missed T1 | change |",
              "|---|---|---|---|---|---|"]
        for _, r in w2.iterrows():
            L.append(f"| {r.model} | {r.sensitivity_T0:.3f} | {r.sensitivity_T1:.3f} | "
                     f"{r.missed_cases_per_100k_T0:,.0f} | {r.missed_cases_per_100k_T1:,.0f} | "
                     f"{r.missed_cases_per_100k_change:+,.0f} |")
        L.append("")

    recl = results.get("A2r")
    if recl is not None and not recl.empty:
        g = recl[recl.rule == "screening"].groupby("model")[
            ["decision_differs_per_100k", "right_T0_wrong_T1_per_100k",
             "wrong_T0_right_T1_per_100k", "net_extra_wrong_decisions_per_100k"]].mean()
        L += ["## A2c. People whose screening decision changes (per 100,000, mean over seeds)", "",
              "| model | decision differs | right under T0, wrong under T1 | wrong under T0, right under T1 | net extra wrong |",
              "|---|---|---|---|---|"]
        for mdl, r in g.iterrows():
            L.append(f"| {mdl} | {r.iloc[0]:,.0f} | {r.iloc[1]:,.0f} | {r.iloc[2]:,.0f} | {r.iloc[3]:,.0f} |")
        L.append("")

    mech = results.get("A3")
    if mech is not None and not mech.empty:
        g = mech.groupby("model").mean(numeric_only=True)
        L += ["## A3. Who the leaky model detects (mean over seeds)", "",
              "| model | cases with a prior coronary marker | share of T0 detections with marker | AUROC T0 / T1 among people without any marker |",
              "|---|---|---|---|"]
        for mdl, r in g.iterrows():
            a0 = r.get("auroc_T0_no_marker", np.nan)
            a1 = r.get("auroc_T1_no_marker", np.nan)
            L.append(f"| {mdl} | {100*r.share_of_cases_with_marker:.1f}% | "
                     f"{100*r.share_of_T0_detections_with_marker:.1f}% | {a0:.4f} / {a1:.4f} |")
        L.append("")

    sx = results.get("A5")
    if sx is not None and not sx.empty:
        g = sx.groupby(["model", "sex"]).mean(numeric_only=True).reset_index()
        L += ["## A5. Sex audit in plain counts (T1, screening threshold, mean over seeds)", "",
              "| model | sex | people | cases | detected | missed | sensitivity | specificity | prevalence |",
              "|---|---|---|---|---|---|---|---|---|"]
        for _, r in g.iterrows():
            L.append(f"| {r.model} | {r.sex} | {r.n:,.0f} | {r.cases:,.0f} | {r.detected:,.0f} | "
                     f"{r.missed:,.0f} | {r.sensitivity:.3f} | {r.specificity:.3f} | {100*r.prevalence:.2f}% |")
        L.append("")

    cv = results.get("A6")
    if cv is not None and not cv.empty:
        g = cv.groupby(["sex", "people"]).mean(numeric_only=True).reset_index()
        L += ["## A6. Conformal coverage split by sex and true class (EBM, T1, mean over seeds)", "",
              "| sex | people | share of group | coverage |", "|---|---|---|---|"]
        for _, r in g.iterrows():
            L.append(f"| {r.sex} | {r.people} | {100*r.share_of_sex_group:.1f}% | {r.coverage:.3f} |")
        L.append("")

    dl = results.get("A11")
    if dl is not None and not dl.empty and "auc_ebm_old_file" in dl:
        gap = (dl.auc_ebm_old_file - dl.auc_ebm_sklearn).abs().max()
        L += ["## A11. Check of results/delong.csv", "",
              f"Largest AUROC error in the old delong.csv: {gap:.4f}. "
              "If this is above 0.001, any p-value taken from that file must be "
              "replaced by delong_recheck.csv. (The paper's non-inferiority tests "
              "came from analysis/equivalence_rev.py, which was already correct.)", ""]

    (c.out / "SUMMARY.md").write_text("\n".join(L) + "\n")
    log(f"summary written to {c.out / 'SUMMARY.md'}")


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    a = ap.parse_args()
    cfg = load_config(a.config)
    c = Ctx(cfg)

    steps = [("A1", a1_significance), ("A2", a2_decisions), ("A3", a3_mechanism),
             ("A4", a4_subgroups), ("A5", a5_sex_plain), ("A6", a6_coverage),
             ("A7", a7_missingness), ("A8", a8_cost), ("A9", a9_temporal),
             ("A10", a10_calibration), ("A11", a11_delong_recheck)]
    results, status = {}, {}
    for key, fn in steps:
        log(f"--- {key} ---")
        try:
            out = fn(c)
            if key == "A2":
                _, recl, wide = out
                results["A2"], results["A2r"] = wide, recl
            else:
                results[key] = out
            status[key] = "ok"
        except Exception as e:                              # noqa: BLE001
            status[key] = f"FAILED: {type(e).__name__}: {e}"
            traceback.print_exc()
            log(f"{key} failed; continuing with the others")
    write_summary(c, results)
    (c.out / "run_status.json").write_text(json.dumps(status, indent=2))
    bad = {k: v for k, v in status.items() if v != "ok"}
    log("ALL ANALYSES COMPLETED" if not bad else f"completed with failures: {list(bad)}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
