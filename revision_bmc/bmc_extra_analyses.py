"""Additional analyses for the BMC Medical Informatics and Decision Making
submission (target leakage in BRFSS-based MI screening models).

Most analyses read the saved predictions; parts B6 and B7 retrain the
classical models (no foundation models) for two new comparisons.

Inputs (written by the original pipeline):
    results/predictions/brfss2022/<tier>/<model>/seed<k>.parquet
    results/models/brfss2022/<tier>/ebm/seed0/model.joblib   (optional)
    results/tuned/<model>__<tier>.yaml                        (optional)
    data/analytic/brfss_2022_analytic.parquet  (and 2023, optional)

Outputs: results/bmc_extra/*.csv, SUMMARY.md, run_status.json

"No marker" means the respondent did not answer "yes" to either of the two
items removed between T0 and T1 (prior angina/CHD diagnosis, chest CT);
this is the same definition as revision_jamiao A3.

  B1  no_marker_decisions.csv   decisions among people WITHOUT either item,
                                at the overall threshold and at a threshold
                                re-set within no-marker validation data
  B2  no_marker_sens_ci.csv     bootstrap 95% CI for the T1 - T0 sensitivity
                                difference among no-marker cases
  B3  dca.csv                   net benefit (decision curves), all
                                respondents and no-marker respondents
  B4  subgroup_by_age.csv       sex and race/ethnicity results within age bands
  B5  subgroup_thresholds.csv   group-specific thresholds (sex, age, race)
  B6  ablation_*.csv            T0 minus only the angina/CHD item, and T0
                                minus only the chest CT item (retrained)
  B7  intended_population.csv   models trained, thresholded and evaluated in
                                people without either item (retrained)
  B8  weighted.csv              survey-weighted prevalence and decisions
  B9  complete_case.csv         complete-case shares per feature set
  B10 item_pattern.csv          who answered the two items (age, smoking)
  B11 no_marker_profile.csv     profile of cases with and without the items
  B12 ebm_importance.csv        EBM term importances, T0 and T1 (seed 0)
  B13 calibration_summary.csv   O/E ratio and calibration slope, raw and
                                isotonic, T0 and T1

Usage (from the repository root):
    python -m revision_bmc.bmc_extra_analyses --config configs/default.yaml
    options: --skip-retrain   (B6/B7 off)
             --models logreg,ebm,...   (models for B6/B7)
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
from sklearn.metrics import roc_auc_score

from revision_jamiao.jamiao_analyses import (CONFIRM, MODEL_ORDER, SCREEN, Ctx,
                                             is_yes, load_pred, log, seeds_for)
from src.evaluation.metrics import confusion_at, threshold_for
from src.utils.common import load_config

PER = 100_000
N_BOOT = 1000
RETRAIN_MODELS = ["logreg", "logreg_spline", "random_forest", "xgboost",
                  "lightgbm", "catboost", "ebm", "mlp"]
PT_GRID = np.round(np.arange(0.01, 0.301, 0.01), 2)


# --------------------------------------------------------------------------
class X(Ctx):
    """Ctx from revision_jamiao plus the no-marker mask and output folder."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.out = self.res_dir / "bmc_extra"
        self.out.mkdir(parents=True, exist_ok=True)
        flags = [is_yes(self.df[f]) for f in self.leak]
        self.known = np.logical_or.reduce(flags)
        self.angina_lab = next((f for f in self.leak if "angina" in f.lower()), self.leak[0])
        self.ct_lab = next((f for f in self.leak if "chest" in f.lower()), self.leak[-1])
        log(f"no-marker respondents: {int((~self.known).sum()):,} of {len(self.known):,}")

    def mask(self, rid):
        return ~self.known[rid]


def rates(y, d, n_pop=None):
    """Decision counts per 100,000 of the population evaluated."""
    y, d = np.asarray(y).astype(int), np.asarray(d).astype(bool)
    n = len(y) if n_pop is None else n_pop
    tp = int((d & (y == 1)).sum()); fp = int((d & (y == 0)).sum())
    fn = int((~d & (y == 1)).sum()); tn = int((~d & (y == 0)).sum())
    return dict(n=len(y), cases=int(y.sum()), prevalence=y.mean(),
                sensitivity=tp / max(tp + fn, 1), specificity=tn / max(tn + fp, 1),
                ppv=tp / max(tp + fp, 1),
                flagged_per_100k=(tp + fp) / n * PER, false_pos_per_100k=fp / n * PER,
                found_per_100k=tp / n * PER, missed_per_100k=fn / n * PER,
                tp=tp, fp=fp, fn=fn, tn=tn)


def lab(series) -> pd.Series:
    """String labels with missing values as 'nan' (safe to sort and group)."""
    s = series.astype(object)
    return s.where(series.notna(), "nan").astype(str)


def iso(v, t):
    m = IsotonicRegression(out_of_bounds="clip").fit(v.p.values, v.y.values)
    return m.predict(t.p.values)


def net_benefit(y, p, pt):
    n = len(y); d = p >= pt
    tp = (d & (y == 1)).sum(); fp = (d & (y == 0)).sum()
    return tp / n - fp / n * pt / (1 - pt)


# --------------------------------------------------------------------------
# B1 + B2: people without either item
# --------------------------------------------------------------------------

def b1_no_marker(c: X):
    rows, store = [], {}
    for m in c.models:
        for s in seeds_for(c.pred, "T1", m):
            if s not in seeds_for(c.pred, "T0", m):
                continue
            v0, t0, v1, t1 = c.pair(m, s)
            yv, yt = v0.y.values.astype(int), t0.y.values.astype(int)
            nm_v, nm_t = c.mask(v0.row_id.values), c.mask(t0.row_id.values)
            for tier, v, t in (("T0", v0, t0), ("T1", v1, t1)):
                thr_all = threshold_for(yv, v.p.values, *SCREEN)
                thr_nm = threshold_for(yv[nm_v], v.p.values[nm_v], *SCREEN)
                for src, thr in (("overall_validation", thr_all),
                                 ("no_marker_validation", thr_nm)):
                    d = t.p.values >= thr
                    for pop, mk in (("no_marker", nm_t), ("marker", ~nm_t), ("all", np.ones_like(nm_t))):
                        r = rates(yt[mk], d[mk])
                        r.update(model=m, seed=s, tier=tier, threshold_set_on=src,
                                 population=pop, threshold=thr)
                        if pop == "no_marker":
                            r["auroc"] = roc_auc_score(yt[mk], t.p.values[mk])
                        rows.append(r)
                    store[(m, s, tier, src)] = d[nm_t & (yt == 1)]
        log(f"B1 {m}: done")
    df = pd.DataFrame(rows)
    df.to_csv(c.out / "no_marker_decisions.csv", index=False)
    return df, store


def b2_ci(c: X, store) -> pd.DataFrame:
    """Paired bootstrap over no-marker cases (same cases for T0 and T1, and
    for all models on a seed, since test rows are shared)."""
    rng = np.random.default_rng(2026)
    out = []
    for src in ("overall_validation", "no_marker_validation"):
        keys = sorted({(m, s) for (m, s, t, q) in store if q == src})
        models = sorted({m for m, _ in keys}, key=MODEL_ORDER.index)
        seeds = sorted({s for _, s in keys})
        point = {m: np.mean([store[(m, s, "T1", src)].mean() - store[(m, s, "T0", src)].mean()
                             for s in seeds if (m, s) in keys]) for m in models}
        boots = {m: [] for m in models + ["ALL_MODELS"]}
        n_cases = {s: len(store[(models[0], s, "T0", src)]) for s in seeds}
        for _ in range(N_BOOT):
            idx = {s: rng.integers(0, n_cases[s], n_cases[s]) for s in seeds}
            per_model = []
            for m in models:
                dd = [store[(m, s, "T1", src)][idx[s]].mean() - store[(m, s, "T0", src)][idx[s]].mean()
                      for s in seeds if (m, s) in keys]
                boots[m].append(np.mean(dd)); per_model.append(np.mean(dd))
            boots["ALL_MODELS"].append(np.mean(per_model))
        point["ALL_MODELS"] = np.mean([point[m] for m in models])
        for m in models + ["ALL_MODELS"]:
            lo, hi = np.percentile(boots[m], [2.5, 97.5])
            out.append(dict(threshold_set_on=src, model=m, diff_T1_minus_T0=point[m],
                            ci_lo=lo, ci_hi=hi, n_boot=N_BOOT,
                            mean_no_marker_cases=np.mean(list(n_cases.values()))))
    df = pd.DataFrame(out)
    df.to_csv(c.out / "no_marker_sens_ci.csv", index=False)
    log("B2: done")
    return df


# --------------------------------------------------------------------------
# B3: decision curves on isotonic-recalibrated probabilities
# --------------------------------------------------------------------------

def b3_dca(c: X) -> pd.DataFrame:
    rows = []
    for m in c.models:
        for s in seeds_for(c.pred, "T1", m):
            if s not in seeds_for(c.pred, "T0", m):
                continue
            v0, t0, v1, t1 = c.pair(m, s)
            yt = t0.y.values.astype(int)
            nm = c.mask(t0.row_id.values)
            for tier, v, t in (("T0", v0, t0), ("T1", v1, t1)):
                p = iso(v, t)
                for pop, mk in (("all", np.ones_like(nm)), ("no_marker", nm)):
                    for pt in PT_GRID:
                        rows.append(dict(model=m, seed=s, tier=tier, population=pop, pt=pt,
                                         net_benefit=net_benefit(yt[mk], p[mk], pt)))
            for pop, mk in (("all", np.ones_like(nm)), ("no_marker", nm)):
                prev = yt[mk].mean()
                for pt in PT_GRID:
                    rows.append(dict(model=m, seed=s, tier="treat_all", population=pop, pt=pt,
                                     net_benefit=prev - (1 - prev) * pt / (1 - pt)))
        log(f"B3 {m}: done")
    df = pd.DataFrame(rows)
    df.to_csv(c.out / "dca.csv", index=False)
    return df


# --------------------------------------------------------------------------
# B4 + B5: subgroups within age bands; group-specific thresholds
# --------------------------------------------------------------------------

def b4_b5_subgroups(c: X):
    rows, thr_rows = [], []
    race = c.race
    for m in c.models:
        for s in seeds_for(c.pred, "T1", m):
            if s not in seeds_for(c.pred, "T0", m):
                continue
            v0, t0, v1, t1 = c.pair(m, s)
            yv, yt = v0.y.values.astype(int), t0.y.values.astype(int)
            gv = c.df.iloc[v0.row_id.values].reset_index(drop=True)
            gt = c.df.iloc[t0.row_id.values].reset_index(drop=True)
            for tier, v, t in (("T0", v0, t0), ("T1", v1, t1)):
                thr = threshold_for(yv, v.p.values, *SCREEN)
                d = t.p.values >= thr
                for gname in ["Sex"] + ([race] if race else []):
                    keyg, keya = lab(gt[gname]), lab(gt["AgeBand"])
                    for (g, a), ix in gt.groupby([keyg, keya]).indices.items():
                        ix = np.asarray(ix)
                        if a == "nan" or g == "nan":
                            continue
                        r = rates(yt[ix], d[ix])
                        r.update(model=m, seed=s, tier=tier, variable=gname, group=g, age_band=a)
                        rows.append(r)
                if tier != "T1":
                    continue
                for gname in ["Sex", "AgeBand"] + ([race] if race else []):
                    for g in sorted(lab(gt[gname]).unique()):
                        if g == "nan":
                            continue
                        mv = (lab(gv[gname]) == g).values
                        mt = (lab(gt[gname]) == g).values
                        if yv[mv].sum() < 5:
                            continue
                        thr_g = threshold_for(yv[mv], v.p.values[mv], *SCREEN)
                        for kind, th in (("shared", thr), ("group_specific", thr_g)):
                            r = rates(yt[mt], t.p.values[mt] >= th)
                            r.update(model=m, seed=s, tier=tier, variable=gname, group=g,
                                     threshold_kind=kind, threshold=th)
                            thr_rows.append(r)
        log(f"B4/B5 {m}: done")
    a = pd.DataFrame(rows); b = pd.DataFrame(thr_rows)
    a.to_csv(c.out / "subgroup_by_age.csv", index=False)
    b.to_csv(c.out / "subgroup_thresholds.csv", index=False)
    return a, b


# --------------------------------------------------------------------------
# B6 + B7: retraining (classical models only)
# --------------------------------------------------------------------------

def _fit_predict(c: X, model, params_tier, feats, tr, va, te, seed):
    from experiments.run_benchmark import model_params
    from src.models.registry import make_model
    from src.utils.common import seed_everything
    seed_everything(seed)
    Xa = c.df[feats].copy(); y = c.df[c.outcome].values.astype(int)
    # nullable-integer (Int8) yes/no columns -> float: identical values, but
    # median imputation in the adapters can then fill 0.5 without a dtype error
    for col in Xa.columns:
        if str(Xa[col].dtype) in ("Int8", "Int16", "Int32", "Int64"):
            Xa[col] = Xa[col].astype("float64")
    m = make_model(model, model_params(c.cfg, model, params_tier), seed)
    t0 = time.time()
    m.fit(Xa.iloc[tr], y[tr], Xa.iloc[va], y[va])
    fit_s = time.time() - t0
    return m.predict_proba_pos(Xa.iloc[va]), m.predict_proba_pos(Xa.iloc[te]), fit_s


def b6_b7_retrain(c: X, models: list[str]):
    from src.utils.common import stratified_splits
    y = c.df[c.outcome].values.astype(int)
    t0f = list(c.t0)
    arms = {
        "T0_minus_angina": dict(feats=[f for f in t0f if f != c.angina_lab], params="T0", pop="all"),
        "T0_minus_chestct": dict(feats=[f for f in t0f if f != c.ct_lab], params="T0", pop="all"),
        "T1_intended_population": dict(feats=list(c.t1), params="T1", pop="no_marker"),
    }
    pred_root = c.out / "predictions"
    rows = []
    seeds = c.cfg.get("seeds", [0, 1, 2, 3, 4])
    for arm, spec in arms.items():
        for model in models:
            for s in seeds:
                f = pred_root / arm / model / f"seed{s}.parquet"
                idx = stratified_splits(y, s, tuple(c.cfg["split_fracs"]))
                tr, va, te = idx["train"], idx["val"], idx["test"]
                if spec["pop"] == "no_marker":
                    tr, va = tr[~c.known[tr]], va[~c.known[va]]
                if f.exists():
                    d = pd.read_parquet(f)
                    pv = d[d.split == "val"].sort_values("row_id").p.values
                    pt_ = d[d.split == "test"].sort_values("row_id").p.values
                    fit_s = float(d.attrs.get("fit_s", np.nan)) if hasattr(d, "attrs") else np.nan
                else:
                    try:
                        pv, pt_, fit_s = _fit_predict(c, model, spec["params"], spec["feats"], tr, va, te, s)
                    except Exception as e:                      # noqa: BLE001
                        log(f"B6/B7 {arm} {model} seed{s} FAILED: {type(e).__name__}: {e}")
                        continue
                    f.parent.mkdir(parents=True, exist_ok=True)
                    pd.concat([pd.DataFrame({"row_id": va, "split": "val", "y": y[va], "p": pv}),
                               pd.DataFrame({"row_id": te, "split": "test", "y": y[te], "p": pt_})]
                              ).to_parquet(f, index=False)
                yv, yt = y[va], y[te]
                nm_t, nm_v = ~c.known[te], ~c.known[va]
                thr_all = threshold_for(yv, pv, *SCREEN)
                thr_nm = threshold_for(yv[nm_v], pv[nm_v], *SCREEN)
                base = dict(arm=arm, model=model, seed=s, n_features=len(spec["feats"]),
                            fit_seconds=fit_s,
                            auroc_all=roc_auc_score(yt, pt_),
                            auroc_no_marker=roc_auc_score(yt[nm_t], pt_[nm_t]))
                for src, thr in (("overall_validation", thr_all), ("no_marker_validation", thr_nm)):
                    if spec["pop"] == "no_marker" and src == "overall_validation":
                        continue   # its validation set is already no-marker only
                    d_ = pt_ >= thr
                    for pop, mk in (("all", np.ones_like(nm_t)), ("no_marker", nm_t)):
                        if spec["pop"] == "no_marker" and pop == "all":
                            continue
                        r = rates(yt[mk], d_[mk]); r.update(base, threshold_set_on=src,
                                                           population=pop, threshold=thr)
                        rows.append(r)
                log(f"B6/B7 {arm} {model} seed{s}: AUROC no-marker {base['auroc_no_marker']:.4f}")
            pd.DataFrame(rows).to_csv(c.out / "retrain_results.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(c.out / "retrain_results.csv", index=False)
    return df


# --------------------------------------------------------------------------
# B8: survey weights
# --------------------------------------------------------------------------

def b8_weighted(c: X) -> pd.DataFrame:
    if "SurveyWeight" not in c.df.columns:
        log("B8: no SurveyWeight column; skipped")
        return pd.DataFrame()
    w_all = c.df["SurveyWeight"].values.astype(float)
    y_all = c.df[c.outcome].values.astype(int)
    rows = [dict(model="(cohort)", seed=-1, tier="", population=pop,
                 weighted_prevalence=float(np.average(y_all[mk], weights=w_all[mk])),
                 unweighted_prevalence=float(y_all[mk].mean()),
                 weighted_share_of_population=float(w_all[mk].sum() / w_all.sum()))
            for pop, mk in (("all", np.ones(len(y_all), bool)), ("no_marker", ~c.known),
                            ("marker", c.known))]
    for m in c.models:
        for s in seeds_for(c.pred, "T1", m):
            if s not in seeds_for(c.pred, "T0", m):
                continue
            v0, t0, v1, t1 = c.pair(m, s)
            rid = t0.row_id.values
            yt, w = t0.y.values.astype(int), w_all[rid]
            nm = c.mask(rid)
            for tier, v, t in (("T0", v0, t0), ("T1", v1, t1)):
                thr = threshold_for(v.y.values, v.p.values, *SCREEN)
                d = t.p.values >= thr
                for pop, mk in (("all", np.ones_like(nm)), ("no_marker", nm)):
                    ww, yy, dd = w[mk], yt[mk], d[mk]
                    W = ww.sum()
                    tp = ww[dd & (yy == 1)].sum(); fp = ww[dd & (yy == 0)].sum()
                    fn = ww[~dd & (yy == 1)].sum(); tn = ww[~dd & (yy == 0)].sum()
                    rows.append(dict(model=m, seed=s, tier=tier, population=pop,
                                     weighted_prevalence=(tp + fn) / W,
                                     sensitivity_w=tp / (tp + fn), specificity_w=tn / (tn + fp),
                                     flagged_per_100k_w=(tp + fp) / W * PER,
                                     false_pos_per_100k_w=fp / W * PER,
                                     missed_per_100k_w=fn / W * PER,
                                     auroc_w=roc_auc_score(yy, t.p.values[mk], sample_weight=ww)))
        log(f"B8 {m}: done")
    df = pd.DataFrame(rows)
    df.to_csv(c.out / "weighted.csv", index=False)
    return df


# --------------------------------------------------------------------------
# B9-B13: descriptive checks
# --------------------------------------------------------------------------

def b9_complete_case(c: X) -> pd.DataFrame:
    from experiments.run_benchmark import load_analytic
    frames = {"2022": c.df}
    try:
        frames["2023"] = load_analytic(c.cfg, "brfss2023")
    except SystemExit:
        pass
    rows = []
    for yr, d in frames.items():
        for t in ("T0", "T1", "T1_portable", "T2"):
            cols = [f for f in c.tier_labels(t) if f in d.columns]
            missing_cols = [f for f in c.tier_labels(t) if f not in d.columns]
            cc = d[cols].notna().all(axis=1)
            rows.append(dict(year=yr, feature_set=t, n=len(d), n_complete=int(cc.sum()),
                             share_complete=float(cc.mean()),
                             share_removed=float(1 - cc.mean()),
                             columns_not_in_file=";".join(missing_cols)))
        if "complete_case" in d.columns:
            rows.append(dict(year=yr, feature_set="complete_case column (all mapped predictors)",
                             n=len(d), n_complete=int(d["complete_case"].sum()),
                             share_complete=float(d["complete_case"].mean()),
                             share_removed=float(1 - d["complete_case"].mean())))
    df = pd.DataFrame(rows)
    df.to_csv(c.out / "complete_case.csv", index=False)
    log("B9: done")
    return df


def b10_item_pattern(c: X) -> pd.DataFrame:
    d = c.df
    y = d[c.outcome].values.astype(int)
    rows = []
    for item in (c.angina_lab, c.ct_lab):
        state = np.where(d[item].isna(), "missing", np.where(is_yes(d[item]), "yes", "no"))
        for by in ["AgeBand", "SmokerStatus", None]:
            groups = lab(d[by]) if by else pd.Series("all", index=d.index)
            for g in sorted(groups.unique()):
                mk = (groups == g).values
                rows.append(dict(item=item, by=by or "all", group=g, n=int(mk.sum()),
                                 share_missing=float((state[mk] == "missing").mean()),
                                 share_yes=float((state[mk] == "yes").mean()),
                                 share_yes_among_cases=float((state[mk & (y == 1)] == "yes").mean()) if (mk & (y == 1)).any() else np.nan,
                                 share_yes_among_noncases=float((state[mk & (y == 0)] == "yes").mean())))
        for st in ("yes", "no", "missing"):
            mk = state == st
            rows.append(dict(item=item, by="item_value", group=st, n=int(mk.sum()),
                             mi_prevalence=float(y[mk].mean()) if mk.any() else np.nan))
    rows.append(dict(item="either", by="all", group="all", n=len(y),
                     share_yes=float(c.known.mean()),
                     share_yes_among_cases=float(c.known[y == 1].mean()),
                     share_yes_among_noncases=float(c.known[y == 0].mean())))
    df = pd.DataFrame(rows)
    df.to_csv(c.out / "item_pattern.csv", index=False)
    log("B10: done")
    return df


def b11_profile(c: X) -> pd.DataFrame:
    d = c.df
    y = d[c.outcome].values.astype(int)

    def share(col, test):
        s = d[col]
        return lambda mk: float(test(s[mk]).mean()) if col in d.columns else np.nan

    feats = {
        "female": share("Sex", lambda s: s.astype(str) == "Female"),
        "age_18_39": share("AgeBand", lambda s: s == "18-39"),
        "age_40_59": share("AgeBand", lambda s: s == "40-59"),
        "age_60_plus": share("AgeBand", lambda s: s == "60+"),
        "checkup_within_1y": share("LastCheckupTime", lambda s: pd.to_numeric(s, errors="coerce") == 1),
        "fair_or_poor_health": share("GeneralHealth", lambda s: pd.to_numeric(s, errors="coerce") >= 4),
        "diabetes": share("DiabetesStatus", lambda s: s.astype(str) == "Yes"),
        "prior_stroke": share("HadStroke", lambda s: pd.to_numeric(s, errors="coerce") == 1),
        "current_smoker": share("SmokerStatus", lambda s: s.astype(str).str.startswith("Current")),
        "difficulty_walking": share("DiffWalking", lambda s: pd.to_numeric(s, errors="coerce") == 1),
        "flu_vaccine_12m": share("FluVaxLast12", lambda s: pd.to_numeric(s, errors="coerce") == 1),
    }
    rows = []
    for grp, mk in (("cases_with_item", (y == 1) & c.known), ("cases_without_item", (y == 1) & ~c.known),
                    ("noncases", y == 0)):
        r = dict(group=grp, n=int(mk.sum()))
        for k, fn in feats.items():
            try:
                r[k] = fn(mk)
            except Exception:                               # noqa: BLE001
                r[k] = np.nan
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(c.out / "no_marker_profile.csv", index=False)
    log("B11: done")
    return df


def b12_ebm_importance(c: X) -> pd.DataFrame:
    rows = []
    for tier in ("T0", "T1"):
        p = c.res_dir / "models" / "brfss2022" / tier / "ebm" / "seed0" / "model.joblib"
        if not p.exists():
            log(f"B12: {p} not found; skipped {tier}")
            continue
        import joblib
        m = joblib.load(p)
        ebm = getattr(m, "model_", m)
        names = list(ebm.term_names_)
        imp = np.asarray(ebm.term_importances())
        total = imp.sum()
        for rank, j in enumerate(np.argsort(-imp), 1):
            rows.append(dict(tier=tier, rank=rank, term=names[j], importance=float(imp[j]),
                             share_of_total=float(imp[j] / total)))
    df = pd.DataFrame(rows)
    df.to_csv(c.out / "ebm_importance.csv", index=False)
    log("B12: done")
    return df


def _slope(y, p):
    from sklearn.linear_model import LogisticRegression
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    return float(LogisticRegression(C=1e6, max_iter=1000).fit(z, y).coef_[0, 0])


def b13_calibration(c: X) -> pd.DataFrame:
    rows = []
    for m in c.models:
        for s in seeds_for(c.pred, "T1", m):
            if s not in seeds_for(c.pred, "T0", m):
                continue
            v0, t0, v1, t1 = c.pair(m, s)
            yt = t0.y.values.astype(int)
            for tier, v, t in (("T0", v0, t0), ("T1", v1, t1)):
                for kind, p in (("raw", t.p.values), ("isotonic", iso(v, t))):
                    rows.append(dict(model=m, seed=s, tier=tier, kind=kind,
                                     mean_predicted=float(p.mean()), observed=float(yt.mean()),
                                     oe_ratio=float(yt.mean() / p.mean()),
                                     cal_slope=_slope(yt, p),
                                     brier=float(np.mean((p - yt) ** 2))))
        log(f"B13 {m}: done")
    df = pd.DataFrame(rows)
    df.to_csv(c.out / "calibration_summary.csv", index=False)
    return df


# --------------------------------------------------------------------------
def summary(c: X, R: dict):
    L = ["# BMC extra analyses: key numbers", "",
         f"Generated {time.strftime('%Y-%m-%d %H:%M')}. No-marker = neither {c.angina_lab} nor {c.ct_lab} = yes.", ""]
    b1 = R.get("B1")
    if b1 is not None and len(b1):
        g = b1[b1.population == "no_marker"].groupby(["threshold_set_on", "tier"])[
            ["n", "cases", "sensitivity", "specificity", "flagged_per_100k",
             "false_pos_per_100k", "missed_per_100k", "auroc"]].mean()
        L += ["## B1 no-marker population (mean over models and seeds)", "", g.round(4).to_string(), ""]
    b2 = R.get("B2")
    if b2 is not None and len(b2):
        L += ["## B2 T1 - T0 sensitivity among no-marker cases", "",
              b2[b2.model == "ALL_MODELS"].round(4).to_string(index=False), ""]
    rt = R.get("B6B7")
    if rt is not None and len(rt):
        g = rt[rt.population == "no_marker"].groupby(["arm", "threshold_set_on"])[
            ["auroc_all", "auroc_no_marker", "sensitivity", "specificity", "false_pos_per_100k"]].mean()
        L += ["## B6/B7 retrained arms (no-marker population, mean)", "", g.round(4).to_string(), ""]
    cc = R.get("B9")
    if cc is not None and len(cc):
        L += ["## B9 complete case", "", cc.round(4).to_string(index=False), ""]
    (c.out / "SUMMARY.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--skip-retrain", action="store_true")
    ap.add_argument("--models", default=",".join(RETRAIN_MODELS))
    a = ap.parse_args()
    cfg = load_config(a.config)
    c = X(cfg)
    retrain_models = [m for m in a.models.split(",") if m]

    R, status = {}, {}

    def run(key, fn, *args):
        log(f"--- {key} ---")
        try:
            R[key] = fn(*args); status[key] = "ok"
        except Exception as e:                              # noqa: BLE001
            status[key] = f"FAILED: {type(e).__name__}: {e}"
            traceback.print_exc()

    log("--- B1 ---")
    try:
        b1, store = b1_no_marker(c); R["B1"] = b1; status["B1"] = "ok"
        run("B2", b2_ci, c, store)
    except Exception as e:                                  # noqa: BLE001
        status["B1"] = f"FAILED: {type(e).__name__}: {e}"; traceback.print_exc()
    run("B3", b3_dca, c)
    run("B4B5", b4_b5_subgroups, c)
    run("B8", b8_weighted, c)
    run("B9", b9_complete_case, c)
    run("B10", b10_item_pattern, c)
    run("B11", b11_profile, c)
    run("B12", b12_ebm_importance, c)
    run("B13", b13_calibration, c)
    if not a.skip_retrain:
        run("B6B7", b6_b7_retrain, c, retrain_models)
    else:
        status["B6B7"] = "skipped"
    summary(c, R)
    (c.out / "run_status.json").write_text(json.dumps(status, indent=2))
    bad = {k: v for k, v in status.items() if v not in ("ok", "skipped")}
    log("ALL ANALYSES COMPLETED" if not bad else f"completed with failures: {list(bad)}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
