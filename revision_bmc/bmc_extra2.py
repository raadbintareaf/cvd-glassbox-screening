"""Second round of checks for the BMC submission (reads saved predictions only;
no retraining; about 10-15 minutes on CPU).

  C1 strict_decisions.csv   B1 repeated with a STRICT definition of the
                            group: respondents who answered "no" to both
                            items (missing answers excluded), next to the
                            main definition (not "yes" to either item)
  C2 strict_ci.csv          paired case bootstrap CIs (as B2) for both
                            definitions
  C3 threshold_ci.csv       CIs that also re-estimate the threshold: each
                            replicate resamples the validation set
                            (stratified by outcome), re-derives the threshold,
                            and resamples the test cases of the group
  C4 dca_group_recal.csv    decision curves in the group with isotonic
                            recalibration fitted on validation respondents of
                            the same group
  C5 group_composition.csv  composition of the group: missing answers and
                            their prevalence

Usage (from the repository root, paper environment active):
    python -m revision_bmc.bmc_extra2 --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from revision_bmc.bmc_extra_analyses import PT_GRID, X, net_benefit, rates
from revision_jamiao.jamiao_analyses import MODEL_ORDER, SCREEN, is_yes, log, seeds_for
from src.evaluation.metrics import threshold_for
from src.utils.common import load_config

N_BOOT_CASES = 1000
N_BOOT_THR = 200


class Y(X):
    def __init__(self, cfg):
        super().__init__(cfg)
        from pathlib import Path
        self.out = self.res_dir / "bmc_extra2"
        self.out.mkdir(parents=True, exist_ok=True)
        ang, ct = self.df[self.angina_lab], self.df[self.ct_lab]
        no_ang = (~is_yes(ang)) & ang.notna().values
        no_ct = (~is_yes(ct)) & ct.notna().values
        self.strict = no_ang & no_ct            # answered "no" to both
        self.main = ~self.known                 # not "yes" to either (main)
        self.ang_missing = ang.isna().values
        self.ct_missing = ct.isna().values
        log(f"main group {int(self.main.sum()):,}; strict group {int(self.strict.sum()):,}")


def c5_composition(c: Y) -> pd.DataFrame:
    y = c.df[c.outcome].values.astype(int)
    rows = []
    for name, mk in (("main (not yes to either)", c.main), ("strict (no to both)", c.strict),
                     ("main, angina answer missing", c.main & c.ang_missing),
                     ("main, chest CT answer missing", c.main & c.ct_missing),
                     ("main, either answer missing", c.main & (c.ang_missing | c.ct_missing))):
        rows.append(dict(group=name, n=int(mk.sum()), cases=int(y[mk].sum()),
                         prevalence=float(y[mk].mean()) if mk.any() else np.nan,
                         share_of_main_n=float(mk.sum() / c.main.sum()),
                         share_of_main_cases=float(y[mk].sum() / y[c.main].sum())))
    df = pd.DataFrame(rows)
    df.to_csv(c.out / "group_composition.csv", index=False)
    log("C5: done")
    return df


def c1_c2(c: Y):
    rows, store = [], {}
    for m in c.models:
        for s in seeds_for(c.pred, "T1", m):
            if s not in seeds_for(c.pred, "T0", m):
                continue
            v0, t0, v1, t1 = c.pair(m, s)
            yv, yt = v0.y.values.astype(int), t0.y.values.astype(int)
            rv, rt_ = v0.row_id.values, t0.row_id.values
            for dname, mask in (("main", c.main), ("strict", c.strict)):
                gv, gt = mask[rv], mask[rt_]
                for tier, v, t in (("T0", v0, t0), ("T1", v1, t1)):
                    thr_all = threshold_for(yv, v.p.values, *SCREEN)
                    thr_g = threshold_for(yv[gv], v.p.values[gv], *SCREEN)
                    for src, thr in (("overall_validation", thr_all), ("group_validation", thr_g)):
                        d = t.p.values >= thr
                        r = rates(yt[gt], d[gt])
                        r.update(model=m, seed=s, tier=tier, definition=dname,
                                 threshold_set_on=src, auroc=roc_auc_score(yt[gt], t.p.values[gt]))
                        rows.append(r)
                        store[(dname, src, m, s, tier)] = d[gt & (yt == 1)]
        log(f"C1 {m}: done")
    df = pd.DataFrame(rows)
    df.to_csv(c.out / "strict_decisions.csv", index=False)

    rng = np.random.default_rng(7)
    out = []
    for dname in ("main", "strict"):
        for src in ("overall_validation", "group_validation"):
            keys = {(m, s) for (d_, q, m, s, t) in store if d_ == dname and q == src}
            models = sorted({m for m, _ in keys}, key=MODEL_ORDER.index)
            seeds = sorted({s for _, s in keys})
            ncase = {s: len(store[(dname, src, models[0], s, "T0")]) for s in seeds}
            point = np.mean([np.mean([store[(dname, src, m, s, "T1")].mean() - store[(dname, src, m, s, "T0")].mean()
                                      for s in seeds]) for m in models])
            boots = []
            for _ in range(N_BOOT_CASES):
                idx = {s: rng.integers(0, ncase[s], ncase[s]) for s in seeds}
                boots.append(np.mean([np.mean([store[(dname, src, m, s, "T1")][idx[s]].mean()
                                               - store[(dname, src, m, s, "T0")][idx[s]].mean() for s in seeds])
                                      for m in models]))
            lo, hi = np.percentile(boots, [2.5, 97.5])
            out.append(dict(definition=dname, threshold_set_on=src, diff_T1_minus_T0=point,
                            ci_lo=lo, ci_hi=hi, mean_cases=np.mean(list(ncase.values()))))
    ci = pd.DataFrame(out)
    ci.to_csv(c.out / "strict_ci.csv", index=False)
    log("C2: done")
    return df, ci


def _strat_idx(rng, y):
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    return np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])


def c3_threshold_ci(c: Y) -> pd.DataFrame:
    """Bootstrap that includes threshold estimation (main definition)."""
    rng = np.random.default_rng(11)
    data = {}
    for m in c.models:
        for s in seeds_for(c.pred, "T1", m):
            if s not in seeds_for(c.pred, "T0", m):
                continue
            v0, t0, v1, t1 = c.pair(m, s)
            data[(m, s)] = (v0.y.values.astype(int), v0.p.values, v1.p.values, c.main[v0.row_id.values],
                            t0.y.values.astype(int), t0.p.values, t1.p.values, c.main[t0.row_id.values])
    models = sorted({m for m, _ in data}, key=MODEL_ORDER.index)
    seeds = sorted({s for _, s in data})
    res = {src: [] for src in ("overall_validation", "group_validation")}
    for b in range(N_BOOT_THR):
        vidx, tidx = {}, {}
        for s in seeds:
            yv, *_ , gv = data[(models[0], s)][:4]
            yt, gt = data[(models[0], s)][4], data[(models[0], s)][7]
            vidx[s] = (_strat_idx(rng, yv), None)
            cases = np.where(gt & (yt == 1))[0]
            tidx[s] = rng.choice(cases, len(cases))
        for src in res:
            diffs = []
            for m in models:
                dd = []
                for s in seeds:
                    yv, pv0, pv1, gv, yt, pt0, pt1, gt = data[(m, s)]
                    vi = vidx[s][0]
                    if src == "group_validation":
                        vi = vi[gv[vi]]
                    th0 = threshold_for(yv[vi], pv0[vi], *SCREEN)
                    th1 = threshold_for(yv[vi], pv1[vi], *SCREEN)
                    ti = tidx[s]
                    dd.append((pt1[ti] >= th1).mean() - (pt0[ti] >= th0).mean())
                diffs.append(np.mean(dd))
            res[src].append(np.mean(diffs))
        if (b + 1) % 20 == 0:
            log(f"C3 bootstrap {b + 1}/{N_BOOT_THR}")
    out = [dict(threshold_set_on=src, ci_lo=np.percentile(v, 2.5), ci_hi=np.percentile(v, 97.5),
                boot_mean=np.mean(v), n_boot=N_BOOT_THR) for src, v in res.items()]
    df = pd.DataFrame(out)
    df.to_csv(c.out / "threshold_ci.csv", index=False)
    return df


def c4_dca(c: Y) -> pd.DataFrame:
    rows = []
    for m in c.models:
        for s in seeds_for(c.pred, "T1", m):
            if s not in seeds_for(c.pred, "T0", m):
                continue
            v0, t0, v1, t1 = c.pair(m, s)
            gv, gt = c.main[v0.row_id.values], c.main[t0.row_id.values]
            yt = t0.y.values.astype(int)
            for tier, v, t in (("T0", v0, t0), ("T1", v1, t1)):
                iso = IsotonicRegression(out_of_bounds="clip").fit(v.p.values[gv], v.y.values[gv])
                p = iso.predict(t.p.values[gt])
                for pt in PT_GRID:
                    rows.append(dict(model=m, seed=s, tier=tier, pt=pt, net_benefit=net_benefit(yt[gt], p, pt)))
            prev = yt[gt].mean()
            for pt in PT_GRID:
                rows.append(dict(model=m, seed=s, tier="treat_all", pt=pt, net_benefit=prev - (1 - prev) * pt / (1 - pt)))
        log(f"C4 {m}: done")
    df = pd.DataFrame(rows)
    df.to_csv(c.out / "dca_group_recal.csv", index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    a = ap.parse_args()
    c = Y(load_config(a.config))
    status = {}
    for key, fn in (("C5", c5_composition), ("C1C2", c1_c2), ("C4", c4_dca), ("C3", c3_threshold_ci)):
        log(f"--- {key} ---")
        try:
            fn(c); status[key] = "ok"
        except Exception as e:                              # noqa: BLE001
            status[key] = f"FAILED: {type(e).__name__}: {e}"; traceback.print_exc()
    (c.out / "run_status.json").write_text(json.dumps(status, indent=2))
    bad = [k for k, v in status.items() if v != "ok"]
    log("ALL ANALYSES COMPLETED" if not bad else f"completed with failures: {bad}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
