"""Subgroup fairness audit and mitigation baselines.

Everything is computed from saved predictions + group columns at EXPLICIT
operating points (thresholds chosen on validation), fixing the draft's
unstated-threshold flaw. Small cells get Wilson CIs and are reported, never
narrated as point facts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import confusion_at, threshold_for


def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    ph = k / n
    den = 1 + z ** 2 / n
    c = (ph + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(ph * (1 - ph) / n + z ** 2 / (4 * n ** 2)) / den
    return float(max(0, c - h)), float(min(1, c + h))


def subgroup_audit(y, p, groups: pd.DataFrame, thr: float) -> pd.DataFrame:
    """Per-subgroup confusion metrics at one threshold, with Wilson CIs on
    sensitivity/FNR and specificity."""
    rows = []
    y = np.asarray(y)
    p = np.asarray(p)
    for col in groups.columns:
        for level, idx in groups.groupby(col, observed=True).groups.items():
            ii = groups.index.get_indexer(idx)
            c = confusion_at(y[ii], p[ii], thr)
            sens_lo, sens_hi = wilson_ci(c["tp"], c["tp"] + c["fn"])
            spec_lo, spec_hi = wilson_ci(c["tn"], c["tn"] + c["fp"])
            rows.append({"group_var": col, "group": str(level),
                         "n": int(len(ii)), "n_pos": int(c["tp"] + c["fn"]),
                         **c, "sens_lo": sens_lo, "sens_hi": sens_hi,
                         "spec_lo": spec_lo, "spec_hi": spec_hi})
    return pd.DataFrame(rows)


def equalized_odds_gaps(audit: pd.DataFrame) -> pd.DataFrame:
    out = []
    for gv, sub in audit.groupby("group_var"):
        out.append({"group_var": gv,
                    "tpr_gap": float(sub.sensitivity.max()
                                     - sub.sensitivity.min()),
                    "fpr_gap": float(sub.fpr.max() - sub.fpr.min())})
    return pd.DataFrame(out)


def group_thresholds(y_val, p_val, groups_val: pd.Series, rule: str,
                     target: float) -> dict:
    """Mitigation arm (a): per-group thresholds on validation data."""
    thrs = {}
    for level in groups_val.dropna().unique():
        m = (groups_val == level).values
        if m.sum() < 50 or y_val[m].sum() < 5:
            thrs[str(level)] = threshold_for(y_val, p_val, rule, target)
        else:
            thrs[str(level)] = threshold_for(y_val[m], p_val[m], rule, target)
    return thrs


def apply_group_thresholds(p, groups: pd.Series, thrs: dict,
                           fallback: float) -> np.ndarray:
    t = groups.astype(str).map(thrs).fillna(fallback).astype(float).values
    return (np.asarray(p) >= t).astype(int)


def reweighing_weights(y: np.ndarray, group: pd.Series) -> np.ndarray:
    """Kamiran & Calders reweighing: w(g,y) = P(g)P(y) / P(g,y).
    Mitigation arm (c): multiply into training sample weights."""
    g = group.astype(str).values
    n = len(y)
    w = np.ones(n)
    for gl in np.unique(g):
        for yl in (0, 1):
            m = (g == gl) & (y == yl)
            pj = m.mean()
            if pj > 0:
                w[m] = ((g == gl).mean() * (y == yl).mean()) / pj
    return w / w.mean()
