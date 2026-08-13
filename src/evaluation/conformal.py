"""Split and Mondrian (group-conditional) conformal prediction for binary
screening, computed from saved probabilities.

Nonconformity: LAC score s(x,y) = 1 - p_y(x). Prediction set at level alpha:
{y : s(x,y) <= q_hat}, with q_hat the ceil((n+1)(1-alpha))/n calibration
quantile. Mondrian variant computes q_hat per group (e.g., Sex x AgeBand),
giving finite-sample coverage within each group (Vovk; Gibbs et al. 2023).

We report per-group coverage, set-size distribution, and a deferral analysis:
treating ambiguous sets {0,1} as 'defer to clinician', what error rate remains
on confidently-labeled cases? Set sizes per group are reported alongside
coverage because equalized coverage alone can hide burden disparities
(Cresswell et al., 2024).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _qhat(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    if n == 0:
        return 1.0
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(max(k, 1), n)
    return float(np.sort(scores)[k - 1])


def _sets_from_q(p: np.ndarray, q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    """Return set code per row: 0={0}, 1={1}, 2={0,1}, 3=empty."""
    in0 = p <= q0        # s(x,0) = 1 - p_model(y=0) = p  ->  include 0 iff p <= q0
    in1 = (1 - p) <= q1
    code = np.full(len(p), 3)
    code[in0 & ~in1] = 0
    code[~in0 & in1] = 1
    code[in0 & in1] = 2
    return code


def split_conformal(y_cal, p_cal, p_test, alpha=0.1):
    s0 = p_cal[np.asarray(y_cal) == 0]          # s(x,0) = p
    s1 = 1 - p_cal[np.asarray(y_cal) == 1]      # s(x,1) = 1-p
    q0, q1 = _qhat(s0, alpha), _qhat(s1, alpha)
    code = _sets_from_q(np.asarray(p_test),
                        np.full(len(p_test), q0), np.full(len(p_test), q1))
    return code, {"q0": q0, "q1": q1}


def mondrian_conformal(y_cal, p_cal, g_cal: pd.Series,
                       p_test, g_test: pd.Series, alpha=0.1,
                       min_group=200):
    """Per-group quantiles; groups below min_group calibration rows fall back
    to the marginal quantiles (recorded in the returned info dict)."""
    y_cal = np.asarray(y_cal)
    p_cal = np.asarray(p_cal)
    s0_all, s1_all = p_cal[y_cal == 0], 1 - p_cal[y_cal == 1]
    q0m, q1m = _qhat(s0_all, alpha), _qhat(s1_all, alpha)
    q0g, q1g, fallback = {}, {}, []
    for lvl in g_cal.dropna().unique():
        m = (g_cal == lvl).values
        s0 = p_cal[m & (y_cal == 0)]
        s1 = 1 - p_cal[m & (y_cal == 1)]
        if m.sum() < min_group or len(s1) < 10:
            q0g[lvl], q1g[lvl] = q0m, q1m
            fallback.append(str(lvl))
        else:
            q0g[lvl], q1g[lvl] = _qhat(s0, alpha), _qhat(s1, alpha)
    q0 = g_test.map(q0g).fillna(q0m).astype(float).values
    q1 = g_test.map(q1g).fillna(q1m).astype(float).values
    code = _sets_from_q(np.asarray(p_test), q0, q1)
    return code, {"marginal": (q0m, q1m), "fallback_groups": fallback}


def summarize(code: np.ndarray, y_test, groups: pd.DataFrame) -> pd.DataFrame:
    """Coverage, set sizes, singleton error, deferral rate — overall and per
    subgroup level."""
    y = np.asarray(y_test)
    covered = ((code == 2) | ((code == 0) & (y == 0)) |
               ((code == 1) & (y == 1)))
    singleton = (code == 0) | (code == 1)
    sing_err = singleton & (((code == 0) & (y == 1)) |
                            ((code == 1) & (y == 0)))
    rows = [{"group_var": "ALL", "group": "ALL", "n": len(y),
             "coverage": float(covered.mean()),
             "avg_set_size": float(np.where(code == 2, 2,
                                   np.where(code == 3, 0, 1)).mean()),
             "deferral_rate": float((code == 2).mean()),
             "singleton_error": float(sing_err.sum() / max(singleton.sum(),
                                                           1)),
             "empty_rate": float((code == 3).mean())}]
    for col in groups.columns:
        for lvl, idx in groups.groupby(col, observed=True).groups.items():
            ii = groups.index.get_indexer(idx)
            rows.append({
                "group_var": col, "group": str(lvl), "n": len(ii),
                "coverage": float(covered[ii].mean()),
                "avg_set_size": float(np.where(code[ii] == 2, 2,
                                      np.where(code[ii] == 3, 0, 1)).mean()),
                "deferral_rate": float((code[ii] == 2).mean()),
                "singleton_error": float(
                    sing_err[ii].sum() / max(singleton[ii].sum(), 1)),
                "empty_rate": float((code[ii] == 3).mean())})
    return pd.DataFrame(rows)
