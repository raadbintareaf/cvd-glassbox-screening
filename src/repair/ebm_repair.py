"""Glass-box fairness repair by direct shape-function editing (contribution C4).

The EBM's additive structure makes targeted, auditable interventions possible:
every edit below is (i) a pure function of the fitted model, (ii) recorded as
a machine-readable edit log for the paper's supplement, and (iii) reversible.
This operationalizes the interaction paradigm of GAM Changer (Wang et al.,
KDD 2022) as a quantified repair protocol rather than an interactive tool.

Implemented edits:
  neutralize_terms(labels)   — zero the shape contribution of the listed
                               feature(s) and any interaction involving them
                               (e.g., remove the direct Sex main effect).
  scale_terms(labels, gamma) — attenuate rather than remove (gamma in [0,1]).
  equalize_group_intercept(...) — after neutralizing, re-center per-group
                               mean logit on validation data so base rates
                               are not silently shifted (records deltas).

Each function returns (edited_model, edit_log). The original model is never
mutated.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd


def _terms_involving(ebm, labels: set[str]) -> list[int]:
    idxs = []
    for i, name in enumerate(ebm.term_names_):
        parts = {p.strip() for p in name.split("&")}
        if parts & labels:
            idxs.append(i)
    return idxs


def neutralize_terms(ebm, labels: list[str]):
    m = copy.deepcopy(ebm)
    labelset = set(labels)
    edited = []
    for i in _terms_involving(m, labelset):
        before_range = float(np.ptp(m.term_scores_[i]))
        m.term_scores_[i] = np.zeros_like(m.term_scores_[i])
        edited.append({"term": m.term_names_[i], "action": "zeroed",
                       "prior_score_range": before_range})
    log = {"edit": "neutralize_terms", "targets": labels, "terms": edited}
    return m, log


def scale_terms(ebm, labels: list[str], gamma: float):
    m = copy.deepcopy(ebm)
    labelset = set(labels)
    edited = []
    for i in _terms_involving(m, labelset):
        m.term_scores_[i] = m.term_scores_[i] * gamma
        edited.append({"term": m.term_names_[i], "action": f"scaled x{gamma}"})
    log = {"edit": "scale_terms", "targets": labels, "gamma": gamma,
           "terms": edited}
    return m, log


def equalize_group_intercept(ebm, X_val: pd.DataFrame, y_val: np.ndarray,
                             group: pd.Series):
    """Re-center per-group mean predicted logit to the overall mean via an
    additive per-group offset applied at prediction time. Returns a wrapped
    predictor and the offset log. Used after neutralization so that removing
    a group term does not silently change group-specific operating behavior."""
    from scipy.special import logit
    p = ebm.predict_proba(X_val)[:, 1].clip(1e-6, 1 - 1e-6)
    lp = logit(p)
    overall = lp.mean()
    offsets = {str(g): float(overall - lp[(group == g).values].mean())
               for g in group.dropna().unique()}

    class _Wrapped:
        def __init__(self, base, offsets):
            self.base, self.offsets = base, offsets

        def predict_proba_pos(self, X, group_series):
            from scipy.special import expit, logit as _lg
            pp = self.base.predict_proba(X)[:, 1].clip(1e-6, 1 - 1e-6)
            off = group_series.astype(str).map(self.offsets).fillna(0.0)
            return expit(_lg(pp) + off.values)

    log = {"edit": "equalize_group_intercept", "offsets": offsets}
    return _Wrapped(ebm, offsets), log
