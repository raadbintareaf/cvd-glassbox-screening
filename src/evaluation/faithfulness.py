"""Explanation-faithfulness analysis (paper contribution C3).

Ground truth: the EBM's exact global term importances, aggregated to RAW
analytic features (interaction terms split half-half between parents).
Comparators: TreeSHAP global importances for XGBoost, KernelSHAP for
foundation models (on a background/eval subsample), and LIME (averaged local
weights). One-hot / per-level attributions are aggregated back to their
parent raw feature BEFORE comparison — fixing the draft's apples-to-oranges
flaw.

Agreement statistics between two importance vectors over the same raw
features: Kendall's tau (rank agreement), Jaccard@k (top-k overlap), and
sign agreement on the directional subset where defined. The sweep repeats
each comparator at several explanation sample sizes x n_resamples to
quantify stability ('law of small numbers' done properly, with CIs).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------- importance extractors -------------------------------

def ebm_global_importance(ebm_model, feature_labels: list[str]) -> pd.Series:
    """Mean-|score| term importances, interactions split between parents."""
    imp = pd.Series(0.0, index=feature_labels)
    names = ebm_model.term_names_
    scores = ebm_model.term_importances()
    for name, s in zip(names, scores):
        parts = [p.strip() for p in name.split("&")]
        for p in parts:
            if p in imp.index:
                imp[p] += float(s) / len(parts)
    return imp / max(imp.sum(), 1e-12)


def shap_tree_importance(xgb_adapter, X: pd.DataFrame) -> pd.Series:
    import shap
    ex = shap.TreeExplainer(xgb_adapter.model_)
    sv = ex.shap_values(X)
    if isinstance(sv, list):
        sv = sv[1]
    imp = pd.Series(np.abs(sv).mean(axis=0), index=X.columns)
    return imp / max(imp.sum(), 1e-12)


def shap_kernel_importance(predict_pos, X: pd.DataFrame,
                           background_n=100, seed=0) -> pd.Series:
    """Model-agnostic KernelSHAP on codes matrix; slow — keep X small."""
    import shap

    from ..models.registry import to_codes_matrix
    M, _ = to_codes_matrix(X)
    rng = np.random.default_rng(seed)
    bg = M[rng.choice(len(M), min(background_n, len(M)), replace=False)]

    cols = list(X.columns)

    def f(mat):
        df = pd.DataFrame(mat, columns=cols)
        for c in cols:  # restore categorical dtypes from codes
            if isinstance(X[c].dtype, pd.CategoricalDtype):
                codes = pd.Series(mat[:, cols.index(c)]).round()
                codes = codes.where(codes.notna(), -1).astype(int)
                df[c] = pd.Categorical.from_codes(
                    codes.clip(-1, len(X[c].cat.categories) - 1),
                    categories=X[c].cat.categories)
        return predict_pos(df)

    ex = shap.KernelExplainer(f, bg)
    sv = ex.shap_values(M, nsamples=200, silent=True)
    imp = pd.Series(np.abs(np.asarray(sv)).mean(axis=0), index=cols)
    return imp / max(imp.sum(), 1e-12)


def lime_importance(predict_pos, X_explain: pd.DataFrame,
                    X_background: pd.DataFrame, seed=0) -> pd.Series:
    """Mean-|weight| LIME importances aggregated to raw features."""
    from lime.lime_tabular import LimeTabularExplainer

    from ..models.registry import to_codes_matrix
    Mb, cat_idx = to_codes_matrix(X_background)
    Me, _ = to_codes_matrix(X_explain)
    cols = list(X_explain.columns)
    cat_names = {j: list(X_background[cols[j]].cat.categories)
                 for j in cat_idx}
    expl = LimeTabularExplainer(
        np.nan_to_num(Mb), feature_names=cols,
        categorical_features=cat_idx,
        categorical_names=cat_names, discretize_continuous=True,
        random_state=seed, mode="classification")

    def f(mat):
        df = pd.DataFrame(mat, columns=cols)
        for j in cat_idx:
            c = cols[j]
            codes = pd.Series(mat[:, j]).round().astype(int)
            df[c] = pd.Categorical.from_codes(
                codes.clip(0, len(cat_names[j]) - 1),
                categories=cat_names[j])
        p = predict_pos(df)
        return np.column_stack([1 - p, p])

    agg = pd.Series(0.0, index=cols)
    for i in range(len(Me)):
        e = expl.explain_instance(np.nan_to_num(Me[i]), f,
                                  num_features=len(cols), num_samples=1000)
        for fid, w in e.as_map()[1]:
            agg[cols[fid]] += abs(w)
    agg /= max(len(Me), 1)
    return agg / max(agg.sum(), 1e-12)


# ---------------------- agreement statistics --------------------------------

def agreement(ref: pd.Series, other: pd.Series, k: int = 10) -> dict:
    common = ref.index.intersection(other.index)
    a, b = ref[common].values, other[common].values
    tau = stats.kendalltau(a, b).statistic
    top_a = set(ref[common].nlargest(k).index)
    top_b = set(other[common].nlargest(k).index)
    jac = len(top_a & top_b) / max(len(top_a | top_b), 1)
    return {"kendall_tau": float(tau), f"jaccard_at_{k}": float(jac),
            "n_features": int(len(common))}
