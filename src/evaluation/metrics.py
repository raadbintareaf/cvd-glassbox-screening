"""All metrics implemented once, imported everywhere.

Includes: AUROC/AUPRC/Brier, calibration slope & intercept & ECE, operating-
point selection and confusion metrics, fast DeLong AUC comparison, stratified
bootstrap CIs, Holm-Bonferroni, and decision-curve net benefit.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score


# ---------------------------- core scores ----------------------------------

def brier(y, p):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def calibration_slope_intercept(y, p, eps=1e-7):
    """Logistic recalibration: fit y ~ logit(p); slope 1 & intercept 0 ideal."""
    from sklearn.linear_model import LogisticRegression
    lp = np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1 - eps))
    lr = LogisticRegression(C=np.inf, max_iter=1000)  # unpenalized
    lr.fit(lp.reshape(-1, 1), y)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def ece(y, p, n_bins=15):
    y, p = np.asarray(y), np.asarray(p)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def threshold_for(y, p, rule: str, target: float) -> float:
    """Pick threshold on (validation) data. rule in {'sens_at','spec_at'}."""
    order = np.argsort(-p)
    ys, ps = np.asarray(y)[order], np.asarray(p)[order]
    P, N = ys.sum(), (1 - ys).sum()
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    sens = tp / max(P, 1)
    spec = 1 - fp / max(N, 1)
    if rule == "sens_at":            # smallest threshold achieving sens>=t
        ok = np.where(sens >= target)[0]
        j = ok[0] if len(ok) else len(ps) - 1
    elif rule == "spec_at":          # largest threshold keeping spec>=t
        ok = np.where(spec >= target)[0]
        j = ok[-1] if len(ok) else 0
    else:
        raise ValueError(rule)
    return float(ps[j])


def confusion_at(y, p, thr):
    y, yhat = np.asarray(y), (np.asarray(p) >= thr).astype(int)
    tp = int(((y == 1) & (yhat == 1)).sum())
    fn = int(((y == 1) & (yhat == 0)).sum())
    tn = int(((y == 0) & (yhat == 0)).sum())
    fp = int(((y == 0) & (yhat == 1)).sum())
    out = dict(tp=tp, fn=fn, tn=tn, fp=fp,
               sensitivity=tp / max(tp + fn, 1),
               specificity=tn / max(tn + fp, 1),
               ppv=tp / max(tp + fp, 1), npv=tn / max(tn + fn, 1),
               fnr=fn / max(tp + fn, 1), fpr=fp / max(tn + fp, 1))
    return {k: (float(v) if isinstance(v, float) else v)
            for k, v in out.items()}


def all_probability_metrics(y, p) -> dict:
    slope, intercept = calibration_slope_intercept(y, p)
    return {"auroc": float(roc_auc_score(y, p)),
            "auprc": float(average_precision_score(y, p)),
            "brier": brier(y, p),
            "cal_slope": slope, "cal_intercept": intercept,
            "ece": ece(y, p)}


# ---------------------------- DeLong ---------------------------------------

def _midrank(x):
    order = np.argsort(x)
    ranks = np.empty(len(x))
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i
        while j < len(x) and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1
        i = j
    return ranks


def delong_test(y, p1, p2):
    """Paired DeLong test for AUC(p1) vs AUC(p2). Returns (auc1, auc2, p)."""
    y = np.asarray(y)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    m, n = len(pos), len(neg)
    preds = np.vstack([np.asarray(p1), np.asarray(p2)])
    k = preds.shape[0]
    tx, ty, tz = (np.empty((k, m)), np.empty((k, n)),
                  np.empty((k, m + n)))
    for r in range(k):
        tx[r] = _midrank(preds[r, pos])
        ty[r] = _midrank(preds[r, neg])
        tz[r] = _midrank(preds[r])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx, sy = np.cov(v01), np.cov(v10)
    S = sx / m + sy / n
    d = np.array([1, -1])
    var = float(d @ S @ d)
    if var <= 0:
        return float(aucs[0]), float(aucs[1]), 1.0
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    pval = float(2 * stats.norm.sf(abs(z)))
    return float(aucs[0]), float(aucs[1]), pval


def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


# ---------------------------- bootstrap ------------------------------------

def bootstrap_ci(y, p, fn, n_boot=2000, seed=0, stratified=True):
    """Percentile CI for a metric fn(y, p)."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    stats_ = []
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    for _ in range(n_boot):
        if stratified:
            idx = np.concatenate([rng.choice(pos, len(pos), replace=True),
                                  rng.choice(neg, len(neg), replace=True)])
        else:
            idx = rng.choice(len(y), len(y), replace=True)
        stats_.append(fn(y[idx], p[idx]))
    lo, hi = np.percentile(stats_, [2.5, 97.5])
    return float(lo), float(hi)


# ---------------------------- decision curve --------------------------------

def net_benefit(y, p, thresholds=None):
    """Net benefit vs threshold; also treat-all and treat-none lines."""
    y, p = np.asarray(y), np.asarray(p)
    if thresholds is None:
        thresholds = np.arange(0.01, 0.31, 0.005)
    n = len(y)
    prev = y.mean()
    rows = []
    for t in thresholds:
        yhat = p >= t
        tp = ((y == 1) & yhat).sum() / n
        fp = ((y == 0) & yhat).sum() / n
        nb = tp - fp * t / (1 - t)
        nb_all = prev - (1 - prev) * t / (1 - t)
        rows.append({"threshold": float(t), "nb_model": float(nb),
                     "nb_all": float(nb_all), "nb_none": 0.0})
    return rows
