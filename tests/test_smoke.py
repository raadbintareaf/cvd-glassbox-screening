"""Synthetic end-to-end smoke test (<2 min, CPU, no data download).

Run:  python -m tests.test_smoke      (or pytest tests/test_smoke.py)
Covers: adapters (CPU set), metrics, thresholding, DeLong, fairness audit +
repair, conformal, faithfulness agreement — the full integration surface.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    age = rng.integers(1, 14, n).astype(float)
    bmi = rng.normal(28, 6, n).clip(15, 60)
    sex = rng.choice(["Male", "Female"], n)
    smoke = rng.choice(["Never", "Former", "Current daily"], n,
                       p=[0.55, 0.3, 0.15])
    genhlth = rng.integers(1, 6, n).astype(float)
    lin = (-4.2 + 0.28 * age + 0.05 * (bmi - 28) + 0.7 * (sex == "Male")
           + 0.9 * (smoke == "Current daily") + 0.35 * (genhlth - 3))
    y = (rng.random(n) < 1 / (1 + np.exp(-lin))).astype(int)
    X = pd.DataFrame({
        "AgeCategory": age, "BMI": bmi,
        "Sex": pd.Categorical(sex),
        "SmokerStatus": pd.Categorical(smoke),
        "GeneralHealth": genhlth,
    })
    X.loc[rng.random(n) < 0.05, "BMI"] = np.nan  # exercise NaN paths
    return X, y


def run():
    from src.evaluation.conformal import mondrian_conformal, summarize
    from src.evaluation.fairness import subgroup_audit
    from src.evaluation.faithfulness import agreement, ebm_global_importance
    from src.evaluation.metrics import (all_probability_metrics, delong_test,
                                        threshold_for)
    from src.models.registry import make_model
    from src.repair.ebm_repair import neutralize_terms
    from src.utils.common import seed_everything, stratified_splits

    seed_everything(0)
    X, y = make_synthetic()
    idx = stratified_splits(y, 0)
    Xtr, ytr = X.iloc[idx["train"]], y[idx["train"]]
    Xva, yva = X.iloc[idx["val"]], y[idx["val"]]
    Xte, yte = X.iloc[idx["test"]], y[idx["test"]]

    probs = {}
    for name in ["logreg", "logreg_spline", "ebm", "xgboost", "lightgbm"]:
        m = make_model(name, {}, 0).fit(Xtr, ytr, Xva, yva)
        p = m.predict_proba_pos(Xte)
        met = all_probability_metrics(yte, p)
        assert met["auroc"] > 0.62, f"{name} AUROC too low: {met['auroc']}"
        probs[name] = p
        if name == "ebm":
            ebm = m
        print(f"  [ok] {name:14s} auroc={met['auroc']:.3f} "
              f"cal_slope={met['cal_slope']:.2f}")

    a1, a2, pde = delong_test(yte, probs["ebm"], probs["logreg"])
    assert 0 <= pde <= 1
    thr = threshold_for(yva, ebm.predict_proba_pos(Xva), "sens_at", 0.85)
    G = X.iloc[idx["test"]][["Sex"]].reset_index(drop=True)
    aud = subgroup_audit(yte, probs["ebm"], G, thr)
    assert set(aud.group) == {"Male", "Female"}

    edited, log = neutralize_terms(ebm.model_, ["Sex"])
    assert any("Sex" in t["term"] for t in log["terms"])
    p_ed = edited.predict_proba(Xte)[:, 1]
    assert abs(p_ed.mean() - probs["ebm"].mean()) < 0.5

    code, _ = mondrian_conformal(
        yva, ebm.predict_proba_pos(Xva),
        X.iloc[idx["val"]]["Sex"].astype(str),
        probs["ebm"], G["Sex"].astype(str), alpha=0.1)
    tab = summarize(code, yte, G)
    cov = tab[tab.group == "ALL"].coverage.iloc[0]
    assert cov > 0.85, f"conformal coverage {cov}"
    print(f"  [ok] conformal ALL coverage={cov:.3f}")

    imp = ebm_global_importance(ebm.model_, list(X.columns))
    ag = agreement(imp, imp.sample(frac=1.0, random_state=0), k=3)
    assert ag["kendall_tau"] > 0.99
    print("  [ok] fairness / repair / conformal / faithfulness plumbing")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    run()
