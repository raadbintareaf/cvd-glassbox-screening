"""Optuna hyperparameter search on the VALIDATION split only (seed 0 split),
per (model, tier). Writes results/tuned/<model>__<tier>.yaml, which
run_benchmark then picks up for all seeds.

Usage:
  python -m experiments.run_tuning --config configs/default.yaml \
      --tier T1 --model xgboost
"""
from __future__ import annotations

import argparse
from pathlib import Path

import optuna
import yaml

from src.data.variable_map import OUTCOME_LABEL, tier_labels
from src.models.registry import make_model
from src.utils.common import load_config, seed_everything, stratified_splits
from experiments.run_benchmark import load_analytic


def space(trial, model):
    if model == "xgboost":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 1500),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2,
                                              log=True),
            max_depth=trial.suggest_int("max_depth", 3, 9),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 20),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5,
                                                 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 0.1, 10, log=True))
    if model == "lightgbm":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 300, 2000),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2,
                                              log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 10,
                                                200, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5,
                                                 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 0.1, 10, log=True))
    if model == "ebm":
        return dict(
            max_bins=trial.suggest_categorical("max_bins", [128, 256, 512]),
            learning_rate=trial.suggest_float("learning_rate", 0.005, 0.05,
                                              log=True),
            interactions=trial.suggest_categorical("interactions",
                                                   [0, 10, 20]),
            outer_bags=14)
    if model == "mlp":
        return dict(width=trial.suggest_categorical("width",
                                                    [128, 256, 512]),
                    depth=trial.suggest_int("depth", 2, 4),
                    dropout=trial.suggest_float("dropout", 0.0, 0.4),
                    lr=trial.suggest_float("lr", 3e-4, 3e-3, log=True),
                    wd=trial.suggest_float("wd", 1e-6, 1e-3, log=True))
    if model == "catboost":
        return dict(
            iterations=trial.suggest_int("iterations", 300, 1500),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2,
                                              log=True),
            depth=trial.suggest_int("depth", 4, 9),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1, 10, log=True))
    if model == "random_forest":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 300, 800),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 50,
                                               log=True),
            max_features=trial.suggest_categorical("max_features",
                                                   ["sqrt", 0.3, 0.5]))
    if model in ("logreg", "logreg_spline"):
        return dict(C=trial.suggest_float("C", 1e-3, 100, log=True))
    return {}  # foundation models: no tuning (defaults; that's the point)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--tier", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-trials", type=int, default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    n_trials = a.n_trials or cfg.get("tuning_trials", {}).get(a.model, 25)
    if n_trials == 0:
        print(f"[tuning] {a.model}: 0 trials configured — using defaults")
        return
    seed_everything(0)
    df = load_analytic(cfg, cfg.get("dataset", "brfss2022"))
    feats = tier_labels(a.tier, drop_stroke=cfg.get("drop_stroke", False))
    X, y = df[feats], df[OUTCOME_LABEL].values.astype(int)
    idx = stratified_splits(y, 0, tuple(cfg["split_fracs"]))
    Xtr, ytr = X.iloc[idx["train"]], y[idx["train"]]
    Xva, yva = X.iloc[idx["val"]], y[idx["val"]]

    from sklearn.metrics import roc_auc_score

    def objective(trial):
        m = make_model(a.model, space(trial, a.model), 0)
        m.fit(Xtr, ytr, Xva, yva)
        return roc_auc_score(yva, m.predict_proba_pos(Xva))

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    out = Path(cfg["results_dir"]) / "tuned"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{a.model}__{a.tier}.yaml", "w") as f:
        yaml.safe_dump(study.best_params, f)
    print(f"[tuning] {a.model} x {a.tier}: best val AUROC="
          f"{study.best_value:.4f}; params -> {out}")


if __name__ == "__main__":
    main()
