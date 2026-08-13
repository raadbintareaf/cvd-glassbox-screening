"""Model adapters with a single interface used by every experiment script.

Adapter contract:
    m = make_model(name, params, seed)
    m.fit(X_tr, y_tr, X_va, y_va)      # DataFrames in the analytic schema
    p = m.predict_proba_pos(X)         # np.ndarray of positive-class probs
    m.save(dir_path) / load(dir_path)  # CPU models only (FMs re-fit cheaply)

Class imbalance is handled by 'balanced' weighting inside each adapter
(sample weights / scale_pos_weight / loss weights) — the pre-registered
primary strategy. SMOTE-NC lives in experiments/run_benchmark.py as an
explicit ablation, never here.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Shared preprocessing
# --------------------------------------------------------------------------

def split_cols(X: pd.DataFrame):
    cat = [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]
    num = [c for c in X.columns if c not in cat]
    return num, cat


def to_codes_matrix(X: pd.DataFrame):
    """Float32 matrix; categoricals as integer codes (NaN preserved).
    Returns (M, cat_idx) for models with native categorical/NaN support."""
    num, cat = split_cols(X)
    cols, cat_idx = [], []
    for j, c in enumerate(X.columns):
        if c in cat:
            cols.append(X[c].cat.codes.replace(-1, np.nan).astype("float32"))
            cat_idx.append(j)
        else:
            cols.append(pd.to_numeric(X[c], errors="coerce").astype("float32"))
    return np.column_stack(cols), cat_idx


class OneHotPipe:
    """Impute(+indicators)+scale numerics; explicit-'Missing' one-hot cats.
    Fit on train only; used by LR / MLP."""

    def fit(self, X: pd.DataFrame):
        from sklearn.preprocessing import StandardScaler
        self.num_, self.cat_ = split_cols(X)
        self.medians_ = X[self.num_].median()
        self.cats_levels_ = {c: list(X[c].cat.categories) + ["__MISSING__"]
                             for c in self.cat_}
        Z = self._numeric_block(X)
        self.scaler_ = StandardScaler().fit(Z)
        return self

    def _numeric_block(self, X):
        Z = X[self.num_].copy()
        ind = Z.isna().astype("float32").add_suffix("__isna")
        Z = Z.fillna(self.medians_)
        return pd.concat([Z.astype("float32"), ind], axis=1).values

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        Zn = self.scaler_.transform(self._numeric_block(X)).astype("float32")
        blocks = [Zn]
        for c in self.cat_:
            s = X[c].astype(object).where(X[c].notna(), "__MISSING__")
            oh = np.zeros((len(X), len(self.cats_levels_[c])), dtype="float32")
            lut = {lv: k for k, lv in enumerate(self.cats_levels_[c])}
            idx = s.map(lambda v: lut.get(v, lut["__MISSING__"])).values
            oh[np.arange(len(X)), idx] = 1.0
            blocks.append(oh)
        return np.concatenate(blocks, axis=1)

    def feature_names(self) -> list[str]:
        names = list(self.num_) + [f"{c}__isna" for c in self.num_]
        for c in self.cat_:
            names += [f"{c}={lv}" for lv in self.cats_levels_[c]]
        return names


def balanced_sample_weight(y: np.ndarray) -> np.ndarray:
    pos = y.mean()
    w = np.where(y == 1, 0.5 / max(pos, 1e-9), 0.5 / max(1 - pos, 1e-9))
    return (w / w.mean()).astype("float64")


# --------------------------------------------------------------------------
# Adapters
# --------------------------------------------------------------------------

class _Base:
    requires_gpu = False

    def __init__(self, params: dict, seed: int):
        self.params, self.seed = dict(params or {}), seed

    def save(self, d: str):
        Path(d).mkdir(parents=True, exist_ok=True)
        joblib.dump(self, Path(d) / "model.joblib")

    @staticmethod
    def load(d: str):
        return joblib.load(Path(d) / "model.joblib")


class EBMAdapter(_Base):
    name = "ebm"

    def fit(self, X, y, Xv=None, yv=None):
        from interpret.glassbox import ExplainableBoostingClassifier
        num, cat = split_cols(X)
        ftypes = ["nominal" if c in cat else "continuous" for c in X.columns]
        p = dict(interactions=self.params.get("interactions", 10),
                 outer_bags=self.params.get("outer_bags", 14),
                 max_bins=self.params.get("max_bins", 256),
                 learning_rate=self.params.get("learning_rate", 0.01),
                 random_state=self.seed, n_jobs=-2)
        self.model_ = ExplainableBoostingClassifier(
            feature_names=list(X.columns), feature_types=ftypes, **p)
        self.model_.fit(X, y, sample_weight=balanced_sample_weight(y))
        return self

    def predict_proba_pos(self, X):
        return self.model_.predict_proba(X)[:, 1]


class LogRegAdapter(_Base):
    name = "logreg"
    spline = False

    def fit(self, X, y, Xv=None, yv=None):
        from sklearn.linear_model import LogisticRegression
        self.pipe_ = OneHotPipe().fit(X)
        Z = self.pipe_.transform(X)
        if self.spline:
            from sklearn.preprocessing import SplineTransformer
            n_num = len(self.pipe_.num_)
            self.spl_ = SplineTransformer(
                n_knots=self.params.get("n_knots", 5), degree=3,
                include_bias=False).fit(Z[:, :n_num])
            Z = np.hstack([self.spl_.transform(Z[:, :n_num]), Z[:, n_num:]])
        self.model_ = LogisticRegression(
            C=self.params.get("C", 1.0), max_iter=2000,
            class_weight="balanced", solver="lbfgs")
        self.model_.fit(Z, y)
        return self

    def predict_proba_pos(self, X):
        Z = self.pipe_.transform(X)
        if self.spline:
            n_num = len(self.pipe_.num_)
            Z = np.hstack([self.spl_.transform(Z[:, :n_num]), Z[:, n_num:]])
        return self.model_.predict_proba(Z)[:, 1]


class LogRegSplineAdapter(LogRegAdapter):
    name = "logreg_spline"
    spline = True


class XGBAdapter(_Base):
    name = "xgboost"

    def fit(self, X, y, Xv=None, yv=None):
        import xgboost as xgb
        spw = float((y == 0).sum() / max((y == 1).sum(), 1))
        p = dict(n_estimators=self.params.get("n_estimators", 600),
                 learning_rate=self.params.get("learning_rate", 0.05),
                 max_depth=self.params.get("max_depth", 6),
                 min_child_weight=self.params.get("min_child_weight", 5),
                 subsample=self.params.get("subsample", 0.8),
                 colsample_bytree=self.params.get("colsample_bytree", 0.8),
                 reg_lambda=self.params.get("reg_lambda", 1.0),
                 scale_pos_weight=spw, tree_method="hist",
                 enable_categorical=True, random_state=self.seed,
                 n_jobs=-2, eval_metric="auc",
                 early_stopping_rounds=50 if Xv is not None else None)
        self.model_ = xgb.XGBClassifier(**{k: v for k, v in p.items()
                                           if v is not None})
        Xc, Xvc = X.copy(), (Xv.copy() if Xv is not None else None)
        if Xv is not None:
            self.model_.fit(Xc, y, eval_set=[(Xvc, yv)], verbose=False)
        else:
            self.model_.fit(Xc, y, verbose=False)
        return self

    def predict_proba_pos(self, X):
        return self.model_.predict_proba(X)[:, 1]


class LGBMAdapter(_Base):
    name = "lightgbm"

    def fit(self, X, y, Xv=None, yv=None):
        import lightgbm as lgb
        p = dict(n_estimators=self.params.get("n_estimators", 800),
                 learning_rate=self.params.get("learning_rate", 0.05),
                 num_leaves=self.params.get("num_leaves", 63),
                 min_child_samples=self.params.get("min_child_samples", 50),
                 subsample=self.params.get("subsample", 0.8),
                 colsample_bytree=self.params.get("colsample_bytree", 0.8),
                 reg_lambda=self.params.get("reg_lambda", 1.0),
                 class_weight="balanced", random_state=self.seed, n_jobs=-2,
                 verbosity=-1)
        self.model_ = lgb.LGBMClassifier(**p)
        cb = ([lgb.early_stopping(50, verbose=False)] if Xv is not None
              else None)
        self.model_.fit(X, y, eval_set=([(Xv, yv)] if Xv is not None else
                        None), eval_metric="auc", callbacks=cb,
                        categorical_feature=[c for c in X.columns if
                                             isinstance(X[c].dtype,
                                                        pd.CategoricalDtype)])
        return self

    def predict_proba_pos(self, X):
        return self.model_.predict_proba(X)[:, 1]


class MLPAdapter(_Base):
    name = "mlp"
    requires_gpu = False  # runs on CPU too, just slower

    def fit(self, X, y, Xv, yv):
        import torch
        import torch.nn as nn
        torch.manual_seed(self.seed)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe_ = OneHotPipe().fit(X)
        Z = torch.tensor(self.pipe_.transform(X), device=dev)
        Zv = torch.tensor(self.pipe_.transform(Xv), device=dev)
        t = torch.tensor(y.astype("float32"), device=dev)
        tv = yv.astype("float32")
        width = self.params.get("width", 256)
        depth = self.params.get("depth", 3)
        drop = self.params.get("dropout", 0.2)
        layers, d_in = [], Z.shape[1]
        for _ in range(depth):
            layers += [nn.Linear(d_in, width), nn.ReLU(), nn.Dropout(drop)]
            d_in = width
        layers += [nn.Linear(d_in, 1)]
        self.net_ = nn.Sequential(*layers).to(dev)
        pos_w = torch.tensor([(y == 0).sum() / max((y == 1).sum(), 1)],
                             device=dev)
        lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        opt = torch.optim.AdamW(self.net_.parameters(),
                                lr=self.params.get("lr", 1e-3),
                                weight_decay=self.params.get("wd", 1e-4))
        from sklearn.metrics import roc_auc_score
        best, best_state, patience, bad = -1, None, 8, 0
        n, bs = len(Z), self.params.get("batch_size", 4096)
        for epoch in range(self.params.get("max_epochs", 60)):
            self.net_.train()
            perm = torch.randperm(n, device=dev)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                opt.zero_grad()
                loss = lossf(self.net_(Z[idx]).squeeze(1), t[idx])
                loss.backward()
                opt.step()
            self.net_.eval()
            with torch.no_grad():
                pv = torch.sigmoid(self.net_(Zv).squeeze(1)).cpu().numpy()
            auc = roc_auc_score(tv, pv)
            if auc > best + 1e-4:
                best, bad = auc, 0
                best_state = {k: v.detach().clone()
                              for k, v in self.net_.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        self.net_.load_state_dict(best_state)
        self.device_ = dev
        return self

    def predict_proba_pos(self, X):
        import torch
        Z = torch.tensor(self.pipe_.transform(X), device=self.device_)
        self.net_.eval()
        with torch.no_grad():
            out = []
            for i in range(0, len(Z), 65536):
                out.append(torch.sigmoid(
                    self.net_(Z[i:i + 65536]).squeeze(1)).cpu().numpy())
        return np.concatenate(out)

    def save(self, d):
        import torch
        Path(d).mkdir(parents=True, exist_ok=True)
        torch.save({"state": self.net_.state_dict(),
                    "params": self.params, "seed": self.seed}, 
                   Path(d) / "mlp.pt")
        joblib.dump(self.pipe_, Path(d) / "pipe.joblib")


class TabPFNAdapter(_Base):
    """TabPFN v2 (Hollmann et al., Nature 2025) with a stratified
    subsampled-context ensemble to exceed the native 10k-sample context."""
    name = "tabpfn_v2"
    requires_gpu = True

    def fit(self, X, y, Xv=None, yv=None):
        from tabpfn import TabPFNClassifier
        M, cat_idx = to_codes_matrix(X)
        self._cat_idx = cat_idx
        n_ctx = int(self.params.get("context_size", 10000))
        k = int(self.params.get("n_contexts", 8))
        rng = np.random.default_rng(self.seed)
        self.members_ = []
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        n_pos = max(1, int(round(n_ctx * y.mean())))
        for _ in range(k):
            idx = np.concatenate([
                rng.choice(pos, size=min(n_pos, len(pos)), replace=False),
                rng.choice(neg, size=min(n_ctx - n_pos, len(neg)),
                           replace=False)])
            clf = TabPFNClassifier(
                categorical_features_indices=cat_idx or None,
                random_state=self.seed)
            clf.fit(M[idx], y[idx])
            self.members_.append(clf)
        return self

    def predict_proba_pos(self, X):
        M, _ = to_codes_matrix(X)
        preds = np.zeros(len(M))
        for clf in self.members_:
            preds += clf.predict_proba(M)[:, 1]
        return preds / len(self.members_)

    def save(self, d):  # FMs are cheap to refit; persist config only
        Path(d).mkdir(parents=True, exist_ok=True)
        joblib.dump({"params": self.params, "seed": self.seed},
                    Path(d) / "fm_config.joblib")


class TabICLAdapter(_Base):
    """TabICL / TabICLv2 (Qu et al., ICML 2025; 2026): full-training-set
    in-context learning — the flagship foundation-model arm."""
    name = "tabicl"
    requires_gpu = True

    def fit(self, X, y, Xv=None, yv=None):
        from tabicl import TabICLClassifier
        self.model_ = TabICLClassifier(random_state=self.seed)
        self.model_.fit(X, y)  # accepts DataFrame; auto dtype handling
        return self

    def predict_proba_pos(self, X):
        out = []
        bs = int(self.params.get("predict_batch", 20000))
        for i in range(0, len(X), bs):
            out.append(self.model_.predict_proba(X.iloc[i:i + bs])[:, 1])
        return np.concatenate(out)

    def save(self, d):
        Path(d).mkdir(parents=True, exist_ok=True)
        joblib.dump({"params": self.params, "seed": self.seed},
                    Path(d) / "fm_config.joblib")


REGISTRY = {c.name: c for c in
            [EBMAdapter, LogRegAdapter, LogRegSplineAdapter, XGBAdapter,
             LGBMAdapter, MLPAdapter, TabPFNAdapter, TabICLAdapter]}


def make_model(name: str, params: dict, seed: int):
    if name not in REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Known: {sorted(REGISTRY)}")
    return REGISTRY[name](params, seed)
