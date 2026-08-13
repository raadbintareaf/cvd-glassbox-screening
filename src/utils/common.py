"""Seeding, config, deterministic splits, and append-only results I/O."""
from __future__ import annotations

import datetime as _dt
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

RESULTS_CSV_COLUMNS = [
    "timestamp", "run_id", "stage", "dataset", "tier", "model", "seed",
    "split", "subgroup", "metric", "value", "extra",
]


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    base = cfg.pop("inherit", None)
    if base:
        parent = load_config(str(Path(path).parent / base))
        parent.update(cfg)
        cfg = parent
    return cfg


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def stratified_splits(y: np.ndarray, seed: int, fracs=(0.6, 0.2, 0.2)):
    """Deterministic stratified train/val/test index arrays."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    parts = {"train": [], "val": [], "test": []}
    for cls in np.unique(y):
        cls_idx = idx[y == cls]
        rng.shuffle(cls_idx)
        n = len(cls_idx)
        n_tr, n_va = int(round(fracs[0] * n)), int(round(fracs[1] * n))
        parts["train"].append(cls_idx[:n_tr])
        parts["val"].append(cls_idx[n_tr:n_tr + n_va])
        parts["test"].append(cls_idx[n_tr + n_va:])
    return {k: np.sort(np.concatenate(v)) for k, v in parts.items()}


def append_rows(results_csv: str, rows: list[dict]) -> None:
    """Append result rows (long format); create file with header if absent."""
    path = Path(results_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    for c in RESULTS_CSV_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    df = df[RESULTS_CSV_COLUMNS]
    df["timestamp"] = _dt.datetime.now().isoformat(timespec="seconds")
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False)


def completed_keys(results_csv: str) -> set[tuple]:
    """(stage, dataset, tier, model, seed) keys already present — for resume."""
    path = Path(results_csv)
    if not path.exists():
        return set()
    df = pd.read_csv(path, usecols=["stage", "dataset", "tier", "model",
                                    "seed"], dtype=str)
    return set(map(tuple, df.drop_duplicates().values))


def save_json(obj, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)
