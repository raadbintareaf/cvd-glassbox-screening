"""Build the analytic dataset from a raw CDC BRFSS LLCP XPT file.

Usage:
    python -m src.data.build_dataset --year 2022 --xpt data/raw/LLCP2022.XPT \
        --out data/analytic
    python -m src.data.build_dataset --year 2023 --xpt data/raw/LLCP2023.XPT \
        --out data/analytic

Reads ONLY the mapped columns (memory-light), applies codebook recodes,
sets pandas dtypes ('category' for categoricals, float for numeric/ordinal,
Int8 for binaries with NaN), and writes:
    <out>/brfss_<year>_analytic.parquet
    <out>/brfss_<year>_cohort_log.json   (row accounting for the paper's flow diagram)

Cohort rule (pre-registered): keep rows with an OBSERVED outcome (CVDINFR4 in
{1,2}); predictors keep NaN (native-missing primary analysis). Complete-case
subset membership is stored as a boolean column `complete_case` for the
sensitivity analysis — no rows are dropped for predictor missingness here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .variable_map import (DESIGN_LABELS, OUTCOME_LABEL, STATE_FIPS,
                           spec_for_year)


def _read_raw(xpt_path: str, usecols: list[str]) -> pd.DataFrame:
    try:
        import pyreadstat
        df, _ = pyreadstat.read_xport(xpt_path, usecols=usecols,
                                      encoding="LATIN1")
        return df
    except ImportError:
        # Fallback: pandas SAS reader in chunks (slower, no usecols support)
        chunks = []
        for ch in pd.read_sas(xpt_path, format="xport", chunksize=100_000,
                              encoding="latin-1"):
            chunks.append(ch[[c for c in usecols if c in ch.columns]])
        return pd.concat(chunks, ignore_index=True)


def _recode(series: pd.Series, rule: dict) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype("float64")
    for na_val in rule.get("na", []):
        s = s.mask(s == na_val)
    vmap = rule.get("map")
    if vmap:
        if all(isinstance(v, (int, float)) for v in vmap.values()):
            s = s.replace(vmap)
        else:  # value -> string label => categorical
            out = s.map(vmap)
            return out.astype("category")
    if rule.get("scale"):
        s = s * rule["scale"]
    if rule.get("kind") == "binary":
        return s.astype("Int8")
    return s


def build(year: int, xpt_path: str, out_dir: str) -> Path:
    spec = spec_for_year(year)
    xpt = Path(xpt_path)
    if not xpt.exists():
        raise SystemExit(
            f"[build_dataset] {xpt} not found.\n"
            f"Download it manually from "
            f"https://www.cdc.gov/brfss/annual_data/{year}/files/"
            f"LLCP{year}XPT.zip and unzip into data/raw/ "
            f"(note: the zip's inner filename has a trailing space — rename it)."
        )
    raw = _read_raw(str(xpt), usecols=list(spec.keys()))
    n_raw = len(raw)

    df = pd.DataFrame(index=raw.index)
    for raw_name, rule in spec.items():
        label = rule.get("label", DESIGN_LABELS.get(raw_name, raw_name))
        col = _recode(raw[raw_name], rule)
        if raw_name == "_STATE":
            col = pd.to_numeric(raw[raw_name], errors="coerce").map(
                STATE_FIPS).astype("category")
        df[label] = col

    # Cohort rule: observed outcome only
    y = df[OUTCOME_LABEL]
    keep = y.notna()
    df = df.loc[keep].reset_index(drop=True)
    df[OUTCOME_LABEL] = df[OUTCOME_LABEL].astype("int8")

    pred_cols = [c for c in df.columns
                 if c not in {OUTCOME_LABEL, *DESIGN_LABELS.values()}]
    df["complete_case"] = df[pred_cols].notna().all(axis=1)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pq = out / f"brfss_{year}_analytic.parquet"
    df.to_parquet(pq, index=False)

    log = {
        "year": year,
        "n_raw_records": int(n_raw),
        "n_outcome_observed": int(len(df)),
        "n_outcome_missing_dropped": int(n_raw - len(df)),
        "n_positive": int(df[OUTCOME_LABEL].sum()),
        "prevalence": float(df[OUTCOME_LABEL].mean()),
        "n_complete_case": int(df["complete_case"].sum()),
        "predictor_missing_fraction": {
            c: float(df[c].isna().mean()) for c in pred_cols},
        "columns": {c: str(df[c].dtype) for c in df.columns},
    }
    with open(out / f"brfss_{year}_cohort_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"[build_dataset] {year}: raw={n_raw:,} -> outcome-observed="
          f"{len(df):,} (prev={log['prevalence']:.4f}); complete-case="
          f"{log['n_complete_case']:,}; wrote {pq}")
    return pq


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True, choices=[2022, 2023])
    ap.add_argument("--xpt", required=True)
    ap.add_argument("--out", default="data/analytic")
    a = ap.parse_args()
    build(a.year, a.xpt, a.out)
