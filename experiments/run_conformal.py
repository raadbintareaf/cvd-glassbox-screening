"""Conformal prediction analysis from saved predictions (no refitting).

Calibration = validation split; evaluation = test split. Reports marginal
split-conformal and Mondrian (Sex, and Sex x AgeBand) coverage / set sizes /
deferral, appending rows and writing tables.

Usage:
  python -m experiments.run_conformal --config configs/default.yaml \
      --tier T1 --model ebm --seed 0 --alpha 0.1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.evaluation.conformal import (mondrian_conformal, split_conformal,
                                      summarize)
from src.utils.common import append_rows, load_config


def age_band(agecat: pd.Series) -> pd.Series:
    # _AGEG5YR ordinal: 1..4 -> 18-39, 5..8 -> 40-59, 9..13 -> 60+
    def f(v):
        if pd.isna(v):
            return None
        v = float(v)
        return "18-39" if v <= 4 else ("40-59" if v <= 8 else "60+")
    return agecat.map(f).astype("object")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--tier", required=True)
    ap.add_argument("--model", default="ebm")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.1)
    a = ap.parse_args()
    cfg = load_config(a.config)
    dataset = cfg.get("dataset", "brfss2022")

    pq = (Path(cfg["results_dir"]) / "predictions" / dataset / a.tier /
          a.model / f"seed{a.seed}.parquet")
    if not pq.exists():
        raise SystemExit(f"{pq} missing — run run_benchmark first.")
    df = pd.read_parquet(pq)
    cal = df[df.split == "val"].reset_index(drop=True)
    tst = df[df.split == "test"].reset_index(drop=True)
    tst_groups = pd.DataFrame({
        "Sex": tst["Sex"].astype("object"),
        "AgeBand": age_band(tst["AgeCategory"]),
    })
    tst_groups["Sex_x_AgeBand"] = (tst_groups.Sex.astype(str) + "|" +
                                   tst_groups.AgeBand.astype(str))
    cal_sexage = (cal["Sex"].astype(str) + "|" +
                  age_band(cal["AgeCategory"]).astype(str))

    out_dir = (Path(cfg["results_dir"]) / "conformal" / dataset / a.tier /
               a.model / f"seed{a.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)

    code_m, _ = split_conformal(cal.y.values, cal.p.values, tst.p.values,
                                alpha=a.alpha)
    tab_m = summarize(code_m, tst.y.values, tst_groups)
    tab_m.insert(0, "method", "marginal")

    code_s, info_s = mondrian_conformal(
        cal.y.values, cal.p.values, cal["Sex"].astype(str),
        tst.p.values, tst_groups["Sex"].astype(str), alpha=a.alpha)
    tab_s = summarize(code_s, tst.y.values, tst_groups)
    tab_s.insert(0, "method", "mondrian_sex")

    code_sa, info_sa = mondrian_conformal(
        cal.y.values, cal.p.values, cal_sexage,
        tst.p.values, tst_groups["Sex_x_AgeBand"], alpha=a.alpha)
    tab_sa = summarize(code_sa, tst.y.values, tst_groups)
    tab_sa.insert(0, "method", "mondrian_sex_age")

    tab = pd.concat([tab_m, tab_s, tab_sa], ignore_index=True)
    tab.to_csv(out_dir / f"coverage_alpha{a.alpha}.csv", index=False)

    results_csv = str(Path(cfg["results_dir"]) / "raw_results.csv")
    append_rows(results_csv, [
        dict(run_id=f"conf-{dataset}-{a.tier}-{a.model}-s{a.seed}",
             stage="conformal", dataset=dataset, tier=a.tier, model=a.model,
             seed=a.seed, split="test",
             subgroup=f"{r.group_var}={r.group}",
             metric=f"{r.method}_{m}", value=getattr(r, m),
             extra=f"alpha={a.alpha}")
        for r in tab.itertuples()
        for m in ("coverage", "avg_set_size", "deferral_rate",
                  "singleton_error")])
    print(tab[tab.group_var.isin(["ALL", "Sex"])].to_string(index=False))


if __name__ == "__main__":
    main()
