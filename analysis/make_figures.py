"""All paper figures from results/ in the unified theme.

  F1 leakage_decay      — test AUROC per model across tiers (seed error bars)
  F2 calibration_dca    — reliability curves + decision curves (T1, seed 0)
  F3 fairness_frontier  — TPR gap vs overall sensitivity across arms
  F4 conformal_coverage — per-group coverage & set size, marginal vs Mondrian
  F5 faithfulness_sweep — explainer agreement vs explanation sample size
  F6 ebm_shapes         — key EBM shape functions, 2022 (with bag SD) vs
                          2023-refit overlay

Usage: python -m analysis.make_figures --config configs/default.yaml
Each figure is skipped gracefully if its inputs are not present yet.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.figure_style import (DOUBLE_COL, MODEL_LABEL, MODEL_ORDER,
                                   PALETTE, SINGLE_COL, TIER_LABEL,
                                   apply_style, save)
from src.utils.common import load_config


def f1_leakage_decay(cfg, rdir):
    f = rdir / "summary.csv"
    if not f.exists():
        return
    s = pd.read_csv(f)
    s = s[(s.stage == "benchmark") & (s.split == "test")
          & (s.subgroup == "ALL") & (s.metric == "auroc")
          & (s.dataset == "brfss2022")]
    tiers = [t for t in ["T0", "T1", "T1_portable", "T2"]
             if t in s.tier.unique()]
    if not len(s) or not tiers:
        return
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.6))
    xs = np.arange(len(tiers))
    for m in [m for m in MODEL_ORDER if m in s.model.unique()]:
        y = [s[(s.tier == t) & (s.model == m)]["mean"].mean()
             for t in tiers]
        e = [s[(s.tier == t) & (s.model == m)]["std"].mean()
             for t in tiers]
        ax.errorbar(xs, y, yerr=e, label=MODEL_LABEL.get(m, m),
                    color=PALETTE.get(m, "#333"), marker="o", capsize=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([TIER_LABEL[t].split(" (")[0] for t in tiers])
    ax.set_ylabel("Test AUROC")
    ax.set_xlabel("Feature tier (decreasing leakage risk \u2192)")
    ax.legend(ncol=2, loc="lower left")
    save(fig, "f1_leakage_decay")


def f2_calibration_dca(cfg, rdir):
    from src.evaluation.metrics import net_benefit
    pdir = rdir / "predictions" / "brfss2022" / "T1"
    if not pdir.exists():
        return
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.7))
    for m in ["logreg", "ebm", "xgboost", "tabicl", "tabpfn_v2"]:
        pq = pdir / m / "seed0.parquet"
        if not pq.exists():
            continue
        df = pd.read_parquet(pq)
        df = df[df.split == "test"]
        bins = np.quantile(df.p, np.linspace(0, 1, 11))
        idx = np.clip(np.digitize(df.p, bins) - 1, 0, 9)
        obs = [df.y.values[idx == b].mean() if (idx == b).any() else np.nan
               for b in range(10)]
        pred = [df.p.values[idx == b].mean() if (idx == b).any() else np.nan
                for b in range(10)]
        axes[0].plot(pred, obs, marker="o", color=PALETTE.get(m),
                     label=MODEL_LABEL.get(m, m))
        nb = pd.DataFrame(net_benefit(df.y.values, df.p.values))
        axes[1].plot(nb.threshold, nb.nb_model, color=PALETTE.get(m),
                     label=MODEL_LABEL.get(m, m))
        if m == "ebm":
            axes[1].plot(nb.threshold, nb.nb_all, color="#000",
                         ls=":", lw=1, label="Treat all")
            axes[1].axhline(0, color="#000", lw=0.8, ls="--",
                            label="Treat none")
    lim = axes[0].get_xlim()
    axes[0].plot(lim, lim, color="#000", lw=0.8, ls="--")
    axes[0].set_xlabel("Predicted probability")
    axes[0].set_ylabel("Observed frequency")
    axes[0].legend()
    axes[1].set_xlabel("Decision threshold")
    axes[1].set_ylabel("Net benefit")
    axes[1].set_ylim(bottom=-0.005)
    axes[1].legend(ncol=2)
    save(fig, "f2_calibration_dca")


def f3_fairness_frontier(cfg, rdir):
    rows = []
    for f in (rdir / "fairness").glob("*/*/*/seed*/frontier.csv"):
        df = pd.read_csv(f)
        df["model"] = f.parts[-3]
        rows.append(df)
    if not rows:
        return
    fr = pd.concat(rows)
    agg = fr.groupby(["model", "arm"]).agg(
        sens=("sensitivity", "mean"), gap=("tpr_gap", "mean"),
        sens_sd=("sensitivity", "std"), gap_sd=("tpr_gap", "std")
        ).reset_index()
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.8))
    markers = {"baseline": "o", "group_thr": "s", "reweigh": "^",
               "repair_half": "D", "repair_zero": "v", "repair_eq": "P"}
    for r in agg.itertuples():
        ax.errorbar(r.gap, r.sens, xerr=r.gap_sd, yerr=r.sens_sd,
                    marker=markers.get(r.arm, "o"),
                    color=PALETTE.get(r.model, "#333"), capsize=2,
                    ls="none")
        ax.annotate(r.arm.replace("_", " "), (r.gap, r.sens), fontsize=6,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("TPR gap across Sex (lower = fairer)")
    ax.set_ylabel("Overall sensitivity")
    handles = [plt.Line2D([], [], color=PALETTE[m], marker="o", ls="none",
                          label=MODEL_LABEL[m])
               for m in agg.model.unique() if m in PALETTE]
    ax.legend(handles=handles, loc="lower right")
    save(fig, "f3_fairness_frontier")


def f4_conformal(cfg, rdir):
    rows = []
    for f in (rdir / "conformal").glob("*/*/*/seed*/coverage_alpha*.csv"):
        df = pd.read_csv(f)
        df["model"] = f.parts[-3]
        rows.append(df)
    if not rows:
        return
    cv = pd.concat(rows)
    cv = cv[(cv.model == "ebm") & (cv.group_var.isin(["Sex", "AgeBand"]))]
    if not len(cv):
        return
    agg = cv.groupby(["method", "group_var", "group"]).agg(
        cov=("coverage", "mean"), cov_sd=("coverage", "std"),
        size=("avg_set_size", "mean")).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.6))
    alpha = load_config("configs/default.yaml").get("conformal_alpha", 0.1)
    groups = agg[["group_var", "group"]].drop_duplicates()
    xs = np.arange(len(groups))
    width = 0.38
    for k, meth in enumerate(["marginal", "mondrian_sex_age"]):
        sub = agg[agg.method == meth].set_index(["group_var", "group"])
        cov = [sub["cov"].get((gv, g), np.nan) for gv, g in groups.values]
        sd = [sub["cov_sd"].get((gv, g), 0) for gv, g in groups.values]
        sz = [sub["size"].get((gv, g), np.nan) for gv, g in groups.values]
        col = PALETTE["ebm"] if k else "#999999"
        axes[0].bar(xs + (k - 0.5) * width, cov, width, yerr=sd,
                    color=col, label=meth.replace("_", " "), capsize=2)
        axes[1].bar(xs + (k - 0.5) * width, sz, width, color=col,
                    label=meth.replace("_", " "))
    axes[0].axhline(1 - alpha, color="#000", ls="--", lw=0.8)
    for ax in axes:
        ax.set_xticks(xs)
        ax.set_xticklabels([g for _, g in groups.values], rotation=45,
                           ha="right")
    axes[0].set_ylabel(f"Coverage (target {1-alpha:.0%})")
    axes[0].set_ylim(0.8, 1.0)
    axes[1].set_ylabel("Average set size")
    axes[0].legend()
    save(fig, "f4_conformal_coverage")


def f5_faithfulness(cfg, rdir):
    fdir = rdir / "faithfulness"
    files = list(fdir.glob("*/*/seed*/agreement_raw.csv"))
    if not files:
        return
    df = pd.concat([pd.read_csv(f) for f in files])
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.6))
    colors = {"xgb_treeshap": PALETTE["xgboost"],
              "ebm_kernelshap": PALETTE["ebm"],
              "xgb_lime": PALETTE["mlp"],
              "fm_kernelshap": PALETTE["tabicl"]}
    for comp, g in df.groupby("comparator"):
        agg = g.groupby("n_explain").kendall_tau.agg(["mean", "std"])
        ax.errorbar(agg.index, agg["mean"], yerr=agg["std"],
                    label=comp.replace("_", " "), marker="o",
                    color=colors.get(comp, "#333"), capsize=2)
    ax.set_xscale("log")
    ax.set_xlabel("Explanation sample size")
    ax.set_ylabel("Kendall $\\tau$ vs. exact EBM importances")
    ax.legend()
    save(fig, "f5_faithfulness_sweep")


def f6_ebm_shapes(cfg, rdir):
    from src.models.registry import EBMAdapter
    d22 = rdir / "models" / "brfss2022" / "T1_portable" / "ebm" / "seed0"
    d23 = rdir / "models" / "brfss2023" / "T1_portable" / "ebm" / "seed0"
    if not (d22 / "model.joblib").exists():
        d22 = rdir / "models" / "brfss2022" / "T1" / "ebm" / "seed0"
        d23 = None
        if not (d22 / "model.joblib").exists():
            return
    ebm22 = EBMAdapter.load(str(d22)).model_
    ebm23 = (EBMAdapter.load(str(d23)).model_
             if d23 and (d23 / "model.joblib").exists() else None)
    wanted = [t for t in ["AgeCategory", "GeneralHealth", "BMI",
                          "SleepHours"] if t in ebm22.term_names_]
    if not wanted:
        wanted = [n for n in ebm22.term_names_ if "&" not in n][:4]
    fig, axes = plt.subplots(1, len(wanted),
                             figsize=(DOUBLE_COL, 2.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, term in zip(axes, wanted):
        i = ebm22.term_names_.index(term)
        ys = ebm22.term_scores_[i]
        sd = ebm22.standard_deviations_[i] \
            if ebm22.standard_deviations_ is not None else None
        core, csd = ys[1:-1], (sd[1:-1] if sd is not None else None)
        xs = np.arange(len(core))
        ax.step(xs, core, where="mid", color=PALETTE["ebm"],
                label="2022")
        if csd is not None:
            ax.fill_between(xs, core - csd, core + csd, step="mid",
                            alpha=0.25, color=PALETTE["ebm"])
        if ebm23 is not None and term in ebm23.term_names_:
            j = ebm23.term_names_.index(term)
            c23 = ebm23.term_scores_[j][1:-1]
            n = min(len(core), len(c23))
            ax.step(np.arange(n), c23[:n], where="mid",
                    color=PALETTE["xgboost"], ls="--", label="2023 refit")
        ax.set_title(term, fontsize=8)
        ax.set_xlabel("Bin")
    axes[0].set_ylabel("Contribution to log-odds")
    axes[0].legend()
    save(fig, "f6_ebm_shapes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    a = ap.parse_args()
    cfg = load_config(a.config)
    rdir = Path(cfg["results_dir"])
    apply_style()
    for fn in (f1_leakage_decay, f2_calibration_dca, f3_fairness_frontier,
               f4_conformal, f5_faithfulness, f6_ebm_shapes):
        try:
            fn(cfg, rdir)
        except Exception as e:  # keep going; report
            print(f"[fig][skip] {fn.__name__}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
