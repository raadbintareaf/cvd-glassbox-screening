"""One visual theme for every figure in the paper (and the graphical
abstract): Okabe-Ito colorblind-safe palette, consistent fonts and sizes,
vector PDF output at journal widths."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# Okabe-Ito
PALETTE = {
    "ebm": "#0072B2",          # blue      — the glass-box protagonist
    "xgboost": "#E69F00",      # orange
    "lightgbm": "#D55E00",     # vermilion
    "mlp": "#CC79A7",          # purple-pink
    "tabpfn_v2": "#009E73",    # green
    "tabicl": "#56B4E9",       # sky
    "catboost": "#F0E442",     # yellow
    "random_forest": "#000000",
    "logreg": "#999999",       # grey
    "logreg_spline": "#666666",
    "neutral": "#000000",
    "accent": "#F0E442",       # yellow (highlights only)
}
MODEL_ORDER = ["logreg", "logreg_spline", "random_forest", "ebm",
               "xgboost", "lightgbm", "catboost", "mlp", "tabpfn_v2",
               "tabicl"]
MODEL_LABEL = {
    "logreg": "LogReg", "logreg_spline": "LogReg (splines)", "ebm": "EBM",
    "xgboost": "XGBoost", "lightgbm": "LightGBM", "mlp": "MLP", "catboost": "CatBoost",
    "random_forest": "Random Forest",
    "tabpfn_v2": "TabPFN v2", "tabicl": "TabICL",
}
TIER_LABEL = {"T0": "T0 (full, incl. post-dx)", "T1": "T1 (screening)",
              "T1_portable": "T1-portable", "T2": "T2 (self-report min.)"}
SINGLE_COL, DOUBLE_COL = 3.35, 6.9  # inches (npj / Nature portfolio)


def apply_style():
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300,
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
        "lines.linewidth": 1.4, "lines.markersize": 4,
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def save(fig, name: str, out_dir="paper/figures"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[fig] {out / name}.pdf")
