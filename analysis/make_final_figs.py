"""Final-style f2 (calibration + decision curves) and f6 (EBM shape gallery
with SleepHours U-curve and 2023 replication overlay)."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression

PAL = {"logreg": "#999999", "ebm": "#0072B2", "catboost": "#8B6F00",
       "tabicl": "#56B4E9"}
LBL = {"logreg": "LogReg", "ebm": "EBM", "catboost": "CatBoost",
       "tabicl": "TabICL"}
plt.rcParams.update({
    "font.size": 7.5, "axes.labelsize": 8, "legend.fontsize": 6.4,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": .25, "grid.linewidth": .5, "legend.frameon": False,
    "pdf.fonttype": 42, "savefig.dpi": 300})


def save(fig, name):
    fig.savefig(f"paper/figures/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"paper/figures/{name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("[fig]", name)


def load_preds(model):
    pq = Path(f"results/predictions/brfss2022/T1/{model}/seed0.parquet")
    df = pd.read_parquet(pq)
    val, tst = df[df.split == "val"], df[df.split == "test"]
    iso = IsotonicRegression(out_of_bounds="clip").fit(val.p, val.y)
    return tst.y.values, tst.p.values, iso.predict(tst.p)


def reliability(y, p, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    xs, ys = [], []
    for b in range(bins):
        m = idx == b
        if m.sum() >= 50:
            xs.append(p[m].mean())
            ys.append(y[m].mean())
    return np.array(xs), np.array(ys)


def net_benefit(y, p, ts):
    n = len(y)
    out = []
    for t in ts:
        pred = p >= t
        tp = np.sum(pred & (y == 1))
        fp = np.sum(pred & (y == 0))
        out.append(tp / n - fp / n * t / (1 - t))
    return np.array(out)


# ------------------------------ f2 ----------------------------------
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
ax = axes[0]
ax.plot([0, 1], [0, 1], color="#bbb", lw=.8, ls=":")
data = {m: load_preds(m) for m in ["ebm", "catboost", "tabicl"]}
for m in ["ebm", "catboost"]:
    y, p, _ = data[m]
    xs, ys = reliability(y, p)
    ax.plot(xs, ys, marker="o", ms=2.2, lw=1.1, ls="--", color=PAL[m],
            label=f"{LBL[m]} raw (class-weighted)")
y, p, _ = data["tabicl"]
xs, ys = reliability(y, p)
ax.plot(xs, ys, marker="o", ms=2.2, lw=1.3, color=PAL["tabicl"],
        label="TabICL raw (native)")
y, _, piso = data["ebm"]
xs, ys = reliability(y, piso)
ax.plot(xs, ys, marker="s", ms=2.2, lw=1.3, color=PAL["ebm"],
        label="EBM + isotonic")
ax.set_xlabel("Predicted probability")
ax.set_ylabel("Observed frequency")
ax.set_title("a  Class weighting breaks calibration; isotonic repairs it",
             loc="left", fontsize=8, weight="bold")
ax.legend(loc="upper left", fontsize=5.8)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

ax = axes[1]
ts = np.linspace(0.01, 0.30, 60)
y0 = data["ebm"][0]
prev = y0.mean()
ax.plot(ts, prev - (1 - prev) * ts / (1 - ts), color="#bbb", lw=.9, ls="--",
        label="treat all")
ax.axhline(0, color="#bbb", lw=.9, ls=":", label="treat none")
for m in ["ebm", "catboost", "tabicl"]:
    y, _, piso = data[m]
    ax.plot(ts, net_benefit(y, piso, ts), color=PAL[m], lw=1.3, label=LBL[m])
ax.set_xlabel("Decision threshold")
ax.set_ylabel("Net benefit")
ax.set_ylim(-0.005, prev * 1.15)
ax.set_title("b  Decision curves after recalibration (T1, seed 0)",
             loc="left", fontsize=8, weight="bold")
ax.legend(loc="upper right", fontsize=5.8, ncol=1)
save(fig, "f2_calibration_dca")

# ------------------------------ f6 ----------------------------------
def get_ebm(path):
    obj = joblib.load(path)
    return getattr(obj, "model_", obj)


ebm22 = get_ebm("results/models/brfss2022/T1/ebm/seed0/model.joblib")
cand = sorted(Path("results/temporal").rglob("*ebm*2023*/model.joblib")) + \
       sorted(Path("results/temporal").rglob("ebm_2023*/*.joblib")) + \
       sorted(Path("results/temporal").rglob("*2023*ebm*.joblib"))
ebm23 = get_ebm(cand[0]) if cand else None
print("2023 refit found:", bool(ebm23))

g22 = ebm22.explain_global()
names22 = list(ebm22.term_names_)
g23, names23 = (ebm23.explain_global(), list(ebm23.term_names_)) if ebm23 \
    else (None, [])

PANELS = [("SleepHours", "Sleep duration (h)"),
          ("AgeCategory", "Age band"),
          ("BMI", "Body-mass index"),
          ("GeneralHealth", "Self-rated general health")]
fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.1))
for ax, (feat, xlabel) in zip(axes, PANELS):
    i = names22.index(feat)
    d = g22.data(i)
    edges = np.asarray(d["names"], dtype=float) if not isinstance(
        d["names"][0], str) else np.arange(len(d["scores"]) + 1)
    sc = np.asarray(d["scores"], dtype=float)
    lo = np.asarray(d.get("lower_bounds", sc), dtype=float)
    hi = np.asarray(d.get("upper_bounds", sc), dtype=float)
    ax.stairs(sc, edges, color=PAL["ebm"], lw=1.4, label="2022",
              baseline=None)
    ax.stairs(hi, edges, baseline=lo, fill=True, color=PAL["ebm"],
              alpha=.15, lw=0)
    if ebm23 is not None and feat in names23:
        d3 = g23.data(names23.index(feat))
        e3 = np.asarray(d3["names"], dtype=float) if not isinstance(
            d3["names"][0], str) else np.arange(len(d3["scores"]) + 1)
        ax.stairs(np.asarray(d3["scores"], dtype=float), e3,
                  color="#009E73", lw=1.2, ls="--", label="2023 refit",
                  baseline=None)
    ax.axhline(0, color="#bbb", lw=.6)
    ax.set_xlabel(xlabel)
    if feat == "SleepHours":
        ax.set_xlim(3, 11)
        ax.annotate("U-shaped risk:\nshort & long sleep",
                    xy=(0.5, 0.92), xycoords="axes fraction", ha="center",
                    fontsize=5.8, color="#333")
    if feat == "BMI":
        ax.set_xlim(15, 55)
axes[0].set_ylabel("Contribution to log-odds")
axes[1].legend(loc="upper left", fontsize=5.8)
fig.suptitle("", y=1.02)
save(fig, "f6_ebm_shapes")
