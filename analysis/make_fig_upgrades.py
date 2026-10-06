"""Extended Fig. 7 shape gallery + Supplementary Fig. 4 interaction net."""
from pathlib import Path
import joblib
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

PAL = {"ebm": "#0072B2", "y23": "#009E73", "sexhl": "#D55E00"}
plt.rcParams.update({
    "font.size": 7.5, "axes.labelsize": 7.2, "legend.fontsize": 6.0,
    "xtick.labelsize": 6.2, "ytick.labelsize": 6.2,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": .25, "grid.linewidth": .5,
    "legend.frameon": False, "pdf.fonttype": 42, "savefig.dpi": 300})


def save(fig, name):
    fig.savefig(f"paper/figures/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"paper/figures/{name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("[fig]", name)


def get_ebm(path):
    obj = joblib.load(path)
    return getattr(obj, "model_", obj)


ebm22 = get_ebm("results/models/brfss2022/T1/ebm/seed0/model.joblib")
g22, names22 = ebm22.explain_global(), list(ebm22.term_names_)

# ---- 2023 portable refit (guaranteed) --------------------------------
cand = sorted(Path("results/temporal").rglob("*ebm*2023*/model.joblib")) + \
       sorted(Path("results/temporal").rglob("*2023*ebm*.joblib"))
if cand:
    ebm23 = get_ebm(cand[0])
else:
    print("refitting EBM on 2023 (fallback)...")
    from src.utils.common import load_config, seed_everything, stratified_splits
    from src.data.variable_map import OUTCOME_LABEL, tier_labels
    from src.models.registry import make_model
    from experiments.run_benchmark import load_analytic, model_params
    cfg = load_config("configs/default.yaml")
    seed_everything(0)
    d23 = load_analytic(cfg, "brfss2023")
    feats = tier_labels("T1_portable")
    X, y = d23[feats], d23[OUTCOME_LABEL].values.astype(int)
    idx = stratified_splits(y, 0, tuple(cfg["split_fracs"]))
    m = make_model("ebm", model_params(cfg, "ebm", "T1_portable"), 0)
    m.fit(X.iloc[idx["train"]], y[idx["train"]],
          X.iloc[idx["val"]], y[idx["val"]])
    ebm23 = m.model_
g23, names23 = ebm23.explain_global(), list(ebm23.term_names_)

PANELS = [
    ("SleepHours", "Sleep duration (h) — 2022 only"),
    ("AgeCategory", "Age band"),
    ("BMI", "Body-mass index"),
    ("GeneralHealth", "Self-rated general health"),
    ("SmokerStatus", "Smoking status"),
    ("PhysicalHealthDays", "Poor physical-health days (30d)"),
    ("LastCheckupTime", "Time since last checkup"),
    ("HadStroke", "Prior stroke"),
]
TICKS = {
    "AgeCategory": ([1, 5, 9, 13], ["18-24", "40-44", "60-64", "80+"]),
    "GeneralHealth": ([1, 2, 3, 4, 5], ["Exc", "VG", "G", "F", "Poor"]),
    "LastCheckupTime": ([1, 2, 3, 4], ["<1y", "<2y", "<5y", "5y+"]),
    "HadStroke": ([0.25, 0.75], ["No", "Yes"]),
}

fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.0))
for ax, (feat, xlabel) in zip(axes.ravel(), PANELS):
    d = g22.data(names22.index(feat))
    cat = isinstance(d["names"][0], str)
    sc = np.asarray(d["scores"], dtype=float)
    lo = np.asarray(d.get("lower_bounds", sc), dtype=float)
    hi = np.asarray(d.get("upper_bounds", sc), dtype=float)
    if cat:
        x = np.arange(len(sc))
        ax.bar(x, sc, .62, color=PAL["ebm"], alpha=.85,
               yerr=np.vstack([sc - lo, hi - sc]), error_kw=dict(lw=.7),
               label="2022")
        labels = [str(n)[:9] for n in d["names"]]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=5.2)
        if feat in names23:
            d3 = g23.data(names23.index(feat))
            m3 = {str(n): s for n, s in zip(d3["names"], d3["scores"])}
            y3 = [m3.get(str(n), np.nan) for n in d["names"]]
            ax.plot(x, y3, "D", ms=3.4, color=PAL["y23"], mfc="none",
                    mew=1.2, label="2023 refit")
    else:
        edges = np.asarray(d["names"], dtype=float)
        ax.stairs(sc, edges, color=PAL["ebm"], lw=1.3, baseline=None,
                  label="2022")
        ax.stairs(hi, edges, baseline=lo, fill=True, color=PAL["ebm"],
                  alpha=.15, lw=0)
        if feat in names23:
            d3 = g23.data(names23.index(feat))
            ax.stairs(np.asarray(d3["scores"], dtype=float),
                      np.asarray(d3["names"], dtype=float),
                      color=PAL["y23"], lw=1.1, ls="--", baseline=None,
                      label="2023 refit")
    ax.axhline(0, color="#bbb", lw=.6)
    ax.set_xlabel(xlabel)
    if feat in TICKS:
        ax.set_xticks(TICKS[feat][0])
        ax.set_xticklabels(TICKS[feat][1], fontsize=5.6)
    if feat == "SleepHours":
        ax.set_xlim(3, 11)
        ax.annotate("U-shaped risk", xy=(0.5, 0.9),
                    xycoords="axes fraction", ha="center",
                    fontsize=5.8, color="#333")
    if feat == "BMI":
        ax.set_xlim(15, 55)
    if feat == "PhysicalHealthDays":
        ax.set_xlim(0, 30)
for ax in axes[:, 0]:
    ax.set_ylabel("Contribution to log-odds")
axes[0, 1].legend(loc="upper left", fontsize=5.6)
fig.tight_layout(w_pad=0.9, h_pad=1.2)
save(fig, "f6_ebm_shapes")

# ---- Supplementary Fig. 4: interaction network -----------------------
imp = dict(zip(ebm22.term_names_, ebm22.term_importances()))
mains = {n: v for n, v in imp.items() if " & " not in n and " x " not in n}
pairs = {}
for n, v in imp.items():
    for sep in (" & ", " x "):
        if sep in n:
            a, b = n.split(sep)
            pairs[(a.strip(), b.strip())] = v
print(f"terms: {len(mains)} mains, {len(pairs)} interactions")
Gr = nx.Graph()
for n, v in mains.items():
    Gr.add_node(n, w=v)
for (a, b), v in pairs.items():
    Gr.add_edge(a, b, w=v)
Gr.remove_nodes_from([n for n in list(Gr.nodes)
                      if Gr.degree(n) == 0 and mains.get(n, 0) <
                      np.percentile(list(mains.values()), 40)])
pos = nx.spring_layout(Gr, seed=0, k=1.35 / np.sqrt(len(Gr)), iterations=200)
fig, ax = plt.subplots(figsize=(6.4, 4.6))
wmax = max(pairs.values()) if pairs else 1
for (a, b), v in pairs.items():
    if a in pos and b in pos:
        sex = "Sex" in (a, b)
        ax.plot(*zip(pos[a], pos[b]), lw=0.6 + 3.4 * v / wmax,
                color=PAL["sexhl"] if sex else "#8ab4d8",
                alpha=.9 if sex else .65, zorder=1)
nmax = max(mains.values())
for n, p in pos.items():
    ax.scatter(*p, s=40 + 640 * mains.get(n, 0) / nmax, color=PAL["ebm"],
               alpha=.88, zorder=2, edgecolors="white", linewidths=.7)
    ax.annotate(n, p, fontsize=4.9, ha="center", va="center", zorder=3,
                color="white" if mains.get(n, 0) / nmax > .45 else "#1a1a1a")
ax.plot([], [], color=PAL["sexhl"], lw=2, label="interaction involving Sex")
ax.plot([], [], color="#8ab4d8", lw=2, label="other pairwise interaction")
ax.scatter([], [], s=160, color=PAL["ebm"], label="node size = term importance")
ax.legend(loc="lower left", fontsize=6)
ax.axis("off")
save(fig, "supp_f4_interactions")
