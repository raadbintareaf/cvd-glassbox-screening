"""Supp Fig 4: EBM pairwise-interaction chord diagram.
Nodes on a ring ordered by main-term importance; chords = selected
interactions, width ~ importance; Sex-involving chords highlighted."""
import joblib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path as MPath
import matplotlib.patches as mpatches

plt.rcParams.update({"font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "pdf.fonttype": 42, "savefig.dpi": 300})

obj = joblib.load("results/models/brfss2022/T1/ebm/seed0/model.joblib")
ebm = getattr(obj, "model_", obj)
imp = dict(zip(ebm.term_names_, ebm.term_importances()))
mains, pairs = {}, {}
for n, v in imp.items():
    hit = None
    for sep in (" & ", " x "):
        if sep in n:
            hit = sep
            break
    if hit:
        a, b = (t.strip() for t in n.split(hit))
        pairs[(a, b)] = v
    else:
        mains[n] = v
print(f"terms: {len(mains)} mains, {len(pairs)} interactions")

nodes = sorted(mains, key=mains.get, reverse=True)
N = len(nodes)
ang = {n: np.pi / 2 - 2 * np.pi * i / N for i, n in enumerate(nodes)}
R = 1.0
pos = {n: (R * np.cos(a), R * np.sin(a)) for n, a in ang.items()}

fig, ax = plt.subplots(figsize=(6.6, 6.6))
ax.set_xlim(-1.62, 1.62)
ax.set_ylim(-1.62, 1.62)
ax.set_aspect("equal")
ax.axis("off")

wmax = max(pairs.values()) if pairs else 1.0
order = sorted(pairs.items(), key=lambda kv: kv[1])          # weak first
for (a, b), v in order:
    if a not in pos or b not in pos:
        continue
    p1, p2 = np.array(pos[a]), np.array(pos[b])
    mid = (p1 + p2) / 2 * 0.25                                # pull to centre
    path = MPath([p1, mid, p2],
                 [MPath.MOVETO, MPath.CURVE3, MPath.CURVE3])
    sex = "Sex" in (a, b)
    ax.add_patch(mpatches.PathPatch(
        path, fc="none",
        ec="#D55E00" if sex else "#7fb2d8",
        lw=0.5 + 4.0 * v / wmax,
        alpha=0.95 if sex else 0.55,
        zorder=3 if sex else 2))

nmax = max(mains.values())
for n in nodes:
    x, y = pos[n]
    ax.scatter(x, y, s=30 + 520 * mains[n] / nmax, color="#0072B2",
               edgecolors="white", linewidths=.8, zorder=4)
    a = ang[n]
    lx, ly = 1.12 * np.cos(a), 1.12 * np.sin(a)
    rot = np.degrees(a)
    ha = "left"
    if 90 < (rot % 360) < 270:
        rot += 180
        ha = "right"
    ax.text(lx, ly, n, rotation=rot, rotation_mode="anchor",
            ha=ha, va="center", fontsize=6.0,
            color="#1a1a1a", zorder=5)

ax.plot([], [], color="#D55E00", lw=2.2, label="interaction involving Sex")
ax.plot([], [], color="#7fb2d8", lw=2.2, label="other pairwise interaction")
ax.scatter([], [], s=200, color="#0072B2",
           label="node size = main-term importance")
ax.legend(loc="lower left", bbox_to_anchor=(-0.06, -0.06), fontsize=6.6,
          frameon=False)
fig.savefig("paper/figures/supp_f4_interactions.pdf", bbox_inches="tight")
fig.savefig("paper/figures/supp_f4_interactions.png", bbox_inches="tight",
            dpi=300)
print("[fig] supp_f4_interactions (chord)")
