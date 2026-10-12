"""Fig. 6: lexical similarity vs detection rate per CWE (data: exp31_similarity_vs_difficulty.csv)."""
import sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = pd.read_csv(sys.argv[1]); out = sys.argv[2]
x, y, n = d.mean_similarity.values, d.mean_frac_correct.values, d.n_samples.values

# label offsets in points (dx, dy); points in the crowded 0.933-0.937 band get leader lines
off = {"CWE-617": (-8, 8), "CWE-476": (-6, 10), "CWE-415": (8, 4), "CWE-119": (8, 4),
       "CWE-200": (0, 11), "CWE-416": (-8, -16), "CWE-20": (-46, -14),
       "CWE-703": (-50, 14), "CWE-125": (-58, 2), "CWE-787": (26, 20),
       "CWE-190": (28, -2), "CWE-369": (20, -22)}
leader = {"CWE-703", "CWE-125", "CWE-787", "CWE-190", "CWE-369", "CWE-20", "CWE-416"}

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.6})
fig, ax = plt.subplots(figsize=(5.2, 3.6), dpi=300)
ax.grid(True, color="#e6e6e6", linewidth=0.5, zorder=0)

b, a = np.polyfit(x, y, 1); xs = np.linspace(x.min() - .002, x.max() + .002, 50)
ax.plot(xs, a + b * xs, ls="--", lw=1, color="#8a8a8a", zorder=1, label="Linear trend")

sizes = 18 + n * 2.2
is617 = d.primary_cwe.eq("CWE-617").values
ax.scatter(x[~is617], y[~is617], s=sizes[~is617], color="#4a78b5", edgecolor="white", linewidth=0.8, zorder=3)
ax.scatter(x[is617], y[is617], s=sizes[is617], color="#c8553d", edgecolor="white", linewidth=0.8, zorder=4)

for cwe, xi, yi in zip(d.primary_cwe, x, y):
    dx, dy = off[cwe]
    ax.annotate(cwe, (xi, yi), xytext=(dx, dy), textcoords="offset points", fontsize=7.5,
                color="#222222", fontweight="bold" if cwe == "CWE-617" else "normal",
                ha="center" if dx == 0 else ("left" if dx > 0 else "right"), va="center",
                arrowprops=dict(arrowstyle="-", lw=0.5, color="#9a9a9a", shrinkA=0, shrinkB=3) if cwe in leader else None)

ax.set_xlabel("Mean lexical similarity (vulnerable vs. patched)")
ax.set_ylabel("Mean detection rate (higher = easier)")
ax.set_xlim(0.914, 0.966); ax.set_ylim(0.41, 0.76)
for s_, lab in [(10, "n = 10"), (40, "n = 40"), (70, "n = 70")]:
    ax.scatter([], [], s=18 + s_ * 2.2, color="#b9b9b9", edgecolor="white", label=lab)
leg = ax.legend(loc="upper right", fontsize=7, frameon=False, labelspacing=1.1, borderpad=0.6, scatterpoints=1,
                title="Dot size = vulnerable\ntest samples per CWE", title_fontsize=7)
leg._legend_box.align = "left"
ax.text(0.015, 0.03, r"Spearman $\rho = -0.30$, $p = 0.34$", transform=ax.transAxes, fontsize=7.5, color="#444444")
fig.tight_layout(); fig.savefig(out, dpi=300); print("saved", out)
