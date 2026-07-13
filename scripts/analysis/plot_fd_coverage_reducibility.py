#!/usr/bin/env python3
"""Discovery figure: coverage difficulty is not enough; reducibility decides.

Left panel  - coverage difficulty D/free by traffic state (breakdown and
              sustained congestion are comparably hard to cover).
Right panel - excess reduced-data degradation over free flow, with a 95%
              detector-cluster interval, coloured by reducibility s.
The pairing shows breakdown and sustained congestion sit at comparable coverage
difficulty, yet only breakdown degrades, and the sign of degradation tracks
reducibility.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors

REPO = Path(__file__).resolve().parents[2]
ANA = REPO / "experiments/result/analysis"
OUT = REPO / "writing/CoresetSelection-paper/figures"

ORDER = ["free_to_free", "breakdown", "recovery", "congested_to_congested"]
LABEL = {"free_to_free": "Free", "breakdown": "Breakdown",
         "recovery": "Recovery", "congested_to_congested": "Congestion"}

# coverage difficulty D decomposed into history coverage (K J_X) and future
# divergence (rho), dataset-averaged
dec = pd.read_csv(ANA / "fd_state_difficulty_decomposition.csv")
g = dec.groupby("traffic_state_transition")[["geometry", "residual", "D"]].mean()
geo = {s: float(g.loc[s, "geometry"]) for s in ORDER}
res = {s: float(g.loc[s, "residual"]) for s in ORDER}
Dtot = {s: float(g.loc[s, "D"]) for s in ORDER}

# reducibility s, mean over detectors and datasets
red = pd.read_csv(ANA / "fd_state_reducibility.csv")
skill = red.groupby("traffic_state_transition")["skill"].mean()
# free-flow reducibility is not in that file (it holds non-free states + free); handle
if "free_to_free" not in skill.index:
    skill["free_to_free"] = np.nan

# raw reduced-data degradation per state, detector-cluster bootstrap CI
det = red.groupby(["dataset", "sensor", "traffic_state_transition"])["degradation"].mean().reset_index()
rng = np.random.default_rng(42)
deg = {}
for s in ORDER:
    v = det[det.traffic_state_transition == s].groupby(["dataset", "sensor"])["degradation"].mean().to_numpy()
    boot = np.array([rng.choice(v, len(v), replace=True).mean() for _ in range(5000)])
    deg[s] = (float(v.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975)))

x = np.arange(len(ORDER))
fig, (axL, axR) = plt.subplots(1, 2, figsize=(8.4, 3.3))

# --- left: coverage difficulty D = history coverage (K J_X) + future divergence (rho) ---
gvals = [geo[s] for s in ORDER]
rvals = [res[s] for s in ORDER]
axL.bar(x, gvals, width=0.62, color="#c6dbef", edgecolor="#3182bd",
        label=r"History coverage $K J_X$")
axL.bar(x, rvals, bottom=gvals, width=0.62, color="#08519c", edgecolor="#3182bd",
        label=r"Future divergence $\rho$")
axL.set_ylabel(r"Coverage difficulty $D$")
axL.set_title(r"(a) $D = K J_X + \rho$ by traffic state", fontsize=10)
axL.set_xticks(x); axL.set_xticklabels([LABEL[s] for s in ORDER], fontsize=8)
axL.legend(fontsize=7, loc="upper left")
axL.set_ylim(0, max(Dtot.values()) * 1.22)

# --- right: degradation coloured by reducibility ---
cmap = cm.get_cmap("viridis")
norm = colors.Normalize(vmin=0.0, vmax=0.45)
vals = [deg[s][0] for s in ORDER]
cols = [cmap(norm(skill[s])) if not np.isnan(skill[s]) else "#cccccc" for s in ORDER]
err_lo = [deg[s][0] - deg[s][1] for s in ORDER]
err_hi = [deg[s][2] - deg[s][0] for s in ORDER]
axR.bar(x, vals, color=cols, edgecolor="#333333", width=0.62,
        yerr=[err_lo, err_hi], capsize=3, error_kw=dict(lw=1))
axR.set_ylabel("Reduced-data degradation (MAE)")
axR.set_title("(b) Degradation by traffic state", fontsize=10)
axR.set_xticks(x); axR.set_xticklabels([LABEL[s] for s in ORDER], fontsize=8)
for xi, s in zip(x, ORDER):
    if not np.isnan(skill[s]):
        axR.text(xi, deg[s][2] + 0.05, f"$s$={skill[s]:.2f}", ha="center", va="bottom", fontsize=8)
axR.set_ylim(0, max(deg[s][2] for s in ORDER) * 1.16)

sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
cb = fig.colorbar(sm, ax=axR, pad=0.02, fraction=0.05)
cb.set_label("Reducibility $s$", fontsize=9)

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fd_coverage_reducibility.{ext}", dpi=200, bbox_inches="tight")
print("wrote", OUT / "fd_coverage_reducibility.png")
print("geo+res:", {LABEL[s]: (round(geo[s],2), round(res[s],2)) for s in ORDER})
print("skill :", {LABEL[s].replace(chr(10), ' '): round(float(skill[s]), 2) for s in ORDER})
print("deg   :", {LABEL[s].replace(chr(10), ' '): round(deg[s][0], 2) for s in ORDER})
