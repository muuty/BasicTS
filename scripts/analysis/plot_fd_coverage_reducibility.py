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

# coverage difficulty D, dataset-averaged, relative to free
dec = pd.read_csv(ANA / "fd_state_difficulty_decomposition.csv")
D = dec.groupby("traffic_state_transition")["D"].mean()
Dfree = D["free_to_free"]
Dratio = {s: D[s] / Dfree for s in ORDER}

# reducibility s, mean over detectors and datasets
red = pd.read_csv(ANA / "fd_state_reducibility.csv")
skill = red.groupby("traffic_state_transition")["skill"].mean()
# free-flow reducibility is not in that file (it holds non-free states + free); handle
if "free_to_free" not in skill.index:
    skill["free_to_free"] = np.nan

# excess degradation over free flow, detector-cluster median + CI
pt = pd.read_csv(ANA / "fd_state_paired_test.csv").set_index("state")
deg = {"free_to_free": (0.0, 0.0, 0.0)}
for s in ("breakdown", "recovery", "congested_to_congested"):
    deg[s] = (pt.loc[s, "median_excess"], pt.loc[s, "median_ci_lo"], pt.loc[s, "median_ci_hi"])

x = np.arange(len(ORDER))
fig, (axL, axR) = plt.subplots(1, 2, figsize=(8.4, 3.3))

# --- left: coverage difficulty ---
barsL = axL.bar(x, [Dratio[s] for s in ORDER], color="#9ecae1", edgecolor="#3182bd", width=0.62)
axL.axhline(1.0, color="grey", lw=0.8, ls="--")
axL.set_ylabel(r"Coverage difficulty $D/\mathrm{free}$")
axL.set_title("(a) Coverage difficulty by traffic state", fontsize=10)
axL.set_xticks(x); axL.set_xticklabels([LABEL[s] for s in ORDER], fontsize=8)
for xi, s in zip(x, ORDER):
    axL.text(xi, Dratio[s] + 0.06, f"{Dratio[s]:.1f}", ha="center", va="bottom", fontsize=8)
axL.set_ylim(0, max(Dratio.values()) * 1.18)

# --- right: degradation coloured by reducibility ---
cmap = cm.get_cmap("viridis")
norm = colors.Normalize(vmin=0.0, vmax=0.45)
vals = [deg[s][0] for s in ORDER]
cols = [cmap(norm(skill[s])) if not np.isnan(skill[s]) else "#cccccc" for s in ORDER]
err_lo = [deg[s][0] - deg[s][1] for s in ORDER]
err_hi = [deg[s][2] - deg[s][0] for s in ORDER]
axR.bar(x, vals, color=cols, edgecolor="#333333", width=0.62,
        yerr=[err_lo, err_hi], capsize=3, error_kw=dict(lw=1))
axR.axhline(0.0, color="grey", lw=0.8)
axR.set_ylabel("Degradation over free flow (MAE)")
axR.set_title("(b) Excess degradation by traffic state", fontsize=10)
axR.set_xticks(x); axR.set_xticklabels([LABEL[s] for s in ORDER], fontsize=8)
for xi, s in zip(x, ORDER):
    if not np.isnan(skill[s]):
        axR.text(xi, deg[s][2] + 0.06 if deg[s][0] >= 0 else deg[s][1] - 0.06,
                 f"$s$={skill[s]:.2f}", ha="center",
                 va="bottom" if deg[s][0] >= 0 else "top", fontsize=8)

sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
cb = fig.colorbar(sm, ax=axR, pad=0.02, fraction=0.05)
cb.set_label("Reducibility $s$", fontsize=9)

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fd_coverage_reducibility.{ext}", dpi=200, bbox_inches="tight")
print("wrote", OUT / "fd_coverage_reducibility.png")
print("D/free:", {LABEL[s].replace(chr(10), ' '): round(Dratio[s], 2) for s in ORDER})
print("skill :", {LABEL[s].replace(chr(10), ' '): round(float(skill[s]), 2) for s in ORDER})
print("deg   :", {LABEL[s].replace(chr(10), ' '): round(deg[s][0], 2) for s in ORDER})
