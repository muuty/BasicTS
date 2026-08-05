#!/usr/bin/env python3
"""Publication figures for the FD state-margin analysis (coreset paper).

Produces three figures (PNG dpi 200 + PDF) into
writing/CoresetSelection-paper/figures/ :

  (1) fd_vulnerability_axes        - scatter x=D (coverage difficulty),
                                      y=M (reducible margin), area proportional
                                      to Delta (gain lost), colour = kappa
                                      (margin retention).
  (2) fd_reducibility_corr_heatmap - 5 backbones x 4 states heatmap of
                                      within-state Pearson r(reducibility,
                                      degradation); diverging cmap centred at 0.
  (3) fd_coverage_decomposition    - stacked bars D = K J_X + rho by state.

All quantities are averaged over the 5 backbones and 2 datasets unless a
per-backbone/per-state cell is the plotted object (figure 2).
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
OUT.mkdir(parents=True, exist_ok=True)

ORDER = ["free_to_free", "breakdown", "recovery", "congested_to_congested"]
LABEL = {"free_to_free": "Free", "breakdown": "Breakdown",
         "recovery": "Recovery", "congested_to_congested": "Sustained congestion"}
LABEL_NL = {"free_to_free": "Free", "breakdown": "Breakdown",
            "recovery": "Recovery",
            "congested_to_congested": "Sustained\ncongestion"}

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "figure.dpi": 200,
})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    print("wrote", OUT / f"{name}.png", "and .pdf")


# ---------------------------------------------------------------------------
# Figure (1): vulnerability axes
# ---------------------------------------------------------------------------
def fig_vulnerability_axes():
    df = pd.read_csv(ANA / "fd_state_margin_by_state.csv").set_index(
        "traffic_state_transition")
    exp = pd.read_csv(ANA / "fd_traffic_state_exposure.csv")
    # pooled exposure share = sum(window_detector_atoms per state)/grand total
    atoms = exp.groupby("traffic_state_transition")["window_detector_atoms"].sum()
    pooled_exp = atoms / atoms.sum()

    D = {s: float(df.loc[s, "D"]) for s in ORDER}
    M = {s: float(df.loc[s, "full_gain"]) for s in ORDER}
    Delta = {s: float(df.loc[s, "gain_lost"]) for s in ORDER}
    kappa = {s: float(df.loc[s, "gain_retention_ratio_of_means"]) for s in ORDER}

    fig, ax = plt.subplots(figsize=(6.0, 4.4))

    # marker area proportional to Delta (gain lost)
    area_per_unit = 620.0
    sizes = np.array([Delta[s] * area_per_unit for s in ORDER])

    cmap = plt.get_cmap("viridis")
    norm = colors.Normalize(vmin=0.70, vmax=0.95)
    cols = [cmap(norm(kappa[s])) for s in ORDER]

    xs = [D[s] for s in ORDER]
    ys = [M[s] for s in ORDER]
    ax.scatter(xs, ys, s=sizes, c=cols, edgecolors="#222222", linewidths=0.9,
               zorder=3, alpha=0.95)

    # annotate state names with explicit per-state placement (offset in points)
    # to avoid colliding with the colourbar, the size legend, and each other
    ann = {
        "free_to_free":           dict(xytext=(20, 6),  ha="left",  va="bottom"),
        "recovery":               dict(xytext=(0, 26),  ha="center", va="bottom"),
        "breakdown":              dict(xytext=(0, 28),  ha="center", va="bottom"),
        "congested_to_congested": dict(xytext=(-16, 4), ha="right", va="center"),
    }
    for s in ORDER:
        ax.annotate(LABEL[s], (D[s], M[s]), textcoords="offset points",
                    fontsize=9, fontweight="bold", **ann[s])

    ax.set_xlabel("Coverage difficulty $D$")
    ax.set_ylabel("Reducible margin $M$ (MAE)")
    ax.set_xlim(0.05, 0.46)
    ax.set_ylim(0, 17.5)
    ax.grid(True, ls=":", lw=0.6, color="#cccccc", zorder=0)
    ax.set_axisbelow(True)

    # colourbar for kappa (retention)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.05)
    cb.set_label(r"Margin retention $\kappa$")

    # size legend for Delta (gain lost)
    leg_vals = [1.0, 2.0, 3.0]
    handles = [plt.scatter([], [], s=v * area_per_unit, facecolor="#dddddd",
                           edgecolor="#222222", linewidths=0.9)
               for v in leg_vals]
    labels = [f"{v:.0f}" for v in leg_vals]
    leg = ax.legend(handles, labels, title=r"Gain lost $\Delta$ (MAE)",
                    loc="lower left", labelspacing=1.4, borderpad=1.0,
                    handletextpad=1.2, frameon=True, framealpha=0.9)
    leg.get_title().set_fontsize(8)

    fig.tight_layout()
    save(fig, "fd_vulnerability_axes")

    print("\n[fig1] vulnerability axes summary (state: D, M, Delta, kappa, exposure):")
    for s in ORDER:
        print(f"  {LABEL[s]:22s}  D={D[s]:.4f}  M={M[s]:.4f}  "
              f"Delta={Delta[s]:.4f}  kappa={kappa[s]:.4f}  "
              f"exposure={float(pooled_exp[s]):.4f}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure (2): within-state r(reducibility, degradation) heatmap
# ---------------------------------------------------------------------------
def fig_corr_heatmap():
    corr = pd.read_csv(ANA / "fd_state_margin_correlations.csv")
    backbones = ["STGCN", "STID", "DCRNN", "AGCRN", "STAEformer"]
    piv = corr.pivot(index="backbone", columns="traffic_state_transition",
                     values="r_reducibility_degradation")
    piv = piv.reindex(index=backbones, columns=ORDER)
    Mmat = piv.to_numpy()

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    vmax = np.nanmax(np.abs(Mmat))
    norm = colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.get_cmap("RdBu_r")
    im = ax.imshow(Mmat, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels([LABEL_NL[s] for s in ORDER])
    ax.set_yticks(range(len(backbones)))
    ax.set_yticklabels(backbones)

    for i in range(len(backbones)):
        for j in range(len(ORDER)):
            v = Mmat[i, j]
            # text colour: dark on light cells, white on saturated cells
            tc = "white" if abs(v) > 0.55 * vmax else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=9, color=tc)

    cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.05)
    cb.set_label(r"within-state $r(s,\ \Delta)$")
    ax.set_title("Reducibility-degradation correlation by backbone and state",
                 fontsize=10)
    fig.tight_layout()
    save(fig, "fd_reducibility_corr_heatmap")

    print("\n[fig2] heatmap: r_reducibility_degradation")
    print(f"  positive cells: {int((Mmat > 0).sum())}/20  "
          f"median r = {np.nanmedian(Mmat):.3f}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure (3): coverage difficulty decomposition D = K J_X + rho
# ---------------------------------------------------------------------------
def fig_coverage_decomposition():
    dec = pd.read_csv(ANA / "fd_state_difficulty_decomposition.csv")
    g = dec.groupby("traffic_state_transition")[["geometry", "residual", "D"]].mean()
    geo = {s: float(g.loc[s, "geometry"]) for s in ORDER}
    res = {s: float(g.loc[s, "residual"]) for s in ORDER}
    Dtot = {s: float(g.loc[s, "D"]) for s in ORDER}

    x = np.arange(len(ORDER))
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    gvals = [geo[s] for s in ORDER]
    rvals = [res[s] for s in ORDER]
    ax.bar(x, gvals, width=0.62, color="#c6dbef", edgecolor="#3182bd",
           label=r"History coverage $K J_X$")
    ax.bar(x, rvals, bottom=gvals, width=0.62, color="#08519c",
           edgecolor="#3182bd", label=r"Future divergence $\rho$")
    ax.set_ylabel(r"Coverage difficulty $D$")
    ax.set_title(r"$D = K J_X + \rho$ by traffic state", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL_NL[s] for s in ORDER])
    ax.legend(loc="upper left")
    ax.set_ylim(0, max(Dtot.values()) * 1.22)
    fig.tight_layout()
    save(fig, "fd_coverage_decomposition")

    print("\n[fig3] coverage decomposition (state: K J_X, rho, D):")
    for s in ORDER:
        print(f"  {LABEL[s]:22s}  geo={geo[s]:.4f}  rho={res[s]:.4f}  "
              f"D={Dtot[s]:.4f}")
    plt.close(fig)


if __name__ == "__main__":
    fig_vulnerability_axes()
    fig_corr_heatmap()
    fig_coverage_decomposition()
