#!/usr/bin/env python3
"""Concentration of breakdown atoms over network-wide time windows.

Panel (a): windows are ranked by the number of detectors simultaneously in a
breakdown transition, and the cumulative share of all breakdown atoms is plotted
against the cumulative share of windows. The observed curve is compared with a
circular-shift null that destroys cross-detector synchronisation while leaving
every detector's marginal state frequency and its own autocorrelation intact,
and with the diagonal a state spread evenly over windows would follow.

Panel (b): the share of breakdown atoms that a 10% window budget retains, for
the four audited selection objectives and for the ranking that panel (a) traces.

Labels, screening, transition coding and the chronological training split are
taken unchanged from fd_sensor_resolved.py, so the curve and the archived
fd_spatial_concentration.csv anchors come from one pipeline. The script asserts
that reproduction before plotting.

Usage:  conda activate cuda && python scripts/analysis/fd_concentration_plot.py
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fd_sensor_resolved import (  # noqa: E402
    OUT,
    classify_observations,
    fit_all_detectors,
    load_raw,
    transition_codes,
)

REPO = Path(__file__).resolve().parents[2]
PAPER = REPO / "writing" / "CoresetSelection-paper" / "figures"
CACHE = Path(tempfile.gettempdir()) / "fd_concentration_curves.npz"

DATASETS = ("SAN_BERNARDINO", "CONTRA_COSTA")
NICE = {"SAN_BERNARDINO": "San Bernardino", "CONTRA_COSTA": "Contra Costa"}
BREAKDOWN_CODE = 1
NULL_SEED = 20260720          # identical to fd_spatial_synchronisation.py
N_NULL = 200
BUDGET = 0.10
OBJECTIVES = [
    ("k_medoids", "K-medoids"),
    ("stride", "Stride"),
    ("random", "Random"),
    ("graph_cut", "Graph Cut"),
]


# --------------------------------------------------------------------------
def label_matrix(dataset: str, args) -> np.ndarray:
    raw = load_raw(dataset)
    screens, _ = fit_all_detectors(
        raw,
        n_bins=args.n_bins,
        min_branch_bins=args.min_branch_bins,
        min_free_rise=args.min_free_rise,
        min_congested_drop=args.min_congested_drop,
        min_fit_improvement=args.min_fit_improvement,
    )
    accepted = screens[screens["identifiable"]].sort_values("sensor")
    observation_state = classify_observations(raw, accepted)
    return transition_codes(
        observation_state,
        min_valid_fraction=args.min_valid_fraction,
        min_agreement=args.min_agreement,
    )


def cumulative_curve(counts: np.ndarray) -> np.ndarray:
    """Cumulative share of atoms when windows are ranked by their atom count."""
    ordered = np.sort(counts.astype(float))[::-1]
    total = ordered.sum()
    return np.cumsum(ordered) / total


def null_curves(transition: np.ndarray, n_rep: int, seed: int) -> np.ndarray:
    """Circular-shift null, one independent phase shift per detector.

    The rng is consumed exactly as in fd_spatial_synchronisation.py (one draw of
    n_sensors offsets per replicate), so the top-share anchors reproduce.
    """
    rng = np.random.default_rng(seed)
    n_windows, n_sensors = transition.shape
    indicator = transition == BREAKDOWN_CODE
    base = np.arange(n_windows)
    out = np.empty((n_rep, n_windows), dtype=np.float64)
    for r in range(n_rep):
        offsets = rng.integers(0, n_windows, n_sensors)
        rows = (base[:, None] - offsets[None, :]) % n_windows
        counts = np.take_along_axis(indicator, rows, axis=0).sum(axis=1)
        out[r] = cumulative_curve(counts.astype(float))
    return out


def thin(n: int, head: int = 200, step: int = 10) -> np.ndarray:
    """Indices for drawing: every point over the steep head, every step after.

    Only the rendered path is thinned; every reported number is computed on the
    full curve. The tail is monotone and near linear, so a step of 10 windows
    (0.07% of the axis) is invisible at print size and keeps the vector file
    small.
    """
    tail = np.arange(head, n, step)
    return np.concatenate([np.arange(min(head, n)), tail, [n - 1]])


def top_share(curve: np.ndarray, frac: float) -> float:
    n = curve.size
    return float(curve[max(1, int(round(frac * n))) - 1])


# --------------------------------------------------------------------------
def compute(args) -> dict:
    if CACHE.exists() and not args.recompute:
        return dict(np.load(CACHE))
    store: dict[str, np.ndarray] = {}
    for dataset in DATASETS:
        print(f"[{dataset}] fitting detector FDs and coding transitions", flush=True)
        transition = label_matrix(dataset, args)
        counts = (transition == BREAKDOWN_CODE).sum(axis=1).astype(float)
        print(f"  windows={transition.shape[0]} detectors={transition.shape[1]} "
              f"breakdown atoms={int(counts.sum()):,}", flush=True)
        store[f"{dataset}_obs"] = cumulative_curve(counts)
        store[f"{dataset}_null"] = null_curves(transition, N_NULL, NULL_SEED)
        store[f"{dataset}_atoms"] = np.array([counts.sum()])
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, **store)
    return store


def verify(store: dict) -> pd.DataFrame:
    """Reproduce the archived anchors before anything is plotted."""
    archived = pd.read_csv(OUT / "fd_spatial_concentration.csv")
    archived = archived[archived.traffic_state_transition == "breakdown"]
    rows = []
    for dataset in DATASETS:
        ref = archived[archived.dataset == dataset].iloc[0]
        obs = store[f"{dataset}_obs"]
        null = store[f"{dataset}_null"]
        for frac, col in ((0.01, "01"), (0.05, "05"), (0.10, "10")):
            got_obs = top_share(obs, frac)
            got_null = float(np.mean([top_share(c, frac) for c in null]))
            rows.append({
                "dataset": dataset, "frac": frac,
                "obs": got_obs, "obs_archived": ref[f"obs_top{col}pct_share"],
                "null": got_null, "null_archived": ref[f"null_mean_top{col}pct_share"],
            })
    frame = pd.DataFrame(rows)
    frame["obs_err"] = (frame.obs - frame.obs_archived).abs()
    frame["null_err"] = (frame.null - frame.null_archived).abs()
    print("\nreproduction of fd_spatial_concentration.csv anchors:")
    print(frame.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    worst = max(frame.obs_err.max(), frame.null_err.max())
    if worst > 1e-9:
        raise AssertionError(f"anchors do not reproduce (max abs error {worst:.3g})")
    print(f"  max absolute error {worst:.3g}")
    return archived.set_index("dataset")


# --------------------------------------------------------------------------
def plot(store: dict, archived: pd.DataFrame) -> None:
    plt.rcParams.update({
        "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.4,
        "lines.solid_capstyle": "round",
    })
    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(6.5, 3.0), gridspec_kw={"width_ratios": [1.16, 1.0]}
    )

    # ---- panel (a): cumulative concentration curve --------------------
    style = {"CONTRA_COSTA": ("0.66", 2.8), "SAN_BERNARDINO": ("black", 1.2)}
    handles = {}
    for dataset in ("CONTRA_COSTA", "SAN_BERNARDINO"):
        obs = store[f"{dataset}_obs"]
        null = store[f"{dataset}_null"]
        keep = thin(obs.size)
        x = ((np.arange(obs.size) + 1) / obs.size)[keep]
        colour, width = style[dataset]
        lo, hi = np.percentile(null, [2.5, 97.5], axis=0)
        ax.fill_between(x, lo[keep], hi[keep], color="0.85", linewidth=0,
                        zorder=1)
        ax.plot(x, null.mean(axis=0)[keep], color=colour, lw=width,
                ls=(0, (3.5, 2)), zorder=3)
        handles[dataset], = ax.plot(x, obs[keep], color=colour, lw=width,
                                    zorder=4, label=NICE[dataset])
    ax.plot([0, 1], [0, 1], color="0.55", lw=0.8, ls=(0, (1, 2.2)), zorder=2)
    ax.axvline(BUDGET, color="0.30", lw=0.7, ls=(0, (5, 3)), zorder=2)

    ax.annotate("even over windows", xy=(0.70, 0.70), xytext=(0.60, 0.44),
                fontsize=7.2, color="0.35", ha="left",
                arrowprops=dict(arrowstyle="-", color="0.55", lw=0.6))
    ax.annotate("circular-shift null", xy=(0.28, 0.585), xytext=(0.37, 0.26),
                fontsize=7.2, color="0.20", ha="left",
                arrowprops=dict(arrowstyle="-", color="0.35", lw=0.6))
    ax.text(BUDGET + 0.02, 0.02, r"$r=0.1$", fontsize=7.2, color="0.30")

    ax.set_xlabel("share of windows, ranked by simultaneous\nbreakdown detectors")
    ax.set_ylabel("cumulative share of breakdown atoms")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.005)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(handles=[handles[d] for d in DATASETS], loc="lower right",
              frameon=False, handlelength=2.0, borderpad=0.2, labelspacing=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.text(-0.30, 1.03, "(a)", transform=ax.transAxes, fontsize=9, va="bottom")

    # ---- panel (b): what a 10% budget retains -------------------------
    rows = [("Ranked on\nbreakdown count", "obs_top10pct_share")]
    rows += [(nice, f"{key}_010_atom_share") for key, nice in OBJECTIVES]
    y = np.arange(len(rows))[::-1]
    printed = []
    for i, (label, column) in enumerate(rows):
        values = [float(archived.loc[d, column]) for d in DATASETS]
        printed.append((label.replace("\n", " "), values))
        bx.plot([0, max(values)], [y[i], y[i]], color="0.85", lw=0.8, zorder=1)
        bx.plot(values[0], y[i], marker="o", ms=5.0, mfc="black", mec="black",
                ls="none", zorder=3,
                label=NICE[DATASETS[0]] if i == 0 else None)
        bx.plot(values[1], y[i], marker="o", ms=5.0, mfc="white", mec="black",
                mew=1.0, ls="none", zorder=3,
                label=NICE[DATASETS[1]] if i == 0 else None)
        bx.text(max(values) + 0.035, y[i],
                f"{values[0]:.3f} / {values[1]:.3f}", fontsize=7.0,
                va="center", color="0.15")
    bx.set_yticks(y)
    bx.set_yticklabels([label for label, _ in rows], fontsize=7.8)
    bx.set_xlim(0, 0.90)
    bx.set_ylim(-1.15, len(rows) - 0.4)
    bx.set_xticks([0, 0.2, 0.4, 0.6])
    bx.set_xlabel("share of breakdown atoms retained\nat a 10% window budget")
    bx.legend(loc="lower center", bbox_to_anchor=(0.56, 0.0), ncol=2,
              frameon=False, handlelength=1.0, borderpad=0.2,
              columnspacing=1.4, numpoints=1)
    for s in ("top", "right", "left"):
        bx.spines[s].set_visible(False)
    bx.tick_params(axis="y", length=0)
    bx.text(-0.46, 1.03, "(b)", transform=bx.transAxes, fontsize=9, va="bottom")

    fig.tight_layout(w_pad=1.6)
    PAPER.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(PAPER / f"fd_concentration.{ext}", dpi=300, bbox_inches="tight")
    print(f"\nwrote {PAPER / 'fd_concentration.png'} and .pdf")
    print("panel (b) values (San Bernardino / Contra Costa):")
    for label, values in printed:
        print(f"  {label:<26s} {values[0]:.4f} / {values[1]:.4f}")


# --------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-bins", type=int, default=40)
    parser.add_argument("--min-branch-bins", type=int, default=6)
    parser.add_argument("--min-free-rise", type=float, default=0.10)
    parser.add_argument("--min-congested-drop", type=float, default=0.05)
    parser.add_argument("--min-fit-improvement", type=float, default=0.10)
    parser.add_argument("--min-valid-fraction", type=float, default=2 / 3)
    parser.add_argument("--min-agreement", type=float, default=0.75)
    parser.add_argument("--recompute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = compute(args)
    archived = verify(store)
    plot(store, archived)


if __name__ == "__main__":
    main()
