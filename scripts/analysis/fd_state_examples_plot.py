#!/usr/bin/env python3
"""Representative 24-step examples of the four fundamental-diagram traffic states.

For each transition state (free->free, breakdown, recovery, sustained
congestion) we pick one clean detector-window and plot its 12-step history and
12-step future, in both the speed trace and the flow-occupancy plane, marking
the fitted critical point and the history/future boundary.

Reuses the labelling in fd_sensor_resolved.py unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "analysis"))

import fd_sensor_resolved as fr  # noqa: E402

OUT = REPO / "experiments" / "result" / "analysis"
PAPER = REPO / "writing" / "CoresetSelection-paper" / "figures"

# single unified colour for every panel's title and speed trace
STATE_COLOR = "#333333"
STATES = [
    (0, "Free flow", STATE_COLOR),
    (1, "Breakdown", STATE_COLOR),
    (2, "Recovery", STATE_COLOR),
    (3, "Congestion", STATE_COLOR),
]
T = fr.T_IN  # 12
DATASET = "SAN_BERNARDINO"


def accepted_detectors(raw):
    """Fit every detector, return the accepted table (with oc / vc)."""
    screens, _ = fr.fit_all_detectors(
        raw,
        n_bins=40,
        min_branch_bins=6,
        min_free_rise=0.10,
        min_congested_drop=0.05,
        min_fit_improvement=0.10,
    )
    accepted = screens[screens["identifiable"]].copy()
    return accepted[
        ["sensor", "critical_occupancy", "speed_at_capacity", "congested_drop",
         "capacity_flow", "free_slope", "congested_slope"]
    ].reset_index(drop=True)


def _state_clarity(hist, fut, qwin, vc, state):
    """Clarity of a candidate window, in speed units, comparable across states.
    qwin is the 24-step flow trace; free flow prefers busier (non-empty) roads
    so its FD point sits on the free branch, not at the origin."""
    spread = hist.std() + fut.std()
    if state == 0:  # free->free: stay well above critical, stable, and busy
        return (min(hist.min(), fut.min()) - vc) - 0.5 * spread + 0.15 * qwin.mean()
    if state == 1:  # breakdown: drop from above to below critical
        return (hist.mean() - fut.mean()) - 0.3 * spread
    if state == 2:  # recovery: rise from below to above critical
        return (fut.mean() - hist.mean()) - 0.3 * spread
    # sustained: stay well below critical, stable
    return (vc - max(hist.max(), fut.max())) - 0.5 * spread


def _best_window_per_state(v_det, q_det, code_det, vc):
    """For one detector: best (start, clarity) for each state, or None."""
    out = {}
    for state, _, _ in STATES:
        best = None
        for w in np.flatnonzero(code_det == state):
            hist, fut = v_det[w : w + T], v_det[w + T : w + 2 * T]
            qwin = q_det[w : w + 2 * T]
            if not (np.all(np.isfinite(hist)) and np.all(np.isfinite(fut))
                    and np.all(np.isfinite(qwin))):
                continue
            c = _state_clarity(hist, fut, qwin, vc, state)
            if best is None or c > best[1]:
                best = (int(w), float(c))
        out[state] = best
    return out


def pick_examples(raw, accepted, single_sensor=True):
    """Pick example windows. If single_sensor, all four come from the detector
    whose weakest state example is clearest (so one location, four regimes,
    one shared critical point)."""
    sensors = accepted["sensor"].to_numpy(int)
    obs = fr.classify_observations(raw, accepted)
    code = fr.transition_codes(obs, min_valid_fraction=2 / 3, min_agreement=0.75)
    vc = accepted["speed_at_capacity"].to_numpy(float)

    if single_sensor:
        best_det, best_score, best_wins = None, -np.inf, None
        for d in range(len(sensors)):
            per = _best_window_per_state(raw[:, sensors[d], 2], raw[:, sensors[d], 0], code[:, d], vc[d])
            if any(per[s] is None for s, _, _ in STATES):
                continue
            score = min(per[s][1] for s, _, _ in STATES)  # weakest state
            if score > best_score:
                best_score, best_det = score, d
                best_wins = {s: per[s][0] for s, _, _ in STATES}
        if best_det is None:
            raise SystemExit("no single detector has clean windows for all states")
        return {s: (int(sensors[best_det]), best_wins[s], best_det) for s, _, _ in STATES}

    chosen = {}
    for state, _, _ in STATES:
        best = None
        for d in range(len(sensors)):
            per = _best_window_per_state(raw[:, sensors[d], 2], raw[:, sensors[d], 0], code[:, d], vc[d])
            if per[state] is None:
                continue
            if best is None or per[state][1] > best[0]:
                best = (per[state][1], int(sensors[d]), per[state][0], d)
        chosen[state] = best[1:]
    return chosen


def main():
    raw = fr.load_raw(DATASET)
    accepted = accepted_detectors(raw)
    print(f"accepted detectors: {len(accepted)}")
    chosen = pick_examples(raw, accepted)

    steps = np.arange(2 * T)

    def pad(lo, hi, frac=0.06):
        m = (hi - lo) * frac
        return lo - m, hi + m

    # speed axis from the four example windows (+ critical speeds)
    v_win = np.concatenate(
        [raw[st : st + 2 * T, sn, 2] for sn, st, _ in (chosen[s] for s, _, _ in STATES)]
    )
    vc_all = [accepted.iloc[chosen[s][2]]["speed_at_capacity"] for s, _, _ in STATES]
    v_lo, v_hi = pad(min(v_win.min(), min(vc_all)), max(v_win.max(), max(vc_all)))

    # FD axes from the FULL empirical cloud of the (single) chosen detector,
    # so the background density shows the real fundamental diagram.
    fd_sensor = chosen[STATES[0][0]][0]
    o_full = raw[:, fd_sensor, 1]
    q_full = raw[:, fd_sensor, 0]
    fin = np.isfinite(o_full) & np.isfinite(q_full)
    o_full, q_full = o_full[fin], q_full[fin]
    # cover both the cloud and every example window so no trajectory is clipped
    win_o = np.concatenate(
        [raw[st : st + 2 * T, sn, 1] for sn, st, _ in (chosen[s] for s, _, _ in STATES)]
    )
    win_q = np.concatenate(
        [raw[st : st + 2 * T, sn, 0] for sn, st, _ in (chosen[s] for s, _, _ in STATES)]
    )
    o_lo, o_hi = pad(0.0, max(np.percentile(o_full, 99.8), win_o.max()))
    q_lo, q_hi = pad(0.0, max(np.percentile(q_full, 99.8), win_q.max()))

    fig = plt.figure(figsize=(11, 5.2))
    gs = GridSpec(2, 4, figure=fig, hspace=0.34, wspace=0.18)

    for col, (state, label, color) in enumerate(STATES):
        sensor, start, di = chosen[state]
        row = accepted.iloc[di]
        vc = row["speed_at_capacity"]
        oc = row["critical_occupancy"]
        qc, fs, cs = row["capacity_flow"], row["free_slope"], row["congested_slope"]
        sl = slice(start, start + 2 * T)
        q = raw[sl, sensor, 0]
        o = raw[sl, sensor, 1]
        v = raw[sl, sensor, 2]

        # top: speed trace
        ax = fig.add_subplot(gs[0, col])
        ax.axvspan(-0.5, T - 0.5, color="0.92", zorder=0)
        ax.plot(steps[:T], v[:T], "-o", ms=3, color=color, label="history")
        ax.plot(
            steps[T - 1 :], v[T - 1 :], "--o", ms=3, color=color,
            markerfacecolor="white",
        )
        ax.axhline(vc, color="0.4", lw=0.8, ls=":")
        ax.axvline(T - 0.5, color="0.6", lw=0.8)
        ax.set_title(label, fontsize=10, color=color)
        ax.set_ylim(v_lo, v_hi)
        if col == 0:
            ax.set_ylabel("Speed")
        else:
            ax.set_yticklabels([])
        ax.set_xticks([0, T, 2 * T - 1])
        ax.tick_params(labelsize=8)

        # bottom: flow-occupancy (fundamental diagram) trajectory, coloured by time
        ax2 = fig.add_subplot(gs[1, col])
        # background: empirical FD density of the whole training period (real data)
        ax2.hexbin(o_full, q_full, gridsize=45, cmap="Greys", bins="log",
                   mincnt=1, linewidths=0, alpha=0.55, zorder=0,
                   extent=(o_lo, o_hi, q_lo, q_hi))
        # faint fitted piecewise-linear FD over the occupancy range it was fit on
        og = np.linspace(max(o_lo, o_full.min()), min(o_hi, np.percentile(o_full, 99.8)), 200)
        qg = qc + fs * np.minimum(og - oc, 0.0) + cs * np.maximum(og - oc, 0.0)
        ax2.plot(og, qg, color="#e08214", lw=1.3, ls="-", alpha=0.9, zorder=1)
        pts = np.column_stack([o, q]).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap="viridis", norm=plt.Normalize(0, 2 * T - 1),
                            linewidth=2, zorder=2)
        lc.set_array(steps[:-1])
        ax2.add_collection(lc)
        # all markers coloured by time; history vs future is shown in the top row
        ax2.scatter(o, q, c=steps, cmap="viridis", norm=plt.Normalize(0, 2 * T - 1),
                    s=24, zorder=3, edgecolor="0.3", linewidth=0.4)
        # start (square) and end (diamond) markers + direction arrow
        ax2.scatter(o[0], q[0], s=95, marker="s", facecolor="none",
                    edgecolor="black", linewidth=1.3, zorder=5)
        ax2.scatter(o[-1], q[-1], s=95, marker="D", facecolor="none",
                    edgecolor="black", linewidth=1.3, zorder=5)
        ax2.annotate("", xy=(o[-1], q[-1]), xytext=(o[-3], q[-3]),
                     arrowprops=dict(arrowstyle="-|>", color="0.25", lw=1.4), zorder=4)
        ax2.axvline(oc, color="0.4", lw=0.8, ls=":")
        ax2.set_xlim(o_lo, o_hi)
        ax2.set_ylim(q_lo, q_hi)
        if col == 0:
            ax2.set_ylabel("Flow")
        else:
            ax2.set_yticklabels([])
        ax2.set_xlabel("Occupancy")
        ax2.tick_params(labelsize=8)

    # shared legend
    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], color="0.3", marker="o", ms=5, ls="-", label="History (12 steps)"),
        Line2D([], [], color="0.3", marker="o", ms=5, ls="--",
               markerfacecolor="white", label="Future (12 steps)"),
        Line2D([], [], color="black", marker="s", ms=7, ls="none",
               markerfacecolor="none", label="Window start"),
        Line2D([], [], color="black", marker="D", ms=7, ls="none",
               markerfacecolor="none", label="Window end"),
        Line2D([], [], color="#e08214", ls="-", alpha=0.9, label="Fitted FD"),
        Line2D([], [], marker="h", color="0.5", ls="none", ms=7, label="Empirical FD density"),
        Line2D([], [], color="0.4", ls=":", label="Fitted critical point"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=6, fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.5, 1.03))

    # shared colorbar for the FD time gradient
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, 2 * T - 1))
    cax = fig.add_axes([0.25, -0.02, 0.5, 0.02])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("Time step within window (0 = start of history, 23 = end of future)",
                 fontsize=8)
    cb.ax.tick_params(labelsize=7)

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        p = OUT / f"fd_state_examples.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"wrote {p}")
    # place a copy in the paper figures directory
    for ext in ("png", "pdf"):
        fig.savefig(PAPER / f"fd_state_examples.{ext}", dpi=200, bbox_inches="tight")
    print("copied to paper figures/")
    for state, label, _ in STATES:
        s, w, _ = chosen[state]
        print(f"  {label:22} sensor={s} start={w}")


if __name__ == "__main__":
    main()
