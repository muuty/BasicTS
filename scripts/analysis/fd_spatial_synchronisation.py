#!/usr/bin/env python3
"""Spatial synchronisation of FD traffic states, and what it does to the
detector-clustered bootstrap.

Coresets are network-wide time windows: one selected sample carries every
detector at once. The paper's state-conditional results are reported with a
bootstrap that resamples physical detectors as independent clusters. If
breakdown fires at many detectors in the same windows, detectors are not
independent clusters and the intervals are too narrow. The same synchronisation
also decides whether a small window budget can reach breakdown at all: atoms
concentrated in few windows are cheap to cover, atoms spread thin are not.

Four analyses, one CSV each, all on the detector-resolved FD labels produced by
fd_sensor_resolved.py (same fit, same screening, same transition coding, same
chronological training split the selectors run on):

  1. fd_spatial_concentration.csv  -- how breakdown / recovery / sustained
     congestion atoms distribute over windows, against a circular-shift null
     that destroys cross-detector synchronisation while preserving every
     detector's marginal state frequency and its own temporal autocorrelation.
  2. fd_spatial_events.csv         -- atoms collapsed into time-connected
     events; atoms per event is the simultaneity design effect.
  3. fd_spatial_cluster_bootstrap.csv -- the paper's paired retention contrasts
     recomputed with detector CLUSTERS as the resampling unit, where clusters
     come from agglomerative clustering on 1 - correlation of the detectors'
     breakdown-indicator series (no adjacency matrix required).
  4. fd_spatial_loss_concentration.csv -- concentration of the per-detector
     reduction loss (gain_lost) across the network, per state.

Post-hoc only; no retraining, no re-inference.
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fd_sensor_resolved import (  # noqa: E402
    DATASETS,
    OUT,
    classify_observations,
    fit_all_detectors,
    load_indices,
    load_raw,
    transition_codes,
)

PANEL = OUT / "fd_state_margin_panel.csv"
ORDER = ["free_to_free", "breakdown", "recovery", "congested_to_congested"]
LABEL = {"free_to_free": "free", "breakdown": "breakdown",
         "recovery": "recovery", "congested_to_congested": "sustained"}
EVENT_STATES = {1: "breakdown", 2: "recovery", 3: "congested_to_congested"}
TOP_FRACTIONS = (0.01, 0.05, 0.10)
N_BOOT = 10_000
BOOT_SEED = 42
NULL_SEED = 20260720


# --------------------------------------------------------------------------
# labels
# --------------------------------------------------------------------------
def label_matrix(dataset: str, args) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """[window, detector] transition codes on the training split.

    Straight reuse of fd_sensor_resolved: same two-branch fit, same screening
    thresholds, same speed-confirmed observation states, same 12/12 history and
    future horizon coding. -1 is unresolved.
    """
    raw = load_raw(dataset)
    screens, _ = fit_all_detectors(
        raw,
        n_bins=args.n_bins,
        min_branch_bins=args.min_branch_bins,
        min_free_rise=args.min_free_rise,
        min_congested_drop=args.min_congested_drop,
        min_fit_improvement=args.min_fit_improvement,
    )
    screens["dataset"] = dataset
    accepted = screens[screens["identifiable"]].sort_values("sensor")
    observation_state = classify_observations(raw, accepted)
    transition = transition_codes(
        observation_state,
        min_valid_fraction=args.min_valid_fraction,
        min_agreement=args.min_agreement,
    )
    return accepted["sensor"].to_numpy(dtype=int), transition, screens


# --------------------------------------------------------------------------
# concentration statistics
# --------------------------------------------------------------------------
def gini(x: np.ndarray) -> float:
    """Gini coefficient of a non-negative vector; 0 = flat, 1 = one holder."""
    x = np.asarray(x, dtype=float)
    if x.size == 0 or x.sum() <= 0:
        return float("nan")
    xs = np.sort(x)
    n = xs.size
    idx = np.arange(1, n + 1)
    return float(2.0 * (idx * xs).sum() / (n * xs.sum()) - (n + 1) / n)


def top_share(counts: np.ndarray, frac: float) -> float:
    n = counts.size
    total = counts.sum()
    if total <= 0:
        return float("nan")
    k = max(1, int(round(frac * n)))
    return float(np.sort(counts)[::-1][:k].sum() / total)


def concentration_stats(counts: np.ndarray) -> dict:
    out = {"gini": gini(counts)}
    for frac in TOP_FRACTIONS:
        out[f"top{int(round(frac * 100)):02d}pct_share"] = top_share(counts, frac)
    return out


def circular_shift_null(
    transition: np.ndarray, n_rep: int, seed: int
) -> dict[int, list[dict]]:
    """Independent circular shift per detector, all states from the same shift.

    A circular shift of a detector's whole label series leaves its marginal
    state frequencies exactly unchanged and its autocorrelation unchanged apart
    from one wrap seam, while randomising its phase relative to every other
    detector. Under the null, windows still differ in how many detectors are in
    a state, purely by Poisson-like superposition of independent series.
    """
    rng = np.random.default_rng(seed)
    n_windows, n_sensors = transition.shape
    indicators = {code: (transition == code) for code in EVENT_STATES}
    stats: dict[int, list[dict]] = {code: [] for code in EVENT_STATES}
    base = np.arange(n_windows)
    for _ in range(n_rep):
        offsets = rng.integers(0, n_windows, n_sensors)
        rows = (base[:, None] - offsets[None, :]) % n_windows
        for code, ind in indicators.items():
            counts = np.take_along_axis(ind, rows, axis=0).sum(axis=1)
            stats[code].append(concentration_stats(counts.astype(float)))
    return stats


def selector_coverage(dataset: str, counts: np.ndarray) -> dict:
    """Share of a state's atoms that sit inside an archived coreset.

    top10pct_share is the oracle: the best any 10% window budget could do if it
    were allowed to rank windows by the state's own atom count. The selectors
    never see state labels, so the gap between the two is the price of a
    label-blind, network-wide window budget.
    """
    out = {}
    total = counts.sum()
    for method in ("k_medoids", "random", "stride", "graph_cut"):
        for ratio in (0.1, 0.3):
            tag = f"{method}_{int(round(100 * ratio)):03d}_atom_share"
            try:
                selected = load_indices(dataset, method, ratio, 42)
            except (FileNotFoundError, ValueError):
                out[tag] = float("nan")
                continue
            out[tag] = float(counts[selected].sum() / total) if total > 0 else np.nan
    return out


def analyse_concentration(
    dataset: str, transition: np.ndarray, n_rep: int, seed: int
) -> list[dict]:
    null = circular_shift_null(transition, n_rep, seed)
    rows = []
    for code, name in EVENT_STATES.items():
        counts = (transition == code).sum(axis=1).astype(float)
        observed = concentration_stats(counts)
        null_frame = pd.DataFrame(null[code])
        row = {
            "dataset": dataset,
            "traffic_state_transition": name,
            "n_windows": int(transition.shape[0]),
            "n_detectors": int(transition.shape[1]),
            "n_atoms": int(counts.sum()),
            "atom_share_of_all_cells": float(
                counts.sum() / (transition.shape[0] * transition.shape[1])
            ),
            "windows_with_any": int((counts > 0).sum()),
            "window_share_with_any": float((counts > 0).mean()),
            "mean_detectors_per_active_window": float(
                counts[counts > 0].mean() if (counts > 0).any() else np.nan
            ),
            "max_detectors_in_a_window": int(counts.max()),
            "n_null_replicates": int(n_rep),
        }
        for key, value in observed.items():
            null_values = null_frame[key].to_numpy(dtype=float)
            row[f"obs_{key}"] = value
            row[f"null_mean_{key}"] = float(null_values.mean())
            row[f"null_sd_{key}"] = float(null_values.std(ddof=1))
            row[f"null_p_{key}"] = float(
                (1 + (null_values >= value).sum()) / (len(null_values) + 1)
            )
            row[f"ratio_{key}"] = float(value / null_values.mean())
        row.update(selector_coverage(dataset, counts))
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
# time-connected events
# --------------------------------------------------------------------------
def analyse_events(
    dataset: str, transition: np.ndarray, gaps: tuple[int, ...]
) -> list[dict]:
    """Collapse atoms into components connected in time.

    Two active windows join the same event when they are within `gap` windows
    of each other; every detector active anywhere in that stretch belongs to the
    event. A corridor-wide breakdown that lasts several windows therefore counts
    once, which is the quantity a window budget actually has to buy.
    """
    rows = []
    for code, name in EVENT_STATES.items():
        indicator = transition == code
        counts = indicator.sum(axis=1)
        active = np.flatnonzero(counts > 0)
        n_atoms = int(counts.sum())
        for gap in gaps:
            if active.size == 0:
                rows.append({
                    "dataset": dataset, "traffic_state_transition": name,
                    "gap_tolerance_windows": gap, "n_atoms": 0, "n_events": 0,
                })
                continue
            breaks = np.flatnonzero(np.diff(active) > gap)
            starts = np.concatenate([[0], breaks + 1])
            stops = np.concatenate([breaks + 1, [active.size]])
            n_events = len(starts)
            det_per_event, atoms_per_event, span_per_event = [], [], []
            for begin, end in zip(starts, stops):
                windows = active[begin:end]
                block = indicator[windows[0]:windows[-1] + 1]
                det_per_event.append(int(block.any(axis=0).sum()))
                atoms_per_event.append(int(indicator[windows].sum()))
                span_per_event.append(int(windows[-1] - windows[0] + 1))
            rows.append({
                "dataset": dataset,
                "traffic_state_transition": name,
                "gap_tolerance_windows": int(gap),
                "n_atoms": n_atoms,
                "n_active_windows": int(active.size),
                "n_events": int(n_events),
                "atoms_per_event": float(n_atoms / n_events),
                "active_windows_per_event": float(active.size / n_events),
                "mean_detectors_per_event": float(np.mean(det_per_event)),
                "median_detectors_per_event": float(np.median(det_per_event)),
                "max_detectors_per_event": int(np.max(det_per_event)),
                "mean_event_span_windows": float(np.mean(span_per_event)),
                "median_event_span_windows": float(np.median(span_per_event)),
                "max_event_span_windows": int(np.max(span_per_event)),
                "mean_atoms_per_event": float(np.mean(atoms_per_event)),
            })
    return rows


# --------------------------------------------------------------------------
# detector clustering and cluster bootstrap
# --------------------------------------------------------------------------
def breakdown_correlation(transition: np.ndarray) -> np.ndarray:
    """Pearson correlation of the detectors' breakdown-indicator series."""
    ind = (transition == 1).astype(np.float64)
    sd = ind.std(axis=0)
    corr = np.corrcoef(ind, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    dead = sd <= 0
    corr[dead, :] = 0.0
    corr[:, dead] = 0.0
    np.fill_diagonal(corr, 1.0)
    return corr


def cluster_detectors(corr: np.ndarray, n_clusters: int) -> np.ndarray:
    if n_clusters >= corr.shape[0]:
        return np.arange(corr.shape[0])
    distance = np.clip(1.0 - corr, 0.0, 2.0)
    distance = 0.5 * (distance + distance.T)
    np.fill_diagonal(distance, 0.0)
    link = linkage(squareform(distance, checks=False), method="average")
    return fcluster(link, t=n_clusters, criterion="maxclust") - 1


def cluster_diagnostics(corr: np.ndarray, labels: np.ndarray) -> dict:
    iu = np.triu_indices_from(corr, k=1)
    same = labels[iu[0]] == labels[iu[1]]
    sizes = np.bincount(labels)
    sizes = sizes[sizes > 0]
    return {
        "n_clusters": int(len(sizes)),
        "max_cluster_size": int(sizes.max()),
        "mean_cluster_size": float(sizes.mean()),
        "mean_within_cluster_corr": float(corr[iu][same].mean()) if same.any()
        else float("nan"),
        "mean_between_cluster_corr": float(corr[iu][~same].mean())
        if (~same).any() else float("nan"),
    }


def intra_cluster_correlation(
    diff: np.ndarray, dataset: np.ndarray, cluster: np.ndarray
) -> dict:
    """One-way random-effects ICC of the per-detector contrast within clusters.

    This is the quantity that decides whether the detector-clustered bootstrap
    is valid. Synchronised traffic states do not by themselves invalidate it:
    what matters is whether the estimated per-detector contrast is correlated
    within a synchronised group. ICC near zero means the design effect
    1 + (n0 - 1) * ICC is near one and detector-level resampling is adequate.
    Values are computed after removing the dataset main effect.
    """
    centred = diff.astype(float).copy()
    for name in np.unique(dataset):
        mask = dataset == name
        centred[mask] -= centred[mask].mean()
    keys, inverse = np.unique(cluster, return_inverse=True)
    n_total, n_groups = len(centred), len(keys)
    if n_groups < 2 or n_groups >= n_total:
        return {"icc": float("nan"), "n0": float("nan"), "design_effect": float("nan")}
    sizes = np.bincount(inverse).astype(float)
    means = np.bincount(inverse, weights=centred) / sizes
    grand = centred.mean()
    ss_between = float((sizes * (means - grand) ** 2).sum())
    ss_within = float(((centred - means[inverse]) ** 2).sum())
    ms_between = ss_between / (n_groups - 1)
    ms_within = ss_within / (n_total - n_groups)
    n0 = (n_total - (sizes ** 2).sum() / n_total) / (n_groups - 1)
    denominator = ms_between + (n0 - 1) * ms_within
    icc = float((ms_between - ms_within) / denominator) if denominator > 0 else np.nan
    return {
        "icc": icc,
        "n0": float(n0),
        "design_effect": float(1.0 + (n0 - 1) * max(icc, 0.0)),
    }


def leave_one_out_leverage(diff: np.ndarray, dataset: np.ndarray) -> dict:
    """How much of the point estimate rests on the single most extreme detector."""
    def estimate(mask):
        return float(np.mean([
            diff[mask & (dataset == d)].mean() for d in np.unique(dataset[mask])
        ]))
    full = estimate(np.ones(len(diff), bool))
    drop = np.ones(len(diff), bool)
    drop[int(np.argmax(np.abs(diff - diff.mean())))] = False
    return {
        "diff_sd": float(diff.std(ddof=1)),
        "diff_max_abs": float(np.abs(diff).max()),
        "mean_diff_drop_extreme": estimate(drop),
        "extreme_leverage": float(abs(full - estimate(drop)) / abs(full))
        if full != 0 else float("nan"),
    }


def clustered_bootstrap(
    diff: np.ndarray,
    dataset: np.ndarray,
    cluster: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
) -> dict:
    """Dataset-stratified bootstrap over resampling units given by `cluster`.

    cluster = detector index reproduces the paper's detector-clustered
    bootstrap; coarser clusters resample whole synchronised groups, so a group's
    detectors enter or leave the replicate together.
    """
    per_dataset_point, boot_parts = [], []
    for name in np.unique(dataset):
        mask = dataset == name
        d = diff[mask]
        keys, inverse = np.unique(cluster[mask], return_inverse=True)
        sums = np.bincount(inverse, weights=d, minlength=len(keys))
        sizes = np.bincount(inverse, minlength=len(keys)).astype(float)
        per_dataset_point.append(float(d.mean()))
        draw = rng.integers(0, len(keys), size=(n_boot, len(keys)))
        boot_parts.append(sums[draw].sum(axis=1) / sizes[draw].sum(axis=1))
    boot = np.mean(np.vstack(boot_parts), axis=0)
    low, high = np.quantile(boot, [0.025, 0.975])
    point = float(np.mean(per_dataset_point))
    return {
        "mean_diff": point,
        "ci_low": float(low),
        "ci_high": float(high),
        "ci_width": float(high - low),
        "boot_mean": float(boot.mean()),
        "boot_bias": float(boot.mean() - point),
        "boot_sd": float(boot.std(ddof=1)),
        "frac_positive": float((diff > 0).mean()),
        "excludes_zero": bool(low > 0 or high < 0),
    }


def retention_panel() -> pd.DataFrame:
    """Per-(detector, state) retention, M and Delta, as in the paper's script."""
    p = pd.read_csv(PANEL)
    p["M"] = p.persist_mae - p.full_mae
    p["Delta"] = p.red_mae - p.full_mae
    p = p[p.M > 0].copy()
    p["retention"] = 1.0 - p.Delta / p.M
    det = (p.groupby(["dataset", "sensor", "traffic_state_transition"], as_index=False)
             .agg(retention=("retention", "mean"), M=("M", "mean"),
                  Delta=("Delta", "mean")))
    return det.pivot_table(index=["dataset", "sensor"],
                           columns="traffic_state_transition",
                           values=["retention", "M", "Delta"])


def analyse_bootstrap(
    corr: dict[str, np.ndarray],
    sensors: dict[str, np.ndarray],
    fractions: tuple[float, ...],
    n_boot: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    wide = retention_panel()
    wide = wide[wide.index.get_level_values("dataset").isin(sensors)]
    granularities: list[tuple[str, dict[str, np.ndarray]]] = [
        ("detector", {d: np.arange(len(sensors[d])) for d in sensors})
    ]
    for frac in fractions:
        assignment, tag = {}, f"cluster_{int(round(frac * 100)):02d}pct"
        for d in sensors:
            target = max(2, int(round(frac * len(sensors[d]))))
            assignment[d] = cluster_detectors(corr[d], target)
        granularities.append((tag, assignment))

    diag_rows = []
    for tag, assignment in granularities:
        for d in sensors:
            diag_rows.append({
                "dataset": d, "granularity": tag,
                "n_detectors_screened": int(len(sensors[d])),
                **cluster_diagnostics(corr[d], assignment[d]),
            })

    rows = []
    for tag, assignment in granularities:
        rng = np.random.default_rng(BOOT_SEED)
        lookup = {d: dict(zip(sensors[d], assignment[d])) for d in sensors}
        for a, b in combinations(ORDER, 2):
            for col in ["retention", "M", "Delta"]:
                sub = wide[[(col, a), (col, b)]].dropna()
                if len(sub) < 10:
                    continue
                diff = (sub[(col, a)] - sub[(col, b)]).to_numpy()
                ds = sub.index.get_level_values("dataset").to_numpy()
                sen = sub.index.get_level_values("sensor").to_numpy()
                keep = np.array([s in lookup[d] for d, s in zip(ds, sen)])
                cl = np.array([f"{d}:{lookup[d][s]}" if k else ""
                               for d, s, k in zip(ds, sen, keep)])
                result = clustered_bootstrap(
                    diff[keep], ds[keep], cl[keep], n_boot, rng
                )
                result.update(intra_cluster_correlation(
                    diff[keep], ds[keep], cl[keep]
                ))
                # the retention statistic is heavy tailed; a single extreme
                # detector inside a small cluster inflates the between-cluster
                # mean square, so report the ICC without it as well
                kept_diff, kept_ds, kept_cl = diff[keep], ds[keep], cl[keep]
                trim = np.ones(len(kept_diff), bool)
                trim[int(np.argmax(np.abs(kept_diff - kept_diff.mean())))] = False
                trimmed = intra_cluster_correlation(
                    kept_diff[trim], kept_ds[trim], kept_cl[trim]
                )
                result["icc_drop_extreme"] = trimmed["icc"]
                result["design_effect_drop_extreme"] = trimmed["design_effect"]
                result.update(leave_one_out_leverage(diff[keep], ds[keep]))
                result.update({
                    "granularity": tag, "quantity": col,
                    "state_a": LABEL[a], "state_b": LABEL[b],
                    "n_detectors": int(keep.sum()),
                    "n_clusters": int(len(np.unique(cl[keep]))),
                    "n_dropped_unscreened": int((~keep).sum()),
                })
                rows.append(result)
    columns = ["granularity", "quantity", "state_a", "state_b", "n_detectors",
               "n_clusters", "mean_diff", "ci_low", "ci_high", "ci_width",
               "boot_mean", "boot_bias", "boot_sd", "icc", "n0",
               "design_effect", "icc_drop_extreme",
               "design_effect_drop_extreme", "frac_positive", "excludes_zero",
               "diff_sd", "diff_max_abs", "mean_diff_drop_extreme",
               "extreme_leverage", "n_dropped_unscreened"]
    return pd.DataFrame(rows)[columns], pd.DataFrame(diag_rows)


# --------------------------------------------------------------------------
# concentration of reduction loss over detectors
# --------------------------------------------------------------------------
def analyse_loss_concentration() -> pd.DataFrame:
    p = pd.read_csv(PANEL)
    det = (p.groupby(["dataset", "sensor", "traffic_state_transition"],
                     as_index=False)
             .agg(gain_lost=("gain_lost", "mean"),
                  full_gain=("full_gain", "mean")))
    rows = []
    for state in ORDER:
        for scope, frame in [("pooled", det[det.traffic_state_transition == state])] + [
            (d, det[(det.traffic_state_transition == state) & (det.dataset == d)])
            for d in sorted(det.dataset.unique())
        ]:
            x = frame["gain_lost"].to_numpy(dtype=float)
            if x.size == 0:
                continue
            positive = np.clip(x, 0.0, None)
            order = np.argsort(x)[::-1]
            k = max(1, int(round(0.10 * x.size)))
            top10 = x[order][:k]
            rows.append({
                "traffic_state_transition": LABEL[state],
                "scope": scope,
                "n_detectors": int(x.size),
                "n_negative_loss": int((x < 0).sum()),
                "frac_negative_loss": float((x < 0).mean()),
                "mean_gain_lost": float(x.mean()),
                "total_gain_lost": float(x.sum()),
                "gini_positive_part": gini(positive),
                "total_positive_gain_lost": float(positive.sum()),
                "top10pct_share_of_net_loss": float(top10.sum() / x.sum())
                if x.sum() != 0 else float("nan"),
                "top10pct_share_of_positive_loss": float(
                    np.clip(top10, 0, None).sum() / positive.sum()
                ) if positive.sum() > 0 else float("nan"),
                "top10pct_n_detectors": int(k),
                "mean_full_gain": float(frame["full_gain"].mean()),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--n-bins", type=int, default=40)
    parser.add_argument("--min-branch-bins", type=int, default=6)
    parser.add_argument("--min-free-rise", type=float, default=0.10)
    parser.add_argument("--min-congested-drop", type=float, default=0.05)
    parser.add_argument("--min-fit-improvement", type=float, default=0.10)
    parser.add_argument("--min-valid-fraction", type=float, default=2 / 3)
    parser.add_argument("--min-agreement", type=float, default=0.75)
    parser.add_argument("--null-replicates", type=int, default=200)
    parser.add_argument("--event-gaps", nargs="+", type=int, default=(1, 12))
    parser.add_argument("--cluster-fractions", nargs="+", type=float,
                        default=(0.50, 0.25, 0.10))
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    sensors, corr, conc_rows, event_rows = {}, {}, [], []
    for dataset in args.datasets:
        print(f"[{dataset}] fitting detector FDs and coding transitions", flush=True)
        sensor_ids, transition, _ = label_matrix(dataset, args)
        sensors[dataset] = sensor_ids
        resolved = (transition >= 0).mean()
        print(f"  detectors={len(sensor_ids)} windows={transition.shape[0]} "
              f"resolved_cells={resolved:.3f}", flush=True)
        conc_rows.extend(analyse_concentration(
            dataset, transition, args.null_replicates, NULL_SEED
        ))
        event_rows.extend(analyse_events(dataset, transition, tuple(args.event_gaps)))
        corr[dataset] = breakdown_correlation(transition)

    concentration = pd.DataFrame(conc_rows)
    events = pd.DataFrame(event_rows)
    concentration.to_csv(OUT / "fd_spatial_concentration.csv", index=False)
    events.to_csv(OUT / "fd_spatial_events.csv", index=False)

    boot, diag = analyse_bootstrap(
        corr, sensors, tuple(args.cluster_fractions), args.n_boot
    )
    boot.to_csv(OUT / "fd_spatial_cluster_bootstrap.csv", index=False)
    diag.to_csv(OUT / "fd_spatial_cluster_diagnostics.csv", index=False)

    loss = analyse_loss_concentration()
    loss.to_csv(OUT / "fd_spatial_loss_concentration.csv", index=False)

    report(concentration, events, boot, diag, loss)


def report(concentration, events, boot, diag, loss) -> None:
    print("\n" + "=" * 78)
    print("(1) SPATIAL CONCENTRATION OF STATE ATOMS OVER WINDOWS")
    print("=" * 78)
    for _, r in concentration.iterrows():
        print(f"\n{r.dataset} / {r.traffic_state_transition}: "
              f"{r.n_atoms:,} atoms over {r.n_windows:,} windows x "
              f"{r.n_detectors} detectors "
              f"({100 * r.atom_share_of_all_cells:.2f}% of cells)")
        print(f"  active windows {r.windows_with_any:,} "
              f"({100 * r.window_share_with_any:.1f}%), "
              f"mean {r.mean_detectors_per_active_window:.1f} detectors when active, "
              f"max {r.max_detectors_in_a_window}")
        for key in ["top01pct_share", "top05pct_share", "top10pct_share", "gini"]:
            print(f"  {key:<16s} obs={r[f'obs_{key}']:.4f}  "
                  f"null={r[f'null_mean_{key}']:.4f} "
                  f"(sd {r[f'null_sd_{key}']:.4f})  "
                  f"ratio={r[f'ratio_{key}']:.2f}x  p={r[f'null_p_{key}']:.4f}")
        covered = [c for c in concentration.columns if c.endswith("_atom_share")]
        print("  atom share inside archived coresets (seed 42), "
              f"oracle top-10% = {r['obs_top10pct_share']:.3f}:")
        print("   " + "  ".join(
            f"{c.replace('_atom_share', '')}={r[c]:.3f}" for c in covered
        ))

    print("\n" + "=" * 78)
    print("(2) TIME-CONNECTED EVENTS (simultaneity design effect)")
    print("=" * 78)
    for _, r in events.iterrows():
        print(f"{r.dataset:<15s} {r.traffic_state_transition:<24s} "
              f"gap<={r.gap_tolerance_windows:<3d} "
              f"atoms={r.n_atoms:>7,d} events={r.n_events:>5,d} "
              f"atoms/event={r.atoms_per_event:>8.1f} "
              f"detectors/event={r.mean_detectors_per_event:>6.1f} "
              f"span={r.mean_event_span_windows:>6.1f}w")

    print("\n" + "=" * 78)
    print("(3) RETENTION CONTRASTS UNDER DETECTOR VS CLUSTER RESAMPLING")
    print("=" * 78)
    print("\ncluster structure (breakdown-indicator correlation, average linkage):")
    for _, r in diag.iterrows():
        print(f"  {r.dataset:<15s} {r.granularity:<15s} "
              f"clusters={r.n_clusters:>4d} max_size={r.max_cluster_size:>4d} "
              f"within_corr={r.mean_within_cluster_corr:+.4f} "
              f"between_corr={r.mean_between_cluster_corr:+.4f}")
    key_pairs = [("free", "breakdown"), ("free", "recovery"),
                 ("breakdown", "recovery")]
    for quantity in ["retention", "M", "Delta"]:
        print(f"\n--- {quantity}: state_a minus state_b, 95% bootstrap CI ---")
        for a, b in key_pairs:
            sub = boot[(boot.quantity == quantity) & (boot.state_a == a)
                       & (boot.state_b == b)]
            for _, r in sub.iterrows():
                mark = "  *" if r.excludes_zero else "   (spans 0)"
                icc = "     n/a" if not np.isfinite(r.icc) else f"{r.icc:+7.4f}"
                icc_t = "     n/a" if not np.isfinite(r.icc_drop_extreme) \
                    else f"{r.icc_drop_extreme:+7.4f}"
                print(f"  {a:>9s} - {b:<9s} {r.granularity:<15s} "
                      f"clusters={r.n_clusters:>4d} "
                      f"{r.mean_diff:+7.3f} [{r.ci_low:+7.3f},{r.ci_high:+7.3f}] "
                      f"width={r.ci_width:6.3f} boot_mean={r.boot_mean:+7.3f} "
                      f"icc={icc} icc_trim={icc_t}{mark}")
    print("\n  * interval excludes zero")
    print("  icc      = intra-cluster correlation of the per-detector contrast")
    print("  icc_trim = same, after dropping the single most extreme detector")
    print("  boot_mean drifting away from mean_diff means the cluster estimator "
          "is diluting an outlier detector")
    print("\nsensitivity of the point estimate to its single most extreme detector:")
    lev = boot[(boot.granularity == "detector") & (boot.quantity == "retention")]
    for _, r in lev.iterrows():
        print(f"  retention {r.state_a:>9s} - {r.state_b:<9s} "
              f"mean={r.mean_diff:+7.3f} drop_extreme={r.mean_diff_drop_extreme:+7.3f} "
              f"leverage={100 * r.extreme_leverage:5.1f}%  "
              f"sd={r.diff_sd:7.3f} max|diff|={r.diff_max_abs:8.3f}")

    print("\n" + "=" * 78)
    print("(4) CONCENTRATION OF REDUCTION LOSS OVER DETECTORS")
    print("=" * 78)
    for _, r in loss.iterrows():
        print(f"  {r.traffic_state_transition:<10s} {r.scope:<15s} "
              f"n={r.n_detectors:>4d} neg={100 * r.frac_negative_loss:>4.0f}% "
              f"gini+={r.gini_positive_part:.3f} "
              f"top10%_of_net={r.top10pct_share_of_net_loss:>7.3f} "
              f"top10%_of_pos={r.top10pct_share_of_positive_loss:.3f} "
              f"mean_lost={r.mean_gain_lost:+.3f}")

    print("\nwrote:")
    for name in ["fd_spatial_concentration.csv", "fd_spatial_events.csv",
                 "fd_spatial_cluster_bootstrap.csv",
                 "fd_spatial_cluster_diagnostics.csv",
                 "fd_spatial_loss_concentration.csv"]:
        print(f"  {OUT / name}")


if __name__ == "__main__":
    main()
