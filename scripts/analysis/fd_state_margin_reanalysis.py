#!/usr/bin/env python3
"""Robust traffic-state reanalysis of coverage and reducibility.

This is a post-hoc analysis of existing detector-level predictions and the
fundamental-diagram (FD) support audit.  It does not train or rerun a model.

The analysis separates three quantities that were partially conflated in the
original reducibility framing:

    R = (L_persist - L_full) / L_persist   normalized full-data gain
    M = L_persist - L_full                 absolute full-data gain (margin)
    Delta = L_reduced - L_full             gain lost after data reduction

The gain retained by the reduced model is M_reduced = M - Delta.  The ratio
Delta / M is reported only for detectors with R >= 0.05, because ratios are
unstable when the full-data model has negligible or negative gain over the
persistence baseline.

Outputs are written below ``experiments/result/analysis``.  Bootstrap units are
physical detectors, stratified by dataset.  State summaries first average the
five backbones within detector and then give the two datasets equal weight.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, pearsonr, spearmanr


REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "experiments" / "result" / "analysis"
PANEL_CSV = OUT / "fd_state_reducibility_multimodel.csv"
MAE_CSV = OUT / "fd_state_conditional_mae_by_detector.csv"
FITS_CSV = OUT / "fd_sensor_resolved_fits.csv"

DATASETS = ("SAN_BERNARDINO", "CONTRA_COSTA")
BACKBONES = ("STGCN", "STID", "DCRNN", "AGCRN", "STAEformer")
STATES = (
    "free_to_free",
    "breakdown",
    "recovery",
    "congested_to_congested",
)
STATE_LABELS = {
    "free_to_free": "Free→Free",
    "breakdown": "Breakdown",
    "recovery": "Recovery",
    "congested_to_congested": "Cong.→Cong.",
}
KEY = ["dataset", "sensor", "traffic_state_transition", "backbone"]
RANGE_END = 24_192
VALID_STEPS = int(RANGE_END * 0.2)
TEST_STEPS = int(RANGE_END * 0.2)
TRAIN_STEPS = RANGE_END - VALID_STEPS - TEST_STEPS
RNG_SEED = 20_260_720


def finite_corr(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray, method="pearson"):
    """Correlation and p-value after finite-value filtering."""
    a, b = np.asarray(x, float), np.asarray(y, float)
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    if len(a) < 5 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan, np.nan, len(a)
    result = pearsonr(a, b) if method == "pearson" else spearmanr(a, b)
    statistic = result.statistic if hasattr(result, "statistic") else result[0]
    pvalue = result.pvalue if hasattr(result, "pvalue") else result[1]
    return float(statistic), float(pvalue), len(a)


def residualized_corr(
    frame: pd.DataFrame,
    x: str,
    y: str,
    control_full_mae: bool = False,
):
    """Pearson correlation adjusted for dataset and optionally full-data MAE."""
    cols = [x, y, "dataset"] + (["full_mae"] if control_full_mae else [])
    d = frame[cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 5:
        return np.nan
    design = [np.ones(len(d))]
    if d.dataset.nunique() > 1:
        design.append((d.dataset == sorted(d.dataset.unique())[-1]).to_numpy(float))
    if control_full_mae:
        design.append(d.full_mae.to_numpy(float))
    z = np.column_stack(design)
    xv, yv = d[x].to_numpy(float), d[y].to_numpy(float)
    xr = xv - z @ np.linalg.lstsq(z, xv, rcond=None)[0]
    yr = yv - z @ np.linalg.lstsq(z, yv, rcond=None)[0]
    return finite_corr(xr, yr)[0]


def residualized_corr_arrays(x, y, dataset_indicator, full_mae=None):
    """Fast array equivalent of :func:`residualized_corr` for bootstrap draws."""
    design = [np.ones(len(x))]
    if np.unique(dataset_indicator).size > 1:
        design.append(dataset_indicator)
    if full_mae is not None:
        design.append(full_mae)
    z = np.column_stack(design)
    xr = x - z @ np.linalg.lstsq(z, x, rcond=None)[0]
    yr = y - z @ np.linalg.lstsq(z, y, rcond=None)[0]
    xsd, ysd = np.std(xr), np.std(yr)
    if len(xr) < 5 or xsd == 0 or ysd == 0:
        return np.nan
    return float(np.mean((xr - xr.mean()) * (yr - yr.mean())) / (xsd * ysd))


def bootstrap_corr(
    frame: pd.DataFrame,
    x: str,
    y: str,
    reps: int,
    seed: int,
    control_full_mae: bool = False,
):
    columns = [x, y, "dataset"] + (["full_mae"] if control_full_mae else [])
    clean = frame[columns].replace([np.inf, -np.inf], np.nan).dropna()
    groups = []
    for dataset_idx, (_, d) in enumerate(clean.groupby("dataset", sort=False)):
        groups.append(
            (
                d[x].to_numpy(float),
                d[y].to_numpy(float),
                np.full(len(d), dataset_idx, dtype=float),
                d.full_mae.to_numpy(float) if control_full_mae else None,
            )
        )
    rng = np.random.default_rng(seed)
    values = np.empty(reps, dtype=float)
    for i in range(reps):
        bx, by, bd, bf = [], [], [], []
        for gx, gy, gd, gf in groups:
            idx = rng.integers(0, len(gx), len(gx))
            bx.append(gx[idx])
            by.append(gy[idx])
            bd.append(gd[idx])
            if control_full_mae:
                bf.append(gf[idx])
        values[i] = residualized_corr_arrays(
            np.concatenate(bx),
            np.concatenate(by),
            np.concatenate(bd),
            np.concatenate(bf) if control_full_mae else None,
        )
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    return tuple(np.quantile(values, [0.025, 0.975]))


def equal_dataset_mean(frame: pd.DataFrame, metric: str):
    return float(frame.groupby("dataset")[metric].mean().mean())


def bootstrap_equal_dataset_means(
    detector_frame: pd.DataFrame,
    metrics: list[str],
    reps: int,
    seed: int,
):
    """Vector bootstrap of equal-dataset detector means."""
    arrays = []
    for _, d in detector_frame.groupby("dataset", sort=False):
        arrays.append(d[metrics].to_numpy(float))
    rng = np.random.default_rng(seed)
    draws = np.full((reps, len(metrics)), np.nan, dtype=float)
    for i in range(reps):
        dataset_means = []
        for a in arrays:
            idx = rng.integers(0, len(a), len(a))
            dataset_means.append(np.nanmean(a[idx], axis=0))
        draws[i] = np.nanmean(dataset_means, axis=0)
    return np.nanquantile(draws, [0.025, 0.975], axis=0)


def load_panel() -> pd.DataFrame:
    panel = pd.read_csv(PANEL_CSV)
    mae = pd.read_csv(MAE_CSV)
    support = (
        mae[mae.method == "full"]
        .groupby(KEY, as_index=False)
        .agg(n_windows=("n_windows", "max"))
    )
    panel = panel.merge(support, on=KEY, how="left", validate="one_to_one")
    if panel.n_windows.isna().any():
        raise ValueError("Missing n_windows after merging the full-data support table")

    panel["full_gain"] = panel.persist_mae - panel.full_mae
    panel["reduced_gain"] = panel.persist_mae - panel.red_mae
    panel["gain_lost"] = panel.red_mae - panel.full_mae
    panel["relative_degradation"] = panel.gain_lost / panel.full_mae
    panel["gain_loss_fraction"] = panel.gain_lost / panel.full_gain
    panel["gain_retention"] = panel.reduced_gain / panel.full_gain
    panel["full_beats_persistence"] = panel.full_gain > 0
    panel["stable_positive_gain"] = panel.reducibility >= 0.05
    panel["reduced_beats_persistence"] = panel.reduced_gain > 0

    # Cross-model quantities break the shared L_full term between a target
    # backbone's degradation and its reducibility.  They ask whether the same
    # detector-state is reducible for the other four architectures.
    detector_state = panel.groupby(
        ["dataset", "sensor", "traffic_state_transition"]
    )
    for metric in ("reducibility", "full_gain"):
        count = detector_state[metric].transform("count")
        panel[f"loo_{metric}"] = (
            detector_state[metric].transform("sum") - panel[metric]
        ) / (count - 1)
    if (count < 2).any():
        raise AssertionError("Leave-one-model-out analysis requires at least two backbones")

    expected_r = panel.full_gain / panel.persist_mae
    if not np.allclose(panel.reducibility, expected_r, rtol=1e-9, atol=1e-9):
        raise AssertionError("Stored reducibility does not equal full_gain / persist_mae")
    if not np.allclose(panel.gain_lost, panel.full_gain - panel.reduced_gain):
        raise AssertionError("Gain decomposition identity failed")
    return panel


def state_summaries(panel: pd.DataFrame, reps: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = [
        "persist_mae",
        "full_mae",
        "red_mae",
        "D",
        "reducibility",
        "full_gain",
        "reduced_gain",
        "gain_lost",
        "relative_degradation",
    ]
    rows = []
    model_rows = []
    for state_idx, state in enumerate(STATES):
        s = panel[panel.traffic_state_transition == state]
        # A physical detector is the inferential unit; average architectures first.
        detector = s.groupby(["dataset", "sensor"], as_index=False)[metrics].mean()
        ci = bootstrap_equal_dataset_means(
            detector, metrics, reps, RNG_SEED + 100 * state_idx
        )
        row = {
            "traffic_state_transition": state,
            "n_detectors": int(detector.groupby("dataset").size().sum()),
            "n_rows": len(s),
            "n_windows_median": float(s.groupby(["dataset", "sensor"]).n_windows.max().median()),
            "full_beats_persistence_fraction": float(s.full_beats_persistence.mean()),
            "reduced_beats_persistence_fraction": float(s.reduced_beats_persistence.mean()),
        }
        for j, metric in enumerate(metrics):
            row[metric] = equal_dataset_mean(detector, metric)
            row[f"{metric}_ci_low"] = ci[0, j]
            row[f"{metric}_ci_high"] = ci[1, j]
        # Ratio of aggregate gains is more stable than a mean of detector ratios.
        row["gain_loss_fraction_ratio_of_means"] = row["gain_lost"] / row["full_gain"]
        row["gain_retention_ratio_of_means"] = row["reduced_gain"] / row["full_gain"]
        rows.append(row)

        for backbone in BACKBONES:
            b = s[s.backbone == backbone]
            r = {
                "backbone": backbone,
                "traffic_state_transition": state,
                "n_detectors": len(b),
                "n_windows_median": float(b.n_windows.median()),
            }
            for metric in metrics:
                r[metric] = equal_dataset_mean(b, metric)
            r["gain_loss_fraction_ratio_of_means"] = r["gain_lost"] / r["full_gain"]
            r["gain_retention_ratio_of_means"] = r["reduced_gain"] / r["full_gain"]
            model_rows.append(r)
    return pd.DataFrame(rows), pd.DataFrame(model_rows)


def correlation_table(panel: pd.DataFrame, reps: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, dataset_rows = [], []
    corr_specs = [
        ("reducibility", False, "r_reducibility_degradation"),
        ("reducibility", True, "partial_r_reducibility_given_full_mae"),
        ("loo_reducibility", False, "r_loo_reducibility_degradation"),
        ("full_gain", False, "r_full_gain_degradation"),
        ("loo_full_gain", False, "r_loo_full_gain_degradation"),
        ("D", False, "r_D_degradation"),
    ]
    cell = 0
    for backbone in BACKBONES:
        for state in STATES:
            s = panel[
                (panel.backbone == backbone)
                & (panel.traffic_state_transition == state)
            ].copy()
            raw_r, raw_p, _ = finite_corr(s.reducibility, s.gain_lost)
            rank_r, rank_p, _ = finite_corr(s.reducibility, s.gain_lost, "spearman")
            loo_rank_r, loo_rank_p, _ = finite_corr(
                s.loo_reducibility, s.gain_lost, "spearman"
            )
            row = {
                "backbone": backbone,
                "traffic_state_transition": state,
                "n_detectors": len(s),
                "raw_r_reducibility_degradation": raw_r,
                "raw_p_reducibility_degradation": raw_p,
                "raw_spearman_reducibility_degradation": rank_r,
                "raw_spearman_p_reducibility_degradation": rank_p,
                "raw_spearman_loo_reducibility_degradation": loo_rank_r,
                "raw_spearman_p_loo_reducibility_degradation": loo_rank_p,
            }
            for spec_idx, (x, control, name) in enumerate(corr_specs):
                row[name] = residualized_corr(s, x, "gain_lost", control)
                low, high = bootstrap_corr(
                    s,
                    x,
                    "gain_lost",
                    reps,
                    RNG_SEED + 10_000 * cell + 100 * spec_idx,
                    control,
                )
                row[f"{name}_ci_low"] = low
                row[f"{name}_ci_high"] = high

            stable = s[s.stable_positive_gain].copy()
            row["n_stable_positive_gain"] = len(stable)
            row["r_D_gain_loss_fraction_stable"] = residualized_corr(
                stable, "D", "gain_loss_fraction"
            )
            if len(stable) >= 10:
                low, high = bootstrap_corr(
                    stable,
                    "D",
                    "gain_loss_fraction",
                    reps,
                    RNG_SEED + 10_000 * cell + 900,
                )
            else:
                low, high = np.nan, np.nan
            row["r_D_gain_loss_fraction_stable_ci_low"] = low
            row["r_D_gain_loss_fraction_stable_ci_high"] = high
            rows.append(row)

            for dataset, d in s.groupby("dataset"):
                dr, dp, dn = finite_corr(d.reducibility, d.gain_lost)
                ds, dsp, _ = finite_corr(d.reducibility, d.gain_lost, "spearman")
                dataset_rows.append(
                    {
                        "dataset": dataset,
                        "backbone": backbone,
                        "traffic_state_transition": state,
                        "n_detectors": dn,
                        "r_reducibility_degradation": dr,
                        "p_reducibility_degradation": dp,
                        "spearman_reducibility_degradation": ds,
                        "spearman_p_reducibility_degradation": dsp,
                        "r_loo_reducibility_degradation": finite_corr(
                            d.loo_reducibility, d.gain_lost
                        )[0],
                        "r_full_gain_degradation": finite_corr(d.full_gain, d.gain_lost)[0],
                        "r_loo_full_gain_degradation": finite_corr(
                            d.loo_full_gain, d.gain_lost
                        )[0],
                        "r_D_degradation": finite_corr(d.D, d.gain_lost)[0],
                    }
                )
            cell += 1
    return pd.DataFrame(rows), pd.DataFrame(dataset_rows)


def sensitivity_table(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    support_specs = [("all_windows", 1), ("min20", 20), ("min50", 50), ("min100", 100)]
    gain_specs = [
        ("all_gains", lambda d: np.ones(len(d), dtype=bool)),
        ("positive_gain", lambda d: d.full_gain > 0),
        ("reducibility_ge_0.05", lambda d: d.reducibility >= 0.05),
    ]
    for support_name, minimum in support_specs:
        for gain_name, gain_filter in gain_specs:
            for backbone in BACKBONES:
                for state in STATES:
                    s = panel[
                        (panel.backbone == backbone)
                        & (panel.traffic_state_transition == state)
                        & (panel.n_windows >= minimum)
                    ].copy()
                    s = s[gain_filter(s)]
                    rows.append(
                        {
                            "support_filter": support_name,
                            "gain_filter": gain_name,
                            "backbone": backbone,
                            "traffic_state_transition": state,
                            "n_detectors": len(s),
                            "r_reducibility_degradation": residualized_corr(
                                s, "reducibility", "gain_lost"
                            ),
                            "partial_r_reducibility_given_full_mae": residualized_corr(
                                s, "reducibility", "gain_lost", True
                            ),
                            "r_loo_reducibility_degradation": residualized_corr(
                                s, "loo_reducibility", "gain_lost"
                            ),
                            "r_full_gain_degradation": residualized_corr(
                                s, "full_gain", "gain_lost"
                            ),
                            "r_loo_full_gain_degradation": residualized_corr(
                                s, "loo_full_gain", "gain_lost"
                            ),
                            "r_D_degradation": residualized_corr(s, "D", "gain_lost"),
                        }
                    )
    return pd.DataFrame(rows)


def selector_panel(base_panel: pd.DataFrame, method: str) -> pd.DataFrame:
    """Construct the same outcome panel for an archived reduced-data selector."""
    keys = ["dataset", "sensor", "traffic_state_transition"]
    keys_model = keys + ["backbone"]
    mae = pd.read_csv(MAE_CSV)
    audit = pd.read_csv(OUT / "fd_sensor_resolved_by_detector.csv")
    full = (
        mae[mae.method == "full"]
        .groupby(keys_model, as_index=False)
        .mae.mean()
        .rename(columns={"mae": "full_mae"})
    )
    reduced = (
        mae[(mae.method == method) & (mae.ratio == 0.3)]
        .groupby(keys_model, as_index=False)
        .mae.mean()
        .rename(columns={"mae": "red_mae"})
    )
    difficulty = (
        audit[
            (audit.method == method)
            & (audit.ratio == 0.3)
            & (audit.K_label == "K50")
        ]
        .groupby(keys, as_index=False)
        .D.mean()
    )
    invariant = base_panel[keys_model + ["persist_mae", "n_windows"]].drop_duplicates(
        keys_model
    )
    panel = (
        reduced.merge(full, on=keys_model)
        .merge(difficulty, on=keys)
        .merge(invariant, on=keys_model)
    )
    panel["method"] = method
    panel["gain_lost"] = panel.red_mae - panel.full_mae
    panel["full_gain"] = panel.persist_mae - panel.full_mae
    panel["reducibility"] = panel.full_gain / panel.persist_mae
    group = panel.groupby(keys)
    for metric in ("reducibility", "full_gain"):
        count = group[metric].transform("count")
        panel[f"loo_{metric}"] = (group[metric].transform("sum") - panel[metric]) / (
            count - 1
        )
    return panel.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["reducibility", "loo_reducibility", "gain_lost", "D"]
    )


def selector_robustness(
    base_panel: pd.DataFrame, reps: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Repeat the detector-state association for K-medoids and random selection."""
    cell_rows, state_rows = [], []
    methods = ("k_medoids", "random")
    specs = [
        ("reducibility", "r_reducibility_degradation"),
        ("loo_reducibility", "r_loo_reducibility_degradation"),
        ("D", "r_D_degradation"),
    ]
    for method_idx, method in enumerate(methods):
        panel = selector_panel(base_panel, method)
        if method == "k_medoids":
            check = panel.merge(
                base_panel[KEY + ["gain_lost"]], on=KEY, suffixes=("", "_base")
            )
            if not np.allclose(check.gain_lost, check.gain_lost_base):
                raise AssertionError("Reconstructed K-medoids panel disagrees with main panel")
        for backbone in BACKBONES:
            for state_idx, state in enumerate(STATES):
                d = panel[
                    (panel.backbone == backbone)
                    & (panel.traffic_state_transition == state)
                ]
                row = {
                    "method": method,
                    "backbone": backbone,
                    "traffic_state_transition": state,
                    "n_detectors": len(d),
                    "gain_lost_mean": equal_dataset_mean(d, "gain_lost"),
                }
                for spec_idx, (x, name) in enumerate(specs):
                    row[name] = residualized_corr(d, x, "gain_lost")
                    low, high = bootstrap_corr(
                        d,
                        x,
                        "gain_lost",
                        reps,
                        RNG_SEED
                        + 900_000
                        + 100_000 * method_idx
                        + 10_000 * BACKBONES.index(backbone)
                        + 1000 * state_idx
                        + 100 * spec_idx,
                    )
                    row[f"{name}_ci_low"] = low
                    row[f"{name}_ci_high"] = high
                cell_rows.append(row)

        for state in STATES:
            d = panel[panel.traffic_state_transition == state]
            detector = d.groupby(["dataset", "sensor"], as_index=False)[
                ["D", "reducibility", "full_gain", "gain_lost"]
            ].mean()
            row = {
                "method": method,
                "traffic_state_transition": state,
                "n_detectors": len(detector),
            }
            for metric in ("D", "reducibility", "full_gain", "gain_lost"):
                row[metric] = equal_dataset_mean(detector, metric)
            row["gain_loss_fraction_ratio_of_means"] = (
                row["gain_lost"] / row["full_gain"]
            )
            state_rows.append(row)
    return pd.DataFrame(cell_rows), pd.DataFrame(state_rows)


def state_contrasts(panel: pd.DataFrame, reps: int) -> pd.DataFrame:
    metrics = ["D", "reducibility", "full_gain", "reduced_gain", "gain_lost"]
    rows = []
    for metric_idx, metric in enumerate(metrics):
        wide = panel.pivot_table(
            index=["dataset", "sensor", "backbone"],
            columns="traffic_state_transition",
            values=metric,
        )
        for state_idx, state in enumerate(STATES[1:]):
            d = wide[["free_to_free", state]].dropna().copy()
            d["difference"] = d[state] - d["free_to_free"]
            d = d.reset_index()
            detector = d.groupby(["dataset", "sensor"], as_index=False).difference.mean()
            estimate = equal_dataset_mean(detector, "difference")
            ci = bootstrap_equal_dataset_means(
                detector,
                ["difference"],
                reps,
                RNG_SEED + 50_000 + 1000 * metric_idx + state_idx,
            )
            rows.append(
                {
                    "metric": metric,
                    "contrast": f"{state}_minus_free_to_free",
                    "n_detectors": len(detector),
                    "mean_paired_difference": estimate,
                    "ci_low": ci[0, 0],
                    "ci_high": ci[1, 0],
                }
            )
    return pd.DataFrame(rows)


def exposure_table() -> pd.DataFrame:
    mae = pd.read_csv(MAE_CSV)
    support = (
        mae[mae.method == "full"]
        .groupby(["dataset", "sensor", "traffic_state_transition"], as_index=False)
        .n_windows.max()
    )
    rows = []
    for dataset, d in support.groupby("dataset"):
        total = d.n_windows.sum()
        for state in STATES:
            s = d[d.traffic_state_transition == state]
            rows.append(
                {
                    "dataset": dataset,
                    "traffic_state_transition": state,
                    "n_detectors_with_state": s.sensor.nunique(),
                    "window_detector_atoms": int(s.n_windows.sum()),
                    "exposure_share": float(s.n_windows.sum() / total),
                    "n_windows_min": int(s.n_windows.min()),
                    "n_windows_p25": float(s.n_windows.quantile(0.25)),
                    "n_windows_median": float(s.n_windows.median()),
                    "n_windows_p75": float(s.n_windows.quantile(0.75)),
                }
            )
    return pd.DataFrame(rows)


def timestamp_state_mixing(fits: pd.DataFrame) -> pd.DataFrame:
    """Quantify how often a timestamp has different detector-level states."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from fd_state_conditional_followups import test_code  # noqa: PLC0415

    rows = []
    for dataset in DATASETS:
        sensors, code = test_code(dataset, fits)
        valid = code >= 0
        n_valid = valid.sum(axis=1)
        counts = np.column_stack([(code == i).sum(axis=1) for i in range(4)])
        n_distinct = (counts > 0).sum(axis=1)
        keep = n_valid > 0
        majority_share = counts.max(axis=1)[keep] / n_valid[keep]
        majority_state = counts.argmax(axis=1)[keep]
        base = {
            "dataset": dataset,
            "n_detectors": len(sensors),
            "n_timestamp_windows": len(code),
            "fraction_with_any_resolved_detector": float(keep.mean()),
            "fraction_mixed_among_resolved": float((n_distinct[keep] > 1).mean()),
            "fraction_three_or_more_states": float((n_distinct[keep] >= 3).mean()),
            "mean_valid_detectors": float(n_valid[keep].mean()),
            "median_valid_detectors": float(np.median(n_valid[keep])),
            "mean_majority_share": float(majority_share.mean()),
            "p10_majority_share": float(np.quantile(majority_share, 0.10)),
            "median_majority_share": float(np.median(majority_share)),
            "p90_majority_share": float(np.quantile(majority_share, 0.90)),
        }
        for i, state in enumerate(STATES):
            base[f"fraction_majority_{state}"] = float((majority_state == i).mean())
            base[f"fraction_present_{state}"] = float((counts[keep, i] > 0).mean())
            base[f"mean_detector_share_{state}"] = float(
                np.mean(counts[keep, i] / n_valid[keep])
            )
        rows.append(base)
    return pd.DataFrame(rows)


def detector_raw_profiles(fits: pd.DataFrame) -> pd.DataFrame:
    """Raw training-split characteristics for accepted and rejected FD detectors."""
    frames = []
    for dataset in DATASETS:
        desc = json.loads((REPO / "datasets" / dataset / "desc.json").read_text())
        raw = np.memmap(
            REPO / "datasets" / dataset / "data.dat",
            dtype=np.float32,
            mode="r",
            shape=tuple(desc["shape"]),
        )
        q = np.asarray(raw[:TRAIN_STEPS, :, 0], dtype=np.float32)
        o = np.asarray(raw[:TRAIN_STEPS, :, 1], dtype=np.float32)
        v = np.asarray(raw[:TRAIN_STEPS, :, 2], dtype=np.float32)
        valid = np.isfinite(q) & np.isfinite(o) & np.isfinite(v) & (q > 0) & (v > 0)
        qn = np.where(valid, q, np.nan)
        on = np.where(valid, o, np.nan)
        vn = np.where(valid, v, np.nan)
        # Fully missing detectors are intentionally retained as NaN profiles;
        # they are a real part of the FD-screen selection profile.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            median_o = np.nanmedian(on, axis=0)
            median_v = np.nanmedian(vn, axis=0)
        congested_proxy = valid & (o > 1.2 * median_o[None, :]) & (v < 0.8 * median_v[None, :])
        n_valid = valid.sum(axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            profile = pd.DataFrame({
                "dataset": dataset,
                "sensor": np.arange(q.shape[1]),
                "valid_fraction": valid.mean(axis=0),
                "mean_flow": np.nanmean(qn, axis=0),
                "std_flow": np.nanstd(qn, axis=0),
                "p95_flow": np.nanpercentile(qn, 95, axis=0),
                "mean_occupancy": np.nanmean(on, axis=0),
                "std_occupancy": np.nanstd(on, axis=0),
                "p95_occupancy": np.nanpercentile(on, 95, axis=0),
                "mean_speed": np.nanmean(vn, axis=0),
                "std_speed": np.nanstd(vn, axis=0),
                "p05_speed": np.nanpercentile(vn, 5, axis=0),
                "low_speed_fraction": np.divide(
                    (valid & (v < 45)).sum(axis=0),
                    n_valid,
                    out=np.full(q.shape[1], np.nan),
                    where=n_valid > 0,
                ),
                "congested_proxy_fraction": np.divide(
                    congested_proxy.sum(axis=0),
                    n_valid,
                    out=np.full(q.shape[1], np.nan),
                    where=n_valid > 0,
                ),
            })
        profile["flow_cv"] = profile.std_flow / profile.mean_flow
        label = fits[fits.dataset == dataset][["sensor", "identifiable", "reason"]]
        frames.append(profile.merge(label, on="sensor", how="left", validate="one_to_one"))
        del q, o, v, qn, on, vn, valid, congested_proxy
    return pd.concat(frames, ignore_index=True)


def bh_adjust(pvalues: pd.Series) -> pd.Series:
    p = pvalues.to_numpy(float)
    out = np.full(len(p), np.nan)
    keep = np.flatnonzero(np.isfinite(p))
    if not len(keep):
        return pd.Series(out, index=pvalues.index)
    order = keep[np.argsort(p[keep])]
    ranked = p[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out[order] = np.minimum(ranked, 1.0)
    return pd.Series(out, index=pvalues.index)


def screening_comparison(profile: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "valid_fraction",
        "mean_flow",
        "std_flow",
        "flow_cv",
        "p95_flow",
        "mean_occupancy",
        "std_occupancy",
        "p95_occupancy",
        "mean_speed",
        "std_speed",
        "p05_speed",
        "low_speed_fraction",
        "congested_proxy_fraction",
    ]
    rows = []
    scopes = [(dataset, d.copy()) for dataset, d in profile.groupby("dataset")]
    pooled = profile.copy()
    for metric in metrics:
        pooled[f"z_{metric}"] = pooled.groupby("dataset")[metric].transform(
            lambda x: (x - x.mean()) / x.std(ddof=0)
        )
    scopes.append(("POOLED_DATASET_STANDARDIZED", pooled))
    for scope, d in scopes:
        for metric in metrics:
            value_col = f"z_{metric}" if scope == "POOLED_DATASET_STANDARDIZED" else metric
            a = d.loc[d.identifiable, value_col].dropna().to_numpy(float)
            r = d.loc[~d.identifiable, value_col].dropna().to_numpy(float)
            pooled_sd = math.sqrt((np.var(a, ddof=1) + np.var(r, ddof=1)) / 2)
            smd = (a.mean() - r.mean()) / pooled_sd if pooled_sd > 0 else np.nan
            test = mannwhitneyu(a, r, alternative="two-sided")
            rank_biserial = 2 * float(test.statistic) / (len(a) * len(r)) - 1
            rows.append(
                {
                    "scope": scope,
                    "metric": metric,
                    "n_accepted": len(a),
                    "n_rejected": len(r),
                    "accepted_mean": float(a.mean()),
                    "rejected_mean": float(r.mean()),
                    "accepted_median": float(np.median(a)),
                    "rejected_median": float(np.median(r)),
                    "standardized_mean_difference": smd,
                    "rank_biserial": rank_biserial,
                    "mannwhitney_p": float(test.pvalue),
                }
            )
    result = pd.DataFrame(rows)
    result["mannwhitney_q_bh_within_scope"] = result.groupby("scope").mannwhitney_p.transform(
        bh_adjust
    )
    return result


def screening_reasons(fits: pd.DataFrame) -> pd.DataFrame:
    rejected = fits[~fits.identifiable].copy()
    rejected["reason_component"] = rejected.reason.str.split(";")
    return (
        rejected.explode("reason_component")
        .groupby(["dataset", "reason_component"], as_index=False)
        .size()
        .rename(columns={"size": "n_detectors"})
    )


def make_figure(
    state: pd.DataFrame,
    model_state: pd.DataFrame,
    correlations: pd.DataFrame,
    exposure: pd.DataFrame,
):
    colors = ["#4C78A8", "#E45756", "#F2CF5B", "#72B7B2"]
    labels = [STATE_LABELS[x] for x in STATES]
    x = np.arange(len(STATES))
    s = state.set_index("traffic_state_transition").loc[list(STATES)]

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.4), constrained_layout=True)
    ax = axes[0, 0]
    ax.bar(x, s.reduced_gain, color=colors, alpha=0.80, label="Gain retained")
    ax.bar(x, s.gain_lost, bottom=s.reduced_gain, color="#B9B9B9", alpha=0.9, label="Gain lost")
    err = np.vstack([s.full_gain - s.full_gain_ci_low, s.full_gain_ci_high - s.full_gain])
    ax.errorbar(x, s.full_gain, yerr=err, fmt="none", ecolor="black", capsize=3, lw=1)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, labels, rotation=12)
    ax.set_ylabel("MAE gain over persistence")
    ax.set_title("a  Full-data gain = retained gain + gain lost")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[0, 1]
    ax.bar(x, s.D, color=colors, alpha=0.85)
    exposure_mean = exposure.groupby("traffic_state_transition").exposure_share.mean().reindex(STATES)
    for i, (d_value, share) in enumerate(zip(s.D, exposure_mean)):
        ax.text(i, d_value + 0.012, f"{share:.1%} atoms", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, labels, rotation=12)
    ax.set_ylabel("FD support difficulty, D")
    ax.set_title("b  Coverage difficulty and observed exposure")
    ax.set_ylim(0, max(s.D) * 1.28)

    ax = axes[1, 0]
    matrix = (
        correlations.pivot(
            index="backbone",
            columns="traffic_state_transition",
            values="r_loo_reducibility_degradation",
        )
        .reindex(index=BACKBONES, columns=STATES)
    )
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    for i, backbone in enumerate(BACKBONES):
        for j, state_name in enumerate(STATES):
            row = correlations[
                (correlations.backbone == backbone)
                & (correlations.traffic_state_transition == state_name)
            ].iloc[0]
            excludes_zero = (
                row.r_loo_reducibility_degradation_ci_low > 0
                or row.r_loo_reducibility_degradation_ci_high < 0
            )
            ax.text(j, i, f"{matrix.iloc[i, j]:+.2f}{'*' if excludes_zero else ''}", ha="center", va="center", fontsize=8)
    ax.set_xticks(x, labels, rotation=15)
    ax.set_yticks(np.arange(len(BACKBONES)), BACKBONES)
    ax.set_title("c  Within-state r(leave-one-model-out R, gain lost)")
    fig.colorbar(image, ax=ax, shrink=0.82, label="Pearson r")

    ax = axes[1, 1]
    ms = model_state.copy()
    for state_name, color in zip(STATES, colors):
        d = ms[ms.traffic_state_transition == state_name]
        sizes = 40 + 35 * np.maximum(d.gain_lost, 0)
        ax.scatter(d.D, d.full_gain, s=sizes, c=color, alpha=0.78, edgecolor="white", lw=0.7, label=STATE_LABELS[state_name])
    ax.set_xlabel("FD support difficulty, D")
    ax.set_ylabel("Full-data gain M")
    ax.set_title("d  Complementary coverage and gain axes\n(point size = mean gain lost)")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"fd_state_margin_reanalysis.{suffix}", dpi=220)
    plt.close(fig)


def print_summary(
    panel: pd.DataFrame,
    state: pd.DataFrame,
    corr: pd.DataFrame,
    sensitivity: pd.DataFrame,
    mixing: pd.DataFrame,
    screening: pd.DataFrame,
):
    display = state.set_index("traffic_state_transition").loc[list(STATES)]
    print("\n=== Equal-dataset, detector-cluster state summary ===")
    print(
        display[
            ["D", "reducibility", "full_gain", "reduced_gain", "gain_lost", "gain_loss_fraction_ratio_of_means"]
        ].round(3).to_string()
    )
    main = corr.r_reducibility_degradation
    ci_positive = corr.r_reducibility_degradation_ci_low > 0
    print("\n=== Main within-state association (dataset-adjusted) ===")
    print(
        f"positive point estimates: {(main > 0).sum()}/{len(main)}; "
        f"95% detector-bootstrap CI above zero: {ci_positive.sum()}/{len(main)}; "
        f"range={main.min():+.3f} to {main.max():+.3f}"
    )
    partial = corr.partial_r_reducibility_given_full_mae
    partial_ci = corr.partial_r_reducibility_given_full_mae_ci_low > 0
    print(
        f"after full-MAE + dataset adjustment: positive={(partial > 0).sum()}/{len(partial)}, "
        f"CI above zero={partial_ci.sum()}/{len(partial)}"
    )
    loo = corr.r_loo_reducibility_degradation
    loo_ci = corr.r_loo_reducibility_degradation_ci_low > 0
    print(
        f"leave-one-model-out R: positive={(loo > 0).sum()}/{len(loo)}, "
        f"CI above zero={loo_ci.sum()}/{len(loo)}, median r={loo.median():+.3f}"
    )
    base = sensitivity[
        (sensitivity.support_filter == "min50")
        & (sensitivity.gain_filter == "all_gains")
    ]
    print(
        f"min 50 windows: positive r(R, Delta)={(base.r_reducibility_degradation > 0).sum()}/"
        f"{base.r_reducibility_degradation.notna().sum()} model-state cells"
    )
    print("\n=== Timestamp mixing ===")
    print(
        mixing[
            ["dataset", "fraction_mixed_among_resolved", "mean_majority_share", "median_valid_detectors"]
        ].round(3).to_string(index=False)
    )
    pooled = screening[screening.scope == "POOLED_DATASET_STANDARDIZED"].copy()
    pooled = pooled.reindex(pooled.standardized_mean_difference.abs().sort_values(ascending=False).index)
    print("\n=== Largest accepted-vs-rejected detector profile differences ===")
    print(
        pooled[["metric", "standardized_mean_difference", "rank_biserial", "mannwhitney_q_bh_within_scope"]]
        .head(6).round(3).to_string(index=False)
    )
    print(f"\nPanel rows: {len(panel)}; outputs written to {OUT}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument(
        "--skip-raw-profiles",
        action="store_true",
        help="Reuse an existing detector profile CSV instead of scanning raw training data.",
    )
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    panel = load_panel()
    fits = pd.read_csv(FITS_CSV)
    state, model_state = state_summaries(panel, args.bootstrap_reps)
    corr, corr_by_dataset = correlation_table(panel, args.bootstrap_reps)
    sensitivity = sensitivity_table(panel)
    selector_cells, selector_states = selector_robustness(panel, args.bootstrap_reps)
    contrasts = state_contrasts(panel, args.bootstrap_reps)
    exposure = exposure_table()
    mixing = timestamp_state_mixing(fits)

    profile_path = OUT / "fd_detector_screening_profile.csv"
    if args.skip_raw_profiles and profile_path.exists():
        profile = pd.read_csv(profile_path)
    else:
        profile = detector_raw_profiles(fits)
    screening = screening_comparison(profile)
    reasons = screening_reasons(fits)

    panel.to_csv(OUT / "fd_state_margin_panel.csv", index=False)
    state.to_csv(OUT / "fd_state_margin_by_state.csv", index=False)
    model_state.to_csv(OUT / "fd_state_margin_by_model_state.csv", index=False)
    corr.to_csv(OUT / "fd_state_margin_correlations.csv", index=False)
    corr_by_dataset.to_csv(OUT / "fd_state_margin_correlations_by_dataset.csv", index=False)
    sensitivity.to_csv(OUT / "fd_state_margin_sensitivity.csv", index=False)
    selector_cells.to_csv(OUT / "fd_state_margin_selector_robustness.csv", index=False)
    selector_states.to_csv(
        OUT / "fd_state_margin_selector_state_summary.csv", index=False
    )
    contrasts.to_csv(OUT / "fd_state_margin_contrasts.csv", index=False)
    exposure.to_csv(OUT / "fd_traffic_state_exposure.csv", index=False)
    mixing.to_csv(OUT / "fd_timestamp_state_mixing_summary.csv", index=False)
    profile.to_csv(profile_path, index=False)
    screening.to_csv(OUT / "fd_detector_screening_comparison.csv", index=False)
    reasons.to_csv(OUT / "fd_detector_screening_reasons.csv", index=False)
    make_figure(state, model_state, corr, exposure)
    print_summary(panel, state, corr, sensitivity, mixing, screening)


if __name__ == "__main__":
    main()
