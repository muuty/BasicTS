#!/usr/bin/env python3
"""Within-detector, across-budget test of coverage difficulty vs reduction loss.

RESEARCH QUESTION
    Does coverage difficulty ``D`` predict the state-conditional reduction loss
    ``Delta = e_r - e_f`` WITHIN a detector as the sampling budget (ratio)
    changes?  The paper currently reports only a fixed-budget CROSS-detector
    correlation ``r(D, Delta) ~ 0``.  Here we build the within-detector,
    across-ratio panel, which is more causal: when a given detector-state gets
    more/less budget, do its coverage and its loss co-move, and -- crucially --
    does coverage carry any signal BEYOND the raw budget?

WHAT THIS REUSES (does not reinvent)
    * ``fd_state_conditional_mae.py`` -- state labelling on the TEST split and
      per-(detector, state) masked MAE from each run's ``test_results.npz``.
    * ``fd_sensor_resolved.py`` -- coverage difficulty ``D`` per detector-state
      for a k_medoids coreset at a given ratio (K50 curve, to match the paper).
    * Existing artifacts, reused rather than recomputed:
        - ``fd_state_conditional_mae_by_detector.csv``  (MAE at ratio 0.3 and
          the full-data reference; both from phase_c_method_comparison).
        - ``fd_sensor_resolved_by_detector.csv``  (D at ratios 0.1 and 0.3).
      Only the MISSING pieces are computed: MAE at {0.1, 0.5, 0.7, 0.9} and
      D at {0.5, 0.7, 0.9}.

CONSISTENCY NOTES
    * DCRNN exists in three checkpoint roots (method_comparison [curriculum
      learning], extra_ratios, and dcrnn_no_cl [no curriculum]).  The reused
      ratio-0.3 and full-data MAE rows come from method_comparison (the CL
      variant), so every DCRNN ratio here is sourced from method_comparison
      (0.5, 0.7) and extra_ratios (0.1, 0.9) to keep a SINGLE training regime
      across ratios.  dcrnn_no_cl is deliberately NOT used, otherwise a
      detector's across-ratio trajectory would mix training regimes -- exactly
      the axis this analysis varies.
    * D is model-free: the same D applies to every backbone at a given
      (dataset, detector, state, ratio).

OUTPUTS (experiments/result/analysis/)
    * fd_within_detector_ratio_panel.csv   -- tidy panel (one row per
      dataset,detector,state,backbone,ratio) with e_f, e_r, Delta, D.
    * fd_within_detector_ratio_summary.csv -- per-state association statistics.
    * fd_within_detector_D_extra.csv       -- computed D at 0.5/0.7/0.9 (cache).
    * a printed paragraph answering the question per state and overall.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
ANALYSIS = REPO / "scripts" / "analysis"
sys.path.insert(0, str(ANALYSIS))

import fd_sensor_resolved as fsr  # noqa: E402
import fd_state_conditional_mae as mae_mod  # noqa: E402

OUT = REPO / "experiments" / "result" / "analysis"
MAE_BY_DETECTOR = OUT / "fd_state_conditional_mae_by_detector.csv"
D_BY_DETECTOR = OUT / "fd_sensor_resolved_by_detector.csv"

RATIOS = (0.1, 0.3, 0.5, 0.7, 0.9)
# Which checkpoint root supplies the k_medoids MAE at each ratio that is not
# already in the reused CSV (0.3 and full come from the reused CSV).
MAE_RATIO_ROOT = {
    0.1: "phase_c_extra_ratios",
    0.5: "phase_c_method_comparison",
    0.7: "phase_c_method_comparison",
    0.9: "phase_c_extra_ratios",
}
D_NEW_RATIOS = (0.5, 0.7, 0.9)  # 0.1, 0.3 reused from existing D CSV
D_SEEDS = (42, 123, 456)
K_LABEL = "K50"
STATES = ("free_to_free", "breakdown", "recovery", "congested_to_congested")
CONGESTED_STATES = ("breakdown", "recovery", "congested_to_congested")


# --------------------------------------------------------------------------- #
# Part 1: state-conditional MAE for the missing ratios
# --------------------------------------------------------------------------- #
def leaves_for(root: str, backbone_dir: str, dataset: str,
               strategy: str, ratio: float) -> list[Path]:
    base = (REPO / "checkpoints" / root / backbone_dir / "xtraffic"
            / f"{dataset}_50_12_12")
    hits: list[Path] = []
    for npz in sorted(base.glob("*/*/test_results.npz")):
        cfg = npz.parent / "cfg.txt"
        if not cfg.exists():
            continue
        strat, r = mae_mod.parse_cfg(cfg)
        if strat == strategy and abs(r - ratio) < 1e-6:
            hits.append(npz)
    return hits


def compute_missing_mae(datasets, backbones, max_seeds: int) -> pd.DataFrame:
    fits = pd.read_csv(mae_mod.FITS_CSV)
    min_valid = int(np.ceil(mae_mod.T_IN * (2 / 3)))
    rows: list[dict] = []
    for dataset in datasets:
        accepted = (fits[(fits.dataset == dataset) & fits.identifiable]
                    .sort_values("sensor").reset_index(drop=True))
        sensors, code = mae_mod.test_transitions(dataset, accepted, min_valid, 0.75)
        print(f"[MAE {dataset}] detectors={len(sensors)}", flush=True)
        for backbone in backbones:
            backbone_dir = mae_mod.BACKBONE_DIRS[backbone]
            for ratio, root in MAE_RATIO_ROOT.items():
                paths = leaves_for(root, backbone_dir, dataset, "k_medoids", ratio)
                if not paths:
                    print(f"  {backbone:11s} r={ratio} [{root}]: NO npz", flush=True)
                    continue
                wd_mae = mae_mod.window_detector_mae(paths, sensors, max_seeds)
                rows.extend(mae_mod.detector_state_rows(
                    wd_mae, code, sensors, dataset, backbone, "k_medoids", ratio))
                print(f"  {backbone:11s} r={ratio} [{root}]: "
                      f"{min(len(paths), max_seeds)}/{len(paths)} runs", flush=True)
    return pd.DataFrame(rows)


def assemble_mae_panel(datasets, backbones, max_seeds: int) -> pd.DataFrame:
    """k_medoids MAE at all 5 ratios (e_r) merged with full-data MAE (e_f)."""
    existing = pd.read_csv(MAE_BY_DETECTOR)
    existing = existing[existing.dataset.isin(datasets) & existing.backbone.isin(backbones)]
    km_03 = existing[(existing.method == "k_medoids") & (existing.ratio == 0.3)]
    full = existing[existing.method == "full"]

    new = compute_missing_mae(datasets, backbones, max_seeds)
    km = pd.concat([km_03, new], ignore_index=True)
    km = km.rename(columns={"mae": "e_r"})[
        ["dataset", "backbone", "sensor", "traffic_state_transition", "ratio", "e_r", "n_windows"]]

    ef = (full.rename(columns={"mae": "e_f"})
          [["dataset", "backbone", "sensor", "traffic_state_transition", "e_f"]])
    panel = km.merge(ef, on=["dataset", "backbone", "sensor", "traffic_state_transition"], how="inner")
    panel["Delta"] = panel["e_r"] - panel["e_f"]
    return panel


# --------------------------------------------------------------------------- #
# Part 2: coverage difficulty D for the missing ratios
# --------------------------------------------------------------------------- #
def compute_missing_D(datasets, device: str) -> pd.DataFrame:
    cache = OUT / "fd_within_detector_D_extra.csv"
    if cache.exists():
        print(f"[D] reusing cache {cache.name}", flush=True)
        return pd.read_csv(cache)
    fits = pd.read_csv(mae_mod.FITS_CSV)
    all_rows: list[pd.DataFrame] = []
    for dataset in datasets:
        raw = fsr.load_raw(dataset)
        accepted = (fits[(fits.dataset == dataset) & fits.identifiable]
                    .sort_values("sensor").reset_index(drop=True))
        obs = fsr.classify_observations(raw, accepted)
        transition = fsr.transition_codes(obs, min_valid_fraction=2 / 3, min_agreement=0.75)
        print(f"[D {dataset}] detectors={len(accepted)} device={device}", flush=True)
        rows = fsr.analyse_support(
            dataset, raw, accepted, transition,
            methods=("k_medoids",), ratios=D_NEW_RATIOS, seeds=D_SEEDS,
            device=device, sensor_batch=8, query_batch=256)
        all_rows.append(rows)
    out = pd.concat(all_rows, ignore_index=True)
    out.to_csv(cache, index=False)
    print(f"[D] wrote {cache.name}", flush=True)
    return out


def assemble_D_panel(datasets) -> pd.DataFrame:
    """D at all 5 ratios, K50 k_medoids, averaged over selection seeds."""
    existing = pd.read_csv(D_BY_DETECTOR)
    existing = existing[(existing.method == "k_medoids") & (existing.K_label == K_LABEL)
                        & existing.dataset.isin(datasets) & existing.ratio.isin([0.1, 0.3])]
    new = compute_missing_D(datasets, device_str())
    new = new[(new.method == "k_medoids") & (new.K_label == K_LABEL)]
    cols = ["dataset", "sensor", "traffic_state_transition", "ratio", "D"]
    both = pd.concat([existing[cols + ["selection_seed"]], new[cols + ["selection_seed"]]],
                     ignore_index=True)
    dpan = (both.groupby(["dataset", "sensor", "traffic_state_transition", "ratio"], as_index=False)
            .agg(D=("D", "mean"), n_seeds=("selection_seed", "nunique")))
    return dpan


def device_str() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# --------------------------------------------------------------------------- #
# Part 3: statistics
# --------------------------------------------------------------------------- #
def ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean()
    denom = float((x * x).sum())
    return float((x * y).sum() / denom) if denom > 0 else np.nan


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def cluster_bootstrap_ci(values_by_cluster: dict, stat_fn, n_boot=2000, seed=0):
    """Percentile CI resampling clusters (detectors) with replacement."""
    rng = np.random.default_rng(seed)
    clusters = list(values_by_cluster.keys())
    if not clusters:
        return (np.nan, np.nan)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(len(clusters), size=len(clusters), replace=True)
        pooled = np.concatenate([values_by_cluster[clusters[i]] for i in pick])
        pooled = pooled[np.isfinite(pooled)]
        if len(pooled):
            boots.append(stat_fn(pooled))
    if not boots:
        return (np.nan, np.nan)
    return (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


def within_detector_analysis(panel: pd.DataFrame) -> pd.DataFrame:
    """(a) Per (dataset,sensor,backbone,state) slope/corr of Delta vs D across ratios."""
    recs = []
    grp = panel.dropna(subset=["D", "Delta"]).groupby(
        ["dataset", "sensor", "backbone", "traffic_state_transition"])
    for (ds, sensor, bb, state), g in grp:
        if g.ratio.nunique() < 3:
            continue
        g = g.sort_values("ratio")
        recs.append({
            "dataset": ds, "sensor": sensor, "backbone": bb,
            "traffic_state_transition": state, "n_ratios": g.ratio.nunique(),
            "slope_Delta_on_D": ols_slope(g.D.to_numpy(), g.Delta.to_numpy()),
            "corr_Delta_D": pearson(g.D.to_numpy(), g.Delta.to_numpy()),
        })
    return pd.DataFrame(recs)


def summarise_within(within: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for state in STATES + ("ALL_congested", "ALL"):
        if state == "ALL":
            sub = within
        elif state == "ALL_congested":
            sub = within[within.traffic_state_transition.isin(CONGESTED_STATES)]
        else:
            sub = within[within.traffic_state_transition == state]
        sub = sub.dropna(subset=["slope_Delta_on_D"])
        if sub.empty:
            continue
        slopes = sub.slope_Delta_on_D.to_numpy()
        corrs = sub.corr_Delta_D.dropna().to_numpy()
        by_cluster = {c: gg.slope_Delta_on_D.to_numpy()
                      for c, gg in sub.groupby(["dataset", "sensor"])}
        lo, hi = cluster_bootstrap_ci(by_cluster, np.nanmedian)
        rows.append({
            "state": state, "n_groups": len(sub),
            "mean_slope": float(np.nanmean(slopes)),
            "median_slope": float(np.nanmedian(slopes)),
            "median_slope_ci_lo": lo, "median_slope_ci_hi": hi,
            "frac_slope_positive": float((slopes > 0).mean()),
            "mean_corr": float(np.nanmean(corrs)) if len(corrs) else np.nan,
            "median_corr": float(np.nanmedian(corrs)) if len(corrs) else np.nan,
            "frac_corr_positive": float((corrs > 0).mean()) if len(corrs) else np.nan,
        })
    return pd.DataFrame(rows)


def fixed_budget_analysis(panel: pd.DataFrame) -> pd.DataFrame:
    """(b) Cross-detector r(D,Delta) at each ratio, per (dataset,backbone,state)."""
    recs = []
    for (ds, bb, state, ratio), g in panel.dropna(subset=["D", "Delta"]).groupby(
            ["dataset", "backbone", "traffic_state_transition", "ratio"]):
        if g.sensor.nunique() < 5:
            continue
        recs.append({
            "dataset": ds, "backbone": bb, "traffic_state_transition": state,
            "ratio": ratio, "n_detectors": g.sensor.nunique(),
            "r_cross_detector": pearson(g.D.to_numpy(), g.Delta.to_numpy()),
        })
    cell = pd.DataFrame(recs)
    summary = (cell.groupby(["traffic_state_transition", "ratio"], as_index=False)
               .agg(median_r=("r_cross_detector", "median"),
                    mean_r=("r_cross_detector", "mean"),
                    n_cells=("r_cross_detector", "size"),
                    frac_r_positive=("r_cross_detector", lambda s: float((s > 0).mean()))))
    return cell, summary


def within_demean(df: pd.DataFrame, cols, by) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c + "_dm"] = out[c] - out.groupby(by)[c].transform("mean")
    return out


def partial_mediation(panel: pd.DataFrame) -> pd.DataFrame:
    """(c) Delta ~ ratio + D with detector*backbone fixed effects, by state.

    Within transformation demeans Delta, ratio, D by (dataset,sensor,backbone).
    The D coefficient isolates whether coverage adds signal beyond the budget.
    A detector-clustered bootstrap gives the CI and a partial correlation gives
    a scale-free effect size.
    """
    fe = ["dataset", "sensor", "backbone"]
    recs = []
    for state in STATES + ("ALL_congested", "ALL"):
        if state == "ALL":
            sub = panel
        elif state == "ALL_congested":
            sub = panel[panel.traffic_state_transition.isin(CONGESTED_STATES)]
        else:
            sub = panel[panel.traffic_state_transition == state]
        sub = sub.dropna(subset=["D", "Delta", "ratio"]).copy()
        # keep only groups with within variation
        sub = sub.groupby(fe).filter(lambda g: g.ratio.nunique() >= 3)
        if len(sub) < 20:
            continue
        dm = within_demean(sub, ["Delta", "ratio", "D"], fe)
        X = np.column_stack([dm.ratio_dm.to_numpy(), dm.D_dm.to_numpy()])
        y = dm.Delta_dm.to_numpy()
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)  # no intercept (demeaned)
        b_ratio, b_D = float(beta[0]), float(beta[1])

        # partial correlation of D and Delta given ratio (within FE)
        rx = dm.D_dm.to_numpy() - ols_slope(dm.ratio_dm.to_numpy(), dm.D_dm.to_numpy()) * (
            dm.ratio_dm.to_numpy() - dm.ratio_dm.mean())
        ry = dm.Delta_dm.to_numpy() - ols_slope(dm.ratio_dm.to_numpy(), dm.Delta_dm.to_numpy()) * (
            dm.ratio_dm.to_numpy() - dm.ratio_dm.mean())
        partial_r = pearson(rx, ry)

        # cluster bootstrap CI on b_D and partial_r, clustering by (dataset,sensor)
        clusters = {c: idx.to_numpy() for c, idx in sub.reset_index().groupby(["dataset", "sensor"]).groups.items()}
        # rebuild with positional indices into sub
        sub_r = sub.reset_index(drop=True)
        cl_pos = {c: g.index.to_numpy() for c, g in sub_r.groupby(["dataset", "sensor"])}
        rng = np.random.default_rng(0)
        keys = list(cl_pos.keys())
        bD_boot, pr_boot = [], []
        for _ in range(1000):
            pick = rng.choice(len(keys), size=len(keys), replace=True)
            rows_idx = np.concatenate([cl_pos[keys[i]] for i in pick])
            bsub = sub_r.iloc[rows_idx]
            bdm = within_demean(bsub, ["Delta", "ratio", "D"], fe)
            Xi = np.column_stack([bdm.ratio_dm.to_numpy(), bdm.D_dm.to_numpy()])
            yi = bdm.Delta_dm.to_numpy()
            if np.linalg.matrix_rank(Xi) < 2:
                continue
            bi, *_ = np.linalg.lstsq(Xi, yi, rcond=None)
            bD_boot.append(float(bi[1]))
            rxi = bdm.D_dm.to_numpy() - ols_slope(bdm.ratio_dm.to_numpy(), bdm.D_dm.to_numpy()) * (
                bdm.ratio_dm.to_numpy() - bdm.ratio_dm.mean())
            ryi = bdm.Delta_dm.to_numpy() - ols_slope(bdm.ratio_dm.to_numpy(), bdm.Delta_dm.to_numpy()) * (
                bdm.ratio_dm.to_numpy() - bdm.ratio_dm.mean())
            pr_boot.append(pearson(rxi, ryi))
        bD_ci = (float(np.percentile(bD_boot, 2.5)), float(np.percentile(bD_boot, 97.5))) if bD_boot else (np.nan, np.nan)
        pr_ci = (float(np.nanpercentile(pr_boot, 2.5)), float(np.nanpercentile(pr_boot, 97.5))) if pr_boot else (np.nan, np.nan)

        recs.append({
            "state": state, "n_obs": len(sub), "n_groups": sub.groupby(fe).ngroups,
            "coef_ratio": b_ratio, "coef_D": b_D,
            "coef_D_ci_lo": bD_ci[0], "coef_D_ci_hi": bD_ci[1],
            "partial_r_D_given_ratio": partial_r,
            "partial_r_ci_lo": pr_ci[0], "partial_r_ci_hi": pr_ci[1],
            "D_survives": bool((bD_ci[0] > 0) or (bD_ci[1] < 0)),
        })
    return pd.DataFrame(recs)


def sanity_trends(panel: pd.DataFrame) -> pd.DataFrame:
    """(3) Within-detector D-vs-ratio and Delta-vs-ratio slopes, by state."""
    fe = ["dataset", "sensor", "backbone"]
    recs = []
    for state in STATES + ("ALL_congested", "ALL"):
        if state == "ALL":
            sub = panel
        elif state == "ALL_congested":
            sub = panel[panel.traffic_state_transition.isin(CONGESTED_STATES)]
        else:
            sub = panel[panel.traffic_state_transition == state]
        sub = sub.dropna(subset=["D", "Delta", "ratio"]).copy()
        sub = sub.groupby(fe).filter(lambda g: g.ratio.nunique() >= 3)
        if len(sub) < 20:
            continue
        dm = within_demean(sub, ["Delta", "ratio", "D"], fe)
        recs.append({
            "state": state,
            "slope_D_on_ratio": ols_slope(dm.ratio_dm.to_numpy(), dm.D_dm.to_numpy()),
            "slope_Delta_on_ratio": ols_slope(dm.ratio_dm.to_numpy(), dm.Delta_dm.to_numpy()),
        })
    return pd.DataFrame(recs)


# --------------------------------------------------------------------------- #
def report(within_sum, fixed_sum, partial, sanity) -> None:
    def g(df, state, col):
        r = df[df.state == state]
        return r[col].iloc[0] if len(r) else np.nan

    print("\n" + "=" * 78)
    print("WITHIN-DETECTOR, ACROSS-BUDGET TEST OF COVERAGE DIFFICULTY D vs LOSS Delta")
    print("=" * 78)

    print("\n[3] SANITY -- common budget trend (within-detector slopes vs ratio):")
    for _, r in sanity.iterrows():
        print(f"  {r.state:22s} dD/dratio={r.slope_D_on_ratio:+.4f}  "
              f"dDelta/dratio={r.slope_Delta_on_ratio:+.3f}")

    print("\n[a] WITHIN-DETECTOR across-ratio Delta-vs-D slope (per detector-backbone-state):")
    for _, r in within_sum.iterrows():
        print(f"  {r.state:22s} n={int(r.n_groups):4d}  median_slope={r.median_slope:+.3f} "
              f"[{r.median_slope_ci_lo:+.3f},{r.median_slope_ci_hi:+.3f}]  "
              f"frac_slope>0={r.frac_slope_positive:.2f}  median_corr={r.median_corr:+.3f}")

    print("\n[b] FIXED-BUDGET cross-detector r(D,Delta), median over backbone*dataset cells:")
    for state in STATES:
        sub = fixed_sum[fixed_sum.traffic_state_transition == state].sort_values("ratio")
        if sub.empty:
            continue
        cells = "  ".join(f"r@{rw.ratio:.1f}={rw.median_r:+.2f}" for _, rw in sub.iterrows())
        print(f"  {state:22s} {cells}")

    print("\n[c] PARTIAL/MEDIATION -- Delta ~ ratio + D, detector*backbone FE:")
    for _, r in partial.iterrows():
        verdict = "SURVIVES" if r.D_survives else "null (CI incl. 0)"
        print(f"  {r.state:22s} coef_D={r.coef_D:+.3f} "
              f"[{r.coef_D_ci_lo:+.3f},{r.coef_D_ci_hi:+.3f}]  "
              f"partial_r={r.partial_r_D_given_ratio:+.3f} "
              f"[{r.partial_r_ci_lo:+.3f},{r.partial_r_ci_hi:+.3f}]  "
              f"(coef_ratio={r.coef_ratio:+.3f})  -> {verdict}")

    # Prose paragraph
    print("\n" + "-" * 78)
    print("ANSWER")
    print("-" * 78)
    cong_slope = g(within_sum, "ALL_congested", "median_slope")
    cong_slope_lo = g(within_sum, "ALL_congested", "median_slope_ci_lo")
    cong_slope_hi = g(within_sum, "ALL_congested", "median_slope_ci_hi")
    cong_fpos = g(within_sum, "ALL_congested", "frac_slope_positive")
    cong_pr = g(partial, "ALL_congested", "partial_r_D_given_ratio")
    cong_pr_lo = g(partial, "ALL_congested", "partial_r_ci_lo")
    cong_pr_hi = g(partial, "ALL_congested", "partial_r_ci_hi")
    cong_surv = g(partial, "ALL_congested", "D_survives")
    print(
        f"Raw within-detector across-budget slope of Delta on D (congested states pooled) "
        f"is median {cong_slope:+.3f} (detector-clustered 95% CI [{cong_slope_lo:+.3f},"
        f"{cong_slope_hi:+.3f}]), positive in {cong_fpos:.0%} of detector-backbone groups. "
        f"But both D and Delta fall mechanically with budget (see sanity slopes), so this raw "
        f"co-movement is confounded by the common budget trend. Controlling for ratio with "
        f"detector*backbone fixed effects, the partial correlation of D with Delta given ratio "
        f"is {cong_pr:+.3f} (CI [{cong_pr_lo:+.3f},{cong_pr_hi:+.3f}]); the D coefficient "
        f"{'DOES' if cong_surv else 'does NOT'} survive controlling for the budget itself. "
        f"{'Coverage difficulty carries signal beyond budget.' if cong_surv else 'Coverage difficulty adds nothing beyond the budget: a null.'}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=list(mae_mod.DATASETS))
    ap.add_argument("--backbones", nargs="+", default=list(mae_mod.BACKBONE_DIRS))
    ap.add_argument("--max-seeds", type=int, default=3)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    mae_panel = assemble_mae_panel(args.datasets, args.backbones, args.max_seeds)
    d_panel = assemble_D_panel(args.datasets)
    panel = mae_panel.merge(
        d_panel, on=["dataset", "sensor", "traffic_state_transition", "ratio"], how="left")
    panel = panel.rename(columns={"sensor": "detector", "traffic_state_transition": "state"})
    panel = panel[["dataset", "detector", "state", "backbone", "ratio",
                   "e_f", "e_r", "Delta", "D", "n_windows", "n_seeds"]].sort_values(
        ["dataset", "detector", "state", "backbone", "ratio"]).reset_index(drop=True)
    panel.to_csv(OUT / "fd_within_detector_ratio_panel.csv", index=False)
    print(f"\nwrote panel: {len(panel)} rows, "
          f"{panel.dropna(subset=['D','Delta']).shape[0]} with both D and Delta", flush=True)

    # rename back for analysis helpers that expect original column names
    ap_panel = panel.rename(columns={"detector": "sensor", "state": "traffic_state_transition"})
    within = within_detector_analysis(ap_panel)
    within_sum = summarise_within(within)
    fixed_cell, fixed_sum = fixed_budget_analysis(ap_panel)
    partial = partial_mediation(ap_panel)
    sanity = sanity_trends(ap_panel)

    within_sum.to_csv(OUT / "fd_within_detector_ratio_summary.csv", index=False)
    fixed_sum.to_csv(OUT / "fd_within_detector_fixedbudget_summary.csv", index=False)
    partial.to_csv(OUT / "fd_within_detector_partial.csv", index=False)
    sanity.to_csv(OUT / "fd_within_detector_sanity.csv", index=False)

    report(within_sum, fixed_sum, partial, sanity)
    print("\nwrote fd_within_detector_ratio_{panel,summary}.csv, "
          "fd_within_detector_{fixedbudget_summary,partial,sanity}.csv", flush=True)


if __name__ == "__main__":
    main()
