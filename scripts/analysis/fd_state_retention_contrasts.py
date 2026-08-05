#!/usr/bin/env python3
"""Paired per-detector contrasts on margin retention between traffic states.

Delta = M(1-kappa) splits the reduction loss into how much forecasting gain was
at stake (M) and what share of it was lost (1-kappa). M is close to definitional
as a driver of Delta, so the retention factor carries the non-mechanical part of
the state contrast. The paper reports kappa per state as a point estimate only;
this script gives every pairwise state contrast an interval, using the physical
detector as the resampling unit.

For each detector, quantities are averaged over architectures first, so the five
architectures do not count as five observations. Contrasts are paired within
detector and the bootstrap resamples detectors, stratified by dataset. Cells with
a non-positive margin are dropped, because the retention ratio is undefined there.

Writes fd_state_retention_contrasts.csv.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from fd_state_conditional_followups import OUT, ORDER

PANEL = OUT / "fd_state_margin_panel.csv"
N_BOOT = 10_000
SEED = 42
LABEL = {"free_to_free": "free", "breakdown": "breakdown",
         "recovery": "recovery", "congested_to_congested": "sustained"}


def paired_bootstrap(wide: pd.DataFrame, a: str, b: str, col: str, rng):
    """Detector-clustered, dataset-stratified bootstrap of the paired mean of a-b."""
    sub = wide[[(col, a), (col, b)]].dropna()
    if len(sub) < 10:
        return None
    diff = (sub[(col, a)] - sub[(col, b)]).to_numpy()
    ds = wide.loc[sub.index].index.get_level_values("dataset").to_numpy()
    groups = [np.where(ds == d)[0] for d in np.unique(ds)]
    boot = np.empty(N_BOOT)
    for i in range(N_BOOT):
        # equal weight per dataset, resample detectors within dataset
        means = [diff[g[rng.integers(0, len(g), len(g))]].mean() for g in groups]
        boot[i] = float(np.mean(means))
    point = float(np.mean([diff[g].mean() for g in groups]))
    return {
        "n_detectors": int(len(diff)),
        "mean_diff": point,
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "frac_positive": float((diff > 0).mean()),
    }


def main() -> None:
    p = pd.read_csv(PANEL)
    p["M"] = p.persist_mae - p.full_mae
    p["Delta"] = p.red_mae - p.full_mae
    p = p[p.M > 0].copy()                      # retention undefined otherwise
    p["retention"] = 1.0 - p.Delta / p.M
    p["share_lost"] = p.Delta / p.M

    # one value per physical detector and state: average architectures first
    det = (p.groupby(["dataset", "sensor", "traffic_state_transition"], as_index=False)
             .agg(retention=("retention", "mean"), share_lost=("share_lost", "mean"),
                  M=("M", "mean"), Delta=("Delta", "mean")))
    wide = det.pivot_table(index=["dataset", "sensor"],
                           columns="traffic_state_transition",
                           values=["retention", "share_lost", "M", "Delta"])

    rng = np.random.default_rng(SEED)
    rows = []
    for a, b in combinations(ORDER, 2):
        for col in ["retention", "M", "Delta"]:
            r = paired_bootstrap(wide, a, b, col, rng)
            if r is None:
                continue
            r.update({"quantity": col, "state_a": LABEL[a], "state_b": LABEL[b]})
            rows.append(r)
    out = pd.DataFrame(rows)[["quantity", "state_a", "state_b", "n_detectors",
                              "mean_diff", "ci_low", "ci_high", "frac_positive"]]
    out.to_csv(OUT / "fd_state_retention_contrasts.csv", index=False)

    for q in ["retention", "M", "Delta"]:
        print(f"\n=== paired {q}: state_a minus state_b (detector-clustered 95% CI) ===")
        s = out[out.quantity == q]
        for _, r in s.iterrows():
            excl = "  *" if (r.ci_low > 0) or (r.ci_high < 0) else ""
            print(f"  {r.state_a:>10s} - {r.state_b:<10s} "
                  f"{r.mean_diff:+7.3f}  [{r.ci_low:+7.3f},{r.ci_high:+7.3f}]  "
                  f"n={r.n_detectors:4d}  frac+={r.frac_positive:.2f}{excl}")
    print("\n* interval excludes zero")
    print("wrote", OUT / "fd_state_retention_contrasts.csv")


if __name__ == "__main__":
    main()
