#!/usr/bin/env python3
"""Where does the quantization-versus-MAE concordance actually come from?

The headline 69.6% pools 750 within-budget pairwise comparisons. Two structural
concerns make that number hard to read:

  1. Two selectors, Graph Cut and K-center, are separable from the rest by
     inspection (+17.6 to +21.1% and +7.1 to +11.4% MAE). Nine of the fifteen
     pairs in every block involve one of them, so a large part of the pooled
     statistic may encode a single ordering fact replicated many times.
  2. At high sampling ratios the MAE differences between competitive methods are
     small, so a share of the comparisons is decided by noise.

This script recomputes concordance overall, by sampling ratio, and restricted to
the competitive selectors, and checks the arithmetic of the leave-group-out table:
the dataset folds and the model folds each partition the same 750 pairs, so their
pair-weighted mean must return the pooled value.

Writes proxy_concordance_stratified.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from proxy_heldout_validation import build_cells, build_pairs, RESULT_DIR

OUT = RESULT_DIR / "analysis"
SEPARABLE = {"graph_cut", "k_center"}
N_BOOT = 10_000
SEED = 0


def concordance(df: pd.DataFrame) -> float:
    return float(df.concordant.mean()) if len(df) else np.nan


def clustered_ci(df: pd.DataFrame, keys: list[str], rng) -> tuple[float, float]:
    """Bootstrap over clusters defined by `keys` (blocks are not independent)."""
    groups = [g.concordant.to_numpy() for _, g in df.groupby(keys)]
    if len(groups) < 3:
        return (np.nan, np.nan)
    boot = np.empty(N_BOOT)
    for i in range(N_BOOT):
        pick = rng.integers(0, len(groups), len(groups))
        boot[i] = np.concatenate([groups[j] for j in pick]).mean()
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def main() -> None:
    cells = build_cells()
    pairs = build_pairs(cells)
    rng = np.random.default_rng(SEED)

    mcols = [c for c in pairs.columns if "method" in c]
    both = pairs[mcols].apply(lambda r: set(r.dropna()), axis=1)
    pairs["has_separable"] = both.apply(lambda s: bool(s & SEPARABLE))

    rows = []

    def add(label, sub):
        lo, hi = clustered_ci(sub, ["dataset", "ratio"], rng)
        rows.append({"subset": label, "n_pairs": len(sub),
                     "concordance": concordance(sub), "ci_low": lo, "ci_high": hi})

    add("all pairs", pairs)
    add("excluding Graph Cut and K-center", pairs[~pairs.has_separable])
    add("involving Graph Cut or K-center", pairs[pairs.has_separable])
    for r in sorted(pairs.ratio.unique()):
        add(f"ratio {r:g}, all pairs", pairs[pairs.ratio == r])
    for r in sorted(pairs.ratio.unique()):
        sub = pairs[(pairs.ratio == r) & (~pairs.has_separable)]
        add(f"ratio {r:g}, competitive only", sub)

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "proxy_concordance_stratified.csv", index=False)
    print(f"{'subset':38s} {'n':>5s} {'conc':>7s}  95% CI")
    for _, r in out.iterrows():
        ci = f"[{r.ci_low:.3f},{r.ci_high:.3f}]" if np.isfinite(r.ci_low) else "     --     "
        print(f"{r.subset:38s} {r.n_pairs:5d} {r.concordance:7.3f}  {ci}")

    # arithmetic check on the leave-group-out folds
    print("\n=== fold arithmetic: dataset and model folds each partition the 750 pairs ===")
    for axis, col in [("dataset", "dataset"), ("model", "model")]:
        g = pairs.groupby(col).concordant.agg(["mean", "size"])
        weighted = float((g["mean"] * g["size"]).sum() / g["size"].sum())
        print(f"  {axis:8s} folds={len(g)}  pair-weighted mean {weighted:.4f}  "
              f"pooled {concordance(pairs):.4f}  match={abs(weighted-concordance(pairs))<1e-9}")
        print("      per fold: " + ", ".join(f"{k}={v:.3f}(n={int(n)})"
                                             for k, (v, n) in g.iterrows()))
    print("\nwrote", OUT / "proxy_concordance_stratified.csv")


if __name__ == "__main__":
    main()
