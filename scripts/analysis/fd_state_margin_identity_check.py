#!/usr/bin/env python3
"""Is the reducibility-to-reduction-loss association mechanical?

Delta = M(1-kappa) and M = s * e_p, so Delta contains s as a factor. Correlating
s with Delta therefore carries an algebraic component. This script separates the
mechanical part from any residual signal:

  1. r(s, Delta) as reported, adjusted for dataset only.
  2. r(s, Delta) controlling full-data MAE -- the control used in the paper.
  3. r(s, Delta) controlling full-data AND persistence MAE, which removes the
     route by which s enters Delta.
  4. r(s, 1-kappa) = r(s, Delta/M): the identity-free question, whether more
     reducible detectors give up a larger SHARE of their margin.
  5. A permutation null that keeps each detector's full-data MAE and permutes
     the reduced-data MAE within a cell, so the mechanical -e_f term survives.
     The correct null for r(s, Delta) is this distribution, not zero.

Writes fd_state_margin_identity_check.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fd_state_conditional_followups import OUT

PANEL = OUT / "fd_state_margin_panel.csv"
N_PERM = 2000
SEED = 0


def residualise(y: np.ndarray, controls: list[np.ndarray]) -> np.ndarray:
    A = np.column_stack([np.ones(len(y))] + [np.asarray(c, float) for c in controls])
    beta, *_ = np.linalg.lstsq(A, np.asarray(y, float), rcond=None)
    return np.asarray(y, float) - A @ beta


def corr(a, b, controls):
    return float(np.corrcoef(residualise(a, controls), residualise(b, controls))[0, 1])


def main() -> None:
    p = pd.read_csv(PANEL)
    p["M"] = p.persist_mae - p.full_mae
    p["Delta"] = p.red_mae - p.full_mae
    p["s"] = p.M / p.persist_mae
    p["ds"] = (p.dataset == "SAN_BERNARDINO").astype(float)
    rng = np.random.default_rng(SEED)

    rows = []
    for (bb, st), g in p.groupby(["backbone", "traffic_state_transition"]):
        g = g.dropna(subset=["s", "Delta", "full_mae", "persist_mae"])
        if len(g) < 20:
            continue
        ds = [g.ds]
        r_raw = corr(g.s, g.Delta, ds)
        r_ef = corr(g.s, g.Delta, ds + [g.full_mae])
        r_efep = corr(g.s, g.Delta, ds + [g.full_mae, g.persist_mae])
        pos = g[g.M > 0]
        r_share = corr(pos.s, pos.Delta / pos.M, [pos.ds]) if len(pos) >= 20 else np.nan
        r_red = corr(g.s, g.red_mae, ds)

        null = np.empty(N_PERM)
        s_res = residualise(g.s.to_numpy(), ds)
        ef = g.full_mae.to_numpy()
        red = g.red_mae.to_numpy()
        for i in range(N_PERM):
            null[i] = np.corrcoef(s_res, residualise(rng.permutation(red) - ef, ds))[0, 1]
        rows.append({
            "backbone": bb, "state": st, "n_detectors": len(g),
            "r_raw": r_raw, "r_ctrl_full": r_ef, "r_ctrl_full_persist": r_efep,
            "r_share_1_minus_kappa": r_share, "r_reduced_mae": r_red,
            "null_median": float(np.median(null)),
            "null_q975": float(np.quantile(null, 0.975)),
            "p_perm": float((null >= r_raw).mean()),
            "n_positive_margin": int((g.M > 0).sum()),
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "fd_state_margin_identity_check.csv", index=False)

    def line(lbl, col):
        v = out[col].dropna()
        print(f"  {lbl:44s} median {v.median():+.3f}   positive {int((v > 0).sum())}/{len(v)}")

    print(f"cells: {len(out)}")
    line("r(s, Delta) as reported", "r_raw")
    line("r(s, Delta) | full MAE   [paper's control]", "r_ctrl_full")
    line("r(s, Delta) | full AND persistence MAE", "r_ctrl_full_persist")
    line("r(s, 1-kappa)  identity-free", "r_share_1_minus_kappa")
    line("r(s, reduced MAE)", "r_reduced_mae")
    print(f"  {'permutation null for r(s,Delta)':44s} median {out.null_median.median():+.3f}")
    print(f"  cells with r_raw above their own null median: "
          f"{int((out.r_raw > out.null_median).sum())}/{len(out)}")
    print(f"  cells with permutation p < 0.05: {int((out.p_perm < 0.05).sum())}/{len(out)}")
    print("\nwrote", OUT / "fd_state_margin_identity_check.csv")


if __name__ == "__main__":
    main()
