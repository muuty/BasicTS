#!/usr/bin/env python3
"""What state retention a budget of windows admits.

A selection objective chooses windows; the traffic state belongs to the
window--detector pair. Write $m_a(i)$ for the number of detectors in state $a$ at
window $i$ and $|I_a|=\\sum_i m_a(i)$, so a retained set $C$ realises the retention
profile $\\pi_a(C)=\\sum_{i\\in C}m_a(i)/|I_a|$. This is a profile of per-state
retention rates and not a division of the budget: one window contributes to every
state at once, so the coordinates cannot be set independently, and $\\sum_a\\pi_a$ is
unconstrained.

Let $M$ have entries $m_a(i)/|I_a|$, so $\\pi(C)=M\\mathbf 1_C$, and let
$\\Delta_{k,n}=\\{x\\in[0,1]^n:\\sum_i x_i=k\\}$ be the hypersimplex, the base
polytope of the uniform matroid and the convex hull of the indicators of $k$-subsets.
Then $\\operatorname{conv}\\{\\pi(C):|C|=k\\}=M\\Delta_{k,n}$, and the guarded frontier

    F(tau) = max { pi_b(C) : |C| = k, pi_g(C) >= tau }

is bounded by the value of the same programme relaxed to the hypersimplex. This script
computes that relaxation exactly and rounds it to a feasible set:

  lp            the relaxation value, an upper bound on the integer optimum
  attainable    a feasible retained set obtained from a basic optimum by one exchange
  certificate   max_i m_b(i)/|I_b|, one window's worth of the target state

A basic optimum of a programme with one guard row besides the cardinality has at most
two fractional coordinates summing to one, so the exchange loses at most the
certificate. The certificate is known before the programme is solved, which is what
makes the frontier located; a small observed gap on its own would prove nothing.

Guard levels above what k windows can hold are infeasible and are reported as such
rather than as a bound.

Everything uses the chronological training split alone, so the frontier is computable
before any model is fitted.

Writes experiments/result/analysis/attainable_allocation{,_selectors}.csv, and the
retained index set at the requested guard level.
"""
from __future__ import annotations

import argparse
import json
import types
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "experiments" / "result" / "analysis"
INDEX = REPO / "coreset_indices"

STATES = {0: "free", 1: "breakdown", 2: "recovery", 3: "congestion"}
SCREEN = types.SimpleNamespace(
    n_bins=40, min_branch_bins=6, min_free_rise=0.10, min_congested_drop=0.05,
    min_fit_improvement=0.10, min_valid_fraction=2 / 3, min_agreement=0.75)


def state_counts(dataset: str) -> dict[int, np.ndarray]:
    """Per-window detector counts in each state, on the training split."""
    import importlib.util
    import sys
    sys.path.insert(0, str(REPO / "scripts" / "analysis"))
    spec = importlib.util.spec_from_file_location(
        "fx", REPO / "scripts" / "analysis" / "fd_selection_experiment.py")
    fx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fx)
    transition = fx.training_labels(dataset, SCREEN)
    return {code: (transition == code).sum(axis=1).astype(float) for code in STATES}


def top_k_sum(values: np.ndarray, k: int) -> float:
    """Support function of the convex hull of k-subset sums in one direction."""
    if k <= 0:
        return 0.0
    part = np.partition(values, -k)[-k:]
    return float(part.sum())


def frontier(target: np.ndarray, guarded: np.ndarray, k: int, tau: float) -> dict:
    """Certified bounds on the guarded frontier, by linear programming and one exchange.

    Maximise the retained share of `target` at budget k while holding the retained
    share of `guarded` at or above tau. Relaxing the indicator to the hypersimplex
    gives a linear programme whose value $F_{LP}$ upper-bounds the integer optimum,
    because the hypersimplex is the convex hull of the indicators. A basic optimum has
    at most two fractional coordinates, and they sum to one, so setting the one with
    the larger guarded count to one and the other to zero returns a feasible set whose
    target share is below the relaxation value by at most one window's worth of the target state. The width of that certificate is known before the programme is
    solved, so tightness is proved and not inferred from a small observed gap.
    """
    n_target, n_guard = target.sum(), guarded.sum()
    t_share, g_share = target / n_target, guarded / n_guard

    reachable = float(np.sort(g_share)[-k:].sum())
    if tau > reachable + 1e-12:
        return {"tau": tau, "lp": np.nan, "attainable": np.nan, "gap": np.nan,
                "certificate": np.nan, "feasible": False, "max_guarded": reachable,
                "guarded_at_attainable": np.nan, "index": None}

    # max t_share . x  s.t.  sum x = k,  g_share . x >= tau,  0 <= x <= 1
    result = linprog(
        c=-t_share,
        A_ub=np.vstack([-g_share]), b_ub=np.array([-tau]),
        A_eq=np.ones((1, len(t_share))), b_eq=np.array([float(k)]),
        bounds=(0.0, 1.0), method="highs")
    if not result.success:
        raise RuntimeError(f"linear programme failed at tau={tau}: {result.message}")
    lp_value = float(-result.fun)
    x = result.x

    # Round to a feasible set: keep the integral ones, break ties by guarded count.
    integral = np.isclose(x, 1.0, atol=1e-9)
    fractional = np.flatnonzero(~integral & ~np.isclose(x, 0.0, atol=1e-9))
    keep = list(np.flatnonzero(integral))
    if fractional.size:
        keep += list(fractional[np.argsort(-g_share[fractional], kind="stable")][:k - len(keep)])
    keep = np.asarray(sorted(keep)[:k], dtype=int)
    if keep.size < k:                       # degenerate basis, top up on the guard
        spare = np.setdiff1d(np.argsort(-g_share, kind="stable"), keep, assume_unique=False)
        keep = np.sort(np.concatenate([keep, spare[:k - keep.size]]))

    attainable = float(t_share[keep].sum())
    achieved_guard = float(g_share[keep].sum())
    certificate = float(t_share.max())      # one window's worth of the target state
    return {"tau": tau, "lp": lp_value, "attainable": attainable,
            "gap": lp_value - attainable, "certificate": certificate,
            "feasible": achieved_guard >= tau - 1e-9, "max_guarded": reachable,
            "guarded_at_attainable": achieved_guard, "index": keep}


def main() -> None:
    args = parse_args()
    rows, selector_rows = [], []
    for dataset in args.datasets:
        counts = state_counts(dataset)
        n_windows = len(counts[0])
        print(f"\n=== {dataset}: {n_windows} windows, "
              f"{int(sum(c.sum() for c in counts.values())):,} resolved pairs ===", flush=True)

        for ratio in args.ratios:
            k = int(round(ratio * n_windows))
            for target_code, guard_code in args.pairs:
                target, guarded = counts[target_code], counts[guard_code]
                for tau in args.taus:
                    result = frontier(target, guarded, k, tau)
                    index = result.pop("index")
                    result.update(dataset=dataset, ratio=ratio,
                                  target=STATES[target_code], guarded=STATES[guard_code])
                    rows.append(result)
                    if (index is not None and args.emit_index and ratio == 0.1
                            and target_code == 1 and guard_code == 0
                            and abs(tau - args.emit_index) < 1e-9):
                        name = f"fd_front{int(round(tau*100)):02d}"
                        out = INDEX / dataset / f"{name}_euclidean_010_seed42.json"
                        out.write_text(json.dumps([int(i) for i in index]))
                        print(f"    wrote {out.name}  ({len(index)} windows)", flush=True)
                    if np.isnan(result["lp"]):
                        print(f"  r={ratio}  {STATES[target_code]:<10} guarded "
                              f"{STATES[guard_code]:<10} tau={tau:.3f}   infeasible "
                              f"(largest guard a budget of {k} admits: "
                              f"{result['max_guarded']:.4f})", flush=True)
                    else:
                        print(f"  r={ratio}  {STATES[target_code]:<10} guarded "
                              f"{STATES[guard_code]:<10} tau={tau:.3f}   "
                              f"LP={result['lp']:.4f}  attained={result['attainable']:.4f}  "
                              f"gap={result['gap']:.5f}  certificate={result['certificate']:.5f}",
                              flush=True)

            # Where each audited objective sits against the bound at its own guard level.
            ratio_text = f"{ratio:.2f}".replace(".", "")
            for name in args.objectives:
                path = INDEX / dataset / f"{name}_euclidean_{ratio_text}_seed{args.seed}.json"
                if not path.exists():
                    continue
                idx = np.asarray(json.loads(path.read_text()), dtype=int)
                idx = idx[idx < n_windows]
                for target_code, guard_code in args.pairs:
                    target, guarded = counts[target_code], counts[guard_code]
                    realised_t = float(target[idx].sum() / target.sum())
                    realised_g = float(guarded[idx].sum() / guarded.sum())
                    bound = frontier(target, guarded, k, realised_g)
                    bound.pop("index", None)
                    selector_rows.append({
                        "dataset": dataset, "ratio": ratio, "objective": name,
                        "target": STATES[target_code], "guarded": STATES[guard_code],
                        "pi_target": realised_t, "pi_guarded": realised_g,
                        "lp_at_pi_guarded": bound["lp"],
                        "attained_share": realised_t / bound["lp"] if bound["lp"] else np.nan,
                        "guard_binds": bool(bound["lp"] < frontier(target, guarded, k, 0.0)["lp"] - 1e-9),
                    })

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "attainable_allocation.csv", index=False)
    selectors = pd.DataFrame(selector_rows)
    selectors.to_csv(OUT / "attainable_allocation_selectors.csv", index=False)
    print("\n=== objectives against the bound ===")
    if len(selectors):
        view = selectors[(selectors.target == "breakdown") & (selectors.guarded == "free")]
        print(view.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nwrote {OUT / 'attainable_allocation.csv'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["SAN_BERNARDINO", "CONTRA_COSTA"])
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.1, 0.3])
    parser.add_argument("--taus", nargs="+", type=float,
                        default=[0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--emit-index", type=float, default=0.10,
                        help="write the frontier index set at this guard level")
    parser.add_argument("--objectives", nargs="+", default=[
        "random", "stride", "recent", "k_medoids", "k_center", "graph_cut",
        "fd_grp00", "fd_strat00", "fd_hyb0802", "fd_hyb0604", "fd_hyb2703", "fd_hyb1812", "fd_front10"])
    args = parser.parse_args()
    args.pairs = [(1, 0), (2, 0), (1, 3)]  # breakdown|free, recovery|free, breakdown|congestion
    return args


if __name__ == "__main__":
    main()
