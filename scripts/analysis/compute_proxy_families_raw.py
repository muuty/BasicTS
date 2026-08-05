#!/usr/bin/env python3
"""All four pre-training score families in one representation.

The scores compared in the manuscript were computed in two different spaces: the
coverage scores on a 50-dimensional PCA projection, the rest on the raw standardised
trajectories. A table built from that mixture measures the representation as much as
the property, so this script recomputes every family in the single raw space that
Proposition 1 attaches to: the standardised history and future of a window, flattened,
under the $\\ell_1$ metric.

Everything derives from one distance matrix per retained set, the $n \\times |S|$
distances from each training window to each retained window:

  coverage               mean, median and maximum of the row minimum
  diversity              summed similarity inside the retained set
  distribution matching  entropic transport cost from the full empirical measure to the
                         uniform measure on the retained set. Proposition 1 makes the
                         free-weight version of this identical to the coverage mean, so
                         the family differs from coverage exactly by holding the weights
                         uniform
  temporal diversity     entropy of the retained set over hour of day and day of week,
                         which needs no distance at all

Similarities use an RBF kernel whose bandwidth is the median pairwise distance, so the
kernel is fixed by the data and not by a chosen constant.

Writes experiments/result/analysis/proxy_families_raw.csv.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
os.chdir(REPO)  # import_config resolves a config path as a module, so it must be relative

from coreset.distance import extract_features, get_features_by_type  # noqa: E402

OUT = REPO / "experiments" / "result" / "analysis"
INDEX = REPO / "coreset_indices"
CONFIGS = {
    "SAN_BERNARDINO": "baselines/STGCN/SAN_BERNARDINO/SAN_BERNARDINO_no_dropout.py",
    "CONTRA_COSTA": "baselines/STGCN/CONTRA_COSTA/CONTRA_COSTA.py",
}


def distances(feat: torch.Tensor, sel: torch.Tensor, batch: int = 256) -> torch.Tensor:
    """(n, |S|) L1 distances, computed in batches over the full set."""
    out = torch.empty((feat.shape[0], sel.shape[0]), dtype=torch.float32, device=feat.device)
    for i in range(0, feat.shape[0], batch):
        out[i:i + batch] = torch.cdist(feat[i:i + batch], sel, p=1)
    return out


def sinkhorn(cost: torch.Tensor, epsilon: float, iters: int = 200) -> float:
    """Entropic transport cost from the uniform measure on rows to that on columns."""
    n, m = cost.shape
    K = torch.exp(-cost / epsilon)
    u = torch.ones(n, device=cost.device) / n
    v = torch.ones(m, device=cost.device) / m
    a = torch.full((n,), 1.0 / n, device=cost.device)
    b = torch.full((m,), 1.0 / m, device=cost.device)
    for _ in range(iters):
        u = a / (K @ v + 1e-30)
        v = b / (K.t() @ u + 1e-30)
    plan = u.unsqueeze(1) * K * v.unsqueeze(0)
    return float((plan * cost).sum())


def calendar_entropy(idx: np.ndarray) -> tuple[float, float]:
    """Normalised entropy of the retained set over hour of day and day of week."""
    def h(labels: np.ndarray, k: int) -> float:
        counts = np.bincount(labels, minlength=k).astype(float)
        p = counts[counts > 0] / counts.sum()
        return float(-(p * np.log(p)).sum() / np.log(k))
    return h((idx % 288) // 12, 24), h((idx // 288) % 7, 7)


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []

    for dataset, cfg_path in CONFIGS.items():
        if dataset not in args.datasets:
            continue
        sys.path.insert(0, str(REPO))
        from easytorch.config import import_config
        from basicts.data import TimeSeriesForecastingDataset

        cfg = import_config(cfg_path, verbose=False)
        ds = TimeSeriesForecastingDataset(mode="train", **cfg["DATASET"]["PARAM"])
        inputs, targets = extract_features(ds, cfg["MODEL"])
        feat = torch.from_numpy(
            get_features_by_type(inputs, targets, "euclidean").astype(np.float32)).to(device)
        n, d = feat.shape
        print(f"[{dataset}] {n} windows, {d} dimensions, device {device}", flush=True)

        # One bandwidth for the whole dataset, the median distance over a random subset.
        probe = feat[torch.randperm(n, device=device)[:512]]
        sigma = float(torch.cdist(probe, probe, p=1).median())
        print(f"  RBF bandwidth (median pairwise L1): {sigma:.1f}", flush=True)

        files = sorted(p for p in (INDEX / dataset).glob("*_euclidean_*.json"))
        for k, path in enumerate(files, 1):
            idx = np.asarray(json.loads(path.read_text()), dtype=int)
            idx = idx[idx < n]
            if idx.size < 10:
                continue
            t0 = time.time()
            sel = feat[torch.from_numpy(idx).to(device)]
            cost = distances(feat, sel)
            nearest = cost.min(dim=1).values

            sim_full = torch.exp(-cost / sigma)
            sub = torch.cdist(sel, sel, p=1)
            h_tod, h_dow = calendar_entropy(idx)
            rows.append({
                "dataset": dataset, "index_file": path.name,
                "qc_mean": float(nearest.mean()), "qc_median": float(nearest.median()),
                "qc_max": float(nearest.max()),
                "fl_objective": float(sim_full.max(dim=1).values.sum()),
                "redundancy": float(torch.exp(-sub / sigma).sum()),
                "sinkhorn": sinkhorn(cost, epsilon=sigma / 10.0),
                "h_tod": h_tod, "h_dow": h_dow,
            })
            del cost, sim_full, sub
            if device == "cuda":
                torch.cuda.empty_cache()
            print(f"  [{k}/{len(files)}] {path.name}  {time.time() - t0:.1f}s", flush=True)

    import pandas as pd
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "proxy_families_raw.csv", index=False)
    print(f"\nwrote {OUT / 'proxy_families_raw.csv'} with {len(rows)} rows")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(CONFIGS))
    return parser.parse_args()


if __name__ == "__main__":
    main()
