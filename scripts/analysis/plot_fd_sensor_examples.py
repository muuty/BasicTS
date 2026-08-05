#!/usr/bin/env python3
"""Plot algorithmically chosen detector-level FD examples for the paper."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
ANALYSIS = REPO / "experiments" / "result" / "analysis"
FIGURES = REPO / "writing" / "CoresetSelection-paper" / "figures"
DATASETS = ["SAN_BERNARDINO", "CONTRA_COSTA"]
LABELS = {"SAN_BERNARDINO": "San Bernardino", "CONTRA_COSTA": "Contra Costa"}


def load_raw(dataset: str) -> np.ndarray:
    desc = json.loads((REPO / "datasets" / dataset / "desc.json").read_text())
    data = np.memmap(
        REPO / "datasets" / dataset / "data.dat",
        dtype=np.float32,
        mode="r",
        shape=tuple(desc["shape"]),
    )
    return np.asarray(data[:14_516, :, :3])


def representative_sensors(fits: pd.DataFrame) -> list[int]:
    accepted = fits[fits.identifiable].sort_values("critical_occupancy")
    targets = accepted.critical_occupancy.quantile([0.25, 0.50, 0.75]).to_numpy()
    selected = []
    for target in targets:
        available = accepted[~accepted.sensor.isin(selected)]
        sensor = int(
            available.loc[(available.critical_occupancy - target).abs().idxmin(), "sensor"]
        )
        selected.append(sensor)
    return selected


def main() -> None:
    fits = pd.read_csv(ANALYSIS / "fd_sensor_resolved_fits.csv")
    bins = pd.read_csv(ANALYSIS / "fd_sensor_resolved_bins.csv")
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.3), sharex=False, sharey=False)
    rng = np.random.default_rng(42)
    for row, dataset in enumerate(DATASETS):
        raw = load_raw(dataset)
        dataset_fits = fits[fits.dataset.eq(dataset)]
        sensors = representative_sensors(dataset_fits)
        for column, sensor in enumerate(sensors):
            ax = axes[row, column]
            fit = dataset_fits[dataset_fits.sensor.eq(sensor)].iloc[0]
            detector_bins = bins[
                bins.dataset.eq(dataset) & bins.sensor.eq(sensor)
            ].sort_values("occupancy")
            q = raw[:, sensor, 0]
            o = raw[:, sensor, 1]
            v = raw[:, sensor, 2]
            oc = fit.critical_occupancy
            vc = fit.speed_at_capacity
            free = (o < oc) & (v > vc)
            congested = (o > oc) & (v < vc)
            unresolved = ~(free | congested)
            sample = rng.choice(len(o), size=min(4500, len(o)), replace=False)
            for mask, color, label in [
                (unresolved, "0.72", "unresolved"),
                (free, "#2878B5", "free"),
                (congested, "#D1495B", "congested"),
            ]:
                idx = sample[mask[sample]]
                ax.scatter(o[idx], q[idx], s=3, alpha=0.18, color=color,
                           edgecolors="none", label=label)
            ax.plot(
                detector_bins.occupancy,
                detector_bins.fitted_flow,
                color="black",
                linewidth=1.8,
                label="two-branch fit",
            )
            ax.scatter(
                detector_bins.occupancy,
                detector_bins.flow,
                s=11,
                facecolors="white",
                edgecolors="black",
                linewidths=0.6,
                zorder=4,
                label="bin median",
            )
            ax.axvline(oc, color="black", linestyle="--", linewidth=0.9)
            ax.set_title(f"{LABELS[dataset]}, detector {sensor}", fontsize=9)
            if column == 0:
                ax.set_ylabel("Flow")
            if row == 1:
                ax.set_xlabel("Occupancy")
            ax.grid(alpha=0.15, linewidth=0.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 1.015), fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "fd_sensor_examples.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fd_sensor_examples.png", dpi=220, bbox_inches="tight")
    print(f"wrote detector examples for {DATASETS}")


if __name__ == "__main__":
    main()
