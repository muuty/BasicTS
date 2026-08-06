#!/usr/bin/env python3
"""Build the detector-resolved traffic-state support table."""

from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[2]
SOURCE = (
    REPO / "experiments" / "result" / "analysis"
    / "fd_sensor_resolved_state_contrasts.csv"
)
SUMMARY = (
    REPO / "experiments" / "result" / "analysis"
    / "fd_sensor_resolved_clustered_summary.csv"
)
OUT = REPO / "writing" / "CoresetSelection-paper" / "tables" / "fd_sensor_resolved.tex"
METHODS = ["k_medoids", "random", "stride", "graph_cut"]
METHOD_LABEL = {
    "k_medoids": "K-medoids",
    "random": "Random",
    "stride": "Stride",
    "graph_cut": "Graph Cut",
}
DATASETS = ["SAN_BERNARDINO", "CONTRA_COSTA"]
DATASET_LABEL = {
    "SAN_BERNARDINO": "San Bernardino",
    "CONTRA_COSTA": "Contra Costa",
}
STATES = ["breakdown", "recovery", "congested_to_congested"]


def interval(row: pd.Series) -> str:
    return (
        f"{row.ratio_to_free_mean:.2f} "
        f"[{row.ratio_to_free_ci_low:.2f}, {row.ratio_to_free_ci_high:.2f}]"
    )


def main() -> None:
    contrasts = pd.read_csv(SOURCE)
    contrasts = contrasts[
        contrasts.metric.eq("D") & contrasts.K_label.eq("K50")
    ]
    summary = pd.read_csv(SUMMARY)
    summary = summary[
        summary.K_label.eq("K50")
        & summary.traffic_state_transition.eq("free_to_free")
    ]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\setlength{\tabcolsep}{3.4pt}",
        r"\renewcommand{\arraystretch}{1.06}",
        (
            r"\caption{Detector-resolved support mismatch by traffic-state transition. "
            r"$D_{\rm FF}$ is the detector mean of $D=KJ_X+\bar\rho$ for "
            r"free$\rightarrow$free windows. Remaining entries are the mean "
            r"detector-paired ratio to that detector's free$\rightarrow$free value, "
            r"with 95\% detector-cluster bootstrap intervals. States use each "
            r"detector's fitted flow--occupancy branches with speed confirmation, "
            r"and ambiguous observations remain unresolved. $K_{50}$ is calibrated per "
            r"detector from selector-independent calendar-lag pairs. These quantities "
            r"diagnose local support coverage and are not neural-network risk bounds.}"
        ),
        r"\label{tab:fd_sensor_resolved}",
        r"\scriptsize",
        r"\begin{tabular}{llcrlll}",
        r"\toprule",
        "Dataset & Ratio & Method & $D_{\\rm FF}$ & Breakdown/FF & Recovery/FF & Congested/FF \\\\",
        r"\midrule",
    ]
    for dataset_i, dataset in enumerate(DATASETS):
        for ratio in [0.1, 0.3]:
            for method_i, method in enumerate(METHODS):
                sub = contrasts[
                    contrasts.dataset.eq(dataset)
                    & contrasts.ratio.eq(ratio)
                    & contrasts.method.eq(method)
                ].set_index("compared_state")
                ff = summary[
                    summary.dataset.eq(dataset)
                    & summary.ratio.eq(ratio)
                    & summary.method.eq(method)
                ].iloc[0]
                dataset_cell = (
                    rf"\multirow{{8}}{{*}}{{{DATASET_LABEL[dataset]}}}"
                    if ratio == 0.1 and method_i == 0 else ""
                )
                ratio_cell = (
                    rf"\multirow{{4}}{{*}}{{{ratio:.1f}}}"
                    if method_i == 0 else ""
                )
                values = [interval(sub.loc[state]) for state in STATES]
                lines.append(
                    " & ".join([
                        dataset_cell,
                        ratio_cell,
                        METHOD_LABEL[method],
                        f"{ff.D_detector_mean:.3f}",
                        *values,
                    ])
                    + " \\\\"
                )
            if ratio == 0.1:
                lines.append(r"\cmidrule(lr){2-7}")
        if dataset_i == 0:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
