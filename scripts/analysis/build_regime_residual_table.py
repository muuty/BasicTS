#!/usr/bin/env python3
"""Build the compact TR-B regime-residual table from the K50 audit."""

from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[2]
CSV = REPO / "experiments" / "result" / "analysis" / "transport_regime_residual_summary.csv"
OUT = REPO / "writing" / "CoresetSelection-paper" / "tables" / "regime_residual.tex"

METHODS = ["k_medoids", "random", "stride", "recent", "k_center", "graph_cut"]
LABEL = {
    "k_medoids": "K-medoids",
    "random": "Random",
    "stride": "Stride",
    "recent": "Recent",
    "k_center": "K-center",
    "graph_cut": "Graph Cut",
}
REGIMES_FOR_WORST = [
    "peak", "off_peak", "weekday", "weekend", "congested_q20",
    "noncongested", "abrupt_speed_drop_q10", "non_drop",
]


def value(df: pd.DataFrame, method: str, ratio: float, regime: str, col: str) -> float:
    row = df[(df.method == method) & (df.ratio == ratio) & (df.regime == regime)]
    return float(row[col].iloc[0])


def build() -> str:
    df = pd.read_csv(CSV)
    df = df[df.K_label == "K50"]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\setlength{\tabcolsep}{5pt}",
        r"\renewcommand{\arraystretch}{1.05}",
        (
            r"\caption{History-only target-continuity audit, averaged over two datasets and "
            r"the available selection seeds. $K_{50}$ is calibrated separately on each "
            r"dataset as the median local nearest-history slope $d_Y/d_X$. $J_X$ is "
            r"history quantization; $p_\rho$ is the fraction of windows with positive "
            r"future-trajectory residual; and $D=KJ_X+\bar\rho$ is the target-continuity "
            r"component of Corollary~\ref{sec:input_only_bound}. Cong. denotes the bottom "
            r"speed quintile and Drop the bottom decile of future-minus-history network "
            r"speed. $D_{\max}$ is the maximum over the declared peak, weekday/weekend, "
            r"congestion, and speed-drop regime pairs. These are validation diagnostics, "
            r"not certified neural-network risk bounds.}"
        ),
        r"\label{tab:regime_residual}",
        r"\small",
        r"\begin{tabular}{clrrrrrr}",
        r"\toprule",
        r"Ratio & Method & $J_X$ & $p_\rho$(all) & $p_\rho$(cong.) & $p_\rho$(drop) & $D_{\rm all}$ & $D_{\max}$ \\",
        r"\midrule",
    ]
    for ratio_idx, ratio in enumerate([0.1, 0.3]):
        sub = df[df.ratio == ratio]
        for method_idx, method in enumerate(METHODS):
            ratio_cell = rf"\multirow{{{len(METHODS)}}}{{*}}{{{ratio:.1f}}}" if method_idx == 0 else ""
            all_j = value(sub, method, ratio, "all", "J_X")
            p_all = value(sub, method, ratio, "all", "residual_positive_rate")
            p_cong = value(sub, method, ratio, "congested_q20", "residual_positive_rate")
            p_drop = value(sub, method, ratio, "abrupt_speed_drop_q10", "residual_positive_rate")
            d_all = value(sub, method, ratio, "all", "data_term_KJ_plus_residual")
            d_max = max(
                value(sub, method, ratio, regime, "data_term_KJ_plus_residual")
                for regime in REGIMES_FOR_WORST
            )
            lines.append(
                " & ".join([
                    ratio_cell, LABEL[method], f"{all_j:.3f}", f"{p_all:.3f}",
                    f"{p_cong:.3f}", f"{p_drop:.3f}", f"{d_all:.3f}", f"{d_max:.3f}",
                ]) + r" \\"
            )
        if ratio_idx == 0:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    OUT.write_text(build())
    print(f"wrote {OUT}")
