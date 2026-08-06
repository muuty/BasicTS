"""Generate paper-ready wall-clock LaTeX table from phase_c_wall_clock_gpu.csv.

The table reports median per-run wall-clock per (model, ratio) on a single
GPU type, plus the speedup factor relative to the full-data baseline. To keep
the comparison apples-to-apples, we filter to one GPU type (default MI300X,
the most populated bucket), include only completed runs, and aggregate over
selection methods, seeds, and the two datasets.

A second compact table reports the same speedup factor on the secondary GPU
type (MI210) as a robustness check.

Outputs:
    tables/wall_clock_mi300x.tex
    tables/wall_clock_mi210.tex
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CSV = REPO / "experiments" / "result" / "analysis" / "phase_c_wall_clock_gpu.csv"
TABLES = REPO / "writing" / "CoresetSelection-paper" / "tables"

MODELS_ORD = ["STGCN", "AGCRN", "DCRNN", "STID", "STAEformer"]
RATIOS = [0.3, 0.5, 0.7, 1.0]
RATIO_LABEL = {r: ("Full" if r == 1.0 else f"{r:.1f}") for r in RATIOS}

# Restrict to a single experiment directory + single GPU type for an
# apples-to-apples wall-clock comparison. phase_c_method_comparison covers
# all five backbones at ratios {0.3, 0.5, 0.7, 1.0} and the MI300X bucket
# is by far the largest within that experiment (459 completed runs).
PHASE = "phase_c_method_comparison"


def load() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    df["model_short"] = df["model"].replace({"STGCNChebGraphConv": "STGCN"})
    df = df[df.completed == True]
    df = df[df.phase == PHASE]
    return df


def build(df: pd.DataFrame, gpu: str, label: str) -> str:
    sub = df[df.gpu == gpu]
    pivot = sub.groupby(["model_short", "ratio"]).agg(
        med=("wall_seconds", "median"),
        n=("wall_seconds", "size"),
    )
    # Reshape to (ratio, model)
    table = {}
    counts = {}
    for model in MODELS_ORD:
        for r in RATIOS:
            try:
                row = pivot.loc[(model, r)]
                table[(model, r)] = row["med"]
                counts[(model, r)] = int(row["n"])
            except KeyError:
                table[(model, r)] = None
                counts[(model, r)] = 0

    # Build LaTeX
    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\setlength{\tabcolsep}{4.5pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.05}")
    lines.append(
        r"\caption{Per-run training wall-clock on a single AMD " + label + r" GPU. "
        r"Each cell is the median over selection methods, selection-seed labels, and the two datasets, "
        r"in seconds. Numbers in parentheses are the speedup over the full-data baseline. "
        r"The training seed is fixed at 42. "
        r"Selection cost is performed once per dataset and shared across all five backbones, "
        r"so it is amortised over repeated model runs.}"
    )
    lines.append(r"\label{tab:wall_clock_" + gpu.lower() + "}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{l" + "r" * len(MODELS_ORD) + "}")
    lines.append(r"\toprule")
    header = " & ".join(["Ratio"] + MODELS_ORD) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    for r in RATIOS:
        cells = [RATIO_LABEL[r]]
        for model in MODELS_ORD:
            val = table.get((model, r))
            if val is None:
                cells.append("--")
                continue
            full = table.get((model, 1.0))
            if r == 1.0 or full is None or full == 0:
                cells.append(f"{val:,.0f}")
            else:
                speedup = full / val
                cells.append(f"{val:,.0f} ({speedup:.1f}$\\times$)")
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def main() -> None:
    df = load()
    print(f"Phase {PHASE}, completed runs by GPU:", df.gpu.value_counts().to_dict())
    for gpu, label, fname in [
        ("MI300X", "MI300X", "wall_clock_mi300x.tex"),
        ("MI210", "MI210", "wall_clock_mi210.tex"),
    ]:
        TABLES.joinpath(fname).write_text(build(df, gpu, label))
        print(f"wrote {TABLES / fname}")


if __name__ == "__main__":
    main()
