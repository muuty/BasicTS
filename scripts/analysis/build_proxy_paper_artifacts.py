"""Generate paper-ready proxy-metric artifacts for Section 7.

Outputs:
1. tables/proxy_correlations.tex  - Pearson/Spearman r per metric, per model.
2. tables/coreset_advantage.tex   - per-method delta-W1 summary.
3. figures/quant_vs_mae.pdf       - quantization-cost vs MAE scatter.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
ANA = REPO / "experiments" / "result" / "analysis"
PAPER = REPO / "writing" / "CoresetSelection-paper"
TABLES = PAPER / "tables"
FIGS = PAPER / "figures"

# ----- Proxy correlation table ---------------------------------------------

CORR_CSV = ANA / "phase_c_proxy_correlations.csv"

METRIC_LABEL = {
    "ot_cost": ("OT cost", "down"),
    "sinkhorn_divergence": ("Sinkhorn divergence", "down"),
    "fl_objective": ("Facility-location obj.", "down"),
    "redundancy": ("Intra-set redundancy", "down"),
    "information_gain": ("Information gain", "up"),
    "h_tod": ("Time-of-day entropy", "up"),
    "h_dow": ("Day-of-week entropy", "up"),
    "quantization_cost": ("Quantization cost (mean)", "down"),
    "quantization_median": ("Quantization cost (median)", "down"),
    "quantization_max": ("Quantization cost (max)", "down"),
}
METRIC_ORDER = [
    "quantization_cost", "quantization_median", "quantization_max",
    "fl_objective", "information_gain", "redundancy",
    "h_tod", "h_dow",
    "sinkhorn_divergence", "ot_cost",
]
MODELS = ["STGCN", "AGCRN", "DCRNN", "STID", "STAEformer"]


def build_correlation_table() -> str:
    df = pd.read_csv(CORR_CSV)
    df = df[df.level == "per_model"]
    df["model"] = df["model"].replace({"STGCNChebGraphConv": "STGCN"})

    pivot = df.pivot_table(index="proxy_metric", columns="model",
                           values="pearson_r", aggfunc="first")
    spear = df.pivot_table(index="proxy_metric", columns="model",
                           values="spearman_r", aggfunc="first")

    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.05}")
    lines.append(
        r"\caption{Per-model Pearson correlation between proxy metrics and test MAE on the paper grid (179--180 samples per model, pooling all selection methods, ratios, subset-selection seeds, and the two datasets; training seed fixed at 42). Sign convention follows each metric's natural direction; large $|r|$ indicates a strong predictor of MAE. \textbf{Bold} marks the strongest predictor per model.}"
    )
    lines.append(r"\label{tab:proxy_correlations}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{l" + "c" * len(MODELS) + "}")
    lines.append(r"\toprule")
    lines.append(" & ".join(["Proxy metric"] + MODELS) + r" \\")
    lines.append(r"\midrule")

    # Determine best |r| per model
    best_per_model = {}
    available = [m for m in METRIC_ORDER if m in pivot.index]
    for col in MODELS:
        if col not in pivot.columns:
            continue
        col_abs = pivot.loc[available, col].abs()
        best_per_model[col] = col_abs.idxmax()

    for metric in METRIC_ORDER:
        if metric not in pivot.index:
            continue
        label, _direction = METRIC_LABEL[metric]
        row = [label]
        for model in MODELS:
            if model not in pivot.columns:
                row.append("--")
                continue
            r = pivot.loc[metric, model]
            cell = f"{r:+.2f}"
            if best_per_model.get(model) == metric:
                cell = r"\textbf{" + cell + "}"
            row.append(cell)
        lines.append(" & ".join(row) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    out = TABLES / "proxy_correlations.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")
    return str(out)


# ----- Coreset advantage table ---------------------------------------------

ADV_CSV = ANA / "coreset_advantage.csv"


def build_coreset_advantage_table() -> str:
    df = pd.read_csv(ADV_CSV)
    df["model_short"] = df["model"].replace({"STGCNChebGraphConv": "STGCN"})
    summary = (
        df.groupby(["method", "ratio"])
        .agg(
            sw1_test_train=("sw1_test_train", "first"),
            sw1_test_core_mean=("sw1_test_core", "mean"),
            delta_w1_mean=("delta_w1", "mean"),
            delta_w1_min=("delta_w1", "min"),
            delta_w1_max=("delta_w1", "max"),
            n=("delta_w1", "size"),
            n_pos=("delta_w1", lambda s: int((s > 0).sum())),
        )
        .reset_index()
    )

    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.05}")
    lines.append(
        r"\caption{Coreset advantage on San Bernardino (Phase~B grid, $n$ checkpoints per cell across distance pipelines, seeds, and backbones). $\Delta W_1=\mathrm{SW}_1(P_{\text{test}},P_{\text{train}})-\mathrm{SW}_1(P_{\text{test}},P_{\text{core}})$, computed on PCA-10 features. Positive $\Delta W_1$ means the coreset is closer to the test distribution than the full training set is. Baseline $\mathrm{SW}_1(P_{\text{test}},P_{\text{train}})\!=\!3.04$. $n_{+}$ = number of cells with $\Delta W_1\!>\!0$.}"
    )
    lines.append(r"\label{tab:coreset_advantage}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llrrrrrr}")
    lines.append(r"\toprule")
    lines.append(
        r"Method & Ratio & $\mathrm{SW}_1(\text{test},\text{core})$ & $\overline{\Delta W_1}$ & $\Delta W_1^{\min}$ & $\Delta W_1^{\max}$ & $n$ & $n_{+}$ \\"
    )
    lines.append(r"\midrule")

    method_label = {"k_medoids": "K-medoids", "k_center": "K-center", "graph_cut": "Graph Cut"}
    method_order = ["k_medoids", "k_center", "graph_cut"]
    for method in method_order:
        sub = summary[summary.method == method].sort_values("ratio")
        for _, row in sub.iterrows():
            mean = row.delta_w1_mean
            cell_mean = f"{mean:+.3f}"
            if mean > 0:
                cell_mean = r"\textbf{" + cell_mean + "}"
            lines.append(
                " & ".join([
                    method_label[method],
                    f"{row.ratio:.1f}",
                    f"{row.sw1_test_core_mean:.2f}",
                    cell_mean,
                    f"{row.delta_w1_min:+.3f}",
                    f"{row.delta_w1_max:+.3f}",
                    str(int(row.n)),
                    str(int(row.n_pos)),
                ]) + r" \\"
            )
        lines.append(r"\midrule")
    # Drop the last \midrule (replace with bottomrule)
    if lines[-1] == r"\midrule":
        lines.pop()

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    out = TABLES / "coreset_advantage.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")
    return str(out)


# ----- Quantization cost vs MAE scatter ------------------------------------

BOUND_CSV = ANA / "bound_terms_cpu.csv"


def build_quant_vs_mae_plot() -> str:
    df = pd.read_csv(BOUND_CSV)
    df["model_short"] = df["model"].replace({"STGCNChebGraphConv": "STGCN"})

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), sharey=False)
    method_color = {
        "k_medoids": "#1f77b4",
        "k_center": "#2ca02c",
        "graph_cut": "#d62728",
    }
    method_marker = {
        "k_medoids": "o",
        "k_center": "s",
        "graph_cut": "^",
    }

    for ax, model in zip(axes, ["STGCN", "AGCRN"]):
        sub = df[df.model_short == model]
        for method, color in method_color.items():
            s = sub[sub.method == method]
            ax.scatter(
                s.quant_pca_mean, s.mae_test,
                c=color, marker=method_marker[method],
                s=40, alpha=0.85,
                label=method.replace("_", "-"),
                edgecolors="white", linewidth=0.4,
            )
        # Linear fit (Pearson r in title)
        x = sub.quant_pca_mean.to_numpy()
        y = sub.mae_test.to_numpy()
        if len(x) > 2:
            r = np.corrcoef(x, y)[0, 1]
            ax.set_title(f"{model} ($r={r:+.3f}$)")
        ax.set_xlabel("Quantization cost (PCA-10)")
        ax.set_ylabel("Test MAE")
        ax.grid(True, alpha=0.3, linestyle=":")
        ax.legend(loc="lower right", fontsize=8, frameon=True)
    fig.tight_layout()
    pdf_out = FIGS / "quant_vs_mae.pdf"
    png_out = FIGS / "quant_vs_mae.png"
    fig.savefig(pdf_out, bbox_inches="tight")
    fig.savefig(png_out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"wrote {pdf_out} and {png_out}")
    return str(pdf_out)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    build_correlation_table()
    build_coreset_advantage_table()
    build_quant_vs_mae_plot()


if __name__ == "__main__":
    main()
