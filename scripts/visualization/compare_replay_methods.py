"""
Compare performance across replay methods (baseline, difficulty, representative) for each model.
"""

from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Optional, Union


def load_data(result_dir: Union[str, Path]) -> dict:
    """Load all experiment result CSV files."""
    result_dir = Path(result_dir)

    data = {}

    # Load representative.csv (contains baseline + representative results)
    representative_path = result_dir / "representative.csv"
    if representative_path.exists():
        data["representative"] = pd.read_csv(representative_path)

    # Load difficult.csv
    difficult_path = result_dir / "difficult.csv"
    if difficult_path.exists():
        data["difficult"] = pd.read_csv(difficult_path)

    # Load experience_replay.csv
    experience_replay_path = result_dir / "experience_replay.csv"
    if experience_replay_path.exists():
        data["experience_replay"] = pd.read_csv(experience_replay_path)

    return data


def extract_results(data: dict) -> pd.DataFrame:
    """
    Extract and organize results by model and method.

    Returns a DataFrame with columns:
    - model: model name
    - method: baseline, difficult, representative
    - MAE_mean, MAE_std, inc_MAE_mean, inc_MAE_std, non_inc_MAE_mean, non_inc_MAE_std
    """
    results = []

    # Extract baseline and representative from representative.csv
    if "representative" in data:
        df = data["representative"]

        for _, row in df.iterrows():
            model = row["model"]
            method = row.get("experience_replay_method", None)

            # Baseline: no method specified
            if pd.isna(method) or method == "":
                results.append({
                    "model": model,
                    "method": "baseline",
                    "MAE_mean": row["MAE_mean"],
                    "MAE_std": row.get("MAE_std", np.nan),
                    "inc_MAE_mean": row.get("inc_MAE_mean", np.nan),
                    "inc_MAE_std": row.get("inc_MAE_std", np.nan),
                    "non_inc_MAE_mean": row.get("non_inc_MAE_mean", np.nan),
                    "non_inc_MAE_std": row.get("non_inc_MAE_std", np.nan),
                    "weight": np.nan,
                    "n_runs": row.get("n_runs", 1),
                })
            elif method == "representative":
                results.append({
                    "model": model,
                    "method": "representative",
                    "MAE_mean": row["MAE_mean"],
                    "MAE_std": row.get("MAE_std", np.nan),
                    "inc_MAE_mean": row.get("inc_MAE_mean", np.nan),
                    "inc_MAE_std": row.get("inc_MAE_std", np.nan),
                    "non_inc_MAE_mean": row.get("non_inc_MAE_mean", np.nan),
                    "non_inc_MAE_std": row.get("non_inc_MAE_std", np.nan),
                    "weight": row.get("experience_replay_weight", np.nan),
                    "n_runs": row.get("n_runs", 1),
                })

    # Extract difficult results
    if "difficult" in data:
        df = data["difficult"]

        for _, row in df.iterrows():
            model = row["model"]
            method = row.get("experience_replay_method", None)

            if method == "difficult":
                results.append({
                    "model": model,
                    "method": "difficult",
                    "MAE_mean": row["MAE_mean"],
                    "MAE_std": row.get("MAE_std", np.nan),
                    "inc_MAE_mean": row.get("inc_MAE_mean", np.nan),
                    "inc_MAE_std": row.get("inc_MAE_std", np.nan),
                    "non_inc_MAE_mean": row.get("non_inc_MAE_mean", np.nan),
                    "non_inc_MAE_std": row.get("non_inc_MAE_std", np.nan),
                    "weight": row.get("experience_replay_weight", np.nan),
                    "n_runs": row.get("n_runs", 1),
                })

    return pd.DataFrame(results)


def plot_model_comparison(
    results_df: pd.DataFrame,
    metric: str = "MAE_mean",
    std_col: str = "MAE_std",
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (12, 6),
):
    """
    Create a bar plot comparing methods for each model.

    Args:
        results_df: DataFrame with model, method, and metric columns
        metric: Column name for the metric to plot
        std_col: Column name for the standard deviation
        save_path: Path to save the figure
        figsize: Figure size
    """
    # Get unique models and methods
    models = results_df["model"].unique()
    methods = ["baseline", "difficult", "representative"]

    # Filter methods that exist in data
    available_methods = results_df["method"].unique()
    methods = [m for m in methods if m in available_methods]

    # Set up the plot
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(models))
    width = 0.25
    multiplier = 0

    colors = {
        "baseline": "#2ecc71",      # green
        "difficult": "#e74c3c",     # red
        "representative": "#3498db", # blue
    }

    for method in methods:
        method_data = results_df[results_df["method"] == method]

        means = []
        stds = []

        for model in models:
            model_method_data = method_data[method_data["model"] == model]

            if len(model_method_data) > 0:
                # If multiple entries (different weights), take the best one
                best_idx = model_method_data[metric].idxmin()
                means.append(model_method_data.loc[best_idx, metric])
                std_val = model_method_data.loc[best_idx, std_col]
                stds.append(std_val if not pd.isna(std_val) else 0)
            else:
                means.append(np.nan)
                stds.append(0)

        offset = width * multiplier
        bars = ax.bar(
            x + offset,
            means,
            width,
            label=method.capitalize(),
            color=colors.get(method, "gray"),
            yerr=stds,
            capsize=3,
            alpha=0.8,
        )

        # Add value labels on bars
        for bar, mean_val in zip(bars, means):
            if not np.isnan(mean_val):
                ax.annotate(
                    f"{mean_val:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        multiplier += 1

    # Customize plot
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_title(f"Comparison of Replay Methods by Model\n({metric.replace('_', ' ')})", fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {save_path}")

    plt.show()

    return fig, ax


def plot_all_metrics(
    results_df: pd.DataFrame,
    save_dir: Optional[Union[str, Path]] = None,
    figsize: tuple = (14, 5),
):
    """
    Create subplots for Overall MAE, Incident MAE, and Non-Incident MAE.
    """
    # Get unique models and methods
    models = results_df["model"].unique()
    methods = ["baseline", "difficult", "representative"]

    # Filter methods that exist in data
    available_methods = results_df["method"].unique()
    methods = [m for m in methods if m in available_methods]

    metrics = [
        ("MAE_mean", "MAE_std", "Overall MAE"),
        ("inc_MAE_mean", "inc_MAE_std", "Incident MAE"),
        ("non_inc_MAE_mean", "non_inc_MAE_std", "Non-Incident MAE"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    colors = {
        "baseline": "#2ecc71",      # green
        "difficult": "#e74c3c",     # red
        "representative": "#3498db", # blue
    }

    for ax, (metric, std_col, title) in zip(axes, metrics):
        x = np.arange(len(models))
        width = 0.25
        multiplier = 0

        for method in methods:
            method_data = results_df[results_df["method"] == method]

            means = []
            stds = []

            for model in models:
                model_method_data = method_data[method_data["model"] == model]

                if len(model_method_data) > 0:
                    # If multiple entries (different weights), take the best one based on MAE_mean
                    best_idx = model_method_data["MAE_mean"].idxmin()
                    means.append(model_method_data.loc[best_idx, metric])
                    std_val = model_method_data.loc[best_idx, std_col]
                    stds.append(std_val if not pd.isna(std_val) else 0)
                else:
                    means.append(np.nan)
                    stds.append(0)

            offset = width * multiplier
            bars = ax.bar(
                x + offset,
                means,
                width,
                label=method.capitalize(),
                color=colors.get(method, "gray"),
                yerr=stds,
                capsize=2,
                alpha=0.8,
            )

            # Add value labels
            for bar, mean_val in zip(bars, means):
                if not np.isnan(mean_val):
                    ax.annotate(
                        f"{mean_val:.1f}",
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 2),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        rotation=90,
                    )

            multiplier += 1

        ax.set_xlabel("Model", fontsize=10)
        ax.set_ylabel("MAE", fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(x + width)
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        if ax == axes[-1]:
            ax.legend(loc="upper right", fontsize=9)

    plt.suptitle("Performance Comparison: Baseline vs Replay Methods", fontsize=13, y=1.02)
    plt.tight_layout()

    if save_dir:
        save_path = Path(save_dir) / "replay_method_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {save_path}")

    plt.show()

    return fig, axes


def print_summary_table(results_df: pd.DataFrame):
    """Print a summary table of results."""
    print("\n" + "=" * 80)
    print("SUMMARY: Best Performance by Model and Method")
    print("=" * 80)

    models = results_df["model"].unique()
    methods = ["baseline", "difficult", "representative"]

    # Create summary table
    summary_data = []

    for model in models:
        model_data = results_df[results_df["model"] == model]
        row = {"Model": model}

        for method in methods:
            method_data = model_data[model_data["method"] == method]

            if len(method_data) > 0:
                best_idx = method_data["MAE_mean"].idxmin()
                mae = method_data.loc[best_idx, "MAE_mean"]
                weight = method_data.loc[best_idx, "weight"]

                if pd.isna(weight):
                    row[method.capitalize()] = f"{mae:.2f}"
                else:
                    row[method.capitalize()] = f"{mae:.2f} (w={weight})"
            else:
                row[method.capitalize()] = "-"

        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    print("=" * 80)


def main():
    # Set paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    result_dir = project_root / "experiments" / "result"
    save_dir = script_dir / "figures"

    # Create save directory
    save_dir.mkdir(exist_ok=True)

    # Load data
    print("Loading data...")
    data = load_data(result_dir)

    print(f"Loaded files: {list(data.keys())}")

    # Extract results
    results_df = extract_results(data)

    print(f"\nFound {len(results_df)} result entries")
    print(f"Models: {results_df['model'].unique().tolist()}")
    print(f"Methods: {results_df['method'].unique().tolist()}")

    # Print summary table
    print_summary_table(results_df)

    # Plot comparison
    print("\nGenerating plots...")

    # Single metric plot
    plot_model_comparison(
        results_df,
        metric="MAE_mean",
        save_path=save_dir / "mae_comparison.png",
    )

    # All metrics plot
    plot_all_metrics(
        results_df,
        save_dir=save_dir,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
