"""
Visualize coreset selection results: MAE vs selection ratio for each model.
Shows Overall MAE, Non-Incident MAE, and Incident MAE.
"""

from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Optional, Union


def load_coreset_data(result_dir: Union[str, Path]) -> pd.DataFrame:
    """Load coreset selection results."""
    result_dir = Path(result_dir)
    coreset_path = result_dir / "coreset.csv"

    if not coreset_path.exists():
        raise FileNotFoundError(f"coreset.csv not found in {result_dir}")

    df = pd.read_csv(coreset_path)

    # Filter out invalid rows (MAE_mean == 0)
    df = df[df["MAE_mean"] > 0]

    return df


def plot_coreset_by_model(
    df: pd.DataFrame,
    save_dir: Optional[Union[str, Path]] = None,
    figsize: tuple = (16, 10),
):
    """
    Create a figure with subplots for each model.
    Each subplot shows MAE vs ratio with three lines: Overall, Non-Incident, Incident.
    """
    models = df["model"].unique()
    n_models = len(models)

    # Calculate grid layout
    n_cols = 2
    n_rows = (n_models + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_models > 1 else [axes]

    colors = {
        "overall": "#3498db",      # blue
        "non_incident": "#2ecc71", # green
        "incident": "#e74c3c",     # red
    }

    markers = {
        "overall": "o",
        "non_incident": "s",
        "incident": "^",
    }

    for idx, model in enumerate(models):
        ax = axes[idx]
        model_data = df[df["model"] == model].sort_values("coreset_selection_ratio")

        ratios = model_data["coreset_selection_ratio"].values

        # Plot Overall MAE
        mae_mean = model_data["MAE_mean"].values
        mae_std = model_data["MAE_std"].fillna(0).values
        ax.errorbar(
            ratios, mae_mean, yerr=mae_std,
            label="Overall MAE",
            color=colors["overall"],
            marker=markers["overall"],
            markersize=6,
            capsize=3,
            linewidth=2,
            alpha=0.8,
        )

        # Plot Non-Incident MAE
        non_inc_mae_mean = model_data["non_inc_MAE_mean"].values
        non_inc_mae_std = model_data["non_inc_MAE_std"].fillna(0).values
        ax.errorbar(
            ratios, non_inc_mae_mean, yerr=non_inc_mae_std,
            label="Non-Incident MAE",
            color=colors["non_incident"],
            marker=markers["non_incident"],
            markersize=6,
            capsize=3,
            linewidth=2,
            alpha=0.8,
        )

        # Plot Incident MAE
        inc_mae_mean = model_data["inc_MAE_mean"].values
        inc_mae_std = model_data["inc_MAE_std"].fillna(0).values
        ax.errorbar(
            ratios, inc_mae_mean, yerr=inc_mae_std,
            label="Incident MAE",
            color=colors["incident"],
            marker=markers["incident"],
            markersize=6,
            capsize=3,
            linewidth=2,
            alpha=0.8,
        )

        ax.set_xlabel("Coreset Selection Ratio", fontsize=11)
        ax.set_ylabel("MAE", fontsize=11)
        ax.set_title(model, fontsize=12, fontweight="bold")
        ax.set_xticks(np.arange(0.1, 1.1, 0.1))
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)

    # Hide empty subplots
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(
        "Coreset Selection: MAE vs Selection Ratio\n(k-medoids strategy)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    if save_dir:
        save_path = Path(save_dir) / "coreset_selection_by_model.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {save_path}")

    plt.show()

    return fig, axes


def plot_coreset_comparison(
    df: pd.DataFrame,
    save_dir: Optional[Union[str, Path]] = None,
    figsize: tuple = (16, 5),
):
    """
    Create a figure with three subplots: Overall MAE, Non-Incident MAE, Incident MAE.
    Each subplot shows all models as separate lines.
    """
    models = df["model"].unique()

    metrics = [
        ("MAE_mean", "MAE_std", "Overall MAE"),
        ("non_inc_MAE_mean", "non_inc_MAE_std", "Non-Incident MAE"),
        ("inc_MAE_mean", "inc_MAE_std", "Incident MAE"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    markers = ["o", "s", "^", "D", "v", "<", ">", "p"]

    for ax, (metric, std_col, title) in zip(axes, metrics):
        for idx, model in enumerate(models):
            model_data = df[df["model"] == model].sort_values("coreset_selection_ratio")

            ratios = model_data["coreset_selection_ratio"].values
            values = model_data[metric].values
            stds = model_data[std_col].fillna(0).values

            ax.errorbar(
                ratios, values, yerr=stds,
                label=model,
                color=colors[idx],
                marker=markers[idx % len(markers)],
                markersize=6,
                capsize=3,
                linewidth=2,
                alpha=0.8,
            )

        ax.set_xlabel("Coreset Selection Ratio", fontsize=11)
        ax.set_ylabel("MAE", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(np.arange(0.1, 1.1, 0.1))
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)

    plt.suptitle(
        "Coreset Selection: Model Comparison by Metric",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    if save_dir:
        save_path = Path(save_dir) / "coreset_selection_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {save_path}")

    plt.show()

    return fig, axes


def print_best_ratios(df: pd.DataFrame):
    """Print the best ratio for each model based on different metrics."""
    print("\n" + "=" * 80)
    print("BEST CORESET SELECTION RATIOS BY MODEL")
    print("=" * 80)

    models = df["model"].unique()

    for model in models:
        model_data = df[df["model"] == model]

        # Best overall MAE
        best_overall_idx = model_data["MAE_mean"].idxmin()
        best_overall_ratio = model_data.loc[best_overall_idx, "coreset_selection_ratio"]
        best_overall_mae = model_data.loc[best_overall_idx, "MAE_mean"]

        # Best incident MAE
        best_inc_idx = model_data["inc_MAE_mean"].idxmin()
        best_inc_ratio = model_data.loc[best_inc_idx, "coreset_selection_ratio"]
        best_inc_mae = model_data.loc[best_inc_idx, "inc_MAE_mean"]

        # Best non-incident MAE
        best_non_inc_idx = model_data["non_inc_MAE_mean"].idxmin()
        best_non_inc_ratio = model_data.loc[best_non_inc_idx, "coreset_selection_ratio"]
        best_non_inc_mae = model_data.loc[best_non_inc_idx, "non_inc_MAE_mean"]

        print(f"\n{model}:")
        print(f"  Overall MAE:      best ratio = {best_overall_ratio:.1f}, MAE = {best_overall_mae:.2f}")
        print(f"  Non-Incident MAE: best ratio = {best_non_inc_ratio:.1f}, MAE = {best_non_inc_mae:.2f}")
        print(f"  Incident MAE:     best ratio = {best_inc_ratio:.1f}, MAE = {best_inc_mae:.2f}")

    print("\n" + "=" * 80)


def main():
    # Set paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    result_dir = project_root / "experiments" / "result"
    save_dir = script_dir / "figures"

    # Create save directory
    save_dir.mkdir(exist_ok=True)

    # Load data
    print("Loading coreset selection data...")
    df = load_coreset_data(result_dir)

    print(f"Loaded {len(df)} result entries")
    print(f"Models: {df['model'].unique().tolist()}")
    print(f"Ratios: {sorted(df['coreset_selection_ratio'].unique().tolist())}")

    # Print best ratios
    print_best_ratios(df)

    # Generate plots
    print("\nGenerating plots...")

    # Plot by model (each model in separate subplot)
    plot_coreset_by_model(df, save_dir=save_dir)

    # Plot comparison (all models in same plot for each metric)
    plot_coreset_comparison(df, save_dir=save_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
