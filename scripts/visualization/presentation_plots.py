"""
Presentation Plots for Monthly Meeting
Generated: 2026-01-26
Updated with accurate data from experiments/get_results.py
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns

# Set style
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    plt.style.use('seaborn-whitegrid')
sns.set_palette("husl")
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 150

OUTPUT_DIR = Path("scripts/visualization/presentation_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR = Path("experiments/result")


# =============================================================================
# 1. Node Identity Experiment (from CSV)
# =============================================================================
def plot_node_identity_experiment():
    """Compare graph message passing approaches"""

    csv_path = RESULT_DIR / "node_identity_exp.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        print("Node Identity data loaded from CSV:")
        print(df[['model', 'MAE_mean', 'MAPE_mean', 'RMSE_mean', 'inc_MAE_mean', 'non_inc_MAE_mean']])
    else:
        print("CSV not found, using hardcoded values")
        df = pd.DataFrame({
            'model': ['STGCNChebGraphConv', 'STGCNWeakenedMP', 'STGCNNodeIdentity'],
            'MAE_mean': [14.23, 14.70, 17.18],
            'MAPE_mean': [0.302, 0.412, 0.393],
            'RMSE_mean': [26.72, 26.38, 31.26],
            'inc_MAE_mean': [13.56, 16.20, 19.15],
            'non_inc_MAE_mean': [14.40, 14.36, 16.73]
        })

    # Sort by MAE
    df = df.sort_values('MAE_mean')

    models = df['model'].tolist()
    models_display = ['STGCN\n(Baseline)', 'WeakenedMP', 'NodeIdentity']

    mae = df['MAE_mean'].tolist()
    mape = df['MAPE_mean'].tolist()
    rmse = df['RMSE_mean'].tolist()
    mae_incident = df['inc_MAE_mean'].tolist()
    mae_non_incident = df['non_inc_MAE_mean'].tolist()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    x = np.arange(len(models))
    width = 0.6
    colors = ['#2ecc71', '#3498db', '#e74c3c']

    # MAE
    bars1 = axes[0].bar(x, mae, width, color=colors, edgecolor='black', linewidth=1.2)
    axes[0].set_ylabel('MAE')
    axes[0].set_title('Mean Absolute Error (Lower is Better)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models_display)
    axes[0].set_ylim(0, max(mae) * 1.2)
    for bar, val in zip(bars1, mae):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     f'{val:.2f}', ha='center', va='bottom', fontweight='bold')

    # MAPE
    bars2 = axes[1].bar(x, [m*100 for m in mape], width, color=colors, edgecolor='black', linewidth=1.2)
    axes[1].set_ylabel('MAPE (%)')
    axes[1].set_title('Mean Absolute Percentage Error (Lower is Better)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models_display)
    axes[1].set_ylim(0, max(mape)*100 * 1.2)
    for bar, val in zip(bars2, mape):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f'{val*100:.1f}%', ha='center', va='bottom', fontweight='bold')

    # RMSE
    bars3 = axes[2].bar(x, rmse, width, color=colors, edgecolor='black', linewidth=1.2)
    axes[2].set_ylabel('RMSE')
    axes[2].set_title('Root Mean Square Error (Lower is Better)')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(models_display)
    axes[2].set_ylim(0, max(rmse) * 1.2)
    for bar, val in zip(bars3, rmse):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f'{val:.2f}', ha='center', va='bottom', fontweight='bold')

    plt.suptitle('Node Identity Experiment: Graph Message Passing Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '1_node_identity_overall.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / '1_node_identity_overall.png'}")

    # Incident vs Non-Incident comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width/2, mae_non_incident, width, label='Non-Incident', color='#3498db', edgecolor='black')
    bars2 = ax.bar(x + width/2, mae_incident, width, label='Incident', color='#e74c3c', edgecolor='black')

    ax.set_ylabel('MAE')
    ax.set_title('Incident vs Non-Incident Performance by Model', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models_display)
    ax.legend()
    ax.set_ylim(0, max(max(mae_incident), max(mae_non_incident)) * 1.2)

    for bar, val in zip(bars1, mae_non_incident):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    for bar, val in zip(bars2, mae_incident):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '1_node_identity_incident_comparison.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / '1_node_identity_incident_comparison.png'}")


# =============================================================================
# 2. Experience Replay Experiment
# =============================================================================
def plot_experience_replay_experiment():
    """Compare experience replay configurations"""

    # Accurate data from checkpoints
    configs = ['Baseline', 'Config 1\n(1bbe12e8)', 'Config 2\n(468267f1)', 'Config 3\n(a1ff5f4f)']
    mae = [17.91, 13.84, 14.71, 17.34]
    mape = [0.364, 0.192, 0.195, 0.235]
    rmse = [30.74, 26.07, 27.01, 30.83]

    inc_mae = [19.01, 14.93, 15.89, 18.83]
    non_inc_mae = [17.63, 13.57, 14.42, 16.97]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    x = np.arange(len(configs))
    width = 0.6
    colors = ['#95a5a6', '#2ecc71', '#3498db', '#f39c12']

    # MAE
    bars1 = axes[0].bar(x, mae, width, color=colors, edgecolor='black', linewidth=1.2)
    axes[0].set_ylabel('MAE')
    axes[0].set_title('Mean Absolute Error')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(configs)
    axes[0].axhline(y=mae[0], color='red', linestyle='--', alpha=0.5, label='Baseline')
    for bar, val in zip(bars1, mae):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     f'{val:.2f}', ha='center', va='bottom', fontweight='bold')

    # MAPE
    bars2 = axes[1].bar(x, [m*100 for m in mape], width, color=colors, edgecolor='black', linewidth=1.2)
    axes[1].set_ylabel('MAPE (%)')
    axes[1].set_title('Mean Absolute Percentage Error')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(configs)
    axes[1].axhline(y=mape[0]*100, color='red', linestyle='--', alpha=0.5, label='Baseline')
    for bar, val in zip(bars2, mape):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f'{val*100:.1f}%', ha='center', va='bottom', fontweight='bold')

    # RMSE
    bars3 = axes[2].bar(x, rmse, width, color=colors, edgecolor='black', linewidth=1.2)
    axes[2].set_ylabel('RMSE')
    axes[2].set_title('Root Mean Square Error')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(configs)
    axes[2].axhline(y=rmse[0], color='red', linestyle='--', alpha=0.5, label='Baseline')
    for bar, val in zip(bars3, rmse):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f'{val:.2f}', ha='center', va='bottom', fontweight='bold')

    plt.suptitle('Experience Replay Experiment: Configuration Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '2_experience_replay_comparison.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / '2_experience_replay_comparison.png'}")

    # Incident vs Non-Incident
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(configs))
    width = 0.35

    bars1 = ax.bar(x - width/2, non_inc_mae, width, label='Non-Incident', color='#3498db', edgecolor='black')
    bars2 = ax.bar(x + width/2, inc_mae, width, label='Incident', color='#e74c3c', edgecolor='black')

    ax.set_ylabel('MAE')
    ax.set_title('Experience Replay: Incident vs Non-Incident Performance', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.legend()

    for bar, val in zip(bars1, non_inc_mae):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, inc_mae):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '2_experience_replay_incident.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / '2_experience_replay_incident.png'}")

    # Improvement chart
    fig, ax = plt.subplots(figsize=(8, 5))

    improvements = [(mae[0] - m) / mae[0] * 100 for m in mae[1:]]
    config_names = ['Config 1', 'Config 2', 'Config 3']
    colors_imp = ['#2ecc71' if imp > 0 else '#e74c3c' for imp in improvements]

    bars = ax.barh(config_names, improvements, color=colors_imp, edgecolor='black')
    ax.set_xlabel('MAE Improvement (%)')
    ax.set_title('Experience Replay: MAE Improvement over Baseline', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=0.8)

    for bar, val in zip(bars, improvements):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', ha='left', va='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '2_experience_replay_improvement.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / '2_experience_replay_improvement.png'}")


# =============================================================================
# 3. Coreset Selection Experiment (from CSV)
# =============================================================================
def plot_coreset_experiment():
    """Compare coreset selection across models"""

    csv_path = RESULT_DIR / "coreset.csv"
    if not csv_path.exists():
        print("Coreset CSV not found!")
        return

    df = pd.read_csv(csv_path)
    print("\nCoreset data loaded from CSV:")
    print(f"Models: {df['model'].unique().tolist()}")
    print(f"Selection ratios: {sorted(df['coreset_selection_ratio'].unique().tolist())}")

    # Filter out zero MAE rows (invalid results)
    df = df[df['MAE_mean'] > 0]

    # Plot 1: MAE vs Selection Ratio by Model
    fig, ax = plt.subplots(figsize=(12, 6))

    models = df['model'].unique()
    colors = plt.cm.Set1(np.linspace(0, 1, len(models)))
    markers = ['o', 's', '^', 'D']

    for i, model in enumerate(models):
        model_df = df[df['model'] == model].sort_values('coreset_selection_ratio')
        ax.plot(model_df['coreset_selection_ratio'], model_df['MAE_mean'],
                marker=markers[i % len(markers)], label=model, color=colors[i],
                linewidth=2, markersize=8)

        # Add error bars if std is available
        if 'MAE_std' in model_df.columns:
            std_vals = model_df['MAE_std'].fillna(0)
            ax.fill_between(model_df['coreset_selection_ratio'],
                           model_df['MAE_mean'] - std_vals,
                           model_df['MAE_mean'] + std_vals,
                           alpha=0.2, color=colors[i])

    ax.set_xlabel('Selection Ratio')
    ax.set_ylabel('MAE')
    ax.set_title('Coreset Selection: MAE vs Selection Ratio', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_xlim(0, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '3_coreset_selection_ratio.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / '3_coreset_selection_ratio.png'}")

    # Plot 2: Best MAE by Model (at optimal selection ratio)
    best_results = df.loc[df.groupby('model')['MAE_mean'].idxmin()]
    best_results = best_results.sort_values('MAE_mean')

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(best_results))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(best_results)))

    bars = ax.bar(x, best_results['MAE_mean'], color=colors, edgecolor='black', linewidth=1.2)

    ax.set_ylabel('MAE')
    ax.set_title('Coreset Selection: Best MAE by Model', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{row['model']}\n(ratio={row['coreset_selection_ratio']})"
                        for _, row in best_results.iterrows()], fontsize=10)

    for bar, (_, row) in zip(bars, best_results.iterrows()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{row["MAE_mean"]:.2f}', ha='center', va='bottom', fontweight='bold')

    # Highlight best
    bars[0].set_color('#2ecc71')
    bars[0].set_edgecolor('black')
    bars[0].set_linewidth(2)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '3_coreset_best_by_model.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / '3_coreset_best_by_model.png'}")

    # Plot 3: Incident vs Non-Incident for best configs
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(best_results))
    width = 0.35

    bars1 = ax.bar(x - width/2, best_results['non_inc_MAE_mean'], width,
                   label='Non-Incident', color='#3498db', edgecolor='black')
    bars2 = ax.bar(x + width/2, best_results['inc_MAE_mean'], width,
                   label='Incident', color='#e74c3c', edgecolor='black')

    ax.set_ylabel('MAE')
    ax.set_title('Coreset Selection: Incident vs Non-Incident (Best Configs)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(best_results['model'].tolist())
    ax.legend()

    for bar, val in zip(bars1, best_results['non_inc_MAE_mean']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, best_results['inc_MAE_mean']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '3_coreset_incident_comparison.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / '3_coreset_incident_comparison.png'}")


# =============================================================================
# 4. Incident Type Analysis (from node_identity best model)
# =============================================================================
def plot_incident_analysis():
    """Analyze performance across different incident types"""

    # Load from actual incident metrics file
    incident_metrics_path = Path("checkpoints/node_identity_exp/STGCNChebGraphConv/xtraffic/SAN_BERNARDINO_100_12_12/1/3bdf69ea375f4f707bc033f96085beed/test_incident_metrics.json")

    if incident_metrics_path.exists():
        with open(incident_metrics_path, 'r') as f:
            data = json.load(f)

        incident_types = ['Hazard', 'NoInj', 'UnknInj', 'Fire', 'CarFire', '1141', 'Other', 'AHazard']
        mae_values = [data[f'{t}_overall']['MAE'] for t in incident_types]

        # Sample counts from the log
        sample_counts = [2008, 849, 611, 67, 53, 318, 261, 74]
    else:
        print("Incident metrics file not found, using estimated values")
        incident_types = ['Hazard', 'NoInj', 'UnknInj', 'Fire', 'CarFire', '1141', 'Other', 'AHazard']
        mae_values = [13.55, 13.32, 13.97, 13.70, 13.79, 14.79, 11.05, 16.16]
        sample_counts = [2008, 849, 611, 67, 53, 318, 261, 74]

    # Sort by MAE
    sorted_data = sorted(zip(incident_types, mae_values, sample_counts), key=lambda x: x[1])
    incident_types = [x[0] for x in sorted_data]
    mae_values = [x[1] for x in sorted_data]
    sample_counts = [x[2] for x in sorted_data]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    x = np.arange(len(incident_types))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(incident_types)))

    bars = ax1.bar(x, mae_values, color=colors, edgecolor='black', linewidth=1.2)
    ax1.set_ylabel('MAE', color='black')
    ax1.set_xlabel('Incident Type')
    ax1.set_title('Performance by Incident Type (STGCNChebGraphConv)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(incident_types, rotation=45, ha='right')

    # Add sample counts as secondary axis
    ax2 = ax1.twinx()
    ax2.plot(x, sample_counts, 'ko-', linewidth=2, markersize=8, label='Sample Count')
    ax2.set_ylabel('Sample Count', color='black')

    for bar, val in zip(bars, mae_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                 f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '4_incident_type_analysis.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / '4_incident_type_analysis.png'}")


# =============================================================================
# 5. Summary Table
# =============================================================================
def create_summary_table():
    """Create a summary comparison table"""

    # Load actual data
    node_df = pd.read_csv(RESULT_DIR / "node_identity_exp.csv") if (RESULT_DIR / "node_identity_exp.csv").exists() else None
    coreset_df = pd.read_csv(RESULT_DIR / "coreset.csv") if (RESULT_DIR / "coreset.csv").exists() else None

    # Get best results
    data = []

    # Node Identity baseline
    if node_df is not None:
        baseline = node_df[node_df['model'] == 'STGCNChebGraphConv'].iloc[0]
        data.append(['STGCNChebGraphConv (Baseline)', f"{baseline['MAE_mean']:.2f}",
                     f"{baseline['MAPE_mean']*100:.1f}%", f"{baseline['RMSE_mean']:.2f}", 'Baseline'])

        for _, row in node_df[node_df['model'] != 'STGCNChebGraphConv'].iterrows():
            diff = (row['MAE_mean'] - baseline['MAE_mean']) / baseline['MAE_mean'] * 100
            sign = '+' if diff > 0 else ''
            data.append([row['model'], f"{row['MAE_mean']:.2f}",
                        f"{row['MAPE_mean']*100:.1f}%", f"{row['RMSE_mean']:.2f}",
                        f"{sign}{diff:.1f}% MAE"])

    # Experience Replay best
    data.append(['Experience Replay (Best)', '13.84', '19.2%', '26.07', '-22.7% MAE'])

    # Coreset best
    if coreset_df is not None:
        coreset_df = coreset_df[coreset_df['MAE_mean'] > 0]
        best_coreset = coreset_df.loc[coreset_df['MAE_mean'].idxmin()]
        if node_df is not None:
            diff = (best_coreset['MAE_mean'] - baseline['MAE_mean']) / baseline['MAE_mean'] * 100
            sign = '+' if diff > 0 else ''
            data.append([f"Coreset {best_coreset['model']} (Best)", f"{best_coreset['MAE_mean']:.2f}",
                        f"{best_coreset['MAPE_mean']*100:.1f}%", f"{best_coreset['RMSE_mean']:.2f}",
                        f"{sign}{diff:.1f}% MAE"])

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis('off')

    columns = ['Model/Method', 'MAE', 'MAPE', 'RMSE', 'vs Baseline']

    table = ax.table(cellText=data, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Style header
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    # Highlight improvements (negative %)
    for i, row in enumerate(data):
        if 'MAE' in row[-1] and row[-1].startswith('-'):
            table[(i+1, 1)].set_facecolor('#d4edda')
            table[(i+1, 4)].set_facecolor('#d4edda')

    plt.title('Summary: Model Performance Comparison', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '5_summary_table.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / '5_summary_table.png'}")


# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    print("Generating presentation plots with accurate data...")
    print("=" * 50)

    plot_node_identity_experiment()
    plot_experience_replay_experiment()
    plot_coreset_experiment()
    plot_incident_analysis()
    create_summary_table()

    print("=" * 50)
    print(f"All plots saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    for f in sorted(OUTPUT_DIR.glob('*.png')):
        print(f"  - {f.name}")
