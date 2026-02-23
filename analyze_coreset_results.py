#!/usr/bin/env python
"""Analyze coreset experiment results."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def analyze_coreset_results():
    """Analyze and visualize coreset experiment results."""

    # Load results
    df = pd.read_csv('experiments/result/coreset.csv')

    print("=" * 60)
    print("CORESET EXPERIMENT ANALYSIS")
    print("=" * 60)

    # Basic information
    print(f"Total experiments: {len(df)}")
    print(f"Models tested: {df['model'].unique().tolist()}")
    print(f"Selection ratios: {sorted(df['coreset_selection_ratio'].unique())}")
    print()

    # Performance summary by model
    print("PERFORMANCE SUMMARY BY MODEL (Overall MAE)")
    print("-" * 50)

    model_performance = df.groupby('model')['MAE_mean'].agg(['min', 'max', 'mean']).round(3)
    model_performance['range'] = model_performance['max'] - model_performance['min']
    model_performance = model_performance.sort_values('mean')

    print(model_performance)
    print()

    # Best performance by selection ratio
    print("BEST PERFORMANCE BY SELECTION RATIO")
    print("-" * 50)

    ratio_analysis = []
    for ratio in sorted(df['coreset_selection_ratio'].unique()):
        ratio_data = df[df['coreset_selection_ratio'] == ratio]
        best_model = ratio_data.loc[ratio_data['MAE_mean'].idxmin()]

        ratio_analysis.append({
            'ratio': ratio,
            'best_model': best_model['model'],
            'mae_mean': round(best_model['MAE_mean'], 3),
            'mae_std': round(best_model['MAE_std'], 3),
            'non_incident_mae': round(best_model['non_inc_MAE_mean'], 3),
            'incident_mae': round(best_model['inc_MAE_mean'], 3)
        })

    ratio_df = pd.DataFrame(ratio_analysis)
    print(ratio_df.to_string(index=False))
    print()

    # Performance trends by model and ratio
    print("PERFORMANCE TRENDS BY MODEL")
    print("-" * 50)

    for model in df['model'].unique():
        model_data = df[df['model'] == model].sort_values('coreset_selection_ratio')
        print(f"\n{model}:")
        print("  Ratio  |  MAE    |  Non-Inc |  Inc")
        print("  -------|---------|----------|--------")

        for _, row in model_data.iterrows():
            print(f"  {row['coreset_selection_ratio']:5.1f}  | {row['MAE_mean']:7.3f} | {row['non_inc_MAE_mean']:8.3f} | {row['inc_MAE_mean']:7.3f}")

    print()

    # Optimal selection ratios
    print("OPTIMAL SELECTION RATIO ANALYSIS")
    print("-" * 50)

    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        best_row = model_data.loc[model_data['MAE_mean'].idxmin()]

        print(f"{model}:")
        print(f"  Best ratio: {best_row['coreset_selection_ratio']}")
        print(f"  Best MAE: {best_row['MAE_mean']:.3f} (±{best_row['MAE_std']:.3f})")

        # Compare with full dataset (ratio=1.0)
        full_data = model_data[model_data['coreset_selection_ratio'] == 1.0]
        if not full_data.empty:
            full_mae = full_data['MAE_mean'].iloc[0]
            improvement = ((full_mae - best_row['MAE_mean']) / full_mae) * 100
            print(f"  Improvement vs full data: {improvement:.1f}%")

        print()

    # Incident vs Non-incident analysis
    print("INCIDENT vs NON-INCIDENT PERFORMANCE")
    print("-" * 50)

    df['incident_improvement'] = ((df['non_inc_MAE_mean'] - df['inc_MAE_mean']) / df['non_inc_MAE_mean']) * 100

    incident_analysis = df.groupby('model')['incident_improvement'].agg(['mean', 'std']).round(1)
    incident_analysis.columns = ['avg_improvement_%', 'std_improvement_%']
    print("Average improvement in incident periods vs non-incident:")
    print(incident_analysis.sort_values('avg_improvement_%', ascending=False))
    print()

    # Efficiency analysis (performance vs data usage)
    print("EFFICIENCY ANALYSIS (Performance per Data Used)")
    print("-" * 50)

    df['efficiency'] = (1 / df['MAE_mean']) / df['coreset_selection_ratio']

    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        best_efficiency = model_data.loc[model_data['efficiency'].idxmax()]

        print(f"{model}:")
        print(f"  Most efficient ratio: {best_efficiency['coreset_selection_ratio']}")
        print(f"  MAE: {best_efficiency['MAE_mean']:.3f}")
        print(f"  Efficiency score: {best_efficiency['efficiency']:.3f}")
        print()

    return df

if __name__ == "__main__":
    df = analyze_coreset_results()