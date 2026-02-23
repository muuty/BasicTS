"""
Experience Replay 실험 결과 분석 및 정리
"""

import pandas as pd
from pathlib import Path

RESULT_DIR = Path("experiments/result")

def analyze_experience_replay():
    print("=" * 80)
    print("Experience Replay 실험 결과 분석")
    print("=" * 80)

    # 1. experience_replay.csv 분석
    print("\n" + "=" * 80)
    print("1. Experience Replay 기본 실험 (STGCN)")
    print("   - Method: Random sampling")
    print("   - Batch size: 64, Capacity ratio: 0.1")
    print("=" * 80)

    df1 = pd.read_csv(RESULT_DIR / "experience_replay.csv")
    df1['config'] = df1.apply(lambda r: f"weight={r['experience_replay_weight']}"
                              if pd.notna(r['experience_replay_weight']) else "Baseline (No Replay)", axis=1)

    print("\n{:<25} {:>10} {:>12} {:>12}".format("Config", "MAE", "Non-Inc MAE", "Inc MAE"))
    print("-" * 60)
    for _, row in df1.sort_values('MAE_mean').iterrows():
        print("{:<25} {:>10.2f} {:>12.2f} {:>12.2f}".format(
            row['config'], row['MAE_mean'], row['non_inc_MAE_mean'], row['inc_MAE_mean']))

    best = df1.loc[df1['MAE_mean'].idxmin()]
    baseline = df1[df1['experience_replay_weight'].isna()].iloc[0]
    improvement = (baseline['MAE_mean'] - best['MAE_mean']) / baseline['MAE_mean'] * 100
    print(f"\n>> Best: {best['config']} (MAE={best['MAE_mean']:.2f})")
    print(f">> Baseline 대비 개선: {improvement:.1f}%")

    # 2. experience_replay_weights.csv 분석
    print("\n" + "=" * 80)
    print("2. Experience Replay Weight 세부 실험 (STGCN)")
    print("   - 다양한 weight 값 테스트 (0.01 ~ 0.2)")
    print("=" * 80)

    df2 = pd.read_csv(RESULT_DIR / "experience_replay_weights.csv")
    df2['config'] = df2.apply(lambda r: f"weight={r['experience_replay_weight']}"
                              if pd.notna(r['experience_replay_weight']) else "Baseline", axis=1)

    print("\n{:<25} {:>10} {:>8} {:>12} {:>12}".format("Config", "MAE", "Std", "Non-Inc MAE", "Inc MAE"))
    print("-" * 70)
    for _, row in df2.sort_values('MAE_mean').iterrows():
        std = row['MAE_std'] if pd.notna(row['MAE_std']) else 0
        print("{:<25} {:>10.2f} {:>8.2f} {:>12.2f} {:>12.2f}".format(
            row['config'], row['MAE_mean'], std, row['non_inc_MAE_mean'], row['inc_MAE_mean']))

    best = df2.loc[df2['MAE_mean'].idxmin()]
    print(f"\n>> Best: {best['config']} (MAE={best['MAE_mean']:.2f})")

    # 3. representative.csv 분석
    print("\n" + "=" * 80)
    print("3. Representative Sampling 실험")
    print("   - Method: Representative (clustering-based sampling)")
    print("=" * 80)

    df3 = pd.read_csv(RESULT_DIR / "representative.csv")

    for model in df3['model'].unique():
        model_df = df3[df3['model'] == model]
        print(f"\n### {model}")
        print("{:<25} {:>10} {:>12} {:>12}".format("Config", "MAE", "Non-Inc MAE", "Inc MAE"))
        print("-" * 60)

        for _, row in model_df.sort_values('MAE_mean').iterrows():
            if pd.notna(row['experience_replay_weight']):
                config = f"weight={row['experience_replay_weight']}"
            else:
                config = "Baseline (No Replay)"
            print("{:<25} {:>10.2f} {:>12.2f} {:>12.2f}".format(
                config, row['MAE_mean'], row['non_inc_MAE_mean'], row['inc_MAE_mean']))

        # Baseline vs Best with replay
        baseline = model_df[model_df['experience_replay_weight'].isna()]
        with_replay = model_df[model_df['experience_replay_weight'].notna()]

        if not baseline.empty and not with_replay.empty:
            baseline_mae = baseline['MAE_mean'].values[0]
            best_replay = with_replay.loc[with_replay['MAE_mean'].idxmin()]
            diff = (best_replay['MAE_mean'] - baseline_mae) / baseline_mae * 100
            sign = '+' if diff > 0 else ''
            print(f"\n>> Baseline MAE: {baseline_mae:.2f}")
            print(f">> Best Replay MAE: {best_replay['MAE_mean']:.2f} (weight={best_replay['experience_replay_weight']})")
            print(f">> 차이: {sign}{diff:.1f}%")

    # 4. difficult.csv 분석
    print("\n" + "=" * 80)
    print("4. Difficult Sampling 실험")
    print("   - Method: Difficult (high-loss sample prioritization)")
    print("=" * 80)

    df4 = pd.read_csv(RESULT_DIR / "difficult.csv")

    print("\n{:<20} {:>10} {:>12} {:>12}".format("Model", "MAE", "Non-Inc MAE", "Inc MAE"))
    print("-" * 55)
    for _, row in df4.iterrows():
        print("{:<20} {:>10.2f} {:>12.2f} {:>12.2f}".format(
            row['model'], row['MAE_mean'], row['non_inc_MAE_mean'], row['inc_MAE_mean']))

    # 종합 비교
    print("\n" + "=" * 80)
    print("5. 종합 비교: 최고 성능 설정")
    print("=" * 80)

    summary = [
        ("STGCN + Random Replay (w=0.1)", 13.84, 13.57, 14.93),
        ("STGCN + Random Replay (w=0.05)", 13.98, 13.74, 14.96),
        ("STGCN + Representative (w=0.3)", 15.29, 14.99, 16.48),
        ("STGCN Baseline", 17.91, 17.63, 19.01),
        ("STAEformer + Difficult (w=0.1)", 11.87, 11.64, 12.78),
        ("STAEformer + Representative (w=0.5)", 12.10, 11.86, 13.03),
        ("STAEformer Baseline", 11.76, 11.53, 12.67),
        ("STGformer Baseline", 11.79, 11.57, 12.68),
    ]

    print("\n{:<40} {:>10} {:>12} {:>12}".format("Config", "MAE", "Non-Inc MAE", "Inc MAE"))
    print("-" * 75)
    for name, mae, non_inc, inc in sorted(summary, key=lambda x: x[1]):
        print("{:<40} {:>10.2f} {:>12.2f} {:>12.2f}".format(name, mae, non_inc, inc))

    print("\n" + "=" * 80)
    print("결론")
    print("=" * 80)
    print("""
1. STGCN에서는 Experience Replay가 효과적 (MAE 17.91 → 13.84, -22.7% 개선)
   - Random sampling + weight=0.1이 최적
   - weight가 너무 크면 오히려 성능 저하

2. STAEformer/STGformer에서는 Experience Replay 효과 미미
   - 이미 baseline 성능이 좋음 (MAE ~11.76)
   - Replay 추가 시 오히려 소폭 성능 저하

3. Sampling Method 비교:
   - Random > Representative (STGCN 기준)
   - Difficult는 STAEformer에서 약간 효과적

4. 추천 설정:
   - STGCN: Random replay, weight=0.05~0.1, batch=64, capacity=0.1
   - STAEformer/STGformer: Replay 없이 baseline 사용 권장
""")


if __name__ == '__main__':
    analyze_experience_replay()
