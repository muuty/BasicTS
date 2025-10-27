import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple

def load_learning_rate_metrics() -> Tuple[List[float], List[float]]:
    """learning_rate별 파일을 찾아 overall_MAE를 로드합니다.
    
    Returns:
        learning_rates: learning_rate 리스트
        overall_maes: overall_MAE 리스트
    """
    overall_metrics_files = []
    incidents_metrics_files = []
    
    pattern = r"SAN_BERNARDINO_CL_([0-9.]+)_100_12_12"
    
    for root, dirs, files in os.walk("/home/uqtyu7/github/BasicTS/checkpoints/STGCNChebGraphConv/xtraffic/"):
        match = re.search(pattern, root)
        if match:
            overall_metrics_files.append((float(match.group(1)), root + "/test_metrics.json"))
            incidents_metrics_files.append((float(match.group(1)), root + "/test_incident_metrics.json"))
    
    # learning_rate순으로 정렬
    overall_metrics_files.sort(key=lambda x: x[0])
    incidents_metrics_files.sort(key=lambda x: x[0])
    
    learning_rates = []
    overall_maes = []
    
    for lr, file_path in overall_metrics_files:
        try:
            with open(file_path, 'r') as f:
                metrics = json.load(f)
                overall_mae = metrics.get('overall', {}).get('MAE', None)
                if overall_mae is not None:
                    learning_rates.append(lr)
                    overall_maes.append(overall_mae)
                    print(f"Learning Rate: {lr}, Overall MAE: {overall_mae:.4f}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return learning_rates, overall_maes


def plot_learning_rate_comparison(learning_rates: List[float], overall_maes: List[float]) -> None:
    """learning_rate에 따른 overall_MAE 변화를 그래프로 표시합니다.
    
    Args:
        learning_rates: learning_rate 리스트
        overall_maes: overall_MAE 리스트
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Line plot
    ax.plot(learning_rates, overall_maes, marker='o', linewidth=2.5, markersize=10, 
            color='steelblue', markerfacecolor='lightcoral', markeredgewidth=2)
    
    # 각 점에 값 표시
    for i, (lr, mae) in enumerate(zip(learning_rates, overall_maes)):
        ax.text(lr, mae + 0.1, f'{mae:.3f}', ha='center', va='bottom', 
                fontsize=10, fontweight='bold')
    
    # 그래프 설정
    ax.set_xlabel('Learning Rate', fontsize=14, fontweight='bold')
    ax.set_ylabel('Overall MAE', fontsize=14, fontweight='bold')
    ax.set_title('Overall MAE vs Learning Rate', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # x축을 log scale로 변경 (학습률은 보통 로그 스케일)
    ax.set_xscale('log')
    
    plt.tight_layout()
    plt.show()
    
    # 요약 통계 출력
    print("\n" + "="*60)
    print("Learning Rate Comparison Summary")
    print("="*60)
    print(f"{'Learning Rate':<15} {'Overall MAE':<15} {'Rank':<10}")
    print("-"*60)
    
    # MAE가 낮은 순으로 정렬
    sorted_data = sorted(zip(learning_rates, overall_maes), key=lambda x: x[1])
    
    for rank, (lr, mae) in enumerate(sorted_data, 1):
        marker = "🏆" if rank == 1 else f"{rank}"
        print(f"{lr:<15.4f} {mae:<15.4f} {marker:<10}")
    
    # 통계
    print("-"*60)
    print(f"Best: LR={sorted_data[0][0]:.4f}, MAE={sorted_data[0][1]:.4f}")
    print(f"Worst: LR={sorted_data[-1][0]:.4f}, MAE={sorted_data[-1][1]:.4f}")
    print(f"Mean MAE: {np.mean(overall_maes):.4f}")
    print(f"Std MAE: {np.std(overall_maes):.4f}")


def create_comparison_table(learning_rates: List[float], overall_maes: List[float]) -> pd.DataFrame:
    """비교 테이블을 DataFrame으로 생성합니다.
    
    Args:
        learning_rates: learning_rate 리스트
        overall_maes: overall_MAE 리스트
        
    Returns:
        비교 결과를 담은 DataFrame
    """
    df = pd.DataFrame({
        'Learning_Rate': learning_rates,
        'Overall_MAE': overall_maes
    })
    
    df = df.sort_values('Learning_Rate')
    df['Rank'] = df['Overall_MAE'].rank(ascending=True).astype(int)
    df = df.sort_values('Rank')
    
    return df


if __name__ == "__main__":
    # 데이터 로드
    print("Loading metrics files...\n")
    learning_rates, overall_maes = load_learning_rate_metrics()
    
    if not learning_rates:
        print("No valid metrics files found!")
    else:
        print(f"\nFound {len(learning_rates)} valid models\n")
        
        # 그래프 그리기
        plot_learning_rate_comparison(learning_rates, overall_maes)
        
        # 비교 테이블 생성
        comparison_df = create_comparison_table(learning_rates, overall_maes)
        print("\n" + "="*60)
        print("Comparison Table")
        print("="*60)
        print(comparison_df.to_string(index=False))
