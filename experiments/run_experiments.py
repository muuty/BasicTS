#!/usr/bin/env python3
"""
실험 실행 스크립트 (파라미터 sweep 및 병렬 실행 지원)

사용법:
    # 순차 실행 (GPU 1개)
    python experiments/run_experiments.py --config experiments/configs/config.yaml -g 0
    
    # 병렬 실행 (GPU 2개 → 2개 동시 실행)
    python experiments/run_experiments.py --config experiments/configs/config.yaml -g 0,1
"""

import argparse
import yaml
from pathlib import Path
import os
import sys

sys.path.append(os.path.abspath(__file__ + '/../..'))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from experiments.utils import (
    start_tensorboard,
    collect_rows_from_checkpoints,
    save_rows_to_csv,
    expand_experiments,
    print_experiment_header,
    print_experiment_summary,
)
from experiments.utils import run_experiments_parallel


def main():
    parser = argparse.ArgumentParser(description='실험 실행 스크립트 (파라미터 sweep 및 병렬 실행 지원)')
    parser.add_argument('--config', '-c', required=True, help='YAML 설정 파일')
    parser.add_argument('-g', '--gpu', default='0', help='GPU ID 또는 GPU 리스트 (예: 0 또는 0,1,2)')
    args = parser.parse_args()

    # Config 로드
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # GPU 리스트 파싱
    gpus = [g.strip() for g in args.gpu.split(',') if g.strip() != '']
    num_workers = len(gpus)
    
    # 디렉토리 설정
    exp_name = Path(args.config).stem
    checkpoint_dir = Path('checkpoints') / exp_name
    results_dir = Path('experiments_results') / exp_name
    generated_config_dir = results_dir / 'generated_configs'
    
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    generated_config_dir.mkdir(parents=True, exist_ok=True)

    # 실험 목록 확장 (config 파일 생성 포함)
    all_runs = expand_experiments(config, generated_config_dir)

    # TensorBoard 시작
    tensorboard_url = start_tensorboard(logdir=str(checkpoint_dir))
    
    # 헤더 출력
    print_experiment_header(
        exp_name=exp_name,
        tensorboard_url=tensorboard_url,
        checkpoint_dir=checkpoint_dir,
        total_runs=len(all_runs),
        results_dir=results_dir,
        num_workers=num_workers,
        gpus=gpus,
    )

    # 큐 기반 실행
    run_experiments_parallel(
        runs=all_runs,
        gpus=gpus,
        exp_name=exp_name,
        results_dir=results_dir,
    )

    # 결과 수집 및 저장
    rows = collect_rows_from_checkpoints(checkpoint_dir)
    csv_path = results_dir / f"{exp_name}_results.csv"
    save_rows_to_csv(rows, csv_path)
    
    # 요약 출력
    print_experiment_summary(
        rows=rows,
        tensorboard_url=tensorboard_url,
        results_dir=results_dir,
        csv_path=csv_path,
    )


if __name__ == '__main__':
    main()