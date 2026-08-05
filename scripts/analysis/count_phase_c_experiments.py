"""
Phase C 실험 완료 현황 집계 스크립트

각 checkpoint 디렉토리의 cfg.txt에서 실험 설정을 파싱하고,
test_metrics.json 유무로 완료 여부를 판단합니다.
"""

import os
import re
import json
from pathlib import Path
from collections import defaultdict
import itertools

CKPT_ROOT = Path("/home/uqtyu7/github/BasicTS/checkpoints/phase_c_method_comparison")

# Phase C 계획 (phase_c_method_comparison.yaml 기준)
PLANNED = {
    "models": ["STGCNChebGraphConv", "AGCRN", "DCRNN", "STID", "STAEformer"],
    "datasets": ["SAN_BERNARDINO", "CONTRA_COSTA"],
    "smart_methods": ["k_center", "k_medoids", "graph_cut"],
    "baseline_methods": ["random", "stride", "recent"],
    "ratios": [0.3, 0.5, 0.7],
    "seeds": [42, 123, 456],
    "full_ratio": [1.0],
}


def parse_cfg_txt(cfg_path):
    """cfg.txt에서 주요 설정을 파싱"""
    info = {}
    with open(cfg_path, "r") as f:
        content = f.read()

    # Model name (MODEL: 섹션 아래의 NAME)
    m = re.search(r"MODEL:\s*\n\s+NAME:\s+(\w+)", content)
    if m:
        info["model"] = m.group(1)

    # Dataset
    m = re.search(r"dataset_name:\s+xtraffic/(\w+)", content)
    if m:
        info["dataset"] = m.group(1)

    # Coreset settings (CORESET: 섹션)
    m = re.search(r"SELECTION_STRATEGY:\s+(\w+)", content)
    if m:
        info["strategy"] = m.group(1)

    m = re.search(r"SELECTION_RATIO:\s+([\d.]+)", content)
    if m:
        info["ratio"] = float(m.group(1))

    m = re.search(r"DISTANCE_TYPE:\s+(\w+)", content)
    if m:
        info["distance"] = m.group(1)

    # Coreset SEED (CORESET 섹션 내부)
    m = re.search(r"CORESET:.*?SEED:\s+(\d+)", content, re.DOTALL)
    if m:
        info["coreset_seed"] = int(m.group(1))

    # ENV SEED
    m = re.search(r"ENV:.*?SEED:\s+(\d+)", content, re.DOTALL)
    if m:
        info["env_seed"] = int(m.group(1))

    return info


def main():
    # 1. 모든 실험 디렉토리 스캔
    experiments = []
    for cfg_path in CKPT_ROOT.rglob("cfg.txt"):
        exp_dir = cfg_path.parent
        info = parse_cfg_txt(cfg_path)
        info["dir"] = str(exp_dir)
        info["has_test_metrics"] = (exp_dir / "test_metrics.json").exists()
        info["has_best_model"] = any(exp_dir.glob("*best_val*"))
        experiments.append(info)

    print(f"총 스캔된 실험 디렉토리: {len(experiments)}")
    print(f"  - test_metrics.json 있음 (완료): {sum(1 for e in experiments if e['has_test_metrics'])}")
    print(f"  - best_model 있음: {sum(1 for e in experiments if e['has_best_model'])}")
    print(f"  - 둘 다 없음 (실패): {sum(1 for e in experiments if not e['has_test_metrics'] and not e['has_best_model'])}")
    print()

    # 2. 완료된 실험만 필터 (중복 제거: 같은 config에 여러 시도가 있을 수 있음)
    completed = [e for e in experiments if e["has_test_metrics"]]

    # 고유 키로 중복 체크
    unique_keys = set()
    unique_completed = []
    duplicates = 0
    for e in completed:
        key = (e.get("model"), e.get("dataset"), e.get("strategy"),
               e.get("ratio"), e.get("coreset_seed"))
        if key in unique_keys:
            duplicates += 1
        else:
            unique_keys.add(key)
            unique_completed.append(e)

    print(f"완료된 실험 (중복 포함): {len(completed)}")
    print(f"완료된 실험 (중복 제거): {len(unique_completed)}")
    print(f"중복 실험: {duplicates}")
    print()

    # 3. 모델별 집계
    print("=" * 80)
    print("모델별 완료 현황")
    print("=" * 80)
    by_model = defaultdict(list)
    for e in unique_completed:
        by_model[e.get("model", "unknown")].append(e)

    for model in PLANNED["models"]:
        exps = by_model.get(model, [])
        # 모델별 상세
        model_name = model
        if model == "STGCNChebGraphConv":
            model_name = "STGCN"
        print(f"\n{model_name} ({model}): {len(exps)}개 완료")

        by_dataset = defaultdict(list)
        for e in exps:
            by_dataset[e.get("dataset", "unknown")].append(e)

        for ds in PLANNED["datasets"]:
            ds_exps = by_dataset.get(ds, [])
            by_strategy = defaultdict(list)
            for e in ds_exps:
                by_strategy[(e.get("strategy"), e.get("ratio"))].append(e)

            print(f"  {ds}: {len(ds_exps)}개")

            # 전략 × 비율 매트릭스
            all_strategies = PLANNED["smart_methods"] + PLANNED["baseline_methods"] + ["random"]
            all_ratios = [0.3, 0.5, 0.7, 1.0]

            print(f"    {'strategy':<15} {'r=0.3':>6} {'r=0.5':>6} {'r=0.7':>6} {'r=1.0':>6}")
            print(f"    {'-'*15} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")

            for strategy in ["k_center", "k_medoids", "graph_cut", "random", "stride", "recent"]:
                row = f"    {strategy:<15}"
                for ratio in all_ratios:
                    count = len(by_strategy.get((strategy, ratio), []))
                    seeds = sorted([e.get("coreset_seed") for e in by_strategy.get((strategy, ratio), [])])
                    if count == 0:
                        row += f"  {'--':>4}"
                    else:
                        row += f"  {count:>2}/3 "
                row += f"  seeds: {seeds}" if count > 0 else ""
                print(row)

    # 4. 계획 대비 누락 분석
    print("\n" + "=" * 80)
    print("계획 대비 누락 실험")
    print("=" * 80)

    missing = []

    for model in PLANNED["models"]:
        for ds in PLANNED["datasets"]:
            # Smart methods: 3 strategies × 3 ratios × 3 seeds
            for strategy in PLANNED["smart_methods"]:
                for ratio in PLANNED["ratios"]:
                    for seed in PLANNED["seeds"]:
                        key = (model, ds, strategy, ratio, seed)
                        if key not in unique_keys:
                            missing.append(key)

            # Baseline methods: 3 strategies × 3 ratios × 3 seeds
            for strategy in PLANNED["baseline_methods"]:
                for ratio in PLANNED["ratios"]:
                    for seed in PLANNED["seeds"]:
                        key = (model, ds, strategy, ratio, seed)
                        if key not in unique_keys:
                            missing.append(key)

            # Full data: random × 1.0 × 3 seeds
            for seed in PLANNED["seeds"]:
                key = (model, ds, "random", 1.0, seed)
                if key not in unique_keys:
                    missing.append(key)

    total_planned = len(PLANNED["models"]) * len(PLANNED["datasets"]) * (
        len(PLANNED["smart_methods"]) * len(PLANNED["ratios"]) * len(PLANNED["seeds"]) +
        len(PLANNED["baseline_methods"]) * len(PLANNED["ratios"]) * len(PLANNED["seeds"]) +
        1 * 1 * len(PLANNED["seeds"])  # full data
    )

    print(f"\n총 계획: {total_planned}개")
    print(f"완료 (고유): {len(unique_completed)}개")
    print(f"누락: {len(missing)}개")
    print(f"완료율: {len(unique_completed) / total_planned * 100:.1f}%")

    if missing:
        print(f"\n누락 실험 상세 ({len(missing)}개):")
        # 모델별 그룹핑
        missing_by_model = defaultdict(list)
        for m in missing:
            missing_by_model[m[0]].append(m)

        for model in PLANNED["models"]:
            model_missing = missing_by_model.get(model, [])
            if model_missing:
                print(f"\n  {model}: {len(model_missing)}개 누락")
                # 데이터셋별
                for ds in PLANNED["datasets"]:
                    ds_missing = [m for m in model_missing if m[1] == ds]
                    if ds_missing:
                        print(f"    {ds}:")
                        for _, _, strategy, ratio, seed in sorted(ds_missing):
                            print(f"      - {strategy} r={ratio} seed={seed}")

    # 5. 요약 테이블
    print("\n" + "=" * 80)
    print("요약 테이블: 모델 × 데이터셋 (완료/계획)")
    print("=" * 80)

    per_dataset_planned = (
        len(PLANNED["smart_methods"]) * len(PLANNED["ratios"]) * len(PLANNED["seeds"]) +
        len(PLANNED["baseline_methods"]) * len(PLANNED["ratios"]) * len(PLANNED["seeds"]) +
        len(PLANNED["seeds"])  # full
    )

    header = f"{'Model':<20}"
    for ds in PLANNED["datasets"]:
        header += f"  {ds:>16}"
    header += f"  {'Total':>10}"
    print(header)
    print("-" * len(header))

    for model in PLANNED["models"]:
        row = f"{model:<20}"
        model_total = 0
        for ds in PLANNED["datasets"]:
            count = sum(1 for e in unique_completed
                       if e.get("model") == model and e.get("dataset") == ds)
            model_total += count
            row += f"  {count:>6}/{per_dataset_planned:<8}"
        row += f"  {model_total:>4}/{per_dataset_planned*2}"
        print(row)

    # 6. Strategy 요약
    print("\n" + "=" * 80)
    print("Strategy별 완료 현황 (전체 모델/데이터셋 합산)")
    print("=" * 80)

    by_strategy_ratio = defaultdict(int)
    for e in unique_completed:
        key = (e.get("strategy"), e.get("ratio"))
        by_strategy_ratio[key] += 1

    per_cell_planned = len(PLANNED["models"]) * len(PLANNED["datasets"]) * len(PLANNED["seeds"])  # 5*2*3=30

    print(f"{'strategy':<15} {'r=0.3':>8} {'r=0.5':>8} {'r=0.7':>8} {'r=1.0':>8} {'Total':>8}")
    print(f"{'-'*15} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for strategy in ["k_center", "k_medoids", "graph_cut", "random", "stride", "recent"]:
        row = f"{strategy:<15}"
        stotal = 0
        for ratio in [0.3, 0.5, 0.7, 1.0]:
            count = by_strategy_ratio.get((strategy, ratio), 0)
            stotal += count
            if ratio == 1.0 and strategy != "random":
                row += f"  {'--':>6}"
            else:
                row += f"  {count:>3}/{per_cell_planned:<3}"
        row += f"  {stotal:>6}"
        print(row)


if __name__ == "__main__":
    main()
