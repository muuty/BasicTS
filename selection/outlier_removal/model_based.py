import torch
import numpy as np
from typing import List
from torch.utils.data import Dataset
from selection.outlier_removal.base import OutlierRemoval

from torch.utils.data import DataLoader
def collate_fn(batch):
    indices = [sample["index"] for sample in batch]
    inputs = torch.stack([torch.Tensor(sample["inputs"]) for sample in batch])
    targets = torch.stack([torch.Tensor(sample["target"]) for sample in batch])

    return {"inputs": inputs, "target": targets, "indices": indices}

class ModelBasedOutlierRemoval(OutlierRemoval):
    def __init__(self, model_path: str,
                 removal_ratio: float,
                 data_config: dict,
                 model_config: dict):
        """
        모델 기반 Outlier 제거 클래스.
        모델을 사용하여 예측을 수행한 후, 오차가 큰 샘플을 제거함.

        Args:
            model_path (str): 저장된 모델 파일 경로 (예: 'checkpoints/model.pt')
            top_n (int): 제거할 Outlier 샘플 개수
            device (str): 모델 실행에 사용할 디바이스 (CPU/GPU)
        """
        self.model_path = model_path
        self.removal_ratio = removal_ratio
        self.data_config = data_config
        self.runner = data_config.get('OUTLIER_REMOVAL_RUNNER')(model_config)
        self.runner.load_model(self.model_path)

    def _compute_errors(self, dataset: Dataset) -> dict:
        data_loader = DataLoader(
            dataset,
            collate_fn=collate_fn,
            batch_size=self.data_config.get('BATCH_SIZE', 1),
            num_workers=self.data_config.get('NUM_WORKERS', 0),
            pin_memory=self.data_config.get('PIN_MEMORY', False)
        )


        previous_errors = {}
        for data in data_loader:
            indices = data["indices"]  # 배치 내 샘플의 원본 인덱스
            outputs = self.runner.forward(data, epoch=None, iter_num=None, train=False)

            predictions, targets = outputs['prediction'], outputs['target']
            errors = torch.abs((targets - predictions)).mean(dim=[1, 2, 3])  # .item() 제거

            # 인덱스별로 previous_error 저장
            for idx, error in zip(indices, errors):
                previous_errors[idx] = error.item()  # 여기에서 개별적으로 .item() 호출

        return previous_errors

    def get_normal_indices(self, dataset: Dataset) -> List[int]:
        """모델 기반 예측 에러를 이용하여 Outlier를 제거한 인덱스 리스트 반환"""
        errors = self._compute_errors(dataset)
        sorted_errors = sorted(errors.items(), key=lambda x: x[1], reverse=True)

        # 상위 5%의 인덱스 추출
        outlier_count = int(self.removal_ratio * len(sorted_errors))  # 개수 계산
        outlier_indicies = [idx for idx, _ in sorted_errors[:outlier_count]]

        # 정상 데이터 인덱스 반환
        inlier_indices = list(set(range(len(dataset))) - set(outlier_indicies))

        return inlier_indices
