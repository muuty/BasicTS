import torch
import numpy as np
from typing import List, Dict
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from selection.outlier_removal.base import OutlierRemoval
from selection.outlier_removal.model_based import ModelBasedOutlierRemoval
from selection.outlier_removal.hdbscan import HDBSCANOutlierRemoval


def get_outlier_removal(type: str, outlier_removal_ratio: float, data_config: dict, model_config: dict) -> OutlierRemoval:
    if type == "model_based":
        return ModelBasedOutlierRemoval(
            model_path=data_config.get('OUTLIER_REMOVAL_MODEL_PATH'),
            removal_ratio=outlier_removal_ratio,
            data_config=data_config,
            model_config=model_config)
    elif type == "hdbscan":
        return HDBSCANOutlierRemoval(
            outlier_removal_ratio,
            data_config)
    return None
