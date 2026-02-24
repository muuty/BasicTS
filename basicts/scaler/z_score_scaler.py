import json
from typing import List, Union

import numpy as np
import torch

from .base_scaler import BaseScaler


class ZScoreScaler(BaseScaler):
    """
    ZScoreScaler performs Z-score normalization on the dataset.

    Supports single-channel (default, backward compatible) or multi-channel normalization
    via the `target_channel` parameter.

    Args:
        target_channel: Which channel(s) to normalize. Default 0.
            - int: single channel (backward compatible)
            - list[int]: multiple channels, each normalized independently
    """

    def __init__(self, dataset_name: str, train_ratio: float, norm_each_channel: bool, rescale: bool,
                 target_channel: Union[int, List[int]] = 0):
        super().__init__(dataset_name, train_ratio, norm_each_channel, rescale)

        # Normalize target_channel to always be a list internally
        if isinstance(target_channel, int):
            self.target_channels = [target_channel]
        else:
            self.target_channels = list(target_channel)

        # For backward compat: single-channel case keeps self.target_channel
        self.target_channel = self.target_channels[0]
        self._multi_channel = len(self.target_channels) > 1

        # Load data
        description_file_path = f'datasets/{dataset_name}/desc.json'
        with open(description_file_path, 'r') as f:
            description = json.load(f)
        data_file_path = f'datasets/{dataset_name}/data.dat'
        data = np.memmap(data_file_path, dtype='float32', mode='r', shape=tuple(description['shape']))

        train_size = int(len(data) * train_ratio)

        # Compute mean/std for each target channel
        self.means = {}
        self.stds = {}
        for ch in self.target_channels:
            train_data = data[:train_size, :, ch].copy()
            if norm_each_channel:
                mean = np.mean(train_data, axis=0, keepdims=True)
                std = np.std(train_data, axis=0, keepdims=True)
                std[std == 0] = 1.0
            else:
                mean = np.mean(train_data)
                std = np.std(train_data)
                if std == 0:
                    std = 1.0
            self.means[ch] = torch.tensor(mean)
            self.stds[ch] = torch.tensor(std)

        # Backward compat: self.mean / self.std point to the first channel
        self.mean = self.means[self.target_channels[0]]
        self.std = self.stds[self.target_channels[0]]

    def transform(self, input_data: torch.Tensor) -> torch.Tensor:
        for ch in self.target_channels:
            mean = self.means[ch].to(input_data.device)
            std = self.stds[ch].to(input_data.device)
            input_data[..., ch] = (input_data[..., ch] - mean) / std
        return input_data

    def inverse_transform(self, input_data: torch.Tensor) -> torch.Tensor:
        input_data = input_data.clone()
        for ch in self.target_channels:
            if ch >= input_data.shape[-1]:
                continue
            mean = self.means[ch].to(input_data.device)
            std = self.stds[ch].to(input_data.device)
            input_data[..., ch] = input_data[..., ch] * std + mean
        return input_data
