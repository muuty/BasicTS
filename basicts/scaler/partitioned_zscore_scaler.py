import json
from typing import List, Optional

import numpy as np
import torch

from .base_scaler import BaseScaler


class PartitionedZScoreScaler(BaseScaler):
    """
    Client-wise (partitioned) Z-score scaler.

    - Each node belongs to exactly one client (given by client_nodes_list).
    - Statistics (mean/std) are computed using ONLY the nodes owned by each client
      from the TRAIN split portion of the dataset.

    Behavior matches original ZScoreScaler:
    - Only target_channel is normalized (default 0).
    - transform modifies input_data[..., target_channel] in-place.
    - inverse_transform returns a cloned tensor with inverse transform applied.

    Important:
    - Internally we always store mean/std as shape [1, N] tensors so we can apply
      per-node scaling even when norm_each_channel=False (client-scalar stats are
      broadcast to all nodes in that client).
    """

    def __init__(
        self,
        dataset_name: str,
        train_ratio: float,
        norm_each_channel: bool,
        rescale: bool,
        client_nodes_list: List[List[int]],
        target_channel: int = 0,
    ):
        super().__init__(dataset_name, train_ratio, norm_each_channel, rescale)
        self.target_channel = target_channel

        # load dataset description and data
        description_file_path = f"datasets/{dataset_name}/desc.json"
        with open(description_file_path, "r") as f:
            description = json.load(f)
        data_file_path = f"datasets/{dataset_name}/data.dat"
        data = np.memmap(
            data_file_path, dtype="float32", mode="r", shape=tuple(description["shape"])
        )

        # Expect data shape like [T, N, C] (Base ZScoreScaler used [:train_size, :, channel])
        train_size = int(len(data) * train_ratio)
        train_data = data[:train_size, :, self.target_channel].copy()  # [T_train, N]

        N = train_data.shape[1]
        mean_full = np.zeros((1, N), dtype=np.float32)
        std_full = np.ones((1, N), dtype=np.float32)

        # fill mean/std per node using client-local stats
        for nodes in client_nodes_list:
            if len(nodes) == 0:
                continue
            sub = train_data[:, nodes]  # [T_train, n_client]

            if norm_each_channel:
                # per-node stats within the client (same as global per-node, but computed client-local)
                mean = np.mean(sub, axis=0, keepdims=True)  # [1, n_client]
                std = np.std(sub, axis=0, keepdims=True)    # [1, n_client]
                std[std == 0] = 1.0
                mean_full[:, nodes] = mean
                std_full[:, nodes] = std
            else:
                # client-scalar stats over its nodes
                m = float(np.mean(sub))
                s = float(np.std(sub))
                if s == 0:
                    s = 1.0
                mean_full[:, nodes] = m
                std_full[:, nodes] = s

        self.mean = torch.tensor(mean_full)
        self.std = torch.tensor(std_full)

    def transform(self, input_data: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(input_data.device)
        std = self.std.to(input_data.device)
        input_data[..., self.target_channel] = (input_data[..., self.target_channel] - mean) / std
        return input_data

    def inverse_transform(self, input_data: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(input_data.device)
        std = self.std.to(input_data.device)
        input_data = input_data.clone()
        input_data[..., self.target_channel] = input_data[..., self.target_channel] * std + mean
        return input_data

class ClientZScoreScaler(BaseScaler):
    """
    Z-score scaler but statistics are computed using ONLY client_nodes.

    - input_data expected node dim matches the client subset size (n_client)
    - mean/std shapes:
        norm_each_channel=True  -> [1, n_client]
        norm_each_channel=False -> scalar
    """

    def __init__(
        self,
        dataset_name: str,
        train_ratio: float,
        norm_each_channel: bool,
        rescale: bool,
        client_nodes: List[int],
        target_channel: int = 0,
    ):
        super().__init__(dataset_name, train_ratio, norm_each_channel, rescale)
        self.target_channel = target_channel
        self.client_nodes = list(client_nodes)

        # load dataset description and data (same as original)
        description_file_path = f"datasets/{dataset_name}/desc.json"
        with open(description_file_path, "r") as f:
            description = json.load(f)
        data_file_path = f"datasets/{dataset_name}/data.dat"
        data = np.memmap(data_file_path, dtype="float32", mode="r", shape=tuple(description["shape"]))

        train_size = int(len(data) * train_ratio)
        train_data = data[:train_size, :, self.target_channel].copy()  # [T_train, N]
        train_data = train_data[:, self.client_nodes]                  # [T_train, n_client]

        if norm_each_channel:
            mean = np.mean(train_data, axis=0, keepdims=True)  # [1, n_client]
            std = np.std(train_data, axis=0, keepdims=True)    # [1, n_client]
            std[std == 0] = 1.0
        else:
            mean = float(np.mean(train_data))
            std = float(np.std(train_data))
            if std == 0:
                std = 1.0

        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def transform(self, input_data: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(input_data.device)
        std = self.std.to(input_data.device)
        input_data[..., self.target_channel] = (input_data[..., self.target_channel] - mean) / std
        return input_data

    def inverse_transform(self, input_data: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(input_data.device)
        std = self.std.to(input_data.device)
        input_data = input_data.clone()
        input_data[..., self.target_channel] = input_data[..., self.target_channel] * std + mean
        return input_data