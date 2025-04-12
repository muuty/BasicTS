import numpy as np
import torch

from basicts.data.simple_tsf_dataset import TimeSeriesForecastingDataset


def get_distance_matrix(x_np: np.ndarray, device: str = 'cuda', batch_size: int = 512) -> np.ndarray:
    N, D = x_np.shape
    dist_mat = torch.empty((N, N), dtype=torch.float32, device='cpu')  # CPU에 저장

    # Precompute norms of all samples (on CPU for now)
    x_norm_cpu = np.sum(x_np**2, axis=1)

    for i in range(0, N, batch_size):
        end_i = min(i + batch_size, N)
        x_batch = torch.from_numpy(x_np[i:end_i]).to(device).float()         # (B, D)
        x_batch_norm = torch.from_numpy(x_norm_cpu[i:end_i]).to(device).float().unsqueeze(1)  # (B, 1)

        # Compute against full x_np (CPU-side → GPU-side in chunks too)
        for j in range(0, N, batch_size):
            end_j = min(j + batch_size, N)
            x_ref = torch.from_numpy(x_np[j:end_j]).to(device).float()  # (B, D)
            x_ref_norm = torch.from_numpy(x_norm_cpu[j:end_j]).to(device).float().unsqueeze(0)  # (1, B)

            dot = x_batch @ x_ref.T  # (B1, B2)
            dist_sq = x_batch_norm + x_ref_norm - 2 * dot
            dist_sq = torch.clamp(dist_sq, min=0.0)
            dist = torch.sqrt(dist_sq).cpu()  # move to CPU

            dist_mat[i:end_i, j:end_j] = dist

    return dist_mat.numpy()        # GPU → CPU


def get_normalized_input(dataset: TimeSeriesForecastingDataset, model_config: dict) -> np.ndarray:
    mean = np.mean(dataset.data, axis=(0,1), keepdims=True)
    std = np.std(dataset.data, axis=(0,1), keepdims=True)
    std[std == 0] = 1.0

    def transform(input_data):
        return (input_data - mean) / std

    inputs = np.array([
        transform(dataset[i]['inputs'])[:, :, model_config.MODEL.FORWARD_FEATURES] +
        transform(dataset[i]['target'])[:, :, model_config.MODEL.TARGET_FEATURES]
        for i in range(len(dataset))
    ])
    return inputs.reshape(len(dataset), -1)

def distance_to_similarity_rbf(distance_matrix: torch.Tensor, sigma: float = None) -> torch.Tensor:
    """
    Convert a distance matrix D into similarity matrix using RBF kernel.
    """
    if sigma is None:
        sigma = torch.median(distance_matrix).item()

    similarity_matrix = torch.exp(-distance_matrix ** 2 / (2 * sigma ** 2))
    return similarity_matrix
