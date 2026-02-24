"""MICL: Missing-Invariant Contrastive Learning.

Asymmetric teacher-student SSL that learns representations
invariant to missing data patterns.

Components:
- MLPEncoder: pointwise MLP encoder (registered for downstream reuse)
- MICLModel: full pretraining model (encoder + predictor + augmentation)
- micl_loss: observation-weighted cosine similarity loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.ContextContrastive.arch.base_encoder import BaseRepresentationEncoder, register_encoder
from .augmentation import MissingAugmentation


@register_encoder('MLPEncoder')
class MLPEncoder(BaseRepresentationEncoder):
    """Pointwise MLP encoder for traffic data.

    Transforms per-(node, timestep) features into learned representations.
    Input: [B, T, N, input_dim] -> Output: [B, T, N, d_model]
    """

    def __init__(self, input_dim=8, d_model=32, hidden_dim=64, **kwargs):
        super().__init__(input_dim, d_model, **kwargs)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, d_model)

    def encode(self, x, **kwargs):
        return self.fc2(self.relu(self.fc1(x)))

    def get_config(self):
        config = super().get_config()
        config['type'] = 'MLPEncoder'
        config['hidden_dim'] = self.fc1.out_features
        return config

    def _extract_encoder_weights(self, state_dict):
        """Extract encoder weights from MICL pretrain checkpoint."""
        encoder_state = {}
        for key, value in state_dict.items():
            if key.startswith('encoder.'):
                encoder_state[key[len('encoder.'):]] = value
        return encoder_state


class MICLModel(nn.Module):
    """MICL pretraining model.

    Asymmetric teacher-student:
    - Teacher (view1): original data (stop-gradient)
    - Student (view2): augmented with additional missing patterns
    - Loss: cosine similarity from predictor(z_student) -> stopgrad(z_teacher)
    """

    def __init__(self, input_dim=8, hidden_dim=64, output_dim=32,
                 pred_hidden=16, aug_types=('node_death', 'intermittent', 'block'),
                 rate_range=(0.1, 0.5), curriculum=None):
        super().__init__()

        self.encoder = MLPEncoder(input_dim=input_dim, d_model=output_dim,
                                  hidden_dim=hidden_dim)

        self.predictor = nn.Sequential(
            nn.Linear(output_dim, pred_hidden),
            nn.ReLU(),
            nn.Linear(pred_hidden, output_dim),
        )

        self.augmentation = MissingAugmentation(
            aug_types=aug_types,
            rate_range=rate_range,
        )

        # Curriculum: gradually increase max augmentation rate
        # e.g. {'min_rate': 0.01, 'max_rate_start': 0.1, 'max_rate_end': 0.5, 'num_epochs': 30}
        self.curriculum = curriculum

    def forward(self, history_data, future_data=None, batch_seen=None,
                epoch=None, train=True, **kwargs):
        """
        Args:
            history_data: [B, T, N, 8] preprocessed input

        Returns:
            dict with prediction (student), target (teacher stop-grad), obs_weight
        """
        # Curriculum: adjust augmentation rate range based on epoch
        if self.curriculum and epoch is not None:
            progress = min(epoch / self.curriculum['num_epochs'], 1.0)
            max_rate = (self.curriculum['max_rate_start'] +
                        progress * (self.curriculum['max_rate_end'] - self.curriculum['max_rate_start']))
            self.augmentation.rate_range = (self.curriculum['min_rate'], max_rate)

        # Create augmented view (student)
        view2, obs_weight = self.augmentation(history_data)

        # Encode both views
        z1 = self.encoder(history_data)  # teacher [B, T, N, D]
        z2 = self.encoder(view2)         # student [B, T, N, D]

        # Pool over time
        z1_pool = z1.mean(dim=1)  # [B, N, D]
        z2_pool = z2.mean(dim=1)

        # Predictor on student only
        p2 = self.predictor(z2_pool)  # [B, N, D]

        return {
            'prediction': p2,
            'target': z1_pool.detach(),  # stop-gradient on teacher
            'obs_weight': obs_weight,
        }


def micl_loss(prediction, target, obs_weight=None):
    """Observation-weighted cosine similarity loss.

    Args:
        prediction: [B, N, D] student predictor output
        target: [B, N, D] teacher representation (stop-grad)
        obs_weight: [B, N] per-node observation weight
    """
    cos_sim = F.cosine_similarity(prediction, target, dim=-1)  # [B, N]
    loss = 1 - cos_sim  # [0, 2]
    if obs_weight is not None:
        loss = loss * obs_weight
    return loss.mean()
