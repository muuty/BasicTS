# Legacy encoders (for backward compatibility)
from .context_aware_encoder import ContextAwareEncoder, TemporalEncoder
from .spatiotemporal_encoder import SpatioTemporalEncoder, SpatialEncoder

# New unified encoder interface
from .base_encoder import BaseRepresentationEncoder, build_encoder, ENCODER_REGISTRY
from .encoders import (
    TransformerEncoder,
    DilatedConvEncoder,
    SpatioTemporalEncoder as SpatioTemporalEncoderV2,
    MaskedAutoEncoderWrapper,
)

# Pretrain models
from .contrastive_pretrain_model import ContrastivePretrainModel, Augmentation
from .temporal_neighbors_model import TemporalNeighborsModel, TemporalAugmentation
from .configurable_pretrain_model import ConfigurablePretrainModel
from .gat_pretrain_model import GATPretrainModel
from .temporal_aware_pretrain_model import TemporalAwarePretrainModel
from .unified_pretrain_model import UnifiedPretrainModel
from .disentangled_encoder import DisentangledEncoder
from .disentangled_pretrain_model import DisentangledPretrainModel
from .disentangled_temporal_model import DisentangledTemporalModel
from .predictive_contrastive_model import PredictiveContrastiveModel
from .cross_variable_pretrain_model import CrossVariablePretrainModel
from .denoising_encoder import DenoisingEncoder
from .denoising_pretrain_model import DenoisingPretrainModel
from .linear_attention_denoising_encoder import LinearAttentionDenoisingEncoder
from .contrastive_reliability_encoder import ContrastiveReliabilityEncoder
from .contrastive_reliability_pretrain_model import ContrastiveReliabilityPretrainModel
from .noise_invariant_pretrain_model import NoiseInvariantPretrainModel
from .simple_mlp_encoder import SimpleMLPEncoder, SimpleMLPPretrainModel
from .gated_mlp_encoder import GatedMLPEncoder, GatedMLPPretrainModel
from .multitask_staeformer import MultiTaskSTAEformer
from .dual_head_forecaster import DualHeadForecaster
from .residual_mlp import ResidualMLP
from .gated_residual_mlp import GatedResidualMLP
from .input_spillover_corrector import InputSpilloverCorrector
from .input_corrector_pretrain_model import InputCorrectorPretrainModel
from .input_spillover_corrector_v2 import InputSpilloverCorrectorV2
from .input_corrector_pretrain_model_v2 import InputCorrectorPretrainModelV2

# Augmentations
from .augmentations import (
    ConfigurableAugmentation,
    TemporalMasking,
    FeatureMasking,
    NodeMasking,
    GaussianNoise,
    Scaling,
)

__all__ = [
    # Unified encoder interface
    'BaseRepresentationEncoder',
    'build_encoder',
    'ENCODER_REGISTRY',
    'TransformerEncoder',
    'DilatedConvEncoder',
    'SpatioTemporalEncoderV2',
    'MaskedAutoEncoderWrapper',
    # Legacy encoders
    'ContextAwareEncoder',
    'TemporalEncoder',
    'SpatioTemporalEncoder',
    'SpatialEncoder',
    # Pretrain models
    'ContrastivePretrainModel',
    'Augmentation',
    'TemporalNeighborsModel',
    'TemporalAugmentation',
    'ConfigurablePretrainModel',
    'GATPretrainModel',
    'TemporalAwarePretrainModel',
    'UnifiedPretrainModel',
    'DisentangledEncoder',
    'DisentangledPretrainModel',
    'DisentangledTemporalModel',
    'PredictiveContrastiveModel',
    'CrossVariablePretrainModel',
    'DenoisingEncoder',
    'DenoisingPretrainModel',
    'ContrastiveReliabilityEncoder',
    'ContrastiveReliabilityPretrainModel',
    'NoiseInvariantPretrainModel',
    'SimpleMLPEncoder',
    'SimpleMLPPretrainModel',
    'DualHeadForecaster',
    'ResidualMLP',
    'GatedResidualMLP',
    'InputSpilloverCorrector',
    'InputCorrectorPretrainModel',
    'InputSpilloverCorrectorV2',
    'InputCorrectorPretrainModelV2',
    # Augmentations
    'ConfigurableAugmentation',
    'TemporalMasking',
    'FeatureMasking',
    'NodeMasking',
    'GaussianNoise',
    'Scaling',
]
