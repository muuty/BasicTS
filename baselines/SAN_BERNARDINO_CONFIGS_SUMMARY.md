# SAN_BERNARDINO Dataset Configuration Summary

This document summarizes all config files created for the 4 pre-training models on the SAN_BERNARDINO dataset.

## Dataset Settings (Common across all models)

- **Dataset Name**: `xtraffic/SAN_BERNARDINO`
- **Number of Nodes**: 893
- **Input Length**: 12 (1 hour at 5-min intervals)
- **Output Length**: 12 (1 hour prediction)
- **Data Range**: (0, 26280) - 3 months of data
- **Train/Val/Test Ratio**: From regular_settings (typically [0.6, 0.2, 0.2])
- **Batch Size**: 16 for training
- **Number of Epochs**: 30
- **Learning Rate Scheduler**: MultiStepLR with milestones [20, 25], gamma 0.1

## Model 1: STSSL (Self-Supervised Spatio-Temporal Learning)

### Config File
- **Location**: `baselines/STSSL/SAN_BERNARDINO.py`
- **Type**: End-to-end training (no separate pre-training/fine-tuning)

### Key Parameters
```python
MODEL_PARAM = {
    "num_nodes": 893,
    "in_steps": 12,
    "out_steps": 12,
    "input_dim": 1,  # Only speed feature
    "output_dim": 1,
    "d_model": 64,
    "dropout": 0.1,
    "nmb_prototype": 10,
    "shm_temp": 0.5,
    "aug_percent": 0.2,
    "loss_weights": [1.0, 0.1, 0.1],  # pred, temporal, spatial
    "mode": "end2end",
    "batch_size": 16,
    "Kt": 3,
    "Ks": 3,
}
```

### Features Used
- **Forward**: [0] - speed only
- **Target**: [0] - speed only

### Usage
```bash
python -c "from basicts import launch_training; launch_training('baselines/STSSL/SAN_BERNARDINO.py', gpus='0')"
```

---

## Model 2: GPTST (Graph Pre-Training for Spatio-Temporal)

### Config Files
1. **Pre-training**: `baselines/GPTST/SAN_BERNARDINO_pretrain.py`
2. **Fine-tuning**: `baselines/GPTST/SAN_BERNARDINO_finetune.py`

### Pre-training Parameters
```python
MODEL_PARAM = {
    "num_nodes": 893,
    "in_steps": 12,
    "out_steps": 12,
    "input_dim": 3,  # flow + tod + dow
    "output_dim": 1,
    "input_base_dim": 1,
    "hidden_dim": 64,
    "embed_dim": 16,
    "embed_dim_spa": 8,
    "HS": 4,  # Spatial hyperedge heads
    "HT": 4,  # Temporal hyperedge heads
    "HT_Tem": 4,
    "num_route": 3,
    "mode": "pretrain",
    "mask_ratio": 0.3,
    "ada_mask_ratio": 1.0,
    "ada_type": "all",
    "change_epoch": 10,
    "epochs": 30,
}
```

### Fine-tuning Parameters
- Same architecture as pre-training
- `mode`: "finetune"
- `pretrained_path`: Points to best pre-trained checkpoint
- `freeze_encoder`: False (allows encoder fine-tuning)

### Features Used
- **Forward**: [0, 3, 4] - flow, tod, dow
- **Target**: [0] - flow only

### Usage
```bash
# Step 1: Pre-training
python -c "from basicts import launch_training; launch_training('baselines/GPTST/SAN_BERNARDINO_pretrain.py', gpus='0')"

# Step 2: Fine-tuning
python -c "from basicts import launch_training; launch_training('baselines/GPTST/SAN_BERNARDINO_finetune.py', gpus='0')"
```

---

## Model 3: STEP (Spatial-Temporal Efficient Pre-training)

### Config Files
1. **Pre-training (TSFormer)**: `baselines/STEP/TSFormer_SAN_BERNARDINO.py`
2. **Fine-tuning (STEP)**: `baselines/STEP/STEP_SAN_BERNARDINO.py`

### TSFormer Pre-training Parameters
```python
PATCH_SIZE = 12  # 1 hour patches
NUM_PATCHES = 168  # 7 days
INPUT_LEN = 2016  # 7 days for pre-training

MODEL_PARAM = {
    "patch_size": 12,
    "in_channel": 1,
    "embed_dim": 96,
    "num_heads": 4,
    "mlp_ratio": 4,
    "dropout": 0.1,
    "num_token": 168,
    "mask_ratio": 0.75,
    "encoder_depth": 4,
    "decoder_depth": 1,
    "mode": "pre-train",
}
```

### STEP Fine-tuning Parameters
```python
MODEL_PARAM = {
    "num_nodes": 893,
    # TSFormer args (must match pre-training)
    "tsformer_patch_size": 12,
    "tsformer_in_channel": 1,
    "tsformer_embed_dim": 96,
    "tsformer_num_heads": 4,
    "tsformer_mlp_ratio": 4,
    "tsformer_dropout": 0.1,
    "tsformer_num_token": 168,
    "tsformer_mask_ratio": 0.75,
    "tsformer_encoder_depth": 4,
    "tsformer_decoder_depth": 1,
    # Backend (GraphWaveNet)
    "backend_in_dim": 2,
    "backend_out_dim": 12,
    "backend_residual_channels": 32,
    "backend_dilation_channels": 32,
    "backend_skip_channels": 256,
    "backend_end_channels": 512,
    "backend_kernel_size": 2,
    "backend_blocks": 4,
    "backend_layers": 2,
    # Dynamic Graph Learning
    "dgl_k": 10,
    "dgl_node_feature_dim": 100,
    "dgl_temperature": 0.5,
    # Pre-trained model
    "pre_trained_tsformer_path": "checkpoints/TSFormer_pretrain/...",
}
```

### Features Used
- **Pre-training**: [0] - speed only
- **Fine-tuning**: [0, 3, 4] - speed, tod, dow

### Usage
```bash
# Step 1: TSFormer Pre-training
python -c "from basicts import launch_training; launch_training('baselines/STEP/TSFormer_SAN_BERNARDINO.py', gpus='0')"

# Step 2: STEP Fine-tuning
python -c "from basicts import launch_training; launch_training('baselines/STEP/STEP_SAN_BERNARDINO.py', gpus='0')"
```

---

## Model 4: STMAE (Spatio-Temporal Masked Autoencoder)

### Config Files
1. **Pre-training**: `baselines/STMAE/SAN_BERNARDINO_pretrain.py`
2. **Fine-tuning**: `baselines/STMAE/SAN_BERNARDINO_finetune.py`

### Pre-training Parameters (AGCRN Backbone)
```python
AGCRN_PARAMS = {
    "num_nodes": 893,
    "input_dim": 1,
    "rnn_units": 64,
    "num_layers": 2,
    "cheb_k": 2,
    "embed_dim": 10,
}

MODEL_PARAM = {
    "num_nodes": 893,
    "input_dim": 1,
    "hidden_dim": 64,
    "input_len": 12,
    "output_len": 12,
    "backbone_class": AGCRN,
    "backbone_params": AGCRN_PARAMS,
    # Masking configuration
    "mask_f_ratio": 0.5,  # Feature masking
    "mask_s_ratio": 0.3,  # Structure masking
    "patch_length": 1,
    "walks_per_node": 10,
    "walk_length": 20,
    # Loss weights
    "sl_weight": 1.0,  # Structure loss
    "fl_weight": 1.0,  # Feature loss
    "embed_dim": 10,
}
```

### Fine-tuning Parameters (STAEformer Backbone)
```python
MODEL_PARAM = {
    "num_nodes": 893,
    "in_steps": 12,
    "out_steps": 12,
    "steps_per_day": 288,
    "input_dim": 3,  # flow + tod + dow
    "output_dim": 1,
    "input_embedding_dim": 24,
    "tod_embedding_dim": 24,
    "dow_embedding_dim": 24,
    "spatial_embedding_dim": 0,
    "adaptive_embedding_dim": 24,
    "feed_forward_dim": 256,
    "num_heads": 4,
    "num_layers": 1,
    "dropout": 0.1,
    "use_mixed_proj": True,
    "pretrained_path": "checkpoints/STMAE_pretrain/...",
    "freeze_encoder": False,
}
```

### Features Used
- **Pre-training**: [0] - flow only
- **Fine-tuning**: [0, 3, 4] - flow, tod, dow

### Usage
```bash
# Step 1: Pre-training
python -c "from basicts import launch_training; launch_training('baselines/STMAE/SAN_BERNARDINO_pretrain.py', gpus='0')"

# Step 2: Fine-tuning
python -c "from basicts import launch_training; launch_training('baselines/STMAE/SAN_BERNARDINO_finetune.py', gpus='0')"
```

---

## Checkpoint Locations

All models save checkpoints to:
```
checkpoints/{MODEL_NAME}/xtraffic_SAN_BERNARDINO_30_{INPUT_LEN}_{OUTPUT_LEN}/
```

Specific patterns:
- **STSSL**: `checkpoints/STSSL/xtraffic_SAN_BERNARDINO_30_12_12/`
- **GPTST Pretrain**: `checkpoints/GPTSTForForecasting/xtraffic_SAN_BERNARDINO_30_12_12_pretrain/`
- **GPTST Finetune**: `checkpoints/GPTSTForForecasting/xtraffic_SAN_BERNARDINO_30_12_12_finetune/`
- **TSFormer**: `checkpoints/TSFormer_pretrain/xtraffic_SAN_BERNARDINO_30/`
- **STEP**: `checkpoints/STEP/xtraffic_SAN_BERNARDINO_30_12_12/`
- **STMAE Pretrain**: `checkpoints/STMAE_pretrain/xtraffic_SAN_BERNARDINO_30_12_12/`
- **STMAE Finetune**: `checkpoints/STMAE_finetune/xtraffic_SAN_BERNARDINO_30_12_12/`

---

## Testing and Evaluation

After training, test results are saved to `test_metrics.json` in the checkpoint directory.

To evaluate a trained model:
```bash
python -c "from basicts import launch_evaluation; launch_evaluation('path/to/config.py', 'path/to/checkpoint.pt', gpus='0')"
```

Check results:
```bash
find checkpoints -name "test_metrics.json" -exec cat {} \;
```

---

## Notes

1. All configs use 3 months of data (26,280 time steps) for fair comparison
2. All models trained for 30 epochs with batch size 16
3. Learning rate scheduler: MultiStepLR with milestones [20, 25], gamma 0.1
4. Pre-training and fine-tuning configs exist for GPTST, STEP, and STMAE
5. STSSL uses end-to-end training (no separate pre-training stage)
6. All configs include incident metadata path for robust evaluation
7. Evaluation horizons: [3, 6, 12] steps ahead
