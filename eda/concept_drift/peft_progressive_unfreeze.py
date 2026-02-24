"""Progressive Unfreezing: Convergence-driven capacity scaling.

Hypothesis: Start with PB+head (efficient), auto-unlock embedding when loss plateaus.
- Phase 1: PB + head (21K params)
- Phase 2: + adaptive_embedding (unlocked when loss plateaus)

Quick validation: 2 pairs × 6 time budgets.
"""
import sys, os, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.append("/data/pretrainingbasicts")
from baselines.STAEformer.arch import STAEformer

MODEL_PARAM = {
    "num_nodes": 893, "in_steps": 12, "out_steps": 12, "steps_per_day": 288,
    "input_dim": 3, "output_dim": 1, "input_embedding_dim": 24,
    "tod_embedding_dim": 24, "dow_embedding_dim": 24, "spatial_embedding_dim": 0,
    "adaptive_embedding_dim": 24, "feed_forward_dim": 256, "num_heads": 4,
    "num_layers": 1, "dropout": 0.1, "use_mixed_proj": True,
}

INSTNORM_CHECKPOINTS = {
    2022: "checkpoints/ConceptDrift_InstanceNorm/SAN_BERNARDINO_2022_Q1_30_12_12/ee3765011653c94baa0ab81045fb239e/STAEformer_best_val_MAE.pt",
    2023: "checkpoints/ConceptDrift_InstanceNorm/SAN_BERNARDINO_2023_Q1_30_12_12/d2940bc326d7f29eedeedb573e026973/STAEformer_best_val_MAE.pt",
    2024: "checkpoints/ConceptDrift_InstanceNorm/SAN_BERNARDINO_2024_Q1_30_12_12/16c8094c3cf1ef5013e0384708d2e954/STAEformer_best_val_MAE.pt",
}

DATASETS = {
    2022: "datasets/SAN_BERNARDINO_2022_Q1",
    2023: "datasets/SAN_BERNARDINO_2023_Q1",
    2024: "datasets/SAN_BERNARDINO_2024_Q1",
}

DEVICE = "cuda:1"
BATCH_SIZE = 16
K = 8
FINETUNE_HOURS = [3, 6, 12, 24, 72, 168]
# Quick validation: 2 pairs only
PAIRS = [(2022, 2023), (2023, 2022)]


class STAEformerWithPB(nn.Module):
    def __init__(self, backbone, num_nodes=893, adp_dim=24, K=8):
        super().__init__()
        self.backbone = backbone
        self.pattern_bank = nn.Parameter(torch.randn(K, adp_dim) * 0.01)
        self.node_weights = nn.Parameter(torch.zeros(num_nodes, K))

    def forward(self, history_data, future_data, batch_seen, epoch, train, **kwargs):
        x = history_data
        batch_size = x.shape[0]
        enc = self.backbone.encoder
        if enc.tod_embedding_dim > 0:
            tod = x[..., enc.tod_index] * enc.steps_per_day
        if enc.dow_embedding_dim > 0:
            dow = x[..., enc.dow_index] * 7
        x = x[..., :enc.input_dim]
        x = enc.input_proj(x)
        features = [x]
        if enc.tod_embedding_dim > 0:
            features.append(enc.tod_embedding(tod.long()))
        if enc.dow_embedding_dim > 0:
            features.append(enc.dow_embedding(dow.long()))
        if enc.adaptive_embedding_dim > 0:
            adp_emb = enc.adaptive_embedding.expand(batch_size, *enc.adaptive_embedding.shape)
            delta = F.softmax(self.node_weights, dim=-1) @ self.pattern_bank
            adp_emb = adp_emb + delta.unsqueeze(0).unsqueeze(0)
            features.append(adp_emb)
        x = torch.cat(features, dim=-1)
        for attn in enc.attn_layers_t:
            x = attn(x, dim=1)
        x = self.backbone.spatial(x, None)
        out = self.backbone.decoder(x)
        return {"prediction": out}


def load_backbone(ckpt_path):
    backbone = STAEformer(**MODEL_PARAM)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    backbone.load_state_dict(ckpt["model_state_dict"])
    return backbone


def load_data_and_scaler(dataset_dir):
    desc = json.load(open(os.path.join(dataset_dir, "desc.json")))
    shape = desc["shape"]
    data = np.memmap(os.path.join(dataset_dir, "data.dat"), dtype="float32", mode="r").reshape(shape)
    n_total = shape[0]
    n_train = int(n_total * 0.6)
    mean = float(np.mean(data[:n_train, :, 0]))
    std = float(np.std(data[:n_train, :, 0]))
    return data, mean, std, n_total


def create_samples(data, input_len=12, output_len=12):
    samples_x, samples_y = [], []
    for i in range(len(data) - input_len - output_len + 1):
        samples_x.append(data[i:i+input_len])
        samples_y.append(data[i+input_len:i+input_len+output_len, :, 0:1])
    return np.array(samples_x), np.array(samples_y)


def normalize_input(x, mean, std):
    x_norm = x.copy()
    x_norm[:, :, :, 0] = (x[:, :, :, 0] - mean) / std
    return x_norm


def apply_instance_norm(bx):
    flow = bx[:, :, :, 0]
    inst_mean = flow.mean(dim=1, keepdim=True)
    inst_std = flow.std(dim=1, keepdim=True) + 1e-5
    bx_normed = bx.clone()
    bx_normed[:, :, :, 0] = (flow - inst_mean) / inst_std
    return bx_normed, inst_mean, inst_std


def denorm_instance_norm(pred, inst_mean, inst_std):
    return pred * inst_std.unsqueeze(-1) + inst_mean.unsqueeze(-1)


def evaluate(model, test_x, test_y, mean, std, batch_size=64):
    model.eval()
    test_x_norm = normalize_input(test_x, mean, std)
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(test_x_norm), batch_size):
            bx = torch.FloatTensor(test_x_norm[i:i+batch_size]).to(DEVICE)
            bx_normed, inst_mean, inst_std = apply_instance_norm(bx)
            out = model(bx_normed, None, 0, 0, False)["prediction"]
            out = denorm_instance_norm(out, inst_mean, inst_std)
            pred = out.cpu().numpy() * std + mean
            all_preds.append(pred)
    preds = np.concatenate(all_preds, axis=0)
    return float(np.mean(np.abs(preds - test_y)))


def train_one_epoch(model, optimizer, train_x_norm, train_y, mean, std):
    """Train one epoch, return average loss."""
    model.train()
    indices = np.random.permutation(len(train_x_norm))
    total_loss = 0
    n_batches = 0
    for i in range(0, len(indices), BATCH_SIZE):
        batch_idx = indices[i:i+BATCH_SIZE]
        bx = torch.FloatTensor(train_x_norm[batch_idx]).to(DEVICE)
        by = torch.FloatTensor(train_y[batch_idx]).to(DEVICE)
        bx_normed, inst_mean, inst_std = apply_instance_norm(bx)
        pred = model(bx_normed, None, 0, 0, True)["prediction"]
        pred = denorm_instance_norm(pred, inst_mean, inst_std)
        loss = nn.L1Loss()(pred * std + mean, by)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


def set_trainable(model, trainable_names):
    for name, param in model.named_parameters():
        param.requires_grad = any(t in name for t in trainable_names)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return n


def train_fixed(model, trainable_names, train_x, train_y, mean, std, epochs, lr):
    """Fixed training (baseline)."""
    model = model.to(DEVICE)
    n_trainable = set_trainable(model, trainable_names)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    train_x_norm = normalize_input(train_x, mean, std)
    for ep in range(epochs):
        train_one_epoch(model, optimizer, train_x_norm, train_y, mean, std)
    return model, n_trainable


def train_progressive(model, train_x, train_y, mean, std,
                      max_epochs=15, patience=2, min_imp_rate=0.005,
                      lr_phase1=0.001, lr_emb_min=0.0003, lr_emb_max=0.001):
    """Progressive unfreezing: PB+head → auto-unlock embedding on plateau.

    Uses relative improvement rate instead of absolute delta:
      improvement_rate = (prev_loss - cur_loss) / prev_loss
      if improvement_rate < min_imp_rate for `patience` consecutive epochs → unlock

    Adaptive Phase2 LR based on convergence speed:
      convergence_speed = switch_epoch / max_epochs  (0~1)
      Fast convergence (low ratio) → more data → higher emb LR OK
      Slow convergence (high ratio) → less data → lower emb LR needed
      lr_emb = lr_emb_min + (lr_emb_max - lr_emb_min) * (1 - convergence_speed)
    """
    model = model.to(DEVICE)
    train_x_norm = normalize_input(train_x, mean, std)

    # Phase 1: PB + head
    phase = 1
    set_trainable(model, ["pattern_bank", "node_weights", "decoder"])
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_phase1)

    prev_loss = None
    no_improve = 0
    phase_switch_epoch = None
    actual_lr_emb = None
    losses = []

    for ep in range(max_epochs):
        loss = train_one_epoch(model, optimizer, train_x_norm, train_y, mean, std)
        losses.append((ep, phase, loss))

        if phase == 1 and prev_loss is not None:
            imp_rate = (prev_loss - loss) / prev_loss
            if imp_rate < min_imp_rate:
                no_improve += 1
            else:
                no_improve = 0

            if no_improve >= patience:
                # Plateau detected → unlock embedding
                phase = 2
                phase_switch_epoch = ep + 1
                # Adaptive LR: faster convergence → higher emb LR
                convergence_speed = (ep + 1) / max_epochs
                actual_lr_emb = lr_emb_min + (lr_emb_max - lr_emb_min) * (1 - convergence_speed)
                set_trainable(model, ["pattern_bank", "node_weights", "decoder", "adaptive_embedding"])
                param_groups = [
                    {"params": [p for n, p in model.named_parameters()
                                if p.requires_grad and "adaptive_embedding" not in n],
                     "lr": lr_phase1},
                    {"params": [p for n, p in model.named_parameters()
                                if p.requires_grad and "adaptive_embedding" in n],
                     "lr": actual_lr_emb},
                ]
                optimizer = torch.optim.Adam(param_groups)
                no_improve = 0

        prev_loss = loss

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model, n_trainable, phase_switch_epoch, actual_lr_emb, losses


def main():
    print("=" * 70)
    print("PROGRESSIVE UNFREEZING: Quick Validation")
    print("=" * 70)

    data_cache = {}
    for year in [2022, 2023]:
        data, mean, std, n_total = load_data_and_scaler(DATASETS[year])
        n_train = int(n_total * 0.6)
        n_val = int(n_total * 0.2)
        test_data = data[n_train + n_val:]
        test_x, test_y = create_samples(test_data)
        data_cache[year] = {"full_data": data, "test_x": test_x, "test_y": test_y,
                            "mean": mean, "std": std}
        print(f"  {year}: test={len(test_x)}")

    results = {}

    for source_year, target_year in PAIRS:
        ckpt_path = INSTNORM_CHECKPOINTS[source_year]
        src = data_cache[source_year]
        tgt = data_cache[target_year]
        pair = f"{source_year}_{target_year}"
        print(f"\n{'='*50} {source_year}->{target_year} {'='*10}")

        for hours in FINETUNE_HOURS:
            n_steps = hours * 12
            ft_data = tgt["full_data"][:n_steps]
            ft_x, ft_y = create_samples(ft_data)
            h_label = f"{hours}h" if hours < 24 else f"{hours//24}d"
            print(f"\n  --- {h_label} ({len(ft_x)} samples) ---")

            # 1) PB+head fixed (baseline)
            backbone = load_backbone(ckpt_path)
            model = STAEformerWithPB(backbone)
            model, np1 = train_fixed(model, ["pattern_bank", "node_weights", "decoder"],
                                     ft_x, ft_y, src["mean"], src["std"], epochs=10, lr=0.001)
            mae = evaluate(model, tgt["test_x"], tgt["test_y"], src["mean"], src["std"])
            results[f"pb+head_{h_label}_{pair}"] = mae
            print(f"    PB+head fixed    {mae:.2f}  ({np1:,}p)")
            del model, backbone; torch.cuda.empty_cache()

            # 2) emb_only fixed (baseline)
            backbone = load_backbone(ckpt_path)
            model = STAEformerWithPB(backbone)
            model, np2 = train_fixed(model, ["adaptive_embedding"],
                                     ft_x, ft_y, src["mean"], src["std"], epochs=10, lr=0.001)
            mae = evaluate(model, tgt["test_x"], tgt["test_y"], src["mean"], src["std"])
            results[f"emb_only_{h_label}_{pair}"] = mae
            print(f"    emb_only fixed   {mae:.2f}  ({np2:,}p)")
            del model, backbone; torch.cuda.empty_cache()

            # 3) Progressive unfreezing
            backbone = load_backbone(ckpt_path)
            model = STAEformerWithPB(backbone)
            model, np3, switch_ep, lr_emb, losses = train_progressive(
                model, ft_x, ft_y, src["mean"], src["std"],
                max_epochs=15, patience=2, min_imp_rate=0.005)
            mae = evaluate(model, tgt["test_x"], tgt["test_y"], src["mean"], src["std"])
            results[f"progressive_{h_label}_{pair}"] = mae
            results[f"prog_switch_{h_label}_{pair}"] = switch_ep
            results[f"prog_lr_emb_{h_label}_{pair}"] = lr_emb
            phase_str = f"switch@ep{switch_ep}, lr_emb={lr_emb:.4f}" if switch_ep else "stayed Phase1"
            print(f"    Progressive      {mae:.2f}  ({np3:,}p, {phase_str})")
            # Show loss curve with improvement rate
            for i, (ep, ph, l) in enumerate(losses):
                marker = " <<<UNLOCK" if switch_ep and ep + 1 == switch_ep else ""
                imp_rate = (losses[i-1][2] - l) / losses[i-1][2] * 100 if i > 0 else 0
                if ep % 3 == 0 or (switch_ep and abs(ep + 1 - switch_ep) <= 1):
                    print(f"      ep{ep:2d} phase{ph} loss={l:.4f} imp={imp_rate:+.2f}%{marker}")
            del model, backbone; torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY (2-pair avg)")
    print("=" * 70)
    print(f"{'Time':<6} {'PB+head':>10} {'emb_only':>10} {'Progress.':>10} {'Switch@':>10}")
    print("-" * 50)
    for hours in FINETUNE_HOURS:
        h = f"{hours}h" if hours < 24 else f"{hours//24}d"
        pb_h = np.mean([results[f"pb+head_{h}_{s}_{t}"] for s, t in PAIRS])
        emb = np.mean([results[f"emb_only_{h}_{s}_{t}"] for s, t in PAIRS])
        prog = np.mean([results[f"progressive_{h}_{s}_{t}"] for s, t in PAIRS])
        switches = [results.get(f"prog_switch_{h}_{s}_{t}") for s, t in PAIRS]
        sw_str = "/".join([str(s) if s else "—" for s in switches])
        best = min(pb_h, emb, prog)
        markers = ["*" if abs(v - best) < 0.005 else " " for v in [pb_h, emb, prog]]
        print(f"{h:<6} {markers[0]}{pb_h:>9.2f} {markers[1]}{emb:>9.2f} {markers[2]}{prog:>9.2f}   ep:{sw_str}")

    output_path = "eda/concept_drift/peft_progressive_unfreeze_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
