import torch

def quantile_resample(sorted_x, n):
    # sorted_x: (K, P)
    K = sorted_x.shape[0]
    if K <= 1:
        return sorted_x.repeat(n, 1)
    q = torch.linspace(0, K - 1, n, device=sorted_x.device)  # fractional index
    lo = torch.floor(q).long()
    hi = torch.clamp(lo + 1, max=K - 1)
    w = (q - lo.float()).unsqueeze(1)  # (n,1)
    return sorted_x[lo] * (1 - w) + sorted_x[hi] * w

def binary_concrete_st(logits, tau):
    # logits: (N,)
    u = torch.rand_like(logits)
    g = torch.log(u) - torch.log(1 - u)  # logistic noise
    z_soft = torch.sigmoid((logits + g) / tau)
    z_hard = (z_soft > 0.5).float()
    z = z_hard.detach() - z_soft.detach() + z_soft  # straight-through
    return z, z_soft, z_hard

class GumbelSigmoidCoresetOptimizer:
    def __init__(self, train_proj, val_sorted, n_common=4096, device="cuda", seed=42):
        torch.manual_seed(seed)
        self.device = device

        # train_proj: (N, P), val_sorted: (M, P) sorted along dim=0
        self.train_proj = train_proj.to(device)
        self.val_sorted = val_sorted.to(device)

        self.N, self.P = self.train_proj.shape
        self.M = self.val_sorted.shape[0]
        self.n_common = n_common

        # fixed target quantiles
        self.val_interp = quantile_resample(self.val_sorted, self.n_common)

        self.best_loss = float("inf")
        self.best_logits = None

    @torch.no_grad()
    def decode(self, logits, thresh=0.5):
        # deterministic decode (no sampling): sigmoid(logits) > thresh
        probs = torch.sigmoid(logits)
        idx = torch.nonzero(probs > thresh).squeeze(1)
        return idx

    def sw_loss_from_mask(self, mask, z_values):
        if not getattr(self, "_joint_ready", False):
            self.all_values = torch.cat([self.train_proj, self.val_sorted], dim=0)
            self.all_values_sorted, self.sort_idx = torch.sort(self.all_values, dim=0)
            self.delta_x = self.all_values_sorted[1:] - self.all_values_sorted[:-1]
            self._joint_ready = True

        # ---- weights ----
        z = z_values.clamp(0.0, 1.0)             # (N,)
        k = int(mask.sum().item())               # 로깅용

        z_sum = z.sum()
        if z_sum.item() < 1.0:
            return torch.tensor(1e3, device=self.device), k

        # (balanced) source mass를 1로 정규화
        w = z / (z_sum + 1e-8)                   # (N,)

        # target은 균일 질량 1/M
        # all_weights: (N+M, P) 를 만들기 위해 broadcasting 사용
        w_source = w[:, None].expand(self.N, self.P)                     # (N, P)
        w_target = (-1.0 / self.M) * torch.ones(self.M, self.P, device=self.device)  # (M, P)
        all_w = torch.cat([w_source, w_target], dim=0)                   # (N+M, P)

        # 값 정렬 순서에 맞춰 weight도 정렬
        all_w_sorted = torch.gather(all_w, 0, self.sort_idx)             # (N+M, P)

        # diff_cdf = F_source - F_target
        diff_cdf = torch.cumsum(all_w_sorted, dim=0)                     # (N+M, P)

        # 적분
        height = torch.abs(diff_cdf[:-1])                                # (N+M-1, P)
        sw = (height * self.delta_x).sum(dim=0).mean()

        return sw, k


    def optimize(
        self,
        steps=2000,
        lr=5e-2,
        tau_start=1.0,
        tau_end=0.2,
        lambda_sparsity=0.0,     # (1)에서는 0.0, (2)에서는 >0
        min_keep=2,
        log_every=50,
    ):
        logits = torch.zeros(self.N, device=self.device, requires_grad=True)
        opt = torch.optim.Adam([logits], lr=lr)

        for t in range(steps):
            # anneal temperature
            frac = t / max(1, steps - 1)
            tau = tau_start * (tau_end / tau_start) ** frac  # geometric anneal

            z, z_soft, z_hard = binary_concrete_st(logits, tau)
            mask = z_hard.bool()

            sw, k = self.sw_loss_from_mask(mask, z_soft)

            # sparsity term: encourage fewer selections (expected keep ratio)
            sparsity = z_soft.mean()
            loss = sw + lambda_sparsity * sparsity

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([logits], 1.0)
            opt.step()

            # track best (by full objective)
            if loss.item() < self.best_loss:
                self.best_loss = loss.item()
                self.best_logits = logits.detach().clone()

            if (t % log_every) == 0 or t == steps - 1:
                with torch.no_grad():
                    w = z_soft.detach().clone()
                    w_clamped = w.clamp(min=0.0)
                    s1 = w_clamped.sum()
                    s2 = (w_clamped ** 2).sum()

                    ess = (s1 * s1) / (s2 + 1e-12)   # Effective Sample Size

                    print(f"ESS={ess.item():.1f} | sum_w={s1.item():.3f} | mean_w={w_clamped.mean().item():.6f}")
                    keep_ratio = float(z_soft.mean().item())
                print(f"[{t:4d}/{steps}] loss={loss.item():.6f} sw={sw.item():.6f} "
                      f"k={k} ({k/self.N:.2%}) E[keep]={keep_ratio:.3f} tau={tau:.3f}")

        best_idx = self.decode(self.best_logits, thresh=0.5)
        return best_idx, self.best_loss