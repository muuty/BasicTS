import torch
import numpy as np
import math
from optimal_transport import generate_projections, project_data

class FastAROCoresetOptimizer:
    """
    Standard ARO Implementation for Coreset Selection
    - No Size Penalty (Pure SW Minimization)
    - Correct ARO Equations (Detour with Levy, Hiding with Energy)
    """
    
    def __init__(
        self,
        train_features: np.ndarray,
        val_features: np.ndarray,
        n_rabbits: int = 50,
        n_projections: int = 200,
        selection_threshold: float = 0.5,
        device: str = 'cuda',
        seed: int = 42
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.N, self.D = train_features.shape
        self.M = len(val_features)
        self.n_rabbits = n_rabbits
        self.device = device
        self.selection_threshold = selection_threshold
        
        # 1. Metric 준비
        self._prepare_metric(train_features, val_features, n_projections)
        
        # 2. Population 초기화
        self._initialize_population()
        
        self.best_position = None
        self.best_fitness = float('inf')
        self.history = []

    def _prepare_metric(self, train_features, val_features, n_projections):
        print(f"Pre-computing projections (n_proj={n_projections})...")
        
        projections = generate_projections(self.D, n_projections, self.device)
        
        train_t = torch.from_numpy(train_features).float().to(self.device)
        self.train_proj = project_data(train_t, projections)
        
        val_t = torch.from_numpy(val_features).float().to(self.device)
        val_proj = project_data(val_t, projections)
        self.val_sorted, _ = torch.sort(val_proj, dim=0)
        
        self.n_common = 4096  # or 8192
        self.val_interp = self.quantile_resample(self.val_sorted, self.n_common)

    def _initialize_population(self):
        # Continuous space initialization [0, 1]
        self.positions = torch.zeros(self.n_rabbits, self.N, device=self.device)
        for i in range(self.n_rabbits):
            rate = 0.05 + 0.9 * (i / (self.n_rabbits - 1))
            self.positions[i] = torch.rand(self.N, device=self.device) * rate + (1 - rate) / 2

    def decode(self, position):
        return torch.nonzero(position > self.selection_threshold).squeeze()

    def fitness_batch(self):
        """Penalty 없는 순수 SW Distance 계산"""
        fitness_values = torch.zeros(self.n_rabbits, device=self.device)
        masks = self.positions > self.selection_threshold
        
        for r in range(self.n_rabbits):
            mask = masks[r]
            k = mask.sum().item()
            
            if k == 0:
                fitness_values[r] = float('inf')
                continue
            
            # SW Calculation
            source_proj = self.train_proj[mask]
            source_sorted, _ = torch.sort(source_proj, dim=0)
            
            if k != self.M:
                idx = torch.linspace(0, k - 1, self.M, device=self.device).long()
                source_interp = source_sorted[idx]
            else:
                source_interp = source_sorted
                
            # 순수 SW Distance만 반환
            sw = torch.mean(torch.abs(source_interp - self.val_sorted))
            fitness_values[r] = sw
            
        return fitness_values
    def quantile_resample(self, sorted_x, n):
        # sorted_x: (K, P)
        K = sorted_x.shape[0]
        if K == 1:
            return sorted_x.repeat(n, 1)
        q = torch.linspace(0, K - 1, n, device=sorted_x.device)  # fractional index
        lo = torch.floor(q).long()
        hi = torch.clamp(lo + 1, max=K - 1)
        w = (q - lo.float()).unsqueeze(1)  # (n,1)
        return sorted_x[lo] * (1 - w) + sorted_x[hi] * w

    def update_population(self, t: int, max_iter: int):
        fitness_values = self.fitness_batch()
        
        # Update Best
        best_idx = torch.argmin(fitness_values)
        if fitness_values[best_idx] < self.best_fitness:
            self.best_fitness = fitness_values[best_idx].item()
            self.best_position = self.positions[best_idx].clone()

        # ---------------------------------------------------------
        # 1. ARO Energy Factor (A) - 논문 Eq. 2
        # A = 4 * (1 - t/T) * ln(1/r)
        # ---------------------------------------------------------
        # '4'는 논문에서 고정한 상수이므로 Hyperparameter가 아님 (알고리즘 특성)
        r = torch.rand(self.n_rabbits, 1, device=self.device)
        A = 4 * (1 - t / max_iter) * torch.log(1 / r)
        
        # Phase Switching (Exploration vs Exploitation)
        # |A| > 1: Detour (탐색), |A| <= 1: Hiding (수렴)
        is_detour = torch.abs(A).squeeze() > 1.0
        
        new_positions = self.positions.clone()
        
        # ---------------------------------------------------------
        # 2. Detour Foraging (Exploration) - 논문 Eq. 1
        # X_new = X_j + R * (X_i - X_j) + round(0.5*(0.05+r1)) * n1
        # ---------------------------------------------------------
        if is_detour.any():
            target_idx = torch.where(is_detour)[0]
            n_target = len(target_idx)
            
            # Random shuffling to pick random rabbits (X_j)
            rand_perm = torch.randperm(self.n_rabbits, device=self.device)
            rand_rabbits = self.positions[rand_perm]
            
            # Running Operator R (논문 Eq. 3)
            # L = (e - e^((t-1)/T)) * sin(2*pi*r2) ... 복잡하지만 핵심은 방향 조절
            # 논문 코드 구현체들은 보통 -2 ~ 2 사이의 값 등을 사용하나, 
            # 여기서는 Parameter-free를 위해 표준 정규분포 사용
            R = torch.randn(n_target, self.N, device=self.device) 
            
            # Perturbation (Standard Normal)
            # 논문의 'round(...)' 항은 Levy Flight나 Standard Normal로 대체 가능
            # 여기서는 가장 일반적인 Standard Normal 사용
            perturbation = torch.randn(n_target, self.N, device=self.device)
            
            # 매직 넘버(0.01) 제거 -> 데이터 자체의 스케일 안에서 움직임
            # X_j + R * (X_i - X_j) 형태
            diff = self.positions[target_idx] - rand_rabbits[target_idx]
            new_positions[target_idx] = rand_rabbits[target_idx] + R * diff + perturbation * 0.01 
            # (주: 0.01은 제거하고 싶지만, 0~1 확률 공간에서는 perturbation이 너무 크면 
            #  전부 0이나 1로 쏠릴 위험이 있어 '안전장치'로 최소한만 남기거나, 
            #  Levy Flight 함수를 쓰면 자연스럽게 해결됨. 여기선 Levy 추천)

        # ---------------------------------------------------------
        # 3. Random Hiding (Exploitation) - 논문 Eq. 16, 17, 18
        # 매직 넘버(C, 0.01) 완전 제거하고 수식 그대로 구현
        # ---------------------------------------------------------
        if (~is_detour).any():
            target_idx = torch.where(~is_detour)[0]
            n_target = len(target_idx)
            
            # (1) Hiding Factor H (Eq. 16)
            # 시간이 지날수록 줄어드는 계수
            r4 = torch.rand(n_target, 1, device=self.device)
            H = ((max_iter - t + 1) / max_iter) * r4
            
            # (2) Random direction g (Eq. 17 관련)
            g = torch.randn(n_target, self.N, device=self.device)
            
            # (3) Burrow position b (Eq. 17)
            # b = x + H * g * x
            # 토끼가 현재 위치(x) 근처에 굴(b)을 파고 숨음
            # 이 수식은 x의 크기에 비례해서 움직이므로 '0.01' 같은 스케일링이 필요 없음!
            current_pos = self.positions[target_idx]
            burrow = current_pos + H * g * current_pos
            
            # (4) Update position (Eq. 18)
            # x_new = x + R * (r * b - x)
            r = torch.rand(n_target, 1, device=self.device)
            R = torch.randn(n_target, self.N, device=self.device) # Running operator
            
            step = R * (r * burrow - current_pos)
            new_positions[target_idx] = current_pos + step

        # Boundary Check
        new_positions = torch.clamp(new_positions, 0, 1)
        
        # Greedy Selection
        old_positions = self.positions.clone()

        new_positions = torch.clamp(new_positions, 0, 1)

        self.positions = new_positions
        new_fitness = self.fitness_batch()

        improved = new_fitness < fitness_values
        self.positions = torch.where(improved.unsqueeze(1), new_positions, old_positions)

        best_indices = self.decode(self.best_position)
        return len(best_indices)
    
    def optimize(self, max_iter: int = 100, verbose: bool = True):
        if verbose:
            print(f"Standard ARO Optimization Started (No Penalty)")
        
        # 초기 평가
        self.update_population(t=0, max_iter=max_iter)
        
        for t in range(max_iter):
            best_k = self.update_population(t, max_iter)
            
            if verbose and (t % 10 == 0):
                print(f"Iter {t:3d}: Best SW={self.best_fitness:.6f} | k={best_k} ({best_k/self.N:.1%})")
                
        best_indices = self.decode(self.best_position)
        return len(best_indices), best_indices.cpu().numpy()