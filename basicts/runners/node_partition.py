#!/usr/bin/env python3
"""
Node partitioning utilities for s2former package
"""

import numpy as np
import networkx as nx
from sklearn.cluster import SpectralClustering
import torch


class NodePartitioner:
    """Node partitioning utilities for S2Former"""
    
    def __init__(
        self,
        num_nodes,
        num_groups,
        grouping_method='metis',
        adj_matrix=None,
        # ✅ 추가
        imbalance: float | None = None,          # 0~1 (0=균등, 1=극단 불균형)
        imbalance_alpha: float | None = None,    # Dirichlet alpha 직접 지정 (우선권)
        min_nodes_per_group: int = 1,            # 각 클라 최소 노드
        random_seed: int | None = None,          # 재현 seed (None이면 결정적 seed)
    ):
        self.num_nodes = num_nodes
        self.num_groups = num_groups
        self.grouping_method = grouping_method
        self.adj_matrix = adj_matrix

        # ✅ 추가 저장
        self.imbalance = imbalance
        self.imbalance_alpha = imbalance_alpha
        self.min_nodes_per_group = int(min_nodes_per_group)
        self.random_seed = random_seed
        
        if self.adj_matrix is not None:
            self._analyze_adjacency_matrix()
    
    def get_subgraph_adjacency(self, group_nodes):
        """그룹 내 노드들의 subgraph adjacency matrix 추출
        
        Args:
            group_nodes: List of node indices in the group
            
        Returns:
            torch.Tensor: [num_nodes_in_group, num_nodes_in_group] subgraph adjacency matrix
        """
        if self.adj_matrix is None:
            raise ValueError("adjacency matrix가 없습니다.")
        
        # Convert to tensor if needed
        if hasattr(self.adj_matrix, 'cpu'):
            adj = self.adj_matrix
        else:
            adj = torch.tensor(self.adj_matrix, dtype=torch.float32)
        
        # Extract subgraph adjacency matrix
        subgraph_adj = adj[group_nodes][:, group_nodes]
        
        return subgraph_adj
    
    def get_subgraph_edges(self, group_nodes):
        """그룹 내 노드들의 edge_index, edge_weight 추출
        
        Args:
            group_nodes: List of global node indices in the group
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - edge_index: [2, E] local indices
                - edge_weight: [E] edge weights
        """
        N = len(group_nodes)
        
        if self.adj_matrix is None:
            # Fallback: self-loop only
            idx = torch.arange(N, dtype=torch.long)
            return torch.stack([idx, idx], dim=0), torch.ones(N)
        
        # Convert to tensor if needed
        if hasattr(self.adj_matrix, 'cpu'):
            adj = self.adj_matrix
        else:
            adj = torch.tensor(self.adj_matrix, dtype=torch.float32)
        
        # 글로벌 인덱스 -> 로컬 인덱스 매핑
        edges = []
        weights = []
        for i, gi in enumerate(group_nodes):
            for j, gj in enumerate(group_nodes):
                w = adj[gi, gj].item()
                if w > 0:
                    edges.append([i, j])
                    weights.append(w)
        
        if len(edges) == 0:
            # No edges: self-loop fallback
            idx = torch.arange(N, dtype=torch.long)
            return torch.stack([idx, idx], dim=0), torch.ones(N)
        
        edge_index = torch.tensor(edges, dtype=torch.long).T
        edge_weight = torch.tensor(weights, dtype=torch.float32)
        return edge_index, edge_weight
    
    def create_node_groups(self):
        """Create node groups based on specified method"""
        if self.grouping_method == 'random':
            return self._random_grouping()
        elif self.grouping_method in ['dirichlet', 'random_imbalanced', 'imbalance_random']:
            return self._dirichlet_grouping()
        elif self.grouping_method == 'metis':
            return self._metis_grouping()
        elif self.grouping_method == 'maxcut':
            return self._maxcut_grouping()
        else:
            raise ValueError(f"Unknown grouping method: {self.grouping_method}")
    
    def _random_grouping(self):
        print(f"\n🎲 Random 그룹핑 진행 중...")

        import hashlib
        seed_str = f"{self.num_nodes}_{self.num_groups}_imb{self.imbalance}_a{self.imbalance_alpha}_m{self.min_nodes_per_group}"
        seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16) % (2**32)
        if self.random_seed is not None:
            seed = int(self.random_seed) % (2**32)

        rng = np.random.default_rng(seed)
        print(f"  - Random seed: {seed} (재현성 보장)")

        nodes = np.arange(self.num_nodes, dtype=np.int64)
        rng.shuffle(nodes)

        # ✅ 불균형이 지정되면 dirichlet sizes 사용
        if self.imbalance is not None or self.imbalance_alpha is not None:
            sizes = self._sample_group_sizes_dirichlet(rng)
            sizes = sizes[rng.permutation(self.num_groups)]  # 큰 그룹이 특정 client에 고정되지 않게
            print(f"  - Dirichlet alpha: {self._resolve_dirichlet_alpha():.6g}")
            print(f"  - 그룹 크기: {sizes.tolist()}")

            groups = []
            start = 0
            for s in sizes:
                end = start + int(s)
                groups.append(nodes[start:end].tolist())
                start = end
            print(f"  ✅ Random(Imbalanced) 그룹핑 완료")
            return groups

        # ✅ (기존 균등 분할 로직 유지)
        group_size = self.num_nodes // self.num_groups
        groups = []
        for i in range(self.num_groups):
            start_idx = i * group_size
            end_idx = self.num_nodes if i == self.num_groups - 1 else (i + 1) * group_size
            groups.append(nodes[start_idx:end_idx].tolist())
        print(f"  ✅ Random 그룹핑 완료")
        return groups
    
    def _metis_grouping(self):
        """METIS-based graph partitioning for balanced groups"""
        if self.adj_matrix is None:
            print("⚠️ Warning: adj_matrix is None, falling back to random grouping")
            return self._random_grouping()
        
        try:
            print(f"\n📊 METIS Graph Partitioning 진행 중...")
            
            try:
                import metis
                print(f"  - METIS 라이브러리 사용")
                
                # Convert adjacency matrix to METIS format
                adj_np = self.adj_matrix.cpu().numpy() if hasattr(self.adj_matrix, 'cpu') else self.adj_matrix
                
                # Create graph for METIS (need to convert to the right format)
                G = nx.from_numpy_array(adj_np)
                
                # Convert NetworkX graph to METIS format
                adjacency_list = [list(G.neighbors(node)) for node in range(len(G.nodes))]
                
                # Perform METIS partitioning
                (cut, groups_metis) = metis.part_graph(adjacency_list, self.num_groups)
                
                print(f"  - Graph cut value: {cut}")
                
                # Convert METIS output to our format
                groups = [[] for _ in range(self.num_groups)]
                for node, group_id in enumerate(groups_metis):
                    groups[group_id].append(node)
                
                print(f"  ✅ METIS 그래프 분할 완료")
                return groups
                
            except ImportError:
                print("  ⚠️ Warning: METIS not available, using recursive spectral bisection")
                return self._recursive_spectral_bisection()
                
        except Exception as e:
            print(f"  ❌ Warning: METIS grouping failed ({e}), falling back to random grouping")
            return self._random_grouping()
    
    def _recursive_spectral_bisection(self):
        """Recursive spectral bisection for graph partitioning"""
        print(f"  - Recursive Spectral Bisection 사용")
        
        adj_np = self.adj_matrix.cpu().numpy() if hasattr(self.adj_matrix, 'cpu') else self.adj_matrix
        
        # Start with all nodes in one group
        groups = [list(range(self.num_nodes))]
        
        # Recursively split until we have enough groups
        while len(groups) < self.num_groups:
            # Find the largest group to split
            largest_group_idx = max(range(len(groups)), key=lambda i: len(groups[i]))
            largest_group = groups[largest_group_idx]
            
            if len(largest_group) <= 1:
                # Can't split further, break
                break
            
            # Extract subgraph adjacency matrix
            subgraph_adj = adj_np[np.ix_(largest_group, largest_group)]
            
            # Use spectral clustering to split into 2 groups
            clustering = SpectralClustering(n_clusters=2, affinity='precomputed', random_state=42)
            labels = clustering.fit_predict(subgraph_adj)
            
            # Create two new groups
            group1 = [largest_group[i] for i in range(len(largest_group)) if labels[i] == 0]
            group2 = [largest_group[i] for i in range(len(largest_group)) if labels[i] == 1]
            
            # Replace the largest group with the two new groups
            groups[largest_group_idx] = group1
            groups.append(group2)
            
            print(f"    Split group of size {len(largest_group)} into {len(group1)} and {len(group2)}")
        
        # If we have more groups than needed, merge smallest ones
        while len(groups) > self.num_groups:
            groups.sort(key=len)
            print(f"    Merging groups of size {len(groups[0])} and {len(groups[1])}")
            groups[0].extend(groups[1])
            groups.pop(1)
        
        print(f"  ✅ Recursive Spectral Bisection 완료")
        return groups

    def _maxcut_grouping(self):
        """
        Balanced max-cut style partitioning (heuristic).

        Goal:
        maximize inter-group edge weight while keeping groups balanced.

        Strategy (greedy, deterministic):
        - build weighted neighbor lists
        - maintain per-node "affinity" score to each group (sum of weights to nodes already in group)
        - when placing a node, prefer the group where its affinity is LOW (to maximize cut),
            subject to capacity constraints.
        """
        if self.adj_matrix is None:
            print("⚠️ Warning: adj_matrix is None, falling back to random grouping")
            return self._random_grouping()

        print(f"\n✂️  MaxCut-like Balanced Partitioning 진행 중...")

        # deterministic seed (same style as random)
        import hashlib
        seed_str = f"maxcut_{self.num_nodes}_{self.num_groups}"
        seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16) % (2**32)
        rng = np.random.default_rng(seed)
        print(f"  - MaxCut seed: {seed} (재현성 보장)")

        adj = self.adj_matrix.cpu().numpy() if hasattr(self.adj_matrix, "cpu") else self.adj_matrix
        adj = np.asarray(adj, dtype=np.float32)

        N = self.num_nodes
        G = self.num_groups

        # capacity (balanced or imbalanced)
        if self.imbalance is not None or self.imbalance_alpha is not None:
            caps = self._sample_group_sizes_dirichlet(rng)
            caps = caps[rng.permutation(G)]
            print(f"  - Dirichlet alpha: {self._resolve_dirichlet_alpha():.6g}")
            print(f"  - target caps: {caps.tolist()}")
        else:
            base = N // G
            rem = N % G
            caps = np.array([base + (1 if i < rem else 0) for i in range(G)], dtype=np.int64)

        # precompute neighbor list for speed
        neighbors = []
        for i in range(N):
            js = np.nonzero(adj[i] > 0)[0]
            ws = adj[i, js]
            neighbors.append((js.astype(np.int64), ws.astype(np.float32)))

        # group assignments
        groups = [[] for _ in range(G)]
        group_sizes = np.zeros(G, dtype=np.int64)
        assign = -np.ones(N, dtype=np.int64)

        # score[node, g] = sum of weights from node to nodes already in group g
        # We update incrementally.
        score = np.zeros((N, G), dtype=np.float32)

        # choose seeds: pick high-degree nodes to spread out
        degrees = np.array([neighbors[i][0].size for i in range(N)], dtype=np.int64)
        seed_nodes = np.argsort(-degrees)[:G].tolist()
        # if graph is tiny / isolated, fill with random
        if len(seed_nodes) < G:
            remaining = [i for i in range(N) if i not in seed_nodes]
            rng.shuffle(remaining)
            seed_nodes += remaining[: (G - len(seed_nodes))]

        # place one seed per group
        for g, node in enumerate(seed_nodes[:G]):
            groups[g].append(int(node))
            group_sizes[g] += 1
            assign[node] = g
            # update scores of neighbors
            js, ws = neighbors[node]
            score[js, g] += ws

        # remaining nodes
        unassigned = [i for i in range(N) if assign[i] < 0]

        # order: hard nodes first (high degree)
        unassigned.sort(key=lambda i: degrees[i], reverse=True)

        for node in unassigned:
            # candidate groups with remaining capacity
            feasible = np.where(group_sizes < caps)[0]
            if feasible.size == 0:
                # should not happen
                feasible = np.arange(G)

            # MaxCut heuristic: put node into group with MIN affinity score (less internal edges)
            # tie-break: choose smaller group, then random
            s = score[node, feasible]
            min_s = np.min(s)
            cand = feasible[np.where(s == min_s)[0]]

            # tie-break by group size
            min_size = np.min(group_sizes[cand])
            cand2 = cand[group_sizes[cand] == min_size]

            # deterministic random tie-break
            g = int(rng.choice(cand2))

            groups[g].append(int(node))
            group_sizes[g] += 1
            assign[node] = g

            # update neighbor scores
            js, ws = neighbors[node]
            score[js, g] += ws

        print("  ✅ MaxCut-like 그룹핑 완료")
        return groups

    def _make_seed(self, tag: str) -> int:
        """결정적 seed 생성 (random_seed가 있으면 그걸 사용)"""
        if self.random_seed is not None:
            return int(self.random_seed) % (2**32)

        import hashlib
        seed_str = f"{tag}_{self.num_nodes}_{self.num_groups}_{self.imbalance}_{self.imbalance_alpha}_{self.min_nodes_per_group}"
        return int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16) % (2**32)

    def _resolve_dirichlet_alpha(self) -> float:
        """
        imbalance(0~1) -> alpha로 매핑.
        - imbalance=0  -> alpha=100 (거의 균등)
        - imbalance=1  -> alpha=0.05 (극단 불균형)
        """
        if self.imbalance_alpha is not None:
            alpha = float(self.imbalance_alpha)
        else:
            if self.imbalance is None:
                # 기본값: 적당히 불균형
                imb = 0.8
            else:
                imb = float(self.imbalance)
            imb = max(0.0, min(1.0, imb))
            alpha = 0.05 * (2000.0 ** (1.0 - imb))  # 0->100, 1->0.05

        # 너무 작으면 수치적으로 불안정할 수 있으니 하한
        return max(alpha, 1e-4)


    def _sample_group_sizes_dirichlet(self, *args) -> np.ndarray:
        """
        Backward/forward compatible:
        - _sample_group_sizes_dirichlet(rㄴng)
        - _sample_group_sizes_dirichlet(alpha, rng)
        """
        if len(args) == 1:
            rng = args[0]
            alpha = self._resolve_dirichlet_alpha()
        elif len(args) == 2:
            alpha, rng = args
        else:
            raise TypeError("Expected (rng) or (alpha, rng)")

        N = int(self.num_nodes)
        G = int(self.num_groups)
        m = int(self.min_nodes_per_group)

        if N < G * m:
            raise ValueError(f"num_nodes({N}) < num_groups({G}) * min_nodes_per_group({m}).")

        remaining = N - G * m
        if remaining == 0:
            return np.full(G, m, dtype=np.int64)

        props = rng.dirichlet(np.full(G, float(alpha), dtype=np.float64))
        raw = props * remaining
        extra = np.floor(raw).astype(np.int64)
        diff = remaining - int(extra.sum())

        if diff > 0:
            frac = raw - extra
            order = np.argsort(-frac)
            extra[order[:diff]] += 1
        elif diff < 0:
            order = np.argsort(-extra)
            k = 0
            while diff < 0 and k < 10 * G:
                i = order[k % G]
                if extra[i] > 0:
                    extra[i] -= 1
                    diff += 1
                k += 1

        sizes = extra + m
        sizes[0] += N - int(sizes.sum())
        assert int(sizes.sum()) == N
        return sizes.astype(np.int64)


    def _analyze_adjacency_matrix(self):
        """Analyze and print adjacency matrix statistics"""
        if hasattr(self.adj_matrix, 'cpu'):
            adj = self.adj_matrix.cpu().numpy()
        else:
            adj = self.adj_matrix
        
        # Calculate node degrees (number of neighbors)
        degrees = np.sum(adj > 0, axis=1)  # Count non-zero connections
        
        print(f"\n📊 그래프 분석 결과:")
        print(f"  - 총 노드 수: {self.num_nodes}")
        print(f"  - 총 엣지 수: {np.sum(adj > 0) // 2}")  # Undirected graph
        print(f"  - 평균 차수: {np.mean(degrees):.2f}")
        print(f"  - 최대 차수: {np.max(degrees)}")
        print(f"  - 최소 차수: {np.min(degrees)}")
        print(f"  - 차수 표준편차: {np.std(degrees):.2f}")
        
        # Print degree distribution
        unique_degrees, counts = np.unique(degrees, return_counts=True)
        print(f"  - 차수 분포:")
        for deg, count in zip(unique_degrees, counts):
            print(f"    차수 {deg}: {count}개 노드 ({count/self.num_nodes*100:.1f}%)")
        
        # Print nodes with highest/lowest degrees
        max_degree_nodes = np.where(degrees == np.max(degrees))[0]
        min_degree_nodes = np.where(degrees == np.min(degrees))[0]
        
        print(f"  - 최고 차수 노드들: {max_degree_nodes[:5].tolist()}")  # Show first 5
        print(f"  - 최저 차수 노드들: {min_degree_nodes[:5].tolist()}")  # Show first 5
    
    def print_grouping_results(self, node_groups):
        """Print node grouping results"""
        print(f"\n🔗 노드 그룹핑 결과:")
        print(f"  - 그룹핑 방법: {self.grouping_method}")
        print(f"  - 총 그룹 수: {len(node_groups)}")

        # Group size statistics
        group_sizes = [len(group) for group in node_groups]
        print(f"  - 그룹 크기 분포:")
        print(f"    평균: {np.mean(group_sizes):.1f}")
        print(f"    최대: {np.max(group_sizes)}")
        print(f"    최소: {np.min(group_sizes)}")
        print(f"    표준편차: {np.std(group_sizes):.2f}")
        
        # Print each group details
        print(f"  - 각 그룹별 상세:")
        for i, group in enumerate(node_groups):
            print(f"    Group {i}: {len(group)}개 노드 - {group[:10]}{'...' if len(group) > 10 else ''}")
        
        # Verify all nodes are included
        all_nodes = set()
        for group in node_groups:
            all_nodes.update(group)
        
        if len(all_nodes) == self.num_nodes and all_nodes == set(range(self.num_nodes)):
            print(f"  ✅ 모든 노드가 올바르게 그룹에 할당됨")
        else:
            print(f"  ❌ 노드 할당 오류: {len(all_nodes)}/{self.num_nodes} 노드 할당됨")


def setup_split_learning_nodes(
    num_nodes: int,
    num_clients: int,
    grouping_method: str = 'metis',
    adj_matrix: torch.Tensor = None,
    dataset_name: str = None,
    load_adj_func = None,
    # ✅ 추가
    imbalance: float | None = None,
    imbalance_alpha: float | None = None,
    min_nodes_per_client: int = 1,
    random_seed: int | None = None,
) -> tuple:
    """
    Split Learning을 위한 노드 그룹 및 subgraph adjacency 리스트 생성
    
    Args:
        num_nodes: 전체 노드 수
        num_clients: 클라이언트 수
        grouping_method: 노드 분할 방법 ('metis' or 'random')
        adj_matrix: 인접 행렬 (None이면 로드 시도)
        dataset_name: 데이터셋 이름 (adj_matrix가 None일 때 로드 시도)
        load_adj_func: adjacency matrix 로드 함수 (선택적)
        
    Returns:
        tuple: (client_nodes_list, subgraph_adj_list, node_partitioner)
            - client_nodes_list: 각 클라이언트의 노드 리스트
            - subgraph_adj_list: 각 클라이언트의 subgraph adjacency matrix 리스트 (None일 수 있음)
            - node_partitioner: NodePartitioner 인스턴스
    """
    # Adjacency matrix 로드 (없는 경우)
    if adj_matrix is None and dataset_name is not None and load_adj_func is not None:
        try:
            adj_path = f"datasets/{dataset_name}/adj_mx.pkl"
            adj_list, adj_mx_raw = load_adj_func(adj_path, "doubletransition")
            adj_matrix = torch.tensor(adj_mx_raw, dtype=torch.float32)
        except Exception as e:
            print(f"Adjacency matrix 로드 실패: {e}, random grouping 사용")
            adj_matrix = None
    
    # 실제 grouping method 결정
    actual_grouping = grouping_method
    if adj_matrix is None and grouping_method in ['metis', 'maxcut']:
        actual_grouping = 'random'
        print("⚠️ Warning: adj_matrix가 없어 random grouping으로 변경")
    
    # NodePartitioner 생성 및 노드 그룹 생성
    node_partitioner = NodePartitioner(
        num_nodes=num_nodes,
        num_groups=num_clients,
        grouping_method=actual_grouping,
        adj_matrix=adj_matrix,
        # ✅ 추가 전달
        imbalance=imbalance,
        imbalance_alpha=imbalance_alpha,
        min_nodes_per_group=min_nodes_per_client,
        random_seed=random_seed,
    )
    client_nodes_list = node_partitioner.create_node_groups()
    node_partitioner.print_grouping_results(client_nodes_list)
    
    subgraph_adj_list = []
    if adj_matrix is None:
        subgraph_adj_list = [None for _ in client_nodes_list]
    else:
        for client_nodes in client_nodes_list:
            subgraph_adj_list.append(node_partitioner.get_subgraph_adjacency(client_nodes))
    
    return client_nodes_list, subgraph_adj_list, node_partitioner 