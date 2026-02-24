import numpy as np
import torch
from typing import List, Optional
import networkx as nx
import metis
from sklearn.cluster import SpectralClustering

import csv
from pathlib import Path
from typing import Dict, List

class NodePartitioner:
    """Node partitioning utilities for S2Former"""
    
    def __init__(self, num_nodes, num_groups, grouping_method, adj_matrix, metadata):
        self.num_nodes = num_nodes
        self.num_groups = num_groups
        self.grouping_method = grouping_method
        self.adj_matrix = adj_matrix
        
        self.metadata = metadata
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
        elif self.grouping_method == 'metis':
            return self._metis_grouping()
        elif self.grouping_method == 'city':
            return city_partition(self.metadata)
        elif self.grouping_method == 'road_type':
            return road_type_partition(self.metadata)
        else:
            raise ValueError(f"Unknown grouping method: {self.grouping_method}")
    
    def _random_grouping(self):
        return random_partition(self.num_nodes, self.num_groups)

    
    def _metis_grouping(self):
        return metis_partition(self.adj_matrix)

    
    def _recursive_spectral_bisection(self):
        return spectral_partition(self.adj_matrix)

    
    def _analyze_adjacency_matrix(self):
        if self.adj_matrix is None:
            return
        
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
    metadata: Dict[str, List[str]] = None,
) -> tuple:
    """
    Split Learning을 위한 노드 그룹 및 subgraph adjacency 리스트 생성
    
    Args:
        num_nodes: 전체 노드 수
        num_clients: 클라이언트 수
        grouping_method: 노드 분할 방법 ('metis' or 'random')
        adj_matrix: 인접 행렬 (None이면 로드 시도)
        metadata: 메타데이터 ({column_name: [row0, row1, ...]})
    Returns:
        tuple: (client_nodes_list, subgraph_adj_list, node_partitioner)
            - client_nodes_list: 각 클라이언트의 노드 리스트
            - subgraph_adj_list: 각 클라이언트의 subgraph adjacency matrix 리스트 (None일 수 있음)
            - node_partitioner: NodePartitioner 인스턴스
    """
    # NodePartitioner 생성 및 노드 그룹 생성
    node_partitioner = NodePartitioner(
        num_nodes=num_nodes,
        num_groups=num_clients,
        grouping_method=grouping_method,
        adj_matrix=adj_matrix,
        metadata=metadata,
    )
    client_nodes_list = node_partitioner.create_node_groups()
    node_partitioner.print_grouping_results(client_nodes_list)
    
    subgraph_adj_list = []
    for client_nodes in client_nodes_list:
        subgraph_adj = node_partitioner.get_subgraph_adjacency(client_nodes)
        subgraph_adj_list.append(subgraph_adj)
    
    return client_nodes_list, subgraph_adj_list, node_partitioner 


def random_partition(total_nodes: int, num_clients: int) -> List[List[int]]:
    """랜덤 노드 분할"""
    all_nodes = list(range(total_nodes))
    np.random.seed(42)
    np.random.shuffle(all_nodes)
    
    base_size = total_nodes // num_clients
    remainder = total_nodes % num_clients
    
    groups = []
    start = 0
    for i in range(num_clients):
        size = base_size + (1 if i < remainder else 0)
        groups.append(all_nodes[start:start + size])
        start += size
    
    return groups

def metis_partition(adj_matrix: Optional[torch.Tensor]) -> List[List[int]]:
    """METIS 기반 그래프 분할"""
    adj_np = adj_matrix.cpu().numpy() if hasattr(adj_matrix, 'cpu') else adj_matrix
    G = nx.from_numpy_array(adj_np)
    _, parts = metis.part_graph(G, self.num_clients)
    
    groups = [[] for _ in range(self.num_clients)]
    for node_idx, part_idx in enumerate(parts):
        groups[part_idx].append(node_idx)
    
    return groups

def spectral_partition(adj_matrix: Optional[torch.Tensor]) -> List[List[int]]:
    """Spectral Clustering 기반 분할"""
    adj_np = adj_matrix.cpu().numpy() if hasattr(adj_matrix, 'cpu') else adj_matrix
    clustering = SpectralClustering(
        n_clusters=self.num_clients,
        affinity='precomputed',
        random_state=42
    )
    labels = clustering.fit_predict(adj_np)
    
    groups = [[] for _ in range(self.num_clients)]
    for node_idx, label in enumerate(labels):
        groups[label].append(node_idx)
    
    return groups



def load_metadata_csv(metadata_csv_path: str) -> Dict[str, List[str]]:
    """
    metadata.csv를 읽어서 {column_name: [row0, row1, ...]} 형태로 반환합니다.

    주의: row index가 node index(0..N-1)와 1:1로 대응된다는 가정입니다.
    """
    path = Path(metadata_csv_path)
    if not path.exists():
        raise FileNotFoundError(f"metadata.csv not found: {metadata_csv_path}")

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"metadata.csv has no header: {metadata_csv_path}")

        columns: Dict[str, List[str]] = {k: [] for k in reader.fieldnames}
        for row in reader:
            for k in columns.keys():
                columns[k].append("" if row.get(k) is None else str(row.get(k)))

    return columns


def road_type_partition(metadata: Dict[str, List[str]]) -> List[List[int]]:
    """road_type 컬럼 값 기준으로 노드를 그룹핑합니다."""
    road_types = metadata["Type"]
    groups: Dict[str, List[int]] = {}
    for i, road_type in enumerate(road_types):
        if road_type not in groups:
            groups[road_type] = []
        groups[road_type].append(i)
    return list(groups.values())


def city_partition(metadata: Dict[str, List[str]]) -> List[List[int]]:
    """city 컬럼 값 기준으로 노드를 그룹핑합니다."""
    cities = metadata["City"]
    groups: Dict[str, List[int]] = {}
    for i, city in enumerate(cities):
        if city not in groups:
            groups[city] = []
        groups[city].append(i)
    return list(groups.values())
