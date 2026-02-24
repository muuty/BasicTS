import numpy as np
import torch
from typing import List, Optional
import networkx as nx
import metis
from sklearn.cluster import SpectralClustering


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

