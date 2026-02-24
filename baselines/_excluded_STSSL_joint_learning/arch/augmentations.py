"""
Augmentation functions for ST-SSL.

Reference: https://github.com/Echo-Ji/ST-SSL/blob/master/model/aug.py
"""
import copy
import numpy as np
import torch
import torch.nn.functional as F


def sim_global(flow_data: torch.Tensor, sim_type: str = 'cos') -> torch.Tensor:
    """Calculate the global similarity of traffic flow data.

    Args:
        flow_data: tensor, original flow [n,l,v,c] or location embedding [n,v,c]
        sim_type: str, type of similarity, attention or cosine. ['att', 'cos']

    Returns:
        sim: tensor, symmetric similarity, [v,v]
    """
    if len(flow_data.shape) == 4:
        n, l, v, c = flow_data.shape
        att_scaling = n * l * c
        cos_scaling = torch.norm(flow_data, p=2, dim=(0, 1, 3)) ** -1  # cal 2-norm of each node, dim N
        sim = torch.einsum('btnc, btmc->nm', flow_data, flow_data)
    elif len(flow_data.shape) == 3:
        n, v, c = flow_data.shape
        att_scaling = n * c
        cos_scaling = torch.norm(flow_data, p=2, dim=(0, 2)) ** -1  # cal 2-norm of each node, dim N
        sim = torch.einsum('bnc, bmc->nm', flow_data, flow_data)
    else:
        raise ValueError(f'sim_global only support shape length in [3, 4] but got {len(flow_data.shape)}.')

    if sim_type == 'cos':
        # cosine similarity
        scaling = torch.einsum('i, j->ij', cos_scaling, cos_scaling)
        sim = sim * scaling
    elif sim_type == 'att':
        # scaled dot product similarity
        scaling = float(att_scaling) ** -0.5
        sim = torch.softmax(sim * scaling, dim=-1)
    else:
        raise ValueError('sim_global only support sim_type in [att, cos].')

    return sim


def aug_topology(sim_mx: torch.Tensor, input_graph: torch.Tensor, percent: float = 0.2) -> torch.Tensor:
    """Generate the data augmentation from topology (graph structure) perspective
        for undirected graph without self-loop.

    Args:
        sim_mx: tensor, symmetric similarity, [v,v]
        input_graph: tensor, adjacency matrix without self-loop, [v,v]
        percent: float, percentage of edges to drop/add

    Returns:
        aug_graph: tensor, augmented adjacency matrix on cuda, [v,v]
    """
    device = input_graph.device

    # Edge dropping starts here
    drop_percent = percent / 2

    index_list = input_graph.nonzero()  # list of edges [row_idx, col_idx]

    edge_num = int(index_list.shape[0] / 2)  # treat one undirected edge as two edges
    edge_mask = (input_graph > 0).tril(diagonal=-1)
    add_drop_num = int(edge_num * drop_percent / 2)

    if add_drop_num == 0:
        return input_graph.clone()

    aug_graph = input_graph.clone()

    # Get similarity values for existing edges (lower triangular)
    edge_sim = sim_mx[edge_mask].cpu()
    drop_prob = torch.softmax(-edge_sim, dim=0).numpy()  # lower similarity -> higher drop prob
    drop_prob = drop_prob / drop_prob.sum()

    # Get indices of lower triangular edges
    lower_tri_indices = edge_mask.nonzero().cpu()

    if len(lower_tri_indices) > 0 and add_drop_num > 0:
        drop_list = np.random.choice(len(lower_tri_indices), size=min(add_drop_num, len(lower_tri_indices)),
                                     p=drop_prob, replace=False)
        drop_index = lower_tri_indices[drop_list]

        zeros = torch.zeros(1, device=device, dtype=aug_graph.dtype)
        for idx in drop_index:
            aug_graph[idx[0], idx[1]] = zeros
            aug_graph[idx[1], idx[0]] = zeros

    # Edge adding starts here
    node_num = input_graph.shape[0]

    # Get lower triangular mask for non-edges
    non_edge_mask = (input_graph == 0).tril(diagonal=-1)
    non_edge_indices = non_edge_mask.nonzero().cpu()

    if len(non_edge_indices) > 0 and add_drop_num > 0:
        # Get similarity for non-edges
        non_edge_sim = sim_mx[non_edge_mask].cpu()
        add_prob = torch.softmax(non_edge_sim, dim=0).numpy()  # higher similarity -> higher add prob
        add_prob = add_prob / add_prob.sum()

        add_list = np.random.choice(len(non_edge_indices), size=min(add_drop_num, len(non_edge_indices)),
                                    p=add_prob, replace=False)
        add_index = non_edge_indices[add_list]

        ones = torch.ones(1, device=device, dtype=aug_graph.dtype)
        for idx in add_index:
            aug_graph[idx[0], idx[1]] = ones
            aug_graph[idx[1], idx[0]] = ones

    return aug_graph


def aug_traffic(t_sim_mx: torch.Tensor, flow_data: torch.Tensor, percent: float = 0.2) -> torch.Tensor:
    """Generate the data augmentation from traffic (node attribute) perspective.

    Args:
        t_sim_mx: temporal similarity matrix after softmax, [l, n, v]
        flow_data: input flow data, [n, l, v, c]
        percent: float, percentage of positions to mask

    Returns:
        aug_flow: augmented flow data, [n, l, v, c]
    """
    l, n, v = t_sim_mx.shape
    mask_num = int(n * l * v * percent)
    aug_flow = flow_data.clone()

    # Lower temporal similarity -> higher probability to be masked
    mask_prob = (1. - t_sim_mx.permute(1, 0, 2).reshape(-1)).cpu().numpy()
    mask_prob = np.clip(mask_prob, 0, None)  # Ensure non-negative
    mask_prob = mask_prob / mask_prob.sum()

    # Create index meshgrid
    x, y, z = np.meshgrid(range(n), range(l), range(v), indexing='ij')
    mask_list = np.random.choice(n * l * v, size=mask_num, p=mask_prob, replace=False)

    # Apply mask
    zeros = torch.zeros_like(aug_flow[0, 0, 0])
    aug_flow[
        x.reshape(-1)[mask_list],
        y.reshape(-1)[mask_list],
        z.reshape(-1)[mask_list]] = zeros

    return aug_flow
