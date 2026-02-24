import torch
import torch.nn as nn

from torch_geometric.nn import MetaLayer


def scatter_add(src, index, dim=0, dim_size=None):
    """
    Simple scatter_add implementation using native PyTorch.

    src   : tensor with shape [..., E, ...] (we scatter along `dim`)
    index : 1D long tensor of length E
    dim   : scatter dimension
    dim_size : size of output along `dim`
    """
    if dim_size is None:
        dim_size = int(index.max()) + 1

    # out shape: same as src but size[dim] = dim_size
    out_shape = list(src.shape)
    out_shape[dim] = dim_size
    out = src.new_zeros(out_shape)

    # expand index to match src shape along other dims
    # index: [E] -> [E, 1, 1, ...] and broadcast
    expand_dims = [index.shape[0]] + [1] * (src.dim() - 1)
    index_expanded = index.view(*expand_dims).expand_as(src)

    out.scatter_add_(dim, index_expanded, src)
    return out


class MLP_GN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, 
        hidden_layer_num, activation='ReLU', dropout=0.0):
        super().__init__()
        self.net = []
        last_layer_size = input_size
        for _ in range(hidden_layer_num):
            self.net.append(nn.Linear(last_layer_size, hidden_size))
            self.net.append(getattr(nn, activation)())
            self.net.append(nn.Dropout(p=dropout))
            last_layer_size = hidden_size
        self.net.append(nn.Linear(last_layer_size, output_size))
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        return self.net(x)


class EdgeModel(nn.Module):
    def __init__(self, 
        node_input_size, edge_input_size, global_input_size, 
        hidden_size, edge_output_size, activation, dropout):
        super(EdgeModel, self).__init__()
        edge_mlp_input_size = 2 * node_input_size + edge_input_size + global_input_size
        self.edge_mlp = MLP_GN(edge_mlp_input_size, hidden_size, edge_output_size, 2, activation, dropout)

    def forward(self, src, dest, edge_attr, u, batch):
        # source, target: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        out = torch.cat([src, dest, edge_attr], -1)
        if u is not None:
            out = torch.cat([out, u[batch]], -1)
        return self.edge_mlp(out)


class NodeModel(nn.Module):
    def __init__(self,
        node_input_size, edge_input_size, global_input_size,
        hidden_size, node_output_size, activation, dropout):
        super(NodeModel, self).__init__()
        node_mlp_input_size = node_input_size + edge_input_size + global_input_size
        self.node_mlp = MLP_GN(node_mlp_input_size, hidden_size, node_output_size, 2, activation, dropout)

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index
        received_msg = scatter_add(edge_attr, col, dim=0, dim_size=x.size(0))
        out = torch.cat([x, received_msg], dim=-1)
        if u is not None:
            out = torch.cat([out, u[batch]], dim=-1)
        return self.node_mlp(out)


class GlobalModel(nn.Module):
    def __init__(self,
        node_input_size, edge_input_size, global_input_size,
        hidden_size, global_output_size, activation, dropout):
        super(GlobalModel, self).__init__()
        global_mlp_input_size = node_input_size + edge_input_size + global_input_size
        self.global_mlp = MLP_GN(global_mlp_input_size, hidden_size, global_output_size, 2, activation, dropout)

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index
        agg_node = scatter_add(x, batch, dim=0)
        agg_edge = scatter_add(scatter_add(edge_attr, col, dim=0, dim_size=x.size(0)), batch, dim=0)
        out = torch.cat([agg_node, agg_edge, u], dim=-1)
        return self.global_mlp(out)


class GraphNet(nn.Module):
    """Graph Network module for graph-based message passing.
    
    This module implements a Graph Network (GN) that performs message passing
    between nodes using edge and global information.
    """
    
    def __init__(self, node_input_size, edge_input_size, global_input_size, 
        hidden_size,
        updated_node_size, updated_edge_size, updated_global_size,
        node_output_size,
        gn_layer_num, activation='ReLU', dropout=0.0):
        super().__init__()

        self.global_input_size = global_input_size

        self.net = []
        last_node_input_size = node_input_size
        last_edge_input_size = edge_input_size
        last_global_input_size = global_input_size
        for _ in range(gn_layer_num):
            edge_model = EdgeModel(last_node_input_size, last_edge_input_size, last_global_input_size, hidden_size, updated_edge_size,
                activation, dropout)
            last_edge_input_size += updated_edge_size
            node_model = NodeModel(last_node_input_size, updated_edge_size, last_global_input_size, hidden_size, updated_node_size,
                activation, dropout)
            last_node_input_size += updated_node_size
            global_model = GlobalModel(updated_node_size, updated_edge_size, last_global_input_size, hidden_size, updated_global_size,
                activation, dropout)
            last_global_input_size += updated_global_size
            self.net.append(MetaLayer(
                edge_model, node_model, global_model
            ))
        self.net = nn.ModuleList(self.net)
        self.node_out_net = nn.Linear(last_node_input_size, node_output_size)
    
    def forward(self, x, edge_index, edge_attr, batch=None):
        """Forward pass of the GraphNet.
        
        Args:
            x: Node features [N, B, L, F] where N=nodes, B=batch, L=layers, F=features
            edge_index: Edge indices [2, E]
            edge_attr: Edge attributes [E, 1, 1, 1] (will be expanded)
            batch: Batch assignment for each node [N]
            
        Returns:
            Node output features [N, B, L, node_output_size]
        """
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)
        batch = batch.to(x.device)
        edge_attr = edge_attr.expand(-1, x.shape[1], x.shape[2], -1)
        u = x.new_zeros(*([batch[-1] + 1] + list(x.shape[1:-1]) + [self.global_input_size]))
        for net in self.net:
            updated_x, updated_edge_attr, updated_u = net(x, edge_index, edge_attr, u, batch)
            x = torch.cat([updated_x, x], dim=-1)
            edge_attr = torch.cat([updated_edge_attr, edge_attr], dim=-1)
            u = torch.cat([updated_u, u], dim=-1)
        node_out = self.node_out_net(x)
        return node_out

