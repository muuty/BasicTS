import numpy as np
import torch
import torch.nn as nn

from .graph_nets import GraphNet


class GRUSeq2SeqWithGraphNet(nn.Module):
    """
    GRU-based Seq2Seq model with Graph Neural Network for spatial-temporal forecasting.
    
    This model combines:
    1. GRU Encoder: Encodes historical time series data
    2. GraphNet: Learns spatial relationships between nodes
    3. GRU Decoder: Generates predictions with optional curriculum learning
    
    Original paper reference: KDD 2021 CNFGNN
    """
    
    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        output_dim: int,
        hidden_size: int = 128,
        gru_num_layers: int = 2,
        dropout: float = 0.0,
        cl_decay_steps: int = 1000,
        use_curriculum_learning: bool = True,
        gn_layer_num: int = 2,
        gn_hidden_size: int = 256,
        gn_updated_node_size: int = 128,
        gn_updated_edge_size: int = 128,
        gn_updated_global_size: int = 128,
        seq_len: int = 12,
        horizon: int = 12,
        adj_mx: torch.Tensor = None,
        edge_index: torch.Tensor = None,
        edge_attr: torch.Tensor = None,
    ):
        """
        Initialize GRUSeq2SeqWithGraphNet model.
        
        Args:
            num_nodes: Number of nodes in the graph
            input_dim: Input feature dimension
            output_dim: Output feature dimension
            hidden_size: Hidden size of GRU
            gru_num_layers: Number of GRU layers
            dropout: Dropout rate
            cl_decay_steps: Curriculum learning decay steps
            use_curriculum_learning: Whether to use curriculum learning
            gn_layer_num: Number of GraphNet layers
            gn_hidden_size: Hidden size in GraphNet MLPs
            gn_updated_node_size: Updated node feature size in GraphNet
            gn_updated_edge_size: Updated edge feature size in GraphNet
            gn_updated_global_size: Updated global feature size in GraphNet
            seq_len: Input sequence length
            horizon: Output sequence length (prediction horizon)
            adj_mx: Adjacency matrix (optional, for compatibility)
            edge_index: Edge indices [2, E] for graph structure
            edge_attr: Edge attributes [E]
        """
        super().__init__()
        
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.gru_num_layers = gru_num_layers
        self.seq_len = seq_len
        self.horizon = horizon
        self.cl_decay_steps = cl_decay_steps
        self.use_curriculum_learning = use_curriculum_learning
        
        # Store graph structure
        if edge_index is not None:
            self.register_buffer('edge_index', edge_index)
        else:
            self.edge_index = None
            
        if edge_attr is not None:
            self.register_buffer('edge_attr', edge_attr)
        else:
            self.edge_attr = None
        
        # Encoder GRU
        self.encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=gru_num_layers,
            dropout=dropout if gru_num_layers > 1 else 0,
            batch_first=False
        )
        
        # GraphNet for spatial encoding
        self.gcn = GraphNet(
            node_input_size=hidden_size,
            edge_input_size=1,
            global_input_size=hidden_size,
            hidden_size=gn_hidden_size,
            updated_node_size=gn_updated_node_size,
            updated_edge_size=gn_updated_edge_size,
            updated_global_size=gn_updated_global_size,
            node_output_size=hidden_size,
            gn_layer_num=gn_layer_num,
            activation='ReLU',
            dropout=dropout
        )
        
        # Decoder GRU (input: features, hidden: encoder_hidden + graph_encoding)
        self.decoder = nn.GRU(
            input_size=input_dim,
            hidden_size=2 * hidden_size,  # doubled for concatenated hidden state
            num_layers=gru_num_layers,
            dropout=dropout if gru_num_layers > 1 else 0,
            batch_first=False
        )
        
        # Output projection
        self.out_net = nn.Linear(2 * hidden_size, output_dim)

    def _compute_sampling_threshold(self, batches_seen):
        """Compute sampling threshold for curriculum learning."""
        if self.cl_decay_steps == 0:
            return 0
        else:
            return self.cl_decay_steps / (
                self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def forward(
        self, 
        history_data: torch.Tensor, 
        future_data: torch.Tensor = None, 
        batch_seen: int = 0, 
        epoch: int = 0, 
        train: bool = True, 
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass of GRUSeq2SeqWithGraphNet.
        
        Args:
            history_data: Historical input data [B, L, N, C]
            future_data: Future target data [B, L, N, C] (for teacher forcing)
            batch_seen: Number of batches seen (for curriculum learning)
            epoch: Current epoch
            train: Whether in training mode
            **kwargs: Additional arguments (edge_index, edge_attr can be passed here)
        
        Returns:
            Predictions with shape [B, horizon, N, output_dim]
        """
        # Get graph structure from kwargs or use stored values
        edge_index = kwargs.get('edge_index', self.edge_index)
        edge_attr = kwargs.get('edge_attr', self.edge_attr)
        
        batch_size, seq_len, num_nodes, input_dim = history_data.shape
        
        # Reshape for GRU: [B, L, N, C] -> [L, B*N, C]
        x_input = history_data.permute(1, 0, 2, 3).reshape(seq_len, batch_size * num_nodes, input_dim)
        
        # Encode
        _, h_encode = self.encoder(x_input)  # h_encode: [num_layers, B*N, hidden_size]
        encoder_h = h_encode
        
        # Apply GraphNet for spatial encoding
        # Reshape: [num_layers, B*N, hidden_size] -> [N, B, num_layers, hidden_size]
        graph_input = h_encode.view(self.gru_num_layers, batch_size, num_nodes, self.hidden_size)
        graph_input = graph_input.permute(2, 1, 0, 3)  # [N, B, L, F]
        
        # Prepare edge_attr for GraphNet
        if edge_attr is not None:
            edge_attr_expanded = edge_attr.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        else:
            # Create dummy edge attributes if not provided
            num_edges = edge_index.shape[1] if edge_index is not None else 0
            edge_attr_expanded = torch.ones(num_edges, 1, 1, 1, device=history_data.device)
        
        # GraphNet forward
        graph_encoding = self.gcn(graph_input, edge_index, edge_attr_expanded)  # [N, B, L, F]
        
        # Reshape back: [N, B, L, F] -> [L, B*N, F]
        graph_encoding = graph_encoding.permute(2, 1, 0, 3).reshape(
            self.gru_num_layers, batch_size * num_nodes, self.hidden_size
        )
        
        # Concatenate encoder hidden with graph encoding
        h_decode = torch.cat([h_encode, graph_encoding], dim=-1)  # [L, B*N, 2*hidden]
        
        # Decode
        if self.training and not self.use_curriculum_learning and future_data is not None:
            # Teacher forcing without curriculum learning
            y_input = future_data[..., :self.input_dim]  # [B, horizon, N, C]
            y_input = y_input.permute(1, 0, 2, 3).reshape(self.horizon, batch_size * num_nodes, -1)
            # Prepend last input
            decoder_input = torch.cat([x_input[-1:], y_input[:-1]], dim=0)
            out_hidden, _ = self.decoder(decoder_input, h_decode)
            out = self.out_net(out_hidden)  # [horizon, B*N, output_dim]
            out = out.view(self.horizon, batch_size, num_nodes, self.output_dim)
            out = out.permute(1, 0, 2, 3)  # [B, horizon, N, output_dim]
        else:
            # Autoregressive decoding (with optional curriculum learning)
            last_input = x_input[-1:]  # [1, B*N, C]
            last_hidden = h_decode
            out_steps = []
            
            # Prepare future data for curriculum learning
            if future_data is not None:
                y_gt = future_data[..., :self.input_dim]
                y_gt = y_gt.permute(1, 0, 2, 3).reshape(self.horizon, batch_size * num_nodes, -1)
            
            for t in range(self.horizon):
                out_hidden, last_hidden = self.decoder(last_input, last_hidden)
                out = self.out_net(out_hidden)  # [1, B*N, output_dim]
                out_steps.append(out)
                
                # Prepare next input
                if self.training and self.use_curriculum_learning and future_data is not None:
                    p_gt = self._compute_sampling_threshold(batch_seen)
                    if np.random.uniform(0, 1) < p_gt:
                        # Use ground truth
                        last_input = y_gt[t:t+1]
                    else:
                        # Use model prediction (pad to input_dim if needed)
                        if out.shape[-1] < self.input_dim:
                            padding = torch.zeros(
                                1, batch_size * num_nodes, self.input_dim - out.shape[-1],
                                device=out.device
                            )
                            last_input = torch.cat([out, padding], dim=-1)
                        else:
                            last_input = out[..., :self.input_dim]
                else:
                    # Use model prediction
                    if out.shape[-1] < self.input_dim:
                        padding = torch.zeros(
                            1, batch_size * num_nodes, self.input_dim - out.shape[-1],
                            device=out.device
                        )
                        last_input = torch.cat([out, padding], dim=-1)
                    else:
                        last_input = out[..., :self.input_dim]
            
            out = torch.cat(out_steps, dim=0)  # [horizon, B*N, output_dim]
            out = out.view(self.horizon, batch_size, num_nodes, self.output_dim)
            out = out.permute(1, 0, 2, 3)  # [B, horizon, N, output_dim]
        
        return out

