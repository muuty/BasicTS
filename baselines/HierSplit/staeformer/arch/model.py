# #!/usr/bin/env python3
# """STAEformer Hierarchical Split Learning 통합 모델"""

# from typing import Dict, List, Optional, Tuple
# import torch
# import torch.nn as nn

# from .client import STAEformerHierClientModel
# from .server import STAEformerHierServerModel


# class STAEformerHierModel(nn.Module):
#     def __init__(
#         self,
#         num_clients: int,
#         client_nodes_list: List[List[int]],
#         client_model_params: Dict,
#         server_model_params: Dict,
#         num_tokens: int,
#         pooling_method: str = 'attention',
#         subgraph_adj_list: Optional[List[torch.Tensor]] = None,
#     ):
#         super().__init__()
        
#         self.num_clients = num_clients
#         self.client_nodes_list = client_nodes_list
#         self.num_tokens = num_tokens
        
#         # 클라이언트 모델들 생성
#         self.client_models = nn.ModuleList()
#         for client_id, client_nodes in enumerate(client_nodes_list):
#             params = client_model_params.copy()
#             params['num_nodes'] = len(client_nodes)
#             params['num_tokens'] = num_tokens
#             params['pooling_method'] = pooling_method
            
#             if subgraph_adj_list is not None and len(subgraph_adj_list) > client_id:
#                 params['subgraph_adj'] = subgraph_adj_list[client_id]
#             else:
#                 params['subgraph_adj'] = None

#             client_model = STAEformerHierClientModel(**params)
#             self.client_models.append(client_model)
#         # 서버 모델 생성
#         self.server_model = STAEformerHierServerModel(**server_model_params)
        
#         # 통신 비용 추적
#         self.total_forward_bytes = 0
#         self.total_backward_bytes = 0
    
#     def forward(
#         self,
#         history_data: torch.Tensor,
#         future_data: Optional[torch.Tensor] = None,
#         batch_seen: int = 0,
#         epoch: int = 0,
#         train: bool = True,
#         **kwargs
#     ) -> Dict:
#         """
#         Forward logic:
#         1. Client: Local Attention -> Pooling
#         2. Server: Global Attention (on pooled tokens)
#         3. Client: Expansion (Cross Attn) -> Residual -> Projection
#         """
#         B, L_in, N, _ = history_data.shape
#         L_out = self.client_models[0].out_steps
#         device = history_data.device
        
#         # -----------------------------------------------------------
#         # Phase 1: Client Local Process & Pooling
#         # -----------------------------------------------------------
#         client_spatial_contexts = {} # Expansion때 쓸 Local 정보 저장
#         client_pooled_outputs = []   # 서버로 보낼 압축 정보
        
#         for client_id in range(self.num_clients):
#             client_model = self.client_models[client_id]
#             nodes = self.client_nodes_list[client_id]
#             client_history = history_data[:, :, nodes, :]
            
#             # forward_temporal_spatial_pool -> (Local_Features, Pooled_Features)
#             spatial_feat, pooled_feat = client_model.forward_temporal_spatial_pool(client_history)
            
#             client_spatial_contexts[client_id] = spatial_feat
#             client_pooled_outputs.append(pooled_feat)
        
#         # -----------------------------------------------------------
#         # Phase 2: Aggregation & Send to Server
#         # -----------------------------------------------------------
#         # pooled_feat shape: (B, T, K, D)
#         # all_pooled shape: (B, T, Total_K, D) where Total_K = num_clients * K
#         all_pooled = torch.cat(client_pooled_outputs, dim=2)
        
#         fwd_c2s = all_pooled.numel() * 4
        
#         if train and torch.is_grad_enabled():
#             if all_pooled.requires_grad:
#                 all_pooled.retain_grad()
        
#         # -----------------------------------------------------------
#         # Phase 3: Server Global Process
#         # -----------------------------------------------------------
#         server_features = self.server_model(all_pooled) # (B, T, Total_K, D)
        
#         fwd_s2c = server_features.numel() * 4
        
#         if train and torch.is_grad_enabled():
#             if server_features.requires_grad:
#                 server_features.retain_grad()
                
#         # -----------------------------------------------------------
#         # Phase 4: Distribution & Client Expansion/Projection
#         # -----------------------------------------------------------
#         # 서버 출력을 클라이언트별 토큰 개수(K)만큼 다시 자름
#         # (여기서는 모든 클라이언트가 동일한 K를 가진다고 가정. 다르면 split_size_or_sections에 리스트 전달)
#         server_features_split = torch.split(server_features, self.num_tokens, dim=2)
        
#         pred_all = torch.zeros(B, L_out, N, self.client_models[0].output_dim, device=device)
        
#         for client_id in range(self.num_clients):
#             client_model = self.client_models[client_id]
#             nodes = self.client_nodes_list[client_id]
            
#             # 내 몫의 Global Token 가져오기
#             my_server_feat = server_features_split[client_id] # (B, T, K, D)
#             # 내 Local Context 가져오기
#             my_local_feat = client_spatial_contexts[client_id] # (B, T, N_sub, D)
            
#             # Expansion & Mixing (Cross Attention + Residual)
#             mixed_features = client_model.forward_expansion(my_server_feat, my_local_feat)
            
#             # Final Projection
#             final_pred = client_model.forward_projection(mixed_features)
            
#             # 결과 저장 (Pre-allocate & Assign)
#             pred_all[:, :, nodes, :] = final_pred
        
#         forward_total = fwd_c2s + fwd_s2c
#         if train and torch.is_grad_enabled():
#             self.total_forward_bytes += forward_total
        
#         return {
#             'prediction': pred_all,
#             'all_pooled_features': all_pooled,
#             'server_features': server_features,
#             'forward_communication': {
#                 'client_to_server_bytes': fwd_c2s,
#                 'server_to_client_bytes': fwd_s2c,
#                 'total_bytes': forward_total
#             }
#         }
        
#     def compute_backward_comm(self, forward_return: Dict) -> Dict[str, int]:
#         """Backward 통신량 (C <- S <- C)"""
#         b_c2s = 0
#         b_s2c = 0
        
#         # 1. Client -> Server: server_features의 Gradient
#         # (Client Expansion Layer의 역전파 결과)
#         server_features = forward_return.get('server_features')
#         if server_features is not None and server_features.grad is not None:
#             b_c2s = server_features.grad.numel() * server_features.grad.element_size()
            
#         # 2. Server -> Client: all_pooled_features의 Gradient
#         # (Server Global Attention의 역전파 결과)
#         all_pooled = forward_return.get('all_pooled_features')
#         if all_pooled is not None and all_pooled.grad is not None:
#             b_s2c = all_pooled.grad.numel() * all_pooled.grad.element_size()
            
#         self.total_backward_bytes += (b_c2s + b_s2c)
#         return {'client_to_server': b_c2s, 'server_to_client': b_s2c}

#     def get_communication_stats(self) -> Dict[str, float]:
#         return {
#             'total_forward_mb': self.total_forward_bytes / (1024 * 1024),
#             'total_backward_mb': self.total_backward_bytes / (1024 * 1024),
#             'total_mb': (self.total_forward_bytes + self.total_backward_bytes) / (1024 * 1024),
#         }
    
#     def get_client_models(self) -> List[nn.Module]:
#         return list(self.client_models)
    
#     def get_server_model(self) -> nn.Module:
#         return self.server_model