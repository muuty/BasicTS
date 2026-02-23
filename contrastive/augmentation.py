import torch
from typing import Optional, Union


class BaseGSO:
    """Base Graph Shift Operator wrapper.
    
    Wraps the adjacency matrix and provides a get_value() interface.
    This allows both plain tensors and augmented GSOs to use the same interface.
    Not inheriting from nn.Module to avoid being included in state_dict.
    
    Args:
        original_gso: Graph shift operator (adjacency matrix) with shape [N, N]
    """
    
    def __init__(self, original_gso: torch.Tensor):
        if not isinstance(original_gso, torch.Tensor):
            raise TypeError(f'original_gso must be a torch.Tensor, but got {type(original_gso)}')
        
        if original_gso.dim() != 2:
            raise ValueError(f'original_gso must be 2D tensor [N, N], but got shape {original_gso.shape}')
        
        # Store as attribute (not registered buffer to avoid state_dict issues)
        self.original_gso = original_gso
    
    def get_value(self) -> torch.Tensor:
        """Get the GSO value.
        
        Returns:
            torch.Tensor: GSO with shape [N, N]
        """
        return self.original_gso
    
    def train(self, mode: bool = True) -> 'BaseGSO':
        """Set training mode (no-op for BaseGSO, for interface compatibility).
        
        Args:
            mode: If True, set to training mode. Default: True
        
        Returns:
            BaseGSO: self (for chaining)
        """
        return self
    
    def eval(self) -> 'BaseGSO':
        """Set evaluation mode (no-op for BaseGSO, for interface compatibility).
        
        Returns:
            BaseGSO: self (for chaining)
        """
        return self
    
    def to(self, device) -> 'BaseGSO':
        """Move GSO to device.
        
        Args:
            device: Target device
        
        Returns:
            BaseGSO: self (for chaining)
        """
        self.original_gso = self.original_gso.to(device)
        return self


class AugmentedGSO(BaseGSO):
    """Graph Shift Operator (GSO) with edge drop augmentation.
    
    Extends BaseGSO to apply edge dropout during training.
    The augmentation is applied dynamically each time get_value() is called.
    Use train() and eval() to control whether augmentation is applied.
    
    Args:
        original_gso: Original graph shift operator (adjacency matrix) with shape [N, N]
        edge_drop_rate: Probability of dropping each edge. Default: None (no augmentation)
    """
    
    def __init__(
        self, 
        original_gso: torch.Tensor,
        edge_drop_rate: Optional[float] = None
    ):
        super(AugmentedGSO, self).__init__(original_gso)
        
        if edge_drop_rate is not None and (edge_drop_rate < 0.0 or edge_drop_rate >= 1.0):
            raise ValueError(f'edge_drop_rate must be in [0.0, 1.0), but got {edge_drop_rate}')
        
        self.edge_drop_rate = edge_drop_rate
        self._is_training = True  # Default to training mode
    
    def get_value(self) -> torch.Tensor:
        """Get the GSO value with edge dropout augmentation.
        
        Applies augmentation only when in training mode (via train()/eval()).
        If edge_drop_rate is None or 0.0, returns original GSO.
        
        Returns:
            torch.Tensor: GSO (original or augmented) with shape [N, N]
        """
        
        # If no augmentation is enabled, return original
        if self.edge_drop_rate is None or self.edge_drop_rate == 0.0:
            return self.original_gso
        
        # Only apply augmentation in training mode
        if not self._is_training:
            return self.original_gso
        
        # Apply edge dropout during training
        device = self.original_gso.device
        
        # Create mask: 1 means keep edge, 0 means drop edge
        # Shape: [N, N]
        mask = torch.bernoulli(
            torch.ones_like(self.original_gso) * (1.0 - self.edge_drop_rate)
        ).to(device)
        
        # Apply mask to adjacency matrix
        # Zero out dropped edges
        augmented_gso = self.original_gso * mask
        
        return augmented_gso
    
    def train(self, mode: bool = True) -> 'AugmentedGSO':
        """Set training mode.
        
        Args:
            mode: If True, set to training mode (augmentation enabled). Default: True
        
        Returns:
            AugmentedGSO: self (for chaining)
        """
        self._is_training = mode
        return self
    
    def eval(self) -> 'AugmentedGSO':
        """Set evaluation mode.
        
        Returns:
            AugmentedGSO: self (for chaining)
        """
        return self.train(False)
