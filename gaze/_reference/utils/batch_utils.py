"""
Batch utilities for consistent tensor handling across the codebase.

This module provides standardized helpers to ensure all tensors maintain
consistent batch dimensions throughout the pipeline.
"""

import torch
from torch_geometric.data import Data
from typing import Optional, Tuple, Union


def ensure_batch_dim(tensor: torch.Tensor, expected_batch_size: int, 
                    feature_name: str = "tensor") -> torch.Tensor:
    """
    Ensure tensor has proper batch dimension.
    
    Args:
        tensor: Input tensor that may or may not have batch dimension
        expected_batch_size: Expected batch size
        feature_name: Name of the feature for error messages
        
    Returns:
        Tensor with guaranteed batch dimension [batch_size, ...]
    """
    if tensor is None:
        return tensor
        
    # Add batch dimension if missing
    if tensor.dim() >= 1:
        # Check if first dimension matches batch size
        if tensor.size(0) != expected_batch_size:
            # Assume tensor is missing batch dimension
            if expected_batch_size == 1:
                tensor = tensor.unsqueeze(0)
            else:
                raise ValueError(
                    f"{feature_name} has incompatible shape {tensor.shape} "
                    f"for batch_size={expected_batch_size}"
                )
    
    return tensor


def standardize_batch_data(data: Data) -> Data:
    """
    Standardize data object to always have batch dimensions.
    
    This ensures all node features have shape [batch_size, num_nodes, feature_dim]
    and all targets/masks have shape [batch_size, num_nodes].
    
    Args:
        data: PyG Data object that may or may not be batched
        
    Returns:
        Data object with standardized batch dimensions
    """
    # Determine if data is batched
    if not hasattr(data, 'batch_size'):
        # Single sample - add batch_size attribute and batch dimensions
        data.batch_size = 1
        
        # Add batch dimension to node features
        if hasattr(data, 'x_embedding') and data.x_embedding is not None and data.x_embedding.dim() == 2:
            data.x_embedding = data.x_embedding.unsqueeze(0)
        
        if hasattr(data, 'x_gene') and data.x_gene is not None and data.x_gene.dim() == 2:
            data.x_gene = data.x_gene.unsqueeze(0)
            
        if hasattr(data, 'x_smiles') and data.x_smiles is not None and data.x_smiles.dim() == 2:
            data.x_smiles = data.x_smiles.unsqueeze(0)
            
        if hasattr(data, 'x') and data.x is not None and data.x.dim() == 2:
            data.x = data.x.unsqueeze(0)
            
        # Add batch dimension to targets and masks
        if hasattr(data, 'y') and data.y is not None and data.y.dim() == 1:
            data.y = data.y.unsqueeze(0)
            
        if hasattr(data, 'mask') and data.mask is not None and data.mask.dim() == 1:
            data.mask = data.mask.unsqueeze(0)
    
    # Validate expected shapes
    validate_batch_shapes(data)
    
    return data


def validate_batch_shapes(data: Data) -> None:
    """
    Validate that all tensors have consistent batch dimensions.
    
    Args:
        data: PyG Data object to validate
        
    Raises:
        AssertionError: If shapes are inconsistent
    """
    if not hasattr(data, 'batch_size'):
        return  # Not batched, skip validation
        
    batch_size = data.batch_size
    
    # Check node features
    if hasattr(data, 'x_embedding') and data.x_embedding is not None:
        assert data.x_embedding.dim() == 3, \
            f"x_embedding should be 3D, got {data.x_embedding.dim()}D"
        assert data.x_embedding.size(0) == batch_size, \
            f"x_embedding batch size mismatch: {data.x_embedding.size(0)} != {batch_size}"
    
    if hasattr(data, 'x_gene') and data.x_gene is not None:
        assert data.x_gene.dim() == 3, \
            f"x_gene should be 3D, got {data.x_gene.dim()}D"
        assert data.x_gene.size(0) == batch_size, \
            f"x_gene batch size mismatch: {data.x_gene.size(0)} != {batch_size}"
    
    if hasattr(data, 'x_smiles') and data.x_smiles is not None:
        assert data.x_smiles.dim() == 3, \
            f"x_smiles should be 3D, got {data.x_smiles.dim()}D"
        assert data.x_smiles.size(0) == batch_size, \
            f"x_smiles batch size mismatch: {data.x_smiles.size(0)} != {batch_size}"
    
    # Check targets and masks
    if hasattr(data, 'y') and data.y is not None:
        assert data.y.dim() == 2, \
            f"y should be 2D [batch_size, num_nodes], got {data.y.dim()}D"
        assert data.y.size(0) == batch_size, \
            f"y batch size mismatch: {data.y.size(0)} != {batch_size}"
    
    if hasattr(data, 'mask') and data.mask is not None:
        assert data.mask.dim() == 2, \
            f"mask should be 2D [batch_size, num_nodes], got {data.mask.dim()}D"
        assert data.mask.size(0) == batch_size, \
            f"mask batch size mismatch: {data.mask.size(0)} != {batch_size}"


def get_batch_size(data: Data) -> int:
    """
    Get batch size from data object.
    
    Args:
        data: PyG Data object
        
    Returns:
        Batch size (1 if not batched)
    """
    if hasattr(data, 'batch_size'):
        return data.batch_size
    return 1


def split_batch_output(output: torch.Tensor, batch_size: int) -> list:
    """
    Split batched output tensor into list of individual samples.
    
    Args:
        output: Batched output tensor [batch_size, ...]
        batch_size: Number of samples in batch
        
    Returns:
        List of tensors, one per sample
    """
    if output.dim() < 1 or output.size(0) != batch_size:
        raise ValueError(f"Invalid output shape {output.shape} for batch_size={batch_size}")
    
    return [output[i] for i in range(batch_size)]


def expand_to_batch(tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
    """
    Expand a tensor to match batch size.
    
    Useful for shared features that need to be replicated across batch.
    
    Args:
        tensor: Tensor to expand [num_nodes, feature_dim] or [num_nodes]
        batch_size: Target batch size
        
    Returns:
        Expanded tensor [batch_size, num_nodes, feature_dim] or [batch_size, num_nodes]
    """
    if tensor is None:
        return tensor
    
    if tensor.dim() == 1:
        # [num_nodes] -> [batch_size, num_nodes]
        return tensor.unsqueeze(0).expand(batch_size, -1)
    elif tensor.dim() == 2:
        # [num_nodes, feature_dim] -> [batch_size, num_nodes, feature_dim]
        return tensor.unsqueeze(0).expand(batch_size, -1, -1)
    elif tensor.dim() == 3 and tensor.size(0) == 1:
        # [1, num_nodes, feature_dim] -> [batch_size, num_nodes, feature_dim]
        return tensor.expand(batch_size, -1, -1)
    elif tensor.dim() == 3 and tensor.size(0) == batch_size:
        # Already properly batched
        return tensor
    else:
        raise ValueError(f"Cannot expand tensor of shape {tensor.shape} to batch_size={batch_size}")


def apply_mask_batched(predictions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Apply mask to batched predictions.
    
    Args:
        predictions: Batched predictions [batch_size, num_nodes]
        mask: Batched mask [batch_size, num_nodes]
        
    Returns:
        Masked predictions [num_masked_total]
    """
    assert predictions.shape == mask.shape, \
        f"Shape mismatch: predictions {predictions.shape} vs mask {mask.shape}"
    
    # Apply mask and flatten
    return predictions[mask]


def create_default_features(num_nodes: int, feature_dim: int, 
                           batch_size: int, device: torch.device) -> torch.Tensor:
    """
    Create default zero features with proper batch dimension.
    
    Args:
        num_nodes: Number of nodes in graph
        feature_dim: Dimension of features
        batch_size: Batch size
        device: Device to create tensor on
        
    Returns:
        Zero tensor [batch_size, num_nodes, feature_dim]
    """
    return torch.zeros(batch_size, num_nodes, feature_dim, device=device)