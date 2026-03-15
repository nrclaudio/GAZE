"""Batch utilities for consistent tensor handling across the pipeline."""

import torch
from torch_geometric.data import Data
from typing import Optional


def standardize_batch_data(data: Data) -> Data:
    """Standardize a PyG Data object to always have batch dimensions.

    Ensures all node features have shape ``[batch_size, num_nodes, feature_dim]``
    and all targets / masks have shape ``[batch_size, num_nodes]``.
    """
    if not hasattr(data, "batch_size"):
        data.batch_size = 1

        if hasattr(data, "x_embedding") and data.x_embedding is not None and data.x_embedding.dim() == 2:
            data.x_embedding = data.x_embedding.unsqueeze(0)
        if hasattr(data, "x_gene") and data.x_gene is not None and data.x_gene.dim() == 2:
            data.x_gene = data.x_gene.unsqueeze(0)
        if hasattr(data, "x_smiles") and data.x_smiles is not None and data.x_smiles.dim() == 2:
            data.x_smiles = data.x_smiles.unsqueeze(0)
        if hasattr(data, "x") and data.x is not None and data.x.dim() == 2:
            data.x = data.x.unsqueeze(0)
        if hasattr(data, "y") and data.y is not None and data.y.dim() == 1:
            data.y = data.y.unsqueeze(0)
        if hasattr(data, "mask") and data.mask is not None and data.mask.dim() == 1:
            data.mask = data.mask.unsqueeze(0)

    validate_batch_shapes(data)
    return data


def validate_batch_shapes(data: Data) -> None:
    """Validate that all tensors have consistent batch dimensions."""
    if not hasattr(data, "batch_size"):
        return
    batch_size = data.batch_size

    for attr in ("x_embedding", "x_gene", "x_smiles"):
        t = getattr(data, attr, None)
        if t is not None:
            assert t.dim() == 3, f"{attr} should be 3D, got {t.dim()}D"
            assert t.size(0) == batch_size, f"{attr} batch size mismatch: {t.size(0)} != {batch_size}"

    if hasattr(data, "y") and data.y is not None:
        assert data.y.dim() == 2, f"y should be 2D, got {data.y.dim()}D"
        assert data.y.size(0) == batch_size, f"y batch size mismatch"

    if hasattr(data, "mask") and data.mask is not None:
        assert data.mask.dim() == 2, f"mask should be 2D, got {data.mask.dim()}D"
        assert data.mask.size(0) == batch_size, f"mask batch size mismatch"


def get_batch_size(data: Data) -> int:
    """Return batch size from a Data object (defaults to 1)."""
    return getattr(data, "batch_size", 1)


def create_default_features(num_nodes: int, feature_dim: int,
                            batch_size: int, device: torch.device) -> torch.Tensor:
    """Create zero features with proper batch dimension ``[B, N, D]``."""
    return torch.zeros(batch_size, num_nodes, feature_dim, device=device)


def expand_to_batch(tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Expand a tensor to match *batch_size*.

    Handles ``[N, D] -> [B, N, D]`` and ``[N] -> [B, N]``.
    """
    if tensor is None:
        return tensor
    if tensor.dim() == 1:
        return tensor.unsqueeze(0).expand(batch_size, -1)
    elif tensor.dim() == 2:
        return tensor.unsqueeze(0).expand(batch_size, -1, -1)
    elif tensor.dim() == 3 and tensor.size(0) == 1:
        return tensor.expand(batch_size, -1, -1)
    elif tensor.dim() == 3 and tensor.size(0) == batch_size:
        return tensor
    else:
        raise ValueError(f"Cannot expand tensor of shape {tensor.shape} to batch_size={batch_size}")
