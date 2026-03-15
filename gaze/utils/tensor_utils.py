"""Tensor utility functions for safe tensor operations."""

import torch


def check_for_nans(tensor: torch.Tensor, location: str = "unknown location",
                   raise_on_nan: bool = True) -> torch.Tensor:
    """Check if tensor contains NaN values and optionally raise an error.

    Args:
        tensor: PyTorch tensor to check.
        location: Description of where this tensor came from (for debugging).
        raise_on_nan: If True, raise ValueError when NaNs are found.

    Returns:
        The input tensor unchanged (if no NaNs or *raise_on_nan* is False).

    Raises:
        ValueError: If NaNs are found and *raise_on_nan* is True.
    """
    if torch.isnan(tensor).any():
        nan_count = torch.isnan(tensor).sum().item()
        total_elements = tensor.numel()
        error_msg = (
            f"NaN detected in {location}! "
            f"Found {nan_count} NaN values out of {total_elements} total elements. "
            f"Tensor shape: {tensor.shape}, dtype: {tensor.dtype}"
        )
        if raise_on_nan:
            raise ValueError(error_msg)
        else:
            print(f"WARNING: {error_msg}")
    return tensor
