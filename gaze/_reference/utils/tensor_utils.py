"""
Tensor utility functions for safe tensor operations.
"""
import torch


def check_for_nans(tensor, location="unknown location", raise_on_nan=True):
    """
    Check if tensor contains NaN values and optionally raise an error.
    
    Args:
        tensor: PyTorch tensor to check
        location: String description of where this tensor came from (for debugging)
        raise_on_nan: If True, raise ValueError when NaNs are found
        
    Returns:
        tensor: The input tensor (if no NaNs or raise_on_nan=False)
        
    Raises:
        ValueError: If NaNs are found and raise_on_nan=True
    """
    if torch.isnan(tensor).any():
        nan_count = torch.isnan(tensor).sum().item()
        total_elements = tensor.numel()
        
        error_msg = (
            f"NaN detected in {location}! "
            f"Found {nan_count} NaN values out of {total_elements} total elements. "
            f"This indicates numerical instability. "
            f"Tensor shape: {tensor.shape}, dtype: {tensor.dtype}"
        )
        
        if raise_on_nan:
            raise ValueError(error_msg)
        else:
            print(f"WARNING: {error_msg}")
    
    return tensor


def safe_tensor_operation(tensor, operation_name="tensor operation"):
    """
    Wrapper for tensor operations that might produce NaNs.
    
    Args:
        tensor: Input tensor
        operation_name: Description of the operation for error messages
        
    Returns:
        tensor: The input tensor after checking for NaNs
        
    Raises:
        ValueError: If NaNs are detected
    """
    return check_for_nans(tensor, f"after {operation_name}")


def check_tensor_health(tensor, name="tensor"):
    """
    Comprehensive tensor health check.
    
    Args:
        tensor: Tensor to check
        name: Name for debugging output
        
    Returns:
        dict: Health report containing various statistics
    """
    health_report = {
        'name': name,
        'shape': tensor.shape,
        'dtype': tensor.dtype,
        'device': tensor.device,
        'has_nans': torch.isnan(tensor).any().item(),
        'has_infs': torch.isinf(tensor).any().item(),
        'min_value': tensor.min().item(),
        'max_value': tensor.max().item(),
        'mean_value': tensor.mean().item(),
        'std_value': tensor.std().item()
    }
    
    # Check for potential issues
    issues = []
    if health_report['has_nans']:
        nan_count = torch.isnan(tensor).sum().item()
        issues.append(f"{nan_count} NaN values")
    
    if health_report['has_infs']:
        inf_count = torch.isinf(tensor).sum().item()
        issues.append(f"{inf_count} Inf values")
    
    if abs(health_report['max_value']) > 1e6:
        issues.append("Very large values (>1e6)")
    
    if abs(health_report['min_value']) > 1e6:
        issues.append("Very large negative values (<-1e6)")
    
    health_report['issues'] = issues
    health_report['is_healthy'] = len(issues) == 0
    
    return health_report