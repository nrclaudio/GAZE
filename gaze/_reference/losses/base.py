"""
Base loss functions for metabolite prediction.

This module contains fundamental loss functions used across different model architectures:
- MaskedMSELoss: Basic MSE with masking
- MaskedMetaboliteLoss: More sophisticated masked loss for metabolites
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.batch_utils import ensure_batch_dim


class MaskedMSELoss(nn.Module):
    """MSE loss with masking for missing metabolites"""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, preds, targets, masks):
        # masks: (batch_size, num_metabolites) indicating present metabolites
        loss = F.mse_loss(preds*masks, targets*masks, reduction='none')
        return loss.sum() / masks.sum()


class MaskedMetaboliteLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        
    def forward(self, pred, target, mask):
        """
        Args:
            pred: Model predictions [batch_size, num_nodes]
            target: Target values [batch_size, num_nodes]
            mask: Boolean mask [batch_size, num_nodes]
        """
        # Ensure batched format (add batch dim if needed for compatibility)
        if pred.dim() == 1:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)
            mask = mask.unsqueeze(0)
        
        # Validate shapes
        assert pred.shape == target.shape == mask.shape, \
            f"Shape mismatch: pred {pred.shape}, target {target.shape}, mask {mask.shape}"
        
        # Apply mask and compute loss
        batch_size = pred.size(0)
        total_loss = 0.0
        total_samples = 0
        
        for b in range(batch_size):
            if mask[b].any():
                masked_pred = pred[b][mask[b]]
                masked_target = target[b][mask[b]]
                
                sample_loss = F.mse_loss(masked_pred, masked_target, reduction='mean')
                total_loss += sample_loss
                total_samples += 1
        
        if total_samples > 0:
            return total_loss / total_samples if self.reduction == 'mean' else total_loss
        else:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)


