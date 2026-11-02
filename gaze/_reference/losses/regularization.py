"""
Regularization loss functions for metabolic models.

This module contains regularization losses used to constrain model parameters:
- parse_gene_rule: Utility function for parsing gene rule strings
"""

import torch
import torch.nn as nn

from ..utils.gene_parsing import parse_gene_rule_simple as parse_gene_rule


class WeightDecayLoss(nn.Module):
    """
    Standard L2 weight decay regularization.
    
    Penalizes large parameter values to prevent overfitting and improve generalization.
    """
    
    def __init__(self, weight=1e-4):
        """
        Args:
            weight: Weight decay strength
        """
        super().__init__()
        self.weight = weight
    
    def forward(self, model_params):
        """
        Compute weight decay loss.
        
        Args:
            model_params: Model parameters to regularize
            
        Returns:
            decay_loss: L2 penalty on parameters
        """
        decay_loss = 0.0
        
        if isinstance(model_params, dict):
            for param in model_params.values():
                if param.requires_grad:
                    decay_loss += torch.sum(param ** 2)
        else:
            for param in model_params:
                if param.requires_grad:
                    decay_loss += torch.sum(param ** 2)
        
        return self.weight * decay_loss


class SparsityRegularization(nn.Module):
    """
    L1 sparsity regularization to encourage sparse solutions.
    
    Useful for flux prediction models where we want to identify
    active vs inactive metabolic pathways.
    """
    
    def __init__(self, weight=1e-3):
        """
        Args:
            weight: Sparsity regularization strength
        """
        super().__init__()
        self.weight = weight
    
    def forward(self, predictions):
        """
        Compute L1 sparsity loss.
        
        Args:
            predictions: Model predictions to encourage sparsity
            
        Returns:
            sparsity_loss: L1 penalty promoting sparse solutions
        """
        return self.weight * torch.sum(torch.abs(predictions))