"""
Flux Alignment Losses

This module implements neural network-free loss functions for aligning reaction fluxes
with gene expression levels, designed to be stable for batch_size=1 training.

Key components:
1. Listwise ranking loss using KL divergence between soft labels and flux distributions
2. Pairwise hinge loss for top-vs-bottom reaction comparisons  
3. Robust normalization schemes for numerical stability

Mathematical formulation:
- Gene soft labels: q = softmax(beta * zscore(log1p(phi)))
- Flux distribution: p = softmax(zscore(log1p(|v|)) * sign(v) / tau)  
- Loss: KL(q || p) with beta ∈ [1,5], tau ∈ [0.3,1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings


class ListwiseFluxRankingLoss(nn.Module):
    """
    Listwise ranking loss for enzyme-flux alignment with biological confidence bias.
    
    This loss encourages correlation between enzyme scores (biology) and reaction fluxes
    (model predictions) using asymmetric soft ranking distributions. The formulation
    places higher confidence in enzyme scores (biological ground truth) compared to 
    flux predictions (model outputs).
    
    Key asymmetry: Enzyme side uses sharper, more confident distributions while
    flux side uses softer, more flexible distributions to accommodate learning.
    
    Args:
        enzyme_confidence: Sharpness for enzyme score distribution (default: 3.0, higher = more confident)
        flux_flexibility: Temperature for flux distribution (default: 0.8, higher = more flexible)
        min_reactions: Minimum reactions needed for loss computation (default: 3)
        use_absolute_flux: Whether to use |v| for ranking (default: True)
        flux_sign_weighting: Whether to weight by flux sign (default: False, simplified)
    """
    
    def __init__(self, enzyme_confidence=3.0, flux_flexibility=0.8, min_reactions=3, 
                 use_absolute_flux=True, flux_sign_weighting=False):
        super().__init__()
        
        # Renamed parameters to reflect biological interpretation
        self.enzyme_confidence = enzyme_confidence  # Higher = more confident in biology
        self.flux_flexibility = flux_flexibility    # Higher = more forgiving of model errors
        self.min_reactions = min_reactions
        self.use_absolute_flux = use_absolute_flux
        self.flux_sign_weighting = flux_sign_weighting  # Simplified default
        
        # Debug frequency control
        self._debug_call_count = 0
        
        # For monitoring loss components
        self.last_loss_info = {}
    
    def _safe_zscore(self, x, dim=-1, eps=1e-8):
        """
        Compute z-score with numerical stability.
        Handles cases where std is zero or very small.
        """
        mean = torch.mean(x, dim=dim, keepdim=True)
        std = torch.std(x, dim=dim, keepdim=True, unbiased=False)
        std = torch.clamp(std, min=eps)
        return (x - mean) / std
    
    def _compute_gene_expression_scores(self, gene_expressions, reaction_indices, data):
        """
        Compute gene expression scores for reactions using cached gene rules.
        
        Args:
            gene_expressions: Gene expression tensor [batch_size, num_nodes, gene_dim]
            reaction_indices: Indices of reaction nodes
            data: Data object with gene rule information
            
        Returns:
            gene_scores: Expression scores per reaction [batch_size, num_reactions]
        """
        # Handle single sample
        is_single_sample = (gene_expressions.dim() == 2)
        if is_single_sample:
            gene_expressions = gene_expressions.unsqueeze(0)
        
        batch_size = gene_expressions.size(0)
        
        # Try to use optimized tensor operations if available
        if (hasattr(data, 'reaction_gene_indices') and 
            hasattr(data, 'reaction_gene_operations') and
            hasattr(data, 'reaction_gene_masks')):
            
            gene_scores = self._compute_gene_scores_optimized(
                gene_expressions, reaction_indices, data
            )
        else:
            # Fallback to original gene rule evaluation
            gene_scores = self._compute_gene_scores_fallback(
                gene_expressions, reaction_indices, data
            )
        
        if is_single_sample:
            gene_scores = gene_scores.squeeze(0)
        
        return gene_scores
    
    def _compute_gene_scores_optimized(self, gene_expressions, reaction_indices, data):
        """Use optimized tensor operations for gene score computation."""
        batch_size = gene_expressions.size(0)
        device = gene_expressions.device
        
        # Get pre-computed tensors
        gene_indices = data.reaction_gene_indices.to(device)  # [num_reactions, max_genes]
        operations = data.reaction_gene_operations.to(device)  # [num_reactions]
        masks = data.reaction_gene_masks.to(device)  # [num_reactions, max_genes]
        reaction_nodes = data.reaction_node_indices.to(device)  # [num_reactions]
        
        # Extract reaction node features [batch_size, num_reactions, gene_dim]
        if isinstance(reaction_indices, list):
            rxn_idx = torch.tensor(reaction_indices, device=device)
        else:
            rxn_idx = reaction_indices
        
        reaction_features = gene_expressions[:, rxn_idx, :]  # [B, R, gene_dim]
        
        # Use the optimized batched computation from physics loss
        gene_scores = self._compute_gene_scores_batched_optimized(
            reaction_features, gene_indices, operations, masks, device
        )
        
        return gene_scores  # [B, R]
    
    def _compute_gene_scores_batched_optimized(self, reaction_features, 
                                             gene_indices, operations, masks, device):
        """
        Ultra-fast batched gene score computation (adapted from physics loss).
        """
        B, R, D = reaction_features.shape
        G = gene_indices.shape[1]
        
        # Expand indices along batch dimension
        gene_idx_exp = gene_indices.unsqueeze(0).expand(B, -1, -1)  # [B, R, G]
        masks_exp = masks.unsqueeze(0).expand(B, -1, -1)  # [B, R, G]
        
        # Safe gather: clamp negative indices to 0
        safe_idx = torch.clamp(gene_idx_exp, min=0)
        gathered = torch.take_along_dim(
            reaction_features.unsqueeze(2).expand(-1, -1, G, -1),
            safe_idx.unsqueeze(-1),
            dim=3
        ).squeeze(-1)  # [B, R, G]
        
        # Threshold mask
        thresh_mask = masks_exp & (gathered > 1e-6)
        
        # Prepare for aggregation operations
        gathered_min = torch.where(thresh_mask, gathered, torch.full_like(gathered, float('inf')))
        gathered_max = torch.where(thresh_mask, gathered, torch.full_like(gathered, float('-inf')))
        gathered_mean = torch.where(thresh_mask, gathered, torch.zeros_like(gathered))
        
        # Valid gene counts
        valid_counts = thresh_mask.sum(dim=2)  # [B, R]
        
        # Compute operations
        mean_scores = gathered_mean.sum(dim=2) / torch.clamp(valid_counts, min=1)
        min_scores = gathered_min.min(dim=2).values
        max_scores = gathered_max.max(dim=2).values
        
        # Assemble final scores
        final_scores = torch.empty_like(mean_scores)
        op_mean = operations == 0
        op_min = operations == 1
        op_max = operations == 2
        
        final_scores[:, op_mean] = mean_scores[:, op_mean]
        final_scores[:, op_min] = min_scores[:, op_min]
        final_scores[:, op_max] = max_scores[:, op_max]
        
        # Mark reactions with no valid genes as NaN
        final_scores = final_scores.masked_fill(valid_counts == 0, float('nan'))
        
        return final_scores
    
    def _compute_gene_scores_fallback(self, gene_expressions, reaction_indices, data):
        """Fallback gene score computation using original methods."""
        # This would implement the original gene rule evaluation logic
        # For now, use simple mean of reaction node gene features as fallback
        batch_size = gene_expressions.size(0)
        
        if isinstance(reaction_indices, list):
            rxn_idx = torch.tensor(reaction_indices, device=gene_expressions.device)
        else:
            rxn_idx = reaction_indices
        
        # Simple fallback: mean gene expression at reaction nodes
        reaction_features = gene_expressions[:, rxn_idx, :]  # [B, R, gene_dim]
        gene_scores = torch.mean(reaction_features, dim=-1)  # [B, R]
        
        return gene_scores
    
    def forward(self, fluxes, gene_expressions, reaction_indices, data):
        """
        Compute listwise ranking loss between gene expressions and fluxes.
        
        Args:
            fluxes: Predicted reaction fluxes [batch_size, num_reactions] or [num_reactions]
            gene_expressions: Gene expressions [batch_size, num_nodes, gene_dim] or [num_nodes, gene_dim]  
            reaction_indices: Indices of reaction nodes
            data: Data object with gene rule information
            
        Returns:
            ranking_loss: KL divergence between gene and flux distributions
        """
        # Handle single sample case
        is_single_sample = (fluxes.dim() == 1)
        if is_single_sample:
            fluxes = fluxes.unsqueeze(0)
            if gene_expressions.dim() == 2:
                gene_expressions = gene_expressions.unsqueeze(0)
        
        batch_size = fluxes.size(0)
        num_reactions = fluxes.size(1)
        
        # Check if we have enough reactions for meaningful ranking
        if num_reactions < self.min_reactions:
            print(f"WARNING: ListwiseFluxRankingLoss - not enough reactions ({num_reactions} < {self.min_reactions})")
            return torch.tensor(0.0, device=fluxes.device, requires_grad=True)
        
        # Compute gene expression scores for reactions
        gene_scores = self._compute_gene_expression_scores(
            gene_expressions, reaction_indices, data
        )
        
        # Debug: Check gene score computation (only every 10 calls)
        self._debug_call_count += 1
        if self._debug_call_count % 10 == 1:  # Print on 1st, 11th, 21st, etc.
            if gene_scores.dim() == 1:
                gene_scores_debug = gene_scores
            else:
                gene_scores_debug = gene_scores[0] if batch_size > 0 else gene_scores
            
            nan_count = torch.isnan(gene_scores_debug).sum().item()
            valid_count = (~torch.isnan(gene_scores_debug)).sum().item()
            print(f"DEBUG: Gene scores - Valid: {valid_count}, NaN: {nan_count}, Total: {len(gene_scores_debug)}")
            
            if valid_count > 0:
                valid_scores = gene_scores_debug[~torch.isnan(gene_scores_debug)]
                print(f"DEBUG: Valid gene score range: [{valid_scores.min().item():.4f}, {valid_scores.max().item():.4f}]")
            else:
                print(f"WARNING: All gene scores are NaN!")
        
        # Handle batch dimension for gene scores
        if gene_scores.dim() == 1:
            gene_scores = gene_scores.unsqueeze(0)
        
        # Compute ranking loss for each sample
        losses = []
        valid_samples = 0
        
        for b in range(batch_size):
            flux_b = fluxes[b]  # [R]
            gene_b = gene_scores[b]  # [R]
            
            # Filter out invalid gene scores (NaN values)
            valid_mask = ~torch.isnan(gene_b)
            valid_count = valid_mask.sum().item()
            
            if valid_count < self.min_reactions:
                # Not enough valid reactions for this sample
                if b == 0:  # Only print for first sample to avoid spam
                    print(f"WARNING: ListwiseFluxRankingLoss - not enough valid reactions ({valid_count} < {self.min_reactions}) for sample {b}")
                continue
            
            flux_valid = flux_b[valid_mask]
            gene_valid = gene_b[valid_mask]
            
            # ENZYME SIDE: High confidence biological ground truth
            # Sharp distribution that strongly reflects gene rule-based enzyme capacity
            gene_log = torch.log1p(torch.clamp(gene_valid, min=0))
            gene_norm = self._safe_zscore(gene_log)
            q = F.softmax(self.enzyme_confidence * gene_norm, dim=0)  # Sharp, confident
            
            # FLUX SIDE: Flexible model predictions that should learn to match biology
            # Softer distribution that allows model to gradually align with enzyme expectations
            if self.use_absolute_flux:
                flux_magnitude = torch.log1p(torch.abs(flux_valid))
            else:
                flux_magnitude = torch.log1p(torch.clamp(flux_valid, min=0))
            
            flux_norm = self._safe_zscore(flux_magnitude)
            
            if self.flux_sign_weighting and self.use_absolute_flux:
                # Optional: Weight by flux sign for directional information
                flux_norm = flux_norm * torch.sign(flux_valid)
            
            p = F.softmax(flux_norm / self.flux_flexibility, dim=0)  # Soft, flexible
            
            # Compute KL divergence: KL(q || p) = sum(q * log(q/p))
            # Add small epsilon to avoid log(0)
            eps = 1e-8
            q_safe = torch.clamp(q, min=eps)
            p_safe = torch.clamp(p, min=eps)
            
            kl_loss = torch.sum(q_safe * torch.log(q_safe / p_safe))
            losses.append(kl_loss)
            valid_samples += 1
        
        # Store loss information for monitoring
        self.last_loss_info = {
            'valid_samples': valid_samples,
            'total_samples': batch_size,
            'avg_reactions_per_sample': num_reactions
        }
        
        if valid_samples > 0:
            final_loss = torch.stack(losses).mean()
        else:
            print(f"WARNING: ListwiseFluxRankingLoss - no valid samples ({valid_samples}/{batch_size})")
            final_loss = torch.tensor(0.0, device=fluxes.device, requires_grad=True)
        
        return final_loss


class PairwiseFluxHingeLoss(nn.Module):
    """
    Pairwise hinge loss for flux-gene alignment.
    
    This loss compares top-vs-bottom reaction pairs based on gene expression
    and encourages fluxes to follow the same ranking.
    
    Args:
        margin: Hinge loss margin (default: 1.0)
        top_k: Number of top/bottom pairs to compare (default: 3)
        min_reactions: Minimum reactions needed (default: 6)
    """
    
    def __init__(self, margin=1.0, top_k=3, min_reactions=6):
        super().__init__()
        self.margin = margin
        self.top_k = top_k
        self.min_reactions = min_reactions
    
    def forward(self, fluxes, gene_scores):
        """
        Compute pairwise hinge loss between top and bottom reactions.
        
        Args:
            fluxes: Reaction fluxes [batch_size, R] or [R]
            gene_scores: Gene expression scores [batch_size, R] or [R]
            
        Returns:
            hinge_loss: Pairwise hinge loss
        """
        # Handle single sample
        is_single_sample = (fluxes.dim() == 1)
        if is_single_sample:
            fluxes = fluxes.unsqueeze(0)
            gene_scores = gene_scores.unsqueeze(0)
        
        batch_size = fluxes.size(0)
        num_reactions = fluxes.size(1)
        
        if num_reactions < self.min_reactions:
            return torch.tensor(0.0, device=fluxes.device, requires_grad=True)
        
        losses = []
        
        for b in range(batch_size):
            flux_b = fluxes[b]
            gene_b = gene_scores[b]
            
            # Filter valid reactions
            valid_mask = ~torch.isnan(gene_b)
            if valid_mask.sum() < self.min_reactions:
                continue
            
            flux_valid = flux_b[valid_mask]
            gene_valid = gene_b[valid_mask]
            
            # Find top-k and bottom-k reactions by gene expression
            _, gene_indices = torch.sort(gene_valid, descending=True)
            
            top_k = min(self.top_k, len(gene_indices) // 3)
            if top_k < 1:
                continue
            
            top_indices = gene_indices[:top_k]
            bottom_indices = gene_indices[-top_k:]
            
            # Get corresponding fluxes
            top_fluxes = flux_valid[top_indices]  # Should be high
            bottom_fluxes = flux_valid[bottom_indices]  # Should be low
            
            # Compute pairwise hinge loss: max(0, margin - (top_flux - bottom_flux))
            pairwise_losses = []
            for i in range(top_k):
                for j in range(top_k):
                    diff = top_fluxes[i] - bottom_fluxes[j]
                    hinge = torch.clamp(self.margin - diff, min=0)
                    pairwise_losses.append(hinge)
            
            if pairwise_losses:
                losses.append(torch.stack(pairwise_losses).mean())
        
        if losses:
            return torch.stack(losses).mean()
        else:
            return torch.tensor(0.0, device=fluxes.device, requires_grad=True)


class FluxAlignmentLoss(nn.Module):
    """
    Combined flux alignment loss using listwise ranking and optional pairwise hinge.
    
    This loss ensures that reaction fluxes are aligned with gene expression patterns
    without requiring additional neural network parameters.
    
    Args:
        use_listwise: Whether to use listwise ranking loss (default: True)
        use_pairwise: Whether to use pairwise hinge loss (default: False)
        listwise_weight: Weight for listwise loss (default: 1.0)
        pairwise_weight: Weight for pairwise loss (default: 0.1)
        **kwargs: Arguments passed to individual loss components
    """
    
    def __init__(self, use_listwise=True, use_pairwise=False,
                 listwise_weight=1.0, pairwise_weight=0.1, **kwargs):
        super().__init__()
        
        self.use_listwise = use_listwise
        self.use_pairwise = use_pairwise
        self.listwise_weight = listwise_weight
        self.pairwise_weight = pairwise_weight
        
        if use_listwise:
            # Default: High confidence in enzyme scores, flexible on flux predictions
            self.listwise_loss = ListwiseFluxRankingLoss(**kwargs)
        
        if use_pairwise:
            self.pairwise_loss = PairwiseFluxHingeLoss(**kwargs)
    
    def forward(self, fluxes, gene_expressions, reaction_indices, data):
        """
        Compute combined flux alignment loss.
        
        Args:
            fluxes: Predicted reaction fluxes
            gene_expressions: Gene expression data
            reaction_indices: Reaction node indices
            data: Data object with gene rule information
            
        Returns:
            total_loss: Combined alignment loss
        """
        total_loss = torch.tensor(0.0, device=fluxes.device, requires_grad=True)
        
        if self.use_listwise:
            listwise = self.listwise_loss(fluxes, gene_expressions, reaction_indices, data)
            total_loss = total_loss + self.listwise_weight * listwise
        
        if self.use_pairwise:
            # Compute gene scores for pairwise loss (reuse from listwise if available)
            if hasattr(self, 'listwise_loss') and hasattr(self.listwise_loss, 'last_gene_scores'):
                gene_scores = self.listwise_loss.last_gene_scores
            else:
                # Compute gene scores independently
                gene_scores = self.listwise_loss._compute_gene_expression_scores(
                    gene_expressions, reaction_indices, data
                )
            
            pairwise = self.pairwise_loss(fluxes, gene_scores)
            total_loss = total_loss + self.pairwise_weight * pairwise
        
        return total_loss