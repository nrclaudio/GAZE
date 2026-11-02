"""
Contrastive learning losses for metabolic networks.

This module provides contrastive learning functions that work with 
batched training for metabolic network models:
- StandardContrastiveLoss: Simple contrastive loss using in-batch negative sampling
- ReactionMetaboliteContrastive: Contrastive learning between reactions and metabolite patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StandardContrastiveLoss(nn.Module):
    """
    Simple contrastive loss using in-batch negative sampling.
    
    This is much simpler than MoCo and works better with batch_size > 1.
    Uses all other samples in the batch as negatives.
    """
    
    def __init__(self, temperature=0.07, normalize=True):
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize
        
    def forward(self, anchor_features, positive_features, batch_mask=None):
        """
        Compute contrastive loss between anchor and positive features.
        
        Args:
            anchor_features: [batch_size, feature_dim] - e.g., reaction representations
            positive_features: [batch_size, feature_dim] - e.g., metabolite representations
            batch_mask: [batch_size] - optional mask for valid samples
            
        Returns:
            contrastive_loss: scalar loss value
        """
        batch_size = anchor_features.size(0)
        
        if batch_size == 1:
            # Can't compute contrastive loss with single sample
            return torch.tensor(0.0, device=anchor_features.device, requires_grad=True)
        
        # Normalize features if requested
        if self.normalize:
            anchor_features = F.normalize(anchor_features, dim=1)
            positive_features = F.normalize(positive_features, dim=1)
        
        # Compute similarity matrix: [batch_size, batch_size]
        logits = torch.mm(anchor_features, positive_features.t()) / self.temperature
        
        # Create labels: diagonal elements are positives
        labels = torch.arange(batch_size, device=anchor_features.device)
        
        # Apply mask if provided
        if batch_mask is not None:
            # Only compute loss for valid samples
            valid_indices = batch_mask.nonzero(as_tuple=True)[0]
            if valid_indices.numel() <= 1:
                return torch.tensor(0.0, device=anchor_features.device, requires_grad=True)
            
            logits = logits.index_select(0, valid_indices).index_select(1, valid_indices)
            labels = torch.arange(valid_indices.numel(), device=anchor_features.device)
        
        # Compute InfoNCE loss
        loss = F.cross_entropy(logits, labels)
        
        return loss


class ReactionMetaboliteContrastive(nn.Module):
    """
    Simplified contrastive learning between reaction representations and metabolite patterns.
    
    Uses reaction embeddings directly and creates simple metabolite representations
    from abundance patterns. No complex projection heads needed.
    """
    
    def __init__(self, reaction_dim, metabolite_dim, temperature=0.07):
        super().__init__()
        
        # Simple metabolite encoder (just normalize and project to reaction dim)
        self.metabolite_encoder = nn.Linear(metabolite_dim, reaction_dim)
        
        self.contrastive_loss = StandardContrastiveLoss(temperature=temperature, normalize=True)
        
    def forward(self, reaction_embeddings, metabolite_values, metabolite_mask, reaction_indices):
        """
        Compute contrastive loss between reactions and metabolite patterns.
        
        Args:
            reaction_embeddings: [batch_size, num_nodes, hidden_dim] - node embeddings from GNN
            metabolite_values: [batch_size, num_nodes] - ground truth metabolite values
            metabolite_mask: [batch_size, num_nodes] - mask for measured metabolites
            reaction_indices: list/tensor of reaction node indices
            
        Returns:
            contrastive_loss: scalar loss value
        """
        batch_size = reaction_embeddings.size(0)
        
        if batch_size == 1:
            return torch.tensor(0.0, device=reaction_embeddings.device, requires_grad=True)
        
        # Convert reaction indices to tensor if needed
        if isinstance(reaction_indices, list):
            reaction_idx_tensor = torch.tensor(reaction_indices, device=reaction_embeddings.device)
        else:
            reaction_idx_tensor = reaction_indices
        
        # Extract reaction representations: [batch_size, num_reactions, hidden_dim]
        batch_reaction_emb = reaction_embeddings[:, reaction_idx_tensor]
        
        # Pool reaction representations (mean pooling): [batch_size, hidden_dim]  
        reaction_repr = batch_reaction_emb.mean(dim=1)
        
        # Create metabolite representations from ground truth values
        # Only use measured metabolites: [batch_size, num_measured_metabolites]
        batch_metabolite_patterns = []
        
        for b in range(batch_size):
            measured_metabolites = metabolite_values[b][metabolite_mask[b]]
            
            if len(measured_metabolites) == 0:
                # No measured metabolites, use zeros
                measured_metabolites = torch.zeros(1, device=metabolite_values.device)
            
            # Pad or truncate to fixed size for consistent batching
            max_metabolites = 50  # Reasonable upper bound
            if len(measured_metabolites) > max_metabolites:
                measured_metabolites = measured_metabolites[:max_metabolites]
            else:
                padding = torch.zeros(max_metabolites - len(measured_metabolites), 
                                    device=metabolite_values.device)
                measured_metabolites = torch.cat([measured_metabolites, padding])
            
            batch_metabolite_patterns.append(measured_metabolites)
        
        # Stack into batch: [batch_size, max_metabolites]
        metabolite_patterns = torch.stack(batch_metabolite_patterns)
        
        # Simple approach: Use reaction embeddings directly, encode metabolite patterns
        reaction_features = reaction_repr                                    # [batch_size, reaction_dim] 
        metabolite_features = self.metabolite_encoder(metabolite_patterns)   # [batch_size, reaction_dim]
        
        # Compute contrastive loss
        contrastive_loss = self.contrastive_loss(reaction_features, metabolite_features)
        
        return contrastive_loss


def create_contrastive_module(hidden_dim, temperature=0.07, **_unused_kwargs):
    """
    Factory function to create contrastive learning module.
    
    Args:
        hidden_dim: Hidden dimension from GNN (reaction embeddings)
        temperature: Temperature parameter for InfoNCE loss
        
    Returns:
        ReactionMetaboliteContrastive module
    """
    return ReactionMetaboliteContrastive(
        reaction_dim=hidden_dim,
        metabolite_dim=50,  # Fixed metabolite pattern size
        temperature=temperature
    )
