import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CrossModalAttention(nn.Module):
    """
    Simplified cross-modal attention mechanism that allows metabolites to attend to reactions
    and vice versa before entering the GNN layers.
    
    Key improvements:
    - Uses ONLY cross-modal attention (no self-attention to reduce complexity)
    - Adds residual connections to preserve original information
    - Helps metabolites leverage reaction context (gene expression patterns)
    """
    
    def __init__(self, reaction_dim, metabolite_dim, hidden_dim, num_heads=4, dropout=0.1, temperature=1.0):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.temperature = temperature
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Query, Key, Value projections for reactions and metabolites
        self.reaction_q = nn.Linear(reaction_dim, hidden_dim)
        self.reaction_k = nn.Linear(reaction_dim, hidden_dim)
        self.reaction_v = nn.Linear(reaction_dim, hidden_dim)
        
        self.metabolite_q = nn.Linear(metabolite_dim, hidden_dim)
        self.metabolite_k = nn.Linear(metabolite_dim, hidden_dim)
        self.metabolite_v = nn.Linear(metabolite_dim, hidden_dim)
        
        # Output projections
        self.reaction_output = nn.Linear(hidden_dim, hidden_dim)
        self.metabolite_output = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

        # Residual projection to align original feature dimensions with hidden_dim
        self.reaction_residual_proj = (
            nn.Linear(reaction_dim, hidden_dim)
            if reaction_dim != hidden_dim else nn.Identity()
        )
        self.metabolite_residual_proj = (
            nn.Linear(metabolite_dim, hidden_dim)
            if metabolite_dim != hidden_dim else nn.Identity()
        )
        
    def forward(self, reaction_features, metabolite_features, reaction_indices, metabolite_indices, mask=None):
        """
        Args:
            reaction_features: [num_reactions, reaction_dim] or [batch_size, num_reactions, reaction_dim]
            metabolite_features: [num_metabolites, metabolite_dim] or [batch_size, num_metabolites, metabolite_dim]
            reaction_indices: indices of reaction nodes in the full graph
            metabolite_indices: indices of metabolite nodes in the full graph
            mask: [batch_size, num_metabolites, num_reactions] boolean mask (1=connected, 0=not connected)
            
        Returns:
            enhanced_reaction_features: [num_reactions, hidden_dim] or [batch_size, num_reactions, hidden_dim]
            enhanced_metabolite_features: [num_metabolites, hidden_dim] or [batch_size, num_metabolites, hidden_dim]
        """
        is_batched = len(reaction_features.shape) == 3
        
        if is_batched:
            batch_size = reaction_features.size(0)
            num_reactions = reaction_features.size(1)
            num_metabolites = metabolite_features.size(1)
        else:
            batch_size = 1
            num_reactions = reaction_features.size(0)
            num_metabolites = metabolite_features.size(0)
            # Add batch dimension for uniform processing
            reaction_features = reaction_features.unsqueeze(0)
            metabolite_features = metabolite_features.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)
        
        # Compute Q, K, V for both node types
        reaction_q = self.reaction_q(reaction_features)  # [B, num_reactions, hidden_dim]
        reaction_k = self.reaction_k(reaction_features)
        reaction_v = self.reaction_v(reaction_features)
        
        metabolite_q = self.metabolite_q(metabolite_features)  # [B, num_metabolites, hidden_dim]
        metabolite_k = self.metabolite_k(metabolite_features)
        metabolite_v = self.metabolite_v(metabolite_features)
        
        # Reshape for multi-head attention: [B, num_heads, seq_len, head_dim]
        reaction_q = self._reshape_for_attention(reaction_q)    # [B, heads, num_reactions, head_dim]
        reaction_k = self._reshape_for_attention(reaction_k)
        reaction_v = self._reshape_for_attention(reaction_v)
        
        metabolite_q = self._reshape_for_attention(metabolite_q)  # [B, heads, num_metabolites, head_dim]
        metabolite_k = self._reshape_for_attention(metabolite_k)
        metabolite_v = self._reshape_for_attention(metabolite_v)
        
        # Cross-modal attention: Metabolites attend to Reactions
        # If mask is provided (M->R), we use it directly
        metabolite_enhanced = self._cross_attention(
            metabolite_q, reaction_k, reaction_v, mask=mask
        )  # [B, heads, num_metabolites, head_dim]
        
        # Cross-modal attention: Reactions attend to Metabolites
        # For R->M, we need to transpose the mask if it exists
        reaction_mask = mask.transpose(-2, -1) if mask is not None else None
        
        reaction_enhanced = self._cross_attention(
            reaction_q, metabolite_k, metabolite_v, mask=reaction_mask
        )  # [B, heads, num_reactions, head_dim]
        
        # Use ONLY cross-modal attention (removed self-attention to reduce complexity)
        metabolite_combined = metabolite_enhanced
        reaction_combined = reaction_enhanced
        
        # Reshape back: [B, seq_len, hidden_dim]
        metabolite_combined = self._reshape_from_attention(metabolite_combined)
        reaction_combined = self._reshape_from_attention(reaction_combined)
        
        # Apply output projections with residual connections
        metabolite_output = self.metabolite_output(metabolite_combined)
        reaction_output = self.reaction_output(reaction_combined)

        # Always provide residual path by projecting base features to hidden_dim
        metabolite_base = self.metabolite_residual_proj(metabolite_features)
        reaction_base = self.reaction_residual_proj(reaction_features)

        enhanced_metabolite_features = metabolite_output + metabolite_base
        enhanced_reaction_features = reaction_output + reaction_base
        
        # Remove batch dimension if input was not batched
        if not is_batched:
            enhanced_metabolite_features = enhanced_metabolite_features.squeeze(0)
            enhanced_reaction_features = enhanced_reaction_features.squeeze(0)
        
        return enhanced_reaction_features, enhanced_metabolite_features
    
    def _reshape_for_attention(self, x):
        """Reshape tensor for multi-head attention: [B, seq_len, hidden_dim] -> [B, heads, seq_len, head_dim]"""
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)  # [B, heads, seq_len, head_dim]
    
    def _reshape_from_attention(self, x):
        """Reshape tensor from multi-head attention: [B, heads, seq_len, head_dim] -> [B, seq_len, hidden_dim]"""
        batch_size, _, seq_len, _ = x.shape
        x = x.transpose(1, 2)  # [B, seq_len, heads, head_dim]
        return x.contiguous().view(batch_size, seq_len, self.hidden_dim)
    
    def _cross_attention(self, q, k, v, mask=None):
        """
        Compute cross-attention between q and k,v
        Args:
            q: [B, heads, seq_len_q, head_dim]
            k: [B, heads, seq_len_k, head_dim]
            v: [B, heads, seq_len_k, head_dim]
            mask: optional attention mask
        """
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.scale * self.temperature)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, v)
        
        return output
