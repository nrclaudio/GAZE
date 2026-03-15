"""Bidirectional cross-modal attention between reaction and metabolite embeddings."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttention(nn.Module):
    """Cross-modal attention that allows metabolites to attend to reactions
    and vice versa.

    Uses ONLY cross-modal attention (no self-attention) with residual connections
    to preserve original information.
    """

    def __init__(self, reaction_dim: int, metabolite_dim: int, hidden_dim: int,
                 num_heads: int = 4, dropout: float = 0.1, temperature: float = 1.0):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.temperature = temperature

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Q/K/V projections for both node types
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

        # Residual projections to align dimensions with hidden_dim
        self.reaction_residual_proj = (
            nn.Linear(reaction_dim, hidden_dim) if reaction_dim != hidden_dim else nn.Identity()
        )
        self.metabolite_residual_proj = (
            nn.Linear(metabolite_dim, hidden_dim) if metabolite_dim != hidden_dim else nn.Identity()
        )

    def forward(self, reaction_features, metabolite_features,
                reaction_indices, metabolite_indices, mask=None):
        """Compute bidirectional cross-modal attention.

        Args:
            reaction_features: ``[N_r, D_r]`` or ``[B, N_r, D_r]``
            metabolite_features: ``[N_m, D_m]`` or ``[B, N_m, D_m]``
            reaction_indices: Indices of reaction nodes in the full graph.
            metabolite_indices: Indices of metabolite nodes in the full graph.
            mask: ``[B, N_m, N_r]`` boolean mask (1=connected).

        Returns:
            enhanced_reaction_features, enhanced_metabolite_features
        """
        is_batched = reaction_features.dim() == 3

        if not is_batched:
            reaction_features = reaction_features.unsqueeze(0)
            metabolite_features = metabolite_features.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)

        # Q/K/V computation
        reaction_q = self._reshape_for_attention(self.reaction_q(reaction_features))
        reaction_k = self._reshape_for_attention(self.reaction_k(reaction_features))
        reaction_v = self._reshape_for_attention(self.reaction_v(reaction_features))

        metabolite_q = self._reshape_for_attention(self.metabolite_q(metabolite_features))
        metabolite_k = self._reshape_for_attention(self.metabolite_k(metabolite_features))
        metabolite_v = self._reshape_for_attention(self.metabolite_v(metabolite_features))

        # Metabolites attend to Reactions
        metabolite_enhanced = self._cross_attention(metabolite_q, reaction_k, reaction_v, mask=mask)
        # Reactions attend to Metabolites
        reaction_mask = mask.transpose(-2, -1) if mask is not None else None
        reaction_enhanced = self._cross_attention(reaction_q, metabolite_k, metabolite_v, mask=reaction_mask)

        # Reshape back
        metabolite_combined = self._reshape_from_attention(metabolite_enhanced)
        reaction_combined = self._reshape_from_attention(reaction_enhanced)

        # Output projections + residual
        metabolite_output = self.metabolite_output(metabolite_combined)
        reaction_output = self.reaction_output(reaction_combined)

        metabolite_base = self.metabolite_residual_proj(metabolite_features)
        reaction_base = self.reaction_residual_proj(reaction_features)

        enhanced_metabolite = metabolite_output + metabolite_base
        enhanced_reaction = reaction_output + reaction_base

        if not is_batched:
            enhanced_metabolite = enhanced_metabolite.squeeze(0)
            enhanced_reaction = enhanced_reaction.squeeze(0)

        return enhanced_reaction, enhanced_metabolite

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reshape_for_attention(self, x):
        """``[B, S, D] -> [B, H, S, D_h]``"""
        B, S, _ = x.shape
        return x.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

    def _reshape_from_attention(self, x):
        """``[B, H, S, D_h] -> [B, S, D]``"""
        B, _, S, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, S, self.hidden_dim)

    def _cross_attention(self, q, k, v, mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.scale * self.temperature)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        return torch.matmul(weights, v)
