"""Metabolite-Conditioned Reader (MCR): 1-hop attention-based metabolite predictor.

Uses metabolite SMILES embeddings as queries and connected reaction embeddings
as keys/values to predict metabolite concentrations via scaled dot-product attention.
All metabolites share the same parameters -- no metabolite-ID leakage.
"""

from collections import defaultdict
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Neighbor index utilities
# ---------------------------------------------------------------------------

def precompute_neighbor_indices(
    edge_index: torch.Tensor,
    metabolite_indices: torch.Tensor,
    reaction_indices: torch.Tensor,
    max_neighbors: int = 10,
    two_hop: bool = False,
    exclude_metabolites: Optional[set] = None,
) -> Dict[int, torch.Tensor]:
    """Pre-compute and cache 1-hop (or 2-hop) reaction neighbor indices per metabolite.

    Args:
        edge_index: ``[2, E]`` graph connectivity.
        metabolite_indices: ``[M]`` metabolite node indices.
        reaction_indices: ``[R]`` reaction node indices.
        max_neighbors: Maximum neighbors to keep per metabolite.
        two_hop: Include 2-hop reaction neighbors.
        exclude_metabolites: Metabolite indices to exclude from neighbour search.

    Returns:
        Mapping from global metabolite index to a tensor of global reaction indices.
    """
    edge_list = edge_index.cpu().numpy()
    metabolite_np = metabolite_indices.cpu().numpy()
    reaction_np = reaction_indices.cpu().numpy()

    metabolite_set = set(metabolite_np.tolist())
    reaction_set = set(reaction_np.tolist())

    adjacency: Dict[int, list] = defaultdict(list)
    for i in range(edge_list.shape[1]):
        src, tgt = int(edge_list[0, i]), int(edge_list[1, i])
        adjacency[src].append(tgt)
        adjacency[tgt].append(src)

    reaction_degree = {
        r: len([n for n in adjacency[r] if n in metabolite_set])
        for r in reaction_set
    }

    exclude_set = {int(m) for m in exclude_metabolites} if exclude_metabolites else set()
    neighbor_cache: Dict[int, torch.Tensor] = {}
    device = edge_index.device

    for met_idx in metabolite_np:
        met_idx = int(met_idx)
        if met_idx in exclude_set:
            neighbor_cache[met_idx] = torch.empty(0, dtype=torch.long, device=device)
            continue

        direct = [n for n in adjacency[met_idx] if n in reaction_set]

        if two_hop and direct:
            extra = set()
            for rxn in direct:
                connected_mets = [
                    n for n in adjacency[rxn]
                    if n in metabolite_set and n != met_idx and n not in exclude_set
                ]
                for other_met in connected_mets:
                    extra.update(n for n in adjacency[other_met] if n in reaction_set)
            direct.extend(extra)

        unique = list(set(direct))
        unique.sort(key=lambda r: reaction_degree.get(r, 0), reverse=True)
        unique = unique[:max_neighbors]

        if unique:
            neighbor_cache[met_idx] = torch.tensor(unique, dtype=torch.long, device=device)
        else:
            neighbor_cache[met_idx] = torch.empty(0, dtype=torch.long, device=device)

    return neighbor_cache


def extract_reaction_neighbors_cached(
    neighbor_cache: Dict[int, torch.Tensor],
    metabolite_indices: torch.Tensor,
    reaction_embeddings: torch.Tensor,
    max_neighbors: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract reaction neighbor embeddings using pre-computed neighbor indices.

    Args:
        neighbor_cache: Mapping from metabolite index to reaction neighbor indices.
        metabolite_indices: ``[M]`` metabolite node indices.
        reaction_embeddings: ``[B, N, D]`` all node embeddings.
        max_neighbors: Maximum neighbors per metabolite.

    Returns:
        neighbor_embeddings: ``[B, M, K, D]``
        neighbor_mask: ``[B, M, K]`` boolean mask for valid neighbors.
    """
    batch_size, num_nodes, hidden_dim = reaction_embeddings.shape
    num_metabolites = len(metabolite_indices)
    device = reaction_embeddings.device

    neighbor_indices = torch.full(
        (num_metabolites, max_neighbors), -1, dtype=torch.long, device=device
    )
    metabolite_cpu = metabolite_indices.cpu().numpy()

    for i, met_idx in enumerate(metabolite_cpu):
        met_idx = int(met_idx)
        if met_idx in neighbor_cache:
            neighbors = neighbor_cache[met_idx][:max_neighbors].to(device)
            n = len(neighbors)
            if n > 0:
                neighbor_indices[i, :n] = neighbors

    valid_mask = neighbor_indices != -1
    safe_indices = torch.where(valid_mask, neighbor_indices, 0)

    gathered = torch.gather(
        reaction_embeddings.unsqueeze(1).expand(-1, num_metabolites, -1, -1),
        2,
        safe_indices.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, hidden_dim),
    )
    neighbor_embeddings = gathered * valid_mask.unsqueeze(0).unsqueeze(-1)
    neighbor_mask = valid_mask.unsqueeze(0).expand(batch_size, -1, -1)

    return neighbor_embeddings, neighbor_mask


# ---------------------------------------------------------------------------
# MetaboliteConditionedReader
# ---------------------------------------------------------------------------

class MetaboliteConditionedReader(nn.Module):
    """Attention-based reader that conditions metabolite predictions on their
    chemical descriptors (1-hop only).

    For each metabolite *m*:
      1. Use SMILES embedding as query.
      2. Gather connected reaction embeddings as keys/values.
      3. Compute attention weights and context vector.
      4. Predict concentration from ``concat(context, projected_smiles)``.
    """

    def __init__(
        self,
        smiles_dim: int,
        gnn_hidden_dim: int,
        hidden_dim: int = 128,
        mlp_hidden: int = 128,
        normalize_descriptors: bool = True,
        dropout: float = 0.1,
        neighbor_dropout_rate: float = 0.0,
        neighbor_min_keep: int = 1,
        max_hops: int = 1,
        hop_decay_init: float = -0.5,
        use_gene_skip: bool = False,
        gene_dim: int = 0,
    ):
        super().__init__()

        self.smiles_dim = smiles_dim
        self.gnn_hidden_dim = gnn_hidden_dim
        self.hidden_dim = hidden_dim
        self.mlp_hidden = mlp_hidden
        self.normalize_descriptors = normalize_descriptors
        self.neighbor_dropout_rate = neighbor_dropout_rate
        self.neighbor_min_keep = int(max(0, neighbor_min_keep))

        # Attention projections
        self.query_projection = nn.Linear(smiles_dim, hidden_dim)
        self.key_projection = nn.Linear(gnn_hidden_dim, hidden_dim)
        self.value_projection = nn.Linear(gnn_hidden_dim, hidden_dim)

        # SMILES projection for concatenation
        self.smiles_projection = nn.Sequential(
            nn.Linear(smiles_dim, hidden_dim),
            nn.ReLU(),
        )

        # Prediction MLP
        self.prediction_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

        self.dropout = nn.Dropout(dropout)

        # Multi-hop bias (kept for checkpoint compatibility even if max_hops=1)
        self.max_hops = max_hops
        if max_hops > 1:
            self.hop_bias = nn.Parameter(
                torch.tensor([hop_decay_init * i for i in range(max_hops)])
            )
        else:
            self.hop_bias = None

        # Gene skip connection
        self.use_gene_skip = use_gene_skip
        if use_gene_skip:
            self.gene_projection = nn.Sequential(
                nn.Linear(gene_dim, hidden_dim),
                nn.ReLU(),
            )
            self.gene_gate = nn.Parameter(torch.tensor(0.01))
            self.prediction_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 3, mlp_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden, 1),
            )

        self._init_weights()

    def _init_weights(self):
        for mod in [self.query_projection, self.key_projection, self.value_projection]:
            nn.init.xavier_uniform_(mod.weight)
            nn.init.zeros_(mod.bias)
        for mod in self.smiles_projection:
            if isinstance(mod, nn.Linear):
                nn.init.xavier_uniform_(mod.weight)
                nn.init.zeros_(mod.bias)
        for mod in self.prediction_mlp:
            if isinstance(mod, nn.Linear):
                nn.init.xavier_uniform_(mod.weight)
                nn.init.zeros_(mod.bias)
        if self.use_gene_skip:
            for mod in self.gene_projection:
                if isinstance(mod, nn.Linear):
                    nn.init.xavier_uniform_(mod.weight)
                    nn.init.zeros_(mod.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        metabolite_smiles: torch.Tensor,
        reaction_neighbors: torch.Tensor,
        neighbor_mask: Optional[torch.Tensor] = None,
        hop_distances: Optional[torch.Tensor] = None,
        gene_neighbors: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Batched forward pass.

        Args:
            metabolite_smiles: ``[B, M, smiles_dim]``
            reaction_neighbors: ``[B, M, K, gnn_hidden_dim]``
            neighbor_mask: ``[B, M, K]`` boolean mask.
            hop_distances: ``[B, M, K]`` integer hop distances (optional).
            gene_neighbors: ``[B, M, K, gene_dim]`` (optional).

        Returns:
            ``[B, M]`` concentration predictions.
        """
        B, M, _ = metabolite_smiles.shape
        K = reaction_neighbors.size(2)

        descriptors = (
            F.normalize(metabolite_smiles, p=2, dim=-1)
            if self.normalize_descriptors else metabolite_smiles
        )

        queries = self.query_projection(descriptors)
        flat = reaction_neighbors.reshape(-1, self.gnn_hidden_dim)
        keys = self.key_projection(flat).view(B, M, K, self.hidden_dim)
        values = self.value_projection(flat).view(B, M, K, self.hidden_dim)

        scale = self.hidden_dim ** -0.5
        scores = torch.einsum("bnh,bnmh->bnm", queries, keys) * scale

        if self.hop_bias is not None and hop_distances is not None:
            hop_idx = (hop_distances.clamp(1, self.max_hops) - 1).long()
            scores = scores + self.hop_bias[hop_idx]

        scores = scores.unsqueeze(-2)

        if neighbor_mask is not None:
            mask_exp = neighbor_mask.unsqueeze(-2)
            scores = scores.masked_fill(~mask_exp, float("-inf"))
            valid = neighbor_mask.any(dim=-1, keepdim=True)
            if not valid.all():
                no_nb = ~valid.unsqueeze(-1)
                scores = torch.where(no_nb.expand_as(scores), torch.full_like(scores, -1e9), scores)

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        context = torch.einsum("bnm,bnmh->bnh", weights.squeeze(-2), values)
        smiles_feat = self.smiles_projection(descriptors)

        if self.use_gene_skip and gene_neighbors is not None:
            gene_proj = self.gene_projection(gene_neighbors)
            gene_ctx = torch.einsum("bnm,bnmh->bnh", weights.squeeze(-2), gene_proj)
            gene_ctx = self.gene_gate * gene_ctx
            combined = torch.cat([context, smiles_feat, gene_ctx], dim=-1)
        else:
            combined = torch.cat([context, smiles_feat], dim=-1)

        return self.prediction_mlp(combined).squeeze(-1)

    def forward_single_optimized(
        self,
        metabolite_smiles: torch.Tensor,
        reaction_neighbors: torch.Tensor,
        neighbor_mask: Optional[torch.Tensor] = None,
        hop_distances: Optional[torch.Tensor] = None,
        gene_neighbors: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Optimized forward for single sample (no batch dim).

        Args:
            metabolite_smiles: ``[M, smiles_dim]``
            reaction_neighbors: ``[M, K, gnn_hidden_dim]``
            neighbor_mask: ``[M, K]``
            hop_distances: ``[M, K]`` (optional)
            gene_neighbors: ``[M, K, gene_dim]`` (optional)

        Returns:
            ``[M]`` concentration predictions.
        """
        M, K = reaction_neighbors.shape[:2]

        descriptors = (
            F.normalize(metabolite_smiles, p=2, dim=-1)
            if self.normalize_descriptors else metabolite_smiles
        )

        queries = self.query_projection(descriptors)
        flat = reaction_neighbors.reshape(-1, self.gnn_hidden_dim)
        keys = self.key_projection(flat).view(M, K, self.hidden_dim)
        values = self.value_projection(flat).view(M, K, self.hidden_dim)

        scale = self.hidden_dim ** -0.5
        scores = torch.einsum("nh,nmh->nm", queries, keys) * scale

        if self.hop_bias is not None and hop_distances is not None:
            hop_idx = (hop_distances.clamp(1, self.max_hops) - 1).long()
            scores = scores + self.hop_bias[hop_idx]

        if neighbor_mask is not None:
            scores = scores.masked_fill(~neighbor_mask, float("-inf"))
            valid = neighbor_mask.any(dim=-1, keepdim=True)
            if not valid.all():
                no_nb = ~valid
                scores = torch.where(no_nb.expand_as(scores), torch.full_like(scores, -1e9), scores)

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        context = torch.einsum("nm,nmh->nh", weights, values)
        smiles_feat = self.smiles_projection(descriptors)

        if self.use_gene_skip and gene_neighbors is not None:
            gene_proj = self.gene_projection(gene_neighbors)
            gene_ctx = torch.einsum("nm,nmh->nh", weights, gene_proj)
            gene_ctx = self.gene_gate * gene_ctx
            combined = torch.cat([context, smiles_feat, gene_ctx], dim=-1)
        else:
            combined = torch.cat([context, smiles_feat], dim=-1)

        return self.prediction_mlp(combined).squeeze(-1)
