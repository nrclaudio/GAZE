"""
Metabolite-Conditioned Reader (MCR): Attention-based metabolite concentration predictor.

This module implements a chemical similarity-aware attention mechanism for metabolite
concentration prediction. The key innovation is using metabolite SMILES embeddings as
queries to aggregate relevant reaction context through attention.

Architecture:
    1. Query Generation: Metabolite SMILES embeddings → attention queries
    2. Context Retrieval: Connected reaction embeddings → keys/values
    3. Attention Computation: Scaled dot-product attention with optional masking
    4. Prediction: Concatenated [context, SMILES] → MLP → concentration

Key Design Principles:
    - Parameter sharing: All metabolites use identical weights (no ID leakage)
    - Chemical awareness: SMILES embeddings capture structural similarity
    - Graph locality: Only connected reactions participate in attention
    - Scalability: Handles variable numbers of reaction neighbors
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import numpy as np


def precompute_neighbor_indices(
    edge_index: torch.Tensor,
    metabolite_indices: torch.Tensor,
    reaction_indices: torch.Tensor,
    max_neighbors: int = 10,
    two_hop: bool = False,
    exclude_metabolites: Optional[set] = None
) -> Dict[int, torch.Tensor]:
    """
    Pre-compute and cache neighbor indices for each metabolite.
    Optimized version with reduced memory allocations and faster processing.
    
    Args:
        edge_index: [2, num_edges] - Graph connectivity
        metabolite_indices: [num_metabolites] - Metabolite node indices
        reaction_indices: [num_reactions] - Reaction node indices
        max_neighbors: Maximum neighbors per metabolite
        two_hop: Whether to include 2-hop neighbors
        exclude_metabolites: Set of metabolite indices to exclude from being neighbors
        
    Returns:
        neighbor_cache: Dict mapping metabolite index to reaction neighbor indices
    """
    # Convert to numpy once for efficiency
    edge_list = edge_index.cpu().numpy()
    metabolite_np = metabolite_indices.cpu().numpy()
    reaction_np = reaction_indices.cpu().numpy()
    
    # Use sets for O(1) lookup performance
    metabolite_set = set(metabolite_np.tolist())
    reaction_set = set(reaction_np.tolist())
    
    # Build adjacency lists using defaultdict for cleaner code
    from collections import defaultdict
    adjacency = defaultdict(list)
    
    # Single pass through edges to build adjacency
    for i in range(edge_list.shape[1]):
        src, tgt = int(edge_list[0, i]), int(edge_list[1, i])
        adjacency[src].append(tgt)
        adjacency[tgt].append(src)
    
    # Pre-compute reaction degrees (number of connected metabolites) for ranking
    reaction_degree = {}
    for reaction in reaction_set:
        neighbours = [n for n in adjacency[reaction] if n in metabolite_set]
        reaction_degree[reaction] = len(neighbours)

    # Pre-compute neighbors for each metabolite
    neighbor_cache = {}
    device = edge_index.device
    
    # Convert exclude_metabolites to set of ints if provided
    if exclude_metabolites is not None:
        exclude_set = {int(m) for m in exclude_metabolites}
    else:
        exclude_set = set()
    
    for met_idx in metabolite_np:
        met_idx = int(met_idx)
        
        # Skip if this metabolite should be excluded
        if met_idx in exclude_set:
            neighbor_cache[met_idx] = torch.empty(0, dtype=torch.long, device=device)
            continue
        
        # Get direct reaction neighbors with list comprehension (faster)
        direct_neighbors = [n for n in adjacency[met_idx] if n in reaction_set]
        
        # Add 2-hop neighbors if requested
        if two_hop and direct_neighbors:
            two_hop_neighbors = set()
            for reaction in direct_neighbors:
                # Find metabolites connected to this reaction (excluding self and excluded metabolites)
                connected_mets = [n for n in adjacency[reaction] 
                                if n in metabolite_set and n != met_idx and n not in exclude_set]
                # Find reactions connected to those metabolites
                for other_met in connected_mets:
                    two_hop_neighbors.update(n for n in adjacency[other_met] 
                                           if n in reaction_set)
            direct_neighbors.extend(two_hop_neighbors)
        
        # Remove duplicates, rank by reaction degree, and limit
        unique_neighbors = list(set(direct_neighbors))
        unique_neighbors.sort(key=lambda r: reaction_degree.get(r, 0), reverse=True)
        unique_neighbors = unique_neighbors[:max_neighbors]
        if unique_neighbors:
            neighbor_cache[met_idx] = torch.tensor(unique_neighbors, 
                                                 dtype=torch.long, device=device)
        else:
            neighbor_cache[met_idx] = torch.empty(0, dtype=torch.long, device=device)
    
    return neighbor_cache


def precompute_neighbor_indices_multihop(
    edge_index: torch.Tensor,
    metabolite_indices: torch.Tensor,
    reaction_indices: torch.Tensor,
    max_neighbors: int = 10,
    max_hops: int = 2,
    exclude_metabolites: Optional[set] = None
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Pre-compute multi-hop neighbor indices with hop distance tracking via BFS.

    For each metabolite, performs BFS over the bipartite graph to find reactions
    within `max_hops` hops (where a hop is metabolite -> reaction -> metabolite -> reaction).

    Args:
        edge_index: [2, num_edges] - Graph connectivity
        metabolite_indices: [num_metabolites] - Metabolite node indices
        reaction_indices: [num_reactions] - Reaction node indices
        max_neighbors: Maximum neighbors per metabolite
        max_hops: Maximum number of hops (1=direct, 2=through one intermediate metabolite, etc.)
        exclude_metabolites: Set of metabolite indices to exclude (for LOMO safety)

    Returns:
        neighbor_cache: Dict mapping metabolite index to
            (reaction_neighbor_indices, hop_distances) tensors
    """
    from collections import defaultdict, deque

    # Convert to numpy for efficiency
    edge_list = edge_index.cpu().numpy()
    metabolite_np = metabolite_indices.cpu().numpy()
    reaction_np = reaction_indices.cpu().numpy()

    metabolite_set = set(metabolite_np.tolist())
    reaction_set = set(reaction_np.tolist())

    # Build adjacency
    adjacency = defaultdict(list)
    for i in range(edge_list.shape[1]):
        src, tgt = int(edge_list[0, i]), int(edge_list[1, i])
        adjacency[src].append(tgt)
        adjacency[tgt].append(src)

    # Pre-compute reaction degrees for ranking
    reaction_degree = {}
    for reaction in reaction_set:
        neighbours = [n for n in adjacency[reaction] if n in metabolite_set]
        reaction_degree[reaction] = len(neighbours)

    exclude_set = {int(m) for m in exclude_metabolites} if exclude_metabolites else set()

    neighbor_cache = {}
    device = edge_index.device

    for met_idx in metabolite_np:
        met_idx = int(met_idx)

        if met_idx in exclude_set:
            empty_idx = torch.empty(0, dtype=torch.long, device=device)
            empty_hops = torch.empty(0, dtype=torch.long, device=device)
            neighbor_cache[met_idx] = (empty_idx, empty_hops)
            continue

        # BFS to find reactions at each hop level
        # hop 1: direct reaction neighbors of met_idx
        # hop 2: reactions reachable via met_idx -> rxn -> other_met -> rxn2
        # (hop path: met -> rxn (hop1) -> met2 -> rxn2 (hop2))
        reaction_hops: Dict[int, int] = {}  # reaction_id -> min_hop_distance
        visited_mets = {met_idx}
        frontier_mets = {met_idx}

        for hop in range(1, max_hops + 1):
            next_frontier_mets = set()
            for m in frontier_mets:
                for rxn in adjacency[m]:
                    if rxn not in reaction_set:
                        continue
                    if rxn not in reaction_hops:
                        reaction_hops[rxn] = hop
                    # Explore metabolites connected to this reaction for next hop
                    if hop < max_hops:
                        for m2 in adjacency[rxn]:
                            if m2 in metabolite_set and m2 not in visited_mets and m2 not in exclude_set:
                                next_frontier_mets.add(m2)
            visited_mets.update(next_frontier_mets)
            frontier_mets = next_frontier_mets

        if not reaction_hops:
            empty_idx = torch.empty(0, dtype=torch.long, device=device)
            empty_hops = torch.empty(0, dtype=torch.long, device=device)
            neighbor_cache[met_idx] = (empty_idx, empty_hops)
            continue

        # Sort by (hop_distance, reaction_degree descending), truncate
        sorted_reactions = sorted(
            reaction_hops.items(),
            key=lambda item: (item[1], -reaction_degree.get(item[0], 0))
        )
        sorted_reactions = sorted_reactions[:max_neighbors]

        rxn_ids = [r for r, _ in sorted_reactions]
        hop_dists = [h for _, h in sorted_reactions]

        neighbor_cache[met_idx] = (
            torch.tensor(rxn_ids, dtype=torch.long, device=device),
            torch.tensor(hop_dists, dtype=torch.long, device=device),
        )

    return neighbor_cache


def extract_reaction_neighbors_cached_multihop(
    neighbor_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    metabolite_indices: torch.Tensor,
    reaction_embeddings: torch.Tensor,
    max_neighbors: int = 10
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract reaction neighbor embeddings and hop distances from multihop cache.

    Args:
        neighbor_cache: Pre-computed mapping of metabolite -> (neighbor_indices, hop_distances)
        metabolite_indices: [num_metabolites] - Metabolite node indices
        reaction_embeddings: [batch_size, num_nodes, hidden_dim] - All node embeddings
        max_neighbors: Maximum neighbors per metabolite

    Returns:
        neighbor_embeddings: [batch_size, num_metabolites, max_neighbors, hidden_dim]
        neighbor_mask: [batch_size, num_metabolites, max_neighbors] - Valid neighbor mask
        hop_distances: [batch_size, num_metabolites, max_neighbors] - Hop distance per neighbor (int)
    """
    batch_size, num_nodes, hidden_dim = reaction_embeddings.shape
    num_metabolites = len(metabolite_indices)
    device = reaction_embeddings.device

    neighbor_embeddings = torch.zeros(
        batch_size, num_metabolites, max_neighbors, hidden_dim,
        device=device, dtype=reaction_embeddings.dtype
    )
    neighbor_mask = torch.zeros(
        batch_size, num_metabolites, max_neighbors,
        dtype=torch.bool, device=device
    )
    hop_dist_tensor = torch.zeros(
        num_metabolites, max_neighbors,
        dtype=torch.long, device=device
    )

    metabolite_cpu = metabolite_indices.cpu().numpy()

    neighbor_indices = torch.full(
        (num_metabolites, max_neighbors), -1, dtype=torch.long, device=device
    )

    for i, met_idx in enumerate(metabolite_cpu):
        met_idx = int(met_idx)
        if met_idx in neighbor_cache:
            rxn_indices, hops = neighbor_cache[met_idx]
            rxn_indices = rxn_indices[:max_neighbors].to(device)
            hops = hops[:max_neighbors].to(device)
            num_neighbors = len(rxn_indices)
            if num_neighbors > 0:
                neighbor_indices[i, :num_neighbors] = rxn_indices
                hop_dist_tensor[i, :num_neighbors] = hops

    valid_mask = neighbor_indices != -1
    neighbor_mask[:, :, :] = valid_mask.unsqueeze(0)

    safe_indices = torch.where(valid_mask, neighbor_indices, 0)

    gathered_embeddings = torch.gather(
        reaction_embeddings.unsqueeze(1).expand(-1, num_metabolites, -1, -1),
        2,
        safe_indices.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, hidden_dim)
    )
    neighbor_embeddings = gathered_embeddings * valid_mask.unsqueeze(0).unsqueeze(-1)

    # Expand hop distances to batch dimension
    hop_distances = hop_dist_tensor.unsqueeze(0).expand(batch_size, -1, -1)

    return neighbor_embeddings, neighbor_mask, hop_distances


class MetaboliteConditionedReader(nn.Module):
    """
    Attention-based reader that conditions metabolite predictions on their chemical descriptors.
    
    For each metabolite m in sample s:
    1. Use SMILES embedding e_m as query
    2. Gather reaction neighbor embeddings as keys/values
    3. Compute attention weights and context vector
    4. Predict concentration from concat(context, e_m)
    
    All metabolites share the same parameters - no ID leakage.
    """
    
    def __init__(self,
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
                 gene_dim: int = 0):
        """Initialize the Metabolite-Conditioned Reader.

        Args:
            smiles_dim: Dimension of input SMILES embeddings
            gnn_hidden_dim: Dimension of GNN reaction embeddings (keys/values)
            hidden_dim: Hidden dimension for attention projections
            mlp_hidden: Hidden dimension for final prediction MLP
            normalize_descriptors: Whether to L2 normalize SMILES before attention
            dropout: Dropout probability for regularization
            neighbor_dropout_rate: Probability of dropping an entire neighbour before attention (stochastic regularization)
            neighbor_min_keep: Minimum number of neighbors to keep after dropout
            max_hops: Maximum hops for multi-hop MCR (1=standard, 2+=multi-hop with hop bias)
            hop_decay_init: Initial attention bias decay per hop level
            use_gene_skip: Whether to skip-connect raw gene expression to prediction head
            gene_dim: Dimension of raw gene expression (required when use_gene_skip=True)

        Raises:
            ValueError: If dimensions are non-positive or dropout is invalid
        """
        if smiles_dim <= 0 or gnn_hidden_dim <= 0 or hidden_dim <= 0 or mlp_hidden <= 0:
            raise ValueError("All dimensions must be positive")
        if not 0 <= dropout <= 1:
            raise ValueError("Dropout must be between 0 and 1")
        if not 0 <= neighbor_dropout_rate <= 1:
            raise ValueError("Neighbor dropout rate must be between 0 and 1")
        super().__init__()

        self.smiles_dim = smiles_dim
        self.gnn_hidden_dim = gnn_hidden_dim
        self.hidden_dim = hidden_dim
        self.mlp_hidden = mlp_hidden
        self.normalize_descriptors = normalize_descriptors
        self.neighbor_dropout_rate = neighbor_dropout_rate
        self.neighbor_min_keep = int(max(0, neighbor_min_keep))
        
        # Attention mechanism components
        self.query_projection = nn.Linear(smiles_dim, hidden_dim)
        self.key_projection = nn.Linear(gnn_hidden_dim, hidden_dim)
        self.value_projection = nn.Linear(gnn_hidden_dim, hidden_dim)

        # Project SMILES descriptors into interaction space before concatenation
        self.smiles_projection = nn.Sequential(
            nn.Linear(smiles_dim, hidden_dim),
            nn.ReLU(),
        )

        # Concentration prediction MLP
        self.prediction_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1)
        )
        
        self.dropout = nn.Dropout(dropout)

        # Multi-hop: learnable per-hop attention bias
        self.max_hops = max_hops
        if max_hops > 1:
            # hop_bias[k] is added to attention scores for neighbors at hop k+1
            # Initialized with decay so farther hops get lower initial attention
            self.hop_bias = nn.Parameter(
                torch.tensor([hop_decay_init * i for i in range(max_hops)])
            )
        else:
            self.hop_bias = None

        # Gene expression skip connection
        self.use_gene_skip = use_gene_skip
        if use_gene_skip:
            self.gene_projection = nn.Sequential(
                nn.Linear(gene_dim, hidden_dim),
                nn.ReLU(),
            )
            # Learnable gate initialized near zero for safe warm-start
            self.gene_gate = nn.Parameter(torch.tensor(0.01))
            # Prediction MLP takes 3 * hidden_dim (context + smiles + gene)
            self.prediction_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 3, mlp_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden, 1)
            )

        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in [self.query_projection, self.key_projection, self.value_projection]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

        # Initialize MLP weights and SMILES projection
        for module in self.smiles_projection:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        # Initialize MLP weights
        for module in self.prediction_mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        # Initialize gene projection weights if present
        if self.use_gene_skip:
            for module in self.gene_projection:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
    
    def forward(self,
                metabolite_smiles: torch.Tensor,
                reaction_neighbors: torch.Tensor,
                neighbor_mask: Optional[torch.Tensor] = None,
                hop_distances: Optional[torch.Tensor] = None,
                gene_neighbors: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass: predict metabolite concentrations using attention.

        Args:
            metabolite_smiles: [batch_size, num_metabolites, smiles_dim]
                SMILES embeddings for query generation
            reaction_neighbors: [batch_size, num_metabolites, max_neighbors, gnn_hidden_dim]
                Connected reaction embeddings for each metabolite
            neighbor_mask: [batch_size, num_metabolites, max_neighbors], optional
                Boolean mask indicating valid neighbors (True = valid)
            hop_distances: [batch_size, num_metabolites, max_neighbors], optional
                Integer hop distances for multi-hop attention bias
            gene_neighbors: [batch_size, num_metabolites, max_neighbors, gene_dim], optional
                Raw gene expression of neighboring reactions for gene skip connection

        Returns:
            torch.Tensor: [batch_size, num_metabolites] concentration predictions

        Raises:
            ValueError: If input tensor shapes are incompatible
        """
        self._validate_forward_inputs(metabolite_smiles, reaction_neighbors, neighbor_mask)
        
        batch_size, num_metabolites, _ = metabolite_smiles.shape
        max_neighbors = reaction_neighbors.size(2)
        
        # Apply L2 normalization if enabled (improves attention stability)
        descriptors = (F.normalize(metabolite_smiles, p=2, dim=-1) 
                      if self.normalize_descriptors else metabolite_smiles)
        
        # Generate attention queries from SMILES embeddings
        queries = self.query_projection(descriptors)  # [batch_size, num_metabolites, hidden_dim]
        
        # Optimized batch projection of keys and values
        # Reshape for efficient batch matrix multiplication
        neighbors_flat = reaction_neighbors.view(-1, self.gnn_hidden_dim)  # [batch_size * num_metabolites * max_neighbors, gnn_hidden_dim]
        
        # Batch compute all projections at once
        keys_flat = self.key_projection(neighbors_flat)    # [batch_size * num_metabolites * max_neighbors, hidden_dim]
        values_flat = self.value_projection(neighbors_flat)  # [batch_size * num_metabolites * max_neighbors, hidden_dim]
        
        # Reshape back to original structure
        keys = keys_flat.view(batch_size, num_metabolites, max_neighbors, self.hidden_dim)
        values = values_flat.view(batch_size, num_metabolites, max_neighbors, self.hidden_dim)
        
        # Compute scaled dot-product attention scores with optimized einsum
        # [batch_size, num_metabolites, 1, hidden_dim] @ [batch_size, num_metabolites, hidden_dim, max_neighbors]
        # -> [batch_size, num_metabolites, 1, max_neighbors]
        scale = self.hidden_dim ** -0.5
        attention_scores = torch.einsum('bnh,bnmh->bnm', queries, keys) * scale

        # Multi-hop attention bias: add learnable per-hop bias to scores
        if self.hop_bias is not None and hop_distances is not None:
            # hop_distances: [B, M, N] with values 1..max_hops (0 for invalid)
            # Clamp to valid range and subtract 1 for 0-based indexing
            hop_idx = (hop_distances.clamp(1, self.max_hops) - 1).long()
            hop_bias_values = self.hop_bias[hop_idx]  # [B, M, N]
            attention_scores = attention_scores + hop_bias_values

        attention_scores = attention_scores.unsqueeze(-2)  # Add dimension for consistency

        # Apply neighbor mask to exclude invalid reactions (with stochastic dropout if enabled)
        effective_mask = self._prepare_neighbor_mask(
            neighbor_mask,
            batch_size=batch_size,
            num_metabolites=num_metabolites,
            max_neighbors=max_neighbors,
            device=reaction_neighbors.device,
        )

        if effective_mask is not None:
            attention_scores = self._apply_attention_mask(attention_scores, effective_mask)

            # CRITICAL FIX: Handle edge case where all neighbors are masked (would cause NaN in softmax)
            # This happens during LOMO cross-validation when held-out metabolites are excluded from cache
            valid_neighbors = effective_mask.any(dim=-1, keepdim=True)  # [batch_size, num_metabolites, 1]

            if not valid_neighbors.all():
                # Replace -inf with large negative values for metabolites with no valid neighbors
                # This prevents softmax from producing NaN while maintaining near-zero attention
                no_neighbors_mask = ~valid_neighbors.unsqueeze(-1)  # [batch_size, num_metabolites, 1, 1]
                attention_scores = torch.where(
                    no_neighbors_mask.expand_as(attention_scores),
                    torch.full_like(attention_scores, -1e9),  # Large negative but not -inf
                    attention_scores
                )
        
        # Compute attention weights and aggregate reaction context
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Optimized weighted aggregation using einsum
        # [batch_size, num_metabolites, 1, max_neighbors] @ [batch_size, num_metabolites, max_neighbors, hidden_dim]
        # -> [batch_size, num_metabolites, hidden_dim]
        context_vector = torch.einsum('bnm,bnmh->bnh', attention_weights.squeeze(-2), values)
        
        # Combine aggregated context with processed SMILES features to align with baseline head
        smiles_features = self.smiles_projection(descriptors)

        if self.use_gene_skip and gene_neighbors is not None:
            gene_proj = self.gene_projection(gene_neighbors)  # [B, M, N, hidden_dim]
            # Reuse same attention weights to aggregate gene features
            gene_context = torch.einsum('bnm,bnmh->bnh', attention_weights.squeeze(-2), gene_proj)
            gene_context = self.gene_gate * gene_context
            combined_features = torch.cat([context_vector, smiles_features, gene_context], dim=-1)
        else:
            combined_features = torch.cat([context_vector, smiles_features], dim=-1)

        # Generate final concentration predictions
        predictions = self.prediction_mlp(combined_features).squeeze(-1)

        return predictions

    def forward_single_optimized(self,
                                metabolite_smiles: torch.Tensor,
                                reaction_neighbors: torch.Tensor,
                                neighbor_mask: Optional[torch.Tensor] = None,
                                hop_distances: Optional[torch.Tensor] = None,
                                gene_neighbors: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Optimized forward pass for single sample (batch_size=1) inference.

        Bypasses batch operations for maximum single-sample performance.

        Args:
            metabolite_smiles: [num_metabolites, smiles_dim] Single sample SMILES
            reaction_neighbors: [num_metabolites, max_neighbors, gnn_hidden_dim] Single sample neighbors
            neighbor_mask: [num_metabolites, max_neighbors] Single sample mask
            hop_distances: [num_metabolites, max_neighbors] Hop distances (optional, for multi-hop)
            gene_neighbors: [num_metabolites, max_neighbors, gene_dim] Raw gene expression (optional)

        Returns:
            torch.Tensor: [num_metabolites] concentration predictions
        """
        num_metabolites, max_neighbors = reaction_neighbors.shape[:2]
        
        # Apply L2 normalization if enabled
        descriptors = (F.normalize(metabolite_smiles, p=2, dim=-1) 
                      if self.normalize_descriptors else metabolite_smiles)
        
        # Generate queries
        queries = self.query_projection(descriptors)  # [num_metabolites, hidden_dim]
        
        # Efficient single-sample key/value projection
        neighbors_flat = reaction_neighbors.view(-1, self.gnn_hidden_dim)
        keys_flat = self.key_projection(neighbors_flat)
        values_flat = self.value_projection(neighbors_flat)
        
        keys = keys_flat.view(num_metabolites, max_neighbors, self.hidden_dim)
        values = values_flat.view(num_metabolites, max_neighbors, self.hidden_dim)
        
        # Single-sample attention computation
        scale = self.hidden_dim ** -0.5
        attention_scores = torch.einsum('nh,nmh->nm', queries, keys) * scale

        # Multi-hop attention bias (single-sample)
        if self.hop_bias is not None and hop_distances is not None:
            hop_idx = (hop_distances.clamp(1, self.max_hops) - 1).long()
            hop_bias_values = self.hop_bias[hop_idx]  # [M, N]
            attention_scores = attention_scores + hop_bias_values

        # Apply mask if provided (with stochastic dropout when training)
        effective_mask = self._prepare_neighbor_mask(
            neighbor_mask.unsqueeze(0) if neighbor_mask is not None else None,
            batch_size=1,
            num_metabolites=num_metabolites,
            max_neighbors=max_neighbors,
            device=reaction_neighbors.device,
        )

        if effective_mask is not None:
            effective_mask = effective_mask.squeeze(0)
            attention_scores = attention_scores.masked_fill(~effective_mask, float('-inf'))
            
            # CRITICAL FIX: Handle edge case where all neighbors are masked
            valid_neighbors = effective_mask.any(dim=-1, keepdim=True)
            if not valid_neighbors.all():
                no_neighbors_mask = ~valid_neighbors
                attention_scores = torch.where(
                    no_neighbors_mask.expand_as(attention_scores),
                    torch.full_like(attention_scores, -1e9),
                    attention_scores
                )

        # Compute weights and context
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Aggregate context
        context_vector = torch.einsum('nm,nmh->nh', attention_weights, values)
        
        # Final prediction with processed SMILES features for consistency
        smiles_features = self.smiles_projection(descriptors)

        if self.use_gene_skip and gene_neighbors is not None:
            gene_proj = self.gene_projection(gene_neighbors)  # [M, N, hidden_dim]
            gene_context = torch.einsum('nm,nmh->nh', attention_weights, gene_proj)
            gene_context = self.gene_gate * gene_context
            combined_features = torch.cat([context_vector, smiles_features, gene_context], dim=-1)
        else:
            combined_features = torch.cat([context_vector, smiles_features], dim=-1)

        predictions = self.prediction_mlp(combined_features).squeeze(-1)

        return predictions

    def get_attention_weights(self,
                            metabolite_smiles: torch.Tensor,
                            reaction_neighbors: torch.Tensor,
                            neighbor_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Extract attention weights for interpretability analysis.
        
        Args:
            metabolite_smiles: [batch_size, num_metabolites, smiles_dim]
            reaction_neighbors: [batch_size, num_metabolites, max_neighbors, gnn_hidden_dim]
            neighbor_mask: [batch_size, num_metabolites, max_neighbors], optional
            
        Returns:
            torch.Tensor: [batch_size, num_metabolites, max_neighbors] attention weights
        """
        batch_size, num_metabolites, _ = metabolite_smiles.shape
        max_neighbors = reaction_neighbors.size(2)
        
        # Apply same preprocessing as forward pass
        descriptors = (F.normalize(metabolite_smiles, p=2, dim=-1)
                      if self.normalize_descriptors else metabolite_smiles)
        
        # Generate queries
        queries = self.query_projection(descriptors)
        
        # Optimized key projection using batch computation
        neighbors_flat = reaction_neighbors.view(-1, self.gnn_hidden_dim)
        keys_flat = self.key_projection(neighbors_flat)
        keys = keys_flat.view(batch_size, num_metabolites, max_neighbors, self.hidden_dim)
        
        # Compute attention scores using optimized einsum
        scale = self.hidden_dim ** -0.5
        attention_scores = torch.einsum('bnh,bnmh->bnm', queries, keys) * scale
        attention_scores = attention_scores.unsqueeze(-2)  # Add dimension for masking
        
        # Apply masking if provided
        if neighbor_mask is not None:
            attention_scores = self._apply_attention_mask(attention_scores, neighbor_mask)
        
        # Return normalized attention weights
        return F.softmax(attention_scores, dim=-1).squeeze(-2)
    
    def _validate_forward_inputs(self, metabolite_smiles: torch.Tensor,
                               reaction_neighbors: torch.Tensor,
                               neighbor_mask: Optional[torch.Tensor]) -> None:
        """Validate input tensor shapes and compatibility."""
        if metabolite_smiles.dim() != 3:
            raise ValueError(f"metabolite_smiles must be 3D, got {metabolite_smiles.dim()}D")
        if reaction_neighbors.dim() != 4:
            raise ValueError(f"reaction_neighbors must be 4D, got {reaction_neighbors.dim()}D")
        
        batch_size, num_metabolites, smiles_dim = metabolite_smiles.shape
        batch_size_r, num_metabolites_r, max_neighbors, gnn_dim = reaction_neighbors.shape
        
        if batch_size != batch_size_r or num_metabolites != num_metabolites_r:
            raise ValueError("Batch size and metabolite count mismatch between inputs")
        if smiles_dim != self.smiles_dim:
            raise ValueError(f"SMILES dimension mismatch: expected {self.smiles_dim}, got {smiles_dim}")
        if gnn_dim != self.gnn_hidden_dim:
            raise ValueError(f"GNN dimension mismatch: expected {self.gnn_hidden_dim}, got {gnn_dim}")
            
        if neighbor_mask is not None:
            if neighbor_mask.shape != (batch_size, num_metabolites, max_neighbors):
                raise ValueError("Neighbor mask shape incompatible with reaction_neighbors")
    
    def _compute_attention_scores(self, queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """Compute scaled dot-product attention scores with optimized operations."""
        # Optimized einsum computation for better performance
        scale = self.hidden_dim ** -0.5
        attention_scores = torch.einsum('bnh,bnmh->bnm', queries, keys) * scale
        return attention_scores.unsqueeze(-2)  # Add dimension for consistency with masking

    def _prepare_neighbor_mask(
        self,
        neighbor_mask: Optional[torch.Tensor],
        *,
        batch_size: int,
        num_metabolites: int,
        max_neighbors: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Combine base neighbor mask with stochastic neighbor dropout when enabled."""
        if neighbor_mask is not None:
            mask = neighbor_mask.to(device)
            if mask.dtype != torch.bool:
                mask = mask.to(dtype=torch.bool)
        else:
            mask = None

        if (
            not self.training
            or self.neighbor_dropout_rate <= 0
            or max_neighbors == 0
        ):
            return mask

        dropout_shape = (batch_size, num_metabolites, max_neighbors)
        dropout_mask = torch.rand(dropout_shape, device=device) > self.neighbor_dropout_rate

        base_mask = mask if mask is not None else torch.ones_like(dropout_mask, dtype=torch.bool)
        # If a metabolite has very few neighbors (<= neighbor_min_keep), skip dropout entirely for it
        if self.neighbor_min_keep > 0:
            base_counts = base_mask.sum(dim=-1, keepdim=True)  # [B, M, 1]
            scarce = base_counts <= self.neighbor_min_keep
        else:
            scarce = torch.zeros((batch_size, num_metabolites, 1), dtype=torch.bool, device=device)

        effective_mask = base_mask & dropout_mask

        # Ensure we keep at least neighbor_min_keep neighbors when possible
        kept_counts = effective_mask.sum(dim=-1, keepdim=True)  # [B, M, 1]
        available = base_mask.any(dim=-1, keepdim=True)

        # Need fix when: (kept < min_keep) and (there are available neighbors)
        if self.neighbor_min_keep > 0:
            needs_fix = (kept_counts < self.neighbor_min_keep) & available
        else:
            needs_fix = torch.zeros_like(available, dtype=torch.bool)

        if needs_fix.any():
            # Build a mask that keeps the first K True positions from base_mask along neighbor dim
            # This is deterministic and cheap; avoids random instability for low-degree cases.
            # first_k_mask = base_mask & (cumsum(base_mask) <= K)
            cumsum = base_mask.cumsum(dim=-1)
            first_k_mask = base_mask & (cumsum <= max(1, self.neighbor_min_keep))
            # For entries needing fix, replace effective_mask with first_k_mask
            effective_mask = torch.where(needs_fix.expand_as(effective_mask), first_k_mask, effective_mask)

        # For metabolites marked scarce, use the base_mask (no dropout applied)
        if scarce.any():
            effective_mask = torch.where(scarce.expand_as(effective_mask), base_mask, effective_mask)

        return effective_mask
    
    def _apply_attention_mask(self, attention_scores: torch.Tensor, neighbor_mask: torch.Tensor) -> torch.Tensor:
        """Apply neighbor mask to exclude invalid reactions from attention."""
        mask_expanded = neighbor_mask.unsqueeze(-2)  # [batch_size, num_metabolites, 1, max_neighbors]
        return attention_scores.masked_fill(~mask_expanded, float('-inf'))


def extract_reaction_neighbors_cached(
    neighbor_cache: Dict[int, torch.Tensor],
    metabolite_indices: torch.Tensor,
    reaction_embeddings: torch.Tensor,
    max_neighbors: int = 10
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract reaction neighbor embeddings using pre-computed neighbor indices.
    Fully vectorized implementation for maximum performance.
    
    Args:
        neighbor_cache: Pre-computed mapping of metabolite -> reaction neighbors
        metabolite_indices: [num_metabolites] - Metabolite node indices
        reaction_embeddings: [batch_size, num_nodes, hidden_dim] - All node embeddings
        max_neighbors: Maximum neighbors per metabolite
        
    Returns:
        neighbor_embeddings: [batch_size, num_metabolites, max_neighbors, hidden_dim]
        neighbor_mask: [batch_size, num_metabolites, max_neighbors] - Valid neighbor mask
    """
    batch_size, num_nodes, hidden_dim = reaction_embeddings.shape
    num_metabolites = len(metabolite_indices)
    device = reaction_embeddings.device
    
    # Pre-allocate output tensors with correct device and dtype
    neighbor_embeddings = torch.zeros(
        batch_size, num_metabolites, max_neighbors, hidden_dim, 
        device=device, dtype=reaction_embeddings.dtype
    )
    neighbor_mask = torch.zeros(
        batch_size, num_metabolites, max_neighbors, 
        dtype=torch.bool, device=device
    )
    
    # Convert metabolite indices to CPU once for cache lookup
    metabolite_cpu = metabolite_indices.cpu().numpy()
    
    # Build vectorized neighbor index tensor and counts
    neighbor_indices = torch.full((num_metabolites, max_neighbors), -1, 
                                 dtype=torch.long, device=device)
    neighbor_counts = torch.zeros(num_metabolites, dtype=torch.long, device=device)
    
    # Fill neighbor indices tensor efficiently
    for i, met_idx in enumerate(metabolite_cpu):
        met_idx = int(met_idx)
        if met_idx in neighbor_cache:
            neighbors = neighbor_cache[met_idx][:max_neighbors].to(device)
            num_neighbors = len(neighbors)
            if num_neighbors > 0:
                neighbor_indices[i, :num_neighbors] = neighbors
                neighbor_counts[i] = num_neighbors
    
    # Create mask for valid neighbors
    valid_mask = neighbor_indices != -1
    neighbor_mask[:, :, :] = valid_mask.unsqueeze(0)
    
    # Fully vectorized embedding extraction using gather
    # Replace invalid indices (-1) with 0 to avoid indexing errors
    safe_indices = torch.where(valid_mask, neighbor_indices, 0)
    
    # Gather embeddings for all positions at once
    # [batch_size, num_nodes, hidden_dim] -> [batch_size, num_metabolites, max_neighbors, hidden_dim]
    gathered_embeddings = torch.gather(
        reaction_embeddings.unsqueeze(1).expand(-1, num_metabolites, -1, -1),  # Expand for broadcasting
        2,  # Gather along node dimension
        safe_indices.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, hidden_dim)
    )
    
    # Apply mask to zero out invalid positions
    neighbor_embeddings = gathered_embeddings * valid_mask.unsqueeze(0).unsqueeze(-1)
    
    return neighbor_embeddings, neighbor_mask


def extract_reaction_neighbors(edge_index: torch.Tensor,
                              metabolite_indices: torch.Tensor, 
                              reaction_indices: torch.Tensor,
                              reaction_embeddings: torch.Tensor,
                              max_neighbors: int = 50,
                              two_hop: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract reaction neighbor embeddings for metabolite attention computation.
    
    For each metabolite, finds connected reactions within the metabolic network graph.
    Supports both 1-hop (direct connections) and 2-hop (metabolite-reaction-metabolite-reaction) paths.
    
    Args:
        edge_index: [2, num_edges] Graph connectivity matrix
        metabolite_indices: [num_metabolites] Metabolite node indices
        reaction_indices: [num_reactions] Reaction node indices  
        reaction_embeddings: [batch_size, num_nodes, hidden_dim] All node embeddings
        max_neighbors: Maximum neighbors per metabolite (for memory efficiency)
        two_hop: Include reactions connected via other metabolites
        
    Returns:
        tuple containing:
            - neighbor_embeddings: [batch_size, num_metabolites, max_neighbors, hidden_dim]
            - neighbor_mask: [batch_size, num_metabolites, max_neighbors] boolean validity mask
            
    Raises:
        AssertionError: If tensor shapes are invalid or indices are out of bounds
    """
    # Shape validation
    assert reaction_embeddings.dim() == 3, \
        f"reaction_embeddings should be 3D [batch_size, num_nodes, hidden_dim], got {reaction_embeddings.dim()}D with shape {reaction_embeddings.shape}"
    assert edge_index.dim() == 2 and edge_index.size(0) == 2, \
        f"edge_index should be [2, num_edges], got shape {edge_index.shape}"
    
    batch_size = reaction_embeddings.shape[0]
    num_nodes = reaction_embeddings.shape[1]
    num_metabolites = len(metabolite_indices)
    hidden_dim = reaction_embeddings.shape[-1]
    
    # Validate indices are within bounds
    assert max(metabolite_indices) < num_nodes, \
        f"Max metabolite index {max(metabolite_indices)} >= num_nodes {num_nodes}"
    assert max(reaction_indices) < num_nodes, \
        f"Max reaction index {max(reaction_indices)} >= num_nodes {num_nodes}"
    
    # Convert to numpy for efficient processing
    metabolite_set = set(metabolite_indices.cpu().numpy().tolist())
    reaction_set = set(reaction_indices.cpu().numpy().tolist())
    edge_list = edge_index.cpu().numpy()
    
    # Build neighbor dictionary for each metabolite
    neighbors_dict = {}
    
    for met_idx in metabolite_indices.cpu().numpy():
        direct_neighbors = _find_direct_reaction_neighbors(
            met_idx, edge_list, reaction_set
        )
        
        # Add 2-hop neighbors if requested
        if two_hop:
            two_hop_neighbors = _find_two_hop_reaction_neighbors(
                met_idx, direct_neighbors, edge_list, metabolite_set, reaction_set
            )
            direct_neighbors.extend(two_hop_neighbors)
        
        # Remove duplicates and limit to max_neighbors
        neighbors = list(set(direct_neighbors))[:max_neighbors]
        neighbors_dict[met_idx] = neighbors
    
    # Build tensors
    device = reaction_embeddings.device
    neighbor_embeddings = torch.zeros(batch_size, num_metabolites, max_neighbors, hidden_dim, device=device)
    neighbor_mask = torch.zeros(batch_size, num_metabolites, max_neighbors, dtype=torch.bool, device=device)
    
    for i, met_idx in enumerate(metabolite_indices.cpu().numpy()):
        neighbors = neighbors_dict[met_idx]
        
        for j, neighbor_idx in enumerate(neighbors):
            if j >= max_neighbors:
                break
                
            # Extract embeddings for all batches
            neighbor_embeddings[:, i, j, :] = reaction_embeddings[:, neighbor_idx, :]
            neighbor_mask[:, i, j] = True
    
    return neighbor_embeddings, neighbor_mask


def _find_direct_reaction_neighbors(metabolite_idx: int, edge_list, reaction_set) -> list[int]:
    """Find reactions directly connected to a metabolite."""
    neighbors = []
    for i in range(edge_list.shape[1]):
        src, tgt = edge_list[0, i], edge_list[1, i]
        
        # Check both directions: metabolite -> reaction or reaction -> metabolite
        if src == metabolite_idx and tgt in reaction_set:
            neighbors.append(tgt)
        elif tgt == metabolite_idx and src in reaction_set:
            neighbors.append(src)
    
    return neighbors


def _find_two_hop_reaction_neighbors(metabolite_idx: int, direct_neighbors: list[int],
                                   edge_list, metabolite_set, reaction_set) -> list[int]:
    """Find reactions connected via metabolite-reaction-metabolite-reaction paths."""
    two_hop_neighbors = set()
    
    for reaction_neighbor in direct_neighbors:
        # Find metabolites connected to this reaction (excluding the original)
        connected_metabolites = _find_connected_metabolites(
            reaction_neighbor, edge_list, metabolite_set, exclude=metabolite_idx
        )
        
        # For each connected metabolite, find its reaction neighbors
        for other_metabolite in connected_metabolites:
            reaction_neighbors = _find_direct_reaction_neighbors(
                other_metabolite, edge_list, reaction_set
            )
            two_hop_neighbors.update(reaction_neighbors)
    
    return list(two_hop_neighbors)


def _find_connected_metabolites(reaction_idx: int, edge_list, metabolite_set, exclude: int) -> list[int]:
    """Find metabolites connected to a reaction, excluding specified metabolite."""
    connected = []
    for i in range(edge_list.shape[1]):
        src, tgt = edge_list[0, i], edge_list[1, i]
        
        if src == reaction_idx and tgt in metabolite_set and tgt != exclude:
            connected.append(tgt)
        elif tgt == reaction_idx and src in metabolite_set and src != exclude:
            connected.append(src)
    
    return connected


def verify_no_metabolite_specific_parameters(module: nn.Module, num_metabolites: int) -> bool:
    """Verify MCR module has no metabolite-specific parameters to prevent ID leakage.
    
    Checks that no parameter tensor has a dimension matching the number of metabolites,
    which would indicate metabolite-specific weights that could lead to identifier leakage.
    
    Args:
        module: MCR module to validate
        num_metabolites: Total number of metabolites in dataset
        
    Returns:
        bool: True if module passes validation, False if ID leakage detected
    """
    problematic_params = []
    
    for param_name, param_tensor in module.named_parameters():
        if num_metabolites in param_tensor.shape:
            problematic_params.append((param_name, param_tensor.shape))
    
    if problematic_params:
        print("⚠️  WARNING: Potential ID leakage detected in MCR parameters:")
        for name, shape in problematic_params:
            print(f"    {name}: {shape} (matches num_metabolites={num_metabolites})")
        return False
    
    print("✅ MCR parameter validation passed - no ID leakage detected")
    return True


def benchmark_mcr_performance(mcr_model: MetaboliteConditionedReader,
                             batch_size: int = 1,
                             num_metabolites: int = 50,
                             max_neighbors: int = 10,
                             smiles_dim: int = 768,
                             hidden_dim: int = 128,
                             num_iterations: int = 100,
                             device: str = 'cuda') -> Dict[str, float]:
    """
    Benchmark MCR performance to validate optimizations.
    
    Returns timing results for different batch sizes and configurations.
    """
    import time
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    mcr_model = mcr_model.to(device)
    mcr_model.eval()
    
    # Create synthetic test data
    metabolite_smiles = torch.randn(batch_size, num_metabolites, smiles_dim, device=device)
    reaction_neighbors = torch.randn(batch_size, num_metabolites, max_neighbors, hidden_dim, device=device)
    neighbor_mask = torch.ones(batch_size, num_metabolites, max_neighbors, dtype=torch.bool, device=device)
    
    # Warm-up runs
    with torch.no_grad():
        for _ in range(10):
            _ = mcr_model(metabolite_smiles, reaction_neighbors, neighbor_mask)
    
    # Benchmark standard forward pass
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_iterations):
            predictions = mcr_model(metabolite_smiles, reaction_neighbors, neighbor_mask)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    standard_time = (time.time() - start_time) / num_iterations
    
    # Benchmark optimized single-sample path (if batch_size=1)
    optimized_time = None
    if batch_size == 1:
        single_smiles = metabolite_smiles.squeeze(0)
        single_neighbors = reaction_neighbors.squeeze(0)
        single_mask = neighbor_mask.squeeze(0)
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_iterations):
                predictions = mcr_model.forward_single_optimized(single_smiles, single_neighbors, single_mask)
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        optimized_time = (time.time() - start_time) / num_iterations
    
    results = {
        'standard_time_ms': standard_time * 1000,
        'samples_per_second': batch_size / standard_time,
    }
    
    if optimized_time is not None:
        results.update({
            'optimized_time_ms': optimized_time * 1000,
            'speedup_factor': standard_time / optimized_time,
            'optimized_samples_per_second': 1.0 / optimized_time
        })
    
    return results


def shuffle_metabolite_descriptors(x_smiles: torch.Tensor, 
                                  metabolite_indices: torch.Tensor,
                                  seed: Optional[int] = None) -> torch.Tensor:
    """Shuffle SMILES embeddings across metabolites for ablation studies.
    
    Breaks the association between metabolites and their chemical descriptors to test
    whether chemical similarity information contributes to MCR prediction performance.
    This is a key ablation to validate that MCR benefits from chemical awareness.
    
    Args:
        x_smiles: [batch_size, num_nodes, smiles_dim] Original SMILES embeddings
        metabolite_indices: [num_metabolites] Indices of metabolite nodes
        seed: Random seed for reproducible shuffling
        
    Returns:
        torch.Tensor: [batch_size, num_nodes, smiles_dim] SMILES with shuffled metabolite descriptors
        
    Note:
        Shuffling is performed independently for each batch to maintain batch structure.
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    x_smiles_shuffled = x_smiles.clone()
    metabolite_smiles = x_smiles[:, metabolite_indices, :]
    
    # Apply random permutation independently to each batch
    for batch_idx in range(metabolite_smiles.shape[0]):
        permutation = torch.randperm(metabolite_smiles.shape[1], device=x_smiles.device)
        metabolite_smiles[batch_idx] = metabolite_smiles[batch_idx][permutation]
    
    # Update the original tensor with shuffled metabolite embeddings
    x_smiles_shuffled[:, metabolite_indices, :] = metabolite_smiles
    
    return x_smiles_shuffled
