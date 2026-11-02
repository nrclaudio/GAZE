"""
Graph Positional Encodings (GPE): Explicit topology features for metabolic graphs.

Computes per-node structural features from graph topology:
- Degree features (4 scalars): in-degree, out-degree, total degree, log(degree+1)
- Random-Walk Structural Encodings (RWSE) (K scalars): diagonal of RW^1..RW^K

RWSE captures local structure (cycles, clustering) without eigendecomposition
sign ambiguity issues that plague Laplacian positional encodings.

Total GPE dim = 4 + K (default K=16, so 20 per node).
Computed once at dataset init time.
"""

import torch
import numpy as np
from typing import Optional


class GraphPositionalEncoder:
    """Computes graph positional/structural encodings from edge_index.

    Usage:
        encoder = GraphPositionalEncoder(rwse_steps=16)
        x_gpe = encoder(edge_index, num_nodes)  # [num_nodes, 20]
    """

    def __init__(self, rwse_steps: int = 16):
        self.rwse_steps = rwse_steps

    @property
    def output_dim(self) -> int:
        """Total dimension of GPE features: 4 degree + rwse_steps."""
        return 4 + self.rwse_steps

    def __call__(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        node_types: Optional[list] = None,
    ) -> torch.Tensor:
        """Compute GPE features for all nodes.

        Args:
            edge_index: [2, num_edges] graph connectivity
            num_nodes: total number of nodes
            node_types: optional list of node type strings (unused, reserved)

        Returns:
            x_gpe: [num_nodes, output_dim] positional/structural features
        """
        device = edge_index.device
        src, dst = edge_index[0], edge_index[1]

        # --- Degree features (4 scalars) ---
        in_degree = torch.zeros(num_nodes, device=device)
        out_degree = torch.zeros(num_nodes, device=device)

        # Count in-degree and out-degree
        in_degree.scatter_add_(0, dst, torch.ones(dst.size(0), device=device))
        out_degree.scatter_add_(0, src, torch.ones(src.size(0), device=device))

        total_degree = in_degree + out_degree
        log_degree = torch.log(total_degree + 1.0)

        degree_features = torch.stack([in_degree, out_degree, total_degree, log_degree], dim=-1)
        # Normalize degree features to zero mean, unit variance for stable training
        degree_features = self._normalize(degree_features)

        # --- RWSE features (rwse_steps scalars) ---
        rwse_features = self._compute_rwse(edge_index, num_nodes)

        # Concatenate: [num_nodes, 4 + rwse_steps]
        x_gpe = torch.cat([degree_features, rwse_features], dim=-1)

        return x_gpe

    def _compute_rwse(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Compute Random-Walk Structural Encodings.

        RWSE[i, k] = (A_rw^k)[i, i] = probability of returning to node i after k steps.

        This captures:
        - k=1: self-loops (usually 0 in our graph)
        - k=2: proportion of neighbors that are mutual
        - k=3: triangles through node
        - k>3: higher-order local structure

        We use the row-normalized adjacency (random walk) matrix.
        """
        device = edge_index.device
        src, dst = edge_index[0], edge_index[1]

        # Build sparse adjacency matrix
        num_edges = src.size(0)
        ones = torch.ones(num_edges, device=device)
        adj = torch.sparse_coo_tensor(
            torch.stack([dst, src]),  # transpose: A[dst, src] = 1
            ones,
            size=(num_nodes, num_nodes),
        ).coalesce()

        # Row-normalize to get random walk matrix: D^{-1} A
        # Compute row sums (out-degree for transposed = in-degree for original)
        row_sum = torch.sparse.sum(adj, dim=1).to_dense()
        row_sum = row_sum.clamp(min=1.0)  # Avoid division by zero for isolated nodes

        # Create D^{-1} as diagonal
        inv_degree = 1.0 / row_sum

        # We'll iterate: p = D^{-1} A p, tracking diagonal
        # Start with identity (each node starts at itself)
        # p_k[i] = (RW^k)[i, i]

        rwse = torch.zeros(num_nodes, self.rwse_steps, device=device)

        # Current distribution: each node has probability 1 at itself
        # We represent the full matrix as dense columns would be expensive,
        # so we use sparse matrix powers applied to identity columns.
        #
        # More efficient: iterate RW matrix multiplication on a diagonal tracker.
        # p^(0) = I (identity), p^(k) = D^{-1} A p^{k-1}
        # We only need the diagonal: trace of RW^k
        #
        # For efficiency with sparse ops:
        # current = I as sparse, multiply by RW, extract diagonal

        # Use batched sparse matmul approach for reasonable num_nodes
        # For metabolic graphs (~3000 nodes), dense is fine
        if num_nodes <= 10000:
            # Dense approach - fast for moderate graphs
            adj_dense = adj.to_dense()
            # Row-normalize
            rw = adj_dense * inv_degree.unsqueeze(1)

            # Iteratively compute powers and extract diagonal
            rw_power = rw.clone()
            for k in range(self.rwse_steps):
                rwse[:, k] = rw_power.diagonal()
                if k < self.rwse_steps - 1:
                    rw_power = rw_power @ rw
        else:
            # Sparse approach for very large graphs
            # Build sparse RW matrix
            adj_indices = adj.indices()
            adj_values = adj.values()
            rw_values = adj_values * inv_degree[adj_indices[0]]
            rw_sparse = torch.sparse_coo_tensor(
                adj_indices, rw_values, (num_nodes, num_nodes)
            ).coalesce()

            # Track diagonal through repeated sparse-dense multiplication
            # Use a dense "probe" matrix that's just the identity
            probe = torch.eye(num_nodes, device=device)
            for k in range(self.rwse_steps):
                probe = torch.sparse.mm(rw_sparse, probe)
                rwse[:, k] = probe.diagonal()

        return rwse

    @staticmethod
    def _normalize(features: torch.Tensor) -> torch.Tensor:
        """Z-score normalize features (zero mean, unit variance)."""
        mean = features.mean(dim=0, keepdim=True)
        std = features.std(dim=0, keepdim=True).clamp(min=1e-6)
        return (features - mean) / std
