"""Structural consistency loss: forces manifold alignment between GNN embeddings
and chemical structure by reconstructing SMILES from aggregated reaction neighbors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeighborReconstructionLoss(nn.Module):
    """Predict a metabolite's SMILES embedding from its aggregated reaction neighbors.

    On a real graph the functional context (enzymes) should predict the chemistry;
    on a random graph the context is noise and the loss explodes.

    Args:
        hidden_dim: Dimension of GNN hidden embeddings.
        smiles_dim: Dimension of SMILES embeddings.
        project_dim: Optional projection dimension (defaults to *smiles_dim*).
    """

    def __init__(self, hidden_dim: int, smiles_dim: int, project_dim: int | None = None):
        super().__init__()
        output_dim = project_dim if project_dim is not None else smiles_dim

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, neighbor_messages: torch.Tensor,
                target_smiles: torch.Tensor) -> torch.Tensor:
        """Compute 1 - mean cosine similarity between decoded and target SMILES.

        Args:
            neighbor_messages: ``[M, hidden_dim]`` aggregated reaction features.
            target_smiles: ``[M, smiles_dim]`` ground-truth SMILES embeddings.

        Returns:
            Scalar loss.
        """
        predicted = self.decoder(neighbor_messages)
        pred_norm = F.normalize(predicted, p=2, dim=-1)
        target_norm = F.normalize(target_smiles, p=2, dim=-1)
        return 1.0 - (pred_norm * target_norm).sum(dim=-1).mean()
