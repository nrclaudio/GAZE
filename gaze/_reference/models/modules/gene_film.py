"""Gene-expression FiLM conditioning for GNN layers."""

import torch
import torch.nn as nn


class GeneExpressionFiLM(nn.Module):
    """Feature-wise Linear Modulation conditioned on raw gene expression.

    Applied after each GNN layer to reaction nodes:
        h_rxn = gamma(x_gene) * h_rxn + beta(x_gene)

    Identity-initialized: gamma -> 1, beta -> 0 at init, so the model
    starts identical to the non-FiLM version.
    """

    def __init__(self, gene_dim: int, hidden_dim: int):
        super().__init__()
        self.gamma_net = nn.Linear(gene_dim, hidden_dim)
        self.beta_net = nn.Linear(gene_dim, hidden_dim)
        # Identity init
        nn.init.zeros_(self.gamma_net.weight)
        nn.init.ones_(self.gamma_net.bias)
        nn.init.zeros_(self.beta_net.weight)
        nn.init.zeros_(self.beta_net.bias)

    def forward(self, h_rxn: torch.Tensor, x_gene_rxn: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma_net(x_gene_rxn)  # [num_rxn, hidden_dim]
        beta = self.beta_net(x_gene_rxn)
        return gamma * h_rxn + beta
