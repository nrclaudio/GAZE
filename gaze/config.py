"""GAZEConfig: minimal configuration dataclass for inference."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GAZEConfig:
    """Configuration for the GAZE deploy model.

    All deploy features are always-on; this dataclass captures their
    hyperparameters.

    Args:
        hidden_dim: Hidden dimension for GNN and attention layers.
        num_layers: Number of GNN (RelationalMetabolicConv) layers.
        dropout: Dropout probability.
        cross_modal_heads: Number of attention heads in CrossModalAttention.
        cross_modal_dropout: Dropout inside cross-modal attention.
        cross_modal_temperature: Softmax temperature for cross-modal attention.
        cross_modal_residual_weight: Blending weight for cross-modal residual.
        gat_heads: Number of heads in GATv2Conv layers.
        gat_edge_dim: Edge feature dimension for GAT layers.
        flux_message_depth: GNN layer index (1-indexed) for flux message injection.
        flux_compress_alpha: Exponent for flux dynamic range compression.
        flux_use_transport: Enable transport term in flux messages.
        flux_transport_hops: Number of transport hops.
        reader_hidden_dim: Hidden dim for MCR attention projections.
        reader_mlp_hidden: Hidden dim for MCR prediction MLP.
        reader_normalize_descriptors: L2-normalize SMILES before MCR attention.
        reader_max_neighbors: Max reaction neighbors per metabolite for MCR.
        structural_loss_weight: Weight for structural reconstruction loss.
        mass_balance_max_reg: Maximum regularization for Cholesky factorization.
    """

    # Core architecture
    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.2

    # Cross-modal attention
    cross_modal_heads: int = 4
    cross_modal_dropout: float = 0.1
    cross_modal_temperature: float = 1.0
    cross_modal_residual_weight: float = 0.5

    # GATv2 (inside RelationalMetabolicConv)
    gat_heads: int = 4
    gat_edge_dim: int = 1

    # Flux messages
    flux_message_depth: int = 1
    flux_compress_alpha: float = 0.5
    flux_use_transport: bool = True
    flux_transport_hops: int = 1

    # MCR reader
    reader_hidden_dim: int = 128
    reader_mlp_hidden: int = 128
    reader_normalize_descriptors: bool = True
    reader_max_neighbors: int = 10

    # Structural loss
    structural_loss_weight: float = 0.1

    # Mass balance
    mass_balance_max_reg: float = 1e-2
