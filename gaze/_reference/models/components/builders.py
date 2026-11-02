"""Factory helpers for assembling Concat-style model components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch.nn as nn
from torch_geometric.nn import GATv2Conv

from ..modules.directional_conv import DirectionalConv
from ..modules.pna_layer import PNALayer
from ..modules.relational_conv import RelationalMetabolicConv
from .met_reader import MetaboliteConditionedReader
from ..modules.flux_messages import AdditiveFluxMessageLayer


@dataclass
class MCRReaderConfig:
    """Configuration for the metabolite-conditioned reader."""

    smiles_dim: int
    hidden_dim: int
    mlp_hidden: int
    normalize_descriptors: bool = True
    dropout: float = 0.2
    neighbor_dropout_rate: float = 0.0
    # Minimum number of neighbors to keep after stochastic neighbor dropout.
    # If a metabolite has <= this many valid neighbors, no neighbor dropout is applied to it.
    neighbor_min_keep: int = 1
    # Multi-hop MCR parameters
    max_hops: int = 1
    hop_decay_init: float = -0.5
    # Gene expression skip connection
    use_gene_skip: bool = False
    gene_dim: int = 0  # Required when use_gene_skip=True


def build_encoders(embedding_dim: int, gene_dim: int, smiles_dim: int, hidden_dim: int) -> tuple[nn.Linear, nn.Linear]:
    """Create reaction and metabolite encoders."""
    combined_dim = embedding_dim + gene_dim
    reaction_encoder = nn.Linear(combined_dim, hidden_dim)
    metabolite_encoder = nn.Linear(smiles_dim, hidden_dim)
    return reaction_encoder, metabolite_encoder


def build_flux_head(hidden_dim: int) -> nn.Sequential:
    """Create the reaction flux prediction head used by physics-informed losses."""
    return nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim // 2),
        nn.ReLU(),
        nn.Linear(hidden_dim // 2, 1),
        nn.Softplus(),
    )


def build_detection_gate(hidden_dim: int, dropout: float = 0.2) -> nn.Sequential:
    """Detection gate MLP predicting P(metabolite detected).

    Outputs raw logits (no sigmoid). Sigmoid is applied externally:
    - ``torch.sigmoid(logit)`` for gating the concentration prediction
    - ``F.binary_cross_entropy_with_logits(logit, target)`` for the loss
    This split provides better numerical stability.
    """
    return nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim // 2),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim // 2, 1),
    )


def build_metabolite_head(
    *,
    use_mcr: bool,
    smiles_dim: int,
    hidden_dim: int,
    dropout: float,
    reader_config: Optional[MCRReaderConfig] = None,
) -> tuple[Optional[MetaboliteConditionedReader], Optional[nn.Module]]:
    """Create either the MCR reader or a standard prediction MLP."""
    if use_mcr:
        if reader_config is None:
            reader_config = MCRReaderConfig(
                smiles_dim=smiles_dim,
                hidden_dim=hidden_dim,
                mlp_hidden=hidden_dim,
                normalize_descriptors=True,
                dropout=dropout,
            )
        reader = MetaboliteConditionedReader(
            smiles_dim=smiles_dim,
            gnn_hidden_dim=hidden_dim,
            hidden_dim=reader_config.hidden_dim,
            mlp_hidden=reader_config.mlp_hidden,
            normalize_descriptors=reader_config.normalize_descriptors,
            dropout=reader_config.dropout,
            neighbor_dropout_rate=reader_config.neighbor_dropout_rate,
            neighbor_min_keep=reader_config.neighbor_min_keep,
            max_hops=reader_config.max_hops,
            hop_decay_init=reader_config.hop_decay_init,
            use_gene_skip=reader_config.use_gene_skip,
            gene_dim=reader_config.gene_dim,
        )
        return reader, None

    predictor = nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim * 2),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim * 2, 1),
    )
    return None, predictor


def build_flux_message_layer(hidden_dim: int) -> AdditiveFluxMessageLayer:
    """Return the additive flux message layer with default settings."""
    return AdditiveFluxMessageLayer(
        hidden_dim=hidden_dim,
        compress_alpha=0.5,
        use_transport=True,
        transport_hops=1,
    )


def build_gnn_layers(
    *,
    num_layers: int,
    hidden_dim: int,
    dropout: float,
    use_directional: bool,
    directional_concat: bool,
    layer_type: str = "GAT",
    deg: Optional[torch.Tensor] = None,
    use_residual: bool = False,
    residual_learning_rate: float = 0.1,
    use_layer_norm: bool = True,
) -> nn.ModuleList:
    """Create a stack of GNN layers for the baseline model."""
    from ..modules.residual_layers import ResidualGNNLayer
    
    layers = nn.ModuleList()
    for _ in range(num_layers):
        if layer_type == "Relational":
            # Relational layer handles residuals and norms internally
            layer = RelationalMetabolicConv(
                hidden_dim=hidden_dim,
                heads=4,
                dropout=dropout,
                edge_dim=1
            )
        elif use_residual:
            # Note: Current ResidualGNNLayer only supports 'gat' internally
            # We might want to extend it to support other types if needed
            layer = ResidualGNNLayer(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                gnn_type='gat',
                edge_dim=1,
                heads=4,
                dropout=dropout,
                residual_learning_rate=residual_learning_rate,
                use_layer_norm=use_layer_norm
            )
        elif layer_type == "PNA":
            if deg is None:
                raise ValueError("Degree histogram 'deg' is required for PNA layers.")
            layer = PNALayer(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                deg=deg,
                edge_dim=1,
                dropout=dropout,
            )
        elif use_directional or layer_type == "Directional":
            layer = DirectionalConv(
                hidden_dim,
                hidden_dim,
                edge_dim=1,
                dropout=dropout,
                heads=4,
                concat=directional_concat,
            )
        else:
            layer = GATv2Conv(
                hidden_dim,
                hidden_dim,
                edge_dim=1,
                heads=4,
                concat=False,
                dropout=dropout,
            )
        layers.append(layer)
    return layers
