"""Helpers to construct models from configuration objects."""

from __future__ import annotations

from typing import Optional

from gaze._reference.config import ModelConfig
from .concat_gnn_smiles import ConcatGNNWithSMILES
from .enhanced_concat_gnn_smiles import EnhancedConcatGNNWithSMILES
from .components import EnhancedModelConfig


def resolve_enhanced_config(raw_config: Optional[dict | EnhancedModelConfig]) -> Optional[EnhancedModelConfig]:
    """Return an EnhancedModelConfig instance from serialized data."""
    if raw_config is None:
        return None
    if isinstance(raw_config, EnhancedModelConfig):
        return raw_config
    return EnhancedModelConfig.from_dict(raw_config)


def create_model(
    model_cfg: ModelConfig,
    *,
    embedding_dim: int,
    gene_dim: int,
    smiles_dim: int,
) -> ConcatGNNWithSMILES | EnhancedConcatGNNWithSMILES:
    """Instantiate the appropriate model variant from configuration."""
    hidden_dim = model_cfg.hidden_dim or 128
    num_layers = model_cfg.num_layers or 3
    dropout = model_cfg.dropout if model_cfg.dropout is not None else 0.2

    if model_cfg.variant == 'enhanced':
        enhanced_config = resolve_enhanced_config(model_cfg.enhanced_config)
        return EnhancedConcatGNNWithSMILES(
            embedding_dim=embedding_dim,
            gene_dim=gene_dim,
            smiles_dim=smiles_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            config=enhanced_config,
            use_flux_messages=model_cfg.use_flux_messages,
            flux_message_depth=model_cfg.flux_message_depth,
            use_mass_balance_projection=model_cfg.use_mass_balance_projection,
            subtract_sample_mean=model_cfg.subtract_sample_mean,
        )

    return ConcatGNNWithSMILES(
        embedding_dim=embedding_dim,
        gene_dim=gene_dim,
        smiles_dim=smiles_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        use_directional=model_cfg.use_directionality,
        use_flux_messages=model_cfg.use_flux_messages,
        flux_message_depth=model_cfg.flux_message_depth,
        use_mass_balance_projection=model_cfg.use_mass_balance_projection,
    )
