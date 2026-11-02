"""Reusable building blocks for Concat-style models."""

from .builders import (
    build_encoders,
    build_flux_head,
    build_metabolite_head,
    build_gnn_layers,
    MCRReaderConfig,
)
from .met_reader import (
    MetaboliteConditionedReader,
    precompute_neighbor_indices,
    precompute_neighbor_indices_multihop,
    extract_reaction_neighbors,
    extract_reaction_neighbors_cached,
    extract_reaction_neighbors_cached_multihop,
    verify_no_metabolite_specific_parameters,
    shuffle_metabolite_descriptors,
    benchmark_mcr_performance,
)
from .enhanced_config import EnhancedModelConfig
from .base import BaseConcatModel

__all__ = [
    "build_encoders",
    "build_flux_head",
    "build_metabolite_head",
    "build_gnn_layers",
    "MCRReaderConfig",
    "MetaboliteConditionedReader",
    "precompute_neighbor_indices",
    "precompute_neighbor_indices_multihop",
    "extract_reaction_neighbors",
    "extract_reaction_neighbors_cached",
    "extract_reaction_neighbors_cached_multihop",
    "verify_no_metabolite_specific_parameters",
    "shuffle_metabolite_descriptors",
    "benchmark_mcr_performance",
    "EnhancedModelConfig",
    "BaseConcatModel",
]
