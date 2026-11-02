"""
Data path utilities for the concat-smiles-model repository.

This module provides centralized path management for all data files,
making the repository self-contained and portable.
"""

import os
from typing import Tuple


def get_project_root() -> str:
    """Get the root directory of the project."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up from src/utils/ to project root
    return os.path.dirname(os.path.dirname(current_dir))


def get_data_dir() -> str:
    """Get the data directory path."""
    return os.path.join(get_project_root(), 'data')


def get_mudata_path() -> str:
    """Get the path to the main mudata file."""
    return os.path.join(
        get_data_dir(), 
        'raw', 
        'mudata_correct_HMDB_new_no_negatives.h5mu'
    )


def get_gene_mapping_path() -> str:
    """Get the path to the gene mapping JSON file."""
    return os.path.join(
        get_data_dir(),
        'raw',
        'gene_mapping.json'
    )


def get_graph_path(normalized: bool = False, use_gene_features: bool = False, identifier: str = "") -> str:
    """
    Get the path to the metabolic network graph.
    
    Args:
        normalized: Whether to use the normalized version
        use_gene_features: Whether to use gene features instead of embeddings
        identifier: Optional suffix for directory (e.g. 'masked')
        
    Returns:
        Path to the appropriate graph file
    """
    suffix = f"_{identifier}" if identifier else ""
    
    if use_gene_features:
        graph_dir = f'PyG_merged_graph_with_SMILES_with_gene_features{suffix}'
    elif normalized:
        graph_dir = f'PyG_merged_graph_with_SMILES_with_embeddings_normalised{suffix}'
    else:
        graph_dir = f'PyG_merged_graph_with_SMILES_with_embeddings{suffix}'
    
    return os.path.join(
        get_data_dir(),
        'graphs',
        graph_dir,
        'merged_metabolic_network.pt'
    )


def get_standard_data_paths(normalized: bool = False, identifier: str = "") -> Tuple[str, str, str]:
    """
    Get the standard data paths used by most training scripts.
    
    Args:
        normalized: Whether to use normalized graph
        identifier: Optional suffix for graph directory
        
    Returns:
        Tuple of (mudata_path, graph_path, gene_mapping_path)
    """
    return (
        get_mudata_path(),
        get_graph_path(normalized=normalized, identifier=identifier),
        get_gene_mapping_path()
    )


def verify_data_files() -> bool:
    """
    Verify that all essential data files exist.
    
    Returns:
        True if all files exist, False otherwise
    """
    essential_files = [
        get_mudata_path(),
        get_gene_mapping_path(),
        get_graph_path(normalized=False),
        get_graph_path(normalized=True),
        get_graph_path(use_gene_features=True)
    ]
    
    for file_path in essential_files:
        if not os.path.exists(file_path):
            print(f"Missing data file: {file_path}")
            return False
    
    print("✅ All essential data files verified")
    return True


if __name__ == "__main__":
    # Quick verification when run as script
    print("Data paths verification:")
    print(f"Project root: {get_project_root()}")
    print(f"Data directory: {get_data_dir()}")
    print(f"MuData: {get_mudata_path()}")
    print(f"Gene mapping: {get_gene_mapping_path()}")
    print(f"Graph (standard): {get_graph_path()}")
    print(f"Graph (normalized): {get_graph_path(normalized=True)}")
    print(f"Graph (gene features): {get_graph_path(use_gene_features=True)}")
    print()
    verify_data_files()