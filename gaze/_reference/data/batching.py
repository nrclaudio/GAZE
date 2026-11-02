"""
Batching utilities for metabolic network datasets.

This module provides PyG-compatible batching for graphs with shared structure
but different sample-specific features (gene expression, metabolite values).
"""

import torch
from torch_geometric.data import Data, Batch
from typing import List, Dict, Any
import numpy as np


def create_pyg_batch_from_samples(data_list: List[Data]) -> Data:
    """
    Create a PyG Batch from samples that share the same graph structure.
    
    Key insight: All samples share the same metabolic network topology,
    so we can batch efficiently by stacking node features while reusing
    the same edge_index and edge_attr.
    
    Args:
        data_list: List of Data objects from MetabolicNetworkDataset
        
    Returns:
        Batched Data object with proper PyG structure
    """
    if not data_list:
        raise ValueError("Cannot create batch from empty data list")
    
    batch_size = len(data_list)
    first_sample = data_list[0]
    
    # Verify all samples have the same graph structure
    num_nodes = first_sample.x_embedding.size(0) if hasattr(first_sample, 'x_embedding') else first_sample.x.size(0)
    
    for i, data in enumerate(data_list[1:], 1):
        sample_nodes = data.x_embedding.size(0) if hasattr(data, 'x_embedding') else data.x.size(0)
        if sample_nodes != num_nodes:
            raise ValueError(f"Sample {i} has {sample_nodes} nodes, expected {num_nodes}")
    
    # Create batched data object
    batch = Data()
    
    # === SHARED GRAPH STRUCTURE (reused across all samples) ===
    batch.edge_index = first_sample.edge_index  # [2, num_edges]
    
    if hasattr(first_sample, 'pathway_edge_mask'):
        batch.pathway_edge_mask = first_sample.pathway_edge_mask
    
    if hasattr(first_sample, 'edge_attr') and first_sample.edge_attr is not None:
        batch.edge_attr = first_sample.edge_attr  # [num_edges, edge_features]
    
    # Node type information (shared)
    batch.node_types = first_sample.node_types
    batch.metabolite_indices = first_sample.metabolite_indices
    batch.reaction_indices = first_sample.reaction_indices
    if hasattr(first_sample, 'measurable_mask'):
        batch.measurable_mask = first_sample.measurable_mask
    
    # === BATCHED NODE FEATURES ===
    # Stack sample-specific features into batch dimension
    
    # Embeddings (static, but we stack for consistency)
    if hasattr(first_sample, 'x_embedding'):
        batch.x_embedding = torch.stack([data.x_embedding for data in data_list], dim=0)  # [batch_size, num_nodes, embed_dim]
    
    # Gene expression (varies per sample)
    if hasattr(first_sample, 'x_gene'):
        batch.x_gene = torch.stack([data.x_gene for data in data_list], dim=0)  # [batch_size, num_nodes, gene_dim]
    
    # SMILES embeddings (static, but we stack for consistency)  
    if hasattr(first_sample, 'x_smiles'):
        batch.x_smiles = torch.stack([data.x_smiles for data in data_list], dim=0)  # [batch_size, num_nodes, smiles_dim]
    
    # Legacy x features (fallback)
    if hasattr(first_sample, 'x') and not hasattr(first_sample, 'x_embedding'):
        batch.x = torch.stack([data.x for data in data_list], dim=0)  # [batch_size, num_nodes, features]
    
    # === BATCHED TARGETS AND MASKS ===
    batch.y = torch.stack([data.y for data in data_list], dim=0)  # [batch_size, num_nodes]
    batch.mask = torch.stack([data.mask for data in data_list], dim=0)  # [batch_size, num_nodes]

    # === PHYSICS COMPONENTS (shared structure) ===
    if hasattr(first_sample, 'incidence_matrix'):
        batch.incidence_matrix = first_sample.incidence_matrix
    
    # === TENSOR OPERATIONS FOR CORRELATION LOSS ===
    if hasattr(first_sample, 'reaction_gene_indices'):
        # These are shared across samples (same reaction-gene mappings)
        batch.reaction_gene_indices = first_sample.reaction_gene_indices
        batch.reaction_gene_operations = first_sample.reaction_gene_operations 
        batch.reaction_gene_masks = first_sample.reaction_gene_masks
        batch.reaction_node_indices = first_sample.reaction_node_indices
        batch.max_genes_per_reaction = first_sample.max_genes_per_reaction
        batch.num_valid_reactions = first_sample.num_valid_reactions
    
    # === SPATIAL COMPONENTS (if present) ===
    if hasattr(first_sample, 'xy'):
        batch.xy = torch.stack([data.xy for data in data_list], dim=0)  # [batch_size, num_spots, 2]
        
    if hasattr(first_sample, 'edge_index_spatial'):
        # Spatial edges might be shared or per-sample - handle both cases
        if first_sample.edge_index_spatial.dim() == 2:
            # Shared spatial structure
            batch.edge_index_spatial = first_sample.edge_index_spatial
        else:
            # Per-sample spatial structure  
            batch.edge_index_spatial = torch.stack([data.edge_index_spatial for data in data_list], dim=0)
    
    # === BATCH METADATA ===
    batch.batch_size = batch_size
    batch.num_nodes = num_nodes
    
    # Mark as GPU-cached if all samples are
    if hasattr(first_sample, '_static_on_gpu'):
        batch._static_on_gpu = all(getattr(data, '_static_on_gpu', False) for data in data_list)
    
    return batch


def collate_metabolic_data(data_list: List[Data]) -> Data:
    """
    Collate function for DataLoader to create proper batches.
    
    This is the main entry point used by PyTorch DataLoader.
    """
    result = create_pyg_batch_from_samples(data_list)
    return result


def extract_sample_predictions(batch_pred: torch.Tensor, batch_size: int, num_nodes: int) -> List[torch.Tensor]:
    """
    Extract individual sample predictions from batched tensor.
    
    Args:
        batch_pred: [batch_size, num_nodes] or [batch_size * num_nodes]
        batch_size: Number of samples in batch
        num_nodes: Number of nodes per sample
        
    Returns:
        List of [num_nodes] prediction tensors
    """
    if batch_pred.dim() == 1:
        # Reshape from flattened to batched
        batch_pred = batch_pred.view(batch_size, num_nodes)
    
    return [batch_pred[i] for i in range(batch_size)]


def unbatch_tensors(batched_tensor: torch.Tensor, batch_size: int) -> List[torch.Tensor]:
    """
    Unbatch a tensor along the first dimension.
    
    Args:
        batched_tensor: [batch_size, ...] tensor
        batch_size: Size of batch dimension
        
    Returns:
        List of tensors with batch dimension removed
    """
    return [batched_tensor[i] for i in range(batch_size)]