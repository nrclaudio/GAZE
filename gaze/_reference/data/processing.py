import mudata as md
import numpy as np
import anndata as ad
import pandas as pd
import torch
from typing import Dict, List, Tuple

from ..utils.gene_parsing import parse_gene_rule


def create_mudata_object(rna_data, metab_data, sample_names):
    """Create MuData object from RNA and metabolite matrices"""
    rna_adata = ad.AnnData(
        X=rna_data,
        obs=pd.DataFrame(index=sample_names),
        var=pd.DataFrame(index=[f"Gene_{i}" for i in range(rna_data.shape[1])])
    )
    
    metab_adata = ad.AnnData(
        X=metab_data,
        obs=pd.DataFrame(index=sample_names),
        var=pd.DataFrame(index=[f"Metab_{i}" for i in range(metab_data.shape[1])])
    )
    
    return md.MuData({'rna': rna_adata, 'metabolomics': metab_data})

def validate_metabolite_measurements(mudata, min_samples=0.5):
    """Filter metabolites with insufficient measurements"""
    metab_data = mudata.mod['metabolomics']
    n_samples = metab_data.shape[0]
    valid_metabs = (np.sum(~np.isnan(metab_data.X), axis=0) / n_samples) >= min_samples
    mudata.mod['metabolomics'] = metab_data[:, valid_metabs].copy()
    return mudata 



# parse_gene_rule is imported from gaze._reference.utils.gene_parsing


def add_tensor_gene_operations(data, gene_to_index: Dict[str, int], max_genes_per_reaction: int = 10):
    """
    🚀 AUTO-OPTIMIZATION: Add pre-computed gene operations to graph during loading.
    
    This function automatically converts string-based gene rules into tensor operations
    when the graph is loaded, eliminating the need for separate preprocessing.
    
    Args:
        data: PyG Data object with gene_rules
        gene_to_index: Mapping from gene ID to feature index
        max_genes_per_reaction: Maximum number of genes to store per reaction
        
    Returns:
        Updated data object with tensor-based gene operations
    """
    # Skip if already has tensor operations
    if hasattr(data, 'reaction_gene_indices'):
        print(f"    ✅ Graph already has tensor-based gene operations")
        return data
    
    # Skip if no gene rules
    if not hasattr(data, 'gene_rules') or not hasattr(data, 'node_types'):
        print(f"    ⚠️  Graph missing gene_rules or node_types, skipping tensor optimization")
        return data
    
    print(f"    🚀 Adding tensor-based gene operations for faster correlation...")
    print(f"    📊 Gene mapping contains {len(gene_to_index)} genes")
    
    # Find reaction nodes
    reaction_indices = []
    for i in range(data.num_nodes):
        if data.node_types[i] == 'reaction':
            reaction_indices.append(i)
    
    if not reaction_indices:
        print(f"    ⚠️  No reaction nodes found, skipping tensor optimization")
        return data
    
    print(f"    📊 Processing {len(reaction_indices)} reaction nodes")
    
    # Initialize tensors
    num_reactions = len(reaction_indices)
    
    # reaction_gene_indices: [num_reactions, max_genes] - gene feature indices (-1 for padding)
    reaction_gene_indices = torch.full((num_reactions, max_genes_per_reaction), -1, dtype=torch.long)
    
    # reaction_gene_operations: [num_reactions] - operation codes (0=simple/mean, 1=and/min, 2=or/max)
    reaction_gene_operations = torch.zeros(num_reactions, dtype=torch.long)
    
    # reaction_gene_masks: [num_reactions, max_genes] - True for valid genes
    reaction_gene_masks = torch.zeros((num_reactions, max_genes_per_reaction), dtype=torch.bool)
    
    # reaction_node_indices: [num_reactions] - maps back to original node indices
    reaction_node_indices = torch.tensor(reaction_indices, dtype=torch.long)
    
    valid_reactions = 0
    skipped_no_genes = 0
    skipped_too_many_genes = 0
    
    for r_idx, node_idx in enumerate(reaction_indices):
        gene_rule = data.gene_rules[node_idx] if hasattr(data, 'gene_rules') else ''
        
        if not gene_rule or gene_rule.strip() == '':
            skipped_no_genes += 1
            continue
            
        # Parse gene rule
        genes, operation = parse_gene_rule(gene_rule)
        
        if not genes:
            skipped_no_genes += 1
            continue
            
        if len(genes) > max_genes_per_reaction:
            genes = genes[:max_genes_per_reaction]
            skipped_too_many_genes += 1
        
        # Convert operation to code
        operation_code = {'simple': 0, 'and': 1, 'or': 2}.get(operation, 0)
        reaction_gene_operations[r_idx] = operation_code
        
        # Map genes to indices
        valid_gene_count = 0
        for gene_id in genes:
            if gene_id in gene_to_index:
                feature_idx = gene_to_index[gene_id]
                reaction_gene_indices[r_idx, valid_gene_count] = feature_idx
                reaction_gene_masks[r_idx, valid_gene_count] = True
                valid_gene_count += 1
        
        if valid_gene_count > 0:
            valid_reactions += 1
    
    print(f"    ✅ Tensor operations added: {valid_reactions} valid reactions")
    if skipped_no_genes > 0:
        print(f"    ⚠️  Skipped {skipped_no_genes} reactions with no genes")
    if skipped_too_many_genes > 0:
        print(f"    ⚠️  Truncated {skipped_too_many_genes} reactions with >10 genes")
    
    # Add tensors to data object
    data.reaction_gene_indices = reaction_gene_indices
    data.reaction_gene_operations = reaction_gene_operations  
    data.reaction_gene_masks = reaction_gene_masks
    data.reaction_node_indices = reaction_node_indices
    
    # Store metadata
    data.max_genes_per_reaction = max_genes_per_reaction
    data.num_valid_reactions = valid_reactions
    
    # Verify tensors were added
    print(f"    🔍 Verification - added tensors:")
    print(f"      - reaction_gene_indices: {hasattr(data, 'reaction_gene_indices')} {data.reaction_gene_indices.shape if hasattr(data, 'reaction_gene_indices') else 'N/A'}")
    print(f"      - reaction_gene_operations: {hasattr(data, 'reaction_gene_operations')} {data.reaction_gene_operations.shape if hasattr(data, 'reaction_gene_operations') else 'N/A'}")
    print(f"      - reaction_gene_masks: {hasattr(data, 'reaction_gene_masks')} {data.reaction_gene_masks.shape if hasattr(data, 'reaction_gene_masks') else 'N/A'}")
    
    return data


def apply_filters(metabolites_data, filters):
    """Apply filters to metabolites data"""
    result = metabolites_data
    for filter_func in filters:
        result = filter_func(result)
    return result 