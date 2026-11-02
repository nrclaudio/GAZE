import os
import torch
import numpy as np
import mudata as mu
import pandas as pd
import json
from torch_geometric.data import Data, Dataset
from .gpu_cached_graph import get_or_create_gpu_cache
from ..utils.gene_parsing import parse_gene_rule_simple as parse_gene_rule

class MetabolicNetworkDataset(Dataset):
    def __init__(self, mudata_path, graph_path, gene_mapping_path=None, use_embeddings=True, device=None):
        """
        Args:
            mudata_path: Path to mudata_correct_HMDB.h5mu
            graph_path: Path to the merged graph (.pt file)
            gene_mapping_path: Path to gene_mapping.json (required when use_embeddings=False)
            use_embeddings: Whether to use EC2Vec embeddings or gene features
            device: Target device for GPU caching (optional)
        """
        super().__init__()
        self.use_embeddings = use_embeddings
        
        # Load the mudata file
        print(f"Loading MuData from {mudata_path}")
        self.mudata = mu.read_h5mu(mudata_path)
        self.num_samples = self.mudata.n_obs
        print("Number of samples: ", self.num_samples)
        
        # Load the merged graph
        print(f"Loading graph from {graph_path}")
        self.graph = torch.load(graph_path, weights_only=False)
        
        # Extract node information
        self.node_types = self.graph.node_types
        self.metabolite_indices = [i for i, t in enumerate(self.node_types) if t == 'metabolite']
        self.reaction_indices = [i for i, t in enumerate(self.node_types) if t == 'reaction']

        # Static mask: True for metabolite nodes with a CAMP panel match
        if hasattr(self.graph, 'measured_flags'):
            self.measurable_mask = torch.tensor(
                [bool(self.graph.measured_flags[i]) for i in range(len(self.node_types))],
                dtype=torch.bool,
            )
        else:
            # Fallback for legacy graphs: treat all metabolites as measurable
            self.measurable_mask = torch.tensor(
                [t == 'metabolite' for t in self.node_types], dtype=torch.bool,
            )
        
        # Prepare metabolite ID mapping
        self.hmdb_to_idx = {}
        self.kegg_to_idx = {}
        
        for i in self.metabolite_indices:
            hmdb_id = self.graph.hmdb_ids[i]
            kegg_id = self.graph.kegg_ids[i]
            
            if hmdb_id:
                self.hmdb_to_idx[hmdb_id] = i
            if kegg_id:
                self.kegg_to_idx[kegg_id] = i
        
        # If using gene features, load gene mapping
        if not use_embeddings and gene_mapping_path:
            with open(gene_mapping_path, 'r') as f:
                mapping_data = json.load(f)
                self.gene_to_index = mapping_data['gene_to_index']
                
            # 🚀 AUTO-OPTIMIZATION: Add tensor-based gene operations for faster training
            from .processing import add_tensor_gene_operations
            self.graph = add_tensor_gene_operations(self.graph, self.gene_to_index)
                
            # Map RNA expression data to gene indices
            self.gene_id_to_expression_idx = {}
            for i, gene_id in enumerate(self.mudata['rna'].var['ensembl_gene_id']):
                self.gene_id_to_expression_idx[gene_id] = i
        
        # 🚀 OPTIMIZATION: Create GPU cache for static graph components
        self.gpu_cache = None
        if device is not None:
            self.gpu_cache = get_or_create_gpu_cache(self.graph, device)
            print(f"✅ Created GPU cache for static graph components on {device}")
        
    def len(self):
        return self.num_samples
    
    def get(self, idx):
        """Get a single sample with metabolite values and mask"""
        # 🚀 OPTIMIZATION: Use GPU cache if available, otherwise fallback to copy
        data = Data()
        
        if self.gpu_cache is not None:
            # HYBRID APPROACH: Get GPU tensors directly (avoid DataLoader transfer)
            static_data = self.gpu_cache.get_static_data(force_cpu=False)
            
            # Use cached GPU components (no copying needed!)
            data.edge_index = static_data['edge_index']
            if 'edge_attr' in static_data:
                data.edge_attr = static_data['edge_attr']
            if 'x_embedding' in static_data:
                data.x_embedding = static_data['x_embedding']
            if 'x_smiles' in static_data:
                data.x_smiles = static_data['x_smiles']
                
            # Add gene operation tensors for fast correlation
            if 'reaction_gene_indices' in static_data:
                data.reaction_gene_indices = static_data['reaction_gene_indices']
                data.reaction_gene_operations = static_data['reaction_gene_operations']
                data.reaction_gene_masks = static_data['reaction_gene_masks']
                data.reaction_node_indices = static_data['reaction_node_indices']
                data.max_genes_per_reaction = static_data['max_genes_per_reaction']
                data.num_valid_reactions = static_data['num_valid_reactions']
            
            # Mark that static components are already on GPU
            data._static_on_gpu = True
        else:
            # Fallback: Copy from graph (old path)
            data.edge_index = self.graph.edge_index
            
            if hasattr(self.graph, 'edge_attr'):
                data.edge_attr = self.graph.edge_attr
            
            data._static_on_gpu = False
        
        # Initialize node features (only if not using GPU cache)
        if self.gpu_cache is None:
            if self.use_embeddings:
                # If using embeddings, directly use the graph's features
                data.x = self.graph.x.clone()
            else:
                # If using gene features, update reaction features with current sample's RNA data
                data.x = torch.zeros_like(self.graph.x)
                
                # For non-reaction nodes, just copy the original features
                non_reaction_mask = torch.tensor([t != 'reaction' for t in self.node_types])
                data.x[non_reaction_mask] = self.graph.x[non_reaction_mask]
                
                # Update reaction nodes with RNA data from this sample
                rna_data = self.mudata['rna'][idx].X.toarray().squeeze()
                
                for r_idx in self.reaction_indices:
                    gene_rule = self.graph.gene_rules[r_idx]
                    genes = parse_gene_rule(gene_rule)
                    
                    # Fill in expression values for measured genes
                    for gene_id in genes:
                        if gene_id in self.gene_to_index and gene_id in self.gene_id_to_expression_idx:
                            gene_feature_idx = self.gene_to_index[gene_id]
                            expression_idx = self.gene_id_to_expression_idx[gene_id]
                            
                            # ADD THIS CHECK: Only assign non-NaN values
                            value = rna_data[expression_idx]
                            if not np.isnan(value):
                                data.x[r_idx, gene_feature_idx] = value
                            # If value is NaN, leave it as 0 (the default)
        
        # Set up metabolite values and mask for this sample
        metabolomics_data = self.mudata['metabolomics'][idx].X.toarray().squeeze()
        metabolite_ids = self.mudata['metabolomics'].var_names
        hmdb_ids = self.mudata['metabolomics'].var['HMDB_trailing_zeros']
        kegg_ids = self.mudata['metabolomics'].var['KEGG']
        
        # Create target values and mask
        # Initialize with NaN (will be ignored in loss computation)
        y = torch.full((len(self.node_types),), float('nan'))
        mask = torch.zeros(len(self.node_types), dtype=torch.bool)
        
        # Fill in measured metabolite values
        for i, (met_id, hmdb_id, kegg_id) in enumerate(zip(metabolite_ids, hmdb_ids, kegg_ids)):
            value = metabolomics_data[i]
            
            # Skip NaN values
            if np.isnan(value):
                continue
                
            # Try to find this metabolite in the graph
            graph_idx = None
            if hmdb_id in self.hmdb_to_idx:
                graph_idx = self.hmdb_to_idx[hmdb_id]
            elif kegg_id in self.kegg_to_idx:
                graph_idx = self.kegg_to_idx[kegg_id]
                
            if graph_idx is not None:
                y[graph_idx] = value
                mask[graph_idx] = True
        
        data.y = y
        data.mask = mask
        data.measurable_mask = self.measurable_mask
        data.metabolite_indices = torch.tensor(self.metabolite_indices)
        data.reaction_indices = torch.tensor(self.reaction_indices)
        data.node_types = self.node_types  # Keep as list for easier inspection
        
        # 🚀 OPTIMIZATION: Only copy tensor operations if not using GPU cache
        if self.gpu_cache is None and hasattr(self.graph, 'reaction_gene_indices'):
            data.reaction_gene_indices = self.graph.reaction_gene_indices.clone()
            data.reaction_gene_operations = self.graph.reaction_gene_operations.clone()
            data.reaction_gene_masks = self.graph.reaction_gene_masks.clone()
            data.reaction_node_indices = self.graph.reaction_node_indices.clone()
            # Also copy metadata
            if hasattr(self.graph, 'max_genes_per_reaction'):
                data.max_genes_per_reaction = self.graph.max_genes_per_reaction
            if hasattr(self.graph, 'num_valid_reactions'):
                data.num_valid_reactions = self.graph.num_valid_reactions
        
        return data


# parse_gene_rule is imported from gaze._reference.utils.gene_parsing
