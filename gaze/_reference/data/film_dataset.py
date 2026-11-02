import torch
import numpy as np
import mudata as mu
import pandas as pd
from torch_geometric.data import Data, Dataset
import json

from ..utils.gene_parsing import parse_gene_rule_simple as parse_gene_rule
from .metabolite_mapping import metabolite_feature_nodes, dense_row

class FiLMDataset(Dataset):
    def __init__(self, mudata_path, embeddings_graph_path, gene_mapping_path):
        """
        Dataset for FiLM-modulated GNN that combines enzyme embeddings with gene expression
        
        Args:
            mudata_path: Path to mudata file with RNA expression
            embeddings_graph_path: Path to the graph with enzyme embeddings
            gene_mapping_path: Path to gene_mapping.json for gene feature positions
        """
        super().__init__()
        
        # Load the mudata file
        print(f"Loading MuData from {mudata_path}")
        self.mudata = mu.read_h5mu(mudata_path)
        self.num_samples = self.mudata.n_obs
        
        # Load the graph with enzyme embeddings
        print(f"Loading embeddings graph from {embeddings_graph_path}")
        self.graph = torch.load(embeddings_graph_path, weights_only=False)
        
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
        
        self.feature_nodes = metabolite_feature_nodes(
            self.mudata['metabolomics'].var, self.hmdb_to_idx, self.kegg_to_idx)
        self.excluded_target_nodes = set()

        # Load gene mapping for gene expression features
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
        
        # Store embedding dimension and gene dimension
        self.embedding_dim = self.graph.x.size(1)
        self.gene_dim = len(self.gene_to_index)
        print(f"Embedding dimension: {self.embedding_dim}, Gene dimension: {self.gene_dim}")
        
    def len(self):
        return self.num_samples
    
    def _targets_for_values(self, values):
        y = torch.full((len(self.node_types),), float('nan'))
        mask = torch.zeros(len(self.node_types), dtype=torch.bool)
        valid = np.isfinite(values) & (self.feature_nodes >= 0)
        nodes = self.feature_nodes[valid]
        if len(nodes):
            node_tensor = torch.as_tensor(nodes, dtype=torch.long)
            y[node_tensor] = torch.as_tensor(values[valid], dtype=torch.float32)
            mask[node_tensor] = True
        if self.excluded_target_nodes:
            excluded = torch.tensor(sorted(self.excluded_target_nodes), dtype=torch.long)
            y[excluded] = float('nan')
            mask[excluded] = False
        return y, mask

    def get(self, idx):
        """Get a single sample with both embeddings and gene expression features"""
        # Start with graph structure from the merged graph
        data = Data()
        data.edge_index = self.graph.edge_index
        
        # Explicitly set num_nodes attribute
        data.num_nodes = len(self.node_types)
        
        if hasattr(self.graph, 'edge_attr'):
            data.edge_attr = self.graph.edge_attr
        
        # Clone the enzyme embeddings for all nodes (these are the base features)
        data.x_embedding = self.graph.x.clone()
        
        # Initialize gene expression features (all zeros by default)
        data.x_gene = torch.zeros(len(self.node_types), self.gene_dim)
        
        # Fill gene expression features for reaction nodes
        rna_data = dense_row(self.mudata['rna'][idx].X)
        
        for r_idx in self.reaction_indices:
            gene_rule = self.graph.gene_rules[r_idx]
            genes = parse_gene_rule(gene_rule)
            
            # Track which genes are measured to avoid NaNs
            measured_genes = 0
            
            # Fill in expression values for measured genes
            for gene_id in genes:
                if gene_id in self.gene_to_index and gene_id in self.gene_id_to_expression_idx:
                    gene_feature_idx = self.gene_to_index[gene_id]
                    expression_idx = self.gene_id_to_expression_idx[gene_id]
                    
                    # Only assign non-NaN values
                    value = rna_data[expression_idx]
                    if not np.isnan(value):
                        data.x_gene[r_idx, gene_feature_idx] = float(value)
                        measured_genes += 1
        
        # Set up metabolite values and mask for this sample
        metabolomics_data = dense_row(self.mudata['metabolomics'][idx].X)
        data.y, data.mask = self._targets_for_values(metabolomics_data)
        data.measurable_mask = self.measurable_mask
        data.metabolite_indices = torch.tensor(self.metabolite_indices)
        data.reaction_indices = torch.tensor(self.reaction_indices)
        data.node_types = self.node_types  # Keep as list for easier inspection
        
        # ✅ ADD MISSING GRAPH ATTRIBUTES for enhanced physics loss
        data.gene_rules = self.graph.gene_rules
        data.gene_to_index = self.gene_to_index  # For gene rule evaluation
        data.gene_id_to_expression_idx = self.gene_id_to_expression_idx  # For finding expression data
        
        # 🚀 OPTIMIZATION: Copy tensor operations from dataset graph to training sample
        if hasattr(self.graph, 'reaction_gene_indices'):
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
