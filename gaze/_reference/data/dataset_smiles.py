import torch
import numpy as np
import os
from torch_geometric.data import Dataset
from .dataset import MetabolicNetworkDataset
from .film_dataset import FiLMDataset
from .metabolite_mapping import dense_row
from ..utils.gene_parsing import parse_gene_rule_simple as parse_gene_rule
from .gpu_cached_graph import get_or_create_gpu_cache
from ..utils.graph_randomization import randomize_graph_edges
from ..utils.masking import create_pathway_edge_mask


class MetabolicNetworkDatasetWithSMILES(MetabolicNetworkDataset):
    """Extends the standard dataset to use SMILES embeddings for metabolites"""
    
    def __init__(self, mudata_path, graph_path, gene_mapping_path=None, use_embeddings=True):
        """
        Args:
            mudata_path: Path to the mudata file
            graph_path: Path to the graph with SMILES embeddings
            gene_mapping_path: Path to gene mapping file
            use_embeddings: Whether to use embeddings (vs. gene features)
        """
        super().__init__(mudata_path, graph_path, gene_mapping_path, use_embeddings)
        
        # Check if node_embeddings from SMILES are available
        if hasattr(self.graph, 'x_smiles'):
            self.has_smiles_embeddings = True
            self.smiles_dim = self.graph.x_smiles.size(1)
        else:
            self.has_smiles_embeddings = False
    
    def get(self, idx):
        """Override to include SMILES embeddings"""
        # Get the base data object from parent class
        data = super().get(idx)
        
        # Add SMILES embeddings if available
        if self.has_smiles_embeddings:
            data.x_smiles = self.graph.x_smiles.clone()
            
            # Verify for NaN values and replace them
            data.x_smiles = torch.nan_to_num(data.x_smiles, nan=0.0)
        
        return data


class FiLMDatasetWithSMILES(FiLMDataset):
    """Extends the FiLM dataset to use SMILES embeddings for metabolites"""

    def __init__(self, mudata_path, embeddings_graph_path, gene_mapping_path, device=None,
                 randomize_smiles=False, randomize_genes=False, randomize_embeddings=False,
                 random_graph=False, randomize_gpe=False):
        """
        Args:
            mudata_path: Path to the mudata file
            embeddings_graph_path: Path to the graph with SMILES embeddings
            gene_mapping_path: Path to gene mapping file
            device: Target device for GPU caching (optional)
            randomize_smiles: If True, replace SMILES embeddings with fixed random noise
            randomize_genes: If True, replace gene expression with fixed random noise
            randomize_embeddings: If True, replace enzyme embeddings with fixed random noise
            random_graph: If True, randomize graph topology (edges)
            randomize_gpe: If True, replace GPE features with fixed random noise
        """
        super().__init__(mudata_path, embeddings_graph_path, gene_mapping_path)
        
        self.randomize_smiles = randomize_smiles
        self.randomize_genes = randomize_genes
        self.randomize_embeddings = randomize_embeddings
        self.random_graph = random_graph
        
        # Check if node_embeddings from SMILES are available
        if hasattr(self.graph, 'x_smiles'):
            self.has_smiles_embeddings = True
            self.smiles_dim = self.graph.x_smiles.size(1)
        else:
            self.has_smiles_embeddings = False
            self.smiles_dim = 0
            
        # === FIXED RANDOM FEATURES INITIALIZATION ===
        # If randomization is requested, we generate the noise tensors ONCE during initialization.
        # This ensures the "random features" are consistent across epochs (static noise baseline).
        
        if self.randomize_smiles and self.has_smiles_embeddings:
            print(f"⚠️ RANDOMIZATION: Replacing SMILES embeddings with fixed Gaussian noise (dim={self.smiles_dim})")
            self.random_smiles_tensor = torch.randn_like(self.graph.x_smiles)
            
        if self.randomize_embeddings:
            print(f"⚠️ RANDOMIZATION: Replacing Enzyme embeddings with fixed Gaussian noise (dim={self.graph.x.size(1)})")
            self.random_embedding_tensor = torch.randn_like(self.graph.x)
            
        # === TOPOLOGICAL MASKING (PRUNING) ===
        # NOTE: Topological masking is now handled by loading pre-pruned graph files.
        # See metabolicatlas_api.py and data_paths.py
        self.pathway_edge_mask = None
            
        if self.random_graph:
            print(f"⚠️ RANDOMIZATION: Randomizing graph topology (edges)")
            # Clear any existing GPU cache to ensure random edges are moved to GPU
            from .gpu_cached_graph import GPUCachedGraph
            GPUCachedGraph.clear_cache()
            self.graph = randomize_graph_edges(self.graph)

        # === GRAPH POSITIONAL ENCODINGS ===
        # Compute GPE features from (possibly randomized) graph topology.
        # These are static per-node features computed once at init.
        self.randomize_gpe = randomize_gpe
        self.x_gpe = None
        self._gpe_dim = 0
        try:
            from ..models.modules.graph_positional_encodings import GraphPositionalEncoder
            gpe_encoder = GraphPositionalEncoder(rwse_steps=16)
            num_nodes = len(self.node_types) if hasattr(self, 'node_types') else self.graph.x.size(0)
            self.x_gpe = gpe_encoder(self.graph.edge_index, num_nodes)
            self._gpe_dim = gpe_encoder.output_dim
            if self.randomize_gpe:
                print(f"⚠️ RANDOMIZATION: Replacing GPE features with fixed random noise (dim={self._gpe_dim})")
                torch.manual_seed(123)
                self.x_gpe = torch.randn_like(self.x_gpe)
            else:
                print(f"Computed GPE features: {self.x_gpe.shape}")
        except Exception as e:
            print(f"Warning: Could not compute GPE features: {e}")

        # For genes, we can't easily pre-allocate [num_samples, gene_dim] due to potential memory issues
        # if the dataset is huge, but for this dataset (~900 samples), it's trivial.
        # However, parent class loads data into dense lists.
        # Let's handle gene randomization by overriding the dense arrays or generating on fly.
        # Given we want *fixed* random features per sample, we should pre-generate a random seed per sample
        # or a random tensor per sample.
        if self.randomize_genes:
            print(f"⚠️ RANDOMIZATION: Gene expression will be replaced with fixed Gaussian noise")
            # We will handle this in the dense matrix creation step below
        
        # OPTIMIZATION: Create GPU cache for static graph components
        self.gpu_cache = None
        if device is not None:
            # Temporarily swap real data with random data for cache creation if needed
            original_x = self.graph.x
            original_smiles = self.graph.x_smiles if hasattr(self.graph, 'x_smiles') else None
            
            if self.randomize_embeddings:
                self.graph.x = self.random_embedding_tensor
            
            if self.randomize_smiles and self.has_smiles_embeddings:
                self.graph.x_smiles = self.random_smiles_tensor
                
            self.gpu_cache = get_or_create_gpu_cache(self.graph, device)
            
            # Restore original graph data
            self.graph.x = original_x
            if original_smiles is not None:
                self.graph.x_smiles = original_smiles
        
        # OPTIMIZATION: Pre-convert sparse matrices to dense for faster access
        print("Pre-converting sparse metabolomics matrices to dense format...")
        self.metabolomics_dense = []
        for i in range(self.num_samples):
            self.metabolomics_dense.append(dense_row(self.mudata['metabolomics'][i].X))
        print(f"Pre-converted {self.num_samples} metabolomics samples to dense format")
        
        # OPTIMIZATION: Pre-convert RNA matrices to dense
        print("Processing RNA matrices...")
        self.rna_dense = []
        if self.randomize_genes:
            # Generate fixed random noise for each sample's RNA profile
            # We need to know the RNA dimension.
            rna_dim = self.mudata['rna'].shape[1]
            torch.manual_seed(42) # Ensure reproducibility of the random features
            for i in range(self.num_samples):
                # Create random noise instead of real expression
                # Using numpy for compatibility with downstream code that expects arrays
                # Note: No explicit astype(np.float32) to avoid PyTorch assignment issues
                self.rna_dense.append(np.random.randn(rna_dim))
            print(f"Generated fixed random noise for {self.num_samples} RNA samples")
        else:
            for i in range(self.num_samples):
                self.rna_dense.append(dense_row(self.mudata['rna'][i].X))
            print(f"Pre-converted {self.num_samples} RNA samples to dense format")
        
        # Pre-compute incidence matrix (same for all samples)
        print("Pre-computing incidence matrix...")
        self._precompute_incidence_matrix()
        
        # NOTE: Topological masking is now applied BEFORE randomization (see above)
        # self.pathway_edge_mask is effectively None because we prune edges directly.
        self.pathway_edge_mask = None

    def _precompute_incidence_matrix(self):
        """Pre-compute the incidence matrix that's shared across all samples.

        Key fixes:
        1. Filter out reverse edges (is_reverse_edge == True) so that forward +s
           and reverse -s don't cancel to 0 after coalesce().
        2. Use raw stoichiometry (edge_attr_raw) instead of tanh-squashed weights
           so that the mass-balance loss Sv = 0 operates on true ratios.
        """
        met_idx = {node_idx: i for i, node_idx in enumerate(self.metabolite_indices)}
        rxn_idx = {node_idx: i for i, node_idx in enumerate(self.reaction_indices)}

        M = len(self.metabolite_indices)
        R = len(self.reaction_indices)

        rows, cols, vals = [], [], []

        edge_index = self.graph.edge_index
        edge_attr_raw = getattr(self.graph, 'edge_attr_raw', None)
        edge_attr = self.graph.edge_attr if hasattr(self.graph, 'edge_attr') else None
        is_reverse_edge = getattr(self.graph, 'is_reverse_edge', None)

        for e in range(edge_index.size(1)):
            # Skip reverse edges — only forward edges contribute to the incidence matrix
            if is_reverse_edge is not None and is_reverse_edge[e].item():
                continue

            u, v = edge_index[:, e]
            u_idx, v_idx = u.item(), v.item()

            if (u_idx in met_idx and v_idx in rxn_idx):
                rows.append(met_idx[u_idx])
                cols.append(rxn_idx[v_idx])
                if edge_attr_raw is not None:
                    vals.append(edge_attr_raw[e].item())
                elif edge_attr is not None:
                    vals.append(edge_attr[e].item())
                else:
                    vals.append(-1.0)
            elif (u_idx in rxn_idx and v_idx in met_idx):
                rows.append(met_idx[v_idx])
                cols.append(rxn_idx[u_idx])
                if edge_attr_raw is not None:
                    vals.append(edge_attr_raw[e].item())
                elif edge_attr is not None:
                    vals.append(edge_attr[e].item())
                else:
                    vals.append(1.0)

        if len(rows) > 0:
            indices = torch.tensor([rows, cols], dtype=torch.long)
            values = torch.tensor(vals, dtype=torch.float32)
            self.incidence_matrix = torch.sparse_coo_tensor(indices, values, (M, R)).coalesce()
        else:
            self.incidence_matrix = torch.sparse_coo_tensor(
                torch.zeros(2, 0, dtype=torch.long),
                torch.zeros(0, dtype=torch.float32),
                (M, R)
            )

        print(f"  Pre-computed incidence matrix: {self.incidence_matrix.shape}")
        print(f"  Non-zero entries: {self.incidence_matrix._nnz()}")
        if is_reverse_edge is not None:
            n_reverse = is_reverse_edge.sum().item()
            print(f"  Reverse edges filtered out: {n_reverse}")

    def get(self, idx):
        """Override to include SMILES embeddings and incidence matrix"""
        
        # OPTIMIZATION: Bypass parent class when GPU cache available
        if self.gpu_cache is not None:
            # Create data object from scratch with GPU-cached components
            from torch_geometric.data import Data
            
            # Get GPU tensors directly (avoid DataLoader transfer)
            static_data = self.gpu_cache.get_static_data(force_cpu=False)
            
            # Create data object with pre-cached GPU tensors
            data = Data(
                edge_index=static_data['edge_index'],
                num_nodes=len(self.node_types),
            )
            
            # Add all static GPU components
            if 'x_embedding' in static_data:
                data.x_embedding = static_data['x_embedding']
            if 'x_smiles' in static_data:
                data.x_smiles = static_data['x_smiles']
            if 'edge_attr' in static_data:
                data.edge_attr = static_data['edge_attr']
            
            # Add gene operation tensors for fast correlation
            if 'reaction_gene_indices' in static_data:
                data.reaction_gene_indices = static_data['reaction_gene_indices']
                data.reaction_gene_operations = static_data['reaction_gene_operations']
                data.reaction_gene_masks = static_data['reaction_gene_masks']
                data.reaction_node_indices = static_data['reaction_node_indices']
                data.max_genes_per_reaction = static_data['max_genes_per_reaction']
                data.num_valid_reactions = static_data['num_valid_reactions']
            
            # Create gene expression features (CPU, will be transferred)
            data.x_gene = torch.zeros(len(self.node_types), self.gene_dim)
            
            # Fill gene expression features for reaction nodes (same as parent class)
            # Use pre-converted dense RNA data (which might be random noise now)
            rna_data = self.rna_dense[idx]
            
            for r_idx in self.reaction_indices:
                gene_rule = self.graph.gene_rules[r_idx]
                genes = parse_gene_rule(gene_rule)
                
                # Fill in expression values for measured genes
                for gene_id in genes:
                    if gene_id in self.gene_to_index and gene_id in self.gene_id_to_expression_idx:
                        gene_feature_idx = self.gene_to_index[gene_id]
                        expression_idx = self.gene_id_to_expression_idx[gene_id]
                        
                        # Only assign non-NaN values
                        value = rna_data[expression_idx]
                        if not np.isnan(value):
                            data.x_gene[r_idx, gene_feature_idx] = float(value)
            
            # Add per-sample data (y, mask, indices) from parent class logic
            # Use pre-converted dense metabolomics data
            metabolomics_data = self.metabolomics_dense[idx]
            data.y, data.mask = self._targets_for_values(metabolomics_data)
            data.measurable_mask = self.measurable_mask
            data.metabolite_indices = torch.tensor(self.metabolite_indices)
            data.reaction_indices = torch.tensor(self.reaction_indices)
            data.node_types = self.node_types
            
            # Mark that static components are already on GPU
            data._static_on_gpu = True
            
        else:
            # Fallback: Use parent class (creates CPU tensors)
            data = super().get(idx)

            # If randomizing embeddings, override x from parent
            if self.randomize_embeddings:
                data.x = self.random_embedding_tensor.clone()
            
            # Add SMILES embeddings (random or real)
            if self.has_smiles_embeddings:
                if self.randomize_smiles:
                    data.x_smiles = self.random_smiles_tensor.clone()
                else:
                    data.x_smiles = self.graph.x_smiles.clone()
                data.x_smiles = torch.nan_to_num(data.x_smiles, nan=0.0)
            
            # If randomizing genes, we need to override x_gene if it exists?
            # Wait, parent get() logic populates x based on rna_data.
            # But parent get() reads from self.mudata['rna'] directly!
            # We need to ensure we use our randomized self.rna_dense if set.
            # Actually, `super().get(idx)` reads from `self.mudata`.
            # To properly support gene randomization in fallback mode, we must override the x/x_gene logic.
            # But `super().get` is `MetabolicNetworkDataset.get` which does the heavy lifting.
            # It's better to duplicate the logic here to ensure control, OR accept that fallback mode 
            # might not support gene randomization perfectly if we don't patch parent.
            # However, we DID set self.rna_dense with random values. 
            # The parent `get` reads from `self.mudata`.
            # Let's override `x` if `randomize_genes` is True by re-running the gene filling logic
            # using `self.rna_dense`.
            if self.randomize_genes:
                # Reset gene features
                data.x = torch.zeros_like(self.graph.x)
                if self.randomize_embeddings:
                     data.x = self.random_embedding_tensor.clone()
                else:
                     # For non-reaction nodes, copy original features
                     non_reaction_mask = torch.tensor([t != 'reaction' for t in self.node_types])
                     data.x[non_reaction_mask] = self.graph.x[non_reaction_mask]
                
                # Update reaction nodes with randomized RNA data
                rna_data = self.rna_dense[idx]
                for r_idx in self.reaction_indices:
                    gene_rule = self.graph.gene_rules[r_idx]
                    genes = parse_gene_rule(gene_rule)
                    for gene_id in genes:
                        if gene_id in self.gene_to_index and gene_id in self.gene_id_to_expression_idx:
                            gene_feature_idx = self.gene_to_index[gene_id]
                            expression_idx = self.gene_id_to_expression_idx[gene_id]
                            value = rna_data[expression_idx]
                            if not np.isnan(value):
                                data.x[r_idx, gene_feature_idx] = float(value)

            data._static_on_gpu = False
        
        # Add SMILES embeddings to GPU cache path if not already added
        if self.gpu_cache is not None and self.has_smiles_embeddings and not hasattr(data, 'x_smiles'):
            if self.randomize_smiles:
                data.x_smiles = self.random_smiles_tensor.clone()
            else:
                data.x_smiles = self.graph.x_smiles.clone()
            data.x_smiles = torch.nan_to_num(data.x_smiles, nan=0.0)
        
        # Attach GPE features if available
        if self.x_gpe is not None:
            data.x_gpe = self.x_gpe.clone()

        # OPTIMIZATION: Use pre-computed incidence matrix
        data.incidence_matrix = self.incidence_matrix
        data.pathway_edge_mask = self.pathway_edge_mask
        
        # OPTIMIZATION: Only copy tensor operations if not using GPU cache
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
    
