import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules.directional_conv import DirectionalConv
from .components.met_reader import (
    extract_reaction_neighbors,
    extract_reaction_neighbors_cached,
    precompute_neighbor_indices,
    verify_no_metabolite_specific_parameters,
    shuffle_metabolite_descriptors,
)
from ..utils.tensor_utils import check_for_nans
from ..utils.batch_utils import standardize_batch_data, validate_batch_shapes, get_batch_size, expand_to_batch, create_default_features
from .components import BaseConcatModel, build_gnn_layers, MCRReaderConfig

class ConcatGNNWithSMILES(BaseConcatModel):
    """
    Graph Neural Network that concatenates enzyme embeddings with gene expression
    and incorporates SMILES embeddings for metabolites.

    Extended with physics-informed flux prediction for reaction nodes.

    NOTE: The flux head and physics-informed forward pass can be reused in other models:
    - models/film_gnn_smiles.py: Add reaction_flux_head and modify forward()
    - models/gnn_smiles.py: Add reaction_flux_head and modify forward()
    - models/concat_gnn.py: Add reaction_flux_head and modify forward()

    The same pattern applies to all model variants.
    """
    def __init__(self, embedding_dim, gene_dim, smiles_dim, hidden_dim=128, num_layers=3, dropout=0.2,
                 use_directional=False, directional_concat=False,
                 use_flux_messages=False, flux_message_depth=1, use_mass_balance_projection=False,
                 # MCR parameters
                 use_mcr=False, reader_hidden_dim=128, reader_mlp_hidden=128, 
                 reader_two_hop=False, reader_normalize_descriptors=True, reader_max_neighbors=50,
                 reader_shuffle_descriptors=False, reader_neighbor_dropout=0.0,
                 reader_neighbor_min_keep: int = 1,
                 # Layer configuration
                 layer_type: str = "GAT", deg=None):
        reader_cfg = None
        if use_mcr:
            reader_cfg = MCRReaderConfig(
                smiles_dim=smiles_dim,
                hidden_dim=reader_hidden_dim,
                mlp_hidden=reader_mlp_hidden,
                normalize_descriptors=reader_normalize_descriptors,
                dropout=dropout,
                neighbor_dropout_rate=reader_neighbor_dropout,
                neighbor_min_keep=reader_neighbor_min_keep,
            )

        super().__init__(
            embedding_dim=embedding_dim,
            gene_dim=gene_dim,
            smiles_dim=smiles_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_flux_messages=use_flux_messages,
            flux_message_depth=flux_message_depth,
            use_mass_balance_projection=use_mass_balance_projection,
            use_mcr=use_mcr,
            reader_config=reader_cfg,
        )

        # Track if we're using directionality
        self.use_directional = use_directional

        # Store MCR-specific configuration for downstream utilities
        self.reader_hidden_dim = reader_hidden_dim
        self.reader_mlp_hidden = reader_mlp_hidden
        self.reader_two_hop = reader_two_hop
        self.reader_normalize_descriptors = reader_normalize_descriptors
        self.reader_max_neighbors = reader_max_neighbors
        self.reader_shuffle_descriptors = reader_shuffle_descriptors
        self.reader_neighbor_dropout = reader_neighbor_dropout
        self.reader_neighbor_min_keep = reader_neighbor_min_keep

        # Graph convolution layers for metabolic network
        self.convs = build_gnn_layers(
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_directional=use_directional,
            directional_concat=directional_concat,
            layer_type=layer_type,
            deg=deg,
        )

        # === CONTRASTIVE LEARNING COMPONENTS (OPTIONAL) ===
        self.contrastive_module = None  # Will be initialized if contrastive learning is enabled
        
        # Contrastive learning configuration (default values, can be overridden)
        self.contrastive_config = {
            'gene_dropout_rate': 0.1,
            'embedding_dropout_rate': 0.1,
            'smiles_dropout_rate': 0.05,
            'temperature': 0.07
        }
    
    def forward(self, data, return_embeddings=False, compute_physics=True):
        """
        Unified forward pass that always handles batched data.

        Single samples are automatically converted to batch_size=1.
        All tensors maintain batch dimensions throughout processing.

        Args:
            data: PyG Data object (automatically batched if needed)
            return_embeddings: whether to return final node embeddings
            compute_physics: whether to compute physics-informed outputs

        Returns:
            pred: [batch_size, num_nodes] metabolite predictions
            delta: [batch_size, num_metabolites] mass balance residuals
            flux_vec: [batch_size, num_reactions] reaction fluxes
            embeddings: [batch_size, num_nodes, hidden_dim] (if requested)
        """
        # Standardize data to always have batch dimensions
        data = standardize_batch_data(data)

        # Standard batched forward pass
        return self._forward_batched_unified(data, return_embeddings, compute_physics)
    
    def _forward_batched_unified(self, data, return_embeddings=False, compute_physics=True):
        """
        Unified batched forward pass for all samples.
        
        Expects data to already be standardized with batch dimensions.
        
        Args:
            data: Batched Data object with:
                - x_embedding: [batch_size, num_nodes, embedding_dim] 
                - x_gene: [batch_size, num_nodes, gene_dim]
                - x_smiles: [batch_size, num_nodes, smiles_dim]
                - edge_index: [2, num_edges] (shared across samples)
                - edge_attr: [num_edges, edge_features] (shared across samples)
                
        Returns:
            pred: [batch_size, num_nodes] - metabolite predictions
            delta: [batch_size, num_metabolites] - mass balance residuals  
            flux_vec: [batch_size, num_reactions] - reaction fluxes
            embeddings: [batch_size, num_nodes, hidden_dim] (if requested)
        """
        batch_size = get_batch_size(data)
        
        # Extract batched features (already standardized with batch dims)
        x_embedding = getattr(data, 'x_embedding', getattr(data, 'x', None))  # [batch_size, num_nodes, embedding_dim]
        if x_embedding is None:
            raise ValueError("Data object must have either 'x_embedding' or 'x' attribute")
        num_nodes = x_embedding.size(1)
        
        # Get or create default features if missing
        if hasattr(data, 'x_gene'):
            x_gene = data.x_gene  # [batch_size, num_nodes, gene_dim]
        else:
            x_gene = create_default_features(num_nodes, self.gene_dim, batch_size, x_embedding.device)
        
        if hasattr(data, 'x_smiles'):
            x_smiles = data.x_smiles  # [batch_size, num_nodes, smiles_dim]
        else:
            x_smiles = create_default_features(num_nodes, self.smiles_dim, batch_size, x_embedding.device)
        
        edge_index = data.edge_index    # [2, num_edges] - shared
        edge_attr = data.edge_attr      # [num_edges, edge_features] - shared
        reaction_indices = data.reaction_indices
        metabolite_indices = data.metabolite_indices
        
        # Validate input features
        x_embedding = check_for_nans(x_embedding, "input x_embedding")
        x_gene = check_for_nans(x_gene, "input x_gene")
        
        # Features should already be 3D from standardize_batch_data
        assert x_embedding.dim() == 3, f"x_embedding should be 3D, got {x_embedding.dim()}D"
        assert x_gene.dim() == 3, f"x_gene should be 3D, got {x_gene.dim()}D"
        assert x_smiles.dim() == 3, f"x_smiles should be 3D, got {x_smiles.dim()}D"
        
        # Rest of the method continues below...
        
        # === EFFICIENT BULK TRANSCRIPTOMICS BATCHING ===
        
        # Convert indices to proper 1D tensors (handles scalars, lists, and tensors)
        if isinstance(reaction_indices, (int, float)):
            reaction_idx_tensor = torch.tensor([reaction_indices], device=x_embedding.device)
        elif isinstance(reaction_indices, list):
            reaction_idx_tensor = torch.tensor(reaction_indices, device=x_embedding.device)
        elif isinstance(reaction_indices, torch.Tensor):
            if reaction_indices.dim() == 0:  # Scalar tensor
                reaction_idx_tensor = reaction_indices.unsqueeze(0)
            else:
                reaction_idx_tensor = reaction_indices
        else:
            reaction_idx_tensor = reaction_indices
            
        if isinstance(metabolite_indices, (int, float)):
            metabolite_idx_tensor = torch.tensor([metabolite_indices], device=x_embedding.device)
        elif isinstance(metabolite_indices, list):
            metabolite_idx_tensor = torch.tensor(metabolite_indices, device=x_embedding.device) 
        elif isinstance(metabolite_indices, torch.Tensor):
            if metabolite_indices.dim() == 0:  # Scalar tensor
                metabolite_idx_tensor = metabolite_indices.unsqueeze(0)
            else:
                metabolite_idx_tensor = metabolite_indices
        else:
            metabolite_idx_tensor = metabolite_indices
        
        num_nodes = x_embedding.size(1)
        
        # Initialize batched transformed features: [batch_size, num_nodes, hidden_dim]
        x_transformed = torch.zeros(
            batch_size, num_nodes, self.hidden_dim,
            device=x_embedding.device
        )
        
        # === ENCODE REACTION NODES ===
        # Extract reaction features: [batch_size, num_reactions, feature_dim]
        reaction_embeddings = x_embedding[:, reaction_idx_tensor]  # [batch_size, num_reactions, embedding_dim]
        reaction_gene_expr = x_gene[:, reaction_idx_tensor]        # [batch_size, num_reactions, gene_dim]
        
        # Concatenate features: [batch_size, num_reactions, embedding_dim + gene_dim]
        combined_features = torch.cat([reaction_embeddings, reaction_gene_expr], dim=2)
        
        # Apply reaction encoder: [batch_size, num_reactions, hidden_dim]
        encoded_reactions = self.reaction_encoder(combined_features)
        x_transformed[:, reaction_idx_tensor] = encoded_reactions
        
        # === ENCODE METABOLITE NODES ===
        # Extract metabolite SMILES: [batch_size, num_metabolites, smiles_dim]
        metabolite_smiles = x_smiles[:, metabolite_idx_tensor]
        
        # Apply metabolite encoder: [batch_size, num_metabolites, hidden_dim] 
        encoded_metabolites = self.metabolite_encoder(metabolite_smiles)
        x_transformed[:, metabolite_idx_tensor] = encoded_metabolites
        
        # === EFFICIENT GRAPH CONVOLUTIONS ===
        # Key insight: Since all samples share the same graph structure,
        # we can process all samples simultaneously by reshaping tensors
        
        h = x_transformed  # [batch_size, num_nodes, hidden_dim]
        
        # Initialize mass balance projector if needed
        if self.use_mass_balance_projection and self.mass_balance_projector is None:
            self.initialize_mass_balance_projector(data)
            
        # Helper to get/create batched edge index
        if not hasattr(self, 'batched_edge_index_cache'):
            self.batched_edge_index_cache = {}
            
        cache_key = (batch_size, num_nodes, x_embedding.device)
        
        if cache_key in self.batched_edge_index_cache:
            batched_edge_index, batched_edge_attr = self.batched_edge_index_cache[cache_key]
        else:
            # Create batched edge indices by adding node offsets
            # edge_index: [2, num_edges] -> batched_edge_index: [2, batch_size * num_edges]
            batched_edge_index_list = []
            batched_edge_attr_list = []
            
            for b in range(batch_size):
                # Add offset for this batch
                offset_edge_index = edge_index + b * num_nodes
                batched_edge_index_list.append(offset_edge_index)
                
                # Edge attributes are the same for all samples
                if edge_attr is not None:
                    batched_edge_attr_list.append(edge_attr)
            
            # Concatenate all batch edge indices: [2, batch_size * num_edges]
            batched_edge_index = torch.cat(batched_edge_index_list, dim=1)
            
            if edge_attr is not None:
                batched_edge_attr = torch.cat(batched_edge_attr_list, dim=0)
            else:
                batched_edge_attr = None
                
            # Cache the result
            self.batched_edge_index_cache[cache_key] = (batched_edge_index, batched_edge_attr)

        for layer_idx, conv in enumerate(self.convs):
            # Reshape to [batch_size * num_nodes, hidden_dim] for PyG convolution
            h_flat = h.view(batch_size * num_nodes, self.hidden_dim)
            
            # Apply convolution to all samples simultaneously
            h_flat_new = conv(h_flat, batched_edge_index, batched_edge_attr)
            h_flat_new = check_for_nans(h_flat_new, "batched GNN convolution output")
            
            # Reshape back to [batch_size, num_nodes, hidden_dim]
            h = h_flat_new.view(batch_size, num_nodes, self.hidden_dim)
            h = F.elu(h)
            h = self.dropout(h)
            
            # === ADD FLUX MESSAGES AT SPECIFIED DEPTH ===
            if (self.use_flux_messages and 
                layer_idx + 1 == self.flux_message_depth and  # 1-indexed depth
                hasattr(data, 'incidence_matrix')):

                # Predict fluxes from current reaction embeddings
                reaction_emb_current = h[:, reaction_idx_tensor]  # [batch_size, num_reactions, hidden_dim]
                flux_predictions = self.reaction_flux_head(reaction_emb_current).squeeze(-1)  # [batch_size, num_reactions]

                # Project fluxes to mass-balanced subspace if enabled
                if self.use_mass_balance_projection and self.mass_balance_projector is not None:
                    flux_balanced = self.mass_balance_projector.project_to_mass_balance(
                        flux_predictions, data.incidence_matrix
                    )
                else:
                    flux_balanced = flux_predictions

                # Extract metabolite embeddings
                metabolite_emb_current = h[:, metabolite_idx_tensor]  # [batch_size, num_metabolites, hidden_dim]

                # Add flux messages to metabolite embeddings  
                metabolite_emb_updated = self.flux_message_layer(
                    metabolite_embeddings=metabolite_emb_current,
                    reaction_embeddings=reaction_emb_current,
                    fluxes=flux_balanced,
                    stoichiometric_matrix=data.incidence_matrix,
                    metabolite_indices=metabolite_indices,
                    reaction_indices=reaction_indices
                )

                # Update metabolite embeddings in h
                h[:, metabolite_idx_tensor] = metabolite_emb_updated
        
        # === PREDICT METABOLITE CONCENTRATIONS ===
        pred = torch.zeros(batch_size, num_nodes, device=x_embedding.device)
        
        # Extract metabolite embeddings: [batch_size, num_metabolites, hidden_dim]
        metabolite_emb = h[:, metabolite_idx_tensor]
        
        # Apply predictor: [batch_size, num_metabolites]
        if self.use_mcr:
            # Use MCR for batched prediction
            metabolite_pred = self._predict_with_mcr_batched_base(
                h, metabolite_idx_tensor, reaction_idx_tensor, edge_index, x_smiles
            )
        else:
            # Use traditional MLP head
            metabolite_pred = self.metabolite_predictor(metabolite_emb).squeeze(-1)
        pred[:, metabolite_idx_tensor] = metabolite_pred
        
        engages_flux_head = (
            compute_physics
            or (self.use_flux_messages and hasattr(data, 'incidence_matrix'))
            or (self.use_mass_balance_projection and hasattr(data, 'incidence_matrix'))
        )

        if engages_flux_head:
            reaction_emb = h[:, reaction_idx_tensor]
            flux_vec_raw = self.reaction_flux_head(reaction_emb).squeeze(-1)

            if (self.use_mass_balance_projection and
                self.mass_balance_projector is not None and
                hasattr(data, 'incidence_matrix')):
                flux_vec_projected = self.mass_balance_projector.project_to_mass_balance(
                    flux_vec_raw, data.incidence_matrix
                )
                if False:  # Optional debug verification
                    verification = self.mass_balance_projector.verify_mass_balance(
                        flux_vec_projected, data.incidence_matrix, tolerance=1e-6
                    )
                    if not verification['passes_tolerance']:
                        print(
                            f"Warning: Mass balance tolerance not met. Max residual: {verification['max_residual']:.2e}"
                        )
            else:
                flux_vec_projected = flux_vec_raw
        else:
            flux_vec_raw = torch.zeros(batch_size, len(reaction_indices), device=x_embedding.device)
            flux_vec_projected = flux_vec_raw

        # === COMPUTE MASS-BALANCE RESIDUALS ===
        if compute_physics and hasattr(data, 'incidence_matrix'):
            incidence_matrix = data.incidence_matrix.to(flux_vec_raw.device)

            if incidence_matrix.is_sparse:
                delta = torch.sparse.mm(incidence_matrix, flux_vec_raw.T).T
            else:
                delta = flux_vec_raw @ incidence_matrix.T
        else:
            delta = torch.zeros(batch_size, len(metabolite_indices), device=flux_vec_raw.device)

        # Return final embeddings if requested
        embeddings = h if return_embeddings else None

        return pred, delta, flux_vec_raw, embeddings
    
    def extract_embeddings(self, data, normalize=True):
        """
        Extract L2-normalized embeddings after GNN processing for contrastive pretraining.

        This method processes data through the GNN layers and returns the embeddings
        at the optimal stage for contrastive learning (after DirectionalConv layers
        but before prediction heads).

        Args:
            data: PyG Data object (automatically batched if needed)
            normalize: Whether to apply L2 normalization to embeddings (default: True)

        Returns:
            dict: Dictionary containing:
                - 'reaction_embeddings': [batch_size, num_reactions, hidden_dim] L2-normalized reaction embeddings
                - 'metabolite_embeddings': [batch_size, num_metabolites, hidden_dim] L2-normalized metabolite embeddings
                - 'all_embeddings': [batch_size, num_nodes, hidden_dim] All node embeddings
        """
        # Ensure data has batch dimensions and disable physics computation for efficiency
        data = standardize_batch_data(data)
        batch_size = get_batch_size(data)

        # Process through the model but only extract embeddings (no predictions)
        # Note: We need gradients enabled for training, so no torch.no_grad() here
        _, _, _, embeddings = self._forward_batched_unified(
            data, return_embeddings=True, compute_physics=False
        )
        
        # Convert indices to tensors for indexing
        reaction_indices = data.reaction_indices
        metabolite_indices = data.metabolite_indices
        
        if isinstance(reaction_indices, (int, float)):
            reaction_idx_tensor = torch.tensor([reaction_indices], device=embeddings.device)
        elif isinstance(reaction_indices, list):
            reaction_idx_tensor = torch.tensor(reaction_indices, device=embeddings.device)
        else:
            reaction_idx_tensor = reaction_indices
            
        if isinstance(metabolite_indices, (int, float)):
            metabolite_idx_tensor = torch.tensor([metabolite_indices], device=embeddings.device)
        elif isinstance(metabolite_indices, list):
            metabolite_idx_tensor = torch.tensor(metabolite_indices, device=embeddings.device)
        else:
            metabolite_idx_tensor = metabolite_indices
        
        # Extract embeddings by node type
        reaction_embeddings = embeddings[:, reaction_idx_tensor]  # [batch_size, num_reactions, hidden_dim]
        metabolite_embeddings = embeddings[:, metabolite_idx_tensor]  # [batch_size, num_metabolites, hidden_dim]

        # Retrieve SMILES embeddings for contrastive alignment
        if hasattr(data, 'x_smiles') and data.x_smiles is not None:
            smiles_embeddings = data.x_smiles
        else:
            raise ValueError(
                "SMILES embeddings (`x_smiles`) are required for contrastive pretraining but were not found in the data object. "
                "Ensure the dataset includes SMILES features or disable contrastive objectives."
            )

        # Apply L2 normalization if requested
        if normalize:
            reaction_embeddings = F.normalize(reaction_embeddings, p=2, dim=-1)
            metabolite_embeddings = F.normalize(metabolite_embeddings, p=2, dim=-1)
            smiles_embeddings = F.normalize(smiles_embeddings, p=2, dim=-1)
            all_embeddings = F.normalize(embeddings, p=2, dim=-1)
        else:
            all_embeddings = embeddings

        return {
            'reaction_embeddings': reaction_embeddings,
            'metabolite_embeddings': metabolite_embeddings,
            'smiles_embeddings': smiles_embeddings,
            'all_embeddings': all_embeddings,
            'batch_size': batch_size
        }
    
    def extract_augmented_embeddings(self, data, augmentation_fn=None, normalize=True):
        """
        Extract embeddings with augmentation applied for contrastive pretraining.
        
        This method applies augmentation to the input features and then processes
        through the GNN to obtain augmented embeddings for within-modal contrastive learning.
        
        Args:
            data: PyG Data object (automatically batched if needed)
            augmentation_fn: Function to augment features. If None, applies default augmentation
                           (dropout-based feature masking)
            normalize: Whether to apply L2 normalization to embeddings (default: True)
            
        Returns:
            dict: Same format as extract_embeddings() but with augmented features
        """
        # Ensure data has batch dimensions  
        data = standardize_batch_data(data)
        batch_size = get_batch_size(data)
        
        # Create augmented version of the data
        augmented_data = self._apply_feature_augmentation(data, augmentation_fn)

        # Process augmented data through the model
        # Note: We need gradients enabled for training, so no torch.no_grad() here
        _, _, _, embeddings = self._forward_batched_unified(
            augmented_data, return_embeddings=True, compute_physics=False
        )
        
        # Convert indices to tensors for indexing
        reaction_indices = data.reaction_indices
        metabolite_indices = data.metabolite_indices
        
        if isinstance(reaction_indices, (int, float)):
            reaction_idx_tensor = torch.tensor([reaction_indices], device=embeddings.device)
        elif isinstance(reaction_indices, list):
            reaction_idx_tensor = torch.tensor(reaction_indices, device=embeddings.device)
        else:
            reaction_idx_tensor = reaction_indices
            
        if isinstance(metabolite_indices, (int, float)):
            metabolite_idx_tensor = torch.tensor([metabolite_indices], device=embeddings.device)
        elif isinstance(metabolite_indices, list):
            metabolite_idx_tensor = torch.tensor(metabolite_indices, device=embeddings.device)
        else:
            metabolite_idx_tensor = metabolite_indices
        
        # Extract embeddings by node type
        reaction_embeddings = embeddings[:, reaction_idx_tensor]  # [batch_size, num_reactions, hidden_dim]
        metabolite_embeddings = embeddings[:, metabolite_idx_tensor]  # [batch_size, num_metabolites, hidden_dim]

        # Retrieve SMILES embeddings for contrastive alignment (with augmentation applied)
        if hasattr(augmented_data, 'x_smiles') and augmented_data.x_smiles is not None:
            smiles_embeddings = augmented_data.x_smiles
        else:
            raise ValueError(
                "Augmented data is missing SMILES embeddings (`x_smiles`), which are required for contrastive pretraining."
            )

        # Apply L2 normalization if requested
        if normalize:
            reaction_embeddings = F.normalize(reaction_embeddings, p=2, dim=-1)
            metabolite_embeddings = F.normalize(metabolite_embeddings, p=2, dim=-1)
            smiles_embeddings = F.normalize(smiles_embeddings, p=2, dim=-1)
            all_embeddings = F.normalize(embeddings, p=2, dim=-1)
        else:
            all_embeddings = embeddings

        return {
            'reaction_embeddings': reaction_embeddings,
            'metabolite_embeddings': metabolite_embeddings,
            'smiles_embeddings': smiles_embeddings,
            'all_embeddings': all_embeddings,
            'batch_size': batch_size
        }
    
    def _apply_feature_augmentation(self, data, augmentation_fn=None):
        """
        Apply augmentation to input features for contrastive learning.
        
        Args:
            data: Original PyG Data object
            augmentation_fn: Custom augmentation function, or None for default
            
        Returns:
            Augmented data object with modified features
        """
        # Create a copy of the data to avoid modifying the original
        from copy import deepcopy
        augmented_data = deepcopy(data)
        
        if augmentation_fn is not None:
            # Use custom augmentation function
            augmented_data = augmentation_fn(augmented_data)
        else:
            # Apply default augmentation: configurable feature dropout
            gene_dropout_rate = self.contrastive_config['gene_dropout_rate']
            embedding_dropout_rate = self.contrastive_config['embedding_dropout_rate']
            smiles_dropout_rate = self.contrastive_config['smiles_dropout_rate']
            
            # Augment gene expression features (if present)
            if hasattr(augmented_data, 'x_gene'):
                x_gene = augmented_data.x_gene
                if x_gene.dim() == 3:  # Batched: [batch_size, num_nodes, gene_dim]
                    mask = torch.rand_like(x_gene) > gene_dropout_rate
                    augmented_data.x_gene = x_gene * mask.float()
                else:  # Single sample: [num_nodes, gene_dim]
                    mask = torch.rand_like(x_gene) > gene_dropout_rate
                    augmented_data.x_gene = x_gene * mask.float()
            
            # Augment reaction embeddings (if present)
            if hasattr(augmented_data, 'x_embedding'):
                x_embedding = augmented_data.x_embedding
                if x_embedding.dim() == 3:  # Batched: [batch_size, num_nodes, embedding_dim]
                    mask = torch.rand_like(x_embedding) > embedding_dropout_rate
                    augmented_data.x_embedding = x_embedding * mask.float()
                else:  # Single sample: [num_nodes, embedding_dim]
                    mask = torch.rand_like(x_embedding) > embedding_dropout_rate
                    augmented_data.x_embedding = x_embedding * mask.float()
            
            # Augment SMILES embeddings (if present) - conservative rate to preserve chemical info
            if hasattr(augmented_data, 'x_smiles'):
                x_smiles = augmented_data.x_smiles
                if x_smiles.dim() == 3:  # Batched: [batch_size, num_nodes, smiles_dim]
                    mask = torch.rand_like(x_smiles) > smiles_dropout_rate
                    augmented_data.x_smiles = x_smiles * mask.float()
                else:  # Single sample: [num_nodes, smiles_dim]
                    mask = torch.rand_like(x_smiles) > smiles_dropout_rate
                    augmented_data.x_smiles = x_smiles * mask.float()
        
        return augmented_data
        
    def enable_contrastive_learning(self, temperature=None, **kwargs):
        """
        Enable contrastive learning for this model.
        
        Args:
            temperature: Temperature for InfoNCE loss (uses config default if None)
            **kwargs: Additional configuration parameters (gene_dropout_rate, embedding_dropout_rate, smiles_dropout_rate)
        """
        # Update configuration with provided parameters
        if temperature is not None:
            self.contrastive_config['temperature'] = temperature
        for key, value in kwargs.items():
            if key in self.contrastive_config:
                self.contrastive_config[key] = value
        
        # Initialize contrastive module if available
        try:
            from gaze._reference.losses.contrastive import create_contrastive_module

            self.contrastive_module = create_contrastive_module(
                hidden_dim=self.hidden_dim,
                temperature=self.contrastive_config['temperature']
            )
        except ImportError:
            print("Warning: Contrastive losses module not available. Contrastive learning disabled.")
            self.contrastive_module = None
        
        
    def compute_contrastive_loss(self, embeddings, metabolite_values, metabolite_mask, reaction_indices, metabolite_indices):
        """
        Compute contrastive loss if contrastive learning is enabled.
        
        Args:
            embeddings: [batch_size, num_nodes, hidden_dim] - node embeddings from forward pass
            metabolite_values: [batch_size, num_nodes] - ground truth metabolite values  
            metabolite_mask: [batch_size, num_nodes] - mask for measured metabolites
            reaction_indices: indices of reaction nodes
            metabolite_indices: indices of metabolite nodes
            
        Returns:
            contrastive_loss: scalar loss value, or 0 if contrastive learning disabled
        """
        if self.contrastive_module is None:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        return self.contrastive_module(
            reaction_embeddings=embeddings,
            metabolite_values=metabolite_values,
            metabolite_mask=metabolite_mask,
            reaction_indices=reaction_indices
        )
    
    def _predict_with_mcr_batched_base(self, h, metabolite_indices, reaction_indices, edge_index, x_smiles):
        """Generate metabolite predictions using MCR for batched samples (base model version).
        
        Args:
            h: [batch_size, num_nodes, hidden_dim] Batched GNN node embeddings
            metabolite_indices: Metabolite node indices
            reaction_indices: Reaction node indices  
            edge_index: [2, num_edges] Graph connectivity (shared across batch)
            x_smiles: [batch_size, num_nodes, smiles_dim] Batched SMILES embeddings
            
        Returns:
            torch.Tensor: [batch_size, num_metabolites] concentration predictions
        """
        # Apply descriptor shuffling for ablation if enabled
        smiles_input = self._apply_descriptor_shuffling_base(x_smiles, metabolite_indices)
        
        # Extract metabolite descriptors for all samples
        metabolite_smiles = smiles_input[:, metabolite_indices]
        
        # Extract reaction neighbor context for attention
        neighbor_embeddings, neighbor_mask = self._extract_neighbor_context_base(
            h, edge_index, metabolite_indices, reaction_indices
        )
        
        # Generate batched predictions via MCR
        return self.metabolite_reader(
            metabolite_smiles=metabolite_smiles,
            reaction_neighbors=neighbor_embeddings,
            neighbor_mask=neighbor_mask
        )
    
    def _apply_descriptor_shuffling_base(self, x_smiles: torch.Tensor, metabolite_indices: torch.Tensor) -> torch.Tensor:
        """Apply SMILES descriptor shuffling for ablation studies if enabled (base model version)."""
        if not self.reader_shuffle_descriptors:
            return x_smiles
        return shuffle_metabolite_descriptors(x_smiles, metabolite_indices, seed=42)
    
    def _extract_neighbor_context_base(self, h: torch.Tensor, edge_index: torch.Tensor,
                                      metabolite_indices: torch.Tensor, reaction_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract reaction neighbor embeddings and mask for MCR attention (base model version)."""
        # Initialize neighbor cache if needed (first forward pass)
        if self.use_mcr and self.neighbor_cache is None:
            self.neighbor_cache = precompute_neighbor_indices(
                edge_index=edge_index,
                metabolite_indices=metabolite_indices,
                reaction_indices=reaction_indices,
                max_neighbors=self.reader_max_neighbors,
                two_hop=self.reader_two_hop,
                exclude_metabolites=None  # No exclusion by default
            )

        # Ensure h has batch dimension for neighbor extraction
        if h.dim() == 2:
            h = h.unsqueeze(0)

        # Use cached neighbor extraction for optimal performance
        if self.neighbor_cache is not None:
            return extract_reaction_neighbors_cached(
                neighbor_cache=self.neighbor_cache,
                metabolite_indices=metabolite_indices,
                reaction_embeddings=h,
                max_neighbors=self.reader_max_neighbors
            )
        else:
            # Fallback to non-cached version
            return extract_reaction_neighbors(
                edge_index=edge_index,
                metabolite_indices=metabolite_indices,
                reaction_indices=reaction_indices,
                reaction_embeddings=h,
                max_neighbors=self.reader_max_neighbors,
                two_hop=self.reader_two_hop
            )
    
    def clear_neighbor_cache(self):
        """Clear the MCR neighbor cache to force recomputation.

        This is needed for LOMO cross-validation where we need to exclude
        held-out metabolites from the neighbor computation.
        """
        if self.use_mcr:
            self.neighbor_cache = None

    def rebuild_neighbor_cache(self, edge_index, metabolite_indices, reaction_indices,
                             exclude_metabolites=None):
        """Rebuild the neighbor cache with optional metabolite exclusion.

        Args:
            edge_index: Graph connectivity
            metabolite_indices: Metabolite node indices
            reaction_indices: Reaction node indices
            exclude_metabolites: Set of metabolite indices to exclude
        """
        if self.use_mcr:
            self.neighbor_cache = precompute_neighbor_indices(
                edge_index=edge_index,
                metabolite_indices=metabolite_indices,
                reaction_indices=reaction_indices,
                max_neighbors=self.reader_max_neighbors,
                two_hop=self.reader_two_hop,
                exclude_metabolites=exclude_metabolites
            )

    def verify_mcr_parameters(self, dataset):
        """Verify MCR has no metabolite-specific parameters to avoid ID leakage."""
        if self.use_mcr and self.metabolite_reader is not None:
            # Get number of metabolites from dataset
            first_data = dataset[0]
            num_metabolites = len(first_data.metabolite_indices)
            
            return verify_no_metabolite_specific_parameters(self.metabolite_reader, num_metabolites)
        return True
