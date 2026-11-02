import torch
import torch.nn as nn
import torch.nn.functional as F

# Import our enhanced components
from .modules.cross_modal_attention import CrossModalAttention
from .modules.residual_layers import ResidualGNNStack, ResidualGNNLayer
from .losses.similarity_regularization import ChemicalSimilarityRegularization
from .losses.structural_consistency import NeighborReconstructionLoss
from .modules.directional_conv import DirectionalConv
from .modules.relational_conv import BipartiteGraphPreparer
# Shared model components
from .components import (
    BaseConcatModel,
    EnhancedModelConfig,
    MCRReaderConfig,
    benchmark_mcr_performance,
    extract_reaction_neighbors,
    extract_reaction_neighbors_cached,
    extract_reaction_neighbors_cached_multihop,
    precompute_neighbor_indices,
    precompute_neighbor_indices_multihop,
    shuffle_metabolite_descriptors,
    verify_no_metabolite_specific_parameters,
)
from ..utils.tensor_utils import check_for_nans
from ..utils.batch_utils import standardize_batch_data, validate_batch_shapes, get_batch_size, create_default_features

class EnhancedConcatGNNWithSMILES(BaseConcatModel):
    """
    Enhanced version of ConcatGNNWithSMILES with improved generalization to unseen metabolites.
    
    Key improvements:
    1. Cross-modal attention between reactions and metabolites
    2. Residual connections and layer normalization to prevent over-smoothing
    3. Unified heterogeneous encoder for better cross-type information flow
    4. Chemical similarity regularization to preserve SMILES relationships
    5. Edge-type-aware attention for different interaction types
    
    The model is fully backward compatible - setting all enhancement flags to False
    makes it equivalent to the original model.
    """
    
    def __init__(self, embedding_dim, gene_dim, smiles_dim, hidden_dim=128, num_layers=3,
                 dropout=0.2, use_directional=False, directional_concat=False,
                 config=None,
                 # Backward compatibility
                 use_flux_messages=False, flux_message_depth=1, use_mass_balance_projection=False,
                 subtract_sample_mean=False):
        # Architectural feature configuration
        self.config = config if config is not None else EnhancedModelConfig()
        self.num_layers = num_layers
        self.cross_modal_residual_weight = getattr(self.config, 'cross_modal_residual_weight', 0.5)
        
        # Store normalization flag
        self.subtract_sample_mean = subtract_sample_mean

        # Determine feature toggles from configuration/backwards-compatible kwargs
        flux_messages_enabled = (
            self.config.use_flux_messages if config is not None else use_flux_messages
        )
        flux_message_depth = (
            self.config.flux_message_depth if config is not None else flux_message_depth
        )
        mass_balance_projection = (
            self.config.use_mass_balance_projection
            if config is not None
            else use_mass_balance_projection
        )
        use_mcr = self.config.use_mcr

        reader_cfg = None
        if use_mcr:
            reader_cfg = MCRReaderConfig(
                smiles_dim=smiles_dim,
                hidden_dim=self.config.reader_hidden_dim,
                mlp_hidden=self.config.reader_mlp_hidden,
                normalize_descriptors=self.config.reader_normalize_descriptors,
                dropout=dropout,
                max_hops=self.config.reader_max_hops,
                hop_decay_init=self.config.reader_hop_decay_init,
                use_gene_skip=self.config.use_gene_skip,
                gene_dim=gene_dim if self.config.use_gene_skip else 0,
            )

        super().__init__(
            embedding_dim=embedding_dim,
            gene_dim=gene_dim,
            smiles_dim=smiles_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_flux_messages=flux_messages_enabled,
            flux_message_depth=flux_message_depth,
            use_mass_balance_projection=mass_balance_projection,
            use_mcr=use_mcr,
            reader_config=reader_cfg,
        )

        # Ensure neighbor_cache exists even if MCR is disabled
        if not hasattr(self, 'neighbor_cache'):
            self.neighbor_cache = None
        # Separate cache for multi-hop neighbors (not overloading legacy cache)
        self.neighbor_cache_multihop = None

        # Track usage of directional components
        self.use_directional = use_directional

        # Ensure configuration mirrors resolved flags
        self.config.use_flux_messages = self.use_flux_messages
        self.config.use_mass_balance_projection = self.use_mass_balance_projection

        # Initialize components based on configuration
        self._initialize_cross_modal_attention()
        self._initialize_gnn_layers(dropout, use_directional, directional_concat)
        self._initialize_similarity_regularization()
        
        # Initialize GPE projection
        if self.config.use_graph_positional_encodings:
            from .modules.graph_positional_encodings import GraphPositionalEncoder
            gpe_dim = GraphPositionalEncoder(rwse_steps=self.config.gpe_rwse_steps).output_dim
            self.gpe_projection = nn.Linear(gpe_dim, hidden_dim)
        else:
            self.gpe_projection = None

        # Initialize Structural Reconstruction Loss
        self.structural_loss_weight = getattr(self.config, 'structural_loss_weight', 0.0)
        if self.structural_loss_weight > 0:
            self.structural_consistency_loss = NeighborReconstructionLoss(
                hidden_dim=hidden_dim, 
                smiles_dim=smiles_dim
            )
        else:
            self.structural_consistency_loss = None

        # Gene expression FiLM conditioning layers
        if self.config.use_gene_film:
            from .modules.gene_film import GeneExpressionFiLM
            self.gene_film_layers = nn.ModuleList([
                GeneExpressionFiLM(gene_dim, hidden_dim) for _ in range(num_layers)
            ])
        else:
            self.gene_film_layers = None

        # Detection gate for sparse spatial inference
        if self.config.use_detection_gate:
            from .components.builders import build_detection_gate
            self.detection_gate = build_detection_gate(hidden_dim, dropout)
            self.detection_gate_weight = self.config.detection_gate_weight

    def _initialize_cross_modal_attention(self):
        """Initialize cross-modal attention mechanism (Recursive)"""
        if self.config.use_cross_modal_attention:
            combined_dim = self.embedding_dim + self.gene_dim
            
            self.attention_layers = nn.ModuleList()
            
            # Input Layer (Raw Features -> Hidden)
            self.attention_layers.append(CrossModalAttention(
                reaction_dim=combined_dim,  
                metabolite_dim=self.smiles_dim,
                hidden_dim=self.hidden_dim,
                num_heads=self.config.cross_modal_attention_heads,
                dropout=self.config.cross_modal_attention_dropout,
                temperature=self.config.cross_modal_temperature
            ))
            
            # Hidden Layers (Hidden -> Hidden) - One per GNN layer
            for _ in range(self.num_layers):
                self.attention_layers.append(CrossModalAttention(
                    reaction_dim=self.hidden_dim,
                    metabolite_dim=self.hidden_dim,
                    hidden_dim=self.hidden_dim,
                    num_heads=self.config.cross_modal_attention_heads,
                    dropout=self.config.cross_modal_attention_dropout,
                    temperature=self.config.cross_modal_temperature
                ))
            
            # Backward compatibility
            self.cross_modal_attention = self.attention_layers[0]
    
    def _initialize_gnn_layers(self, dropout, use_directional, directional_concat):
        """Initialize GNN layers based on configuration"""
        from .components.builders import build_gnn_layers
        
        # Determine layer type from config or backward-compatible flags
        layer_type = getattr(self.config, 'layer_type', "GAT")
        if layer_type == "GAT" and use_directional:
            layer_type = "Directional"
            
        # PNA layers require degree histogram
        deg = getattr(self.config, 'pna_deg', None)
        
        # Use centralized factory for layer building
        self.convs = build_gnn_layers(
            num_layers=self.num_layers,
            hidden_dim=self.hidden_dim,
            dropout=dropout,
            use_directional=(layer_type == "Directional"),
            directional_concat=directional_concat,
            layer_type=layer_type,
            deg=deg,
            use_residual=getattr(self.config, 'use_residual_connections', False),
            residual_learning_rate=getattr(self.config, 'residual_learning_rate', 0.1),
            use_layer_norm=getattr(self.config, 'use_layer_normalization', True)
        )
    
    def _initialize_similarity_regularization(self):
        """Initialize chemical similarity regularization"""
        if self.config.use_similarity_regularization:
            self.similarity_regularizer = ChemicalSimilarityRegularization(
                weight=self.config.similarity_loss_weight,
                temperature=self.config.similarity_temperature,
                top_k=self.config.similarity_top_k,
                similarity_threshold=0.7
            )
    
    def forward(self, data, return_embeddings=False, compute_physics=True,
                dropout_metabolites=False, mask_target_smiles=False):
        """
        Unified forward pass with enhanced processing pipeline.
        
        Always handles batched data - single samples are converted to batch_size=1.
        
        Args:
            data: PyTorch Geometric data object (automatically batched if needed)
            return_embeddings: whether to return final node embeddings
            compute_physics: whether to return auxiliary physics outputs; the prediction path is unchanged
            dropout_metabolites: whether to apply metabolite dropout (masking) for LOMO training
            mask_target_smiles: whether to zero out metabolite SMILES (Blindfold Test)
            
        Returns:
            pred: [batch_size, num_nodes] metabolite predictions
            delta: [batch_size, num_metabolites] mass balance residuals
            flux_vec: [batch_size, num_reactions] reaction fluxes
            embeddings: [batch_size, num_nodes, hidden_dim] (if requested)
        """
        # Standardize data to always have batch dimensions
        data = standardize_batch_data(data)
        batch_size = get_batch_size(data)
        
        # Use unified batched processing
        return self._forward_batched_unified(data, return_embeddings, compute_physics, 
                                           dropout_metabolites, mask_target_smiles)
    
    def _forward_batched_unified(self, data, return_embeddings=False, compute_physics=True, 
                               dropout_metabolites=False, mask_target_smiles=False):
        """Unified enhanced forward pass for all samples"""
        batch_size = get_batch_size(data)
        
        # Extract batched features (already standardized)
        x_embedding = getattr(data, 'x_embedding', data.x)
        num_nodes = x_embedding.size(1)
        
        # Get or create default features if missing
        if hasattr(data, 'x_gene'):
            x_gene = data.x_gene
        else:
            x_gene = create_default_features(num_nodes, self.gene_dim, batch_size, x_embedding.device)
        
        if hasattr(data, 'x_smiles'):
            x_smiles = data.x_smiles
        else:
            x_smiles = create_default_features(num_nodes, self.smiles_dim, batch_size, x_embedding.device)
            
        # === SAMPLE-MEAN SUBTRACTION (Shortcut Killer) ===
        if self.subtract_sample_mean:
            # Remove global expression level to force relative topological learning
            if x_gene.dim() == 3:
                # [B, N, D] -> Mean over N -> [B, 1, D]
                gene_mean = x_gene.mean(dim=1, keepdim=True)
                x_gene = x_gene - gene_mean
            
        # Capture Ground Truth SMILES for Structural Loss BEFORE masking
        # We use data.metabolite_indices which are shared across batch
        ground_truth_smiles = x_smiles.clone()
        
        # === BLINDFOLD TEST (Masking SMILES) ===
        # Support both boolean (100%) and float (probabilistic) masking
        mask_prob = 0.0
        if isinstance(mask_target_smiles, bool):
            mask_prob = 1.0 if mask_target_smiles else 0.0
        else:
            mask_prob = float(mask_target_smiles)
            
        if self.training and mask_prob > 0.0:
            # Create mask for metabolite nodes
            # 1.0 = Keep, 0.0 = Drop
            # We want to drop with probability 'mask_prob'
            # So we keep if rand > mask_prob
            
            # Create random tensor for each metabolite in the batch
            # Shape: [batch_size, num_nodes]
            rand_tensor = torch.rand((batch_size, num_nodes), device=x_smiles.device)
            
            # Create binary mask: 1 if we keep, 0 if we mask
            keep_mask = (rand_tensor > mask_prob).float().unsqueeze(-1) # [B, N, 1]
            
            # Apply mask ONLY to metabolite indices
            # We need to construct a full node mask
            full_mask = torch.ones_like(x_smiles)
            
            # Get indices for all samples
            # data.metabolite_indices are [num_metabolites] (shared)
            # We need to apply the specific random mask to these columns
            # Expand metabolite indices to batch
            
            # Optimized masking:
            # x_smiles is [B, N, D]
            # keep_mask is [B, N, 1]
            
            # We only want to apply this to metabolites.
            # Create a static mask for metabolite nodes [N]
            is_metabolite = torch.zeros(num_nodes, dtype=torch.bool, device=x_smiles.device)
            is_metabolite[data.metabolite_indices] = True
            is_metabolite = is_metabolite.unsqueeze(0).unsqueeze(-1) # [1, N, 1]
            
            # Final mask:
            # If node is NOT metabolite -> 1.0
            # If node IS metabolite -> keep_mask
            final_mask = torch.where(is_metabolite, keep_mask, torch.tensor(1.0, device=x_smiles.device))
            
            x_smiles = x_smiles * final_mask

        # Features should already be 3D from standardize_batch_data
        assert x_embedding.dim() == 3, f"x_embedding should be 3D, got {x_embedding.dim()}D"
        assert x_gene.dim() == 3, f"x_gene should be 3D, got {x_gene.dim()}D"
        assert x_smiles.dim() == 3, f"x_smiles should be 3D, got {x_smiles.dim()}D"
        
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        reaction_indices = data.reaction_indices
        metabolite_indices = data.metabolite_indices
        
        # Handle NaN values
        x_embedding = check_for_nans(x_embedding, "input x_embedding")
        x_gene = check_for_nans(x_gene, "input x_gene")
        
        # Process batched data
        return self._process_enhanced_pipeline_batched(
            x_embedding, x_gene, x_smiles, edge_index, edge_attr,
            reaction_indices, metabolite_indices, data,
            return_embeddings, compute_physics, batch_size,
            dropout_metabolites=dropout_metabolites,
            ground_truth_smiles=ground_truth_smiles
        )
    
    def _process_enhanced_pipeline(self, x_embedding, x_gene, x_smiles, edge_index, edge_attr,
                                 reaction_indices, metabolite_indices, data,
                                 return_embeddings, compute_physics, dropout_metabolites=False,
                                 ground_truth_smiles=None, _batch_idx=None):
        """Enhanced processing pipeline for single sample"""
        device = x_embedding.device
        num_nodes = x_embedding.size(0)
        
        # === TOPOLOGICAL MASKING (Hub/Teleporter Removal) ===
        # Use pathway_edge_mask if available to restrict GNN message passing
        gnn_edge_index = edge_index
        gnn_edge_attr = edge_attr
        
        if hasattr(data, 'pathway_edge_mask') and data.pathway_edge_mask is not None:
            mask = data.pathway_edge_mask
            # Ensure mask length matches edge_index
            if mask.size(0) == edge_index.size(1):
                gnn_edge_index = edge_index[:, mask]
                if edge_attr is not None:
                    gnn_edge_attr = edge_attr[mask]
        
        # === SYNTHETIC LOMO AUGMENTATION ===
        # 1. Topological Dropout (Graph Structure)
        if dropout_metabolites:
            dropout_prob = 0.2
            metabolite_drop_mask = torch.rand(len(metabolite_indices), device=device) < dropout_prob
            dropped_indices = metabolite_indices[metabolite_drop_mask]
            
            if len(dropped_indices) > 0:
                # Disconnect dropped nodes from GNN (Topological Dropout)
                # Note: We apply this on top of the already masked gnn_edge_index
                src, dst = gnn_edge_index
                node_is_dropped = torch.zeros(num_nodes, device=device, dtype=torch.bool)
                node_is_dropped[dropped_indices] = True
                drop_mask = ~(node_is_dropped[src] | node_is_dropped[dst])
                gnn_edge_index = gnn_edge_index[:, drop_mask]
                if gnn_edge_attr is not None:
                    gnn_edge_attr = gnn_edge_attr[drop_mask]
            
            # 2. SMILES Jitter (Feature Noise) - Prevents memorization of exact chemical ID
            # Apply to ALL metabolites (not just dropped ones) to smooth the latent space
            if self.training:
                jitter_scale = 0.05
                noise = torch.randn_like(x_smiles) * jitter_scale
                # Only apply noise to metabolite nodes
                mask = torch.zeros_like(x_smiles, dtype=torch.bool)
                mask[metabolite_indices] = True
                x_smiles = torch.where(mask, x_smiles + noise, x_smiles)

        # Step 1: Initial feature encoding (always use separate encoders)
        h = torch.zeros(num_nodes, self.hidden_dim, device=device)
        
        # Encode reactions
        reaction_embeddings = x_embedding[reaction_indices]
        reaction_gene_expr = x_gene[reaction_indices]
        combined_features = torch.cat([reaction_embeddings, reaction_gene_expr], dim=1)
        encoded_reactions = self.reaction_encoder(combined_features)
        
        # Encode metabolites
        metabolite_smiles = x_smiles[metabolite_indices]
        encoded_metabolites = self.metabolite_encoder(metabolite_smiles)
        
        # Step 2: Cross-modal attention (if enabled)
        if self.config.use_cross_modal_attention:
            # Create local attention mask [batch_size, num_metabolites, num_reactions]
            # Attention adjacency is independent of the MCR reader cache
            cross_modal_mask = self._create_cross_modal_mask(
                edge_index, metabolite_indices, reaction_indices, 
                batch_size=1, num_metabolites=len(metabolite_indices), num_reactions=len(reaction_indices),
                device=device
            )
            
            # Use original separate features for cross-modal attention
            reaction_features = torch.cat([x_embedding[reaction_indices], x_gene[reaction_indices]], dim=1)
            metabolite_features = x_smiles[metabolite_indices]
            
            enhanced_reactions, enhanced_metabolites = self.attention_layers[0](
                reaction_features, metabolite_features, reaction_indices, metabolite_indices,
                mask=cross_modal_mask
            )
            
            # Fuse enhancements with encoded features instead of replacing them outright
            encoded_reactions = encoded_reactions + self.cross_modal_residual_weight * enhanced_reactions
            encoded_metabolites = encoded_metabolites + self.cross_modal_residual_weight * enhanced_metabolites

        # Populate hidden state with fused encodings
        h[reaction_indices] = encoded_reactions
        h[metabolite_indices] = encoded_metabolites

        # GPE injection: add projected graph positional encodings to h
        if self.gpe_projection is not None and hasattr(data, 'x_gpe') and data.x_gpe is not None:
            x_gpe = data.x_gpe
            if x_gpe.dim() == 3:
                x_gpe = x_gpe.squeeze(0)  # Remove batch dim if present
            if x_gpe.device != device:
                x_gpe = x_gpe.to(device)
            h = h + self.gpe_projection(x_gpe)

        # Initialize mass balance projector if needed
        if self.use_mass_balance_projection and self.mass_balance_projector is None:
            self.initialize_mass_balance_projector(data)
        
        # Step 3: Graph neural network layers
        layer_type = getattr(self.config, 'layer_type', "GAT")
        
        # Prepare bipartite data once if using Relational layers
        if layer_type == "Relational":
            # Check if cache is valid for current edge_index
            if (not hasattr(self, '_bipartite_cache') or 
                self._bipartite_cache['edge_index_ptr'] != gnn_edge_index.data_ptr() or
                self._bipartite_cache['num_nodes'] != num_nodes):
                
                edge_index_dict, edge_attr_dict, rxn_idx, met_idx = BipartiteGraphPreparer.prepare_bipartite_edges(
                    gnn_edge_index, 
                    gnn_edge_attr, 
                    data.node_types, 
                    num_nodes
                )
                self._bipartite_cache = {
                    'edge_index_ptr': gnn_edge_index.data_ptr(),
                    'num_nodes': num_nodes,
                    'edge_index_dict': edge_index_dict,
                    'edge_attr_dict': edge_attr_dict,
                    'rxn_idx': rxn_idx,
                    'met_idx': met_idx
                }
            
            edge_index_dict = self._bipartite_cache['edge_index_dict']
            edge_attr_dict = self._bipartite_cache['edge_attr_dict']
            rxn_idx = self._bipartite_cache['rxn_idx']
            met_idx = self._bipartite_cache['met_idx']
        
        for i, conv in enumerate(self.convs):
            if layer_type == "Relational":
                # Split h into dictionary for HeteroConv
                x_dict = {
                    'reaction': h[rxn_idx],
                    'metabolite': h[met_idx]
                }
                
                # Forward pass through Relational Layer
                # Returns {'reaction': h_r, 'metabolite': h_m}
                out_dict = conv(x_dict, edge_index_dict, edge_attr_dict)
                
                # Merge back into flat h tensor
                # We use scatter/index_put to place results back into their original positions
                h_new = torch.zeros_like(h)
                h_new[rxn_idx] = out_dict['reaction']
                h_new[met_idx] = out_dict['metabolite']
                h = h_new
                
            elif self.config.use_residual_connections:
                h = conv(h, gnn_edge_index, gnn_edge_attr)
            else:
                # Original GNN processing
                h_new = conv(h, gnn_edge_index, gnn_edge_attr)
                h_new = check_for_nans(h_new, "enhanced GNN layer output")
                h = F.elu(h_new)
                h = self.dropout(h)
            
            # === GENE EXPRESSION FiLM CONDITIONING ===
            if self.gene_film_layers is not None:
                h[reaction_indices] = self.gene_film_layers[i](
                    h[reaction_indices], x_gene[reaction_indices]
                )

            # === RECURSIVE CROSS-MODAL ALIGNMENT ===
            # Apply attention after every GNN layer to re-align manifolds
            if self.config.use_cross_modal_attention and i + 1 < len(self.attention_layers):
                # Extract current latent features
                current_rxn = h[reaction_indices]
                current_met = h[metabolite_indices]
                
                # Apply attention (Hidden -> Hidden)
                # Note: We reuse the cross_modal_mask computed earlier
                aligned_rxn, aligned_met = self.attention_layers[i+1](
                    current_rxn, current_met, reaction_indices, metabolite_indices,
                    mask=cross_modal_mask
                )
                
                # Fuse with residual connection
                h[reaction_indices] = current_rxn + self.cross_modal_residual_weight * aligned_rxn
                h[metabolite_indices] = current_met + self.cross_modal_residual_weight * aligned_met
            
            # === ADD FLUX MESSAGES AT SPECIFIED DEPTH ===
            if (self.use_flux_messages and
                i + 1 == self.flux_message_depth and  # 1-indexed depth
                hasattr(data, 'incidence_matrix')):
                
                # Predict fluxes from current reaction embeddings
                reaction_emb_current = h[reaction_indices]  # [num_reactions, hidden_dim]
                flux_predictions = self.reaction_flux_head(reaction_emb_current).squeeze(-1)  # [num_reactions]
                
                # Project fluxes to mass-balanced subspace if enabled
                if self.use_mass_balance_projection and self.mass_balance_projector is not None:
                    flux_balanced = self.mass_balance_projector.project_to_mass_balance(
                        flux_predictions, data.incidence_matrix
                    )
                else:
                    flux_balanced = flux_predictions
                
                # Extract metabolite embeddings
                metabolite_emb_current = h[metabolite_indices]  # [num_metabolites, hidden_dim]
                
                # Add flux messages to metabolite embeddings  
                metabolite_emb_updated = self.flux_message_layer(
                    metabolite_embeddings=metabolite_emb_current.unsqueeze(0),  # Add batch dim
                    reaction_embeddings=reaction_emb_current.unsqueeze(0),      # Add batch dim
                    fluxes=flux_balanced.unsqueeze(0),                          # Add batch dim
                    stoichiometric_matrix=data.incidence_matrix,
                    metabolite_indices=metabolite_indices,
                    reaction_indices=reaction_indices
                ).squeeze(0)  # Remove batch dim
                
                # Update metabolite embeddings in h
                h[metabolite_indices] = metabolite_emb_updated
        
        # Step 4: Output predictions
        pred = torch.zeros(num_nodes, device=device)
        
        if self.config.use_mcr:
            # Use Metabolite-Conditioned Reader
            # Uncomment the next line for debugging tensor shapes:
            # self.debug_mcr_tensor_shapes(data, h, metabolite_indices, reaction_indices)
            metabolite_pred = self._predict_with_mcr_single(
                h, metabolite_indices, reaction_indices, edge_index, x_smiles,
                x_gene=x_gene,
            )
        else:
            # Use traditional MLP head
            metabolite_emb = h[metabolite_indices]
            metabolite_pred = self.metabolite_predictor(metabolite_emb).squeeze(-1)
        
        # === DETECTION GATE ===
        metabolite_emb_for_gate = h[metabolite_indices]
        metabolite_pred, gate_logit = self._apply_detection_gate(
            metabolite_emb_for_gate, metabolite_pred
        )
        if gate_logit is not None:
            self.last_detection_scores = torch.sigmoid(gate_logit).detach()
            # Extract per-sample mask when data has been batched
            sample_mask = data.mask
            if sample_mask is not None and sample_mask.dim() == 2:
                if _batch_idx is not None:
                    sample_mask = sample_mask[_batch_idx]
                else:
                    sample_mask = sample_mask.squeeze(0)
            self.last_detection_loss = self._compute_detection_bce(
                gate_logit, sample_mask, metabolite_indices,
            )
        else:
            self.last_detection_scores = None
            self.last_detection_loss = None

        pred[metabolite_indices] = metabolite_pred

        # Physics-informed outputs
        if compute_physics:
            reaction_emb = h[reaction_indices]
            flux_vec_raw = self.reaction_flux_head(reaction_emb).squeeze(-1)
            
            # === MASS BALANCE PROJECTION ===
            if (self.use_mass_balance_projection and 
                self.mass_balance_projector is not None and 
                hasattr(data, 'incidence_matrix')):
                
                # Project to mass-balanced subspace: exact enforcement of Sv = 0
                flux_vec = self.mass_balance_projector.project_to_mass_balance(
                    flux_vec_raw, data.incidence_matrix
                )
            else:
                # No projection - use raw predictions
                flux_vec = flux_vec_raw
            
            # Compute mass-balance residual
            if hasattr(data, 'incidence_matrix'):
                incidence_matrix = data.incidence_matrix.to(flux_vec.device)
                # If projection was used, residuals should be near zero
                delta = torch.sparse.mm(incidence_matrix, flux_vec.unsqueeze(1)).squeeze()
            else:
                delta = torch.zeros(len(metabolite_indices), device=flux_vec.device)
        else:
            flux_vec = torch.zeros(len(reaction_indices), device=device)
            delta = torch.zeros(len(metabolite_indices), device=device)
        
        # === STRUCTURAL CONSISTENCY LOSS ===
        structural_loss = torch.tensor(0.0, device=device)
        if self.structural_consistency_loss is not None and self.training:
            # We need aggregated neighbor messages. 
            # We can re-use the GNN aggregation or just do a simple aggregation here.
            # For robustness, we do a simple mean aggregation of Reaction neighbors.
            
            # 1. Identify Rxn->Met edges
            # We need standard edge_index here
            src, dst = data.edge_index
            is_rxn_src = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
            is_rxn_src[data.reaction_indices] = True
            
            # Filter edges where source is reaction
            rxn_edges_mask = is_rxn_src[src]
            rxn_src = src[rxn_edges_mask]
            met_dst = dst[rxn_edges_mask]
            
            # 2. Gather Reaction Embeddings (from 'h' - the latent space)
            rxn_features = h[rxn_src]
            
            # 3. Aggregate messages to Metabolites
            # Use scatter_mean to aggregate
            from torch_geometric.utils import scatter
            neighbor_messages = scatter(rxn_features, met_dst, dim=0, dim_size=data.num_nodes, reduce='mean')
            
            # Verify integrity
            neighbor_messages = check_for_nans(neighbor_messages, "neighbor_messages_for_loss")
            
            # 4. Select only metabolite nodes for loss
            met_neighbor_messages = neighbor_messages[data.metabolite_indices]
            
            # 5. Get Target SMILES (Ground Truth)
            # Use original SMILES features (before encoder)
            # Use ground_truth_smiles passed from top level to ensure we use unmasked features
            source_smiles = ground_truth_smiles if ground_truth_smiles is not None else x_smiles
            target_smiles = source_smiles[data.metabolite_indices]
            
            # Verify integrity
            target_smiles = check_for_nans(target_smiles, "target_smiles_for_loss")
            
            # 6. Compute Loss
            structural_loss = self.structural_consistency_loss(met_neighbor_messages, target_smiles) * self.structural_loss_weight
            
        # Store for trainer access
        self.last_structural_loss = structural_loss
        
        # Return embeddings if requested
        embeddings = h if return_embeddings else None

        return pred, delta, flux_vec, embeddings
    
    def _process_enhanced_pipeline_batched(self, x_embedding, x_gene, x_smiles, edge_index, edge_attr,
                                         reaction_indices, metabolite_indices, data,
                                         return_embeddings, compute_physics, batch_size, dropout_metabolites=False,
                                         ground_truth_smiles=None):
        """Enhanced processing pipeline for batched samples"""
        device = x_embedding.device
        num_nodes = x_embedding.size(1)
        
        # For simplicity, process each sample individually and stack results
        # TODO: Implement fully batched processing for all enhancements
        batch_pred = []
        batch_delta = []
        batch_flux = []
        batch_embeddings = [] if return_embeddings else None
        batch_detection_losses = []
        batch_detection_scores = []

        for b in range(batch_size):
            # Extract single sample
            sample_x_embedding = x_embedding[b]
            sample_x_gene = x_gene[b]
            sample_x_smiles = x_smiles[b]
            sample_ground_truth = ground_truth_smiles[b] if ground_truth_smiles is not None else None

            # Process single sample
            pred, delta, flux_vec, embeddings = self._process_enhanced_pipeline(
                sample_x_embedding, sample_x_gene, sample_x_smiles,
                edge_index, edge_attr, reaction_indices, metabolite_indices,
                data, return_embeddings, compute_physics,
                dropout_metabolites=dropout_metabolites,
                ground_truth_smiles=sample_ground_truth,
                _batch_idx=b,
            )

            batch_pred.append(pred)
            batch_delta.append(delta)
            batch_flux.append(flux_vec)
            if return_embeddings:
                batch_embeddings.append(embeddings)

            # Accumulate detection gate outputs
            if self.last_detection_loss is not None:
                batch_detection_losses.append(self.last_detection_loss)
            if self.last_detection_scores is not None:
                batch_detection_scores.append(self.last_detection_scores)

        # Stack results
        pred_stacked = torch.stack(batch_pred, dim=0)
        delta_stacked = torch.stack(batch_delta, dim=0)
        flux_stacked = torch.stack(batch_flux, dim=0)
        embeddings_stacked = torch.stack(batch_embeddings, dim=0) if return_embeddings else None

        # Aggregate detection gate outputs across batch
        if batch_detection_losses:
            self.last_detection_loss = torch.stack(batch_detection_losses).mean()
        else:
            self.last_detection_loss = None
        if batch_detection_scores:
            self.last_detection_scores = torch.stack(batch_detection_scores)
        else:
            self.last_detection_scores = None

        return pred_stacked, delta_stacked, flux_stacked, embeddings_stacked
    
    def extract_embeddings(self, data, normalize=True):
        """
        Extract L2-normalized embeddings after GNN processing for contrastive pretraining.
        
        This method processes data through the enhanced GNN layers and returns the embeddings
        at the optimal stage for contrastive learning (after DirectionalConv/ResidualGNN layers
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
        
        # Process through the enhanced model but only extract embeddings (no predictions)
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
                "Ensure the graph includes SMILES features or disable contrastive objectives."
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
        through the enhanced GNN to obtain augmented embeddings for within-modal contrastive learning.
        
        Args:
            data: PyG Data object (automatically batched if needed)
            augmentation_fn: Function to augment features. If None, applies default augmentation
                           (dropout-based feature masking with enhanced features considered)
            normalize: Whether to apply L2 normalization to embeddings (default: True)
            
        Returns:
            dict: Same format as extract_embeddings() but with augmented features
        """
        # Ensure data has batch dimensions  
        data = standardize_batch_data(data)
        batch_size = get_batch_size(data)
        
        # Create augmented version of the data
        augmented_data = self._apply_feature_augmentation(data, augmentation_fn)
        
        # Process augmented data through the enhanced model
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
        Apply augmentation to input features for contrastive learning with enhanced model support.
        
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
            # Apply default augmentation using configuration parameters
            gene_dropout_rate = (self.config.contrastive_enhanced_dropout_rate 
                               if self.config.use_cross_modal_attention 
                               else self.config.contrastive_gene_dropout_rate)
            embedding_dropout_rate = (self.config.contrastive_enhanced_dropout_rate 
                                    if self.config.use_cross_modal_attention 
                                    else self.config.contrastive_embedding_dropout_rate)
            smiles_dropout_rate = self.config.contrastive_smiles_dropout_rate
            
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
            
            # Enhanced model: augment SMILES embeddings using configurable rate
            if hasattr(augmented_data, 'x_smiles'):
                x_smiles = augmented_data.x_smiles
                if x_smiles.dim() == 3:  # Batched: [batch_size, num_nodes, smiles_dim]
                    mask = torch.rand_like(x_smiles) > smiles_dropout_rate
                    augmented_data.x_smiles = x_smiles * mask.float()
                else:  # Single sample: [num_nodes, smiles_dim]
                    mask = torch.rand_like(x_smiles) > smiles_dropout_rate
                    augmented_data.x_smiles = x_smiles * mask.float()
        
        return augmented_data
    
    def enable_contrastive_learning(self):
        """
        Enable contrastive learning for this enhanced model.
        
        Uses the configuration parameters from self.config.
        """
        # Update config to enable contrastive learning
        self.config.use_contrastive_pretraining = True
        
        # Initialize contrastive module if available
        try:
            from gaze._reference.losses.contrastive import create_contrastive_module

            self.contrastive_module = create_contrastive_module(
                hidden_dim=self.hidden_dim,
                temperature=self.config.contrastive_temperature,
                projection_dim=self.config.contrastive_projection_dim
            )
        except ImportError:
            print("Warning: Contrastive losses module not available. Contrastive learning disabled.")
            self.contrastive_module = None
    
    def compute_similarity_loss(self, embeddings, smiles_embeddings, metabolite_indices, mask=None):
        """
        Compute chemical similarity regularization loss.
        
        Args:
            embeddings: learned node embeddings from forward pass
            smiles_embeddings: original ChemBERTa SMILES embeddings
            metabolite_indices: indices of metabolite nodes
            mask: optional mask for which metabolites to include
            
        Returns:
            similarity_loss: regularization loss to encourage chemical similarity preservation
        """
        if not self.config.use_similarity_regularization:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        return self.similarity_regularizer(
            embeddings, smiles_embeddings, metabolite_indices, mask
        )
    
    def _predict_with_mcr_single(self, h, metabolite_indices, reaction_indices, edge_index, x_smiles,
                                x_gene=None):
        """Generate metabolite predictions using MCR for a single sample.

        Args:
            h: [num_nodes, hidden_dim] GNN node embeddings
            metabolite_indices: Metabolite node indices
            reaction_indices: Reaction node indices
            edge_index: [2, num_edges] Graph connectivity
            x_smiles: [num_nodes, smiles_dim] or [1, num_nodes, smiles_dim] SMILES embeddings
            x_gene: [num_nodes, gene_dim] Raw gene expression (optional, for gene skip)

        Returns:
            torch.Tensor: [num_metabolites] concentration predictions
        """
        # Normalize SMILES tensor to 2D
        x_smiles_2d = self._normalize_smiles_tensor(x_smiles)
        self._validate_mcr_inputs(x_smiles_2d, metabolite_indices)

        # Apply descriptor shuffling for ablation if enabled
        smiles_input = self._apply_descriptor_shuffling(x_smiles_2d, metabolite_indices)

        # Extract metabolite descriptors and reaction neighbors
        metabolite_smiles = smiles_input[metabolite_indices]
        neighbor_result = self._extract_neighbor_context(
            h, edge_index, metabolite_indices, reaction_indices
        )
        neighbor_embeddings, neighbor_mask, hop_distances = neighbor_result

        # Extract gene expression neighbors for gene skip connection
        gene_neighbors = None
        if self.config.use_gene_skip and x_gene is not None:
            gene_neighbor_result = self._extract_neighbor_context(
                x_gene, edge_index, metabolite_indices, reaction_indices
            )
            gene_neighbors = gene_neighbor_result[0]  # [1, M, N, gene_dim]

        # Generate predictions via MCR attention mechanism
        # Use optimized single-sample path if batch_size=1
        if neighbor_embeddings.size(0) == 1:
            return self.metabolite_reader.forward_single_optimized(
                metabolite_smiles=metabolite_smiles,
                reaction_neighbors=neighbor_embeddings.squeeze(0),
                neighbor_mask=neighbor_mask.squeeze(0) if neighbor_mask is not None else None,
                hop_distances=hop_distances.squeeze(0) if hop_distances is not None else None,
                gene_neighbors=gene_neighbors.squeeze(0) if gene_neighbors is not None else None,
            )
        else:
            return self.metabolite_reader(
                metabolite_smiles=metabolite_smiles.unsqueeze(0),
                reaction_neighbors=neighbor_embeddings,
                neighbor_mask=neighbor_mask,
                hop_distances=hop_distances,
                gene_neighbors=gene_neighbors,
            ).squeeze(0)
    
    def _predict_with_mcr_batched(self, h, metabolite_indices, reaction_indices, edge_index, x_smiles,
                                  x_gene=None):
        """Generate metabolite predictions using MCR for batched samples.

        Args:
            h: [batch_size, num_nodes, hidden_dim] Batched GNN node embeddings
            metabolite_indices: Metabolite node indices
            reaction_indices: Reaction node indices
            edge_index: [2, num_edges] Graph connectivity (shared across batch)
            x_smiles: [batch_size, num_nodes, smiles_dim] Batched SMILES embeddings
            x_gene: [batch_size, num_nodes, gene_dim] Raw gene expression (optional, for gene skip)

        Returns:
            torch.Tensor: [batch_size, num_metabolites] concentration predictions
        """
        # Apply descriptor shuffling for ablation if enabled
        smiles_input = self._apply_descriptor_shuffling(x_smiles, metabolite_indices)

        # Extract metabolite descriptors for all samples
        metabolite_smiles = smiles_input[:, metabolite_indices]

        # Extract reaction neighbor context for attention
        neighbor_result = self._extract_neighbor_context(
            h, edge_index, metabolite_indices, reaction_indices, batched=True
        )
        neighbor_embeddings, neighbor_mask, hop_distances = neighbor_result

        # Extract gene expression neighbors for gene skip connection
        gene_neighbors = None
        if self.config.use_gene_skip and x_gene is not None:
            gene_neighbor_result = self._extract_neighbor_context(
                x_gene, edge_index, metabolite_indices, reaction_indices, batched=True
            )
            gene_neighbors = gene_neighbor_result[0]  # [B, M, N, gene_dim]

        # Generate batched predictions via MCR
        return self.metabolite_reader(
            metabolite_smiles=metabolite_smiles,
            reaction_neighbors=neighbor_embeddings,
            neighbor_mask=neighbor_mask,
            hop_distances=hop_distances,
            gene_neighbors=gene_neighbors,
        )
    
    def _normalize_smiles_tensor(self, x_smiles: torch.Tensor) -> torch.Tensor:
        """Normalize SMILES tensor to 2D format for processing."""
        if x_smiles.dim() == 3 and x_smiles.size(0) == 1:
            return x_smiles.squeeze(0)  # [1, num_nodes, smiles_dim] -> [num_nodes, smiles_dim]
        return x_smiles
    
    def _validate_mcr_inputs(self, x_smiles_2d: torch.Tensor, metabolite_indices: torch.Tensor) -> None:
        """Validate MCR input tensor shapes and indices."""
        if x_smiles_2d.dim() != 2:
            raise ValueError(f"Expected 2D SMILES tensor, got {x_smiles_2d.dim()}D with shape {x_smiles_2d.shape}")
        
        max_metabolite_idx = max(metabolite_indices.cpu().numpy())
        if max_metabolite_idx >= x_smiles_2d.size(0):
            raise IndexError(f"Max metabolite index {max_metabolite_idx} >= SMILES tensor size {x_smiles_2d.size(0)}")
    
    def _apply_descriptor_shuffling(self, x_smiles: torch.Tensor, metabolite_indices: torch.Tensor) -> torch.Tensor:
        """Apply SMILES descriptor shuffling for ablation studies if enabled."""
        if not self.config.reader_shuffle_descriptors:
            return x_smiles
        
        # Ensure tensor has batch dimension for shuffling function
        if x_smiles.dim() == 2:
            x_smiles = x_smiles.unsqueeze(0)
            return shuffle_metabolite_descriptors(x_smiles, metabolite_indices, seed=42).squeeze(0)
        else:
            return shuffle_metabolite_descriptors(x_smiles, metabolite_indices, seed=42)
    
    def _extract_neighbor_context(self, h: torch.Tensor, edge_index: torch.Tensor,
                                 metabolite_indices: torch.Tensor, reaction_indices: torch.Tensor,
                                 batched: bool = False):
        """Extract reaction neighbor embeddings and mask for MCR attention.

        Returns:
            tuple: (neighbor_embeddings, neighbor_mask) for 1-hop, or
                   (neighbor_embeddings, neighbor_mask, hop_distances) for multi-hop
        """
        use_multihop = self.config.effective_max_hops > 1

        # Initialize neighbor cache if needed (first forward pass)
        if self.config.use_mcr:
            if use_multihop and self.neighbor_cache_multihop is None:
                self.neighbor_cache_multihop = precompute_neighbor_indices_multihop(
                    edge_index=edge_index,
                    metabolite_indices=metabolite_indices,
                    reaction_indices=reaction_indices,
                    max_neighbors=self.config.reader_max_neighbors,
                    max_hops=self.config.effective_max_hops,
                    exclude_metabolites=None,
                )
            elif not use_multihop and self.neighbor_cache is None:
                self.neighbor_cache = precompute_neighbor_indices(
                    edge_index=edge_index,
                    metabolite_indices=metabolite_indices,
                    reaction_indices=reaction_indices,
                    max_neighbors=self.config.reader_max_neighbors,
                    two_hop=self.config.reader_two_hop,
                    exclude_metabolites=None,
                )

        # Ensure h has batch dimension for neighbor extraction
        if not batched and h.dim() == 2:
            h = h.unsqueeze(0)

        if use_multihop and self.neighbor_cache_multihop is not None:
            return extract_reaction_neighbors_cached_multihop(
                neighbor_cache=self.neighbor_cache_multihop,
                metabolite_indices=metabolite_indices,
                reaction_embeddings=h,
                max_neighbors=self.config.reader_max_neighbors,
            )
        elif self.neighbor_cache is not None:
            emb, mask = extract_reaction_neighbors_cached(
                neighbor_cache=self.neighbor_cache,
                metabolite_indices=metabolite_indices,
                reaction_embeddings=h,
                max_neighbors=self.config.reader_max_neighbors,
            )
            return emb, mask, None
        else:
            # Emergency fallback
            print("Warning: Using non-cached neighbor extraction. Performance may be degraded.")
            emb, mask = extract_reaction_neighbors(
                edge_index=edge_index,
                metabolite_indices=metabolite_indices,
                reaction_indices=reaction_indices,
                reaction_embeddings=h,
                max_neighbors=self.config.reader_max_neighbors,
                two_hop=self.config.reader_two_hop,
            )
            return emb, mask, None
    
    def debug_mcr_tensor_shapes(self, data, h, metabolite_indices, reaction_indices):
        """Debug method to print all tensor shapes in MCR pipeline."""
        print(f"=== MCR Tensor Shape Debug ===")
        print(f"data.x_smiles.shape: {data.x_smiles.shape if hasattr(data, 'x_smiles') else 'None'}")
        print(f"h.shape: {h.shape}")
        print(f"len(metabolite_indices): {len(metabolite_indices)}")
        print(f"len(reaction_indices): {len(reaction_indices)}")
        print(f"max(metabolite_indices): {max(metabolite_indices)}")
        print(f"max(reaction_indices): {max(reaction_indices)}")
        print(f"data.edge_index.shape: {data.edge_index.shape}")
        print(f"data.batch_size: {getattr(data, 'batch_size', 'Not set')}")
        
        if hasattr(data, 'x_smiles'):
            x_smiles = data.x_smiles
            print(f"x_smiles dimensionality: {x_smiles.dim()}D")
            if x_smiles.dim() == 3:
                print(f"x_smiles batch shape: [batch_size={x_smiles.size(0)}, num_nodes={x_smiles.size(1)}, smiles_dim={x_smiles.size(2)}]")
            elif x_smiles.dim() == 2:
                print(f"x_smiles 2D shape: [num_nodes={x_smiles.size(0)}, smiles_dim={x_smiles.size(1)}]")
        print(f"===============================")
    
    def clear_neighbor_cache(self):
        """Clear the MCR neighbor cache to force recomputation.

        This is needed for LOMO cross-validation where we need to exclude
        held-out metabolites from the neighbor computation.
        """
        if self.config.use_mcr:
            self.neighbor_cache = None
            self.neighbor_cache_multihop = None

    def rebuild_neighbor_cache(self, edge_index, metabolite_indices, reaction_indices,
                             exclude_metabolites=None):
        """Rebuild the neighbor cache with optional metabolite exclusion.

        Args:
            edge_index: Graph connectivity
            metabolite_indices: Metabolite node indices
            reaction_indices: Reaction node indices
            exclude_metabolites: Set of metabolite indices to exclude
        """
        if self.config.use_mcr:
            if self.config.effective_max_hops > 1:
                self.neighbor_cache_multihop = precompute_neighbor_indices_multihop(
                    edge_index=edge_index,
                    metabolite_indices=metabolite_indices,
                    reaction_indices=reaction_indices,
                    max_neighbors=self.config.reader_max_neighbors,
                    max_hops=self.config.effective_max_hops,
                    exclude_metabolites=exclude_metabolites,
                )
                # Clear legacy cache to avoid confusion
                self.neighbor_cache = None
            else:
                self.neighbor_cache = precompute_neighbor_indices(
                    edge_index=edge_index,
                    metabolite_indices=metabolite_indices,
                    reaction_indices=reaction_indices,
                    max_neighbors=self.config.reader_max_neighbors,
                    two_hop=self.config.reader_two_hop,
                    exclude_metabolites=exclude_metabolites,
                )
                self.neighbor_cache_multihop = None
    
    def _create_cross_modal_mask(self, edge_index, metabolite_indices, reaction_indices,
                               batch_size, num_metabolites, num_reactions, device):
        """Use graph adjacency, independently of the capped MCR reader cache.

        Every call uses the same one-hop attention neighborhood. Reader neighbor
        limits and cache initialization must not change the prediction function.
        """
        mask = torch.zeros(batch_size, num_metabolites, num_reactions,
                           dtype=torch.bool, device=device)
        if edge_index.numel() == 0 or num_metabolites == 0 or num_reactions == 0:
            return mask
        edge_index = edge_index.to(device)
        metabolite_indices = metabolite_indices.to(device)
        reaction_indices = reaction_indices.to(device)
        size = max(int(edge_index.max()), int(metabolite_indices.max()),
                   int(reaction_indices.max())) + 1
        met_to_local = torch.full((size,), -1, dtype=torch.long, device=device)
        rxn_to_local = torch.full((size,), -1, dtype=torch.long, device=device)
        met_to_local[metabolite_indices] = torch.arange(num_metabolites, device=device)
        rxn_to_local[reaction_indices] = torch.arange(num_reactions, device=device)
        src, dst = edge_index
        forward = (met_to_local[src] >= 0) & (rxn_to_local[dst] >= 0)
        reverse = (rxn_to_local[src] >= 0) & (met_to_local[dst] >= 0)
        mask[:, met_to_local[src[forward]], rxn_to_local[dst[forward]]] = True
        mask[:, met_to_local[dst[reverse]], rxn_to_local[src[reverse]]] = True
        return mask

    def verify_mcr_parameters(self, dataset):
        """Verify MCR has no metabolite-specific parameters to avoid ID leakage."""
        if self.config.use_mcr and self.metabolite_reader is not None:
            # Get number of metabolites from dataset
            first_data = dataset[0]
            num_metabolites = len(first_data.metabolite_indices)
            
            return verify_no_metabolite_specific_parameters(self.metabolite_reader, num_metabolites)
        return True
    
    def get_config_summary(self):
        """Get a summary of the current model configuration"""
        return {
            'model_type': 'EnhancedConcatGNNWithSMILES',
            'base_params': {
                'embedding_dim': self.embedding_dim,
                'gene_dim': self.gene_dim,
                'smiles_dim': self.smiles_dim,
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
            },
            'enhancements': self.config.to_dict(),
            'config_str': str(self.config)
        }
