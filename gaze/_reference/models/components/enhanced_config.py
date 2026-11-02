import torch
from dataclasses import dataclass
from typing import Iterable

@dataclass
class ArchitecturalFeatureConfig:
    """Configuration for architectural enhancements to the base ConcatGNN model.
    
    Provides granular control over individual architectural components for systematic
    ablation studies and performance benchmarking. When all features are disabled,
    the model becomes functionally equivalent to the baseline implementation.
    
    Design Philosophy:
        - Independent toggles: Each feature can be enabled/disabled separately
        - Composable architecture: Multiple features can be combined seamlessly
        - Ablation-friendly: Single-feature configurations for controlled experiments
        - Performance tuning: Hyperparameters exposed for each major component
    """
    
    # === ATTENTION MECHANISMS ===
    # Cross-modal attention: Enables communication between reaction and metabolite embeddings
    use_cross_modal_attention: bool = True
    cross_modal_attention_heads: int = 4
    cross_modal_attention_dropout: float = 0.1
    cross_modal_temperature: float = 1.0
    cross_modal_residual_weight: float = 0.5
    
    # === NEURAL ARCHITECTURE ENHANCEMENTS ===
    # Residual connections: Prevent over-smoothing in deep GNN layers
    use_residual_connections: bool = True
    use_layer_normalization: bool = True
    residual_learning_rate: float = 0.1  # Residual blend factor (0=skip, 1=full)
    
    # GNN Layer type (GAT, Directional, PNA)
    layer_type: str = "GAT"
    pna_deg: torch.Tensor = None  # Degree histogram for PNA layers
    
    # === CHEMICAL SIMILARITY REGULARIZATION ===
    # Enforce consistency between learned embeddings and chemical structure
    use_similarity_regularization: bool = True
    similarity_loss_weight: float = 0.1
    similarity_temperature: float = 0.1
    similarity_top_k: int = 5  # Number of nearest neighbors for similarity computation
    
    # === STRUCTURAL CONSISTENCY (NEW) ===
    # Reconstruct SMILES from Reaction Neighbors to force manifold alignment
    structural_loss_weight: float = 0.0

    # === PHYSICS-INFORMED COMPONENTS ===
    # Flux-based message passing: Incorporates metabolic flux predictions into GNN layers
    use_flux_messages: bool = False
    flux_message_depth: int = 1  # GNN layer for flux message injection (1-indexed)
    
    # Mass balance projection: Enforces stoichiometric constraints on flux predictions
    use_mass_balance_projection: bool = False
    
    # === METABOLITE-CONDITIONED READER (MCR) ===
    # Alternative attention-based readout head using chemical descriptors
    use_mcr: bool = False
    reader_hidden_dim: int = 128        # Attention projection dimension
    reader_mlp_hidden: int = 128        # Final prediction MLP dimension
    reader_two_hop: bool = False        # Include reactions connected via other metabolites
    reader_normalize_descriptors: bool = True  # L2 normalize SMILES before attention
    reader_max_neighbors: int = 10      # Memory-efficiency limit for neighbors (optimized for speed)
    reader_shuffle_descriptors: bool = False  # Ablation: break chemical similarity
    reader_max_hops: int = 1            # Max hops for multi-hop MCR (1=original, 2+=multi-hop)
    reader_hop_decay_init: float = -0.5 # Initial hop bias decay per hop level

    # === GRAPH POSITIONAL ENCODINGS (GPE) ===
    # Explicit topology features derived from graph structure
    use_graph_positional_encodings: bool = False
    gpe_rwse_steps: int = 16            # Number of random walk steps for RWSE
    
    # === CONTRASTIVE LEARNING ===
    # Contrastive pretraining support for self-supervised learning
    use_contrastive_pretraining: bool = False
    contrastive_temperature: float = 0.07       # Temperature for InfoNCE loss
    contrastive_projection_dim: int = 128       # Dimension for contrastive projections
    contrastive_cross_modal_weight: float = 1.0 # Weight for cross-modal contrastive loss
    contrastive_within_modal_weight: float = 0.5 # Weight for within-modal contrastive loss
    contrastive_symmetric_loss: bool = True     # Bidirectional contrastive alignment
    contrastive_min_batch_size: int = 16       # Minimum batch size for effective contrastive learning
    
    # === GENE EXPRESSION SENSITIVITY ===
    use_gene_skip: bool = False     # Skip-connect raw gene expression to MCR head
    use_gene_film: bool = False     # FiLM conditioning on gene expression per GNN layer

    # === DETECTION GATE (for sparse spatial inference) ===
    use_detection_gate: bool = False      # Sigmoid gate for sparse spatial output
    detection_gate_weight: float = 1.0    # BCE loss weight for gate supervision

    # Feature augmentation parameters for contrastive learning
    contrastive_gene_dropout_rate: float = 0.1   # Dropout rate for gene expression features
    contrastive_embedding_dropout_rate: float = 0.1  # Dropout rate for reaction embeddings  
    contrastive_smiles_dropout_rate: float = 0.05    # Conservative dropout for SMILES (preserve chemical info)
    contrastive_enhanced_dropout_rate: float = 0.15  # Higher dropout when using cross-modal attention
    
    @property
    def effective_max_hops(self) -> int:
        """Effective max hops bridging legacy reader_two_hop and new reader_max_hops."""
        if self.reader_max_hops > 1:
            return self.reader_max_hops
        if self.reader_two_hop:
            return 2
        return 1

    def to_dict(self) -> dict:
        """Convert configuration to dictionary for logging and serialization.
        
        Returns:
            dict: Organized configuration dictionary with grouped parameters
        """
        return {
            # Core architectural toggles
            'features': {
                'cross_modal_attention': self.use_cross_modal_attention,
                'residual_connections': self.use_residual_connections,
                'layer_normalization': self.use_layer_normalization,
                'similarity_regularization': self.use_similarity_regularization,
                'flux_messages': self.use_flux_messages,
                'mass_balance_projection': self.use_mass_balance_projection,
                'metabolite_conditioned_reader': self.use_mcr,
                'contrastive_pretraining': self.use_contrastive_pretraining,
                'graph_positional_encodings': self.use_graph_positional_encodings,
                'gene_skip': self.use_gene_skip,
                'gene_film': self.use_gene_film,
                'detection_gate': self.use_detection_gate,
            },
            
            # Attention hyperparameters
            'attention': {
                'cross_modal_heads': self.cross_modal_attention_heads,
                'cross_modal_dropout': self.cross_modal_attention_dropout,
                'cross_modal_temperature': self.cross_modal_temperature,
                'cross_modal_residual_weight': self.cross_modal_residual_weight,
            },
            
            # Architecture hyperparameters
            'architecture': {
                'residual_learning_rate': self.residual_learning_rate,
                'layer_type': self.layer_type,
            },
            
            # Regularization parameters
            'regularization': {
                'similarity_weight': self.similarity_loss_weight,
                'similarity_temperature': self.similarity_temperature,
                'similarity_top_k': self.similarity_top_k,
                'structural_loss_weight': self.structural_loss_weight,
            },
            
            # Physics-informed parameters
            'physics': {
                'flux_message_depth': self.flux_message_depth,
            },
            
            # MCR-specific parameters
            'mcr': {
                'hidden_dim': self.reader_hidden_dim,
                'mlp_hidden': self.reader_mlp_hidden,
                'two_hop': self.reader_two_hop,
                'normalize_descriptors': self.reader_normalize_descriptors,
                'max_neighbors': self.reader_max_neighbors,
                'shuffle_descriptors': self.reader_shuffle_descriptors,
                'max_hops': self.reader_max_hops,
                'hop_decay_init': self.reader_hop_decay_init,
            },

            # GPE parameters
            'gpe': {
                'enabled': self.use_graph_positional_encodings,
                'rwse_steps': self.gpe_rwse_steps,
            },
            
            # Contrastive learning parameters
            'contrastive': {
                'enabled': self.use_contrastive_pretraining,
                'temperature': self.contrastive_temperature,
                'projection_dim': self.contrastive_projection_dim,
                'cross_modal_weight': self.contrastive_cross_modal_weight,
                'within_modal_weight': self.contrastive_within_modal_weight,
                'symmetric_loss': self.contrastive_symmetric_loss,
                'min_batch_size': self.contrastive_min_batch_size,
                'gene_dropout_rate': self.contrastive_gene_dropout_rate,
                'embedding_dropout_rate': self.contrastive_embedding_dropout_rate,
                'smiles_dropout_rate': self.contrastive_smiles_dropout_rate,
                'enhanced_dropout_rate': self.contrastive_enhanced_dropout_rate,
            }
        }
    
    @classmethod
    def original_model_config(cls) -> 'ArchitecturalFeatureConfig':
        """Create configuration equivalent to the original baseline model.
        
        Disables all architectural enhancements to provide a clean baseline
        for performance comparison and ablation studies.
        
        Returns:
            ArchitecturalFeatureConfig: Configuration with all enhancements disabled
        """
        return cls(
            # Disable all architectural enhancements
            use_cross_modal_attention=False,
            use_residual_connections=False,
            use_layer_normalization=False,
            use_similarity_regularization=False,
            use_flux_messages=False,
            use_mass_balance_projection=False,
            use_mcr=False,
            use_contrastive_pretraining=False,
        )
    
    @classmethod
    def _feature_toggles(cls):
        """Return mapping of feature names to their enabling operations."""
        def _enable_cross_modal_attention(cfg):
            cfg.use_cross_modal_attention = True
            cfg.cross_modal_attention_dropout = max(cfg.cross_modal_attention_dropout, 0.2)

        return {
            'cross_modal_attention': _enable_cross_modal_attention,
            'residual_connections': lambda cfg: (
                setattr(cfg, 'use_residual_connections', True),
                setattr(cfg, 'use_layer_normalization', True)
            ),
            'similarity_regularization': lambda cfg: setattr(cfg, 'use_similarity_regularization', True),
            'flux_messages': lambda cfg: setattr(cfg, 'use_flux_messages', True),
            'mass_balance_projection': lambda cfg: setattr(cfg, 'use_mass_balance_projection', True),
            'mcr': lambda cfg: setattr(cfg, 'use_mcr', True),
            'contrastive_pretraining': lambda cfg: setattr(cfg, 'use_contrastive_pretraining', True),
            'graph_positional_encodings': lambda cfg: setattr(cfg, 'use_graph_positional_encodings', True),
            'mcr_multihop': lambda cfg: (
                setattr(cfg, 'use_mcr', True),
                setattr(cfg, 'reader_max_hops', 2),
            ),
            'gene_skip': lambda cfg: setattr(cfg, 'use_gene_skip', True),
            'gene_film': lambda cfg: setattr(cfg, 'use_gene_film', True),
            'detection_gate': lambda cfg: setattr(cfg, 'use_detection_gate', True),
        }

    @classmethod
    def with_features(cls, feature_names: Iterable[str]) -> 'ArchitecturalFeatureConfig':
        """Create configuration enabling the requested set of features."""
        toggles = cls._feature_toggles()
        feature_list = list(feature_names)
        if not feature_list:
            return cls.original_model_config()

        unknown = [name for name in feature_list if name not in toggles]
        if unknown:
            available = ', '.join(sorted(toggles.keys()))
            if len(unknown) == 1:
                raise ValueError(f"Unknown feature '{unknown[0]}'. Available: {available}")
            unknown_features = ', '.join(unknown)
            raise ValueError(f"Unknown features '{unknown_features}'. Available: {available}")

        config = cls.original_model_config()
        for name in feature_list:
            toggles[name](config)
        return config

    @classmethod
    def ablation_config(cls, feature_name: str) -> 'ArchitecturalFeatureConfig':
        """Create configuration with single architectural feature enabled.

        Generates isolated configurations for systematic ablation studies by enabling
        only one enhancement at a time against the baseline model.

        Args:
            feature_name: Name of feature to enable. Options:
                - 'cross_modal_attention': Cross-modal attention mechanism
                - 'residual_connections': Residual connections + layer norm
                - 'similarity_regularization': Chemical similarity loss
                - 'flux_messages': Physics-informed flux message passing
                - 'mass_balance_projection': Stoichiometric constraint enforcement  
                - 'mcr': Metabolite-Conditioned Reader
                - 'contrastive_pretraining': Contrastive pretraining support

        Returns:
            ArchitecturalFeatureConfig: Baseline config with specified feature enabled

        Raises:
            ValueError: If feature_name is not recognized
        """
        return cls.with_features([feature_name])
    
    def __str__(self) -> str:
        """Generate human-readable string representation of active features.
        
        Returns:
            str: Compact description of enabled architectural components
        """
        enabled_features = []
        
        # Core architectural features
        if self.use_cross_modal_attention:
            enabled_features.append("CrossModal")
        if self.use_residual_connections:
            enabled_features.append("Residual")
        if self.use_layer_normalization and not self.use_residual_connections:
            # Only show LayerNorm separately if not bundled with Residual
            enabled_features.append("LayerNorm")
        if self.use_similarity_regularization:
            enabled_features.append("SimReg")
        if self.use_flux_messages:
            enabled_features.append("FluxMsg")
        if self.use_mass_balance_projection:
            enabled_features.append("MassBalProj")
        if self.use_mcr:
            if self.reader_max_hops > 1:
                enabled_features.append(f"MCR({self.reader_max_hops}hop)")
            else:
                enabled_features.append("MCR")
        if self.use_graph_positional_encodings:
            enabled_features.append("GPE")
        if self.use_contrastive_pretraining:
            enabled_features.append("Contrastive")
        if self.use_gene_skip:
            enabled_features.append("GeneSkip")
        if self.use_gene_film:
            enabled_features.append("GeneFiLM")
        if self.use_detection_gate:
            enabled_features.append("DetGate")

        if not enabled_features:
            return "BaselineModel"
        
        return f"Enhanced[{'+'.join(enabled_features)}]"
    
    def get_enabled_features(self) -> list[str]:
        """Get list of enabled feature names for programmatic access.
        
        Returns:
            list[str]: Names of all enabled architectural features
        """
        features = []
        
        if self.use_cross_modal_attention:
            features.append('cross_modal_attention')
        if self.use_residual_connections:
            features.append('residual_connections')
        if self.use_layer_normalization:
            features.append('layer_normalization')
        if self.use_similarity_regularization:
            features.append('similarity_regularization')
        if self.use_flux_messages:
            features.append('flux_messages')
        if self.use_mass_balance_projection:
            features.append('mass_balance_projection')
        if self.use_mcr:
            features.append('mcr')
        if self.use_graph_positional_encodings:
            features.append('graph_positional_encodings')
        if self.use_contrastive_pretraining:
            features.append('contrastive_pretraining')
        if self.use_gene_skip:
            features.append('gene_skip')
        if self.use_gene_film:
            features.append('gene_film')
        if self.use_detection_gate:
            features.append('detection_gate')

        return features

    @classmethod
    def from_dict(cls, data: dict) -> 'ArchitecturalFeatureConfig':
        """Create a configuration instance from a flat or nested dictionary.
        
        Handles both the nested structure produced by to_dict() and flat structures
        from direct instantiation.
        """
        if not data:
            return cls()
            
        # If it's the nested format from to_dict()
        if 'features' in data and isinstance(data['features'], dict):
            flat_data = {}
            
            # Map nested features back to flat attributes
            feature_map = {
                'cross_modal_attention': 'use_cross_modal_attention',
                'residual_connections': 'use_residual_connections',
                'layer_normalization': 'use_layer_normalization',
                'similarity_regularization': 'use_similarity_regularization',
                'flux_messages': 'use_flux_messages',
                'mass_balance_projection': 'use_mass_balance_projection',
                'metabolite_conditioned_reader': 'use_mcr',
                'contrastive_pretraining': 'use_contrastive_pretraining',
                'graph_positional_encodings': 'use_graph_positional_encodings',
                'gene_skip': 'use_gene_skip',
                'gene_film': 'use_gene_film',
                'detection_gate': 'use_detection_gate',
            }
            for k, v in data['features'].items():
                if k in feature_map: flat_data[feature_map[k]] = v
                
            # Map other nested groups
            if 'attention' in data:
                attn = data['attention']
                if 'cross_modal_heads' in attn: flat_data['cross_modal_attention_heads'] = attn['cross_modal_heads']
                if 'cross_modal_dropout' in attn: flat_data['cross_modal_attention_dropout'] = attn['cross_modal_dropout']
                if 'cross_modal_temperature' in attn: flat_data['cross_modal_temperature'] = attn['cross_modal_temperature']
                if 'cross_modal_residual_weight' in attn: flat_data['cross_modal_residual_weight'] = attn['cross_modal_residual_weight']
                
            if 'architecture' in data:
                arch = data['architecture']
                if 'residual_learning_rate' in arch: flat_data['residual_learning_rate'] = arch['residual_learning_rate']
                if 'layer_type' in arch: flat_data['layer_type'] = arch['layer_type']
                
            if 'regularization' in data:
                reg = data['regularization']
                if 'similarity_weight' in reg: flat_data['similarity_loss_weight'] = reg['similarity_weight']
                if 'similarity_temperature' in reg: flat_data['similarity_temperature'] = reg['similarity_temperature']
                if 'similarity_top_k' in reg: flat_data['similarity_top_k'] = reg['similarity_top_k']
                if 'structural_loss_weight' in reg: flat_data['structural_loss_weight'] = reg['structural_loss_weight']
                
            if 'physics' in data:
                phys = data['physics']
                if 'flux_message_depth' in phys: flat_data['flux_message_depth'] = phys['flux_message_depth']
                
            if 'mcr' in data:
                mcr = data['mcr']
                if 'hidden_dim' in mcr: flat_data['reader_hidden_dim'] = mcr['hidden_dim']
                if 'mlp_hidden' in mcr: flat_data['reader_mlp_hidden'] = mcr['mlp_hidden']
                if 'two_hop' in mcr: flat_data['reader_two_hop'] = mcr['two_hop']
                if 'normalize_descriptors' in mcr: flat_data['reader_normalize_descriptors'] = mcr['normalize_descriptors']
                if 'max_neighbors' in mcr: flat_data['reader_max_neighbors'] = mcr['max_neighbors']
                if 'shuffle_descriptors' in mcr: flat_data['reader_shuffle_descriptors'] = mcr['shuffle_descriptors']
                if 'max_hops' in mcr: flat_data['reader_max_hops'] = mcr['max_hops']
                if 'hop_decay_init' in mcr: flat_data['reader_hop_decay_init'] = mcr['hop_decay_init']

            if 'gpe' in data:
                gpe = data['gpe']
                if 'enabled' in gpe: flat_data['use_graph_positional_encodings'] = gpe['enabled']
                if 'rwse_steps' in gpe: flat_data['gpe_rwse_steps'] = gpe['rwse_steps']
                
            # Filter valid dataclass fields
            import inspect
            fields = {f.name for f in inspect.signature(cls).parameters.values()}
            final_data = {k: v for k, v in flat_data.items() if k in fields}
            return cls(**final_data)
            
        # Fallback for flat dictionary (filter unknown args)
        import inspect
        fields = {f.name for f in inspect.signature(cls).parameters.values()}
        final_data = {k: v for k, v in data.items() if k in fields}
        return cls(**final_data)


# Backward compatibility alias
EnhancedModelConfig = ArchitecturalFeatureConfig
