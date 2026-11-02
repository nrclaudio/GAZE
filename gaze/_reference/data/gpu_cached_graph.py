import torch
from typing import Dict, Any, Optional

class GPUCachedGraph:
    """
    Manages GPU-cached static graph components to eliminate repeated tensor transfers.
    
    Key optimization: Static graph tensors (x_embedding, x_smiles, edge_index, etc.) 
    are moved to GPU once and reused across all samples, avoiding costly .clone() + .to(device) 
    operations in the training loop.
    
    Usage:
        # In dataset.__init__:
        self.gpu_cache = GPUCachedGraph(self.graph, device)
        
        # In dataset.__getitem__:
        static_data = self.gpu_cache.get_static_data()
        # Only add per-sample tensors (x_gene, y, mask)
    """
    
    _instances: Dict[int, 'GPUCachedGraph'] = {}  # Cache per device
    
    def __init__(self, graph, device: Optional[torch.device] = None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.device = device
        device_id = device.index if device.type == 'cuda' else -1
        
        # Check if we already have a cached instance for this device
        if device_id in self._instances:
            existing = self._instances[device_id]
            if self._graphs_equal(existing.original_graph, graph):
                # Reuse existing cache
                self.__dict__.update(existing.__dict__)
                return
        
        self.original_graph = graph
        self._cache_static_tensors(graph)
        
        # Store weak reference to avoid memory leaks
        self._instances[device_id] = self
    
    def _graphs_equal(self, graph1, graph2) -> bool:
        """Quick check if two graphs have same structure/tensors"""
        try:
            if not torch.equal(graph1.edge_index, graph2.edge_index):
                return False
            if hasattr(graph1, 'x') and hasattr(graph2, 'x'):
                return torch.equal(graph1.x, graph2.x)
            return True
        except (AttributeError, RuntimeError, TypeError):
            return False
    
    def _cache_static_tensors(self, graph):
        """Move static graph components to GPU and cache them"""
        print(f"📦 Caching static graph tensors on {self.device}...")
        
        # Core graph structure
        self.x_embedding = graph.x.to(self.device, non_blocking=True) if hasattr(graph, 'x') else None
        self.x_smiles = graph.x_smiles.to(self.device, non_blocking=True) if hasattr(graph, 'x_smiles') else None
        self.edge_index = graph.edge_index.to(self.device, non_blocking=True)
        self.edge_attr = graph.edge_attr.to(self.device, non_blocking=True) if hasattr(graph, 'edge_attr') else None
        
        # Metadata (keep on CPU - lightweight)
        self.node_types = graph.node_types if hasattr(graph, 'node_types') else None
        self.hmdb_ids = graph.hmdb_ids if hasattr(graph, 'hmdb_ids') else None
        self.kegg_ids = graph.kegg_ids if hasattr(graph, 'kegg_ids') else None
        
        # Pre-computed gene operation tensors (for fast correlation loss)
        if hasattr(graph, 'reaction_gene_indices'):
            self.reaction_gene_indices = graph.reaction_gene_indices.to(self.device, non_blocking=True)
            self.reaction_gene_operations = graph.reaction_gene_operations.to(self.device, non_blocking=True)
            self.reaction_gene_masks = graph.reaction_gene_masks.to(self.device, non_blocking=True)
            self.reaction_node_indices = graph.reaction_node_indices.to(self.device, non_blocking=True)
            
            # Metadata for gene operations
            self.max_genes_per_reaction = getattr(graph, 'max_genes_per_reaction', None)
            self.num_valid_reactions = getattr(graph, 'num_valid_reactions', None)
            
            print("  ✅ Cached tensor gene operations for fast correlation loss")
        else:
            print("  ⚠️  No tensor gene operations found - correlation may be slow")
        
        # Calculate memory usage
        total_bytes = 0
        for attr_name in ['x_embedding', 'x_smiles', 'edge_index', 'edge_attr', 
                         'reaction_gene_indices', 'reaction_gene_operations', 
                         'reaction_gene_masks', 'reaction_node_indices']:
            tensor = getattr(self, attr_name, None)
            if tensor is not None:
                total_bytes += tensor.nelement() * tensor.element_size()
        
        print(f"  📊 Cached {total_bytes / 1024**2:.1f} MB of static graph data on GPU")
    
    def get_static_data(self, force_cpu=False) -> Dict[str, Any]:
        """
        Return dictionary of cached static tensors.
        
        Args:
            force_cpu: If True, return CPU copies (safe for multiprocessing)
        """
        static_data = {}
        
        if force_cpu:
            # Return CPU copies for multiprocessing safety
            if self.x_embedding is not None:
                static_data['x_embedding'] = self.x_embedding.cpu()
            if self.x_smiles is not None:
                static_data['x_smiles'] = self.x_smiles.cpu()
            
            static_data['edge_index'] = self.edge_index.cpu()
            if self.edge_attr is not None:
                static_data['edge_attr'] = self.edge_attr.cpu()
            
            # Add gene operation tensors as CPU copies
            if hasattr(self, 'reaction_gene_indices'):
                static_data.update({
                    'reaction_gene_indices': self.reaction_gene_indices.cpu(),
                    'reaction_gene_operations': self.reaction_gene_operations.cpu(),
                    'reaction_gene_masks': self.reaction_gene_masks.cpu(),
                    'reaction_node_indices': self.reaction_node_indices.cpu(),
                    'max_genes_per_reaction': self.max_genes_per_reaction,
                    'num_valid_reactions': self.num_valid_reactions,
                })
        else:
            # Return GPU tensors (main process only)
            if self.x_embedding is not None:
                static_data['x_embedding'] = self.x_embedding
            if self.x_smiles is not None:
                static_data['x_smiles'] = self.x_smiles
            
            static_data['edge_index'] = self.edge_index
            if self.edge_attr is not None:
                static_data['edge_attr'] = self.edge_attr
            
            # Add gene operation tensors if available
            if hasattr(self, 'reaction_gene_indices'):
                static_data.update({
                    'reaction_gene_indices': self.reaction_gene_indices,
                    'reaction_gene_operations': self.reaction_gene_operations,
                    'reaction_gene_masks': self.reaction_gene_masks,
                    'reaction_node_indices': self.reaction_node_indices,
                    'max_genes_per_reaction': self.max_genes_per_reaction,
                    'num_valid_reactions': self.num_valid_reactions,
                })
        
        # Add metadata (CPU objects)
        if self.node_types is not None:
            static_data['node_types'] = self.node_types
        
        return static_data
    
    def update_device(self, new_device: torch.device):
        """Move cached tensors to a different device if needed"""
        if new_device == self.device:
            return
        
        print(f"🔄 Moving cached graph from {self.device} to {new_device}...")
        self.device = new_device
        
        # Move all cached tensors
        for attr_name in ['x_embedding', 'x_smiles', 'edge_index', 'edge_attr',
                         'reaction_gene_indices', 'reaction_gene_operations',
                         'reaction_gene_masks', 'reaction_node_indices']:
            tensor = getattr(self, attr_name, None)
            if tensor is not None:
                setattr(self, attr_name, tensor.to(new_device, non_blocking=True))
    
    @classmethod
    def clear_cache(cls):
        """Clear all cached instances (useful for testing)"""
        cls._instances.clear()
        print("🧹 Cleared GPU graph cache")

def get_or_create_gpu_cache(graph, device: Optional[torch.device] = None) -> GPUCachedGraph:
    """
    Get or create a GPU cache for the given graph and device.
    
    This is the main entry point for datasets to use.
    """
    return GPUCachedGraph(graph, device) 