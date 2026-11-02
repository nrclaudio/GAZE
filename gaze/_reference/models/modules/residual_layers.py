import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from ...utils.tensor_utils import check_for_nans

class ResidualGNNLayer(nn.Module):
    """
    GNN layer with residual connections to prevent over-smoothing.
    
    Follows the architecture: ELU(GNN(x) + x)
    This helps preserve node-specific information through deep networks.
    """
    
    def __init__(self, in_dim, out_dim, gnn_type='gat', edge_dim=1, heads=4, 
                 dropout=0.2, residual_learning_rate=0.1, use_layer_norm=True):
        super().__init__()
        
        # GNN layer (defaulting to GAT)
        if gnn_type == 'gat':
            self.gnn = GATv2Conv(in_dim, out_dim, edge_dim=edge_dim, heads=heads, 
                               concat=False, dropout=dropout)
        else:
            raise NotImplementedError(f"GNN type {gnn_type} not implemented")
        
        # Residual projection if input/output dimensions don't match
        self.residual_proj = None
        if in_dim != out_dim:
            self.residual_proj = nn.Linear(in_dim, out_dim)
            
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(out_dim) if use_layer_norm else None
        
        self.dropout = nn.Dropout(dropout)
        self.residual_learning_rate = residual_learning_rate
        
    def forward(self, x, edge_index, edge_attr=None):
        """
        Args:
            x: [num_nodes, in_dim] or [batch_size * num_nodes, in_dim] node features
            edge_index: [2, num_edges] edge indices
            edge_attr: [num_edges, edge_dim] edge attributes (optional)
        """
        # Store input for residual connection
        residual = x
        
        # Apply GNN layer
        out = self.gnn(x, edge_index, edge_attr)
        out = check_for_nans(out, "residual GNN layer output")
        
        # Project residual if needed
        if self.residual_proj is not None:
            residual = self.residual_proj(residual)
        
        # Add residual connection
        # We use a learnable weight or fixed weight if specified, but simple addition is standard
        out = out + residual
        
        # Apply LayerNorm if enabled (Pre-activation)
        if self.layer_norm is not None:
            out = self.layer_norm(out)
        
        # Apply activation and dropout
        out = F.elu(out)
        out = self.dropout(out)
        
        return out


class MultiScaleResidualBlock(nn.Module):
    """
    Multi-scale residual block that processes information at different scales
    to capture both local and global patterns.
    """
    
    def __init__(self, hidden_dim, num_scales=3, edge_dim=1, heads=4, dropout=0.2):
        super().__init__()
        
        self.num_scales = num_scales
        
        # Multi-scale GNN layers with different receptive fields
        self.scale_layers = nn.ModuleList()
        for scale in range(num_scales):
            # Use different number of heads for different scales
            scale_heads = max(1, heads // (scale + 1))
            layer = ResidualGNNLayer(
                hidden_dim, hidden_dim, 
                edge_dim=edge_dim, heads=scale_heads,
                dropout=dropout, residual_learning_rate=0.1 / (scale + 1)
            )
            self.scale_layers.append(layer)
        
        # Attention mechanism to combine multi-scale features
        self.scale_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4, dropout=dropout, batch_first=True
        )
        
        # Final projection
        self.final_proj = nn.Linear(hidden_dim * num_scales, hidden_dim)
        self.final_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x, edge_index, edge_attr=None):
        """Apply multi-scale processing with attention-based combination"""
        scale_outputs = []
        
        # Apply each scale layer
        for scale_layer in self.scale_layers:
            scale_out = scale_layer(x, edge_index, edge_attr)
            scale_outputs.append(scale_out)
        
        # Stack scale outputs: [num_scales, num_nodes, hidden_dim] 
        stacked_outputs = torch.stack(scale_outputs, dim=0)
        
        # Transpose for attention: [num_nodes, num_scales, hidden_dim]
        stacked_outputs = stacked_outputs.transpose(0, 1)
        
        # Apply self-attention across scales for each node
        attended_output, _ = self.scale_attention(
            stacked_outputs, stacked_outputs, stacked_outputs
        )
        
        # Combine attended features: [num_nodes, num_scales * hidden_dim]
        combined = attended_output.flatten(start_dim=1)
        
        # Final projection back to hidden_dim
        output = self.final_proj(combined)
        output = self.final_norm(output)
        
        # Add residual connection to original input
        if output.size() == x.size():
            output = output + 0.1 * x
        
        return output


class AdaptiveResidualGNN(nn.Module):
    """
    Adaptive residual GNN that learns different residual weights for different node types.
    
    This is particularly useful for heterogeneous graphs where different node types
    may benefit from different levels of residual connections.
    """
    
    def __init__(self, in_dim, out_dim, num_node_types=2, edge_dim=1, heads=4, dropout=0.2):
        super().__init__()
        
        self.num_node_types = num_node_types
        
        # Base GNN layer
        self.gnn = GATv2Conv(in_dim, out_dim, edge_dim=edge_dim, heads=heads,
                           concat=False, dropout=dropout)
        
        # Type-specific residual weights
        self.type_residual_weights = nn.Parameter(torch.ones(num_node_types) * 0.1)
        
        # Type-specific layer norms
        self.type_layer_norms = nn.ModuleList([
            nn.LayerNorm(out_dim) for _ in range(num_node_types)
        ])
        
        # Residual projection if needed
        self.residual_proj = None
        if in_dim != out_dim:
            self.residual_proj = nn.Linear(in_dim, out_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index, edge_attr=None, node_types=None):
        """
        Args:
            x: [num_nodes, in_dim] node features
            edge_index: [2, num_edges] edge indices  
            edge_attr: [num_edges, edge_dim] edge attributes (optional)
            node_types: [num_nodes] node type indices (0=reaction, 1=metabolite, etc.)
        """
        # Store input for residual connection
        residual = x
        
        # Apply GNN layer
        out = self.gnn(x, edge_index, edge_attr)
        out = check_for_nans(out, "adaptive residual GNN output")
        out = F.elu(out)
        out = self.dropout(out)
        
        # Project residual if needed
        if self.residual_proj is not None:
            residual = self.residual_proj(residual)
        
        # Apply type-specific residual connections and layer norms
        if node_types is not None:
            # Group by node type
            for node_type in range(self.num_node_types):
                type_mask = (node_types == node_type)
                if type_mask.any():
                    type_indices = type_mask.nonzero(as_tuple=True)[0]
                    
                    # Apply type-specific residual weight
                    residual_weight = self.type_residual_weights[node_type]
                    out[type_indices] = out[type_indices] + residual_weight * residual[type_indices]
                    
                    # Apply type-specific layer norm
                    out[type_indices] = self.type_layer_norms[node_type](out[type_indices])
        else:
            # Fallback: use average residual weight
            avg_weight = torch.mean(self.type_residual_weights)
            out = out + avg_weight * residual
            out = self.type_layer_norms[0](out)  # Use first layer norm as fallback
        
        return out


class ResidualGNNStack(nn.Module):
    """
    Stack of residual GNN layers with optional skip connections across multiple layers.
    """
    
    def __init__(self, hidden_dim, num_layers=3, edge_dim=1, heads=4, dropout=0.2,
                 use_multi_layer_skip=True, use_adaptive_residual=False, num_node_types=2):
        super().__init__()
        
        self.num_layers = num_layers
        self.use_multi_layer_skip = use_multi_layer_skip
        self.use_adaptive_residual = use_adaptive_residual
        
        # Create residual layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if use_adaptive_residual:
                layer = AdaptiveResidualGNN(
                    hidden_dim, hidden_dim, num_node_types=num_node_types,
                    edge_dim=edge_dim, heads=heads, dropout=dropout
                )
            else:
                layer = ResidualGNNLayer(
                    hidden_dim, hidden_dim, edge_dim=edge_dim, heads=heads,
                    dropout=dropout, residual_learning_rate=0.1
                )
            self.layers.append(layer)
        
        # Multi-layer skip connections
        if use_multi_layer_skip and num_layers > 1:
            self.skip_connections = nn.ModuleList([
                nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)
            ])
            self.skip_weights = nn.Parameter(torch.ones(num_layers - 1) * 0.1)
        
        self.final_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x, edge_index, edge_attr=None, node_types=None):
        """
        Args:
            x: [num_nodes, hidden_dim] input features
            edge_index: [2, num_edges] edge indices
            edge_attr: [num_edges, edge_dim] edge attributes (optional)
            node_types: [num_nodes] node type indices (optional, for adaptive residual)
        """
        layer_outputs = [x]  # Store outputs from each layer
        current = x
        
        # Apply each layer
        for i, layer in enumerate(self.layers):
            if self.use_adaptive_residual:
                current = layer(current, edge_index, edge_attr, node_types)
            else:
                current = layer(current, edge_index, edge_attr)
            
            layer_outputs.append(current)
        
        # Add multi-layer skip connections
        if self.use_multi_layer_skip and len(layer_outputs) > 2:
            final_output = current
            
            # Add weighted skip connections from previous layers
            for i in range(len(self.skip_connections)):
                skip_input = layer_outputs[i]  # Output from layer i
                skip_weight = self.skip_weights[i]
                skip_contribution = self.skip_connections[i](skip_input)
                final_output = final_output + skip_weight * skip_contribution
        else:
            final_output = current
        
        # Final normalization
        final_output = self.final_norm(final_output)
        
        return final_output