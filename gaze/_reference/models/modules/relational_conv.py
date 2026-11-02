import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, GATv2Conv
import numpy as np
from ...utils.tensor_utils import check_for_nans

class RelationalMetabolicConv(nn.Module):
    """
    Bipartite-Aware Relational Convolution for Metabolic Networks.
    
    Features:
    1. Heterogeneous Parameters: Separate weights for Rxn->Met and Met->Rxn.
    2. Stoichiometric Attention: Uses edge_attr (squashed stoichiometry) to gate messages.
    3. Heterophily Awareness: Explicit separate processing of self-features vs. neighbor-features.
    """
    def __init__(self, hidden_dim, heads=4, dropout=0.2, edge_dim=1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Define the heterogeneous convolutions
        # We use GATv2Conv with add_self_loops=False because we handle self-loops explicitely
        # to solve the Negative Homophily problem.
        self.hetero_conv = HeteroConv({
            # Reaction -> Metabolite (Product/Substrate link)
            ('reaction', 'to', 'metabolite'): GATv2Conv(
                hidden_dim, hidden_dim, 
                edge_dim=edge_dim, 
                heads=heads, 
                concat=False, 
                dropout=dropout,
                add_self_loops=False 
            ),
            # Metabolite -> Reaction (Substrate/Product link)
            ('metabolite', 'to', 'reaction'): GATv2Conv(
                hidden_dim, hidden_dim, 
                edge_dim=edge_dim, 
                heads=heads, 
                concat=False, 
                dropout=dropout,
                add_self_loops=False
            )
        }, aggr='sum')
        
        # Self-Loop Processors (The "Trust Myself" path)
        # Allows the model to weight its own features differently from the (dissimilar) neighbors
        self.self_reaction = nn.Linear(hidden_dim, hidden_dim)
        self.self_metabolite = nn.Linear(hidden_dim, hidden_dim)
        
        # Gating/Fusion Layer (Root Weight Bias)
        # Decides how to combine [Self, Neighbor_Message]
        self.fusion_reaction = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fusion_metabolite = nn.Linear(2 * hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.norm_reaction = nn.LayerNorm(hidden_dim)
        self.norm_metabolite = nn.LayerNorm(hidden_dim)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        """
        Args:
            x_dict: {'reaction': [N_r, D], 'metabolite': [N_m, D]}
            edge_index_dict: Dictionary of edge indices for each relation
            edge_attr_dict: Dictionary of edge attributes (stoichiometry)
        """
        # 1. Compute Messages from Neighbors (The Graph Signal)
        # This returns a dictionary of outputs {key: output_tensor}
        # Note: We must pass edge_attr to the GAT layers
        out_dict = self.hetero_conv(x_dict, edge_index_dict, edge_attr_dict)
        
        # 2. Compute Self-Features (The Node Signal)
        self_r = self.self_reaction(x_dict['reaction'])
        self_m = self.self_metabolite(x_dict['metabolite'])
        
        # 3. Fuse Signals (Heterophily-Aware Aggregation)
        # If neighbors are noise (Random Graph), the gate can learn to ignore out_dict
        
        # Reaction Fusion
        # Handle case where a node might not receive messages (isolated)
        msg_r = out_dict.get('reaction', torch.zeros_like(self_r))
        msg_r = check_for_nans(msg_r, "relational_msg_r")
        
        combined_r = torch.cat([self_r, msg_r], dim=1)
        h_r = self.fusion_reaction(combined_r)
        h_r = self.norm_reaction(h_r + x_dict['reaction']) # Residual
        h_r = self.dropout(torch.relu(h_r))
        h_r = check_for_nans(h_r, "relational_h_r")
        
        # Metabolite Fusion
        msg_m = out_dict.get('metabolite', torch.zeros_like(self_m))
        msg_m = check_for_nans(msg_m, "relational_msg_m")
        
        combined_m = torch.cat([self_m, msg_m], dim=1)
        h_m = self.fusion_metabolite(combined_m)
        h_m = self.norm_metabolite(h_m + x_dict['metabolite']) # Residual
        h_m = self.dropout(torch.relu(h_m))
        h_m = check_for_nans(h_m, "relational_h_m")
        
        return {
            'reaction': h_r,
            'metabolite': h_m
        }

class BipartiteGraphPreparer:
    """Helper to split homogeneous graph data into bipartite components."""
    
    @staticmethod
    def prepare_bipartite_edges(edge_index, edge_attr, node_types, num_nodes):
        """Extract r2m and m2r edges based on node types."""
        # Check inputs
        if node_types is None:
            raise ValueError("node_types must be provided for Relational Conv.")
        if len(node_types) != num_nodes:
            raise ValueError(f"Length of node_types ({len(node_types)}) does not match num_nodes ({num_nodes})")
            
        node_types = np.array(node_types)
        src, dst = edge_index
        
        if src.max() >= num_nodes or dst.max() >= num_nodes:
             raise ValueError(f"Edge index contains indices ({src.max()}, {dst.max()}) >= num_nodes ({num_nodes})")
        
        device = edge_index.device
        
        # Identify node indices
        # We assume node_types is a list of strings
        # Convert to boolean masks for efficiency if possible, but list comprehension is robust
        is_rxn = torch.tensor([t == 'reaction' for t in node_types], device=device)
        is_met = torch.tensor([t == 'metabolite' for t in node_types], device=device)
        
        rxn_indices = torch.where(is_rxn)[0]
        met_indices = torch.where(is_met)[0]
        
        # Create local mappings (Global Index -> Local Type Index)
        # Initialize with -1 to catch errors
        rxn_map = torch.full((num_nodes,), -1, device=device, dtype=torch.long)
        rxn_map[rxn_indices] = torch.arange(len(rxn_indices), device=device)
        
        met_map = torch.full((num_nodes,), -1, device=device, dtype=torch.long)
        met_map[met_indices] = torch.arange(len(met_indices), device=device)
        
        # --- Edge Type 1: Reaction -> Metabolite ---
        # Source is Reaction, Target is Metabolite
        r2m_mask = is_rxn[src] & is_met[dst]
        r2m_idx = torch.stack([rxn_map[src[r2m_mask]], met_map[dst[r2m_mask]]])
        r2m_attr = edge_attr[r2m_mask] if edge_attr is not None else None
        
        # --- Edge Type 2: Metabolite -> Reaction ---
        # Source is Metabolite, Target is Reaction
        m2r_mask = is_met[src] & is_rxn[dst]
        m2r_idx = torch.stack([met_map[src[m2r_mask]], rxn_map[dst[m2r_mask]]])
        m2r_attr = edge_attr[m2r_mask] if edge_attr is not None else None
        
        if (r2m_idx < 0).any() or (m2r_idx < 0).any():
            raise ValueError("BipartiteGraphPreparer generated invalid (-1) indices. Check node_types and edge_index alignment.")
        
        edge_index_dict = {
            ('reaction', 'to', 'metabolite'): r2m_idx,
            ('metabolite', 'to', 'reaction'): m2r_idx
        }
        
        edge_attr_dict = {
            ('reaction', 'to', 'metabolite'): r2m_attr,
            ('metabolite', 'to', 'reaction'): m2r_attr
        }
        
        return edge_index_dict, edge_attr_dict, rxn_indices, met_indices
