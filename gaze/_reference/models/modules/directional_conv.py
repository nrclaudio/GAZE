import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv

class DirectionalConv(nn.Module):
    """
    A graph convolution layer that explicitly handles edge directionality
    by processing incoming and outgoing edges separately.
    """
    def __init__(self, in_dim, out_dim, edge_dim=1, heads=4, dropout=0.2, concat=True):
        super().__init__()
        # If concat=False, both convs output out_dim
        # If concat=True, both convs output out_dim/2 so the concatenated result is out_dim
        conv_out_dim = out_dim if not concat else out_dim // 2
        
        self.in_conv = GATv2Conv(in_dim, conv_out_dim, edge_dim=edge_dim, 
                                heads=heads, concat=False, dropout=dropout)
        self.out_conv = GATv2Conv(in_dim, conv_out_dim, edge_dim=edge_dim, 
                                 heads=heads, concat=False, dropout=dropout)
        self.concat = concat
    
    def forward(self, x, edge_index, edge_attr):
        # Process incoming edges (original direction)
        h_in = self.in_conv(x, edge_index, edge_attr)
        
        # Process outgoing edges (reversed direction)
        edge_index_rev = torch.stack([edge_index[1], edge_index[0]], dim=0)
        h_out = self.out_conv(x, edge_index_rev, edge_attr)
        
        if self.concat:
            # Combine the representations
            h_combined = torch.cat([h_in, h_out], dim=1)
            return h_combined
        else:
            # Average the representations
            return (h_in + h_out) / 2 