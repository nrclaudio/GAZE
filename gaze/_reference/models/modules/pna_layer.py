import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import PNAConv
from torch_geometric.utils import degree
from ...utils.tensor_utils import check_for_nans

class PNALayer(nn.Module):
    """
    Principal Neighbourhood Aggregation (PNA) Layer.
    
    Why this works for Metabolism:
    - Mass balance is additive: PNA supports 'sum' aggregation (unlike GAT/GCN 'mean').
    - Heterophily: High enzyme expression -> Low substrate, High product. 
      PNA's multiple aggregators (min, max, mean, std) can capture these quantitative relationships
      better than smoothing filters.
    
    Args:
        in_dim (int): Input feature dimension
        out_dim (int): Output feature dimension
        deg (torch.Tensor): Histogram of in-degrees from the training set. 
                            Required by PNA to normalize scalers.
        edge_dim (int): Edge feature dimension (default: 1)
        dropout (float): Dropout probability
    """
    def __init__(self, in_dim, out_dim, deg, edge_dim=1, dropout=0.2, 
                 aggregators=['mean', 'min', 'max', 'std', 'sum'], 
                 scalers=['identity', 'amplification', 'attenuation']):
        super().__init__()
        
        # PNAConv internally uses MLPs for the aggregators
        self.conv = PNAConv(in_dim, out_dim, 
                           aggregators=aggregators, 
                           scalers=scalers, 
                           deg=deg,
                           edge_dim=edge_dim, 
                           towers=4,          # Use multiple towers (like heads in GAT)
                           pre_layers=1,      # MLP layers before aggregation
                           post_layers=1,     # MLP layers after aggregation
                           divide_input=False)
                           
        self.dropout = nn.Dropout(dropout)
        
        # LayerNorm is crucial for PNA stability, especially with 'amplification' scalers
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, edge_index, edge_attr=None):
        out = self.conv(x, edge_index, edge_attr)
        out = check_for_nans(out, "PNA layer output")
        
        # Normalization -> Activation -> Dropout pattern
        out = self.norm(out)
        out = F.elu(out)
        out = self.dropout(out)
        return out

    @staticmethod
    def compute_degree_histogram(data):
        """
        Helper to compute in-degree histogram from a PyG Data object.
        Pass the result of this to the PNALayer constructor.
        """
        # Get edge targets (col) to calculate in-degree
        row, col = data.edge_index
        deg = degree(col, data.num_nodes, dtype=torch.long)
        
        # Compute histogram (count occurrences of each degree)
        deg_histogram = torch.bincount(deg, minlength=1)
        
        return deg_histogram
