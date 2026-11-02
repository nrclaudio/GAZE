import torch
import torch.nn as nn
import torch.nn.functional as F

class NeighborReconstructionLoss(nn.Module):
    """
    Structural Consistency Loss: Forces the model to learn Manifold Alignment.
    
    Objective:
    Predict a metabolite's chemical identity (SMILES embedding) using ONLY 
    the aggregated information from its neighboring reactions.
    
    Hypothesis:
    - On a Real Graph: The functional context (enzymes) predicts the chemistry.
    - On a Random Graph: The context is noise, prediction fails, loss explodes.
    """
    def __init__(self, hidden_dim, smiles_dim, project_dim=None):
        super().__init__()
        
        output_dim = project_dim if project_dim is not None else smiles_dim
        
        # The "Chemical Decoder" MLP
        # Input: Aggregated Reaction Neighbor Features (from GNN)
        # Output: Predicted SMILES Embedding
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, neighbor_messages, target_smiles):
        """
        Args:
            neighbor_messages: [num_metabolites, hidden_dim] 
                               Aggregated features from reaction neighbors.
            target_smiles: [num_metabolites, smiles_dim]
                           The ground-truth chemical embedding.
        """
        # Predict SMILES from biology
        predicted_smiles = self.decoder(neighbor_messages)
        
        # Normalize if comparing cosine similarity, or use MSE
        # Using MSE on normalized vectors is strictly equivalent to Cosine Loss
        pred_norm = F.normalize(predicted_smiles, p=2, dim=-1)
        target_norm = F.normalize(target_smiles, p=2, dim=-1)
        
        # Loss = 1 - CosineSimilarity
        # This is more stable for high-dimensional embeddings than raw MSE
        loss = 1.0 - (pred_norm * target_norm).sum(dim=-1).mean()
        
        return loss
