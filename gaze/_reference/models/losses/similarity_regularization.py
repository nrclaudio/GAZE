import torch
import torch.nn as nn
import torch.nn.functional as F

class ChemicalSimilarityRegularization(nn.Module):
    """
    Chemical similarity regularization loss using ONLY contrastive learning.
    
    Encourages metabolites with very similar ChemBERTa SMILES embeddings (>0.85 similarity)
    to be pulled closer together in learned space, while pushing dissimilar ones apart.
    
    Removed the preservation loss component which was too restrictive.
    """
    
    def __init__(self, weight=0.1, temperature=0.1, top_k=5, similarity_threshold=0.85):
        super().__init__()
        self.weight = weight
        self.temperature = temperature
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        
    def forward(self, learned_embeddings, smiles_embeddings, metabolite_indices, mask=None):
        """
        Args:
            learned_embeddings: [num_nodes, hidden_dim] or [batch_size, num_nodes, hidden_dim] - learned node embeddings
            smiles_embeddings: [num_nodes, smiles_dim] or [batch_size, num_nodes, smiles_dim] - ChemBERTa embeddings
            metabolite_indices: indices of metabolite nodes in the graph
            mask: optional mask for which metabolites to include
            
        Returns:
            similarity_loss: scalar regularization loss
        """
        is_batched = len(learned_embeddings.shape) == 3
        device = learned_embeddings.device
        
        if is_batched:
            batch_size = learned_embeddings.size(0)
            total_loss = 0.0
            
            for b in range(batch_size):
                batch_learned = learned_embeddings[b]  # [num_nodes, hidden_dim]
                batch_smiles = smiles_embeddings[b] if smiles_embeddings.dim() == 3 else smiles_embeddings  # [num_nodes, smiles_dim]
                batch_mask = mask[b] if mask is not None and mask.dim() == 2 else mask
                
                loss = self._compute_single_sample_loss(
                    batch_learned, batch_smiles, metabolite_indices, batch_mask
                )
                total_loss += loss
            
            return total_loss / batch_size
        else:
            return self._compute_single_sample_loss(
                learned_embeddings, smiles_embeddings, metabolite_indices, mask
            )
    
    def _compute_single_sample_loss(self, learned_embeddings, smiles_embeddings, metabolite_indices, mask=None):
        """Compute similarity loss for a single sample"""
        device = learned_embeddings.device
        
        # Extract metabolite embeddings only
        if isinstance(metabolite_indices, list):
            metabolite_indices = torch.tensor(metabolite_indices, device=device)
        
        metabolite_learned = learned_embeddings[metabolite_indices]  # [num_metabolites, hidden_dim]
        metabolite_smiles = smiles_embeddings[metabolite_indices]    # [num_metabolites, smiles_dim]
        
        # Apply mask if provided
        if mask is not None:
            metabolite_mask = mask[metabolite_indices]
            if metabolite_mask.any():
                metabolite_learned = metabolite_learned[metabolite_mask]
                metabolite_smiles = metabolite_smiles[metabolite_mask]
            else:
                return torch.tensor(0.0, device=device, requires_grad=True)
        
        num_metabolites = metabolite_learned.size(0)
        
        if num_metabolites < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Compute pairwise similarities in SMILES space (cosine similarity)
        smiles_similarities = self._compute_cosine_similarity_matrix(metabolite_smiles)
        
        # Compute pairwise similarities in learned embedding space
        learned_similarities = self._compute_cosine_similarity_matrix(metabolite_learned)
        
        # Create mask for similar pairs (above threshold in SMILES space)
        similar_pairs_mask = (smiles_similarities > self.similarity_threshold) & (smiles_similarities < 0.999)  # Exclude self-similarity
        
        if not similar_pairs_mask.any():
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Use ONLY contrastive loss (removed problematic preservation loss)
        contrastive_loss = self._compute_contrastive_loss(
            metabolite_learned, smiles_similarities
        )
        
        # Return only contrastive loss
        total_loss = contrastive_loss
        
        return self.weight * total_loss
    
    def _compute_cosine_similarity_matrix(self, embeddings):
        """Compute pairwise cosine similarities between all embeddings"""
        # Normalize embeddings
        normalized = F.normalize(embeddings, p=2, dim=1)
        
        # Compute cosine similarity matrix
        similarity_matrix = torch.matmul(normalized, normalized.T)
        
        return similarity_matrix
    
    def _compute_similarity_preservation_loss(self, smiles_similarities, learned_similarities, mask):
        """
        Loss that encourages preservation of pairwise similarities between SMILES and learned spaces
        """
        # Extract similarities for valid pairs only
        smiles_sim_valid = smiles_similarities[mask]
        learned_sim_valid = learned_similarities[mask]
        
        # L2 loss between similarity values
        preservation_loss = F.mse_loss(learned_sim_valid, smiles_sim_valid)
        
        return preservation_loss
    
    def _compute_contrastive_loss(self, learned_embeddings, smiles_similarities):
        """
        Contrastive loss that pulls together embeddings of chemically similar metabolites
        and pushes apart embeddings of dissimilar metabolites
        """
        num_metabolites = learned_embeddings.size(0)
        
        if num_metabolites < 2:
            return torch.tensor(0.0, device=learned_embeddings.device, requires_grad=True)
        
        total_loss = 0.0
        num_pairs = 0
        
        for i in range(num_metabolites):
            # Get top-k most similar metabolites (excluding self)
            similarities_i = smiles_similarities[i]
            similarities_i[i] = -1  # Exclude self
            
            # Get top-k similar and bottom-k dissimilar
            top_k_indices = torch.topk(similarities_i, min(self.top_k, num_metabolites - 1))[1]
            bottom_k_indices = torch.topk(similarities_i, min(self.top_k, num_metabolites - 1), largest=False)[1]
            
            anchor_embedding = learned_embeddings[i]
            
            # Positive pairs (similar metabolites should be close)
            for pos_idx in top_k_indices:
                pos_embedding = learned_embeddings[pos_idx]
                pos_similarity = F.cosine_similarity(
                    anchor_embedding.unsqueeze(0), pos_embedding.unsqueeze(0)
                )
                positive_loss = -torch.log(torch.sigmoid(pos_similarity / self.temperature) + 1e-8)
                total_loss += positive_loss
                num_pairs += 1
            
            # Negative pairs (dissimilar metabolites should be far)
            for neg_idx in bottom_k_indices:
                neg_embedding = learned_embeddings[neg_idx]
                neg_similarity = F.cosine_similarity(
                    anchor_embedding.unsqueeze(0), neg_embedding.unsqueeze(0)
                )
                negative_loss = -torch.log(torch.sigmoid(-neg_similarity / self.temperature) + 1e-8)
                total_loss += negative_loss
                num_pairs += 1
        
        if num_pairs > 0:
            return total_loss / num_pairs
        else:
            return torch.tensor(0.0, device=learned_embeddings.device, requires_grad=True)


class InfoNCELoss(nn.Module):
    """
    InfoNCE (Normalized Temperature-scaled Cross Entropy) loss for contrastive learning.
    
    Alternative implementation for chemical similarity regularization using contrastive learning principles.
    """
    
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features, similarity_matrix, positive_threshold=0.7):
        """
        Args:
            features: [num_samples, feature_dim] learned embeddings
            similarity_matrix: [num_samples, num_samples] pairwise similarities (e.g., from SMILES)
            positive_threshold: threshold above which samples are considered positive pairs
        """
        batch_size = features.size(0)
        device = features.device
        
        if batch_size < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Compute pairwise similarities in embedding space
        features_norm = F.normalize(features, p=2, dim=1)
        similarities = torch.matmul(features_norm, features_norm.T) / self.temperature
        
        # Create positive pair mask based on similarity threshold
        positive_mask = (similarity_matrix > positive_threshold) & (similarity_matrix < 0.999)  # Exclude self
        
        if not positive_mask.any():
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Compute InfoNCE loss
        total_loss = 0.0
        num_valid_anchors = 0
        
        for i in range(batch_size):
            positive_indices = positive_mask[i].nonzero(as_tuple=True)[0]
            
            if len(positive_indices) == 0:
                continue
            
            # Numerator: similarities with positive samples
            pos_similarities = similarities[i, positive_indices]
            
            # Denominator: similarities with all samples (excluding self)
            all_similarities = torch.cat([similarities[i, :i], similarities[i, i+1:]])
            
            # Compute log-sum-exp for numerical stability
            max_sim = torch.max(all_similarities)
            denominator = torch.log(torch.sum(torch.exp(all_similarities - max_sim))) + max_sim
            
            # InfoNCE loss for this anchor
            for pos_sim in pos_similarities:
                loss = -pos_sim + denominator
                total_loss += loss
            
            num_valid_anchors += len(positive_indices)
        
        if num_valid_anchors > 0:
            return total_loss / num_valid_anchors
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)