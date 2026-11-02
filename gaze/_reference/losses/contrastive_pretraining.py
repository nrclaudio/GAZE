"""
Contrastive pretraining loss for self-supervised learning of reaction-SMILES alignments.

This module implements InfoNCE-based contrastive learning to align reaction context
embeddings (from GNN) with ChemBERTa SMILES embeddings before supervised fine-tuning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math


class ContrastivePretrainingLoss(nn.Module):
    """
    Self-supervised contrastive loss for aligning reaction contexts with SMILES embeddings.
    
    Features:
    - Cross-modal alignment: reaction-context ↔ SMILES embeddings
    - Within-modal alignment: augmented reaction contexts (optional)
    - InfoNCE loss with cosine similarity and temperature scaling
    - Symmetric loss computation (bidirectional)
    - Batch-wise negative sampling
    """
    
    def __init__(
        self,
        cross_modal_temperature: float = 0.07,
        within_modal_temperature: float = 0.10,
        cross_modal_weight: float = 1.0,
        within_modal_weight: float = 0.5,
        symmetric_loss: bool = True,
        min_batch_size: int = 16,
        use_hard_negative_mining: bool = False,
        chemical_similarity_threshold: float = 0.3,
        smiles_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        
        self.cross_modal_temperature = cross_modal_temperature
        self.within_modal_temperature = within_modal_temperature
        self.cross_modal_weight = cross_modal_weight
        self.within_modal_weight = within_modal_weight
        self.symmetric_loss = symmetric_loss
        self.min_batch_size = min_batch_size
        self.use_hard_negative_mining = use_hard_negative_mining
        self.chemical_similarity_threshold = chemical_similarity_threshold
        self._smiles_projection: Optional[nn.Linear] = None
        
        # Validation of parameters
        if cross_modal_temperature <= 0 or within_modal_temperature <= 0:
            raise ValueError("Temperature parameters must be positive")
        if min_batch_size < 2:
            raise ValueError("Minimum batch size must be at least 2 for contrastive learning")
        
        # If dimensions are provided, initialize a persistent, learnable projection
        if smiles_dim is not None and hidden_dim is not None:
            self._smiles_projection = nn.Linear(smiles_dim, hidden_dim, bias=False)
            nn.init.kaiming_uniform_(self._smiles_projection.weight, a=math.sqrt(5))
            
    def forward(
        self,
        reaction_embeddings: torch.Tensor,
        smiles_embeddings: torch.Tensor,
        metabolite_indices: torch.Tensor,
        edge_index: torch.Tensor,
        reaction_indices: torch.Tensor,
        augmented_reaction_embeddings: Optional[torch.Tensor] = None,
        compute_within_modal: bool = False,
        return_metrics: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Compute contrastive pretraining loss.
        
        Args:
            reaction_embeddings: [batch_size, num_nodes, hidden_dim] GNN node embeddings
            smiles_embeddings: [batch_size, num_nodes, smiles_dim] SMILES embeddings
            metabolite_indices: [num_metabolites] indices of metabolite nodes (global)
            edge_index: [2, num_edges] graph connectivity (global indices)
            reaction_indices: [num_reactions] indices of reaction nodes (global)
            augmented_reaction_embeddings: [batch_size, num_nodes, hidden_dim] augmented embeddings
            compute_within_modal: whether to compute within-modality loss
            return_metrics: whether to return additional metrics
            
        Returns:
            Dict containing:
                - total_loss: combined contrastive loss
                - cross_modal_loss: cross-modal alignment loss
                - within_modal_loss: within-modal alignment loss (if computed)
                - metrics: additional evaluation metrics (if requested)
        """
        batch_size = reaction_embeddings.size(0)
        
        # Validate batch size for effective contrastive learning
        if batch_size < self.min_batch_size:
            raise ValueError(f"Batch size {batch_size} < minimum required {self.min_batch_size} for contrastive learning")
        
        # Extract metabolite embeddings for each (sample, metabolite) pair
        reaction_contexts, smiles_contexts = self._extract_metabolite_pairs(
            reaction_embeddings,
            smiles_embeddings,
            metabolite_indices,
            reaction_indices,
            edge_index
        )
        
        # Compute cross-modal contrastive loss
        cross_modal_loss = self._compute_cross_modal_loss(reaction_contexts, smiles_contexts)
        
        # Initialize within-modal loss
        within_modal_loss = torch.tensor(0.0, device=reaction_embeddings.device)
        
        # Compute within-modal loss if requested and augmented embeddings provided
        if compute_within_modal and augmented_reaction_embeddings is not None:
            augmented_contexts, _ = self._extract_metabolite_pairs(
                augmented_reaction_embeddings,
                smiles_embeddings,
                metabolite_indices,
                reaction_indices,
                edge_index
            )
            within_modal_loss = self._compute_within_modal_loss(reaction_contexts, augmented_contexts)
        
        # Combine losses
        total_loss = (self.cross_modal_weight * cross_modal_loss + 
                     self.within_modal_weight * within_modal_loss)
        
        results = {
            'total_loss': total_loss,
            'cross_modal_loss': cross_modal_loss,
            'within_modal_loss': within_modal_loss
        }
        
        # Add evaluation metrics if requested
        if return_metrics:
            metrics = self._compute_evaluation_metrics(reaction_contexts, smiles_contexts)
            results['metrics'] = metrics
            
        return results
    
    def _extract_metabolite_pairs(
        self,
        reaction_embeddings: torch.Tensor,
        smiles_embeddings: torch.Tensor,
        metabolite_indices: torch.Tensor,
        reaction_indices: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract (sample, metabolite) pairs for contrastive learning.
        
        For each sample and each metabolite, create pairs:
        - Reaction context: pooled reaction embeddings for that sample
        - SMILES embedding: metabolite's SMILES embedding
        
        Args:
            reaction_embeddings: [batch_size, num_nodes, hidden_dim]
            smiles_embeddings: [batch_size, num_nodes, smiles_dim] 
            metabolite_indices: [num_metabolites]
            
        Returns:
            reaction_contexts: [total_pairs, hidden_dim] L2-normalized
            smiles_contexts: [total_pairs, hidden_dim] L2-normalized (projected if needed)
        """
        device = reaction_embeddings.device
        batch_size, num_reactions, hidden_dim = reaction_embeddings.shape

        if isinstance(reaction_indices, torch.Tensor):
            reaction_idx_tensor = reaction_indices.to(device=device, dtype=torch.long)
        else:
            reaction_idx_tensor = torch.tensor(reaction_indices, device=device, dtype=torch.long)

        if isinstance(metabolite_indices, torch.Tensor):
            metabolite_idx_tensor = metabolite_indices.to(device=device, dtype=torch.long)
        else:
            metabolite_idx_tensor = torch.tensor(metabolite_indices, device=device, dtype=torch.long)

        num_metabolites = metabolite_idx_tensor.numel()

        # Ensure tensors on CPU for neighbour lookup
        reaction_indices_cpu = reaction_idx_tensor.detach().cpu().tolist()
        metabolite_indices_cpu = metabolite_idx_tensor.detach().cpu().tolist()
        edge_index_cpu = edge_index.detach().cpu().numpy()

        reaction_global_to_local = {int(node_idx): idx for idx, node_idx in enumerate(reaction_indices_cpu)}
        metabolite_set = {int(idx) for idx in metabolite_indices_cpu}

        # Build metabolite -> local reaction neighbours map
        neighbour_map: dict[int, list[int]] = {int(m): [] for m in metabolite_indices_cpu}
        for src, dst in edge_index_cpu.T:
            src_i = int(src)
            dst_i = int(dst)

            if src_i in metabolite_set and dst_i in reaction_global_to_local:
                neighbour_map[src_i].append(reaction_global_to_local[dst_i])
            if dst_i in metabolite_set and src_i in reaction_global_to_local:
                neighbour_map[dst_i].append(reaction_global_to_local[src_i])

        # Remove duplicates while preserving insertion order
        for key, values in neighbour_map.items():
            seen = set()
            deduped = []
            for val in values:
                if val not in seen:
                    seen.add(val)
                    deduped.append(val)
            neighbour_map[key] = deduped

        batch_metabolite_smiles = smiles_embeddings.index_select(1, metabolite_idx_tensor)

        reaction_contexts = []
        smiles_contexts = []

        reaction_means = reaction_embeddings.mean(dim=1)  # fallback when no neighbours

        for batch_idx in range(batch_size):
            reaction_sample = reaction_embeddings[batch_idx]
            fallback = reaction_means[batch_idx]

            for local_met_idx, met_global in enumerate(metabolite_indices_cpu):
                neighbour_ids = neighbour_map.get(int(met_global), [])

                if neighbour_ids:
                    neighbour_tensor = torch.tensor(neighbour_ids, device=device, dtype=torch.long)
                    context = reaction_sample.index_select(0, neighbour_tensor).mean(dim=0)
                else:
                    context = fallback

                reaction_contexts.append(context)
                smiles_contexts.append(batch_metabolite_smiles[batch_idx, local_met_idx])

        reaction_contexts = torch.stack(reaction_contexts)
        smiles_contexts = torch.stack(smiles_contexts)

        # Project SMILES to hidden_dim if needed
        if smiles_contexts.size(-1) != hidden_dim:
            smiles_contexts = self._project_smiles_contexts(smiles_contexts, hidden_dim)
        
        # L2 normalize both embeddings
        reaction_contexts = F.normalize(reaction_contexts, p=2, dim=1)
        smiles_contexts = F.normalize(smiles_contexts, p=2, dim=1)
        
        return reaction_contexts, smiles_contexts
    
    def _compute_cross_modal_loss(
        self, 
        reaction_contexts: torch.Tensor, 
        smiles_contexts: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cross-modal InfoNCE loss between reaction contexts and SMILES.
        
        Args:
            reaction_contexts: [num_pairs, hidden_dim] L2-normalized
            smiles_contexts: [num_pairs, hidden_dim] L2-normalized
            
        Returns:
            cross_modal_loss: scalar loss value
        """
        num_pairs = reaction_contexts.size(0)
        
        # Compute similarity matrix: [num_pairs, num_pairs]
        similarities = torch.matmul(reaction_contexts, smiles_contexts.t()) / self.cross_modal_temperature
        
        # Create labels (diagonal elements are positives)
        labels = torch.arange(num_pairs, device=reaction_contexts.device)
        
        if self.symmetric_loss:
            # Bidirectional InfoNCE: reaction→SMILES and SMILES→reaction
            loss_r2s = F.cross_entropy(similarities, labels)  # reaction to SMILES
            loss_s2r = F.cross_entropy(similarities.t(), labels)  # SMILES to reaction
            return (loss_r2s + loss_s2r) / 2.0
        else:
            # Unidirectional: reaction→SMILES only
            return F.cross_entropy(similarities, labels)
    
    def _compute_within_modal_loss(
        self, 
        original_contexts: torch.Tensor, 
        augmented_contexts: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute within-modal InfoNCE loss between original and augmented reaction contexts.
        
        Args:
            original_contexts: [num_pairs, hidden_dim] L2-normalized
            augmented_contexts: [num_pairs, hidden_dim] L2-normalized
            
        Returns:
            within_modal_loss: scalar loss value
        """
        num_pairs = original_contexts.size(0)
        
        # Compute similarity matrix
        similarities = torch.matmul(original_contexts, augmented_contexts.t()) / self.within_modal_temperature
        
        # Create labels (diagonal elements are positives)
        labels = torch.arange(num_pairs, device=original_contexts.device)
        
        if self.symmetric_loss:
            # Bidirectional InfoNCE
            loss_o2a = F.cross_entropy(similarities, labels)  # original to augmented
            loss_a2o = F.cross_entropy(similarities.t(), labels)  # augmented to original
            return (loss_o2a + loss_a2o) / 2.0
        else:
            # Unidirectional: original→augmented only
            return F.cross_entropy(similarities, labels)
    
    def _compute_evaluation_metrics(
        self, 
        reaction_contexts: torch.Tensor, 
        smiles_contexts: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute evaluation metrics for monitoring training progress.
        
        Args:
            reaction_contexts: [num_pairs, hidden_dim] L2-normalized
            smiles_contexts: [num_pairs, hidden_dim] L2-normalized
            
        Returns:
            Dict of evaluation metrics
        """
        with torch.no_grad():
            # Compute similarity matrix
            similarities = torch.matmul(reaction_contexts, smiles_contexts.t())
            
            # Retrieval@1: for each reaction context, is its corresponding SMILES the most similar?
            _, top1_indices = similarities.topk(k=1, dim=1)
            correct_top1 = (top1_indices.squeeze() == torch.arange(len(reaction_contexts), device=similarities.device))
            retrieval_at_1 = correct_top1.float().mean()
            
            # Retrieval@5
            _, top5_indices = similarities.topk(k=min(5, similarities.size(1)), dim=1)
            correct_top5 = torch.any(
                top5_indices == torch.arange(len(reaction_contexts), device=similarities.device).unsqueeze(1),
                dim=1
            )
            retrieval_at_5 = correct_top5.float().mean()
            
            # Cosine similarity statistics
            positive_similarities = torch.diag(similarities)  # Positive pairs
            negative_similarities = similarities[~torch.eye(len(similarities), dtype=bool, device=similarities.device)]  # Negative pairs
            
            positive_sim_mean = positive_similarities.mean()
            negative_sim_mean = negative_similarities.mean()
            cosine_gap = positive_sim_mean - negative_sim_mean  # Should be positive and large
            
        return {
            'retrieval_at_1': retrieval_at_1,
            'retrieval_at_5': retrieval_at_5,
            'positive_sim_mean': positive_sim_mean,
            'negative_sim_mean': negative_sim_mean,
            'cosine_gap': cosine_gap
        }

    def _project_smiles_contexts(
        self,
        smiles_contexts: torch.Tensor,
        hidden_dim: int
    ) -> torch.Tensor:
        """Project SMILES embeddings to the reaction hidden dimension with a stable mapping."""
        in_features = smiles_contexts.size(-1)

        # Lazily initialize projection to avoid recreating it every forward pass.
        if self._smiles_projection is None or \
           self._smiles_projection.in_features != in_features or \
           self._smiles_projection.out_features != hidden_dim:
            self._smiles_projection = nn.Linear(in_features, hidden_dim, bias=False)
            nn.init.kaiming_uniform_(self._smiles_projection.weight, a=math.sqrt(5))

        projection = self._smiles_projection.to(smiles_contexts.device)
        return projection(smiles_contexts)


class ReactionAugmentor(nn.Module):
    """
    Data augmentation for reaction contexts to enable within-modal contrastive learning.
    
    Augmentation strategies:
    - Gene expression masking: randomly mask ~10-20% of reaction scalars
    - Edge dropping: randomly drop ~5-10% of local edges (biologically-aware)
    """
    
    def __init__(
        self,
        gene_mask_ratio: float = 0.15,
        edge_drop_ratio: float = 0.05,
        preserve_essential_genes: bool = True
    ):
        super().__init__()
        self.gene_mask_ratio = gene_mask_ratio
        self.edge_drop_ratio = edge_drop_ratio
        self.preserve_essential_genes = preserve_essential_genes
        
    def forward(
        self,
        reaction_embeddings: torch.Tensor,
        x_gene: torch.Tensor,
        edge_index: torch.Tensor,
        reaction_indices: torch.Tensor,
        essential_reaction_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply augmentations to create alternative views of reaction contexts.
        
        Args:
            reaction_embeddings: [batch_size, num_nodes, hidden_dim] 
            x_gene: [batch_size, num_nodes, gene_dim] gene expression features
            edge_index: [2, num_edges] graph edges
            reaction_indices: [num_reactions] reaction node indices
            essential_reaction_mask: [num_reactions] mask for essential reactions to preserve
            
        Returns:
            augmented_embeddings: [batch_size, num_nodes, hidden_dim] augmented reaction embeddings
            augmented_edge_index: [2, num_edges_reduced] augmented edge index
        """
        batch_size, num_nodes, hidden_dim = reaction_embeddings.shape
        
        # Clone inputs to avoid in-place modifications
        aug_x_gene = x_gene.clone()
        aug_edge_index = edge_index.clone()
        
        # 1. Gene expression masking for reaction nodes
        if self.gene_mask_ratio > 0:
            aug_x_gene = self._mask_gene_expression(
                aug_x_gene, reaction_indices, essential_reaction_mask
            )
        
        # 2. Edge dropping (structural augmentation)
        if self.edge_drop_ratio > 0:
            aug_edge_index = self._drop_edges(aug_edge_index, reaction_indices)
        
        # Note: In a full implementation, you would re-run the GNN forward pass
        # with augmented inputs to get augmented_embeddings. For now, we return
        # the original embeddings as a placeholder.
        augmented_embeddings = reaction_embeddings
        
        return augmented_embeddings, aug_edge_index
    
    def _mask_gene_expression(
        self,
        x_gene: torch.Tensor,
        reaction_indices: torch.Tensor,
        essential_reaction_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Randomly mask gene expression values for reaction nodes.
        
        Args:
            x_gene: [batch_size, num_nodes, gene_dim]
            reaction_indices: [num_reactions]
            essential_reaction_mask: [num_reactions] optional mask to preserve essential reactions
            
        Returns:
            Augmented gene expression tensor
        """
        batch_size, num_nodes, gene_dim = x_gene.shape
        
        for batch_idx in range(batch_size):
            for reaction_idx in reaction_indices:
                # Skip essential reactions if specified
                if (essential_reaction_mask is not None and 
                    essential_reaction_mask[reaction_idx].item()):
                    continue
                
                # Randomly mask some gene expression values
                mask = torch.rand(gene_dim, device=x_gene.device) < self.gene_mask_ratio
                x_gene[batch_idx, reaction_idx, mask] = 0.0
                
        return x_gene
    
    def _drop_edges(
        self,
        edge_index: torch.Tensor,
        reaction_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Randomly drop edges to create structural variation.
        
        Args:
            edge_index: [2, num_edges]
            reaction_indices: [num_reactions] 
            
        Returns:
            Augmented edge index with some edges removed
        """
        num_edges = edge_index.size(1)
        
        # Create a mask for edges to keep
        keep_mask = torch.rand(num_edges, device=edge_index.device) > self.edge_drop_ratio
        
        # TODO: Add biologically-aware edge dropping logic here
        # - Preserve essential metabolic connections
        # - Weight dropping by edge importance/stoichiometry
        
        return edge_index[:, keep_mask]
