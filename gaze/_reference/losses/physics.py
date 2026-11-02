"""
Physics-informed loss functions for metabolic modeling.

This module contains loss functions that incorporate physical/biological constraints:
- PhysicsLoss: Basic physics constraints
- EnhancedPhysicsLoss: Advanced physics with gene-flux correlation
- Helper functions for gene rule parsing and correlation computation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from .flux_alignment import FluxAlignmentLoss
from ..utils.batch_utils import ensure_batch_dim
from ..utils.constants import FLUX_VARIANCE_EPS, CORRELATION_EPS, EXPRESSION_THRESHOLD
from ..utils.gene_parsing import parse_gene_rule_simple as parse_gene_rule


class PhysicsLoss(nn.Module):
    """
    Physics-informed loss that combines metabolite prediction loss with
    mass-balance constraint and flux non-negativity penalty.

    NOTE: This loss function can be used with any model that returns:
    - c_hat: predicted concentrations
    - delta: mass-balance residuals
    - flux_vec: predicted fluxes

    Simply replace MaskedMetaboliteLoss with PhysicsLoss in any training script.
    Recommended hyperparameters:
    - λ_mse=1.0 (standard MSE weight)
    - λ_bal=1.0 (mass-balance weight, tune in range [0.1, 3.0])

    Note: Non-negativity is enforced architecturally via ELU + ReLU.
    """
    def __init__(self, λ_mse=1.0, λ_bal=1.0):
        super().__init__()
        self.λ_mse = λ_mse
        self.λ_bal = λ_bal
        
    def forward(self, c_hat, c_true, mask, delta, flux_vec, epoch, data=None,
                **kwargs):
        """
        Args:
            c_hat: Predicted metabolite concentrations [batch_size, num_nodes]
            c_true: True metabolite concentrations [batch_size, num_nodes]
            mask: Boolean mask for measured metabolites [batch_size, num_nodes]
            delta: Mass-balance residuals [batch_size, num_metabolites]
            flux_vec: Predicted reaction fluxes [batch_size, num_reactions]
            epoch: Current training epoch
            data: PyG data object (optional, for spatial information)
        """
        # Ensure batched format
        if c_hat.dim() == 1:
            c_hat = c_hat.unsqueeze(0)
            c_true = c_true.unsqueeze(0)
            mask = mask.unsqueeze(0)
            delta = delta.unsqueeze(0)
            flux_vec = flux_vec.unsqueeze(0)
        
        batch_size = c_hat.size(0)
        
        # Standard MSE loss on masked predictions
        if mask.any():
            # Create a combined mask ensuring we only compute loss on valid, masked entries
            # c_hat and c_true might have NaNs (though ideally handled elsewhere), so we can be safe
            valid_mask = mask # & ~torch.isnan(c_hat) & ~torch.isnan(c_true)
            
            if valid_mask.any():
                mse = F.mse_loss(c_hat[valid_mask], c_true[valid_mask])
            else:
                mse = torch.tensor(0.0, device=c_hat.device, requires_grad=True)
        else:
            mse = torch.tensor(0.0, device=c_hat.device, requires_grad=True)
        
        # Mass-balance constraint: minimize squared residuals
        balance = torch.mean(delta ** 2)

        # Combined loss
        total_loss = self.λ_mse * mse + self.λ_bal * balance

        return total_loss


# ===== Removed divergence and correlation loss utilities (corr_loss, pairwise_divergence_loss,
#       pairwise_divergence_flux_loss, build_reaction_gene_scores, evaluate_gene_rule_logic,
#       evaluate_and_expression, build_metabolite_gene_scores) as part of codebase cleanup. =====



# parse_gene_rule is imported from gaze._reference.utils.gene_parsing


class EnhancedPhysicsLoss(nn.Module):
    """
    Enhanced physics-informed loss with flux-based message passing and listwise ranking.
    
    Features:
    1. Mass balance constraint: Nv = 0 (can be enforced by projection or soft penalty)
    2. Non-negativity constraint: v ≥ 0 (flux directions, enforced architecturally) 
    3. Gene-flux correlation: Traditional correlation loss
    4. Listwise flux ranking: Neural network-free alignment using KL divergence
    5. Optional pairwise hinge loss for top-vs-bottom reaction pairs
    
    When mass balance projection is used in the model, λ_bal should be set to 0 or very small
    since the constraint is already exactly satisfied.
    
    Usage examples:
    
    # Standard soft constraints (no projection)
    loss = EnhancedPhysicsLoss(λ_mse=1.0, λ_bal=1.0, λ_corr=0.5, λ_kl=1.0)
    
    # With mass balance projection (reduce λ_bal)
    loss = EnhancedPhysicsLoss(λ_mse=1.0, λ_bal=0.01, λ_corr=0.5, λ_kl=1.0)
    """
    def __init__(self, λ_mse=1.0, λ_bal=1.0, λ_corr=0.5, λ_var=1.0, λ_kl=1.0,
                 use_listwise_ranking=True, listwise_config=None):
        super().__init__()
        self.λ_mse = λ_mse
        self.λ_bal = λ_bal
        self.λ_corr = λ_corr
        self.λ_var = λ_var
        self.λ_kl = λ_kl
        self.use_listwise_ranking = use_listwise_ranking
        
        # Cache for gene rule parsing (PERFORMANCE OPTIMIZATION)
        self._gene_rule_cache = {}
        self._reaction_gene_mapping = None
        
        # Initialize flux alignment loss (listwise ranking)
        if self.use_listwise_ranking:
            if listwise_config is None:
                listwise_config = {
                    'enzyme_confidence': 3.0,   # High confidence in biological enzyme scores
                    'flux_flexibility': 0.8,    # Flexible model predictions can learn
                    'min_reactions': 3,
                    'use_absolute_flux': True,
                    'flux_sign_weighting': False  # Simplified, no sign weighting
                }
            
            self.flux_alignment_loss = FluxAlignmentLoss(
                use_listwise=True,
                use_pairwise=False,  # Can be enabled for additional constraint
                listwise_weight=1.0,
                **listwise_config
            )
        
        # For external logging
        self.last_loss_components = {}

    def forward(self, c_hat, c_true, mask, delta, flux_vec, epoch, data=None,
                **kwargs):
        """
        Args:
            c_hat: Predicted metabolite concentrations [num_nodes] or [batch_size, num_nodes]
            c_true: True metabolite concentrations [num_nodes] or [batch_size, num_nodes]
            mask: Boolean mask for measured metabolites [num_nodes] or [batch_size, num_nodes]
            delta: Mass-balance residuals [num_metabolites] or [batch_size, num_metabolites]
            flux_vec: Predicted reaction fluxes [num_reactions] or [batch_size, num_reactions]
            epoch: Current training epoch
            data: PyG data object with gene rules and expression data
        """
        forward_start_time = time.time()
        
        device = c_hat.device
        
        # Ensure batched format (for compatibility during transition)
        if c_hat.dim() == 1:
            c_hat = c_hat.unsqueeze(0)
            c_true = c_true.unsqueeze(0)
            mask = mask.unsqueeze(0)
            delta = delta.unsqueeze(0)
            flux_vec = flux_vec.unsqueeze(0)
        
        batch_size = c_hat.size(0)
        
        # 1. Standard MSE loss on masked predictions (if labels available)
        mse_start = time.time()
        if mask.any():
            # Compute MSE only on masked (measured) metabolites
            # Create a combined mask ensuring we only compute loss on valid, masked entries
            valid_mask = mask # & ~torch.isnan(c_hat) & ~torch.isnan(c_true)
            
            if valid_mask.any():
                mse_loss = F.mse_loss(c_hat[valid_mask], c_true[valid_mask])
            else:
                mse_loss = torch.tensor(0.0, device=device, requires_grad=True)
        else:
            # No metabolite labels available (e.g., spatial transcriptomics)
            mse_loss = torch.tensor(0.0, device=device, requires_grad=True)
        mse_time = time.time() - mse_start
        
        # 2. Mass-balance penalty 
        balance_loss = torch.mean(delta ** 2)
        
        # 3. Gene-flux correlation term (biologically-inspired)
        correlation_loss = self._compute_gene_flux_correlation_loss(flux_vec, data, device)
        
        # 4. Variance constraint - prevent all fluxes from collapsing to near-zero
        flux_std = torch.std(flux_vec, dim=-1)
        variance_loss = torch.mean(1.0 / (flux_std + FLUX_VARIANCE_EPS))
        
        # 5. Flux alignment loss (listwise ranking)
        flux_alignment_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # Debug: Check why flux alignment loss might be 0
        if not self.use_listwise_ranking or self.λ_kl == 0:
            pass  # FluxAlignment disabled by configuration or zero weight
        elif data is None:
            print(f"WARNING: FluxAlignment loss is 0 - data is None")
        elif not hasattr(data, 'x_gene'):
            print(f"WARNING: FluxAlignment loss is 0 - data.x_gene missing")
        elif not hasattr(data, 'reaction_indices'):
            print(f"WARNING: FluxAlignment loss is 0 - data.reaction_indices missing")
        else:
            # Debug: Print data shapes and content (only every 10 epochs)
            if epoch % 10 == 0:
                print(f"DEBUG: FluxAlignment conditions met:")
                print(f"  - x_gene shape: {data.x_gene.shape}")
                print(f"  - reaction_indices length: {len(data.reaction_indices) if hasattr(data.reaction_indices, '__len__') else 'scalar'}")
                print(f"  - flux_vec shape: {flux_vec.shape}")
                print(f"  - x_gene has NaN: {torch.isnan(data.x_gene).sum().item()} / {data.x_gene.numel()}")
                print(f"  - x_gene non-zero: {(data.x_gene != 0).sum().item()} / {data.x_gene.numel()}")
            
            # All conditions met - compute the loss
            flux_alignment_loss = self.flux_alignment_loss(
                fluxes=flux_vec,
                gene_expressions=data.x_gene,
                reaction_indices=data.reaction_indices,
                data=data
            )
            
            # Debug: Report the computed loss value (only every 10 epochs)
            if epoch % 10 == 0:
                print(f"DEBUG: FluxAlignment loss computed: {flux_alignment_loss.item():.6f}")
            if flux_alignment_loss.item() == 0.0:
                print(f"WARNING: FluxAlignment loss is exactly 0 - check ListwiseFluxRankingLoss details")
        
        # Combined loss
        # Store individual loss components for external monitoring
        self.last_loss_components = {
            'MSE': mse_loss.detach().item(),
            'Balance': balance_loss.detach().item(),
            'Corr': correlation_loss.detach().item(),
            'Variance': variance_loss.detach().item(),
            'FluxAlignment': flux_alignment_loss.detach().item(),
        }

        total_loss = (self.λ_mse * mse_loss +
                     self.λ_bal * balance_loss +
                     self.λ_corr * correlation_loss +
                     self.λ_var * variance_loss +
                     self.λ_kl * flux_alignment_loss)
        
        forward_total_time = time.time() - forward_start_time
        
        return total_loss

    def _compute_gene_flux_correlation_loss(self, flux_vec, data, device):
        """
        TENSOR-OPTIMIZED: Compute correlation using pre-computed tensor operations.
        
        Uses pre-computed tensors to eliminate all dictionary lookups and string parsing:
        - reaction_gene_indices: [num_reactions, max_genes] gene feature indices
        - reaction_gene_operations: [num_reactions] operation codes (0=mean, 1=min, 2=max)
        - reaction_gene_masks: [num_reactions, max_genes] validity masks
        
        Args:
            flux_vec: Predicted reaction fluxes [batch_size, num_reactions]
            data: PyG data object with pre-computed gene tensors
            device: torch device
            
        Returns:
            correlation_loss: Scalar loss encouraging flux-gene correlation
        """
        corr_start_time = time.time()
        
        # Check for pre-computed tensor operations
        has_tensor_ops = (hasattr(data, 'reaction_gene_indices') and 
                         hasattr(data, 'reaction_gene_operations') and
                         hasattr(data, 'reaction_gene_masks'))
        
        if has_tensor_ops:
            return self._compute_correlation_tensor_optimized(flux_vec, data, device, corr_start_time)
        else:
            missing_attrs = []
            if not hasattr(data, 'reaction_gene_indices'):
                missing_attrs.append('reaction_gene_indices')
            if not hasattr(data, 'reaction_gene_operations'):
                missing_attrs.append('reaction_gene_operations')
            if not hasattr(data, 'reaction_gene_masks'):
                missing_attrs.append('reaction_gene_masks')
            
            return self._compute_correlation_original(flux_vec, data, device, corr_start_time)
    
    def _compute_correlation_tensor_optimized(self, flux_vec, data, device, start_time):
        """
        ULTRA-FAST: Pure tensor operations for gene-flux correlation.
        """
        batch_size = flux_vec.size(0)
        
        # Get pre-computed tensors (detach to avoid graph conflicts)
        gene_indices = data.reaction_gene_indices.detach().to(device)  # [num_reactions, max_genes]
        operations = data.reaction_gene_operations.detach().to(device)  # [num_reactions]
        masks = data.reaction_gene_masks.detach().to(device)  # [num_reactions, max_genes]
        reaction_nodes = data.reaction_node_indices.detach().to(device)  # [num_reactions]
        
        # Get gene expression data  
        if not hasattr(data, 'x_gene'):
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        x_gene = data.x_gene
        if len(x_gene.shape) > 1:
            if x_gene.size(0) == batch_size:
                # Batched: [batch_size, num_nodes, gene_dim]
                batched_gene_expr = x_gene
            else:
                # Single: expand to [batch_size, num_nodes, gene_dim]
                batched_gene_expr = x_gene.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            batched_gene_expr = x_gene.unsqueeze(0).expand(batch_size, -1, -1)
        
        # OLD per-sample path removed – replaced by batched implementation
        # Compute gene scores for all batches in a single fused GPU pass
        gene_scores = self._compute_gene_scores_batched(batched_gene_expr, reaction_nodes,
                                                        gene_indices, operations, masks, device)
        
        # Build validity mask & zero-out invalid positions
        valid_mask = ~torch.isnan(gene_scores)                               # [B,R]

        # If *all* positions are invalid just return 0
        if valid_mask.sum() < 2:
            total_time = time.time() - start_time
            pass
            return torch.tensor(0.0, device=device, requires_grad=True)

        flux_valid = flux_vec.clone()
        gene_valid = gene_scores.clone()
        flux_valid[~valid_mask] = 0.0
        gene_valid[~valid_mask] = 0.0

        # Compute per-batch means over valid entries only
        counts = torch.clamp(valid_mask.sum(dim=1, keepdim=True), min=1)
        flux_mean = flux_valid.sum(dim=1, keepdim=True) / counts
        gene_mean = gene_valid.sum(dim=1, keepdim=True) / counts

        # Center the data (zero contributions from invalid stay zero)
        flux_centered = flux_valid - flux_mean
        gene_centered = gene_valid - gene_mean

        # Mask again after centering to be safe
        flux_centered[~valid_mask] = 0.0
        gene_centered[~valid_mask] = 0.0

        # Compute covariance & variances
        numerator   = (flux_centered * gene_centered).sum(dim=1)
        flux_var    = (flux_centered ** 2).sum(dim=1)
        gene_var    = (gene_centered ** 2).sum(dim=1)
        denominator = torch.sqrt(flux_var * gene_var + CORRELATION_EPS)

        corr = numerator / denominator
        corr_loss = 1.0 - torch.abs(corr)

        total_time = time.time() - start_time

        # Return mean loss across batch
        return corr_loss.mean()
    
    def _compute_all_gene_scores_vectorized(self, gene_expr, reaction_nodes, gene_indices, operations, masks, device):
        """
        ULTRA-VECTORIZED: Compute gene scores for all reactions using fully parallel operations.
        
        Args:
            gene_expr: [num_nodes, gene_dim] gene expression for current sample
            reaction_nodes: [num_reactions] reaction node indices
            gene_indices: [num_reactions, max_genes] gene feature indices (-1 for padding)
            operations: [num_reactions] operation codes (0=mean, 1=min, 2=max)
            masks: [num_reactions, max_genes] validity masks
            
        Returns:
            gene_scores: [num_reactions] tensor with NaN for invalid reactions
        """
        num_reactions = reaction_nodes.size(0)
        max_genes = gene_indices.size(1)
        
        # Filter out reactions with invalid node indices
        valid_reaction_mask = reaction_nodes < gene_expr.size(0)
        valid_reactions = torch.nonzero(valid_reaction_mask).squeeze(-1)
        
        if valid_reactions.numel() == 0:
            return torch.full((num_reactions,), float('nan'), device=device)
        
        # Get all reaction features at once [num_valid_reactions, gene_dim]
        valid_reaction_nodes = reaction_nodes[valid_reactions]
        reaction_features = gene_expr[valid_reaction_nodes]  # [num_valid_reactions, gene_dim]
        
        # Get corresponding gene indices and masks
        valid_gene_indices = gene_indices[valid_reactions]  # [num_valid_reactions, max_genes]
        valid_masks = masks[valid_reactions]  # [num_valid_reactions, max_genes]
        valid_operations = operations[valid_reactions]  # [num_valid_reactions]
        
        # Create a large tensor to hold all gene expressions
        # [num_valid_reactions, max_genes]
        expanded_features = reaction_features.unsqueeze(1).expand(-1, max_genes, -1)  # [num_valid_reactions, max_genes, gene_dim]
        
        # Create valid index mask
        gene_dim = reaction_features.size(1)
        valid_idx_mask = (valid_gene_indices >= 0) & (valid_gene_indices < gene_dim) & valid_masks
        
        # Extract gene expressions using advanced indexing
        # This is the tricky part - we need to gather from each reaction's features using its gene indices
        batch_idx = torch.arange(valid_reactions.size(0), device=device).unsqueeze(1).expand(-1, max_genes)
        
        # Only gather where we have valid indices
        gene_expressions = torch.zeros((valid_reactions.size(0), max_genes), device=device)
        
        # Use advanced indexing to gather gene expressions
        valid_positions = valid_idx_mask.nonzero()  # [num_valid_positions, 2]
        
        if valid_positions.numel() > 0:
            reaction_idx = valid_positions[:, 0]  # Which reaction
            gene_slot_idx = valid_positions[:, 1]  # Which gene slot
            gene_feature_idx = valid_gene_indices[reaction_idx, gene_slot_idx]  # Which gene feature
            
            # Extract the actual expressions
            extracted_expressions = reaction_features[reaction_idx, gene_feature_idx]
            gene_expressions[reaction_idx, gene_slot_idx] = extracted_expressions
        
        # Apply threshold mask
        threshold_mask = (gene_expressions > EXPRESSION_THRESHOLD) & valid_idx_mask
        
        # Set invalid/below-threshold values to special values for operations
        gene_expressions_masked = gene_expressions.clone()
        
        # For min operations, set invalid to +inf; for max/mean set to 0
        operations_expanded = valid_operations.unsqueeze(1).expand(-1, max_genes)
        min_mask = (operations_expanded == 1) & ~threshold_mask
        other_mask = (operations_expanded != 1) & ~threshold_mask
        
        gene_expressions_masked[min_mask] = float('inf')
        gene_expressions_masked[other_mask] = 0.0
        
        # Count valid genes per reaction
        valid_gene_counts = threshold_mask.sum(dim=1)  # [num_valid_reactions]
        
        # Compute operations vectorized
        min_scores = torch.min(gene_expressions_masked, dim=1)[0]  # [num_valid_reactions]
        max_scores = torch.max(gene_expressions_masked, dim=1)[0]  # [num_valid_reactions] 
        sum_scores = torch.sum(gene_expressions_masked, dim=1)  # [num_valid_reactions]
        mean_scores = sum_scores / torch.clamp(valid_gene_counts.float(), min=1.0)
        
        # Select the appropriate score based on operation
        final_scores = torch.zeros(valid_reactions.size(0), device=device)
        
        mean_mask = valid_operations == 0
        min_mask = valid_operations == 1  
        max_mask = valid_operations == 2
        
        final_scores[mean_mask] = mean_scores[mean_mask]
        final_scores[min_mask] = min_scores[min_mask]
        final_scores[max_mask] = max_scores[max_mask]
        
        # Handle invalid cases (no valid genes or inf values)
        invalid_mask = (valid_gene_counts == 0) | (final_scores == float('inf'))
        final_scores[invalid_mask] = float('nan')
        
        # Create full results tensor
        gene_scores = torch.full((num_reactions,), float('nan'), device=device)
        gene_scores[valid_reactions] = final_scores
        
        return gene_scores

    # ------------------------------------------------------------------
    # NEW: fully-batched ultra-fast gene score computation
    # ------------------------------------------------------------------
    def _compute_gene_scores_batched(self, batched_gene_expr, reaction_nodes,
                                     gene_indices, operations, masks, device):
        """
        Ultra-fast batched version that computes gene scores for *all* batches in
        a single fused GPU pass – no Python loops.

        Args:
            batched_gene_expr: Tensor [B, num_nodes, gene_dim] gene expressions
            reaction_nodes:    Tensor [num_reactions] reaction node indices
            gene_indices:      Tensor [num_reactions, max_genes] gene feature idx
            operations:        Tensor [num_reactions] 0=mean,1=min,2=max
            masks:             Bool tensor [num_reactions, max_genes] valid mask
            device:            torch device
        Returns:
            gene_scores: Tensor [B, num_reactions] (NaN where no valid genes)
        """
        B, N, D = batched_gene_expr.shape
        R, G = gene_indices.shape

        # 1) Gather reaction node features: [B, R, D]
        reaction_features = batched_gene_expr[:, reaction_nodes, :]

        # 2) Expand indices & masks along batch dim
        gene_idx_exp = gene_indices.unsqueeze(0).expand(B, -1, -1)           # [B,R,G]
        masks_exp    = masks.unsqueeze(0).expand(B, -1, -1)                  # [B,R,G]

        # 3) Safe gather: clamp negative padding indices to 0
        safe_idx = torch.clamp(gene_idx_exp, min=0)
        gathered = torch.take_along_dim(
            reaction_features.unsqueeze(2).expand(-1, -1, G, -1),
            safe_idx.unsqueeze(-1),
            dim=3,
        ).squeeze(-1)                                                        # [B,R,G]

        # 4) Threshold mask
        thresh_mask = masks_exp & (gathered > EXPRESSION_THRESHOLD)

        # 5) Prepare sentinel-filled tensors
        gathered_min  = torch.where(thresh_mask, gathered, torch.full_like(gathered, float('inf')))
        gathered_max  = torch.where(thresh_mask, gathered, torch.full_like(gathered, float('-inf')))
        gathered_mean = torch.where(thresh_mask, gathered, torch.zeros_like(gathered))

        # 6) Valid gene counts per reaction
        valid_counts = thresh_mask.sum(dim=2)                                # [B,R]

        # 7) Compute aggregation ops
        mean_scores = gathered_mean.sum(dim=2) / torch.clamp(valid_counts, min=1)
        min_scores  = gathered_min.min(dim=2).values
        max_scores  = gathered_max.max(dim=2).values

        # 8) Assemble final scores
        final_scores = torch.empty_like(mean_scores)
        op_mean = operations == 0
        op_min  = operations == 1
        op_max  = operations == 2

        final_scores[:, op_mean] = mean_scores[:, op_mean]
        final_scores[:, op_min]  = min_scores[:,  op_min]
        final_scores[:, op_max]  = max_scores[:,  op_max]

        # 9) Mark reactions with zero valid genes as NaN
        final_scores = final_scores.masked_fill(valid_counts == 0, float('nan'))

        return final_scores
    
    def _compute_correlation_original(self, flux_vec, data, device, start_time):
        """Fallback to original correlation computation method."""
        has_gene_rules = hasattr(data, 'gene_rules')
        has_x_gene = hasattr(data, 'x_gene')
        
        if not has_gene_rules or not has_x_gene:
            pass
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Use the optimized cached approach as fallback
        gene_rules_list = data.gene_rules
        if len(gene_rules_list) == 1 and hasattr(gene_rules_list[0], '__len__') and not isinstance(gene_rules_list[0], str):
            gene_rules_list = gene_rules_list[0]
        
        if self._reaction_gene_mapping is None:
            self._build_reaction_gene_mapping(gene_rules_list, data)
        
        batch_size = flux_vec.size(0)
        reaction_indices = data.reaction_indices
        if isinstance(reaction_indices, list):
            reaction_indices = torch.tensor(reaction_indices, device=device)
        
        correlation_losses = []
        
        for b in range(batch_size):
            batch_flux = flux_vec[b]
            
            if hasattr(data, 'x_gene') and len(data.x_gene.shape) > 1:
                if data.x_gene.size(0) == batch_size:
                    gene_expr = data.x_gene[b]
                else:
                    gene_expr = data.x_gene
            else:
                gene_expr = data.x_gene
            
            valid_gene_scores = []
            valid_fluxes = []
            
            for i, r_idx in enumerate(reaction_indices):
                r_idx_int = int(r_idx.item()) if hasattr(r_idx, 'item') else int(r_idx)
                
                if r_idx_int in self._reaction_gene_mapping:
                    gene_score = self._compute_cached_gene_score_fast(r_idx_int, gene_expr, data)
                    
                    # More robust filtering
                    if (gene_score is not None and 
                        torch.is_tensor(gene_score) and 
                        not torch.isnan(gene_score).any() and 
                        gene_score.numel() > 0):
                        valid_gene_scores.append(gene_score)
                        valid_fluxes.append(batch_flux[i])
            
            if len(valid_gene_scores) >= 2:
                try:
                    gene_scores_tensor = torch.stack(valid_gene_scores)
                    flux_scores_tensor = torch.stack(valid_fluxes)
                    correlation = self._compute_pearson_correlation(flux_scores_tensor, gene_scores_tensor)
                    correlation_loss = 1.0 - torch.abs(correlation)
                    correlation_losses.append(correlation_loss)
                except Exception as e:
                    correlation_losses.append(torch.tensor(0.0, device=device, requires_grad=True))
            else:
                correlation_losses.append(torch.tensor(0.0, device=device, requires_grad=True))
        
        if correlation_losses:
            final_loss = torch.stack(correlation_losses).mean()
            total_time = time.time() - start_time
            pass
            return final_loss
        else:
            total_time = time.time() - start_time
            pass
            return torch.tensor(0.0, device=device, requires_grad=True)

    def _build_reaction_gene_mapping(self, gene_rules_list, data):
        """
        PERFORMANCE: Pre-compute and cache gene rule parsing for all reactions.
        
        This avoids repeated regex operations and string parsing during training.
        """
        self._reaction_gene_mapping = {}
        
        for r_idx, gene_rule in enumerate(gene_rules_list):
            if gene_rule and gene_rule.strip():
                # Parse gene rule once and cache the result
                genes = self._parse_gene_rule_for_evaluation(gene_rule)
                if genes:
                    # Store parsed genes and logical operation type
                    op_type = 'and' if 'and' in gene_rule.lower() else ('or' if 'or' in gene_rule.lower() else 'single')
                    self._reaction_gene_mapping[r_idx] = {
                        'genes': genes,
                        'operation': op_type,
                        'raw_rule': gene_rule
                    }

    def _compute_cached_gene_score_fast(self, reaction_idx, gene_expr, data):
        """
        OPTIMIZED: Faster version of gene score computation with minimal allocations.
        """
        if reaction_idx not in self._reaction_gene_mapping:
            # Fallback: use gene expression directly from reaction node
            if reaction_idx < gene_expr.size(0):
                return torch.mean(gene_expr[reaction_idx])
            return None
        
        mapping = self._reaction_gene_mapping[reaction_idx]
        genes = mapping['genes']
        operation = mapping['operation']
        
        # Early exit if reaction index is out of bounds
        if reaction_idx >= gene_expr.size(0):
            return None
        
        # Get reaction gene features: [gene_dim] - single tensor access
        reaction_gene_features = gene_expr[reaction_idx]
        
        # Optimize gene expression collection
        if not hasattr(data, 'gene_to_index'):
            return None
            
        # Pre-allocate list for better performance
        valid_expressions = []
        gene_to_index = data.gene_to_index  # Cache reference
        
        # Process genes with optimized loop
        for gene_id in genes:
            gene_feature_idx = gene_to_index.get(gene_id)  # Use dict.get() for efficiency
            if gene_feature_idx is not None and gene_feature_idx < reaction_gene_features.size(0):
                expr_val = reaction_gene_features[gene_feature_idx]
                if expr_val > EXPRESSION_THRESHOLD:  # Vectorized comparison
                    valid_expressions.append(expr_val)
        
        if not valid_expressions:
            return None
        
        # Efficient tensor operations based on operation type
        if len(valid_expressions) == 1:
            # Ensure scalar tensor for consistency
            result = valid_expressions[0]
            # Convert to scalar if needed, but preserve gradients
            if result.dim() > 0:
                result = result.squeeze()
            return result
        
        expr_tensor = torch.stack(valid_expressions)
        
        # Use optimized operations - ensure scalar output
        if operation == 'and':
            result = torch.min(expr_tensor)
        elif operation == 'or':
            result = torch.max(expr_tensor)
        else:
            result = torch.mean(expr_tensor)
        
        # Ensure scalar tensor with gradients preserved
        if result.dim() > 0:
            result = result.squeeze()
        return result



    def _compute_cached_gene_score(self, reaction_idx, gene_expr, data):
        """
        PERFORMANCE: Use cached gene mapping to quickly compute gene score.
        """
        if reaction_idx not in self._reaction_gene_mapping:
            # Fallback: use gene expression directly from reaction node
            if reaction_idx < gene_expr.size(0):
                return torch.mean(gene_expr[reaction_idx])
            return None
        
        mapping = self._reaction_gene_mapping[reaction_idx]
        genes = mapping['genes']
        operation = mapping['operation']
        
        # Extract expression values for specific genes
        gene_expressions = []
        
        if reaction_idx < gene_expr.size(0):
            reaction_gene_features = gene_expr[reaction_idx]  # [gene_dim]
            
            for gene_id in genes:
                if hasattr(data, 'gene_to_index') and gene_id in data.gene_to_index:
                    gene_feature_idx = data.gene_to_index[gene_id]
                    
                    # Get the expression value for this specific gene
                    if gene_feature_idx < reaction_gene_features.size(0):
                        expr_val = reaction_gene_features[gene_feature_idx]
                        
                        # Only use non-zero values (zero means gene not measured or not expressed)
                        if expr_val > EXPRESSION_THRESHOLD:  # Small threshold to avoid floating point issues
                            gene_expressions.append(expr_val)
        
        if not gene_expressions:
            return None
        
        # Apply cached logical operation
        if operation == 'and':
            # AND operation: take minimum (bottleneck)
            return torch.min(torch.stack(gene_expressions))
        elif operation == 'or':
            # OR operation: take maximum (best alternative)
            return torch.max(torch.stack(gene_expressions))
        else:
            # Simple gene: take mean
            return torch.mean(torch.stack(gene_expressions))
    
    def _parse_gene_rule_for_evaluation(self, gene_rule):
        """Parse gene rule string to extract individual gene IDs."""
        return parse_gene_rule(gene_rule)
    
    def _compute_pearson_correlation(self, x, y):
        """
        Compute Pearson correlation coefficient between two tensors.
        
        Args:
            x, y: 1D tensors of same length
            
        Returns:
            correlation: scalar correlation coefficient
        """
        if len(x) < 2 or len(y) < 2:
            return torch.tensor(0.0, device=x.device)
        
        # Center the data
        x_centered = x - torch.mean(x)
        y_centered = y - torch.mean(y)
        
        # Compute correlation
        numerator = torch.sum(x_centered * y_centered)
        denominator = torch.sqrt(torch.sum(x_centered ** 2) * torch.sum(y_centered ** 2))
        
        # Avoid division by zero
        if denominator < CORRELATION_EPS:
            return torch.tensor(0.0, device=x.device)
        
        correlation = numerator / denominator
        return torch.clamp(correlation, -1.0, 1.0)  # Ensure valid correlation range


