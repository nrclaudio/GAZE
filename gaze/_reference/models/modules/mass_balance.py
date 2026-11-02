"""
Mass Balance Projection Module

This module implements Cholesky-based projection of unconstrained flux predictions
onto the mass-balanced subspace, ensuring exact satisfaction of Sv = 0 constraints.

Key mathematical operations:
1. Precompute M = S @ S.T + eps * I with adaptive regularization
2. Cache Cholesky factor L = cholesky(M) for efficient projection
3. Project fluxes: v = y - S.T @ (L \\ (S @ y))

The projection solves: min ||v - y||² subject to Sv = 0
"""

import torch
import torch.nn as nn
import warnings
from ...utils.constants import MASS_BALANCE_BASE_EPS_FACTOR


class MassBalanceProjector(nn.Module):
    """
    Efficient mass balance projection using cached Cholesky factorization.
    
    This module ensures that flux predictions exactly satisfy stoichiometric 
    constraints Sv = 0 by projecting onto the null space of the stoichiometric matrix.
    
    Key features:
    - Caches Cholesky factor for O(R²) projection instead of O(R³) 
    - Adaptive epsilon regularization for numerical stability
    - Handles device/dtype changes automatically
    - Fallback to iterative methods if Cholesky fails
    
    Args:
        stoichiometric_matrix: Sparse tensor [M, R] where M=metabolites, R=reactions
        max_regularization: Maximum epsilon value to try (default: 1e-2)
        use_float64: Use double precision for Cholesky operations (default: False)
    """
    
    def __init__(self, stoichiometric_matrix=None, max_regularization=1e-2, use_float64=False):
        super().__init__()
        
        self.max_regularization = max_regularization
        self.use_float64 = use_float64
        
        # Cached components (will be computed when needed)
        self.L = None  # Cholesky factor
        self.eps = None  # Regularization parameter used
        self.cached_device = None
        self.cached_dtype = None
        
        # Store original matrix shape for validation
        self.num_metabolites = None
        self.num_reactions = None
        
        if stoichiometric_matrix is not None:
            self._precompute_cholesky_factor(stoichiometric_matrix)
    
    def _precompute_cholesky_factor(self, S):
        """
        Precompute and cache the Cholesky factor for efficient projection.
        
        Computes M = S @ S.T + eps * I and L = cholesky(M), trying multiple
        eps values if needed for numerical stability.
        
        Args:
            S: Stoichiometric matrix [M, R] (can be sparse)
        """
        device = S.device
        dtype = S.dtype
        
        # Use higher precision for factorization if requested
        working_dtype = torch.float64 if self.use_float64 else torch.float32
        
        # Convert to dense and move to working precision
        if S.is_sparse:
            S_dense = S.to_dense().to(dtype=working_dtype)
        else:
            S_dense = S.to(dtype=working_dtype)
        
        # Compute M = S @ S.T
        M = torch.mm(S_dense, S_dense.T)  # [M, M]
        
        # Try different regularization levels
        base_eps = MASS_BALANCE_BASE_EPS_FACTOR * torch.mean(torch.diag(M))
        eps = base_eps
        
        for attempt in range(5):  # Try up to 5 different eps values
            try:
                # Add regularization: M_reg = M + eps * I
                M_reg = M + eps * torch.eye(M.size(0), dtype=working_dtype, device=device)
                
                # Attempt Cholesky decomposition
                L = torch.linalg.cholesky(M_reg)
                
                # Success! Cache the results
                self.L = L.to(dtype=dtype, device=device)
                self.eps = eps.item() if torch.is_tensor(eps) else eps
                self.cached_device = device
                self.cached_dtype = dtype
                self.num_metabolites = S.size(0)
                self.num_reactions = S.size(1)
                
                print(f"Mass balance Cholesky factor computed with eps={self.eps:.2e}")
                return
                
            except RuntimeError as e:
                if "cholesky" in str(e).lower():
                    # Cholesky failed - try larger eps
                    eps = min(eps * 10, self.max_regularization)
                    if eps >= self.max_regularization:
                        break
                else:
                    raise e
        
        # If we reach here, Cholesky failed even with max regularization
        warnings.warn(f"Cholesky decomposition failed even with eps={eps:.2e}. "
                     f"Consider using iterative projection fallback.")
        raise RuntimeError(f"Could not compute stable Cholesky decomposition for mass balance projection")
    
    def _validate_and_refresh_cache(self, S):
        """
        Check if cached Cholesky factor is valid for current stoichiometric matrix.
        Recompute if device/dtype changed or if matrix dimensions don't match.
        """
        current_device = S.device
        current_dtype = S.dtype
        
        # Check if we need to recompute
        needs_recompute = (
            self.L is None or
            self.cached_device != current_device or
            self.cached_dtype != current_dtype or
            self.num_metabolites != S.size(0) or
            self.num_reactions != S.size(1)
        )
        
        if needs_recompute:
            self._precompute_cholesky_factor(S)
    
    def project_to_mass_balance(self, y, S):
        """
        Project unconstrained flux predictions onto mass-balanced subspace.
        
        Solves: min ||v - y||^2 subject to Sv = 0
        Solution: v = y - S.T @ w where (S @ S.T + eps*I) @ w = S @ y
        
        Args:
            y: Unconstrained flux predictions [batch_size, R] or [R]
            S: Stoichiometric matrix [M, R] (can be sparse)
            
        Returns:
            v: Mass-balanced flux predictions (same shape as y)
        """
        # Handle single sample case
        is_single_sample = (y.dim() == 1)
        if is_single_sample:
            y = y.unsqueeze(0)  # [1, R]
        
        batch_size, num_reactions = y.shape

        # Ensure matrix is on same device as y for subsequent matmuls
        if S.device != y.device:
            S = S.to(y.device)
        
        # Validate and refresh cache if needed (recomputes if device/dtype changed)
        self._validate_and_refresh_cache(S)
        
        # Convert to dense if sparse (for matrix multiplication)
        if S.is_sparse:
            S_dense = S.to_dense()
        else:
            S_dense = S
        
        try:
            # Compute r = S @ y for all samples: [M, R] @ [B, R].T = [M, B]
            r = torch.mm(S_dense, y.T)  # [M, B]
            
            # Solve (S @ S.T + eps*I) @ w = r using cached Cholesky factor
            # L @ L.T @ w = r  =>  w = L.T \ (L \ r)
            L_device = self.L.to(device=y.device, dtype=y.dtype)
            
            # Forward solve: L @ z = r
            z = torch.linalg.solve_triangular(L_device, r, upper=False)  # [M, B]
            
            # Backward solve: L.T @ w = z  
            w = torch.linalg.solve_triangular(L_device.T, z, upper=True)  # [M, B]
            
            # Compute correction: S.T @ w = [R, M] @ [M, B] = [R, B]
            correction = torch.mm(S_dense.T, w)  # [R, B]
            
            # Apply projection: v = y - correction.T = [B, R] - [B, R]
            v = y - correction.T
            
        except RuntimeError as e:
            # Fallback to pseudo-inverse if triangular solve fails
            warnings.warn(f"Triangular solve failed: {e}. Using pseudo-inverse fallback.")
            v = self._project_pseudoinverse_fallback(y, S_dense)
        
        # Return original shape
        if is_single_sample:
            v = v.squeeze(0)
        
        return v
    
    def _project_pseudoinverse_fallback(self, y, S_dense):
        """
        Fallback projection using pseudo-inverse when Cholesky methods fail.
        Less efficient but more robust.
        """
        batch_size = y.size(0)
        
        try:
            # Compute projection matrix P = I - S.T @ (S @ S.T)^+ @ S
            SST = torch.mm(S_dense, S_dense.T)
            
            # Add small regularization for pseudo-inverse
            reg = 1e-6 * torch.eye(SST.size(0), device=SST.device, dtype=SST.dtype)
            SST_reg = SST + reg
            
            # Pseudo-inverse
            SST_pinv = torch.pinverse(SST_reg)
            
            # Compute projection for each sample
            projected = []
            for b in range(batch_size):
                y_b = y[b]  # [R]
                r_b = torch.mv(S_dense, y_b)  # [M]
                w_b = torch.mv(SST_pinv, r_b)  # [M] 
                correction_b = torch.mv(S_dense.T, w_b)  # [R]
                v_b = y_b - correction_b
                projected.append(v_b)
            
            return torch.stack(projected, dim=0)
            
        except Exception as e:
            warnings.warn(f"Pseudo-inverse fallback also failed: {e}. Returning original fluxes.")
            return y
    
    def verify_mass_balance(self, v, S, tolerance=1e-6):
        """
        Verify that projected fluxes satisfy mass balance constraints.
        
        Args:
            v: Projected flux predictions [batch_size, R] or [R]
            S: Stoichiometric matrix [M, R]
            tolerance: Maximum allowed residual norm
            
        Returns:
            dict: Verification results with residual norms and pass/fail status
        """
        is_single_sample = (v.dim() == 1)
        if is_single_sample:
            v = v.unsqueeze(0)
        
        # Convert to dense if needed
        if S.is_sparse:
            S_dense = S.to_dense()
        else:
            S_dense = S
        
        # Compute residuals: delta = S @ v.T  [M, B]
        residuals = torch.mm(S_dense, v.T)
        
        # Compute norms for each sample
        residual_norms = torch.norm(residuals, dim=0)  # [B]
        
        results = {
            'residual_norms': residual_norms.squeeze() if is_single_sample else residual_norms,
            'max_residual': torch.max(residual_norms).item(),
            'mean_residual': torch.mean(residual_norms).item(),
            'passes_tolerance': torch.all(residual_norms <= tolerance).item()
        }
        
        return results
    
    def forward(self, y, S):
        """Forward pass for nn.Module compatibility."""
        return self.project_to_mass_balance(y, S)


def split_reversible_reactions(S, reaction_reversibility=None):
    """
    Split reversible reactions into separate forward and backward components.
    
    This is required for proper mass balance projection since reversible reactions
    can have negative fluxes, but the model should predict non-negative values.
    
    Args:
        S: Original stoichiometric matrix [M, R]
        reaction_reversibility: Boolean tensor [R] indicating which reactions are reversible
                               If None, assumes all reactions are irreversible
    
    Returns:
        S_split: Extended stoichiometric matrix [M, R_split] where R_split >= R
        reaction_mapping: Dict mapping original reaction indices to split indices
    """
    if reaction_reversibility is None:
        # No reversible reactions - return original matrix
        return S, {i: i for i in range(S.size(1))}
    
    device = S.device
    dtype = S.dtype
    M, R = S.shape
    
    # Convert to dense for easier manipulation
    if S.is_sparse:
        S_dense = S.to_dense()
    else:
        S_dense = S.clone()
    
    # Find reversible reactions
    reversible_indices = torch.nonzero(reaction_reversibility).squeeze(-1)
    irreversible_indices = torch.nonzero(~reaction_reversibility).squeeze(-1)
    
    num_reversible = len(reversible_indices)
    num_irreversible = len(irreversible_indices)
    R_split = num_irreversible + 2 * num_reversible
    
    # Create extended matrix
    S_split = torch.zeros(M, R_split, device=device, dtype=dtype)
    reaction_mapping = {}
    
    current_col = 0
    
    # Add irreversible reactions as-is
    for i, orig_idx in enumerate(irreversible_indices):
        S_split[:, current_col] = S_dense[:, orig_idx]
        reaction_mapping[orig_idx.item()] = current_col
        current_col += 1
    
    # Split reversible reactions
    for orig_idx in reversible_indices:
        # Forward reaction: same stoichiometry
        S_split[:, current_col] = S_dense[:, orig_idx]
        reaction_mapping[f"{orig_idx.item()}_fwd"] = current_col
        current_col += 1
        
        # Backward reaction: negated stoichiometry  
        S_split[:, current_col] = -S_dense[:, orig_idx]
        reaction_mapping[f"{orig_idx.item()}_bwd"] = current_col
        current_col += 1
    
    return S_split, reaction_mapping


def create_mass_balance_projector_from_dataset(dataset):
    """
    Convenience function to create MassBalanceProjector from a dataset.
    
    Args:
        dataset: Dataset object with incidence_matrix attribute
        
    Returns:
        MassBalanceProjector: Configured projector ready for use
    """
    if not hasattr(dataset, 'incidence_matrix'):
        raise ValueError("Dataset must have 'incidence_matrix' attribute")
    
    # The incidence matrix is typically metabolites x reactions
    S = dataset.incidence_matrix
    
    print(f"Creating mass balance projector for {S.size(0)} metabolites, {S.size(1)} reactions")
    
    # Check for reversible reactions (this would need to be added to dataset)
    reversible = getattr(dataset, 'reaction_reversibility', None)
    
    if reversible is not None:
        print(f"Splitting {reversible.sum()} reversible reactions")
        S_split, mapping = split_reversible_reactions(S, reversible)
        projector = MassBalanceProjector(S_split)
        projector.reaction_mapping = mapping
        return projector
    else:
        return MassBalanceProjector(S)
