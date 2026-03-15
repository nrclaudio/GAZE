"""Mass balance projection via cached Cholesky factorization.

Projects unconstrained flux predictions onto the mass-balanced subspace,
ensuring exact satisfaction of Sv = 0 constraints.
"""

import warnings

import torch
import torch.nn as nn

# Inline the constant to avoid an extra constants module
MASS_BALANCE_BASE_EPS_FACTOR = 1e-6


class MassBalanceProjector(nn.Module):
    """Efficient mass balance projection using cached Cholesky factorization.

    Ensures flux predictions exactly satisfy stoichiometric constraints ``Sv = 0``
    by projecting onto the null space of the stoichiometric matrix.

    Args:
        stoichiometric_matrix: Sparse tensor ``[M, R]``.
        max_regularization: Maximum epsilon value to try.
        use_float64: Use double precision for Cholesky operations.
    """

    def __init__(self, stoichiometric_matrix=None, max_regularization: float = 1e-2,
                 use_float64: bool = False):
        super().__init__()
        self.max_regularization = max_regularization
        self.use_float64 = use_float64

        self.L = None
        self.eps = None
        self.cached_device = None
        self.cached_dtype = None
        self.num_metabolites = None
        self.num_reactions = None

        if stoichiometric_matrix is not None:
            self._precompute_cholesky_factor(stoichiometric_matrix)

    # ------------------------------------------------------------------
    def _precompute_cholesky_factor(self, S):
        device = S.device
        dtype = S.dtype
        working_dtype = torch.float64 if self.use_float64 else torch.float32

        S_dense = S.to_dense().to(dtype=working_dtype) if S.is_sparse else S.to(dtype=working_dtype)
        M = torch.mm(S_dense, S_dense.T)

        base_eps = MASS_BALANCE_BASE_EPS_FACTOR * torch.mean(torch.diag(M))
        eps = base_eps

        for _ in range(5):
            try:
                M_reg = M + eps * torch.eye(M.size(0), dtype=working_dtype, device=device)
                L = torch.linalg.cholesky(M_reg)
                self.L = L.to(dtype=dtype, device=device)
                self.eps = eps.item() if torch.is_tensor(eps) else eps
                self.cached_device = device
                self.cached_dtype = dtype
                self.num_metabolites = S.size(0)
                self.num_reactions = S.size(1)
                return
            except RuntimeError as e:
                if "cholesky" in str(e).lower():
                    eps = min(eps * 10, self.max_regularization)
                    if eps >= self.max_regularization:
                        break
                else:
                    raise

        raise RuntimeError("Could not compute stable Cholesky decomposition for mass balance projection")

    def _validate_and_refresh_cache(self, S):
        needs = (
            self.L is None
            or self.cached_device != S.device
            or self.cached_dtype != S.dtype
            or self.num_metabolites != S.size(0)
            or self.num_reactions != S.size(1)
        )
        if needs:
            self._precompute_cholesky_factor(S)

    # ------------------------------------------------------------------
    def project_to_mass_balance(self, y, S):
        """Project unconstrained flux predictions onto mass-balanced subspace.

        Solves: ``min ||v - y||^2  s.t.  Sv = 0``

        Args:
            y: Unconstrained flux predictions ``[B, R]`` or ``[R]``.
            S: Stoichiometric matrix ``[M, R]`` (can be sparse).

        Returns:
            Mass-balanced flux predictions (same shape as *y*).
        """
        is_single = y.dim() == 1
        if is_single:
            y = y.unsqueeze(0)

        if S.device != y.device:
            S = S.to(y.device)
        self._validate_and_refresh_cache(S)
        S_dense = S.to_dense() if S.is_sparse else S

        try:
            r = torch.mm(S_dense, y.T)
            L_dev = self.L.to(device=y.device, dtype=y.dtype)
            z = torch.linalg.solve_triangular(L_dev, r, upper=False)
            w = torch.linalg.solve_triangular(L_dev.T, z, upper=True)
            correction = torch.mm(S_dense.T, w)
            v = y - correction.T
        except RuntimeError:
            warnings.warn("Triangular solve failed; using pseudo-inverse fallback.")
            v = self._project_pseudoinverse_fallback(y, S_dense)

        return v.squeeze(0) if is_single else v

    def _project_pseudoinverse_fallback(self, y, S_dense):
        try:
            SST = torch.mm(S_dense, S_dense.T)
            reg = 1e-6 * torch.eye(SST.size(0), device=SST.device, dtype=SST.dtype)
            SST_pinv = torch.pinverse(SST + reg)

            projected = []
            for b in range(y.size(0)):
                r_b = torch.mv(S_dense, y[b])
                w_b = torch.mv(SST_pinv, r_b)
                correction_b = torch.mv(S_dense.T, w_b)
                projected.append(y[b] - correction_b)
            return torch.stack(projected, dim=0)
        except Exception:
            warnings.warn("Pseudo-inverse fallback also failed. Returning original fluxes.")
            return y

    def forward(self, y, S):
        """Forward pass for ``nn.Module`` compatibility."""
        return self.project_to_mass_balance(y, S)
