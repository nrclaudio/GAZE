"""Flux message module: context-aware flux messages for metabolite embeddings.

Computes production, consumption, and activity messages from predicted reaction
fluxes and injects them additively into metabolite embeddings.
"""

import torch
import torch.nn as nn


def _assert_finite(x: torch.Tensor, name: str) -> None:
    if not torch.isfinite(x).all():
        raise RuntimeError(f"{name} has NaN/Inf")


class FluxMessageComputer(nn.Module):
    """Computes production, consumption, and activity messages from reaction fluxes.

    Args:
        hidden_dim: Dimension of metabolite embeddings.
        compress_alpha: Exponent for dynamic range compression (default: 0.5).
        use_transport: Whether to include transport term (default: True).
        transport_hops: Number of transport hops (default: 1).
    """

    def __init__(self, hidden_dim: int, compress_alpha: float = 0.5,
                 use_transport: bool = False, transport_hops: int = 1):
        super().__init__()
        self.compress_alpha = compress_alpha
        self.use_transport = use_transport
        self.transport_hops = transport_hops

        self.flux_message_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )
        self.msg_norm = nn.LayerNorm(3, eps=1e-5)

        if self.use_transport:
            self.transport_reaction_transform = nn.Linear(hidden_dim, hidden_dim)
            if transport_hops >= 2:
                self.transport_metabolite_transform = nn.Linear(hidden_dim, hidden_dim)

    def compute_flux_messages(self, fluxes, stoichiometric_matrix, return_components=False):
        """Compute ``[p, c, a]`` messages from reaction fluxes."""
        is_single = fluxes.dim() == 1
        if is_single:
            fluxes = fluxes.unsqueeze(0)

        matrix = stoichiometric_matrix
        if matrix.device != fluxes.device:
            matrix = matrix.to(fluxes.device)
        S = matrix.to_dense() if matrix.is_sparse else matrix

        S_plus = torch.clamp(S, min=0)
        S_minus = torch.clamp(-S, min=0)
        eps = 1e-12

        production = torch.mm(S_plus, fluxes.T).T
        _assert_finite(production, "production")

        consumption = torch.mm(S_minus, fluxes.T).T
        _assert_finite(consumption, "consumption")

        S_abs = torch.abs(S)
        flux_mag_comp = torch.pow(torch.abs(fluxes) + eps, self.compress_alpha)
        activity = torch.mm(S_abs, flux_mag_comp.T).T
        _assert_finite(activity, "activity")

        messages = torch.stack([production, consumption, activity], dim=-1)
        messages = self.msg_norm(messages)
        _assert_finite(messages, "normalized_messages")

        if is_single:
            messages = messages.squeeze(0)
            production, consumption, activity = production.squeeze(0), consumption.squeeze(0), activity.squeeze(0)

        if return_components:
            return messages, {"production": production, "consumption": consumption, "activity": activity}
        return messages

    def compute_transport_messages(self, metabolite_embeddings, reaction_embeddings,
                                   fluxes, stoichiometric_matrix,
                                   metabolite_indices, reaction_indices):
        """Compute flux-weighted transport messages between metabolites via reactions."""
        is_single = fluxes.dim() == 1
        if is_single:
            fluxes = fluxes.unsqueeze(0)
            metabolite_embeddings = metabolite_embeddings.unsqueeze(0)
            reaction_embeddings = reaction_embeddings.unsqueeze(0)

        batch_size = fluxes.size(0)
        num_metabolites = len(metabolite_indices)
        hidden_dim = metabolite_embeddings.size(-1)

        matrix = stoichiometric_matrix
        target_device = fluxes.device
        if matrix.device != target_device:
            matrix = matrix.to(target_device)
        S = matrix.to_dense() if matrix.is_sparse else matrix

        num_reactions = reaction_embeddings.size(-2)
        met_idx = torch.arange(num_metabolites, device=S.device)
        rxn_idx = torch.arange(num_reactions, device=S.device)

        if num_metabolites > S.size(0) or num_reactions > S.size(1):
            out = torch.zeros(batch_size, num_metabolites, hidden_dim, device=target_device)
            return out.squeeze(0) if is_single else out

        S_sub = S[met_idx][:, rxn_idx]
        S_abs = torch.abs(S_sub)
        eps = 1e-12

        transport_messages = []
        for b in range(batch_size):
            flux_weights = torch.pow(torch.abs(fluxes[b]) + eps, self.compress_alpha)
            T_MR = S_abs * flux_weights.unsqueeze(0)
            row_sums = torch.clamp(T_MR.sum(dim=1, keepdim=True), min=1e-8)
            T_MR_norm = T_MR / row_sums
            _assert_finite(T_MR_norm, "T_MR_norm")

            rxn_messages = self.transport_reaction_transform(reaction_embeddings[b])
            one_hop = torch.mm(T_MR_norm, rxn_messages)

            if self.transport_hops >= 2:
                T_RM = T_MR_norm.T
                T_MM = torch.mm(T_MR_norm, T_RM)
                mm_sums = torch.clamp(T_MM.sum(dim=1, keepdim=True), min=1e-8)
                T_MM_norm = T_MM / mm_sums
                met_messages = self.transport_metabolite_transform(metabolite_embeddings[b])
                two_hop = torch.mm(T_MM_norm, met_messages)
                total = one_hop + two_hop
            else:
                total = one_hop

            _assert_finite(total, "transport_messages")
            transport_messages.append(total)

        transport_messages = torch.stack(transport_messages, dim=0)
        return transport_messages.squeeze(0) if is_single else transport_messages

    def forward(self, fluxes, stoichiometric_matrix, metabolite_embeddings=None,
                reaction_embeddings=None, metabolite_indices=None, reaction_indices=None):
        flux_messages = self.compute_flux_messages(fluxes, stoichiometric_matrix)
        flux_updates = self.flux_message_mlp(flux_messages)
        _assert_finite(flux_updates, "flux_updates")

        transport_updates = None
        if self.use_transport and metabolite_embeddings is not None and reaction_embeddings is not None:
            transport_updates = self.compute_transport_messages(
                metabolite_embeddings, reaction_embeddings, fluxes,
                stoichiometric_matrix, metabolite_indices, reaction_indices,
            )
        return flux_updates, transport_updates


class AdditiveFluxMessageLayer(nn.Module):
    """Additively incorporates flux messages into metabolite embeddings.

    Args:
        hidden_dim: Dimension of embeddings.
        compress_alpha: Exponent for dynamic range compression.
        use_transport: Whether to include transport term.
        transport_hops: Number of transport hops.
    """

    def __init__(self, hidden_dim: int, compress_alpha: float = 0.5,
                 use_transport: bool = False, transport_hops: int = 1):
        super().__init__()
        self.flux_message_computer = FluxMessageComputer(
            hidden_dim=hidden_dim,
            compress_alpha=compress_alpha,
            use_transport=use_transport,
            transport_hops=transport_hops,
        )
        self.flux_gate_log = nn.Parameter(torch.tensor(-6.0))
        self.transport_gate_log = nn.Parameter(torch.tensor(-6.0)) if use_transport else None

    def forward(self, metabolite_embeddings, reaction_embeddings, fluxes,
                stoichiometric_matrix, metabolite_indices, reaction_indices):
        flux_updates, transport_updates = self.flux_message_computer(
            fluxes=fluxes,
            stoichiometric_matrix=stoichiometric_matrix,
            metabolite_embeddings=metabolite_embeddings,
            reaction_embeddings=reaction_embeddings,
            metabolite_indices=metabolite_indices,
            reaction_indices=reaction_indices,
        )
        gate = torch.exp(self.flux_gate_log)
        updated = metabolite_embeddings + gate * flux_updates

        if transport_updates is not None and self.transport_gate_log is not None:
            tgate = torch.exp(self.transport_gate_log)
            updated = updated + tgate * transport_updates

        return updated
