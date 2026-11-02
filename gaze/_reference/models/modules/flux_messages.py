"""
Flux Message Module

This module implements context-aware flux messages that inform metabolite embeddings
about their production, consumption, and metabolic activity in the network.

Key components:
1. Production messages: p = S_plus @ v (metabolites produced by reactions)
2. Consumption messages: c = S_minus @ v (metabolites consumed by reactions)  
3. Activity messages: a = |S| @ |v|^alpha (metabolic throughput/activity)
4. Transport term: flux-weighted message passing between metabolites via reactions

These messages are additive and provide physical context for unseen metabolites.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _assert_finite(x, name):
    """Helper function to check for NaN/Inf values and raise clear error messages."""
    if not torch.isfinite(x).all():
        raise RuntimeError(f"{name} has NaN/Inf")


class FluxMessageComputer(nn.Module):
    """
    Computes production, consumption, and activity messages from reaction fluxes.
    
    These messages provide metabolite embeddings with context about their role
    in the metabolic network based on predicted reaction fluxes.
    
    Key insights:
    - Production (p): How much of metabolite is being produced
    - Consumption (c): How much of metabolite is being consumed  
    - Activity (a): Nonnegative metabolic throughput involving metabolite
    - Transport: Multi-hop connectivity via flux-weighted adjacency
    
    Args:
        hidden_dim: Dimension of metabolite embeddings
        compress_alpha: Exponent for dynamic range compression (default: 0.5)
        use_transport: Whether to include transport term (default: True)
        transport_hops: Number of transport hops (1 or 2, default: 1)
    """
    
    def __init__(self, hidden_dim, compress_alpha=0.5, use_transport=False, transport_hops=1):
        super().__init__()
        
        self.compress_alpha = compress_alpha
        self.use_transport = use_transport  
        self.transport_hops = transport_hops
        
        
        # MLP to transform [p, c, a] into embedding updates
        self.flux_message_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),  # 3 inputs: production, consumption, activity
            nn.LeakyReLU(0.1),  # Better gradient flow than ReLU
            nn.Linear(hidden_dim // 2, hidden_dim)
            # Removed LayerNorm - causes NaNs with batch_size=1
        )
        self.msg_norm = nn.LayerNorm(3, eps=1e-5)
        # Transport term MLPs (if enabled)
        if self.use_transport:
            # Transform reaction embeddings for transport
            self.transport_reaction_transform = nn.Linear(hidden_dim, hidden_dim)
            
            if transport_hops >= 2:
                # Two-hop transport: metabolite -> reaction -> metabolite -> metabolite
                self.transport_metabolite_transform = nn.Linear(hidden_dim, hidden_dim)
    
    def compute_flux_messages(self, fluxes, stoichiometric_matrix, return_components=False):
        """
        Compute production, consumption, and activity messages from reaction fluxes.
        
        Args:
            fluxes: Reaction flux predictions [batch_size, R] or [R]
            stoichiometric_matrix: Stoichiometric matrix [M, R] (can be sparse)
            return_components: If True, return individual p, c, a components
            
        Returns:
            messages: Flux message tensor [batch_size, M, 3] or [M, 3]
            components: Dict with 'production', 'consumption', 'activity' (if requested)
        """
        # Handle single sample case
        is_single_sample = (fluxes.dim() == 1)
        if is_single_sample:
            fluxes = fluxes.unsqueeze(0)  # [1, R]
        
        batch_size, num_reactions = fluxes.shape
        
        # Align stoichiometric matrix device with fluxes before dense computations
        matrix = stoichiometric_matrix
        if matrix.device != fluxes.device:
            matrix = matrix.to(fluxes.device)

        # Convert stoichiometric matrix to dense for computations
        if matrix.is_sparse:
            S = matrix.to_dense()
        else:
            S = matrix
        
        num_metabolites = S.size(0)
        
        # Split stoichiometric matrix into production and consumption parts
        # S_plus: positive coefficients (products)
        # S_minus: negative coefficients made positive (reactants)
        S_plus = torch.clamp(S, min=0)  # [M, R] - products
        S_minus = torch.clamp(-S, min=0)  # [M, R] - reactants (made positive)
        
        # Compute messages with numerical stability
        eps = 1e-12
        
        # Production: p = S_plus @ v  [M, R] @ [B, R].T = [M, B]
        production = torch.mm(S_plus, fluxes.T)  # [M, B]
        _assert_finite(production, "production")
        
        # Consumption: c = S_minus @ v  [M, R] @ [B, R].T = [M, B]  
        consumption = torch.mm(S_minus, fluxes.T)  # [M, B]
        _assert_finite(consumption, "consumption")
        
        # Activity: a = |S| @ |v|^alpha for dynamic range compression
        S_abs = torch.abs(S)  # [M, R]
        flux_mag = torch.abs(fluxes)
        flux_mag_comp = torch.pow(flux_mag + eps, self.compress_alpha)
        activity = torch.mm(S_abs, flux_mag_comp.T)  # [M, B]
        _assert_finite(activity, "activity")
        
        # Transpose to batch-first format and stack
        production = production.T  # [B, M]
        consumption = consumption.T  # [B, M] 
        activity = activity.T  # [B, M]
        
        # Stack into message tensor [B, M, 3]
        messages = torch.stack([production, consumption, activity], dim=-1)
        
        # Normalize across metabolites (dimension 1) for each feature
        # This compares metabolite activities relative to each other
        messages = self.msg_norm(messages)
        _assert_finite(messages, "normalized_messages")
        
        # Return original shape if single sample
        if is_single_sample:
            messages = messages.squeeze(0)  # [M, 3]
            production = production.squeeze(0)  # [M]
            consumption = consumption.squeeze(0)  # [M]
            activity = activity.squeeze(0)  # [M]
        
        if return_components:
            components = {
                'production': production,
                'consumption': consumption,  
                'activity': activity
            }
            return messages, components
        
        return messages
    
    def compute_transport_messages(self, metabolite_embeddings, reaction_embeddings, 
                                 fluxes, stoichiometric_matrix, metabolite_indices, reaction_indices):
        """
        Compute flux-weighted transport messages between metabolites via reactions.
        
        This implements metabolite-reaction-metabolite message passing weighted by flux magnitudes.
        Provides connectivity information for metabolites that may not be directly connected.
        
        Args:
            metabolite_embeddings: Metabolite embeddings [batch_size, M, hidden_dim]
            reaction_embeddings: Reaction embeddings [batch_size, R, hidden_dim]
            fluxes: Reaction fluxes [batch_size, R]
            stoichiometric_matrix: Stoichiometric matrix [M_total, R_total] for full graph
            metabolite_indices: Indices of metabolite nodes in full graph
            reaction_indices: Indices of reaction nodes in full graph
            
        Returns:
            transport_messages: Transport messages [batch_size, M, hidden_dim]
        """
        # Handle single sample case
        is_single_sample = (fluxes.dim() == 1)
        if is_single_sample:
            fluxes = fluxes.unsqueeze(0)
            metabolite_embeddings = metabolite_embeddings.unsqueeze(0)
            reaction_embeddings = reaction_embeddings.unsqueeze(0)
        
        batch_size = fluxes.size(0)
        num_metabolites = len(metabolite_indices)
        hidden_dim = metabolite_embeddings.size(-1)
        
        # Align stoichiometric matrix device with fluxes before dense computations
        matrix = stoichiometric_matrix
        target_device = fluxes.device
        if matrix.device != target_device:
            matrix = matrix.to(target_device)

        # Convert to dense and extract relevant submatrix
        if matrix.is_sparse:
            S = matrix.to_dense()
        else:
            S = matrix
        
        # Use local indices for stoichiometric matrix access
        # Since embeddings are already properly extracted, we just need sequential indices
        # for accessing the stoichiometric matrix which represents the same metabolites/reactions
        num_reactions = reaction_embeddings.size(-2)  # Get number of reactions from embeddings
        
        # For stoichiometric matrix access, use local indices (0, 1, 2, ...)
        met_idx = torch.arange(num_metabolites, device=S.device)
        rxn_idx = torch.arange(num_reactions, device=S.device)
        
        # Check bounds to avoid indexing errors
        if num_metabolites > S.size(0) or num_reactions > S.size(1):
            # Dimension mismatch - return zero transport messages
            print(f"Warning: Embedding dimensions don't match stoichiometric matrix. "
                  f"Metabolites: {num_metabolites} > {S.size(0)}, Reactions: {num_reactions} > {S.size(1)}")
            transport_messages = torch.zeros(batch_size, num_metabolites, hidden_dim, device=S.device)
            if is_single_sample:
                transport_messages = transport_messages.squeeze(0)
            return transport_messages
            
        # Extract relevant submatrix [M, R]
        S_sub = S[met_idx][:, rxn_idx]
        
        # Create flux-weighted adjacency matrix
        # T_MR[m, r] = |S[m,r]| * |flux[r]|^alpha (connectivity weighted by flux)
        S_abs = torch.abs(S_sub)  # [M, R]
        
        # Compute transport messages for each sample
        transport_messages = []
        
        for b in range(batch_size):
            flux_b = fluxes[b]  # [R]
            met_emb_b = metabolite_embeddings[b]  # [M, hidden_dim]
            rxn_emb_b = reaction_embeddings[b]  # [R, hidden_dim]
            
            # Flux-weighted adjacency: T_MR = |S| * |flux|^alpha
            eps = 1e-12
            flux_weights = torch.pow(torch.abs(flux_b) + eps, self.compress_alpha)  # [R]
            T_MR = S_abs * flux_weights.unsqueeze(0)  # [M, R]
            
            # Row-normalize to get transition probabilities
            row_sums = T_MR.sum(dim=1, keepdim=True)  # [M, 1]
            row_sums = torch.clamp(row_sums, min=1e-8)  # Avoid division by zero
            T_MR_norm = T_MR / row_sums  # [M, R]
            _assert_finite(T_MR_norm, "T_MR_norm")
            
            # One-hop transport: metabolite -> reaction
            # Transform reaction embeddings for message passing
            rxn_messages = self.transport_reaction_transform(rxn_emb_b)  # [R, hidden_dim]
            
            # Aggregate messages: [M, R] @ [R, hidden_dim] = [M, hidden_dim]
            one_hop_messages = torch.mm(T_MR_norm, rxn_messages)
            
            if self.transport_hops >= 2:
                # Two-hop transport: metabolite -> reaction -> metabolite -> metabolite
                # Create metabolite-to-metabolite adjacency via reactions
                T_RM = T_MR_norm.T  # [R, M] - transpose for reverse direction
                T_MM = torch.mm(T_MR_norm, T_RM)  # [M, R] @ [R, M] = [M, M]
                
                # Normalize metabolite-to-metabolite transitions
                mm_row_sums = T_MM.sum(dim=1, keepdim=True)
                mm_row_sums = torch.clamp(mm_row_sums, min=1e-8)
                T_MM_norm = T_MM / mm_row_sums
                _assert_finite(T_MM_norm, "T_MM_norm")
                
                # Transform metabolite embeddings for two-hop messages
                met_messages = self.transport_metabolite_transform(met_emb_b)  # [M, hidden_dim]
                
                # Two-hop messages: [M, M] @ [M, hidden_dim] = [M, hidden_dim]  
                two_hop_messages = torch.mm(T_MM_norm, met_messages)
            
                # Combine one-hop and two-hop messages
                total_messages = one_hop_messages + two_hop_messages
            else:
                total_messages = one_hop_messages
                
            _assert_finite(total_messages, "transport_messages")
            
            transport_messages.append(total_messages)
        
        # Stack batch results
        transport_messages = torch.stack(transport_messages, dim=0)  # [B, M, hidden_dim]
        
        # Return original shape if single sample
        if is_single_sample:
            transport_messages = transport_messages.squeeze(0)
        
        return transport_messages
    
    def forward(self, fluxes, stoichiometric_matrix, metabolite_embeddings=None, 
               reaction_embeddings=None, metabolite_indices=None, reaction_indices=None):
        """
        Forward pass: compute flux messages and optionally transport messages.
        
        Args:
            fluxes: Reaction flux predictions [batch_size, R] or [R]
            stoichiometric_matrix: Stoichiometric matrix [M, R] or full graph matrix
            metabolite_embeddings: Current metabolite embeddings (for transport, optional)
            reaction_embeddings: Current reaction embeddings (for transport, optional)
            metabolite_indices: Metabolite indices in full graph (optional)
            reaction_indices: Reaction indices in full graph (optional)
            
        Returns:
            flux_updates: Embedding updates from flux messages [batch_size, M, hidden_dim]
            transport_updates: Transport updates (if enabled and embeddings provided)
        """
        # Compute basic flux messages [p, c, a]
        flux_messages = self.compute_flux_messages(fluxes, stoichiometric_matrix)
        
        # Transform to embedding updates
        flux_updates = self.flux_message_mlp(flux_messages)
        _assert_finite(flux_updates, "flux_updates")
        
        # Compute transport messages if requested and embeddings provided
        transport_updates = None
        if (self.use_transport and metabolite_embeddings is not None and 
            reaction_embeddings is not None):
            transport_updates = self.compute_transport_messages(
                metabolite_embeddings, reaction_embeddings, fluxes,
                stoichiometric_matrix, metabolite_indices, reaction_indices
            )
        
        return flux_updates, transport_updates


class AdditiveFluxMessageLayer(nn.Module):
    """
    Layer that additively incorporates flux messages into metabolite embeddings.
    
    This is designed to be inserted after GNN layers to provide metabolites with
    context about their production, consumption, and activity in the network.
    
    Args:
        hidden_dim: Dimension of embeddings
        flux_message_config: Configuration for flux message computer
    """
    
    def __init__(self, hidden_dim, compress_alpha=0.5, use_transport=False, transport_hops=1):
        super().__init__()
        
        self.flux_message_computer = FluxMessageComputer(
            hidden_dim=hidden_dim,
            compress_alpha=compress_alpha,
            use_transport=use_transport,
            transport_hops=transport_hops
        )
        
        # Exponential gates for flux message contributions (start very small)
        self.flux_gate_log = nn.Parameter(torch.tensor(-6.0))
        self.transport_gate_log = nn.Parameter(torch.tensor(-6.0)) if use_transport else None
    
    def forward(self, metabolite_embeddings, reaction_embeddings, fluxes, 
               stoichiometric_matrix, metabolite_indices, reaction_indices):
        """
        Add flux messages to metabolite embeddings.
        
        Args:
            metabolite_embeddings: Current metabolite embeddings [batch_size, M, hidden_dim]
            reaction_embeddings: Current reaction embeddings [batch_size, R, hidden_dim]  
            fluxes: Predicted reaction fluxes [batch_size, R]
            stoichiometric_matrix: Stoichiometric matrix
            metabolite_indices: Metabolite node indices
            reaction_indices: Reaction node indices
            
        Returns:
            updated_embeddings: Metabolite embeddings with flux context added
        """
        # Compute flux messages
        flux_updates, transport_updates = self.flux_message_computer(
            fluxes=fluxes,
            stoichiometric_matrix=stoichiometric_matrix,
            metabolite_embeddings=metabolite_embeddings,
            reaction_embeddings=reaction_embeddings,
            metabolite_indices=metabolite_indices,
            reaction_indices=reaction_indices
        )
        
        # Gated additive approach with exponential gates
        gate = torch.exp(self.flux_gate_log)
        updated_embeddings = metabolite_embeddings + gate * flux_updates
        
        # Add gated transport updates if available
        if transport_updates is not None and self.transport_gate_log is not None:
            tgate = torch.exp(self.transport_gate_log)
            updated_embeddings = updated_embeddings + tgate * transport_updates
        
        return updated_embeddings
