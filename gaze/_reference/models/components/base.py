"""Base class offering shared functionality for Concat-style models."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .builders import (
    build_encoders,
    build_flux_head,
    build_metabolite_head,
    build_flux_message_layer,
    MCRReaderConfig,
)
from ..modules.mass_balance import MassBalanceProjector


class BaseConcatModel(nn.Module):
    """Provides common encoders, heads, and utilities for Concat variants."""

    def __init__(
        self,
        *,
        embedding_dim: int,
        gene_dim: int,
        smiles_dim: int,
        hidden_dim: int,
        dropout: float,
        use_flux_messages: bool,
        flux_message_depth: int,
        use_mass_balance_projection: bool,
        use_mcr: bool,
        reader_config: Optional[MCRReaderConfig] = None,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.gene_dim = gene_dim
        self.smiles_dim = smiles_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.use_flux_messages = use_flux_messages
        self.flux_message_depth = flux_message_depth
        self.use_mass_balance_projection = use_mass_balance_projection
        self.use_mcr = use_mcr

        self.reaction_encoder, self.metabolite_encoder = build_encoders(
            embedding_dim=embedding_dim,
            gene_dim=gene_dim,
            smiles_dim=smiles_dim,
            hidden_dim=hidden_dim,
        )
        self.reaction_flux_head = build_flux_head(hidden_dim)
        self.metabolite_reader, self.metabolite_predictor = build_metabolite_head(
            use_mcr=use_mcr,
            smiles_dim=smiles_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            reader_config=reader_config,
        )

        if use_mcr:
            self.neighbor_cache = None

        # Detection gate (initialized by subclasses if enabled)
        self.detection_gate: Optional[nn.Module] = None
        self.last_detection_loss: Optional[torch.Tensor] = None
        self.last_detection_scores: Optional[torch.Tensor] = None

        self.mass_balance_projector: Optional[MassBalanceProjector] = None
        if self.use_flux_messages:
            self.flux_message_layer = build_flux_message_layer(hidden_dim)
        else:
            self.flux_message_layer = None

    # ------------------------------------------------------------------
    # Mass balance utilities
    # ------------------------------------------------------------------

    def initialize_mass_balance_projector(self, dataset_or_data) -> None:
        """Initialise the projector the first time stoichiometry is available."""
        if not self.use_mass_balance_projection or self.mass_balance_projector is not None:
            return

        if hasattr(dataset_or_data, 'incidence_matrix'):
            incidence_matrix = dataset_or_data.incidence_matrix
        elif (
            hasattr(dataset_or_data, 'dataset')
            and hasattr(dataset_or_data.dataset, 'incidence_matrix')
        ):
            incidence_matrix = dataset_or_data.dataset.incidence_matrix
        else:
            print("Warning: No incidence matrix found. Mass balance projection disabled.")
            self.use_mass_balance_projection = False
            return

        self.mass_balance_projector = MassBalanceProjector(
            stoichiometric_matrix=incidence_matrix,
            max_regularization=1e-2,
            use_float64=False,
        )
        print(
            "Mass balance projector initialized for "
            f"{incidence_matrix.size(0)} metabolites, {incidence_matrix.size(1)} reactions"
        )

    # ------------------------------------------------------------------
    # Detection gate utilities
    # ------------------------------------------------------------------

    def _apply_detection_gate(self, metabolite_emb, metabolite_pred):
        """Apply detection gate: returns ``(gated_pred, gate_logit)`` or ``(pred, None)``."""
        if self.detection_gate is None:
            return metabolite_pred, None
        gate_logit = self.detection_gate(metabolite_emb).squeeze(-1)
        gate_prob = torch.sigmoid(gate_logit)
        return gate_prob * metabolite_pred, gate_logit

    def _compute_detection_bce(self, gate_logit, mask, metabolite_indices,
                               measurable_mask=None):
        """BCE loss from gate logits and mask. Returns scalar or ``None``.

        When *measurable_mask* is provided, the loss is restricted to
        CAMP-measurable metabolites so that zero-shot targets (never in
        CAMP) do not bias the gate.
        """
        if gate_logit is None or mask is None or not self.training:
            return None
        if mask.dim() == 1:
            target = mask[metabolite_indices].float()
        else:
            target = mask[:, metabolite_indices].float()

        # Restrict to CAMP-measurable metabolites only
        if measurable_mask is not None:
            measurable = measurable_mask[metabolite_indices]
            if not measurable.any():
                return None
            if target.dim() == 1:
                return F.binary_cross_entropy_with_logits(
                    gate_logit[measurable], target[measurable])
            else:
                return F.binary_cross_entropy_with_logits(
                    gate_logit[:, measurable], target[:, measurable])

        return F.binary_cross_entropy_with_logits(gate_logit, target)

    # ------------------------------------------------------------------
    # Contrastive utilities (optional, implemented by subclasses)
    # ------------------------------------------------------------------

    def enable_contrastive_learning(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError

    def compute_contrastive_loss(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError
