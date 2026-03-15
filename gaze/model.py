"""GAZEModel: inference model for zero-shot metabolite prediction.

Implements the full GAZE architecture:
  - Relational bipartite convolutions (GATv2)
  - Residual connections + LayerNorm
  - Bidirectional cross-modal attention
  - Metabolite-Conditioned Reader (MCR)
  - Flux messages + mass balance projection
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import HeteroConv, GATv2Conv

from .config import GAZEConfig
from .modules.cross_modal_attention import CrossModalAttention
from .modules.flux_messages import AdditiveFluxMessageLayer
from .modules.mass_balance import MassBalanceProjector
from .losses.structural_consistency import NeighborReconstructionLoss
from .components.met_reader import (
    MetaboliteConditionedReader,
    precompute_neighbor_indices,
    extract_reaction_neighbors_cached,
)
from .utils.batch_utils import standardize_batch_data, get_batch_size, create_default_features
from .utils.tensor_utils import check_for_nans


# ---------------------------------------------------------------------------
# RelationalMetabolicConv (inline to avoid extra file)
# ---------------------------------------------------------------------------

class RelationalMetabolicConv(nn.Module):
    """Bipartite-aware relational convolution for metabolic networks.

    Separate GATv2 weights for rxn->met and met->rxn, explicit self-loop
    processing, and gated fusion with residual + LayerNorm.
    """

    def __init__(self, hidden_dim: int, heads: int = 4, dropout: float = 0.2,
                 edge_dim: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.hetero_conv = HeteroConv({
            ("reaction", "to", "metabolite"): GATv2Conv(
                hidden_dim, hidden_dim, edge_dim=edge_dim, heads=heads,
                concat=False, dropout=dropout, add_self_loops=False,
            ),
            ("metabolite", "to", "reaction"): GATv2Conv(
                hidden_dim, hidden_dim, edge_dim=edge_dim, heads=heads,
                concat=False, dropout=dropout, add_self_loops=False,
            ),
        }, aggr="sum")

        self.self_reaction = nn.Linear(hidden_dim, hidden_dim)
        self.self_metabolite = nn.Linear(hidden_dim, hidden_dim)

        self.fusion_reaction = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fusion_metabolite = nn.Linear(2 * hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.norm_reaction = nn.LayerNorm(hidden_dim)
        self.norm_metabolite = nn.LayerNorm(hidden_dim)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        out_dict = self.hetero_conv(x_dict, edge_index_dict, edge_attr_dict)

        self_r = self.self_reaction(x_dict["reaction"])
        self_m = self.self_metabolite(x_dict["metabolite"])

        msg_r = out_dict.get("reaction", torch.zeros_like(self_r))
        combined_r = torch.cat([self_r, msg_r], dim=1)
        h_r = self.fusion_reaction(combined_r)
        h_r = self.norm_reaction(h_r + x_dict["reaction"])
        h_r = self.dropout(torch.relu(h_r))

        msg_m = out_dict.get("metabolite", torch.zeros_like(self_m))
        combined_m = torch.cat([self_m, msg_m], dim=1)
        h_m = self.fusion_metabolite(combined_m)
        h_m = self.norm_metabolite(h_m + x_dict["metabolite"])
        h_m = self.dropout(torch.relu(h_m))

        return {"reaction": h_r, "metabolite": h_m}


# ---------------------------------------------------------------------------
# BipartiteGraphPreparer (inline)
# ---------------------------------------------------------------------------

class BipartiteGraphPreparer:
    """Split homogeneous graph data into bipartite edge dicts for HeteroConv."""

    @staticmethod
    def prepare_bipartite_edges(edge_index, edge_attr, node_types, num_nodes):
        node_types_arr = np.array(node_types)
        src, dst = edge_index
        device = edge_index.device

        is_rxn = torch.tensor([t == "reaction" for t in node_types_arr], device=device)
        is_met = torch.tensor([t == "metabolite" for t in node_types_arr], device=device)

        rxn_indices = torch.where(is_rxn)[0]
        met_indices = torch.where(is_met)[0]

        rxn_map = torch.full((num_nodes,), -1, device=device, dtype=torch.long)
        rxn_map[rxn_indices] = torch.arange(len(rxn_indices), device=device)

        met_map = torch.full((num_nodes,), -1, device=device, dtype=torch.long)
        met_map[met_indices] = torch.arange(len(met_indices), device=device)

        r2m_mask = is_rxn[src] & is_met[dst]
        r2m_idx = torch.stack([rxn_map[src[r2m_mask]], met_map[dst[r2m_mask]]])
        r2m_attr = edge_attr[r2m_mask] if edge_attr is not None else None

        m2r_mask = is_met[src] & is_rxn[dst]
        m2r_idx = torch.stack([met_map[src[m2r_mask]], rxn_map[dst[m2r_mask]]])
        m2r_attr = edge_attr[m2r_mask] if edge_attr is not None else None

        edge_index_dict = {
            ("reaction", "to", "metabolite"): r2m_idx,
            ("metabolite", "to", "reaction"): m2r_idx,
        }
        edge_attr_dict = {
            ("reaction", "to", "metabolite"): r2m_attr,
            ("metabolite", "to", "reaction"): m2r_attr,
        }
        return edge_index_dict, edge_attr_dict, rxn_indices, met_indices


# ---------------------------------------------------------------------------
# GAZEModel
# ---------------------------------------------------------------------------

class GAZEModel(nn.Module):
    """GAZE deploy model for metabolite level prediction from gene expression.

    All deploy features are always enabled:
      - Relational bipartite GNN (``convs``)
      - Bidirectional cross-modal attention (``attention_layers``)
      - MCR readout head (``metabolite_reader``)
      - Flux messages + mass balance projection
      - Structural reconstruction auxiliary loss

    Args:
        config: :class:`GAZEConfig` hyperparameters.
        embedding_dim: Dimension of reaction node embeddings (e.g. 200).
        gene_dim: Dimension of gene expression features (e.g. 1).
        smiles_dim: Dimension of SMILES embeddings (e.g. 768).
        graph_data: Optional PyG Data object with graph structure for
            pre-computing bipartite edges and neighbor indices.
    """

    def __init__(
        self,
        config: GAZEConfig,
        embedding_dim: int,
        gene_dim: int,
        smiles_dim: int,
        graph_data: Optional[Data] = None,
    ):
        super().__init__()
        self.config = config
        self.embedding_dim = embedding_dim
        self.gene_dim = gene_dim
        self.smiles_dim = smiles_dim
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers

        combined_dim = embedding_dim + gene_dim

        # --- Encoders ---
        self.reaction_encoder = nn.Linear(combined_dim, config.hidden_dim)
        self.metabolite_encoder = nn.Linear(smiles_dim, config.hidden_dim)

        # --- Cross-modal attention layers ---
        # Layer 0: raw features -> hidden (reaction_dim=combined_dim, met_dim=smiles_dim)
        # Layers 1..num_layers: hidden -> hidden (after each GNN layer)
        self.attention_layers = nn.ModuleList()
        self.attention_layers.append(CrossModalAttention(
            reaction_dim=combined_dim,
            metabolite_dim=smiles_dim,
            hidden_dim=config.hidden_dim,
            num_heads=config.cross_modal_heads,
            dropout=config.cross_modal_dropout,
            temperature=config.cross_modal_temperature,
        ))
        for _ in range(config.num_layers):
            self.attention_layers.append(CrossModalAttention(
                reaction_dim=config.hidden_dim,
                metabolite_dim=config.hidden_dim,
                hidden_dim=config.hidden_dim,
                num_heads=config.cross_modal_heads,
                dropout=config.cross_modal_dropout,
                temperature=config.cross_modal_temperature,
            ))

        # Backward compatibility alias
        self.cross_modal_attention = self.attention_layers[0]

        # --- GNN layers (Relational) ---
        self.convs = nn.ModuleList([
            RelationalMetabolicConv(
                hidden_dim=config.hidden_dim,
                heads=config.gat_heads,
                dropout=config.dropout,
                edge_dim=config.gat_edge_dim,
            )
            for _ in range(config.num_layers)
        ])

        # --- Flux head ---
        self.reaction_flux_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Softplus(),
        )

        # --- MCR reader ---
        self.metabolite_reader = MetaboliteConditionedReader(
            smiles_dim=smiles_dim,
            gnn_hidden_dim=config.hidden_dim,
            hidden_dim=config.reader_hidden_dim,
            mlp_hidden=config.reader_mlp_hidden,
            normalize_descriptors=config.reader_normalize_descriptors,
            dropout=config.dropout,
        )
        # Not using MLP predictor -- MCR only
        self.metabolite_predictor = None

        # --- Flux message layer ---
        self.flux_message_layer = AdditiveFluxMessageLayer(
            hidden_dim=config.hidden_dim,
            compress_alpha=config.flux_compress_alpha,
            use_transport=config.flux_use_transport,
            transport_hops=config.flux_transport_hops,
        )

        # --- Mass balance projector (lazy-init from stoichiometric matrix) ---
        self.mass_balance_projector: Optional[MassBalanceProjector] = None

        # --- Structural reconstruction loss ---
        self.structural_consistency_loss = NeighborReconstructionLoss(
            hidden_dim=config.hidden_dim,
            smiles_dim=smiles_dim,
        )
        self.structural_loss_weight = config.structural_loss_weight

        # --- Dropout layer ---
        self.dropout = nn.Dropout(config.dropout)

        # --- Runtime caches ---
        self.neighbor_cache: Optional[Dict] = None
        self._bipartite_cache: Optional[Dict] = None

        # Store graph data for prediction convenience
        self._graph_data = graph_data

    # ------------------------------------------------------------------
    # Mass balance lazy init
    # ------------------------------------------------------------------

    def initialize_mass_balance_projector(self, data_or_matrix) -> None:
        """Lazily initialize the mass balance projector.

        Accepts either a ``Data`` object with ``incidence_matrix`` attribute,
        or a raw stoichiometric matrix tensor.
        """
        if self.mass_balance_projector is not None:
            return

        if isinstance(data_or_matrix, torch.Tensor):
            matrix = data_or_matrix
        elif hasattr(data_or_matrix, "incidence_matrix"):
            matrix = data_or_matrix.incidence_matrix
        elif hasattr(data_or_matrix, "dataset") and hasattr(data_or_matrix.dataset, "incidence_matrix"):
            matrix = data_or_matrix.dataset.incidence_matrix
        else:
            return

        self.mass_balance_projector = MassBalanceProjector(
            stoichiometric_matrix=matrix,
            max_regularization=self.config.mass_balance_max_reg,
        )

    # ------------------------------------------------------------------
    # Cross-modal mask
    # ------------------------------------------------------------------

    def _create_cross_modal_mask(self, edge_index, metabolite_indices,
                                  reaction_indices, device):
        """Create boolean mask ``[1, M, R]`` for cross-modal attention."""
        num_m = len(metabolite_indices)
        num_r = len(reaction_indices)
        mask = torch.zeros(1, num_m, num_r, dtype=torch.bool, device=device)

        if self.neighbor_cache is not None:
            # Build global->local reaction index map
            max_idx = reaction_indices.max().item()
            rxn_g2l = torch.full((max_idx + 1,), -1, device=device, dtype=torch.long)
            rxn_g2l[reaction_indices] = torch.arange(num_r, device=device)

            met_cpu = metabolite_indices.cpu().numpy()
            for local_m, global_m in enumerate(met_cpu):
                global_m = int(global_m)
                if global_m in self.neighbor_cache:
                    nbrs = self.neighbor_cache[global_m]
                    if len(nbrs) > 0:
                        local_nbrs = rxn_g2l[nbrs]
                        valid = local_nbrs >= 0
                        mask[:, local_m, local_nbrs[valid]] = True
        else:
            # Fallback: build mask from edge_index
            src, dst = edge_index
            met_set = set(metabolite_indices.cpu().numpy().tolist())
            rxn_set = set(reaction_indices.cpu().numpy().tolist())

            met_g2l = {int(v): i for i, v in enumerate(metabolite_indices.cpu().numpy())}
            rxn_g2l_dict = {int(v): i for i, v in enumerate(reaction_indices.cpu().numpy())}

            for i in range(src.size(0)):
                s, d = int(src[i]), int(dst[i])
                if s in met_set and d in rxn_set:
                    mask[0, met_g2l[s], rxn_g2l_dict[d]] = True
                elif s in rxn_set and d in met_set:
                    mask[0, met_g2l[d], rxn_g2l_dict[s]] = True

        return mask

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, data: Data, return_embeddings: bool = False,
                compute_physics: bool = True) -> tuple:
        """Forward pass.

        Args:
            data: PyG Data object with ``x_embedding``, ``x_gene``, ``x_smiles``,
                ``edge_index``, ``edge_attr``, ``reaction_indices``,
                ``metabolite_indices``, ``node_types``.
            return_embeddings: Return GNN embeddings as fourth output.
            compute_physics: Compute flux / mass-balance outputs.

        Returns:
            pred: ``[B, N]`` metabolite predictions (zero for non-metabolite nodes).
            delta: ``[B, M]`` mass balance residuals.
            flux_vec: ``[B, R]`` reaction fluxes.
            embeddings: ``[B, N, D]`` or None.
        """
        data = standardize_batch_data(data)
        batch_size = get_batch_size(data)

        x_embedding = getattr(data, "x_embedding", data.x)
        num_nodes = x_embedding.size(1)

        x_gene = getattr(data, "x_gene", None)
        if x_gene is None:
            x_gene = create_default_features(num_nodes, self.gene_dim, batch_size, x_embedding.device)

        x_smiles = getattr(data, "x_smiles", None)
        if x_smiles is None:
            x_smiles = create_default_features(num_nodes, self.smiles_dim, batch_size, x_embedding.device)

        edge_index = data.edge_index
        edge_attr = data.edge_attr
        reaction_indices = data.reaction_indices
        metabolite_indices = data.metabolite_indices

        # Process each sample individually (matching training model)
        batch_pred, batch_delta, batch_flux, batch_emb = [], [], [], []

        for b in range(batch_size):
            pred, delta, flux, emb = self._forward_single(
                x_embedding[b], x_gene[b], x_smiles[b],
                edge_index, edge_attr,
                reaction_indices, metabolite_indices,
                data, return_embeddings, compute_physics,
            )
            batch_pred.append(pred)
            batch_delta.append(delta)
            batch_flux.append(flux)
            if return_embeddings:
                batch_emb.append(emb)

        pred = torch.stack(batch_pred)
        delta = torch.stack(batch_delta)
        flux = torch.stack(batch_flux)
        emb = torch.stack(batch_emb) if return_embeddings else None

        return pred, delta, flux, emb

    def _forward_single(self, x_embedding, x_gene, x_smiles,
                         edge_index, edge_attr,
                         reaction_indices, metabolite_indices,
                         data, return_embeddings, compute_physics):
        """Process a single sample through the full pipeline."""
        device = x_embedding.device
        num_nodes = x_embedding.size(0)

        # --- Step 1: Encode ---
        reaction_features = torch.cat([
            x_embedding[reaction_indices],
            x_gene[reaction_indices],
        ], dim=1)
        encoded_reactions = self.reaction_encoder(reaction_features)
        encoded_metabolites = self.metabolite_encoder(x_smiles[metabolite_indices])

        # --- Step 2: Pre-GNN cross-modal attention ---
        cross_modal_mask = self._create_cross_modal_mask(
            edge_index, metabolite_indices, reaction_indices, device,
        )
        raw_rxn_features = reaction_features  # combined_dim
        raw_met_features = x_smiles[metabolite_indices]  # smiles_dim

        enhanced_rxn, enhanced_met = self.attention_layers[0](
            raw_rxn_features, raw_met_features,
            reaction_indices, metabolite_indices,
            mask=cross_modal_mask,
        )
        w = self.config.cross_modal_residual_weight
        encoded_reactions = encoded_reactions + w * enhanced_rxn
        encoded_metabolites = encoded_metabolites + w * enhanced_met

        # Populate hidden state
        h = torch.zeros(num_nodes, self.hidden_dim, device=device)
        h[reaction_indices] = encoded_reactions
        h[metabolite_indices] = encoded_metabolites

        # Lazy-init mass balance projector
        if self.mass_balance_projector is None and hasattr(data, "incidence_matrix"):
            self.initialize_mass_balance_projector(data)

        # --- Step 3: Prepare bipartite edges ---
        if (self._bipartite_cache is None
                or self._bipartite_cache.get("edge_ptr") != edge_index.data_ptr()
                or self._bipartite_cache.get("num_nodes") != num_nodes):
            eid, ead, rxn_idx, met_idx = BipartiteGraphPreparer.prepare_bipartite_edges(
                edge_index, edge_attr, data.node_types, num_nodes,
            )
            self._bipartite_cache = {
                "edge_ptr": edge_index.data_ptr(),
                "num_nodes": num_nodes,
                "edge_index_dict": eid,
                "edge_attr_dict": ead,
                "rxn_idx": rxn_idx,
                "met_idx": met_idx,
            }

        edge_index_dict = self._bipartite_cache["edge_index_dict"]
        edge_attr_dict = self._bipartite_cache["edge_attr_dict"]
        rxn_idx = self._bipartite_cache["rxn_idx"]
        met_idx = self._bipartite_cache["met_idx"]

        # --- Step 4: GNN layers with cross-modal attention + flux messages ---
        for i, conv in enumerate(self.convs):
            x_dict = {"reaction": h[rxn_idx], "metabolite": h[met_idx]}
            out_dict = conv(x_dict, edge_index_dict, edge_attr_dict)

            h_new = torch.zeros_like(h)
            h_new[rxn_idx] = out_dict["reaction"]
            h_new[met_idx] = out_dict["metabolite"]
            h = h_new

            # Recursive cross-modal alignment after each GNN layer
            if i + 1 < len(self.attention_layers):
                cur_rxn = h[reaction_indices]
                cur_met = h[metabolite_indices]
                aligned_rxn, aligned_met = self.attention_layers[i + 1](
                    cur_rxn, cur_met,
                    reaction_indices, metabolite_indices,
                    mask=cross_modal_mask,
                )
                h[reaction_indices] = cur_rxn + w * aligned_rxn
                h[metabolite_indices] = cur_met + w * aligned_met

            # Flux messages at specified depth
            if (compute_physics
                    and i + 1 == self.config.flux_message_depth
                    and hasattr(data, "incidence_matrix")):
                rxn_emb = h[reaction_indices]
                flux_pred = self.reaction_flux_head(rxn_emb).squeeze(-1)

                if self.mass_balance_projector is not None:
                    flux_bal = self.mass_balance_projector.project_to_mass_balance(
                        flux_pred, data.incidence_matrix,
                    )
                else:
                    flux_bal = flux_pred

                met_emb = h[metabolite_indices]
                met_updated = self.flux_message_layer(
                    metabolite_embeddings=met_emb.unsqueeze(0),
                    reaction_embeddings=rxn_emb.unsqueeze(0),
                    fluxes=flux_bal.unsqueeze(0),
                    stoichiometric_matrix=data.incidence_matrix,
                    metabolite_indices=metabolite_indices,
                    reaction_indices=reaction_indices,
                ).squeeze(0)
                h[metabolite_indices] = met_updated

        # --- Step 5: MCR prediction ---
        metabolite_pred = self._predict_with_mcr(
            h, metabolite_indices, reaction_indices, edge_index, x_smiles,
        )

        pred = torch.zeros(num_nodes, device=device)
        pred[metabolite_indices] = metabolite_pred

        # --- Step 6: Physics outputs ---
        if compute_physics:
            rxn_emb = h[reaction_indices]
            flux_raw = self.reaction_flux_head(rxn_emb).squeeze(-1)

            if self.mass_balance_projector is not None and hasattr(data, "incidence_matrix"):
                flux_vec = self.mass_balance_projector.project_to_mass_balance(
                    flux_raw, data.incidence_matrix,
                )
            else:
                flux_vec = flux_raw

            if hasattr(data, "incidence_matrix"):
                inc = data.incidence_matrix.to(flux_vec.device)
                delta = torch.sparse.mm(inc, flux_vec.unsqueeze(1)).squeeze()
            else:
                delta = torch.zeros(len(metabolite_indices), device=device)
        else:
            flux_vec = torch.zeros(len(reaction_indices), device=device)
            delta = torch.zeros(len(metabolite_indices), device=device)

        # --- Structural loss (training only) ---
        structural_loss = torch.tensor(0.0, device=device)
        if self.structural_consistency_loss is not None and self.training:
            src, dst = edge_index
            is_rxn = torch.zeros(num_nodes, dtype=torch.bool, device=device)
            is_rxn[reaction_indices] = True
            rxn_mask = is_rxn[src]
            rxn_src = src[rxn_mask]
            met_dst = dst[rxn_mask]

            from torch_geometric.utils import scatter
            nb_msgs = scatter(h[rxn_src], met_dst, dim=0, dim_size=num_nodes, reduce="mean")
            met_nb = nb_msgs[metabolite_indices]
            target_smiles = x_smiles[metabolite_indices]
            structural_loss = (
                self.structural_consistency_loss(met_nb, target_smiles)
                * self.structural_loss_weight
            )
        self.last_structural_loss = structural_loss

        embeddings = h if return_embeddings else None
        return pred, delta, flux_vec, embeddings

    # ------------------------------------------------------------------
    # MCR helper
    # ------------------------------------------------------------------

    def _predict_with_mcr(self, h, metabolite_indices, reaction_indices,
                           edge_index, x_smiles):
        """Predict metabolite concentrations with MCR reader."""
        x_smiles_2d = x_smiles.squeeze(0) if x_smiles.dim() == 3 and x_smiles.size(0) == 1 else x_smiles
        metabolite_smiles = x_smiles_2d[metabolite_indices]

        # Lazy-init neighbor cache
        if self.neighbor_cache is None:
            self.neighbor_cache = precompute_neighbor_indices(
                edge_index=edge_index,
                metabolite_indices=metabolite_indices,
                reaction_indices=reaction_indices,
                max_neighbors=self.config.reader_max_neighbors,
                two_hop=False,
            )

        h_3d = h.unsqueeze(0) if h.dim() == 2 else h
        neighbor_emb, neighbor_mask = extract_reaction_neighbors_cached(
            neighbor_cache=self.neighbor_cache,
            metabolite_indices=metabolite_indices,
            reaction_embeddings=h_3d,
            max_neighbors=self.config.reader_max_neighbors,
        )

        # Use optimized single-sample path
        return self.metabolite_reader.forward_single_optimized(
            metabolite_smiles=metabolite_smiles,
            reaction_neighbors=neighbor_emb.squeeze(0),
            neighbor_mask=neighbor_mask.squeeze(0),
        )

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def clear_caches(self):
        """Clear all runtime caches (neighbor indices, bipartite edges)."""
        self.neighbor_cache = None
        self._bipartite_cache = None

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        graph_path: Optional[Union[str, Path]] = None,
        gene_mapping_path: Optional[Union[str, Path]] = None,
        config: Optional[GAZEConfig] = None,
        device: Optional[torch.device] = None,
    ) -> "GAZEModel":
        """Load a trained GAZEModel from a checkpoint file.

        Loads weights with ``strict=False`` so that extra keys from the
        training model (detection gate, similarity regularizer, etc.)
        are silently ignored.

        Args:
            checkpoint_path: Path to ``*.pt`` checkpoint.
            graph_path: Path to ``graph.pt`` file with pre-saved graph data.
            gene_mapping_path: Path to ``gene_mapping.json`` for gene index mapping.
            config: Optional :class:`GAZEConfig`. If *None*, the config is
                inferred from the checkpoint's ``model_state_dict`` shapes.
            device: Device to load the model onto.

        Returns:
            Loaded :class:`GAZEModel` in eval mode.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint_path = Path(checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Support both raw state_dict and wrapped checkpoint
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif isinstance(ckpt, dict) and any(k.startswith("reaction_encoder") for k in ckpt):
            state_dict = ckpt
        else:
            state_dict = ckpt

        # Infer dimensions from state_dict
        # nn.Linear(in, out) weight shape is (out, in)
        hidden_dim, combined_dim = state_dict["reaction_encoder.weight"].shape
        smiles_dim = state_dict["metabolite_encoder.weight"].shape[1]

        # Infer gene_dim from combined_dim
        # gene_dim = combined_dim - embedding_dim, but we need embedding_dim.
        # We'll try to infer from stored config or fallback.
        if config is None:
            config = GAZEConfig(hidden_dim=hidden_dim)

        # Count GNN layers from convs keys
        conv_keys = [k for k in state_dict if k.startswith("convs.")]
        if conv_keys:
            max_layer = max(int(k.split(".")[1]) for k in conv_keys)
            config.num_layers = max_layer + 1

        # Try to infer embedding_dim from checkpoint metadata
        stored_config = None
        if isinstance(ckpt, dict):
            stored_config = ckpt.get("config", None)

        if stored_config and hasattr(stored_config, "embedding_dim"):
            embedding_dim = stored_config.embedding_dim
        elif stored_config and isinstance(stored_config, dict) and "embedding_dim" in stored_config:
            embedding_dim = stored_config["embedding_dim"]
        else:
            # Heuristic: metabolite_encoder input is smiles_dim.
            # For reaction_encoder: input is embedding_dim + gene_dim.
            # Common defaults: gene_dim=1 for single expression value.
            embedding_dim = combined_dim - 1  # assume gene_dim=1

        gene_dim = combined_dim - embedding_dim

        # Load graph data if provided
        graph_data = None
        if graph_path is not None:
            graph_data = torch.load(Path(graph_path), map_location=device, weights_only=False)

        # Build model
        model = cls(
            config=config,
            embedding_dim=embedding_dim,
            gene_dim=gene_dim,
            smiles_dim=smiles_dim,
            graph_data=graph_data,
        )

        # Load weights (strict=False to ignore removed features)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if unexpected:
            print(f"Ignored {len(unexpected)} unexpected keys from checkpoint "
                  f"(e.g. {unexpected[:3]})")
        if missing:
            print(f"Missing {len(missing)} keys (will use fresh init): "
                  f"{missing[:5]}")

        model = model.to(device)
        model.eval()

        # Store gene mapping if provided
        if gene_mapping_path is not None:
            with open(gene_mapping_path) as f:
                model._gene_mapping = json.load(f)
        else:
            model._gene_mapping = None

        return model

    # ------------------------------------------------------------------
    # High-level inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        gene_expression: torch.Tensor,
        metabolite_indices: Optional[torch.Tensor] = None,
        graph_data: Optional[Data] = None,
    ) -> torch.Tensor:
        """High-level inference method.

        Args:
            gene_expression: Gene expression matrix ``[num_samples, num_genes]``
                or ``[num_genes]`` for a single sample.
            metabolite_indices: Optional subset of metabolite indices to return.
                If *None*, returns all metabolite predictions.
            graph_data: PyG Data object with graph structure. If *None*, uses
                the graph data stored at construction time.

        Returns:
            Predictions ``[num_samples, num_metabolites]`` or
            ``[num_metabolites]`` for a single sample.
        """
        self.eval()
        data = graph_data if graph_data is not None else self._graph_data
        if data is None:
            raise ValueError(
                "No graph data available. Pass graph_data to predict() or "
                "provide graph_path when loading the model."
            )

        is_single = gene_expression.dim() == 1
        if is_single:
            gene_expression = gene_expression.unsqueeze(0)

        device = next(self.parameters()).device
        gene_expression = gene_expression.to(device)

        num_samples = gene_expression.size(0)
        num_nodes = data.x_embedding.size(0) if hasattr(data, "x_embedding") else data.num_nodes

        all_preds = []
        for i in range(num_samples):
            # Build per-sample Data with gene expression injected
            sample_data = data.clone()

            # Inject gene expression into the graph
            if hasattr(sample_data, "x_gene"):
                reaction_idx = sample_data.reaction_indices
                sample_data.x_gene = sample_data.x_gene.clone()
                if sample_data.x_gene.dim() == 2:
                    sample_data.x_gene[reaction_idx] = gene_expression[i].unsqueeze(-1) \
                        if gene_expression[i].dim() == 1 and sample_data.x_gene.size(-1) == 1 \
                        else gene_expression[i]

            pred, _, _, _ = self.forward(sample_data, compute_physics=False)
            pred = pred.squeeze(0)  # remove batch dim

            if metabolite_indices is not None:
                pred = pred[metabolite_indices]
            else:
                met_idx = sample_data.metabolite_indices
                pred = pred[met_idx]

            all_preds.append(pred)

        result = torch.stack(all_preds)
        return result.squeeze(0) if is_single else result
