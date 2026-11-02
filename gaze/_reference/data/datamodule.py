"""Reusable data module abstractions for training pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .batching import collate_metabolic_data
from .dataset_smiles import FiLMDatasetWithSMILES
from ..utils.data_paths import get_standard_data_paths
from ..utils.graph_randomization import create_random_graph_version
from ..config import DataConfig
from .sample_splits import manifest_splits, verify_sha256


class DataModule:
    """Lightweight base class for dataset orchestration."""

    def setup(self, *args, **kwargs) -> None:  # pragma: no cover - interface definition
        raise NotImplementedError

    def train_dataloader(self) -> DataLoader:
        raise NotImplementedError

    def val_dataloader(self) -> DataLoader:
        raise NotImplementedError

    def test_dataloader(self) -> DataLoader:
        raise NotImplementedError


@dataclass
class DataModuleState:
    """Runtime metadata produced during data module setup."""

    embedding_dim: int
    gene_dim: int
    smiles_dim: int
    node_types: Sequence[str]


class FiLMDataModule(DataModule):
    """Data module that wraps FiLM + SMILES datasets with optional spatial support."""

    def __init__(
        self,
        config: DataConfig,
        seed: int,
        device: Optional[torch.device] = None,
    ) -> None:
        self.config = config
        self.seed = seed
        self.device = device

        self._dataset = None
        self._train_subset: Optional[Subset] = None
        self._val_subset: Optional[Subset] = None
        self._test_subset: Optional[Subset] = None
        self._state: Optional[DataModuleState] = None
        self._splits: Optional[dict] = None

    @property
    def state(self) -> DataModuleState:
        if self._state is None:
            raise RuntimeError("Data module has not been set up")
        return self._state

    @property
    def dataset(self):
        if self._dataset is None:
            raise RuntimeError("Data module has not been set up")
        return self._dataset

    def setup(self, **kwargs) -> DataModuleState:
        default_mudata, default_graph, default_gene_mapping = get_standard_data_paths(
            normalized=self.config.normalized,
            identifier=self.config.graph_identifier
        )

        mudata_path = self.config.mudata_path or default_mudata
        graph_path = self.config.graph_path or default_graph
        gene_mapping_path = self.config.gene_mapping_path or default_gene_mapping
        verify_sha256(mudata_path, self.config.mudata_sha256)
        if self.config.split_manifest_path:
            verify_sha256(self.config.split_manifest_path, self.config.split_manifest_sha256)

        dataset = FiLMDatasetWithSMILES(
            mudata_path=mudata_path,
            embeddings_graph_path=graph_path,
            gene_mapping_path=gene_mapping_path,
            device=self.device,
            randomize_smiles=self.config.randomize_smiles,
            randomize_genes=self.config.randomize_genes,
            randomize_embeddings=self.config.randomize_embeddings,
            random_graph=self.config.random_graph,
        )

        n_samples = len(dataset)
        dataset.excluded_target_nodes = set(self.config.excluded_target_nodes)
        if self.config.split_manifest_path:
            splits = manifest_splits(self.config.split_manifest_path,
                                     dataset.mudata['rna'].obs, dataset.mudata['metabolomics'].obs)
            train_indices, val_indices, test_indices = splits['train'], splits['val'], splits['test']
        else:
            train_indices, val_indices, test_indices = self._split_indices(n_samples)
            splits = {'train': train_indices, 'val': val_indices, 'test': test_indices}

        self._train_subset = Subset(dataset, train_indices)
        self._val_subset = Subset(dataset, val_indices)
        self._test_subset = Subset(dataset, test_indices)
        self._splits = splits

        if self.config.random_graph:
            self._train_subset = create_random_graph_version(self._train_subset, self.seed)
            self._val_subset = create_random_graph_version(self._val_subset, self.seed)
            self._test_subset = create_random_graph_version(self._test_subset, self.seed)

        self._dataset = dataset
        self._state = DataModuleState(
            embedding_dim=dataset.embedding_dim,
            gene_dim=dataset.gene_dim,
            smiles_dim=dataset.smiles_dim,
            node_types=dataset.node_types,
        )
        return self._state

    # ------------------------------------------------------------------
    # DataLoader helpers
    # ------------------------------------------------------------------

    def train_dataloader(self) -> DataLoader:
        if self._train_subset is None:
            raise RuntimeError("Data module not initialised. Call setup() first.")
        return DataLoader(
            self._train_subset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            collate_fn=collate_metabolic_data,
        )

    def val_dataloader(self) -> DataLoader:
        if self._val_subset is None:
            raise RuntimeError("Data module not initialised. Call setup() first.")
        return DataLoader(
            self._val_subset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            collate_fn=collate_metabolic_data,
        )

    def test_dataloader(self) -> DataLoader:
        if self._test_subset is None:
            raise RuntimeError("Data module not initialised. Call setup() first.")
        return DataLoader(
            self._test_subset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            collate_fn=collate_metabolic_data,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split_indices(self, n_samples: int) -> Tuple[List[int], List[int], List[int]]:
        indices = list(range(n_samples))
        rng = np.random.RandomState(self.seed)
        rng.shuffle(indices)

        train_count = int(self.config.train_fraction * n_samples)
        val_count = int(self.config.val_fraction * n_samples)
        test_count = n_samples - train_count - val_count

        if train_count < 0 or val_count < 0 or test_count < 0:
            raise ValueError(
                "Invalid dataset split configuration: fractions sum to more than 1"
            )

        train_indices = indices[:train_count]
        val_indices = indices[train_count : train_count + val_count]
        test_indices = indices[train_count + val_count : train_count + val_count + test_count]
        return train_indices, val_indices, test_indices

    @property
    def splits(self) -> dict:
        if self._splits is None:
            raise RuntimeError("Data module has not been set up")
        return self._splits
