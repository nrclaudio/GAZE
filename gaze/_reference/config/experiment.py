"""Experiment configuration schemas and loaders."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
import json

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    yaml = None


def _with_aliases(raw: Optional[Dict[str, Any]], alias_map: Dict[str, str]) -> Dict[str, Any]:
    """Return a copy of *raw* with legacy aliases mapped to canonical keys."""
    data = dict(raw or {})
    for alias, canonical in alias_map.items():
        if alias in data and canonical not in data:
            data[canonical] = data.pop(alias)
    return data


@dataclass
class PhysicsConfig:
    """Weights for physics-inspired regularisation terms."""

    lambda_balance: float = 0.0
    lambda_correlation: float = 0.0
    lambda_variance: float = 0.0
    lambda_kl: float = 0.0
    use_listwise_ranking: bool = False
    enzyme_confidence: float = 3.0
    flux_flexibility: float = 0.8

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> "PhysicsConfig":
        data = _with_aliases(
            raw,
            {
                "λ_bal": "lambda_balance",
                "λ_corr": "lambda_correlation",
                "λ_var": "lambda_variance",
                "λ_kl": "lambda_kl",
            },
        )
        # Drop removed keys that may linger in old configs
        data.pop("lambda_detection", None)
        data.pop("λ_det", None)
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lambda_balance": self.lambda_balance,
            "lambda_correlation": self.lambda_correlation,
            "lambda_variance": self.lambda_variance,
            "lambda_kl": self.lambda_kl,
            "use_listwise_ranking": self.use_listwise_ranking,
            "enzyme_confidence": self.enzyme_confidence,
            "flux_flexibility": self.flux_flexibility,
        }


@dataclass
class ContrastiveConfig:
    """Configuration for contrastive learning auxiliary objective."""

    enabled: bool = False
    weight: float = 0.1
    temperature: float = 0.07
    pretrained_encoders_path: Optional[str] = None

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> "ContrastiveConfig":
        data = _with_aliases(
            raw,
            {
                "contrastive_weight": "weight",
                "contrastive_temperature": "temperature",
                "pretrained_encoders": "pretrained_encoders_path",
                "enable_contrastive": "enabled",
            },
        )
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "weight": self.weight,
            "temperature": self.temperature,
            "pretrained_encoders_path": self.pretrained_encoders_path,
        }


@dataclass
class ModelConfig:
    """Hyperparameters for model instantiation."""

    variant: str = "concat"  # "concat" or "enhanced"
    hidden_dim: Optional[int] = None
    num_layers: Optional[int] = None
    dropout: Optional[float] = None
    use_directionality: bool = False
    directional_concat: bool = False
    use_flux_messages: bool = False
    flux_message_depth: int = 1
    use_mass_balance_projection: bool = False
    subtract_sample_mean: bool = False # New Flag
    enhanced_config: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> "ModelConfig":
        data = _with_aliases(
            raw,
            {
                "use_flux_messages": "use_flux_messages",
                "flux_message_depth": "flux_message_depth",
                "use_mass_balance_projection": "use_mass_balance_projection",
                "subtract_sample_mean": "subtract_sample_mean",
            },
        )
        # Drop legacy use_spatial key if present in configs
        data.pop("use_spatial", None)
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        enhanced_config = self.enhanced_config
        if hasattr(enhanced_config, 'to_dict'):
            enhanced_config = enhanced_config.to_dict()
        return {
            "variant": self.variant,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "use_directionality": self.use_directionality,
            "directional_concat": self.directional_concat,
            "use_flux_messages": self.use_flux_messages,
            "flux_message_depth": self.flux_message_depth,
            "use_mass_balance_projection": self.use_mass_balance_projection,
            "subtract_sample_mean": self.subtract_sample_mean,
            "enhanced_config": enhanced_config,
        }


@dataclass
class DataConfig:
    """Dataset and loader configuration."""

    normalized: bool = False
    use_gene_features: bool = False
    mudata_path: Optional[str] = None
    graph_path: Optional[str] = None
    gene_mapping_path: Optional[str] = None
    split_manifest_path: Optional[str] = None
    mudata_sha256: Optional[str] = None
    split_manifest_sha256: Optional[str] = None
    excluded_target_nodes: list[int] = field(default_factory=list)
    batch_size: int = 1
    num_workers: int = 4
    pin_memory: bool = True
    gradient_accumulation_steps: int = 1
    train_fraction: float = 0.7
    val_fraction: float = 0.15
    random_graph: bool = False
    graph_identifier: str = "" # e.g. 'masked'
    randomize_smiles: bool = False
    randomize_genes: bool = False
    randomize_embeddings: bool = False
    randomize_gpe: bool = False

    @property
    def test_fraction(self) -> float:
        return max(0.0, 1.0 - self.train_fraction - self.val_fraction)

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> "DataConfig":
        data = _with_aliases(
            raw,
            {
                "use_normalised": "normalized",
                "mudata_path": "mudata_path",
                "graph_path": "graph_path",
                "gene_mapping_path": "gene_mapping_path",
                "batch_size": "batch_size",
                "gradient_accumulation_steps": "gradient_accumulation_steps",
            },
        )
        # Drop removed keys that may linger in old configs
        for key in ("use_spatial", "spatial", "spatial_data_path"):
            data.pop(key, None)
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "normalized": self.normalized,
            "use_gene_features": self.use_gene_features,
            "mudata_path": self.mudata_path,
            "graph_path": self.graph_path,
            "gene_mapping_path": self.gene_mapping_path,
            "split_manifest_path": self.split_manifest_path,
            "mudata_sha256": self.mudata_sha256,
            "split_manifest_sha256": self.split_manifest_sha256,
            "excluded_target_nodes": self.excluded_target_nodes,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "train_fraction": self.train_fraction,
            "val_fraction": self.val_fraction,
            "random_graph": self.random_graph,
            "graph_identifier": self.graph_identifier,
            "randomize_smiles": self.randomize_smiles,
            "randomize_genes": self.randomize_genes,
            "randomize_embeddings": self.randomize_embeddings,
            "randomize_gpe": self.randomize_gpe,
        }


@dataclass
class TrainingConfig:
    """Optimisation and runtime parameters."""

    learning_rate: float = 7.5e-4
    max_epochs: Optional[int] = None
    patience: Optional[int] = 20
    save_dir: Optional[str] = None
    enable_metabolite_dropout: bool = False
    mask_target_smiles: bool = False

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> "TrainingConfig":
        data = _with_aliases(
            raw,
            {
                "learning_rate": "learning_rate",
                "max_epochs": "max_epochs",
                "enable_metabolite_dropout": "enable_metabolite_dropout",
                "mask_target_smiles": "mask_target_smiles",
            },
        )
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "save_dir": self.save_dir,
            "enable_metabolite_dropout": self.enable_metabolite_dropout,
            "mask_target_smiles": self.mask_target_smiles,
        }


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""

    seed: int = 42
    config_name: Optional[str] = None
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    contrastive: ContrastiveConfig = field(default_factory=ContrastiveConfig)

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> "ExperimentConfig":
        data = dict(raw or {})
        data.setdefault("model", {})
        data.setdefault("data", {})
        data.setdefault("training", {})
        data.setdefault("physics", {})
        data.setdefault("contrastive", {})

        data["model"] = ModelConfig.from_dict(data["model"])
        data["data"] = DataConfig.from_dict(data["data"])
        data["training"] = TrainingConfig.from_dict(data["training"])
        data["physics"] = PhysicsConfig.from_dict(data["physics"])
        data["contrastive"] = ContrastiveConfig.from_dict(data["contrastive"])

        # Drop removed keys that may linger in old configs
        for key in ("pretrained_model", "pretrained_model_path", "lambda_distill",
                     "distill", "distillation_weight"):
            data.pop(key, None)
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "config_name": self.config_name,
            "model": self.model.to_dict(),
            "data": self.data.to_dict(),
            "training": self.training.to_dict(),
            "physics": self.physics.to_dict(),
            "contrastive": self.contrastive.to_dict(),
        }


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment configuration from JSON or YAML."""

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    ext = config_path.suffix.lower()
    if ext in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to load YAML configuration files")
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    elif ext == ".json":
        with config_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    else:
        raise ValueError(f"Unsupported config extension '{ext}'. Use .json or .yaml/.yml")

    return ExperimentConfig.from_dict(raw)
