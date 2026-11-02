# Backwards-compatible re-exports of loss utilities now housed under src.losses
from gaze._reference.losses import (
    MaskedMSELoss,
    MaskedMetaboliteLoss,
    PhysicsLoss,
    EnhancedPhysicsLoss,
    WeightDecayLoss,
    SparsityRegularization,
    parse_gene_rule,
    ContrastivePretrainingLoss,
)
from .similarity_regularization import ChemicalSimilarityRegularization, InfoNCELoss

__all__ = [
    'MaskedMSELoss',
    'MaskedMetaboliteLoss',
    'PhysicsLoss',
    'EnhancedPhysicsLoss',
    'WeightDecayLoss',
    'SparsityRegularization',
    'parse_gene_rule',
    'ContrastivePretrainingLoss',
    'ChemicalSimilarityRegularization',
    'InfoNCELoss',
]
