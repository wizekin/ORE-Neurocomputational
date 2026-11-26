"""Knowledge-graph embeddings: native models, leakage-safe training, calibration (Phase B)."""
from .model_adapters import (
    MODEL_NAMES,
    ComplEx,
    DistMult,
    KGEModel,
    RelationBehaviorProfile,
    RotatE,
    TransE,
    create_model,
    profile_relation_behavior,
    sigmoid,
)
from .splitting import (
    SPLIT_POLICIES,
    FilteredNegativeSampler,
    TripleRecord,
    build_split_manifest,
)
from .trainer import (
    AdamOptimizer,
    EmbeddingScorer,
    EmbeddingTrainer,
    FilteredRankingEvaluator,
    IsotonicCalibrator,
    TemperatureCalibrator,
    TrainedEmbedding,
    TrainingReport,
)

__all__ = [
    "MODEL_NAMES", "SPLIT_POLICIES",
    "AdamOptimizer", "ComplEx", "DistMult", "EmbeddingScorer", "EmbeddingTrainer",
    "FilteredNegativeSampler", "FilteredRankingEvaluator", "IsotonicCalibrator", "KGEModel",
    "RelationBehaviorProfile", "RotatE", "TemperatureCalibrator", "TrainedEmbedding", "TransE",
    "TrainingReport", "TripleRecord", "build_split_manifest", "create_model",
    "profile_relation_behavior", "sigmoid",
]
