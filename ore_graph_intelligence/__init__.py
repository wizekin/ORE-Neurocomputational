"""ORE Point 8 - Graph Neural Networks and Neural Algorithmic Reasoning (integrated unit).

A bounded USG Graph Intelligence Layer that projects authoritative USG concepts and
relations into typed graphs, learns structural representations, proposes candidate
relations, and sends every candidate through symbolic and evidence-based verification
before any authorized knowledge update - plus an isolated Neural Algorithmic
Reasoning research track compared against deterministic algorithms.

Core invariant: learned scores guide search; they do not establish truth. An unobserved
edge is not necessarily false. A high score is not a proof.
"""
from .discovery import (
    AbstentionDecision,
    AbstentionPolicy,
    CandidateGenerator,
    ComplementaryScorer,
    ExplanationNeighborhood,
    GraphCandidateService,
    LinkPredictor,
    NeighborhoodExplainer,
    NoveltyContext,
    ScoreNormalization,
    UtilityRanker,
)
from .embeddings import (
    MODEL_NAMES,
    SPLIT_POLICIES,
    EmbeddingScorer,
    EmbeddingTrainer,
    FilteredRankingEvaluator,
    TrainedEmbedding,
    build_split_manifest,
    create_model,
    profile_relation_behavior,
)
from .gnn import (
    DeterministicGraphBaselines,
    GNNInferenceEngine,
    OverSmoothingMonitor,
    OverSmoothingReport,
    RGCNLinkPredictor,
    RGCNTrainingReport,
    build_snapshot_features,
)
from .nar import (
    NAREvaluationReport,
    NAREvaluator,
    NARGateDecision,
    NARProductionGate,
    NARTrainer,
    NARUsePolicy,
    NarrowTask,
    NeuralAlgorithmicService,
    TaskGenerator,
    TrainedNARModel,
    reference_solve,
)
from .observability import (
    AuditTrail,
    CandidateLifecycleAudit,
    GraphEventRecorder,
    SemanticSafetyMetrics,
    expected_calibration_error,
    false_positive_rate,
    graph_prediction_metrics,
    hits_at_k,
    mean_reciprocal_rank,
    relation_type_subgroup_metrics,
)
from .orchestration import (
    GraphIntelligenceCoordinator,
    GraphIntelligenceResult,
    GraphIntelligenceTask,
    GraphResultCache,
    ModelRegistry,
)
from .projection import (
    IntegrityReport,
    IntegrityValidator,
    TypedGraphSnapshot,
    USGGraphProjector,
    compute_graph_hash,
)
from .schemas import (
    CandidateDisposition,
    EvidenceClassification,
    GraphCandidate,
    GraphQuery,
    GraphQueryAnchor,
    GraphSchema,
    ModelManifest,
    new_hypothesis_namespace,
)
from .verification import (
    AuthorizedUpdateProposal,
    CandidateVerifier,
    PromotionGate,
    VerificationResult,
)

__version__ = "1.0.0"

__all__ = [
    # discovery
    "AbstentionDecision", "AbstentionPolicy", "CandidateGenerator",
    "ComplementaryScorer", "ExplanationNeighborhood", "GraphCandidateService",
    "LinkPredictor", "NeighborhoodExplainer", "NoveltyContext", "ScoreNormalization",
    "UtilityRanker",
    # embeddings
    "MODEL_NAMES", "SPLIT_POLICIES", "EmbeddingScorer", "EmbeddingTrainer",
    "FilteredRankingEvaluator", "TrainedEmbedding", "build_split_manifest",
    "create_model", "profile_relation_behavior",
    # gnn
    "DeterministicGraphBaselines", "GNNInferenceEngine", "OverSmoothingMonitor",
    "OverSmoothingReport", "RGCNLinkPredictor", "RGCNTrainingReport",
    "build_snapshot_features",
    # nar
    "NAREvaluationReport", "NAREvaluator", "NARGateDecision", "NARProductionGate",
    "NARTrainer", "NARUsePolicy", "NarrowTask", "NeuralAlgorithmicService",
    "TaskGenerator", "TrainedNARModel", "reference_solve",
    # observability
    "AuditTrail", "CandidateLifecycleAudit", "GraphEventRecorder",
    "SemanticSafetyMetrics", "expected_calibration_error", "false_positive_rate",
    "graph_prediction_metrics", "hits_at_k", "mean_reciprocal_rank",
    "relation_type_subgroup_metrics",
    # orchestration
    "GraphIntelligenceCoordinator", "GraphIntelligenceResult",
    "GraphIntelligenceTask", "GraphResultCache", "ModelRegistry",
    # projection
    "IntegrityReport", "IntegrityValidator", "TypedGraphSnapshot",
    "USGGraphProjector", "compute_graph_hash",
    # schemas
    "AuthorizedUpdateProposal", "CandidateDisposition", "EvidenceClassification",
    "GraphCandidate", "GraphQuery", "GraphQueryAnchor", "GraphSchema", "ModelManifest",
    "new_hypothesis_namespace",
    # verification
    "CandidateVerifier", "PromotionGate", "VerificationResult",
    "__version__",
]
