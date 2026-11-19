"""ORE Point 03: spaced repetition, adaptive review timing, and memory-state tracking."""
from .annotations import register_annotations
from .biology import (
    BiologicalSpacedRepetitionNetwork,
    CorticalSpacedPlasticityNetwork,
    HippocampalReactivationNetwork,
    NeuralPopulation,
    NeuromodulatoryLearningGate,
    OfflineConsolidationNetwork,
    PrefrontalRetrievalControlNetwork,
)
from .contracts import (
    LearningItem,
    MemoryRecord,
    OfflineConsolidationObservation,
    ReviewObservation,
    SpacedRepetitionConfig,
    SpacedRepetitionRequest,
    SpacedRepetitionResult,
    SpacedRepetitionState,
    normalize_utc,
)
from .functional import (
    add_hours,
    apply_offline_consolidation,
    assess_calibration,
    assess_maintenance,
    choose_initial_interval,
    choose_next_interval,
    clamp,
    decide_review_action,
    elapsed_hours,
    estimate_accessibility,
    estimate_encoding_quality,
    initialize_stability,
    parse_utc,
    score_review,
    update_memory_state,
)
from .pipeline import SpacedRepetitionProcessor, record_result_to_sqlite
from .registry import OPERATION_REGISTRY, OperationSpec, registry_as_dicts

__all__ = [
    "LearningItem", "ReviewObservation", "OfflineConsolidationObservation",
    "SpacedRepetitionConfig", "SpacedRepetitionRequest", "SpacedRepetitionState",
    "MemoryRecord", "SpacedRepetitionResult", "normalize_utc",
    "NeuralPopulation", "HippocampalReactivationNetwork",
    "CorticalSpacedPlasticityNetwork", "PrefrontalRetrievalControlNetwork",
    "OfflineConsolidationNetwork", "NeuromodulatoryLearningGate",
    "BiologicalSpacedRepetitionNetwork", "OperationSpec", "OPERATION_REGISTRY",
    "registry_as_dicts", "SpacedRepetitionProcessor", "register_annotations",
    "record_result_to_sqlite", "clamp", "parse_utc", "add_hours", "elapsed_hours",
    "estimate_encoding_quality", "initialize_stability", "choose_initial_interval",
    "estimate_accessibility", "decide_review_action", "score_review",
    "update_memory_state", "choose_next_interval", "assess_maintenance",
    "apply_offline_consolidation", "assess_calibration",
]
