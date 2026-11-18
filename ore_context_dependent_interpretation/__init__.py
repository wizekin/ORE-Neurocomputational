"""ORE Point 12 context-dependent interpretation package."""

from .annotations import register_annotations
from .biology import (
    BiologicalContextNetwork,
    HippocampalDiscourseMemoryNetwork,
    LanguageEncodingNetwork,
    MedialPrefrontalGoalContextNetwork,
    PerceptualSceneNetwork,
    PosteriorMidlineSituationModelNetwork,
    PrefrontalSelectionControlNetwork,
    TemporalSemanticMemoryNetwork,
)
from .contracts import (
    CandidateHypothesis,
    ContextEvidence,
    ContextInterpretationConfig,
    ContextInterpretationResult,
    ContextState,
    InputEvent,
    ScoredCandidate,
    SituationModel,
)
from .functional import (
    adapt_context_weights,
    bayesian_update,
    build_evidence_ledger,
    detect_conflicts,
    entropy,
    estimate_context_priors,
    generate_candidates,
    resolve_competition,
    retrieve_context,
    revise_candidates,
    score_input_evidence,
    select_goal,
    similarity,
    softmax,
    tokenize,
    update_situation_model,
)
from .pipeline import ContextDependentInterpretationProcessor, record_result_to_sqlite
from .registry import OPERATION_REGISTRY, OperationSpec, registry_as_dicts

__all__ = [
    "ContextEvidence", "CandidateHypothesis", "InputEvent", "SituationModel",
    "ContextState", "ScoredCandidate", "ContextInterpretationConfig",
    "ContextInterpretationResult", "LanguageEncodingNetwork",
    "HippocampalDiscourseMemoryNetwork", "TemporalSemanticMemoryNetwork",
    "MedialPrefrontalGoalContextNetwork", "PerceptualSceneNetwork",
    "PosteriorMidlineSituationModelNetwork", "PrefrontalSelectionControlNetwork",
    "BiologicalContextNetwork", "ContextDependentInterpretationProcessor",
    "OPERATION_REGISTRY", "OperationSpec", "registry_as_dicts",
    "register_annotations", "record_result_to_sqlite", "tokenize", "similarity",
    "retrieve_context", "select_goal", "generate_candidates",
    "estimate_context_priors", "score_input_evidence", "softmax",
    "bayesian_update", "entropy", "resolve_competition", "detect_conflicts",
    "revise_candidates", "update_situation_model", "adapt_context_weights",
    "build_evidence_ledger",
]