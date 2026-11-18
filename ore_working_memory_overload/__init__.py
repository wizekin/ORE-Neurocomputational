"""ORE Point 19: working-memory overload and information loss during complex interpretation."""
from .biology import DistributedWorkingMemoryNetwork, RatePopulation
from .contracts import (
    Dependency, Point19Config, Point19Request, Point19Result, Point19State, Representation,
)
from .functional import (
    adapt_difficulty, allocate_attention, apply_recovery, check_coherence,
    choose_recovery_action, clip01, commit_interpretation, create_dependencies,
    detect_overload, diagnose_failures, encode_unit, estimate_interference,
    estimate_load, feature_similarity, form_chunks, identify_at_risk,
    integrate_retrieval, reconstruct_supported, retrieve_by_cue, segment_units,
    sigmoid, softmax, update_accessibility,
)
from .pipeline import Point19WorkingMemoryOverloadProcessor, record_result_to_sqlite, register_annotations
from .registry import OPERATION_REGISTRY, Point19OperationSpec, registry_as_dicts

__all__ = [
    "Dependency", "Representation", "Point19Config", "Point19Request", "Point19State",
    "Point19Result", "RatePopulation", "DistributedWorkingMemoryNetwork",
    "Point19OperationSpec", "OPERATION_REGISTRY", "registry_as_dicts",
    "Point19WorkingMemoryOverloadProcessor", "register_annotations", "record_result_to_sqlite",
    "clip01", "sigmoid", "softmax", "segment_units", "feature_similarity", "encode_unit",
    "create_dependencies", "update_accessibility", "estimate_load", "form_chunks",
    "allocate_attention", "estimate_interference", "retrieve_by_cue", "integrate_retrieval",
    "detect_overload", "identify_at_risk", "choose_recovery_action", "apply_recovery",
    "diagnose_failures", "reconstruct_supported", "check_coherence",
    "commit_interpretation", "adapt_difficulty",
]
