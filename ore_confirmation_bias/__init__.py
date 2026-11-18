"""ORE Point 21: confirmation bias and belief-consistent attention, interpretation, and memory."""
from .biology import ConfirmationBiasNetwork, FunctionalPopulation
from .contracts import (
    EvidenceItem, EvidenceSource, Hypothesis, Point21Config, Point21Request,
    Point21Result, Point21State,
)
from .functional import (
    allocate_attention, apply_corrective_control, assess_evidence_quality,
    build_evidence_audit, build_expectation_template, clamp,
    compare_evidence_symmetrically, detect_confirmation_feedback, entropy,
    encode_evidence_with_provenance, initialize_belief_state,
    interpret_ambiguous_evidence, readout_evidence_for_judgment,
    retrieve_belief_related_memories, select_information_sources, softmax,
    update_belief_confidence, validate_point21_stages,
)
from .pipeline import (
    Point21ConfirmationBiasProcessor, record_result_to_sqlite, register_annotations,
)
from .registry import OPERATION_REGISTRY, Point21OperationSpec, registry_as_dicts

__all__ = [
    "Hypothesis", "EvidenceSource", "EvidenceItem", "Point21Config", "Point21Request",
    "Point21State", "Point21Result", "FunctionalPopulation", "ConfirmationBiasNetwork",
    "Point21OperationSpec", "OPERATION_REGISTRY", "registry_as_dicts",
    "Point21ConfirmationBiasProcessor", "register_annotations", "record_result_to_sqlite",
    "clamp", "softmax", "entropy", "initialize_belief_state", "build_expectation_template",
    "select_information_sources", "allocate_attention", "encode_evidence_with_provenance",
    "interpret_ambiguous_evidence", "assess_evidence_quality", "compare_evidence_symmetrically",
    "readout_evidence_for_judgment", "update_belief_confidence",
    "retrieve_belief_related_memories", "detect_confirmation_feedback",
    "apply_corrective_control", "validate_point21_stages", "build_evidence_audit",
]
