"""ORE Point 18: attentional failure, distraction, fatigue, and missed information."""
from .annotations import register_annotations
from .biology import AttentionPopulation, AttentionalBiologicalNetwork
from .contracts import (
    AttentionRequirement, AttentionalConfig, AttentionalExperience,
    AttentionalResult, AttentionalState, InformationRequirement,
)
from .functional import (
    assess_competing_inputs, audit_completeness, bias_sensory_representation,
    choose_recovery_action, clip01, compute_priority_map, cosine_similarity,
    decide_downstream_release, detect_attention_lapse, establish_task_requirements,
    estimate_attentional_state, gate_working_memory, preserve_uncertainty,
    reencode_recovered_information, reconstruct_missing_information, reorient_to_task,
    sigmoid, softmax, update_sustained_attention, verify_repair, adapt_future_attention,
)
from .pipeline import AttentionalFailureProcessor, record_result_to_sqlite
from .registry import AttentionOperationSpec, OPERATION_REGISTRY, registry_as_dicts

__all__ = [
    "AttentionPopulation", "AttentionalBiologicalNetwork", "AttentionRequirement",
    "AttentionalConfig", "AttentionalExperience", "AttentionalResult", "AttentionalState",
    "InformationRequirement", "AttentionalFailureProcessor", "AttentionOperationSpec",
    "OPERATION_REGISTRY", "registry_as_dicts", "register_annotations", "record_result_to_sqlite",
    "clip01", "sigmoid", "softmax", "cosine_similarity", "establish_task_requirements",
    "estimate_attentional_state", "assess_competing_inputs", "compute_priority_map",
    "bias_sensory_representation", "gate_working_memory", "update_sustained_attention",
    "detect_attention_lapse", "reconstruct_missing_information", "audit_completeness",
    "choose_recovery_action", "reorient_to_task", "reencode_recovered_information",
    "verify_repair", "adapt_future_attention", "preserve_uncertainty",
    "decide_downstream_release",
]
