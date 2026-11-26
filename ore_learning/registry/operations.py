"""Operation specifications for the 56-step RLMS pipeline (Point 14, Section 3).

Each source step has a stable identifier, a phase, an operation name, and a short
responsibility statement.  The pipeline records one trace entry per step; the
registry uses these specs to describe what the subsystem does.
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = ("OPERATION_SPECS", "OperationSpec", "phase_for_step", "steps_for_phase")


@dataclass(frozen=True, slots=True)
class OperationSpec:
    """One source step of the Point 14 specification."""

    step: int
    phase: str
    operation: str
    description: str


_OPERATION_ROWS: tuple[tuple[int, str, str, str], ...] = (
    (1, "A", "normalize_task", "receive and normalize the cognitive task"),
    (2, "A", "necessity_check", "decide whether learned inference is necessary"),
    (3, "A", "build_capability_request", "express the need as typed capabilities"),
    (4, "A", "decompose_task", "decompose at verifiable boundaries"),
    (5, "A", "set_authority_policy", "set evidence and authority policy"),
    (6, "B", "query_registry", "query the learned-model registry"),
    (7, "B", "check_compatibility", "enforce E = S and C and P and R and L before ranking"),
    (8, "B", "rank_candidates", "rank eligible candidates by measured utility"),
    (9, "B", "moe_routing_reference", "use sparse MoE as a routing pattern only"),
    (10, "B", "select_model_or_ensemble", "select one model or a bounded ensemble"),
    (11, "B", "reserve_resources", "reserve resources through CROS"),
    (12, "C", "frozen_feature_test", "prefer frozen feature extraction when sufficient"),
    (13, "C", "select_adaptation_method", "select the smallest suitable adaptation method"),
    (14, "C", "build_adaptation_dataset", "build a reproducible adaptation dataset"),
    (15, "C", "train_isolated_candidate", "train an isolated candidate version"),
    (16, "C", "compare_adapted_and_base", "compare adapted and base on held-out tasks"),
    (17, "C", "ttt_experiment", "treat test-time training as an opt-in experiment"),
    (18, "D", "receive_modality_input", "receive modality-specific input"),
    (19, "D", "validate_modality", "validate and normalize modality data"),
    (20, "D", "run_perception_specialists", "run modality-specific perception specialists"),
    (21, "D", "map_to_semantics", "map observations into shared semantic representations"),
    (22, "D", "fuse_modalities", "fuse modalities with explicit conflict handling"),
    (23, "E", "gan_objective_gate", "admit GANs only for a defined objective"),
    (24, "E", "validate_synthetic_data", "validate synthetic data before training use"),
    (25, "E", "evaluate_diffusion", "evaluate diffusion-style generation as alternative"),
    (26, "E", "gate_generated_outputs", "gate all generated outputs as untrusted"),
    (27, "F", "maintain_two_channels", "maintain two distinct explanation channels"),
    (28, "F", "apply_shap", "apply SHAP when feature attribution is meaningful"),
    (29, "F", "apply_lime", "apply LIME for local surrogate diagnostics"),
    (30, "F", "counterfactuals", "use counterfactuals and feature importance"),
    (31, "F", "record_explanation_limits", "record explanation limitations"),
    (32, "G", "represent_reasoning_steps", "represent reasoning as verifiable transitions"),
    (33, "G", "prm_screening", "use PRM-like scores for intermediate screening"),
    (34, "G", "deterministic_validation", "apply deterministic validators"),
    (35, "G", "construct_rlvr_feedback", "construct RLVR feedback from outcomes"),
    (36, "G", "separate_reward_from_correctness", "keep reward signals separate"),
    (37, "G", "offline_improvement", "improve candidate models offline"),
    (38, "H", "edge_decision", "decide whether inference belongs at the edge"),
    (39, "H", "compress_edge_specialists", "compress with measured trade-offs"),
    (40, "H", "split_execution", "use split execution where useful"),
    (41, "H", "federated_round", "enable federated learning only when required"),
    (42, "H", "protect_federated_updates", "protect federated updates"),
    (43, "H", "validate_aggregated_model", "validate aggregates before redistribution"),
    (44, "I", "identify_combinatorial_problem", "identify suitable combinatorial problems"),
    (45, "I", "formulate_qubo", "formulate and validate the QUBO objective"),
    (46, "I", "conventional_solver_baseline", "use conventional solvers as baselines"),
    (47, "I", "verify_optimization_output", "treat Ising results as candidate solutions"),
    (48, "J", "monitor_quality_calibration", "monitor task quality and calibration"),
    (49, "J", "detect_drift", "detect data and concept drift"),
    (50, "J", "detect_behavioral_regressions", "detect behavioral and operational regressions"),
    (51, "J", "quarantine_and_rollback", "quarantine and roll back deterministically"),
    (52, "J", "canary_deployment", "use canary deployment for replacements"),
    (53, "J", "maintain_lineage", "maintain reproducible model lineage"),
    (54, "J", "log_lifecycle_events", "log lifecycle transitions in SQLite"),
    (55, "J", "human_approval", "preserve human approval for sensitive transitions"),
    (56, "J", "retire_specialists", "retire unnecessary specialists"),
)

OPERATION_SPECS: tuple[OperationSpec, ...] = tuple(
    OperationSpec(step=step, phase=phase, operation=operation, description=description)
    for step, phase, operation, description in _OPERATION_ROWS)

_PHASES: dict[str, str] = {
    "A": "task interpretation and capability planning",
    "B": "registry, selection and routing",
    "C": "transfer learning and adaptation",
    "D": "multimodal perception and representation",
    "E": "generative models and synthetic data",
    "F": "explainability and diagnostics",
    "G": "process reward models and verifiable learning",
    "H": "edge, TinyML and federated learning",
    "I": "Ising and QUBO optimization",
    "J": "monitoring, governance and lifecycle",
}


def phase_for_step(step: int) -> str:
    """Return the phase letter for a step number."""
    if not 1 <= step <= len(OPERATION_SPECS):
        raise ValueError(f"step {step} is outside 1..{len(OPERATION_SPECS)}")
    return OPERATION_SPECS[step - 1].phase


def phase_description(phase: str) -> str:
    """Return the human-readable phase description."""
    return _PHASES.get(phase, "unknown phase")


def steps_for_phase(phase: str) -> tuple[OperationSpec, ...]:
    """Return the operation specs of one phase."""
    return tuple(spec for spec in OPERATION_SPECS if spec.phase == phase)


def step_description(step: int) -> str:
    """Return the responsibility statement of one step."""
    if not 1 <= step <= len(OPERATION_SPECS):
        raise ValueError(f"step {step} is outside 1..{len(OPERATION_SPECS)}")
    return OPERATION_SPECS[step - 1].description
