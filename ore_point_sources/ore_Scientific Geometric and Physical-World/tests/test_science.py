"""Acceptance tests for the ore_science core: contracts, interpretation, validation, and pipeline.

Each test maps to one or more acceptance criteria in Section 8 of the integrated
algorithm specification ("ORE Integrated Algorithm - Point 12: Scientific, Geometric
and Physical-World AI").
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from ore_science import (
    UNITS,
    CandidateKnowledgeUpdate,
    EquationModel,
    KnowledgeUpdatePolicy,
    ModelRegistry,
    OutcomeKind,
    ProblemClass,
    ProvenanceRecorder,
    RepresentationBuilder,
    ResourceAdmissionControl,
    ResultStatus,
    RunManifest,
    ScienceConfig,
    ScienceRequest,
    ScienceResourceBudget,
    ScientificDomain,
    ScientificOrchestrator,
    ScientificReasoner,
    ScientificTaskParser,
    StatusLadder,
    ValidationEngine,
    quantity_from_string,
    redact_secrets,
    richardson_convergence,
    validate_status_transition,
)
from ore_science.knowledge import ScienceKnowledgeBase
from ore_science.representations import FactorGraphRepresentation, FactorSpec, RepresentationKind
from ore_science.units import Quantity


def _config() -> ScienceConfig:
    """Return a fast test configuration."""
    return ScienceConfig(
        collocation_points=41,
        time_steps=160,
        refinement_levels=3,
        ensemble_members=24,
        pin_basis_size=6,
        pin_max_iterations=60,
    )


def _knowledge() -> ScienceKnowledgeBase:
    """Return the default scientific knowledge base."""
    return ScienceKnowledgeBase()


def _heat_request() -> ScienceRequest:
    """Return the canonical heat-diffusion request."""
    return ScienceRequest(
        question=(
            "Predict the temperature field of a 1 m rod after 60 s with thermal "
            "diffusivity alpha = 1e-4 m2/s, initial temperature 100 degC, ends held "
            "at 0 degC"
        ),
        request_id="test_heat",
    )


# ---------------------------------------------------------------------------
# Phase I: task interpretation (Steps 1-7)
# ---------------------------------------------------------------------------
def test_natural_language_request_converts_to_typed_task() -> None:
    """A natural-language scientific request becomes a typed task with objective and domain."""
    task = ScientificTaskParser(_knowledge()).parse(_heat_request())
    assert task.domains == (ScientificDomain.PHYSICAL_SIMULATION,)
    assert task.objective_kind.value == "ESTIMATE_FIELD"
    assert task.problem_class is ProblemClass.FORWARD
    assert "temperature" in task.objective.lower() or "predict" in task.objective.lower()


def test_multi_domain_task_decomposes_into_linked_subtasks() -> None:
    """Multi-domain tasks decompose into typed subtasks with explicit dependencies."""
    request = ScienceRequest(
        question=(
            "Discover crystal materials for a battery and validate their stability "
            "with a simulation of the heat diffusion during charging"
        )
    )
    task = ScientificTaskParser(_knowledge()).parse(request)
    assert len(task.domains) >= 2
    assert task.subtasks, "multi-domain tasks must produce linked subtasks"
    for index, link in enumerate(task.subtasks[1:], start=1):
        assert link.depends_on, "later subtasks must declare dependencies"


def test_every_variable_records_meaning_unit_domain_and_provenance() -> None:
    """Every variable registry entry records meaning, unit, domain, and provenance."""
    task = ScientificTaskParser(_knowledge()).parse(_heat_request())
    for variable in task.variables:
        record = variable.to_dict()
        assert record["meaning"]
        assert record["unit"]
        assert record["domain"]
        assert record["provenance"]


def test_incompatible_dimensions_cause_formalization_failure() -> None:
    """Incompatible dimensions in the request facts fail formalization loudly."""
    request = ScienceRequest(
        question="Predict the temperature field of a rod with alpha = 1e-4 m2/s",
        facts={"L": "1 m", "L_conflicting": "60 s"},
    )
    request.facts["L"] = "1 m"
    request = ScienceRequest(
        question="Predict the temperature field of a 1 m rod with alpha = 1e-4 m2/s",
        facts={"L": "60 s"},
    )
    with pytest.raises(ValueError):
        ScientificTaskParser(_knowledge()).parse(request)


def test_unit_conversion_preserves_physical_value() -> None:
    """Unit conversion preserves the physical value within declared tolerance."""
    boiling = quantity_from_string("100 degC")
    in_kelvin = boiling.in_si()
    assert in_kelvin.value == pytest.approx(373.15, abs=1.0e-9)
    distance = quantity_from_string("1 km").convert_to(UNITS["m"])
    assert distance.value == pytest.approx(1000.0, abs=1.0e-9)
    fahrenheit = quantity_from_string("100 degC").convert_to(UNITS["degF"])
    assert fahrenheit.value == pytest.approx(212.0, abs=1.0e-6)
    round_trip = fahrenheit.convert_to(UNITS["degC"])
    assert round_trip.value == pytest.approx(100.0, abs=1.0e-6)


def test_dimension_checked_arithmetic_rejects_mixed_units() -> None:
    """Adding quantities of different dimensions raises instead of coercing."""
    with pytest.raises(ValueError):
        Quantity(1.0, UNITS["m"]) + Quantity(1.0, UNITS["s"])
    with pytest.raises(ValueError):
        Quantity(1.0, UNITS["K"]) - Quantity(1.0, UNITS["m"])
    allowed = Quantity(1.0, UNITS["m"]) + Quantity(2.0, UNITS["km"])
    assert allowed.in_si().value == pytest.approx(2001.0)


def test_initial_and_boundary_conditions_are_separate_constraints() -> None:
    """Initial and boundary conditions are represented separately in the task."""
    task = ScientificTaskParser(_knowledge()).parse(_heat_request())
    kinds = {constraint.kind for constraint in task.constraints}
    assert "initial_condition" in kinds
    assert "boundary_condition" in kinds
    initial = next(c for c in task.constraints if c.kind == "initial_condition")
    boundary = next(c for c in task.constraints if c.kind == "boundary_condition")
    assert initial.expression != boundary.expression


# ---------------------------------------------------------------------------
# Phase II: representations (Steps 8-15)
# ---------------------------------------------------------------------------
def test_representation_builder_selects_molecular_graph_for_structure_task() -> None:
    """The representation builder selects a molecular graph for molecular tasks."""
    task = ScientificTaskParser(_knowledge()).parse(
        ScienceRequest(question="Predict the structure of a protein backbone")
    )
    artifact, decision = RepresentationBuilder(_config()).build(task)
    assert artifact.kind is RepresentationKind.MOLECULAR_GRAPH
    assert decision.selected is RepresentationKind.MOLECULAR_GRAPH


def test_molecular_graph_preserves_atom_and_bond_attributes() -> None:
    """Molecular graph construction preserves required atom and bond attributes."""
    task = ScientificTaskParser(_knowledge()).parse(
        ScienceRequest(question="Predict the structure of a protein backbone")
    )
    artifact, _ = RepresentationBuilder(_config()).build(task)
    graph = artifact.payload
    for element, charge, position in graph.atoms:
        assert element
        assert charge is not None
        assert position is not None
    for a, b, order, bond_type in graph.bonds:
        assert a != b
        assert order >= 1
        assert bond_type
    serialized = graph.to_dict()
    assert len(serialized["atoms"]) == len(graph.atoms)
    assert len(serialized["bonds"]) == len(graph.bonds)


def test_geometry_metadata_includes_frame_and_unit_conventions() -> None:
    """Geometry metadata includes coordinate frame and unit conventions."""
    task = ScientificTaskParser(_knowledge()).parse(
        ScienceRequest(question="Check the rotation equivariance of a geometric transform")
    )
    artifact, _ = RepresentationBuilder(_config()).build(task)
    assert artifact.coordinate_frame
    assert artifact.unit_symbol
    assert artifact.payload.transform_convention


def test_materials_representation_preserves_periodicity_and_lattice() -> None:
    """Materials representations preserve periodicity and lattice metadata."""
    task = ScientificTaskParser(_knowledge()).parse(
        ScienceRequest(question="Discover stable crystal materials with a perovskite oxide lattice")
    )
    artifact, _ = RepresentationBuilder(_config()).build(task)
    graph = artifact.payload
    assert graph.periodic
    assert len(graph.lattice) == 3
    assert graph.unit_cell_volume_nm3() > 0.0
    serialized = artifact.to_dict()
    assert serialized["payload"]["periodic"] is True
    assert len(serialized["payload"]["lattice"]) == 3


def test_field_representation_preserves_domain_metadata() -> None:
    """Field representations preserve spatial and temporal domain metadata."""
    task = ScientificTaskParser(_knowledge()).parse(_heat_request())
    artifact, _ = RepresentationBuilder(_config()).build(task)
    field = artifact.payload
    assert field.spatial_domain == (0.0, 1.0)
    assert field.time_domain == (0.0, 60.0)
    assert field.unit.symbol == "K"
    assert field.boundary_condition
    assert field.initial_condition


def test_factor_graphs_reject_factors_referencing_undefined_variables() -> None:
    """Factor graphs reject factors referencing undefined variables."""
    graph = FactorGraphRepresentation(variables=("x0", "x1"))
    factor = FactorSpec(
        factor_id="bad",
        factor_type="odometry",
        scope=("x0", "x9"),
        residual=lambda state: np.zeros(1),
        covariance=(0.01,),
    )
    with pytest.raises(ValueError):
        graph.add_factor(factor)


# ---------------------------------------------------------------------------
# Model registry (Section 3.D, 5.8)
# ---------------------------------------------------------------------------
def test_model_registry_rejects_incompatible_input_schema() -> None:
    """The registry rejects models whose input schema is incompatible with the task."""
    registry = ModelRegistry()
    candidates = registry.find_compatible(
        (ScientificDomain.PHYSICAL_SIMULATION,),
        ProblemClass.STATE_ESTIMATION,
        RepresentationKind.FIELD,
    )
    assert not candidates.models
    assert candidates.rejections
    reasons = " ".join(rejection.reason for rejection in candidates.rejections)
    assert "problem class" in reasons or "input schema" in reasons


def test_interaction_tasks_not_routed_to_protein_only_model() -> None:
    """Molecular interaction tasks are not routed to models lacking the interaction class."""
    registry = ModelRegistry()
    candidates = registry.find_compatible(
        (ScientificDomain.MOLECULAR_INTERACTION,),
        ProblemClass.FORWARD,
        RepresentationKind.MOLECULAR_GRAPH,
        interaction_class="protein_small_molecule",
    )
    routed = {model.model_id for model in candidates.models}
    assert "molecular.protein_only_structure" not in routed
    interaction_rejections = [
        rejection for rejection in candidates.rejections
        if "interaction class" in rejection.reason
    ]
    assert interaction_rejections, "the protein-only model must be explicitly rejected"


def test_external_services_are_rejected_as_not_configured() -> None:
    """Unconfigured external services are rejected explicitly, never silently used."""
    registry = ModelRegistry()
    candidates = registry.find_compatible(
        (ScientificDomain.MOLECULAR_STRUCTURE,),
        ProblemClass.FORWARD,
        RepresentationKind.MOLECULAR_GRAPH,
    )
    routed = {model.model_id for model in candidates.models}
    assert "molecular.alphafold_service" not in routed
    assert any(
        "not configured" in rejection.reason for rejection in candidates.rejections
    )


def test_selection_score_ranks_only_gate_passing_models() -> None:
    """The Section 5.8 score ranks candidates that passed the hard gates in order."""
    registry = ModelRegistry()
    candidates = registry.find_compatible(
        (ScientificDomain.PHYSICAL_SIMULATION,),
        ProblemClass.FORWARD,
        RepresentationKind.FIELD,
    )
    assert candidates.models
    scores = [
        score
        for score in (
            __import__("ore_science").selection_score(model) for model in candidates.models
        )
    ]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Phase IV: validation engine (Steps 35-44)
# ---------------------------------------------------------------------------
def test_dimensional_validation_catches_inconsistent_equation() -> None:
    """Dimensional validation catches a deliberately inconsistent equation."""
    inconsistent = EquationModel(
        equation_id="EQ-BAD",
        name="Inconsistent equation",
        form="du/dt = alpha * u",
        terms=(("du/dt", UNITS["K_s"]), ("alpha * u", UNITS["m2_s"])),
        symbols=(("u", UNITS["K"]), ("alpha", UNITS["m2_s"])),
    )
    report = ValidationEngine().dimensional_validation([inconsistent])
    assert not report.passed
    assert any("dimension" in check.detail.lower() for check in report.checks)


def test_conservation_checks_detect_perturbed_solution() -> None:
    """Conservation checks detect a deliberately perturbed solution."""
    engine = ValidationEngine()
    x_grid = np.linspace(0.0, 1.0, 21)
    fields = [
        np.full(21, 300.0) - 10.0 * index for index in range(5)
    ]
    clean = engine.conservation_checks(fields, x_grid, tolerance=1.0e-6)
    assert clean.passed
    perturbed = [field.copy() for field in fields]
    perturbed[2] += 25.0
    detected = engine.conservation_checks(perturbed, x_grid, tolerance=1.0e-6)
    assert not detected.passed


def test_numerical_refinement_reports_convergence_explicitly() -> None:
    """Numerical refinement studies report convergence or non-convergence explicitly."""
    converging = [1.0, 1.02, 1.025, 1.0256]
    outcome = richardson_convergence(converging)
    assert outcome["converged"] is True
    assert outcome["status"] == "converged"
    diverging = [1.0, 1.5, 2.6, 5.0]
    outcome_bad = richardson_convergence(diverging)
    assert outcome_bad["converged"] is False
    assert outcome_bad["status"] == "not_converged"


def test_independent_baseline_comparison_records_discrepancy() -> None:
    """Independent baseline comparison records discrepancy and tolerance."""
    engine = ValidationEngine()
    baseline = np.array([300.0, 302.0, 304.0])
    good = baseline + 0.01
    report = engine.baseline_comparison(
        good, baseline, quantity_name="field", tolerance=0.01,
        baseline_identity="analytic reference",
    )
    assert report.passed
    assert "tolerance" in report.checks[0].detail
    bad = baseline + 5.0
    failed = engine.baseline_comparison(
        bad, baseline, quantity_name="field", tolerance=0.01,
        baseline_identity="analytic reference",
    )
    assert not failed.passed


# ---------------------------------------------------------------------------
# Status ladder (Step 43)
# ---------------------------------------------------------------------------
def test_status_ladder_rejects_conflating_evidence_states() -> None:
    """Candidate, predicted, validated, and confirmed statuses cannot be conflated."""
    assert not validate_status_transition(ResultStatus.CANDIDATE, ResultStatus.COMPUTATIONALLY_VALIDATED)
    assert not validate_status_transition(ResultStatus.PREDICTED, ResultStatus.EXPERIMENTALLY_CONFIRMED)
    assert not validate_status_transition(ResultStatus.EXPERIMENTALLY_CONFIRMED, ResultStatus.PREDICTED)
    with pytest.raises(ValueError):
        StatusLadder().request_promotion(
            ResultStatus.CANDIDATE, ResultStatus.COMPUTATIONALLY_VALIDATED,
            gate="shortcut", reviewer="nobody", evidence_reference="none",
        )


def test_status_promotion_requires_named_gate_and_reviewer() -> None:
    """Legal promotion requires a named gate, reviewer, and evidence reference."""
    ladder = StatusLadder()
    status, transition = ladder.request_promotion(
        ResultStatus.PREDICTED, ResultStatus.COMPUTATIONALLY_VALIDATED,
        gate="domain_review_gate:PHYSICAL_SIMULATION",
        reviewer="ore_science.validation_engine",
        evidence_reference="validation report aggregate",
    )
    assert status is ResultStatus.COMPUTATIONALLY_VALIDATED
    assert transition is not None
    assert transition.gate.startswith("domain_review_gate")


# ---------------------------------------------------------------------------
# Uncertainty (Steps 39-42, Sections 5.6-5.7)
# ---------------------------------------------------------------------------
def test_sensitivity_analysis_identifies_high_influence_parameter() -> None:
    """Sensitivity analysis identifies a parameter with known high influence."""
    from ore_science import identifiability_flags, sensitivity_analysis

    def function(parameters: np.ndarray) -> np.ndarray:
        return np.array([3.0 * parameters[0] + 0.001 * parameters[1]])

    result = sensitivity_analysis(function, np.array([1.0, 1.0]))
    assert result["column_norms"][0] > 100.0 * result["column_norms"][1]
    flags = identifiability_flags(
        result["normalized_jacobian"], ("influential", "negligible")
    )
    assert "negligible" in flags["insensitive_parameters"]


def test_non_identifiable_parameters_are_flagged() -> None:
    """Non-identifiable (perfectly correlated) parameters are flagged, not given false precision."""
    from ore_science import identifiability_flags

    jacobian = np.array([[1.0, 1.0, 0.2], [1.0, 1.0, 0.2], [1.0, 1.0, 0.2]])
    flags = identifiability_flags(jacobian, ("a", "b", "c"))
    assert flags["non_identifiable"]
    correlated = {(pair["a"], pair["b"]) for pair in flags["correlated_parameter_pairs"]}
    assert ("a", "b") in correlated
    assert "false precision" in flags["guidance"]


def test_out_of_domain_inputs_trigger_warning() -> None:
    """Out-of-domain inputs trigger a warning with a conservative action."""
    from ore_science import out_of_domain_check

    result = out_of_domain_check(
        {"alpha": 5.0}, {"alpha": (1.0e-7, 1.0e-2)}
    )
    assert not result["in_domain"]
    assert result["warnings"]
    assert result["action"] == "conservative solver or expert review"
    in_domain = out_of_domain_check({"alpha": 1.0e-4}, {"alpha": (1.0e-7, 1.0e-2)})
    assert in_domain["in_domain"]


# ---------------------------------------------------------------------------
# Provenance (Step 50)
# ---------------------------------------------------------------------------
def test_repeated_execution_records_reproducibility_metadata() -> None:
    """Repeated execution with fixed deterministic settings records reproducibility metadata."""
    recorder = ProvenanceRecorder()
    common = dict(
        task_id="task_repro",
        question="repro question",
        models=({"model_id": "m", "model_version": "1.0.0"},),
        solver_settings={"grid": 41},
        random_seed=7,
        equation_versions=("EQ-HEAT-1D",),
        input_payload={"q": 1},
        output_payload={"r": 2},
        validation_summary={"passed": True},
        result_status="PREDICTED",
    )
    first = recorder.build_manifest(**common)
    second = recorder.build_manifest(**common)
    check = ProvenanceRecorder.reproducibility_check(first, second)
    assert check["reproducible"]
    assert check["output_hash_match"]
    third = recorder.build_manifest(**{**common, "random_seed": 8})
    assert not ProvenanceRecorder.reproducibility_check(first, third)["reproducible"]


def test_provenance_manifest_redacts_secrets() -> None:
    """Scientific provenance does not expose protected credentials."""
    redacted = redact_secrets(
        {"api_key": "abc123", "model": "x", "nested": {"token": "t", "value": 1}}
    )
    assert redacted["api_key"] == "[REDACTED]"
    assert redacted["nested"]["token"] == "[REDACTED]"
    assert redacted["model"] == "x"
    manifest = RunManifest(
        task_id="t", question="q", sensitive_notes=("none",),
    )
    serialized = manifest.to_dict()
    assert json.dumps(serialized)


# ---------------------------------------------------------------------------
# Orchestration (Steps 48-49)
# ---------------------------------------------------------------------------
def test_resource_admission_rejects_exceeding_budget() -> None:
    """Resource admission rejects a task exceeding its CPU/GPU/memory budget."""
    control = ResourceAdmissionControl(_config())
    over_memory = ScienceResourceBudget(cpu_cores=1.0, memory_mb=99999.0, time_seconds=10.0)
    decision = control.submit(over_memory)
    assert not decision.admitted
    assert "memory" in decision.reason
    over_gpu = ScienceResourceBudget(
        cpu_cores=1.0, memory_mb=64.0, time_seconds=10.0, gpu_units=4.0
    )
    gpu_decision = control.submit(over_gpu)
    assert not gpu_decision.admitted
    assert "gpu" in gpu_decision.reason
    within = ScienceResourceBudget(cpu_cores=1.0, memory_mb=64.0, time_seconds=10.0)
    assert control.submit(within).admitted


def test_cancellation_releases_reservation_and_preserves_provenance() -> None:
    """Cancellation releases reservations and preserves partial-run provenance."""
    control = ResourceAdmissionControl(_config())
    budget = ScienceResourceBudget(cpu_cores=1.0, memory_mb=64.0, time_seconds=5.0)
    admission = control.submit(budget)
    assert admission.admitted
    cancellation = control.cancel(admission.reservation_id)
    assert cancellation.released
    assert "partial-run provenance preserved" in cancellation.reason
    assert control.active_reservations() == 0


def test_multi_domain_orchestration_preserves_per_stage_assumptions() -> None:
    """Multi-domain orchestration preserves per-stage assumptions and uncertainty."""
    task = ScientificTaskParser(_knowledge()).parse(
        ScienceRequest(
            question=(
                "Discover crystal materials for a battery and validate stability "
                "with a heat diffusion simulation"
            )
        )
    )
    plan = ScientificOrchestrator().decompose(task)
    serialized = plan.to_dict()
    assert serialized["uncertainty_policy"]
    assert len(plan.subtasks) >= 2
    for link in plan.subtasks[1:]:
        assert link.depends_on


# ---------------------------------------------------------------------------
# Knowledge update policy (Steps 47, 51)
# ---------------------------------------------------------------------------
def test_usg_candidate_updates_require_provenance_and_review() -> None:
    """USG candidate updates require provenance and the configured review process."""
    policy = KnowledgeUpdatePolicy()
    with pytest.raises(ValueError):
        CandidateKnowledgeUpdate(
            subject_concept="heat diffusion",
            predicate="related_to",
            object_concept="alpha",
            evidence_reference="",
            provenance_digest="digest",
            confidence=0.9,
        )
    with pytest.raises(ValueError):
        CandidateKnowledgeUpdate(
            subject_concept="heat diffusion",
            predicate="related_to",
            object_concept="alpha",
            evidence_reference="ref",
            provenance_digest="digest",
            confidence=0.9,
            review_status="accepted",
        )
    update = CandidateKnowledgeUpdate(
        subject_concept="heat diffusion",
        predicate="supported_by_computational_evidence",
        object_concept="pinn.basis_heat_1d",
        evidence_reference="validation report",
        provenance_digest="digest",
        confidence=0.9,
    )
    policy.submit_candidate(update)
    outcome = policy.review_outcome(0, accepted=True, reviewer="reviewer")
    assert outcome["accepted"]
    assert policy.to_dict()["candidates"][0]["review_status"] == "review_accepted"


def test_model_selection_policy_update_is_versioned_and_rollbackable() -> None:
    """Model-selection policy updates are versioned and can be rolled back."""
    policy = KnowledgeUpdatePolicy()
    assert policy.policy_version == 1
    policy.record_prediction_error("model_a", 0.5)
    policy.record_prediction_error("model_a", -0.2)
    assert policy.policy_version == 3
    rollback = policy.rollback(1)
    assert rollback["rolled_back"]
    assert policy.policy_version == 1
    with pytest.raises(ValueError):
        policy.rollback(99)


# ---------------------------------------------------------------------------
# End-to-end pipeline (all phases)
# ---------------------------------------------------------------------------
def test_full_pipeline_heat_diffusion_completes_with_validation() -> None:
    """The full 52-step loop completes a heat-diffusion task with independent validation."""
    reasoner = ScientificReasoner(_config())
    result = reasoner.process(_heat_request())
    assert result.outcome is OutcomeKind.COMPLETED
    assert result.diagnostics["valid"]
    assert result.diagnostics["operation_count"] == 52
    assert result.diagnostics["trace_complete"]
    assert result.result_status == "COMPUTATIONALLY_VALIDATED"
    model_ids = {prediction["model_id"] for prediction in result.predictions}
    assert "solver.fd_heat_1d" in model_ids
    assert "pinn.basis_heat_1d" in model_ids
    assert all(report["passed"] for report in result.validation_reports)
    assert result.provenance_ref and result.provenance_ref["digest"]


def test_explanation_identifies_model_representation_and_limitations() -> None:
    """A result explanation identifies model, representation, assumptions, checks, and limitations."""
    reasoner = ScientificReasoner(_config())
    result = reasoner.process(_heat_request())
    explanation = result.explanation
    assert explanation["representation_rationale"]
    assert explanation["solver_rationale"]
    assert explanation["applied_constraints"]
    assert explanation["validation_summary"]
    assert explanation["remaining_uncertainty"]
    assert "mathematical_derivation" in explanation["statement_classes"]
    assert "model_prediction" in explanation["statement_classes"]


def test_invalid_request_returns_typed_outcome_not_exception() -> None:
    """Invalid input produces a typed INVALID_INPUT outcome, not an exception."""
    reasoner = ScientificReasoner(_config())
    result = reasoner.process(
        ScienceRequest(question="rod alpha = 1 m", facts={"L": "60 s"})
    )
    assert result.outcome is OutcomeKind.INVALID_INPUT
    assert result.diagnostics["operation_count"] == 52
    blocked = [
        row for row in result.operation_trace if row["functional_status"] == "blocked"
    ]
    assert blocked


def test_result_serializes_to_json() -> None:
    """The full result serializes to JSON for logs and workspace publication."""
    reasoner = ScientificReasoner(_config())
    result = reasoner.process(_heat_request())
    payload = result.to_dict()
    encoded = json.dumps(payload)
    assert "ore.science.52_step.v1" in payload["schema"]
    assert len(payload["operation_trace"]) == 52
    assert encoded


def test_pipeline_status_history_records_gated_transition() -> None:
    """The status history records the review-gated promotion with gate and reviewer."""
    reasoner = ScientificReasoner(_config())
    result = reasoner.process(_heat_request())
    assert result.status_history
    transition = result.status_history[0]
    assert transition["from_status"] == "PREDICTED"
    assert transition["to_status"] == "COMPUTATIONALLY_VALIDATED"
    assert transition["gate"].startswith("domain_review_gate")
    assert transition["reviewer"]
