"""Acceptance tests for the seventeen Point 22 cognitive-heuristics operations."""
from __future__ import annotations
import sqlite3
from pathlib import Path
import pytest
from ore_cognitive_heuristics import (
    JudgmentProblem, Point22CognitiveHeuristicsProcessor, Point22Config, Point22State,
    adjust_anchor, bayesian_base_rate_check, check_correlation_causation,
    estimate_sample_uncertainty, record_result_to_sqlite, register_annotations,
    representativeness_heuristic,
)

def make_problem(**overrides) -> JudgmentProblem:
    """Build a two-category case with a vivid rare-category example and base-rate evidence."""
    values = {
        "problem_id": "rare-pattern",
        "question": "Which explanation best fits the observed case?",
        "target_type": "probability",
        "categories": ("rare", "common"),
        "examples": (
            {"example_id": "vivid-1", "category": "rare", "recency": 1.0,
             "salience": 1.0, "fluency": 1.0, "strength": 1.0,
             "source_id": "news-a", "independence_group": "source-a"},
            {"example_id": "ordinary-1", "category": "common", "recency": 0.2,
             "salience": 0.2, "fluency": 0.2, "strength": 0.7,
             "source_id": "survey-b", "independence_group": "source-b"},
        ),
        "base_rates": {"rare": 0.05, "common": 0.95},
        "features": {"red": 0.9, "round": 0.8},
        "prototypes": {
            "rare": {"red": 1.0, "round": 1.0},
            "common": {"red": 0.1, "round": 0.2},
        },
        "feature_weights": {"red": 1.0, "round": 1.0},
        "similarity_scores": {"rare": 0.98, "common": 0.2},
        "evidence_confidence": 0.65, "fluency": 0.8,
        "time_pressure": 0.2, "emotional_salience": 0.3, "decision_cost": 0.2,
        "cue_validity": {"availability": 0.65, "representativeness": 0.7, "base_rate": 0.95},
        "environment_stability": 0.9,
        "evidence": {
            "likelihoods": {"rare": 0.8, "common": 0.15},
            "exposure_count": 6, "independent_source_count": 2,
        },
    }
    values.update(overrides)
    return JudgmentProblem(**values)

def test_pipeline_executes_seventeen_ordered_operations_with_two_output_tracks() -> None:
    """All source steps produce ordered functional and biological-reference outputs."""
    result = Point22CognitiveHeuristicsProcessor().process(make_problem())
    assert len(result.operation_trace) == 17
    assert [row["operation_id"] for row in result.operation_trace] == [
        f"P22-{index:02d}" for index in range(1, 18)
    ]
    assert all(row["status"] == "completed" for row in result.operation_trace)
    assert all("functional_output" in row and "biological_output" in row
               for row in result.operation_trace)
    assert result.diagnostics["validation"]["operation_order_verified"] is True
    assert result.metadata["source_sequence_steps"] == 17

def test_base_rate_and_likelihood_are_combined_with_bayes_rule() -> None:
    """A salient rare-category match remains constrained by its low prior."""
    result = bayesian_base_rate_check(make_problem(), ("rare", "common"))
    expected = (0.05 * 0.8) / ((0.05 * 0.8) + (0.95 * 0.15))
    assert result["posterior_distribution"]["rare"] == pytest.approx(expected)
    assert result["posterior_distribution"]["common"] == pytest.approx(1.0 - expected)
    assert result["bayes_rule_applied"] is True

def test_missing_likelihoods_leave_prior_distribution_unchanged() -> None:
    """A missing diagnostic likelihood is not silently fabricated."""
    result = bayesian_base_rate_check(make_problem(evidence={}), ("rare", "common"))
    assert result["posterior_distribution"] == pytest.approx({"rare": 0.05, "common": 0.95})
    assert result["posterior_updated"] is False

def test_availability_score_tracks_accessibility_not_population_frequency() -> None:
    """A vivid rare example can dominate recall while the actual prior remains separate."""
    result = Point22CognitiveHeuristicsProcessor(
        Point22Config(simulate_biology=False)
    ).process(make_problem())
    diagnostic = result.diagnostics["availability"]
    assert diagnostic["category_accessibility"]["rare"] > diagnostic["category_accessibility"]["common"]
    assert result.metadata["retrieval_fluency_separated_from_evidence"] is True
    assert result.probabilities["common"] > result.probabilities["rare"]

def test_representativeness_uses_similarity_without_claiming_probability() -> None:
    """Prototype similarity is exposed as its own signal."""
    result = representativeness_heuristic(make_problem(), ("rare", "common"))
    assert result["similarity_scores"]["rare"] > result["similarity_scores"]["common"]
    assert result["similarity_is_not_probability"] is True

def test_anchor_adjustment_moves_part_way_toward_an_independent_estimate() -> None:
    """The adjustment fraction controls the residual pull of the anchor."""
    problem = make_problem(candidate_anchors=(100.0,), evidence={"evidence_estimate": 20.0})
    result = adjust_anchor(problem, Point22Config(adjustment_fraction=0.25))
    assert result["estimate"] == pytest.approx(80.0)
    assert result["anchor_influence_remaining"] == pytest.approx(60.0)

def test_anchor_is_not_used_as_evidence_when_no_external_estimate_exists() -> None:
    """An arbitrary starting value is disclosed but not turned into a verified answer."""
    result = adjust_anchor(make_problem(candidate_anchors=(500.0,), evidence={}), Point22Config())
    assert result["applied"] is False
    assert result["estimate"] is None

def test_small_sample_standard_error_and_interval_are_reported() -> None:
    """The uncertainty interval widens as the sample size becomes smaller."""
    small = estimate_sample_uncertainty(
        make_problem(evidence={"sample_n": 4, "sample_sd": 2.0}), Point22Config())
    large = estimate_sample_uncertainty(
        make_problem(evidence={"sample_n": 100, "sample_sd": 2.0}), Point22Config())
    assert small["standard_error"] == pytest.approx(1.0)
    assert large["standard_error"] == pytest.approx(0.2)
    assert small["standard_error"] > large["standard_error"]
    assert any("Small sample" in item for item in small["warnings"])

def test_binomial_standard_error_uses_sample_proportion_and_effective_n() -> None:
    """A supplied design effect reduces effective sample size."""
    result = estimate_sample_uncertainty(
        make_problem(evidence={"sample_n": 100, "sample_proportion": 0.5, "design_effect": 2.0}),
        Point22Config())
    assert result["effective_sample_size"] == pytest.approx(50.0)
    assert result["standard_error"] == pytest.approx((0.25 / 50.0) ** 0.5)

def test_correlation_is_computed_but_never_reported_as_causation() -> None:
    """Strong correlation without temporal/confounder checks produces explicit caveats."""
    result = check_correlation_causation(make_problem(
        target_type="cause", evidence={"x_values": [1, 2, 3, 4], "y_values": [2, 4, 6, 8]}))
    assert result["correlation"] == pytest.approx(1.0)
    assert result["causal_status"] == "causal_identification_incomplete"
    assert result["association_is_not_causation"] is True
    assert len(result["warnings"]) >= 4

def test_intervention_evidence_changes_causal_status_but_keeps_caveats() -> None:
    """An intervention is a stronger design cue, not a universal guarantee."""
    result = check_correlation_causation(make_problem(target_type="cause", evidence={
        "observational_association": True, "temporal_order_known": True,
        "confounders_examined": True, "reverse_causality_examined": True,
        "intervention_evidence": True,
    }))
    assert result["causal_status"] == "stronger_causal_design_available"
    assert result["warnings"] == ["Observed association alone does not establish a causal effect."]

def test_urgency_and_stakes_raise_verification_priority() -> None:
    """High stakes trigger independent review rather than stronger shortcut confidence."""
    result = Point22CognitiveHeuristicsProcessor(
        Point22Config(simulate_biology=False)
    ).process(make_problem(decision_cost=0.95, time_pressure=0.9))
    assert result.diagnostics["affect_and_urgency"]["verification_priority"] > 0.7
    assert result.diagnostics["conflict"]["verification_triggered"] is True
    assert result.decision["review_required"] is True

def test_fluency_changes_subjective_confidence_not_bayesian_posterior() -> None:
    """Repeated exposure can raise fluency while verified category probabilities stay fixed."""
    base = make_problem(fluency=0.0, evidence={"likelihoods": {"rare": 0.8, "common": 0.15}})
    fluent = make_problem(fluency=1.0, evidence={
        "likelihoods": {"rare": 0.8, "common": 0.15}, "exposure_count": 10})
    processor = Point22CognitiveHeuristicsProcessor(Point22Config(simulate_biology=False))
    first, second = processor.process(base), processor.process(fluent)
    assert first.probabilities == pytest.approx(second.probabilities)
    assert second.diagnostics["fluency"]["subjective_confidence"] > first.diagnostics["fluency"]["subjective_confidence"]

def test_feedback_records_brier_error_and_updates_only_attributed_cues() -> None:
    """Outcome feedback is context-tagged and cue weights change only with cue-level attribution."""
    state = Point22State()
    result = Point22CognitiveHeuristicsProcessor(
        Point22Config(simulate_biology=False), state=state
    ).process(make_problem(evidence={
        "likelihoods": {"rare": 0.8, "common": 0.15}, "observed_outcome": "rare",
        "cue_feedback": {"availability": 0.5},
    }))
    assert result.feedback["brier_score"] is not None
    assert result.feedback["cue_updates"]["availability"]["after"] == pytest.approx(1.05)
    assert result.final_state["prediction_history"]

def test_feedback_does_not_reassign_error_to_cues_without_attribution() -> None:
    """An outcome alone is insufficient to identify which cue caused the error."""
    state = Point22State()
    processor = Point22CognitiveHeuristicsProcessor(Point22Config(simulate_biology=False), state=state)
    before = dict(state.cue_weights)
    result = processor.process(make_problem(evidence={
        "likelihoods": {"rare": 0.8, "common": 0.15}, "observed_outcome": "rare"}))
    assert state.cue_weights == before
    assert result.feedback["cue_weights_updated"] is False

def test_ecological_fit_is_unknown_without_environmental_validity_data() -> None:
    """The package does not assume one heuristic is best across all environments."""
    result = Point22CognitiveHeuristicsProcessor(
        Point22Config(simulate_biology=False)
    ).process(make_problem(cue_validity={}, environment_stability=0.9))
    assert result.ecological_fit["robust_in_supplied_environment"] is None
    assert result.ecological_fit["out_of_sample_evaluation_required"] is True

def test_biological_reference_track_can_be_disabled() -> None:
    """Disabling population-rate dynamics preserves the seventeen functional stages."""
    result = Point22CognitiveHeuristicsProcessor(
        Point22Config(simulate_biology=False)
    ).process(make_problem())
    assert len(result.operation_trace) == 17
    assert result.biological["model"] == "disabled"
    assert all(row["biological_output"]["model"] == "disabled" for row in result.operation_trace)

def test_input_contract_rejects_invalid_probabilities_and_sample_sizes() -> None:
    """Invalid normalized cues or sampling requests fail before judgment is produced."""
    with pytest.raises(ValueError, match="fluency"):
        make_problem(fluency=1.5)
    with pytest.raises(ValueError, match="sample_n"):
        Point22CognitiveHeuristicsProcessor(
            Point22Config(simulate_biology=False)
        ).process(make_problem(evidence={"sample_n": 0, "sample_sd": 1.0}))

def test_sqlite_registers_symbols_and_logs_all_seventeen_operations(tmp_path: Path) -> None:
    """The shared annotations table and runs/events tables persist Point 22 metadata."""
    root = Path(__file__).resolve().parents[1]
    database_path = tmp_path / "ore_logger.sqlite3"
    result = Point22CognitiveHeuristicsProcessor(
        Point22Config(simulate_biology=False)
    ).process(make_problem())
    report = record_result_to_sqlite(result, database_path, repository_root=root)
    assert report["database_summary"]["events"] == 17
    with sqlite3.connect(database_path) as connection:
        annotations = connection.execute(
            "SELECT COUNT(*) FROM annotations WHERE module_name LIKE 'ore_cognitive_heuristics%'"
        ).fetchone()[0]
        events = connection.execute(
            "SELECT COUNT(*) FROM events WHERE run_id=?", (report["run_id"],)).fetchone()[0]
        status = connection.execute(
            "SELECT status FROM runs WHERE run_id=?", (report["run_id"],)).fetchone()[0]
    assert annotations >= 20
    assert events == 17
    assert status == "completed"


def test_step_seventeen_persists_context_and_both_judgment_tracks_without_feedback() -> None:
    """The final operation stores context, assumptions, initial and verified outputs."""
    state = Point22State()
    result = Point22CognitiveHeuristicsProcessor(
        Point22Config(simulate_biology=False), state=state
    ).process(make_problem(evidence={"likelihoods": {"rare": 0.8, "common": 0.15}}))
    stored = result.final_state["judgment_history"][-1]
    assert stored["problem_id"] == "rare-pattern"
    assert stored["question"] == "Which explanation best fits the observed case?"
    assert stored["initial_answer"] == result.initial_answer
    assert stored["verified_answer"] == result.verified_answer
    assert stored["assumptions"]
    assert result.operation_trace[-1]["functional_output"]["stored_judgment"] == stored


def test_magnitude_judgment_uses_numeric_anchor_for_initial_answer_and_evidence_for_verified() -> None:
    """Magnitude tasks expose the anchor-shifted intuition beside the evidence estimate."""
    problem = make_problem(
        target_type="magnitude",
        candidate_anchors=(100.0,),
        evidence={"evidence_estimate": 20.0},
    )
    result = Point22CognitiveHeuristicsProcessor(
        Point22Config(simulate_biology=False, adjustment_fraction=0.25)
    ).process(problem)
    assert result.initial_answer == pytest.approx(80.0)
    assert result.verified_answer == pytest.approx(20.0)

def test_annotation_registration_is_scoped_to_point22_package(tmp_path: Path) -> None:
    """Manual annotation synchronization uses the shared repository catalogue."""
    root = Path(__file__).resolve().parents[1]
    report = register_annotations(tmp_path / "catalog.sqlite3", repository_root=root)
    assert report["scope"] == "ore_cognitive_heuristics"
    assert report["annotation_table"] == "annotations"
