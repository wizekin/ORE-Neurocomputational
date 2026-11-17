"""Acceptance tests for Point 13 individualized language-comprehension profiling."""
from __future__ import annotations

import inspect
import shutil
from pathlib import Path

import pytest

from ore_language_comprehension import (
    POINT13_COMPONENTS,
    POINT13_STEP_SPECS,
    Point13IndividualizedProcessor,
    Point13Request,
    Point13Result,
    register_point13_annotations,
)
from ore_language_comprehension import point13
from ore_logger import LoggerDatabase


def make_request(**overrides):
    """Build a synthetic multi-domain request with measured and missing components."""
    values = {
        "participant_id": "participant-01",
        "target_operation": "indirect-request-comprehension",
        "stimulus": {"modality": "auditory", "language": "en"},
        "discourse_context": {"scene": "open-window"},
        "task_requirements": {
            "pragmatic_operations": ["indirect_request", "belief_inference"],
            "length": 4,
            "dependency_load": 2,
            "competitor_count": 1,
            "revision_demand": 1,
        },
        "observations": [
            {"response_time": 500, "accuracy": 1, "confidence": .8, "correct": 1,
             "session_id": "s1", "state": "alert", "cue_weight_estimates": {"context": .7},
             "candidate_scores": {"close_window": 2, "temperature_only": 1},
             "held_out_observed": 1, "held_out_predicted": .9,
             "model_frozen": True, "independent_test_set": True},
            {"response_time": 550, "accuracy": 1, "confidence": .7, "correct": 1,
             "session_id": "s1", "state": "alert", "cue_weight_estimates": {"context": .8},
             "candidate_scores": {"close_window": 3, "temperature_only": 0},
             "held_out_observed": 0, "held_out_predicted": .2,
             "model_frozen": True, "independent_test_set": True},
            {"response_time": 650, "accuracy": 0, "confidence": .6, "correct": 0,
             "session_id": "s2", "state": "fatigued", "interference_cost": 100},
        ],
        "component_measures": {
            "lexical_access_efficiency": {"values": [.7, .8, .75], "quality": "good"},
            "vocabulary_knowledge": [.8, .9, .85],
            "perceptual_reliability": [.9, .8, .85],
            "working_memory_support": [.6, .7, .65],
            "inhibitory_control": [.4, .5, .45],
            "world_knowledge": [.8, .9],
            "pragmatic_perspective_taking": [.5, .6],
            "contextual_cue_use": [.7, .8],
            "reanalysis_efficiency": [.4, .5],
            "pragmatic_tasks": {"indirect_request": [.7, .8]},
        },
        "experience_profile": {
            "chronological_age": 28,
            "languages": {
                "English": {"age_of_acquisition": 7, "exposure_hours": 3000,
                            "proficiency": {"overall": .8}, "recency_of_use": "daily"},
                "Arabic": {"age_of_acquisition": 0, "exposure_hours": 20000,
                           "proficiency": {"overall": .95}, "recency_of_use": "daily"},
            },
            "domain_experience": {"window_context": "familiar"},
        },
        "baseline_conditions": {
            "low_context": {"response_time": [700, 720], "accuracy": [0, 1]},
            "high_context": {"response_time": [550, 570], "accuracy": [1, 1]},
        },
        "longitudinal_measures": [
            {"component": "vocabulary_knowledge", "time": 1, "score": .4},
            {"component": "vocabulary_knowledge", "time": 2, "score": .6},
            {"component": "vocabulary_knowledge", "time": 3, "score": .8},
        ],
        "cohort_observations": [
            {"effect": .2, "age_group": "young", "language": "English"},
            {"effect": .5, "age_group": "older", "language": "English"},
        ],
        "candidate_strategies": [
            {"strategy_id": "lexical-first", "held_out_observed": [1, 0],
             "held_out_predicted": [.8, .2], "complexity": 2, "penalty_weight": .01},
            {"strategy_id": "context-supported", "held_out_observed": [1, 0],
             "held_out_predicted": [.9, .1], "complexity": 3, "penalty_weight": .01},
        ],
        "previous_profile": {"components": {"lexical_access_efficiency": .5}},
        "metadata": {"learning_rate": .5},
    }
    values.update(overrides)
    return Point13Request(**values)


def test_point13_exposes_18_source_ordered_steps():
    assert len(POINT13_STEP_SPECS) == 18
    assert [item["operation_id"] for item in POINT13_STEP_SPECS] == [
        f"LC-P13-{index:02d}" for index in range(1, 19)
    ]


def test_process_returns_complete_profile_and_18_step_trace():
    result = Point13IndividualizedProcessor().process(make_request())
    assert isinstance(result, Point13Result)
    assert len(result.trace) == 18
    assert result.diagnostics["step_count"] == 18
    assert result.diagnostics["clinical_diagnosis_not_performed"] is True
    assert set(result.profile["components"]) == set(POINT13_COMPONENTS)


def test_contextual_facilitation_uses_matched_low_and_high_context():
    result = Point13IndividualizedProcessor().process(make_request())
    assert result.profile["contextual_facilitation"]["response_time_benefit"] == pytest.approx(150)
    assert result.profile["contextual_facilitation"]["accuracy_benefit"] == pytest.approx(.5)
    assert result.trace[5]["result"]["core_efficiency_and_context_benefit_are_distinct"] is True


def test_component_abilities_are_separate_and_missing_is_not_zero():
    request = make_request(component_measures={"lexical_access_efficiency": [.4, .6]})
    result = Point13IndividualizedProcessor().process(request)
    assert result.profile["components"]["lexical_access_efficiency"] == pytest.approx(.5)
    assert result.profile["components"]["vocabulary_knowledge"] is None
    assert "vocabulary_knowledge" in result.uncertainty_estimates["missing_components"]


def test_language_profiles_keep_proficiency_separate_by_language():
    result = Point13IndividualizedProcessor().process(make_request())
    languages = result.profile["language_experience"]["languages"]
    assert set(languages) == {"English", "Arabic"}
    assert languages["English"]["proficiency"]["overall"] == .8
    assert languages["Arabic"]["proficiency"]["overall"] == .95


def test_cue_weights_and_candidate_softmax_are_normalized():
    result = Point13IndividualizedProcessor().process(make_request())
    cue = result.profile["cue_profile"]
    assert cue["cue_weight_estimates"]["context"] == pytest.approx(.75)
    assert sum(cue["candidate_probabilities"].values()) == pytest.approx(1.0)


def test_developmental_trajectory_requires_distinct_measurement_times():
    result = Point13IndividualizedProcessor().process(make_request())
    slope = result.profile["developmental_trajectories"]["component_trajectories"][
        "vocabulary_knowledge"
    ]
    assert slope["available"] is True
    assert slope["slope"] == pytest.approx(.2)


def test_pragmatic_operations_do_not_collapse_to_one_theory_of_mind_score():
    result = Point13IndividualizedProcessor().process(make_request())
    pragmatic = result.profile["pragmatic_profile"]
    assert "indirect_request" in pragmatic["required_operations"]
    assert pragmatic["theory_of_mind_is_not_assumed_for_every_pragmatic_task"] is True


def test_aging_profile_does_not_infer_uniform_aging_from_age():
    result = Point13IndividualizedProcessor().process(make_request())
    aging = result.profile["aging_and_interference_profile"]
    assert aging["aging_relevance_explicitly_supplied"] is False
    assert aging["no_uniform_aging_profile_assumed"] is True


def test_state_and_group_variation_are_preserved():
    result = Point13IndividualizedProcessor().process(make_request())
    assert result.profile["state_stability"]["repeated_sessions"] == 2
    assert result.profile["group_heterogeneity"]["participant_summary"]["n"] == 2


def test_strategy_comparison_uses_held_out_predictions():
    result = Point13IndividualizedProcessor().process(make_request())
    ranked = result.profile["strategy_comparison"]["ranked_strategy_ids"]
    assert ranked[0] == "context-supported"


def test_adaptation_is_reversible_and_profile_update_is_bounded():
    result = Point13IndividualizedProcessor().process(make_request())
    assert result.adapted_parameters["adaptation_is_reversible"] is True
    assert result.profile["components"]["lexical_access_efficiency"] == pytest.approx(.625)
    assert result.trace[17]["result"]["learning_rate"] == .5


def test_calibration_requires_observations_and_is_reported_separately():
    result = Point13IndividualizedProcessor().process(make_request())
    calibration = result.profile["calibration"]
    assert calibration["calibration_n"] == 3
    assert calibration["held_out_mse"] == pytest.approx(.025)
    assert calibration["generalization_supported"] is True


def test_empty_evidence_preserves_uncertainty_without_fabricating_profile():
    request = make_request(
        observations=[],
        component_measures={},
        experience_profile={},
        baseline_conditions={},
        candidate_strategies=[],
    )
    result = Point13IndividualizedProcessor().process(request)
    assert result.unresolved_hypotheses
    assert result.profile["contextual_facilitation"]["available"] is False
    assert result.profile["components"]["lexical_access_efficiency"] is None


def test_request_deep_copies_nested_observations():
    observations = [{"response_time": 500}]
    request = make_request(observations=observations)
    observations[0]["response_time"] = 1
    assert request.observations[0]["response_time"] == 500


def test_invalid_request_requires_participant_and_operation():
    with pytest.raises(ValueError):
        make_request(participant_id="")
    with pytest.raises(ValueError):
        make_request(target_operation="")


def test_public_point13_definitions_have_docstrings():
    public = [
        value for name, value in vars(point13).items()
        if not name.startswith("_") and (inspect.isclass(value) or inspect.isfunction(value))
    ]
    assert public
    assert all(inspect.getdoc(value) for value in public)


def test_annotation_registration_uses_shared_sqlite_catalogue(tmp_path):
    source_root = Path(__file__).resolve().parents[1]
    isolated = tmp_path / "ore-project"
    isolated.mkdir()
    for package in ("ore_logger", "ore_language_comprehension", "ore_brain_networks"):
        shutil.copytree(source_root / package, isolated / package)
    (isolated / "pyproject.toml").write_text(
        "[project]\nname='point13-test'\n", encoding="utf-8"
    )
    database_path = tmp_path / "point13.sqlite3"
    report = register_point13_annotations(isolated, database_path)
    rows = LoggerDatabase(database_path).search_annotations(
        "ore_language_comprehension", limit=4000
    )
    assert report["symbols_written"] > 0
    assert any(row["qualified_name"] == "Point13IndividualizedProcessor.process" for row in rows)
    assert any(row["qualified_name"] == "Point13Request.__post_init__" for row in rows)
    assert all(row["source_docstring"] for row in rows)
