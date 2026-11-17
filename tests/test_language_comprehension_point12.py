"""Acceptance tests for Point 12 methodological model evaluation."""
from __future__ import annotations

import inspect
import shutil
from pathlib import Path

import pytest

from ore_language_comprehension import (
    POINT12_METHODS, POINT12_STEP_SPECS, Point12ModelEvaluator, Point12Request,
    Point12Result, register_point12_annotations,
)
from ore_language_comprehension import point12
from ore_logger import LoggerDatabase


def make_request(**overrides):
    """Build a minimal falsifiable research request with explicit observable outcome."""
    values = {
        "target_phenomenon": "sarcasm interpretation",
        "observable_target": {"measure": "interpretation_choice", "separates_comprehension_from_response": True},
        "candidate_hypotheses": [
            {"name": "contextual_revision", "operation": "revise literal meaning using discourse",
             "predictions": ["context changes interpretation"], "falsifiers": ["context manipulation has no effect"]},
            {"name": "general_conflict", "operation": "resolve response conflict",
             "predictions": ["conflict predicts response"], "falsifiers": ["matched non-social conflict has no effect"]},
        ],
        "conditions": {}, "behavioral_data": {}, "neural_data": {},
        "participant_metadata": [], "analysis_plan": {"preregistered": True, "multiple_comparison_control": True},
        "metadata": {"study_id": "synthetic-study"},
    }
    values.update(overrides)
    return Point12Request(**values)


def test_point12_registers_25_source_ordered_operations():
    assert len(POINT12_STEP_SPECS) == 25
    assert [s["operation_id"] for s in POINT12_STEP_SPECS] == [f"LC-P12-{i:02d}" for i in range(1, 26)]


def test_process_returns_all_25_steps_and_typed_research_result():
    result = Point12ModelEvaluator().process(make_request())
    assert isinstance(result, Point12Result)
    assert [step["step"] for step in result.trace] == list(range(1, 26))
    assert result.diagnostics["research_analysis_not_brain_simulation"] is True


def test_missing_empirical_data_does_not_invent_evidence_or_rank_hypotheses():
    result = Point12ModelEvaluator().process(make_request())
    assert result.ranked_hypotheses == []
    assert result.evidence_by_method["fmri"]["available"] is False
    assert result.uncertainty_report["missing_evidence_is_not_evidence_of_absence"] is True


def test_causal_graph_cycle_is_detected_and_reported():
    result = Point12ModelEvaluator().process(make_request(analysis_plan={
        "causal_edges": [["context", "interpretation"], ["interpretation", "context"]]
    }))
    assert result.confound_audit["causal_graph"]["cycle_detected"] is True
    assert "causal_graph_contains_cycle" in result.uncertainty_report["quality_control_flags"]


def test_target_control_feature_mismatch_flags_potential_confound():
    result = Point12ModelEvaluator().process(make_request(
        conditions={"target": {"length": [10, 11], "social": [1, 1]},
                    "control": {"length": [5, 6], "social": [0, 0]}},
        analysis_plan={"manipulated_variables": ["social"], "matched_variables": []}))
    assert "length" in result.design_audit["potentially_confounded_features"]


def test_behavior_and_neural_measurements_remain_distinct():
    result = Point12ModelEvaluator().process(make_request(
        behavioral_data={"target": {"accuracy": [1, 1, 0], "response_time": [400, 420, 410]},
                         "control": {"accuracy": [1, 0, 0], "response_time": [500, 520, 510]}},
        neural_data={"fmri": {"target": [2, 3], "control": [1, 1],
                              "independent_localizer": True, "motion_controlled": True,
                              "multiple_comparison_control": True}}))
    assert result.evidence_by_method["behavior"]["accuracy_difference"] == pytest.approx(1/3)
    assert result.evidence_by_method["fmri"]["contrast"] == pytest.approx(1.5)
    assert "does not localize" in result.evidence_by_method["behavior"]["interpretation"]


def test_erp_labels_are_not_treated_as_unique_cognitive_processes():
    result = Point12ModelEvaluator(minimum_temporal_trials=5).process(make_request(
        neural_data={"eeg": {"target": [1, 2], "control": [0, 1],
                             "target_trial_count": 2, "control_trial_count": 2,
                             "component_label": "N400", "artifact_controlled": True}}))
    assert result.trace[6]["result"]["component_is_unique_process_label"] is False
    assert result.evidence_by_method["eeg"]["sufficient_trials"] is False


def test_language_selectivity_is_a_contrast_not_absolute_exclusivity():
    result = Point12ModelEvaluator().process(make_request(
        neural_data={"task_response_profile": {"language": [3, 4], "non_language_control": [1, 2]}}))
    assert result.trace[7]["result"]["selectivity_contrast"] == pytest.approx(2.0)
    assert result.trace[7]["result"]["absolute_exclusivity_claimed"] is False


def test_hypotheses_are_ranked_only_with_comparable_explicit_likelihoods():
    candidates = [
        {"name": "A", "predictions": ["p"], "falsifiers": ["f"], "prior_probability": .5,
         "likelihoods": {"dataset-1": .8}},
        {"name": "B", "predictions": ["q"], "falsifiers": ["g"], "prior_probability": .5,
         "likelihoods": {"dataset-1": .2}},
    ]
    result = Point12ModelEvaluator().process(make_request(candidate_hypotheses=candidates))
    assert result.ranked_hypotheses == ["A", "B"]
    assert result.trace[14]["result"]["posterior_probabilities"]["A"] == pytest.approx(.8)


def test_incomparable_likelihoods_do_not_produce_posterior():
    candidates = [
        {"name": "A", "predictions": ["p"], "falsifiers": ["f"], "likelihoods": {"d1": .8}},
        {"name": "B", "predictions": ["q"], "falsifiers": ["g"], "likelihoods": {"d2": .2}},
    ]
    result = Point12ModelEvaluator().process(make_request(candidate_hypotheses=candidates))
    assert result.ranked_hypotheses == []
    assert result.trace[14]["result"]["available"] is False


def test_held_out_metrics_require_frozen_model_and_independent_test_set_for_claim():
    result = Point12ModelEvaluator().process(make_request(analysis_plan={
        "held_out_observed": [1, 2, 3], "held_out_predicted": [1, 2, 2],
        "model_was_frozen": True, "independent_test_set": True}))
    assert result.generalization_report["mse"] == pytest.approx(1/3)
    assert result.generalization_report["mae"] == pytest.approx(1/3)
    assert result.generalization_report["generalization_supported"] is True


def test_request_snapshots_mutable_inputs():
    hypotheses = [{"name": "A", "predictions": ["p"], "falsifiers": ["f"]}]
    request = make_request(candidate_hypotheses=hypotheses)
    hypotheses[0]["name"] = "mutated"
    assert request.candidate_hypotheses[0]["name"] == "A"


def test_invalid_request_requires_at_least_one_candidate():
    with pytest.raises(ValueError):
        make_request(candidate_hypotheses=[])


def test_all_public_point12_classes_and_functions_have_docstrings():
    public = [obj for name, obj in vars(point12).items()
              if not name.startswith("_") and (inspect.isclass(obj) or inspect.isfunction(obj))]
    assert public
    assert all(inspect.getdoc(obj) for obj in public)


def test_annotation_registration_uses_shared_sqlite_catalogue(tmp_path):
    source_root = Path(__file__).resolve().parents[1]
    isolated = tmp_path / "ore-project"
    isolated.mkdir()
    shutil.copytree(source_root / "ore_logger", isolated / "ore_logger")
    shutil.copytree(source_root / "ore_language_comprehension", isolated / "ore_language_comprehension")
    shutil.copytree(source_root / "ore_brain_networks", isolated / "ore_brain_networks")
    (isolated / "pyproject.toml").write_text("[project]\nname='point12-test'\n", encoding="utf-8")
    database_path = tmp_path / "point12.sqlite3"
    report = register_point12_annotations(isolated, database_path)
    rows = LoggerDatabase(database_path).search_annotations("ore_language_comprehension", limit=3000)
    assert report["symbols_written"] > 0
    assert any(row["qualified_name"] == "Point12ModelEvaluator.process" for row in rows)
    assert any(row["qualified_name"] == "Point12Request.__post_init__" for row in rows)
    assert all(row["source_docstring"] for row in rows)
