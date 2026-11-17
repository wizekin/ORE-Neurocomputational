"""Acceptance tests for the Point 10 evidence-based validation workflow."""
from __future__ import annotations

import inspect
import shutil
from pathlib import Path

import numpy as np
import pytest

from ore_language_comprehension import (
    POINT10_METHODS,
    POINT10_STEP_SPECS,
    Point10EvidenceValidator,
    Point10Request,
    Point10Result,
    register_point10_annotations,
)
from ore_language_comprehension import point10
from ore_logger import LoggerDatabase


def request(**overrides):
    """Create a minimal, explicitly falsifiable research-validation request."""
    values = {
        "mechanism": {
            "name": "lexical_semantic_access",
            "operation": "retrieve lexical-semantic information after word-form recognition",
            "predictions": ["semantic constraint changes behavior and neural response"],
            "alternatives": ["general difficulty", "working-memory load"],
        },
        "study_plan": {
            "target_condition": "high semantic constraint",
            "control_condition": "low semantic constraint",
            "counterbalanced": True,
            "confounds_controlled": ["word frequency", "length", "response format"],
            "participant_and_item_effects": True,
            "multiple_comparison_control": True,
        },
        "datasets": {},
        "metadata": {"study_id": "synthetic-01"},
    }
    values.update(overrides)
    return Point10Request(**values)


def test_point10_has_fifteen_ordered_research_operations():
    assert len(POINT10_STEP_SPECS) == 15
    assert [item["operation_id"] for item in POINT10_STEP_SPECS] == [
        f"LC-P10-{i:02d}" for i in range(1, 16)
    ]


def test_validation_runs_all_fifteen_steps_and_returns_typed_result():
    result = Point10EvidenceValidator().process(request())
    assert isinstance(result, Point10Result)
    assert len(result.trace) == 15
    assert result.diagnostics["research_analysis_not_brain_simulation"] is True


def test_unprovided_evidence_is_not_fabricated():
    result = Point10EvidenceValidator().process(request())
    assert result.report["confidence_label"] == "insufficient_evidence"
    assert result.evidence["fmri"]["available"] is False
    assert result.evidence["lesion"]["available"] is False


def test_behavioral_accuracy_rt_and_effect_sizes_are_summarized():
    result = Point10EvidenceValidator().process(request(datasets={
        "behavior": {
            "target": {"accuracy": [1, 1, 1, 0], "response_time": [500, 520, 510, 530]},
            "control": {"accuracy": [1, 0, 0, 0], "response_time": [600, 620, 610, 630]},
        }
    }))
    behavior = result.evidence["behavior"]
    assert behavior["accuracy_difference"] == pytest.approx(0.5)
    assert behavior["response_time_difference"] == pytest.approx(-100.0)
    assert behavior["accuracy_effect_size_d"] is not None


def test_fmri_evidence_is_association_not_necessity():
    result = Point10EvidenceValidator().process(request(datasets={
        "fmri": {"target": [2.0, 2.1, 2.2], "control": [1.0, 1.1, 1.2],
                "independent_roi": True, "motion_controlled": True,
                "multiple_comparison_control": True}
    }))
    assert result.evidence["fmri"]["contrast"] == pytest.approx(1.0)
    assert "not necessity" in result.evidence["fmri"]["inference"]


def test_fmri_flags_circular_roi_and_missing_motion_or_multiplicity_controls():
    result = Point10EvidenceValidator().process(request(datasets={
        "fmri": {"target": [1, 2], "control": [0, 1]}
    }, study_plan={"target_condition": "A", "control_condition": "B",
                   "roi_selected_from_test_contrast": True}))
    flags = result.quality_control["flags"]
    assert "fMRI_ROI_independence_not_documented" in flags
    assert "fMRI_motion_control_not_documented" in flags
    assert "circular_ROI_selection_possible" in flags


def test_eeg_insufficient_trial_count_is_flagged():
    result = Point10EvidenceValidator(minimum_temporal_trials=10).process(request(datasets={
        "eeg": {"target": [1, 2, 3], "control": [0, 1, 2],
                "target_trial_count": 3, "control_trial_count": 3,
                "artifact_controlled": True, "pre_registered_window": True}
    }))
    assert result.evidence["eeg"]["sufficient_trials_by_config"] is False
    assert "eeg_trial_count_below_configured_minimum" in result.quality_control["flags"]


def test_meg_and_eeg_are_reported_as_distinct_methods():
    result = Point10EvidenceValidator().process(request(datasets={
        "eeg": {"target": [1, 2], "control": [0, 1]},
        "meg": {"target": [3, 4], "control": [2, 3]},
    }))
    assert result.evidence["eeg"]["method"] == "EEG/ERP"
    assert result.evidence["meg"]["method"] == "MEG"


def test_lesion_analysis_requires_matched_numeric_arrays_and_keeps_caveats():
    result = Point10EvidenceValidator().process(request(datasets={
        "lesion": {"lesion_feature": [0, 1, 2, 3], "performance": [1, .8, .5, .2],
                   "lesion_volume_controlled": True, "disconnection_assessed": True,
                   "non_language_abilities_assessed": True}
    }))
    assert result.evidence["lesion"]["lesion_performance_correlation"] == pytest.approx(-0.99591, abs=1e-5)
    assert "network effects" in result.evidence["lesion"]["interpretation"]


def test_lesion_covariate_safeguards_are_flagged():
    result = Point10EvidenceValidator().process(request(datasets={
        "lesion": {"lesion_feature": [0, 1, 2], "performance": [1, .5, .2]}
    }))
    assert "lesion_volume_not_controlled" in result.quality_control["flags"]
    assert "lesion_disconnection_not_assessed" in result.quality_control["flags"]


def test_stimulation_compares_active_and_sham_but_does_not_assume_focality():
    result = Point10EvidenceValidator().process(request(datasets={
        "stimulation": {"active_response_time": [450, 470], "sham_response_time": [500, 520],
                        "active_accuracy": [.9, 1], "sham_accuracy": [.9, .9],
                        "sham_or_control_site": True, "non_specific_effects_checked": True,
                        "safety_protocol_documented": True}
    }))
    assert result.evidence["stimulation"]["response_time_difference"] == pytest.approx(-50)
    assert "not a uniquely localized" in result.evidence["stimulation"]["interpretation"]


def test_stimulation_missing_controls_are_flagged():
    result = Point10EvidenceValidator().process(request(datasets={
        "stimulation": {"active_response_time": [1, 2], "sham_response_time": [2, 3]}
    }))
    assert "stimulation_sham_or_control_site_missing" in result.quality_control["flags"]
    assert "stimulation_safety_protocol_not_documented" in result.quality_control["flags"]


def test_named_competing_explanations_are_preserved_and_marked_unresolved():
    result = Point10EvidenceValidator().process(request())
    assert result.alternatives["general difficulty"]["status"] == "unresolved"
    assert result.alternatives["working-memory load"]["status"] == "unresolved"


def test_supplied_alternative_evidence_is_not_silently_ignored():
    result = Point10EvidenceValidator().process(request(datasets={
        "alternative_evidence": {"general difficulty": {"matched_difficulty": True}}
    }))
    assert result.alternatives["general difficulty"]["status"] == "evaluated_from_supplied_evidence"


def test_evidence_from_same_independence_group_is_not_double_counted():
    result = Point10EvidenceValidator().process(request(datasets={
        "behavior": {"target": {"accuracy": [1, 1]}, "control": {"accuracy": [0, 1]},
                     "independence_group": "cohort-A"},
        "fmri": {"target": [2, 3], "control": [1, 2], "independence_group": "cohort-A"},
    }))
    assert result.synthesis["double_counting_warning"] is True
    assert result.synthesis["independent_source_group_count"] == 1


def test_distinct_source_groups_are_counted_separately():
    result = Point10EvidenceValidator().process(request(datasets={
        "behavior": {"target": {"accuracy": [1, 1]}, "control": {"accuracy": [0, 1]},
                     "independence_group": "cohort-A"},
        "fmri": {"target": [2, 3], "control": [1, 2], "independence_group": "cohort-B"},
    }))
    assert result.synthesis["independent_source_group_count"] == 2
    assert result.synthesis["double_counting_warning"] is False


def test_held_out_predictions_compute_error_metrics_and_check_freezing():
    result = Point10EvidenceValidator().process(request(datasets={
        "held_out": {"observed": [1, 2, 3], "predicted": [1, 2, 2],
                     "model_was_frozen": True, "independent_test_set": True}
    }))
    assert result.generalization["available"] is True
    assert result.generalization["mse"] == pytest.approx(1 / 3)
    assert result.generalization["model_was_frozen"] is True


def test_held_out_mismatched_arrays_are_not_scored():
    result = Point10EvidenceValidator().process(request(datasets={
        "held_out": {"observed": [1, 2], "predicted": [1]}
    }))
    assert result.generalization["available"] is False


def test_missing_operational_definition_predictions_and_alternatives_are_flagged():
    result = Point10EvidenceValidator().process(request(mechanism={"name": "incomplete", "operation": ""}))
    assert result.diagnostics["valid"] is False
    assert "target_operation_not_operationally_defined" in result.quality_control["flags"]
    assert "no_falsifiable_predictions" in result.quality_control["flags"]


def test_null_results_and_multiple_comparisons_are_not_overinterpreted():
    result = Point10EvidenceValidator().process(request(study_plan={
        "target_condition": "A", "control_condition": "B", "multiple_comparisons_ignored": True,
        "null_result_claimed_as_absence": True,
    }))
    assert "multiple_comparisons_ignored" in result.quality_control["flags"]
    assert "null_result_overinterpreted_as_proof_of_absence" in result.quality_control["flags"]


def test_result_serializes_to_plain_data():
    result = Point10EvidenceValidator().process(request())
    plain = result.to_dict()
    assert plain["trace"][0]["step"] == 1
    assert isinstance(plain["trace"], list)


def test_request_snapshots_input_mappings():
    mechanism = {"name": "m", "operation": "op", "predictions": ["p"], "alternatives": ["a"]}
    req = Point10Request(mechanism, {"target_condition": "A", "control_condition": "B"})
    mechanism["operation"] = "mutated"
    assert req.mechanism["operation"] == "op"


def test_all_public_point10_functions_and_classes_have_docstrings():
    public = [obj for name, obj in vars(point10).items()
              if not name.startswith("_") and (inspect.isclass(obj) or inspect.isfunction(obj))]
    assert public
    assert all(inspect.getdoc(obj) for obj in public)


def test_annotation_registration_uses_shared_sqlite_catalogue(tmp_path):
    source_root = Path(__file__).resolve().parents[1]
    isolated = tmp_path / "ore-project"
    isolated.mkdir()
    shutil.copytree(source_root / "ore_logger", isolated / "ore_logger")
    shutil.copytree(source_root / "ore_language_comprehension", isolated / "ore_language_comprehension")
    database_path = tmp_path / "point10-annotations.sqlite3"

    report = register_point10_annotations(isolated, database_path)
    rows = LoggerDatabase(database_path).search_annotations("ore_language_comprehension", limit=2500)

    assert report["symbols_written"] > 0
    assert report["database_path"] == str(database_path)
    assert any(row["qualified_name"] == "Point10EvidenceValidator.process" for row in rows)
    assert any(row["qualified_name"] == "Point10Request.__post_init__" for row in rows)
    assert all(row["source_docstring"] for row in rows)
