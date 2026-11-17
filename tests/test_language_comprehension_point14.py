"""Acceptance tests for the 19-step Point 14 clinical/research profile."""
from __future__ import annotations

import inspect
import shutil
from pathlib import Path

import pytest

from ore_language_comprehension import (
    POINT14_COMPONENTS,
    POINT14_PROFILE_FAMILIES,
    POINT14_STEP_SPECS,
    Point14ClinicalProfileProcessor,
    Point14Request,
    Point14Result,
    register_point14_annotations,
)
from ore_language_comprehension import point14
from ore_logger import LoggerDatabase


def make_request(**overrides):
    """Create synthetic matched tasks spanning literal, inference, discourse, and support."""
    values = {
        "participant_id": "research-participant-14",
        "target_outcome": "understanding-implied-speaker-intention",
        "clinical_contexts": ["right_hemisphere", "autism_related", "psychosis_spectrum",
                             "neurodegenerative", "aphasia"],
        "observations": [
            {"task_id": "literal", "task_type": "literal", "meaning_level": "literal",
             "accuracy": .9, "memory_load": 1, "demands": {"lexical_access": .2}},
            {"task_id": "inference", "task_type": "inference", "meaning_level": "inference",
             "accuracy": .6, "memory_load": 2, "inference_domain": "social",
             "speaker_knowledge_explicit": True, "cue_weights": {"prosody": .4, "context": .8},
             "demands": {"pragmatic_inference": .9, "social_state_inference": .8}},
            {"task_id": "physical", "task_type": "causal", "accuracy": .8, "memory_load": 3,
             "inference_domain": "nonsocial", "speaker_knowledge_explicit": False,
             "cue_weights": {"context": .6}, "demands": {"world_knowledge": .5}},
            {"task_id": "baseline", "task_type": "support-test", "accuracy": .5,
             "pair_id": "matched-1", "support_present": False},
            {"task_id": "supported", "task_type": "support-test", "accuracy": .8,
             "pair_id": "matched-1", "support_present": True, "support_generalization": True},
            {"task_id": "discourse-graph", "task_type": "discourse", "accuracy": .7,
             "discourse_graph": {"required_edges": [("a", "b"), ("b", "c")],
                                 "produced_edges": [("a", "b"), ("x", "y")]}},
            {"task_id": "rhd-sentence", "task_type": "sentence", "accuracy": .9,
             "profile_family": "right_hemisphere"},
            {"task_id": "rhd-discourse", "task_type": "discourse", "accuracy": .5,
             "profile_family": "right_hemisphere"},
            {"task_id": "autism-irony", "task_type": "irony", "accuracy": .6,
             "profile_family": "autism_related", "communication_preferences": ["written context"]},
            {"task_id": "psychosis-understand", "task_type": "literal", "accuracy": .8,
             "profile_family": "psychosis_spectrum", "response_mode": "comprehension"},
            {"task_id": "psychosis-produce", "task_type": "narrative", "accuracy": .4,
             "profile_family": "psychosis_spectrum", "response_mode": "production"},
            {"task_id": "nd-task", "task_type": "semantic", "accuracy": .7,
             "profile_family": "neurodegenerative"},
            {"task_id": "aphasia-language", "task_type": "matched", "accuracy": .4,
             "profile_family": "aphasia", "presentation": "linguistic"},
            {"task_id": "aphasia-nonlanguage", "task_type": "matched", "accuracy": .8,
             "profile_family": "aphasia", "presentation": "nonlinguistic"},
            {"task_id": "held-out", "task_type": "validation", "accuracy": 1,
             "predicted_probability": .8, "split": "validation"},
        ],
        "component_measures": {
            "perceptual_access": [.9, .8],
            "lexical_access": [.8, .7],
            "syntactic_parsing": [.9],
            "semantic_composition": [.7, .8],
            "working_memory": [.6, .5],
            "discourse_coherence": [.5, .6],
            "pragmatic_inference": [.5, .6],
            "social_state_inference": [.6],
            "world_knowledge": [.8, .9],
            "executive_control": [.5, .6],
        },
        "candidate_explanations": [
            {"hypothesis_id": "cue-integration", "description": "Cue weighting hypothesis",
             "predictions": {"literal": .9, "inference": .5, "held-out": .8}, "complexity": 2},
            {"hypothesis_id": "uniform-performance", "description": "Uniform response hypothesis",
             "predictions": {"literal": .5, "inference": .5}, "complexity": 1},
        ],
        "longitudinal_observations": [
            {"component": "semantic_composition", "time": 1, "score": .8},
            {"component": "semantic_composition", "time": 2, "score": .7},
            {"component": "semantic_composition", "time": 3, "score": .6},
            {"component": "working_memory", "time": 1, "score": .6},
        ],
        "previous_profile": {"components": {"semantic_composition": .9, "lexical_access": .4}},
        "metadata": {"learning_rate": .5},
    }
    values.update(overrides)
    return Point14Request(**values)


def test_point14_has_19_source_ordered_steps():
    assert len(POINT14_STEP_SPECS) == 19
    assert len(POINT14_PROFILE_FAMILIES) == 5
    assert [item[0] for item in POINT14_STEP_SPECS] == [
        "define_outcome", "task_demands", "basic_access", "literal_inference",
        "cue_integration", "social_nonsocial", "discourse_coherence",
        "memory_executive_load", "right_hemisphere_profile", "autism_profile",
        "psychosis_profile", "neurodegeneration_profile", "aphasia_profile",
        "component_profile", "competing_models", "support_conditions",
        "independent_validation", "longitudinal_update", "transparent_report",
    ]


def test_process_returns_auditable_result_and_19_step_trace():
    result = Point14ClinicalProfileProcessor().process(make_request())
    assert isinstance(result, Point14Result)
    assert len(result.trace) == 19
    assert result.diagnostics["step_count"] == 19
    assert result.diagnostics["clinical_diagnosis_not_performed"] is True
    assert [entry["operation_id"] for entry in result.trace] == [
        f"LC-P14-{index:02d}" for index in range(1, 20)
    ]
    assert all(entry["functional_operation"] for entry in result.trace)
    assert all(entry["possible_biological_implementation"] for entry in result.trace)


def test_literal_inference_cost_and_social_nonsocial_are_separate():
    result = Point14ClinicalProfileProcessor().process(make_request())
    assert result.profile["literal_vs_inferred_meaning"]["inference_cost"] == pytest.approx(.3)
    inference = result.profile["social_vs_nonsocial_inference"]
    assert inference["social_inference"] == pytest.approx(.6)
    assert inference["nonsocial_inference"] == pytest.approx(.8)
    assert inference["theory_of_mind_not_assumed_for_all_pragmatics"] is True


def test_cue_weights_are_normalized_and_missing_cues_stay_empty():
    result = Point14ClinicalProfileProcessor().process(make_request())
    cue = result.profile["cue_integration"]["normalized_cue_weights"]
    assert sum(cue.values()) == pytest.approx(1.0)
    empty = Point14ClinicalProfileProcessor().process(
        make_request(observations=[{"task_id": "none", "accuracy": .5}])
    )
    assert empty.profile["cue_integration"]["normalized_cue_weights"] == {}


def test_task_demand_matrix_exposes_rank_and_confounds():
    result = Point14ClinicalProfileProcessor().process(make_request())
    demand = result.profile["task_demands"]
    assert demand["rank"] >= 1
    assert demand["observed_demand_cells"] > 0
    confounded = Point14ClinicalProfileProcessor().process(make_request(
        observations=[
            {"task_id": "a", "accuracy": 1, "demands": {"working_memory": .5}},
            {"task_id": "b", "accuracy": 0, "demands": {"working_memory": .8}},
        ],
        clinical_contexts=[],
    ))
    assert confounded.profile["task_demands"]["identifiability_warning"] is True


def test_discourse_graph_metrics_and_distractor_contrast_are_available():
    result = Point14ClinicalProfileProcessor().process(make_request(
        observations=[
            {"task_id": "with", "accuracy": .4, "distractor_present": True},
            {"task_id": "without", "accuracy": .9, "distractor_present": False},
            {"task_id": "graph", "accuracy": .7,
             "discourse_graph": {"required_edges": [("a", "b")],
                                 "produced_edges": [("a", "b"), ("x", "y")]}},
        ]
    ))
    discourse = result.profile["discourse_coherence"]
    assert discourse["mean_edge_recall"] == pytest.approx(1.0)
    assert discourse["mean_edge_precision"] == pytest.approx(.5)
    assert discourse["distractor_cost"] == pytest.approx(.5)


def test_memory_load_slope_uses_multiple_load_levels():
    result = Point14ClinicalProfileProcessor().process(make_request())
    assert result.profile["working_memory_and_executive_load"]["memory_load_accuracy_slope"] == pytest.approx(-.2)


def test_context_profiles_are_task_specific_and_not_diagnostic():
    result = Point14ClinicalProfileProcessor().process(make_request())
    profiles = result.profile["context_specific_profiles"]
    assert profiles["right_hemisphere"]["sentence_vs_discourse_contrast"] == pytest.approx(.4)
    assert profiles["autism_related"]["communication_preferences_recorded"] is True
    assert profiles["psychosis_spectrum"]["comprehension_score"] == pytest.approx(.8)
    assert profiles["psychosis_spectrum"]["production_score"] == pytest.approx(.4)
    assert profiles["aphasia"]["language_concept_dissociation"] == pytest.approx(-.4)
    assert all(item["diagnostic_inference_performed"] is False for item in profiles.values())


def test_neurodegenerative_trajectories_are_component_specific():
    result = Point14ClinicalProfileProcessor().process(make_request())
    trajectory = result.profile["longitudinal_trajectories"]["semantic_composition"]
    assert trajectory["available"] is True
    assert trajectory["slope_per_time_unit"] == pytest.approx(-.1)
    assert result.profile["context_specific_profiles"]["neurodegenerative"][
        "component_trajectories_are_separate"
    ] is True


def test_component_profiles_preserve_unknown_and_report_measurement_counts():
    result = Point14ClinicalProfileProcessor().process(make_request(component_measures={}))
    assert set(result.profile["functional_components"]["components"]) == set(POINT14_COMPONENTS)
    assert result.profile["functional_components"]["components"]["perceptual_access"] is None
    assert "perceptual_access" in result.uncertainty_estimates["missing_components"]
    assert result.uncertainty_estimates["unmeasured_is_not_zero"] is True


def test_competing_explanations_use_supplied_predictions_only():
    result = Point14ClinicalProfileProcessor().process(make_request())
    ranked = result.candidate_explanations
    assert ranked[0]["hypothesis_id"] == "cue-integration"
    assert ranked[0]["prediction_count"] == 3


def test_support_benefit_requires_matched_pair_and_explicit_support():
    result = Point14ClinicalProfileProcessor().process(make_request())
    assert result.support_predictions["mean_support_benefit"] == pytest.approx(.3)
    assert result.support_predictions["generalization_items"] == ["supported"]


def test_validation_uses_only_explicit_held_out_items():
    result = Point14ClinicalProfileProcessor().process(make_request())
    assert result.validation["validation_n"] == 1
    assert result.validation["held_out_brier_score"] == pytest.approx(.04)
    assert result.validation["validation_available"] is True


def test_previous_profile_is_historical_when_current_measurement_missing():
    request = make_request(
        component_measures={"semantic_composition": [.5, .7]},
        observations=[],
        previous_profile={"components": {"semantic_composition": .9, "lexical_access": .4}},
    )
    result = Point14ClinicalProfileProcessor().process(request)
    update = result.profile["profile_update"]
    assert update["updated_components_with_current_evidence_only"]["semantic_composition"] == pytest.approx(.7)
    assert "lexical_access" not in update["updated_components_with_current_evidence_only"]
    assert update["historical_previous_components"]["lexical_access"] == pytest.approx(.4)


def test_empty_evidence_does_not_fabricate_component_estimates():
    result = Point14ClinicalProfileProcessor().process(make_request(
        task_set=[], observations=[], component_measures={}, clinical_contexts=[],
        candidate_explanations=[], longitudinal_observations=[], previous_profile={}
    ))
    assert result.profile["functional_components"]["components"]["working_memory"] is None
    assert result.validation["validation_available"] is False
    assert result.diagnostics["missing_data_preserved"] is True


def test_request_deep_copies_nested_records_and_validates_identifiers():
    observations = [{"task_id": "a", "nested": {"value": 1}}]
    request = make_request(observations=observations)
    observations[0]["nested"]["value"] = 99
    assert request.observations[0]["nested"]["value"] == 1
    with pytest.raises(ValueError):
        make_request(participant_id="")
    with pytest.raises(ValueError):
        make_request(target_outcome="")


def test_public_point14_definitions_have_docstrings():
    public = [
        value for name, value in vars(point14).items()
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
    (isolated / "pyproject.toml").write_text("[project]\nname='point14-test'\n", encoding="utf-8")
    database_path = tmp_path / "point14.sqlite3"
    report = register_point14_annotations(isolated, database_path)
    rows = LoggerDatabase(database_path).search_annotations("ore_language_comprehension", limit=5000)
    assert report["symbols_written"] > 0
    assert any(row["qualified_name"] == "Point14ClinicalProfileProcessor.process" for row in rows)
    assert any(row["qualified_name"] == "Point14Request.__post_init__" for row in rows)
    assert all(row["source_docstring"] for row in rows)
