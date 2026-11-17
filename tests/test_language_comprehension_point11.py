"""Acceptance tests for Point 11 functional separation and cooperation."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from ore_language_comprehension import (
    POINT11_STEP_SPECS,
    POINT11_SYSTEMS,
    Point11FunctionalSeparationCoordinator,
    Point11LIFReference,
    Point11Request,
    register_point11_annotations,
)
from ore_logger import LoggerDatabase


def stage_outputs():
    """Return small upstream stage records with separate literal and conceptual sources."""
    return {
        "lexical": {"candidates": [{"word": "Maya", "confidence": .9}]},
        "syntactic": {"candidates": [{"structure": "subject-verb-object"}]},
        "semantic": {"propositions": [{"predicate": "OPEN", "arguments": ["child", "door"]}]},
        "working_memory": {"items": [{"content": "literal proposition"}]},
    }


def test_point11_has_six_distinct_systems_and_sixteen_operations():
    assert len(POINT11_SYSTEMS) == 6
    assert len(POINT11_STEP_SPECS) == 16
    assert [x["operation_id"] for x in POINT11_STEP_SPECS] == [f"LC-P11-{i:02d}" for i in range(1, 17)]


def test_literal_sentence_does_not_force_mentalizing_or_executive_control():
    result = Point11FunctionalSeparationCoordinator().process(Point11Request(
        sequence_id="literal", stage_outputs=stage_outputs(),
        context={"text": "The child opened the door."},
    ))
    assert result.functional_trace[5]["result"]["recruited"] is False
    assert result.functional_trace[6]["result"]["recruited"] is False
    assert result.system_states["mentalizing"]["activation"] == 0.0
    assert result.system_states["multiple_demand"]["activation"] == 0.0


def test_explicit_belief_task_conditionally_recruits_mentalizing():
    result = Point11FunctionalSeparationCoordinator().process(Point11Request(
        sequence_id="belief", stage_outputs=stage_outputs(),
        task={"mentalizing_required": True},
        agent_candidates=[{"agent": "Maya", "belief": "keys_in_drawer", "confidence": .8}],
    ))
    assert result.functional_trace[5]["result"]["recruited"] is True
    assert result.system_states["mentalizing"]["activation"] > 0
    assert result.interpretation["provenance_preserved"] is True


def test_social_context_cues_raise_social_demand_without_erasing_literal_content():
    result = Point11FunctionalSeparationCoordinator().process(Point11Request(
        sequence_id="irony", stage_outputs=stage_outputs(),
        context={"text": "The speaker's intention is ironic and sarcastic."},
    ))
    assert result.demand_profile["social"] > 0
    assert result.functional_trace[5]["result"]["recruited"] is True
    assert result.functional_trace[5]["result"]["literal_proposition_overwritten"] is False


def test_executive_control_is_gated_separately_from_language_processing():
    result = Point11FunctionalSeparationCoordinator().process(Point11Request(
        sequence_id="dual-task", stage_outputs=stage_outputs(),
        task={"working_memory_load": .9, "response_selection_complexity": .8},
    ))
    assert result.functional_trace[6]["result"]["recruited"] is True
    assert result.functional_trace[6]["result"]["language_computation_assigned_to_multiple_demand"] is False


def test_individual_localizer_takes_precedence_over_literature_prior():
    result = Point11FunctionalSeparationCoordinator().process(Point11Request(
        sequence_id="localizer",
        individual_profile={"language_localizer": {"language": [2, 3], "control": [1, 1],
                                                   "region_labels": ["left temporal"]},
                            "literature_prior": {"laterality": "left"}},
    ))
    assert result.localization["source"] == "individual_functional_localizer"
    assert result.localization["selectivity_contrast"] == pytest.approx(1.5)


def test_missing_localizer_is_not_replaced_with_fabricated_personal_map():
    result = Point11FunctionalSeparationCoordinator().process(Point11Request(sequence_id="no-localizer"))
    assert result.localization["source"] == "not_localized"
    assert result.localization["individual_measurement_supplied"] is False


def test_literature_prior_is_explicitly_not_an_individual_measurement():
    result = Point11FunctionalSeparationCoordinator().process(Point11Request(
        sequence_id="prior", individual_profile={"literature_prior": {"laterality": "left-dominant"}}
    ))
    assert result.localization["source"] == "approximate_literature_prior_only"


def test_literal_and_conceptual_features_keep_separate_provenance():
    result = Point11FunctionalSeparationCoordinator().process(Point11Request(
        sequence_id="provenance", stage_outputs=stage_outputs(),
        conceptual_candidates=[{"feature": "doors are movable barriers", "confidence": .7}],
    ))
    assert result.interpretation["literal_propositions"][0]["predicate"] == "OPEN"
    assert result.provenance[0]["source"] == "conceptual_knowledge"
    assert result.interpretation["provenance_preserved"] is True


def test_upstream_points_are_mapped_to_specialized_channels():
    result = Point11FunctionalSeparationCoordinator().process(Point11Request(
        sequence_id="stage-map",
        stage_outputs={"point2": {"candidates": ["word"]}, "point3": {"candidates": ["parse"]},
                       "point4": {"propositions": ["proposition"]}, "point6": {"items": ["wm"]}},
    ))
    assert result.system_states["language"]["representation"][0]["lexical_candidates"] == ["word"]
    assert result.system_states["memory"]["representation"] == ["wm"]


def test_functional_graph_is_sparse_and_not_called_anatomical_wiring():
    result = Point11FunctionalSeparationCoordinator().process(Point11Request(sequence_id="graph"))
    assert result.communication_graph["edges"]
    assert all(e["edge_type"] == "functional_information_dependency_not_anatomical_wiring"
               for e in result.communication_graph["edges"])


def test_custom_communication_weight_changes_graph():
    result = Point11FunctionalSeparationCoordinator().process(Point11Request(
        sequence_id="custom", network_weights={"language->conceptual_knowledge": .95}
    ))
    edge = next(e for e in result.communication_graph["edges"]
                if e["source"] == "language" and e["target"] == "conceptual_knowledge")
    assert edge["weight"] == pytest.approx(.95)


def test_invalid_network_edges_and_weights_are_rejected():
    with pytest.raises(ValueError):
        Point11FunctionalSeparationCoordinator().process(Point11Request(
            sequence_id="bad-edge", network_weights={"unknown->language": .2}
        ))
    with pytest.raises(ValueError):
        Point11FunctionalSeparationCoordinator().process(Point11Request(
            sequence_id="bad-weight", network_weights={"language->memory": 1.5}
        ))


def test_task_response_profile_is_preserved_without_claiming_absolute_exclusivity():
    result = Point11FunctionalSeparationCoordinator().process(Point11Request(
        sequence_id="selectivity",
        task_response_profile={"language": [3, 4], "non_language_control": [1, 1],
                               "math": [0, 0]},
    ))
    assert result.selectivity["language_vs_control"] == pytest.approx(2.5)
    assert result.selectivity["selectivity_is_profile_not_absolute_exclusivity"] is True


def test_coactivation_is_not_reported_as_causal_evidence():
    result = Point11FunctionalSeparationCoordinator().process(Point11Request(
        sequence_id="evidence",
        individual_profile={"association_evidence": {"fMRI": "activation"},
                            "lesion_evidence": {}, "perturbation_evidence": {}},
    ))
    assert result.contribution_evidence["coactivation_is_not_causation"] is True
    assert result.contribution_evidence["lesion"] == {}


def test_language_and_nonlinguistic_thought_measures_remain_distinct():
    measures = {"language": {"score": .2}, "non_linguistic_reasoning": {"score": .9}}
    result = Point11FunctionalSeparationCoordinator().process(Point11Request(
        sequence_id="dissociation", individual_profile={"language_thought_measures": measures}
    ))
    assert result.functional_trace[10]["result"]["separate_measures"] == measures
    assert result.functional_trace[10]["result"]["complete_independence_claimed"] is False


def test_right_hemisphere_demand_is_task_sensitive_and_does_not_invent_activation():
    result = Point11FunctionalSeparationCoordinator().process(Point11Request(
        sequence_id="narrative", task={"narrative_level": .9}
    ))
    right = result.functional_trace[12]["result"]
    assert right["task_demand_score"] == pytest.approx(.9)
    assert right["activation_claim_made"] is False
    assert right["not_assumed_irrelevant"] is True


def test_right_hemisphere_evidence_is_kept_separate_from_task_demand():
    result = Point11FunctionalSeparationCoordinator().process(Point11Request(
        sequence_id="right-evidence", task={"narrative_level": .8},
        individual_profile={"right_hemisphere_evidence": {"measured_contrast": .4}},
    ))
    right = result.functional_trace[12]["result"]
    assert right["activation_claim_made"] is True
    assert right["measured_evidence"]["measured_contrast"] == .4


def test_incremental_updates_advance_state_and_history():
    coordinator = Point11FunctionalSeparationCoordinator()
    first = coordinator.process(Point11Request(sequence_id="incremental", task={"linguistic_demand": .4}))
    second = coordinator.process(Point11Request(sequence_id="incremental", task={"linguistic_demand": .8}))
    assert first.time_index == 1
    assert second.time_index == 2
    assert len(coordinator.state.history) == 2
    assert second.demand_profile["linguistic"] == pytest.approx(.8)


def test_additional_inference_stops_when_cost_exceeds_utility():
    result = Point11FunctionalSeparationCoordinator().process(Point11Request(
        sequence_id="stop", task={"additional_inference_utility": .2, "additional_inference_cost": .7}
    ))
    stopping = result.functional_trace[15]["result"]
    assert stopping["continue_additional_inference"] is False
    assert stopping["speculative_detail_not_added_when_cost_exceeds_utility"] is True


def test_optional_lif_reference_has_six_separate_population_traces():
    result = Point11FunctionalSeparationCoordinator(simulate_biology=True, lif_population_size=3).process(
        Point11Request(sequence_id="lif", task={"mentalizing_required": True})
    )
    assert result.biological_summary["population_names"] == list(POINT11_SYSTEMS)
    assert len(result.biological_trace) == 16
    assert result.diagnostics["biological_reference_is_measured_human_data"] is False


def test_lif_reference_validates_inputs():
    lif = Point11LIFReference(population_size=2)
    with pytest.raises(KeyError):
        lif.run({"unknown": .5})
    with pytest.raises(ValueError):
        lif.run({"language": float("nan")})
    with pytest.raises(ValueError):
        lif.run({"language": .5}, steps=0)


def test_reset_clears_sequence_state_but_preserves_configuration():
    coordinator = Point11FunctionalSeparationCoordinator(social_threshold=.6, recurrent_gain=.2)
    coordinator.process(Point11Request(sequence_id="reset", task={"mentalizing_required": True}))
    coordinator.reset()
    assert coordinator.state.time_index == 0
    assert coordinator.state.sourced_features == []
    assert coordinator.social_threshold == pytest.approx(.6)
    assert coordinator.recurrent_gain == pytest.approx(.2)


def test_result_serializes_to_plain_data():
    result = Point11FunctionalSeparationCoordinator().process(Point11Request(
        sequence_id="serialize", stage_outputs=stage_outputs()
    ))
    plain = result.to_dict()
    assert plain["functional_trace"][0]["step"] == 1
    assert isinstance(plain["system_states"], dict)


def test_point11_annotations_are_indexed_in_shared_sqlite_catalogue(tmp_path):
    source_root = Path(__file__).resolve().parents[1]
    isolated = tmp_path / "ore-project"
    isolated.mkdir()
    shutil.copytree(source_root / "ore_logger", isolated / "ore_logger")
    shutil.copytree(source_root / "ore_language_comprehension", isolated / "ore_language_comprehension")
    shutil.copytree(source_root / "ore_brain_networks", isolated / "ore_brain_networks")
    (isolated / "pyproject.toml").write_text("[project]\nname='point11-annotation-test'\n", encoding="utf-8")
    database_path = tmp_path / "point11-annotations.sqlite3"

    report = register_point11_annotations(isolated, database_path)
    rows = LoggerDatabase(database_path).search_annotations("ore_language_comprehension", limit=2500)
    assert report["symbols_written"] > 0
    assert any(row["qualified_name"] == "Point11FunctionalSeparationCoordinator.process" for row in rows)
    assert any(row["qualified_name"] == "Point11Request.__post_init__" for row in rows)
    assert all(row["source_docstring"] for row in rows)
