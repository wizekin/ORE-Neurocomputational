"""Focused tests for the 17-stage developmental language/cognition model."""
from __future__ import annotations

import sqlite3
import pytest

from ore_developmental_language_cognition import (
    DevelopmentalBiologicalNetwork, DevelopmentalConfig, DevelopmentalExperience,
    DevelopmentalLanguageCognitionProcessor, DevelopmentalLearnerState,
    OPERATION_REGISTRY, integrate_multimodal_referents, refine_developing_circuits,
)


def experience(**overrides):
    """Build a realistic multi-domain learning episode with independent evidence channels."""
    values = {
        "experience_id": "p16-example-001",
        "age_months": 96.0,
        "acoustic_frames": ((0.1, 0.8, 0.2), (0.2, 0.75, 0.25), (0.8, 0.1, 0.7)),
        "syllables": ("ba", "na", "na"),
        "boundary_cues": (
            {"pause": 0.95, "stress": 0.8, "phonotactic_boundary": 0.7},
            {"pause": 0.05, "stress": 0.1},
        ),
        "phonetic_label": "ba-like",
        "native_language": "environment-A",
        "phonetic_feedback": {
            "native_contrasts": ({"contrast": "ba-pa", "discrimination_success": 0.95},),
            "unfamiliar_contrasts": ({"contrast": "click-contrast", "discrimination_success": 0.4},),
        },
        "scene_referents": (
            {"referent_id": "fruit-1", "label": "banana", "salience": 0.9,
             "features": (1.0, 0.2, 0.1), "attributes": {"yellow": True, "edible": True}},
            {"referent_id": "toy-1", "label": "toy banana", "salience": 0.5,
             "features": (0.8, 0.2, 0.4), "attributes": {"yellow": True, "edible": False}},
        ),
        "gaze_target": "fruit-1",
        "gesture_target": "fruit-1",
        "speaker_intention": "label_object",
        "social_contingency": 0.9,
        "target_word": "banana",
        "syntactic_hints": {"semantic_type": "entity", "part_of_speech": "noun"},
        "context_tokens": ("fruit", "yellow", "eat"),
        "concept_examples": (
            {"label": "banana", "parent": "fruit", "semantic_type": "entity",
             "features": (1.0, 0.2, 0.1),
             "attributes": {"yellow": True, "edible": True, "spots": False}},
            {"label": "fruit", "parent": "food", "semantic_type": "category",
             "features": (0.8, 0.3, 0.1),
             "attributes": {"edible": True, "grows_on_plant": True}},
        ),
        "memory_cues": ("banana", "fruit", "eat"),
        "working_memory_items": ("banana", "yellow", "fruit", "snack"),
        "distractors": ("toy",),
        "current_goal": "find fruit and remember its color",
        "action_candidates": (
            {"action_id": "choose-fruit", "goal_match": 1.0, "relevance": 0.95,
             "distraction": 0.0, "prepotency": 0.1, "conflict": 0.0},
            {"action_id": "choose-toy", "goal_match": 0.1, "relevance": 0.2,
             "distraction": 0.8, "prepotency": 1.0, "conflict": 1.0},
        ),
        "current_rule": "choose edible items",
        "previous_rule": "choose colored items",
        "rule_feedback": {"error": 0.1},
        "circuit_activations": {"auditory": 0.8, "temporal": 0.7, "visual": 0.6},
        "active_pathways": {"auditory-temporal": 0.8, "temporal-frontal": 0.7},
        "path_lengths": {"auditory-temporal": 2.0, "temporal-frontal": 1.5},
        "plan_steps": (
            {"step_id": "observe", "depends_on": ()},
            {"step_id": "identify", "depends_on": ("observe",)},
            {"step_id": "select", "depends_on": ("identify",)},
        ),
        "plan_observations": {"observe": {"completed": True}},
        "performance_feedback": {
            "domains": {
                "speech_perception": {"success": 0.82},
                "working_memory": {"success": 0.65},
                "inhibition": {"success": 0.90},
            }
        },
    }
    values.update(overrides)
    return DevelopmentalExperience(**values)


def test_registry_has_seventeen_source_ordered_operations():
    """Keep the registry aligned with the 17 source stages."""
    assert len(OPERATION_REGISTRY) == 17
    assert [item.operation_id for item in OPERATION_REGISTRY] == [
        f"P16-{index:02d}" for index in range(1, 18)
    ]


def test_raw_waveform_feature_extraction_preserves_time_order():
    """Derive simple overlapping acoustic features from a synthetic waveform."""
    waveform = [0.4 * ((-1.0) ** index) for index in range(600)]
    result = DevelopmentalLanguageCognitionProcessor().process(experience(
        acoustic_frames=(), audio_samples=waveform, syllables=(),
        scene_referents=(), concept_examples=(), working_memory_items=(),
        action_candidates=(), plan_steps=(), performance_feedback={},
    ))
    assert result.acoustic_features["source"] == "raw_waveform_frame_features"
    assert result.acoustic_features["frame_count"] > 1
    assert result.acoustic_features["feature_dimension"] == 6


def test_phonetic_categories_are_provisional_and_experience_sensitive():
    """Update a labelled sound prototype across exposure without assuming adult categories."""
    state = DevelopmentalLearnerState()
    processor = DevelopmentalLanguageCognitionProcessor(state=state)
    first = processor.process(experience())
    before = list(state.phonetic_prototypes["ba-like"])
    second = processor.process(experience(experience_id="p16-example-002",
                                          acoustic_frames=((0.2, 0.7, 0.25),)))
    after = state.phonetic_prototypes["ba-like"]
    assert first.phonetic_categories["assignments"]
    assert second.phonetic_categories["assignments"]
    assert before != after
    assert second.phonetic_categories["adult_category_structure_assumed"] is False


def test_statistical_learning_accumulates_frequencies_and_transitions():
    """Repeated syllable sequences should accumulate measurable conditional regularities."""
    state = DevelopmentalLearnerState()
    processor = DevelopmentalLanguageCognitionProcessor(state=state)
    processor.process(experience())
    result = processor.process(experience(experience_id="p16-repeat"))
    assert state.syllable_counts["na"] >= 4
    assert result.statistical_learning["transition_probabilities"]
    assert result.statistical_learning["observations"] >= 6


def test_segmentation_combines_multiple_cues_not_one_fixed_transition():
    """Strong pause/prosodic evidence should raise candidate boundary confidence."""
    result = DevelopmentalLanguageCognitionProcessor(
        config=DevelopmentalConfig(segmentation_threshold=0.2)
    ).process(experience(
        syllables=("ba", "na"),
        boundary_cues=({"pause": 1.0, "stress": 1.0, "phonotactic_boundary": 1.0},),
    ))
    assert result.word_segmentation["boundary_indices"] == [1]
    assert result.word_segmentation["candidate_word_forms"] == ["ba", "na"]
    assert result.word_segmentation["single_transition_rule_not_assumed"] is True


def test_native_tuning_does_not_delete_unfamiliar_contrasts():
    """Native weighting can strengthen while unfamiliar categories remain represented."""
    state = DevelopmentalLearnerState(
        unfamiliar_contrast_strengths={"click-contrast": 0.9}
    )
    result = DevelopmentalLanguageCognitionProcessor(state=state).process(experience())
    assert state.native_contrast_strengths["ba-pa"] > 0.5
    assert state.unfamiliar_contrast_strengths["click-contrast"] > 0.0
    assert result.perceptual_tuning["unfamiliar_contrasts_are_not_deleted"] is True


def test_gaze_and_gesture_constrain_multimodal_referent_selection():
    """Jointly aligned gaze and pointing should increase a candidate referent posterior."""
    result = integrate_multimodal_referents(experience(), DevelopmentalConfig())
    assert result["selected_referent"] == "fruit-1"
    assert result["referent_posterior"][0]["probability"] > result["referent_posterior"][1]["probability"]
    assert result["cue_independence_is_only_an_approximation"] is True


def test_repeated_exposure_establishes_a_vocabulary_entry():
    """Provisional word knowledge becomes established only after repeated confident exposures."""
    state = DevelopmentalLearnerState()
    processor = DevelopmentalLanguageCognitionProcessor(state=state)
    processor.process(experience())
    processor.process(experience(experience_id="p16-word-2"))
    result = processor.process(experience(experience_id="p16-word-3"))
    assert state.lexicon["banana"]["exposures"] == 3
    meanings = [row["meaning"] for row in result.word_meaning_mapping["candidate_meanings"]]
    assert len(meanings) == len(set(meanings))
    assert result.word_meaning_mapping["meaning_confidence"] >= 0.55
    assert result.vocabulary_growth["status"] == "established"
    assert result.vocabulary_growth["learning_efficiency"] >= 0.35


def test_concepts_retain_hierarchy_and_distinguish_stable_from_incidental_features():
    """Repeated grounded examples should preserve parent relations and feature recurrence."""
    state = DevelopmentalLearnerState()
    processor = DevelopmentalLanguageCognitionProcessor(state=state)
    for index in range(4):
        processor.process(experience(
            experience_id=f"p16-concept-{index}",
            concept_examples=({
                "label": "bird", "parent": "animal", "semantic_type": "entity",
                "features": (0.8, 0.2, 0.4),
                "attributes": {"feathers": True, "spots": index == 0},
            },),
        ))
    bird = state.concept_store["bird"]
    assert bird["parent"] == "animal"
    assert "feathers" in bird["stable_features"]
    assert "spots" in bird["incidental_features"]


def test_memory_keeps_episodes_separate_from_general_associations():
    """Source-linked episodes accumulate while word-to-meaning strength is generalized separately."""
    state = DevelopmentalLearnerState()
    processor = DevelopmentalLanguageCognitionProcessor(state=state)
    for index in range(3):
        result = processor.process(experience(experience_id=f"p16-memory-{index}"))
    assert len(state.episodic_memory) == 3
    assert state.semantic_associations
    assert result.memory_stabilization["episodic_and_generalized_knowledge_separated"] is True


def test_working_memory_uses_chunking_and_distraction_sensitive_retention():
    """A familiar schema chunk should reduce the number of separate units to maintain."""
    result = DevelopmentalLanguageCognitionProcessor().process(experience(
        working_memory_items=("red", "apple", "blue", "berry"),
        distractors=("toy",),
        metadata={"schema_chunks": ({"name": "fruit-color", "items": ("red", "apple")},)},
    ))
    assert result.working_memory["chunk_count"] == 1
    assert result.working_memory["effective_capacity"] >= 0.5
    assert result.working_memory["capacity_is_flexible_and_chunk_sensitive"] is True


def test_executive_control_prefers_goal_relevant_action_and_tracks_rule_switch():
    """Goal maintenance should outscore a distracting prepotent response and register switching."""
    result = DevelopmentalLanguageCognitionProcessor().process(experience())
    assert result.executive_control["selected_action"] == "choose-fruit"
    assert result.executive_control["rule_switched"] is True
    assert result.executive_control["inhibition_required"] is False


def test_circuit_refinement_is_activity_dependent_and_myelination_changes_delay_proxy():
    """Repeated pathway use should strengthen links and reduce the modeled conduction delay."""
    state = DevelopmentalLearnerState(circuit_weights={"a->b": 0.2})
    event = experience(circuit_activations={"a": 1.0, "b": 1.0},
                       active_pathways={"a-b": 1.0}, path_lengths={"a-b": 2.0})
    config = DevelopmentalConfig()
    first = refine_developing_circuits(event, state, config)
    before_delay = first["conduction_delay_proxy"]["a-b"]
    second = refine_developing_circuits(event, state, config)
    after_delay = second["conduction_delay_proxy"]["a-b"]
    assert state.circuit_weights["a->b"] > 0.2
    assert after_delay < before_delay
    assert second["biological_synapses_not_equated_with_regularization"] is True


def test_multi_step_plan_keeps_completed_and_pending_dependencies_separate():
    """A dependent step cannot complete before its prerequisite is observed as completed."""
    result = DevelopmentalLanguageCognitionProcessor().process(experience(
        plan_steps=(
            {"step_id": "one", "depends_on": ()},
            {"step_id": "two", "depends_on": ("one",)},
            {"step_id": "three", "depends_on": ("two",)},
        ),
        plan_observations={"one": {"completed": True}, "three": {"completed": True}},
    ))
    assert result.multi_step_reasoning["completed_steps"] == ["one"]
    assert result.multi_step_reasoning["pending_steps"] == ["three", "two"]
    assert "two" in result.multi_step_reasoning["ready_steps"]
    assert result.multi_step_reasoning["premature_action_suppressed"] is True


def test_developmental_calibration_tracks_domains_separately_not_as_one_age_score():
    """Different performance domains should receive different difficulty and support updates."""
    low_age = DevelopmentalLanguageCognitionProcessor().process(experience(
        age_months=18, performance_feedback={"domains": {"vocabulary": {"success": 0.4}}}
    ))
    high_age = DevelopmentalLanguageCognitionProcessor().process(experience(
        experience_id="older-trajectory", age_months=180,
        performance_feedback={"domains": {
            "vocabulary": {"success": 0.4}, "working_memory": {"success": 0.9}
        }},
    ))
    assert "vocabulary" in low_age.developmental_calibration["domain_updates"]
    assert "working_memory" in high_age.developmental_calibration["domain_updates"]
    assert high_age.developmental_calibration["age_is_not_used_as_a_single_maturity_score"] is True


def test_all_seventeen_stages_run_with_separate_biological_network_traces():
    """Return exactly 17 ordered operation records with distinct network diagnostics."""
    result = DevelopmentalLanguageCognitionProcessor().process(experience())
    assert len(result.operation_trace) == 17
    assert [row["operation_id"] for row in result.operation_trace] == [
        item.operation_id for item in OPERATION_REGISTRY
    ]
    assert "auditory_perception" in result.biological
    assert "prefrontal_executive_control" in result.biological
    assert "hippocampal_prefrontal_integration" in result.biological
    assert result.biological["scientific_scope"].startswith("Rate-coded distributed")


def test_rate_coded_developmental_network_has_recurrent_activity():
    """A network update should carry previous activity into the next population-rate state."""
    network = DevelopmentalBiologicalNetwork()
    first = network.engage("auditory_perception", "broad-input", 0.6)
    second = network.engage("auditory_perception", "recurrent", 0.0)
    assert second["activation_before"] == pytest.approx(first["activation_after"])
    assert second["activation_after"] != pytest.approx(first["activation_after"])


def test_sqlite_registers_every_package_annotation_and_seventeen_operation_events(tmp_path):
    """Index classes/functions and store all developmental operation traces in SQLite."""
    from ore_logger.database import LoggerDatabase
    from ore_developmental_language_cognition import register_annotations, record_result_to_sqlite

    path = tmp_path / "point16.sqlite3"
    result = DevelopmentalLanguageCognitionProcessor().process(experience())
    catalogue = register_annotations(path)
    assert catalogue["files_seen"] >= 6
    assert LoggerDatabase(path).summary()["symbols"] > 0
    logged = record_result_to_sqlite(result, path)
    with sqlite3.connect(path) as connection:
        row = connection.execute(
            "SELECT status, event_count FROM runs WHERE run_id = ?", (logged["run_id"],)
        ).fetchone()
        count = connection.execute(
            "SELECT COUNT(*) FROM events WHERE run_id = ?", (logged["run_id"],)
        ).fetchone()[0]
    assert row == ("completed", 17)
    assert count == 17


def test_contract_rejects_invalid_age_and_nonfinite_acoustic_values():
    """Reject nonsensical developmental metadata and acoustic feature values."""
    with pytest.raises(ValueError):
        DevelopmentalExperience(experience_id="bad-age", age_months=300)
    with pytest.raises(ValueError):
        DevelopmentalExperience(experience_id="bad-audio", acoustic_frames=((0.1, float("nan")),))
    with pytest.raises(ValueError):
        DevelopmentalConfig(target_success=2.0)
