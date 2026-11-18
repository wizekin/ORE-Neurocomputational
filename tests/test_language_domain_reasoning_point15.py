"""Focused functional, biological, separation, and SQLite tests for Point 15."""
from __future__ import annotations

import sqlite3
import pytest

from ore_language_domain_reasoning import (
    LanguageDomainReasoningProcessor, LanguageEvent, LanguageLearnerState,
    LanguageProcessingConfig, OPERATION_REGISTRY, encode_sensory_signal,
    infer_social_pragmatics, update_learned_representations,
)


def event(**overrides):
    """Build a reusable, explicitly contextualized Point 15 language event."""
    values = {
        "event_id": "p15-event-1",
        "sensory_stream": "The metal expanded after heating.",
        "modality": "text",
        "linguistic_context": {"preceding_words": "metal expands when heated"},
        "lexicon": {
            "metal": {"frequency": 0.9, "grammar": "noun"},
            "expanded": {"frequency": 0.8, "grammar": "verb"},
            "heating": {"frequency": 0.75, "grammar": "noun"},
        },
        "conceptual_memory": {
            "thermal_expansion": {
                "description": "many materials expand when temperature rises",
                "reliability": 0.95,
            },
        },
        "situational_context": {"domain": "physics"},
        "task_goal": "comprehend",
        "cognitive_state": {"control_capacity": 0.7, "fatigue": 0.0},
    }
    values.update(overrides)
    return LanguageEvent(**values)


def test_registry_contains_exact_fifteen_source_ordered_operations():
    """Keep the registry aligned to all 15 stages in the source algorithm."""
    assert len(OPERATION_REGISTRY) == 15
    assert [item.operation_id for item in OPERATION_REGISTRY] == [
        f"P15-{index:02d}" for index in range(1, 16)
    ]


def test_sensory_encoding_preserves_modality_order_quality_and_speech_scope():
    """Preserve input order and note that speech arrives here as symbolic transcript text."""
    text_features = encode_sensory_signal(event(sensory_stream=("A red", "metal rod.")))
    speech_features = encode_sensory_signal(event(modality="speech"))
    assert text_features["ordered_tokens"] == ["A", "red", "metal", "rod", "."]
    assert speech_features["transcript_assumption_for_speech"] is True
    assert speech_features["signal_quality"] == 1.0


def test_lexical_alternatives_are_retained_with_probabilities():
    """Maintain competing perceptual candidates when signal evidence is ambiguous."""
    result = LanguageDomainReasoningProcessor().process(event(
        sensory_stream="The bat moved.",
        lexical_alternatives={"bat": ("bat", "bag")},
        lexicon={"bat": {"frequency": 0.8}, "bag": {"frequency": 0.8}},
    ))
    candidates = result.linguistic_representation["lexical"]["lexical_candidates"]["bat"]
    assert len(candidates) == 2
    assert sum(row["probability"] for row in candidates) == pytest.approx(1.0)
    assert result.linguistic_representation["lexical"]["alternatives_preserved"] is True


def test_conceptual_retrieval_is_separate_from_explicit_utterance_content():
    """Retrieved scientific knowledge must not be mislabeled as something the speaker said."""
    result = LanguageDomainReasoningProcessor().process(event())
    explicit = result.linguistic_representation["said_or_encoded"]
    concepts = result.conceptual_interpretation["activated_concepts"]
    assert explicit["utterance"] == "The metal expanded after heating."
    assert concepts
    assert "thermal_expansion" not in explicit["tokens"]
    assert result.integrated_interpretation["provenance_map"]["conceptual_knowledge"] == "retrieved_long_term_knowledge"


def test_control_demand_increases_when_capacity_is_low_and_memory_load_is_high():
    """Estimate domain-general demand separately from sentence identity."""
    easy = LanguageDomainReasoningProcessor().process(event(
        sensory_stream="Birds fly.", conceptual_memory={},
        cognitive_state={"control_capacity": 1.0, "fatigue": 0.0},
    ))
    hard = LanguageDomainReasoningProcessor().process(event(
        sensory_stream="The report that the editor who the journalist interviewed praised was published.",
        cognitive_state={"control_capacity": 0.15, "fatigue": 0.9},
    ))
    assert hard.control_request["demand_score"] >= easy.control_request["demand_score"]
    assert hard.control_request["control_allocation"] > easy.control_request["control_allocation"]


def test_reasoning_conclusion_is_marked_inferred_not_said():
    """A rule conclusion is separate from the directly encoded sentence."""
    result = LanguageDomainReasoningProcessor().process(event(
        sensory_stream="Every A is B. This is A.", task_goal="derive",
        reasoning_rules=({"rule_id": "modus-1", "if_all": ("Every A is B", "This is A"),
                          "then": "This is B"},),
    ))
    assert result.reasoning_result["reasoning_requested"] is True
    assert result.reasoning_result["inferred"][0]["conclusion"] == "This is B"
    assert result.reasoning_result["said_or_encoded"]["utterance"] == "Every A is B. This is A."
    assert result.reasoning_result["linguistic_content_not_rewritten_by_inference"] is True


def test_pragmatic_inference_is_not_recruited_for_every_sentence():
    """Routine literal comprehension should not automatically invoke social inference."""
    literal = infer_social_pragmatics(event(), {
        "said_or_encoded": {"utterance": "The metal expanded after heating."}
    }, LanguageProcessingConfig())
    pragmatic = infer_social_pragmatics(event(
        sensory_stream="It is cold in here.", task_goal="infer speaker intention",
        speaker_model={"candidate_intentions": ("inform", "request_action")},
        situational_context={"open_window": True, "requires_intention_inference": True},
    ), {"said_or_encoded": {"utterance": "It is cold in here."}}, LanguageProcessingConfig())
    assert literal["recruited"] is False
    assert pragmatic["recruited"] is True
    assert pragmatic["intention_hypotheses"]


def test_mismatch_classification_targets_lexical_layer_and_preserves_raw_input():
    """A lexical correction reopens lexical alternatives without rewriting the original utterance."""
    result = LanguageDomainReasoningProcessor().process(event(
        new_evidence={"lexical_correction": {"token": "metal", "candidate": "medal"}}
    ))
    assert result.mismatch_and_revision["mismatch_detected"] is True
    assert "lexical" in result.mismatch_and_revision["mismatch_types"]
    assert result.mismatch_and_revision["original_explicit_content_preserved"] is True
    assert result.linguistic_representation["said_or_encoded"]["utterance"].startswith("The metal")


def test_one_off_unreliable_feedback_does_not_update_long_term_lexicon():
    """Require reliable repeated evidence before changing a general lexical representation."""
    state = LanguageLearnerState()
    weak = update_learned_representations(
        event(feedback={"kind": "confirmation", "token": "metal", "correct": True,
                        "reliability": 0.4, "repetitions": 1}),
        state, {}, {}, LanguageProcessingConfig(),
    )
    assert weak["eligible"] is False
    assert "metal" not in state.lexical_strengths


def test_repeated_reliable_feedback_updates_lexical_and_construction_strengths():
    """Use bounded reliability-weighted updates for repeated evidence."""
    state = LanguageLearnerState()
    updated = update_learned_representations(
        event(feedback={"kind": "confirmation", "token": "metal", "construction": "noun verb",
                        "correct": True, "reliability": 0.95, "repetitions": 5}),
        state, {}, {}, LanguageProcessingConfig(),
    )
    assert updated["eligible"] is True
    assert state.lexical_strengths["metal"] > 0.5
    assert state.construction_strengths["noun verb"] > 0.5


def test_full_pipeline_runs_fifteen_stages_and_keeps_networks_distinct():
    """Return 15 ordered operations and separate biological subsystem activity."""
    result = LanguageDomainReasoningProcessor().process(event())
    assert len(result.operation_trace) == 15
    assert [row["operation_id"] for row in result.operation_trace] == [
        item.operation_id for item in OPERATION_REGISTRY
    ]
    assert result.biological["frontotemporal_language_selective"]["engagement_count"] == 4
    assert "multiple_demand_control" in result.biological
    assert "social_pragmatic_cognition" in result.biological
    assert result.integrated_interpretation["system_separation_preserved"] is True


def test_sqlite_annotation_registration_and_operation_logging(tmp_path):
    """Index package annotations and persist every Point 15 operation in ORE SQLite."""
    from ore_logger.database import LoggerDatabase
    from ore_language_domain_reasoning import register_annotations, record_result_to_sqlite

    path = tmp_path / "point15.sqlite3"
    result = LanguageDomainReasoningProcessor().process(event())
    catalogue = register_annotations(path)
    assert catalogue["files_seen"] >= 5
    assert LoggerDatabase(path).summary()["symbols"] > 0
    logged = record_result_to_sqlite(result, path)

    with sqlite3.connect(path) as connection:
        run = connection.execute(
            "SELECT status, event_count FROM runs WHERE run_id = ?", (logged["run_id"],)
        ).fetchone()
        count = connection.execute(
            "SELECT COUNT(*) FROM events WHERE run_id = ?", (logged["run_id"],)
        ).fetchone()[0]
    assert run == ("completed", 15)
    assert count == 15


def test_contract_rejects_unknown_modality_and_invalid_probabilities():
    """Reject unsupported modalities and learning-rate values outside the unit interval."""
    with pytest.raises(ValueError):
        LanguageEvent(event_id="bad", sensory_stream="hello", modality="telepathy")
    with pytest.raises(ValueError):
        LanguageProcessingConfig(learning_rate=1.5)
