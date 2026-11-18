"""Focused tests for Point 14 social interaction and conversational learning."""

from __future__ import annotations

import sqlite3

import pytest

from ore_logger.database import LoggerDatabase
from ore_social_conversational_learning import (
    LearnerSocialState,
    NarrativeEvent,
    PartnerModel,
    ReferentCandidate,
    SocialFeedback,
    SocialInteractionEvent,
    SocialInteractionLearningProcessor,
    SocialLearningConfig,
    SocialSignal,
    OPERATION_REGISTRY,
    compare_perspectives,
    design_audience_response,
    detect_conversational_mismatch,
    evaluate_explanation,
    infer_communicative_intention,
    infer_joint_attention,
    integrate_multimodal_signals,
    learn_from_social_feedback,
    predict_turn_state,
    retrieve_social_memories,
    select_scaffold,
)


def social_event(**overrides):
    """Create a multimodal shared-attention event with an ambiguous target word."""
    values = {
        "event_id": "social-1",
        "learner_id": "learner-1",
        "partner_id": "partner-1",
        "utterance": "Look at the red cup.",
        "signals": (
            SocialSignal(
                "speech-1", "speech", "Look at the red cup.",
                reliability=0.95, salience=0.8, directed_to_learner=0.9,
                timestamp=1.0,
            ),
            SocialSignal(
                "gaze-1", "gaze", "partner gaze toward cup",
                reliability=0.9, salience=0.8, directed_to_learner=0.95,
                target_id="cup", timestamp=1.1,
            ),
            SocialSignal(
                "point-1", "gesture", "partner points toward cup",
                reliability=0.95, salience=0.95, directed_to_learner=0.95,
                target_id="cup", timestamp=1.2,
            ),
        ),
        "referents": (
            ReferentCandidate(
                "cup", "red cup", features=("red", "cup", "container"),
                salience=0.9, context_fit=0.95,
            ),
            ReferentCandidate(
                "bowl", "blue bowl", features=("blue", "bowl", "container"),
                salience=0.6, context_fit=0.4,
            ),
        ),
        "learner_focus": "cup",
        "learning_goal": "learn_object_words",
        "target_word": "cup",
        "candidate_mappings": {"cup": 0.95, "bowl": 0.05},
        "predicted_partner_response": "Yes, that is the cup.",
        "partner_response": "Yes, that is the cup.",
        "target_concept": "cup",
        "required_explanation_terms": ("cup", "container", "red"),
        "required_relations": ("cup is a container",),
        "learner_attempt": "A red cup is a container.",
        "difficulty": 0.55,
        "metadata": {"timestamp": "t1", "response_ready": True},
    }
    values.update(overrides)
    return SocialInteractionEvent(**values)


def test_registry_contains_exact_sixteen_source_ordered_operations():
    """Keep the processor aligned with all 16 source algorithmic stages."""
    assert len(OPERATION_REGISTRY) == 16
    assert [item.operation_id for item in OPERATION_REGISTRY] == [
        f"P14-{index:02d}" for index in range(1, 17)
    ]


def test_joint_attention_requires_learner_partner_focus_agreement():
    """Confirm shared focus when reliable gaze and pointing align with learner attention."""
    event = social_event()
    config = SocialLearningConfig()
    attention = __import__(
        "ore_social_conversational_learning", fromlist=["estimate_social_attention"]
    ).estimate_social_attention(event.signals, event.learning_goal, config)
    result = infer_joint_attention(event, attention, config)
    assert result["partner_focus"] == "cup"
    assert result["status"] == "coordinated"
    assert result["shared_attention_confidence"] >= config.joint_attention_threshold


def test_joint_attention_preserves_conflicting_referent_cues():
    """Avoid forcing agreement when learner and high-reliability partner cues point to different objects."""
    event = social_event(
        event_id="focus-conflict",
        learner_focus="bowl",
        signals=(
            SocialSignal("gaze", "gaze", "looks at cup", reliability=0.95,
                         salience=0.9, target_id="cup"),
            SocialSignal("point", "gesture", "points at cup", reliability=0.95,
                         salience=0.9, target_id="cup"),
        ),
    )
    config = SocialLearningConfig()
    attention = __import__(
        "ore_social_conversational_learning", fromlist=["estimate_social_attention"]
    ).estimate_social_attention(event.signals, event.learning_goal, config)
    result = infer_joint_attention(event, attention, config)
    assert result["status"] == "uncertain_or_misaligned"
    assert result["clarification_needed"] is True


def test_partner_intention_inference_uses_learner_partner_belief_difference():
    """Estimate belief mismatch from separate learner and partner knowledge states."""
    event = social_event(
        utterance="Why is the cup red?",
        candidate_intentions=("questioning", "informing"),
    )
    partner = PartnerModel(
        partner_id="partner-1", knowledge_state={"color": 0.9, "shape": 0.2}
    )
    result = infer_communicative_intention(
        event, partner, SocialLearningConfig(),
        learner_knowledge={"color": 0.1, "shape": 0.8},
    )
    assert result["most_likely_intention"] == "questioning"
    assert result["partner_belief_mismatch_proxy"] > 0.5
    assert result["uncertainty_nats"] >= 0.0


def test_multimodal_integration_weights_reliable_signals_and_preserves_conflict():
    """Fuse modalities by precision while retaining conflicting target evidence and provenance."""
    event = social_event(
        signals=(
            SocialSignal("speech", "speech", "cup", reliability=0.95, target_id="cup"),
            SocialSignal("gesture", "gesture", "points elsewhere", reliability=0.35, target_id="bowl"),
        )
    )
    config = SocialLearningConfig()
    attention = __import__(
        "ore_social_conversational_learning", fromlist=["estimate_social_attention"]
    ).estimate_social_attention(event.signals, event.learning_goal, config)
    result = integrate_multimodal_signals(event, attention, config)
    assert result["modality_weights"]["speech"] > result["modality_weights"]["gesture"]
    assert result["modality_conflicts"]
    assert result["source_preserved"] is True


def test_grounded_word_mapping_selects_jointly_attended_referent_and_updates_association():
    """Learn a tentative word-to-object mapping from the partner's shared focus and context."""
    state = LearnerSocialState(learner_id="learner-1")
    event = social_event()
    processor = SocialInteractionLearningProcessor("learner-1", state=state)
    result = processor.process(event)
    assert result.semantic_grounding["selected_referent"] == "cup"
    key = "cup::cup"
    assert state.word_referent_associations[key] > 0.5
    assert result.semantic_grounding["confidence"] > 0.5


def test_turn_prediction_responds_to_clear_yield_cues():
    """Predict a yielded turn when a question, long pause, gaze, and gesture invite response."""
    event = social_event(
        utterance="What shape is it?",
        signals=(
            SocialSignal("speech", "speech", "What shape is it?", reliability=0.95),
            SocialSignal("pause", "timing", "pause", reliability=0.95,
                         attributes={"pause_seconds": 1.5}),
            SocialSignal("gaze-yield", "gaze", "looks toward learner", reliability=0.95,
                         attributes={"turn_yielding": True}),
            SocialSignal("gesture-yield", "gesture", "open hand", reliability=0.9,
                         attributes={"inviting_response": True}),
        ),
    )
    result = predict_turn_state(event, {}, SocialLearningConfig())
    assert result["yield_probability"] > 0.62
    assert result["learner_should_respond"] is True


def test_audience_design_changes_detail_for_novice_and_expert_partners():
    """Adapt vocabulary and explanation depth to different estimated partner knowledge states."""
    event = social_event(target_concept="geometry")
    novice = design_audience_response(
        event, PartnerModel("novice", knowledge_state={"geometry": 0.1}),
        {"most_likely_intention": "informing"},
    )
    expert = design_audience_response(
        event, PartnerModel("expert", knowledge_state={"geometry": 0.95}),
        {"most_likely_intention": "informing"},
    )
    assert novice["detail_level"] == "high"
    assert expert["detail_level"] == "concise"
    assert novice["response_plan"] != expert["response_plan"]


def test_misunderstanding_triggers_repair_but_matching_response_does_not():
    """Detect a lexical/response mismatch and retain normal monitoring when responses agree."""
    event = social_event(
        predicted_partner_response="Yes, that is the cup.",
        partner_response="No, I meant the bowl.",
        feedback=(
            SocialFeedback(
                "correction", "informational_correction",
                content="I meant the bowl.", target_key="cup",
                reliability=0.95, confirms=False,
            ),
        ),
    )
    repair = detect_conversational_mismatch(
        event, {"uncertainty_nats": 0.5}, SocialLearningConfig()
    )
    assert repair["mismatch_detected"] is True
    assert repair["alternatives_reopened"] is True
    assert "referential" in repair["mismatch_types"]

    matching = detect_conversational_mismatch(
        social_event(), {"uncertainty_nats": 0.1}, SocialLearningConfig()
    )
    assert matching["mismatch_detected"] is False


def test_social_approval_changes_action_value_not_factual_confidence():
    """Keep praise/reward separate from factual knowledge certainty."""
    state = LearnerSocialState(learner_id="learner-1", semantic_knowledge={"cup": 0.4})
    event = social_event(feedback=(
        SocialFeedback("praise", "social_affective", content="Good job.",
                       reward=1.0, reliability=0.95),
    ))
    result = learn_from_social_feedback(
        event, state, PartnerModel("partner-1"), SocialLearningConfig()
    )
    assert state.semantic_knowledge["cup"] == pytest.approx(0.4)
    assert state.action_values["learn_object_words"] > 0.5
    assert not result["factual_corrections"]
    assert result["factual_confidence_is_separate_from_approval"] is True


def test_informational_feedback_updates_claim_and_preserves_source():
    """Store factual correction with its source rather than treating it as generic praise."""
    state = LearnerSocialState(learner_id="learner-1")
    event = social_event(feedback=(
        SocialFeedback(
            "correction", "informational_correction",
            content="A cup is a container.", target_key="cup",
            factual_value="container", reliability=0.95, confirms=True,
            source_id="teacher-1",
        ),
    ))
    partner = PartnerModel("teacher-1", reliability=0.9)
    result = learn_from_social_feedback(event, state, partner, SocialLearningConfig())
    assert state.semantic_knowledge["cup"] > 0.35
    assert state.knowledge_claims["cup"] == "container"
    assert state.source_ledger[-1]["source_id"] == "teacher-1"
    assert result["source_ledger_entries"] == 1


def test_scaffolding_increases_for_a_difficult_task_and_fades_after_competence():
    """Match hint strength to competence gap and reduce support after observed success."""
    event = social_event(difficulty=0.95, competence_observation=0.1)
    state = LearnerSocialState(learner_id="learner-1")
    config = SocialLearningConfig()
    high = select_scaffold(event, state, config)
    skilled_event = social_event(difficulty=0.3, competence_observation=0.98)
    low = select_scaffold(skilled_event, state, config)
    assert high["support_level"] > low["support_level"]
    assert low["scaffold_type"] == "fade_support_and_test_transfer"


def test_self_explanation_scores_missing_terms_and_improves_retrieval_strength():
    """Measure term/relation coverage and record gaps instead of accepting fluent but incomplete explanation."""
    state = LearnerSocialState(learner_id="learner-1")
    event = social_event(
        learner_attempt="A cup is a container.",
        required_explanation_terms=("cup", "container", "red"),
        required_relations=("cup is a container",),
    )
    result = evaluate_explanation(event, state, SocialLearningConfig())
    assert result["completeness"] < 1.0
    assert "missing_term:red" in result["knowledge_gaps"]
    assert state.retrieval_strengths["cup"] > 0.0


def test_perspective_comparison_weights_source_and_retains_disagreement():
    """Move beliefs toward a reliable partner claim without erasing the disagreement record."""
    state = LearnerSocialState(learner_id="learner-1", semantic_knowledge={"color": 0.1})
    partner = PartnerModel("partner-1", reliability=0.9)
    event = social_event(
        partner_claims={"color": 0.95},
        partner_evidence={"color": 0.95},
    )
    result = compare_perspectives(event, state, partner, SocialLearningConfig())
    assert state.semantic_knowledge["color"] > 0.1
    assert result["disagreements"]
    assert result["agreement_is_not_truth"] is True
    assert state.source_ledger[-1]["independence_assumed"] is False


def test_narrative_model_keeps_character_knowledge_agent_specific():
    """Assign story facts to characters who receive them, not automatically to the narrator or every agent."""
    event = social_event(
        narrative_events=(
            NarrativeEvent(
                "e1", "Ava sees a key.", agent_id="Ava",
                action="sees_key", goal="open_box", facts={"key_location": "table"},
                reveals_to=("Ava",),
            ),
            NarrativeEvent(
                "e2", "Ben enters the room.", agent_id="Ben",
                action="enters", goal="find_box",
                causal_parents=("e1",), reveals_to=(),
            ),
        )
    )
    state = LearnerSocialState(learner_id="learner-1")
    processor = SocialInteractionLearningProcessor("learner-1", state=state)
    result = processor.process(event)
    assert "key_location" in state.narrative_agent_states["Ava"]["known_facts"]
    assert "key_location" not in state.narrative_agent_states.get("Ben", {}).get("known_facts", [])
    assert result.narrative_model["graph_node_count"] == 2


def test_social_memory_binds_content_partner_context_and_feedback_source():
    """Persist a social episode with source claims, modality, confidence, and retrieval cues."""
    state = LearnerSocialState(learner_id="learner-1")
    event = social_event(
        feedback=(
            SocialFeedback("f1", "confirmation", "correct", target_key="cup",
                           factual_value="container", reliability=0.9, confirms=True,
                           source_id="partner-1"),
        )
    )
    result = SocialInteractionLearningProcessor("learner-1", state=state).process(event)
    memory = state.episodic_memories[-1]
    assert memory.partner_id == "partner-1"
    assert "gesture" in memory.modalities
    assert memory.source_claims["partner_id"] == "partner-1"
    retrieved = retrieve_social_memories(
        ("cup", "red", "look"), state.episodic_memories
    )
    assert retrieved
    assert result.social_memory["source_binding_preserved"] is True


def test_cross_partner_transfer_requires_multiple_partners_and_consistency():
    """Do not generalize one speaker's word usage until a second partner supplies matching evidence."""
    state = LearnerSocialState(learner_id="learner-1")
    processor = SocialInteractionLearningProcessor("learner-1", state=state)
    first = processor.process(social_event(event_id="partner-a", target_word="cup"))
    assert first.transfer["cross_partner_pattern"]["generalizable"] is False
    second = processor.process(
        social_event(event_id="partner-b", partner_id="partner-2", target_word="cup")
    )
    assert second.transfer["cross_partner_pattern"]["partner_count"] == 2
    assert second.transfer["cross_partner_pattern"]["generalizable"] is True


def test_full_processor_returns_sixteen_operations_and_biological_diagnostics():
    """Run all stages and expose the coordinated distributed social-learning surrogates."""
    result = SocialInteractionLearningProcessor("learner-1").process(social_event())
    assert len(result.operation_trace) == 16
    assert [item["operation_id"] for item in result.operation_trace] == [
        item.operation_id for item in OPERATION_REGISTRY
    ]
    assert result.biological["superior_temporal"]["processing_count"] == 1
    assert "temporoparietal_joint_attention" in result.biological
    assert "hippocampal_memory" in result.biological
    assert result.output["agreement_is_not_truth"] is True
    assert result.output["social_synchrony_is_not_proof_of_understanding"] is True


def test_sqlite_annotation_registration_and_all_sixteen_operation_events(tmp_path):
    """Index every Point 14 callable and persist all operation traces in the ORE logger."""
    database_path = tmp_path / "point14.sqlite3"
    result = SocialInteractionLearningProcessor("learner-1").process(social_event())
    from ore_social_conversational_learning import register_annotations, record_result_to_sqlite

    catalogue = register_annotations(database_path)
    assert catalogue["files_seen"] >= 5
    database = LoggerDatabase(database_path)
    assert database.summary()["symbols"] > 0
    logged = record_result_to_sqlite(result, database_path)

    with sqlite3.connect(database_path) as connection:
        run = connection.execute(
            "SELECT status, event_count FROM runs WHERE run_id = ?",
            (logged["run_id"],),
        ).fetchone()
        event_count = connection.execute(
            "SELECT COUNT(*) FROM events WHERE run_id = ?",
            (logged["run_id"],),
        ).fetchone()[0]
    assert run == ("completed", 16)
    assert event_count == 16


def test_invalid_contracts_reject_duplicate_signal_ids_and_empty_partner():
    """Protect the social learning boundary against invalid identity or cue records."""
    with pytest.raises(ValueError):
        SocialInteractionEvent(
            event_id="bad", learner_id="learner", partner_id="partner",
            signals=(
                SocialSignal("same", "gaze"),
                SocialSignal("same", "gesture"),
            ),
        )
    with pytest.raises(ValueError):
        PartnerModel("")
