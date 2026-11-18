"""Focused tests for Point 13 affective modulation, motivation, attention, and memory."""

from __future__ import annotations

import sqlite3

import pytest

from ore_logger.database import LoggerDatabase
from ore_emotion_motivation import (
    AffectiveAppraisal,
    EmotionLearningConfig,
    EmotionMotivationMemoryProcessor,
    EventFeature,
    LearningExperience,
    LearningState,
    MemoryTrace,
    RewardFeedback,
    allocate_attention,
    allocate_effort,
    compute_attentional_priorities,
    consolidate_memory_trace,
    estimate_control_capacity,
    estimate_neuromodulatory_gain,
    measure_attention_memory_tradeoffs,
    retrieve_memories,
    update_reward_values,
    OPERATION_REGISTRY,
)


def make_experience(**overrides):
    """Create a multi-detail learning event with salient, peripheral, and contextual features."""
    values = {
        "event_id": "test-event",
        "goal": "learn_the_procedure",
        "task_demands": 0.45,
        "novelty": 0.35,
        "uncertainty": 0.15,
        "threat": 0.20,
        "reward_relevance": 0.85,
        "valence_hint": 0.25,
        "arousal_hint": 0.45,
        "sleep_replay_support": 0.60,
        "features": (
            EventFeature(
                "main_instruction", value=1.0, salience=0.95,
                goal_relevance=1.0, centrality=1.0, valence=0.2,
                arousal=0.55, tags=("procedure", "safety"), role="central",
            ),
            EventFeature(
                "warning_signal", value=0.9, salience=0.8,
                goal_relevance=0.8, centrality=0.85, valence=-0.4,
                arousal=0.9, tags=("procedure", "warning"), role="central",
            ),
            EventFeature(
                "room_background", value=0.6, salience=0.15,
                goal_relevance=0.1, centrality=0.1, valence=0.0,
                arousal=0.0, tags=("environment",), role="peripheral",
            ),
            EventFeature(
                "time_location", value=0.7, salience=0.3,
                goal_relevance=0.6, centrality=0.35, valence=0.0,
                arousal=0.1, tags=("context", "environment"), role="context",
            ),
        ),
        "context": {"feature_relevance": {"time_location": 0.85}, "room": "lab"},
        "retrieval_cues": ("procedure", "safety"),
        "metadata": {"elapsed_hours": 2.0},
    }
    values.update(overrides)
    return LearningExperience(**values)


def test_registry_contains_exact_sixteen_source_ordered_operations():
    """Require all Point 13 algorithmic stages exactly once and in their source order."""
    assert len(OPERATION_REGISTRY) == 16
    assert [item.operation_id for item in OPERATION_REGISTRY] == [
        f"P13-{index:02d}" for index in range(1, 17)
    ]


def test_valence_and_arousal_are_separate_dimensions():
    """Permit an event to be highly arousing without requiring negative valence."""
    experience = make_experience(
        threat=0.0, reward_relevance=0.0, valence_hint=0.0,
        arousal_hint=0.95,
        features=(
            EventFeature(
                "loud_music", salience=0.9, goal_relevance=0.3,
                valence=0.0, arousal=0.95, role="central",
            ),
        ),
    )
    result = EmotionMotivationMemoryProcessor().process(experience)
    assert result.appraisal.arousal > 0.7
    assert abs(result.appraisal.valence) < 0.15


def test_goal_relevance_can_outweigh_a_more_salient_distractor():
    """Show that top-down relevance can change priority even when a distractor is more salient."""
    features = (
        EventFeature(
            "goal_item", salience=0.30, goal_relevance=1.0,
            centrality=0.5, role="central",
        ),
        EventFeature(
            "salient_distractor", salience=1.0, goal_relevance=0.10,
            centrality=0.5, role="peripheral",
        ),
    )
    appraisal = AffectiveAppraisal(
        valence=0.0, arousal=0.2, significance=0.1,
        threat=0.0, reward_relevance=0.0, novelty=0.0, uncertainty=0.0,
    )
    priorities = compute_attentional_priorities(
        features, goal_strength=1.0, appraisal=appraisal,
        config=EmotionLearningConfig(),
    )
    assert priorities["goal_item"] > priorities["salient_distractor"]


def test_attention_respects_its_finite_resource_budget():
    """Check that softmax competition distributes no more than the available capacity."""
    config = EmotionLearningConfig(attention_capacity=0.55)
    allocation = allocate_attention(
        {"one": 0.9, "two": 0.4, "three": 0.1}, config, control_capacity=0.8
    )
    assert sum(allocation.weights.values()) == pytest.approx(0.44)
    assert allocation.capacity_budget == pytest.approx(0.44)
    assert allocation.weights["one"] > allocation.weights["three"]


def test_neuromodulatory_gain_has_a_task_dependent_inverted_u_shape():
    """Verify that moderate arousal can support higher model gain than very low or high arousal."""
    config = EmotionLearningConfig(arousal_optimum=0.5)
    middle = estimate_neuromodulatory_gain(
        {"fast_arousal": 0.5, "norepinephrine_gain": 0.8}, config
    )
    low = estimate_neuromodulatory_gain(
        {"fast_arousal": 0.0, "norepinephrine_gain": 0.8}, config
    )
    high = estimate_neuromodulatory_gain(
        {"fast_arousal": 1.0, "norepinephrine_gain": 0.8}, config
    )
    assert middle > low
    assert middle > high


def test_slow_stress_load_is_distinct_from_fast_arousal_and_ne_proxy():
    """Ensure the state stores separate arousal, LC-gain, and slower HPA-like values."""
    result = EmotionMotivationMemoryProcessor().process(make_experience())
    assert result.arousal_state.fast_arousal > result.arousal_state.slow_stress_load
    assert 0.0 <= result.arousal_state.norepinephrine_gain <= 1.0
    assert 0.0 <= result.arousal_state.glucocorticoid_load <= 1.0
    assert "hpa_axis" in result.biological
    assert "locus_coeruleus" in result.biological


def test_control_capacity_is_not_monotonic_in_arousal_and_is_fatigue_sensitive():
    """Model reduced broad control at extreme arousal or high fatigue without erasing memory effects."""
    config = EmotionLearningConfig(arousal_optimum=0.5)
    mid = estimate_control_capacity(
        {"fast_arousal": 0.5, "slow_stress_load": 0.0}, 0.0, 0.2, config
    )
    low = estimate_control_capacity(
        {"fast_arousal": 0.0, "slow_stress_load": 0.0}, 0.0, 0.2, config
    )
    high = estimate_control_capacity(
        {"fast_arousal": 1.0, "slow_stress_load": 0.0}, 0.0, 0.2, config
    )
    tired = estimate_control_capacity(
        {"fast_arousal": 0.5, "slow_stress_load": 0.0}, 0.9, 0.2, config
    )
    assert mid > low
    assert mid > high
    assert mid > tired


def test_emotional_and_goal_relevant_central_features_encode_more_strongly():
    """Confirm selective encoding permits central salient details to outlast peripheral details."""
    result = EmotionMotivationMemoryProcessor().process(make_experience())
    strengths = result.memory_trace.feature_strengths
    assert strengths["main_instruction"] > strengths["room_background"]
    assert strengths["warning_signal"] > strengths["room_background"]
    assert result.memory_trace.central_feature_ids
    assert result.memory_trace.peripheral_feature_ids


def test_reward_prediction_error_updates_value_estimates_in_expected_direction():
    """Update a state value from an unexpectedly positive reward using a TD-style rule."""
    state = LearningState()
    state.motivation.state_values["before"] = 0.2
    state.motivation.state_values["after"] = 0.5
    delta, diagnostics = update_reward_values(
        RewardFeedback(
            state_id="before", action_id="study",
            reward=1.0, next_state_id="after",
        ),
        state, EmotionLearningConfig(reward_learning_rate=0.2, discount_factor=0.9),
    )
    assert delta == pytest.approx(1.25)
    assert state.motivation.state_values["before"] == pytest.approx(0.45)
    assert diagnostics["updated_action_value"] > 0.0


def test_effort_allocation_accounts_for_reward_value_and_cost():
    """Reduce selected effort when the expected benefit is low and cognitive/fatigue cost is high."""
    experience = make_experience()
    fresh = LearningState()
    high_value = allocate_effort(
        experience, fresh, control_capacity=0.9,
        expected_value=1.0, config=EmotionLearningConfig(),
    )
    tired = LearningState()
    tired.arousal.fatigue = 1.0
    low_value = allocate_effort(
        experience, tired, control_capacity=0.2,
        expected_value=0.05, config=EmotionLearningConfig(),
    )
    assert high_value.effort > low_value.effort
    assert low_value.recommended_strategy in {
        "change_strategy_or_seek_support", "pause_or_disengage"
    }


def test_elaborative_processing_adds_retrieval_links_when_effort_is_justified():
    """Connect new material to prior semantic tags only when effort is allocated."""
    state = LearningState()
    state.concept_strengths["safety"] = 0.9
    experience = make_experience()
    result = EmotionMotivationMemoryProcessor(state=state).process(experience)
    assert "safety" in result.memory_trace.context_tags
    assert result.memory_trace.elaboration_links == ("safety",)
    assert result.operation_trace[10]["functional_output"]["elaboration_performed"] is True


def test_consolidation_is_separate_from_encoding_and_decays_with_delay():
    """Give near-term modulation and replay more influence than a much later single update."""
    trace = MemoryTrace(
        event_id="trace",
        feature_strengths={"central": 0.8, "peripheral": 0.3},
        encoded_feature_ids=("central", "peripheral"),
        central_feature_ids=("central",),
        peripheral_feature_ids=("peripheral",),
        relation_links=(), context_tags=("context",),
        encoding_strength=0.6, consolidation_strength=0.6,
        amygdala_modulation=0.8,
    )
    config = EmotionLearningConfig(consolidation_rate=0.3)
    immediate = consolidate_memory_trace(trace, 0.0, 0.8, config)
    delayed = consolidate_memory_trace(trace, 48.0, 0.8, config)
    assert immediate.encoding_strength == trace.encoding_strength
    assert immediate.consolidation_strength > delayed.consolidation_strength
    assert immediate.consolidation_strength >= trace.consolidation_strength


def test_retrieval_keeps_vividness_confidence_and_accuracy_distinct():
    """Allow a vivid, confident selective memory to have incomplete objective detail coverage."""
    state = LearningState()
    trace = MemoryTrace(
        event_id="emotional-episode",
        feature_strengths={"flash": 0.98, "background": 0.05, "location": 0.08},
        encoded_feature_ids=("flash", "background", "location"),
        central_feature_ids=("flash",),
        peripheral_feature_ids=("background", "location"),
        relation_links=(), context_tags=("flash", "storm"),
        encoding_strength=0.85, consolidation_strength=0.90,
        amygdala_modulation=0.95,
    )
    state.memory_traces.append(trace.to_dict())
    recalled = retrieve_memories(
        ("flash", "storm"), state,
        ground_truth_event_id="emotional-episode",
    )
    assert len(recalled) == 1
    assert recalled[0].confidence > recalled[0].accuracy
    assert recalled[0].accuracy == pytest.approx(1 / 3)
    assert recalled[0].recalled_central_ids == ("flash",)


def test_attention_memory_tradeoff_metrics_capture_contextual_omission():
    """Measure attention concentration and unequal central/peripheral memory strength."""
    experience = make_experience()
    result = EmotionMotivationMemoryProcessor().process(experience)
    metrics = measure_attention_memory_tradeoffs(
        experience, result.attention, result.memory_trace
    )
    assert 0.0 <= metrics["attention_entropy"] <= 1.0
    assert metrics["central_mean_strength"] > metrics["peripheral_mean_strength"]
    assert metrics["central_peripheral_ratio"] > 1.0
    assert 0.0 <= metrics["context_completeness"] <= 1.0


def test_reappraisal_updates_future_associations_without_rewriting_episode_trace():
    """Use feedback to revise future cue associations while preserving the prior event record."""
    state = LearningState()
    experience = make_experience()
    result = EmotionMotivationMemoryProcessor(state=state).process(experience)
    old_trace = result.memory_trace.to_dict()
    state.associative_weights["main_instruction"] = 0.5
    from ore_emotion_motivation import reappraise_and_update_associations

    updated = reappraise_and_update_associations(
        experience, result.appraisal, prediction_error=-0.8,
        state=state, learning_rate=0.4,
    )
    assert updated["association_count_updated"] > 0
    assert state.associative_weights["main_instruction"] < 0.5
    assert result.memory_trace.to_dict() == old_trace


def test_full_processor_returns_all_stages_and_persists_state():
    """Run all 16 stages and verify state, memory, effort, and reward diagnostics."""
    state = LearningState()
    processor = EmotionMotivationMemoryProcessor(state=state)
    result = processor.process(
        make_experience(),
        reward_feedback=RewardFeedback(
            state_id="learning", action_id="study", reward=0.9,
            next_state_id="practiced",
        ),
    )
    assert len(result.operation_trace) == 16
    assert [item["operation_id"] for item in result.operation_trace] == [
        item.operation_id for item in OPERATION_REGISTRY
    ]
    assert result.operation_trace[8]["functional_output"]["reward_prediction_error"] > 0
    assert state.event_count == 1
    assert len(state.memory_traces) == 1
    assert result.output["confidence_is_not_accuracy"] is True
    assert result.biological["scientific_scope"]


def test_sqlite_annotation_registration_and_sixteen_operation_events(tmp_path):
    """Index all Point 13 classes/functions and persist the full operation trace in ORE SQLite."""
    database_path = tmp_path / "point13.sqlite3"
    result = EmotionMotivationMemoryProcessor().process(make_experience())
    catalogue = __import__(
        "ore_emotion_motivation", fromlist=["register_annotations"]
    ).register_annotations(database_path)
    assert catalogue["files_seen"] >= 5
    database = LoggerDatabase(database_path)
    assert database.summary()["symbols"] > 0

    from ore_emotion_motivation import record_result_to_sqlite

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


def test_invalid_feature_and_reward_contracts_are_rejected():
    """Reject duplicate feature identities and invalid normalized metadata."""
    with pytest.raises(ValueError):
        LearningExperience(
            event_id="bad",
            features=(
                EventFeature("duplicate"),
                EventFeature("duplicate"),
            ),
        )
    with pytest.raises(ValueError):
        EventFeature("bad", salience=1.5)
