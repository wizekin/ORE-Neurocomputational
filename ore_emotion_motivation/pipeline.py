"""End-to-end 16-stage Point 13 emotion, motivation, attention, and memory processor."""
from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Mapping

from .biology import BiologicalEmotionNetwork
from .contracts import (
    EmotionLearningConfig, EmotionLearningResult, LearningExperience,
    LearningState, MemoryTrace, RewardFeedback,
)
from .functional import (
    allocate_attention, compute_attentional_priorities, divisive_normalization,
    estimate_control_capacity, estimate_motivational_value,
    measure_attention_memory_tradeoffs, modulate_feature_activity,
    reappraise_and_update_associations, retrieve_memories,
    update_arousal_state, update_generalized_knowledge,
)
from .registry import OPERATION_REGISTRY


def _trace_operation(
    trace: list[dict[str, Any]],
    index: int,
    functional: Mapping[str, Any],
    biological: Mapping[str, Any] | None = None,
) -> None:
    """Append one canonical source-ordered operation trace record."""
    spec = OPERATION_REGISTRY[index]
    trace.append({
        "step": index + 1,
        "operation_id": spec.operation_id,
        "operation": spec.name,
        "status": "completed",
        "explanation": spec.purpose,
        "evidence_level": spec.evidence_level,
        "mathematical_form": spec.mathematical_form,
        "functional_output": dict(functional),
        "biological_output": dict(biological or {}),
    })


class EmotionMotivationMemoryProcessor:
    """Coordinate affective appraisal, attention, motivation, episodic encoding, and consolidation."""

    def __init__(
        self,
        config: EmotionLearningConfig | None = None,
        state: LearningState | None = None,
        biological: BiologicalEmotionNetwork | None = None,
    ) -> None:
        """Initialize a reusable Point 13 processor and its persistent learning state."""
        self.config = config or EmotionLearningConfig()
        self.state = state or LearningState()
        self.biological = biological or BiologicalEmotionNetwork()
        self.process_count = 0

    def process(
        self,
        experience: LearningExperience,
        reward_feedback: RewardFeedback | None = None,
        state: LearningState | None = None,
    ) -> EmotionLearningResult:
        """Execute all 16 Point 13 stages for a new experience and optional observed outcome."""
        self.process_count += 1
        active_state = state if state is not None else self.state
        bio = self.biological
        config = self.config
        trace: list[dict[str, Any]] = []

        # 1. Represent the experience without making early assumptions about its final meaning.
        event_vector = {
            item.feature_id: float(item.value) for item in experience.features
        }
        relation_candidates: dict[str, list[str]] = {}
        for item in experience.features:
            for tag in item.tags:
                relation_candidates.setdefault(tag, []).append(item.feature_id)
        _trace_operation(trace, 0, {
            "event_vector": event_vector,
            "feature_count": len(event_vector),
            "relation_tag_groups": relation_candidates,
            "context_keys": sorted(str(key) for key in experience.context),
        }, {"hippocampal_binding_input": len(event_vector)})

        # 2. Appraise affective significance; valence and arousal remain distinct.
        appraisal = bio.amygdala.appraise(experience)
        _trace_operation(trace, 1, appraisal.to_dict(), bio.amygdala.snapshot())

        # 3. Update separate fast and slow arousal proxies.
        arousal_update = update_arousal_state(active_state, appraisal, experience, config)
        glucocorticoid = bio.hpa_axis.update(active_state, appraisal)
        arousal_update["hpa_glucocorticoid_load_proxy"] = glucocorticoid
        _trace_operation(trace, 2, arousal_update, {
            "locus_coeruleus": bio.locus_coeruleus.snapshot(),
            "hpa_axis": bio.hpa_axis.snapshot(),
        })

        # 4. Estimate the active goal and allocate attention by salience and relevance.
        active_state.motivation.goal = experience.goal or active_state.motivation.goal
        bio.prefrontal.maintain_goal(active_state.motivation.goal, active_state)
        context_relevance_raw = experience.context.get("feature_relevance", {})
        context_relevance = (
            context_relevance_raw if isinstance(context_relevance_raw, Mapping) else {}
        )
        priorities = compute_attentional_priorities(
            experience.features, active_state.motivation.goal_strength,
            appraisal, config, context_relevance,
        )
        control_capacity = estimate_control_capacity(
            active_state.arousal.to_dict(), active_state.arousal.fatigue,
            experience.task_demands, config,
        )
        attention = allocate_attention(priorities, config, control_capacity)
        _trace_operation(trace, 3, {
            "priorities": priorities,
            "attention_weights": dict(attention.weights),
            "selected_feature_ids": list(attention.selected_feature_ids),
            "attention_entropy": attention.entropy,
            "capacity_budget": attention.capacity_budget,
            "control_capacity": control_capacity,
        }, bio.prefrontal.snapshot())

        # 5. Apply an illustrative inverted-U state-dependent gain and population normalization.
        gain = bio.locus_coeruleus.update(active_state, config)
        modulated = modulate_feature_activity(experience, attention, gain)
        normalized_activity = divisive_normalization(
            modulated, attention.weights, epsilon=1e-8
        )
        _trace_operation(trace, 4, {
            "neuromodulatory_gain_proxy": gain,
            "modulated_activity": modulated,
            "divisively_normalized_activity": normalized_activity,
            "gain_is_task_and_state_dependent": True,
        }, bio.locus_coeruleus.snapshot())

        # 6. Encode attended information into a selective episodic trace.
        memory_trace = bio.hippocampus.encode(
            experience, attention, appraisal, active_state, config
        )
        amygdala_modulation = bio.amygdala.modulate_memory(
            appraisal,
            max((item.salience for item in experience.features), default=0.0),
        )
        _trace_operation(trace, 5, {
            "feature_strengths": dict(memory_trace.feature_strengths),
            "central_feature_ids": list(memory_trace.central_feature_ids),
            "peripheral_feature_ids": list(memory_trace.peripheral_feature_ids),
            "encoding_strength": memory_trace.encoding_strength,
            "amygdala_modulation": amygdala_modulation,
        }, {
            "hippocampus": bio.hippocampus.snapshot(),
            "amygdala": bio.amygdala.snapshot(),
        })

        # 7. Evaluate stress-related control capacity and selective detail trade-offs.
        _trace_operation(trace, 6, {
            "control_capacity": control_capacity,
            "slow_stress_load": active_state.arousal.slow_stress_load,
            "fatigue": active_state.arousal.fatigue,
            "encoding_and_control_capacity_are_separate": True,
        }, {
            "prefrontal": bio.prefrontal.snapshot(),
            "hpa_axis": bio.hpa_axis.snapshot(),
        })

        # 8. Estimate expected motivational value before observed feedback.
        expected_reward, initial_utility = estimate_motivational_value(experience, active_state)
        _trace_operation(trace, 7, {
            "expected_reward": expected_reward,
            "pre_feedback_utility": initial_utility,
            "effort_cost_weight": config.effort_cost_weight,
            "fatigue_cost_weight": config.fatigue_cost_weight,
        }, {"dopamine_striatal": bio.dopamine_striatal.snapshot()})

        # 9. If an outcome is supplied, update expected state/action values with a TD-style error.
        prediction_error, value_diagnostics = bio.dopamine_striatal.update(
            reward_feedback, active_state, config
        )
        effective_value = (
            active_state.motivation.expected_reward
            if reward_feedback is not None else expected_reward
        )
        _trace_operation(trace, 8, {
            "reward_prediction_error": prediction_error,
            "value_diagnostics": value_diagnostics,
            "effective_value_for_effort": effective_value,
        }, bio.dopamine_striatal.snapshot())

        # 10. Allocate effort under expected benefit, fatigue, and resource costs.
        effort = bio.prefrontal.allocate_effort(
            experience, active_state, max(0.0, effective_value), config
        )
        monitoring = bio.cingulo_opercular.monitor(effort, prediction_error)
        _trace_operation(trace, 9, {
            "effort_decision": effort.to_dict(),
            "monitoring": monitoring,
        }, {
            "prefrontal": bio.prefrontal.snapshot(),
            "cingulo_opercular": bio.cingulo_opercular.snapshot(),
        })

        # 11. Elaborate only when the expected value justifies extra processing.
        memory_trace = bio.cortical_elaboration.elaborate(
            experience, memory_trace, active_state, effort
        )
        _trace_operation(trace, 10, {
            "elaboration_links": list(memory_trace.elaboration_links),
            "feature_strengths_after_elaboration": dict(memory_trace.feature_strengths),
            "elaboration_performed": effort.continue_task,
        }, bio.cortical_elaboration.snapshot())

        # 12. Consolidate separately from initial encoding; use a time-limited modulation window.
        elapsed_hours = float(experience.metadata.get("elapsed_hours", 1.0))
        if elapsed_hours < 0:
            raise ValueError("metadata.elapsed_hours must be non-negative")
        memory_trace = bio.hippocampus.consolidate(
            memory_trace, elapsed_hours,
            experience.sleep_replay_support, config,
        )
        _trace_operation(trace, 11, {
            "elapsed_hours": elapsed_hours,
            "sleep_replay_support": experience.sleep_replay_support,
            "encoding_strength": memory_trace.encoding_strength,
            "consolidation_strength": memory_trace.consolidation_strength,
        }, bio.hippocampus.snapshot())

        # 13. Retrieve older memories by current cues before storing the new trace.
        retrieval_cues = list(experience.retrieval_cues)
        retrieval_cues.extend(item.feature_id for item in experience.features)
        retrieval_cues.extend(tag for item in experience.features for tag in item.tags)
        retrieval_cues.extend(str(key).lower() for key in experience.context)
        ground_truth_id = experience.metadata.get("ground_truth_event_id")
        retrieved = retrieve_memories(
            retrieval_cues, active_state,
            temperature=config.retrieval_temperature,
            limit=5,
            ground_truth_event_id=str(ground_truth_id) if ground_truth_id else None,
            threshold=config.minimum_retrieval_similarity,
        )
        _trace_operation(trace, 12, {
            "retrieval_cues": sorted(set(retrieval_cues)),
            "retrieved_event_ids": [item.event_id for item in retrieved],
            "vividness_confidence_accuracy_separated": True,
        }, bio.hippocampus.snapshot())

        # 14. Quantify narrowed attention and central/peripheral/context trade-offs.
        tradeoffs = measure_attention_memory_tradeoffs(
            experience, attention, memory_trace
        )
        _trace_operation(trace, 13, tradeoffs, {
            "amygdala": bio.amygdala.snapshot(),
            "prefrontal": bio.prefrontal.snapshot(),
        })

        # 15. Reappraise outcomes and revise future associations, not the stored event itself.
        reappraisal = reappraise_and_update_associations(
            experience, appraisal, prediction_error, active_state,
            config.reward_learning_rate,
        )
        _trace_operation(trace, 14, reappraisal, {
            "amygdala": bio.amygdala.snapshot(),
            "dopamine_striatal": bio.dopamine_striatal.snapshot(),
        })

        # 16. Retain the episode and update generalized knowledge only through shared evidence.
        generalized = update_generalized_knowledge(
            experience, memory_trace, active_state, config.encoding_learning_rate
        )
        active_state.memory_traces.append(memory_trace.to_dict())
        if len(active_state.memory_traces) > config.max_memory_traces:
            active_state.memory_traces = active_state.memory_traces[-config.max_memory_traces:]
        active_state.event_count += 1
        active_state.motivation.progress = clamp(
            active_state.motivation.progress + effort.expected_progress * 0.10
        )
        active_state.motivation.active = effort.continue_task
        _trace_operation(trace, 15, {
            "stored_event_count": len(active_state.memory_traces),
            "generalized_knowledge_updates": generalized,
            "updated_motivation": active_state.motivation.to_dict(),
            "uncertainty_about_omitted_details": 1.0 - memory_trace.encoding_strength,
        }, bio.snapshot())

        output = {
            "event_id": experience.event_id,
            "appraisal": appraisal.to_dict(),
            "attention_selected": list(attention.selected_feature_ids),
            "recommended_strategy": effort.recommended_strategy,
            "effort_allocated": effort.effort,
            "memory_encoding_strength": memory_trace.encoding_strength,
            "memory_consolidation_strength": memory_trace.consolidation_strength,
            "retrieved_prior_events": [item.event_id for item in retrieved],
            "attention_memory_tradeoffs": tradeoffs,
            "prediction_error": prediction_error,
            "confidence_is_not_accuracy": True,
            "biological_model_is_a_functional_surrogate": True,
        }
        active_state.appraisal_history.append({
            "event_id": experience.event_id,
            "appraisal": appraisal.to_dict(),
            "prediction_error": prediction_error,
        })
        return EmotionLearningResult(
            event_id=experience.event_id,
            appraisal=appraisal,
            arousal_state=active_state.arousal,
            attention=attention,
            effort=effort,
            memory_trace=memory_trace,
            retrieved_memories=retrieved,
            reward_prediction_error=prediction_error,
            prediction_values=value_diagnostics,
            tradeoff_metrics=tradeoffs,
            operation_trace=tuple(trace),
            biological=bio.snapshot(),
            final_state=active_state.to_dict(),
            output=output,
        )


def register_annotations(
    database_path: str | Path,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Index every Point 13 package class and callable docstring in the ORE SQLite catalogue."""
    from ore_logger.catalog import sync_repository
    from ore_logger.database import LoggerDatabase

    root = (Path(repository_root).expanduser().resolve() if repository_root
            else Path(__file__).resolve().parents[1])
    return sync_repository(
        LoggerDatabase(database_path), root,
        include_paths={"ore_emotion_motivation"},
    )


def record_result_to_sqlite(
    result: EmotionLearningResult,
    database_path: str | Path,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Persist the 16-stage Point 13 operation trace and index all package annotations."""
    from ore_logger.catalog import sync_repository
    from ore_logger.database import LoggerDatabase

    root = (Path(repository_root).expanduser().resolve() if repository_root
            else Path(__file__).resolve().parents[1])
    database = LoggerDatabase(database_path)
    catalogue = sync_repository(
        database, root, include_paths={"ore_emotion_motivation"}
    )
    run_id = database.create_run(
        script_path="ore_emotion_motivation/point13",
        module_name="ore_emotion_motivation.pipeline",
        entry_kind="emotion_motivation_memory",
        working_directory=str(root),
    )
    started = time.perf_counter()
    try:
        for sequence, item in enumerate(result.operation_trace, start=1):
            database.record_event(
                run_id=run_id,
                sequence_no=sequence,
                event_kind="operation",
                source_path="ore_emotion_motivation/pipeline.py",
                elapsed_ms=0.0,
                metadata={
                    "operation_id": item.get("operation_id"),
                    "operation": item.get("operation"),
                    "status": item.get("status"),
                    "evidence_level": item.get("evidence_level"),
                    "mathematical_form": item.get("mathematical_form"),
                    "functional_output": item.get("functional_output"),
                    "biological_output": item.get("biological_output"),
                },
            )
        database.finish_run(
            run_id, "completed",
            duration_ms=round((time.perf_counter() - started) * 1000.0, 3),
        )
    except Exception as exc:
        database.finish_run(
            run_id, "failed",
            duration_ms=round((time.perf_counter() - started) * 1000.0, 3),
            error_summary=type(exc).__name__ + ": " + str(exc),
        )
        raise
    return {
        "run_id": run_id,
        "annotation_catalogue": catalogue,
        "database_summary": database.summary(),
    }
