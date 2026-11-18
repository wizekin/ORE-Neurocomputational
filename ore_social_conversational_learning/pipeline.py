"""End-to-end 16-stage Point 14 social interaction and conversational-learning processor."""
from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Mapping

from .biology import BiologicalSocialLearningNetwork
from .contracts import (
    LearnerSocialState, PartnerModel, SocialInteractionEvent,
    SocialLearningConfig, SocialLearningResult,
)
from .functional import (
    build_narrative_model, compare_perspectives, create_social_memory,
    detect_conversational_mismatch, evaluate_explanation,
    get_or_create_partner, learn_from_social_feedback,
    update_partner_and_cross_partner_models,
)
from .registry import OPERATION_REGISTRY


def _trace_operation(
    trace: list[dict[str, Any]],
    index: int,
    functional: Mapping[str, Any],
    biological: Mapping[str, Any] | None = None,
) -> None:
    """Append one source-ordered Point 14 operation with its mathematical and biological notes."""
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


class SocialInteractionLearningProcessor:
    """Coordinate joint attention, intention inference, dialogue repair, scaffolding, and social memory."""

    def __init__(
        self,
        learner_id: str,
        config: SocialLearningConfig | None = None,
        state: LearnerSocialState | None = None,
        biological: BiologicalSocialLearningNetwork | None = None,
    ) -> None:
        """Initialize one learner's persistent social-learning state and biological surrogates."""
        self.learner_id = learner_id
        self.config = config or SocialLearningConfig()
        self.state = state or LearnerSocialState(learner_id=learner_id)
        if self.state.learner_id != learner_id:
            raise ValueError("processor learner_id must match the supplied LearnerSocialState")
        self.biological = biological or BiologicalSocialLearningNetwork()
        self.process_count = 0

    def process(
        self,
        event: SocialInteractionEvent,
        state: LearnerSocialState | None = None,
    ) -> SocialLearningResult:
        """Run the full 16-stage social interaction and conversational-learning sequence."""
        if event.learner_id != self.learner_id:
            raise ValueError("interaction learner_id must match this processor")
        self.process_count += 1
        active_state = state if state is not None else self.state
        if active_state.learner_id != event.learner_id:
            raise ValueError("interaction learner_id must match the supplied state")
        config = self.config
        bio = self.biological
        trace: list[dict[str, Any]] = []
        partner = get_or_create_partner(active_state, event.partner_id)
        active_state.current_goal = event.learning_goal or active_state.current_goal

        # 1. Establish the learner, partner, shared environment, and event stream.
        episode = {
            "event_id": event.event_id,
            "learner_id": event.learner_id,
            "partner_id": event.partner_id,
            "goal": active_state.current_goal,
            "utterance": event.utterance,
            "signal_count": len(event.signals),
            "referent_ids": [item.referent_id for item in event.referents],
            "context_keys": sorted(str(key) for key in event.context),
            "partner_model_before": partner.to_dict(),
            "social_and_linguistic_channels_separate": True,
        }
        _trace_operation(trace, 0, episode, {
            "distributed_network_initialized": True,
            "hippocampal_memory": bio.hippocampal_memory.snapshot(),
        })

        # 2. Detect socially relevant cues and allocate attention to the partner.
        perception = bio.superior_temporal.process(event, config)
        attention = perception["attention"]
        attention_weights = attention.get("attention_weights", {})
        attention_quality = sum(
            signal.reliability * float(attention_weights.get(signal.signal_id, 0.0))
            for signal in event.signals
        )
        attention["quality"] = attention_quality
        _trace_operation(trace, 1, {
            **attention,
            "partner_signal_count": len(event.signals),
            "partner_directed_signals": [
                item.signal_id for item in event.signals if item.directed_to_learner >= 0.5
            ],
        }, bio.superior_temporal.snapshot())

        # 3. Establish shared attention, retaining uncertainty if the referent is not confirmed.
        joint_attention = bio.temporoparietal_joint_attention.update(
            event, attention, config
        )
        active_state.current_shared_focus = (
            str(joint_attention["partner_focus"])
            if joint_attention["status"] == "coordinated"
            else active_state.current_shared_focus
        )
        _trace_operation(trace, 2, joint_attention, bio.temporoparietal_joint_attention.snapshot())

        # 4. Infer communicative intention and the partner's likely knowledge state.
        intention = bio.medial_prefrontal_mentalizing.infer(event, partner, config)
        _trace_operation(trace, 3, intention, bio.medial_prefrontal_mentalizing.snapshot())

        # 5. Integrate speech, gesture, gaze, prosody, face, and action with source tags retained.
        multimodal = bio.frontotemporal_language.integrate(event, attention, config)
        _trace_operation(trace, 4, multimodal, bio.frontotemporal_language.snapshot())

        # 6. Map the target word or relation to a shared-scene referent using uncertain evidence.
        from .functional import ground_word_to_shared_referent

        grounding = ground_word_to_shared_referent(
            event, active_state, joint_attention, intention, config
        )
        bio.frontotemporal_language.record_grounding()
        _trace_operation(trace, 5, grounding, bio.frontotemporal_language.snapshot())

        # 7. Predict whether the partner is continuing, yielding, or requesting repair.
        turn = bio.inferior_frontal_turns.predict(event, attention, config)
        active_state.last_turn_state = str(turn["predicted_state"])
        _trace_operation(trace, 6, turn, bio.inferior_frontal_turns.snapshot())

        # 8. Adapt a response plan to the partner's estimated knowledge and information need.
        audience = bio.prefrontal_control.design_response(event, partner, intention)
        _trace_operation(trace, 7, audience, bio.prefrontal_control.snapshot())

        # 9. Detect conversational mismatch and retain alternatives for repair.
        repair = detect_conversational_mismatch(event, intention, config)
        if repair["mismatch_detected"]:
            active_state.unresolved_questions.append(
                "Repair needed in event " + event.event_id + ": "
                + ", ".join(repair["mismatch_types"])
            )
        _trace_operation(trace, 8, repair, {
            "language_monitoring": "prediction_error_and_repair_surrogate",
            "prefrontal_revision": "alternative_interpretations_reopened"
            if repair["mismatch_detected"] else "monitoring",
        })

        # 10. Update evidence and reward as separate channels.
        feedback = learn_from_social_feedback(event, active_state, partner, config)
        social_errors = [
            item["prediction_error"] for item in feedback["social_and_outcome_rewards"]
        ]
        reward_diagnostics = bio.ventral_striatal_reward.update(social_errors)
        feedback["reward_diagnostics"] = reward_diagnostics
        _trace_operation(trace, 9, feedback, bio.ventral_striatal_reward.snapshot())

        # 11. Choose an adaptive scaffold matched to competence and task difficulty.
        scaffolding = bio.prefrontal_control.scaffold(event, active_state, config)
        _trace_operation(trace, 10, scaffolding, bio.prefrontal_control.snapshot())

        # 12. Evaluate learner explanation, retrieval, relational coverage, and knowledge gaps.
        explanation = evaluate_explanation(event, active_state, config)
        _trace_operation(trace, 11, explanation, {
            "language_formulation": "explanation_generation_surrogate",
            "hippocampal_retrieval_support": "effortful_retrieval_and_reconstruction",
        })

        # 13. Compare the partner's perspective with the learner's belief using source reliability.
        perspective = compare_perspectives(event, active_state, partner, config)
        _trace_operation(trace, 12, perspective, {
            "mentalizing": bio.medial_prefrontal_mentalizing.snapshot(),
            "source_binding": "partner_claims_remain_source_attributed",
        })

        # 14. Construct story/event graphs with agent-specific knowledge, goals, and causal links.
        narrative = build_narrative_model(event, active_state, config)
        narrative_bio = bio.posterior_medial_narrative.integrate(narrative)
        _trace_operation(trace, 13, narrative, {
            **bio.posterior_medial_narrative.snapshot(),
            **narrative_bio,
        })

        # 15. Bind learned content to partner, context, feedback, confidence, and episode time.
        memory = create_social_memory(
            event, active_state, partner, attention, intention,
            grounding, feedback, config,
        )
        bio.hippocampal_memory.encode(event.event_id)
        bio.hippocampal_memory.retrieved(
            len(memory["retrieved_prior_episodes"])
        )
        _trace_operation(trace, 14, memory, bio.hippocampal_memory.snapshot())

        # 16. Adapt partner estimates and generalize only patterns repeated across partners.
        transfer = update_partner_and_cross_partner_models(
            event, active_state, partner, attention, intention, grounding, config
        )
        active_state.event_history.append({
            "event_id": event.event_id,
            "partner_id": event.partner_id,
            "goal": event.learning_goal,
            "intention": intention.get("most_likely_intention"),
            "joint_attention": joint_attention.get("status"),
            "grounded_referent": grounding.get("selected_referent"),
            "repair_detected": repair.get("mismatch_detected"),
            "timestamp": event.metadata.get("timestamp", ""),
        })
        _trace_operation(trace, 15, transfer, bio.snapshot())

        output = {
            "event_id": event.event_id,
            "partner_id": event.partner_id,
            "joint_attention_status": joint_attention["status"],
            "joint_attention_confidence": joint_attention["shared_attention_confidence"],
            "likely_intention": intention["most_likely_intention"],
            "intention_uncertainty_nats": intention["uncertainty_nats"],
            "grounded_referent": grounding.get("selected_referent"),
            "clarification_needed": bool(
                joint_attention["clarification_needed"]
                or intention["clarification_needed"]
                or grounding["clarification_needed"]
                or repair["mismatch_detected"]
            ),
            "turn_state": turn["predicted_state"],
            "learner_should_respond": turn["learner_should_respond"],
            "audience_response_plan": audience["response_plan"],
            "repair_action": repair["repair_action"],
            "scaffold_type": scaffolding["scaffold_type"],
            "explanation_completeness": explanation["completeness"],
            "perspective_disagreement_count": len(perspective["disagreements"]),
            "social_episode_memory_id": event.event_id,
            "cross_partner_generalization_available": transfer["cross_partner_pattern"].get(
                "generalizable", False
            ),
            "agreement_is_not_truth": True,
            "social_synchrony_is_not_proof_of_understanding": True,
        }
        return SocialLearningResult(
            event_id=event.event_id,
            partner_id=event.partner_id,
            joint_attention=joint_attention,
            intention=intention,
            multimodal_representation=multimodal,
            semantic_grounding=grounding,
            turn_decision=turn,
            audience_design=audience,
            repair=repair,
            feedback_learning=feedback,
            scaffolding=scaffolding,
            explanation=explanation,
            perspective_update=perspective,
            narrative_model=narrative,
            social_memory=memory,
            transfer=transfer,
            operation_trace=tuple(trace),
            biological=bio.snapshot(),
            final_state=active_state.to_dict(),
            output=output,
        )


def register_annotations(
    database_path: str | Path,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Index all Point 14 classes and callable annotations in the ORE SQLite catalogue."""
    from ore_logger.catalog import sync_repository
    from ore_logger.database import LoggerDatabase

    root = (
        Path(repository_root).expanduser().resolve()
        if repository_root else Path(__file__).resolve().parents[1]
    )
    return sync_repository(
        LoggerDatabase(database_path),
        root,
        include_paths={"ore_social_conversational_learning"},
    )


def record_result_to_sqlite(
    result: SocialLearningResult,
    database_path: str | Path,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Persist each of the 16 Point 14 operations and register class/function annotations in SQLite."""
    from ore_logger.catalog import sync_repository
    from ore_logger.database import LoggerDatabase

    root = (
        Path(repository_root).expanduser().resolve()
        if repository_root else Path(__file__).resolve().parents[1]
    )
    database = LoggerDatabase(database_path)
    catalogue = sync_repository(
        database, root, include_paths={"ore_social_conversational_learning"}
    )
    run_id = database.create_run(
        script_path="ore_social_conversational_learning/point14",
        module_name="ore_social_conversational_learning.pipeline",
        entry_kind="social_conversational_learning",
        working_directory=str(root),
    )
    started = time.perf_counter()
    try:
        for sequence, item in enumerate(result.operation_trace, start=1):
            database.record_event(
                run_id=run_id,
                sequence_no=sequence,
                event_kind="operation",
                source_path="ore_social_conversational_learning/pipeline.py",
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
