"""End-to-end 22-stage Point 04 testing-effect processor and SQLite recorder."""
from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from .biology import BiologicalTestingEffectNetwork
from .contracts import (
    Feedback, LearningTarget, RetrievalAttempt, TargetMemoryState,
    TestingEffectRequest, TestingEffectResult, TestingEffectState,
)
from .functional import (
    assess_attempt, assess_cue_quality, calculate_future_priority,
    choose_learning_action, classify_errors, delayed_retention_assessment,
    detect_test_failures, record_attempt, schedule_handoff, select_feedback,
    update_accessibility, update_cue_diversity, update_error_history,
    update_knowledge_graph,
)
from .registry import OPERATION_REGISTRY


def _append_operation(
    trace: list[dict[str, Any]], index: int,
    functional_output: Mapping[str, Any],
    biological_output: Mapping[str, Any] | None = None,
) -> None:
    """Append one source-ordered operation with its declared math and biological annotations."""
    spec = OPERATION_REGISTRY[index]
    trace.append({
        "step": index + 1,
        "source_step": spec.source_step,
        "operation_id": spec.operation_id,
        "operation": spec.name,
        "purpose": spec.purpose,
        "status": "completed",
        "evidence_level": spec.evidence_level,
        "mathematical_form": spec.mathematical_form,
        "functional_output": dict(functional_output),
        "biological_output": dict(biological_output or {}),
    })


def _target_content(target: LearningTarget) -> list[str]:
    """Collect target text fragments used to construct the distributed biological representation."""
    return [target.prompt, *target.components, *target.relations, *target.conditions]


def _assessment_summary(assessment: Mapping[str, Any]) -> dict[str, Any]:
    """Extract compact behavioral measures without discarding the full stored assessment."""
    return {
        "weighted_coverage": assessment["weighted_coverage"],
        "component_coverage": assessment["component_coverage"],
        "relation_coverage": assessment["relation_coverage"],
        "order_accuracy": assessment["order_accuracy"],
        "condition_accuracy": assessment["condition_accuracy"],
        "evidenced_accuracy": assessment["evidenced_accuracy"],
        "confidence": assessment["confidence"],
        "independent_retrieval": assessment["independent_retrieval"],
    }


class TestingEffectProcessor:
    """Coordinate retrieval attempts, correction, cue variation, delayed probes, and biological surrogates."""

    def __init__(
        self,
        state: TestingEffectState | None = None,
        biological: BiologicalTestingEffectNetwork | None = None,
    ) -> None:
        """Initialize persistent target memory and the interacting biological model components."""
        self.state = state or TestingEffectState()
        self.biological = biological or BiologicalTestingEffectNetwork(dimensions=64)

    def process(
        self,
        request: TestingEffectRequest,
        state: TestingEffectState | None = None,
    ) -> TestingEffectResult:
        """Process one target through all 22 source-aligned testing-effect operations."""
        active = state if state is not None else self.state
        target = request.target
        attempt = request.initial_attempt
        memory = active.targets.setdefault(target.target_id, TargetMemoryState(target.target_id))
        trace: list[dict[str, Any]] = []
        biological = self.biological

        # 1. Define and register the target before inspecting the learner's response.
        bio_encoding = biological.encode_target(target.target_id, _target_content(target), target.context)
        _append_operation(trace, 0, {
            "target_id": target.target_id,
            "target_kind": target.target_kind,
            "required_components": list(target.components),
            "required_relations": list(target.relations),
            "expected_order": list(target.expected_order),
            "criterion_source": target.criterion_source,
            "criterion_reliability": target.criterion_reliability,
        }, bio_encoding)

        # 2. Select and evaluate the retrieval cue.
        cue_quality = assess_cue_quality(target, attempt)
        _append_operation(trace, 1, {
            "cue": attempt.cue, "cue_type": attempt.cue_type,
            "cue_alignment": cue_quality,
            "valid_cue_type": attempt.cue_type in target.valid_cue_types
            or attempt.cue_type == "free_recall",
        }, {"prefrontal_goal_bias": "task-relevant cue selection; no answer representation exposed"})

        # 3. Establish the retrieval condition and record all assistance.
        independent_condition = attempt.aid_level == 0 and not attempt.guessed
        _append_operation(trace, 2, {
            "answer_source_hidden": attempt.aid_level == 0,
            "aid_level": attempt.aid_level,
            "guessed": attempt.guessed,
            "independent_retrieval_condition": independent_condition,
            "recognition_not_equated_with_recall": attempt.cue_type != "recognition",
        }, {"executive_goal_maintenance": "maintain target while answer source is externally absent"})

        # 4. Activate candidate memories from the partial cue.
        difficulty = 1.0 - (target.initial_encoding_quality
                            if target.initial_encoding_quality is not None else 0.5)
        bio_retrieval = biological.retrieve(
            target.target_id, attempt.cue, difficulty,
            cue_quality["relevance"], request.fatigue,
        )
        _append_operation(trace, 3, {
            "cue": attempt.cue, "candidate_search_requested": True,
            "cue_relevance": cue_quality["relevance"],
            "irrelevant_competition_control": True,
            "multiple_candidates_retained": True,
        }, bio_retrieval)

        # 5. Reconstruct and score content and relations while preserving inference labels.
        initial_assessment = assess_attempt(target, attempt)
        _append_operation(trace, 4, {
            "response": attempt.response,
            "direct_recollections": list(attempt.recalled_components),
            "inferred_components": list(attempt.inferred_components),
            "assessment": _assessment_summary(initial_assessment),
            "uncertain_units_preserved": True,
        }, {"hippocampal_cortical_reinstatement": bio_retrieval["pattern_completion"]})

        # 6. Maintain a compact response workspace and check structural completeness.
        workspace = {
            "active_components": list(attempt.recalled_components),
            "active_relations": list(attempt.recalled_relations),
            "missing_components": [key for key, value in initial_assessment["component_status"].items()
                                   if value == "missing"],
            "missing_relations": [key for key, value in initial_assessment["relation_status"].items()
                                  if value == "missing"],
            "chunk_count": len(attempt.recalled_components) + len(attempt.recalled_relations),
            "supported_coverage": initial_assessment["weighted_coverage"],
        }
        _append_operation(trace, 5, workspace, {
            "frontoparietal_workspace": "functional abstraction for goal maintenance and response organization",
            "working_set_size": workspace["chunk_count"],
        })

        # 7. Monitor confidence and prediction discrepancy before showing the criterion.
        monitor = biological.monitor_attempt(
            memory.accessibility_ema, initial_assessment["evidenced_accuracy"],
            {"retrieved": initial_assessment["weighted_coverage"],
             "inferred": len(attempt.inferred_components) / max(1, len(target.components)),
             "missing": len(workspace["missing_components"]) / max(1, len(target.components))},
        )
        _append_operation(trace, 6, {
            "confidence": attempt.confidence,
            "claim_confidence": dict(attempt.claim_confidence),
            "confidence_error_proxy": initial_assessment["confidence_error_proxy"],
            "response_time_ms": attempt.response_time_ms,
            "monitoring_discrepancy": monitor,
        }, {"anterior_cingulate_monitor": monitor})

        # 8. Commit the first attempt as immutable evidence.
        record_attempt(memory, attempt, initial_assessment, "initial_unaided_or_assisted_attempt")
        _append_operation(trace, 7, {
            "attempt_recorded": True,
            "original_response": attempt.response,
            "aid_level": attempt.aid_level,
            "confidence": attempt.confidence,
            "first_response_immutable": True,
            "attempt_history_count": len(memory.attempt_history),
        }, {"immutable_record": "software audit constraint, not a biological storage claim"})

        # 9. Retrieve and validate the answer criterion.
        criterion_trusted = target.criterion_reliability >= 0.70
        _append_operation(trace, 8, {
            "criterion_available": True,
            "criterion_source": target.criterion_source,
            "criterion_reliability": target.criterion_reliability,
            "criterion_trusted": criterion_trusted,
            "untrusted_criterion_blocks_definitive_error_claims": not criterion_trusted,
        }, {"criterion_validation": "external answer-source reliability gate"})

        # 10. Compare content, relations, order, and conditions independently.
        _append_operation(trace, 9, {
            "content_coverage": initial_assessment["component_coverage"],
            "relation_coverage": initial_assessment["relation_coverage"],
            "order_accuracy": initial_assessment["order_accuracy"],
            "condition_accuracy": initial_assessment["condition_accuracy"],
            "weighted_discrepancy": 1.0 - initial_assessment["weighted_coverage"],
            "criterion_trusted": initial_assessment["criterion_trusted"],
        }, {"comparison_networks": "distributed semantic representations and executive evaluation"})

        # 11. Diagnose likely error categories while preserving competing explanations.
        diagnosis = classify_errors(target, attempt, initial_assessment)
        update_error_history(memory, diagnosis)
        _append_operation(trace, 10, {
            "diagnosis": diagnosis,
            "competing_explanations_retained": True,
            "diagnosis_not_a_neural_error_label": True,
        }, {"conflict_monitoring_signal": monitor["absolute_error_signal"],
            "candidate_conflict": monitor["candidate_conflict"]})

        # 12. Select feedback appropriate to the detected gap and criterion reliability.
        feedback_decision = select_feedback(target, attempt, diagnosis, request.feedback)
        _append_operation(trace, 11, feedback_decision, {
            "feedback_attention_priority": feedback_decision["component_feedback_emphasis"],
            "neuromodulation_not_assumed_from_feedback_alone": True,
        })

        # 13. Correct the representation only from reliable evidence.
        verified_learning_evidence = (
            target.criterion_reliability >= 0.70
            and (initial_assessment["evidenced_accuracy"] >= 0.70
                 or feedback_decision["verified_feedback_accepted"])
        )
        corrected_representation = (
            feedback_decision["correct_representation"]
            if feedback_decision["verified_feedback_accepted"] else None
        )
        biological_learning = biological.learn_from_verified_evidence(
            _target_content(target),
            corrected_representation or attempt.response,
            initial_assessment["evidenced_accuracy"],
            monitor["absolute_error_signal"],
            cue_quality["relevance"],
            min(1.0, attempt.response_time_ms / 60000.0),
            verified_learning_evidence,
        )
        _append_operation(trace, 12, {
            "correction_applied": feedback_decision["verified_feedback_accepted"],
            "corrected_representation": corrected_representation,
            "feedback_explanation": feedback_decision["explanation"],
            "original_response_preserved": True,
            "correction_signal": (1.0 - initial_assessment["weighted_coverage"])
            if verified_learning_evidence else 0.0,
        }, biological_learning)

        # 14. Hide feedback and evaluate the second retrieval attempt, if supplied.
        retest_assessment = (assess_attempt(target, request.retest_attempt)
                             if request.retest_attempt is not None else None)
        immediate_gain = (
            float(retest_assessment["evidenced_accuracy"])
            - float(initial_assessment["evidenced_accuracy"])
            if retest_assessment is not None else None
        )
        retest_diagnosis = (
            classify_errors(target, request.retest_attempt, retest_assessment)
            if request.retest_attempt is not None and retest_assessment is not None else None
        )
        if request.retest_attempt is not None and retest_assessment is not None:
            record_attempt(memory, request.retest_attempt, retest_assessment, "second_retrieval")
            if retest_diagnosis:
                update_error_history(memory, retest_diagnosis)
        _append_operation(trace, 13, {
            "second_attempt_performed": request.retest_attempt is not None,
            "second_attempt": _assessment_summary(retest_assessment)
            if retest_assessment is not None else None,
            "immediate_correction_gain": immediate_gain,
            "persistent_error_diagnosis": retest_diagnosis,
            "immediate_gain_is_not_delayed_mastery": True,
        }, {"second_pattern_reactivation": request.retest_attempt is not None})

        # 15. Test valid alternative cues and estimate access-route diversity.
        cue_attempts = [attempt]
        if request.retest_attempt is not None:
            cue_attempts.append(request.retest_attempt)
        cue_attempts.extend(request.alternative_attempts)
        cue_diversity = update_cue_diversity(target, cue_attempts)
        alternative_assessments = [
            assess_attempt(target, item) for item in request.alternative_attempts
        ]
        transfer_scores = [float(item["evidenced_accuracy"]) for item in alternative_assessments]
        transfer_accuracy = float(np.mean(transfer_scores)) if transfer_scores else None
        for index, (alternative, assessment) in enumerate(
            zip(request.alternative_attempts, alternative_assessments), start=1
        ):
            record_attempt(memory, alternative, assessment, f"alternative_cue_{index}")
            update_error_history(memory, classify_errors(target, alternative, assessment))
        _append_operation(trace, 14, {
            "cue_diversity": cue_diversity,
            "alternative_attempt_count": len(request.alternative_attempts),
            "alternative_attempt_scores": transfer_scores,
            "transfer_accuracy_estimate": transfer_accuracy,
            "valid_target_cues_only": True,
        }, {"overlapping_and_nonidentical_cue_reinstatement": True})

        # 16. Update cue-conditioned accessibility and confidence calibration from evidence.
        accessibility_updates = [update_accessibility(memory, attempt, initial_assessment)]
        if request.retest_attempt is not None and retest_assessment is not None:
            accessibility_updates.append(
                update_accessibility(memory, request.retest_attempt, retest_assessment)
            )
        for alternative, assessment in zip(request.alternative_attempts, alternative_assessments):
            accessibility_updates.append(update_accessibility(memory, alternative, assessment))
        _append_operation(trace, 15, {
            "accessibility_updates": accessibility_updates,
            "cue_routes": dict(memory.cue_routes),
            "confidence_calibration_pairs": len(memory.confidence_pairs),
            "subjective_confidence_not_used_as_accuracy": True,
        }, {"activity_dependent_retrieval_history": len(memory.attempt_history)})

        # 17. Choose an adaptive next learning action using quality, fatigue, and time.
        action = choose_learning_action(
            initial_assessment, feedback_decision, request.fatigue,
            request.available_time_minutes,
        )
        _append_operation(trace, 16, {
            "learning_action": action,
            "retrieval_and_restudy_are_distinct_operations": True,
            "failed_retrieval_does_not_automatically_strengthen_content": True,
        }, {"prefrontal_policy_control": biological.prefrontal.snapshot()})

        # 18. Detect failures of the test itself, not just failures of the learner.
        test_failures = detect_test_failures(target, attempt, initial_assessment)
        test_failures["flags"]["narrow_cue_dependence"] = not cue_diversity["multiple_routes_tested"]
        test_failures["flags"]["excessive_repetition_risk"] = (
            len(cue_attempts) + len(request.delayed_attempts) > 10
        )
        test_failures["flags"]["uncorrected_high_confidence_error"] = (
            attempt.confidence >= 0.80
            and initial_assessment["evidenced_accuracy"] < 0.50
            and not feedback_decision["verified_feedback_accepted"]
        )
        _append_operation(trace, 17, {
            "test_quality": test_failures,
            "validity_warning_count": sum(bool(value)
                                          for value in test_failures["flags"].values()),
            "performance_not_interpreted_as_fixed_ability": True,
        }, {"attention_fatigue_and_arousal": "behavioral confounds; not directly measured here"})

        # 19. Update the conceptual network only when the correction or relation is supported.
        graph_update = update_knowledge_graph(
            target, initial_assessment, feedback_decision, request.feedback, memory,
        )
        _append_operation(trace, 18, graph_update, {
            "cortical_network": biological.cortex.snapshot(),
            "verified_plasticity_only": graph_update["criterion_reliability_gate"],
        })

        # 20. Measure delayed retrieval separately from the immediate retest.
        delayed = delayed_retention_assessment(target, request.delayed_attempts)
        delayed_scores = [
            assess_attempt(target, item) for item in request.delayed_attempts
        ]
        for index, (delayed_attempt, assessment) in enumerate(
            zip(request.delayed_attempts, delayed_scores), start=1
        ):
            record_attempt(memory, delayed_attempt, assessment, f"delayed_retrieval_{index}")
            update_accessibility(memory, delayed_attempt, assessment)
            memory.delayed_accuracy.append(float(assessment["evidenced_accuracy"]))
        memory.delayed_accuracy[:] = memory.delayed_accuracy[-200:]
        _append_operation(trace, 19, {
            "delayed_retention": delayed,
            "delayed_accuracy_history": list(memory.delayed_accuracy),
            "immediate_retest_not_used_as_delayed_evidence": True,
        }, {"offline_consolidation_not_assumed_or_guaranteed": True})

        # 21. Rank future learning priorities from performance, uncertainty, errors, and transfer.
        priority = calculate_future_priority(
            target, initial_assessment, diagnosis, memory, transfer_accuracy,
        )
        _append_operation(trace, 20, {
            "future_priority": priority,
            "repeated_error_counts": dict(memory.error_counts),
            "single_failure_not_a_fixed_ability_measure": True,
        }, {"adaptive_control_state": "model-level future-practice priority estimate"})

        # 22. Prepare the delayed-practice handoff and return an auditable state.
        handoff = schedule_handoff(target, priority, initial_assessment, delayed)
        output = {
            "target_id": target.target_id,
            "initial_accuracy": initial_assessment["evidenced_accuracy"],
            "initial_independent_retrieval": initial_assessment["independent_retrieval"],
            "dominant_error": diagnosis["primary_error"],
            "feedback_action": feedback_decision["action"],
            "feedback_verified": feedback_decision["verified_feedback_accepted"],
            "retest_accuracy": (retest_assessment["evidenced_accuracy"]
                                if retest_assessment is not None else None),
            "immediate_gain": immediate_gain,
            "delayed_accuracy": delayed["mean_delayed_accuracy"],
            "cue_type_coverage": cue_diversity["cue_type_coverage"],
            "future_priority": priority["priority_score"],
            "recommended_action": action["selected_action"],
            "review_handoff": handoff,
            "biological_fidelity": "functional computational surrogate",
        }
        _append_operation(trace, 21, {
            "review_handoff": handoff,
            "auditable_output": output,
            "original_attempt_retained": True,
            "delayed_scheduler_is_external": True,
        }, {"biological_snapshot": biological.snapshot(),
            "neural_equations_are_approximations_not_literal_brain_arithmetic": True})

        active.process_count += 1
        target_state = memory.to_dict()
        return TestingEffectResult(
            target_id=target.target_id,
            operation_trace=tuple(trace),
            initial_assessment=initial_assessment,
            retest_assessment=retest_assessment,
            feedback_decision=feedback_decision,
            future_priority=priority,
            target_state=target_state,
            biological=biological.snapshot(),
            final_state=active.to_dict(),
            output=output,
        )


def register_annotations(
    database_path: str | Path,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Index every Point 04 class and callable in the shared ORE SQLite annotation catalogue."""
    from ore_logger.catalog import sync_repository
    from ore_logger.database import LoggerDatabase

    root = (Path(repository_root).expanduser().resolve() if repository_root
            else Path(__file__).resolve().parents[1])
    return sync_repository(LoggerDatabase(database_path), root,
                           include_paths={"ore_testing_effect"})


def record_result_to_sqlite(
    result: TestingEffectResult,
    database_path: str | Path,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Persist the 22 ordered Point 04 operations and their math/biology outputs to SQLite."""
    from ore_logger.catalog import sync_repository
    from ore_logger.database import LoggerDatabase

    root = (Path(repository_root).expanduser().resolve() if repository_root
            else Path(__file__).resolve().parents[1])
    database = LoggerDatabase(database_path)
    catalogue = sync_repository(database, root, include_paths={"ore_testing_effect"})
    run_id = database.create_run(
        script_path="ore_testing_effect/point04",
        module_name="ore_testing_effect.pipeline",
        entry_kind="testing_effect",
        working_directory=str(root),
    )
    started = time.perf_counter()
    try:
        for sequence, operation in enumerate(result.operation_trace, start=1):
            database.record_event(
                run_id=run_id,
                sequence_no=sequence,
                event_kind="operation",
                source_path="ore_testing_effect/pipeline.py",
                elapsed_ms=round((time.perf_counter() - started) * 1000.0, 3),
                metadata={
                    "operation_id": operation["operation_id"],
                    "operation": operation["operation"],
                    "source_step": operation["source_step"],
                    "status": operation["status"],
                    "evidence_level": operation["evidence_level"],
                    "mathematical_form": operation["mathematical_form"],
                    "functional_output": operation["functional_output"],
                    "biological_output": operation["biological_output"],
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
    return {"run_id": run_id, "annotation_catalogue": catalogue,
            "database_summary": database.summary()}
