"""End-to-end 12-stage Point 03 spaced-repetition processor and SQLite recorder."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import time
from typing import Any, Mapping

from .biology import BiologicalSpacedRepetitionNetwork
from .contracts import (
    LearningItem, MemoryRecord, SpacedRepetitionConfig, SpacedRepetitionRequest,
    SpacedRepetitionResult, SpacedRepetitionState,
)
from .functional import (
    add_hours, apply_offline_consolidation, assess_calibration, assess_maintenance,
    choose_initial_interval, choose_next_interval, decide_review_action,
    elapsed_hours, estimate_accessibility, estimate_encoding_quality, parse_utc,
    score_review, update_memory_state,
)
from .registry import OPERATION_REGISTRY


def _append_operation(
    trace: list[dict[str, Any]], index: int, functional_output: Mapping[str, Any],
    biological_output: Mapping[str, Any] | None = None,
) -> None:
    """Append one ordered operation with its mathematical and biological annotations."""
    spec = OPERATION_REGISTRY[index]
    trace.append({
        "step": index + 1, "source_step": spec.source_step,
        "operation_id": spec.operation_id, "operation": spec.name,
        "purpose": spec.purpose, "status": "completed",
        "evidence_level": spec.evidence_level,
        "mathematical_form": spec.mathematical_form,
        "functional_output": dict(functional_output),
        "biological_output": dict(biological_output or {}),
    })


def _validate_repeated_item(existing: LearningItem, supplied: LearningItem) -> None:
    """Reject attempts to reuse an item ID for materially different learning content."""
    fields = ("content", "components", "relations", "context", "difficulty",
              "target_retention_hours", "encoding_measurements", "source")
    changed = [name for name in fields if getattr(existing, name) != getattr(supplied, name)]
    if changed:
        raise ValueError(f"item_id already exists with different item fields: {changed}")


class SpacedRepetitionProcessor:
    """Coordinate item registration, accessibility prediction, review outcomes, and adaptive scheduling."""

    def __init__(
        self, learner_id: str, config: SpacedRepetitionConfig | None = None,
        state: SpacedRepetitionState | None = None,
        biological: BiologicalSpacedRepetitionNetwork | None = None,
    ) -> None:
        """Initialize learner-specific scheduler state and interacting biological surrogates."""
        if not isinstance(learner_id, str) or not learner_id.strip():
            raise ValueError("learner_id must be a non-empty string")
        self.learner_id = learner_id.strip()
        self.config = config or SpacedRepetitionConfig()
        self.state = state or SpacedRepetitionState(self.learner_id)
        if self.state.learner_id != self.learner_id:
            raise ValueError("processor learner_id must match state learner_id")
        self.biological = biological or BiologicalSpacedRepetitionNetwork(
            dimensions=64, learning_rate=self.config.learning_rate
        )

    def process(
        self, request: SpacedRepetitionRequest,
        state: SpacedRepetitionState | None = None,
    ) -> SpacedRepetitionResult:
        """Process one item registration or review update through all twelve source stages."""
        if request.learner_id != self.learner_id:
            raise ValueError("request learner_id must match processor learner_id")
        active = state if state is not None else self.state
        if active.learner_id != self.learner_id:
            raise ValueError("state learner_id must match processor learner_id")
        config = self.config
        bio = self.biological
        item_input = request.item
        now = request.current_time_utc
        trace: list[dict[str, Any]] = []
        record = active.items.get(item_input.item_id)
        is_new = record is None
        if is_new:
            created_at = item_input.created_at_utc or now
            item = replace(item_input, created_at_utc=created_at)
            bio_encoding = bio.encode(
                [item.content, *item.components, *item.relations],
                {"item_id": item.item_id, "source": item.source, "context": dict(item.context)},
            )
        else:
            assert record is not None
            _validate_repeated_item(record.item, item_input)
            item = record.item
            bio_encoding = {"item_already_bound": True,
                            "hippocampal_state": bio.hippocampus.snapshot()}

        # 1. Register a learning episode and retain a stable memory-item identity.
        _append_operation(trace, 0, {
            "item_id": item.item_id, "new_memory_item": is_new,
            "content": item.content, "components": list(item.components),
            "relations": list(item.relations), "source": item.source,
            "context": dict(item.context),
            "segmentation_note": "one caller-defined learning item; components remain linked",
        }, bio_encoding)

        # 2. Estimate the quality of initial encoding and initialize durability only once.
        if is_new:
            encoding = estimate_encoding_quality(item.encoding_measurements)
            stability = bio.initial_stability_hours if hasattr(bio, "initial_stability_hours") else (
                config.base_stability_hours
                * (0.50 + 1.50 * encoding["encoding_quality"])
                * (1.0 - 0.55 * item.difficulty)
            )
            stability = min(config.maximum_stability_hours,
                            max(config.minimum_stability_hours, stability))
            record = MemoryRecord(
                item=item, created_at_utc=str(item.created_at_utc),
                last_learning_at_utc=str(item.created_at_utc),
                encoding_quality=encoding["encoding_quality"],
                encoding_uncertainty=encoding["encoding_uncertainty"],
                stability_hours=stability, memory_accessibility=1.0,
                memory_uncertainty=encoding["encoding_uncertainty"],
                next_interval_hours=config.minimum_interval_hours,
                next_due_at_utc=str(item.created_at_utc),
            )
            active.items[item.item_id] = record
        else:
            assert record is not None
            encoding = {
                "encoding_quality": record.encoding_quality,
                "encoding_uncertainty": record.encoding_uncertainty,
                "measurement_coverage": 1.0,
                "measurements_used": sorted(item.encoding_measurements),
                "retained_from_prior_state": True,
            }
        _append_operation(trace, 1, {
            "encoding_estimate": encoding,
            "initial_stability_hours": record.stability_hours,
            "stability_is_not_a_single_synaptic_measure": True,
        }, {"distributed_population": bio.population.snapshot(),
            "hippocampus": bio.hippocampus.snapshot()})

        # 3. Establish the temporal origin and first non-zero review delay.
        if is_new:
            initial_schedule = choose_initial_interval(item, record.encoding_quality, config)
            record.next_interval_hours = initial_schedule["interval_hours"]
            record.next_due_at_utc = add_hours(record.last_learning_at_utc,
                                               record.next_interval_hours)
        else:
            initial_schedule = {
                "interval_hours": record.next_interval_hours,
                "next_due_at_utc": record.next_due_at_utc,
                "retained_existing_schedule": True,
            }
        _append_operation(trace, 2, {
            "last_learning_at_utc": record.last_learning_at_utc,
            "initial_schedule": initial_schedule,
            "next_due_at_utc": record.next_due_at_utc,
            "calendar_is_external_scheduler_state": True,
        }, {"temporal_separation_hours": record.next_interval_hours})

        # 4. Estimate memory accessibility and uncertainty at the current timestamp.
        elapsed = elapsed_hours(record.last_learning_at_utc, now)
        prior_scores = [float(row.get("quality_score", 0.0)) for row in record.review_history]
        accessibility = estimate_accessibility(
            record.stability_hours, elapsed, len(record.review_history),
            record.memory_uncertainty, prior_scores, config,
        )
        record.memory_accessibility = float(accessibility["memory_accessibility"])
        record.memory_uncertainty = float(accessibility["memory_uncertainty"])
        _append_operation(trace, 3, {"accessibility_estimate": accessibility,
                                     "subjective_familiarity_used_as_recall_proxy": False},
                          {"hippocampus": bio.hippocampus.snapshot(),
                           "distributed_population": bio.population.snapshot()})

        # 5. Determine whether the item is ready for review, should be deferred, or needs support.
        deadline = add_hours(record.created_at_utc, item.target_retention_hours)
        decision = decide_review_action(
            float(accessibility["predicted_recall_probability"]), now,
            deadline, record.next_due_at_utc, config,
        )
        _append_operation(trace, 4, {"review_decision": decision,
                                     "target_recall_band": [config.target_recall_low,
                                                            config.target_recall_high]},
                          {"prefrontal_control": bio.prefrontal.snapshot()})

        # 6. Capture the actual retrieval attempt only when an observation is supplied.
        review_event: dict[str, Any] | None = None
        previous_activity = bio.population.activity.copy()
        review_score: dict[str, Any] | None = None
        if request.review_observation is not None:
            observation = request.review_observation
            if parse_utc(observation.occurred_at_utc) > parse_utc(now):
                raise ValueError("review observation cannot occur after current_time_utc")
            event_elapsed = elapsed_hours(record.last_learning_at_utc,
                                          observation.occurred_at_utc)
            event_prediction = estimate_accessibility(
                record.stability_hours, event_elapsed, len(record.review_history),
                record.memory_uncertainty, prior_scores, config,
            )
            review_score = score_review(observation)
            review_event = {
                "occurred_at_utc": observation.occurred_at_utc,
                "elapsed_since_last_learning_hours": event_elapsed,
                "predicted_recall_probability": event_prediction["predicted_recall_probability"],
                **review_score,
                "learner_response": observation.learner_response,
                "corrected_response": observation.corrected_response,
                "early_review": parse_utc(observation.occurred_at_utc) < parse_utc(record.next_due_at_utc),
                "first_response_preserved": True,
            }
            bio_review = bio.review(
                item.item_id, item.content,
                1.0 - float(event_prediction["predicted_recall_probability"]),
                float(review_score["quality_score"]), float(review_score["effort"]),
                event_elapsed, previous_activity,
            )
        else:
            bio_review = {"review_attempt_performed": False,
                          "reason": "no learner retrieval observation supplied"}
        _append_operation(trace, 5, {
            "review_attempt": review_event,
            "attempt_performed": review_event is not None,
            "recall_recognition_distinction": True,
            "biological_reactivation_is_a_surrogate": True,
        }, bio_review)

        # 7. Correct incomplete/wrong attempts and keep corrected relearning distinct.
        if review_event is None:
            correction = {"correction_required": False, "status": "awaiting_review_observation"}
        elif review_event["outcome"] in {"partial", "failure", "corrected_relearning"}:
            correction = {
                "correction_required": review_event["outcome"] != "corrected_relearning",
                "feedback_provided": review_event["feedback_provided"],
                "correction_status": (
                    "corrected_relearning_recorded" if review_event["feedback_provided"]
                    else "correction_needed_before_retesting"
                ),
                "missing_or_inaccurate_response_preserved": True,
                "corrected_response_available": bool(review_event["corrected_response"]),
            }
        else:
            correction = {"correction_required": False,
                          "status": "no_correction_flagged_for_this_outcome"}
        _append_operation(trace, 6, {"correction": correction,
                                     "retrieval_error_feedback_not_equated_with_success": True},
                          {"error_monitoring": "prefrontal/retrieval-control model-level analogy",
                           "relearning_event": bool(review_event and review_event["feedback_provided"])})

        # 8. Update durability and event history using the observed outcome, not recognition alone.
        if review_event is not None and review_score is not None:
            update = update_memory_state(record.stability_hours,
                                         record.memory_uncertainty, review_score, config)
            record.stability_hours = float(update["stability_hours"])
            record.memory_uncertainty = float(update["memory_uncertainty"])
            record.review_history.append(dict(review_event))
            meaningful_learning = (
                bool(review_event["independent_retrieval"])
                and review_event["outcome"] in {"success", "partial"}
            ) or bool(review_event["feedback_provided"])
            if meaningful_learning:
                record.last_learning_at_utc = str(observation.occurred_at_utc)
        else:
            update = {"update_type": "no_review_observed",
                      "stability_hours": record.stability_hours,
                      "memory_uncertainty": record.memory_uncertainty,
                      "state_changed": False}
        _append_operation(trace, 7, {"memory_state_update": update,
                                     "review_history_count": len(record.review_history)},
                          {"cortical_plasticity": bio.cortex.snapshot(),
                           "learning_gate": bio.learning_gate.snapshot()})

        # 9. Choose the next interval from the updated model and actual review outcome.
        if review_event is not None:
            schedule = choose_next_interval(
                record.stability_hours, item, observation,
                observation.occurred_at_utc, config,
            )
            record.next_interval_hours = float(schedule["interval_hours"])
            record.next_due_at_utc = add_hours(observation.occurred_at_utc,
                                               record.next_interval_hours)
        else:
            schedule = {
                "interval_hours": record.next_interval_hours,
                "next_due_at_utc": record.next_due_at_utc,
                "schedule_reason": "retain existing due time until a review outcome is observed",
            }
        _append_operation(trace, 8, {"adaptive_schedule": schedule,
                                     "next_due_at_utc": record.next_due_at_utc},
                          {"interval_is_external_scheduler_decision": True})

        # 10. Assess provisional maintenance from repeated independent successes over time.
        maintenance = assess_maintenance(
            record.review_history, record.created_at_utc, now, config
        )
        record.provisional_maintenance = bool(maintenance["provisionally_stable"])
        if record.provisional_maintenance:
            maintenance_interval = min(
                config.maximum_interval_hours,
                max(config.minimum_interval_hours, item.target_retention_hours * 0.20),
            )
            if review_event is not None:
                record.next_interval_hours = max(record.next_interval_hours,
                                                 maintenance_interval)
                record.next_due_at_utc = add_hours(now, record.next_interval_hours)
                schedule["maintenance_interval_applied"] = True
        _append_operation(trace, 9, {"maintenance_assessment": maintenance,
                                     "review_history_count": len(record.review_history),
                                     "maintenance_due_policy": "provisional_maintenance_not_permanent_mastery"},
                          {"continued_encoding_and_reactivation_possible": True})

        # 11. Apply only an explicitly supplied offline/sleep observation.
        offline_payload = {
            "observed": request.offline_consolidation.observed,
            "duration_hours": request.offline_consolidation.duration_hours,
            "quality": request.offline_consolidation.quality,
            "interference": request.offline_consolidation.interference,
            "evidence_confidence": request.offline_consolidation.evidence_confidence,
        }
        consolidation = apply_offline_consolidation(
            record.stability_hours, record.memory_uncertainty, offline_payload, config
        )
        record.stability_hours = float(consolidation["stability_hours"])
        record.memory_uncertainty = float(consolidation["memory_uncertainty"])
        if request.offline_consolidation.observed:
            record.consolidation_history.append({
                "at_utc": now, **consolidation,
            })
        bio_offline = bio.offline.observe(offline_payload)
        _append_operation(trace, 10, {"offline_consolidation": consolidation,
                                      "consolidation_history_count": len(record.consolidation_history)},
                          {"offline_network": bio_offline,
                           "sleep_is_not_a_guaranteed_gain": True})

        # 12. Calibrate only against independent binary retrieval outcomes.
        if review_event is not None and review_event["independent_retrieval"]:
            if review_event["outcome"] in {"success", "failure"}:
                observed = 1.0 if review_event["outcome"] == "success" else 0.0
                record.predicted_outcomes.append({
                    "predicted": float(review_event["predicted_recall_probability"]),
                    "observed": observed,
                })
        calibration = assess_calibration(record.predicted_outcomes, config)
        _append_operation(trace, 11, {
            "calibration": calibration,
            "recognition_only_excluded": True,
            "independent_retrieval_history": sum(
                1 for event in record.review_history if event.get("independent_retrieval")
            ),
            "uncertainty_and_predictions_remain_model_estimates": True,
        }, {"learning_gate": bio.learning_gate.snapshot(),
            "biological_fidelity": "computational surrogate"})

        active.process_count += 1
        output = {
            "learner_id": self.learner_id, "item_id": item.item_id,
            "new_item_registered": is_new,
            "review_action": decision["action"],
            "predicted_recall_probability": float(accessibility["predicted_recall_probability"]),
            "next_interval_hours": record.next_interval_hours,
            "next_due_at_utc": record.next_due_at_utc,
            "review_attempt_recorded": review_event is not None,
            "provisionally_stable": record.provisional_maintenance,
            "review_history_count": len(record.review_history),
            "biological_fidelity": "functional computational surrogate",
        }
        return SpacedRepetitionResult(
            learner_id=self.learner_id, item_id=item.item_id,
            operation_trace=tuple(trace), memory_record=record.to_dict(),
            review_event=review_event, calibration=calibration,
            biological=bio.snapshot(), final_state=active.to_dict(), output=output,
        )


def register_annotations(
    database_path: str | Path, repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Index every Point 03 class and callable in the shared SQLite annotation catalogue."""
    from ore_logger.catalog import sync_repository
    from ore_logger.database import LoggerDatabase
    root = Path(repository_root).expanduser().resolve() if repository_root else Path(__file__).resolve().parents[1]
    return sync_repository(LoggerDatabase(database_path), root, include_paths={"ore_spaced_repetition"})


def record_result_to_sqlite(
    result: SpacedRepetitionResult, database_path: str | Path,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Persist all twelve numbered operations and AST annotations in the shared ORE SQLite database."""
    from ore_logger.catalog import sync_repository
    from ore_logger.database import LoggerDatabase
    root = Path(repository_root).expanduser().resolve() if repository_root else Path(__file__).resolve().parents[1]
    database = LoggerDatabase(database_path)
    catalogue = sync_repository(database, root, include_paths={"ore_spaced_repetition"})
    run_id = database.create_run(
        script_path="ore_spaced_repetition/point03",
        module_name="ore_spaced_repetition.pipeline",
        entry_kind="spaced_repetition", working_directory=str(root),
    )
    started = time.perf_counter()
    try:
        for sequence, item in enumerate(result.operation_trace, start=1):
            database.record_event(
                run_id=run_id, sequence_no=sequence, event_kind="operation",
                source_path="ore_spaced_repetition/pipeline.py",
                elapsed_ms=round((time.perf_counter() - started) * 1000.0, 3),
                metadata={
                    "operation_id": item["operation_id"], "operation": item["operation"],
                    "source_step": item["source_step"], "status": item["status"],
                    "evidence_level": item["evidence_level"],
                    "mathematical_form": item["mathematical_form"],
                    "functional_output": item["functional_output"],
                    "biological_output": item["biological_output"],
                },
            )
        database.finish_run(run_id, "completed",
                            duration_ms=round((time.perf_counter() - started) * 1000.0, 3))
    except Exception as exc:
        database.finish_run(run_id, "failed",
                            duration_ms=round((time.perf_counter() - started) * 1000.0, 3),
                            error_summary=type(exc).__name__ + ": " + str(exc))
        raise
    return {"run_id": run_id, "annotation_catalogue": catalogue,
            "database_summary": database.summary()}
