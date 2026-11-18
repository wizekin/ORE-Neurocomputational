"""End-to-end 17-stage Point 18 attentional-failure processor and SQLite logging."""
from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Mapping

from .biology import AttentionalBiologicalNetwork
from .contracts import (
    AttentionalConfig, AttentionalExperience, AttentionalResult, AttentionalState,
)
from .functional import (
    adapt_future_attention, assess_competing_inputs, audit_completeness,
    bias_sensory_representation, choose_recovery_action, clip01,
    compute_priority_map, decide_downstream_release, detect_attention_lapse,
    establish_task_requirements, estimate_attentional_state, gate_working_memory,
    preserve_uncertainty, reencode_recovered_information, reconstruct_missing_information,
    reorient_to_task, update_sustained_attention, verify_repair,
)
from .registry import OPERATION_REGISTRY


def _record(
    trace: list[dict[str, Any]],
    index: int,
    output: Mapping[str, Any],
    biological: Mapping[str, Any],
) -> None:
    """Append a source-ordered operation with its equation and both output channels."""
    spec = OPERATION_REGISTRY[index]
    trace.append({
        "step": index + 1,
        "operation_id": spec.operation_id,
        "operation": spec.name,
        "purpose": spec.purpose,
        "evidence_level": spec.evidence_level,
        "mathematical_form": spec.mathematical_form,
        "status": "completed",
        "functional_output": dict(output),
        "biological_output": dict(biological),
    })


class AttentionalFailureProcessor:
    """Coordinate detect-select-encode-audit-recover processing for attentional failures."""

    def __init__(
        self,
        config: AttentionalConfig | None = None,
        state: AttentionalState | None = None,
        biological: AttentionalBiologicalNetwork | None = None,
    ) -> None:
        """Initialize task-calibratable parameters, persistent state, and network surrogates."""
        self.config = config or AttentionalConfig()
        self.state = state or AttentionalState()
        self.biological = biological or AttentionalBiologicalNetwork()
        self.process_count = 0

    def process(
        self,
        event: AttentionalExperience,
        state: AttentionalState | None = None,
    ) -> AttentionalResult:
        """Execute all 17 stages, preserve missing evidence, and gate downstream release."""
        current = state if state is not None else self.state
        config = self.config
        network = self.biological
        trace: list[dict[str, Any]] = []
        self.process_count += 1

        # 1. Establish the current goal and required information.
        requirements = establish_task_requirements(event)
        activity = network.engage(
            "top_down_goal_control", "maintain_task_goal",
            max(requirements["importance_weights"].values(), default=0.0),
        )
        _record(trace, 0, requirements, activity)

        # 2. Estimate alertness, fatigue, distraction, and task-specific availability.
        state_estimate = estimate_attentional_state(event, current, config)
        activity = network.engage(
            "arousal_state_modulation", "estimate_alertness_and_processing_availability",
            state_estimate["attentional_availability"],
        )
        _record(trace, 1, state_estimate, activity)

        # 3. Detect competing external/internal inputs while retaining safety-relevant signals.
        competitors = assess_competing_inputs(event)
        activity = network.engage(
            "ventral_attention_reorienting", "detect_competing_inputs",
            competitors["distraction_load"],
        )
        _record(trace, 2, competitors, activity)

        # 4. Compute an adaptive priority map from task and stimulus features.
        priority = compute_priority_map(event, requirements, current, config)
        activity = network.engage(
            "dorsal_attention_selection", "rank_goal_relevant_input",
            max((x["selection_probability"] for x in priority["ranked_items"]), default=0.0),
        )
        _record(trace, 3, priority, activity)

        # 5. Bias sensory processing while preserving the original input and low-priority traces.
        sensory = bias_sensory_representation(priority)
        activity = network.engage(
            "sensory_gain_modulation", "bias_relevant_sensory_features",
            max((x["gain"] / 1.5 for x in sensory["items"]), default=0.0),
        )
        _record(trace, 4, sensory, activity)

        # 6. Gate candidate information into working memory under a dynamic budget.
        gate = gate_working_memory(priority, state_estimate, config)
        activity = network.engage(
            "working_memory_admission", "admit_or_defer_candidate_information",
            min(1.0, len(gate["selected_items"]) / max(1, len(priority["ranked_items"]))),
        )
        _record(trace, 5, gate, activity)

        # 7. Track sustained attention across time and estimate stability.
        sustained = update_sustained_attention(state_estimate, current, event, config)
        activity = network.engage(
            "sustained_attention_monitor", "update_attention_over_time",
            sustained["stability"],
        )
        _record(trace, 6, sustained, activity)

        # 8. Detect possible lapses using convergent evidence and hysteresis.
        lapse = detect_attention_lapse(event, state_estimate, sustained, current, config)
        activity = network.engage(
            "lapse_detection_monitor", "detect_and_localize_possible_lapse",
            lapse["confidence"] if lapse["detected"] else lapse["score"],
        )
        _record(trace, 7, lapse, activity)

        # 9. Reconstruct missing information only from source-supported records.
        missing = reconstruct_missing_information(event, gate, lapse, config)
        activity = network.engage(
            "evidence_reconstruction", "identify_missing_and_recoverable_information",
            min(1.0, len(missing["missing"]) / max(1, len(event.required_information))),
        )
        _record(trace, 8, missing, activity)

        # Build the first-pass representation from selected, actually processed input.
        initial_items: dict[str, dict[str, Any]] = {}
        selected_keys = {str(x["key"]) for x in gate["selected_items"]}
        for item in gate["selected_items"]:
            key = str(item["key"])
            confidence = clip01(
                float(item.get("signal_quality", 1.0))
                * float(item.get("integration_support", 1.0))
                * state_estimate["attentional_availability"]
            )
            initial_items[key] = {
                "key": key, "content": item.get("content"),
                "confidence": confidence,
                "source_id": item.get("source_id", "incoming"),
                "provenance": "observed_input",
            }
        representation = {
            "items": initial_items,
            "selected_keys": sorted(selected_keys),
            "deferred_keys": [x["key"] for x in gate["deferred_items"]],
        }

        # 10. Audit the first-pass representation against required information and dependencies.
        audit = audit_completeness(event.required_information, representation, config)
        activity = network.engage(
            "completeness_audit", "compare_encoded_representation_with_requirements",
            audit["coverage"],
        )
        _record(trace, 9, audit, activity)

        # 11. Select a feasible recovery action by expected benefit minus cost.
        recovery = choose_recovery_action(event, state_estimate, audit)
        activity = network.engage(
            "recovery_control", "select_recovery_strategy",
            max(0.0, min(1.0, recovery.get("net_value", 0.0) + 0.5)),
        )
        _record(trace, 10, recovery, activity)

        # 12. Re-establish the current task goal and last verified processing position.
        reoriented = reorient_to_task(event, current, priority, audit)
        activity = network.engage(
            "goal_reorientation", "reinstate_task_goal_and_continuation_point",
            reoriented["goal_alignment"],
        )
        _record(trace, 11, reoriented, activity)

        # 13. Re-encode recoverable information with explicit source and confidence labels.
        reencoded = reencode_recovered_information(event, representation, missing, recovery)
        activity = network.engage(
            "hippocampal_reencoding", "integrate_verified_recovered_input",
            min(1.0, len(reencoded["recovered_items"]) / max(1, len(event.required_information))),
        )
        _record(trace, 12, reencoded, activity)

        # 14. Recheck the repaired representation and identify residual high-priority gaps.
        verification = verify_repair(event.required_information, reencoded, config)
        activity = network.engage(
            "repair_verification", "verify_coverage_and_residual_gaps",
            verification["coverage"],
        )
        _record(trace, 13, verification, activity)

        # 15. Adapt future attention allocation from context-specific feedback.
        future_policy = adapt_future_attention(event, current, lapse, verification)
        activity = network.engage(
            "selection_history_learning", "adapt_context_specific_attention_policy",
            future_policy["recovery_success_rate"],
        )
        _record(trace, 14, future_policy, activity)

        # 16. Preserve unrecovered input as unknown instead of filling it from expectation.
        uncertainty = preserve_uncertainty(missing, verification)
        activity = network.engage(
            "uncertainty_maintenance", "maintain_unknowns_and_missing_evidence",
            min(1.0, uncertainty["unresolved_count"] / max(1, len(event.required_information))),
        )
        _record(trace, 15, uncertainty, activity)

        # 17. Release only a sufficiently complete, confidence-calibrated representation.
        release = decide_downstream_release(verification, uncertainty, event, config)
        activity = network.engage(
            "downstream_release_gate", "gate_downstream_comprehension",
            1.0 if release["release_downstream"] else 0.0,
        )
        _record(trace, 16, release, activity)

        # Commit the revisable state only after all operations have returned successfully.
        current.alertness = state_estimate["alertness"]
        current.fatigue = state_estimate["fatigue"]
        current.distraction = state_estimate["distraction"]
        current.stability = sustained["stability"]
        current.lapse_probability = state_estimate["lapse_probability"]
        current.attention_history = list(sustained["history"])
        current.lapse_history.append({
            "score": lapse["score"], "detected": lapse["detected"],
            "experience_id": event.experience_id,
        })
        current.lapse_history = current.lapse_history[-max(8, config.lapse_hysteresis_samples):]
        verified_positions = [
            index for index, item in enumerate(event.incoming_items)
            if str(item.get("key", "")) in reencoded["items"]
            and float(reencoded["items"][str(item.get("key", ""))].get("confidence", 0.0))
            >= config.requirement_threshold
        ]
        if verified_positions:
            current.last_verified_position = max(current.last_verified_position, max(verified_positions))
        current.experience_count += 1

        return AttentionalResult(
            experience_id=event.experience_id,
            task_requirements=requirements,
            state_estimate=state_estimate,
            competing_input_assessment=competitors,
            priority_map=priority,
            sensory_bias=sensory,
            working_memory_gate=gate,
            sustained_attention=sustained,
            lapse_event=lapse,
            missing_information=missing,
            completeness_audit=audit,
            recovery_plan=recovery,
            reoriented_task=reoriented,
            reencoded_representation=reencoded,
            repair_verification=verification,
            future_attention_policy=future_policy,
            preserved_uncertainty=uncertainty,
            release_decision=release,
            operation_trace=tuple(trace),
            biological=network.snapshot(),
            final_state=current.to_dict(),
        )


def register_annotations(
    database_path: str | Path,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Index Point 18 classes and callables in the ORE SQLite annotation catalogue."""
    from ore_logger.catalog import sync_repository
    from ore_logger.database import LoggerDatabase

    root = (
        Path(repository_root).expanduser().resolve()
        if repository_root else Path(__file__).resolve().parents[1]
    )
    return sync_repository(
        LoggerDatabase(database_path), root,
        include_paths={"ore_attentional_failure"},
    )


def record_result_to_sqlite(
    result: AttentionalResult,
    database_path: str | Path,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Persist every Point 18 operation output and synchronize package annotations."""
    from ore_logger.catalog import sync_repository
    from ore_logger.database import LoggerDatabase

    root = (
        Path(repository_root).expanduser().resolve()
        if repository_root else Path(__file__).resolve().parents[1]
    )
    database = LoggerDatabase(database_path)
    catalogue = sync_repository(
        database, root, include_paths={"ore_attentional_failure"}
    )
    run_id = database.create_run(
        script_path="ore_attentional_failure/point18",
        module_name="ore_attentional_failure.pipeline",
        entry_kind="attentional_failure", working_directory=str(root),
    )
    started = time.perf_counter()
    try:
        for sequence, item in enumerate(result.operation_trace, start=1):
            database.record_event(
                run_id=run_id,
                sequence_no=sequence,
                event_kind="operation",
                source_path="ore_attentional_failure/pipeline.py",
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
