"""Nineteen-step Point 19 working-memory overload and recovery pipeline."""
from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Mapping

from .biology import DistributedWorkingMemoryNetwork
from .contracts import Point19Config, Point19Request, Point19Result, Point19State
from .functional import (
    adapt_difficulty, allocate_attention, apply_recovery, check_coherence,
    choose_recovery_action, clip01, commit_interpretation, create_dependencies,
    detect_overload, diagnose_failures, encode_unit, estimate_interference,
    estimate_load, form_chunks, identify_at_risk, integrate_retrieval,
    reconstruct_supported, retrieve_by_cue, segment_units, softmax, update_accessibility,
)
from .registry import OPERATION_REGISTRY


_POPULATION_BY_OPERATION = (
    "prefrontal_goal_control",
    "phonological_lexical_maintenance",
    "phonological_lexical_maintenance",
    "syntactic_dependency_workspace",
    "phonological_lexical_maintenance",
    "prefrontal_goal_control",
    "semantic_event_integration",
    "parietal_attention_selection",
    "cingulo_opercular_conflict_monitor",
    "medial_temporal_pattern_completion",
    "semantic_event_integration",
    "cingulo_opercular_conflict_monitor",
    "parietal_attention_selection",
    "prefrontal_goal_control",
    "cingulo_opercular_conflict_monitor",
    "medial_temporal_pattern_completion",
    "semantic_event_integration",
    "syntactic_dependency_workspace",
    "prefrontal_goal_control",
)


def _record(
    trace: list[dict[str, Any]],
    index: int,
    functional_output: Mapping[str, Any],
    biological_output: Mapping[str, Any],
) -> None:
    """Append one source-ordered operation with its equation and separate output channels."""
    spec = OPERATION_REGISTRY[index]
    trace.append({
        "step": index + 1,
        "operation_id": spec.operation_id,
        "operation": spec.name,
        "purpose": spec.purpose,
        "evidence_level": spec.evidence_level,
        "mathematical_form": spec.mathematical_form,
        "biological_reference": spec.biological_reference,
        "status": "completed",
        "functional_output": dict(functional_output),
        "biological_output": dict(biological_output),
    })


class Point19WorkingMemoryOverloadProcessor:
    """Coordinate bounded maintenance, cue retrieval, overload detection, and recovery for Point 19.

    The functional model maintains an inspectable representation/dependency state. The optional
    rate-coded network is a separate biological-reference channel, not measured or fitted neural
    activity. A processor instance may be reused to model successive input increments.
    """

    def __init__(
        self,
        config: Point19Config | None = None,
        state: Point19State | None = None,
        biological: DistributedWorkingMemoryNetwork | None = None,
    ) -> None:
        """Initialize task-calibratable parameters, a revisable state, and optional network proxy."""
        self.config = config or Point19Config()
        self.state = state or Point19State()
        self.biological = biological or DistributedWorkingMemoryNetwork(self.config)
        self.process_count = 0

    def process(
        self,
        request: Point19Request,
        state: Point19State | None = None,
    ) -> Point19Result:
        """Execute all 19 source operations and return trace, interpretation, diagnostics, and biology."""
        if not isinstance(request, Point19Request):
            raise TypeError("request must be a Point19Request")
        current = state if state is not None else self.state
        config = self.config
        network = self.biological if config.simulate_biology else None
        self.process_count += 1
        trace: list[dict[str, Any]] = []
        units = list(request.units) if request.units is not None else segment_units(request.text or "")
        if request.unit_annotations and len(request.unit_annotations) != len(units):
            raise ValueError("unit_annotations must have one item per segmented unit")
        annotations = list(request.unit_annotations) or [{} for _ in units]

        def biological(index: int, drive: float) -> dict[str, Any]:
            """Advance the target network population or report that biological simulation is disabled."""
            if network is None:
                return {
                    "model": "disabled", "operation": OPERATION_REGISTRY[index].operation_id,
                    "scope": "Biological-reference simulation disabled by configuration.",
                }
            return network.engage(
                OPERATION_REGISTRY[index].operation_id,
                _POPULATION_BY_OPERATION[index], clip01(drive),
            )

        # 1. Establish the goal and normalized priority weights.
        goal_values = {str(key): clip01(max(0.0, float(value))) for key, value in request.goals.items()}
        context_relevance = request.context.get("goal_context_relevance", {})
        task_importance = request.context.get("goal_task_importance", {})
        if not isinstance(context_relevance, Mapping) or not isinstance(task_importance, Mapping):
            raise TypeError("goal_context_relevance and goal_task_importance must be mappings")
        feature_keys = sorted(
            set(goal_values) | {str(key) for key in context_relevance}
            | {str(key) for key in task_importance}
        )
        goal_features = {
            key: {
                "goal_relevance": goal_values.get(key, 0.0),
                "context_relevance": clip01(float(context_relevance.get(key, 0.0))),
                "task_importance": clip01(float(task_importance.get(key, 0.0))),
            }
            for key in feature_keys
        }
        goal_scores = [
            0.50 * values["goal_relevance"]
            + 0.25 * values["context_relevance"]
            + 0.25 * values["task_importance"]
            for values in goal_features.values()
        ]
        goal_probabilities = softmax(goal_scores, config.attention_temperature)
        current.goal_weights = dict(zip(feature_keys, goal_probabilities))
        goal_output = {
            "sequence_id": request.sequence_id,
            "goal_features": goal_features,
            "goal_scores": dict(zip(feature_keys, goal_scores)),
            "goal_weights": dict(current.goal_weights),
            "goal_is_explicit": bool(feature_keys),
            "task_priority_policy": "weighted goal/context/task relevance followed by temperature-scaled softmax",
        }
        _record(trace, 0, goal_output, biological(0, max(current.goal_weights.values(), default=0.0)))

        # 2. Segment the incoming stream, retaining unit boundaries and punctuation.
        segmentation = {
            "units": units,
            "unit_count": len(units),
            "segmentation_method": "caller_supplied_units" if request.units is not None else "regex_word_punctuation",
            "full_syntactic_parse_performed": False,
        }
        _record(trace, 1, segmentation, biological(1, min(1.0, len(units) / 20.0)))

        # 3. Encode only observed/supplied features with source and position information.
        encoded: list[dict[str, Any]] = []
        for offset, (unit, annotation) in enumerate(zip(units, annotations)):
            position = current.step_index + offset
            representation = encode_unit(
                request.sequence_id, unit, position, annotation,
                goal_relevance=float(annotation.get("goal_relevance", 0.5)),
            )
            if representation.identifier in current.representations:
                raise ValueError(f"duplicate representation identifier: {representation.identifier}")
            current.representations[representation.identifier] = representation
            encoded.append(representation.to_dict())
        _record(trace, 2, {"representations": encoded, "encoded_count": len(encoded)},
                biological(2, min(1.0, len(encoded) / 16.0)))

        # 4. Build open dependencies only from explicit request or unit annotations.
        dependencies = create_dependencies(request.dependencies, annotations, request.sequence_id)
        duplicate_ids = set(current.dependencies).intersection(item.identifier for item in dependencies)
        if duplicate_ids:
            raise ValueError(f"duplicate dependency identifiers: {sorted(duplicate_ids)}")
        for dependency in dependencies:
            current.dependencies[dependency.identifier] = dependency
        structure_output = {
            "new_dependencies": [item.to_dict() for item in dependencies],
            "open_dependencies": sum(item.open for item in current.dependencies.values()),
            "structure_source": "explicit_dependency_specs_or_unit_annotations",
            "alternatives_fabricated": False,
        }
        _record(trace, 3, structure_output,
                biological(3, min(1.0, len(dependencies) / 5.0)))

        # 5. Maintain prior items using context support, refreshing, distance, and interference.
        contextual_support = request.context.get("item_support", {})
        if not isinstance(contextual_support, Mapping):
            raise TypeError("context.item_support must be a mapping when supplied")
        raw_refresh_ids = request.context.get("refresh_ids", ())
        if isinstance(raw_refresh_ids, (str, bytes)):
            raw_refresh_ids = (raw_refresh_ids,)
        refresh_ids = {str(value) for value in raw_refresh_ids}
        maintenance = update_accessibility(
            current, current.step_index + max(0, len(units) - 1), config,
            contextual_support={str(k): float(v) for k, v in contextual_support.items()},
            refreshed_ids=refresh_ids,
        )
        _record(trace, 4, maintenance,
                biological(4, sum(x["after"] for x in maintenance["items"].values())
                           / max(1, len(maintenance["items"]))))

        # 6. Estimate current load from unresolved structure, complexity, distance, competition, and context.
        structural_complexity = clip01(float(request.metrics.get(
            "structural_complexity",
            min(1.0, (len(dependencies) + sum(bool(x.get("ambiguous")) for x in annotations)) / 8.0),
        )))
        active = [item for item in current.representations.values() if item.active]
        uncertainty_level = sum(1.0 - item.confidence for item in active) / max(1, len(active))
        context_support = clip01(float(request.context.get("context_support", 0.0)))
        load = estimate_load(
            current, structural_complexity, request.extra_task_load,
            uncertainty_level, context_support, config,
            current_position=current.step_index + max(0, len(units) - 1),
        )
        _record(trace, 5, load, biological(5, load["load"]))

        # 7. Form reversible chunks only from explicit chunk annotations.
        chunks = form_chunks(current)
        _record(trace, 6, chunks, biological(6, min(1.0, chunks["chunk_count"] / 5.0)))

        # 8. Allocate attention between currently active representations.
        attention = allocate_attention(current, config.attention_temperature)
        _record(trace, 7, attention,
                biological(7, max((x["attention"] for x in attention["ranked_items"]), default=0.0)))

        # 9. Estimate similarity-based competition before retrieval.
        interference = estimate_interference(current)
        _record(trace, 8, interference, biological(8, interference["mean_interference"]))

        # 10. Retrieve candidate antecedents/relations using dependency-specific cues.
        candidates = [item for item in current.representations.values() if item.active]
        retrievals: list[dict[str, Any]] = []
        for dependency in current.dependencies.values():
            if dependency.open:
                retrievals.append(retrieve_by_cue(dependency, candidates, config))
        retrieval_output = {
            "dependencies": retrievals,
            "retrieval_count": len(retrievals),
            "selection_policy": "cue match plus accessibility and task relevance, normalized by softmax",
        }
        _record(trace, 9, retrieval_output,
                biological(9, sum(x["confidence"] for x in retrievals) / max(1, len(retrievals))))

        # 11. Integrate only retrieved candidates that pass the evidence-confidence threshold.
        integrations = []
        retrieval_by_id = {str(item["dependency_id"]): item for item in retrievals}
        for dependency in current.dependencies.values():
            if dependency.open and dependency.identifier in retrieval_by_id:
                integrations.append(integrate_retrieval(
                    current, dependency, retrieval_by_id[dependency.identifier],
                ))
        integration_output = {
            "integrations": integrations,
            "integrated_count": sum(bool(item.get("integrated")) for item in integrations),
            "event_model": [dict(item) for item in current.event_model],
        }
        _record(trace, 10, integration_output,
                biological(10, integration_output["integrated_count"] / max(1, len(integrations))))

        # 12. Detect possible overload from multiple converging indicators.
        overload = detect_overload(
            current, request.metrics, context_support, config.overload_threshold,
        )
        _record(trace, 11, overload, biological(11, overload["probability"]))

        # 13. Identify representations that are most vulnerable to loss or competition.
        at_risk = identify_at_risk(current)
        _record(trace, 12, at_risk,
                biological(12, max((x["risk"] for x in at_risk["at_risk_items"]), default=0.0)))

        # 14. Select and apply a recovery strategy only when the state contains a genuine problem.
        open_count = sum(item.open for item in current.dependencies.values())
        needs_recovery = overload["detected"] or open_count > 0 or bool(at_risk["at_risk_items"] and at_risk["at_risk_items"][0]["risk"] >= 0.65)
        if needs_recovery:
            recovery_choice = choose_recovery_action(
                request.context, load["load"], len(at_risk["at_risk_items"]), config,
            )
            recovery_application = apply_recovery(current, recovery_choice["selected_action"])
        else:
            recovery_choice = {
                "selected_action": {"action": "no_action", "net_value": 0.0},
                "ranked_actions": [], "at_risk_count": 0,
                "decision_rule": "no recovery required by current evidence",
            }
            recovery_application = {"action": "no_action", "affected_ids": [],
                                    "remaining_open_dependencies": 0,
                                    "recovered_from_existing_evidence_only": True}
        recovery_output = {**recovery_choice, "application": recovery_application,
                           "recovery_was_needed": needs_recovery}
        _record(trace, 13, recovery_output,
                biological(13, min(1.0, len(recovery_application["affected_ids"]) / 5.0)))

        # 15. Diagnose weak stages without collapsing distinct failure causes.
        diagnosis = diagnose_failures(current, retrievals)
        _record(trace, 14, diagnosis,
                biological(14, min(1.0, diagnosis["failure_count"] / 5.0)))

        # 16. Reconstruct only when evidence references existing representations; preserve unknowns.
        reconstruction = reconstruct_supported(current, request.context)
        _record(trace, 15, reconstruction,
                biological(15, min(1.0, len(reconstruction["supported_relations"]) / 5.0)))

        # 17. Check global coherence using supplied checks; unprovided checks remain marked missing.
        coherence = check_coherence(current, request.context)
        _record(trace, 16, coherence, biological(16, coherence["score"]))

        # 18. Commit a revisable partial interpretation with provenance and explicit unknowns.
        interpretation = commit_interpretation(current, reconstruction, coherence)
        _record(trace, 17, interpretation,
                biological(17, len(interpretation["supported"]) / max(1, len(current.dependencies))))

        # 19. Adapt future task weights from observed performance error, not from guessed neural gradients.
        adaptation = adapt_difficulty(current, request.metrics, config)
        current.step_index += len(units)
        _record(trace, 18, adaptation,
                biological(18, min(1.0, adaptation["mean_absolute_weight_change"])))

        biology_snapshot = network.snapshot() if network else {
            "model": "disabled", "operation_count": 0,
            "scientific_limitations": ["Biological-reference simulation disabled by configuration."],
        }
        return Point19Result(
            sequence_id=request.sequence_id,
            interpretation=interpretation,
            load_assessment={"load": load, "overload": overload, "at_risk": at_risk},
            retrieval={"retrieval": retrieval_output, "integration": integration_output,
                         "maintenance": maintenance, "interference": interference},
            recovery=recovery_output,
            failure_diagnosis=diagnosis,
            coherence=coherence,
            adaptation=adaptation,
            biological=biology_snapshot,
            operation_trace=tuple(trace),
            final_state=current.to_dict(),
            metadata={
                "model": "ORE Point 19 functional plus biological-reference implementation",
                "source_sequence_steps": 19,
                "token_count": len(units),
                "persistent_processor_state": state is None,
                "durable_episodic_memory_write": False,
                "scientific_status": "computational synthesis; not a validated human neural model",
            },
        )


def register_annotations(
    database_path: str | Path | None = None,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Index every Point 19 class, function, and method in the shared ORE logger SQLite catalogue."""
    from ore_logger.catalog import sync_repository
    from ore_logger.database import LoggerDatabase

    root = Path(repository_root).expanduser().resolve() if repository_root else Path(__file__).resolve().parents[1]
    target = Path(database_path).expanduser().resolve() if database_path else root / "ore_logs" / "ore_logger.sqlite3"
    catalogue = sync_repository(LoggerDatabase(target), root,
                                include_paths={"ore_working_memory_overload"})
    return {**catalogue, "database_path": str(target),
            "scope": "ore_working_memory_overload", "annotation_table": "annotations"}


def record_result_to_sqlite(
    result: Point19Result,
    database_path: str | Path | None = None,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Persist all 19 operation events and synchronize every Point 19 symbol annotation."""
    from ore_logger.catalog import sync_repository
    from ore_logger.database import LoggerDatabase

    if not isinstance(result, Point19Result):
        raise TypeError("result must be a Point19Result")
    root = Path(repository_root).expanduser().resolve() if repository_root else Path(__file__).resolve().parents[1]
    target = Path(database_path).expanduser().resolve() if database_path else root / "ore_logs" / "ore_logger.sqlite3"
    database = LoggerDatabase(target)
    catalogue = sync_repository(database, root, include_paths={"ore_working_memory_overload"})
    run_id = database.create_run(
        script_path="ore_working_memory_overload/point19",
        module_name="ore_working_memory_overload.pipeline",
        entry_kind="working_memory_overload_point19",
        working_directory=str(root),
    )
    started = time.perf_counter()
    try:
        for sequence, item in enumerate(result.operation_trace, start=1):
            database.record_event(
                run_id=run_id,
                sequence_no=sequence,
                event_kind="operation",
                source_path="ore_working_memory_overload/pipeline.py",
                elapsed_ms=0.0,
                metadata={
                    "operation_id": item["operation_id"],
                    "operation": item["operation"],
                    "status": item["status"],
                    "evidence_level": item["evidence_level"],
                    "mathematical_form": item["mathematical_form"],
                    "biological_reference": item["biological_reference"],
                    "functional_output": item["functional_output"],
                    "biological_output": item["biological_output"],
                },
            )
        database.finish_run(
            run_id, "completed", duration_ms=round((time.perf_counter() - started) * 1000.0, 3),
        )
    except Exception as exc:
        database.finish_run(
            run_id, "failed", duration_ms=round((time.perf_counter() - started) * 1000.0, 3),
            error_summary=f"{type(exc).__name__}: {exc}",
        )
        raise
    return {"run_id": run_id, "annotation_catalogue": catalogue,
            "database_path": str(target), "database_summary": database.summary()}
