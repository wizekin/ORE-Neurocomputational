"""Sixteen-step Point 20 schema-driven inference and reconstructive-memory processor."""
from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Mapping

from .biology import ReconstructiveMemoryNetwork
from .contracts import (
    Point20Config, Point20Request, Point20Result, Point20State,
)
from .functional import (
    activate_candidate_schemas, assess_familiarity_and_recollection,
    build_feature_ledger, build_provenance_graph, calibrate_detail_confidence,
    check_spatiotemporal_causal_constraints, choose_response_policy,
    define_retrieval_target, evaluate_candidates_bayesian,
    finalize_reconstruction, generate_inferred_completions,
    monitor_candidate_sources, monitor_retelling_and_post_event_information,
    reactivate_episodic_fragments, search_competing_episodes,
    update_general_schema_knowledge, update_reconstruction_from_feedback,
)
from .registry import OPERATION_REGISTRY


_POPULATION_DRIVES = (
    {"lateral_prefrontal_source_control": 0.80, "distributed_event_integration": 0.35},
    {"hippocampal_pattern_completion": 0.90, "sensory_association_reinstatement": 0.55,
     "parietal_episodic_detail": 0.35},
    {"parietal_episodic_detail": 0.65, "lateral_prefrontal_source_control": 0.55},
    {"medial_prefrontal_schema": 0.85, "temporal_semantic_memory": 0.70},
    {"medial_prefrontal_schema": 0.65, "temporal_semantic_memory": 0.80},
    {"hippocampal_pattern_completion": 0.75, "lateral_prefrontal_source_control": 0.85},
    {"lateral_prefrontal_source_control": 0.75, "distributed_event_integration": 0.65},
    {"lateral_prefrontal_source_control": 0.90, "hippocampal_pattern_completion": 0.50},
    {"parietal_episodic_detail": 0.70, "lateral_prefrontal_source_control": 0.60},
    {"hippocampal_pattern_completion": 0.80, "lateral_prefrontal_source_control": 0.85},
    {"lateral_prefrontal_source_control": 0.80, "parietal_episodic_detail": 0.55},
    {"lateral_prefrontal_source_control": 0.90, "distributed_event_integration": 0.55},
    {"hippocampal_pattern_completion": 0.65, "lateral_prefrontal_source_control": 0.75},
    {"hippocampal_pattern_completion": 0.65, "lateral_prefrontal_source_control": 0.80},
    {"medial_prefrontal_schema": 0.75, "temporal_semantic_memory": 0.80},
    {"distributed_event_integration": 0.90, "hippocampal_pattern_completion": 0.65},
)


def _record_operation(
    trace: list[dict[str, Any]],
    index: int,
    functional_output: Mapping[str, Any],
    biological_output: Mapping[str, Any],
) -> None:
    """Append one source-aligned operation with equations and two separate output tracks."""
    spec = OPERATION_REGISTRY[index]
    trace.append({
        "step": index + 1,
        "operation_id": spec.operation_id,
        "operation": spec.name,
        "purpose": spec.purpose,
        "mathematical_form": spec.mathematical_form,
        "biological_reference": spec.biological_reference,
        "evidence_level": spec.evidence_level,
        "status": "completed",
        "functional_output": dict(functional_output),
        "biological_output": dict(biological_output),
    })


class ReconstructiveMemoryProcessor:
    """Coordinate evidence retrieval, schema inference, source monitoring, and corrective learning.

    The processor keeps event-specific records separate from general schema counts. A schema can
    propose an inference but cannot turn that proposal into an original observation. The optional
    biological-reference network is a separate coarse population-rate model.
    """

    def __init__(
        self,
        config: Point20Config | None = None,
        state: Point20State | None = None,
        biological: ReconstructiveMemoryNetwork | None = None,
    ) -> None:
        """Initialize calibrated model settings, revisable memory state, and optional network proxy."""
        self.config = config or Point20Config()
        self.state = state or Point20State()
        self.biological = biological or ReconstructiveMemoryNetwork(self.config)
        self.process_count = 0

    def process(
        self,
        request: Point20Request,
        state: Point20State | None = None,
    ) -> Point20Result:
        """Execute the complete 16-operation reconstructive-memory algorithm."""
        if not isinstance(request, Point20Request):
            raise TypeError("request must be a Point20Request")
        current = state if state is not None else self.state
        config = self.config
        network = self.biological if config.simulate_biology else None
        trace: list[dict[str, Any]] = []
        self.process_count += 1
        current.run_count += 1

        def biological(index: int, salience: float = 0.0) -> dict[str, Any]:
            """Advance the biological-reference network or return a disabled-track record."""
            if network is None:
                return {
                    "model": "disabled",
                    "operation_id": OPERATION_REGISTRY[index].operation_id,
                    "scope": "Biological-reference simulation disabled by configuration.",
                }
            return network.advance(
                OPERATION_REGISTRY[index].operation_id,
                _POPULATION_DRIVES[index],
                salience=salience,
            )

        # 1. Define the retrieval target and the precision required by the question.
        target = define_retrieval_target(request)
        _record_operation(
            trace, 0, target,
            biological(0, float(request.context.get("emotional_salience", 0.0))),
        )

        # 2. Reactivate the particular event's available episodic fragments.
        retrieval = reactivate_episodic_fragments(request, target, config)
        _record_operation(
            trace, 1, retrieval,
            biological(1, float(request.context.get("emotional_salience", 0.0))),
        )

        # 3. Separate direct/retrieved evidence from weak, missing, contradicted, and unqueried details.
        ledger = build_feature_ledger(request, retrieval)
        _record_operation(trace, 2, ledger, biological(2))

        # 4. Activate several relevant schemas without treating them as episode records.
        schema_activation = activate_candidate_schemas(request, ledger, config)
        _record_operation(
            trace, 3, schema_activation,
            biological(3, max(
                (item["activation_probability"] for item in schema_activation["ranked_schemas"]),
                default=0.0,
            )),
        )

        # 5. Generate top-K alternatives for missing/weak features; all start as inferred.
        generation = generate_inferred_completions(request, ledger, schema_activation, config)
        candidates = list(generation["candidates"])
        _record_operation(trace, 4, {
            **{key: value for key, value in generation.items() if key != "candidates"},
            "candidates": [item.to_dict() for item in candidates],
        }, biological(4))

        # 6. Combine event-specific likelihoods and schema priors without allowing priors to erase evidence.
        evaluation = evaluate_candidates_bayesian(candidates, request.episodic_fragments, config)
        candidates = list(evaluation["candidates"])
        _record_operation(trace, 5, {
            **{key: value for key, value in evaluation.items() if key != "candidates"},
            "candidates": [item.to_dict() for item in candidates],
        }, biological(5))

        # 7. Test each proposed detail against explicit temporal, spatial, participant, and causal constraints.
        constraint_check = check_spatiotemporal_causal_constraints(candidates, request)
        candidates = list(constraint_check["candidates"])
        _record_operation(trace, 6, {
            **{key: value for key, value in constraint_check.items() if key != "candidates"},
            "candidates": [item.to_dict() for item in candidates],
        }, biological(6))

        # 8. Estimate whether details derive from perception, recollection, schema, imagination, or later sources.
        source_monitoring = monitor_candidate_sources(
            candidates, request.episodic_fragments, request.source_history, config,
        )
        candidates = list(source_monitoring["candidates"])
        _record_operation(trace, 7, {
            **{key: value for key, value in source_monitoring.items() if key != "candidates"},
            "candidates": [item.to_dict() for item in candidates],
        }, biological(7))

        # 9. Keep familiarity separate from source-specific episodic recollection.
        familiarity = assess_familiarity_and_recollection(
            candidates, request.episodic_fragments, request.source_history, request.precision,
        )
        candidates = list(familiarity["candidates"])
        _record_operation(trace, 8, {
            **{key: value for key, value in familiarity.items() if key != "candidates"},
            "candidates": [item.to_dict() for item in candidates],
        }, biological(8, float(request.context.get("emotional_salience", 0.0))))

        # 10. Search similar episodes for overlapping material and possible source blending.
        competition = search_competing_episodes(candidates, request, request.episodic_fragments)
        candidates = list(competition["candidates"])
        _record_operation(trace, 9, {
            **{key: value for key, value in competition.items() if key != "candidates"},
            "candidates": [item.to_dict() for item in candidates],
        }, biological(9))

        # 11. Calibrate confidence for every candidate independently of narrative coherence.
        confidence = calibrate_detail_confidence(candidates, config)
        candidates = list(confidence["candidates"])
        _record_operation(trace, 10, {
            **{key: value for key, value in confidence.items() if key != "candidates"},
            "candidates": [item.to_dict() for item in candidates],
        }, biological(10))

        # 12. Choose report, qualify, clarify, or omit by expected loss and task precision.
        policy = choose_response_policy(request, ledger, candidates, config)
        _record_operation(trace, 11, policy, biological(11))

        # 13. Build an immutable-source timeline and provenance DAG before applying new corrections.
        retelling = monitor_retelling_and_post_event_information(request, current, candidates)
        _record_operation(trace, 12, retelling, biological(12))

        # 14. Apply reliable, independent feedback to candidate odds; keep version history.
        feedback_update = update_reconstruction_from_feedback(
            candidates, request.feedback, current, config,
        )
        _record_operation(trace, 13, feedback_update, biological(13))

        # 15. Update general schema counts only from verified feedback authorized for schema learning.
        schema_update = update_general_schema_knowledge(request, current, config)
        _record_operation(trace, 14, schema_update, biological(14))

        # 16. Finalize the evidence-faithful report and refresh the provenance graph with feedback sources.
        provenance_graph = build_provenance_graph(request, candidates)
        reconstruction = finalize_reconstruction(
            request, ledger, candidates, policy, provenance_graph,
        )
        if request.event_id not in current.episode_records:
            current.episode_records[request.event_id] = {"versions": []}
        current.episode_records[request.event_id]["versions"].append({
            "version": len(current.episode_records[request.event_id]["versions"]) + 1,
            "reconstruction": reconstruction,
            "feedback_update": feedback_update,
        })
        current.reconstruction_versions.append({
            "event_id": request.event_id,
            "version": len(current.episode_records[request.event_id]["versions"]),
            "candidate_count": len(candidates),
            "supported_count": len(reconstruction["supported_details"]),
            "inferred_count": len(reconstruction["inferred_details"]),
        })
        _record_operation(trace, 15, {
            "reconstruction": reconstruction,
            "candidate_count": len(candidates),
            "supported_detail_count": len(reconstruction["supported_details"]),
            "inferred_detail_count": len(reconstruction["inferred_details"]),
            "unresolved_feature_count": len(reconstruction["unresolved_features"]),
            "provenance_node_count": len(provenance_graph["nodes"]),
            "provenance_edge_count": len(provenance_graph["edges"]),
        }, biological(15))

        biological_snapshot = network.snapshot() if network else {
            "model": "disabled",
            "operation_count": 0,
            "scientific_limitations": ["Biological-reference simulation disabled by configuration."],
        }
        result = Point20Result(
            event_id=request.event_id,
            reconstruction=reconstruction,
            retrieval=retrieval,
            feature_ledger=ledger,
            schema_activation=schema_activation,
            candidate_evaluation={
                "generation": {**{k: v for k, v in generation.items() if k != "candidates"},
                               "candidates": [item.to_dict() for item in generation["candidates"]]},
                "evaluation": {**{k: v for k, v in evaluation.items() if k != "candidates"},
                               "candidates": [item.to_dict() for item in evaluation["candidates"]]},
                "constraints": {**{k: v for k, v in constraint_check.items() if k != "candidates"},
                                "candidates": [item.to_dict() for item in constraint_check["candidates"]]},
                "competing_episodes": {**{k: v for k, v in competition.items() if k != "candidates"},
                                       "candidates": [item.to_dict() for item in competition["candidates"]]},
            },
            source_monitoring={**{k: v for k, v in source_monitoring.items() if k != "candidates"},
                               "source_assignments": source_monitoring["source_assignments"]},
            confidence={**{k: v for k, v in confidence.items() if k != "candidates"},
                        "per_detail": confidence["per_detail"]},
            response_policy=policy,
            retelling_monitor=retelling,
            feedback_update=feedback_update,
            schema_update=schema_update,
            provenance_graph=provenance_graph,
            biological=biological_snapshot,
            operation_trace=tuple(trace),
            final_state=current.to_dict(),
            metadata={
                "model": "ORE Point 20 schema-driven inference and reconstructive memory",
                "source_sequence_steps": 16,
                "task_type": request.task_type,
                "precision": request.precision,
                "fragment_count": len(request.episodic_fragments),
                "schema_count": len(request.schemas),
                "candidate_count": len(candidates),
                "biological_track_enabled": network is not None,
                "durable_event_record_versioned": True,
                "schema_learning_requires_verified_feedback": True,
                "not_retrieved_is_not_absent": True,
                "scientific_status": "computational synthesis; not a validated neural simulator",
            },
        )
        return result


def register_annotations(
    database_path: str | Path | None = None,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Register every Point 20 class, function, and method in the shared ORE SQLite annotation table."""
    from ore_logger.catalog import sync_repository
    from ore_logger.database import LoggerDatabase

    root = Path(repository_root).expanduser().resolve() if repository_root else Path(__file__).resolve().parents[1]
    target = Path(database_path).expanduser().resolve() if database_path else root / "ore_logs" / "ore_logger.sqlite3"
    catalogue = sync_repository(
        LoggerDatabase(target), root,
        include_paths={"ore_reconstructive_memory"},
    )
    return {
        **catalogue,
        "database_path": str(target),
        "scope": "ore_reconstructive_memory",
        "annotation_table": "annotations",
    }


def record_result_to_sqlite(
    result: Point20Result,
    database_path: str | Path | None = None,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Persist all 16 Point 20 operation events and synchronize package symbol annotations."""
    from ore_logger.catalog import sync_repository
    from ore_logger.database import LoggerDatabase

    if not isinstance(result, Point20Result):
        raise TypeError("result must be a Point20Result")
    root = Path(repository_root).expanduser().resolve() if repository_root else Path(__file__).resolve().parents[1]
    target = Path(database_path).expanduser().resolve() if database_path else root / "ore_logs" / "ore_logger.sqlite3"
    database = LoggerDatabase(target)
    catalogue = sync_repository(
        database, root,
        include_paths={"ore_reconstructive_memory"},
    )
    run_id = database.create_run(
        script_path="ore_reconstructive_memory/point20",
        module_name="ore_reconstructive_memory.pipeline",
        entry_kind="reconstructive_memory_point20",
        working_directory=str(root),
    )
    started = time.perf_counter()
    try:
        for sequence, item in enumerate(result.operation_trace, start=1):
            database.record_event(
                run_id=run_id,
                sequence_no=sequence,
                event_kind="operation",
                source_path="ore_reconstructive_memory/pipeline.py",
                elapsed_ms=0.0,
                metadata={
                    "operation_id": item["operation_id"],
                    "operation": item["operation"],
                    "status": item["status"],
                    "mathematical_form": item["mathematical_form"],
                    "biological_reference": item["biological_reference"],
                    "evidence_level": item["evidence_level"],
                    "functional_output": item["functional_output"],
                    "biological_output": item["biological_output"],
                },
            )
        database.finish_run(
            run_id,
            "completed",
            duration_ms=round((time.perf_counter() - started) * 1000.0, 3),
        )
    except Exception as exc:
        database.finish_run(
            run_id,
            "failed",
            duration_ms=round((time.perf_counter() - started) * 1000.0, 3),
            error_summary=f"{type(exc).__name__}: {exc}",
        )
        raise
    return {
        "run_id": run_id,
        "annotation_catalogue": catalogue,
        "database_path": str(target),
        "database_summary": database.summary(),
    }
