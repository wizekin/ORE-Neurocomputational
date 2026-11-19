"""End-to-end 16-step functional and biological-reference processor for Point 06."""
from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Mapping

from .biology import DistributedCognitiveLoadNetwork
from .contracts import Point06Config, Point06Request, Point06Result, Point06State
from .functional import (
    adapt_learning_step, build_external_representation, build_guided_scaffold,
    clip01, estimate_element_interactivity, form_meaningful_chunks,
    integrate_chunks, monitor_overload, order_by_prerequisites, reduce_presentation_demands,
    retrieve_prerequisites, segment_material, select_principal_subgoal,
    select_working_set, update_retrieval_and_support, validate_dependencies,
    verify_comprehension,
)
from .registry import OPERATION_REGISTRY

_POPULATION_BY_STEP = (
    "prefrontal_goal_control", "parietal_attention_selection",
    "hippocampal_associative_retrieval", "temporal_semantic_language",
    "cortical_schema_integration", "cortical_schema_integration",
    "prefrontal_goal_control", "parietal_attention_selection",
    "sensory_external_representation", "prefrontal_goal_control",
    "cingulo_opercular_monitoring", "prefrontal_goal_control",
    "cortical_schema_integration", "temporal_semantic_language",
    "hippocampal_associative_retrieval", "cingulo_opercular_monitoring",
)


def _append_operation(
    trace: list[dict[str, Any]], index: int,
    functional_output: Mapping[str, Any], biological_output: Mapping[str, Any],
) -> None:
    """Append one source-ordered step with its equation, evidence level, and separate output channels."""
    spec = OPERATION_REGISTRY[index]
    trace.append({
        "step": spec.source_step, "source_step": spec.source_step,
        "operation_id": spec.operation_id, "operation": spec.name,
        "status": "completed", "purpose": spec.purpose,
        "evidence_level": spec.evidence_level,
        "mathematical_form": spec.mathematical_form,
        "biological_reference": spec.biological_reference,
        "functional_output": dict(functional_output),
        "biological_output": dict(biological_output),
    })


class Point06CognitiveLoadProcessor:
    """Coordinate goal selection, load estimation, chunking, external support, and comprehension checks."""

    def __init__(
        self, config: Point06Config | None = None, state: Point06State | None = None,
        biological: DistributedCognitiveLoadNetwork | None = None,
    ) -> None:
        """Initialize task-calibrated parameters, persistent episode state, and an optional rate-coded network."""
        self.config = config or Point06Config()
        self.state = state or Point06State()
        self.biological = biological or DistributedCognitiveLoadNetwork(self.config)
        self.process_count = 0

    def process(
        self, request: Point06Request, state: Point06State | None = None,
    ) -> Point06Result:
        """Execute all sixteen source steps and return diagnostics, external structure, and biology proxy."""
        if not isinstance(request, Point06Request):
            raise TypeError("request must be a Point06Request")
        current = state if state is not None else self.state
        config = self.config
        network = self.biological if config.simulate_biology else None
        self.process_count += 1
        current.process_count += 1
        current.active_goal = request.goal
        trace: list[dict[str, Any]] = []
        units = segment_material(request)
        dependencies = validate_dependencies(units, request.dependencies)

        def biological(index: int, drive: float) -> dict[str, Any]:
            """Advance the mapped population or mark the biological-reference channel as disabled."""
            if network is None:
                return {"model": "disabled", "operation": OPERATION_REGISTRY[index].operation_id,
                        "scope": "Biological-reference simulation disabled by configuration."}
            return network.engage(OPERATION_REGISTRY[index].operation_id,
                                  _POPULATION_BY_STEP[index], clip01(drive))

        # Step 1 — define the goal and select a candidate unit using normalized relevance scores.
        goal_selection = select_principal_subgoal(request, units, dependencies, config)
        _append_operation(trace, 0, goal_selection,
                          biological(0, max((item["utility"] for item in goal_selection["unit_candidate_scores"].values()), default=0.0)))

        # Step 2 — estimate interacting elements and the effect of prior familiarity.
        complexity = estimate_element_interactivity(units, dependencies, config)
        current.estimated_load = complexity["effective_complexity"]
        _append_operation(trace, 1, complexity, biological(1, complexity["effective_complexity"]))

        # Step 3 — retrieve only caller-supplied memories associated with prerequisites and the current goal.
        selected_id = goal_selection["selected_unit_id"]
        prerequisite_retrieval = retrieve_prerequisites(
            units, dependencies, request.prior_knowledge,
            target_ids=[selected_id] if selected_id else None,
        )
        _append_operation(trace, 2, prerequisite_retrieval,
                          biological(2, min(1.0, len(prerequisite_retrieval["retrieved_memories"]) / 5.0)))

        # Step 4 — make source-grounded information units and preserve segmentation provenance.
        segmentation = {"units": [unit.to_dict() for unit in units], "unit_count": len(units),
                        "segmentation_method": "caller_supplied_units" if request.units is not None
                        else "sentence_paragraph_and_semicolon_boundaries",
                        "source_text_preserved": request.text is not None}
        _append_operation(trace, 3, segmentation,
                          biological(3, min(1.0, len(units) / 16.0)))

        # Step 5 — build reversible chunks only from explicit or observable local coherence evidence.
        chunking = form_meaningful_chunks(units, dependencies, config.chunk_similarity_threshold)
        current.chunks = [dict(item) for item in chunking["chunks"]]
        _append_operation(trace, 4, chunking,
                          biological(4, min(1.0, chunking["chunk_count"] / max(1, len(units)))))

        # Step 6 — sequence prerequisites first and expose cycles rather than silently breaking them.
        ordering = order_by_prerequisites(units, dependencies)
        _append_operation(trace, 5, ordering,
                          biological(5, min(1.0, len(ordering["ordered_unit_ids"]) / max(1, len(units)))))

        # Step 7 — select one current subgoal and preserve the deferred alternatives.
        current.active_subgoal = goal_selection["selected_subgoal"]
        _append_operation(trace, 6, goal_selection,
                          biological(6, max((item["probability"] for item in goal_selection["unit_candidate_scores"].values()), default=0.0)))

        # Step 8 — quantify avoidable presentation costs without deleting essential information.
        presentation = reduce_presentation_demands(request.presentation_costs)
        _append_operation(trace, 7, presentation,
                          biological(7, presentation["avoidable_load"]))

        # Step 9 — externalize concepts and only supplied prerequisite/semantic relations.
        external = build_external_representation(
            units, chunking["chunks"], dependencies, request.relations, request.goal,
        )
        current.external_representation = external
        _append_operation(trace, 8, external,
                          biological(8, min(1.0, len(external["nodes"]) / 20.0)))

        # Step 10 — select an engineering-bounded active set for the current subgoal.
        working_set = select_working_set(
            units, dependencies, selected_id, config.working_set_limit,
        )
        current.active_working_set = list(working_set["active_unit_ids"])
        _append_operation(trace, 9, working_set,
                          biological(9, min(1.0, len(working_set["active_unit_ids"]) / config.working_set_limit)))

        # Step 11 — combine error, rereading, switching, fatigue, effort, and task-demand indicators.
        monitor = monitor_overload(request.metrics, complexity["effective_complexity"], config.overload_threshold)
        _append_operation(trace, 10, monitor,
                          biological(10, monitor["overload_score"]))

        # Step 12 — adapt complexity and scaffolding conservatively to evidence and capacity estimate.
        adaptation = adapt_learning_step(
            current, complexity["effective_complexity"], request.resource_capacity,
            monitor, config,
        )
        _append_operation(trace, 11, adaptation,
                          biological(11, clip01(adaptation["utilization"] / max(config.utilization_high, 1e-9))))

        # Step 13 — use supplied examples and prompts; do not invent subject-matter facts.
        scaffolding = build_guided_scaffold(chunking["chunks"], request.examples, current.support_level)
        _append_operation(trace, 12, scaffolding,
                          biological(12, current.support_level))

        # Step 14 — integrate chunks through supported explicit edges and retain unresolved links.
        integration = integrate_chunks(chunking["chunks"], dependencies, request.relations)
        _append_operation(trace, 13, integration,
                          biological(13, min(1.0, len(integration["cross_chunk_links"]) / max(1, len(units)))))

        # Step 15 — test reconstruction before fading support and retain the support state when evidence is absent.
        consolidation = update_retrieval_and_support(current, request.metrics, config)
        _append_operation(trace, 14, consolidation,
                          biological(14, clip01((consolidation["reconstruction_accuracy"] or 0.0))))

        # Step 16 — verify local, relational, transfer, and explanation evidence independently.
        comprehension = verify_comprehension(request.metrics)
        _append_operation(trace, 15, comprehension,
                          biological(15, clip01(comprehension["score"] if comprehension["score"] is not None else 0.0)))

        explicitly_completed = request.learner_profile.get("completed_unit_ids", ())
        if isinstance(explicitly_completed, (str, bytes)):
            explicitly_completed = (explicitly_completed,)
        for completed_id in explicitly_completed:
            if completed_id not in {unit.identifier for unit in units}:
                raise ValueError(f"completed_unit_ids contains unknown unit: {completed_id}")
            if completed_id not in current.completed_units:
                current.completed_units.append(str(completed_id))
        biology_snapshot = network.snapshot() if network else {
            "model": "disabled", "operation_count": 0,
            "scientific_limitations": ["Biological-reference simulation disabled by configuration."],
        }
        return Point06Result(
            sequence_id=request.sequence_id, goal=request.goal,
            goal_selection=goal_selection, complexity=complexity,
            prerequisite_retrieval=prerequisite_retrieval, chunking=chunking,
            presentation=presentation, external_representation=external,
            working_set=working_set, overload_monitor=monitor,
            adaptation=adaptation, scaffolding=scaffolding,
            integration=integration, consolidation=consolidation,
            comprehension=comprehension, biological=biology_snapshot,
            operation_trace=tuple(trace), final_state=current.to_dict(),
            metadata={
                "model": "ORE Point 06 functional and distributed biological-reference implementation",
                "source_sequence_steps": 16,
                "input_unit_count": len(units),
                "dependency_count": len(dependencies),
                "completed_unit_ids_observed": list(current.completed_units),
                "durable_memory_write": False,
                "scientific_status": "computational synthesis; not a validated quantitative model of human neural activity",
                "equations_are_neural_claims": False,
                "source_reference": request.source_reference,
            },
        )


def register_annotations(
    database_path: str | Path | None = None,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Index every Point 06 class, function, and method in the shared ORE SQLite annotations table."""
    from ore_logger.catalog import sync_repository
    from ore_logger.database import LoggerDatabase

    root = Path(repository_root).expanduser().resolve() if repository_root else Path(__file__).resolve().parents[1]
    target = Path(database_path).expanduser().resolve() if database_path else root / "ore_logs" / "ore_logger.sqlite3"
    catalogue = sync_repository(LoggerDatabase(target), root, include_paths={"ore_cognitive_load"})
    return {**catalogue, "database_path": str(target),
            "scope": "ore_cognitive_load", "annotation_table": "annotations"}


def record_result_to_sqlite(
    result: Point06Result,
    database_path: str | Path | None = None,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Persist all sixteen operation events and synchronize all Point 06 symbol annotations to SQLite."""
    from ore_logger.catalog import sync_repository
    from ore_logger.database import LoggerDatabase

    if not isinstance(result, Point06Result):
        raise TypeError("result must be a Point06Result")
    root = Path(repository_root).expanduser().resolve() if repository_root else Path(__file__).resolve().parents[1]
    target = Path(database_path).expanduser().resolve() if database_path else root / "ore_logs" / "ore_logger.sqlite3"
    database = LoggerDatabase(target)
    catalogue = sync_repository(database, root, include_paths={"ore_cognitive_load"})
    run_id = database.create_run(
        script_path="ore_cognitive_load/point06", module_name="ore_cognitive_load.pipeline",
        entry_kind="cognitive_load_point06", working_directory=str(root),
    )
    started = time.perf_counter()
    try:
        for sequence, item in enumerate(result.operation_trace, start=1):
            database.record_event(
                run_id=run_id, sequence_no=sequence, event_kind="operation",
                source_path="ore_cognitive_load/pipeline.py",
                elapsed_ms=round((time.perf_counter() - started) * 1000.0, 3),
                metadata={
                    "operation_id": item["operation_id"], "operation": item["operation"],
                    "status": item["status"], "source_step": item["source_step"],
                    "evidence_level": item["evidence_level"],
                    "mathematical_form": item["mathematical_form"],
                    "biological_reference": item["biological_reference"],
                    "functional_output": item["functional_output"],
                    "biological_output": item["biological_output"],
                },
            )
        database.finish_run(run_id, "completed",
                            duration_ms=round((time.perf_counter() - started) * 1000.0, 3))
    except Exception as exc:
        database.finish_run(run_id, "failed",
                            duration_ms=round((time.perf_counter() - started) * 1000.0, 3),
                            error_summary=f"{type(exc).__name__}: {exc}")
        raise
    return {"run_id": run_id, "annotation_catalogue": catalogue,
            "database_path": str(target), "database_summary": database.summary()}