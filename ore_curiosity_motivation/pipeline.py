"""Twenty-four-step orchestration and shared SQLite annotation/event persistence for Point 08."""
from __future__ import annotations

from pathlib import Path
import time
from typing import Any

from .biology import DistributedMotivationNetwork, biological_drives_for_step
from .contracts import LearningRequest, LearningResult, LearningState, MotivationConfig
from .functional import FUNCTIONAL_STEPS
from .registry import OPERATION_REGISTRY


class Point08MotivatedLearningProcessor:
    """Execute the 24 source-aligned functional operations and separate biological-reference trace."""

    def __init__(self, config: MotivationConfig | None = None, simulate_biology: bool = True,
                 biological_network: DistributedMotivationNetwork | None = None,
                 reset_biology_each_process: bool = True) -> None:
        """Configure numerical parameters and optional recurrent biological-reference simulation."""
        self.config = config or MotivationConfig()
        self.simulate_biology = bool(simulate_biology)
        self.biological_network = biological_network or DistributedMotivationNetwork()
        self.reset_biology_each_process = bool(reset_biology_each_process)

    def process(self, request: LearningRequest,
                state: LearningState | None = None) -> LearningResult:
        """Run every source operation once and return its functional and population-rate outputs."""
        if not isinstance(request, LearningRequest):
            raise TypeError("request must be a LearningRequest")
        current_state = state if state is not None else LearningState()
        if self.simulate_biology and self.reset_biology_each_process:
            self.biological_network.reset()
        outputs: dict[str, Any] = {}
        operation_trace: list[dict[str, Any]] = []
        biological_trace: list[dict[str, Any]] = []
        for index, (spec, function) in enumerate(zip(OPERATION_REGISTRY, FUNCTIONAL_STEPS), start=1):
            functional_output = function(request, outputs, current_state, self.config)
            key = f"step_{index:02d}_{function.__name__[8:]}"
            outputs[key] = functional_output
            rpe = None
            if index == 12:
                errors = functional_output["prediction_errors"]
                observed = [item["prediction_error"] for item in errors
                            if item["prediction_error"] is not None]
                rpe = sum(observed)/len(observed) if observed else None
            if self.simulate_biology:
                drives = biological_drives_for_step(request, outputs)
                biological_output = self.biological_network.advance(
                    drives, reward_prediction_error=rpe)
            else:
                biological_output = {"simulated": False,
                                     "reason": "Biological-reference track disabled by configuration.",
                                     "biological_scope": "not simulated"}
            biological_trace.append(biological_output)
            operation_trace.append({
                "operation_id": spec.operation_id,
                "source_step": spec.source_step,
                "operation": spec.name,
                "function": function.__name__,
                "status": "completed",
                "mathematical_form": spec.mathematical_form,
                "biological_reference": spec.biological_reference,
                "evidence_level": spec.evidence_level,
                "functional_output": functional_output,
                "biological_output": biological_output,
            })
        diagnostics = {
            "valid": len(operation_trace) == 24 and all(item["status"] == "completed"
                                                         for item in operation_trace),
            "operation_count": len(operation_trace),
            "functional_step_count": len(outputs),
            "biological_step_count": len(biological_trace),
            "biological_simulation_enabled": self.simulate_biology,
            "biological_scope": "distributed population-rate engineering abstraction; not a validated brain simulation",
            "target_memory_count": len(current_state.memory_candidates),
            "incidental_memory_count": len(current_state.incidental_memory_candidates),
            "delayed_learning_measured": outputs["step_23_test_durable_understanding"]["durable_learning"] is not None,
            "limitations": [
                "Material segmentation is supplied through LearningUnit objects.",
                "Actions, evidence, and outcome labels are supplied by the caller; no hidden web search is performed.",
                "Interest and curiosity are not treated as proof of successful learning.",
                "Reward-prediction-error input is computed only from explicitly observed learning outcomes.",
            ],
        }
        return LearningResult(
            request_id=request.request_id,
            operation_trace=tuple(operation_trace),
            functional_outputs=outputs,
            biological_outputs=tuple(biological_trace),
            diagnostics=diagnostics,
            final_state=current_state.to_dict(),
        )


def register_annotations(database_path: str | Path | None = None,
                         repository_root: str | Path | None = None) -> dict[str, Any]:
    """Index every Point 08 package module, class, function, and method in the shared SQLite catalogue."""
    from ore_logger.catalog import sync_repository
    from ore_logger.database import LoggerDatabase

    root = Path(repository_root).expanduser().resolve() if repository_root else Path(__file__).resolve().parents[1]
    target = Path(database_path).expanduser().resolve() if database_path else root / "ore_logs" / "ore_logger.sqlite3"
    catalogue = sync_repository(LoggerDatabase(target), root,
                                include_paths={"ore_curiosity_motivation"})
    return {**catalogue, "database_path": str(target),
            "scope": "ore_curiosity_motivation", "annotation_table": "annotations"}


def record_result_to_sqlite(result: LearningResult, database_path: str | Path | None = None,
                            repository_root: str | Path | None = None) -> dict[str, Any]:
    """Persist 24 operation events linked to their function and processor-class annotations."""
    from ore_logger.catalog import sync_repository
    from ore_logger.database import LoggerDatabase

    if not isinstance(result, LearningResult):
        raise TypeError("result must be a LearningResult")
    root = Path(repository_root).expanduser().resolve() if repository_root else Path(__file__).resolve().parents[1]
    target = Path(database_path).expanduser().resolve() if database_path else root / "ore_logs" / "ore_logger.sqlite3"
    database = LoggerDatabase(target)
    catalogue = sync_repository(database, root, include_paths={"ore_curiosity_motivation"})
    run_id = database.create_run(
        script_path="ore_curiosity_motivation/point08",
        module_name="ore_curiosity_motivation.pipeline",
        entry_kind="motivated_learning",
        working_directory=str(root),
    )
    class_row = database.get_annotation(
        "ore_curiosity_motivation.pipeline", "Point08MotivatedLearningProcessor",
        "class", "ore_curiosity_motivation/pipeline.py")
    started = time.perf_counter()
    try:
        for sequence, (item, function) in enumerate(zip(result.operation_trace, FUNCTIONAL_STEPS), start=1):
            row = database.get_annotation(
                "ore_curiosity_motivation.functional", function.__name__, "function",
                "ore_curiosity_motivation/functional.py")
            database.record_event(
                run_id=run_id, sequence_no=sequence, event_kind="operation",
                symbol_id=row["symbol_id"] if row else None,
                class_symbol_id=class_row["symbol_id"] if class_row else None,
                source_path="ore_curiosity_motivation/functional.py",
                line_number=row.get("line_start") if row else None,
                elapsed_ms=round((time.perf_counter()-started)*1000.0, 3),
                metadata={
                    "operation_id": item["operation_id"], "operation": item["operation"],
                    "function": item["function"], "status": item["status"],
                    "mathematical_form": item["mathematical_form"],
                    "evidence_level": item["evidence_level"],
                    "functional_output": item["functional_output"],
                    "biological_output": item["biological_output"],
                })
        database.finish_run(run_id, "completed",
                            duration_ms=round((time.perf_counter()-started)*1000.0, 3))
    except Exception as exc:
        database.finish_run(run_id, "failed",
                            duration_ms=round((time.perf_counter()-started)*1000.0, 3),
                            error_summary=type(exc).__name__+": "+str(exc))
        raise
    return {"run_id": run_id, "annotation_catalogue": catalogue,
            "database_summary": database.summary(),
            "events_written": len(result.operation_trace),
            "class_annotation_linked": class_row is not None}
