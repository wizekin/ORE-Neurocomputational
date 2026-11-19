"""Orchestration and SQLite persistence for the Point 07 feedback-seeking pipeline."""
from __future__ import annotations

from pathlib import Path
import time
from typing import Any

from .biology import DistributedFeedbackNetwork, biological_drives_for_step
from .contracts import FeedbackConfig, FeedbackRequest, FeedbackResult, FeedbackState
from .functional import FUNCTIONAL_STEPS
from .registry import OPERATION_REGISTRY


class Point07FeedbackProcessor:
    """Run the eighteen functional feedback operations with an optional distributed biological reference."""

    def __init__(self, config: FeedbackConfig | None = None,
                 simulate_biology: bool = True,
                 biological_network: DistributedFeedbackNetwork | None = None,
                 reset_biology_each_process: bool = True) -> None:
        """Configure thresholds, network reuse, and whether each episode resets biological state."""
        self.config = config or FeedbackConfig()
        self.simulate_biology = bool(simulate_biology)
        self.biological_network = biological_network or DistributedFeedbackNetwork()
        self.reset_biology_each_process = bool(reset_biology_each_process)

    def reset_biology(self) -> None:
        """Reset recurrent population states before processing an unrelated episode."""
        self.biological_network.reset()

    def process(self, request: FeedbackRequest,
                state: FeedbackState | None = None) -> FeedbackResult:
        """Execute all eighteen source-ordered steps and return separate functional and biological traces."""
        if not isinstance(request, FeedbackRequest):
            raise TypeError("request must be a FeedbackRequest")
        current_state = state if state is not None else FeedbackState()
        outputs: dict[str, Any] = {}
        biological_trace: list[dict[str, Any]] = []
        operation_trace: list[dict[str, Any]] = []
        if self.simulate_biology and self.reset_biology_each_process:
            self.biological_network.reset()
        for index, (spec, operation) in enumerate(zip(OPERATION_REGISTRY, FUNCTIONAL_STEPS), start=1):
            functional_output = operation(request, outputs, current_state, self.config)
            key = f"step_{index:02d}_{operation.__name__[8:]}"
            outputs[key] = functional_output
            if self.simulate_biology:
                drives = biological_drives_for_step(index, request, outputs)
                biological_output = self.biological_network.advance(drives)
            else:
                biological_output = {
                    "simulated": False,
                    "reason": "Biological reference track disabled by configuration.",
                    "biological_scope": "not simulated",
                }
            biological_trace.append(biological_output)
            operation_trace.append({
                "operation_id": spec.operation_id,
                "source_step": spec.source_step,
                "operation": spec.name,
                "function": operation.__name__,
                "status": "completed",
                "mathematical_form": spec.mathematical_form,
                "biological_reference": spec.biological_reference,
                "evidence_level": spec.evidence_level,
                "functional_output": functional_output,
                "biological_output": biological_output,
            })
        retrieval = outputs.get("step_13_test_retrieval", {})
        transfer = outputs.get("step_16_test_transfer", {})
        diagnostics = {
            "valid": len(operation_trace) == 18 and all(item["status"] == "completed" for item in operation_trace),
            "operation_count": len(operation_trace),
            "functional_step_count": len(outputs),
            "biological_step_count": len(biological_trace),
            "biological_simulation_enabled": self.simulate_biology,
            "biological_scope": "distributed population-rate engineering analogy; not a validated human brain simulation",
            "retrieval_verified": retrieval.get("retrieval_success") is not None,
            "transfer_verified": transfer.get("transfer_score") is not None,
            "correction_applied": outputs.get("step_10_apply_targeted_correction", {}).get("correction_applied", False),
            "source_verification_passed": outputs.get("step_07_verify_feedback", {}).get("feedback_trustworthy", False),
            "limitations": [
                "The caller supplies evidence, labels, sources, and transfer probes; the processor does not fetch or invent them.",
                "Lexical overlap is only a fallback proxy for semantic correctness.",
                "Biological populations are abstract, distributed reference signals, not measured neural data.",
                "A correction remains provisional when its evidence or independent verification is insufficient.",
            ],
        }
        return FeedbackResult(
            request_id=request.request_id,
            operation_trace=tuple(operation_trace),
            functional_outputs=outputs,
            biological_outputs=tuple(biological_trace),
            diagnostics=diagnostics,
            final_state=current_state.snapshot(),
        )


def register_annotations(database_path: str | Path | None = None,
                         repository_root: str | Path | None = None) -> dict[str, Any]:
    """Index every Point 07 module, class, function, and method in the shared logger SQLite catalogue."""
    from ore_logger.catalog import sync_repository
    from ore_logger.database import LoggerDatabase

    root = Path(repository_root).expanduser().resolve() if repository_root else Path(__file__).resolve().parents[1]
    target = Path(database_path).expanduser().resolve() if database_path else root / "ore_logs" / "ore_logger.sqlite3"
    catalogue = sync_repository(LoggerDatabase(target), root, include_paths={"ore_feedback_seeking"})
    return {**catalogue, "database_path": str(target),
            "scope": "ore_feedback_seeking", "annotation_table": "annotations"}


def record_result_to_sqlite(result: FeedbackResult, database_path: str | Path,
                            repository_root: str | Path | None = None) -> dict[str, Any]:
    """Persist all eighteen functional/biological operation pairs and link each event to SQLite annotations."""
    from ore_logger.catalog import sync_repository
    from ore_logger.database import LoggerDatabase

    if not isinstance(result, FeedbackResult):
        raise TypeError("result must be a FeedbackResult")
    root = Path(repository_root).expanduser().resolve() if repository_root else Path(__file__).resolve().parents[1]
    database = LoggerDatabase(database_path)
    catalogue = sync_repository(database, root, include_paths={"ore_feedback_seeking"})
    run_id = database.create_run(
        script_path="ore_feedback_seeking/point07",
        module_name="ore_feedback_seeking.pipeline",
        entry_kind="feedback_seeking_learning",
        working_directory=str(root),
    )
    class_annotation = database.get_annotation(
        "ore_feedback_seeking.pipeline", "Point07FeedbackProcessor", "class",
        "ore_feedback_seeking/pipeline.py",
    )
    started = time.perf_counter()
    try:
        for sequence, (item, function) in enumerate(zip(result.operation_trace, FUNCTIONAL_STEPS), start=1):
            annotation = database.get_annotation(
                "ore_feedback_seeking.functional", function.__name__, "function",
                "ore_feedback_seeking/functional.py",
            )
            database.record_event(
                run_id=run_id,
                sequence_no=sequence,
                event_kind="operation",
                symbol_id=annotation["symbol_id"] if annotation else None,
                class_symbol_id=class_annotation["symbol_id"] if class_annotation else None,
                source_path="ore_feedback_seeking/functional.py",
                line_number=annotation.get("line_start") if annotation else None,
                elapsed_ms=round((time.perf_counter() - started) * 1000.0, 3),
                metadata={
                    "operation_id": item["operation_id"],
                    "operation": item["operation"],
                    "function": item["function"],
                    "status": item["status"],
                    "mathematical_form": item["mathematical_form"],
                    "evidence_level": item["evidence_level"],
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
            "database_summary": database.summary(),
            "events_written": len(result.operation_trace),
            "class_annotation_linked": class_annotation is not None}
