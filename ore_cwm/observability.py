"""SQLite observability bridge into ORE's existing logger conventions.

Predictions, selected actions, observed outcomes, prediction errors, calibration,
drift, and fallback decisions are recorded as events in the shared ore_logger database
and as alignment rows in a dedicated table, mirroring the repository's existing
resource_module_history pattern.
"""
from __future__ import annotations

import sqlite3
import time
from collections.abc import Mapping
from pathlib import Path
from threading import RLock
from typing import Any

from .contracts import ObservedOutcome, PredictionRecord
from .learning import PredictionErrorReport

PREDICTION_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS cwm_prediction_outcomes (
    prediction_id TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    workspace_revision INTEGER NOT NULL,
    goal_id TEXT NOT NULL,
    operation_id TEXT NOT NULL,
    module_id TEXT NOT NULL,
    model_version TEXT NOT NULL,
    advisory_mode TEXT NOT NULL,
    predicted_success REAL NOT NULL,
    predicted_duration_ms REAL,
    predicted_goal_progress REAL,
    observed_success INTEGER,
    observed_duration_ms REAL,
    observed_goal_progress REAL,
    execution_status TEXT,
    prediction_error_success REAL,
    brier_score REAL,
    excluded_from_learning INTEGER NOT NULL DEFAULT 0,
    recorded_at_utc TEXT NOT NULL,
    PRIMARY KEY(prediction_id, operation_id)
)
"""


def register_cwm_annotations(
    database_path: str | Path | None = None,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Index every ore_cwm module, class, function, and method in ore_logger.annotations."""
    from ore_logger.catalog import sync_repository
    from ore_logger.database import LoggerDatabase

    root = Path(repository_root).expanduser().resolve() if repository_root else (
        Path(__file__).resolve().parents[1])
    target = Path(database_path).expanduser().resolve() if database_path else (
        root / "ore_logs" / "ore_logger.sqlite3")
    database = LoggerDatabase(target)
    report = sync_repository(database, root, include_paths={"ore_cwm"})
    return {**report, "database_path": str(target), "scope": "ore_cwm",
            "annotation_table": "annotations"}


class SQLiteCWMObserver:
    """Record CWM operations as method-linked events in ORE's shared SQLite logger."""

    def __init__(
        self,
        database_path: str | Path | None = None,
        repository_root: str | Path | None = None,
    ) -> None:
        """Resolve the logger database path and repository root without opening a run."""
        self.root = Path(repository_root).expanduser().resolve() if repository_root else (
            Path(__file__).resolve().parents[1])
        self.path = Path(database_path).expanduser().resolve() if database_path else (
            self.root / "ore_logs" / "ore_logger.sqlite3")
        self.database: Any | None = None
        self.run_id: str | None = None
        self.sequence_no = 0
        self.started = 0.0
        self._lock = RLock()

    def open(self) -> SQLiteCWMObserver:
        """Synchronize CWM annotations and create a single ORE logger execution run."""
        from ore_logger.database import LoggerDatabase

        with self._lock:
            if self.run_id is not None:
                return self
            register_cwm_annotations(self.path, self.root)
            self.database = LoggerDatabase(self.path)
            self.run_id = self.database.create_run(
                script_path="ore_cwm", module_name="ore_cwm",
                entry_kind="cognitive_world_model", working_directory=str(self.root),
            )
            self.started = time.perf_counter()
        return self

    def record(
        self,
        method_name: str,
        metadata: Mapping[str, Any],
        *,
        class_name: str,
        module_name: str,
        event_kind: str = "cognitive_world_model",
    ) -> dict[str, Any] | None:
        """Write an event linked to the method and class annotation records when available."""
        with self._lock:
            if self.database is None or self.run_id is None:
                return None
            source_path = module_name.replace(".", "/") + ".py"
            qualified_name = f"{class_name}.{method_name}" if class_name else method_name
            symbol = self.database.get_annotation(module_name, qualified_name, "function", source_path)
            if symbol is None:
                symbol = self.database.get_annotation(
                    module_name, qualified_name, "async_function", source_path)
            class_symbol = (
                self.database.get_annotation(module_name, class_name, "class", source_path)
                if class_name else None)
            self.sequence_no += 1
            return self.database.record_event(
                run_id=self.run_id,
                sequence_no=self.sequence_no,
                event_kind=event_kind,
                symbol_id=symbol["symbol_id"] if symbol else None,
                class_symbol_id=class_symbol["symbol_id"] if class_symbol else None,
                source_path=source_path,
                line_number=symbol.get("line_start") if symbol else None,
                elapsed_ms=round((time.perf_counter() - self.started) * 1000.0, 3),
                metadata={
                    **dict(metadata), "annotated_method": qualified_name,
                    "method_annotation_found": symbol is not None,
                    "class_annotation_found": class_symbol is not None,
                },
            )

    def close(self, status: str = "completed", error: BaseException | None = None) -> None:
        """Finish the logger run and retain failure details without altering workspace state."""
        with self._lock:
            if self.database is not None and self.run_id is not None:
                self.database.finish_run(
                    self.run_id, "failed" if error else status,
                    duration_ms=round((time.perf_counter() - self.started) * 1000.0, 3),
                    error_summary=f"{type(error).__name__}: {error}" if error else None,
                )
                self.run_id = None

    def __enter__(self) -> SQLiteCWMObserver:
        """Open this observer and begin a linked logger run for the context body."""
        return self.open()

    def __exit__(self, exc_type: object, exc: BaseException | None, traceback: object) -> bool:
        """Close the logger run while allowing exceptions to propagate to the caller."""
        self.close(error=exc)
        return False


class SQLitePredictionStore:
    """Persist prediction/outcome alignment rows in the shared SQLite logger database."""

    def __init__(self, database_path: str | Path) -> None:
        """Open (creating if needed) the prediction outcome table in one database."""
        self.path = Path(database_path).expanduser().resolve()
        self._lock = RLock()
        with self._lock, sqlite3.connect(self.path) as connection:
            connection.execute(PREDICTION_TABLE_SCHEMA)

    def record_alignment(
        self,
        prediction: PredictionRecord,
        outcome: ObservedOutcome,
        error: PredictionErrorReport | None = None,
    ) -> None:
        """Write one prediction-versus-outcome alignment row with its error metrics."""
        row = _alignment_row(prediction, outcome, error)
        with self._lock, sqlite3.connect(self.path) as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO cwm_prediction_outcomes (
                    prediction_id, workspace_id, workspace_revision, goal_id, operation_id,
                    module_id, model_version, advisory_mode, predicted_success,
                    predicted_duration_ms, predicted_goal_progress, observed_success,
                    observed_duration_ms, observed_goal_progress, execution_status,
                    prediction_error_success, brier_score, excluded_from_learning,
                    recorded_at_utc)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                row,
            )

    def recent(self, limit: int = 200) -> tuple[dict[str, Any], ...]:
        """Return the most recent alignment rows as dictionaries."""
        if int(limit) < 1:
            raise ValueError("limit must be a positive integer")
        with self._lock, sqlite3.connect(self.path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT * FROM cwm_prediction_outcomes ORDER BY rowid DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return tuple(dict(row) for row in rows)


def _alignment_row(
    prediction: PredictionRecord,
    outcome: ObservedOutcome,
    error: PredictionErrorReport | None,
) -> tuple[Any, ...]:
    """Flatten one alignment into the persisted row format, keeping missing values NULL."""
    from .contracts import MISSING

    predicted = prediction.predicted
    duration = predicted.expected_cost.expected_duration_ms
    observed_success: int | None = None
    if not error or not error.excluded_from_learning:
        observed_success = 1 if outcome.artifact_lifecycle.value == "accepted" else 0
    observed_duration = outcome.dimensions.elapsed_ms
    observed_progress = outcome.dimensions.goal_progress_delta
    return (
        prediction.prediction_id,
        prediction.workspace_id,
        prediction.workspace_revision,
        prediction.goal_id,
        prediction.operation_id,
        outcome.module_id,
        prediction.predictor_version,
        prediction.advisory_mode,
        float(predicted.success_probability),
        None if duration is MISSING else float(duration),
        float(predicted.goal_progress_estimate),
        observed_success,
        None if observed_duration is MISSING else float(observed_duration),
        None if observed_progress is MISSING else float(observed_progress),
        outcome.execution_status.value,
        None if error is None else error.success_error,
        None if error is None else error.brier_score,
        1 if (error is not None and error.excluded_from_learning) else 0,
        outcome.timestamp,
    )


__all__ = [
    "PREDICTION_TABLE_SCHEMA",
    "SQLiteCWMObserver",
    "SQLitePredictionStore",
    "register_cwm_annotations",
]
