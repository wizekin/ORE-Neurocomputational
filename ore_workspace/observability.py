"""Bridge shared-workspace annotations and runtime events into ORE's existing SQLite logger."""
from __future__ import annotations

import time
from threading import RLock
from pathlib import Path
from typing import Any, Mapping


def register_workspace_annotations(
    database_path: str | Path | None = None,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Index every workspace module, class, function and method in ore_logger.annotations."""
    from ore_logger.catalog import sync_repository
    from ore_logger.database import LoggerDatabase

    root = Path(repository_root).expanduser().resolve() if repository_root else Path(__file__).resolve().parents[1]
    target = Path(database_path).expanduser().resolve() if database_path else root / "ore_logs" / "ore_logger.sqlite3"
    database = LoggerDatabase(target)
    report = sync_repository(database, root, include_paths={"ore_workspace"})
    return {**report, "database_path": str(target), "scope": "ore_workspace", "annotation_table": "annotations"}


class SQLiteWorkspaceObserver:
    """Record workspace operations as method-linked events in ORE's shared SQLite database."""

    def __init__(
        self,
        database_path: str | Path | None = None,
        repository_root: str | Path | None = None,
    ) -> None:
        """Resolve the existing logger database path and repository root without opening a run."""
        self.root = Path(repository_root).expanduser().resolve() if repository_root else Path(__file__).resolve().parents[1]
        self.path = Path(database_path).expanduser().resolve() if database_path else self.root / "ore_logs" / "ore_logger.sqlite3"
        self.database: Any | None = None
        self.run_id: str | None = None
        self.sequence_no = 0
        self.started = 0.0
        self._lock = RLock()

    def open(self) -> "SQLiteWorkspaceObserver":
        """Synchronize workspace annotations and create a single ORE logger execution run."""
        from ore_logger.database import LoggerDatabase

        with self._lock:
            if self.run_id is not None:
                return self
            register_workspace_annotations(self.path, self.root)
            self.database = LoggerDatabase(self.path)
            self.run_id = self.database.create_run(
                script_path="ore_workspace", module_name="ore_workspace",
                entry_kind="shared_workspace", working_directory=str(self.root),
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
        event_kind: str = "shared_workspace",
    ) -> dict[str, Any] | None:
        """Write an event linked to the method and class annotation records when available."""
        with self._lock:
            if self.database is None or self.run_id is None:
                return None
            source_path = module_name.replace(".", "/") + ".py"
            qualified_name = f"{class_name}.{method_name}" if class_name else method_name
            symbol = self.database.get_annotation(module_name, qualified_name, "function", source_path)
            if symbol is None:
                symbol = self.database.get_annotation(module_name, qualified_name, "async_function", source_path)
            class_symbol = self.database.get_annotation(module_name, class_name, "class", source_path) if class_name else None
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

    def __enter__(self) -> "SQLiteWorkspaceObserver":
        """Open this observer and begin a linked logger run for the context body."""
        return self.open()

    def __exit__(self, exc_type: object, exc: BaseException | None, traceback: object) -> bool:
        """Close the logger run while allowing exceptions to propagate to the caller."""
        self.close(error=exc)
        return False