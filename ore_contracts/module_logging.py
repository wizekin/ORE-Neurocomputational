"""Register ORE communication definitions and link registry events to the shared SQLite logger."""
from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Mapping



def register_module_annotations(
    database_path: str | Path | None = None,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Index every ore_contracts and ore_usg module, class, function, and method in ore_logger.annotations."""
    from ore_logger.catalog import sync_repository
    from ore_logger.database import LoggerDatabase

    root = Path(repository_root).expanduser().resolve() if repository_root else Path(__file__).resolve().parents[1]
    target = (
        Path(database_path).expanduser().resolve()
        if database_path else root / "ore_logs" / "ore_logger.sqlite3"
    )
    database = LoggerDatabase(target)
    result = sync_repository(database, root, include_paths={"ore_contracts", "ore_usg"})
    return {
        **result, "database_path": str(target), "scope": ("ore_contracts", "ore_usg"),
        "annotation_table": "annotations",
    }


class SQLiteModuleEventLogger:
    """Persist method-linked registry events using ore_logger's existing runs/events/annotations tables."""

    def __init__(
        self,
        database_path: str | Path | None = None,
        repository_root: str | Path | None = None,
        display_label: str = "Universal module registry",
    ) -> None:
        """Configure the existing ORE SQLite database location and one run label."""
        self.root = Path(repository_root).expanduser().resolve() if repository_root else Path(__file__).resolve().parents[1]
        self.path = (
            Path(database_path).expanduser().resolve()
            if database_path else self.root / "ore_logs" / "ore_logger.sqlite3"
        )
        self.display_label = display_label
        self.database: Any | None = None
        self.run_id: str | None = None
        self.sequence_no = 0
        self._started = 0.0

    def __enter__(self) -> "SQLiteModuleEventLogger":
        """Synchronize AST annotations and open one logger run before recording registry events."""
        from ore_logger.catalog import sync_repository
        from ore_logger.database import LoggerDatabase

        self.database = LoggerDatabase(self.path)
        sync_repository(self.database, self.root, include_paths={"ore_contracts", "ore_usg"})
        self.run_id = self.database.create_run(
            script_path="ore_contracts/module_registry",
            module_name="ore_contracts.module_registry",
            entry_kind="module_registry",
            working_directory=str(self.root),
        )
        self._started = time.perf_counter()
        return self

    def record(
        self,
        method_name: str,
        metadata: Mapping[str, Any],
        *,
        class_name: str = "ModuleRegistry",
        module_name: str = "ore_contracts.module_registry",
    ) -> dict[str, Any]:
        """Write one event linked to the exact method annotation and containing class annotation."""
        if self.database is None or self.run_id is None:
            raise RuntimeError("SQLiteModuleEventLogger must be used as a context manager")
        source_path = module_name.replace(".", "/") + ".py"
        qualified_name = f"{class_name}.{method_name}" if class_name else method_name
        symbol = self.database.get_annotation(module_name, qualified_name, "function", source_path)
        class_symbol = (
            self.database.get_annotation(module_name, class_name, "class", source_path)
            if class_name else None
        )
        self.sequence_no += 1
        return self.database.record_event(
            run_id=self.run_id,
            sequence_no=self.sequence_no,
            event_kind="module_registry",
            symbol_id=symbol["symbol_id"] if symbol else None,
            class_symbol_id=class_symbol["symbol_id"] if class_symbol else None,
            source_path=source_path,
            line_number=symbol.get("line_start") if symbol else None,
            elapsed_ms=round((time.perf_counter() - self._started) * 1000.0, 3),
            metadata={
                **dict(metadata),
                "annotated_method": qualified_name,
                "method_annotation_found": symbol is not None,
                "class_annotation_found": class_symbol is not None,
            },
        )

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        """Finish the SQLite run and preserve failure details without suppressing exceptions."""
        if self.database is not None and self.run_id is not None:
            status = "failed" if exc is not None else "completed"
            duration = round((time.perf_counter() - self._started) * 1000.0, 3)
            self.database.finish_run(
                self.run_id, status, duration_ms=duration,
                error_summary=(f"{type(exc).__name__}: {exc}" if exc is not None else None),
            )
        return False


def record_registry_event(
    method_name: str,
    metadata: Mapping[str, Any],
    *,
    database_path: str | Path | None = None,
    repository_root: str | Path | None = None,
    class_name: str = "ModuleRegistry",
    module_name: str = "ore_contracts.module_registry",
) -> dict[str, Any]:
    """Persist one isolated registry event with its method/class annotation references."""
    with SQLiteModuleEventLogger(database_path, repository_root) as logger:
        return logger.record(
            method_name, metadata, class_name=class_name, module_name=module_name
        )
