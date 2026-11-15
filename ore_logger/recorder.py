"""SQLite-backed recorder shared by automatic script tracing and the Workbench."""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from .catalog import sync_repository
from .database import LoggerDatabase


def _compact_metadata(value: Any, depth: int = 0) -> Any:
    if depth >= 5:
        return "..."
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:240]
    if isinstance(value, Mapping):
        return {str(key)[:80]: _compact_metadata(item, depth + 1)
                for key, item in list(value.items())[:30]}
    if isinstance(value, (list, tuple)):
        items = [_compact_metadata(item, depth + 1) for item in value[:12]]
        if len(value) > 12:
            items.append(f"... {len(value) - 12} more value(s)")
        return items
    shape = getattr(value, "shape", None)
    if shape is not None:
        return {"array_shape": list(shape), "dtype": str(getattr(value, "dtype", "unknown"))}
    return repr(value)[:240]


class SQLiteRunRecorder:
    """Persist execution events as references to reusable annotation records."""

    def __init__(
        self,
        database_path: str | Path,
        repository_root: str | Path,
        run_id: str,
        component: str,
        callable_name: str,
        started_at_utc: str,
        on_event: Callable[[dict[str, Any]], None] | None = None,
        entry_kind: str = "workbench_component",
    ) -> None:
        self.database = LoggerDatabase(database_path)
        self.repository_root = Path(repository_root).resolve()
        self.run_id = run_id
        self.component = component
        self.callable_name = callable_name
        self.on_event = on_event
        self._clock = time.perf_counter()
        self._sequence = 0
        self._event_count = 0
        self._closed = False
        self._operation_ids: set[tuple[str, str]] = set()
        self._errors: list[str] = []
        self.catalog_report = sync_repository(self.database, self.repository_root)
        script_path = f"workbench/{component}"
        module_name = component
        actual_id = self.database.create_run(
            script_path=script_path,
            module_name=module_name,
            entry_kind=entry_kind,
            working_directory=str(self.repository_root),
            process_id=None,
            python_version=None,
            run_id=run_id,
            started_at_utc=started_at_utc,
        )
        if actual_id != run_id:
            raise RuntimeError("SQLite run identifier mismatch.")
        self._emit(
            "run_started",
            explanation=f"Started {callable_name} in component {component}.",
            metadata={"catalog": self.catalog_report, "input_values_logged": False},
        )

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "database_path": str(self.database.path),
            "run_id": self.run_id,
            "event_count": self._event_count,
            "annotation_catalog": self.catalog_report,
            "logger_errors": list(self._errors),
        }

    def _emit(
        self,
        kind: str,
        symbol: Mapping[str, Any] | None = None,
        class_symbol: Mapping[str, Any] | None = None,
        source_path: str | None = None,
        line: int | None = None,
        duration_ms: float | None = None,
        exception_type: str | None = None,
        exception_message: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        explanation: str | None = None,
        display_fields: Mapping[str, Any] | None = None,
    ) -> None:
        if self._closed:
            return
        self._sequence += 1
        elapsed = round((time.perf_counter() - self._clock) * 1000, 3)
        try:
            row = self.database.record_event(
                run_id=self.run_id,
                sequence_no=self._sequence,
                event_kind=kind,
                symbol_id=symbol.get("symbol_id") if symbol else None,
                class_symbol_id=class_symbol.get("symbol_id") if class_symbol else None,
                source_path=source_path,
                line_number=line,
                elapsed_ms=elapsed,
                duration_ms=duration_ms,
                exception_type=exception_type,
                exception_message=exception_message,
                metadata=_compact_metadata(dict(metadata or {})),
            )
            self._event_count += 1
            symbol_data = row.get("symbol") or {}
            class_data = row.get("class_symbol") or {}
            event = {
                "sequence": self._sequence,
                "run_id": self.run_id,
                "component": self.component,
                "callable": self.callable_name,
                "elapsed_ms": elapsed,
                "kind": kind,
                "function": symbol_data.get("qualified_name", ""),
                "module": symbol_data.get("module_name", ""),
                "annotation": symbol_data.get("annotation", ""),
                "class_annotation": class_data.get("annotation", ""),
                "explanation": explanation or symbol_data.get("annotation", ""),
                "source": source_path,
                "file": source_path,
                "line": line,
                "duration_ms": duration_ms,
                "exception_type": exception_type,
                "exception_message": exception_message,
                "details": _compact_metadata(dict(metadata or {})),
            }
            if display_fields:
                event.update(dict(display_fields))
            if self.on_event is not None:
                try:
                    self.on_event(event)
                except Exception:
                    pass
        except Exception as exc:
            self._errors.append(f"{type(exc).__name__}: {exc}")

    def _symbol_for_python_event(self, row: Mapping[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        module = str(row.get("module") or "")
        qualname = str(row.get("qualname") or row.get("short_name") or "")
        symbol = self.database.get_annotation(module, qualname, "function")
        if symbol is None:
            symbol = self.database.get_annotation(module, qualname, "async_function")
        class_symbol = None
        if ".<locals>." in qualname:
            outer, nested = qualname.split(".<locals>.", 1)
            if "." in outer:
                class_name = outer.split(".", 1)[0]
            elif "." in nested:
                class_name = nested.split(".", 1)[0]
            else:
                class_name = None
        else:
            class_name = qualname.rsplit(".", 1)[0] if "." in qualname else None
        if class_name:
            class_symbol = self.database.get_annotation(module, class_name, "class")
        return symbol, class_symbol

    def log_python_event(self, event: Mapping[str, Any]) -> None:
        row = dict(event)
        kind = str(row.get("kind", "python_event"))
        symbol, class_symbol = self._symbol_for_python_event(row)
        label = str(row.get("qualname") or row.get("short_name") or row.get("function") or "Python function")
        annotation = symbol.get("annotation") if symbol else f"Python is executing {label.replace('_', ' ')}."
        explanation = {
            "call": f"Entered {label}. {annotation}",
            "return": f"Finished {label}. {annotation}",
            "exception": f"{row.get('exception_type', 'An exception')} occurred in {label}. {annotation}",
            "line": f"Executing line {row.get('line')} in {label}. {annotation}",
        }.get(kind, annotation)
        display_fields = {
            "short_name": row.get("short_name"),
            "active_stack": row.get("active_stack", []),
        }
        if row.get("source") is not None:
            display_fields["source"] = row.get("source")
        self._emit(
            kind, symbol=symbol, class_symbol=class_symbol,
            source_path=(symbol.get("source_path") if symbol else None)
                or row.get("source_path") or row.get("file"),
            line=row.get("line"),
            duration_ms=row.get("duration_ms"),
            exception_type=row.get("exception_type"),
            exception_message=row.get("exception_message"),
            metadata={
                "depth": row.get("depth"),
                "return_type": row.get("return_type"),
                "module": row.get("module"),
                "qualname": row.get("qualname"),
                "active_stack": row.get("active_stack", []),
                "source_text": row.get("source"),
            },
            explanation=explanation,
            display_fields=display_fields,
        )

    def _register_synthetic(self, component: str, name: str, kind: str, annotation: str) -> dict[str, Any]:
        raw = f"{component}|{kind}|{name}".encode("utf-8")
        symbol_id = hashlib.sha256(raw).hexdigest()[:24]
        source_path = f"{component}::<{kind}-catalog>"
        self.database.register_annotation(
            symbol_id=symbol_id,
            module_name=component,
            qualified_name=name,
            symbol_kind=kind,
            source_path=source_path,
            annotation=annotation,
            annotation_source="operation_catalog",
            source_hash=hashlib.sha256(annotation.encode("utf-8")).hexdigest(),
        )
        return self.database.get_annotation_by_id(symbol_id) or {
            "symbol_id": symbol_id, "annotation": annotation,
            "qualified_name": name, "module_name": component,
        }

    def log_operation(self, operation_event: Mapping[str, Any]) -> None:
        row = dict(operation_event)
        name = str(row.get("operation", row.get("name", "unnamed_operation")))
        step = row.get("step", row.get("number"))
        status = row.get("status", "ok")
        explain = row.get("explanation")
        if not explain:
            explain = f"ORE is executing operation {name.replace('_', ' ')}."
        symbol = self._register_synthetic(self.component, name, "operation", explain)
        self._operation_ids.add((str(step), name))
        details = {
            key: value for key, value in row.items()
            if key not in {"kind", "step", "number", "operation", "name", "status", "explanation"}
        }
        self._emit(
            "operation", symbol=symbol, source_path=self.component,
            metadata={"step": step, "operation": name, "status": status, **details},
            explanation=explain,
            display_fields={
                "step": step, "operation": name, "title": name.replace("_", " "),
                "phase": row.get("phase", ""), "status": status,
                "status_explanation": row.get("status_explanation", ""),
                "phase_explanation": row.get("phase_explanation", ""),
            },
        )

    def log_component_traces(self, rows: list[Mapping[str, Any]]) -> None:
        for index, source_row in enumerate(rows):
            row = dict(source_row)
            name = str(row.get("operation", row.get("name", row.get("stage", row.get("step_name", f"trace_{index + 1}")))))
            step = row.get("step", row.get("index", index + 1))
            if (str(step), name) in self._operation_ids:
                continue
            annotation = str(row.get("explanation") or f"ORE completed the component trace: {name.replace('_', ' ')}.")
            symbol = self._register_synthetic(self.component, name, "component_trace", annotation)
            self._emit(
                "component_trace", symbol=symbol,
                source_path=str(row.get("source", self.component)),
                line=row.get("line"),
                metadata={"step": step, **dict(row)},
                explanation=annotation,
                display_fields={"step": step, "operation": name, "title": name.replace("_", " ")},
            )

    def finish(
        self, status: str, elapsed_ms: float | None,
        output_type: str | None = None, error: str | None = None,
    ) -> dict[str, Any]:
        if not self._closed:
            self._emit(
                "run_finished",
                metadata={"status": status, "duration_ms": elapsed_ms, "output_type": output_type},
                explanation=(
                    f"Execution {status}. Output type: {output_type or 'none'}."
                    if not error else f"Execution failed: {error[:500]}"
                ),
            )
            self.database.finish_run(self.run_id, status, elapsed_ms, error)
            self._closed = True
        return self.metadata
