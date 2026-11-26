"""Automatic, low-overhead Python execution tracing for ORE repository scripts."""
from __future__ import annotations

import atexit
import importlib.util
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

from .catalog import _module_name, sync_repository
from .database import LoggerDatabase


class RuntimeSession:
    """Capture calls to repository-defined functions and link them to SQLite annotations."""

    def __init__(
        self,
        repository_root: str | Path,
        script_path: str | Path,
        entry_kind: str = "script",
        database_path: str | Path | None = None,
        max_events: int = 200000,
    ) -> None:
        self.root = Path(repository_root).resolve()
        self.script = Path(script_path).resolve()
        self.database_path = Path(database_path or self.root / "ore_logs" / "ore_logger.sqlite3")
        self.database = LoggerDatabase(self.database_path)
        self.catalog_report = sync_repository(self.database, self.root)
        relative = self.script.relative_to(self.root).as_posix()
        module_name = _module_name(relative)
        self.run_id = self.database.create_run(
            script_path=relative,
            module_name=module_name,
            entry_kind=entry_kind,
            working_directory=os.getcwd(),
            process_id=os.getpid(),
            python_version=sys.version.split()[0],
        )
        self._max_events = max(100, int(max_events))
        self._sequence = 0
        self._event_count = 0
        self._lock = threading.RLock()
        self._started = time.perf_counter()
        self._active: dict[tuple[int, int], float] = {}
        self._seen_classes: set[str] = set()
        self._closed = False
        self._previous_profile = None
        self._previous_trace = None
        self._previous_excepthook = sys.excepthook
        self._install()
        self._emit(
            "run_started",
            metadata={"catalog": self.catalog_report, "database": str(self.database_path)},
            source_path=relative,
        )

    def _install(self) -> None:
        self._previous_profile = sys.getprofile()
        self._previous_trace = sys.gettrace()
        if self._previous_profile is None:
            sys.setprofile(self._profile)
            threading.setprofile(self._profile)
        if self._previous_trace is None:
            sys.settrace(self._trace)
            threading.settrace(self._trace)
        sys.excepthook = self._excepthook

    def _source(self, frame: Any) -> tuple[str | None, str | None, str]:
        raw_filename = frame.f_code.co_filename

# Ignore Python internal and synthetic filenames.
        if not raw_filename or (
            raw_filename.startswith("<") and raw_filename.endswith(">")
        ):
            return None, "", ""

        try:
            filename = Path(raw_filename).resolve()
        except (OSError, ValueError, RuntimeError):
            return None, "", ""
        try:
            relative = filename.relative_to(self.root).as_posix()
        except ValueError:
            return None, None, ""
        if relative.startswith("ore_logger/") or relative == "sitecustomize.py":
            return None, None, relative
        module_name = _module_name(relative)
        qualname = getattr(frame.f_code, "co_qualname", frame.f_code.co_name)
        return relative, module_name, qualname

    def _lookup_symbol(self, source_path: str, module: str, qualname: str, kind: str = "function"):
        result = self.database.get_annotation(module, qualname, kind, source_path)
        if result is None and kind == "function":
            result = self.database.get_annotation(module, qualname, "async_function", source_path)
        return result

    def _emit(
        self,
        kind: str,
        symbol_id: str | None = None,
        class_symbol_id: str | None = None,
        source_path: str | None = None,
        line: int | None = None,
        duration_ms: float | None = None,
        exception_type: str | None = None,
        exception_message: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            if self._closed or self._event_count >= self._max_events:
                return
            self._sequence += 1
            self._event_count += 1
            self.database.record_event(
                run_id=self.run_id,
                sequence_no=self._sequence,
                event_kind=kind,
                symbol_id=symbol_id,
                class_symbol_id=class_symbol_id,
                source_path=source_path,
                line_number=line,
                elapsed_ms=round((time.perf_counter() - self._started) * 1000, 3),
                duration_ms=duration_ms,
                thread_id=threading.get_ident(),
                thread_name=threading.current_thread().name,
                exception_type=exception_type,
                exception_message=(exception_message or "")[:500] or None,
                metadata=metadata or {},
            )

    def _profile(self, frame: Any, event: str, arg: Any) -> None:
        if self._previous_profile is not None:
            try:
                self._previous_profile(frame, event, arg)
            except Exception:
                pass
        if self._closed:
            return
        source_path, module, qualname = self._source(frame)
        if source_path is None or module is None:
            return
        key = (threading.get_ident(), id(frame))
        if event == "call":
            symbol = self._lookup_symbol(source_path, module, qualname)
            if symbol is None:
                return
            self._active[key] = time.perf_counter()
            class_symbol = None
            class_name = qualname.split(".<locals>.", 1)[0].split(".", 1)[0]
            if "." in qualname and class_name:
                class_symbol = self._lookup_symbol(source_path, module, class_name, "class")
                if class_symbol and class_name not in self._seen_classes:
                    self._seen_classes.add(class_name)
                    self._emit(
                        "class_enter", symbol_id=class_symbol["symbol_id"],
                        source_path=source_path, line=frame.f_code.co_firstlineno,
                    )
            self._emit(
                "call", symbol_id=symbol["symbol_id"],
                class_symbol_id=class_symbol["symbol_id"] if class_symbol else None,
                source_path=source_path, line=frame.f_code.co_firstlineno,
            )
        elif event == "return":
            start = self._active.pop(key, None)
            if start is None:
                return
            symbol = self._lookup_symbol(source_path, module, qualname)
            if symbol is not None:
                self._emit(
                    "return", symbol_id=symbol["symbol_id"],
                    source_path=source_path, line=frame.f_lineno,
                    duration_ms=round((time.perf_counter() - start) * 1000, 4),
                )

    def _trace(self, frame: Any, event: str, arg: Any):
        if self._closed or event != "exception":
            return self._trace
        source_path, module, qualname = self._source(frame)
        if source_path is None or module is None:
            return self._trace
        try:
            exc_type, exc_value, traceback_obj = arg
            # Python repeats an exception trace event as it unwinds. Record its
            # original raising frame once, not once for every caller.
            if traceback_obj is not None and traceback_obj.tb_frame is not frame:
                return self._trace
            symbol = self._lookup_symbol(source_path, module, qualname)
            if symbol is not None:
                self._emit(
                    "exception", symbol_id=symbol["symbol_id"],
                    source_path=source_path, line=frame.f_lineno,
                    exception_type=getattr(exc_type, "__name__", str(exc_type)),
                    exception_message=str(exc_value),
                )
        except Exception:
            pass
        return self._trace

    def _excepthook(self, exc_type: Any, exc_value: Any, traceback_obj: Any) -> None:
        self.close("failed", f"{getattr(exc_type, '__name__', exc_type)}: {exc_value}")
        if self._previous_excepthook is not None:
            self._previous_excepthook(exc_type, exc_value, traceback_obj)

    def close(self, status: str = "completed", error: str | None = None) -> None:
        with self._lock:
            if self._closed:
                return
            duration = round((time.perf_counter() - self._started) * 1000, 3)
            self._emit(
                "run_finished",
                source_path=self.script.relative_to(self.root).as_posix(),
                metadata={"status": status, "duration_ms": duration},
            )
            self._closed = True
            try:
                if sys.getprofile() == self._profile:
                    sys.setprofile(self._previous_profile)
                if sys.gettrace() == self._trace:
                    sys.settrace(self._previous_trace)
                threading.setprofile(self._previous_profile)
                threading.settrace(self._previous_trace)
                if sys.excepthook == self._excepthook:
                    sys.excepthook = self._previous_excepthook
            except Exception:
                pass
            self.database.finish_run(self.run_id, status, duration, error)


def _candidate_script(arguments: list[str], root: Path) -> tuple[Path | None, str]:
    # The module target is present in sys.orig_argv even when sitecustomize runs
    # before runpy has replaced sys.argv[0] with the module's file path.
    combined = list(arguments) + list(getattr(sys, "orig_argv", []))
    if "-m" in combined:
        try:
            module_index = combined.index("-m")
            module_target = combined[module_index + 1]
            spec = importlib.util.find_spec(module_target)
            if spec is not None:
                candidate = None
                if spec.submodule_search_locations:
                    package_dir = Path(next(iter(spec.submodule_search_locations))).resolve()
                    package_main = package_dir / "__main__.py"
                    candidate = package_main if package_main.is_file() else None
                if candidate is None and spec.origin:
                    candidate = Path(spec.origin).resolve()
                if candidate is not None:
                    relative = candidate.relative_to(root)
                    if relative.parts and relative.parts[0] != "ore_logger" and candidate.is_file():
                        return candidate, "module"
        except (IndexError, ImportError, ModuleNotFoundError, ValueError, OSError):
            pass

    for value in combined:
        if not value or value.startswith("-"):
            continue
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        try:
            resolved = candidate.resolve()
            relative = resolved.relative_to(root)
        except (OSError, ValueError):
            continue
        if resolved.suffix.lower() != ".py" or not resolved.is_file():
            continue
        if relative.parts and relative.parts[0] == "ore_logger":
            continue
        return resolved, "module" if "__main__.py" in relative.parts else "script"
    return None, "script"


def install_for_current_script() -> RuntimeSession | None:
    """Install automatically when the launched .py entry point belongs to this repository."""
    if (
    os.getenv("ORE_AUTO_LOG", "1").strip().lower()
    in {"0", "false", "no", "off"}
    or "pytest" in sys.modules
    or any("pytest" in str(arg).lower() for arg in sys.argv)
):
        return None
    root = Path(__file__).resolve().parents[1]
    script, entry_kind = _candidate_script(sys.argv, root)
    if script is None:
        return None

    
    # Streamlit's server process is long-lived and reruns app.py repeatedly.
    # Individual component runs are recorded by the Workbench adapter instead.
    if script.relative_to(root).as_posix() == "experiment_workbench/app.py":
        return None
    try:
        session = RuntimeSession(
            root, script, entry_kind=entry_kind,
            database_path=os.getenv("ORE_LOGGER_DB") or root / "ore_logs" / "ore_logger.sqlite3",
            max_events=int(os.getenv("ORE_LOG_MAX_EVENTS", "200000")),
        )
        atexit.register(session.close)
        return session
    except Exception:
        # Optional diagnostics must not prevent the user's script from running.
        return None


_ACTIVE_SESSION = install_for_current_script()
