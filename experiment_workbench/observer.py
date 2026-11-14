"""Execution telemetry and result observation for the ORE Experiment Workbench.

The observer reports Python execution facts and data already emitted by a component.
It does not infer hidden neural operations or rewrite component results.
"""
from __future__ import annotations

import dataclasses
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError:  # pragma: no cover - the ORE project normally requires NumPy
    np = None


class PythonCallObserver:
    """Collect timed calls into one selected package without changing its code."""

    def __init__(self, package_name: str, max_events: int = 1200) -> None:
        self.package_name = package_name
        self.max_events = max_events
        self.events: list[dict[str, Any]] = []
        self.counts: Counter[str] = Counter()
        self._active: dict[int, dict[str, Any]] = {}
        self._sequence = 0
        self._dropped = 0
        self._previous_profile = None

    def _belongs(self, frame) -> bool:
        module = str(frame.f_globals.get("__name__", ""))
        return module == self.package_name or module.startswith(self.package_name + ".")

    def _profile(self, frame, event, arg):
        previous = self._previous_profile
        if previous is not None:
            previous(frame, event, arg)
        if not self._belongs(frame):
            return
        ident = id(frame)
        if event == "call":
            self._sequence += 1
            code = frame.f_code
            label = f"{frame.f_globals.get('__name__', '')}.{code.co_name}"
            self.counts[label] += 1
            self._active[ident] = {
                "sequence": self._sequence,
                "function": label,
                "module": frame.f_globals.get("__name__", ""),
                "file": Path(code.co_filename).name,
                "line": int(code.co_firstlineno),
                "depth": len(self._active),
                "started_ns": time.perf_counter_ns(),
            }
        elif event == "return":
            started = self._active.pop(ident, None)
            if started is None:
                return
            if len(self.events) >= self.max_events:
                self._dropped += 1
                return
            duration = (time.perf_counter_ns() - started["started_ns"]) / 1_000_000
            self.events.append({
                "sequence": started["sequence"],
                "function": started["function"],
                "module": started["module"],
                "file": started["file"],
                "line": started["line"],
                "depth": started["depth"],
                "duration_ms": round(duration, 4),
                "return_type": type(arg).__name__,
            })

    def __enter__(self):
        self._previous_profile = sys.getprofile()
        sys.setprofile(self._profile)
        return self

    def __exit__(self, exc_type, exc, tb):
        sys.setprofile(self._previous_profile)
        self._previous_profile = None
        self.events.sort(key=lambda item: item["sequence"])
        return False

    def report(self) -> dict[str, Any]:
        grouped: dict[str, dict[str, Any]] = {}
        for event in self.events:
            item = grouped.setdefault(event["function"], {
                "function": event["function"], "calls": 0, "total_ms": 0.0,
                "max_ms": 0.0,
            })
            item["calls"] += 1
            item["total_ms"] += event["duration_ms"]
            item["max_ms"] = max(item["max_ms"], event["duration_ms"])
        top = sorted(grouped.values(), key=lambda x: x["total_ms"], reverse=True)
        for item in top:
            item["total_ms"] = round(item["total_ms"], 4)
            item["max_ms"] = round(item["max_ms"], 4)
        return {
            "package": self.package_name,
            "observed_call_count": sum(self.counts.values()),
            "captured_call_count": len(self.events),
            "dropped_event_count": self._dropped,
            "events": self.events,
            "function_summary": top,
        }


def _field_items(value: Any):
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            try:
                yield field.name, getattr(value, field.name)
            except Exception:
                continue
    elif isinstance(value, dict):
        yield from ((str(key), item) for key, item in value.items())
    elif hasattr(value, "__dict__") and not isinstance(value, type):
        try:
            yield from vars(value).items()
        except Exception:
            return


def _array_summary(value: Any, path: str) -> dict[str, Any] | None:
    if np is None:
        return None
    if not isinstance(value, np.ndarray):
        return None
    if value.size == 0:
        return {"field": path, "shape": list(value.shape), "dtype": str(value.dtype),
                "size": 0, "finite": True, "empty": True}
    if not np.issubdtype(value.dtype, np.number) and value.dtype != np.bool_:
        return {"field": path, "shape": list(value.shape), "dtype": str(value.dtype),
                "size": int(value.size), "numeric": False}
    finite_mask = np.isfinite(value)
    finite = bool(finite_mask.all())
    safe = value[finite_mask]
    summary = {
        "field": path, "shape": list(value.shape), "dtype": str(value.dtype),
        "size": int(value.size), "finite": finite,
        "finite_count": int(safe.size),
    }
    if safe.size:
        summary.update({
            "min": float(np.min(safe)), "max": float(np.max(safe)),
            "mean": float(np.mean(safe)), "std": float(np.std(safe)),
            "l2_norm": float(np.linalg.norm(safe.astype(np.float64).ravel())),
            "nonzero_count": int(np.count_nonzero(safe)),
        })
    return summary


def inspect_output(output: Any, max_arrays: int = 300, max_depth: int = 7) -> dict[str, Any]:
    """Extract public traces, numerical summaries, diagnostics and warnings."""
    traces: list[dict[str, Any]] = []
    arrays: list[dict[str, Any]] = []
    warnings: list[str] = []
    checks: list[dict[str, Any]] = []
    seen: set[int] = set()
    explicit_trace_keys = {
        "operation_trace", "stage_trace", "iteration_trace", "trace",
        "training_trace", "steps_trace", "stages",
    }
    visited_trace_paths: set[str] = set()

    def walk(value: Any, path: str, depth: int):
        if depth > max_depth or id(value) in seen:
            return
        if isinstance(value, (dict, list, tuple)) or dataclasses.is_dataclass(value) or hasattr(value, "__dict__"):
            seen.add(id(value))
        array = _array_summary(value, path)
        if array is not None:
            if len(arrays) < max_arrays:
                arrays.append(array)
            return
        if isinstance(value, dict):
            for key, child in value.items():
                key_text = str(key)
                child_path = f"{path}.{key_text}" if path else key_text
                if key_text in {"valid", "finite_outputs", "converged"} and isinstance(child, (bool, int)):
                    checks.append({"check": child_path, "value": bool(child)})
                if key_text in {"warning", "warnings"}:
                    if isinstance(child, str):
                        warnings.append(child)
                    elif isinstance(child, (list, tuple)):
                        warnings.extend(str(x) for x in child)
                if key_text in explicit_trace_keys and isinstance(child, (list, tuple)):
                    if child_path not in visited_trace_paths:
                        visited_trace_paths.add(child_path)
                        for index, row in enumerate(child):
                            if isinstance(row, dict):
                                traces.append({"source": child_path, "index": index, **row})
                            else:
                                traces.append({"source": child_path, "index": index, "value": repr(row)})
                if key_text == "records" and isinstance(child, dict):
                    for record_name, record in child.items():
                        if isinstance(record, dict):
                            traces.append({"source": child_path, "stage": str(record_name), **record})
                walk(child, child_path, depth + 1)
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]", depth + 1)
        else:
            for key, child in _field_items(value):
                child_path = f"{path}.{key}" if path else str(key)
                if key in explicit_trace_keys and isinstance(child, (list, tuple)):
                    if child_path not in visited_trace_paths:
                        visited_trace_paths.add(child_path)
                        for index, row in enumerate(child):
                            if isinstance(row, dict):
                                traces.append({"source": child_path, "index": index, **row})
                            else:
                                traces.append({"source": child_path, "index": index, "value": repr(row)})
                walk(child, child_path, depth + 1)

    walk(output, "output", 0)
    # Avoid duplicate warning messages from nested diagnostics.
    warnings = list(dict.fromkeys(warnings))
    numeric_failures = [item for item in arrays if not item.get("finite", True)]
    checks.extend([
        {"check": "numeric_arrays_finite", "value": not numeric_failures,
         "details": f"{len(numeric_failures)} non-finite array(s)"}])
    output_type = type(output).__name__
    fields = [name for name, _ in _field_items(output)]
    return {
        "output_type": output_type,
        "output_fields": fields,
        "trace_rows": traces[:2000],
        "trace_count": len(traces),
        "numerical_summaries": arrays,
        "array_count": len(arrays),
        "warnings": warnings,
        "checks": checks,
        "summary": {
            "trace_records": len(traces),
            "numeric_arrays": len(arrays),
            "warning_count": len(warnings),
            "check_count": len(checks),
        },
    }
