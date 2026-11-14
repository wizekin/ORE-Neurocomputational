"""Execution telemetry and result observation for the ORE Experiment Workbench.

The observer reports Python execution facts and data already emitted by a component.
It does not infer hidden neural operations or rewrite component results.
"""
from __future__ import annotations

import dataclasses
import linecache
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

try:
    import numpy as np
except ImportError:  # pragma: no cover - the ORE project normally requires NumPy
    np = None


_FRIENDLY_FUNCTIONS = {
    "process": "Coordinate this component's main processing steps.",
    "fit": "Learn or prepare the internal model from supplied examples.",
    "recognize": "Use a prepared model to recognize an input sequence.",
    "as_grayscale_float": "Convert the image into a grayscale numeric intensity array.",
    "photoreceptor_adaptation": "Adapt image responses to local brightness, like retinal light adaptation.",
    "retinal_on_off": "Separate brightness increases from brightness decreases.",
    "center_surround": "Compare each location with its surroundings to emphasize local contrast.",
    "lgn_gain_control": "Adjust signal strength to keep visual responses in a useful range.",
    "simple_cell_responses": "Measure oriented visual patterns such as edges and bars.",
    "complex_cell_energy": "Combine oriented responses into a more position-tolerant signal.",
    "divisive_normalization": "Scale activity relative to neighboring feature activity.",
    "spatial_pooling": "Summarize local visual features into a compact spatial representation.",
    "adapt_n1_result": "Check and adapt the upstream N1 result for the N2 input contract.",
    "contextual_prior": "Estimate likely alternatives using neighboring positions and learned patterns.",
    "weighted_belief_update": "Combine sensory evidence with contextual predictions.",
    "precision_from_prediction": "Estimate how strongly contextual predictions should influence the current belief.",
    "precision_from_reliability": "Estimate sensory confidence from input reliability and attention.",
    "classwise_prediction_error": "Measure the mismatch between the current belief and its prediction.",
    "feature_prediction_residual": "Measure the remaining mismatch in the underlying feature representation.",
    "normalize_rows": "Normalize each row into a comparable probability-like distribution.",
}

def explain_event(event: dict[str, Any]) -> str:
    kind = event.get("kind")
    name = str(event.get("short_name") or event.get("function", "operation"))
    if kind == "call":
        return _FRIENDLY_FUNCTIONS.get(name, f"Started {name.replace('_', ' ')}.")
    if kind == "return":
        duration = event.get("duration_ms")
        timing = f" in {duration:.2f} ms" if isinstance(duration, (int, float)) else ""
        return f"Finished {name.replace('_', ' ')} and returned {event.get('return_type', 'a value')}{timing}."
    if kind == "exception":
        return f"{event.get('exception_type', 'An error')} was raised inside {name.replace('_', ' ')}."
    if kind == "line":
        return f"Executing line {event.get('line')} in {name.replace('_', ' ')}."
    return f"Observed {name.replace('_', ' ')}."


class PythonCallObserver:
    """Observe package-local Python calls and optionally stream events in real time."""

    def __init__(self, package_name: str, max_events: int = 2400,
                 on_event: Callable[[dict[str, Any]], None] | None = None,
                 trace_lines: bool = False) -> None:
        self.package_name = package_name
        self.max_events = max_events
        self.on_event = on_event
        self.trace_lines = trace_lines
        self.events: list[dict[str, Any]] = []       # completed calls, retained for compatibility
        self.timeline: list[dict[str, Any]] = []     # call/return/exception/line events
        self.counts: Counter[str] = Counter()
        self._active: dict[int, dict[str, Any]] = {}
        self._sequence = 0
        self._dropped = 0
        self._previous_profile = None
        self._previous_trace = None
        self._started_ns = 0

    def _belongs(self, frame) -> bool:
        module = str(frame.f_globals.get("__name__", ""))
        return module == self.package_name or module.startswith(self.package_name + ".")

    def _stack(self) -> list[dict[str, Any]]:
        active = sorted(self._active.values(), key=lambda item: item["sequence"])
        return [{"function": x["function"], "short_name": x["short_name"],
                 "file": x["file"], "line": x["line"]} for x in active[-30:]]

    def _emit(self, event: dict[str, Any]) -> None:
        if len(self.timeline) >= self.max_events:
            self._dropped += 1
            return
        event["elapsed_ms"] = round((time.perf_counter_ns() - self._started_ns) / 1_000_000, 3)
        event["active_stack"] = self._stack()
        event["explanation"] = explain_event(event)
        self.timeline.append(event)
        if self.on_event is not None:
            try:
                self.on_event(dict(event))
            except Exception:
                pass  # Instrumentation must never interrupt the component.

    def _profile(self, frame, event, arg):
        if self._previous_profile is not None:
            self._previous_profile(frame, event, arg)
        if not self._belongs(frame):
            return
        ident = id(frame)
        if event == "call":
            self._sequence += 1
            code = frame.f_code
            module = frame.f_globals.get("__name__", "")
            short_name = code.co_name
            label = f"{module}.{short_name}"
            self.counts[label] += 1
            record = {"sequence": self._sequence, "function": label,
                      "short_name": short_name, "module": module,
                      "file": Path(code.co_filename).name,
                      "line": int(code.co_firstlineno), "depth": len(self._active),
                      "started_ns": time.perf_counter_ns()}
            self._active[ident] = record
            self._emit({k: record[k] for k in ("sequence", "function", "short_name", "module", "file", "line", "depth")} | {"kind": "call"})
        elif event == "exception":
            exc_type, exc_value, _ = arg
            current = self._active.get(ident)
            if current:
                self._emit({"sequence": current["sequence"], "kind": "exception",
                            "function": current["function"], "short_name": current["short_name"],
                            "module": current["module"], "file": current["file"],
                            "line": int(frame.f_lineno), "depth": current["depth"],
                            "exception_type": getattr(exc_type, "__name__", str(exc_type)),
                            "exception_message": str(exc_value)[:500]})
        elif event == "return":
            record = self._active.pop(ident, None)
            if record is None:
                return
            duration = (time.perf_counter_ns() - record["started_ns"]) / 1_000_000
            completed = {"sequence": record["sequence"], "kind": "return",
                         "function": record["function"], "short_name": record["short_name"],
                         "module": record["module"], "file": record["file"],
                         "line": record["line"], "depth": record["depth"],
                         "duration_ms": round(duration, 4), "return_type": type(arg).__name__}
            if len(self.events) < self.max_events:
                self.events.append(completed)
            self._emit(dict(completed))

    def _trace(self, frame, event, arg):
        if not self._belongs(frame):
            return None
        if event == "line" and self.trace_lines:
            current = self._active.get(id(frame))
            if current:
                line_number = int(frame.f_lineno)
                source = linecache.getline(frame.f_code.co_filename, line_number).strip()
                self._emit({"sequence": current["sequence"], "kind": "line",
                            "function": current["function"], "short_name": current["short_name"],
                            "module": current["module"], "file": current["file"],
                            "line": line_number, "depth": current["depth"],
                            "source": source[:260]})
        elif event == "exception":
            exc_type, exc_value, _ = arg
            current = self._active.get(id(frame))
            if current:
                self._emit({"sequence": current["sequence"], "kind": "exception",
                            "function": current["function"], "short_name": current["short_name"],
                            "module": current["module"], "file": current["file"],
                            "line": int(frame.f_lineno), "depth": current["depth"],
                            "exception_type": getattr(exc_type, "__name__", str(exc_type)),
                            "exception_message": str(exc_value)[:500]})
        return self._trace

    def __enter__(self):
        self._started_ns = time.perf_counter_ns()
        self._previous_profile = sys.getprofile()
        self._previous_trace = sys.gettrace()
        sys.setprofile(self._profile)
        # Trace hooks are needed for exception events even when line events are off.
        sys.settrace(self._trace)
        return self

    def __exit__(self, exc_type, exc, tb):
        sys.setprofile(self._previous_profile)
        sys.settrace(self._previous_trace)
        self._previous_profile = None
        self._previous_trace = None
        self.events.sort(key=lambda item: item["sequence"])
        return False

    def report(self) -> dict[str, Any]:
        grouped: dict[str, dict[str, Any]] = {}
        for event in self.events:
            item = grouped.setdefault(event["function"], {
                "function": event["function"], "calls": 0, "total_ms": 0.0, "max_ms": 0.0})
            item["calls"] += 1
            item["total_ms"] += event["duration_ms"]
            item["max_ms"] = max(item["max_ms"], event["duration_ms"])
        top = sorted(grouped.values(), key=lambda item: item["total_ms"], reverse=True)
        for item in top:
            item["total_ms"] = round(item["total_ms"], 4)
            item["max_ms"] = round(item["max_ms"], 4)
        return {"package": self.package_name, "observed_call_count": sum(self.counts.values()),
                "captured_call_count": len(self.events), "timeline_event_count": len(self.timeline),
                "dropped_event_count": self._dropped, "events": self.events,
                "timeline": self.timeline, "function_summary": top}

def _trace_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        try:
            return dataclasses.asdict(value)
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return dict(vars(value))
        except Exception:
            pass
    return {"value": repr(value)}


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
                            row_data = _trace_mapping(row)
                            traces.append({"source": child_path, "index": index, **row_data})
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
                            row_data = _trace_mapping(row)
                            traces.append({"source": child_path, "index": index, **row_data})
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
