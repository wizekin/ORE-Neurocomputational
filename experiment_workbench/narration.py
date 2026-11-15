"""Plain-language narration and durable logging for ORE execution."""
from __future__ import annotations

import json
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

try:
    from ore_n9.pipeline import N9_ALGORITHM_STEPS
except Exception:  # Keep the logger usable for isolated component experiments.
    N9_ALGORITHM_STEPS = ()

N9_PHASES = (
    (1, 12, "A — Input, provenance and state", "ORE checks incoming evidence, identifies its format, preserves source identity, and prepares processing state and module connections."),
    (13, 24, "B — Visual evidence and organization", "ORE extracts or reuses visual signals, organizes local features and contours, and packages available visual evidence. Character segmentation is supplied by the caller; this controller does not invent character crops."),
    (25, 36, "C — Grapheme identity and sequence", "ORE forms grapheme alternatives, records positions, compares features, uses transition memory when available, and builds candidate sequences and word spans."),
    (37, 48, "D — Attention and precision", "ORE estimates uncertainty and prediction error, allocates attention across positions, adjusts sensory and prior precision, and prepares these signals for later processing."),
    (49, 60, "E — Sequence, lexical and phonological access", "ORE combines forward and backward sequence expectations with visual evidence, then requests lexical, morphological and pronunciation information where an adapter is available."),
    (61, 72, "F — Syntax and semantic composition", "ORE looks for grammatical structure and composes supported roles, phrases, propositions and typed semantic relations. Missing adapters remain explicitly unavailable."),
    (73, 84, "G — Discourse and situation context", "ORE gathers supplied discourse, semantic-memory and episodic context, then requests intent, coherence, temporal/causal links and uncertainty from the context adapter."),
    (85, 96, "H — Bounded feedback and conflict resolution", "ORE sends higher-level predictions back toward lexical and grapheme alternatives, measures disagreement, performs bounded belief revision and preserves separate evidence channels."),
    (97, 108, "I — Integration, decision and output", "ORE combines available representations, evaluates consistency and confidence, chooses a candidate or abstains, constructs the result and performs integrity checks."),
    (109, 120, "J — Learning governance and scientific acceptance", "ORE freezes the inference result before considering learning. Any model update requires explicit authorization and must be proposed on a detached child model; validation requirements are recorded separately."),
)

_OPERATION_ANNOTATIONS_PATH = (
    Path(__file__).resolve().parents[1] / "ore_logger" / "operation_annotations.json"
)
with _OPERATION_ANNOTATIONS_PATH.open("r", encoding="utf-8") as annotation_file:
    N9_OPERATION_EXPLANATIONS = json.load(annotation_file).get("n9_operations", {})

if N9_ALGORITHM_STEPS and set(N9_OPERATION_EXPLANATIONS) != set(N9_ALGORITHM_STEPS):
    raise RuntimeError("N9 operation annotations do not match the operation registry.")


def _plain_name(value: Any) -> str:
    text = str(value or "operation").strip()
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    return text.replace("_", " ").replace(".", " ")


def _safe_detail(value: Any, depth: int = 0) -> Any:
    """Bound event details; never expand full image or array payloads."""
    if depth >= 5:
        return "..."
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:240]
    if isinstance(value, Mapping):
        return {str(key)[:80]: _safe_detail(item, depth + 1)
                for key, item in list(value.items())[:30]}
    if isinstance(value, (list, tuple)):
        items = [_safe_detail(item, depth + 1) for item in value[:12]]
        if len(value) > 12:
            items.append(f"... {len(value) - 12} more value(s)")
        return items
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None:
        return {"array_shape": list(shape), "dtype": str(dtype or "unknown")}
    return repr(value)[:240]


def _phase_for_step(step: Any) -> tuple[str, str]:
    try:
        number = int(step)
    except (TypeError, ValueError):
        return ("Component operation", "A component-authored diagnostic step.")
    for start, end, title, description in N9_PHASES:
        if start <= number <= end:
            return title, description
    return ("Component operation", "A component-authored diagnostic step.")


def explain_operation(operation: Any, step: Any = None, status: Any = "ok") -> dict[str, str]:
    name = str(operation or "unnamed_operation")
    if name in N9_OPERATION_EXPLANATIONS:
        phase, phase_description = _phase_for_step(step)
        explanation = N9_OPERATION_EXPLANATIONS[name]
    else:
        phase = "Component operation"
        phase_description = "A component-authored diagnostic step; this is not an N9 phase label."
        explanation = f"ORE is carrying out the component step: {_plain_name(name)}."
    status_text = str(status or "ok").lower()
    status_explanations = {
        "ok": "Completed.",
        "returned": "Completed and the result was packaged.",
        "not_applicable": "This input did not contain the evidence needed for this operation, so the step was correctly marked not applicable.",
        "not_configured": "The required optional adapter or model is not configured. ORE preserves that absence instead of inventing an output.",
        "fallback": "The configured component was unavailable, so ORE used the explicit fallback described in the operation details.",
        "abstention_fallback": "The required recognition evidence was unavailable. ORE used an unknown-only fallback and avoids claiming a recognized symbol.",
        "abstain": "The evidence did not meet the configured confidence requirement, so ORE abstains.",
        "ambiguous": "More than one interpretation remains too close to select confidently.",
        "not_authorized": "Learning was not authorized; the inference result remains separate from model updates.",
        "skipped_without_explicit_authorization": "Skipped because explicit learning authorization was not provided.",
        "authorized_but_N5_unavailable": "Authorization was present, but the N5 learning adapter was not configured.",
        "error": "The operation failed. Inspect the exception and traceback for the cause.",
    }
    return {
        "title": _plain_name(name), "phase": phase,
        "phase_explanation": phase_description, "explanation": explanation,
        "status_explanation": status_explanations.get(status_text, f"Recorded status: {status_text}."),
    }


class OREExecutionLogger:
    """Compatibility façade that stores Workbench events in the shared SQLite database."""

    def __init__(self, output_dir: str | Path, run_id: str, component: str,
                 callable_name: str, started_at: str,
                 on_event: Callable[[dict[str, Any]], None] | None = None) -> None:
        from ore_logger.recorder import SQLiteRunRecorder

        repository_root = Path(__file__).resolve().parents[1]
        database_path = Path(output_dir).expanduser() / "ore_logger.sqlite3"
        self._recorder = SQLiteRunRecorder(
            database_path=database_path,
            repository_root=repository_root,
            run_id=run_id,
            component=component,
            callable_name=callable_name,
            started_at_utc=started_at,
            on_event=on_event,
        )
        self._operation_keys: set[tuple[str, str]] = set()

    @property
    def metadata(self) -> dict[str, Any]:
        return self._recorder.metadata

    def log_python_event(self, event: Mapping[str, Any]) -> None:
        self._recorder.log_python_event(event)

    def log_operation(self, operation_event: Mapping[str, Any]) -> None:
        row = dict(operation_event)
        name = str(row.get("operation", row.get("name", "unnamed_operation")))
        step = row.get("step", row.get("number"))
        status = row.get("status", "ok")
        translated = explain_operation(name, step, status)
        self._operation_keys.add((str(step), name))
        self._recorder.log_operation({
            **row,
            **translated,
            "phase": translated["phase"],
            "phase_explanation": translated["phase_explanation"],
            "status_explanation": translated["status_explanation"],
        })

    def log_component_traces(self, rows: list[Mapping[str, Any]]) -> None:
        pending = []
        for index, source_row in enumerate(rows):
            row = dict(source_row)
            name = str(row.get("operation", row.get("name", row.get("stage", row.get("step_name", f"trace_{index + 1}")))))
            step = row.get("step", row.get("index", index + 1))
            if (str(step), name) in self._operation_keys:
                continue
            status = row.get("status", "recorded")
            translated = explain_operation(name, step, status)
            pending.append({
                **row,
                "step": step,
                "operation": name,
                "status": status,
                "explanation": translated["explanation"],
                "phase": translated["phase"],
                "phase_explanation": translated["phase_explanation"],
                "status_explanation": translated["status_explanation"],
            })
        self._recorder.log_component_traces(pending)

    def finish(self, status: str, elapsed_ms: float | None,
               output_type: str | None = None, error: str | None = None) -> dict[str, Any]:
        return self._recorder.finish(status, elapsed_ms, output_type, error)
