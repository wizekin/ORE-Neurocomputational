"""Specialist adapters: the typed boundary around every learned model (Section 2.1).

Adapters isolate framework- and modality-specific interfaces behind stable ORE
contracts.  Every adapter declares its input and output schemas; the executor
validates adapter output against the declared schema before anything downstream can
consume it.  Adapters receive an :class:`ExecutionContext` and must call
``checkpoint()`` periodically so deadlines and cancellation are honoured.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

from ..contracts import AdapterKind, SchemaDescriptor

__all__ = ("AdapterOutputError", "AdapterRegistry", "ExecutionContext",
           "SpecialistAdapter", "validate_payload_against_schema")


class AdapterOutputError(ValueError):
    """Raised when an adapter's output violates its declared output schema."""


_NUMERIC_FRACTION_FIELDS = ("score", "confidence", "probability", "support")
_TEXTUAL_FIELDS = ("text", "caption", "transcript", "label", "language",
                   "code_text", "region", "event_label", "frame")


def validate_payload_against_schema(payload: Mapping[str, Any],
                                    schema: SchemaDescriptor) -> tuple[bool, str]:
    """Validate a typed payload against its schema descriptor (acceptance 10).

    Every declared field must be present and non-null; fraction-like fields
    (score/confidence/probability/support) must be numbers in [0, 1]; textual fields
    must be strings.  The check is intentionally conservative: schema violations are
    rejected before the payload can travel further.
    """
    if not isinstance(payload, Mapping):
        return False, "payload must be a mapping"
    for field_name in schema.fields:
        if field_name not in payload:
            return False, f"missing required field {field_name!r}"
        value = payload[field_name]
        if value is None:
            return False, f"field {field_name!r} is null"
        lowered = field_name.lower()
        if any(marker in lowered for marker in _NUMERIC_FRACTION_FIELDS):
            numeric = _as_float(value)
            if numeric is None or not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
                return False, (f"field {field_name!r} must be a fraction in [0, 1], "
                               f"got {value!r}")
        if any(lowered == marker or lowered.endswith(marker) for marker in _TEXTUAL_FIELDS):
            if isinstance(value, str) and not value.strip():
                return False, f"field {field_name!r} must be a non-empty string"
    return True, "payload satisfies the schema"


def _as_float(value: Any) -> float | None:
    """Best-effort float conversion; returns None for non-numeric values."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


class ExecutionContext:
    """Deadline, budget, and cancellation state for one adapter execution."""

    def __init__(self, *, deadline_seconds: float = 5.0, budget_ms: float = 1000.0,
                 cancellation: Any = None, max_steps: int = 10_000) -> None:
        """Configure the execution envelope."""
        self.deadline_seconds = float(deadline_seconds)
        self.budget_ms = float(budget_ms)
        self.cancellation = cancellation
        self.max_steps = int(max_steps)
        self.started_at = time.perf_counter()
        self.steps = 0
        self.cancelled = False
        self.cancel_reason = ""
        self.checkpoints: list[tuple[int, float]] = []

    @property
    def elapsed_ms(self) -> float:
        """Milliseconds since execution start."""
        return (time.perf_counter() - self.started_at) * 1000.0

    @property
    def timed_out(self) -> bool:
        """True when the wall-clock deadline or step budget is exhausted."""
        elapsed = time.perf_counter() - self.started_at
        return (elapsed >= self.deadline_seconds
                or self.elapsed_ms >= self.budget_ms
                or self.steps > self.max_steps)

    def checkpoint(self) -> None:
        """Cooperative cancellation point; adapters call this between work units."""
        self.steps += 1
        self.checkpoints.append((self.steps, self.elapsed_ms))
        if self.cancellation is not None and getattr(self.cancellation, "cancelled",
                                                     False):
            self.cancelled = True
            self.cancel_reason = getattr(self.cancellation, "reason", "cancelled")
            raise TimeoutError("execution cancelled: " + self.cancel_reason)
        if self.timed_out:
            self.cancelled = True
            self.cancel_reason = "deadline exceeded"
            raise TimeoutError("execution cancelled: deadline exceeded")

    def describe(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {"deadline_seconds": self.deadline_seconds, "budget_ms": self.budget_ms,
                "steps": self.steps, "elapsed_ms": round(self.elapsed_ms, 3),
                "cancelled": self.cancelled, "cancel_reason": self.cancel_reason}


@runtime_checkable
class SpecialistAdapter(Protocol):
    """The stable ORE-side interface every specialist implements."""

    adapter_kind: AdapterKind
    input_schema: SchemaDescriptor
    output_schema: SchemaDescriptor

    def invoke(self, payload: Mapping[str, Any],
               execution: ExecutionContext) -> dict[str, Any]: ...


class AdapterRegistry:
    """Registry of available adapters keyed by adapter kind and framework."""

    def __init__(self) -> None:
        """Start empty."""
        self._adapters: dict[tuple[str, str], Any] = {}
        self._default: dict[str, Any] = {}

    def register(self, adapter: Any, framework: str = "numpy-reference",
                 *, make_default: bool = True) -> None:
        """Register an adapter instance for its kind and a framework tag."""
        if not isinstance(adapter, SpecialistAdapter):
            raise TypeError("adapter must satisfy the SpecialistAdapter protocol")
        self._adapters[(adapter.adapter_kind.value, framework)] = adapter
        if make_default:
            self._default[adapter.adapter_kind.value] = adapter

    def get(self, kind: AdapterKind,
            framework: str | None = None) -> Any:
        """Look up an adapter by kind, optionally for a specific framework."""
        if framework is not None:
            adapter = self._adapters.get((kind.value, framework))
            if adapter is not None:
                return adapter
        adapter = self._default.get(kind.value)
        if adapter is None:
            raise KeyError(f"no adapter registered for kind {kind.value}")
        return adapter

    def describe(self) -> dict[str, Any]:
        """List registered adapters."""
        return {"adapters": sorted(f"{kind}:{framework}"
                                   for (kind, framework) in self._adapters),
                "defaults": sorted(self._default)}
