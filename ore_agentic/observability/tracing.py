"""Distributed tracing with secret redaction (Step 43).

Trace, workflow, task and parent-span IDs propagate across coordinators,
workers and tools through the protocol envelopes.  Spans record latency,
status, retries, queue wait and provenance, and secrets are redacted from
logs and model-visible outputs before persistence.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import re
import time
from typing import Any, Iterator, Mapping

from ..contracts import new_id, utc_now

__all__ = (
    "REDACTED",
    "Span",
    "TraceContext",
    "Tracer",
    "redact",
    "redact_value",
)

REDACTED = "***REDACTED***"

_FORBIDDEN_KEYS = frozenset({
    "secret", "secrets", "password", "token", "api_key", "apikey",
    "credentials", "authorization", "cookie", "session_key", "private_key",
    "access_key",
})

_SECRET_PATTERNS = tuple(re.compile(pattern) for pattern in (
    r"sk-[A-Za-z0-9]{8,}",
    r"Bearer\s+[A-Za-z0-9._\-]{8,}",
    r"(?i)password=[^\s&]+",
    r"(?i)token=[^\s&]+",
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
))


def redact_value(value: Any) -> Any:
    """Redact secret-looking strings inside any value (Step 43)."""
    if isinstance(value, str):
        redacted = value
        for pattern in _SECRET_PATTERNS:
            redacted = pattern.sub(REDACTED, redacted)
        return redacted
    if isinstance(value, Mapping):
        return redact(value)
    if isinstance(value, (list, tuple)):
        sequence = [redact_value(item) for item in value]
        return type(value)(sequence) if isinstance(value, tuple) else sequence
    return value


def redact(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Redact secret-bearing keys and secret-looking values (Step 43)."""
    result: dict[str, Any] = {}
    for key, value in payload.items():
        if str(key).lower() in _FORBIDDEN_KEYS:
            result[key] = REDACTED
        else:
            result[key] = redact_value(value)
    return result


@dataclass(frozen=True, slots=True)
class TraceContext:
    """The propagated trace identity (Step 43)."""

    trace_id: str
    workflow_id: str = ""
    task_id: str = ""
    parent_span_id: str = ""

    @classmethod
    def create(cls, workflow_id: str = "", task_id: str = "",
               parent_span_id: str = "", trace_id: str = "") -> TraceContext:
        """Create a fresh trace context."""
        return cls(trace_id=trace_id or new_id("trace"), workflow_id=workflow_id,
                   task_id=task_id, parent_span_id=parent_span_id)

    def for_task(self, task_id: str, parent_span_id: str = "") -> TraceContext:
        """Derive the context of one task span."""
        return TraceContext(trace_id=self.trace_id, workflow_id=self.workflow_id,
                            task_id=task_id, parent_span_id=parent_span_id)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the context for propagation."""
        return {"trace_id": self.trace_id, "workflow_id": self.workflow_id,
                "task_id": self.task_id, "parent_span_id": self.parent_span_id}


@dataclass(slots=True)
class Span:
    """One traced operation with timing and redacted attributes (Step 43)."""

    span_id: str
    trace_id: str
    operation: str
    started_at: float
    ended_at: float | None = None
    status: str = "running"
    attributes: dict[str, Any] = field(default_factory=dict)
    redacted_fields: tuple[str, ...] = ()
    workflow_id: str = ""
    task_id: str = ""
    parent_span_id: str = ""

    @property
    def duration_ms(self) -> float:
        """Return the span duration in milliseconds."""
        if self.ended_at is None:
            return 0.0
        return round((self.ended_at - self.started_at) * 1000.0, 3)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the span with redacted attributes only."""
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "operation": self.operation,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "attributes": dict(self.attributes),
            "redacted_fields": list(self.redacted_fields),
            "workflow_id": self.workflow_id,
            "task_id": self.task_id,
            "parent_span_id": self.parent_span_id,
            "recorded_at": utc_now(),
        }


class Tracer:
    """Collect spans for one trace with redaction on close (Step 43)."""

    def __init__(self, context: TraceContext, *,
                 sink: Any = None) -> None:
        """Bind the tracer to a trace context and optional persistence sink."""
        self._context = context
        self._sink = sink
        self._spans: list[Span] = []

    @property
    def context(self) -> TraceContext:
        """Return the trace context."""
        return self._context

    @contextmanager
    def span(self, operation: str, **attributes: Any) -> Iterator[Span]:
        """Open one span, redacting attributes, closing with status (Step 43)."""
        safe = redact(attributes)
        redacted_fields = tuple(key for key in attributes
                                if key not in safe or safe[key] == REDACTED)
        current = Span(
            span_id=new_id("span"), trace_id=self._context.trace_id,
            operation=operation, started_at=time.monotonic(),
            attributes=safe, redacted_fields=redacted_fields,
            workflow_id=self._context.workflow_id,
            task_id=self._context.task_id,
            parent_span_id=self._context.parent_span_id)
        self._spans.append(current)
        try:
            yield current
            current.status = "completed"
        except BaseException as exc:  # noqa: BLE001 - span status then re-raise
            current.status = f"failed: {type(exc).__name__}"
            raise
        finally:
            current.ended_at = time.monotonic()
            if self._sink is not None:
                try:
                    self._sink(current.to_dict())
                except Exception:  # noqa: BLE001 - tracing must never break work
                    pass

    def spans(self) -> tuple[Span, ...]:
        """Return all spans of the trace."""
        return tuple(self._spans)

    def summary(self) -> dict[str, Any]:
        """Return a redacted, serializable trace summary (Step 43)."""
        return {
            "trace_id": self._context.trace_id,
            "workflow_id": self._context.workflow_id,
            "spans": len(self._spans),
            "operations": [span.operation for span in self._spans],
            "total_duration_ms": round(
                sum(span.duration_ms for span in self._spans), 3),
        }
