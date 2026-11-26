"""Graph intelligence lifecycle events with ore_logger/SQLite persistence (step 40).

Every step of a graph task lifecycle is emitted as a structured event carrying the
correlation ID, snapshot hash, model version, query, candidates, scores, thresholds,
evidence, verifier rules, disposition, latency, resources and errors. When an
:class:`ore_logger.database.LoggerDatabase` is supplied, events are persisted to
SQLite through the shared ORE logger so the full candidate lifecycle can be
reconstructed later (acceptance test 25).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

#: Lifecycle event kinds emitted by the coordinator (steps 1-56).
EVENT_KINDS = (
    "task_received", "eligibility_decided", "snapshot_projected",
    "integrity_validated", "admission_decided", "path_selected",
    "cache_lookup", "candidates_generated", "candidates_scored",
    "candidates_ranked", "abstention_evaluated", "hypotheses_published",
    "verification_completed", "promotion_gated", "nar_proposed",
    "metrics_recorded", "cycle_closed", "error", "model_registered",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass(frozen=True, slots=True)
class GraphLifecycleEvent:
    """One structured lifecycle event (step 40)."""

    kind: str
    correlation_id: str
    sequence_no: int
    occurred_at_utc: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    duration_ms: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in EVENT_KINDS:
            raise ValueError(f"event kind must be one of {list(EVENT_KINDS)}")
        if not self.correlation_id.strip():
            raise ValueError("correlation_id must be non-empty")

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind, "correlation_id": self.correlation_id,
            "sequence_no": self.sequence_no, "occurred_at_utc": self.occurred_at_utc,
            "payload": dict(self.payload), "duration_ms": self.duration_ms,
        }


class GraphEventRecorder:
    """Record lifecycle events in memory and, optionally, ore_logger SQLite (step 40)."""

    def __init__(
        self,
        *,
        logger_database: Any | None = None,
        script_path: str = "ore_graph_intelligence.coordinator",
        module_name: str = "ore_graph_intelligence",
    ) -> None:
        self.events: list[GraphLifecycleEvent] = []
        self._sequence = 0
        self._run_id: str | None = None
        self._database = None
        self._script_path = script_path
        self._module_name = module_name
        if logger_database is not None:
            self._database = logger_database
            self._run_id = logger_database.create_run(
                script_path=script_path, module_name=module_name,
                entry_kind="service", working_directory=".",
            )

    @property
    def run_id(self) -> str | None:
        """SQLite run ID when persistence is active, otherwise None."""
        return self._run_id

    def emit(
        self,
        kind: str,
        correlation_id: str,
        payload: Mapping[str, Any] | None = None,
        *,
        duration_ms: float | None = None,
    ) -> GraphLifecycleEvent:
        """Emit one event; persist to SQLite when a logger database is attached."""
        self._sequence += 1
        event = GraphLifecycleEvent(
            kind=kind, correlation_id=correlation_id, sequence_no=self._sequence,
            occurred_at_utc=_utc_now(), payload=dict(payload or {}),
            duration_ms=duration_ms,
        )
        self.events.append(event)
        if self._database is not None and self._run_id is not None:
            self._database.record_event(
                self._run_id, event.sequence_no, event.kind,
                elapsed_ms=float(self._sequence), duration_ms=event.duration_ms,
                metadata={
                    "correlation_id": event.correlation_id,
                    **event.payload,
                },
            )
        return event

    def by_correlation_id(self, correlation_id: str) -> list[GraphLifecycleEvent]:
        """Return all events for one correlation ID in emission order."""
        return [event for event in self.events if event.correlation_id == correlation_id]

    def kinds(self) -> tuple[str, ...]:
        """Return the emitted kinds in first-emission order."""
        seen: list[str] = []
        for event in self.events:
            if event.kind not in seen:
                seen.append(event.kind)
        return tuple(seen)

    def finish(self, status: str = "completed") -> None:
        """Close the SQLite run when persistence is active."""
        if self._database is not None and self._run_id is not None:
            self._database.finish_run(self._run_id, status=status)

    def as_dicts(self) -> list[dict[str, object]]:
        return [event.as_dict() for event in self.events]


__all__ = ["EVENT_KINDS", "GraphEventRecorder", "GraphLifecycleEvent"]
