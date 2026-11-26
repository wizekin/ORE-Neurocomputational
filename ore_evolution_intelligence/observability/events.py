"""Evolution lifecycle events with ore_logger/SQLite persistence (section 5).

Every step of an evolution run is emitted as a structured event carrying the run ID,
objective, candidate identifiers, metric summaries, decisions and errors. When an
:class:`ore_logger.database.LoggerDatabase` is supplied, events persist through the
shared ORE logger so the full experiment history can be reconstructed later (SQLite /
ore_logger integration row of the ORE integration map).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

#: Lifecycle event kinds emitted during evolution runs.
EVENT_KINDS: tuple[str, ...] = (
    "run_started", "objective_narrowed", "baseline_captured", "benchmark_versioned",
    "split_declared", "budget_granted", "population_initialized",
    "candidate_generated", "candidate_evaluated", "generation_completed",
    "search_stopped", "archive_initialized", "archive_cell_updated",
    "archive_metrics_recorded", "verification_completed", "promotion_requested",
    "promotion_decided", "registry_registered", "canary_started", "canary_evaluated",
    "drift_detected", "rollback_triggered", "strategy_deactivated",
    "budget_exhausted", "workspace_published", "run_completed", "run_failed", "error",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass(frozen=True, slots=True)
class EvolutionEvent:
    """One structured evolution event."""

    kind: str
    run_id: str
    sequence_no: int
    occurred_at_utc: str
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in EVENT_KINDS:
            raise ValueError(f"event kind must be one of {list(EVENT_KINDS)}")
        if not self.run_id.strip():
            raise ValueError("run_id must be non-empty")

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind, "run_id": self.run_id,
            "sequence_no": self.sequence_no, "occurred_at_utc": self.occurred_at_utc,
            "payload": dict(self.payload),
        }


class EvolutionEventRecorder:
    """Record evolution events in memory and, optionally, ore_logger SQLite."""

    def __init__(
        self,
        *,
        logger_database: Any | None = None,
        script_path: str = "ore_evolution_intelligence.evolution_service",
        module_name: str = "ore_evolution_intelligence",
    ) -> None:
        self.events: list[EvolutionEvent] = []
        self._sequence = 0
        self._database = None
        self._run_id: Any | None = None
        self._script_path = script_path
        self._module_name = module_name
        if logger_database is not None:
            self._database = logger_database
            self._run_id = logger_database.create_run(
                script_path=script_path, module_name=module_name,
                entry_kind="service", working_directory=".",
            )

    @property
    def logger_run_id(self) -> Any | None:
        """SQLite run ID when ore_logger persistence is active."""
        return self._run_id

    def emit(
        self,
        kind: str,
        run_id: str,
        payload: Mapping[str, Any] | None = None,
    ) -> EvolutionEvent:
        """Emit one event; persist to SQLite when a logger database is attached."""
        self._sequence += 1
        event = EvolutionEvent(
            kind=kind, run_id=run_id, sequence_no=self._sequence,
            occurred_at_utc=_utc_now(), payload=dict(payload or {}),
        )
        self.events.append(event)
        if self._database is not None and self._run_id is not None:
            self._database.record_event(
                self._run_id, event.sequence_no, event.kind,
                elapsed_ms=float(self._sequence), duration_ms=None,
                metadata={"run_id": event.run_id, **event.payload},
            )
        return event

    def by_run(self, run_id: str) -> list[EvolutionEvent]:
        """All events of one run in emission order."""
        return [event for event in self.events if event.run_id == run_id]

    def kinds(self) -> tuple[str, ...]:
        seen: list[str] = []
        for event in self.events:
            if event.kind not in seen:
                seen.append(event.kind)
        return tuple(seen)

    def finish(self, status: str = "completed") -> None:
        """Close the SQLite run when persistence is active."""
        if self._database is not None and self._run_id is not None:
            self._database.finish_run(self._run_id, status=status)

    def as_dicts(self) -> list[dict[str, Any]]:
        return [event.as_dict() for event in self.events]


__all__ = ["EVENT_KINDS", "EvolutionEvent", "EvolutionEventRecorder"]
