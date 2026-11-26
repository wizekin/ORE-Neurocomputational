"""Durable event log with per-aggregate order, replay, snapshots, projections."""
from __future__ import annotations

import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..contracts import new_id, utc_now
from ..events.schemas import EventEnvelope


@dataclass(frozen=True, slots=True)
class QuarantinedEvent:
    """Hold one out-of-order event rejected from an aggregate's ordered log."""

    envelope: EventEnvelope
    expected_version: int
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable quarantine record."""
        return {
            "event_id": self.envelope.event_id,
            "aggregate_id": self.envelope.aggregate_id,
            "aggregate_version": self.envelope.aggregate_version,
            "expected_version": self.expected_version,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class StoredEvent:
    """Position one committed event in the global durable sequence."""

    global_sequence: int
    envelope: EventEnvelope
    committed_at: str


@dataclass(slots=True)
class AggregateSnapshot:
    """Bound replay cost by retaining a materialized aggregate state."""

    aggregate_id: str
    version: int
    state: Mapping[str, Any]
    saved_at: str


@dataclass(frozen=True, slots=True)
class Projection:
    """Serve one read-optimized projection with explicit staleness reporting."""

    name: str
    current_through_sequence: int
    state: Mapping[str, Any]

    @property
    def freshness(self) -> int:
        """Return the global sequence the projection is current through."""
        return self.current_through_sequence

    def staleness(self, latest_sequence: int) -> int:
        """Return how many committed events this projection has not consumed."""
        return max(0, latest_sequence - self.current_through_sequence)


class EventStore:
    """Append committed events once, per-aggregate ordered, and rebuild by replay.

    Appending is the linearization point of the log: each aggregate's version
    must extend the aggregate's current version exactly, and out-of-order
    envelopes are quarantined instead of silently reordered. State is rebuilt
    deterministically as State_n = Reduce(State_0, [E_1, ..., E_n]).
    """

    def __init__(self) -> None:
        """Initialize the empty durable log, snapshots, and quarantine."""
        self._lock = threading.RLock()
        self._events: list[StoredEvent] = []
        self._by_aggregate: dict[str, list[StoredEvent]] = {}
        self._aggregate_versions: dict[str, int] = {}
        self._snapshots: dict[str, AggregateSnapshot] = {}
        self._quarantine: list[QuarantinedEvent] = []
        self._global_sequence = 0

    # --------------------------------------------------------------- append

    def append(
        self, envelope: EventEnvelope, *, expected_version: int | None = None,
        strict: bool = True,
    ) -> StoredEvent | QuarantinedEvent:
        """Append one committed event with per-aggregate sequence enforcement.

        When ``strict`` is true, an envelope whose aggregate_version does not
        exactly extend the aggregate's current version is quarantined and the
        call raises; when false, the event is quarantined and returned so the
        caller can log it without corrupting the ordered log.
        """
        with self._lock:
            current = self._aggregate_versions.get(envelope.aggregate_id, 0)
            if expected_version is None:
                expected_version = current
            if envelope.aggregate_version != expected_version:
                quarantined = QuarantinedEvent(
                    envelope=envelope,
                    expected_version=expected_version,
                    reason=(
                        f"aggregate '{envelope.aggregate_id}' is at version "
                        f"{current}; envelope carries {envelope.aggregate_version}"
                    ),
                )
                self._quarantine.append(quarantined)
                if strict:
                    raise ValueError(quarantined.reason)
                return quarantined
            stored = StoredEvent(
                global_sequence=self._global_sequence + 1,
                envelope=envelope,
                committed_at=utc_now(),
            )
            self._global_sequence += 1
            self._events.append(stored)
            self._by_aggregate.setdefault(envelope.aggregate_id, []).append(stored)
            self._aggregate_versions[envelope.aggregate_id] = (
                envelope.aggregate_version + 1
            )
            return stored

    def append_committed(
        self, envelopes: Sequence[EventEnvelope],
    ) -> tuple[StoredEvent, ...]:
        """Append several committed envelopes in order under one critical section."""
        with self._lock:
            stored: list[StoredEvent] = []
            for envelope in envelopes:
                stored.append(self.append(envelope))
            return tuple(stored)

    # --------------------------------------------------------------- queries

    @property
    def global_sequence(self) -> int:
        """Return the current global sequence number of the durable log."""
        return self._global_sequence

    def latest_aggregate_version(self, aggregate_id: str) -> int:
        """Return an aggregate's next expected aggregate version."""
        with self._lock:
            return self._aggregate_versions.get(aggregate_id, 0)

    def events(self) -> tuple[StoredEvent, ...]:
        """Return every stored event in global sequence order."""
        with self._lock:
            return tuple(self._events)

    def events_for_aggregate(self, aggregate_id: str) -> tuple[EventEnvelope, ...]:
        """Return one aggregate's events in per-aggregate sequence order."""
        with self._lock:
            return tuple(
                stored.envelope
                for stored in self._by_aggregate.get(aggregate_id, ())
            )

    def quarantined(self) -> tuple[QuarantinedEvent, ...]:
        """Return every quarantined out-of-order event for audit."""
        with self._lock:
            return tuple(self._quarantine)

    def by_correlation(self, correlation_id: str) -> tuple[StoredEvent, ...]:
        """Reconstruct one causal event chain by its correlation identifier."""
        with self._lock:
            return tuple(
                stored for stored in self._events
                if stored.envelope.correlation_id == correlation_id
            )

    def by_causation(self, event_id: str) -> tuple[StoredEvent, ...]:
        """Return the events directly caused by one parent event identifier."""
        with self._lock:
            return tuple(
                stored for stored in self._events
                if stored.envelope.causation_id == event_id
            )

    def causal_chain(self, correlation_id: str) -> tuple[dict[str, Any], ...]:
        """Return the correlation's events with their causal links for the audit."""
        stored_events = self.by_correlation(correlation_id)
        return tuple(
            {
                "event_id": stored.envelope.event_id,
                "event_type": stored.envelope.event_type,
                "causation_id": stored.envelope.causation_id,
                "sequence": stored.global_sequence,
            }
            for stored in stored_events
        )

    # ---------------------------------------------------- replay and reduce

    def replay(
        self, reducer: Callable[[Any, EventEnvelope], Any], initial: Any = None,
        *,
        aggregate_id: str | None = None,
        upto_sequence: int | None = None,
    ) -> Any:
        """Reduce committed events into a rebuilt state deterministically."""
        with self._lock:
            selected = list(self._events)
        if aggregate_id is not None:
            selected = [
                stored for stored in selected
                if stored.envelope.aggregate_id == aggregate_id
            ]
        if upto_sequence is not None:
            selected = [
                stored for stored in selected
                if stored.global_sequence <= upto_sequence
            ]
        state = initial
        for stored in selected:
            state = reducer(state, stored.envelope)
        return state

    def save_snapshot(self, aggregate_id: str, state: Mapping[str, Any]) -> None:
        """Persist a materialized aggregate state to bound replay cost."""
        with self._lock:
            version = self._aggregate_versions.get(aggregate_id, 0)
            self._snapshots[aggregate_id] = AggregateSnapshot(
                aggregate_id=aggregate_id, version=version,
                state=dict(state), saved_at=utc_now(),
            )

    def load_snapshot(self, aggregate_id: str) -> AggregateSnapshot | None:
        """Return the retained snapshot for one aggregate when present."""
        with self._lock:
            snapshot = self._snapshots.get(aggregate_id)
            return snapshot

    # ------------------------------------------------------------ projection

    def build_projection(
        self, name: str, reducer: Callable[[Any, EventEnvelope], Any],
        initial: Any = None, *,
        upto_sequence: int | None = None,
    ) -> Projection:
        """Build one read projection and expose its freshness explicitly."""
        with self._lock:
            latest = self._global_sequence
        state = self.replay(reducer, initial, upto_sequence=upto_sequence)
        current = (
            latest if upto_sequence is None
            else min(upto_sequence, latest)
        )
        return Projection(
            name=name,
            current_through_sequence=current,
            state=state if isinstance(state, Mapping) else {"value": state},
        )

    def stats(self) -> dict[str, Any]:
        """Return log, snapshot, and quarantine counts for observability."""
        with self._lock:
            return {
                "global_sequence": self._global_sequence,
                "event_count": len(self._events),
                "aggregates": len(self._by_aggregate),
                "snapshots": len(self._snapshots),
                "quarantined": len(self._quarantine),
            }
