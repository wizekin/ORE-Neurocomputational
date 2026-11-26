"""Transactional outbox records and delivery retry with backpressure on backlog."""
from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..contracts import utc_now
from ..exceptions import OutboxPublishError
from .schemas import EventEnvelope, assert_valid_envelope


class OutboxStatus(str, Enum):
    """Track one outbox row from pending through delivered or failed."""

    PENDING = "pending"
    DELIVERED = "delivered"
    FAILED = "failed"


@dataclass(slots=True)
class OutboxRow:
    """Hold one outbound event record waiting for reliable publication."""

    outbox_id: str
    envelope: EventEnvelope
    created_at: float
    created_at_iso: str
    status: OutboxStatus = OutboxStatus.PENDING
    attempts: int = 0
    last_attempt_at: float | None = None
    delivered_at: float | None = None
    last_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable outbox record for diagnostics."""
        return {
            "outbox_id": self.outbox_id,
            "event_id": self.envelope.event_id,
            "event_type": self.envelope.event_type,
            "status": self.status.value,
            "attempts": self.attempts,
            "created_at_iso": self.created_at_iso,
            "last_error": self.last_error,
        }


class TransactionalOutbox:
    """Keep committed state changes paired with their outbound event records.

    Rows are appended inside the same critical section as the versioned-store
    commit (the engine passes the append as the commit transaction callback),
    so a committed state transition can never silently lose its event. A
    publisher drains pending rows later and marks them delivered only after
    successful publication; consumers remain idempotent across retries.
    """

    def __init__(
        self, *, max_attempts: int = 5, backoff_seconds: float = 0.01,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Initialize the outbox with bounded retry policy."""
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1.")
        if backoff_seconds < 0.0:
            raise ValueError("backoff_seconds cannot be negative.")
        self._lock = threading.RLock()
        self._rows: list[OutboxRow] = []
        self._max_attempts = max_attempts
        self._backoff = backoff_seconds
        self._clock = clock
        self._drained = 0
        self._failed_publications = 0

    # --------------------------------------------------------------- recording

    def record(self, envelope: EventEnvelope) -> OutboxRow:
        """Validate and append one pending row for later publication."""
        assert_valid_envelope(envelope)
        with self._lock:
            row = OutboxRow(
                outbox_id=f"outbox_{envelope.event_id}",
                envelope=envelope,
                created_at=self._clock(),
                created_at_iso=utc_now(),
            )
            self._rows.append(row)
            return row

    def record_all(self, envelopes: Iterable[EventEnvelope]) -> tuple[OutboxRow, ...]:
        """Append several pending rows atomically within one critical section."""
        with self._lock:
            return tuple(self.record(envelope) for envelope in envelopes)

    def as_transaction(self, envelopes: Iterable[EventEnvelope]) -> Callable[[], list[OutboxRow]]:
        """Return a transaction callback the versioned store executes on commit.

        The callback runs only when the optimistic version check has passed, so
        the outbox rows exist exactly when the state change commits.
        """
        pending = list(envelopes)
        for envelope in pending:
            assert_valid_envelope(envelope)

        def transaction() -> list[OutboxRow]:
            """Append the paired outbox rows inside the store's critical section."""
            return list(self.record_all(pending))

        return transaction

    # ---------------------------------------------------------------- draining

    def drain(
        self, publisher: Callable[[EventEnvelope], Any],
        *, max_rows: int | None = None,
    ) -> tuple[OutboxRow, ...]:
        """Publish pending rows in order and mark them delivered on success.

        Failed publications leave their rows pending with an incremented
        attempt count and a backoff delay, so a crashed publisher loses
        nothing: the next drain retries the same durable rows.
        """
        with self._lock:
            candidates = [
                row for row in self._rows if row.status is OutboxStatus.PENDING
            ]
            if max_rows is not None:
                candidates = candidates[:max_rows]
        delivered: list[OutboxRow] = []
        for row in candidates:
            now = self._clock()
            if row.last_attempt_at is not None:
                if now - row.last_attempt_at < self._backoff * row.attempts:
                    continue
            with self._lock:
                if row.status is not OutboxStatus.PENDING:
                    continue
                row.attempts += 1
                row.last_attempt_at = now
            try:
                publisher(row.envelope)
            except Exception as exc:  # noqa: BLE001 - publication isolation
                with self._lock:
                    row.status = OutboxStatus.FAILED
                    row.last_error = f"{type(exc).__name__}: {exc}"
                    self._failed_publications += 1
                    if row.attempts >= self._max_attempts:
                        row.status = OutboxStatus.FAILED
                    else:
                        row.status = OutboxStatus.PENDING
                continue
            with self._lock:
                row.status = OutboxStatus.DELIVERED
                row.delivered_at = self._clock()
                self._drained += 1
                delivered.append(row)
        return tuple(delivered)

    def pending_rows(self) -> tuple[OutboxRow, ...]:
        """Return every row that still awaits successful publication."""
        with self._lock:
            return tuple(
                row for row in self._rows if row.status is OutboxStatus.PENDING
            )

    def failed_rows(self) -> tuple[OutboxRow, ...]:
        """Return rows that exhausted their retry budget."""
        with self._lock:
            return tuple(
                row for row in self._rows
                if row.status is OutboxStatus.FAILED
                and row.attempts >= self._max_attempts
            )

    def raise_if_backlog(self, max_pending: int) -> None:
        """Raise when the durable backlog exceeds the configured bound."""
        if len(self.pending_rows()) > max_pending:
            raise OutboxPublishError(
                f"outbox backlog exceeds {max_pending} pending rows"
            )

    # ---------------------------------------------------------------- queries

    def backlog_age(self) -> float:
        """Return the age of the oldest pending row for lag monitoring."""
        with self._lock:
            pending = [
                row for row in self._rows if row.status is OutboxStatus.PENDING
            ]
            if not pending:
                return 0.0
            return self._clock() - min(row.created_at for row in pending)

    def rows(self) -> tuple[OutboxRow, ...]:
        """Return every outbox row in creation order."""
        with self._lock:
            return tuple(self._rows)

    def stats(self) -> dict[str, Any]:
        """Return outbox statistics for the metrics report."""
        with self._lock:
            pending = sum(
                1 for row in self._rows if row.status is OutboxStatus.PENDING
            )
            return {
                "total_rows": len(self._rows),
                "pending": pending,
                "delivered": self._drained,
                "failed_publications": self._failed_publications,
                "backlog_age": self.backlog_age(),
                "max_attempts": self._max_attempts,
            }

    def to_dicts(self) -> list[dict[str, Any]]:
        """Return serializable row records for replay manifests."""
        return [row.to_dict() for row in self.rows()]
