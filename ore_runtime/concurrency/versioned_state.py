"""Immutable versioned snapshots, optimistic commits, and atomic publication."""
from __future__ import annotations

import copy
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..contracts import CommitDecision, utc_now


@dataclass(frozen=True, slots=True)
class Snapshot:
    """Represent one immutable workspace revision read by tasks and projections."""

    revision: int
    data: Mapping[str, Any]
    index: Mapping[str, int]
    provenance: Mapping[str, Any]
    created_at: str

    def __post_init__(self) -> None:
        """Deep-copy caller data so the snapshot stays immutable for every reader."""
        object.__setattr__(self, "data", copy.deepcopy(dict(self.data)))
        object.__setattr__(self, "index", dict(self.index))
        object.__setattr__(self, "provenance", copy.deepcopy(dict(self.provenance)))

    def read(self, key: str, default: Any = None) -> Any:
        """Read one value from the immutable snapshot data."""
        return self.data.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable snapshot record for replay manifests."""
        return {
            "revision": self.revision,
            "data": dict(self.data),
            "index": dict(self.index),
            "provenance": dict(self.provenance),
            "created_at": self.created_at,
        }


@dataclass(frozen=True, slots=True)
class ReaderTicket:
    """Pin one revision for a reader so reclamation waits for the grace period."""

    ticket_id: int
    revision: int


@dataclass(frozen=True, slots=True)
class PreparedCommit:
    """Carry a candidate update prepared against a specific read revision."""

    read_version: int
    changes: Mapping[str, Any]
    change_keys: tuple[str, ...]
    validate: Callable[[Mapping[str, Any]], bool] | None
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Detach caller mappings from the prepared commit."""
        object.__setattr__(self, "changes", dict(self.changes))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class CommitResult:
    """Report an optimistic commit outcome with the resulting revision."""

    decision: CommitDecision
    read_version: int
    committed_revision: int | None
    expected_current: int | None
    reason: str

    @property
    def committed(self) -> bool:
        """Return whether the commit published a new authoritative revision."""
        return self.decision is CommitDecision.COMMITTED


class VersionedStore:
    """Own authoritative versioned state with optimistic validation and RCU reclamation.

    Commit linearization point: the single lock-protected pointer swap in
    :meth:`commit`. Readers never observe partially applied updates because a
    complete immutable revision is constructed off to the side and published by
    one assignment; older revisions are reclaimed only after all pre-existing
    readers of that revision have exited (grace-period condition).
    """

    def __init__(
        self, initial_data: Mapping[str, Any] | None = None,
        *, retained_revision_limit: int = 8,
        validation: Callable[[Mapping[str, Any]], bool] | None = None,
    ) -> None:
        """Initialize revision zero with detached data and bounded retention."""
        if retained_revision_limit < 1:
            raise ValueError("retained_revision_limit must be at least 1.")
        self._lock = threading.RLock()
        self._revision = 0
        self._current = Snapshot(
            revision=0,
            data=copy.deepcopy(dict(initial_data or {})),
            index={key: 0 for key in (initial_data or {})},
            provenance={"origin": "initialization"},
            created_at=utc_now(),
        )
        self._retained: dict[int, Snapshot] = {0: self._current}
        self._retained_revision_limit = retained_revision_limit
        self._validation = validation
        self._readers: dict[int, set[int]] = {}
        self._ticket_counter = 0
        self._commit_count = 0
        self._conflict_count = 0

    # ------------------------------------------------------------------ reads

    def latest(self) -> Snapshot:
        """Return the current authoritative snapshot without pinning a revision."""
        with self._lock:
            return self._current

    def revision(self, version: int) -> Snapshot:
        """Return a retained historical snapshot by revision number."""
        with self._lock:
            if version == self._revision:
                return self._current
            snapshot = self._retained.get(version)
            if snapshot is None:
                raise KeyError(f"revision {version} is not retained by this store.")
            return snapshot

    def capture(self) -> Snapshot:
        """Capture a detached deep copy of the current revision as a read context."""
        with self._lock:
            return Snapshot(
                revision=self._current.revision,
                data=copy.deepcopy(dict(self._current.data)),
                index=dict(self._current.index),
                provenance=copy.deepcopy(dict(self._current.provenance)),
                created_at=self._current.created_at,
            )

    def pin(self, snapshot: Snapshot) -> ReaderTicket:
        """Pin a read revision so RCU reclamation waits for this reader to exit."""
        with self._lock:
            self._ticket_counter += 1
            ticket = ReaderTicket(ticket_id=self._ticket_counter,
                                  revision=snapshot.revision)
            self._readers.setdefault(snapshot.revision, set()).add(ticket.ticket_id)
            return ticket

    def unpin(self, ticket: ReaderTicket) -> None:
        """Release a reader pin and reclaim unobserved revisions after the grace period."""
        with self._lock:
            holders = self._readers.get(ticket.revision)
            if holders is None or ticket.ticket_id not in holders:
                raise KeyError(f"reader ticket {ticket.ticket_id} is not pinned.")
            holders.discard(ticket.ticket_id)
            if not holders:
                self._readers.pop(ticket.revision, None)
            self._reclaim_locked()

    def _reclaim_locked(self) -> None:
        """Drop the oldest retained revisions that no reader still observes."""
        oldest_kept = max(
            0, self._revision - self._retained_revision_limit + 1
        )
        for version in sorted(self._retained):
            if version >= oldest_kept or version == self._revision:
                break
            if self._readers.get(version):
                continue
            self._retained.pop(version, None)

    def pinned_revisions(self) -> tuple[int, ...]:
        """Return the revisions currently pinned by active readers."""
        with self._lock:
            return tuple(sorted(rev for rev, holders in self._readers.items() if holders))

    def retained_revisions(self) -> tuple[int, ...]:
        """Return the revisions currently retained for replay and rollback."""
        with self._lock:
            return tuple(sorted(self._retained))

    # ----------------------------------------------------------------- writes

    def prepare(
        self, read_version: int, changes: Mapping[str, Any],
        *, validate: Callable[[Mapping[str, Any]], bool] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> PreparedCommit:
        """Prepare a candidate update from a specific read revision."""
        if read_version < 0:
            raise ValueError("read_version cannot be negative.")
        return PreparedCommit(
            read_version=read_version,
            changes=dict(changes),
            change_keys=tuple(sorted(changes)),
            validate=validate,
            metadata=dict(metadata or {}),
        )

    def commit(
        self, prepared: PreparedCommit,
        transaction: Callable[[], Sequence[Any]] | None = None,
    ) -> CommitResult:
        """Commit optimistically, publishing a complete revision atomically.

        The optional ``transaction`` callback runs inside the same critical
        section as the pointer swap: it must return side records (for example
        outbox rows) that are persisted only when the commit itself succeeds.
        This makes state change and event record one local transaction.
        """
        side_records: list[Any] = []
        with self._lock:
            if prepared.read_version != self._revision:
                self._conflict_count += 1
                return CommitResult(
                    decision=CommitDecision.CONFLICT,
                    read_version=prepared.read_version,
                    committed_revision=None,
                    expected_current=self._revision,
                    reason=(
                        "authoritative revision moved from "
                        f"{prepared.read_version} to {self._revision}"
                    ),
                )
            candidate = copy.deepcopy(self._current.data)
            candidate.update(copy.deepcopy(dict(prepared.changes)))
            validator = prepared.validate or self._validation
            if validator is not None and not validator(candidate):
                self._conflict_count += 1
                return CommitResult(
                    decision=CommitDecision.CONFLICT,
                    read_version=prepared.read_version,
                    committed_revision=None,
                    expected_current=self._revision,
                    reason="candidate revision failed validation",
                )
            next_revision = self._revision + 1
            index = dict(self._current.index)
            for key in prepared.change_keys:
                index[key] = next_revision
            new_snapshot = Snapshot(
                revision=next_revision,
                data=candidate,
                index=index,
                provenance=dict(prepared.metadata or {"origin": "runtime_commit"}),
                created_at=utc_now(),
            )
            if transaction is not None:
                side_records = list(transaction())
            self._retained[next_revision] = new_snapshot
            self._revision = next_revision
            self._current = new_snapshot   # single pointer swap = linearization point
            self._commit_count += 1
            self._reclaim_locked()
        for record in side_records:
            self.on_transaction_record(record)
        return CommitResult(
            decision=CommitDecision.COMMITTED,
            read_version=prepared.read_version,
            committed_revision=self._revision,
            expected_current=self._revision,
            reason="committed",
        )

    def on_transaction_record(self, record: Any) -> None:
        """Handle one side record produced by a commit transaction (overridable)."""

    def commit_batch(self, commits: Sequence[PreparedCommit]) -> tuple[CommitResult, ...]:
        """Apply a batch of prepared commits sequentially, reporting each outcome."""
        results: list[CommitResult] = []
        for prepared in commits:
            results.append(self.commit(prepared))
        return tuple(results)

    # ---------------------------------------------------------------- metrics

    def stats(self) -> dict[str, Any]:
        """Return commit, conflict, and retention statistics for observability."""
        with self._lock:
            return {
                "revision": self._revision,
                "commits": self._commit_count,
                "conflicts": self._conflict_count,
                "retained_revisions": self.retained_revisions(),
                "pinned_revisions": self.pinned_revisions(),
            }
