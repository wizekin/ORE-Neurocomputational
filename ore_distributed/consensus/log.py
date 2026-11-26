"""Replicated log entries and majority quorum arithmetic (Steps 14 and 16).

A log entry is committed when replicated to a majority quorum under the
protocol's current-term and leader rules; any two majorities of the same fixed
voter set intersect, which is the basis for preventing disjoint majorities from
independently committing conflicting histories.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ore_distributed.consensus.state_machine import StateMachineCommand


def quorum_size(voters: int) -> int:
    """Return the majority quorum size ``Q = floor(N / 2) + 1`` for N voters."""
    if not isinstance(voters, int) or voters < 1:
        raise ValueError("voters must be a positive integer")
    return voters // 2 + 1


def majority_quorum(voter_ids: Sequence[str]) -> frozenset[str]:
    """Return the smallest majority of a sorted voter set as a frozenset."""
    ids = sorted(set(str(node) for node in voter_ids))
    if not ids:
        raise ValueError("voter set must not be empty")
    keep = quorum_size(len(ids))
    return frozenset(ids[:keep])


@dataclass(frozen=True, slots=True)
class LogEntry:
    """One replicated log entry carrying a deterministic state-machine command."""

    index: int
    term: int
    command: StateMachineCommand
    committed: bool = False

    def __post_init__(self) -> None:
        """Validate index and term bounds and command type."""
        if not isinstance(self.index, int) or self.index < 1:
            raise ValueError("log index starts at 1 and must be a positive integer")
        if not isinstance(self.term, int) or self.term < 0:
            raise ValueError("term must be a non-negative integer")
        if not isinstance(self.command, StateMachineCommand):
            raise TypeError("command must be a StateMachineCommand")

    def with_committed(self, committed: bool) -> "LogEntry":
        """Return a copy flagged as committed (or not)."""
        return LogEntry(
            index=self.index,
            term=self.term,
            command=self.command,
            committed=committed,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the entry as a plain dictionary."""
        return {
            "index": self.index,
            "term": self.term,
            "command": self.command.to_dict(),
            "committed": self.committed,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LogEntry":
        """Rebuild a log entry from its dictionary form."""
        return cls(
            index=int(payload["index"]),
            term=int(payload["term"]),
            command=StateMachineCommand.from_dict(payload["command"]),
            committed=bool(payload.get("committed", False)),
        )


class ReplicatedLog:
    """In-memory log holding entries indexed from 1 with append and truncate rules."""

    def __init__(self) -> None:
        """Create an empty log."""
        self._entries: list[LogEntry] = []

    def __len__(self) -> int:
        """Return the number of stored entries."""
        return len(self._entries)

    @property
    def last_index(self) -> int:
        """Return the highest stored index, or 0 when empty."""
        return len(self._entries)

    @property
    def last_term(self) -> int:
        """Return the term of the highest stored entry, or 0 when empty."""
        return self._entries[-1].term if self._entries else 0

    def entry_at(self, index: int) -> LogEntry | None:
        """Return the entry at a 1-based index or None when out of range."""
        if index < 1 or index > len(self._entries):
            return None
        return self._entries[index - 1]

    def term_at(self, index: int) -> int | None:
        """Return the term at a 1-based index or None when out of range."""
        entry = self.entry_at(index)
        return entry.term if entry is not None else None

    def append(self, term: int, command: StateMachineCommand) -> LogEntry:
        """Append one entry and return it."""
        if self._entries and term < self._entries[-1].term:
            raise ValueError("terms must not move backward within the log")
        entry = LogEntry(index=len(self._entries) + 1, term=term, command=command)
        self._entries.append(entry)
        return entry

    def append_entry(self, entry: LogEntry) -> LogEntry:
        """Append a prebuilt entry; index must extend the log exactly."""
        if entry.index != len(self._entries) + 1:
            raise ValueError("appended entry must extend the log contiguously")
        if self._entries and entry.term < self._entries[-1].term:
            raise ValueError("appended entry term must not move backward")
        self._entries.append(entry)
        return entry

    def truncate_suffix(self, from_index: int) -> int:
        """Remove every entry at or beyond ``from_index``; returns entries dropped.

        Truncation implements the follower rule for conflicting uncommitted
        suffixes: Raft permits followers to delete entries that conflict with
        the leader's log because such entries were never committed.
        """
        from_index = int(from_index)
        if from_index < 1:
            raise ValueError("truncation index starts at 1")
        dropped = max(0, len(self._entries) - (from_index - 1))
        self._entries = self._entries[: from_index - 1]
        return dropped

    def entries_from(self, start_index: int, limit: int | None = None) -> tuple[LogEntry, ...]:
        """Return committed-suffix-agnostic entries at or beyond a 1-based index."""
        start_index = int(start_index)
        if start_index < 1:
            raise ValueError("start index starts at 1")
        slice_ = self._entries[start_index - 1:]
        if limit is not None:
            slice_ = slice_[: int(limit)]
        return tuple(slice_)

    def prefix(self, through_index: int) -> tuple[LogEntry, ...]:
        """Return entries at or before a 1-based index."""
        through_index = int(through_index)
        if through_index < 0:
            raise ValueError("through index must be non-negative")
        return tuple(self._entries[:through_index])

    def snapshot(self) -> tuple[dict[str, Any], ...]:
        """Return every entry as a plain dictionary tuple for persistence."""
        return tuple(entry.to_dict() for entry in self._entries)

    def restore(self, entries: Sequence[Mapping[str, Any]]) -> None:
        """Restore entries from a persisted snapshot, replacing current content."""
        self._entries = [LogEntry.from_dict(item) for item in entries]
        for position, entry in enumerate(self._entries, start=1):
            if entry.index != position:
                raise ValueError("restored log indices must be contiguous from 1")


def quorum_intersection(voters: Sequence[str], quorum_a: set[str], quorum_b: set[str]) -> int:
    """Return ``|Qa ∩ Qb|`` proving two majorities of one voter set intersect.

    For majorities of the same fixed voter set this is always at least
    ``2Q - N > 0``, the property that blocks two disjoint majorities from
    independently committing conflicting histories.
    """
    total = len(set(voters))
    if not quorum_a or not quorum_b:
        raise ValueError("quorums must be non-empty")
    if (quorum_a | quorum_b) - set(voters):
        raise ValueError("quorums must be subsets of the voter set")
    return len(quorum_a & quorum_b)


__all__ = [
    "LogEntry",
    "ReplicatedLog",
    "quorum_size",
    "majority_quorum",
    "quorum_intersection",
]
