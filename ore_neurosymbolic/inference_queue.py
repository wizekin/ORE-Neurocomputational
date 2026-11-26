"""Bounded, deduplicating inference queue (Steps B7, Section 2).

Forward chaining processes new facts through this queue. Every entry is
content-addressed, duplicates are silently absorbed (deduplication is a
correctness requirement of the fixed-point equation F_{k+1} = F_k ∪ T_R(F_k)),
and a hard size budget guarantees the loop can never explode (Section 2:
"Bound rule fires, recursion, depth, candidate count").
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .contracts import SemanticFact


@dataclass(frozen=True, slots=True)
class QueueEntry:
    """One queued derivation awaiting propagation."""

    fact: SemanticFact
    origin_rule_ref: tuple[str, str] | None
    depth: int

    def key(self) -> str:
        """Return the content key used for deduplication."""
        return self.fact.fingerprint()


class BoundedInferenceQueue:
    """A FIFO queue with content deduplication and a hard capacity budget."""

    def __init__(self, max_size: int = 4096) -> None:
        """Create an empty bounded queue."""
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        self.max_size = max_size
        self._entries: list[QueueEntry] = []
        self._seen: set[str] = set()
        self.duplicates_absorbed: int = 0
        self.overflow_count: int = 0

    def push(self, entry: QueueEntry) -> bool:
        """Push one derivation; return True when accepted, False when absorbed."""
        key = entry.key()
        if key in self._seen:
            self.duplicates_absorbed += 1
            return False
        if len(self._entries) >= self.max_size:
            self.overflow_count += 1
            return False
        self._seen.add(key)
        self._entries.append(entry)
        return True

    def pop(self) -> QueueEntry | None:
        """Pop the oldest entry, or None when the queue is empty."""
        if not self._entries:
            return None
        entry = self._entries.pop(0)
        return entry

    def __len__(self) -> int:
        """Return the current queue depth."""
        return len(self._entries)

    def stats(self) -> dict[str, int]:
        """Return the deduplication and overflow counters for observability."""
        return {
            "depth": len(self._entries),
            "duplicates_absorbed": self.duplicates_absorbed,
            "overflow_count": self.overflow_count,
        }


__all__ = ["BoundedInferenceQueue", "QueueEntry"]
