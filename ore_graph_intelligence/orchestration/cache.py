"""Bounded inference cache keyed by snapshot hash and model version (step 51).

Caches are keyed by (snapshot hash, model version) so a snapshot or model change can
never serve stale entries (acceptance test 24). Capacity is bounded with
least-recently-used eviction; expansion, candidates, batches, retries and timeouts all
stay bounded through the cache's own limits.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class CacheKey:
    """Composite cache identity: snapshot hash, model version, namespace."""

    snapshot_hash: str
    model_version: str
    namespace: str

    def __post_init__(self) -> None:
        for name in ("snapshot_hash", "model_version", "namespace"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be non-empty")


class GraphResultCache:
    """LRU cache bounded by entry count, keyed by snapshot hash and model version."""

    def __init__(self, *, capacity: int = 128) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self.capacity = int(capacity)
        self._entries: OrderedDict[CacheKey, Any] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.invalidations = 0

    def get(self, key: CacheKey) -> Any | None:
        """Return the cached value or None; records hit/miss statistics."""
        if key in self._entries:
            self._entries.move_to_end(key)
            self.hits += 1
            return self._entries[key]
        self.misses += 1
        return None

    def put(self, key: CacheKey, value: Any) -> None:
        """Store one value; evict the least-recently-used entry when over capacity."""
        self._entries[key] = value
        self._entries.move_to_end(key)
        while len(self._entries) > self.capacity:
            self._entries.popitem(last=False)

    def invalidate(self, key: CacheKey) -> bool:
        """Drop one exact key; returns whether it was present."""
        if key in self._entries:
            del self._entries[key]
            self.invalidations += 1
            return True
        return False

    def invalidate_stale(
        self,
        *,
        snapshot_hash: str | None = None,
        model_version: str | None = None,
    ) -> int:
        """Invalidate entries whose snapshot hash or model version is stale (test 24)."""
        stale = [
            key for key in self._entries
            if (snapshot_hash is not None and key.snapshot_hash != snapshot_hash)
            or (model_version is not None and key.model_version != model_version)
        ]
        for key in stale:
            self.invalidate(key)
        return len(stale)

    def snapshot_keys(self) -> tuple[CacheKey, ...]:
        """Return current keys in least-to-most-recently-used order."""
        return tuple(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def stats(self) -> dict[str, object]:
        return {
            "capacity": self.capacity, "entries": len(self._entries),
            "hits": self.hits, "misses": self.misses,
            "invalidations": self.invalidations,
        }


__all__ = ["CacheKey", "GraphResultCache"]
