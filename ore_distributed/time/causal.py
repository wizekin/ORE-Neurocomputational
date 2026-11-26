"""Explicit causal dependency metadata (Step 35).

HLC provides a practical total-ordering key, but causality is recorded explicitly:
parent event identifiers, dependency versions, and vector-clock components
distinguish causally ordered events from concurrent ones.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence


def _non_empty(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True, slots=True)
class CausalHistory:
    """Explicit causal parents and dependency versions for one event (Step 35)."""

    event_id: str
    parents: frozenset[str] = frozenset()
    dependency_versions: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the event identity and normalize dependency metadata."""
        object.__setattr__(self, "event_id", _non_empty("event_id", self.event_id))
        object.__setattr__(self, "parents", frozenset(self.parents))
        versions = {str(k): int(v) for k, v in dict(self.dependency_versions).items()}
        if any(v < 0 for v in versions.values()):
            raise ValueError("dependency versions must be non-negative")
        object.__setattr__(self, "dependency_versions", versions)

    def includes(self, other: "CausalHistory") -> bool:
        """Return whether this history transitively includes another event."""
        return self.event_id == other.event_id or other.event_id in self.parents

    def to_dict(self) -> dict[str, object]:
        """Return the causal record as a plain dictionary."""
        return {
            "event_id": self.event_id,
            "parents": sorted(self.parents),
            "dependency_versions": dict(self.dependency_versions),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "CausalHistory":
        """Rebuild causal metadata from its dictionary form."""
        return cls(
            event_id=str(payload["event_id"]),
            parents=frozenset(str(p) for p in payload.get("parents", ())),
            dependency_versions=dict(payload.get("dependency_versions", {})),
        )


class VectorClock:
    """Per-node counters distinguishing happens-before from concurrency."""

    def __init__(self, counters: Mapping[str, int] | None = None) -> None:
        """Create a vector clock with an optional initial counter set."""
        self._counters: dict[str, int] = {
            str(node): int(value) for node, value in (counters or {}).items()
        }
        if any(value < 0 for value in self._counters.values()):
            raise ValueError("vector clock counters must be non-negative")

    def increment(self, node_id: str) -> "VectorClock":
        """Increment one node's counter for a local event and return self."""
        node_id = _non_empty("node_id", node_id)
        self._counters[node_id] = self._counters.get(node_id, 0) + 1
        return self

    def merge(self, other: "VectorClock") -> "VectorClock":
        """Fold an observed vector clock into this one (componentwise max)."""
        if not isinstance(other, VectorClock):
            raise TypeError("merge expects a VectorClock")
        for node, value in other._counters.items():
            self._counters[node] = max(self._counters.get(node, 0), value)
        return self

    def snapshot(self) -> dict[str, int]:
        """Return an independent copy of the current counters."""
        return dict(self._counters)

    def knows(self, node_id: str, at_least: int) -> bool:
        """Return whether this clock has observed a node counter value."""
        return self._counters.get(node_id, 0) >= at_least


def happens_before(left: Mapping[str, int], right: Mapping[str, int]) -> bool:
    """Return whether vector clock ``left`` causally precedes ``right``."""
    keys = set(left) | set(right)
    strictly_less = False
    for key in keys:
        a = left.get(key, 0)
        b = right.get(key, 0)
        if a > b:
            return False
        if a < b:
            strictly_less = True
    return strictly_less


def events_concurrent(left: Mapping[str, int], right: Mapping[str, int]) -> bool:
    """Return whether two vector clocks are causally concurrent."""
    return not happens_before(left, right) and not happens_before(right, left)


def missing_dependencies(
    history: CausalHistory, known_event_ids: Iterable[str]
) -> tuple[str, ...]:
    """Return parent identifiers not present in the known event set (quarantine check)."""
    known = set(known_event_ids)
    return tuple(sorted(parent for parent in history.parents if parent not in known))


def order_by_hlc_then_causal(
    events: Sequence[tuple[str, object]],
    hlc_key: str = "hlc",
    causal_key: str = "causal",
) -> tuple[str, ...]:
    """Sort event ids by HLC order key, reporting ties broken without claiming causality.

    The returned ordering is a practical total order; Step 35 explicitly warns
    that HLC ordering does not prove causality, so the caller keeps causal
    metadata separately through :class:`CausalHistory`.
    """
    def key(item: tuple[str, object]) -> tuple[int, int, str]:
        payload = item[1]
        hlc = payload.get(hlc_key) if isinstance(payload, Mapping) else None
        if isinstance(hlc, Mapping):
            return (
                int(hlc.get("physical", 0)),
                int(hlc.get("logical", 0)),
                str(hlc.get("node_id", "")),
            )
        causal = payload.get(causal_key) if isinstance(payload, Mapping) else None
        if isinstance(causal, Mapping):
            return (int(causal.get("physical", 0)), 0, "")
        return (0, 0, item[0])

    ordered = sorted(events, key=key)
    return tuple(item[0] for item in ordered)


__all__ = [
    "CausalHistory",
    "VectorClock",
    "happens_before",
    "events_concurrent",
    "missing_dependencies",
    "order_by_hlc_then_causal",
]
