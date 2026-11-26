"""Hybrid Logical Clock implementation (Step 34).

An HLC is the triple (physical component, logical counter, node identifier). The
merge algorithm keeps timestamps monotonic and preserves observed ordering even
when physical clocks drift. HLC ordering is useful metadata; it does not by itself
prove causality, which is why causal parents are recorded separately.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Mapping


def _non_empty(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True, slots=True, order=True)
class HLCTimestamp:
    """One Hybrid Logical Clock timestamp (physical, logical, node)."""

    physical: int
    logical: int = 0
    node_id: str = ""

    def __post_init__(self) -> None:
        """Validate components and default the node identifier."""
        if not isinstance(self.physical, int) or self.physical < 0:
            raise ValueError("physical component must be a non-negative integer")
        if not isinstance(self.logical, int) or self.logical < 0:
            raise ValueError("logical component must be a non-negative integer")
        if not isinstance(self.node_id, str):
            raise ValueError("node_id must be a string")

    @property
    def order_key(self) -> tuple[int, int, str]:
        """Return the total-order comparison key (physical, logical, node)."""
        return (self.physical, self.logical, self.node_id)

    def to_dict(self) -> dict[str, Any]:
        """Return the timestamp as a plain dictionary."""
        return {
            "physical": self.physical,
            "logical": self.logical,
            "node_id": self.node_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HLCTimestamp":
        """Rebuild a timestamp from its dictionary form."""
        return cls(
            physical=int(payload["physical"]),
            logical=int(payload.get("logical", 0)),
            node_id=str(payload.get("node_id", "")),
        )


class HybridLogicalClock:
    """Maintain one node's Hybrid Logical Clock following the merge algorithm.

    ``tick`` stamps a local event: the physical component advances with wall time
    (or the last observed value), incrementing the logical counter when wall time
    did not move. ``merge`` stamps a receive event: physical becomes the maximum
    of local, message, and wall time, and logical becomes one past the max of the
    local and message counters when they tie.
    """

    def __init__(self, node_id: str, wall_clock=time.time_ns) -> None:
        """Create a clock for one node with an injectable wall-clock function."""
        self.node_id = _non_empty("node_id", node_id)
        self._wall_clock = wall_clock
        self._physical = self._wall_clock()
        self._logical = 0

    @property
    def now(self) -> HLCTimestamp:
        """Return the current clock reading without advancing it."""
        return HLCTimestamp(self._physical, self._logical, self.node_id)

    def tick(self) -> HLCTimestamp:
        """Stamp a local event and keep the clock strictly monotonic."""
        wall = self._wall_clock()
        if wall > self._physical:
            self._physical = wall
            self._logical = 0
        else:
            self._logical += 1
        return HLCTimestamp(self._physical, self._logical, self.node_id)

    def merge(self, remote: HLCTimestamp) -> HLCTimestamp:
        """Stamp a receive event against a remote timestamp using the HLC merge."""
        if not isinstance(remote, HLCTimestamp):
            raise TypeError("merge expects an HLCTimestamp")
        wall = self._wall_clock()
        physical = max(self._physical, remote.physical, wall)
        if physical == self._physical and physical == remote.physical:
            logical = max(self._logical, remote.logical) + 1
        elif physical == self._physical:
            logical = self._logical + 1
        elif physical == remote.physical:
            logical = remote.logical + 1
        else:
            logical = 0
        self._physical = physical
        self._logical = logical
        return HLCTimestamp(self._physical, self._logical, self.node_id)

    def observe(self, remote: HLCTimestamp) -> HLCTimestamp:
        """Record an observed remote timestamp without generating a local event."""
        if remote.order_key > self.now.order_key:
            self._physical = remote.physical
            self._logical = remote.logical + 1
        return self.now

    def to_state(self) -> dict[str, Any]:
        """Return persisted clock state; restoring it prevents moving backward."""
        return {"physical": self._physical, "logical": self._logical, "node_id": self.node_id}

    def restore_state(self, state: Mapping[str, Any]) -> None:
        """Restore persisted state so the clock never moves backward across restarts."""
        physical = int(state["physical"])
        logical = int(state.get("logical", 0))
        node_id = str(state.get("node_id", self.node_id))
        if node_id != self.node_id:
            raise ValueError("persisted clock state belongs to a different node")
        if physical > self._physical:
            self._physical = physical
            self._logical = logical
        elif physical == self._physical and logical > self._logical:
            self._logical = logical


def monotonic_restart_state(clock: HybridLogicalClock, previous: HLCTimestamp) -> HLCTimestamp:
    """Fold a persisted timestamp into a restarted clock (Step 34 persistence)."""
    clock.observe(previous)
    return clock.tick()


def persist_clock(clock: HybridLogicalClock, path: str | Path) -> None:
    """Persist clock state so restarts cannot move the clock backward."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(clock.to_state()), encoding="utf-8")


def load_clock_state(path: str | Path) -> dict[str, Any] | None:
    """Load persisted clock state, returning None when no state exists yet."""
    file_path = Path(path)
    if not file_path.is_file():
        return None
    payload: dict[str, Any] = json.loads(file_path.read_text(encoding="utf-8"))
    return payload


__all__ = [
    "HLCTimestamp",
    "HybridLogicalClock",
    "monotonic_restart_state",
    "persist_clock",
    "load_clock_state",
]
