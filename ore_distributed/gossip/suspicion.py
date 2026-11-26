"""Failure suspicion with explicit states before removal (Step 33).

Network delay and partitions can mimic failure, so a missed heartbeat moves a
peer to unreachable, then suspected, and only timeouts plus policy move it to
removed; exclusive work is never reassigned from a single missed heartbeat.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SuspicionState(str, Enum):
    """The suspicion lifecycle of one peer (Step 33)."""

    ALIVE = "alive"
    UNREACHABLE = "unreachable"
    SUSPECTED = "suspected"
    DRAINING = "draining"
    REMOVED = "removed"


@dataclass(frozen=True, slots=True)
class SuspicionTransition:
    """One recorded state transition in the suspicion lifecycle."""

    peer: str
    from_state: str
    to_state: str
    reason: str


class FailureSuspicion:
    """Track peer liveness with bounded timeouts and explicit suspicion."""

    def __init__(
        self,
        *,
        unreachable_after: int = 3,
        suspected_after: int = 6,
        removed_after: int = 12,
    ) -> None:
        """Create a tracker with monotone timeout thresholds."""
        if not 0 < unreachable_after <= suspected_after < removed_after:
            raise ValueError("timeouts must satisfy 0 < unreachable <= suspected < removed")
        self.unreachable_after = int(unreachable_after)
        self.suspected_after = int(suspected_after)
        self.removed_after = int(removed_after)
        self._missed: dict[str, int] = {}
        self._states: dict[str, SuspicionState] = {}
        self._transitions: list[SuspicionTransition] = []
        self._suspected_once: set[str] = set()

    def state_of(self, peer: str) -> SuspicionState:
        """Return the current suspicion state of a peer."""
        return self._states.get(peer, SuspicionState.ALIVE)

    def transitions(self) -> tuple[SuspicionTransition, ...]:
        """Return every recorded transition in order."""
        return tuple(self._transitions)

    def _record(self, peer: str, to_state: SuspicionState, reason: str) -> None:
        """Record a state change for audit."""
        previous = self.state_of(peer)
        self._states[peer] = to_state
        self._transitions.append(
            SuspicionTransition(
                peer=peer,
                from_state=previous.value,
                to_state=to_state.value,
                reason=reason,
            )
        )

    def record_heartbeat(self, peer: str) -> SuspicionState:
        """Record a received heartbeat; liveness resets the miss counter."""
        if not peer.strip():
            raise ValueError("peer must be non-empty")
        self._missed.pop(peer, None)
        if self.state_of(peer) is not SuspicionState.ALIVE:
            self._record(peer, SuspicionState.ALIVE, "heartbeat received")
        return self.state_of(peer)

    def record_missed_heartbeat(self, peer: str) -> SuspicionState:
        """Record one missed heartbeat and return the resulting state.

        A single miss only moves the peer to unreachable; suspicion and
        removal require sustained timeouts (Step 33).
        """
        if not peer.strip():
            raise ValueError("peer must be non-empty")
        missed = self._missed.get(peer, 0) + 1
        self._missed[peer] = missed
        current = self.state_of(peer)
        if missed >= self.removed_after and current is SuspicionState.SUSPECTED:
            self._record(peer, SuspicionState.REMOVED, f"{missed} missed heartbeats")
        elif missed >= self.suspected_after and current in {
            SuspicionState.ALIVE,
            SuspicionState.UNREACHABLE,
        }:
            self._record(peer, SuspicionState.SUSPECTED, f"{missed} missed heartbeats")
            self._suspected_once.add(peer)
        elif missed >= self.unreachable_after and current is SuspicionState.ALIVE:
            self._record(peer, SuspicionState.UNREACHABLE, f"{missed} missed heartbeats")
        return self.state_of(peer)

    def begin_draining(self, peer: str) -> SuspicionState:
        """Mark a peer as draining ahead of an authorized removal."""
        if self.state_of(peer) is SuspicionState.REMOVED:
            raise ValueError("peer already removed")
        self._record(peer, SuspicionState.DRAINING, "drain requested")
        return self.state_of(peer)

    def remove(self, peer: str, reason: str = "explicit removal") -> SuspicionState:
        """Remove a peer through policy, not mere timeout observation."""
        self._record(peer, SuspicionState.REMOVED, reason)
        return self.state_of(peer)

    def was_suspected_before_removal(self, peer: str) -> bool:
        """Return whether removal was preceded by an explicit suspicion phase."""
        return peer in self._suspected_once

    def may_reassign_exclusive_work(self, peer: str) -> bool:
        """Gate exclusive-work reassignment on the suspicion policy.

        Work reassignment requires the suspected state (or removal), never a
        single missed heartbeat; the authoritative ownership advance itself
        stays in the consensus state machine (Step 40).
        """
        return self.state_of(peer) in {SuspicionState.SUSPECTED, SuspicionState.REMOVED}


__all__ = [
    "SuspicionState",
    "SuspicionTransition",
    "FailureSuspicion",
]
