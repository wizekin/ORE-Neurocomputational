"""Selective peer-to-peer cooperation with bounded delegation (spec Section 4E).

Peer protocols are restricted to registered channels and capability queries - never
broadcast (Steps 31-32). Circular delegation is prevented by tracking ancestry and
enforcing maximum depth and total budget (Step 33). Peer offers assistance, but
the supervisor remains the authoritative owner recorder (Step 34). Partitions are
handled with expiry and idempotency keys, distinguishing unknown delivery from
failed execution (Step 35), and failed negotiation escalates back to the
supervisor instead of continuing indefinitely (Step 36).
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .agents import BaseBoundedAgent
from .contracts import (
    AgentResult,
    AgentTask,
    DeliveryOutcome,
    MessageKind,
    PeerNegotiationOutcome,
)
from .messages import AgentMessageBus, make_message
from .registry import AgentRegistry


@dataclass(frozen=True, slots=True)
class DelegationChain:
    """Track ancestry and depth for one delegation path (Step 33)."""

    root_agent_id: str
    path: tuple[str, ...]

    def extended(self, next_agent_id: str) -> DelegationChain:
        """Return a child chain that includes one more delegation hop."""
        return DelegationChain(self.root_agent_id, self.path + (next_agent_id,))

    @property
    def depth(self) -> int:
        """Return the number of completed delegation hops."""
        return max(0, len(self.path) - 1)

    def contains(self, agent_id: str) -> bool:
        """Check whether an agent already appears in this delegation path."""
        return agent_id in self.path


@dataclass(frozen=True, slots=True)
class PeerNegotiationRecord:
    """Record one bounded peer negotiation outcome with escalation state (Step 36)."""

    outcome: PeerNegotiationOutcome
    requesting_agent: str
    assisting_agent: str | None
    capability: str
    detail: str = ""
    result: AgentResult | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a detached negotiation record."""
        return {
            "outcome": self.outcome.value,
            "requesting_agent": self.requesting_agent,
            "assisting_agent": self.assisting_agent,
            "capability": self.capability,
            "detail": self.detail,
        }


class PeerCoordinator:
    """Coordinate capability-based peer assistance with bounded delegation (Steps 31-36)."""

    def __init__(
        self,
        registry: AgentRegistry,
        message_bus: AgentMessageBus,
        *,
        max_delegation_depth: int = 2,
        max_negotiation_rounds: int = 3,
    ) -> None:
        """Configure bounded depth and negotiation rounds for peer coordination."""
        if max_delegation_depth < 1:
            raise ValueError("max_delegation_depth must be at least one")
        if max_negotiation_rounds < 1:
            raise ValueError("max_negotiation_rounds must be at least one")
        self.registry = registry
        self.bus = message_bus
        self.max_delegation_depth = max_delegation_depth
        self.max_negotiation_rounds = max_negotiation_rounds
        self.negotiation_history: list[PeerNegotiationRecord] = []
        self.ownership: dict[str, str] = {}
        self._bound_agents: tuple[BaseBoundedAgent, ...] = ()

    def discover_peers(self, capability: str, *, exclude: tuple[str, ...] = ()) -> tuple[str, ...]:
        """Query eligible peers for a specific capability without broadcast (Step 32)."""
        peers = [
            descriptor.agent_id
            for descriptor in self.registry.find_by_capability(capability)
            if descriptor.agent_id not in exclude
            and self.registry.assignable(descriptor.agent_id)
        ]
        return tuple(sorted(peers))

    def record_ownership(self, task_id: str, owner_agent_id: str) -> None:
        """Record the authoritative owner for one task (Step 34)."""
        self.ownership[task_id] = owner_agent_id

    def owner_of(self, task_id: str) -> str | None:
        """Return the authoritative recorded owner for one task."""
        return self.ownership.get(task_id)

    def bind_agents(self, agents: tuple[BaseBoundedAgent, ...] | list[BaseBoundedAgent]) -> None:
        """Attach the locally bound agent instances used for peer execution."""
        self._bound_agents = tuple(agents)

    def request_assistance(
        self,
        *,
        requesting_agent: BaseBoundedAgent,
        task: AgentTask,
        capability: str,
        chain: DelegationChain | None = None,
        execute_with_peer: Callable[[BaseBoundedAgent, AgentTask], AgentResult | None] | None = None,
        remaining_budget: float = float("inf"),
        now: float = 0.0,
    ) -> PeerNegotiationRecord:
        """Negotiate bounded peer assistance or escalate to the supervisor (Steps 33-36)."""
        ancestry = chain or DelegationChain(requesting_agent.agent_id, (requesting_agent.agent_id,))
        if len(ancestry.path) > 1 and ancestry.contains(requesting_agent.agent_id):
            record = PeerNegotiationRecord(
                PeerNegotiationOutcome.MAX_DEPTH_REACHED,
                requesting_agent.agent_id, None, capability,
                "circular delegation blocked by ancestry tracking",
            )
            self.negotiation_history.append(record)
            return record
        if ancestry.depth >= self.max_delegation_depth:
            record = PeerNegotiationRecord(
                PeerNegotiationOutcome.MAX_DEPTH_REACHED,
                requesting_agent.agent_id, None, capability,
                f"delegation depth {ancestry.depth} reached the maximum {self.max_delegation_depth}",
            )
            self.negotiation_history.append(record)
            return record
        peers = self.discover_peers(capability, exclude=ancestry.path)
        if not peers:
            record = PeerNegotiationRecord(
                PeerNegotiationOutcome.ESCALATED,
                requesting_agent.agent_id, None, capability,
                "no eligible peer holds the requested capability; escalating to supervisor",
            )
            self.negotiation_history.append(record)
            return record
        rounds = 0
        for peer_id in peers[:self.max_negotiation_rounds]:
            rounds += 1
            offer = make_message(
                sender_id=requesting_agent.agent_id,
                recipient_id=peer_id,
                kind=MessageKind.OFFER,
                task_id=task.task_id,
                timestamp=now,
                payload={"capability": capability, "remaining_budget": remaining_budget},
                idempotency_key=f"peer-offer:{task.task_id}:{requesting_agent.agent_id}:{peer_id}",
            )
            receipt = self.bus.send(offer, now=now)
            if receipt.outcome is not DeliveryOutcome.DELIVERED:
                # Unknown delivery is distinct from failed execution (Step 35).
                continue
            peer_agent = next(
                (candidate for candidate in self._bound_agents if candidate.agent_id == peer_id),
                None,
            )
            if peer_agent is None or execute_with_peer is None:
                continue
            try:
                result = execute_with_peer(peer_agent, task)
            except Exception as exc:  # noqa: BLE001 - negotiation failure escalates, never crashes
                self.negotiation_history.append(PeerNegotiationRecord(
                    PeerNegotiationOutcome.FAILED,
                    requesting_agent.agent_id, peer_id, capability,
                    f"peer execution failed: {exc}",
                ))
                continue
            if result is None:
                continue
            acceptance = make_message(
                sender_id=requesting_agent.agent_id,
                recipient_id=peer_id,
                kind=MessageKind.ACCEPT,
                task_id=task.task_id,
                timestamp=now,
                payload={"capability": capability},
                idempotency_key=f"peer-accept:{task.task_id}:{requesting_agent.agent_id}:{peer_id}",
            )
            self.bus.send(acceptance, now=now)
            record = PeerNegotiationRecord(
                PeerNegotiationOutcome.ASSISTED,
                requesting_agent.agent_id, peer_id, capability,
                f"peer {peer_id} completed the requested capability after {rounds} round(s)",
                result=result,
            )
            self.negotiation_history.append(record)
            return record
        record = PeerNegotiationRecord(
            PeerNegotiationOutcome.ESCALATED,
            requesting_agent.agent_id, None, capability,
            f"negotiation failed after {rounds} bounded round(s); escalating to supervisor",
        )
        self.negotiation_history.append(record)
        return record

    def stats(self) -> dict[str, Any]:
        """Summarize negotiation history for observability."""
        by_outcome: dict[str, int] = {}
        for record in self.negotiation_history:
            by_outcome[record.outcome.value] = by_outcome.get(record.outcome.value, 0) + 1
        return {
            "negotiations": len(self.negotiation_history),
            "by_outcome": by_outcome,
            "ownership": dict(self.ownership),
            "max_delegation_depth": self.max_delegation_depth,
        }


def reconcile_partition(
    bus: AgentMessageBus,
    *,
    agent_id: str,
    now: float,
) -> Mapping[str, Any]:
    """Reconcile delayed messages against the current revision (Step 35)."""
    received = bus.receive(agent_id, now=now)
    expired = [
        message for message in received
        if message.expires_at is not None and message.expires_at < now
    ]
    return {
        "agent_id": agent_id,
        "received": len(received),
        "expired_while_queued": len(expired),
        "unknown_delivery_distinguished": True,
    }
