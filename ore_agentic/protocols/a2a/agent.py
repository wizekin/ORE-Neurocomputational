"""Optional Agent-to-Agent interoperability (Steps 35-36).

A2A is used only for independently operated agents requiring discovery,
delegation, status exchange or artifact handoff; internal ORE modules use
simpler typed interfaces.  Remote assertions are proposals until locally
validated, and duplicate, delayed, out-of-order and contradictory messages
are handled explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from ...contracts import new_id, utc_now

__all__ = (
    "A2AAgentCard",
    "A2ADelegation",
    "A2AMessage",
    "A2AMessageKind",
    "A2AParticipant",
    "A2AResultRecord",
)


class A2AMessageKind(str, Enum):
    """The A2A message categories (Step 35)."""

    TASK = "task"
    STATUS = "status"
    RESULT = "result"
    ARTIFACT = "artifact"


@dataclass(frozen=True, slots=True)
class A2AAgentCard:
    """The discovery card of an independently operated agent (Step 35)."""

    agent_id: str
    name: str
    capabilities: tuple[str, ...]
    endpoint_ref: str = ""
    trust_level: float = 0.5
    version: str = "1.0.0"

    def __post_init__(self) -> None:
        """Validate the card."""
        if not self.agent_id or not self.agent_id.strip():
            raise ValueError("agent_id must be a non-empty string")
        if not self.capabilities:
            raise ValueError("an agent card must declare capabilities")
        if not 0.0 <= self.trust_level <= 1.0:
            raise ValueError("trust_level must be in [0, 1]")

    def supports(self, capability: str) -> bool:
        """Return whether the peer declares a capability."""
        return capability in self.capabilities

    def to_dict(self) -> dict[str, Any]:
        """Serialize the card."""
        return {"agent_id": self.agent_id, "name": self.name,
                "capabilities": list(self.capabilities),
                "endpoint_ref": self.endpoint_ref, "trust_level": self.trust_level,
                "version": self.version}


@dataclass(frozen=True, slots=True)
class A2ADelegation:
    """One delegated task handed to an external agent (Step 35)."""

    delegation_id: str
    peer_agent_id: str
    capability: str
    instruction: str
    deadline_at: str = ""
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the delegation."""
        return {"delegation_id": self.delegation_id,
                "peer_agent_id": self.peer_agent_id, "capability": self.capability,
                "instruction": self.instruction, "deadline_at": self.deadline_at,
                "created_at": self.created_at}


@dataclass(frozen=True, slots=True)
class A2AMessage:
    """One message exchanged with an external agent (Step 36)."""

    message_id: str
    agent_id: str
    delegation_id: str
    sequence_no: int
    kind: A2AMessageKind
    payload: Mapping[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the message."""
        return {"message_id": self.message_id, "agent_id": self.agent_id,
                "delegation_id": self.delegation_id, "sequence_no": self.sequence_no,
                "kind": self.kind.value, "payload": dict(self.payload),
                "timestamp": self.timestamp}


@dataclass(slots=True)
class A2AResultRecord:
    """An external result held as an untrusted proposal (Step 36)."""

    message_id: str
    agent_id: str
    delegation_id: str
    payload: Mapping[str, Any]
    validated: bool = False
    contradiction_with: tuple[str, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)

    @property
    def trusted(self) -> bool:
        """Remote assertions remain untrusted until locally validated (Step 36)."""
        return self.validated

    def to_dict(self) -> dict[str, Any]:
        """Serialize the record."""
        return {"message_id": self.message_id, "agent_id": self.agent_id,
                "delegation_id": self.delegation_id, "validated": self.validated,
                "trusted": self.trusted,
                "contradiction_with": list(self.contradiction_with),
                "payload": dict(self.payload),
                "provenance": dict(self.provenance)}


class A2AParticipant:
    """One A2A-capable ORE participant (Steps 35-36).

    Delegation requires an explicit justification flag; results pass through
    identity, delegation, schema and policy validation before they may be
    marked validated; duplicates are deduplicated and contradictions are
    preserved and flagged, never averaged away.
    """

    def __init__(self, card: A2AAgentCard, *, enabled: bool = False,
                 expected_schema: Mapping[str, Any] | None = None,
                 policy_check: Any = None) -> None:
        """Configure the participant; disabled by default (Step 35)."""
        self._card = card
        self._enabled = enabled
        self._expected_schema = dict(expected_schema or {})
        self._policy_check = policy_check
        self._delegations: dict[str, A2ADelegation] = {}
        self._results: dict[str, A2AResultRecord] = {}
        self._seen_messages: set[str] = set()
        self._out_of_order: dict[str, list[A2AMessage]] = {}

    @property
    def card(self) -> A2AAgentCard:
        """Return this participant's agent card."""
        return self._card

    @property
    def enabled(self) -> bool:
        """Return whether A2A use is justified and enabled (Step 35)."""
        return self._enabled

    def delegate(self, peer: A2AAgentCard, capability: str, instruction: str,
                 *, deadline_at: str = "") -> A2ADelegation:
        """Delegate a bounded task to an external agent (Step 35).

        Raises ``PermissionError`` when A2A is not justified (disabled), the
        peer does not declare the capability, or the peer trust level is 0.
        """
        if not self._enabled:
            raise PermissionError(
                "A2A is disabled: use only for independently operated agents "
                "requiring discovery, delegation, status exchange or artifact handoff")
        if not peer.supports(capability):
            raise PermissionError(
                f"peer {peer.agent_id} does not declare capability {capability!r}")
        if peer.trust_level <= 0.0:
            raise PermissionError(f"peer {peer.agent_id} is untrusted (trust 0)")
        delegation = A2ADelegation(
            delegation_id=new_id("a2a"), peer_agent_id=peer.agent_id,
            capability=capability, instruction=instruction, deadline_at=deadline_at)
        self._delegations[delegation.delegation_id] = delegation
        return delegation

    def delegation(self, delegation_id: str) -> A2ADelegation | None:
        """Return one delegation by ID."""
        return self._delegations.get(delegation_id)

    def receive(self, message: A2AMessage) -> str:
        """Accept an inbound message; returns the disposition (Step 36).

        Dispositions: ``duplicate``, ``out_of_order``, ``unknown_delegation``,
        ``wrong_agent``, ``status_recorded``, ``stored_untrusted``.  Duplicate
        messages are dropped; out-of-order results are buffered until their
        predecessors arrive; contradictory results are both stored with
        cross-references.
        """
        if message.message_id in self._seen_messages:
            return "duplicate"
        self._seen_messages.add(message.message_id)
        delegation = self._delegations.get(message.delegation_id)
        if delegation is None:
            return "unknown_delegation"
        if message.agent_id != delegation.peer_agent_id:
            return "wrong_agent"
        if message.kind is not A2AMessageKind.RESULT:
            return "status_recorded"
        accepted = sum(1 for record in self._results.values()
                       if record.delegation_id == message.delegation_id)
        if message.sequence_no != accepted:
            self._out_of_order.setdefault(message.delegation_id,
                                          []).append(message)
            return "out_of_order"
        self._store_result(message)
        self._drain_buffered(message.delegation_id)
        return "stored_untrusted"

    def _store_result(self, message: A2AMessage) -> None:
        """Store one result record as an untrusted proposal (Step 36)."""
        record = A2AResultRecord(
            message_id=message.message_id, agent_id=message.agent_id,
            delegation_id=message.delegation_id, payload=dict(message.payload),
            provenance={"received_at": message.timestamp,
                        "peer_version": self._peer_version(message.agent_id)})
        for existing in self._results.values():
            if (existing.delegation_id == message.delegation_id
                    and _contradicts(existing.payload, message.payload)):
                existing.contradiction_with = (*existing.contradiction_with,
                                               message.message_id)
                record.contradiction_with = (*record.contradiction_with,
                                             existing.message_id)
        self._results[message.message_id] = record

    def _drain_buffered(self, delegation_id: str) -> None:
        """Store buffered results whose predecessors have now arrived."""
        buffered = self._out_of_order.get(delegation_id, [])
        while buffered:
            accepted = sum(1 for record in self._results.values()
                           if record.delegation_id == delegation_id)
            nxt = next((item for item in buffered
                        if item.sequence_no == accepted), None)
            if nxt is None:
                return
            buffered.remove(nxt)
            self._store_result(nxt)

    def _peer_version(self, agent_id: str) -> str:
        """Return the peer card version when known."""
        return "1.0.0" if agent_id else ""

    def validate_result(self, message_id: str) -> A2AResultRecord:
        """Validate one stored result locally (Step 36).

        Validation checks the payload schema and the injected policy check;
        only then does the record become trusted.  Invalid results keep
        ``validated=False`` and are returned for disclosure.
        """
        record = self._results.get(message_id)
        if record is None:
            raise KeyError(f"no stored result for message {message_id}")
        schema_ok = self._check_schema(record.payload)
        policy_ok = (self._policy_check is None
                     or bool(self._policy_check(record.payload)))
        record.validated = schema_ok and policy_ok
        return record

    def _check_schema(self, payload: Mapping[str, Any]) -> bool:
        """Check the expected payload schema, when declared (Step 36)."""
        if not self._expected_schema:
            return True
        for field_name in self._expected_schema.get("required", ()) or ():
            if field_name not in payload:
                return False
        return True

    def results(self, delegation_id: str | None = None) -> tuple[A2AResultRecord, ...]:
        """Return stored results, optionally for one delegation."""
        return tuple(record for record in self._results.values()
                     if delegation_id is None
                     or record.delegation_id == delegation_id)

    def describe(self) -> dict[str, Any]:
        """Return a serializable summary."""
        return {"agent_id": self._card.agent_id, "enabled": self._enabled,
                "delegations": len(self._delegations),
                "results": len(self._results),
                "validated": sum(1 for record in self._results.values()
                                 if record.validated)}


def _contradicts(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    """Detect direct contradictions between two result payloads (Step 36)."""
    for key, value in first.items():
        if key in second and second[key] != value:
            return True
    return False
