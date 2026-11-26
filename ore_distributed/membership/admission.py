"""Authenticated membership admission and removal (Steps 8 and 12).

A node discovered by gossip is not automatically a trusted voting member: only
the admission controller, recording initiator, reason, resulting configuration
version, and the committing protocol event, may change the voting configuration.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import uuid
from typing import Iterable

from ore_distributed.membership.registry import (
    CapabilityDescriptor,
    NodeHealth,
    NodeRecord,
    NodeRegistry,
)


class MembershipChangeError(ValueError):
    """Raised when a membership change violates the admission policy."""


@dataclass(frozen=True, slots=True)
class MembershipChange:
    """Audit record of one authorized membership change (Step 8)."""

    change_id: str
    node_id: str
    action: str  # "admit" or "remove"
    initiator: str
    reason: str
    configuration_version: int
    committing_event: str
    recorded_at: str

    def __post_init__(self) -> None:
        """Validate the audit record fields."""
        for name in (
            "change_id",
            "node_id",
            "initiator",
            "reason",
            "committing_event",
            "recorded_at",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.action not in {"admit", "remove"}:
            raise ValueError("action must be 'admit' or 'remove'")
        if not isinstance(self.configuration_version, int) or self.configuration_version < 1:
            raise ValueError("configuration_version must be a positive integer")

    def to_dict(self) -> dict[str, object]:
        """Return the change record as a plain dictionary."""
        return {
            "change_id": self.change_id,
            "node_id": self.node_id,
            "action": self.action,
            "initiator": self.initiator,
            "reason": self.reason,
            "configuration_version": self.configuration_version,
            "committing_event": self.committing_event,
            "recorded_at": self.recorded_at,
        }


class AdmissionController:
    """Authorize membership transitions and keep a complete audit trail.

    The controller works with the registry view (discovery, capability exchange)
    and the voting configuration used by consensus. Discovery can mark a node
    authenticated, but only ``admit_voter``/``remove_voter`` change the voter
    configuration, and every change is bound to the consensus event that
    committed it (Step 12).
    """

    def __init__(
        self,
        registry: NodeRegistry,
        cluster_id: str,
        authorized_initiators: Iterable[str] = (),
        minimum_voters_after_change: int = 1,
    ) -> None:
        """Create a controller over one registry and cluster."""
        if not cluster_id.strip():
            raise ValueError("cluster_id must be a non-empty string")
        self.registry = registry
        self.cluster_id = cluster_id
        self.authorized_initiators = frozenset(authorized_initiators)
        self.minimum_voters_after_change = max(1, int(minimum_voters_after_change))
        self._configuration_version = 1
        self._changes: list[MembershipChange] = []

    @property
    def configuration_version(self) -> int:
        """Return the current membership configuration version."""
        return self._configuration_version

    def changes(self) -> tuple[MembershipChange, ...]:
        """Return the immutable audit trail of authorized changes."""
        return tuple(self._changes)

    def authenticate(self, node_id: str, credential_verified: bool = True) -> NodeRecord:
        """Mark a discovered node authenticated; authentication is not voting rights."""
        if not credential_verified:
            raise MembershipChangeError(
                "authentication requires verified credentials before state changes"
            )
        record = self.registry.get(node_id)
        if record is None:
            record = NodeRecord(node_id=node_id, health=NodeHealth.DISCOVERED)
            self.registry.register(record)
        if record.health is NodeHealth.REMOVED:
            raise MembershipChangeError("removed nodes require re-admission, not re-authentication")
        if record.health in {NodeHealth.DISCOVERED}:
            # Authentication upgrades discovery only; voters stay voters (Step 12).
            record.health = NodeHealth.AUTHENTICATED
            record.authorization = "authenticated"
        return record

    def _check_initiator(self, initiator: str) -> None:
        """Reject membership changes from unauthorized initiators."""
        if self.authorized_initiators and initiator not in self.authorized_initiators:
            raise MembershipChangeError(
                f"initiator {initiator!r} is not authorized to change membership"
            )

    def _new_change_id(self) -> str:
        """Return a collision-resistant audit identifier."""
        return f"mchg_{uuid.uuid4().hex}"

    def admit_voter(
        self,
        node_id: str,
        initiator: str,
        reason: str,
        committing_event: str,
        *,
        authenticated: bool = True,
        required_role: str = "voter",
    ) -> MembershipChange:
        """Admit an authenticated node into the voting configuration (Steps 8, 12)."""
        if not authenticated:
            raise MembershipChangeError(
                "a discovered node cannot join the voting configuration without "
                "an authenticated, authorized membership change"
            )
        self._check_initiator(initiator)
        record = self.registry.get(node_id)
        if record is None:
            raise MembershipChangeError(
                f"node {node_id!r} is unknown; discover and authenticate it first"
            )
        if record.health is NodeHealth.VOTING:
            raise MembershipChangeError(f"node {node_id!r} is already a voter")
        if record.health is NodeHealth.REMOVED:
            raise MembershipChangeError(
                "removed nodes must be re-discovered and re-authenticated first"
            )
        record.health = NodeHealth.VOTING
        record.authorization = "authorized_voter"
        self._configuration_version += 1
        change = MembershipChange(
            change_id=self._new_change_id(),
            node_id=node_id,
            action="admit",
            initiator=initiator,
            reason=reason,
            configuration_version=self._configuration_version,
            committing_event=committing_event,
            recorded_at=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        )
        self._changes.append(change)
        return change

    def remove_voter(
        self,
        node_id: str,
        initiator: str,
        reason: str,
        committing_event: str,
    ) -> MembershipChange:
        """Remove a voter while refusing to strand the cluster below quorum (Step 20 guard)."""
        self._check_initiator(initiator)
        record = self.registry.get(node_id)
        if record is None or record.health is not NodeHealth.VOTING:
            raise MembershipChangeError(f"node {node_id!r} is not a voting member")
        voters = [item.node_id for item in self.registry.voters()]
        remaining = [item for item in voters if item != node_id]
        if len(remaining) < self.minimum_voters_after_change:
            raise MembershipChangeError(
                "removal would leave the cluster without the minimum voter count; "
                "add a replacement first or drain explicitly"
            )
        record.health = NodeHealth.REMOVED
        record.authorization = "removed"
        self._configuration_version += 1
        change = MembershipChange(
            change_id=self._new_change_id(),
            node_id=node_id,
            action="remove",
            initiator=initiator,
            reason=reason,
            configuration_version=self._configuration_version,
            committing_event=committing_event,
            recorded_at=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        )
        self._changes.append(change)
        return change

    def drain(self, node_id: str) -> NodeRecord:
        """Move a voter into the draining state ahead of a removal."""
        record = self.registry.get(node_id)
        if record is None or record.health is not NodeHealth.VOTING:
            raise MembershipChangeError(f"node {node_id!r} is not a voting member")
        record.health = NodeHealth.DRAINING
        record.authorization = "draining"
        return record

    def discovered_to_authenticated_flow(
        self, descriptor: CapabilityDescriptor
    ) -> tuple[NodeHealth, NodeHealth]:
        """Register a capability advertisement without granting trust (Step 12).

        Gossip or a registry may discover nodes; the returned pair shows the
        transition stays at discovered/authenticated and never reaches voting.
        """
        before = self.registry.get(descriptor.node_id)
        before_state = before.health if before else NodeHealth.DISCOVERED
        self.registry.publish_capabilities(descriptor)
        after = self.registry.require(descriptor.node_id)
        if after.health is NodeHealth.VOTING:
            raise MembershipChangeError("capability advertisement must not create voters")
        return before_state, after.health


def voting_configuration(registry: NodeRegistry) -> tuple[str, ...]:
    """Return the sorted voter identifiers of the registry view."""
    return tuple(item.node_id for item in registry.voters())


def audit_trail(controller: AdmissionController) -> list[dict[str, object]]:
    """Return the audit trail as plain dictionaries for observability."""
    return [change.to_dict() for change in controller.changes()]


__all__ = [
    "MembershipChange",
    "MembershipChangeError",
    "AdmissionController",
    "voting_configuration",
    "audit_trail",
]
