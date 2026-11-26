"""Model lifecycle events and approvals (Point 14, Steps 51-56).

Lifecycle transitions are append-only events: corrections become new linked events,
never in-place rewrites.  Every event carries the model identity, the task link when
one exists, the acting authority, and the evidence it was based on.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any

from .manifests import LifecycleState

__all__ = ("ApprovalRecord", "LifecycleEventType", "ModelLifecycleEvent")


def _nonempty(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


class LifecycleEventType(str, Enum):
    """Auditable lifecycle transitions (Step 54)."""

    REGISTERED = "registered"
    REJECTED = "rejected"
    VALIDATED = "validated"
    CANARY_STARTED = "canary_started"
    ACTIVATED = "activated"
    DEPRECATED = "deprecated"
    QUARANTINED = "quarantined"
    ROLLED_BACK = "rolled_back"
    REEVALUATED = "reevaluated"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_GRANTED = "approval_granted"
    PROMOTED = "promoted"
    RETIRED = "retired"
    REVOKED = "revoked"


@dataclass(frozen=True, slots=True)
class ApprovalRecord:
    """Step 55 -- explicit human approval for a sensitive transition."""

    approval_id: str
    approver: str
    scope: str
    reviewed_evidence: tuple[str, ...] = ()
    decision: str = "approved"
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        """Validate the approval record."""
        object.__setattr__(self, "approval_id", _nonempty("approval_id", self.approval_id))
        object.__setattr__(self, "approver", _nonempty("approver", self.approver))
        object.__setattr__(self, "scope", _nonempty("scope", self.scope))
        if self.decision not in ("approved", "rejected"):
            raise ValueError("decision must be approved/rejected")

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {"approval_id": self.approval_id, "approver": self.approver,
                "scope": self.scope,
                "reviewed_evidence": list(self.reviewed_evidence),
                "decision": self.decision, "timestamp": self.timestamp}


@dataclass(frozen=True, slots=True)
class ModelLifecycleEvent:
    """One append-only lifecycle transition record (Steps 53-54)."""

    event_id: str
    event_type: LifecycleEventType
    model_id: str
    model_version: str
    from_state: LifecycleState | None
    to_state: LifecycleState
    task_id: str = ""
    actor: str = "ore_learning.lifecycle"
    reason: str = ""
    evidence_refs: tuple[str, ...] = ()
    sequence: int = 0
    timestamp: float = 0.0
    approval: ApprovalRecord | None = None

    def __post_init__(self) -> None:
        """Validate the event record."""
        object.__setattr__(self, "event_id", _nonempty("event_id", self.event_id))
        object.__setattr__(self, "model_id", _nonempty("model_id", self.model_id))
        object.__setattr__(self, "model_version",
                           _nonempty("model_version", self.model_version))
        if not math.isfinite(float(self.timestamp)) or float(self.timestamp) < 0.0:
            raise ValueError("timestamp must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {"event_id": self.event_id, "event_type": self.event_type.value,
                "model_id": self.model_id, "model_version": self.model_version,
                "from_state": self.from_state.value if self.from_state else None,
                "to_state": self.to_state.value, "task_id": self.task_id,
                "actor": self.actor, "reason": self.reason,
                "evidence_refs": list(self.evidence_refs),
                "sequence": self.sequence, "timestamp": self.timestamp,
                "approval": self.approval.to_dict() if self.approval else None}


#: Transitions that require recorded human approval when the task risk class is high.
HIGH_RISK_TRANSITIONS: frozenset[str] = frozenset({
    LifecycleEventType.PROMOTED.value,
    LifecycleEventType.ACTIVATED.value,
})
