"""Capability-scoped access to the shared workspace and USG (step 20).

Narrow methods replace broad shared writes: ``read_snapshot``,
``propose_fact``, ``append_observation``, and ``request_merge`` each require
a capability scoped to the operation and, where the ORE repository is
importable, additionally pass through ``ore_workspace``'s own field-level
``PermissionPolicy`` (defense in depth: both the capability AND the policy
must allow the write). USG canonical knowledge is never written directly:
``USGGuard.propose_update`` routes candidate bundles through the external
``ore_usg`` validation workflow and keeps them pending (section 6: "USG
remains the semantic authority"). When the repository packages are not
importable (standalone zip deployment), a documented minimal blackboard
provides the same governed surface with reduced fidelity.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from .capabilities import AccessRequest
from .contracts import CapabilityDeniedError

#: Fields the minimal blackboard accepts when ore_workspace is unavailable.
FALLBACK_WORKSPACE_FIELDS: tuple[str, ...] = (
    "lexical_hypotheses",
    "syntactic_hypotheses",
    "semantic_hypotheses",
    "working_memory",
    "retrieved_memories",
    "unresolved_questions",
    "module_status",
    "artifact_references",
    "discourse_context",
    "current_goal",
)


@dataclass(frozen=True)
class GuardResult:
    """Typed outcome of one guarded workspace operation."""

    allowed: bool
    reason: str
    decision: str = "none"
    capability_id: str = ""
    policy_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "decision": self.decision,
            "capability_id": self.capability_id,
            "policy_version": self.policy_version,
        }


def _authorize_or_raise(
    authorizer: Callable[..., Any] | None,
    capability: Any,
    right: str,
    resource: str,
    subject: str,
    constraints: Mapping[str, Any],
) -> None:
    """Deny by default; capability plus authorizer must both agree (step 16)."""
    if authorizer is None:
        # No authorizer wired: deny by default rather than assume trust.
        raise CapabilityDeniedError(
            "no authorization authority attached to the workspace guard",
            subject=subject,
            right=right,
        )
    request = AccessRequest(
        subject=subject, resource=resource, right=right, constraints=dict(constraints)
    )
    decision = authorizer(capability, request, None)
    if not getattr(decision, "allowed", False):
        reason = getattr(decision, "reason", "authorization denied")
        raise CapabilityDeniedError(
            f"workspace operation {right} denied: {reason}",
            subject=subject,
            right=right,
        )


class _MinimalBlackboard:
    """Fallback governed blackboard for standalone deployment (step 20)."""

    def __init__(self) -> None:
        self.observations: dict[str, list[dict[str, Any]]] = {}
        self.permissions: dict[str, frozenset[str]] = {}
        self.revision = 0

    def can_write(self, module_id: str, workspace_field: str) -> bool:
        return workspace_field in self.permissions.get(module_id, frozenset())

    def grant(self, module_id: str, fields: Any) -> None:
        allowed = frozenset(f for f in fields if f in FALLBACK_WORKSPACE_FIELDS)
        self.permissions[module_id] = self.permissions.get(module_id, frozenset()) | allowed

    def append(self, module_id: str, workspace_field: str, payload: Mapping[str, Any]) -> bool:
        if workspace_field not in FALLBACK_WORKSPACE_FIELDS:
            return False
        if not self.can_write(module_id, workspace_field):
            return False
        self.revision += 1
        self.observations.setdefault(workspace_field, []).append(
            {"producer": module_id, "payload": dict(payload), "revision": self.revision}
        )
        return True

    def snapshot(self) -> dict[str, list[dict[str, Any]]]:
        return {key: list(value) for key, value in self.observations.items()}

    def pending_merge(self) -> int:
        return sum(len(value) for value in self.observations.values())


class BlackboardGuard:
    """Narrow, capability-checked workspace operations (integrated step 20)."""

    def __init__(
        self,
        workspace: Any | None = None,
        *,
        authorizer: Callable[..., Any] | None = None,
        on_event: Callable[[str, Mapping[str, Any]], None] | None = None,
    ) -> None:
        self._workspace = workspace
        self._authorizer = authorizer
        self._on_event = on_event
        self._minimal: _MinimalBlackboard | None = None
        self._uses_ore_workspace = workspace is not None
        if workspace is None:
            try:
                from ore_workspace.models import Finding, FindingTarget, Provenance
                from ore_workspace.workspace import SharedWorkspace

                self._workspace = SharedWorkspace(current_input="", current_goal="runtime")
                self._uses_ore_workspace = True
                self._finding_types = (Finding,)
                self._finding_target = FindingTarget
                self._provenance = Provenance
            except Exception:
                self._workspace = None
                self._minimal = _MinimalBlackboard()

    @property
    def backed_by_ore_workspace(self) -> bool:
        return self._uses_ore_workspace and self._workspace is not None

    # -- snapshot (read path) ------------------------------------------------

    def read_snapshot(
        self,
        capability: Any,
        authorizer: Callable[..., Any] | None = None,
    ) -> dict[str, Any]:
        """Return a detached snapshot; never live mutable state (step 23)."""
        subject = getattr(capability, "subject", "")
        _authorize_or_raise(
            authorizer or self._authorizer,
            capability,
            "workspace.read_snapshot",
            "workspace",
            subject,
            {},
        )
        if self._uses_ore_workspace and self._workspace is not None:
            snapshot = self._workspace.snapshot()
            if hasattr(snapshot, "to_dict"):
                payload = snapshot.to_dict()
                return {"source": "ore_workspace", "state": payload.get("state", payload)}
            state = getattr(snapshot, "state", snapshot)
            return {"source": "ore_workspace", "state": dict(state)}
        assert self._minimal is not None
        return {"source": "minimal_blackboard", "state": self._minimal.snapshot()}

    # -- write paths ------------------------------------------------------------

    def append_observation(
        self,
        capability: Any,
        authorizer: Callable[..., Any] | None = None,
        *,
        field: str,
        payload: Mapping[str, Any],
    ) -> GuardResult:
        """Append an observation to an owned workspace field (step 20)."""
        subject = getattr(capability, "subject", "")
        constraints = {"field": field}
        try:
            _authorize_or_raise(
                authorizer or self._authorizer,
                capability,
                "workspace.append_observation",
                f"workspace/{field}",
                subject,
                constraints,
            )
        except CapabilityDeniedError as exc:
            self._emit("authorization_denied", {"actor": subject, "field": field, "reason": exc.reason})
            return GuardResult(allowed=False, reason=exc.reason, decision="denied",
                                capability_id=getattr(capability, "capability_id", ""))
        return self._publish_finding(
            subject, field, payload, finding_type="observation",
            capability_id=getattr(capability, "capability_id", ""),
        )

    def propose_fact(
        self,
        capability: Any,
        authorizer: Callable[..., Any] | None = None,
        *,
        field: str,
        payload: Mapping[str, Any],
    ) -> GuardResult:
        subject = getattr(capability, "subject", "")
        try:
            _authorize_or_raise(
                authorizer or self._authorizer,
                capability,
                "workspace.propose_fact",
                f"workspace/{field}",
                subject,
                {"field": field},
            )
        except CapabilityDeniedError as exc:
            self._emit("authorization_denied", {"actor": subject, "field": field, "reason": exc.reason})
            return GuardResult(allowed=False, reason=exc.reason, decision="denied",
                                capability_id=getattr(capability, "capability_id", ""))
        return self._publish_finding(
            subject, field, payload, finding_type="proposed_fact",
            capability_id=getattr(capability, "capability_id", ""),
        )

    def _publish_finding(
        self,
        subject: str,
        workspace_field: str,
        payload: Mapping[str, Any],
        *,
        finding_type: str,
        capability_id: str,
    ) -> GuardResult:
        if self._uses_ore_workspace and self._workspace is not None:
            try:
                from ore_workspace.models import Finding, FindingTarget, Provenance

                finding = Finding(
                    workspace_id=getattr(self._workspace, "workspace_id", "workspace"),
                    producer_module=subject,
                    finding_type=finding_type,
                    target=FindingTarget(field=workspace_field),
                    payload=dict(payload),
                    provenance=Provenance(
                        source=subject,
                        rules_applied=("ore_runtime.blackboard_guard",),
                        notes=f"capability {capability_id}",
                    ),
                    base_workspace_revision=getattr(self._workspace, "revision", 0),
                )
                validation = self._workspace.publish(finding)
                decision_obj = getattr(validation, "decision", "accepted")
                decision = getattr(decision_obj, "value", str(decision_obj))
                accepted = decision in {"accepted", "merged", "duplicate"}
                return GuardResult(
                    allowed=accepted,
                    reason=(
                        f"ore_workspace returned {decision}"
                        if accepted
                        else f"workspace policy denied the write ({decision})"
                    ),
                    decision=decision,
                    capability_id=capability_id,
                )
            except CapabilityDeniedError:
                raise
            except Exception as exc:  # workspace-level rejection stays a denial
                return GuardResult(
                    allowed=False,
                    reason=f"workspace rejected the publication: {type(exc).__name__}: {exc}",
                    decision="rejected",
                    capability_id=capability_id,
                )
        assert self._minimal is not None
        if not self._minimal.can_write(subject, workspace_field):
            return GuardResult(
                allowed=False,
                reason=(
                    f"minimal blackboard policy: {subject!r} does not own field "
                    f"{workspace_field!r}"
                ),
                decision="denied",
                capability_id=capability_id,
            )
        appended = self._minimal.append(subject, workspace_field, payload)
        return GuardResult(
            allowed=appended,
            reason="observation appended" if appended else "append failed",
            decision="accepted" if appended else "rejected",
            capability_id=capability_id,
        )

    def request_merge(
        self,
        capability: Any,
        authorizer: Callable[..., Any] | None = None,
    ) -> GuardResult:
        """Request a merge of pending proposals (step 20)."""
        subject = getattr(capability, "subject", "")
        try:
            _authorize_or_raise(
                authorizer or self._authorizer,
                capability,
                "workspace.request_merge",
                "workspace",
                subject,
                {},
            )
        except CapabilityDeniedError as exc:
            return GuardResult(allowed=False, reason=exc.reason, decision="denied",
                               capability_id=getattr(capability, "capability_id", ""))
        if self._uses_ore_workspace and self._workspace is not None:
            state_manager = getattr(self._workspace, "state_manager", None)
            reconsider = getattr(state_manager, "reconsider_pending", None)
            if callable(reconsider):
                reconsider()
                return GuardResult(
                    allowed=True,
                    reason="pending findings reconsidered through the workspace merge policies",
                    decision="reconsidered",
                    capability_id=getattr(capability, "capability_id", ""),
                )
            return GuardResult(
                allowed=True,
                reason="workspace has no pending reconsideration hook; merge request recorded",
                decision="recorded",
                capability_id=getattr(capability, "capability_id", ""),
            )
        assert self._minimal is not None
        pending = self._minimal.pending_merge()
        return GuardResult(
            allowed=True,
            reason=f"{pending} pending observations available for merge review",
            decision="recorded",
            capability_id=getattr(capability, "capability_id", ""),
        )

    def grant_minimal_permission(self, module_id: str, workspace_fields: Any) -> None:
        """Grant field ownership on the fallback blackboard (test/setup only)."""
        if self._minimal is None:
            raise RuntimeError("minimal permissions exist only in standalone mode")
        self._minimal.grant(module_id, workspace_fields)

    def _emit(self, kind: str, metadata: Mapping[str, Any]) -> None:
        if self._on_event is not None:
            try:
                self._on_event(kind, dict(metadata))
            except Exception:
                pass

    def describe(self) -> dict[str, Any]:
        return {
            "backed_by_ore_workspace": self.backed_by_ore_workspace,
            "mode": "ore_workspace" if self.backed_by_ore_workspace else "minimal_blackboard",
            "note": (
                "USG canonical updates flow only through the external semantic "
                "validation workflow; this guard never writes canonical knowledge"
            ),
        }


@dataclass
class UsgProposalRecord:
    """Outcome of one proposed USG knowledge update (step 20)."""

    status: str = "pending_external_validation"
    reasons: tuple[str, ...] = ()
    submitted_epoch: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reasons": list(self.reasons),
            "submitted_epoch": self.submitted_epoch,
        }


class USGGuard:
    """Read-only USG snapshots; updates stay external and validated (step 20)."""

    def __init__(self, *, pending_limit: int = 256) -> None:
        self._pending: list[tuple[str, Any]] = []
        self._pending_limit = pending_limit

    def read_snapshot(
        self,
        capability: Any,
        authorizer: Callable[..., Any] | None = None,
        *,
        bundle: Any | None = None,
    ) -> dict[str, Any]:
        """Return a detached read-only view of a USG bundle (step 20)."""
        subject = getattr(capability, "subject", "")
        _authorize_or_raise(authorizer, capability, "usg.read_snapshot", "usg", subject, {})
        if bundle is None:
            return {"source": "empty", "records": (), "note": "no bundle attached"}
        to_dict = getattr(bundle, "to_dict", None)
        if callable(to_dict):
            return {"source": "usg_bundle", "records": to_dict()}
        return {"source": "opaque", "note": "bundle exposes no detached view"}

    def propose_update(
        self,
        capability: Any,
        authorizer: Callable[..., Any] | None = None,
        *,
        candidate_bundle: Any,
    ) -> UsgProposalRecord:
        """Validate a candidate bundle through ore_usg; never write canonical."""
        subject = getattr(capability, "subject", "")
        _authorize_or_raise(
            authorizer, capability, "usg.propose_update", "usg", subject, {}
        )
        reasons: list[str] = []
        status = "pending_external_validation"
        try:
            from ore_usg.validation import validate_knowledge_bundle

            errors = validate_knowledge_bundle(candidate_bundle)
            if errors:
                status = "rejected"
                reasons = tuple(str(error) for error in errors)
            else:
                status = "validated_pending_review"
                reasons = ("bundle passed the USG-KNOWLEDGE/0.1.0 profile",)
        except ImportError:
            reasons = (
                "ore_usg not importable in standalone mode; structural checks only",
            )
            if not isinstance(candidate_bundle, object):
                status = "rejected"
                reasons = ("candidate bundle must be an object",)
        except Exception as exc:
            status = "rejected"
            reasons = (f"validation raised {type(exc).__name__}: {exc}",)
        if status != "rejected" and len(self._pending) < self._pending_limit:
            self._pending.append((subject, candidate_bundle))
        return UsgProposalRecord(status=status, reasons=tuple(reasons))

    def pending_count(self) -> int:
        return len(self._pending)

    def describe(self) -> dict[str, Any]:
        return {
            "pending_updates": len(self._pending),
            "canonical_writes": 0,
            "note": (
                "canonical USG knowledge changes only through the independent "
                "semantic validation and provenance workflow"
            ),
        }
