"""Integration bridge to the shared cognitive workspace (Point 14, Section 2.1).

Model proposals are published to the Shared Cognitive Workspace as findings --
append-only, provenance-carrying, and behind the documented permission-grant
handshake.  A proposal is evidence, not a state change: the workspace receives the
typed record while USG remains reachable only through independently validated and
approved update requests (see :class:`~ore_learning.contracts.AuthorityBoundary`).
"""
from __future__ import annotations

from typing import Any

from .contracts import ModelProposal

__all__ = ("WorkspaceBridge", "build_coordinator_notification")


class WorkspaceBridge:
    """Publish model proposals to the shared workspace (Section 2.1)."""

    producer_module = "ore_learning"
    workspace_field = "artifact_references"
    authorized_fields = ("artifact_references", "unresolved_questions")

    def request_authorization(self, workspace: Any) -> bool:
        """Perform the documented permission-grant handshake with a workspace."""
        permissions = getattr(workspace, "permissions", None)
        if permissions is None:
            return False
        try:
            if all(permissions.can_write(self.producer_module, field)
                   for field in self.authorized_fields):
                return False
            permissions.grant(self.producer_module, self.authorized_fields)
            return True
        except (AttributeError, TypeError, ValueError):
            return False

    def publish(self, proposal: ModelProposal, workspace: Any | None,
                approval: Any = None) -> dict[str, Any]:
        """Publish one proposal as a workspace finding; never overwrite evidence."""
        record: dict[str, Any] = {
            "published": False, "workspace_id": None, "finding_id": None,
            "decision": "no_workspace", "permission_granted": False,
            "payload_digest": None}
        if workspace is None:
            record["decision"] = "no_workspace_attached"
            return record
        try:
            from ore_workspace import FindingPublisher, Provenance  # noqa: PLC0415
        except ImportError:
            record["decision"] = "ore_workspace_unavailable"
            return record
        if not getattr(workspace.permissions, "can_write", lambda *_: False)(
                self.producer_module, self.workspace_field):
            record["permission_granted"] = self.request_authorization(workspace)
        publisher = FindingPublisher(workspace, self.producer_module)
        payload = self._payload(proposal)
        key = f"learning:{proposal.task_id}:{proposal.proposal_id}"
        finding = publisher.create_finding(
            self.workspace_field, payload, key=key,
            finding_type="model_proposal",
            provenance=Provenance(
                source="module_inference",
                rules_applied=("ore_learning.point14.pipeline",),
                evidence=({"input_ref": reference}
                          for reference in proposal.input_provenance.input_refs),
                notes=(f"model={proposal.model_id}@{proposal.model_version}; "
                       f"status={proposal.status.value}; approval="
                       + (getattr(approval, "approval_id", "none") or "none"))),
            confidence=proposal.uncertainty.confidence)
        validation = workspace.publish(finding)
        record.update({
            "published": validation.decision.value in ("accepted", "merged"),
            "workspace_id": getattr(workspace, "workspace_id", None),
            "finding_id": finding.finding_id,
            "decision": validation.decision.value,
            "workspace_revision": validation.workspace_revision,
            "payload_digest": payload.get("payload_digest")})
        return record

    @staticmethod
    def _payload(proposal: ModelProposal) -> dict[str, Any]:
        """Bounded publication payload (no raw sensitive content)."""
        return {
            "proposal_id": proposal.proposal_id,
            "task_id": proposal.task_id,
            "model_id": proposal.model_id,
            "model_version": proposal.model_version,
            "artifact_digest": proposal.artifact_digest,
            "status": proposal.status.value,
            "confidence": proposal.uncertainty.confidence,
            "verified": proposal.validation.verified,
            "is_actionable": proposal.is_actionable,
            "input_refs": list(proposal.input_provenance.input_refs),
            "payload_digest": (proposal.payload.get("digest")
                               or str(sorted(proposal.payload.keys()))),
            "append_only": True,
            "note": "model proposals are evidence records, not state changes",
        }


def build_coordinator_notification(proposal: ModelProposal) -> dict[str, Any]:
    """A concise, bounded message for the Main Brain Coordinator."""
    return {
        "message_kind": "model_proposal",
        "task_id": proposal.task_id,
        "proposal_id": proposal.proposal_id,
        "model": f"{proposal.model_id}@{proposal.model_version}",
        "status": proposal.status.value,
        "verified": proposal.validation.verified,
        "confidence": proposal.uncertainty.confidence,
        "bounded": True,
        "note": ("the coordinator arbitrates; a learned model never approves its "
                 "own output"),
    }
