"""WorkspaceBridge: publication into the Shared Cognitive Workspace.

The Shared Cognitive Workspace remains the authoritative cognitive state;
the workflow only references its revisions.  The bridge publishes typed
findings (artifact references, module status, unresolved questions) through
the workspace's own publication path with full provenance, and never mutates
workspace state directly.
"""
from __future__ import annotations

from typing import Any, Mapping

from ore_workspace import FindingPublisher, SharedWorkspace
from ore_workspace.models import Finding, Provenance

__all__ = ("WorkspaceBridge", "AGENTIC_MODULE_ID")

AGENTIC_MODULE_ID = "ore_agentic"


class WorkspaceBridge:
    """Publish agentic results into the authoritative workspace (Step 50).

    Usage requires the workspace permissions to grant the published fields
    to the ``ore_agentic`` module (see ore_workspace.permissions); the bridge
    reports but never bypasses a permission denial.
    """

    def __init__(self, workspace: SharedWorkspace) -> None:
        """Bind the bridge to one workspace and its publisher."""
        self._workspace = workspace
        self._publisher = FindingPublisher(workspace, AGENTIC_MODULE_ID)
        self.published_findings: list[str] = []
        self.denied_publications: list[str] = []

    @property
    def workspace_id(self) -> str:
        """Return the authoritative workspace ID."""
        return self._workspace.workspace_id

    @property
    def workspace_revision(self) -> int:
        """Return the current workspace revision (referenced, not forked)."""
        return self._workspace.revision

    def publish_workflow_result(self, result: Mapping[str, Any]) -> Mapping[str, Any]:
        """Publish the workflow close-out artifact (Step 50).

        A permission denial is reported through ``denied_publications`` and
        the rejected validation result; the bridge never bypasses the
        workspace's own permission policy.
        """
        provenance = Provenance(
            source="ore_agentic.workflow",
            rules_applied=("point15.workflow.close",),
            notes=f"workflow {result.get('workflow_id', '')} "
                  f"status {result.get('status', '')}")
        validation = self._publisher.publish(
            "artifact_references",
            {"workflow_id": result.get("workflow_id", ""),
             "status": result.get("status", ""),
             "outcomes": dict(result.get("outcomes", {}))},
            finding_type="agentic_workflow_result",
            provenance=provenance,
            confidence=float(result.get("confidence", 0.8)))
        decision = str(getattr(validation, "decision", "accepted"))
        if "REJECT" in decision.upper():
            self.denied_publications.append(
                str(result.get("workflow_id", "")))
        else:
            self.published_findings.append(
                getattr(validation, "finding_id", "") or "")
        return validation

    def publish_unresolved_question(self, question: str) -> Mapping[str, Any]:
        """Publish a clarification request as an unresolved question (Step 2)."""
        return self._publisher.publish(
            "unresolved_questions",
            {"question": question, "origin": AGENTIC_MODULE_ID},
            finding_type="agentic_clarification_request")

    def publish_module_status(self, status: Mapping[str, Any]) -> Mapping[str, Any]:
        """Publish the operational module status (Step 43)."""
        return self._publisher.publish(
            "module_status",
            {"module": AGENTIC_MODULE_ID, "status": dict(status)},
            finding_type="agentic_module_status")

    def describe(self) -> dict[str, Any]:
        """Return a serializable bridge summary."""
        return {"workspace_id": self.workspace_id,
                "workspace_revision": self.workspace_revision,
                "published_findings": len(self.published_findings),
                "denied_publications": len(self.denied_publications)}


def build_agentic_finding(workspace: SharedWorkspace, field: str,
                          payload: Mapping[str, Any], *,
                          finding_type: str = "agentic_finding",
                          base_revision: int | None = None) -> Finding:
    """Build an unpublished Finding for callers composing their own batches."""
    publisher = FindingPublisher(workspace, AGENTIC_MODULE_ID)
    return publisher.create_finding(
        field, payload, finding_type=finding_type,
        base_workspace_revision=base_revision)
