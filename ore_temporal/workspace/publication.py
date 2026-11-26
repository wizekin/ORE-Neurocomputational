"""Blackboard publication, coordinator notification, and task decomposition (Steps 35-37)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..contracts import TemporalResult
from ..representations.evidence import ProvenanceRecord

__all__ = ("WorkspaceBridge", "build_coordinator_notification", "build_downstream_task_hints")


@dataclass(slots=True)
class WorkspaceBridge:
    """Publish versioned temporal findings to the shared cognitive workspace (Step 35).

    Findings target an append-only workspace field so earlier observations are never
    overwritten; each publication carries provenance, confidence, and evidence
    references.  When no workspace is wired in, publications are recorded locally and
    reported as such.
    """

    producer_module: str = "ore_temporal"
    workspace_field: str = "artifact_references"
    #: Fields this producer is authorized to write after the documented grant handshake.
    authorized_fields: tuple[str, ...] = ("artifact_references", "unresolved_questions")

    def request_authorization(self, workspace: Any) -> bool:
        """Perform the documented permission-grant handshake with a workspace.

        Uses the workspace's public PermissionPolicy.grant API so that field ownership
        stays explicit and auditable; deployments may pre-grant instead.
        """
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

    def publish(self, result: TemporalResult, workspace: Any | None,
                provenance: ProvenanceRecord | None = None,
                usg_mapping: Mapping[str, str] | None = None) -> dict[str, Any]:
        """Publish one versioned record per result; never overwrite earlier evidence."""
        payload = self._payload(result, usg_mapping or {})
        record = {
            "published": False,
            "workspace_id": None,
            "finding_id": None,
            "decision": "no_workspace",
            "payload": payload,
            "permission_granted": False,
        }
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
        key = f"temporal:{result.task_id}:{result.result_type.value}"
        finding = publisher.create_finding(
            self.workspace_field,
            payload,
            key=key,
            finding_type="temporal_result",
            provenance=Provenance(
                source="module_inference",
                rules_applied=("ore_temporal.point13.pipeline",),
                evidence=tuple(payload.get("evidence", []) or ()),
                notes=(provenance.to_dict().get("notes", "") if provenance else ""),
            ),
            confidence=(max((item.confidence for item in result.events
                             if item.confidence is not None), default=0.5)),
        )
        validation = workspace.publish(finding)
        record.update({
            "published": validation.decision.value in ("accepted", "merged"),
            "workspace_id": getattr(workspace, "workspace_id", None),
            "finding_id": finding.finding_id,
            "decision": validation.decision.value,
            "workspace_revision": validation.workspace_revision,
        })
        return record

    def _payload(self, result: TemporalResult,
                 usg_mapping: Mapping[str, str]) -> dict[str, Any]:
        """Build the bounded publication payload (no raw streams)."""
        payload = result.to_dict()
        payload.pop("operation_trace", None)
        payload.pop("diagnostics", None)
        payload["usg"] = {
            event.event_type: usg_mapping.get(
                event.event_type, "usg:temporal:unknown@1")
            for event in result.events}
        payload["append_only"] = True
        return payload


def build_coordinator_notification(result: TemporalResult, max_events: int = 8) -> dict[str, Any]:
    """Step 36 -- a concise, bounded event/result message for the Main Brain Coordinator.

    The coordinator chooses downstream consumers; raw samples are never broadcast.
    """
    events = result.events[:max_events]
    return {
        "message_kind": "temporal_result",
        "task_id": result.task_id,
        "source_id": result.source_id,
        "interval": list(result.interval),
        "result_type": result.result_type.value,
        "validation_status": result.validation_status.value,
        "n_estimates": len(result.estimates),
        "n_events": len(result.events),
        "event_types": [event.event_type for event in events],
        "top_event_severities": [event.severity.value for event in events],
        "model_family": (result.model_identity.family.value if result.model_identity
                         else "baseline"),
        "uncertainty": (result.uncertainty.to_dict() if result.uncertainty else None),
        "bounded": True,
        "note": "concise summary; the coordinator selects downstream consumers",
    }


def build_downstream_task_hints(result: TemporalResult) -> list[dict[str, Any]]:
    """Step 37 -- bounded task-decomposition hints for CognitiveTaskCoordinator.

    Each hint carries a bounded evidence slice and one explicit question.
    """
    hints: list[dict[str, Any]] = []
    if result.events:
        hints.append({
            "task_kind": "explanation",
            "question": "Which candidate cause best explains the detected temporal events?",
            "evidence": [event.to_dict() for event in result.events[:4]],
            "evidence_bound": 4,
        })
    if result.estimates:
        hints.append({
            "task_kind": "cross_source_comparison",
            "question": "Do other sources corroborate the forecast trajectory?",
            "evidence": [estimate.to_dict() for estimate in result.estimates[:4]],
            "evidence_bound": 4,
        })
    if result.baseline_comparison is not None:
        hints.append({
            "task_kind": "resource_prediction",
            "question": "Does the measured model benefit justify the resource cost?",
            "evidence": [result.baseline_comparison.to_dict()],
            "evidence_bound": 1,
        })
    return hints
