"""Blackboard publication adapter (Phase G).

Publishes versioned, metadata-only optimization evidence envelopes to the
shared cognitive workspace (ore_workspace) when it is importable, and always
returns a structured publication report. Envelopes carry configuration,
error, validation status, and provenance -- never huge tensors.
"""
from __future__ import annotations

from typing import Any, Mapping

from .contracts import OptimizationResult


def build_blackboard_envelope(result: OptimizationResult) -> dict[str, Any]:
    """Build the metadata-only envelope; tensors are never attached."""
    envelope = result.to_envelope()
    for key, value in envelope.items():
        if hasattr(value, "shape"):
            raise TypeError(f"envelope field {key!r} carries array data; metadata only")
    envelope["summary_kind"] = "hacol_optimization_evidence"
    envelope["producer_module"] = "ore_hacol"
    return envelope


def publish_to_blackboard(
    envelope: Mapping[str, Any],
    workspace: Any | None = None,
    *,
    field: str = "hacol_optimization_evidence",
    key: str | None = None,
) -> dict[str, Any]:
    """Submit one envelope through the controlled publication pipeline."""
    if workspace is None:
        return {
            "published": False,
            "channel": "blackboard",
            "reason": "no workspace instance provided; envelope retained in the result",
            "envelope_keys": sorted(envelope.keys()),
        }
    try:
        from ore_workspace.publication import FindingPublisher
    except ImportError:
        return {
            "published": False,
            "channel": "blackboard",
            "reason": "ore_workspace not importable in this environment",
            "envelope_keys": sorted(envelope.keys()),
        }
    publisher = FindingPublisher(workspace, "ore_hacol")
    validation = publisher.publish(field, dict(envelope), key=key)
    return {
        "published": True,
        "channel": "blackboard",
        "workspace_id": workspace.workspace_id,
        "revision": workspace.revision,
        "validation": getattr(validation, "decision", None) or str(validation),
        "field": field,
        "envelope_keys": sorted(envelope.keys()),
    }


def publish_result(
    result: OptimizationResult,
    workspace: Any | None = None,
    *,
    field: str = "hacol_optimization_evidence",
) -> dict[str, Any]:
    """Publish one OptimizationResult envelope to the Blackboard."""
    envelope = build_blackboard_envelope(result)
    return publish_to_blackboard(envelope, workspace, field=field)
