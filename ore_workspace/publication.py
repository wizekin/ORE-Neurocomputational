"""Module-facing finding publication APIs that enforce the common workspace envelope."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .models import Finding, FindingTarget, Provenance, ValidationResult


class FindingPublisher:
    """Create immutable finding envelopes and submit them through the workspace state manager."""

    def __init__(self, workspace, producer_module: str) -> None:
        """Bind a module identity to one workspace without granting direct mutable-state access."""
        if not producer_module.strip():
            raise ValueError("producer_module must be non-empty")
        self.workspace = workspace
        self.producer_module = producer_module

    def create_finding(
        self,
        field: str,
        payload: Mapping[str, Any],
        *,
        key: str | None = None,
        finding_type: str = "module_hypothesis",
        provenance: Provenance | None = None,
        dependencies: Iterable[str] = (),
        confidence: float | None = None,
        base_workspace_revision: int | None = None,
    ) -> Finding:
        """Build a typed publication envelope against a selected immutable workspace revision."""
        snapshot = self.workspace.snapshot()
        return Finding(
            workspace_id=self.workspace.workspace_id,
            producer_module=self.producer_module,
            finding_type=finding_type,
            target=FindingTarget(field=field, key=key),
            payload=dict(payload),
            provenance=provenance or Provenance(source="module_inference"),
            dependencies=tuple(dependencies),
            base_workspace_revision=(snapshot.revision if base_workspace_revision is None else base_workspace_revision),
            confidence=confidence,
        )

    def publish(
        self,
        field: str,
        payload: Mapping[str, Any],
        **options: Any,
    ) -> ValidationResult:
        """Create a new finding and submit it to the workspace's controlled validation pipeline."""
        finding = self.create_finding(field, payload, **options)
        return self.workspace.publish(finding)

    def publish_many(self, findings: Iterable[Finding]) -> tuple[ValidationResult, ...]:
        """Submit independently produced findings in order and report each validation decision."""
        results = []
        for finding in findings:
            if finding.producer_module != self.producer_module:
                raise ValueError("publisher cannot submit a finding for a different producer module")
            results.append(self.workspace.publish(finding))
        return tuple(results)