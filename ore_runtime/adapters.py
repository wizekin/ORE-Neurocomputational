"""Integration adapters for the shared cognitive workspace and USG sources."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .concurrency.task_graph import TaskNode
from .contracts import content_fingerprint


class WorkspaceRuntimeAdapter:
    """Publish runtime task results into the shared cognitive workspace.

    The shared workspace stays authoritative for its own fields: the runtime
    engine never mutates workspace internals directly, it publishes validated
    Finding envelopes through the workspace's own commit path and keeps its
    authoritative versioned state inside the runtime VersionedStore.
    """

    def __init__(self, workspace: Any, *, producer_module: str = "ore_runtime.engine") -> None:
        """Wrap one SharedWorkspace instance with engine publication rights."""
        self._workspace = workspace
        self._producer = producer_module

    @property
    def workspace(self) -> Any:
        """Return the wrapped shared workspace."""
        return self._workspace

    @property
    def revision(self) -> int:
        """Return the workspace's authoritative revision."""
        return int(self._workspace.revision)

    def snapshot(self) -> Mapping[str, Any]:
        """Return a detached workspace snapshot mapping for read contexts."""
        return self._workspace.snapshot().state

    def publish_task_result(
        self, node: TaskNode, output: Mapping[str, Any], *,
        read_revision: int, correlation_id: str,
    ) -> tuple[Any | None, str]:
        """Publish one validated task result as a workspace finding.

        Returns the published Finding and its decision when the workspace
        accepted (or merged) the contribution, otherwise (None, decision).
        """
        from ore_workspace import Finding, FindingTarget, Provenance, UpdateDecision
        workspace_id = getattr(self._workspace, "workspace_id", "")
        finding = Finding(
            workspace_id=workspace_id,
            producer_module=self._producer,
            finding_type="runtime_task_result",
            target=FindingTarget(field="artifact_references", key=node.task_id),
            payload={
                "task_id": node.task_id,
                "capability": node.spec.capability,
                "output": dict(output),
            },
            provenance=Provenance(
                source=f"ore_runtime::{node.task_id}",
                rules_applied=tuple(node.spec.metadata.get("rules_applied", ())),
                notes=(
                    "published by the Point 19 concurrent execution engine; "
                    f"correlation {correlation_id}"
                ),
            ),
            base_workspace_revision=read_revision,
            dependencies=tuple(node.spec.dependencies),
            confidence=None,
        )
        result = self._workspace.publish(finding)
        decision = result.decision
        if decision is UpdateDecision.ACCEPTED or decision is UpdateDecision.MERGED:
            return finding, decision.value
        return None, decision.value

    def register_module_status(self, module_id: str, status: Mapping[str, Any]) -> Any:
        """Publish one module status finding through the validated workspace path."""
        from ore_workspace import Finding, FindingTarget, Provenance, UpdateDecision
        workspace_id = getattr(self._workspace, "workspace_id", "")
        finding = Finding(
            workspace_id=workspace_id,
            producer_module=self._producer,
            finding_type="module_status",
            target=FindingTarget(field="module_status", key=module_id),
            payload=dict(status),
            provenance=Provenance(source=f"ore_runtime::{module_id}"),
            base_workspace_revision=int(self._workspace.revision),
        )
        result = self._workspace.publish(finding)
        if result.decision is UpdateDecision.ACCEPTED:
            return finding
        return None


class USGSourceAdapter:
    """Bridge versioned USG source records into the runtime dependency graph.

    USG remains read-only from the runtime's perspective: this adapter only
    registers records and their validation contract, and versioned changes are
    declared through the engine's explicit write path so no concurrency
    mechanism can bypass USG validation or mutate authoritative knowledge.
    """

    def __init__(self, dependency_graph: Any) -> None:
        """Bind the adapter to the runtime dependency graph."""
        self._graph = dependency_graph
        self._validators: dict[str, Callable[[Mapping[str, Any]], bool]] = {}

    def register(
        self, record_id: str, payload: Mapping[str, Any], *,
        validator: Callable[[Mapping[str, Any]], bool] | None = None,
    ) -> Any:
        """Register one USG record with its validation contract."""
        if validator is not None:
            self._validators[record_id] = validator
        return self._graph.register_source(
            record_id, payload, validator=validator,
        )

    def declare_change(
        self, record_id: str, payload: Mapping[str, Any],
    ) -> Any:
        """Version one USG record through its registered validation contract."""
        return self._graph.register_source(
            record_id, payload, validator=self._validators.get(record_id),
        )

    def record(self, record_id: str) -> Any | None:
        """Return the current versioned USG record."""
        return self._graph.source(record_id)

    def fingerprint(self, payload: Mapping[str, Any]) -> str:
        """Hash one USG payload so records can be referenced by content."""
        return content_fingerprint(payload)


def build_workspace_permissions() -> Any:
    """Return a permission policy granting the runtime engine workspace writes."""
    from ore_workspace import DEFAULT_WRITE_PERMISSIONS, PermissionPolicy
    permissions: dict[str, frozenset[str]] = {
        module_id: frozenset(fields)
        for module_id, fields in DEFAULT_WRITE_PERMISSIONS.items()
    }
    permissions["ore_runtime.engine"] = frozenset({"module_status", "artifact_references"})
    return PermissionPolicy(permissions)


def connect_workspace(workspace: Any) -> WorkspaceRuntimeAdapter:
    """Wrap an existing SharedWorkspace with the runtime publication adapter."""
    return WorkspaceRuntimeAdapter(workspace)


def module_task_executor(
    module: Any, input_builder: Callable[[Mapping[str, Any]], Mapping[str, Any]],
) -> Callable[[Any], Mapping[str, Any]]:
    """Adapt one ORE module protocol backend into a runtime task executor."""

    def execute(context: Any) -> Mapping[str, Any]:
        """Run the module through its protocol and collect its artifacts."""
        inputs = input_builder(context.snapshot)
        run_context = getattr(module, "last_run_context", None)
        result = module.execute(inputs, run_context)
        artifacts = {}
        for artifact in getattr(result, "artifacts", ()) or ():
            artifacts[getattr(artifact, "artifact_id", str(artifact))] = artifact
        return {
            "module_id": getattr(result, "module_id", ""),
            "status": getattr(result, "status", ""),
            "artifacts": artifacts,
        }

    return execute


def summarize_sources(sources: Sequence[Any]) -> list[dict[str, Any]]:
    """Summarize versioned USG source records for diagnostics."""
    return [
        {
            "record_id": source.record_id,
            "version": source.version,
            "content_hash": source.content_hash,
        }
        for source in sources
    ]
