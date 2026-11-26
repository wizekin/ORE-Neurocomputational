"""Bounded specialist execution against detached workspace snapshots (Steps 10, 15, 16).

Specialists are invoked through the existing ``ore_workspace.ModuleScheduler``
bounded executor: parallel workers, per-module timeouts, activation caps, and
isolated failure handling. This module adds the invocation identity, computation
suppression registry, and the manifest-backed invocation adapter that converts
typed module results into findings through the artifact bridge.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from ore_contracts import ComputeBudget, OperationMode, RunContext
from ore_contracts.module_artifacts import ContractArtifact
from ore_workspace.models import WorkspaceSnapshot

from .bridge import validate_module_result
from .config import BlackboardPolicy
from .contracts import sha256_json


@dataclass(slots=True)
class InvocationRecord:
    """Record one specialist invocation with its identity and outcome (Step 10)."""

    module_id: str
    module_version: str
    invocation_id: str
    base_revision: int
    relevant_fingerprint: str
    status: str = "pending"
    started_at_ms: float = 0.0
    duration_ms: float = 0.0
    produced_finding_ids: tuple[str, ...] = ()
    suppressed: bool = False
    error: str | None = None


class ComputationRegistry:
    """Suppress repeated computations with unchanged relevant inputs (Steps 10, 23).

    A computation is suppressed when the tuple
    ``(module_version, input_fingerprints, workspace_revision_of_relevant_fields)``
    already produced a reusable successful output.
    """

    def __init__(self) -> None:
        """Initialize the per-run computation cache."""
        self._records: dict[str, InvocationRecord] = {}
        self._successful: dict[tuple[str, str], InvocationRecord] = {}
        self.suppressed_count = 0

    @staticmethod
    def invocation_id(
        task_id: str, module_id: str, module_version: str, input_hash: str, revision: int,
    ) -> str:
        """Return ``id_call = H(task_id, module_id, version, input_hashes, r_t)`` (Step 10)."""
        return sha256_json({
            "task_id": task_id, "module_id": module_id, "module_version": module_version,
            "input_hash": input_hash, "revision": revision,
        })

    def relevant_fingerprint(self, module_id: str, snapshot: WorkspaceSnapshot,
                             subscribed_fields: tuple[str, ...]) -> str:
        """Fingerprint the module's subscribed fields in one snapshot revision."""
        state = snapshot.state
        return sha256_json({
            field: state.get(field) for field in sorted(subscribed_fields)
        })

    def check_suppressed(
        self, module_id: str, module_version: str, relevant_fingerprint: str,
    ) -> InvocationRecord | None:
        """Return the earlier successful record when the computation is redundant."""
        return self._successful.get((module_id, relevant_fingerprint))

    def record(self, invocation: InvocationRecord) -> None:
        """Store one invocation outcome and index reusable successful outputs."""
        self._records[invocation.invocation_id] = invocation
        if invocation.status == "success" and not invocation.error:
            self._successful.setdefault(
                (invocation.module_id, invocation.relevant_fingerprint), invocation,
            )

    def count_suppressed(self) -> int:
        """Return the number of suppressed redundant computations in this run."""
        return self.suppressed_count

    def mark_suppressed(self) -> None:
        """Count one suppressed redundant computation."""
        self.suppressed_count += 1

    def records(self) -> tuple[InvocationRecord, ...]:
        """Return all invocation records in insertion order."""
        return tuple(self._records.values())


def build_snapshot_artifact(
    snapshot: WorkspaceSnapshot, run_id: str,
) -> ContractArtifact:
    """Wrap a detached workspace snapshot as a typed input artifact (Step 10)."""
    from .bridge import BLACKBOARD_CONTRACT_VERSION, BLACKBOARD_SNAPSHOT_CONTRACT

    return ContractArtifact(
        envelope=_envelope(
            BLACKBOARD_SNAPSHOT_CONTRACT, BLACKBOARD_CONTRACT_VERSION, run_id,
            "ore.blackboard.coordinator", {"revision": snapshot.revision},
        ),
        payload={
            "workspace_id": snapshot.workspace_id,
            "revision": snapshot.revision,
            "state": dict(snapshot.state),
        },
        metadata={"detached": True, "snapshot_created_at": snapshot.created_at},
    )


def build_input_root_artifact(
    input_root, run_id: str, task_id: str = "",
) -> ContractArtifact:
    """Wrap the immutable input root as a typed input artifact (Steps 4, 10)."""
    from .bridge import BLACKBOARD_CONTRACT_VERSION, BLACKBOARD_INPUT_CONTRACT

    return ContractArtifact(
        envelope=_envelope(
            BLACKBOARD_INPUT_CONTRACT, BLACKBOARD_CONTRACT_VERSION, run_id,
            "ore.blackboard.coordinator", {"input_digest": input_root.input_digest},
        ),
        payload={
            "task_id": task_id,
            "text": input_root.primary_text(),
            "input_digest": input_root.input_digest,
            "tokens": [
                {
                    "token": span.token, "source_id": span.source_id,
                    "start": span.start, "end": span.end,
                }
                for span in input_root.tokens
            ],
            "sources": [
                {"source_id": item.source_id, "source_group": item.source_group}
                for item in input_root.sources
            ],
        },
        metadata={"immutable": True, "root_id": input_root.root_id},
    )


def build_memory_artifact(episodes: tuple[Mapping[str, Any], ...], run_id: str) -> ContractArtifact:
    """Wrap caller-supplied episodic memories as a typed input artifact (Step 1 input 5)."""
    from .bridge import BLACKBOARD_CONTRACT_VERSION, BLACKBOARD_MEMORY_STORE_CONTRACT

    return ContractArtifact(
        envelope=_envelope(
            BLACKBOARD_MEMORY_STORE_CONTRACT, BLACKBOARD_CONTRACT_VERSION, run_id,
            "ore.blackboard.coordinator", {"memory_count": len(episodes)},
        ),
        payload={"episodes": [dict(item) for item in episodes]},
        metadata={"caller_supplied": True},
    )


def build_rules_artifact(rules: tuple[Mapping[str, Any], ...], run_id: str) -> ContractArtifact:
    """Wrap caller-supplied verification rules as a typed input artifact."""
    from .bridge import BLACKBOARD_CONTRACT_VERSION, BLACKBOARD_VERIFICATION_RULES_CONTRACT

    return ContractArtifact(
        envelope=_envelope(
            BLACKBOARD_VERIFICATION_RULES_CONTRACT, BLACKBOARD_CONTRACT_VERSION, run_id,
            "ore.blackboard.coordinator", {"rule_count": len(rules)},
        ),
        payload={"rules": [dict(item) for item in rules]},
        metadata={"caller_supplied": True},
    )


def _envelope(contract_id: str, version: str, run_id: str, producer_id: str,
              config: Mapping[str, Any]) -> Any:
    """Create a contract envelope with a deterministic config hash."""
    from ore_contracts import ContractEnvelope, EvidenceKind
    from ore_workspace.models import utc_now

    return ContractEnvelope(
        contract_id=contract_id,
        schema_version=version,
        run_id=run_id,
        producer_id=producer_id,
        config_hash=sha256_json(dict(config)),
        created_at=utc_now(),
        evidence_kind=EvidenceKind.OBSERVED,
    )


class SpecialistInvocation:
    """Adapter binding one manifest-backed specialist into the workspace scheduler (Step 16).

    The scheduler invokes ``__call__(snapshot)`` with a detached snapshot; the
    adapter assembles the manifest's typed inputs, executes the registered
    backend, validates the typed result, bridges artifacts into findings, and
    records the invocation. Exceptions and timeouts are allowed to propagate so
    the scheduler isolates them without fabricating an empty successful result.
    """

    def __init__(
        self,
        specialist: Any,
        *,
        run_context: RunContext,
        catalog: Any,
        task_artifacts: Mapping[str, ContractArtifact],
        bridge_factory: Callable[[WorkspaceSnapshot], Any],
        computation_registry: ComputationRegistry,
        task_id: str,
        subscribed_fields: tuple[str, ...],
        measurement_sink: Callable[[str, float], None] | None = None,
        artifact_provider: Callable[[WorkspaceSnapshot], Mapping[str, ContractArtifact]] | None = None,
    ) -> None:
        """Bind the specialist with its immutable task-level input artifacts."""
        self.specialist = specialist
        self.run_context = run_context
        self.catalog = catalog
        self.task_artifacts = dict(task_artifacts)
        self.bridge_factory = bridge_factory
        self.computation_registry = computation_registry
        self.task_id = task_id
        self.subscribed_fields = subscribed_fields
        self.measurement_sink = measurement_sink
        self.artifact_provider = artifact_provider
        self.manifest = specialist.describe()
        self.module_id = self.manifest.identity.module_id
        self.module_version = self.manifest.identity.module_version
        self.last_record: InvocationRecord | None = None

    def _assemble_inputs(self, snapshot: WorkspaceSnapshot) -> Mapping[str, ContractArtifact]:
        """Assemble typed port inputs from task artifacts and the detached snapshot."""

        artifacts = dict(self.task_artifacts)
        if self.artifact_provider is not None:
            dynamic_artifacts = self.artifact_provider(snapshot)
            if not isinstance(dynamic_artifacts, Mapping):
                raise TypeError("artifact_provider must return a mapping of port IDs to artifacts")
            for port_id, artifact in dynamic_artifacts.items():
                if not isinstance(artifact, ContractArtifact):
                    raise TypeError(f"artifact_provider returned a non-ContractArtifact for {port_id!r}")
                artifacts[str(port_id)] = artifact
        artifacts.setdefault("workspace_snapshot", build_snapshot_artifact(
            snapshot, self.run_context.run_id,
        ))
        inputs: dict[str, ContractArtifact] = {}
        for port in self.manifest.input_ports:
            artifact = artifacts.get(port.port_id)
            if artifact is None:
                continue
            if (artifact.envelope.contract_id != port.contract_id
                    or artifact.envelope.schema_version not in port.accepted_versions):
                raise ValueError(
                    f"input artifact for port {port.port_id!r} does not match contract"
                )
            validation = self.catalog.validate_artifact(artifact)
            if not validation.valid:
                raise ValueError(
                    f"invalid input artifact for port {port.port_id!r}: "
                    + "; ".join(validation.errors)
                )
            inputs[port.port_id] = artifact
        return inputs

    def __call__(self, snapshot: WorkspaceSnapshot) -> tuple[Any, ...]:
        """Execute the specialist and return findings for scheduler publication."""
        started = time.perf_counter()
        fingerprint = self.computation_registry.relevant_fingerprint(
            self.module_id, snapshot, self.subscribed_fields,
        )
        invocation_id = self.computation_registry.invocation_id(
            self.task_id, self.module_id, self.module_version, fingerprint, snapshot.revision,
        )
        cached = self.computation_registry.check_suppressed(
            self.module_id, self.module_version, fingerprint,
        )
        if cached is not None:
            self.computation_registry.mark_suppressed()
            self.last_record = InvocationRecord(
                self.module_id, self.module_version, invocation_id, snapshot.revision,
                fingerprint, status="suppressed", suppressed=True,
                error=f"reuses successful output from revision {cached.base_revision}",
            )
            return ()
        record = InvocationRecord(
            self.module_id, self.module_version, invocation_id, snapshot.revision, fingerprint,
        )
        self.computation_registry.record(record)
        try:
            inputs = self._assemble_inputs(snapshot)
            missing_required = [
                port.port_id for port in self.manifest.input_ports
                if port.required and port.port_id not in inputs
            ]
            if missing_required:
                raise ValueError(
                    "missing required input port(s): " + ", ".join(missing_required)
                )
            raw_result = self.specialist.execute(inputs, self.run_context)
            artifacts, errors = validate_module_result(
                raw_result,
                module_id=self.module_id,
                module_version=self.module_version,
                run_id=self.run_context.run_id,
                catalog=self.catalog,
            )
            if errors:
                record.status = "failed"
                record.error = "; ".join(errors)
                raise ValueError("; ".join(errors))
            record.status = raw_result.status.value
            bridge = self.bridge_factory(snapshot)
            findings: list[Any] = []
            for artifact in artifacts:
                findings.extend(bridge.convert(
                    artifact, base_workspace_revision=snapshot.revision,
                ))
            record.produced_finding_ids = tuple(
                finding.finding_id for finding in findings
            )
            self.last_record = record
            return tuple(findings)
        except Exception as exc:
            record.status = "failed"
            record.error = f"{type(exc).__name__}: {exc}"
            self.last_record = record
            raise
        finally:
            record.duration_ms = (time.perf_counter() - started) * 1000.0
            if self.measurement_sink is not None:
                self.measurement_sink(self.module_id, record.duration_ms)


def make_run_context(
    run_id: str, policy: BlackboardPolicy, execution_mode: str,
) -> RunContext:
    """Create the task-scoped run context with compute budgets (Step 1, 15)."""
    return RunContext(
        run_id=run_id,
        mode=OperationMode(execution_mode),
        deterministic_seed=policy.deterministic_seed,
        compute_budget=ComputeBudget(
            wall_clock_seconds=policy.budgets.max_time_seconds,
            max_recurrent_iterations=policy.budgets.max_rounds,
            max_memory_mb=policy.budgets.max_memory_mb,
        ),
    )


__all__ = [
    "ComputationRegistry",
    "InvocationRecord",
    "SpecialistInvocation",
    "build_input_root_artifact",
    "build_memory_artifact",
    "build_rules_artifact",
    "build_snapshot_artifact",
    "make_run_context",
]
