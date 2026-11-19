"""Dependency-aware coordination for registered ORE cognitive modules.

The coordinator plans from declared capabilities, exact contract versions, declared
dependencies, and validated artifacts. It preserves the canonical B-N9 controller
and delegates specialist computation to the registered ORE module backends.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from contextlib import nullcontext
from dataclasses import dataclass, field
import math
from pathlib import Path
import threading
import time
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4

from . import ComputeBudget, OperationMode, RunContext, sha256_json, utc_now_iso
from .advanced import TraceEvent
from .contracts_registry import ContractCatalog
from .module_artifacts import ContractArtifact
from .module_logging import SQLiteModuleEventLogger
from .module_manifest import InputPort, ModuleManifest
from .module_protocol import ModuleResult, ModuleStatus
from .module_registry import ModuleRegistry
from ore_workspace import FindingPublisher, Provenance, WorkspaceManager


@dataclass(frozen=True, slots=True)
class CognitiveTask:
    """Declare a task goal, required capabilities, typed inputs, and execution constraints."""

    goal: str
    required_capabilities: tuple[str, ...]
    input_artifacts: Mapping[str, Any] = field(default_factory=dict)
    optional_capabilities: tuple[str, ...] = ()
    expected_outputs: tuple[str, ...] = ()
    execution_mode: str = "FUNCTIONAL"
    priority: float = 0.5
    max_replans: int = 1
    metadata: Mapping[str, Any] = field(default_factory=dict)
    task_id: str = field(default_factory=lambda: f"cognitive_task_{uuid4().hex}")

    def __post_init__(self) -> None:
        """Validate declared requirements without guessing capabilities from free-form text."""
        if not isinstance(self.goal, str) or not self.goal.strip():
            raise ValueError("goal must be a non-empty string")
        required = tuple(self.required_capabilities)
        optional = tuple(self.optional_capabilities)
        outputs = tuple(self.expected_outputs)
        if any(not isinstance(item, str) or not item.strip() for item in required + optional + outputs):
            raise ValueError("capabilities and expected outputs must be non-empty strings")
        if len(set(required)) != len(required) or len(set(optional)) != len(optional):
            raise ValueError("capability requirements must not contain duplicates")
        if set(required).intersection(optional):
            raise ValueError("a capability cannot be both required and optional")
        if not isinstance(self.input_artifacts, Mapping) or not isinstance(self.metadata, Mapping):
            raise TypeError("input_artifacts and metadata must be mappings")
        mode = self.execution_mode.value if isinstance(self.execution_mode, OperationMode) else str(self.execution_mode).upper()
        OperationMode(mode)
        if not math.isfinite(float(self.priority)) or not 0.0 <= float(self.priority) <= 1.0:
            raise ValueError("priority must be finite and in [0, 1]")
        if self.max_replans < 0:
            raise ValueError("max_replans cannot be negative")
        if not isinstance(self.task_id, str) or not self.task_id.strip():
            raise ValueError("task_id must be non-empty")
        object.__setattr__(self, "required_capabilities", required)
        object.__setattr__(self, "optional_capabilities", optional)
        object.__setattr__(self, "expected_outputs", outputs)
        object.__setattr__(self, "execution_mode", mode)
        object.__setattr__(self, "input_artifacts", dict(self.input_artifacts))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def input_fingerprint(self) -> str:
        """Hash canonical input envelopes and payload digests without logging payload contents."""
        rows = []
        for label, value in _iter_task_artifacts(self):
            if isinstance(value, ContractArtifact):
                rows.append({
                    "label": label,
                    "envelope": value.envelope.to_dict(),
                    "payload_hash": value.payload_hash,
                })
            else:
                rows.append({"label": label, "invalid_input_type": type(value).__qualname__})
        return sha256_json({
            "goal": self.goal,
            "required_capabilities": list(self.required_capabilities),
            "optional_capabilities": list(self.optional_capabilities),
            "expected_outputs": list(self.expected_outputs),
            "execution_mode": self.execution_mode,
            "inputs": rows,
        })


@dataclass(frozen=True, slots=True)
class ModuleAvailability:
    """Separate backend existence and validation from dependency readiness."""

    module_id: str
    executable: bool
    status: str
    dependencies_ready: bool
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ModuleInputPlan:
    """Describe the external or upstream source for one module input port."""

    module_id: str
    port_id: str
    contract_id: str
    accepted_versions: tuple[str, ...]
    required: bool
    multiple: bool
    external_artifact_ids: tuple[str, ...] = ()
    producer_module_ids: tuple[str, ...] = ()
    missing: bool = False


@dataclass(frozen=True, slots=True)
class CognitiveTaskPlan:
    """Hold the dependency graph, execution layers, input compatibility matrix, and gaps."""

    selected_modules: tuple[str, ...]
    required_modules: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    layers: tuple[tuple[str, ...], ...]
    input_plans: Mapping[str, tuple[ModuleInputPlan, ...]]
    missing_capabilities: tuple[str, ...] = ()
    missing_dependencies: tuple[str, ...] = ()
    missing_inputs: tuple[str, ...] = ()
    cycle_nodes: tuple[str, ...] = ()
    optional_gaps: tuple[str, ...] = ()
    capability_coverage: float = 1.0
    compatibility_matrix: Mapping[str, Mapping[str, int]] = field(default_factory=dict)

    @property
    def executable(self) -> bool:
        """Indicate whether all mandatory planning constraints are satisfied."""
        return not (
            self.missing_capabilities or self.missing_dependencies
            or self.missing_inputs or self.cycle_nodes
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize planning decisions without including backends or artifact payloads."""
        return {
            "selected_modules": list(self.selected_modules),
            "required_modules": list(self.required_modules),
            "edges": [list(item) for item in self.edges],
            "layers": [list(item) for item in self.layers],
            "input_plans": {
                key: [
                    {
                        "port_id": item.port_id,
                        "contract_id": item.contract_id,
                        "accepted_versions": list(item.accepted_versions),
                        "required": item.required,
                        "multiple": item.multiple,
                        "external_artifact_ids": list(item.external_artifact_ids),
                        "producer_module_ids": list(item.producer_module_ids),
                        "missing": item.missing,
                    }
                    for item in value
                ]
                for key, value in self.input_plans.items()
            },
            "compatibility_matrix": {
                key: dict(value) for key, value in self.compatibility_matrix.items()
            },
            "missing_capabilities": list(self.missing_capabilities),
            "missing_dependencies": list(self.missing_dependencies),
            "missing_inputs": list(self.missing_inputs),
            "cycle_nodes": list(self.cycle_nodes),
            "optional_gaps": list(self.optional_gaps),
            "capability_coverage": self.capability_coverage,
            "executable": self.executable,
        }


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    """Store a typed artifact with immutable identity, hash, and lineage metadata."""

    artifact_id: str
    contract_id: str
    schema_version: str
    producer_module: str
    payload_hash: str
    parent_artifact_ids: tuple[str, ...]
    external: bool
    artifact: ContractArtifact = field(compare=False, repr=False)

    def to_reference(self) -> dict[str, Any]:
        """Return a compact JSON-compatible artifact reference for workspace publication."""
        return {
            "artifact_id": self.artifact_id,
            "contract_id": self.contract_id,
            "schema_version": self.schema_version,
            "producer_module": self.producer_module,
            "payload_hash": self.payload_hash,
            "parent_artifact_ids": list(self.parent_artifact_ids),
            "external": self.external,
            "provenance": self.artifact.envelope.to_dict(),
            "metadata": dict(self.artifact.metadata),
        }


@dataclass(frozen=True, slots=True)
class CognitiveTaskResult:
    """Return final task status, module results, artifact references, workspace revision, and trace."""

    task_id: str
    run_id: str
    status: str
    plan: CognitiveTaskPlan
    module_results: Mapping[str, ModuleResult]
    artifacts: tuple[ArtifactRecord, ...]
    workspace_id: str
    workspace_revision: int
    workspace_state: Mapping[str, Any]
    trace: tuple[TraceEvent, ...]
    input_errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize execution metadata while omitting large cognitive payloads."""
        return {
            "task_id": self.task_id,
            "run_id": self.run_id,
            "status": self.status,
            "plan": self.plan.to_dict(),
            "module_results": {
                key: {
                    "status": value.status.value,
                    "version": value.module_version,
                    "diagnostics": dict(value.diagnostics),
                    "warnings": list(value.warnings),
                    "artifact_ids": [item.envelope.fingerprint() for item in value.artifacts],
                }
                for key, value in self.module_results.items()
            },
            "artifacts": [item.to_reference() for item in self.artifacts],
            "workspace_id": self.workspace_id,
            "workspace_revision": self.workspace_revision,
            "input_errors": list(self.input_errors),
            "trace": [
                {
                    "event_index": item.event_index,
                    "module_id": item.module_id,
                    "operation": item.operation,
                    "status": item.status,
                    "input_artifact_ids": list(item.input_artifact_ids),
                    "output_artifact_ids": list(item.output_artifact_ids),
                }
                for item in self.trace
            ],
        }


class ArtifactLedger:
    """Retain typed ContractArtifact objects and index their stable provenance references."""

    def __init__(self) -> None:
        """Initialize an isolated, thread-safe ledger for one cognitive task."""
        self._records: dict[str, ArtifactRecord] = {}
        self._lock = threading.RLock()

    def add(
        self,
        artifact: ContractArtifact,
        *,
        producer_module: str | None = None,
        external: bool = False,
    ) -> ArtifactRecord:
        """Index an artifact by envelope fingerprint and reject identity reuse with new content."""
        if not isinstance(artifact, ContractArtifact):
            raise TypeError("artifact must be a ContractArtifact")
        artifact_id = artifact.envelope.fingerprint()
        record = ArtifactRecord(
            artifact_id=artifact_id,
            contract_id=artifact.envelope.contract_id,
            schema_version=artifact.envelope.schema_version,
            producer_module=producer_module or artifact.envelope.producer_id,
            payload_hash=artifact.payload_hash,
            parent_artifact_ids=tuple(artifact.envelope.parent_artifact_ids),
            external=bool(external),
            artifact=artifact,
        )
        with self._lock:
            previous = self._records.get(artifact_id)
            if previous is not None:
                if previous.payload_hash != record.payload_hash:
                    raise ValueError("artifact envelope was reused for different payload content")
                return previous
            self._records[artifact_id] = record
        return record

    def get(self, artifact_id: str) -> ArtifactRecord | None:
        """Retrieve one record by its envelope fingerprint."""
        with self._lock:
            return self._records.get(artifact_id)

    def records(self) -> tuple[ArtifactRecord, ...]:
        """Return all ledger records in deterministic ID order."""
        with self._lock:
            return tuple(self._records[key] for key in sorted(self._records))

    def validate_lineage(self, artifact: ContractArtifact) -> tuple[str, ...]:
        """Return parent artifact IDs not present in this task's known artifact history."""
        with self._lock:
            known = set(self._records)
        return tuple(parent for parent in artifact.envelope.parent_artifact_ids if parent not in known)


class ModuleAvailabilityMonitor:
    """Inspect manifest registration, backend binding, validation, enablement, and dependencies."""

    def __init__(self, registry: ModuleRegistry) -> None:
        """Connect availability inspection to the module registry."""
        if not isinstance(registry, ModuleRegistry):
            raise TypeError("registry must be a ModuleRegistry")
        self.registry = registry

    def inspect(self, module_id: str) -> ModuleAvailability:
        """Return executable state even when upstream module dependencies still need planning."""
        manifest = self.registry.get_manifest(module_id)
        if manifest is None:
            return ModuleAvailability(module_id, False, "NOT_REGISTERED", False)
        diagnostics = self.registry.diagnostics(module_id)
        backend_bound = self.registry.get_backend(module_id) is not None
        validated = bool(diagnostics.get("backend_validated", False))
        disabled = bool(diagnostics.get("disabled", False))
        dependencies_ready = bool(diagnostics.get("dependency_complete", True))
        executable = backend_bound and validated and not disabled
        if disabled:
            status = "DISABLED"
        elif not backend_bound:
            status = "BACKEND_NOT_BOUND"
        elif not validated:
            status = "BACKEND_NOT_VALIDATED"
        elif not dependencies_ready:
            status = "DEPENDENCIES_PENDING"
        else:
            status = "READY"
        return ModuleAvailability(
            module_id, executable, status, dependencies_ready, dict(diagnostics),
        )


class DependencyGraphPlanner:
    """Discover suitable modules and construct a contract-aware directed acyclic execution plan."""

    def __init__(self, registry: ModuleRegistry, contract_catalog: ContractCatalog) -> None:
        """Bind planning to the same catalog used to validate registered module manifests."""
        if registry.contract_catalog is not contract_catalog:
            raise ValueError("registry and planner must use the same ContractCatalog")
        self.registry = registry
        self.catalog = contract_catalog
        self.availability = ModuleAvailabilityMonitor(registry)

    def _runnable(self, module_id: str, mode: str) -> bool:
        """Check for a validated backend and declared support for the requested operation mode."""
        manifest = self.registry.get_manifest(module_id)
        return bool(
            manifest is not None
            and mode in manifest.execution_modes
            and self.availability.inspect(module_id).executable
        )

    def _external_for_port(
        self, task: CognitiveTask, module_id: str, port: InputPort,
    ) -> tuple[ContractArtifact, ...]:
        """Resolve globally keyed or module-scoped task artifacts compatible with an input port."""
        value = task.input_artifacts.get(module_id)
        if isinstance(value, Mapping):
            value = value.get(port.port_id)
        if value is None:
            value = task.input_artifacts.get(port.port_id)
        if value is None:
            return ()
        values = value if isinstance(value, (tuple, list)) else (value,)
        return tuple(
            artifact for artifact in values
            if isinstance(artifact, ContractArtifact)
            and artifact.envelope.contract_id == port.contract_id
            and artifact.envelope.schema_version in port.accepted_versions
            and self.catalog.validate_artifact(artifact).valid
        )

    def _find_producer(self, port: InputPort, consumer: str, mode: str) -> str | None:
        """Choose a stable, executable producer whose declared output exactly matches the input contract."""
        matches = []
        for manifest in self.registry.list_manifests():
            module_id = manifest.identity.module_id
            if module_id == consumer or not self._runnable(module_id, mode):
                continue
            if any(
                output.contract_id == port.contract_id
                and output.contract_version in port.accepted_versions
                for output in manifest.output_ports
            ):
                matches.append(module_id)
        return sorted(matches)[0] if matches else None

    def build(self, task: CognitiveTask) -> CognitiveTaskPlan:
        """Discover capabilities, close required dependencies, resolve ports, and layer the task DAG."""
        selected: dict[str, ModuleManifest] = {}
        required_modules: set[str] = set()
        edges: set[tuple[str, str]] = set()
        port_producers: dict[tuple[str, str], str] = {}
        missing_capabilities: list[str] = []
        missing_dependencies: list[str] = []
        missing_inputs: list[str] = []
        optional_gaps: list[str] = []
        pending: deque[str] = deque()
        visited: set[str] = set()
        matched = 0

        def include(module_id: str, required: bool = False) -> None:
            """Add a runnable manifest once and queue it for recursive dependency inspection."""
            manifest = self.registry.get_manifest(module_id)
            if manifest is None or not self._runnable(module_id, task.execution_mode):
                if required:
                    missing_dependencies.append(module_id)
                return
            if module_id not in selected:
                selected[module_id] = manifest
                pending.append(module_id)
            if required:
                required_modules.add(module_id)

        for capability in task.required_capabilities:
            providers = sorted(
                record.module_id
                for record in self.registry.find_by_capability(capability)
                if self._runnable(record.module_id, task.execution_mode)
            )
            if not providers:
                missing_capabilities.append(capability)
            else:
                include(providers[0], required=True)
                matched += 1

        for capability in task.optional_capabilities:
            providers = sorted(
                record.module_id
                for record in self.registry.find_by_capability(capability)
                if self._runnable(record.module_id, task.execution_mode)
            )
            if providers:
                include(providers[0])
            else:
                optional_gaps.append(capability)

        while pending:
            module_id = pending.popleft()
            if module_id in visited:
                continue
            visited.add(module_id)
            manifest = selected[module_id]
            for dependency in manifest.dependencies:
                dependency_id = dependency.module_id
                if self._runnable(dependency_id, task.execution_mode):
                    include(dependency_id, required=dependency.required)
                    if dependency_id in selected:
                        edges.add((dependency_id, module_id))
                elif dependency.required:
                    missing_dependencies.append(f"{module_id} requires {dependency_id}")
                else:
                    optional_gaps.append(f"{module_id} optional dependency {dependency_id}")

            for port in manifest.input_ports:
                if self._external_for_port(task, module_id, port):
                    continue
                producer = self._find_producer(port, module_id, task.execution_mode)
                if producer is not None:
                    include(producer)
                    port_producers[(module_id, port.port_id)] = producer
                    edges.add((producer, module_id))
                elif port.required:
                    missing_inputs.append(f"{module_id}.{port.port_id} requires {port.contract_id}")
                else:
                    optional_gaps.append(f"{module_id}.{port.port_id}")

        input_plans: dict[str, tuple[ModuleInputPlan, ...]] = {}
        compatibility: dict[str, dict[str, int]] = {}
        for module_id, manifest in sorted(selected.items()):
            rows = []
            compatibility[module_id] = {}
            for port in manifest.input_ports:
                external = self._external_for_port(task, module_id, port)
                producer = port_producers.get((module_id, port.port_id))
                missing = bool(port.required and not external and producer is None)
                compatible = bool(external or producer)
                compatibility[module_id][port.port_id] = int(compatible)
                rows.append(ModuleInputPlan(
                    module_id, port.port_id, port.contract_id,
                    tuple(port.accepted_versions), port.required, port.multiple,
                    tuple(item.envelope.fingerprint() for item in external),
                    (producer,) if producer else (), missing,
                ))
            input_plans[module_id] = tuple(rows)

        layers, cycles = self._topological_layers(set(selected), edges)
        coverage = matched / len(task.required_capabilities) if task.required_capabilities else 1.0
        return CognitiveTaskPlan(
            selected_modules=tuple(sorted(selected)),
            required_modules=tuple(sorted(required_modules)),
            edges=tuple(sorted(edges)),
            layers=layers,
            input_plans=input_plans,
            missing_capabilities=tuple(sorted(set(missing_capabilities))),
            missing_dependencies=tuple(sorted(set(missing_dependencies))),
            missing_inputs=tuple(sorted(set(missing_inputs))),
            cycle_nodes=cycles,
            optional_gaps=tuple(sorted(set(optional_gaps))),
            capability_coverage=coverage,
            compatibility_matrix=compatibility,
        )

    @staticmethod
    def _topological_layers(
        nodes: set[str],
        edges: Iterable[tuple[str, str]],
    ) -> tuple[tuple[tuple[str, ...], ...], tuple[str, ...]]:
        """Apply Kahn's algorithm in parallelizable layers and return nodes blocked by cycles."""
        remaining = set(nodes)
        connections = tuple(edges)
        layers = []
        while remaining:
            ready = sorted(
                node for node in remaining
                if not any(target == node and source in remaining for source, target in connections)
            )
            if not ready:
                break
            layers.append(tuple(ready))
            remaining.difference_update(ready)
        return tuple(layers), tuple(sorted(remaining))


def calculate_module_priority(
    *,
    relevance: float,
    downstream_importance: float,
    urgency: float,
    reliability: float | None = None,
    resource_cost: float | None = None,
    alpha: float = 0.4,
    beta: float = 0.3,
    gamma: float = 0.3,
    delta: float = 0.0,
    resource_penalty: float = 0.0,
) -> float:
    """Calculate a normalized priority score and require data for any enabled optional term."""
    values = (relevance, downstream_importance, urgency, alpha, beta, gamma, delta, resource_penalty)
    if any(not math.isfinite(float(item)) for item in values):
        raise ValueError("priority inputs and weights must be finite")
    if any(not 0.0 <= float(item) <= 1.0 for item in (relevance, downstream_importance, urgency)):
        raise ValueError("relevance, downstream_importance and urgency must be in [0, 1]")
    if delta and reliability is None:
        raise ValueError("reliability is required when its weight is non-zero")
    if resource_penalty and resource_cost is None:
        raise ValueError("resource_cost is required when its penalty is non-zero")
    if reliability is not None and not 0.0 <= reliability <= 1.0:
        raise ValueError("reliability must be in [0, 1]")
    if resource_cost is not None and not 0.0 <= resource_cost <= 1.0:
        raise ValueError("resource_cost must be in [0, 1]")
    return (
        alpha * relevance
        + beta * downstream_importance
        + gamma * urgency
        + delta * (reliability or 0.0)
        - resource_penalty * (resource_cost or 0.0)
    )


def normalized_entropy(probabilities: Sequence[float], epsilon: float = 1e-12) -> float | None:
    """Compute normalized Shannon entropy only for a finite probability distribution."""
    values = tuple(float(item) for item in probabilities)
    if len(values) < 2:
        return None
    if any(not math.isfinite(item) or item < 0.0 or item > 1.0 for item in values):
        raise ValueError("probabilities must be finite values in [0, 1]")
    if not math.isclose(sum(values), 1.0, rel_tol=1e-7, abs_tol=1e-9):
        raise ValueError("probabilities must sum to one")
    if epsilon <= 0.0 or not math.isfinite(epsilon):
        raise ValueError("epsilon must be finite and positive")
    entropy = -sum(item * math.log(item + epsilon) for item in values)
    return min(1.0, max(0.0, entropy / math.log(len(values))))


def weighted_disagreement(
    observations: Iterable[tuple[float, float] | Mapping[str, float]],
) -> float | None:
    """Calculate weighted population variance for comparable finite scores and positive weights."""
    pairs = []
    for row in observations:
        score, weight = (
            (float(row["score"]), float(row["weight"]))
            if isinstance(row, Mapping)
            else (float(row[0]), float(row[1]))
        )
        if not math.isfinite(score) or not math.isfinite(weight) or weight <= 0.0:
            raise ValueError("scores must be finite and weights finite and positive")
        pairs.append((score, weight))
    if not pairs:
        return None
    total = sum(weight for _, weight in pairs)
    mean = sum(score * weight for score, weight in pairs) / total
    return sum(weight * (score - mean) ** 2 for score, weight in pairs) / total


class CognitiveTaskCoordinator:
    """Execute the planned DAG with bounded workers and controlled artifact publication."""

    def __init__(
        self,
        registry: ModuleRegistry,
        contract_catalog: ContractCatalog,
        *,
        workspace_manager: WorkspaceManager | None = None,
        repository_root: str | Path | None = None,
        logger_database_path: str | Path | None = None,
        enable_logging: bool = True,
        max_workers: int = 4,
        default_timeout_seconds: float | None = None,
    ) -> None:
        """Connect the task coordinator to registries, shared workspace, worker limits, and SQLite."""
        if registry.contract_catalog is not contract_catalog:
            raise ValueError("registry and coordinator must use the same ContractCatalog")
        if max_workers < 1:
            raise ValueError("max_workers must be at least one")
        if default_timeout_seconds is not None and default_timeout_seconds <= 0:
            raise ValueError("default_timeout_seconds must be positive")
        self.registry = registry
        self.catalog = contract_catalog
        self.workspace_manager = workspace_manager
        self.repository_root = (
            Path(repository_root).resolve() if repository_root else Path(__file__).resolve().parents[1]
        )
        self.logger_database_path = Path(logger_database_path).resolve() if logger_database_path else None
        self.enable_logging = enable_logging
        self.max_workers = max_workers
        self.default_timeout_seconds = default_timeout_seconds
        self.planner = DependencyGraphPlanner(registry, contract_catalog)

    @staticmethod
    def _record(
        logger: SQLiteModuleEventLogger | None,
        class_name: str,
        method_name: str,
        metadata: Mapping[str, Any],
    ) -> None:
        """Persist a compact event linked to the exact class and method annotations."""
        if logger is not None:
            logger.record(
                method_name, metadata, class_name=class_name,
                module_name="ore_contracts.coordinator",
            )

    @staticmethod
    def _context(task: CognitiveTask, context: RunContext | None) -> RunContext:
        """Reuse a supplied run context or create a reproducible task-scoped context."""
        if context is None:
            return RunContext(
                task.task_id, OperationMode(task.execution_mode), 0, ComputeBudget(),
            )
        if not isinstance(context, RunContext):
            raise TypeError("context must be a RunContext")
        if context.mode.value != task.execution_mode:
            raise ValueError("task execution mode and RunContext.mode must match")
        return context

    @staticmethod
    def _inputs(
        manifest: ModuleManifest,
        plans: tuple[ModuleInputPlan, ...],
        results: Mapping[str, ModuleResult],
        ledger: ArtifactLedger,
    ) -> tuple[dict[str, Any], tuple[str, ...], tuple[str, ...]]:
        """Assemble exact-version port data from external artifacts and completed producers."""
        inputs = {}
        missing = []
        consumed = []
        by_port = {item.port_id: item for item in plans}
        for port in manifest.input_ports:
            plan = by_port.get(port.port_id)
            artifacts = []
            if plan is not None:
                artifacts.extend(
                    record.artifact
                    for artifact_id in plan.external_artifact_ids
                    if (record := ledger.get(artifact_id)) is not None
                )
                for producer_id in plan.producer_module_ids:
                    upstream = results.get(producer_id)
                    if upstream is not None:
                        artifacts.extend(
                            artifact for artifact in upstream.artifacts
                            if artifact.envelope.contract_id == port.contract_id
                            and artifact.envelope.schema_version in port.accepted_versions
                        )
            unique = {item.envelope.fingerprint(): item for item in artifacts}
            artifacts = [unique[key] for key in sorted(unique)]
            if not artifacts:
                if port.required:
                    missing.append(port.port_id)
                continue
            if not port.multiple and len(artifacts) != 1:
                missing.append(f"{port.port_id}: expected one artifact, received {len(artifacts)}")
                continue
            inputs[port.port_id] = tuple(artifacts) if port.multiple else artifacts[0]
            consumed.extend(item.envelope.fingerprint() for item in artifacts)
        return inputs, tuple(missing), tuple(sorted(set(consumed)))

    def _validate_result(
        self,
        module_id: str,
        manifest: ModuleManifest,
        result: ModuleResult,
        context: RunContext,
        ledger: ArtifactLedger,
    ) -> tuple[ModuleResult, tuple[ArtifactRecord, ...], tuple[str, ...]]:
        """Reject results with wrong identity, schema, producer, run, output port, or lineage."""
        if not isinstance(result, ModuleResult):
            raise TypeError("module backend must return ModuleResult")
        if result.module_id != module_id or result.module_version != manifest.identity.module_version:
            raise ValueError("module result identity/version does not match its registered manifest")
        candidates = []
        errors = []
        for artifact in result.artifacts:
            if not isinstance(artifact, ContractArtifact):
                errors.append("output is not a ContractArtifact")
                continue
            envelope = artifact.envelope
            declared = any(
                port.contract_id == envelope.contract_id
                and port.contract_version == envelope.schema_version
                for port in manifest.output_ports
            )
            if not declared:
                errors.append(f"undeclared output {envelope.contract_id}@{envelope.schema_version}")
                continue
            if envelope.producer_id != module_id or envelope.run_id != context.run_id:
                errors.append("artifact producer_id or run_id does not match this execution")
                continue
            validation = self.catalog.validate_artifact(artifact)
            if not validation.valid:
                errors.extend(validation.errors)
                continue
            unknown = ledger.validate_lineage(artifact)
            if unknown:
                errors.append("unknown artifact parents: " + ", ".join(unknown))
                continue
            candidates.append(artifact)
        if errors:
            failed = ModuleResult(
                module_id, manifest.identity.module_version, ModuleStatus.FAILED,
                diagnostics={**dict(result.diagnostics), "output_validation_errors": errors},
                warnings=tuple(result.warnings) + ("Rejected invalid output artifacts.",),
                provenance=dict(result.provenance),
                processing_metadata=dict(result.processing_metadata),
            )
            return failed, (), tuple(errors)

        records = tuple(
            ledger.add(item, producer_module=module_id)
            for item in candidates
        )
        status = result.status
        missing_always = [
            port.port_id for port in manifest.output_ports
            if port.availability == "always"
            and not any(
                item.envelope.contract_id == port.contract_id
                and item.envelope.schema_version == port.contract_version
                for item in candidates
            )
        ]
        if status is ModuleStatus.SUCCESS and missing_always:
            status = ModuleStatus.PARTIAL
            result = ModuleResult(
                module_id, manifest.identity.module_version, status, tuple(candidates),
                diagnostics={**dict(result.diagnostics), "missing_declared_outputs": missing_always},
                warnings=tuple(result.warnings) + ("Always-available output ports were not all produced.",),
                provenance=dict(result.provenance),
                processing_metadata=dict(result.processing_metadata),
            )
        return result, records, ()

    @staticmethod
    def _publish_status(workspace: Any, module_id: str, status: str) -> None:
        """Publish module lifecycle state through the coordinator's declared workspace permission."""
        publisher = FindingPublisher(workspace, "cognitive_control")
        finding = publisher.create_finding(
            "module_status", {"status": status}, key=module_id,
            finding_type="module_execution_status",
            provenance=Provenance(
                source="cognitive_task_coordinator",
                rules_applied=("module_result_status",),
            ),
        )
        workspace.publish(finding)

    @staticmethod
    def _publish_artifact(workspace: Any, record: ArtifactRecord) -> str:
        """Publish artifact identity, hash, exact schema, and parent lineage without its payload."""
        publisher = FindingPublisher(workspace, "cognitive_control")
        finding = publisher.create_finding(
            "artifact_references", record.to_reference(),
            finding_type="typed_artifact_reference",
            provenance=Provenance(
                source="cognitive_task_coordinator",
                rules_applied=("contract_validation", "artifact_ledger"),
                evidence=({"artifact_id": record.artifact_id, "payload_hash": record.payload_hash},),
            ),
        )
        return workspace.publish(finding).decision.value

    def coordinate(
        self,
        task: CognitiveTask,
        context: RunContext | None = None,
    ) -> CognitiveTaskResult:
        """Plan, execute, validate, publish, and trace one dependency-aware cognitive task."""
        if not isinstance(task, CognitiveTask):
            raise TypeError("task must be a CognitiveTask")
        run_context = self._context(task, context)
        own_manager = self.workspace_manager is None
        manager = self.workspace_manager or WorkspaceManager(
            repository_root=self.repository_root,
            logger_database_path=self.logger_database_path,
            enable_logging=False,
        )
        logging = (
            SQLiteModuleEventLogger(
                self.logger_database_path, self.repository_root,
                display_label="ORE cognitive task coordinator",
            )
            if self.enable_logging else nullcontext(None)
        )
        try:
            with logging as logger:
                input_text = task.metadata.get("input_text", task.goal)
                if not isinstance(input_text, str):
                    raise TypeError("metadata.input_text must be a string")
                workspace = manager.create_workspace(input_text, task.goal)
                ledger = ArtifactLedger()
                input_errors = []
                for label, artifact in _iter_task_artifacts(task):
                    if not isinstance(artifact, ContractArtifact):
                        input_errors.append(f"{label}: input is not a ContractArtifact")
                        continue
                    validation = self.catalog.validate_artifact(artifact)
                    if not validation.valid:
                        input_errors.extend(f"{label}: {item}" for item in validation.errors)
                        continue
                    try:
                        ledger.add(artifact, external=True)
                    except (TypeError, ValueError) as exc:
                        input_errors.append(f"{label}: {type(exc).__name__}: {exc}")

                self._record(logger, "CognitiveTaskCoordinator", "coordinate", {
                    "task_id": task.task_id,
                    "run_id": run_context.run_id,
                    "input_fingerprint": task.input_fingerprint(),
                    "required_capabilities": list(task.required_capabilities),
                    "max_workers": self.max_workers,
                })
                plan = self.planner.build(task)
                self._record(logger, "DependencyGraphPlanner", "build", {
                    "task_id": task.task_id,
                    "selected_modules": list(plan.selected_modules),
                    "layers": [list(item) for item in plan.layers],
                    "edges": [list(item) for item in plan.edges],
                    "capability_coverage": plan.capability_coverage,
                    "cycle_nodes": list(plan.cycle_nodes),
                })

                results: dict[str, ModuleResult] = {}
                statuses: dict[str, str] = {}
                trace: list[TraceEvent] = []
                started = time.monotonic()
                timeout_map = dict(run_context.timeout_limits_seconds)
                executor = ThreadPoolExecutor(
                    max_workers=self.max_workers, thread_name_prefix="ore-brain-module",
                )
                try:
                    for layer in plan.layers:
                        pending: dict[str, tuple[Future, tuple[str, ...], float | None]] = {}
                        order = sorted(
                            layer,
                            key=lambda module_id: (
                                -calculate_module_priority(
                                    relevance=float(module_id in plan.required_modules),
                                    downstream_importance=(
                                        sum(1 for source, _ in plan.edges if source == module_id)
                                        / max(1, len(plan.selected_modules) - 1)
                                    ),
                                    urgency=float(task.priority),
                                ),
                                module_id,
                            ),
                        )
                        for module_id in order:
                            manifest = self.registry.get_manifest(module_id)
                            if manifest is None:
                                continue
                            blocked = [
                                dependency.module_id for dependency in manifest.dependencies
                                if dependency.required and statuses.get(dependency.module_id) != "SUCCESS"
                            ]
                            if blocked:
                                statuses[module_id] = "BLOCKED"
                                results[module_id] = ModuleResult(
                                    module_id, manifest.identity.module_version, ModuleStatus.UNAVAILABLE,
                                    diagnostics={"blocked_dependencies": blocked},
                                )
                                self._publish_status(workspace, module_id, "BLOCKED")
                                trace.append(TraceEvent(
                                    len(trace), utc_now_iso(), module_id, "block_on_dependency",
                                    (), (), "BLOCKED",
                                    numerical_diagnostics={"blocked_dependencies": blocked},
                                ))
                                continue

                            inputs, missing, consumed = self._inputs(
                                manifest, plan.input_plans.get(module_id, ()), results, ledger,
                            )
                            if missing:
                                statuses[module_id] = "BLOCKED"
                                results[module_id] = ModuleResult(
                                    module_id, manifest.identity.module_version, ModuleStatus.UNAVAILABLE,
                                    diagnostics={"missing_inputs": list(missing)},
                                )
                                self._publish_status(workspace, module_id, "BLOCKED")
                                trace.append(TraceEvent(
                                    len(trace), utc_now_iso(), module_id, "block_on_missing_input",
                                    consumed, (), "BLOCKED",
                                    numerical_diagnostics={"missing_inputs": list(missing)},
                                ))
                                continue

                            availability = self.planner.availability.inspect(module_id)
                            backend = self.registry.get_backend(module_id)
                            if not availability.executable or backend is None:
                                statuses[module_id] = "UNAVAILABLE"
                                results[module_id] = ModuleResult(
                                    module_id, manifest.identity.module_version, ModuleStatus.UNAVAILABLE,
                                    diagnostics={"availability": availability.status},
                                )
                                self._publish_status(workspace, module_id, "UNAVAILABLE")
                                continue

                            budget = run_context.compute_budget.wall_clock_seconds
                            remaining = None if budget is None else max(
                                0.0, budget - (time.monotonic() - started),
                            )
                            if remaining is not None and remaining <= 0:
                                statuses[module_id] = "FAILED"
                                results[module_id] = ModuleResult(
                                    module_id, manifest.identity.module_version, ModuleStatus.FAILED,
                                    diagnostics={"error": "task compute budget exhausted"},
                                )
                                self._publish_status(workspace, module_id, "FAILED")
                                continue
                            timeout = timeout_map.get(module_id, self.default_timeout_seconds)
                            if remaining is not None:
                                timeout = remaining if timeout is None else min(timeout, remaining)
                            future = executor.submit(backend.execute, inputs, run_context)
                            pending[module_id] = (future, consumed, timeout)

                        for module_id in sorted(pending):
                            future, consumed, timeout = pending[module_id]
                            manifest = self.registry.get_manifest(module_id)
                            call_started = time.monotonic()
                            try:
                                raw = future.result(timeout=timeout)
                                validated, records, errors = self._validate_result(
                                    module_id, manifest, raw, run_context, ledger,
                                )
                                results[module_id] = validated
                                statuses[module_id] = "FAILED" if errors else validated.status.value.upper()
                                output_ids = tuple(item.artifact_id for item in records)
                                for record in records:
                                    decision = self._publish_artifact(workspace, record)
                                    self._record(logger, "CognitiveTaskCoordinator", "_publish_artifact", {
                                        "task_id": task.task_id,
                                        "module_id": module_id,
                                        "artifact_id": record.artifact_id,
                                        "payload_hash": record.payload_hash,
                                        "workspace_decision": decision,
                                        "workspace_revision": workspace.revision,
                                    })
                                self._publish_status(workspace, module_id, statuses[module_id])
                                elapsed = round((time.monotonic() - call_started) * 1000.0, 3)
                                trace.append(TraceEvent(
                                    len(trace), utc_now_iso(), module_id, "execute_module",
                                    consumed, output_ids, statuses[module_id],
                                    simulation_time_ms=elapsed,
                                    numerical_diagnostics={
                                        "input_count": len(consumed),
                                        "output_count": len(output_ids),
                                        "validation_errors": list(errors),
                                    },
                                ))
                                self._record(logger, "CognitiveTaskCoordinator", "coordinate", {
                                    "task_id": task.task_id,
                                    "module_id": module_id,
                                    "operation": "execute_module",
                                    "input_artifact_ids": list(consumed),
                                    "output_artifact_ids": list(output_ids),
                                    "workspace_revision": workspace.revision,
                                    "execution_status": statuses[module_id],
                                    "duration_ms": elapsed,
                                })
                            except FutureTimeoutError:
                                future.cancel()
                                statuses[module_id] = "FAILED"
                                results[module_id] = ModuleResult(
                                    module_id, manifest.identity.module_version, ModuleStatus.FAILED,
                                    diagnostics={"error": "module execution timed out", "timeout_seconds": timeout},
                                )
                                self._publish_status(workspace, module_id, "FAILED")
                            except Exception as exc:
                                statuses[module_id] = "FAILED"
                                results[module_id] = ModuleResult(
                                    module_id, manifest.identity.module_version, ModuleStatus.FAILED,
                                    diagnostics={"exception_type": type(exc).__name__, "error": str(exc)},
                                )
                                self._publish_status(workspace, module_id, "FAILED")
                            if statuses[module_id] == "FAILED":
                                trace.append(TraceEvent(
                                    len(trace), utc_now_iso(), module_id, "module_execution_failed",
                                    consumed, (), "FAILED",
                                ))
                finally:
                    executor.shutdown(wait=False, cancel_futures=True)

                for module_id in plan.cycle_nodes:
                    manifest = self.registry.get_manifest(module_id)
                    if manifest is not None and module_id not in results:
                        results[module_id] = ModuleResult(
                            module_id, manifest.identity.module_version, ModuleStatus.FAILED,
                            diagnostics={"error": "dependency cycle prevents execution"},
                        )
                    statuses[module_id] = "FAILED"
                    self._publish_status(workspace, module_id, "FAILED")
                    trace.append(TraceEvent(
                        len(trace), utc_now_iso(), module_id, "reject_dependency_cycle",
                        (), (), "FAILED",
                    ))

                missing_questions = (
                    [f"Missing capability: {item}" for item in plan.missing_capabilities]
                    + [f"Missing dependency: {item}" for item in plan.missing_dependencies]
                    + [f"Missing required input: {item}" for item in plan.missing_inputs]
                )
                question_publisher = FindingPublisher(workspace, "cognitive_control")
                for index, question in enumerate(missing_questions):
                    finding = question_publisher.create_finding(
                        "unresolved_questions",
                        {
                            "question_id": f"{task.task_id}:missing:{index}",
                            "question": question,
                            "status": "unresolved",
                            "task_id": task.task_id,
                        },
                        finding_type="missing_information_request",
                        provenance=Provenance(
                            source="cognitive_task_coordinator",
                            rules_applied=("dependency_plan_validation",),
                        ),
                    )
                    workspace.publish(finding)

                produced = [item for item in ledger.records() if not item.external]
                output_gaps = [
                    item for item in task.expected_outputs
                    if not any(record.contract_id == item for record in produced)
                    and not any(
                        output.port_id == item and any(
                            record.producer_module == module_id
                            and record.contract_id == output.contract_id
                            and record.schema_version == output.contract_version
                            for record in produced
                        )
                        for module_id in plan.selected_modules
                        for output in self.registry.get_manifest(module_id).output_ports
                    )
                ]
                open_conflicts = workspace.conflicts.list(workspace.workspace_id, unresolved_only=True)
                success_count = sum(item.status is ModuleStatus.SUCCESS for item in results.values())
                failure_count = sum(item.status is ModuleStatus.FAILED for item in results.values())
                partial_count = sum(item.status is ModuleStatus.PARTIAL for item in results.values())
                unavailable_count = sum(item.status is ModuleStatus.UNAVAILABLE for item in results.values())
                if open_conflicts:
                    final_status = "AMBIGUOUS"
                elif plan.cycle_nodes or (failure_count and not success_count):
                    final_status = "FAILED"
                elif not plan.selected_modules and (
                    plan.missing_capabilities or plan.missing_dependencies or plan.missing_inputs
                ):
                    final_status = "BLOCKED"
                elif (
                    plan.missing_capabilities or plan.missing_dependencies or plan.missing_inputs
                    or output_gaps or failure_count or partial_count or unavailable_count
                ):
                    final_status = "PARTIAL" if success_count or produced else "BLOCKED"
                elif any(item.status is ModuleStatus.ABSTAINED for item in results.values()):
                    final_status = "ABSTAINED"
                elif produced:
                    final_status = "RESOLVED"
                else:
                    final_status = "ABSTAINED"

                trace.append(TraceEvent(
                    len(trace), utc_now_iso(), "B-N9", "finalize_cognitive_task",
                    (), tuple(item.artifact_id for item in ledger.records()), final_status,
                    numerical_diagnostics={
                        "capability_coverage": plan.capability_coverage,
                        "output_gaps": output_gaps,
                        "artifact_count": len(ledger.records()),
                    },
                ))
                snapshot = workspace.snapshot()
                self._record(logger, "CognitiveTaskCoordinator", "coordinate", {
                    "task_id": task.task_id,
                    "final_status": final_status,
                    "workspace_revision": snapshot.revision,
                    "artifact_count": len(ledger.records()),
                    "input_error_count": len(input_errors),
                })
                return CognitiveTaskResult(
                    task_id=task.task_id,
                    run_id=run_context.run_id,
                    status=final_status,
                    plan=plan,
                    module_results=dict(results),
                    artifacts=ledger.records(),
                    workspace_id=workspace.workspace_id,
                    workspace_revision=snapshot.revision,
                    workspace_state=dict(snapshot.state),
                    trace=tuple(trace),
                    input_errors=tuple(input_errors),
                )
        finally:
            if own_manager:
                manager.close()


def _artifact_values(value: Any) -> tuple[Any, ...]:
    """Normalize an optional artifact or artifact sequence without hiding invalid values."""
    if value is None:
        return ()
    return tuple(value) if isinstance(value, (tuple, list)) else (value,)


def _iter_task_artifacts(task: CognitiveTask) -> Iterable[tuple[str, Any]]:
    """Yield flat port inputs and nested module-to-port inputs with stable diagnostic labels."""
    for key, value in task.input_artifacts.items():
        if isinstance(value, Mapping):
            for port_id, nested in value.items():
                for index, artifact in enumerate(_artifact_values(nested)):
                    yield f"{key}.{port_id}[{index}]", artifact
        else:
            for index, artifact in enumerate(_artifact_values(value)):
                yield f"{key}[{index}]", artifact


__all__ = [
    "ArtifactLedger", "ArtifactRecord", "CognitiveTask", "CognitiveTaskCoordinator",
    "CognitiveTaskPlan", "CognitiveTaskResult", "DependencyGraphPlanner",
    "ModuleAvailability", "ModuleAvailabilityMonitor", "ModuleInputPlan",
    "calculate_module_priority", "normalized_entropy", "weighted_disagreement",
]
