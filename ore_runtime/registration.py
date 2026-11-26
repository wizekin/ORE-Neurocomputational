"""Register the runtime engine as an ORE module with contracts and capabilities."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .contracts import EngineConfig, TaskSpec
from .engine import CognitiveExecutionEngine

RUNTIME_MODULE_ID = "ore.runtime.engine"
RUNTIME_MODULE_VERSION = "0.1.0"
RUNTIME_CAPABILITIES = (
    "runtime.concurrent_execution",
    "runtime.event_coordination",
    "runtime.incremental_recompute",
)
RUNTIME_CONTRACT_IDS = {
    "goal": "ORE.RUNTIME.GOAL",
    "task_manifest": "ORE.RUNTIME.TASK_MANIFEST",
    "execution_report": "ORE.RUNTIME.EXECUTION_REPORT",
}
RUNTIME_CONTRACT_VERSION = "0.1.0"


def register_runtime_contracts(catalog: Any) -> None:
    """Register the runtime goal, manifest, and report contracts in the catalog."""
    from ore_contracts.contracts_registry import ContractDefinition
    catalog.register_contract(ContractDefinition(
        contract_id=RUNTIME_CONTRACT_IDS["goal"],
        version=RUNTIME_CONTRACT_VERSION,
        model_type=dict,
        required_fields=("goal_id", "description", "correlation_id"),
        optional_fields=("constraints", "expected_outputs", "inputs"),
        description="A validated cognitive goal submitted to the runtime engine.",
    ))
    catalog.register_contract(ContractDefinition(
        contract_id=RUNTIME_CONTRACT_IDS["task_manifest"],
        version=RUNTIME_CONTRACT_VERSION,
        model_type=dict,
        required_fields=("tasks",),
        optional_fields=("iterative_regions", "executors"),
        description="A typed task manifest for concurrent execution planning.",
    ))
    catalog.register_contract(ContractDefinition(
        contract_id=RUNTIME_CONTRACT_IDS["execution_report"],
        version=RUNTIME_CONTRACT_VERSION,
        model_type=dict,
        required_fields=("goal_id", "status", "correlation_id"),
        optional_fields=(
            "succeeded_tasks", "failed_tasks", "committed_revisions",
            "events_emitted", "final_outputs", "trace",
        ),
        description="An audited goal execution report returned by the engine.",
    ))


def build_runtime_manifest() -> Any:
    """Build the runtime engine's module manifest for the ORE registry."""
    from ore_contracts.module_manifest import (
        InputPort, ModuleDependency, ModuleIdentity, ModuleManifest, OutputPort,
    )
    return ModuleManifest(
        identity=ModuleIdentity(
            module_id=RUNTIME_MODULE_ID,
            module_version=RUNTIME_MODULE_VERSION,
            module_name="ORE Runtime Concurrent Execution Engine",
            package_path="ore_runtime",
            biological_reference=(
                "Point 19: concurrency, synchronization, event-driven systems, "
                "and incremental computation for cognitive execution"
            ),
        ),
        responsibilities=(
            "Compile cognitive goals into dependency-aware concurrent task plans",
            "Coordinate versioned shared state with optimistic commits",
            "Coordinate modules through durable, idempotent event publication",
            "Invalidate and recompute derived conclusions incrementally",
            "Reproduce concurrency failures with deterministic schedules",
        ),
        input_ports=(
            InputPort(
                port_id="goal",
                contract_id=RUNTIME_CONTRACT_IDS["goal"],
                accepted_versions=(RUNTIME_CONTRACT_VERSION,),
                required=True,
            ),
            InputPort(
                port_id="task_manifest",
                contract_id=RUNTIME_CONTRACT_IDS["task_manifest"],
                accepted_versions=(RUNTIME_CONTRACT_VERSION,),
                required=True,
            ),
        ),
        output_ports=(
            OutputPort(
                port_id="execution_report",
                contract_id=RUNTIME_CONTRACT_IDS["execution_report"],
                contract_version=RUNTIME_CONTRACT_VERSION,
                output_data_type="object",
                description="Audited goal execution report with full trace",
            ),
        ),
        capabilities=RUNTIME_CAPABILITIES,
        dependencies=(
            ModuleDependency(
                module_id="ore_workspace", required=False,
                purpose="Shared cognitive workspace publication of task findings",
            ),
        ),
        execution_modes=("FUNCTIONAL",),
        limitations=(
            "Single-process implementation; distributed stream processing is "
            "deliberately deferred until workload measurements justify it",
        ),
    )


class RuntimeEngineModuleBackend:
    """Serve the runtime engine through the ORE module protocol."""

    def __init__(self, config: EngineConfig | None = None) -> None:
        """Initialize the backend with its own engine instance."""
        self._config = config or EngineConfig()
        self._engine = CognitiveExecutionEngine(self._config)
        self._manifest = build_runtime_manifest()

    @property
    def engine(self) -> CognitiveExecutionEngine:
        """Return the backing cognitive execution engine."""
        return self._engine

    def describe(self) -> Any:
        """Return the module manifest describing ports and capabilities."""
        return self._manifest

    def execute(self, inputs: Mapping[str, Any], context: Any) -> Any:
        """Execute one goal manifest submitted through the module protocol."""
        from ore_contracts.module_protocol import ModuleResult, ModuleStatus
        goal = dict(inputs.get("goal", {}))
        manifest = dict(inputs.get("task_manifest", {}))
        specs = [
            TaskSpec(**spec) if isinstance(spec, dict) else spec
            for spec in manifest.get("tasks", ())
        ]
        executors = {
            key: callable for key, callable in
            (inputs.get("executors") or {}).items()
        }
        report = self._engine.execute_goal(goal, specs, executors)
        return ModuleResult(
            module_id=RUNTIME_MODULE_ID,
            module_version=RUNTIME_MODULE_VERSION,
            status=ModuleStatus.SUCCESS if report.status == "succeeded"
            else ModuleStatus.PARTIAL,
            diagnostics={"report": report.to_dict()},
            provenance={"correlation_id": report.correlation_id},
            processing_metadata={"rounds": report.rounds},
        )

    def close(self) -> None:
        """Close the backing engine deterministically."""
        self._engine.close()


def register_runtime_module(registry: Any, catalog: Any) -> RuntimeEngineModuleBackend:
    """Register contracts, the runtime manifest, and bind the engine backend."""
    register_runtime_contracts(catalog)
    registry.register(build_runtime_manifest())
    backend = RuntimeEngineModuleBackend()
    registry.bind_backend(RUNTIME_MODULE_ID, backend)
    return backend


def runtime_module_specs() -> dict[str, Any]:
    """Return the runtime module identity and contract identifiers."""
    return {
        "module_id": RUNTIME_MODULE_ID,
        "module_version": RUNTIME_MODULE_VERSION,
        "capabilities": list(RUNTIME_CAPABILITIES),
        "contracts": dict(RUNTIME_CONTRACT_IDS),
        "contract_version": RUNTIME_CONTRACT_VERSION,
    }
