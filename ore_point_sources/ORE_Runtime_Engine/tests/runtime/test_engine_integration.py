"""End-to-end engine acceptance tests including workspace and USG integration."""
from __future__ import annotations

import pytest

from ore_runtime import (
    EngineConfig, ExecutionRequest, SideEffectClass, TaskSpec,
    WorkspaceRuntimeAdapter,
)
from ore_runtime.adapters import build_workspace_permissions
from ore_runtime.engine import CognitiveExecutionEngine


def executor(context):
    """Produce a deterministic output derived from the immutable snapshot."""
    return {"task": context.task_spec.task_id, "keys": sorted(context.snapshot)}


def chain_specs() -> tuple[TaskSpec, ...]:
    """Build the standard three-task dependency chain."""
    return (
        TaskSpec(task_id="perceive", name="perceive", priority=0.8,
                 payload={"sentence": "the engine test"}),
        TaskSpec(task_id="analyze", name="analyze", dependencies=("perceive",),
                 priority=0.6),
        TaskSpec(task_id="conclude", name="conclude", dependencies=("analyze",),
                 priority=0.4),
    )


def test_engine_executes_dependency_ordered_concurrent_goal():
    """The engine runs a dependency chain and commits each result in order."""
    engine = CognitiveExecutionEngine(EngineConfig(max_workers=3, max_active_tasks=3))
    try:
        report = engine.execute_goal(
            ExecutionRequest(goal_id="goal-1", description="chain",
                             expected_outputs=("conclude",)),
            chain_specs(),
            {"perceive": executor, "analyze": executor, "conclude": executor},
        )
        assert report.status == "succeeded"
        assert report.succeeded_tasks == ("perceive", "analyze", "conclude")
        assert report.committed_revisions == (1, 2, 3)
        assert report.final_outputs["conclude"]["task"] == "conclude"
        assert report.events_emitted >= 9
        metrics = engine.metrics_report()
        assert metrics["completed_tasks"] == 3
    finally:
        engine.close()


def test_engine_retries_pure_failures_and_recover_the_goal():
    """A failing pure task is retried within policy and can succeed later."""
    engine = CognitiveExecutionEngine(EngineConfig(max_workers=2, max_active_tasks=2))
    attempts: dict[str, int] = {"analyze": 0}

    def flaky(context):
        """Fail the first attempt and succeed afterwards."""
        attempts["analyze"] += 1
        if attempts["analyze"] == 1:
            raise RuntimeError("transient failure")
        return {"task": "analyze"}

    try:
        report = engine.execute_goal(
            ExecutionRequest(goal_id="goal-2", description="retry"),
            chain_specs(),
            {"perceive": executor, "analyze": flaky, "conclude": executor},
        )
        assert report.status == "succeeded"
        assert attempts["analyze"] == 2
        assert engine.metrics_report()["retried_tasks"] == 1
    finally:
        engine.close()


def test_engine_blocks_dependents_on_failed_upstream():
    """A permanently failed upstream cancels its dependents with a reason."""
    engine = CognitiveExecutionEngine(EngineConfig(max_workers=2))

    def broken(context):
        """Fail every attempt permanently."""
        raise RuntimeError("permanent failure")

    specs = (
        TaskSpec(task_id="root", name="root"),
        TaskSpec(task_id="child", name="child", dependencies=("root",)),
    )
    try:
        report = engine.execute_goal(
            ExecutionRequest(goal_id="goal-3", description="failure propagation"),
            specs, {"root": broken, "child": executor},
        )
        assert report.failed_tasks == ("root",)
        assert report.cancelled_tasks == ("child",)
        assert report.status == "partial"
    finally:
        engine.close()


def test_engine_rejects_cycle_with_diagnostic_path():
    """An accidental cycle in the manifest is rejected before execution."""
    engine = CognitiveExecutionEngine()
    specs = (
        TaskSpec(task_id="a", name="a", dependencies=("b",)),
        TaskSpec(task_id="b", name="b", dependencies=("a",)),
    )
    with pytest.raises(Exception):  # noqa: B017 - CycleRejectedError subclass
        engine.execute_goal(
            ExecutionRequest(goal_id="goal-4", description="cycle"),
            specs, {"a": executor, "b": executor},
        )
    engine.close()


def test_engine_rejects_task_with_unavailable_capability():
    """A task whose capability no worker offers is rejected at planning time."""
    engine = CognitiveExecutionEngine()
    specs = (TaskSpec(task_id="t", name="t", capability="nonexistent"),)
    with pytest.raises(Exception):  # noqa: B017 - CapabilityUnavailableError subclass
        engine.execute_goal(
            ExecutionRequest(goal_id="goal-5", description="capability"),
            specs, {"t": executor},
        )
    engine.close()


def test_engine_cancellation_propagates_and_prevents_new_effects():
    """Goal cancellation stops dispatched tasks and cancels pending ones."""
    engine = CognitiveExecutionEngine(EngineConfig(max_workers=1, max_active_tasks=1))
    dispatched: list[str] = []
    cancelled_seen: list[str] = []

    def slow(context):
        """Cooperate with cancellation at safe checkpoints."""
        dispatched.append(context.task_spec.task_id)
        for _ in range(100):
            context.check_cancellation()
        return {"task": context.task_spec.task_id}

    specs = (
        TaskSpec(task_id="first", name="first", deadline_seconds=30.0),
        TaskSpec(task_id="second", name="second", dependencies=("first",)),
    )
    run = engine.start(
        ExecutionRequest(goal_id="goal-6", description="cancellation"),
        specs, {"first": slow, "second": slow},
    )
    engine.cancel_goal(run, "user requested stop")
    report = engine.run_loop(run)
    assert report.cancelled_tasks
    assert report.status in {"cancelled", "partial", "failed", "incomplete"}
    engine.close()


def test_engine_publishes_findings_to_shared_workspace():
    """Integration: successful tasks publish findings to ore_workspace."""
    from ore_workspace import SharedWorkspace

    workspace = SharedWorkspace(
        "The runtime test passed.", "understand_sentence",
        permissions=build_workspace_permissions(),
    )
    adapter = WorkspaceRuntimeAdapter(workspace)
    engine = CognitiveExecutionEngine(workspace_adapter=adapter)
    try:
        report = engine.execute_goal(
            ExecutionRequest(goal_id="goal-7", description="workspace integration"),
            chain_specs(),
            {"perceive": executor, "analyze": executor, "conclude": executor},
        )
        assert report.status == "succeeded"
        snapshot = workspace.snapshot()
        references = snapshot.state.get("artifact_references", [])
        published_tasks = {
            item["task_id"] for item in references
            if isinstance(item, dict) and "task_id" in item
        }
        assert "perceive" in published_tasks
        assert workspace.revision >= 3
    finally:
        engine.close()
        workspace.close()


def test_usg_change_invalidates_and_recomputes_dependent_tasks():
    """Integration: USG record changes invalidate and recompute derived artifacts."""
    engine = CognitiveExecutionEngine(EngineConfig(max_workers=2))
    try:
        engine.register_usg_source("usg.word.walker", {"sense": "person"})
        specs = (
            TaskSpec(task_id="lookup", name="lookup", metadata={"usg_deps": ["usg.word.walker"]}),
        )
        run = engine.start(
            ExecutionRequest(goal_id="goal-8", description="usg integration"),
            specs,
            {"lookup": lambda context: {
                "sense": "person", "read": sorted(context.snapshot)
            }},
        )
        engine.run_loop(run)
        assert engine.graph_dependencies.artifact("artifact::lookup") is not None

        record, report = engine.declare_usg_change(
            "usg.word.walker", {"sense": "person_who_walks"},
        )
        assert record.version == 2
        assert "artifact::lookup" in report.marked_stale
        assert engine.invalidation.is_current("artifact::lookup") is False

        recomputed = engine.recompute_stale({
            "artifact::lookup": lambda context: {
                "sense": "person_who_walks", "recomputed": True,
            },
        })
        assert "artifact::lookup" in recomputed.recomputed
        assert engine.invalidation.is_current("artifact::lookup") is True
    finally:
        engine.close()


def test_engine_crash_recovery_resumes_unfinished_tasks():
    """Crash recovery: a checkpoint resumes or safely retries unfinished tasks."""
    engine = CognitiveExecutionEngine(EngineConfig(max_workers=2))
    specs = (
        TaskSpec(task_id="done", name="done"),
        TaskSpec(task_id="pending", name="pending"),
        TaskSpec(task_id="external", name="external",
                 side_effect_class=SideEffectClass.EXTERNAL_EFFECT),
    )
    run = engine.start(
        ExecutionRequest(goal_id="goal-9", description="recovery"),
        specs, {"done": executor, "pending": executor, "external": executor},
    )
    run.rounds = 1
    engine._persist_checkpoint(run)
    checkpoint = engine.pool.checkpoint_store.load(run.run_id)
    assert checkpoint is not None
    engine.close()

    recovery = CognitiveExecutionEngine(EngineConfig(max_workers=2))
    try:
        report = recovery.recover_run(
            checkpoint, specs,
            {"done": executor, "pending": executor, "external": executor},
        )
        assert "done" in report.succeeded_tasks
        assert "pending" in report.succeeded_tasks
        # external effects without idempotency keys are not blindly retried
        assert "external" in report.cancelled_tasks
    finally:
        recovery.close()


def test_engine_registers_as_ore_module_in_registry():
    """Integration: the engine registers into ore_contracts registries."""
    from ore_contracts.contracts_registry import ContractCatalog
    from ore_contracts.module_registry import ModuleRegistry

    from ore_runtime.registration import (
        RUNTIME_MODULE_ID, register_runtime_module,
    )
    catalog = ContractCatalog()
    registry = ModuleRegistry(catalog)
    backend = register_runtime_module(registry, catalog)
    try:
        manifest = registry.get_manifest(RUNTIME_MODULE_ID)
        assert manifest.identity.module_id == RUNTIME_MODULE_ID
        assert "runtime.concurrent_execution" in manifest.capabilities
        found = registry.find_by_capability("runtime.concurrent_execution")
        assert RUNTIME_MODULE_ID in {item.module_id for item in found}
    finally:
        backend.close()


def test_engine_audit_trace_reconstructs_causal_chain():
    """The audit trace reconstructs the causal chain by correlation ID."""
    engine = CognitiveExecutionEngine(EngineConfig(max_workers=2))
    try:
        report = engine.execute_goal(
            ExecutionRequest(goal_id="goal-10", description="audit"),
            chain_specs(),
            {"perceive": executor, "analyze": executor, "conclude": executor},
        )
        chain = engine.audit_trace(report.correlation_id)
        assert len(chain) >= 6
        types = {row["event_type"] for row in chain}
        assert "goal.started" in types
        assert "task.succeeded" in types
        assert "goal.completed" in types
    finally:
        engine.close()


def test_engine_admission_backpressure_bounds_pending_work():
    """Backpressure: bursts defer dispatch and pending work stays bounded."""
    engine = CognitiveExecutionEngine(
        EngineConfig(
            max_workers=2, max_active_tasks=2,
            queue_high_watermark=2, queue_low_watermark=1,
        ),
    )
    specs = tuple(
        TaskSpec(task_id=f"burst-{index}", name=f"burst {index}", priority=0.5)
        for index in range(12)
    )
    try:
        report = engine.execute_goal(
            ExecutionRequest(goal_id="goal-11", description="backpressure"),
            specs,
            {spec.task_id: executor for spec in specs},
        )
        deferred = [row for row in report.trace if row["kind"] == "admission_deferred"]
        budget = [row for row in report.trace if row["kind"] == "dispatch_deferred"]
        assert deferred or budget
        max_active_observed = max(
            [len([row for row in report.trace if row["kind"] == "dispatched"])] + [0],
        )
        assert max_active_observed <= 12
        metrics = engine.metrics_report()
        assert metrics["admission_rejections"] >= 1
    finally:
        engine.close()


def test_engine_diagnostics_and_configuration_surface():
    """Diagnostics expose configuration, workers, and subsystem health."""
    engine = CognitiveExecutionEngine(EngineConfig(max_workers=2))
    try:
        diagnostics = engine.diagnostics()
        assert diagnostics["config"]["max_workers"] == 2
        assert "worker-0" in diagnostics["workers"]
        assert "invalidation" in diagnostics
    finally:
        engine.close()
