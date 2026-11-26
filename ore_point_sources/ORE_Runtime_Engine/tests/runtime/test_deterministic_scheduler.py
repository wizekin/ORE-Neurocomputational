"""Acceptance tests for deterministic replay, invariants, and fuzzing."""
from __future__ import annotations

import pytest

from ore_runtime import (
    DeterministicScheduler, EngineConfig, ExecutionRequest, TaskSpec,
)
from ore_runtime.engine import CognitiveExecutionEngine
from ore_runtime.exceptions import InvariantViolationError
from ore_runtime.testing.deterministic_scheduler import (
    ScheduleStep, default_invariants,
)


def manual_engine(**overrides) -> CognitiveExecutionEngine:
    """Build one manual-mode engine with two generic workers."""
    settings = {
        "max_workers": 2, "max_active_tasks": 2, "execution_mode": "manual",
        "queue_high_watermark": 32, "queue_low_watermark": 16,
        "max_ready_queue_size": 64,
    }
    settings.update(overrides)
    return CognitiveExecutionEngine(EngineConfig(**settings))


def demo_specs() -> tuple[TaskSpec, ...]:
    """Build a three-task dependency chain for deterministic tests."""
    return (
        TaskSpec(task_id="a", name="first", priority=0.8),
        TaskSpec(task_id="b", name="second", dependencies=("a",), priority=0.6),
        TaskSpec(task_id="c", name="third", dependencies=("b",), priority=0.4),
    )


def executor(context):
    """Produce a deterministic output derived from the snapshot."""
    return {"task": context.task_spec.task_id}


def test_deterministic_replay_reproduces_invariant_outcomes():
    """Deterministic replay: a captured schedule reproduces invariant outcomes."""
    engine = manual_engine()
    scheduler = DeterministicScheduler(engine, seed=7)
    run = engine.start(
        ExecutionRequest(goal_id="g1", description="deterministic replay",
                        expected_outputs=("c",)),
        demo_specs(), {"a": executor, "b": executor, "c": executor},
    )
    schedule = scheduler.natural_schedule(run)
    original_status, report = scheduler.run(run, schedule)
    assert original_status == "succeeded"
    manifest = scheduler.capture_replay_manifest(run)
    assert manifest.fingerprint()

    replay_engine = manual_engine()
    replay_scheduler = DeterministicScheduler(replay_engine, seed=7)
    replay_run = replay_engine.start(
        ExecutionRequest(goal_id="g1", description="deterministic replay",
                        expected_outputs=("c",)),
        demo_specs(), {"a": executor, "b": executor, "c": executor},
    )
    replayed_status, _ = replay_scheduler.run(replay_run, schedule)
    assert replay_scheduler.assert_replay_deterministic(
        original_status, replayed_status,
    ) is True
    engine.close()
    replay_engine.close()


def test_invariants_hold_under_natural_schedule():
    """Invariant checks hold over the standard schedule."""
    engine = manual_engine()
    scheduler = DeterministicScheduler(engine, seed=1)
    run = engine.start(
        ExecutionRequest(goal_id="g2", description="invariants"),
        demo_specs(), {"a": executor, "b": executor, "c": executor},
    )
    scheduler.run(run, scheduler.natural_schedule(run))
    scheduler.assert_invariants(run, default_invariants(engine.config))
    engine.close()


def test_fuzzing_injects_faults_without_violating_invariants():
    """Fuzzing: generated adversarial schedules keep task-state invariants."""
    engine = manual_engine(max_rounds=64)
    scheduler = DeterministicScheduler(engine, seed=11)

    def run_factory():
        """Create one fresh prepared run for each fuzz case."""
        return engine.start(
            ExecutionRequest(goal_id=f"g-fuzz", description="fuzz"),
            demo_specs(), {"a": executor, "b": executor, "c": executor},
        )

    summary = scheduler.fuzz(
        run_factory, invariants=default_invariants(engine.config),
        cases=8, seed=3,
    )
    assert summary["cases"] == 8
    assert summary["invariant_violations"] == 0
    engine.close()


def test_invariant_violation_is_reported_with_evidence():
    """Broken invariants raise InvariantViolationError naming the check."""
    engine = manual_engine()
    scheduler = DeterministicScheduler(engine, seed=2)
    run = engine.start(
        ExecutionRequest(goal_id="g3", description="violation"),
        demo_specs(), {"a": executor, "b": executor, "c": executor},
    )
    scheduler.run(run, scheduler.natural_schedule(run))
    from ore_runtime.testing.deterministic_scheduler import Invariant
    broken = Invariant(
        name="always_false",
        check=lambda engine_ref, run_ref: False,
    )
    with pytest.raises(InvariantViolationError) as excinfo:
        scheduler.assert_invariants(run, (broken,))
    assert excinfo.value.invariant == "always_false"
    engine.close()


def test_random_schedule_generates_adversarial_fault_steps():
    """Random schedules include injected failure, timeout, and cancellation faults."""
    engine = manual_engine()
    scheduler = DeterministicScheduler(engine, seed=5)
    run = engine.start(
        ExecutionRequest(goal_id="g4", description="adversarial"),
        demo_specs(), {"a": executor, "b": executor, "c": executor},
    )
    schedule = scheduler.random_schedule(run, seed=9, faults=3)
    injected = [step for step in schedule if step.kind == "inject"]
    assert len(injected) == 3
    assert all(step.outcome in {"failed", "timed_out", "cancelled"} for step in injected)
    engine.close()


def test_schedule_survives_duplicate_event_redelivery():
    """Duplicate delivery under manual stepping applies effects only once."""
    engine = manual_engine()
    scheduler = DeterministicScheduler(engine, seed=4)
    run = engine.start(
        ExecutionRequest(goal_id="g5", description="duplicate events"),
        demo_specs(), {"a": executor, "b": executor, "c": executor},
    )
    scheduler.run(run, scheduler.natural_schedule(run))
    delivered = [event for event in engine.event_store.events()]
    duplicated = 0
    for stored in delivered:
        outcomes = engine.bus.publish(stored.envelope)
        duplicated += sum(
            1 for outcome in outcomes if outcome.status.value == "already_applied"
        )
    assert duplicated > 0
    assert all(node.state.value == "succeeded" for node in run.graph.nodes)
    engine.close()
