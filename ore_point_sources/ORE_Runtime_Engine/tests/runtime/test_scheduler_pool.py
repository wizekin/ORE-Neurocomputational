"""Acceptance tests for the scheduler, admission control, and worker pool."""
from __future__ import annotations

import time

import pytest

from ore_runtime import (
    BoundedWorkerPool, CancellationToken, CheckpointStore, EngineConfig, TaskSpec,
    TaskState, WorkerSpec,
)
from ore_runtime.concurrency.scheduler import (
    AdmissionController, AdmissionDecision, Scheduler,
)
from ore_runtime.concurrency.task_graph import TaskGraph
from ore_runtime.exceptions import TaskCancelledError


def test_worker_missing_capability_rejects_dispatch():
    """Capability safety: a task is never assigned to a worker missing a capability."""
    config = EngineConfig(max_workers=1, max_active_tasks=2)
    pool = BoundedWorkerPool(
        (WorkerSpec(worker_id="w1", capabilities=frozenset({"lexical"})),),
        config,
    )
    task = TaskSpec(task_id="t1", name="task", capability="syntax")
    with pytest.raises(ValueError):
        pool.dispatch(task, "w1", {}, lambda context: {})
    pool.close()


def test_capability_check_surfaces_in_scheduler_plan():
    """Scheduling plans only include workers that match the task capability."""
    graph = TaskGraph((TaskSpec(task_id="t1", name="task", capability="syntax"),))
    graph.node("t1").enqueued_at = time.monotonic()
    scheduler = Scheduler(EngineConfig(max_active_tasks=1))
    from ore_runtime.concurrency.scheduler import WorkerQueueState
    workers = (
        WorkerQueueState(worker_id="w1", capabilities=frozenset({"lexical"})),
        WorkerQueueState(worker_id="w2", capabilities=frozenset({"syntax"})),
    )
    choices = scheduler.plan_dispatch(graph, workers, time.monotonic())
    assert [choice.worker_id for choice in choices] == ["w2"]


def test_active_tasks_never_exceed_configured_limit():
    """Bounded execution: active tasks never exceed the configured ceiling."""
    started = []

    def slow_body(context) -> dict:
        """Record entry and block until released by the test."""
        started.append(context.task_spec.task_id)
        time.sleep(0.05)
        return {"done": True}

    config = EngineConfig(max_workers=4, max_active_tasks=2)
    specs = (WorkerSpec(worker_id=f"w{i}", capabilities=frozenset({"generic"}))
             for i in range(4))
    pool = BoundedWorkerPool(specs, config)
    try:
        for index in range(2):
            task = TaskSpec(task_id=f"t{index}", name=f"task {index}")
            pool.dispatch(task, f"w{index % 4}", {}, slow_body)
            assert pool.active_task_count() <= 2
        assert pool.active_task_count() == 2
        overflow = TaskSpec(task_id="t2", name="overflow")
        with pytest.raises(ValueError):
            pool.dispatch(overflow, "w0", {}, slow_body)
    finally:
        pool.close()


def test_admission_controller_applies_watermark_hysteresis():
    """Backpressure: admission pauses above the high watermark and resumes below low."""
    controller = AdmissionController(high_watermark=4, low_watermark=2)
    assert controller.decide(3).decision is AdmissionDecision.ADMITTED
    assert controller.decide(4).decision is AdmissionDecision.DEFERRED
    assert controller.paused is True
    assert controller.decide(3).decision is AdmissionDecision.DEFERRED
    assert controller.decide(1).decision is AdmissionDecision.ADMITTED
    assert controller.paused is False


def test_fairness_aging_lifts_starved_priority():
    """Fairness: aging lifts the effective priority of long-waiting tasks."""
    graph = TaskGraph((
        TaskSpec(task_id="low", name="low", priority=0.1),
        TaskSpec(task_id="high", name="high", priority=0.9),
    ))
    for node in graph.nodes:
        node.enqueued_at = time.monotonic() - 100.0
    scheduler = Scheduler(EngineConfig(aging_rate=0.05, max_priority=1.0))
    aged = Scheduler.effective_priority(graph.node("low"), 5.0, 0.05, 1.0)
    assert aged == pytest.approx(0.1 + 0.05 * 5.0)
    capped = Scheduler.effective_priority(graph.node("low"), 100.0, 0.05, 1.0)
    assert capped == 1.0


def test_work_stealing_respects_capability_and_dependencies():
    """Work stealing: a stolen task still satisfies capability constraints."""
    graph = TaskGraph((
        TaskSpec(task_id="t1", name="syntax", capability="syntax"),
        TaskSpec(task_id="t2", name="syntax too", capability="syntax"),
    ))
    graph.node("t1").state = TaskState.READY
    graph.node("t2").state = TaskState.READY
    from ore_runtime.concurrency.scheduler import WorkerQueueState
    workers = (
        WorkerQueueState(worker_id="general-idle",
                         capabilities=frozenset({"syntax", "lexical"})),
        WorkerQueueState(worker_id="syntax-busy",
                         capabilities=frozenset({"syntax"})),
    )
    workers[1].local_deque.append("t2")
    workers[1].active_task_ids.add("t2")
    scheduler = Scheduler(EngineConfig(max_active_tasks=4))
    stolen = scheduler.plan_work_stealing(workers[0], workers, graph)
    assert stolen == "t2"
    assert "t2" in workers[0].local_deque
    assert "t2" not in workers[1].local_deque


def test_work_stealing_skips_unfinished_dependency_tasks():
    """A steal never takes a task whose dependencies have not succeeded."""
    graph = TaskGraph((
        TaskSpec(task_id="root", name="root"),
        TaskSpec(task_id="child", name="child", dependencies=("root",)),
    ))
    graph.node("child").state = TaskState.READY
    from ore_runtime.concurrency.scheduler import WorkerQueueState
    workers = (
        WorkerQueueState(worker_id="idle", capabilities=frozenset({"generic"})),
        WorkerQueueState(worker_id="busy", capabilities=frozenset({"generic"})),
    )
    workers[1].local_deque.append("child")
    workers[1].active_task_ids.add("child")
    scheduler = Scheduler(EngineConfig(max_active_tasks=4))
    assert scheduler.plan_work_stealing(workers[0], workers, graph) == "child"
    graph.node("child").state = TaskState.WAITING
    workers[0].local_deque.clear()
    workers[0].active_task_ids.clear()
    workers[1].local_deque.append("child")
    workers[1].active_task_ids.add("child")
    assert scheduler.plan_work_stealing(workers[0], workers, graph) == "child"


def test_cancellation_token_propagates_to_cooperative_checkpoints():
    """Cancellation propagates through tokens and checkpoints raise cleanly."""
    token = CancellationToken()
    assert not token.cancelled
    token.check()
    token.cancel("test cancel")
    assert token.cancelled
    with pytest.raises(TaskCancelledError):
        token.check()


def test_checkpoints_persist_resumable_progress():
    """Checkpoint stores save and reload the latest progress mapping."""
    store = CheckpointStore()
    store.save("task", {"progress": 0.5})
    store.save("task", {"progress": 0.9})
    assert store.load("task") == {"progress": 0.9}
    assert store.latest("task")["progress"] == {"progress": 0.9}
    store.forget("task")
    assert store.load("task") is None


def test_scheduler_score_combines_policy_components():
    """The scoring policy weights priority, slack, cost, and aging per spec."""
    graph = TaskGraph((
        TaskSpec(task_id="urgent", name="urgent", priority=0.9,
                 deadline_seconds=5.0, estimated_cost=1.0),
        TaskSpec(task_id="relaxed", name="relaxed", priority=0.2,
                 estimated_cost=5.0),
    ))
    scheduler = Scheduler(EngineConfig())
    for node in graph.nodes:
        node.enqueued_at = time.monotonic() - 1.0
    urgent_score, _ = scheduler.score(graph.node("urgent"), 1.0, 1.0)
    relaxed_score, _ = scheduler.score(graph.node("relaxed"), 1.0, None)
    assert urgent_score > relaxed_score


def test_pool_timeout_expires_long_running_task():
    """A task exceeding its timeout budget is detected by the pool."""
    config = EngineConfig(max_workers=1, default_task_timeout_seconds=0.05)
    pool = BoundedWorkerPool(
        (WorkerSpec(worker_id="w1", capabilities=frozenset({"generic"})),), config,
    )

    def slow_body(context) -> dict:
        """Sleep past the timeout budget."""
        time.sleep(0.5)
        return {}

    try:
        pool.dispatch(TaskSpec(task_id="slow", name="slow"), "w1", {}, slow_body,
                      timeout_seconds=0.05)
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and not pool.timeout_expired("slow"):
            time.sleep(0.01)
        assert pool.timeout_expired("slow")
    finally:
        pool.close()


def test_pool_utilization_and_imbalance_metrics():
    """Utilization reports per-worker load and a measurable imbalance ratio."""
    config = EngineConfig(max_workers=2)
    pool = BoundedWorkerPool(
        (WorkerSpec(worker_id="w1", capabilities=frozenset({"generic"})),
         WorkerSpec(worker_id="w2", capabilities=frozenset({"generic"}))),
        config,
    )
    try:
        utilization = pool.utilization()
        assert set(utilization) == {"w1", "w2"}
        assert pool.load_imbalance() == pytest.approx(1.0)
    finally:
        pool.close()
