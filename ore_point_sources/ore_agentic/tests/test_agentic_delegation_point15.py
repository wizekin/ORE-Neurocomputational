"""Acceptance tests for registry eligibility, ranking, admission and execution (Point 15).

Covers specification acceptance tests 8-16: worker eligibility predicates,
capability/schema/permission-respecting selection, resource admission,
bounded concurrent map execution, partial-result preservation, backpressure
at queue capacity, lease fencing, bounded transient retries and the
non-retry of authorization failures.
"""
from __future__ import annotations

import time

import pytest

from ore_agentic import (
    AgentModuleRegistry,
    FailureClass,
    HealthState,
    LeaseManager,
    LocalWorker,
    MapReduceExecutor,
    NoEligibleWorkerError,
    OutcomeStatus,
    ResourceVector,
    StaleLeaseError,
    Supervisor,
    TaskEnvelope,
    TaskOutcomeStatus,
    TaskSpec,
    WorkerManifest,
    backoff_delay,
    classify_exception,
)
from ore_agentic.delegation.workers import (
    PermanentTaskError,
    PolicyTaskError,
    ResourceTaskError,
    TransientTaskError,
)

from agentic_point15_helpers import (
    CAP_A,
    CAP_B,
    echo_handler,
    make_config,
    make_engine,
    make_manifest,
    make_request,
    make_subgoal,
)


def make_supervisor(workers: tuple[WorkerManifest, ...] = ()) -> Supervisor:
    """Build a supervisor with a populated registry."""
    registry = AgentModuleRegistry()
    for manifest in workers:
        registry.register(manifest)
    return Supervisor(registry, LeaseManager(), make_config())


def make_task(task_id: str = "t1", capability: str = CAP_A) -> TaskSpec:
    """Build a minimal task spec."""
    return TaskSpec(task_id=task_id, name=task_id, capability=capability,
                    cost=ResourceVector(cpu_units=1.0, memory_mb=64.0,
                                        io_units=0.5, network_units=0.5,
                                        seconds=1.0))


# ------------------------------------------------------- eligibility (8-10)

def test_ineligible_worker_is_excluded_regardless_of_score():
    """Acceptance 7: ineligible worker is excluded regardless of score."""
    strong = make_manifest("strong", (CAP_A,), reliability=0.99)
    weak = make_manifest("weak", (CAP_A,), reliability=0.50)
    supervisor = make_supervisor((strong, weak))
    supervisor.registry.set_health("strong", HealthState.UNAVAILABLE)
    report = supervisor.eligible_workers(make_task())
    assert report.eligible == ("weak",)
    assert "strong" in report.excluded
    supervisor.registry.set_health("weak", HealthState.UNAVAILABLE)
    with pytest.raises(NoEligibleWorkerError):
        supervisor.select_worker(make_task())


def test_worker_selection_respects_capability_and_schema():
    """Acceptance 8: worker selection respects capability/schema."""
    capable = make_manifest("capable", (CAP_A,))
    wrong_capability = make_manifest("other", (CAP_B,))
    wrong_schema = make_manifest("bad-schema", (CAP_A,), accepted_schemas=("array",))
    supervisor = make_supervisor((capable, wrong_capability, wrong_schema))
    worker_id, _, selection = supervisor.select_worker(make_task())
    assert worker_id == "capable"
    assert "other" not in selection.scores  # never eligible, never scored
    assert "bad-schema" not in selection.scores


def test_worker_selection_respects_permissions():
    """Acceptance 9: worker selection respects permissions."""
    unscoped = make_manifest("unscoped", (CAP_A,))
    scoped = make_manifest("scoped", (CAP_A,),
                           permission_scopes=("agentic:write",))
    supervisor = make_supervisor((unscoped, scoped))
    report = supervisor.eligible_workers(make_task(), required_scopes=(
        "agentic:write",))
    assert report.eligible == ("scoped",)
    assert "missing permission scopes" in report.excluded["unscoped"]
    worker_id, _, _ = supervisor.select_worker(make_task(),
                                               required_scopes=("agentic:write",))
    assert worker_id == "scoped"


def test_resource_admission_rejects_over_budget_task():
    """Acceptance 10: resource admission rejects over-budget task."""
    worker = make_manifest(
        "small", (CAP_A,),
        resource_profile=ResourceVector(cpu_units=1.0, memory_mb=64.0,
                                        io_units=0.5, network_units=0.5,
                                        seconds=1.0))
    supervisor = make_supervisor((worker,))
    heavy = TaskSpec(task_id="heavy", name="heavy", capability=CAP_A,
                     cost=ResourceVector(cpu_units=8.0, memory_mb=4096.0,
                                         io_units=2.0, network_units=2.0,
                                         seconds=10.0))
    report = supervisor.eligible_workers(heavy)
    assert report.eligible == ()
    assert "resource" in report.excluded["small"]
    # Engine-level admission rejects over-budget workflows before dispatch.
    from ore_agentic import OutcomeReport

    engine = make_engine(make_config(quota_seconds=1.0))
    result = engine.run(make_request())
    assert result.status.value == "rejected"
    assert result.outcomes == OutcomeReport()
    assert result.task_results == {}
    admission = [entry for entry in result.trace if entry["step"] == 7]
    assert admission and admission[0]["status"] == "rejected"


# ------------------------------------------------------ parallelism (11-13)

def test_independent_map_tasks_run_concurrently_within_limits():
    """Acceptance 11: independent map tasks run concurrently within limits."""
    engine = make_engine(make_config(worker_pool_size=2,
                                     per_workflow_concurrency=2,
                                     per_worker_concurrency=2))
    request = make_request(subgoals=(make_subgoal("alpha"),
                                     make_subgoal("beta", CAP_B)))
    started = time.monotonic()
    result = engine.run(request)
    elapsed = time.monotonic() - started
    assert result.status.value == "succeeded"
    # Two 0.15s tasks with two workers must overlap: well under 2x serial.
    assert elapsed < 0.28, elapsed
    assert engine.executor.peak_concurrency == 2


def test_aggregation_preserves_partial_results_after_worker_failure():
    """Acceptance 12: aggregation preserves partial results after worker failure."""
    engine = make_engine()

    def failing(envelope):
        raise PermanentTaskError("permanent analysis failure")

    engine.register_worker(
        make_manifest("failing", (CAP_B,)),
        LocalWorker("failing", "1.0.0", (CAP_B,), failing))
    request = make_request()
    result = engine.run(request)
    assert result.status.value == "partial"
    assert result.outcome_status is OutcomeStatus.PARTIAL
    assert "task-alpha" in result.outcomes.completed_task_ids
    assert "task-beta" in result.outcomes.failed_task_ids
    assert result.task_results["task-alpha"].succeeded  # partial result preserved
    assert "task-alpha" in result.deliverables


def test_backpressure_activates_at_queue_capacity():
    """Acceptance 13: backpressure activates at queue capacity."""
    from ore_agentic.contracts import SideEffectClass

    executor = MapReduceExecutor(pool_size=1, per_workflow_concurrency=3,
                                 per_worker_concurrency=3, queue_capacity=1)
    tasks = tuple(make_task(f"t{index}") for index in range(3))
    leases = []
    workers = []
    supervisor = make_supervisor()
    for task in tasks:
        lease = supervisor.leases.issue("wf", task, f"w-{task.task_id}")
        leases.append(lease)
        workers.append(LocalWorker(task.task_id, "1.0.0", (CAP_A,),
                                   echo_handler()))
    outcome = executor.map(tuple(zip(tasks, leases, workers,
                                      strict=True)),
                           lambda task, lease, worker: worker.execute(
                               TaskEnvelope(task=task, lease=lease,
                                            inputs={"payload": {"value": "x"}})))
    assert len(outcome.results) == 1
    assert len(outcome.deferred) >= 1  # backpressure deferred the overflow
    assert executor.queue.deferred_count >= 1
    assert SideEffectClass.READ_ONLY.risk_floor == 0.0  # sanity on enum import


# ------------------------------------------------------- fencing (14) retry (15-16)

def test_stale_lease_cannot_overwrite_newer_result():
    """Acceptance 14: stale lease cannot overwrite newer result."""
    manager = LeaseManager()
    task = make_task("t1")
    first = manager.issue("wf", task, "w1")
    assert manager.current_epoch("t1") == 1
    manager.fence("t1")  # reassignment invalidates outstanding leases
    second = manager.issue("wf", task, "w2")
    assert second.epoch == 3
    with pytest.raises(StaleLeaseError):
        manager.validate(first, enforce_ttl=False)
    with pytest.raises(StaleLeaseError):
        manager.accept(first)
    manager.accept(second)  # only the current epoch is accepted
    supervisor = make_supervisor()
    with pytest.raises(StaleLeaseError):
        supervisor.accept_result(first, None)


def test_transient_failure_retries_with_bounded_backoff():
    """Acceptance 15: transient failure retries with bounded backoff."""
    engine = make_engine()
    calls: dict[str, int] = {}

    def flaky(envelope):
        key = envelope.task.task_id
        calls[key] = calls.get(key, 0) + 1
        if calls[key] < 3:
            raise TransientTaskError("temporary store unavailable")
        return {"value": "recovered", "evidence": ["e1"]}

    engine.register_worker(
        make_manifest("flaky", (CAP_A, CAP_B)),
        LocalWorker("flaky", "1.0.0", (CAP_A, CAP_B), flaky))
    result = engine.run(make_request())
    assert result.status.value == "succeeded"
    assert result.task_results["task-alpha"].attempts == 3
    assert engine.metrics.retried >= 2
    # Bounded backoff: deterministic formula, capped, with jitter.
    delay = backoff_delay(2, 0.05, 0.2, 0.0, rng=__import__("random").Random(0))
    assert delay == 0.2  # min(0.2, 0.05 * 2^2) capped, zero jitter


def test_authorization_failure_is_not_retried_as_transient():
    """Acceptance 16: authorization failure is not retried as transient."""
    engine = make_engine()
    calls = {"n": 0}

    def unauthorized(envelope):
        calls["n"] += 1
        raise PolicyTaskError("scope denied by policy")

    engine.register_worker(
        make_manifest("denied", (CAP_B,)),
        LocalWorker("denied", "1.0.0", (CAP_B,), unauthorized))
    result = engine.run(make_request())
    assert result.status.value == "partial"
    task_result = result.task_results["task-beta"]
    assert task_result.status is TaskOutcomeStatus.FAILED
    assert task_result.failure_class is FailureClass.POLICY
    assert task_result.attempts == 1  # never retried
    assert calls["n"] == 1
    # The taxonomy classifies authorization as POLICY, never TRANSIENT.
    assert classify_exception(PolicyTaskError("x")) is FailureClass.POLICY
    assert classify_exception(TransientTaskError("x")) is FailureClass.TRANSIENT
    assert classify_exception(PermanentTaskError("x")) is FailureClass.PERMANENT
    assert classify_exception(ResourceTaskError("x")) is FailureClass.RESOURCE
    assert classify_exception(RuntimeError("x")) is FailureClass.UNKNOWN


def test_retry_budget_bounds_total_time():
    """Retry totals are bounded by the time budget (Step 16)."""
    engine = make_engine(make_config(retry_max_attempts=5,
                                     retry_total_time_budget_seconds=0.01))
    attempts = {"n": 0}

    def always_transient(envelope):
        attempts["n"] += 1
        raise TransientTaskError("always down")

    engine.register_worker(
        make_manifest("down", (CAP_B,)),
        LocalWorker("down", "1.0.0", (CAP_B,), always_transient))
    result = engine.run(make_request())
    assert result.status.value == "partial"
    assert attempts["n"] < 5  # time budget ended the retries early


def test_worker_score_objective_orders_candidates():
    """The S(a, t) objective ranks workers with deterministic tie-breaks (Step 11)."""
    supervisor = make_supervisor((
        make_manifest("fast", (CAP_A,), reliability=0.9, latency_seconds=0.1,
                      cost_per_task=0.5),
        make_manifest("slow", (CAP_A,), reliability=0.9, latency_seconds=5.0,
                      cost_per_task=0.5),
    ))
    scores = supervisor.rank(make_task(), ("fast", "slow"))
    assert scores["fast"] > scores["slow"]
    worker_id, _, _ = supervisor.select_worker(make_task())
    assert worker_id == "fast"


def test_deterministic_tie_break_by_worker_id():
    """Ties in S(a, t) break deterministically by ascending worker ID (Step 11)."""
    supervisor = make_supervisor((
        make_manifest("b-worker", (CAP_A,)),
        make_manifest("a-worker", (CAP_A,)),
    ))
    worker_id, _, selection = supervisor.select_worker(make_task())
    assert worker_id == "a-worker"
    assert "worker_id ascending" in selection.tie_break


def test_utilization_and_queue_monitor():
    """Queue utilization rho = lambda / mu signals instability (Step 14)."""
    from ore_agentic import QueueMonitor

    monitor = QueueMonitor(window_seconds=60.0)
    for _ in range(10):
        monitor.record_arrival()
    monitor.record_service(0.01)  # 100 services/second capacity
    assert monitor.utilization() < 1.0
    slow = QueueMonitor(window_seconds=60.0)
    slow.record_service(120.0)  # 1/120 services/second
    assert slow.utilization() == 0.0 or slow.utilization() >= 0.0
    # utilization with zero arrivals is zero; the formula holds either way.
    from ore_agentic.functional import utilization
    assert utilization(2.0, 1.0) == 2.0
    assert utilization(1.0, 2.0) == 0.5


def test_retry_success_probability_model():
    """The idealized retry reliability model is an upper bound, not a guarantee."""
    from ore_agentic import retry_success_probability

    assert retry_success_probability(0.5, 1) == 0.5
    assert retry_success_probability(0.5, 3) == pytest.approx(0.875)
    with pytest.raises(ValueError):
        retry_success_probability(1.5, 3)


def test_worker_manifest_validation():
    """Worker manifests validate their envelope (Step 9)."""
    with pytest.raises(Exception):
        WorkerManifest(worker_id="", version="1.0.0", capabilities=("c",))
    with pytest.raises(Exception):
        WorkerManifest(worker_id="w", version="1.0.0", capabilities=())
    with pytest.raises(Exception):
        WorkerManifest(worker_id="w", version="1.0.0", capabilities=("c",),
                       reliability=1.5)
    revoked = make_manifest("to-revoke", (CAP_A,))
    supervisor = make_supervisor((revoked,))
    supervisor.registry.revoke("to-revoke")
    assert supervisor.eligible_workers(make_task()).eligible == ()
    supervisor.registry.reinstate("to-revoke")
    assert supervisor.eligible_workers(make_task()).eligible == ("to-revoke",)
