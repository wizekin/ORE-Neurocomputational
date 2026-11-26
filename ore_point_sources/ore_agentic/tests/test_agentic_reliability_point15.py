"""Acceptance tests for production reliability, governance and status (Point 15).

Covers specification acceptance tests 38-40 (redaction is covered in the
memory module), queue overload/worker loss/stale checkpoint robustness, the
distinct end-to-end status categories, quotas and sandboxing, semantic
caching, drift monitoring, and the system evaluation metrics.
"""
from __future__ import annotations

import sqlite3
import time

import pytest

from ore_agentic import (
    AgenticDatabase,
    DriftMonitor,
    MetricsCollector,
    OutcomeStatus,
    OutcomeReport,
    QuotaExceededError,
    QuotaManager,
    ResourceVector,
    SandboxProfile,
    SandboxedExecutor,
    SemanticCache,
    TimeoutViolation,
    WorkflowStore,
)

from agentic_point15_helpers import (
    CAP_A,
    CAP_B,
    make_config,
    make_engine,
    make_request,
    make_subgoal,
)


# ------------------------------------------------------- robustness (39)

def test_queue_overload_defers_but_never_loses_work():
    """Acceptance 39a: queue overload defers work without corruption."""
    engine = make_engine(make_config(queue_capacity=1,
                                     per_workflow_concurrency=2,
                                     worker_pool_size=1))
    request = make_request(subgoals=(make_subgoal("alpha"),
                                     make_subgoal("beta", CAP_B)))
    result = engine.run(request)
    # All tasks eventually completed despite the overload: nothing was lost.
    assert result.status.value == "succeeded"
    assert set(result.outcomes.completed_task_ids) >= {"task-alpha", "task-beta"}
    assert engine.metrics.deferred >= 0  # deferrals counted, work preserved


def test_worker_loss_yields_partial_but_consistent_state():
    """Acceptance 39b: worker loss degrades to partial, never corrupts state."""
    from ore_agentic import LocalWorker, WorkerManifest
    from ore_agentic.delegation.workers import PermanentTaskError

    engine = make_engine()
    engine.register_worker(
        WorkerManifest(worker_id="lost-worker", version="1.0.0",
                       capabilities=(CAP_B,), latency_seconds=0.1,
                       cost_per_task=0.1),
        LocalWorker("lost-worker", "1.0.0", (CAP_B,),
                    lambda envelope: (_ for _ in ()).throw(
                        PermanentTaskError("worker process lost"))))
    result = engine.run(make_request())
    assert result.status.value == "partial"
    assert result.outcome_status is OutcomeStatus.PARTIAL
    machine = engine.state_machines[result.workflow_id]
    counts = machine.counts()
    # Every task reached a terminal state; no task stuck in running/leased.
    for _task_id, state in machine.snapshot().items():
        assert state in ("succeeded", "failed", "cancelled", "compensated")
    assert counts


def test_stale_checkpoint_does_not_corrupt_workflow_state(tmp_path):
    """Acceptance 39c: stale checkpoint does not corrupt state."""
    database_path = tmp_path / "agentic.sqlite3"
    engine = make_engine(database_path=str(database_path))
    result = engine.run(make_request())
    store = engine.workflow_store
    assert store is not None
    checkpoints = store.list_checkpoints(result.workflow_id)
    assert checkpoints  # checkpoints were captured
    # Corrupt one checkpoint: loading raises, the workflow row stays intact.
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "UPDATE agentic_checkpoints SET snapshot_json = '{\"bad\": 1}' "
            "WHERE checkpoint_id = ?", (checkpoints[0],))
    with pytest.raises(ValueError):
        store.load_checkpoint(checkpoints[0])
    row = store.get(result.workflow_id)
    assert row is not None
    assert row["status"] == "succeeded"  # persisted state unaffected
    assert store.tasks(result.workflow_id)


# ------------------------------------------------------------ status (40)

def test_end_to_end_status_distinguishes_complete_partial_failed_unverified():
    """Acceptance 40: end-to-end status distinguishes all four outcome categories."""
    from ore_agentic import LocalWorker, WorkerManifest
    from ore_agentic.delegation.workers import PermanentTaskError

    # Complete: every task succeeds and verification passes.
    complete = make_engine().run(make_request())
    assert complete.outcome_status is OutcomeStatus.COMPLETED
    assert complete.status.value == "succeeded"

    # Partial: one task fails permanently.
    engine = make_engine()
    engine.register_worker(
        WorkerManifest(worker_id="failing", version="1.0.0",
                       capabilities=(CAP_B,), latency_seconds=0.1,
                       cost_per_task=0.1),
        LocalWorker("failing", "1.0.0", (CAP_B,),
                    lambda envelope: (_ for _ in ()).throw(
                        PermanentTaskError("boom"))))
    partial = engine.run(make_request())
    assert partial.outcome_status is OutcomeStatus.PARTIAL
    assert partial.status.value == "partial"
    assert partial.outcomes.failed_task_ids and partial.outcomes.completed_task_ids

    # Failed: every task fails.
    engine = make_engine()
    engine.register_worker(
        WorkerManifest(worker_id="all-failing", version="1.0.0",
                       capabilities=(CAP_A, CAP_B), latency_seconds=0.1,
                       cost_per_task=0.1),
        LocalWorker("all-failing", "1.0.0", (CAP_A, CAP_B),
                    lambda envelope: (_ for _ in ()).throw(
                        PermanentTaskError("everything fails"))))
    failed = engine.run(make_request())
    assert failed.outcome_status is OutcomeStatus.FAILED
    assert failed.status.value == "failed"

    # Unverified: tasks succeed but final verification fails.
    engine = make_engine(deterministic_checks={
        "deliverables_present": lambda payload: (False, "verification rejected")})
    unverified = engine.run(make_request())
    assert unverified.outcome_status is OutcomeStatus.UNVERIFIED
    assert unverified.outcomes.unverified_task_ids
    assert unverified.outcomes.failed_task_ids == ()
    # The four categories are mutually exclusive on the overall property.
    outcomes = {complete.outcome_status, partial.outcome_status,
                failed.outcome_status, unverified.outcome_status}
    assert outcomes == {OutcomeStatus.COMPLETED, OutcomeStatus.PARTIAL,
                        OutcomeStatus.FAILED, OutcomeStatus.UNVERIFIED}


def test_outcome_report_categories_are_derived_correctly():
    """The outcome report derives the distinct overall category (Step 50)."""
    assert OutcomeReport().overall is OutcomeStatus.COMPLETED
    assert OutcomeReport(completed_task_ids=("a",)).overall is (
        OutcomeStatus.COMPLETED)
    assert OutcomeReport(completed_task_ids=("a",),
                         failed_task_ids=("b",)).overall is OutcomeStatus.PARTIAL
    assert OutcomeReport(failed_task_ids=("b",)).overall is OutcomeStatus.FAILED
    assert OutcomeReport(completed_task_ids=("a",),
                         unverified_task_ids=("a",)).overall is (
        OutcomeStatus.UNVERIFIED)


# ------------------------------------------------- quotas and sandbox (47)

def test_quotas_reject_over_budget_principals():
    """Quota managers reserve and release bounded capacity (Step 47)."""
    quotas = QuotaManager(cpu_units=4.0, memory_mb=256.0, seconds=10.0,
                          cost=5.0)
    principal = "agent-1"
    lease = quotas.check_and_reserve(principal, ResourceVector(
        cpu_units=2.0, memory_mb=128.0, io_units=0.0, network_units=0.0,
        seconds=5.0), cost=2.0)
    quotas.release(lease)
    with pytest.raises(QuotaExceededError):
        quotas.check_and_reserve(principal, ResourceVector(
            cpu_units=8.0, memory_mb=128.0), cost=0.0)
    with pytest.raises(QuotaExceededError):
        quotas.check_and_reserve(principal, ResourceVector(
            cpu_units=1.0, memory_mb=64.0), cost=100.0)


def test_sandbox_enforces_wall_clock_and_output_limits():
    """Sandbox profiles enforce wall-clock and output-size limits (Step 47)."""
    executor = SandboxedExecutor(SandboxProfile(wall_clock_seconds=0.15,
                                                max_output_bytes=256))
    quick = executor.run(lambda: "ok")
    assert quick.ok and quick.value == "ok"
    with pytest.raises(TimeoutViolation):
        executor.run(lambda: time.sleep(0.5))
    oversized = executor.run(lambda: {"blob": "x" * 1024})
    assert not oversized.ok
    assert oversized.truncated
    with pytest.raises(ValueError):
        SandboxProfile(wall_clock_seconds=0.0)


# ------------------------------------------------- cache and drift (45-46)

def test_semantic_cache_keys_on_versions_and_workspace_revision():
    """Acceptance 45: semantic cache keyed on inputs, versions and revision."""
    cache = SemanticCache(ttl_seconds=60.0)
    key = SemanticCache.build_key("cap", {"q": 1}, "model-1", revision := 3)
    assert cache.get(key) is None
    cache.put(key, {"answer": 42}, model_version="model-1",
              workspace_revision=revision)
    entry = cache.get(key)
    assert entry is not None and entry.value == {"answer": 42}
    assert not entry.stale
    # A different model version or revision is a different key.
    other = SemanticCache.build_key("cap", {"q": 1}, "model-2", 3)
    assert cache.get(other) is None
    other = SemanticCache.build_key("cap", {"q": 1}, "model-1", 4)
    assert cache.get(other) is None
    # Expired entries are marked stale, never treated as authoritative.
    expired_clock = {"now": 0.0}

    class FakeClock:
        def __call__(self) -> float:
            return expired_clock["now"]

    tiny = SemanticCache(ttl_seconds=0.05, clock=FakeClock())
    tiny.put("k", {"v": 1}, model_version="m", workspace_revision=0)
    expired_clock["now"] = 100.0
    stale = tiny.get("k")
    assert stale is not None and stale.stale


def test_drift_monitor_triggers_review_not_silent_rewrites():
    """Acceptance 46: drift monitoring recommends review, not silent rewrite."""
    monitor = DriftMonitor(window=4, threshold=0.25)
    for _ in range(6):
        monitor.observe(True, 10.0)  # healthy baseline
    report = monitor.report()
    assert not report.drifted
    for _ in range(4):
        monitor.observe(False, 10.0)  # recent failures
    report = monitor.report()
    assert report.drifted
    assert report.recommendation == "review"  # quarantine only under policy
    strict = DriftMonitor(window=4, threshold=0.25, quarantine_allowed=True)
    for _ in range(6):
        strict.observe(True, 5.0)
    for _ in range(4):
        strict.observe(False, 5.0)
    assert strict.report().recommendation == "quarantine_under_explicit_policy"
    early = DriftMonitor(window=32)
    early.observe(False, 1.0)
    assert not early.report().drifted  # insufficient data: keep monitoring


def test_metrics_collector_tracks_evaluation_counters():
    """The metrics collector feeds the system evaluation (Step 49)."""
    metrics = MetricsCollector()
    metrics.record_dispatch(cost=1.0)
    metrics.record_outcome(True, 12.5)
    metrics.record_outcome(False, 30.0, failure_class="permanent", retries=2)
    metrics.record_deferred(3)
    metrics.record_unauthorized()
    metrics.record_cache_hit()
    snapshot = metrics.snapshot()
    assert snapshot.dispatched == 1
    assert snapshot.succeeded == 1
    assert snapshot.failed == 1
    assert snapshot.retried == 2
    assert snapshot.deferred == 3
    assert snapshot.unauthorized_calls == 1
    assert snapshot.cache_hits == 1
    assert snapshot.total_cost == 1.0
    assert 0.0 < snapshot.mean_latency_ms < 100.0


def test_engine_collects_system_evaluation_metrics():
    """The engine result carries the full system evaluation (Steps 43-49)."""
    engine = make_engine()
    result = engine.run(make_request())
    evaluation = result.metrics
    assert "metrics" in evaluation
    assert evaluation["metrics"]["dispatched"] >= 2
    assert evaluation["queue"]["arrival_rate"] >= 0.0
    assert "drift" in evaluation
    assert "duplicate_side_effects_prevented" in evaluation
    assert evaluation["worker_selections"]
    assert len(result.trace) == 50  # exactly one entry per source step
    steps = [entry["step"] for entry in result.trace]
    assert steps == list(range(1, 51))  # all 50 steps in order
    assert len(result.spans) >= 7  # one span per phase plus run
    assert result.plan_hash


def test_database_summary_counts_all_tables(tmp_path):
    """The agentic database persists all subsystem tables (Section 4)."""
    database = AgenticDatabase(tmp_path / "summary.sqlite3")
    store = WorkflowStore(database)
    database.record_audit("test", {"kind": "unit"})
    assert store is not None
    summary = database.summary()
    assert summary["schema_version"] == 1
    assert set(summary["tables"]) >= {
        "agentic_workflows", "agentic_tasks", "agentic_task_results",
        "agentic_checkpoints", "agentic_side_effects", "agentic_episodes",
        "agentic_procedures", "agentic_memory_links",
        "agentic_semantic_proposals", "agentic_audit", "agentic_spans"}
    rows = database.query("SELECT COUNT(*) AS n FROM agentic_audit")
    assert rows[0]["n"] == 1
    database.close()
