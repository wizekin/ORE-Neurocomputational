"""Acceptance tests for the MAPE-K stages (Steps 9-15)."""
import pytest

from ore_cros.actions import (
    ActionType, PolicyOrigin, ResourceAction, RuntimeControlState,
    keep_current_policy_action,
)
from ore_cros.config import CROSConfig
from ore_cros.mape.analyze import Analyzer, PressureDetector, ResourceCondition
from ore_cros.mape.execute import ControlledExecutionManager, OutcomeMeasurer
from ore_cros.mape.monitor import Monitor, TaskEventCollector
from ore_cros.mape.plan import (
    KnowledgeBase, KnowledgeEntry, Planner, select_baseline_or_constrained,
)
from ore_cros.resource_model import (
    EnvironmentProfile, ResourceKind, ResourceDemand, TaskRecord, TaskState,
    WorkloadClass, build_resource_state, machine_scope,
)
from ore_cros.safety.constraints import HardConstraintSet
from ore_cros.telemetry.collectors import CollectorSet, SimulatedCollector
from ore_cros.telemetry.history import TimeWindowedResourceHistory
from ore_cros.telemetry.quality import TelemetryQualityAssessor
from ore_cros.telemetry.schemas import Measurement, MetricKind

NOW = 1_700_000_000.0


def _profile():
    """Return a test capacity profile."""
    return EnvironmentProfile(
        profile_version="test", host_os="TestOS", kernel_release="1",
        capacity={
            ResourceKind.CPU: 4.0, ResourceKind.RAM: 8192.0,
            ResourceKind.IO: 100.0, ResourceKind.NET: 100.0,
            ResourceKind.GPU: 8192.0, ResourceKind.STORAGE: 10_000.0,
            ResourceKind.QUEUE: 64.0,
        },
    )


def _task(task_id, cls=WorkloadClass.CORE_COGNITIVE):
    """Build a minimal task for MAPE tests."""
    return TaskRecord(
        task_id=task_id, parent_goal_id=None, module_id=f"m-{task_id}",
        workload_class=cls, demand=ResourceDemand(cpu=0.5), remaining_work=1.0,
        ready_time=0.0,
    )


def _monitor():
    """Build a Monitor wired to a deterministic simulated collector."""
    collector_set = CollectorSet([SimulatedCollector(seed=7, load_profile=lambda s: 0.9)])
    history = TimeWindowedResourceHistory()
    return Monitor(collector_set, TelemetryQualityAssessor(), history), history


def test_monitor_stage_emits_versioned_coherent_snapshots():
    """Step 9: Monitor publishes timestamped, versioned snapshots."""
    monitor, _ = _monitor()
    first = monitor.collect([_task("t1")], now=NOW)
    second = monitor.collect([_task("t1")], now=NOW + 1.0)
    assert first.version == 1 and second.version == 2
    assert second.timestamp == NOW + 1.0
    assert first.measurements
    assert first.ready_tasks[0].task_id == "t1"


def test_task_event_collector_splits_ready_and_blocked():
    """Task events separate ready work from dependency-blocked work."""
    blocked = TaskRecord(
        task_id="b", parent_goal_id=None, module_id="m", workload_class=WorkloadClass.BATCH,
        demand=ResourceDemand(), remaining_work=1.0, dependencies=("upstream",),
    )
    ready = _task("r")
    assert TaskEventCollector.collect_ready_and_running([blocked, ready]) == (ready,)
    assert TaskEventCollector.blocked_tasks([blocked, ready]) == (blocked,)


def test_analyze_stage_diagnoses_cpu_bound_with_evidence():
    """Step 10: the analyzer ranks bottleneck candidates with evidence."""
    monitor, history = _monitor()
    snapshot = monitor.collect([_task("t1"), _task("t2")], now=NOW)
    detector = PressureDetector()
    pressure = detector.detect(
        snapshot.measurements, history, now=NOW, window_seconds=10.0,
    )
    diagnosis = Analyzer().diagnose(snapshot, pressure)
    assert diagnosis.condition in (
        ResourceCondition.CPU_BOUND, ResourceCondition.MEMORY_BOUND,
        ResourceCondition.IO_BOUND, ResourceCondition.NETWORK_BOUND,
    )
    assert diagnosis.candidates
    assert diagnosis.candidates[0].rank_score >= diagnosis.candidates[-1].rank_score


def test_pressure_detector_separates_sustained_saturation_from_spikes():
    """Step 6: sustained saturation and short spikes are distinguished."""
    detector = PressureDetector(saturation_threshold=0.8, sustained_windows=3)
    history = TimeWindowedResourceHistory()
    measurements = []
    for index in range(5):
        timestamp = NOW - 5.0 + index
        measurement = Measurement(
            resource=ResourceKind.CPU, metric=MetricKind.PRESSURE,
            value=0.9, unit="ratio", timestamp=timestamp,
            scope=machine_scope(), source="test",
        )
        measurements.append(measurement)
        history.record([measurement])
    vector = detector.detect(
        measurements[-1:], history, now=NOW, window_seconds=60.0,
    )
    reading = vector.reading(ResourceKind.CPU)
    assert reading.sustained is True and reading.spike is False

    spike_history = TimeWindowedResourceHistory()
    spike_measurement = Measurement(
        resource=ResourceKind.CPU, metric=MetricKind.PRESSURE, value=0.95,
        unit="ratio", timestamp=NOW, scope=machine_scope(), source="test",
    )
    spike_history.record([spike_measurement])
    spike_vector = detector.detect(
        [spike_measurement], spike_history, now=NOW, window_seconds=60.0,
    )
    spike_reading = spike_vector.reading(ResourceKind.CPU)
    assert spike_reading.spike is True and spike_reading.sustained is False


def test_littles_law_check_in_queue_reading():
    """Little's Law L = lambda * W is checked on queue observations (spec 3.2)."""
    measurements = [
        Measurement(
            resource=ResourceKind.QUEUE, metric=MetricKind.QUEUE_DEPTH, value=6.0,
            unit="jobs", timestamp=NOW, scope=machine_scope(), source="test",
        ),
        Measurement(
            resource=ResourceKind.QUEUE, metric=MetricKind.ARRIVAL_RATE, value=2.0,
            unit="jobs/s", timestamp=NOW, scope=machine_scope(), source="test",
        ),
        Measurement(
            resource=ResourceKind.QUEUE, metric=MetricKind.WAIT_SECONDS, value=3.0,
            unit="s", timestamp=NOW, scope=machine_scope(), source="test",
        ),
    ]
    vector = PressureDetector().detect(
        measurements, TimeWindowedResourceHistory(), now=NOW, window_seconds=10.0,
    )
    assert vector.queue.littles_law_work == pytest.approx(6.0)
    assert vector.queue.depth == 6.0


def test_planner_generates_candidates_with_costs_and_rollback():
    """Step 11: candidates carry predicted effects, costs and rollback conditions."""
    monitor, history = _monitor()
    snapshot = monitor.collect([_task("t1")], now=NOW)
    pressure = PressureDetector().detect(
        snapshot.measurements, history, now=NOW, window_seconds=10.0,
    )
    diagnosis = Analyzer().diagnose(snapshot, pressure)
    knowledge = KnowledgeBase()
    candidates = Planner(CROSConfig()).generate(
        diagnosis, knowledge.retrieve(diagnosis), current_worker_count=4,
    )
    assert candidates
    for action in candidates:
        assert action.rollback_condition
        assert action.rationale
        assert action.policy_version
    assert any(
        action.action_type is ActionType.KEEP_CURRENT_POLICY for action in candidates
    )


def test_knowledge_base_distinguishes_measured_and_learned():
    """Step 12: knowledge entries carry provenance and bounded retention."""
    from types import SimpleNamespace

    knowledge = KnowledgeBase()
    knowledge.record_fact(KnowledgeEntry(
        key="condition/CPU_BOUND", fact="deferring exploration helped",
        provenance="measured", value=0.8,
    ))
    knowledge.record_fact(KnowledgeEntry(
        key="general/default-workers", fact="4 workers fit this machine",
        provenance="declared", value=4.0,
    ))
    with pytest.raises(ValueError):
        knowledge.record_fact(KnowledgeEntry(
            key="x", fact="bad provenance", provenance="rumor",
        ))
    diagnosis = SimpleNamespace(condition=ResourceCondition.CPU_BOUND)
    context = knowledge.retrieve(diagnosis)
    keys = {entry.key for entry in context.entries}
    # The condition-specific entry is retrieved; unrelated conditions are not.
    assert "condition/CPU_BOUND" in keys
    assert "condition/MEMORY_BOUND" not in keys
    assert context.knowledge_version


def test_each_mape_stage_is_independently_testable():
    """Acceptance: each MAPE-K stage can be tested with typed inputs/outputs."""
    # Monitor, Analyzer, Planner and Execute all run above with typed
    # records; here Execute is exercised in isolation.
    constraints = HardConstraintSet(min_workers=1, max_workers=8)
    executor = ControlledExecutionManager(constraints)
    control = RuntimeControlState(worker_count=2)
    action = ResourceAction(
        action_id="scale-up", action_type=ActionType.ADJUST_WORKER_COUNT,
        parameters={"target_worker_count": 4}, rationale="absorb backlog",
        policy_origin=PolicyOrigin.DETERMINISTIC, policy_version="v1",
        state_version=1,
    )
    result = executor.apply(action, control, now=NOW)
    assert result.success and control.worker_count == 4
    assert executor.rollback(result.undo_token, control)
    assert control.worker_count == 2


def test_constraint_filter_rejects_violating_candidates():
    """Acceptance: candidate plans violating hard constraints are rejected
    (Step 13)."""
    constraints = HardConstraintSet(min_workers=1, max_workers=8)
    state = build_resource_state(_profile(), machine_scope(), version=7)
    control = RuntimeControlState()
    good = ResourceAction(
        action_id="good", action_type=ActionType.ADJUST_WORKER_COUNT,
        parameters={"target_worker_count": 6}, rationale="within bounds",
        policy_origin=PolicyOrigin.DETERMINISTIC, policy_version="v1",
        state_version=7,
    )
    bad = ResourceAction(
        action_id="bad", action_type=ActionType.ADJUST_WORKER_COUNT,
        parameters={"target_worker_count": 99}, rationale="out of bounds",
        policy_origin=PolicyOrigin.DETERMINISTIC, policy_version="v1",
        state_version=7,
    )
    stale = ResourceAction(
        action_id="stale", action_type=ActionType.ADJUST_WORKER_COUNT,
        parameters={"target_worker_count": 6}, rationale="wrong state version",
        policy_origin=PolicyOrigin.DETERMINISTIC, policy_version="v1",
        state_version=6,
    )
    safe, rejections = constraints.filter_safe_actions(
        [good, bad, stale], state, control,
    )
    assert [action.action_id for action in safe] == ["good"]
    assert "worker count 99 outside [1, 8]" in rejections["bad"]
    assert rejections["stale"]


def test_policy_engine_cannot_bypass_execution_manager():
    """Acceptance: the policy engine cannot invoke arbitrary OS commands or
    bypass the execution manager (Step 14)."""
    constraints = HardConstraintSet()
    executor = ControlledExecutionManager(constraints)
    control = RuntimeControlState()
    # A well-formed action of every known type passes through typed paths...
    keep = keep_current_policy_action(0, "v1")
    result = executor.apply(keep, control, now=NOW)
    assert result.success
    # ...and unknown parameter shapes cannot even be constructed.
    with pytest.raises(ValueError):
        ResourceAction(
            action_id="weird", action_type=ActionType.ADJUST_WORKER_COUNT,
            parameters={"shell_command": "rm -rf /"}, rationale="arbitrary",
            policy_origin=PolicyOrigin.LEARNED, policy_version="v1",
            state_version=0,
        )
    # The executor only mutates the typed control state.
    assert executor._mutate(keep, control) == {}


def test_every_decision_records_full_provenance():
    """Acceptance: every action records state version, policy version,
    constraints, rationale, predicted and measured outcome."""
    monitor, history = _monitor()
    snapshot = monitor.collect([_task("t1")], now=NOW)
    pressure = PressureDetector().detect(
        snapshot.measurements, history, now=NOW, window_seconds=10.0,
    )
    diagnosis = Analyzer().diagnose(snapshot, pressure)
    config = CROSConfig()
    planner = Planner(config)
    candidates = planner.generate(
        diagnosis, KnowledgeBase().retrieve(diagnosis), current_worker_count=4,
    )
    selected = select_baseline_or_constrained(candidates, canary_active=False)
    predicted = {"latency_p95": 0.1, "queue_delay": 0.05}
    outcome = OutcomeMeasurer.measure(
        snapshot.measurements, predicted=predicted,
    )
    from ore_cros.mape.plan import DecisionRecord

    record = DecisionRecord(
        state_version=selected.state_version, policy_version=selected.policy_version,
        action_id=selected.action_id, action_type=selected.action_type.value,
        policy_origin=selected.policy_origin.value, rationale=selected.rationale,
        constraints_checked=("all hard constraints",),
        predicted_outcome=predicted, measured_outcome=outcome.to_dict(),
        timestamp=NOW,
    )
    payload = record.to_dict()
    for key in (
        "state_version", "policy_version", "action_id", "action_type",
        "policy_origin", "rationale", "constraints_checked",
        "predicted_outcome", "measured_outcome",
    ):
        assert key in payload
    assert payload["state_version"] == selected.state_version


def test_outcome_measurer_computes_residuals():
    """Step 15: residual = observed outcome - predicted outcome."""
    measurements = [
        Measurement(
            resource=ResourceKind.IO, metric=MetricKind.LATENCY_SECONDS,
            value=0.2, unit="s", timestamp=NOW, scope=machine_scope(), source="t",
        ),
        Measurement(
            resource=ResourceKind.QUEUE, metric=MetricKind.WAIT_SECONDS,
            value=0.3, unit="s", timestamp=NOW, scope=machine_scope(), source="t",
        ),
    ]
    outcome = OutcomeMeasurer.measure(
        measurements, predicted={"latency_p95": 0.1, "queue_delay": 0.2},
    )
    assert outcome.residuals["latency_p95"] == pytest.approx(0.1)
    assert outcome.residuals["queue_delay"] == pytest.approx(0.1)
