"""End-to-end acceptance tests: the adaptive resource cycle and ORE integration."""
import pytest

from ore_cros.config import CROSConfig
from ore_cros.cycle import CROSCoordinator, run_replay_benchmark
from ore_cros.evaluation.benchmarks import (
    ReplayBenchmark, TaskTraceEvent, compare_policies, long_replay_stability,
)
from ore_cros.integration import (
    OreWorkloadClassMapper, SqliteDecisionStore, cros_task_from_metadata,
    describe_boundaries,
)
from ore_cros.learning.bandit import ConstrainedContextualBandit
from ore_cros.policies.baseline import DeterministicBaselineScheduler
from ore_cros.prediction.demand import DemandPredictor
from ore_cros.resource_model import ResourceDemand, ResourceKind, TaskRecord, WorkloadClass
from ore_cros.telemetry.collectors import CollectorSet, SimulatedCollector

NOW = 1_700_000_000.0


def _coordinator(load=0.9, seed=7, **coordinator_kwargs):
    """Build a coordinator driven by a deterministic simulated collector."""
    collector_set = CollectorSet([
        SimulatedCollector(seed=seed, load_profile=lambda step: load)
    ])
    defaults = dict(config=CROSConfig(), collector_set=collector_set)
    defaults.update(coordinator_kwargs)
    return CROSCoordinator(**defaults)


def _tasks():
    """Return a mixed task set: integrity-critical, optional and batch work."""
    return [
        TaskRecord(
            task_id="verify-1", parent_goal_id="goal", module_id="workspace-verify",
            workload_class=WorkloadClass.INTEGRITY_CRITICAL,
            demand=ResourceDemand(cpu=0.2), remaining_work=1.0,
        ),
        TaskRecord(
            task_id="explore-1", parent_goal_id="goal", module_id="hypothesis-search",
            workload_class=WorkloadClass.EXPLORATORY,
            demand=ResourceDemand(cpu=0.3), remaining_work=2.0,
        ),
        TaskRecord(
            task_id="index-1", parent_goal_id="goal", module_id="bulk-index",
            workload_class=WorkloadClass.BATCH,
            demand=ResourceDemand(cpu=0.3), remaining_work=3.0,
        ),
    ]


def test_full_cycle_returns_state_and_decision_trace():
    """Section 5: the loop returns the resource state and decision trace."""
    coordinator = _coordinator(load=0.2)
    result = coordinator.adaptive_resource_cycle(_tasks(), now=NOW)
    assert result.cycle_number == 1
    assert result.state.version >= 1
    assert result.diagnosis.condition.value == "HEALTHY"
    assert result.action.action_type.value == "KEEP_CURRENT_POLICY"
    assert result.decision is not None
    assert result.execution is not None and result.execution.success
    assert result.to_dict()["cycle_number"] == 1


def test_pressure_defers_optional_before_integrity_critical():
    """Acceptance: optional exploration/indexing is deferred before
    integrity-critical operations (Step 47 ordering)."""
    coordinator = _coordinator(load=0.95)
    result = coordinator.adaptive_resource_cycle(_tasks(), now=NOW)
    assert result.action.action_type.value == "DEFER_BACKGROUND"
    deferred = {item.value for item in coordinator.control.deferred_classes}
    assert "EXPLORATORY" in deferred and "BATCH" in deferred
    # Integrity-critical and interactive work are never deferred.
    assert "INTEGRITY_CRITICAL" not in deferred
    assert "INTERACTIVE" not in deferred


def test_cycle_versions_state_and_stores_decisions():
    """Every cycle increments state version and records provenance (Step 15)."""
    coordinator = _coordinator(load=0.9)
    versions = []
    for index in range(3):
        result = coordinator.adaptive_resource_cycle(_tasks(), now=NOW + index * 10.0)
        versions.append(result.state.version)
    assert versions == sorted(versions) and len(set(versions)) == 3
    assert len(coordinator.decisions) == 3
    record = coordinator.decisions[-1]
    assert record.policy_version == coordinator.config.policy_version
    assert record.measured_outcome is not None


def test_cycle_updates_demand_predictors_by_class():
    """Step 49: predictors observe telemetry from every cycle."""
    coordinator = _coordinator(load=0.9)
    coordinator.adaptive_resource_cycle(_tasks(), now=NOW)
    report = coordinator.demand_predictor.error_report()
    assert report  # error tracked by resource/workload class
    prediction = coordinator.demand_predictor.predict(
        ResourceKind.CPU, WorkloadClass.INTERACTIVE, horizon=1.0,
    )
    assert prediction.sample_count > 0
    assert prediction.interval_low <= prediction.point <= prediction.interval_high


def test_prediction_error_and_uncertainty_measured_by_class():
    """Acceptance: predictor error and uncertainty are measured by
    resource/workload class."""
    predictor = DemandPredictor()
    first = predictor.predict(ResourceKind.CPU, WorkloadClass.BATCH)
    predictor.observe(
        ResourceKind.CPU, WorkloadClass.BATCH, 1.0, previous_prediction=first,
    )
    second = predictor.predict(ResourceKind.CPU, WorkloadClass.BATCH)
    predictor.observe(
        ResourceKind.CPU, WorkloadClass.BATCH, 1.2, previous_prediction=second,
    )
    report = predictor.error_report()["CPU/BATCH"]
    assert report["samples"] == 2.0
    assert report["mean_absolute_error"] >= 0.0
    assert report["interval_coverage"] in (0.0, 1.0)


def test_long_replay_stability_no_drift_or_growth():
    """Acceptance: long replay shows no unbounded memory growth, oscillation
    or policy drift."""
    coordinator = _coordinator(load=0.7)
    action_counts: dict[str, int] = {}
    for index in range(50):
        result = coordinator.adaptive_resource_cycle(
            _tasks(), now=NOW + index * 10.0,
        )
        action_counts[result.action.action_type.value] = (
            action_counts.get(result.action.action_type.value, 0) + 1
        )
        assert coordinator.history.total_samples() <= coordinator.history.memory_bound()
    stability = long_replay_stability(
        coordinator.history.total_samples(),
        coordinator.history.memory_bound(),
        oscillation_reversals=coordinator.oscillation.reversal_count(),
        oscillation_limit=coordinator.config.limits.oscillation_reversal_limit,
        action_distribution=action_counts,
        reference_distribution={
            key: 0.5 for key in action_counts
        },
        drift_tolerance=0.6,
    )
    assert stability.bounded_memory
    assert not stability.oscillation_detected
    assert stability.stable


def test_benchmark_reports_required_metrics():
    """Acceptance: benchmarks report throughput, p50/p95/p99 latency, queue
    delay, pressure, energy where available and policy overhead."""
    trace = [
        TaskTraceEvent(
            task_id=f"t{index}", module_id="m",
            workload_class=WorkloadClass.CORE_COGNITIVE,
            arrival_time=float(index) * 0.1, service_time=0.5,
        )
        for index in range(30)
    ]
    benchmark = ReplayBenchmark(
        DeterministicBaselineScheduler(), worker_count=4, energy_available=True,
    )
    report = benchmark.run(trace, pressure_by_class={WorkloadClass.CORE_COGNITIVE: 0.3})
    assert report.completed == 30
    assert report.throughput > 0
    assert 0.0 <= report.latency_p50 <= report.latency_p95 <= report.latency_p99
    assert report.queue_delay_p50 >= 0
    assert report.mean_pressure == pytest.approx(0.3)
    assert report.energy_joules is not None and report.energy_joules > 0
    assert report.policy_overhead_seconds >= 0.0
    assert report.constraint_violations == 0


def test_policy_comparison_acceptance_criteria():
    """Policy comparison applies the section-8 acceptance criteria."""
    trace = [
        TaskTraceEvent(
            task_id=f"t{index}", module_id="m", workload_class=WorkloadClass.BATCH,
            arrival_time=0.0, service_time=0.3,
        )
        for index in range(20)
    ]
    baseline = run_replay_benchmark(trace, worker_count=4)
    candidate = run_replay_benchmark(trace, worker_count=4)
    comparison = compare_policies(baseline, candidate)
    assert comparison.accepted
    assert comparison.to_dict()["acceptance"]["throughput_not_reduced"]


def test_integration_mapper_is_policy_controlled():
    """ORE metadata maps to CROS classes through the policy mapper only."""
    mapper = OreWorkloadClassMapper()
    assert mapper.classify(module_id="workspace-consistency") is WorkloadClass.INTEGRITY_CRITICAL
    assert mapper.classify(module_id="dialogue-agent") is WorkloadClass.INTERACTIVE
    assert mapper.classify(module_id="bulk-indexer") is WorkloadClass.BATCH
    assert mapper.classify(module_id="hypothesis-expander") is WorkloadClass.EXPLORATORY
    assert mapper.classify(module_id="backup-compaction") is WorkloadClass.MAINTENANCE
    assert mapper.classify(module_id="reasoning-core") is WorkloadClass.CORE_COGNITIVE
    declared = mapper.classify(
        module_id="anything", metadata={"workload_class": "BATCH"},
    )
    assert declared is WorkloadClass.BATCH
    bogus = mapper.classify(
        module_id="anything", metadata={"workload_class": "SUPER_URGENT"},
    )
    assert bogus is WorkloadClass.CORE_COGNITIVE  # conservative default


def test_cros_task_from_metadata_projection():
    """Task metadata projects into the CROS task model (section 6 boundary)."""
    task = cros_task_from_metadata(
        task_id="ore-1", module_id="bulk-index",
        metadata={"parent_goal_id": "goal-7", "demand_cpu": 0.4, "demand_ram": 128.0},
        remaining_work=2.0, deadline=100.0, dependencies=("upstream",),
    )
    assert task.workload_class is WorkloadClass.BATCH
    assert task.parent_goal_id == "goal-7"
    assert task.demand.cpu == 0.4 and task.demand.ram == 128.0
    assert task.deadline == 100.0 and task.dependencies == ("upstream",)


def test_sqlite_decision_store_bounded_persistence(tmp_path):
    """Decision provenance persists to an ore_logger-style database."""
    from ore_cros.mape.plan import DecisionRecord

    store = SqliteDecisionStore(tmp_path / "ore_logger.sqlite3")
    record = DecisionRecord(
        state_version=1, policy_version="v1", action_id="defer-optional-work",
        action_type="DEFER_BACKGROUND", policy_origin="DETERMINISTIC",
        rationale="cpu pressure", constraints_checked=("all hard constraints",),
        predicted_outcome={"latency_p95": 0.1}, measured_outcome={"latency_p95": 0.12},
        timestamp=NOW,
    )
    store.store(record)
    store.store(record)
    assert store.count() == 2


def test_integration_boundaries_documented():
    """The section-6 responsibility map is available as structured data."""
    boundaries = describe_boundaries()
    components = {item["component"] for item in boundaries}
    assert "AdaptiveResourceCoordinator" in components
    assert "SharedWorkspace" in components
    assert "ore_usg" in components
    shared = next(item for item in boundaries if item["component"] == "SharedWorkspace")
    assert "not a general telemetry database" in shared["boundary"]


def test_bandit_shadow_cycle_in_integration():
    """A shadow-mode bandit integrates with the full cycle without control."""
    bandit = ConstrainedContextualBandit(
        approved_actions=("keep-current", "defer-optional-work", "scale-workers-up"),
        feature_names=(
            "cpu_pressure", "ram_pressure", "io_pressure",
            "net_pressure", "queue_depth", "worker_load",
        ),
        epsilon=0.0, shadow=True,
    )
    coordinator = _coordinator(load=0.95, bandit=bandit)
    coordinator.adaptive_resource_cycle(_tasks(), now=NOW)
    assert coordinator.shadow_evaluator.dataset()
    assert bandit.log_records()
    assert all(
        record.policy_origin != "LEARNED" for record in coordinator.decisions
    )
