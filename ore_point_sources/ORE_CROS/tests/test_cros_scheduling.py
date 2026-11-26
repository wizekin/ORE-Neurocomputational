"""Acceptance tests for deterministic and adaptive CPU scheduling (Steps 16-26)."""
import pytest

from ore_cros.actions import ActionType, PolicyOrigin, ResourceAction, RuntimeControlState
from ore_cros.config import AdaptationLimits
from ore_cros.policies.baseline import (
    AdmissionController, DeterministicBaselineScheduler,
)
from ore_cros.policies.cpu import (
    ConstrainedAdaptation, EarliestDeadlineFirstScheduler, GangScheduler,
    MultilevelFeedbackQueue, QuotaManager, RateMonotonicScheduler,
    ShadowModeEvaluator, StarvationGuard, WeightedFairScheduler,
)
from ore_cros.resource_model import (
    EnvironmentProfile, ModuleBudget, ResourceDemand, ResourceKind, TaskRecord,
    TaskState, WorkloadClass, build_resource_state, machine_scope,
)


def _profile():
    """Return a test capacity profile: 4 cores, 8 GiB RAM."""
    return EnvironmentProfile(
        profile_version="test", host_os="TestOS", kernel_release="1.0",
        capacity={
            ResourceKind.CPU: 4.0, ResourceKind.RAM: 8192.0,
            ResourceKind.IO: 100.0, ResourceKind.NET: 100.0,
            ResourceKind.GPU: 8192.0, ResourceKind.STORAGE: 10_000.0,
            ResourceKind.QUEUE: 64.0,
        },
    )


def _state(**overrides):
    """Build a resource state from the test profile."""
    return build_resource_state(_profile(), machine_scope(), **overrides)


def _task(task_id, cls=WorkloadClass.BATCH, cpu=1.0, **overrides):
    """Build a schedulable task for scheduling tests."""
    defaults = dict(
        task_id=task_id, parent_goal_id=None, module_id=f"mod-{task_id}",
        workload_class=cls, demand=ResourceDemand(cpu=cpu), remaining_work=1.0,
        ready_time=0.0,
    )
    defaults.update(overrides)
    return TaskRecord(**defaults)


def test_identical_traces_produce_identical_decisions():
    """Acceptance: identical task traces under the deterministic policy
    produce equivalent allocation decisions."""
    scheduler = DeterministicBaselineScheduler(available_capabilities=["cap1"])
    tasks = [
        _task("b1", WorkloadClass.BATCH, capabilities=("cap1",)),
        _task("i1", WorkloadClass.INTERACTIVE, capabilities=("cap1",)),
        _task("c1", WorkloadClass.CORE_COGNITIVE, capabilities=("cap1",)),
        _task("v1", WorkloadClass.INTEGRITY_CRITICAL, capabilities=("cap1",)),
    ]
    control = RuntimeControlState(worker_count=2)
    first = scheduler.schedule(tasks, _state(), control, now=10.0)
    second = scheduler.schedule(list(reversed(tasks)), _state(), control, now=10.0)
    assert first.admitted_order == second.admitted_order
    assert first.admitted_order[0] == "v1"  # integrity-critical first


def test_admission_defers_optionals_before_ceilings_breach():
    """Acceptance: admission control defers or rejects work before resource
    ceilings are exceeded; optional classes defer first."""
    controller = AdmissionController()
    state = _state(allocations={ResourceKind.CPU: 3.9})
    optional = _task("explore", WorkloadClass.EXPLORATORY, cpu=0.5)
    critical = _task("verify", WorkloadClass.INTEGRITY_CRITICAL, cpu=0.5)
    optional_decision = controller.decide(optional, state)
    critical_decision = controller.decide(critical, state)
    assert optional_decision.deferred and not optional_decision.admitted
    assert not critical_decision.admitted and not critical_decision.deferred


def test_integrity_critical_retains_reserved_capacity_under_batch_overload():
    """Acceptance: integrity-critical tasks retain reserved capacity under
    batch overload."""
    controller = AdmissionController()
    state = _state(
        allocations={ResourceKind.CPU: 2.0},
        reservations={ResourceKind.CPU: 2.0},
    )
    batch = _task("bulk", WorkloadClass.BATCH, cpu=1.0)
    integrity = _task("check", WorkloadClass.INTEGRITY_CRITICAL, cpu=1.0)
    assert controller.decide(batch, state).admitted is False
    assert controller.decide(integrity, state, reserved=ResourceDemand(cpu=2.0)).admitted


def test_weighted_fairness_matches_configured_weights():
    """Acceptance: general workloads receive service consistent with
    configured weights (Step 17)."""
    scheduler = WeightedFairScheduler()
    scheduler.register("a", weight=3.0)
    scheduler.register("b", weight=1.0)
    for _ in range(400):
        chosen = scheduler.pick_next()
        scheduler.account(chosen, 0.01)
    report = scheduler.fairness_report()
    assert report.ideal_shares["a"] == pytest.approx(0.75)
    assert report.achieved_shares["a"] == pytest.approx(0.75, abs=0.05)
    # Jain 0.8 is the index of a perfect 3:1 weighted split between two tasks.
    assert report.jain_index == pytest.approx(0.8, abs=0.02)
    assert report.max_deviation < 0.01
    received, owed = scheduler.service_tables()
    assert set(received) == {"a", "b"} and owed == {"a": 0.0, "b": 0.0}


def test_mlfq_aging_prevents_indefinite_demotion():
    """Acceptance: aging and reservations prevent indefinite waiting
    (Step 21)."""
    queue = MultilevelFeedbackQueue(levels=3, quanta=(1.0, 2.0, 4.0), aging_threshold=10.0)
    queue.submit("long-runner", now=0.0)
    for _ in range(6):
        queue.account("long-runner", 1.0)  # exhausts quanta and demotes
    assert queue.level_of("long-runner") == 2
    queue.submit("quiet", now=20.0)
    # Aging lifts the demoted task one level per threshold window.
    first = queue.pick_next(now=30.0)  # long-runner promoted to level 1; quiet at 0 wins
    assert first == "quiet"
    queue.account("quiet", 0.5)
    queue.complete("quiet")
    second = queue.pick_next(now=45.0)  # long-runner promoted to level 0 and picked
    assert second == "long-runner"
    assert queue.level_of("long-runner") == 0


def test_edf_orders_by_deadline_and_checks_feasibility():
    """Acceptance: EDF orders eligible deadline jobs correctly and checks
    feasibility assumptions (Step 19)."""
    tasks = [
        _task("late", deadline=10.0),
        _task("early", deadline=5.0),
        _task("none"),
    ]
    order = EarliestDeadlineFirstScheduler.order(tasks, now=0.0)
    assert order == ("early", "late")  # 'none' has no deadline, excluded
    feasible, _ = EarliestDeadlineFirstScheduler.feasible(
        tasks, now=0.0, cpu_capacity=1.0,
    )
    assert feasible
    overloaded = [
        _task("big", deadline=1.0, remaining_work=5.0),
    ]
    feasible, reason = EarliestDeadlineFirstScheduler.feasible(
        overloaded, now=0.0, cpu_capacity=1.0,
    )
    assert not feasible and "needs" in reason


def test_rate_monotonic_priorities_and_bound():
    """Acceptance: Rate Monotonic tests validate configured periods and
    execution assumptions (Step 20)."""
    scheduler = RateMonotonicScheduler()
    priorities = scheduler.priorities({"monitor-1s": 1.0, "monitor-10s": 10.0})
    assert priorities["monitor-1s"] < priorities["monitor-10s"]
    feasible, _ = scheduler.feasible(
        {"a": 10.0, "b": 20.0}, {"a": 1.0, "b": 2.0},
    )
    assert feasible
    infeasible, _ = scheduler.feasible(
        {"a": 10.0, "b": 10.1, "c": 20.0}, {"a": 4.0, "b": 4.0, "c": 8.0},
    )
    assert not infeasible
    with pytest.raises(ValueError):
        scheduler.priorities({"bad": -1.0})
    bad_wcet, reason = scheduler.feasible({"a": 2.0}, {"a": 5.0})
    assert not bad_wcet and "exceeds" in reason


def test_gang_scheduler_dispatches_only_complete_gangs():
    """Gang execution coordinates groups that progress together (Step 22)."""
    gang = GangScheduler()
    gang.declare_gang("g1", ["a", "b", "c"])
    gang.mark_ready("a")
    gang.mark_ready("b")
    assert gang.dispatchable_gangs() == ()
    gang.mark_ready("c")
    assert gang.dispatchable_gangs() == ("g1",)
    assert gang.dispatch()["g1"] == ("a", "b", "c")


def test_quotas_isolate_reserved_budgets():
    """Acceptance: one module cannot silently consume another module's
    reserved budget (Step 23)."""
    quotas = QuotaManager({
        "indexer": ModuleBudget("indexer", ResourceDemand(cpu=1.0), ResourceDemand(cpu=2.0)),
        "verifier": ModuleBudget("verifier", ResourceDemand(cpu=1.0), ResourceDemand(cpu=2.0)),
    })
    ok, _ = quotas.admit("indexer", ResourceDemand(cpu=1.5))
    assert ok
    denied, reason = quotas.admit("indexer", ResourceDemand(cpu=3.0))
    assert not denied and "exceeds" in reason
    # Borrowing without authorization is refused and recorded nowhere.
    allowed, _ = quotas.borrow("indexer", "verifier", ResourceDemand(cpu=0.5), authorized=False)
    assert not allowed and quotas.borrow_events() == ()


def test_starvation_guard_flags_and_ages_waiting_tasks():
    """Starvation guards age waiting tasks and flag bounded-wait breaches (Step 24)."""
    guard = StarvationGuard(age_gain=0.2, wait_alert_threshold=30.0)
    waiting = _task("waiting", ready_time=0.0)
    fresh = _task("fresh", ready_time=45.0)
    assert guard.effective_priority(waiting, now=50.0) < guard.effective_priority(fresh, now=50.0)
    assert guard.starved_tasks([waiting], now=50.0) == ("waiting",)
    report = guard.wait_report([waiting, fresh], now=50.0)
    assert set(report) == {"NORMAL"}


def test_shadow_mode_records_without_controlling():
    """Acceptance: learned actions cannot change allocation in shadow mode
    is prepared here; the evaluator collects a dataset and executes nothing
    (Step 25)."""
    evaluator = ShadowModeEvaluator()
    baseline = ResourceAction(
        action_id="keep", action_type=ActionType.KEEP_CURRENT_POLICY,
        parameters={}, rationale="baseline", policy_origin=PolicyOrigin.BASELINE,
        policy_version="v1", state_version=1,
    )
    proposal = ResourceAction(
        action_id="scale", action_type=ActionType.ADJUST_WORKER_COUNT,
        parameters={"target_worker_count": 5}, rationale="proposal",
        policy_origin=PolicyOrigin.LEARNED, policy_version="v1", state_version=1,
    )
    evaluator.observe(proposal, baseline, context={"now": 1.0})
    evaluator.observe(proposal, baseline, context={"now": 2.0})
    assert len(evaluator.dataset()) == 2
    assert evaluator.applied_actions == []  # shadow mode never applies
    assert evaluator.agreement_rate() == 0.0


def test_constrained_adaptation_reverts_to_baseline():
    """Constrained adaptation only permits reversible actions under a
    healthy watchdog and active canary (Step 26)."""
    limits = AdaptationLimits()
    adaptation = ConstrainedAdaptation(limits)
    baseline = ResourceAction(
        action_id="keep", action_type=ActionType.KEEP_CURRENT_POLICY,
        parameters={}, rationale="baseline", policy_origin=PolicyOrigin.BASELINE,
        policy_version="v1", state_version=1,
    )
    irreversible = ResourceAction(
        action_id="admission", action_type=ActionType.ADJUST_ADMISSION,
        parameters={"admission_enabled": False}, rationale="risky",
        policy_origin=PolicyOrigin.LEARNED, policy_version="v1", state_version=1,
    )
    assert adaptation.decide(
        irreversible, canary_active=True, watchdog_healthy=True, baseline_action=baseline,
    ) is baseline
    reversible = ResourceAction(
        action_id="scale", action_type=ActionType.ADJUST_WORKER_COUNT,
        parameters={"target_worker_count": 5}, rationale="safe scaling",
        policy_origin=PolicyOrigin.LEARNED, policy_version="v1", state_version=1,
    )
    assert adaptation.decide(
        reversible, canary_active=True, watchdog_healthy=True,
    ) is reversible
    assert adaptation.decide(
        reversible, canary_active=False, watchdog_healthy=True, baseline_action=baseline,
    ) is baseline
    assert adaptation.decide(
        reversible, canary_active=True, watchdog_healthy=False, baseline_action=baseline,
    ) is baseline
    assert adaptation.decide(
        reversible, canary_active=True, watchdog_healthy=True,
        prediction_error=0.9, baseline_action=baseline,
    ) is baseline


def test_worker_pool_bounds_admitted_concurrency():
    """The bounded worker pool caps admitted tasks per round (Step 16)."""
    scheduler = DeterministicBaselineScheduler()
    tasks = [_task(f"t{index}") for index in range(10)]
    control = RuntimeControlState(worker_count=3)
    decision = scheduler.schedule(tasks, _state(), control, now=0.0)
    assert len(decision.admitted_order) == 3
    # Tasks beyond the pool wait explicitly rather than disappearing.
    waiting = [line for line in decision.order_rationale if "bounded worker pool" in line]
    assert len(waiting) == 7
    assert decision.deferred == () and decision.rejected == ()


def test_dependency_and_capability_readiness():
    """Dependency readiness and capability checks gate scheduling (Step 16)."""
    scheduler = DeterministicBaselineScheduler(available_capabilities=["cap"])
    blocked = _task("blocked", dependencies=("upstream",))
    incapable = _task("incapable", capabilities=("missing-cap",))
    ready = _task("ready")
    decision = scheduler.schedule(
        [blocked, incapable, ready], _state(), RuntimeControlState(),
        completed=["upstream"], now=0.0,
    )
    # 'blocked' has its dependency satisfied via completed; 'incapable' cannot run.
    assert "incapable" not in decision.admitted_order
    assert decision.admitted_order in (("ready", "blocked"), ("blocked", "ready"))
    rationale = " ".join(decision.order_rationale)
    assert "missing capabilities" in rationale
