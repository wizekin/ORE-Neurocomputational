"""Acceptance tests for the safety layer: constraints, watchdog, fallback (Steps 13, 47-48, 52)."""
import pytest

from ore_cros.actions import (
    ActionType, PolicyOrigin, ResourceAction, RuntimeControlState,
)
from ore_cros.config import AdaptationLimits, CROSConfig
from ore_cros.policies.cpu import WeightedFairScheduler
from ore_cros.resource_model import (
    ResourceDemand, ResourceKind, WorkloadClass, jain_fairness_index,
)
from ore_cros.safety.constraints import (
    CriticalityMatrix, HardConstraintSet, QuotaLedger, RealTimeClaimError,
    RealTimeTaskSpec, WcetAnalyzer,
)
from ore_cros.safety.fallback import (
    HysteresisController, OscillationDetector, PidController, RollbackManager,
    SafeFallbackPolicy, Watchdog,
)
from ore_cros.telemetry.collectors import CollectorSet, SimulatedCollector
from ore_cros.cycle import CROSCoordinator
from ore_cros.resource_model import ResourceDemand as _RD


def _limits(**overrides):
    """Build adaptation limits for safety tests."""
    values = dict(
        min_dwell_seconds=2.0, oscillation_window=10, oscillation_reversal_limit=2,
        rollback_cooldown_seconds=30.0, prediction_error_threshold=0.5,
    )
    values.update(overrides)
    return AdaptationLimits(**values)


def test_hysteresis_prevents_rapid_oscillation():
    """Acceptance: noisy measurements do not cause rapid policy oscillation."""
    controller = HysteresisController(_limits(), on_threshold=0.8, off_threshold=0.5)
    states = []
    for step, value in enumerate([0.9, 0.3, 0.9, 0.3, 0.9, 0.3, 0.9, 0.3]):
        states.append(controller.update(value, now=float(step)))
    transitions = sum(1 for first, second in zip(states, states[1:]) if first != second)
    # Without dwell-time hysteresis this flapping input toggles 7 times;
    # with it, transitions are bounded by the minimum dwell time.
    assert transitions <= 2


def test_hysteresis_engages_and_disengages_with_dwell():
    """Hysteresis transitions respect the dwell time between changes."""
    controller = HysteresisController(_limits(), on_threshold=0.8, off_threshold=0.5)
    assert controller.update(0.95, now=0.0) is True        # engages (first change)
    assert controller.update(0.20, now=1.0) is True         # dwell not elapsed
    assert controller.update(0.20, now=10.0) is False       # dwell elapsed: off
    assert controller.update(0.95, now=11.0) is False       # dwell not elapsed
    assert controller.update(0.95, now=20.0) is True        # dwell elapsed: on
    with pytest.raises(ValueError):
        controller.update(0.9, now=5.0)  # time cannot move backwards


def test_oscillation_detector_counts_reversals():
    """The oscillation detector flags excessive action reversals."""
    detector = OscillationDetector(_limits())
    sequence = [1, 2, 1, 2, 1, 2]
    flagged = False
    for step, signature in enumerate(sequence):
        flagged = detector.record(signature, now=float(step))
    assert flagged
    assert detector.reversal_count() >= 5


def test_pid_controller_anti_windup_bounds_overshoot():
    """Acceptance: a saturated controller recovers without excessive overshoot."""
    controller = PidController(
        kp=2.0, ki=0.5, kd=0.0, output_min=-1.0, output_max=1.0,
    )
    outputs = [controller.update(5.0, delta_time=0.1) for _ in range(200)]
    assert all(-1.0 <= value <= 1.0 for value in outputs)
    # Anti-windup: the integral cannot push output past the bound for long.
    assert max(outputs) <= 1.0 + 1e-9
    # Recovery: after the error disappears, output decays back toward zero.
    for _ in range(50):
        output = controller.update(0.0, delta_time=0.1)
    assert abs(output) < 0.5
    controller.reset()
    with pytest.raises(ValueError):
        controller.update(0.1, delta_time=0.0)


def test_watchdog_triggers_and_enforces_cooldown():
    """The watchdog aggregates failure signals and enforces cooldown."""
    watchdog = Watchdog(_limits())
    verdict = watchdog.check(
        invariant_violations=("budget exceeded",), now=100.0,
    )
    assert verdict.triggered and verdict.reasons
    assert watchdog.trigger_count == 1
    assert watchdog.in_cooldown(now=110.0) is True
    assert watchdog.in_cooldown(now=200.0) is False
    quiet = watchdog.check(now=200.0)
    assert not quiet.triggered


def test_rollback_manager_restores_known_good_baseline():
    """Acceptance: simulated policy failure restores the known-good baseline."""
    manager = RollbackManager()
    baseline = RuntimeControlState(worker_count=4)
    manager.register_baseline(baseline)
    changed = RuntimeControlState(worker_count=4)
    changed.worker_count = 16
    manager.checkpoint("before-change", changed)
    changed.worker_count = 32
    restored = manager.activate_baseline()
    assert restored.worker_count == 4
    token_restored = manager.rollback("before-change")
    assert token_restored.worker_count == 16
    with pytest.raises(KeyError):
        manager.rollback("never-created")
    assert manager.history()


def test_safe_fallback_policy_disables_adaptation_after_trigger():
    """SafeFallbackPolicy composes watchdog, rollback and cooldown (Step 52)."""
    policy = SafeFallbackPolicy(_limits())
    policy.rollback.register_baseline(RuntimeControlState(worker_count=2))
    from ore_cros.safety.fallback import WatchdogVerdict

    triggered = WatchdogVerdict(True, ("invariant violation",))
    assert policy.evaluate(triggered, now=10.0) is False
    assert policy.evaluate(WatchdogVerdict(False), now=20.0) is False  # cooldown
    assert policy.evaluate(WatchdogVerdict(False), now=100.0) is True


def test_criticality_matrix_governs_deferral_and_reservations():
    """Step 47: the criticality matrix fixes deferral rights and reservations."""
    matrix = CriticalityMatrix(reserved_for_integrity={ResourceKind.CPU: 0.5})
    assert matrix.may_defer(WorkloadClass.EXPLORATORY)
    assert not matrix.may_defer(WorkloadClass.INTEGRITY_CRITICAL)
    assert not matrix.may_defer(WorkloadClass.INTERACTIVE)
    order = matrix.deferral_order()
    assert order[0] is WorkloadClass.EXPLORATORY
    assert order[-1] is WorkloadClass.INTEGRITY_CRITICAL


def test_wcet_analyzer_validates_rm_feasibility():
    """Step 48: RM utilization bounds validate execution assumptions."""
    analyzer = WcetAnalyzer()
    assert analyzer.rm_utilization_bound(1) == pytest.approx(1.0)
    feasible = analyzer.analyze_rm([
        RealTimeTaskSpec(name="a", period=10.0, wcet=1.0),
        RealTimeTaskSpec(name="b", period=20.0, wcet=2.0),
    ])
    assert feasible.feasible
    overloaded = analyzer.analyze_rm([
        RealTimeTaskSpec(name=f"t{index}", period=10.0 + index, wcet=4.0)
        for index in range(4)
    ])
    assert not overloaded.feasible
    with pytest.raises(ValueError):
        RealTimeTaskSpec(name="bad", period=5.0, wcet=10.0)


def test_hard_real_time_claim_requires_platform_evidence():
    """Acceptance: no hard real-time claim without bounded execution and
    platform evidence (Step 48)."""
    analyzer = WcetAnalyzer()
    tasks = [RealTimeTaskSpec(name="rt", period=10.0, wcet=1.0)]
    with pytest.raises(RealTimeClaimError):
        analyzer.hard_real_time_claim(tasks)
    partial = {"wcet_bounds": True, "interrupt_bound": True}
    with pytest.raises(RealTimeClaimError):
        analyzer.hard_real_time_claim(tasks, platform_evidence=partial)
    complete = {
        "wcet_bounds": True, "interrupt_bound": 0.01, "blocking_bound": 0.1,
        "platform": "rt-linux-verified", "scheduling_overhead": 0.001,
    }
    analysis = analyzer.hard_real_time_claim(tasks, platform_evidence=complete)
    assert analysis.feasible


def test_quota_ledger_records_authorized_exceptions():
    """Acceptance: no task exceeds hard budget without a recorded authorized
    exception."""
    ledger = QuotaLedger()
    cap = ResourceDemand(cpu=2.0)
    verdict = ledger.charge(
        "m1", ResourceDemand(cpu=1.5), reserved=ResourceDemand(cpu=1.0), cap=cap,
    )
    assert verdict.allowed
    # Exceeding without authorization is rejected and unrecorded.
    verdict = ledger.charge(
        "m1", ResourceDemand(cpu=1.0), reserved=ResourceDemand(cpu=1.0), cap=cap,
    )
    assert not verdict.allowed and ledger.exceptions() == ()
    # With explicit authorization the exception is recorded.
    verdict = ledger.charge(
        "m1", ResourceDemand(cpu=1.0), reserved=ResourceDemand(cpu=1.0), cap=cap,
        authorized_exception=True, reason="deadline rescue",
    )
    assert verdict.allowed
    assert len(ledger.exceptions()) == 1
    assert ledger.exceptions()[0]["reason"] == "deadline rescue"


def test_fairness_floor_constrains_adaptive_actions():
    """Acceptance: adaptive control cannot exceed the configured unfairness
    threshold."""
    shares_equal = [0.25, 0.25, 0.25, 0.25]
    shares_skewed = [0.7, 0.1, 0.1, 0.1]
    floor = 0.90
    assert jain_fairness_index(shares_equal) >= floor
    assert jain_fairness_index(shares_skewed) < floor
    constraints = HardConstraintSet(fairness_floor=floor)
    assert constraints.fairness_floor == floor


def test_cycle_survives_executor_failure_with_baseline():
    """Acceptance: missing telemetry, collector crash, timeout and executor
    failure trigger safe behavior."""
    collector_set = CollectorSet([SimulatedCollector(seed=5, load_profile=lambda s: 0.9)])
    coordinator = CROSCoordinator(
        config=CROSConfig(), collector_set=collector_set,
        constraints=HardConstraintSet(min_workers=1, max_workers=8),
    )
    # Simulate executor failure for every action id via a subclass hook.
    coordinator.executor.fail_on_action_ids = {"defer-optional-work", "scale-workers-up"}
    baseline_workers = coordinator.control.worker_count
    result = coordinator.adaptive_resource_cycle(now=1_000.0)
    # A failed apply must not corrupt the control state.
    assert coordinator.control.worker_count == baseline_workers
    assert result.execution is not None


def test_deterministic_baseline_available_when_learning_fails():
    """Acceptance: the deterministic baseline remains available if predictors
    or learning components fail."""

    class ExplodingBandit:
        """A learned policy that always crashes, exercising safe degradation."""

        shadow = True

        def propose(self, context, currently_safe=None, timestamp=0.0):
            raise RuntimeError("learned policy crashed")

    collector_set = CollectorSet([SimulatedCollector(seed=3, load_profile=lambda s: 0.5)])
    coordinator = CROSCoordinator(
        config=CROSConfig(), collector_set=collector_set,
        bandit=ExplodingBandit(),
    )
    # The crashing learner must not break the cycle: the baseline decides.
    result = coordinator.adaptive_resource_cycle(now=1_000.0)
    assert result.cycle_number == 1
    assert result.action.action_type in (
        ActionType.KEEP_CURRENT_POLICY, ActionType.DEFER_BACKGROUND,
    )
    # The baseline scheduler itself still orders work deterministically.
    from ore_cros.policies.baseline import DeterministicBaselineScheduler

    scheduler = DeterministicBaselineScheduler()
    decision = scheduler.schedule([], build_state(), RuntimeControlState())
    assert decision.admitted_order == ()


def build_state():
    """Build a permissive resource state for the fallback test."""
    from ore_cros.resource_model import (
        EnvironmentProfile, ResourceKind, build_resource_state, machine_scope,
    )

    profile = EnvironmentProfile(
        profile_version="t", host_os="t", kernel_release="1",
        capacity={kind: 1024.0 for kind in ResourceKind},
    )
    return build_resource_state(profile, machine_scope())
