"""End-to-end CROS adaptive resource cycle (ORE Point 23, section 5).

Implements the ``ADAPTIVE_RESOURCE_CYCLE`` pseudocode of the specification:
discover, collect telemetry, assess quality, update the versioned
resource state, account, detect pressure, analyze, retrieve knowledge,
generate candidates, filter by hard constraints, keep the current safe
policy when nothing safe remains, select baseline-or-constrained,
estimate effects, validate, execute through the restricted interface,
measure outcomes, store provenance, update predictors and policy
evaluation, and roll back to the baseline on any watchdog failure.

The loop reacts to events (sustained pressure, queue growth, worker
failure, task bursts, hardware changes) and uses hysteresis and minimum
dwell times to avoid oscillation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
import time as _time
from typing import Any, Sequence

from .actions import (
    ActionType, ResourceAction, RuntimeControlState,
    deterministic_fallback_action, keep_current_policy_action,
)
from .config import CROSConfig
from .accounting.task_costs import TaskUsageAccountant
from .distributed.placement import FailureDetector
from .evaluation.benchmarks import (
    BenchmarkReport, ReplayBenchmark, TaskTraceEvent, compare_policies,
)
from .learning.bandit import ConstrainedContextualBandit
from .mape.analyze import Analyzer, Diagnosis, PressureDetector
from .mape.execute import ControlledExecutionManager, ExecutionResult, OutcomeMeasurer
from .mape.monitor import Monitor, ObservationSnapshot, TaskEventCollector
from .mape.plan import DecisionRecord, KnowledgeBase, Planner, select_baseline_or_constrained
from .policies.baseline import AdmissionController, DeterministicBaselineScheduler
from .policies.cpu import ShadowModeEvaluator
from .prediction.demand import CompletionTimePredictor, DemandPredictor
from .resource_model import (
    EnvironmentProfile, ResourceDimension, ResourceKind, ResourceState,
    TaskRecord, WorkloadClass, machine_scope,
)
from .safety.constraints import HardConstraintSet
from .safety.fallback import OscillationDetector, RollbackManager, Watchdog
from .telemetry.collectors import (
    CollectorSet, HostProfiler, default_machine_collectors,
)
from .telemetry.history import TimeWindowedResourceHistory
from .telemetry.quality import TelemetryQualityAssessor
from .telemetry.schemas import Measurement, MetricKind

#: Context feature names used by the constrained contextual bandit.
BANDIT_FEATURES: tuple[str, ...] = (
    "cpu_pressure", "ram_pressure", "io_pressure",
    "net_pressure", "queue_depth", "worker_load",
)


@dataclass(frozen=True)
class CycleResult:
    """Outcome of one adaptive resource cycle (spec section 5)."""

    state: ResourceState
    action: ResourceAction
    execution: ExecutionResult | None
    diagnosis: Diagnosis
    outcome: Any
    watchdog_triggered: bool
    watchdog_reasons: tuple[str, ...]
    cycle_number: int
    decision: DecisionRecord | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible snapshot of the cycle result."""
        return {
            "cycle_number": self.cycle_number,
            "state": self.state.to_dict(), "action": self.action.to_dict(),
            "execution": self.execution.to_dict() if self.execution else None,
            "diagnosis": self.diagnosis.to_dict(),
            "outcome": self.outcome.to_dict() if hasattr(self.outcome, "to_dict") else None,
            "watchdog_triggered": self.watchdog_triggered,
            "watchdog_reasons": list(self.watchdog_reasons),
            "decision": self.decision.to_dict() if self.decision else None,
        }


class CROSCoordinator:
    """The CROS adaptive resource coordinator (spec sections 5-6).

    CROS is the higher-level resource intelligence working with existing
    ORE components: it owns global resource policy, budgets and allocation
    through the restricted execution interface, and never bypasses the
    host operating system.
    """

    def __init__(
        self,
        *,
        config: CROSConfig | None = None,
        profile: EnvironmentProfile | None = None,
        collector_set: CollectorSet | None = None,
        constraints: HardConstraintSet | None = None,
        bandit: ConstrainedContextualBandit | None = None,
        seed: int = 7,
    ) -> None:
        """Assemble the MAPE-K loop, safety layers and learned policy."""
        self.config = config or CROSConfig()
        self.profile = profile or HostProfiler().discover()
        self.collector_set = collector_set or default_machine_collectors(self.profile)
        self.constraints = constraints or HardConstraintSet(
            min_workers=self.config.limits.min_worker_count,
            max_workers=self.config.limits.max_worker_count,
            fairness_floor=self.config.limits.fairness_floor,
        )
        self.scope = machine_scope()
        self.history = TimeWindowedResourceHistory(self.config.windows)
        self.assessor = TelemetryQualityAssessor(self.config.quality)
        self.monitor = Monitor(
            self.collector_set, self.assessor, self.history, config=self.config,
        )
        self.pressure_detector = PressureDetector()
        self.analyzer = Analyzer()
        self.planner = Planner(self.config)
        self.knowledge = KnowledgeBase()
        self.executor = ControlledExecutionManager(self.constraints)
        self.rollback = RollbackManager()
        self.watchdog = Watchdog(self.config.limits)
        self.oscillation = OscillationDetector(self.config.limits)
        self.accountant = TaskUsageAccountant()
        self.demand_predictor = DemandPredictor()
        self.completion_predictor = CompletionTimePredictor()
        self.baseline_scheduler = DeterministicBaselineScheduler(
            admission=AdmissionController(),
        )
        self.control = RuntimeControlState()
        self.rollback.register_baseline(self.control)
        self.bandit = bandit
        self.shadow_evaluator = ShadowModeEvaluator()
        self.failure_detector = FailureDetector()
        self._rng = random.Random(seed)
        self._cycle_count = 0
        self._state_version = 0
        self._last_cycle_time = -math.inf
        self._last_action_type: ActionType | None = None
        self._latest_measurements: list[Measurement] = []
        self._last_observed_metrics: dict[str, float] = {}
        self._bandit_failure_count = 0
        self.decisions: list[DecisionRecord] = []
        self._baseline_known_good = self.control.snapshot()

    # ------------------------------------------------------------------
    # Section 5 pseudocode, one method per step
    # ------------------------------------------------------------------

    def discover_environment(self) -> EnvironmentProfile:
        """Step 1 entry point: return the cached host and topology profile."""
        return self.profile

    def update_versioned_resource_state(
        self, snapshot: ObservationSnapshot,
    ) -> ResourceState:
        """Fold one observation snapshot into the versioned resource state."""
        self._state_version += 1
        utilizations: dict[ResourceKind, float] = {}
        allocations: dict[ResourceKind, float] = {}
        pressures: dict[ResourceKind, float] = {}
        for measurement in snapshot.measurements:
            if measurement.metric is MetricKind.UTILIZATION:
                utilizations[measurement.resource] = measurement.value
            elif measurement.metric is MetricKind.ALLOCATED:
                allocations[measurement.resource] = measurement.value
            elif measurement.metric is MetricKind.PRESSURE:
                pressures[measurement.resource] = measurement.value
        dimensions: dict[ResourceKind, ResourceDimension] = {}
        for kind in ResourceKind:
            capacity = self.profile.capacity.get(kind)
            if (
                kind in utilizations and capacity is not None and capacity > 0.0
                and kind not in allocations
            ):
                allocations[kind] = utilizations[kind] * capacity
            dimensions[kind] = ResourceDimension(
                kind=kind, capacity=capacity,
                allocated=allocations.get(kind, 0.0),
                reserved=self._reserved_for(kind),
                utilization=utilizations.get(kind),
                pressure=pressures.get(kind, 0.0),
                confidence=snapshot.quality.confidence(kind),
            )
        return ResourceState(
            version=self._state_version, timestamp=snapshot.timestamp,
            scope=self.scope, dimensions=dimensions,
        )

    def _reserved_for(self, kind: ResourceKind) -> float:
        """Return the integrity-critical reservation floor for one kind."""
        if kind is ResourceKind.CPU:
            return min(
                float(self.profile.capacity.get(ResourceKind.CPU) or 0.0) * 0.1, 0.5,
            )
        if kind is ResourceKind.RAM:
            return min(
                float(self.profile.capacity.get(ResourceKind.RAM) or 0.0) * 0.1, 256.0,
            )
        return 0.0

    def attribute_resource_usage(
        self, workloads: Sequence[TaskRecord], state: ResourceState,
        *, now: float,
    ) -> None:
        """Attribute measured usage to running tasks (spec Step 5)."""
        self.accountant.attribute_machine_measurements(
            _machine_measurements(self.monitor_history_measurements()),
            [task for task in workloads if task.state.value == "RUNNING"],
            now=now,
        )

    def monitor_history_measurements(self) -> Sequence[Measurement]:
        """Return the most recent observation's measurements."""
        return self._latest_measurements

    def adaptive_resource_cycle(
        self,
        tasks: Sequence[TaskRecord] = (),
        *,
        now: float | None = None,
        event: str | None = None,
    ) -> CycleResult:
        """Run one full ADAPTIVE_RESOURCE_CYCLE (spec section 5)."""
        moment = _time.time() if now is None else float(now)
        self._cycle_count += 1

        # Observation and resource accounting (Phase A).
        observations = self.monitor.collect(tasks, now=moment)
        self._latest_measurements = list(observations.measurements)
        quality = observations.quality
        state = self.update_versioned_resource_state(observations)
        workloads = TaskEventCollector.collect_ready_and_running(tasks)
        self.attribute_resource_usage(workloads, state, now=moment)
        pressure = self.pressure_detector.detect(
            observations.measurements, self.history,
            now=moment, window_seconds=self.config.windows.medium_window_seconds,
        )
        blocked = TaskEventCollector.blocked_tasks(tasks)
        diagnosis = self.analyzer.diagnose(
            observations, pressure, blocked_tasks=blocked,
        )

        # MAPE-K plan and knowledge (Phase B).
        knowledge = self.knowledge.retrieve(diagnosis)
        candidates = self.planner.generate(
            diagnosis, knowledge, current_worker_count=self.control.worker_count,
        )
        safe_actions, rejections = self.constraints.filter_safe_actions(
            candidates, state, self.control, quality=quality,
        )
        if not safe_actions:
            action = keep_current_policy_action(
                state.version, self.config.policy_version,
                rationale="no safe action available; keeping current safe policy",
            )
        else:
            bandit_proposal = self._bandit_proposal(
                diagnosis, safe_actions, moment,
            )
            action = select_baseline_or_constrained(
                safe_actions,
                canary_active=self._canary_active(),
                bandit_proposal=bandit_proposal,
            )

        # Estimate, validate and execute through the restricted interface.
        plan = self.planner.estimate(action)
        verdict = self.constraints.evaluate_action(
            plan, state, self.control, quality=quality,
        )
        if not verdict.allowed:
            action = deterministic_fallback_action(
                state.version, self.config.policy_version,
                rationale="plan failed hard-constraint validation: "
                          f"{'; '.join(verdict.violated)}",
            )
        result = self.executor.apply(action, self.control, now=moment)
        execution: ExecutionResult | None = result
        if not result.success:
            restored = self.rollback.activate_baseline()
            self._adopt_control_state(restored)
            execution = result

        # Measure outcomes and close the loop.
        had_prior_observation = bool(self._last_observed_metrics)
        predicted = self._predicted_metrics(action)
        outcome = OutcomeMeasurer.measure(
            observations.measurements,
            predicted=predicted,
            constraint_violations=0 if result.success else 1,
        )
        self._last_observed_metrics = {
            "latency_p95": outcome.latency_p95,
            "queue_delay": outcome.queue_delay,
            "pressure": outcome.pressure,
            "throughput": outcome.throughput,
        }
        decision = DecisionRecord(
            state_version=state.version, policy_version=self.config.policy_version,
            action_id=action.action_id, action_type=action.action_type.value,
            policy_origin=action.policy_origin.value, rationale=action.rationale,
            constraints_checked=tuple(verdict.violated) or ("all hard constraints",),
            predicted_outcome=dict(predicted),
            measured_outcome=outcome.to_dict(), timestamp=moment,
        )
        self.knowledge.record_decision(decision)
        self.decisions.append(decision)
        self._update_predictors(observations, moment)

        # Watchdog, rollback and baseline restoration (Step 52).
        oscillating = self.oscillation.record(
            self._action_signature(action), now=moment,
        )
        prediction_error = max(
            (abs(value) for value in outcome.residuals.values()), default=0.0,
        )
        watchdog_verdict = self.watchdog.check(
            invariant_violations=() if result.success else ("execution failed",),
            # Without a prior observation there is no calibrated prediction
            # to be wrong about; the error gate stays disabled on cycle one.
            prediction_error=(
                prediction_error if had_prior_observation and outcome.residuals else None
            ),
            oscillating=oscillating, now=moment,
        )
        if watchdog_verdict.triggered:
            restored = self.rollback.activate_baseline()
            self._adopt_control_state(restored)

        self._last_cycle_time = moment
        self._last_action_type = action.action_type
        return CycleResult(
            state=state, action=action, execution=execution,
            diagnosis=diagnosis, outcome=outcome,
            watchdog_triggered=watchdog_verdict.triggered,
            watchdog_reasons=watchdog_verdict.reasons,
            cycle_number=self._cycle_count, decision=decision,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _canary_active(self) -> bool:
        """Return True when this cycle belongs to the learning canary."""
        if self.bandit is None or self.bandit.shadow:
            return False
        fraction = self.config.limits.canary_fraction
        if fraction <= 0.0:
            return False
        if fraction >= 1.0:
            return True
        return self._rng.random() < fraction

    def _bandit_proposal(
        self, diagnosis: Diagnosis, safe_actions: Sequence[ResourceAction],
        now: float,
    ) -> str | None:
        """Return the learned proposal, kept in shadow mode when configured.

        A crashing learned component degrades safely: the proposal is
        skipped for this cycle and the deterministic baseline decides.
        """
        if self.bandit is None:
            return None
        context = self._bandit_context(diagnosis)
        safe_ids = [action.action_id for action in safe_actions]
        try:
            proposal_id, _propensity = self.bandit.propose(
                context, currently_safe=safe_ids, timestamp=now,
            )
        except Exception:  # noqa: BLE001 - learned components must fail safely
            self._bandit_failure_count += 1
            if self._bandit_failure_count >= 3:
                self.bandit = None  # permanently disable a broken learner
            return None
        if self.bandit.shadow:
            baseline = keep_current_policy_action(
                diagnosis.snapshot_version, self.config.policy_version,
            )
            self.shadow_evaluator.observe(
                _proposal_action(proposal_id, safe_actions), baseline,
                context={"now": now, **context},
            )
            return None
        return proposal_id

    def _bandit_context(self, diagnosis: Diagnosis) -> dict[str, float]:
        """Project the diagnosis into the fixed bandit feature vector."""
        queue = diagnosis.pressure.queue
        cpu_capacity = float(self.profile.capacity.get(ResourceKind.CPU) or 1.0) or 1.0
        return {
            "cpu_pressure": _pressure_of(diagnosis, ResourceKind.CPU),
            "ram_pressure": _pressure_of(diagnosis, ResourceKind.RAM),
            "io_pressure": _pressure_of(diagnosis, ResourceKind.IO),
            "net_pressure": _pressure_of(diagnosis, ResourceKind.NET),
            "queue_depth": (queue.depth / 64.0) if queue else 0.0,
            "worker_load": self.control.worker_count / max(1.0, cpu_capacity),
        }

    def _predicted_metrics(self, action: ResourceAction) -> dict[str, float]:
        """Return the metric predictions recorded for one action (Step 15).

        No-op actions (keep-current, deterministic fallback) predict
        persistence: the last observed values carried forward. Real
        adjustments shift latency by the predicted objective delta and
        queue delay by the action cost.
        """
        persistence = dict(self._last_observed_metrics) or {
            "latency_p95": 0.0, "queue_delay": 0.0,
            "pressure": 0.0, "throughput": 0.0,
        }
        if action.action_type in (
            ActionType.KEEP_CURRENT_POLICY, ActionType.DETERMINISTIC_FALLBACK,
        ):
            return persistence
        predicted = dict(persistence)
        predicted["latency_p95"] = max(
            0.0, persistence.get("latency_p95", 0.0)
            + action.predicted_objective_delta,
        )
        predicted["queue_delay"] = max(
            0.0, persistence.get("queue_delay", 0.0) - action.cost,
        )
        # Pressure and throughput carry forward unchanged: the action model
        # does not claim effects it cannot predict.
        return predicted

    def _update_predictors(self, snapshot: ObservationSnapshot, moment: float) -> None:
        """Feed observed values into the demand predictors (Step 49)."""
        for measurement in snapshot.measurements:
            if measurement.metric is not MetricKind.UTILIZATION:
                continue
            if measurement.resource not in (ResourceKind.CPU, ResourceKind.RAM):
                continue
            for workload_class in WorkloadClass:
                self.demand_predictor.observe(
                    measurement.resource, workload_class, measurement.value,
                )

    @staticmethod
    def _action_signature(action: ResourceAction) -> int:
        """Return a stable oscillation signature for one action."""
        return hash(action.action_type.value)

    def _adopt_control_state(self, restored: RuntimeControlState) -> None:
        """Adopt a restored control state (rollback path)."""
        self.control.admission_enabled = restored.admission_enabled
        self.control.allow_overcommit = restored.allow_overcommit
        self.control.worker_count = restored.worker_count
        self.control.cpu_weights = dict(restored.cpu_weights)
        self.control.deferred_classes = set(restored.deferred_classes)
        self.control.tier_sizes = dict(restored.tier_sizes)

    @property
    def cycle_count(self) -> int:
        """Return the number of cycles executed so far."""
        return self._cycle_count


def _pressure_of(diagnosis: Diagnosis, kind: ResourceKind) -> float:
    """Return the pressure reading for one resource kind."""
    reading = diagnosis.pressure.reading(kind)
    return reading.pressure if reading else 0.0


def _machine_measurements(
    measurements: Sequence[Measurement],
) -> list[Measurement]:
    """Filter measurements down to the machine scope."""
    return [
        item for item in measurements
        if item.scope.level.value == "MACHINE"
    ]


def _proposal_action(
    proposal_id: str, safe_actions: Sequence[ResourceAction],
) -> ResourceAction:
    """Return the safe action matching one bandit proposal id."""
    for action in safe_actions:
        if action.action_id == proposal_id:
            return action
    return keep_current_policy_action(0, "unknown")


def run_replay_benchmark(
    trace: Sequence[TaskTraceEvent], *,
    worker_count: int = 4,
    energy_available: bool = False,
) -> BenchmarkReport:
    """Run the deterministic replay benchmark for one trace (section 9)."""
    benchmark = ReplayBenchmark(
        DeterministicBaselineScheduler(), worker_count=worker_count,
        energy_available=energy_available,
    )
    return benchmark.run(trace)


def compare_against_baseline(
    baseline: BenchmarkReport, candidate: BenchmarkReport,
) -> Any:
    """Compare a candidate policy report against the baseline (section 8)."""
    return compare_policies(baseline, candidate)


__all__ = [
    "CROSCoordinator", "CycleResult", "BANDIT_FEATURES", "run_replay_benchmark",
    "compare_against_baseline",
]
