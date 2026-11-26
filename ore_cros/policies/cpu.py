"""CPU and module scheduling policies (ORE Point 23, Steps 17-26).

Implements proportional fairness with virtual-runtime accounting (Step 17),
EDF (Step 19), Rate Monotonic (Step 20), multilevel feedback with aging
(Step 21), gang scheduling (Step 22), module quotas and reservations
(Step 23), starvation guards (Step 24), shadow-mode evaluation (Step 25)
and constrained adaptation with fallback (Step 26).

CFS/EEVDF are design references (Step 18): nothing here claims to
reproduce kernel semantics; ``FairnessReport`` merely compares achieved
service against configured weights.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from ..actions import ActionType, ResourceAction
from ..config import AdaptationLimits
from ..resource_model import (
    ModuleBudget, ResourceDemand, ResourceKind, TaskRecord, jain_fairness_index,
)
from ..safety.constraints import WcetAnalyzer


class WeightedFairScheduler:
    """Proportional fairness via virtual-runtime accounting (Steps 17-18).

    ``share_i = weight_i / sum(weights)``; each accounted service quantum
    increases the task's virtual runtime by ``service / weight``, so heavier
    weights slow the growth of virtual runtime and receive proportionally
    more CPU. Service received and service owed are tracked explicitly.
    """

    def __init__(self) -> None:
        """Initialize empty accounting tables."""
        self._weights: dict[str, float] = {}
        self._virtual_runtime: dict[str, float] = {}
        self._service_received: dict[str, float] = {}
        self._service_owed: dict[str, float] = {}

    def register(self, task_id: str, weight: float) -> None:
        """Register one runnable task with a positive weight."""
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("task_id must be a non-empty string")
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError("weight must be finite and positive")
        self._weights[task_id] = weight
        self._virtual_runtime.setdefault(task_id, 0.0)
        self._service_received.setdefault(task_id, 0.0)
        self._service_owed.setdefault(task_id, 0.0)

    def unregister(self, task_id: str) -> None:
        """Remove one task from fair-share accounting."""
        for table in (
            self._weights, self._virtual_runtime,
            self._service_received, self._service_owed,
        ):
            table.pop(task_id, None)

    def shares(self) -> dict[str, float]:
        """Return the ideal proportional share per registered task."""
        total = sum(self._weights.values())
        if total <= 0.0:
            return {}
        return {task: weight / total for task, weight in self._weights.items()}

    def pick_next(self) -> str | None:
        """Return the task with the smallest virtual runtime (weighted).

        Virtual runtime already grows as ``service / weight``, so the
        plain minimum virtual runtime yields proportional service (spec 3.3).
        """
        if not self._weights:
            return None
        return min(
            self._weights,
            key=lambda task: (self._virtual_runtime[task], task),
        )

    def account(self, task_id: str, cpu_seconds: float) -> None:
        """Account consumed CPU time for one task."""
        if task_id not in self._weights:
            raise KeyError(f"task {task_id!r} is not registered")
        if not math.isfinite(cpu_seconds) or cpu_seconds < 0.0:
            raise ValueError("cpu_seconds must be finite and non-negative")
        weight = self._weights[task_id]
        self._virtual_runtime[task_id] += cpu_seconds / weight
        self._service_received[task_id] += cpu_seconds

    def service_tables(self) -> tuple[dict[str, float], dict[str, float]]:
        """Return (service_received, service_owed) accounting tables."""
        return dict(self._service_received), dict(self._service_owed)

    def achieved_shares(self) -> dict[str, float]:
        """Return the achieved service share per task."""
        total = sum(self._service_received.values())
        if total <= 0.0:
            return {task: 0.0 for task in self._weights}
        return {
            task: received / total
            for task, received in self._service_received.items()
        }

    def fairness_report(self) -> "FairnessReport":
        """Compare achieved service against configured weights (Step 18)."""
        ideal = self.shares()
        achieved = self.achieved_shares()
        tasks = sorted(ideal)
        index = (
            jain_fairness_index([achieved[task] for task in tasks])
            if tasks else 1.0
        )
        max_deviation = max(
            (abs(achieved[task] - ideal[task]) for task in tasks), default=0.0,
        )
        return FairnessReport(
            ideal_shares=ideal, achieved_shares=achieved,
            jain_index=index, max_deviation=max_deviation,
        )


@dataclass(frozen=True)
class FairnessReport:
    """Comparison data between ideal and achieved service (Step 18)."""

    ideal_shares: Mapping[str, float]
    achieved_shares: Mapping[str, float]
    jain_index: float
    max_deviation: float

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible snapshot of the report."""
        return {
            "ideal_shares": dict(self.ideal_shares),
            "achieved_shares": dict(self.achieved_shares),
            "jain_index": self.jain_index,
            "max_deviation": self.max_deviation,
        }


class EarliestDeadlineFirstScheduler:
    """Deadline-aware ordering with feasibility checks (Step 19)."""

    @staticmethod
    def order(tasks: Sequence[TaskRecord], *, now: float) -> tuple[str, ...]:
        """Order eligible jobs by increasing absolute deadline (ties by id)."""
        eligible = [task for task in tasks if task.deadline is not None]
        return tuple(
            task.task_id for task in sorted(
                eligible, key=lambda item: (item.deadline, item.task_id),
            )
        )

    @staticmethod
    def slack(task: TaskRecord, *, now: float) -> float | None:
        """Return slack: deadline minus now minus remaining work (spec 3.3)."""
        return task.slack(now)

    @staticmethod
    def feasible(
        tasks: Sequence[TaskRecord], *, now: float, cpu_capacity: float,
    ) -> tuple[bool, str]:
        """Check demand-bounded feasibility for the deadline set (Step 19)."""
        eligible = [task for task in tasks if task.deadline is not None]
        for task in eligible:
            available_time = task.deadline - now
            if available_time <= 0:
                return False, f"task {task.task_id} deadline already passed"
            if task.remaining_work > available_time * cpu_capacity + 1e-9:
                return (
                    False,
                    f"task {task.task_id} needs {task.remaining_work:.3f} cpu-seconds "
                    f"but only {available_time * cpu_capacity:.3f} remain before deadline",
                )
        total_demand = sum(task.remaining_work for task in eligible)
        horizon = max((task.deadline - now for task in eligible), default=0.0)
        if horizon > 0 and total_demand > horizon * cpu_capacity + 1e-9:
            return False, "aggregate demand exceeds capacity over the deadline horizon"
        return True, "feasible"


class RateMonotonicScheduler:
    """Fixed-priority Rate Monotonic scheduling for periodic tasks (Step 20).

    Suitable only for periodic monitoring/control jobs with known periods
    and bounded execution times; feasibility uses the Liu-Layland bound.
    """

    def __init__(self, analyzer: WcetAnalyzer | None = None) -> None:
        """Configure the WCET analyzer used for feasibility tests."""
        self.analyzer = analyzer or WcetAnalyzer()

    @staticmethod
    def priorities(periods: Mapping[str, float]) -> dict[str, int]:
        """Return static priorities: shorter period means higher priority."""
        if not periods:
            return {}
        for name, period in periods.items():
            if not math.isfinite(period) or period <= 0.0:
                raise ValueError(f"period for {name!r} must be finite and positive")
        order = sorted(periods, key=lambda name: (periods[name], name))
        return {name: index + 1 for index, name in enumerate(order)}

    def feasible(
        self, periods: Mapping[str, float], wcets: Mapping[str, float],
        *, scheduling_overhead: float = 0.0,
    ) -> tuple[bool, str]:
        """Validate configured periods and execution assumptions (Step 20)."""
        for name, period in periods.items():
            if name not in wcets:
                return False, f"missing WCET for periodic task {name!r}"
            if wcets[name] > period:
                return False, f"task {name!r} WCET exceeds its period"
        from ..safety.constraints import RealTimeTaskSpec

        specs = [
            RealTimeTaskSpec(name=name, period=periods[name], wcet=wcets[name])
            for name in sorted(periods)
        ]
        analysis = self.analyzer.analyze_rm(specs, scheduling_overhead=scheduling_overhead)
        return analysis.feasible, "; ".join(analysis.notes)


class MultilevelFeedbackQueue:
    """Multilevel feedback with aging for mixed workloads (Step 21).

    Tasks start at the top level and are demoted when they exhaust their
    quantum; aging promotes long-waiting tasks so demotion is never
    indefinite.
    """

    def __init__(
        self,
        *,
        levels: int = 3,
        quanta: Sequence[float] = (1.0, 2.0, 4.0),
        aging_threshold: float = 10.0,
    ) -> None:
        """Configure levels, per-level quanta and the aging threshold."""
        if levels <= 0 or len(quanta) != levels:
            raise ValueError("quanta must provide one quantum per level")
        if any(not math.isfinite(q) or q <= 0.0 for q in quanta):
            raise ValueError("quanta must be finite and positive")
        if not math.isfinite(aging_threshold) or aging_threshold <= 0.0:
            raise ValueError("aging_threshold must be finite and positive")
        self.quanta = tuple(float(q) for q in quanta)
        self.aging_threshold = float(aging_threshold)
        self._entries: dict[str, list[Any]] = {}
        self._entered_at: dict[str, float] = {}

    def submit(self, task_id: str, *, now: float, level: int = 0) -> None:
        """Enqueue one task at the given level (default: top level)."""
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("task_id must be a non-empty string")
        if not 0 <= level < len(self.quanta):
            raise ValueError(f"level must be within [0, {len(self.quanta) - 1}]")
        if task_id in self._entries:
            raise ValueError(f"task {task_id!r} already queued")
        self._entries[task_id] = [level, 0.0, 0.0]  # level, consumed_at_level, waited_since
        self._entered_at[task_id] = now

    def pick_next(self, *, now: float) -> str | None:
        """Return the next task to run, applying aging before selection."""
        if not self._entries:
            return None
        self._apply_aging(now)
        top_level = min(entry[0] for entry in self._entries.values())
        candidates = [
            task for task, entry in self._entries.items() if entry[0] == top_level
        ]
        return min(
            candidates,
            key=lambda task: (self._entered_at[task], task),
        )

    def _apply_aging(self, now: float) -> None:
        """Promote tasks whose total wait exceeds the aging threshold."""
        for task, entry in self._entries.items():
            waited = now - self._entered_at[task]
            if waited >= self.aging_threshold and entry[0] > 0:
                entry[0] -= 1
                entry[1] = 0.0
                self._entered_at[task] = now

    def account(self, task_id: str, cpu_seconds: float) -> None:
        """Account consumed CPU; demote when the level quantum is exhausted."""
        if task_id not in self._entries:
            raise KeyError(f"task {task_id!r} is not queued")
        if cpu_seconds < 0.0 or not math.isfinite(cpu_seconds):
            raise ValueError("cpu_seconds must be finite and non-negative")
        entry = self._entries[task_id]
        entry[1] += cpu_seconds
        if entry[1] >= self.quanta[entry[0]] and entry[0] < len(self.quanta) - 1:
            entry[0] += 1
            entry[1] = 0.0

    def level_of(self, task_id: str) -> int:
        """Return the current level of one queued task."""
        return self._entries[task_id][0]

    def complete(self, task_id: str) -> None:
        """Remove one finished task from the queue."""
        self._entries.pop(task_id, None)
        self._entered_at.pop(task_id, None)


class GangScheduler:
    """Coordinated gang execution for explicit parallel groups (Step 22).

    A gang becomes dispatchable only when every member is ready; the whole
    group is dispatched together or not at all.
    """

    def __init__(self) -> None:
        """Initialize empty gang registries."""
        self._gangs: dict[str, set[str]] = {}
        self._ready: set[str] = set()

    def declare_gang(self, gang_id: str, members: Sequence[str]) -> None:
        """Declare one gang and its member task ids."""
        if not isinstance(gang_id, str) or not gang_id.strip():
            raise ValueError("gang_id must be a non-empty string")
        if not members:
            raise ValueError("a gang requires at least one member")
        if len(set(members)) != len(members):
            raise ValueError("gang members must be unique")
        self._gangs[gang_id] = set(members)

    def mark_ready(self, task_id: str) -> None:
        """Mark one task as ready for gang dispatch."""
        self._ready.add(task_id)

    def dispatchable_gangs(self) -> tuple[str, ...]:
        """Return gang ids whose members are all ready, in declaration order."""
        return tuple(
            gang_id for gang_id, members in self._gangs.items()
            if members and members <= self._ready
        )

    def dispatch(self) -> dict[str, tuple[str, ...]]:
        """Return the full member sets of every dispatchable gang."""
        return {
            gang_id: tuple(sorted(self._gangs[gang_id]))
            for gang_id in self.dispatchable_gangs()
        }


class QuotaManager:
    """Module quotas and reservations with explicit borrowing (Step 23)."""

    def __init__(self, budgets: Mapping[str, ModuleBudget] | None = None) -> None:
        """Register per-module budgets."""
        self.budgets: dict[str, ModuleBudget] = dict(budgets or {})
        self._borrow_events: list[dict[str, Any]] = []

    def register(self, budget: ModuleBudget) -> None:
        """Add or replace one module budget."""
        self.budgets[budget.module_id] = budget

    def admit(self, module_id: str, demand: ResourceDemand) -> tuple[bool, str]:
        """Check one demand against the module's reserved+cap budget."""
        budget = self.budgets.get(module_id)
        if budget is None:
            return True, "no budget registered; defaulting to allow"
        for kind, field_name in (
            (ResourceKind.CPU, "cpu"), (ResourceKind.RAM, "ram"),
            (ResourceKind.IO, "io"), (ResourceKind.NET, "net"),
            (ResourceKind.GPU, "gpu"), (ResourceKind.STORAGE, "storage"),
        ):
            if demand.get(kind) > budget.cap.get(kind) + 1e-9:
                return (
                    False,
                    f"{module_id} demand {demand.get(kind):.3f} {field_name} exceeds "
                    f"cap {budget.cap.get(kind):.3f}",
                )
        return True, "within cap"

    def borrow(
        self, from_module: str, to_module: str, amount: ResourceDemand, *,
        authorized: bool,
    ) -> tuple[bool, str]:
        """Attempt to lend unused reservation from one module to another.

        Borrowing is only legal when the lending budget is marked borrowable
        and the transfer is explicitly authorized; every event is recorded.
        """
        lender = self.budgets.get(from_module)
        if lender is None:
            return False, f"unknown lending module {from_module!r}"
        if not lender.borrowable:
            return False, f"{from_module} reservation is not borrowable"
        if not authorized:
            return False, "borrowing requires explicit authorization"
        self._borrow_events.append({
            "from": from_module, "to": to_module,
            "amount": amount.to_dict(), "authorized": True,
        })
        return True, "borrowing recorded"

    def borrow_events(self) -> tuple[dict[str, Any], ...]:
        """Return the recorded borrowing events."""
        return tuple(dict(item) for item in self._borrow_events)


class StarvationGuard:
    """Aging and bounded-wait monitoring by priority class (Step 24)."""

    def __init__(self, *, age_gain: float = 0.1, wait_alert_threshold: float = 30.0) -> None:
        """Configure the aging gain and the bounded-wait alert threshold."""
        if age_gain <= 0.0 or wait_alert_threshold <= 0.0:
            raise ValueError("age_gain and wait_alert_threshold must be positive")
        self.age_gain = age_gain
        self.wait_alert_threshold = wait_alert_threshold

    def effective_priority(self, task: TaskRecord, *, now: float) -> float:
        """Return base criticality priority increased by waiting time."""
        waited = max(0.0, now - task.ready_time)
        return float(task.criticality.value) - self.age_gain * waited

    def starved_tasks(self, tasks: Sequence[TaskRecord], *, now: float) -> tuple[str, ...]:
        """Return ids of tasks whose wait exceeds the alert threshold."""
        return tuple(
            task.task_id for task in sorted(tasks, key=lambda item: item.task_id)
            if (now - task.ready_time) > self.wait_alert_threshold
        )

    def wait_report(self, tasks: Sequence[TaskRecord], *, now: float) -> dict[str, list[float]]:
        """Return waiting times grouped by criticality class."""
        report: dict[str, list[float]] = {}
        for task in tasks:
            report.setdefault(task.criticality.name, []).append(max(0.0, now - task.ready_time))
        return report


@dataclass(frozen=True)
class ShadowEvaluationRecord:
    """One shadow-mode observation: proposal vs baseline (Step 25)."""

    timestamp: float
    state_version: int
    proposed_action: str
    baseline_action: str
    proposed_predicted_objective: float
    baseline_predicted_objective: float
    context: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible snapshot of the record."""
        return {
            "timestamp": self.timestamp, "state_version": self.state_version,
            "proposed_action": self.proposed_action,
            "baseline_action": self.baseline_action,
            "proposed_predicted_objective": self.proposed_predicted_objective,
            "baseline_predicted_objective": self.baseline_predicted_objective,
            "context": dict(self.context),
        }


class ShadowModeEvaluator:
    """Non-invasive evaluation of learned policies (Step 25).

    The learned policy observes state and proposes actions without
    controlling execution; choices are recorded next to the deterministic
    scheduler's decisions for offline evaluation.
    """

    def __init__(self) -> None:
        """Initialize the empty evaluation dataset."""
        self._records: list[ShadowEvaluationRecord] = []
        self.applied_actions: list[str] = []  # stays empty in shadow mode

    def observe(
        self, proposal: ResourceAction, baseline: ResourceAction, *,
        context: Mapping[str, Any] | None = None,
    ) -> ShadowEvaluationRecord:
        """Record one proposal-baseline pair; nothing is executed."""
        record = ShadowEvaluationRecord(
            timestamp=float(context.get("now", 0.0)) if context else 0.0,
            state_version=proposal.state_version,
            proposed_action=proposal.action_type.value,
            baseline_action=baseline.action_type.value,
            proposed_predicted_objective=proposal.predicted_objective_delta,
            baseline_predicted_objective=baseline.predicted_objective_delta,
            context=dict(context or {}),
        )
        self._records.append(record)
        return record

    def dataset(self) -> tuple[ShadowEvaluationRecord, ...]:
        """Return the collected offline evaluation dataset."""
        return tuple(self._records)

    def agreement_rate(self) -> float:
        """Return the fraction of proposals matching the baseline decision."""
        if not self._records:
            return 0.0
        agreeing = sum(
            1 for record in self._records
            if record.proposed_action == record.baseline_action
        )
        return agreeing / len(self._records)


#: Reversible action types permitted for constrained adaptation (Step 26).
REVERSIBLE_ACTIONS: frozenset[ActionType] = frozenset({
    ActionType.ADJUST_WORKER_COUNT,
    ActionType.ADJUST_CPU_WEIGHTS,
    ActionType.DEFER_BACKGROUND,
})


class ConstrainedAdaptation:
    """Bounded adaptive scheduling with deterministic fallback (Step 26).

    After shadow evaluation, a limited set of reversible actions may act,
    under strict budgets, a canary fraction and watchdog supervision. Any
    constraint, prediction-error, tail-latency or stability failure
    switches control back to the baseline.
    """

    def __init__(self, limits: AdaptationLimits) -> None:
        """Configure the adaptation limits governing reversibility."""
        self.limits = limits

    def decide(
        self,
        proposal: ResourceAction,
        *,
        canary_active: bool,
        watchdog_healthy: bool,
        prediction_error: float | None = None,
        baseline_action: ResourceAction | None = None,
    ) -> ResourceAction:
        """Return the action to execute: proposal or baseline fallback.

        Only reversible action types may act, and only while the canary is
        active, the watchdog is healthy and prediction error stays bounded;
        every other path returns the baseline (or the proposal itself when
        no baseline was supplied).
        """
        fallback = baseline_action if baseline_action is not None else proposal
        if proposal.action_type not in REVERSIBLE_ACTIONS:
            return fallback
        if not canary_active or not watchdog_healthy:
            return fallback
        if (
            prediction_error is not None
            and prediction_error > self.limits.prediction_error_threshold
        ):
            return fallback
        return proposal


__all__ = [
    "WeightedFairScheduler", "FairnessReport", "EarliestDeadlineFirstScheduler",
    "RateMonotonicScheduler", "MultilevelFeedbackQueue", "GangScheduler",
    "QuotaManager", "StarvationGuard", "ShadowModeEvaluator",
    "ShadowEvaluationRecord", "ConstrainedAdaptation", "REVERSIBLE_ACTIONS",
]
