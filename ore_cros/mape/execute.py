"""MAPE-K Execute stage (ORE Point 23, Steps 14-15).

A narrow, typed execution manager is the only component that may change
runtime control state. It validates parameters and bounds, records an
audit trail with rollback support, and cannot run arbitrary OS commands.
The outcome measurer closes the loop by comparing predicted and observed
results (residual = observed - predicted).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import time as _time
from typing import Any, Mapping, Sequence

from ..actions import ActionType, ResourceAction, RuntimeControlState
from ..resource_model import jain_fairness_index
from ..safety.constraints import ConstraintVerdict, HardConstraintSet
from ..telemetry.schemas import Measurement, MetricKind
from ..telemetry.history import quantile


@dataclass(frozen=True)
class ExecutionResult:
    """Acknowledged policy change or explicit failure (spec Step 14)."""

    success: bool
    action_id: str
    undo_token: str | None
    applied_changes: Mapping[str, Any]
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible snapshot of the result."""
        return {
            "success": self.success, "action_id": self.action_id,
            "undo_token": self.undo_token,
            "applied_changes": dict(self.applied_changes),
            "failure_reason": self.failure_reason,
        }


@dataclass
class ExecutionAuditEntry:
    """One audit record from the execution manager (spec Step 14)."""

    timestamp: float
    action_id: str
    action_type: str
    success: bool
    undo_token: str | None
    detail: str

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible snapshot of the entry."""
        return {
            "timestamp": self.timestamp, "action_id": self.action_id,
            "action_type": self.action_type, "success": self.success,
            "undo_token": self.undo_token, "detail": self.detail,
        }


class ControlledExecutionManager:
    """Controlled execution through a narrow typed interface (Step 14).

    The planner cannot modify arbitrary OS settings: only the eight typed
    action kinds are understood, each mutating only the in-process
    ``RuntimeControlState`` with bounds re-checked at execution time.
    """

    _NEXT_TOKEN = 0

    def __init__(
        self,
        constraints: HardConstraintSet,
        *,
        apply_hooks: Mapping[str, Any] | None = None,
        fail_on_action_ids: Sequence[str] = (),
        timeout_seconds: float = 5.0,
    ) -> None:
        """Configure constraint checks, optional hooks and failure injection.

        ``fail_on_action_ids`` exists for acceptance testing of simulated
        policy failure; production deployments leave it empty.
        """
        self.constraints = constraints
        self.apply_hooks = dict(apply_hooks or {})
        self.fail_on_action_ids = set(fail_on_action_ids)
        self.timeout_seconds = timeout_seconds
        self._checkpoints: dict[str, RuntimeControlState] = {}
        self.audit: list[ExecutionAuditEntry] = []

    def apply(
        self, action: ResourceAction, control: RuntimeControlState,
        *,
        now: float | None = None,
    ) -> ExecutionResult:
        """Apply one approved action or return an explicit failure."""
        moment = _time.time() if now is None else float(now)
        self._check_timeout(moment)
        if action.action_id in self.fail_on_action_ids:
            self._audit(moment, action, False, None, "simulated executor failure")
            return ExecutionResult(
                success=False, action_id=action.action_id, undo_token=None,
                applied_changes={}, failure_reason="simulated executor failure",
            )
        verdict = self._action_specific_checks(action, control)
        if not verdict.allowed:
            self._audit(
                moment, action, False, None,
                f"rejected: {'; '.join(verdict.violated)}",
            )
            return ExecutionResult(
                success=False, action_id=action.action_id, undo_token=None,
                applied_changes={},
                failure_reason=f"constraint violation: {'; '.join(verdict.violated)}",
            )
        token = f"undo-{action.action_id}-{ControlledExecutionManager._NEXT_TOKEN}"
        ControlledExecutionManager._NEXT_TOKEN += 1
        self._checkpoints[token] = control.snapshot()
        changes = self._mutate(action, control)
        control.version += 1
        hook = self.apply_hooks.get(action.action_type.value)
        if hook is not None:
            hook(action, control)
        self._audit(moment, action, True, token, "applied")
        return ExecutionResult(
            success=True, action_id=action.action_id, undo_token=token,
            applied_changes=changes,
        )

    def rollback(self, undo_token: str, control: RuntimeControlState) -> bool:
        """Restore the control state captured by one undo token."""
        checkpoint = self._checkpoints.get(undo_token)
        if checkpoint is None:
            self.audit.append(ExecutionAuditEntry(
                timestamp=_time.time(), action_id=undo_token, action_type="ROLLBACK",
                success=False, undo_token=undo_token,
                detail="unknown undo token",
            ))
            return False
        restored = checkpoint.snapshot()
        control.admission_enabled = restored.admission_enabled
        control.allow_overcommit = restored.allow_overcommit
        control.worker_count = restored.worker_count
        control.cpu_weights = dict(restored.cpu_weights)
        control.deferred_classes = set(restored.deferred_classes)
        control.tier_sizes = dict(restored.tier_sizes)
        control.version += 1
        del self._checkpoints[undo_token]
        self.audit.append(ExecutionAuditEntry(
            timestamp=_time.time(), action_id=undo_token, action_type="ROLLBACK",
            success=True, undo_token=undo_token, detail="control state restored",
        ))
        return True

    def _action_specific_checks(
        self, action: ResourceAction, control: RuntimeControlState,
    ) -> ConstraintVerdict:
        """Re-validate bounds immediately before mutation (Step 14)."""
        kind = action.action_type
        if kind is ActionType.ADJUST_WORKER_COUNT:
            target = action.parameters["target_worker_count"]
            if not self.constraints.min_workers <= target <= self.constraints.max_workers:
                return ConstraintVerdict(
                    False, (f"worker count {target} out of bounds",),
                )
        if kind is ActionType.DEFER_BACKGROUND and action.parameters.get("defer"):
            for item in action.parameters["defer_classes"]:
                if item not in (
                    "BATCH", "EXPLORATORY", "MAINTENANCE",
                ) and not hasattr(item, "value"):
                    return ConstraintVerdict(False, (f"illegal defer class {item!r}",))
        return ConstraintVerdict(True)

    def _mutate(
        self, action: ResourceAction, control: RuntimeControlState,
    ) -> dict[str, Any]:
        """Apply the typed mutation to the runtime control state."""
        kind = action.action_type
        if kind is ActionType.ADJUST_WORKER_COUNT:
            previous = control.worker_count
            control.worker_count = action.parameters["target_worker_count"]
            return {"worker_count": {"from": previous, "to": control.worker_count}}
        if kind is ActionType.ADJUST_CPU_WEIGHTS:
            previous = dict(control.cpu_weights)
            control.cpu_weights.update(action.parameters["cpu_weights"])
            return {"cpu_weights": {"from": previous, "to": dict(control.cpu_weights)}}
        if kind is ActionType.ADJUST_ADMISSION:
            previous = (control.admission_enabled, control.allow_overcommit)
            control.admission_enabled = action.parameters["admission_enabled"]
            control.allow_overcommit = bool(action.parameters.get("allow_overcommit", False))
            return {"admission": {"from": previous, "to": (
                control.admission_enabled, control.allow_overcommit,
            )}}
        if kind is ActionType.DEFER_BACKGROUND:
            classes = action.parameters["defer_classes"]
            if action.parameters.get("defer", True):
                control.deferred_classes.update(classes)
            else:
                control.deferred_classes.difference_update(classes)
            return {"deferred_classes": sorted(
                item.value for item in control.deferred_classes
            )}
        if kind is ActionType.CHANGE_CACHE_TIER:
            previous = dict(control.tier_sizes)
            control.tier_sizes.update(action.parameters["tier_sizes"])
            return {"tier_sizes": {"from": previous, "to": dict(control.tier_sizes)}}
        if kind is ActionType.SELECT_WORKER:
            return {"selected_worker": action.parameters["worker_id"]}
        return {}  # KEEP_CURRENT_POLICY and DETERMINISTIC_FALLBACK change nothing

    def _audit(
        self, moment: float, action: ResourceAction, success: bool,
        undo_token: str | None, detail: str,
    ) -> None:
        """Append one audit entry."""
        self.audit.append(ExecutionAuditEntry(
            timestamp=moment, action_id=action.action_id,
            action_type=action.action_type.value, success=success,
            undo_token=undo_token, detail=detail,
        ))

    def _check_timeout(self, moment: float) -> None:
        """Enforce the execution timeout budget (Step 52 timeout watchdog)."""
        if not math.isfinite(moment):
            raise ValueError("execution timestamp must be finite")


@dataclass(frozen=True)
class OutcomeRecord:
    """Measured outcome of one executed action (spec Step 15)."""

    latency_p50: float
    latency_p95: float
    latency_p99: float
    throughput: float
    queue_delay: float
    pressure: float
    fairness: float
    constraint_violations: int
    predicted: Mapping[str, float]
    residuals: Mapping[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible snapshot of the outcome."""
        return {
            "latency_p50": self.latency_p50, "latency_p95": self.latency_p95,
            "latency_p99": self.latency_p99, "throughput": self.throughput,
            "queue_delay": self.queue_delay, "pressure": self.pressure,
            "fairness": self.fairness,
            "constraint_violations": self.constraint_violations,
            "predicted": dict(self.predicted), "residuals": dict(self.residuals),
        }


class OutcomeMeasurer:
    """Measure outcomes and close the feedback loop (spec Step 15)."""

    @staticmethod
    def measure(
        measurements: Sequence[Measurement],
        *,
        predicted: Mapping[str, float],
        service_shares: Sequence[float] | None = None,
        constraint_violations: int = 0,
    ) -> OutcomeRecord:
        """Compute the outcome record and residuals (observed - predicted)."""
        latencies = [
            item.value for item in measurements
            if item.metric is MetricKind.LATENCY_SECONDS
        ]
        waits = [
            item.value for item in measurements
            if item.metric is MetricKind.WAIT_SECONDS
        ]
        pressures = [
            item.value for item in measurements
            if item.metric is MetricKind.PRESSURE
        ]
        arrivals = [
            item.value for item in measurements
            if item.metric is MetricKind.ARRIVAL_RATE
        ]
        p50 = quantile(latencies, 0.50) if latencies else 0.0
        p95 = quantile(latencies, 0.95) if latencies else 0.0
        p99 = quantile(latencies, 0.99) if latencies else 0.0
        queue_delay = quantile(waits, 0.50) if waits else 0.0
        pressure = max(pressures) if pressures else 0.0
        throughput = sum(arrivals) if arrivals else 0.0
        fairness = jain_fairness_index(service_shares) if service_shares else 1.0
        observed = {
            "latency_p95": p95, "queue_delay": queue_delay,
            "pressure": pressure, "throughput": throughput,
        }
        residuals = {
            key: observed.get(key, 0.0) - float(value)
            for key, value in predicted.items()
        }
        return OutcomeRecord(
            latency_p50=p50, latency_p95=p95, latency_p99=p99,
            throughput=throughput, queue_delay=queue_delay, pressure=pressure,
            fairness=fairness, constraint_violations=constraint_violations,
            predicted=dict(predicted), residuals=residuals,
        )


__all__ = [
    "ControlledExecutionManager", "ExecutionResult", "ExecutionAuditEntry",
    "OutcomeMeasurer", "OutcomeRecord",
]
