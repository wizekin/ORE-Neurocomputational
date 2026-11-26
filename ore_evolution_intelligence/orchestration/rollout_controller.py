"""Canary rollout, live monitoring and rollback (section 7; section 9 safeguards).

Deployment regression is guarded by canary monitoring, version pinning and rollback
(acceptance tests 33-34): an APPROVED strategy enters CANARY on a bounded task subset,
its canary metrics are compared against the validation envelope (drift) and the
baseline (regression gate); any drift or regression triggers rollback to the pinned
baseline configuration, records the event and deactivates the strategy. Live
monitoring re-checks active strategies against the same envelope.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from ..domain.evaluation_record import EvaluationRecord
from ..domain.strategy import StrategyCandidate
from ..evaluation.evaluator import StrategyEvaluator
from ..evaluation.heldout_validation import DriftReport, ValidationEnvelope
from ..registry.strategy_registry import ValidatedStrategyRegistry
from ..verification.promotion_policy import CanaryPlan
from ..verification.regression_gate import RegressionDecision, RegressionGate
from .coordinator_adapter import CoordinatorAdapter, ControlledConfigurationInterface


@dataclass(frozen=True, slots=True)
class CanaryDecision:
    """Composite canary verdict consumed by the promotion policy (CANARY -> ACTIVE)."""

    approved: bool
    reasons: tuple[str, ...] = ()
    drift: DriftReport | None = None
    regression: RegressionDecision | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "approved": self.approved, "reasons": list(self.reasons),
            "drift": self.drift.as_dict() if self.drift else None,
            "regression": self.regression.as_dict() if self.regression else None,
        }


@dataclass(frozen=True, slots=True)
class CanaryOutcome:
    """Result of one canary rollout (steps: canary -> active or rollback)."""

    promoted_to_active: bool
    rolled_back: bool
    drift_report: DriftReport | None
    regression_decision: RegressionDecision | None
    canary_record: EvaluationRecord | None
    reasons: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "promoted_to_active": self.promoted_to_active,
            "rolled_back": self.rolled_back,
            "drift": self.drift_report.as_dict() if self.drift_report else None,
            "regression": self.regression_decision.as_dict() if self.regression_decision else None,
            "canary_candidate_id": self.canary_record.candidate_id if self.canary_record else None,
            "reasons": list(self.reasons),
        }


class RolloutController:
    """Canary rollout with drift monitoring and rollback (section 7)."""

    def __init__(
        self,
        evaluator: StrategyEvaluator,
        *,
        registry: ValidatedStrategyRegistry | None = None,
        adapter: CoordinatorAdapter | None = None,
        regression_gate: RegressionGate | None = None,
        event_recorder: Any | None = None,
        run_id: str = "run",
    ) -> None:
        self.evaluator = evaluator
        self.registry = registry
        self.adapter = adapter
        self.regression_gate = regression_gate or RegressionGate()
        self.event_recorder = event_recorder
        self.run_id = run_id

    def canary(
        self,
        candidate: StrategyCandidate,
        baseline: StrategyCandidate,
        canary_tasks: Sequence[Any],
        *,
        envelope: ValidationEnvelope,
        seeds: Sequence[int],
        budget: Any | None = None,
    ) -> tuple[StrategyCandidate, CanaryOutcome]:
        """Run one canary rollout; returns the (possibly rejected) candidate."""
        if self.event_recorder is not None:
            self.event_recorder.emit(
                "canary_started", self.run_id,
                {"candidate_id": candidate.candidate_id,
                 "canary_tasks": [t.task_id for t in canary_tasks]},
            )
        # Apply the approved strategy through the controlled interface.
        if self.adapter is not None:
            self.adapter.apply_approved(candidate)
        canary_record = self.evaluator.evaluate_on_tasks(
            candidate, canary_tasks, split="canary", seeds=seeds, budget=budget,
        )
        baseline_record = self.evaluator.evaluate_on_tasks(
            baseline, canary_tasks, split="canary", seeds=seeds, budget=budget,
        )
        regression = self.regression_gate.check(
            canary_record.metrics, baseline_record.metrics,
        )
        drift = envelope.check(canary_record.metrics)
        reasons: list[str] = []
        if not regression.approved:
            reasons.extend(regression.reasons)
        if drift.drifted:
            reasons.extend(drift.violations)
        decision = CanaryDecision(
            approved=regression.approved and not drift.drifted,
            reasons=tuple(reasons),
            drift=drift,
            regression=regression,
        )
        if self.event_recorder is not None:
            self.event_recorder.emit(
                "canary_evaluated", self.run_id,
                {"candidate_id": candidate.candidate_id,
                 "approved": decision.approved,
                 "drifted": drift.drifted},
            )
        if not decision.approved:
            rejected = self._rollback(candidate, reasons)
            return rejected, CanaryOutcome(
                promoted_to_active=False, rolled_back=True,
                drift_report=drift, regression_decision=regression,
                canary_record=canary_record, reasons=tuple(reasons),
            )
        return candidate, CanaryOutcome(
            promoted_to_active=True, rolled_back=False,
            drift_report=drift, regression_decision=regression,
            canary_record=canary_record, reasons=(),
        )

    def monitor_active(
        self,
        candidate: StrategyCandidate,
        live_metrics: Any,
        *,
        envelope: ValidationEnvelope,
    ) -> tuple[bool, DriftReport]:
        """Live monitoring of an ACTIVE strategy against the validation envelope."""
        drift = envelope.check(live_metrics)
        if drift.drifted:
            if self.event_recorder is not None:
                self.event_recorder.emit(
                    "drift_detected", self.run_id,
                    {"candidate_id": candidate.candidate_id,
                     "violations": list(drift.violations)},
                )
            self._rollback(candidate, list(drift.violations))
            return True, drift
        return False, drift

    def _rollback(self, candidate: StrategyCandidate, reasons: Sequence[str]) -> StrategyCandidate:
        """Restore the pinned baseline configuration (acceptance test 34)."""
        if self.adapter is not None:
            self.adapter.interface.restore_rollback_target()
        if self.event_recorder is not None:
            self.event_recorder.emit(
                "rollback_triggered", self.run_id,
                {"candidate_id": candidate.candidate_id,
                 "reasons": list(reasons)},
            )
        rejected = candidate.with_state("REJECTED")
        if self.registry is not None:
            self.registry.reject(rejected, "; ".join(reasons) or "canary regression",
                                 stage="canary")
        if self.event_recorder is not None:
            self.event_recorder.emit(
                "strategy_deactivated", self.run_id,
                {"candidate_id": candidate.candidate_id, "state": "REJECTED"},
            )
        return rejected


def canary_plan_from_tasks(canary_tasks: Sequence[Any]) -> CanaryPlan:
    """Build the canary plan for one rollout (APPROVED -> CANARY evidence)."""
    return CanaryPlan(canary_task_ids=tuple(task.task_id for task in canary_tasks))


__all__ = ["CanaryDecision", "CanaryOutcome", "RolloutController", "canary_plan_from_tasks"]
