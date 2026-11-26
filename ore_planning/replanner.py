"""Outcome evaluation, revision-validated replanning, and safe termination rules.

The evaluator compares predicted and actual outcomes, recomputes preconditions
and goal progress after meaningful workspace revisions, applies the documented
replanning threshold, and enforces the stop conditions. Partial work is always
reported as explicit incomplete status with unresolved subgoals; an incomplete
plan can never be labeled successful.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .contracts import (
    ExecutionFeedback,
    GoalSpec,
    OperatorSpec,
    PlanNode,
    PlanNodeStatus,
    PlanRunResult,
    PlanRunStatus,
    PlanState,
)
from .cost import CostModel, GoalProgress
from .partial_order import PartialOrderPlan

REPLAN_COST_UNITS = 1.0
DISRUPTION_COST_UNITS = 0.5


@dataclass(frozen=True, slots=True)
class PredictionError:
    """Record the deviation between predicted and actual cost for one operator."""

    operator_id: str
    predicted: float
    actual: float
    deviation: float

    def to_dict(self) -> dict[str, Any]:
        """Return the prediction-error record."""
        return {
            "operator_id": self.operator_id,
            "predicted": self.predicted,
            "actual": self.actual,
            "deviation": self.deviation,
        }


@dataclass(frozen=True, slots=True)
class OutcomeEvaluation:
    """Summarize prediction errors and calibration guidance across feedback."""

    errors: tuple[PredictionError, ...]
    mean_absolute_error: float
    failed_operators: tuple[str, ...]
    succeeded_operators: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible evaluation record."""
        return {
            "errors": [item.to_dict() for item in self.errors],
            "mean_absolute_error": self.mean_absolute_error,
            "failed_operators": list(self.failed_operators),
            "succeeded_operators": list(self.succeeded_operators),
        }


@dataclass(frozen=True, slots=True)
class ReplanDecision:
    """Explain whether replanning is justified under the expected-benefit threshold."""

    should_replan: bool
    rationale: str
    expected_benefit: float
    replan_cost: float

    def to_dict(self) -> dict[str, Any]:
        """Return the replanning decision record."""
        return {
            "should_replan": self.should_replan,
            "rationale": self.rationale,
            "expected_benefit": self.expected_benefit,
            "replan_cost": self.replan_cost,
        }


class OutcomeEvaluator:
    """Evaluate execution outcomes and decide when replanning is justified."""

    def __init__(self, cost_model: CostModel) -> None:
        """Bind the evaluator to the shared cost model."""
        self.cost_model = cost_model

    def evaluate_outcomes(
        self, feedback: Sequence[ExecutionFeedback]
    ) -> OutcomeEvaluation:
        """Compare predicted and actual costs for every executed operator."""
        errors: list[PredictionError] = []
        failed: list[str] = []
        succeeded: list[str] = []
        for item in feedback:
            predicted = self.cost_model.vector_cost(item.predicted_cost)
            actual = self.cost_model.vector_cost(item.actual_cost)
            errors.append(PredictionError(
                operator_id=item.operator_id or item.task_id,
                predicted=predicted,
                actual=actual,
                deviation=actual - predicted,
            ))
            if item.success and item.operator_id:
                succeeded.append(item.operator_id)
            elif not item.success and item.operator_id:
                failed.append(item.operator_id)
        mae = (
            sum(abs(item.deviation) for item in errors) / len(errors) if errors else 0.0
        )
        return OutcomeEvaluation(
            errors=tuple(errors),
            mean_absolute_error=mae,
            failed_operators=tuple(sorted(set(failed))),
            succeeded_operators=tuple(sorted(set(succeeded))),
        )

    @staticmethod
    def should_replan(
        *,
        expected_quality_gain: float,
        expected_saved_cost: float,
        replan_cost: float = REPLAN_COST_UNITS,
        disruption_cost: float = DISRUPTION_COST_UNITS,
        failed_operators: Iterable[str] = (),
        invalidated_nodes: Iterable[str] = (),
    ) -> ReplanDecision:
        """Apply the replanning threshold E[dQ + dC_saved] > C_replan + C_disruption.

        All terms must already be expressed in one common cost scale; the defaults
        document the planner's unit conventions. A failed or invalidated operator
        always justifies replanning because the current plan can no longer reach
        its success conditions.
        """
        failed = tuple(failed_operators)
        invalidated = tuple(invalidated_nodes)
        expected_benefit = float(expected_quality_gain + expected_saved_cost)
        threshold = float(replan_cost + disruption_cost)
        if failed or invalidated:
            return ReplanDecision(
                should_replan=True,
                rationale=(
                    f"plan invalidated: failed={list(failed)}, invalidated={list(invalidated)}"
                ),
                expected_benefit=expected_benefit,
                replan_cost=threshold,
            )
        justified = expected_benefit > threshold
        return ReplanDecision(
            should_replan=justified,
            rationale=(
                f"expected benefit {expected_benefit:.3f} "
                f"{'exceeds' if justified else 'does not exceed'} "
                f"replanning threshold {threshold:.3f}"
            ),
            expected_benefit=expected_benefit,
            replan_cost=threshold,
        )

    def validate_preconditions(
        self,
        plan: PartialOrderPlan,
        operators: Mapping[str, OperatorSpec],
        state: PlanState,
    ) -> tuple[str, ...]:
        """Recompute node preconditions and return nodes invalidated by new facts."""
        invalidated: list[str] = []
        achieved = state.achieved_facts
        for node in plan.nodes:
            if node.status not in (
                PlanNodeStatus.PENDING, PlanNodeStatus.READY,
            ):
                continue
            for operator in operators.values():
                if operator.module_capability != node.task.task_type:
                    continue
                if not all(
                    precondition.satisfied_by(achieved)
                    for precondition in operator.preconditions
                ):
                    invalidated.append(node.node_id)
                    break
        return tuple(sorted(set(invalidated)))

    @staticmethod
    def revision_invalidated(
        plan: PartialOrderPlan, current_revision: int
    ) -> bool:
        """Check whether the workspace advanced beyond the plan's bound revision."""
        return current_revision > plan.workspace_revision

    @staticmethod
    def stop_conditions(
        goal: GoalSpec,
        progress: GoalProgress,
        *,
        budget_exhausted: bool = False,
        no_valid_plan: bool = False,
        human_decision_required: bool = False,
    ) -> str | None:
        """Return the first satisfied stop condition, or None to continue."""
        del goal
        if progress.critical_satisfied and progress.score >= 1.0:
            return "verified_success_conditions"
        if budget_exhausted:
            return "budget_or_deadline_reached"
        if no_valid_plan:
            return "no_valid_plan_exists"
        if human_decision_required:
            return "human_decision_required"
        return None

    @staticmethod
    def build_run_result(
        plan: PartialOrderPlan,
        progress: GoalProgress,
        termination_reason: str,
        *,
        feedback: Sequence[ExecutionFeedback] = (),
        replans: int = 0,
        workspace_revision: int = 0,
        evidence_refs: Iterable[str] = (),
    ) -> PlanRunResult:
        """Assemble the final run result with explicit partial status semantics."""
        complete = (
            progress.critical_satisfied and progress.score >= 1.0
            and termination_reason == "verified_success_conditions"
        )
        status = PlanRunStatus.COMPLETE if complete else PlanRunStatus.INCOMPLETE
        if termination_reason in {"budget_or_deadline_reached", "human_decision_required"}:
            status = PlanRunStatus.STOPPED
        return PlanRunResult(
            plan_id=plan.plan_id,
            status=status,
            goal_progress_score=progress.score,
            critical_satisfied=progress.critical_satisfied,
            unresolved_subgoals=tuple(progress.unsatisfied_conditions),
            termination_reason=termination_reason,
            feedback=tuple(feedback),
            replans=replans,
            workspace_revision=workspace_revision,
            evidence_refs=tuple(evidence_refs),
        )


def nodes_from_plan(plan: PartialOrderPlan) -> tuple[PlanNode, ...]:
    """Return the plan nodes for external inspection."""
    return plan.nodes


__all__ = [
    "DISRUPTION_COST_UNITS",
    "REPLAN_COST_UNITS",
    "OutcomeEvaluation",
    "OutcomeEvaluator",
    "PredictionError",
    "ReplanDecision",
]
