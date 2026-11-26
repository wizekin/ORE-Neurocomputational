"""Bounded-horizon model-predictive planning over existing feasible candidates.

The planner rolls the statistical transition model forward under strict horizon,
branching, and runtime budgets, filters sequences through hard constraints evaluated
independently of the objective, and selects a safe feasible sequence whose first
action is the only one ever dispatched.
"""
from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .contracts import (
    MISSING,
    CognitiveOperation,
    CognitiveStateFeatures,
    CWMBudget,
    CWMConfig,
    FallbackDecision,
    FallbackTrigger,
    OperationContext,
    OperationType,
    PredictionRecord,
    RetryPolicy,
    SafetyClass,
    _new_id,
    _utc_now,
)
from .policy import (
    ConfidenceGate,
    FallbackPolicy,
    HardConstraintSet,
)
from .predictor import PredictedTransition, StatisticalTransitionModel, classify_operation

VERIFY_MODULE_ID = "cwm_planner"
VERIFY_CAPABILITY = "cognition.world_model.verification"
STOP_CAPABILITY = "cognition.world_model.stop"


@dataclass(frozen=True, slots=True)
class MPCObjectiveWeights:
    """Documented MPC objective weights tuned on historical replay, not hand optimism.

    The score is minimized: resource cost, risk, uncertainty, delay, and terminal goal
    shortfall add penalty while expected information and goal benefit subtract from it.
    A conservative uncertainty margin is subtracted from benefits and added to costs
    before scoring, and hard constraints are filtered separately, never as penalties.
    """

    cost_weight: float = 1.0
    benefit_weight: float = 2.0
    risk_weight: float = 2.0
    uncertainty_weight: float = 1.0
    delay_weight: float = 0.25
    terminal_shortfall_weight: float = 2.0
    uncertainty_margin: float = 0.5

    def __post_init__(self) -> None:
        """Validate the documented weights."""
        for name in ("cost_weight", "benefit_weight", "risk_weight", "uncertainty_weight",
                     "delay_weight", "terminal_shortfall_weight", "uncertainty_margin"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class PredictedStep:
    """Hold one rollout step with its transition and cumulative aggregates."""

    step_index: int
    transition: PredictedTransition
    cumulative_cost: float
    cumulative_duration_ms: float
    cumulative_benefit: float
    cumulative_risk: float
    cumulative_uncertainty: float

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible step record."""
        return {
            "step_index": self.step_index,
            "operation_id": self.transition.operation.operation_id,
            "module_id": self.transition.operation.module_id,
            "operation_class": self.transition.operation_class,
            "cumulative_cost": self.cumulative_cost,
            "cumulative_duration_ms": self.cumulative_duration_ms,
            "cumulative_benefit": self.cumulative_benefit,
            "cumulative_risk": self.cumulative_risk,
            "cumulative_uncertainty": self.cumulative_uncertainty,
        }


@dataclass(frozen=True, slots=True)
class PredictedSequence:
    """Represent one bounded rollout with aggregates, verdict, and score."""

    sequence_id: str = field(default_factory=lambda: _new_id("cwm_sequence"))
    steps: tuple[PredictedStep, ...] = ()
    total_cost: float = 0.0
    total_duration_ms: float = 0.0
    total_benefit: float = 0.0
    total_risk: float = 0.0
    total_uncertainty: float = 0.0
    terminal_goal_score: float = 0.0
    produced_fields: tuple[str, ...] = ()
    module_ids: tuple[str, ...] = ()
    includes_verification: bool = False
    includes_stop: bool = False
    feasible: bool = True
    violations: tuple[tuple[str, str], ...] = ()
    score: float | None = None
    explanation: str = ""

    def __post_init__(self) -> None:
        """Normalize aggregate tuples."""
        object.__setattr__(self, "steps", tuple(self.steps))
        object.__setattr__(self, "produced_fields", tuple(self.produced_fields))
        object.__setattr__(self, "module_ids", tuple(self.module_ids))
        object.__setattr__(self, "violations", tuple(self.violations))

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible sequence record."""
        return {
            "sequence_id": self.sequence_id,
            "steps": [item.to_dict() for item in self.steps],
            "total_cost": self.total_cost,
            "total_duration_ms": self.total_duration_ms,
            "total_benefit": self.total_benefit,
            "total_risk": self.total_risk,
            "total_uncertainty": self.total_uncertainty,
            "terminal_goal_score": self.terminal_goal_score,
            "produced_fields": list(self.produced_fields),
            "module_ids": list(self.module_ids),
            "includes_verification": self.includes_verification,
            "includes_stop": self.includes_stop,
            "feasible": self.feasible,
            "violations": [{"constraint": name, "detail": detail}
                          for name, detail in self.violations],
            "score": self.score,
            "explanation": self.explanation,
        }


@dataclass(frozen=True, slots=True)
class CWMPlan:
    """One planning outcome: ranked feasible sequences, the selected first action, or fallback."""

    workspace_revision: int
    goal: str
    horizon: int
    ranked_sequences: tuple[PredictedSequence, ...] = ()
    rejected_sequences: tuple[PredictedSequence, ...] = ()
    selected: PredictedSequence | None = None
    first_action: CognitiveOperation | None = None
    predictions: tuple[PredictionRecord, ...] = ()
    fallback: FallbackDecision | None = None
    stopped: bool = False
    explanation: str = ""
    created_at: str = field(default_factory=_utc_now)

    @property
    def deferred_to_coordinator_defaults(self) -> bool:
        """Report whether this plan defers execution to existing deterministic policies."""
        return self.fallback is not None

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible plan record."""
        return {
            "workspace_revision": self.workspace_revision,
            "goal": self.goal,
            "horizon": self.horizon,
            "ranked_sequences": [item.to_dict() for item in self.ranked_sequences],
            "rejected_sequences": [item.to_dict() for item in self.rejected_sequences],
            "selected_sequence_id": (
                self.selected.sequence_id if self.selected is not None else None),
            "first_action": (
                self.first_action.to_dict() if self.first_action is not None else None),
            "predictions": [item.to_dict() for item in self.predictions],
            "fallback": self.fallback.to_dict() if self.fallback is not None else None,
            "stopped": self.stopped,
            "explanation": self.explanation,
            "created_at": self.created_at,
        }


def verification_action() -> CognitiveOperation:
    """Build the planner-internal verification control action (never dispatched)."""
    return CognitiveOperation(
        operation_type=OperationType.VERIFY,
        capability_id=VERIFY_CAPABILITY,
        module_id=VERIFY_MODULE_ID,
        output_schema=("cwm.verification.report",),
        retry_policy=RetryPolicy.AT_MOST_ONCE,
        safety=SafetyClass.SAFE,
    )


def stop_action() -> CognitiveOperation:
    """Build the planner-internal stop control action (never dispatched to modules)."""
    return CognitiveOperation(
        operation_type=OperationType.STOP,
        capability_id=STOP_CAPABILITY,
        module_id=VERIFY_MODULE_ID,
        retry_policy=RetryPolicy.AT_MOST_ONCE,
        safety=SafetyClass.SAFE,
    )


class MPCPlanner:
    """Receding-horizon planner: rollout, hard-constraint filtering, scored selection."""

    def __init__(
        self,
        transition_model: StatisticalTransitionModel,
        hard_constraints: HardConstraintSet | None = None,
        objective: MPCObjectiveWeights | None = None,
        confidence_gate: ConfidenceGate | None = None,
        fallback_policy: FallbackPolicy | None = None,
        config: CWMConfig | None = None,
    ) -> None:
        """Configure the planner with its model, constraints, objective, and gates."""
        self.transition_model = transition_model
        self.hard_constraints = hard_constraints or HardConstraintSet()
        self.objective = objective or MPCObjectiveWeights()
        self.confidence_gate = confidence_gate
        self.fallback_policy = fallback_policy or FallbackPolicy()
        self.config = config or transition_model.config

    def plan(
        self,
        features: CognitiveStateFeatures,
        candidates: Sequence[CognitiveOperation],
        *,
        budget: CWMBudget | None = None,
        goal_threshold: float | None = None,
        expected_revision: int | None = None,
        dependency_depth: int = 0,
        belief_probabilities: Sequence[float] | None = None,
        advisory_mode: str = "shadow",
    ) -> CWMPlan:
        """Plan one bounded rollout and return ranked sequences or a deterministic fallback."""
        budget = budget or CWMBudget(horizon=self.config.default_horizon)
        threshold = (
            self.config.goal_threshold if goal_threshold is None else float(goal_threshold))
        if expected_revision is not None and features.workspace_revision != expected_revision:
            return self._fallback_plan(
                features, FallbackTrigger.STALE_REVISION,
                "feature revision does not match the authoritative workspace revision")
        if budget.cancelled:
            return self._fallback_plan(
                features, FallbackTrigger.TIMEOUT, "planning budget was cancelled")
        started = time.perf_counter()
        deadline = started + budget.planning_time_budget_ms / 1000.0
        verified = self._check_candidates(features, candidates)
        if not verified:
            return self._fallback_plan(
                features, FallbackTrigger.NO_SAFE_CANDIDATE,
                "no candidate satisfied its safety and dependency preconditions")
        planning_context = OperationContext(
            operation_type="planning",
            module_id=verified[0].module_id,
            goal_class=features.goal,
            dependency_depth=dependency_depth,
        )
        first_transitions = [
            self.transition_model.transition(
                features, operation,
                _context_for(planning_context, operation, features, dependency_depth),
                belief_probabilities)
            for operation in verified
        ]
        pruned = self._prune(first_transitions, budget.max_candidates)
        control_actions: list[CognitiveOperation] = [verification_action(), stop_action()]
        collected: list[tuple[PredictedSequence, CognitiveStateFeatures]] = []
        self._enumerate(
            current_features=features,
            available=[item.operation for item in pruned] + control_actions,
            steps=[],
            horizon=budget.horizon,
            budget=budget,
            deadline=deadline,
            collected=collected,
            belief_probabilities=belief_probabilities,
        )
        sequences = [self._finalize(sequence, terminal_features, threshold, features)
                     for sequence, terminal_features in collected]
        ranked = self._rank(sequences, threshold)
        rejected = tuple(sequence for sequence in sequences if not sequence.feasible)
        predictions = self._prediction_records(features, pruned, budget, advisory_mode)
        selected = self._select(ranked)
        if ranked and selected is None:
            return self._fallback_plan(
                features, FallbackTrigger.LOW_CONFIDENCE,
                "no feasible sequence cleared the conservative confidence gates",
                predictions=predictions, ranked_sequences=tuple(ranked),
                rejected_sequences=rejected)
        if not ranked:
            return self._fallback_plan(
                features, FallbackTrigger.NO_SAFE_CANDIDATE,
                "every enumerated sequence violated a hard constraint",
                predictions=predictions, ranked_sequences=tuple(sequences),
                rejected_sequences=rejected)
        return CWMPlan(
            workspace_revision=features.workspace_revision,
            goal=features.goal,
            horizon=budget.horizon,
            ranked_sequences=tuple(ranked),
            rejected_sequences=rejected,
            selected=selected,
            first_action=selected.steps[0].transition.operation,
            predictions=predictions,
            stopped=(
                selected.steps[0].transition.operation.operation_type is OperationType.STOP),
            explanation=self._plan_explanation(selected, ranked),
        )

    # -- internals -----------------------------------------------------------

    def _check_candidates(
        self, features: CognitiveStateFeatures, candidates: Sequence[CognitiveOperation],
    ) -> list[CognitiveOperation]:
        """Apply safety and dependency precondition checks (Steps 10 and 11)."""
        verified: list[CognitiveOperation] = []
        candidate_ids = {item.operation_id for item in candidates}
        for operation in candidates:
            if operation.safety is not SafetyClass.SAFE:
                continue
            if any(reference not in candidate_ids for reference in operation.depends_on):
                continue
            verified.append(operation)
        return verified

    def _prune(
        self, transitions: Sequence[PredictedTransition], max_candidates: int,
    ) -> list[PredictedTransition]:
        """Greedy prune to the candidate budget by predicted benefit per unit cost."""
        def ordering(item: PredictedTransition) -> float:
            outcome = item.predicted_outcome
            cpu = outcome.expected_cost.expected_cpu_cost
            cost = max(1.0, float(cpu) if cpu is not MISSING else 1.0)
            benefit = outcome.goal_progress_estimate + outcome.information_gain.value
            return benefit / cost

        ordered = sorted(transitions, key=ordering, reverse=True)
        return ordered[:max_candidates]

    def _enumerate(
        self,
        *,
        current_features: CognitiveStateFeatures,
        available: list[CognitiveOperation],
        steps: list[PredictedStep],
        horizon: int,
        budget: CWMBudget,
        deadline: float,
        collected: list[tuple[PredictedSequence, CognitiveStateFeatures]],
        belief_probabilities: Sequence[float] | None,
    ) -> None:
        """Depth-first enumeration of bounded sequences under branch and time budgets.

        Every non-empty prefix is recorded as a candidate sequence; prefixes that already
        violate monotone constraints (resource budget, deadline, permissions, retries)
        cannot be repaired by appending and are pruned immediately.
        """
        if time.perf_counter() > deadline or len(collected) >= budget.max_sequences:
            return
        if steps:
            aggregate = self._aggregate(steps)
            collected.append((self._skeleton(steps, aggregate), current_features))
            if self._monotone_violated(aggregate, steps):
                return
            includes_stop = any(
                step.transition.operation.operation_type is OperationType.STOP
                for step in steps)
            if includes_stop or len(steps) >= horizon:
                return
        for operation in available:
            if len(collected) >= budget.max_sequences:
                return
            context = OperationContext(
                operation_type=operation.operation_type.value,
                module_id=operation.module_id,
                goal_class=current_features.goal,
                dependency_depth=len(steps),
            )
            transition = self.transition_model.transition(
                current_features, operation, context, belief_probabilities)
            step = _step(len(steps), transition, self._aggregate(steps))
            remaining = [item for item in available if item.operation_id != operation.operation_id]
            self._enumerate(
                current_features=transition.next_features,
                available=remaining,
                steps=steps + [step],
                horizon=horizon,
                budget=budget,
                deadline=deadline,
                collected=collected,
                belief_probabilities=belief_probabilities,
            )

    def _monotone_violated(self, aggregate: dict[str, float], steps: Sequence[PredictedStep]) -> bool:
        """Check the constraints that appending operations can only worsen."""
        module_ids = [step.transition.operation.module_id for step in steps]
        verdict = self.hard_constraints.evaluate(
            module_ids=module_ids,
            total_cost=aggregate["cost"],
            total_duration_ms=aggregate["duration"],
            produced_fields=(),
            includes_verification=False,
            evidence_coverage=1.0,
            retries=0,
            monotone_only=True,
        )
        return not verdict.feasible

    def _skeleton(
        self, steps: Sequence[PredictedStep], aggregate: dict[str, float],
    ) -> PredictedSequence:
        """Freeze aggregate and identity fields of one enumerated sequence."""
        return PredictedSequence(
            steps=tuple(steps),
            total_cost=aggregate["cost"],
            total_duration_ms=aggregate["duration"],
            total_benefit=aggregate["benefit"],
            total_risk=aggregate["risk"],
            total_uncertainty=aggregate["uncertainty"],
            produced_fields=tuple(dict.fromkeys(
                field_name
                for step in steps
                for field_name in step.transition.operation.produced_fields)),
            module_ids=tuple(
                dict.fromkeys(step.transition.operation.module_id for step in steps)),
            includes_verification=any(
                step.transition.operation.operation_type is OperationType.VERIFY
                for step in steps),
            includes_stop=any(
                step.transition.operation.operation_type is OperationType.STOP
                for step in steps),
        )

    def _finalize(
        self,
        sequence: PredictedSequence,
        terminal_features: CognitiveStateFeatures,
        goal_threshold: float,
        initial_features: CognitiveStateFeatures,
    ) -> PredictedSequence:
        """Apply terminal hard constraints, goal score, and objective score to one sequence."""
        goal_score = terminal_features.value("goal_completion")
        initial_coverage = initial_features.value("evidence_coverage")
        coverage_now = terminal_features.value("evidence_coverage")
        verdict = self.hard_constraints.evaluate(
            module_ids=list(sequence.module_ids),
            total_cost=sequence.total_cost,
            total_duration_ms=sequence.total_duration_ms,
            produced_fields=sequence.produced_fields,
            includes_verification=sequence.includes_verification,
            evidence_coverage=(
                float(coverage_now) if coverage_now is not MISSING else
                (float(initial_coverage) if initial_coverage is not MISSING else 0.0)),
        )
        margin = self.objective.uncertainty_margin * sequence.total_uncertainty
        effective_cost = sequence.total_cost + margin
        effective_benefit = max(0.0, sequence.total_benefit - margin)
        shortfall = max(0.0, goal_threshold - (
            float(goal_score) if goal_score is not MISSING else 0.0))
        score = (
            self.objective.cost_weight * effective_cost
            + self.objective.risk_weight * sequence.total_risk
            + self.objective.uncertainty_weight * sequence.total_uncertainty
            + self.objective.delay_weight * (sequence.total_duration_ms / 1000.0)
            + self.objective.terminal_shortfall_weight * shortfall
            - self.objective.benefit_weight * effective_benefit
        )
        violations = tuple((item.constraint, item.detail) for item in verdict.violations)
        return PredictedSequence(
            sequence_id=sequence.sequence_id,
            steps=sequence.steps,
            total_cost=sequence.total_cost,
            total_duration_ms=sequence.total_duration_ms,
            total_benefit=sequence.total_benefit,
            total_risk=sequence.total_risk,
            total_uncertainty=sequence.total_uncertainty,
            terminal_goal_score=(
                float(goal_score) if goal_score is not MISSING else 0.0),
            produced_fields=sequence.produced_fields,
            module_ids=sequence.module_ids,
            includes_verification=sequence.includes_verification,
            includes_stop=sequence.includes_stop,
            feasible=verdict.feasible,
            violations=violations,
            score=score,
        )

    def _rank(
        self, sequences: Sequence[PredictedSequence], goal_threshold: float,
    ) -> list[PredictedSequence]:
        """Order feasible sequences by the conservative objective, lowest score first."""
        feasible = [item for item in sequences if item.feasible]
        feasible.sort(key=lambda item: (item.score if item.score is not None else 0.0,
                                        item.sequence_id))
        return feasible

    def _select(self, ranked: Sequence[PredictedSequence]) -> PredictedSequence | None:
        """Select the best sequence whose first action clears the confidence gate."""
        if not ranked:
            return None
        if self.confidence_gate is None:
            return ranked[0]
        for sequence in ranked:
            first = sequence.steps[0].transition
            verdict = self.confidence_gate.assess(
                first.predicted_outcome, classify_operation(first.operation))
            if verdict.supported:
                return sequence
        return None

    def _prediction_records(
        self,
        features: CognitiveStateFeatures,
        pruned: Sequence[PredictedTransition],
        budget: CWMBudget,
        advisory_mode: str,
    ) -> tuple[PredictionRecord, ...]:
        """Build one provisional prediction record per first-step candidate."""
        records = []
        for transition in pruned:
            if transition.operation.operation_type is not OperationType.MODULE_EXECUTION:
                continue
            records.append(PredictionRecord(
                workspace_id=features.workspace_id,
                workspace_revision=features.workspace_revision,
                goal_id=features.goal,
                operation_id=transition.operation.operation_id,
                feature_snapshot_hash=features.feature_snapshot_hash,
                predicted=transition.predicted_outcome,
                predictor_version=self.transition_model.predictor_version,
                horizon=budget.horizon,
                constraints_considered=self.hard_constraints.constraint_names(),
                advisory_mode=advisory_mode,
                notes=(
                    f"operation_class={classify_operation(transition.operation)};"
                    f"module_id={transition.operation.module_id}"
                ),
            ))
        return tuple(records)

    def _fallback_plan(
        self,
        features: CognitiveStateFeatures,
        trigger: FallbackTrigger,
        reason: str,
        *,
        predictions: tuple[PredictionRecord, ...] = (),
        ranked_sequences: tuple[PredictedSequence, ...] = (),
        rejected_sequences: tuple[PredictedSequence, ...] = (),
    ) -> CWMPlan:
        """Return a plan that defers to existing deterministic coordinator policies."""
        decision = self.fallback_policy.decide(trigger, reason)
        return CWMPlan(
            workspace_revision=features.workspace_revision,
            goal=features.goal,
            horizon=1,
            ranked_sequences=ranked_sequences,
            rejected_sequences=rejected_sequences,
            predictions=predictions,
            fallback=decision,
            explanation=f"CWM deferred to {decision.deferred_to}: {reason}",
        )

    def _plan_explanation(
        self, selected: PredictedSequence, ranked: Sequence[PredictedSequence],
    ) -> str:
        """Explain the selected recommendation with benefits, costs, risks, and support."""
        first = selected.steps[0].transition
        outcome = first.predicted_outcome
        return (
            f"selected sequence of {len(selected.steps)} actions starting with "
            f"{first.operation.module_id}; predicted benefit {selected.total_benefit:.3f}, "
            f"cost {selected.total_cost:.3f}, risk {selected.total_risk:.3f}, "
            f"uncertainty {selected.total_uncertainty:.3f}; historical support "
            f"{outcome.uncertainty.sample_count} validated executions; "
            f"{len(ranked)} feasible sequences compared"
        )

    def _aggregate(self, steps: Sequence[PredictedStep]) -> dict[str, float]:
        """Return the cumulative aggregates of one step prefix."""
        if not steps:
            return {"cost": 0.0, "duration": 0.0, "benefit": 0.0,
                    "risk": 0.0, "uncertainty": 0.0}
        return {
            "cost": steps[-1].cumulative_cost,
            "duration": steps[-1].cumulative_duration_ms,
            "benefit": steps[-1].cumulative_benefit,
            "risk": steps[-1].cumulative_risk,
            "uncertainty": steps[-1].cumulative_uncertainty,
        }


def _step(index: int, transition: PredictedTransition, aggregate: dict[str, float]) -> PredictedStep:
    """Build one rollout step from a transition and the prior cumulative aggregates."""
    outcome = transition.predicted_outcome
    duration = outcome.expected_cost.expected_duration_ms
    cpu = outcome.expected_cost.expected_cpu_cost
    return PredictedStep(
        step_index=index,
        transition=transition,
        cumulative_cost=aggregate["cost"] + (float(cpu) if cpu is not MISSING else 0.0),
        cumulative_duration_ms=aggregate["duration"] + (
            float(duration) if duration is not MISSING else 0.0),
        cumulative_benefit=aggregate["benefit"] + (
            outcome.goal_progress_estimate + outcome.information_gain.value),
        cumulative_risk=aggregate["risk"] + outcome.risks.aggregate(),
        cumulative_uncertainty=aggregate["uncertainty"] + (
            outcome.uncertainty.success_credible_interval[1]
            - outcome.uncertainty.success_credible_interval[0]),
    )


def _context_for(
    planning_context: OperationContext,
    operation: CognitiveOperation,
    features: CognitiveStateFeatures,
    dependency_depth: int,
) -> OperationContext:
    """Build the conditioning context for one candidate operation (Step 12)."""
    return OperationContext(
        operation_type=operation.operation_type.value,
        module_id=operation.module_id,
        goal_class=features.goal,
        input_size=planning_context.input_size,
        dependency_depth=dependency_depth,
        resource_load=planning_context.resource_load,
    )


__all__ = [
    "STOP_CAPABILITY",
    "VERIFY_CAPABILITY",
    "VERIFY_MODULE_ID",
    "CWMPlan",
    "MPCObjectiveWeights",
    "MPCPlanner",
    "PredictedSequence",
    "PredictedStep",
    "stop_action",
    "verification_action",
]
