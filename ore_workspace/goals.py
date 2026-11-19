"""Weighted but constraint-aware evaluation of progress toward a workspace goal."""
from __future__ import annotations

from typing import Mapping

from .models import GoalEvaluation, GoalOutcome, WorkspaceSnapshot


DEFAULT_GOAL_WEIGHTS = {"lexical": 0.30, "syntactic": 0.25, "semantic": 0.30, "discourse": 0.15}


class GoalEvaluator:
    """Evaluate task coverage while preventing aggregate scores from masking critical omissions."""

    def __init__(
        self,
        weights: Mapping[str, float] | None = None,
        threshold: float = 0.75,
        critical_requirements: Mapping[str, tuple[str, ...]] | None = None,
    ) -> None:
        """Configure component weights, the completion threshold and explicit goal requirements."""
        selected = dict(weights or DEFAULT_GOAL_WEIGHTS)
        if not selected or any(value < 0 for value in selected.values()) or sum(selected.values()) <= 0:
            raise ValueError("goal weights must be non-negative with positive total")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        self.weights = selected
        self.threshold = float(threshold)
        self.critical_requirements = dict(critical_requirements or {
            "understand_sentence": ("lexical_hypotheses", "syntactic_hypotheses", "semantic_hypotheses")
        })

    def evaluate(
        self,
        snapshot: WorkspaceSnapshot,
        component_scores: Mapping[str, float] | None = None,
    ) -> GoalEvaluation:
        """Compute a normalized weighted score and apply the current goal's hard requirements."""
        state = snapshot.state
        scores = dict(component_scores or self._derive_scores(state))
        for component, score in scores.items():
            if not 0.0 <= float(score) <= 1.0:
                raise ValueError(f"component score {component!r} must be in [0, 1]")
        relevant_weights = {key: value for key, value in self.weights.items() if key in scores}
        weight_sum = sum(relevant_weights.values())
        score = (
            sum(relevant_weights[key] * float(scores[key]) for key in relevant_weights) / weight_sum
            if weight_sum else 0.0
        )
        goal = str(state.get("current_goal", ""))
        requirements = self.critical_requirements.get(goal, ())
        missing = tuple(field for field in requirements if not self._field_satisfied(state, field))
        unresolved = tuple(
            str(item.get("question_id", item.get("finding_id", "unresolved question")))
            for item in state.get("unresolved_questions", [])
            if not bool(item.get("resolved", False))
        )
        module_states = state.get("module_status", {})
        statuses = {str(value).casefold() for value in module_states.values()} if isinstance(module_states, Mapping) else set()
        if "failed" in statuses or "error" in statuses:
            outcome = GoalOutcome.FAILED
            message = "A required processing module reported a failure."
        elif "blocked" in statuses or "unavailable" in statuses:
            outcome = GoalOutcome.BLOCKED
            message = "A required processing module is blocked or unavailable."
        elif missing:
            outcome = GoalOutcome.NEEDS_MORE_PROCESSING
            message = "Critical representations are still missing."
        elif score < self.threshold:
            outcome = GoalOutcome.NEEDS_MORE_PROCESSING
            message = "The weighted coverage score is below the completion threshold."
        elif unresolved:
            outcome = GoalOutcome.COMPLETED_WITH_UNCERTAINTY
            message = "The required representations are present, but questions remain unresolved."
        else:
            outcome = GoalOutcome.COMPLETED
            message = "The current goal's required representations and score threshold are satisfied."
        return GoalEvaluation(
            outcome=outcome, score=score, threshold=self.threshold,
            component_scores=scores, missing_requirements=missing,
            uncertainty_reasons=unresolved, message=message,
        )

    def _derive_scores(self, state: Mapping[str, object]) -> dict[str, float]:
        """Estimate transparent coverage ratios from the structured hypotheses currently present."""
        lexical = state.get("lexical_hypotheses", {})
        token_count = max(1, len(str(state.get("current_input", "")).split()))
        lexical_count = len(lexical) if isinstance(lexical, Mapping) else 0
        return {
            "lexical": min(1.0, lexical_count / token_count),
            "syntactic": 1.0 if state.get("syntactic_hypotheses") else 0.0,
            "semantic": 1.0 if state.get("semantic_hypotheses") else 0.0,
            "discourse": 1.0 if state.get("discourse_context") else 0.0,
        }

    def _field_satisfied(self, state: Mapping[str, object], field: str) -> bool:
        """Check whether a critical field contains a minimally viable structured result."""
        value = state.get(field)
        return bool(value)