"""Meta-cognitive monitoring for the control layer (Step 29).

Tracks uncertainty, specialist disagreement, resource consumption, repeated
failures, stale evidence, and calibration status each cycle. Low confidence
or disagreement triggers targeted retrieval, independent verification, or a
human-review escalation; uncertainty is never converted into false certainty
merely to force completion.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .config import ControlPolicy


@dataclass(frozen=True, slots=True)
class MetacognitiveReport:
    """One cycle's monitoring summary with escalation triggers (Step 29)."""

    uncertainty_level: float
    disagreement_present: bool
    repeated_failures: int
    stale_evidence: bool
    budget_consumption: float
    calibration_flagged: bool
    escalations: tuple[str, ...] = field(default_factory=())

    def to_dict(self) -> dict[str, Any]:
        """Return the serializable monitoring record."""
        return {
            "uncertainty_level": float(self.uncertainty_level),
            "disagreement_present": bool(self.disagreement_present),
            "repeated_failures": int(self.repeated_failures),
            "stale_evidence": bool(self.stale_evidence),
            "budget_consumption": float(self.budget_consumption),
            "calibration_flagged": bool(self.calibration_flagged),
            "escalations": list(self.escalations),
        }


def monitor(
    state_summary: Mapping[str, Any],
    *,
    policy: ControlPolicy,
    goal_evaluation: Mapping[str, Any] | None = None,
    repeated_failures: int = 0,
    budget_consumption: float = 0.0,
) -> MetacognitiveReport:
    """Compute the meta-cognitive report for one cycle (Step 29).

    ``state_summary`` is the compact cycle state summary; ``goal_evaluation``
    is the authoritative evaluator output for the current snapshot. Monitoring
    never changes the goal outcome itself - it only proposes escalations.
    """
    open_questions = [
        item for item in state_summary.get("unresolved_questions", [])
        if isinstance(item, Mapping) and not item.get("resolved", False)
    ]
    competing = int(state_summary.get("competing_hypotheses", 0) or 0)
    unresolved_conflicts = int(state_summary.get("unresolved_conflicts", 0) or 0)

    uncertainty = min(
        1.0,
        (len(open_questions) * 0.2) + (competing * 0.1) + (0.1 if unresolved_conflicts else 0.0),
    )
    disagreement = bool(competing or unresolved_conflicts)
    evidence_revision = int(state_summary.get("latest_evidence_revision", 0) or 0)
    current_revision = int(state_summary.get("workspace_revision", 0) or 0)
    stale = (
        current_revision - evidence_revision
    ) >= policy.stale_evidence_revision_gap
    calibration_flagged = uncertainty >= policy.metaconfidence_threshold

    escalations: list[str] = []
    if calibration_flagged and open_questions:
        escalations.append("targeted_retrieval")
    if disagreement and repeated_failures >= policy.budgets.max_retries:
        escalations.append("independent_verification")
    if open_questions and all(
        item.get("awaiting_external_evidence", False) for item in open_questions
    ):
        escalations.append("human_review_path")
    if budget_consumption >= 0.9:
        escalations.append("resource_pressure")

    return MetacognitiveReport(
        uncertainty_level=uncertainty,
        disagreement_present=disagreement,
        repeated_failures=int(repeated_failures),
        stale_evidence=bool(stale),
        budget_consumption=float(min(1.0, budget_consumption)),
        calibration_flagged=bool(calibration_flagged),
        escalations=tuple(escalations),
    )


def apply_escalation_preference(
    candidate_operator_ids: Sequence[str],
    report: MetacognitiveReport,
) -> tuple[str, ...]:
    """Reorder candidate operators to honour escalation triggers.

    Targeted retrieval lifts retrieval operators; independent verification
    lifts the verification operator. This is a preference, not a bypass: hard
    constraints are still applied afterwards (Step 20).
    """
    ordered = list(candidate_operator_ids)
    if "targeted_retrieval" in report.escalations:
        ordered.sort(
            key=lambda operator_id: (0 if "retrieve" in operator_id else 1, operator_id),
        )
    if "independent_verification" in report.escalations:
        ordered.sort(
            key=lambda operator_id: (0 if "verify" in operator_id else 1, operator_id),
        )
    return tuple(ordered)


__all__ = [
    "MetacognitiveReport",
    "apply_escalation_preference",
    "monitor",
]
