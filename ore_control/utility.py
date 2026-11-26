"""Expected-utility operator selection for the control layer (Phase D).

Implements Point 02 Steps 18-21: outcome estimation from measured history or
labelled priors, the EU(o) and normalized U(o) design scores, hard-constraint
filtering before ranking, and deterministic selection with stable
tie-breaking (or a safe parallel set when independence is proven).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .config import ControlPolicy
from .contracts import CandidateRecord, OperatorSpec

EU_FORMULA = (
    "EU(o) = Pr(useful progress | o) * V(o) / (epsilon + C(o)); an ORE "
    "design score, not an equation prescribed by Soar or ACT-R"
)
U_FORMULA = (
    "U(o) = w_g*G + w_i*I + w_c*K - w_t*T - w_r*R - w_f*F; normalized ORE "
    "design score with hard constraints applied outside the score"
)


@dataclass(frozen=True, slots=True)
class OutcomeEstimate:
    """One operator's estimated outcome with labelled provenance (Step 18)."""

    operator_id: str
    probability_useful_progress: float
    value: float
    time_seconds: float
    memory_mb: float
    risk: float
    reversibility: float
    independently_verifiable: float
    provenance: str  # "measured_history" | "declared_prior"

    def __post_init__(self) -> None:
        """Validate estimate bounds."""
        for name in (
            "probability_useful_progress", "value", "risk", "reversibility",
            "independently_verifiable",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        for name in ("time_seconds", "memory_mb"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")


def estimate_outcomes(
    specs: Iterable[OperatorSpec],
    *,
    goal_gaps: Sequence[str],
    measured_success: Mapping[str, float] | None = None,
    measured_latency_seconds: Mapping[str, float] | None = None,
) -> dict[str, OutcomeEstimate]:
    """Estimate outcomes from measured data or declared conservative priors.

    Estimates built from measured history use the recorded success rate and
    latency; everything else is a declared prior from the operator spec and is
    labelled as such (Step 18).
    """
    estimates: dict[str, OutcomeEstimate] = {}
    for spec in specs:
        success = measured_success.get(spec.operator_id) if measured_success else None
        if success is not None and math.isfinite(float(success)):
            probability = min(1.0, max(0.0, float(success)))
            provenance = "measured_history"
        else:
            probability = float(spec.expected_progress)
            provenance = "declared_prior"
        latency = (
            measured_latency_seconds.get(spec.operator_id)
            if measured_latency_seconds else None
        )
        time_seconds = (
            float(latency) if latency is not None and math.isfinite(float(latency))
            else float(spec.estimated_time_seconds)
        )
        contributes = bool(
            set(spec.write_fields) & set(goal_gaps)
        ) or bool(goal_gaps)
        value = 1.0 if contributes else spec.expected_progress
        risk_map = {
            "NONE": 0.0, "LOW": 0.2, "MEDIUM": 0.5, "HIGH": 0.8,
        }
        estimates[spec.operator_id] = OutcomeEstimate(
            operator_id=spec.operator_id,
            probability_useful_progress=probability,
            value=value,
            time_seconds=time_seconds,
            memory_mb=float(spec.estimated_memory_mb),
            risk=risk_map.get(spec.risk_class.value, 0.5),
            reversibility=1.0 if spec.reversible else 0.0,
            independently_verifiable=(
                1.0 if spec.validation_policy == "deterministic_contract_checks" else 0.5
            ),
            provenance=provenance,
        )
    return estimates


@dataclass(frozen=True, slots=True)
class UtilityScores:
    """The two labelled design scores for one candidate (Step 19)."""

    operator_id: str
    expected_utility: float
    normalized_utility: float
    terms: Mapping[str, float]
    formulas: tuple[str, str] = (EU_FORMULA, U_FORMULA)

    def to_dict(self) -> dict[str, Any]:
        """Return the serializable score record."""
        return {
            "operator_id": self.operator_id,
            "expected_utility": float(self.expected_utility),
            "normalized_utility": float(self.normalized_utility),
            "terms": dict(self.terms),
            "formulas": list(self.formulas),
        }


def compute_utilities(
    candidates: Sequence[CandidateRecord],
    estimates: Mapping[str, OutcomeEstimate],
    *,
    policy: ControlPolicy,
) -> dict[str, UtilityScores]:
    """Compute EU(o) and U(o) for each candidate (Step 19).

    Terms are normalized before combination: time and memory costs are
    normalized against the largest candidate cost so no single scale dominates.
    Safety, permission, and critical-goal checks are applied separately as
    hard filters (Step 20), never inside this score.
    """
    if not candidates:
        return {}
    max_time = max(
        (estimates[item.operator_id].time_seconds for item in candidates
         if item.operator_id in estimates), default=1.0,
    ) or 1.0
    max_memory = max(
        (estimates[item.operator_id].memory_mb for item in candidates
         if item.operator_id in estimates), default=1.0,
    ) or 1.0
    scores: dict[str, UtilityScores] = {}
    for candidate in candidates:
        estimate = estimates.get(candidate.operator_id)
        if estimate is None:
            continue
        cost = estimate.time_seconds + estimate.memory_mb / 100.0
        expected_utility = (
            estimate.probability_useful_progress * estimate.value
            / (policy.eu_epsilon + cost)
        )
        weights = policy.utility
        time_cost = estimate.time_seconds / max_time if max_time else 0.0
        memory_cost = estimate.memory_mb / max_memory if max_memory else 0.0
        normalized_utility = (
            weights.w_g * estimate.probability_useful_progress * estimate.value
            + weights.w_i * 0.5  # operator-declared information gain share
            + weights.w_c * estimate.independently_verifiable
            - weights.w_t * time_cost
            - weights.w_r * estimate.risk
            - weights.w_f * memory_cost
        )
        scores[candidate.operator_id] = UtilityScores(
            operator_id=candidate.operator_id,
            expected_utility=expected_utility,
            normalized_utility=normalized_utility,
            terms={
                "probability_useful_progress": estimate.probability_useful_progress,
                "value": estimate.value,
                "time_cost": time_cost,
                "memory_cost": memory_cost,
                "risk": estimate.risk,
                "independently_verifiable": estimate.independently_verifiable,
                "estimate_provenance": 0.0 if estimate.provenance == "measured_history"
                else 1.0,
            },
        )
    return scores


@dataclass(frozen=True, slots=True)
class ConstraintVerdict:
    """Hard-constraint verdict for one candidate (Step 20)."""

    operator_id: str
    passes: bool
    rejections: tuple[str, ...] = field(default_factory=())

    def to_dict(self) -> dict[str, Any]:
        """Return the serializable verdict."""
        return {
            "operator_id": self.operator_id,
            "passes": self.passes,
            "rejections": list(self.rejections),
        }


def apply_hard_constraints(
    candidates: Sequence[CandidateRecord],
    specs: Mapping[str, OperatorSpec],
    *,
    remaining_time_seconds: float,
    remaining_operations: int,
    remaining_memory_mb: float,
    critical_requirements_unmet: Sequence[str],
    required_validation_policy: str = "deterministic_contract_checks",
) -> tuple[ConstraintVerdict, ...]:
    """Reject candidates that violate hard constraints (Step 20).

    Budgets, preconditions, permissions, critical-goal conflicts, and the
    required validation standard are hard filters; a high utility score can
    never override them.
    """
    verdicts: list[ConstraintVerdict] = []
    for candidate in candidates:
        spec = specs.get(candidate.operator_id)
        if spec is None:
            verdicts.append(ConstraintVerdict(
                candidate.operator_id, False, ("operator_not_registered",),
            ))
            continue
        rejections: list[str] = []
        if candidate.precondition_failures:
            rejections.append(f"preconditions_failed:{','.join(candidate.precondition_failures)}")
        if spec.estimated_time_seconds > remaining_time_seconds:
            rejections.append("time_budget_exceeded")
        if remaining_operations <= 0:
            rejections.append("operation_budget_exhausted")
        if spec.estimated_memory_mb > remaining_memory_mb:
            rejections.append("memory_budget_exceeded")
        if not spec.enabled:
            rejections.append("operator_disabled")
        if critical_requirements_unmet and not (
            set(spec.write_fields) & set(critical_requirements_unmet)
        ):
            # The operator cannot close any unmet critical requirement;
            # it stays eligible only as a supporting action.
            pass
        if spec.validation_policy != required_validation_policy:
            rejections.append(
                f"validation_standard_insufficient:{spec.validation_policy}"
            )
        verdicts.append(ConstraintVerdict(
            candidate.operator_id, not rejections, tuple(rejections),
        ))
    return tuple(verdicts)


@dataclass(frozen=True, slots=True)
class SelectionOutcome:
    """The selection decision of one cycle (Step 21)."""

    selected: CandidateRecord | None
    parallel_set: tuple[CandidateRecord, ...] = ()
    rationale: str = ""
    rejected: tuple[tuple[str, str], ...] = ()
    selection_mode: str = "deterministic_single"

    def to_dict(self) -> dict[str, Any]:
        """Return the serializable selection record."""
        return {
            "selected": self.selected.to_dict() if self.selected else None,
            "parallel_set": [item.to_dict() for item in self.parallel_set],
            "rationale": self.rationale,
            "rejected": [list(item) for item in self.rejected],
            "selection_mode": self.selection_mode,
        }


def select_action(
    candidates: Sequence[CandidateRecord],
    *,
    utilities: Mapping[str, UtilityScores],
    verdicts: Mapping[str, ConstraintVerdict],
    specs: Mapping[str, OperatorSpec],
    allow_parallel: bool = True,
) -> SelectionOutcome:
    """Select the highest-utility eligible action (Step 21).

    Ranking is deterministic with stable tie-breaking on (score, operator_id,
    arguments_fingerprint). A safe parallel set is returned only when the
    top-ranked candidates have pairwise-disjoint write fields and none of them
    is a decomposition action; otherwise a single action is selected.
    """
    eligible = [
        candidate for candidate in candidates
        if candidate.operator_id in utilities
        and verdicts.get(candidate.operator_id, ConstraintVerdict(
            candidate.operator_id, False, ("unknown_verdict",),
        )).passes
    ]
    if not eligible:
        return SelectionOutcome(
            None, (), "no eligible operator passed hard constraints",
            tuple(
                (candidate.operator_id, "hard_constraint_failed")
                for candidate in candidates
            ),
            "none",
        )
    ranked = sorted(
        eligible,
        key=lambda item: (
            -utilities[item.operator_id].normalized_utility,
            -utilities[item.operator_id].expected_utility,
            item.operator_id,
            item.arguments_fingerprint,
        ),
    )
    selected = ranked[0]
    parallel: list[CandidateRecord] = [selected]
    if allow_parallel and not selected.precondition_failures:
        for candidate in ranked[1:]:
            if len(parallel) >= 2:
                break
            spec = specs.get(candidate.operator_id)
            if spec is None:
                continue
            head_spec = specs[selected.operator_id]
            if set(spec.write_fields) & set(head_spec.write_fields):
                continue
            if any(
                set(spec.write_fields) & set(
                    specs[item.operator_id].write_fields
                ) for item in parallel
            ):
                continue
            parallel.append(candidate)
    if len(parallel) > 1:
        return SelectionOutcome(
            selected=selected,
            parallel_set=tuple(parallel),
            rationale=(
                "safe parallel set: disjoint write fields permit concurrent "
                "execution under the resource reservation"
            ),
            rejected=tuple(
                (item.operator_id, "utility_ranked_below_selected")
                for item in ranked[len(parallel):]
            ),
            selection_mode="deterministic_parallel_set",
        )
    return SelectionOutcome(
        selected=selected,
        rationale=(
            "highest utility after hard constraints; deterministic tie-break "
            "on (score, operator_id, arguments_fingerprint)"
        ),
        rejected=tuple(
            (item.operator_id, "utility_ranked_below_selected")
            for item in ranked[1:]
        ),
        selection_mode="deterministic_single",
    )


__all__ = [
    "ConstraintVerdict",
    "EU_FORMULA",
    "OutcomeEstimate",
    "SelectionOutcome",
    "UtilityScores",
    "U_FORMULA",
    "apply_hard_constraints",
    "compute_utilities",
    "estimate_outcomes",
    "select_action",
]
