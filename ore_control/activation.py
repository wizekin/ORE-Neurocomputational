"""Activation-based retrieval scoring for the cognitive control layer.

This module implements the ACT-R-inspired retrieval mathematics of Point 02
Phase C: base-level activation ``B_i``, cue association ``S_i``, the combined
retrieval score ``R_i``, competition/ambiguity preservation, and the
budget-aware retrieval-depth ladder. Every formula is an engineering
approximation labelled as such; missing usage history is reported as a
missing feature, never fabricated (Point 02 §5.1, Step 13).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .config import AmbiguityPolicy, RetrievalDepthPolicy
from .contracts import RetrievalCandidate, sha256_json

ACTIVATION_FORMULA = (
    "B_i = ln(epsilon + sum_j t_ij^-d): ACT-R-inspired engineering "
    "approximation of recency/frequency activation, not the full ACT-R model"
)
CUE_FORMULA = "S_i = sum_k W_k * A_ik with per-cue contribution traces"
RETRIEVAL_FORMULA = (
    "R_i = alpha*B_i + beta*S_i + gamma*Q_i + delta*P_i - lambda*C_i; "
    "configurable task-profile coefficients, not calibrated probabilities; "
    "missing features drop their term and flag the candidate instead of "
    "silently substituting zero"
)


def base_level_activation(
    usage_timestamps: Sequence[float],
    *,
    now: float,
    decay: float,
    epsilon: float,
) -> float | None:
    """Compute ``B_i = ln(eps + sum_j t_ij^-d)`` (Step 13).

    Returns ``None`` when usage history is absent or unreliable so the caller
    can mark the feature unavailable instead of inventing a value. Elapsed
    times must be strictly positive; non-positive gaps (bad clocks) also make
    the feature unavailable rather than being silently clamped.
    """
    if decay <= 0 or epsilon <= 0:
        raise ValueError("decay and epsilon must be positive")
    if not usage_timestamps:
        return None
    total = 0.0
    for timestamp in usage_timestamps:
        elapsed = now - float(timestamp)
        if not math.isfinite(elapsed) or elapsed <= 0:
            return None
        total += elapsed ** (-decay)
    if total <= 0.0 or not math.isfinite(total):
        return None
    value = math.log(epsilon + total)
    return value if math.isfinite(value) else None


def cue_association(
    cue_weights: Mapping[str, float],
    associations: Mapping[str, float],
) -> tuple[float, dict[str, float]]:
    """Compute ``S_i = sum_k W_k * A_ik`` with per-cue contributions (Step 14).

    Cue weights are normalized so correlated cues cannot double-count scale;
    associations must already be normalized by the caller. The per-cue
    contribution trace is returned alongside the score.
    """
    if not cue_weights:
        return 0.0, {}
    negative = [name for name, weight in cue_weights.items() if weight < 0]
    if negative:
        raise ValueError(f"cue weights must be non-negative: {negative}")
    total_weight = sum(cue_weights.values())
    if total_weight <= 0:
        return 0.0, {}
    contributions: dict[str, float] = {}
    score = 0.0
    for cue, weight in cue_weights.items():
        strength = float(associations.get(cue, 0.0))
        if strength < 0:
            raise ValueError(f"association for cue {cue!r} must be non-negative")
        contribution = (weight / total_weight) * strength
        contributions[cue] = contribution
        score += contribution
    return score, contributions


def retrieval_score(
    candidate: RetrievalCandidate,
    *,
    coefficients,
    task_relevance: float,
    provenance_support: float,
) -> RetrievalCandidate:
    """Compute ``R_i`` for one candidate (Step 15).

    ``task_relevance`` (Q_i) and ``provenance_support`` (P_i) are supplied by
    the caller from declared task metadata; the retrieval cost ``C_i`` is the
    candidate's declared cost. When ``B_i`` or ``S_i`` is unavailable the term
    is dropped, the dropped terms are recorded on the candidate, and the
    candidate is flagged so ranking cannot mistake missingness for a low
    score.
    """
    dropped: list[str] = []
    score = 0.0
    if candidate.base_level_activation is not None and candidate.activation_available:
        score += coefficients.alpha * candidate.base_level_activation
    else:
        dropped.append("B_i")
    if candidate.cue_association is not None:
        score += coefficients.beta * candidate.cue_association
    else:
        dropped.append("S_i")
    score += coefficients.gamma * max(0.0, task_relevance)
    score += coefficients.delta * max(0.0, provenance_support)
    score -= coefficients.lam * max(0.0, candidate.retrieval_cost)
    if not math.isfinite(score):
        raise ValueError("retrieval score diverged for candidate")
    return RetrievalCandidate(
        record_reference=candidate.record_reference,
        source_type=candidate.source_type,
        cue_match_features=candidate.cue_match_features,
        usage_timestamps=candidate.usage_timestamps,
        semantic_relevance=candidate.semantic_relevance,
        confidence=candidate.confidence,
        provenance=candidate.provenance,
        retrieval_cost=candidate.retrieval_cost,
        access_state=candidate.access_state,
        version_reference=candidate.version_reference,
        inclusion_reasons=candidate.inclusion_reasons,
        exclusion_reasons=candidate.exclusion_reasons,
        base_level_activation=candidate.base_level_activation,
        activation_available=candidate.activation_available,
        cue_association=candidate.cue_association,
        cue_contributions=candidate.cue_contributions,
        retrieval_score=score,
        dropped_terms=tuple(dropped),
    )


@dataclass(frozen=True, slots=True)
class RetrievalShortlist:
    """Ranked retrieval shortlist with preserved ambiguity (Step 16)."""

    ranked: tuple[RetrievalCandidate, ...]
    shortlist: tuple[RetrievalCandidate, ...]
    winner: RetrievalCandidate | None
    ambiguity_records: tuple[Mapping[str, Any], ...]
    formula: str = RETRIEVAL_FORMULA

    def to_dict(self) -> dict[str, Any]:
        """Return the serializable shortlist record."""
        return {
            "ranked": [item.to_dict() for item in self.ranked],
            "shortlist": [item.to_dict() for item in self.shortlist],
            "winner": self.winner.to_dict() if self.winner else None,
            "ambiguity_records": [dict(item) for item in self.ambiguity_records],
            "formula": self.formula,
        }


def rank_candidates(
    candidates: Sequence[RetrievalCandidate],
    *,
    policy: AmbiguityPolicy,
) -> RetrievalShortlist:
    """Rank retrieval candidates while preserving competition (Step 16).

    Candidates without a score (never scored) sort last with their missingness
    preserved. When the top two scores fall inside the ambiguity margin, no
    winner is forced: the shortlist is retained and an ambiguity record is
    emitted for the cycle trace.
    """
    scored = [item for item in candidates if item.retrieval_score is not None]
    unscored = [item for item in candidates if item.retrieval_score is None]
    ranked = sorted(
        scored,
        key=lambda item: (-(item.retrieval_score or 0.0), item.record_reference),
    ) + list(unscored)
    shortlist = tuple(ranked[: policy.shortlist_size])
    winner = None
    ambiguities: list[Mapping[str, Any]] = []
    if ranked:
        top = ranked[0]
        if top.retrieval_score is not None and top.activation_available:
            winner = top
            if len(ranked) > 1 and ranked[1].retrieval_score is not None:
                margin = top.retrieval_score - ranked[1].retrieval_score
                if policy.require_winner_margin and margin < policy.margin:
                    winner = None
                    ambiguities.append({
                        "ambiguity_id": sha256_json({
                            "top": top.record_reference,
                            "runner_up": ranked[1].record_reference,
                        })[:16],
                        "kind": "close_retrieval_scores",
                        "candidates": [top.record_reference, ranked[1].record_reference],
                        "margin": margin,
                        "policy_margin": policy.margin,
                        "scheduled_action": "discriminating retrieval or hypothesis test",
                    })
        else:
            ambiguities.append({
                "ambiguity_id": sha256_json({"top": top.record_reference})[:16],
                "kind": "missing_activation_feature",
                "candidates": [top.record_reference],
                "scheduled_action": "evidence gathering before winner selection",
            })
    return RetrievalShortlist(
        ranked=tuple(ranked),
        shortlist=shortlist,
        winner=winner,
        ambiguity_records=tuple(ambiguities),
    )


@dataclass(frozen=True, slots=True)
class RetrievalDepthDecision:
    """The chosen retrieval depth level under budget (Step 17)."""

    level: str
    expected_gain: float
    expected_cost: float
    reason: str
    skipped_levels: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the serializable depth decision."""
        return {
            "level": self.level,
            "expected_gain": float(self.expected_gain),
            "expected_cost": float(self.expected_cost),
            "reason": self.reason,
            "skipped_levels": list(self.skipped_levels),
        }


def choose_retrieval_depth(
    *,
    remaining_budget: float,
    sufficient_evidence_met: bool,
    depth_policy: RetrievalDepthPolicy,
) -> RetrievalDepthDecision:
    """Choose how deep to search under the remaining budget (Step 17).

    Escalation is marginal: start at the local cache and escalate to the next
    store only while its marginal expected gain exceeds its marginal cost and
    the cumulative cost stays within the remaining budget; deeper levels are
    recorded as skipped with the stopping reason.
    """
    levels = depth_policy.levels()
    if sufficient_evidence_met:
        return RetrievalDepthDecision(
            "none", 0.0, 0.0, "sufficient evidence threshold met",
            tuple(level for level, _, _ in levels),
        )
    chosen_level, chosen_gain, chosen_cost = levels[0]
    skipped: list[str] = []
    if chosen_cost > remaining_budget:
        return RetrievalDepthDecision(
            "none", 0.0, 0.0, "retrieval budget exhausted before any store",
            tuple(level for level, _, _ in levels),
        )
    for (level, gain, cost), (next_level, next_gain, next_cost) in zip(
        levels, levels[1:],
    ):
        if next_gain - gain <= next_cost - cost:
            skipped.append(next_level)
            continue
        if chosen_cost + next_cost > remaining_budget:
            skipped.append(next_level)
            continue
        chosen_level, chosen_gain, chosen_cost = next_level, next_gain, next_cost
    unselected = tuple(
        level for level, _, _ in levels if level != chosen_level
    )
    return RetrievalDepthDecision(
        chosen_level, chosen_gain, chosen_cost,
        "marginal expected gain exceeds marginal cost within budget",
        tuple(skipped + [level for level in unselected if level not in skipped]),
    )


__all__ = [
    "ACTIVATION_FORMULA",
    "CUE_FORMULA",
    "RETRIEVAL_FORMULA",
    "RetrievalDepthDecision",
    "RetrievalShortlist",
    "base_level_activation",
    "choose_retrieval_depth",
    "cue_association",
    "rank_candidates",
    "retrieval_score",
]
