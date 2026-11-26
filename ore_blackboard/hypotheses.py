"""Evidence aggregation, Bayesian updates, and explicit conflict resolution (Step 21).

The evidence score ``E(h) = sum_g rho_g Aggregate_i(w_i e_i(h))`` is implemented by
the existing ``ore_workspace.HypothesisScorer`` with a conservative maximum
aggregation inside each correlation group. The score is an evidence score, not
automatically a probability. The Bayesian updater is provided only for callers
that can declare defensible conditional independence and unique evidence items.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from ore_workspace.evidence import (
    EvidenceContribution,
    HypothesisScorer,
)
from ore_workspace.models import GoalEvaluation

SCORE_DEFINITION = (
    "E(h) = sum_g rho_g * Aggregate_i(w_i * e_i(h)) with Aggregate = max within each"
    " correlation group; rho_g = 1 for each used group. Transparent evidence score,"
    " not automatically a calibrated probability."
)


@dataclass(frozen=True, slots=True)
class RankedHypothesis:
    """Pair one retained hypothesis with its transparent evidence score (Step 21)."""

    hypothesis_id: str
    label: str
    evidence: tuple[EvidenceContribution, ...]


def aggregate_evidence(hypotheses: Iterable[RankedHypothesis]) -> tuple[Any, ...]:
    """Score each hypothesis with the shared conservative evidence scorer (Step 21)."""
    scorer = HypothesisScorer()
    return tuple(scorer.score(item.evidence) for item in hypotheses)


class BayesianHypothesisUpdater:
    """Log-odds Bayesian updating only under defensible independence (Step 21)."""

    def posterior_probability(
        self,
        prior: float,
        likelihood_ratios: Sequence[float],
        *,
        evidence_ids: Sequence[str],
        independence_declared: bool,
    ) -> float:
        """Apply ``log odds posterior = log odds prior + sum_i log LR_i`` (Step 21).

        Refuses duplicated evidence identifiers and refuses updates when
        conditional independence has not been explicitly declared. Duplicate or
        dependent evidence must never be multiplied as though independent.
        """
        if not 0.0 <= prior <= 1.0:
            raise ValueError("prior must be in [0, 1]")
        if not independence_declared:
            raise ValueError(
                "Bayesian updating requires an explicit conditional independence declaration"
            )
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("duplicated evidence identifiers cannot be counted as independent")
        if not evidence_ids:
            return prior
        if any(not math.isfinite(value) or value <= 0 for value in likelihood_ratios):
            raise ValueError("likelihood ratios must be finite and positive")
        if len(likelihood_ratios) != len(evidence_ids):
            raise ValueError("one likelihood ratio is required per evidence item")
        log_odds = math.log(prior / (1.0 - prior)) if 0.0 < prior < 1.0 else (
            -math.inf if prior == 0.0 else math.inf
        )
        for ratio in likelihood_ratios:
            log_odds += math.log(ratio)
        if log_odds == math.inf:
            return 1.0
        if log_odds == -math.inf:
            return 0.0
        return 1.0 / (1.0 + math.exp(-log_odds))


def summarize_open_conflicts(workspace, hypothesis_conflict_handling: bool = True) -> tuple[dict[str, Any], ...]:
    """Summarize unresolved conflicts with their retained alternatives (Steps 21, 28)."""
    if not hypothesis_conflict_handling:
        return ()
    conflicts = workspace.conflicts.list(workspace.workspace_id, unresolved_only=True)
    summaries: list[dict[str, Any]] = []
    for record in conflicts:
        alternatives = []
        for finding_id in record.conflicting_findings:
            finding = workspace.findings.get(finding_id)
            alternatives.append({
                "finding_id": finding_id,
                "producer_module": finding.producer_module if finding else None,
                "candidate": (
                    finding.payload.get("candidate") if finding else None
                ),
            })
        summaries.append({
            "conflict_id": record.conflict_id,
            "conflict_type": record.conflict_type,
            "affected_fields": list(record.affected_fields),
            "status": record.status,
            "alternatives": alternatives,
            "note": "Both alternatives are retained until an explicit resolution action.",
        })
    return tuple(summaries)


def resolve_conflict_with_evidence(
    workspace, conflict_id: str, *, preferred_finding_id: str,
) -> Any:
    """Apply an explicit, evidence-backed resolution to one conflict (Step 21).

    Resolution is a separate control action; it never silently overwrites either
    alternative and both findings remain addressable in the workspace.
    """
    return workspace.state_manager.resolve_conflict(
        conflict_id, preferred_finding_id=preferred_finding_id,
        resolution_candidates=(),
    )


def goal_progress(previous: GoalEvaluation | None, current: GoalEvaluation) -> float:
    """Return ``Delta C_G`` between two goal evaluations for the stagnation test (Step 24)."""
    if previous is None:
        return current.score
    return current.score - previous.score


__all__ = [
    "BayesianHypothesisUpdater",
    "RankedHypothesis",
    "SCORE_DEFINITION",
    "aggregate_evidence",
    "goal_progress",
    "resolve_conflict_with_evidence",
    "summarize_open_conflicts",
]
