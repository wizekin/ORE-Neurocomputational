"""Candidate abstention rules (ORE Point 8 step 31).

Abstention applies when confidence is low, calibration is poor, evidence is insufficient,
or alternatives are near-tied. The rule set never forces a top-1 answer: an abstained
query is returned explicitly as such, together with the reasons, instead of silently
surfacing a weak candidate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..schemas.candidate_schema import CandidateStatus, GraphCandidate


@dataclass(frozen=True, slots=True)
class AbstentionDecision:
    """One abstention outcome with its triggering rule and diagnostic detail."""

    abstained: bool
    rule: str | None
    detail: str = ""

    def as_dict(self) -> dict[str, object]:
        return {"abstained": self.abstained, "rule": self.rule, "detail": self.detail}


class AbstentionPolicy:
    """Decide whether a ranked candidate list should abstain (step 31)."""

    def __init__(
        self,
        *,
        min_utility: float = 0.05,
        utility_margin: float = 0.05,
        max_calibration_error: float | None = None,
        min_evidence_ids: int = 0,
    ) -> None:
        if not 0.0 <= min_utility <= 1.0:
            raise ValueError("min_utility must lie in [0, 1]")
        if not 0.0 <= utility_margin <= 1.0:
            raise ValueError("utility_margin must lie in [0, 1]")
        if max_calibration_error is not None and max_calibration_error < 0:
            raise ValueError("max_calibration_error cannot be negative")
        if min_evidence_ids < 0:
            raise ValueError("min_evidence_ids cannot be negative")
        self.min_utility = float(min_utility)
        self.utility_margin = float(utility_margin)
        self.max_calibration_error = max_calibration_error
        self.min_evidence_ids = int(min_evidence_ids)

    def evaluate(
        self,
        ranked: Sequence[GraphCandidate],
        *,
        calibration_error: float | None = None,
    ) -> AbstentionDecision:
        """Evaluate abstention rules over one ranked candidate list."""
        if not ranked:
            return AbstentionDecision(True, "no_candidates", "no candidate passed pre-ranking filters")
        top = ranked[0]
        if top.utility is None or top.utility < self.min_utility:
            return AbstentionDecision(
                True, "low_confidence",
                f"top utility {top.utility if top.utility is not None else 'n/a'} "
                f"below threshold {self.min_utility}",
            )
        if self.max_calibration_error is not None and calibration_error is not None \
                and calibration_error > self.max_calibration_error:
            return AbstentionDecision(
                True, "poor_calibration",
                f"calibration error {calibration_error:.3f} exceeds {self.max_calibration_error:.3f}",
            )
        if self.min_evidence_ids > 0 and len(top.evidence_ids) < self.min_evidence_ids:
            return AbstentionDecision(
                True, "insufficient_evidence",
                f"top candidate carries {len(top.evidence_ids)} evidence ids "
                f"(< {self.min_evidence_ids})",
            )
        if len(ranked) > 1 and ranked[1].utility is not None:
            margin = top.utility - ranked[1].utility
            if margin < self.utility_margin:
                return AbstentionDecision(
                    True, "near_tied_alternatives",
                    f"top-two utility margin {margin:.4f} below {self.utility_margin}",
                )
        return AbstentionDecision(False, None, "")

    def apply(
        self,
        ranked: Sequence[GraphCandidate],
        *,
        calibration_error: float | None = None,
    ) -> tuple[list[GraphCandidate], AbstentionDecision]:
        """Mark candidates as abstained when the policy abstains (never force top-1)."""
        decision = self.evaluate(ranked, calibration_error=calibration_error)
        if decision.abstained:
            for candidate in ranked:
                candidate.status = CandidateStatus.ABSTAINED
            return [], decision
        return list(ranked), decision


__all__ = ["AbstentionDecision", "AbstentionPolicy"]
