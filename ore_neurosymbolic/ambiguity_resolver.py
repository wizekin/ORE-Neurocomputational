"""Candidate generation, Section 6.3 ranking and ambiguity preservation (Phase D).

The ranking score is ``S(c) = w_t*T(c) + w_g*G(c) + w_e*E(c) + w_m*M(c)
- w_v*V(c)`` where T is type compatibility, G is grammar/context fit,
E is evidence support, M is calibrated model similarity and V counts
soft-constraint violations. Hard violations are filtered before ranking.
Weights are versioned (``RankingWeights``) and must be evaluated on held-out
data before any quality claim (Section 6.3; Section 10).

Step D4 (preserve ambiguity): when several surviving candidates remain close
(within ``epsilon``), the resolver publishes the alternatives together with
their discriminating evidence instead of forcing one interpretation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from .contracts import CandidateInterpretation, CandidateStatus, make_id


@dataclass(frozen=True, slots=True)
class RankingWeights:
    """Versioned ranking weights for Section 6.3; defaults sum to 1.0 on the positive side."""

    version: str = "1.0.0"
    w_t: float = 0.3
    w_g: float = 0.25
    w_e: float = 0.25
    w_m: float = 0.2
    w_v: float = 0.3

    def __post_init__(self) -> None:
        """Require finite, non-negative weights and a semver-style version."""
        values = (self.w_t, self.w_g, self.w_e, self.w_m, self.w_v)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("ranking weights must be finite and non-negative")
        if not isinstance(self.version, str) or not self.version:
            raise ValueError("RankingWeights.version must be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "version": self.version,
            "w_t": self.w_t, "w_g": self.w_g, "w_e": self.w_e,
            "w_m": self.w_m, "w_v": self.w_v,
        }


@dataclass(frozen=True, slots=True)
class ScoredCandidate:
    """One candidate with its computed Section 6.3 score and components."""

    candidate_id: str
    score: float
    components: Mapping[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "candidate_id": self.candidate_id,
            "score": round(self.score, 6),
            "components": dict(self.components),
        }


@dataclass(frozen=True, slots=True)
class AmbiguityGroup:
    """A preserved alternative set with discriminating evidence (Step D4)."""

    group_id: str
    member_ids: tuple[str, ...]
    score_spread: float
    discriminating_evidence: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "group_id": self.group_id,
            "member_ids": list(self.member_ids),
            "score_spread": round(self.score_spread, 6),
            "discriminating_evidence": list(self.discriminating_evidence),
        }


class AmbiguityResolver:
    """Generate, rank and preserve alternative interpretations (Steps D1, D3, D4)."""

    def __init__(
        self,
        weights: RankingWeights | None = None,
        *,
        epsilon: float = 0.05,
        max_candidates: int = 256,
    ) -> None:
        """Bind the versioned weights, closeness epsilon and candidate budget."""
        self.weights = weights or RankingWeights()
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        self.epsilon = epsilon
        if max_candidates <= 0:
            raise ValueError("max_candidates must be positive")
        self.max_candidates = max_candidates

    def generate_candidates(
        self,
        readings: Mapping[str, tuple[str, ...]],
        evidence_refs: tuple[str, ...] = (),
        *,
        structure_template: Mapping[str, Any] | None = None,
        neural_scores: Mapping[str, Mapping[str, float]] | None = None,
    ) -> tuple[CandidateInterpretation, ...]:
        """Generate the finite candidate set from lexical/USG alternatives (Step D1).

        ``readings`` maps one reference (word/entity) to its possible concept
        IDs; ambiguous references therefore produce one candidate per reading.
        Candidates combine grammar/type/context/relation constraints through
        their declared structure template.
        """
        candidates: list[CandidateInterpretation] = []
        template = dict(structure_template or {})
        neural = neural_scores or {}
        for reference, alternatives in sorted(readings.items()):
            for concept_id in alternatives:
                structure = {
                    **template,
                    "reference": reference,
                    "concept_id": concept_id,
                    "entities": (reference, concept_id),
                }
                candidate_id = make_id("candidate")
                candidates.append(CandidateInterpretation(
                    candidate_id=candidate_id,
                    semantic_structure=structure,
                    evidence_refs=evidence_refs,
                    neural_scores=dict(neural.get(concept_id, {})),
                    constraint_status={},
                    ranking_features={"type_compatibility": 1.0},
                ))
        if len(candidates) > self.max_candidates:
            candidates = candidates[:self.max_candidates]
        return tuple(candidates)

    def score(
        self,
        candidate: CandidateInterpretation,
        soft_scores: Mapping[str, float] | None = None,
    ) -> ScoredCandidate:
        """Compute ``S(c)`` per Section 6.3 from recorded features (Step D3)."""
        features = candidate.ranking_features
        soft = soft_scores or {}
        t_value = float(features.get("type_compatibility", 0.0))
        g_value = float(soft.get("context_fit", features.get("context_fit", 0.0)))
        e_value = float(soft.get("evidence_support", features.get("evidence_support", 0.0)))
        m_value = max(candidate.neural_scores.values()) if candidate.neural_scores else 0.0
        v_value = float(features.get("soft_violations", 0.0))
        w = self.weights
        raw = (
            w.w_t * t_value + w.w_g * g_value + w.w_e * e_value + w.w_m * m_value
            - w.w_v * v_value
        )
        score = min(1.0, max(0.0, raw))
        components = {
            "type_compatibility": t_value,
            "context_fit": g_value,
            "evidence_support": e_value,
            "model_similarity": m_value,
            "soft_violations": v_value,
        }
        return ScoredCandidate(
            candidate_id=candidate.candidate_id, score=score, components=components,
        )

    def rank(
        self,
        candidates: tuple[CandidateInterpretation, ...],
        soft_scores: Mapping[str, Mapping[str, float]] | None = None,
    ) -> tuple[ScoredCandidate, ...]:
        """Rank survivors deterministically; ties order by candidate ID (Step D3)."""
        soft = soft_scores or {}
        scored = [
            self.score(candidate, soft.get(candidate.candidate_id, {}))
            for candidate in candidates
        ]
        scored.sort(key=lambda item: (-item.score, item.candidate_id))
        return tuple(scored)

    def preserve_ambiguity(
        self,
        ranked: tuple[ScoredCandidate, ...],
        discriminating_evidence: tuple[str, ...] = (),
    ) -> tuple[AmbiguityGroup, ...]:
        """Group close survivors into preserved alternatives (Step D4).

        Any maximal run of consecutive candidates whose scores stay within
        ``epsilon`` of the group leader is published as one alternative group;
        a clear winner outside such a run stays single (no forced grouping).
        """
        groups: list[AmbiguityGroup] = []
        index = 0
        while index < len(ranked):
            leader = ranked[index]
            members = [leader]
            cursor = index + 1
            while cursor < len(ranked) and (leader.score - ranked[cursor].score) <= self.epsilon:
                members.append(ranked[cursor])
                cursor += 1
            if len(members) > 1:
                groups.append(AmbiguityGroup(
                    group_id=make_id("ambiguity"),
                    member_ids=tuple(item.candidate_id for item in members),
                    score_spread=leader.score - members[-1].score,
                    discriminating_evidence=discriminating_evidence,
                ))
            index = cursor
        return tuple(groups)

    def mark_alternatives(
        self,
        candidates: tuple[CandidateInterpretation, ...],
        groups: tuple[AmbiguityGroup, ...],
    ) -> tuple[CandidateInterpretation, ...]:
        """Tag candidates that belong to a preserved ambiguity group (Step D4)."""
        member_ids = {
            item for group in groups for item in group.member_ids
        }
        return tuple(
            CandidateInterpretation(
                candidate_id=item.candidate_id,
                semantic_structure=dict(item.semantic_structure),
                evidence_refs=item.evidence_refs,
                neural_scores=dict(item.neural_scores),
                constraint_status=dict(item.constraint_status),
                ranking_features=dict(item.ranking_features),
                status=(
                    CandidateStatus.ALTERNATIVE
                    if item.candidate_id in member_ids else item.status
                ),
            )
            for item in candidates
        )


__all__ = [
    "AmbiguityGroup",
    "AmbiguityResolver",
    "RankingWeights",
    "ScoredCandidate",
]
