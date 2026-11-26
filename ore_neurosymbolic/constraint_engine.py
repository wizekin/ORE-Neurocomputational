"""Hard/soft constraint reasoning for candidate interpretations (Phase D, Steps D2-D3).

Hard constraints eliminate only candidates *proven* to violate an explicit
constraint (type, arity, exclusivity, temporal order, required evidence). Soft
constraints score context fit, semantic coherence and evidence support; they
rank but never prove. Every hard elimination carries the violated constraint ID
so rejections stay auditable (Section 5: retain bounded rejection reasons).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .contracts import CandidateInterpretation, CandidateStatus


@dataclass(frozen=True, slots=True)
class HardConstraint:
    """One explicit hard constraint with a stable identifier."""

    constraint_id: str
    kind: str  # type | arity | exclusivity | temporal_order | required_evidence
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the constraint kind and identifier."""
        if not isinstance(self.constraint_id, str) or not self.constraint_id:
            raise ValueError("HardConstraint.constraint_id must be a non-empty string")
        allowed = {
            "type", "arity", "exclusivity", "temporal_order", "required_evidence",
        }
        if self.kind not in allowed:
            raise ValueError(f"HardConstraint.kind must be one of {sorted(allowed)}")

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "constraint_id": self.constraint_id,
            "kind": self.kind,
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True, slots=True)
class SoftConstraint:
    """One soft preference scored between 0 and 1; never a proof."""

    constraint_id: str
    target: str  # context_fit | semantic_coherence | evidence_support
    weight: float = 1.0

    def __post_init__(self) -> None:
        """Validate the target and weight bounds."""
        if not isinstance(self.constraint_id, str) or not self.constraint_id:
            raise ValueError("SoftConstraint.constraint_id must be a non-empty string")
        allowed = {"context_fit", "semantic_coherence", "evidence_support"}
        if self.target not in allowed:
            raise ValueError(f"SoftConstraint.target must be one of {sorted(allowed)}")
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError("SoftConstraint.weight must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "constraint_id": self.constraint_id,
            "target": self.target,
            "weight": self.weight,
        }


@dataclass(slots=True)
class ConstraintReport:
    """Outcome of one constraint pass over the candidate set."""

    active: tuple[CandidateInterpretation, ...] = ()
    hard_filtered: tuple[tuple[str, str], ...] = ()  # (candidate_id, constraint_id)
    soft_scores: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "active_count": len(self.active),
            "hard_filtered": [
                {"candidate_id": item[0], "constraint_id": item[1]}
                for item in self.hard_filtered
            ],
            "soft_scores": {
                key: dict(value) for key, value in sorted(self.soft_scores.items())
            },
        }


def _structure_features(candidate: CandidateInterpretation) -> dict[str, Any]:
    """Read the declared structure features of one candidate."""
    structure = candidate.semantic_structure
    return {
        "type": structure.get("type"),
        "arity": structure.get("arity"),
        "role": structure.get("role"),
        "entities": tuple(structure.get("entities", ())),
        "temporal": structure.get("temporal"),
        "evidence_refs": candidate.evidence_refs,
        "context_tags": tuple(structure.get("context_tags", ())),
    }


class ConstraintEngine:
    """Apply hard filters then soft scores to the candidate set (Steps D2-D3)."""

    def __init__(
        self,
        hard: tuple[HardConstraint, ...] = (),
        soft: tuple[SoftConstraint, ...] = (),
    ) -> None:
        """Bind the declared constraint sets."""
        self.hard = hard
        self.soft = soft

    def apply_hard(
        self, candidates: tuple[CandidateInterpretation, ...],
    ) -> tuple[tuple[CandidateInterpretation, ...], tuple[tuple[str, str], ...]]:
        """Eliminate only proven constraint violations; keep everything else (Step D2)."""
        active: list[CandidateInterpretation] = []
        filtered: list[tuple[str, str]] = []
        for candidate in candidates:
            violation = self._first_hard_violation(candidate)
            if violation is None:
                active.append(candidate)
            else:
                filtered.append((candidate.candidate_id, violation))
        return tuple(active), tuple(filtered)

    def _first_hard_violation(self, candidate: CandidateInterpretation) -> str | None:
        """Return the ID of the first constraint the candidate provably violates."""
        features = _structure_features(candidate)
        for constraint in self.hard:
            violated = False
            if constraint.kind == "type":
                required = str(constraint.parameters.get("required_type", "any"))
                violated = (
                    required != "any"
                    and features["type"] is not None
                    and features["type"] != required
                )
            elif constraint.kind == "arity":
                required = int(constraint.parameters.get("required_arity", 0))
                violated = (
                    features["arity"] is not None and int(features["arity"]) != required
                )
            elif constraint.kind == "exclusivity":
                group = str(constraint.parameters.get("group", ""))
                exclusive = tuple(constraint.parameters.get("exclusive_values", ()))
                role_value = features["role"]
                violated = bool(group) and role_value in exclusive
            elif constraint.kind == "temporal_order":
                required_order = tuple(
                    str(item) for item in constraint.parameters.get("ordered_values", ())
                )
                temporal = features["temporal"]
                violated = (
                    temporal is not None
                    and required_order
                    and str(temporal) not in required_order
                )
            elif constraint.kind == "required_evidence":
                minimum = int(constraint.parameters.get("minimum_count", 1))
                violated = len(features["evidence_refs"]) < minimum
            if violated:
                return constraint.constraint_id
        return None

    def apply_soft(
        self, candidates: tuple[CandidateInterpretation, ...],
    ) -> dict[str, dict[str, float]]:
        """Score context fit, semantic coherence and evidence support (Step D3).

        Soft scores rank candidates but are never treated as proof; they are
        recorded per candidate for the ranking step and for auditability.
        """
        scores: dict[str, dict[str, float]] = {}
        for candidate in candidates:
            features = _structure_features(candidate)
            context_tags = set(features["context_tags"])
            declared_context = self._context_tags()
            overlap = len(context_tags & declared_context)
            context_fit = overlap / max(1, len(context_tags)) if context_tags else 0.5
            coherence = min(1.0, len(features["entities"]) / 2.0)
            evidence_support = min(1.0, len(features["evidence_refs"]) / 2.0)
            for constraint in self.soft:
                if constraint.target == "context_fit":
                    context_fit = min(1.0, context_fit * constraint.weight + 0.0)
                elif constraint.target == "semantic_coherence":
                    coherence = min(1.0, coherence * constraint.weight + 0.0)
                elif constraint.target == "evidence_support":
                    evidence_support = min(1.0, evidence_support * constraint.weight + 0.0)
            scores[candidate.candidate_id] = {
                "context_fit": round(min(1.0, max(0.0, context_fit)), 6),
                "semantic_coherence": round(min(1.0, max(0.0, coherence)), 6),
                "evidence_support": round(min(1.0, max(0.0, evidence_support)), 6),
            }
        return scores

    def _context_tags(self) -> set[str]:
        """Collect the declared context tags from soft-constraint parameters."""
        tags: set[str] = set()
        for constraint in self.hard:
            tags.update(str(item) for item in constraint.parameters.get("context_tags", ()))
        return tags

    def evaluate(
        self, candidates: tuple[CandidateInterpretation, ...],
    ) -> ConstraintReport:
        """Run hard filtering then soft scoring over the whole candidate set."""
        active, filtered = self.apply_hard(candidates)
        soft_scores = self.apply_soft(active)
        updated: list[CandidateInterpretation] = []
        for item in candidates:
            was_filtered = filtered_by(filtered, item.candidate_id)
            updated.append(CandidateInterpretation(
                candidate_id=item.candidate_id,
                semantic_structure=dict(item.semantic_structure),
                evidence_refs=item.evidence_refs,
                neural_scores=dict(item.neural_scores),
                constraint_status={
                    "hard_violations": tuple(
                        constraint_id
                        for candidate_id, constraint_id in filtered
                        if candidate_id == item.candidate_id
                    ),
                    "soft_scores": soft_scores.get(item.candidate_id, {}),
                },
                ranking_features={
                    **item.ranking_features,
                    **soft_scores.get(item.candidate_id, {}),
                },
                status=(
                    CandidateStatus.HARD_FILTERED if was_filtered
                    else CandidateStatus.RANKED
                ),
            ))
        surviving = tuple(
            item for item in updated if item.status is not CandidateStatus.HARD_FILTERED
        )
        return ConstraintReport(
            active=surviving,
            hard_filtered=filtered,
            soft_scores=soft_scores,
        )


def filtered_by(filtered: tuple[tuple[str, str], ...], candidate_id: str) -> bool:
    """Return whether one candidate was hard-filtered."""
    return any(item[0] == candidate_id for item in filtered)


__all__ = [
    "ConstraintEngine",
    "ConstraintReport",
    "HardConstraint",
    "SoftConstraint",
]
