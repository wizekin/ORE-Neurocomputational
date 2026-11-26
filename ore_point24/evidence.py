"""Phase II, Step 9 - attach evidence and provenance to semantic claims.

Preserve source identity, version, rights and attribution; record whether
evidence supports, contradicts or merely contextualises a claim; keep source
text, normalised representation and derived interpretation distinguishable; and
track source dependence so correlated evidence is not counted as independent
corroboration.  The support score ``S(h) = sum_i w_i e_i(h)`` is an
inspectable ranking score, **not** a calibrated probability.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .core import new_id, require_unit_interval, sha256_digest, utc_now_iso

EVIDENCE_SCHEMA = "ore.point24.evidence.v1"

SUPPORTS = "supports"
CONTRADICTS = "contradicts"
CONTEXTUALIZES = "contextualizes"


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    """One evidence record with declared weight, relation and source identity."""

    source_id: str
    relation: str
    weight: float
    evidence_type: str
    source_version: str | None = None
    rights: str | None = None
    attribution: str | None = None
    derived_by: str | None = None
    source_text: str | None = None
    normalized_representation: Any = None
    derived_interpretation: Any = None

    def __post_init__(self) -> None:
        if self.relation not in {SUPPORTS, CONTRADICTS, CONTEXTUALIZES}:
            raise ValueError(f"unknown evidence relation: {self.relation!r}")
        require_unit_interval("evidence weight", self.weight)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "relation": self.relation,
            "weight": self.weight,
            "evidence_type": self.evidence_type,
            "source_version": self.source_version,
            "rights": self.rights,
            "attribution": self.attribution,
            "derived_by": self.derived_by,
            "source_text": self.source_text,
            "normalized_representation": self.normalized_representation,
            "derived_interpretation": self.derived_interpretation,
        }


@dataclass(slots=True)
class SemanticClaim:
    """A provenance-aware semantic claim with its full evidence set."""

    statement: str
    claim_id: str = field(default_factory=lambda: new_id("claim"))
    evidence: list[EvidenceItem] = field(default_factory=list)
    dependence_groups: dict[str, list[str]] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)
    status: str = "hypothesis"

    def add_evidence(self, item: EvidenceItem, *, dependence_group: str | None = None) -> None:
        self.evidence.append(item)
        if dependence_group:
            self.dependence_groups.setdefault(dependence_group, []).append(item.source_id)

    def support_score(self) -> float:
        """S(h) = sum_i w_i e_i(h) over *independent* evidence groups.

        Correlated sources are grouped and contribute the maximum member
        weight once (they must not be counted as independent corroboration).
        The result is a ranking score, never a probability.
        """
        group_of: dict[str, str] = {
            source_id: group for group, members in self.dependence_groups.items()
            for source_id in members
        }
        grouped: dict[str, float] = {}
        total = 0.0
        for item in self.evidence:
            sign = 1.0 if item.relation == SUPPORTS else (-1.0 if item.relation == CONTRADICTS else 0.0)
            contribution = item.weight * sign
            group = group_of.get(item.source_id)
            if group is None:
                total += contribution
            else:
                key = f"{group}:{item.relation}"
                if key not in grouped or abs(contribution) > abs(grouped[key]):
                    grouped[key] = contribution
        return total + sum(grouped.values())

    def independent_corroboration_count(self) -> int:
        """Count evidence items excluding correlated duplicates."""
        group_of: dict[str, str] = {
            source_id: group for group, members in self.dependence_groups.items()
            for source_id in members
        }
        seen_groups: set[str] = set()
        count = 0
        for item in self.evidence:
            group = group_of.get(item.source_id)
            if group is None:
                count += 1
            elif group not in seen_groups:
                seen_groups.add(group)
                count += 1
        return count

    def promote(self, status: str) -> None:
        """Change claim status through an explicit, recorded transition."""
        if status not in {"hypothesis", "supported", "contradicted", "verified", "abstained"}:
            raise ValueError(f"unknown claim status: {status!r}")
        self.status = status

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": EVIDENCE_SCHEMA,
            "claim_id": self.claim_id,
            "statement": self.statement,
            "status": self.status,
            "created_at": self.created_at,
            "evidence": [item.to_dict() for item in self.evidence],
            "dependence_groups": {k: list(v) for k, v in self.dependence_groups.items()},
            "support_score": self.support_score(),
            "independent_corroboration": self.independent_corroboration_count(),
        }

    def digest(self) -> str:
        return sha256_digest(self.to_dict())


__all__ = [
    "CONTEXTUALIZES",
    "CONTRADICTS",
    "EVIDENCE_SCHEMA",
    "SUPPORTS",
    "EvidenceItem",
    "SemanticClaim",
]
