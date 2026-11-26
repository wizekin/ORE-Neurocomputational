"""Evidence retrieval contracts, quality assessment and conflict handling (Steps 26-27).

Evidence retains source identity, timestamp, version, location and
relevance.  Correlated sources collapse into single confirmations, primary
evidence is distinguished from summaries and model-generated claims, and
material disagreement is preserved rather than averaged away.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

from ..contracts import utc_now
from ..functional import bayes_posterior, independent_confirmation_count

__all__ = (
    "EvidenceAssessment",
    "EvidenceBundle",
    "EvidenceItem",
    "EvidenceKind",
    "MaterialConflict",
    "assess_evidence",
)


class EvidenceKind(str, Enum):
    """Distinguishes primary evidence from summaries and model claims (Step 26)."""

    PRIMARY = "primary"
    SUMMARY = "summary"
    MODEL_GENERATED = "model_generated"
    EXTERNAL_REFERENCE = "external_reference"


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    """One piece of retrieved evidence with full provenance (Step 26)."""

    item_id: str
    source_id: str
    claim: str
    kind: EvidenceKind = EvidenceKind.PRIMARY
    source_locator: str = ""
    timestamp: str = field(default_factory=utc_now)
    version: str = ""
    reliability: float = 0.5
    directness: float = 0.5
    recency_seconds: float = 0.0
    applicability: float = 0.5
    correlation_group: str | None = None
    supports: bool = True
    relevance: float = 0.5

    def __post_init__(self) -> None:
        """Validate the evidence envelope."""
        if not self.item_id or not self.item_id.strip():
            raise ValueError("item_id must be a non-empty string")
        if not self.source_id or not self.source_id.strip():
            raise ValueError("source_id must be a non-empty string")
        if not self.claim or not self.claim.strip():
            raise ValueError("claim must be a non-empty string")
        for name in ("reliability", "directness", "applicability", "relevance"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the item with its provenance."""
        return {
            "item_id": self.item_id,
            "source_id": self.source_id,
            "claim": self.claim,
            "kind": self.kind.value,
            "source_locator": self.source_locator,
            "timestamp": self.timestamp,
            "version": self.version,
            "reliability": self.reliability,
            "directness": self.directness,
            "recency_seconds": self.recency_seconds,
            "applicability": self.applicability,
            "correlation_group": self.correlation_group,
            "supports": self.supports,
            "relevance": self.relevance,
        }


@dataclass(frozen=True, slots=True)
class MaterialConflict:
    """Preserved material disagreement between evidence positions (Step 27)."""

    topic: str
    positions: tuple[Mapping[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the conflict."""
        return {"topic": self.topic,
                "positions": [dict(position) for position in self.positions]}


@dataclass(frozen=True, slots=True)
class EvidenceAssessment:
    """The result of assessing an evidence bundle (Step 27)."""

    independent_confirmations: int
    supporting: tuple[str, ...]
    opposing: tuple[str, ...]
    suppressed_correlated: tuple[str, ...]
    conflicts: tuple[MaterialConflict, ...]
    mean_reliability: float
    bayesian_posterior: float | None
    assumptions: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the assessment."""
        return {
            "independent_confirmations": self.independent_confirmations,
            "supporting": list(self.supporting),
            "opposing": list(self.opposing),
            "suppressed_correlated": list(self.suppressed_correlated),
            "conflicts": [conflict.to_dict() for conflict in self.conflicts],
            "mean_reliability": self.mean_reliability,
            "bayesian_posterior": self.bayesian_posterior,
            "assumptions": list(self.assumptions),
        }


class EvidenceBundle:
    """An ordered collection of evidence items (Steps 26-27)."""

    def __init__(self, items: Sequence[EvidenceItem] = ()) -> None:
        """Hold the items; identity is preserved, never merged."""
        self._items = list(items)

    def add(self, item: EvidenceItem) -> None:
        """Add one item."""
        self._items.append(item)

    @property
    def items(self) -> tuple[EvidenceItem, ...]:
        """Return all items in insertion order."""
        return tuple(self._items)

    def supporting(self) -> tuple[EvidenceItem, ...]:
        """Return the supporting items."""
        return tuple(item for item in self._items if item.supports)

    def opposing(self) -> tuple[EvidenceItem, ...]:
        """Return the opposing items."""
        return tuple(item for item in self._items if not item.supports)

    def primary(self) -> tuple[EvidenceItem, ...]:
        """Return primary evidence only (Step 26)."""
        return tuple(item for item in self._items
                     if item.kind is EvidenceKind.PRIMARY)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the bundle."""
        return {"items": [item.to_dict() for item in self._items]}


def assess_evidence(bundle: EvidenceBundle, topic: str = "",
                    prior: float = 0.5,
                    allow_bayes: bool = False) -> EvidenceAssessment:
    """Assess quality, independence and conflict of an evidence bundle (Step 27).

    Correlated copies (a shared non-None correlation group) collapse to one
    confirmation.  Material disagreement is preserved, never averaged.  The
    Bayesian posterior is computed only when the caller justifies independence
    by setting ``allow_bayes`` and every supporting item is primary evidence.
    """
    items = bundle.items
    supporting = bundle.supporting()
    opposing = bundle.opposing()
    groups = [item.correlation_group for item in supporting]
    confirmations = independent_confirmation_count(groups)
    group_counts: dict[str, int] = {}
    for group in groups:
        if group is not None:
            group_counts[group] = group_counts.get(group, 0) + 1
    suppressed = tuple(sorted(
        item.item_id for item in supporting
        if item.correlation_group is not None
        and group_counts[item.correlation_group] > 1))
    conflicts: list[MaterialConflict] = []
    if supporting and opposing and topic:
        conflicts.append(MaterialConflict(
            topic=topic,
            positions=tuple(
                [{"item_id": item.item_id, "source_id": item.source_id,
                  "claim": item.claim, "supports": item.supports}
                 for item in (*supporting, *opposing)])))
    mean_reliability = (
        sum(item.reliability for item in items) / len(items) if items else 0.0)
    posterior: float | None = None
    assumptions: list[str] = []
    if allow_bayes and supporting:
        all_primary = all(item.kind is EvidenceKind.PRIMARY for item in supporting)
        all_independent = all(
            item.correlation_group is None
            or group_counts.get(item.correlation_group, 0) == 1
            for item in supporting)
        if all_primary and all_independent:
            posterior = prior
            for item in supporting:
                likelihood = 0.5 + item.reliability * 0.5
                posterior = bayes_posterior(posterior, likelihood, 1.0 - likelihood)
            assumptions.append("independent primary evidence; single-topic update")
        else:
            assumptions.append(
                "posterior withheld: correlated or non-primary evidence present")
    else:
        assumptions.append("posterior not requested or no supporting evidence")
    return EvidenceAssessment(
        independent_confirmations=confirmations,
        supporting=tuple(item.item_id for item in supporting),
        opposing=tuple(item.item_id for item in opposing),
        suppressed_correlated=suppressed,
        conflicts=tuple(conflicts),
        mean_reliability=mean_reliability,
        bayesian_posterior=posterior,
        assumptions=tuple(assumptions))
