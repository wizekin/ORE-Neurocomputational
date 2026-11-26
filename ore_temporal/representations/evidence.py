"""Evidence references and provenance records (specification Steps 34 and 31)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ("EvidenceBundle", "EvidenceRef", "ProvenanceRecord")


@dataclass(frozen=True, slots=True)
class EvidenceRef:
    """A stable reference to one piece of evidence backing a temporal conclusion."""

    evidence_id: str
    kind: str
    description: str
    reference: str = ""

    def __post_init__(self) -> None:
        """Validate the reference identity."""
        for name in ("evidence_id", "kind", "description"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the evidence reference."""
        return {
            "evidence_id": self.evidence_id,
            "kind": self.kind,
            "description": self.description,
            "reference": self.reference,
        }


@dataclass(frozen=True, slots=True)
class ProvenanceRecord:
    """Step 34 -- explicit provenance distinguishing measured, imputed, and predicted data."""

    source_ids: tuple[str, ...]
    time_range: tuple[float, float]
    preprocessing_version: str
    model_family: str
    model_version: str
    parameter_count: int
    uncertainty_method: str
    baseline_comparison: str
    validation_status: str
    evidence_refs: tuple[EvidenceRef, ...] = ()
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize the provenance record."""
        return {
            "source_ids": list(self.source_ids),
            "time_range": list(self.time_range),
            "preprocessing_version": self.preprocessing_version,
            "model_family": self.model_family,
            "model_version": self.model_version,
            "parameter_count": self.parameter_count,
            "uncertainty_method": self.uncertainty_method,
            "baseline_comparison": self.baseline_comparison,
            "validation_status": self.validation_status,
            "evidence_refs": [item.to_dict() for item in self.evidence_refs],
            "notes": self.notes,
        }


@dataclass(slots=True)
class EvidenceBundle:
    """Mutable collector used while a run assembles its evidence references."""

    _items: list[EvidenceRef] = field(default_factory=list)

    def add(self, evidence_id: str, kind: str, description: str,
            reference: str = "") -> EvidenceRef:
        """Append a new evidence reference with a unique id."""
        if any(item.evidence_id == evidence_id for item in self._items):
            raise ValueError(f"duplicate evidence id {evidence_id!r}")
        item = EvidenceRef(evidence_id, kind, description, reference)
        self._items.append(item)
        return item

    def refs(self) -> tuple[str, ...]:
        """Return the evidence ids in insertion order."""
        return tuple(item.evidence_id for item in self._items)

    def __len__(self) -> int:
        """Return the number of collected references."""
        return len(self._items)

    def to_dict(self) -> list[dict[str, Any]]:
        """Serialize the collected evidence."""
        return [item.to_dict() for item in self._items]
