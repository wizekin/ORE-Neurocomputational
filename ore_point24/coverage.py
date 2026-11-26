"""Phases I, Steps 3-4 - point-to-layer map and the evidence-based capability ledger.

Step 3 assigns every source capability to one primary layer (secondary links
allowed) so that no source point disappears merely because it overlaps another
point.  Step 4 classifies every capability into an evidence-based
implementation state using the strict gate

    g_c = I_contract * I_backend * I_tests * I_integration

whose indicators are 1 only when independently evidenced.  Intermediate states
stay visible rather than collapsing into one percentage; "examined" and
"implemented" are never interchangeable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .core import (
    EvidenceState,
    ModuleStateVector,
    derive_evidence_state,
    new_id,
    sha256_digest,
    utc_now_iso,
)
from .spec import LAYER_IDS, POINT_FAMILIES, point_layer_incidence

LEDGER_SCHEMA = "ore.point24.coverage_ledger.v1"


@dataclass(frozen=True, slots=True)
class LayerAssignment:
    """One row of the Step 3 point-to-layer map."""

    point: int
    family: str
    primary_layer: str
    secondary_layers: tuple[str, ...]
    responsibility: str
    producing_package: str | None
    consuming_package: str | None
    required_contract: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "point": self.point,
            "family": self.family,
            "primary_layer": self.primary_layer,
            "secondary_layers": list(self.secondary_layers),
            "responsibility": self.responsibility,
            "producing_package": self.producing_package,
            "consuming_package": self.consuming_package,
            "required_contract": self.required_contract,
        }


def build_point_layer_map() -> tuple[LayerAssignment, ...]:
    """Build the traceable Step 3 map from the static point families.

    Every one of the 24 source points maps to at least one layer and a
    documented responsibility; shared algorithms are recorded once and
    referenced from all relevant points.
    """
    assignments: list[LayerAssignment] = []
    for family in POINT_FAMILIES:
        assignments.append(LayerAssignment(
            point=family.point,
            family=family.family,
            primary_layer=family.primary_layer,
            secondary_layers=family.secondary_layers,
            responsibility=family.primary_responsibility,
            producing_package=family.package,
            consuming_package="ore_point24" if family.point != 24 else None,
            required_contract=f"ore.point24.point_{family.point:02d}.v1",
        ))
    return tuple(assignments)


@dataclass(frozen=True, slots=True)
class CapabilityLedgerEntry:
    """One capability with its strict gate indicators and evidence state."""

    capability_id: str
    point: int
    layer: str
    owner_module: str
    state_vector: ModuleStateVector
    evidence_state: EvidenceState
    gate: float
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability_id": self.capability_id,
            "point": self.point,
            "layer": self.layer,
            "owner_module": self.owner_module,
            "state_vector": self.state_vector.to_dict(),
            "evidence_state": self.evidence_state.value,
            "gate": self.gate,
            "notes": list(self.notes),
        }


@dataclass(slots=True)
class CapabilityLedger:
    """The Step 4 evidence-labelled ledger over all identified capabilities."""

    ledger_id: str = field(default_factory=lambda: new_id("point24_ledger"))
    created_at: str = field(default_factory=utc_now_iso)
    entries: dict[str, CapabilityLedgerEntry] = field(default_factory=dict)

    def add(
        self,
        capability_id: str,
        *,
        point: int,
        layer: str,
        owner_module: str,
        vector: ModuleStateVector,
        notes: Iterable[str] = (),
    ) -> CapabilityLedgerEntry:
        """Record one capability; the state is derived, never asserted."""
        if capability_id in self.entries:
            raise ValueError(f"capability already recorded: {capability_id}")
        state = derive_evidence_state(vector)
        gate = float(
            vector.manifest_registered
            and vector.backend_bound
            and vector.validated
            and vector.integration_tested
        )
        entry = CapabilityLedgerEntry(
            capability_id=capability_id,
            point=point,
            layer=layer,
            owner_module=owner_module,
            state_vector=vector,
            evidence_state=state,
            gate=gate,
            notes=tuple(notes),
        )
        self.entries[capability_id] = entry
        return entry

    def entry(self, capability_id: str) -> CapabilityLedgerEntry:
        return self.entries[capability_id]

    def state_counts(self) -> dict[str, int]:
        """Count entries per evidence state; intermediate states stay visible."""
        counts = {state.value: 0 for state in EvidenceState}
        for entry in self.entries.values():
            counts[entry.evidence_state.value] += 1
        return counts

    def weighted_coverage(self, weights: Mapping[str, float] | None = None) -> float:
        """Weighted implementation coverage C = sum(w_c g_c) / sum(w_c).

        Published alongside - never instead of - the state counts.  A family
        marked "examined" is not automatically counted as implemented because
        the gate requires every indicator to be independently evidenced.
        """
        if not self.entries:
            return 0.0
        total_weight = 0.0
        covered = 0.0
        for capability_id, entry in self.entries.items():
            weight = float(weights.get(capability_id, 1.0)) if weights else 1.0
            if weight < 0:
                raise ValueError("capability weights must be non-negative")
            total_weight += weight
            covered += weight * entry.gate
        if total_weight <= 0:
            return 0.0
        return covered / total_weight

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": LEDGER_SCHEMA,
            "ledger_id": self.ledger_id,
            "created_at": self.created_at,
            "entries": {key: entry.to_dict() for key, entry in sorted(self.entries.items())},
            "state_counts": self.state_counts(),
            "weighted_coverage": self.weighted_coverage(),
            "incidence_matrix": point_layer_incidence(),
        }

    def digest(self) -> str:
        return sha256_digest(self.to_dict())


def assert_full_point_coverage(ledger: CapabilityLedger) -> None:
    """Control condition of Step 3: all 24 points have at least one entry."""
    covered = {entry.point for entry in ledger.entries.values()}
    missing = sorted(set(range(1, 25)) - covered)
    if missing:
        raise ValueError(f"points without any capability ledger entry: {missing}")
    for entry in ledger.entries.values():
        if entry.layer not in LAYER_IDS:
            raise ValueError(f"unknown layer {entry.layer!r} on capability {entry.capability_id}")


__all__ = [
    "LEDGER_SCHEMA",
    "CapabilityLedger",
    "CapabilityLedgerEntry",
    "LayerAssignment",
    "assert_full_point_coverage",
    "build_point_layer_map",
]
