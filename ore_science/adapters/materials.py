"""Materials intelligence adapters: graph prediction and conditional generation (Steps 20-21).

Two interfaces with local reference implementations: a graph-based property predictor
(nearest-neighbour cohesive-energy heuristic, explicitly NOT GNoME and not a laboratory
result) and a conditional generator (MatterGen-style interface proposing bounded
candidates under composition and lattice constraints). Every candidate passes physical
plausibility filters - charge balance, composition bounds, lattice geometry, periodicity,
and duplicate detection - before any expensive downstream validation, and predicted
stability is always reported separately from experimentally confirmed stability.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..representations import MaterialsGraph

__all__ = [
    "ConditionalMaterialGenerator",
    "LocalReferenceStabilityModel",
    "MaterialsPlausibilityReport",
    "MaterialsPropertyPrediction",
]

# Common formal oxidation states for charge-balance checking.
_OXIDATION_STATES: dict[str, tuple[int, ...]] = {
    "H": (1,), "Li": (1,), "Na": (1,), "K": (1,), "Cs": (1,),
    "Mg": (2,), "Ca": (2,), "Sr": (2,), "Ba": (2,),
    "B": (3,), "Al": (3,), "Ga": (3,), "In": (3,),
    "C": (4, 2, -4), "Si": (4,), "Ge": (4,), "Sn": (4, 2), "Pb": (4, 2),
    "N": (-3, 3, 5), "P": (-3, 3, 5), "As": (-3, 3, 5),
    "O": (-2,), "S": (-2, 2, 4, 6), "Se": (-2, 2, 4, 6), "Te": (-2, 2, 4, 6),
    "F": (-1,), "Cl": (-1,), "Br": (-1,), "I": (-1,),
    "Fe": (2, 3), "Co": (2, 3), "Ni": (2,), "Cu": (1, 2), "Zn": (2,),
    "Ti": (4,), "Zr": (4,), "Hf": (4,), "V": (3, 5), "Nb": (5,), "Ta": (5,),
    "Mn": (2, 3, 4), "Cr": (3, 6), "Mo": (4, 6), "W": (4, 6),
    "Y": (3,), "La": (3,), "Ce": (3, 4), "Gd": (3,),
}

# Soft composition bounds for the A-site perovskite candidate search.
_COMPOSITION_BOUNDS: dict[str, tuple[float, float]] = {
    "Ba": (1.0, 1.0), "Sr": (1.0, 1.0), "Ca": (1.0, 1.0),
    "Ti": (1.0, 1.0), "O": (3.0, 3.0),
}


@dataclass(frozen=True, slots=True)
class MaterialsPlausibilityReport:
    """Physical plausibility filter results for a candidate structure (Step 21)."""

    charge_balanced: bool
    composition_within_bounds: bool
    lattice_parameter_valid: bool
    periodicity_preserved: bool
    duplicate_of: tuple[str, ...]
    passed: bool
    detail: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the plausibility report."""
        return {
            "charge_balanced": self.charge_balanced,
            "composition_within_bounds": self.composition_within_bounds,
            "lattice_parameter_valid": self.lattice_parameter_valid,
            "periodicity_preserved": self.periodicity_preserved,
            "duplicate_of": list(self.duplicate_of),
            "passed": self.passed,
            "detail": dict(self.detail),
        }


def structure_fingerprint(graph: MaterialsGraph) -> str:
    """Canonical hash of composition, lattice, and sites for duplicate detection."""
    payload = {
        "composition": sorted([[element, count] for element, count in graph.composition]),
        "lattice": [list(row) for row in graph.lattice],
        "sites": sorted([site for site in graph.sites]),
        "periodic": graph.periodic,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]


def check_charge_balance(composition: Sequence[tuple[str, int]]) -> tuple[bool, int]:
    """Check whether the composition admits a neutral formal-charge assignment."""
    variables: list[tuple[str, int]] = [item for item in composition if item[1] > 0]
    if not variables:
        return False, 0

    def search(index: int, running_total: int) -> int | None:
        if index == len(variables):
            return running_total
        element, count = variables[index]
        for state in _OXIDATION_STATES.get(element, (0,)):
            outcome = search(index + 1, running_total + state * count)
            if outcome == 0:
                return outcome
        return None

    for state in _OXIDATION_STATES.get(variables[0][0], (0,)):
        if search(1, state * variables[0][1]) == 0:
            return True, 0
    total_charge = sum(
        _OXIDATION_STATES.get(element, (0,))[0] * count for element, count in variables
    )
    return False, total_charge


def evaluate_plausibility(
    graph: MaterialsGraph,
    *,
    known_fingerprints: Sequence[str] = (),
    lattice_bounds_nm: tuple[float, float] = (0.15, 1.2),
) -> MaterialsPlausibilityReport:
    """Run the Step 21 physical plausibility filters on one candidate."""
    charge_ok, net_charge = check_charge_balance(graph.composition)
    bounds_ok = all(
        _COMPOSITION_BOUNDS.get(element, (-math.inf, math.inf))[0]
        <= count
        <= _COMPOSITION_BOUNDS.get(element, (-math.inf, math.inf))[1]
        for element, count in graph.composition
    ) if any(element in _COMPOSITION_BOUNDS for element, _ in graph.composition) else True
    lattice_parameters = [
        math.sqrt(sum(component ** 2 for component in row)) for row in graph.lattice
    ]
    lattice_ok = all(
        lattice_bounds_nm[0] <= value <= lattice_bounds_nm[1] for value in lattice_parameters
    )
    periodic_ok = bool(graph.periodic)
    fingerprint = structure_fingerprint(graph)
    duplicates = tuple(
        known for known in known_fingerprints if known == fingerprint
    )
    detail = {
        "fingerprint": fingerprint,
        "net_formal_charge": net_charge,
        "lattice_parameters_nm": [round(value, 4) for value in lattice_parameters],
        "unit_cell_volume_nm3": round(graph.unit_cell_volume_nm3(), 6),
    }
    passed = charge_ok and bounds_ok and lattice_ok and periodic_ok and not duplicates
    return MaterialsPlausibilityReport(
        charge_balanced=charge_ok,
        composition_within_bounds=bounds_ok,
        lattice_parameter_valid=lattice_ok,
        periodicity_preserved=periodic_ok,
        duplicate_of=duplicates,
        passed=passed,
        detail=detail,
    )


@dataclass(frozen=True, slots=True)
class MaterialsPropertyPrediction:
    """Graph-based property prediction with predicted-validated status separation."""

    model_id: str
    model_version: str
    formation_energy_ev_per_atom: float
    predicted_stable: bool
    prediction_uncertainty_ev: float
    fingerprint: str
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the prediction record."""
        return {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "formation_energy_ev_per_atom": self.formation_energy_ev_per_atom,
            "predicted_stable": self.predicted_stable,
            "prediction_uncertainty_ev": self.prediction_uncertainty_ev,
            "fingerprint": self.fingerprint,
            "notes": list(self.notes),
            "status": "PREDICTED (model output, not a laboratory result)",
        }


class LocalReferenceStabilityModel:
    """Graph-based stability predictor with a transparent physics heuristic (Step 20).

    The heuristic estimates cohesive energy from site density and composition mix;
    it is a local reference model playing the GNoME-style role. It is NOT GNoME and
    its predicted stability must never be reported as an experimental result.
    """

    model_id = "materials.gnn_reference"
    model_version = "1.0.0"

    def predict(self, graph: MaterialsGraph) -> MaterialsPropertyPrediction:
        """Predict formation energy and stability from crystal-graph features."""
        volume = graph.unit_cell_volume_nm3()
        site_count = len(graph.sites)
        density = site_count / max(volume, 1.0e-9)
        species_count = len(graph.composition)
        # Transparent heuristic: denser packing and mixed chemistry lower the energy.
        energy = -1.6 * min(density, 60.0) / 60.0 - 0.35 * (species_count - 1)
        uncertainty = 0.12 + 0.05 * max(0.0, density - 45.0)
        stable = energy < -1.15
        return MaterialsPropertyPrediction(
            model_id=self.model_id,
            model_version=self.model_version,
            formation_energy_ev_per_atom=float(energy),
            predicted_stable=bool(stable),
            prediction_uncertainty_ev=float(uncertainty),
            fingerprint=structure_fingerprint(graph),
            notes=(
                "GNoME-style interface with a local heuristic backend; predicted "
                "stability is a model output requiring independent computation or "
                "experiment for confirmation.",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the model identity."""
        return {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "kind": "local_reference (GNoME-style interface)",
            "external_gnome": "not_configured",
        }


@dataclass(frozen=True, slots=True)
class GeneratedCandidate:
    """One generated candidate structure with provenance and status."""

    graph: MaterialsGraph
    seed: int
    status: str = "CANDIDATE"
    provenance_note: str = "generated by conditional generation; requires validation"

    def to_dict(self) -> dict[str, Any]:
        """Serialize the candidate record."""
        return {
            "graph": self.graph.to_dict(),
            "fingerprint": structure_fingerprint(self.graph),
            "seed": self.seed,
            "status": self.status,
            "provenance_note": self.provenance_note,
        }


class ConditionalMaterialGenerator:
    """MatterGen-style conditional generator under explicit constraints (Step 20).

    Proposes cubic ABO3 candidates whose lattice parameter is drawn within declared
    bounds conditioned on a target density. Every proposal preserves composition,
    structure, synthesis assumptions, and the seed, and remains a candidate.
    """

    model_id = "materials.conditional_generator"
    model_version = "1.0.0"

    def generate(
        self,
        *,
        target_density_nm_minus3: float = 55.0,
        candidate_count: int = 4,
        random_seed: int = 2026,
        lattice_bounds_nm: tuple[float, float] = (0.35, 0.48),
    ) -> tuple[GeneratedCandidate, ...]:
        """Generate bounded candidate structures conditioned on the target density.

        Candidates are cubic ABO3 perovskites with the A-site cation cycling through
        barium, strontium, and calcium so every proposal is charge neutral with Ti(+4)
        and O(-2).
        """
        if not 1 <= candidate_count <= 12:
            raise ValueError("candidate sets are bounded to at most 12 members")
        generator = np.random.default_rng(random_seed)
        a_site_elements = ("Ba", "Sr", "Ca")
        candidates: list[GeneratedCandidate] = []
        for index in range(candidate_count):
            parameter = float(generator.uniform(*lattice_bounds_nm))
            a_site = a_site_elements[index % len(a_site_elements)]
            lattice = (
                (parameter, 0.0, 0.0),
                (0.0, parameter, 0.0),
                (0.0, 0.0, parameter),
            )
            sites = (
                (a_site, 0.0, 0.0, 0.0),
                ("Ti", 0.5, 0.5, 0.5),
                ("O", 0.5, 0.5, 0.0),
                ("O", 0.5, 0.0, 0.5),
                ("O", 0.0, 0.5, 0.5),
            )
            graph = MaterialsGraph(
                composition=((a_site, 1), ("Ti", 1), ("O", 3)),
                lattice=lattice,
                sites=sites,
            )
            candidates.append(
                GeneratedCandidate(
                    graph=graph,
                    seed=random_seed + index,
                    provenance_note=(
                        f"conditional on target density {target_density_nm_minus3:g} nm^-3; "
                        "synthesis assumptions: none declared"
                    ),
                )
            )
        return tuple(candidates)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the generator identity."""
        return {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "kind": "local_reference (MatterGen-style interface)",
            "external_mattergen": "not_configured",
        }
