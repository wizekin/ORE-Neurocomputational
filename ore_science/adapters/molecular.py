"""Molecular intelligence adapters: structure, interaction, and design (Steps 16-19).

Three service interfaces are provided with local reference implementations:
structure evaluation (geometric chemistry checks), interaction evaluation
(Lennard-Jones pair energy over declared interaction classes), and generative design
(bounded, seeded backbone candidate generation). External AlphaFold-family or
RFdiffusion services integrate through the same interfaces when access, licensing,
and compute budgets are configured; until then the registry marks them
``not_configured`` and they are never silently substituted. Generated candidates
remain in CANDIDATE status until independent validation completes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..representations import MolecularGraph

__all__ = [
    "GenerativeDesignResult",
    "InteractionEnergyReport",
    "LocalReferenceDesignEngine",
    "LocalReferenceInteractionService",
    "LocalReferenceStructureService",
    "StructureCheckReport",
]

# Covalent radii in nanometres for supported elements (reference values).
_COVALENT_RADII_NM: dict[str, float] = {
    "H": 0.031, "C": 0.076, "N": 0.071, "O": 0.066, "S": 0.105,
    "P": 0.107, "F": 0.057, "Cl": 0.102, "Br": 0.120, "Fe": 0.132,
}


@dataclass(frozen=True, slots=True)
class StructureCheckReport:
    """Independent geometric checks on a molecular structure prediction."""

    bond_length_violations: tuple[str, ...]
    clash_pairs: tuple[str, ...]
    coordinate_completeness: bool
    max_bond_deviation_nm: float
    checks_passed: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialize the structure check report."""
        return {
            "bond_length_violations": list(self.bond_length_violations),
            "clash_pairs": list(self.clash_pairs),
            "coordinate_completeness": self.coordinate_completeness,
            "max_bond_deviation_nm": self.max_bond_deviation_nm,
            "checks_passed": self.checks_passed,
        }


class LocalReferenceStructureService:
    """Geometric chemistry checks for predicted molecular structures (Step 16).

    This is a local reference evaluator, not an AlphaFold-family predictor: it checks
    bond lengths against covalent radii sums (with bond-type-specific expected
    lengths) and detects steric clashes. The AlphaFold-family interface (sequence in,
    structure plus model confidence out) remains a declared service boundary.
    """

    model_id = "molecular.structure_reference"
    model_version = "1.0.0"
    supported_components = ("protein", "peptide", "small_molecule")

    def check_structure(self, graph: MolecularGraph) -> StructureCheckReport:
        """Validate bond lengths and steric clashes against reference geometry."""
        violations: list[str] = []
        max_deviation = 0.0
        clash_pairs: list[str] = []
        positions = np.array(
            [position if position is not None else (0.0, 0.0, 0.0) for _, _, position in graph.atoms]
        )
        for index_a, index_b, order, bond_type in graph.bonds:
            element_a = graph.atoms[index_a][0]
            element_b = graph.atoms[index_b][0]
            distance = float(np.linalg.norm(positions[index_a] - positions[index_b]))
            expected, tolerance = self._expected_bond_length(
                element_a, element_b, order, bond_type
            )
            deviation = abs(distance - expected)
            max_deviation = max(max_deviation, deviation)
            if deviation > tolerance:
                violations.append(
                    f"bond {index_a}-{index_b} ({element_a}-{element_b}, {bond_type}): "
                    f"{distance:.3f} nm vs expected {expected:.3f} nm"
                )
        bonded_pairs = {(a, b) for a, b, _, _ in graph.bonds} | {
            (b, a) for a, b, _, _ in graph.bonds
        }
        for index_a in range(len(graph.atoms)):
            for index_b in range(index_a + 1, len(graph.atoms)):
                if (index_a, index_b) in bonded_pairs:
                    continue
                distance = float(np.linalg.norm(positions[index_a] - positions[index_b]))
                radii_sum = _COVALENT_RADII_NM.get(
                    graph.atoms[index_a][0], 0.08
                ) + _COVALENT_RADII_NM.get(graph.atoms[index_b][0], 0.08)
                if distance < 0.62 * radii_sum:
                    clash_pairs.append(
                        f"non-bonded pair {index_a}-{index_b} at {distance:.3f} nm"
                    )
        completeness = all(position is not None for _, _, position in graph.atoms) \
            if graph.has_coordinates else True
        passed = not violations and not clash_pairs and completeness
        return StructureCheckReport(
            bond_length_violations=tuple(violations),
            clash_pairs=tuple(clash_pairs),
            coordinate_completeness=bool(completeness),
            max_bond_deviation_nm=float(max_deviation),
            checks_passed=bool(passed),
        )

    @staticmethod
    def _expected_bond_length(element_a: str, element_b: str, order: int, bond_type: str) -> tuple[float, float]:
        """Expected bond length and tolerance by element pair and bond type.

        Backbone pseudo-bonds connect consecutive C-alpha atoms (~0.38 nm virtual
        bond, not a single covalent bond); peptide bonds are ~0.133 nm C-N; double
        bonds are ~13% shorter than the radii sum.
        """
        if bond_type == "backbone":
            return 0.38, 0.06
        if bond_type == "peptide":
            return 0.133, 0.03
        radii_sum = _COVALENT_RADII_NM.get(element_a, 0.08) + _COVALENT_RADII_NM.get(
            element_b, 0.08
        )
        if order == 2 or bond_type == "double":
            return 0.87 * radii_sum, 0.045
        return radii_sum, 0.045

    def to_dict(self) -> dict[str, Any]:
        """Serialize the service identity."""
        return {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "kind": "local_reference",
            "supported_components": list(self.supported_components),
            "external_alphafold": "not_configured (requires licensing and API access)",
        }


@dataclass(frozen=True, slots=True)
class InteractionEnergyReport:
    """Pairwise interaction energy evaluation with model identity and units."""

    interaction_class: str
    energy_ev: float
    pair_count: int
    model_id: str
    model_version: str
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the interaction report."""
        return {
            "interaction_class": self.interaction_class,
            "energy_ev": self.energy_ev,
            "pair_count": self.pair_count,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "notes": list(self.notes),
        }


class LocalReferenceInteractionService:
    """Lennard-Jones interaction evaluator for declared interaction classes (Step 17).

    Supports only ``protein_small_molecule`` and ``small_molecule_small_molecule``
    interaction classes. A protein-only structure service must not be routed other
    classes; that compatibility gate lives in the model registry and is enforced here
    as well by rejecting unsupported classes.
    """

    model_id = "molecular.interaction_reference"
    model_version = "1.0.0"
    interaction_classes = ("protein_small_molecule", "small_molecule_small_molecule")
    epsilon_ev = 0.01
    sigma_nm = 0.30

    def evaluate(
        self, graph: MolecularGraph, *, interaction_class: str
    ) -> InteractionEnergyReport:
        """Compute the pairwise Lennard-Jones energy for the requested class."""
        if interaction_class not in self.interaction_classes:
            raise ValueError(
                f"interaction class {interaction_class!r} not supported by "
                f"{self.model_id}; supported classes: {self.interaction_classes}"
            )
        positions = np.array(
            [position if position is not None else (0.0, 0.0, 0.0) for _, _, position in graph.atoms]
        )
        sigma = self.sigma_nm
        epsilon = self.epsilon_ev
        total = 0.0
        pairs = 0
        bonded = {(a, b) for a, b, _, _ in graph.bonds} | {
            (b, a) for a, b, _, _ in graph.bonds
        }
        for index_a in range(len(positions)):
            for index_b in range(index_a + 1, len(positions)):
                if (index_a, index_b) in bonded:
                    continue
                distance = max(float(np.linalg.norm(positions[index_a] - positions[index_b])), 1.0e-9)
                ratio = sigma / distance
                total += 4.0 * epsilon * (ratio ** 12 - ratio ** 6)
                pairs += 1
        return InteractionEnergyReport(
            interaction_class=interaction_class,
            energy_ev=float(total),
            pair_count=pairs,
            model_id=self.model_id,
            model_version=self.model_version,
            notes=(
                "Lennard-Jones reference potential; pairwise additive, non-bonded only.",
                "Interaction-class compatibility is enforced before evaluation.",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the service identity."""
        return {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "interaction_classes": list(self.interaction_classes),
            "potential": "Lennard-Jones 12-6",
        }


@dataclass(frozen=True, slots=True)
class GenerativeDesignResult:
    """A bounded set of generated candidates with seeds and conditions."""

    candidates: tuple[MolecularGraph, ...]
    random_seed: int
    sampling_parameters: dict[str, Any]
    structural_conditions: tuple[str, ...]
    model_id: str
    model_version: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize the candidate set metadata."""
        return {
            "candidate_count": len(self.candidates),
            "random_seed": self.random_seed,
            "sampling_parameters": dict(self.sampling_parameters),
            "structural_conditions": list(self.structural_conditions),
            "model_id": self.model_id,
            "model_version": self.model_version,
            "status": "CANDIDATE (generated, not experimentally established)",
        }


class LocalReferenceDesignEngine:
    """RFdiffusion-style generative candidate engine, locally implemented (Step 18).

    Generates bounded, seeded candidate backbones parameterized by helix rise per
    residue. The engine converts the design objective into explicit structural
    conditions and preserves random seeds and sampling parameters for reproducibility
    (Step 18 requirements).
    """

    model_id = "molecular.rfdiffusion_reference"
    model_version = "1.0.0"

    def generate(
        self,
        *,
        residue_count: int = 12,
        candidate_count: int = 3,
        random_seed: int = 2026,
        rise_nm_range: tuple[float, float] = (0.14, 0.16),
    ) -> GenerativeDesignResult:
        """Generate a bounded candidate set of ideal helix backbones."""
        if residue_count < 2:
            raise ValueError("residue count must be at least 2")
        if not 1 <= candidate_count <= 10:
            raise ValueError("candidate sets are bounded to at most 10 members")
        generator = np.random.default_rng(random_seed)
        candidates: list[MolecularGraph] = []
        for _ in range(candidate_count):
            rise = float(generator.uniform(*rise_nm_range))
            atoms = []
            for residue in range(residue_count):
                angle = residue * (100.0 / 180.0 * math.pi)
                radius_nm = 0.23
                atoms.append(
                    (
                        "C",
                        0.0,
                        (
                            radius_nm * math.cos(angle),
                            radius_nm * math.sin(angle),
                            residue * rise,
                        ),
                    )
                )
            bonds = tuple(
                (index, index + 1, 1, "backbone") for index in range(residue_count - 1)
            )
            candidates.append(MolecularGraph(atoms=tuple(atoms), bonds=bonds))
        return GenerativeDesignResult(
            candidates=tuple(candidates),
            random_seed=random_seed,
            sampling_parameters={
                "residue_count": residue_count,
                "candidate_count": candidate_count,
                "rise_nm_range": list(rise_nm_range),
            },
            structural_conditions=(
                "ideal alpha-helix backbone geometry",
                f"{residue_count} residues per candidate",
            ),
            model_id=self.model_id,
            model_version=self.model_version,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the engine identity."""
        return {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "kind": "local_reference (RFdiffusion-style interface)",
            "external_rfdiffusion": "not_configured",
        }
