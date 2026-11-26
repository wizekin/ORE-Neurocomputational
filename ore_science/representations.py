"""Representation builder: molecular, materials, field, geometric, and factor-graph objects.

This module implements Phase II of the integrated algorithm (Steps 8-15). Each candidate
representation records what it preserves, what it discards, its expected computational
cost, invariances, and known limitations; the builder selects the simplest representation
that preserves the task's essential structure and stores full construction provenance so
that any transformation affecting scientific interpretation is reproducible.
"""
from __future__ import annotations

import enum
import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .contracts import ScientificDomain, ScientificTask
from .units import UNITS, Unit

__all__ = [
    "FactorGraphRepresentation",
    "FactorSpec",
    "FieldRepresentation",
    "GeometricRepresentation",
    "MaterialsGraph",
    "MolecularGraph",
    "RepresentationArtifact",
    "RepresentationBuilder",
    "RepresentationDecision",
    "RepresentationKind",
    "TrajectoryEnsembleRepresentation",
]


class RepresentationKind(enum.Enum):
    """Representation families supported by the builder (Steps 9-14)."""

    MOLECULAR_GRAPH = "MOLECULAR_GRAPH"
    MATERIALS_GRAPH = "MATERIALS_GRAPH"
    FIELD = "FIELD"
    GEOMETRIC = "GEOMETRIC"
    FACTOR_GRAPH = "FACTOR_GRAPH"
    TRAJECTORY_ENSEMBLE = "TRAJECTORY_ENSEMBLE"
    STATE_SPACE = "STATE_SPACE"


@dataclass(frozen=True, slots=True)
class MolecularGraph:
    """Attributed molecular graph plus 3D geometry (Step 9).

    Atoms are nodes carrying element, formal charge, and optional 3D coordinates;
    bonds or interactions are edges carrying bond order and type. A graph alone does
    not preserve all conformational information, which is why coordinates are attached
    when spatial interactions matter.
    """

    atoms: tuple[tuple[str, float, tuple[float, float, float] | None], ...]
    bonds: tuple[tuple[int, int, int, str], ...]
    has_coordinates: bool = True
    coordinate_frame: str = "cartesian_nanometre"
    coordinate_unit: Unit = UNITS["m"]

    def __post_init__(self) -> None:
        """Validate atom indices referenced by bonds and coordinate presence."""
        if not self.atoms:
            raise ValueError("molecular graph requires at least one atom")
        for index_a, index_b, _, _ in self.bonds:
            if not (0 <= index_a < len(self.atoms)) or not (0 <= index_b < len(self.atoms)):
                raise ValueError(f"bond ({index_a}, {index_b}) references an undefined atom")
            if index_a == index_b:
                raise ValueError("bonds must connect two distinct atoms")
        if self.has_coordinates:
            for element, _, position in self.atoms:
                if position is None:
                    raise ValueError(
                        f"atom {element!r} lacks 3D coordinates while the graph declares geometry"
                    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the molecular graph with full attributes."""
        return {
            "atom_count": len(self.atoms),
            "atoms": [
                {
                    "element": element,
                    "charge": charge,
                    "position": list(position) if position else None,
                }
                for element, charge, position in self.atoms
            ],
            "bonds": [
                {"a": a, "b": b, "order": order, "type": bond_type}
                for a, b, order, bond_type in self.bonds
            ],
            "has_coordinates": self.has_coordinates,
            "coordinate_frame": self.coordinate_frame,
            "coordinate_unit": self.coordinate_unit.symbol,
        }


@dataclass(frozen=True, slots=True)
class MaterialsGraph:
    """Structured crystal graph with periodicity and lattice metadata (Step 10)."""

    composition: tuple[tuple[str, int], ...]
    lattice: tuple[tuple[float, float, float], ...]
    sites: tuple[tuple[str, float, float, float], ...]
    periodic: bool = True
    neighbor_cutoff_nm: float = 0.35
    coordinate_unit: Unit = UNITS["m"]

    def __post_init__(self) -> None:
        """Validate lattice shape, site coordinates, and composition coverage."""
        if len(self.lattice) != 3 or any(len(row) != 3 for row in self.lattice):
            raise ValueError("lattice must be a 3x3 matrix")
        for species, _, _, _ in self.sites:
            if not any(element == species for element, _ in self.composition):
                raise ValueError(f"site species {species!r} is absent from the composition")
        if not any(count > 0 for _, count in self.composition):
            raise ValueError("composition must be non-empty")
        if not 0.0 < self.neighbor_cutoff_nm <= 2.0:
            raise ValueError("neighbour cutoff must lie in (0, 2] nm")

    def unit_cell_volume_nm3(self) -> float:
        """Return the unit-cell volume in cubic nanometres."""
        matrix = np.array(self.lattice, dtype=float)
        return float(abs(np.linalg.det(matrix)))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the crystal graph."""
        return {
            "composition": [[element, count] for element, count in self.composition],
            "lattice": [list(row) for row in self.lattice],
            "sites": [
                {"species": species, "fractional": [fx, fy, fz]}
                for species, fx, fy, fz in self.sites
            ],
            "periodic": self.periodic,
            "neighbor_cutoff_nm": self.neighbor_cutoff_nm,
            "unit_cell_volume_nm3": self.unit_cell_volume_nm3(),
        }


@dataclass(frozen=True, slots=True)
class FieldRepresentation:
    """Continuous physical quantity discretized over a domain (Step 11).

    Records the field domain, units, temporal sampling, boundary and initial
    conditions, and discretization, and states whether the underlying mathematical
    model is continuous while the stored representation is only sampled.
    """

    quantity_symbol: str
    quantity_name: str
    unit: Unit
    spatial_domain: tuple[float, float]
    time_domain: tuple[float, float]
    grid_size: int
    time_steps: int
    boundary_condition: str
    initial_condition: str
    boundary_value_si: float | None
    initial_profile: str
    mathematically_continuous: bool = True
    coordinate_frame: str = "rod_centred_1d"

    def __post_init__(self) -> None:
        """Validate domains and discretization parameters."""
        if self.spatial_domain[0] >= self.spatial_domain[1]:
            raise ValueError("spatial domain must satisfy x_min < x_max")
        if self.time_domain[0] >= self.time_domain[1]:
            raise ValueError("time domain must satisfy t_min < t_max")
        if self.grid_size < 3 or self.time_steps < 1:
            raise ValueError("grid size must be >= 3 and time steps >= 1")
        if not self.boundary_condition.strip() or not self.initial_condition.strip():
            raise ValueError("fields require explicit boundary and initial condition records")

    def grid(self) -> np.ndarray:
        """Return the spatial collocation grid in SI units (metres)."""
        return np.linspace(self.spatial_domain[0], self.spatial_domain[1], self.grid_size)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the field representation metadata."""
        return {
            "quantity_symbol": self.quantity_symbol,
            "quantity_name": self.quantity_name,
            "unit": self.unit.symbol,
            "spatial_domain": list(self.spatial_domain),
            "time_domain": list(self.time_domain),
            "grid_size": self.grid_size,
            "time_steps": self.time_steps,
            "boundary_condition": self.boundary_condition,
            "initial_condition": self.initial_condition,
            "boundary_value_si": self.boundary_value_si,
            "initial_profile": self.initial_profile,
            "mathematically_continuous": self.mathematically_continuous,
            "coordinate_frame": self.coordinate_frame,
        }


@dataclass(frozen=True, slots=True)
class GeometricRepresentation:
    """Spatial objects with frames and transformation conventions (Step 12)."""

    points: tuple[tuple[float, ...], ...]
    vectors: tuple[tuple[float, ...], ...] = ()
    frames: tuple[str, ...] = ("world",)
    transform_convention: str = "active_rotation_then_translation"
    dimension: int = 2

    def __post_init__(self) -> None:
        """Validate point and vector shapes against the declared dimension."""
        for point in self.points:
            if len(point) != self.dimension:
                raise ValueError("every point must match the declared dimension")
        for vector in self.vectors:
            if len(vector) != self.dimension:
                raise ValueError("every vector must match the declared dimension")
        if self.dimension not in (2, 3):
            raise ValueError("geometric dimension must be 2 or 3")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the geometric representation."""
        return {
            "point_count": len(self.points),
            "vector_count": len(self.vectors),
            "frames": list(self.frames),
            "transform_convention": self.transform_convention,
            "dimension": self.dimension,
        }


@dataclass(frozen=True, slots=True)
class FactorSpec:
    """One factor: residual function, covariance, scope, and robust loss (Step 13)."""

    factor_id: str
    factor_type: str
    scope: tuple[str, ...]
    residual: Callable[[Mapping[str, Any]], Any]
    covariance: tuple[float, ...]
    robust_loss: str = "quadratic"
    frame: str = "world"

    def __post_init__(self) -> None:
        """Factors require a scope, a positive covariance, and a known loss."""
        if not self.factor_id.strip() or not self.scope:
            raise ValueError("factor requires an id and a non-empty variable scope")
        if not self.covariance or any(variance <= 0.0 for variance in self.covariance):
            raise ValueError("factor covariances must be positive")
        if self.robust_loss not in ("quadratic", "huber"):
            raise ValueError("robust loss must be 'quadratic' or 'huber'")


@dataclass(slots=True)
class FactorGraphRepresentation:
    """Factor graph of variables and measurement factors (Step 13).

    Factors referencing variables that are not declared are rejected at construction,
    which is exactly the acceptance test for representation validation.
    """

    variables: tuple[str, ...]
    factors: list[FactorSpec] = field(default_factory=list)
    variable_frames: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize variable frames to a default world frame."""
        for name in self.variables:
            self.variable_frames.setdefault(name, "world")

    def add_factor(self, factor: FactorSpec) -> None:
        """Add a factor, rejecting references to undefined variables."""
        for variable in factor.scope:
            if variable not in self.variables:
                raise ValueError(
                    f"factor {factor.factor_id!r} references undefined variable {variable!r}"
                )
        self.factors.append(factor)

    def frame_consistency_errors(self) -> list[str]:
        """Report factors whose frame disagrees with any scoped variable's frame."""
        errors = []
        for factor in self.factors:
            for variable in factor.scope:
                if self.variable_frames.get(variable, "world") != factor.frame:
                    errors.append(
                        f"factor {factor.factor_id} uses frame {factor.frame!r} but variable "
                        f"{variable!r} is declared in frame {self.variable_frames.get(variable)!r}"
                    )
        return errors

    def to_dict(self) -> dict[str, Any]:
        """Serialize the factor graph structure without the callables."""
        return {
            "variables": list(self.variables),
            "variable_frames": dict(self.variable_frames),
            "factors": [
                {
                    "factor_id": factor.factor_id,
                    "factor_type": factor.factor_type,
                    "scope": list(factor.scope),
                    "covariance": list(factor.covariance),
                    "robust_loss": factor.robust_loss,
                    "frame": factor.frame,
                }
                for factor in self.factors
            ],
        }


@dataclass(frozen=True, slots=True)
class TrajectoryEnsembleRepresentation:
    """A set of possible futures rather than one deterministic trajectory (Step 14)."""

    member_count: int
    lead_times: tuple[float, ...]
    quantity_name: str
    unit: Unit
    current_state_summary: str
    coordinate_frame: str = "regional_grid"

    def __post_init__(self) -> None:
        """Ensembles require members and lead times."""
        if self.member_count < 1:
            raise ValueError("trajectory ensembles require at least one member")
        if not self.lead_times:
            raise ValueError("trajectory ensembles require lead times")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the ensemble representation metadata."""
        return {
            "member_count": self.member_count,
            "lead_times": list(self.lead_times),
            "quantity_name": self.quantity_name,
            "unit": self.unit.symbol,
            "current_state_summary": self.current_state_summary,
            "coordinate_frame": self.coordinate_frame,
        }


@dataclass(slots=True)
class RepresentationArtifact:
    """A built representation plus the decision record and construction provenance."""

    kind: RepresentationKind
    payload: Any
    preserves: list[str] = field(default_factory=list)
    discards: list[str] = field(default_factory=list)
    invariances: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    expected_cost: str = "low"
    construction_settings: dict[str, Any] = field(default_factory=dict)
    source_hash: str = ""
    coordinate_frame: str = "unspecified"
    unit_symbol: str = "1"
    domain: ScientificDomain = ScientificDomain.GENERAL_SCIENCE

    def payload_hash(self) -> str:
        """Hash the serialized payload so construction is verifiable (Step 15)."""
        if hasattr(self.payload, "to_dict"):
            serialized = self.payload.to_dict()
        else:
            serialized = repr(self.payload)
        return hashlib.sha256(json.dumps(serialized, sort_keys=True, default=str).encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the artifact, replacing the payload with its typed summary."""
        payload_dict = self.payload.to_dict() if hasattr(self.payload, "to_dict") else repr(self.payload)
        return {
            "kind": self.kind.value,
            "payload": payload_dict,
            "preserves": list(self.preserves),
            "discards": list(self.discards),
            "invariances": list(self.invariances),
            "limitations": list(self.limitations),
            "expected_cost": self.expected_cost,
            "construction_settings": dict(self.construction_settings),
            "source_hash": self.source_hash,
            "payload_hash": self.payload_hash(),
            "coordinate_frame": self.coordinate_frame,
            "unit_symbol": self.unit_symbol,
            "domain": self.domain.value,
        }


@dataclass(frozen=True, slots=True)
class RepresentationDecision:
    """Decision record over candidate representations (Step 8)."""

    candidates: tuple[dict[str, Any], ...]
    selected: RepresentationKind
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize the decision record."""
        return {
            "candidates": [dict(candidate) for candidate in self.candidates],
            "selected": self.selected.value,
            "rationale": self.rationale,
        }


_CANDIDATE_LIBRARY: dict[ScientificDomain, tuple[RepresentationKind, ...]] = {
    ScientificDomain.MOLECULAR_STRUCTURE: (RepresentationKind.MOLECULAR_GRAPH, RepresentationKind.GEOMETRIC),
    ScientificDomain.MOLECULAR_INTERACTION: (RepresentationKind.MOLECULAR_GRAPH, RepresentationKind.GEOMETRIC),
    ScientificDomain.PROTEIN_DESIGN: (RepresentationKind.MOLECULAR_GRAPH,),
    ScientificDomain.MATERIALS: (RepresentationKind.MATERIALS_GRAPH,),
    ScientificDomain.PHYSICAL_SIMULATION: (RepresentationKind.FIELD, RepresentationKind.STATE_SPACE),
    ScientificDomain.GEOMETRIC: (RepresentationKind.GEOMETRIC, RepresentationKind.MOLECULAR_GRAPH),
    ScientificDomain.FORECASTING: (RepresentationKind.TRAJECTORY_ENSEMBLE, RepresentationKind.FIELD),
    ScientificDomain.LOCALIZATION: (RepresentationKind.FACTOR_GRAPH, RepresentationKind.STATE_SPACE),
}


_CANDIDATE_TRAITS: dict[RepresentationKind, dict[str, Any]] = {
    RepresentationKind.MOLECULAR_GRAPH: {
        "preserves": "atom identity, bond connectivity, element, charge, 3D coordinates",
        "discards": "electron density detail, solvent effects beyond the model",
        "invariances": "permutation of atom ordering (with attributes)",
        "limitations": "graph alone may not preserve all conformational information",
        "expected_cost": "low",
    },
    RepresentationKind.MATERIALS_GRAPH: {
        "preserves": "composition, periodicity, lattice parameters, site positions",
        "discards": "defects, surfaces, finite-size effects",
        "invariances": "lattice translations and site relabelling",
        "limitations": "periodic boundary conventions must be validated before property prediction",
        "expected_cost": "low",
    },
    RepresentationKind.FIELD: {
        "preserves": "spatial and temporal domain, units, boundary and initial conditions",
        "discards": "sub-grid structure below the discretization",
        "invariances": "grid translation under periodic conventions only",
        "limitations": "sampled field is not the continuous mathematical solution",
        "expected_cost": "medium",
    },
    RepresentationKind.GEOMETRIC: {
        "preserves": "points, vectors, frames, transform conventions",
        "discards": "non-geometric attributes",
        "invariances": "explicit per declared group",
        "limitations": "composed transforms require compatible frames",
        "expected_cost": "low",
    },
    RepresentationKind.FACTOR_GRAPH: {
        "preserves": "variables, measurement likelihoods, covariances, priors",
        "discards": "smooth dynamics between states unless modelled by factors",
        "invariances": "none assumed",
        "limitations": "covariance estimates depend on local linearization",
        "expected_cost": "medium",
    },
    RepresentationKind.TRAJECTORY_ENSEMBLE: {
        "preserves": "member identity, lead times, conditional distribution shape",
        "discards": "single deterministic future (deliberately)",
        "invariances": "member relabelling",
        "limitations": "ensemble must be calibrated per event, lead time, geography",
        "expected_cost": "medium",
    },
    RepresentationKind.STATE_SPACE: {
        "preserves": "state variables and parameters over time",
        "discards": "spatial structure unless encoded in the state",
        "invariances": "none assumed",
        "limitations": "state-space size grows with modelled detail",
        "expected_cost": "medium",
    },
}


class RepresentationBuilder:
    """Select and build the representation for a scientific task (Steps 8-15)."""

    def __init__(self, config: Any) -> None:
        """Bind the builder to pipeline configuration for discretization defaults."""
        self.config = config

    def build(self, task: ScientificTask) -> tuple[RepresentationArtifact, RepresentationDecision]:
        """Return the selected artifact and the decision record for the primary domain."""
        domain = task.domains[0]
        candidates = _CANDIDATE_LIBRARY.get(domain, (RepresentationKind.STATE_SPACE,))
        primary = candidates[0]
        decision = RepresentationDecision(
            candidates=tuple(
                {**_CANDIDATE_TRAITS[kind], "kind": kind.value} for kind in candidates
            ),
            selected=primary,
            rationale=(
                f"Selected {primary.value} as the simplest representation preserving the "
                f"essential structure of a {domain.value.lower().replace('_', ' ')} task."
            ),
        )
        builder = {
            RepresentationKind.FIELD: self._build_field,
            RepresentationKind.FACTOR_GRAPH: self._build_factor_graph,
            RepresentationKind.TRAJECTORY_ENSEMBLE: self._build_ensemble,
            RepresentationKind.GEOMETRIC: self._build_geometric,
            RepresentationKind.MOLECULAR_GRAPH: self._build_molecular,
            RepresentationKind.MATERIALS_GRAPH: self._build_materials,
            RepresentationKind.STATE_SPACE: self._build_state_space,
        }[primary]
        artifact = builder(task)
        return artifact, decision

    # -- Step 11: fields ------------------------------------------------------
    def _build_field(self, task: ScientificTask) -> RepresentationArtifact:
        """Build a field representation with domain, units, and conditions from the registry."""
        length = task.variable("L")
        horizon = task.variable("T")
        alpha = task.variable("alpha")
        initial = task.variable("u_initial")
        boundary = task.variable("u_boundary")
        x_max = float(length.value) if length and length.value is not None else 1.0
        t_max = float(horizon.value) if horizon and horizon.value is not None else 60.0
        boundary_si = boundary.unit.to_si(boundary.value) if boundary and boundary.value is not None else 0.0
        initial_si = initial.unit.to_si(initial.value) if initial and initial.value is not None else 0.0
        field = FieldRepresentation(
            quantity_symbol="u",
            quantity_name="temperature",
            unit=UNITS["K"],
            spatial_domain=(0.0, x_max),
            time_domain=(0.0, t_max),
            grid_size=int(self.config.collocation_points),
            time_steps=int(self.config.time_steps),
            boundary_condition="Dirichlet: u = boundary_value at x = 0 and x = L",
            initial_condition=f"uniform initial field u(x, 0) = {initial_si:g} K",
            boundary_value_si=boundary_si,
            initial_profile="uniform",
        )
        settings = {
            "grid_size": field.grid_size,
            "time_steps": field.time_steps,
            "alpha_si": float(alpha.value) if alpha and alpha.value is not None else None,
            "discretization": "uniform collocation grid",
            "feature_schema": "u: K",
        }
        return RepresentationArtifact(
            kind=RepresentationKind.FIELD,
            payload=field,
            preserves=["spatial domain", "temporal domain", "units", "boundary conditions",
                       "initial condition"],
            discards=["sub-grid structure below the discretization"],
            invariances=["none assumed (Dirichlet boundaries)"],
            limitations=["sampled field approximates the continuous solution"],
            expected_cost="medium",
            construction_settings=settings,
            source_hash=hashlib.sha256(task.question.encode("utf-8")).hexdigest()[:16],
            coordinate_frame=field.coordinate_frame,
            unit_symbol="K",
            domain=ScientificDomain.PHYSICAL_SIMULATION,
        )

    # -- Step 13: factor graphs ----------------------------------------------
    def _build_factor_graph(self, task: ScientificTask) -> RepresentationArtifact:
        """Build a pose-landmark factor-graph skeleton for localization tasks."""
        pose_count = 3
        landmark_count = 2
        variables = tuple(
            [f"x{i}" for i in range(pose_count)] + [f"l{j}" for j in range(landmark_count)]
        )
        graph = FactorGraphRepresentation(variables=variables)
        for index in range(pose_count):
            graph.variable_frames[f"x{index}"] = "world"
        for index in range(landmark_count):
            graph.variable_frames[f"l{index}"] = "world"
        artifact = RepresentationArtifact(
            kind=RepresentationKind.FACTOR_GRAPH,
            payload=graph,
            preserves=["pose and landmark variables", "measurement factor scopes",
                       "covariances", "frames"],
            discards=["continuous dynamics between poses"],
            invariances=[],
            limitations=["covariance estimates depend on local linearization"],
            expected_cost="medium",
            construction_settings={
                "pose_count": pose_count,
                "landmark_count": landmark_count,
                "solver": "gauss_newton_levenberg_marquardt",
                "measurement_models": "relative odometry and landmark bearings in the world frame",
            },
            source_hash=hashlib.sha256(task.question.encode("utf-8")).hexdigest()[:16],
            coordinate_frame="world",
            unit_symbol="m",
            domain=ScientificDomain.LOCALIZATION,
        )
        return artifact

    # -- Step 14: trajectory ensembles ---------------------------------------
    def _build_ensemble(self, task: ScientificTask) -> RepresentationArtifact:
        """Build an ensemble forecast representation from the config member count."""
        ensemble = TrajectoryEnsembleRepresentation(
            member_count=int(self.config.ensemble_members),
            lead_times=tuple(round(6.0 * step, 1) for step in range(1, 9)),
            quantity_name="wind speed",
            unit=UNITS["m_s"],
            current_state_summary="analysed state at initialization time",
        )
        return RepresentationArtifact(
            kind=RepresentationKind.TRAJECTORY_ENSEMBLE,
            payload=ensemble,
            preserves=["member identity", "lead times", "conditional distribution shape"],
            discards=["a single deterministic future (deliberately)"],
            invariances=["member relabelling"],
            limitations=["ensemble must be calibrated per event, lead time, and geography"],
            expected_cost="medium",
            construction_settings={"members": ensemble.member_count,
                                   "lead_times_hours": list(ensemble.lead_times)},
            source_hash=hashlib.sha256(task.question.encode("utf-8")).hexdigest()[:16],
            coordinate_frame=ensemble.coordinate_frame,
            unit_symbol="m/s",
            domain=ScientificDomain.FORECASTING,
        )

    # -- Step 12: geometric objects -------------------------------------------
    def _build_geometric(self, task: ScientificTask) -> RepresentationArtifact:
        """Build a 2D point-set representation for equivariance testing."""
        rng = np.random.default_rng(self.config.random_seed)
        base = rng.normal(loc=0.0, scale=1.0, size=(5, 2))
        points = tuple(tuple(row) for row in base)
        geometry = GeometricRepresentation(points=points)
        return RepresentationArtifact(
            kind=RepresentationKind.GEOMETRIC,
            payload=geometry,
            preserves=["point positions", "coordinate frame", "transform conventions"],
            discards=["non-geometric attributes"],
            invariances=["per declared group under equivariant models"],
            limitations=["composed transforms require compatible frames"],
            expected_cost="low",
            construction_settings={"point_count": len(points), "dimension": 2,
                                   "seed": self.config.random_seed},
            source_hash=hashlib.sha256(task.question.encode("utf-8")).hexdigest()[:16],
            coordinate_frame="world",
            unit_symbol="m",
            domain=ScientificDomain.GEOMETRIC,
        )

    # -- Step 9: molecular graphs ---------------------------------------------
    def _build_molecular(self, task: ScientificTask) -> RepresentationArtifact:
        """Build a reference molecular graph (small peptide-like fragment)."""
        atoms = (
            ("N", 0.0, (0.00, 0.00, 0.00)),
            ("C", 0.0, (0.15, 0.00, 0.00)),
            ("C", 0.0, (0.27, 0.12, 0.00)),
            ("O", 0.0, (0.24, 0.25, 0.00)),
            ("N", 0.0, (0.41, 0.08, 0.00)),
            ("C", 0.0, (0.54, 0.16, 0.00)),
        )
        bonds = ((0, 1, 1, "peptide"), (1, 2, 1, "single"), (2, 3, 2, "double"),
                 (2, 4, 1, "single"), (4, 5, 1, "single"))
        graph = MolecularGraph(atoms=atoms, bonds=bonds)
        return RepresentationArtifact(
            kind=RepresentationKind.MOLECULAR_GRAPH,
            payload=graph,
            preserves=["atom identity", "bond connectivity", "element", "charge",
                       "3D coordinates"],
            discards=["electron density detail", "solvent effects"],
            invariances=["permutation of atom ordering with attributes"],
            limitations=["graph alone may not preserve all conformational information"],
            expected_cost="low",
            construction_settings={"atom_count": len(atoms), "bond_count": len(bonds),
                                   "coordinates": "nm, cartesian"},
            source_hash=hashlib.sha256(task.question.encode("utf-8")).hexdigest()[:16],
            coordinate_frame=graph.coordinate_frame,
            unit_symbol="m",
            domain=task.domains[0],
        )

    # -- Step 10: materials graphs --------------------------------------------
    def _build_materials(self, task: ScientificTask) -> RepresentationArtifact:
        """Build a cubic perovskite BaTiO3 crystal graph as the reference structure."""
        lattice = ((0.4, 0.0, 0.0), (0.0, 0.4, 0.0), (0.0, 0.0, 0.4))
        sites = (
            ("Ba", 0.0, 0.0, 0.0),
            ("Ti", 0.5, 0.5, 0.5),
            ("O", 0.5, 0.5, 0.0),
            ("O", 0.5, 0.0, 0.5),
            ("O", 0.0, 0.5, 0.5),
        )
        graph = MaterialsGraph(
            composition=(("Ba", 1), ("Ti", 1), ("O", 3)),
            lattice=lattice,
            sites=sites,
        )
        return RepresentationArtifact(
            kind=RepresentationKind.MATERIALS_GRAPH,
            payload=graph,
            preserves=["composition", "periodicity", "lattice parameters", "site positions",
                       "neighbor relationships"],
            discards=["defects", "surfaces", "finite-size effects"],
            invariances=["lattice translations", "site relabelling"],
            limitations=["periodic boundary conventions must be validated before prediction"],
            expected_cost="low",
            construction_settings={
                "prototype": "cubic perovskite BaTiO3",
                "lattice_parameter_nm": 0.4,
                "periodic": True,
            },
            source_hash=hashlib.sha256(task.question.encode("utf-8")).hexdigest()[:16],
            coordinate_frame="fractional_unit_cell",
            unit_symbol="m",
            domain=ScientificDomain.MATERIALS,
        )

    # -- fallback: state space ------------------------------------------------
    def _build_state_space(self, task: ScientificTask) -> RepresentationArtifact:
        """Build a minimal state-space representation for unclassified domains."""
        symbols = [variable.symbol for variable in task.variables] or ["x"]
        return RepresentationArtifact(
            kind=RepresentationKind.STATE_SPACE,
            payload={"state_symbols": symbols},
            preserves=["state variables and parameters"],
            discards=["spatial structure not encoded in the state"],
            invariances=[],
            limitations=["generic state-space; domain detail is not represented"],
            expected_cost="low",
            construction_settings={"symbols": symbols},
            source_hash=hashlib.sha256(task.question.encode("utf-8")).hexdigest()[:16],
            coordinate_frame="unspecified",
            unit_symbol="1",
            domain=task.domains[0],
        )
