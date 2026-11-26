"""Model and solver registry with capability metadata and compatibility gating.

This module implements the ``ore_science.registry`` boundary from the integrated
algorithm. Every model family declares its domains, problem classes, input and output
representations, validation regime, interaction classes, risk, and cost. The registry
performs hard compatibility and safety gates first; the Section 5.8 decision score
ranks only candidates that pass those gates, so a cheap or high-scoring model can never
bypass mandatory validation.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .contracts import ProblemClass, ScientificDomain
from .representations import RepresentationKind

__all__ = [
    "SELECTION_WEIGHTS",
    "CandidateModels",
    "CompatibilityRejection",
    "ModelCapabilitySpec",
    "ModelRegistry",
    "selection_score",
]


@dataclass(frozen=True, slots=True)
class ModelCapabilitySpec:
    """Capability metadata for one model or solver family."""

    model_id: str
    family: str
    name: str
    version: str
    domains: tuple[ScientificDomain, ...]
    problem_classes: tuple[ProblemClass, ...]
    input_representation: RepresentationKind
    output_representation: RepresentationKind
    validation_regime: str
    quality: float
    calibration: float
    domain_compatibility: float
    risk: float
    resource_cost: float
    interaction_classes: tuple[str, ...] = ()
    external_service: bool = False
    availability: str = "local_reference"
    notes: str = ""

    def __post_init__(self) -> None:
        """Validate identifiers and score components."""
        if not self.model_id.strip() or not self.name.strip():
            raise ValueError("model id and name must be non-empty")
        if not self.version.strip():
            raise ValueError("model version must be non-empty")
        if not self.domains or not self.problem_classes:
            raise ValueError(f"model {self.model_id} must declare domains and problem classes")
        for label, value in (
            ("quality", self.quality),
            ("calibration", self.calibration),
            ("domain_compatibility", self.domain_compatibility),
            ("risk", self.risk),
            ("resource_cost", self.resource_cost),
        ):
            if not math.isfinite(float(value)) or value < 0.0:
                raise ValueError(f"{label} must be finite and non-negative")
        if self.availability not in ("local_reference", "external_service", "not_configured"):
            raise ValueError("availability must be a known service state")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the capability spec."""
        return {
            "model_id": self.model_id,
            "family": self.family,
            "name": self.name,
            "version": self.version,
            "domains": [domain.value for domain in self.domains],
            "problem_classes": [cls.value for cls in self.problem_classes],
            "input_representation": self.input_representation.value,
            "output_representation": self.output_representation.value,
            "validation_regime": self.validation_regime,
            "quality": self.quality,
            "calibration": self.calibration,
            "domain_compatibility": self.domain_compatibility,
            "risk": self.risk,
            "resource_cost": self.resource_cost,
            "interaction_classes": list(self.interaction_classes),
            "external_service": self.external_service,
            "availability": self.availability,
            "notes": self.notes,
        }


@dataclass(frozen=True, slots=True)
class CompatibilityRejection:
    """Why a model was rejected by the hard compatibility gates."""

    model_id: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize the rejection record."""
        return {"model_id": self.model_id, "reason": self.reason}


@dataclass(frozen=True, slots=True)
class CandidateModels:
    """Compatible model candidates plus rejection records for the audit trail."""

    models: tuple[ModelCapabilitySpec, ...]
    rejections: tuple[CompatibilityRejection, ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the candidate set."""
        return {
            "models": [model.to_dict() for model in self.models],
            "rejections": [rejection.to_dict() for rejection in self.rejections],
        }


# Selection score weights from Section 5.8: Score = w_q Q + w_c C + w_d D - w_r R - w_k K.
SELECTION_WEIGHTS: Mapping[str, float] = {
    "quality": 1.0,
    "calibration": 0.5,
    "domain_compatibility": 0.75,
    "risk": 0.6,
    "resource_cost": 0.2,
}


def selection_score(model: ModelCapabilitySpec) -> float:
    """Compute the Section 5.8 decision score for a gate-passing model."""
    return (
        SELECTION_WEIGHTS["quality"] * model.quality
        + SELECTION_WEIGHTS["calibration"] * model.calibration
        + SELECTION_WEIGHTS["domain_compatibility"] * model.domain_compatibility
        - SELECTION_WEIGHTS["risk"] * model.risk
        - SELECTION_WEIGHTS["resource_cost"] * model.resource_cost
    )


def _default_catalog() -> tuple[ModelCapabilitySpec, ...]:
    """Declare the local reference model and solver families."""
    return (
        ModelCapabilitySpec(
            model_id="solver.fd_heat_1d",
            family="numerical_solver",
            name="Crank-Nicolson finite-difference heat solver",
            version="1.0.0",
            domains=(ScientificDomain.PHYSICAL_SIMULATION,),
            problem_classes=(ProblemClass.FORWARD, ProblemClass.PARAMETER_ESTIMATION),
            input_representation=RepresentationKind.FIELD,
            output_representation=RepresentationKind.FIELD,
            validation_regime="numerical convergence study against analytic solution",
            quality=0.9,
            calibration=0.0,
            domain_compatibility=0.95,
            risk=0.1,
            resource_cost=0.25,
            notes="Conventional numerical reference used as the independent baseline.",
        ),
        ModelCapabilitySpec(
            model_id="pinn.basis_heat_1d",
            family="pinn",
            name="Basis-parametrized physics-informed surrogate",
            version="1.0.0",
            domains=(ScientificDomain.PHYSICAL_SIMULATION,),
            problem_classes=(ProblemClass.FORWARD,),
            input_representation=RepresentationKind.FIELD,
            output_representation=RepresentationKind.FIELD,
            validation_regime="compared with numerical reference solution and analytic test",
            quality=0.75,
            calibration=0.0,
            domain_compatibility=0.9,
            risk=0.35,
            resource_cost=0.4,
            notes="Research capability; low combined loss does not establish correctness.",
        ),
        ModelCapabilitySpec(
            model_id="operator.fourier_diag",
            family="neural_operator",
            name="Diagonal Fourier neural operator surrogate",
            version="1.0.0",
            domains=(ScientificDomain.PHYSICAL_SIMULATION,),
            problem_classes=(ProblemClass.FORWARD,),
            input_representation=RepresentationKind.FIELD,
            output_representation=RepresentationKind.FIELD,
            validation_regime="multi-resolution evaluation against reference solutions",
            quality=0.8,
            calibration=0.0,
            domain_compatibility=0.85,
            risk=0.3,
            resource_cost=0.3,
            notes="Linear diagonal spectral operator; periodic boundary convention.",
        ),
        ModelCapabilitySpec(
            model_id="forecast.ensemble_ar1",
            family="forecasting",
            name="Ensemble stochastic forecast model",
            version="1.0.0",
            domains=(ScientificDomain.FORECASTING,),
            problem_classes=(ProblemClass.PROBABILISTIC_PREDICTION,),
            input_representation=RepresentationKind.TRAJECTORY_ENSEMBLE,
            output_representation=RepresentationKind.TRAJECTORY_ENSEMBLE,
            validation_regime="calibration metrics against verifying observations",
            quality=0.7,
            calibration=0.8,
            domain_compatibility=0.9,
            risk=0.2,
            resource_cost=0.2,
            notes="Preserves ensemble members and probabilistic outputs.",
        ),
        ModelCapabilitySpec(
            model_id="factorgraph.gauss_newton",
            family="factor_graph",
            name="Gauss-Newton factor-graph optimizer",
            version="1.0.0",
            domains=(ScientificDomain.LOCALIZATION,),
            problem_classes=(ProblemClass.STATE_ESTIMATION,),
            input_representation=RepresentationKind.FACTOR_GRAPH,
            output_representation=RepresentationKind.FACTOR_GRAPH,
            validation_regime="synthetic ground-truth recovery and outlier robustness",
            quality=0.85,
            calibration=0.0,
            domain_compatibility=0.9,
            risk=0.2,
            resource_cost=0.3,
            notes="GTSAM-compatible interface implemented locally in numpy.",
        ),
        ModelCapabilitySpec(
            model_id="geometric.equivariant_points",
            family="geometric",
            name="SE(2)-equivariant point-set module",
            version="1.0.0",
            domains=(ScientificDomain.GEOMETRIC,),
            problem_classes=(ProblemClass.FORWARD,),
            input_representation=RepresentationKind.GEOMETRIC,
            output_representation=RepresentationKind.GEOMETRIC,
            validation_regime="explicit equivariance and invariance transformation tests",
            quality=0.8,
            calibration=0.0,
            domain_compatibility=0.95,
            risk=0.15,
            resource_cost=0.1,
            notes="Equivariance is tested, never assumed.",
        ),
        ModelCapabilitySpec(
            model_id="geometric.naive_mlp",
            family="geometric",
            name="Non-equivariant reference MLP",
            version="1.0.0",
            domains=(ScientificDomain.GEOMETRIC,),
            problem_classes=(ProblemClass.FORWARD,),
            input_representation=RepresentationKind.GEOMETRIC,
            output_representation=RepresentationKind.GEOMETRIC,
            validation_regime="equivariance error detection (expected to fail)",
            quality=0.5,
            calibration=0.0,
            domain_compatibility=0.5,
            risk=0.8,
            resource_cost=0.1,
            notes="Deliberately non-equivariant baseline for transformation testing.",
        ),
        ModelCapabilitySpec(
            model_id="molecular.structure_reference",
            family="molecular",
            name="Local reference molecular structure evaluator",
            version="1.0.0",
            domains=(ScientificDomain.MOLECULAR_STRUCTURE,),
            problem_classes=(ProblemClass.FORWARD,),
            input_representation=RepresentationKind.MOLECULAR_GRAPH,
            output_representation=RepresentationKind.MOLECULAR_GRAPH,
            validation_regime="geometric chemistry checks: bond lengths, angles, clashes",
            quality=0.6,
            calibration=0.0,
            domain_compatibility=0.8,
            risk=0.4,
            resource_cost=0.2,
            notes="Structure prediction service interface; local reference implementation.",
        ),
        ModelCapabilitySpec(
            model_id="molecular.interaction_reference",
            family="molecular",
            name="Lennard-Jones interaction evaluator",
            version="1.0.0",
            domains=(ScientificDomain.MOLECULAR_INTERACTION,),
            problem_classes=(ProblemClass.FORWARD,),
            input_representation=RepresentationKind.MOLECULAR_GRAPH,
            output_representation=RepresentationKind.MOLECULAR_GRAPH,
            validation_regime="energy sanity checks and interaction-class compatibility",
            quality=0.6,
            calibration=0.0,
            domain_compatibility=0.75,
            risk=0.4,
            resource_cost=0.2,
            interaction_classes=("protein_small_molecule", "small_molecule_small_molecule"),
            notes="Supports only the declared interaction classes.",
        ),
        ModelCapabilitySpec(
            model_id="molecular.protein_only_structure",
            family="molecular",
            name="Protein-only structure service stub",
            version="1.0.0",
            domains=(ScientificDomain.MOLECULAR_INTERACTION,),
            problem_classes=(ProblemClass.FORWARD,),
            input_representation=RepresentationKind.MOLECULAR_GRAPH,
            output_representation=RepresentationKind.MOLECULAR_GRAPH,
            validation_regime="service interface contract checks",
            quality=0.5,
            calibration=0.0,
            domain_compatibility=0.4,
            risk=0.6,
            resource_cost=0.3,
            interaction_classes=("protein_protein",),
            notes="Protein-only prediction service; must not receive other interaction classes.",
        ),
        ModelCapabilitySpec(
            model_id="molecular.rfdiffusion_reference",
            family="molecular",
            name="Generative backbone candidate engine (reference)",
            version="1.0.0",
            domains=(ScientificDomain.PROTEIN_DESIGN,),
            problem_classes=(ProblemClass.GENERATIVE_DESIGN,),
            input_representation=RepresentationKind.MOLECULAR_GRAPH,
            output_representation=RepresentationKind.MOLECULAR_GRAPH,
            validation_regime="candidate structure checks; output remains CANDIDATE",
            quality=0.55,
            calibration=0.0,
            domain_compatibility=0.8,
            risk=0.5,
            resource_cost=0.45,
            notes="RFdiffusion-style interface; generates bounded seeded candidate sets.",
        ),
        ModelCapabilitySpec(
            model_id="materials.gnn_reference",
            family="materials",
            name="Graph-based property predictor (reference)",
            version="1.0.0",
            domains=(ScientificDomain.MATERIALS,),
            problem_classes=(ProblemClass.FORWARD,),
            input_representation=RepresentationKind.MATERIALS_GRAPH,
            output_representation=RepresentationKind.MATERIALS_GRAPH,
            validation_regime="physical plausibility filters and independent computation",
            quality=0.6,
            calibration=0.0,
            domain_compatibility=0.8,
            risk=0.45,
            resource_cost=0.25,
            notes="GNoME-style interface; predicted stability is not a laboratory result.",
        ),
        ModelCapabilitySpec(
            model_id="materials.conditional_generator",
            family="materials",
            name="Conditional material generator (reference)",
            version="1.0.0",
            domains=(ScientificDomain.MATERIALS,),
            problem_classes=(ProblemClass.GENERATIVE_DESIGN,),
            input_representation=RepresentationKind.MATERIALS_GRAPH,
            output_representation=RepresentationKind.MATERIALS_GRAPH,
            validation_regime="charge balance, composition bounds, periodicity, duplicates",
            quality=0.55,
            calibration=0.0,
            domain_compatibility=0.8,
            risk=0.5,
            resource_cost=0.35,
            notes="MatterGen-style interface; candidates require downstream validation.",
        ),
        ModelCapabilitySpec(
            model_id="molecular.alphafold_service",
            family="molecular",
            name="AlphaFold-family structure service",
            version="external",
            domains=(ScientificDomain.MOLECULAR_STRUCTURE,),
            problem_classes=(ProblemClass.FORWARD,),
            input_representation=RepresentationKind.MOLECULAR_GRAPH,
            output_representation=RepresentationKind.MOLECULAR_GRAPH,
            validation_regime="service contract and independent structure checks",
            quality=0.95,
            calibration=0.0,
            domain_compatibility=0.95,
            risk=0.3,
            resource_cost=0.9,
            external_service=True,
            availability="not_configured",
            notes="Optional integration: requires licensing, API access, and compute budget.",
        ),
        ModelCapabilitySpec(
            model_id="forecast.graphcast_service",
            family="forecasting",
            name="GraphCast-family deterministic weather service",
            version="external",
            domains=(ScientificDomain.FORECASTING,),
            problem_classes=(ProblemClass.FORWARD,),
            input_representation=RepresentationKind.TRAJECTORY_ENSEMBLE,
            output_representation=RepresentationKind.TRAJECTORY_ENSEMBLE,
            validation_regime="held-out observations and benchmark comparison",
            quality=0.9,
            calibration=0.0,
            domain_compatibility=0.9,
            risk=0.3,
            resource_cost=0.95,
            external_service=True,
            availability="not_configured",
            notes="Optional integration; deterministic medium-range forecasts.",
        ),
        ModelCapabilitySpec(
            model_id="forecast.gencast_service",
            family="forecasting",
            name="GenCast-style ensemble weather service",
            version="external",
            domains=(ScientificDomain.FORECASTING,),
            problem_classes=(ProblemClass.PROBABILISTIC_PREDICTION,),
            input_representation=RepresentationKind.TRAJECTORY_ENSEMBLE,
            output_representation=RepresentationKind.TRAJECTORY_ENSEMBLE,
            validation_regime="calibration against verifying observations",
            quality=0.9,
            calibration=0.85,
            domain_compatibility=0.9,
            risk=0.3,
            resource_cost=0.95,
            external_service=True,
            availability="not_configured",
            notes="Optional integration; probabilistic ensemble trajectories.",
        ),
    )


class ModelRegistry:
    """Registry mapping task requirements to compatible model families (Section 3.D)."""

    def __init__(self, models: Sequence[ModelCapabilitySpec] | None = None) -> None:
        """Register the default catalog unless an explicit catalog is supplied."""
        self._models: dict[str, ModelCapabilitySpec] = {}
        for model in _default_catalog() if models is None else models:
            self.register(model)

    def register(self, model: ModelCapabilitySpec) -> None:
        """Add a model, rejecting duplicate identifiers."""
        if model.model_id in self._models:
            raise ValueError(f"duplicate model id {model.model_id!r}")
        self._models[model.model_id] = model

    def get(self, model_id: str) -> ModelCapabilitySpec:
        """Return a registered model by identifier."""
        model = self._models.get(model_id)
        if model is None:
            raise KeyError(f"unknown model {model_id!r}")
        return model

    def all_models(self) -> tuple[ModelCapabilitySpec, ...]:
        """Return every registered model."""
        return tuple(self._models.values())

    def find_compatible(
        self,
        domains: Sequence[ScientificDomain],
        problem_class: ProblemClass,
        representation: RepresentationKind,
        *,
        interaction_class: str | None = None,
    ) -> CandidateModels:
        """Run the hard compatibility gates and return passing candidates in score order.

        Gates: domain match, problem-class match, input-representation match, interaction
        class support (for molecular interaction tasks), and availability. Models marked
        ``not_configured`` (external services without access) are rejected with an
        explicit reason rather than silently dropped.
        """
        compatible: list[ModelCapabilitySpec] = []
        rejections: list[CompatibilityRejection] = []
        for model in self._models.values():
            if not any(domain in model.domains for domain in domains):
                rejections.append(
                    CompatibilityRejection(
                        model.model_id,
                        f"domain mismatch: model supports {model.domains!r}",
                    )
                )
                continue
            if problem_class not in model.problem_classes:
                rejections.append(
                    CompatibilityRejection(
                        model.model_id,
                        f"problem class mismatch: model supports {model.problem_classes!r}",
                    )
                )
                continue
            if model.input_representation != representation:
                rejections.append(
                    CompatibilityRejection(
                        model.model_id,
                        "input schema incompatible: model expects "
                        f"{model.input_representation.value} but the task provides "
                        f"{representation.value}",
                    )
                )
                continue
            if interaction_class and model.interaction_classes and interaction_class not in model.interaction_classes:
                rejections.append(
                    CompatibilityRejection(
                        model.model_id,
                        f"interaction class {interaction_class!r} not supported; model "
                        f"supports {model.interaction_classes!r}",
                    )
                )
                continue
            if model.availability == "not_configured":
                rejections.append(
                    CompatibilityRejection(
                        model.model_id,
                        "external service not configured: requires licensing, API access, "
                        "and compute budget before use",
                    )
                )
                continue
            compatible.append(model)
        ranked = sorted(compatible, key=selection_score, reverse=True)
        return CandidateModels(models=tuple(ranked), rejections=tuple(rejections))
