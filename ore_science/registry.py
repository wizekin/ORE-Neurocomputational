"""Stage and operation registry for the 52-step scientific reasoning loop.

This module follows the repository convention of stage/operation registries (as in
``ore_csc.registry``). The 52 steps of the integrated algorithm are organized into
five phases and twelve architectural stations (the components A-H of Section 3 plus
the four routing stations for specialized intelligence). Steps execute in canonical
order; a step that cannot run records an explicit status instead of being skipped
silently.
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ALL_OPERATIONS", "OPERATION_REGISTRY", "STATION_REGISTRY", "OperationSpec", "StationSpec"]


@dataclass(frozen=True, slots=True)
class OperationSpec:
    """One numbered step of the integrated algorithm."""

    step: int
    operation_id: str
    phase: str
    station: str
    name: str
    mathematical_operation: str
    functional_responsibility: str


@dataclass(frozen=True, slots=True)
class StationSpec:
    """One architectural component (A-H plus routing stations)."""

    station_id: str
    name: str
    objective: str
    steps: tuple[int, ...]


def _op(
    step: int,
    phase: str,
    station: str,
    name: str,
    math_op: str,
    responsibility: str,
) -> OperationSpec:
    """Construct one operation spec with a stable identifier."""
    return OperationSpec(
        step=step,
        operation_id=f"SCI-{step:02d}",
        phase=phase,
        station=station,
        name=name,
        mathematical_operation=math_op,
        functional_responsibility=responsibility,
    )


OPERATION_REGISTRY: tuple[OperationSpec, ...] = (
    # Phase I - Interpret and formalize the scientific task (Steps 1-7)
    _op(1, "I", "A", "Detect scientific intent", "keyword-evidence domain routing",
        "Classify the task domain and preserve separate subtask identities."),
    _op(2, "I", "A", "Extract the objective", "objective-kind mapping",
        "Translate the request into an explicit scientific objective."),
    _op(3, "I", "A", "Extract variables and observables", "variable registry construction",
        "Record symbol, meaning, unit, value, domain, frame, and provenance per variable."),
    _op(4, "I", "A", "Identify constraints and assumptions", "constraint extraction",
        "Extract laws, boundary/initial conditions, and tag assumption sources."),
    _op(5, "I", "B", "Resolve units and dimensions", "dimensional homogeneity check",
        "Normalize units and reject dimensionally inconsistent formalizations."),
    _op(6, "I", "A", "Identify the mathematical problem class", "problem-class classification",
        "Determine forward, inverse, estimation, optimization, or generative class."),
    _op(7, "I", "A", "Establish evidence and precision requirements", "requirement specification",
        "Set accuracy, confidence, resolution, horizon, and consequence level."),
    # Phase II - Select the scientific representation (Steps 8-15)
    _op(8, "II", "C", "Build a representation decision record", "candidate comparison",
        "Compare candidates on preservation, cost, invariances, and limitations."),
    _op(9, "II", "C", "Represent molecular systems as attributed graphs", "graph construction",
        "Build molecular graph plus 3D coordinates with validated attributes."),
    _op(10, "II", "C", "Represent materials as crystal graphs", "periodic graph construction",
        "Build composition, lattice, sites, periodicity, and neighbour metadata."),
    _op(11, "II", "C", "Represent continuous quantities as fields", "field discretization",
        "Define domains, units, sampling, boundary and initial conditions."),
    _op(12, "II", "C", "Represent spatial objects geometrically", "geometric data selection",
        "Choose points, vectors, transforms, meshes with frame conventions."),
    _op(13, "II", "C", "Represent uncertain dependencies as factor graphs", "factor graph assembly",
        "Create variable and factor nodes with residual, covariance, and scope."),
    _op(14, "II", "C", "Represent forecasting as trajectory distributions", "ensemble construction",
        "Separate current state from predicted future-state distributions."),
    _op(15, "II", "C", "Preserve representation provenance", "hash and settings recording",
        "Store construction settings, schemas, and source data hashes."),
    # Phase III - Select and execute domain-specialized intelligence (Steps 16-34)
    _op(16, "III", "D", "Route molecular structure prediction", "service routing",
        "Route to a compatible structure service when one is configured."),
    _op(17, "III", "D", "Route molecular interaction questions", "interaction-class routing",
        "Select a model whose interaction class covers the request."),
    _op(18, "III", "D", "Route protein design to generative engine", "candidate generation routing",
        "Generate bounded, seeded candidate sets for de novo design."),
    _op(19, "III", "D", "Evaluate generated molecular candidates", "independent structure checks",
        "Apply structure, stability, and safety screening to candidates."),
    _op(20, "III", "D", "Route materials discovery", "predictor and generator routing",
        "Select graph property prediction or conditional generation."),
    _op(21, "III", "D", "Apply materials plausibility filters", "chemical constraint checks",
        "Check charge balance, composition, geometry, periodicity, duplicates."),
    _op(22, "III", "D", "Build the governing equation model", "equation assembly",
        "Define variables, parameters, conditions, and source terms."),
    _op(23, "III", "D", "Select PINN when equation-informed learning fits", "residual definition",
        "Define R[u] and confirm PDE, domain, and boundary correctness."),
    _op(24, "III", "D", "Construct the PINN objective with weights", "L = ld Ld + lp Lp + lb Lb + li Li",
        "Train with explicit weights and monitor component losses."),
    _op(25, "III", "D", "Select a neural operator for function-to-function problems",
        "operator selection", "Map input fields to output fields with operators."),
    _op(26, "III", "D", "Use Fourier neural operators where suitable", "spectral mode learning",
        "Learn transformations over retained physical-frequency modes."),
    _op(27, "III", "D", "Combine learned operators with numerical solvers", "hybrid execution",
        "Use numerical reference solutions and quantify discrepancies."),
    _op(28, "III", "D", "Select equivariant models for geometric tasks", "group selection",
        "Choose architecture with the exact required equivariance."),
    _op(29, "III", "D", "Test equivariance explicitly",
        "eps_eq = ||f(gx)-g f(x)|| / (||g f(x)||+eps)",
        "Measure normalized equivariance error over representative transformations."),
    _op(30, "III", "D", "Route global forecasting to weather models", "forecast model routing",
        "Choose deterministic or ensemble forecasting per the objective."),
    _op(31, "III", "D", "Preserve probabilistic weather outputs", "ensemble statistics",
        "Keep member identity; compute quantiles, probabilities, spread."),
    _op(32, "III", "D", "Use factor graphs for localization", "factor graph modelling",
        "Add pose, landmark, odometry, GPS, and prior factors with frames."),
    _op(33, "III", "D", "Optimize the factor graph",
        "x* = argmin sum_k r_k(x)^T Sigma_k^-1 r_k(x)",
        "Solve weighted nonlinear least squares with robust losses."),
    _op(34, "III", "D", "Use GTSAM-compatible backend through adapter", "backend adaptation",
        "Encapsulate factor creation, optimization, and marginals behind an interface."),
    # Phase IV - Independent validation and scientific uncertainty (Steps 35-44)
    _op(35, "IV", "F", "Run dimensional and algebraic validation", "unit and domain checks",
        "Check units, domains, equation structure, and parameter ranges."),
    _op(36, "IV", "F", "Validate numerical convergence", "refinement study",
        "Vary resolution and tolerance; report convergence explicitly."),
    _op(37, "IV", "F", "Check conservation laws and invariants", "invariant residuals",
        "Evaluate mass, energy, momentum, charge, normalization invariants."),
    _op(38, "IV", "F", "Compare against independent baselines", "reference comparison",
        "Compare with analytic or numerical references and record discrepancy."),
    _op(39, "IV", "F", "Quantify prediction uncertainty", "ensemble and sensitivity methods",
        "Store distributions, intervals, and calibration evidence."),
    _op(40, "IV", "F", "Calibrate probabilistic predictions", "Brier, reliability, coverage",
        "Evaluate calibration and sharpness of probabilistic outputs."),
    _op(41, "IV", "F", "Detect out-of-distribution inputs", "domain range checks",
        "Warn or fall back conservatively on out-of-domain inputs."),
    _op(42, "IV", "F", "Sensitivity and identifiability analysis", "J_ij = dy_i/dtheta_j",
        "Estimate sensitivities and flag non-identifiable parameters."),
    _op(43, "IV", "F", "Separate result evidence states", "status ladder",
        "Distinguish candidate, predicted, validated, experimentally confirmed."),
    _op(44, "IV", "F", "Apply domain-specific review gates", "gate selection by task class",
        "Require the validation gate appropriate to the domain."),
    # Phase V - Integrate results into ORE cognition and knowledge (Steps 45-52)
    _op(45, "V", "G", "Return a structured scientific result", "result assembly",
        "Include interpretation, representation, solver, uncertainty, limitations."),
    _op(46, "V", "G", "Publish through the Shared Cognitive Workspace", "workspace publication",
        "Publish immutable artifacts and structured, status-tagged observations."),
    _op(47, "V", "G", "Link results to USG concepts without rewriting",
        "candidate knowledge workflow", "Attach concepts via a reviewed candidate workflow."),
    _op(48, "V", "H", "Coordinate multi-domain scientific workflows", "typed decomposition",
        "Preserve per-stage assumptions; never merge incompatible uncertainties."),
    _op(49, "V", "H", "Allocate resources according to model requirements",
        "budget admission", "Reserve CPU, memory, time, and accelerators before dispatch."),
    _op(50, "V", "H", "Record complete scientific provenance", "manifest hashing",
        "Log data, model, solver, seeds, hardware, hashes, and validation."),
    _op(51, "V", "H", "Learn from prediction error without corrupting knowledge",
        "versioned evaluation", "Update selection statistics only under review."),
    _op(52, "V", "G", "Produce a calibrated explanation", "statement classification",
        "Distinguish derivation, model prediction, and empirical observation."),
)

ALL_OPERATIONS: tuple[OperationSpec, ...] = OPERATION_REGISTRY

STATION_REGISTRY: tuple[StationSpec, ...] = (
    StationSpec("A", "Scientific task interpreter",
                "Convert requests into structured scientific tasks.",
                tuple(range(1, 8))),
    StationSpec("B", "USG scientific knowledge interface",
                "Retrieve concepts, equations, units, and domain rules read-only.",
                (5,)),
    StationSpec("C", "Representation builder",
                "Construct the mathematical object appropriate to the task.",
                tuple(range(8, 16))),
    StationSpec("D", "Scientific model and solver registry",
                "Route to compatible specialized models and solvers.",
                tuple(range(16, 35))),
    StationSpec("E", "Execution and resource coordinator",
                "Admit tasks within CPU, memory, time, and accelerator budgets.",
                (49,)),
    StationSpec("F", "Validation and uncertainty engine",
                "Check units, invariants, convergence, calibration, and uncertainty.",
                tuple(range(35, 45))),
    StationSpec("G", "Scientific result workspace",
                "Publish validated results, candidates, uncertainty, and provenance.",
                (45, 46, 47, 52)),
    StationSpec("H", "Audit and learning loop",
                "Record provenance and improve selection under versioned review.",
                (48, 49, 50, 51)),
)


def registry_as_dicts() -> list[dict]:
    """Serialize the operation registry for logs and documentation."""
    return [
        {
            "step": spec.step,
            "operation_id": spec.operation_id,
            "phase": spec.phase,
            "station": spec.station,
            "name": spec.name,
            "mathematical_operation": spec.mathematical_operation,
            "functional_responsibility": spec.functional_responsibility,
        }
        for spec in OPERATION_REGISTRY
    ]
