"""Domain-specialized scientific model adapters (Steps 16-34).

Local reference implementations of the specialized intelligence families named in the
integrated algorithm: numerical solvers, PINNs, neural operators, equivariant
geometric models, ensemble forecasting, factor-graph state estimation, molecular
intelligence, and materials intelligence. External services (AlphaFold-family,
RFdiffusion, GNoME/MatterGen, GraphCast/GenCast, GTSAM bindings) integrate through the
same interfaces when licensing, API access, compute budgets, and schema contracts are
configured; the model registry marks them ``not_configured`` until then.
"""
from __future__ import annotations

from .factor_graphs import (
    Factor,
    FactorGraphAdapter,
    FactorGraphBuilder,
    FactorGraphSolution,
    GaussNewtonFactorGraphOptimizer,
    build_synthetic_slam_problem,
)
from .forecasting import (
    EnsembleForecast,
    StochasticForecastSimulator,
    brier_score,
    calibration_report,
)
from .geometric import (
    EquivarianceResult,
    EquivarianceTestProtocol,
    EquivariantPointModule,
    NonEquivariantPointModule,
    Rotation2D,
)
from .materials import (
    ConditionalMaterialGenerator,
    GeneratedCandidate,
    LocalReferenceStabilityModel,
    MaterialsPlausibilityReport,
    MaterialsPropertyPrediction,
    check_charge_balance,
    evaluate_plausibility,
    structure_fingerprint,
)
from .molecular import (
    GenerativeDesignResult,
    InteractionEnergyReport,
    LocalReferenceDesignEngine,
    LocalReferenceInteractionService,
    LocalReferenceStructureService,
    StructureCheckReport,
)
from .numerical import CrankNicolsonHeatSolver, HeatSolverResult, analytic_heat_solution
from .operators import FourierModeOperator, OperatorFitReport
from .pinn import PDEProblem, PinnBasisSurrogate, PinnLossReport, PinnWeights, heat_residual_grid

__all__ = [
    "ConditionalMaterialGenerator",
    "CrankNicolsonHeatSolver",
    "EnsembleForecast",
    "EquivarianceResult",
    "EquivarianceTestProtocol",
    "EquivariantPointModule",
    "Factor",
    "FactorGraphAdapter",
    "FactorGraphBuilder",
    "FactorGraphSolution",
    "FourierModeOperator",
    "GaussNewtonFactorGraphOptimizer",
    "GeneratedCandidate",
    "GenerativeDesignResult",
    "HeatSolverResult",
    "InteractionEnergyReport",
    "LocalReferenceDesignEngine",
    "LocalReferenceInteractionService",
    "LocalReferenceStabilityModel",
    "LocalReferenceStructureService",
    "MaterialsPlausibilityReport",
    "MaterialsPropertyPrediction",
    "NonEquivariantPointModule",
    "OperatorFitReport",
    "PDEProblem",
    "PinnBasisSurrogate",
    "PinnLossReport",
    "PinnWeights",
    "Rotation2D",
    "StochasticForecastSimulator",
    "StructureCheckReport",
    "analytic_heat_solution",
    "brier_score",
    "build_synthetic_slam_problem",
    "calibration_report",
    "check_charge_balance",
    "evaluate_plausibility",
    "heat_residual_grid",
    "structure_fingerprint",
]
