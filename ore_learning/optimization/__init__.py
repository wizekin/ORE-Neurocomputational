"""QUBO formulation, solver adapters and independent feasibility checks (Section 5)."""
from .qubo import (  # noqa: F401
    CombinatorialProblem,
    OptimizationPolicyError,
    QuboFormulation,
    formulate_qubo,
    verify_small_instance,
)
from .solvers import (  # noqa: F401
    ConventionalBaseline,
    IsingAdapter,
    OptimizationPipeline,
    SolverResult,
)
from .verify import (  # noqa: F401
    OptimizationPolicyError,
    OptimizationVerification,
    apply_to_cros_or_policy,
    decode_assignment,
    independent_feasibility_check,
)

__all__ = (
    "CombinatorialProblem",
    "ConventionalBaseline",
    "IsingAdapter",
    "OptimizationPipeline",
    "OptimizationPolicyError",
    "OptimizationVerification",
    "QuboFormulation",
    "SolverResult",
    "apply_to_cros_or_policy",
    "decode_assignment",
    "formulate_qubo",
    "independent_feasibility_check",
    "verify_small_instance",
)
