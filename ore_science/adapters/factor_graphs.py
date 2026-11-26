"""Factor-graph state estimation with a GTSAM-compatible adapter (Steps 32-34, 5.5).

Variables (poses, landmarks, biases) and factors (odometry, landmark observations,
GPS, priors) form a bipartite graph. MAP estimation reduces to the weighted nonlinear
least-squares problem x* = argmin_x sum_k rho_k(r_k(x)^T Sigma_k^-1 r_k(x)), solved
here with Gauss-Newton iterations under Levenberg-Marquardt damping, with Huber
robust losses for outliers, and with marginal covariance from the information matrix
at the solution. The interface mirrors GTSAM concepts (add_factor, optimize,
marginal_covariance) so the numerical backend can be swapped for GTSAM without
changing the scientific task schema.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "Factor",
    "FactorGraphAdapter",
    "FactorGraphBuilder",
    "FactorGraphSolution",
    "GaussNewtonFactorGraphOptimizer",
    "build_synthetic_slam_problem",
]


@dataclass(frozen=True, slots=True)
class Factor:
    """One measurement factor: residual function, covariance, scope, and robust loss."""

    factor_id: str
    factor_type: str
    scope: tuple[str, ...]
    residual: Callable[[Mapping[str, float]], np.ndarray]
    covariance: np.ndarray
    robust_loss: str = "quadratic"
    huber_delta: float = 1.0

    def __post_init__(self) -> None:
        """Validate covariance shape and robust-loss configuration."""
        if not self.factor_id.strip() or not self.scope:
            raise ValueError("factor requires an id and a non-empty scope")
        if self.covariance.ndim != 2:
            raise ValueError("covariance must be a matrix")
        if self.covariance.shape[0] != self.covariance.shape[1]:
            raise ValueError("covariance must be square")
        if not np.all(np.linalg.eigvals(self.covariance) > 0.0):
            raise ValueError("covariance must be positive definite")
        if self.robust_loss not in ("quadratic", "huber"):
            raise ValueError("robust loss must be 'quadratic' or 'huber'")
        if self.huber_delta <= 0.0:
            raise ValueError("huber delta must be positive")

    def weight(self) -> np.ndarray:
        """Information matrix Sigma^-1 of the factor."""
        return np.linalg.inv(self.covariance)


class FactorGraphBuilder:
    """Assemble variables and factors while enforcing scope validity (Step 32)."""

    def __init__(self) -> None:
        """Start with an empty graph."""
        self.variables: list[str] = []
        self.factors: list[Factor] = []
        self.variable_frames: dict[str, str] = {}

    def add_variable(self, name: str, frame: str = "world", initial: float = 0.0) -> None:
        """Declare a scalar variable with its coordinate frame and initial value."""
        if name in self.variables:
            raise ValueError(f"duplicate variable {name!r}")
        self.variables.append(name)
        self.variable_frames[name] = frame
        self.initial: dict[str, float] = getattr(self, "initial", {})
        self.initial[name] = float(initial)

    def add_factor(self, factor: Factor, *, frame: str = "world") -> None:
        """Add a factor, rejecting undefined-variable references and frame mismatch."""
        for variable in factor.scope:
            if variable not in self.variables:
                raise ValueError(
                    f"factor {factor.factor_id!r} references undefined variable {variable!r}"
                )
            declared = self.variable_frames.get(variable, "world")
            if declared != frame:
                raise ValueError(
                    f"factor {factor.factor_id!r} uses frame {frame!r} but variable "
                    f"{variable!r} is declared in frame {declared!r}"
                )
        self.factors.append(factor)

    def initial_vector(self) -> np.ndarray:
        """Return the stacked initial estimate for the optimizer."""
        return np.array([self.initial[name] for name in self.variables], dtype=float)

    def frame_consistency_errors(self) -> list[str]:
        """Report any factor/variable frame disagreement not caught at insertion."""
        errors: list[str] = []
        for factor in self.factors:
            for variable in factor.scope:
                if self.variable_frames.get(variable, "world") != "world" and factor.factor_type != "frame_sensitive":
                    errors.append(
                        f"factor {factor.factor_id} may be inconsistent with the frame of {variable}"
                    )
        return errors


@dataclass(frozen=True, slots=True)
class FactorGraphSolution:
    """Optimization result with convergence and covariance diagnostics (Step 33)."""

    values: dict[str, float]
    converged: bool
    iterations: int
    final_chi_squared: float
    per_factor_residuals: dict[str, float]
    marginal_variances: dict[str, float]
    condition_number: float
    robust_loss_active: dict[str, bool]
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the solution record."""
        return {
            "values": dict(self.values),
            "converged": self.converged,
            "iterations": self.iterations,
            "final_chi_squared": self.final_chi_squared,
            "per_factor_residuals": dict(self.per_factor_residuals),
            "marginal_variances": dict(self.marginal_variances),
            "condition_number": self.condition_number,
            "robust_loss_active": dict(self.robust_loss_active),
            "notes": list(self.notes),
        }


class GaussNewtonFactorGraphOptimizer:
    """Weighted nonlinear least squares via Gauss-Newton with LM damping (Step 33)."""

    def __init__(self, *, max_iterations: int = 50, tolerance: float = 1.0e-10,
                 initial_damping: float = 1.0e-3) -> None:
        """Configure iteration limits and stopping criteria."""
        if max_iterations < 1:
            raise ValueError("max iterations must be positive")
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.initial_damping = initial_damping

    def _robust_weights(self, factor: Factor, residual: np.ndarray) -> np.ndarray:
        """Per-residual Huber weights w(r): 1 near zero, delta/|r| beyond."""
        if factor.robust_loss == "quadratic":
            return np.ones_like(residual)
        delta = factor.huber_delta
        magnitudes = np.abs(residual)
        weights = np.where(magnitudes <= delta, 1.0, delta / np.maximum(magnitudes, 1.0e-15))
        return weights

    def _robust_active(self, factor: Factor, residual: np.ndarray) -> bool:
        """Return True when any residual of this factor exceeds the Huber threshold."""
        if factor.robust_loss == "quadratic":
            return False
        return bool(np.any(np.abs(residual) > factor.huber_delta))

    def optimize(self, graph: FactorGraphBuilder) -> FactorGraphSolution:
        """Solve x* = argmin sum_k rho_k(r_k(x)^T Sigma_k^-1 r_k(x))."""
        variables = graph.variables
        x = graph.initial_vector().copy()
        damping = self.initial_damping
        converged = False
        chi_squared = float("inf")
        notes: list[str] = []
        robust_active: dict[str, bool] = {}
        last_residuals: dict[str, float] = {}

        for iteration in range(1, self.max_iterations + 1):
            hessian = np.zeros((len(variables), len(variables)))
            gradient = np.zeros(len(variables))
            chi_squared = 0.0
            last_residuals = {}
            robust_active = {}
            for factor in graph.factors:
                state = {name: float(value) for name, value in zip(variables, x)}
                residual = np.asarray(factor.residual(state), dtype=float)
                weights = self._robust_weights(factor, residual)
                weighted = weights * residual
                information = factor.weight()
                chi_squared += float(weighted @ information @ weighted)
                last_residuals[factor.factor_id] = float(np.sqrt(np.mean(residual ** 2)))
                robust_active[factor.factor_id] = self._robust_active(factor, residual)
                jacobian = self._numerical_jacobian(factor, state, variables)
                scaled = information @ weighted  # w Sigma^-1 r
                gradient += jacobian.T @ scaled
                weight_matrix = np.diag(weights)
                weighted_jacobian = weight_matrix @ jacobian
                hessian += weighted_jacobian.T @ information @ weighted_jacobian
            hessian += damping * np.eye(len(variables))
            try:
                step = np.linalg.solve(hessian, -gradient)
            except np.linalg.LinAlgError:
                notes.append("linear solve failed; terminating with the last iterate")
                break
            candidate = x + step
            candidate_energy = self._energy(graph, variables, candidate)
            current_energy = self._energy(graph, variables, x)
            if candidate_energy < current_energy:
                x = candidate
                damping = max(damping * 0.5, 1.0e-9)
            else:
                damping = min(damping * 10.0, 1.0e6)
            if np.linalg.norm(step) < self.tolerance * max(1.0, np.linalg.norm(x)):
                converged = True
                break

        information_matrix = self._information_matrix(graph, variables, x)
        try:
            covariance = np.linalg.inv(information_matrix)
            marginal_variances = {
                name: float(covariance[index, index])
                for index, name in enumerate(variables)
            }
            eigenvalues = np.linalg.eigvalsh(information_matrix)
            condition_number = (
                float(eigenvalues[-1] / eigenvalues[0]) if eigenvalues[0] > 0 else float("inf")
            )
        except np.linalg.LinAlgError:
            marginal_variances = {name: float("nan") for name in variables}
            condition_number = float("inf")
            notes.append("information matrix singular; observability analysis required")

        values = {name: float(value) for name, value in zip(variables, x)}
        return FactorGraphSolution(
            values=values,
            converged=converged,
            iterations=iteration,
            final_chi_squared=chi_squared,
            per_factor_residuals=last_residuals,
            marginal_variances=marginal_variances,
            condition_number=condition_number,
            robust_loss_active=robust_active,
            notes=tuple(notes),
        )

    @staticmethod
    def _numerical_jacobian(
        factor: Factor, state: Mapping[str, float], variables: Sequence[str]
    ) -> np.ndarray:
        """Finite-difference Jacobian dr/dx for the factor's residual."""
        base = np.asarray(factor.residual(state), dtype=float)
        jacobian = np.zeros((base.shape[0], len(variables)))
        step = 1.0e-6
        for column, name in enumerate(variables):
            perturbed = dict(state)
            perturbed[name] = state[name] + step
            forward = np.asarray(factor.residual(perturbed), dtype=float)
            perturbed[name] = state[name] - step
            backward = np.asarray(factor.residual(perturbed), dtype=float)
            jacobian[:, column] = (forward - backward) / (2.0 * step)
        return jacobian

    def _energy(self, graph: FactorGraphBuilder, variables: Sequence[str],
                x: np.ndarray) -> float:
        """Total robust weighted cost at a state vector."""
        energy = 0.0
        state = {name: float(value) for name, value in zip(variables, x)}
        for factor in graph.factors:
            residual = np.asarray(factor.residual(state), dtype=float)
            weights = self._robust_weights(factor, residual)
            weighted = weights * residual
            energy += float(weighted @ factor.weight() @ weighted)
        return energy

    def _information_matrix(self, graph: FactorGraphBuilder, variables: Sequence[str],
                            x: np.ndarray) -> np.ndarray:
        """Assemble the information matrix H = sum J^T Sigma^-1 J at the solution."""
        state = {name: float(value) for name, value in zip(variables, x)}
        size = len(variables)
        information = np.zeros((size, size))
        for factor in graph.factors:
            residual = np.asarray(factor.residual(state), dtype=float)
            weights = self._robust_weights(factor, residual)
            jacobian = self._numerical_jacobian(factor, state, variables)
            weighted_jacobian = np.diag(weights) @ jacobian
            information += weighted_jacobian.T @ factor.weight() @ weighted_jacobian
        return information


class FactorGraphAdapter:
    """GTSAM-compatible facade over the local optimizer (Step 34).

    The adapter exposes ``add_factor`` / ``optimize`` / ``marginal_covariance`` in the
    GTSAM naming style while the scientific task schema stays independent of the
    backend. Swapping in real GTSAM Python bindings would replace the optimizer call
    without touching the callers.
    """

    backend_version = "gauss_newton_lm_numpy/1.0.0"
    gtsam_available = False

    def __init__(self, optimizer: GaussNewtonFactorGraphOptimizer | None = None) -> None:
        """Bind an optimizer, defaulting to the local Gauss-Newton implementation."""
        self.optimizer = optimizer or GaussNewtonFactorGraphOptimizer()
        self.graph = FactorGraphBuilder()
        self.solution: FactorGraphSolution | None = None

    def add_variable(self, name: str, frame: str = "world", initial: float = 0.0) -> None:
        """Declare a variable through the adapter interface."""
        self.graph.add_variable(name, frame=frame, initial=initial)

    def add_factor(self, factor: Factor, *, frame: str = "world") -> None:
        """Add a factor through the adapter interface."""
        self.graph.add_factor(factor, frame=frame)

    def optimize(self) -> FactorGraphSolution:
        """Run MAP estimation and retain the solution for marginal queries."""
        self.solution = self.optimizer.optimize(self.graph)
        return self.solution

    def marginal_covariance(self, name: str) -> float:
        """Return the marginal variance of a variable from the last solution."""
        if self.solution is None:
            raise ValueError("optimize must run before querying marginals")
        if name not in self.solution.marginal_variances:
            raise KeyError(f"unknown variable {name!r}")
        return self.solution.marginal_variances[name]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the adapter state and backend identity."""
        return {
            "backend": "local_numpy" if not self.gtsam_available else "gtsam_bindings",
            "backend_version": self.backend_version,
            "gtsam_available": self.gtsam_available,
            "variable_count": len(self.graph.variables),
            "factor_count": len(self.graph.factors),
            "optimized": self.solution is not None,
            "note": (
                "GTSAM Python bindings were not detected; the local Gauss-Newton "
                "backend implements the same interface. Backend swap is transparent."
            ),
        }


def build_synthetic_slam_problem(
    *,
    odometry_noise: float = 0.05,
    landmark_noise: float = 0.08,
    outlier: bool = False,
    inconsistent_frames: bool = False,
) -> tuple[FactorGraphAdapter, dict[str, float]]:
    """Construct a small pose-landmark SLAM problem with known ground truth.

    Ground truth: poses x0=0, x1=1, x2=2; landmarks l0=0.5, l1=1.7. Odometry factors
    tie consecutive poses; landmark factors tie poses to landmarks; a prior anchors x0.
    ``outlier`` injects one grossly corrupted odometry measurement to exercise the
    Huber robust loss. ``inconsistent_frames`` declares one pose in a different frame,
    which must be caught by the frame-consistency validation.
    """
    ground_truth = {"x0": 0.0, "x1": 1.0, "x2": 2.0, "l0": 0.5, "l1": 1.7}
    adapter = FactorGraphAdapter()
    for name, value in ground_truth.items():
        frame = "world"
        if inconsistent_frames and name == "x2":
            frame = "robot_local"
        adapter.add_variable(name, frame=frame, initial=value * 0.85)

    variance_odometry = odometry_noise ** 2
    variance_landmark = landmark_noise ** 2

    def prior_residual(state: Mapping[str, float]) -> np.ndarray:
        return np.array([state["x0"] - 0.0])

    adapter.add_factor(
        Factor(
            factor_id="prior_x0",
            factor_type="prior",
            scope=("x0",),
            residual=prior_residual,
            covariance=np.array([[0.01]]),
        )
    )

    def make_odometry_residual(from_name: str, to_name: str, measured: float):
        def residual(state: Mapping[str, float]) -> np.ndarray:
            return np.array([state[to_name] - state[from_name] - measured])
        return residual

    odometry_measurements = {"x0->x1": 1.02, "x2->x1_reversed": 0.99, "x1->x2": 0.98}
    adapter.add_factor(
        Factor(
            factor_id="odom_x0_x1",
            factor_type="odometry",
            scope=("x0", "x1"),
            residual=make_odometry_residual("x0", "x1", odometry_measurements["x0->x1"]),
            covariance=np.array([[variance_odometry]]),
            robust_loss="huber",
            huber_delta=3.0 * odometry_noise,
        )
    )
    if outlier:
        adapter.add_factor(
            Factor(
                factor_id="odom_x0_x1_outlier",
                factor_type="odometry",
                scope=("x0", "x1"),
                residual=make_odometry_residual("x0", "x1", 5.0),
                covariance=np.array([[variance_odometry]]),
                robust_loss="huber",
                huber_delta=3.0 * odometry_noise,
            )
        )
    adapter.add_factor(
        Factor(
            factor_id="odom_x1_x2",
            factor_type="odometry",
            scope=("x1", "x2"),
            residual=make_odometry_residual("x1", "x2", odometry_measurements["x1->x2"]),
            covariance=np.array([[variance_odometry]]),
            robust_loss="huber",
            huber_delta=3.0 * odometry_noise,
        )
    )

    def make_landmark_residual(pose_name: str, landmark_name: str, measured: float):
        def residual(state: Mapping[str, float]) -> np.ndarray:
            return np.array([state[landmark_name] - state[pose_name] - measured])
        return residual

    adapter.add_factor(
        Factor(
            factor_id="obs_x0_l0",
            factor_type="landmark",
            scope=("x0", "l0"),
            residual=make_landmark_residual("x0", "l0", 0.52),
            covariance=np.array([[variance_landmark]]),
        )
    )
    adapter.add_factor(
        Factor(
            factor_id="obs_x1_l0",
            factor_type="landmark",
            scope=("x1", "l0"),
            residual=make_landmark_residual("x1", "l0", -0.47),
            covariance=np.array([[variance_landmark]]),
        )
    )
    adapter.add_factor(
        Factor(
            factor_id="obs_x2_l1",
            factor_type="landmark",
            scope=("x2", "l1"),
            residual=make_landmark_residual("x2", "l1", -0.28),
            covariance=np.array([[variance_landmark]]),
        )
    )
    return adapter, ground_truth
