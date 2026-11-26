"""Physics-Informed Neural Network adapter: residuals, objectives, and training (Steps 23-24).

The PINN family here is a basis-parametrized field surrogate u_phi = sum_k c_k psi_k(x,t)
trained by minimizing the Section 5.1 objective

    L(phi) = lambda_d L_data + lambda_p L_physics + lambda_b L_boundary + lambda_i L_initial

with residuals evaluated by finite differences on a collocation grid. Component losses
are reported independently so that a small combined loss cannot conceal a large error
in one critical component. Training uses scipy optimizers with numerical gradients;
automatic differentiation is not required by the contract, and the adapter documents
this explicitly. Low residual at sampled points is never treated as proof of
correctness over the full domain: the pipeline always compares PINN output with an
independent numerical reference solution.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize

__all__ = [
    "PDEProblem",
    "PinnBasisSurrogate",
    "PinnLossReport",
    "PinnWeights",
    "heat_residual_grid",
]


@dataclass(frozen=True, slots=True)
class PDEProblem:
    """A 1-D parabolic PDE with parameters, domain, and boundary/initial specifications."""

    name: str
    alpha: float
    length: float
    time_horizon: float
    u_boundary: float
    u_initial: float
    equation_id: str = "EQ-HEAT-1D"
    declared_form: str = "du/dt = alpha * d2u/dx2"

    def __post_init__(self) -> None:
        """Validate the physical parameters of the problem."""
        if self.alpha <= 0.0 or self.length <= 0.0 or self.time_horizon <= 0.0:
            raise ValueError("alpha, length, and time horizon must be positive")

    def exact_solution(self, x: np.ndarray, t: float) -> np.ndarray:
        """Analytic solution used as an independent test case (not for training)."""
        total = np.zeros_like(x, dtype=float)
        for n in range(1, 61, 2):
            coefficient = 4.0 * self.u_initial / (math.pi * n)
            decay = math.exp(-self.alpha * (math.pi * n / self.length) ** 2 * t)
            total += coefficient * np.sin(math.pi * n * x / self.length) * decay
        return total


@dataclass(frozen=True, slots=True)
class PinnWeights:
    """Explicit loss weights lambda_d, lambda_p, lambda_b, lambda_i (Step 24)."""

    data: float = 1.0
    physics: float = 1.0
    boundary: float = 10.0
    initial: float = 10.0

    def __post_init__(self) -> None:
        """Weights must be finite and non-negative."""
        for name in ("data", "physics", "boundary", "initial"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"weight {name} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class PinnLossReport:
    """Independently reported loss components for one evaluation."""

    data_loss: float
    physics_loss: float
    boundary_loss: float
    initial_loss: float
    total_loss: float
    weights: PinnWeights

    def to_dict(self) -> dict[str, Any]:
        """Serialize the loss decomposition."""
        return {
            "data_loss": self.data_loss,
            "physics_loss": self.physics_loss,
            "boundary_loss": self.boundary_loss,
            "initial_loss": self.initial_loss,
            "total_loss": self.total_loss,
            "weights": {
                "lambda_d": self.weights.data,
                "lambda_p": self.weights.physics,
                "lambda_b": self.weights.boundary,
                "lambda_i": self.weights.initial,
            },
        }


def heat_residual_grid(
    field: np.ndarray, x_grid: np.ndarray, times: np.ndarray, alpha: float
) -> np.ndarray:
    """Evaluate R = du/dt - alpha d2u/dx2 by finite differences on the grid.

    ``field`` has shape (T, X). Interior points only; boundaries are excluded because
    they are handled by the boundary loss.
    """
    if field.ndim != 2 or field.shape[0] != len(times) or field.shape[1] != len(x_grid):
        raise ValueError("field shape must match (times, x_grid)")
    dx = x_grid[1] - x_grid[0]
    interior_x = slice(1, -1)
    laplacian = np.zeros_like(field)
    laplacian[:, interior_x] = (
        field[:, :-2] - 2.0 * field[:, 1:-1] + field[:, 2:]
    ) / (dx * dx)
    du_dt = np.gradient(field, times, axis=0, edge_order=2)
    residual = du_dt - alpha * laplacian
    return residual[:, interior_x]


class PinnBasisSurrogate:
    """Basis-parametrized PINN surrogate for the 1-D heat equation.

    The surrogate is u_phi(x, t) = u_boundary + (u_initial - u_boundary) * g_phi(x, t)
    with g_phi(x, t) = sum_k a_k exp(-d_k t) sin(n_k pi x / L). The initial amplitudes
    a_k and decay rates d_k are trainable: d_k is initialized at half the analytic
    modal decay rate so training must genuinely learn the physics from the PDE
    residual. The basis satisfies zero Dirichlet boundaries by construction; the
    boundary loss is still computed and reported independently rather than assumed.
    """

    def __init__(self, problem: PDEProblem, basis_size: int, seed: int = 0) -> None:
        """Initialize modal amplitudes and decay-rate parameters deterministically."""
        if basis_size < 1:
            raise ValueError("basis size must be positive")
        self.problem = problem
        self.basis_size = basis_size
        self.modes = np.arange(1, basis_size + 1)
        # Spectral amplitudes of a constant initial profile (odd modes only).
        self.amplitudes = np.where(
            self.modes % 2 == 1,
            4.0 / (np.pi * self.modes),
            0.0,
        )
        analytic_rates = self.problem.alpha * (np.pi * self.modes / self.problem.length) ** 2
        # Deliberately off by a factor of two: training must correct these rates.
        self.decay_rates = 0.5 * analytic_rates
        self.seed = seed
        self.training_report: dict[str, Any] = {}
        self.model_version = "pinn.basis_heat_1d/1.0.0"

    def _pack(self) -> np.ndarray:
        """Pack trainable parameters into one flat vector."""
        return np.concatenate([self.amplitudes, self.decay_rates])

    def _unpack(self, flat: np.ndarray) -> None:
        """Unpack a flat parameter vector into amplitudes and decay rates."""
        size = self.basis_size
        self.amplitudes = np.asarray(flat[:size], dtype=float)
        self.decay_rates = np.asarray(flat[size:], dtype=float)

    # -- field evaluation -----------------------------------------------------
    def field_on_grid(self, x_grid: np.ndarray, times: np.ndarray) -> np.ndarray:
        """Evaluate the surrogate on a space-time collocation grid."""
        scale = self.problem.u_initial - self.problem.u_boundary
        spatial = np.sin(
            np.pi * np.outer(self.modes, x_grid) / self.problem.length
        )  # (basis, X)
        decay = np.exp(-np.outer(times, self.decay_rates))  # (T, basis)
        modal = decay[:, :, None] * spatial[None, :, :]  # (T, basis, X)
        g = np.einsum("k,tkx->tx", self.amplitudes, modal)
        return self.problem.u_boundary + scale * g

    def evaluate_losses(
        self,
        x_grid: np.ndarray,
        times: np.ndarray,
        weights: PinnWeights,
        *,
        data_field: np.ndarray | None = None,
        alpha_override: float | None = None,
    ) -> PinnLossReport:
        """Report data, physics, boundary, and initial losses independently.

        ``data_field`` supplies optional observations; when absent the data term uses
        the analytic initial profile sampled on the grid. ``alpha_override`` lets the
        caller test a deliberately incorrect PDE (wrong parameter) - used by the
        wrong-PDE detection test.
        """
        field = self.field_on_grid(x_grid, times)
        alpha = self.problem.alpha if alpha_override is None else alpha_override
        residual = heat_residual_grid(field, x_grid, times, alpha)
        physics_loss = float(np.mean(residual ** 2))

        boundary_values = np.concatenate([field[:, 0], field[:, -1]])
        boundary_loss = float(np.mean((boundary_values - self.problem.u_boundary) ** 2))

        initial_row = self.field_on_grid(x_grid, np.array([times[0]]))[0]
        expected_initial = np.full_like(x_grid, self.problem.u_initial, dtype=float)
        initial_loss = float(np.mean((initial_row - expected_initial) ** 2))

        if data_field is not None:
            data_loss = float(np.mean((field - data_field) ** 2))
        else:
            data_loss = float(np.mean((initial_row - expected_initial) ** 2)) * 0.5

        total = (
            weights.data * data_loss
            + weights.physics * physics_loss
            + weights.boundary * boundary_loss
            + weights.initial * initial_loss
        )
        return PinnLossReport(
            data_loss=data_loss,
            physics_loss=physics_loss,
            boundary_loss=boundary_loss,
            initial_loss=initial_loss,
            total_loss=total,
            weights=weights,
        )

    # -- training (Step 24) ---------------------------------------------------
    def train(
        self,
        x_grid: np.ndarray,
        times: np.ndarray,
        weights: PinnWeights,
        *,
        max_iterations: int = 200,
    ) -> dict[str, Any]:
        """Fit amplitudes and decay rates by minimizing the weighted PINN objective.

        Optimization uses scipy's BFGS with finite-difference gradients, which the
        Section 5.7 formulation explicitly permits when derivatives are unavailable.
        The training report records the loss decomposition before and after training
        so that a small combined loss cannot conceal a large component error.
        """
        before = self.evaluate_losses(x_grid, times, weights)

        def objective(flat: np.ndarray) -> float:
            self._unpack(flat)
            return self.evaluate_losses(x_grid, times, weights).total_loss

        result = minimize(
            objective,
            self._pack(),
            method="BFGS",
            options={"maxiter": max_iterations, "gtol": 1.0e-10},
        )
        self._unpack(np.asarray(result.x, dtype=float))
        after = self.evaluate_losses(x_grid, times, weights)
        analytic_rates = self.problem.alpha * (np.pi * self.modes / self.problem.length) ** 2
        self.training_report = {
            "optimizer": "scipy.optimize.minimize/BFGS",
            "gradient": "finite_difference (Section 5.7 permission)",
            "iterations": int(getattr(result, "nit", 0)),
            "converged": bool(getattr(result, "success", False)),
            "message": str(getattr(result, "message", "")),
            "before": before.to_dict(),
            "after": after.to_dict(),
            "collocation_points": int(len(x_grid) * len(times)),
            "basis_size": self.basis_size,
            "learned_decay_rates": [float(value) for value in self.decay_rates],
            "analytic_decay_rates": [float(value) for value in analytic_rates],
            "rate_learning_relative_error": float(
                np.mean(np.abs(self.decay_rates - analytic_rates) / analytic_rates)
            ),
        }
        return self.training_report

    def to_dict(self) -> dict[str, Any]:
        """Serialize the surrogate identity and training report."""
        return {
            "model_version": self.model_version,
            "family": "pinn",
            "basis_size": self.basis_size,
            "amplitudes": [float(value) for value in self.amplitudes],
            "decay_rates": [float(value) for value in self.decay_rates],
            "training_report": dict(self.training_report),
        }
