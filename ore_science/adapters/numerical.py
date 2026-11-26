"""Conventional numerical solvers serving as independent baselines (Step 27, Stage 3).

The Crank-Nicolson finite-difference solver is the reference against which learned
surrogates (PINNs, neural operators) are compared. It carries its own convergence
metadata so validation can verify stability and refinement behaviour explicitly
instead of trusting the last iterate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["CrankNicolsonHeatSolver", "HeatSolverResult", "analytic_heat_solution"]


@dataclass(frozen=True, slots=True)
class HeatSolverResult:
    """Sampled solution field plus grid and numerical metadata."""

    field: np.ndarray
    x_grid: np.ndarray
    times: np.ndarray
    alpha: float
    method: str
    grid_size: int
    time_steps: int
    stable: bool
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize solver output metadata without inlining the field arrays."""
        return {
            "method": self.method,
            "alpha": self.alpha,
            "grid_size": self.grid_size,
            "time_steps": self.time_steps,
            "time_horizon": float(self.times[-1]),
            "stable": self.stable,
            "metadata": {
                key: value
                for key, value in self.metadata.items()
                if key not in ("snapshots", "snapshot_times")
            },
            "snapshot_count": int(
                self.metadata.get("snapshots", np.zeros((1, 1))).shape[0]
            ),
            "field_shape": list(self.field.shape),
        }


def analytic_heat_solution(
    x: np.ndarray, t: float, *, length: float, alpha: float, u_initial: float,
    modes: int = 60,
) -> np.ndarray:
    """Analytic Fourier-series solution for a uniform rod with Dirichlet ends.

    The initial condition is the constant u_initial on (0, L) with zero ends; the
    series converges to u_initial at t -> 0 in the interior.
    """
    total = np.zeros_like(x, dtype=float)
    for n in range(1, modes + 1):
        coefficient = 4.0 * u_initial / (math.pi * n) * (1.0 if n % 2 == 1 else 0.0)
        if coefficient == 0.0:
            continue
        decay = math.exp(-alpha * (math.pi * n / length) ** 2 * t)
        total += coefficient * np.sin(math.pi * n * x / length) * decay
    return total


class CrankNicolsonHeatSolver:
    """Crank-Nicolson implicit solver for the 1-D heat equation (independent baseline)."""

    method = "crank_nicolson_implicit"

    def solve(
        self,
        *,
        length: float,
        time_horizon: float,
        alpha: float,
        u_initial: float,
        u_boundary: float,
        grid_size: int,
        time_steps: int,
        snapshot_count: int = 1,
    ) -> HeatSolverResult:
        """Integrate du/dt = alpha d2u/dx2 with Dirichlet boundaries at both ends.

        ``snapshot_count`` controls how many evenly spaced time snapshots (including
        the final field) are retained for conservation and convergence checks.
        """
        if length <= 0.0 or time_horizon <= 0.0:
            raise ValueError("length and time horizon must be positive")
        if alpha <= 0.0:
            raise ValueError("thermal diffusivity must be positive")
        if grid_size < 3 or time_steps < 1:
            raise ValueError("grid size must be >= 3 and time steps >= 1")
        if snapshot_count < 1 or snapshot_count > time_steps + 1:
            raise ValueError("snapshot count must lie in [1, time_steps + 1]")

        x_grid = np.linspace(0.0, length, grid_size)
        dx = x_grid[1] - x_grid[0]
        dt = time_horizon / time_steps
        field = np.full(grid_size, u_initial, dtype=float)
        field[0] = u_boundary
        field[-1] = u_boundary

        r = alpha * dt / (2.0 * dx * dx)
        interior = grid_size - 2
        # Tridiagonal system: -r u_{i-1}^{n+1} + (1+2r) u_i^{n+1} - r u_{i+1}^{n+1}
        #                    =  r u_{i-1}^n     + (1-2r) u_i^n     + r u_{i+1}^n
        lower = np.full(interior, -r)
        diagonal = np.full(interior, 1.0 + 2.0 * r)
        upper = np.full(interior, -r)
        times = np.linspace(0.0, time_horizon, time_steps + 1)
        snapshot_interval = max(1, time_steps // (snapshot_count - 1)) if snapshot_count > 1 else time_steps
        snapshots: list[np.ndarray] = [field.copy()]
        snapshot_times: list[float] = [0.0]

        for step_index in range(time_steps):
            right = (
                r * field[:-2] + (1.0 - 2.0 * r) * field[1:-1] + r * field[2:]
            )
            right[0] += r * u_boundary
            right[-1] += r * u_boundary
            solution = _tridiagonal_solve(lower, diagonal, upper, right)
            field[1:-1] = solution
            field[0] = u_boundary
            field[-1] = u_boundary
            if (step_index + 1) % snapshot_interval == 0 or step_index == time_steps - 1:
                snapshots.append(field.copy())
                snapshot_times.append(times[step_index + 1])

        stable = bool(np.all(np.isfinite(field)))
        finite_difference_residual = self._residual_estimate(field, x_grid, dt, alpha)
        return HeatSolverResult(
            field=field,
            x_grid=x_grid,
            times=times,
            alpha=alpha,
            method=self.method,
            grid_size=grid_size,
            time_steps=time_steps,
            stable=stable,
            metadata={
                "r": float(r),
                "dx": float(dx),
                "dt": float(dt),
                "finite_difference_residual": float(finite_difference_residual),
                "boundary": "Dirichlet both ends",
                "unconditionally_stable": True,
                "snapshots": np.asarray(snapshots),
                "snapshot_times": np.asarray(snapshot_times),
            },
        )

    @staticmethod
    def _residual_estimate(field: np.ndarray, x_grid: np.ndarray, dt: float, alpha: float) -> float:
        """Estimate the mean |du/dt - alpha d2u/dx2| residual via finite differences.

        Uses the last two time snapshots implicitly approximated by the steady decay;
        here we evaluate the spatial Laplacian consistency of the final field, which
        is the component available without storing full history.
        """
        if field.shape[0] < 3:
            return float("inf")
        laplacian = (field[:-2] - 2.0 * field[1:-1] + field[2:]) / (
            (x_grid[1] - x_grid[0]) ** 2
        )
        mean_curvature = float(np.mean(np.abs(laplacian)))
        return mean_curvature * alpha


def _tridiagonal_solve(
    lower: np.ndarray, diagonal: np.ndarray, upper: np.ndarray, right: np.ndarray
) -> np.ndarray:
    """Thomas algorithm for a tridiagonal linear system."""
    n = diagonal.shape[0]
    work_diagonal = diagonal.astype(float).copy()
    work_right = right.astype(float).copy()
    for index in range(1, n):
        factor = lower[index] / work_diagonal[index - 1]
        work_diagonal[index] -= factor * upper[index - 1]
        work_right[index] -= factor * work_right[index - 1]
    solution = np.zeros(n, dtype=float)
    solution[-1] = work_right[-1] / work_diagonal[-1]
    for index in range(n - 2, -1, -1):
        solution[index] = (
            work_right[index] - upper[index] * solution[index + 1]
        ) / work_diagonal[index]
    return solution
