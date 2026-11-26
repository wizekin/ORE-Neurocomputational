"""Bounded Particle Swarm Optimization for offline parameter search (Steps 63, 65).

Particles represent module-selection weights or resource-policy coefficients. The
objective must be a fixed replay or simulation evaluator; candidates are clamped
to declared bounds, hard constraints are enforced, iterations and evaluations are
bounded, and runs are reproducible through an explicit seed. PSO proposes
parameter sets only - existing ORE policies validate and authorize them, and the
optimizer can never alter permissions or critical constraints.
"""
from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .errors import MACLError


@dataclass(frozen=True, slots=True)
class PSOConfig:
    """Bound the swarm's size, iterations, coefficients, and evaluation budget."""

    num_particles: int = 12
    iterations: int = 25
    inertia: float = 0.6
    cognitive: float = 1.4
    social: float = 1.4
    seed: int = 11
    max_evaluations: int = 1000

    def __post_init__(self) -> None:
        """Validate swarm configuration bounds."""
        if self.num_particles < 2:
            raise ValueError("num_particles must be at least two")
        if self.iterations < 1:
            raise ValueError("iterations must be at least one")
        if not 0.0 <= self.inertia <= 1.0:
            raise ValueError("inertia must be in [0, 1]")
        if self.cognitive < 0 or self.social < 0:
            raise ValueError("cognitive and social coefficients cannot be negative")
        if self.max_evaluations < self.num_particles:
            raise ValueError("max_evaluations must cover at least one swarm pass")


@dataclass
class PSOResult:
    """Record the bounded search outcome with reproducibility metadata."""

    best_position: tuple[float, ...]
    best_score: float
    iterations: int
    evaluations: int
    history: list[float] = field(default_factory=list)
    feasible: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return a detached result record."""
        return {
            "best_position": list(self.best_position),
            "best_score": self.best_score,
            "iterations": self.iterations,
            "evaluations": self.evaluations,
            "history": list(self.history),
            "feasible": self.feasible,
        }


class ParticleSwarmOptimizer:
    """Optimize bounded parameter vectors against a fixed objective (Step 63)."""

    def __init__(
        self,
        config: PSOConfig,
        objective: Callable[[Sequence[float]], float],
        *,
        bounds: Sequence[tuple[float, float]],
        constraint: Callable[[Sequence[float]], bool] | None = None,
    ) -> None:
        """Bind the swarm to an objective, per-dimension bounds, and a hard constraint."""
        if not bounds:
            raise ValueError("bounds must declare at least one dimension")
        for index, (lower, upper) in enumerate(bounds):
            if lower >= upper:
                raise ValueError(f"bounds for dimension {index} must satisfy lower < upper")
        self.config = config
        self.objective = objective
        self.bounds = tuple((float(lower), float(upper)) for lower, upper in bounds)
        self.constraint = constraint
        self.evaluations = 0

    def _evaluate(self, position: Sequence[float]) -> float:
        """Evaluate one clamped candidate under the evaluation budget."""
        if self.evaluations >= self.config.max_evaluations:
            raise MACLError("PSO evaluation budget exhausted")
        self.evaluations += 1
        return float(self.objective(tuple(position)))

    def _feasible(self, position: Sequence[float]) -> bool:
        """Apply the hard constraint gate; infeasible candidates never score (Step 65)."""
        return self.constraint is None or self.constraint(tuple(position))

    def _clamp(self, position: list[float]) -> list[float]:
        """Clamp every dimension back into its declared bounds."""
        return [
            max(lower, min(upper, value))
            for value, (lower, upper) in zip(position, self.bounds)
        ]

    def optimize(self) -> PSOResult:
        """Run the bounded, reproducible swarm search and return the best candidate."""
        rng = random.Random(self.config.seed)
        dimensions = len(self.bounds)
        span = [upper - lower for lower, upper in self.bounds]
        positions = [
            self._clamp([rng.uniform(lower, upper) for lower, upper in self.bounds])
            for _ in range(self.config.num_particles)
        ]
        velocities = [
            [rng.uniform(-0.2 * width, 0.2 * width) for width in span]
            for _ in range(self.config.num_particles)
        ]
        personal_best = [list(position) for position in positions]
        personal_scores = []
        for position in positions:
            if self._feasible(position):
                personal_scores.append(self._evaluate(position))
            else:
                personal_scores.append(float("-inf"))
        global_index = max(range(len(personal_scores)), key=lambda index: personal_scores[index])
        global_best = list(personal_best[global_index])
        global_score = personal_scores[global_index]
        history = [global_score if global_score != float("-inf") else 0.0]
        iterations = 0
        for _ in range(self.config.iterations):
            iterations += 1
            for index in range(self.config.num_particles):
                for dimension in range(dimensions):
                    cognitive_pull = self.config.cognitive * rng.random() * (
                        personal_best[index][dimension] - positions[index][dimension]
                    )
                    social_pull = self.config.social * rng.random() * (
                        global_best[dimension] - positions[index][dimension]
                    )
                    velocities[index][dimension] = (
                        self.config.inertia * velocities[index][dimension]
                        + cognitive_pull + social_pull
                    )
                positions[index] = self._clamp([
                    value + velocities[index][dimension]
                    for dimension, value in enumerate(positions[index])
                ])
                if not self._feasible(positions[index]):
                    continue
                score = self._evaluate(positions[index])
                if score > personal_scores[index]:
                    personal_scores[index] = score
                    personal_best[index] = list(positions[index])
                    if score > global_score:
                        global_score = score
                        global_best = list(positions[index])
            history.append(global_score if global_score != float("-inf") else 0.0)
        feasible = global_score != float("-inf")
        return PSOResult(
            best_position=tuple(global_best),
            best_score=global_score if feasible else 0.0,
            iterations=iterations,
            evaluations=self.evaluations,
            history=history,
            feasible=feasible,
        )


def resource_policy_objective(
    weights: Sequence[float],
    *,
    reference_solution: Sequence[float],
) -> float:
    """Evaluate a replay objective: closeness to a validated reference policy."""
    if len(weights) != len(reference_solution):
        raise ValueError("weights and reference solution must share dimensions")
    distance = sum(
        (float(weight) - float(reference)) ** 2
        for weight, reference in zip(weights, reference_solution)
    )
    return -distance
