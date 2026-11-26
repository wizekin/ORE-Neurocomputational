"""CMA-ES adapter for correlated continuous parameters (section 2.3, step 30).

A compact, dependency-free CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
over the *continuous* dimensions of a bounded parameter genome. CMA-ES adapts the
search covariance and step size, which makes it useful for correlated continuous
coordinator parameters — and useless for arbitrary discrete program structures, so
the adapter refuses non-continuous search spaces (step 30: apply CMA-ES selectively;
acceptance test: CMA-ES only for continuous correlated parameter vectors).

Constraint handling follows the gate philosophy (step 21): infeasible candidates are
ranked as worst and never contribute to the covariance update beyond displacement.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from ..domain.genome import ParameterGenome


@dataclass(frozen=True, slots=True)
class CMAESResult:
    """CMA-ES run summary."""

    best_genome: ParameterGenome | None
    best_fitness: float | None
    generations: int
    evaluations: int
    stop_reason: str
    sigma_final: float
    axis_ratio_final: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "best_fitness": self.best_fitness,
            "best_genome": self.best_genome.as_dict() if self.best_genome else None,
            "generations": self.generations, "evaluations": self.evaluations,
            "stop_reason": self.stop_reason, "sigma_final": self.sigma_final,
            "axis_ratio_final": self.axis_ratio_final,
        }


class CMAESAdapter:
    """CMA-ES over the continuous subspace of one parameter genome (step 30)."""

    def __init__(
        self,
        *,
        sigma0: float = 0.20,
        generation_limit: int = 40,
        seed: int = 20261122,
        tol: float = 1e-8,
    ) -> None:
        if sigma0 <= 0:
            raise ValueError("sigma0 must be positive")
        self.sigma0 = sigma0
        self.generation_limit = generation_limit
        self.seed = seed
        self.tol = tol

    def optimize(
        self,
        baseline: ParameterGenome,
        objective: Callable[[ParameterGenome], float | None],
        *,
        budget: Any | None = None,
    ) -> CMAESResult:
        """Optimize the continuous parameters of ``baseline``.

        ``objective`` maps a genome to a scalar fitness or ``None`` when the candidate
        violates a hard constraint (gates, not penalties — step 21).
        """
        continuous = [d for d in baseline.space if d.kind == "continuous"]
        if len(continuous) != len(baseline.space):
            raise ValueError(
                "CMA-ES applies to continuous correlated parameter vectors only; "
                "the search space contains discrete dimensions "
                f"({[d.name for d in baseline.space if d.kind != 'continuous']})"
            )
        n = len(continuous)
        rng = np.random.default_rng(self.seed)
        x_mean = np.array([
            _normalize(baseline.values[d.name], d) for d in continuous
        ])
        sigma = self.sigma0
        covariance = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)

        popsize = 4 + int(3 * math.log(max(n, 1)))
        mu = max(popsize // 2, 1)
        weights = np.array([
            math.log(mu + 0.5) - math.log(i + 1) for i in range(mu)
        ])
        weights /= weights.sum()
        mu_eff = 1.0 / float(np.sum(weights ** 2))

        chi_n = math.sqrt(n) * (
            1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n)
        )
        c_sigma = (mu_eff + 2.0) / (n + mu_eff + 5.0)
        d_sigma = 1.0 + 2.0 * max(math.sqrt((mu_eff - 1.0) / (n + 2.0)) - 1.0, 0.0) + c_sigma
        c_c = (4.0 + mu_eff / n) / (n + 4.0 + 2.0 * mu_eff / n)
        c1 = 2.0 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1.0 - c1, 2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((n + 2.0) ** 2 + mu_eff))

        best_genome = baseline
        best_fitness = objective(baseline)
        evaluations = 1
        stop_reason = "generation_limit"

        for generation in range(1, self.generation_limit + 1):
            try:
                if budget is not None:
                    budget.consume(popsize)
            except Exception as error:
                if type(error).__name__ == "BudgetExhausted":
                    stop_reason = "budget_exhausted"
                    break
                raise
            # Sample in the adapted covariance distribution.
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            eigenvalues = np.maximum(eigenvalues, 1e-12)
            b_d = eigenvectors * np.sqrt(eigenvalues)
            samples = []
            for _ in range(popsize):
                z = rng.standard_normal(n)
                step = x_mean + sigma * (b_d @ z)
                genome = _genome_from_vector(baseline, np.clip(step, 0.0, 1.0))
                samples.append((genome, step, z))
            scored: list[tuple[float, int]] = []
            for index, (genome, _step, _z) in enumerate(samples):
                fitness = objective(genome)
                evaluations += 1
                if fitness is not None and (best_fitness is None or fitness > best_fitness):
                    best_fitness = fitness
                    best_genome = genome
                score = fitness if fitness is not None else -1e12
                scored.append((score, index))
            scored.sort(key=lambda pair: -pair[0])
            ranked_indices = [index for _score, index in scored[:mu]]

            # Weighted recombination of the selected steps.
            selected = np.array([samples[i][1] for i in ranked_indices])
            x_new = np.average(selected, axis=0, weights=weights)
            y_w = (x_new - x_mean) / max(sigma, 1e-12)

            # Cumulation: evolution path for step size and covariance.
            p_sigma = (1.0 - c_sigma) * p_sigma + math.sqrt(
                c_sigma * (2.0 - c_sigma) * mu_eff
            ) * (eigenvectors.T @ y_w / np.sqrt(np.maximum(eigenvalues, 1e-12)))
            p_c = (1.0 - c_c) * p_c + math.sqrt(
                c_c * (2.0 - c_c) * mu_eff
            ) * y_w

            # Step-size adaptation.
            ratio = float(np.linalg.norm(p_sigma)) / chi_n + 1e-12
            sigma *= math.exp((c_sigma / d_sigma) * (ratio - 1.0))
            sigma = min(max(sigma, 1e-6), 0.5)

            # Covariance update: rank-one plus rank-mu.
            c_s = (1.0 - c1 - c_mu)
            covariance = c_s * covariance + c1 * np.outer(p_c, p_c) + c_mu * (
                sum(
                    weights[i] * np.outer(
                        (samples[ranked_indices[i]][1] - x_mean) / max(sigma, 1e-12),
                        (samples[ranked_indices[i]][1] - x_mean) / max(sigma, 1e-12),
                    )
                    for i in range(len(ranked_indices))
                )
            )
            covariance = np.triu(covariance) + np.triu(covariance, 1).T
            x_mean = np.clip(x_new, 0.0, 1.0)

            # Convergence checks.
            axis_ratio = float(
                math.sqrt(max(eigenvalues) / max(min(eigenvalues), 1e-12))
            )
            if sigma * math.sqrt(max(eigenvalues)) < self.tol:
                stop_reason = "convergence"
                break
            if axis_ratio > 1e7:
                stop_reason = "degenerate_covariance"
                break
        eigenvalues_final, _ = np.linalg.eigh(covariance)
        return CMAESResult(
            best_genome=best_genome,
            best_fitness=best_fitness,
            generations=self.generation_limit,
            evaluations=evaluations,
            stop_reason=stop_reason,
            sigma_final=sigma,
            axis_ratio_final=float(
                math.sqrt(max(eigenvalues_final) / max(min(eigenvalues_final), 1e-12))
            ),
        )


def _normalize(value: float, definition: Any) -> float:
    return (float(value) - definition.low) / (definition.high - definition.low)


def _genome_from_vector(baseline: ParameterGenome, vector: np.ndarray) -> ParameterGenome:
    updates: dict[str, float] = {}
    for index, definition in enumerate(baseline.space):
        if definition.kind != "continuous":
            continue
        updates[definition.name] = definition.low + float(vector[index]) * (
            definition.high - definition.low
        )
    return baseline.clone_with(updates)


__all__ = ["CMAESAdapter", "CMAESResult"]
