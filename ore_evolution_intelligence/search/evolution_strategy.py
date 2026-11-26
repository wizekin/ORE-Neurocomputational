"""Evolution strategies for continuous parameter vectors (section 2.3, step 30).

A (mu + lambda) evolution strategy with Gaussian perturbation and deterministic seeds:
for ``theta_i = mu + sigma * epsilon_i, epsilon ~ N(0, I)``. Only *feasible* offspring
enter the parent set (step 21); the known-good baseline is always preserved (step 18)
and the run stops responsibly on budget or generation caps (step 28).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ..domain.genome import ParameterGenome
from ..evaluation.evaluator import BudgetExhausted, SeededRandom


@dataclass(frozen=True, slots=True)
class ESResult:
    """Evolution-strategy run summary."""

    best_genome: ParameterGenome | None
    best_fitness: float | None
    generations: int
    evaluations: int
    stop_reason: str
    sigma_final: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "best_fitness": self.best_fitness,
            "best_genome": self.best_genome.as_dict() if self.best_genome else None,
            "generations": self.generations, "evaluations": self.evaluations,
            "stop_reason": self.stop_reason, "sigma_final": self.sigma_final,
        }


class EvolutionStrategyOptimizer:
    """(mu + lambda) ES over a bounded continuous parameter space."""

    def __init__(
        self,
        *,
        mu: int = 4,
        lambda_: int = 8,
        sigma: float = 0.15,
        sigma_decay: float = 0.92,
        generation_limit: int = 10,
        seed: int = 20261122,
    ) -> None:
        if mu < 1 or lambda_ < 1:
            raise ValueError("mu and lambda must be positive")
        self.mu = mu
        self.lambda_ = lambda_
        self.sigma = sigma
        self.sigma_decay = sigma_decay
        self.generation_limit = generation_limit
        self.seed = seed

    def optimize(
        self,
        baseline: ParameterGenome,
        objective: Callable[[ParameterGenome], float | None],
        *,
        budget: Any | None = None,
    ) -> ESResult:
        """Optimize one bounded objective; ``objective`` returns fitness or None (infeasible)."""
        if baseline.kind != "parameters":
            raise ValueError("evolution strategies operate on parameter genomes")
        rng = SeededRandom(self.seed)
        parents: list[tuple[float, ParameterGenome]] = []
        baseline_fitness = objective(baseline)
        best_fitness = baseline_fitness
        best_genome = baseline if baseline_fitness is not None else None
        if baseline_fitness is not None:
            parents.append((baseline_fitness, baseline))
        evaluations = 1
        sigma = self.sigma
        stop_reason = "generation_limit"
        for generation in range(1, self.generation_limit + 1):
            offspring: list[tuple[float, ParameterGenome]] = []
            for _ in range(self.lambda_):
                if not parents:
                    genome = self._perturb(baseline, rng, sigma)
                else:
                    parent = parents[rng.randint(0, len(parents) - 1)]
                    genome = self._perturb(parent[1], rng, sigma)
                try:
                    fitness = objective(genome)
                except BudgetExhausted:
                    stop_reason = "budget_exhausted"
                    return ESResult(
                        best_genome=best_genome, best_fitness=best_fitness,
                        generations=generation - 1, evaluations=evaluations,
                        stop_reason=stop_reason, sigma_final=sigma,
                    )
                evaluations += 1
                if fitness is not None:
                    offspring.append((fitness, genome))
            if not offspring and not parents:
                stop_reason = "unacceptable_failure_rate"
                break
            # (mu + lambda): parents and feasible offspring compete.
            combined = sorted(parents + offspring, key=lambda pair: -pair[0])[: self.mu]
            parents = combined
            if combined and (best_fitness is None or combined[0][0] > best_fitness):
                best_fitness = combined[0][0]
                best_genome = combined[0][1]
            sigma *= self.sigma_decay
            if sigma < 1e-4:
                stop_reason = "convergence"
                break
        return ESResult(
            best_genome=best_genome, best_fitness=best_fitness,
            generations=self.generation_limit, evaluations=evaluations,
            stop_reason=stop_reason, sigma_final=sigma,
        )

    def _perturb(self, genome: ParameterGenome, rng: SeededRandom, sigma: float) -> ParameterGenome:
        updates: dict[str, Any] = {}
        for definition in genome.space:
            if definition.kind != "continuous":
                continue
            span = definition.high - definition.low
            updates[definition.name] = (
                genome.values[definition.name] + rng.gauss(0.0, sigma * span)
            )
        return genome.clone_with(updates)


__all__ = ["ESResult", "EvolutionStrategyOptimizer"]
