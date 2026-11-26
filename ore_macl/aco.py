"""Ant Colony Optimization for offline discrete path search (Steps 64-65).

Ants construct operation or knowledge-graph paths; useful validated paths are
reinforced while dependencies and budgets are enforced at every extension - a
path that violates a constraint is simply unreachable, so the optimizer cannot
bypass ORE validation. Pheromone update uses evaporation plus bounded deposits,
and runs are reproducible through an explicit seed.
"""
from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ACOConfig:
    """Bound the colony's size, iterations, pheromone parameters, and path length."""

    num_ants: int = 8
    iterations: int = 20
    alpha: float = 1.0
    beta: float = 2.0
    evaporation: float = 0.15
    deposit: float = 1.0
    seed: int = 7
    max_path_length: int = 12
    elitist_deposits: int = 2

    def __post_init__(self) -> None:
        """Validate colony configuration bounds."""
        if self.num_ants < 1:
            raise ValueError("num_ants must be at least one")
        if self.iterations < 1:
            raise ValueError("iterations must be at least one")
        if self.alpha < 0 or self.beta < 0:
            raise ValueError("alpha and beta cannot be negative")
        if not 0.0 < self.evaporation < 1.0:
            raise ValueError("evaporation must be in (0, 1)")
        if self.deposit <= 0:
            raise ValueError("deposit must be positive")
        if self.max_path_length < 1:
            raise ValueError("max_path_length must be at least one")
        if self.elitist_deposits < 1:
            raise ValueError("elitist_deposits must be at least one")


@dataclass
class OperationGraph:
    """Declare an operation graph with costs, desirability, and dependencies (Step 64)."""

    operations: tuple[str, ...]
    dependencies: Mapping[str, tuple[str, ...]]
    edge_cost: Mapping[tuple[str, str], float]
    heuristic_desirability: Mapping[tuple[str, str], float]
    start: str = "start"
    terminal: str | None = None

    def __post_init__(self) -> None:
        """Validate graph structure and dependency closure."""
        known = set(self.operations) | ({self.start} if self.start else set())
        if self.terminal is not None and self.terminal not in known:
            raise ValueError("terminal must be a declared operation or the start")
        for operation, dependencies in self.dependencies.items():
            if operation not in known:
                raise ValueError(f"dependency declared for unknown operation: {operation}")
            for dependency in dependencies:
                if dependency not in known:
                    raise ValueError(
                        f"operation {operation} depends on unknown operation {dependency}"
                    )
        for (source, target) in list(self.edge_cost) + list(self.heuristic_desirability):
            if source not in known or target not in known:
                raise ValueError(f"edge references unknown operation: {source} -> {target}")

    def predecessors_of(self, operation: str) -> tuple[str, ...]:
        """Return the declared dependencies of one operation."""
        return tuple(self.dependencies.get(operation, ()))


@dataclass
class ACOResult:
    """Record the bounded colony search outcome."""

    best_path: tuple[str, ...]
    best_score: float
    iterations: int
    paths_evaluated: int
    validated_best: bool = False
    history: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a detached result record."""
        return {
            "best_path": list(self.best_path),
            "best_score": self.best_score,
            "iterations": self.iterations,
            "paths_evaluated": self.paths_evaluated,
            "validated_best": self.validated_best,
            "history": list(self.history),
        }


class AntColonyOptimizer:
    """Search discrete operation paths with pheromone reinforcement (Step 64)."""

    def __init__(
        self,
        config: ACOConfig,
        graph: OperationGraph,
        *,
        validator: Callable[[Sequence[str]], bool] | None = None,
        budget: float = 10.0,
    ) -> None:
        """Bind the colony to a graph, a path validator, and a cost budget."""
        self.config = config
        self.graph = graph
        self.validator = validator
        self.budget = budget
        self._pheromones: dict[tuple[str, str], float] = {
            edge: 1.0 for edge in list(graph.edge_cost) + list(graph.heuristic_desirability)
        }
        self.paths_evaluated = 0

    def _feasible_successors(
        self,
        path: Sequence[str],
        spent: float,
    ) -> tuple[tuple[str, float], ...]:
        """Return affordable, dependency-satisfied successors of the path head (Step 64)."""
        completed = set(path)
        head = path[-1] if path else self.graph.start
        candidates = []
        for (source, target), cost in self.graph.edge_cost.items():
            if source != head or target in completed:
                continue
            if spent + cost > self.budget:
                continue
            if any(dependency not in completed for dependency in self.graph.predecessors_of(target)):
                continue
            desirability = self.graph.heuristic_desirability.get((source, target), 1.0)
            candidates.append((target, cost + desirability))
        return tuple(candidates)

    def _construct_path(self, rng: random.Random) -> tuple[tuple[str, ...], float]:
        """Construct one dependency- and budget-feasible path stochastically."""
        path: list[str] = []
        spent = 0.0
        while len(path) < self.config.max_path_length:
            head = path[-1] if path else self.graph.start
            successors = self._feasible_successors(path, spent)
            if not successors:
                break
            weights = []
            for target, cost in successors:
                edge = (head, target)
                pheromone = self._pheromones.get(edge, 1.0) ** self.config.alpha
                desirability = self.graph.heuristic_desirability.get(edge, 1.0) ** self.config.beta
                weights.append(pheromone * desirability)
            chosen = rng.choices(
                [target for target, _ in successors], weights=weights, k=1,
            )[0]
            cost = next(cost for target, cost in successors if target == chosen)
            spent += cost
            path.append(chosen)
            if self.graph.terminal is not None and chosen == self.graph.terminal:
                break
        return tuple(path), spent

    def _path_score(self, path: Sequence[str], spent: float) -> float:
        """Score a path by length minus cost; validation gates reinforcement (Step 64)."""
        self.paths_evaluated += 1
        if self.validator is not None and not self.validator(path):
            return float("-inf")
        return len(path) / (1.0 + spent)

    def optimize(self) -> ACOResult:
        """Run the bounded, reproducible colony search and reinforce validated paths."""
        rng = random.Random(self.config.seed)
        best_path: tuple[str, ...] = ()
        best_score = float("-inf")
        history: list[float] = []
        iterations = 0
        for _ in range(self.config.iterations):
            iterations += 1
            for edge in self._pheromones:
                self._pheromones[edge] *= (1.0 - self.config.evaporation)
            candidates: list[tuple[tuple[str, ...], float]] = []
            for _ in range(self.config.num_ants):
                path, spent = self._construct_path(rng)
                if not path:
                    continue
                score = self._path_score(path, spent)
                if score == float("-inf"):
                    continue
                candidates.append((path, score))
                if score > best_score:
                    best_score = score
                    best_path = path
            candidates.sort(key=lambda item: (-item[1], item[0]))
            for path, score in candidates[:self.config.elitist_deposits]:
                head = self.graph.start
                for operation in path:
                    edge = (head, operation)
                    self._pheromones[edge] = self._pheromones.get(edge, 1.0) + self.config.deposit * score
                    head = operation
            history.append(best_score if best_score != float("-inf") else 0.0)
        validated = self.validator(best_path) if (self.validator and best_path) else True
        return ACOResult(
            best_path=best_path,
            best_score=best_score if best_score != float("-inf") else 0.0,
            iterations=iterations,
            paths_evaluated=self.paths_evaluated,
            validated_best=bool(best_path) and validated,
            history=history,
        )


def dependency_feasible(path: Sequence[str], graph: OperationGraph) -> bool:
    """Check that a path satisfies every declared dependency at each extension."""
    completed: set[str] = set()
    for operation in path:
        if any(dependency not in completed for dependency in graph.predecessors_of(operation)):
            return False
        completed.add(operation)
    return True
