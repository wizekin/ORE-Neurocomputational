"""Shared fixtures for the ORE Point 9 evolution-intelligence tests."""
from __future__ import annotations

import pytest

from ore_evolution_intelligence import (
    HardConstraints, ScoringConfig, ScalarFitness, StrategyEvaluator,
    build_benchmark,
)
from ore_evolution_intelligence.domain.genome import (
    ParameterGenome, default_search_space,
)
from ore_evolution_intelligence.domain.strategy import (
    LineageRecord, StrategyCandidate,
)

SEED = 20261122


def make_benchmark(seed: int = SEED, tasks_per_family: int = 10):
    """Versioned benchmark with the default scoring configuration."""
    return build_benchmark(
        benchmark_id="ore_module_selection_v1",
        seed=seed,
        tasks_per_family=tasks_per_family,
        scoring=ScoringConfig(
            fitness_config=ScalarFitness().as_dict(),
            constraints_config=HardConstraints().as_dict(),
        ),
    )


def make_evaluator(seed: int = SEED, trials_per_task: int = 3, **kwargs):
    """Independent strategy evaluator over the versioned benchmark."""
    return StrategyEvaluator(
        make_benchmark(seed), trials_per_task=trials_per_task, **kwargs,
    )


def make_baseline_genome():
    """The deterministic baseline parameter genome (step 3)."""
    space = default_search_space()
    return ParameterGenome(
        {definition.name: definition.default_value() for definition in space}, space,
    )


def make_baseline_candidate(run_id: str = "fixture-run"):
    """The immutable baseline candidate."""
    return StrategyCandidate.from_genome(
        make_baseline_genome(),
        lineage=LineageRecord(operation="initial", generation=0, run_id=run_id),
        run_id=run_id,
    )


def make_aggressive_genome(**overrides):
    """A strong hand-crafted candidate: aggressive escalation, broad activation."""
    space = default_search_space()
    values = {definition.name: definition.default_value() for definition in space}
    for definition in space:
        if definition.name.startswith("weight__"):
            values[definition.name] = 1.4
        elif definition.name.startswith("threshold__"):
            values[definition.name] = 0.25
        elif definition.name.startswith("route_weight__"):
            values[definition.name] = 0.6
    values["cpu_budget_share"] = 0.9
    values["memory_budget_share"] = 0.8
    values["retrieval_depth"] = 1
    values["escalation_mode"] = "aggressive"
    values.update(overrides)
    return ParameterGenome(values, space)


def make_candidate(values_overrides=None, run_id: str = "fixture-run"):
    genome = make_aggressive_genome(**(values_overrides or {}))
    return StrategyCandidate.from_genome(
        genome,
        lineage=LineageRecord(operation="initial", generation=0, run_id=run_id),
        run_id=run_id,
    )


__all__ = [
    "SEED", "make_aggressive_genome", "make_baseline_candidate",
    "make_baseline_genome", "make_benchmark", "make_candidate", "make_evaluator",
]
