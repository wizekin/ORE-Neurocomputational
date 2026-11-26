"""Genome operators: bounded crossover and mutation (steps 24-25, 27).

Crossover combines compatible parameter groups or policy components and *validates
schema after recombination* (step 24): operators return ``None`` — never an invalid
genome — when recombination would break the schema (acceptance test 15: crossover and
mutation yield valid genomes or are rejected).

Mutation schedule (step 25, documented): per-parameter mutation probability 0.25;
continuous parameters receive Gaussian perturbation with sigma proportional to the
validated range and clipping at the bounds; integer parameters move to a uniform
neighbour; choice parameters re-draw uniformly. Policy tables mutate a random subset
of cells (add/drop approved actions or defer); policy-IR mutation delegates to the
typed-grammar subtree operators.
"""
from __future__ import annotations

import math
from typing import Any

from ..domain.actions import ORE_MODULES, ROUTE_QUEUES
from ..domain.genome import (
    ParameterDefinition, ParameterGenome, PolicyTableGenome, table_context_key,
)
from ..domain.actions import ActionCall
from ..evaluation.evaluator import SeededRandom

#: Parameter groups used by parameter crossover (compatible groups, step 24).
PARAMETER_GROUPS: tuple[str, ...] = (
    "weights", "thresholds", "routing", "resources", "discrete",
)


def _parameter_group(name: str) -> str:
    if name.startswith("weight__"):
        return "weights"
    if name.startswith("threshold__"):
        return "thresholds"
    if name.startswith("route_weight__"):
        return "routing"
    if name in ("cpu_budget_share", "memory_budget_share"):
        return "resources"
    return "discrete"


def parameter_crossover(
    parent_a: ParameterGenome, parent_b: ParameterGenome, rng: SeededRandom,
) -> ParameterGenome | None:
    """Exchange whole compatible parameter groups; schema validated afterwards."""
    if parent_a.space != parent_b.space:
        return None  # incompatible parameter groups (step 24)
    group_source: dict[str, bool] = {}
    for group in PARAMETER_GROUPS:
        group_source[group] = rng.uniform() < 0.5
    merged: dict[str, Any] = {}
    for name in parent_a.values:
        genome = parent_a if group_source[_parameter_group(name)] else parent_b
        merged[name] = genome.values[name]
    offspring = ParameterGenome(merged, parent_a.space)
    return offspring if not offspring.validate() else None


def parameter_mutate(
    genome: ParameterGenome,
    rng: SeededRandom,
    *,
    probability: float = 0.25,
    sigma: float = 0.12,
) -> ParameterGenome | None:
    """Documented mutation schedule: gaussian for continuous, neighbours otherwise."""
    updates: dict[str, Any] = {}
    for definition in genome.space:
        if rng.uniform() > probability:
            continue
        name = definition.name
        current = genome.values[name]
        if definition.kind == "continuous":
            span = definition.high - definition.low
            updates[name] = float(current) + rng.gauss(0.0, sigma * span)
        elif definition.kind == "integer":
            low, high = int(definition.low), int(definition.high)
            options = [v for v in range(low, high + 1) if v != int(current)]
            updates[name] = rng.choice(options) if options else current
        else:
            options = [c for c in definition.choices if c != current]
            updates[name] = rng.choice(options) if options else current
    if not updates:
        return genome
    offspring = genome.clone_with(updates)
    return offspring if not offspring.validate() else None


def parameter_random(
    space: tuple[ParameterDefinition, ...], rng: SeededRandom,
) -> ParameterGenome:
    """Uniform random genome inside the validated ranges (diverse initialization)."""
    values: dict[str, Any] = {}
    for definition in space:
        if definition.kind == "continuous":
            values[definition.name] = round(
                definition.low + rng.uniform() * (definition.high - definition.low), 6,
            )
        elif definition.kind == "integer":
            values[definition.name] = rng.randint(
                int(definition.low), int(definition.high),
            )
        else:
            values[definition.name] = rng.choice(definition.choices)
    return ParameterGenome(values, space)


def parameter_safe_perturbation(
    baseline: ParameterGenome, rng: SeededRandom, *, sigma: float = 0.05,
) -> ParameterGenome:
    """Small perturbation of the known-good baseline (step 18: safe perturbations)."""
    updates: dict[str, Any] = {}
    for definition in baseline.space:
        if definition.kind == "continuous":
            span = definition.high - definition.low
            updates[definition.name] = (
                baseline.values[definition.name] + rng.gauss(0.0, sigma * span)
            )
    return baseline.clone_with(updates)


# ------------------------------------------------------------------ policy tables
def _random_actions(rng: SeededRandom, modules: tuple[str, ...]) -> frozenset[ActionCall]:
    count = rng.randint(1, min(3, len(modules)))
    chosen = rng.sample(list(modules), count)
    actions: set[ActionCall] = {ActionCall("activate_module", {"module": m}) for m in chosen}
    return frozenset(actions)


def policy_table_random(
    rng: SeededRandom,
    *,
    defer_probability: float = 0.15,
    escalate_safety: bool = True,
) -> PolicyTableGenome:
    """Random valid policy table across the finite context schema."""
    from ..evaluation.benchmark_manifest import RESOURCE_STATES, DIFFICULTY_TIERS
    from ..evaluation.benchmark_manifest import TASK_FAMILIES
    rules: dict[tuple[str, ...], frozenset[ActionCall]] = {}
    for family in TASK_FAMILIES:
        for difficulty in DIFFICULTY_TIERS:
            for safety in ("no", "yes"):
                for resource in RESOURCE_STATES:
                    key = table_context_key(family, difficulty, safety == "yes", resource)
                    if rng.uniform() < defer_probability:
                        rules[key] = frozenset({
                            ActionCall("defer_task", {"reason": "low_confidence"}),
                        })
                        continue
                    actions = set(_random_actions(rng, ORE_MODULES))
                    if safety == "yes" and escalate_safety and rng.uniform() < 0.8:
                        actions.add(ActionCall("escalate_verification", {}))
                    if rng.uniform() < 0.3:
                        actions.add(ActionCall(
                            "request_retrieval", {"depth": rng.randint(0, 3)},
                        ))
                    rules[key] = frozenset(actions)
    return PolicyTableGenome(rules)


def policy_table_crossover(
    parent_a: PolicyTableGenome, parent_b: PolicyTableGenome, rng: SeededRandom,
) -> PolicyTableGenome | None:
    """Per-cell recombination; rejected when the offspring breaks the schema."""
    if parent_a.context_dimensions != parent_b.context_dimensions:
        return None
    merged: dict[tuple[str, ...], frozenset[ActionCall]] = {}
    keys = set(parent_a.rules) | set(parent_b.rules)
    for key in keys:
        take_a = key not in parent_b.rules or (key in parent_a.rules and rng.uniform() < 0.5)
        source = parent_a if take_a else parent_b
        merged[key] = source.rules.get(key, frozenset())
    offspring = PolicyTableGenome(merged)
    return offspring if not offspring.validate() else None


def policy_table_mutate(
    genome: PolicyTableGenome,
    rng: SeededRandom,
    *,
    cell_probability: float = 0.3,
) -> PolicyTableGenome | None:
    """Mutate a random subset of cells with approved actions only."""
    rules: dict[tuple[str, ...], frozenset[ActionCall]] = dict(genome.rules)
    for key in list(rules):
        if rng.uniform() > cell_probability:
            continue
        current = set(rules.get(key, frozenset()))
        mode = rng.uniform()
        if mode < 0.4:  # add an approved action
            module = rng.choice(ORE_MODULES)
            current.add(ActionCall("activate_module", {"module": module}))
        elif mode < 0.7 and current:  # drop one action
            victim = rng.choice(sorted(current, key=lambda a: a.name))
            current.discard(victim)
        elif mode < 0.85:  # toggle verification escalation
            current.add(ActionCall("escalate_verification", {}))
        else:  # defer the cell
            current = {ActionCall("defer_task", {"reason": rng.choice(("low_confidence", "resource_pressure"))})}
        rules[key] = frozenset(current)
    offspring = PolicyTableGenome(rules)
    return offspring if not offspring.validate() else None


# ------------------------------------------------------------------ policy IR
def policy_ir_crossover(parent_a: Any, parent_b: Any, rng: SeededRandom):
    """Subtree crossover with type matching (delegates to the typed grammar)."""
    from .genetic_programming import subtree_crossover
    return subtree_crossover(parent_a, parent_b, rng)


def policy_ir_mutate(genome: Any, rng: SeededRandom):
    """Subtree mutation bounded by the grammar (step 34)."""
    from .genetic_programming import subtree_mutation
    return subtree_mutation(genome, rng)


def policy_ir_random(grammar: Any, rng: SeededRandom, max_depth: int | None = None):
    """Random bounded tree within the typed grammar (step 32)."""
    from .genetic_programming import random_tree
    return random_tree(grammar, rng, max_depth=max_depth)


# ------------------------------------------------------------------ dispatch
def crossover_genomes(parent_a: Any, parent_b: Any, rng: SeededRandom):
    """Dispatch crossover by genome kind; ``None`` means rejected (step 24)."""
    if parent_a.kind != parent_b.kind:
        return None
    if parent_a.kind == "parameters":
        return parameter_crossover(parent_a, parent_b, rng)
    if parent_a.kind == "policy_table":
        return policy_table_crossover(parent_a, parent_b, rng)
    if parent_a.kind == "policy_ir":
        return policy_ir_crossover(parent_a, parent_b, rng)
    raise ValueError(f"no crossover operator for genome kind {parent_a.kind!r}")


def mutate_genome(genome: Any, rng: SeededRandom):
    """Dispatch mutation by genome kind; ``None`` means rejected (step 25)."""
    if genome.kind == "parameters":
        return parameter_mutate(genome, rng)
    if genome.kind == "policy_table":
        return policy_table_mutate(genome, rng)
    if genome.kind == "policy_ir":
        return policy_ir_mutate(genome, rng)
    raise ValueError(f"no mutation operator for genome kind {genome.kind!r}")


def random_genome(genome_kind: str, rng: SeededRandom, *, space=None, grammar=None):
    """Random genome of the requested kind for diverse initialization (step 18)."""
    if genome_kind == "parameters":
        return parameter_random(space or _default_space(), rng)
    if genome_kind == "policy_table":
        return policy_table_random(rng)
    if genome_kind == "policy_ir":
        if grammar is None:
            raise ValueError("policy_ir random generation requires a typed grammar")
        from .genetic_programming import PolicyIRGenome
        return PolicyIRGenome(policy_ir_random(grammar, rng), grammar=grammar)
    raise ValueError(f"no random generator for genome kind {genome_kind!r}")


def _default_space() -> tuple[ParameterDefinition, ...]:
    from ..domain.genome import default_search_space
    return default_search_space()


def parsimony_preferred(candidate_a: tuple[float, int], candidate_b: tuple[float, int], *, epsilon: float = 0.005) -> bool:
    """Prefer the simpler candidate only when quality is comparable (step 35).

    Inputs are ``(fitness, complexity)`` pairs; returns True when ``a`` should rank
    ahead of ``b``.
    """
    fitness_a, complexity_a = candidate_a
    fitness_b, complexity_b = candidate_b
    if math.isclose(fitness_a, fitness_b, abs_tol=epsilon):
        return complexity_a <= complexity_b
    return fitness_a > fitness_b


__all__ = [
    "PARAMETER_GROUPS", "crossover_genomes", "mutate_genome", "parameter_crossover",
    "parameter_mutate", "parameter_random", "parameter_safe_perturbation",
    "parsimony_preferred", "policy_ir_crossover", "policy_ir_mutate",
    "policy_ir_random", "policy_table_crossover", "policy_table_mutate",
    "policy_table_random", "random_genome",
]
