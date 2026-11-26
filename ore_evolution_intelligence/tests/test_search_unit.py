"""Search-layer unit tests (Phase C; acceptance tests 15, 22-23, 25)."""
from __future__ import annotations

import math

import pytest

from ore_evolution_intelligence.evaluation.evaluator import SeededRandom
from ore_evolution_intelligence.search import operators
from ore_evolution_intelligence.search.cmaes import CMAESAdapter
from ore_evolution_intelligence.search.evolution_strategy import (
    EvolutionStrategyOptimizer,
)
from ore_evolution_intelligence.search.genetic_optimizer import (
    GeneticOptimizer, GeneticSearchConfig, genome_complexity,
)
from ore_evolution_intelligence.search.novelty_search import (
    NoveltyArchive, NoveltyGuidedSelector, ParetoArchive, dominates,
    pareto_front_is_consistent,
)

from .fixtures import (
    SEED, make_baseline_candidate, make_baseline_genome, make_candidate,
    make_evaluator,
)


class TestOperators:
    """Steps 24-25: crossover and mutation yield valid genomes or are rejected."""

    def test_parameter_crossover_property(self):
        # Acceptance test 15: crossover yields valid genomes or is rejected.
        rng = SeededRandom(123)
        a = operators.parameter_random(make_baseline_genome().space, rng)
        b = operators.parameter_random(make_baseline_genome().space, rng)
        for _ in range(120):
            offspring = operators.parameter_crossover(a, b, rng)
            if offspring is not None:
                assert offspring.validate() == []
            else:
                assert a.space != b.space

    def test_parameter_mutation_property(self):
        rng = SeededRandom(456)
        genome = make_baseline_genome()
        for _ in range(120):
            mutated = operators.parameter_mutate(genome, rng)
            assert mutated is None or mutated.validate() == []

    def test_policy_table_operators_property(self):
        rng = SeededRandom(789)
        for _ in range(40):
            a = operators.policy_table_random(rng)
            b = operators.policy_table_random(rng)
            assert a.validate() == []
            child = operators.policy_table_crossover(a, b, rng)
            assert child is None or child.validate() == []
            mutant = operators.policy_table_mutate(a, rng)
            assert mutant is None or mutant.validate() == []

    def test_crossover_between_kinds_rejected(self):
        rng = SeededRandom(11)
        table = operators.policy_table_random(rng)
        assert operators.crossover_genomes(make_baseline_genome(), table, rng) is None

    def test_parsimony_prefers_simpler_when_comparable(self):
        # Step 35: prefer simpler candidates only when quality is comparable.
        assert operators.parsimony_preferred((0.900, 5), (0.9001, 50))
        assert not operators.parsimony_preferred((0.800, 5), (0.900, 50))


class TestGeneticSearch:
    """Steps 18-29: bounded GA with gates, elites, diversity and stopping."""

    def _optimizer(self, **config) -> GeneticOptimizer:
        evaluator = make_evaluator(trials_per_task=2)
        values = dict(
            population_size=8, generation_limit=3, seed=SEED,
        )
        values.update(config)
        return GeneticOptimizer(evaluator, config=GeneticSearchConfig(**values))

    def test_ga_runs_and_reports(self):
        optimizer = self._optimizer()
        report = optimizer.search(make_baseline_genome(), run_id="unit-ga")
        assert report.stop_reason in (
            "generation_limit", "convergence", "budget_exhausted",
            "unacceptable_failure_rate",
        )
        assert report.generations_executed <= 3
        assert report.evaluations_consumed >= 8
        # Acceptance test 3: every run records the objective-side evidence.
        assert report.config["seed"] == SEED
        assert report.train_task_count > 0 and report.validation_task_count > 0

    def test_ga_budget_exhaustion_stops_cleanly(self):
        # Acceptance test 24: budget exhaustion stops the run cleanly.
        from ore_evolution_intelligence.evaluation.evaluator import EvaluationBudget
        evaluator = make_evaluator(trials_per_task=2)
        optimizer = GeneticOptimizer(
            evaluator, config=GeneticSearchConfig(
                population_size=8, generation_limit=5, seed=SEED,
            ),
        )
        budget = EvaluationBudget(total_evaluations=10)
        report = optimizer.search(
            make_baseline_genome(), run_id="budget-ga", budget=budget,
        )
        assert report.stop_reason == "budget_exhausted"
        assert report.evaluations_consumed <= 12

    def test_ga_improves_or_reports_baseline(self):
        optimizer = self._optimizer(generation_limit=6)
        report = optimizer.search(make_baseline_genome(), run_id="improve-ga")
        # Either an improved candidate or an honest no-improvement outcome.
        if report.best_candidate is not None:
            assert report.best_genome_dict is not None
        else:
            assert report.best_candidate_id is None

    def test_genome_complexity_by_kind(self):
        assert genome_complexity(make_baseline_genome()) == 0
        rng = SeededRandom(3)
        table = operators.policy_table_random(rng)
        assert genome_complexity(table) == len(table.rules)


class TestEvolutionStrategy:
    """Section 2.3: (mu + lambda) ES with gaussian perturbation."""

    def test_es_improves_simple_landscape(self):
        space = make_baseline_genome().space
        baseline = make_baseline_genome()

        def objective(genome):
            # Maximize the sum of normalized continuous coordinates.
            return sum(genome.normalized_vector())

        optimizer = EvolutionStrategyOptimizer(
            mu=3, lambda_=6, sigma=0.2, generation_limit=12, seed=SEED,
        )
        result = optimizer.optimize(baseline, objective)
        assert result.best_fitness is not None
        assert result.best_fitness > objective(baseline)

    def test_es_stops_on_budget(self):
        from ore_evolution_intelligence.evaluation.evaluator import EvaluationBudget

        baseline = make_baseline_genome()
        remaining = EvaluationBudget(total_evaluations=6)

        def objective(genome):
            remaining.consume(1)  # budget enforced inside the objective
            return sum(genome.normalized_vector())

        optimizer = EvolutionStrategyOptimizer(
            mu=2, lambda_=4, generation_limit=50, seed=SEED,
        )
        result = optimizer.optimize(baseline, objective)
        assert result.stop_reason == "budget_exhausted"


class TestCMAES:
    """Step 30: CMA-ES for continuous correlated vectors only."""

    def test_cmaes_rejects_discrete_spaces(self):
        baseline = make_baseline_genome()  # has integer + choice params
        adapter = CMAESAdapter(generation_limit=5)
        with pytest.raises(ValueError, match="continuous"):
            adapter.optimize(baseline, lambda genome: 0.0)

    def test_cmaes_converges_on_quadratic(self):
        space = make_baseline_genome().space
        # A purely continuous space: drop the discrete dimensions.
        from ore_evolution_intelligence.domain.genome import ParameterDefinition
        continuous_space = tuple(
            ParameterDefinition(
                d.name, "continuous", low=d.low, high=d.high,
                default=d.default, description=d.description,
            )
            for d in space if d.kind == "continuous"
        )
        values = {d.name: d.default_value() for d in continuous_space}
        # Target: all continuous params at their upper bound (normalized 1.0).
        from ore_evolution_intelligence.domain.genome import ParameterGenome
        baseline = ParameterGenome(values, continuous_space)

        def objective(genome):
            vector = genome.normalized_vector()
            return -sum((v - 1.0) ** 2 for v in vector)

        adapter = CMAESAdapter(sigma0=0.25, generation_limit=25, seed=SEED)
        result = adapter.optimize(baseline, objective)
        assert result.best_fitness is not None
        assert result.best_fitness > objective(baseline)
        best_vector = result.best_genome.normalized_vector()
        assert sum(best_vector) / len(best_vector) > 0.7

    def test_cmaes_handles_infeasible_as_worst(self):
        from ore_evolution_intelligence.domain.genome import (
            ParameterDefinition, ParameterGenome,
        )
        space = tuple(
            ParameterDefinition(f"p{i}", "continuous", low=0.0, high=1.0)
            for i in range(4)
        )
        baseline = ParameterGenome(
            {d.name: 0.5 for d in space}, space,
        )

        def objective(genome):
            # Infeasible whenever the first parameter exceeds 0.5.
            if genome.values["p0"] > 0.5:
                return None
            return genome.values["p0"] + 0.001 * genome.values["p1"]

        adapter = CMAESAdapter(sigma0=0.2, generation_limit=15, seed=SEED)
        result = adapter.optimize(baseline, objective)
        assert result.best_genome.values["p0"] <= 0.5


class TestNoveltyAndPareto:
    """Section 2.5: novelty and Pareto archives (acceptance tests 22, 23)."""

    def test_novelty_is_knn_mean_distance(self):
        archive = NoveltyArchive(k=3)
        descriptors = []
        for index in range(6):
            descriptor = _descriptor(speed=100.0 + 50.0 * index)
            archive.observe(descriptor)
            descriptors.append(descriptor)
        far = _descriptor(speed=900.0)
        near = _descriptor(speed=115.0)
        report_far = archive.novelty(far)
        report_near = archive.novelty(near)
        assert report_far.novelty > report_near.novelty
        assert report_far.neighbors == 3

    def test_novelty_cannot_bypass_quality_gates(self):
        # Acceptance test 22: novelty cannot bypass quality/safety gates.
        selector = NoveltyGuidedSelector(novelty_weight=5.0)
        selector.observe(_descriptor(speed=100.0))
        infeasible_score = selector.combined_score(None, _descriptor(speed=900.0))
        feasible_score = selector.combined_score(0.5, _descriptor(speed=900.0))
        assert infeasible_score == float("-inf")
        assert feasible_score > float("-inf")

    def test_domination_definition(self):
        a = {"quality": 0.9, "reliability": 0.9, "latency_mean_ms": 100.0, "cost": 10.0}
        b = {"quality": 0.8, "reliability": 0.9, "latency_mean_ms": 120.0, "cost": 10.0}
        assert dominates(a, b)
        assert not dominates(b, a)
        # Equal on all objectives: no domination.
        assert not dominates(a, a)

    def test_pareto_archive_contains_only_non_dominated(self):
        # Acceptance test 23: Pareto archive contains only non-dominated candidates.
        archive = ParetoArchive()
        archive.insert("a", {"quality": 0.9, "reliability": 0.8,
                              "latency_mean_ms": 100.0, "cost": 10.0})
        archive.insert("b", {"quality": 0.8, "reliability": 0.9,
                              "latency_mean_ms": 110.0, "cost": 12.0})
        # Dominated by a and b on all axes: rejected.
        update = archive.insert("c", {"quality": 0.7, "reliability": 0.7,
                                      "latency_mean_ms": 150.0, "cost": 20.0})
        assert not update.accepted
        # Dominates b strictly on quality and latency: replaces it.
        update = archive.insert("d", {"quality": 0.95, "reliability": 0.9,
                                      "latency_mean_ms": 90.0, "cost": 11.0})
        assert update.accepted
        entries = archive.non_dominated()
        assert {entry.key for entry in entries} == {"a", "d"}
        assert pareto_front_is_consistent(entries)

    def test_pareto_missing_objective_rejected(self):
        archive = ParetoArchive()
        update = archive.insert("x", {"quality": 0.9})
        assert not update.accepted
        assert "missing objective" in update.reason


def _descriptor(speed: float):
    from ore_evolution_intelligence.domain.behavior_descriptor import (
        BehaviorDescriptor,
    )
    return BehaviorDescriptor(
        {
            "speed": speed, "reasoning_depth": 4.0, "resource_tier": 0.5,
            "reliability": 0.8, "module_cooperation": 0.6,
        },
        "calibration-v1",
    )


class TestGeneticOptimizerHelpers:
    def test_tournament_prefers_higher_fitness(self):
        evaluator = make_evaluator(trials_per_task=1)
        optimizer = GeneticOptimizer(
            evaluator, config=GeneticSearchConfig(
                population_size=6, generation_limit=1, tournament_size=3, seed=SEED,
            ),
        )
        class FakeRecord:
            def __init__(self, fitness, feasible):
                self.fitness = fitness
                self.gate = type("G", (), {"feasible": feasible})()

        from ore_evolution_intelligence.domain.strategy import StrategyCandidate
        candidates = []
        records = {}
        for index in range(6):
            candidate = StrategyCandidate.from_genome(
                operators.parameter_random(make_baseline_genome().space, SeededRandom(index)),
            )
            candidates.append(candidate)
            records[candidate.candidate_id] = FakeRecord(0.1 * index, True)
        winner = optimizer._tournament(candidates, records, SeededRandom(5))
        assert records[winner.candidate_id].fitness >= 0.4
