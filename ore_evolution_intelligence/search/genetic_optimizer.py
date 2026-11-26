"""Genetic algorithm for bounded cognitive-strategy search (Phase C, steps 18-29).

Population evolution with tournament selection (step 23), group/subtree crossover
(step 24), documented mutation (step 25), elitism with a diversity guard (step 26),
ancestry-preserving replacement (step 27), responsible stopping (step 28) and
baseline comparison on validation tasks (step 29). Hard constraints are gates, never
penalties (step 21): infeasible candidates are excluded from parent selection — speed
cannot compensate for unsafe behavior. Duplicate candidate hashes are deduplicated and
candidate lineage stays reconstructable (acceptance tests 23, 24).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..domain.genome import default_search_space
from ..domain.strategy import LineageRecord, StrategyCandidate
from ..evaluation.evaluator import BudgetExhausted, SeededRandom, StrategyEvaluator
from . import operators


@dataclass(frozen=True, slots=True)
class GeneticSearchConfig:
    """Documented GA policy choices (steps 18, 22, 23, 26, 28)."""

    population_size: int = 12
    tournament_size: int = 3
    crossover_probability: float = 0.7
    mutation_probability: float = 0.4
    elitism: int = 2
    diverse_injections: int = 2
    generation_limit: int = 8
    stall_generations: int = 3
    min_lineages: int = 4
    unacceptable_failure_generations: int = 3
    parsimony_epsilon: float = 0.005
    operator_retry_limit: int = 3
    seed: int = 20261122

    def as_dict(self) -> dict[str, Any]:
        return {
            "population_size": self.population_size,
            "tournament_size": self.tournament_size,
            "crossover_probability": self.crossover_probability,
            "mutation_probability": self.mutation_probability,
            "elitism": self.elitism,
            "diverse_injections": self.diverse_injections,
            "generation_limit": self.generation_limit,
            "stall_generations": self.stall_generations,
            "min_lineages": self.min_lineages,
            "unacceptable_failure_generations": self.unacceptable_failure_generations,
            "parsimony_epsilon": self.parsimony_epsilon,
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class GenerationStats:
    """One generation's record (step 27: ancestry and deterministic seeds kept)."""

    generation: int
    best_fitness: float | None
    mean_fitness: float | None
    feasible_fraction: float
    unique_lineages: int
    population_size: int
    duplicates_rejected: int
    crossover_attempts: int = 0
    crossover_accepted: int = 0
    mutation_attempts: int = 0
    mutation_accepted: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "generation": self.generation, "best_fitness": self.best_fitness,
            "mean_fitness": self.mean_fitness, "feasible_fraction": self.feasible_fraction,
            "unique_lineages": self.unique_lineages,
            "population_size": self.population_size,
            "duplicates_rejected": self.duplicates_rejected,
            "crossover_attempts": self.crossover_attempts,
            "crossover_accepted": self.crossover_accepted,
            "mutation_attempts": self.mutation_attempts,
            "mutation_accepted": self.mutation_accepted,
        }


@dataclass(frozen=True, slots=True)
class EvolutionSearchReport:
    """Search-level report (steps 27-29; section 8 evolution-efficiency metrics)."""

    run_id: str
    genome_kind: str
    config: dict[str, Any]
    stop_reason: str
    generations_executed: int
    evaluations_consumed: int
    evaluations_to_improvement: int | None
    best_candidate_id: str | None
    best_train_fitness: float | None
    best_validation_fitness: float | None
    baseline_validation_fitness: float | None
    quality_improvement: float | None
    history: tuple[GenerationStats, ...] = ()
    duplicate_rejections: int = 0
    mutation_acceptance_rate: float | None = None
    crossover_acceptance_rate: float | None = None
    best_genome_dict: dict[str, Any] | None = None
    finalist_ids: tuple[str, ...] = ()
    train_task_count: int = 0
    validation_task_count: int = 0
    # Candidate objects carried for the orchestration layer (not serialized).
    best_candidate: Any = None
    finalists: tuple = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id, "genome_kind": self.genome_kind,
            "config": self.config, "stop_reason": self.stop_reason,
            "generations_executed": self.generations_executed,
            "evaluations_consumed": self.evaluations_consumed,
            "evaluations_to_improvement": self.evaluations_to_improvement,
            "best_candidate_id": self.best_candidate_id,
            "best_train_fitness": self.best_train_fitness,
            "best_validation_fitness": self.best_validation_fitness,
            "baseline_validation_fitness": self.baseline_validation_fitness,
            "quality_improvement": self.quality_improvement,
            "history": [entry.as_dict() for entry in self.history],
            "duplicate_rejections": self.duplicate_rejections,
            "mutation_acceptance_rate": self.mutation_acceptance_rate,
            "crossover_acceptance_rate": self.crossover_acceptance_rate,
            "best_genome_dict": self.best_genome_dict,
            "finalist_ids": list(self.finalist_ids),
            "train_task_count": self.train_task_count,
            "validation_task_count": self.validation_task_count,
        }


def genome_complexity(genome: Any) -> int:
    """Complexity metric for parsimony ranking (step 35)."""
    if genome.kind == "parameters":
        return 0
    if genome.kind == "policy_table":
        return len(genome.rules)
    if genome.kind == "policy_ir":
        return genome.root.size()
    return 0


class GeneticOptimizer:
    """Bounded genetic search over one narrow objective (steps 18-29)."""

    def __init__(
        self,
        evaluator: StrategyEvaluator,
        *,
        config: GeneticSearchConfig | None = None,
        event_recorder: Any | None = None,
        grammar: Any | None = None,
    ) -> None:
        self.evaluator = evaluator
        self.config = config or GeneticSearchConfig()
        self.event_recorder = event_recorder
        self.grammar = grammar

    # ------------------------------------------------------------------ public
    def search(
        self,
        baseline_genome: Any,
        *,
        run_id: str = "run",
        budget: Any | None = None,
    ) -> EvolutionSearchReport:
        """Run the bounded GA; never mutates the baseline genome in place."""
        rng = SeededRandom(self.config.seed)
        baseline = StrategyCandidate.from_genome(
            baseline_genome,
            lineage=LineageRecord(operation="initial", generation=0, run_id=run_id),
            run_id=run_id,
        )
        # Step 18: baseline + safe perturbations + diverse configurations.
        population = self._initial_population(baseline, rng, run_id)
        if self.event_recorder is not None:
            self.event_recorder.emit(
                "population_initialized", run_id,
                {"size": len(population), "kind": baseline_genome.kind},
            )

        train_seeds = tuple(
            self.config.seed + 7 * i + 1 for i in range(self.evaluator.trials_per_task)
        )
        evaluation_records: dict[str, Any] = {}
        candidates_by_id: dict[str, StrategyCandidate] = {
            candidate.candidate_id: candidate for candidate in population
        }
        seen_hashes: set[str] = {baseline.content_hash}
        duplicate_rejections = 0
        crossover_attempts = crossover_accepted = 0
        mutation_attempts = mutation_accepted = 0
        history: list[GenerationStats] = []
        stop_reason = "generation_limit"
        best_fitness: float | None = None
        evaluations_at_improvement: int | None = None
        all_infeasible_streak = 0

        for generation in range(1, self.config.generation_limit + 1):
            try:
                self._evaluate_population(
                    population, evaluation_records, train_seeds, budget,
                )
            except BudgetExhausted:
                stop_reason = "budget_exhausted"
                history.append(GenerationStats(
                    generation=generation, best_fitness=best_fitness,
                    mean_fitness=None, feasible_fraction=0.0,
                    unique_lineages=len({c.lineage.parents[0] if c.lineage.parents else c.candidate_id for c in population}),
                    population_size=len(population), duplicates_rejected=duplicate_rejections,
                ))
                break
            for candidate in population:
                candidates_by_id.setdefault(candidate.candidate_id, candidate)

            feasible, generation_best, generation_mean = self._rank_population(
                population, evaluation_records,
            )
            history.append(GenerationStats(
                generation=generation,
                best_fitness=generation_best,
                mean_fitness=generation_mean,
                feasible_fraction=len(feasible) / max(len(population), 1),
                unique_lineages=self._unique_lineages(population),
                population_size=len(population),
                duplicates_rejected=duplicate_rejections,
            ))
            if self.event_recorder is not None:
                self.event_recorder.emit(
                    "generation_completed", run_id,
                    {"generation": generation, "best_fitness": generation_best,
                     "feasible_fraction": len(feasible) / max(len(population), 1),
                     "unique_lineages": self._unique_lineages(population)},
                )

            # Improvement tracking (evaluations to improvement, section 8).
            if generation_best is not None and (best_fitness is None or generation_best > best_fitness + 1e-9):
                best_fitness = generation_best
                evaluations_at_improvement = self._total_evaluations(
                    evaluation_records,
                )

            # Step 28: responsible stopping.
            if not feasible:
                all_infeasible_streak += 1
                if all_infeasible_streak >= self.config.unacceptable_failure_generations:
                    stop_reason = "unacceptable_failure_rate"
                    break
            else:
                all_infeasible_streak = 0
            if best_fitness is not None and self._stalled(history, self.config.stall_generations):
                stop_reason = "convergence"
                break
            if generation >= self.config.generation_limit:
                break

            # Steps 23-27: build the next generation.
            population, stats = self._next_generation(
                population, feasible, evaluation_records, rng, run_id,
                seen_hashes,
            )
            duplicate_rejections += stats["duplicates"]
            crossover_attempts += stats["crossover_attempts"]
            crossover_accepted += stats["crossover_accepted"]
            mutation_attempts += stats["mutation_attempts"]
            mutation_accepted += stats["mutation_accepted"]
            for candidate in population:
                candidates_by_id.setdefault(candidate.candidate_id, candidate)

        # Step 29: finalists re-evaluated with the baseline on validation tasks.
        best, best_validation, baseline_validation, finalist_candidates = self._validation_comparison(
            population, evaluation_records, candidates_by_id, baseline, budget,
        )
        if self.event_recorder is not None:
            self.event_recorder.emit(
                "search_stopped", run_id,
                {"stop_reason": stop_reason, "best_candidate_id": best.candidate_id if best else None},
            )
        quality_improvement = None
        if best_validation is not None and baseline_validation is not None:
            quality_improvement = (
                best_validation.metrics.quality - baseline_validation.metrics.quality
            )
        return EvolutionSearchReport(
            run_id=run_id,
            genome_kind=baseline_genome.kind,
            config=self.config.as_dict(),
            stop_reason=stop_reason,
            generations_executed=len(history),
            evaluations_consumed=self._total_evaluations(evaluation_records),
            evaluations_to_improvement=evaluations_at_improvement,
            best_candidate_id=best.candidate_id if best else None,
            best_train_fitness=(
                evaluation_records[best.candidate_id].fitness if best and best.candidate_id in evaluation_records else None
            ),
            best_validation_fitness=(
                best_validation.fitness if best_validation is not None else None
            ),
            baseline_validation_fitness=(
                baseline_validation.fitness if baseline_validation is not None else None
            ),
            quality_improvement=quality_improvement,
            history=tuple(history),
            duplicate_rejections=duplicate_rejections,
            mutation_acceptance_rate=(
                mutation_accepted / mutation_attempts if mutation_attempts else None
            ),
            crossover_acceptance_rate=(
                crossover_accepted / crossover_attempts if crossover_attempts else None
            ),
            best_genome_dict=best.genome.as_dict() if best else None,
            finalist_ids=tuple(c.candidate_id for c in finalist_candidates),
            train_task_count=len(self.evaluator.benchmark.training_tasks()),
            validation_task_count=len(self.evaluator.benchmark.validation_tasks()),
            best_candidate=best,
            finalists=tuple(finalist_candidates),
        )

    # -------------------------------------------------------------- internals
    def _initial_population(
        self, baseline: StrategyCandidate, rng: SeededRandom, run_id: str,
    ) -> list[StrategyCandidate]:
        config = self.config
        population: list[StrategyCandidate] = [baseline]
        # Safe perturbations of the known-good baseline (step 18).
        for _index in range(max(config.population_size // 4, 1)):
            genome = operators.parameter_safe_perturbation(baseline.genome, rng) \
                if baseline.genome.kind == "parameters" else \
                operators.mutate_genome(baseline.genome, rng)
            if genome is None:
                continue
            population.append(StrategyCandidate.from_genome(
                genome,
                lineage=LineageRecord(
                    parents=(baseline.candidate_id,), operation="mutation",
                    generation=0, run_id=run_id, notes="safe perturbation",
                ),
                run_id=run_id,
            ))
        # Diverse configurations (step 18).
        while len(population) < config.population_size:
            genome = operators.random_genome(
                baseline.genome.kind, rng,
                space=getattr(baseline.genome, "space", None) or default_search_space()
                if baseline.genome.kind == "parameters" else None,
                grammar=self.grammar,
            )
            candidate = StrategyCandidate.from_genome(
                genome,
                lineage=LineageRecord(operation="initial", generation=0, run_id=run_id,
                                      notes="diverse initialization"),
                run_id=run_id,
            )
            if candidate.content_hash in {c.content_hash for c in population}:
                continue
            population.append(candidate)
        return population

    def _evaluate_population(
        self,
        population: Sequence[StrategyCandidate],
        records: dict[str, Any],
        seeds: tuple[int, ...],
        budget: Any | None,
    ) -> None:
        for candidate in population:
            if candidate.candidate_id in records:
                continue
            records[candidate.candidate_id] = self.evaluator.evaluate_candidate(
                candidate, split="train", seeds=seeds, budget=budget,
            )
            if self.event_recorder is not None:
                self.event_recorder.emit(
                    "candidate_evaluated", candidate.run_id or "run",
                    {"candidate_id": candidate.candidate_id,
                     "feasible": records[candidate.candidate_id].gate.feasible,
                     "fitness": records[candidate.candidate_id].fitness},
                )

    def _rank_population(
        self, population: Sequence[StrategyCandidate], records: Mapping[str, Any],
    ) -> tuple[list[StrategyCandidate], float | None, float | None]:
        feasible: list[StrategyCandidate] = []
        for candidate in population:
            record = records.get(candidate.candidate_id)
            if record is not None and record.gate.feasible:
                feasible.append(candidate)
        ranked = sorted(
            feasible,
            key=lambda c: (
                -(records[c.candidate_id].fitness or 0.0),
                genome_complexity(c.genome),
            ),
        )
        fitnesses = [records[c.candidate_id].fitness or 0.0 for c in ranked]
        best = fitnesses[0] if fitnesses else None
        mean = sum(fitnesses) / len(fitnesses) if fitnesses else None
        return ranked, best, mean

    def _tournament(
        self,
        feasible: Sequence[StrategyCandidate],
        records: Mapping[str, Any],
        rng: SeededRandom,
    ) -> StrategyCandidate | None:
        """Tournament selection (step 23): best of k randomly sampled candidates."""
        if not feasible:
            return None
        k = min(self.config.tournament_size, len(feasible))
        contestants = rng.sample(list(feasible), k)
        winner = contestants[0]
        for other in contestants[1:]:
            a = (records[winner.candidate_id].fitness or 0.0, genome_complexity(winner.genome))
            b = (records[other.candidate_id].fitness or 0.0, genome_complexity(other.genome))
            if not operators.parsimony_preferred(a, b, epsilon=self.config.parsimony_epsilon):
                winner = other
        return winner

    def _next_generation(
        self,
        population: Sequence[StrategyCandidate],
        feasible: Sequence[StrategyCandidate],
        records: Mapping[str, Any],
        rng: SeededRandom,
        run_id: str,
        seen_hashes: set[str],
    ) -> tuple[list[StrategyCandidate], dict[str, int]]:
        config = self.config
        stats = {
            "duplicates": 0, "crossover_attempts": 0, "crossover_accepted": 0,
            "mutation_attempts": 0, "mutation_accepted": 0,
        }
        # Step 26: elites (feasible, high quality) keep their slots.
        elites = list(feasible[: config.elitism])
        next_population: list[StrategyCandidate] = []
        # Known-good baseline always retained (step 18: preserve a known-good candidate).
        baseline_entry = next(
            (c for c in population if c.lineage.operation == "initial" and not c.lineage.parents),
            None,
        )
        if baseline_entry is not None:
            next_population.append(baseline_entry)
        next_population.extend(elites)

        # Offspring via tournament parents (steps 23-25). When no feasible
        # candidate exists yet, fall back to exploring from *any* population
        # member so the search keeps moving instead of collapsing (step 26:
        # never let the infeasible known-good baseline eliminate exploration).
        parent_pool = feasible if feasible else list(population)
        target = config.population_size
        attempts = 0
        while (
            len(next_population) < target
            and parent_pool
            and attempts < target * config.operator_retry_limit * 2
        ):
            attempts += 1
            if feasible:
                parent_a = self._tournament(feasible, records, rng)
                parent_b = self._tournament(feasible, records, rng)
            else:
                parent_a = parent_pool[rng.randint(0, len(parent_pool) - 1)]
                parent_b = parent_pool[rng.randint(0, len(parent_pool) - 1)]
            if parent_a is None or parent_b is None:
                break
            genome = parent_a.genome
            if rng.uniform() < config.crossover_probability:
                stats["crossover_attempts"] += 1
                offspring = operators.crossover_genomes(parent_a.genome, parent_b.genome, rng)
                if offspring is not None:
                    stats["crossover_accepted"] += 1
                    genome = offspring
            if rng.uniform() < config.mutation_probability:
                stats["mutation_attempts"] += 1
                mutated = operators.mutate_genome(genome, rng)
                if mutated is not None:
                    stats["mutation_accepted"] += 1
                    genome = mutated
            lineage = LineageRecord(
                parents=(parent_a.candidate_id, parent_b.candidate_id),
                operation="crossover" if genome is not parent_a.genome else "mutation",
                generation=len(next_population),
                run_id=run_id,
            )
            candidate = StrategyCandidate.from_genome(genome, lineage=lineage, run_id=run_id)
            if candidate.content_hash in seen_hashes or any(
                c.content_hash == candidate.content_hash for c in next_population
            ):
                stats["duplicates"] += 1
                continue
            seen_hashes.add(candidate.content_hash)
            next_population.append(candidate)

        # Step 26 diversity guard: never let one lineage eliminate all alternatives.
        unique_lineages = self._unique_lineages(next_population)
        injected = 0
        while (
            unique_lineages < config.min_lineages
            and len(next_population) < target + config.diverse_injections
            and injected < config.diverse_injections
        ):
            genome = operators.random_genome(
                population[0].genome.kind, rng,
                space=getattr(population[0].genome, "space", None) or default_search_space()
                if population[0].genome.kind == "parameters" else None,
                grammar=self.grammar,
            )
            candidate = StrategyCandidate.from_genome(
                genome,
                lineage=LineageRecord(operation="initial", generation=0, run_id=run_id,
                                      notes="diversity injection"),
                run_id=run_id,
            )
            if any(c.content_hash == candidate.content_hash for c in next_population):
                break
            next_population.append(candidate)
            injected += 1
            unique_lineages = self._unique_lineages(next_population)
        return next_population[: max(target, len(elites) + 1)], stats

    def _validation_comparison(
        self,
        population: Sequence[StrategyCandidate],
        records: Mapping[str, Any],
        candidates_by_id: Mapping[str, StrategyCandidate],
        baseline: StrategyCandidate,
        budget: Any | None,
    ) -> tuple[StrategyCandidate | None, Any | None, Any | None, list[str]]:
        """Step 29: finalists and baseline on identical held-out validation tasks."""
        validation_seeds = tuple(
            self.config.seed + 31 * i + 11 for i in range(self.evaluator.trials_per_task)
        )
        ranked = [
            c for c in population if records.get(c.candidate_id) is not None
            and records[c.candidate_id].gate.feasible
        ]
        ranked = sorted(
            ranked,
            key=lambda c: (-(records[c.candidate_id].fitness or 0.0), genome_complexity(c.genome)),
        )
        finalists = ranked[:3]
        if not finalists:
            return None, None, None, []
        best_validation = None
        best = None
        baseline_validation = None
        try:
            baseline_validation = self.evaluator.evaluate_candidate(
                baseline, split="validation", seeds=validation_seeds, budget=budget,
            )
            for candidate in finalists:
                record = self.evaluator.evaluate_candidate(
                    candidate, split="validation", seeds=validation_seeds, budget=budget,
                )
                if record.gate.feasible and (
                    best_validation is None
                    or (record.fitness or 0.0) > (best_validation.fitness or 0.0)
                ):
                    best_validation = record
                    best = candidate
        except BudgetExhausted:
            pass
        return best, best_validation, baseline_validation, finalists

    def _unique_lineages(self, population: Sequence[StrategyCandidate]) -> int:
        roots: set[str] = set()
        for candidate in population:
            if candidate.lineage.parents:
                roots.add(candidate.lineage.parents[0])
            else:
                roots.add(candidate.candidate_id)
        return len(roots)

    def _stalled(self, history: Sequence[GenerationStats], stall_generations: int) -> bool:
        if len(history) < stall_generations + 1:
            return False
        recent = [entry.best_fitness for entry in history[-stall_generations - 1: -1]]
        if any(value is None for value in recent):
            return False
        current = history[-1].best_fitness
        if current is None:
            return False
        return all(current <= value + 1e-9 for value in recent if value is not None)

    @staticmethod
    def _total_evaluations(records: Mapping[str, Any]) -> int:
        return len(records)


__all__ = ["EvolutionSearchReport", "GeneticOptimizer", "GeneticSearchConfig", "GenerationStats", "genome_complexity"]
