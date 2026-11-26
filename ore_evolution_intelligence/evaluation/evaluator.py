"""Independent multi-objective strategy evaluator (Phase B, steps 14-17, 20-22).

The evaluator is separate from candidates (step 14): genomes are data whose
interpretation happens inside the sandbox; candidates cannot reach scoring code,
labels, the environment clock, task selection or validation data. Stochastic trials
(step 15) use multiple seeds per task and record mean, dispersion, confidence intervals
and worst-case behavior. Raw observations are always preserved (step 17) — never only
a fitness scalar. Benchmark-hardcoding detection (step 16) flags suspicious constant
output across distinct-feature tasks.
"""
from __future__ import annotations

from hashlib import sha256
from typing import Any, Iterable, Sequence

import numpy as np

from ..domain.evaluation_record import (
    AggregateMetrics, EvaluationRecord, GateDecision, TaskOutcomeRecord,
)
from ..domain.strategy import StrategyCandidate
from .benchmark_manifest import BenchmarkManifest, TaskCase
from .sandbox import SandboxedStrategyExecutor


class BudgetExhausted(RuntimeError):
    """Raised when the evaluation budget is exhausted (step 28: clean stop)."""


class EvaluationBudget:
    """Mutable evaluation-budget tracker enforced across the whole run (step 12).

    One budget unit is one candidate evaluation (all tasks x trials of one candidate
    on one split). Generation and population caps live in the search configuration;
    total evaluation cap lives here.
    """

    def __init__(self, total_evaluations: int) -> None:
        if total_evaluations < 1:
            raise ValueError("total_evaluations must be positive")
        self.total_evaluations = total_evaluations
        self.consumed = 0

    def consume(self, units: int = 1) -> None:
        if self.consumed + units > self.total_evaluations:
            raise BudgetExhausted(
                f"evaluation budget exhausted ({self.consumed}/{self.total_evaluations})"
            )
        self.consumed += units

    @property
    def remaining(self) -> int:
        return max(self.total_evaluations - self.consumed, 0)

    def as_dict(self) -> dict[str, int]:
        return {"total": self.total_evaluations, "consumed": self.consumed}


class SeededRandom:
    """Deterministic RNG facade (``seed`` attribute preserved for traceability)."""

    def __init__(self, seed: int) -> None:
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed % (2**63))

    def gauss(self, mu: float, sigma: float) -> float:
        return float(self._rng.normal(mu, sigma))

    def uniform(self) -> float:
        return float(self._rng.random())

    def randint(self, low: int, high: int) -> int:  # inclusive range
        return int(self._rng.integers(low, high + 1))

    def choice(self, sequence: Sequence[Any]) -> Any:
        return sequence[int(self._rng.integers(0, len(sequence)))]

    def sample(self, sequence: Sequence[Any], k: int) -> list[Any]:
        return list(self._rng.choice(np.asarray(list(sequence), dtype=object), size=k, replace=False))

    def shuffle(self, sequence: list[Any]) -> None:
        order = list(self._rng.permutation(len(sequence)))
        original = list(sequence)
        sequence[:] = [original[i] for i in order]


def task_stream_seed(seed: int, task_id: str) -> int:
    """Deterministic per-task RNG seed (stable across processes and runs)."""
    digest = sha256(f"{seed}:{task_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % (2**31)


#: Number of distinct-feature tasks required before hardcoding detection applies.
HARDCODING_MIN_TASKS = 6


class StrategyEvaluator:
    """Evaluates candidates on search-visible splits with repeated trials (steps 15, 20)."""

    def __init__(
        self,
        benchmark: BenchmarkManifest,
        *,
        sandbox: SandboxedStrategyExecutor | None = None,
        constraints: Any = None,
        fitness: Any = None,
        trials_per_task: int = 3,
    ) -> None:
        if trials_per_task < 1:
            raise ValueError("trials_per_task must be >= 1")
        self.benchmark = benchmark
        self.sandbox = sandbox or SandboxedStrategyExecutor()
        from ..domain.evaluation_record import HardConstraints, ScalarFitness
        self.constraints = constraints or HardConstraints()
        self.fitness = fitness or ScalarFitness()
        self.trials_per_task = trials_per_task

    # --------------------------------------------------------------- search-facing
    def evaluate_candidate(
        self,
        candidate: StrategyCandidate,
        *,
        split: str = "train",
        seeds: Sequence[int] | None = None,
        budget: EvaluationBudget | None = None,
    ) -> EvaluationRecord:
        """Evaluate one candidate on a search-visible split (step 20).

        Only ``train`` and ``validation`` splits are reachable here; the held-out test
        split is opened solely by the held-out validator after search ends.
        """
        if split not in ("train", "validation"):
            raise ValueError(
                f"search-facing evaluation cannot use split {split!r} "
                "(held-out test data is sealed)"
            )
        tasks = (
            self.benchmark.training_tasks() if split == "train"
            else self.benchmark.validation_tasks()
        )
        if seeds is None:
            seeds = [self.benchmark.seed + i for i in range(self.trials_per_task)]
        return self._evaluate_on_tasks(candidate, tasks, split, tuple(seeds), budget)

    def evaluate_on_tasks(
        self,
        candidate: StrategyCandidate,
        tasks: Sequence[TaskCase],
        *,
        split: str,
        seeds: Sequence[int],
        budget: EvaluationBudget | None = None,
    ) -> EvaluationRecord:
        """Evaluate on an explicit task list — held-out validator use only (step 29)."""
        if split not in ("heldout_test", "canary"):
            raise ValueError(f"explicit task evaluation requires split 'heldout_test' or 'canary', got {split!r}")
        return self._evaluate_on_tasks(candidate, list(tasks), split, tuple(seeds), budget)

    # -------------------------------------------------------------------- internals
    def _evaluate_on_tasks(
        self,
        candidate: StrategyCandidate,
        tasks: Sequence[TaskCase],
        split: str,
        seeds: tuple[int, ...],
        budget: EvaluationBudget | None,
    ) -> EvaluationRecord:
        if budget is not None:
            budget.consume(1)
        outcomes: list[TaskOutcomeRecord] = []
        for trial, seed in enumerate(seeds):
            for task in tasks:
                rng = SeededRandom(task_stream_seed(seed, task.task_id))
                record, _trace = self.sandbox.execute(candidate.genome, task, rng)
                outcomes.append(record)

        aggregates = AggregateMetrics.from_outcomes(outcomes, n_tasks=len(tasks))
        gate: GateDecision = self.constraints.check(aggregates)
        fitness_value: float | None = None
        if gate.feasible:
            fitness_value = self.fitness.score(aggregates)

        invalid_reasons = self._collect_invalid_reasons(outcomes)
        hardcoding = self._detect_hardcoding(tasks, outcomes, aggregates)
        if hardcoding is not None:
            invalid_reasons.append(hardcoding)
            gate = GateDecision(
                feasible=False, reasons=tuple((*gate.reasons, hardcoding)),
            )
            fitness_value = None

        behavior_measurement = {
            "speed": aggregates.latency_mean_ms,
            "reasoning_depth": aggregates.mean_active_modules,
            "resource_tier": aggregates.cost_mean,
            "reliability": aggregates.reliability,
            "module_cooperation": aggregates.cooperation,
        }
        return EvaluationRecord(
            candidate_id=candidate.candidate_id,
            candidate_version=candidate.version,
            genome_hash=candidate.content_hash,
            benchmark_id=self.benchmark.benchmark_id,
            benchmark_version=self.benchmark.version,
            split=split,
            seeds=tuple(seeds),
            metrics=aggregates,
            outcomes=tuple(outcomes),
            gate=gate,
            fitness=fitness_value,
            invalid_reasons=tuple(invalid_reasons),
            behavior_measurement=behavior_measurement,
        )

    def _collect_invalid_reasons(self, outcomes: Iterable[TaskOutcomeRecord]) -> list[str]:
        reasons: list[str] = []
        seen: set[str] = set()
        for outcome in outcomes:
            if outcome.outcome != "INVALID":
                continue
            for entry in (outcome.evidence.get("trace", {}).get("exceptions") or [])[:1]:
                if entry not in seen:
                    seen.add(entry)
                    reasons.append(entry)
            for note in (outcome.evidence.get("trace", {}).get("notes") or [])[:1]:
                if note not in seen:
                    seen.add(note)
                    reasons.append(note)
        return reasons

    def _detect_hardcoding(
        self,
        tasks: Sequence[TaskCase],
        outcomes: Sequence[TaskOutcomeRecord],
        aggregates: AggregateMetrics,
    ) -> str | None:
        """Detect benchmark hardcoding: constant output over distinct-feature tasks.

        A real strategy evaluated with seeded noise cannot produce bit-identical
        quality across many distinct tasks; a candidate hardcoding the benchmark can
        (step 16: reject benchmark hardcoding).
        """
        if len(tasks) < HARDCODING_MIN_TASKS:
            return None
        if aggregates.quality < self.constraints.q_min:
            return None  # infeasible already; nothing to protect
        distinct_features = {tuple(sorted(task.features().items())) for task in tasks}
        if len(distinct_features) < HARDCODING_MIN_TASKS:
            return None
        per_task_quality: dict[str, float] = {}
        for outcome in outcomes:
            per_task_quality.setdefault(outcome.task_id, outcome.quality)
        if len(per_task_quality) < HARDCODING_MIN_TASKS:
            return None
        qualities = list(per_task_quality.values())
        if all(q == qualities[0] for q in qualities) and qualities[0] > 0:
            return (
                "benchmark hardcoding suspected: identical quality across "
                f"{len(qualities)} distinct-feature tasks"
            )
        return None


__all__ = [
    "BudgetExhausted", "EvaluationBudget", "HARDCODING_MIN_TASKS", "SeededRandom",
    "StrategyEvaluator", "task_stream_seed",
]
