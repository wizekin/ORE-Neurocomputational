"""Versioned benchmark manifests with sealed task splits (steps 4-5, 9-10, step 29).

Representative tasks (step 4) cover task families, difficulty tiers, resource states,
failure cases and rare safety-critical cases. The evaluation protocol is *versioned*
(step 9): task set, scoring configuration, constraints, seeds, runtime environment and
evaluator version are hashed — a changed evaluator yields a new benchmark version.

Evaluation data is split by a declared policy (step 10): training, validation and
held-out test tasks. The held-out test set is *sealed*: search-facing APIs expose only
training and validation tasks, and only :class:`HeldoutValidator` (this package) opens
the sealed set after search finishes (acceptance tests 11, 12).
"""
from __future__ import annotations

import platform
import sys
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..domain.evaluation_record import EVALUATOR_VERSION, OUTCOME_CREDIT
from ..domain.genome import canonical_json, content_digest
from ..domain.strategy import StrategyCandidate  # noqa: F401  (typing convenience)

#: Task families covered by the representative task suite (step 4).
TASK_FAMILIES: tuple[str, ...] = (
    "lexical_lookup", "semantic_composition", "syntax_parse",
    "reading", "discourse", "consolidation",
)

DIFFICULTY_TIERS: tuple[str, ...] = ("low", "medium", "high")
RESOURCE_STATES: tuple[str, ...] = ("normal", "pressured")

#: Declared split policy (step 10): stratified by source within each family.
SPLIT_POLICY = "by_source_stratified_v1"

#: Split fractions within each family (train / validation / test).
SPLIT_RATIONALE = "60/20/20 stratified by task family and source; test tasks sealed"


@dataclass(frozen=True, slots=True)
class TaskCase:
    """One representative evaluation task (step 4)."""

    task_id: str
    family: str
    difficulty: str
    safety_critical: bool
    resource_state: str
    complexity: float
    novelty: float
    latency_sensitivity: float
    latency_budget_ms: float
    memory_budget_mb: float
    source: str
    sequence_index: int
    is_failure_case: bool = False

    def __post_init__(self) -> None:
        if self.family not in TASK_FAMILIES:
            raise ValueError(f"unknown task family {self.family!r}")
        if self.difficulty not in DIFFICULTY_TIERS:
            raise ValueError(f"unknown difficulty tier {self.difficulty!r}")
        if self.resource_state not in RESOURCE_STATES:
            raise ValueError(f"unknown resource state {self.resource_state!r}")
        for name, value in (
            ("complexity", self.complexity),
            ("novelty", self.novelty),
            ("latency_sensitivity", self.latency_sensitivity),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"task feature {name} must be within [0, 1]")

    def features(self) -> dict[str, float]:
        """Read-only feature view handed to candidate strategies (step 14)."""
        return {
            "complexity": self.complexity,
            "novelty": self.novelty,
            "latency_sensitivity": self.latency_sensitivity,
            "resource_pressure": 1.0 if self.resource_state == "pressured" else 0.0,
            "safety_critical": 1.0 if self.safety_critical else 0.0,
        }

    def context_key(self) -> tuple[str, ...]:
        from ..domain.genome import table_context_key
        return table_context_key(self.family, self.difficulty, self.safety_critical, self.resource_state)

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id, "family": self.family,
            "difficulty": self.difficulty, "safety_critical": self.safety_critical,
            "resource_state": self.resource_state, "complexity": self.complexity,
            "novelty": self.novelty, "latency_sensitivity": self.latency_sensitivity,
            "latency_budget_ms": self.latency_budget_ms,
            "memory_budget_mb": self.memory_budget_mb,
            "source": self.source, "sequence_index": self.sequence_index,
            "is_failure_case": self.is_failure_case,
        }

    def content_payload(self) -> dict[str, Any]:
        return self.as_dict()


def _difficulty_complexity(difficulty: str, jitter: float) -> float:
    base = {"low": 0.28, "medium": 0.52, "high": 0.74}[difficulty]
    return min(base + jitter, 0.95)


def build_task_suite(
    *,
    tasks_per_family: int = 10,
    include_failure_cases: bool = True,
) -> tuple[TaskCase, ...]:
    """Deterministically build the representative task suite (step 4).

    Each family contributes ``tasks_per_family`` stratified tasks covering the three
    difficulty tiers, both resource states, varied latency sensitivity and novelty,
    plus rare safety-critical cases. When ``include_failure_cases`` is set, each family
    also contributes one deliberately hard failure case (high complexity, pressured
    resources, tight latency budget).
    """
    tasks: list[TaskCase] = []
    for family in TASK_FAMILIES:
        for index in range(tasks_per_family):
            difficulty = DIFFICULTY_TIERS[index % 3]
            complexity = _difficulty_complexity(difficulty, 0.03 * (index % 4))
            latency_sensitivity = round((index % 4) / 3.0, 3)
            novelty = round((index % 7) / 7.0, 3)
            resource_state = "pressured" if index % 5 == 4 else "normal"
            safety_critical = index % 8 == 3
            latency_budget = round(
                260.0 + 240.0 * (1.0 - latency_sensitivity)
                - (60.0 if resource_state == "pressured" else 0.0),
                1,
            )
            tasks.append(TaskCase(
                task_id=f"{family}_{index:02d}",
                family=family,
                difficulty=difficulty,
                safety_critical=safety_critical,
                resource_state=resource_state,
                complexity=round(complexity, 3),
                novelty=novelty,
                latency_sensitivity=latency_sensitivity,
                latency_budget_ms=max(latency_budget, 120.0),
                memory_budget_mb=72.0 if resource_state == "pressured" else 96.0,
                source=f"{family}_suite_v1",
                sequence_index=index,
            ))
        if include_failure_cases:
            tasks.append(TaskCase(
                task_id=f"{family}_edge",
                family=family,
                difficulty="high",
                safety_critical=family in ("reading", "discourse", "consolidation"),
                resource_state="pressured",
                complexity=0.92,
                novelty=0.85,
                latency_sensitivity=1.0,
                latency_budget_ms=150.0,
                memory_budget_mb=64.0,
                source=f"{family}_failure_case_v1",
                sequence_index=tasks_per_family,
                is_failure_case=True,
            ))
    return tuple(tasks)


def split_tasks(
    tasks: Sequence[TaskCase],
) -> tuple[tuple[TaskCase, ...], tuple[TaskCase, ...], tuple[TaskCase, ...]]:
    """Apply the declared stratified split policy (step 10).

    Within each family, non-edge tasks are assigned 60% train / 20% validation /
    20% test by sequence index modulo 10; failure cases alternate between validation
    and test so generalization is checked on hard cases too.
    """
    train: list[TaskCase] = []
    validation: list[TaskCase] = []
    test: list[TaskCase] = []
    for task in tasks:
        if task.is_failure_case:
            family_parity = sum(ord(c) for c in task.family) % 2
            (validation if family_parity == 0 else test).append(task)
            continue
        slot = task.sequence_index % 10
        if slot < 6:
            train.append(task)
        elif slot < 8:
            validation.append(task)
        else:
            test.append(task)
    return tuple(train), tuple(validation), tuple(test)


@dataclass(frozen=True, slots=True)
class ScoringConfig:
    """Scoring-code identity: fitness weights, scales, constraints, credits (step 9)."""

    fitness_config: Mapping[str, Any]
    constraints_config: Mapping[str, Any]

    def content_payload(self) -> dict[str, Any]:
        return {
            "fitness": dict(self.fitness_config),
            "constraints": dict(self.constraints_config),
            "outcome_credit": {k: OUTCOME_CREDIT[k] for k in sorted(OUTCOME_CREDIT)},
        }

    def digest(self) -> str:
        return content_digest(self.content_payload())


@dataclass(frozen=True, slots=True)
class BenchmarkManifest:
    """Immutable, versioned evaluation protocol (steps 4-10, 9).

    Hashes cover the task set, scoring configuration, constraints, seeds, runtime
    environment and evaluator version. A changed evaluator produces a new benchmark
    version (acceptance test 3).
    """

    benchmark_id: str
    tasks: tuple[TaskCase, ...]
    train: tuple[TaskCase, ...]
    validation: tuple[TaskCase, ...]
    test: tuple[TaskCase, ...]
    split_policy: str
    split_rationale: str
    seed: int
    scoring: ScoringConfig
    environment: Mapping[str, Any] = field(default_factory=dict)
    evaluator_version: str = EVALUATOR_VERSION
    version: str = ""
    task_set_digest: str = ""

    def __post_init__(self) -> None:
        if not self.version or not self.task_set_digest:
            task_digest = content_digest(
                {"tasks": [task.content_payload() for task in self.tasks]}
            )
            protocol = content_digest({
                "scoring": self.scoring.content_payload(),
                "seed": self.seed,
                "environment": dict(self.environment),
                "evaluator_version": self.evaluator_version,
                "split_policy": self.split_policy,
            })
            version = content_digest({
                "task_set": task_digest, "protocol": protocol,
            })
            if not self.task_set_digest:
                object.__setattr__(self, "task_set_digest", task_digest)
            if not self.version:
                object.__setattr__(self, "version", version[:16])

    # -- search-visible APIs (step 10; test set stays sealed) ----------------
    def training_tasks(self) -> tuple[TaskCase, ...]:
        return self.train

    def validation_tasks(self) -> tuple[TaskCase, ...]:
        return self.validation

    def test_task_digests(self) -> tuple[str, ...]:
        """Digests only — search code never sees held-out test content (step 10)."""
        return tuple(
            content_digest({"task": task.content_payload()}) for task in self.test
        )

    def open_sealed_test_set(self) -> tuple[TaskCase, ...]:
        """Open the sealed held-out set — post-search use only (step 29).

        Called exclusively by :class:`..evaluation.heldout_validation.HeldoutValidator`
        after search terminates. Nothing in the search layer holds a reference to this
        method's result.
        """
        return self.test

    def split_manifest(self) -> dict[str, Any]:
        """Declared split policy with per-split digests (acceptance test 3)."""
        return {
            "policy": self.split_policy,
            "rationale": self.split_rationale,
            "train_digest": content_digest(
                {"ids": [task.task_id for task in self.train]}),
            "validation_digest": content_digest(
                {"ids": [task.task_id for task in self.validation]}),
            "test_digest": content_digest(
                {"ids": [task.task_id for task in self.test]}),
            "test_task_count": len(self.test),
        }

    def protocol_digest(self) -> str:
        """Hash of scoring code, constraints, seeds and runtime environment (step 9)."""
        return content_digest({
            "scoring": self.scoring.content_payload(),
            "seed": self.seed,
            "environment": dict(self.environment),
            "evaluator_version": self.evaluator_version,
        })

    def as_dict(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "version": self.version,
            "task_set_digest": self.task_set_digest,
            "split_policy": self.split_policy,
            "split_rationale": self.split_rationale,
            "seed": self.seed,
            "scoring": self.scoring.content_payload(),
            "environment": dict(self.environment),
            "evaluator_version": self.evaluator_version,
            "split": self.split_manifest(),
            "tasks": [task.as_dict() for task in self.tasks],
        }


def runtime_environment() -> dict[str, Any]:
    """Runtime environment and dependency fingerprint (step 9)."""
    try:
        import numpy as np
        numpy_version = np.__version__
    except Exception:  # pragma: no cover - numpy is a hard dependency
        numpy_version = "unavailable"
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "numpy": numpy_version,
        "platform": sys.platform,
    }


def build_benchmark(
    *,
    benchmark_id: str = "ore_coordinator_parameter_v1",
    seed: int = 20261122,
    tasks_per_family: int = 10,
    scoring: ScoringConfig,
) -> BenchmarkManifest:
    """Build the versioned benchmark with sealed splits (steps 4-10)."""
    tasks = build_task_suite(
        tasks_per_family=tasks_per_family, include_failure_cases=True,
    )
    train, validation, test = split_tasks(tasks)
    return BenchmarkManifest(
        benchmark_id=benchmark_id,
        tasks=tasks,
        train=train,
        validation=validation,
        test=test,
        split_policy=SPLIT_POLICY,
        split_rationale=SPLIT_RATIONALE,
        seed=seed,
        scoring=scoring,
        environment=runtime_environment(),
    )


__all__ = [
    "BenchmarkManifest", "DIFFICULTY_TIERS", "RESOURCE_STATES", "SPLIT_POLICY",
    "SPLIT_RATIONALE", "ScoringConfig", "TASK_FAMILIES", "TaskCase",
    "build_benchmark", "build_task_suite", "runtime_environment", "split_tasks",
]
