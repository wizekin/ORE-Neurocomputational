"""Evaluation records, hard-constraint gates and scalar fitness (sections 2.2, steps 5-6, 15-17).

Multi-objective evaluation (section 2.2): ``F(s, x) = (Q, R, T, M, C, S)`` — verified
task quality, reliability, latency, memory/compute use, resource cost and
safety/compliance. Raw metrics and supporting evidence are always preserved (step 17);
never only a fitness scalar.

Hard correctness and safety conditions are *gates, not small penalties* (section 2.2):
``feasible`` requires ``quality >= q_min``, ``safety == pass`` (zero violations),
``latency <= T_max`` and ``cost <= C_max`` regardless of the aggregate score.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from .genome import canonical_json, content_digest

#: Outcome semantics (step 5): observable criteria only.
OUTCOME_KINDS: tuple[str, ...] = (
    "SUCCESS", "PARTIAL", "FAILURE", "ABSTAIN", "TIMEOUT", "INVALID",
)

#: Credit granted per outcome kind (verified partial credit, section 8).
OUTCOME_CREDIT: Mapping[str, float] = {
    "SUCCESS": 1.0, "PARTIAL": 0.5, "ABSTAIN": 0.1,
    "FAILURE": 0.0, "TIMEOUT": 0.0, "INVALID": 0.0,
}

#: Evaluator algorithm version — part of the benchmark protocol version (step 9).
EVALUATOR_VERSION = "ore_evolution_intelligence.evaluator-1.0.0"

#: 95% Student-t critical values for df 1..29 (confidence intervals, step 15).
_T95 = (12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
        2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
        2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045)


def t95(df: int) -> float:
    """Student-t 95% critical value (df capped at 29, normal approx above)."""
    if df < 1:
        return _T95[0]
    if df >= len(_T95):
        return 1.96
    return _T95[df - 1]


@dataclass(frozen=True, slots=True)
class TaskOutcomeRecord:
    """Raw per-trial, per-task observation (steps 13, 17: never discard evidence)."""

    task_id: str
    trial: int
    seed: int
    outcome: str
    quality: float
    latency_ms: float
    memory_mb: float
    cpu_ms: float
    cost: float
    safety_violations: int
    actions: tuple[str, ...] = ()
    steps: int = 0
    terminated: bool = False
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.outcome not in OUTCOME_KINDS:
            raise ValueError(f"unknown outcome kind {self.outcome!r}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id, "trial": self.trial, "seed": self.seed,
            "outcome": self.outcome, "quality": self.quality,
            "latency_ms": self.latency_ms, "memory_mb": self.memory_mb,
            "cpu_ms": self.cpu_ms, "cost": self.cost,
            "safety_violations": self.safety_violations,
            "actions": list(self.actions), "steps": self.steps,
            "terminated": self.terminated, "evidence": dict(self.evidence),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TaskOutcomeRecord":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * q
    lower = int(math.floor(position))
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


@dataclass(frozen=True, slots=True)
class AggregateMetrics:
    """Aggregated multi-objective metrics over trials (section 2.2, step 15)."""

    quality: float            # Q: mean verified quality (0..1)
    reliability: float         # R: fraction of non-failure, non-timeout, non-invalid trials
    latency_mean_ms: float    # T
    latency_p95_ms: float
    memory_peak_mb: float     # M
    cpu_total_ms: float
    cost: float               # C: total resource cost across all trials
    cost_mean: float           # C per trial (gate semantics: mean per task)
    safety_violations: int    # S: total violations across trials (gate must be zero)
    failure_rate: float
    timeout_rate: float
    invalid_rate: float
    abstain_rate: float
    quality_std: float
    quality_ci95: float
    worst_quality: float
    n_trials: int
    n_tasks: int
    mean_active_modules: float
    cooperation: float        # module cooperation: fraction of tasks with >=2 jointly active modules

    def as_dict(self) -> dict[str, Any]:
        return {
            "quality": self.quality, "reliability": self.reliability,
            "latency_mean_ms": self.latency_mean_ms, "latency_p95_ms": self.latency_p95_ms,
            "memory_peak_mb": self.memory_peak_mb, "cpu_total_ms": self.cpu_total_ms,
            "cost": self.cost, "cost_mean": self.cost_mean, "safety_violations": self.safety_violations,
            "failure_rate": self.failure_rate, "timeout_rate": self.timeout_rate,
            "invalid_rate": self.invalid_rate, "abstain_rate": self.abstain_rate,
            "quality_std": self.quality_std, "quality_ci95": self.quality_ci95,
            "worst_quality": self.worst_quality, "n_trials": self.n_trials,
            "n_tasks": self.n_tasks, "mean_active_modules": self.mean_active_modules,
            "cooperation": self.cooperation,
        }

    @classmethod
    def from_outcomes(cls, outcomes: list[TaskOutcomeRecord], n_tasks: int) -> "AggregateMetrics":
        if not outcomes:
            raise ValueError("cannot aggregate an empty trial set")
        n = len(outcomes)
        qualities = [o.quality for o in outcomes]
        latencies = sorted(o.latency_ms for o in outcomes)
        mean_quality = sum(qualities) / n
        variance = sum((q - mean_quality) ** 2 for q in qualities) / max(n - 1, 1)
        std = math.sqrt(variance)
        ci95 = t95(max(n - 1, 1)) * std / math.sqrt(n) if n > 1 else 0.0
        reliable = sum(1 for o in outcomes if o.outcome in ("SUCCESS", "PARTIAL"))
        failures = sum(1 for o in outcomes if o.outcome == "FAILURE")
        timeouts = sum(1 for o in outcomes if o.outcome == "TIMEOUT")
        invalid = sum(1 for o in outcomes if o.outcome == "INVALID")
        abstains = sum(1 for o in outcomes if o.outcome == "ABSTAIN")
        mean_active = sum(o.evidence.get("active_modules", 0) for o in outcomes) / n
        cooperation = sum(
            1 for o in outcomes if o.evidence.get("active_modules", 0) >= 2
        ) / n
        return cls(
            quality=mean_quality,
            reliability=reliable / n,
            latency_mean_ms=sum(latencies) / n,
            latency_p95_ms=_percentile(latencies, 0.95),
            memory_peak_mb=max(o.memory_mb for o in outcomes),
            cpu_total_ms=sum(o.cpu_ms for o in outcomes),
            cost=sum(o.cost for o in outcomes),
            cost_mean=sum(o.cost for o in outcomes) / n,
            safety_violations=sum(o.safety_violations for o in outcomes),
            failure_rate=failures / n,
            timeout_rate=timeouts / n,
            invalid_rate=invalid / n,
            abstain_rate=abstains / n,
            quality_std=std,
            quality_ci95=ci95,
            worst_quality=min(qualities),
            n_trials=n,
            n_tasks=n_tasks,
            mean_active_modules=mean_active,
            cooperation=cooperation,
        )


@dataclass(frozen=True, slots=True)
class GateDecision:
    """Result of applying hard constraints (step 6, step 21)."""

    feasible: bool
    reasons: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {"feasible": self.feasible, "reasons": list(self.reasons)}


@dataclass(frozen=True, slots=True)
class HardConstraints:
    """Hard safety/correctness/latency/cost limits (step 6; section 2.2 gates).

    Semantics: quality, latency p95, failure rate, safety violations are aggregate
    statistics over trials; ``c_max`` limits the *mean cost per task evaluation*
    (``cost_mean``), matching the per-task resource-cost semantics of section 2.2.
    Any violation makes a candidate ineligible regardless of aggregate score — speed
    can never compensate for unsafe behavior (step 21).
    """

    q_min: float = 0.55
    t_max_ms: float = 480.0
    c_max: float = 36.0
    memory_max_mb: float = 96.0
    max_failure_rate: float = 0.35

    def check(self, metrics: AggregateMetrics) -> GateDecision:
        reasons: list[str] = []
        if metrics.quality < self.q_min:
            reasons.append(f"quality {metrics.quality:.3f} < q_min {self.q_min:.3f}")
        if metrics.safety_violations > 0:
            reasons.append(f"safety violations present ({metrics.safety_violations})")
        if metrics.latency_p95_ms > self.t_max_ms:
            reasons.append(f"latency p95 {metrics.latency_p95_ms:.1f}ms > T_max {self.t_max_ms:.1f}ms")
        if metrics.cost_mean > self.c_max:
            reasons.append(f"cost per task {metrics.cost_mean:.1f} > C_max {self.c_max:.1f}")
        if metrics.memory_peak_mb > self.memory_max_mb:
            reasons.append(f"memory {metrics.memory_peak_mb:.1f}MB > limit {self.memory_max_mb:.1f}MB")
        if metrics.failure_rate > self.max_failure_rate:
            reasons.append(f"failure rate {metrics.failure_rate:.3f} > {self.max_failure_rate:.3f}")
        return GateDecision(feasible=not reasons, reasons=tuple(reasons))

    def as_dict(self) -> dict[str, Any]:
        return {
            "q_min": self.q_min, "t_max_ms": self.t_max_ms, "c_max": self.c_max,
            "memory_max_mb": self.memory_max_mb, "max_failure_rate": self.max_failure_rate,
            "c_max_semantics": "mean cost per task evaluation",
        }


@dataclass(frozen=True, slots=True)
class NormalizationScales:
    """Fixed benchmark-derived scales for the scalar fitness (section 2.2).

    Scales are documented policy choices fixed per benchmark version (step 22);
    they are *not* adapted during search.
    """

    latency_ms: float = 500.0
    memory_mb: float = 100.0
    cpu_ms: float = 400.0
    cost: float = 30.0
    safety_violation: float = 1.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "latency_ms": self.latency_ms, "memory_mb": self.memory_mb,
            "cpu_ms": self.cpu_ms, "cost": self.cost,
            "safety_violation": self.safety_violation,
        }


@dataclass(frozen=True, slots=True)
class ScalarFitness:
    """Documented weighted scalar fitness ``J(s)`` for controlled experiments (2.2).

    ``J = wQ*Q + wR*R - wT*T_hat - wM*M_hat - wC*C_hat - wS*S_hat`` with fixed
    normalization scales. Only used *after* hard-constraint gates: infeasible
    candidates never receive a fitness (step 21-22). Weights are explicit,
    documented policy choices, recorded in every run report.
    """

    w_quality: float = 1.0
    w_reliability: float = 0.35
    w_latency: float = 0.10
    w_memory: float = 0.02
    w_cost: float = 0.05
    w_safety: float = 1.0
    scales: NormalizationScales = field(default_factory=NormalizationScales)

    def score(self, metrics: AggregateMetrics) -> float:
        """Compute ``J(s)`` for one aggregate (gates applied by the caller)."""
        t_hat = metrics.latency_mean_ms / self.scales.latency_ms
        m_hat = metrics.memory_peak_mb / self.scales.memory_mb
        c_hat = metrics.cost_mean / self.scales.cost
        s_hat = metrics.safety_violations / self.scales.safety_violation
        return (
            self.w_quality * metrics.quality
            + self.w_reliability * metrics.reliability
            - self.w_latency * t_hat
            - self.w_memory * m_hat
            - self.w_cost * c_hat
            - self.w_safety * s_hat
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "w_quality": self.w_quality, "w_reliability": self.w_reliability,
            "w_latency": self.w_latency, "w_memory": self.w_memory,
            "w_cost": self.w_cost, "w_safety": self.w_safety,
            "scales": self.scales.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    """Complete evaluation of one candidate on one split (steps 15-17, 20)."""

    candidate_id: str
    candidate_version: int
    genome_hash: str
    benchmark_id: str
    benchmark_version: str
    split: str  # "train" | "validation" | "canary"
    seeds: tuple[int, ...]
    metrics: AggregateMetrics
    outcomes: tuple[TaskOutcomeRecord, ...]  # raw observations preserved (step 17)
    gate: GateDecision
    fitness: float | None  # scalar J only for feasible candidates (step 22)
    evaluator_version: str = EVALUATOR_VERSION
    behavior_measurement: Mapping[str, float | None] = field(default_factory=dict)
    invalid_reasons: tuple[str, ...] = ()
    evaluated_at_utc: str = ""

    def __post_init__(self) -> None:
        if self.split not in ("train", "validation", "canary", "heldout_test"):
            raise ValueError(
                f"evaluation split {self.split!r} is not a known split; search-visible "
                "splits are train/validation, canary and heldout_test are "
                "post-search splits"
            )
        if not self.evaluated_at_utc:
            object.__setattr__(self, "evaluated_at_utc", _utc_now_iso())

    def is_feasible(self) -> bool:
        return self.gate.feasible

    def measurement(self, name: str) -> float | None:
        """Raw component metric by name (never only a fitness scalar)."""
        if name in ("quality", "reliability"):
            return getattr(self.metrics, name)
        if name == "latency_mean_ms":
            return self.metrics.latency_mean_ms
        if name == "memory_peak_mb":
            return self.metrics.memory_peak_mb
        if name == "cost":
            return self.metrics.cost
        if name in self.behavior_measurement:
            return self.behavior_measurement.get(name)
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "candidate_version": self.candidate_version,
            "genome_hash": self.genome_hash,
            "benchmark_id": self.benchmark_id,
            "benchmark_version": self.benchmark_version,
            "split": self.split,
            "seeds": list(self.seeds),
            "metrics": self.metrics.as_dict(),
            "n_outcomes": len(self.outcomes),
            "gate": self.gate.as_dict(),
            "fitness": self.fitness,
            "evaluator_version": self.evaluator_version,
            "behavior_measurement": dict(self.behavior_measurement),
            "invalid_reasons": list(self.invalid_reasons),
            "evaluated_at_utc": self.evaluated_at_utc,
        }

    def content_digest(self) -> str:
        return content_digest({
            "candidate_id": self.candidate_id,
            "benchmark_version": self.benchmark_version,
            "split": self.split,
            "metrics": self.metrics.as_dict(),
            "gate": self.gate.as_dict(),
            "fitness": self.fitness,
        })


def _utc_now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


__all__ = [
    "AggregateMetrics", "EVALUATOR_VERSION", "EvaluationRecord", "GateDecision",
    "HardConstraints", "NormalizationScales", "OUTCOME_CREDIT", "OUTCOME_KINDS",
    "ScalarFitness", "TaskOutcomeRecord", "t95",
]
