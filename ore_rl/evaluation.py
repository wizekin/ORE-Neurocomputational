"""Benchmark suites, policy evaluation, and promotion thresholds.

Point 10, Step 43 and section 8: learned policies are compared against the
deterministic coordinator baseline (and each other) under identical task
splits, budgets, evaluators, and reward versions. Reports cover the metrics
of section 8.4 — task success rate, independently evaluated quality,
evidence validity, latency percentiles, resource cost, cost per successful
task, invalid-action rejection rate, failure/recovery, stop-decision
precision/recall, calibration, and improvement relative to the baseline with
bootstrap confidence intervals, broken out by task category.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .contracts import OREState, PolicyDecision
from .environment import OREEnvironment
from .policies.base import PolicyService


@dataclass(frozen=True, slots=True)
class PromotionThresholds:
    """Predeclared criteria a candidate must meet before promotion (8.3)."""

    min_episodes: int = 20
    min_success_rate: float = 0.0
    min_baseline_improvement: float = 0.0
    max_constraint_violations: int = 0
    min_ci_lower_bound: float = -1.0
    min_evidence_validity: float = 0.0


@dataclass(slots=True)
class EpisodeRecord:
    """One evaluated episode with per-step detail for audit."""

    episode_id: str
    task_id: str
    task_family: str
    success: bool
    quality: float
    evidence: float
    latency_ms: float
    cpu_cost: float
    memory_mb: float
    steps: int
    module_calls: int
    terminal_kind: str
    stop_label: str
    stop_correct: bool
    total_reward: float
    rejected_proposals: int = 0
    constraint_violations: int = 0
    wasted_actions: int = 0


@dataclass(slots=True)
class MetricSnapshot:
    """Aggregate metrics over one policy's evaluation (section 8.4)."""

    policy_id: str
    episodes: int
    success_rate: float
    mean_quality: float
    mean_evidence: float
    evidence_validity: float
    latency_mean_ms: float
    latency_p95_ms: float
    mean_cpu_cost: float
    mean_memory_mb: float
    cost_per_success: float | None
    invalid_action_rejection_rate: float
    failure_rate: float
    mean_steps: float
    stop_precision: float | None
    stop_recall: float | None
    by_family: Mapping[str, Mapping[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "episodes": self.episodes,
            "success_rate": self.success_rate,
            "mean_quality": self.mean_quality,
            "mean_evidence": self.mean_evidence,
            "evidence_validity": self.evidence_validity,
            "latency_mean_ms": self.latency_mean_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "mean_cpu_cost": self.mean_cpu_cost,
            "mean_memory_mb": self.mean_memory_mb,
            "cost_per_success": self.cost_per_success,
            "invalid_action_rejection_rate": self.invalid_action_rejection_rate,
            "failure_rate": self.failure_rate,
            "mean_steps": self.mean_steps,
            "stop_precision": self.stop_precision,
            "stop_recall": self.stop_recall,
            "by_family": {
                family: dict(metrics) for family, metrics in self.by_family.items()
            },
        }


@dataclass(slots=True)
class EvaluationReport:
    """A complete evaluation result with baseline comparison and CI."""

    policy_id: str
    baseline_id: str
    metrics: MetricSnapshot
    baseline_metrics: MetricSnapshot | None
    improvement_vs_baseline: float | None = None
    improvement_ci95: tuple[float, float] | None = None
    constraint_violations: int = 0
    meets_promotion_thresholds: bool | None = None
    thresholds: PromotionThresholds | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "baseline_id": self.baseline_id,
            "metrics": self.metrics.to_dict(),
            "baseline_metrics": (
                self.baseline_metrics.to_dict() if self.baseline_metrics else None
            ),
            "improvement_vs_baseline": self.improvement_vs_baseline,
            "improvement_ci95": list(self.improvement_ci95) if self.improvement_ci95 else None,
            "constraint_violations": self.constraint_violations,
            "meets_promotion_thresholds": self.meets_promotion_thresholds,
            "thresholds": asdict(self.thresholds) if self.thresholds else None,
            "details": dict(self.details),
        }


def bootstrap_improvement_ci(
    treatment: Sequence[float], baseline: Sequence[float], *, iterations: int = 1000, seed: int = 0
) -> tuple[float, float]:
    """Bootstrap 95% CI for the mean difference (independent samples)."""
    rng = np.random.default_rng(seed)
    treatment_array = np.asarray(treatment, dtype=np.float64)
    baseline_array = np.asarray(baseline, dtype=np.float64)
    if treatment_array.size == 0 or baseline_array.size == 0:
        return (float("nan"), float("nan"))
    differences = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        t_sample = treatment_array[rng.integers(0, treatment_array.size, treatment_array.size)]
        b_sample = baseline_array[rng.integers(0, baseline_array.size, baseline_array.size)]
        differences[index] = t_sample.mean() - b_sample.mean()
    return (
        float(np.percentile(differences, 2.5)),
        float(np.percentile(differences, 97.5)),
    )


def _run_episode(
    env: OREEnvironment,
    policy: PolicyService,
    *,
    depth: int = 3,
    act_hook: Callable[[PolicyDecision, OREState], tuple[str, dict[str, Any]]] | None = None,
) -> EpisodeRecord:
    """Run one episode with the policy and collect audit metrics."""
    state, info = env.reset()
    rejected = 0
    total_reward = 0.0
    step_budget = {"latency_ms": 50.0}
    terminal_kind = ""
    stop_label = ""
    stop_correct = False
    constraint_violations = 0
    while True:
        eligible = info["feasible_action_ids"] if "feasible_action_ids" in info else env.feasible_actions()
        decision = policy.propose_actions(state, eligible, step_budget)
        if act_hook is not None:
            action, params = act_hook(decision, state)
        else:
            action = decision.chosen_action_id
            params = {}
        state, reward, terminated, truncated, info = env.step((action, params))
        total_reward += reward
        if info.get("rejected"):
            rejected += 1
            constraint_violations += len(info.get("violations", ()))
            continue
        if terminated or truncated:
            terminal_kind = info.get("terminal_kind", "TRUNCATED_BUDGET")
            stop_label = info.get("stop_label", "")
            stop_correct = stop_label == "correct"
            break
    summary = env.episode_history()[-1]
    record = EpisodeRecord(
        episode_id=summary["episode_id"],
        task_id=summary["task_id"],
        task_family=summary["task_family"],
        success=bool(summary["success"]),
        quality=float(summary["quality"]),
        evidence=float(summary["evidence"]),
        latency_ms=float(summary["latency_ms"]),
        cpu_cost=float(summary["cpu_cost"]),
        memory_mb=float(summary["memory_mb"]),
        steps=int(summary["steps"]),
        module_calls=int(summary["module_calls"]),
        terminal_kind=terminal_kind,
        stop_label=stop_label,
        stop_correct=stop_correct,
        total_reward=total_reward,
        rejected_proposals=rejected,
        constraint_violations=constraint_violations,
        wasted_actions=int(summary["wasted_actions"]),
    )
    return record


def _metrics_from_records(
    records: Sequence[EpisodeRecord], policy_id: str
) -> MetricSnapshot:
    successes = [record for record in records if record.success]
    failures = [record for record in records if record.terminal_kind in ("EXCEPTION", "FAILED")]
    stop_events = [record for record in records if record.stop_label]
    true_positives = sum(1 for record in stop_events if record.stop_correct)
    false_positives = len(stop_events) - true_positives
    relevant = sum(1 for record in records if record.quality >= 0.75)
    precision = true_positives / max(1, true_positives + false_positives) if stop_events else None
    recall = true_positives / max(1, relevant) if relevant else None
    by_family: dict[str, dict[str, float]] = {}
    for family in sorted({record.task_family for record in records}):
        family_records = [record for record in records if record.task_family == family]
        by_family[family] = {
            "episodes": float(len(family_records)),
            "success_rate": sum(1 for r in family_records if r.success) / len(family_records),
            "mean_quality": sum(r.quality for r in family_records) / len(family_records),
            "mean_latency_ms": sum(r.latency_ms for r in family_records) / len(family_records),
        }
    latencies = sorted(record.latency_ms for record in records)
    total_steps = sum(record.steps for record in records) or 1
    return MetricSnapshot(
        policy_id=policy_id,
        episodes=len(records),
        success_rate=len(successes) / len(records) if records else 0.0,
        mean_quality=(
            sum(record.quality for record in records) / len(records) if records else 0.0
        ),
        mean_evidence=(
            sum(record.evidence for record in records) / len(records) if records else 0.0
        ),
        evidence_validity=(
            sum(record.evidence for record in successes) / len(successes)
            if successes
            else 0.0
        ),
        latency_mean_ms=(
            sum(record.latency_ms for record in records) / len(records) if records else 0.0
        ),
        latency_p95_ms=(
            latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))] if latencies else 0.0
        ),
        mean_cpu_cost=(
            sum(record.cpu_cost for record in records) / len(records) if records else 0.0
        ),
        mean_memory_mb=(
            sum(record.memory_mb for record in records) / len(records) if records else 0.0
        ),
        cost_per_success=(
            sum(record.latency_ms for record in successes) / len(successes)
            if successes
            else None
        ),
        invalid_action_rejection_rate=sum(record.rejected_proposals for record in records)
        / total_steps,
        failure_rate=len(failures) / len(records) if records else 0.0,
        mean_steps=(sum(record.steps for record in records) / len(records) if records else 0.0),
        stop_precision=precision,
        stop_recall=recall,
        by_family=by_family,
    )


@dataclass(frozen=True, slots=True)
class BenchmarkSuite:
    """Fixed benchmark definition: identical splits, budgets, evaluators (Step 43)."""

    name: str
    n_episodes: int
    seed: int
    depth: int = 3
    reward_version: str = "reward-weights-1.0.0"
    evaluator_version: str = "ore-rl-env-1.0.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_episodes": self.n_episodes,
            "seed": self.seed,
            "depth": self.depth,
            "reward_version": self.reward_version,
            "evaluator_version": self.evaluator_version,
        }


def evaluate_policy(
    policy: PolicyService,
    env: OREEnvironment,
    *,
    suite: BenchmarkSuite | None = None,
    baseline: PolicyService | None = None,
    thresholds: PromotionThresholds | None = None,
    act_hook: Callable[[PolicyDecision, OREState], tuple[str, dict[str, Any]]] | None = None,
) -> EvaluationReport:
    """Run the benchmark for one policy and compare against the baseline."""
    suite = suite or BenchmarkSuite(name="default", n_episodes=20, seed=7)
    records = [
        _run_episode(env, policy, depth=suite.depth, act_hook=act_hook)
        for _ in range(suite.n_episodes)
    ]
    metrics = _metrics_from_records(records, policy.metadata.policy_id)
    baseline_metrics = None
    improvement = None
    ci = None
    if baseline is not None:
        baseline_records = [
            _run_episode(env, baseline, depth=suite.depth)
            for _ in range(suite.n_episodes)
        ]
        baseline_metrics = _metrics_from_records(
            baseline_records, baseline.metadata.policy_id
        )
        treatment_scores = [1.0 if r.success else 0.0 for r in records]
        baseline_scores = [1.0 if r.success else 0.0 for r in baseline_records]
        improvement = metrics.success_rate - baseline_metrics.success_rate
        ci = bootstrap_improvement_ci(treatment_scores, baseline_scores, seed=suite.seed)
    constraint_violations = sum(record.constraint_violations for record in records)
    meets = None
    if thresholds is not None:
        meets = (
            metrics.episodes >= thresholds.min_episodes
            and metrics.success_rate >= thresholds.min_success_rate
            and (improvement is None or improvement >= thresholds.min_baseline_improvement)
            and constraint_violations <= thresholds.max_constraint_violations
            and metrics.evidence_validity >= thresholds.min_evidence_validity
            and (ci is None or ci[0] >= thresholds.min_ci_lower_bound)
        )
    return EvaluationReport(
        policy_id=policy.metadata.policy_id,
        baseline_id=baseline.metadata.policy_id if baseline else "none",
        metrics=metrics,
        baseline_metrics=baseline_metrics,
        improvement_vs_baseline=improvement,
        improvement_ci95=ci,
        constraint_violations=constraint_violations,
        meets_promotion_thresholds=meets,
        thresholds=thresholds,
        details={"suite": suite.to_dict()},
    )


@dataclass(slots=True)
class ComparisonReport:
    """Side-by-side comparison of multiple policies (Step 43)."""

    suite: BenchmarkSuite
    policy_metrics: dict[str, MetricSnapshot] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite": self.suite.to_dict(),
            "policies": {
                policy_id: metrics.to_dict()
                for policy_id, metrics in self.policy_metrics.items()
            },
        }

    def ranking(self) -> list[tuple[str, float]]:
        return sorted(
            (
                (policy_id, metrics.success_rate)
                for policy_id, metrics in self.policy_metrics.items()
            ),
            key=lambda item: -item[1],
        )


def compare_policies(
    policies: Mapping[str, PolicyService],
    env: OREEnvironment,
    *,
    suite: BenchmarkSuite | None = None,
    act_hook: Callable[[PolicyDecision, OREState], tuple[str, dict[str, Any]]] | None = None,
) -> ComparisonReport:
    """Compare deterministic coordinator, bandit, DQN, PPO... under identical conditions."""
    suite = suite or BenchmarkSuite(name="comparison", n_episodes=15, seed=7)
    report = ComparisonReport(suite=suite)
    for policy_id, policy in policies.items():
        records = [
            _run_episode(env, policy, depth=suite.depth, act_hook=act_hook)
            for _ in range(suite.n_episodes)
        ]
        report.policy_metrics[policy_id] = _metrics_from_records(records, policy_id)
    return report
