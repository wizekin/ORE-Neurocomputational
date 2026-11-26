"""Benchmark and ablation harnesses for the reasoning layer (Section 10; Stages 6-7).

The harness runs synthetic labeled corpora through the coordinator and reports
software-level metrics: precision/recall where labels exist, proof validity,
latency, rule fires, memory estimate and verification overhead. Following the
separation demanded by Section 10, these numbers measure the reasoning
mechanism on synthetic data — they are not claims of scientific reasoning
quality on real corpora, which requires independent labeled evaluation.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .config import NSRLPolicy
from .coordinator import NeuroSymbolicReasoningCoordinator
from .contracts import (
    EpistemicStatus,
    RuleClass,
    SemanticFact,
)
from .request_models import ReasoningRequest
from .rule_validation import RuleProfile

_SUCCESS_STATUSES = ("completed", "completed_with_uncertainty")


@dataclass(frozen=True, slots=True)
class ReasoningRunMetrics:
    """Per-task metrics for one reasoning configuration."""

    task_id: str
    status: str
    success: bool
    goal_satisfied: bool
    conclusion_count: int
    published_count: int
    gated_out_count: int
    conflict_count: int
    ambiguity_group_count: int
    proof_node_count: int
    verified_node_count: int
    proof_validity_rate: float
    rule_fires: int
    latency_ms: float
    memory_estimate_mb: float
    model_calls: int
    operation_coverage: tuple[str, ...]
    trace_completeness: float


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    """Aggregate benchmark over a labeled synthetic corpus."""

    task_count: int
    summary: Mapping[str, float]
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return the detached JSON-compatible report."""
        return {
            "task_count": self.task_count,
            "summary": dict(self.summary),
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class AblationReport:
    """Mechanism contribution table across ablation configurations."""

    metric_names: tuple[str, ...]
    full_configuration: Mapping[str, float]
    ablations: Mapping[str, Mapping[str, float]]

    def to_dict(self) -> dict[str, Any]:
        """Return the detached JSON-compatible ablation report."""
        return {
            "metric_names": list(self.metric_names),
            "full_configuration": dict(self.full_configuration),
            "ablations": {
                key: dict(value) for key, value in sorted(self.ablations.items())
            },
            "contributions": {
                key: {
                    metric: self.full_configuration.get(metric, 0.0)
                    - value.get(metric, 0.0)
                    for metric in self.metric_names
                }
                for key, value in self.ablations.items()
            },
        }


def run_reasoning(
    coordinator: NeuroSymbolicReasoningCoordinator,
    request: ReasoningRequest,
) -> ReasoningRunMetrics:
    """Run one task through the coordinator and collect its metrics."""
    started = time.perf_counter()
    result = coordinator.run(request)
    latency_ms = (time.perf_counter() - started) * 1000.0
    proof_summary = result.proof_summary
    node_count = int(proof_summary.get("node_count", 0))
    verified_count = int(proof_summary.get("verified_nodes", 0))
    strict_classes = sum(
        1 for value in proof_summary.get("classification", {}).values()
        if value == "strict_verified"
    )
    validity = verified_count / max(1, strict_classes) if strict_classes else 1.0
    return ReasoningRunMetrics(
        task_id=request.task_id,
        status=result.status,
        success=result.status in _SUCCESS_STATUSES,
        goal_satisfied=result.goal_satisfied,
        conclusion_count=len(result.conclusions),
        published_count=len(result.published_finding_ids),
        gated_out_count=sum(1 for item in result.conclusions
                            if item.publication_state.value == "gated_out"),
        conflict_count=len(result.conflicts),
        ambiguity_group_count=len(result.ambiguity_groups),
        proof_node_count=node_count,
        verified_node_count=verified_count,
        proof_validity_rate=min(1.0, validity),
        rule_fires=int(result.metrics.get("rule_fires", 0)),
        latency_ms=latency_ms,
        memory_estimate_mb=float(result.metrics.get("memory_estimate_mb", 0.0)),
        model_calls=int(result.metrics.get("model_calls", 0)),
        operation_coverage=result.operation_coverage,
        trace_completeness=result.trace_completeness,
    )


def evaluate_corpus(
    request_factory: Callable[[int], ReasoningRequest],
    labels: Sequence[Mapping[str, bool]],
    *,
    policy: NSRLPolicy | None = None,
    task_count: int | None = None,
) -> BenchmarkReport:
    """Evaluate precision/recall against labeled conclusions (Section 10).

    ``labels[i]`` maps one expected conclusion signature per task to whether
    the corpus considers it true. Precision/recall are computed from derived
    conclusion signatures against these labels; when no labels are supplied
    the metrics report None instead of invented values.
    """
    count = task_count if task_count is not None else len(labels)
    predicted_true: list[set[str]] = []
    labeled_true: list[set[str]] = []
    latencies: list[float] = []
    validity_rates: list[float] = []
    successes = 0
    for index in range(count):
        request = request_factory(index)
        coordinator = NeuroSymbolicReasoningCoordinator(policy=policy)
        started = time.perf_counter()
        result = coordinator.run(request)
        latencies.append((time.perf_counter() - started) * 1000.0)
        successes += 1 if result.status in _SUCCESS_STATUSES else 0
        validity_rates.append(float(result.metrics.get("proof_validity_rate", 0.0)))
        predicted = {
            finding.conclusion.signature()
            for finding in result.conclusions
            if finding.publication_state.value in {"published", "review_requested"}
        }
        predicted_true.append(predicted)
        label_map = labels[index] if index < len(labels) else {}
        labeled_true.append({key for key, value in label_map.items() if value})

    true_positives = sum(
        len(predicted & labeled)
        for predicted, labeled in zip(predicted_true, labeled_true)
    )
    predicted_total = sum(len(item) for item in predicted_true)
    labeled_total = sum(len(item) for item in labeled_true)
    precision = true_positives / predicted_total if predicted_total else None
    recall = true_positives / labeled_total if labeled_total else None

    def mean(values: Sequence[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    summary: dict[str, float] = {
        "success_rate": successes / max(1, count),
        "mean_latency_ms": mean(latencies),
        "mean_proof_validity_rate": mean(validity_rates),
        "true_positives": float(true_positives),
    }
    if precision is not None:
        summary["precision"] = precision
    if recall is not None:
        summary["recall"] = recall
    return BenchmarkReport(
        task_count=count,
        summary=summary,
        notes=(
            "Metrics measure the reasoning mechanism on synthetic labeled data.",
            "Scientific quality claims require independent labeled evaluation "
            "on real corpora (Section 10).",
        ),
    )


ABLATION_MECHANISMS: tuple[str, ...] = (
    "learned_assistance",
    "probabilistic_combination",
    "independent_verification",
    "constraint_filtering",
    "ambiguity_preservation",
    "defeasible_resolution",
    "usg_retrieval",
)


def run_ablation(
    request: ReasoningRequest,
    *,
    base_policy: NSRLPolicy | None = None,
) -> AblationReport:
    """Disable one mechanism at a time and measure its contribution."""
    from dataclasses import replace as _replace

    from .config import NSRLAblationSwitches

    policy = base_policy or NSRLPolicy()
    full_metrics = run_reasoning(
        NeuroSymbolicReasoningCoordinator(policy=policy), request,
    )
    full = _single_run_metrics(full_metrics)
    ablations: dict[str, Mapping[str, float]] = {}
    for mechanism in ABLATION_MECHANISMS:
        switches = _replace(policy.ablations, **{mechanism: False})
        ablated_policy = _replace(policy, ablations=switches)
        metrics = run_reasoning(
            NeuroSymbolicReasoningCoordinator(policy=ablated_policy), request,
        )
        ablations[mechanism] = _single_run_metrics(metrics)
    return AblationReport(
        metric_names=tuple(sorted(full)),
        full_configuration=full,
        ablations=ablations,
    )


def _single_run_metrics(metrics: ReasoningRunMetrics) -> dict[str, float]:
    """Flatten one run into comparable numeric metrics."""
    return {
        "conclusion_count": float(metrics.conclusion_count),
        "published_count": float(metrics.published_count),
        "conflict_count": float(metrics.conflict_count),
        "ambiguity_group_count": float(metrics.ambiguity_group_count),
        "proof_node_count": float(metrics.proof_node_count),
        "proof_validity_rate": metrics.proof_validity_rate,
        "rule_fires": float(metrics.rule_fires),
        "latency_ms": metrics.latency_ms,
        "trace_completeness": metrics.trace_completeness,
    }


def safe_rule_profile() -> RuleProfile:
    """Return the documented safe rule profile used by the harnesses."""
    return RuleProfile()


__all__ = [
    "ABLATION_MECHANISMS",
    "AblationReport",
    "BenchmarkReport",
    "ReasoningRunMetrics",
    "evaluate_corpus",
    "run_ablation",
    "run_reasoning",
    "safe_rule_profile",
]
