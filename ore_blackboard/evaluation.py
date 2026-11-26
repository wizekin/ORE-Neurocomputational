"""Integration benchmark, paired comparison, and ablation harnesses (Steps 29-30).

Step 29 runs the same representative task set through (a) a simple sequential
execution baseline and (b) the blackboard coordination path, holding model
versions, inputs, and evaluation criteria constant. Step 30 disables one
integration mechanism at a time and measures its contribution. Neither harness
claims real-world quality from internally generated examples alone; the metrics
here measure coordination behaviour (activation selectivity, recovery, provenance
completeness), not language-model accuracy.
"""
from __future__ import annotations

import random
import statistics
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

from .config import BlackboardPolicy
from .contracts import BlackboardTaskRequest, BlackboardTaskResult
from .registry import operation_by_step

_SUCCESS_STATUSES = ("completed", "completed_with_uncertainty")


@dataclass(frozen=True, slots=True)
class TaskRunMetrics:
    """Per-task metrics for one execution path (Step 29)."""

    task_id: str
    path: str
    success: bool
    goal_score: float
    missing_requirement_count: int
    module_calls: int
    activations: int
    unnecessary_activation_count: int
    latency_ms: float
    conflict_count: int
    deferred_recovery_count: int
    recovery_count: int
    recoverable_failure_count: int
    provenance_completeness: float
    usg_contribution: int
    operation_coverage: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    """Aggregate benchmark comparison with paired statistics (Step 29)."""

    task_count: int
    blackboard: Mapping[str, float]
    baseline: Mapping[str, float]
    paired_differences: Mapping[str, tuple[float, float, float]]
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible benchmark report."""
        return {
            "task_count": self.task_count,
            "blackboard": dict(self.blackboard),
            "baseline": dict(self.baseline),
            "paired_differences": {
                key: {"mean": value[0], "ci_low": value[1], "ci_high": value[2]}
                for key, value in self.paired_differences.items()
            },
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class AblationReport:
    """Mechanism contribution table across ablation runs (Step 30)."""

    metric_names: tuple[str, ...]
    full_configuration: Mapping[str, float]
    ablations: Mapping[str, Mapping[str, float]]

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible ablation report."""
        return {
            "metric_names": list(self.metric_names),
            "full_configuration": dict(self.full_configuration),
            "ablations": {
                key: dict(value) for key, value in sorted(self.ablations.items())
            },
            "contributions": {
                key: {
                    metric: self.full_configuration.get(metric, 0.0) - value.get(metric, 0.0)
                    for metric in self.metric_names
                }
                for key, value in self.ablations.items()
            },
        }


def run_blackboard(coordinator: Any, request: BlackboardTaskRequest) -> TaskRunMetrics:
    """Run one task through the blackboard path and collect its metrics (Step 29)."""
    started = time.perf_counter()
    result: BlackboardTaskResult = coordinator.run(request)
    latency_ms = (time.perf_counter() - started) * 1000.0
    failure_summary = result.failure_summary
    recoverable = [item for item in failure_summary if not item.get("recovered")]
    return TaskRunMetrics(
        task_id=request.task_id,
        path="blackboard",
        success=result.status in _SUCCESS_STATUSES,
        goal_score=result.goal_score,
        missing_requirement_count=len(result.missing_requirements),
        module_calls=result.module_calls,
        activations=result.activations,
        unnecessary_activation_count=(
            result.suppressed_computations + _duplicate_publications(result)
        ),
        latency_ms=latency_ms,
        conflict_count=result.conflict_count,
        deferred_recovery_count=0,
        recovery_count=sum(
            1 for item in failure_summary if item.get("recovered")
        ),
        recoverable_failure_count=len(recoverable),
        provenance_completeness=result.trace_completeness,
        usg_contribution=len(result.imported_knowledge_references),
        operation_coverage=result.operation_coverage,
    )


def _duplicate_publications(result: BlackboardTaskResult) -> int:
    """Count duplicate publication decisions recorded in the run trace (Step 29)."""
    return sum(
        1 for record in result.trace
        if record["operation_id"] == "BBA-017"
        for publication in record["details"].get("publications", ())
        if publication.get("decision") == "duplicate"
    )


class SequentialBaselineRunner:
    """Simple sequential execution baseline over the same specialists (Step 29).

    Every module runs exactly once in dependency order with the same typed
    inputs and the same publication/validation path; the only difference is the
    absence of blackboard coordination: no relevance gating, no event routing,
    no computation suppression, no resource-aware recovery.
    """

    def __init__(self, coordinator: Any) -> None:
        """Reuse the coordinator's registry, catalog, and specialists."""
        self.coordinator = coordinator
        self.catalog = coordinator.catalog
        self.registry = coordinator.registry
        self.specialists = coordinator.specialists
        self.policy = coordinator.policy

    def run(self, request: BlackboardTaskRequest) -> TaskRunMetrics:
        """Run every specialist once in fixed order and collect baseline metrics."""
        from ore_workspace import (
            FindingPublisher,
            GoalEvaluator,
            SubscriptionRegistry,
            WorkspaceManager,
        )
        from ore_workspace.models import Provenance as WorkspaceProvenance

        from .contracts import COGNITIVE_CONTROL_PRODUCER, build_input_root, new_id
        from .execution import (
            SpecialistInvocation,
            build_input_root_artifact,
            build_memory_artifact,
            build_rules_artifact,
            make_run_context,
        )
        from .planning import build_permission_policy
        from .trace import ProvenanceTraceBuilder
        from .usg_integration import BlackboardUSGSnapshot, publish_usg_reference

        started = time.perf_counter()
        run_id = new_id("run")
        run_context = make_run_context(run_id, self.policy, request.execution_mode)
        manager = WorkspaceManager(enable_logging=False)
        input_root = build_input_root(request.input)
        authority = {
            specialist.describe().identity.module_id: frozenset(
                specialist.write_fields.values()
            )
            for specialist in self.specialists
        }
        workspace = manager.create_workspace(
            input_root.primary_text(),
            request.goal,
            permissions=build_permission_policy(authority),
            subscriptions=SubscriptionRegistry(defaults={}),
        )
        publisher = FindingPublisher(workspace, COGNITIVE_CONTROL_PRODUCER)
        input_root_result = workspace.publish(publisher.create_finding(
            "artifact_references",
            input_root.to_dict(),
            key=f"input_root:{input_root.input_digest[:16]}",
            finding_type="input_root_observation",
            provenance=WorkspaceProvenance(
                source="task_request",
                rules_applied=("sha256_input_digest",),
            ),
        ))
        artifacts: dict[str, Any] = {
            "input_root": build_input_root_artifact(input_root, run_id, request.task_id),
        }
        usg = None
        if request.usg_bundle is not None:
            usg = BlackboardUSGSnapshot(
                request.usg_bundle, expected_digest=request.usg_expected_digest,
            )
            publish_usg_reference(workspace, usg.reference)
            artifacts["usg_lexicon"] = usg.lexicon_artifact(run_id)
        if request.memory_store:
            artifacts["memory_store"] = build_memory_artifact(
                tuple(request.memory_store), run_id,
            )
        if request.verification_rules:
            artifacts["verification_rules"] = build_rules_artifact(
                tuple(request.verification_rules), run_id,
            )
        computation_registry = _ComputationRegistryProxy()
        module_calls = 0
        unnecessary_calls = 0
        for specialist in _dependency_order(self.specialists):
            snapshot = workspace.snapshot()
            invocation = SpecialistInvocation(
                specialist,
                run_context=run_context,
                catalog=self.catalog,
                task_artifacts=artifacts,
                bridge_factory=lambda snap, s=specialist: _make_baseline_bridge(
                    workspace, s, input_root,
                ),
                computation_registry=computation_registry,
                task_id=request.task_id,
                subscribed_fields=(),
            )
            findings = invocation(snapshot)
            module_calls += 1
            if not findings:
                unnecessary_calls += 1
            for finding in findings:
                workspace.publish(finding)
        goal = workspace.evaluate_goal(GoalEvaluator(
            weights=self.policy.goal_weights,
            threshold=self.policy.goal_threshold,
            critical_requirements=self.policy.critical_requirements,
        ))
        trace_builder = ProvenanceTraceBuilder(input_root)
        if usg is not None:
            trace_builder.add_usg_bundle(usg.reference.bundle_id, usg.reference.bundle_digest)
        if input_root_result.finding_id:
            trace_builder.link_input_root(input_root_result.finding_id)
        for finding_id, finding in sorted(workspace.findings.items()):
            trace_builder.add_finding(
                finding_id,
                producer_module=finding.producer_module,
                decision="accepted",
                target_field=finding.target.field,
                payload=dict(finding.payload),
                dependencies=finding.dependencies,
                usg_references=finding.payload.get("usg_references", ()),
            )
        completeness, _untraceable = trace_builder.completeness(list(workspace.findings))
        usg_contribution = (1 if usg else 0) + sum(
            len(finding.payload.get("usg_references", ()))
            for finding in workspace.findings.values()
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        manager.close()
        return TaskRunMetrics(
            task_id=request.task_id,
            path="sequential_baseline",
            success=goal.outcome.value in _SUCCESS_STATUSES,
            goal_score=goal.score,
            missing_requirement_count=len(goal.missing_requirements),
            module_calls=module_calls,
            activations=module_calls,
            unnecessary_activation_count=unnecessary_calls,
            latency_ms=latency_ms,
            conflict_count=len(workspace.conflicts.list(workspace.workspace_id)),
            deferred_recovery_count=0,
            recovery_count=0,
            recoverable_failure_count=0,
            provenance_completeness=completeness,
            usg_contribution=usg_contribution,
            operation_coverage=(),
        )


def _make_baseline_bridge(workspace: Any, specialist: Any, input_root: Any):
    """Create the artifact-to-finding bridge for the sequential baseline (Step 17)."""
    from .bridge import ArtifactFindingBridge

    return ArtifactFindingBridge(
        workspace,
        specialist.describe().identity.module_id,
        allowed_fields=frozenset(specialist.write_fields.values()),
        input_length=len(input_root.primary_text()),
    )


class _ComputationRegistryProxy:
    """No-op computation registry: the baseline never suppresses recomputation."""

    @staticmethod
    def invocation_id(task_id, module_id, module_version, input_hash, revision):
        """Return a placeholder invocation identity for tracing only."""
        return f"baseline:{module_id}:{revision}"

    @staticmethod
    def relevant_fingerprint(module_id, snapshot, subscribed_fields):
        """Return a placeholder fingerprint; the baseline does not deduplicate."""
        return f"baseline:{module_id}:{snapshot.revision}"

    @staticmethod
    def check_suppressed(module_id, module_version, relevant_fingerprint):
        """Never suppress: the baseline recomputes on every invocation."""
        return None

    @staticmethod
    def record(invocation):
        """Record the invocation without indexing reusable outputs."""

    @staticmethod
    def count_suppressed():
        """Return zero: the baseline performs no computation suppression."""
        return 0

    @staticmethod
    def records():
        """Return no invocation records."""
        return ()

    @staticmethod
    def mark_suppressed():
        """No-op: the baseline does not track suppressed computations."""


def _dependency_order(specialists: Iterable[Any]) -> tuple[Any, ...]:
    """Order specialists by their declared manifest dependencies (Kahn's algorithm)."""
    items = list(specialists)
    identifiers = {
        item.describe().identity.module_id: item for item in items
    }
    pending = {module_id: set() for module_id in identifiers}
    dependents: dict[str, set[str]] = {module_id: set() for module_id in identifiers}
    for module_id, specialist in identifiers.items():
        for dependency in specialist.describe().dependencies:
            if dependency.module_id in pending:
                pending[module_id].add(dependency.module_id)
                dependents[dependency.module_id].add(module_id)
    ordered: list[Any] = []
    ready = sorted(module_id for module_id, blockers in pending.items() if not blockers)
    while ready:
        module_id = ready.pop(0)
        ordered.append(identifiers[module_id])
        for dependent in sorted(dependents[module_id]):
            pending[dependent].discard(module_id)
            if not pending[dependent]:
                ready.append(dependent)
        ready.sort()
    if len(ordered) != len(items):  # pragma: no cover - manifests are validated first
        raise ValueError("specialist dependency graph contains a cycle")
    return tuple(ordered)


def _aggregate(metrics: Sequence[TaskRunMetrics]) -> dict[str, float]:
    """Aggregate one path's per-task metrics into comparable summary statistics."""
    values = list(metrics)

    def mean(items: Sequence[float]) -> float:
        return sum(items) / len(items) if items else 0.0

    latencies = [item.latency_ms for item in values]
    sorted_latencies = sorted(latencies)
    p95 = (
        sorted_latencies[min(len(sorted_latencies) - 1, max(0, int(0.95 * len(sorted_latencies))))]
        if sorted_latencies else 0.0
    )
    successful = sum(1 for item in values if item.success)
    all_activations = sum(item.activations for item in values)
    unnecessary = sum(item.unnecessary_activation_count for item in values)
    return {
        "success_rate": successful / max(1, len(values)),
        "mean_goal_score": mean([item.goal_score for item in values]),
        "mean_module_calls": mean([item.module_calls for item in values]),
        "total_activations": float(all_activations),
        "unnecessary_activation_rate": unnecessary / max(1, all_activations),
        "mean_latency_ms": mean(latencies),
        "median_latency_ms": float(statistics.median(latencies)) if latencies else 0.0,
        "p95_latency_ms": float(p95),
        "mean_conflicts_preserved": mean([float(item.conflict_count) for item in values]),
        "recovery_rate": (
            sum(item.recovery_count for item in values)
            / max(1, sum(item.recoverable_failure_count for item in values) + sum(
                item.recovery_count for item in values
            ))
        ),
        "mean_provenance_completeness": mean(
            [item.provenance_completeness for item in values]
        ),
        "mean_usg_contribution": mean([float(item.usg_contribution) for item in values]),
    }


def paired_bootstrap(
    values_a: Sequence[float],
    values_b: Sequence[float],
    *,
    metric_name: str,
    iterations: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return mean difference and a bootstrap CI for paired differences (Step 29)."""
    if len(values_a) != len(values_b):
        raise ValueError("paired bootstrap requires equal-length samples")
    if not values_a:
        return 0.0, 0.0, 0.0
    differences = [a - b for a, b in zip(values_a, values_b)]
    mean = sum(differences) / len(differences)
    generator = random.Random(seed)
    samples: list[float] = []
    for _ in range(iterations):
        draws = [
            differences[generator.randrange(len(differences))]
            for _ in range(len(differences))
        ]
        samples.append(sum(draws) / len(draws))
    samples.sort()
    alpha = (1.0 - confidence) / 2.0
    low = samples[int(alpha * len(samples))]
    high = samples[min(len(samples) - 1, int((1.0 - alpha) * len(samples)))]
    return mean, low, high


def run_benchmark(
    coordinator_factory: Callable[[], Any],
    requests: Sequence[BlackboardTaskRequest],
    *,
    iterations: int = 1000,
    seed: int = 0,
) -> tuple[BenchmarkReport, tuple[TaskRunMetrics, ...], tuple[TaskRunMetrics, ...]]:
    """Compare blackboard coordination against the sequential baseline (Step 29)."""
    blackboard_metrics: list[TaskRunMetrics] = []
    baseline_metrics: list[TaskRunMetrics] = []
    for request in requests:
        coordinator = coordinator_factory()
        blackboard_metrics.append(run_blackboard(coordinator, request))
        baseline = SequentialBaselineRunner(coordinator)
        baseline_metrics.append(baseline.run(request))
    blackboard_summary = _aggregate(blackboard_metrics)
    baseline_summary = _aggregate(baseline_metrics)
    paired: dict[str, tuple[float, float, float]] = {}
    paired["goal_score"] = paired_bootstrap(
        [item.goal_score for item in blackboard_metrics],
        [item.goal_score for item in baseline_metrics],
        metric_name="goal_score", iterations=iterations, seed=seed,
    )
    paired["module_calls"] = paired_bootstrap(
        [float(item.module_calls) for item in blackboard_metrics],
        [float(item.module_calls) for item in baseline_metrics],
        metric_name="module_calls", iterations=iterations, seed=seed,
    )
    paired["latency_ms"] = paired_bootstrap(
        [item.latency_ms for item in blackboard_metrics],
        [item.latency_ms for item in baseline_metrics],
        metric_name="latency_ms", iterations=iterations, seed=seed,
    )
    paired["provenance_completeness"] = paired_bootstrap(
        [item.provenance_completeness for item in blackboard_metrics],
        [item.provenance_completeness for item in baseline_metrics],
        metric_name="provenance_completeness", iterations=iterations, seed=seed,
    )
    report = BenchmarkReport(
        task_count=len(requests),
        blackboard=blackboard_summary,
        baseline=baseline_summary,
        paired_differences=paired,
        notes=(
            "Success here measures goal-completion of the coordination harness, "
            "not independent real-world task quality.",
            "Paired bootstrap confidence intervals estimate uncertainty; one "
            "demonstration run never establishes an improvement.",
        ),
    )
    return report, tuple(blackboard_metrics), tuple(baseline_metrics)


ABLATION_MECHANISMS: tuple[str, ...] = (
    "selective_subscriptions",
    "resource_aware_choice",
    "usg_retrieval",
    "hypothesis_conflict_handling",
    "dependency_deferral",
    "provenance_checks",
    "bounded_parallelism",
)


def run_ablation(
    coordinator_factory: Callable[[BlackboardPolicy], Any],
    request: BlackboardTaskRequest,
    *,
    base_policy: BlackboardPolicy | None = None,
) -> AblationReport:
    """Disable one integration mechanism at a time and measure contributions (Step 30)."""
    from dataclasses import replace

    policy = base_policy or BlackboardPolicy()
    full_metrics = run_blackboard(coordinator_factory(policy), request)
    full = _single_run_metrics(full_metrics)
    ablations: dict[str, Mapping[str, float]] = {}
    for mechanism in ABLATION_MECHANISMS:
        ablated_switches = replace(
            policy.ablations, **{mechanism: False},
        )
        ablated_policy = replace(policy, ablations=ablated_switches)
        metrics = run_blackboard(coordinator_factory(ablated_policy), request)
        ablations[mechanism] = _single_run_metrics(metrics)
    metric_names = tuple(sorted(full))
    return AblationReport(
        metric_names=metric_names,
        full_configuration=full,
        ablations=ablations,
    )


def _single_run_metrics(metrics: TaskRunMetrics) -> dict[str, float]:
    """Flatten one task run into comparable numeric metrics (Step 30)."""
    return {
        "goal_score": metrics.goal_score,
        "module_calls": float(metrics.module_calls),
        "activations": float(metrics.activations),
        "unnecessary_activation_count": float(metrics.unnecessary_activation_count),
        "latency_ms": metrics.latency_ms,
        "conflict_count": float(metrics.conflict_count),
        "provenance_completeness": metrics.provenance_completeness,
        "usg_contribution": float(metrics.usg_contribution),
    }


def evaluation_operation_records() -> tuple[Mapping[str, Any], ...]:
    """Return the operation trace records for the evaluation harness (Steps 29-30)."""
    records = []
    for step in (29, 30):
        spec = operation_by_step(step)
        records.append({
            "operation_id": spec.operation_id,
            "step": spec.step,
            "title": spec.title,
            "cycle": None,
            "details": {"purpose": spec.purpose},
        })
    return tuple(records)


__all__ = [
    "ABLATION_MECHANISMS",
    "AblationReport",
    "BenchmarkReport",
    "SequentialBaselineRunner",
    "TaskRunMetrics",
    "evaluation_operation_records",
    "paired_bootstrap",
    "run_ablation",
    "run_benchmark",
    "run_blackboard",
]
