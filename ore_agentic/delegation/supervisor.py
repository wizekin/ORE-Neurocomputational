"""The supervisor: eligibility, ranking, bounded delegation and joins (Steps 10-13, 16, 19).

The supervisor owns logical assignment only; lower-level execution scheduling
belongs to ModuleScheduler.  A worker failing any mandatory predicate is
excluded regardless of its score; candidates are ranked with the calibrated
objective S(a, t) and deterministic tie-breaks; failures are classified and
only transient failures are retried with bounded exponential backoff.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from ..config import AgenticConfig
from ..contracts import (
    FailureClass,
    ProvenanceRecord,
    ResourceVector,
    RetryPolicy,
    TaskOutcomeStatus,
    TaskResult,
    TaskSpec,
)
from ..functional import backoff_delay, quorum_satisfied, unit_scale, worker_score
from ..registry.agents import AgentModuleRegistry
from .leases import LeaseManager, StaleLeaseError, TaskLease
from .workers import (
    CancelledError,
    PermanentTaskError,
    PolicyTaskError,
    ResourceTaskError,
    TaskEnvelope,
    TransientTaskError,
    Worker,
)

__all__ = (
    "EligibilityReport",
    "NoEligibleWorkerError",
    "SelectionRecord",
    "Supervisor",
    "classify_exception",
)


class NoEligibleWorkerError(RuntimeError):
    """No worker satisfies the mandatory predicates (Step 10)."""


def classify_exception(exc: BaseException) -> FailureClass:
    """Classify an execution error into the failure taxonomy (Step 16)."""
    if isinstance(exc, PolicyTaskError):
        return FailureClass.POLICY
    if isinstance(exc, PermanentTaskError):
        return FailureClass.PERMANENT
    if isinstance(exc, ResourceTaskError):
        return FailureClass.RESOURCE
    if isinstance(exc, TransientTaskError):
        return FailureClass.TRANSIENT
    if isinstance(exc, CancelledError):
        return FailureClass.UNKNOWN
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return FailureClass.TRANSIENT
    return FailureClass.UNKNOWN


@dataclass(frozen=True, slots=True)
class EligibilityReport:
    """Why each candidate was eligible or excluded (Step 10)."""

    task_id: str
    eligible: tuple[str, ...]
    excluded: Mapping[str, str]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report."""
        return {"task_id": self.task_id, "eligible": list(self.eligible),
                "excluded": dict(self.excluded)}


@dataclass(frozen=True, slots=True)
class SelectionRecord:
    """One logged worker-selection decision with features (Step 11)."""

    task_id: str
    worker_id: str
    scores: Mapping[str, float]
    features: Mapping[str, float]
    tie_break: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize the record."""
        return {"task_id": self.task_id, "worker_id": self.worker_id,
                "scores": dict(self.scores), "features": dict(self.features),
                "tie_break": self.tie_break}


class Supervisor:
    """Bounded delegation of workflow tasks to eligible workers (Steps 10-16)."""

    def __init__(self, registry: AgentModuleRegistry, lease_manager: LeaseManager,
                 config: AgenticConfig | None = None,
                 rng: random.Random | None = None) -> None:
        """Compose the supervisor from the registry, leases and configuration."""
        self.registry = registry
        self.leases = lease_manager
        self.config = config or AgenticConfig()
        self._rng = rng or random.Random(self.config.seed)
        self.selections: list[SelectionRecord] = []
        self.eligibility_reports: list[EligibilityReport] = []
        self.retry_log: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ Step 10

    def eligible_workers(self, task: TaskSpec,
                         required_scopes: Sequence[str] = ()) -> EligibilityReport:
        """Filter candidates by mandatory predicates (Step 10).

        Capability, schema acceptance, permission scope, health, policy and
        resource profile are all mandatory: failing any one excludes the
        worker regardless of its score.
        """
        eligible: list[str] = []
        excluded: dict[str, str] = {}
        for record in self.registry.by_capability(task.capability):
            manifest = record.manifest
            if not record.eligible:
                excluded[manifest.worker_id] = "registration ineligible (revoked/expired)"
                continue
            if "object" not in manifest.accepted_schemas and (
                    "object" in task.input_schema or not task.input_schema):
                if "object" not in manifest.accepted_schemas:
                    excluded[manifest.worker_id] = "schema not accepted"
                    continue
            if required_scopes:
                missing = [scope for scope in required_scopes
                           if scope not in manifest.permission_scopes]
                if missing:
                    excluded[manifest.worker_id] = f"missing permission scopes: {missing}"
                    continue
            if not task.cost.fits_within(manifest.resource_profile):
                excluded[manifest.worker_id] = "task cost exceeds resource profile"
                continue
            eligible.append(manifest.worker_id)
        report = EligibilityReport(task.task_id, tuple(sorted(eligible)), excluded)
        self.eligibility_reports.append(report)
        return report

    # ------------------------------------------------------------------ Step 11

    def rank(self, task: TaskSpec, eligible: Sequence[str]) -> dict[str, float]:
        """Score eligible workers with S(a, t) (Step 11)."""
        scores: dict[str, float] = {}
        weights = self.config.worker_score_weights
        for worker_id in eligible:
            record = self.registry.get(worker_id)
            if record is None:
                continue
            manifest = record.manifest
            measured = record.measured_success_rate
            quality = measured if measured is not None else manifest.reliability
            fit = 1.0 if manifest.supports(task.capability) else 0.0
            features = {
                "fit": fit,
                "quality": quality,
                "latency": unit_scale(manifest.latency_seconds, 10.0),
                "cost": unit_scale(manifest.cost_per_task, 10.0),
                "uncertainty": 0.0 if measured is not None else 0.5,
                "load": manifest.load,
            }
            scores[worker_id] = worker_score(
                weights={name: getattr(weights, name) for name in
                         ("fit", "quality", "latency", "cost", "uncertainty", "load")},
                **features)
        return scores

    def select_worker(self, task: TaskSpec, required_scopes: Sequence[str] = ()
                      ) -> tuple[str, Worker | None, SelectionRecord]:
        """Select the best eligible worker with deterministic tie-breaks (Step 11)."""
        report = self.eligible_workers(task, required_scopes)
        if not report.eligible:
            raise NoEligibleWorkerError(
                f"no eligible worker for task {task.task_id}: {dict(report.excluded)}")
        scores = self.rank(task, report.eligible)
        best_score = max(scores.values())
        tied = sorted(worker_id for worker_id, value in scores.items()
                      if abs(value - best_score) < 1e-12)
        winner = tied[0]
        record = self.registry.get(winner)
        manifest = record.manifest if record else None
        selection = SelectionRecord(
            task_id=task.task_id, worker_id=winner, scores=scores,
            features={"reliability": manifest.reliability if manifest else 0.0,
                      "latency_seconds": manifest.latency_seconds if manifest else 0.0},
            tie_break="worker_id ascending" if len(tied) > 1 else "")
        self.selections.append(selection)
        return winner, None, selection

    # ------------------------------------------------------------------ Step 12

    def assign(self, workflow_id: str, task: TaskSpec, worker_id: str,
               inputs: Mapping[str, Any], effects: Any = None,
               cancellation: Any = None) -> tuple[TaskLease, TaskEnvelope]:
        """Issue a bounded lease and build the least-privilege envelope (Step 12)."""
        lease = self.leases.issue(workflow_id, task, worker_id,
                                  permitted_tools=task.permitted_tools)
        envelope = TaskEnvelope(task=task, lease=lease, inputs=dict(inputs),
                                effects=effects, cancellation=cancellation)
        return lease, envelope

    # ------------------------------------------------------------------ Step 16

    def run_with_retry(self, task: TaskSpec, envelope: TaskEnvelope,
                       worker: Worker, worker_version: str,
                       policy: RetryPolicy | None = None) -> TaskResult:
        """Execute one task with classified retries and bounded backoff (Step 16)."""
        retry = policy or task.retry
        deadline_budget = time.monotonic() + retry.total_time_budget_seconds
        attempts = 0
        last_error = ""
        last_class: FailureClass | None = None
        while attempts < retry.max_attempts:
            attempts += 1
            envelope.check_cancelled()
            try:
                result = worker.execute(envelope)
            except BaseException as exc:  # noqa: BLE001 - classified below
                failure_class = classify_exception(exc)
                last_error = f"{type(exc).__name__}: {exc}"
                last_class = failure_class
                if failure_class is not FailureClass.TRANSIENT:
                    break
                self.retry_log.append({
                    "task_id": task.task_id, "attempt": attempts,
                    "failure_class": failure_class.value, "error": last_error[:200]})
                if attempts >= retry.max_attempts or time.monotonic() >= deadline_budget:
                    break
                delay = backoff_delay(attempts - 1, retry.initial_delay_seconds,
                                      retry.max_delay_seconds, retry.jitter_seconds,
                                      rng=self._rng)
                time.sleep(min(delay, max(0.0, deadline_budget - time.monotonic())))
                continue
            if result.status is TaskOutcomeStatus.SUCCEEDED:
                self.registry.record_outcome(envelope.lease.worker_id, True)
                if attempts > result.attempts:
                    result = replace(result, attempts=attempts)
                return result
            failure_class = result.failure_class or FailureClass.UNKNOWN
            self.registry.record_outcome(envelope.lease.worker_id, False)
            if failure_class is not FailureClass.TRANSIENT:
                return result
            self.retry_log.append({
                "task_id": task.task_id, "attempt": attempts,
                "failure_class": failure_class.value, "error": result.error[:200]})
            if attempts >= retry.max_attempts or time.monotonic() >= deadline_budget:
                break
            delay = backoff_delay(attempts - 1, retry.initial_delay_seconds,
                                  retry.max_delay_seconds, retry.jitter_seconds,
                                  rng=self._rng)
            time.sleep(min(delay, max(0.0, deadline_budget - time.monotonic())))
        return TaskResult(
            task_id=task.task_id, worker_id=envelope.lease.worker_id,
            status=TaskOutcomeStatus.FAILED, error=last_error or "exhausted retries",
            failure_class=last_class, attempts=attempts,
            provenance=ProvenanceRecord(
                request_id=envelope.lease.lease_id, worker_id=envelope.lease.worker_id,
                worker_version=worker_version, verification="unverified"),
            resource_use=ResourceVector(seconds=0.0))

    # ------------------------------------------------------------------ Step 19

    def join(self, task: TaskSpec, results: Mapping[str, TaskResult]) -> TaskResult:
        """Aggregate dependency results under the declared join semantics (Step 19)."""
        task_ids = list(results)
        votes = [results[tid].status is TaskOutcomeStatus.SUCCEEDED for tid in task_ids]
        semantics = task.join.semantics.value
        satisfied = quorum_satisfied(votes, semantics, task.join.quorum,
                                     dict(task.join.weights) or
                                     {tid: 1.0 for tid in task_ids})
        payload: dict[str, Any] = {}
        dissent: list[str] = []
        for tid in task_ids:
            result = results[tid]
            payload[tid] = {
                "status": result.status.value,
                "payload": dict(result.payload),
                "warnings": list(result.warnings),
            }
            if not result.succeeded:
                dissent.append(tid)
        status = TaskOutcomeStatus.SUCCEEDED if satisfied else TaskOutcomeStatus.FAILED
        missing = [dep for dep in task.depends_on if dep not in results]
        return TaskResult(
            task_id=task.task_id, worker_id="supervisor.join",
            status=status,
            payload={"joined": payload, "satisfied": satisfied,
                     "missing": missing, "dissent": dissent},
            warnings=[f"missing results distinguished from negative: {missing}"]
            if missing else (),
            provenance=ProvenanceRecord(
                request_id=task.task_id,
                evidence_refs=tuple(
                    ref for result in results.values() for ref in result.evidence_refs),
                worker_id="supervisor", verification="aggregated"),
            attempts=1)

    # ------------------------------------------------------------------ Step 15

    def accept_result(self, lease: Any, result: TaskResult) -> TaskResult:
        """Accept a result only from the current lease epoch (Step 15)."""
        try:
            self.leases.accept(lease)
        except StaleLeaseError:
            raise
        return result
