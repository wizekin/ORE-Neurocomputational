"""Bounded ready queues, priority scoring, admission control, and work stealing."""
from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..contracts import EngineConfig, TaskState
from .task_graph import TaskGraph, TaskNode


class AdmissionDecision(str, Enum):
    """Report what admission control decided for one candidate task."""

    ADMITTED = "admitted"
    DEFERRED = "deferred"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class AdmissionOutcome:
    """Carry the admission decision with the reason and current queue depth."""

    decision: AdmissionDecision
    reason: str
    queue_depth: int
    admitted_count: int


@dataclass(frozen=True, slots=True)
class SchedulerChoice:
    """Record one scheduling decision with its score components for the trace."""

    task_id: str
    worker_id: str
    score: float
    effective_priority: float
    slack: float
    aging: float
    estimated_finish: float


@dataclass(slots=True)
class WorkerQueueState:
    """Track one capability worker's local queue and live load estimate."""

    worker_id: str
    capabilities: frozenset[str]
    local_deque: deque[str] = field(default_factory=deque)
    active_task_ids: set[str] = field(default_factory=set)
    completed_count: int = 0
    busy_seconds: float = 0.0

    @property
    def estimated_load(self) -> float:
        """Sum the estimated cost of queued and active tasks for this worker."""
        return float(len(self.local_deque) + len(self.active_task_ids))

    def can_claim(self, capability: str, max_active: int) -> bool:
        """Check capability match and an available active-task budget slot."""
        return capability in self.capabilities and len(self.active_task_ids) < max_active


class AdmissionController:
    """Enforce high/low watermark backpressure so bursts cannot exhaust the machine."""

    def __init__(self, *, high_watermark: int, low_watermark: int) -> None:
        """Validate watermark ordering that provides hysteresis against oscillation."""
        if high_watermark < 1:
            raise ValueError("high_watermark must be at least 1.")
        if not 0 <= low_watermark < high_watermark:
            raise ValueError("low_watermark must lie in [0, high_watermark).")
        self._high = high_watermark
        self._low = low_watermark
        self._paused = False
        self._rejections = 0
        self._deferrals = 0
        self._admissions = 0

    def decide(self, queue_depth: int) -> AdmissionOutcome:
        """Decide admission with hysteresis: pause above high, resume below low."""
        if self._paused:
            if queue_depth < self._low:
                self._paused = False
            else:
                self._rejections += 1
                return AdmissionOutcome(
                    decision=AdmissionDecision.DEFERRED,
                    reason=f"admission paused until depth falls below {self._low}",
                    queue_depth=queue_depth, admitted_count=self._admissions,
                )
        if queue_depth >= self._high:
            self._paused = True
            self._rejections += 1
            return AdmissionOutcome(
                decision=AdmissionDecision.DEFERRED,
                reason=f"queue depth {queue_depth} reached high watermark {self._high}",
                queue_depth=queue_depth, admitted_count=self._admissions,
            )
        self._admissions += 1
        return AdmissionOutcome(
            decision=AdmissionDecision.ADMITTED,
            reason="queue depth within configured watermarks",
            queue_depth=queue_depth, admitted_count=self._admissions,
        )

    @property
    def paused(self) -> bool:
        """Return whether admission is currently paused under backpressure."""
        return self._paused

    def stats(self) -> dict[str, int | bool]:
        """Return admission counters for the metrics report."""
        return {
            "admissions": self._admissions,
            "deferrals": self._deferrals,
            "rejections": self._rejections,
            "paused": self._paused,
        }


class Scheduler:
    """Select and assign ready tasks with bounded queues, aging, and quotas.

    The scheduling score follows the specification: alpha*priority + beta/(slack +
    epsilon) - gamma*estimated_cost + delta*aging. The score is a policy aid
    only; capability match, dependency completion, and resource ceilings are
    enforced separately by the worker pool and admission controller.
    """

    def __init__(self, config: EngineConfig) -> None:
        """Bind scheduler policy to the validated engine configuration."""
        self._config = config
        self._lock = threading.RLock()
        self._queue_wait_samples: list[float] = []
        self._dispatch_log: list[dict[str, Any]] = []
        self._module_counts: dict[str, int] = {}

    # ---------------------------------------------------------------- scoring

    @staticmethod
    def effective_priority(node: TaskNode, wait_seconds: float,
                           aging_rate: float, max_priority: float) -> float:
        """Age a task's priority while it waits so starvation stays bounded."""
        return min(max_priority, node.spec.priority + aging_rate * max(0.0, wait_seconds))

    def score(self, node: TaskNode, wait_seconds: float,
              elapsed_deadline_seconds: float | None) -> tuple[float, dict[str, float]]:
        """Compute the scheduling score and its components for one ready task."""
        weights = self._config.scheduler_weights
        priority = self.effective_priority(
            node, wait_seconds, self._config.aging_rate, self._config.max_priority,
        )
        if elapsed_deadline_seconds is None:
            slack = weights.epsilon + node.spec.estimated_cost
        else:
            remaining = node.spec.deadline_seconds - elapsed_deadline_seconds
            slack = max(remaining, 0.0) + weights.epsilon
        aging = wait_seconds
        value = (
            weights.alpha_priority * priority
            + weights.beta_deadline / (slack + weights.epsilon)
            - weights.gamma_cost * node.spec.estimated_cost
            + weights.delta_aging * aging
        )
        return value, {"priority": priority, "slack": slack, "aging": aging}

    def rank(self, nodes: Sequence[TaskNode], now: float) -> tuple[tuple[float, TaskNode], ...]:
        """Rank ready tasks by scheduling score in descending order."""
        ranked: list[tuple[float, TaskNode]] = []
        for node in nodes:
            wait = 0.0 if node.enqueued_at is None else max(0.0, now - node.enqueued_at)
            deadline_elapsed = (
                None if node.spec.deadline_seconds is None
                else max(0.0, now - (node.enqueued_at or now))
            )
            value, _ = self.score(node, wait, deadline_elapsed)
            ranked.append((value, node))
        ranked.sort(key=lambda pair: (-pair[0], pair[1].task_id))
        return tuple(ranked)

    # --------------------------------------------------------------- quotas

    def _module_of(self, node: TaskNode) -> str:
        """Derive the module identity used for per-module queue quotas."""
        return str(node.spec.metadata.get("module_id", node.spec.capability))

    def within_quota(self, node: TaskNode, queued_by_module: Mapping[str, int]) -> bool:
        """Check the per-module queue share so one module cannot monopolize queues."""
        return queued_by_module.get(self._module_of(node), 0) < self._config.module_queue_quota

    # --------------------------------------------------------------- dispatch

    def plan_dispatch(
        self, graph: TaskGraph, workers: Sequence[WorkerQueueState], now: float,
    ) -> tuple[SchedulerChoice, ...]:
        """Plan task-worker assignments honoring capability and active-task ceilings.

        The plan is pure: worker queues are mutated only by the pool when the
        engine actually dispatches, so a plan is reusable and side-effect free.
        """
        ready = graph.ready_tasks()
        if not ready:
            return ()
        assignments: list[SchedulerChoice] = []
        queued_by_module: dict[str, int] = {}
        for value, node in self.rank(ready, now):
            eligible = [
                state for state in workers
                if state.can_claim(node.spec.capability, self._config.max_active_tasks)
            ]
            if not eligible:
                continue
            if not self.within_quota(node, queued_by_module):
                continue
            wait = 0.0 if node.enqueued_at is None else max(0.0, now - node.enqueued_at)
            deadline_elapsed = (
                None if node.spec.deadline_seconds is None
                else max(0.0, now - (node.enqueued_at or now))
            )
            choice_worker = min(
                eligible,
                key=lambda state: (state.estimated_load, state.worker_id),
            )
            _score, components = self.score(node, wait, deadline_elapsed)
            assignments.append(SchedulerChoice(
                task_id=node.task_id,
                worker_id=choice_worker.worker_id,
                score=value,
                effective_priority=components["priority"],
                slack=components["slack"],
                aging=components["aging"],
                estimated_finish=choice_worker.estimated_load + node.spec.estimated_cost,
            ))
            queued_by_module[self._module_of(node)] = (
                queued_by_module.get(self._module_of(node), 0) + 1
            )
        return tuple(assignments)

    # ---------------------------------------------------------- work stealing

    def plan_work_stealing(
        self, idle_worker: WorkerQueueState, workers: Sequence[WorkerQueueState],
        graph: TaskGraph,
    ) -> str | None:
        """Steal one eligible queued task from the most loaded other worker.

        Stealing is conservative: the stolen task must satisfy capability match
        and must not be blocked by an unfinished dependency, so dependency,
        priority, and resource constraints are preserved.
        """
        victims = [state for state in workers if state.worker_id != idle_worker.worker_id]
        if not victims:
            return None
        victim = max(victims, key=lambda state: (state.estimated_load, state.worker_id))
        while victim.local_deque:
            candidate_id = victim.local_deque.pop()
            node = graph.node(candidate_id)
            if node.state in {TaskState.READY, TaskState.CREATED, TaskState.WAITING} \
                    and idle_worker.can_claim(node.spec.capability, self._config.max_active_tasks):
                victim.active_task_ids.discard(candidate_id)
                idle_worker.local_deque.append(candidate_id)
                idle_worker.active_task_ids.add(candidate_id)
                return candidate_id
            victim.local_deque.appendleft(candidate_id)
            if candidate_id == victim.local_deque[-1]:
                break
        return None

    # --------------------------------------------------------------- metrics

    def record_queue_wait(self, wait_seconds: float) -> None:
        """Record one queue-wait sample for percentile reporting."""
        with self._lock:
            self._queue_wait_samples.append(max(0.0, wait_seconds))

    def record_dispatch(self, choice: SchedulerChoice) -> None:
        """Record one dispatch decision for the audit trace."""
        with self._lock:
            self._dispatch_log.append({
                "task_id": choice.task_id,
                "worker_id": choice.worker_id,
                "score": round(choice.score, 6),
                "effective_priority": round(choice.effective_priority, 6),
            })

    @staticmethod
    def _percentile(samples: Sequence[float], fraction: float) -> float:
        """Return one nearest-rank percentile over a sample list."""
        if not samples:
            return 0.0
        ordered = sorted(samples)
        rank = max(1, min(len(ordered), int(round(fraction * len(ordered)))))
        return ordered[rank - 1]

    def stats(self) -> dict[str, Any]:
        """Return scheduler statistics including queue-wait percentiles."""
        with self._lock:
            samples = list(self._queue_wait_samples)
            return {
                "dispatch_count": len(self._dispatch_log),
                "queue_wait_p50": self._percentile(samples, 0.50),
                "queue_wait_p95": self._percentile(samples, 0.95),
                "queue_wait_p99": self._percentile(samples, 0.99),
                "dispatch_log": list(self._dispatch_log),
            }
