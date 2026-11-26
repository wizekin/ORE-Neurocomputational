"""Operational metrics for the Point 19 runtime engine."""
from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from typing import Any


class RuntimeMetrics:
    """Collect the specification's operational metrics for engine reports."""

    def __init__(self) -> None:
        """Initialize all counters and observation windows."""
        self._lock = threading.RLock()
        self._completed_tasks = 0
        self._failed_tasks = 0
        self._retried_tasks = 0
        self._cancelled_tasks = 0
        self._queue_wait_samples: list[float] = []
        self._lock_wait_samples: list[float] = []
        self._conflict_retries = 0
        self._event_lag_samples: list[float] = []
        self._duplicate_deliveries = 0
        self._outbox_backlog_age_samples: list[float] = []
        self._stale_counts: list[int] = []
        self._recompute_volumes: list[int] = []
        self._change_volumes: list[int] = []
        self._admission_rejections = 0
        self._replay_deterministic: bool | None = None
        self._recovery_durations: list[float] = []

    # -------------------------------------------------------------- counters

    def record_completion(self, *, failed: bool = False, retried: bool = False,
                          cancelled: bool = False) -> None:
        """Record one task completion, failure, retry, or cancellation."""
        with self._lock:
            if failed:
                self._failed_tasks += 1
            elif cancelled:
                self._cancelled_tasks += 1
            else:
                self._completed_tasks += 1
            if retried:
                self._retried_tasks += 1

    def record_queue_wait(self, seconds: float) -> None:
        """Sample one queue wait duration."""
        with self._lock:
            self._queue_wait_samples.append(max(0.0, seconds))

    def record_lock_wait(self, seconds: float) -> None:
        """Sample one lock wait duration."""
        with self._lock:
            self._lock_wait_samples.append(max(0.0, seconds))

    def record_conflict_retry(self) -> None:
        """Count one optimistic-write conflict that triggered a retry."""
        with self._lock:
            self._conflict_retries += 1

    def record_event_lag(self, seconds: float) -> None:
        """Sample commit-to-consumer processing lag for one event."""
        with self._lock:
            self._event_lag_samples.append(max(0.0, seconds))

    def record_duplicate_delivery(self, count: int = 1) -> None:
        """Count duplicate deliveries observed by consumers."""
        with self._lock:
            self._duplicate_deliveries += count

    def record_outbox_backlog_age(self, seconds: float) -> None:
        """Sample the outbox backlog age."""
        with self._lock:
            self._outbox_backlog_age_samples.append(max(0.0, seconds))

    def record_stale_artifacts(self, count: int) -> None:
        """Sample the current stale artifact count."""
        with self._lock:
            self._stale_counts.append(count)

    def record_recompute_volume(self, recomputed: int, changed: int) -> None:
        """Record one incremental recompute volume against its change volume."""
        with self._lock:
            self._recompute_volumes.append(max(0, recomputed))
            self._change_volumes.append(max(0, changed))

    def record_admission_rejection(self) -> None:
        """Count one admission-control deferral or rejection."""
        with self._lock:
            self._admission_rejections += 1

    def record_recovery_duration(self, seconds: float) -> None:
        """Sample one crash-recovery duration."""
        with self._lock:
            self._recovery_durations.append(max(0.0, seconds))

    def record_replay_determinism(self, deterministic: bool) -> None:
        """Record whether a captured schedule reproduced invariant outcomes."""
        with self._lock:
            self._replay_deterministic = bool(deterministic)

    # ---------------------------------------------------------------- report

    @staticmethod
    def _percentiles(samples: Sequence[float]) -> dict[str, float]:
        """Return p50/p95/p99 using the nearest-rank method."""
        if not samples:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
        ordered = sorted(samples)

        def at(fraction: float) -> float:
            """Compute one nearest-rank percentile."""
            rank = max(1, min(len(ordered), int(round(fraction * len(ordered)))))
            return ordered[rank - 1]

        return {"p50": at(0.50), "p95": at(0.95), "p99": at(0.99)}

    def snapshot(self) -> dict[str, Any]:
        """Return the full operational metrics report."""
        with self._lock:
            total = self._completed_tasks + self._failed_tasks + self._cancelled_tasks
            recompute_total = sum(self._recompute_volumes)
            change_total = sum(self._change_volumes)
            return {
                "task_throughput": total,
                "completed_tasks": self._completed_tasks,
                "failed_tasks": self._failed_tasks,
                "retried_tasks": self._retried_tasks,
                "cancelled_tasks": self._cancelled_tasks,
                "queue_wait": self._percentiles(self._queue_wait_samples),
                "lock_wait": self._percentiles(self._lock_wait_samples),
                "conflict_retry_rate": (
                    round(self._conflict_retries / total, 4) if total else 0.0
                ),
                "event_lag": self._percentiles(self._event_lag_samples),
                "duplicate_deliveries": self._duplicate_deliveries,
                "outbox_backlog_age": self._percentiles(self._outbox_backlog_age_samples),
                "stale_artifacts": (
                    self._stale_counts[-1] if self._stale_counts else 0
                ),
                "recompute_to_change_ratio": (
                    round(recompute_total / change_total, 4)
                    if change_total else 0.0
                ),
                "admission_rejections": self._admission_rejections,
                "replay_determinism": self._replay_deterministic,
                "recovery_duration": self._percentiles(self._recovery_durations),
            }

    def merge(self, other: Mapping[str, Any]) -> dict[str, Any]:
        """Return the snapshot merged with component-specific statistics."""
        return {**self.snapshot(), **dict(other)}
