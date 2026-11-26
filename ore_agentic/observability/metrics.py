"""Queue metrics, utilization, drift monitoring and semantic caching (Steps 14, 45-46).

Metrics track dispatched tasks, successes, failures, retries, deferred
admissions, queue wait, latency, cost and unauthorized calls.  The queue
monitor computes utilization rho = lambda / mu; sustained rho >= 1 signals
instability in the simple single-server approximation.  The drift monitor
compares recent and baseline success/latency at validated thresholds and
recommends review; quarantine only happens under explicit policy.  The
semantic cache keys on inputs, permissions, versions and workspace revision;
stale cache is never authoritative evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Mapping

from ..contracts import content_digest, utc_now
from ..functional import utilization

__all__ = (
    "CacheEntry",
    "DriftMonitor",
    "DriftReport",
    "MetricsCollector",
    "MetricsSnapshot",
    "QueueMonitor",
    "SemanticCache",
)


@dataclass(frozen=True, slots=True)
class MetricsSnapshot:
    """A point-in-time snapshot of engine metrics (Step 49)."""

    dispatched: int
    succeeded: int
    failed: int
    retried: int
    deferred: int
    unauthorized_calls: int
    cache_hits: int
    mean_latency_ms: float
    total_cost: float
    captured_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the snapshot."""
        return {
            "dispatched": self.dispatched, "succeeded": self.succeeded,
            "failed": self.failed, "retried": self.retried,
            "deferred": self.deferred, "unauthorized_calls": self.unauthorized_calls,
            "cache_hits": self.cache_hits, "mean_latency_ms": self.mean_latency_ms,
            "total_cost": self.total_cost, "captured_at": self.captured_at,
        }


class MetricsCollector:
    """Accumulate engine execution metrics (Steps 43, 49)."""

    def __init__(self) -> None:
        """Start with zeroed counters."""
        self.dispatched = 0
        self.succeeded = 0
        self.failed = 0
        self.retried = 0
        self.deferred = 0
        self.unauthorized_calls = 0
        self.cache_hits = 0
        self.total_cost = 0.0
        self.latencies_ms: list[float] = []
        self.failure_classes: dict[str, int] = {}

    def record_dispatch(self, cost: float = 0.0) -> None:
        """Record one task dispatch with its cost."""
        self.dispatched += 1
        self.total_cost += max(0.0, cost)

    def record_outcome(self, succeeded: bool, latency_ms: float,
                       failure_class: str = "", retries: int = 0) -> None:
        """Record one task outcome with latency and retries (Step 49)."""
        if succeeded:
            self.succeeded += 1
        else:
            self.failed += 1
            if failure_class:
                self.failure_classes[failure_class] = (
                    self.failure_classes.get(failure_class, 0) + 1)
        self.retried += retries
        self.latencies_ms.append(max(0.0, float(latency_ms)))

    def record_deferred(self, count: int = 1) -> None:
        """Record deferred admissions (backpressure, Step 14)."""
        self.deferred += count

    def record_unauthorized(self) -> None:
        """Record one unauthorized call attempt (Step 49)."""
        self.unauthorized_calls += 1

    def record_cache_hit(self) -> None:
        """Record one semantic-cache hit (Step 45)."""
        self.cache_hits += 1

    @property
    def mean_latency_ms(self) -> float:
        """Return the mean recorded latency."""
        return sum(self.latencies_ms) / len(self.latencies_ms) if self.latencies_ms else 0.0

    def snapshot(self) -> MetricsSnapshot:
        """Capture the current metrics (Step 49)."""
        return MetricsSnapshot(
            dispatched=self.dispatched, succeeded=self.succeeded, failed=self.failed,
            retried=self.retried, deferred=self.deferred,
            unauthorized_calls=self.unauthorized_calls, cache_hits=self.cache_hits,
            mean_latency_ms=round(self.mean_latency_ms, 3),
            total_cost=round(self.total_cost, 6))


class QueueMonitor:
    """Observe arrival and service rates for utilization (Step 14)."""

    def __init__(self, window_seconds: float = 60.0) -> None:
        """Configure the observation window."""
        self._window = window_seconds
        self._arrivals: list[float] = []
        self._services: list[float] = []

    def record_arrival(self) -> None:
        """Record one work arrival."""
        self._arrivals.append(time.monotonic())

    def record_service(self, service_seconds: float) -> None:
        """Record one completed service with its duration."""
        self._services.append(max(1e-9, float(service_seconds)))

    def rates(self) -> tuple[float, float]:
        """Return (arrival_rate, service_rate) over the window (Step 14)."""
        now = time.monotonic()
        recent = [stamp for stamp in self._arrivals if now - stamp <= self._window]
        arrival_rate = len(recent) / self._window if self._window > 0 else 0.0
        if not self._services:
            return arrival_rate, 0.0
        mean_service = sum(self._services) / len(self._services)
        service_rate = 1.0 / mean_service if mean_service > 0 else 0.0
        return arrival_rate, service_rate

    def utilization(self) -> float:
        """Return rho = lambda / mu with the single-server caveat (Step 14)."""
        arrival_rate, service_rate = self.rates()
        return utilization(arrival_rate, service_rate)

    def overloaded(self) -> bool:
        """Return True when sustained utilization is at or above 1 (Step 14)."""
        arrival_rate, service_rate = self.rates()
        return service_rate > 0.0 and arrival_rate / service_rate >= 1.0

    def describe(self) -> dict[str, Any]:
        """Return a serializable summary."""
        arrival_rate, service_rate = self.rates()
        return {"arrival_rate": round(arrival_rate, 6),
                "service_rate": round(service_rate, 6),
                "utilization": round(self.utilization(), 6),
                "overloaded": self.overloaded()}


@dataclass(frozen=True, slots=True)
class DriftReport:
    """The outcome of one drift evaluation (Step 46)."""

    drifted: bool
    metric: str
    recent: float
    baseline: float
    recommendation: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report."""
        return {"drifted": self.drifted, "metric": self.metric,
                "recent": self.recent, "baseline": self.baseline,
                "recommendation": self.recommendation}


class DriftMonitor:
    """Monitor success-rate and latency drift (Step 46).

    Quarantine or downgrade is only recommended under explicit policy;
    memory is never silently rewritten.
    """

    def __init__(self, window: int = 32, threshold: float = 0.25,
                 *, quarantine_allowed: bool = False) -> None:
        """Configure the drift window, threshold and quarantine policy."""
        if window < 2:
            raise ValueError("window must be at least 2")
        self._window = window
        self._threshold = threshold
        self._quarantine_allowed = quarantine_allowed
        self._successes: list[bool] = []
        self._latencies_ms: list[float] = []

    def observe(self, succeeded: bool, latency_ms: float) -> None:
        """Record one observed outcome (Step 46)."""
        self._successes.append(succeeded)
        self._latencies_ms.append(max(0.0, float(latency_ms)))
        if len(self._successes) > self._window * 2:
            self._successes = self._successes[-self._window * 2:]
            self._latencies_ms = self._latencies_ms[-self._window * 2:]

    def report(self) -> DriftReport:
        """Compare the recent window against the baseline (Step 46)."""
        if len(self._successes) < self._window:
            return DriftReport(False, "insufficient_data", 0.0, 0.0,
                               "continue monitoring")
        recent = self._successes[-self._window:]
        baseline = self._successes[:-self._window]
        recent_rate = sum(recent) / len(recent)
        baseline_rate = (sum(baseline) / len(baseline)) if baseline else recent_rate
        drop = baseline_rate - recent_rate
        if drop > self._threshold:
            recommendation = ("quarantine_under_explicit_policy"
                              if self._quarantine_allowed else "review")
            return DriftReport(True, "success_rate", round(recent_rate, 4),
                               round(baseline_rate, 4), recommendation)
        recent_latency = (sum(self._latencies_ms[-self._window:]) / self._window)
        baseline_latency = (sum(self._latencies_ms[:-self._window])
                            / max(1, len(self._latencies_ms) - self._window))
        if baseline_latency > 0 and (recent_latency / baseline_latency - 1.0) > self._threshold:
            return DriftReport(True, "latency", round(recent_latency, 3),
                               round(baseline_latency, 3), "review")
        return DriftReport(False, "success_rate", round(recent_rate, 4),
                           round(baseline_rate, 4), "stable")


@dataclass(frozen=True, slots=True)
class CacheEntry:
    """One semantic-cache entry with its authority bounds (Step 45)."""

    key: str
    value: Mapping[str, Any]
    model_version: str
    workspace_revision: int
    created_at: str
    expires_at: float
    stale: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize the entry."""
        return {"key": self.key, "value": dict(self.value),
                "model_version": self.model_version,
                "workspace_revision": self.workspace_revision,
                "created_at": self.created_at, "expires_at": self.expires_at,
                "stale": self.stale}


class SemanticCache:
    """Cache results keyed on inputs, versions and workspace revision (Step 45).

    Cache hits are explicitly marked non-authoritative: a cached entry is a
    performance artifact, never evidence.
    """

    def __init__(self, ttl_seconds: float = 300.0,
                 clock: Any = None) -> None:
        """Configure the TTL and injectable clock."""
        self._ttl = ttl_seconds
        self._clock = clock or time.monotonic
        self._entries: dict[str, CacheEntry] = {}

    @staticmethod
    def build_key(capability: str, inputs: Mapping[str, Any],
                  model_version: str, workspace_revision: int) -> str:
        """Build the cache key including versions and revision (Step 45)."""
        return content_digest({
            "capability": capability, "inputs": dict(inputs),
            "model_version": model_version,
            "workspace_revision": workspace_revision})

    def get(self, key: str) -> CacheEntry | None:
        """Return a fresh entry, marking expired entries stale (Step 45)."""
        entry = self._entries.get(key)
        if entry is None:
            return None
        if self._clock() > entry.expires_at:
            stale = CacheEntry(**{**entry.to_dict(), "stale": True,
                                  "value": entry.value})
            self._entries[key] = stale
            return stale
        return entry

    def put(self, key: str, value: Mapping[str, Any], *,
            model_version: str, workspace_revision: int) -> CacheEntry:
        """Store one entry with its authority bounds (Step 45)."""
        entry = CacheEntry(key=key, value=dict(value),
                           model_version=model_version,
                           workspace_revision=workspace_revision,
                           created_at=utc_now(),
                           expires_at=self._clock() + self._ttl)
        self._entries[key] = entry
        return entry

    def invalidate(self, key: str) -> None:
        """Explicitly invalidate one entry."""
        self._entries.pop(key, None)

    def size(self) -> int:
        """Return the number of stored entries."""
        return len(self._entries)
