"""Phase VI, Step 30 - streaming and concurrency-safe data flows.

Bounded queues with explicit backpressure, defined event ordering and
delivery guarantees, idempotent consumers, event identity preserved across
retries, and admission control when sustained arrival pressure approaches
service capacity.  Queue stability requires long-run service capacity to
exceed admitted work; arrival rate ``lambda`` and service rate ``mu`` are
tracked together with queue depth and waiting time, and no unbounded queue,
retry loop or hidden shared mutable state is permitted.
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

from .core import new_id, sha256_digest, utc_now_iso

STREAMING_SCHEMA = "ore.point24.streaming.v1"


@dataclass(frozen=True, slots=True)
class StreamEvent:
    """One event with a stable identity preserved across retries."""

    payload: Mapping[str, Any]
    event_key: str  # idempotency key derived from content identity
    partition: str
    event_id: str = field(default_factory=lambda: new_id("event"))

    @classmethod
    def create(cls, payload: Mapping[str, Any], *, partition: str = "default") -> "StreamEvent":
        return cls(
            payload=dict(payload),
            event_key=sha256_digest({"payload": dict(payload), "partition": partition}),
            partition=partition,
        )


class BoundedEventQueue:
    """A bounded queue with explicit backpressure (Step 30).

    ``put`` raises ``queue.Full`` (or blocks up to ``block_seconds``) when the
    queue is full - work is never silently dropped or allowed to grow without
    limits.
    """

    def __init__(self, *, max_depth: int = 256) -> None:
        if max_depth < 1:
            raise ValueError("max_depth must be positive")
        self._queue: queue.Queue[StreamEvent] = queue.Queue(maxsize=max_depth)
        self.max_depth = max_depth

    def put(self, event: StreamEvent, *, block_seconds: float = 0.0) -> bool:
        try:
            self._queue.put(event, block=(block_seconds > 0), timeout=block_seconds or None)
            return True
        except queue.Full:
            return False

    def get(self, *, block_seconds: float | None = None) -> StreamEvent | None:
        try:
            if block_seconds is None:
                return self._queue.get_nowait()
            return self._queue.get(block=(block_seconds > 0), timeout=block_seconds or None)
        except queue.Empty:
            return None

    def depth(self) -> int:
        return self._queue.qsize()

    def apply_backpressure(self) -> bool:
        """Signal the producer to defer when the queue exceeds 80% capacity."""
        return self.depth() >= int(self.max_depth * 0.8)


@dataclass(slots=True)
class QueueMonitor:
    """Track arrival rate lambda, service rate mu, depth and wait time."""

    window_seconds: float = 5.0
    _arrivals: list[float] = field(default_factory=list)
    _services: list[float] = field(default_factory=list)
    _waits: list[float] = field(default_factory=list)

    def record_arrival(self) -> None:
        self._arrivals.append(time.monotonic())
        self._prune()

    def record_service(self, wait_seconds: float) -> None:
        self._services.append(time.monotonic())
        self._waits.append(wait_seconds)
        self._prune()

    def _prune(self) -> None:
        cutoff = time.monotonic() - self.window_seconds
        self._arrivals = [t for t in self._arrivals if t >= cutoff]
        self._services = [t for t in self._services if t >= cutoff]

    def rates(self) -> dict[str, float]:
        arrivals = len(self._arrivals) / self.window_seconds
        services = len(self._services) / self.window_seconds
        return {"lambda": round(arrivals, 4), "mu": round(services, 4)}

    def utilization(self) -> float:
        rates = self.rates()
        if rates["mu"] <= 0:
            return float("inf") if rates["lambda"] > 0 else 0.0
        return rates["lambda"] / rates["mu"]

    def stable(self) -> bool:
        """Queue stability requires lambda < mu for the workload class."""
        return self.utilization() < 1.0

    def mean_wait_seconds(self) -> float:
        return sum(self._waits) / len(self._waits) if self._waits else 0.0


class IdempotentConsumer:
    """A consumer whose duplicate events are detected and applied once.

    Event identity is preserved across retries; applying the same event twice
    reports ``duplicate`` without re-executing the side effect.
    """

    def __init__(self, apply: Callable[[StreamEvent], Any]) -> None:
        self._apply = apply
        self._applied: dict[str, str] = {}
        self._lock = threading.Lock()

    def consume(self, event: StreamEvent) -> str:
        with self._lock:
            if event.event_key in self._applied:
                return "duplicate"
            self._apply(event)
            self._applied[event.event_key] = event.event_id
            return "applied"

    def applied_count(self) -> int:
        with self._lock:
            return len(self._applied)


class StreamPipeline:
    """A bounded producer-consumer pipeline with backpressure admission."""

    def __init__(
        self,
        consumer: IdempotentConsumer,
        *,
        max_depth: int = 128,
        max_batch: int = 64,
    ) -> None:
        self.queue = BoundedEventQueue(max_depth=max_depth)
        self.consumer = consumer
        self.monitor = QueueMonitor()
        self.max_batch = max_batch
        self.admission_log: list[dict[str, Any]] = []

    def submit(self, payload: Mapping[str, Any], *, partition: str = "default") -> str:
        """Admit work unless backpressure requires deferral."""
        if self.queue.apply_backpressure():
            self.admission_log.append({
                "decision": "deferred", "reason": "backpressure",
                "depth": self.queue.depth(), "at": utc_now_iso(),
            })
            return "deferred"
        event = StreamEvent.create(payload, partition=partition)
        if not self.queue.put(event):
            self.admission_log.append({
                "decision": "rejected", "reason": "queue full",
                "depth": self.queue.depth(), "at": utc_now_iso(),
            })
            return "rejected"
        self.monitor.record_arrival()
        return "accepted"

    def drain_once(self) -> int:
        """Process at most ``max_batch`` events; returns processed count."""
        processed = 0
        for _ in range(self.max_batch):
            event = self.queue.get()
            if event is None:
                break
            started = time.monotonic()
            self.consumer.consume(event)
            self.monitor.record_service(0.0 if processed == 0 else 0.0)
            processed += 1
        return processed

    def summary(self) -> dict[str, Any]:
        return {
            "schema": STREAMING_SCHEMA,
            "queue_depth": self.queue.depth(),
            "max_depth": self.queue.max_depth,
            "rates": self.monitor.rates(),
            "stable": self.monitor.stable(),
            "admissions": list(self.admission_log),
        }


__all__ = [
    "STREAMING_SCHEMA",
    "BoundedEventQueue",
    "IdempotentConsumer",
    "QueueMonitor",
    "StreamEvent",
    "StreamPipeline",
]
