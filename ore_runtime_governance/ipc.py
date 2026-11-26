"""IPC plane: typed envelopes, bounded queues, back-pressure (steps 10-14).

``MessageBus`` validates every envelope (type, size, schema version, sender
authentication, capability authorization) before dispatch, keeps queues
bounded with an admission factor a(q)=max(0, 1-q/K) and discrete thresholds,
reserves capacity for critical messages, and never silently discards a
committed task (integrated step 12). Requests carry deadlines on a monotonic
clock and cancellation tokens; timeout is distinguished from cancellation and
from confirmed failure (step 13). Effectively-once delivery is obtained by
deduplicating idempotent requests (step 33); queue stability follows the
lambda < mu condition of section 5.4, monitored by queue age, not length alone.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .capabilities import AccessRequest
from .contracts import (
    ENVELOPE_SCHEMA_VERSION,
    CapabilityDeniedError,
    EnvelopeValidationError,
    ImpersonationError,
    MessageEnvelope,
    OperationCancelledError,
    OperationTimedOutError,
    OutcomeKind,
    QueueBackpressureError,
    ResponseEnvelope,
)

CRITICAL_PRIORITY = 9


def admission_factor(queue_length: int, capacity: int) -> float:
    """a(q) = max(0, 1 - q/K) (integrated step 12)."""
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    return max(0.0, 1.0 - queue_length / capacity)


def queue_stable(arrival_rate: float, service_rate: float) -> bool:
    """Long-run queue stability requires lambda < mu (section 5.4)."""
    if arrival_rate < 0 or service_rate <= 0:
        raise ValueError("rates must be positive magnitudes")
    return arrival_rate < service_rate


class CancellationToken:
    """Cooperative cancellation token propagated with requests (step 13)."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._reason = ""

    def cancel(self, reason: str = "cancelled by operator") -> None:
        with self._lock:
            self._reason = reason
        self._event.set()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str:
        with self._lock:
            return self._reason

    def wait(self, timeout: float | None = None) -> bool:
        return self._event.wait(timeout)

    def raise_if_cancelled(self) -> None:
        if self.is_cancelled:
            raise OperationCancelledError(f"operation cancelled: {self.reason}")


class QueueDecisionKind:
    ACCEPT = "accept"
    DEFER = "defer"
    REJECT = "reject"


@dataclass(frozen=True)
class QueueDecision:
    """Typed bounded-queue admission decision (step 12)."""

    kind: str
    reason: str
    admission_factor_value: float
    queue_length: int
    capacity: int
    priority: int

    @property
    def accepted(self) -> bool:
        return self.kind == QueueDecisionKind.ACCEPT


@dataclass(frozen=True)
class BackpressurePolicy:
    """Discrete admission thresholds with a critical reserve (step 12).

    ``defer_threshold``/``reject_threshold`` are fractions of capacity K;
    ``critical_reserve`` slots are reserved for messages with priority at or
    above ``critical_priority``.
    """

    capacity: int = 64
    defer_threshold: float = 0.75
    reject_threshold: float = 0.9
    critical_reserve: int = 2
    critical_priority: int = CRITICAL_PRIORITY

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        if not 0.0 <= self.defer_threshold <= self.reject_threshold <= 1.0:
            raise ValueError("thresholds must satisfy 0 <= defer <= reject <= 1")
        if not 0 <= self.critical_reserve <= self.capacity:
            raise ValueError("critical reserve must fit inside the capacity")
        if not 0 <= self.critical_priority <= 10:
            raise ValueError("critical priority must be within 0..10")

    def admission_factor(self, queue_length: int) -> float:
        return admission_factor(queue_length, self.capacity)

    def decide(self, queue_length: int, priority: int) -> QueueDecision:
        if queue_length < 0:
            raise ValueError("queue length must be non-negative")
        factor = self.admission_factor(queue_length)
        if queue_length >= self.capacity:
            return QueueDecision(
                kind=QueueDecisionKind.REJECT,
                reason=(
                    f"queue full: {queue_length}/{self.capacity} entries; rejecting "
                    "typed (committed work is never silently discarded)"
                ),
                admission_factor_value=factor,
                queue_length=queue_length,
                capacity=self.capacity,
                priority=priority,
            )
        reserve_line = self.capacity - self.critical_reserve
        if queue_length >= reserve_line and priority < self.critical_priority:
            return QueueDecision(
                kind=QueueDecisionKind.REJECT,
                reason=(
                    f"critical reserve exhausted: {self.critical_reserve} slots are "
                    f"reserved for priority >= {self.critical_priority}"
                ),
                admission_factor_value=factor,
                queue_length=queue_length,
                capacity=self.capacity,
                priority=priority,
            )
        ratio = queue_length / self.capacity
        if ratio >= self.reject_threshold and priority < self.critical_priority:
            return QueueDecision(
                kind=QueueDecisionKind.REJECT,
                reason=(
                    f"queue pressure {ratio:.2f} >= reject threshold "
                    f"{self.reject_threshold}; shedding non-critical work"
                ),
                admission_factor_value=factor,
                queue_length=queue_length,
                capacity=self.capacity,
                priority=priority,
            )
        if ratio >= self.defer_threshold and priority < self.critical_priority:
            return QueueDecision(
                kind=QueueDecisionKind.DEFER,
                reason=(
                    f"queue pressure {ratio:.2f} >= defer threshold "
                    f"{self.defer_threshold}; deferring non-critical work"
                ),
                admission_factor_value=factor,
                queue_length=queue_length,
                capacity=self.capacity,
                priority=priority,
            )
        return QueueDecision(
            kind=QueueDecisionKind.ACCEPT,
            reason="queue capacity available",
            admission_factor_value=factor,
            queue_length=queue_length,
            capacity=self.capacity,
            priority=priority,
        )


@dataclass
class DispatchStatistics:
    """Counters for IPC observability (sections 5.4 and 9)."""

    offered: int = 0
    accepted: int = 0
    rejected: int = 0
    deferred: int = 0
    deduplicated: int = 0
    denied_authorization: int = 0
    timed_out: int = 0
    cancelled: int = 0
    internal_failures: int = 0
    unavailable_recipients: int = 0
    max_queue_age_seconds: float = 0.0

    def snapshot(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in vars(self)}


class DeduplicationCache:
    """Bounded message-id -> response cache for effectively-once delivery (step 33)."""

    def __init__(self, max_entries: int = 4096) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self._max_entries = max_entries
        self._responses: dict[str, ResponseEnvelope] = {}
        self._order: deque[str] = deque()

    def record(self, message_id: str, response: ResponseEnvelope) -> None:
        if message_id in self._responses:
            return
        if len(self._order) >= self._max_entries:
            evicted = self._order.popleft()
            self._responses.pop(evicted, None)
        self._responses[message_id] = response
        self._order.append(message_id)

    def lookup(self, message_id: str) -> ResponseEnvelope | None:
        return self._responses.get(message_id)

    def size(self) -> int:
        return len(self._responses)


class BoundedQueue:
    """Thread-safe bounded FIFO with admission policy and age tracking (step 12)."""

    def __init__(self, policy: BackpressurePolicy | None = None) -> None:
        self.policy = policy or BackpressurePolicy()
        self._items: deque[tuple[MessageEnvelope, float]] = deque()
        self._lock = threading.Lock()

    def offer(self, envelope: MessageEnvelope) -> QueueDecision:
        with self._lock:
            decision = self.policy.decide(len(self._items), envelope.priority)
            if decision.accepted:
                self._items.append((envelope, time.monotonic()))
            return decision

    def pop(self) -> MessageEnvelope | None:
        with self._lock:
            if not self._items:
                return None
            return self._items.popleft()[0]

    def qsize(self) -> int:
        with self._lock:
            return len(self._items)

    def queue_age_seconds(self, now: float | None = None) -> float:
        """Age of the oldest entry -- bursts overload queues before length shows it."""
        with self._lock:
            if not self._items:
                return 0.0
            oldest = self._items[0][1]
            return (now if now is not None else time.monotonic()) - oldest

    def clear(self) -> int:
        with self._lock:
            dropped = len(self._items)
            self._items.clear()
            return dropped


class MessageBus:
    """Typed request/reply bus with validation, authority, and back-pressure.

    Dispatch is synchronous (coordinator event loops stay non-blocking by
    moving CPU-heavy work to worker backends, integrated step 14). Every
    denial is typed and explainable; committed tasks are never dropped.
    """

    def __init__(
        self,
        *,
        capacity: int = 64,
        max_payload_bytes: int = 65536,
        policy: BackpressurePolicy | None = None,
        capability_resolver: Callable[[str], Any] | None = None,
        authorizer: Callable[[Any, Any, float | None], Any] | None = None,
        authenticator: Callable[[str], Any] | None = None,
        on_event: Callable[[str, Mapping[str, Any]], None] | None = None,
        clock=None,
    ) -> None:
        self.policy = policy or BackpressurePolicy(capacity=capacity)
        self.max_payload_bytes = int(max_payload_bytes)
        if self.max_payload_bytes <= 0:
            raise ValueError("max_payload_bytes must be positive")
        self.queue = BoundedQueue(self.policy)
        self.dedupe = DeduplicationCache()
        self.statistics = DispatchStatistics()
        self._handlers: dict[str, Callable[[MessageEnvelope], Mapping[str, Any]]] = {}
        self._capability_resolver = capability_resolver
        self._authorizer = authorizer
        self._authenticator = authenticator
        self._on_event = on_event
        self._clock = clock or time.monotonic
        self._paused = False
        self._pause_lock = threading.Lock()
        self._pause_event = threading.Event()
        self._pause_event.set()

    # -- registration ---------------------------------------------------------

    def register_handler(
        self, recipient: str, handler: Callable[[MessageEnvelope], Mapping[str, Any]]
    ) -> None:
        if not recipient or not callable(handler):
            raise ValueError("recipient must be non-empty and handler callable")
        self._handlers[recipient] = handler

    def unregister_handler(self, recipient: str) -> bool:
        return self._handlers.pop(recipient, None) is not None

    def handlers(self) -> tuple[str, ...]:
        return tuple(sorted(self._handlers))

    # -- pause/resume (fault injection) ----------------------------------------

    def pause(self) -> None:
        with self._pause_lock:
            self._paused = True
            self._pause_event.clear()

    def resume(self) -> None:
        with self._pause_lock:
            self._paused = False
            self._pause_event.set()

    @property
    def paused(self) -> bool:
        return self._paused

    # -- validation -------------------------------------------------------------

    def validate_envelope(self, envelope: MessageEnvelope) -> int:
        """Validate type, schema, payload shape, and size (steps 10 and 29)."""
        if not isinstance(envelope, MessageEnvelope):
            raise EnvelopeValidationError(
                f"cross-process requests require MessageEnvelope objects, got {type(envelope)!r}"
            )
        if envelope.schema_version != ENVELOPE_SCHEMA_VERSION:
            raise EnvelopeValidationError(
                f"unsupported envelope schema version {envelope.schema_version}"
            )
        if not isinstance(envelope.payload, Mapping):
            raise EnvelopeValidationError("payload must be a mapping")
        size = envelope.estimated_payload_bytes()
        if size > self.max_payload_bytes:
            raise EnvelopeValidationError(
                f"payload of {size} bytes exceeds the {self.max_payload_bytes} byte "
                "limit; oversized messages are rejected before dispatch"
            )
        return size

    # -- request path -------------------------------------------------------------

    def request(
        self,
        envelope: MessageEnvelope,
        *,
        cancellation: CancellationToken | None = None,
    ) -> ResponseEnvelope:
        """Validate, authorize, admit, and dispatch one request (step 10)."""
        self.statistics.offered += 1
        self.validate_envelope(envelope)

        self._authenticate_sender(envelope)
        self._authorize_sender(envelope)
        if cancellation is not None and cancellation.is_cancelled:
            self.statistics.cancelled += 1
            raise OperationCancelledError(
                f"request cancelled before dispatch: {cancellation.reason}"
            )

        self._await_if_paused(envelope)

        if envelope.delivery.value == "effectively_once":
            cached = self.dedupe.lookup(envelope.message_id)
            if cached is not None:
                self.statistics.deduplicated += 1
                return cached

        decision = self.queue.offer(envelope)
        if decision.kind == QueueDecisionKind.REJECT:
            self.statistics.rejected += 1
            self._emit_event(
                "queue_backpressure",
                {
                    "message_id": envelope.message_id,
                    "decision": decision.kind,
                    "reason": decision.reason,
                    "admission_factor": decision.admission_factor_value,
                    "queue_length": decision.queue_length,
                    "capacity": decision.capacity,
                },
            )
            raise QueueBackpressureError(
                decision.reason,
                kind=decision.kind,
                admission_factor=decision.admission_factor_value,
            )

        deadline = envelope.deadline_monotonic_s
        if decision.kind == QueueDecisionKind.DEFER:
            self.statistics.deferred += 1
            self._await_admission(envelope, deadline)
            self.statistics.accepted += 1
        else:
            self.statistics.accepted += 1

        envelope = self.queue.pop() or envelope
        if deadline is not None and deadline - self._clock() <= 0:
            self.statistics.timed_out += 1
            raise OperationTimedOutError(
                "deadline exhausted before dispatch; task not started "
                "(D_remaining <= 0)"
            )
        if cancellation is not None and cancellation.is_cancelled:
            self.statistics.cancelled += 1
            raise OperationCancelledError(
                f"request cancelled while queued: {cancellation.reason}"
            )

        response = self._dispatch(envelope, cancellation, deadline)
        if envelope.delivery.value == "effectively_once":
            self.dedupe.record(envelope.message_id, response)
        return response

    # -- internals ------------------------------------------------------------

    def _authenticate_sender(self, envelope: MessageEnvelope) -> None:
        if self._authenticator is None:
            return
        identity = self._authenticator(envelope.sender)
        known = bool(getattr(identity, "known", False))
        if not known:
            note = getattr(identity, "note", "")
            raise ImpersonationError(
                f"sender {envelope.sender!r} is not an authenticated component "
                f"({note}); impersonation is denied"
            )

    def _authorize_sender(self, envelope: MessageEnvelope) -> None:
        if self._authorizer is None or self._capability_resolver is None:
            return
        capability_id = envelope.auth_capability_id
        if not capability_id:
            self.statistics.denied_authorization += 1
            self._emit_event(
                "authorization_denied",
                {
                    "actor": envelope.sender,
                    "resource": f"ipc/{envelope.recipient}",
                    "reason": "no capability attached; deny by default",
                },
            )
            raise CapabilityDeniedError(
                "request carries no capability; authorization denies by default",
                subject=envelope.sender,
                right="ipc.send",
            )
        capability = self._capability_resolver(capability_id)
        # The bus checks the dispatch right only (integrated step 16: "check
        # capabilities at registration, dispatch, data access, artifact
        # retrieval, and side-effect execution"): domain rights such as
        # workspace.append_observation are enforced by the receiving guard,
        # not re-checked here.
        request = AccessRequest(
            subject=envelope.sender,
            resource=f"ipc/{envelope.recipient}",
            right="ipc.send",
            constraints={"task_id": envelope.idempotency_key or envelope.message_id},
        )
        decision = self._authorizer(capability, request, None)
        if not getattr(decision, "allowed", False):
            self.statistics.denied_authorization += 1
            reason = getattr(decision, "reason", "authorization denied")
            self._emit_event(
                "authorization_denied",
                {
                    "actor": envelope.sender,
                    "resource": f"ipc/{envelope.recipient}",
                    "right": "ipc.send",
                    "policy_version": getattr(decision, "policy_version", 1),
                    "reason": reason,
                },
            )
            raise CapabilityDeniedError(
                f"ipc authorization denied: {reason}",
                subject=envelope.sender,
                right="ipc.send",
            )

    def _await_if_paused(self, envelope: MessageEnvelope) -> None:
        deadline = envelope.deadline_monotonic_s
        if not self._paused:
            return
        remaining = None if deadline is None else deadline - self._clock()
        if remaining is not None and remaining <= 0:
            raise OperationTimedOutError("ipc plane paused past the request deadline")
        self._pause_event.wait(timeout=remaining if remaining is not None else 60.0)
        if self._paused:
            raise OperationTimedOutError("ipc plane paused; deadline exhausted")

    def _await_admission(self, envelope: MessageEnvelope, deadline: float | None) -> None:
        deadline = deadline if deadline is not None else self._clock() + 5.0
        while self._clock() < deadline:
            if self.queue.qsize() < self.policy.capacity:
                return
            time.sleep(0.01)
        raise QueueBackpressureError(
            "deferral window elapsed without admission; request stays uncommitted",
            kind="deferred",
            admission_factor=admission_factor(self.queue.qsize(), self.policy.capacity),
        )

    def _dispatch(
        self,
        envelope: MessageEnvelope,
        cancellation: CancellationToken | None,
        deadline: float | None,
    ) -> ResponseEnvelope:
        handler = self._handlers.get(envelope.recipient)
        started = self._clock()
        if handler is None:
            self.statistics.unavailable_recipients += 1
            return self._reply(
                envelope,
                OutcomeKind.UNAVAILABLE,
                result=None,
                reason=f"recipient {envelope.recipient!r} has no registered handler",
                elapsed=0.0,
            )
        if cancellation is not None and cancellation.is_cancelled:
            self.statistics.cancelled += 1
            return self._reply(
                envelope,
                OutcomeKind.CANCELLED,
                result=None,
                reason=f"cancelled before execution: {cancellation.reason}",
                elapsed=0.0,
            )
        try:
            result = handler(envelope)
        except OperationTimedOutError as exc:
            self.statistics.timed_out += 1
            return self._reply(
                envelope, OutcomeKind.TIMED_OUT, None, f"operation timed out: {exc}", 0.0
            )
        except OperationCancelledError as exc:
            self.statistics.cancelled += 1
            return self._reply(
                envelope, OutcomeKind.CANCELLED, None, f"operation cancelled: {exc}", 0.0
            )
        except Exception as exc:  # confirmed failure, distinct from timeout
            self.statistics.internal_failures += 1
            return self._reply(
                envelope,
                OutcomeKind.INTERNAL,
                None,
                f"handler failed: {type(exc).__name__}: {exc}",
                self._clock() - started,
            )
        elapsed = self._clock() - started
        if cancellation is not None and cancellation.is_cancelled:
            self.statistics.cancelled += 1
            return self._reply(
                envelope,
                OutcomeKind.CANCELLED,
                result if isinstance(result, Mapping) else None,
                reason=f"cancelled after execution: {cancellation.reason}",
                elapsed=elapsed,
            )
        if deadline is not None and elapsed > max(0.0, deadline - started):
            # The worker may still have completed the operation (step 13):
            # report TIMED_OUT while preserving the produced result.
            self.statistics.timed_out += 1
            return self._reply(
                envelope,
                OutcomeKind.TIMED_OUT,
                result if isinstance(result, Mapping) else None,
                reason="deadline exceeded during execution; result retained",
                elapsed=elapsed,
            )
        if not isinstance(result, Mapping):
            return self._reply(
                envelope,
                OutcomeKind.INTERNAL,
                None,
                f"handler returned a non-mapping result ({type(result).__name__})",
                elapsed=elapsed,
            )
        return self._reply(envelope, OutcomeKind.OK, result, "ok", elapsed)

    def _reply(
        self,
        envelope: MessageEnvelope,
        status: OutcomeKind,
        result: Mapping[str, Any] | None,
        reason: str,
        elapsed: float,
    ) -> ResponseEnvelope:
        return ResponseEnvelope(
            correlation_id=envelope.message_id,
            sender=envelope.recipient,
            recipient=envelope.sender,
            status=status,
            result=result,
            reason=reason,
            elapsed_monotonic_s=max(0.0, elapsed),
            trace=envelope.trace.child(envelope.message_id),
        )

    def _emit_event(self, kind: str, metadata: Mapping[str, Any]) -> None:
        if self._on_event is not None:
            try:
                self._on_event(kind, dict(metadata))
            except Exception:
                # Observability must never break the dispatch path.
                pass

    def statistics_snapshot(self) -> dict[str, Any]:
        snapshot = self.statistics.snapshot()
        snapshot["queue_length"] = self.queue.qsize()
        snapshot["queue_age_seconds"] = self.queue.queue_age_seconds()
        snapshot["deduplicated_cache_size"] = self.dedupe.size()
        snapshot["paused"] = self._paused
        return snapshot
