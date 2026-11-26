"""Local event routing with typed subscriptions and idempotent delivery."""
from __future__ import annotations

import threading
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..contracts import DeliveryGuarantee
from ..exceptions import EventSchemaError
from .idempotency import ApplyStatus, IdempotencyStore
from .schemas import EventEnvelope, assert_valid_envelope


class DeliveryStatus(str, Enum):
    """Report the delivery outcome for one consumer of one event."""

    DELIVERED = "delivered"
    ALREADY_APPLIED = "already_applied"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class DeliveryOutcome:
    """Record one delivery attempt for the bus trace and metrics."""

    consumer_id: str
    event_id: str
    status: DeliveryStatus
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable delivery record."""
        return {
            "consumer_id": self.consumer_id,
            "event_id": self.event_id,
            "status": self.status.value,
            "error": self.error,
        }


@dataclass(slots=True)
class Subscription:
    """Bind one consumer to selected event types with an optional predicate."""

    consumer_id: str
    callback: Callable[[EventEnvelope], Any]
    event_types: frozenset[str] | None = None
    predicate: Callable[[EventEnvelope], bool] | None = None
    delivery_guarantee: DeliveryGuarantee = DeliveryGuarantee.AT_LEAST_ONCE

    def matches(self, envelope: EventEnvelope) -> bool:
        """Check whether this subscription filters the envelope in."""
        if self.event_types is not None and envelope.event_type not in self.event_types:
            return False
        if self.predicate is not None and not self.predicate(envelope):
            return False
        return True


@dataclass(slots=True)
class PublishRecord:
    """Track one publish call and its consumer outcomes."""

    envelope: EventEnvelope
    outcomes: tuple[DeliveryOutcome, ...] = field(default_factory=())


class EventBus:
    """Route committed events to filtered consumers with honest delivery semantics.

    Consumers subscribe by event type and predicate, never by direct calls to
    other modules. Bus delivery is at-least-once; consumers that declare an
    EFFECTIVELY_ONCE guarantee are deduplicated through the idempotency store,
    which is what makes redelivery safe rather than any physical exactly-once
    transport claim.
    """

    def __init__(
        self, *, idempotency: IdempotencyStore | None = None,
        max_delivery_failures: int = 3,
    ) -> None:
        """Initialize the bus with optional durable consumer deduplication."""
        self._lock = threading.RLock()
        self._subscriptions: dict[str, Subscription] = {}
        self._idempotency = idempotency or IdempotencyStore()
        self._max_delivery_failures = max_delivery_failures
        self._publish_log: list[PublishRecord] = []
        self._published_count = 0
        self._delivery_failures = 0
        self._committed_at: dict[str, float] = {}
        self._consumed_at: dict[str, float] = {}
        self._event_lag_samples: list[float] = []

    # ------------------------------------------------------------ subscribe

    def subscribe(
        self, consumer_id: str, callback: Callable[[EventEnvelope], Any],
        *, event_types: Iterable[str] | None = None,
        predicate: Callable[[EventEnvelope], bool] | None = None,
        delivery_guarantee: DeliveryGuarantee = DeliveryGuarantee.AT_LEAST_ONCE,
    ) -> Subscription:
        """Register one consumer subscription filtered by type or predicate."""
        if not consumer_id.strip():
            raise ValueError("consumer_id must be non-empty.")
        if not callable(callback):
            raise TypeError("callback must be callable.")
        with self._lock:
            if consumer_id in self._subscriptions:
                raise ValueError(f"consumer '{consumer_id}' is already subscribed.")
            subscription = Subscription(
                consumer_id=consumer_id,
                callback=callback,
                event_types=frozenset(event_types) if event_types is not None else None,
                predicate=predicate,
                delivery_guarantee=DeliveryGuarantee(delivery_guarantee),
            )
            self._subscriptions[consumer_id] = subscription
            return subscription

    def unsubscribe(self, consumer_id: str) -> bool:
        """Remove one subscription and report whether it existed."""
        with self._lock:
            return self._subscriptions.pop(consumer_id, None) is not None

    def subscriptions(self) -> tuple[str, ...]:
        """Return every subscribed consumer in deterministic order."""
        with self._lock:
            return tuple(sorted(self._subscriptions))

    # -------------------------------------------------------------- publish

    def publish(self, envelope: EventEnvelope) -> tuple[DeliveryOutcome, ...]:
        """Publish one validated envelope and route it to matching consumers.

        Producers must call publish only after the corresponding state
        transition has committed (or through the transactional outbox), which
        the engine enforces; the bus itself never fabricates causal ordering.
        """
        assert_valid_envelope(envelope)
        import time as _time

        occurred = _time.monotonic()
        with self._lock:
            self._published_count += 1
            self._committed_at.setdefault(envelope.event_id, occurred)
            matching = sorted(
                (subscription for subscription in self._subscriptions.values()
                 if subscription.matches(envelope)),
                key=lambda subscription: subscription.consumer_id,
            )
        outcomes: list[DeliveryOutcome] = []
        for subscription in matching:
            outcomes.append(self._deliver(subscription, envelope))
        record = PublishRecord(envelope=envelope, outcomes=tuple(outcomes))
        with self._lock:
            self._publish_log.append(record)
        return tuple(outcomes)

    def _deliver(self, subscription: Subscription,
                 envelope: EventEnvelope) -> DeliveryOutcome:
        """Deliver one event to one consumer honoring its delivery guarantee."""
        if subscription.delivery_guarantee is DeliveryGuarantee.AT_MOST_ONCE:
            try:
                subscription.callback(envelope)
            except Exception as exc:  # noqa: BLE001 - delivery isolation
                self._delivery_failures += 1
                return DeliveryOutcome(
                    consumer_id=subscription.consumer_id,
                    event_id=envelope.event_id,
                    status=DeliveryStatus.FAILED,
                    error=f"{type(exc).__name__}: {exc}",
                )
            return DeliveryOutcome(
                consumer_id=subscription.consumer_id,
                event_id=envelope.event_id,
                status=DeliveryStatus.DELIVERED,
            )
        if subscription.delivery_guarantee is DeliveryGuarantee.EFFECTIVELY_ONCE:
            status, _ = self._idempotency.apply_once(
                subscription.consumer_id, envelope.event_id,
                effect=lambda: subscription.callback(envelope),
            )
            if status is ApplyStatus.ALREADY_APPLIED:
                return DeliveryOutcome(
                    consumer_id=subscription.consumer_id,
                    event_id=envelope.event_id,
                    status=DeliveryStatus.ALREADY_APPLIED,
                )
            return DeliveryOutcome(
                consumer_id=subscription.consumer_id,
                event_id=envelope.event_id,
                status=DeliveryStatus.DELIVERED,
            )
        # at-least-once: deliver and record failures without deduplication.
        try:
            subscription.callback(envelope)
        except Exception as exc:  # noqa: BLE001 - delivery isolation
            self._delivery_failures += 1
            return DeliveryOutcome(
                consumer_id=subscription.consumer_id,
                event_id=envelope.event_id,
                status=DeliveryStatus.FAILED,
                error=f"{type(exc).__name__}: {exc}",
            )
        return DeliveryOutcome(
            consumer_id=subscription.consumer_id,
            event_id=envelope.event_id,
            status=DeliveryStatus.DELIVERED,
        )

    def redeliver(self, envelope: EventEnvelope) -> tuple[DeliveryOutcome, ...]:
        """Redeliver one envelope to test duplicate-delivery safety explicitly."""
        return self.publish(envelope)

    # --------------------------------------------------------------- queries

    def delivery_failures(self) -> int:
        """Return the count of failed delivery attempts."""
        return self._delivery_failures

    def publish_records(self) -> tuple[PublishRecord, ...]:
        """Return every publish record in publication order."""
        with self._lock:
            return tuple(self._publish_log)

    def event_lag_samples(self) -> tuple[float, ...]:
        """Return observed commit-to-consumption lag samples for metrics."""
        with self._lock:
            return tuple(self._event_lag_samples)

    def stats(self) -> dict[str, Any]:
        """Return bus statistics for the observability report."""
        with self._lock:
            return {
                "published": self._published_count,
                "subscriptions": len(self._subscriptions),
                "delivery_failures": self._delivery_failures,
                "idempotency": self._idempotency.stats(),
            }

    def close(self) -> None:
        """Clear subscriptions so the bus can be discarded deterministically."""
        with self._lock:
            self._subscriptions.clear()
