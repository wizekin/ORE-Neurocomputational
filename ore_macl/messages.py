"""Typed agent message bus with idempotency, TTL expiry, and sender quotas.

Messages are versioned envelopes referencing workspace artifacts (Step 25). The
bus deduplicates by idempotency key, drops expired envelopes, enforces per-sender
quotas, and keeps detached inboxes and outboxes with full lineage so partitions
can be reconciled: expiry or quota rejection distinguishes unknown delivery from
failed execution (Steps 31-32, 35).
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from threading import RLock
from typing import Any

from .contracts import AgentMessage, DeliveryOutcome, MessageKind


@dataclass(frozen=True, slots=True)
class DeliveryReceipt:
    """Record one send decision without conflating duplicates with failures."""

    outcome: DeliveryOutcome
    message_id: str
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a detached receipt record."""
        return {"outcome": self.outcome.value, "message_id": self.message_id, "detail": self.detail}


@dataclass
class MessageWindow:
    """Track one sender's message count inside the current quota window."""

    limit: int
    count: int = 0

    def consume(self) -> bool:
        """Consume one quota slot or report exhaustion."""
        if self.count >= self.limit:
            return False
        self.count += 1
        return True


class AgentMessageBus:
    """Route typed messages with idempotency, expiry, quotas, and lineage (Section 2)."""

    def __init__(self, *, default_quota: int = 128) -> None:
        """Create an empty bus with a bounded default per-sender quota."""
        if default_quota < 1:
            raise ValueError("default_quota must be at least one")
        self._lock = RLock()
        self._inboxes: dict[str, list[AgentMessage]] = {}
        self._outbox: dict[str, list[AgentMessage]] = {}
        self._receipts: list[DeliveryReceipt] = []
        self._seen_idempotency_keys: dict[str, str] = {}
        self._seen_message_ids: set[str] = set()
        self._windows: dict[str, MessageWindow] = {}
        self._default_quota = default_quota
        self._sequence = 0

    def send(self, message: AgentMessage, *, now: float = 0.0) -> DeliveryReceipt:
        """Deliver one message unless duplicate, expired, or over quota."""
        if not isinstance(message, AgentMessage):
            raise TypeError("message must be an AgentMessage")
        with self._lock:
            if message.message_id in self._seen_message_ids:
                return DeliveryReceipt(
                    DeliveryOutcome.DUPLICATE, message.message_id, "message ID already delivered",
                )
            if message.idempotency_key in self._seen_idempotency_keys:
                return DeliveryReceipt(
                    DeliveryOutcome.DUPLICATE, message.message_id,
                    f"idempotency key already used by {self._seen_idempotency_keys[message.idempotency_key]}",
                )
            if message.expires_at is not None and message.expires_at < now:
                return DeliveryReceipt(
                    DeliveryOutcome.EXPIRED, message.message_id,
                    f"message expired at {message.expires_at}",
                )
            window = self._windows.get(message.sender_id)
            if window is None:
                window = MessageWindow(self._default_quota)
                self._windows[message.sender_id] = window
            if not window.consume():
                return DeliveryReceipt(
                    DeliveryOutcome.QUOTA_EXCEEDED, message.message_id,
                    f"sender quota of {window.limit} messages exceeded",
                )
            self._sequence += 1
            self._seen_message_ids.add(message.message_id)
            self._seen_idempotency_keys[message.idempotency_key] = message.message_id
            self._inboxes.setdefault(message.recipient_id, []).append(message)
            self._outbox.setdefault(message.sender_id, []).append(message)
            receipt = DeliveryReceipt(DeliveryOutcome.DELIVERED, message.message_id)
            self._receipts.append(receipt)
            return receipt

    def receive(self, agent_id: str, *, now: float = 0.0) -> tuple[AgentMessage, ...]:
        """Drain one agent's inbox, dropping envelopes that expired while queued."""
        with self._lock:
            queued = self._inboxes.pop(agent_id, [])
        delivered = []
        for message in queued:
            if message.expires_at is not None and message.expires_at < now:
                self._receipts.append(DeliveryReceipt(
                    DeliveryOutcome.EXPIRED, message.message_id, "expired before delivery",
                ))
            else:
                delivered.append(message)
        return tuple(delivered)

    def peek(self, agent_id: str) -> tuple[AgentMessage, ...]:
        """Inspect one inbox without draining it."""
        with self._lock:
            return tuple(self._inboxes.get(agent_id, ()))

    def outbox(self, sender_id: str) -> tuple[AgentMessage, ...]:
        """Return one sender's dispatched messages in order."""
        with self._lock:
            return tuple(self._outbox.get(sender_id, ()))

    def receipts(self) -> tuple[DeliveryReceipt, ...]:
        """Return every send decision for lineage and audits."""
        with self._lock:
            return tuple(self._receipts)

    def reset_window(self, sender_id: str) -> None:
        """Open a fresh quota window for one sender."""
        with self._lock:
            self._windows[sender_id] = MessageWindow(self._default_quota)

    def set_quota(self, sender_id: str, limit: int) -> None:
        """Install an explicit per-sender quota for the current window."""
        if limit < 1:
            raise ValueError("quota limit must be at least one")
        with self._lock:
            self._windows[sender_id] = MessageWindow(limit)

    def has_been_delivered(self, idempotency_key: str) -> bool:
        """Check whether an idempotency key has already been consumed."""
        with self._lock:
            return idempotency_key in self._seen_idempotency_keys

    def stats(self) -> dict[str, Any]:
        """Return compact bus statistics for observability."""
        with self._lock:
            by_outcome: dict[str, int] = {}
            for receipt in self._receipts:
                by_outcome[receipt.outcome.value] = by_outcome.get(receipt.outcome.value, 0) + 1
            return {
                "messages_sent": len(self._receipts),
                "by_outcome": by_outcome,
                "tracked_senders": sorted(self._outbox),
                "pending_inboxes": {
                    agent_id: len(messages) for agent_id, messages in sorted(self._inboxes.items())
                    if messages
                },
                "sequence": self._sequence,
            }


def make_message(
    *,
    sender_id: str,
    recipient_id: str,
    kind: MessageKind,
    task_id: str,
    timestamp: float,
    payload: Mapping[str, Any] | None = None,
    **options: Any,
) -> AgentMessage:
    """Build a validated message envelope with sensible defaults."""
    return AgentMessage(
        sender_id=sender_id,
        recipient_id=recipient_id,
        kind=kind,
        task_id=task_id,
        timestamp=timestamp,
        payload=dict(payload or {}),
        **options,
    )
