"""Retry-safe task execution: idempotency, outbox, and inbox (Step 41).

Idempotency keys, attempt records, and deduplication separate at-least-once
task delivery from exactly-once effect claims; protected external effects use
the transactional outbox/inbox pattern instead of a universal exactly-once
promise.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping


@dataclass(frozen=True, slots=True)
class EffectRecord:
    """One executed effect recorded under its idempotency key."""

    idempotency_key: str
    task_id: str
    effect_digest: str
    result: Any = None

    def __post_init__(self) -> None:
        """Validate the record identity fields."""
        if not self.idempotency_key.strip() or not self.task_id.strip():
            raise ValueError("idempotency_key and task_id must be non-empty")


class IdempotencyRegistry:
    """Deduplicate effects by idempotency key (Step 41)."""

    def __init__(self) -> None:
        """Create an empty registry."""
        self._effects: dict[str, EffectRecord] = {}
        self.duplicates_blocked = 0

    def try_execute(
        self,
        idempotency_key: str,
        task_id: str,
        effect: Callable[[], Any],
        effect_digest: str = "",
    ) -> tuple[bool, Any]:
        """Execute an effect once per key; repeats return the stored result."""
        if not idempotency_key.strip():
            raise ValueError("idempotency_key must be non-empty")
        if idempotency_key in self._effects:
            self.duplicates_blocked += 1
            record = self._effects[idempotency_key]
            return False, record.result
        result = effect()
        self._effects[idempotency_key] = EffectRecord(
            idempotency_key=idempotency_key,
            task_id=task_id,
            effect_digest=effect_digest or str(idempotency_key),
            result=result,
        )
        return True, result

    def has_executed(self, idempotency_key: str) -> bool:
        """Return whether a key has already produced its effect."""
        return idempotency_key in self._effects

    def records(self) -> tuple[EffectRecord, ...]:
        """Return every recorded effect."""
        return tuple(self._effects.values())


class TransactionalOutbox:
    """Buffer outbound side effects with their producing command (Step 41)."""

    def __init__(self, limit: int = 512) -> None:
        """Create a bounded outbox."""
        if limit < 1:
            raise ValueError("limit must be positive")
        self._entries: list[dict[str, Any]] = []
        self.limit = int(limit)

    def enqueue(self, command_id: str, effect: Mapping[str, Any]) -> int:
        """Record one effect to publish after its command commits."""
        if not command_id.strip():
            raise ValueError("command_id must be non-empty")
        if len(self._entries) >= self.limit:
            raise RuntimeError("transactional outbox is full; apply admission control")
        self._entries.append({"command_id": command_id, "effect": dict(effect)})
        return len(self._entries)

    def drain(self) -> tuple[dict[str, Any], ...]:
        """Return and clear every pending outbox entry."""
        entries = tuple(self._entries)
        self._entries.clear()
        return entries

    def pending(self) -> tuple[dict[str, Any], ...]:
        """Return the pending entries without clearing them."""
        return tuple(self._entries)


class TransactionalInbox:
    """Deduplicate inbound effect deliveries on the receiving side (Step 41)."""

    def __init__(self, limit: int = 512) -> None:
        """Create a bounded inbox."""
        if limit < 1:
            raise ValueError("limit must be positive")
        self._seen: dict[str, dict[str, Any]] = {}
        self.limit = int(limit)
        self.duplicates_dropped = 0

    def receive(self, message_id: str, effect: Mapping[str, Any]) -> str:
        """Accept one delivery; duplicates are dropped by message identifier."""
        if not message_id.strip():
            raise ValueError("message_id must be non-empty")
        if len(self._seen) >= self.limit and message_id not in self._seen:
            raise RuntimeError("transactional inbox is full; apply admission control")
        if message_id in self._seen:
            self.duplicates_dropped += 1
            return "duplicate"
        self._seen[message_id] = dict(effect)
        return "new"

    def messages(self) -> dict[str, dict[str, Any]]:
        """Return the accepted messages."""
        return dict(self._seen)


def at_least_once_vs_exactly_once_note() -> str:
    """Return the standing policy statement for retry semantics (Step 41)."""
    return (
        "Task delivery is at-least-once; exactly-once *effect* claims require "
        "idempotency keys plus the transactional outbox/inbox pattern, or "
        "application-specific idempotency at the effect boundary."
    )


def retries_do_not_duplicate_effects(
    registry: IdempotencyRegistry,
    task_id: str,
    delivery_attempts: int,
) -> tuple[int, int]:
    """Deliver one task several times; return (executions, duplicates blocked)."""
    executions = 0
    for attempt in range(1, delivery_attempts + 1):
        executed, _ = registry.try_execute(
            f"idem-{task_id}",
            task_id,
            lambda: f"result-{task_id}",
            effect_digest="digest",
        )
        if executed:
            executions += 1
    return executions, registry.duplicates_blocked


__all__ = [
    "EffectRecord",
    "IdempotencyRegistry",
    "TransactionalOutbox",
    "TransactionalInbox",
    "at_least_once_vs_exactly_once_note",
    "retries_do_not_duplicate_effects",
]
