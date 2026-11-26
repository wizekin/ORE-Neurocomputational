"""Side-effect safety: idempotency, receipts, outbox/inbox, reconciliation (Step 23).

Retries cannot duplicate externally visible side effects.  Every effect is
executed at most once per idempotency key, produces an operation receipt, and
an uncertain outcome must be reconciled (or confirmed by a human) before any
retry is allowed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping

from ..contracts import content_digest, new_id, utc_now

__all__ = (
    "CompensationNotAuthorizedError",
    "CompensationNotDeclaredError",
    "EffectStatus",
    "OutboxEntry",
    "OutboxState",
    "ReconciliationRequiredError",
    "SideEffectHandle",
    "SideEffectLedger",
    "SideEffectReceipt",
)


class CompensationNotDeclaredError(RuntimeError):
    """No compensation was declared for an effect's operation (Step 22)."""


class CompensationNotAuthorizedError(RuntimeError):
    """Compensation was declared but not authorized (Step 22)."""


class EffectStatus(str, Enum):
    """Status of one externally visible operation."""

    CONFIRMED = "confirmed"
    UNCERTAIN = "uncertain"
    FAILED = "failed"


class ReconciliationRequiredError(RuntimeError):
    """Raised when an uncertain side effect blocks a retry (Step 23)."""


@dataclass(frozen=True, slots=True)
class SideEffectReceipt:
    """The durable receipt of one side-effecting operation (Step 23)."""

    effect_id: str
    idempotency_key: str
    operation: str
    status: EffectStatus
    result_digest: str
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the receipt."""
        return {
            "effect_id": self.effect_id,
            "idempotency_key": self.idempotency_key,
            "operation": self.operation,
            "status": self.status.value,
            "result_digest": self.result_digest,
            "created_at": self.created_at,
        }


class OutboxState(str, Enum):
    """Outbox entry lifecycle."""

    PENDING = "pending"
    DISPATCHED = "dispatched"
    ABANDONED = "abandoned"


@dataclass(slots=True)
class OutboxEntry:
    """One queued external operation awaiting dispatch (outbox pattern)."""

    entry_id: str
    workflow_id: str
    task_id: str
    operation: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    state: OutboxState = OutboxState.PENDING
    receipt: SideEffectReceipt | None = None
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the entry."""
        return {
            "entry_id": self.entry_id,
            "workflow_id": self.workflow_id,
            "task_id": self.task_id,
            "operation": self.operation,
            "payload": dict(self.payload),
            "state": self.state.value,
            "receipt": self.receipt.to_dict() if self.receipt else None,
            "created_at": self.created_at,
        }


class SideEffectLedger:
    """Deduplication ledger and outbox for side-effecting operations (Step 23).

    The ledger is the single authority for "did this effect already happen";
    callers must consult it before executing any external operation and must
    not bypass it with direct calls.
    """

    def __init__(self) -> None:
        """Start with an empty ledger and outbox."""
        self._receipts: dict[str, SideEffectReceipt] = {}
        self._outbox: dict[str, OutboxEntry] = {}
        self._inbox_seen: set[str] = set()
        self.deduplicated_hits = 0
        self._compensations: dict[str, Callable[[SideEffectReceipt], Any]] = {}
        self.compensations_invoked: list[str] = []

    def receipt_for(self, idempotency_key: str) -> SideEffectReceipt | None:
        """Return the recorded receipt for a key, if any."""
        return self._receipts.get(idempotency_key)

    def requires_reconciliation(self, idempotency_key: str) -> bool:
        """Return whether an uncertain outcome blocks further attempts."""
        receipt = self._receipts.get(idempotency_key)
        return receipt is not None and receipt.status is EffectStatus.UNCERTAIN

    def begin(self, idempotency_key: str, operation: str,
              workflow_id: str = "", task_id: str = "") -> OutboxEntry:
        """Enqueue an intended external operation in the outbox."""
        entry = OutboxEntry(entry_id=new_id("outbox"), workflow_id=workflow_id,
                            task_id=task_id, operation=operation)
        self._outbox[entry.entry_id] = entry
        return entry

    def commit(self, idempotency_key: str, operation: str,
               result: Any, status: EffectStatus = EffectStatus.CONFIRMED) -> SideEffectReceipt:
        """Record the receipt of one executed effect (at most once per key).

        A confirmed or uncertain receipt can never be overwritten; a failed
        receipt (the effect provably did not happen) may be replaced by a
        re-attempt.
        """
        existing = self._receipts.get(idempotency_key)
        if existing is not None and existing.status is not EffectStatus.FAILED:
            raise ReconciliationRequiredError(
                f"effect {idempotency_key} already has a receipt")
        receipt = SideEffectReceipt(
            effect_id=new_id("effect"), idempotency_key=idempotency_key,
            operation=operation, status=status,
            result_digest=content_digest(result) if result is not None else "")
        self._receipts[idempotency_key] = receipt
        return receipt

    def reconcile(self, idempotency_key: str, confirmed: bool) -> SideEffectReceipt:
        """Resolve an uncertain outcome (Step 23): confirm or mark failed."""
        receipt = self._receipts.get(idempotency_key)
        if receipt is None:
            raise KeyError(f"no receipt for {idempotency_key}")
        if receipt.status is not EffectStatus.UNCERTAIN:
            raise ReconciliationRequiredError(
                f"receipt {idempotency_key} is not uncertain")
        updated = SideEffectReceipt(
            effect_id=receipt.effect_id, idempotency_key=idempotency_key,
            operation=receipt.operation,
            status=EffectStatus.CONFIRMED if confirmed else EffectStatus.FAILED,
            result_digest=receipt.result_digest, created_at=receipt.created_at)
        self._receipts[idempotency_key] = updated
        return updated

    def seen_message(self, message_id: str) -> bool:
        """Inbox deduplication for external callbacks; True if already seen."""
        if message_id in self._inbox_seen:
            return True
        self._inbox_seen.add(message_id)
        return False

    def pending_outbox(self) -> tuple[OutboxEntry, ...]:
        """Return the not-yet-dispatched outbox entries (graceful shutdown)."""
        return tuple(entry for entry in self._outbox.values()
                     if entry.state is OutboxState.PENDING)

    def mark_dispatched(self, entry_id: str,
                        receipt: SideEffectReceipt | None = None) -> None:
        """Mark an outbox entry as dispatched with its receipt."""
        entry = self._outbox.get(entry_id)
        if entry is None:
            raise KeyError(f"unknown outbox entry: {entry_id}")
        entry.state = OutboxState.DISPATCHED
        entry.receipt = receipt

    def execute_once(self, idempotency_key: str, operation: str,
                     action: Callable[[], Any],
                     outcome_status: EffectStatus = EffectStatus.CONFIRMED,
                     ) -> tuple[Any, SideEffectReceipt | None]:
        """Execute an effect exactly once per key, returning (result, receipt).

        A confirmed prior receipt short-circuits execution and returns
        ``(None, receipt)``; an uncertain prior receipt raises
        :class:`ReconciliationRequiredError` until reconciled; a fresh key
        executes the action and records the receipt.
        """
        existing = self._receipts.get(idempotency_key)
        if existing is not None:
            if existing.status is EffectStatus.UNCERTAIN:
                raise ReconciliationRequiredError(
                    f"effect {idempotency_key} is uncertain and must be reconciled "
                    "before any retry")
            if existing.status is EffectStatus.CONFIRMED:
                self.deduplicated_hits += 1
                return None, existing
            # A FAILED receipt proves the effect never happened; execute again.
        result = action()
        receipt = self.commit(idempotency_key, operation, result, outcome_status)
        return result, receipt

    def declare_compensation(self, operation: str,
                              handler: Callable[[SideEffectReceipt], Any]) -> None:
        """Declare the compensation handler for one operation (Step 22)."""
        self._compensations[operation] = handler

    def compensate(self, idempotency_key: str, *, authorized: bool
                   ) -> SideEffectReceipt | None:
        """Invoke declared compensation only when safe and authorized (Step 22).

        Compensation is not guaranteed rollback: it runs the declared handler
        for a confirmed effect and records the invocation.  Raises
        :class:`CompensationNotDeclaredError` when no handler was declared and
        :class:`CompensationNotAuthorizedError` when authorization is missing.
        """
        receipt = self._receipts.get(idempotency_key)
        if receipt is None:
            raise KeyError(f"no receipt for {idempotency_key}")
        handler = self._compensations.get(receipt.operation)
        if handler is None:
            raise CompensationNotDeclaredError(
                f"no compensation declared for operation {receipt.operation!r}")
        if not authorized:
            raise CompensationNotAuthorizedError(
                f"compensation for {idempotency_key} is not authorized")
        handler(receipt)
        self.compensations_invoked.append(idempotency_key)
        return receipt

    def receipts(self) -> tuple[SideEffectReceipt, ...]:
        """Return all recorded receipts."""
        return tuple(self._receipts.values())

    def describe(self) -> dict[str, Any]:
        """Return a serializable summary."""
        return {
            "receipts": len(self._receipts),
            "pending_outbox": len(self.pending_outbox()),
            "effects": [receipt.to_dict() for receipt in self._receipts.values()],
        }


class SideEffectHandle:
    """The narrow handle passed to workers inside the task envelope (Step 12).

    Workers never receive the ledger itself, only this scoped handle bound to
    one workflow, task and permitted operation list, preserving least
    privilege.
    """

    def __init__(self, ledger: SideEffectLedger, workflow_id: str, task_id: str,
                 permitted_operations: tuple[str, ...]) -> None:
        """Bind the handle to one task scope."""
        self._ledger = ledger
        self.workflow_id = workflow_id
        self.task_id = task_id
        self.permitted_operations = permitted_operations

    def execute(self, operation: str, payload: Mapping[str, Any],
                action: Callable[[], Any],
                status: EffectStatus = EffectStatus.CONFIRMED) -> tuple[Any, SideEffectReceipt | None]:
        """Execute one permitted external effect exactly once per key."""
        if operation not in self.permitted_operations:
            raise PermissionError(
                f"operation {operation!r} is not permitted for task {self.task_id}")
        key = f"{self.workflow_id}:{self.task_id}:{operation}"
        self._ledger.begin(key, operation, self.workflow_id, self.task_id)
        result, receipt = self._ledger.execute_once(key, operation, action,
                                                    outcome_status=status)
        return result, receipt

    def uncertain(self, operation: str) -> bool:
        """Return whether the task's effect for an operation is uncertain."""
        key = f"{self.workflow_id}:{self.task_id}:{operation}"
        return self._ledger.requires_reconciliation(key)
