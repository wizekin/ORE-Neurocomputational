"""CROS admission gate (integrated steps 14-16).

HACOL never bypasses CROS: every optimization request passes through a
gate object whose decision is authoritative. The default LocalCROSGate
enforces in-process budgets (memory caps, concurrency slots, queue pressure);
an external AdaptiveResourceCoordinator/CROS can be wrapped through
ExternalCROSAdapter so global allocation authority stays with the caller's
coordinator.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Any, Protocol

from .contracts import AdmissionDecision, ResourceAsk


class CROSGate(Protocol):
    """Minimal admission interface HACOL requires from any coordinator."""

    def admit(self, ask: ResourceAsk, *, task_id: str = "") -> AdmissionDecision: ...


class LocalCROSGate:
    """In-process admission authority with hard caps (integrated step 14).

    Tracks concurrent HACOL optimizations, memory reservations, and queue
    pressure; defers requests whose demands exceed the configured caps or
    whose deadlines cannot plausibly be met.
    """

    def __init__(
        self,
        *,
        memory_cap_bytes: float | None = None,
        max_concurrent: int = 2,
        cpu_share_cap: float = 1.0,
        queue_priority_limit: int = 10,
    ) -> None:
        self.memory_cap_bytes = (
            float(memory_cap_bytes)
            if memory_cap_bytes is not None
            else float(getattr(os, "sysconf", lambda *_: 0)("SC_PAGE_SIZE") * getattr(os, "sysconf", lambda *_: 0)("SC_PHYS_PAGES"))
        )
        self.max_concurrent = max(1, int(max_concurrent))
        self.cpu_share_cap = float(cpu_share_cap)
        self.queue_priority_limit = int(queue_priority_limit)
        self._active = 0
        self._reserved_memory = 0.0
        self._queue_position = 0
        self._lock = threading.Lock()

    def _total_memory(self) -> float:
        return self._reserved_memory

    def admit(self, ask: ResourceAsk, *, task_id: str = "") -> AdmissionDecision:
        """Grant or defer one resource ask; decisions are final for this gate."""
        with self._lock:
            self._queue_position += 1
            queue = self._queue_position
            if self._active >= self.max_concurrent:
                return AdmissionDecision(
                    granted=False,
                    reason=(
                        f"queue pressure: {self._active} active optimizations at "
                        f"max concurrency {self.max_concurrent}"
                    ),
                    queue_position=queue,
                )
            if ask.cpu_share > self.cpu_share_cap:
                return AdmissionDecision(
                    granted=False,
                    reason=f"cpu share {ask.cpu_share} exceeds cap {self.cpu_share_cap}",
                    queue_position=queue,
                )
            if ask.memory_bytes + self._total_memory() > self.memory_cap_bytes:
                return AdmissionDecision(
                    granted=False,
                    reason=(
                        f"memory ask {ask.memory_bytes:.0f}B plus reserved "
                        f"{self._total_memory():.0f}B exceeds cap {self.memory_cap_bytes:.0f}B"
                    ),
                    queue_position=queue,
                )
            if ask.queue_priority > self.queue_priority_limit:
                return AdmissionDecision(
                    granted=False,
                    reason=f"queue priority {ask.queue_priority} above limit",
                    queue_position=queue,
                )
            slack = ask.deadline_seconds
            self._active += 1
            self._reserved_memory += ask.memory_bytes
        return AdmissionDecision(
            granted=True,
            reason="local budgets satisfied",
            granted_memory_bytes=ask.memory_bytes,
            granted_cpu_share=min(ask.cpu_share, self.cpu_share_cap),
            queue_position=queue,
            deadline_slack_seconds=slack,
        )

    def release(self, ask: ResourceAsk) -> None:
        """Release a granted admission slot."""
        with self._lock:
            self._active = max(0, self._active - 1)
            self._reserved_memory = max(0.0, self._reserved_memory - ask.memory_bytes)

    # Context-manager convenience so granted asks are always released.
    def session(self, ask: ResourceAsk, task_id: str = ""):
        return _AdmissionSession(self, ask, task_id)


class _AdmissionSession:
    def __init__(self, gate: LocalCROSGate, ask: ResourceAsk, task_id: str) -> None:
        self.gate = gate
        self.ask = ask
        self.task_id = task_id
        self.decision: AdmissionDecision | None = None

    def __enter__(self) -> "LocalCROSGate":
        self.decision = self.gate.admit(self.ask, task_id=self.task_id)
        return self.gate

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        if self.decision is not None and self.decision.granted:
            self.gate.release(self.ask)
        return False


class ExternalCROSAdapter:
    """Adapter delegating admission to an external coordinator object.

    The external coordinator must expose ``admit(ask, **options)`` returning
    an object with a truthy ``granted`` field, or callable ``admit(ask)``
    returning a bool. Global authority remains with that coordinator; HACOL
    only supplies measured cost estimates alongside its recommendation.
    """

    def __init__(self, coordinator: Any, *, label: str = "external_cros") -> None:
        if coordinator is None:
            raise ValueError("coordinator must be a live coordinator object")
        self.coordinator = coordinator
        self.label = label

    def admit(self, ask: ResourceAsk, *, task_id: str = "") -> AdmissionDecision:
        """Translate the local ask into the coordinator's decision."""
        started = time.perf_counter()
        try:
            result = self.coordinator.admit(ask, task_id=task_id)
        except TypeError:
            result = self.coordinator.admit(ask)
        elapsed = time.perf_counter() - started
        granted = bool(getattr(result, "granted", result))
        reason = getattr(result, "reason", "external coordinator decision")
        return AdmissionDecision(
            granted=granted,
            reason=f"[{self.label}] {reason}",
            granted_memory_bytes=ask.memory_bytes if granted else 0.0,
            granted_cpu_share=ask.cpu_share if granted else 0.0,
            deadline_slack_seconds=ask.deadline_seconds,
        )


def default_cros_gate(
    cros: Any | None = None,
    *,
    memory_cap_bytes: float | None = None,
    max_concurrent: int = 2,
) -> CROSGate:
    """Build the gate HACOL will use: external coordinator if provided."""
    if cros is None:
        return LocalCROSGate(
            memory_cap_bytes=memory_cap_bytes, max_concurrent=max_concurrent
        )
    if isinstance(cros, (LocalCROSGate, ExternalCROSAdapter)):
        return cros
    if hasattr(cros, "admit"):
        return ExternalCROSAdapter(cros)
    raise TypeError("cros must expose admit(ask) or be a CROSGate instance")
