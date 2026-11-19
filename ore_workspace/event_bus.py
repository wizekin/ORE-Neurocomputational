"""Isolated delivery of committed workspace-change notifications to subscribed modules."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from threading import RLock
from typing import Callable

from .models import DeliveryResult, WorkspaceEvent
from .subscriptions import SubscriptionRegistry


class WorkspaceEventBus:
    """Dispatch change events only to interested consumers and isolate callback failures."""

    def __init__(self, subscriptions: SubscriptionRegistry, max_workers: int = 8) -> None:
        """Create a bounded event dispatcher using the shared subscription registry."""
        if max_workers <= 0:
            raise ValueError("max_workers must be positive")
        self.subscriptions = subscriptions
        self._lock = RLock()
        self._callbacks: dict[str, Callable[[WorkspaceEvent], object]] = {}
        self._delivery_attempts: set[tuple[str, str]] = set()
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ore-workspace-event")

    def subscribe(self, module_id: str, callback: Callable[[WorkspaceEvent], object]) -> None:
        """Register or replace one module's event callback without changing its field subscription."""
        if not module_id.strip() or not callable(callback):
            raise ValueError("module_id and a callable callback are required")
        with self._lock:
            self._callbacks[module_id] = callback

    def unsubscribe(self, module_id: str) -> bool:
        """Remove one module callback and report whether it was registered."""
        with self._lock:
            return self._callbacks.pop(module_id, None) is not None

    def publish(self, event: WorkspaceEvent, timeout: float | None = 5.0) -> tuple[DeliveryResult, ...]:
        """Deliver an event concurrently and return per-module failures instead of raising them."""
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be positive")
        eligible = self.subscriptions.subscribers_for(event.changed_fields)
        with self._lock:
            callbacks = {}
            for module in eligible:
                identity = (module, event.event_id)
                callback = self._callbacks.get(module)
                if callback is not None and identity not in self._delivery_attempts:
                    self._delivery_attempts.add(identity)
                    callbacks[module] = callback
        futures = {self._executor.submit(callback, event): module for module, callback in callbacks.items()}
        results: list[DeliveryResult] = []
        try:
            for future in as_completed(futures, timeout=timeout):
                module_id = futures[future]
                try:
                    future.result()
                    results.append(DeliveryResult(module_id, event.event_id, True))
                except Exception as exc:  # Subscriber failures are isolated by design.
                    results.append(DeliveryResult(module_id, event.event_id, False,
                                                  f"{type(exc).__name__}: {exc}"))
        except TimeoutError:
            completed = {item.module_id for item in results}
            for future, module_id in futures.items():
                if module_id not in completed:
                    future.cancel()
                    results.append(DeliveryResult(
                        module_id, event.event_id, False,
                        "subscriber delivery timed out", timed_out=True,
                    ))
        return tuple(sorted(results, key=lambda item: item.module_id))

    def close(self, wait: bool = True) -> None:
        """Stop accepting notifications and optionally wait for already-running callbacks."""
        self._executor.shutdown(wait=wait, cancel_futures=True)

    def __enter__(self) -> "WorkspaceEventBus":
        """Return this event bus for context-managed lifecycle handling."""
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        """Shut down the dispatcher without suppressing exceptions from the context body."""
        self.close(wait=True)
        return False