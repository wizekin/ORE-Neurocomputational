"""Bounded event-driven module activation with dependency-aware task execution."""
from __future__ import annotations

import asyncio
import inspect
import queue
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

from .exceptions import SchedulerLimitError
from .models import Finding, ModuleActivationResult, WorkspaceEvent, WorkspaceSnapshot


@dataclass(frozen=True, slots=True)
class SchedulerReport:
    """Summarize one bounded activation session and its publication outcomes."""

    rounds: int
    activations: tuple[ModuleActivationResult, ...]
    publication_results: tuple[Any, ...]
    stopped_due_to_limit: bool = False


@dataclass(slots=True)
class _ModuleHandler:
    """Hold one registered module callback and its configured execution timeout."""

    callback: Callable[[WorkspaceSnapshot], Any]
    timeout_seconds: float


class ModuleScheduler:
    """Trigger only relevant modules and bound rounds, execution concurrency and task duration."""

    def __init__(
        self,
        *,
        max_workers: int = 8,
        max_rounds: int = 8,
        max_activations: int = 64,
        default_timeout_seconds: float = 10.0,
    ) -> None:
        """Configure scheduler limits that prevent unbounded activation loops or stuck modules."""
        if min(max_workers, max_rounds, max_activations) <= 0 or default_timeout_seconds <= 0:
            raise ValueError("scheduler limits and timeout must be positive")
        self.max_workers = max_workers
        self.max_rounds = max_rounds
        self.max_activations = max_activations
        self.default_timeout_seconds = float(default_timeout_seconds)
        self._handlers: dict[str, _ModuleHandler] = {}
        self._queue: queue.Queue[tuple[str, WorkspaceEvent]] = queue.Queue()
        self._subscriptions: dict[str, Any] = {}
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ore-workspace-module")
        self._closed = False

    def register_module(
        self,
        workspace,
        module_id: str,
        callback: Callable[[WorkspaceSnapshot], Any],
        fields: Iterable[str],
        *,
        sensitivity: Mapping[str, float] | None = None,
        threshold: float = 0.0,
        timeout_seconds: float | None = None,
    ) -> None:
        """Register a processor, its event subscriptions and its relevance sensitivity."""
        if not callable(callback):
            raise TypeError("module callback must be callable")
        timeout = self.default_timeout_seconds if timeout_seconds is None else float(timeout_seconds)
        if timeout <= 0:
            raise ValueError("timeout_seconds must be positive")
        if module_id in self._subscriptions:
            self.unregister_module(module_id)
        workspace.subscriptions.register(
            module_id, fields, sensitivity=sensitivity, threshold=threshold,
        )
        self._handlers[module_id] = _ModuleHandler(callback, timeout)

        def enqueue(event: WorkspaceEvent, selected_module: str = module_id) -> None:
            """Queue the event for the module whose subscription caused this callback."""
            self._queue.put((selected_module, event))

        workspace.event_bus.subscribe(module_id, enqueue)
        self._subscriptions[module_id] = workspace

    def unregister_module(self, module_id: str) -> bool:
        """Remove a handler and its notification callback from the associated workspace."""
        handler = self._handlers.pop(module_id, None)
        workspace = self._subscriptions.pop(module_id, None)
        if workspace is not None:
            workspace.event_bus.unsubscribe(module_id)
            workspace.subscriptions.unregister(module_id)
        return handler is not None

    def _invoke(self, callback: Callable[[WorkspaceSnapshot], Any], snapshot: WorkspaceSnapshot) -> Any:
        """Run synchronous or asynchronous module handlers within a worker thread."""
        result = callback(snapshot)
        if inspect.isawaitable(result):
            return asyncio.run(result)
        return result

    def run_round(
        self,
        workspace,
        changed_fields: Iterable[str],
        deltas: Mapping[str, float] | None = None,
    ) -> tuple[tuple[ModuleActivationResult, ...], tuple[Any, ...]]:
        """Run one parallel activation round for subscribed modules meeting their relevance threshold."""
        if self._closed:
            raise RuntimeError("scheduler is closed")
        fields = tuple(changed_fields)
        magnitudes = dict(deltas or {field: 1.0 for field in fields})
        eligible = workspace.subscriptions.eligible(magnitudes)
        snapshot = workspace.snapshot()
        all_selected = [(module, score) for module, score in eligible if module in self._handlers]
        selected = all_selected[:self.max_activations]
        futures = {
            self._executor.submit(self._invoke, self._handlers[module].callback, snapshot): (module, score)
            for module, score in selected
        }
        activations: list[ModuleActivationResult] = []
        publications: list[Any] = []
        for future in tuple(futures):
            module_id, score = futures[future]
            timeout = self._handlers[module_id].timeout_seconds
            try:
                output = future.result(timeout=timeout)
                activations.append(ModuleActivationResult(
                    module_id, "manual_round", score,
                    workspace.subscriptions.get(module_id).threshold, True, "module completed",
                ))
                for finding in self._normalize_findings(output):
                    if finding.producer_module != module_id:
                        raise ValueError("module handler returned a finding owned by a different producer")
                    publications.append(workspace.publish(finding))
            except TimeoutError:
                future.cancel()
                activations.append(ModuleActivationResult(
                    module_id, "manual_round", score,
                    workspace.subscriptions.get(module_id).threshold, False, "module timed out",
                ))
            except Exception as exc:
                activations.append(ModuleActivationResult(
                    module_id, "manual_round", score,
                    workspace.subscriptions.get(module_id).threshold, False,
                    f"{type(exc).__name__}: {exc}",
                ))
        for module_id, score in all_selected[self.max_activations:]:
            subscription = workspace.subscriptions.get(module_id)
            activations.append(ModuleActivationResult(
                module_id, "manual_round", score,
                subscription.threshold if subscription else 0.0,
                False, "activation budget exhausted",
            ))
        return tuple(activations), tuple(publications)

    def _normalize_findings(self, output: Any) -> tuple[Finding, ...]:
        """Normalize a module callback result into zero or more typed finding objects."""
        if output is None:
            return ()
        if isinstance(output, Finding):
            return (output,)
        if isinstance(output, (str, bytes, Mapping)):
            raise TypeError("module handlers must return Finding objects, an iterable of Findings, or None")
        try:
            findings = tuple(output)
        except TypeError as exc:
            raise TypeError("module handler returned an unsupported result") from exc
        if any(not isinstance(item, Finding) for item in findings):
            raise TypeError("every module handler output must be a Finding")
        return findings

    def run_until_idle(self, workspace, *, max_rounds: int | None = None) -> SchedulerReport:
        """Consume queued notification tasks until idle or a configured processing budget is reached."""
        round_limit = self.max_rounds if max_rounds is None else max_rounds
        if round_limit <= 0:
            raise ValueError("max_rounds must be positive")
        activations: list[ModuleActivationResult] = []
        publications: list[Any] = []
        seen: set[tuple[str, str]] = set()
        rounds = 0
        while rounds < min(round_limit, self.max_rounds) and len(activations) < self.max_activations:
            batch: list[tuple[str, WorkspaceEvent]] = []
            while True:
                try:
                    item = self._queue.get_nowait()
                except queue.Empty:
                    break
                identity = (item[0], item[1].event_id)
                if identity not in seen:
                    seen.add(identity)
                    batch.append(item)
            if not batch:
                break
            rounds += 1
            grouped: dict[str, list[WorkspaceEvent]] = {}
            for module_id, event in batch:
                grouped.setdefault(module_id, []).append(event)
            for module_id, events in grouped.items():
                if len(activations) >= self.max_activations:
                    break
                handler = self._handlers.get(module_id)
                if handler is None:
                    continue
                event = events[-1]
                deltas = {field: 1.0 for field in event.changed_fields}
                score = workspace.subscriptions.relevance_score(module_id, deltas)
                subscription = workspace.subscriptions.get(module_id)
                threshold = subscription.threshold if subscription else 0.0
                if score < threshold or score <= 0:
                    activations.append(ModuleActivationResult(
                        module_id, event.event_id, score, threshold, False,
                        "change relevance below activation threshold",
                    ))
                    continue
                snapshot = workspace.snapshot()
                future = self._executor.submit(self._invoke, handler.callback, snapshot)
                try:
                    output = future.result(timeout=handler.timeout_seconds)
                    activations.append(ModuleActivationResult(
                        module_id, event.event_id, score, threshold, True, "module completed",
                    ))
                    for finding in self._normalize_findings(output):
                        if finding.producer_module != module_id:
                            raise ValueError("module handler returned a finding owned by a different producer")
                        publications.append(workspace.publish(finding))
                except TimeoutError:
                    future.cancel()
                    activations.append(ModuleActivationResult(
                        module_id, event.event_id, score, threshold, False, "module timed out",
                    ))
                except Exception as exc:
                    activations.append(ModuleActivationResult(
                        module_id, event.event_id, score, threshold, False,
                        f"{type(exc).__name__}: {exc}",
                    ))
        stopped = not self._queue.empty() and (
            rounds >= min(round_limit, self.max_rounds) or len(activations) >= self.max_activations
        )
        return SchedulerReport(rounds, tuple(activations), tuple(publications), stopped)

    def close(self, wait_for_tasks: bool = True) -> None:
        """Stop future scheduling and close the bounded worker pool."""
        if not self._closed:
            self._executor.shutdown(wait=wait_for_tasks, cancel_futures=True)
            self._closed = True

    def __enter__(self) -> "ModuleScheduler":
        """Return this scheduler for context-managed resource handling."""
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        """Shut down module workers while allowing exceptions to propagate."""
        self.close(wait_for_tasks=True)
        return False