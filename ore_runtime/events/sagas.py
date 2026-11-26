"""Sagas with ordered local transactions and explicit compensating actions."""
from __future__ import annotations

import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..contracts import new_id, utc_now
from ..exceptions import SagaCompensationIncompleteError


@dataclass(frozen=True, slots=True)
class SagaStep:
    """Declare one local transaction with its optional compensating action."""

    name: str
    action: Callable[[Mapping[str, Any]], Any]
    compensate: Callable[[Mapping[str, Any], Any], Any] | None = None
    irreversible: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate step identity and detach metadata."""
        if not self.name.strip():
            raise ValueError("step name must be non-empty.")
        if not callable(self.action):
            raise TypeError("action must be callable.")
        if self.compensate is not None and not callable(self.compensate):
            raise TypeError("compensate must be callable when provided.")
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class SagaStepOutcome:
    """Record one executed saga step with its result or error."""

    step_index: int
    step_name: str
    succeeded: bool
    result: Any = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class SagaOutcome:
    """Summarize one saga run including compensation and residual effects."""

    saga_id: str
    completed: bool
    executed_steps: tuple[SagaStepOutcome, ...]
    compensated_steps: tuple[str, ...]
    incomplete_compensations: tuple[str, ...]
    residual_effects: tuple[str, ...]
    failed_step: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable saga outcome record."""
        return {
            "saga_id": self.saga_id,
            "completed": self.completed,
            "executed_steps": [
                {
                    "step_index": step.step_index,
                    "step_name": step.step_name,
                    "succeeded": step.succeeded,
                    "error": step.error,
                }
                for step in self.executed_steps
            ],
            "compensated_steps": list(self.compensated_steps),
            "incomplete_compensations": list(self.incomplete_compensations),
            "residual_effects": list(self.residual_effects),
            "failed_step": self.failed_step,
            "error": self.error,
        }


@dataclass(slots=True)
class SagaState:
    """Persist one saga's position and step outcomes for recovery."""

    saga_id: str
    steps: tuple[SagaStep, ...]
    next_step: int = 0
    outcomes: list[SagaStepOutcome] = field(default_factory=list)
    started_at: str = field(default_factory=utc_now)
    finished_at: str | None = None
    status: str = "running"


class SagaCoordinator:
    """Run multi-step side-effectful workflows with compensation and audit logs.

    Compensation is modeled as a new auditable action, never as erasure of
    history: every executed step and every compensation is recorded in the
    saga log, and steps marked irreversible leave residual effects that
    require human review instead of a silent undo.
    """

    def __init__(self, on_step: Callable[[str, SagaStepOutcome], None] | None = None,
                 on_compensation: Callable[[str, str], None] | None = None) -> None:
        """Initialize the coordinator with optional audit event callbacks."""
        self._lock = threading.RLock()
        self._logs: dict[str, SagaState] = {}
        self._on_step = on_step
        self._on_compensation = on_compensation

    # ------------------------------------------------------------------- run

    def start(self, saga_id: str, steps: Sequence[SagaStep]) -> SagaState:
        """Create and persist a new saga state without executing any step."""
        if not saga_id.strip():
            raise ValueError("saga_id must be non-empty.")
        if not steps:
            raise ValueError("a saga needs at least one step.")
        names = [step.name for step in steps]
        if len(set(names)) != len(names):
            raise ValueError("saga step names must be unique.")
        with self._lock:
            if saga_id in self._logs:
                raise ValueError(f"saga '{saga_id}' already exists.")
            state = SagaState(saga_id=saga_id, steps=tuple(steps))
            self._logs[saga_id] = state
            return state

    def run(self, saga_id: str, steps: Sequence[SagaStep],
            context: Mapping[str, Any] | None = None) -> SagaOutcome:
        """Execute a saga in order, compensating in reverse when a step fails."""
        state = self.start(saga_id, steps)
        return self.resume(saga_id, context or {})

    def resume(self, saga_id: str, context: Mapping[str, Any]) -> SagaOutcome:
        """Resume a persisted saga from its next unfinished step."""
        with self._lock:
            state = self._logs.get(saga_id)
            if state is None:
                raise KeyError(f"unknown saga '{saga_id}'.")
        local_context = dict(context)
        while state.next_step < len(state.steps):
            step = state.steps[state.next_step]
            try:
                result = step.action(local_context)
            except Exception as exc:  # noqa: BLE001 - saga isolation
                outcome = SagaStepOutcome(
                    step_index=state.next_step, step_name=step.name,
                    succeeded=False, error=f"{type(exc).__name__}: {exc}",
                )
                state.outcomes.append(outcome)
                state.status = "compensating"
                self._notify_step(saga_id, outcome)
                return self._compensate(saga_id, state, local_context,
                                         failed_step=step.name,
                                         error=str(exc))
            outcome = SagaStepOutcome(
                step_index=state.next_step, step_name=step.name,
                succeeded=True, result=result,
            )
            state.outcomes.append(outcome)
            state.next_step += 1
            self._notify_step(saga_id, outcome)
        state.status = "completed"
        state.finished_at = utc_now()
        return SagaOutcome(
            saga_id=saga_id, completed=True,
            executed_steps=tuple(state.outcomes),
            compensated_steps=(), incomplete_compensations=(),
            residual_effects=(),
        )

    def _compensate(self, saga_id: str, state: SagaState,
                    context: Mapping[str, Any], *,
                    failed_step: str, error: str) -> SagaOutcome:
        """Run compensating actions in reverse for completed steps."""
        compensated: list[str] = []
        incomplete: list[str] = []
        residuals: list[str] = []
        for outcome in reversed(
                [step for step in state.outcomes if step.succeeded]):
            step = state.steps[outcome.step_index]
            if step.irreversible:
                incomplete.append(step.name)
                residuals.append(
                    f"step '{step.name}' is irreversible; manual review required"
                )
                continue
            if step.compensate is None:
                incomplete.append(step.name)
                residuals.append(f"step '{step.name}' has no compensating action")
                continue
            try:
                step.compensate(context, outcome.result)
                compensated.append(step.name)
                if self._on_compensation is not None:
                    self._on_compensation(saga_id, step.name)
            except Exception as exc:  # noqa: BLE001 - compensation isolation
                incomplete.append(step.name)
                residuals.append(
                    f"compensation for '{step.name}' failed: "
                    f"{type(exc).__name__}: {exc}"
                )
        state.status = "compensated"
        state.finished_at = utc_now()
        if incomplete:
            raise SagaCompensationIncompleteError(
                f"saga '{saga_id}' compensation incomplete for steps: "
                + ", ".join(sorted(incomplete))
                + f"; residual effects recorded: {'; '.join(residuals)}",
            )
        return SagaOutcome(
            saga_id=saga_id, completed=False,
            executed_steps=tuple(state.outcomes),
            compensated_steps=tuple(compensated),
            incomplete_compensations=(),
            residual_effects=tuple(residuals),
            failed_step=failed_step,
            error=error,
        )

    def _notify_step(self, saga_id: str, outcome: SagaStepOutcome) -> None:
        """Forward one step outcome to the audit callback when registered."""
        if self._on_step is not None:
            self._on_step(saga_id, outcome)

    # ---------------------------------------------------------------- queries

    def saga_state(self, saga_id: str) -> dict[str, Any]:
        """Return one persisted saga's position for recovery planning."""
        with self._lock:
            state = self._logs.get(saga_id)
            if state is None:
                raise KeyError(f"unknown saga '{saga_id}'.")
            return {
                "saga_id": state.saga_id,
                "next_step": state.next_step,
                "status": state.status,
                "started_at": state.started_at,
                "finished_at": state.finished_at,
                "steps": [
                    {
                        "step_index": index,
                        "step_name": step.name,
                        "compensatable": step.compensate is not None,
                        "irreversible": step.irreversible,
                    }
                    for index, step in enumerate(state.steps)
                ],
                "outcomes": [
                    {
                        "step_index": outcome.step_index,
                        "step_name": outcome.step_name,
                        "succeeded": outcome.succeeded,
                        "error": outcome.error,
                    }
                    for outcome in state.outcomes
                ],
            }

    def known_sagas(self) -> tuple[str, ...]:
        """Return every persisted saga identifier."""
        with self._lock:
            return tuple(sorted(self._logs))

    def stats(self) -> dict[str, Any]:
        """Return saga counts by status for observability."""
        with self._lock:
            statuses: dict[str, int] = {}
            for state in self._logs.values():
                statuses[state.status] = statuses.get(state.status, 0) + 1
            return {"sagas": len(self._logs), "statuses": statuses}


def build_compensatable_workflow(
    steps: Sequence[tuple[str, Callable[[Mapping[str, Any]], Any],
                        Callable[[Mapping[str, Any], Any], Any]]],
) -> tuple[SagaStep, ...]:
    """Build compensatable saga steps from (name, action, compensation) triples."""
    return tuple(
        SagaStep(name=name, action=action, compensate=compensation)
        for name, action, compensation in steps
    )
