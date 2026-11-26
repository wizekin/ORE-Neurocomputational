"""Component lifecycle state machine and supervised recovery (steps 8-9, 35).

States: CREATED, VALIDATED, AUTHORIZED, READY, RUNNING, QUIESCING, STOPPED,
FAILED, QUARANTINED. Transitions are declared in one auditable table;
illegal transitions raise :class:`IllegalTransitionError` and terminal states
never silently return to RUNNING -- revival requires a new authorized
generation (integrated step 8). The supervisor bounds restart attempts with
exponential backoff and jitter, ``delay_k = min(delay_max, delay_base * 2^k)
+ jitter`` (section 5.5), and moves repeatedly failing components to
QUARANTINED while preserving diagnostic evidence (steps 9 and 35).
"""
from __future__ import annotations

import random
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from .contracts import IllegalTransitionError

TERMINAL_STATES = ("stopped", "failed", "quarantined")


class LifecycleState:
    """Lifecycle state constants (kept a plain namespace for table clarity)."""

    CREATED = "created"
    VALIDATED = "validated"
    AUTHORIZED = "authorized"
    READY = "ready"
    RUNNING = "running"
    QUIESCING = "quiescing"
    STOPPED = "stopped"
    FAILED = "failed"
    QUARANTINED = "quarantined"

    ALL: tuple[str, ...] = (
        CREATED,
        VALIDATED,
        AUTHORIZED,
        READY,
        RUNNING,
        QUIESCING,
        STOPPED,
        FAILED,
        QUARANTINED,
    )
    TERMINAL: tuple[str, ...] = TERMINAL_STATES


#: Explicit transition table (integrated step 8). Terminal revival rows only
#: become reachable when the caller presents a new authorized generation.
ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    LifecycleState.CREATED: frozenset({LifecycleState.VALIDATED, LifecycleState.FAILED}),
    LifecycleState.VALIDATED: frozenset(
        {LifecycleState.AUTHORIZED, LifecycleState.FAILED, LifecycleState.STOPPED}
    ),
    LifecycleState.AUTHORIZED: frozenset(
        {LifecycleState.READY, LifecycleState.FAILED, LifecycleState.STOPPED, LifecycleState.QUARANTINED}
    ),
    LifecycleState.READY: frozenset(
        {LifecycleState.RUNNING, LifecycleState.FAILED, LifecycleState.STOPPED, LifecycleState.QUARANTINED}
    ),
    LifecycleState.RUNNING: frozenset(
        {
            LifecycleState.QUIESCING,
            LifecycleState.STOPPED,
            LifecycleState.FAILED,
            LifecycleState.QUARANTINED,
        }
    ),
    LifecycleState.QUIESCING: frozenset(
        {LifecycleState.STOPPED, LifecycleState.FAILED, LifecycleState.QUARANTINED}
    ),
    LifecycleState.STOPPED: frozenset({LifecycleState.READY, LifecycleState.QUARANTINED}),
    LifecycleState.FAILED: frozenset({LifecycleState.READY, LifecycleState.QUARANTINED}),
    LifecycleState.QUARANTINED: frozenset({LifecycleState.READY}),
}


@dataclass(frozen=True)
class TransitionRecord:
    """Auditable record of one lifecycle transition (step 8)."""

    component_id: str
    from_state: str
    to_state: str
    event: str
    actor: str
    reason: str
    policy_version: int
    generation: int
    timestamp_epoch: float
    monotonic: float
    transition_id: str = field(default_factory=lambda: f"tr-{uuid.uuid4().hex}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "transition_id": self.transition_id,
            "component_id": self.component_id,
            "from_state": self.from_state,
            "to_state": self.to_state,
            "event": self.event,
            "actor": self.actor,
            "reason": self.reason,
            "policy_version": self.policy_version,
            "generation": self.generation,
            "timestamp_epoch": self.timestamp_epoch,
            "monotonic": self.monotonic,
        }


def allowed_transitions(state: str) -> frozenset[str]:
    """Return the declared successor set for a state (model-checking helper)."""
    if state not in ALLOWED_TRANSITIONS:
        raise KeyError(f"unknown lifecycle state {state!r}")
    return ALLOWED_TRANSITIONS[state]


def compute_backoff(
    attempt: int,
    *,
    base: float = 0.25,
    cap: float = 30.0,
    jitter: float = 0.0,
    rng: random.Random | None = None,
) -> float:
    """Bounded exponential backoff with jitter (section 5.5)."""
    if attempt < 0:
        raise ValueError("attempt must be non-negative")
    if base <= 0 or cap <= 0 or jitter < 0:
        raise ValueError("base/cap must be positive and jitter non-negative")
    delay = min(cap, base * (2**attempt))
    jitter_term = rng.uniform(0.0, jitter) if rng is not None else jitter
    return min(cap, delay + jitter_term)


class LifecycleMachine:
    """Tracks per-component lifecycle states and validates every transition."""

    def __init__(self, clock=None) -> None:
        self._clock = clock or time.time
        self._states: dict[str, str] = {}
        self._generations: dict[str, int] = {}
        self._history: dict[str, list[TransitionRecord]] = {}
        self._lock = threading.RLock()

    # -- queries -------------------------------------------------------------

    def state(self, component_id: str) -> str | None:
        with self._lock:
            return self._states.get(component_id)

    def generation(self, component_id: str) -> int:
        with self._lock:
            return self._generations.get(component_id, 0)

    def history(self, component_id: str) -> tuple[TransitionRecord, ...]:
        with self._lock:
            return tuple(self._history.get(component_id, ()))

    def describe(self) -> dict[str, str]:
        with self._lock:
            return dict(self._states)

    # -- transitions ----------------------------------------------------------

    def apply(
        self,
        component_id: str,
        to_state: str,
        *,
        actor: str = "system",
        event: str = "transition",
        reason: str = "",
        policy_version: int = 1,
        new_generation: bool = False,
    ) -> TransitionRecord:
        """Apply one transition or raise :class:`IllegalTransitionError`."""
        if to_state not in ALLOWED_TRANSITIONS:
            raise IllegalTransitionError(
                f"unknown target state {to_state!r}",
                from_state=None,
                to_state=to_state,
            )
        with self._lock:
            from_state = self._states.get(component_id, LifecycleState.CREATED)
            if to_state not in ALLOWED_TRANSITIONS[from_state]:
                raise IllegalTransitionError(
                    f"component {component_id}: transition {from_state} -> {to_state} "
                    f"is not authorized; allowed: {sorted(ALLOWED_TRANSITIONS[from_state])}",
                    from_state=from_state,
                    to_state=to_state,
                )
            if from_state in TERMINAL_STATES and not new_generation:
                raise IllegalTransitionError(
                    f"component {component_id}: terminal state {from_state} requires a "
                    f"new authorized generation before revival (got none)",
                    from_state=from_state,
                    to_state=to_state,
                )
            generation = self._generations.get(component_id, 0)
            if new_generation:
                generation += 1
                self._generations[component_id] = generation
            record = TransitionRecord(
                component_id=component_id,
                from_state=from_state,
                to_state=to_state,
                event=event,
                actor=actor,
                reason=reason,
                policy_version=policy_version,
                generation=generation,
                timestamp_epoch=self._clock(),
                monotonic=time.monotonic(),
            )
            self._states[component_id] = to_state
            self._history.setdefault(component_id, []).append(record)
            return record

    def force_state(self, component_id: str, state: str, *, reason: str = "test setup") -> None:
        """Move a component directly to a state (fault injection only)."""
        if state not in ALLOWED_TRANSITIONS:
            raise ValueError(f"unknown lifecycle state {state!r}")
        with self._lock:
            self._states[component_id] = state
            self._history.setdefault(component_id, [])
            self._history[component_id].append(
                TransitionRecord(
                    component_id=component_id,
                    from_state="<forced>",
                    to_state=state,
                    event="force_state",
                    actor="fault_injection",
                    reason=reason,
                    policy_version=0,
                    generation=self._generations.get(component_id, 0),
                    timestamp_epoch=self._clock(),
                    monotonic=time.monotonic(),
                )
            )


class SupervisorDecision:
    """Kinds of supervised recovery outcomes."""

    RETRY = "retry"
    QUARANTINE = "quarantine"
    GIVE_UP = "give_up"


@dataclass(frozen=True)
class SupervisorReport:
    """Typed supervisor decision for one failure report (steps 9 and 35)."""

    component_id: str
    decision: str
    attempt: int
    delay_seconds: float
    reason: str
    restart_limit: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "decision": self.decision,
            "attempt": self.attempt,
            "delay_seconds": self.delay_seconds,
            "reason": self.reason,
            "restart_limit": self.restart_limit,
        }


@dataclass(frozen=True)
class RestartPolicy:
    """Bounded restart behaviour (integrated step 9)."""

    restart_limit: int = 3
    backoff_base_seconds: float = 0.25
    backoff_max_seconds: float = 30.0
    jitter_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.restart_limit < 1:
            raise ValueError("restart_limit must be at least 1")
        if self.backoff_base_seconds <= 0 or self.backoff_max_seconds <= 0:
            raise ValueError("backoff bounds must be positive")
        if self.jitter_seconds < 0:
            raise ValueError("jitter_seconds must be non-negative")

    def backoff(self, attempt: int, rng: random.Random | None = None) -> float:
        return compute_backoff(
            attempt,
            base=self.backoff_base_seconds,
            cap=self.backoff_max_seconds,
            jitter=self.jitter_seconds,
            rng=rng,
        )


class Supervisor:
    """Bounded failure recovery with backoff and quarantine (steps 9, 35)."""

    def __init__(
        self,
        policy: RestartPolicy | None = None,
        *,
        on_quarantine: Callable[[str, str], None] | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.policy = policy or RestartPolicy()
        self._attempts: dict[str, int] = {}
        self._quarantined: set[str] = set()
        self._on_quarantine = on_quarantine
        self._rng = rng

    def attempts(self, component_id: str) -> int:
        return self._attempts.get(component_id, 0)

    def is_quarantined(self, component_id: str) -> bool:
        return component_id in self._quarantined

    def report_success(self, component_id: str) -> None:
        """A successful run resets the backoff ladder for the component."""
        self._attempts[component_id] = 0

    def report_failure(self, component_id: str, *, reason: str = "worker failure") -> SupervisorReport:
        """Classify one failure: bounded retry, then quarantine (step 35)."""
        if component_id in self._quarantined:
            return SupervisorReport(
                component_id=component_id,
                decision=SupervisorDecision.QUARANTINE,
                attempt=self.attempts(component_id),
                delay_seconds=0.0,
                reason="component already quarantined",
                restart_limit=self.policy.restart_limit,
            )
        attempt = self._attempts.get(component_id, 0) + 1
        self._attempts[component_id] = attempt
        if attempt >= self.policy.restart_limit:
            self._quarantined.add(component_id)
            report = SupervisorReport(
                component_id=component_id,
                decision=SupervisorDecision.QUARANTINE,
                attempt=attempt,
                delay_seconds=0.0,
                reason=(
                    f"restart budget exhausted after {attempt} attempts "
                    f"({reason}); quarantined and coordinator alerted"
                ),
                restart_limit=self.policy.restart_limit,
            )
            if self._on_quarantine is not None:
                self._on_quarantine(component_id, report.reason)
            return report
        delay = self.policy.backoff(attempt, self._rng)
        return SupervisorReport(
            component_id=component_id,
            decision=SupervisorDecision.RETRY,
            attempt=attempt,
            delay_seconds=delay,
            reason=f"retryable failure ({reason}); bounded restart {attempt}/"
            f"{self.policy.restart_limit} after {delay:.3f}s backoff",
            restart_limit=self.policy.restart_limit,
        )

    def release_quarantine(self, component_id: str) -> bool:
        """Operators may release a quarantined component for a new generation."""
        if component_id not in self._quarantined:
            return False
        self._quarantined.discard(component_id)
        self._attempts[component_id] = 0
        return True

    def describe(self) -> dict[str, Any]:
        return {
            "restart_limit": self.policy.restart_limit,
            "attempts": dict(sorted(self._attempts.items())),
            "quarantined": sorted(self._quarantined),
        }
