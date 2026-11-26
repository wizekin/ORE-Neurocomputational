"""Watchdog, rollback and deterministic fallback (ORE Point 23, Step 52).

Every adaptive policy must define activation thresholds, watchdogs,
rollback triggers, known-good settings and a safe fallback. Adaptive
behaviour can never permanently displace the deterministic baseline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Sequence

from ..actions import RuntimeControlState
from ..config import AdaptationLimits


@dataclass(frozen=True)
class WatchdogVerdict:
    """Outcome of one watchdog evaluation."""

    triggered: bool
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible snapshot of the verdict."""
        return {"triggered": self.triggered, "reasons": list(self.reasons)}


class HysteresisController:
    """Threshold transitions with dwell time to avoid oscillation (spec 3.5).

    The controller engages only above ``on_threshold`` and disengages only
    below ``off_threshold``; both transitions additionally respect the
    minimum dwell time, so noisy measurements cannot toggle policy quickly.
    """

    def __init__(self, limits: AdaptationLimits, *, on_threshold: float, off_threshold: float):
        """Configure thresholds and shared stability limits."""
        if not 0.0 <= off_threshold < on_threshold:
            raise ValueError("thresholds must satisfy 0 <= off_threshold < on_threshold")
        self.limits = limits
        self.on_threshold = on_threshold
        self.off_threshold = off_threshold
        self.engaged = False
        self._last_transition = -math.inf

    def update(self, value: float, *, now: float) -> bool:
        """Return the engagement state after observing one value."""
        if not math.isfinite(value):
            raise ValueError("value must be finite")
        if now < self._last_transition:
            raise ValueError("time cannot move backwards")
        dwell_elapsed = (now - self._last_transition) >= self.limits.min_dwell_seconds
        if not self.engaged and value >= self.on_threshold and dwell_elapsed:
            self.engaged = True
            self._last_transition = now
        elif self.engaged and value <= self.off_threshold and dwell_elapsed:
            self.engaged = False
            self._last_transition = now
        return self.engaged


class OscillationDetector:
    """Detect rapid policy reversal within a bounded window (spec section 5)."""

    def __init__(self, limits: AdaptationLimits) -> None:
        """Configure the reversal window and limit from adaptation limits."""
        self.limits = limits
        self._history: list[tuple[float, int]] = []

    def record(self, signature: int, *, now: float) -> bool:
        """Record one action signature and return True when oscillating."""
        self._history.append((now, signature))
        cutoff = now - self.limits.oscillation_window
        self._history = [item for item in self._history if item[0] >= cutoff]
        reversals = 0
        for (_, previous), (_, current) in zip(self._history, self._history[1:]):
            if previous != current:
                reversals += 1
        return reversals > self.limits.oscillation_reversal_limit

    def reversal_count(self) -> int:
        """Return the number of direction changes in the current window."""
        count = 0
        for (_, previous), (_, current) in zip(self._history, self._history[1:]):
            if previous != current:
                count += 1
        return count


class PidController:
    """Bounded PID controller with anti-windup and rate limiting (spec 3.5).

    Outputs are clamped and the integral term is frozen whenever the output
    saturates, preventing windup; a rate limit bounds how fast the control
    signal may change between updates.
    """

    def __init__(
        self, *, kp: float, ki: float = 0.0, kd: float = 0.0,
        output_min: float = -1.0, output_max: float = 1.0,
        rate_limit: float | None = None,
    ) -> None:
        """Configure gains and bounds; all gains must be non-negative."""
        for name, value in (("kp", kp), ("ki", ki), ("kd", kd)):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if output_min >= output_max:
            raise ValueError("output_min must be below output_max")
        if rate_limit is not None and rate_limit <= 0.0:
            raise ValueError("rate_limit must be positive when provided")
        self.kp, self.ki, self.kd = kp, ki, kd
        self.output_min, self.output_max = output_min, output_max
        self.rate_limit = rate_limit
        self._integral = 0.0
        self._previous_error: float | None = None
        self._previous_output = 0.0

    def update(self, error: float, *, delta_time: float) -> float:
        """Return the bounded control signal for one error sample."""
        if not math.isfinite(error):
            raise ValueError("error must be finite")
        if not math.isfinite(delta_time) or delta_time <= 0.0:
            raise ValueError("delta_time must be finite and positive")
        proportional = self.kp * error
        candidate_integral = self._integral + self.ki * error * delta_time
        derivative = 0.0
        if self._previous_error is not None:
            derivative = self.kd * (error - self._previous_error) / delta_time
        raw = proportional + candidate_integral + derivative
        output = max(self.output_min, min(self.output_max, raw))
        if output == raw:
            self._integral = candidate_integral
        if self.rate_limit is not None:
            delta = output - self._previous_output
            bound = self.rate_limit
            if delta > bound:
                output = self._previous_output + bound
            elif delta < -bound:
                output = self._previous_output - bound
            output = max(self.output_min, min(self.output_max, output))
        self._previous_error = error
        self._previous_output = output
        return output

    def reset(self) -> None:
        """Clear integral and derivative memory."""
        self._integral = 0.0
        self._previous_error = None
        self._previous_output = 0.0


class Watchdog:
    """Aggregate watchdog for adaptive control (spec Step 52).

    Triggers on invariant failure, excessive prediction error, tail-latency
    regression, oscillation or execution timeout. After a trigger the
    watchdog enforces a cooldown before adaptive control may re-engage.
    """

    def __init__(self, limits: AdaptationLimits) -> None:
        """Configure thresholds from the adaptation limits."""
        self.limits = limits
        self._last_trigger = -math.inf
        self._trigger_count = 0

    def check(
        self,
        *,
        invariant_violations: Sequence[str] = (),
        prediction_error: float | None = None,
        tail_latency_regression: float | None = None,
        oscillating: bool = False,
        execution_timeout: bool = False,
        now: float = 0.0,
    ) -> WatchdogVerdict:
        """Evaluate every watchdog condition and return the verdict."""
        reasons: list[str] = []
        if invariant_violations:
            reasons.append(f"invariant violations: {list(invariant_violations)}")
        error_limit = self.limits.prediction_error_threshold
        if prediction_error is not None and prediction_error > error_limit:
            reasons.append(
                f"prediction error {prediction_error:.3f} exceeds "
                f"{self.limits.prediction_error_threshold}"
            )
        if tail_latency_regression is not None and (
            tail_latency_regression > self.limits.tail_latency_regression_threshold
        ):
            reasons.append(
                f"tail latency regression {tail_latency_regression:.3f} exceeds "
                f"{self.limits.tail_latency_regression_threshold}"
            )
        if oscillating:
            reasons.append("policy oscillation detected")
        if execution_timeout:
            reasons.append("execution timeout")
        triggered = bool(reasons)
        if triggered:
            self._last_trigger = now
            self._trigger_count += 1
        return WatchdogVerdict(triggered=triggered, reasons=tuple(reasons))

    def in_cooldown(self, *, now: float) -> bool:
        """Return True while adaptive control must stay disabled after a trigger."""
        return (now - self._last_trigger) < self.limits.rollback_cooldown_seconds

    def note_trigger(self, *, now: float) -> None:
        """Record an external trigger so the cooldown clock starts."""
        self._last_trigger = now
        self._trigger_count += 1

    @property
    def trigger_count(self) -> int:
        """Return the number of watchdog triggers observed so far."""
        return self._trigger_count


class RollbackManager:
    """Checkpoint and restore known-good control state (spec Step 52)."""

    def __init__(self) -> None:
        """Initialize without checkpoints or a registered baseline."""
        self._checkpoints: dict[str, RuntimeControlState] = {}
        self._baseline: RuntimeControlState | None = None
        self._history: list[dict[str, Any]] = []

    def register_baseline(self, state: RuntimeControlState) -> None:
        """Register the known-good baseline used by deterministic fallback."""
        self._baseline = state.snapshot()

    def baseline(self) -> RuntimeControlState:
        """Return a snapshot of the registered baseline."""
        if self._baseline is None:
            raise RuntimeError("no baseline registered; call register_baseline first")
        return self._baseline.snapshot()

    def checkpoint(self, token: str, state: RuntimeControlState) -> None:
        """Store one named checkpoint of the current control state."""
        if not isinstance(token, str) or not token.strip():
            raise ValueError("token must be a non-empty string")
        self._checkpoints[token] = state.snapshot()

    def rollback(self, token: str) -> RuntimeControlState:
        """Restore a named checkpoint; raises when the token is unknown."""
        if token not in self._checkpoints:
            raise KeyError(f"unknown rollback token {token!r}")
        restored = self._checkpoints[token].snapshot()
        self._history.append({"action": "rollback", "token": token})
        return restored

    def activate_baseline(self) -> RuntimeControlState:
        """Restore the registered baseline (deterministic fallback)."""
        restored = self.baseline()
        self._history.append({"action": "activate_baseline"})
        return restored

    def history(self) -> tuple[dict[str, Any], ...]:
        """Return the rollback audit history."""
        return tuple(dict(item) for item in self._history)


class SafeFallbackPolicy:
    """Compose watchdog, rollback and cooldown into one safe-fallback rule."""

    def __init__(self, limits: AdaptationLimits) -> None:
        """Create the watchdog and rollback manager from adaptation limits."""
        self.limits = limits
        self.watchdog = Watchdog(limits)
        self.rollback = RollbackManager()
        self.adaptive_enabled = True

    def evaluate(self, verdict: WatchdogVerdict, *, now: float) -> bool:
        """Process one watchdog verdict, disabling adaptation when triggered."""
        if verdict.triggered:
            self.adaptive_enabled = False
            self.rollback.activate_baseline()
            self.watchdog.note_trigger(now=now)
        elif self.watchdog.in_cooldown(now=now):
            self.adaptive_enabled = False
        else:
            self.adaptive_enabled = True
        return self.adaptive_enabled


__all__ = [
    "HysteresisController", "OscillationDetector", "PidController", "Watchdog",
    "WatchdogVerdict", "RollbackManager", "SafeFallbackPolicy",
]
