"""Numerical health checks that invalidate output instead of clipping (Step 25)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

__all__ = ("HealthReport", "numerical_health_check")


@dataclass(frozen=True, slots=True)
class HealthReport:
    """The outcome of one numerical health check."""

    valid: bool
    failures: tuple[str, ...]
    checks: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the health report."""
        return {"valid": self.valid, "failures": list(self.failures), "checks": self.checks}


def numerical_health_check(state: Sequence[float] | None,
                           prediction: Sequence[float] | None, *,
                           state_norm_bound: float = 1e6,
                           derivative_bound: float = 1e4,
                           output_bounds: tuple[float, float] | None = None,
                           spike_rate: float | None = None,
                           max_spike_rate: float = 1e6,
                           domain_invariants: dict[str, tuple[float, float]] | None = None,
                           ) -> HealthReport:
    """Check finite values, state norms, output bounds, spike rates, and invariants.

    A failed check invalidates the output; values are never silently clipped.
    """
    failures: list[str] = []
    checks: dict[str, Any] = {}
    if state is not None:
        state_array = np.asarray(state, dtype=float).reshape(-1)
        state_finite = bool(np.all(np.isfinite(state_array)))
        state_norm = float(np.linalg.norm(state_array)) if state_array.size else 0.0
        checks["state_finite"] = state_finite
        checks["state_norm"] = state_norm
        checks["state_norm_bound"] = state_norm_bound
        if not state_finite:
            failures.append("state contains non-finite values")
        if state_norm > state_norm_bound:
            failures.append(f"state norm {state_norm} exceeds bound {state_norm_bound}")
    if prediction is not None:
        prediction_array = np.asarray(prediction, dtype=float).reshape(-1)
        prediction_finite = bool(np.all(np.isfinite(prediction_array)))
        checks["prediction_finite"] = prediction_finite
        checks["derivative_bound"] = derivative_bound
        if not prediction_finite:
            failures.append("prediction contains non-finite values")
        if prediction_array.size and float(np.max(np.abs(prediction_array))) > derivative_bound:
            failures.append("prediction magnitude exceeds the derivative/output guard")
        if output_bounds is not None:
            low, high = output_bounds
            within = bool(np.all((prediction_array >= low) & (prediction_array <= high)))
            checks["output_bounds"] = [low, high]
            checks["within_output_bounds"] = within
            if not within:
                failures.append("prediction violates configured output bounds")
    if spike_rate is not None:
        checks["spike_rate"] = float(spike_rate)
        checks["max_spike_rate"] = float(max_spike_rate)
        if spike_rate > max_spike_rate:
            failures.append(f"spike rate {spike_rate} exceeds the configured maximum")
    for name, (low, high) in (domain_invariants or {}).items():
        if prediction is None:
            continue
        prediction_array = np.asarray(prediction, dtype=float).reshape(-1)
        if prediction_array.size and not (low <= float(prediction_array[0]) <= high):
            failures.append(f"domain invariant {name!r} violated")
        checks[f"invariant_{name}"] = [low, high]
    return HealthReport(valid=not failures, failures=tuple(failures), checks=checks)
