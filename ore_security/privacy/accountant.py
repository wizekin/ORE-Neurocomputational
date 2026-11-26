"""Privacy accounting and release approval (steps 41-42).

The accountant tracks every release's mechanism, privacy unit, epsilon, delta,
composition method, dataset version and purpose. Sequential releases compose under
the configured accountant; a release that would exceed the remaining budget is
rejected before any noise is drawn or data touched.
"""
from __future__ import annotations

import math
from typing import Any

from ..contracts import PrivacyAccountantRecord, PrivacyReleaseRequest
from ..exceptions import BudgetExceededError, PrivacyError

COMPOSITION_METHODS = frozenset({"sequential", "advanced"})


class PrivacyAccountant:
    """Budget ledger for differential-privacy releases (fail closed on exhaustion)."""

    def __init__(self, *, epsilon_budget: float, delta_budget: float,
                 composition: str = "sequential") -> None:
        if epsilon_budget <= 0:
            raise PrivacyError("epsilon budget must be positive")
        if not 0.0 < delta_budget < 1.0:
            raise PrivacyError("delta budget must be in (0, 1)")
        if composition not in COMPOSITION_METHODS:
            raise PrivacyError(
                f"unknown composition method {composition!r}; choose from {sorted(COMPOSITION_METHODS)}"
            )
        self.epsilon_budget = float(epsilon_budget)
        self.delta_budget = float(delta_budget)
        self.composition = composition
        self._records: list[PrivacyAccountantRecord] = []
        self.composed_epsilon = 0.0
        self.composed_delta = 0.0

    def request_release(self, request: PrivacyReleaseRequest) -> PrivacyAccountantRecord:
        """Approve or reject one release against the remaining budget (step 41)."""
        if not isinstance(request, PrivacyReleaseRequest):
            raise TypeError("request must be a PrivacyReleaseRequest")
        if request.mechanism not in {"Gaussian", "Laplace"}:
            raise PrivacyError(
                f"mechanism {request.mechanism!r} is not reviewed; only Gaussian and Laplace are approved"
            )
        if request.mechanism == "Gaussian" and request.delta <= 0.0:
            raise PrivacyError("the Gaussian mechanism requires delta > 0")
        if request.mechanism == "Laplace" and request.delta != 0.0:
            raise PrivacyError("the Laplace mechanism must request delta = 0")
        if request.contribution_limit and request.contribution_limit < request.sensitivity:
            raise PrivacyError(
                "contribution limit tighter than stated sensitivity: review the adjacency analysis"
            )

        projected_epsilon, projected_delta = self._compose(request.epsilon, request.delta)
        if projected_epsilon > self.epsilon_budget or projected_delta > self.delta_budget:
            record = PrivacyAccountantRecord(
                release_id=request.release_id, epsilon=request.epsilon,
                delta=request.delta,
                composed_epsilon=max(projected_epsilon, request.epsilon),
                composed_delta=max(projected_delta, request.delta),
                composition_method=self.composition, approved=False,
                purpose=request.purpose,
            )
            self._records.append(record)
            raise BudgetExceededError(
                f"release {request.release_id} would spend ({projected_epsilon:.4f}, "
                f"{projected_delta:.4f}) but the remaining budget is "
                f"({self.epsilon_budget - self.composed_epsilon:.4f}, "
                f"{self.delta_budget - self.composed_delta:.4f})"
            )
        self.composed_epsilon = projected_epsilon
        self.composed_delta = projected_delta
        record = PrivacyAccountantRecord(
            release_id=request.release_id, epsilon=request.epsilon,
            delta=request.delta, composed_epsilon=self.composed_epsilon,
            composed_delta=self.composed_delta,
            composition_method=self.composition, approved=True,
            purpose=request.purpose,
        )
        self._records.append(record)
        return record

    def _compose(self, epsilon: float, delta: float) -> tuple[float, float]:
        """Apply the configured composition theorem."""
        if self.composition == "sequential":
            return self.composed_epsilon + epsilon, self.composed_delta + delta
        # advanced composition (Dwork-Rothblum-Vadhan style) over the releases so far,
        # using the running count as k and the new release as the k-th
        k = len([record for record in self._records if record.approved]) + 1
        delta_term = max(1e-12, self.delta_budget * 1e-3)
        advanced = (math.sqrt(2.0 * k * math.log(1.0 / delta_term)) * (epsilon / math.sqrt(k))
                    + k * epsilon * (math.exp(epsilon) - 1.0))
        return advanced, self.composed_delta + delta + delta_term

    def remaining_budget(self) -> dict[str, float]:
        return {
            "epsilon_remaining": max(0.0, self.epsilon_budget - self.composed_epsilon),
            "delta_remaining": max(0.0, self.delta_budget - self.composed_delta),
        }

    def records(self) -> tuple[PrivacyAccountantRecord, ...]:
        return tuple(self._records)

    def ledger_report(self) -> dict[str, Any]:
        return {
            "budget": {"epsilon": self.epsilon_budget, "delta": self.delta_budget},
            "spent": {"epsilon": self.composed_epsilon, "delta": self.composed_delta},
            "remaining": self.remaining_budget(),
            "composition": self.composition,
            "release_count": len(self._records),
            "approved_count": len([record for record in self._records if record.approved]),
        }
