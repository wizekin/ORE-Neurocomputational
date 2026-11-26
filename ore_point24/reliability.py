"""Phase VII, Step 33 - reuse measured reliability and resource-cost estimation.

``SQLiteResourceHistory`` is the persistent observation source; execution
reliability is estimated from observed successes and failures using the
repository's documented Beta(2, 2) prior ``p_hat = (alpha0 + s) / (alpha0 +
beta0 + s + f)``; circuit breakers exclude repeatedly failing modules; moving
averages track measured latency, memory and CPU costs; declared estimates are
retained only where measurements are absent and are labelled as assumptions;
and every estimate exposes its confidence and sample count.  Insufficient
observations stay visible and are never mistaken for proven reliability.
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ore_contracts.resource_coordination import (
    ModuleReliabilityTracker,
    SQLiteResourceHistory,
)

from .core import sha256_digest, utc_now_iso

RELIABILITY_SCHEMA = "ore.point24.reliability.v1"
DEFAULT_PRIOR_ALPHA = 2.0
DEFAULT_PRIOR_BETA = 2.0


@dataclass(frozen=True, slots=True)
class ReliabilityEstimate:
    """One Beta-posterior estimate with its evidence exposed."""

    task_family: str
    module_id: str
    point_estimate: float
    successes: int
    failures: int
    sample_count: int
    prior: str
    evidence_state: str  # measured | insufficient_observations

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_family": self.task_family,
            "module_id": self.module_id,
            "point_estimate": self.point_estimate,
            "successes": self.successes,
            "failures": self.failures,
            "sample_count": self.sample_count,
            "prior": self.prior,
            "evidence_state": self.evidence_state,
        }


class MeasuredReliabilityFacade:
    """Wrap the repository reliability tracker with explicit evidence states.

    Reliability changes only after an actual recorded outcome; the facade
    exposes the posterior, the raw counts and whether the sample is
    sufficient, so an unmeasured module can never masquerade as proven.
    """

    def __init__(
        self,
        *,
        database_path: str | Path | None = None,
        alpha: float = DEFAULT_PRIOR_ALPHA,
        beta: float = DEFAULT_PRIOR_BETA,
        min_samples: int = 5,
    ) -> None:
        if database_path is None:
            handle = tempfile.TemporaryDirectory(prefix="point24_reliability_")
            database_path = Path(handle.name) / "reliability.sqlite3"
            self._temporary_dir = handle
        else:
            self._temporary_dir = None
        self.history = SQLiteResourceHistory(Path(database_path))
        self.tracker = ModuleReliabilityTracker(
            self.history, prior_alpha=alpha, prior_beta=beta,
        )
        self.alpha = alpha
        self.beta = beta
        self.min_samples = min_samples

    def close(self) -> None:
        """Release temporary storage after all SQLite operations have finished.

        SQLiteResourceHistory owns no long-lived connection; its individual
        operations close their connections. This method is therefore safe to
        call more than once and only removes storage created by this facade.
        """
        temporary_dir = self._temporary_dir
        self._temporary_dir = None
        if temporary_dir is not None:
            temporary_dir.cleanup()

    def __enter__(self) -> "MeasuredReliabilityFacade":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def record_outcome(self, task_family: str, module_id: str, *, success: bool) -> None:
        """Reliability changes only through an actual recorded outcome."""
        self.tracker.record_outcome(task_family, module_id, success)

    def estimate(self, task_family: str, module_id: str) -> ReliabilityEstimate:
        row = self.history.get(task_family, module_id)
        successes = int(row.get("successes", 0) or 0)
        failures = int(row.get("failures", 0) or 0)
        total = successes + failures
        posterior = self.tracker.probability(task_family, module_id)
        return ReliabilityEstimate(
            task_family=task_family,
            module_id=module_id,
            point_estimate=round(posterior, 6),
            successes=successes,
            failures=failures,
            sample_count=total,
            prior=f"Beta({self.alpha:g}, {self.beta:g})",
            evidence_state=(
                "measured" if total >= self.min_samples else "insufficient_observations"
            ),
        )

    def circuit_open(self, task_family: str, module_id: str) -> bool:
        """A repeatedly failing module is excluded while its breaker is open."""
        return self.tracker.is_circuit_open(task_family, module_id)

    def failure_penalty(self, task_family: str, module_id: str) -> float:
        return self.tracker.failure_penalty(task_family, module_id)


class DeclaredCostEstimates:
    """Declared (not measured) resource costs, always labelled as assumptions."""

    def __init__(self) -> None:
        self._declared: dict[tuple[str, str], Mapping[str, float]] = {}

    def declare(self, task_family: str, module_id: str, costs: Mapping[str, float]) -> None:
        self._declared[(task_family, module_id)] = dict(costs)

    def get(self, task_family: str, module_id: str) -> dict[str, Any]:
        declared = self._declared.get((task_family, module_id))
        if declared is None:
            return {"available": False, "note": "no measured or declared cost available"}
        return {
            "available": True,
            "kind": "declared_assumption",
            "values": dict(declared),
            "note": "declared estimate; labelled as assumption until measured",
        }


__all__ = [
    "RELIABILITY_SCHEMA",
    "DEFAULT_PRIOR_ALPHA",
    "DEFAULT_PRIOR_BETA",
    "DeclaredCostEstimates",
    "MeasuredReliabilityFacade",
    "ReliabilityEstimate",
]
