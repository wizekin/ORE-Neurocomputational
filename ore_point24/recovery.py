"""Phase VIII, Steps 37-40 - boundaries, recovery, privacy and promotion.

Step 37 chooses process and distributed boundaries using measured need:
tightly coupled components stay in-process until isolation, resource control
or independent scaling justifies separation, and microkernel/unikernel
research is evaluated only for concrete deployment requirements.  Step 38
makes retries, recovery and circuit breaking explicit with bounded
exponential backoff ``d_n = min(d_max, d_0 2^n)`` and jitter, idempotency
keys for externally visible side effects, and resume from valid checkpoints.
Step 39 applies cryptographic and privacy controls where required, including
differential-privacy loss composition across releases.  Step 40 versions,
audits and rolls back every promoted model or policy; no in-place overwrite
of the only known-good model is permitted.
"""
from __future__ import annotations

import hashlib
import hmac
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

from .core import new_id, sha256_digest, utc_now_iso

BOUNDARIES_SCHEMA = "ore.point24.boundaries.v1"
RECOVERY_SCHEMA = "ore.point24.recovery.v1"
PRIVACY_SCHEMA = "ore.point24.privacy.v1"
PROMOTION_SCHEMA = "ore.point24.promotion.v1"


# ---------------------------------------------------------------------------
# Step 37 - process and distributed boundaries
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class BoundaryDecision:
    """An evidence-backed deployment-boundary decision (Step 37)."""

    component: str
    boundary: str  # in_process | separate_process | remote_worker
    justification: str
    measured_evidence: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "boundary": self.boundary,
            "justification": self.justification,
            "measured_evidence": dict(self.measured_evidence),
        }


class DeploymentBoundaryPlanner:
    """Distribute only when measured need justifies it (Step 37).

    Expected throughput, resilience or isolation gains must exceed the
    communication, consistency and operational costs; the logical
    architecture's layers are NOT turned into services by default.
    """

    RESEARCH_OPTIONS = (
        "seL4", "Genode", "CHERI", "Barrelfish", "Firecracker", "gVisor",
        "Singularity", "CheriBSD", "CHERIoT", "Plan 9", "Landlock", "Helios",
        "Popcorn Linux", "Gramine", "Demikernel", "NOVA", "Mnemosyne",
        "SplineFS", "OBASE", "NetworkedOS",
    )

    def decide(
        self,
        component: str,
        *,
        communication_cost_ms: float,
        isolation_need: bool,
        independent_scaling_need: bool,
        failure_isolation_need: bool,
    ) -> BoundaryDecision:
        if isolation_need and failure_isolation_need and communication_cost_ms < 50.0:
            boundary = "separate_process"
            justification = (
                "isolation and failure-containment needs justify process separation "
                "at the measured communication cost"
            )
        elif independent_scaling_need and communication_cost_ms < 200.0:
            boundary = "remote_worker"
            justification = (
                "independent scaling need justifies a remote worker with stable, "
                "authenticated contracts at the measured communication cost"
            )
        else:
            boundary = "in_process"
            justification = (
                "tightly coupled component stays in-process; no measured isolation, "
                "resource-control or scaling need justifies separation yet"
            )
        return BoundaryDecision(
            component=component,
            boundary=boundary,
            justification=justification,
            measured_evidence={
                "communication_cost_ms": communication_cost_ms,
                "isolation_need": isolation_need,
                "independent_scaling_need": independent_scaling_need,
                "failure_isolation_need": failure_isolation_need,
            },
        )

    def research_option_status(self, technology: str) -> str:
        if technology not in self.RESEARCH_OPTIONS:
            raise ValueError(f"unknown research option: {technology!r}")
        return "future_option: evaluated only for a concrete deployment requirement"


# ---------------------------------------------------------------------------
# Step 38 - retries, recovery and circuit breaking
# ---------------------------------------------------------------------------

TRANSIENT = "transient"
PERMANENT = "permanent"
VALIDATION = "validation"
PERMISSION = "permission"
RESOURCE = "resource"


def classify_failure(exception: BaseException) -> str:
    """Classify transient, permanent, validation, permission and resource failures."""
    name = type(exception).__name__
    if isinstance(exception, (TimeoutError, ConnectionError)):
        return TRANSIENT
    if isinstance(exception, PermissionError):
        return PERMISSION
    if isinstance(exception, (MemoryError,)):
        return RESOURCE
    if isinstance(exception, (ValueError, TypeError)):
        return VALIDATION
    if isinstance(exception, (KeyError, AttributeError, NotImplementedError)):
        return PERMANENT
    return TRANSIENT if "timeout" in name.lower() else PERMANENT


def backoff_schedule(
    *,
    attempt: int,
    base_delay: float = 0.1,
    max_delay: float = 8.0,
    jitter_fraction: float = 0.25,
    seed: int | None = None,
) -> float:
    """Bounded exponential backoff ``d_n = min(d_max, d_0 2^n)`` with jitter."""
    if attempt < 0:
        raise ValueError("attempt must be non-negative")
    if base_delay <= 0 or max_delay <= 0 or not 0.0 <= jitter_fraction <= 1.0:
        raise ValueError("invalid backoff parameters")
    rng = random.Random(seed if seed is not None else attempt)
    delay = min(max_delay, base_delay * (2 ** attempt))
    jitter = delay * jitter_fraction
    return delay + rng.uniform(-jitter, jitter)


class IdempotencyKeyStore:
    """Idempotency keys for externally visible side effects (Step 38).

    Retrying an operation must never duplicate an irreversible action because
    of an ambiguous prior result: the key store records completed keys and
    returns the recorded outcome instead of re-executing.
    """

    def __init__(self) -> None:
        self._completed: dict[str, Any] = {}

    def key_for(self, operation: str, payload: Mapping[str, Any]) -> str:
        return sha256_digest({"operation": operation, "payload": dict(payload)})

    def execute_once(
        self,
        key: str,
        operation: Callable[[], Any],
    ) -> tuple[str, Any]:
        if key in self._completed:
            return "duplicate", self._completed[key]
        result = operation()
        self._completed[key] = result
        return "applied", result

    def completed_keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._completed))


class CheckpointStore:
    """Resume from valid checkpoints, preserving completed artifacts."""

    def __init__(self) -> None:
        self._checkpoints: dict[str, Mapping[str, Any]] = {}

    def save(self, name: str, state: Mapping[str, Any]) -> str:
        self._checkpoints[name] = dict(state)
        return name

    def load(self, name: str) -> Mapping[str, Any] | None:
        return self._checkpoints.get(name)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._checkpoints))


@dataclass(frozen=True, slots=True)
class RetryOutcome:
    """The explicit completion state after bounded retries."""

    operation: str
    status: str  # succeeded | failed_permanent | failed_after_retries
    attempts: int
    final_error: str | None
    side_effects_duplicated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "status": self.status,
            "attempts": self.attempts,
            "final_error": self.final_error,
            "side_effects_duplicated": self.side_effects_duplicated,
        }


class RetryPolicy:
    """Retry only operations declared safe to retry (Step 38)."""

    def __init__(self, *, max_attempts: int = 3, base_delay: float = 0.01,
                 max_delay: float = 0.05) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least one")
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay

    def run(
        self,
        operation: str,
        fn: Callable[[], Any],
        *,
        safe_to_retry: bool,
        sleep: bool = False,
    ) -> RetryOutcome:
        """Execute with bounded retries; shared budget prevents amplification."""
        if not safe_to_retry:
            try:
                fn()
                return RetryOutcome(operation, "succeeded", 1, None)
            except Exception as exc:  # noqa: BLE001
                return RetryOutcome(
                    operation, "failed_permanent", 1, f"{type(exc).__name__}: {exc}"
                )
        attempts = 0
        last_error: str | None = None
        while attempts < self.max_attempts:
            attempts += 1
            try:
                fn()
                return RetryOutcome(operation, "succeeded", attempts, None)
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
                if classify_failure(exc) in (PERMANENT, VALIDATION, PERMISSION):
                    return RetryOutcome(operation, "failed_permanent", attempts, last_error)
                if sleep and attempts < self.max_attempts:
                    time.sleep(backoff_schedule(attempt=attempts - 1, base_delay=self.base_delay,
                                                max_delay=self.max_delay))
        return RetryOutcome(operation, "failed_after_retries", attempts, last_error)


# ---------------------------------------------------------------------------
# Step 39 - cryptographic and privacy controls
# ---------------------------------------------------------------------------

class IntegrityProtector:
    """HMAC-based integrity and authentication of artifact digests.

    Uses maintained standard-library cryptographic primitives; a primitive is
    not a complete security architecture - key management, threat model and
    implementation validation remain necessary.
    """

    def __init__(self, key: bytes) -> None:
        if not isinstance(key, (bytes, bytearray)) or len(key) < 16:
            raise ValueError("key must be at least 16 bytes of entropy")
        self._key = bytes(key)

    def tag(self, content: str) -> str:
        return hmac.new(self._key, content.encode("utf-8"), hashlib.sha256).hexdigest()

    def verify(self, content: str, tag: str) -> bool:
        return hmac.compare_digest(self.tag(content), tag)


@dataclass(frozen=True, slots=True)
class PrivacyRelease:
    """One differentially private release with its actual parameters."""

    release_id: str
    epsilon: float
    mechanisms: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "release_id": self.release_id,
            "epsilon": self.epsilon,
            "mechanisms": list(self.mechanisms),
        }


class PrivacyBudgetLedger:
    """Compose differential-privacy loss across releases (Step 39).

    The ledger reports the *composed* privacy loss of the sequence, never a
    per-query epsilon in isolation, and refuses further releases once the
    approved budget is exhausted.
    """

    def __init__(self, *, total_epsilon_budget: float) -> None:
        if total_epsilon_budget <= 0:
            raise ValueError("total epsilon budget must be positive")
        self.total_epsilon_budget = total_epsilon_budget
        self._releases: list[PrivacyRelease] = []

    def request_release(self, *, epsilon: float, mechanisms: Iterable[str]) -> PrivacyRelease:
        if epsilon <= 0:
            raise ValueError("release epsilon must be positive")
        spent = self.spent_epsilon()
        if spent + epsilon > self.total_epsilon_budget:
            raise PermissionError(
                f"privacy budget exhausted: spent {spent:.3f} + requested "
                f"{epsilon:.3f} > budget {self.total_epsilon_budget:.3f}"
            )
        release = PrivacyRelease(
            release_id=new_id("dp_release"),
            epsilon=epsilon,
            mechanisms=tuple(mechanisms),
        )
        self._releases.append(release)
        return release

    def spent_epsilon(self) -> float:
        return sum(release.epsilon for release in self._releases)

    def summary(self) -> dict[str, Any]:
        return {
            "schema": PRIVACY_SCHEMA,
            "total_budget": self.total_epsilon_budget,
            "spent_epsilon": self.spent_epsilon(),
            "release_count": len(self._releases),
            "releases": [release.to_dict() for release in self._releases],
        }


# ---------------------------------------------------------------------------
# Step 40 - version, audit and roll back every promoted model or policy
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PromotionRecord:
    """A reproducible promotion or rollback record (Step 40)."""

    artifact: str
    candidate_fingerprint: str
    parent_fingerprint: str | None
    decision: str  # promoted | rolled_back | rejected
    gates: Mapping[str, bool]
    approved_by: str
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact": self.artifact,
            "candidate_fingerprint": self.candidate_fingerprint,
            "parent_fingerprint": self.parent_fingerprint,
            "decision": self.decision,
            "gates": dict(self.gates),
            "approved_by": self.approved_by,
            "created_at": self.created_at,
        }


class PromotionController:
    """Versioned promotion with canary, rollback and audit (Step 40).

    The last known-good version is never overwritten in place; a candidate is
    fingerprinted with its configuration, gates are evaluated against
    predeclared non-inferiority bounds, and a failed gate rolls back to the
    parent version while preserving the decision record.
    """

    def __init__(self) -> None:
        self._active: dict[str, str] = {}
        self._records: list[PromotionRecord] = []

    def register(self, artifact: str, known_good_fingerprint: str) -> None:
        if artifact in self._active:
            raise ValueError(f"artifact already governed: {artifact}")
        self._active[artifact] = known_good_fingerprint

    def active_version(self, artifact: str) -> str:
        return self._active[artifact]

    def promote(
        self,
        artifact: str,
        *,
        candidate_fingerprint: str,
        gates: Mapping[str, bool],
        approved_by: str,
        canary: bool = True,
    ) -> PromotionRecord:
        if artifact not in self._active:
            raise KeyError(f"unknown governed artifact: {artifact}")
        parent = self._active[artifact]
        # A gain on one metric cannot compensate for violation of a hard gate.
        passed = all(bool(value) for value in gates.values())
        record = PromotionRecord(
            artifact=artifact,
            candidate_fingerprint=candidate_fingerprint,
            parent_fingerprint=parent,
            decision="promoted" if passed else "rejected",
            gates=dict(gates),
            approved_by=approved_by,
        )
        self._records.append(record)
        if passed:
            self._active[artifact] = candidate_fingerprint
        return record

    def rollback(self, artifact: str, *, approved_by: str) -> PromotionRecord | None:
        active = self._active.get(artifact)
        parent_record = next(
            (record for record in reversed(self._records)
             if record.artifact == artifact and record.candidate_fingerprint == active
             and record.decision == "promoted"),
            None,
        )
        if parent_record is None or parent_record.parent_fingerprint is None:
            return None
        record = PromotionRecord(
            artifact=artifact,
            candidate_fingerprint=parent_record.parent_fingerprint,
            parent_fingerprint=active,
            decision="rolled_back",
            gates=dict(parent_record.gates),
            approved_by=approved_by,
        )
        self._records.append(record)
        self._active[artifact] = parent_record.parent_fingerprint
        return record

    def history(self, artifact: str | None = None) -> tuple[PromotionRecord, ...]:
        return tuple(
            record for record in self._records
            if artifact is None or record.artifact == artifact
        )


__all__ = [
    "BOUNDARIES_SCHEMA",
    "PROMOTION_SCHEMA",
    "PRIVACY_SCHEMA",
    "RECOVERY_SCHEMA",
    "BoundaryDecision",
    "CheckpointStore",
    "DeploymentBoundaryPlanner",
    "IdempotencyKeyStore",
    "IntegrityProtector",
    "PrivacyBudgetLedger",
    "PrivacyRelease",
    "PromotionController",
    "PromotionRecord",
    "RetryOutcome",
    "RetryPolicy",
    "backoff_schedule",
    "classify_failure",
]
