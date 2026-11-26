"""Typed exceptions for the Point 19 concurrent and incremental cognitive execution engine."""
from __future__ import annotations


class RuntimeEngineError(Exception):
    """Base class for ore_runtime failures that carry engine-specific diagnostics."""


class PlanValidationError(RuntimeEngineError):
    """Reject a cognitive plan that is malformed, unsatisfiable, or missing inputs."""

    def __init__(self, message: str, diagnostic_path: tuple[str, ...] = ()) -> None:
        """Store the human-readable reason and the ordered path of offending nodes."""
        super().__init__(message)
        self.diagnostic_path = tuple(diagnostic_path)


class CycleRejectedError(PlanValidationError):
    """Report an accidental dependency cycle with the cycle's diagnostic path."""


class CapabilityUnavailableError(PlanValidationError):
    """Reject a task when no worker can ever satisfy its capability requirements."""


class CommitConflictError(RuntimeEngineError):
    """Signal that an optimistic commit raced with a newer authoritative revision."""

    def __init__(self, read_version: int, current_version: int) -> None:
        """Capture the stale read version and the authoritative version observed."""
        super().__init__(
            f"optimistic commit conflict: read revision {read_version} is stale, "
            f"current revision is {current_version}"
        )
        self.read_version = read_version
        self.current_version = current_version


class TaskCancelledError(RuntimeEngineError):
    """Raise inside a task body when a cooperative checkpoint observes cancellation."""


class DeadlineExpiredError(RuntimeEngineError):
    """Raise when a task exceeded its deadline and cannot make progress."""


class EventSchemaError(RuntimeEngineError):
    """Reject an event envelope that fails its registered schema contract."""

    def __init__(self, message: str, errors: tuple[str, ...] = ()) -> None:
        """Store the schema violation details for diagnostics."""
        super().__init__(message)
        self.errors = tuple(errors)


class OutboxPublishError(RuntimeEngineError):
    """Report a publication failure that leaves an outbox row pending for retry."""


class SagaCompensationIncompleteError(RuntimeEngineError):
    """Report a saga whose compensating actions did not fully reverse executed steps."""


class InvariantViolationError(RuntimeEngineError):
    """Report a broken engine invariant with the failing check name and evidence."""

    def __init__(self, invariant: str, evidence: str) -> None:
        """Capture the invariant name and the observed violating state."""
        super().__init__(f"invariant '{invariant}' violated: {evidence}")
        self.invariant = invariant
        self.evidence = evidence


class DeterminismError(RuntimeEngineError):
    """Report a replay whose invariant outcomes diverge from the recorded execution."""


class RecoveryError(RuntimeEngineError):
    """Report a checkpoint that cannot be resumed after a simulated crash."""
