"""Exception hierarchy for the Formal Reasoning and Constraint Engine.

Failures are classified, never silently swallowed, and never converted into
misleading solver statuses. A failure status such as ``unknown``, ``timeout``
or ``unsupported`` is never reported as ``false`` (Point 5, Section 1).
"""
from __future__ import annotations


class FormalReasoningError(Exception):
    """Base class for all FRCE errors."""


class InvalidFormalizationError(FormalReasoningError):
    """The request or IR failed validation before any solver was invoked."""


class UnsupportedTheoryError(FormalReasoningError):
    """A required theory is not permitted or not supported by any backend."""


class SolverBudgetExceededError(FormalReasoningError):
    """A wall-clock, CPU, memory, or output budget was exhausted."""


class SolverCrashError(FormalReasoningError):
    """An external solver or worker process crashed; diagnostics are retained."""


class MalformedSolverOutputError(FormalReasoningError):
    """Solver output failed schema or integrity validation."""


class UnsafeRuleError(FormalReasoningError):
    """A rule is unsafe (unbound variables) or uses unbounded constructs."""


class NonStratifiableProgramError(FormalReasoningError):
    """Negation through recursion cannot be stratified for Datalog evaluation."""


class UnitMismatchError(InvalidFormalizationError):
    """Dimensionally incompatible units were combined in arithmetic."""


class AmbiguousEntityError(InvalidFormalizationError):
    """Entity resolution produced several candidates without a discriminator."""


class InsufficientFormalizationError(FormalReasoningError):
    """The request cannot be operationalized without clarification (Step 7)."""


class GovernanceViolationError(FormalReasoningError):
    """An attempted knowledge write violated the read-only/promotion policy."""


class ProofCheckFailureError(FormalReasoningError):
    """A proof candidate failed the trusted kernel check."""


__all__ = [
    "FormalReasoningError",
    "InvalidFormalizationError",
    "UnsupportedTheoryError",
    "SolverBudgetExceededError",
    "SolverCrashError",
    "MalformedSolverOutputError",
    "UnsafeRuleError",
    "NonStratifiableProgramError",
    "UnitMismatchError",
    "AmbiguousEntityError",
    "InsufficientFormalizationError",
    "GovernanceViolationError",
    "ProofCheckFailureError",
]
