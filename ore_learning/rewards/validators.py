"""Deterministic validators for checkable transformations (Point 14, Step 34).

Unit tests, type checks, mathematical solvers, constraint solvers, and arithmetic
checks apply where they exist: for every transition s_i -> s_{i+1} the validator must
return PASS before the step is marked verified.  When no deterministic validator
exists the step is labelled model-assessed or human-reviewed -- never proven.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Sequence

from ..contracts import (
    CheckOutcome,
    DeterministicCheck,
    HumanReview,
    PRMDiagnostic,
    ValidationReport,
)
from .process import ReasoningStep, ReasoningTrace, StepStatus

__all__ = ("ArithmeticCheck", "ConstraintCheck", "DeterministicValidator",
           "TypeCheck", "UnitCheck", "apply_validators", "available_validators")


@dataclass(frozen=True, slots=True)
class DeterministicValidator:
    """A named, versioned deterministic check over a step's declared payload."""

    name: str
    version: str
    check: Callable[[dict[str, Any]], tuple[bool, str]]

    def __call__(self, payload: dict[str, Any]) -> tuple[bool, str]:
        """Run the check."""
        return self.check(payload)


def arithmetic_check(payload: dict[str, Any]) -> tuple[bool, str]:
    """Verify a declared arithmetic identity: a == b + c when supplied."""
    left = payload.get("a")
    right = payload.get("b")
    increment = payload.get("c")
    if None in (left, right, increment):
        return False, "arithmetic payload requires a, b and c"
    if not all(isinstance(value, (int, float)) for value in (left, right, increment)):
        return False, "arithmetic operands must be numeric"
    if not math.isclose(float(left), float(right) + float(increment), rel_tol=1e-9,
                        abs_tol=1e-9):
        return False, f"{left} != {right} + {increment}"
    return True, "identity holds exactly within tolerance"


def type_check(payload: dict[str, Any]) -> tuple[bool, str]:
    """Verify declared fields have the declared types."""
    expectations = payload.get("type_expectations")
    if not isinstance(expectations, dict):
        return False, "type_expectations mapping is required"
    for field_name, expected in expectations.items():
        value = payload.get("values", {}).get(field_name)
        if value is None or not isinstance(value, _TYPE_MAP.get(str(expected), object)):
            return False, f"field {field_name!r} is not of type {expected}"
    return True, "all declared types hold"


_TYPE_MAP: dict[str, type] = {"int": int, "float": float, "str": str, "list": list,
                              "dict": dict, "bool": bool}


def constraint_check(payload: dict[str, Any]) -> tuple[bool, str]:
    """Verify declared inequality constraints: value <= bound for every entry."""
    constraints = payload.get("constraints")
    if not isinstance(constraints, dict) or not constraints:
        return False, "constraints mapping is required"
    values = payload.get("values", {})
    for name, bound in constraints.items():
        value = values.get(name)
        if not isinstance(value, (int, float)) or float(value) > float(bound):
            return False, f"constraint {name}={value} exceeds bound {bound}"
    return True, "all constraints satisfied"


def unit_check(payload: dict[str, Any]) -> tuple[bool, str]:
    """Verify declared unit compatibility between operands."""
    units = payload.get("units")
    if not isinstance(units, dict) or len(units) < 2:
        return False, "at least two declared units are required"
    distinct = set(units.values())
    if len(distinct) != 1:
        return False, f"mixed units {sorted(distinct)} are not compatible"
    return True, f"units compatible ({next(iter(distinct))})"


#: The validator registry; PRM models are structurally absent (Step 34).
BUILTIN_VALIDATORS: dict[str, DeterministicValidator] = {
    "arithmetic-identity": DeterministicValidator("arithmetic-identity", "1.0.0",
                                                  arithmetic_check),
    "type-check": DeterministicValidator("type-check", "1.0.0", type_check),
    "constraint-check": DeterministicValidator("constraint-check", "1.0.0",
                                               constraint_check),
    "unit-check": DeterministicValidator("unit-check", "1.0.0", unit_check),
}


def available_validators() -> list[str]:
    """Names of the registered deterministic validators."""
    return sorted(BUILTIN_VALIDATORS)


def apply_validators(trace: ReasoningTrace,
                     payloads: dict[str, dict[str, Any]] | None = None) -> ValidationReport:
    """Run each step's declared validator and update step statuses (Step 34).

    Steps whose validator returns PASS become ``verified``; failures block verified
    status (a later PRM score can never repair them); steps without a validator are
    labelled ``model_assessed`` or ``human_reviewed`` when a review is attached.
    """
    payloads = payloads or {}
    checks: list[DeterministicCheck] = []
    for step in trace.steps:
        if not step.validator_name:
            step.status = StepStatus.MODEL_ASSESSED
            step.validator_result = "unavailable"
            checks.append(DeterministicCheck(
                check_id=step.step_id, validator="none-registered",
                outcome=CheckOutcome.NOT_APPLICABLE,
                detail=("no deterministic validator exists; the step is "
                        "model-assessed, not proven")))
            continue
        validator = BUILTIN_VALIDATORS.get(step.validator_name)
        if validator is None:
            step.status = StepStatus.MODEL_ASSESSED
            step.validator_result = "unavailable"
            checks.append(DeterministicCheck(
                check_id=step.step_id, validator=step.validator_name,
                outcome=CheckOutcome.NOT_APPLICABLE,
                detail="validator not registered; step is model-assessed"))
            continue
        payload = payloads.get(step.step_id, {})
        passed, detail = validator(payload)
        step.validator_result = "pass" if passed else "fail"
        step.status = StepStatus.VERIFIED if passed else StepStatus.FAILED
        checks.append(DeterministicCheck(
            check_id=step.step_id, validator=validator.name,
            outcome=CheckOutcome.PASS if passed else CheckOutcome.FAIL,
            detail=detail, validator_version=validator.version))
    return ValidationReport(checks=tuple(checks))
