"""Structural, semantic and deterministic validation (Steps 25, 30).

The schema validator implements a small auditable subset of JSON-schema
semantics (types, required fields, bounds, enums, lengths).  Schema validity
alone is not correctness: domain invariants and deterministic checks are
validated separately, and a failed deterministic test rejects the output in
final verification.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from ..contracts import TaskResult
from .contracts import (
    ContractViolation,
    CriterionKind,
    CriterionSpec,
    OutputContract,
    ValidationOutcome,
)

__all__ = (
    "ContractValidator",
    "DeterministicCheckRunner",
    "SchemaValidator",
    "SchemaViolationError",
    "validate_schema",
)

_TYPE_CHECKS: dict[str, Callable[[Any], bool]] = {
    "object": lambda value: isinstance(value, Mapping),
    "array": lambda value: isinstance(value, (list, tuple)),
    "string": lambda value: isinstance(value, str),
    "integer": lambda value: isinstance(value, int) and not isinstance(value, bool),
    "number": lambda value: (isinstance(value, (int, float))
                             and not isinstance(value, bool)),
    "boolean": lambda value: isinstance(value, bool),
}


class SchemaViolationError(ValueError):
    """Raised by the strict variant of the schema validator (Step 25)."""


def validate_schema(payload: Any, schema: Mapping[str, Any]) -> list[str]:
    """Validate a payload against a declarative schema subset (Step 25).

    Supported constraints per field: ``type`` (object/array/string/integer/
    number/boolean), ``required`` (top-level list), ``properties`` (nested
    object fields), ``items`` (array element schema), ``minimum``/``maximum``,
    ``min_length``/``max_length``, and ``enum``.  Returns the list of
    violation messages; an empty list means structurally valid.
    """
    errors: list[str] = []

    def _check_field(name: str, value: Any, spec: Mapping[str, Any]) -> None:
        expected_type = spec.get("type")
        if expected_type is not None:
            check = _TYPE_CHECKS.get(str(expected_type))
            if check is None:
                errors.append(f"{name}: unknown type {expected_type!r}")
            elif not check(value):
                errors.append(f"{name}: expected {expected_type}, got "
                              f"{type(value).__name__}")
                return
        if "enum" in spec and value not in spec["enum"]:
            errors.append(f"{name}: {value!r} not in enum {list(spec['enum'])}")
        if "minimum" in spec:
            try:
                if float(value) < float(spec["minimum"]):
                    errors.append(f"{name}: below minimum {spec['minimum']}")
            except (TypeError, ValueError):
                pass
        if "maximum" in spec:
            try:
                if float(value) > float(spec["maximum"]):
                    errors.append(f"{name}: above maximum {spec['maximum']}")
            except (TypeError, ValueError):
                pass
        if "min_length" in spec:
            try:
                if len(value) < int(spec["min_length"]):
                    errors.append(f"{name}: shorter than min_length {spec['min_length']}")
            except TypeError:
                pass
        if "max_length" in spec:
            try:
                if len(value) > int(spec["max_length"]):
                    errors.append(f"{name}: longer than max_length {spec['max_length']}")
            except TypeError:
                pass
        if str(expected_type) == "object" and isinstance(value, Mapping):
            for child, child_spec in (spec.get("properties") or {}).items():
                if child not in value:
                    if (child_spec or {}).get("required", False):
                        errors.append(f"{name}.{child}: required field missing")
                    continue
                _check_field(f"{name}.{child}", value[child], child_spec or {})
        if str(expected_type) == "array" and isinstance(value, (list, tuple)) \
                and "items" in spec:
            for index, item in enumerate(value):
                _check_field(f"{name}[{index}]", item, spec["items"])

    if not isinstance(payload, Mapping):
        return [f"payload: expected object, got {type(payload).__name__}"]
    for field_name in schema.get("required", ()) or ():
        if field_name not in payload:
            errors.append(f"{field_name}: required field missing")
    for field_name, field_spec in (schema.get("properties") or {}).items():
        if field_name in payload:
            _check_field(field_name, payload[field_name], field_spec or {})
    return errors


class SchemaValidator:
    """Reusable schema validator with strict and non-strict modes (Step 25)."""

    def validate(self, payload: Any, schema: Mapping[str, Any]) -> list[str]:
        """Return the list of violation messages."""
        return validate_schema(payload, schema)

    def validate_strict(self, payload: Any, schema: Mapping[str, Any]) -> None:
        """Raise SchemaViolationError listing every violation (Step 13)."""
        errors = validate_schema(payload, schema)
        if errors:
            raise SchemaViolationError("; ".join(errors))


class DeterministicCheckRunner:
    """Registry of deterministic checks run by final verification (Step 30)."""

    def __init__(self) -> None:
        """Start with an empty check registry."""
        self._checks: dict[str, Callable[[Mapping[str, Any]], tuple[bool, str]]] = {}

    def register(self, check_id: str,
                 check: Callable[[Mapping[str, Any]], tuple[bool, str]],
                 *, replace: bool = False) -> None:
        """Register a deterministic check returning (passed, detail)."""
        if check_id in self._checks and not replace:
            raise ValueError(f"duplicate deterministic check id: {check_id}")
        self._checks[check_id] = check

    def run(self, check_id: str, payload: Mapping[str, Any]) -> tuple[bool, str]:
        """Run one registered check; unknown checks fail closed (Step 30)."""
        check = self._checks.get(check_id)
        if check is None:
            return False, f"unknown deterministic check: {check_id}"
        return check(payload)

    def run_all(self, check_ids: Sequence[str],
                payload: Mapping[str, Any]) -> dict[str, tuple[bool, str]]:
        """Run several checks and return their results keyed by id."""
        return {check_id: self.run(check_id, payload) for check_id in check_ids}


class ContractValidator:
    """Validate an artifact against its predeclared contract (Steps 25, 28)."""

    def __init__(self, schema_validator: SchemaValidator | None = None,
                 deterministic_runner: DeterministicCheckRunner | None = None,
                 policy_check: Callable[[Mapping[str, Any]], tuple[bool, str]]
                 | None = None) -> None:
        """Compose the validator; the policy check is injected (Step 30)."""
        self._schema = schema_validator or SchemaValidator()
        self._deterministic = deterministic_runner or DeterministicCheckRunner()
        self._policy_check = policy_check

    def validate(self, artifact: Mapping[str, Any],
                 contract: OutputContract) -> ValidationOutcome:
        """Map every criterion to its checker and aggregate violations (Step 25)."""
        violations: list[ContractViolation] = []
        unverified: list[str] = []
        for criterion in contract.criteria:
            violation = self._check_criterion(artifact, criterion)
            if violation == "unverified":
                unverified.append(criterion.criterion_id)
            elif violation is not None:
                violations.append(violation)
        if contract.schema:
            for message in self._schema.validate(artifact, contract.schema):
                violations.append(ContractViolation(
                    criterion_id="schema", severity="major", message=message))
        required_unmet = [item.criterion_id for item in contract.criteria
                          if item.required and item.criterion_id in unverified]
        passed = not violations and not required_unmet
        return ValidationOutcome(passed=passed, violations=tuple(violations),
                                 unverified_criteria=tuple(unverified))

    def _check_criterion(self, artifact: Mapping[str, Any],
                         criterion: CriterionSpec) -> ContractViolation | None | str:
        """Return a violation, None (pass), or the literal 'unverified'."""
        kind = criterion.kind
        params = dict(criterion.params)
        if kind is CriterionKind.SCHEMA:
            schema = params.get("schema") or {}
            errors = self._schema.validate(artifact, schema)
            if errors:
                return ContractViolation(criterion.criterion_id, criterion.severity,
                                          "; ".join(errors))
            return None
        if kind is CriterionKind.DETERMINISTIC:
            check_id = str(params.get("check_id", ""))
            if not check_id:
                return "unverified"
            passed, detail = self._deterministic.run(check_id, artifact)
            return None if passed else ContractViolation(
                criterion.criterion_id, criterion.severity, detail)
        if kind is CriterionKind.EVIDENCE:
            minimum = int(params.get("minimum_refs", 1))
            refs = artifact.get("evidence_refs", ())
            if len(refs) < minimum:
                return ContractViolation(
                    criterion.criterion_id, criterion.severity,
                    f"evidence below minimum: {len(refs)} < {minimum}")
            return None
        if kind is CriterionKind.POLICY:
            if self._policy_check is None:
                return "unverified"
            passed, detail = self._policy_check(artifact)
            return None if passed else ContractViolation(
                criterion.criterion_id, criterion.severity, detail)
        if kind is CriterionKind.PERFORMANCE:
            threshold = float(params.get("max_seconds", 1e9))
            elapsed = float(artifact.get("elapsed_seconds", 0.0))
            return None if elapsed <= threshold else ContractViolation(
                criterion.criterion_id, criterion.severity,
                f"elapsed {elapsed}s exceeds budget {threshold}s")
        if kind is CriterionKind.COMPLETENESS:
            required_keys = list(params.get("required_keys", ()))
            missing = [key for key in required_keys if key not in artifact]
            return None if not missing else ContractViolation(
                criterion.criterion_id, criterion.severity,
                f"missing required output keys: {missing}")
        return "unverified"

    def validate_task_result(self, result: TaskResult) -> ValidationOutcome:
        """Validate one worker result against its task output schema (Step 13)."""
        schema = result.payload.get("output_schema") if isinstance(
            result.payload, Mapping) else None
        if not isinstance(schema, Mapping) or not schema:
            return ValidationOutcome(passed=True)
        errors = self._schema.validate(result.payload, schema)
        violations = tuple(ContractViolation("schema", "major", message)
                           for message in errors)
        return ValidationOutcome(passed=not violations, violations=violations)
