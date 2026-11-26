"""Output contracts declared before execution (Step 24).

Correctness, completeness, evidence, safety, format and performance criteria
are specified in advance, versioned, and never fitted to the produced output
afterwards.  Criteria are deterministic where possible.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from ..contracts import utc_now

__all__ = (
    "CriterionKind",
    "CriterionSpec",
    "ContractViolation",
    "OutputContract",
    "ValidationOutcome",
    "ApprovalRequirement",
)


class CriterionKind(str, Enum):
    """The criterion families of an output contract (Step 24)."""

    SCHEMA = "schema"
    DETERMINISTIC = "deterministic"
    EVIDENCE = "evidence"
    POLICY = "policy"
    PERFORMANCE = "performance"
    COMPLETENESS = "completeness"


@dataclass(frozen=True, slots=True)
class CriterionSpec:
    """One declared criterion with its parameters and severity (Step 24)."""

    criterion_id: str
    kind: CriterionKind
    description: str = ""
    params: Mapping[str, Any] = field(default_factory=dict)
    severity: str = "major"
    required: bool = True

    def __post_init__(self) -> None:
        """Validate the criterion envelope."""
        if not self.criterion_id or not self.criterion_id.strip():
            raise ValueError("criterion_id must be a non-empty string")
        if not isinstance(self.params, Mapping):
            raise ValueError("params must be a mapping")
        if self.severity not in ("info", "minor", "major", "critical"):
            raise ValueError("severity must be info/minor/major/critical")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the criterion."""
        return {"criterion_id": self.criterion_id, "kind": self.kind.value,
                "description": self.description, "params": dict(self.params),
                "severity": self.severity, "required": self.required}


@dataclass(frozen=True, slots=True)
class ApprovalRequirement:
    """Human approval gate for consequential actions (Step 30)."""

    consequential: bool = False
    approver_role: str = ""
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize the requirement."""
        return {"consequential": self.consequential,
                "approver_role": self.approver_role, "rationale": self.rationale}


@dataclass(frozen=True, slots=True)
class OutputContract:
    """The predeclared, versioned contract for one output (Step 24)."""

    contract_id: str
    version: str
    criteria: tuple[CriterionSpec, ...] = ()
    schema: Mapping[str, Any] = field(default_factory=dict)
    approval: ApprovalRequirement = ApprovalRequirement()
    declared_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        """Validate the contract envelope."""
        if not self.contract_id or not self.contract_id.strip():
            raise ValueError("contract_id must be a non-empty string")
        if not self.version or not self.version.strip():
            raise ValueError("version must be a non-empty string")
        if not isinstance(self.schema, Mapping):
            raise ValueError("schema must be a mapping")
        ids = [criterion.criterion_id for criterion in self.criteria]
        if len(set(ids)) != len(ids):
            raise ValueError("criterion ids must be unique within a contract")

    def criteria_of_kind(self, kind: CriterionKind) -> tuple[CriterionSpec, ...]:
        """Return the criteria of one kind."""
        return tuple(item for item in self.criteria if item.kind is kind)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the contract."""
        return {
            "contract_id": self.contract_id,
            "version": self.version,
            "criteria": [item.to_dict() for item in self.criteria],
            "schema": dict(self.schema),
            "approval": self.approval.to_dict(),
            "declared_at": self.declared_at,
        }


@dataclass(frozen=True, slots=True)
class ContractViolation:
    """One detected violation mapped to its contract criterion (Step 28)."""

    criterion_id: str
    severity: str
    message: str
    location: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize the violation."""
        return {"criterion_id": self.criterion_id, "severity": self.severity,
                "message": self.message, "location": self.location}


@dataclass(frozen=True, slots=True)
class ValidationOutcome:
    """The result of validating an artifact against its contract (Step 25)."""

    passed: bool
    violations: tuple[ContractViolation, ...] = ()
    unverified_criteria: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the outcome."""
        return {"passed": self.passed,
                "violations": [item.to_dict() for item in self.violations],
                "unverified_criteria": list(self.unverified_criteria)}
