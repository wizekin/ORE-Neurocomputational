"""Procedural memory: reusable, test-gated procedures (Step 39).

Procedures store reusable algorithms, workflows, cognitive procedures and
tool routines with preconditions, inputs, outputs, dependencies, versions,
originating episodes and test results.  Promotion from draft to validated is
refused without passing tests, and supersession is explicit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

from ..contracts import new_id, utc_now

__all__ = (
    "ProcedureNotPromotedError",
    "ProcedureSpec",
    "ProcedureStatus",
    "ProcedureTestRecord",
    "ProceduralMemory",
)


class ProcedureStatus(str, Enum):
    """Lifecycle of a procedure (Step 39)."""

    DRAFT = "draft"
    VALIDATED = "validated"
    SUPERSEDED = "superseded"
    RETIRED = "retired"


class ProcedureNotPromotedError(RuntimeError):
    """Promotion refused: the procedure lacks required passing tests (Step 39)."""


@dataclass(frozen=True, slots=True)
class ProcedureSpec:
    """A reusable procedure declaration (Step 39)."""

    procedure_id: str
    name: str
    version: str
    steps: tuple[str, ...] = ()
    preconditions: tuple[str, ...] = ()
    inputs_schema: Mapping[str, Any] = field(default_factory=dict)
    outputs_schema: Mapping[str, Any] = field(default_factory=dict)
    dependencies: tuple[str, ...] = ()
    originating_episodes: tuple[str, ...] = ()
    status: ProcedureStatus = ProcedureStatus.DRAFT
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        """Validate the procedure envelope."""
        if not self.name or not self.name.strip():
            raise ValueError("name must be a non-empty string")
        if not self.version or not self.version.strip():
            raise ValueError("version must be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the procedure."""
        return {
            "procedure_id": self.procedure_id,
            "name": self.name,
            "version": self.version,
            "steps": list(self.steps),
            "preconditions": list(self.preconditions),
            "inputs_schema": dict(self.inputs_schema),
            "outputs_schema": dict(self.outputs_schema),
            "dependencies": list(self.dependencies),
            "originating_episodes": list(self.originating_episodes),
            "status": self.status.value,
            "created_at": self.created_at,
        }


@dataclass(frozen=True, slots=True)
class ProcedureTestRecord:
    """One executed test of a procedure (Step 39)."""

    test_id: str
    procedure_id: str
    passed: bool
    detail: str = ""
    executed_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the record."""
        return {"test_id": self.test_id, "procedure_id": self.procedure_id,
                "passed": self.passed, "detail": self.detail,
                "executed_at": self.executed_at}


class ProceduralMemory:
    """Store procedures and enforce the test-gated promotion (Step 39)."""

    def __init__(self, min_passing_tests: int = 1) -> None:
        """Configure the minimum passing tests for promotion."""
        self._procedures: dict[str, ProcedureSpec] = {}
        self._tests: dict[str, list[ProcedureTestRecord]] = {}
        self._min_passing = max(1, int(min_passing_tests))

    def propose(self, name: str, version: str, *, steps: Sequence[str] = (),
                preconditions: Sequence[str] = (),
                inputs_schema: Mapping[str, Any] | None = None,
                outputs_schema: Mapping[str, Any] | None = None,
                dependencies: Sequence[str] = (),
                originating_episodes: Sequence[str] = ()) -> ProcedureSpec:
        """Record a draft procedure (Step 39)."""
        procedure = ProcedureSpec(
            procedure_id=new_id("proc"), name=name, version=version,
            steps=tuple(steps), preconditions=tuple(preconditions),
            inputs_schema=dict(inputs_schema or {}),
            outputs_schema=dict(outputs_schema or {}),
            dependencies=tuple(dependencies),
            originating_episodes=tuple(originating_episodes))
        self._procedures[procedure.procedure_id] = procedure
        self._tests[procedure.procedure_id] = []
        return procedure

    def attach_test(self, procedure_id: str, passed: bool,
                    detail: str = "") -> ProcedureTestRecord:
        """Attach one test result to a procedure (Step 39)."""
        self._require(procedure_id)
        record = ProcedureTestRecord(
            test_id=new_id("proctest"), procedure_id=procedure_id,
            passed=passed, detail=detail)
        self._tests[procedure_id].append(record)
        return record

    def passing_tests(self, procedure_id: str) -> int:
        """Return the count of passing tests for a procedure."""
        return sum(1 for record in self._tests.get(procedure_id, [])
                   if record.passed)

    def promote(self, procedure_id: str) -> ProcedureSpec:
        """Promote a draft procedure to validated, or refuse (Step 39).

        Promotion requires at least ``min_passing_tests`` passing tests;
        otherwise :class:`ProcedureNotPromotedError` is raised and the
        procedure stays a draft.
        """
        procedure = self._require(procedure_id)
        if procedure.status is not ProcedureStatus.DRAFT:
            raise ProcedureNotPromotedError(
                f"procedure {procedure_id} is {procedure.status.value}, not draft")
        if self.passing_tests(procedure_id) < self._min_passing:
            raise ProcedureNotPromotedError(
                f"procedure {procedure_id} has "
                f"{self.passing_tests(procedure_id)} passing tests; "
                f"{self._min_passing} required for promotion")
        promoted = ProcedureSpec(
            procedure_id=procedure.procedure_id, name=procedure.name,
            version=procedure.version, steps=procedure.steps,
            preconditions=procedure.preconditions,
            inputs_schema=procedure.inputs_schema,
            outputs_schema=procedure.outputs_schema,
            dependencies=procedure.dependencies,
            originating_episodes=procedure.originating_episodes,
            status=ProcedureStatus.VALIDATED, created_at=procedure.created_at)
        self._procedures[procedure_id] = promoted
        return promoted

    def supersede(self, procedure_id: str, successor_id: str) -> ProcedureSpec:
        """Mark a procedure superseded by an explicit successor (Steps 40, 42)."""
        procedure = self._require(procedure_id)
        self._require(successor_id)
        superseded = ProcedureSpec(
            procedure_id=procedure.procedure_id, name=procedure.name,
            version=procedure.version, steps=procedure.steps,
            preconditions=procedure.preconditions,
            inputs_schema=procedure.inputs_schema,
            outputs_schema=procedure.outputs_schema,
            dependencies=procedure.dependencies,
            originating_episodes=procedure.originating_episodes,
            status=ProcedureStatus.SUPERSEDED, created_at=procedure.created_at)
        self._procedures[procedure_id] = superseded
        return superseded

    def procedure(self, procedure_id: str) -> ProcedureSpec | None:
        """Return one procedure by ID."""
        return self._procedures.get(procedure_id)

    def procedures(self, *, status: ProcedureStatus | None = None
                   ) -> tuple[ProcedureSpec, ...]:
        """List procedures, optionally filtered by status."""
        items = list(self._procedures.values())
        if status is not None:
            items = [item for item in items if item.status is status]
        return tuple(items)

    def tests(self, procedure_id: str) -> tuple[ProcedureTestRecord, ...]:
        """Return the test history of a procedure."""
        return tuple(self._tests.get(procedure_id, ()))

    def _require(self, procedure_id: str) -> ProcedureSpec:
        """Return the procedure or raise."""
        procedure = self._procedures.get(procedure_id)
        if procedure is None:
            raise KeyError(f"unknown procedure: {procedure_id}")
        return procedure

    def describe(self) -> dict[str, Any]:
        """Return a serializable summary."""
        return {"procedures": len(self._procedures),
                "statuses": {procedure.procedure_id: procedure.status.value
                             for procedure in self._procedures.values()}}
