"""Typed request, status, evidence-tag, and result records for FRCE (Point 5, Section 3).

The engine must not collapse different kinds of evidence into a single ``true``
label. Every record below keeps the six epistemic categories distinct:

1. retrieved source fact;
2. rule-derived inference;
3. SAT/SMT satisfiability or unsatisfiability result;
4. optimization result (optimality or bounded/unknown status);
5. machine-checked proof under explicit assumptions;
6. empirical observation or prediction that is not formally proved.

``UNKNOWN``, timeout, unsupported theory and ``no proof found`` are first-class
statuses and are never reported as ``false``.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from .version import FORMAL_REASONING_RESULT_VERSION, FORMAL_REASONING_SCHEMA_VERSION


def canonical_json(value: Any) -> str:
    """Serialize nested values deterministically for fingerprints and digests."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    )


def sha256_json(value: Any) -> str:
    """Return a SHA-256 digest of the canonical JSON encoding of a value."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_text(text: str) -> str:
    """Return a SHA-256 digest of one UTF-8 string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ResultStatus(str, Enum):
    """The exhaustive, mutually distinct result statuses (Section 3).

    ``UNKNOWN``, ``TIMEOUT``, ``UNSUPPORTED``, ``INVALID_INPUT`` and ``ERROR``
    are grouped as non-success statuses: they never entail falsity of the
    statement under test.
    """

    PROVED = "proved"
    REFUTED_UNDER_MODEL = "refuted-under-model"
    SATISFIABLE = "satisfiable"
    UNSATISFIABLE = "unsatisfiable"
    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    UNKNOWN = "unknown"
    TIMEOUT = "timeout"
    UNSUPPORTED = "unsupported"
    INVALID_INPUT = "invalid-input"
    ERROR = "error"


NON_SUCCESS_STATUSES: frozenset[ResultStatus] = frozenset({
    ResultStatus.UNKNOWN,
    ResultStatus.TIMEOUT,
    ResultStatus.UNSUPPORTED,
    ResultStatus.INVALID_INPUT,
    ResultStatus.ERROR,
})


class ResultKind(str, Enum):
    """The six epistemic categories that must never be collapsed (Section 1)."""

    RETRIEVED_SOURCE_FACT = "retrieved_source_fact"
    RULE_DERIVED_INFERENCE = "rule_derived_inference"
    SATISFIABILITY_RESULT = "satisfiability_result"
    OPTIMIZATION_RESULT = "optimization_result"
    MACHINE_CHECKED_PROOF = "machine_checked_proof"
    EMPIRICAL_OBSERVATION = "empirical_observation"


class EvidenceTag(str, Enum):
    """Input classification carried through every proof and result (Step 5)."""

    OBSERVED_FACT = "observed_fact"                # observed / source-backed fact
    STIPULATED_ASSUMPTION = "stipulated_assumption"  # stipulated assumption
    INFERRED_RESULT = "inferred_result"            # previously derived result
    HYPOTHESIS = "hypothesis"                      # unverified hypothesis
    POLICY_CONSTRAINT = "policy_constraint"        # policy/authorization constraint


class TaskClass(str, Enum):
    """Reasoning problem classes routed by Step 2; not everything is 'theorem proving'."""

    RULE_INFERENCE = "rule_inference"
    BOOLEAN_CONSISTENCY = "boolean_consistency"
    ARITHMETIC_CONSTRAINTS = "arithmetic_constraints"
    COMBINATORIAL_OPTIMIZATION = "combinatorial_optimization"
    MATHEMATICAL_PROOF = "mathematical_proof"
    PROGRAM_PROPERTY_VERIFICATION = "program_property_verification"
    MIXED = "mixed"


class SolverStatus(str, Enum):
    """Raw solver outcomes kept distinct from result statuses (Step 26)."""

    SAT = "sat"
    UNSAT = "unsat"
    UNKNOWN = "unknown"
    TIMEOUT = "timeout"
    ERROR = "error"


class AssuranceLevel(str, Enum):
    """Requested assurance level; ``HIGH`` triggers proof obligations (Step 37)."""

    NONE = "none"
    BASIC = "basic"
    STANDARD = "standard"
    HIGH = "high"


class PermittedTheory(str, Enum):
    """Theory allowlist entries used by the solver-selection policy (Step 23)."""

    BOOL = "bool"
    LINEAR_INTEGER_ARITHMETIC = "lia"
    LINEAR_REAL_ARITHMETIC = "lra"
    NONLINEAR_ARITHMETIC = "nra"
    BITVECTORS = "bv"
    STRINGS = "strings"
    ARRAYS = "arrays"
    UNINTERPRETED_FUNCTIONS = "uf"
    DATETIME_INTERVALS = "temporal_intervals"


class TemporalScope:
    """Validity interval attached to time-scoped facts (Step 6)."""

    __slots__ = ("valid_from", "valid_until")

    def __init__(self, valid_from: int | None = None, valid_until: int | None = None) -> None:
        if valid_from is not None and valid_until is not None and valid_from > valid_until:
            raise ValueError("temporal scope requires valid_from <= valid_until")
        self.valid_from = valid_from
        self.valid_until = valid_until

    def overlaps(self, other: "TemporalScope") -> bool:
        """Return whether two validity intervals share at least one time point."""
        lower = max(
            self.valid_from if self.valid_from is not None else -(2**62),
            other.valid_from if other.valid_from is not None else -(2**62),
        )
        upper = min(
            self.valid_until if self.valid_until is not None else 2**62,
            other.valid_until if other.valid_until is not None else 2**62,
        )
        return lower <= upper

    def as_tuple(self) -> tuple[int | None, int | None]:
        """Return the raw interval pair for hashing."""
        return (self.valid_from, self.valid_until)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TemporalScope) and self.as_tuple() == other.as_tuple()

    def __hash__(self) -> int:
        return hash(self.as_tuple())

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"TemporalScope({self.valid_from}, {self.valid_until})"


@dataclass(frozen=True, slots=True)
class SourceRef:
    """One provenance link back to a USG record, source, or statement (Step 6)."""

    source_id: str
    record_id: str | None = None
    version_id: str | None = None
    timestamp: str | None = None
    authority: str | None = None
    locator: str | None = None

    def __post_init__(self) -> None:
        if not self.source_id.strip():
            raise ValueError("SourceRef.source_id must be non-empty")


@dataclass(frozen=True, slots=True)
class ResourceBudget:
    """Resource and time budgets supplied with a request (Step 47)."""

    wall_clock_ms: int = 30000
    solver_timeout_ms: int = 10000
    cpu_seconds: int = 30
    memory_mb: int = 1024
    max_output_bytes: int = 1_000_000
    max_retries: int = 1
    retry_backoff_ms: int = 100

    def __post_init__(self) -> None:
        for name in (
            "wall_clock_ms", "solver_timeout_ms", "cpu_seconds", "memory_mb",
            "max_output_bytes", "max_retries", "retry_backoff_ms",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"ResourceBudget.{name} must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class NamedAssumption:
    """A separately addressable assumption with a stable name (Step 25)."""

    name: str
    description: str
    tag: EvidenceTag = EvidenceTag.STIPULATED_ASSUMPTION
    source_refs: tuple[SourceRef, ...] = ()
    premise_status: str = "stipulated"  # stipulated | unvalidated | observed

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("NamedAssumption.name must be non-empty")
        if not self.description.strip():
            raise ValueError("NamedAssumption.description must be non-empty")
        if self.premise_status not in {"stipulated", "unvalidated", "observed"}:
            raise ValueError("premise_status must be stipulated, unvalidated, or observed")


@dataclass(frozen=True, slots=True)
class FormalizationRequest:
    """A formalization request contract (Point 5, Section 3, Step 1).

    The coordinator submits a narrowly stated objective: derive a fact, test
    consistency, verify a property, or optimize a finite decision.
    """

    request_id: str
    objective: str
    task_classes: tuple[TaskClass, ...]
    required_property: str | None = None        # conclusion/property to establish
    usg_entity_ids: tuple[str, ...] = ()        # selected USG identifiers
    source_refs: tuple[SourceRef, ...] = ()
    assumptions: tuple[NamedAssumption, ...] = ()
    scope_context: str = "default"
    temporal_scope: TemporalScope | None = None
    permitted_theories: frozenset[PermittedTheory] = frozenset({
        PermittedTheory.BOOL,
        PermittedTheory.LINEAR_INTEGER_ARITHMETIC,
        PermittedTheory.LINEAR_REAL_ARITHMETIC,
    })
    budget: ResourceBudget = ResourceBudget()
    assurance_level: AssuranceLevel = AssuranceLevel.STANDARD
    policy_context: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = FORMAL_REASONING_SCHEMA_VERSION
    optimization_problem: Any = None            # SelectionProblem when optimizing (Step 31)
    proof_obligation: Any = None                # ProofObligation input when proving (Step 37)
    clarification_rules: tuple[str, ...] = ()   # open-world / closed-world domain declarations

    def __post_init__(self) -> None:
        """Validate the request contract before any computation begins."""
        if not self.request_id.strip():
            raise ValueError("FormalizationRequest.request_id must be non-empty")
        if not self.objective.strip():
            raise ValueError("FormalizationRequest.objective must be non-empty")
        if not self.task_classes:
            raise ValueError("FormalizationRequest requires at least one task class")
        if len(set(self.task_classes)) != len(self.task_classes):
            raise ValueError("task classes must not repeat")
        if TaskClass.MIXED in self.task_classes and len(self.task_classes) > 1:
            raise ValueError("task class MIXED cannot be combined with specific classes")
        object.__setattr__(self, "usg_entity_ids", tuple(self.usg_entity_ids))
        object.__setattr__(self, "source_refs", tuple(self.source_refs))
        object.__setattr__(self, "assumptions", tuple(self.assumptions))
        object.__setattr__(self, "clarification_rules", tuple(self.clarification_rules))
        names = [item.name for item in self.assumptions]
        if len(names) != len(set(names)):
            raise ValueError("assumption names must be unique within a request")

    @property
    def fingerprint(self) -> str:
        """Return the deterministic fingerprint used as an idempotency key (Step 1)."""
        return sha256_json({
            "request_id": self.request_id,
            "objective": self.objective,
            "task_classes": [item.value for item in self.task_classes],
            "required_property": self.required_property,
            "usg_entity_ids": list(self.usg_entity_ids),
            "sources": [_source_ref_payload(item) for item in self.source_refs],
            "assumptions": [
                {"name": a.name, "description": a.description, "tag": a.tag.value,
                 "premise_status": a.premise_status}
                for a in self.assumptions
            ],
            "scope_context": self.scope_context,
            "temporal": list(self.temporal_scope.as_tuple()) if self.temporal_scope else None,
            "theories": sorted(item.value for item in self.permitted_theories),
            "assurance": self.assurance_level.value,
            "optimization": _safe_repr_payload(self.optimization_problem),
            "proof_obligation": _safe_repr_payload(self.proof_obligation),
            "clarification_rules": list(self.clarification_rules),
        })


def _source_ref_payload(ref: SourceRef) -> dict[str, Any]:
    """Return a stable payload for one provenance link."""
    return {
        "source_id": ref.source_id,
        "record_id": ref.record_id,
        "version_id": ref.version_id,
        "timestamp": ref.timestamp,
        "authority": ref.authority,
        "locator": ref.locator,
    }


def _safe_repr_payload(value: Any) -> Any:
    """Return a stable payload for optional problem objects used in fingerprints."""
    if value is None:
        return None
    to_payload = getattr(value, "fingerprint_payload", None)
    if callable(to_payload):
        return to_payload()
    return repr(value)


@dataclass(frozen=True, slots=True)
class SolverIdentity:
    """Solver name, version and configuration recorded with every result."""

    name: str
    version: str
    configuration: Mapping[str, Any] = field(default_factory=dict)
    backend: str = "in_process"

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.version.strip():
            raise ValueError("SolverIdentity requires non-empty name and version")


@dataclass(frozen=True, slots=True)
class TheoryProfile:
    """The theories actually used by an encoding (recorded with results)."""

    theories: tuple[PermittedTheory, ...]
    nonlinear: bool = False
    exact_arithmetic: bool = True
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "theories", tuple(self.theories))
        object.__setattr__(self, "notes", tuple(self.notes))


@dataclass(frozen=True, slots=True)
class RuntimeMeasurement:
    """Runtime/resource measurements attached to results (Section 3)."""

    wall_clock_ms: float = 0.0
    cpu_seconds: float = 0.0
    peak_memory_mb: float | None = None

    def __post_init__(self) -> None:
        if self.wall_clock_ms < 0 or self.cpu_seconds < 0:
            raise ValueError("runtime measurements cannot be negative")


@dataclass(frozen=True, slots=True)
class FormalResult:
    """One typed result record (Point 5, Section 3).

    ``status`` uses the exhaustive ``ResultStatus`` vocabulary; the ``kind``
    field preserves which epistemic category the result belongs to. Solver
    textual explanations are never treated as proofs: ``proof`` evidence must
    carry a machine-check status or be explicitly marked unverified.
    """

    result_id: str
    request_id: str
    kind: ResultKind
    status: ResultStatus
    statement: str                                # exact formal statement
    statement_ir_refs: tuple[str, ...] = ()       # normalized IR node references
    assumptions: tuple[NamedAssumption, ...] = ()
    theory_profile: TheoryProfile | None = None
    solver: SolverIdentity | None = None
    model: Mapping[str, Any] | None = None        # satisfying assignment, if any
    model_independently_validated: bool | None = None
    counterexample: Mapping[str, Any] | None = None
    unsat_core: tuple[str, ...] = ()
    unsat_core_minimized: bool = False
    derivation_refs: tuple[str, ...] = ()         # explanation-graph fact references
    proof_ref: str | None = None                  # proof artifact identifier
    proof_check_status: str | None = None         # kernel_checked | unverified | rejected | none
    provenance: tuple[SourceRef, ...] = ()
    runtime: RuntimeMeasurement = RuntimeMeasurement()
    limitations: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    reproducibility: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = FORMAL_REASONING_RESULT_VERSION

    def __post_init__(self) -> None:
        """Enforce the structural invariants that independent validation relies on."""
        if not self.result_id.strip() or not self.request_id.strip():
            raise ValueError("FormalResult requires non-empty result_id and request_id")
        if not self.statement.strip():
            raise ValueError("FormalResult.statement must be non-empty")
        if self.kind == ResultKind.MACHINE_CHECKED_PROOF:
            if self.status not in {ResultStatus.PROVED, ResultStatus.UNKNOWN,
                                   ResultStatus.TIMEOUT, ResultStatus.UNSUPPORTED,
                                   ResultStatus.INVALID_INPUT, ResultStatus.ERROR,
                                   ResultStatus.REFUTED_UNDER_MODEL}:
                raise ValueError(f"proof result cannot carry status {self.status.value}")
            if self.status == ResultStatus.PROVED and self.proof_check_status != "kernel_checked":
                raise ValueError(
                    "a PROVED result must carry proof_check_status='kernel_checked'; "
                    "unverified explanations are never proofs"
                )
        if self.status == ResultStatus.OPTIMAL and self.kind != ResultKind.OPTIMIZATION_RESULT:
            raise ValueError("only optimization results may claim OPTIMAL")
        if self.model is not None and self.model_independently_validated is None:
            raise ValueError(
                "a decoded model must record whether it was independently validated"
            )
        object.__setattr__(self, "statement_ir_refs", tuple(self.statement_ir_refs))
        object.__setattr__(self, "assumptions", tuple(self.assumptions))
        object.__setattr__(self, "unsat_core", tuple(self.unsat_core))
        object.__setattr__(self, "derivation_refs", tuple(self.derivation_refs))
        object.__setattr__(self, "provenance", tuple(self.provenance))
        object.__setattr__(self, "limitations", tuple(self.limitations))
        object.__setattr__(self, "warnings", tuple(self.warnings))

    @property
    def is_success(self) -> bool:
        """Return whether the status is a successful, established outcome."""
        return self.status not in NON_SUCCESS_STATUSES

    @property
    def establishes_false(self) -> bool:
        """Return whether the result establishes falsity (never true for non-success)."""
        return self.status in {
            ResultStatus.REFUTED_UNDER_MODEL,
            ResultStatus.UNSATISFIABLE,
            ResultStatus.INFEASIBLE,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible representation for audit and workspace use."""
        return {
            "schema_version": self.schema_version,
            "result_id": self.result_id,
            "request_id": self.request_id,
            "kind": self.kind.value,
            "status": self.status.value,
            "statement": self.statement,
            "statement_ir_refs": list(self.statement_ir_refs),
            "assumptions": [
                {"name": a.name, "description": a.description, "tag": a.tag.value,
                 "premise_status": a.premise_status}
                for a in self.assumptions
            ],
            "theory_profile": None if self.theory_profile is None else {
                "theories": [t.value for t in self.theory_profile.theories],
                "nonlinear": self.theory_profile.nonlinear,
                "exact_arithmetic": self.theory_profile.exact_arithmetic,
                "notes": list(self.theory_profile.notes),
            },
            "solver": None if self.solver is None else {
                "name": self.solver.name,
                "version": self.solver.version,
                "configuration": dict(self.solver.configuration),
                "backend": self.solver.backend,
            },
            "model": None if self.model is None else dict(self.model),
            "model_independently_validated": self.model_independently_validated,
            "counterexample": None if self.counterexample is None else dict(self.counterexample),
            "unsat_core": list(self.unsat_core),
            "unsat_core_minimized": self.unsat_core_minimized,
            "derivation_refs": list(self.derivation_refs),
            "proof_ref": self.proof_ref,
            "proof_check_status": self.proof_check_status,
            "provenance": [
                {f: getattr(item, f) for f in item.__slots__} for item in self.provenance
            ],
            "runtime": {
                "wall_clock_ms": self.runtime.wall_clock_ms,
                "cpu_seconds": self.runtime.cpu_seconds,
                "peak_memory_mb": self.runtime.peak_memory_mb,
            },
            "limitations": list(self.limitations),
            "warnings": list(self.warnings),
            "reproducibility": dict(self.reproducibility),
        }


__all__ = [
    "canonical_json",
    "sha256_json",
    "sha256_text",
    "ResultStatus",
    "NON_SUCCESS_STATUSES",
    "ResultKind",
    "EvidenceTag",
    "TaskClass",
    "SolverStatus",
    "AssuranceLevel",
    "PermittedTheory",
    "TemporalScope",
    "SourceRef",
    "ResourceBudget",
    "NamedAssumption",
    "FormalizationRequest",
    "SolverIdentity",
    "TheoryProfile",
    "RuntimeMeasurement",
    "FormalResult",
]
