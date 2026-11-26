"""Shared contracts for the ORE Human Idea-to-Algorithm Compiler (HIC, Point 16).

These types implement the canonical artifact and provenance contracts of the
Point 16 specification. Every artifact is versioned, deterministically
serializable, and digestable so that a full compilation lineage can be
reconstructed and audited. Uncertainty is never silently converted into fact:
unresolved items stay explicitly unresolved and high-impact assumptions block
unsafe downstream generation until they are confirmed.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

HIC_SCHEMA_VERSION = "0.1.0"
HIC_GENERATOR_VERSION = "ore-hic/0.1.0"


class HICError(Exception):
    """Base class for HIC pipeline errors."""


class SchemaError(HICError):
    """Raised when an artifact violates its schema."""


class GateViolation(HICError):
    """Raised when a pipeline stage is attempted before its gate is satisfied."""


class InputModality(str, Enum):
    """Supported input modalities (spec 2.1)."""

    TEXT = "TEXT"
    SPEECH_TRANSCRIPT = "SPEECH_TRANSCRIPT"
    DIAGRAM = "DIAGRAM"
    TABLE = "TABLE"
    CODE = "CODE"
    STRUCTURED_FILE = "STRUCTURED_FILE"
    MIXED = "MIXED"
    BCI_STREAM = "BCI_STREAM"


class QualityFlag(str, Enum):
    """Modality-specific quality warnings (spec Step 3)."""

    LOW_TRANSCRIPTION_CONFIDENCE = "LOW_TRANSCRIPTION_CONFIDENCE"
    OCR_UNCERTAINTY = "OCR_UNCERTAINTY"
    MISSING_PAGES = "MISSING_PAGES"
    ENCODING_CORRECTED = "ENCODING_CORRECTED"
    UNCONFIRMED_EXTRACTION = "UNCONFIRMED_EXTRACTION"
    EMPTY_INPUT = "EMPTY_INPUT"
    UNSUPPORTED_MODALITY = "UNSUPPORTED_MODALITY"


class FragmentKind(str, Enum):
    """Segment kinds for idea fragmentation (spec Step 5)."""

    GOAL = "GOAL"
    DESCRIPTION = "DESCRIPTION"
    EXAMPLE = "EXAMPLE"
    CONSTRAINT = "CONSTRAINT"
    PREFERENCE = "PREFERENCE"
    QUESTION = "QUESTION"
    FACT = "FACT"
    ACTION = "ACTION"


class DeliverableKind(str, Enum):
    """Requested deliverable kinds (spec Step 6)."""

    EXPLANATION = "EXPLANATION"
    MATH_MODEL = "MATH_MODEL"
    ALGORITHM = "ALGORITHM"
    PSEUDOCODE = "PSEUDOCODE"
    IMPLEMENTATION = "IMPLEMENTATION"
    TEST_SUITE = "TEST_SUITE"
    WORKFLOW = "WORKFLOW"
    COMBINATION = "COMBINATION"
    UNDETERMINED = "UNDETERMINED"


class Modality(str, Enum):
    """Linguistic modality distinguishing must/may/should statements (spec Step 10)."""

    MUST = "MUST"
    MAY = "MAY"
    SHOULD = "SHOULD"
    USUALLY = "USUALLY"
    DESCRIPTIVE = "DESCRIPTIVE"


class RequirementKind(str, Enum):
    PRIMARY = "PRIMARY"
    SECONDARY = "SECONDARY"
    PROHIBITED = "PROHIBITED"


class ConstraintKind(str, Enum):
    HARD = "HARD"
    SOFT = "SOFT"


class VariableRole(str, Enum):
    DECISION = "DECISION"
    OBSERVED = "OBSERVED"
    PARAMETER = "PARAMETER"
    LATENT = "LATENT"
    DERIVED = "DERIVED"


class MathStructureKind(str, Enum):
    CLASSIFICATION = "CLASSIFICATION"
    SEARCH = "SEARCH"
    OPTIMIZATION = "OPTIMIZATION"
    SCHEDULING = "SCHEDULING"
    ASSIGNMENT = "ASSIGNMENT"
    GRAPH = "GRAPH"
    PROBABILISTIC_INFERENCE = "PROBABILISTIC_INFERENCE"
    CONTROL = "CONTROL"
    SIMULATION = "SIMULATION"
    SYMBOLIC_TRANSFORMATION = "SYMBOLIC_TRANSFORMATION"
    NUMERICAL_COMPUTATION = "NUMERICAL_COMPUTATION"
    UNKNOWN = "UNKNOWN"


class SolverToolKind(str, Enum):
    SAT_SMT = "SAT_SMT"
    CONSTRAINT_PROGRAMMING = "CONSTRAINT_PROGRAMMING"
    LINEAR_INTEGER_PROGRAMMING = "LINEAR_INTEGER_PROGRAMMING"
    GRAPH_ALGORITHM = "GRAPH_ALGORITHM"
    NUMERICAL_OPTIMIZATION = "NUMERICAL_OPTIMIZATION"
    BAYESIAN_INFERENCE = "BAYESIAN_INFERENCE"
    SIMULATION = "SIMULATION"
    EXHAUSTIVE_ENUMERATION = "EXHAUSTIVE_ENUMERATION"
    HEURISTIC = "HEURISTIC"


class NodeType(str, Enum):
    """Required AlgoIR node types (spec 4.1)."""

    GOAL = "GOAL"
    ENTITY = "ENTITY"
    TYPE = "TYPE"
    VARIABLE = "VARIABLE"
    INPUT = "INPUT"
    OUTPUT = "OUTPUT"
    PRECONDITION = "PRECONDITION"
    POSTCONDITION = "POSTCONDITION"
    OPERATION = "OPERATION"
    CONDITION = "CONDITION"
    LOOP = "LOOP"
    DEPENDENCY = "DEPENDENCY"
    CONSTRAINT = "CONSTRAINT"
    INVARIANT = "INVARIANT"
    FAILURE_CASE = "FAILURE_CASE"
    COMPLEXITY = "COMPLEXITY"
    TEST = "TEST"
    PROVENANCE = "PROVENANCE"


class ValidationState(str, Enum):
    UNVALIDATED = "UNVALIDATED"
    VALID = "VALID"
    INVALID = "INVALID"
    REJECTED = "REJECTED"
    BLOCKED = "BLOCKED"


class AssumptionSource(str, Enum):
    USER = "USER"
    PROJECT = "PROJECT"
    COMPILER = "COMPILER"


class Impact(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class ArtifactKind(str, Enum):
    """Canonical compilation artifacts (spec 2.2)."""

    INPUT_RECORD = "INPUT_RECORD"
    SEMANTIC_GRAPH = "SEMANTIC_GRAPH"
    AMBIGUITY_SET = "AMBIGUITY_SET"
    REQUIREMENT_SPEC = "REQUIREMENT_SPEC"
    MATH_MODEL = "MATH_MODEL"
    ALGO_IR = "ALGO_IR"
    CANDIDATE_SET = "CANDIDATE_SET"
    VERIFICATION_REPORT = "VERIFICATION_REPORT"
    CODE_ARTIFACT = "CODE_ARTIFACT"
    EXECUTION_REPORT = "EXECUTION_REPORT"
    APPROVAL_RECORD = "APPROVAL_RECORD"
    LEARNING_RECORD = "LEARNING_RECORD"


class TerminationPolicy(str, Enum):
    BOUNDED = "BOUNDED"
    CONVERGENCE = "CONVERGENCE"
    NONTERMINATING_DECLARED = "NONTERMINATING_DECLARED"


class CandidateFamily(str, Enum):
    """Candidate algorithm families (spec Step 40)."""

    GREEDY_SELECTION = "GREEDY_SELECTION"
    BIPARTITE_MATCHING = "BIPARTITE_MATCHING"
    HUNGARIAN_ASSIGNMENT = "HUNGARIAN_ASSIGNMENT"
    EXHAUSTIVE_OPTIMAL = "EXHAUSTIVE_OPTIMAL"
    DYNAMIC_PROGRAMMING = "DYNAMIC_PROGRAMMING"
    GRAPH_SEARCH = "GRAPH_SEARCH"
    LINEAR_INTEGER_PROGRAMMING = "LINEAR_INTEGER_PROGRAMMING"
    CONSTRAINT_PROGRAMMING = "CONSTRAINT_PROGRAMMING"
    GENETIC = "GENETIC"
    REINFORCEMENT_LEARNING = "REINFORCEMENT_LEARNING"
    ROUND_ROBIN = "ROUND_ROBIN"


class Capability(str, Enum):
    """Declared execution capabilities for resource access policy (spec 4.3)."""

    COMPUTE = "COMPUTE"
    MEMORY = "MEMORY"
    FILESYSTEM_READ = "FILESYSTEM_READ"
    FILESYSTEM_WRITE = "FILESYSTEM_WRITE"
    NETWORK = "NETWORK"
    SUBPROCESS = "SUBPROCESS"
    CREDENTIALS = "CREDENTIALS"


@dataclass(frozen=True, slots=True)
class SourceSpan:
    """A traceable location inside one immutable source artifact (spec Step 2)."""

    artifact_id: str
    start: int
    end: int
    text: str = ""

    def __post_init__(self) -> None:
        if not self.artifact_id:
            raise SchemaError("SourceSpan requires an artifact_id")
        if self.start < 0 or self.end < self.start:
            raise SchemaError(f"invalid span [{self.start}, {self.end})")

    def to_dict(self) -> dict[str, Any]:
        return {"artifact_id": self.artifact_id, "start": self.start, "end": self.end, "text": self.text}


@dataclass(frozen=True, slots=True)
class USGReference:
    """A citation of the USG concept, relation, or rule supporting an item (spec 3.1)."""

    concept_id: str
    rule_id: str
    mapping_type: str = "UNRESOLVED"
    lifecycle_state: str = "UNRESOLVED"

    def to_dict(self) -> dict[str, Any]:
        return {
            "concept_id": self.concept_id,
            "rule_id": self.rule_id,
            "mapping_type": self.mapping_type,
            "lifecycle_state": self.lifecycle_state,
        }


def _json_default(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    raise TypeError(f"not canonically serializable: {type(value)!r}")


def canonical_json(payload: Any) -> str:
    """Serialize deterministically: sorted keys, stable separators, no data loss."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=_json_default)


def digest_payload(payload: Any) -> str:
    """Return the SHA-256 digest of the canonical JSON serialization of ``payload``."""
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def digest_text(text: str) -> str:
    """Return the SHA-256 digest of raw text content, used for integrity checks."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class CanonicalArtifact:
    """Mixin-style base for canonical artifacts: versioned, digestable, serializable."""

    kind: ArtifactKind

    def content_payload(self) -> dict[str, Any]:
        """Return the digestable semantic content of the artifact.

        Timestamps and runtime identities are excluded by design: two compilations
        of the same versioned input under the same deterministic settings must
        produce the same content digest.
        """
        raise NotImplementedError

    @property
    def content_digest(self) -> str:
        return digest_payload(self.content_payload())

    @property
    def semantic_digest(self) -> str:
        """Digest over semantic content only, excluding housekeeping ids.

        Two compilations of the same versioned input under the same
        deterministic settings produce equal semantic digests even when their
        session-scoped artifact ids differ; approvals bind to this digest.
        """
        payload = self.content_payload()
        payload.pop("artifact_id", None)
        payload.pop("record_id", None)
        payload.pop("graph_id", None)
        payload.pop("spec_id", None)
        payload.pop("model_id", None)
        payload.pop("ir_id", None)
        payload.pop("candidate_set_id", None)
        payload.pop("code_artifact_id", None)
        payload.pop("execution_report_id", None)
        return digest_payload(payload)

    def to_dict(self) -> dict[str, Any]:
        """Return the full artifact dictionary including digest metadata."""
        kind_value = self.kind.value if hasattr(self.kind, "value") else self.kind
        base = {"kind": kind_value, "schema_version": HIC_SCHEMA_VERSION}
        base.update(self.content_payload())
        base["content_digest"] = self.content_digest
        return base

    def to_json(self) -> str:
        return canonical_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class ArtifactPointer:
    """A stable reference to one versioned artifact inside a session."""

    artifact_kind: str
    artifact_id: str
    version: int
    content_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": self.artifact_kind,
            "artifact_id": self.artifact_id,
            "version": self.version,
            "content_digest": self.content_digest,
        }


@dataclass(frozen=True, slots=True)
class UnitDeclaration:
    """A declared measurement unit attached to a value or type (spec Steps 10, 18)."""

    unit: str
    dimension: str

    def to_dict(self) -> dict[str, Any]:
        return {"unit": self.unit, "dimension": self.dimension}


def spans_to_dict(spans: Sequence[SourceSpan]) -> list[dict[str, Any]]:
    return [span.to_dict() for span in spans]


def ensure_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{field_name} must be a non-empty string")
    return value
