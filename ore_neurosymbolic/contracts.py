"""Typed contracts for the ORE Neuro-Symbolic Reasoning Layer (Point 3, Section 4).

Every record carries an explicit schema version and validates itself at the
boundary. The invariants from Section 2 of ``ORE —Neurocomputational Point 03
Neuro-Symbolic Reasoning Algorithm.txt`` are enforced structurally here:

- observations, hypotheses, model scores, probabilistic beliefs, strict facts
  and formally derived conclusions are separate ``EpistemicStatus`` values and
  are never collapsed;
- ``UNKNOWN`` is a first-class status distinct from negation (``polarity=False``);
- learned models can only produce ``MODEL_PROPOSAL`` records, never facts;
- every conclusion keeps its premises, rule ID/version, source lineage,
  workspace revision and inference-run ID.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from ore_workspace.models import new_id, utc_now

NEUROSYMBOLIC_PACKAGE_VERSION = "1.0.0"
NEUROSYMBOLIC_SCHEMA_VERSION = "1.0.0"
NSRL_PRODUCER = "ore_neurosymbolic"
RULESET_CONTRACT = "ORE.NSRL.RULESET/1.0.0"


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


class EpistemicStatus(str, Enum):
    """The seven distinct epistemic categories (Sections 2 and 5, Step A6).

    ``UNKNOWN`` means no evidence exists either way; it is never equivalent to
    a negated fact and missing evidence is never stored as negative evidence.
    """

    OBSERVED = "observed_evidence"
    ASSERTED = "asserted_fact"
    DERIVED = "derived_fact"
    HYPOTHESIS = "hypothesis"
    PROBABILISTIC = "probabilistic_claim"
    MODEL_PROPOSAL = "model_proposal"
    UNKNOWN = "unknown"


class RuleClass(str, Enum):
    """Strict rules fire as a logical gate; defeasible rules admit exceptions."""

    STRICT = "strict"
    DEFEASIBLE = "defeasible"


class ProofClass(str, Enum):
    """The five proof labels that must never be conflated (Step C5)."""

    STRICT_VERIFIED = "strict_verified"
    DEFEASIBLE = "defeasible"
    PROBABILISTIC_SUPPORT = "probabilistic_support"
    NEURAL_PROPOSAL = "neural_proposal"
    UNVERIFIED_CANDIDATE = "unverified_candidate"


class VerificationStatus(str, Enum):
    """Independent-verifier outcome for one proof node (Step C3)."""

    UNVERIFIED = "unverified"
    VERIFIED = "verified"
    TAMPERED = "tampered"
    FAILED = "failed"


class CandidateStatus(str, Enum):
    """Constraint pipeline status of one candidate interpretation (Phase D)."""

    GENERATED = "generated"
    HARD_FILTERED = "hard_filtered"
    RANKED = "ranked"
    ALTERNATIVE = "alternative_preserved"
    SELECTED = "selected"


class InferenceMode(str, Enum):
    """Forward chaining, goal-directed backward chaining, or both."""

    FORWARD = "forward"
    BACKWARD = "backward"
    HYBRID = "hybrid"


class VerificationLevel(str, Enum):
    """How much independent verification the request demands (Step A1)."""

    NONE = "none"
    BASIC = "basic"
    STRICT = "strict"


class PublicationState(str, Enum):
    """Publication-gate lifecycle of one reasoning finding (Phase G)."""

    UNREVIEWED = "unreviewed"
    GATED_OUT = "gated_out"
    PUBLISHED = "published"
    REVIEW_REQUESTED = "review_requested"
    DEFERRED = "deferred"


@dataclass(frozen=True, slots=True)
class LogicalTerm:
    """A typed constant term (individual, value or concept identifier)."""

    value: str
    type: str = "any"

    def __post_init__(self) -> None:
        """Reject empty constants and unknown types."""
        if not isinstance(self.value, str) or not self.value:
            raise ValueError("LogicalTerm.value must be a non-empty string")
        if not isinstance(self.type, str) or not self.type:
            raise ValueError("LogicalTerm.type must be a non-empty string")

    def to_dict(self) -> dict[str, str]:
        """Return the JSON-compatible record."""
        return {"kind": "term", "value": self.value, "type": self.type}


@dataclass(frozen=True, slots=True)
class LogicalVariable:
    """A typed logical variable used inside rule premises and conclusions."""

    name: str
    type: str = "any"

    def __post_init__(self) -> None:
        """Reject empty or reserved variable names."""
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("LogicalVariable.name must be a non-empty string")
        if not isinstance(self.type, str) or not self.type:
            raise ValueError("LogicalVariable.type must be a non-empty string")

    def to_dict(self) -> dict[str, str]:
        """Return the JSON-compatible record."""
        return {"kind": "variable", "name": self.name, "type": self.type}


TermOrVariable = LogicalTerm | LogicalVariable


def parse_term(record: Any) -> TermOrVariable:
    """Parse one term record from a mapping or pass an already-typed term through."""
    if isinstance(record, (LogicalTerm, LogicalVariable)):
        return record
    if isinstance(record, Mapping) and record.get("kind") == "variable":
        return LogicalVariable(str(record["name"]), str(record.get("type", "any")))
    if isinstance(record, Mapping):
        return LogicalTerm(str(record["value"]), str(record.get("type", "any")))
    raise TypeError(f"cannot parse logical term from {record!r}")


def term_key(term: TermOrVariable) -> str:
    """Return the canonical key used for indexing and hashing one term."""
    if isinstance(term, LogicalVariable):
        return f"?{term.name}"
    return term.value


@dataclass(frozen=True, slots=True)
class Literal:
    """A typed predicate application with polarity and an optional temporal scope."""

    predicate: str
    args: tuple[TermOrVariable, ...]
    negated: bool = False
    temporal_qualifier: str | None = None

    def __post_init__(self) -> None:
        """Validate the predicate name, argument types and temporal qualifier."""
        if not isinstance(self.predicate, str) or not self.predicate:
            raise ValueError("Literal.predicate must be a non-empty string")
        if not isinstance(self.args, tuple) or not all(
            isinstance(item, (LogicalTerm, LogicalVariable)) for item in self.args
        ):
            raise TypeError("Literal.args must be a tuple of LogicalTerm/LogicalVariable")
        if not isinstance(self.negated, bool):
            raise TypeError("Literal.negated must be a boolean")
        if self.temporal_qualifier is not None and not isinstance(self.temporal_qualifier, str):
            raise TypeError("Literal.temporal_qualifier must be a string when supplied")

    @property
    def arity(self) -> int:
        """Return the number of arguments."""
        return len(self.args)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "predicate": self.predicate,
            "args": [item.to_dict() for item in self.args],
            "negated": self.negated,
            "temporal_qualifier": self.temporal_qualifier,
        }

    @staticmethod
    def from_dict(record: Mapping[str, Any]) -> "Literal":
        """Rebuild one literal from its ``to_dict`` form."""
        return Literal(
            predicate=str(record["predicate"]),
            args=tuple(parse_term(item) for item in record.get("args", ())),
            negated=bool(record.get("negated", False)),
            temporal_qualifier=record.get("temporal_qualifier"),
        )

    def signature(self, substitution: Mapping[str, Any] | None = None) -> str:
        """Return the canonical signature used for deduplication and hashing.

        ``substitution`` maps variable names to terms or to already-canonical
        term-key strings (as returned by ``Substitution.as_dict``).
        """
        parts = []
        for item in self.args:
            resolved: Any = item
            if isinstance(item, LogicalVariable) and substitution:
                resolved = substitution.get(item.name, item)
            parts.append(
                resolved if isinstance(resolved, str) else term_key(resolved)
            )
        prefix = "not:" if self.negated else ""
        temporal = f"@{self.temporal_qualifier}" if self.temporal_qualifier else ""
        return f"{prefix}{self.predicate}({','.join(parts)}){temporal}"


@dataclass(frozen=True, slots=True)
class SemanticFact:
    """One typed, versioned fact with an explicit epistemic status (Section 4)."""

    fact_id: str
    subject: str
    predicate: str
    object_value: str
    qualifiers: Mapping[str, Any] = field(default_factory=dict)
    polarity: bool = True
    epistemic_status: EpistemicStatus = EpistemicStatus.ASSERTED
    source_refs: tuple[str, ...] = ()
    validity: Mapping[str, Any] | None = None
    usg_version: str | None = None

    def __post_init__(self) -> None:
        """Validate identifiers, value types, polarity and provenance."""
        if not isinstance(self.fact_id, str) or not self.fact_id:
            raise ValueError("SemanticFact.fact_id must be a non-empty string")
        for name in ("subject", "predicate", "object_value"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"SemanticFact.{name} must be a non-empty string")
        if not isinstance(self.polarity, bool):
            raise TypeError("SemanticFact.polarity must be a boolean")
        if not isinstance(self.epistemic_status, EpistemicStatus):
            raise TypeError("SemanticFact.epistemic_status must be an EpistemicStatus")
        if self.source_refs and not all(
            isinstance(item, str) and item for item in self.source_refs
        ):
            raise ValueError("SemanticFact.source_refs must contain non-empty strings")

    def literal(self) -> Literal:
        """Project the fact onto the typed logical literal used by the engine."""
        return Literal(
            predicate=self.predicate,
            args=(LogicalTerm(self.subject, "entity"), LogicalTerm(self.object_value, "entity")),
            negated=not self.polarity,
            temporal_qualifier=(
                str(self.qualifiers.get("temporal"))
                if self.qualifiers.get("temporal") is not None else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "fact_id": self.fact_id,
            "subject": self.subject,
            "predicate": self.predicate,
            "object_value": self.object_value,
            "qualifiers": dict(self.qualifiers),
            "polarity": self.polarity,
            "epistemic_status": self.epistemic_status.value,
            "source_refs": list(self.source_refs),
            "validity": dict(self.validity) if self.validity else None,
            "usg_version": self.usg_version,
        }

    @staticmethod
    def from_dict(record: Mapping[str, Any]) -> "SemanticFact":
        """Rebuild one fact from its ``to_dict`` form."""
        return SemanticFact(
            fact_id=str(record["fact_id"]),
            subject=str(record["subject"]),
            predicate=str(record["predicate"]),
            object_value=str(record["object_value"]),
            qualifiers=dict(record.get("qualifiers", {})),
            polarity=bool(record.get("polarity", True)),
            epistemic_status=EpistemicStatus(str(record.get("epistemic_status", "asserted_fact"))),
            source_refs=tuple(record.get("source_refs", ())),
            validity=record.get("validity"),
            usg_version=record.get("usg_version"),
        )

    def fingerprint(self) -> str:
        """Return the content digest used for deduplication."""
        return sha256_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class RuleDefinition:
    """One versioned inference rule with class, priority, scope and exceptions."""

    rule_id: str
    version: str
    typed_variables: Mapping[str, str]
    premises: tuple[Literal, ...]
    conclusion: Literal
    rule_class: RuleClass = RuleClass.STRICT
    priority: float = 0.5
    scope: Mapping[str, Any] = field(default_factory=dict)
    exceptions: tuple[Literal, ...] = ()
    safety_policy: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate identifiers, priority bounds, premise/conclusion shape."""
        if not isinstance(self.rule_id, str) or not self.rule_id:
            raise ValueError("RuleDefinition.rule_id must be a non-empty string")
        if not isinstance(self.version, str) or not self.version:
            raise ValueError("RuleDefinition.version must be a non-empty string")
        if not isinstance(self.rule_class, RuleClass):
            raise TypeError("RuleDefinition.rule_class must be a RuleClass")
        if not 0.0 <= self.priority <= 1.0:
            raise ValueError("RuleDefinition.priority must be in [0, 1]")
        if not isinstance(self.premises, tuple) or not self.premises:
            raise ValueError("RuleDefinition.premises must be a non-empty tuple of Literal")
        if not all(isinstance(item, Literal) for item in self.premises):
            raise TypeError("RuleDefinition.premises must contain only Literal records")
        if not isinstance(self.conclusion, Literal):
            raise TypeError("RuleDefinition.conclusion must be one Literal")
        if not all(isinstance(item, Literal) for item in self.exceptions):
            raise TypeError("RuleDefinition.exceptions must contain only Literal records")
        if not isinstance(self.typed_variables, Mapping):
            raise TypeError("RuleDefinition.typed_variables must be a mapping")
        for name, variable_type in self.typed_variables.items():
            if not isinstance(name, str) or not name.startswith("?"):
                raise ValueError("typed_variables keys must be variable names starting with '?'")
            if not isinstance(variable_type, str) or not variable_type:
                raise ValueError("typed_variables values must be non-empty type names")

    @property
    def rule_ref(self) -> tuple[str, str]:
        """Return the ``(rule_id, version)`` pair recorded on proof nodes."""
        return (self.rule_id, self.version)

    def declared_variables(self) -> set[str]:
        """Return the set of variable names declared for this rule."""
        declared = set(self.typed_variables)
        for premise in self.premises:
            declared.update(
                item.name for item in premise.args if isinstance(item, LogicalVariable)
            )
        declared.update(
            item.name for item in self.conclusion.args if isinstance(item, LogicalVariable)
        )
        return declared

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "rule_id": self.rule_id,
            "version": self.version,
            "typed_variables": dict(self.typed_variables),
            "premises": [item.to_dict() for item in self.premises],
            "conclusion": self.conclusion.to_dict(),
            "rule_class": self.rule_class.value,
            "priority": self.priority,
            "scope": dict(self.scope),
            "exceptions": [item.to_dict() for item in self.exceptions],
            "safety_policy": dict(self.safety_policy),
            "provenance": dict(self.provenance),
        }

    @staticmethod
    def from_dict(record: Mapping[str, Any]) -> "RuleDefinition":
        """Rebuild one rule from its ``to_dict`` form."""
        return RuleDefinition(
            rule_id=str(record["rule_id"]),
            version=str(record["version"]),
            typed_variables=dict(record.get("typed_variables", {})),
            premises=tuple(Literal.from_dict(item) for item in record.get("premises", ())),
            conclusion=Literal.from_dict(record["conclusion"]),
            rule_class=RuleClass(str(record.get("rule_class", "strict"))),
            priority=float(record.get("priority", 0.5)),
            scope=dict(record.get("scope", {})),
            exceptions=tuple(Literal.from_dict(item) for item in record.get("exceptions", ())),
            safety_policy=dict(record.get("safety_policy", {})),
            provenance=dict(record.get("provenance", {})),
        )

    def fingerprint(self) -> str:
        """Return the content digest that pins one rule version."""
        return sha256_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    """One evidence record with reliability, uncertainty model and lineage (Section 4)."""

    evidence_id: str
    source: str
    observation: Mapping[str, Any]
    reliability: float
    timestamp: str = field(default_factory=utc_now)
    uncertainty_model: Mapping[str, Any] = field(default_factory=dict)
    lineage: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate identifiers and the reliability range."""
        if not isinstance(self.evidence_id, str) or not self.evidence_id:
            raise ValueError("EvidenceItem.evidence_id must be a non-empty string")
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("EvidenceItem.source must be a non-empty string")
        if not isinstance(self.observation, Mapping):
            raise TypeError("EvidenceItem.observation must be a mapping")
        if not math.isfinite(self.reliability) or not 0.0 <= self.reliability <= 1.0:
            raise ValueError("EvidenceItem.reliability must be in [0, 1]")
        if not isinstance(self.uncertainty_model, Mapping):
            raise TypeError("EvidenceItem.uncertainty_model must be a mapping")

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "evidence_id": self.evidence_id,
            "source": self.source,
            "observation": dict(self.observation),
            "reliability": self.reliability,
            "timestamp": self.timestamp,
            "uncertainty_model": dict(self.uncertainty_model),
            "lineage": list(self.lineage),
        }

    @staticmethod
    def from_dict(record: Mapping[str, Any]) -> "EvidenceItem":
        """Rebuild one evidence item from its ``to_dict`` form."""
        return EvidenceItem(
            evidence_id=str(record["evidence_id"]),
            source=str(record["source"]),
            observation=dict(record.get("observation", {})),
            reliability=float(record.get("reliability", 0.0)),
            timestamp=str(record.get("timestamp", utc_now())),
            uncertainty_model=dict(record.get("uncertainty_model", {})),
            lineage=tuple(record.get("lineage", ())),
        )


@dataclass(frozen=True, slots=True)
class ConfidenceVector:
    """Decomposed confidence ``q = (q_source, q_observation, q_rule, q_coverage, q_conflict)``.

    Section 6.5 forbids collapsing this vector into one opaque score; callers
    that need a scalar must apply their own, explicitly declared aggregation.
    """

    q_source: float
    q_observation: float
    q_rule: float
    q_coverage: float
    q_conflict: float

    def __post_init__(self) -> None:
        """Require finite components in [0, 1]."""
        values = (self.q_source, self.q_observation, self.q_rule, self.q_coverage, self.q_conflict)
        for value in values:
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("ConfidenceVector components must be in [0, 1]")

    def to_dict(self) -> dict[str, float]:
        """Return the five named dimensions without collapsing them."""
        return {
            "q_source": self.q_source,
            "q_observation": self.q_observation,
            "q_rule": self.q_rule,
            "q_coverage": self.q_coverage,
            "q_conflict": self.q_conflict,
        }

    @staticmethod
    def from_dict(record: Mapping[str, Any]) -> "ConfidenceVector":
        """Rebuild one confidence vector from its ``to_dict`` form."""
        return ConfidenceVector(
            q_source=float(record["q_source"]),
            q_observation=float(record["q_observation"]),
            q_rule=float(record["q_rule"]),
            q_coverage=float(record["q_coverage"]),
            q_conflict=float(record["q_conflict"]),
        )


@dataclass(frozen=True, slots=True)
class CandidateInterpretation:
    """One candidate interpretation produced by Phase D (Section 4)."""

    candidate_id: str
    semantic_structure: Mapping[str, Any]
    evidence_refs: tuple[str, ...] = ()
    neural_scores: Mapping[str, float] = field(default_factory=dict)
    constraint_status: Mapping[str, Any] = field(default_factory=dict)
    ranking_features: Mapping[str, float] = field(default_factory=dict)
    status: CandidateStatus = CandidateStatus.GENERATED

    def __post_init__(self) -> None:
        """Validate the identifier, feature ranges and status type."""
        if not isinstance(self.candidate_id, str) or not self.candidate_id:
            raise ValueError("CandidateInterpretation.candidate_id must be a non-empty string")
        if not isinstance(self.semantic_structure, Mapping):
            raise TypeError("CandidateInterpretation.semantic_structure must be a mapping")
        for name, value in self.ranking_features.items():
            if not math.isfinite(float(value)):
                raise ValueError(f"ranking feature {name!r} must be finite")
        if not isinstance(self.status, CandidateStatus):
            raise TypeError("CandidateInterpretation.status must be a CandidateStatus")

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "candidate_id": self.candidate_id,
            "semantic_structure": dict(self.semantic_structure),
            "evidence_refs": list(self.evidence_refs),
            "neural_scores": dict(self.neural_scores),
            "constraint_status": dict(self.constraint_status),
            "ranking_features": dict(self.ranking_features),
            "status": self.status.value,
        }

    @staticmethod
    def from_dict(record: Mapping[str, Any]) -> "CandidateInterpretation":
        """Rebuild one candidate from its ``to_dict`` form."""
        return CandidateInterpretation(
            candidate_id=str(record["candidate_id"]),
            semantic_structure=dict(record.get("semantic_structure", {})),
            evidence_refs=tuple(record.get("evidence_refs", ())),
            neural_scores=dict(record.get("neural_scores", {})),
            constraint_status=dict(record.get("constraint_status", {})),
            ranking_features=dict(record.get("ranking_features", {})),
            status=CandidateStatus(str(record.get("status", "generated"))),
        )


@dataclass(frozen=True, slots=True)
class ProofNode:
    """One immutable, content-addressed derivation-DAG node (Section 4, Step C1).

    ``content_hash`` implements Section 6.8:
    ``h_v = H(conclusion || ruleVersion || h_p1 || ... || h_pn)``.
    """

    node_id: str
    conclusion: Literal
    rule_ref: tuple[str, str]
    premise_node_ids: tuple[str, ...]
    source_fact_ids: tuple[str, ...]
    run_id: str
    verifier_status: VerificationStatus = VerificationStatus.UNVERIFIED
    content_hash: str = ""

    def __post_init__(self) -> None:
        """Validate the node identity, references and verifier status."""
        if not isinstance(self.node_id, str) or not self.node_id:
            raise ValueError("ProofNode.node_id must be a non-empty string")
        if not isinstance(self.conclusion, Literal):
            raise TypeError("ProofNode.conclusion must be a Literal")
        if (
            not isinstance(self.rule_ref, tuple) or len(self.rule_ref) != 2
            or not all(isinstance(item, str) and item for item in self.rule_ref)
        ):
            raise ValueError("ProofNode.rule_ref must be a (rule_id, version) string pair")
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("ProofNode.run_id must be a non-empty string")
        if not isinstance(self.verifier_status, VerificationStatus):
            raise TypeError("ProofNode.verifier_status must be a VerificationStatus")

    def compute_content_hash(self, premise_hashes: tuple[str, ...]) -> str:
        """Compute ``h_v`` over conclusion, rule version and premise hashes (6.8).

        ``h_v = H(conclusion || ruleVersion || h_p1 || ... || h_pn)`` — the
        recorded substitution is reconstruction data for the verifier, not a
        hash input.
        """
        return sha256_json({
            "conclusion": self.conclusion.to_dict(),
            "rule_version": self.rule_ref[1],
            "premise_hashes": list(premise_hashes),
        })

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "node_id": self.node_id,
            "conclusion": self.conclusion.to_dict(),
            "rule_ref": [self.rule_ref[0], self.rule_ref[1]],
            "premise_node_ids": list(self.premise_node_ids),
            "source_fact_ids": list(self.source_fact_ids),
            "run_id": self.run_id,
            "verifier_status": self.verifier_status.value,
            "content_hash": self.content_hash,
        }


@dataclass(frozen=True, slots=True)
class InferenceRun:
    """One bounded inference-run record with outcome and metrics (Section 4)."""

    run_id: str
    workspace_revision: int
    ruleset_version: str
    config: Mapping[str, Any] = field(default_factory=dict)
    budgets: Mapping[str, Any] = field(default_factory=dict)
    timestamps: Mapping[str, str] = field(default_factory=dict)
    outcome: str = "running"
    metrics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate identity and revision bounds."""
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("InferenceRun.run_id must be a non-empty string")
        if not isinstance(self.workspace_revision, int) or self.workspace_revision < 0:
            raise ValueError("InferenceRun.workspace_revision must be a non-negative integer")
        if not isinstance(self.ruleset_version, str) or not self.ruleset_version:
            raise ValueError("InferenceRun.ruleset_version must be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "run_id": self.run_id,
            "workspace_revision": self.workspace_revision,
            "ruleset_version": self.ruleset_version,
            "config": dict(self.config),
            "budgets": dict(self.budgets),
            "timestamps": dict(self.timestamps),
            "outcome": self.outcome,
            "metrics": dict(self.metrics),
        }


@dataclass(frozen=True, slots=True)
class ReasoningFinding:
    """One publishable reasoning finding with proof reference and support profile."""

    finding_id: str
    conclusion: Literal
    epistemic_status: EpistemicStatus
    proof_ref: str | None
    support_profile: ConfidenceVector
    conflicts: tuple[str, ...] = ()
    publication_state: PublicationState = PublicationState.UNREVIEWED
    proof_class: ProofClass = ProofClass.UNVERIFIED_CANDIDATE
    source_fact_ids: tuple[str, ...] = ()
    run_id: str = ""

    def __post_init__(self) -> None:
        """Validate identity, statuses and proof references."""
        if not isinstance(self.finding_id, str) or not self.finding_id:
            raise ValueError("ReasoningFinding.finding_id must be a non-empty string")
        if not isinstance(self.conclusion, Literal):
            raise TypeError("ReasoningFinding.conclusion must be a Literal")
        if not isinstance(self.epistemic_status, EpistemicStatus):
            raise TypeError("ReasoningFinding.epistemic_status must be an EpistemicStatus")
        if not isinstance(self.support_profile, ConfidenceVector):
            raise TypeError("ReasoningFinding.support_profile must be a ConfidenceVector")
        if not isinstance(self.publication_state, PublicationState):
            raise TypeError("ReasoningFinding.publication_state must be a PublicationState")
        if not isinstance(self.proof_class, ProofClass):
            raise TypeError("ReasoningFinding.proof_class must be a ProofClass")

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "finding_id": self.finding_id,
            "conclusion": self.conclusion.to_dict(),
            "epistemic_status": self.epistemic_status.value,
            "proof_ref": self.proof_ref,
            "support_profile": self.support_profile.to_dict(),
            "conflicts": list(self.conflicts),
            "publication_state": self.publication_state.value,
            "proof_class": self.proof_class.value,
            "source_fact_ids": list(self.source_fact_ids),
            "run_id": self.run_id,
        }

    @staticmethod
    def from_dict(record: Mapping[str, Any]) -> "ReasoningFinding":
        """Rebuild one finding from its ``to_dict`` form."""
        return ReasoningFinding(
            finding_id=str(record["finding_id"]),
            conclusion=Literal.from_dict(record["conclusion"]),
            epistemic_status=EpistemicStatus(str(record["epistemic_status"])),
            proof_ref=record.get("proof_ref"),
            support_profile=ConfidenceVector.from_dict(record["support_profile"]),
            conflicts=tuple(record.get("conflicts", ())),
            publication_state=PublicationState(str(record.get("publication_state", "unreviewed"))),
            proof_class=ProofClass(str(record.get("proof_class", "unverified_candidate"))),
            source_fact_ids=tuple(record.get("source_fact_ids", ())),
            run_id=str(record.get("run_id", "")),
        )


@dataclass(frozen=True, slots=True)
class ModelProposal:
    """One validated learned-model response (Section 4, Phase F).

    ``score_semantics`` records what the scores mean (e.g. ``softmax_logit``,
    ``calibrated_probability``); raw softmax outputs are never treated as
    calibrated probabilities unless the adapter declares that semantics.
    """

    model_id: str
    model_version: str
    input_hash: str
    output_schema: str
    candidates: tuple[CandidateInterpretation, ...]
    score_semantics: str
    calibration: Mapping[str, Any] = field(default_factory=dict)
    limitations: tuple[str, ...] = ()
    latency_ms: float = 0.0

    def __post_init__(self) -> None:
        """Validate model identity, hashes, schema and candidate payloads."""
        for name in ("model_id", "model_version", "input_hash", "output_schema", "score_semantics"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"ModelProposal.{name} must be a non-empty string")
        if not isinstance(self.candidates, tuple) or not all(
            isinstance(item, CandidateInterpretation) for item in self.candidates
        ):
            raise TypeError("ModelProposal.candidates must be a tuple of CandidateInterpretation")
        if not isinstance(self.calibration, Mapping):
            raise TypeError("ModelProposal.calibration must be a mapping")
        if self.latency_ms < 0:
            raise ValueError("ModelProposal.latency_ms cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "input_hash": self.input_hash,
            "output_schema": self.output_schema,
            "candidates": [item.to_dict() for item in self.candidates],
            "score_semantics": self.score_semantics,
            "calibration": dict(self.calibration),
            "limitations": list(self.limitations),
            "latency_ms": self.latency_ms,
        }


@dataclass(frozen=True, slots=True)
class NSRLConflictRecord:
    """One preserved contradiction with provenance and resolution options (Step G2)."""

    conflict_id: str
    claims: tuple[Mapping[str, Any], ...]
    provenance: tuple[Mapping[str, Any], ...]
    priority: float
    time_scope: Mapping[str, Any] | None
    resolution_options: tuple[str, ...]
    status: str = "unresolved"
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        """Validate the claim list, priority and resolution options."""
        if not isinstance(self.conflict_id, str) or not self.conflict_id:
            raise ValueError("NSRLConflictRecord.conflict_id must be a non-empty string")
        if not self.claims or not isinstance(self.claims, tuple):
            raise ValueError("NSRLConflictRecord.claims must be a non-empty tuple")
        if not 0.0 <= self.priority <= 1.0:
            raise ValueError("NSRLConflictRecord.priority must be in [0, 1]")
        if not isinstance(self.resolution_options, tuple):
            raise TypeError("NSRLConflictRecord.resolution_options must be a tuple")

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "conflict_id": self.conflict_id,
            "claims": [dict(item) for item in self.claims],
            "provenance": [dict(item) for item in self.provenance],
            "priority": self.priority,
            "time_scope": dict(self.time_scope) if self.time_scope else None,
            "resolution_options": list(self.resolution_options),
            "status": self.status,
            "created_at": self.created_at,
        }


def make_id(prefix: str) -> str:
    """Return one repository-native identifier with the given prefix."""
    return new_id(prefix)


__all__ = [
    "NEUROSYMBOLIC_PACKAGE_VERSION",
    "NEUROSYMBOLIC_SCHEMA_VERSION",
    "NSRL_PRODUCER",
    "RULESET_CONTRACT",
    "CandidateInterpretation",
    "CandidateStatus",
    "ConfidenceVector",
    "EpistemicStatus",
    "EvidenceItem",
    "InferenceMode",
    "InferenceRun",
    "Literal",
    "LogicalTerm",
    "LogicalVariable",
    "ModelProposal",
    "NSRLConflictRecord",
    "ProofClass",
    "ProofNode",
    "PublicationState",
    "ReasoningFinding",
    "RuleClass",
    "RuleDefinition",
    "SemanticFact",
    "TermOrVariable",
    "VerificationLevel",
    "VerificationStatus",
    "canonical_json",
    "make_id",
    "parse_term",
    "sha256_json",
    "sha256_text",
    "term_key",
]
