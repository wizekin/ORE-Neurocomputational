"""Candidate, query, disposition, and utility schemas for graph hypothesis generation.

Graph candidates are *hypotheses only*: learned scores guide search and never establish
truth (ORE Point 8 core invariant). Every candidate carries component scores, provenance,
a status, and - after symbolic verification - a disposition with an evidence classification.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Mapping


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _candidate_id(subject: str, relation_type: str, obj: str, seed: str) -> str:
    digest = sha256(f"{subject}|{relation_type}|{obj}|{seed}".encode("utf-8")).hexdigest()
    return f"graph-candidate:{digest[:24]}"


class CandidateStatus(str):
    """Lifecycle states of a candidate hypothesis before verification."""

    PROPOSED = "proposed"
    RANKED = "ranked"
    ABSTAINED = "abstained"
    REJECTED_PRE_RANK = "rejected_pre_rank"
    PUBLISHED = "published"
    VERIFIED = "verified"


class CandidateDisposition(str):
    """Symbolic-verification dispositions assigned only by the verification phase."""

    SUPPORTED = "supported"
    REJECTED = "rejected"
    AMBIGUOUS = "ambiguous"
    UNVERIFIABLE = "unverifiable"


class EvidenceClassification(str):
    """How the evidence supporting a candidate was obtained."""

    DIRECT_SOURCE = "direct_source"
    RULE_DERIVED = "rule_derived"
    MODEL_ONLY = "model_only"
    ABSENT = "absent"


class CandidateRejectionReason(str):
    """Reasons a candidate is hard-rejected before or during ranking."""

    TYPE_VIOLATION = "type_violation"
    CARDINALITY_VIOLATION = "cardinality_violation"
    DUPLICATE = "duplicate"
    INVERSE_DUPLICATE = "inverse_duplicate"
    CONTRADICTION = "contradiction"
    PREVIOUSLY_REJECTED = "previously_rejected"


@dataclass(frozen=True, slots=True)
class ComponentScores:
    """Per-model raw scores preserved separately; they are never merged into one truth value."""

    embedding_score: float | None = None
    gnn_score: float | None = None
    baseline_score: float | None = None

    def available(self) -> tuple[str, ...]:
        """Return the component names with a numeric score attached."""
        names = []
        for name in ("baseline_score", "embedding_score", "gnn_score"):
            if getattr(self, name) is not None:
                names.append(name)
        return tuple(names)

    def as_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in self.available()}


@dataclass(frozen=True, slots=True)
class UtilityComponents:
    """Normalized components of the task utility score S(c) = w1*R+w2*E+w3*T+w4*N-w5*C."""

    relevance: float
    evidence: float
    type_compatibility: float
    novelty: float
    cost: float

    def __post_init__(self) -> None:
        for name in ("relevance", "evidence", "type_compatibility", "novelty", "cost"):
            value = getattr(self, name)
            if not isinstance(value, float) or not 0.0 <= value <= 1.0:
                raise ValueError(f"utility component {name} must be normalized into [0, 1]")

    def weighted(self, weights: Mapping[str, float]) -> float:
        """Apply the utility weights; hard semantic violations are handled before this call."""
        return (
            weights.get("relevance", 0.0) * self.relevance
            + weights.get("evidence", 0.0) * self.evidence
            + weights.get("type_compatibility", 0.0) * self.type_compatibility
            + weights.get("novelty", 0.0) * self.novelty
            - weights.get("cost", 0.0) * self.cost
        )


@dataclass(frozen=True, slots=True)
class GraphQueryAnchor:
    """One query anchor: an alias/label or an explicit stable node id."""

    label: str | None = None
    node_id: str | None = None
    node_kind: str | None = None

    def __post_init__(self) -> None:
        if (self.label is None) == (self.node_id is None):
            raise ValueError("exactly one of label or node_id must be provided")


class QueryAnchorResolution:
    """Result of resolving query anchors to stable node ids (ORE Point 8 step 25)."""

    def __init__(self, node_ids: tuple[str, ...], *, ambiguous_labels: tuple[str, ...] = ()) -> None:
        self.node_ids = node_ids
        self.ambiguous_labels = ambiguous_labels
        self.abstained = bool(ambiguous_labels)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"QueryAnchorResolution(nodes={len(self.node_ids)}, ambiguous={self.ambiguous_labels})"


@dataclass(frozen=True, slots=True)
class GraphQuery:
    """A bounded graph-intelligence query submitted to the candidate service."""

    anchor: GraphQueryAnchor
    relation_types: tuple[str, ...] = ()
    target_node_id: str | None = None
    task_hint: str = "link_discovery"
    budget_candidates: int = 32
    neighborhood_depth: int = 3

    def __post_init__(self) -> None:
        if self.budget_candidates < 1:
            raise ValueError("budget_candidates must be positive")
        if self.neighborhood_depth < 1 or self.neighborhood_depth > 6:
            raise ValueError("neighborhood_depth must be between 1 and 6")


@dataclass(slots=True)
class GraphCandidate:
    """One typed relation hypothesis; scores guide search and never establish truth."""

    subject_id: str
    relation_type: str
    object_id: str
    component_scores: ComponentScores = field(default_factory=ComponentScores)
    utility: float | None = None
    utility_components: UtilityComponents | None = None
    status: str = CandidateStatus.PROPOSED
    rejection_reasons: tuple[str, ...] = ()
    calibrated_probability: float | None = None
    calibrated: bool = False
    model_version: str = ""
    snapshot_hash: str = ""
    evidence_ids: tuple[str, ...] = ()
    source_ids: tuple[str, ...] = ()
    supporting_edge_ids: tuple[str, ...] = ()
    counter_edge_ids: tuple[str, ...] = ()
    explanation_paths: tuple[tuple[str, ...], ...] = ()
    disposition: str | None = None
    evidence_classification: str | None = None
    rules_applied: tuple[str, ...] = ()
    verifier_notes: tuple[str, ...] = ()
    created_at: str = field(default_factory=_utc_now)
    candidate_id: str = field(default="")

    def __post_init__(self) -> None:
        if not self.subject_id or not self.object_id:
            raise ValueError("candidate endpoints must be non-empty node ids")
        if not self.relation_type.strip():
            raise ValueError("relation_type must be non-empty")
        if not self.candidate_id:
            object.__setattr__(self, "candidate_id", _candidate_id(
                self.subject_id, self.relation_type, self.object_id, self.created_at,
            ))

    def triple(self) -> tuple[str, str, str]:
        return (self.subject_id, self.relation_type, self.object_id)

    def as_payload(self) -> dict[str, Any]:
        """Return the JSON-compatible workspace payload for hypothesis publication."""
        return {
            "candidate_id": self.candidate_id,
            "triple": list(self.triple()),
            "status": self.status,
            "component_scores": self.component_scores.as_dict(),
            "utility": self.utility,
            "utility_components": (
                {
                    "relevance": self.utility_components.relevance,
                    "evidence": self.utility_components.evidence,
                    "type_compatibility": self.utility_components.type_compatibility,
                    "novelty": self.utility_components.novelty,
                    "cost": self.utility_components.cost,
                }
                if self.utility_components is not None else None
            ),
            "calibrated_probability": self.calibrated_probability,
            "calibrated": self.calibrated,
            "model_version": self.model_version,
            "snapshot_hash": self.snapshot_hash,
            "disposition": self.disposition,
            "evidence_classification": self.evidence_classification,
            "evidence_ids": list(self.evidence_ids),
            "source_ids": list(self.source_ids),
            "supporting_edge_ids": list(self.supporting_edge_ids),
            "counter_edge_ids": list(self.counter_edge_ids),
            "rules_applied": list(self.rules_applied),
            "verifier_notes": list(self.verifier_notes),
            "created_at": self.created_at,
        }


def new_hypothesis_namespace(model_version: str, snapshot_hash: str) -> str:
    """Build the dedicated workspace hypothesis namespace for one model/snapshot pair."""
    digest = sha256(f"{model_version}|{snapshot_hash}".encode("utf-8")).hexdigest()[:16]
    return f"graph_hypotheses:{digest}"


__all__ = [
    "CandidateDisposition", "CandidateRejectionReason", "CandidateStatus", "ComponentScores",
    "EvidenceClassification", "GraphCandidate", "GraphQuery", "GraphQueryAnchor",
    "QueryAnchorResolution", "UtilityComponents", "new_hypothesis_namespace",
]
