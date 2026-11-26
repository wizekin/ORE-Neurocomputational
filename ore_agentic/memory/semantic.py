"""Semantic memory through the canonical USG path (Steps 37, 42).

USG remains canonical: agent output can never directly modify authoritative
knowledge.  Proposals are recorded as lifecycle candidates and can only be
promoted through the canonical USG bundle import path (validation and
rights-policy filtering), never by writing into the live resource.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from ..contracts import new_id, utc_now

__all__ = (
    "LifecycleState",
    "PromotionReview",
    "SemanticMemory",
    "SemanticProposal",
    "USGPromotionGate",
)


class LifecycleState(str, Enum):
    """Lifecycle of a semantic proposal, mirroring USG states (Step 37)."""

    CANDIDATE = "CANDIDATE"
    VALIDATED_FOR_IMPORT = "VALIDATED_FOR_IMPORT"
    ACCEPTED = "ACCEPTED"
    DEPRECATED = "DEPRECATED"
    REJECTED = "REJECTED"


@dataclass(frozen=True, slots=True)
class SemanticProposal:
    """A proposed semantic fact originating from agent output (Step 37).

    A proposal is a hypothesis with provenance, never an authoritative fact.
    """

    proposal_id: str
    statement: str
    concepts: tuple[str, ...] = ()
    relations: tuple[tuple[str, str, str], ...] = ()
    evidence_refs: tuple[str, ...] = ()
    origin_workflow_id: str = ""
    origin_task_id: str = ""
    lifecycle: LifecycleState = LifecycleState.CANDIDATE
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        """Validate the proposal envelope."""
        if not self.statement or not self.statement.strip():
            raise ValueError("statement must be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the proposal."""
        return {
            "proposal_id": self.proposal_id,
            "statement": self.statement,
            "concepts": list(self.concepts),
            "relations": [list(relation) for relation in self.relations],
            "evidence_refs": list(self.evidence_refs),
            "origin_workflow_id": self.origin_workflow_id,
            "origin_task_id": self.origin_task_id,
            "lifecycle": self.lifecycle.value,
            "created_at": self.created_at,
        }


@dataclass(frozen=True, slots=True)
class PromotionReview:
    """The gate's decision on a semantic proposal (Step 37)."""

    eligible: bool
    blockers: tuple[str, ...]
    required_path: str = "canonical_usg_bundle_import"

    def to_dict(self) -> dict[str, Any]:
        """Serialize the review."""
        return {"eligible": self.eligible, "blockers": list(self.blockers),
                "required_path": self.required_path}


class USGPromotionGate:
    """The canonical promotion gate for semantic proposals (Step 37).

    Even a fully evidenced proposal can only be marked
    ``VALIDATED_FOR_IMPORT``; authoritative acceptance happens exclusively
    through the canonical USG bundle import path (structural validation,
    rights-policy filtering, operator approval), never by mutating the live
    USG resource from agent code.
    """

    def __init__(self, *, require_evidence: bool = True,
                 require_relations: bool = False) -> None:
        """Configure the gate's mandatory predicates."""
        self._require_evidence = require_evidence
        self._require_relations = require_relations

    def review(self, proposal: SemanticProposal) -> PromotionReview:
        """Return the blockers preventing promotion, or eligibility."""
        blockers: list[str] = []
        if self._require_evidence and not proposal.evidence_refs:
            blockers.append("proposal lacks evidence references")
        if self._require_relations and not proposal.relations:
            blockers.append("proposal declares no typed relations")
        if proposal.lifecycle in (LifecycleState.REJECTED, LifecycleState.DEPRECATED):
            blockers.append(f"proposal lifecycle is {proposal.lifecycle.value}")
        return PromotionReview(eligible=not blockers, blockers=tuple(blockers))

    def validate_for_import(self, proposal: SemanticProposal) -> SemanticProposal:
        """Mark an eligible proposal VALIDATED_FOR_IMPORT (Step 37).

        This transitions the proposal record only.  It does not, and cannot,
        modify the USG resource itself.
        """
        review = self.review(proposal)
        if not review.eligible:
            raise ValueError(
                f"proposal {proposal.proposal_id} not eligible: {list(review.blockers)}")
        return SemanticProposal(
            proposal_id=proposal.proposal_id, statement=proposal.statement,
            concepts=proposal.concepts, relations=proposal.relations,
            evidence_refs=proposal.evidence_refs,
            origin_workflow_id=proposal.origin_workflow_id,
            origin_task_id=proposal.origin_task_id,
            lifecycle=LifecycleState.VALIDATED_FOR_IMPORT,
            created_at=proposal.created_at)


class SemanticMemory:
    """Read-only USG retrieval plus candidate proposals (Steps 37, 42).

    The memory exposes canonical lookup through the USG index and stores
    agent proposals separately; there is deliberately no write path into the
    USG resource.
    """

    def __init__(self, usg_index: Any = None,
                 gate: USGPromotionGate | None = None) -> None:
        """Wrap the canonical index (duck-typed) and the promotion gate."""
        self._usg_index = usg_index
        self._gate = gate or USGPromotionGate()
        self._proposals: dict[str, SemanticProposal] = {}

    @property
    def usg_index(self) -> Any:
        """Return the wrapped canonical index (read-only usage)."""
        return self._usg_index

    def lookup(self, query: str) -> tuple[Mapping[str, Any], ...]:
        """Look up canonical semantic records with provenance (Step 37)."""
        if self._usg_index is None:
            return ()
        try:
            result = self._usg_index.lookup(query)
        except Exception:  # noqa: BLE001 - external index is untrusted input
            return ()
        return tuple(
            {"record_id": item.record_id, "label": item.label,
             "kind": item.kind,
             "source_reference": getattr(item, "source_reference", "")}
            for item in getattr(result, "items", ()) or ())

    def propose(self, statement: str, *, concepts: tuple[str, ...] = (),
                relations: tuple[tuple[str, str, str], ...] = (),
                evidence_refs: tuple[str, ...] = (),
                origin_workflow_id: str = "",
                origin_task_id: str = "") -> SemanticProposal:
        """Record an agent proposal as a lifecycle candidate (Step 37)."""
        proposal = SemanticProposal(
            proposal_id=new_id("semprop"), statement=statement, concepts=concepts,
            relations=relations, evidence_refs=evidence_refs,
            origin_workflow_id=origin_workflow_id, origin_task_id=origin_task_id)
        self._proposals[proposal.proposal_id] = proposal
        return proposal

    def proposal(self, proposal_id: str) -> SemanticProposal | None:
        """Return one stored proposal."""
        return self._proposals.get(proposal_id)

    def proposals(self) -> tuple[SemanticProposal, ...]:
        """Return all stored proposals in insertion order."""
        return tuple(self._proposals.values())

    @property
    def gate(self) -> USGPromotionGate:
        """Return the canonical promotion gate."""
        return self._gate

    def describe(self) -> dict[str, Any]:
        """Return a serializable summary."""
        return {
            "canonical": self._usg_index is not None,
            "proposals": len(self._proposals),
            "proposal_lifecycle": {
                proposal.proposal_id: proposal.lifecycle.value
                for proposal in self._proposals.values()},
        }
