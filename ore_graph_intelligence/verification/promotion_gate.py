"""Candidate disposition, promotion gate, and verification results (steps 37-39).

The promotion gate assigns SUPPORTED / REJECTED / AMBIGUOUS / UNVERIFIABLE from evidence
and rules - never from scores alone - and gates supported candidates into an *authorized
update workflow*. The prediction service itself cannot write authoritative USG knowledge:
``AuthorizedUpdateProposal`` is a data object consumed by a separate authorized workflow,
and this module performs no USG writes at all.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping, Sequence

from ..projection.typed_graph import TypedGraphSnapshot
from ..schemas.candidate_schema import (
    CandidateDisposition,
    CandidateStatus,
    EvidenceClassification,
    GraphCandidate,
)
from .contradiction_search import ContradictionSearch, CounterEvidenceReport
from .evidence_checker import EvidenceCheck, EvidenceChecker
from .type_checker import TypeCheckResult, CandidateTypeChecker
from .usg_rule_adapter import RuleViolation, USGRuleAdapter


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass(frozen=True, slots=True)
class AuthorizedUpdateProposal:
    """A gated proposal for the authorized USG update workflow (steps 38-39).

    Carrying this object does NOT update USG: an authorized workflow must consume it and
    run its own authorization checks. The graph intelligence layer never writes authoritative
    knowledge directly.
    """

    candidate_id: str
    triple: tuple[str, str, str]
    evidence_ids: tuple[str, ...]
    supporting_edge_ids: tuple[str, ...]
    rules_applied: tuple[str, ...]
    model_version: str
    snapshot_hash: str
    proposed_at_utc: str
    requires_authorization: str = "authorized_usg_update_workflow"

    def as_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "triple": list(self.triple),
            "evidence_ids": list(self.evidence_ids),
            "supporting_edge_ids": list(self.supporting_edge_ids),
            "rules_applied": list(self.rules_applied),
            "model_version": self.model_version,
            "snapshot_hash": self.snapshot_hash,
            "proposed_at_utc": self.proposed_at_utc,
            "requires_authorization": self.requires_authorization,
        }


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """Full verification outcome for one candidate (step 37-39)."""

    candidate_id: str
    triple: tuple[str, str, str]
    disposition: str
    evidence_classification: str | None
    evidence_ids: tuple[str, ...]
    supporting_edge_ids: tuple[str, ...]
    counter_edge_ids: tuple[str, ...]
    rules_applied: tuple[str, ...]
    rule_violations: tuple[RuleViolation, ...]
    type_violations: tuple[str, ...]
    notes: tuple[str, ...]
    update_proposal: AuthorizedUpdateProposal | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "triple": list(self.triple),
            "disposition": self.disposition,
            "evidence_classification": self.evidence_classification,
            "evidence_ids": list(self.evidence_ids),
            "supporting_edge_ids": list(self.supporting_edge_ids),
            "counter_edge_ids": list(self.counter_edge_ids),
            "rules_applied": list(self.rules_applied),
            "rule_violations": [item.as_dict() for item in self.rule_violations],
            "type_violations": list(self.type_violations),
            "notes": list(self.notes),
            "update_proposal": self.update_proposal.as_dict() if self.update_proposal else None,
        }


class CandidateVerifier:
    """Verify candidates: types, rules, evidence, counter-evidence, disposition (steps 33-39)."""

    def __init__(
        self,
        snapshot: TypedGraphSnapshot,
        *,
        rejected_ledger: Mapping[tuple[str, str, str], str] | None = None,
    ) -> None:
        self.snapshot = snapshot
        self.type_checker = CandidateTypeChecker(snapshot)
        self.rule_adapter = USGRuleAdapter(snapshot)
        self.evidence_checker = EvidenceChecker(snapshot)
        self.contradiction_search = ContradictionSearch(
            snapshot, rejected_ledger=rejected_ledger,
        )

    def verify(self, candidate: GraphCandidate) -> VerificationResult:
        """Assign a disposition from evidence and rules; scores are never sufficient."""
        type_check: TypeCheckResult = self.type_checker.check(candidate)
        violations: list[RuleViolation] = self.rule_adapter.check(candidate)
        evidence: EvidenceCheck = self.evidence_checker.check(candidate)
        counter: CounterEvidenceReport = self.contradiction_search.search(candidate)
        notes: list[str] = []

        if not type_check.valid:
            notes.append("type check failed; rejected regardless of learned score")
            disposition = CandidateDisposition.REJECTED
        elif violations:
            notes.append("logical rule violations detected")
            disposition = CandidateDisposition.REJECTED
        elif counter.counter_edge_ids:
            notes.append("counter-evidence found; verification is ambiguous")
            disposition = CandidateDisposition.AMBIGUOUS
        elif evidence.classification == EvidenceClassification.DIRECT_SOURCE and evidence.evidence_ids:
            disposition = CandidateDisposition.SUPPORTED
        elif evidence.classification == EvidenceClassification.RULE_DERIVED:
            disposition = CandidateDisposition.SUPPORTED
        elif evidence.classification == EvidenceClassification.MODEL_ONLY:
            notes.append(
                "model-only suggestion; a high learned score is not a proof"
            )
            disposition = CandidateDisposition.UNVERIFIABLE
        else:
            notes.append("no evidence accessible either way")
            disposition = CandidateDisposition.UNVERIFIABLE

        candidate.disposition = disposition
        candidate.evidence_classification = evidence.classification
        candidate.evidence_ids = evidence.evidence_ids
        candidate.supporting_edge_ids = evidence.supporting_edge_ids
        candidate.counter_edge_ids = counter.counter_edge_ids
        candidate.rules_applied = tuple(
            rule.rule for rule in violations
        ) if violations else self._rules_applied(evidence, candidate)
        candidate.verifier_notes = tuple(notes)
        candidate.status = CandidateStatus.VERIFIED

        update_proposal: AuthorizedUpdateProposal | None = None
        if disposition == CandidateDisposition.SUPPORTED:
            update_proposal = AuthorizedUpdateProposal(
                candidate_id=candidate.candidate_id,
                triple=candidate.triple(),
                evidence_ids=evidence.evidence_ids,
                supporting_edge_ids=evidence.supporting_edge_ids,
                rules_applied=candidate.rules_applied,
                model_version=candidate.model_version,
                snapshot_hash=candidate.snapshot_hash or self.snapshot.graph_hash,
                proposed_at_utc=_utc_now(),
            )
        return VerificationResult(
            candidate_id=candidate.candidate_id,
            triple=candidate.triple(),
            disposition=disposition,
            evidence_classification=evidence.classification,
            evidence_ids=evidence.evidence_ids,
            supporting_edge_ids=evidence.supporting_edge_ids,
            counter_edge_ids=counter.counter_edge_ids,
            rules_applied=candidate.rules_applied,
            rule_violations=tuple(violations),
            type_violations=type_check.violations,
            notes=tuple(notes),
            update_proposal=update_proposal,
        )

    def verify_all(self, candidates: Sequence[GraphCandidate]) -> list[VerificationResult]:
        """Verify every candidate in deterministic order."""
        return [self.verify(candidate) for candidate in candidates]

    def _rules_applied(
        self, evidence: EvidenceCheck, candidate: GraphCandidate,
    ) -> tuple[str, ...]:
        """Record which rules produced the evidence classification."""
        if evidence.classification == EvidenceClassification.DIRECT_SOURCE:
            return ("direct_source_support",)
        if evidence.classification == EvidenceClassification.RULE_DERIVED:
            return ("rule_derivation_transitive",)
        return ()


class PromotionGate:
    """Gate supported candidates into the authorized update workflow (step 38)."""

    def gate(self, results: Sequence[VerificationResult]) -> list[AuthorizedUpdateProposal]:
        """Return update proposals only for SUPPORTED dispositions."""
        return [
            result.update_proposal for result in results
            if result.disposition == CandidateDisposition.SUPPORTED
            and result.update_proposal is not None
        ]


__all__ = [
    "AuthorizedUpdateProposal", "CandidateVerifier", "PromotionGate", "VerificationResult",
]
