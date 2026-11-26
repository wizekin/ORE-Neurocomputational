"""Publication gates and typed workspace publication (Phase G, Steps G3-G5).

All findings flow through the *existing* shared-workspace publication path
(``FindingPublisher`` → ``WorkspaceStateManager`` validation → atomic commit →
post-commit events). This module adds the reasoning-specific gates in front of
that path (Step G3):

- proof validity: strict-class conclusions require independent verification;
- evidence sufficiency: minimum coverage of supporting source facts;
- permitted epistemic status: only ``DERIVED``, ``ASSERTED``, ``OBSERVED`` and
  ``HYPOTHESIS`` may publish; ``MODEL_PROPOSAL`` and ``UNKNOWN`` may not;
- critical constraints: declared constraint IDs must be satisfied;
- unresolved-conflict level: publication is blocked while too many unresolved
  conflicts involve the finding.

High-impact, novel or contested findings additionally request independent
review (Step G5). Conflicts are mirrored into the workspace ``ConflictRegistry``
so both ledgers stay correlated (Step G2's workspace-side record).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ore_workspace.models import Provenance
from ore_workspace.publication import FindingPublisher
from ore_workspace.workspace import SharedWorkspace

from .contracts import (
    ConfidenceVector,
    EpistemicStatus,
    Literal,
    NSRLConflictRecord,
    ProofClass,
    PublicationState,
    ReasoningFinding,
    make_id,
)

PERMITTED_PUBLICATION_STATUSES: frozenset[EpistemicStatus] = frozenset({
    EpistemicStatus.DERIVED,
    EpistemicStatus.ASSERTED,
    EpistemicStatus.OBSERVED,
    EpistemicStatus.HYPOTHESIS,
})

BLOCKED_PUBLICATION_STATUSES: frozenset[EpistemicStatus] = frozenset({
    EpistemicStatus.MODEL_PROPOSAL,
    EpistemicStatus.UNKNOWN,
    EpistemicStatus.PROBABILISTIC,
})


@dataclass(frozen=True, slots=True)
class PublicationGates:
    """Configured thresholds for the Step G3 gates."""

    require_strict_verification: bool = True
    minimum_source_facts: int = 1
    maximum_unresolved_conflicts: int = 0
    critical_constraints: tuple[str, ...] = ()
    review_impact_threshold: float = 0.9

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "require_strict_verification": self.require_strict_verification,
            "minimum_source_facts": self.minimum_source_facts,
            "maximum_unresolved_conflicts": self.maximum_unresolved_conflicts,
            "critical_constraints": list(self.critical_constraints),
            "review_impact_threshold": self.review_impact_threshold,
        }


@dataclass(slots=True)
class GateDecision:
    """Outcome of one publication-gate evaluation."""

    finding_id: str
    passed: bool
    gate_results: tuple[Mapping[str, Any], ...] = ()
    publication_state: PublicationState = PublicationState.UNREVIEWED
    review_requested: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "finding_id": self.finding_id,
            "passed": self.passed,
            "gate_results": [dict(item) for item in self.gate_results],
            "publication_state": self.publication_state.value,
            "review_requested": self.review_requested,
        }


@dataclass(slots=True)
class PublicationReport:
    """Outcome of one Phase G publication pass."""

    decisions: tuple[GateDecision, ...] = ()
    published_finding_ids: tuple[str, ...] = ()
    review_requested_ids: tuple[str, ...] = ()
    workspace_results: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "decisions": [item.to_dict() for item in self.decisions],
            "published_finding_ids": list(self.published_finding_ids),
            "review_requested_ids": list(self.review_requested_ids),
            "workspace_results": dict(self.workspace_results),
        }


class NSRLWorkspacePublisher:
    """Gate and publish reasoning findings through the existing workspace APIs."""

    def __init__(
        self,
        workspace: SharedWorkspace,
        *,
        producer: str = "ore_neurosymbolic",
        gates: PublicationGates | None = None,
    ) -> None:
        """Bind the workspace, the producer identity and the gate configuration."""
        self.workspace = workspace
        self.producer = producer
        self.gates = gates or PublicationGates()
        self._publisher = FindingPublisher(workspace, producer)

    # ------------------------------------------------------------------ Step G3

    def apply_gates(
        self,
        findings: tuple[ReasoningFinding, ...],
        *,
        verified_node_ids: Mapping[str, bool],
        conflict_manager: Any = None,
        satisfied_constraints: tuple[str, ...] = (),
    ) -> tuple[GateDecision, ...]:
        """Evaluate every gate for every finding in deterministic order (Step G3)."""
        decisions: list[GateDecision] = []
        unresolved_by_finding = self._unresolved_conflicts_by_finding(conflict_manager)
        for finding in findings:
            results: list[Mapping[str, Any]] = []
            passed = True

            proof_ok = self._proof_gate(finding, verified_node_ids)
            results.append(proof_ok)
            passed = passed and bool(proof_ok["passed"])

            evidence_ok = self._evidence_gate(finding)
            results.append(evidence_ok)
            passed = passed and bool(evidence_ok["passed"])

            status_ok = self._epistemic_gate(finding)
            results.append(status_ok)
            passed = passed and bool(status_ok["passed"])

            critical_ok = self._critical_constraint_gate(satisfied_constraints)
            results.append(critical_ok)
            passed = passed and bool(critical_ok["passed"])

            conflict_ok = self._conflict_gate(
                finding, unresolved_by_finding.get(finding.finding_id, ()),
            )
            results.append(conflict_ok)
            passed = passed and bool(conflict_ok["passed"])

            review_requested = self._needs_review(
                finding, verified_node_ids,
            ) and passed
            decision = GateDecision(
                finding_id=finding.finding_id,
                passed=passed,
                gate_results=tuple(results),
                publication_state=(
                    PublicationState.PUBLISHED if passed else PublicationState.GATED_OUT
                ),
                review_requested=review_requested,
            )
            decisions.append(decision)
        return tuple(decisions)

    def _proof_gate(
        self, finding: ReasoningFinding, verified_node_ids: Mapping[str, bool],
    ) -> dict[str, Any]:
        """Require independent verification for strict-class conclusions (G3)."""
        if finding.proof_class is ProofClass.STRICT_VERIFIED:
            verified = bool(verified_node_ids.get(finding.proof_ref or ""))
            return {
                "gate": "proof_validity", "passed": verified,
                "detail": "strict_verified_requires_independent_verification",
            }
        if finding.proof_class is ProofClass.DEFEASIBLE:
            return {
                "gate": "proof_validity", "passed": True,
                "detail": "defeasible_published_with_explicit_label",
            }
        if finding.proof_class in {ProofClass.NEURAL_PROPOSAL, ProofClass.UNVERIFIED_CANDIDATE}:
            return {
                "gate": "proof_validity", "passed": False,
                "detail": "unverified_or_neural_proposals_cannot_publish",
            }
        return {
            "gate": "proof_validity", "passed": True,
            "detail": "probabilistic_support_published_with_decomposed_confidence",
        }

    def _evidence_gate(self, finding: ReasoningFinding) -> dict[str, Any]:
        """Require the minimum number of supporting source facts (G3)."""
        count = len(finding.source_fact_ids)
        minimum = self.gates.minimum_source_facts
        return {
            "gate": "evidence_sufficiency",
            "passed": count >= minimum,
            "detail": f"source_facts={count}, minimum={minimum}",
        }

    def _epistemic_gate(self, finding: ReasoningFinding) -> dict[str, Any]:
        """Only permitted epistemic statuses may publish (G3; Section 2)."""
        permitted = finding.epistemic_status in PERMITTED_PUBLICATION_STATUSES
        return {
            "gate": "permitted_epistemic_status",
            "passed": permitted,
            "detail": f"status={finding.epistemic_status.value}",
        }

    def _critical_constraint_gate(self, satisfied_constraints: tuple[str, ...]) -> dict[str, Any]:
        """Require declared critical constraints to be satisfied (G3)."""
        missing = [
            item for item in self.gates.critical_constraints
            if item not in satisfied_constraints
        ]
        return {
            "gate": "critical_constraints",
            "passed": not missing,
            "detail": f"missing={missing}",
        }

    def _conflict_gate(
        self,
        finding: ReasoningFinding,
        unresolved_conflicts: tuple[str, ...],
    ) -> dict[str, Any]:
        """Block publication above the unresolved-conflict threshold (G3)."""
        count = len(unresolved_conflicts)
        return {
            "gate": "unresolved_conflict_level",
            "passed": count <= self.gates.maximum_unresolved_conflicts,
            "detail": f"unresolved_conflicts={count}",
        }

    def _needs_review(
        self, finding: ReasoningFinding, verified_node_ids: Mapping[str, bool],
    ) -> bool:
        """High-impact, novel or contested findings request independent review (G5)."""
        contested = len(finding.conflicts) > 0
        high_impact = min(
            finding.support_profile.q_source,
            finding.support_profile.q_observation,
        ) >= self.gates.review_impact_threshold
        novel = finding.proof_class is ProofClass.STRICT_VERIFIED and bool(
            verified_node_ids.get(finding.proof_ref or ""),
        )
        return contested or high_impact or novel

    @staticmethod
    def _unresolved_conflicts_by_finding(
        conflict_manager: Any,
    ) -> dict[str, tuple[str, ...]]:
        """Index unresolved conflict IDs by the finding IDs they involve."""
        mapping: dict[str, list[str]] = {}
        if conflict_manager is None:
            return mapping
        for record in conflict_manager.unresolved():
            for claim in record.claims:
                finding_id = claim.get("finding_id")
                if finding_id:
                    mapping.setdefault(str(finding_id), []).append(record.conflict_id)
        return {key: tuple(value) for key, value in mapping.items()}

    # ------------------------------------------------------------------ Step G4

    def publish(
        self,
        findings: tuple[ReasoningFinding, ...],
        decisions: tuple[GateDecision, ...],
        *,
        snapshot_revision: int,
        lineage: Mapping[str, Any] | None = None,
        conflict_manager: Any = None,
    ) -> PublicationReport:
        """Publish gated findings through the atomic workspace path (Step G4).

        Each payload carries the structured finding, its proof reference, the
        source lineage, the decomposed support vector and the snapshot
        revision. Workspace-level decisions (accepted/conflict/duplicate) are
        returned per finding; gated-out findings never reach the workspace.
        """
        decision_by_id = {item.finding_id: item for item in decisions}
        report = PublicationReport()
        for finding in findings:
            decision = decision_by_id.get(finding.finding_id)
            if decision is None or not decision.passed:
                continue
            payload = {
                "finding": finding.to_dict(),
                "proof_ref": finding.proof_ref,
                "lineage": dict(lineage or {}),
                "support_vector": finding.support_profile.to_dict(),
                "snapshot_revision": snapshot_revision,
                "conclusion_signature": finding.conclusion.signature(),
            }
            provenance = Provenance(
                source=f"nsrl:{finding.run_id}",
                rules_applied=(finding.proof_class.value,),
                evidence=tuple(
                    {"fact_id": item} for item in finding.source_fact_ids
                ),
                notes="Proof-carrying neuro-symbolic finding; epistemic status preserved.",
            )
            workspace_finding = self._publisher.create_finding(
                "semantic_hypotheses",
                payload,
                key=f"nsrl:{finding.finding_id}",
                finding_type="neurosymbolic_conclusion",
                provenance=provenance,
                dependencies=(),
                confidence=None,
                base_workspace_revision=snapshot_revision,
            )
            result = self.workspace.publish(workspace_finding)
            report.workspace_results[finding.finding_id] = str(result.decision.value)
            if finding.conflicts and conflict_manager is not None:
                self._mirror_conflicts(finding, conflict_manager)
            if decision.review_requested:
                report.review_requested_ids = (
                    *report.review_requested_ids, finding.finding_id,
                )
            report.published_finding_ids = (*report.published_finding_ids, finding.finding_id)
        report.decisions = decisions
        return report

    def _mirror_conflicts(self, finding: ReasoningFinding, conflict_manager: Any) -> None:
        """Mirror NSRL conflict records into the workspace ConflictRegistry."""
        from ore_workspace.models import Finding, FindingTarget

        for conflict_id in finding.conflicts:
            record = conflict_manager.resolve_lookup(conflict_id) if hasattr(
                conflict_manager, "resolve_lookup",
            ) else None
            claims = record.claims if record is not None else ()
            if len(claims) < 2:
                continue
            from ore_workspace.publication import FindingPublisher as _FP

            mirror_publisher = _FP(self.workspace, self.producer)
            for index in (0, 1):
                mirror_finding = mirror_publisher.create_finding(
                    "semantic_hypotheses",
                    {
                        "candidate": claims[index].get("conclusion_signature", ""),
                        "conflict_id": conflict_id,
                        "origin": "nsrl_conflict_mirror",
                    },
                    key=f"nsrl_conflict:{conflict_id}:{index}",
                    finding_type="neurosymbolic_conflict_alternative",
                    provenance=Provenance(
                        source=f"nsrl:conflict:{conflict_id}",
                        rules_applied=("conflict_preservation",),
                        notes="Conflicting claim preserved; source facts untouched.",
                    ),
                    base_workspace_revision=self.workspace.revision,
                )
                self.workspace.publish(mirror_finding)

    # ------------------------------------------------------------------ Step G5

    def request_independent_review(
        self,
        finding: ReasoningFinding,
        *,
        premises: tuple[str, ...] = (),
        proof_summary: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Publish a review request for high-impact/novel/contested findings (G5).

        The request carries the exact premises and proof so another module can
        re-verify without trusting this run.
        """
        request = {
            "review_id": make_id("nsr_review"),
            "finding_id": finding.finding_id,
            "conclusion": finding.conclusion.to_dict(),
            "proof_ref": finding.proof_ref,
            "premises": list(premises),
            "proof_summary": dict(proof_summary or {}),
            "reasons": {
                "contested": len(finding.conflicts) > 0,
                "support_vector": finding.support_profile.to_dict(),
            },
            "workspace_revision": self.workspace.revision,
        }
        review_finding = self._publisher.create_finding(
            "unresolved_questions",
            request,
            key=f"nsr_review:{finding.finding_id}",
            finding_type="independent_review_request",
            provenance=Provenance(
                source=f"nsrl:{finding.run_id}",
                rules_applied=("independent_review",),
                notes="Premises and proof supplied for independent re-verification.",
            ),
            base_workspace_revision=self.workspace.revision,
        )
        result = self.workspace.publish(review_finding)
        return {
            "review": request,
            "workspace_decision": str(result.decision.value),
        }


__all__ = [
    "BLOCKED_PUBLICATION_STATUSES",
    "GateDecision",
    "PublicationGates",
    "PublicationReport",
    "PERMITTED_PUBLICATION_STATUSES",
    "NSRLWorkspacePublisher",
]
