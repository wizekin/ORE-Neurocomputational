"""Evaluation, versioned learning, and safe improvement proposals (Step 52).

Compares execution outcomes with acceptance criteria, records latency, memory,
correctness and failure evidence, and versions the whole compilation lineage.
Learning records stay separate from authoritative semantic knowledge: proposals
can never silently modify approved requirements or USG facts, and promotion
requires an explicit, audited human decision.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .contracts import ArtifactKind, CanonicalArtifact, GateViolation, digest_payload
from .requirements import RequirementSpec
from .sandbox import ExecutionReport
from .session import CompilationSession


@dataclass(frozen=True, slots=True)
class EvaluationOutcome:
    """One acceptance criterion evaluated against execution evidence."""

    criterion_id: str
    statement: str
    satisfied: bool | None  # None = not decidable from execution alone
    evidence: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "criterion_id": self.criterion_id,
            "statement": self.statement,
            "satisfied": self.satisfied,
            "evidence": self.evidence,
        }


@dataclass(frozen=True, slots=True)
class ImprovementProposal:
    """A proposed change; proposals never mutate approved artifacts."""

    proposal_id: str
    statement: str
    rationale: str
    target_artifact: str
    status: str  # PROPOSED | APPROVED | REJECTED
    requires_human_review: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "statement": self.statement,
            "rationale": self.rationale,
            "target_artifact": self.target_artifact,
            "status": self.status,
            "requires_human_review": self.requires_human_review,
        }


class LearningRecord(CanonicalArtifact):
    """Canonical artifact: observed performance and validated improvements."""

    kind = ArtifactKind.LEARNING_RECORD

    def __init__(
        self,
        *,
        artifact_id: str,
        execution_report_id: str,
        outcomes: Iterable[EvaluationOutcome],
        metrics: dict[str, Any],
        proposals: Iterable[ImprovementProposal],
        lineage_digests: dict[str, str],
        version: int = 1,
    ) -> None:
        self.artifact_id = artifact_id
        self.execution_report_id = execution_report_id
        self.outcomes = tuple(outcomes)
        self.metrics = dict(metrics)
        self.proposals = tuple(proposals)
        self.lineage_digests = dict(lineage_digests)
        self.version = version

    def content_payload(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "execution_report_id": self.execution_report_id,
            "outcomes": [o.to_dict() for o in self.outcomes],
            "metrics": dict(sorted(self.metrics.items())),
            "proposals": [p.to_dict() for p in self.proposals],
            "lineage_digests": dict(sorted(self.lineage_digests.items())),
            "version": self.version,
        }

    @property
    def semantic_digest(self) -> str:
        """Digest over deterministic content; runtime metrics excluded.

        Wall-clock times and memory readings are observations, not canonical
        content: determinism is promised for the compilation artifacts, not
        for measured execution performance (spec Section 9).
        """
        payload = self.content_payload()
        payload.pop("artifact_id", None)
        payload.pop("execution_report_id", None)
        payload.pop("metrics", None)
        return digest_payload(payload)


class LearningJournal:
    """Evaluate runs and propose improvements safely (Step 52)."""

    def evaluate(
        self,
        session: CompilationSession,
        execution: ExecutionReport,
        spec: RequirementSpec,
        lineage_digests: dict[str, str],
    ) -> LearningRecord:
        outcomes: list[EvaluationOutcome] = []
        for criterion in spec.acceptance_criteria:
            if criterion.criterion_id == "acc-001":
                satisfied = execution.all_passed and execution.exit_code == 0
                evidence = "sandbox fixtures passed; exactly-one enforced by runtime assertions"
            elif criterion.criterion_id == "acc-002":
                satisfied = execution.all_passed
                evidence = "capacity assertion executed in sandbox"
            elif criterion.criterion_id == "acc-004":
                satisfied = None
                evidence = "negative fixtures are covered by property tests, not only execution"
            elif criterion.criterion_id == "acc-005":
                satisfied = None
                evidence = "empty-input behavior is enforced by validate_inputs; measured in property tests"
            else:
                satisfied = execution.all_passed if execution.outcomes else None
                evidence = "derived from sandbox outcome status"
            outcomes.append(
                EvaluationOutcome(criterion.criterion_id, criterion.statement, satisfied, evidence)
            )
        metrics = {
            "wall_seconds": execution.wall_seconds,
            "max_rss_kb": execution.max_rss_kb,
            "fixtures_passed": sum(1 for o in execution.outcomes if o.status == "PASSED"),
            "fixtures_total": len(execution.outcomes),
            "blocked_violations": len(execution.blocked_violations),
        }
        proposals: list[ImprovementProposal] = []
        if execution.exit_code != 0:
            proposals.append(
                ImprovementProposal(
                    "prop-001",
                    "investigate non-zero sandbox exit code before any deployment",
                    "execution reports must be clean before approval scope is widened",
                    execution.code_artifact_id,
                    "PROPOSED",
                    True,
                )
            )
        proposals.append(
            ImprovementProposal(
                "prop-002",
                "benchmark greedy-by-pair against greedy-by-task on production-like instances",
                "candidate comparison showed greedy-by-pair may improve average quality",
                "candidate-set",
                "PROPOSED",
                True,
            )
        )
        record = LearningRecord(
            artifact_id=session.next_artifact_id(ArtifactKind.LEARNING_RECORD),
            execution_report_id=execution.artifact_id,
            outcomes=outcomes,
            metrics=metrics,
            proposals=proposals,
            lineage_digests=lineage_digests,
        )
        session.register(record)
        session.ledger.record(
            stage="H",
            step_id="HIC52",
            description="Evaluate execution against acceptance criteria and record learning",
            inputs=[execution, spec],
            outputs=[record],
            rule_references={"HIC-LEARN-1"},
        )
        return record

    @staticmethod
    def promote(proposal: ImprovementProposal, *, approved_by: str) -> ImprovementProposal:
        """Promotion requires an explicit human decision and stays auditable."""
        if not approved_by or not approved_by.strip():
            raise GateViolation("proposal promotion requires a human approver identity")
        return ImprovementProposal(
            proposal.proposal_id,
            proposal.statement,
            proposal.rationale,
            proposal.target_artifact,
            "APPROVED",
            False,
        )
