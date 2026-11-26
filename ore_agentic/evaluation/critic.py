"""The bounded evaluator-optimizer loop (Steps 28-30).

A separate critic reviews the proposed output against the predeclared
contract and evidence, returning structured defects with severity, location,
rationale and suggested correction.  Revisions are bounded by iteration, time
and cost budgets, each revision creates a new artifact version, and the loop
stops when checks pass, the budget ends, or progress stalls.  Final
independent verification runs the required deterministic tests and enforces
approval gates; a failed deterministic test rejects the output.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from ..contracts import ApprovalStatus, utc_now
from .contracts import OutputContract, ValidationOutcome
from .validators import ContractValidator, DeterministicCheckRunner

__all__ = (
    "Critic",
    "CriticLoop",
    "CriticLoopReport",
    "CriticReport",
    "Defect",
    "DeterministicCritic",
    "FinalVerifier",
    "RevisionBudget",
    "VerificationReport",
    "loop_stop_reasons",
)


@dataclass(frozen=True, slots=True)
class Defect:
    """One structured defect mapped to a contract criterion (Step 28)."""

    defect_id: str
    criterion_id: str
    severity: str
    location: str
    rationale: str
    suggested_correction: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize the defect."""
        return {"defect_id": self.defect_id, "criterion_id": self.criterion_id,
                "severity": self.severity, "location": self.location,
                "rationale": self.rationale,
                "suggested_correction": self.suggested_correction}


@dataclass(frozen=True, slots=True)
class CriticReport:
    """One critic review of one artifact version (Step 28)."""

    iteration: int
    verdict: str  # "pass" | "revise" | "reject"
    defects: tuple[Defect, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    reviewed_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report."""
        return {"iteration": self.iteration, "verdict": self.verdict,
                "defects": [item.to_dict() for item in self.defects],
                "evidence_refs": list(self.evidence_refs),
                "reviewed_at": self.reviewed_at}


class Critic:
    """The separate-evaluator protocol (Step 28)."""

    def review(self, artifact: Mapping[str, Any], contract: OutputContract,
               evidence_refs: Sequence[str] = ()) -> CriticReport:
        """Review one artifact version against the contract."""
        raise NotImplementedError


class DeterministicCritic(Critic):
    """An independent deterministic checker as critic (Step 28).

    Deterministic checkers are a different implementation class from the
    producing worker, reducing shared failure modes between producer and
    evaluator.
    """

    def __init__(self, validator: ContractValidator) -> None:
        """Bind the critic to the contract validator."""
        self._validator = validator

    def review(self, artifact: Mapping[str, Any], contract: OutputContract,
               evidence_refs: Sequence[str] = ()) -> CriticReport:
        """Map violations to structured defects (Step 28)."""
        outcome: ValidationOutcome = self._validator.validate(artifact, contract)
        defects = tuple(
            Defect(
                defect_id=f"defect-{index:03d}",
                criterion_id=violation.criterion_id,
                severity=violation.severity,
                location=violation.location or "$",
                rationale=violation.message,
                suggested_correction="correct the artifact to satisfy "
                                     f"criterion {violation.criterion_id}")
            for index, violation in enumerate(outcome.violations, start=1))
        verdict = "pass" if outcome.passed and not defects else "revise"
        if any(defect.severity == "critical" for defect in defects):
            verdict = "reject"
        return CriticReport(iteration=0, verdict=verdict, defects=defects,
                            evidence_refs=tuple(evidence_refs))


@dataclass(frozen=True, slots=True)
class RevisionBudget:
    """Bounded revision limits (Step 29)."""

    max_iterations: int = 3
    cost_budget: float = 16.0
    stall_threshold: float = 0.0

    def __post_init__(self) -> None:
        """Validate the budget."""
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        if self.cost_budget <= 0.0:
            raise ValueError("cost_budget must be positive")


def loop_stop_reasons() -> tuple[str, ...]:
    """Return the exhaustive stop reasons of the critic loop (Step 29)."""
    return ("passed", "iteration_budget", "cost_budget", "stalled",
            "no_revision_capability", "rejected")


@dataclass(slots=True)
class CriticLoopReport:
    """The full trace of one bounded critic loop (Step 29)."""

    iterations: list[CriticReport] = field(default_factory=list)
    artifact_versions: list[Mapping[str, Any]] = field(default_factory=list)
    stop_reason: str = ""
    final_passed: bool = False
    cost_spent: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report."""
        return {
            "iterations": [item.to_dict() for item in self.iterations],
            "artifact_versions": [dict(version) for version in self.artifact_versions],
            "stop_reason": self.stop_reason,
            "final_passed": self.final_passed,
            "cost_spent": self.cost_spent,
        }


class CriticLoop:
    """Bounded revision loop with versioned artifacts (Step 29)."""

    def __init__(self, critic: Critic, budget: RevisionBudget | None = None) -> None:
        """Compose the loop from a critic and its budget."""
        self._critic = critic
        self._budget = budget or RevisionBudget()

    def run(self, artifact: Mapping[str, Any], contract: OutputContract,
            revise: Callable[[Mapping[str, Any], tuple[Defect, ...]],
                             Mapping[str, Any]] | None = None,
            evidence_refs: Sequence[str] = (),
            revision_cost: float = 1.0) -> CriticLoopReport:
        """Iterate review/revision within budgets; never overwrite versions (Step 29)."""
        report = CriticLoopReport()
        current = dict(artifact)
        report.artifact_versions.append(dict(current))
        previous_defect_count: int | None = None
        cost = 0.0
        for iteration in range(1, self._budget.max_iterations + 1):
            review = self._critic.review(current, contract, evidence_refs)
            review = CriticReport(iteration=iteration, verdict=review.verdict,
                                  defects=review.defects,
                                  evidence_refs=review.evidence_refs,
                                  reviewed_at=review.reviewed_at)
            report.iterations.append(review)
            if review.verdict == "pass":
                report.stop_reason = "passed"
                report.final_passed = True
                report.cost_spent = cost
                return report
            if review.verdict == "reject":
                report.stop_reason = "rejected"
                report.cost_spent = cost
                return report
            defect_count = len(review.defects)
            stalled = (previous_defect_count is not None
                       and defect_count >= previous_defect_count
                       + int(self._budget.stall_threshold))
            previous_defect_count = defect_count
            if revise is None:
                report.stop_reason = "no_revision_capability"
                report.cost_spent = cost
                return report
            if stalled:
                report.stop_reason = "stalled"
                report.cost_spent = cost
                return report
            if iteration >= self._budget.max_iterations:
                report.stop_reason = "iteration_budget"
                report.cost_spent = cost
                return report
            cost += revision_cost
            if cost > self._budget.cost_budget:
                report.stop_reason = "cost_budget"
                report.cost_spent = cost
                return report
            current = dict(revise(current, review.defects))
            report.artifact_versions.append(dict(current))
        report.stop_reason = "iteration_budget"
        report.cost_spent = cost
        return report


@dataclass(frozen=True, slots=True)
class VerificationReport:
    """Final independent verification outcome (Step 30)."""

    passed: bool
    failed_checks: tuple[str, ...] = ()
    approval: ApprovalStatus = ApprovalStatus.NOT_REQUIRED
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report."""
        return {"passed": self.passed, "failed_checks": list(self.failed_checks),
                "approval": self.approval.value, "detail": self.detail}


class FinalVerifier:
    """Run required tests and enforce approval gates (Step 30)."""

    def __init__(self, runner: DeterministicCheckRunner, *,
                 approver: Callable[[str], ApprovalStatus] | None = None) -> None:
        """Compose the verifier; the human approver is injectable."""
        self._runner = runner
        self._approver = approver

    def verify(self, artifact: Mapping[str, Any],
               required_checks: Sequence[str],
               contract: OutputContract | None = None) -> VerificationReport:
        """Reject on any failed deterministic test; gate on approval (Step 30)."""
        results = self._runner.run_all(required_checks, artifact)
        failed = [check_id for check_id, (passed, _detail) in results.items()
                  if not passed]
        approval = ApprovalStatus.NOT_REQUIRED
        if contract is not None and contract.approval.consequential:
            if self._approver is None:
                approval = ApprovalStatus.PENDING
            else:
                approval = self._approver(contract.contract_id)
        passed = not failed and approval in (ApprovalStatus.NOT_REQUIRED,
                                             ApprovalStatus.APPROVED)
        detail = "; ".join(f"{check_id}: {results[check_id][1]}"
                           for check_id in failed) or "all required checks passed"
        return VerificationReport(passed=passed, failed_checks=tuple(failed),
                                  approval=approval, detail=detail)
