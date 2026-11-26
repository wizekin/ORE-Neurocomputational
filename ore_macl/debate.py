"""Debate, critic, and independent verification orchestration (spec Section 4F).

For high-impact or uncertain tasks the orchestrator generates independent
proposals along separate evidence paths, measures independence by shared data,
model/policy lineage, and tools, evaluates evidence rather than votes, structures
critic feedback, bounds revision cycles, and preserves minority findings that
carry evidence (Steps 37-43).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .adjudication import AdjudicationReport, ResultAdjudicator
from .agents import BaseBoundedAgent, CriticAgent
from .contracts import AgentResult, AgentTask


@dataclass(frozen=True, slots=True)
class IndependenceAssessment:
    """Quantify how independent two agents' evidence paths actually are (Step 38)."""

    agent_id_a: str
    agent_id_b: str
    shared_fields: tuple[str, ...]
    shared_policy: bool
    shared_tools: tuple[str, ...]
    independence: float

    def to_dict(self) -> dict[str, Any]:
        """Return a detached assessment record."""
        return {
            "agent_id_a": self.agent_id_a,
            "agent_id_b": self.agent_id_b,
            "shared_fields": list(self.shared_fields),
            "shared_policy": self.shared_policy,
            "shared_tools": list(self.shared_tools),
            "independence": self.independence,
        }


def measure_independence(
    result_a: AgentResult,
    result_b: AgentResult,
    *,
    policy_versions: Mapping[str, str] | None = None,
    tools_by_agent: Mapping[str, tuple[str, ...]] | None = None,
) -> IndependenceAssessment:
    """Measure shared data, model/policy lineage, and tools between two agents (Step 38).

    Two agents using the same evidence and the same model are not independent
    witnesses; independence decreases with every shared dependency.
    """
    policies = dict(policy_versions or {})
    tools = dict(tools_by_agent or {})
    evidence_a = {item.artifact_id for item in result_a.evidence_refs}
    evidence_b = {item.artifact_id for item in result_b.evidence_refs}
    shared_fields = tuple(sorted(evidence_a & evidence_b))
    shared_policy = (
        policies.get(result_a.agent_id) is not None
        and policies.get(result_a.agent_id) == policies.get(result_b.agent_id)
    )
    tools_a = set(tools.get(result_a.agent_id, ()))
    tools_b = set(tools.get(result_b.agent_id, ()))
    shared_tools = tuple(sorted(tools_a & tools_b))
    penalty = 0.0
    penalty += 0.2 * len(shared_fields)
    penalty += 0.25 if shared_policy else 0.0
    penalty += 0.15 * len(shared_tools)
    independence = max(0.0, 1.0 - penalty)
    return IndependenceAssessment(
        agent_id_a=result_a.agent_id,
        agent_id_b=result_b.agent_id,
        shared_fields=shared_fields,
        shared_policy=shared_policy,
        shared_tools=shared_tools,
        independence=independence,
    )


@dataclass(frozen=True, slots=True)
class DissentRecord:
    """Preserve a minority finding with its evidence instead of discarding it (Step 43)."""

    agent_id: str
    interpretation: str
    evidence_refs: tuple[Mapping[str, Any], ...]
    uncertainty: float

    def to_dict(self) -> dict[str, Any]:
        """Return a detached dissent record."""
        return {
            "agent_id": self.agent_id,
            "interpretation": self.interpretation,
            "evidence_refs": [dict(item) for item in self.evidence_refs],
            "uncertainty": self.uncertainty,
        }


@dataclass
class DebateRecord:
    """Capture one bounded debate: proposals, critiques, rounds, and the decision basis."""

    task_id: str
    proposals: list[dict[str, Any]] = field(default_factory=list)
    critiques: list[dict[str, Any]] = field(default_factory=list)
    independence_assessments: list[dict[str, Any]] = field(default_factory=list)
    rounds: int = 0
    stop_reason: str = ""
    selected_agent_id: str = ""
    dissent: list[DissentRecord] = field(default_factory=list)
    evidence_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a detached debate record."""
        return {
            "task_id": self.task_id,
            "proposals": list(self.proposals),
            "critiques": list(self.critiques),
            "independence_assessments": list(self.independence_assessments),
            "rounds": self.rounds,
            "stop_reason": self.stop_reason,
            "selected_agent_id": self.selected_agent_id,
            "dissent": [item.to_dict() for item in self.dissent],
            "evidence_scores": dict(self.evidence_scores),
        }


class DebateOrchestrator:
    """Run bounded generate/evaluate/revise debates over competing hypotheses."""

    def __init__(
        self,
        adjudicator: ResultAdjudicator,
        *,
        max_rounds: int = 3,
        min_independence: float = 0.5,
        policy_versions: Mapping[str, str] | None = None,
    ) -> None:
        """Configure bounded rounds and the independence threshold."""
        if max_rounds < 1:
            raise ValueError("max_rounds must be at least one")
        if not 0.0 <= min_independence <= 1.0:
            raise ValueError("min_independence must be in [0, 1]")
        self.adjudicator = adjudicator
        self.max_rounds = max_rounds
        self.min_independence = min_independence
        self.policy_versions = dict(policy_versions or {})

    def run(
        self,
        task: AgentTask,
        proposals: Sequence[AgentResult],
        critic: CriticAgent,
        *,
        on_round: Any = None,
    ) -> DebateRecord:
        """Evaluate proposals by evidence quality, not vote count (Steps 37-43)."""
        record = DebateRecord(task_id=task.task_id)
        if not proposals:
            record.stop_reason = "no proposals submitted"
            return record
        current = list(proposals)
        for round_index in range(1, self.max_rounds + 1):
            record.rounds = round_index
            for proposal in current:
                record.proposals.append({
                    "round": round_index,
                    "agent_id": proposal.agent_id,
                    "interpretation": str(proposal.outputs.get("interpretation", "")),
                    "self_confidence": proposal.self_confidence,
                })
            for index, proposal in enumerate(current):
                counterpart = current[index - 1] if index > 0 else (
                    current[index + 1] if len(current) > index + 1 else None
                )
                critique = critic.critique(
                    claim=str(proposal.outputs.get("interpretation", "unnamed")),
                    proposal=dict(proposal.outputs),
                    counterpart=dict(counterpart.outputs) if counterpart is not None else None,
                )
                critique["round"] = round_index
                critique["agent_id"] = proposal.agent_id
                record.critiques.append(critique)
            for index in range(len(current)):
                for other in range(index + 1, len(current)):
                    assessment = measure_independence(
                        current[index], current[other],
                        policy_versions=self.policy_versions,
                        tools_by_agent={
                            proposal.agent_id: tuple(proposal.provenance.transformation_history)
                            for proposal in current
                        },
                    )
                    record.independence_assessments.append(assessment.to_dict())
            scores = {
                proposal.agent_id: self._evidence_score(proposal, record)
                for proposal in current
            }
            record.evidence_scores.update(scores)
            if on_round is not None:
                on_round(record)
            unaddressed = [
                critique for critique in record.critiques
                if critique.get("violated_rule") != "none" and critique.get("round") == round_index
            ]
            if not unaddressed:
                record.stop_reason = "critic criteria passed"
                break
            if round_index == self.max_rounds:
                record.stop_reason = "maximum rounds reached"
                break
        selected = max(
            sorted(record.evidence_scores),
            key=lambda agent_id: record.evidence_scores[agent_id],
        ) if record.evidence_scores else ""
        record.selected_agent_id = selected
        for proposal in current:
            if proposal.agent_id != selected and proposal.evidence_refs:
                record.dissent.append(DissentRecord(
                    agent_id=proposal.agent_id,
                    interpretation=str(proposal.outputs.get("interpretation", "")),
                    evidence_refs=[item.to_dict() for item in proposal.evidence_refs],
                    uncertainty=proposal.uncertainty,
                ))
        if not record.stop_reason:
            record.stop_reason = "budget exhausted"
        return record

    def _evidence_score(self, proposal: AgentResult, record: DebateRecord) -> float:
        """Score proposals by evidence quality, consistency, and provenance (Step 39).

        The number of agents supporting a claim is never a truth score; the score
        combines evidence coverage, self-consistency, uncertainty, and absence of
        unaddressed critic findings.
        """
        evidence_coverage = min(1.0, len(proposal.evidence_refs) / 2.0)
        provenance_ok = 1.0 if proposal.provenance.operation else 0.0
        unaddressed = sum(
            1 for critique in record.critiques
            if critique.get("agent_id") == proposal.agent_id
            and critique.get("violated_rule") != "none"
        )
        consistency = max(0.0, 1.0 - 0.25 * unaddressed)
        return round(
            0.35 * evidence_coverage + 0.25 * provenance_ok
            + 0.25 * consistency + 0.15 * (1.0 - proposal.uncertainty),
            4,
        )


def verify_independently(
    verifier: BaseBoundedAgent,
    context_factory: Any,
    adjudicator: ResultAdjudicator,
    report: AdjudicationReport,
    *,
    verifier_requirements: Sequence[str] = (),
) -> AdjudicationReport:
    """Run one independent verification pass for a critical output (Step 42)."""
    try:
        verification_result = verifier.execute(context_factory())
    except Exception as exc:  # noqa: BLE001 - verification failure is a report, not a crash
        return AdjudicationReport(
            decision=report.decision, task_id=report.task_id, agent_id=report.agent_id,
            reasons=tuple(report.reasons) + (f"verifier failed: {exc}",),
            evidence_errors=report.evidence_errors, criteria_gaps=report.criteria_gaps,
            fingerprint=report.fingerprint,
            verified_by=tuple(report.verified_by) + (f"{verifier.agent_id}:error",),
        )
    passed = bool(verification_result.outputs.get("verified", False))
    return adjudicator.record_verification(report.task_id, verifier.agent_id, passed, report)
