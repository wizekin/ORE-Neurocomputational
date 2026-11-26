"""Performance ledger: accepted quality, reliability, cost, and calibration (Steps 55-57).

The ledger records accepted-result rate, defect rate, verifier disagreement,
latency, cost, retry rate, calibration, and task-class coverage per agent.
Reliability uses Beta-Binomial posteriors with credible intervals so tiny samples
can never justify precise ranking (Step 56). Learning from validated outcomes
rewards accepted goal progress, correctness, and efficiency, and penalizes
invalid outputs, waste, and policy violations - never verbosity or task count
(Step 57).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .adjudication import AdjudicationReport
from .contracts import AdjudicationDecision, AgentResult, AgentTask
from .reliability import (
    BetaBinomialEstimate,
    ReliabilityRecord,
    reliability_report,
)


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    """Record one adjudicated outcome with its resource and calibration metrics."""

    agent_id: str
    task_id: str
    task_class: str
    accepted: bool
    decision: str
    resource_usage: float
    elapsed_seconds: float
    self_confidence: float
    retried: bool = False
    verifier_disagreed: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a detached entry record."""
        return {
            "agent_id": self.agent_id,
            "task_id": self.task_id,
            "task_class": self.task_class,
            "accepted": self.accepted,
            "decision": self.decision,
            "resource_usage": self.resource_usage,
            "elapsed_seconds": self.elapsed_seconds,
            "self_confidence": self.self_confidence,
            "retried": self.retried,
            "verifier_disagreed": self.verifier_disagreed,
        }


@dataclass
class PerformanceLedger:
    """Track per-agent accepted quality, cost, latency, and failures (Step 55)."""

    prior_alpha: float = 1.0
    prior_beta: float = 1.0
    records: dict[str, ReliabilityRecord] = field(default_factory=dict)
    entries: list[LedgerEntry] = field(default_factory=list)
    conflict_counts: dict[str, int] = field(default_factory=dict)

    def record(
        self,
        task: AgentTask,
        result: AgentResult,
        report: AdjudicationReport,
        *,
        retried: bool = False,
        verifier_disagreed: bool = False,
    ) -> LedgerEntry:
        """Record one adjudicated outcome and update every metric."""
        accepted = report.decision is AdjudicationDecision.ACCEPTED
        task_class = task.required_capabilities[0] if task.required_capabilities else "unknown"
        entry = LedgerEntry(
            agent_id=result.agent_id,
            task_id=task.task_id,
            task_class=task_class,
            accepted=accepted,
            decision=report.decision.value,
            resource_usage=result.resource_usage,
            elapsed_seconds=result.elapsed_seconds,
            self_confidence=result.self_confidence,
            retried=retried,
            verifier_disagreed=verifier_disagreed,
        )
        self.entries.append(entry)
        record = self.records.setdefault(result.agent_id, ReliabilityRecord())
        record.record_outcome(accepted=accepted)
        record.record_calibration(result.self_confidence, accepted)
        record.record_resource(
            elapsed_seconds=result.elapsed_seconds, cost=result.resource_usage,
        )
        record.record_task_class(task_class)
        if retried:
            record.retries += 1
        if verifier_disagreed:
            record.verifier_disagreements += 1
        return entry

    def record_conflict(self, agent_id: str) -> None:
        """Count one preserved conflict involving an agent's proposal."""
        self.conflict_counts[agent_id] = self.conflict_counts.get(agent_id, 0) + 1

    def record_violation(self, agent_id: str) -> None:
        """Count one policy violation that penalizes learned reliability (Step 57)."""
        record = self.records.setdefault(agent_id, ReliabilityRecord())
        record.policy_violations += 1
        record.record_outcome(accepted=False)

    def estimate(self, agent_id: str) -> BetaBinomialEstimate:
        """Return the agent's Beta-Binomial reliability estimate."""
        record = self.records.get(agent_id) or ReliabilityRecord()
        return record.estimate(self.prior_alpha, self.prior_beta)

    def conflict_rate(self, agent_id: str) -> float:
        """Return the agent's conflict involvement rate over recorded entries."""
        agent_entries = [entry for entry in self.entries if entry.agent_id == agent_id]
        if not agent_entries:
            return 0.0
        return self.conflict_counts.get(agent_id, 0) / len(agent_entries)

    def learning_reward(self, agent_id: str) -> float:
        """Compute the validated-outcome reward signal (Step 57).

        Rewards accepted goal progress, correctness, and efficiency; penalizes
        invalid outputs, waste, and policy violations. Verbosity and task count
        are deliberately absent.
        """
        record = self.records.get(agent_id) or ReliabilityRecord()
        estimate = record.estimate(self.prior_alpha, self.prior_beta)
        outcomes = record.successes + record.failures
        if not outcomes:
            return 0.0
        acceptance = estimate.posterior_mean
        calibration_penalty = record.brier_score or 0.0
        waste_penalty = min(1.0, record.retries / outcomes)
        violation_penalty = min(1.0, record.policy_violations / outcomes)
        verifier_penalty = min(1.0, record.verifier_disagreements / outcomes)
        return round(
            max(-1.0, min(1.0, acceptance - 0.5 * calibration_penalty - 0.5 * waste_penalty
                          - 1.0 * violation_penalty - 0.5 * verifier_penalty)),
            4,
        )

    def summary(self) -> dict[str, Any]:
        """Summarize reliability, calibration, and rewards for every tracked agent."""
        return {
            "reliability": reliability_report(
                self.records, prior_alpha=self.prior_alpha, prior_beta=self.prior_beta,
            ),
            "learning_reward": {
                agent_id: self.learning_reward(agent_id) for agent_id in sorted(self.records)
            },
            "conflict_counts": dict(self.conflict_counts),
            "entries": len(self.entries),
        }

    def entries_for(self, agent_id: str) -> tuple[LedgerEntry, ...]:
        """Return the recorded entries for one agent."""
        return tuple(entry for entry in self.entries if entry.agent_id == agent_id)


def sufficient_sample_warning(agent_id: str, estimate: BetaBinomialEstimate) -> str | None:
    """Warn when an agent is being ranked from a tiny sample (Step 56)."""
    return None if estimate.samples >= 5 else (
        f"agent {agent_id} has only {estimate.samples} samples; "
        "reliable ranking requires more accepted-outcome evidence"
    )


def ledger_entries_for_task(entries: Sequence[LedgerEntry], task_id: str) -> tuple[LedgerEntry, ...]:
    """Filter recorded ledger entries down to one task."""
    return tuple(entry for entry in entries if entry.task_id == task_id)
