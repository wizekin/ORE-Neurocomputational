"""Recovery: bounded retries, timeout reconciliation, safe reassignment, quarantine.

Retry only transient failures of idempotent operations; for non-idempotent
actions query status first. A timeout marks the outcome unknown until checked;
timeout does not prove the worker did nothing. Reassignment requires cancellation
or lease expiry with safe duplicate semantics, preserving prior attempt logs.
Repeated schema/permission violations, fabricated provenance, or unsafe actions
trigger quarantine, and quarantined agents receive no new tasks (Steps 51-54).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .agents import BaseBoundedAgent
from .contracts import AgentLifecycleState, AgentTask, TaskOutcome
from .errors import (
    PermanentAgentError,
    QuarantineTriggered,
    TransientAgentError,
)
from .registry import AgentRegistry


@dataclass(frozen=True, slots=True)
class ReconciliationReport:
    """Report the reconciled outcome of an attempt whose delivery was unknown (Step 52)."""

    task_id: str
    agent_id: str
    outcome: TaskOutcome
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a detached report record."""
        return {
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "outcome": self.outcome.value,
            "detail": self.detail,
        }


@dataclass
class AttemptLedger:
    """Preserve prior attempt logs and artifacts across reassignment (Step 53)."""

    attempts: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def record(self, task_id: str, agent_id: str, outcome: TaskOutcome, detail: str = "") -> None:
        """Append one attempt outcome without ever deleting earlier attempts."""
        self.attempts.setdefault(task_id, []).append({
            "agent_id": agent_id, "outcome": outcome.value, "detail": detail,
        })

    def count(self, task_id: str) -> int:
        """Return the number of recorded attempts for one task."""
        return len(self.attempts.get(task_id, ()))

    def history(self, task_id: str) -> tuple[dict[str, Any], ...]:
        """Return the preserved attempt history for one task."""
        return tuple(self.attempts.get(task_id, ()))


class RecoveryManager:
    """Coordinate retries, reconciliation, reassignment, and quarantine (Steps 51-54)."""

    def __init__(
        self,
        registry: AgentRegistry,
        *,
        quarantine_threshold: int = 3,
        lease_ticks: int = 2,
    ) -> None:
        """Configure quarantine sensitivity and lease duration for safe reassignment."""
        if quarantine_threshold < 1:
            raise ValueError("quarantine_threshold must be at least one")
        if lease_ticks < 1:
            raise ValueError("lease_ticks must be at least one")
        self.registry = registry
        self.quarantine_threshold = quarantine_threshold
        self.lease_ticks = lease_ticks
        self.ledger = AttemptLedger()
        self._violation_counts: dict[str, int] = {}
        self._leases: dict[str, int] = {}

    def classify_failure(self, reason: str) -> type[Exception]:
        """Classify a failure reason as transient or permanent (Step 51)."""
        lowered = reason.casefold()
        if "timeout" in lowered or "unknown" in lowered or "transient" in lowered:
            return TransientAgentError
        return PermanentAgentError

    def should_retry(self, task: AgentTask, agent: BaseBoundedAgent, attempt_count: int,
                     *, transient: bool) -> bool:
        """Decide whether policy permits one more attempt by this agent (Step 51)."""
        retry = agent.descriptor.retry
        return retry.allows(
            attempt_count,
            idempotent=task.idempotency_key.strip() != "",
            transient=transient,
        )

    def reconcile_timeout(self, task: AgentTask, agent: BaseBoundedAgent) -> ReconciliationReport:
        """Ask the worker for its recorded outcome before any retry (Step 52)."""
        recorded = agent.reconcile(task.task_id)
        if recorded is not None:
            outcome = TaskOutcome.COMPLETED
            detail = "worker recorded a completed result after the timeout"
            self.ledger.record(task.task_id, agent.agent_id, outcome, detail)
        else:
            outcome = TaskOutcome.UNKNOWN
            detail = "worker reported no recorded outcome; execution state unknown"
            self.ledger.record(task.task_id, agent.agent_id, outcome, detail)
        return ReconciliationReport(task.task_id, agent.agent_id, outcome, detail)

    def issue_lease(self, task_id: str, agent_id: str) -> int:
        """Grant one execution lease that safe reassignment must respect (Step 53)."""
        self._leases[f"{task_id}:{agent_id}"] = self.lease_ticks
        return self.lease_ticks

    def lease_expired(self, task_id: str, agent_id: str) -> bool:
        """Check whether the lease for one attempt has expired."""
        key = f"{task_id}:{agent_id}"
        return self._leases.get(key, 0) <= 0

    def tick_leases(self) -> None:
        """Advance every outstanding lease by one tick toward expiry."""
        for key in list(self._leases):
            self._leases[key] -= 1
            if self._leases[key] <= 0:
                del self._leases[key]

    def allow_reassignment(self, task: AgentTask, agent: BaseBoundedAgent,
                           *, cancelled: bool) -> bool:
        """Permit reassignment only after cancellation or lease expiry (Step 53)."""
        if cancelled:
            return True
        return self.lease_expired(task.task_id, agent.agent_id)

    def record_violation(self, agent_id: str, rule: str, detail: str = "") -> int:
        """Count one policy violation and quarantine past the threshold (Step 54)."""
        self._violation_counts[agent_id] = self._violation_counts.get(agent_id, 0) + 1
        count = self._violation_counts[agent_id]
        self.ledger.record("policy", agent_id, TaskOutcome.FAILED, f"{rule}: {detail}")
        if count >= self.quarantine_threshold:
            self.quarantine(agent_id, rule, detail)
        return count

    def quarantine(self, agent_id: str, rule: str, detail: str = "") -> None:
        """Move an agent into quarantine; quarantined agents receive no new tasks."""
        self.registry.set_state(agent_id, AgentLifecycleState.QUARANTINED)
        raise QuarantineTriggered(agent_id, rule, detail)

    def violations(self, agent_id: str) -> int:
        """Return the recorded violation count for one agent."""
        return self._violation_counts.get(agent_id, 0)

    def to_dict(self) -> dict[str, Any]:
        """Return a detached recovery-state record."""
        return {
            "quarantine_threshold": self.quarantine_threshold,
            "lease_ticks": self.lease_ticks,
            "violations": dict(self._violation_counts),
            "leases": dict(self._leases),
            "quarantined": list(self.registry.quarantined_agents()),
        }


def is_transient(reason: str | None) -> bool:
    """Return whether a failure reason looks transient for retry policy."""
    if not reason:
        return False
    lowered = reason.casefold()
    return "timeout" in lowered or "unknown" in lowered or "transient" in lowered
