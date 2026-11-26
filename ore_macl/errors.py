"""MACL exception hierarchy mapping the specification's failure modes and safeguards."""
from __future__ import annotations


class MACLError(RuntimeError):
    """Base class for multi-agent coordination layer failures."""


class AgentRejectedTask(MACLError):
    """An agent declined a task envelope with an explicit reason (Step 18)."""

    def __init__(self, agent_id: str, reason: str) -> None:
        """Record which agent rejected the envelope and why."""
        super().__init__(f"agent {agent_id!r} rejected the task: {reason}")
        self.agent_id = agent_id
        self.reason = reason


class TransientAgentError(MACLError):
    """A recoverable execution failure that retry policy may consider (Step 51)."""


class PermanentAgentError(MACLError):
    """A non-recoverable execution failure; retrying will not help (Step 51)."""


class PolicyViolation(MACLError):
    """A schema, permission, provenance, or safety rule violation (Step 54)."""

    def __init__(self, agent_id: str, rule: str, detail: str = "") -> None:
        """Record the violating agent, rule, and detail for quarantine accounting."""
        message = f"policy violation by {agent_id!r}: {rule}"
        super().__init__(f"{message}: {detail}" if detail else message)
        self.agent_id = agent_id
        self.rule = rule
        self.detail = detail


class QuarantineTriggered(PolicyViolation):
    """Repeated or severe misbehavior moved an agent into quarantine (Step 54)."""


class BudgetExhausted(MACLError):
    """A task or goal exceeded its declared cost or concurrency budget (Section 10)."""


class DeadlineExceeded(MACLError):
    """A task or goal exceeded its declared wall-clock deadline (Section 10)."""


class TaskBlockedError(MACLError):
    """A task cannot proceed and requires information, fallback, or human escalation."""


class CyclicDependencyError(MACLError):
    """The task dependency graph contains a cycle (Step 9)."""

    def __init__(self, cycle_nodes: tuple[str, ...]) -> None:
        """Report the exact nodes participating in the detected cycle."""
        super().__init__(f"task dependency cycle detected among: {sorted(cycle_nodes)}")
        self.cycle_nodes = tuple(sorted(cycle_nodes))


class OutOfScopeAccess(MACLError):
    """An agent attempted to read or write a field outside its permitted scopes (Step 4)."""


class StaleRevisionError(MACLError):
    """A result or proposal references an obsolete workspace revision (Step 27)."""


class FabricatedProvenanceError(PolicyViolation):
    """A result cited evidence artifacts that do not exist (Sections 9 and 10)."""


class MARLProductionControlError(MACLError):
    """MARL attempted to influence production ORE control without approval (Step 59)."""
