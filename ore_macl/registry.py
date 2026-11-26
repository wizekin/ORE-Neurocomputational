"""Source-ordered MACL stage registry and the bounded agent lifecycle registry.

The registry module follows the ORE point-package convention: a source-ordered,
immutable operation registry that traces every implementation unit back to the
ranked Point 11 specification, plus the runtime ``AgentRegistry`` that owns agent
identity, lifecycle states, health tracking, and safe replacement (spec Steps 1-7).
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from threading import RLock
from typing import Any

from .contracts import AgentDescriptor, AgentLifecycleState, AutonomyLevel
from .errors import MACLError


@dataclass(frozen=True)
class MACLStageSpec:
    """Describe one source-sequenced coordination stage and its reference interpretation."""

    stage_id: str
    name: str
    source_steps: tuple[int, ...]
    purpose: str
    mathematical_form: str
    biological_reference: str
    evidence_level: str = "distributed_systems_account_with_computational_abstraction"

    def to_dict(self) -> dict[str, Any]:
        """Return plain serializable metadata."""
        return asdict(self)


_ROWS = (
    ("A", "Agent lifecycle and registry", tuple(range(1, 8)),
     "Classify autonomy levels, register contracts, validate capabilities, enforce least privilege, track health, version policies, and support safe replacement",
     "state(agent) in {registered, ready, busy, degraded, quarantined, disabled}; contract(agent) = (identity, role, capabilities, schemas, permissions, limits)",
     "Prefrontal supervisor circuits maintain task sets and gates; anterior cingulate and control networks monitor participant state."),
    ("B", "Task decomposition and allocation", tuple(range(8, 17)),
     "Decompose goals into typed tasks, build a cycle-checked DAG, match and score eligible agents, keep deterministic fallbacks, and select a justified topology",
     "owner*(t) = argmax_a sum_j w_j * f_j(a, t) subject to eligibility gates, capacity, dependency order, and budgets",
     "Frontoparietal control and basal ganglia action-selection circuits support task-set decomposition and bounded participant selection."),
    ("C", "Supervisor/worker protocol", tuple(range(17, 24)),
     "Create bounded task envelopes, accept or reject explicitly, execute within scope, emit progress, monitor budgets, and submit typed results validated against contracts",
     "accept(agent, envelope) in {accept, reject(reason)}; completion iff contract(result) satisfied",
     "Hierarchical prefrontal supervision with progressive action gating models bounded delegation and completion criteria."),
    ("D", "Blackboard and shared evidence", tuple(range(24, 31)),
     "Reuse the existing shared workspace, reference artifacts compactly, attach provenance, check revisions, separate proposals from facts, deduplicate idempotently, and preserve conflicts",
     "finding.base_revision <= current_revision; dedup by fingerprint; conflicts retained as explicit records",
     "Hippocampal-cortical contextual binding and persistent memory systems ground revision-bound, provenance-carrying shared state."),
    ("E", "Peer-to-peer cooperation", tuple(range(31, 37)),
     "Restrict peer channels, discover capabilities without broadcast, prevent circular delegation, resolve ownership, handle partitions, and escalate failed negotiation",
     "depth(delegation chain) <= d_max; message expiry via TTL with idempotency keys",
     "Social-circuits perspective: temporoparietal and medial prefrontal regions support modelling other participants within bounded reciprocity."),
    ("F", "Debate, critic, and verification", tuple(range(37, 44)),
     "Generate independent proposals, measure independence, evaluate evidence rather than votes, structure critic feedback, bound revision cycles, verify critical outputs, and preserve minority findings",
     "independence(a1, a2) = 1 - overlap(data, model lineage, tools); accept by evidence score E, not vote count",
     "Anterior cingulate conflict monitoring and lateral prefrontal control support evidence-based evaluation and bounded revision."),
    ("G", "Aggregation and integration", tuple(range(44, 51)),
     "Validate schema and scope, validate evidence lineage, aggregate only compatible results, resolve conflicts explicitly, publish through existing integration, update dependencies, and evaluate goal progress",
     "integrate(result) iff schema(result) valid and lineage(result) resolvable; goal progress from accepted outcomes only",
     "Cortical integration and supervised state consolidation accept only validated contributions into shared representations."),
    ("H", "Recovery and performance learning", tuple(range(51, 59)),
     "Retry transient idempotent failures, reconcile timeouts, reassign safely, quarantine misbehavior, track performance with uncertainty, learn from validated outcomes, and shadow learned delegation",
     "p_i | data ~ Beta(alpha + successes, beta + failures); reward = accepted progress - cost - defects - penalties",
     "Corticostriatal reinforcement and neuromodulatory error signals update reliability and policy selection from validated outcomes."),
    ("I", "Optional MARL and swarm", tuple(range(59, 67)),
     "Isolate MARL environments, use CTDE as a research pattern, select algorithms by structure, address non-stationarity, run bounded PSO parameter search and ACO path search, prevent optimizer authority leakage, and defer swarm robotics",
     "PSO: v <- w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x); ACO: P(edge) proportional to tau^alpha * eta^beta",
     "Population-level analogies only; swarm coordination is a computational abstraction without a direct neural claim."),
)

STAGE_REGISTRY: tuple[MACLStageSpec, ...] = tuple(
    MACLStageSpec(f"MACL-{index}", *row[1:])  # type: ignore[arg-type] - row[0] is the source letter
    for index, row in enumerate(_ROWS, start=1)
)

_STEP_TO_STAGE: dict[int, str] = {
    step: spec.stage_id for spec in STAGE_REGISTRY for step in spec.source_steps
}


def stage_for_step(step: int) -> str:
    """Return the stage identifier that owns a source step number."""
    if step not in _STEP_TO_STAGE:
        raise ValueError(f"unknown source step: {step}")
    return _STEP_TO_STAGE[step]


def registry_as_dicts() -> list[dict[str, Any]]:
    """Return all stages in exact source order."""
    return [item.to_dict() for item in STAGE_REGISTRY]


_VALID_TRANSITIONS: dict[AgentLifecycleState, frozenset[AgentLifecycleState]] = {
    AgentLifecycleState.REGISTERED: frozenset({
        AgentLifecycleState.READY, AgentLifecycleState.DISABLED,
    }),
    AgentLifecycleState.READY: frozenset({
        AgentLifecycleState.BUSY, AgentLifecycleState.DEGRADED,
        AgentLifecycleState.QUARANTINED, AgentLifecycleState.DISABLED,
    }),
    AgentLifecycleState.BUSY: frozenset({
        AgentLifecycleState.READY, AgentLifecycleState.DEGRADED,
        AgentLifecycleState.QUARANTINED, AgentLifecycleState.DISABLED,
    }),
    AgentLifecycleState.DEGRADED: frozenset({
        AgentLifecycleState.READY, AgentLifecycleState.QUARANTINED,
        AgentLifecycleState.DISABLED,
    }),
    AgentLifecycleState.QUARANTINED: frozenset({AgentLifecycleState.DISABLED}),
    AgentLifecycleState.DISABLED: frozenset({AgentLifecycleState.REGISTERED}),
}

#: States in which an agent may be assigned new work.
_ASSIGNABLE_STATES = frozenset({
    AgentLifecycleState.READY, AgentLifecycleState.BUSY, AgentLifecycleState.DEGRADED,
})


class AgentRegistry:
    """Maintain agent descriptors, lifecycle states, and health (spec Steps 2, 5, 7)."""

    def __init__(self) -> None:
        """Create an empty, thread-safe registry."""
        self._lock = RLock()
        self._descriptors: dict[str, AgentDescriptor] = {}
        self._states: dict[str, AgentLifecycleState] = {}
        self._heartbeats: dict[str, int] = {}
        self._active_tasks: dict[str, int] = {}
        self._history: list[dict[str, Any]] = []
        self._release_tokens: set[str] = set()

    def register(self, descriptor: AgentDescriptor, *, replace: bool = False) -> AgentDescriptor:
        """Validate and register one agent contract; replacement is explicit (Steps 2, 7)."""
        if not isinstance(descriptor, AgentDescriptor):
            raise TypeError("descriptor must be an AgentDescriptor")
        with self._lock:
            if descriptor.agent_id in self._descriptors and not replace:
                raise ValueError(f"agent identity is already registered: {descriptor.agent_id}")
            self._descriptors[descriptor.agent_id] = descriptor
            self._states[descriptor.agent_id] = AgentLifecycleState.REGISTERED
            self._heartbeats[descriptor.agent_id] = 0
            self._active_tasks[descriptor.agent_id] = 0
            self._history.append({
                "event": "register", "agent_id": descriptor.agent_id,
                "version": descriptor.agent_version, "replaced_existing": bool(replace),
            })
        return descriptor

    def get(self, agent_id: str) -> AgentDescriptor:
        """Return the registered descriptor or raise for an unknown identity."""
        with self._lock:
            if agent_id not in self._descriptors:
                raise KeyError(f"unknown agent: {agent_id}")
            return self._descriptors[agent_id]

    def contains(self, agent_id: str) -> bool:
        """Return whether an agent identity is registered."""
        with self._lock:
            return agent_id in self._descriptors

    def set_state(self, agent_id: str, state: AgentLifecycleState) -> AgentLifecycleState:
        """Apply one validated lifecycle transition and record the event."""
        if not isinstance(state, AgentLifecycleState):
            raise TypeError("state must be an AgentLifecycleState")
        with self._lock:
            if agent_id not in self._descriptors:
                raise KeyError(f"unknown agent: {agent_id}")
            current = self._states[agent_id]
            if state is not current and state not in _VALID_TRANSITIONS[current]:
                raise MACLError(
                    f"invalid lifecycle transition for {agent_id}: "
                    f"{current.value} -> {state.value}"
                )
            self._states[agent_id] = state
            self._history.append({
                "event": "transition", "agent_id": agent_id,
                "from": current.value, "to": state.value,
            })
            return state

    def release_from_quarantine(self, agent_id: str, token: str) -> AgentLifecycleState:
        """Permit an explicit administrative release back to the registered state (Step 54)."""
        with self._lock:
            if agent_id not in self._descriptors:
                raise KeyError(f"unknown agent: {agent_id}")
            if self._states[agent_id] is not AgentLifecycleState.QUARANTINED:
                raise MACLError("only quarantined agents can be released")
            if not token or not token.strip():
                raise ValueError("release token must be non-empty")
            self._release_tokens.add(token)
            self._states[agent_id] = AgentLifecycleState.REGISTERED
            self._history.append({
                "event": "quarantine_release", "agent_id": agent_id, "token": token,
            })
            return self._states[agent_id]

    def state(self, agent_id: str) -> AgentLifecycleState:
        """Return the agent's current lifecycle state."""
        with self._lock:
            if agent_id not in self._descriptors:
                raise KeyError(f"unknown agent: {agent_id}")
            return self._states[agent_id]

    def heartbeat(self, agent_id: str) -> int:
        """Record one heartbeat; a missed heartbeat never erases history (Step 5)."""
        with self._lock:
            if agent_id not in self._descriptors:
                raise KeyError(f"unknown agent: {agent_id}")
            self._heartbeats[agent_id] += 1
            return self._heartbeats[agent_id]

    def acquire_capacity(self, agent_id: str, amount: int = 1) -> int:
        """Claim one concurrency slot for a dispatched task."""
        with self._lock:
            descriptor = self.get(agent_id)
            if self._states[agent_id] not in _ASSIGNABLE_STATES:
                raise MACLError(
                    f"agent {agent_id} is {self._states[agent_id].value} and cannot take tasks"
                )
            if self._active_tasks[agent_id] + amount > descriptor.resources.max_concurrent_tasks:
                raise MACLError(
                    f"agent {agent_id} is at its concurrency limit "
                    f"({descriptor.resources.max_concurrent_tasks})"
                )
            self._active_tasks[agent_id] += amount
            return self._active_tasks[agent_id]

    def release_capacity(self, agent_id: str, amount: int = 1) -> int:
        """Release one concurrency slot after an attempt terminates."""
        with self._lock:
            if agent_id not in self._descriptors:
                raise KeyError(f"unknown agent: {agent_id}")
            self._active_tasks[agent_id] = max(0, self._active_tasks[agent_id] - amount)
            return self._active_tasks[agent_id]

    def active_tasks(self, agent_id: str) -> int:
        """Return the agent's currently claimed concurrency slots."""
        with self._lock:
            if agent_id not in self._descriptors:
                raise KeyError(f"unknown agent: {agent_id}")
            return self._active_tasks[agent_id]

    def assignable(self, agent_id: str) -> bool:
        """Return whether the agent may receive new work right now (Step 12)."""
        with self._lock:
            if agent_id not in self._descriptors:
                return False
            if self._states[agent_id] not in _ASSIGNABLE_STATES:
                return False
            descriptor = self._descriptors[agent_id]
            return self._active_tasks[agent_id] < descriptor.resources.max_concurrent_tasks

    def quarantined_agents(self) -> tuple[str, ...]:
        """Return quarantined agents, which must receive no new tasks (Section 9)."""
        with self._lock:
            return tuple(sorted(
                agent_id for agent_id, state in self._states.items()
                if state is AgentLifecycleState.QUARANTINED
            ))

    def find_by_capability(self, capability: str) -> tuple[AgentDescriptor, ...]:
        """Return agents whose declared capability contracts include the requested one."""
        if not capability.strip():
            raise ValueError("capability must be non-empty")
        with self._lock:
            return tuple(
                self._descriptors[agent_id]
                for agent_id in sorted(self._descriptors)
                if capability in self._descriptors[agent_id].capability_ids
            )

    def find_by_role(self, role: str) -> tuple[AgentDescriptor, ...]:
        """Return agents registered under one role."""
        if not role.strip():
            raise ValueError("role must be non-empty")
        with self._lock:
            return tuple(
                self._descriptors[agent_id]
                for agent_id in sorted(self._descriptors)
                if self._descriptors[agent_id].role == role
            )

    def by_autonomy(self, level: AutonomyLevel) -> tuple[AgentDescriptor, ...]:
        """Return agents registered at one autonomy level (Step 1)."""
        with self._lock:
            return tuple(
                self._descriptors[agent_id]
                for agent_id in sorted(self._descriptors)
                if self._descriptors[agent_id].autonomy is level
            )

    def replace_agent(self, replacement: AgentDescriptor) -> AgentDescriptor:
        """Replace an agent only when the contract stays compatible (Step 7)."""
        if not isinstance(replacement, AgentDescriptor):
            raise TypeError("replacement must be an AgentDescriptor")
        with self._lock:
            existing = self._descriptors.get(replacement.agent_id)
            if existing is None:
                raise KeyError(f"cannot replace an unregistered agent: {replacement.agent_id}")
            if existing.role != replacement.role:
                raise MACLError("safe replacement requires the same role")
            if not set(existing.capability_ids).issubset(set(replacement.capability_ids)):
                raise MACLError("safe replacement cannot drop declared capabilities")
            if not set(replacement.produced_output_schema.fields) >= set(existing.produced_output_schema.fields):
                raise MACLError("safe replacement cannot remove produced output fields")
            self._descriptors[replacement.agent_id] = replacement
            self._history.append({
                "event": "replace", "agent_id": replacement.agent_id,
                "old_version": existing.agent_version, "new_version": replacement.agent_version,
            })
            return replacement

    def list_descriptors(self) -> tuple[AgentDescriptor, ...]:
        """Return every registered descriptor sorted by agent ID."""
        with self._lock:
            return tuple(self._descriptors[agent_id] for agent_id in sorted(self._descriptors))

    def snapshot(self) -> dict[str, Any]:
        """Return a detached registry status snapshot."""
        with self._lock:
            return {
                "agents": {
                    agent_id: {
                        "state": self._states[agent_id].value,
                        "version": self._descriptors[agent_id].agent_version,
                        "role": self._descriptors[agent_id].role,
                        "autonomy": self._descriptors[agent_id].autonomy.value,
                        "heartbeats": self._heartbeats[agent_id],
                        "active_tasks": self._active_tasks[agent_id],
                        "max_concurrent_tasks": self._descriptors[agent_id].resources.max_concurrent_tasks,
                    }
                    for agent_id in sorted(self._descriptors)
                },
                "quarantined": self.quarantined_agents(),
                "history": list(self._history),
            }

    def history(self) -> tuple[dict[str, Any], ...]:
        """Return the recorded registry event history."""
        with self._lock:
            return tuple(dict(item) for item in self._history)


def classify_component(
    *,
    makes_action_choices: bool,
    uses_tools: bool = False,
    supervises_other_agents: bool = False,
    justification: str = "",
) -> AutonomyLevel:
    """Classify a component's autonomy level and require justification (Step 1)."""
    if supervises_other_agents:
        if not justification.strip():
            raise ValueError("supervisor autonomy requires an explicit justification")
        return AutonomyLevel.SUPERVISOR
    if makes_action_choices:
        if not justification.strip():
            raise ValueError("bounded autonomy requires an explicit justification")
        return AutonomyLevel.BOUNDED_AUTONOMOUS_AGENT
    if uses_tools:
        return AutonomyLevel.TOOL_USING_WORKER
    return AutonomyLevel.DETERMINISTIC_SPECIALIST


def assignable_states() -> frozenset[AgentLifecycleState]:
    """Expose the states in which agents may receive new work."""
    return frozenset(_ASSIGNABLE_STATES)


def valid_transitions() -> Mapping[AgentLifecycleState, frozenset[AgentLifecycleState]]:
    """Expose the validated lifecycle transition table."""
    return dict(_VALID_TRANSITIONS)


def iterate_heartbeats(registry: AgentRegistry, agents: Iterable[str]) -> dict[str, int]:
    """Record one heartbeat for each listed agent."""
    return {agent_id: registry.heartbeat(agent_id) for agent_id in agents}
