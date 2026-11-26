"""State classification, consistency requirements, and threat models (Steps 2-4).

Every ORE state schema must be classified as authoritative operational, mergeable
collaborative, immutable event/provenance, or agent-local transient before it can
participate in the distributed coordination layer. Unclassified schemas are
rejected outright (Step 2).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Mapping


class StateClass(str, Enum):
    """The four core state separations of the Point 17 architecture."""

    AUTHORITATIVE_OPERATIONAL = "authoritative_operational"
    MERGEABLE_COLLABORATIVE = "mergeable_collaborative"
    IMMUTABLE_EVENT_PROVENANCE = "immutable_event_provenance"
    AGENT_LOCAL_TRANSIENT = "agent_local_transient"


class ConsistencyModel(str, Enum):
    """Consistency requirements selectable per record type (Step 3)."""

    LINEARIZABLE = "linearizable"
    EVENTUAL = "eventual"
    CAUSAL = "causal"
    READ_YOUR_WRITES = "read_your_writes"
    LOCAL_ONLY = "local_only"


class FailureModel(str, Enum):
    """Failure assumptions the coordination layer is built to tolerate (Step 4)."""

    CRASH_STOP = "crash_stop"
    CRASH_RECOVERY = "crash_recovery"
    NETWORK_PARTITION = "network_partition"
    DELAYED_MESSAGES = "delayed_messages"
    DUPLICATION = "duplication"
    REORDERING = "reordering"
    CLOCK_DRIFT = "clock_drift"
    DISK_LOSS = "disk_loss"
    OPERATOR_ERROR = "operator_error"


class StateClassificationError(ValueError):
    """Raised when a schema is used without an explicit state classification."""


#: Consistency models each state class may legally require (Step 3).
_ALLOWED_CONSISTENCY: Mapping[StateClass, frozenset[str]] = {
    StateClass.AUTHORITATIVE_OPERATIONAL: frozenset(
        {ConsistencyModel.LINEARIZABLE.value, ConsistencyModel.READ_YOUR_WRITES.value}
    ),
    StateClass.MERGEABLE_COLLABORATIVE: frozenset(
        {
            ConsistencyModel.EVENTUAL.value,
            ConsistencyModel.CAUSAL.value,
            ConsistencyModel.READ_YOUR_WRITES.value,
        }
    ),
    StateClass.IMMUTABLE_EVENT_PROVENANCE: frozenset(
        {ConsistencyModel.CAUSAL.value, ConsistencyModel.EVENTUAL.value}
    ),
    StateClass.AGENT_LOCAL_TRANSIENT: frozenset({ConsistencyModel.LOCAL_ONLY.value}),
}


def _non_empty(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True, slots=True)
class ThreatModel:
    """Record the deployment threat model and the protocol family it justifies (Step 4)."""

    tolerated: tuple[FailureModel, ...]
    byzantine_participants: bool = False
    operator_error_expected: bool = False

    def __post_init__(self) -> None:
        """Normalize the tolerated failure set and keep Byzantine assumptions explicit."""
        values = tuple(FailureModel(item) for item in self.tolerated)
        if not values:
            raise ValueError("tolerated failure set must not be empty")
        object.__setattr__(self, "tolerated", values)

    def requires_byzantine_tolerance(self) -> bool:
        """Return whether the threat model demands BFT rather than crash tolerance."""
        return self.byzantine_participants


def recommended_protocol_family(threat: ThreatModel) -> str:
    """Select the consensus family implied by a threat model (Step 4)."""
    if threat.requires_byzantine_tolerance():
        return "bft"
    return "raft"


@dataclass(frozen=True, slots=True)
class StateObjectSpec:
    """Describe the classification and requirements of one ORE state schema."""

    schema_id: str
    state_class: StateClass
    consistency: ConsistencyModel
    description: str = ""
    replication_factor: int = 1
    retention: str = "session"
    examples: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate schema identity, class/consistency agreement, and replication."""
        object.__setattr__(self, "schema_id", _non_empty("schema_id", self.schema_id))
        if not isinstance(self.state_class, StateClass):
            raise ValueError("state_class must be a StateClass value")
        if not isinstance(self.consistency, ConsistencyModel):
            raise ValueError("consistency must be a ConsistencyModel value")
        allowed = _ALLOWED_CONSISTENCY[self.state_class]
        if self.consistency.value not in allowed:
            raise ValueError(
                f"consistency {self.consistency.value} is not legal for state class "
                f"{self.state_class.value}; allowed: {sorted(allowed)}"
            )
        if not isinstance(self.replication_factor, int) or self.replication_factor < 1:
            raise ValueError("replication_factor must be a positive integer")
        if self.retention not in {"session", "durable", "archival", "ephemeral"}:
            raise ValueError("unsupported retention policy")
        object.__setattr__(self, "examples", tuple(self.examples))

    def to_dict(self) -> dict[str, object]:
        """Return the classification record as a plain dictionary."""
        return {
            "schema_id": self.schema_id,
            "state_class": self.state_class.value,
            "consistency": self.consistency.value,
            "description": self.description,
            "replication_factor": self.replication_factor,
            "retention": self.retention,
            "examples": list(self.examples),
        }


class StateSchemaRegistry:
    """Registry that only lets classified schemas enter the coordination layer."""

    def __init__(self) -> None:
        """Create an empty classification registry."""
        self._specs: dict[str, StateObjectSpec] = {}

    def register(self, spec: StateObjectSpec) -> StateObjectSpec:
        """Register one classification; duplicate schema IDs are rejected."""
        if not isinstance(spec, StateObjectSpec):
            raise TypeError("register expects a StateObjectSpec")
        if spec.schema_id in self._specs:
            raise StateClassificationError(f"schema {spec.schema_id} is already classified")
        self._specs[spec.schema_id] = spec
        return spec

    def register_all(self, specs: Iterable[StateObjectSpec]) -> tuple[StateObjectSpec, ...]:
        """Register several classifications atomically."""
        items = tuple(specs)
        for spec in items:
            self.register(spec)
        return items

    def require(self, schema_id: str) -> StateObjectSpec:
        """Return the classification or raise: unclassified schemas are rejected (Step 2)."""
        schema_id = _non_empty("schema_id", schema_id)
        spec = self._specs.get(schema_id)
        if spec is None:
            raise StateClassificationError(
                f"schema {schema_id} has no state classification; classify it before use"
            )
        return spec

    def get(self, schema_id: str) -> StateObjectSpec | None:
        """Return the classification for a schema or None when absent."""
        return self._specs.get(schema_id)

    def is_classified(self, schema_id: str) -> bool:
        """Return whether a schema carries an accepted classification."""
        return schema_id in self._specs

    def specs(self) -> tuple[StateObjectSpec, ...]:
        """Return all classifications ordered by schema identifier."""
        return tuple(self._specs[key] for key in sorted(self._specs))

    def as_dicts(self) -> list[dict[str, object]]:
        """Return all classification records as plain dictionaries."""
        return [spec.to_dict() for spec in self.specs()]


def default_state_schema_registry() -> StateSchemaRegistry:
    """Build the canonical ORE state classification for the coordination layer.

    The registry encodes the Step 2 core state separation: task ownership, leases,
    coordinator epochs, workflow transitions, committed decisions and configuration
    versions are authoritative; selected agent presence, annotations, observations
    and contribution sets are mergeable; events and provenance are immutable; and
    agent scratch state stays local.
    """
    registry = StateSchemaRegistry()
    registry.register_all(
        (
            StateObjectSpec(
                schema_id="ore.task.ownership",
                state_class=StateClass.AUTHORITATIVE_OPERATIONAL,
                consistency=ConsistencyModel.LINEARIZABLE,
                description="Task ownership records and lease assignments",
                replication_factor=3,
                retention="durable",
                examples=("task_id", "owner", "lease_expiry"),
            ),
            StateObjectSpec(
                schema_id="ore.coordinator.epoch",
                state_class=StateClass.AUTHORITATIVE_OPERATIONAL,
                consistency=ConsistencyModel.LINEARIZABLE,
                description="Coordinator term/epoch and fencing tokens",
                replication_factor=3,
                retention="durable",
                examples=("epoch", "leader_id"),
            ),
            StateObjectSpec(
                schema_id="ore.workflow.transition",
                state_class=StateClass.AUTHORITATIVE_OPERATIONAL,
                consistency=ConsistencyModel.LINEARIZABLE,
                description="Workflow state transitions and committed decisions",
                replication_factor=3,
                retention="durable",
            ),
            StateObjectSpec(
                schema_id="ore.config.version",
                state_class=StateClass.AUTHORITATIVE_OPERATIONAL,
                consistency=ConsistencyModel.LINEARIZABLE,
                description="Cluster configuration versions and membership epochs",
                replication_factor=3,
                retention="durable",
            ),
            StateObjectSpec(
                schema_id="ore.agent.presence",
                state_class=StateClass.MERGEABLE_COLLABORATIVE,
                consistency=ConsistencyModel.EVENTUAL,
                description="Non-authoritative agent presence records",
                replication_factor=3,
                retention="session",
            ),
            StateObjectSpec(
                schema_id="ore.agent.annotation",
                state_class=StateClass.MERGEABLE_COLLABORATIVE,
                consistency=ConsistencyModel.EVENTUAL,
                description="Independently authored notes and annotations",
                replication_factor=3,
                retention="session",
            ),
            StateObjectSpec(
                schema_id="ore.observation.contribution",
                state_class=StateClass.MERGEABLE_COLLABORATIVE,
                consistency=ConsistencyModel.CAUSAL,
                description="Append-only independent observation contribution sets",
                replication_factor=3,
                retention="archival",
            ),
            StateObjectSpec(
                schema_id="ore.event.history",
                state_class=StateClass.IMMUTABLE_EVENT_PROVENANCE,
                consistency=ConsistencyModel.CAUSAL,
                description="Immutable event identifiers, actors, and lineage",
                replication_factor=3,
                retention="archival",
            ),
            StateObjectSpec(
                schema_id="ore.provenance.envelope",
                state_class=StateClass.IMMUTABLE_EVENT_PROVENANCE,
                consistency=ConsistencyModel.CAUSAL,
                description="Provenance envelopes with digests and causal parents",
                replication_factor=3,
                retention="archival",
            ),
            StateObjectSpec(
                schema_id="ore.agent.scratch",
                state_class=StateClass.AGENT_LOCAL_TRANSIENT,
                consistency=ConsistencyModel.LOCAL_ONLY,
                description="Private scratch state, temporary plans, and caches",
                replication_factor=1,
                retention="ephemeral",
            ),
            StateObjectSpec(
                schema_id="ore.usg.semantic.assertion",
                state_class=StateClass.AUTHORITATIVE_OPERATIONAL,
                consistency=ConsistencyModel.LINEARIZABLE,
                description=(
                    "USG semantic assertions promoted only after validation; CRDT "
                    "convergence alone never promotes a claim to truth"
                ),
                replication_factor=3,
                retention="durable",
            ),
        )
    )
    return registry


__all__ = [
    "StateClass",
    "ConsistencyModel",
    "FailureModel",
    "StateClassificationError",
    "StateObjectSpec",
    "StateSchemaRegistry",
    "ThreatModel",
    "recommended_protocol_family",
    "default_state_schema_registry",
]
