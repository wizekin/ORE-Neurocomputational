"""Deterministic authoritative state machine (Steps 13, 18, and 22).

The state machine holds the authoritative operational records: task assignment,
lease ownership, coordinator epochs, workflow transitions, committed policy
versions, and recovery checkpoint references. Commands are deterministic and
versioned; committed commands apply in log order, and command identifiers plus
deduplication records make client retries idempotent. The accepted command set
is deliberately narrow so high-volume observations never touch consensus.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

#: Control-plane command classes allowed into the consensus log (Step 22).
COMMAND_TYPES = frozenset(
    {
        "assign_task",
        "release_task",
        "complete_task",
        "transition_workflow",
        "set_policy",
        "advance_epoch",
        "record_checkpoint",
        "config_change",
        "approve_semantic_assertion",
        "no_op",
    }
)


def _non_empty(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True, slots=True)
class StateMachineCommand:
    """One deterministic, versioned state-machine command (Step 13)."""

    command_id: str
    command_type: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = "0.1.0"

    def __post_init__(self) -> None:
        """Validate identity, type, and payload shape."""
        object.__setattr__(self, "command_id", _non_empty("command_id", self.command_id))
        if self.command_type not in COMMAND_TYPES:
            raise ValueError(
                f"command type {self.command_type!r} is not an authoritative control-plane "
                "operation; route it outside consensus (Step 22)"
            )
        object.__setattr__(self, "payload", dict(self.payload))
        object.__setattr__(self, "schema_version", _non_empty("schema_version", self.schema_version))

    def to_dict(self) -> dict[str, Any]:
        """Return the command as a plain dictionary."""
        return {
            "command_id": self.command_id,
            "command_type": self.command_type,
            "payload": dict(self.payload),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StateMachineCommand":
        """Rebuild a command from its dictionary form."""
        return cls(
            command_id=str(payload["command_id"]),
            command_type=str(payload["command_type"]),
            payload=dict(payload.get("payload", {})),
            schema_version=str(payload.get("schema_version", "0.1.0")),
        )


class CommandStatus:
    """Deduplication outcomes when applying a command to the state machine."""

    APPLIED = "applied"
    DUPLICATE = "duplicate"


@dataclass(slots=True)
class AuthoritativeStateMachine:
    """Deterministic application of committed commands in log order (Step 18).

    Applying the same entries in the same order yields identical state across
    replicas; command IDs with a deduplication record make retries idempotent so
    duplicate client commands never create duplicate task assignments or
    workflow transitions.
    """

    tasks: dict[str, dict[str, Any]] = field(default_factory=dict)
    workflows: dict[str, dict[str, Any]] = field(default_factory=dict)
    policies: dict[str, dict[str, Any]] = field(default_factory=dict)
    epochs: dict[str, int] = field(default_factory=dict)
    checkpoints: dict[str, dict[str, Any]] = field(default_factory=dict)
    semantic_assertions: dict[str, dict[str, Any]] = field(default_factory=dict)
    applied_commands: dict[str, int] = field(default_factory=dict)
    membership_voters: tuple[str, ...] = ()
    configuration_epoch: int = 0

    def apply(self, command: StateMachineCommand, log_index: int) -> str:
        """Apply one committed command idempotently; returns the dedup status."""
        if not isinstance(command, StateMachineCommand):
            raise TypeError("apply expects a StateMachineCommand")
        if command.command_id in self.applied_commands:
            return CommandStatus.DUPLICATE
        self.applied_commands[command.command_id] = log_index
        handler = getattr(self, f"_apply_{command.command_type}")
        handler(command.payload)
        return CommandStatus.APPLIED

    def _target_task(self, payload: Mapping[str, Any]) -> str:
        """Return the validated task identifier from a command payload."""
        return _non_empty("task_id", str(payload.get("task_id", "")))

    def _apply_assign_task(self, payload: Mapping[str, Any]) -> None:
        """Assign or replace task ownership with lease and fencing metadata."""
        task_id = self._target_task(payload)
        owner = _non_empty("owner", str(payload.get("owner", "")))
        previous = self.tasks.get(task_id, {})
        attempt = int(payload.get("attempt", previous.get("attempt", 0)))
        fencing_epoch = int(payload.get("fencing_epoch", previous.get("fencing_epoch", 0))) + 1
        self.tasks[task_id] = {
            "task_id": task_id,
            "owner": owner,
            "lease_expiry": float(payload.get("lease_expiry", 0.0)),
            "attempt": attempt,
            "fencing_epoch": fencing_epoch,
            "status": "assigned",
        }

    def _apply_release_task(self, payload: Mapping[str, Any]) -> None:
        """Release task ownership back to the unassigned pool."""
        task_id = self._target_task(payload)
        record = self.tasks.get(task_id)
        if record is not None:
            record["owner"] = ""
            record["status"] = "released"

    def _apply_complete_task(self, payload: Mapping[str, Any]) -> None:
        """Mark a task completed with output digests, keyed by idempotency."""
        task_id = self._target_task(payload)
        record = self.tasks.get(task_id)
        if record is None:
            record = self.tasks.setdefault(task_id, {"task_id": task_id})
        record["status"] = "completed"
        record["output_digest"] = str(payload.get("output_digest", ""))

    def _apply_transition_workflow(self, payload: Mapping[str, Any]) -> None:
        """Move one workflow through a validated state transition."""
        workflow_id = _non_empty("workflow_id", str(payload.get("workflow_id", "")))
        next_state = _non_empty("state", str(payload.get("state", "")))
        record = self.workflows.setdefault(workflow_id, {"workflow_id": workflow_id, "state": "new"})
        record["previous_state"] = record["state"]
        record["state"] = next_state

    def _apply_set_policy(self, payload: Mapping[str, Any]) -> None:
        """Commit one policy version record."""
        key = _non_empty("policy_key", str(payload.get("policy_key", "")))
        self.policies[key] = {
            "version": int(payload.get("version", 1)),
            "value": dict(payload.get("value", {})),
        }

    def _apply_advance_epoch(self, payload: Mapping[str, Any]) -> None:
        """Advance a monotonically increasing fencing epoch for one resource."""
        key = _non_empty("epoch_key", str(payload.get("epoch_key", "")))
        self.epochs[key] = max(self.epochs.get(key, 0), int(payload.get("epoch", 1))) + 1

    def _apply_record_checkpoint(self, payload: Mapping[str, Any]) -> None:
        """Store one durable recovery checkpoint reference."""
        checkpoint_id = _non_empty("checkpoint_id", str(payload.get("checkpoint_id", "")))
        self.checkpoints[checkpoint_id] = {
            "workflow_state": dict(payload.get("workflow_state", {})),
            "committed_task_ids": tuple(payload.get("committed_task_ids", ())),
            "input_digests": dict(payload.get("input_digests", {})),
            "output_digests": dict(payload.get("output_digests", {})),
            "coordinator_epoch": int(payload.get("coordinator_epoch", 0)),
            "recovery_cursor": int(payload.get("recovery_cursor", 0)),
        }

    def _apply_config_change(self, payload: Mapping[str, Any]) -> None:
        """Apply a committed membership configuration change with its epoch."""
        voters = tuple(sorted(str(v) for v in payload.get("voters", ())))
        if not voters:
            raise ValueError("configuration change requires a non-empty voter set")
        self.membership_voters = voters
        self.configuration_epoch += 1

    def _apply_no_op(self, payload: Mapping[str, Any]) -> None:
        """Leader-establishment no-op that commits prior-term entries indirectly."""
        return None

    def _apply_approve_semantic_assertion(self, payload: Mapping[str, Any]) -> None:
        """Record a consensus decision ordering a semantic approval (Step 30).

        Consensus may order an approval decision; it does not establish
        epistemic truth, so the assertion keeps its USG validation status.
        """
        assertion_id = _non_empty("assertion_id", str(payload.get("assertion_id", "")))
        self.semantic_assertions[assertion_id] = {
            "decision": str(payload.get("decision", "approved")),
            "usg_validation": str(payload.get("usg_validation", "pending")),
            "ordered_by": "consensus",
        }

    def snapshot(self) -> dict[str, Any]:
        """Return a comparable state snapshot for replay determinism checks."""
        return {
            "tasks": {k: dict(v) for k, v in sorted(self.tasks.items())},
            "workflows": {k: dict(v) for k, v in sorted(self.workflows.items())},
            "policies": {k: dict(v) for k, v in sorted(self.policies.items())},
            "epochs": dict(sorted(self.epochs.items())),
            "checkpoints": {k: dict(v) for k, v in sorted(self.checkpoints.items())},
            "semantic_assertions": {k: dict(v) for k, v in sorted(self.semantic_assertions.items())},
            "applied_commands": dict(sorted(self.applied_commands.items())),
            "membership_voters": list(self.membership_voters),
            "configuration_epoch": self.configuration_epoch,
        }

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        """Restore state from a snapshot (deterministic replay support)."""
        self.tasks = {k: dict(v) for k, v in snapshot.get("tasks", {}).items()}
        self.workflows = {k: dict(v) for k, v in snapshot.get("workflows", {}).items()}
        self.policies = {k: dict(v) for k, v in snapshot.get("policies", {}).items()}
        self.epochs = dict(snapshot.get("epochs", {}))
        self.checkpoints = {k: dict(v) for k, v in snapshot.get("checkpoints", {}).items()}
        self.semantic_assertions = {
            k: dict(v) for k, v in snapshot.get("semantic_assertions", {}).items()
        }
        self.applied_commands = dict(snapshot.get("applied_commands", {}))
        self.membership_voters = tuple(snapshot.get("membership_voters", ()))
        self.configuration_epoch = int(snapshot.get("configuration_epoch", 0))


def replay_deterministic(
    commands: tuple[StateMachineCommand, ...],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replay one command sequence twice and return both snapshots (Step 18 check)."""
    def build() -> dict[str, Any]:
        machine = AuthoritativeStateMachine()
        for position, command in enumerate(commands, start=1):
            machine.apply(command, position)
        return machine.snapshot()

    return build(), build()


__all__ = [
    "COMMAND_TYPES",
    "StateMachineCommand",
    "CommandStatus",
    "AuthoritativeStateMachine",
    "replay_deterministic",
]
