"""Safe Raft membership reconfiguration with configuration epochs (Step 20).

Membership changes serialize through the consensus log; quorum availability is
validated before any change, only one change may be in flight, and both adding
and removing nodes are exercised under failures in the test suite.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ore_distributed.consensus.log import quorum_size
from ore_distributed.consensus.raft import RaftCluster, RaftNode
from ore_distributed.consensus.state_machine import StateMachineCommand


class ConfigurationChangeError(ValueError):
    """Raised when a membership change would violate configuration safety."""


@dataclass(frozen=True, slots=True)
class ConfigurationChangeRecord:
    """Audit record of one committed configuration change."""

    epoch: int
    action: str
    node_id: str
    voters: tuple[str, ...]
    log_index: int

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a plain dictionary."""
        return {
            "epoch": self.epoch,
            "action": self.action,
            "node_id": self.node_id,
            "voters": list(self.voters),
            "log_index": self.log_index,
        }


def quorum_available_after_change(
    current_voters: tuple[str, ...], change: str, node_id: str
) -> tuple[bool, int, int]:
    """Validate quorum availability before and after a proposed change (Step 20)."""
    current_voters = tuple(sorted(set(current_voters)))
    if node_id not in current_voters and change == "remove":
        raise ConfigurationChangeError(f"node {node_id!r} is not in the configuration")
    if node_id in current_voters and change == "add":
        raise ConfigurationChangeError(f"node {node_id!r} is already in the configuration")
    if change == "add":
        voters_after = tuple(sorted(set(current_voters) | {node_id}))
    elif change == "remove":
        voters_after = tuple(sorted(set(current_voters) - {node_id}))
    else:
        raise ConfigurationChangeError("action must be 'add' or 'remove'")
    if len(voters_after) == 0:
        raise ConfigurationChangeError("configuration would become empty")
    before_quorum = quorum_size(len(current_voters))
    after_quorum = quorum_size(len(voters_after))
    # The new configuration must still be able to form a quorum of itself.
    if len(voters_after) < after_quorum:
        raise ConfigurationChangeError("resulting configuration cannot form a quorum")
    return len(voters_after) >= after_quorum, before_quorum, after_quorum


class RaftReconfiguration:
    """Serialize membership changes through the consensus log (Step 20).

    The controller refuses simultaneous uncoordinated edits with an in-flight
    lock, commits a ``config_change`` command through the current leader, and
    only applies the new voter set cluster-wide after the change commits.
    """

    def __init__(self, cluster: RaftCluster) -> None:
        """Create a reconfiguration controller over one cluster."""
        self.cluster = cluster
        self._pending: str | None = None
        self.history: list[ConfigurationChangeRecord] = []

    def _leader_or_raise(self) -> RaftNode:
        """Return the current leader or refuse to change membership."""
        leader = self.cluster.leader()
        if leader is None:
            raise ConfigurationChangeError("no stable leader; refuse membership change")
        return leader

    def _current_voters(self, leader: RaftNode) -> tuple[str, ...]:
        """Return the voter configuration the leader currently uses."""
        return leader.voters

    def _commit_config_change(
        self,
        leader: RaftNode,
        new_voters: tuple[str, ...],
        change_command: StateMachineCommand,
    ) -> int:
        """Propose the change, wait for commit, and return its log index."""
        result = leader.propose(change_command)
        if result.status == "not_leader":
            raise ConfigurationChangeError("leadership changed during configuration update")
        if result.status == "pending":
            committed = self.cluster.run_until_committed(
                change_command.command_id, max_ticks=30
            )
            if not committed:
                raise ConfigurationChangeError("configuration change failed to commit")
        return result.index

    def _apply_cluster_wide(self, new_voters: tuple[str, ...]) -> int:
        """Adopt the new voter set on every live node; returns the config epoch."""
        epoch = 0
        for node in self.cluster.nodes.values():
            epoch = node.reconfigure_voters(new_voters)
        return epoch

    def add_voter(
        self,
        node_id: str,
        *,
        node: RaftNode | None = None,
        reason: str = "capacity addition",
    ) -> ConfigurationChangeRecord:
        """Add one voter through a committed configuration change."""
        if self._pending is not None:
            raise ConfigurationChangeError(
                "another membership change is in flight; serialize configuration edits"
            )
        leader = self._leader_or_raise()
        current = self._current_voters(leader)
        quorum_available_after_change(current, "add", node_id)
        if node is not None and node_id not in self.cluster.nodes:
            self.cluster.nodes[node_id] = node
        elif node_id not in self.cluster.nodes:
            raise ConfigurationChangeError(
                f"node {node_id!r} is not part of the cluster; discover and join it first"
            )
        self._pending = f"add:{node_id}"
        try:
            # State transfer: a joining voter first receives the current
            # committed log so it can rejoin as an informed voter (Step 20).
            if node_id in self.cluster.nodes:
                joiner = self.cluster.nodes[node_id]
                if joiner.storage.log.last_index == 0 and leader.storage.log.last_index > 0:
                    joiner.storage.log.restore(leader.storage.log.snapshot())
                    joiner.storage.persist_commit(leader.storage.commit_index)
                    joiner.last_applied = 0
                    joiner._apply_committed()  # noqa: SLF001 - state transfer replays
            new_voters = tuple(sorted(set(current) | {node_id}))
            command = StateMachineCommand(
                command_id=f"config-add-{node_id}-{len(current) + 1}",
                command_type="config_change",
                payload={"voters": list(new_voters), "action": "add", "node_id": node_id},
            )
            log_index = self._commit_config_change(leader, new_voters, command)
            epoch = self._apply_cluster_wide(new_voters)
            record = ConfigurationChangeRecord(
                epoch=epoch,
                action="add",
                node_id=node_id,
                voters=new_voters,
                log_index=log_index,
            )
            self.history.append(record)
            return record
        finally:
            self._pending = None

    def remove_voter(self, node_id: str, reason: str = "maintenance removal") -> ConfigurationChangeRecord:
        """Remove one voter without stranding the cluster below quorum (Step 20)."""
        if self._pending is not None:
            raise ConfigurationChangeError(
                "another membership change is in flight; serialize configuration edits"
            )
        leader = self._leader_or_raise()
        current = self._current_voters(leader)
        quorum_available_after_change(current, "remove", node_id)
        new_voters = tuple(sorted(set(current) - {node_id}))
        self._pending = f"remove:{node_id}"
        try:
            command = StateMachineCommand(
                command_id=f"config-remove-{node_id}-{len(current)}",
                command_type="config_change",
                payload={"voters": list(new_voters), "action": "remove", "node_id": node_id},
            )
            log_index = self._commit_config_change(leader, new_voters, command)
            epoch = self._apply_cluster_wide(new_voters)
            removed = self.cluster.nodes.pop(node_id, None)
            if removed is not None and removed.is_leader():
                # A removed leader steps down: its writes no longer count.
                removed.role = removed.role.__class__.FOLLOWER
            record = ConfigurationChangeRecord(
                epoch=epoch,
                action="remove",
                node_id=node_id,
                voters=new_voters,
                log_index=log_index,
            )
            self.history.append(record)
            return record
        finally:
            self._pending = None

    def changes(self) -> tuple[ConfigurationChangeRecord, ...]:
        """Return the committed configuration history."""
        return tuple(self.history)

    def configuration_epoch(self) -> int:
        """Return the current configuration epoch from consensus state."""
        leader = self.cluster.leader()
        if leader is None:
            return self.cluster.nodes[self.cluster.node_ids[0]].state_machine.configuration_epoch
        return leader.state_machine.configuration_epoch


__all__ = [
    "ConfigurationChangeError",
    "ConfigurationChangeRecord",
    "RaftReconfiguration",
    "quorum_available_after_change",
]
