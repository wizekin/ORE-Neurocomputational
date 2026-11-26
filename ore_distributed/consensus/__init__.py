"""Raft consensus: replicated log, deterministic state machine, election, and commits."""
from ore_distributed.consensus.log import LogEntry, ReplicatedLog, majority_quorum, quorum_size
from ore_distributed.consensus.state_machine import (
    AuthoritativeStateMachine,
    CommandStatus,
    StateMachineCommand,
)
from ore_distributed.consensus.raft import (
    RaftCluster,
    RaftNode,
    RaftRole,
    RaftStorage,
)
from ore_distributed.consensus.reconfiguration import (
    ConfigurationChangeError,
    RaftReconfiguration,
)
from ore_distributed.consensus.leases import (
    FencingRejected,
    LeaseGrant,
    LeaseManager,
)

__all__ = [
    "LogEntry",
    "ReplicatedLog",
    "majority_quorum",
    "quorum_size",
    "AuthoritativeStateMachine",
    "CommandStatus",
    "StateMachineCommand",
    "RaftCluster",
    "RaftNode",
    "RaftRole",
    "RaftStorage",
    "ConfigurationChangeError",
    "RaftReconfiguration",
    "FencingRejected",
    "LeaseGrant",
    "LeaseManager",
]
