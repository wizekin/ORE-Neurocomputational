"""Raft consensus adapter: election, replication, commit, and recovery rules.

Implements Steps 14-19 of the Point 17 algorithm on top of the deterministic
state machine and the replicated log:

* Step 14 - leader election with randomized timeouts to avoid repeated split votes.
* Step 15 - term and vote safety: term and vote persist before responding, at most
  one vote per term, and candidate log freshness is checked before granting.
* Step 16 - leader replication with match/next indices; followers truncate only
  conflicting *uncommitted* suffixes.
* Step 17 - commit advancement only under replication and current-term rules; client
  success is acknowledged only after the durable commit point.
* Step 18 - deterministic application in log order with command-id deduplication.
* Step 19 - leader failure recovery; an old leader steps down when it observes a
  higher term and cannot continue authoritative writes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import random
from typing import Any, Callable, Mapping

from ore_distributed.consensus.log import LogEntry, ReplicatedLog, quorum_size
from ore_distributed.consensus.state_machine import (
    AuthoritativeStateMachine,
    CommandStatus,
    StateMachineCommand,
)


class RaftRole(str, Enum):
    """The three Raft roles of one node."""

    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"


#: Leader-establishment no-op used to commit entries from prior terms.
NO_OP_COMMAND = StateMachineCommand(command_id="leader-no-op", command_type="no_op")


class RaftStorage:
    """Durable Raft state: current term, vote, log, and commit index.

    Persistence happens synchronously before protocol responses, which is the
    property Step 15 requires; a crash simulation keeps only this state.
    """

    def __init__(self) -> None:
        """Create empty durable state for one node."""
        self.current_term = 0
        self.voted_for: str | None = None
        self.log = ReplicatedLog()
        self.commit_index = 0
        self.persistence_events: list[tuple[str, Any]] = []

    def persist_term_and_vote(self, term: int, voted_for: str | None) -> None:
        """Persist term and vote before any vote-dependent response (Step 15)."""
        self.current_term = term
        self.voted_for = voted_for
        self.persistence_events.append(("vote", (term, voted_for)))

    def persist_term(self, term: int) -> None:
        """Persist an observed higher term before further processing."""
        self.current_term = term
        self.persistence_events.append(("term", term))

    def persist_commit(self, commit_index: int) -> None:
        """Persist the commit index durable commit point (Step 17)."""
        self.commit_index = commit_index
        self.persistence_events.append(("commit", commit_index))

    def snapshot(self) -> dict[str, Any]:
        """Return the durable state needed to restart a node."""
        return {
            "current_term": self.current_term,
            "voted_for": self.voted_for,
            "log": list(self.log.snapshot()),
            "commit_index": self.commit_index,
        }


@dataclass(slots=True)
class ProposalResult:
    """Outcome of submitting a client command to a node."""

    status: str  # committed | pending | not_leader | duplicate
    command_id: str
    index: int = 0
    term: int = 0

    @property
    def acknowledged(self) -> bool:
        """True only when the command reached the durable commit point."""
        return self.status == "committed"


class RaftNode:
    """One Raft voting member operating over synchronous in-process delivery."""

    def __init__(
        self,
        node_id: str,
        storage: RaftStorage | None = None,
        state_machine: AuthoritativeStateMachine | None = None,
        voters: tuple[str, ...] = (),
        *,
        election_timeout_min: int = 8,
        election_timeout_max: int = 14,
        heartbeat_interval: int = 3,
        seed: int = 5,
        rpc_send: Callable[[str, str, Mapping[str, Any]], Any] | None = None,
    ) -> None:
        """Create a node; voters default to the single-node configuration."""
        if not node_id.strip():
            raise ValueError("node_id must be non-empty")
        self.node_id = node_id.strip()
        self.storage = storage or RaftStorage()
        self.state_machine = state_machine or AuthoritativeStateMachine()
        self.voters = tuple(sorted(set(voters))) or (self.node_id,)
        self.role = RaftRole.FOLLOWER
        self.leader_id: str | None = None
        self.election_elapsed = 0
        self.heartbeat_elapsed = 0
        self.heartbeat_interval = int(heartbeat_interval)
        self._rng = random.Random(seed ^ hash(self.node_id) & 0xFFFF)
        self.election_timeout_ticks = self._rng.randint(
            int(election_timeout_min), int(election_timeout_max)
        )
        self.election_timeout_min = int(election_timeout_min)
        self.election_timeout_max = int(election_timeout_max)
        self.votes_received: set[str] = set()
        self.next_index: dict[str, int] = {}
        self.match_index: dict[str, int] = {}
        self.last_applied = 0
        self.commit_events: list[tuple[str, int, int]] = []
        self.election_events: list[tuple[int, str]] = []
        self.rpc_send = rpc_send or self._rpc_send_default
        self.stale_messages_rejected = 0
        self.append_rejections = 0

    # ------------------------------------------------------------------ helpers

    def _rpc_send_default(self, recipient: str, rpc_type: str, args: Mapping[str, Any]) -> Any:
        """Default no-network sender that drops every message (harness overrides)."""
        return None

    @property
    def current_term(self) -> int:
        """Return the durable current term."""
        return self.storage.current_term

    @property
    def commit_index(self) -> int:
        """Return the durable commit index."""
        return self.storage.commit_index

    @property
    def log(self) -> ReplicatedLog:
        """Return the replicated log."""
        return self.storage.log

    def is_leader(self) -> bool:
        """Return whether this node is the current leader."""
        return self.role is RaftRole.LEADER

    def quorum(self) -> int:
        """Return the majority quorum size for the current voter configuration."""
        return quorum_size(len(self.voters))

    def _become_follower(self, term: int, leader: str | None = None) -> None:
        """Persist an observed term and revert to follower role."""
        if term > self.storage.current_term:
            self.storage.persist_term(term)
        elif term == self.storage.current_term and self.storage.voted_for is None:
            self.storage.persist_term(term, None)
        self.role = RaftRole.FOLLOWER
        self.leader_id = leader
        self.votes_received.clear()
        self.election_elapsed = 0

    # ------------------------------------------------------------------ election

    def tick(self) -> None:
        """Advance one protocol tick of election and heartbeat timers."""
        if self.role is RaftRole.LEADER:
            self.heartbeat_elapsed += 1
            if self.heartbeat_elapsed >= self.heartbeat_interval:
                self.heartbeat_elapsed = 0
                self._broadcast_append_entries()
            return
        self.election_elapsed += 1
        if self.election_elapsed >= self.election_timeout_ticks:
            self.start_election()

    def start_election(self) -> None:
        """Begin an election: increment term, vote for self, request votes (Step 14)."""
        new_term = self.storage.current_term + 1
        # Step 15: persist the new term and self-vote before requesting votes.
        self.storage.persist_term_and_vote(new_term, self.node_id)
        self.role = RaftRole.CANDIDATE
        self.leader_id = None
        self.votes_received = {self.node_id}
        self.election_elapsed = 0
        self.election_timeout_ticks = self._rng.randint(
            self.election_timeout_min, self.election_timeout_max
        )
        self.election_events.append((new_term, "candidate"))
        for voter in self.voters:
            if voter == self.node_id:
                continue
            self.rpc_send(
                voter,
                "request_vote",
                {
                    "term": new_term,
                    "candidate": self.node_id,
                    "last_log_index": self.storage.log.last_index,
                    "last_log_term": self.storage.log.last_term,
                },
            )

    def _candidate_log_fresh(self, last_log_term: int, last_log_index: int) -> bool:
        """Check candidate log freshness before granting a vote (Step 15)."""
        own_last_term = self.storage.log.last_term
        own_last_index = self.storage.log.last_index
        if last_log_term != own_last_term:
            return last_log_term > own_last_term
        return last_log_index >= own_last_index

    def handle_request_vote(self, args: Mapping[str, Any]) -> dict[str, Any]:
        """Grant or reject one vote following term and vote safety (Step 15)."""
        term = int(args["term"])
        candidate = str(args["candidate"])
        if term < self.storage.current_term:
            self.stale_messages_rejected += 1
            return {"term": self.storage.current_term, "vote_granted": False, "voter": self.node_id}
        if term > self.storage.current_term:
            # Persist the observed term before acting on it (Step 15).
            self.storage.persist_term_and_vote(term, None)
            self.role = RaftRole.FOLLOWER
            self.leader_id = None
            self.votes_received.clear()
        granted = (
            self.storage.voted_for in (None, candidate)
            and self._candidate_log_fresh(int(args["last_log_term"]), int(args["last_log_index"]))
        )
        if granted:
            # Persist the granted vote before responding (Step 15).
            self.storage.persist_term_and_vote(term, candidate)
            self.election_elapsed = 0
        return {"term": self.storage.current_term, "vote_granted": granted, "voter": self.node_id}

    def handle_request_vote_reply(self, reply: Mapping[str, Any]) -> None:
        """Count one vote reply; a majority elects the leader (Step 14)."""
        term = int(reply["term"])
        if term > self.storage.current_term:
            self._become_follower(term)
            return
        if term < self.storage.current_term or self.role is not RaftRole.CANDIDATE:
            return
        if reply.get("vote_granted") and str(reply.get("voter", "")) in self.voters:
            self.votes_received.add(str(reply.get("voter", "")))
        if len(self.votes_received & set(self.voters)) >= self.quorum():
            self._become_leader()

    def _become_leader(self) -> None:
        """Win an election, initialize replication indices, and establish authority."""
        if self.role is not RaftRole.CANDIDATE:
            return
        self.role = RaftRole.LEADER
        self.leader_id = self.node_id
        self.heartbeat_elapsed = 0
        last = self.storage.log.last_index
        for voter in self.voters:
            self.next_index[voter] = last + 1
            self.match_index[voter] = 0
        self.election_events.append((self.storage.current_term, "leader"))
        # Leader-establishment no-op commits entries from earlier terms indirectly.
        self._append_local(NO_OP_COMMAND)
        self._broadcast_append_entries()

    # ------------------------------------------------------------------ replication

    def _append_local(self, command: StateMachineCommand) -> LogEntry:
        """Append a command locally as leader and return the entry."""
        entry = self.storage.log.append(self.storage.current_term, command)
        self.match_index[self.node_id] = entry.index
        self.next_index[self.node_id] = entry.index + 1
        return entry

    def _broadcast_append_entries(self) -> None:
        """Send AppendEntries RPCs to every other voter in the configuration."""
        for voter in self.voters:
            if voter == self.node_id:
                continue
            self._send_append_entries_to(voter)

    def _send_append_entries_to(self, follower: str, limit: int = 64) -> None:
        """Send one AppendEntries RPC built from the follower's next index."""
        start = self.next_index.get(follower, self.storage.log.last_index + 1)
        prev_index = start - 1
        prev_term = self.storage.log.term_at(prev_index) if prev_index >= 1 else 0
        entries = list(self.storage.log.entries_from(start, limit))
        self.rpc_send(
            follower,
            "append_entries",
            {
                "term": self.storage.current_term,
                "leader": self.node_id,
                "prev_log_index": prev_index,
                "prev_log_term": prev_term,
                "entries": entries,
                "leader_commit": self.storage.commit_index,
            },
        )

    def handle_append_entries(self, args: Mapping[str, Any]) -> dict[str, Any]:
        """Accept or reject leader replication following the log-matching rules (Step 16)."""
        term = int(args["term"])
        leader = str(args["leader"])
        if term < self.storage.current_term:
            # Step 19: stale-term messages are rejected outright.
            self.stale_messages_rejected += 1
            return {
                "term": self.storage.current_term,
                "success": False,
                "match_index": 0,
                "follower": self.node_id,
                "conflict_index": 0,
            }
        if term > self.storage.current_term or self.role is RaftRole.CANDIDATE:
            self._become_follower(term, leader)
        elif self.role is RaftRole.LEADER and term == self.storage.current_term and leader != self.node_id:
            self._become_follower(term, leader)
        else:
            self.leader_id = leader
        self.election_elapsed = 0

        prev_index = int(args["prev_log_index"])
        prev_term = int(args["prev_log_term"])
        entries: list[LogEntry] = list(args.get("entries", ()))

        if prev_index > self.storage.log.last_index:
            self.append_rejections += 1
            return {
                "term": self.storage.current_term,
                "success": False,
                "match_index": 0,
                "follower": self.node_id,
                "conflict_index": self.storage.log.last_index + 1,
                "conflict_term": 0,
            }
        if prev_index >= 1 and self.storage.log.term_at(prev_index) != prev_term:
            self.append_rejections += 1
            conflict_term = self.storage.log.term_at(prev_index) or 0
            conflict_index = self.storage.log.last_index
            while conflict_index > 1 and self.storage.log.term_at(conflict_index) == conflict_term:
                conflict_index -= 1
            # Followers may only truncate uncommitted conflicting suffixes.
            if conflict_index >= self.storage.commit_index:
                self.storage.log.truncate_suffix(conflict_index)
            return {
                "term": self.storage.current_term,
                "success": False,
                "match_index": 0,
                "follower": self.node_id,
                "conflict_index": conflict_index,
                "conflict_term": conflict_term,
            }

        # Append new entries, truncating any conflicting uncommitted suffix.
        expected_index = prev_index + 1
        for entry in entries:
            existing = self.storage.log.entry_at(expected_index)
            if existing is not None and existing.term != entry.term:
                if expected_index <= self.storage.commit_index:
                    raise RuntimeError(
                        "committed log prefix conflicts with the leader; invariant violation"
                    )
                # Uncommitted conflicting suffix is discarded as Raft permits.
                self.storage.log.truncate_suffix(expected_index)
                existing = None
            if existing is None:
                self.storage.log.append_entry(entry)
            expected_index += 1
        match_index = expected_index - 1

        leader_commit = int(args.get("leader_commit", 0))
        if leader_commit > self.storage.commit_index:
            self.storage.persist_commit(min(leader_commit, match_index))
            self._apply_committed()
        return {
            "term": self.storage.current_term,
            "success": True,
            "match_index": match_index,
            "follower": self.node_id,
        }

    def handle_append_entries_reply(self, reply: Mapping[str, Any]) -> None:
        """Update replication progress and advance the commit index (Step 17)."""
        term = int(reply["term"])
        if term > self.storage.current_term:
            # A recovering old leader observes a higher term and must step down.
            self._become_follower(term)
            return
        if term < self.storage.current_term or self.role is not RaftRole.LEADER:
            self.stale_messages_rejected += 1
            return
        follower = str(reply.get("follower", ""))
        if follower not in self.voters:
            return
        if reply.get("success"):
            match = int(reply.get("match_index", 0))
            self.match_index[follower] = max(self.match_index.get(follower, 0), match)
            self.next_index[follower] = self.match_index[follower] + 1
            self._advance_commit_index()
        else:
            conflict_index = int(reply.get("conflict_index", 0)) or 1
            self.next_index[follower] = max(1, min(conflict_index, self.storage.log.last_index + 1))
            self._send_append_entries_to(follower)

    def _advance_commit_index(self) -> None:
        """Advance commit only under replication and current-term rules (Step 17)."""
        self.match_index[self.node_id] = self.storage.log.last_index
        for candidate_index in range(self.storage.log.last_index, self.storage.commit_index, -1):
            entry = self.storage.log.entry_at(candidate_index)
            if entry is None:
                continue
            if entry.term != self.storage.current_term:
                # Only current-term entries count toward a direct commit.
                break
            replicated = sum(
                1 for voter in self.voters if self.match_index.get(voter, 0) >= candidate_index
            )
            if replicated >= self.quorum():
                self.storage.persist_commit(candidate_index)
                self._apply_committed()
                return

    def _apply_committed(self) -> None:
        """Apply committed commands in log order with deduplication (Step 18)."""
        while self.last_applied < self.storage.commit_index:
            self.last_applied += 1
            entry = self.storage.log.entry_at(self.last_applied)
            if entry is None:
                continue
            status = self.state_machine.apply(entry.command, entry.index)
            self.commit_events.append((entry.command.command_id, entry.index, self.storage.current_term))
            if status == CommandStatus.DUPLICATE:
                continue

    # ------------------------------------------------------------------ client API

    def propose(self, command: StateMachineCommand) -> ProposalResult:
        """Submit a client command; acknowledged only after durable commit (Step 17)."""
        if self.role is not RaftRole.LEADER:
            return ProposalResult(status="not_leader", command_id=command.command_id)
        if command.command_id in self.state_machine.applied_commands:
            return ProposalResult(status="duplicate", command_id=command.command_id)
        entry = self._append_local(command)
        self._broadcast_append_entries()
        self._advance_commit_index()
        if entry.index <= self.storage.commit_index:
            return ProposalResult(
                status="committed", command_id=command.command_id,
                index=entry.index, term=entry.term,
            )
        return ProposalResult(
            status="pending", command_id=command.command_id, index=entry.index, term=entry.term
        )

    def commit_rule_holds_for(self, index: int) -> bool:
        """Return whether an entry at ``index`` may be committed right now.

        Committing requires replication to a majority of the current voter
        configuration *and* an entry from the current term; entries from
        earlier terms commit only indirectly (Step 17).
        """
        entry = self.storage.log.entry_at(index)
        if entry is None or entry.term != self.storage.current_term:
            return False
        self.match_index[self.node_id] = self.storage.log.last_index
        replicated = sum(
            1 for voter in self.voters if self.match_index.get(voter, 0) >= index
        )
        return replicated >= self.quorum()

    def is_committed(self, command_id: str) -> bool:
        """Return whether a command reached the durable commit point."""
        for index in range(1, self.storage.commit_index + 1):
            entry = self.storage.log.entry_at(index)
            if entry is not None and entry.command.command_id == command_id:
                return True
        return command_id in self.state_machine.applied_commands

    def crash(self) -> RaftStorage:
        """Simulate process loss; only durable storage survives."""
        return self.storage

    def restart_from(self, storage: RaftStorage) -> None:
        """Restart as a follower from durable state only (Step 19 recovery)."""
        self.storage = storage
        self.role = RaftRole.FOLLOWER
        self.leader_id = None
        self.votes_received.clear()
        self.election_elapsed = 0
        self.heartbeat_elapsed = 0
        self.last_applied = 0
        self.commit_events = []
        self._apply_committed()

    # ------------------------------------------------------------------ configuration

    def reconfigure_voters(self, voters: tuple[str, ...]) -> int:
        """Adopt a committed voter configuration; returns the config epoch."""
        voters = tuple(sorted(set(str(v) for v in voters)))
        if not voters:
            raise ValueError("voter configuration must be non-empty")
        self.voters = voters
        if self.role is RaftRole.LEADER:
            last = self.storage.log.last_index
            for voter in voters:
                self.next_index.setdefault(voter, last + 1)
                self.match_index.setdefault(voter, 0)
        return self.state_machine.configuration_epoch


class RaftCluster:
    """Synchronous multi-node harness driving Raft message delivery with partitions.

    Delivery is direct in-process function invocation so protocol behavior stays
    deterministic; partitions drop messages between node groups instead.
    """

    def __init__(self, node_ids: tuple[str, ...], seed: int = 23) -> None:
        """Create a cluster with one RaftNode per identifier."""
        if not node_ids:
            raise ValueError("cluster requires at least one node")
        self.node_ids = tuple(sorted(set(node_ids)))
        self._rng = random.Random(seed)
        self.nodes: dict[str, RaftNode] = {}
        self._partitions: list[tuple[frozenset[str], frozenset[str]]] = []
        self.dropped_messages = 0
        for position, node_id in enumerate(self.node_ids):
            node = RaftNode(
                node_id,
                voters=self.node_ids,
                seed=seed + position,
                rpc_send=self._deliver,
            )
            self.nodes[node_id] = node

    def node(self, node_id: str) -> RaftNode:
        """Return one node or raise."""
        node = self.nodes.get(node_id)
        if node is None:
            raise KeyError(f"unknown node {node_id!r}")
        return node

    def leader(self) -> RaftNode | None:
        """Return the legitimate current leader, or None when elections are unsettled.

        During a partition a stale minority leader may still claim the leader
        role at an old term; the legitimate leader is the claimant with the
        strictly highest term, since a lower-term claimant steps down on the
        first protocol exchange that reaches it.
        """
        claimants = [node for node in self.nodes.values() if node.is_leader()]
        if not claimants:
            return None
        highest = max(node.storage.current_term for node in claimants)
        top = [node for node in claimants if node.storage.current_term == highest]
        if len(top) == 1:
            return top[0]
        return None

    def _cut(self, sender: str, recipient: str) -> bool:
        """Return whether delivery between two nodes is partitioned."""
        for group_a, group_b in self._partitions:
            if (sender in group_a and recipient in group_b) or (sender in group_b and recipient in group_a):
                return True
        return False

    def _deliver(self, recipient: str, rpc_type: str, args: Mapping[str, Any]) -> Any:
        """Deliver one RPC to a recipient, dropping partitioned messages."""
        sender = str(args.get("candidate") or args.get("leader") or "")
        if sender and self._cut(sender, recipient):
            self.dropped_messages += 1
            return None
        target = self.nodes.get(recipient)
        if target is None:
            self.dropped_messages += 1
            return None
        if rpc_type == "request_vote":
            reply = target.handle_request_vote(args)
            origin = self.nodes.get(sender)
            if origin is not None and reply["term"] >= origin.current_term:
                origin.handle_request_vote_reply(reply)
            return reply
        if rpc_type == "append_entries":
            reply = target.handle_append_entries(args)
            origin = self.nodes.get(sender)
            if origin is not None and reply["term"] >= origin.current_term:
                origin.handle_append_entries_reply(reply)
            return reply
        raise ValueError(f"unsupported RPC type {rpc_type!r}")

    def tick(self) -> None:
        """Advance every node by one protocol tick."""
        for node in self.nodes.values():
            node.tick()

    def run_ticks(self, ticks: int) -> None:
        """Run several ticks to drive elections, heartbeats, and replication."""
        for _ in range(int(ticks)):
            self.tick()

    def run_until_leader(self, max_ticks: int = 60) -> RaftNode:
        """Run ticks until exactly one leader exists or raise."""
        for _ in range(int(max_ticks)):
            leader = self.leader()
            if leader is not None:
                return leader
            self.tick()
        raise RuntimeError("no unique leader elected within the tick budget")

    def run_until_committed(self, command_id: str, max_ticks: int = 60) -> bool:
        """Run ticks until a command is committed somewhere; returns success."""
        for _ in range(int(max_ticks)):
            if any(node.is_committed(command_id) for node in self.nodes.values()):
                return True
            self.tick()
        return any(node.is_committed(command_id) for node in self.nodes.values())

    def partition(self, group_a: set[str], group_b: set[str]) -> None:
        """Cut traffic between two node groups (Step 44)."""
        self._partitions.append((frozenset(group_a), frozenset(group_b)))

    def isolate(self, node_id: str) -> None:
        """Cut one node from every other node."""
        others = set(self.node_ids) - {node_id}
        self.partition({node_id}, others)

    def heal(self) -> None:
        """Remove every injected partition."""
        self._partitions.clear()

    def crash_and_restart(self, node_id: str) -> None:
        """Simulate one node's process loss followed by restart from storage."""
        node = self.node(node_id)
        storage = node.crash()
        fresh = RaftNode(
            node_id,
            storage=storage,
            state_machine=AuthoritativeStateMachine(),
            voters=node.voters,
            rpc_send=self._deliver,
        )
        # A restart replays its durable log deterministically.
        fresh.restart_from(storage)
        self.nodes[node_id] = fresh

    def state_machines(self) -> dict[str, dict[str, Any]]:
        """Return state-machine snapshots of every node for convergence checks."""
        return {node_id: self.nodes[node_id].state_machine.snapshot() for node_id in self.node_ids}

    def committed_indices(self) -> dict[str, int]:
        """Return the durable commit index of every node."""
        return {node_id: self.nodes[node_id].storage.commit_index for node_id in self.node_ids}


def majority_partition_writes(
    cluster: RaftCluster, majority_group: set[str]
) -> tuple[bool, bool]:
    """Probe whether authoritative writes survive only in the majority group.

    Returns (majority_writes_succeed, minority_writes_blocked) by attempting a
    proposal from the majority side and from the minority side.
    """
    minority = set(cluster.node_ids) - set(majority_group)
    majority_ok = False
    minority_ok = False
    leader = cluster.leader()
    if leader is not None and leader.node_id in majority_group:
        result = leader.propose(
            StateMachineCommand(
                command_id="probe-majority", command_type="set_policy",
                payload={"policy_key": "probe", "version": 1, "value": {}},
            )
        )
        majority_ok = result.status in {"committed", "pending"}
    if leader is not None and leader.node_id in minority:
        result = leader.propose(
            StateMachineCommand(
                command_id="probe-minority", command_type="set_policy",
                payload={"policy_key": "probe", "version": 1, "value": {}},
            )
        )
        minority_ok = result.acknowledged
    return majority_ok, minority_ok


__all__ = [
    "RaftRole",
    "RaftStorage",
    "RaftNode",
    "RaftCluster",
    "ProposalResult",
    "NO_OP_COMMAND",
    "majority_partition_writes",
]
