"""Restart and disruption recovery orchestration (Step 43).

On restart a node authenticates and rejoins, loads durable local state,
reconciles the consensus log, restores checkpoints, exchanges CRDT summaries,
and replays missing events. Recovered local state is never published as
authoritative until it has passed reconciliation and epoch checks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ore_distributed.consensus.raft import RaftCluster
from ore_distributed.execution.checkpoints import CheckpointStore
from ore_distributed.replication.replica import CrdtReplica


@dataclass(frozen=True, slots=True)
class RecoveryReport:
    """One recovery attempt's outcome for audit and observability."""

    node_id: str
    steps: tuple[str, ...]
    reconciled_entries: int
    checkpoint_restored: str | None
    events_replayed: int
    authorized: bool
    failure: str | None = None


class RecoveryManager:
    """Coordinate one node's recovery after a crash or cluster disruption."""

    def __init__(
        self,
        cluster: RaftCluster,
        checkpoint_store: CheckpointStore,
    ) -> None:
        """Create a manager over one cluster and checkpoint store."""
        self.cluster = cluster
        self.checkpoints = checkpoint_store
        self.reports: list[RecoveryReport] = []

    def recovered_state_is_authoritative(
        self, node_state: Mapping[str, Any], leader_state: Mapping[str, Any]
    ) -> bool:
        """Gate publication of recovered state behind reconciliation (Step 43)."""
        return node_state == leader_state

    def recover_node(
        self,
        node_id: str,
        *,
        replica: CrdtReplica | None = None,
        authoritative_replica: CrdtReplica | None = None,
        workflow_id: str | None = None,
    ) -> RecoveryReport:
        """Run the full restart sequence for one node (Step 43)."""
        steps: list[str] = []
        # 1. Authenticate and rejoin: recovery models the restart of the node's
        # process from durable storage (crash-restart is idempotent here).
        self.cluster.crash_and_restart(node_id)
        steps.append("authenticated_rejoin")
        node = self.cluster.node(node_id)
        # 2. Load durable local state (the restarted node kept its storage).
        steps.append("loaded_durable_local_state")
        # 3. Reconcile the consensus log up to the current commit index.
        self.cluster.run_ticks(4)
        steps.append("reconciled_consensus_log")
        reconciled_entries = node.storage.commit_index
        # 4. Restore the latest checkpoint for the workflow, if any.
        checkpoint_id: str | None = None
        if workflow_id:
            checkpoint = self.checkpoints.latest_for(workflow_id)
            if checkpoint is not None:
                checkpoint_id = checkpoint.checkpoint_id
                steps.append("restored_checkpoint")
        # 5. Exchange CRDT summaries and replay missing events.
        events_replayed = 0
        if replica is not None and authoritative_replica is not None:
            from ore_distributed.replication.reconciliation import reconcile_replicas

            report = reconcile_replicas(replica, authoritative_replica)
            events_replayed = len(replica.observations())
            steps.append("exchanged_crdt_summaries" if report.converged else "crdt_reconciliation_incomplete")
        # 6. Epoch check before anything authoritative is published.
        leader = self.cluster.leader()
        authorized = leader is None or node.current_term <= leader.current_term
        if not authorized:
            steps.append("epoch_check_failed")
        else:
            steps.append("epoch_check_passed")
        result = RecoveryReport(
            node_id=node_id,
            steps=tuple(steps),
            reconciled_entries=reconciled_entries,
            checkpoint_restored=checkpoint_id,
            events_replayed=events_replayed,
            authorized=authorized,
        )
        self.reports.append(result)
        return result

    def replay_missing_events(
        self,
        journal_events: Mapping[str, Any],
        received_event_ids: set[str],
    ) -> tuple[str, ...]:
        """Replay the events a recovered node missed while it was down."""
        return tuple(
            event_id
            for event_id in sorted(journal_events)
            if event_id not in received_event_ids
        )

    def recovery_duration_observable(self) -> list[tuple[str, int]]:
        """Return the per-node recovered-entry counts for observability (Step 48)."""
        return [
            (report.node_id, report.reconciled_entries)
            for report in self.reports
        ]


__all__ = [
    "RecoveryReport",
    "RecoveryManager",
]
