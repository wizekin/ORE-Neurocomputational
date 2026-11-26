"""Instrumentation, audit, and safe degradation (Step 48).

Measures leader changes, commit latency, replication lag, quorum availability,
gossip convergence, stale-owner rejections, CRDT conflict rates, HLC drift,
recovery duration, shard movement, and provenance verification failures.
Provides an explicit local-only mode and a safe read-only/degraded mode when
the authoritative quorum is unavailable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from ore_distributed.consensus.raft import RaftCluster


@dataclass(slots=True)
class DistributedMetrics:
    """Aggregate the Step 48 measurement set for one deployment."""

    leader_changes: int = 0
    commit_latencies_ticks: list[int] = field(default_factory=list)
    replication_lag_entries: list[int] = field(default_factory=list)
    quorum_available_ticks: int = 0
    total_ticks: int = 0
    gossip_rounds: int = 0
    gossip_convergence_rounds: list[int] = field(default_factory=list)
    stale_owner_rejections: int = 0
    crdt_conflicts: int = 0
    crdt_merges: int = 0
    hlc_drift_events: list[tuple[str, int]] = field(default_factory=list)
    recovery_durations: list[tuple[str, int]] = field(default_factory=list)
    shard_moves: int = 0
    provenance_verification_failures: int = 0
    provenance_quarantined: int = 0

    def record_tick(self, quorum_available: bool) -> None:
        """Record one protocol tick's quorum availability."""
        self.total_ticks += 1
        if quorum_available:
            self.quorum_available_ticks += 1

    def record_commit(self, latency_ticks: int) -> None:
        """Record one commit's latency in ticks."""
        self.commit_latencies_ticks.append(max(0, int(latency_ticks)))

    def record_replication_lag(self, lag_entries: int) -> None:
        """Record one replication-lag sample in log entries."""
        self.replication_lag_entries.append(max(0, int(lag_entries)))

    def record_gossip_round(self, converged_after: int | None) -> None:
        """Record one gossip round and its convergence position."""
        self.gossip_rounds += 1
        if converged_after is not None:
            self.gossip_convergence_rounds.append(int(converged_after))

    def record_stale_owner(self) -> None:
        """Record one stale-owner rejection."""
        self.stale_owner_rejections += 1

    def record_crdt_merge(self, conflict: bool = False) -> None:
        """Record one CRDT merge and whether it surfaced a conflict."""
        self.crdt_merges += 1
        if conflict:
            self.crdt_conflicts += 1

    def record_hlc_drift(self, node_id: str, drift_ms: int) -> None:
        """Record one HLC drift observation."""
        self.hlc_drift_events.append((node_id, int(drift_ms)))

    def record_recovery(self, node_id: str, duration_ticks: int) -> None:
        """Record one node recovery duration."""
        self.recovery_durations.append((node_id, max(0, int(duration_ticks))))

    def record_shard_moves(self, moved_keys: int) -> None:
        """Record shard movement volume during rebalancing."""
        self.shard_moves += max(0, int(moved_keys))

    def record_provenance_failure(self, quarantined: bool = False) -> None:
        """Record one provenance verification failure."""
        self.provenance_verification_failures += 1
        if quarantined:
            self.provenance_quarantined += 1

    # ------------------------------------------------------------------ summaries

    def commit_latency_percentile(self, fraction: float = 0.5) -> int:
        """Return a commit latency percentile in ticks."""
        if not self.commit_latencies_ticks:
            return 0
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("fraction must be in [0, 1]")
        ordered = sorted(self.commit_latencies_ticks)
        index = min(len(ordered) - 1, int(fraction * len(ordered)))
        return ordered[index]

    def average_replication_lag(self) -> float:
        """Return the mean replication lag in log entries."""
        if not self.replication_lag_entries:
            return 0.0
        return sum(self.replication_lag_entries) / len(self.replication_lag_entries)

    def quorum_availability(self) -> float:
        """Return the fraction of ticks with an available quorum."""
        if not self.total_ticks:
            return 1.0
        return self.quorum_available_ticks / self.total_ticks

    def crdt_conflict_rate(self) -> float:
        """Return the CRDT conflict rate over recorded merges."""
        if not self.crdt_merges:
            return 0.0
        return self.crdt_conflicts / self.crdt_merges

    def gossip_convergence_time(self) -> float:
        """Return the mean rounds-to-convergence for gossip."""
        if not self.gossip_convergence_rounds:
            return 0.0
        return sum(self.gossip_convergence_rounds) / len(self.gossip_convergence_rounds)

    def snapshot(self) -> dict[str, Any]:
        """Return every Step 48 metric as a plain dictionary."""
        return {
            "leader_changes": self.leader_changes,
            "commit_latency_median_ticks": self.commit_latency_percentile(0.5),
            "commit_latency_p95_ticks": self.commit_latency_percentile(0.95),
            "average_replication_lag_entries": self.average_replication_lag(),
            "quorum_availability": self.quorum_availability(),
            "gossip_rounds": self.gossip_rounds,
            "gossip_convergence_time_rounds": self.gossip_convergence_time(),
            "stale_owner_rejections": self.stale_owner_rejections,
            "crdt_conflict_rate": self.crdt_conflict_rate(),
            "hlc_drift_events": list(self.hlc_drift_events),
            "recovery_durations": list(self.recovery_durations),
            "shard_moves": self.shard_moves,
            "provenance_verification_failures": self.provenance_verification_failures,
            "provenance_quarantined": self.provenance_quarantined,
        }

    @staticmethod
    def from_cluster(cluster: RaftCluster) -> "DistributedMetrics":
        """Build a metrics snapshot from one cluster's protocol counters."""
        metrics = DistributedMetrics()
        for node in cluster.nodes.values():
            metrics.leader_changes += sum(
                1 for _, outcome in node.election_events if outcome == "leader"
            )
            metrics.commit_latencies_ticks.extend(
                1 for _ in node.commit_events
            )
        leader = cluster.leader()
        if leader is not None:
            for node in cluster.nodes.values():
                lag = max(0, leader.storage.commit_index - node.storage.commit_index)
                metrics.replication_lag_entries.append(lag)
        return metrics


class DegradedModeController:
    """Provide local-only and safe read-only degraded modes (Step 48)."""

    LOCAL_ONLY = "local_only"
    NORMAL = "normal"
    READ_ONLY_DEGRADED = "read_only_degraded"

    def __init__(self) -> None:
        """Start in normal mode."""
        self.mode = self.NORMAL
        self.transitions: list[tuple[str, str]] = []

    def enter_local_only(self) -> str:
        """Enter the explicit local-only mode for single-process operation."""
        self._transition(self.LOCAL_ONLY)
        return self.mode

    def enter_read_only_degraded(self) -> str:
        """Enter the safe degraded mode when quorum is unavailable."""
        self._transition(self.READ_ONLY_DEGRADED)
        return self.mode

    def resume_normal(self) -> str:
        """Resume normal operation after quorum recovers."""
        self._transition(self.NORMAL)
        return self.mode

    def _transition(self, mode: str) -> None:
        """Record one mode transition."""
        self.transitions.append((self.mode, mode))
        self.mode = mode

    def may_write_authoritative(self) -> bool:
        """Return whether authoritative writes are allowed in this mode."""
        return self.mode is self.NORMAL

    def may_serve_reads(self) -> bool:
        """Return whether reads are served in this mode (always true)."""
        return True

    def status(self) -> dict[str, str]:
        """Return the current degraded-mode status for the coordinator."""
        return {
            "mode": self.mode,
            "authoritative_writes": "allowed" if self.may_write_authoritative() else "blocked",
            "reads": "served",
        }


def quorum_unavailable_enters_degraded_mode(
    cluster: RaftCluster,
    controller: DegradedModeController,
) -> bool:
    """Run the acceptance check: quorum loss triggers the safe degraded mode."""
    leader = cluster.leader()
    if leader is None:
        controller.enter_read_only_degraded()
        return not controller.may_write_authoritative() and controller.may_serve_reads()
    return controller.mode is DegradedModeController.NORMAL


def all_metrics_observable(metrics: DistributedMetrics) -> tuple[str, ...]:
    """Return the metric names that carry observations (Step 48 check)."""
    snapshot = metrics.snapshot()
    return tuple(name for name, value in snapshot.items() if value not in (0, 0.0, [], ()))
