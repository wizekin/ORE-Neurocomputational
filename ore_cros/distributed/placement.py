"""Distributed placement and migration (ORE Point 23, Steps 36-41).

Covers node resource advertisements with expiry, hard eligibility
filtering, cost-and-locality placement scoring, cost-aware migration
decisions and side-effect-aware failure recovery. Borg, the Kubernetes
scheduler and Decima are documented design references (Step 39): only
principles appropriate to a single-PC or small-cluster ORE deployment are
adopted here.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

from ..resource_model import (
    DEMAND_KINDS, ResourceDemand, SideEffectPolicy, TaskRecord,
)


@dataclass(frozen=True)
class NodeAdvertisement:
    """One node's published resource advertisement (spec Step 36)."""

    node_id: str
    capabilities: tuple[str, ...]
    capacity: ResourceDemand
    load: ResourceDemand
    data_locality: tuple[str, ...]
    healthy: bool
    version: int
    published_at: float
    expires_after_seconds: float

    def __post_init__(self) -> None:
        """Validate the advertisement record."""
        if not isinstance(self.node_id, str) or not self.node_id.strip():
            raise ValueError("node_id must be a non-empty string")
        if self.version < 0:
            raise ValueError("advertisement version must be non-negative")
        if not math.isfinite(self.published_at):
            raise ValueError("published_at must be finite")
        if not math.isfinite(self.expires_after_seconds) or self.expires_after_seconds <= 0:
            raise ValueError("expires_after_seconds must be finite and positive")

    def is_expired(self, *, now: float) -> bool:
        """Return True when the advertisement has expired (Step 36)."""
        return (now - self.published_at) > self.expires_after_seconds

    def headroom(self) -> ResourceDemand:
        """Return capacity minus current load per resource kind."""
        return ResourceDemand(
            cpu=max(0.0, self.capacity.cpu - self.load.cpu),
            ram=max(0.0, self.capacity.ram - self.load.ram),
            io=max(0.0, self.capacity.io - self.load.io),
            net=max(0.0, self.capacity.net - self.load.net),
            gpu=max(0.0, self.capacity.gpu - self.load.gpu),
            storage=max(0.0, self.capacity.storage - self.load.storage),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible snapshot of the advertisement."""
        return {
            "node_id": self.node_id, "capabilities": list(self.capabilities),
            "capacity": self.capacity.to_dict(), "load": self.load.to_dict(),
            "data_locality": list(self.data_locality), "healthy": self.healthy,
            "version": self.version, "published_at": self.published_at,
            "expires_after_seconds": self.expires_after_seconds,
        }


class ClusterResourceMap:
    """Versioned cluster resource map built from advertisements (Step 36)."""

    def __init__(self) -> None:
        """Initialize the empty map."""
        self._nodes: dict[str, NodeAdvertisement] = {}
        self.version = 0

    def publish(self, advertisement: NodeAdvertisement) -> None:
        """Publish or refresh one node advertisement."""
        existing = self._nodes.get(advertisement.node_id)
        if existing is not None and advertisement.version < existing.version:
            raise ValueError(
                f"stale advertisement for {advertisement.node_id!r}: version "
                f"{advertisement.version} predates {existing.version}"
            )
        self._nodes[advertisement.node_id] = advertisement
        self.version += 1

    def prune(self, *, now: float) -> tuple[str, ...]:
        """Remove expired advertisements and return their node ids."""
        expired = tuple(sorted(
            node_id for node_id, advertisement in self._nodes.items()
            if advertisement.is_expired(now=now)
        ))
        for node_id in expired:
            del self._nodes[node_id]
        if expired:
            self.version += 1
        return expired

    def nodes(self) -> tuple[NodeAdvertisement, ...]:
        """Return every live advertisement."""
        return tuple(
            self._nodes[node_id] for node_id in sorted(self._nodes)
        )

    def node(self, node_id: str) -> NodeAdvertisement | None:
        """Return one node's advertisement, if present."""
        return self._nodes.get(node_id)


@dataclass(frozen=True)
class EligibilityCheck:
    """Outcome of one node eligibility evaluation (spec Step 37)."""

    node_id: str
    eligible: bool
    capability_ok: bool
    budget_ok: bool
    policy_ok: bool
    data_access_ok: bool
    health_ok: bool

    @property
    def reason(self) -> str:
        """Return the human-readable explanation of the verdict."""
        if self.eligible:
            return "capability AND budget AND policy AND data access AND health"
        failed = [
            name for name, ok in (
                ("capability", self.capability_ok), ("budget", self.budget_ok),
                ("policy", self.policy_ok), ("data_access", self.data_access_ok),
                ("health", self.health_ok),
            ) if not ok
        ]
        return f"failed: {failed}"

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible snapshot of the check."""
        return {
            "node_id": self.node_id, "eligible": self.eligible,
            "capability_ok": self.capability_ok, "budget_ok": self.budget_ok,
            "policy_ok": self.policy_ok, "data_access_ok": self.data_access_ok,
            "health_ok": self.health_ok, "reason": self.reason,
        }


def filter_eligible_nodes(
    task: TaskRecord,
    cluster: ClusterResourceMap,
    *,
    policy_allowed_nodes: Iterable[str] | None = None,
) -> tuple[NodeAdvertisement, ...]:
    """Filter candidate nodes by hard eligibility (spec Step 37).

    eligibility = capability AND budget AND policy AND data_access AND
    health. A node is eligible only with required capabilities, sufficient
    budget, compatible software, access to the task's data, allowed
    policy and healthy status.
    """
    allowed = set(policy_allowed_nodes) if policy_allowed_nodes is not None else None
    eligible: list[NodeAdvertisement] = []
    for node in cluster.nodes():
        check = check_eligibility(task, node, policy_allowed_nodes=allowed)
        if check.eligible:
            eligible.append(node)
    return tuple(eligible)


def check_eligibility(
    task: TaskRecord, node: NodeAdvertisement,
    *,
    policy_allowed_nodes: set[str] | None = None,
) -> EligibilityCheck:
    """Evaluate every eligibility conjunct for one node (Step 37)."""
    capability_ok = set(task.capabilities) <= set(node.capabilities)
    budget_ok = _budget_ok(task, node.headroom())
    policy_ok = policy_allowed_nodes is None or node.node_id in policy_allowed_nodes
    data_access_ok = (
        not task.input_data_locality
        or set(task.input_data_locality) <= set(node.data_locality)
    )
    health_ok = node.healthy
    return EligibilityCheck(
        node_id=node.node_id,
        eligible=all((capability_ok, budget_ok, policy_ok, data_access_ok, health_ok)),
        capability_ok=capability_ok, budget_ok=budget_ok, policy_ok=policy_ok,
        data_access_ok=data_access_ok, health_ok=health_ok,
    )


def _budget_ok(task: TaskRecord, headroom: ResourceDemand) -> bool:
    """Return True when the task demand fits the node headroom."""
    return all(
        task.demand.get(kind) <= headroom.get(kind) + 1e-9
        for kind in DEMAND_KINDS
    )


@dataclass(frozen=True)
class PlacementDecision:
    """Ranked placement decision with explanation (spec Step 38)."""

    task_id: str
    ranked_nodes: tuple[str, ...]
    scores: Mapping[str, float]
    explanation: Mapping[str, str]
    selected_node: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible snapshot of the decision."""
        return {
            "task_id": self.task_id, "ranked_nodes": list(self.ranked_nodes),
            "scores": dict(self.scores), "explanation": dict(self.explanation),
            "selected_node": self.selected_node,
        }


class PlacementScorer:
    """Score placement by estimated completion cost (spec Step 38).

    Cost model: queue delay + data transfer time + network latency +
    contention penalty + accelerator availability penalty. The ranked
    decision is returned with per-node explanations; hard eligibility is
    a precondition, not part of the score.
    """

    def __init__(
        self,
        *,
        network_bandwidth_mbit: float = 1000.0,
        network_latency_seconds: float = 0.001,
        contention_penalty: float = 0.5,
    ) -> None:
        """Configure the transfer and contention cost model."""
        if network_bandwidth_mbit <= 0.0 or network_latency_seconds < 0.0:
            raise ValueError("bandwidth must be positive and latency non-negative")
        if contention_penalty < 0.0:
            raise ValueError("contention_penalty must be non-negative")
        self.network_bandwidth_mbit = network_bandwidth_mbit
        self.network_latency_seconds = network_latency_seconds
        self.contention_penalty = contention_penalty

    def score(
        self, task: TaskRecord, nodes: Sequence[NodeAdvertisement],
    ) -> PlacementDecision:
        """Return the ranked placement decision for one task (Step 38)."""
        if not nodes:
            return PlacementDecision(
                task_id=task.task_id, ranked_nodes=(), scores={},
                explanation={}, selected_node=None,
            )
        scores: dict[str, float] = {}
        explanations: dict[str, str] = {}
        for node in nodes:
            headroom = node.headroom()
            queue_delay = _estimated_queue_delay(node)
            transfer_mib = task.input_size_mib if not (
                set(task.input_data_locality) <= set(node.data_locality)
            ) else 0.0
            transfer_seconds = (
                (transfer_mib * 8.0 / 1024.0) / self.network_bandwidth_mbit
                if transfer_mib > 0 else 0.0
            )
            utilization = _mean_utilization(node)
            contention = self.contention_penalty * utilization
            accelerator_penalty = 0.5 if task.demand.gpu > headroom.gpu + 1e-9 else 0.0
            cost = queue_delay + transfer_seconds + self.network_latency_seconds
            cost += contention + accelerator_penalty
            scores[node.node_id] = cost
            explanations[node.node_id] = (
                f"queue={queue_delay:.3f}s transfer={transfer_seconds:.3f}s "
                f"latency={self.network_latency_seconds:.3f}s "
                f"contention={contention:.3f} accelerator={accelerator_penalty:.3f}"
            )
        ranked = sorted(scores, key=lambda node_id: (scores[node_id], node_id))
        return PlacementDecision(
            task_id=task.task_id, ranked_nodes=tuple(ranked), scores=scores,
            explanation=explanations, selected_node=ranked[0],
        )


def _estimated_queue_delay(node: NodeAdvertisement) -> float:
    """Estimate queue delay from load relative to capacity."""
    capacity_total = sum(node.capacity.to_dict().values())
    load_total = sum(node.load.to_dict().values())
    if capacity_total <= 0.0:
        return 0.0
    return min(5.0, (load_total / capacity_total) * 1.0)


def _mean_utilization(node: NodeAdvertisement) -> float:
    """Return the mean utilization across resource kinds."""
    capacity = node.capacity.to_dict()
    load = node.load.to_dict()
    ratios = [
        load[kind] / capacity[kind] for kind in capacity
        if capacity[kind] > 0.0
    ]
    return sum(ratios) / len(ratios) if ratios else 0.0


#: Documented cluster-scheduling design references (spec Step 39).
CLUSTER_DESIGN_REFERENCES: tuple[dict[str, Any], ...] = (
    {
        "reference": "Borg",
        "principle_adopted": "explicit resource quotas and priority classes for "
                             "admission control",
        "rejected_for_scale": "cell-level centralized fleet scheduling",
    },
    {
        "reference": "Kubernetes Scheduler",
        "principle_adopted": "predicate (hard constraint) filtering followed by "
                             "priority-based scoring",
        "rejected_for_scale": "cluster-wide controller machinery ORE does not need",
    },
    {
        "reference": "Decima",
        "principle_adopted": "learning-based scheduling must run in shadow mode "
                             "before earning control",
        "rejected_for_scale": "deep-RL job schedulers requiring large training fleets",
    },
)


@dataclass(frozen=True)
class MigrationCosts:
    """Cost breakdown for one migration decision (spec Step 40)."""

    expected_benefit: float
    transfer_cost: float
    restart_cost: float
    risk_penalty: float

    @property
    def total_cost(self) -> float:
        """Return the combined migration cost."""
        return self.transfer_cost + self.restart_cost + self.risk_penalty

    @property
    def advisable(self) -> bool:
        """Migrate only when benefit exceeds total cost (Step 40)."""
        return self.expected_benefit > self.total_cost


def decide_migration(
    task: TaskRecord,
    *,
    expected_benefit: float,
    transfer_cost: float,
    restart_cost: float,
    risk_penalty: float,
) -> MigrationCosts:
    """Cost-aware migration decision requiring a safe restart point (Step 40).

    A task without checkpoint or safe-restart support is never migrated,
    regardless of the cost model.
    """
    for name, value in (
        ("expected_benefit", expected_benefit), ("transfer_cost", transfer_cost),
        ("restart_cost", restart_cost), ("risk_penalty", risk_penalty),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    costs = MigrationCosts(
        expected_benefit=expected_benefit, transfer_cost=transfer_cost,
        restart_cost=restart_cost, risk_penalty=risk_penalty,
    )
    if costs.advisable and task.checkpoint.value == "NONE":
        raise ValueError(
            f"task {task.task_id!r} has no checkpoint or safe-restart point; "
            "migration is not permitted"
        )
    return costs


@dataclass(frozen=True)
class RecoveryDecision:
    """Side-effect-aware recovery decision for one in-flight task (Step 41)."""

    task_id: str
    action: str  # "retry" | "restart-from-checkpoint" | "cancel"
    reason: str
    presumed_running: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible snapshot of the decision."""
        return {
            "task_id": self.task_id, "action": self.action,
            "reason": self.reason, "presumed_running": self.presumed_running,
        }


class FailureDetector:
    """Heartbeat-based failure detection with side-effect-aware recovery (Step 41).

    A timeout does not prove a remote task stopped: until a fencing
    decision is made, affected tasks are marked presumed running.
    """

    def __init__(self, *, heartbeat_timeout_seconds: float = 5.0) -> None:
        """Configure the missed-heartbeat threshold."""
        if heartbeat_timeout_seconds <= 0.0:
            raise ValueError("heartbeat_timeout_seconds must be positive")
        self.heartbeat_timeout_seconds = heartbeat_timeout_seconds
        self._last_heartbeat: dict[str, float] = {}

    def record_heartbeat(self, node_id: str, *, now: float) -> None:
        """Record one observed heartbeat for a node."""
        self._last_heartbeat[node_id] = now

    def unavailable_nodes(self, *, now: float) -> tuple[str, ...]:
        """Return nodes whose heartbeats have timed out (Step 41)."""
        return tuple(sorted(
            node_id for node_id, last in self._last_heartbeat.items()
            if (now - last) > self.heartbeat_timeout_seconds
        ))

    def isolate(self, node_id: str) -> None:
        """Stop tracking an isolated node."""
        self._last_heartbeat.pop(node_id, None)

    @staticmethod
    def recover(
        task: TaskRecord, *, node_presumed_alive: bool,
    ) -> RecoveryDecision:
        """Classify in-flight task recovery by side-effect policy (Step 41)."""
        if node_presumed_alive:
            return RecoveryDecision(
                task_id=task.task_id, action="retry", reason=(
                    "timeout does not prove the remote task stopped; task is "
                    "idempotent, safe to await or retry"
                ) if task.side_effects is SideEffectPolicy.IDEMPOTENT else (
                    "node possibly alive; await fencing decision before acting"
                ),
                presumed_running=True,
            )
        if task.side_effects is SideEffectPolicy.IDEMPOTENT:
            return RecoveryDecision(
                task_id=task.task_id, action="retry",
                reason="side effects are idempotent; retry is safe",
                presumed_running=False,
            )
        if task.side_effects is SideEffectPolicy.NON_IDEMPOTENT_CHECKPOINTED:
            return RecoveryDecision(
                task_id=task.task_id, action="restart-from-checkpoint",
                reason="checkpoint available; restart from the last checkpoint",
                presumed_running=False,
            )
        return RecoveryDecision(
            task_id=task.task_id, action="cancel",
            reason="non-idempotent without checkpoint; cancel with explicit record",
            presumed_running=False,
        )


__all__ = [
    "NodeAdvertisement", "ClusterResourceMap", "EligibilityCheck",
    "filter_eligible_nodes", "check_eligibility", "PlacementScorer",
    "PlacementDecision", "CLUSTER_DESIGN_REFERENCES", "MigrationCosts",
    "decide_migration", "FailureDetector", "RecoveryDecision",
]
