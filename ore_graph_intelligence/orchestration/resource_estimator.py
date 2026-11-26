"""Resource estimation and adaptive admission for graph tasks (ORE Point 8 step 50).

The Main Brain Coordinator estimates nodes/edges, neighborhood size, model load and
latency; the adaptive coordinator (CROS) admits, defers, samples or rejects the task
against its budget. Oversized tasks are sampled/deferred/rejected, never silently run
(acceptance test 23).
"""
from __future__ import annotations

from dataclasses import dataclass

from ..projection.typed_graph import TypedGraphSnapshot
from ..schemas.candidate_schema import GraphQuery

#: Deterministic cost model constants (documented, not learned).
_PER_NODE_BYTES = 512.0
_PER_EDGE_BYTES = 256.0
_EMBEDDING_MODEL_LOAD_MB = 2.0
_GNN_MODEL_LOAD_MB = 4.0
_PER_CANDIDATE_MS = 0.25


@dataclass(frozen=True, slots=True)
class ResourceEstimate:
    """Deterministic estimate of one graph task's footprint and latency."""

    n_nodes: int
    n_edges: int
    neighborhood_nodes: int
    estimated_ram_mb: float
    model_load_mb: float
    estimated_latency_ms: float
    candidate_budget: int

    def as_dict(self) -> dict[str, object]:
        return {
            "n_nodes": self.n_nodes, "n_edges": self.n_edges,
            "neighborhood_nodes": self.neighborhood_nodes,
            "estimated_ram_mb": round(self.estimated_ram_mb, 3),
            "model_load_mb": self.model_load_mb,
            "estimated_latency_ms": round(self.estimated_latency_ms, 3),
            "candidate_budget": self.candidate_budget,
        }


@dataclass(frozen=True, slots=True)
class ResourceBudget:
    """Bounded resource envelope set by the coordinator (CROS admission limits)."""

    deadline_ms: float = 500.0
    max_ram_mb: float = 256.0
    max_neighborhood_nodes: int = 4096
    max_candidates: int = 64
    max_models: int = 2

    def __post_init__(self) -> None:
        if self.deadline_ms <= 0 or self.max_ram_mb <= 0:
            raise ValueError("deadline and RAM budget must be positive")
        if self.max_neighborhood_nodes < 1 or self.max_candidates < 1:
            raise ValueError("neighborhood and candidate bounds must be at least 1")
        if self.max_models < 1:
            raise ValueError("max_models must be at least 1")


class GraphResourceEstimator:
    """Estimate task footprint from snapshot statistics and query bounds (step 50)."""

    def __init__(self, *, include_models: tuple[str, ...] = ("embeddings", "gnn")) -> None:
        self.include_models = tuple(include_models)

    def estimate(
        self,
        snapshot: TypedGraphSnapshot,
        query: GraphQuery | None = None,
    ) -> ResourceEstimate:
        """Estimate nodes, edges, neighborhood, RAM, model load and latency."""
        n_nodes = snapshot.node_count
        n_edges = snapshot.edge_count
        depth = query.neighborhood_depth if query is not None else 3
        budget = query.budget_candidates if query is not None else 32
        degree_out, degree_in = 0, 0
        if n_nodes:
            degrees = [snapshot.degree_of(node.node_id) for node in snapshot.nodes[:64]]
            degree_out = max((d[0] for d in degrees), default=0)
            degree_in = max((d[1] for d in degrees), default=0)
        mean_degree = max(1.0, (degree_out + degree_in) / 2.0)
        neighborhood = int(min(n_nodes, 1 + (mean_degree ** max(0, depth - 1)) * mean_degree))
        graph_mb = (n_nodes * _PER_NODE_BYTES + n_edges * _PER_EDGE_BYTES) / (1024.0 * 1024.0)
        model_mb = 0.0
        if "embeddings" in self.include_models:
            model_mb += _EMBEDDING_MODEL_LOAD_MB
        if "gnn" in self.include_models:
            model_mb += _GNN_MODEL_LOAD_MB
        model_mb = min(model_mb, self._model_cap(len(self.include_models)))
        latency = 5.0 + budget * _PER_CANDIDATE_MS + 0.02 * neighborhood
        return ResourceEstimate(
            n_nodes=n_nodes,
            n_edges=n_edges,
            neighborhood_nodes=neighborhood,
            estimated_ram_mb=graph_mb + model_mb,
            model_load_mb=model_mb,
            estimated_latency_ms=latency,
            candidate_budget=budget,
        )

    @staticmethod
    def _model_cap(count: int) -> float:
        return count * _GNN_MODEL_LOAD_MB


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    """CROS admission verdict for one task (step 50)."""

    decision: str  # admit | sample | defer | reject
    reason: str
    sampled_candidate_budget: int | None = None
    deferred_retry_after_ms: float | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "decision": self.decision, "reason": self.reason,
            "sampled_candidate_budget": self.sampled_candidate_budget,
            "deferred_retry_after_ms": self.deferred_retry_after_ms,
        }


class AdaptiveResourceCoordinator:
    """Admission control: admit, defer, sample or reject tasks under budget (step 50)."""

    def __init__(self, budget: ResourceBudget | None = None) -> None:
        self.budget = budget or ResourceBudget()

    def admit(self, estimate: ResourceEstimate) -> AdmissionDecision:
        """Decide admission from the estimate against the resource budget."""
        if estimate.estimated_ram_mb > self.budget.max_ram_mb:
            return AdmissionDecision(
                "reject",
                f"estimated RAM {estimate.estimated_ram_mb:.1f} MB exceeds "
                f"{self.budget.max_ram_mb:.1f} MB",
            )
        if estimate.neighborhood_nodes > self.budget.max_neighborhood_nodes:
            return AdmissionDecision(
                "reject",
                f"neighborhood {estimate.neighborhood_nodes} exceeds "
                f"{self.budget.max_neighborhood_nodes} nodes",
            )
        if estimate.estimated_latency_ms > self.budget.deadline_ms:
            if estimate.candidate_budget > 8:
                sampled = max(8, self.budget.max_candidates // 4)
                return AdmissionDecision(
                    "sample",
                    "latency exceeds deadline; candidate budget sampled down",
                    sampled_candidate_budget=sampled,
                )
            return AdmissionDecision(
                "defer",
                f"latency {estimate.estimated_latency_ms:.1f} ms exceeds deadline; deferred",
                deferred_retry_after_ms=self.budget.deadline_ms,
            )
        if estimate.candidate_budget > self.budget.max_candidates:
            return AdmissionDecision(
                "sample",
                f"candidate budget {estimate.candidate_budget} exceeds "
                f"{self.budget.max_candidates}; sampled down",
                sampled_candidate_budget=self.budget.max_candidates,
            )
        return AdmissionDecision("admit", "estimate fits the resource budget")


__all__ = [
    "AdmissionDecision", "AdaptiveResourceCoordinator", "GraphResourceEstimator",
    "ResourceBudget", "ResourceEstimate",
]
