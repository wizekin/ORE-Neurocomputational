"""Edge deployment decisions (Point 14, Steps 38 and 40).

Edge deployment is justified by latency, connectivity, privacy, or bandwidth needs,
after comparing local with centralized execution on accuracy, update cost, power,
memory, and maintenance.  Split execution places lightweight observation extraction
and low-risk classification at the edge while central ORE keeps semantic integration
and deeper reasoning; edge components can never bypass central authorization for
consequential actions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..config import LearningConfig

__all__ = ("EdgeDecision", "EdgeDeploymentPlanner", "SplitExecutionPlan")


@dataclass(slots=True)
class EdgeDecision:
    """Step 38 -- the measured local-vs-central comparison and its verdict."""

    deploy_at_edge: bool
    reasons: list[str]
    local_accuracy: float
    central_accuracy: float
    local_latency_ms: float
    central_latency_ms: float
    local_memory_mb: float
    update_cost_local: float
    update_cost_central: float
    maintenance_burden: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {"deploy_at_edge": self.deploy_at_edge, "reasons": list(self.reasons),
                "local_accuracy": round(self.local_accuracy, 6),
                "central_accuracy": round(self.central_accuracy, 6),
                "local_latency_ms": round(self.local_latency_ms, 3),
                "central_latency_ms": round(self.central_latency_ms, 3),
                "local_memory_mb": round(self.local_memory_mb, 3),
                "update_cost_local": self.update_cost_local,
                "update_cost_central": self.update_cost_central,
                "maintenance_burden": self.maintenance_burden}


class EdgeDeploymentPlanner:
    """Steps 38 and 40 -- decide, and plan split execution when useful."""

    def __init__(self, config: LearningConfig | None = None) -> None:
        """Configure the accuracy tolerance for edge deployment."""
        self.config = config or LearningConfig()

    def decide(self, *, local_accuracy: float, central_accuracy: float,
               local_latency_ms: float, central_latency_ms: float,
               local_memory_mb: float, device_memory_mb: float,
               connectivity_available: bool, privacy_sensitive: bool,
               bandwidth_cost_per_mb: float, update_cost_local: float = 0.8,
               update_cost_central: float = 0.2) -> EdgeDecision:
        """Compare local and centralized execution on the measured axes (Step 38)."""
        reasons: list[str] = []
        if local_memory_mb > device_memory_mb:
            reasons.append(f"edge model needs {local_memory_mb:.1f} MB but the device "
                           f"declares {device_memory_mb:.1f} MB")
        accuracy_gap = central_accuracy - local_accuracy
        if accuracy_gap > 0.1:
            reasons.append(f"edge accuracy trails central by {accuracy_gap:.3f}")
        if local_latency_ms >= central_latency_ms and not privacy_sensitive:
            reasons.append("central execution is at least as fast and not "
                           "privacy-constrained")
        if not connectivity_available:
            reasons.append("no connectivity: edge is the only executable option")
        if privacy_sensitive:
            reasons.append("privacy-sensitive inputs favour on-device inference")
        if local_latency_ms < central_latency_ms and connectivity_available:
            reasons.append(f"local inference saves "
                           f"{central_latency_ms - local_latency_ms:.1f} ms of latency "
                           "and bandwidth")
        if bandwidth_cost_per_mb > 1.0:
            reasons.append("high bandwidth cost favours edge pre-processing")
        deploy = not any(reason.startswith(("edge model needs", "edge accuracy"))
                         for reason in reasons) and (
            privacy_sensitive or not connectivity_available
            or local_latency_ms < central_latency_ms)
        return EdgeDecision(
            deploy_at_edge=deploy, reasons=reasons,
            local_accuracy=local_accuracy, central_accuracy=central_accuracy,
            local_latency_ms=local_latency_ms, central_latency_ms=central_latency_ms,
            local_memory_mb=local_memory_mb,
            update_cost_local=update_cost_local,
            update_cost_central=update_cost_central,
            maintenance_burden=("edge fleet requires coordinated rollout and "
                                "monitoring" if deploy else "single central artifact"))

    def plan_split_execution(self, decision: EdgeDecision, *,
                             edge_tasks: tuple[str, ...] = ("observation_extraction",
                                                            "low_risk_classification"),
                             central_tasks: tuple[str, ...] = ("semantic_integration",
                                                               "deep_reasoning",
                                                               "consequential_actions")
                             ) -> SplitExecutionPlan:
        """Step 40 -- the edge/central split with the authorization boundary."""
        return SplitExecutionPlan(
            deploy_at_edge=decision.deploy_at_edge,
            edge_tasks=edge_tasks, central_tasks=central_tasks)


@dataclass(slots=True)
class SplitExecutionPlan:
    """Step 40 -- what runs where, and what the edge may never do."""

    deploy_at_edge: bool
    edge_tasks: tuple[str, ...]
    central_tasks: tuple[str, ...]

    @property
    def authorization_boundary(self) -> str:
        """Edge components cannot bypass central authorization."""
        return ("edge components never authorize consequential actions; central ORE "
                "retains authorization authority")

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {"deploy_at_edge": self.deploy_at_edge,
                "edge_tasks": list(self.edge_tasks),
                "central_tasks": list(self.central_tasks),
                "authorization_boundary": self.authorization_boundary}
