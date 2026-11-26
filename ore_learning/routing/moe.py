"""Sparse Mixture-of-Experts as a routing pattern, never ORE authority (Step 9).

An MoE model combines selected expert subnetworks as y = sum_i g_i(x) E_i(x) with
sum_i g_i(x) ~ 1.  A learned router may *propose* experts, but ORE's coordinator
decides which external specialist is permitted and what budget applies; neural
experts are not equivalent to independently implemented symbolic modules.  This
module also monitors expert utilization and routing concentration to catch
router collapse (Section 9 failure mode).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ..config import LearningConfig
from ..functional import moe_combined_output, moe_routing_concentration, softmax

__all__ = ("MoEProposal", "MoERouter", "MoERoutingDecision", "RoutingConcentrationAlert")


@dataclass(slots=True)
class MoEProposal:
    """What a learned router proposes: experts, gates, and a note that it is a proposal."""

    expert_ids: tuple[str, ...]
    gates: tuple[float, ...]
    rationale: str = "learned router proposal; the coordinator decides"
    utilization: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MoERoutingDecision:
    """The coordinator's decision over a learned routing proposal."""

    approved: bool
    authorized_experts: tuple[str, ...]
    budgets: dict[str, float]
    reason: str
    proposal: MoEProposal

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {"approved": self.approved,
                "authorized_experts": list(self.authorized_experts),
                "budgets": dict(self.budgets), "reason": self.reason,
                "proposed_experts": list(self.proposal.expert_ids),
                "proposed_gates": list(self.proposal.gates)}


@dataclass(frozen=True, slots=True)
class RoutingConcentrationAlert:
    """Router-collapse signal when one expert captures too much routing share."""

    top_expert: str
    concentration: float
    threshold: float
    recommendation: str = "retain fallback and review expert utilization"

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {"top_expert": self.top_expert,
                "concentration": round(self.concentration, 4),
                "threshold": self.threshold, "recommendation": self.recommendation}


class MoERouter:
    """Step 9 -- combine expert outputs under coordinator authorization."""

    def __init__(self, config: LearningConfig | None = None) -> None:
        """Configure concentration thresholds."""
        self.config = config or LearningConfig()
        self.authorized_experts: frozenset[str] = frozenset()
        self.budgets: dict[str, float] = {}
        self.utilization: dict[str, int] = {}

    def authorize(self, expert_ids: Sequence[str],
                  budgets: dict[str, float] | None = None) -> None:
        """The coordinator (not the learned router) authorizes experts and budgets."""
        self.authorized_experts = frozenset(expert_ids)
        self.budgets = dict(budgets or {})

    def decide(self, proposal: MoEProposal) -> MoERoutingDecision:
        """Filter a learned proposal down to authorized experts (coordinator policy)."""
        allowed = [expert for expert in proposal.expert_ids
                   if expert in self.authorized_experts]
        if not allowed:
            return MoERoutingDecision(
                approved=False, authorized_experts=(), budgets={},
                reason="no proposed expert is authorized by the coordinator",
                proposal=proposal)
        missing = [expert for expert in proposal.expert_ids
                   if expert not in self.authorized_experts]
        reason = ("coordinator authorized the intersection of proposal and policy"
                  + (f"; dropped unauthorized experts {missing}" if missing else ""))
        return MoERoutingDecision(approved=True, authorized_experts=tuple(allowed),
                                  budgets={expert: self.budgets.get(expert, 0.0)
                                           for expert in allowed},
                                  reason=reason, proposal=proposal)

    def combine(self, decision: MoERoutingDecision,
                expert_outputs: dict[str, Any]) -> np.ndarray:
        """Compute y = sum_i g_i E_i(x) over the authorized experts only."""
        if not decision.approved:
            raise ValueError("cannot combine outputs of an unauthorized routing proposal")
        missing = [expert for expert in decision.authorized_experts
                   if expert not in expert_outputs]
        if missing:
            raise ValueError(f"missing expert outputs: {missing}")
        outputs = [np.asarray(expert_outputs[expert], dtype=float)
                   for expert in decision.authorized_experts]
        gate_map = dict(zip(decision.proposal.expert_ids, decision.proposal.gates))
        gates = [float(gate_map.get(expert, 0.0)) for expert in decision.authorized_experts]
        total = sum(gates)
        if total <= 0.0:
            gates = [1.0 / len(gates)] * len(gates)
        else:
            gates = [gate / total for gate in gates]
        for expert, gate in zip(decision.authorized_experts, gates):
            self.utilization[expert] = self.utilization.get(expert, 0) + 1
        return moe_combined_output(gates, outputs)

    def concentration(self, decision: MoERoutingDecision) -> float:
        """Routing concentration of a decision (expert-share monitor)."""
        gate_map = dict(zip(decision.proposal.expert_ids, decision.proposal.gates))
        gates = [gate_map.get(expert, 0.0) for expert in decision.authorized_experts]
        return moe_routing_concentration(gates) if gates else 0.0

    def check_concentration(self, decision: MoERoutingDecision
                            ) -> RoutingConcentrationAlert | None:
        """Alert when one expert exceeds the configured share (router collapse)."""
        concentration = self.concentration(decision)
        if concentration > self.config.moe_max_expert_share:
            gate_map = dict(zip(decision.proposal.expert_ids, decision.proposal.gates))
            top = max(decision.authorized_experts,
                      key=lambda expert: gate_map.get(expert, 0.0))
            return RoutingConcentrationAlert(
                top_expert=top, concentration=concentration,
                threshold=self.config.moe_max_expert_share)
        return None

    def describe(self) -> dict[str, Any]:
        """Summarize router state and expert utilization."""
        total = sum(self.utilization.values()) or 1
        return {"authorized_experts": sorted(self.authorized_experts),
                "budgets": dict(self.budgets),
                "utilization_share": {expert: count / total for expert, count
                                      in sorted(self.utilization.items())},
                "concentration_threshold": self.config.moe_max_expert_share,
                "note": ("a learned router proposes experts; the ORE coordinator "
                         "decides which specialist is permitted and the budget")}


def gates_from_scores(scores: Sequence[float]) -> list[float]:
    """Softmax gates from raw expert scores (sum_i g_i(x) ~ 1)."""
    return [float(value) for value in softmax(np.asarray(scores, dtype=float))]
