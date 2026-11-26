"""Distributed rate-coded reference network for MACL (not a neural simulation).

The multi-agent coordination layer maps onto systems-level neuroscience analogies:
a prefrontal supervisor maintaining goals, anterior cingulate monitoring for
conflict, a frontoparietal control network gating tasks, temporal-parietal and
medial prefrontal circuits supporting agent modelling, basal ganglia action
selection, and cerebellar timing for predictive coordination. As elsewhere in ORE,
these coarse population rates are an interpretive reference, not a claim about
neural implementation.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field


def _clamp(value: float) -> float:
    """Bound a population activity to the unit interval."""
    return max(0.0, min(1.0, float(value)))


@dataclass
class FunctionalPopulation:
    """Store a coarse activity value for one functional population."""

    name: str
    activation: float = 0.0
    history: list[float] = field(default_factory=list)

    def update(self, drive: float, recurrent: float = 0.0) -> float:
        """Apply a bounded leaky-rate update with saturating nonlinearity."""
        total = float(drive) + 0.35 * float(recurrent)
        target = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, 4.0 * (total - 0.5)))))
        self.activation = _clamp(0.65 * self.activation + 0.35 * target)
        self.history.append(self.activation)
        return self.activation


class MultiAgentCoordinationNetwork:
    """Advance a recurrent systems-level analogy for hierarchical coordination circuits."""

    POPULATIONS = (
        "dlpfc_supervisor",          # goal decomposition and policy gates
        "acc_conflict_monitor",      # adjudication, conflict and error monitoring
        "frontoparietal_control",    # task-set maintenance and assignment gating
        "tpj_agent_modeling",        # modelling other agents' capabilities and states
        "mpfc_social_coordination",  # peer-to-peer negotiation and offers
        "basal_ganglia_selection",   # bounded action and agent selection
        "cerebellar_timing",         # deadline, heartbeat and progress prediction
        "anterior_insula_uncertainty",  # reliability uncertainty and risk
        "hippocampal_context",       # workspace revision and provenance context
        "striatal_reinforcement",    # performance ledger learning
    )

    def __init__(self, enabled: bool = True) -> None:
        """Create independent population-rate state."""
        self.enabled = enabled
        self.populations = {name: FunctionalPopulation(name) for name in self.POPULATIONS}
        self.step_count = 0

    def advance(self, drives: Mapping[str, float]) -> dict[str, float]:
        """Advance all populations with the supplied per-circuit drive levels."""
        self.step_count += 1
        activities: dict[str, float] = {}
        for name, population in self.populations.items():
            drive = drives.get(name, 0.0)
            recurrent = population.activation
            activities[name] = population.update(drive if self.enabled else 0.0, recurrent)
        return activities

    def snapshot(self) -> dict[str, float]:
        """Return the current activation of every population."""
        return {name: population.activation for name, population in self.populations.items()}


#: Circuit drives associated with each MACL coordination stage (A-I in the source).
STAGE_DRIVES: tuple[Mapping[str, float], ...] = (
    {"dlpfc_supervisor": 0.85, "frontoparietal_control": 0.70},
    {"dlpfc_supervisor": 0.90, "frontoparietal_control": 0.75, "hippocampal_context": 0.50},
    {"dlpfc_supervisor": 0.90, "basal_ganglia_selection": 0.85, "anterior_insula_uncertainty": 0.55},
    {"frontoparietal_control": 0.85, "cerebellar_timing": 0.70, "acc_conflict_monitor": 0.55},
    {"hippocampal_context": 0.85, "acc_conflict_monitor": 0.70},
    {"mpfc_social_coordination": 0.85, "tpj_agent_modeling": 0.75},
    {"acc_conflict_monitor": 0.90, "tpj_agent_modeling": 0.65, "dlpfc_supervisor": 0.70},
    {"striatal_reinforcement": 0.80, "anterior_insula_uncertainty": 0.70},
    {"striatal_reinforcement": 0.85, "basal_ganglia_selection": 0.60},
)
