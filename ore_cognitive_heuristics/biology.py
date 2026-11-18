"""Distributed rate-coded reference network for Point 22 (not a neural simulation)."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping


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


class CognitiveHeuristicsNetwork:
    """Advance a recurrent systems-level analogy for memory, valuation, and cognitive control."""

    POPULATIONS = (
        "episodic_memory", "semantic_prototypes", "frontoparietal_attention",
        "parietal_magnitude", "prefrontal_integration", "vm_pfc_valuation",
        "amygdala_insula_salience", "anterior_cingulate_monitoring",
        "lateral_prefrontal_control", "metacognitive_confidence",
        "corticostriatal_learning", "sensory_association",
    )

    def __init__(self, enabled: bool = True) -> None:
        """Create independent population-rate state."""
        self.enabled = enabled
        self.populations = {name: FunctionalPopulation(name) for name in self.POPULATIONS}
        self.operation_count = 0
        self.last_operation_id: str | None = None

    def advance(
        self,
        operation_id: str,
        drives: Mapping[str, float],
        *,
        salience: float = 0.0,
        conflict: float = 0.0,
    ) -> dict:
        """Update populations and return activity plus recurrent coupling diagnostics."""
        if not self.enabled:
            return {"model": "disabled", "operation_id": operation_id}
        prior = {name: row.activation for name, row in self.populations.items()}
        outputs: dict[str, float] = {}
        for name, population in self.populations.items():
            drive = float(drives.get(name, 0.0))
            if name == "amygdala_insula_salience":
                drive = max(drive, salience)
            if name in {"anterior_cingulate_monitoring", "lateral_prefrontal_control"}:
                drive = max(drive, conflict)
            recurrent = 0.0
            if name in {"frontoparietal_attention", "prefrontal_integration",
                        "lateral_prefrontal_control"}:
                recurrent = prior["anterior_cingulate_monitoring"]
            elif name in {"episodic_memory", "semantic_prototypes"}:
                recurrent = prior["prefrontal_integration"]
            elif name == "metacognitive_confidence":
                recurrent = prior["anterior_cingulate_monitoring"]
            outputs[name] = population.update(drive, recurrent)
        self.operation_count += 1
        self.last_operation_id = operation_id
        return {
            "model": "distributed_functional_population_rate_reference",
            "operation_id": operation_id,
            "population_activity": outputs,
            "mean_activity": sum(outputs.values()) / len(outputs),
            "salience_drive": _clamp(salience),
            "conflict_drive": _clamp(conflict),
            "scientific_limit": "Illustrative systems-level activity; not measured or anatomically fitted neural data.",
        }

    def snapshot(self) -> dict:
        """Return the complete current population state."""
        return {
            "model": "distributed_functional_population_rate_reference" if self.enabled else "disabled",
            "operation_count": self.operation_count,
            "last_operation_id": self.last_operation_id,
            "populations": {
                name: {"activation": row.activation, "samples": len(row.history)}
                for name, row in self.populations.items()
            },
            "scientific_limitations": [
                "Population values are illustrative engineering abstractions.",
                "The network is not a literal neuronal, anatomical, or validated cognitive simulation.",
                "Named regions indicate broad systems-level associations, not one-region localization.",
            ],
        }
