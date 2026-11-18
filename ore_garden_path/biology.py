"""Distributed population-rate reference network for Point 23 sentence comprehension."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping


def _bounded(value: float) -> float:
    """Keep illustrative activity inside [0, 1]."""
    return max(0.0, min(1.0, float(value)))


@dataclass
class FunctionalPopulation:
    """Maintain a coarse rate-coded activity and its operation history."""

    name: str
    activation: float = 0.0
    history: list[float] = field(default_factory=list)

    def update(self, drive: float, recurrent: float = 0.0) -> float:
        """Apply a leaky population-rate update with a bounded sigmoid target."""
        total = float(drive) + 0.30 * float(recurrent)
        z = max(-30.0, min(30.0, 4.0 * (total - 0.5)))
        target = 1.0 / (1.0 + math.exp(-z))
        self.activation = _bounded(0.65 * self.activation + 0.35 * target)
        self.history.append(self.activation)
        return self.activation


class GardenPathNetwork:
    """Model broad frontotemporal, working-memory, cue, and control interactions."""

    POPULATIONS = (
        "auditory_visual_input",
        "lexical_temporal",
        "posterior_temporal_syntax",
        "anterior_temporal_semantics",
        "left_ifg_unification",
        "frontoparietal_control",
        "working_memory",
        "attention",
        "prosody_punctuation",
        "semantic_role_binding",
        "conflict_monitoring",
        "metacognitive_confidence",
        "distributed_learning",
    )

    def __init__(self, enabled: bool = True) -> None:
        """Initialize independent population-rate reference state."""
        self.enabled = enabled
        self.populations = {name: FunctionalPopulation(name) for name in self.POPULATIONS}
        self.operation_count = 0
        self.last_operation_id: str | None = None

    def advance(
        self,
        operation_id: str,
        drives: Mapping[str, float],
        *,
        conflict: float = 0.0,
        memory_load: float = 0.0,
        cue_strength: float = 0.0,
    ) -> dict:
        """Update populations using recurrent signals and return a transparent activity snapshot."""
        if not self.enabled:
            return {"model": "disabled", "operation_id": operation_id}
        previous = {name: item.activation for name, item in self.populations.items()}
        activity: dict[str, float] = {}
        for name, population in self.populations.items():
            drive = float(drives.get(name, 0.0))
            if name == "conflict_monitoring":
                drive = max(drive, conflict)
            elif name == "working_memory":
                drive = max(drive, memory_load)
            elif name == "prosody_punctuation":
                drive = max(drive, cue_strength)
            recurrent = 0.0
            if name in {"left_ifg_unification", "frontoparietal_control"}:
                recurrent = previous["conflict_monitoring"]
            elif name in {"posterior_temporal_syntax", "anterior_temporal_semantics"}:
                recurrent = previous["working_memory"]
            elif name == "semantic_role_binding":
                recurrent = previous["anterior_temporal_semantics"]
            elif name == "metacognitive_confidence":
                recurrent = previous["conflict_monitoring"]
            activity[name] = population.update(drive, recurrent)
        self.operation_count += 1
        self.last_operation_id = operation_id
        return {
            "model": "distributed_frontotemporal_population_rate_reference",
            "operation_id": operation_id,
            "population_activity": activity,
            "mean_activity": sum(activity.values()) / len(activity),
            "conflict_drive": _bounded(conflict),
            "working_memory_load": _bounded(memory_load),
            "cue_strength": _bounded(cue_strength),
            "scientific_limit": "Coarse functional reference only; not measured or anatomically fitted neural activity.",
        }

    def snapshot(self) -> dict:
        """Return final population activity and explicit scientific limitations."""
        return {
            "model": "distributed_frontotemporal_population_rate_reference" if self.enabled else "disabled",
            "operation_count": self.operation_count,
            "last_operation_id": self.last_operation_id,
            "populations": {
                name: {"activation": row.activation, "samples": len(row.history)}
                for name, row in self.populations.items()
            },
            "scientific_limitations": [
                "Population values are illustrative engineering parameters.",
                "The model is not a literal neuronal or anatomical simulation.",
                "No single named brain region is claimed to implement a complete parsing or reanalysis operation.",
                "Contributions of language-selective and domain-general systems vary by construction and task.",
            ],
        }
