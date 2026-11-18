"""Illustrative distributed population-rate reference for Point 24."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping


def _bounded(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass
class Population:
    """A coarse activation variable, not a simulation of individual neurons."""

    name: str
    activation: float = 0.0
    history: list[float] = field(default_factory=list)

    def update(self, drive: float, recurrent: float = 0.0) -> float:
        total = float(drive) + 0.25 * float(recurrent)
        z = max(-30.0, min(30.0, 4.0 * (total - 0.5)))
        target = 1.0 / (1.0 + math.exp(-z))
        self.activation = _bounded(0.70 * self.activation + 0.30 * target)
        self.history.append(self.activation)
        return self.activation


class CorrectiveLearningNetwork:
    """A transparent systems-level reference with recurrent population interactions."""

    POPULATIONS = (
        "sensory_association",
        "frontotemporal_language",
        "medial_frontal_monitoring",
        "prefrontal_control",
        "hippocampal_retrieval",
        "locus_coeruleus_noradrenergic",
        "dopaminergic_feedback_value",
        "semantic_relational",
        "metacognitive_monitoring",
        "distributed_consolidation",
        "cerebellar_calibration",
    )

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.populations = {name: Population(name) for name in self.POPULATIONS}
        self.operation_count = 0

    def advance(
        self,
        operation_id: str,
        drives: Mapping[str, float],
        *,
        mismatch: float = 0.0,
        memory_load: float = 0.0,
        feedback_quality: float = 0.0,
    ) -> dict:
        """Advance all populations with leaky bounded activity and simple recurrent links."""
        if not self.enabled:
            return {"model": "disabled", "operation_id": operation_id}
        previous = {name: unit.activation for name, unit in self.populations.items()}
        activity: dict[str, float] = {}
        for name, unit in self.populations.items():
            drive = float(drives.get(name, 0.0))
            if name == "medial_frontal_monitoring":
                drive = max(drive, mismatch)
            elif name in {"prefrontal_control", "hippocampal_retrieval"}:
                drive = max(drive, memory_load)
            elif name in {"dopaminergic_feedback_value", "locus_coeruleus_noradrenergic"}:
                drive = max(drive, feedback_quality)
            recurrent = 0.0
            if name == "prefrontal_control":
                recurrent = previous["medial_frontal_monitoring"]
            elif name == "hippocampal_retrieval":
                recurrent = previous["prefrontal_control"]
            elif name == "semantic_relational":
                recurrent = previous["frontotemporal_language"]
            elif name == "metacognitive_monitoring":
                recurrent = previous["medial_frontal_monitoring"]
            elif name == "distributed_consolidation":
                recurrent = previous["hippocampal_retrieval"]
            elif name == "cerebellar_calibration":
                recurrent = previous["dopaminergic_feedback_value"]
            activity[name] = unit.update(drive, recurrent)
        self.operation_count += 1
        return {
            "model": "distributed_corrective_learning_population_rate_reference",
            "operation_id": operation_id,
            "activity": activity,
            "mismatch_drive": _bounded(mismatch),
            "feedback_quality_drive": _bounded(feedback_quality),
            "scientific_scope": "illustrative systems-level activity; not measured neural data",
        }

    def snapshot(self) -> dict:
        if not self.enabled:
            return {"model": "disabled", "operation_count": 0}
        return {
            "model": "distributed_corrective_learning_population_rate_reference",
            "operation_count": self.operation_count,
            "populations": {
                name: {
                    "activation": unit.activation,
                    "history": list(unit.history),
                    "history_length": len(unit.history),
                }
                for name, unit in self.populations.items()
            },
            "limitations": [
                "Populations are coarse abstractions, not anatomical compartments or neuron simulations.",
                "Regional contributions are distributed, overlapping, and task-dependent.",
                "Prediction error, reward prediction error, and semantic correction are not interchangeable.",
                "This model has not been fitted to human neural recordings.",
            ],
        }
