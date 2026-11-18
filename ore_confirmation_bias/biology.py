"""Distributed rate-coded biological-reference network for Point 21."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

from .contracts import Point21Config


def _sigmoid(value: float) -> float:
    """Numerically stable logistic activation for the reference populations."""
    value = max(-40.0, min(40.0, float(value)))
    return 1.0 / (1.0 + math.exp(-value))


@dataclass
class FunctionalPopulation:
    """A coarse activity variable representing a distributed functional population."""

    name: str
    rate: float = 0.0

    def update(self, drive: float, recurrent: float, inhibition: float, config: Point21Config) -> float:
        """Apply leaky recurrent excitation and inhibition to one normalized population rate."""
        target = _sigmoid(2.0 * drive + config.population_recurrent_gain * recurrent
                          - config.population_inhibition_gain * inhibition - 1.0)
        self.rate = max(0.0, min(1.0, (1.0 - config.population_decay) * self.rate
                              + config.population_decay * target))
        return self.rate


class ConfirmationBiasNetwork:
    """Simulate interacting control, attention, evidence, salience, and memory functions.

    This is a transparent population-rate software reference. It is not a fitted neural circuit,
    a measurement of a person's brain, or a claim that one anatomical region performs one step.
    """

    POPULATIONS = (
        "prefrontal_prior_expectation",
        "prefrontal_control_monitoring",
        "frontoparietal_attention",
        "parietal_evidence_readout",
        "hippocampal_context_binding",
        "temporal_semantic_associations",
        "salience_valuation",
        "metacognitive_conflict_monitoring",
    )

    def __init__(self, config: Point21Config | None = None) -> None:
        """Create low-baseline populations and an operation-level activity history."""
        self.config = config or Point21Config()
        self.populations = {name: FunctionalPopulation(name) for name in self.POPULATIONS}
        self.history: list[dict[str, Any]] = []

    def advance(
        self,
        operation_id: str,
        drives: Mapping[str, float],
        *,
        salience: float = 0.0,
        conflict: float = 0.0,
    ) -> dict[str, Any]:
        """Advance the recurrent network once and return rates for this operation."""
        unknown = set(drives) - set(self.populations)
        if unknown:
            raise ValueError(f"unknown functional populations: {sorted(unknown)}")
        salience = max(0.0, min(1.0, float(salience)))
        conflict = max(0.0, min(1.0, float(conflict)))
        old = {name: item.rate for name, item in self.populations.items()}
        mean_activity = sum(old.values()) / max(1, len(old))
        updated: dict[str, float] = {}
        for name, population in self.populations.items():
            other = [value for key, value in old.items() if key != name]
            recurrent = sum(other) / max(1, len(other))
            inhibition = max(0.0, mean_activity - old[name])
            drive = float(drives.get(name, 0.0))
            if name == "salience_valuation":
                drive = max(drive, salience)
            if name == "metacognitive_conflict_monitoring":
                drive = max(drive, conflict)
            updated[name] = population.update(drive, recurrent, inhibition, self.config)
        row = {
            "operation_id": operation_id,
            "population_rates": updated,
            "mean_activity": sum(updated.values()) / max(1, len(updated)),
            "salience_input": salience,
            "conflict_input": conflict,
            "interpretation": "simulated functional reference rates; not neural measurements",
        }
        self.history.append(row)
        return row

    def snapshot(self) -> dict[str, Any]:
        """Return the current population rates and explicit scientific limitations."""
        return {
            "model": "distributed_recurrent_population_rate_reference",
            "operation_count": len(self.history),
            "population_rates": {name: item.rate for name, item in self.populations.items()},
            "history": [dict(item) for item in self.history],
            "scientific_limitations": [
                "Rates are computational proxies, not measured biological signals.",
                "Population labels denote broad functions, not one-region/one-operation localization.",
                "Parameters are illustrative and require empirical calibration.",
                "The network does not establish whether a human belief is true or biased.",
            ],
        }
