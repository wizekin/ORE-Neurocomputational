"""Distributed rate-coded biological-reference network for the HACOL layer.

The biological reference is deliberately honest: hardware-aware numerical
optimization is an engineering layer, and its "biological motivation" maps to
well-described resource constraints in neural systems -- metabolic energy
budgets, efficient coding, divisive normalization, working-memory limits, and
compressed representations. Quantum records are explicitly not biologically
grounded. This is a transparent population-rate software reference; it is not
a fitted neural circuit, a measurement of a brain, or a claim that one
anatomical region performs one step.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping


def _sigmoid(value: float) -> float:
    """Numerically stable logistic activation for the reference populations."""
    value = max(-40.0, min(40.0, float(value)))
    return 1.0 / (1.0 + math.exp(-value))


@dataclass
class FunctionalPopulation:
    """A coarse activity variable representing a distributed functional population."""

    name: str
    rate: float = 0.0

    def update(self, drive: float, recurrent: float, inhibition: float) -> float:
        """Leaky recurrent excitation and inhibition on one normalized rate."""
        target = _sigmoid(2.0 * drive + 0.5 * recurrent - 1.0 * inhibition - 1.0)
        self.rate = max(0.0, min(1.0, 0.7 * self.rate + 0.3 * target))
        return self.rate


#: Phase-level biological-reference annotations for the HACOL layers.
BIOLOGY_REGISTRY: tuple[dict[str, str], ...] = (
    {"phase": "A", "layer": "workload discovery",
     "biological_reference": "Attention and cognitive-control systems signal which representations consume resources; salience and error signals flag effortful processing before optimization.",
     "evidence_level": "distributed_function_description_with_engineering_abstraction"},
    {"phase": "B", "layer": "numerical contract and admission",
     "biological_reference": "Metabolic and energetic constraints (cortical energy budgets) bound neural computation; homeostatic and arousal systems enforce global resource allocation rather than local shortcuts.",
     "evidence_level": "distributed_function_description_with_engineering_abstraction"},
    {"phase": "C", "layer": "IO-aware attention",
     "biological_reference": "Cortical processing is streaming and memory-limited: sequential evidence integration with divisive normalization avoids materializing complete association matrices; normalized population responses keep bounded dynamic range.",
     "evidence_level": "normalized_computation_principle_with_engineering_abstraction"},
    {"phase": "D", "layer": "randomized linear algebra",
     "biological_reference": "Compressed and low-dimensional neural codes (sparse and reduced representations in sensory cortex) trade exactness for efficiency under resource constraints.",
     "evidence_level": "efficient_coding_principle_with_engineering_abstraction"},
    {"phase": "E", "layer": "structured transforms and mixed precision",
     "biological_reference": "Neural systems exploit statistical structure of inputs and assign differential reliability to representations: coarse rate codes versus precisely timed codes echo per-operation precision.",
     "evidence_level": "multi_time_scale_coding_principle_with_engineering_abstraction"},
    {"phase": "F", "layer": "classical optimization and quantum gate",
     "biological_reference": "None: QUBO/Ising settling resembles attractor dynamics, but quantum algorithms are not biologically grounded and are kept as research records only.",
     "evidence_level": "not_biologically_grounded_research_records"},
    {"phase": "G", "layer": "verification and lifecycle",
     "biological_reference": "Metacognitive monitoring and confidence signals verify outputs before commitment; prediction-error signals trigger relearning or fallback to reliable paths.",
     "evidence_level": "metacognitive_monitoring_principle_with_engineering_abstraction"},
)


class HACOLBiologyNetwork:
    """Simulate interacting resource-control populations for the HACOL phases.

    Populations represent distributed functions (allocation control, salience,
    normalization, verification monitoring) as normalized rates. The model is
    a transparent engineering reference for tracing phase activity, not a
    validated neural simulator.
    """

    POPULATIONS = (
        "frontoparietal_allocation_control",
        "salience_effort_signal",
        "normalized_response_range",
        "compressed_representation",
        "metacognitive_verification",
    )

    def __init__(self) -> None:
        self.populations = {name: FunctionalPopulation(name) for name in self.POPULATIONS}
        self.history: list[dict[str, Any]] = []

    def advance(
        self,
        phase: str,
        drives: Mapping[str, float],
        *,
        salience: float = 0.0,
        conflict: float = 0.0,
    ) -> dict[str, Any]:
        """Advance the reference network once for one phase and return rates."""
        unknown = set(drives) - set(self.populations)
        if unknown:
            raise ValueError(f"unknown functional populations: {sorted(unknown)}")
        salience = max(0.0, min(1.0, float(salience)))
        conflict = max(0.0, min(1.0, float(conflict)))
        old = {name: item.rate for name, item in self.populations.items()}
        mean_activity = sum(old.values()) / max(1, len(old))
        updated: dict[str, float] = {}
        for name, population in self.populations.items():
            others = [value for key, value in old.items() if key != name]
            recurrent = sum(others) / max(1, len(others))
            inhibition = max(0.0, mean_activity - old[name])
            drive = float(drives.get(name, 0.0))
            if name == "salience_effort_signal":
                drive = max(drive, salience)
            if name == "metacognitive_verification":
                drive = max(drive, conflict)
            updated[name] = population.update(drive, recurrent, inhibition)
        record = {
            "phase": phase,
            "populations": dict(updated),
            "salience": salience,
            "conflict": conflict,
        }
        self.history.append(record)
        return record

    def snapshot(self) -> dict[str, Any]:
        """Return detached population rates and the phase history."""
        return {
            "model": "hacol_biological_reference_network",
            "populations": {name: item.rate for name, item in self.populations.items()},
            "phase_count": len(self.history),
            "scientific_status": "computational synthesis; not a validated neural simulator",
            "quantum_layers_biologically_grounded": False,
        }


PHASE_DRIVES: dict[str, Mapping[str, float]] = {
    "A": {"frontoparietal_allocation_control": 0.6, "salience_effort_signal": 0.5},
    "B": {"frontoparietal_allocation_control": 0.8, "metacognitive_verification": 0.4},
    "C": {"normalized_response_range": 0.8, "frontoparietal_allocation_control": 0.5},
    "D": {"compressed_representation": 0.8, "frontoparietal_allocation_control": 0.4},
    "E": {"normalized_response_range": 0.6, "compressed_representation": 0.5,
          "metacognitive_verification": 0.5},
    "F": {"metacognitive_verification": 0.7, "frontoparietal_allocation_control": 0.3},
    "G": {"metacognitive_verification": 0.9, "frontoparietal_allocation_control": 0.5},
}


def biology_reference_for_phase(phase: str) -> dict[str, str]:
    """Return the biological-reference annotation for one phase letter."""
    for row in BIOLOGY_REGISTRY:
        if row["phase"] == phase:
            return row
    raise KeyError(f"no biological reference registered for phase {phase!r}")
