"""Biological-reference track for the Point 13 continuous-time and neuromorphic families.

This module provides cautious, explicitly labelled biological correspondence notes for
each candidate family (LNN, CfC, Neural ODE, selective SSM, SNN) and a small
membrane-potential population simulator used as the biological-reference trace.  It is
an engineering abstraction for side-by-side comparison, not a claim that ORE reproduces
any particular neural system.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .functional import lif_advance

__all__ = (
    "FAMILY_BIOLOGICAL_REFERENCES",
    "MembranePopulation",
    "family_biological_reference",
)


FAMILY_BIOLOGICAL_REFERENCES: dict[str, str] = {
    "baseline": (
        "Deterministic statistical baselines correspond to engineering control conditions; "
        "they carry no biological claim and exist to make every model benefit measurable."
    ),
    "lnn": (
        "Liquid neural networks echo continuously evolving recurrent circuit dynamics in "
        "cortex, where hidden state evolves through differential equations with "
        "time constants rather than discrete clocked updates."
    ),
    "cfc": (
        "Closed-form continuous-time updates approximate membrane leak toward a driving "
        "target with an exponential time constant, echoing the effective time constants "
        "of synaptic and membrane integration."
    ),
    "neural_ode": (
        "Neural ordinary differential equations learn a vector field, echoing the idea that "
        "cortical population dynamics can be approximated by smooth differential systems "
        "in low-dimensional state spaces."
    ),
    "ssm": (
        "Selective state-space models maintain compact recurrent memory with "
        "input-dependent gating, echoing gated recurrent processing in cortical "
        "microcircuits that carry context across long sequences."
    ),
    "snn": (
        "Spiking neural networks use leaky integrate-and-fire membranes, thresholds, "
        "resets, and refractory periods, matching the event-driven signalling vocabulary "
        "of biological neurons at a functional level of description."
    ),
}


def family_biological_reference(family: str) -> str:
    """Return the cautious biological correspondence note for a model family."""
    return FAMILY_BIOLOGICAL_REFERENCES.get(family, (
        "No biological correspondence has been documented for this family."
    ))


class MembranePopulation:
    """A small LIF population used as the biological-reference trace for temporal runs."""

    def __init__(self, n_neurons: int = 16, tau_m: float = 0.02, v_rest: float = 0.0,
                 v_threshold: float = 1.0, v_reset: float = 0.0, resistance: float = 1.0,
                 refractory_period: float = 0.002) -> None:
        """Initialize deterministic membrane state."""
        if n_neurons <= 0:
            raise ValueError("n_neurons must be positive")
        self.n_neurons = int(n_neurons)
        self.tau_m = float(tau_m)
        self.v_rest = float(v_rest)
        self.v_threshold = float(v_threshold)
        self.v_reset = float(v_reset)
        self.resistance = float(resistance)
        self.refractory_period = float(refractory_period)
        self.reset()

    def reset(self) -> None:
        """Restore the population to its resting deterministic state."""
        self._voltage = np.zeros(self.n_neurons, dtype=float)
        self._refractory = np.zeros(self.n_neurons, dtype=float)
        self._weights = np.random.default_rng(13).normal(
            scale=1.0 / np.sqrt(self.n_neurons), size=(self.n_neurons, self.n_neurons))
        self.total_spikes = 0
        self.steps = 0

    def advance(self, drive: np.ndarray, dt: float) -> dict[str, Any]:
        """Advance the population by dt under an external drive current."""
        drive_vector = np.asarray(drive, dtype=float).reshape(-1)
        if drive_vector.size == 0:
            raise ValueError("drive must not be empty")
        if drive_vector.size != self.n_neurons:
            repeats = int(np.ceil(self.n_neurons / drive_vector.size))
            drive_vector = np.tile(drive_vector, repeats)[: self.n_neurons]
        current = self._weights @ drive_vector + drive_vector
        self._voltage, spikes, self._refractory = lif_advance(
            self._voltage, current, dt, self.tau_m, self.v_rest, self.v_threshold,
            self.v_reset, self.resistance, self.refractory_period, self._refractory)
        self.total_spikes += int(np.count_nonzero(spikes))
        self.steps += 1
        return {
            "spikes": int(np.count_nonzero(spikes)),
            "mean_voltage": float(np.mean(self._voltage)) if self._voltage.size else 0.0,
            "total_spikes": self.total_spikes,
            "steps": self.steps,
            "spike_rate": (self.total_spikes / max(self.steps * self.n_neurons, 1)),
            "biological_scope": (
                "population-rate LIF engineering abstraction; "
                "not a validated simulation of any biological system"
            ),
        }

    def state_summary(self) -> Mapping[str, Any]:
        """Return a compact membrane-state summary for diagnostics."""
        return {
            "n_neurons": self.n_neurons,
            "total_spikes": self.total_spikes,
            "mean_voltage": float(np.mean(self._voltage)) if self._voltage.size else 0.0,
            "max_voltage": float(np.max(self._voltage)) if self._voltage.size else 0.0,
            "steps": self.steps,
        }
