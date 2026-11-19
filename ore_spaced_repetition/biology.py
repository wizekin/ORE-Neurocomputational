"""Biologically inspired network surrogates for Point 03 spaced learning."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .functional import clamp


def _vectorize(texts: Sequence[str], dimensions: int) -> np.ndarray:
    """Encode text fragments as deterministic signed-hash population input."""
    vector = np.zeros(dimensions, dtype=float)
    for text in texts:
        for token in str(text).casefold().split():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            number = int.from_bytes(digest, "big")
            vector[number % dimensions] += 1.0 if ((number >> 8) & 1) else -1.0
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else vector


@dataclass
class NeuralPopulation:
    """Maintain bounded recurrent population activity with gain and inhibitory modulation."""

    name: str
    dimensions: int = 64
    recurrent_scale: float = 0.12
    seed: int = 31
    activity: np.ndarray = field(init=False, repr=False)
    weights: np.ndarray = field(init=False, repr=False)
    update_count: int = 0

    def __post_init__(self) -> None:
        """Initialize a reproducible recurrent matrix and neutral population activity."""
        if self.dimensions < 8 or not math.isfinite(self.recurrent_scale) or self.recurrent_scale < 0:
            raise ValueError("invalid population dimensions or recurrent scale")
        rng = np.random.default_rng(self.seed)
        self.weights = rng.normal(0.0, 1.0 / math.sqrt(self.dimensions),
                                  (self.dimensions, self.dimensions)) * self.recurrent_scale
        self.activity = np.zeros(self.dimensions, dtype=float)

    def update(self, incoming: Sequence[float], gain: float = 1.0,
               inhibition: float = 0.0) -> np.ndarray:
        """Advance recurrent activity through a bounded nonlinear population update."""
        vector = np.asarray(incoming, dtype=float)
        if vector.shape != (self.dimensions,) or not np.all(np.isfinite(vector)):
            raise ValueError("incoming must be a finite vector matching population dimensions")
        if not math.isfinite(gain) or gain < 0 or not math.isfinite(inhibition):
            raise ValueError("gain and inhibition must be finite; gain must be non-negative")
        drive = gain * vector + self.weights @ self.activity - inhibition
        self.activity = np.tanh(np.clip(drive, -20.0, 20.0))
        self.update_count += 1
        return self.activity.copy()

    def snapshot(self) -> dict[str, Any]:
        """Return compact population diagnostics without exposing internal matrices."""
        return {"name": self.name, "dimensions": self.dimensions,
                "update_count": self.update_count,
                "mean_activity": float(np.mean(self.activity)),
                "activity_energy": float(np.mean(self.activity ** 2)),
                "model": "recurrent_population_surrogate"}


class HippocampalReactivationNetwork:
    """Bind item patterns and model cue-driven partial reactivation across learning episodes."""

    def __init__(self, dimensions: int = 64) -> None:
        """Initialize a pattern store for item representations and event context."""
        self.dimensions = dimensions
        self.patterns: dict[str, np.ndarray] = {}
        self.contexts: dict[str, dict[str, Any]] = {}
        self.reactivation_count = 0

    def bind(self, item_id: str, content: Sequence[str],
             context: Mapping[str, Any]) -> dict[str, Any]:
        """Bind a deterministic item pattern and its learning context."""
        if item_id in self.patterns:
            raise ValueError("item_id already bound")
        vector = _vectorize(content, self.dimensions)
        self.patterns[item_id] = vector
        self.contexts[item_id] = dict(context)
        return {"item_id": item_id, "bound": True,
                "context_keys": sorted(str(key) for key in context),
                "model": "hippocampal_item_binding_surrogate"}

    def reactivate(self, item_id: str, cue_text: str) -> dict[str, Any]:
        """Estimate cue-based overlap with a stored pattern as a partial-reinstatement proxy."""
        if item_id not in self.patterns:
            raise KeyError(item_id)
        cue = _vectorize([cue_text], self.dimensions)
        target = self.patterns[item_id]
        denominator = float(np.linalg.norm(cue) * np.linalg.norm(target))
        similarity = float(np.dot(cue, target) / denominator) if denominator > 1e-12 else 0.0
        self.reactivation_count += 1
        return {"item_id": item_id, "cue_pattern_overlap": clamp((similarity + 1.0) / 2.0),
                "reactivation_count": self.reactivation_count,
                "context_available": bool(self.contexts[item_id]),
                "model": "cue_dependent_pattern_reactivation_surrogate"}

    def snapshot(self) -> dict[str, Any]:
        """Report bound item patterns and modeled reactivation count."""
        return {"item_pattern_count": len(self.patterns),
                "reactivation_count": self.reactivation_count,
                "model": "hippocampal_reactivation_surrogate"}


class CorticalSpacedPlasticityNetwork:
    """Represent repeated activity-dependent association updates with spacing-sensitive model terms."""

    def __init__(self, dimensions: int = 64, learning_rate: float = 0.08) -> None:
        """Initialize bounded association weights for a distributed cortical surrogate."""
        if dimensions < 8 or not 0 <= learning_rate <= 1:
            raise ValueError("invalid dimensions or learning_rate")
        self.dimensions = dimensions
        self.learning_rate = float(learning_rate)
        self.weights = np.zeros((dimensions, dimensions), dtype=float)
        self.update_count = 0
        self.last_spacing_factor = 0.0

    def update(self, pre_activity: Sequence[float], post_activity: Sequence[float],
               interval_hours: float, gate: float = 1.0) -> dict[str, Any]:
        """Apply a bounded co-activity update modulated by the time between learning episodes."""
        pre = np.asarray(pre_activity, dtype=float)
        post = np.asarray(post_activity, dtype=float)
        if pre.shape != (self.dimensions,) or post.shape != (self.dimensions,):
            raise ValueError("activity vectors must match plasticity dimensions")
        if not np.all(np.isfinite(pre)) or not np.all(np.isfinite(post)):
            raise ValueError("activity vectors must be finite")
        if not math.isfinite(interval_hours) or interval_hours < 0:
            raise ValueError("interval_hours must be finite and non-negative")
        gate = clamp(gate)
        spacing_factor = 1.0 - math.exp(-interval_hours / 24.0)
        delta = self.learning_rate * gate * spacing_factor * np.outer(pre, post)
        self.weights = np.clip(self.weights + delta, -1.0, 1.0)
        self.update_count += 1
        self.last_spacing_factor = spacing_factor
        return {"update_count": self.update_count, "spacing_factor": spacing_factor,
                "update_norm": float(np.linalg.norm(delta)),
                "weights_bounded": bool(np.all(np.abs(self.weights) <= 1.0)),
                "model": "spacing_modulated_hebbian_surrogate"}

    def snapshot(self) -> dict[str, Any]:
        """Expose plasticity magnitudes and bounded-weight diagnostics."""
        return {"update_count": self.update_count,
                "mean_absolute_weight": float(np.mean(np.abs(self.weights))),
                "maximum_absolute_weight": float(np.max(np.abs(self.weights))),
                "last_spacing_factor": self.last_spacing_factor,
                "weights_bounded": bool(np.all(np.abs(self.weights) <= 1.0)),
                "model": "cortical_spaced_plasticity_surrogate"}


class PrefrontalRetrievalControlNetwork:
    """Maintain a retrieval goal and model cue support, effort monitoring, and adaptive control."""

    def __init__(self, dimensions: int = 64) -> None:
        """Initialize a recurrent control population and empty task goal."""
        self.population = NeuralPopulation("prefrontal_retrieval_control", dimensions,
                                           recurrent_scale=0.16, seed=607)
        self.goal_vector = np.zeros(dimensions, dtype=float)
        self.control_count = 0

    def set_goal(self, goal: str) -> dict[str, Any]:
        """Maintain a task goal in a distributed activity pattern."""
        self.goal_vector = _vectorize([goal], self.population.dimensions)
        activity = self.population.update(self.goal_vector)
        self.control_count += 1
        return {"goal_activity_energy": float(np.mean(activity ** 2)),
                "goal_maintained": True}

    def control_retrieval(self, cue: str, predicted_difficulty: float) -> dict[str, Any]:
        """Modulate cue support and control gain as predicted retrieval difficulty rises."""
        difficulty = clamp(predicted_difficulty)
        cue_vector = _vectorize([cue], self.population.dimensions)
        drive = 0.65 * cue_vector + 0.35 * self.goal_vector
        activity = self.population.update(drive, gain=1.0 + 0.4 * difficulty,
                                           inhibition=0.08 * difficulty)
        self.control_count += 1
        return {"cue_support_level": "supportive" if difficulty > 0.65 else "minimal",
                "control_gain": clamp(0.7 + 0.3 * difficulty),
                "control_activity_energy": float(np.mean(activity ** 2)),
                "control_count": self.control_count}

    def snapshot(self) -> dict[str, Any]:
        """Return the prefrontal control-population diagnostics."""
        return {**self.population.snapshot(), "control_count": self.control_count}


class OfflineConsolidationNetwork:
    """Track optional offline context without assuming a deterministic sleep-related gain."""

    def __init__(self) -> None:
        """Initialize an evidence-gated offline-event counter."""
        self.observed_event_count = 0
        self.last_context: dict[str, Any] = {}

    def observe(self, context: Mapping[str, Any]) -> dict[str, Any]:
        """Record the supplied offline context and leave unobserved periods unchanged."""
        observed = bool(context.get("observed", False))
        if observed:
            self.observed_event_count += 1
        self.last_context = dict(context)
        return {"observed": observed,
                "observed_event_count": self.observed_event_count,
                "automatic_memory_boost_applied": False,
                "model": "offline_consolidation_context_monitor"}


class NeuromodulatoryLearningGate:
    """Represent a bounded task-state gate based on effort, outcome quality, and novelty."""

    def __init__(self) -> None:
        """Initialize a neutral learning gate and update counter."""
        self.last_gate = 0.0
        self.update_count = 0

    def compute(self, quality: float, effort: float, novelty: float) -> dict[str, float]:
        """Compute an illustrative gate without claiming a specific transmitter concentration."""
        gate = clamp(0.50 * clamp(quality) + 0.30 * clamp(effort) + 0.20 * clamp(novelty))
        self.last_gate = gate
        self.update_count += 1
        return {"learning_gate": gate, "quality_signal": clamp(quality),
                "effort_signal": clamp(effort), "novelty_signal": clamp(novelty)}

    def snapshot(self) -> dict[str, Any]:
        """Return gate activity and an explicit model-level interpretation."""
        return {"last_gate": self.last_gate, "update_count": self.update_count,
                "model": "neuromodulatory_learning_gate_surrogate"}


class BiologicalSpacedRepetitionNetwork:
    """Coordinate hippocampal reactivation, cortical plasticity, prefrontal control, and offline context."""

    def __init__(self, dimensions: int = 64, learning_rate: float = 0.08) -> None:
        """Initialize interacting network surrogates with matching representation dimensions."""
        self.dimensions = dimensions
        self.population = NeuralPopulation("distributed_memory_population", dimensions, seed=503)
        self.hippocampus = HippocampalReactivationNetwork(dimensions)
        self.cortex = CorticalSpacedPlasticityNetwork(dimensions, learning_rate)
        self.prefrontal = PrefrontalRetrievalControlNetwork(dimensions)
        self.offline = OfflineConsolidationNetwork()
        self.learning_gate = NeuromodulatoryLearningGate()

    def encode(self, content: Sequence[str], context: Mapping[str, Any]) -> dict[str, Any]:
        """Encode the item in a distributed population and hippocampal-style pattern store."""
        vector = _vectorize(content, self.dimensions)
        activity = self.population.update(vector)
        return {"population_activity_energy": float(np.mean(activity ** 2)),
                "hippocampal_binding": self.hippocampus.bind(
                    str(context["item_id"]), content, context
                )}

    def review(self, item_id: str, cue: str, predicted_difficulty: float,
               outcome_quality: float, effort: float, interval_hours: float,
               previous_activity: Sequence[float]) -> dict[str, Any]:
        """Model cue-driven reactivation, control recruitment, and spacing-sensitive plasticity."""
        reactivation = self.hippocampus.reactivate(item_id, cue)
        control = self.prefrontal.control_retrieval(cue, predicted_difficulty)
        vector = _vectorize([cue], self.dimensions)
        activity = self.population.update(vector, gain=1.0 + predicted_difficulty)
        gate = self.learning_gate.compute(outcome_quality, effort, predicted_difficulty)
        plasticity = self.cortex.update(previous_activity, activity, interval_hours,
                                        gate["learning_gate"])
        return {"reactivation": reactivation, "retrieval_control": control,
                "learning_gate": gate, "plasticity": plasticity}

    def snapshot(self) -> dict[str, Any]:
        """Return all network diagnostics with a clear computational-surrogate boundary."""
        return {"distributed_population": self.population.snapshot(),
                "hippocampus": self.hippocampus.snapshot(),
                "cortex": self.cortex.snapshot(),
                "prefrontal": self.prefrontal.snapshot(),
                "offline_consolidation": self.offline.observe({"observed": False}),
                "learning_gate": self.learning_gate.snapshot(),
                "literal_neuron_simulation": False,
                "biological_fidelity": "functional computational surrogate"}
