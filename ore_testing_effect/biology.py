"""Functional biological surrogates for Point 04 testing-effect and retrieval-practice processes."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
import re
from typing import Any, Mapping, Sequence

import numpy as np

from .functional import clamp, cosine_similarity, softmax


def _sigmoid_vector(values: np.ndarray) -> np.ndarray:
    """Apply a stable logistic activation to a vector of modeled neural activities."""
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


def vectorize_text(texts: Sequence[str], dimensions: int = 64) -> np.ndarray:
    """Encode text fragments as a deterministic signed-hash population input vector."""
    if dimensions < 4:
        raise ValueError("dimensions must be at least four")
    vector = np.zeros(dimensions, dtype=float)
    for text in texts:
        for token in re.findall(r"[a-zA-Z0-9]+", str(text).casefold()):
            if len(token) < 2:
                continue
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            number = int.from_bytes(digest, "big")
            vector[number % dimensions] += 1.0 if ((number >> 9) & 1) else -1.0
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else vector


@dataclass
class NeuralPopulation:
    """Maintain bounded recurrent population activity with excitation and global inhibition."""

    name: str
    dimensions: int = 64
    recurrent_scale: float = 0.12
    seed: int = 41
    activity: np.ndarray = field(init=False, repr=False)
    weights: np.ndarray = field(init=False, repr=False)
    update_count: int = 0

    def __post_init__(self) -> None:
        """Initialize a reproducible recurrent matrix and a neutral activity vector."""
        if self.dimensions < 4 or self.recurrent_scale < 0:
            raise ValueError("dimensions must be >= 4 and recurrent_scale non-negative")
        rng = np.random.default_rng(self.seed)
        raw = rng.normal(0.0, 1.0 / math.sqrt(self.dimensions),
                        size=(self.dimensions, self.dimensions))
        self.weights = raw * self.recurrent_scale
        np.fill_diagonal(self.weights, 0.0)
        self.activity = np.zeros(self.dimensions, dtype=float)

    def update(self, input_vector: Sequence[float], gain: float = 1.0,
               inhibition: float = 0.12) -> np.ndarray:
        """Integrate current input with recurrent activity and inhibitory competition."""
        incoming = np.asarray(input_vector, dtype=float)
        if incoming.shape != (self.dimensions,):
            raise ValueError("input_vector has an incompatible dimension")
        if not np.all(np.isfinite(incoming)):
            raise ValueError("input_vector must be finite")
        recurrent = self.weights @ self.activity
        global_inhibition = float(inhibition) * float(np.mean(self.activity))
        self.activity = _sigmoid_vector(float(gain) * incoming + recurrent - global_inhibition)
        self.update_count += 1
        return self.activity.copy()

    def snapshot(self) -> dict[str, Any]:
        """Return bounded population diagnostics without exposing mutable arrays."""
        return {"name": self.name, "dimensions": self.dimensions,
                "mean_activity": float(np.mean(self.activity)),
                "activity_energy": float(np.mean(self.activity ** 2)),
                "maximum_activity": float(np.max(self.activity)),
                "update_count": self.update_count,
                "model": "recurrent_rate_population_surrogate"}


class HippocampalPatternCompletion:
    """Store item patterns and reinstate the closest trace from a partial retrieval cue."""

    def __init__(self, dimensions: int = 64) -> None:
        """Create an empty item-pattern store with a fixed representation dimension."""
        self.dimensions = dimensions
        self.patterns: dict[str, np.ndarray] = {}
        self.metadata: dict[str, dict[str, Any]] = {}
        self.reactivation_counts: dict[str, int] = {}

    def bind(self, item_id: str, content: Sequence[str],
             metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Bind a learning target to a distributed item vector and retain metadata."""
        vector = vectorize_text(list(content), self.dimensions)
        self.patterns[item_id] = vector
        self.metadata[item_id] = dict(metadata or {})
        self.reactivation_counts.setdefault(item_id, 0)
        return {"item_id": item_id, "bound": True,
                "pattern_norm": float(np.linalg.norm(vector)),
                "component_count": len(content),
                "model": "hippocampal_pattern_binding_surrogate"}

    def complete(self, cue: str, item_id: str | None = None) -> dict[str, Any]:
        """Use cue-to-pattern similarity to estimate partial pattern completion."""
        if not self.patterns:
            return {"retrieved": False, "reason": "no patterns have been bound",
                    "best_match": None, "similarity": 0.0}
        cue_vector = vectorize_text([cue], self.dimensions)
        candidates = self.patterns.items()
        similarities = {key: cosine_similarity(cue_vector, vector)
                        for key, vector in candidates}
        selected = item_id if item_id in similarities else max(similarities, key=similarities.get)
        similarity = similarities[selected]
        self.reactivation_counts[selected] = self.reactivation_counts.get(selected, 0) + 1
        reinstated = self.patterns[selected] * clamp(max(0.0, similarity))
        return {"retrieved": similarity > 0.0, "best_match": selected,
                "target_match": selected == item_id if item_id else None,
                "similarity": similarity, "candidate_similarities": similarities,
                "completion_energy": float(np.mean(reinstated ** 2)),
                "reactivation_count": self.reactivation_counts[selected],
                "model": "hippocampal_pattern_completion_surrogate"}

    def snapshot(self) -> dict[str, Any]:
        """Return item counts and reactivation totals for the pattern-completion model."""
        return {"stored_patterns": len(self.patterns),
                "total_reactivations": sum(self.reactivation_counts.values()),
                "reactivation_counts": dict(self.reactivation_counts),
                "literal_hippocampal_simulation": False}


class PrefrontalRetrievalControl:
    """Represent goal maintenance, cue biasing, and suppression of irrelevant candidates."""

    def __init__(self) -> None:
        """Initialize a neutral task set and retrieval-control diagnostics."""
        self.last_focus = 0.0
        self.last_inhibition = 0.0
        self.update_count = 0

    def configure(self, cue_relevance: float, task_difficulty: float,
                  fatigue: float = 0.0) -> dict[str, float]:
        """Compute illustrative goal focus and inhibition levels from demand and fatigue."""
        relevance = clamp(cue_relevance)
        difficulty = clamp(task_difficulty)
        fatigue_value = clamp(fatigue)
        focus = clamp(0.55 * relevance + 0.30 * difficulty + 0.15 * (1.0 - fatigue_value))
        inhibition = clamp(0.25 + 0.45 * relevance + 0.20 * (1.0 - fatigue_value))
        self.last_focus = focus
        self.last_inhibition = inhibition
        self.update_count += 1
        return {"goal_focus": focus, "irrelevant_competition_inhibition": inhibition,
                "task_difficulty": difficulty, "fatigue": fatigue_value}

    def snapshot(self) -> dict[str, Any]:
        """Return goal-maintenance and inhibition diagnostics."""
        return {"last_focus": self.last_focus, "last_inhibition": self.last_inhibition,
                "update_count": self.update_count,
                "model": "prefrontal_retrieval_control_surrogate"}


class AnteriorCingulateErrorMonitor:
    """Approximate discrepancy, uncertainty, and conflict monitoring across retrieval attempts."""

    def __init__(self) -> None:
        """Initialize the monitoring system with no previous error signal."""
        self.last_error_signal = 0.0
        self.last_conflict = 0.0
        self.update_count = 0

    def monitor(self, predicted_success: float, observed_quality: float,
                candidate_scores: Mapping[str, float] | None = None) -> dict[str, float]:
        """Compute behavioral prediction discrepancy and competition entropy."""
        prediction_error = clamp(predicted_success) - clamp(observed_quality)
        probabilities = softmax(candidate_scores or {"single_candidate": 1.0})
        entropy = -sum(p * math.log(max(p, 1e-12)) for p in probabilities.values())
        max_entropy = math.log(max(2, len(probabilities)))
        conflict = clamp(entropy / max_entropy) if len(probabilities) > 1 else 0.0
        self.last_error_signal = abs(prediction_error)
        self.last_conflict = conflict
        self.update_count += 1
        return {"signed_prediction_error": prediction_error,
                "absolute_error_signal": abs(prediction_error),
                "candidate_conflict": conflict,
                "candidate_entropy": entropy}

    def snapshot(self) -> dict[str, Any]:
        """Return prediction-error and conflict monitoring diagnostics."""
        return {"last_error_signal": self.last_error_signal,
                "last_conflict": self.last_conflict,
                "update_count": self.update_count,
                "model": "anterior_cingulate_monitoring_surrogate"}


class NeuromodulatoryLearningGate:
    """Modulate modeled plasticity using surprise, relevance, effort, and verified feedback."""

    def __init__(self) -> None:
        """Initialize a neutral modulatory gate and reward-prediction estimate."""
        self.value_estimate = 0.5
        self.last_gate = 0.0
        self.update_count = 0

    def compute(self, outcome_quality: float, surprise: float, relevance: float,
                effort: float, feedback_verified: bool) -> dict[str, float]:
        """Compute a bounded learning gate and update a simple reward-prediction error."""
        quality = clamp(outcome_quality)
        expected = self.value_estimate
        prediction_error = quality - expected
        self.value_estimate = clamp(expected + 0.12 * prediction_error)
        verification = 1.0 if feedback_verified else 0.0
        gate = clamp(0.34 * quality + 0.22 * clamp(surprise)
                     + 0.20 * clamp(relevance) + 0.14 * clamp(effort)
                     + 0.10 * verification)
        self.last_gate = gate
        self.update_count += 1
        return {"learning_gate": gate, "outcome_quality": quality,
                "expected_quality_before_update": expected,
                "reward_prediction_error": prediction_error,
                "verified_feedback_signal": verification}

    def snapshot(self) -> dict[str, Any]:
        """Return gate and learned-value diagnostics without labeling them transmitter levels."""
        return {"last_gate": self.last_gate, "value_estimate": self.value_estimate,
                "update_count": self.update_count,
                "literal_neurotransmitter_concentrations": False,
                "model": "neuromodulatory_gate_surrogate"}


class CorticalAssociativePlasticity:
    """Maintain distributed conceptual links with gated Hebbian/Oja-like updates."""

    def __init__(self, dimensions: int = 64, learning_rate: float = 0.04) -> None:
        """Initialize a bounded associative matrix for target and relation representations."""
        if dimensions < 4 or not 0.0 < learning_rate <= 1.0:
            raise ValueError("invalid cortical model dimensions or learning rate")
        self.dimensions = dimensions
        self.learning_rate = learning_rate
        self.weights = np.zeros((dimensions, dimensions), dtype=float)
        self.update_count = 0
        self.last_update_norm = 0.0

    def update(self, pre_activity: Sequence[float], post_activity: Sequence[float],
               gate: float, verified: bool) -> dict[str, Any]:
        """Apply a bounded co-activity update only when the criterion or correction is verified."""
        pre = np.asarray(pre_activity, dtype=float)
        post = np.asarray(post_activity, dtype=float)
        if pre.shape != (self.dimensions,) or post.shape != (self.dimensions,):
            raise ValueError("activity vectors have an incompatible dimension")
        if not verified:
            self.last_update_norm = 0.0
            return {"plasticity_applied": False, "reason": "criterion or feedback is unverified",
                    "update_norm": 0.0}
        rate = self.learning_rate * clamp(gate)
        hebbian = np.outer(pre, post)
        oja_normalizer = np.outer(pre ** 2, np.ones(self.dimensions))
        delta = rate * (hebbian - self.weights * oja_normalizer)
        self.weights = np.clip(self.weights + delta, -1.0, 1.0)
        self.update_count += 1
        self.last_update_norm = float(np.linalg.norm(delta))
        return {"plasticity_applied": True, "update_norm": self.last_update_norm,
                "mean_weight": float(np.mean(self.weights)),
                "maximum_absolute_weight": float(np.max(np.abs(self.weights))),
                "learning_rule": "gated_hebbian_oja_approximation"}

    def snapshot(self) -> dict[str, Any]:
        """Return bounded cortical association diagnostics."""
        return {"dimensions": self.dimensions, "update_count": self.update_count,
                "last_update_norm": self.last_update_norm,
                "maximum_absolute_weight": float(np.max(np.abs(self.weights))),
                "model": "distributed_cortical_association_surrogate"}


class BiologicalTestingEffectNetwork:
    """Coordinate hippocampal retrieval, cortical reinstatement, control, monitoring, and plasticity."""

    def __init__(self, dimensions: int = 64, learning_rate: float = 0.04) -> None:
        """Initialize interacting computational surrogates with a shared feature dimension."""
        self.dimensions = dimensions
        self.population = NeuralPopulation("testing_effect_population", dimensions, seed=419)
        self.hippocampus = HippocampalPatternCompletion(dimensions)
        self.prefrontal = PrefrontalRetrievalControl()
        self.monitoring = AnteriorCingulateErrorMonitor()
        self.modulatory_gate = NeuromodulatoryLearningGate()
        self.cortex = CorticalAssociativePlasticity(dimensions, learning_rate)
        self.previous_activity = np.zeros(dimensions, dtype=float)

    def encode_target(self, target_id: str, content: Sequence[str],
                      context: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Encode target features and context into population and hippocampal pattern stores."""
        vector = vectorize_text(content, self.dimensions)
        activity = self.population.update(vector, gain=1.0)
        binding = self.hippocampus.bind(target_id, content, context)
        self.previous_activity = activity
        return {"population_energy": float(np.mean(activity ** 2)),
                "hippocampal_binding": binding,
                "semantic_feature_norm": float(np.linalg.norm(vector))}

    def retrieve(self, target_id: str, cue: str, difficulty: float,
                 relevance: float, fatigue: float = 0.0) -> dict[str, Any]:
        """Reactivate a target from a partial cue while biasing processing toward the task goal."""
        completion = self.hippocampus.complete(cue, target_id)
        control = self.prefrontal.configure(relevance, difficulty, fatigue)
        cue_vector = vectorize_text([cue], self.dimensions)
        activity = self.population.update(cue_vector, gain=0.8 + 0.6 * control["goal_focus"],
                                          inhibition=0.10 + 0.20 * control["irrelevant_competition_inhibition"])
        self.previous_activity = activity
        return {"pattern_completion": completion, "retrieval_control": control,
                "population_energy": float(np.mean(activity ** 2)),
                "cue_activity_norm": float(np.linalg.norm(activity))}

    def monitor_attempt(self, predicted_success: float, observed_quality: float,
                        candidate_scores: Mapping[str, float] | None = None) -> dict[str, float]:
        """Route behavioral retrieval discrepancy and candidate competition to the monitor."""
        return self.monitoring.monitor(predicted_success, observed_quality, candidate_scores)

    def learn_from_verified_evidence(
        self, target: Sequence[str], response: str, outcome_quality: float,
        surprise: float, relevance: float, effort: float,
        feedback_verified: bool,
    ) -> dict[str, Any]:
        """Use a verified target and corrective evidence to update cortical association surrogates."""
        content_activity = vectorize_text(target, self.dimensions)
        response_activity = vectorize_text([response], self.dimensions)
        gate = self.modulatory_gate.compute(outcome_quality, surprise, relevance,
                                            effort, feedback_verified)
        plasticity = self.cortex.update(content_activity, response_activity,
                                        gate["learning_gate"], feedback_verified)
        return {"neuromodulatory_gate": gate, "cortical_plasticity": plasticity,
                "unverified_error_rehearsal_prevented": not feedback_verified}

    def snapshot(self) -> dict[str, Any]:
        """Return all network diagnostics with explicit model-level scientific limitations."""
        return {"population": self.population.snapshot(),
                "hippocampus": self.hippocampus.snapshot(),
                "prefrontal_control": self.prefrontal.snapshot(),
                "error_monitor": self.monitoring.snapshot(),
                "neuromodulatory_gate": self.modulatory_gate.snapshot(),
                "cortical_plasticity": self.cortex.snapshot(),
                "literal_neuron_simulation": False,
                "biological_fidelity": "functional computational surrogate"}
