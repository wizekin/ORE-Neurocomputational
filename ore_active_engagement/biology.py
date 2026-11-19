"""Biologically motivated population, control, episodic-binding, and plasticity surrogates.

All classes are computational analogues of distributed neural functions. Their matrices,
activation states, and update equations are engineering models, not measured human neurons.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import ActiveEngagementState
from .functional import clamp, softmax, vectorize_texts


@dataclass
class NeuralPopulation:
    """Maintain recurrent population activity with excitation, inhibition, and gain control."""

    name: str
    dimensions: int = 64
    recurrent_scale: float = 0.15
    seed: int = 17
    activity: np.ndarray = field(init=False, repr=False)
    recurrent_weights: np.ndarray = field(init=False, repr=False)
    bias: np.ndarray = field(init=False, repr=False)
    update_count: int = 0

    def __post_init__(self) -> None:
        """Create reproducible recurrent weights and a neutral initial activity state."""
        if self.dimensions < 8:
            raise ValueError("dimensions must be at least 8")
        if not math.isfinite(self.recurrent_scale) or self.recurrent_scale < 0:
            raise ValueError("recurrent_scale must be finite and non-negative")
        rng = np.random.default_rng(self.seed)
        weights = rng.normal(0.0, 1.0 / math.sqrt(self.dimensions),
                             (self.dimensions, self.dimensions))
        self.recurrent_weights = weights * self.recurrent_scale
        self.activity = np.zeros(self.dimensions, dtype=float)
        self.bias = np.zeros(self.dimensions, dtype=float)

    def update(
        self,
        input_vector: Sequence[float],
        gain: float = 1.0,
        inhibition: float = 0.0,
    ) -> np.ndarray:
        """Advance one recurrent neural-population state using bounded nonlinear integration."""
        incoming = np.asarray(input_vector, dtype=float)
        if incoming.shape != (self.dimensions,) or not np.all(np.isfinite(incoming)):
            raise ValueError("input_vector must be a finite vector matching population dimensions")
        if not math.isfinite(gain) or gain < 0 or not math.isfinite(inhibition):
            raise ValueError("gain and inhibition must be finite; gain must be non-negative")
        recurrent = self.recurrent_weights @ self.activity
        drive = gain * incoming + recurrent + self.bias - float(inhibition)
        self.activity = np.tanh(np.clip(drive, -20.0, 20.0))
        self.update_count += 1
        return self.activity.copy()

    def reset(self) -> None:
        """Reset population activity without changing its learned recurrent weights."""
        self.activity.fill(0.0)
        self.update_count = 0

    def snapshot(self) -> dict[str, Any]:
        """Describe activation level and network scale without exporting internal matrices."""
        return {
            "population": self.name,
            "dimensions": self.dimensions,
            "processing_count": self.update_count,
            "mean_activity": float(np.mean(self.activity)),
            "activity_energy": float(np.mean(self.activity ** 2)),
            "recurrent_scale": self.recurrent_scale,
            "implementation": "recurrent_tanh_population_surrogate",
        }


class TemporalLanguageNetwork:
    """Model distributed temporal/language representation and recurrent sequence context."""

    def __init__(self, dimensions: int = 64) -> None:
        """Initialize a deterministic recurrent population for lexical and proposition features."""
        self.population = NeuralPopulation(
            "temporal_language", dimensions, recurrent_scale=0.12, seed=101
        )
        self.stream_state = np.zeros(dimensions, dtype=float)

    def encode(self, text: str) -> dict[str, Any]:
        """Encode one source segment and combine it with the preceding sequence state."""
        vector = vectorize_texts([text], self.population.dimensions)
        activity = self.population.update(vector + 0.15 * self.stream_state, gain=1.0)
        self.stream_state = 0.65 * self.stream_state + 0.35 * activity
        return {
            "embedding": vector.tolist(),
            "population_activity": activity.tolist(),
            "stream_state": self.stream_state.tolist(),
            "sequence_context_retained": True,
            "population": self.population.snapshot(),
        }

    def snapshot(self) -> dict[str, Any]:
        """Return recurrent language-population diagnostics."""
        return {
            **self.population.snapshot(),
            "stream_state_energy": float(np.mean(self.stream_state ** 2)),
        }


class FrontoparietalAttentionNetwork:
    """Model goal-biased selection, divisive competition, and attention-dependent gain."""

    def __init__(self, dimensions: int = 64) -> None:
        """Initialize a population representing distributed attentional control signals."""
        self.population = NeuralPopulation(
            "frontoparietal_attention", dimensions,
            recurrent_scale=0.10, seed=211,
        )

    def allocate(
        self,
        evidence_vectors: Sequence[Sequence[float]],
        relevance_scores: Sequence[float],
        temperature: float = 0.75,
    ) -> dict[str, Any]:
        """Convert evidence relevance into competing gains and update attention activity."""
        if len(evidence_vectors) != len(relevance_scores):
            raise ValueError("evidence_vectors and relevance_scores must have equal lengths")
        weights = softmax(relevance_scores, temperature=temperature)
        if evidence_vectors:
            stacked = np.asarray(evidence_vectors, dtype=float)
            if stacked.shape != (len(evidence_vectors), self.population.dimensions):
                raise ValueError("evidence vectors do not match attention population dimensions")
            pooled = np.average(stacked, axis=0, weights=np.asarray(weights))
        else:
            pooled = np.zeros(self.population.dimensions, dtype=float)
        activity = self.population.update(pooled, gain=1.0)
        gains = [
            clamp(0.5 + 0.5 * float(weight) * (1.0 + float(np.mean(np.abs(vector)))))
            for vector, weight in zip(evidence_vectors, weights)
        ]
        return {
            "attention_probabilities": list(weights),
            "gain_modulation": gains,
            "population_activity": activity.tolist(),
            "competition_normalized": True,
            "inhibition_is_relative_not_global_suppression": True,
        }

    def snapshot(self) -> dict[str, Any]:
        """Return attentional-population activity and processing count."""
        return self.population.snapshot()


class PrefrontalControlNetwork:
    """Model goal maintenance, uncertainty-sensitive control, and learning-gate modulation."""

    def __init__(self, dimensions: int = 64) -> None:
        """Initialize a recurrent prefrontal control-state surrogate."""
        self.population = NeuralPopulation(
            "prefrontal_control", dimensions, recurrent_scale=0.18, seed=307
        )
        self.current_goal_vector = np.zeros(dimensions, dtype=float)

    def maintain_goal(self, goal: str) -> dict[str, Any]:
        """Maintain a task-goal vector and update recurrent control activity."""
        self.current_goal_vector = vectorize_texts([goal], self.population.dimensions)
        activity = self.population.update(self.current_goal_vector, gain=1.05)
        return {
            "goal_activity": activity.tolist(),
            "goal_energy": float(np.mean(activity ** 2)),
            "goal_maintained": True,
        }

    def learning_gate(self, attention_quality: float, uncertainty: float) -> dict[str, float]:
        """Estimate a bounded plasticity gate from attended evidence and task uncertainty."""
        attention_quality = clamp(attention_quality)
        uncertainty = clamp(uncertainty)
        control_drive = 0.65 * attention_quality + 0.35 * (1.0 - uncertainty)
        activity = self.population.update(
            self.current_goal_vector,
            gain=0.5 + control_drive,
            inhibition=0.1 * (1.0 - attention_quality),
        )
        neural_gate = clamp(float(np.mean(np.abs(activity))))
        gate = clamp(0.4 * control_drive + 0.6 * neural_gate)
        return {
            "learning_gate": gate,
            "control_drive": control_drive,
            "mean_control_activity": float(np.mean(activity)),
        }

    def snapshot(self) -> dict[str, Any]:
        """Return prefrontal-control state without exposing private weight matrices."""
        return {
            **self.population.snapshot(),
            "goal_vector_energy": float(np.mean(self.current_goal_vector ** 2)),
        }


class HippocampalEpisodeNetwork:
    """Model hippocampal-style binding and partial-cue reinstatement of episode features."""

    def __init__(self, dimensions: int = 64) -> None:
        """Initialize an episodic population and an empty set of encoded episode vectors."""
        self.population = NeuralPopulation(
            "hippocampal_episode_binding", dimensions,
            recurrent_scale=0.08, seed=401,
        )
        self.episodes: list[dict[str, Any]] = []

    def bind(
        self,
        feature_vectors: Mapping[str, Sequence[float]],
        episode_metadata: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Bind co-present feature vectors into an episode pattern with source and context tags."""
        vectors = [
            np.asarray(value, dtype=float)
            for value in feature_vectors.values()
        ]
        if not vectors:
            raise ValueError("at least one feature vector is required for episode binding")
        if any(item.shape != (self.population.dimensions,) for item in vectors):
            raise ValueError("all feature vectors must match hippocampal population dimensions")
        bound = np.mean(np.stack(vectors, axis=0), axis=0)
        activity = self.population.update(bound, gain=1.0)
        record = {
            "episode_index": len(self.episodes) + 1,
            "feature_names": list(feature_vectors),
            "metadata": dict(episode_metadata),
            "pattern": activity.tolist(),
            "pattern_energy": float(np.mean(activity ** 2)),
            "binding_rule": "distributed_mean_pooling_surrogate",
        }
        self.episodes.append(record)
        return {
            "episode_index": record["episode_index"],
            "feature_names": record["feature_names"],
            "pattern": record["pattern"],
            "pattern_energy": record["pattern_energy"],
            "source_context_binding_preserved": True,
        }

    def reinstate(self, partial_cue: Sequence[float]) -> dict[str, Any]:
        """Retrieve the most similar stored episode pattern from a partial cue."""
        cue = np.asarray(partial_cue, dtype=float)
        if cue.shape != (self.population.dimensions,):
            raise ValueError("partial_cue must match hippocampal population dimensions")
        if not self.episodes:
            return {"retrieved_episode_index": None, "similarity": 0.0}
        cue_norm = float(np.linalg.norm(cue))
        scores = []
        for episode in self.episodes:
            pattern = np.asarray(episode["pattern"], dtype=float)
            denom = cue_norm * float(np.linalg.norm(pattern))
            scores.append(float(np.dot(cue, pattern) / denom) if denom > 1e-12 else 0.0)
        index = int(np.argmax(scores))
        return {
            "retrieved_episode_index": self.episodes[index]["episode_index"],
            "similarity": clamp(scores[index], 0.0, 1.0),
            "pattern_completion_is_approximate": True,
        }

    def snapshot(self) -> dict[str, Any]:
        """Report episodic binding count and hippocampal population diagnostics."""
        return {
            **self.population.snapshot(),
            "episodes_encoded": len(self.episodes),
            "pattern_completion_available": bool(self.episodes),
        }


class SynapticPlasticityNetwork:
    """Model bounded co-activation and timing-sensitive changes to effective synaptic weights."""

    def __init__(self, dimensions: int = 64) -> None:
        """Initialize an effective connection matrix for a simplified neural population."""
        self.dimensions = dimensions
        self.weights = np.zeros((dimensions, dimensions), dtype=float)
        self.update_count = 0

    def update(
        self,
        pre_activity: Sequence[float],
        post_activity: Sequence[float],
        learning_rate: float,
        gate: float,
        delta_t_seconds: float = 0.01,
        potentiation_amplitude: float = 0.01,
        depression_amplitude: float = 0.01,
        tau_seconds: float = 0.02,
        state: ActiveEngagementState | None = None,
    ) -> dict[str, Any]:
        """Apply bounded Hebbian/STDP-inspired updates and persist salient associations."""
        pre = np.asarray(pre_activity, dtype=float)
        post = np.asarray(post_activity, dtype=float)
        expected = (self.dimensions,)
        if pre.shape != expected or post.shape != expected:
            raise ValueError("pre_activity and post_activity must match plasticity dimensions")
        if not np.all(np.isfinite(pre)) or not np.all(np.isfinite(post)):
            raise ValueError("activity vectors must be finite")
        for name, value in (
            ("learning_rate", learning_rate), ("gate", gate),
            ("delta_t_seconds", delta_t_seconds),
            ("potentiation_amplitude", potentiation_amplitude),
            ("depression_amplitude", depression_amplitude), ("tau_seconds", tau_seconds),
        ):
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if learning_rate < 0 or tau_seconds <= 0 or potentiation_amplitude < 0 or depression_amplitude < 0:
            raise ValueError("learning rate/amplitudes must be non-negative and tau positive")
        gate = clamp(gate)
        pre_centered = pre - float(np.mean(pre))
        post_centered = post - float(np.mean(post))
        hebbian = float(learning_rate) * gate * np.outer(pre_centered, post_centered)
        if delta_t_seconds >= 0:
            timing_factor = potentiation_amplitude * math.exp(
                -delta_t_seconds / tau_seconds
            )
        else:
            timing_factor = -depression_amplitude * math.exp(
                delta_t_seconds / tau_seconds
            )
        timing_update = timing_factor * np.outer(pre, post)
        delta = hebbian + timing_update
        self.weights = np.clip(self.weights + delta, -1.0, 1.0)
        self.update_count += 1

        persisted = 0
        if state is not None:
            active_pre = np.argsort(np.abs(pre))[-4:]
            active_post = np.argsort(np.abs(post))[-4:]
            for left in active_pre:
                for right in active_post:
                    key = f"population_{int(left)}::population_{int(right)}"
                    old = state.associative_weights.get(key, 0.0)
                    change = float(delta[int(left), int(right)])
                    state.associative_weights[key] = clamp(old + change)
                    persisted += 1
        return {
            "update_count": self.update_count,
            "hebbian_update_norm": float(np.linalg.norm(hebbian)),
            "timing_update_norm": float(np.linalg.norm(timing_update)),
            "timing_direction": "potentiation" if delta_t_seconds >= 0 else "depression",
            "delta_t_seconds": float(delta_t_seconds),
            "learning_gate": gate,
            "weight_min": float(np.min(self.weights)),
            "weight_max": float(np.max(self.weights)),
            "weight_norm": float(np.linalg.norm(self.weights)),
            "persistent_association_updates": persisted,
            "weights_bounded": bool(np.all(np.abs(self.weights) <= 1.0)),
        }

    def snapshot(self) -> dict[str, Any]:
        """Return weight magnitude and stability diagnostics for the plasticity matrix."""
        return {
            "dimensions": self.dimensions,
            "processing_count": self.update_count,
            "mean_absolute_weight": float(np.mean(np.abs(self.weights))),
            "maximum_absolute_weight": float(np.max(np.abs(self.weights))),
            "weights_bounded": bool(np.all(np.abs(self.weights) <= 1.0)),
            "implementation": "bounded_hebbian_and_stdp_surrogate",
        }


class BiologicalActiveEngagementNetwork:
    """Coordinate temporal, attentional, prefrontal, hippocampal, and plasticity surrogates."""

    def __init__(self, dimensions: int = 64) -> None:
        """Initialize interacting population surrogates with matched vector dimensions."""
        self.dimensions = dimensions
        self.temporal_language = TemporalLanguageNetwork(dimensions)
        self.frontoparietal_attention = FrontoparietalAttentionNetwork(dimensions)
        self.prefrontal_control = PrefrontalControlNetwork(dimensions)
        self.hippocampal_episode = HippocampalEpisodeNetwork(dimensions)
        self.synaptic_plasticity = SynapticPlasticityNetwork(dimensions)

    def snapshot(self) -> dict[str, Any]:
        """Return distributed network diagnostics and explicitly label model limitations."""
        return {
            "temporal_language": self.temporal_language.snapshot(),
            "frontoparietal_attention": self.frontoparietal_attention.snapshot(),
            "prefrontal_control": self.prefrontal_control.snapshot(),
            "hippocampal_episode": self.hippocampal_episode.snapshot(),
            "synaptic_plasticity": self.synaptic_plasticity.snapshot(),
            "biological_fidelity": "functional computational surrogate",
            "literal_neuron_simulation": False,
            "interpretation": (
                "The components model candidate distributed functions; they do not reproduce "
                "the full anatomy, physiology, or dynamics of human learning."
            ),
        }
