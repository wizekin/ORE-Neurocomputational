"""Biologically motivated, explicitly approximate networks for Point 02 integration."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .functional import clamp, cosine_similarity, softmax, vectorize_tokens


@dataclass
class NeuralPopulation:
    """Maintain bounded recurrent population activity with excitatory and inhibitory drive."""

    name: str
    dimensions: int = 64
    recurrent_scale: float = 0.12
    seed: int = 23
    activity: np.ndarray = field(init=False, repr=False)
    weights: np.ndarray = field(init=False, repr=False)
    update_count: int = 0

    def __post_init__(self) -> None:
        """Initialize deterministic recurrent weights and neutral activity."""
        if self.dimensions < 8 or not math.isfinite(self.recurrent_scale) or self.recurrent_scale < 0:
            raise ValueError("invalid neural population dimensions or recurrent scale")
        rng = np.random.default_rng(self.seed)
        self.weights = rng.normal(0.0, 1.0 / math.sqrt(self.dimensions),
                                  (self.dimensions, self.dimensions)) * self.recurrent_scale
        self.activity = np.zeros(self.dimensions, dtype=float)

    def update(self, incoming: Sequence[float], gain: float = 1.0,
               inhibition: float = 0.0) -> np.ndarray:
        """Advance one recurrent activation step using a clipped tanh transfer function."""
        vector = np.asarray(incoming, dtype=float)
        if vector.shape != (self.dimensions,) or not np.all(np.isfinite(vector)):
            raise ValueError("incoming must be a finite vector matching population dimensions")
        if not math.isfinite(gain) or gain < 0 or not math.isfinite(inhibition):
            raise ValueError("gain and inhibition must be finite and gain non-negative")
        drive = gain * vector + self.weights @ self.activity - inhibition
        self.activity = np.tanh(np.clip(drive, -20.0, 20.0))
        self.update_count += 1
        return self.activity.copy()

    def snapshot(self) -> dict[str, Any]:
        """Expose compact population diagnostics without exporting internal weight matrices."""
        return {"name": self.name, "dimensions": self.dimensions,
                "update_count": self.update_count,
                "mean_activity": float(np.mean(self.activity)),
                "activity_energy": float(np.mean(self.activity ** 2)),
                "model": "recurrent_population_surrogate"}


class HippocampalPatternCompletionNetwork:
    """Store episode vectors and reinstate likely patterns from partial retrieval cues."""

    def __init__(self, dimensions: int = 64) -> None:
        """Initialize an episodic vector store with fixed feature dimensionality."""
        self.dimensions = dimensions
        self.episodes: dict[str, np.ndarray] = {}
        self.metadata: dict[str, dict[str, Any]] = {}
        self.bind_count = 0

    def bind(self, episode_id: str, vector: Sequence[float], metadata: Mapping[str, Any]) -> dict[str, Any]:
        """Bind an episode representation and its context metadata without overwriting identity."""
        value = np.asarray(vector, dtype=float)
        if value.shape != (self.dimensions,) or not np.all(np.isfinite(value)):
            raise ValueError("episode vector must match hippocampal dimensions and be finite")
        if episode_id in self.episodes:
            raise ValueError("episode_id already exists")
        self.episodes[episode_id] = value.copy()
        self.metadata[episode_id] = dict(metadata)
        self.bind_count += 1
        return {"episode_id": episode_id, "bound": True,
                "context_keys": sorted(str(x) for x in metadata),
                "binding_model": "hippocampal_relational_binding_surrogate"}

    def reinstate(self, cue: Sequence[float], limit: int = 5) -> list[dict[str, Any]]:
        """Rank stored episodes by cosine match to a partial cue vector."""
        value = np.asarray(cue, dtype=float)
        if value.shape != (self.dimensions,) or not np.all(np.isfinite(value)):
            raise ValueError("cue vector must match hippocampal dimensions and be finite")
        rows = [{"episode_id": key, "similarity": cosine_similarity(value, vector),
                 "metadata": dict(self.metadata[key])} for key, vector in self.episodes.items()]
        return sorted(rows, key=lambda row: (-row["similarity"], row["episode_id"]))[:max(0, limit)]

    def snapshot(self) -> dict[str, Any]:
        """Return episode count and binding count as hippocampal-model diagnostics."""
        return {"episode_count": len(self.episodes), "binding_count": self.bind_count,
                "model": "pattern_completion_and_relational_binding_surrogate"}


class DistributedCorticalAssociationNetwork:
    """Maintain bounded co-activation weights and spread activation through learned links."""

    def __init__(self, learning_rate: float = 0.08) -> None:
        """Initialize associative strengths and a bounded learning rate."""
        if not math.isfinite(learning_rate) or not 0.0 <= learning_rate <= 1.0:
            raise ValueError("learning_rate must lie in [0, 1]")
        self.learning_rate = float(learning_rate)
        self.associations: dict[tuple[str, str], float] = {}
        self.update_count = 0

    def learn(self, cues: Sequence[str], targets: Sequence[str], gate: float = 1.0) -> int:
        """Apply a bounded co-activation update to cue-target associations."""
        gate = clamp(gate)
        updates = 0
        for cue in set(cues):
            for target in set(targets):
                if cue == target:
                    continue
                key = tuple(sorted((str(cue), str(target))))
                old = self.associations.get(key, 0.0)
                delta = self.learning_rate * gate * (1.0 - old)
                self.associations[key] = clamp(old + delta)
                updates += 1
        self.update_count += updates
        return updates

    def spread(self, active: Mapping[str, float], limit: int = 10) -> dict[str, float]:
        """Propagate graded activation one associative hop and retain the strongest outputs."""
        output: dict[str, float] = {}
        for (left, right), strength in self.associations.items():
            if left in active:
                output[right] = max(output.get(right, 0.0), clamp(active[left]) * strength)
            if right in active:
                output[left] = max(output.get(left, 0.0), clamp(active[right]) * strength)
        return dict(sorted(output.items(), key=lambda item: (-item[1], item[0]))[:max(0, limit)])

    def snapshot(self) -> dict[str, Any]:
        """Return association counts and strongest learned links."""
        strongest = [{"left": left, "right": right, "strength": value}
                     for (left, right), value in sorted(
                         self.associations.items(), key=lambda item: (-item[1], item[0]))[:20]]
        return {"association_count": len(self.associations), "update_count": self.update_count,
                "strongest_associations": strongest,
                "model": "distributed_cortical_hebbian_association_surrogate"}


class MedialPrefrontalSchemaNetwork:
    """Model context-sensitive competition among schema candidates and conflict monitoring."""

    def __init__(self, temperature: float = 0.5) -> None:
        """Initialize a positive competition temperature and neutral state."""
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        self.temperature = temperature
        self.selection_count = 0
        self.last_distribution: tuple[float, ...] = ()

    def select(self, scores: Sequence[float]) -> dict[str, Any]:
        """Normalize candidate schema activation and retain alternatives when competition is close."""
        probabilities = softmax(scores, self.temperature)
        self.last_distribution = probabilities
        self.selection_count += 1
        winner = int(np.argmax(probabilities)) if probabilities else None
        return {"probabilities": list(probabilities), "selected_index": winner,
                "selection_entropy": float(-sum(p * math.log(p) for p in probabilities if p > 0)),
                "alternatives_retained": len(probabilities) > 1,
                "model": "medial_prefrontal_schema_selection_surrogate"}

    def snapshot(self) -> dict[str, Any]:
        """Expose schema selection activity and the last competitive distribution."""
        return {"selection_count": self.selection_count,
                "last_distribution": list(self.last_distribution),
                "model": "mPFC_schema_control_surrogate"}


class FrontoparietalRetrievalControlNetwork:
    """Represent task-goal-biased retrieval selection and competition between memory candidates."""

    def __init__(self, dimensions: int = 64) -> None:
        """Initialize a recurrent frontoparietal control population."""
        self.population = NeuralPopulation("frontoparietal_retrieval_control", dimensions, seed=313)

    def focus(self, vector: Sequence[float], competition: float = 0.0) -> dict[str, Any]:
        """Update control activity with competition-dependent inhibitory gain."""
        activity = self.population.update(vector, gain=1.0, inhibition=max(0.0, competition))
        return {"population_activity": activity.tolist(),
                "control_gain": clamp(1.0 / (1.0 + max(0.0, competition))),
                "competition_signal": float(max(0.0, competition))}

    def snapshot(self) -> dict[str, Any]:
        """Return frontoparietal population diagnostics."""
        return self.population.snapshot()


class NeuromodulatoryLearningGate:
    """Combine novelty, evidence reliability, and prediction error into a bounded learning gate."""

    def __init__(self) -> None:
        """Initialize the learning-gate update counter."""
        self.update_count = 0
        self.last_gate = 0.0

    def compute(self, novelty: float, reliability: float, prediction_error: float) -> dict[str, float]:
        """Compute an illustrative salience/reliability gate for associative plasticity."""
        for name, value in (("novelty", novelty), ("reliability", reliability),
                            ("prediction_error", prediction_error)):
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        gate = clamp((0.35 * novelty + 0.35 * reliability + 0.30 * prediction_error)
                     * (0.5 + 0.5 * reliability))
        self.last_gate = gate
        self.update_count += 1
        return {"learning_gate": gate, "novelty_signal": clamp(novelty),
                "evidence_reliability": clamp(reliability),
                "prediction_error_signal": clamp(prediction_error)}

    def snapshot(self) -> dict[str, Any]:
        """Expose gate activity while avoiding claims about a specific neuromodulator level."""
        return {"update_count": self.update_count, "last_gate": self.last_gate,
                "model": "neuromodulatory_salience_gate_surrogate"}


class BiologicalPriorKnowledgeNetwork:
    """Coordinate episodic binding, cortical association, schema selection, and retrieval control."""

    def __init__(self, dimensions: int = 64, learning_rate: float = 0.08) -> None:
        """Create interacting distributed surrogates with consistent representation dimensions."""
        self.dimensions = dimensions
        self.population = NeuralPopulation("distributed_semantic_population", dimensions, seed=401)
        self.hippocampus = HippocampalPatternCompletionNetwork(dimensions)
        self.cortex = DistributedCorticalAssociationNetwork(learning_rate)
        self.mPFC = MedialPrefrontalSchemaNetwork()
        self.frontoparietal = FrontoparietalRetrievalControlNetwork(dimensions)
        self.learning_gate = NeuromodulatoryLearningGate()

    def encode(self, tokens: Sequence[str]) -> np.ndarray:
        """Encode a feature-token set and update a recurrent cortical population surrogate."""
        vector = vectorize_tokens(tokens, self.dimensions)
        return self.population.update(vector)

    def process_step(self, step: int, tokens: Sequence[str], novelty: float,
                     reliability: float, prediction_error: float = 0.0) -> dict[str, Any]:
        """Advance relevant population activity for one functional operation."""
        vector = vectorize_tokens(tokens or ("neutral",), self.dimensions)
        activity = self.population.update(vector, gain=1.0, inhibition=0.02 * step)
        control = self.frontoparietal.focus(vector, competition=prediction_error * 0.2)
        gate = self.learning_gate.compute(novelty, reliability, prediction_error)
        return {"distributed_population": self.population.snapshot(),
                "frontoparietal_control": control, "learning_gate": gate,
                "interpretation": "functional surrogate, not literal neural simulation"}

    def bind_and_learn(self, episode_id: str, tokens: Sequence[str], metadata: Mapping[str, Any],
                       cues: Sequence[str], targets: Sequence[str], gate: float) -> dict[str, Any]:
        """Bind an episode and update cortical associations using a gated Hebbian-style rule."""
        vector = vectorize_tokens(tokens, self.dimensions)
        binding = self.hippocampus.bind(episode_id, vector, metadata)
        updates = self.cortex.learn(cues, targets, gate)
        return {"hippocampal_binding": binding, "cortical_association_updates": updates,
                "cortical_association_state": self.cortex.snapshot()}

    def snapshot(self) -> dict[str, Any]:
        """Return diagnostics for the distributed candidate biological network."""
        return {"distributed_semantic_population": self.population.snapshot(),
                "hippocampus": self.hippocampus.snapshot(),
                "cortical_associations": self.cortex.snapshot(),
                "medial_prefrontal_schema": self.mPFC.snapshot(),
                "frontoparietal_control": self.frontoparietal.snapshot(),
                "neuromodulatory_gate": self.learning_gate.snapshot(),
                "literal_neuron_simulation": False,
                "biological_fidelity": "computational surrogate"}
