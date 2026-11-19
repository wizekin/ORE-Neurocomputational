"""Biologically inspired surrogates for Point 05 contextual variation and interleaved learning."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .functional import clamp, cosine_similarity


def _hash_vector(items: Mapping[str, float] | Mapping[str, str],
                 dimensions: int, seed: int) -> np.ndarray:
    """Project named features into a deterministic signed-hash population vector."""
    vector = np.zeros(dimensions, dtype=float)
    for key, raw_value in items.items():
        token = f"{key}={raw_value}"
        digest = hashlib.blake2b(
            f"{seed}:{token}".encode("utf-8"), digest_size=8
        ).digest()
        number = int.from_bytes(digest, "big")
        index = number % dimensions
        sign = 1.0 if ((number >> 11) & 1) else -1.0
        try:
            magnitude = float(raw_value)
        except (TypeError, ValueError):
            magnitude = 1.0
        vector[index] += sign * magnitude
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else vector


def _activation(values: np.ndarray) -> np.ndarray:
    """Apply a bounded logistic rate activation to a population vector."""
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


@dataclass
class RecurrentPopulation:
    """Maintain recurrent feature-population activity with gain and inhibitory competition."""

    name: str
    dimensions: int = 64
    recurrent_scale: float = 0.10
    seed: int = 505
    activity: np.ndarray = field(init=False, repr=False)
    weights: np.ndarray = field(init=False, repr=False)
    update_count: int = 0

    def __post_init__(self) -> None:
        """Initialize a reproducible recurrent connection matrix and neutral activity."""
        if self.dimensions < 4 or self.recurrent_scale < 0:
            raise ValueError("dimensions must be >= 4 and recurrent_scale non-negative")
        rng = np.random.default_rng(self.seed)
        self.weights = rng.normal(
            0.0, 1.0 / math.sqrt(self.dimensions),
            size=(self.dimensions, self.dimensions),
        ) * self.recurrent_scale
        np.fill_diagonal(self.weights, 0.0)
        self.activity = np.zeros(self.dimensions, dtype=float)

    def update(
        self,
        input_vector: Sequence[float],
        gain: float = 1.0,
        inhibition: float = 0.12,
    ) -> np.ndarray:
        """Integrate input, recurrent activation, and global inhibition into bounded activity."""
        incoming = np.asarray(input_vector, dtype=float)
        if incoming.shape != (self.dimensions,) or not np.all(np.isfinite(incoming)):
            raise ValueError("input_vector must be finite and match population dimensions")
        recurrent = self.weights @ self.activity
        inhibitory = float(inhibition) * float(np.mean(self.activity))
        self.activity = _activation(float(gain) * incoming + recurrent - inhibitory)
        self.update_count += 1
        return self.activity.copy()

    def snapshot(self) -> dict[str, Any]:
        """Return activity statistics without exposing internal mutable arrays."""
        return {"name": self.name, "dimensions": self.dimensions,
                "mean_activity": float(np.mean(self.activity)),
                "activity_energy": float(np.mean(self.activity ** 2)),
                "max_activity": float(np.max(self.activity)),
                "update_count": self.update_count,
                "model": "recurrent_rate_population_surrogate"}


class HippocampalContextMemory:
    """Store episode-context bindings and estimate retrieval from partial contextual cues."""

    def __init__(self, dimensions: int = 64) -> None:
        """Initialize episode vectors and item-specific context associations."""
        self.dimensions = dimensions
        self.episodes: dict[str, dict[str, Any]] = {}
        self.reactivation_counts: dict[str, int] = {}

    def encode_episode(
        self,
        example_id: str,
        concept_id: str,
        concept_features: Mapping[str, float],
        context_features: Mapping[str, str],
        relations: Sequence[str],
    ) -> dict[str, Any]:
        """Bind a specific example, its context, and relational details into one episode record."""
        concept_vector = _hash_vector(concept_features, self.dimensions, 105)
        context_vector = _hash_vector(context_features, self.dimensions, 205)
        self.episodes[example_id] = {
            "concept_id": concept_id,
            "concept_vector": concept_vector,
            "context_vector": context_vector,
            "context_features": dict(context_features),
            "relations": tuple(relations),
        }
        self.reactivation_counts.setdefault(example_id, 0)
        return {"example_id": example_id, "concept_id": concept_id,
                "episode_bound": True, "concept_vector_norm": float(np.linalg.norm(concept_vector)),
                "context_vector_norm": float(np.linalg.norm(context_vector)),
                "relation_count": len(relations),
                "model": "hippocampal_episode_context_binding_surrogate"}

    def pattern_complete(
        self, cue: Mapping[str, str], expected_concept_id: str | None = None,
    ) -> dict[str, Any]:
        """Reinstate an episode from partial context cues using similarity-based pattern completion."""
        if not self.episodes:
            return {"retrieved": False, "best_episode_id": None,
                    "similarity": 0.0, "candidate_scores": {}}
        cue_vector = _hash_vector(cue, self.dimensions, 205)
        scores = {
            key: cosine_similarity(
                {str(i): float(value) for i, value in enumerate(cue_vector)},
                {str(i): float(value) for i, value in enumerate(value["context_vector"])},
            )
            for key, value in self.episodes.items()
        }
        best = max(scores, key=scores.get)
        self.reactivation_counts[best] = self.reactivation_counts.get(best, 0) + 1
        record = self.episodes[best]
        return {"retrieved": scores[best] > 0.0, "best_episode_id": best,
                "concept_id": record["concept_id"],
                "expected_concept_match": (record["concept_id"] == expected_concept_id
                                           if expected_concept_id else None),
                "similarity": scores[best], "candidate_scores": scores,
                "reactivation_count": self.reactivation_counts[best],
                "model": "partial_cue_pattern_completion_surrogate"}

    def snapshot(self) -> dict[str, Any]:
        """Summarize stored episodes and reactivation counts."""
        return {"episode_count": len(self.episodes),
                "total_reactivations": sum(self.reactivation_counts.values()),
                "reactivation_counts": dict(self.reactivation_counts),
                "literal_hippocampal_simulation": False}


class CorticalConceptRepresentation:
    """Maintain shared concept prototypes while leaving episodic records intact."""

    def __init__(self, learning_rate: float = 0.08) -> None:
        """Initialize separate feature prototypes and concept update counters."""
        if not 0.0 < learning_rate <= 1.0:
            raise ValueError("learning_rate must be in (0, 1]")
        self.learning_rate = learning_rate
        self.prototypes: dict[str, dict[str, float]] = {}
        self.update_counts: dict[str, int] = {}

    def update(
        self, concept_id: str, features: Mapping[str, float], verified: bool,
    ) -> dict[str, Any]:
        """Update a concept prototype from verified feature evidence using a bounded EMA."""
        if not verified:
            return {"updated": False, "reason": "example evidence is not verified"}
        old = self.prototypes.setdefault(concept_id, {})
        keys = set(old) | set(features)
        for key in keys:
            previous = old.get(key, 0.0)
            observed = float(features.get(key, 0.0))
            old[key] = previous + self.learning_rate * (observed - previous)
        self.update_counts[concept_id] = self.update_counts.get(concept_id, 0) + 1
        return {"updated": True, "concept_id": concept_id,
                "prototype": dict(old), "update_count": self.update_counts[concept_id],
                "model": "distributed_cortical_concept_prototype_surrogate"}

    def snapshot(self) -> dict[str, Any]:
        """Return prototype counts and learned feature summaries."""
        return {"concept_count": len(self.prototypes),
                "prototypes": {key: dict(value) for key, value in self.prototypes.items()},
                "update_counts": dict(self.update_counts),
                "literal_single_cortical_storage_site": False}


class SensoryAssociationPathway:
    """Represent modality-specific input processing followed by a shared feature projection."""

    def __init__(self, dimensions: int = 64, seed: int = 505) -> None:
        """Initialize fixed deterministic modality projection matrices."""
        self.dimensions = dimensions
        self.seed = seed
        self.pathways: dict[str, RecurrentPopulation] = {}
        self.last_outputs: dict[str, np.ndarray] = {}

    def encode(self, modality: str, features: Mapping[str, float]) -> dict[str, Any]:
        """Process modality-specific concept features through a named population pathway."""
        population = self.pathways.setdefault(
            modality,
            RecurrentPopulation(f"{modality}_association_pathway", self.dimensions,
                                seed=self.seed + len(self.pathways) * 17),
        )
        vector = _hash_vector(features, self.dimensions, self.seed + 11)
        activity = population.update(vector)
        self.last_outputs[modality] = activity
        return {"modality": modality, "pathway_activity_energy": float(np.mean(activity ** 2)),
                "feature_norm": float(np.linalg.norm(vector)),
                "model": "modality_specific_association_surrogate"}

    def compare_modalities(self, left: str, right: str) -> dict[str, Any]:
        """Compare activity patterns from two modalities without requiring identical sensory input."""
        if left not in self.last_outputs or right not in self.last_outputs:
            return {"comparable": False, "similarity": None}
        a = self.last_outputs[left]
        b = self.last_outputs[right]
        denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
        similarity = float(np.dot(a, b) / denominator) if denominator > 1e-12 else 0.0
        return {"comparable": True, "similarity": similarity,
                "left_modality": left, "right_modality": right}

    def snapshot(self) -> dict[str, Any]:
        """Return modality-pathway diagnostics."""
        return {"modalities": {
            key: value.snapshot() for key, value in self.pathways.items()
        }, "model": "sensory_association_pathways_surrogate"}


class PrefrontalTaskController:
    """Model goal maintenance and selection among related concepts under task demands."""

    def __init__(self) -> None:
        """Initialize the active task set and control diagnostics."""
        self.active_concept: str | None = None
        self.last_switch_cost = 0.0
        self.update_count = 0

    def select(
        self,
        candidate_scores: Mapping[str, float],
        relevant_context: float,
        fatigue: float,
    ) -> dict[str, Any]:
        """Select a candidate while tracking switching cost, contextual relevance, and fatigue."""
        if not candidate_scores:
            return {"selected_concept": None, "selection_confidence": 0.0}
        selected = max(candidate_scores, key=candidate_scores.get)
        switched = self.active_concept is not None and selected != self.active_concept
        self.last_switch_cost = (0.20 + 0.35 * clamp(fatigue)
                                 if switched else 0.0)
        self.active_concept = selected
        self.update_count += 1
        ordered = sorted((float(value) for value in candidate_scores.values()), reverse=True)
        margin = ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0]
        return {"selected_concept": selected,
                "selection_confidence": clamp(margin),
                "task_switch": switched,
                "task_switch_cost": self.last_switch_cost,
                "contextual_goal_support": clamp(relevant_context),
                "fatigue": clamp(fatigue),
                "update_count": self.update_count}

    def snapshot(self) -> dict[str, Any]:
        """Return task-set and switching diagnostics."""
        return {"active_concept": self.active_concept,
                "last_switch_cost": self.last_switch_cost,
                "update_count": self.update_count,
                "model": "prefrontal_task_set_control_surrogate"}


class MedialFrontalErrorMonitor:
    """Track unexpected category choices and prediction discrepancies across learning trials."""

    def __init__(self) -> None:
        """Initialize the monitoring state without assuming an error signal is always present."""
        self.last_error = 0.0
        self.mean_error = 0.0
        self.update_count = 0

    def update(self, expected: float, observed: float, confidence: float) -> dict[str, float]:
        """Compute a bounded prediction discrepancy and confidence-weighted surprise estimate."""
        discrepancy = float(expected) - float(observed)
        error = abs(discrepancy)
        surprise = error * (0.5 + 0.5 * clamp(confidence))
        self.update_count += 1
        self.mean_error += (error - self.mean_error) / self.update_count
        self.last_error = error
        return {"signed_prediction_error": discrepancy,
                "absolute_prediction_error": error,
                "confidence_weighted_surprise": surprise,
                "running_mean_error": self.mean_error}

    def snapshot(self) -> dict[str, Any]:
        """Return cumulative discrepancy-monitoring diagnostics."""
        return {"last_error": self.last_error, "mean_error": self.mean_error,
                "update_count": self.update_count,
                "model": "medial_frontal_error_monitor_surrogate"}


class NeuromodulatoryPlasticityGate:
    """Use feedback reliability, prediction discrepancy, and relevance to gate modeled plasticity."""

    def __init__(self) -> None:
        """Initialize a neutral learning gate and expected-outcome estimate."""
        self.expected_outcome = 0.5
        self.last_gate = 0.0
        self.update_count = 0

    def compute(
        self, observed_quality: float, prediction_error: float,
        relevance: float, verified: bool,
    ) -> dict[str, float]:
        """Compute an illustrative bounded gate without estimating transmitter concentrations."""
        quality = clamp(observed_quality)
        discrepancy = abs(float(prediction_error))
        prior = self.expected_outcome
        self.expected_outcome = clamp(prior + 0.10 * (quality - prior))
        verification = 1.0 if verified else 0.0
        gate = clamp(0.40 * quality + 0.25 * clamp(discrepancy)
                     + 0.20 * clamp(relevance) + 0.15 * verification)
        self.last_gate = gate
        self.update_count += 1
        return {"learning_gate": gate, "expected_outcome_before_update": prior,
                "observed_quality": quality, "prediction_discrepancy": discrepancy,
                "verified_evidence": verification}

    def snapshot(self) -> dict[str, Any]:
        """Return gate state and explicitly deny interpreting it as a measured transmitter level."""
        return {"last_gate": self.last_gate,
                "expected_outcome": self.expected_outcome,
                "update_count": self.update_count,
                "literal_neurotransmitter_measurement": False,
                "model": "neuromodulatory_plasticity_gate_surrogate"}


class BiologicalVariedContextsNetwork:
    """Coordinate episodic context memory, cortical concept learning, sensory pathways, and control."""

    def __init__(self, dimensions: int = 64, learning_rate: float = 0.08) -> None:
        """Initialize interacting computational surrogates with a common feature dimension."""
        self.dimensions = dimensions
        self.population = RecurrentPopulation("varied_context_population", dimensions, seed=705)
        self.hippocampus = HippocampalContextMemory(dimensions)
        self.cortex = CorticalConceptRepresentation(learning_rate)
        self.sensory = SensoryAssociationPathway(dimensions)
        self.prefrontal = PrefrontalTaskController()
        self.error_monitor = MedialFrontalErrorMonitor()
        self.modulatory_gate = NeuromodulatoryPlasticityGate()

    def encode_example(self, example: Any) -> dict[str, Any]:
        """Encode an example in a modality pathway, shared population, episode record, and concept model."""
        episode = self.hippocampus.encode_episode(
            example.example_id, example.concept_id, example.concept_features,
            example.context_features, example.relations,
        )
        sensory = self.sensory.encode(example.modality, example.concept_features)
        vector = _hash_vector(example.concept_features, self.dimensions, 805)
        activity = self.population.update(vector)
        cortical = self.cortex.update(
            example.concept_id, example.concept_features, example.verified
        )
        return {"episode_binding": episode, "sensory_processing": sensory,
                "population_energy": float(np.mean(activity ** 2)),
                "cortical_update": cortical}

    def infer_context(
        self, cues: Mapping[str, str], expected_concept_id: str | None = None,
    ) -> dict[str, Any]:
        """Reactivate context-linked episodes from partial cues through hippocampal pattern completion."""
        completion = self.hippocampus.pattern_complete(cues, expected_concept_id)
        cue_vector = _hash_vector(cues, self.dimensions, 205)
        activity = self.population.update(cue_vector, gain=0.9)
        return {"pattern_completion": completion,
                "population_energy": float(np.mean(activity ** 2))}

    def select_concept(
        self,
        candidate_scores: Mapping[str, float],
        relevance: float,
        fatigue: float,
    ) -> dict[str, Any]:
        """Apply a prefrontal task-selection surrogate to concept candidates."""
        return self.prefrontal.select(candidate_scores, relevance, fatigue)

    def learn_from_trial(
        self,
        expected_accuracy: float,
        observed_accuracy: float,
        confidence: float,
        relevance: float,
        verified: bool,
    ) -> dict[str, Any]:
        """Monitor trial discrepancy and compute a gated learning signal for cortical updates."""
        monitor = self.error_monitor.update(expected_accuracy, observed_accuracy, confidence)
        gate = self.modulatory_gate.compute(
            observed_accuracy, monitor["signed_prediction_error"], relevance, verified
        )
        return {"error_monitor": monitor, "plasticity_gate": gate,
                "plasticity_evidence_verified": verified}

    def snapshot(self) -> dict[str, Any]:
        """Return all biological-surrogate diagnostics and scientific boundary markers."""
        return {
            "population": self.population.snapshot(),
            "hippocampal_context_memory": self.hippocampus.snapshot(),
            "cortical_concept_representation": self.cortex.snapshot(),
            "sensory_pathways": self.sensory.snapshot(),
            "prefrontal_controller": self.prefrontal.snapshot(),
            "medial_frontal_monitor": self.error_monitor.snapshot(),
            "neuromodulatory_gate": self.modulatory_gate.snapshot(),
            "literal_neuron_simulation": False,
            "biological_fidelity": "functional computational surrogate",
        }
