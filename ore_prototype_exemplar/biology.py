"""Distributed biological-network surrogates for category representation and control.

The networks model candidate functional contributions of sensory/association cortex, hippocampal
binding, cortical regularity learning, and prefrontal/parietal attention. They are not anatomical
digital twins and do not imply one brain region stores one entire representation type.
"""
from __future__ import annotations

from hashlib import blake2b
import math
import time
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import (
    EpisodeInput, ExemplarTrace, PrototypeExemplarConfig, CategoryState,
)
from .functional import canonical_features, canonical_relations, retrieve_exemplars, stable_softmax


class SensoryAssociationNetwork:
    """Encode structured feature evidence with modality and uncertainty metadata."""

    def encode(self, episode: EpisodeInput, aliases: Mapping[str, str]) -> dict[str, Any]:
        """Create a canonical multimodal feature map while keeping missingness and uncertainty explicit."""
        features = canonical_features(episode, aliases)
        reliability = {
            aliases.get(name, name): episode.feature_reliability.get(name, episode.reliability)
            for name in episode.features
        }
        uncertainty = {
            aliases.get(name, name): episode.feature_uncertainty.get(name, 0.0)
            for name in episode.features
        }
        modalities = {
            aliases.get(name, name): episode.feature_modalities.get(name, "unspecified")
            for name in episode.features
        }
        observed = {name: value for name, value in features.items() if value is not None}
        return {
            "features": observed,
            "observation_mask": {name: value is not None for name, value in features.items()},
            "feature_reliability": reliability,
            "feature_uncertainty": uncertainty,
            "modalities": modalities,
            "observed_feature_count": len(observed),
            "interpretation": "structured feature evidence, not raw sensory neural activity",
        }


class HippocampalEpisodicNetwork:
    """Bind and retrieve episode-specific feature/context/relation conjunctions."""

    def __init__(self, config: PrototypeExemplarConfig) -> None:
        """Initialize independent exemplar indices and episodic retrieval parameters."""
        self.config = config
        self.memories: dict[str, dict[str, ExemplarTrace]] = {}
        self.binding_count = 0
        self.retrieval_count = 0
        self.pattern_completion_count = 0

    def bind(
        self, episode: EpisodeInput, category_id: str, aliases: Mapping[str, str],
        diagnosticity: float = 0.0, is_exception: bool = False,
    ) -> ExemplarTrace:
        """Create an independent exemplar trace with contextual and relational binding."""
        now = episode.timestamp if episode.timestamp is not None else time.time()
        trace = ExemplarTrace(
            episode_id=episode.episode_id,
            category_id=category_id,
            features=canonical_features(episode, aliases),
            context=dict(episode.context),
            relations=canonical_relations(episode, aliases),
            reliability=episode.reliability,
            feature_reliability=dict(episode.feature_reliability),
            created_at=now,
            base_strength=episode.reliability,
            is_exception=is_exception,
            diagnosticity=float(diagnosticity),
        )
        self.binding_count += 1
        return trace

    def store(self, trace: ExemplarTrace) -> bool:
        """Store an episode trace once and reject conflicting duplicate identity."""
        category = self.memories.setdefault(trace.category_id, {})
        previous = category.get(trace.episode_id)
        if previous is not None:
            if previous.to_dict() != trace.to_dict():
                raise ValueError(f"exemplar {trace.episode_id!r} already exists with different content")
            return False
        category[trace.episode_id] = trace
        return True

    def retrieve(
        self, episode: EpisodeInput, state: CategoryState, weights: Mapping[str, float], now: float
    ) -> list[dict[str, Any]]:
        """Retrieve distinct matching episodic patterns using partial cues and memory accessibility."""
        self.retrieval_count += 1
        rows = retrieve_exemplars(episode, state, weights, self.config, now)
        self.pattern_completion_count += int(bool(rows))
        for row in rows:
            trace = state.exemplars.get(str(row["episode_id"]))
            if trace is not None:
                trace.retrieval_count += 1
        return rows

    def rehearse(self, category_id: str, episode_ids: Sequence[str], now: float) -> int:
        """Strengthen selected episodic traces with a decaying replay event."""
        category = self.memories.get(category_id, {})
        count = 0
        for episode_id in episode_ids:
            trace = category.get(episode_id)
            if trace is not None:
                trace.rehearsal_events.append((float(now), self.config.rehearsal_gain))
                count += 1
        return count

    def snapshot(self) -> dict[str, Any]:
        """Describe episodic storage and retrieval without exposing the full personal memory contents."""
        return {
            "model": "hippocampal_pattern_separation_completion_surrogate",
            "category_count": len(self.memories),
            "exemplar_count": sum(len(rows) for rows in self.memories.values()),
            "binding_count": self.binding_count,
            "retrieval_count": self.retrieval_count,
            "pattern_completion_count": self.pattern_completion_count,
            "episode_identity_preserved": True,
        }


class DistributedCorticalPrototypeNetwork:
    """Model gradual extraction of recurring category patterns in distributed cortical populations."""

    def __init__(self, config: PrototypeExemplarConfig) -> None:
        """Initialize deterministic sparse projections and recurrent population dynamics."""
        self.config = config
        self.population_size = config.cortical_units
        rng = np.random.default_rng(config.random_seed)
        recurrent = rng.normal(0.0, 1.0 / math.sqrt(self.population_size),
                               (self.population_size, self.population_size))
        eigen = max(float(np.max(np.abs(np.linalg.eigvals(recurrent)))), 1e-9)
        self.recurrent = recurrent / eigen * 0.82
        self.category_patterns: dict[str, np.ndarray] = {}
        self.synaptic_associations: dict[str, np.ndarray] = {}
        self.learning_update_count = 0

    def _feature_index(self, feature: str) -> tuple[int, float]:
        """Map a feature name reproducibly onto a distributed population unit and signed gain."""
        digest = blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        return value % self.population_size, (1.0 if value & 1 else -1.0)

    def encode(self, features: Mapping[str, Any]) -> np.ndarray:
        """Project sparse symbolic feature activity into a reproducible distributed population vector."""
        vector = np.zeros(self.population_size, dtype=float)
        for name, value in features.items():
            index, sign = self._feature_index(name)
            if isinstance(value, bool):
                magnitude = float(value)
            elif isinstance(value, (int, float)):
                magnitude = math.tanh(float(value))
            elif value is None:
                continue
            else:
                token_digest = blake2b(str(value).encode("utf-8"), digest_size=2).digest()
                magnitude = (int.from_bytes(token_digest, "big") / 32767.5) - 1.0
            vector[index] += sign * magnitude
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 1e-12 else vector

    def settle(self, drive: np.ndarray) -> np.ndarray:
        """Iterate recurrent cortical population activity with bounded nonlinear dynamics."""
        activity = np.asarray(drive, dtype=float).copy()
        for _ in range(self.config.recurrent_iterations):
            activity = np.tanh(drive + self.config.learning_rate * (self.recurrent @ activity))
        return activity

    def learn(self, category_id: str, features: Mapping[str, Any], reliability: float) -> dict[str, Any]:
        """Update a category-level population pattern and Hebbian co-activation association matrix."""
        drive = self.encode(features)
        settled = self.settle(drive)
        previous = self.category_patterns.get(category_id)
        if previous is None:
            updated = settled.copy()
        else:
            rate = self.config.learning_rate * max(0.0, min(1.0, reliability))
            updated = (1.0 - rate) * previous + rate * settled
        norm = float(np.linalg.norm(updated))
        if norm > 1e-12:
            updated = updated / norm
        self.category_patterns[category_id] = updated
        association = self.synaptic_associations.setdefault(
            category_id, np.zeros((self.population_size, self.population_size), dtype=float)
        )
        association += self.config.learning_rate * max(0.0, min(1.0, reliability)) * np.outer(
            settled, settled
        )
        self.learning_update_count += 1
        return {"population_activity": settled.tolist(), "pattern_similarity": float(
            np.dot(updated, settled)), "learning_update_count": self.learning_update_count}

    def similarity(self, category_id: str, features: Mapping[str, Any]) -> float:
        """Measure overlap between current distributed activity and the learned category pattern."""
        pattern = self.category_patterns.get(category_id)
        if pattern is None:
            return 0.0
        activity = self.settle(self.encode(features))
        return max(0.0, min(1.0, (float(np.dot(pattern, activity)) + 1.0) / 2.0))

    def snapshot(self) -> dict[str, Any]:
        """Expose population sizes and learned category summaries without exporting synaptic matrices."""
        return {
            "model": "distributed_cortical_regularities_surrogate",
            "population_size": self.population_size,
            "category_pattern_count": len(self.category_patterns),
            "learning_update_count": self.learning_update_count,
            "recurrent_spectral_radius": float(max(abs(np.linalg.eigvals(self.recurrent)))),
            "interpretation": "repeated co-activation strengthens distributed category regularities",
        }


class PrefrontalParietalControlNetwork:
    """Select task-relevant feature gains and coordinate prototype/exemplar strategy control."""

    def __init__(self, temperature: float = 1.0) -> None:
        """Initialize the competitive attention and task-control network surrogate."""
        self.temperature = float(temperature)
        self.last_feature_gains: dict[str, float] = {}
        self.strategy_history: list[float] = []

    def select_feature_gains(
        self, feature_names: Sequence[str], attention: Mapping[str, float],
        goals: Mapping[str, float], prior: Mapping[str, float],
    ) -> dict[str, float]:
        """Combine bottom-up attention and top-down goal/prior weights using competitive softmax."""
        names = sorted(set(feature_names))
        if not names:
            self.last_feature_gains = {}
            return {}
        raw = [0.50 * float(attention.get(name, 0.5))
               + 0.30 * float(goals.get(name, 0.5))
               + 0.20 * float(prior.get(name, 0.5)) for name in names]
        distribution = stable_softmax(raw, self.temperature)
        peak = max(distribution)
        self.last_feature_gains = {name: float(value / peak)
                                   for name, value in zip(names, distribution)}
        return dict(self.last_feature_gains)

    def register_strategy(self, prototype_weight: float) -> None:
        """Retain the strategy-control signal used for the latest category decision."""
        self.strategy_history.append(max(0.0, min(1.0, float(prototype_weight))))

    def snapshot(self) -> dict[str, Any]:
        """Return the current attention profile and strategy-control history."""
        return {
            "model": "prefrontal_parietal_goal_and_attention_control_surrogate",
            "feature_gains": dict(self.last_feature_gains),
            "strategy_decision_count": len(self.strategy_history),
            "mean_prototype_weight": (float(np.mean(self.strategy_history))
                                      if self.strategy_history else None),
        }


class NeuromodulatoryFeedbackGate:
    """Model outcome-sensitive learning modulation without equating every surprise with reward."""

    def __init__(self, slope: float = 5.0, midpoint: float = 0.1) -> None:
        """Initialize feedback sensitivity and the prediction-error midpoint."""
        self.slope = float(slope)
        self.midpoint = float(midpoint)
        self.last_gate = 0.0
        self.update_count = 0

    def compute_gate(self, prediction_error: float, reliability: float,
                     feedback_available: bool) -> float:
        """Open a bounded learning gate only when explicit feedback is available."""
        if not feedback_available:
            self.last_gate = 0.0
            return self.last_gate
        argument = self.slope * (abs(float(prediction_error)) - self.midpoint)
        self.last_gate = (1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, argument)))))
        self.last_gate *= max(0.0, min(1.0, float(reliability)))
        self.update_count += 1
        return self.last_gate

    def snapshot(self) -> dict[str, Any]:
        """Describe feedback-gated plasticity and its accumulated explicit teaching events."""
        return {"model": "neuromodulatory_feedback_gate_surrogate",
                "last_gate": self.last_gate, "feedback_update_count": self.update_count,
                "unlabelled_surprise_is_not_a_teaching_signal": True}


class BiologicalPrototypeExemplarSystem:
    """Coordinate distributed sensory, episodic, cortical, attentional, and feedback mechanisms."""

    def __init__(self, config: PrototypeExemplarConfig) -> None:
        """Construct the biological surrogate subsystems with a shared configuration."""
        self.config = config
        self.sensory = SensoryAssociationNetwork()
        self.hippocampus = HippocampalEpisodicNetwork(config)
        self.cortex = DistributedCorticalPrototypeNetwork(config)
        self.control = PrefrontalParietalControlNetwork(config.attention_temperature)
        self.feedback = NeuromodulatoryFeedbackGate()

    def snapshot(self) -> dict[str, Any]:
        """Return a distributed network state with explicit biological-scope cautions."""
        return {
            "sensory_association_network": {"model": "multimodal_feature_encoding_surrogate"},
            "hippocampal_episode_network": self.hippocampus.snapshot(),
            "distributed_cortical_network": self.cortex.snapshot(),
            "prefrontal_parietal_control": self.control.snapshot(),
            "neuromodulatory_feedback_gate": self.feedback.snapshot(),
            "anatomical_mapping": "functional correspondences are task-dependent, distributed, and non-exclusive",
        }
