"""Biologically motivated network surrogates for commonality and concept formation.

The models represent functional interactions among attention/control, hippocampal episode retrieval,
and distributed cortical populations. They are coarse computational surrogates, not anatomical
digital twins or direct measurements of neural activity.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import ConceptLearningConfig, EpisodeInput
from .functional import (
    canonical_features,
    competitive_attention_weights,
    retrieve_related_episodes,
)


class AttentionControlNetwork:
    """Represent goal- and prior-sensitive gain modulation in prefrontal/parietal control networks."""

    def __init__(self, temperature: float = 1.0) -> None:
        """Initialize the competitive feature-selection temperature."""
        self.temperature = float(temperature)
        self.last_gains: dict[str, float] = {}

    def select_feature_gains(
        self,
        feature_names: Sequence[str],
        attention: Mapping[str, float],
        goals: Mapping[str, float],
        prior: Mapping[str, float],
    ) -> dict[str, float]:
        """Compute competitive feature gains from task attention, goal relevance, and prior salience."""
        gains = competitive_attention_weights(
            feature_names, attention, goals, prior, self.temperature
        )
        self.last_gains = dict(gains)
        return gains

    def snapshot(self) -> dict[str, Any]:
        """Return the latest feature gain profile and its control-network interpretation."""
        return {
            "model": "prefrontal_parietal_attention_gain_surrogate",
            "temperature": self.temperature,
            "feature_gains": dict(self.last_gains),
            "interpretation": (
                "Feature gains model task-dependent modulation; they are not neuronal firing rates."
            ),
        }


class HippocampalEpisodeNetwork:
    """Maintain episode-specific memory indices and retrieve related experiences by partial cues."""

    def __init__(self, config: ConceptLearningConfig) -> None:
        """Initialize an episode index and the configured retrieval capacity."""
        self.config = config
        self.episode_index: dict[str, EpisodeInput] = {}
        self.retrieval_count = 0

    def register_episodes(self, episodes: Sequence[EpisodeInput]) -> int:
        """Register new episode identities without collapsing their unique contextual records."""
        added = 0
        for episode in episodes:
            existing = self.episode_index.get(episode.episode_id)
            if existing is not None and existing.to_dict() != episode.to_dict():
                raise ValueError("episode_id already exists with different episode content")
            if existing is None:
                self.episode_index[episode.episode_id] = episode
                added += 1
        return added

    def retrieve(
        self,
        current: EpisodeInput,
        aliases: Mapping[str, str],
    ) -> list[dict[str, object]]:
        """Reactivate candidate past episodes using shared observed features and typed relations."""
        self.retrieval_count += 1
        return retrieve_related_episodes(
            current,
            tuple(self.episode_index.values()),
            aliases,
            limit=self.config.max_related_episodes,
        )

    def snapshot(self) -> dict[str, Any]:
        """Return the episode count and retrieval activity without exporting the raw memories."""
        return {
            "model": "hippocampal_episode_index_surrogate",
            "episode_count": len(self.episode_index),
            "retrieval_count": self.retrieval_count,
            "episode_identity_preserved": True,
        }


class DistributedCorticalConceptNetwork:
    """Model gradual shared-pattern learning in distributed sensory/association cortical populations.

    Feature-to-unit and recurrent weights are rate-model parameters. Hebbian-style co-activation
    updates capture gradual plasticity, while episode IDs ensure duplicate requests do not
    repeatedly train the same evidence. The model does not implement prototype/exemplar theories,
    an anterior-temporal semantic hub, or schema/analogy operations.
    """

    def __init__(self, config: ConceptLearningConfig) -> None:
        """Initialize deterministic recurrent-population controls and empty learned state."""
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)
        self.feature_names: tuple[str, ...] = ()
        self.input_to_cortex = np.zeros((0, config.cortical_units), dtype=float)
        self.recurrent_weights = np.zeros(
            (config.cortical_units, config.cortical_units), dtype=float
        )
        self.cortical_pattern = np.zeros(config.cortical_units, dtype=float)
        self.learned_episode_ids: set[str] = set()
        self.learning_updates = 0
        self.last_activity = np.zeros((0, config.cortical_units), dtype=float)

    def _initialize_recurrence(self) -> None:
        """Create a weak recurrent population with a bounded spectral radius."""
        units = self.config.cortical_units
        matrix = self.rng.normal(0.0, 1.0, (units, units))
        matrix = 0.5 * (matrix + matrix.T)
        np.fill_diagonal(matrix, 0.0)
        radius = float(np.max(np.abs(np.linalg.eigvals(matrix))))
        if radius > 1e-12:
            matrix *= self.config.recurrent_gain / radius
        self.recurrent_weights = matrix.astype(float)

    def _align_features(self, feature_names: Sequence[str]) -> None:
        """Expand or reorder input weights while preserving columns for previously learned features."""
        requested = tuple(feature_names)
        if requested == self.feature_names:
            return
        previous = {
            name: self.input_to_cortex[index].copy()
            for index, name in enumerate(self.feature_names)
        }
        weights = self.rng.normal(
            0.0, 0.12, (len(requested), self.config.cortical_units)
        )
        for index, name in enumerate(requested):
            if name in previous:
                weights[index] = previous[name]
        self.feature_names = requested
        self.input_to_cortex = weights
        if self.recurrent_weights.size == 0:
            self._initialize_recurrence()

    def _encode_matrix(
        self,
        episodes: Sequence[EpisodeInput],
        aliases: Mapping[str, str],
        means: Mapping[str, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build a masked episode-by-feature matrix and reliability vector."""
        matrix = np.zeros((len(episodes), len(self.feature_names)), dtype=float)
        reliability = np.asarray([episode.reliability for episode in episodes], dtype=float)
        for row_index, episode in enumerate(episodes):
            values = canonical_features(episode, aliases)
            for column_index, feature in enumerate(self.feature_names):
                value = values.get(feature)
                # An unknown observation contributes the population's current expected value,
                # not a false zero that would bias the learned commonality estimate.
                matrix[row_index, column_index] = (
                    float(value) if value is not None else float(means.get(feature, 0.5))
                )
        return matrix, reliability

    def _settle(self, drive: np.ndarray) -> np.ndarray:
        """Iterate recurrent rate activity until a bounded population representation is obtained."""
        activity = np.zeros((drive.shape[0], self.config.cortical_units), dtype=float)
        for _ in range(self.config.recurrent_iterations):
            recurrent = activity @ self.recurrent_weights
            candidate = np.tanh(drive + recurrent)
            activity = 0.55 * activity + 0.45 * candidate
        return activity

    def learn(
        self,
        episodes: Sequence[EpisodeInput],
        aliases: Mapping[str, str],
        feature_means: Mapping[str, float],
        modulatory_gate: float = 1.0,
    ) -> dict[str, Any]:
        """Update cortical association weights from new episodes under an optional feedback gate."""
        gate = max(0.0, min(1.0, float(modulatory_gate)))
        if not episodes:
            return {
                "new_episode_count": 0,
                "learning_update_performed": False,
                "cortical_pattern": self.cortical_pattern.tolist(),
                "activation_stability": 1.0,
            }
        names = sorted(set(self.feature_names) | set(feature_means))
        self._align_features(names)
        matrix, reliability = self._encode_matrix(episodes, aliases, feature_means)
        centered = 2.0 * matrix - 1.0
        drive = centered @ self.input_to_cortex
        activity = self._settle(drive)
        new_indices = [
            index for index, episode in enumerate(episodes)
            if episode.episode_id not in self.learned_episode_ids
        ]
        old_pattern = self.cortical_pattern.copy()
        if new_indices:
            new_activity = activity[new_indices]
            new_input = centered[new_indices]
            new_reliability = reliability[new_indices]
            total_weight = max(float(np.sum(new_reliability)), 1e-12)
            coactivation = (
                new_input.T @ (new_activity * new_reliability[:, None])
            ) / total_weight
            self.input_to_cortex += (self.config.learning_rate * gate) * (
                coactivation - self.config.weight_decay * self.input_to_cortex
            )
            correlation = (
                new_activity.T @ (new_activity * new_reliability[:, None])
            ) / total_weight
            correlation -= np.diag(np.diag(correlation))
            self.recurrent_weights += 0.15 * self.config.learning_rate * gate * correlation
            self.recurrent_weights = 0.5 * (
                self.recurrent_weights + self.recurrent_weights.T
            )
            np.fill_diagonal(self.recurrent_weights, 0.0)
            radius = float(np.max(np.abs(np.linalg.eigvals(self.recurrent_weights))))
            if radius > self.config.recurrent_gain > 0.0:
                self.recurrent_weights *= self.config.recurrent_gain / radius
            self.learned_episode_ids.update(
                episodes[index].episode_id for index in new_indices
            )
            self.learning_updates += 1
            # Re-settle after plasticity so the output reflects the updated cortical weights.
            activity = self._settle(centered @ self.input_to_cortex)
        total = float(np.sum(reliability))
        if total > 0.0:
            self.cortical_pattern = np.sum(
                activity * reliability[:, None], axis=0
            ) / total
        self.last_activity = activity
        old_norm = float(np.linalg.norm(old_pattern))
        new_norm = float(np.linalg.norm(self.cortical_pattern))
        if old_norm <= 1e-12 or new_norm <= 1e-12:
            stability = 1.0 if old_norm <= 1e-12 and new_norm <= 1e-12 else 0.0
        else:
            stability = float(np.dot(old_pattern, self.cortical_pattern) / (old_norm * new_norm))
            stability = max(-1.0, min(1.0, stability))
        return {
            "new_episode_count": len(new_indices),
            "learning_update_performed": bool(new_indices),
            "learning_update_count": self.learning_updates,
            "population_size": self.config.cortical_units,
            "mean_absolute_input_weight": float(np.mean(np.abs(self.input_to_cortex))),
            "mean_absolute_recurrent_weight": float(np.mean(np.abs(self.recurrent_weights))),
            "cortical_pattern": self.cortical_pattern.tolist(),
            "activation_stability": stability,
            "rate_population_activity": activity.tolist(),
            "learning_rule": "reliability-weighted Hebbian co-activation with decay and recurrent normalization",
            "modulatory_feedback_gate": gate,
        }

    def snapshot(self) -> dict[str, Any]:
        """Return compact cortical population metadata and plasticity history."""
        return {
            "model": "distributed_recurrent_cortical_rate_population",
            "feature_names": list(self.feature_names),
            "population_size": self.config.cortical_units,
            "learned_episode_count": len(self.learned_episode_ids),
            "learning_update_count": self.learning_updates,
            "cortical_pattern": self.cortical_pattern.tolist(),
            "interpretation": (
                "The network is a distributed population surrogate, not a neuron-level simulator "
                "or a prototype/exemplar representation."
            ),
        }


class CorticostriatalFeedbackNetwork:
    """Represent optional feedback-dependent corticostriatal and dopaminergic plasticity gating."""

    def __init__(self, threshold: float = 0.25, slope: float = 8.0) -> None:
        """Initialize the prediction-error sensitivity used to estimate feedback teaching gates."""
        self.threshold = float(threshold)
        self.slope = float(slope)
        self.last_gate = 0.0
        self.update_count = 0

    def compute_learning_gate(
        self, prediction_error: float, reliability: float, feedback_available: bool
    ) -> dict[str, float | bool]:
        """Convert labelled prediction error and evidence reliability into a bounded teaching gate."""
        if not feedback_available:
            self.last_gate = 0.0
            return {
                "feedback_available": False,
                "prediction_error": max(0.0, min(1.0, float(prediction_error))),
                "reliability": max(0.0, min(1.0, float(reliability))),
                "learning_gate": 0.0,
            }
        error = max(0.0, min(1.0, float(prediction_error)))
        confidence = max(0.0, min(1.0, float(reliability)))
        activation = 1.0 / (
            1.0 + np.exp(-np.clip(self.slope * (error - self.threshold), -40.0, 40.0))
        )
        self.last_gate = float(activation) * confidence
        self.update_count += 1
        return {
            "feedback_available": True,
            "prediction_error": error,
            "reliability": confidence,
            "learning_gate": self.last_gate,
        }

    def snapshot(self) -> dict[str, Any]:
        """Return the current feedback-gating state and its computational interpretation."""
        return {
            "model": "corticostriatal_dopaminergic_feedback_gate_surrogate",
            "threshold": self.threshold,
            "slope": self.slope,
            "last_gate": self.last_gate,
            "update_count": self.update_count,
            "interpretation": (
                "The gate models optional error/reward-dependent modulation; it is not a DA measurement."
            ),
        }


class BiologicalCommonalityNetwork:
    """Coordinate attention control, hippocampal episode retrieval, and cortical shared-pattern learning."""

    def __init__(self, config: ConceptLearningConfig) -> None:
        """Construct deterministic attention, episodic-memory, and recurrent-cortex surrogates."""
        self.config = config
        self.attention = AttentionControlNetwork()
        self.hippocampus = HippocampalEpisodeNetwork(config)
        self.cortex = DistributedCorticalConceptNetwork(config)
        self.feedback = CorticostriatalFeedbackNetwork()
        self.concept_updates = 0

    def register_episodes(self, episodes: Sequence[EpisodeInput]) -> int:
        """Preserve episode-specific records for retrieval and future concept revision."""
        return self.hippocampus.register_episodes(episodes)

    def learn_cortical_commonality(
        self,
        episodes: Sequence[EpisodeInput],
        aliases: Mapping[str, str],
        feature_means: Mapping[str, float],
        modulatory_gate: float = 1.0,
    ) -> dict[str, Any]:
        """Train and settle distributed cortical populations using reliability and feedback gates."""
        self.concept_updates += 1
        return self.cortex.learn(
            episodes, aliases, feature_means, modulatory_gate=modulatory_gate
        )

    def estimate_feedback_gates(
        self,
        mismatch_rows: Sequence[Mapping[str, Any]],
        predictions: Sequence[Mapping[str, Any]],
        episodes: Sequence[EpisodeInput],
    ) -> dict[str, Any]:
        """Estimate optional teaching gates only for held-out examples with known category labels."""
        mismatch_by_id = {str(row.get("episode_id")): row for row in mismatch_rows}
        prediction_by_id = {str(row.get("episode_id")): row for row in predictions}
        gates = []
        for episode in episodes:
            if episode.category_label is None:
                continue
            prediction = prediction_by_id.get(episode.episode_id, {})
            mismatch = mismatch_by_id.get(episode.episode_id, {})
            posterior = prediction.get("posterior_category_probability")
            if posterior is None:
                error = float(mismatch.get("mismatch_score", 0.0))
            else:
                error = abs(float(posterior) - float(episode.category_label))
                if prediction.get("predicted_category") is not None and (
                    bool(prediction["predicted_category"]) != episode.category_label
                ):
                    error = max(error, 1.0)
            gate = self.feedback.compute_learning_gate(
                error, episode.reliability, feedback_available=True
            )
            gates.append({"episode_id": episode.episode_id, **gate})
        mean_gate = (
            sum(float(row["learning_gate"]) for row in gates) / len(gates)
            if gates else 0.0
        )
        return {
            "per_episode_gates": gates,
            "mean_learning_gate": mean_gate,
            "labelled_feedback_count": len(gates),
            "unlabelled_mismatch_is_not_a_teaching_signal": True,
        }

    def snapshot(self) -> dict[str, Any]:
        """Return subsystem states while keeping the anatomical and computational caveats explicit."""
        return {
            "model_type": "functional_biological_surrogate",
            "attention_control": self.attention.snapshot(),
            "hippocampal_episode_network": self.hippocampus.snapshot(),
            "distributed_cortical_network": self.cortex.snapshot(),
            "corticostriatal_feedback_network": self.feedback.snapshot(),
            "concept_update_count": self.concept_updates,
            "network_interpretation": (
                "Sensory/association cortex, hippocampal episode retrieval, and control systems "
                "interact; no single region is treated as a concept centre."
            ),
        }
