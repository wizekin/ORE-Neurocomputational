"""Biologically motivated hippocampal indexing and distributed cortical semantic populations.

This module models two linked but non-identical representational supports: sparse event-specific
hippocampal index patterns and distributed semantic population patterns. Projection vectors and
attractor updates are software surrogates, not an anatomically fitted neural simulation.
"""
from __future__ import annotations

import hashlib
from typing import Mapping, Sequence

import numpy as np

from .contracts import EpisodeInput, FEATURE_GROUPS, MemoryTransformationConfig
from .functional import normalize_nested_features


def _normalise(vector: np.ndarray) -> np.ndarray:
    """Return an L2-normalized vector, preserving an all-zero pattern."""
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else np.zeros_like(vector)


def _stable_seed(token: str, seed: int) -> int:
    """Derive a repeatable projection seed from a semantic token and configured seed."""
    digest = hashlib.blake2b(f"{seed}:{token}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False) % (2 ** 32)


class HippocampalIndexNetwork:
    """Store event-specific index patterns and complete them from partial feature cues.

    Distinct event indices remain separate even when their feature content overlaps. A recurrent
    population matrix provides an attractor-like pattern-completion surrogate; exact event details
    remain in the episodic store and are never replaced by the lower-dimensional index.
    """

    def __init__(self, config: MemoryTransformationConfig) -> None:
        """Initialize deterministic feature projections, recurrent connectivity, and event indices."""
        self.config = config
        units = config.hippocampal_units
        rng = np.random.default_rng(config.random_seed)
        recurrent = rng.normal(0.0, 0.10 / np.sqrt(units), size=(units, units))
        recurrent = 0.5 * (recurrent + recurrent.T)
        recurrent *= rng.random(recurrent.shape) < 0.28
        np.fill_diagonal(recurrent, 0.0)
        self.recurrent_weights = recurrent
        self.feature_projections: dict[str, np.ndarray] = {}
        self.event_indices: dict[str, np.ndarray] = {}
        self.update_count = 0

    def _projection(self, token: str) -> np.ndarray:
        """Return a stable random population projection for one feature or relation token."""
        if token not in self.feature_projections:
            rng = np.random.default_rng(_stable_seed(token, self.config.random_seed))
            vector = rng.normal(0.0, 1.0, self.config.hippocampal_units)
            self.feature_projections[token] = _normalise(vector)
        return self.feature_projections[token]

    def encode_episode(
        self, feature_groups: Mapping[str, Mapping[str, float]], relations: Sequence[object]
    ) -> np.ndarray:
        """Converge feature and typed-relation inputs into one event-specific hippocampal index."""
        groups = normalize_nested_features(feature_groups)
        drive = np.zeros(self.config.hippocampal_units, dtype=np.float64)
        for group in FEATURE_GROUPS:
            for feature, value in groups[group].items():
                token = f"feature:{group}:{feature}"
                drive += float(value) * self._projection(token)
        for relation in relations:
            token = (f"relation:{getattr(relation, 'subject')}|"
                     f"{getattr(relation, 'predicate')}|{getattr(relation, 'object')}")
            drive += float(getattr(relation, "confidence", 1.0)) * self._projection(token)
        if not np.any(drive):
            return drive
        activity = np.tanh(1.7 * drive)
        k = max(1, int(np.ceil(self.config.sparse_activity_fraction * activity.size)))
        active = np.argpartition(np.abs(activity), -k)[-k:]
        sparse = np.zeros_like(activity)
        sparse[active] = activity[active]
        return _normalise(sparse)

    def register(self, episode: EpisodeInput) -> dict[str, object]:
        """Create an event-specific hippocampal index once and reject content replacement."""
        pattern = self.encode_episode(episode.feature_groups(), episode.relations)
        existing = self.event_indices.get(episode.episode_id)
        if existing is not None:
            if not np.allclose(existing, pattern):
                raise ValueError("episode_id is already registered with different index content")
            return {"episode_id": episode.episode_id, "status": "already_indexed",
                    "index_norm": float(np.linalg.norm(existing))}
        self.event_indices[episode.episode_id] = pattern
        self.update_count += 1
        return {"episode_id": episode.episode_id, "status": "indexed",
                "index_norm": float(np.linalg.norm(pattern)),
                "active_fraction": float(np.mean(np.abs(pattern) > 1e-12))}

    def complete(
        self,
        episode_id: str,
        cue_features: Mapping[str, Mapping[str, float]],
        cue_relations: Sequence[object],
    ) -> tuple[np.ndarray, dict[str, object]]:
        """Reinstate a stored event pattern from partial cue input using recurrent settling."""
        if episode_id not in self.event_indices:
            raise KeyError(f"unknown episode_id: {episode_id}")
        stored = self.event_indices[episode_id]
        cue = self.encode_episode(cue_features, cue_relations) if cue_features or cue_relations else stored
        similarity_before = self._similarity(stored, cue)
        activity = _normalise(0.60 * stored + 0.40 * cue)
        for _ in range(7):
            recurrent_drive = self.recurrent_weights @ activity
            inhibition = 0.20 * float(np.mean(np.maximum(recurrent_drive, 0.0)))
            settled = np.tanh(1.35 * activity + recurrent_drive - inhibition)
            activity = _normalise(0.60 * activity + 0.40 * settled)
        similarity_after = self._similarity(stored, activity)
        return activity, {"initial_cue_similarity": similarity_before,
                          "completed_pattern_similarity": similarity_after,
                          "settling_iterations": 7,
                          "stored_event_index_preserved": True}

    def _similarity(self, left: np.ndarray, right: np.ndarray) -> float:
        """Compute cosine overlap for same-width hippocampal population patterns."""
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        return float(np.dot(left, right) / denominator) if denominator > 1e-12 else 0.0

    def snapshot(self) -> dict[str, object]:
        """Return index-store size, projection count, recurrent strength, and registration count."""
        return {"event_index_count": len(self.event_indices),
                "feature_projection_count": len(self.feature_projections),
                "recurrent_weight_norm": float(np.linalg.norm(self.recurrent_weights)),
                "index_update_count": self.update_count}


class DistributedSemanticNetwork:
    """Maintain distributed cortical-like population patterns for generalized concepts.

    Feature and relation tokens project into shared population units. A sparse activity transform
    and slow prototype update approximate distributed semantic integration, while concept IDs
    remain distinct so a high overlap does not automatically collapse two categories.
    """

    def __init__(self, config: MemoryTransformationConfig) -> None:
        """Initialize deterministic feature projections and separate concept-level cortical patterns."""
        self.config = config
        self.feature_projections: dict[str, np.ndarray] = {}
        self.concept_patterns: dict[str, np.ndarray] = {}
        self.concept_update_counts: dict[str, int] = {}
        self.association_strengths: dict[tuple[str, str], float] = {}

    def _projection(self, token: str) -> np.ndarray:
        """Return a stable distributed cortical projection for a feature or typed relation."""
        if token not in self.feature_projections:
            rng = np.random.default_rng(_stable_seed(token, self.config.random_seed + 101))
            self.feature_projections[token] = _normalise(
                rng.normal(0.0, 1.0, self.config.semantic_units)
            )
        return self.feature_projections[token]

    def encode_gist(
        self,
        features: Mapping[str, Mapping[str, float]],
        relations: Mapping[str, Mapping[str, object]],
        context_weights: Mapping[str, float],
    ) -> tuple[np.ndarray, dict[str, object]]:
        """Transform weighted gist features and recurring relations into sparse cortical activity."""
        groups = normalize_nested_features(features)
        drive = np.zeros(self.config.semantic_units, dtype=np.float64)
        contribution_count = 0
        for group in FEATURE_GROUPS:
            for feature, value in groups[group].items():
                key = f"{group}:{feature}"
                weight = float(context_weights.get(key, 1.0))
                drive += float(value) * weight * self._projection(f"feature:{key}")
                contribution_count += 1
        for key, relation in relations.items():
            confidence = float(relation.get("mean_reliability", 1.0))
            prevalence = float(relation.get("prevalence", 1.0))
            token = f"relation:{key}"
            drive += confidence * prevalence * self._projection(token)
            contribution_count += 1
        if contribution_count == 0:
            return np.zeros(self.config.semantic_units, dtype=np.float64), {
                "status": "no_supported_semantic_features", "active_fraction": 0.0}
        inhibition = 0.18 * float(np.mean(np.maximum(drive, 0.0)))
        activity = np.tanh(1.4 * (drive - inhibition))
        k = max(1, int(np.ceil(self.config.sparse_activity_fraction * activity.size)))
        active = np.argpartition(np.abs(activity), -k)[-k:]
        sparse = np.zeros_like(activity)
        sparse[active] = activity[active]
        return _normalise(sparse), {
            "status": "distributed_cortical_pattern_created",
            "feature_relation_contribution_count": contribution_count,
            "active_unit_count": int(np.count_nonzero(sparse)),
            "active_fraction": float(np.mean(np.abs(sparse) > 1e-12)),
            "population_norm": float(np.linalg.norm(sparse)),
        }

    def integrate(
        self, concept_id: str, activity: Sequence[float], learning_rate: float | None = None
    ) -> dict[str, object]:
        """Integrate a new semantic pattern by a bounded prototype update without merging concepts."""
        vector = np.asarray(activity, dtype=np.float64).reshape(-1)
        if vector.size != self.config.semantic_units or not np.all(np.isfinite(vector)):
            raise ValueError("semantic activity must be finite and match semantic_units")
        vector = _normalise(vector)
        rate = self.config.semantic_learning_rate if learning_rate is None else float(learning_rate)
        if not 0.0 <= rate <= 1.0:
            raise ValueError("learning_rate must lie in [0, 1]")
        previous = self.concept_patterns.get(concept_id)
        if previous is None:
            updated = vector.copy()
            similarity_before = None
            count = 1
        else:
            similarity_before = self._cosine(previous, vector)
            count = self.concept_update_counts[concept_id] + 1
            updated = _normalise((1.0 - rate) * previous + rate * vector)
        self.concept_patterns[concept_id] = updated
        self.concept_update_counts[concept_id] = count
        return {"concept_id": concept_id, "status": "integrated",
                "similarity_before_update": similarity_before,
                "prototype_norm": float(np.linalg.norm(updated)),
                "support_update_count": count,
                "prototype_changed": bool(previous is None or not np.allclose(previous, updated))}

    def link_concepts(self, concept_id: str, other_id: str, similarity: float) -> None:
        """Store a symmetric graded schema association without merging concept identities."""
        if concept_id == other_id:
            return
        self.association_strengths[(concept_id, other_id)] = float(similarity)
        self.association_strengths[(other_id, concept_id)] = float(similarity)

    def get_pattern(self, concept_id: str) -> np.ndarray | None:
        """Return a defensive copy of one stored concept pattern, or None when unknown."""
        pattern = self.concept_patterns.get(concept_id)
        return None if pattern is None else pattern.copy()

    def _cosine(self, left: np.ndarray, right: np.ndarray) -> float:
        """Compute cosine similarity for two distributed concept patterns."""
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        return float(np.dot(left, right) / denominator) if denominator > 1e-12 else 0.0

    def snapshot(self) -> dict[str, object]:
        """Return concept count, population size, and learned associative-link diagnostics."""
        return {"concept_pattern_count": len(self.concept_patterns),
                "feature_projection_count": len(self.feature_projections),
                "association_link_count": len(self.association_strengths),
                "semantic_population_size": self.config.semantic_units,
                "concept_update_counts": dict(self.concept_update_counts)}
