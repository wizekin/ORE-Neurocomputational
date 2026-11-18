"""Executable 21-stage Hub-and-Spoke semantic processing and lesion model.

The functional pipeline and current-based LIF reference are separate evidence
tracks. All numerical equations are configurable computational analogies, not
measured neural constants or literal descriptions of neuronal arithmetic.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence
import math
import re
import unicodedata

import numpy as np

from ore_semantic.contracts import SemanticConcept, WordFormEpisode

from .biology import (
    MODALITY_POPULATIONS,
    HubSpokeLIFSimulator,
    biological_drives,
)
from .contracts import (
    HubAndSpokeConfig,
    HubAndSpokeRequest,
    HubAndSpokeResult,
    LearningExperience,
    MODALITIES,
    canonical_modality,
)
from .registry import ALL_OPERATIONS, STAGE_REGISTRY, StageSpec

_EPS = 1e-9


def _normalize_text(value: str) -> str:
    """Normalize text for lexical matching while leaving the request surface unchanged."""
    normalized = unicodedata.normalize("NFKC", value).casefold().strip()
    return re.sub(r"^\W+|\W+$", "", normalized, flags=re.UNICODE)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Apply a numerically stable logistic activation elementwise."""
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _softmax(values: Sequence[float], temperature: float = 1.0) -> np.ndarray:
    """Normalize finite modality or candidate scores into a probability distribution."""
    vector = np.asarray(values, dtype=np.float64)
    if vector.size == 0:
        return vector
    scaled = vector / max(float(temperature), _EPS)
    scaled -= float(np.max(scaled))
    exponentials = np.exp(np.clip(scaled, -60.0, 60.0))
    total = float(np.sum(exponentials))
    if total <= _EPS or not math.isfinite(total):
        return np.full(vector.size, 1.0 / vector.size, dtype=np.float64)
    return exponentials / total


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    """Compute cosine similarity with a defined zero-vector result."""
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= _EPS:
        return 0.0
    return float(np.dot(left, right) / denominator)


def _map_cosine(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    """Compute cosine similarity over the union of two sparse feature maps."""
    keys = sorted(set(left) | set(right))
    if not keys:
        return 0.0
    a = np.asarray([float(left.get(key, 0.0)) for key in keys], dtype=np.float64)
    b = np.asarray([float(right.get(key, 0.0)) for key in keys], dtype=np.float64)
    return _cosine(a, b)


def _string_similarity(left: str, right: str) -> float:
    """Calculate normalized edit similarity for short concept labels and aliases."""
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, char_left in enumerate(left, 1):
        current = [i]
        for j, char_right in enumerate(right, 1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (char_left != char_right),
            ))
        previous = current
    return float(max(0.0, 1.0 - previous[-1] / max(len(left), len(right), 1)))


def _safe_vector(value: Mapping[str, float], vocabulary: Sequence[str]) -> np.ndarray:
    """Encode a sparse named feature map in a stable, explicitly ordered vector space."""
    index = {name: position for position, name in enumerate(vocabulary)}
    vector = np.zeros(len(vocabulary), dtype=np.float64)
    for feature, strength in value.items():
        position = index.get(str(feature))
        if position is not None:
            vector[position] = max(vector[position], float(strength))
    return vector


def _plain(value: Any) -> Any:
    """Convert nested numerical state into JSON-compatible values for traces."""
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, set):
        return sorted(_plain(item) for item in value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class HubAndSpokeProcessor:
    """Run the 21-stage semantic algorithm over concepts and modality-specific features."""

    def __init__(
        self,
        concepts: Sequence[SemanticConcept],
        config: HubAndSpokeConfig | None = None,
        associations: Mapping[str, Mapping[str, float]] | None = None,
    ) -> None:
        """Pin the concept inventory and initialize spoke, hub, feedback, and relation weights."""
        self.config = config or HubAndSpokeConfig()
        self.concepts = tuple(concepts)
        self._concept_by_id = {item.concept_id: item for item in self.concepts}
        if len(self._concept_by_id) != len(self.concepts):
            raise ValueError("concept_id values must be unique")
        self.associations = self._validate_associations(associations or {})
        self._rng = np.random.default_rng(self.config.random_seed)
        self._experience_memory: dict[str, dict[str, dict[str, float]]] = {}
        self._feature_probability: dict[str, dict[str, float]] = {}
        self._vocabulary: dict[str, list[str]] = {}
        self._input_weights: dict[str, np.ndarray] = {}
        self._spoke_to_hub: dict[str, np.ndarray] = {}
        self._hub_to_spoke: dict[str, np.ndarray] = {}
        self._feature_associations: dict[str, np.ndarray] = {}
        self._initial_spoke_to_hub: dict[str, np.ndarray] = {}
        self._hub_w1 = self._rng.normal(
            0.0, 1.0 / math.sqrt(self.config.hidden_dim),
            (self.config.hidden_dim, self.config.hidden_dim),
        )
        self._hub_w2 = self._rng.normal(
            0.0, 1.0 / math.sqrt(self.config.hidden_dim),
            (self.config.hidden_dim, self.config.hidden_dim),
        )
        self._hub_bias1 = np.zeros(self.config.hidden_dim, dtype=np.float64)
        self._hub_bias2 = np.zeros(self.config.hidden_dim, dtype=np.float64)
        self._recurrent = (
            0.32 * np.eye(self.config.hidden_dim)
            + self._rng.normal(0.0, 0.015, (self.config.hidden_dim, self.config.hidden_dim))
        )
        self._initialize_vocabulary()
        self._initialize_connections()
        self.model_generation = 0

    def _validate_associations(
        self, associations: Mapping[str, Mapping[str, float]]
    ) -> dict[str, dict[str, float]]:
        """Validate explicit associative links without inferring unprovided real-world facts."""
        validated: dict[str, dict[str, float]] = {}
        for source, targets in associations.items():
            if source not in self._concept_by_id:
                raise ValueError(f"association source is not a known concept: {source}")
            validated[source] = {}
            for target, raw in targets.items():
                strength = float(raw)
                if target not in self._concept_by_id:
                    raise ValueError(f"association target is not a known concept: {target}")
                if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
                    raise ValueError("association strengths must lie in [0, 1]")
                validated[source][target] = strength
        return validated

    def _concept_modality_features(
        self, concept: SemanticConcept, modality: str
    ) -> dict[str, float]:
        """Combine shared semantic properties with the selected modality-specific spoke."""
        features = {
            str(key): float(value) * 0.25
            for key, value in concept.semantic_features.items()
        }
        for raw_modality, raw_features in concept.modality_features.items():
            try:
                canonical = canonical_modality(raw_modality)
            except ValueError:
                continue
            if canonical == modality:
                for key, value in raw_features.items():
                    features[str(key)] = max(features.get(str(key), 0.0), float(value))
        if modality == "linguistic":
            for form in (concept.label, *concept.aliases):
                key = f"WORD:{_normalize_text(form)}"
                if key != "WORD:":
                    features[key] = 1.0
        return features

    def _initialize_vocabulary(self) -> None:
        """Build deterministic per-spoke feature vocabularies from the pinned concept inventory."""
        for modality in MODALITIES:
            names: set[str] = set()
            for concept in self.concepts:
                names.update(self._concept_modality_features(concept, modality))
            self._vocabulary[modality] = sorted(names)

    def _initialize_connections(self) -> None:
        """Initialize input, spoke-to-hub, feedback, and co-occurrence matrices."""
        hidden = self.config.hidden_dim
        for modality in MODALITIES:
            dimension = len(self._vocabulary[modality])
            self._input_weights[modality] = 2.4 * np.eye(dimension, dtype=np.float64)
            self._spoke_to_hub[modality] = self._rng.normal(
                0.0, 1.0 / math.sqrt(max(dimension, 1)), (hidden, dimension)
            )
            self._hub_to_spoke[modality] = self._spoke_to_hub[modality].T.copy()
            self._feature_associations[modality] = np.zeros((dimension, dimension), dtype=np.float64)
            self._initial_spoke_to_hub[modality] = self._spoke_to_hub[modality].copy()
        self._rebuild_feature_associations()

    def _rebuild_feature_associations(self) -> None:
        """Estimate within-spoke co-activation from stored concept feature patterns."""
        for modality in MODALITIES:
            dimension = len(self._vocabulary[modality])
            if not dimension or not self.concepts:
                self._feature_associations[modality] = np.zeros((dimension, dimension))
                continue
            rows = [
                _safe_vector(self._concept_modality_features(item, modality), self._vocabulary[modality])
                for item in self.concepts
            ]
            matrix = np.vstack(rows)
            association = matrix.T @ matrix
            np.fill_diagonal(association, 0.0)
            scale = float(np.max(association)) if association.size else 0.0
            if scale > _EPS:
                association /= scale
            self._feature_associations[modality] = association

    def _ensure_vocabulary(
        self,
        request: HubAndSpokeRequest,
        experiences: Sequence[LearningExperience],
    ) -> None:
        """Expand feature dimensions for new observations while preserving learned matrix columns."""
        additional: dict[str, set[str]] = {modality: set() for modality in MODALITIES}
        normalized_inputs = self._normalized_inputs(request)
        for modality, features in normalized_inputs.items():
            additional[modality].update(features)
        for modality, features in self._normalized_recalled_inputs(request).items():
            additional[modality].update(features)
        for modality, features in _normalize_episode_features(request.episodes).items():
            additional[modality].update(features)
        for experience in experiences:
            for raw_modality, features in experience.modality_features.items():
                additional[canonical_modality(raw_modality)].update(features)
            for modality in MODALITIES:
                additional[modality].update(experience.semantic_features)
                additional[modality].update(
                    f"WORD:{_normalize_text(self._concept_by_id[experience.concept_id].label)}"
                    for _ in (0,)
                    if experience.concept_id in self._concept_by_id and modality == "linguistic"
                )
        cue = _normalize_text(request.concept_cue)
        if cue:
            additional["linguistic"].add(f"WORD:{cue}")
        for modality in MODALITIES:
            old = self._vocabulary[modality]
            expanded = sorted(set(old) | additional[modality])
            if expanded == old:
                continue
            old_index = {name: index for index, name in enumerate(old)}
            new_index = {name: index for index, name in enumerate(expanded)}
            old_size, new_size = len(old), len(expanded)
            input_weights = 2.4 * np.eye(new_size, dtype=np.float64)
            input_weights[:old_size, :old_size] = self._input_weights[modality]
            spoke_to_hub = self._rng.normal(
                0.0, 1.0 / math.sqrt(max(new_size, 1)),
                (self.config.hidden_dim, new_size),
            )
            hub_to_spoke = spoke_to_hub.T.copy()
            for feature, old_position in old_index.items():
                new_position = new_index[feature]
                spoke_to_hub[:, new_position] = self._spoke_to_hub[modality][:, old_position]
                hub_to_spoke[new_position, :] = self._hub_to_spoke[modality][old_position, :]
            self._vocabulary[modality] = expanded
            self._input_weights[modality] = input_weights
            self._spoke_to_hub[modality] = spoke_to_hub
            self._hub_to_spoke[modality] = hub_to_spoke
            associations = np.zeros((new_size, new_size), dtype=np.float64)
            for feature_i, old_i in old_index.items():
                for feature_j, old_j in old_index.items():
                    associations[new_index[feature_i], new_index[feature_j]] = (
                        self._feature_associations[modality][old_i, old_j]
                    )
            self._feature_associations[modality] = associations
            initial = np.zeros_like(spoke_to_hub)
            initial[:, :old_size] = self._initial_spoke_to_hub[modality]
            if new_size > old_size:
                initial[:, old_size:] = spoke_to_hub[:, old_size:]
            self._initial_spoke_to_hub[modality] = initial
        self._rebuild_feature_associations()

    def _normalized_inputs(self, request: HubAndSpokeRequest) -> dict[str, dict[str, float]]:
        """Canonicalize direct modality evidence and the preserved lexical cue only."""
        result: dict[str, dict[str, float]] = {}
        for raw_modality, features in request.modality_inputs.items():
            modality = canonical_modality(raw_modality)
            target = result.setdefault(modality, {})
            for feature, value in features.items():
                target[str(feature)] = max(target.get(str(feature), 0.0), float(value))
        cue = _normalize_text(request.concept_cue)
        if cue:
            result.setdefault("linguistic", {})[f"WORD:{cue}"] = 1.0
        return result

    def _normalized_recalled_inputs(
        self, request: HubAndSpokeRequest
    ) -> dict[str, dict[str, float]]:
        """Canonicalize internally reactivated features without relabelling them as direct observations."""
        result: dict[str, dict[str, float]] = {}
        for raw_modality, features in request.recalled_modality_inputs.items():
            modality = canonical_modality(raw_modality)
            target = result.setdefault(modality, {})
            for feature, value in features.items():
                target[str(feature)] = max(target.get(str(feature), 0.0), float(value))
        return result

    def _concept_candidates(self, cue: str, episodes: Sequence[WordFormEpisode]) -> list[dict[str, Any]]:
        """Retrieve lexical and episodic candidates while preserving non-identical alternatives."""
        normalized = _normalize_text(cue)
        episode_scores: dict[str, float] = defaultdict(float)
        episode_rows: list[dict[str, Any]] = []
        for episode in episodes:
            similarity = _string_similarity(normalized, _normalize_text(episode.orthography)) if normalized else 0.0
            strength = float(episode.strength) * float(episode.recency) * similarity
            episode_rows.append({
                "episode_id": episode.episode_id,
                "concept_id": episode.concept_id,
                "orthography": episode.orthography,
                "similarity": similarity,
                "retrieval_strength": strength,
                "source_reference": episode.source_reference,
            })
            if episode.concept_id and episode.concept_id in self._concept_by_id:
                episode_scores[episode.concept_id] = max(episode_scores[episode.concept_id], strength)
        rows: list[dict[str, Any]] = []
        for concept in self.concepts:
            lexical_similarity = max(
                (_string_similarity(normalized, _normalize_text(form))
                 for form in (concept.label, *concept.aliases)),
                default=0.0,
            ) if normalized else 0.0
            episode_fit = episode_scores.get(concept.concept_id, 0.0)
            if not normalized or lexical_similarity >= self.config.candidate_similarity_floor or episode_fit > 0:
                rows.append({
                    "concept": concept,
                    "lexical_similarity": lexical_similarity,
                    "episode_fit": episode_fit,
                    "episode_ids": [
                        item["episode_id"] for item in episode_rows
                        if item["concept_id"] == concept.concept_id and item["retrieval_strength"] > 0
                    ],
                })
        rows.sort(
            key=lambda item: (
                max(item["lexical_similarity"], item["episode_fit"]),
                item["lexical_similarity"],
                item["concept"].frequency,
            ),
            reverse=True,
        )
        return rows[:self.config.max_candidates]

    def _concept_hub(self, concept: SemanticConcept) -> np.ndarray:
        """Project one stored concept through its seven spokes to make its learned hub template."""
        projections = []
        for modality in MODALITIES:
            features = self._concept_modality_features(concept, modality)
            vector = _safe_vector(features, self._vocabulary[modality])
            projections.append(self._spoke_to_hub[modality] @ vector)
        combined = np.mean(np.vstack(projections), axis=0) if projections else np.zeros(self.config.hidden_dim)
        hidden = np.tanh(self._hub_w1 @ combined + self._hub_bias1)
        return np.tanh(self._hub_w2 @ hidden + self._hub_bias2)

    def _integrate_projections(
        self, projections: Mapping[str, np.ndarray], weights: Mapping[str, float]
    ) -> np.ndarray:
        """Combine spoke projections using the current reliability-weighted modality gates."""
        combined = np.zeros(self.config.hidden_dim, dtype=np.float64)
        for modality, projection in projections.items():
            combined += float(weights.get(modality, 0.0)) * projection
        hidden = np.tanh(self._hub_w1 @ combined + self._hub_bias1)
        return np.tanh(self._hub_w2 @ hidden + self._hub_bias2)

    def _score_candidates(
        self,
        hub: np.ndarray,
        retrieved: Sequence[Mapping[str, Any]],
        raw_inputs: Mapping[str, Mapping[str, float]],
        context: Mapping[str, float],
        selected_features: Mapping[str, float] | None = None,
    ) -> list[dict[str, Any]]:
        """Score candidates using separate lexical, hub, modality, and context evidence channels."""
        scores: list[dict[str, Any]] = []
        for row in retrieved:
            concept = row["concept"]
            prototype = self._concept_hub(concept)
            hub_similarity = _cosine(hub, prototype)
            hub_fit = float(np.clip((hub_similarity + 1.0) / 2.0, 0.0, 1.0))
            modality_fits = []
            for modality, evidence in raw_inputs.items():
                target = self._concept_modality_features(concept, modality)
                modality_fits.append(_map_cosine(evidence, target))
            modality_fit = float(np.mean(modality_fits)) if modality_fits else 0.0
            context_fit = _map_cosine(context, concept.semantic_features) if context else 0.0
            task_fit = (
                _map_cosine(selected_features, concept.semantic_features)
                if selected_features else 0.0
            )
            lexical = float(row["lexical_similarity"])
            episode = float(row["episode_fit"])
            score = (
                0.40 * hub_fit
                + 0.30 * lexical
                + 0.15 * modality_fit
                + 0.08 * context_fit
                + 0.04 * episode
                + 0.03 * task_fit
            )
            scores.append({
                "concept_id": concept.concept_id,
                "label": concept.label,
                "lexical_category": concept.lexical_category,
                "prototype_id": concept.prototype_id,
                "source_reference": concept.source_reference,
                "lexical_similarity": lexical,
                "hub_similarity": hub_similarity,
                "hub_fit": hub_fit,
                "modality_fit": modality_fit,
                "context_fit": context_fit,
                "episode_fit": episode,
                "task_fit": task_fit,
                "score": float(np.clip(score, 0.0, 1.0)),
                "semantic_features": dict(concept.semantic_features),
                "episode_ids": list(row["episode_ids"]),
            })
        scores.sort(key=lambda item: (item["score"], item["lexical_similarity"]), reverse=True)
        probabilities = _softmax([item["score"] for item in scores], temperature=0.12)
        for item, probability in zip(scores, probabilities):
            item["posterior"] = float(probability)
        return scores

    def _stage_01_receive(self, state: dict[str, Any]) -> dict[str, Any]:
        """Encode direct inputs separately from recalled cues and preserve both provenance streams."""
        raw = self._normalized_inputs(state["request"])
        recalled = self._normalized_recalled_inputs(state["request"])
        state["raw_inputs"] = raw
        state["direct_inputs"] = raw
        state["recalled_inputs"] = recalled
        state["available_modalities"] = sorted(raw)
        state["input_vectors"] = {
            modality: _safe_vector(raw.get(modality, {}), self._vocabulary[modality])
            for modality in MODALITIES
        }
        return {
            "available_modalities": list(state["available_modalities"]),
            "direct_modalities": list(state["available_modalities"]),
            "recalled_modalities": sorted(recalled),
            "all_available_modalities": sorted(set(raw) | set(recalled)),
            "input_channel_count": len(raw),
            "recalled_channel_count": len(recalled),
            "direct_evidence_preserved": True,
            "recalled_features_not_treated_as_direct_observations": True,
            "parallel_inputs_retained": True,
            "source_reference": state["request"].source_reference,
        }

    def _stage_02_activate_spokes(self, state: dict[str, Any]) -> dict[str, Any]:
        """Route inputs through modality-specific input transforms into graded spoke activity."""
        activations: dict[str, np.ndarray] = {}
        for modality in MODALITIES:
            vector = state["input_vectors"][modality]
            logits = self._input_weights[modality] @ vector - 1.2
            activations[modality] = _sigmoid(logits) if vector.size else vector.copy()
        state["spoke_activations"] = activations
        return {
            "spoke_dimensions": {key: int(value.size) for key, value in activations.items()},
            "active_spokes": [
                key for key, value in activations.items()
                if value.size and float(np.max(value)) > self.config.feature_threshold
            ],
            "graded_activity": True,
        }

    def _stage_03_extract_features(self, state: dict[str, Any]) -> dict[str, Any]:
        """Threshold feature activity, retain magnitudes, and expose learned co-activation relations."""
        features: dict[str, dict[str, float]] = {}
        association_summaries: dict[str, dict[str, float]] = {}
        for modality in MODALITIES:
            values = state["spoke_activations"][modality]
            vocabulary = self._vocabulary[modality]
            features[modality] = {
                name: float(value)
                for name, value in zip(vocabulary, values)
                if float(value) > self.config.feature_threshold
            }
            association_summaries[modality] = {
                "matrix_shape": [int(values.size), int(values.size)],
                "nonzero_edges": int(np.count_nonzero(self._feature_associations[modality])),
                "max_association": float(np.max(self._feature_associations[modality]))
                    if values.size else 0.0,
            }
        state["extracted_features"] = features
        return {
            "active_feature_count": sum(len(value) for value in features.values()),
            "feature_activations": features,
            "association_matrices": association_summaries,
        }

    def _stage_04_retrieve_memory(self, state: dict[str, Any]) -> dict[str, Any]:
        """Retrieve concepts and episodes, then blend current and remembered feature patterns."""
        request: HubAndSpokeRequest = state["request"]
        retrieved = self._concept_candidates(request.concept_cue, request.episodes)
        raw_evidence = self._normalized_inputs(request)
        if not request.concept_cue and raw_evidence:
            sensory_matches = []
            for row in retrieved:
                concept = row["concept"]
                fits = [
                    _map_cosine(evidence, self._concept_modality_features(concept, modality))
                    for modality, evidence in raw_evidence.items()
                ]
                input_fit = max(fits, default=0.0)
                if input_fit > 0.0:
                    row["input_fit"] = float(input_fit)
                    sensory_matches.append(row)
            retrieved = sorted(
                sensory_matches,
                key=lambda item: (item["input_fit"], item["concept"].frequency),
                reverse=True,
            )[:self.config.max_candidates]
        state["retrieved_candidates"] = retrieved
        episode_rows = []
        episode_memory: dict[str, float] = defaultdict(float)
        for episode in request.episodes:
            similarity = _string_similarity(
                _normalize_text(request.concept_cue), _normalize_text(episode.orthography)
            ) if request.concept_cue else 0.0
            episode_rows.append({
                "episode_id": episode.episode_id,
                "orthography": episode.orthography,
                "concept_id": episode.concept_id,
                "similarity": similarity,
                "retrieval_confidence": float(episode.strength * episode.recency * similarity),
                "source_reference": episode.source_reference,
            })
            for feature, value in episode.semantic_features.items():
                episode_memory[feature] = max(episode_memory[feature], float(value) * similarity)
        combined: dict[str, np.ndarray] = {}
        confidence: dict[str, float] = {}
        for modality in MODALITIES:
            current = state["spoke_activations"][modality]
            memory_map: dict[str, float] = defaultdict(float)
            total_weight = 0.0
            for row in retrieved:
                concept: SemanticConcept = row["concept"]
                weight = max(
                    float(row["lexical_similarity"]),
                    float(row["episode_fit"]),
                    float(row.get("input_fit", 0.0)),
                    0.05,
                )
                if concept.concept_id in self._experience_memory:
                    saved = self._experience_memory[concept.concept_id].get(modality, {})
                    for feature, value in saved.items():
                        memory_map[feature] += float(value) * weight
                features = self._concept_modality_features(concept, modality)
                for feature, value in features.items():
                    memory_map[feature] += float(value) * weight
                total_weight += weight
            if modality == "linguistic":
                for feature, value in episode_memory.items():
                    memory_map[feature] += float(value) * 0.25
            recalled_features = state.get("recalled_inputs", {}).get(modality, {})
            if recalled_features:
                for feature, value in recalled_features.items():
                    memory_map[feature] += float(value) * 0.65
                total_weight += 0.65
            if total_weight > _EPS:
                memory_map = {
                    key: float(np.clip(value / total_weight, 0.0, 1.0))
                    for key, value in memory_map.items()
                }
            remembered = _safe_vector(memory_map, self._vocabulary[modality])
            association_drive = (
                self._feature_associations[modality].T
                @ state["input_vectors"][modality]
            )
            remembered = np.clip(remembered + 0.20 * association_drive, 0.0, 1.0)
            current_quality = (
                float(np.count_nonzero(state["input_vectors"][modality])) /
                max(1, len(self._vocabulary[modality]))
            ) if np.any(state["input_vectors"][modality]) else 0.0
            memory_confidence = min(1.0, total_weight / max(len(retrieved), 1))
            alpha = float(np.clip(0.2 + 0.75 * current_quality, 0.0, 0.95))
            if not np.any(current):
                alpha = 0.0
            if not np.any(remembered):
                alpha = 1.0
            combined[modality] = alpha * current + (1.0 - alpha) * remembered
            confidence[modality] = float(max(current_quality, memory_confidence * 0.8))
        state["memory_combined_spokes"] = combined
        state["memory_confidence"] = confidence
        state["episode_retrieval"] = episode_rows
        return {
            "candidate_count": len(retrieved),
            "episode_retrieval": episode_rows,
            "memory_confidence": confidence,
            "combined_spoke_magnitudes": {
                key: float(np.linalg.norm(value)) for key, value in combined.items()
            },
            "current_memory_blend": "alpha*current + (1-alpha)*retrieved",
        }

    def _stage_05_transmit(self, state: dict[str, Any]) -> dict[str, Any]:
        """Project every retained spoke state to the common hub dimensionality."""
        projections = {
            modality: self._spoke_to_hub[modality] @ state["memory_combined_spokes"][modality]
            for modality in MODALITIES
        }
        state["spoke_projections"] = projections
        return {
            "projection_dimensions": {key: int(value.size) for key, value in projections.items()},
            "projection_norms": {key: float(np.linalg.norm(value)) for key, value in projections.items()},
            "all_spokes_retained": True,
        }

    def _stage_06_weight_modalities(self, state: dict[str, Any]) -> dict[str, Any]:
        """Estimate spoke precision, normalize softmax gates, and retain pairwise disagreement."""
        quality = state["memory_confidence"]
        scores = []
        active = []
        for modality in MODALITIES:
            direct = bool(np.any(state["input_vectors"][modality]))
            recalled = bool(np.any(state["memory_combined_spokes"][modality]))
            if direct or recalled:
                active.append(modality)
                scores.append(1.5 * float(quality[modality]) + (0.15 if direct else 0.0))
            else:
                scores.append(-5.0)
        if active:
            gates = _softmax(scores, self.config.modality_temperature)
            gate_map = {modality: float(gates[index]) for index, modality in enumerate(MODALITIES)}
        else:
            gate_map = {modality: 0.0 for modality in MODALITIES}
        disagreement: dict[str, float] = {}
        for index, left in enumerate(MODALITIES):
            for right in MODALITIES[index + 1:]:
                if left in active and right in active:
                    disagreement[f"{left}|{right}"] = float(
                        1.0 - max(-1.0, min(1.0, _cosine(
                            state["spoke_projections"][left],
                            state["spoke_projections"][right],
                        )))
                    )
        direct_modalities = [
            modality for modality in MODALITIES
            if np.any(state["input_vectors"][modality])
        ]
        reinstated_modalities = [
            modality for modality in active if modality not in direct_modalities
        ]
        state["modality_gates"] = gate_map
        state["cross_modal_disagreement"] = disagreement
        return {
            "reliability": quality,
            "modality_gates": gate_map,
            "gate_sum": float(sum(gate_map.values())),
            "cross_modal_disagreement": disagreement,
            "active_modalities": active,
            "direct_input_modalities": direct_modalities,
            "memory_reinstated_modalities": reinstated_modalities,
            "missing_channels_encoded_as_negative": False,
        }

    def _stage_07_integrate_hub(self, state: dict[str, Any]) -> dict[str, Any]:
        """Apply nonlinear feature binding to the reliability-weighted spoke projections."""
        combined = np.zeros(self.config.hidden_dim, dtype=np.float64)
        for modality, projection in state["spoke_projections"].items():
            combined += state["modality_gates"][modality] * projection
        hidden = np.tanh(self._hub_w1 @ combined + self._hub_bias1)
        hub = np.tanh(self._hub_w2 @ hidden + self._hub_bias2)
        state["weighted_hub_input"] = combined
        state["hub_initial"] = hub
        state["hub_state"] = hub.copy()
        return {
            "hub_dimension": int(hub.size),
            "hub_activation": hub,
            "hidden_layer_norm": float(np.linalg.norm(hidden)),
            "nonlinear_integration": True,
            "feature_binding": "two-layer tanh transformation",
        }

    def _stage_08_stabilize_hub(self, state: dict[str, Any]) -> dict[str, Any]:
        """Iterate recurrent hub dynamics until the state stabilizes or the iteration bound is reached."""
        initial = state["hub_initial"]
        current = initial.copy()
        deltas = []
        for _ in range(self.config.recurrent_steps):
            following = np.tanh(
                self.config.recurrent_gain * (self._recurrent @ current)
                + (1.0 - self.config.recurrent_gain) * initial
            )
            delta = float(np.linalg.norm(following - current))
            deltas.append(delta)
            current = following
            if delta < self.config.convergence_epsilon:
                break
        state["hub_state"] = current
        state["recurrent_deltas"] = deltas
        return {
            "iterations": len(deltas),
            "converged": bool(deltas and deltas[-1] < self.config.convergence_epsilon),
            "final_delta": deltas[-1] if deltas else 0.0,
            "delta_history": deltas,
            "hub_state": current,
        }

    def _stage_09_feedback(self, state: dict[str, Any]) -> dict[str, Any]:
        """Project hub information back into each spoke and keep prediction errors separate."""
        predictions: dict[str, np.ndarray] = {}
        refined: dict[str, np.ndarray] = {}
        errors: dict[str, float] = {}
        for modality in MODALITIES:
            prediction = _sigmoid(self._hub_to_spoke[modality] @ state["hub_state"])
            observed = state["memory_combined_spokes"][modality]
            predictions[modality] = prediction
            refined[modality] = (
                (1.0 - self.config.feedback_gain) * observed
                + self.config.feedback_gain * prediction
            )
            errors[modality] = float(np.linalg.norm(observed - prediction))
        state["feedback_predictions"] = predictions
        state["refined_spokes"] = refined
        state["prediction_errors"] = errors
        return {
            "prediction_error_norms": errors,
            "refined_spoke_norms": {
                key: float(np.linalg.norm(value)) for key, value in refined.items()
            },
            "observations_overwritten": False,
        }

    def _stage_10_modality_invariance(self, state: dict[str, Any]) -> dict[str, Any]:
        """Align route-specific hub patterns and decompose common from modality-specific information."""
        routes: dict[str, np.ndarray] = {}
        active = [
            modality for modality in MODALITIES
            if state["modality_gates"][modality] > 0.0
        ]
        for modality in active:
            projection = self._spoke_to_hub[modality] @ state["refined_spokes"][modality]
            hidden = np.tanh(self._hub_w1 @ projection + self._hub_bias1)
            routes[modality] = np.tanh(self._hub_w2 @ hidden + self._hub_bias2)
        if routes:
            weights = np.asarray([state["modality_gates"][key] for key in routes])
            weights = weights / max(float(np.sum(weights)), _EPS)
            shared = sum(
                weight * routes[key] for weight, key in zip(weights, routes)
            )
        else:
            shared = state["hub_state"].copy()
        residuals = {key: value - shared for key, value in routes.items()}
        similarities = {
            f"{left}|{right}": _cosine(routes[left], routes[right])
            for index, left in enumerate(routes)
            for right in list(routes)[index + 1:]
        }
        state["route_hub_states"] = routes
        state["shared_hub_state"] = shared
        state["modality_residuals"] = residuals
        state["hub_state"] = 0.7 * state["hub_state"] + 0.3 * shared
        return {
            "shared_hub_state": shared,
            "modality_specific_residual_norms": {
                key: float(np.linalg.norm(value)) for key, value in residuals.items()
            },
            "cross_modal_similarities": similarities,
            "routes_compared": list(routes),
        }

    def _stage_11_identify_concept(self, state: dict[str, Any]) -> dict[str, Any]:
        """Rank hub-template matches and estimate confidence without discarding alternatives."""
        ranked = self._score_candidates(
            state["hub_state"], state["retrieved_candidates"],
            state["raw_inputs"], state["request"].context_features,
        )
        state["candidate_scores"] = ranked
        margin = (
            float(ranked[0]["score"] - ranked[1]["score"])
            if len(ranked) > 1 else float(ranked[0]["score"]) if ranked else 0.0
        )
        posterior = float(ranked[0]["posterior"]) if ranked else 0.0
        state["identification_confidence"] = posterior
        state["identification_margin"] = margin
        return {
            "ranked_candidates": ranked,
            "top_candidate": ranked[0]["concept_id"] if ranked else None,
            "confidence": posterior,
            "margin": margin,
            "ambiguous": bool(len(ranked) > 1 and margin < self.config.ambiguity_margin),
        }

    def _stage_12_generalize(self, state: dict[str, Any]) -> dict[str, Any]:
        """Compute shared/distinctive feature relations, category prototypes, and explicit association links."""
        candidate_ids = {item["concept_id"] for item in state["candidate_scores"]}
        candidates = [
            item for item in self.concepts
            if item.concept_id in candidate_ids
        ] or list(self.concepts)
        comparisons: list[dict[str, Any]] = []
        for index, left in enumerate(candidates):
            left_features = {
                key for key, value in left.semantic_features.items()
                if float(value) >= self.config.feature_threshold
            }
            for right in candidates[index + 1:]:
                right_features = {
                    key for key, value in right.semantic_features.items()
                    if float(value) >= self.config.feature_threshold
                }
                comparisons.append({
                    "left_concept_id": left.concept_id,
                    "right_concept_id": right.concept_id,
                    "similarity": _map_cosine(left.semantic_features, right.semantic_features),
                    "shared_features": sorted(left_features & right_features),
                    "left_distinctive_features": sorted(left_features - right_features),
                    "right_distinctive_features": sorted(right_features - left_features),
                    "same_prototype": bool(left.prototype_id and left.prototype_id == right.prototype_id),
                })
        groups: dict[str, list[SemanticConcept]] = defaultdict(list)
        for concept in self.concepts:
            group = concept.prototype_id or concept.lexical_category or "uncategorized"
            groups[group].append(concept)
        prototypes = {}
        for group, members in groups.items():
            keys = sorted({key for member in members for key in member.semantic_features})
            prototype = {
                key: float(np.mean([member.semantic_features.get(key, 0.0) for member in members]))
                for key in keys
            }
            prototypes[group] = {
                "features": prototype,
                "member_concept_ids": [member.concept_id for member in members],
                "member_count": len(members),
            }
        relation_edges = [
            {"source": source, "target": target, "strength": weight, "kind": "supplied_association"}
            for source, targets in self.associations.items()
            for target, weight in targets.items()
        ]
        taxonomy_edges = [
            {
                "source": concept.concept_id,
                "target": f"prototype:{concept.prototype_id}",
                "kind": "prototype_membership",
                "lexical_category": concept.lexical_category,
            }
            for concept in self.concepts if concept.prototype_id
        ]
        state["concept_comparisons"] = comparisons
        state["category_prototypes"] = prototypes
        return {
            "concept_comparisons": comparisons,
            "category_prototypes": prototypes,
            "taxonomic_membership_edges": taxonomy_edges,
            "explicit_association_edges": relation_edges,
            "association_provenance": "caller_supplied_only",
        }

    def _stage_13_semantic_control(self, state: dict[str, Any]) -> dict[str, Any]:
        """Boost task-relevant features, reduce distractors, and bias the current retrieval state."""
        request = state["request"]
        feature_state: dict[str, float] = defaultdict(float)
        for modality, values in state["refined_spokes"].items():
            for feature, activation in zip(self._vocabulary[modality], values):
                feature_state[feature] = max(feature_state[feature], float(activation))
        selected: dict[str, float] = {}
        for feature, activation in feature_state.items():
            relevance = float(request.task_relevance.get(feature, 0.0))
            irrelevant = float(request.irrelevant_features.get(feature, 0.0))
            controlled = float(np.clip(
                activation + 0.35 * relevance - 0.25 * irrelevant, 0.0, 1.0
            ))
            if controlled >= self.config.feature_threshold:
                selected[feature] = controlled
        if request.task_relevance:
            control_logits = []
            for modality in MODALITIES:
                values = state["refined_spokes"][modality]
                relevance = sum(
                    float(value) * float(request.task_relevance.get(feature, 0.0))
                    for feature, value in zip(self._vocabulary[modality], values)
                )
                control_logits.append(
                    math.log(max(state["modality_gates"][modality], _EPS)) + relevance
                )
            controlled_weights = _softmax(control_logits, self.config.modality_temperature)
            controlled_gates = {
                modality: float(controlled_weights[index])
                for index, modality in enumerate(MODALITIES)
            }
        else:
            controlled_gates = dict(state["modality_gates"])
        state["controlled_features"] = selected
        state["controlled_modality_gates"] = controlled_gates
        controlled_scores = self._score_candidates(
            state["hub_state"], state["retrieved_candidates"],
            state["raw_inputs"], request.context_features, selected,
        )
        for row in controlled_scores:
            row["controlled_score"] = float(
                np.clip(row["score"] + 0.08 * row["task_fit"], 0.0, 1.0)
            )
        controlled_scores.sort(key=lambda item: item["controlled_score"], reverse=True)
        state["controlled_candidate_scores"] = controlled_scores
        return {
            "task": request.task,
            "selected_features": selected,
            "controlled_modality_gates": controlled_gates,
            "suppressed_feature_count": sum(
                1 for key, value in request.irrelevant_features.items() if value > 0
            ),
            "controlled_candidate_scores": controlled_scores,
            "semantic_store_mutated": False,
        }

    def _stage_14_generate_output(self, state: dict[str, Any]) -> dict[str, Any]:
        """Produce a task-conditioned conceptual response or abstain when evidence is insufficient."""
        ranked = state["controlled_candidate_scores"]
        top = ranked[0] if ranked else None
        margin = (
            float(top["controlled_score"] - ranked[1]["controlled_score"])
            if top and len(ranked) > 1 else float(top["controlled_score"]) if top else 0.0
        )
        confidence = float(top["posterior"]) if top else 0.0
        accepted = bool(
            top
            and top["controlled_score"] >= self.config.confidence_threshold
            and (len(ranked) == 1 or margin >= self.config.ambiguity_margin)
        )
        selected = self._concept_by_id[top["concept_id"]] if accepted and top else None
        features = dict(state["controlled_features"])
        if selected:
            if state["request"].task.casefold() in {"describe", "description", "explain"}:
                response = {
                    "type": "description",
                    "concept_id": selected.concept_id,
                    "label": selected.label,
                    "features": {
                        key: value for key, value in selected.semantic_features.items()
                        if key in features or not state["request"].task_relevance
                    },
                }
            elif state["request"].task.casefold() in {"compare", "comparison"}:
                response = {
                    "type": "comparison_ready",
                    "concept_id": selected.concept_id,
                    "label": selected.label,
                    "comparisons": state["concept_comparisons"],
                }
            else:
                response = {
                    "type": "concept_identification",
                    "concept_id": selected.concept_id,
                    "label": selected.label,
                    "lexical_category": selected.lexical_category,
                }
        else:
            response = {
                "type": "abstention",
                "reason": "insufficient_confidence_or_ambiguous_candidates",
                "candidate_count": len(ranked),
            }
        state["functional_output"] = {
            **response,
            "accepted": accepted,
            "confidence": confidence,
            "decision_margin": margin,
            "alternatives": ranked,
            "selected_features": features,
            "task": state["request"].task,
        }
        return state["functional_output"]

    def _stage_15_acquire_experiences(self, state: dict[str, Any]) -> dict[str, Any]:
        """Validate sourced learning episodes and estimate feature observations without mutating weights."""
        experiences: Sequence[LearningExperience] = state["learning_experiences"]
        counts: dict[str, int] = defaultdict(int)
        feature_counts: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for experience in experiences:
            if experience.concept_id not in self._concept_by_id:
                raise ValueError(f"learning experience references unknown concept {experience.concept_id!r}")
            counts[experience.concept_id] += 1
            for feature, strength in experience.semantic_features.items():
                feature_counts[experience.concept_id][feature] += strength * experience.confidence
            for modality, values in experience.modality_features.items():
                canonical = canonical_modality(modality)
                for feature, strength in values.items():
                    feature_counts[experience.concept_id][f"{canonical}:{feature}"] += (
                        strength * experience.confidence
                    )
        state["experience_counts"] = dict(counts)
        state["experience_feature_counts"] = {
            concept: dict(values) for concept, values in feature_counts.items()
        }
        return {
            "experience_count": len(experiences),
            "concept_episode_counts": dict(counts),
            "feature_evidence_counts": {
                key: dict(value) for key, value in feature_counts.items()
            },
            "feature_probabilities": {
                concept_id: {
                    feature: float(value / max(
                        sum(item.confidence for item in experiences
                             if item.concept_id == concept_id),
                        _EPS,
                    ))
                    for feature, value in features.items()
                }
                for concept_id, features in feature_counts.items()
            },
            "stable_features": {
                concept_id: sorted(
                    feature for feature, value in features.items()
                    if value / max(
                        sum(item.confidence for item in experiences
                            if item.concept_id == concept_id),
                        _EPS,
                    ) >= 0.5
                )
                for concept_id, features in feature_counts.items()
            },
            "incidental_features": {
                concept_id: sorted(
                    feature for feature, value in features.items()
                    if value / max(
                        sum(item.confidence for item in experiences
                            if item.concept_id == concept_id),
                        _EPS,
                    ) < 0.5
                )
                for concept_id, features in feature_counts.items()
            },
            "all_experiences_sourced": all(
                item.source_reference not in {"", "unspecified"}
                and item.verified_by not in {"", "unspecified"}
                for item in experiences
            ),
        }

    def _stage_16_learn_mappings(self, state: dict[str, Any]) -> dict[str, Any]:
        """Compute gradient-style and Hebbian spoke-to-hub proposals from paired experience."""
        experiences: Sequence[LearningExperience] = state["learning_experiences"]
        deltas = {modality: np.zeros_like(weight) for modality, weight in self._spoke_to_hub.items()}
        losses: dict[str, list[float]] = defaultdict(list)
        for experience in experiences:
            concept = self._concept_by_id[experience.concept_id]
            target = self._concept_hub(concept)
            modality_maps = {
                canonical_modality(modality): values
                for modality, values in experience.modality_features.items()
            }
            for modality in MODALITIES:
                if modality not in modality_maps:
                    continue
                vector = _safe_vector(modality_maps[modality], self._vocabulary[modality])
                if experience.semantic_features:
                    semantic_vector = _safe_vector(
                        experience.semantic_features, self._vocabulary[modality]
                    )
                    vector = np.maximum(vector, semantic_vector)
                if not np.any(vector):
                    continue
                prediction = self._spoke_to_hub[modality] @ vector
                error = target - prediction
                loss = 0.5 * float(np.dot(error, error))
                confidence = float(experience.confidence)
                gradient = np.outer(error, vector)
                hebbian = np.outer(target, vector)
                deltas[modality] += confidence * (
                    self.config.learning_rate * gradient
                    + 0.08 * self.config.learning_rate * hebbian
                )
                losses[modality].append(loss)
        denominator = max(len(experiences), 1)
        deltas = {key: value / denominator for key, value in deltas.items()}
        state["mapping_deltas"] = deltas
        return {
            "mapping_loss_by_modality": {
                key: float(np.mean(values)) for key, values in losses.items()
            },
            "proposed_delta_norms": {
                key: float(np.linalg.norm(value)) for key, value in deltas.items()
            },
            "learning_rule": "gradient_error_plus_hebbian_coactivation",
            "weights_committed": False,
        }

    def _stage_17_align_modalities(self, state: dict[str, Any]) -> dict[str, Any]:
        """Prepare cross-modal alignment updates for routes paired by the same verified concept."""
        experiences: Sequence[LearningExperience] = state["learning_experiences"]
        deltas = {modality: np.zeros_like(weight) for modality, weight in self._spoke_to_hub.items()}
        total_loss = 0.0
        pair_count = 0
        for experience in experiences:
            route_states: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for raw_modality, values in experience.modality_features.items():
                modality = canonical_modality(raw_modality)
                vector = _safe_vector(values, self._vocabulary[modality])
                if experience.semantic_features:
                    vector = np.maximum(
                        vector,
                        _safe_vector(experience.semantic_features, self._vocabulary[modality]),
                    )
                if np.any(vector):
                    route_states[modality] = (
                        vector, self._spoke_to_hub[modality] @ vector
                    )
            if len(route_states) < 2:
                continue
            shared = np.mean(np.vstack([item[1] for item in route_states.values()]), axis=0)
            for modality, (vector, route) in route_states.items():
                error = shared - route
                total_loss += float(np.dot(error, error))
                pair_count += 1
                deltas[modality] += (
                    self.config.learning_rate
                    * float(experience.confidence)
                    * np.outer(error, vector)
                )
        denominator = max(len(experiences), 1)
        state["alignment_deltas"] = {
            key: value / denominator for key, value in deltas.items()
        }
        return {
            "cross_modal_alignment_loss": total_loss / max(pair_count, 1),
            "paired_route_count": pair_count,
            "alignment_delta_norms": {
                key: float(np.linalg.norm(value))
                for key, value in state["alignment_deltas"].items()
            },
            "modality_residuals_preserved": True,
            "weights_committed": False,
        }

    def _stage_18_update_knowledge(self, state: dict[str, Any]) -> dict[str, Any]:
        """Commit authorized bounded learning while anchoring existing mappings against excessive drift."""
        experiences: Sequence[LearningExperience] = state["learning_experiences"]
        authorized = bool(state["learning_authorized"])
        if not experiences:
            status = "skipped_no_experiences"
        elif not authorized:
            status = "preview_only_authorization_required"
        else:
            for modality in MODALITIES:
                proposed = (
                    self._spoke_to_hub[modality]
                    + state["mapping_deltas"][modality]
                    + state["alignment_deltas"][modality]
                )
                anchor = self._initial_spoke_to_hub[modality]
                regularized = proposed - self.config.weight_decay * (
                    self._spoke_to_hub[modality] - anchor
                )
                self._spoke_to_hub[modality] = np.clip(regularized, -2.0, 2.0)
                self._hub_to_spoke[modality] = self._spoke_to_hub[modality].T.copy()
            for experience in experiences:
                concept_memory = self._experience_memory.setdefault(experience.concept_id, {})
                modality_maps = {
                    canonical_modality(modality): values
                    for modality, values in experience.modality_features.items()
                }
                for modality, values in modality_maps.items():
                    old = concept_memory.get(modality, {})
                    keys = set(old) | set(values) | set(experience.semantic_features)
                    updated = {}
                    for feature in keys:
                        target = max(
                            float(values.get(feature, 0.0)),
                            float(experience.semantic_features.get(feature, 0.0)) * 0.25,
                        )
                        updated[feature] = float(np.clip(
                            0.8 * float(old.get(feature, 0.0))
                            + 0.2 * target * experience.confidence,
                            0.0, 1.0,
                        ))
                    concept_memory[modality] = updated
                stats = self._feature_probability.setdefault(experience.concept_id, {})
                for feature, value in experience.semantic_features.items():
                    stats[feature] = float(np.clip(
                        0.8 * stats.get(feature, 0.0) + 0.2 * value * experience.confidence,
                        0.0, 1.0,
                    ))
            self.model_generation += 1
            status = "committed"
        state["learning_status"] = status
        return {
            "status": status,
            "authorized": authorized,
            "experience_count": len(experiences),
            "model_generation": self.model_generation,
            "learned_concept_ids": sorted({item.concept_id for item in experiences})
                if status == "committed" else [],
            "weight_update_rule": "bounded_gradient_hebbian_alignment_with_anchor_regularization",
            "catastrophic_forgetting_guard": "prior_weight_anchor_penalty",
        }

    def _rank_lesion(self, hub: np.ndarray, retrieved: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Measure candidate discriminability from a counterfactual damaged hub state."""
        scored = []
        for row in retrieved:
            concept = row["concept"]
            similarity = _cosine(hub, self._concept_hub(concept))
            hub_fit = (similarity + 1.0) / 2.0
            score = 0.8 * hub_fit + 0.2 * float(row["lexical_similarity"])
            scored.append((concept.concept_id, float(score)))
        scored.sort(key=lambda item: item[1], reverse=True)
        margin = scored[0][1] - scored[1][1] if len(scored) > 1 else (
            scored[0][1] if scored else 0.0
        )
        return {
            "top_candidate": scored[0][0] if scored else None,
            "candidate_scores": [
                {"concept_id": concept_id, "score": score} for concept_id, score in scored
            ],
            "specificity_margin": float(margin),
        }

    def _stage_19_lesion_hub(self, state: dict[str, Any]) -> dict[str, Any]:
        """Apply a partial hub-unit mask and measure loss of cross-modal concept specificity."""
        hub = state["hub_state"].copy()
        count = max(1, int(round(hub.size * self.config.lesion_severity))) if hub.size else 0
        vulnerable = np.argsort(np.abs(hub))[-count:] if count else np.asarray([], dtype=int)
        degradation = np.ones_like(hub)
        degradation[vulnerable] = 1.0 - self.config.lesion_severity
        impaired = hub * degradation
        baseline = self._rank_lesion(hub, state["retrieved_candidates"])
        damaged = self._rank_lesion(impaired, state["retrieved_candidates"])
        state["hub_lesion_state"] = impaired
        return {
            "severity": self.config.lesion_severity,
            "degraded_dimensions": vulnerable.astype(int).tolist(),
            "baseline": baseline,
            "impaired": damaged,
            "specificity_margin_change": (
                damaged["specificity_margin"] - baseline["specificity_margin"]
            ),
            "graded_residual_state": impaired,
        }

    def _stage_20_lesion_spokes(self, state: dict[str, Any]) -> dict[str, Any]:
        """Attenuate each spoke separately and compare its selective contribution to hub performance."""
        baseline = self._rank_lesion(state["hub_state"], state["retrieved_candidates"])
        profiles = {}
        for modality in MODALITIES:
            altered = {
                key: value.copy() for key, value in state["refined_spokes"].items()
            }
            altered[modality] *= 1.0 - self.config.lesion_severity
            projections = {
                key: self._spoke_to_hub[key] @ altered[key] for key in MODALITIES
            }
            impaired_hub = self._integrate_projections(projections, state["modality_gates"])
            score = self._rank_lesion(impaired_hub, state["retrieved_candidates"])
            profiles[modality] = {
                "baseline": baseline,
                "impaired": score,
                "specificity_margin_change": (
                    score["specificity_margin"] - baseline["specificity_margin"]
                ),
                "affected_spoke": modality,
                "other_spokes_preserved": True,
            }
        state["spoke_lesion_profiles"] = profiles
        return {
            "severity": self.config.lesion_severity,
            "individual_spoke_profiles": profiles,
            "selective_ablation": True,
        }

    def _stage_21_lesion_connections(self, state: dict[str, Any]) -> dict[str, Any]:
        """Attenuate each spoke-hub projection matrix in turn while preserving endpoint activity."""
        baseline = self._rank_lesion(state["hub_state"], state["retrieved_candidates"])
        profiles = {}
        for modality in MODALITIES:
            projections = {}
            for key in MODALITIES:
                weights = self._spoke_to_hub[key]
                if key == modality:
                    weights = weights * (1.0 - self.config.lesion_severity)
                projections[key] = weights @ state["refined_spokes"][key]
            impaired_hub = self._integrate_projections(projections, state["modality_gates"])
            score = self._rank_lesion(impaired_hub, state["retrieved_candidates"])
            profiles[modality] = {
                "baseline": baseline,
                "impaired": score,
                "specificity_margin_change": (
                    score["specificity_margin"] - baseline["specificity_margin"]
                ),
                "damaged_connection": f"{modality}_to_atl_hub",
                "endpoint_populations_preserved": True,
            }
        state["connection_lesion_profiles"] = profiles
        return {
            "severity": self.config.lesion_severity,
            "individual_connection_profiles": profiles,
            "endpoint_populations_destroyed": False,
        }

    def _stage_biological_drives(
        self, index: int, state: dict[str, Any], output: Mapping[str, Any]
    ) -> dict[str, float]:
        """Derive bounded population drive from the functional state of one algorithmic stage."""
        modality_activity: dict[str, float] = {}
        for modality in MODALITIES:
            values = state.get("refined_spokes", state.get("spoke_activations", {})).get(modality)
            if values is None or not values.size:
                modality_activity[modality] = 0.0
            else:
                modality_activity[modality] = float(np.clip(np.mean(values), 0.0, 1.0))
        hub = float(np.clip(np.mean(np.abs(state.get("hub_state", np.zeros(1)))), 0.0, 1.0))
        control = float(np.clip(
            len(state.get("controlled_features", {})) /
            max(sum(len(value) for value in state.get("extracted_features", {}).values()), 1),
            0.0, 1.0,
        ))
        memory = float(np.clip(
            len(state.get("episode_retrieval", [])) / max(len(state.get("request").episodes), 1),
            0.0, 1.0,
        )) if state.get("request") else 0.0
        output_activity = float(bool(state.get("functional_output", {}).get("accepted", False)))
        if index == 19:
            hub *= 1.0 - self.config.lesion_severity
        elif index == 20:
            modality_activity = {
                key: value * (1.0 - self.config.lesion_severity)
                for key, value in modality_activity.items()
            }
        elif index == 21:
            modality_activity = {
                key: value * (1.0 - 0.5 * self.config.lesion_severity)
                for key, value in modality_activity.items()
            }
        return biological_drives(modality_activity, hub, control, memory, output_activity)

    def process(
        self,
        request: HubAndSpokeRequest,
        learning_experiences: Sequence[LearningExperience] = (),
        learning_authorized: bool = False,
    ) -> HubAndSpokeResult:
        """Execute all 21 functional stages and the parallel, optional LIF reference."""
        if not isinstance(request, HubAndSpokeRequest):
            raise TypeError("request must be a HubAndSpokeRequest")
        experiences = tuple(learning_experiences)
        if any(not isinstance(item, LearningExperience) for item in experiences):
            raise TypeError("learning_experiences must contain LearningExperience records")
        self._ensure_vocabulary(request, experiences)
        state: dict[str, Any] = {
            "request": request,
            "learning_experiences": experiences,
            "learning_authorized": bool(learning_authorized),
        }
        stage_methods = (
            self._stage_01_receive,
            self._stage_02_activate_spokes,
            self._stage_03_extract_features,
            self._stage_04_retrieve_memory,
            self._stage_05_transmit,
            self._stage_06_weight_modalities,
            self._stage_07_integrate_hub,
            self._stage_08_stabilize_hub,
            self._stage_09_feedback,
            self._stage_10_modality_invariance,
            self._stage_11_identify_concept,
            self._stage_12_generalize,
            self._stage_13_semantic_control,
            self._stage_14_generate_output,
            self._stage_15_acquire_experiences,
            self._stage_16_learn_mappings,
            self._stage_17_align_modalities,
            self._stage_18_update_knowledge,
            self._stage_19_lesion_hub,
            self._stage_20_lesion_spokes,
            self._stage_21_lesion_connections,
        )
        lif = HubSpokeLIFSimulator(self.config) if self.config.simulate_biology else None
        stage_outputs: dict[str, Any] = {}
        stage_trace: list[dict[str, Any]] = []
        operation_trace: list[dict[str, Any]] = []
        for index, (spec, method) in enumerate(zip(STAGE_REGISTRY, stage_methods), 1):
            output = method(state)
            status = "executed"
            if index in {15, 16, 17, 18} and not experiences:
                status = "skipped_no_experiences"
            elif index == 18 and experiences and not learning_authorized:
                status = "preview_only"
            stage_outputs[spec.stage_id] = _plain(output)
            biological_event = None
            if lif is not None:
                biological_event = lif.step(
                    spec.stage_id,
                    self._stage_biological_drives(index, state, output),
                )
            stage_trace.append({
                "stage_id": spec.stage_id,
                "stage_number": index,
                "name": spec.name,
                "objective": spec.objective,
                "status": status,
                "biological_reference": spec.biological_reference,
                "output_summary": _trace_summary(output),
                "biological_event": biological_event,
            })
            for operation in spec.operations:
                operation_trace.append({
                    "operation_id": operation.operation_id,
                    "stage_id": spec.stage_id,
                    "stage_name": spec.name,
                    "operation": operation.name,
                    "mathematical_operation": operation.mathematical_operation,
                    "functional_responsibility": operation.functional_responsibility,
                    "biological_reference": spec.biological_reference,
                    "functional_status": status,
                    "biological_status": "executed" if lif is not None else "skipped_disabled",
                })
        biological = lif.snapshot() if lif is not None else {
            "enabled": False,
            "operation_trace": [
                {"stage_id": stage.stage_id, "status": "skipped_disabled"}
                for stage in STAGE_REGISTRY
            ],
            "stage_count": 0,
            "numerically_valid": True,
            "model_scope": "LIF reference disabled by configuration; functional model still ran.",
        }
        selected_output = state["functional_output"]
        lesional = {
            "hub": stage_outputs["HS-19"],
            "spokes": stage_outputs["HS-20"],
            "connections": stage_outputs["HS-21"],
        }
        finite_hub = bool(np.all(np.isfinite(state["hub_state"])))
        gate_sum = float(sum(state["modality_gates"].values()))
        diagnostics = {
            "valid": bool(
                finite_hub
                and len(stage_trace) == 21
                and len(operation_trace) == len(ALL_OPERATIONS)
                and abs(gate_sum - (1.0 if any(state["modality_gates"].values()) else 0.0)) < 1e-6
                and biological.get("numerically_valid", True)
            ),
            "stage_count": len(stage_trace),
            "expected_stage_count": 21,
            "operation_count": len(operation_trace),
            "expected_operation_count": len(ALL_OPERATIONS),
            "hub_state_finite": finite_hub,
            "modality_gate_sum": gate_sum,
            "biological_state_finite": biological.get("numerically_valid", True),
            "recognition_mutates_model": bool(learning_authorized and experiences),
            "learning_status": state.get("learning_status", "skipped_no_experiences"),
            "scientific_validation": (
                "Passing software invariants does not establish behavioral validity or biological correspondence."
            ),
        }
        return HubAndSpokeResult(
            sequence_id=request.sequence_id,
            original_input=request.concept_cue,
            output=_plain(selected_output),
            stage_outputs=stage_outputs,
            stage_trace=stage_trace,
            operation_trace=operation_trace,
            biological=biological,
            learning={
                "stage_15": stage_outputs["HS-15"],
                "stage_16": stage_outputs["HS-16"],
                "stage_17": stage_outputs["HS-17"],
                "stage_18": stage_outputs["HS-18"],
            },
            lesions=lesional,
            diagnostics=diagnostics,
            metadata={
                "model_generation": self.model_generation,
                "concept_count": len(self.concepts),
                "modalities": list(MODALITIES),
                "feature_dimensions": {
                    modality: len(self._vocabulary[modality]) for modality in MODALITIES
                },
                "hub_dimension": self.config.hidden_dim,
                "source_reference": "Master Algorithm - Hub-and-Spoke Semantic Concept Integration.txt",
                "model_scope": (
                    "Functional Hub-and-Spoke approximation with an optional generic LIF reference. "
                    "It is not a fitted neural model or a clinical assessment."
                ),
            },
        )

    def learned_feature_probabilities(self, concept_id: str) -> dict[str, float]:
        """Return a defensive copy of confidence-weighted feature statistics for one concept."""
        if concept_id not in self._concept_by_id:
            raise KeyError(concept_id)
        return dict(self._feature_probability.get(concept_id, {}))

    def experience_memory(self, concept_id: str) -> dict[str, dict[str, float]]:
        """Return a defensive copy of explicitly learned modality features for one concept."""
        if concept_id not in self._concept_by_id:
            raise KeyError(concept_id)
        return {
            modality: dict(features)
            for modality, features in self._experience_memory.get(concept_id, {}).items()
        }

    def reset_learned_experiences(self) -> None:
        """Clear explicitly acquired episode statistics and restore original spoke-to-hub mappings."""
        self._experience_memory.clear()
        self._feature_probability.clear()
        for modality in MODALITIES:
            self._spoke_to_hub[modality] = self._initial_spoke_to_hub[modality].copy()
            self._hub_to_spoke[modality] = self._spoke_to_hub[modality].T.copy()
        self.model_generation += 1


def _normalize_episode_features(
    episodes: Sequence[WordFormEpisode],
) -> dict[str, dict[str, float]]:
    """Collect semantic feature evidence from supplied episodes for vocabulary preparation."""
    combined: dict[str, dict[str, float]] = {modality: {} for modality in MODALITIES}
    for episode in episodes:
        for modality in MODALITIES:
            for feature, value in episode.semantic_features.items():
                combined[modality][feature] = max(combined[modality].get(feature, 0.0), value * 0.25)
    return combined


def _trace_summary(value: Mapping[str, Any]) -> dict[str, Any]:
    """Keep stage trace summaries compact while retaining key dimensions and confidence values."""
    summary = {}
    for key, item in value.items():
        if isinstance(item, np.ndarray):
            summary[key] = {"shape": list(item.shape), "norm": float(np.linalg.norm(item))}
        elif isinstance(item, Mapping) and len(item) > 20:
            summary[key] = {"item_count": len(item), "keys": list(item)[:20]}
        elif isinstance(item, list) and len(item) > 20:
            summary[key] = {"item_count": len(item)}
        else:
            summary[key] = _plain(item)
    return summary


def request_from_semantic_request(request: Any) -> HubAndSpokeRequest:
    """Adapt the existing ore_semantic request contract without changing its source values."""
    from ore_semantic.contracts import SemanticRequest

    if not isinstance(request, SemanticRequest):
        raise TypeError("request must be an ore_semantic.SemanticRequest")
    return HubAndSpokeRequest(
        sequence_id=request.sequence_id,
        text=request.text,
        tokens=request.tokens,
        modality_inputs=request.modality_evidence,
        context_features=request.context,
        task=request.task,
        episodes=request.episodes,
        source_reference=request.source_reference or "ore_semantic.SemanticRequest",
    )
