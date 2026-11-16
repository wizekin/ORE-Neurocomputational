"""Executable functional pipeline for 19-stage hybrid embodied–amodal concept processing."""
from __future__ import annotations

import hashlib
import math
import time
from typing import Any, Mapping, Sequence

import numpy as np

from .biology import HybridLIFSimulator, stage_drives
from .contracts import (
    HybridConcept,
    HybridConfig,
    HybridLearningExample,
    HybridRequest,
    HybridResult,
    _jsonable,
)
from .registry import ALL_OPERATIONS, STAGE_REGISTRY, get_stage


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Apply a numerically stable logistic activation elementwise."""
    clipped = np.clip(np.asarray(values, dtype=np.float64), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _softmax(values: Sequence[float] | np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Convert finite candidate scores into a stable normalized distribution."""
    scores = np.asarray(values, dtype=np.float64)
    if scores.size == 0:
        return scores
    scaled = (scores - np.max(scores)) / max(float(temperature), 1e-8)
    exponentials = np.exp(np.clip(scaled, -60.0, 0.0))
    denominator = float(np.sum(exponentials))
    return exponentials / denominator if denominator > 0.0 else np.zeros_like(scores)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    """Compute cosine similarity with an explicit zero-vector guard."""
    a = np.asarray(left, dtype=np.float64).ravel()
    b = np.asarray(right, dtype=np.float64).ravel()
    if a.shape != b.shape or a.size == 0:
        return 0.0
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denominator)


def _vector(features: Mapping[str, float], vocabulary: Sequence[str]) -> np.ndarray:
    """Encode a named feature map in a stable vocabulary order."""
    return np.asarray([float(features.get(name, 0.0)) for name in vocabulary], dtype=np.float64)


def _as_feature_map(values: np.ndarray, vocabulary: Sequence[str]) -> dict[str, float]:
    """Decode a numerical feature vector into named values in vocabulary order."""
    vector = np.asarray(values, dtype=np.float64).ravel()
    return {name: float(vector[index]) for index, name in enumerate(vocabulary)}


def _finite(values: np.ndarray) -> bool:
    """Return whether every element of a numerical state is finite."""
    return bool(np.all(np.isfinite(np.asarray(values, dtype=np.float64))))


class HybridEmbodiedAmodalProcessor:
    """Run the ordered 19-stage functional pipeline with an optional LIF reference network."""

    def __init__(
        self,
        concepts: Sequence[Any] = (),
        config: HybridConfig | None = None,
    ) -> None:
        """Prepare validated concept prototypes, deterministic projections, and optional biology."""
        self.config = config or HybridConfig()
        self.concepts = tuple(self._coerce_concept(item) for item in concepts)
        ids = [item.concept_id for item in self.concepts]
        if len(ids) != len(set(ids)):
            raise ValueError("concept_id values must be unique")
        self._learned_memory: dict[str, dict[str, Any]] = {}
        self._learned_associations: dict[str, float] = {}
        self._projection_cache: dict[tuple[str, int, int], np.ndarray] = {}
        rng = np.random.default_rng(self.config.random_seed)
        raw_recurrent = rng.normal(
            0.0, 1.0 / math.sqrt(self.config.hub_dimensions),
            (self.config.hub_dimensions, self.config.hub_dimensions),
        )
        spectral = max(float(np.linalg.norm(raw_recurrent, ord=2)), 1e-8)
        self._hub_recurrent = raw_recurrent * (0.22 / spectral)
        self._biology = (
            HybridLIFSimulator(self.config) if self.config.simulate_biology else None
        )

    @staticmethod
    def _coerce_concept(item: Any) -> HybridConcept:
        """Accept native HybridConcept records or adapt existing ORE SemanticConcept objects."""
        if isinstance(item, HybridConcept):
            return item
        if isinstance(item, Mapping):
            return HybridConcept(**dict(item))
        concept_id = getattr(item, "concept_id", None)
        label = getattr(item, "label", None)
        if not concept_id or not label:
            raise TypeError("concepts must be HybridConcept, mapping, or SemanticConcept-like records")
        return HybridConcept(
            concept_id=str(concept_id),
            label=str(label),
            features=dict(getattr(item, "semantic_features", {}) or {}),
            modality_features=dict(getattr(item, "modality_features", {}) or {}),
            category=str(
                getattr(item, "prototype_id", None)
                or getattr(item, "lexical_category", None)
                or "unknown"
            ),
            aliases=tuple(getattr(item, "aliases", ()) or ()),
            frequency=float(getattr(item, "frequency", 1.0)),
            verified_by=str(getattr(item, "verified_by", "unspecified")),
            source_reference=str(getattr(item, "source_reference", "unspecified")),
        )

    def _all_concepts(self) -> tuple[HybridConcept, ...]:
        """Combine immutable caller prototypes with previously authorized learned prototypes."""
        combined = {concept.concept_id: concept for concept in self.concepts}
        combined.update({
            concept_id: value["concept"]
            for concept_id, value in self._learned_memory.items()
        })
        return tuple(combined.values())

    def _vocabulary(self, request: HybridRequest) -> tuple[str, ...]:
        """Collect a deterministic feature vocabulary from inputs, context, memory, and concepts."""
        names: set[str] = set()
        maps: list[Mapping[str, float]] = [
            request.context_features, request.task_features, request.irrelevant_features,
            request.target_features,
        ]
        for mapping in request.modality_inputs.values():
            maps.append(mapping)
        for mapping in request.spatial_features.values():
            maps.append(mapping)
        for concept in self._all_concepts():
            maps.append(concept.features)
            maps.extend(concept.modality_features.values())
        for example in request.learning_examples:
            maps.append(example.shared_features)
            maps.extend(example.modality_inputs.values())
        for mapping in maps:
            names.update(str(name) for name in mapping)
        return tuple(sorted(names)) or ("__empty_feature__",)

    def _projection(self, name: str, input_size: int) -> np.ndarray:
        """Return a deterministic modality-to-hub or feature-to-hub projection matrix."""
        key = (name, input_size, self.config.hub_dimensions)
        cached = self._projection_cache.get(key)
        if cached is not None:
            return cached
        # A shared semantic basis aligns feature dimensions across modalities. Small,
        # deterministic route perturbations allow modality-specific structure without
        # making identical named features orthogonal merely because they came from
        # different input channels.
        base_rng = np.random.default_rng(self.config.random_seed + 1709)
        base = base_rng.normal(
            0.0, 1.0 / math.sqrt(max(input_size, 1)),
            (self.config.hub_dimensions, input_size),
        )
        if name in MODALITIES_FOR_DIAGNOSTICS:
            digest = hashlib.sha256(name.encode("utf-8")).digest()
            seed_offset = int.from_bytes(digest[:4], "little")
            route_rng = np.random.default_rng(self.config.random_seed + seed_offset)
            perturbation = route_rng.normal(
                0.0, 0.025 / math.sqrt(max(input_size, 1)),
                (self.config.hub_dimensions, input_size),
            )
            matrix = base + perturbation
        else:
            matrix = base
        self._projection_cache[key] = matrix
        return matrix

    def _concept_vector(
        self, concept: HybridConcept, vocabulary: Sequence[str]
    ) -> np.ndarray:
        """Build a shared prototype from semantic features or available modality features."""
        if concept.features:
            return _vector(concept.features, vocabulary)
        channel_vectors = [
            _vector(values, vocabulary) for values in concept.modality_features.values()
        ]
        if not channel_vectors:
            return np.zeros(len(vocabulary), dtype=np.float64)
        return np.mean(np.stack(channel_vectors), axis=0)

    @staticmethod
    def _pair_key(left: str, right: str) -> str:
        """Create an order-independent key for an association between two modalities."""
        return "|".join(sorted((left, right)))

    def _record_stage(
        self,
        number: int,
        payload: Mapping[str, Any],
        modality_strengths: Mapping[str, float],
        hub_strength: float,
        control_strength: float,
        prediction_error: float = 0.0,
        response_strength: float = 0.0,
        plasticity_strength: float = 0.0,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Record stage output, biological reference status, and every registered substep."""
        stage = get_stage(number)
        started = time.perf_counter()
        if self._biology is not None:
            drives = stage_drives(
                number, modality_strengths, hub_strength, control_strength,
                prediction_error, response_strength, plasticity_strength,
            )
            biological_event = self._biology.step(stage.stage_id, drives)
            biological_status = "reference_advanced"
        else:
            biological_event = {
                "stage_id": stage.stage_id,
                "status": "disabled",
                "reason": "simulate_biology=False",
            }
            biological_status = "disabled"
        safe_payload = _jsonable(dict(payload))
        evidence = payload.get("step_evidence", ())
        for index, operation in enumerate(stage.steps):
            output_evidence = evidence[index] if index < len(evidence) else {
                "stage_output_keys": sorted(str(key) for key in payload if key != "step_evidence")
            }
            self._current_operation_trace.append({
                "operation_id": operation.operation_id,
                "stage_id": stage.stage_id,
                "stage_name": stage.name,
                "operation": operation.name,
                "mathematical_operation": operation.mathematical_operation,
                "functional_responsibility": operation.functional_responsibility,
                "biological_reference": operation.biological_reference,
                "functional_status": "completed",
                "biological_status": biological_status,
                "output_evidence": _jsonable(output_evidence),
            })
        duration_ms = (time.perf_counter() - started) * 1000.0
        stage_row = {
            "stage_id": stage.stage_id,
            "stage_number": number,
            "stage_name": stage.name,
            "objective": stage.objective,
            "functional_status": "completed",
            "biological_status": biological_status,
            "duration_ms": float(duration_ms),
            "operation_count": len(stage.steps),
            "biological_event": biological_event,
        }
        self._current_stage_trace.append(stage_row)
        self._current_stage_outputs[stage.stage_id] = safe_payload
        return stage_row, biological_event

    def _rank_concepts(
        self,
        hub: np.ndarray,
        vocabulary: Sequence[str],
        context: np.ndarray,
        task: np.ndarray,
        concepts: Sequence[HybridConcept],
        prediction_fit: Mapping[str, float] | None = None,
    ) -> list[dict[str, Any]]:
        """Score all known concept prototypes while keeping evidence channels independently visible."""
        semantic_projection = self._projection("__semantic__", len(vocabulary))
        rows: list[dict[str, Any]] = []
        raw_scores: list[float] = []
        for concept in concepts:
            concept_vector = self._concept_vector(concept, vocabulary)
            projected = semantic_projection @ concept_vector
            hub_fit = max(0.0, _cosine(hub, projected))
            context_fit = max(0.0, _cosine(concept_vector, context))
            task_fit = max(0.0, _cosine(concept_vector, task))
            frequency_prior = math.log1p(max(concept.frequency, 1e-6))
            prediction_fit_value = float((prediction_fit or {}).get(concept.concept_id, 0.0))
            score = (
                0.48 * hub_fit + 0.18 * context_fit + 0.14 * task_fit
                + 0.08 * min(frequency_prior / 6.0, 1.0)
                + 0.12 * prediction_fit_value
            )
            raw_scores.append(score)
            rows.append({
                "concept_id": concept.concept_id,
                "label": concept.label,
                "category": concept.category,
                "hub_similarity": hub_fit,
                "context_fit": context_fit,
                "task_fit": task_fit,
                "frequency_prior": frequency_prior,
                "prediction_fit": prediction_fit_value,
                "score": score,
            })
        # A softmax over a single stored concept would otherwise report 1.0
        # even when every evidence channel is zero. Frequency is a prior, not
        # independent evidence that a concept is currently present.
        evidence_available = (
            float(np.linalg.norm(hub)) > 1e-8
            or float(np.linalg.norm(context)) > 1e-8
            or float(np.linalg.norm(task)) > 1e-8
            or any(float(value) > 1e-8 for value in (prediction_fit or {}).values())
        )
        posterior = (
            _softmax(raw_scores, temperature=0.20)
            if evidence_available else np.zeros(len(rows), dtype=np.float64)
        )
        for row, probability in zip(rows, posterior):
            row["posterior"] = float(probability)
            row["evidence_available"] = evidence_available
        rows.sort(key=lambda row: row["posterior"], reverse=True)
        return rows

    def process(self, request: HybridRequest) -> HybridResult:
        """Execute all 19 stages and return separate common, embodied, response, and LIF tracks."""
        self._current_operation_trace: list[dict[str, Any]] = []
        self._current_stage_trace: list[dict[str, Any]] = []
        self._current_stage_outputs: dict[str, dict[str, Any]] = {}
        vocabulary = self._vocabulary(request)
        concepts = self._all_concepts()
        original = {
            modality: _vector(features, vocabulary)
            for modality, features in request.modality_inputs.items()
        }
        context_vector = _vector(request.context_features, vocabulary)
        task_vector = _vector(request.task_features, vocabulary)
        irrelevant_vector = _vector(request.irrelevant_features, vocabulary)
        target_vector = _vector(request.target_features, vocabulary)
        modality_strengths = {
            modality: float(np.mean(np.abs(values))) for modality, values in original.items()
        }
        feature_count = len(vocabulary)
        epsilon = 1e-8

        # Stage 1: transduce and normalize channels without losing constant-valued evidence.
        normalized: dict[str, np.ndarray] = {}
        signal_statistics: dict[str, dict[str, float]] = {}
        for modality, values in original.items():
            mean = float(np.mean(values))
            std = float(np.std(values))
            normalized[modality] = (
                (values - mean) / (std + epsilon) if std > epsilon else values.copy()
            )
            signal_statistics[modality] = {"mean": mean, "std": std}
        self._record_stage(1, {
            "transduced_streams": {m: _as_feature_map(v, vocabulary) for m, v in original.items()},
            "normalized_streams": {m: _as_feature_map(v, vocabulary) for m, v in normalized.items()},
            "signal_statistics": signal_statistics,
            "step_evidence": [
                {"available_modalities": list(original)},
                {"statistics": signal_statistics},
                {"normalized_channels": list(normalized)},
                {"parallel_stream_count": len(normalized)},
            ],
        }, modality_strengths, 0.0, 0.0)

        # Stage 2: distribute to separate spoke states; only present modalities are instantiated.
        spokes = {m: np.tanh(values) for m, values in normalized.items()}
        self._record_stage(2, {
            "spokes": {m: _as_feature_map(v, vocabulary) for m, v in spokes.items()},
            "absent_modalities": sorted(set((
                "visual", "auditory", "motor", "somatosensory", "olfactory",
                "gustatory", "emotional", "interoceptive", "linguistic", "internal",
            )) - set(spokes)),
            "step_evidence": [
                {"routed_modalities": list(spokes)},
                {"spoke_activity": {m: float(np.linalg.norm(v)) for m, v in spokes.items()}},
                {"recurrent_path": "higher-order feedback is represented in later hub-to-spoke stages"},
            ],
        }, modality_strengths, 0.0, 0.0)

        # Stage 3: derive modality-specific feature patterns and graded feature similarity.
        extracted = {
            m: np.tanh(v + 0.15 * np.square(v)) for m, v in spokes.items()
        }
        feature_similarity: dict[str, dict[str, float]] = {}
        modalities = list(extracted)
        for left in modalities:
            feature_similarity[left] = {
                right: _cosine(extracted[left], extracted[right])
                for right in modalities
            }
        self._record_stage(3, {
            "features": {m: _as_feature_map(v, vocabulary) for m, v in extracted.items()},
            "feature_similarity": feature_similarity,
            "step_evidence": [
                {"feature_channels": list(extracted)},
                {"hierarchical_transform": "bounded nonlinear feature expansion"},
                {"pairwise_similarity": feature_similarity},
            ],
        }, modality_strengths, 0.0, 0.0)

        # Stage 4: task/context attention and subtractive inhibition of supplied distractors.
        has_goal = bool(request.task_features or request.context_features)
        attention = (
            _sigmoid(self.config.attention_gain * (task_vector + context_vector))
            if has_goal else np.ones(feature_count, dtype=np.float64)
        )
        selected = {
            m: np.maximum(
                0.0,
                values * attention - self.config.inhibition_strength * irrelevant_vector,
            )
            for m, values in extracted.items()
        }
        self._record_stage(4, {
            "attention_weights": _as_feature_map(attention, vocabulary),
            "selected_features": {m: _as_feature_map(v, vocabulary) for m, v in selected.items()},
            "inhibited_features": _as_feature_map(irrelevant_vector, vocabulary),
            "step_evidence": [
                {"goal_present": has_goal},
                {"attention_mean": float(np.mean(attention))},
                {"inhibition_strength": self.config.inhibition_strength},
            ],
        }, modality_strengths, 0.0, float(np.mean(attention)))

        # Stage 5: build nonlinear semantic representations within each modality.
        semantic_representations = {
            m: np.tanh(v + 0.20 * np.square(v) + 0.10 * (v * np.roll(v, 1)))
            for m, v in selected.items()
        }
        self._record_stage(5, {
            "modality_semantic_representations": {
                m: _as_feature_map(v, vocabulary) for m, v in semantic_representations.items()
            },
            "step_evidence": [
                {"constructed_modalities": list(semantic_representations)},
                {"structured_feature_count": feature_count},
                {"nonlinear_interactions": "elementwise feature conjunction proxy"},
            ],
        }, modality_strengths, 0.0, float(np.mean(attention)))

        # Stage 6: retrieve available concept prototypes and spread their supported features.
        pattern_rows: list[dict[str, Any]] = []
        pattern_vectors: list[np.ndarray] = []
        current_pattern = (
            np.mean(np.stack(list(semantic_representations.values())), axis=0)
            if semantic_representations else np.zeros(feature_count)
        )
        for concept in concepts:
            vector = self._concept_vector(concept, vocabulary)
            input_fit = max(0.0, _cosine(current_pattern, vector))
            context_fit = max(0.0, _cosine(context_vector, vector))
            familiarity = min(math.log1p(concept.frequency) / 6.0, 1.0)
            activation = input_fit + 0.20 * context_fit + 0.10 * familiarity
            pattern_vectors.append(vector)
            pattern_rows.append({
                "concept_id": concept.concept_id,
                "label": concept.label,
                "input_similarity": input_fit,
                "context_fit": context_fit,
                "familiarity": familiarity,
                "activation": activation,
            })
        # Associative memory requires a retrieval cue. Familiarity/frequency is a
        # prior that can bias a cue-driven search, but it must not activate a concept
        # when the current modalities, context, and task provide no evidence at all.
        retrieval_evidence = (
            float(np.linalg.norm(current_pattern)) > epsilon
            or float(np.linalg.norm(context_vector)) > epsilon
            or float(np.linalg.norm(task_vector)) > epsilon
        )
        memory_posterior = (
            _softmax([row["activation"] for row in pattern_rows], temperature=0.25)
            if retrieval_evidence else np.zeros(len(pattern_rows), dtype=np.float64)
        )
        for row, probability in zip(pattern_rows, memory_posterior):
            row["posterior"] = float(probability)
        retrieved_memory = (
            sum((float(p) * vector for p, vector in zip(memory_posterior, pattern_vectors)),
                np.zeros(feature_count, dtype=np.float64))
            if retrieval_evidence and pattern_vectors
            else np.zeros(feature_count, dtype=np.float64)
        )
        memory_blended = {
            m: 0.85 * values + 0.15 * retrieved_memory
            for m, values in semantic_representations.items()
        }
        self._record_stage(6, {
            "retrieved_concepts": pattern_rows,
            "retrieval_evidence_available": retrieval_evidence,
            "retrieved_feature_activation": _as_feature_map(retrieved_memory, vocabulary),
            "memory_blended_representations": {
                m: _as_feature_map(v, vocabulary) for m, v in memory_blended.items()
            },
            "step_evidence": [
                {"stored_patterns_compared": len(pattern_rows)},
                {"retrieval_evidence_available": retrieval_evidence},
                {"top_memory_candidates": [row["concept_id"] for row in sorted(
                    pattern_rows, key=lambda row: row["posterior"], reverse=True
                )[:5]]},
                {"spreading_activation_norm": float(np.linalg.norm(retrieved_memory))},
            ],
        }, modality_strengths, 0.0, float(np.mean(attention)))

        # Stage 7: bind channels with explicit temporal, spatial, and learned association evidence.
        binding_matrix: dict[str, dict[str, float]] = {m: {} for m in memory_blended}
        for index, left in enumerate(memory_blended):
            for right in list(memory_blended)[index + 1:]:
                temporal = 0.0
                if left in request.modality_times and right in request.modality_times:
                    delta = abs(
                        float(request.modality_times[left]) - float(request.modality_times[right])
                    )
                    temporal = math.exp(-delta / self.config.temporal_tau)
                spatial = 0.0
                if left in request.spatial_features and right in request.spatial_features:
                    spatial = max(0.0, _cosine(
                        _vector(request.spatial_features[left], vocabulary),
                        _vector(request.spatial_features[right], vocabulary),
                    ))
                key = self._pair_key(left, right)
                learned = float(request.binding_priors.get(
                    key, self._learned_associations.get(key, 0.0)
                ))
                total = (
                    self.config.temporal_weight * temporal
                    + self.config.spatial_weight * spatial
                    + self.config.learned_binding_weight * learned
                )
                binding_matrix[left][right] = float(np.clip(total, 0.0, 1.0))
                binding_matrix[right][left] = binding_matrix[left][right]
        bound: dict[str, np.ndarray] = {}
        for modality, values in memory_blended.items():
            related = [
                (other, strength) for other, strength in binding_matrix[modality].items()
                if strength > 0.0
            ]
            denominator = sum(strength for _, strength in related)
            if denominator > 0.0:
                cross = sum(
                    (strength * memory_blended[other] for other, strength in related),
                    np.zeros(feature_count, dtype=np.float64),
                ) / denominator
                bound[modality] = 0.85 * values + 0.15 * cross
            else:
                bound[modality] = values.copy()
        self._record_stage(7, {
            "binding_strengths": binding_matrix,
            "bound_representations": {m: _as_feature_map(v, vocabulary) for m, v in bound.items()},
            "step_evidence": [
                {"temporal_pairs": sum(
                    1 for left in request.modality_times for right in request.modality_times if left < right
                )},
                {"spatial_pair_count": sum(
                    1 for left in request.spatial_features for right in request.spatial_features if left < right
                )},
                {"learned_association_pairs": len(self._learned_associations)},
                {"bound_channel_count": len(bound)},
            ],
        }, modality_strengths, 0.0, float(np.mean(attention)))

        # Stage 8: project spoke representations into a common ATL input dimensionality.
        projected = {
            modality: self._projection(modality, feature_count) @ values
            for modality, values in bound.items()
        }
        self._record_stage(8, {
            "projected_modalities": {m: v.tolist() for m, v in projected.items()},
            "hub_dimensions": self.config.hub_dimensions,
            "step_evidence": [
                {"projection_shapes": {
                    m: list(self._projection(m, feature_count).shape) for m in projected
                }},
                {"shared_dimension": self.config.hub_dimensions},
            ],
        }, modality_strengths, 0.0, float(np.mean(attention)))

        # Stage 9: estimate precision/reliability, apply task gates, and combine available projections.
        reliability: dict[str, float] = {}
        eligible_modalities: list[str] = []
        reliability_logits: list[float] = []
        for modality in projected:
            uncertainty = float(request.modality_uncertainty.get(modality, 0.25))
            reliability[modality] = 1.0 / (uncertainty * uncertainty + 1e-6)
            task_weight = float(request.task_modality_weights.get(modality, 1.0))
            if task_weight > 0.0:
                eligible_modalities.append(modality)
                reliability_logits.append(
                    math.log(reliability[modality]) + math.log(task_weight)
                )
        modality_weight_values = _softmax(reliability_logits, temperature=1.0)
        modality_weights = {modality: 0.0 for modality in projected}
        modality_weights.update({
            modality: float(weight)
            for modality, weight in zip(eligible_modalities, modality_weight_values)
        })
        combined_input = sum(
            (modality_weights[m] * projected[m] for m in projected),
            np.zeros(self.config.hub_dimensions, dtype=np.float64),
        )
        self._record_stage(9, {
            "reliability": reliability,
            "modality_weights": modality_weights,
            "combined_input": combined_input.tolist(),
            "weight_sum": float(sum(modality_weights.values())),
            "step_evidence": [
                {"reliability": reliability},
                {"normalized_weights": modality_weights},
                {"weight_sum": float(sum(modality_weights.values()))},
                {"task_modality_gates": dict(request.task_modality_weights)},
            ],
        }, modality_strengths, float(np.linalg.norm(combined_input) / math.sqrt(
            self.config.hub_dimensions
        )), float(np.mean(attention)))

        # Stage 10: recurrent nonlinear ATL convergence with bounded leaky updates.
        hub = np.zeros(self.config.hub_dimensions, dtype=np.float64)
        recurrent_deltas: list[float] = []
        for _ in range(self.config.max_recurrent_steps):
            candidate = np.tanh(combined_input + self._hub_recurrent @ hub)
            next_hub = 0.50 * hub + 0.50 * candidate
            delta = float(np.linalg.norm(next_hub - hub))
            recurrent_deltas.append(delta)
            hub = next_hub
            if delta < self.config.convergence_tolerance:
                break
        self._record_stage(10, {
            "hub_input": combined_input.tolist(),
            "integrated_hub": hub.tolist(),
            "recurrent_deltas": recurrent_deltas,
            "converged": bool(
                recurrent_deltas and recurrent_deltas[-1] < self.config.convergence_tolerance
            ),
            "step_evidence": [
                {"weighted_convergence_norm": float(np.linalg.norm(combined_input))},
                {"recurrent_iterations": len(recurrent_deltas)},
                {"final_state_delta": recurrent_deltas[-1] if recurrent_deltas else 0.0},
                {"nonlinear_integration": "tanh with bounded recurrent gain"},
            ],
        }, modality_strengths, float(np.linalg.norm(hub) / math.sqrt(self.config.hub_dimensions)),
            float(np.mean(attention)))

        # Stage 11: estimate the shared route-invariant state and explicitly measure residual diversity.
        route_hubs = {m: np.tanh(values) for m, values in projected.items()}
        route_common = sum(
            (modality_weights[m] * route_hubs[m] for m in route_hubs),
            np.zeros(self.config.hub_dimensions, dtype=np.float64),
        )
        weight_total = max(sum(modality_weights.values()), epsilon)
        route_common /= weight_total
        common = 0.50 * hub + 0.50 * route_common
        cross_modal_similarity = {
            left: {
                right: _cosine(route_hubs[left], route_hubs[right])
                for right in route_hubs
            }
            for left in route_hubs
        }
        invariance_loss = float(sum(
            np.sum(np.square(route_hubs[left] - route_hubs[right]))
            for index, left in enumerate(route_hubs)
            for right in list(route_hubs)[index + 1:]
        ))
        shared_residuals = {
            m: route_hubs[m] - common for m in route_hubs
        }
        self._record_stage(11, {
            "common_representation": common.tolist(),
            "route_specific_hub_states": {m: v.tolist() for m, v in route_hubs.items()},
            "cross_modal_similarity": cross_modal_similarity,
            "invariance_loss": invariance_loss,
            "shared_residual_norms": {
                m: float(np.linalg.norm(v)) for m, v in shared_residuals.items()
            },
            "step_evidence": [
                {"route_count": len(route_hubs)},
                {"cross_modal_similarity": cross_modal_similarity},
                {"invariance_loss": invariance_loss},
                {"unique_residuals_preserved": list(shared_residuals)},
            ],
        }, modality_strengths, float(np.linalg.norm(common) / math.sqrt(
            self.config.hub_dimensions
        )), float(np.mean(attention)))

        # Stage 12: decode the shared state, retain modality residuals, and form task-weighted states.
        modality_reconstructions: dict[str, np.ndarray] = {}
        modality_residuals: dict[str, np.ndarray] = {}
        embodied_representations: dict[str, np.ndarray] = {}
        task_words = request.task.lower()
        for modality, values in semantic_representations.items():
            projection = self._projection(modality, feature_count)
            reconstruction = np.tanh(projection.T @ common)
            residual = values - reconstruction
            common_weight = self.config.common_weight
            embodied_weight = self.config.embodied_weight
            if modality == "visual" and any(word in task_words for word in ("visual", "image", "look", "appear")):
                common_weight, embodied_weight = 0.35, 0.65
            if modality in ("motor", "somatosensory") and any(
                word in task_words for word in ("action", "move", "play", "manipulate")
            ):
                common_weight, embodied_weight = 0.30, 0.70
            modality_reconstructions[modality] = reconstruction
            modality_residuals[modality] = residual
            embodied_representations[modality] = (
                common_weight * reconstruction + embodied_weight * residual
            )
        self._record_stage(12, {
            "reconstructed_modalities": {
                m: _as_feature_map(v, vocabulary) for m, v in modality_reconstructions.items()
            },
            "modality_residuals": {
                m: _as_feature_map(v, vocabulary) for m, v in modality_residuals.items()
            },
            "embodied_representations": {
                m: _as_feature_map(v, vocabulary) for m, v in embodied_representations.items()
            },
            "step_evidence": [
                {"reconstructed_channels": list(modality_reconstructions)},
                {"residual_norms": {
                    m: float(np.linalg.norm(v)) for m, v in modality_residuals.items()
                }},
                {"task": request.task, "common_weight": self.config.common_weight,
                 "embodied_weight": self.config.embodied_weight},
            ],
        }, modality_strengths, float(np.linalg.norm(common) / math.sqrt(
            self.config.hub_dimensions
        )), float(np.mean(attention)))

        # Stage 13: generalize across prototypes but retain concept-level distinctions.
        concept_rows = self._rank_concepts(
            common, vocabulary, context_vector, task_vector, concepts
        )
        category_groups: dict[str, list[np.ndarray]] = {}
        for concept in concepts:
            category_groups.setdefault(concept.category, []).append(
                self._projection("__semantic__", feature_count)
                @ self._concept_vector(concept, vocabulary)
            )
        category_scores: dict[str, float] = {}
        category_posteriors: dict[str, float] = {}
        for category, vectors in category_groups.items():
            prototype = np.mean(np.stack(vectors), axis=0)
            category_scores[category] = max(0.0, _cosine(common, prototype))
        category_evidence_available = (
            float(np.linalg.norm(common)) > epsilon
            or float(np.linalg.norm(context_vector)) > epsilon
            or float(np.linalg.norm(task_vector)) > epsilon
        )
        if category_scores:
            probabilities = (
                _softmax(list(category_scores.values()), temperature=0.25)
                if category_evidence_available
                else np.zeros(len(category_scores), dtype=np.float64)
            )
            category_posteriors = {
                category: float(value)
                for category, value in zip(category_scores, probabilities)
            }
        self._record_stage(13, {
            "concept_candidates": concept_rows,
            "category_scores": category_scores,
            "category_posteriors": category_posteriors,
            "step_evidence": [
                {"concept_prototype_count": len(concepts)},
                {"category_count": len(category_groups)},
                {"category_evidence_available": category_evidence_available},
                {"top_category": (
                    max(category_posteriors, key=category_posteriors.get)
                    if category_posteriors and category_evidence_available else None
                )},
                {"individual_concepts_preserved": [row["concept_id"] for row in concept_rows]},
                {"generalization": "category centroid with distinct concept candidates"},
            ],
        }, modality_strengths, float(np.linalg.norm(common) / math.sqrt(
            self.config.hub_dimensions
        )), float(np.mean(attention)))

        # Stage 14: adapt the shared concept state to current context and task.
        context_projection = self._projection("__context__", feature_count)
        task_projection = self._projection("__task__", feature_count)
        context_hub = context_projection @ context_vector
        task_hub = task_projection @ task_vector
        contextual_state = np.tanh(common + 0.30 * context_hub + 0.20 * task_hub)
        feature_relevance = _softmax(
            np.abs(context_vector) + np.abs(task_vector) + np.abs(current_pattern),
            temperature=0.50,
        )
        weighted_feature_vector = current_pattern * (1.0 + feature_relevance)
        adapted_drive = self._projection("__adapted__", feature_count) @ weighted_feature_vector
        adapted = np.tanh(contextual_state + 0.15 * adapted_drive)
        self._record_stage(14, {
            "context_vector": _as_feature_map(context_vector, vocabulary),
            "contextual_representation": contextual_state.tolist(),
            "contextual_feature_weights": _as_feature_map(feature_relevance, vocabulary),
            "adapted_representation": adapted.tolist(),
            "step_evidence": [
                {"context_feature_count": len(request.context_features)},
                {"context_transform_norm": float(np.linalg.norm(context_hub))},
                {"weighted_features": int(np.count_nonzero(feature_relevance))},
                {"adapted_norm": float(np.linalg.norm(adapted))},
            ],
        }, modality_strengths, float(np.linalg.norm(adapted) / math.sqrt(
            self.config.hub_dimensions
        )), float(np.mean(attention)))

        # Stage 15: reactivate the embodied spokes using task-gated hub-to-modality projections.
        default_gates = {m: 0.50 for m in semantic_representations}
        if any(word in task_words for word in ("look", "visual", "appear", "image")):
            default_gates.update({"visual": 0.90})
        if any(word in task_words for word in ("play", "move", "action", "manipulate")):
            default_gates.update({"motor": 0.90, "somatosensory": 0.75})
        if any(word in task_words for word in ("sound", "hear", "listen")):
            default_gates.update({"auditory": 0.90})
        if any(word in task_words for word in ("say", "name", "describe", "explain")):
            default_gates.update({"linguistic": 0.80})
        modality_gates = {
            m: float(request.task_modality_weights.get(m, default_gates.get(m, 0.50)))
            for m in semantic_representations
        }
        topdown_predictions = {
            m: np.tanh(self._projection(m, feature_count).T @ adapted)
            for m in semantic_representations
        }
        reactivated = {
            m: modality_gates[m] * values for m, values in topdown_predictions.items()
        }
        self._record_stage(15, {
            "topdown_predictions": {
                m: _as_feature_map(v, vocabulary) for m, v in topdown_predictions.items()
            },
            "modality_gates": modality_gates,
            "reactivated_representations": {
                m: _as_feature_map(v, vocabulary) for m, v in reactivated.items()
            },
            "step_evidence": [
                {"topdown_routes": list(topdown_predictions)},
                {"task_gates": modality_gates},
                {"partial_reactivation": {
                    m: float(np.linalg.norm(v)) for m, v in reactivated.items()
                }},
            ],
        }, modality_strengths, float(np.linalg.norm(adapted) / math.sqrt(
            self.config.hub_dimensions
        )), float(np.mean(list(modality_gates.values()))) if modality_gates else 0.0)

        # Stage 16: predict modality evidence, calculate residual errors, and refine the hub.
        prediction_errors: dict[str, np.ndarray] = {}
        prediction_error_norms: dict[str, float] = {}
        prediction_error_hub = np.zeros(self.config.hub_dimensions, dtype=np.float64)
        total_precision = 0.0
        for modality, observed in original.items():
            predicted = topdown_predictions.get(modality, np.zeros(feature_count))
            error = observed - predicted
            prediction_errors[modality] = error
            prediction_error_norms[modality] = float(np.linalg.norm(error))
            uncertainty = float(request.modality_uncertainty.get(modality, 0.25))
            precision = 1.0 / (uncertainty * uncertainty + 1e-6)
            prediction_error_hub += precision * (
                self._projection(modality, feature_count) @ error
            )
            total_precision += precision
        if total_precision > 0.0:
            prediction_error_hub /= total_precision
        predictive_hub = np.tanh(
            adapted + self.config.prediction_update_rate * prediction_error_hub
        )
        mean_error = float(np.mean(list(prediction_error_norms.values()))) if prediction_error_norms else 0.0
        prediction_fit = (
            {
                concept.concept_id: max(
                    0.0,
                    1.0 - float(np.linalg.norm(
                        current_pattern - self._concept_vector(concept, vocabulary)
                    )) / math.sqrt(feature_count),
                )
                for concept in concepts
            }
            if retrieval_evidence else {concept.concept_id: 0.0 for concept in concepts}
        )
        bayesian_rows = self._rank_concepts(
            predictive_hub, vocabulary, context_vector, task_vector, concepts, prediction_fit
        )
        self._record_stage(16, {
            "predicted_modalities": {
                m: _as_feature_map(v, vocabulary) for m, v in topdown_predictions.items()
            },
            "prediction_errors": {
                m: _as_feature_map(v, vocabulary) for m, v in prediction_errors.items()
            },
            "prediction_error_norms": prediction_error_norms,
            "predictive_hub": predictive_hub.tolist(),
            "bayesian_candidate_posterior": bayesian_rows,
            "bayesian_interpretation": "input compatibility combined with concept/context support",
            "step_evidence": [
                {"prediction_routes": list(topdown_predictions)},
                {"mean_prediction_error": mean_error},
                {"hub_update_norm": float(np.linalg.norm(prediction_error_hub))},
                {"candidate_likelihood_fit": prediction_fit},
            ],
        }, modality_strengths, float(np.linalg.norm(predictive_hub) / math.sqrt(
            self.config.hub_dimensions
        )), float(np.mean(list(modality_gates.values()))) if modality_gates else 0.0,
            prediction_error=min(1.0, mean_error / math.sqrt(feature_count)))

        # Stage 17: score responses and retain alternatives when evidence is weak or ambiguous.
        candidate_rows = self._rank_concepts(
            predictive_hub, vocabulary, context_vector, task_vector, concepts, prediction_fit
        )
        best_probability = candidate_rows[0]["posterior"] if candidate_rows else 0.0
        second_probability = candidate_rows[1]["posterior"] if len(candidate_rows) > 1 else 0.0
        confidence_margin = best_probability - second_probability
        # Do not let a frequency prior alone manufacture recognition when all
        # modality channels are gated off and no contextual evidence is present.
        decision_evidence = (
            float(np.linalg.norm(combined_input)) > epsilon
            or float(np.linalg.norm(context_hub)) > epsilon
        )
        selected_row = (
            candidate_rows[0]
            if candidate_rows
            and decision_evidence
            and best_probability >= self.config.confidence_threshold
            and confidence_margin >= self.config.selection_margin
            else None
        )
        selected_pattern = next(
            (item for item in concepts if selected_row and item.concept_id == selected_row["concept_id"]),
            None,
        )
        active_features = sorted(
            _as_feature_map(current_pattern, vocabulary).items(),
            key=lambda item: item[1],
            reverse=True,
        )
        if selected_row is None:
            response = {
                "kind": request.response_mode,
                "abstained": True,
                "reason": "confidence or candidate margin did not meet the configured decision gate",
                "ranked_alternatives": candidate_rows[:5],
                "active_features": [name for name, value in active_features if value > 0.0][:12],
            }
        elif request.response_mode.lower() in ("description", "explain", "describe"):
            response = {
                "kind": "description",
                "concept_id": selected_row["concept_id"],
                "label": selected_row["label"],
                "category": selected_row["category"],
                "supported_features": [
                    name for name, value in sorted(
                        (selected_pattern.features if selected_pattern else {}).items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )[:12]
                ],
                "confidence": best_probability,
            }
        elif request.response_mode.lower() in ("action", "motor", "how_to"):
            response = {
                "kind": "action_reference",
                "concept_id": selected_row["concept_id"],
                "label": selected_row["label"],
                "motor_features": _as_feature_map(
                    reactivated.get("motor", np.zeros(feature_count)), vocabulary
                ),
                "confidence": best_probability,
            }
        else:
            response = {
                "kind": "recognition",
                "concept_id": selected_row["concept_id"],
                "label": selected_row["label"],
                "category": selected_row["category"],
                "confidence": best_probability,
                "alternatives": candidate_rows[1:5],
            }
        self._record_stage(17, {
            "candidate_scores": candidate_rows,
            "selected_concept": selected_row,
            "confidence_margin": confidence_margin,
            "response": response,
            "step_evidence": [
                {"response_candidates": len(candidate_rows)},
                {"top_candidate_score": candidate_rows[0]["score"] if candidate_rows else None},
                {"confidence_margin": confidence_margin},
                {"decision": "selected" if selected_row else "abstained"},
                {"response_mode": request.response_mode},
            ],
        }, modality_strengths, float(np.linalg.norm(predictive_hub) / math.sqrt(
            self.config.hub_dimensions
        )), float(np.mean(list(modality_gates.values()))) if modality_gates else 0.0,
            response_strength=best_probability)

        # Stage 18: compare feedback targets, revise the representation, and schedule bounded reprocessing.
        feedback_available = bool(request.target_features or request.expected_concept_id)
        feedback_target = target_vector.copy()
        if request.expected_concept_id:
            expected = next(
                (item for item in concepts if item.concept_id == request.expected_concept_id), None
            )
            if expected is not None:
                feedback_target = self._concept_vector(expected, vocabulary)
        feedback_error = (
            self._projection("__semantic__", feature_count) @ feedback_target - predictive_hub
            if feedback_available else np.zeros(self.config.hub_dimensions, dtype=np.float64)
        )
        feedback_hub = np.tanh(
            predictive_hub + self.config.prediction_update_rate * feedback_error
        )
        feedback_attention = np.clip(
            attention + 0.10 * (feedback_target - current_pattern)
            if feedback_available else attention,
            0.0, 1.0,
        )
        needs_reprocessing = (
            selected_row is None or best_probability < self.config.confidence_threshold
        )
        reprocessing_requests: list[str] = []
        if needs_reprocessing:
            reprocessing_requests = [
                "feature_extraction",
                "associative_memory_retrieval",
                "contextual_analysis",
                "cross_modal_integration",
            ][:self.config.max_feedback_iterations * 2]
            for _ in range(self.config.max_feedback_iterations):
                feedback_hub = np.tanh(0.80 * feedback_hub + 0.20 * combined_input)
        revised_candidates = self._rank_concepts(
            feedback_hub, vocabulary, context_vector, task_vector, concepts, prediction_fit
        )
        if revised_candidates:
            revised_best = revised_candidates[0]["posterior"]
            revised_second = revised_candidates[1]["posterior"] if len(revised_candidates) > 1 else 0.0
            revised_margin = revised_best - revised_second
        else:
            revised_best = 0.0
            revised_margin = 0.0
        revised_selected = (
            revised_candidates[0]
            if revised_candidates
            and decision_evidence
            and revised_best >= self.config.confidence_threshold
            and revised_margin >= self.config.selection_margin
            else None
        )
        self._record_stage(18, {
            "feedback_available": feedback_available,
            "feedback_error": feedback_error.tolist(),
            "revised_hub": feedback_hub.tolist(),
            "attention_before": _as_feature_map(attention, vocabulary),
            "attention_after": _as_feature_map(feedback_attention, vocabulary),
            "needs_reprocessing": needs_reprocessing,
            "reprocessing_requests": reprocessing_requests,
            "revised_candidates": revised_candidates,
            "revised_selection": revised_selected,
            "step_evidence": [
                {"feedback_available": feedback_available},
                {"feedback_error_norm": float(np.linalg.norm(feedback_error))},
                {"attention_adjustment_norm": float(np.linalg.norm(feedback_attention - attention))},
                {"reprocessing_requests": reprocessing_requests},
            ],
        }, modality_strengths, float(np.linalg.norm(feedback_hub) / math.sqrt(
            self.config.hub_dimensions
        )), float(np.mean(feedback_attention)),
            prediction_error=min(1.0, float(np.linalg.norm(feedback_error)) / math.sqrt(
                self.config.hub_dimensions
            )))

        # Stage 19: propose updates for inspection; commit only verified and explicitly authorized data.
        learning_proposals: list[dict[str, Any]] = []
        committed_examples = 0
        blocked_examples = 0
        plasticity_strength = 0.0
        for example in request.learning_examples:
            shared = (
                dict(example.shared_features)
                if example.shared_features
                else {
                    name: float(np.mean([
                        values.get(name, 0.0) for values in example.modality_inputs.values()
                    ]))
                    for name in vocabulary
                }
            )
            target_shared_vector = _vector(shared, vocabulary)
            target_hub_vector = self._projection("__semantic__", feature_count) @ target_shared_vector
            mapping_proposals: dict[str, float] = {}
            for modality, feature_map in example.modality_inputs.items():
                input_vector = _vector(feature_map, vocabulary)
                mapping = self._projection(modality, feature_count)
                mapping_error = target_hub_vector - mapping @ input_vector
                gradient_update = np.outer(mapping_error, input_vector)
                hebbian_update = np.outer(target_hub_vector, input_vector)
                mapping_proposals[modality] = float(example.confidence * self.config.learning_rate * np.linalg.norm(
                    gradient_update + 0.10 * hebbian_update
                ))
            learning_proposals.append({
                "concept_id": example.concept_id,
                "confidence": example.confidence,
                "hebbian_update_norm": float(example.confidence * np.linalg.norm(
                    target_shared_vector
                )),
                "mapping_update_norms": mapping_proposals,
                "stabilization": "error-driven plus Hebbian update with weight decay and clipping",
                "verified_by": example.verified_by,
                "source_reference": example.source_reference,
            })
            provenance_ok = (
                example.verified_by.strip().lower() not in ("", "unspecified", "unknown")
                and example.source_reference.strip().lower() not in ("", "unspecified", "unknown")
                and example.confidence > 0.0
            )
            if not request.learning_authorized or not provenance_ok:
                blocked_examples += 1
                continue
            # Commit spoke-to-hub mapping changes only after explicit authorization
            # and provenance validation. This is a bounded software learning rule.
            for modality, feature_map in example.modality_inputs.items():
                input_vector = _vector(feature_map, vocabulary)
                mapping = self._projection(modality, feature_count)
                mapping_error = target_hub_vector - mapping @ input_vector
                gradient_update = np.outer(mapping_error, input_vector)
                hebbian_update = np.outer(target_hub_vector, input_vector)
                delta = example.confidence * self.config.learning_rate * (
                    gradient_update + 0.10 * hebbian_update
                ) - self.config.weight_decay * mapping
                mapping[:] = np.clip(mapping + delta, -3.0, 3.0)
            existing = self._learned_memory.get(example.concept_id)
            old_weight = float(existing["weight"]) if existing else 0.0
            new_weight = example.confidence
            denominator = old_weight + new_weight
            if existing:
                old_concept: HybridConcept = existing["concept"]
                old_features = dict(old_concept.features)
                new_features = {
                    key: (
                        old_weight * float(old_features.get(key, 0.0))
                        + new_weight * float(shared.get(key, 0.0))
                    ) / denominator
                    for key in set(old_features) | set(shared)
                }
                old_modalities = {
                    modality: dict(values) for modality, values in old_concept.modality_features.items()
                }
            else:
                new_features = dict(shared)
                old_modalities = {}
            for modality, values in example.modality_inputs.items():
                current = old_modalities.get(modality, {})
                old_modalities[modality] = {
                    key: (
                        old_weight * float(current.get(key, 0.0))
                        + new_weight * float(values.get(key, 0.0))
                    ) / denominator
                    for key in set(current) | set(values)
                }
            static = next(
                (item for item in self.concepts if item.concept_id == example.concept_id), None
            )
            label = static.label if static else example.concept_id
            category = static.category if static else "learned"
            self._learned_memory[example.concept_id] = {
                "concept": HybridConcept(
                    concept_id=example.concept_id,
                    label=label,
                    features=new_features,
                    modality_features=old_modalities,
                    category=category,
                    verified_by=example.verified_by,
                    source_reference=example.source_reference,
                ),
                "weight": denominator,
            }
            active_modalities = list(example.modality_inputs)
            for index, left in enumerate(active_modalities):
                for right in active_modalities[index + 1:]:
                    key = self._pair_key(left, right)
                    coactivation = float(np.mean([
                        min(float(example.modality_inputs[left].get(name, 0.0)),
                            float(example.modality_inputs[right].get(name, 0.0)))
                        for name in set(example.modality_inputs[left]) | set(example.modality_inputs[right])
                    ] or [0.0]))
                    previous = self._learned_associations.get(key, 0.0)
                    delta = self.config.learning_rate * example.confidence * coactivation
                    self._learned_associations[key] = float(np.clip(
                        (1.0 - self.config.weight_decay) * previous + delta, 0.0, 1.0
                    ))
            committed_examples += 1
            plasticity_strength = max(plasticity_strength, example.confidence)
        learning_status = (
            "committed" if committed_examples and request.learning_authorized
            else "preview_only" if learning_proposals and not request.learning_authorized
            else "blocked_missing_provenance" if blocked_examples
            else "no_learning_examples"
        )
        learning = {
            "status": learning_status,
            "authorized": request.learning_authorized,
            "proposed_examples": len(learning_proposals),
            "committed_examples": committed_examples,
            "blocked_examples": blocked_examples,
            "proposals": learning_proposals,
            "learned_memory_size": len(self._learned_memory),
            "learned_association_count": len(self._learned_associations),
            "rules": {
                "hebbian": "Delta W_ij = eta x_i y_j",
                "stabilized": "Delta W_ij = eta x_i y_j - lambda W_ij",
                "experience_average": "H_updated = sum_i alpha_i H_i / sum_i alpha_i",
            },
        }
        self._record_stage(19, {
            "learning": learning,
            "step_evidence": [
                {"new_experiences": len(learning_proposals)},
                {"committed_examples": committed_examples},
                {"bounded_association_links": len(self._learned_associations)},
                {"learning_authorized": request.learning_authorized},
                {"gradual_decay_rate": self.config.weight_decay},
            ],
        }, modality_strengths, float(np.linalg.norm(feedback_hub) / math.sqrt(
            self.config.hub_dimensions
        )), float(np.mean(feedback_attention)), plasticity_strength=plasticity_strength)

        # A feedback-selected concept is preferred only when it passes the same explicit gate.
        final_candidates = revised_candidates if feedback_available or needs_reprocessing else candidate_rows
        final_selected = revised_selected if feedback_available or needs_reprocessing else selected_row
        final_response = dict(response)
        if final_selected is not None:
            final_response.update({
                "abstained": False,
                "concept_id": final_selected["concept_id"],
                "label": final_selected["label"],
                "category": final_selected["category"],
                "confidence": final_selected["posterior"],
            })
        elif final_response.get("kind") == "recognition":
            final_response["abstained"] = True
            final_response["ranked_alternatives"] = final_candidates[:5]

        finite_arrays = [
            common, hub, feedback_hub, combined_input, predictive_hub,
            *original.values(), *semantic_representations.values(),
            *modality_residuals.values(), *reactivated.values(),
        ]
        finite_valid = all(_finite(array) for array in finite_arrays)
        biological = (
            self._biology.snapshot()
            if self._biology is not None
            else {
                "enabled": False,
                "layers": {},
                "operation_trace": [],
                "stage_count": 0,
                "model_scope": "Biological-reference simulation disabled by configuration.",
            }
        )
        diagnostics = {
            "valid": bool(
                finite_valid
                and len(self._current_stage_trace) == 19
                and len(self._current_operation_trace) == len(ALL_OPERATIONS)
            ),
            "finite_numerical_state": finite_valid,
            "stage_count": len(self._current_stage_trace),
            "expected_stage_count": 19,
            "operation_count": len(self._current_operation_trace),
            "expected_operation_count": len(ALL_OPERATIONS),
            "missing_modalities": sorted(set(MODALITIES_FOR_DIAGNOSTICS) - set(original)),
            "selected": final_selected is not None,
            "abstained": final_selected is None,
            "decision_evidence_available": decision_evidence,
            "mean_prediction_error": mean_error,
            "invariance_loss": invariance_loss,
            "biological_numerically_valid": biological.get("numerically_valid", False),
        }
        metadata = {
            "model": "hybrid embodied-amodal concept formation and processing",
            "registry": "ORE-HEC-19",
            "feature_vocabulary": list(vocabulary),
            "available_modalities": list(original),
            "hub_dimensions": self.config.hub_dimensions,
            "recurrent_iterations": len(recurrent_deltas),
            "scientific_scope": (
                "Functional computational approximation plus optional generic LIF reference; "
                "not a fitted or validated reproduction of human neural processing."
            ),
            "input_boundary": (
                "Stage 1 accepts named feature maps from upstream sensors/perception; raw image, "
                "waveform, and chemical-sensor transduction are not implemented in this package."
            ),
        }
        return HybridResult(
            sequence_id=request.sequence_id,
            selected_concept=final_selected,
            candidate_scores=final_candidates,
            common_representation=feedback_hub.tolist(),
            modality_representations={
                m: v.tolist() for m, v in embodied_representations.items()
            },
            modality_residuals={
                m: v.tolist() for m, v in modality_residuals.items()
            },
            reactivated_representations={
                m: v.tolist() for m, v in reactivated.items()
            },
            stage_outputs=self._current_stage_outputs,
            stage_trace=self._current_stage_trace,
            operation_trace=self._current_operation_trace,
            biological=biological,
            learning=learning,
            response=final_response,
            diagnostics=diagnostics,
            metadata=metadata,
        )


MODALITIES_FOR_DIAGNOSTICS = (
    "visual", "auditory", "motor", "somatosensory", "olfactory", "gustatory",
    "emotional", "interoceptive", "linguistic", "internal",
)
