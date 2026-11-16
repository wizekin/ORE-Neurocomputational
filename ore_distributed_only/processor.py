"""Executable distributed-only semantic and embodied concept-processing pipeline.

Each concept is represented by separate property-specific population patterns.
The implementation has recurrent inter-population connections, not a central
semantic hub or a single amodal concept vector. Numerical dynamics are
neurocomputational analogues, not a fitted biological simulation.
"""
from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .biology import DistributedLIFConfig, DistributedLIFSimulator
from .contracts import (
    POPULATIONS,
    POPULATION_ALIASES,
    ConceptPattern,
    DistributedSemanticRequest,
    DistributedSemanticResult,
    EpisodicExperience,
)
from .registry import STAGE_REGISTRY


_EPS = 1e-9
_TASK_DEFAULTS = {
    "recognize": {"linguistic": 1.0, "visual": 0.9, "auditory": 0.9},
    "describe": {"visual": 1.0, "tactile": 0.9, "auditory": 0.7, "linguistic": 0.8},
    "play": {"action": 1.0, "auditory": 0.9, "spatial": 0.8},
    "use": {"action": 1.0, "spatial": 0.9, "tactile": 0.8},
    "identify_sound": {"auditory": 1.0, "linguistic": 0.6},
    "compare": {"visual": 0.8, "auditory": 0.8, "tactile": 0.8, "linguistic": 0.7},
    "feel": {"affective": 1.0, "contextual": 0.8},
}


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Apply a numerically stable logistic activation element by element."""
    clipped = np.clip(np.asarray(values, dtype=np.float64), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _softmax(values: Sequence[float], temperature: float = 1.0) -> np.ndarray:
    """Convert finite candidate scores into a stable normalized distribution."""
    vector = np.asarray(values, dtype=np.float64)
    if vector.size == 0:
        return vector
    scaled = vector / max(float(temperature), _EPS)
    scaled -= float(np.max(scaled))
    exponentials = np.exp(np.clip(scaled, -60.0, 60.0))
    total = float(np.sum(exponentials))
    if not math.isfinite(total) or total <= _EPS:
        return np.full(vector.size, 1.0 / vector.size, dtype=np.float64)
    return exponentials / total


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    """Calculate cosine similarity and return zero when either vector is empty."""
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > _EPS else 0.0


def _canonical_population(name: str) -> str:
    """Resolve a caller's modality label to one of the model's distinct populations."""
    value = str(name).strip().casefold().replace("-", "_").replace(" ", "_")
    value = POPULATION_ALIASES.get(value, value)
    if value not in POPULATIONS:
        raise ValueError(f"Unknown population or modality: {name!r}")
    return value


def _bounded_feature_map(features: Mapping[str, float]) -> dict[str, float]:
    """Validate named feature strengths and constrain them to the unit interval."""
    result: dict[str, float] = {}
    for key, raw_value in features.items():
        strength = float(raw_value)
        if not math.isfinite(strength):
            raise ValueError(f"Feature {key!r} has a non-finite strength.")
        result[str(key)] = float(np.clip(strength, 0.0, 1.0))
    return result


def _stable_bucket(feature: str, dimension: int) -> int:
    """Map a named feature deterministically to a fixed population feature slot."""
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dimension


class DistributedOnlyProcessor:
    """Execute all 21 distributed-only semantic stages over property-specific populations.

    Concepts are stored as separate property maps for visual, auditory, action,
    tactile, affective, linguistic, contextual, and spatial populations.
    Inter-population coupling is distributed; there is no single semantic hub.
    """

    def __init__(
        self,
        concepts: Sequence[ConceptPattern] = (),
        *,
        population_size: int = 48,
        random_seed: int = 17,
        recurrent_strength: float = 0.12,
        inhibition_strength: float = 0.08,
        auto_log: bool = True,
        repository_root: str | Path | None = None,
        database_path: str | Path | None = None,
    ) -> None:
        """Initialize concept-specific population patterns, distributed links, and logger settings."""
        if population_size < 8:
            raise ValueError("population_size must be at least 8.")
        self.population_size = int(population_size)
        self.random_seed = int(random_seed)
        self._rng = np.random.default_rng(self.random_seed)
        self.recurrent_strength = float(np.clip(recurrent_strength, 0.0, 1.0))
        self.inhibition_strength = float(np.clip(inhibition_strength, 0.0, 1.0))
        self.concepts: dict[str, ConceptPattern] = {}
        self._concept_vectors: dict[str, dict[str, np.ndarray]] = {}
        self.episodes: list[EpisodicExperience] = []
        self._activation_history: list[np.ndarray] = []
        self._previous_population_activity = np.zeros(len(POPULATIONS), dtype=np.float64)
        self._population_index = {name: i for i, name in enumerate(POPULATIONS)}
        self.population_connectivity = self._initialize_population_connectivity()
        self.within_population_weights = self._initialize_within_population_weights()
        self.biological_simulator = DistributedLIFSimulator(
            DistributedLIFConfig(neuron_count=self.population_size),
            random_seed=self.random_seed,
        )
        self.auto_log = bool(auto_log)
        self.repository_root = Path(repository_root).expanduser().resolve() if repository_root else Path(__file__).resolve().parents[1]
        self.database_path = Path(database_path).expanduser().resolve() if database_path else self.repository_root / "ore_logs" / "ore_logger.sqlite3"
        for concept in concepts:
            self.add_concept(concept)

    def _initialize_population_connectivity(self) -> np.ndarray:
        """Create a symmetric distributed connection graph with no privileged population."""
        count = len(POPULATIONS)
        matrix = self._rng.uniform(0.025, 0.075, size=(count, count))
        matrix = (matrix + matrix.T) / 2.0
        np.fill_diagonal(matrix, 0.0)
        return matrix.astype(np.float64)

    def _initialize_within_population_weights(self) -> dict[str, np.ndarray]:
        """Create separate local feature-association matrices for every population."""
        result: dict[str, np.ndarray] = {}
        for population in POPULATIONS:
            noise = self._rng.normal(0.0, 0.006, size=(self.population_size, self.population_size))
            matrix = np.eye(self.population_size) * 0.04 + noise
            result[population] = matrix.astype(np.float64)
        return result

    def _vectorize(self, features: Mapping[str, float]) -> np.ndarray:
        """Encode named property strengths into one population-local feature vector."""
        vector = np.zeros(self.population_size, dtype=np.float64)
        for feature, strength in _bounded_feature_map(features).items():
            index = _stable_bucket(feature, self.population_size)
            vector[index] = max(vector[index], strength)
        return vector

    def add_concept(self, concept: ConceptPattern) -> None:
        """Register a concept while retaining its property patterns in separate populations."""
        if not isinstance(concept, ConceptPattern):
            raise TypeError("concept must be a ConceptPattern instance.")
        label = concept.label.strip()
        if not label:
            raise ValueError("Concept labels cannot be empty.")
        if label in self.concepts:
            raise ValueError(f"Concept {label!r} is already registered.")
        normalized: dict[str, dict[str, float]] = {}
        vectors = {population: np.zeros(self.population_size, dtype=np.float64) for population in POPULATIONS}
        for raw_population, raw_features in concept.properties.items():
            population = _canonical_population(raw_population)
            features = _bounded_feature_map(raw_features)
            normalized[population] = features
            vectors[population] = self._vectorize(features)
        stored = ConceptPattern(
            label=label,
            properties=normalized,
            category=concept.category or "unspecified",
            description=concept.description,
            typical_action=concept.typical_action,
            aliases=tuple(concept.aliases),
        )
        self.concepts[label] = stored
        self._concept_vectors[label] = vectors

    def add_experience(self, episode: EpisodicExperience) -> None:
        """Store one caller-supplied episode as a collection of population-specific property cues."""
        if episode.concept_label not in self.concepts:
            raise KeyError(f"Unknown concept in episodic experience: {episode.concept_label}")
        if not 0.0 <= episode.familiarity <= 1.0 or not 0.0 <= episode.durability <= 1.0:
            raise ValueError("Episode familiarity and durability must lie in [0, 1].")
        normalized: dict[str, dict[str, float]] = {}
        for raw_population, features in episode.properties.items():
            normalized[_canonical_population(raw_population)] = _bounded_feature_map(features)
        self.episodes.append(EpisodicExperience(
            concept_label=episode.concept_label,
            properties=normalized,
            context=_bounded_feature_map(episode.context),
            familiarity=float(episode.familiarity),
            durability=float(episode.durability),
            provenance=episode.provenance,
        ))

    def _prepare_input_features(self, request: DistributedSemanticRequest) -> dict[str, dict[str, float]]:
        """Collect explicit sensory features, language tokens, context cues, and signal summaries."""
        features = {population: {} for population in POPULATIONS}
        for raw_population, raw_features in request.modality_inputs.items():
            population = _canonical_population(raw_population)
            features[population].update(_bounded_feature_map(raw_features))
        if request.linguistic_input:
            tokens = re.findall(r"[\w'-]+", request.linguistic_input.casefold(), flags=re.UNICODE)
            for token in tokens:
                features["linguistic"][f"token:{token}"] = 1.0
            for left, right in zip(tokens, tokens[1:]):
                features["linguistic"][f"bigram:{left}:{right}"] = 0.7
        if request.context:
            features["contextual"].update(_bounded_feature_map(request.context))
        if request.episodic_cues:
            features["contextual"].update({
                f"episodic_cue:{key}": value
                for key, value in _bounded_feature_map(request.episodic_cues).items()
            })
        for raw_population, signal in request.signal_inputs.items():
            population = _canonical_population(raw_population)
            samples = np.asarray(signal, dtype=np.float64).reshape(-1)
            if samples.size == 0 or not np.all(np.isfinite(samples)):
                raise ValueError(f"Signal for {population!r} must contain finite samples.")
            centered = samples - float(np.mean(samples))
            spectrum = np.abs(np.fft.rfft(centered))
            rms = float(np.sqrt(np.mean(np.square(samples))))
            peak_bin = int(np.argmax(spectrum)) if spectrum.size else 0
            alpha = math.exp(-1.0 / 8.0)
            filtered = np.empty_like(samples)
            filtered[0] = samples[0]
            for sample_index in range(1, samples.size):
                filtered[sample_index] = alpha * filtered[sample_index - 1] + (1.0 - alpha) * samples[sample_index]
            temporal_envelope = float(np.mean(np.abs(filtered)))
            features[population]["signal:mean"] = float(np.clip(abs(np.mean(samples)), 0.0, 1.0))
            features[population]["signal:rms"] = float(np.clip(rms / (1.0 + rms), 0.0, 1.0))
            features[population]["signal:variance"] = float(np.clip(np.var(samples) / (1.0 + np.var(samples)), 0.0, 1.0))
            features[population]["signal:spectral_peak"] = float(peak_bin / max(spectrum.size - 1, 1))
            features[population]["signal:temporal_envelope"] = float(np.clip(temporal_envelope / (1.0 + temporal_envelope), 0.0, 1.0))
        return features

    def _concept_similarity(
        self,
        activity: np.ndarray,
        label: str,
        *,
        weights: np.ndarray | None = None,
        populations: Sequence[str] = POPULATIONS,
    ) -> float:
        """Compare each active population with the matching population-specific concept pattern."""
        pattern = self._concept_vectors[label]
        values: list[float] = []
        local_weights: list[float] = []
        for population in populations:
            index = self._population_index[population]
            score = _cosine(activity[index], pattern[population])
            weight = float(weights[index]) if weights is not None else 1.0
            if np.linalg.norm(activity[index]) > _EPS and np.linalg.norm(pattern[population]) > _EPS:
                values.append(score)
                local_weights.append(weight)
        if not values:
            return 0.0
        return float(np.average(values, weights=np.maximum(local_weights, _EPS)))

    def _candidate_scores(
        self,
        activity: np.ndarray,
        weights: np.ndarray,
    ) -> dict[str, float]:
        """Aggregate separate population-pattern similarities for every known concept."""
        return {
            label: self._concept_similarity(activity, label, weights=weights)
            for label in self.concepts
        }

    def _cue_score(self, activity: np.ndarray, label: str) -> float:
        """Estimate how well currently active population patterns cue a stored concept."""
        return self._concept_similarity(activity, label)

    def _record_stage(self, context: dict[str, Any], stage_number: int, outputs: Sequence[Any]) -> None:
        """Record every named sub-operation with its formula and biological reference."""
        spec = STAGE_REGISTRY[stage_number - 1]
        if len(outputs) != len(spec.operations):
            raise RuntimeError(
                f"{spec.stage_id} emitted {len(outputs)} outputs for "
                f"{len(spec.operations)} registered sub-operations."
            )
        for operation, output in zip(spec.operations, outputs):
            context["operation_trace"].append({
                "stage_id": spec.stage_id,
                "stage_name": spec.name,
                "operation_id": f"{spec.stage_id}.{operation.name}",
                "operation": operation.name,
                "functional_responsibility": operation.functional_responsibility,
                "mathematical_operation": operation.equation,
                "biological_reference": operation.biological_reference,
                "functional_status": "completed",
                "biological_status": "computational_analogue",
                "output": output,
            })
        means = {
            population: float(np.mean(context["activity"][index]))
            for index, population in enumerate(POPULATIONS)
        }
        biological_event = self.biological_simulator.step(
            spec.stage_id,
            means,
            connectivity=self.population_connectivity,
        )
        context.setdefault("biological_trace", []).append(biological_event)
        context["stage_trace"].append({
            "stage_id": spec.stage_id,
            "stage_name": spec.name,
            "phase": spec.phase,
            "objective": spec.objective,
            "biological_interpretation": spec.biological_interpretation,
            "functional_status": "completed",
            "biological_status": "lif_population_simulation",
            "population_activity": means,
            "biological_population_activity": biological_event["population_activity"],
            "sub_operation_count": len(spec.operations),
        })

    def _stage_01_receive_input(self, request: DistributedSemanticRequest, context: dict[str, Any]) -> None:
        """Sample, decompose, normalize, and collect the incoming multimodal evidence."""
        feature_maps = self._prepare_input_features(request)
        raw = np.vstack([self._vectorize(feature_maps[name]) for name in POPULATIONS])
        observed = raw > 0.0
        signal_count = sum(len(values) for values in request.signal_inputs.values())
        context.update({
            "feature_maps": feature_maps,
            "raw_input": raw,
            "observed_mask": observed,
            "activity": raw.copy(),
            "input_anchor": raw.copy(),
        })
        signal_normalized = np.zeros_like(raw)
        for index in range(len(POPULATIONS)):
            active = raw[index][observed[index]]
            if active.size:
                mu = float(np.mean(active))
                sigma = float(np.std(active))
                signal_normalized[index][observed[index]] = np.clip(
                    (active - mu) / (sigma + 1e-6), -3.0, 3.0
                )
        context["normalized_input"] = signal_normalized
        context["history"] = [raw.copy()]
        outputs = [
            {"input_populations": [p for p in POPULATIONS if feature_maps[p]], "feature_count": sum(len(v) for v in feature_maps.values())},
            {"sampled_signal_series": signal_count, "decomposition": "rFFT applied to supplied non-empty signals"},
            {"extracted_property_channels": sum(bool(v) for v in feature_maps.values())},
            {"normalized_dimensions": int(np.count_nonzero(observed))},
            {"temporal_filter": "causal exponential envelope computed for each supplied signal"},
        ]
        self._record_stage(context, 1, outputs)

    def _stage_02_extract_properties(self, context: dict[str, Any]) -> None:
        """Transform input feature strengths into graded property-specific activations."""
        raw = context["raw_input"]
        mask = context["observed_mask"]
        selected = np.where(mask, raw > 0.05, False)
        graded = np.where(selected, _sigmoid(4.0 * (raw - 0.25)), 0.0)
        projection = np.eye(self.population_size, dtype=np.float64)
        reduced = graded @ projection.T
        context["activity"] = reduced
        context["observed_mask"] = selected
        context["input_anchor"] = reduced.copy()
        outputs = [
            {"active_dimensions": int(np.count_nonzero(selected))},
            {"per_population_dimension": self.population_size, "projection": "identity-preserving population-local projection"},
            {"selected_dimensions": int(np.count_nonzero(selected))},
            {"mean_graded_activation": float(np.mean(graded))},
        ]
        self._record_stage(context, 2, outputs)

    def _stage_03_initial_activation(self, context: dict[str, Any]) -> None:
        """Apply input weights, nonlinear response, and initial population thresholds."""
        activity = context["activity"]
        weighted = 1.15 * activity
        nonlinear = _sigmoid(4.0 * weighted - 1.8)
        nonlinear *= context["observed_mask"]
        thresholded = nonlinear > 0.5
        context["activity"] = nonlinear
        context["input_anchor"] = nonlinear.copy()
        context["initial_activity"] = nonlinear.copy()
        context["history"].append(nonlinear.copy())
        outputs = [
            {"input_gain": 1.15, "active_population_count": int(np.count_nonzero(np.mean(weighted, axis=1)))},
            {"mean_nonlinear_activation": float(np.mean(nonlinear))},
            {"thresholded_feature_count": int(np.count_nonzero(thresholded))},
            {"population_means": {p: float(np.mean(nonlinear[i])) for i, p in enumerate(POPULATIONS)}},
        ]
        self._record_stage(context, 3, outputs)

    def _stage_04_retrieve_associations(self, context: dict[str, Any]) -> None:
        """Spread initial activation through learned distributed population and concept associations."""
        activity = context["activity"]
        before = activity.copy()
        means = np.mean(activity, axis=1)
        spread = self.population_connectivity @ means
        activity += self.recurrent_strength * spread[:, None]
        concept_matches = {
            label: self._cue_score(before, label)
            for label in self.concepts
        }
        for label, score in concept_matches.items():
            for index, population in enumerate(POPULATIONS):
                activity[index] += 0.10 * score * self._concept_vectors[label][population]
        context["activity"] = np.clip(activity, 0.0, 1.0)
        outputs = [
            {"mean_spreading_drive": float(np.mean(spread)), "connected_population_pairs": int(np.count_nonzero(self.population_connectivity))},
            {"population_targets": len(POPULATIONS), "concept_association_matches": concept_matches},
        ]
        self._record_stage(context, 4, outputs)

    def _stage_05_reactivate_perception(self, context: dict[str, Any]) -> None:
        """Reinstate perceptual property patterns through the active population collection."""
        targets = ("visual", "auditory", "tactile", "spatial")
        before = context["activity"].copy()
        matches = {label: self._cue_score(before, label) for label in self.concepts}
        activated = 0
        for label, score in matches.items():
            for population in targets:
                index = self._population_index[population]
                vector = self._concept_vectors[label][population]
                context["activity"][index] += 0.12 * score * vector
                activated += int(np.count_nonzero(vector))
        context["activity"] = np.clip(context["activity"], 0.0, 1.0)
        strengths = {p: float(np.sum(context["activity"][self._population_index[p]])) for p in targets}
        similarity = {label: round(score, 6) for label, score in matches.items()}
        outputs = [
            {"perceptual_populations": list(targets), "reactivated_slots": activated},
            {"population_strengths": strengths},
            {"total_perceptual_activation": float(sum(strengths.values()))},
            {"remembered_pattern_similarity": similarity},
        ]
        self._record_stage(context, 5, outputs)

    def _stage_06_reactivate_action(self, context: dict[str, Any]) -> None:
        """Reactivate manipulation patterns, action sequences, and likely next action states."""
        before = context["activity"].copy()
        action_index = self._population_index["action"]
        spatial_index = self._population_index["spatial"]
        matches = {label: self._cue_score(before, label) for label in self.concepts}
        action_drive = 0.0
        for label, score in matches.items():
            action_vector = self._concept_vectors[label]["action"]
            spatial_vector = self._concept_vectors[label]["spatial"]
            context["activity"][action_index] += 0.18 * score * action_vector
            context["activity"][spatial_index] += 0.08 * score * spatial_vector
            action_drive += score * float(np.mean(action_vector))
        action_sequence = [
            float(np.mean(context["activity"][action_index])),
            float(np.mean(context["activity"][spatial_index])),
        ]
        predicted_next = float(np.clip(action_sequence[-1] + 0.25 * action_drive, 0.0, 1.0))
        context["activity"] = np.clip(context["activity"], 0.0, 1.0)
        outputs = [
            {"action_pattern_strength": action_sequence[0]},
            {"action_sequence_state": action_sequence},
            {"predicted_next_action_strength": predicted_next},
            {"action_pattern_similarity": matches},
        ]
        self._record_stage(context, 6, outputs)

    def _stage_07_reactivate_affect(self, request: DistributedSemanticRequest, context: dict[str, Any]) -> None:
        """Reinstate affective features and calculate context-conditioned valuation."""
        index = self._population_index["affective"]
        before = context["activity"].copy()
        matches = {label: self._cue_score(before, label) for label in self.concepts}
        for label, score in matches.items():
            context["activity"][index] += 0.16 * score * self._concept_vectors[label]["affective"]
        context["activity"][index] += 0.05 * self._vectorize(context["feature_maps"]["contextual"])
        affect = np.clip(context["activity"][index], 0.0, 1.0)
        valuation = float(np.mean(affect) * 2.0 - 1.0)
        context_value = float(np.mean(self._vectorize(request.context)))
        conditioned = float(np.clip(valuation * (0.75 + 0.25 * context_value), -1.0, 1.0))
        context["activity"] = np.clip(context["activity"], 0.0, 1.0)
        outputs = [
            {"affective_dimensions": int(np.count_nonzero(affect)), "mean_affect": float(np.mean(affect))},
            {"affective_population_mean": float(np.mean(affect))},
            {"valuation": valuation},
            {"context_conditioned_valuation": conditioned},
        ]
        self._record_stage(context, 7, outputs)

    def _stage_08_retrieve_episodes(self, context: dict[str, Any]) -> None:
        """Retrieve matching episodes and reinstate their population-specific details."""
        activity = context["activity"]
        episode_scores: list[float] = []
        episode_rows: list[dict[str, Any]] = []
        for episode in self.episodes:
            episode_activity = np.vstack([
                self._vectorize(episode.properties.get(population, {}))
                for population in POPULATIONS
            ])
            episode_activity[self._population_index["contextual"]] += self._vectorize(episode.context)
            episode_activity = np.clip(episode_activity, 0.0, 1.0)
            channel_similarities = [
                _cosine(activity[i], episode_activity[i])
                for i in range(len(POPULATIONS))
                if np.linalg.norm(activity[i]) > _EPS and np.linalg.norm(episode_activity[i]) > _EPS
            ]
            cue_similarity = float(np.mean(channel_similarities)) if channel_similarities else 0.0
            score = cue_similarity + 0.2 * episode.familiarity
            episode_scores.append(score)
            episode_rows.append({
                "concept_label": episode.concept_label,
                "cue_similarity": cue_similarity,
                "familiarity": episode.familiarity,
                "durability": episode.durability,
            })
        probabilities = _softmax(episode_scores, temperature=0.35) if episode_scores else np.asarray([])
        for probability, episode, row in zip(probabilities, self.episodes, episode_rows):
            strength = float(probability * episode.durability)
            row["retrieval_probability"] = float(probability)
            row["memory_strength"] = strength
            for index, population in enumerate(POPULATIONS):
                recalled = self._vectorize(episode.properties.get(population, {}))
                activity[index] += strength * 0.10 * recalled
        context["activity"] = np.clip(activity, 0.0, 1.0)
        context["episodic_retrieval"] = episode_rows
        outputs = [
            {"episode_count": len(self.episodes), "cue_similarities": [row["cue_similarity"] for row in episode_rows]},
            {"retrieval_probabilities": probabilities.astype(float).tolist()},
            {"weighted_memory_strength": [row.get("memory_strength", 0.0) for row in episode_rows]},
        ]
        self._record_stage(context, 8, outputs)

    def _stage_09_recurrent_interaction(self, request: DistributedSemanticRequest, context: dict[str, Any]) -> None:
        """Iterate reciprocal population influences with damping and convergence monitoring."""
        activity = context["activity"].copy()
        deltas: list[float] = []
        for _ in range(max(1, int(request.max_iterations))):
            population_means = np.mean(activity, axis=1)
            cross_drive = self.population_connectivity @ population_means
            updated = np.zeros_like(activity)
            for index, population in enumerate(POPULATIONS):
                local_drive = self.within_population_weights[population] @ activity[index]
                target = np.tanh(np.maximum(
                    activity[index] + self.recurrent_strength * cross_drive[index]
                    + 0.08 * local_drive,
                    0.0,
                ))
                updated[index] = 0.75 * activity[index] + 0.25 * target
            delta = float(np.linalg.norm(updated - activity))
            deltas.append(delta)
            activity = np.clip(updated, 0.0, 1.0)
            if delta < request.convergence_tolerance:
                break
        context["activity"] = activity
        context["history"].append(activity.copy())
        context["recurrent_iterations"] = len(deltas)
        context["recurrent_delta"] = deltas[-1] if deltas else 0.0
        outputs = [
            {"iteration_count": len(deltas), "final_delta": deltas[-1] if deltas else 0.0},
            {"reciprocal_pairs": int(np.count_nonzero(self.population_connectivity))},
            {"damping_factor": 0.25, "activity_norm": float(np.linalg.norm(activity))},
            {"convergence_deltas": deltas},
        ]
        self._record_stage(context, 9, outputs)

    def _stage_10_excitation_inhibition(self, context: dict[str, Any]) -> None:
        """Apply excitatory drive, global competition, nonlinear firing, and divisive normalization."""
        activity = context["activity"]
        excitation = np.maximum(activity, 0.0)
        local_pooled_activity = np.mean(excitation, axis=1, keepdims=True)
        inhibitory_drive = self.inhibition_strength * local_pooled_activity
        net = np.maximum(excitation - inhibitory_drive, 0.0)
        fired = _sigmoid(6.0 * (net - 0.22))
        fired *= net > 0.0
        pooled = np.mean(fired, axis=1, keepdims=True)
        normalized = fired / (0.35 + pooled)
        context["activity"] = np.clip(normalized, 0.0, 1.0)
        outputs = [
            {"mean_excitatory_drive": float(np.mean(excitation))},
            {"local_inhibitory_drive": inhibitory_drive.astype(float).tolist()},
            {"mean_net_drive": float(np.mean(net))},
            {"mean_nonlinear_response": float(np.mean(fired))},
            {"mean_divisive_gain": float(np.mean(1.0 / (0.35 + pooled)))},
        ]
        self._record_stage(context, 10, outputs)

    def _stage_11_dynamic_contribution(self, request: DistributedSemanticRequest, context: dict[str, Any]) -> None:
        """Estimate input reliability and compute soft competitive population weights."""
        activity = context["activity"]
        reliability: list[float] = []
        for population in POPULATIONS:
            explicit = request.reliability.get(population, request.reliability.get(
                next((alias for alias, canonical in POPULATION_ALIASES.items() if canonical == population), ""),
                None,
            ))
            index = self._population_index[population]
            active_fraction = float(np.mean(activity[index] > 0.10))
            value = float(explicit) if explicit is not None else 0.35 + 0.65 * active_fraction
            if not math.isfinite(value):
                raise ValueError(f"Reliability for {population!r} must be finite.")
            reliability.append(float(np.clip(value, 0.0, 1.0)))
        weights = _softmax(reliability, temperature=0.35)
        context["reliability"] = np.asarray(reliability, dtype=np.float64)
        context["population_weights"] = weights
        context["weighted_activity"] = activity * weights[:, None]
        outputs = [
            {"reliability": dict(zip(POPULATIONS, reliability))},
            {"normalized_contribution_weights": dict(zip(POPULATIONS, weights.astype(float).tolist()))},
            {"weighted_population_activity_norm": float(np.linalg.norm(context["weighted_activity"]))},
        ]
        self._record_stage(context, 11, outputs)

    def _stage_12_temporal_coordination(self, context: dict[str, Any]) -> None:
        """Estimate population phase coherence and lagged activation correlation."""
        means = np.mean(context["activity"], axis=1)
        frequencies = np.linspace(5.0, 12.0, len(POPULATIONS))
        simulated_time_seconds = 0.02 * max(1, len(context["history"]))
        phases = 2.0 * math.pi * frequencies * simulated_time_seconds + math.pi * means
        phase_matrix = phases[:, None] - phases[None, :]
        phase_coherence = float(abs(np.mean(np.exp(1j * phases))))
        pairwise = np.cos(phase_matrix)
        history = context["history"]
        cross_correlations: dict[str, float] = {}
        if len(history) >= 2:
            previous = np.mean(history[-2], axis=1)
            current = np.mean(history[-1], axis=1)
            for i, left in enumerate(POPULATIONS):
                for j in range(i + 1, len(POPULATIONS)):
                    right = POPULATIONS[j]
                    denom = float(np.linalg.norm([previous[i], current[i]]) * np.linalg.norm([previous[j], current[j]]))
                    cross_correlations[f"{left}:{right}"] = (
                        float((previous[i] * previous[j] + current[i] * current[j]) / denom)
                        if denom > _EPS else 0.0
                    )
        context["timing_coordination"] = {
            "phase_by_population": dict(zip(POPULATIONS, phases.astype(float).tolist())),
            "global_phase_coherence": phase_coherence,
            "pairwise_phase_cosine": pairwise.astype(float).tolist(),
            "temporal_cross_correlation": cross_correlations,
            "simulated_time_seconds": simulated_time_seconds,
        }
        outputs = [
            {"phase_by_population": dict(zip(POPULATIONS, phases.astype(float).tolist()))},
            {"phase_difference_matrix": phase_matrix.astype(float).tolist()},
            {"phase_coherence": phase_coherence},
            {"temporal_cross_correlation": cross_correlations},
        ]
        self._record_stage(context, 12, outputs)

    def _stage_13_construct_assembly(self, context: dict[str, Any]) -> None:
        """Build a distributed cell-assembly readout from co-active population patterns."""
        activity = context["activity"]
        means = np.mean(activity, axis=1)
        coactivation = np.outer(means, means)
        correlations = np.zeros((len(POPULATIONS), len(POPULATIONS)), dtype=np.float64)
        for i in range(len(POPULATIONS)):
            for j in range(len(POPULATIONS)):
                left, right = activity[i], activity[j]
                if np.std(left) > _EPS and np.std(right) > _EPS:
                    correlations[i, j] = float(np.corrcoef(left, right)[0, 1])
        assembly = {
            population: bool(means[index] > 0.02)
            for index, population in enumerate(POPULATIONS)
        }
        context["assembly_membership"] = assembly
        context["coactivation"] = coactivation
        context["population_correlations"] = correlations
        outputs = [
            {"coactivation_matrix": coactivation.astype(float).tolist()},
            {"population_correlation_matrix": correlations.astype(float).tolist()},
            {"assembly_populations": [p for p, active in assembly.items() if active]},
            {"assembly_membership": assembly},
        ]
        self._record_stage(context, 13, outputs)

    def _stage_14_pattern_completion(self, request: DistributedSemanticRequest, context: dict[str, Any]) -> None:
        """Recover unobserved dimensions while clamping every originally observed input feature."""
        activity = context["activity"].copy()
        observed = context["observed_mask"]
        anchor = context["input_anchor"]
        for _ in range(max(1, int(request.max_iterations))):
            means = np.mean(activity, axis=1)
            cross_drive = self.population_connectivity @ means
            updated = np.zeros_like(activity)
            for index, population in enumerate(POPULATIONS):
                local = self.within_population_weights[population] @ activity[index]
                updated[index] = np.clip(
                    activity[index] + 0.08 * cross_drive[index] + 0.03 * local,
                    0.0,
                    1.0,
                )
            for label in self.concepts:
                score = self._cue_score(activity, label)
                for index, population in enumerate(POPULATIONS):
                    missing = ~observed[index]
                    updated[index, missing] += (
                        0.025 * score * self._concept_vectors[label][population][missing]
                    )
            updated = np.clip(updated, 0.0, 1.0)
            updated[observed] = anchor[observed]
            delta = float(np.linalg.norm(updated - activity))
            activity = updated
            if delta < request.convergence_tolerance:
                break
        inferred = (activity > 0.08) & ~observed
        context["activity"] = activity
        context["inferred_mask"] = inferred
        context["completion_delta"] = delta if "delta" in locals() else 0.0
        outputs = [
            {"observed_dimensions": int(np.count_nonzero(observed))},
            {"completed_dimensions": int(np.count_nonzero(inferred)), "iterations": _ + 1},
            {"observed_dimensions_clamped": bool(np.array_equal(activity[observed], anchor[observed]))},
            {"final_completion_delta": context["completion_delta"]},
        ]
        self._record_stage(context, 14, outputs)

    def _stage_15_category_circuits(self, context: dict[str, Any]) -> None:
        """Construct per-category population prototypes and estimate distributed category activation."""
        categories: dict[str, dict[str, np.ndarray]] = {}
        labels_by_category: dict[str, list[str]] = {}
        for label, concept in self.concepts.items():
            category = concept.category
            labels_by_category.setdefault(category, []).append(label)
            if category not in categories:
                categories[category] = {
                    population: np.zeros(self.population_size, dtype=np.float64)
                    for population in POPULATIONS
                }
        for category, labels in labels_by_category.items():
            for population in POPULATIONS:
                categories[category][population] = np.mean(
                    [self._concept_vectors[label][population] for label in labels],
                    axis=0,
                )
        scores: dict[str, float] = {}
        for category, pattern in categories.items():
            per_population = [
                _cosine(context["activity"][index], pattern[population])
                for index, population in enumerate(POPULATIONS)
                if np.linalg.norm(pattern[population]) > _EPS
            ]
            scores[category] = float(np.mean(per_population)) if per_population else 0.0
        context["category_scores"] = scores
        outputs = [
            {"categories": list(categories), "category_count": len(categories)},
            {"category_activation": scores},
            {"distributed_category_similarity": scores},
        ]
        self._record_stage(context, 15, outputs)

    def _stage_16_context_task_modulation(self, request: DistributedSemanticRequest, context: dict[str, Any]) -> None:
        """Apply task relevance and context bias to current activity without changing stored patterns."""
        task_key = request.task.casefold().replace("-", "_").strip()
        defaults = _TASK_DEFAULTS.get(task_key, {})
        relevance = np.asarray([
            float(request.task_relevance.get(population, defaults.get(population, 0.5)))
            for population in POPULATIONS
        ], dtype=np.float64)
        if not np.all(np.isfinite(relevance)):
            raise ValueError("Task relevance values must be finite.")
        relevance = np.clip(relevance, 0.0, 1.0)
        context_strength = float(np.mean(self._vectorize(request.context))) if request.context else 0.0
        previous_activity = context["activity"].copy()
        modulated = previous_activity * (0.85 + 0.30 * relevance[:, None])
        distractor_penalty = 0.04 * (1.0 - relevance[:, None])
        modulated -= distractor_penalty
        modulated[self._population_index["contextual"]] += 0.02 * context_strength
        context["activity"] = np.clip(modulated, 0.0, 1.0)
        modulation_delta = float(np.mean(context["activity"] - previous_activity))
        context["task_relevance"] = dict(zip(POPULATIONS, relevance.astype(float).tolist()))
        outputs = [
            {"task": request.task, "task_vector": context["task_relevance"]},
            {"feature_relevance": context["task_relevance"]},
            {"mean_task_modulation": modulation_delta},
            {"context_strength": context_strength, "context_population_mean": float(np.mean(context["activity"][self._population_index["contextual"]]))},
        ]
        self._record_stage(context, 16, outputs)

    def _stage_17_resolve_competition(self, context: dict[str, Any]) -> None:
        """Score competing interpretations and quantify their normalized support and entropy."""
        weights = context["population_weights"]
        scores = self._candidate_scores(context["activity"], weights)
        labels = list(scores)
        probabilities = _softmax([scores[label] for label in labels], temperature=0.18)
        probability_map = dict(zip(labels, probabilities.astype(float).tolist()))
        ranked = sorted(scores, key=scores.get, reverse=True)
        margin = scores[ranked[0]] - scores[ranked[1]] if len(ranked) > 1 else (scores[ranked[0]] if ranked else 0.0)
        entropy = float(-sum(p * math.log(max(p, _EPS)) for p in probabilities)) if probabilities.size else 0.0
        context["candidate_scores"] = scores
        context["candidate_probabilities"] = probability_map
        context["ranked_candidates"] = ranked
        context["selection_margin"] = float(margin)
        context["interpretation_entropy"] = entropy
        outputs = [
            {"candidate_scores": scores},
            {"candidate_probabilities": probability_map},
            {"selection_margin": float(margin)},
            {"interpretation_entropy": entropy},
        ]
        self._record_stage(context, 17, outputs)

    def _stage_18_generate_response(self, request: DistributedSemanticRequest, context: dict[str, Any]) -> None:
        """Read out the strongest interpretation into a task-dependent text or action response."""
        ranked = context.get("ranked_candidates", [])
        probability_map = context.get("candidate_probabilities", {})
        winner = ranked[0] if ranked else None
        confidence = float(probability_map.get(winner, 0.0)) if winner else 0.0
        margin = float(context.get("selection_margin", 0.0))
        best_score = float(context.get("candidate_scores", {}).get(winner, 0.0)) if winner else 0.0
        accepted = bool(
            winner and best_score > 0.02 and confidence >= 0.40
            and (len(ranked) == 1 or margin >= 0.015)
        )
        selected = winner if accepted else None
        response = "Insufficient distributed evidence to select a concept."
        category = None
        if selected is not None:
            concept = self.concepts[selected]
            category = concept.category
            task = request.task.casefold()
            if task in {"describe", "description"} and concept.description:
                response = concept.description
            elif task in {"use", "play", "action"} and concept.typical_action:
                response = concept.typical_action
            else:
                response = f"Recognized concept: {concept.label}."
        elif ranked:
            response = "Ambiguous interpretation: " + ", ".join(ranked[:3])
        context["interpretation"] = selected
        context["confidence"] = confidence
        context["response"] = response
        context["selected_category"] = category
        outputs = [
            {"distributed_readout_norm": float(np.linalg.norm(context["activity"]))},
            {"selected_concept": selected, "accepted": accepted},
            {"lexical_alternatives": probability_map},
            {"behavioral_response": response, "task": request.task},
        ]
        self._record_stage(context, 18, outputs)

    def _stage_19_update_connections(self, request: DistributedSemanticRequest, context: dict[str, Any]) -> None:
        """Apply explicit Hebbian learning, bounded decay, and optional spike-timing-dependent plasticity."""
        means = np.mean(context["activity"], axis=1)
        enabled = bool(request.learning)
        target = request.learning_target or context.get("interpretation")
        changed = False
        if enabled:
            if target not in self.concepts:
                raise ValueError("learning=True requires learning_target to name a registered concept.")
            raw_rate = float(request.learning_rate)
            if not math.isfinite(raw_rate):
                raise ValueError("learning_rate must be finite.")
            rate = float(np.clip(raw_rate, 0.0, 0.25))
            hebbian = rate * np.outer(means, means)
            updated = self.population_connectivity + hebbian - 0.002 * self.population_connectivity
            np.fill_diagonal(updated, 0.0)
            self.population_connectivity = np.clip(updated, 0.0, 1.0)
            for index, population in enumerate(POPULATIONS):
                vector = context["activity"][index]
                local_update = rate * np.outer(vector, vector)
                local_decay = 0.001 * self.within_population_weights[population]
                self.within_population_weights[population] = np.clip(
                    self.within_population_weights[population] + local_update - local_decay,
                    -0.5,
                    0.5,
                )
            changed = True
            observed_properties = {
                population: dict(context["feature_maps"][population])
                for population in POPULATIONS
                if context["feature_maps"][population]
            }
            if observed_properties:
                self.add_experience(EpisodicExperience(
                    concept_label=target,
                    properties=observed_properties,
                    context=request.context,
                    familiarity=1.0,
                    durability=0.8,
                    provenance="explicitly_authorized_learning_request",
                ))
        spike_times = request.spike_times_ms
        stdp_applied = False
        if enabled and spike_times:
            timing = np.zeros_like(self.population_connectivity)
            supplied = {
                _canonical_population(population): float(time_ms)
                for population, time_ms in spike_times.items()
            }
            for pre, pre_time in supplied.items():
                for post, post_time in supplied.items():
                    if pre == post:
                        continue
                    delta_t = post_time - pre_time
                    if delta_t > 0:
                        change = 0.005 * math.exp(-delta_t / 20.0)
                    elif delta_t < 0:
                        change = -0.005 * math.exp(delta_t / 20.0)
                    else:
                        change = 0.0
                    timing[self._population_index[post], self._population_index[pre]] = change
            self.population_connectivity = np.clip(self.population_connectivity + timing, 0.0, 1.0)
            np.fill_diagonal(self.population_connectivity, 0.0)
            stdp_applied = True
        self._previous_population_activity = means.copy()
        outputs = [
            {"learning_authorized": enabled, "target": target, "hebbian_update_applied": changed},
            {"connection_weight_range": [float(np.min(self.population_connectivity)), float(np.max(self.population_connectivity))]},
            {"weight_decay": 0.002 if enabled else 0.0, "local_weight_decay": 0.001 if enabled else 0.0},
            {"stdp_applied": stdp_applied, "timing_channels": list(spike_times)},
        ]
        self._record_stage(context, 19, outputs)

    def _stage_20_measure_redundancy(self, context: dict[str, Any]) -> None:
        """Measure overlapping property coverage and connection degree across the distributed graph."""
        feature_populations: dict[str, set[str]] = {}
        represented = 0
        for concept in self.concepts.values():
            for population, features in concept.properties.items():
                for feature in features:
                    feature_populations.setdefault(feature, set()).add(population)
                    represented += 1
        overlap = {
            feature: sorted(populations)
            for feature, populations in feature_populations.items()
        }
        degree = np.count_nonzero(self.population_connectivity > 0.0, axis=1)
        total_features = len(feature_populations)
        multiply_represented = sum(len(populations) > 1 for populations in feature_populations.values())
        coverage = float(multiply_represented / total_features) if total_features else 0.0
        resilience = {
            "unique_features": total_features,
            "multiply_represented_features": multiply_represented,
            "overlap_coverage": coverage,
            "concept_property_entries": represented,
        }
        context["redundancy_metrics"] = {
            **resilience,
            "population_degree": dict(zip(POPULATIONS, degree.astype(int).tolist())),
            "feature_population_overlap": overlap,
        }
        outputs = [
            {"feature_population_overlap": overlap},
            {"overlap_coverage": coverage, "represented_features": total_features},
            {"population_degree": dict(zip(POPULATIONS, degree.astype(int).tolist()))},
        ]
        self._record_stage(context, 20, outputs)

    def _stage_21_simulate_damage(self, request: DistributedSemanticRequest, context: dict[str, Any]) -> None:
        """Apply controlled population and connection lesions and quantify semantic performance change."""
        severity = {population: 0.0 for population in POPULATIONS}
        for raw_population, strength in request.lesion_strengths.items():
            population = _canonical_population(raw_population)
            value = float(strength)
            if not math.isfinite(value):
                raise ValueError(f"Lesion severity for {population!r} must be finite.")
            severity[population] = float(np.clip(value, 0.0, 1.0))
        raw_progressive = float(request.progressive_degradation)
        if not math.isfinite(raw_progressive):
            raise ValueError("progressive_degradation must be finite.")
        progressive = float(np.clip(raw_progressive, 0.0, 1.0))
        severity = {
            population: float(1.0 - (1.0 - value) * (1.0 - progressive))
            for population, value in severity.items()
        }
        scales = np.asarray([1.0 - severity[p] for p in POPULATIONS], dtype=np.float64)
        healthy = context["activity"].copy()
        damaged = healthy * scales[:, None]
        damaged_connectivity = self.population_connectivity * scales[:, None] * scales[None, :]
        damaged_means = np.mean(damaged, axis=1)
        propagated = damaged_connectivity @ damaged_means
        damaged = np.clip(damaged + 0.08 * propagated[:, None], 0.0, 1.0)
        weights = context["population_weights"]
        healthy_scores = self._candidate_scores(healthy, weights)
        damaged_scores = self._candidate_scores(damaged, weights)
        deltas = {
            label: float(damaged_scores.get(label, 0.0) - healthy_scores.get(label, 0.0))
            for label in healthy_scores
        }
        healthy_best = max(healthy_scores, key=healthy_scores.get) if healthy_scores else None
        damaged_best = max(damaged_scores, key=damaged_scores.get) if damaged_scores else None
        feature_coverage = float(np.mean(scales))
        context["lesion_analysis"] = {
            "population_severity": severity,
            "progressive_degradation": progressive,
            "population_mask": scales.astype(float).tolist(),
            "healthy_candidate_scores": healthy_scores,
            "damaged_candidate_scores": damaged_scores,
            "candidate_score_delta": deltas,
            "healthy_best_candidate": healthy_best,
            "damaged_best_candidate": damaged_best,
            "mean_population_coverage": feature_coverage,
            "damaged_population_means": dict(zip(POPULATIONS, np.mean(damaged, axis=1).astype(float).tolist())),
            "interpretation": "Counterfactual network lesion; not a clinical disease or patient-outcome simulator.",
        }
        outputs = [
            {"population_lesion_mask": severity},
            {"connection_lesion_mask": damaged_connectivity.astype(float).tolist()},
            {"progressive_degradation": progressive, "effective_severity": severity},
            {"post_lesion_population_means": context["lesion_analysis"]["damaged_population_means"]},
            {"candidate_score_delta": deltas, "healthy_best": healthy_best, "damaged_best": damaged_best, "coverage": feature_coverage},
        ]
        self._record_stage(context, 21, outputs)

    def process(self, request: DistributedSemanticRequest) -> DistributedSemanticResult:
        """Run the complete 21-stage algorithm and optionally persist annotations and traces to SQLite."""
        if not isinstance(request, DistributedSemanticRequest):
            raise TypeError("request must be a DistributedSemanticRequest.")
        context: dict[str, Any] = {
            "operation_trace": [],
            "stage_trace": [],
            "episodic_retrieval": [],
        }
        self._stage_01_receive_input(request, context)
        self._stage_02_extract_properties(context)
        self._stage_03_initial_activation(context)
        self._stage_04_retrieve_associations(context)
        self._stage_05_reactivate_perception(context)
        self._stage_06_reactivate_action(context)
        self._stage_07_reactivate_affect(request, context)
        self._stage_08_retrieve_episodes(context)
        self._stage_09_recurrent_interaction(request, context)
        self._stage_10_excitation_inhibition(context)
        self._stage_11_dynamic_contribution(request, context)
        self._stage_12_temporal_coordination(context)
        self._stage_13_construct_assembly(context)
        self._stage_14_pattern_completion(request, context)
        self._stage_15_category_circuits(context)
        self._stage_16_context_task_modulation(request, context)
        self._stage_17_resolve_competition(context)
        self._stage_18_generate_response(request, context)
        self._stage_19_update_connections(request, context)
        self._stage_20_measure_redundancy(context)
        self._stage_21_simulate_damage(request, context)
        activity = context["activity"]
        observed = context["observed_mask"]
        inferred = context.get("inferred_mask", np.zeros_like(observed))
        means = np.mean(activity, axis=1)
        finite = bool(np.all(np.isfinite(activity)) and np.all(np.isfinite(self.population_connectivity)))
        diagnostics = {
            "valid": finite and len(context["stage_trace"]) == 21,
            "stage_count": len(context["stage_trace"]),
            "operation_count": len(context["operation_trace"]),
            "expected_operation_count": sum(len(stage.operations) for stage in STAGE_REGISTRY),
            "finite_activity": finite,
            "central_semantic_hub": False,
            "representation": "collection of distinct property-specific population states",
            "convergence_delta": context.get("recurrent_delta", 0.0),
            "pattern_completion_delta": context.get("completion_delta", 0.0),
            "selection_margin": context.get("selection_margin", 0.0),
            "interpretation_entropy": context.get("interpretation_entropy", 0.0),
            "scientific_scope": (
                "Mathematical implementation of a distributed-only theory. "
                "The separate LIF reference track is not fitted to neural recordings."
            ),
            "biological_simulation": self.biological_simulator.snapshot(),
        }
        result = DistributedSemanticResult(
            interpretation=context.get("interpretation"),
            category=context.get("selected_category"),
            response=context.get("response", ""),
            confidence=float(context.get("confidence", 0.0)),
            candidate_probabilities=context.get("candidate_probabilities", {}),
            candidate_scores=context.get("candidate_scores", {}),
            category_scores=context.get("category_scores", {}),
            population_activations={
                population: activity[index].astype(float).tolist()
                for index, population in enumerate(POPULATIONS)
            },
            population_means=dict(zip(POPULATIONS, means.astype(float).tolist())),
            population_reliability=dict(zip(
                POPULATIONS,
                context.get("population_weights", np.zeros(len(POPULATIONS))).astype(float).tolist(),
            )),
            observed_indices={
                population: np.flatnonzero(observed[index]).astype(int).tolist()
                for index, population in enumerate(POPULATIONS)
            },
            inferred_indices={
                population: np.flatnonzero(inferred[index]).astype(int).tolist()
                for index, population in enumerate(POPULATIONS)
            },
            assembly_membership=context.get("assembly_membership", {}),
            episodic_retrieval=context.get("episodic_retrieval", []),
            timing_coordination=context.get("timing_coordination", {}),
            redundancy_metrics=context.get("redundancy_metrics", {}),
            lesion_analysis=context.get("lesion_analysis", {}),
            stage_trace=context["stage_trace"],
            operation_trace=context["operation_trace"],
            diagnostics=diagnostics,
        )
        if self.auto_log:
            try:
                from .annotations import record_distributed_result
                result.logging = record_distributed_result(
                    result,
                    repository_root=self.repository_root,
                    database_path=self.database_path,
                )
            except Exception as exc:
                result.logging = {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "database_path": str(self.database_path),
                }
        return result
