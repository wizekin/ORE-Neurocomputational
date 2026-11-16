"""Executable functional pipeline for 19-stage amodal semantic processing through the ATL.

Functional vector operations and the generic LIF reference simulator are
reported separately. Every stage and sub-operation is traceable. Equations
are modeling choices inspired by neuroscience, not literal neuronal arithmetic.
"""
from __future__ import annotations

from difflib import SequenceMatcher
import hashlib
import math
import re
from typing import Any, Mapping, Sequence

import numpy as np

from .biology import ATLBiologySimulator, biological_drives
from .contracts import (
    AmodalATLConfig,
    AmodalATLRequest,
    AmodalATLResult,
    AmodalConcept,
    MODALITIES,
    normalize_modalities,
)
from .registry import ALL_OPERATIONS, STAGE_REGISTRY


_EPS = 1e-9
_ADJECTIVE_PRIORS = {
    "fake": {"NONAUTHENTIC": 1.0, "IMITATION": 0.9, "DECEPTIVE": 0.7},
    "counterfeit": {"NONAUTHENTIC": 1.0, "IMITATION": 1.0, "DECEPTIVE": 0.8},
    "imitation": {"NONAUTHENTIC": 0.9, "IMITATION": 1.0},
    "small": {"SIZE_SMALL": 1.0, "DIMENSION": 0.7},
    "large": {"SIZE_LARGE": 1.0, "DIMENSION": 0.7},
    "red": {"COLOR_RED": 1.0, "COLOR": 0.8},
    "blue": {"COLOR_BLUE": 1.0, "COLOR": 0.8},
    "real": {"AUTHENTIC": 1.0, "GENUINE": 0.9},
    "genuine": {"AUTHENTIC": 1.0, "GENUINE": 1.0},
}


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Apply a numerically stable logistic activation to every element of a vector."""
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _softmax(values: Sequence[float], temperature: float = 1.0) -> np.ndarray:
    """Convert candidate or modality scores into a normalized probability distribution."""
    vector = np.asarray(values, dtype=np.float64)
    if vector.size == 0:
        return vector
    scaled = vector / max(float(temperature), _EPS)
    scaled -= float(np.max(scaled))
    exp_values = np.exp(np.clip(scaled, -60.0, 60.0))
    total = float(np.sum(exp_values))
    if total <= _EPS or not math.isfinite(total):
        return np.full(vector.size, 1.0 / vector.size, dtype=np.float64)
    return exp_values / total


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    """Calculate cosine similarity while returning zero for a degenerate zero-vector comparison."""
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= _EPS:
        return 0.0
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))


def _unit(values: np.ndarray) -> np.ndarray:
    """Normalize a vector to unit Euclidean length without producing non-finite values."""
    norm = float(np.linalg.norm(values))
    return values.copy() if norm <= _EPS else values / norm


def _tokens(value: str) -> list[str]:
    """Split a cue into normalized Unicode word tokens for simple lexical matching."""
    return re.findall(r"[^\W_]+", str(value).casefold(), flags=re.UNICODE)


def _coerce_concept(value: Any) -> AmodalConcept:
    """Convert an ORE semantic concept, mapping, or compatible object into the amodal ATL concept contract."""
    if isinstance(value, AmodalConcept):
        return value
    if isinstance(value, Mapping):
        read = value.get
    else:
        read = lambda name, default=None: getattr(value, name, default)
    concept_id = str(read("concept_id", read("id", "")))
    label = str(read("label", read("name", concept_id)))
    semantic = read("semantic_features", {}) or {}
    modalities = read("modality_features", {}) or {}
    aliases = tuple(read("aliases", ()) or ())
    category = str(read("category", read("prototype_id", "")) or "")
    source = str(read("source_reference", "unspecified") or "unspecified")
    if not concept_id:
        concept_id = re.sub(r"\W+", "-", label.casefold()).strip("-") or "concept"
    if not label:
        label = concept_id
    return AmodalConcept(
        concept_id=concept_id,
        label=label,
        semantic_features=dict(semantic),
        modality_features={str(k): dict(v) for k, v in modalities.items()},
        aliases=aliases,
        category=category,
        source_reference=source,
    )


class AmodalATLProcessor:
    """Coordinate multimodal encoding, bilateral ATL convergence, phrase composition, feedback, retrieval, and learning."""

    def __init__(
        self,
        concepts: Sequence[AmodalConcept | Mapping[str, Any] | Any] = (),
        config: AmodalATLConfig | None = None,
    ) -> None:
        """Initialize deterministic modality pathways, recurrent bilateral ATL weights, phrase projections, and concept prototypes."""
        self.config = config or AmodalATLConfig()
        self.concepts = tuple(_coerce_concept(item) for item in concepts)
        ids = [concept.concept_id for concept in self.concepts]
        if len(ids) != len(set(ids)):
            raise ValueError("Concept identifiers must be unique")
        self.rng = np.random.default_rng(self.config.random_seed)
        dimension = self.config.hidden_dim
        scale = 1.0 / math.sqrt(dimension)

        def matrix(gain: float = 1.0) -> np.ndarray:
            """Draw one seeded weight matrix with a dimension-scaled initialization."""
            return self.rng.normal(0.0, scale * gain, size=(dimension, dimension))

        self.W_spoke = {name: matrix(0.75) for name in MODALITIES}
        self.b_spoke = {name: np.zeros(dimension, dtype=np.float64) for name in MODALITIES}
        self.W_to_left = {name: matrix(0.85) for name in MODALITIES}
        self.W_to_right = {name: matrix(0.85) for name in MODALITIES}
        self.W_feedback = {name: matrix(0.65) for name in MODALITIES}
        self.W_recurrent_left = matrix(0.20)
        self.W_recurrent_right = matrix(0.20)
        self.W_left_to_right = matrix(0.12)
        self.W_right_to_left = matrix(0.12)
        self.W_context = matrix(0.45)
        self.W_noun = np.eye(dimension, dtype=np.float64) + matrix(0.05)
        self.W_adjective = np.eye(dimension, dtype=np.float64) + matrix(0.05)
        self.W_phrase_noun = np.eye(dimension, dtype=np.float64) * 0.65 + matrix(0.08)
        self.W_phrase_adjective = matrix(0.25)
        self.W_phrase_relation = matrix(0.25)
        self.W_relation = matrix(0.20)
        self.phrase_bias = np.zeros(dimension, dtype=np.float64)
        self._embedding_cache: dict[str, np.ndarray] = {}
        self.model_generation = 0
        self._stage_methods = (
            self._stage_01_receive,
            self._stage_02_extract,
            self._stage_03_activate_spokes,
            self._stage_04_normalize_weight,
            self._stage_05_transmit,
            self._stage_06_integrate_bilateral_atl,
            self._stage_07_common_representation,
            self._stage_08_generalize,
            self._stage_09_feedback,
            self._stage_10_retrieve_concept,
            self._stage_11_activate_noun,
            self._stage_12_maintain_noun,
            self._stage_13_process_adjective,
            self._stage_14_bind_adjective_noun,
            self._stage_15_compose_phrase,
            self._stage_16_denotation,
            self._stage_17_refine_phrase,
            self._stage_18_unified_output,
            self._stage_19_learn_connections,
        )

    def _embedding(self, feature: str, namespace: str) -> np.ndarray:
        """Return a deterministic cached random-feature vector for one named feature and representational namespace."""
        key = f"{namespace}:{feature}".casefold()
        if key not in self._embedding_cache:
            digest = hashlib.sha256(key.encode("utf-8")).digest()
            seed = int.from_bytes(digest[:8], "big", signed=False)
            generator = np.random.default_rng(seed)
            vector = generator.normal(0.0, 1.0, size=self.config.hidden_dim)
            self._embedding_cache[key] = _unit(vector)
        return self._embedding_cache[key]

    def _encode_features(self, features: Mapping[str, float], namespace: str) -> np.ndarray:
        """Project a sparse named feature map into a bounded dense vector using deterministic feature embeddings."""
        if not features:
            return np.zeros(self.config.hidden_dim, dtype=np.float64)
        contributions = [
            float(value) * self._embedding(str(name), namespace)
            for name, value in features.items()
            if float(value) > 0.0
        ]
        if not contributions:
            return np.zeros(self.config.hidden_dim, dtype=np.float64)
        pooled = np.sum(np.stack(contributions), axis=0) / math.sqrt(len(contributions))
        return np.tanh(pooled)

    def _encode_text(self, value: str, namespace: str = "lexical") -> np.ndarray:
        """Represent each cue token as a stable lexical feature and pool the resulting word vectors."""
        tokens = _tokens(value)
        features = {f"LEXEME::{token}": 1.0 for token in tokens}
        return self._encode_features(features, namespace)

    def _concept_vector(self, concept: AmodalConcept) -> np.ndarray:
        """Build a prototype by pooling shared semantic features with the concept's available modality-specific evidence."""
        vectors = []
        if concept.semantic_features:
            vectors.append(self._encode_features(concept.semantic_features, "semantic"))
        for modality, features in normalize_modalities(concept.modality_features).items():
            vectors.append(self._encode_features(features, modality))
        if not vectors:
            return np.zeros(self.config.hidden_dim, dtype=np.float64)
        return _unit(np.tanh(np.mean(np.stack(vectors), axis=0)))

    def _lexical_score(self, cue: str, concept: AmodalConcept) -> float:
        """Score exact, token, alias, and near-string matches between a cue and a stored concept label."""
        cue_tokens = _tokens(cue)
        labels = (concept.label, *concept.aliases)
        if not cue_tokens:
            return 0.0
        best = 0.0
        for label in labels:
            label_tokens = _tokens(label)
            if cue.casefold().strip() == label.casefold().strip():
                best = max(best, 1.0)
            if label_tokens and set(label_tokens).issubset(set(cue_tokens)):
                best = max(best, 0.98)
            for cue_token in cue_tokens:
                for label_token in label_tokens:
                    best = max(best, SequenceMatcher(None, cue_token, label_token).ratio() * 0.85)
        return float(best)

    def _resolve_phrase_parts(self, request: AmodalATLRequest) -> tuple[str, str | None]:
        """Resolve explicit noun and adjective fields or conservatively parse a two-token adjective–noun cue."""
        noun = request.noun or ""
        adjective = request.adjective
        cue_tokens = _tokens(request.text or "")
        if not noun and cue_tokens:
            noun = cue_tokens[-1]
        if adjective is None and request.noun is None and len(cue_tokens) >= 2:
            adjective = cue_tokens[-2]
        return noun, adjective

    def _stage_01_receive(self, state: dict[str, Any]) -> dict[str, Any]:
        """Receive sensory and linguistic evidence while preserving the identity and availability of each input modality."""
        request: AmodalATLRequest = state["request"]
        modalities = normalize_modalities(request.modality_inputs)
        cue = request.text or request.noun or ""
        if cue:
            lexical = modalities.setdefault("linguistic", {})
            for token in _tokens(cue):
                lexical[f"LEXEME::{token}"] = max(lexical.get(f"LEXEME::{token}", 0.0), 1.0)
        if request.noun:
            lexical = modalities.setdefault("linguistic", {})
            for token in _tokens(request.noun):
                lexical[f"NOUN::{token}"] = max(lexical.get(f"NOUN::{token}", 0.0), 1.0)
        state["modalities"] = modalities
        state["available_modalities"] = tuple(sorted(modalities))
        state["context_vector"] = self._encode_features(request.context_features, "context")
        state["noun_text"], state["adjective_text"] = self._resolve_phrase_parts(request)
        return {
            "available_modalities": list(state["available_modalities"]),
            "input_feature_counts": {key: len(value) for key, value in modalities.items()},
            "context_norm": float(np.linalg.norm(state["context_vector"])),
        }

    def _stage_02_extract(self, state: dict[str, Any]) -> dict[str, Any]:
        """Extract modality-specific feature vectors using separate deterministic encoders for each sensory or language channel."""
        state["encoded_inputs"] = {
            modality: self._encode_features(features, modality)
            for modality, features in state["modalities"].items()
        }
        return {
            "encoded_modalities": sorted(state["encoded_inputs"]),
            "encoding_norms": {
                key: float(np.linalg.norm(value))
                for key, value in state["encoded_inputs"].items()
            },
        }

    def _stage_03_activate_spokes(self, state: dict[str, Any]) -> dict[str, Any]:
        """Activate distributed feature populations through a modality-specific nonlinear projection."""
        spokes = {}
        for modality, encoded in state["encoded_inputs"].items():
            activity = _sigmoid(self.W_spoke[modality] @ encoded + self.b_spoke[modality])
            activity[activity < self.config.feature_threshold] = 0.0
            spokes[modality] = activity
        state["spokes"] = spokes
        return {
            "active_spokes": sorted(spokes),
            "population_activity_norms": {
                key: float(np.linalg.norm(value)) for key, value in spokes.items()
            },
            "distributed_population_count": self.config.hidden_dim,
        }

    def _stage_04_normalize_weight(self, state: dict[str, Any]) -> dict[str, Any]:
        """Normalize each spoke pattern and compute reliability-weighted gates for the present modalities."""
        normalized = {}
        reliability = {}
        for modality, activity in state["spokes"].items():
            centered = activity - float(np.mean(activity))
            spread = float(np.std(activity))
            normalized[modality] = np.tanh(centered / (spread + 1e-6))
            coverage = len(state["modalities"][modality]) / (
                len(state["modalities"][modality]) + 4.0
            )
            strength = float(np.mean(np.abs(activity)))
            reliability[modality] = 0.5 * coverage + 0.5 * strength
        names = list(normalized)
        weights = _softmax(
            [reliability[name] for name in names],
            self.config.modality_temperature,
        )
        state["normalized_spokes"] = normalized
        state["modality_reliability"] = reliability
        state["modality_gates"] = dict(zip(names, map(float, weights)))
        return {
            "reliability": reliability,
            "modality_gates": state["modality_gates"],
            "gate_sum": float(np.sum(weights)),
        }

    def _stage_05_transmit(self, state: dict[str, Any]) -> dict[str, Any]:
        """Transmit each normalized spoke pattern through distinct learned left and right ATL projection pathways."""
        left = {}
        right = {}
        for modality, activity in state["normalized_spokes"].items():
            left[modality] = self.W_to_left[modality] @ activity
            right[modality] = self.W_to_right[modality] @ activity
        gates = state["modality_gates"]
        state["left_projections"] = left
        state["right_projections"] = right
        state["left_input"] = sum(
            (gates[name] * value for name, value in left.items()),
            np.zeros(self.config.hidden_dim, dtype=np.float64),
        )
        state["right_input"] = sum(
            (gates[name] * value for name, value in right.items()),
            np.zeros(self.config.hidden_dim, dtype=np.float64),
        )
        return {
            "pathway_modalities": sorted(left),
            "left_input_norm": float(np.linalg.norm(state["left_input"])),
            "right_input_norm": float(np.linalg.norm(state["right_input"])),
        }

    def _stage_06_integrate_bilateral_atl(self, state: dict[str, Any]) -> dict[str, Any]:
        """Integrate sensory projections and context recurrently across modeled left and right ATL populations."""
        left = np.zeros(self.config.hidden_dim, dtype=np.float64)
        right = np.zeros(self.config.hidden_dim, dtype=np.float64)
        context = state["context_vector"]
        deltas = []
        for iteration in range(self.config.recurrent_steps):
            next_left = np.tanh(
                0.35 * (self.W_recurrent_left @ left)
                + 0.12 * (self.W_right_to_left @ right)
                + state["left_input"]
                + 0.12 * (self.W_context @ context)
            )
            next_right = np.tanh(
                0.35 * (self.W_recurrent_right @ right)
                + 0.12 * (self.W_left_to_right @ left)
                + state["right_input"]
                + 0.12 * (self.W_context @ context)
            )
            delta = max(float(np.linalg.norm(next_left - left)),
                        float(np.linalg.norm(next_right - right)))
            deltas.append(delta)
            left, right = next_left, next_right
            if delta <= self.config.convergence_epsilon:
                break
        state["atl_left"] = left
        state["atl_right"] = right
        state["atl_hub"] = _unit(0.5 * (left + right))
        state["integration_iterations"] = len(deltas)
        state["integration_deltas"] = deltas
        return {
            "left_atl_norm": float(np.linalg.norm(left)),
            "right_atl_norm": float(np.linalg.norm(right)),
            "bilateral_hub_norm": float(np.linalg.norm(state["atl_hub"])),
            "iterations": len(deltas),
            "final_delta": deltas[-1] if deltas else 0.0,
        }

    def _stage_07_common_representation(self, state: dict[str, Any]) -> dict[str, Any]:
        """Pool route-specific ATL states into a shared code and retain residuals for modality-unique information."""
        route_states = {}
        for modality, activity in state["normalized_spokes"].items():
            left = self.W_to_left[modality] @ activity
            right = self.W_to_right[modality] @ activity
            route_states[modality] = _unit(np.tanh(0.5 * (left + right)))
        if route_states:
            weights = state["modality_gates"]
            route_mean = sum(
                (weights[name] * vector for name, vector in route_states.items()),
                np.zeros(self.config.hidden_dim, dtype=np.float64),
            )
            shared = _unit(np.tanh(0.5 * state["atl_hub"] + 0.5 * route_mean))
        else:
            shared = state["atl_hub"].copy()
        residuals = {name: vector - shared for name, vector in route_states.items()}
        state["route_states"] = route_states
        state["shared_representation"] = shared
        state["modality_residuals"] = residuals
        similarities = [
            _cosine(route_states[left], route_states[right])
            for index, left in enumerate(route_states)
            for right in list(route_states)[index + 1:]
        ]
        return {
            "shared_representation_norm": float(np.linalg.norm(shared)),
            "route_count": len(route_states),
            "mean_cross_route_similarity": float(np.mean(similarities)) if similarities else None,
            "residual_norms": {
                name: float(np.linalg.norm(value)) for name, value in residuals.items()
            },
        }

    def _stage_08_generalize(self, state: dict[str, Any]) -> dict[str, Any]:
        """Compare the shared state with stored concept prototypes and estimate cross-modal candidate probabilities."""
        prototypes = {
            concept.concept_id: self._concept_vector(concept)
            for concept in self.concepts
        }
        rows = []
        for concept in self.concepts:
            score = _cosine(state["shared_representation"], prototypes[concept.concept_id])
            rows.append({"concept": concept, "prototype_score": score})
        state["concept_prototypes"] = prototypes
        state["generalization_candidates"] = rows
        probabilities = _softmax(
            [row["prototype_score"] for row in rows],
            self.config.modality_temperature,
        )
        for row, probability in zip(rows, probabilities):
            row["probability"] = float(probability)
        return {
            "prototype_count": len(rows),
            "candidate_probabilities": {
                row["concept"].concept_id: row["probability"] for row in rows
            },
        }

    def _stage_09_feedback(self, state: dict[str, Any]) -> dict[str, Any]:
        """Decode the common representation back toward each spoke, calculate prediction errors, and refine working activity."""
        predictions = {}
        errors = {}
        refined = {}
        for modality, observed in state["spokes"].items():
            predicted = np.tanh(self.W_feedback[modality] @ state["shared_representation"])
            predictions[modality] = predicted
            errors[modality] = observed - predicted
            refined[modality] = np.tanh(
                observed + self.config.feedback_gain * predicted
            )
        state["feedback_predictions"] = predictions
        state["feedback_errors"] = errors
        state["refined_spokes"] = refined
        return {
            "prediction_error_norms": {
                key: float(np.linalg.norm(value)) for key, value in errors.items()
            },
            "feedback_gain": self.config.feedback_gain,
            "raw_observations_preserved": True,
        }

    def _stage_10_retrieve_concept(self, state: dict[str, Any]) -> dict[str, Any]:
        """Rank known concepts using shared-state similarity and lexical evidence, and retain uncertainty when evidence is weak."""
        request: AmodalATLRequest = state["request"]
        cue = request.text or state["noun_text"] or ""
        ranked = []
        for row in state["generalization_candidates"]:
            concept = row["concept"]
            semantic_similarity = (1.0 + row["prototype_score"]) / 2.0
            lexical = self._lexical_score(cue, concept)
            score = 0.82 * semantic_similarity + 0.18 * lexical
            ranked.append({
                "concept_id": concept.concept_id,
                "label": concept.label,
                "category": concept.category,
                "semantic_similarity": float(semantic_similarity),
                "lexical_score": float(lexical),
                "score": float(score),
                "source_reference": concept.source_reference,
            })
        ranked.sort(key=lambda item: item["score"], reverse=True)
        selected = ranked[0] if ranked else None
        margin = (
            float(selected["score"] - ranked[1]["score"])
            if selected is not None and len(ranked) > 1
            else (float(selected["score"]) if selected else 0.0)
        )
        accepted = bool(
            selected
            and (selected["lexical_score"] >= 0.95 or selected["score"] >= 0.60)
        )
        state["retrieval"] = {
            "selected": selected if accepted else None,
            "ranked_candidates": ranked,
            "confidence": float(selected["score"]) if selected else 0.0,
            "margin": margin,
            "accepted": accepted,
        }
        return state["retrieval"]

    def _stage_11_activate_noun(self, state: dict[str, Any]) -> dict[str, Any]:
        """Activate a noun prototype from the retrieved concept inventory or encode the supplied noun cue when no prototype is available."""
        request: AmodalATLRequest = state["request"]
        noun = state["noun_text"]
        selected = state["retrieval"]["selected"]
        concept = None
        if noun:
            matching = [
                item for item in self.concepts
                if self._lexical_score(noun, item) >= 0.95
            ]
            concept = matching[0] if matching else None
        elif selected:
            concept = next(
                (item for item in self.concepts
                 if item.concept_id == selected["concept_id"]),
                None,
            )
        if request.noun_features:
            noun_raw = self._encode_features(request.noun_features, "semantic")
        elif concept is not None:
            noun_raw = self._concept_vector(concept)
        else:
            noun_raw = self._encode_text(noun, "semantic")
        noun_state = np.tanh(self.W_noun @ noun_raw)
        noun_state = np.tanh(noun_state + self.W_context @ state["context_vector"] * 0.10)
        state["noun_vector_initial"] = _unit(noun_state)
        state["noun_concept"] = concept
        state["noun_identity"] = noun or (concept.label if concept else "")
        return {
            "noun": state["noun_identity"],
            "matched_concept_id": concept.concept_id if concept else None,
            "noun_representation_norm": float(np.linalg.norm(state["noun_vector_initial"])),
            "noun_source": "concept_prototype" if concept else "lexical_or_feature_encoding",
        }

    def _stage_12_maintain_noun(self, state: dict[str, Any]) -> dict[str, Any]:
        """Maintain noun activation through a leaky recurrent trace while retaining a stable semantic anchor."""
        initial = state["noun_vector_initial"]
        maintained = np.zeros_like(initial)
        rho = self.config.noun_retention
        stability = []
        for iteration in range(max(2, self.config.recurrent_steps // 2)):
            previous = maintained.copy()
            maintained = rho * previous + (1.0 - rho) * initial
            stability.append(_cosine(previous, maintained))
        state["noun_vector"] = _unit(maintained if np.linalg.norm(maintained) > _EPS else initial)
        state["noun_stability"] = stability[-1] if stability else 1.0
        return {
            "retention": rho,
            "maintenance_iterations": len(stability),
            "noun_stability": float(state["noun_stability"]),
            "noun_representation_norm": float(np.linalg.norm(state["noun_vector"])),
        }

    def _stage_13_process_adjective(self, state: dict[str, Any]) -> dict[str, Any]:
        """Encode adjective semantics independently and condition the modifier vector on context and available noun information."""
        request: AmodalATLRequest = state["request"]
        adjective = state["adjective_text"]
        if request.adjective_features:
            features = dict(request.adjective_features)
            source = "caller_features"
        elif adjective and adjective.casefold() in _ADJECTIVE_PRIORS:
            features = _ADJECTIVE_PRIORS[adjective.casefold()]
            source = "built_in_illustrative_prior"
        elif adjective:
            features = {f"MODIFIER::{token}": 1.0 for token in _tokens(adjective)}
            source = "lexical_feature_hash"
        else:
            features = {}
            source = "no_adjective"
        raw = self._encode_features(features, "adjective")
        adjective_state = np.tanh(
            self.W_adjective @ raw + 0.10 * self.W_context @ state["context_vector"]
        )
        state["adjective_vector"] = _unit(adjective_state) if np.linalg.norm(adjective_state) > _EPS else adjective_state
        state["adjective_feature_source"] = source
        return {
            "adjective": adjective,
            "feature_source": source,
            "feature_count": len(features),
            "adjective_representation_norm": float(np.linalg.norm(state["adjective_vector"])),
        }

    def _stage_14_bind_adjective_noun(self, state: dict[str, Any]) -> dict[str, Any]:
        """Construct an adjective–noun interaction using elementwise gating and a learned bilinear compatibility term."""
        noun = state["noun_vector"]
        adjective = state["adjective_vector"]
        elementwise = adjective * noun
        bilinear_scalar = float(adjective @ self.W_relation @ noun) / self.config.hidden_dim
        relation = np.tanh(elementwise + bilinear_scalar * noun)
        state["relation_vector"] = relation
        state["bilinear_interaction"] = bilinear_scalar
        return {
            "elementwise_interaction_norm": float(np.linalg.norm(elementwise)),
            "bilinear_interaction": bilinear_scalar,
            "relation_representation_norm": float(np.linalg.norm(relation)),
        }

    def _stage_15_compose_phrase(self, state: dict[str, Any]) -> dict[str, Any]:
        """Compose noun, adjective, and relation signals into one nonlinear phrase-level semantic representation."""
        if not state["adjective_text"]:
            phrase = state["noun_vector"].copy()
        else:
            phrase = np.tanh(
                self.W_phrase_noun @ state["noun_vector"]
                + self.W_phrase_adjective @ state["adjective_vector"]
                + self.W_phrase_relation @ state["relation_vector"]
                + self.phrase_bias
            )
        state["phrase_vector_initial"] = _unit(phrase)
        return {
            "phrase_representation_norm": float(np.linalg.norm(state["phrase_vector_initial"])),
            "composition_includes_noun": True,
            "composition_includes_adjective": bool(state["adjective_text"]),
            "composition_includes_relation": bool(state["adjective_text"]),
        }

    def _stage_16_denotation(self, state: dict[str, Any]) -> dict[str, Any]:
        """Estimate resemblance and noun-category membership separately so privative modifiers cannot inherit category membership from similarity alone."""
        phrase = state["phrase_vector_initial"]
        noun = state["noun_vector"]
        resemblance = (1.0 + _cosine(phrase, noun)) / 2.0
        adjective_type = state["request"].adjective_type
        if adjective_type == "auto":
            adjective_type = (
                "privative"
                if (state["adjective_text"] or "").casefold() in {"fake", "counterfeit", "imitation"}
                else "subsective"
            )
        if not state["adjective_text"]:
            membership = 1.0
        elif adjective_type == "privative":
            membership = 0.15 * resemblance
        else:
            membership = 0.65 + 0.35 * resemblance
        state["denotation"] = {
            "adjective_type": adjective_type,
            "noun_resemblance": float(resemblance),
            "noun_category_membership": float(membership),
            "belongs_to_noun_category": bool(membership >= 0.50),
        }
        return state["denotation"]

    def _stage_17_refine_phrase(self, state: dict[str, Any]) -> dict[str, Any]:
        """Iteratively update the phrase state toward its current noun–adjective relation target and measure interpretation uncertainty."""
        noun = state["noun_vector"]
        adjective = state["adjective_vector"]
        relation = state["relation_vector"]
        target = np.tanh(
            self.W_phrase_noun @ noun
            + self.W_phrase_adjective @ adjective
            + self.W_phrase_relation @ relation
            + self.phrase_bias
        )
        current = state["phrase_vector_initial"].copy()
        deltas = []
        alpha = self.config.phrase_update_rate
        for _ in range(self.config.recurrent_steps):
            updated = (1.0 - alpha) * current + alpha * target
            deltas.append(float(np.linalg.norm(updated - current)))
            current = updated
            if deltas[-1] <= self.config.convergence_epsilon:
                break
        state["phrase_vector"] = _unit(current)
        membership = float(state["denotation"]["noun_category_membership"])
        bounded = min(max(membership, _EPS), 1.0 - _EPS)
        entropy = -(
            bounded * math.log(bounded)
            + (1.0 - bounded) * math.log(1.0 - bounded)
        )
        state["phrase_uncertainty"] = float(entropy)
        state["phrase_refinement_deltas"] = deltas
        return {
            "iterations": len(deltas),
            "final_delta": deltas[-1] if deltas else 0.0,
            "membership_entropy_nats": float(entropy),
            "phrase_representation_norm": float(np.linalg.norm(state["phrase_vector"])),
        }

    def _stage_18_unified_output(self, state: dict[str, Any]) -> dict[str, Any]:
        """Finalize a unified concept or phrase representation and expose denotation, confidence, and source provenance."""
        phrase_enabled = bool(state["adjective_text"])
        vector = state["phrase_vector"] if phrase_enabled else state["shared_representation"]
        retrieval = state["retrieval"]
        concept = state["noun_concept"]
        noun = state["noun_identity"]
        adjective = state["adjective_text"]
        phrase_label = f"{adjective} {noun}".strip() if adjective else noun
        output = {
            "kind": "phrase" if phrase_enabled else "concept",
            "label": phrase_label or (retrieval["selected"]["label"] if retrieval["selected"] else "unresolved"),
            "noun": noun or None,
            "adjective": adjective,
            "concept_id": concept.concept_id if concept else (
                retrieval["selected"]["concept_id"] if retrieval["selected"] else None
            ),
            "concept_label": concept.label if concept else (
                retrieval["selected"]["label"] if retrieval["selected"] else None
            ),
            "representation": vector.tolist(),
            "representation_norm": float(np.linalg.norm(vector)),
            "retrieval_confidence": float(retrieval["confidence"]),
            "retrieval_accepted": bool(retrieval["accepted"]),
            "denotation": dict(state["denotation"]),
            "uncertainty_nats": float(state["phrase_uncertainty"]),
            "source_reference": state["request"].source_reference,
        }
        state["output"] = output
        return output

    def _learning_pairs(self, state: dict[str, Any]) -> list[tuple[str, np.ndarray, np.ndarray]]:
        """Build modality-to-hub activation pairs from supplied episodes or the current co-active observation."""
        request: AmodalATLRequest = state["request"]
        pairs: list[tuple[str, np.ndarray, np.ndarray]] = []
        for episode in request.learning_experiences:
            modality_features = episode.get("modality_features", {})
            target_features = episode.get("semantic_features", {})
            try:
                normalized = normalize_modalities(modality_features)
            except (TypeError, ValueError):
                continue
            if target_features:
                target = self._encode_features(target_features, "semantic")
            else:
                target = state["shared_representation"]
            for modality, features in normalized.items():
                source = self._encode_features(features, modality)
                pairs.append((modality, source, target))
        if not pairs:
            for modality, source in state["encoded_inputs"].items():
                pairs.append((modality, source, state["shared_representation"]))
        return pairs

    def _stage_19_learn_connections(self, state: dict[str, Any]) -> dict[str, Any]:
        """Compute Hebbian co-activation updates with decay and apply them only when the caller authorizes persistent learning."""
        pairs = self._learning_pairs(state)
        authorized = bool(state["request"].learning_authorized or self.config.learning_authorized)
        learning_rate = self.config.learning_rate
        decay = self.config.weight_decay
        proposals = []
        for modality, source, target in pairs:
            delta_left = learning_rate * (
                np.outer(target, source) - decay * self.W_to_left[modality]
            )
            delta_right = learning_rate * (
                np.outer(target, source) - decay * self.W_to_right[modality]
            )
            proposals.append((modality, delta_left, delta_right))
        norms = {
            modality: {
                "left_delta_norm": float(np.linalg.norm(left)),
                "right_delta_norm": float(np.linalg.norm(right)),
            }
            for modality, left, right in proposals
        }
        if authorized and proposals:
            for modality, left, right in proposals:
                self.W_to_left[modality] = np.clip(
                    self.W_to_left[modality] + left, -2.0, 2.0
                )
                self.W_to_right[modality] = np.clip(
                    self.W_to_right[modality] + right, -2.0, 2.0
                )
            self.model_generation += 1
        state["learning"] = {
            "status": "committed" if authorized and proposals else (
                "preview_only_authorization_required" if proposals else "no_learning_pairs"
            ),
            "learning_authorized": authorized,
            "learning_rate": learning_rate,
            "weight_decay": decay,
            "pair_count": len(proposals),
            "weight_delta_norms": norms,
            "model_generation": self.model_generation,
            "rule": "Delta W = eta * outer(hub_target, modality_input) - eta * lambda * W",
        }
        return state["learning"]

    def process(self, request: AmodalATLRequest) -> AmodalATLResult:
        """Execute all 19 registered functional stages and their corresponding biological reference steps for one request."""
        if not isinstance(request, AmodalATLRequest):
            raise TypeError("request must be an AmodalATLRequest")
        state: dict[str, Any] = {
            "request": request,
            "stage_outputs": {},
            "stage_trace": [],
            "operation_trace": [],
        }
        simulator = ATLBiologySimulator(self.config) if self.config.simulate_biology else None
        for stage_number, (spec, method) in enumerate(
            zip(STAGE_REGISTRY, self._stage_methods), start=1
        ):
            functional_output = method(state)
            modality_activity = state.get("modality_gates", {})
            hub_activity = float(
                np.mean(np.abs(state.get("atl_hub", np.zeros(self.config.hidden_dim))))
            )
            feedback_activity = float(
                np.mean([np.mean(np.abs(v)) for v in state.get("feedback_predictions", {}).values()])
            ) if state.get("feedback_predictions") else 0.0
            phrase_activity = float(
                np.mean(np.abs(state.get("phrase_vector", state.get("phrase_vector_initial",
                               np.zeros(self.config.hidden_dim)))))
            )
            memory_activity = float(
                np.mean(np.abs(state.get("noun_vector", np.zeros(self.config.hidden_dim))))
            )
            if simulator is not None:
                bio_event = simulator.step(
                    spec.stage_id,
                    biological_drives(
                        stage_number,
                        modality_activity,
                        hub_activity,
                        feedback_activity,
                        phrase_activity,
                        memory_activity,
                    ),
                )
                biological_status = "simulated"
            else:
                bio_event = {"stage_id": spec.stage_id, "status": "skipped_disabled"}
                biological_status = "skipped_disabled"
            state["stage_outputs"][spec.stage_id] = functional_output
            stage_trace = {
                "stage_id": spec.stage_id,
                "stage_number": stage_number,
                "name": spec.name,
                "objective": spec.objective,
                "functional_status": "executed",
                "biological_status": biological_status,
                "functional_output": functional_output,
                "biological_event": bio_event,
            }
            state["stage_trace"].append(stage_trace)
            for operation in spec.operations:
                state["operation_trace"].append({
                    "operation_id": operation.operation_id,
                    "stage_id": spec.stage_id,
                    "stage_name": spec.name,
                    "operation": operation.name,
                    "mathematical_operation": operation.mathematical_operation,
                    "functional_responsibility": operation.functional_responsibility,
                    "functional_status": "executed",
                    "biological_status": biological_status,
                    "biological_reference": spec.biological_reference,
                    "explanation": (
                        f"{operation.functional_responsibility} "
                        f"Mathematical model: {operation.mathematical_operation} "
                        f"Biological reference: {spec.biological_reference}"
                    ),
                })

        biological = (
            simulator.snapshot()
            if simulator is not None
            else {
                "enabled": False,
                "stage_count": 0,
                "operation_trace": [],
                "numerically_valid": True,
                "model_scope": "Biological LIF simulation was disabled; functional stages still executed.",
            }
        )
        vectors = [
            state.get(key) for key in (
                "shared_representation", "noun_vector", "adjective_vector",
                "relation_vector", "phrase_vector",
            ) if isinstance(state.get(key), np.ndarray)
        ]
        numerically_valid = all(bool(np.all(np.isfinite(vector))) for vector in vectors)
        diagnostics = {
            "valid": bool(numerically_valid and len(state["stage_trace"]) == 19),
            "stage_count": len(state["stage_trace"]),
            "operation_count": len(state["operation_trace"]),
            "expected_stage_count": 19,
            "expected_operation_count": len(ALL_OPERATIONS),
            "available_modalities": list(state.get("available_modalities", ())),
            "numerically_valid": numerically_valid,
            "model_generation": self.model_generation,
            "scientific_scope": (
                "Computational approximation inspired by ATL hub-and-spoke and semantic "
                "composition theories; not a biologically validated neural simulation."
            ),
        }
        return AmodalATLResult(
            sequence_id=request.sequence_id,
            output=state.get("output", {}),
            stage_outputs=state["stage_outputs"],
            stage_trace=state["stage_trace"],
            operation_trace=state["operation_trace"],
            biological=biological,
            learning=state.get("learning", {}),
            diagnostics=diagnostics,
        )
