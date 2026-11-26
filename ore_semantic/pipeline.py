"""44-operation semantic-cognition pipeline with separate functional and LIF paths."""
from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
from typing import Any, Iterable, Mapping, Sequence
import json
import math
import re
import unicodedata

import numpy as np

from .biology import POPULATIONS, SemanticLIFSimulator
from .contracts import SemanticConfig, SemanticConcept, SemanticRequest, SemanticResult, WordFormEpisode
from .registry import OPERATION_REGISTRY
from .source_mapping import SOURCE_COMPONENTS, source_mapping_as_dicts
from .memory import MEMORY_OPERATION_REGISTRY, SemanticMemoryStore, SemanticPrototypeEvidence

_EPS = 1e-9
_MODALITY_POPULATION = {
    "visual": "visual_spoke",
    "auditory": "auditory_spoke",
    "phonological": "auditory_spoke",
    "action": "action_spoke",
    "motor": "action_spoke",
    "somatosensory": "somatosensory_spoke",
    "tactile": "somatosensory_spoke",
    "olfactory": "chemosensory_interoceptive_spoke",
    "gustatory": "chemosensory_interoceptive_spoke",
    "interoceptive": "chemosensory_interoceptive_spoke",
    "affective": "affective_spoke",
    "emotion": "affective_spoke",
    "spatial": "spatiotemporal_spoke",
    "temporal": "spatiotemporal_spoke",
    "spatiotemporal": "spatiotemporal_spoke",
}


def _normalize(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold().strip()
    return re.sub(r"^\W+|\W+$", "", normalized, flags=re.UNICODE)


def _tokenize(request: SemanticRequest) -> tuple[str, ...]:
    if request.tokens is not None:
        return tuple(str(token) for token in request.tokens)
    assert request.text is not None
    return tuple(re.findall(r"[\w]+(?:['’][\w]+)*", request.text, flags=re.UNICODE))


def _softmax(values: Sequence[float], temperature: float = 1.0) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.size == 0:
        return vector
    scaled = vector / max(float(temperature), _EPS)
    scaled -= np.max(scaled)
    exp = np.exp(np.clip(scaled, -60.0, 60.0))
    total = float(exp.sum())
    if total <= _EPS or not math.isfinite(total):
        return np.full(vector.size, 1.0 / vector.size, dtype=np.float64)
    return exp / total


def _levenshtein(left: str, right: str) -> int:
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
    return previous[-1]


def _string_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return max(0.0, 1.0 - _levenshtein(left, right) / max(len(left), len(right), 1))


def _cosine(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    keys = set(left) | set(right)
    if not keys:
        return 0.0
    a = np.asarray([float(left.get(key, 0.0)) for key in keys], dtype=np.float64)
    b = np.asarray([float(right.get(key, 0.0)) for key in keys], dtype=np.float64)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > _EPS else 0.0


def _entropy(values: Sequence[float]) -> float:
    p = np.asarray(values, dtype=np.float64)
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log(np.maximum(p, _EPS))))


def _weighted_features(
    rows: Sequence[tuple[Mapping[str, float], float]],
) -> dict[str, float]:
    result: dict[str, float] = defaultdict(float)
    total = 0.0
    for features, weight in rows:
        if weight <= 0.0:
            continue
        total += weight
        for name, value in features.items():
            result[str(name)] += float(value) * weight
    if total <= _EPS:
        return {}
    return {name: float(np.clip(value / total, 0.0, 1.0)) for name, value in result.items()}


def _fingerprint(concepts: Sequence[SemanticConcept]) -> str:
    payload = [
        {
            "id": item.concept_id,
            "label": item.label,
            "aliases": list(item.aliases),
            "features": dict(sorted(item.semantic_features.items())),
            "category": item.lexical_category,
            "frequency": item.frequency,
            "prototype_id": item.prototype_id,
            "source": item.source_reference,
        }
        for item in concepts
    ]
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()


def _serializable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_serializable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    return repr(value)


class SemanticCognitionProcessor:
    """Run functional semantic inference plus an optional LIF reference network.

    The processor is read-only: it never changes concept entries, episodes,
    lexical weights, or prototype memory during recognition.
    """

    def __init__(
        self,
        concepts: Iterable[SemanticConcept] = (),
        config: SemanticConfig | None = None,
        memory_store: SemanticMemoryStore | None = None,
    ) -> None:
        """Initialize read-only semantic inference with an optional explicit-learning memory store."""
        self.config = config or SemanticConfig()
        self.memory_store = memory_store
        self.concepts = tuple(concepts)
        ids = [item.concept_id for item in self.concepts]
        if len(ids) != len(set(ids)):
            raise ValueError("concept_id values must be unique")
        self._by_id = {item.concept_id: item for item in self.concepts}
        self._alias_index: dict[str, list[SemanticConcept]] = defaultdict(list)
        for item in self.concepts:
            for alias in (item.label, *item.aliases):
                normalized = _normalize(alias)
                if normalized:
                    self._alias_index[normalized].append(item)
        self.model_fingerprint = _fingerprint(self.concepts)
        self._prototypes = self._build_prototypes()

    def _build_prototypes(self) -> dict[str, dict[str, Any]]:
        grouped: dict[str, list[SemanticConcept]] = defaultdict(list)
        for item in self.concepts:
            group = item.prototype_id or item.lexical_category or "uncategorized"
            grouped[group].append(item)
        prototypes: dict[str, dict[str, Any]] = {}
        for group, members in sorted(grouped.items()):
            features = _weighted_features([
                (item.semantic_features, math.log1p(item.frequency))
                for item in members
            ])
            prototypes[group] = {
                "prototype_id": group,
                "lexical_category": members[0].lexical_category,
                "member_concept_ids": [item.concept_id for item in members],
                "features": features,
                "member_count": len(members),
            }
        return prototypes

    def _retrieve_candidates(
        self,
        token: str,
        episodes: Sequence[Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        normalized = _normalize(token)
        accumulated: dict[str, dict[str, Any]] = {}
        for concept in self._alias_index.get(normalized, ()):
            accumulated[concept.concept_id] = {
                "entry": concept,
                "lexical_similarity": 1.0,
                "exact_match": True,
                "episode_fit": 0.0,
                "episode_ids": [],
                "prototype_evidence_ids": [],
                "phonological_cues": [],
            }
        for concept in self.concepts:
            if concept.concept_id in accumulated:
                continue
            forms = (_normalize(concept.label), *(_normalize(alias) for alias in concept.aliases))
            similarity = max((_string_similarity(normalized, form) for form in forms), default=0.0)
            if similarity >= self.config.candidate_similarity_floor:
                accumulated[concept.concept_id] = {
                    "entry": concept,
                    "lexical_similarity": similarity,
                    "exact_match": False,
                    "episode_fit": 0.0,
                    "episode_ids": [],
                    "prototype_evidence_ids": [],
                    "phonological_cues": [],
                }

        retrieved_episodes: list[dict[str, Any]] = []
        for episode in episodes:
            episode_form = _normalize(episode.orthography)
            similarity = _string_similarity(normalized, episode_form)
            if episode_form == normalized:
                similarity = 1.0
            if similarity < self.config.candidate_similarity_floor:
                continue
            weighted_match = similarity * episode.strength * episode.recency
            evidence_type = getattr(episode, "evidence_type", "episodic_word_form")
            prototype_id = getattr(episode, "prototype_id", None)
            retrieved_episodes.append({
                "episode_id": episode.episode_id,
                "orthography": episode.orthography,
                "concept_id": episode.concept_id,
                "similarity": float(similarity),
                "retrieval_strength": float(weighted_match),
                "phonological_form": list(episode.phonological_form),
                "source_reference": episode.source_reference,
                "evidence_type": evidence_type,
                "prototype_id": prototype_id,
            })
            direct_concept = self._by_id.get(episode.concept_id or "")
            matching_concepts = (
                [direct_concept] if direct_concept is not None else
                [
                    entry for entry in self.concepts
                    if evidence_type == "abstract_lexical_prototype"
                    and entry.prototype_id == prototype_id
                ]
            )
            for concept in matching_concepts:
                item = accumulated.setdefault(concept.concept_id, {
                    "entry": concept,
                    "lexical_similarity": 0.0,
                    "exact_match": False,
                    "episode_fit": 0.0,
                    "episode_ids": [],
                    "prototype_evidence_ids": [],
                    "phonological_cues": [],
                })
                item["episode_fit"] = max(float(item["episode_fit"]), weighted_match)
                if evidence_type == "abstract_lexical_prototype":
                    item["prototype_evidence_ids"].append(str(prototype_id))
                else:
                    item["episode_ids"].append(episode.episode_id)
                if episode.phonological_form:
                    item["phonological_cues"].append(list(episode.phonological_form))
        candidates = list(accumulated.values())
        candidates.sort(key=lambda item: (
            bool(item["exact_match"]),
            float(item["lexical_similarity"]),
            float(item["episode_fit"]),
            math.log1p(item["entry"].frequency),
            item["entry"].concept_id,
        ), reverse=True)
        return candidates[:self.config.max_candidates], retrieved_episodes

    def _candidate_evidence(
        self,
        candidates: list[dict[str, Any]],
        context: Mapping[str, float],
        modality_evidence: Mapping[str, Mapping[str, float]],
    ) -> None:
        for item in candidates:
            concept: SemanticConcept = item["entry"]
            item["context_fit"] = _cosine(concept.semantic_features, context)
            modality_fits: dict[str, float] = {}
            for modality, evidence in modality_evidence.items():
                concept_features = concept.modality_features.get(modality, {})
                modality_fits[modality] = _cosine(concept_features, evidence)
            item["modality_fits"] = modality_fits
            item["modality_fit"] = (
                sum(modality_fits.values()) / len(modality_fits)
                if modality_fits else 0.0
            )
            item["frequency_prior"] = min(math.log1p(concept.frequency) / 10.0, 1.0)
            item["base_score"] = (
                2.0 * float(item["lexical_similarity"])
                + 0.18 * float(item["frequency_prior"])
                + self.config.context_gain * float(item["context_fit"])
                + self.config.episode_gain * float(item["episode_fit"])
                + self.config.modality_gain * float(item["modality_fit"])
            )

    def _spoke_activations(
        self,
        candidates: Sequence[dict[str, Any]],
        modality_evidence: Mapping[str, Mapping[str, float]],
    ) -> tuple[dict[str, float], dict[str, float]]:
        activations: dict[str, float] = {}
        reliability: dict[str, float] = {}
        for modality, evidence in modality_evidence.items():
            population = _MODALITY_POPULATION.get(modality, "distributed_features")
            drive = float(np.mean(tuple(evidence.values()))) if evidence else 0.0
            fits = [float(item.get("modality_fits", {}).get(modality, 0.0)) for item in candidates]
            coverage = sum(1 for item in candidates if item["entry"].modality_features.get(modality)) / max(
                len(candidates), 1
            )
            activations[population] = max(activations.get(population, 0.0), drive * max(fits, default=0.0))
            reliability[modality] = float(np.clip(coverage * (0.5 + 0.5 * drive), 0.0, 1.0))
        return activations, reliability

    def _hub_features(
        self,
        candidates: Sequence[dict[str, Any]],
        probabilities: Sequence[float],
        context: Mapping[str, float],
    ) -> dict[str, float]:
        rows: list[tuple[Mapping[str, float], float]] = []
        for item, probability in zip(candidates, probabilities):
            rows.append((item["entry"].semantic_features, float(probability)))
        if context:
            rows.append((context, 0.25))
        return _weighted_features(rows)

    def _prototype_fit(self, concept: SemanticConcept) -> tuple[str, float]:
        group = concept.prototype_id or concept.lexical_category or "uncategorized"
        prototype = self._prototypes.get(group)
        if not prototype:
            return group, 0.0
        return group, _cosine(concept.semantic_features, prototype["features"])

    def learn_word_form_episode(self, orthography: str, **episode_fields: Any) -> WordFormEpisode:
        """Explicitly capture and store one observed word-form experience."""
        if self.memory_store is None:
            raise RuntimeError("learn_word_form_episode requires a configured SemanticMemoryStore")
        return self.memory_store.capture_word_form(orthography, **episode_fields)

    def consolidate_memory(self, prototype_id: str | None = None, *,
                           learning_rate: float = 0.20) -> dict[str, Any]:
        """Explicitly consolidate stored episodes into durable abstract lexical prototypes."""
        if self.memory_store is None:
            raise RuntimeError("consolidate_memory requires a configured SemanticMemoryStore")
        return self.memory_store.consolidate(prototype_id, learning_rate=learning_rate)

    def process(self, request: SemanticRequest) -> SemanticResult:
        tokens = _tokenize(request)
        if not tokens:
            raise ValueError("request produced no word tokens")
        episodes_by_token: dict[str, tuple[Any, ...]] = {}
        memory_prototypes_by_token: dict[str, list[dict[str, Any]]] = {}
        for token in tokens:
            persistent: list[Any] = []
            prototype_matches: list[dict[str, Any]] = []
            if self.memory_store is not None:
                persistent.extend(
                    self.memory_store.recall(token, context_features=request.context)
                )
                completion = self.memory_store.pattern_complete(
                    orthography=token, context_features=request.context
                )
                if completion.get("status") == "completed":
                    completed_episode = completion["completed_episode"]
                    if all(item.episode_id != completed_episode.episode_id for item in persistent):
                        persistent.append(completed_episode)
                prototype_matches = self.memory_store.generalize(
                    orthography=token, context_features=request.context,
                    limit=3, minimum_similarity=0.25,
                )
                for prototype in prototype_matches:
                    persistent.append(SemanticPrototypeEvidence(
                        episode_id=f"prototype:{prototype['prototype_id']}",
                        orthography=token,
                        concept_id=None,
                        phonological_form=tuple(prototype.get("phonological_sequence", ())),
                        semantic_features=dict(prototype["semantic_features"]),
                        context_features=dict(prototype["context_features"]),
                        strength=float(np.clip(0.4 + 0.4 * prototype["similarity"], 0.0, 1.0)),
                        recency=1.0,
                        source_reference=f"abstract_lexical_prototype:{prototype['prototype_id']}",
                        prototype_id=str(prototype["prototype_id"]),
                    ))
                self.memory_store.record_semantic_feedback(
                    token,
                    episodic_ids=[
                        item.episode_id for item in persistent
                        if getattr(item, "evidence_type", None) != "abstract_lexical_prototype"
                    ],
                    prototype_ids=[item["prototype_id"] for item in prototype_matches],
                    evidence_count=len(persistent),
                )
            memory_prototypes_by_token[token] = prototype_matches
            episodes_by_token[token] = tuple(request.episodes) + tuple(persistent)
        original_text = request.text if request.text is not None else " ".join(request.tokens or ())
        normalized_tokens = tuple(_normalize(token) for token in tokens)
        normalized_text = " ".join(normalized_tokens)
        biological_enabled = bool(self.config.simulate_biology)
        simulator = SemanticLIFSimulator(self.config) if biological_enabled else None
        functional_trace: list[dict[str, Any]] = []
        biological_trace: list[dict[str, Any]] = []

        def tick(
            operation_index: int,
            evidence: Mapping[str, Any],
            drives: Mapping[str, float] | None = None,
        ) -> None:
            spec = OPERATION_REGISTRY[operation_index - 1]
            functional_trace.append({
                "operation_id": spec.operation_id,
                "subsystem": spec.subsystem,
                "name": spec.name,
                "status": "executed",
                "functional_evidence": _serializable(evidence),
            })
            if simulator is not None:
                metrics = simulator.step(drives or {}, spec.operation_id)
                biological_trace.append({
                    "operation_id": spec.operation_id,
                    "status": "executed",
                    **metrics,
                })
            else:
                biological_trace.append({
                    "operation_id": spec.operation_id,
                    "status": "skipped",
                    "reason": "simulate_biology_disabled",
                })

        # 1–6: input integrity, task framing and immutable lexical snapshot.
        tick(1, {"sequence_id": request.sequence_id, "token_count": len(tokens)})
        tick(2, {"surface_text_preserved": True, "original_length": len(original_text)})
        tick(3, {"normalized_text": normalized_text})
        tick(4, {"tokens": list(tokens), "token_indices": list(range(len(tokens)))})
        tick(5, {"task": request.task, "context_feature_count": len(request.context)})
        tick(6, {
            "concept_count": len(self.concepts),
            "model_fingerprint": self.model_fingerprint,
            "inference_is_read_only": True,
        }, {"lexical": 0.20 if self.concepts else 0.0})

        # 7–12: lexical activation and episodic word-form retrieval.
        candidate_sets: list[list[dict[str, Any]]] = []
        episode_records: list[dict[str, Any]] = []
        for token in tokens:
            candidates, retrieved = self._retrieve_candidates(token, episodes_by_token[token])
            candidate_sets.append(candidates)
            episode_records.extend({**row, "token": token} for row in retrieved)
        tick(7, {"tokens_with_lexical_candidates": sum(bool(row) for row in candidate_sets)},
             {"lexical": 0.45})
        tick(8, {"candidate_counts": [len(row) for row in candidate_sets]},
             {"lexical": 0.55})
        phonology_count = sum(
            len(item.get("phonological_cues", ()))
            for candidate_set in candidate_sets for item in candidate_set
        )
        tick(9, {"phonological_episode_cues": phonology_count},
             {"episodic_wordform": 0.35, "auditory_spoke": 0.20})
        tick(10, {"episodes_considered": len(request.episodes),
                  "episodes_retrieved": len(episode_records)},
             {"episodic_wordform": min(1.0, len(episode_records) / max(len(tokens), 1))})
        tick(11, {"episode_matches": len(episode_records),
                  "exact_episode_matches": sum(row["similarity"] >= 1.0 for row in episode_records)},
             {"episodic_wordform": 0.65 if episode_records else 0.0})
        tick(12, {"episode_strength_mean": float(np.mean([
            row["retrieval_strength"] for row in episode_records
        ])) if episode_records else 0.0},
             {"episodic_wordform": 0.75 if episode_records else 0.0})

        # 13–20: modality-specific semantic spokes; absent modalities are not negative evidence.
        self._candidate_evidence(candidate_sets_flat := [
            item for candidate_set in candidate_sets for item in candidate_set
        ], request.context, request.modality_evidence)
        spoke_by_population, spoke_reliability = self._spoke_activations(
            candidate_sets_flat, request.modality_evidence
        )
        modality_values = {
            modality: float(np.mean(tuple(features.values()))) if features else 0.0
            for modality, features in request.modality_evidence.items()
        }
        modality_to_operation = (
            ("visual", 13, "visual_spoke"),
            ("auditory", 14, "auditory_spoke"),
            ("phonological", 14, "auditory_spoke"),
            ("action", 15, "action_spoke"),
            ("motor", 15, "action_spoke"),
            ("somatosensory", 16, "somatosensory_spoke"),
            ("tactile", 16, "somatosensory_spoke"),
            ("olfactory", 17, "chemosensory_interoceptive_spoke"),
            ("gustatory", 17, "chemosensory_interoceptive_spoke"),
            ("interoceptive", 17, "chemosensory_interoceptive_spoke"),
            ("affective", 18, "affective_spoke"),
            ("emotion", 18, "affective_spoke"),
            ("spatial", 19, "spatiotemporal_spoke"),
            ("temporal", 19, "spatiotemporal_spoke"),
            ("spatiotemporal", 19, "spatiotemporal_spoke"),
        )
        operation_drives: dict[int, dict[str, float]] = {
            index: {} for index in range(13, 20)
        }
        for modality, operation_index, population in modality_to_operation:
            if modality in modality_values:
                operation_drives[operation_index][population] = max(
                    operation_drives[operation_index].get(population, 0.0),
                    modality_values[modality],
                )
        tick(13, {"visual_spoke_activation": spoke_by_population.get("visual_spoke", 0.0)},
             operation_drives[13])
        tick(14, {"auditory_spoke_activation": spoke_by_population.get("auditory_spoke", 0.0)},
             operation_drives[14])
        tick(15, {"action_spoke_activation": spoke_by_population.get("action_spoke", 0.0)},
             operation_drives[15])
        tick(16, {"somatosensory_spoke_activation": spoke_by_population.get("somatosensory_spoke", 0.0)},
             operation_drives[16])
        tick(17, {"chemosensory_interoceptive_activation":
                  spoke_by_population.get("chemosensory_interoceptive_spoke", 0.0)},
             operation_drives[17])
        tick(18, {"affective_spoke_activation": spoke_by_population.get("affective_spoke", 0.0)},
             operation_drives[18])
        tick(19, {"spatiotemporal_spoke_activation":
                  spoke_by_population.get("spatiotemporal_spoke", 0.0)},
             operation_drives[19])
        tick(20, {"spoke_reliability": spoke_reliability,
                  "missing_modalities_are_neutral": True},
             {"distributed_features": min(1.0, float(np.mean(list(spoke_reliability.values())))
                                          if spoke_reliability else 0.0)})

        # 21–26: distributed feature networks and evidence-preserving feature state.
        initial_probabilities: list[np.ndarray] = []
        for candidate_set in candidate_sets:
            if not candidate_set:
                initial_probabilities.append(np.zeros(0, dtype=np.float64))
                continue
            scores = [float(item["base_score"]) for item in candidate_set]
            probabilities = _softmax(scores)
            for _ in range(self.config.interactive_steps):
                proposal_scores = (
                    np.asarray(scores)
                    + self.config.feedback_gain * probabilities
                    - self.config.lateral_inhibition * (probabilities.sum() - probabilities)
                )
                proposal = _softmax(proposal_scores)
                probabilities = 0.65 * probabilities + 0.35 * proposal
                probabilities /= max(float(probabilities.sum()), _EPS)
            initial_probabilities.append(probabilities)
        flat_probabilities = [
            float(probability)
            for row in initial_probabilities for probability in row
        ]
        initial_hub = self._hub_features(candidate_sets_flat, flat_probabilities, request.context)
        tick(21, {"distributed_candidate_feature_count": len({
            key for item in candidate_sets_flat for key in item["entry"].semantic_features
        })}, {"distributed_features": 0.45})
        tick(22, {"association_propagation": "bounded candidate-weighted feature pooling"},
             {"distributed_features": 0.60})
        tick(23, {"integrated_feature_count": len(initial_hub),
                  "integrated_features": initial_hub},
             {"distributed_features": 0.70})
        sparse_features = {
            key: value for key, value in initial_hub.items() if value >= 0.05
        }
        tick(24, {"feature_count_before": len(initial_hub),
                  "feature_count_after": len(sparse_features)},
             {"distributed_features": 0.75})
        observed_features = dict(sparse_features)
        tick(25, {"observed_feature_count": len(observed_features),
                  "feature_channels_preserved": True},
             {"distributed_features": 0.80})
        feature_error = {
            key: float(max(0.0, value - request.context.get(key, 0.0)))
            for key, value in observed_features.items()
        }
        tick(26, {"feature_prediction_error": feature_error},
             {"distributed_features": 0.85})

        # 27–32: ATL-like transmodal hub, recurrent settling and lexical feedback.
        hub_state = dict(observed_features)
        for _ in range(self.config.interactive_steps):
            revised = _weighted_features([
                (hub_state, 0.70),
                (initial_hub, 0.20),
                (request.context, 0.10),
            ])
            keys = set(hub_state) | set(revised)
            delta = max((abs(revised.get(k, 0.0) - hub_state.get(k, 0.0)) for k in keys),
                        default=0.0)
            hub_state = revised
            if delta < 1e-4:
                break
        completed_features = dict(hub_state)
        for item in candidate_sets_flat:
            for key, value in item["entry"].semantic_features.items():
                completed_features[key] = max(completed_features.get(key, 0.0),
                                              float(value) * 0.15)
        tick(27, {"hub_seed_feature_count": len(observed_features)},
             {"atl_hub": 0.45})
        tick(28, {"contributing_modalities": sorted(request.modality_evidence),
                  "reliability_weights": spoke_reliability},
             {"atl_hub": 0.65, "distributed_features": 0.50})
        tick(29, {"hub_settling_iterations": self.config.interactive_steps,
                  "settled_feature_count": len(hub_state)},
             {"atl_hub": 0.85})
        tick(30, {"completed_feature_count": len(completed_features),
                  "observed_features_preserved": True},
             {"atl_hub": 0.90})
        tick(31, {"hub_activation_feature_count": len(hub_state)},
             {"atl_hub": 0.90, "lexical": 0.45})
        tick(32, {"hub_to_lexical_feedback_gain": self.config.feedback_gain},
             {"atl_hub": 0.70, "lexical": 0.65})

        # 33–39: inferior-frontal semantic control and bounded concept selection.
        preliminary_posteriors: list[np.ndarray] = []
        control_demands: list[float] = []
        for candidate_set in candidate_sets:
            if not candidate_set:
                preliminary_posteriors.append(np.zeros(0, dtype=np.float64))
                control_demands.append(1.0)
                continue
            raw = []
            for item in candidate_set:
                item["hub_fit"] = _cosine(item["entry"].semantic_features, hub_state)
                item["preliminary_score"] = (
                    float(item["base_score"])
                    + 0.25 * float(item["hub_fit"])
                    + 0.15 * float(item["context_fit"])
                )
                raw.append(float(item["preliminary_score"]))
            posterior = _softmax(raw)
            normalized_entropy = _entropy(posterior) / max(math.log(len(posterior)), 1.0)
            control_demands.append(float(np.clip(normalized_entropy, 0.0, 1.0)))
            preliminary_posteriors.append(posterior)
        mean_control_demand = float(np.mean(control_demands)) if control_demands else 1.0
        tick(33, {"semantic_control_demand": mean_control_demand},
             {"ifg_control": mean_control_demand})
        task_targets = {
            "action": {"action", "motor", "verb"},
            "perception": {"visual", "auditory", "sensory"},
            "emotion": {"affective", "emotion", "feeling"},
            "category": {"noun", "entity", "category"},
        }.get(request.task.casefold(), set())
        task_relevance = {
            item["entry"].concept_id: (
                1.0 if not task_targets or item["entry"].lexical_category.casefold() in task_targets
                else 0.5
            )
            for item in candidate_sets_flat
        }
        tick(34, {"task": request.task, "task_target_categories": sorted(task_targets)},
             {"ifg_control": min(1.0, mean_control_demand + 0.10)})
        tick(35, {"competitor_count": sum(max(0, len(row) - 1) for row in candidate_sets)},
             {"ifg_control": min(1.0, mean_control_demand + 0.20)})
        tick(36, {"lateral_inhibition_gain": self.config.lateral_inhibition},
             {"ifg_control": min(1.0, mean_control_demand + 0.25)})
        tick(37, {"context_feature_count": len(request.context),
                  "task_relevance_entries": len(task_relevance)},
             {"ifg_control": min(1.0, mean_control_demand + 0.15)})
        controlled_posteriors: list[np.ndarray] = []
        for candidate_set in candidate_sets:
            if not candidate_set:
                controlled_posteriors.append(np.zeros(0, dtype=np.float64))
                continue
            scores = [
                float(item["preliminary_score"])
                + self.config.control_gain * (
                    float(item["context_fit"]) + task_relevance.get(item["entry"].concept_id, 1.0) * 0.25
                )
                for item in candidate_set
            ]
            posterior = _softmax(scores)
            for _ in range(self.config.interactive_steps):
                proposal_scores = (
                    np.asarray(scores)
                    + self.config.feedback_gain * posterior
                    - self.config.lateral_inhibition * (posterior.sum() - posterior)
                )
                proposal = _softmax(proposal_scores)
                posterior = 0.60 * posterior + 0.40 * proposal
                posterior /= max(float(posterior.sum()), _EPS)
            controlled_posteriors.append(posterior)
        tick(38, {"controlled_retrieval_cycles": self.config.interactive_steps,
                  "tokens_with_candidates": sum(bool(row) for row in candidate_sets)},
             {"ifg_control": min(1.0, mean_control_demand + 0.30),
              "atl_hub": 0.70})
        ambiguity_rows = []
        for posterior in controlled_posteriors:
            ordered = np.sort(posterior)[::-1]
            entropy = _entropy(posterior)
            margin = float(ordered[0] - ordered[1]) if len(ordered) > 1 else (
                float(ordered[0]) if len(ordered) == 1 else 0.0
            )
            ambiguity_rows.append({"entropy": entropy, "margin": margin,
                                   "candidate_count": int(len(ordered))})
        tick(39, {"ambiguity": ambiguity_rows},
             {"ifg_control": mean_control_demand})

        # 40–42: abstract prototypes; prototypes remain derived and read-only.
        prototype_records: list[dict[str, Any]] = []
        for group, prototype in self._prototypes.items():
            feature_state = hub_state
            activation = _cosine(prototype["features"], feature_state)
            prototype_records.append({
                "prototype_id": group,
                "lexical_category": prototype["lexical_category"],
                "member_concept_ids": list(prototype["member_concept_ids"]),
                "member_count": prototype["member_count"],
                "features": dict(prototype["features"]),
                "activation": activation,
                "source": "frequency_weighted_read_only_concept_aggregate",
            })
        for token, matched_prototypes in memory_prototypes_by_token.items():
            for prototype in matched_prototypes:
                prototype_records.append({
                    "prototype_id": prototype["prototype_id"],
                    "lexical_category": None,
                    "member_concept_ids": [
                        concept.concept_id for concept in self.concepts
                        if concept.prototype_id == prototype["prototype_id"]
                    ],
                    "member_count": prototype["experience_count"],
                    "features": dict(prototype["semantic_features"]),
                    "activation": float(prototype["similarity"]),
                    "source": "sqlite_consolidated_experience_generalization",
                    "matched_token": token,
                    "orthographic_profile": prototype["orthographic_profile"],
                    "phonological_sequence": prototype["phonological_sequence"],
                })
        prototype_records.sort(key=lambda row: row["activation"], reverse=True)
        tick(40, {"prototype_count": len(prototype_records)},
             {"abstract_prototype": min(1.0, len(prototype_records) / max(len(self._prototypes), 1))})
        tick(41, {"prototype_feature_counts": {
            item["prototype_id"]: len(item["features"]) for item in prototype_records
        }}, {"abstract_prototype": 0.75})
        for candidate_set in candidate_sets:
            for item in candidate_set:
                prototype_id, fit = self._prototype_fit(item["entry"])
                item["prototype_id"] = prototype_id
                item["prototype_fit"] = fit
        tick(42, {"prototype_comparison_count": len(candidate_sets_flat)},
             {"abstract_prototype": 0.85, "atl_hub": 0.50})

        # 43: final decision integrates controlled retrieval and bounded prototype support.
        token_results: list[dict[str, Any]] = []
        selected_concepts: list[dict[str, Any]] = []
        semantic_rows: list[tuple[Mapping[str, float], float]] = []
        for token_index, (surface, candidate_set, posterior) in enumerate(
            zip(tokens, candidate_sets, controlled_posteriors)
        ):
            if not candidate_set:
                token_results.append({
                    "token_index": token_index,
                    "surface": surface,
                    "normalized": normalized_tokens[token_index],
                    "candidates": [],
                    "selected_concept": None,
                    "confidence": 0.0,
                    "ambiguity": 1.0,
                    "abstained": True,
                    "reason": "no_lexical_or_episodic_candidate",
                })
                continue
            refined_scores = [
                math.log(float(probability) + _EPS)
                + self.config.prototype_gain * float(item["prototype_fit"])
                for item, probability in zip(candidate_set, posterior)
            ]
            refined = _softmax(refined_scores)
            ranked = sorted(
                zip(candidate_set, refined),
                key=lambda pair: float(pair[1]),
                reverse=True,
            )
            top_item, top_probability = ranked[0]
            second_probability = float(ranked[1][1]) if len(ranked) > 1 else 0.0
            margin = float(top_probability) - second_probability
            entropy = _entropy(refined)
            abstained = (
                float(top_probability) < self.config.confidence_threshold
                or margin < self.config.ambiguity_margin
            )
            candidate_rows = []
            for item, probability in ranked:
                entry: SemanticConcept = item["entry"]
                candidate_rows.append({
                    "concept_id": entry.concept_id,
                    "label": entry.label,
                    "lexical_category": entry.lexical_category,
                    "probability": float(probability),
                    "lexical_similarity": float(item["lexical_similarity"]),
                    "frequency_prior": float(item["frequency_prior"]),
                    "context_fit": float(item["context_fit"]),
                    "episode_fit": float(item["episode_fit"]),
                    "modality_fit": float(item["modality_fit"]),
                    "hub_fit": float(item["hub_fit"]),
                    "prototype_id": item["prototype_id"],
                    "prototype_fit": float(item["prototype_fit"]),
                    "episode_ids": list(item["episode_ids"]),
                    "prototype_evidence_ids": list(item.get("prototype_evidence_ids", [])),
                    "phonological_cues": list(item["phonological_cues"]),
                    "source_reference": entry.source_reference,
                })
            selected = None if abstained else {
                "concept_id": top_item["entry"].concept_id,
                "label": top_item["entry"].label,
                "lexical_category": top_item["entry"].lexical_category,
                "confidence": float(top_probability),
            }
            if selected is not None:
                selected_concepts.append({
                    "token_index": token_index,
                    "surface": surface,
                    **selected,
                })
            for item, probability in zip(candidate_set, refined):
                semantic_rows.append((item["entry"].semantic_features, float(probability)))
            token_results.append({
                "token_index": token_index,
                "surface": surface,
                "normalized": normalized_tokens[token_index],
                "candidates": candidate_rows,
                "selected_concept": selected,
                "confidence": float(top_probability),
                "ambiguity": float(entropy),
                "selection_margin": margin,
                "abstained": bool(abstained),
                "reason": "confidence_or_margin_gate" if abstained else "selected",
                "spoke_activations": spoke_by_population,
            })
        tick(43, {"selected_count": len(selected_concepts),
                  "abstained_count": len(tokens) - len(selected_concepts),
                  "confidence_threshold": self.config.confidence_threshold},
             {"ifg_control": 0.50, "atl_hub": 0.65,
              "abstract_prototype": 0.60})

        semantic_state = _weighted_features(semantic_rows)
        if not semantic_state:
            semantic_state = dict(hub_state)
        finite_functional = all(
            math.isfinite(float(value))
            for value in semantic_state.values()
        )
        trace_complete = (
            len(functional_trace) == 44
            and [row["operation_id"] for row in functional_trace]
            == [spec.operation_id for spec in OPERATION_REGISTRY]
        )
        if simulator is not None:
            biology_snapshot = simulator.snapshot()
            biology = {
                "enabled": True,
                **biology_snapshot,
                "operation_trace": biological_trace,
                "configuration": {
                    "dt_ms": self.config.dt_ms,
                    "steps_per_operation": self.config.steps_per_operation,
                    "membrane_tau_ms": self.config.membrane_tau_ms,
                    "population_size": self.config.population_size,
                    "current_gain_pa": self.config.current_gain_pa,
                    "recurrent_gain_pa": self.config.recurrent_gain_pa,
                    "inhibition_gain_pa": self.config.inhibition_gain_pa,
                },
                "interpretation": (
                    "Neural population observables are an executable LIF reference, "
                    "not evidence of anatomical or physiological equivalence."
                ),
            }
        else:
            biology = {
                "enabled": False,
                "operation_trace": biological_trace,
                "layers": {},
                "simulated_time_ms": 0.0,
                "numerically_valid": True,
                "units": {
                    "membrane_voltage": "mV",
                    "synaptic_current": "pA",
                    "firing_rate": "Hz",
                    "simulated_time": "ms",
                },
                "model_scope": "Biological reference execution disabled by configuration.",
            }
        diagnostics = {
            "valid": bool(trace_complete and finite_functional),
            "trace_complete": bool(trace_complete),
            "operation_count": len(functional_trace),
            "token_count": len(tokens),
            "resolved_token_count": len(selected_concepts),
            "abstained_token_count": len(tokens) - len(selected_concepts),
            "episode_count": len(episode_records),
            "prototype_count": len(prototype_records),
            "finite_functional_state": bool(finite_functional),
            "biological_numerically_valid": bool(biology.get("numerically_valid", True)),
            "missing_modalities_are_neutral": True,
            "recognition_is_read_only": True,
            "acceptance_scope": {
                "software_contracts": "testable by automated tests",
                "behavioral_accuracy": "requires held-out human reading/semantic task data",
                "biological_validity": "requires independent neural data and replication",
            },
        }
        metadata = {
            "algorithm": "ORE Semantic Cognition 44",
            "registry_version": "1.0.0",
            "functional_model": "transparent lexical-feature-control inference",
            "biological_model": "current-based LIF population reference",
            "model_fingerprint": self.model_fingerprint,
            "concept_count": len(self.concepts),
            "persistent_memory_enabled": self.memory_store is not None,
            "persistent_episode_count": self.memory_store.episode_count() if self.memory_store is not None else 0,
            "semantic_source_components": [
                {"component_id": row[0], "name": row[1], "start_step": row[2], "end_step": row[3]}
                for row in SOURCE_COMPONENTS
            ],
            "semantic_source_operation_mapping": source_mapping_as_dicts(),
            "semantic_source_operation_count": len(source_mapping_as_dicts()),
            "persistent_memory_operation_registry": (
                [dict(row) for row in MEMORY_OPERATION_REGISTRY]
                if self.memory_store is not None else []
            ),
            "persistent_memory_operation_trace": (
                self.memory_store.operation_trace[-256:] if self.memory_store is not None else []
            ),
            "config": self.config.to_dict(),
            "source_reference": request.source_reference,
            "language": request.language,
            "task": request.task,
            "contextual_records": [
                {"context_id": item.context_id, "context_type": item.context_type,
                 "value": item.value, "language_profile": item.language_profile, "scope": item.scope}
                for item in request.contextual_records
            ],
            "evidence_records": [
                {"evidence_id": item.evidence_id, "source_id": item.source_id,
                 "evidence_type": item.evidence_type, "support_status": item.support_status,
                 "source_rights_status": item.source_rights_status,
                 "source_redistribution": item.source_redistribution,
                 "source_license_id": item.source_license_id,
                 "usable_for_local_reasoning": item.usable_for_local_reasoning,
                 "verified_fact": False, "interpretation": item.interpretation}
                for item in request.evidence_records
            ],
            "source_provenance": [
                {"source_id": item.source_id, "provider": item.provider,
                 "source_type": item.source_type, "rights_status": item.rights_status,
                 "redistribution": item.redistribution, "attribution": item.attribution,
                 "license_id": item.license_id, "usable_for_local_import": item.usable_for_local_import,
                 "redistributable": item.redistributable}
                for item in request.source_provenance
            ],
            "evidence_is_verified_fact": False,
            "model_scope": (
                "Semantic access and concept selection only; no unrestricted syntax, "
                "discourse, world-model, or clinical inference is claimed."
            ),
        }
        tick(44, {"valid": diagnostics["valid"],
                  "trace_complete": trace_complete,
                  "functional_state_finite": finite_functional,
                  "biological_enabled": biological_enabled},
             {"atl_hub": 0.40, "ifg_control": 0.20})
        # Operation 44 is part of both traces and therefore precedes final result validation.
        trace_complete = (
            len(functional_trace) == 44
            and [row["operation_id"] for row in functional_trace]
            == [spec.operation_id for spec in OPERATION_REGISTRY]
        )
        diagnostics["trace_complete"] = bool(trace_complete)
        diagnostics["operation_count"] = len(functional_trace)
        diagnostics["valid"] = bool(trace_complete and finite_functional)
        if simulator is not None:
            biology.update(simulator.snapshot())
            biology["operation_trace"] = biological_trace

        return SemanticResult(
            sequence_id=request.sequence_id,
            original_text=original_text,
            normalized_text=normalized_text,
            tokens=token_results,
            semantic_state=semantic_state,
            selected_concepts=selected_concepts,
            episodic_retrieval=episode_records,
            abstract_prototypes=prototype_records,
            biological=biology,
            operation_trace=functional_trace,
            diagnostics=diagnostics,
            metadata=metadata,
        )
