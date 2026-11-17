"""Executable 22-stage grounded semantic integration and predictive-language pipeline.

The processor is deliberately explicit and inspectable. It implements transparent feature hashing,
weighted integration, context scoring, graph propagation, retrieval, prediction-error diagnostics,
Hebbian-style updates and learner adaptation. These are computational approximations, not claims
that a human brain runs the same algorithms or equations.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Any, Callable, Mapping
import math
import re

import numpy as np

from .biology import GroundedNeuralReference
from .contracts import GroundedSemanticConfig, GroundedSemanticRequest, GroundedSemanticResult
from .math_ops import (
    categorical_entropy,
    contrastive_alignment_loss,
    cosine_similarity,
    feature_vector,
    hashed_text_vector,
    hebbian_update,
    prediction_residual,
    reliability_from_error,
    stable_softmax,
    temporal_association,
    temporal_difference_error,
    unit_normalize,
    weighted_vector_sum,
)
from .registry import OPERATION_REGISTRY, STAGE_REGISTRY

_VERBS = {
    "be", "is", "are", "was", "were", "have", "has", "do", "does",
    "fly", "flies", "flew", "build", "builds", "built", "sing", "sings", "sang", "eat", "eats",
    "ate", "drink", "drinks", "drank", "open", "opens", "opened",
    "take", "takes", "took", "throw", "throws", "threw", "run", "runs",
    "walk", "walks", "move", "moves", "grasp", "hold", "pour", "use",
}
_DOMAIN_TERMS = {
    "natural_world": {"bird", "animal", "plant", "tree", "flower", "river", "water", "forest"},
    "finance": {"bank", "loan", "money", "account", "credit", "payment", "cash"},
    "geography": {"bank", "river", "stream", "shore", "water", "valley", "coast"},
    "technology": {"computer", "software", "code", "program", "processor", "network", "data"},
    "food": {"apple", "fruit", "eat", "food", "taste", "sweet", "drink", "water"},
    "social": {"person", "boy", "girl", "family", "friend", "doctor", "patient"},
}
_TUTORING_ACTIVITIES = {
    "recall": "retrieve_and_restate",
    "identification": "contrastive_multisensory_identification",
    "action": "demonstrate_object_affordances",
    "contextual": "practice_contextual_usage",
    "generalization": "transfer_to_novel_exemplars",
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to a finite float or return the documented fallback."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _pairwise_scores(vectors: Mapping[str, np.ndarray]) -> dict[str, float]:
    """Calculate named cosine similarities for every pair of available modality vectors."""
    scores: dict[str, float] = {}
    for left, right in combinations(vectors, 2):
        scores[f"{left}|{right}"] = cosine_similarity(vectors[left], vectors[right])
    return scores


class GroundedSemanticIntegrationProcessor:
    """Coordinate all 22 functional stages and optional distributed LIF reference dynamics.

    The processor retains learned co-occurrence, graph, transition and concept-memory state between
    calls. Default encoders are deterministic feature-hash baselines; real modality encoders can be
    injected by name. Every stage has a stable registry ID and produces inspectable output.
    """

    def __init__(
        self,
        config: GroundedSemanticConfig | None = None,
        encoders: Mapping[str, Callable[[Any], Any]] | None = None,
    ) -> None:
        """Initialize numerical controls, optional encoders, associative memory and the biology model."""
        self.config = config or GroundedSemanticConfig()
        self.encoders = dict(encoders or {})
        self.neural_reference = (
            GroundedNeuralReference(self.config) if self.config.simulate_biology else None
        )
        self.word_context_counts: dict[str, Counter[str]] = {}
        self.next_token_counts: dict[str, Counter[str]] = {}
        self.semantic_graph: dict[str, dict[str, float]] = {}
        self.semantic_memory: dict[str, dict[str, Any]] = {}
        self.episodic_memory: list[dict[str, Any]] = []
        self.association_weights: dict[str, np.ndarray] = {}
        self.value_estimates: dict[str, float] = {}
        self.process_count = 0

    def process(self, request: GroundedSemanticRequest | Mapping[str, Any]) -> GroundedSemanticResult:
        """Run the 22 registered stages in order and return functional, biological and logger traces.

        Each stage writes a stable GS-01 through GS-22 result. The processing state preserves
        modality vectors, candidate meanings and memory evidence so later stages can refer to them.
        """
        if isinstance(request, Mapping):
            request = GroundedSemanticRequest(**dict(request))
        if not isinstance(request, GroundedSemanticRequest):
            raise TypeError("request must be GroundedSemanticRequest or a compatible mapping")
        self.process_count += 1
        state: dict[str, Any] = {
            "tokens": list(request.normalized_tokens),
            "raw_inputs": {},
            "timestamps": {},
            "vectors": {},
            "context_features": dict(request.context),
            "stage_outputs": {},
            "stage_trace": [],
            "operation_trace": [],
            "biological_trace": [],
            "integrated_vector": np.zeros(self.config.hidden_dim, dtype=np.float64),
            "attended_vector": np.zeros(self.config.hidden_dim, dtype=np.float64),
            "concept_key": "",
        }
        stage_methods = (
            self._stage_01_acquire,
            self._stage_02_extract,
            self._stage_03_distributional,
            self._stage_04_context,
            self._stage_05_align,
            self._stage_06_affordances,
            self._stage_07_integrate,
            self._stage_08_graph,
            self._stage_09_disambiguate,
            self._stage_10_compose,
            self._stage_11_predict,
            self._stage_12_prediction_error,
            self._stage_13_attention,
            self._stage_14_retrieve,
            self._stage_15_infer,
            self._stage_16_generate,
            self._stage_17_learn,
            self._stage_18_consolidate,
            self._stage_19_teaching,
            self._stage_20_examples,
            self._stage_21_evaluate,
            self._stage_22_adapt,
        )
        for specification, method in zip(STAGE_REGISTRY, stage_methods):
            stage_output = method(request, state)
            state["stage_outputs"][specification.stage_id] = stage_output
            stage_drive = _safe_float(stage_output.get("drive", 0.35), 0.35)
            trace_row = {
                "stage_id": specification.stage_id,
                "stage_number": specification.stage_number,
                "stage": specification.name,
                "objective": specification.objective,
                "functional_status": "completed",
                "explanation": specification.objective,
                "biological_reference": specification.biological_reference,
                "mathematical_operations": [
                    operation.mathematical_form for operation in specification.operations
                ],
                "drive": float(np.clip(stage_drive, 0.0, 1.0)),
            }
            state["stage_trace"].append(trace_row)
            for operation in specification.operations:
                state["operation_trace"].append({
                    "operation_id": operation.operation_id,
                    "stage_id": operation.stage_id,
                    "operation": operation.name,
                    "mathematical_operation": operation.mathematical_form,
                    "functional_responsibility": operation.functional_description,
                    "biological_reference": operation.biological_reference,
                    "functional_status": "completed",
                    "biological_status": "reference_only",
                })
            if self.neural_reference is not None:
                snapshot = self.neural_reference.advance(specification.stage_id, stage_drive)
                snapshot["functional_stage"] = specification.name
                state["biological_trace"].append(snapshot)

        output = dict(state["stage_outputs"]["GS-16"])
        output["recommended_learning_activity"] = state["stage_outputs"]["GS-22"]["activity"]
        diagnostics = {
            "valid": len(state["stage_trace"]) == 22,
            "stage_count": len(state["stage_trace"]),
            "operation_count": len(state["operation_trace"]),
            "modalities_present": sorted(state["vectors"]),
            "integrated_dimension": int(state["integrated_vector"].size),
            "biological_simulation_enabled": self.neural_reference is not None,
            "prediction_error_available": bool(
                state["stage_outputs"]["GS-12"].get("available")
            ),
            "learner_evaluated": bool(state["stage_outputs"]["GS-21"].get("evaluated")),
            "scientific_scope": (
                "Functional computational approximation with an optional abstract LIF reference; "
                "not a literal or empirically validated model of neural semantic processing."
            ),
        }
        metadata = {
            "model": "grounded_semantic_integration",
            "version": "0.1.0",
            "source_reference": request.source_reference,
            "task": request.task,
            "stage_registry": "GS-01..GS-22",
            "registry_operations": len(OPERATION_REGISTRY),
            "process_count": self.process_count,
            "neural_reference": (
                self.neural_reference.summary() if self.neural_reference is not None else None
            ),
        }
        return GroundedSemanticResult(
            sample_id=request.sample_id,
            original_text=request.text or " ".join(request.tokens or ()),
            output=output,
            stage_outputs=state["stage_outputs"],
            stage_trace=state["stage_trace"],
            operation_trace=state["operation_trace"],
            biological_trace=state["biological_trace"],
            diagnostics=diagnostics,
            metadata=metadata,
        )

    def _stage_01_acquire(self, request: GroundedSemanticRequest, state: dict[str, Any]) -> dict[str, Any]:
        """Stage 1: acquire all supplied modalities and estimate temporal coincidence.

        The output preserves raw observations and calculates A_ij=exp(-|t_i-t_j|/tau) for input
        pairs falling within the configured temporal window. No missing modality is fabricated.
        """
        raw = dict(request.modality_inputs)
        if request.text is not None:
            raw["linguistic"] = request.text
        elif request.tokens is not None:
            raw["linguistic"] = " ".join(request.tokens)
        if request.context:
            raw["context"] = request.context
        timestamps: dict[str, float] = {}
        for index, name in enumerate(raw):
            timestamps[name] = _safe_float(
                request.modality_timestamps.get(name, index * 0.001), index * 0.001
            )
        pairs: dict[str, float] = {}
        for left, right in combinations(raw, 2):
            delta = abs(timestamps[left] - timestamps[right])
            if delta <= self.config.temporal_window:
                pairs[f"{left}|{right}"] = temporal_association(delta, self.config.temporal_tau)
        state["raw_inputs"] = raw
        state["timestamps"] = timestamps
        return {
            "modalities": sorted(raw),
            "timestamps": timestamps,
            "temporal_associations": pairs,
            "synchronized_pair_count": len(pairs),
            "drive": min(1.0, len(raw) / 5.0),
        }

    def _stage_02_extract(self, request: GroundedSemanticRequest, state: dict[str, Any]) -> dict[str, Any]:
        """Stage 2: encode each available modality into its own normalized feature vector.

        Callers may inject a trained encoder for a modality. Otherwise named features and numeric
        arrays use deterministic feature hashing/binning. Each channel remains separately available.
        """
        vectors: dict[str, np.ndarray] = {}
        for modality, raw in state["raw_inputs"].items():
            encoder = self.encoders.get(modality)
            encoded = encoder(raw) if encoder is not None else raw
            vector = np.asarray(encoded, dtype=np.float64).reshape(-1) if (
                encoder is not None and not isinstance(encoded, Mapping)
            ) else feature_vector(encoded, self.config.hidden_dim)
            if vector.size != self.config.hidden_dim:
                vector = feature_vector(vector, self.config.hidden_dim)
            vectors[modality] = unit_normalize(vector)
        state["vectors"] = vectors
        return {
            "modalities_encoded": sorted(vectors),
            "dimensions": {name: int(vector.size) for name, vector in vectors.items()},
            "feature_norms": {name: float(np.linalg.norm(vector)) for name, vector in vectors.items()},
            "drive": min(1.0, len(vectors) / 5.0),
        }

    def _stage_03_distributional(self, request: GroundedSemanticRequest, state: dict[str, Any]) -> dict[str, Any]:
        """Stage 3: construct lexical vectors from local context windows and learned co-occurrences.

        For each token, the method combines neighboring terms from this sample with stored context
        counts. This is a transparent distributional baseline, not Word2Vec, BERT or a human neural
        equation. Newly observed pairs are learned later in Stage 17.
        """
        tokens = state["tokens"]
        local_counts: dict[str, Counter[str]] = {}
        for index, token in enumerate(tokens):
            neighbors = tokens[max(0, index - 2):index] + tokens[index + 1:index + 3]
            counts = Counter(neighbors)
            counts.update(self.word_context_counts.get(token, Counter()))
            local_counts[token] = counts
        distributional: dict[str, list[tuple[str, int]]] = {
            token: counts.most_common(8) for token, counts in local_counts.items()
        }
        pooled_counts: Counter[str] = Counter()
        for counts in local_counts.values():
            pooled_counts.update(counts)
        lexical_vector = None
        if tokens:
            lexical_vector = (
                feature_vector(pooled_counts, self.config.hidden_dim)
                if pooled_counts else hashed_text_vector(" ".join(tokens), self.config.hidden_dim)
            )
            if "linguistic" in state["vectors"]:
                lexical_vector = unit_normalize(
                    0.5 * state["vectors"]["linguistic"] + 0.5 * lexical_vector
                )
            state["vectors"]["linguistic"] = lexical_vector
        return {
            "tokens": tokens,
            "distributional_contexts": distributional,
            "lexical_vector": lexical_vector,
            "vocabulary_size": len(set(tokens)),
            "drive": min(1.0, len(tokens) / 12.0),
        }

    def _stage_04_context(self, request: GroundedSemanticRequest, state: dict[str, Any]) -> dict[str, Any]:
        """Stage 4: estimate context hypotheses from explicit features and current lexical cues.

        Category scores are evidence-overlap estimates. The posterior uses a temperature-scaled
        softmax and remains a computational approximation to contextual inference.
        """
        context_features = dict(request.context)
        tokens = set(state["tokens"])
        for domain, terms in _DOMAIN_TERMS.items():
            overlap = len(tokens.intersection(terms))
            if overlap:
                context_features[domain] = max(context_features.get(domain, 0.0), float(overlap))
        scores: dict[str, float] = {}
        for domain, terms in _DOMAIN_TERMS.items():
            scores[domain] = sum(
                max(0.0, _safe_float(context_features.get(term, 0.0)))
                + (1.0 if term in tokens else 0.0)
                for term in terms
            ) + max(0.0, _safe_float(context_features.get(domain, 0.0)))
        if not any(scores.values()):
            scores = {"unspecified": 1.0}
        posterior = stable_softmax(scores, self.config.softmax_temperature)
        state["context_features"] = context_features
        state["context_posterior"] = posterior
        return {
            "context_features": context_features,
            "context_scores": scores,
            "context_posterior": posterior,
            "selected_context": max(posterior, key=posterior.get),
            "entropy_nats": categorical_entropy(posterior),
            "drive": max(posterior.values()) if posterior else 0.0,
        }

    def _stage_05_align(self, request: GroundedSemanticRequest, state: dict[str, Any]) -> dict[str, Any]:
        """Stage 5: compare modality pairs and quantify same-event alignment.

        Pairwise cosine similarity measures representational compatibility. A margin loss compares
        the mean positive compatibility against a neutral negative baseline; the raw channel vectors
        are retained to prevent the alignment score from erasing modality-specific detail.
        """
        vectors = state["vectors"]
        pairwise = _pairwise_scores(vectors)
        positive = float(np.mean(list(pairwise.values()))) if pairwise else 0.0
        loss = contrastive_alignment_loss(positive, [0.0], self.config.alignment_margin)
        state["alignment"] = {"pairwise": pairwise, "mean_similarity": positive, "loss": loss}
        return {
            "pairwise_similarity": pairwise,
            "mean_cross_modal_similarity": positive,
            "contrastive_alignment_loss": loss,
            "aligned": positive >= self.config.confidence_threshold,
            "drive": float(np.clip((positive + 1.0) / 2.0, 0.0, 1.0)),
        }

    def _stage_06_affordances(self, request: GroundedSemanticRequest, state: dict[str, Any]) -> dict[str, Any]:
        """Stage 6: represent supported actions and object affordances from motor and language evidence.

        Explicit motor feature weights take priority. If they are absent, a small transparent verb
        lexicon can identify action cues in text; it does not infer an unobserved physical capability.
        """
        motor = request.modality_inputs.get("motor", {})
        actions: dict[str, float] = {}
        if isinstance(motor, Mapping):
            actions.update({str(key): float(np.clip(_safe_float(value), 0.0, 1.0))
                            for key, value in motor.items()})
        for token in state["tokens"]:
            if token in _VERBS and token not in {"is", "are", "was", "were", "be", "has", "have", "do", "does"}:
                actions[token] = max(actions.get(token, 0.0), 0.5)
        if isinstance(motor, str) and motor.strip():
            actions.setdefault(motor.strip(), 0.75)
        return {
            "affordance_scores": actions,
            "selected_action": max(actions, key=actions.get) if actions else None,
            "evidence_source": "motor_features" if motor else "textual_action_cues",
            "drive": max(actions.values()) if actions else 0.0,
        }

    def _stage_07_integrate(self, request: GroundedSemanticRequest, state: dict[str, Any]) -> dict[str, Any]:
        """Stage 7: integrate channel projections into a shared concept while preserving each spoke.

        Each modality is already mapped to hidden_dim by its encoder. Softmax weights combine
        confidence, availability and task relevance; the original vectors remain in state['vectors'].
        """
        vectors = state["vectors"]
        if not vectors:
            integrated = np.zeros(self.config.hidden_dim, dtype=np.float64)
            weights: dict[str, float] = {}
        else:
            raw_scores: dict[str, float] = {}
            for name, vector in vectors.items():
                confidence = _safe_float(request.modality_confidence.get(name, 0.75), 0.75)
                reliability = reliability_from_error(1.0 - float(np.clip(confidence, 0.0, 1.0)))
                task_bonus = 0.35 if name in request.task.casefold() else 0.0
                context_bonus = 0.15 if name == "context" and state.get("context_features") else 0.0
                raw_scores[name] = reliability + task_bonus + context_bonus + float(np.linalg.norm(vector)) * 0.05
            weights = stable_softmax(raw_scores, self.config.softmax_temperature)
            integrated = unit_normalize(weighted_vector_sum(vectors, weights))
        state["integrated_vector"] = integrated
        state["modality_weights"] = weights
        return {
            "integrated_vector": integrated,
            "modality_weights": weights,
            "preserved_modalities": sorted(vectors),
            "integrated_norm": float(np.linalg.norm(integrated)),
            "drive": min(1.0, float(np.linalg.norm(integrated))),
        }

    def _stage_08_graph(self, request: GroundedSemanticRequest, state: dict[str, Any]) -> dict[str, Any]:
        """Stage 8: construct local semantic edges and spread activation over remembered relationships.

        Adjacent token pairs create temporary undirected links for this interpretation. Persistent
        graph learning occurs in Stage 17 after the current output has been formed.
        """
        tokens = state["tokens"]
        local: dict[str, dict[str, float]] = {}
        for left, right in zip(tokens, tokens[1:]):
            if left == right:
                continue
            local.setdefault(left, {})[right] = max(local.get(left, {}).get(right, 0.0), 1.0)
            local.setdefault(right, {})[left] = max(local.get(right, {}).get(left, 0.0), 1.0)
        activation = {token: 1.0 for token in set(tokens)}
        combined_graph: dict[str, dict[str, float]] = {
            node: dict(edges) for node, edges in self.semantic_graph.items()
        }
        for node, edges in local.items():
            target = combined_graph.setdefault(node, {})
            for neighbor, weight in edges.items():
                target[neighbor] = max(target.get(neighbor, 0.0), weight)
        for _ in range(self.config.graph_steps):
            next_activation: dict[str, float] = {}
            for source, value in activation.items():
                for target, weight in combined_graph.get(source, {}).items():
                    next_activation[target] = next_activation.get(target, 0.0) + value * weight
            activation = {
                name: float(value * 0.75)
                for name, value in next_activation.items()
            } or activation
        state["local_graph"] = local
        state["graph_activation"] = activation
        return {
            "local_edges": sum(len(edges) for edges in local.values()) // 2,
            "remembered_nodes": len(self.semantic_graph),
            "activation": activation,
            "drive": min(1.0, len(activation) / 10.0),
        }

    def _stage_09_disambiguate(self, request: GroundedSemanticRequest, state: dict[str, Any]) -> dict[str, Any]:
        """Stage 9: rank candidate senses against contextual features and retain alternatives.

        Candidate senses are explicit caller-provided weighted feature maps. No sense is invented
        when candidates are absent; the context posterior remains a separate contextual estimate.
        """
        candidates = request.candidate_senses
        evidence = set(state["tokens"]) | set(state["context_features"])
        scores = {
            sense: sum(
                max(0.0, weight) * (
                    max(0.0, _safe_float(state["context_features"].get(feature, 0.0)))
                    + (1.0 if feature in evidence else 0.0)
                )
                for feature, weight in features.items()
            )
            for sense, features in candidates.items()
        }
        probabilities = stable_softmax(scores, self.config.softmax_temperature) if scores else {}
        selected = max(probabilities, key=probabilities.get) if probabilities else None
        state["selected_sense"] = selected
        state["sense_probabilities"] = probabilities
        return {
            "candidate_scores": scores,
            "candidate_probabilities": probabilities,
            "selected_meaning": selected,
            "context_fallback": state.get("context_posterior", {}),
            "uncertain": bool(probabilities and max(probabilities.values()) < self.config.confidence_threshold),
            "drive": max(probabilities.values()) if probabilities else 0.2,
        }

    def _stage_10_compose(self, request: GroundedSemanticRequest, state: dict[str, Any]) -> dict[str, Any]:
        """Stage 10: compose tokens into a sentence vector and a lightweight event-role frame.

        A small verb inventory provides a transparent baseline for agent/action/object roles. More
        complete syntactic parsing can be supplied by an external parser or language encoder.
        """
        tokens = state["tokens"]
        verb_index = next((index for index, token in enumerate(tokens) if token in _VERBS), None)
        if verb_index is None:
            frame = {"agent": None, "action": None, "object": None}
        else:
            frame = {
                "agent": " ".join(tokens[:verb_index]) or None,
                "action": tokens[verb_index],
                "object": " ".join(tokens[verb_index + 1:]) or None,
            }
        sentence_vector = unit_normalize(
            state["integrated_vector"] + hashed_text_vector(" ".join(tokens), self.config.hidden_dim)
        )
        state["sentence_vector"] = sentence_vector
        state["event_frame"] = frame
        return {
            "event_frame": frame,
            "sentence_vector": sentence_vector,
            "token_count": len(tokens),
            "composition_method": "contextual_hash_plus_heuristic_semantic_roles",
            "drive": min(1.0, len(tokens) / 12.0),
        }

    def _stage_11_predict(self, request: GroundedSemanticRequest, state: dict[str, Any]) -> dict[str, Any]:
        """Stage 11: predict next lexical items using learned local transition statistics.

        The baseline estimates P(next token | final token) from previously processed examples. It
        does not claim the brain uses autoregressive token prediction or a Transformer objective.
        """
        tokens = state["tokens"]
        cue = tokens[-1] if tokens else ""
        counts = self.next_token_counts.get(cue, Counter())
        prediction = stable_softmax(
            {token: float(count) for token, count in counts.items()},
            self.config.softmax_temperature,
        ) if counts else {}
        state["next_token_prediction"] = prediction
        state["prediction_cue"] = cue
        return {
            "prediction_cue": cue,
            "next_token_probabilities": prediction,
            "predicted_next_token": max(prediction, key=prediction.get) if prediction else None,
            "prediction_available": bool(prediction),
            "drive": max(prediction.values()) if prediction else 0.1,
        }

    def _stage_12_prediction_error(self, request: GroundedSemanticRequest, state: dict[str, Any]) -> dict[str, Any]:
        """Stage 12: quantify mismatch between the current concept and the closest stored prediction.

        If no earlier episode exists, the result explicitly marks prediction error unavailable.
        When a memory match exists, vector residual, mean absolute error and squared error are stored.
        """
        vector = state["integrated_vector"]
        candidates = []
        for key, record in self.semantic_memory.items():
            similarity = cosine_similarity(vector, record["vector"])
            candidates.append((similarity, key, record))
        if not candidates:
            return {"available": False, "reason": "no_prior_semantic_memory", "drive": 0.0}
        similarity, key, record = max(candidates, key=lambda item: item[0])
        residual = prediction_residual(vector, record["vector"])
        state["prediction_error"] = residual
        return {
            "available": True,
            "nearest_memory": key,
            "prior_similarity": similarity,
            "euclidean_error": residual["euclidean_error"],
            "mean_absolute_error": residual["absolute_error"],
            "squared_error": residual["squared_error"],
            "prediction_error_vector": residual["residual"],
            "drive": float(np.clip(residual["euclidean_error"] / 2.0, 0.0, 1.0)),
        }

    def _stage_13_attention(self, request: GroundedSemanticRequest, state: dict[str, Any]) -> dict[str, Any]:
        """Stage 13: allocate attention according to modality confidence, contextual fit and task value.

        The resulting softmax weights create an attended vector. The unweighted modality vectors
        remain unchanged so selective attention does not erase evidence from other channels.
        """
        vectors = state["vectors"]
        scores: dict[str, float] = {}
        task = request.task.casefold()
        token_set = set(state["tokens"])
        for name, vector in vectors.items():
            confidence = float(np.clip(_safe_float(request.modality_confidence.get(name, 0.75), 0.75), 0.0, 1.0))
            availability = min(1.0, float(np.linalg.norm(vector)))
            relevance = 0.5 if name == "linguistic" and state["tokens"] else 0.0
            relevance += 0.4 if name in task else 0.0
            relevance += 0.2 if name == "context" and state["context_features"] else 0.0
            uncertainty = 1.0 - confidence
            scores[name] = 0.4 * relevance + 0.3 * confidence + 0.2 * availability + 0.1 * uncertainty
        weights = stable_softmax(scores, self.config.attention_temperature) if scores else {}
        attended = weighted_vector_sum(vectors, weights) if vectors else np.zeros(self.config.hidden_dim)
        state["attention_weights"] = weights
        state["attended_vector"] = unit_normalize(attended)
        return {
            "attention_scores": scores,
            "attention_weights": weights,
            "attended_vector": state["attended_vector"],
            "attention_entropy_nats": categorical_entropy(weights),
            "drive": max(weights.values()) if weights else 0.0,
        }

    def _stage_14_retrieve(self, request: GroundedSemanticRequest, state: dict[str, Any]) -> dict[str, Any]:
        """Stage 14: retrieve semantic prototypes and episodic examples similar to the current concept.

        Retrieval scores combine cosine similarity, memory strength and recency. Returned records
        contain summaries rather than mutable references to the processor's internal state.
        """
        query = state["attended_vector"]
        ranked = []
        for key, record in self.semantic_memory.items():
            similarity = cosine_similarity(query, record["vector"])
            score = similarity + 0.15 * _safe_float(record.get("strength", 0.0))
            ranked.append((score, key, similarity, record))
        ranked.sort(reverse=True, key=lambda row: row[0])
        semantic = [
            {"concept_key": key, "score": score, "similarity": similarity,
             "strength": record.get("strength", 0.0), "retrieval_count": record.get("retrieval_count", 0)}
            for score, key, similarity, record in ranked[:self.config.retrieval_limit]
        ]
        episodic_ranked = sorted(
            (
                (cosine_similarity(query, episode["vector"]), episode)
                for episode in self.episodic_memory
            ),
            key=lambda row: row[0],
            reverse=True,
        )
        episodic = [
            {"sample_id": item["sample_id"], "concept_key": item["concept_key"], "similarity": similarity}
            for similarity, item in episodic_ranked[:self.config.retrieval_limit]
        ]
        for item in semantic:
            self.semantic_memory[item["concept_key"]]["retrieval_count"] += 1
        state["retrieved_memories"] = semantic
        return {
            "semantic_memories": semantic,
            "episodic_memories": episodic,
            "retrieval_entropy_nats": categorical_entropy(
                {item["concept_key"]: max(0.0, item["score"]) for item in semantic}
            ),
            "drive": min(1.0, len(semantic) / float(self.config.retrieval_limit)),
        }

    def _stage_15_infer(self, request: GroundedSemanticRequest, state: dict[str, Any]) -> dict[str, Any]:
        """Stage 15: infer nearby semantic relationships and generalize over one- and two-hop paths.

        Edge weights are multiplied along a path and the strongest evidence per inferred concept is
        retained. The bounded graph traversal is a computational inference mechanism, not proof of
        a real-world fact unless the graph edges were supplied from credible evidence.
        """
        inferred: dict[str, float] = {}
        seeds = set(state["tokens"])
        for seed in seeds:
            for neighbor, weight in self.semantic_graph.get(seed, {}).items():
                inferred[neighbor] = max(inferred.get(neighbor, 0.0), float(weight))
                for second, second_weight in self.semantic_graph.get(neighbor, {}).items():
                    if second not in seeds:
                        path_score = float(weight) * float(second_weight)
                        inferred[second] = max(inferred.get(second, 0.0), path_score)
        inferred = {key: value for key, value in inferred.items() if key not in seeds}
        state["inferred_relations"] = inferred
        return {
            "seed_concepts": sorted(seeds),
            "inferred_concepts": inferred,
            "inference_count": len(inferred),
            "drive": min(1.0, len(inferred) / 10.0),
        }

    def _stage_16_generate(self, request: GroundedSemanticRequest, state: dict[str, Any]) -> dict[str, Any]:
        """Stage 16: produce a task-conditioned response from selected, observed semantic evidence.

        The built-in response is deliberately conservative and template-based. It describes which
        cues were supplied rather than pretending a small feature-hash model has a general world model.
        """
        tokens = state["tokens"]
        concept = state.get("selected_sense") or " ".join(tokens[:3]) or "the supplied multimodal input"
        salient = sorted(
            set(tokens) | set(state.get("inferred_relations", {})),
            key=lambda item: (
                state.get("inferred_relations", {}).get(item, 0.0),
                item in tokens,
                item,
            ),
            reverse=True,
        )[:8]
        action_scores = state["stage_outputs"]["GS-06"]["affordance_scores"]
        selected_action = max(action_scores, key=action_scores.get) if action_scores else None
        if "action" in request.task.casefold() and selected_action:
            response_text = f"Supported action cue: {selected_action}."
        elif salient:
            response_text = (
                f"Based on the supplied evidence, the current concept cue is '{concept}'. "
                f"Associated observed or retrieved cues: {', '.join(salient)}."
            )
        else:
            response_text = f"The system received evidence for '{concept}', but has limited supporting features."
        response = {
            "concept": concept,
            "response_text": response_text,
            "salient_features": salient,
            "selected_action": selected_action if "action" in request.task.casefold() else None,
            "confidence": max(state.get("sense_probabilities", {}).values(), default=0.0),
            "evidence_limited": not bool(salient),
        }
        state["response"] = response
        state["concept_key"] = str(concept).casefold()
        return {**response, "drive": min(1.0, len(salient) / 8.0)}

    def _stage_17_learn(self, request: GroundedSemanticRequest, state: dict[str, Any]) -> dict[str, Any]:
        """Stage 17: update word co-occurrence, semantic edges, cross-modal weights and value estimates.

        Adjacent lexical observations update transition counts. Cross-modal co-activation uses a
        clipped Hebbian-style outer-product update, while optional reward uses a temporal-difference
        error. These equations are engineering analogies, not a claim of biological backpropagation.
        """
        tokens = state["tokens"]
        pairs_updated = 0
        for index, token in enumerate(tokens):
            neighbors = tokens[max(0, index - 2):index] + tokens[index + 1:index + 3]
            counter = self.word_context_counts.setdefault(token, Counter())
            counter.update(neighbors)
            for neighbor in neighbors:
                if token != neighbor:
                    left_edges = self.semantic_graph.setdefault(token, {})
                    right_edges = self.semantic_graph.setdefault(neighbor, {})
                    left_edges[neighbor] = left_edges.get(neighbor, 0.0) + 1.0
                    right_edges[token] = right_edges.get(token, 0.0) + 1.0
                    pairs_updated += 1
        for left, right in zip(tokens, tokens[1:]):
            self.next_token_counts.setdefault(left, Counter())[right] += 1
        channels = list(state["vectors"].items())
        hebbian_updates = 0
        for (left_name, left_vector), (right_name, right_vector) in combinations(channels, 2):
            key = f"{left_name}->{right_name}"
            matrix = self.association_weights.get(
                key, np.zeros((self.config.hidden_dim, self.config.hidden_dim), dtype=np.float64)
            )
            self.association_weights[key] = hebbian_update(
                matrix, left_vector, right_vector,
                self.config.learning_rate, self.config.weight_decay,
            )
            hebbian_updates += 1
        concept_key = state.get("concept_key") or "unspecified"
        current_value = self.value_estimates.get(concept_key, 0.0)
        reward = _safe_float(request.feedback.get("reward", 0.0))
        td_error = temporal_difference_error(reward, 0.0, current_value, 0.9)
        self.value_estimates[concept_key] = current_value + self.config.learning_rate * td_error
        return {
            "cooccurrence_tokens_updated": len(tokens),
            "semantic_edges_observed": pairs_updated,
            "transition_pairs_observed": max(0, len(tokens) - 1),
            "hebbian_matrix_updates": hebbian_updates,
            "reward": reward,
            "temporal_difference_error": td_error,
            "updated_value_estimate": self.value_estimates[concept_key],
            "drive": min(1.0, (pairs_updated + hebbian_updates) / 20.0),
        }

    def _stage_18_consolidate(self, request: GroundedSemanticRequest, state: dict[str, Any]) -> dict[str, Any]:
        """Stage 18: consolidate the integrated vector into a stable concept prototype and episodic store.

        Existing strengths decay exponentially; the current representation updates the prototype
        with a bounded learning rate. A separate episode record preserves the current sample cue.
        """
        concept_key = state.get("concept_key") or "multimodal_input"
        vector = unit_normalize(state["integrated_vector"])
        for record in self.semantic_memory.values():
            record["strength"] = float(record.get("strength", 0.0)) * math.exp(-self.config.memory_decay)
        record = self.semantic_memory.get(concept_key)
        if record is None:
            record = {"vector": vector.copy(), "strength": 0.0, "retrieval_count": 0, "observations": 0}
            self.semantic_memory[concept_key] = record
        else:
            old = record["vector"]
            record["vector"] = unit_normalize(
                (1.0 - self.config.learning_rate) * old + self.config.learning_rate * vector
            )
        record["strength"] = min(1.0, float(record["strength"]) + self.config.learning_rate)
        record["observations"] = int(record.get("observations", 0)) + 1
        self.episodic_memory.append({
            "sample_id": request.sample_id,
            "concept_key": concept_key,
            "vector": vector.copy(),
            "tokens": list(state["tokens"]),
            "source_reference": request.source_reference,
        })
        if len(self.episodic_memory) > self.config.max_episodes:
            del self.episodic_memory[:-self.config.max_episodes]
        return {
            "concept_key": concept_key,
            "prototype_strength": record["strength"],
            "observations": record["observations"],
            "semantic_memory_size": len(self.semantic_memory),
            "episodic_memory_size": len(self.episodic_memory),
            "drive": record["strength"],
        }

    def _stage_19_teaching(self, request: GroundedSemanticRequest, state: dict[str, Any]) -> dict[str, Any]:
        """Stage 19: assemble an instructional object from available complementary modalities.

        The lesson keeps each supplied channel separate, computes pairwise coherence where possible,
        and estimates reliability from the request's supplied confidence values.
        """
        vectors = state["vectors"]
        pairwise = _pairwise_scores(vectors)
        coherence = float(np.mean(list(pairwise.values()))) if pairwise else 0.0
        reliability = {
            name: float(np.clip(_safe_float(request.modality_confidence.get(name, 0.75), 0.75), 0.0, 1.0))
            for name in vectors
        }
        teaching = {
            "concept": state.get("concept_key", "multimodal_input"),
            "channels": {name: vector for name, vector in vectors.items()},
            "cross_modal_coherence": coherence,
            "channel_reliability": reliability,
            "text": request.text,
        }
        state["teaching_representation"] = teaching
        return {
            "teaching_representation": teaching,
            "cross_modal_coherence": coherence,
            "channel_reliability": reliability,
            "drive": float(np.clip((coherence + 1.0) / 2.0, 0.0, 1.0)),
        }

    def _stage_20_examples(self, request: GroundedSemanticRequest, state: dict[str, Any]) -> dict[str, Any]:
        """Stage 20: collect real caller examples and create neutral practice prompts for context variation.

        The processor never fabricates external facts. In the absence of supplied examples it creates
        lesson prompts that ask the learner to describe or reuse the observed concept cue.
        """
        concept = state.get("concept_key") or "the concept"
        examples = list(request.usage_examples)
        if not examples and request.text:
            examples.append(request.text)
        prompts = [
            f"Describe '{concept}' using the features observed in this example.",
            f"Use '{concept}' in a different situation and explain what changes with context.",
            f"Identify which supplied feature remains stable across the examples.",
        ]
        return {
            "source_examples": examples,
            "practice_prompts": prompts,
            "shared_context_cue": concept,
            "evidence_note": "Prompts are instructional templates; source examples remain caller-provided.",
            "drive": min(1.0, (len(examples) + len(prompts)) / 6.0),
        }

    def _stage_21_evaluate(self, request: GroundedSemanticRequest, state: dict[str, Any]) -> dict[str, Any]:
        """Stage 21: evaluate recall, identification, action, context and generalization evidence.

        Only supplied learner scores are evaluated. Missing dimensions are excluded from the weighted
        mean, rather than being silently treated as failures or successful learning.
        """
        response = dict(request.learner_response or {})
        dimensions = ("recall", "identification", "action", "contextual", "generalization")
        weights = {
            "recall": 0.20,
            "identification": 0.20,
            "action": 0.15,
            "contextual": 0.20,
            "generalization": 0.25,
        }
        scores = {
            name: float(np.clip(_safe_float(response[name]), 0.0, 1.0))
            for name in dimensions if name in response
        }
        available_weight = sum(weights[name] for name in scores)
        learning_score = (
            sum(weights[name] * value for name, value in scores.items()) / available_weight
            if available_weight > 0 else None
        )
        uncertainty = 1.0 - float(np.mean(list(scores.values()))) if scores else None
        state["learner_scores"] = scores
        return {
            "evaluated": bool(scores),
            "dimension_scores": scores,
            "learning_score": learning_score,
            "estimated_uncertainty": uncertainty,
            "weakest_dimension": min(scores, key=scores.get) if scores else None,
            "weights_used": {name: weights[name] for name in scores},
            "drive": float(np.mean(list(scores.values()))) if scores else 0.0,
        }

    def _stage_22_adapt(self, request: GroundedSemanticRequest, state: dict[str, Any]) -> dict[str, Any]:
        """Stage 22: choose the next teaching activity from measured learner gaps.

        The weakest observed dimension selects a targeted activity. When no learner response exists,
        the system requests a baseline response instead of inferring an ability profile.
        """
        scores = state.get("learner_scores", {})
        if not scores:
            activity = "collect_baseline_response"
            dimension = None
            expected_gain = None
        else:
            dimension = min(scores, key=scores.get)
            activity = _TUTORING_ACTIVITIES[dimension]
            expected_gain = max(0.0, 1.0 - float(scores[dimension]))
        return {
            "activity": activity,
            "target_dimension": dimension,
            "expected_gain_proxy": expected_gain,
            "reason": (
                f"Target the lowest measured dimension: {dimension}."
                if dimension else "No learner evidence supplied; collect a baseline before adapting."
            ),
            "drive": expected_gain if expected_gain is not None else 0.2,
        }


def run_grounded_semantic_pipeline(
    request: GroundedSemanticRequest | Mapping[str, Any],
    config: GroundedSemanticConfig | None = None,
) -> GroundedSemanticResult:
    """Run one request through a fresh 22-stage processor and return its complete structured result."""
    return GroundedSemanticIntegrationProcessor(config=config).process(request)
