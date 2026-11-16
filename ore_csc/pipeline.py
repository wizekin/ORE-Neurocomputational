"""Executable 18-stage Controlled Semantic Cognition functional pipeline.

The functional track manipulates explicit semantic evidence and preserves
alternative meanings. The optional LIF track is a separate computational
reference, not a measured or anatomically fitted model of the human brain.
"""
from __future__ import annotations

from collections import defaultdict
from difflib import SequenceMatcher
import math
import re
import time
import unicodedata
from typing import Any, Mapping, Sequence

import numpy as np

from ore_semantic.contracts import SemanticConcept

from .biology import CSCLIFSimulator, stage_drives
from .contracts import CSCConfig, CSCRequest, CSCResult
from .registry import ALL_OPERATIONS, STAGE_REGISTRY


_EPS = 1e-9


def _normalize_text(value: str) -> str:
    """Normalize a text cue for matching without changing the stored original form."""
    normalized = unicodedata.normalize("NFKC", value).casefold().strip()
    return re.sub(r"[^\w\s'-]+", " ", normalized, flags=re.UNICODE).strip()


def _tokenize(value: str) -> list[str]:
    """Split text into stable word-like units while preserving apostrophe-bearing words."""
    return re.findall(r"[^\W_]+(?:['’][^\W_]+)?", _normalize_text(value), flags=re.UNICODE)


def _feature_map(values: Mapping[str, float]) -> dict[str, float]:
    """Normalize feature names and merge duplicate case variants using maximum activation."""
    output: dict[str, float] = {}
    for key, raw in values.items():
        normalized = str(key).strip().casefold().replace("-", "_").replace(" ", "_")
        if normalized:
            output[normalized] = max(output.get(normalized, 0.0), float(np.clip(raw, 0.0, 1.0)))
    return output


def _merge_maps(*maps: Mapping[str, float]) -> dict[str, float]:
    """Merge sparse feature maps by retaining the strongest supplied activation per feature."""
    result: dict[str, float] = {}
    for values in maps:
        for key, value in _feature_map(values).items():
            result[key] = max(result.get(key, 0.0), value)
    return result


def _cosine(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    """Calculate cosine similarity on sparse named vectors with a defined zero-vector result."""
    keys = sorted(set(left) | set(right))
    if not keys:
        return 0.0
    a = np.asarray([float(left.get(key, 0.0)) for key in keys], dtype=np.float64)
    b = np.asarray([float(right.get(key, 0.0)) for key in keys], dtype=np.float64)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= _EPS:
        return 0.0
    return float(np.clip(np.dot(a, b) / denominator, -1.0, 1.0))


def _softmax(values: Sequence[float], temperature: float = 1.0) -> list[float]:
    """Convert finite candidate scores into a stable probability distribution."""
    vector = np.asarray(values, dtype=np.float64)
    if vector.size == 0:
        return []
    scaled = vector / max(float(temperature), _EPS)
    scaled -= float(np.max(scaled))
    exponentials = np.exp(np.clip(scaled, -60.0, 60.0))
    total = float(np.sum(exponentials))
    if total <= _EPS or not math.isfinite(total):
        return [1.0 / vector.size] * int(vector.size)
    return [float(item) for item in exponentials / total]


def _sigmoid(value: float) -> float:
    """Apply a numerically stable logistic transform to a scalar."""
    clipped = float(np.clip(value, -40.0, 40.0))
    return float(1.0 / (1.0 + math.exp(-clipped)))


def _normalized_entropy(probabilities: Sequence[float]) -> float:
    """Return candidate entropy normalized to [0, 1] for two or more alternatives."""
    values = np.asarray(probabilities, dtype=np.float64)
    if values.size <= 1:
        return 0.0
    entropy = -float(np.sum(values * np.log(np.maximum(values, _EPS))))
    return float(np.clip(entropy / math.log(values.size), 0.0, 1.0))


def _candidate_features(concept: SemanticConcept) -> dict[str, float]:
    """Combine a concept's transmodal and modality-linked features without inventing features."""
    pieces: list[Mapping[str, float]] = [concept.semantic_features]
    pieces.extend(concept.modality_features.values())
    return _merge_maps(*pieces)


def _lexical_match(tokens: Sequence[str], concept: SemanticConcept) -> float:
    """Score exact and partial label/alias matches while retaining ambiguous alternatives."""
    text = " ".join(tokens)
    candidates = (concept.label, *concept.aliases)
    best = 0.0
    for label in candidates:
        normalized = _normalize_text(label)
        label_tokens = _tokenize(label)
        if not normalized or not label_tokens:
            continue
        if normalized in text:
            best = max(best, 1.0)
            continue
        overlap = len(set(tokens) & set(label_tokens)) / max(len(set(label_tokens)), 1)
        sequence = SequenceMatcher(None, text, normalized).ratio()
        best = max(best, 0.72 * overlap, 0.45 * sequence)
    return float(np.clip(best, 0.0, 1.0))


def _clamp_map(values: Mapping[str, float]) -> dict[str, float]:
    """Clip every sparse feature activation to the biologically motivated unit interval."""
    return {key: float(np.clip(value, 0.0, 1.0)) for key, value in values.items()}


class CSCProcessor:
    """Run the 18-stage CSC control algorithm over a provenance-tracked concept inventory."""

    def __init__(
        self,
        concepts: Sequence[SemanticConcept],
        config: CSCConfig | None = None,
        associations: Mapping[str, Mapping[str, float]] | None = None,
    ) -> None:
        """Pin a unique concept inventory and initialize explicit, instance-local learning state."""
        self.config = config or CSCConfig()
        self.concepts = tuple(concepts)
        if not self.concepts:
            raise ValueError("CSCProcessor requires at least one SemanticConcept")
        self._concept_by_id = {item.concept_id: item for item in self.concepts}
        if len(self._concept_by_id) != len(self.concepts):
            raise ValueError("concept_id values must be unique")
        self._feature_by_concept = {
            item.concept_id: _candidate_features(item) for item in self.concepts
        }
        self.associations = self._validate_associations(associations or {})
        self._retrieval_counts: dict[str, int] = defaultdict(int)
        self._learned_priors: dict[str, float] = defaultdict(float)
        self._learning_generation = 0
        self._control_bias = 0.0

    def _validate_associations(
        self, associations: Mapping[str, Mapping[str, float]]
    ) -> dict[str, dict[str, float]]:
        """Validate caller-supplied non-negative feature association strengths."""
        known = {feature for mapping in self._feature_by_concept.values() for feature in mapping}
        result: dict[str, dict[str, float]] = {}
        for source, targets in associations.items():
            source_key = str(source).casefold()
            if source_key not in known:
                raise ValueError("association source is not in the concept inventory: " + source_key)
            result[source_key] = {}
            for target, raw in targets.items():
                target_key = str(target).casefold()
                value = float(raw)
                if target_key not in known or not math.isfinite(value) or value < 0.0:
                    raise ValueError("associations require known features and finite non-negative weights")
                result[source_key][target_key] = value
        return result

    def _build_associations(self) -> dict[str, dict[str, float]]:
        """Combine supplied feature links with normalized co-occurrence across concepts."""
        counts: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for features in self._feature_by_concept.values():
            names = sorted(features)
            for left in names:
                for right in names:
                    if left != right:
                        counts[left][right] += min(features[left], features[right])
        output: dict[str, dict[str, float]] = {}
        for source, targets in counts.items():
            total = sum(targets.values()) or 1.0
            output[source] = {
                target: value / total for target, value in targets.items()
            }
        for source, targets in self.associations.items():
            output.setdefault(source, {})
            for target, value in targets.items():
                output[source][target] = max(output[source].get(target, 0.0), value)
        return output

    def process(
        self,
        request: CSCRequest,
        learning_authorized: bool = False,
        feedback_reward: float | None = None,
    ) -> CSCResult:
        """Execute all 18 stages and optionally commit externally supervised learning updates."""
        if not isinstance(request, CSCRequest):
            raise TypeError("request must be a CSCRequest")
        if learning_authorized:
            if feedback_reward is None or request.feedback_concept_id is None:
                raise ValueError(
                    "authorized learning requires feedback_reward and request.feedback_concept_id"
                )
            if not math.isfinite(float(feedback_reward)) or not -1.0 <= feedback_reward <= 1.0:
                raise ValueError("feedback_reward must be finite and lie in [-1, 1]")
            if request.feedback_concept_id not in self._concept_by_id:
                raise ValueError("feedback_concept_id must identify a supplied concept")
        elif feedback_reward is not None:
            raise ValueError("feedback_reward is accepted only when learning_authorized=True")

        tokens = _tokenize(request.input_text)
        simulator = CSCLIFSimulator(self.config) if self.config.simulate_biology else None
        state: dict[str, Any] = {
            "request": request,
            "tokens": tokens,
            "associations": self._build_associations(),
            "candidate_features": self._feature_by_concept,
            "candidate_scores": [],
            "candidate_probabilities": [],
            "feature_activation": {},
            "semantic_state": {},
            "controlled_state": {},
            "context_state": _feature_map(request.context_features),
            "task_state": _feature_map(request.task_features),
            "task_relevance": {},
            "feature_gain": {},
            "feature_inhibition": {},
            "control_demand": 0.0,
            "control_activity": 0.0,
            "hub_state": {},
            "selected_concept": None,
            "selection": {},
            "integrated_state": {},
            "response": {},
            "control_iterations": [],
            "learning": {},
            "stage_outputs": {},
            "stage_trace": [],
            "operation_trace": [],
        }
        if not state["tokens"]:
            raise ValueError("input text must contain at least one word-like token")
        if not request.task_features:
            state["task_state"] = self._task_features_from_text(request.task, state)

        stages = (
            self._stage_01,
            self._stage_02,
            self._stage_03,
            self._stage_04,
            self._stage_05,
            self._stage_06,
            self._stage_07,
            self._stage_08,
            self._stage_09,
            self._stage_10,
            self._stage_11,
            self._stage_12,
            self._stage_13,
            self._stage_14,
            self._stage_15,
            self._stage_16,
            self._stage_17,
            self._stage_18,
        )
        for number, stage_function in enumerate(stages, 1):
            started = time.perf_counter()
            output = stage_function(state, learning_authorized, feedback_reward)
            duration_ms = (time.perf_counter() - started) * 1000.0
            state["stage_outputs"]["CSC-{0:02d}".format(number)] = output
            biological_event = None
            if simulator is not None:
                drives = stage_drives(
                    number,
                    demand=state["control_demand"],
                    lexical=self._mean(state["candidate_probabilities"]),
                    semantic=self._mean(state["feature_activation"].values()),
                    hub=self._mean(state["hub_state"].values()),
                    context=self._mean(state["context_state"].values()),
                    control=state["control_activity"],
                    selected=state["selection"].get("confidence", 0.0),
                )
                biological_event = simulator.step("CSC-{0:02d}".format(number), drives)
            self._record_stage(state, number, output, duration_ms, biological_event)

        biological = simulator.snapshot() if simulator is not None else {
            "enabled": False,
            "stage_count": 0,
            "operation_trace": [],
            "numerically_valid": True,
            "model_scope": "Biological reference disabled by CSCConfig.simulate_biology.",
        }
        selected = state["selected_concept"]
        diagnostics = {
            "valid": bool(
                len(state["stage_trace"]) == 18
                and len(state["operation_trace"]) == len(ALL_OPERATIONS)
                and all(math.isfinite(float(value)) for value in state["controlled_state"].values())
            ),
            "stage_count": len(state["stage_trace"]),
            "operation_count": len(state["operation_trace"]),
            "biological_stage_count": biological["stage_count"],
            "selected": selected is not None,
            "selection_accepted": bool(state["selection"].get("accepted", False)),
            "control_iterations": len(state["control_iterations"]),
            "warnings": [] if selected is not None else [
                "No candidate passed the configured confidence and selection-margin gates."
            ],
        }
        metadata = {
            "architecture": "Controlled Semantic Cognition",
            "registry": "ORE-CSC-18",
            "registry_version": "1.0.0",
            "source_document": "Controlled Semantic Cognition (CSC).txt",
            "source_reference": request.source_reference,
            "model_scope": (
                "Functional evidence-processing prototype plus optional generic LIF reference; "
                "not a fitted or validated reproduction of human brain activity."
            ),
            "learning_generation": self._learning_generation,
            "learning_persistence": "processor_instance_memory_only",
        }
        result = CSCResult(
            sequence_id=request.sequence_id,
            original_text=request.input_text,
            tokens=tokens,
            selected_concept=selected,
            candidate_scores=state["candidate_scores"],
            semantic_state=_clamp_map(state["semantic_state"]),
            controlled_state=_clamp_map(state["controlled_state"]),
            integrated_state=_clamp_map(state["integrated_state"]),
            response=state["response"],
            stage_outputs=state["stage_outputs"],
            stage_trace=state["stage_trace"],
            operation_trace=state["operation_trace"],
            biological=biological,
            learning=state["learning"],
            diagnostics=diagnostics,
            metadata=metadata,
        )
        return result

    def _record_stage(
        self,
        state: dict[str, Any],
        number: int,
        output: Mapping[str, Any],
        duration_ms: float,
        biological_event: Mapping[str, Any] | None,
    ) -> None:
        """Record the stage outcome and every registered operation with its math and biology note."""
        stage = STAGE_REGISTRY[number - 1]
        biological_status = "simulated" if biological_event is not None else "disabled"
        stage_row = {
            "stage_id": stage.stage_id,
            "stage_name": stage.name,
            "functional_status": "completed",
            "biological_status": biological_status,
            "duration_ms": round(duration_ms, 4),
            "functional_output": dict(output),
            "biological_output": dict(biological_event or {}),
        }
        state["stage_trace"].append(stage_row)
        for operation in stage.operations:
            state["operation_trace"].append({
                "operation_id": operation.operation_id,
                "stage_id": stage.stage_id,
                "operation": operation.name,
                "functional_responsibility": operation.functional_responsibility,
                "mathematical_operation": operation.mathematical_operation,
                "biological_reference": stage.biological_reference,
                "functional_status": "executed",
                "biological_status": biological_status,
            })

    def _task_features_from_text(self, task: str, state: Mapping[str, Any]) -> dict[str, float]:
        """Extract exact task words that match feature names already present in the inventory."""
        tokens = set(_tokenize(task))
        known = {
            feature
            for features in self._feature_by_concept.values()
            for feature in features
        }
        return {feature: 0.7 for feature in known if feature in tokens}

    def _mean(self, values: Sequence[float] | Any) -> float:
        """Return a bounded mean for optional stage drives, including empty collections."""
        items = list(values)
        if not items:
            return 0.0
        return float(np.clip(np.mean(np.asarray(items, dtype=np.float64)), 0.0, 1.0))

    def _stage_01(
        self, state: dict[str, Any], learning_authorized: bool,
        feedback_reward: float | None,
    ) -> dict[str, Any]:
        """Encode text, tokens, contextual features, and supplied modality evidence."""
        request: CSCRequest = state["request"]
        modality_state = {
            name: _feature_map(values) for name, values in request.modality_inputs.items()
        }
        fourier_bins = 0
        signal_mean = 0.0
        signal_std = 0.0
        if request.signal_samples:
            samples = np.asarray(request.signal_samples, dtype=np.float64)
            signal_mean = float(np.mean(samples))
            signal_std = float(np.std(samples))
            standardized = (samples - signal_mean) / (signal_std + _EPS)
            spectrum = np.abs(np.fft.rfft(standardized))
            scale = float(np.max(spectrum)) if spectrum.size else 0.0
            normalized_spectrum = spectrum / (scale + _EPS)
            signal_features = {
                "spectral_bin_{0}".format(index): float(value)
                for index, value in enumerate(normalized_spectrum)
            }
            modality_state.setdefault(request.signal_modality, {}).update(signal_features)
            fourier_bins = len(signal_features)
        temporal_tokens = {
            "token:" + token: float(math.exp(-0.08 * (len(state["tokens"]) - index - 1)))
            for index, token in enumerate(state["tokens"])
        }
        state["input_state"] = _merge_maps(*modality_state.values(), temporal_tokens)
        return {
            "tokens": list(state["tokens"]),
            "token_count": len(state["tokens"]),
            "modalities_received": sorted(modality_state),
            "input_feature_count": len(state["input_state"]),
            "fourier_bins": fourier_bins,
            "signal_sample_count": len(request.signal_samples),
            "signal_mean": signal_mean,
            "signal_standard_deviation": signal_std,
            "z_score_normalization": bool(request.signal_samples),
            "temporal_token_weights": temporal_tokens,
            "input_normalization": "z-score and normalized real FFT for optional samples; bounded feature maps otherwise",
        }

    def _stage_02(
        self, state: dict[str, Any], learning_authorized: bool,
        feedback_reward: float | None,
    ) -> dict[str, Any]:
        """Activate all lexically plausible concepts and add weak frequency priors."""
        tokens = state["tokens"]
        raw = []
        for concept in self.concepts:
            match = _lexical_match(tokens, concept)
            prior = math.log1p(concept.frequency)
            learned = self._learned_priors.get(concept.concept_id, 0.0)
            raw.append({
                "concept_id": concept.concept_id,
                "label": concept.label,
                "lexical_match": match,
                "frequency_prior": prior,
                "learned_prior": learned,
                "initial_score": 1.5 * match + 0.06 * prior + learned,
            })
        probabilities = _softmax([item["initial_score"] for item in raw], 0.45)
        for item, probability in zip(raw, probabilities):
            item["initial_probability"] = probability
            item["initial_activation"] = probability
        state["candidate_scores"] = raw
        state["candidate_probabilities"] = probabilities
        return {
            "candidate_count": len(raw),
            "candidates": raw,
            "posterior_sum": float(sum(probabilities)),
            "alternatives_preserved": True,
        }

    def _stage_03(
        self, state: dict[str, Any], learning_authorized: bool,
        feedback_reward: float | None,
    ) -> dict[str, Any]:
        """Spread activity through co-occurring concept features with bounded recurrence and decay."""
        activation: dict[str, float] = defaultdict(float)
        for candidate, probability in zip(self.concepts, state["candidate_probabilities"]):
            for feature, strength in self._feature_by_concept[candidate.concept_id].items():
                activation[feature] = max(activation[feature], float(probability) * strength)
        associations = state["associations"]
        initial = dict(activation)
        for _ in range(2):
            incoming: dict[str, float] = defaultdict(float)
            for source, value in activation.items():
                for target, weight in associations.get(source, {}).items():
                    incoming[target] += value * weight
            next_state = {}
            for feature in set(activation) | set(incoming):
                retained = (1.0 - self.config.activity_decay) * activation.get(feature, 0.0)
                propagated = self.config.recurrence_gain * incoming.get(feature, 0.0)
                next_state[feature] = float(np.clip(retained + propagated, 0.0, 1.0))
            activation = next_state
        state["feature_activation"] = activation
        state["semantic_state"] = dict(activation)
        return {
            "feature_count": len(activation),
            "active_features": dict(activation),
            "initial_activation_mean": self._mean(initial.values()),
            "recurrent_activation_mean": self._mean(activation.values()),
            "decay_fraction": self.config.activity_decay,
        }

    def _stage_04(
        self, state: dict[str, Any], learning_authorized: bool,
        feedback_reward: float | None,
    ) -> dict[str, Any]:
        """Reliability-weight modality inputs and settle an ATL-inspired shared feature hub."""
        request: CSCRequest = state["request"]
        spoke_maps: dict[str, dict[str, float]] = {}
        for modality, values in request.modality_inputs.items():
            spoke_maps[modality] = _feature_map(values)
        for concept in self.concepts:
            for modality, values in concept.modality_features.items():
                spoke_maps.setdefault(modality, {})
                for key, value in _feature_map(values).items():
                    spoke_maps[modality][key] = max(spoke_maps[modality].get(key, 0.0), value)
        names = sorted(spoke_maps)
        reliability = [
            math.log1p(sum(spoke_maps[name].values())) for name in names
        ]
        weights = _softmax(reliability, 0.75) if names else []
        hub: dict[str, float] = defaultdict(float)
        for name, weight in zip(names, weights):
            for feature, value in spoke_maps[name].items():
                hub[feature] += weight * value
        for feature, value in state["semantic_state"].items():
            hub[feature] = max(hub.get(feature, 0.0), 0.5 * value)
        recurrent = dict(hub)
        for _ in range(3):
            recurrent = {
                feature: float(np.clip(
                    0.65 * recurrent.get(feature, 0.0)
                    + 0.35 * state["semantic_state"].get(feature, 0.0),
                    0.0, 1.0
                ))
                for feature in set(recurrent) | set(state["semantic_state"])
            }
        state["hub_state"] = recurrent
        return {
            "spoke_names": names,
            "spoke_reliability_weights": {
                name: weight for name, weight in zip(names, weights)
            },
            "hub_state": recurrent,
            "hub_similarity_by_candidate": {
                concept.concept_id: _cosine(
                    recurrent, self._feature_by_concept[concept.concept_id]
                )
                for concept in self.concepts
            },
            "pattern_completion": "limited to features present in candidates or caller inputs",
        }

    def _stage_05(
        self, state: dict[str, Any], learning_authorized: bool,
        feedback_reward: float | None,
    ) -> dict[str, Any]:
        """Accumulate explicit context and token cues, then compute contextual candidate fit."""
        current_context = dict(state["context_state"])
        token_set = set(state["tokens"])
        for feature in {
            item for values in self._feature_by_concept.values() for item in values
        }:
            if feature in token_set:
                current_context[feature] = max(current_context.get(feature, 0.0), 0.7)
        previous_context = _feature_map(request.previous_context_features)
        if previous_context:
            alpha = self.config.context_ema_alpha
            context = {
                key: float(np.clip(
                    alpha * previous_context.get(key, 0.0)
                    + (1.0 - alpha) * current_context.get(key, 0.0), 0.0, 1.0
                ))
                for key in set(previous_context) | set(current_context)
            }
        else:
            context = _merge_maps(current_context)
        state["context_state"] = context
        context_scores = {
            concept.concept_id: _cosine(
                self._feature_by_concept[concept.concept_id], context
            )
            for concept in self.concepts
        }
        state["context_scores"] = context_scores
        return {
            "context_features": context,
            "context_similarity": context_scores,
            "context_source": "caller features plus exact feature-token matches only",
            "context_accumulation_alpha": self.config.context_ema_alpha,
            "previous_context_supplied": bool(previous_context),
        }

    def _stage_06(
        self, state: dict[str, Any], learning_authorized: bool,
        feedback_reward: float | None,
    ) -> dict[str, Any]:
        """Create a task target and rank features by task relevance."""
        current_task = dict(state["task_state"])
        previous_task = _feature_map(state["request"].previous_task_features)
        if previous_task:
            alpha = self.config.task_ema_alpha
            target = {
                key: float(np.clip(
                    alpha * previous_task.get(key, 0.0)
                    + (1.0 - alpha) * current_task.get(key, 0.0), 0.0, 1.0
                ))
                for key in set(previous_task) | set(current_task)
            }
        else:
            target = _merge_maps(current_task)
        context = state["context_state"]
        relevance: dict[str, float] = {}
        features = set(state["feature_activation"]) | set(target)
        for feature in features:
            relevance[feature] = (
                float(target.get(feature, 0.0))
                if target else 0.0
            )
        priorities = {key: value * relevance.get(key, 0.0) for key, value in relevance.items()}
        state["task_state"] = target
        state["task_relevance"] = relevance
        return {
            "task": state["request"].task,
            "task_features": target,
            "feature_task_match": relevance,
            "priority_weights": priorities,
            "context_retained_separately": context,
            "previous_task_supplied": bool(previous_task),
            "task_maintenance_alpha": self.config.task_ema_alpha,
        }

    def _stage_07(
        self, state: dict[str, Any], learning_authorized: bool,
        feedback_reward: float | None,
    ) -> dict[str, Any]:
        """Estimate semantic control demand from entropy, conflict, weak target activity, and novelty."""
        probabilities = state["candidate_probabilities"]
        entropy = _normalized_entropy(probabilities)
        ordered = sorted(probabilities, reverse=True)
        margin = ordered[0] - (ordered[1] if len(ordered) > 1 else 0.0)
        conflict = float(np.clip(1.0 - margin, 0.0, 1.0))
        target = state["task_state"] or state["context_state"]
        target_activation = self._mean([
            state["feature_activation"].get(feature, 0.0) * weight
            for feature, weight in target.items()
        ])
        weakness = 1.0 - target_activation if target else 0.5
        novelty = 1.0 - _cosine(target, state["semantic_state"]) if target else 0.5
        demand = _sigmoid(
            1.2 * entropy + 0.85 * conflict + 0.9 * weakness + 0.55 * novelty
            - 1.35 + self._control_bias
        )
        state["control_demand"] = demand
        return {
            "normalized_entropy": entropy,
            "candidate_conflict": conflict,
            "target_activation": target_activation,
            "retrieval_weakness": weakness,
            "task_novelty": novelty,
            "control_demand": demand,
            "equation_scope": "transparent illustrative combination; weights are not measured neural constants",
        }

    def _stage_08(
        self, state: dict[str, Any], learning_authorized: bool,
        feedback_reward: float | None,
    ) -> dict[str, Any]:
        """Recruit a distributed semantic control reference network as demand increases."""
        demand = state["control_demand"]
        context_level = self._mean(state["context_state"].values())
        task_level = self._mean(state["task_state"].values())
        control_signal = float(np.clip(0.45 * task_level + 0.35 * context_level + 0.65 * demand, 0, 1))
        regions = {
            "left_ifg": control_signal,
            "posterior_mtg": 0.92 * control_signal,
            "posterior_itg": 0.68 * control_signal,
            "dorsomedial_pfc": 0.82 * demand,
            "dorsolateral_pfc": 0.78 * task_level * control_signal,
            "parietal_attention": 0.5 * control_signal if demand > 0.45 else 0.0,
        }
        # A damped recurrent network update is represented with an explicit bounded scalar state.
        previous = state.get("control_activity", 0.0)
        control_activity = float(np.clip(
            0.7 * previous + 0.3 * self._mean(regions.values()), 0.0, 1.0
        ))
        state["control_activity"] = control_activity
        state["control_regions"] = regions
        return {
            "control_signal": control_signal,
            "regional_activity": regions,
            "recurrent_network_activity": control_activity,
            "parietal_route": "task-dependent" if regions["parietal_attention"] > 0 else "not_recruited",
            "demand_dependent_coupling": 1.0 + demand * self.config.recurrence_gain,
        }

    def _stage_09(
        self, state: dict[str, Any], learning_authorized: bool,
        feedback_reward: float | None,
    ) -> dict[str, Any]:
        """Translate task and context targets into feature relevance, gains, and weak-feature priority."""
        target = _merge_maps(state["task_state"], state["context_state"])
        activation = state["feature_activation"]
        if target:
            relevance = {
                feature: _cosine({feature: 1.0}, target) for feature in activation
            }
        else:
            relevance = {feature: 0.5 for feature in activation}
        gains = {
            feature: float(np.clip(
                0.20 + self.config.gain_strength * _sigmoid(
                    4.0 * (relevance.get(feature, 0.0) - 0.35)
                    + 2.0 * state["control_activity"]
                ),
                0.0, 1.0
            ))
            for feature in activation
        }
        exploration = {
            feature: (relevance.get(feature, 0.0) + _EPS) / (value + _EPS)
            for feature, value in activation.items()
        }
        state["task_relevance"] = relevance
        state["feature_gain"] = gains
        state["exploration_priority"] = exploration
        return {
            "target_features": target,
            "feature_relevance": relevance,
            "feature_gain": gains,
            "weak_relevant_feature_priority": exploration,
        }

    def _stage_10(
        self, state: dict[str, Any], learning_authorized: bool,
        feedback_reward: float | None,
    ) -> dict[str, Any]:
        """Amplify relevant features through multiplicative, additive, and recurrent operations."""
        activation = state["feature_activation"]
        gains = state["feature_gain"]
        associations = state["associations"]
        amplified: dict[str, float] = {}
        for feature, value in activation.items():
            multiplicative = value * gains.get(feature, 0.45)
            additive = 0.08 * gains.get(feature, 0.45)
            recurrent = sum(
                activation.get(source, 0.0) * associations.get(source, {}).get(feature, 0.0)
                for source in activation
            )
            amplified[feature] = float(np.clip(
                multiplicative + additive + self.config.recurrence_gain * recurrent,
                0.0, 1.0
            ))
        state["amplified_state"] = amplified
        return {
            "pre_gain": activation,
            "gain_vector": gains,
            "amplified_state": amplified,
            "changed_features": sum(
                abs(amplified.get(key, 0.0) - value) > _EPS
                for key, value in activation.items()
            ),
        }

    def _stage_11(
        self, state: dict[str, Any], learning_authorized: bool,
        feedback_reward: float | None,
    ) -> dict[str, Any]:
        """Suppress irrelevant and competing feature activity without deleting concept knowledge."""
        amplified = state.get("amplified_state", state["feature_activation"])
        relevance = state["task_relevance"]
        values = list(amplified.values())
        mean_competition = self._mean(values)
        inhibition: dict[str, float] = {}
        controlled: dict[str, float] = {}
        for feature, value in amplified.items():
            task_inhibition = self.config.inhibition_strength * (
                1.0 - relevance.get(feature, 0.5)
            ) * state["control_activity"]
            lateral = 0.12 * max(0.0, mean_competition - value)
            total = float(np.clip(task_inhibition + lateral, 0.0, 1.0))
            inhibition[feature] = total
            controlled[feature] = float(np.clip(value - total, 0.0, 1.0))
        state["feature_inhibition"] = inhibition
        state["controlled_state"] = controlled
        return {
            "inhibitory_drive": inhibition,
            "controlled_feature_state": controlled,
            "knowledge_deleted": False,
            "rectification": "max(0, activation - inhibition)",
        }

    def _stage_12(
        self, state: dict[str, Any], learning_authorized: bool,
        feedback_reward: float | None,
    ) -> dict[str, Any]:
        """Iteratively retrieve weak but relevant features using target-weighted recurrent spreading."""
        activation = dict(state["controlled_state"])
        relevance = state["task_relevance"]
        gains = state["feature_gain"]
        associations = state["associations"]
        history = []
        converged = False
        for iteration in range(self.config.max_control_iterations):
            incoming: dict[str, float] = defaultdict(float)
            for source, value in activation.items():
                for target, weight in associations.get(source, {}).items():
                    incoming[target] += value * weight
            priority = _softmax([
                2.0 * relevance.get(feature, 0.5) * value
                for feature, value in activation.items()
            ], 0.6)
            feature_names = list(activation)
            priority_by_feature = {
                feature: probability * max(len(activation), 1)
                for feature, probability in zip(feature_names, priority)
            }
            next_state = {}
            for feature, value in activation.items():
                drive = (
                    gains.get(feature, 0.45) * incoming.get(feature, 0.0)
                    * (0.5 + priority_by_feature.get(feature, 0.0))
                )
                retained = 0.72 * value
                inhibited = state["feature_inhibition"].get(feature, 0.0) * 0.15
                next_state[feature] = float(np.clip(
                    retained + 0.28 * drive - inhibited, 0.0, 1.0
                ))
            change = math.sqrt(sum(
                (next_state.get(key, 0.0) - activation.get(key, 0.0)) ** 2
                for key in set(next_state) | set(activation)
            ))
            history.append({
                "iteration": iteration + 1,
                "change_norm": change,
                "priority_distribution": {
                    feature: probability
                    for feature, probability in zip(feature_names, priority)
                },
            })
            activation = next_state
            if change < self.config.convergence_epsilon:
                converged = True
                break
        state["retrieved_state"] = activation
        state["controlled_state"] = activation
        state["retrieval_history"] = history
        return {
            "iterations": len(history),
            "converged": converged,
            "retrieved_state": activation,
            "iteration_history": history,
            "weak_feature_search": "relevance-weighted spreading activation",
        }

    def _evaluate_candidates(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        """Score candidates from independent semantic, task, feature, and contextual evidence."""
        evaluated = []
        for initial in state["candidate_scores"]:
            concept_id = initial["concept_id"]
            features = self._feature_by_concept[concept_id]
            retrieval_state = state.get("controlled_state") or state["semantic_state"]
            semantic_match = _cosine(features, retrieval_state)
            task_match = (
                _cosine(features, state["task_state"])
                if state["task_state"] else 0.5
            )
            context_match = (
                _cosine(features, state["context_state"])
                if state["context_state"] else 0.5
            )
            relevance = state["task_relevance"]
            active_total = sum(features.values()) or 1.0
            feature_match = sum(
                value * relevance.get(feature, 0.5)
                * (0.5 + 0.5 * retrieval_state.get(feature, 0.0))
                for feature, value in features.items()
            ) / active_total
            score = (
                0.10 * initial["lexical_match"]
                + 0.18 * semantic_match
                + 0.28 * task_match
                + 0.22 * feature_match
                + 0.22 * context_match
            )
            evaluated.append({
                **initial,
                "semantic_match": semantic_match,
                "task_match": task_match,
                "feature_relevance": feature_match,
                "context_compatibility": context_match,
                "integrated_score": float(np.clip(score, 0.0, 1.0)),
            })
        probabilities = _softmax(
            [item["integrated_score"] for item in evaluated],
            self.config.softmax_temperature,
        )
        for item, probability in zip(evaluated, probabilities):
            item["posterior"] = probability
        return sorted(evaluated, key=lambda item: item["posterior"], reverse=True)

    def _stage_13(
        self, state: dict[str, Any], learning_authorized: bool,
        feedback_reward: float | None,
    ) -> dict[str, Any]:
        """Score every candidate against integrated semantics, task relevance, and context."""
        evaluated = self._evaluate_candidates(state)
        state["candidate_scores"] = evaluated
        state["candidate_probabilities"] = [item["posterior"] for item in evaluated]
        return {
            "ranked_candidates": evaluated,
            "scoring_weights": {
                "lexical": 0.10, "semantic": 0.18, "task": 0.28,
                "feature": 0.22, "context": 0.22,
            },
            "probability_sum": float(sum(state["candidate_probabilities"])),
            "softmax_temperature": self.config.softmax_temperature,
        }

    def _stage_14(
        self, state: dict[str, Any], learning_authorized: bool,
        feedback_reward: float | None,
    ) -> dict[str, Any]:
        """Apply a posterior and margin gate while preserving alternative interpretations."""
        ranked = state["candidate_scores"]
        best = ranked[0]
        second = ranked[1] if len(ranked) > 1 else None
        margin = float(best["posterior"] - (second["posterior"] if second else 0.0))
        confidence = float(best["posterior"])
        accepted = (
            confidence >= self.config.confidence_threshold
            and margin >= self.config.selection_margin_threshold
        )
        state["selection"] = {
            "accepted": accepted,
            "confidence": confidence,
            "margin": margin,
            "candidate_id": best["concept_id"] if accepted else None,
        }
        state["selected_concept"] = (
            self._serialize_concept(self._concept_by_id[best["concept_id"]])
            if accepted else None
        )
        feature_selection = {
            feature: value for feature, value in state["controlled_state"].items()
            if value >= self.config.feature_threshold
            and state["task_relevance"].get(feature, 0.5) >= 0.35
        }
        state["selected_features"] = feature_selection
        return {
            "accepted": accepted,
            "winner": best,
            "runner_up": second,
            "confidence": confidence,
            "margin": margin,
            "confidence_threshold": self.config.confidence_threshold,
            "margin_threshold": self.config.selection_margin_threshold,
            "selected_feature_set": feature_selection,
            "alternatives_preserved": ranked,
        }

    def _stage_15(
        self, state: dict[str, Any], learning_authorized: bool,
        feedback_reward: float | None,
    ) -> dict[str, Any]:
        """Monitor task-fit error and run a bounded gain/inhibition feedback correction loop."""
        request: CSCRequest = state["request"]
        iterations = []
        best = state["candidate_scores"][0]
        gain_adjustment = 0.0
        inhibition_adjustment = 0.0
        demand = state["control_demand"]
        for iteration in range(self.config.max_control_iterations):
            error = float(np.clip(request.required_fit - best["integrated_score"], -1.0, 1.0))
            interference = 1.0 - state["selection"].get("margin", 0.0)
            if state["selection"].get("accepted") and error <= 0.0:
                iterations.append({
                    "iteration": iteration + 1, "retrieval_error": error,
                    "stopped": "selection_satisfied",
                })
                break
            gain_adjustment = float(np.clip(
                gain_adjustment + self.config.learning_rate * error, -0.4, 0.4
            ))
            inhibition_adjustment = float(np.clip(
                inhibition_adjustment + 0.08 * interference, 0.0, 0.4
            ))
            demand = float(np.clip(demand + 0.18 * max(error, 0.0), 0.0, 1.0))
            controlled = {}
            for feature, value in state["controlled_state"].items():
                relevant = state["task_relevance"].get(feature, 0.5)
                controlled[feature] = float(np.clip(
                    value + gain_adjustment * relevant
                    - inhibition_adjustment * (1.0 - relevant), 0.0, 1.0
                ))
            state["controlled_state"] = controlled
            state["control_demand"] = demand
            reranked = self._evaluate_candidates(state)
            state["candidate_scores"] = reranked
            state["candidate_probabilities"] = [item["posterior"] for item in reranked]
            best = reranked[0]
            second = reranked[1] if len(reranked) > 1 else None
            margin = best["posterior"] - (second["posterior"] if second else 0.0)
            confidence = best["posterior"]
            accepted = (
                confidence >= self.config.confidence_threshold
                and margin >= self.config.selection_margin_threshold
            )
            state["selection"] = {
                "accepted": accepted, "confidence": confidence, "margin": margin,
                "candidate_id": best["concept_id"] if accepted else None,
            }
            state["selected_concept"] = (
                self._serialize_concept(self._concept_by_id[best["concept_id"]])
                if accepted else None
            )
            iterations.append({
                "iteration": iteration + 1,
                "retrieval_error": error,
                "control_demand": demand,
                "gain_adjustment": gain_adjustment,
                "inhibition_adjustment": inhibition_adjustment,
                "best_candidate": best["concept_id"],
                "best_score": best["integrated_score"],
                "confidence": confidence,
                "margin": margin,
                "accepted": accepted,
            })
            if accepted and error <= 0.0:
                break
        state["control_iterations"] = iterations
        state["control_demand"] = demand
        state["selected_features"] = {
            feature: value for feature, value in state["controlled_state"].items()
            if value >= self.config.feature_threshold
            and state["task_relevance"].get(feature, 0.5) >= 0.35
        }
        return {
            "feedback_iterations": iterations,
            "gain_adjustment": gain_adjustment,
            "inhibition_adjustment": inhibition_adjustment,
            "final_control_demand": demand,
            "selection_after_feedback": state["selection"],
            "feedback_loop_bounded": True,
        }

    def _stage_16(
        self, state: dict[str, Any], learning_authorized: bool,
        feedback_reward: float | None,
    ) -> dict[str, Any]:
        """Aggregate selected feature activity and construct inventory-supported feature relations."""
        selected = state["selected_features"]
        concept_id = state["selection"].get("candidate_id")
        concept_features = (
            self._feature_by_concept.get(concept_id, {}) if concept_id else {}
        )
        integrated = {
            feature: float(np.clip(
                value * concept_features.get(feature, 0.5), 0.0, 1.0
            ))
            for feature, value in selected.items()
        }
        relations: dict[str, float] = {}
        keys = sorted(integrated)
        compatible = 0
        evaluated = 0
        for index, left in enumerate(keys):
            for right in keys[index + 1:]:
                weight = max(
                    (features.get(left, 0.0) * features.get(right, 0.0)
                     for features in self._feature_by_concept.values()),
                    default=0.0,
                )
                relations[left + "|" + right] = float(weight)
                evaluated += 1
                compatible += int(weight > 0.0)
        state["integrated_state"] = integrated
        consistency = compatible / evaluated if evaluated else 1.0
        state["feature_relations"] = relations
        return {
            "integrated_features": integrated,
            "feature_relations": relations,
            "relation_consistency": consistency,
            "features_from_selected_candidate": bool(concept_id),
        }

    def _stage_17(
        self, state: dict[str, Any], learning_authorized: bool,
        feedback_reward: float | None,
    ) -> dict[str, Any]:
        """Score supplied response options and emit an interpretation only after selection gates pass."""
        accepted = bool(state["selection"].get("accepted"))
        selected = state["selected_concept"]
        response_options = list(state["request"].response_options)
        option_scores = []
        if accepted and selected:
            selected_label = str(selected["label"]).casefold()
            for option in response_options:
                normalized = _normalize_text(option)
                score = 1.0 if selected_label in normalized else SequenceMatcher(
                    None, selected_label, normalized
                ).ratio()
                option_scores.append({"response": option, "score": float(score)})
            option_scores.sort(key=lambda item: item["score"], reverse=True)
            label = option_scores[0]["response"] if option_scores else selected["label"]
            state["response"] = {
                "emitted": True,
                "response": label,
                "response_kind": "supplied_option" if option_scores else "selected_concept_label",
                "confidence": state["selection"]["confidence"],
            }
        else:
            state["response"] = {
                "emitted": False,
                "response": None,
                "response_kind": "abstention",
                "confidence": state["selection"].get("confidence", 0.0),
            }
        return {
            "response": state["response"],
            "response_option_scores": option_scores,
            "output_threshold": self.config.confidence_threshold,
            "language_generation_scope": "structured candidate label; no unrestricted text generation",
        }

    def _stage_18(
        self, state: dict[str, Any], learning_authorized: bool,
        feedback_reward: float | None,
    ) -> dict[str, Any]:
        """Preview adaptation by default and commit Hebbian/reward updates only with explicit feedback."""
        request: CSCRequest = state["request"]
        if not learning_authorized:
            learning = {
                "status": "preview_only_authorization_required",
                "committed": False,
                "temporary_control_state": {
                    "gain": state["feature_gain"],
                    "inhibition": state["feature_inhibition"],
                    "demand": state["control_demand"],
                },
                "long_term_memory_changed": False,
            }
        else:
            concept_id = request.feedback_concept_id
            assert concept_id is not None
            reward = float(feedback_reward)
            features = self._feature_by_concept[concept_id]
            delta_links: dict[str, float] = {}
            names = sorted(features)
            for index, left in enumerate(names):
                for right in names[index + 1:]:
                    delta = self.config.learning_rate * reward * features[left] * features[right]
                    self.associations.setdefault(left, {})[right] = max(
                        0.0, self.associations.get(left, {}).get(right, 0.0) + delta
                    )
                    self.associations.setdefault(right, {})[left] = max(
                        0.0, self.associations.get(right, {}).get(left, 0.0) + delta
                    )
                    delta_links[left + "|" + right] = delta
            self._retrieval_counts[concept_id] += 1
            self._learned_priors[concept_id] += self.config.learning_rate * reward
            self._control_bias = float(np.clip(
                self._control_bias + self.config.learning_rate * (1.0 - reward),
                -1.0, 1.0
            ))
            self._learning_generation += 1
            learning = {
                "status": "committed",
                "committed": True,
                "feedback_concept_id": concept_id,
                "feedback_reward": reward,
                "hebbian_link_deltas": delta_links,
                "retrieval_frequency": self._retrieval_counts[concept_id],
                "learned_prior": self._learned_priors[concept_id],
                "control_bias": self._control_bias,
                "learning_generation": self._learning_generation,
                "long_term_memory_changed": True,
                "persistence": "processor instance memory only; export/store separately for durable learning",
            }
        state["learning"] = learning
        return {
            "learning": learning,
            "temporary_control_changes_separated_from_long_term_learning": True,
            "authorization_required": True,
        }

    def _serialize_concept(self, concept: SemanticConcept) -> dict[str, Any]:
        """Convert a pinned concept record into a JSON-compatible selected-concept summary."""
        return {
            "concept_id": concept.concept_id,
            "label": concept.label,
            "aliases": list(concept.aliases),
            "semantic_features": _feature_map(concept.semantic_features),
            "modality_features": {
                name: _feature_map(values) for name, values in concept.modality_features.items()
            },
            "source_reference": concept.source_reference,
            "verified_by": concept.verified_by,
        }
