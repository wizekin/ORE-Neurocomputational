"""Functional and biologically motivated implementation of all 38 TDPIA operations."""
from __future__ import annotations

from collections import Counter, defaultdict
import math
import re
from typing import Any, Mapping, Sequence

import numpy as np

from .biology import ComprehensionLIFSimulator, LIFConfig, population_drives
from .contracts import ComprehensionConfig, ComprehensionRequest, ComprehensionResult
from .registry import OPERATION_REGISTRY, STAGE_REGISTRY


_EPSILON = 1e-9
_FEATURE_NAMES = (
    "human", "animate", "physical", "edible", "tool", "action", "abstract",
    "location", "plural", "event", "artifact", "natural", "liquid", "concrete",
)
_PUNCTUATION = re.compile(r"[.!?;:,]$")

# Compact, inspectable lexical priors. Applications can replace or supplement these
# using ComprehensionRequest lexical_categories and semantic_features.
_LEXICON: dict[str, dict[str, Any]] = {
    "the": {"pos": "DET", "features": {}}, "a": {"pos": "DET", "features": {}},
    "an": {"pos": "DET", "features": {}}, "this": {"pos": "DET", "features": {}},
    "these": {"pos": "DET", "features": {"plural": 1.0}},
    "i": {"pos": "PRON", "features": {"human": 1.0, "animate": 1.0}},
    "you": {"pos": "PRON", "features": {"human": 1.0, "animate": 1.0}},
    "he": {"pos": "PRON", "features": {"human": 1.0, "animate": 1.0}},
    "she": {"pos": "PRON", "features": {"human": 1.0, "animate": 1.0}},
    "they": {"pos": "PRON", "features": {"human": 0.8, "animate": 1.0, "plural": 1.0}},
    "we": {"pos": "PRON", "features": {"human": 1.0, "animate": 1.0, "plural": 1.0}},
    "chef": {"pos": "NOUN", "features": {"human": 1.0, "animate": 1.0, "physical": 1.0, "concrete": 1.0}},
    "scientist": {"pos": "NOUN", "features": {"human": 1.0, "animate": 1.0, "physical": 1.0}},
    "doctor": {"pos": "NOUN", "features": {"human": 1.0, "animate": 1.0, "physical": 1.0}},
    "patient": {"pos": "NOUN", "features": {"human": 1.0, "animate": 1.0, "physical": 1.0}},
    "child": {"pos": "NOUN", "features": {"human": 1.0, "animate": 1.0, "physical": 1.0}},
    "children": {"pos": "NOUN", "features": {"human": 1.0, "animate": 1.0, "physical": 1.0, "plural": 1.0}},
    "cat": {"pos": "NOUN", "features": {"animate": 1.0, "physical": 1.0, "concrete": 1.0}},
    "dog": {"pos": "NOUN", "features": {"animate": 1.0, "physical": 1.0, "concrete": 1.0}},
    "stone": {"pos": "NOUN", "features": {"physical": 1.0, "natural": 1.0, "concrete": 1.0}},
    "bread": {"pos": "NOUN", "features": {"physical": 1.0, "edible": 1.0, "concrete": 1.0}},
    "meal": {"pos": "NOUN", "features": {"physical": 1.0, "edible": 1.0, "concrete": 1.0, "event": 0.2}},
    "food": {"pos": "NOUN", "features": {"physical": 1.0, "edible": 1.0, "concrete": 1.0}},
    "ingredients": {"pos": "NOUN", "features": {"physical": 1.0, "edible": 1.0, "plural": 1.0}},
    "knife": {"pos": "NOUN", "features": {"physical": 1.0, "tool": 1.0, "artifact": 1.0, "concrete": 1.0}},
    "fork": {"pos": "NOUN", "features": {"physical": 1.0, "tool": 1.0, "artifact": 1.0, "concrete": 1.0}},
    "spoon": {"pos": "NOUN", "features": {"physical": 1.0, "tool": 1.0, "artifact": 1.0, "concrete": 1.0}},
    "cloud": {"pos": "NOUN", "features": {"physical": 0.7, "natural": 1.0, "concrete": 0.6}},
    "cabinet": {"pos": "NOUN", "features": {"physical": 1.0, "artifact": 1.0, "concrete": 1.0}},
    "telescope": {"pos": "NOUN", "features": {"physical": 1.0, "tool": 0.9, "artifact": 1.0}},
    "dragon": {"pos": "NOUN", "features": {"animate": 1.0, "physical": 1.0, "natural": 0.2}},
    "kitchen": {"pos": "NOUN", "features": {"physical": 1.0, "location": 1.0, "concrete": 1.0}},
    "guests": {"pos": "NOUN", "features": {"human": 1.0, "animate": 1.0, "plural": 1.0}},
    "keys": {"pos": "NOUN", "features": {"physical": 1.0, "artifact": 1.0, "plural": 1.0}},
    "book": {"pos": "NOUN", "features": {"physical": 1.0, "artifact": 1.0, "concrete": 1.0}},
    "water": {"pos": "NOUN", "features": {"physical": 1.0, "liquid": 1.0, "natural": 1.0}},
    "garden": {"pos": "NOUN", "features": {"physical": 1.0, "location": 1.0, "natural": 1.0}},
    "prepared": {"pos": "VERB", "features": {"action": 1.0, "event": 1.0}, "valency": "transitive"},
    "prepare": {"pos": "VERB", "features": {"action": 1.0, "event": 1.0}, "valency": "transitive"},
    "cut": {"pos": "VERB", "features": {"action": 1.0, "event": 1.0}, "valency": "transitive"},
    "cuts": {"pos": "VERB", "features": {"action": 1.0, "event": 1.0}, "valency": "transitive"},
    "ate": {"pos": "VERB", "features": {"action": 1.0, "event": 1.0}, "valency": "transitive"},
    "eat": {"pos": "VERB", "features": {"action": 1.0, "event": 1.0}, "valency": "transitive"},
    "eats": {"pos": "VERB", "features": {"action": 1.0, "event": 1.0}, "valency": "transitive"},
    "cooked": {"pos": "VERB", "features": {"action": 1.0, "event": 1.0}, "valency": "transitive"},
    "cook": {"pos": "VERB", "features": {"action": 1.0, "event": 1.0}, "valency": "transitive"},
    "saw": {"pos": "VERB", "features": {"action": 1.0, "event": 1.0}, "valency": "transitive"},
    "see": {"pos": "VERB", "features": {"action": 1.0, "event": 1.0}, "valency": "transitive"},
    "discovered": {"pos": "VERB", "features": {"action": 1.0, "event": 1.0}, "valency": "transitive"},
    "discover": {"pos": "VERB", "features": {"action": 1.0, "event": 1.0}, "valency": "transitive"},
    "is": {"pos": "AUX", "features": {"action": 0.0}, "number": "singular"},
    "are": {"pos": "AUX", "features": {"action": 0.0}, "number": "plural"},
    "was": {"pos": "AUX", "features": {"action": 0.0}, "number": "singular"},
    "were": {"pos": "AUX", "features": {"action": 0.0}, "number": "plural"},
    "with": {"pos": "ADP", "features": {}}, "to": {"pos": "ADP", "features": {}},
    "in": {"pos": "ADP", "features": {}}, "on": {"pos": "ADP", "features": {}},
    "who": {"pos": "PRON", "features": {"human": 1.0, "animate": 1.0}},
    "that": {"pos": "SCONJ", "features": {}}, "which": {"pos": "PRON", "features": {}},
    "carefully": {"pos": "ADV", "features": {}}, "quickly": {"pos": "ADV", "features": {}},
    "careful": {"pos": "ADJ", "features": {}}, "large": {"pos": "ADJ", "features": {"physical": 0.5}},
    "and": {"pos": "CCONJ", "features": {}}, "because": {"pos": "SCONJ", "features": {}},
    "not": {"pos": "PART", "features": {}},
}

_EVENT_SCHEMAS: dict[str, list[str]] = {
    "cooking": ["obtain ingredients", "prepare ingredients", "cut or mix", "cook when needed", "serve food"],
    "eating": ["obtain food", "prepare food", "eat", "finish meal"],
    "medical": ["patient arrives", "describe symptoms", "examination", "evaluate diagnosis", "recommend treatment"],
    "discovery": ["observe", "investigate", "find evidence", "interpret discovery"],
    "perception": ["perceiver attends", "sensory target is detected", "interpret sensory evidence"],
    "movement": ["agent starts movement", "travels along a path", "reaches destination"],
    "generic": ["situation begins", "participants interact", "situation changes"],
}
_EVENT_CUES = {
    "cooking": {"chef", "prepared", "prepare", "cut", "cooked", "cook", "kitchen", "ingredients"},
    "eating": {"eat", "eats", "ate", "meal", "food", "bread"},
    "medical": {"doctor", "patient", "symptom", "diagnosis", "treatment"},
    "discovery": {"scientist", "discovered", "discover", "evidence"},
    "perception": {"saw", "see", "telescope"},
    "movement": {"travel", "walk", "run", "arrived", "moved"},
}
_ROLE_NAMES = ("agent", "patient", "instrument", "location", "predicate", "other")


def _normalize_word(value: str) -> str:
    """Normalize a surface token for lexical lookup while retaining source text elsewhere."""
    return value.casefold().strip(" \t\r\n.,!?;:()[]{}\"'“”‘’")


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp a numeric score to a bounded interval and avoid propagating non-finite values."""
    number = float(value)
    if not math.isfinite(number):
        return low
    return min(high, max(low, number))


def _softmax(scores: Mapping[str, float], temperature: float = 1.0) -> dict[str, float]:
    """Convert arbitrary candidate scores into a stable normalized probability distribution."""
    if not scores:
        return {}
    scale = max(float(temperature), _EPSILON)
    keys = list(scores)
    values = np.asarray([float(scores[key]) / scale for key in keys], dtype=np.float64)
    values -= float(np.max(values))
    exponentials = np.exp(np.clip(values, -700.0, 0.0))
    total = float(np.sum(exponentials))
    if not math.isfinite(total) or total <= _EPSILON:
        return {key: 1.0 / len(keys) for key in keys}
    return {key: float(value / total) for key, value in zip(keys, exponentials)}


def _entropy(probabilities: Mapping[str, float]) -> float:
    """Calculate Shannon entropy in bits for a normalized or normalizable distribution."""
    values = np.asarray([max(0.0, float(value)) for value in probabilities.values()], dtype=np.float64)
    if values.size == 0 or float(values.sum()) <= _EPSILON:
        return 0.0
    values /= values.sum()
    positive = values[values > 0.0]
    return float(-np.sum(positive * np.log2(positive)))


def _cosine_similarity(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    """Compare sparse semantic-feature mappings using cosine similarity in a shared feature space."""
    dimensions = sorted(set(left) | set(right))
    if not dimensions:
        return 0.0
    a = np.asarray([float(left.get(key, 0.0)) for key in dimensions], dtype=np.float64)
    b = np.asarray([float(right.get(key, 0.0)) for key in dimensions], dtype=np.float64)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if denominator <= _EPSILON else _clamp(float(np.dot(a, b) / denominator), -1.0, 1.0)


def _js_divergence(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    """Calculate bounded Jensen-Shannon divergence in bits between two sparse distributions."""
    keys = sorted(set(left) | set(right))
    if not keys:
        return 0.0
    p = np.asarray([max(0.0, float(left.get(key, 0.0))) for key in keys], dtype=np.float64)
    q = np.asarray([max(0.0, float(right.get(key, 0.0))) for key in keys], dtype=np.float64)
    p = p / max(float(p.sum()), _EPSILON)
    q = q / max(float(q.sum()), _EPSILON)
    middle = 0.5 * (p + q)
    p_mask, q_mask = p > 0, q > 0
    left_kl = float(np.sum(p[p_mask] * np.log2(p[p_mask] / middle[p_mask])))
    right_kl = float(np.sum(q[q_mask] * np.log2(q[q_mask] / middle[q_mask])))
    return _clamp(0.5 * (left_kl + right_kl), 0.0, 1.0)


def _mean_feature_vectors(vectors: Sequence[Mapping[str, float]], weights: Sequence[float]) -> dict[str, float]:
    """Pool semantic feature mappings using normalized non-negative weights."""
    if not vectors:
        return {name: 0.0 for name in _FEATURE_NAMES}
    total = sum(max(0.0, float(weight)) for weight in weights)
    if total <= _EPSILON:
        weights = [1.0] * len(vectors)
        total = float(len(vectors))
    dimensions = sorted(set(_FEATURE_NAMES).union(*(set(vector) for vector in vectors)))
    return {
        dimension: float(sum(
            max(0.0, float(weight)) * float(vector.get(dimension, 0.0))
            for vector, weight in zip(vectors, weights)
        ) / total)
        for dimension in dimensions
    }


def _finite_tree(value: Any) -> bool:
    """Recursively verify that numeric values in a result tree are finite."""
    if isinstance(value, Mapping):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return bool(math.isfinite(float(value)))
    if isinstance(value, (int, np.integer, str, bool)) or value is None:
        return True
    return True


class ComprehensionProcessor:
    """Run the seven-stage, 38-operation top-down predictive and incremental comprehension algorithm.

    The functional track is a deterministic, inspectable symbolic/statistical model. The biological
    track is a separate LIF reference simulation and must not be interpreted as a literal brain model.
    """

    def __init__(self, config: ComprehensionConfig | None = None) -> None:
        """Initialize validated controls and a small experience-dependent association store."""
        self.config = config or ComprehensionConfig()
        self._learning_memory: dict[str, Any] = {
            "bigrams": Counter(),
            "event_counts": Counter(),
            "association_weights": defaultdict(float),
            "learning_updates": 0,
            "processed_sequences": 0,
        }

    def _entry_for(self, word: str, state: Mapping[str, Any]) -> dict[str, Any]:
        """Merge built-in lexical priors with caller-provided category and semantic-feature evidence."""
        request: ComprehensionRequest = state["request"]
        base = _LEXICON.get(word, {})
        category_map = {_normalize_word(key): str(value).upper() for key, value in request.lexical_categories.items()}
        category = category_map.get(word, str(base.get("pos", ""))).upper()
        if not category:
            if word.endswith("ly"):
                category = "ADV"
            elif word.endswith(("ed", "ing")):
                category = "VERB"
            elif word.endswith(("ous", "ful", "ive", "able")):
                category = "ADJ"
            else:
                category = "NOUN"
        features = {name: 0.0 for name in _FEATURE_NAMES}
        features.update({str(key): _clamp(float(value), 0.0, 1.0) for key, value in base.get("features", {}).items()})
        supplied = {
            _normalize_word(key): value for key, value in request.semantic_features.items()
        }.get(word, {})
        features.update({str(key): _clamp(float(value), 0.0, 1.0) for key, value in supplied.items()})
        return {
            "word": word,
            "pos": category,
            "features": features,
            "number": base.get("number", "plural" if features.get("plural", 0.0) >= 0.5 else "singular"),
            "valency": base.get("valency", "unknown"),
            "known_lexeme": bool(base or supplied or word in category_map),
        }

    def _candidate_vocabulary(self, state: Mapping[str, Any], categories: Sequence[str] | None = None) -> list[str]:
        """Build the candidate lexicon from built-in entries and caller-supplied lexical evidence."""
        request: ComprehensionRequest = state["request"]
        candidates = set(_LEXICON)
        candidates.update(_normalize_word(word) for word in request.lexical_categories)
        candidates.update(_normalize_word(word) for word in request.semantic_features)
        candidates.discard("")
        if categories:
            allowed = {str(category).upper() for category in categories}
            candidates = {word for word in candidates if self._entry_for(word, state)["pos"] in allowed}
        candidates.difference_update({"", "the", "a", "an", "and", "or", "not", "to", "in", "on", "with"})
        return sorted(candidates)

    def _expected_pos(self, state: Mapping[str, Any]) -> list[str]:
        """Derive a compact next-category expectation from the current partial grammatical context."""
        tags = [item["pos"] for item in state["tagged_tokens"]]
        words = list(state["received_tokens"])
        last_word = words[-1] if words else ""
        last_tag = tags[-1] if tags else ""
        if last_tag in {"DET", "ADJ"}:
            return ["NOUN", "PRON"]
        if last_tag == "AUX":
            return ["VERB", "ADV"]
        if last_tag == "ADP":
            return ["DET", "NOUN", "PRON"]
        if last_word in {"who", "which", "that"}:
            return ["VERB", "AUX"]
        if "with" in words[-3:]:
            return ["DET", "NOUN"]
        if last_tag in {"NOUN", "PRON"} and not any(tag in {"VERB", "AUX"} for tag in tags):
            return ["VERB", "AUX"]
        if last_tag == "VERB":
            return ["DET", "NOUN", "ADP", "ADV"]
        return ["NOUN", "VERB", "ADJ", "ADV"]

    def _event_name(self, state: Mapping[str, Any]) -> str:
        """Choose the best-supported event schema from the already processed lexical context."""
        words = set(state["received_tokens"])
        scores = {
            event: len(words.intersection(cues))
            for event, cues in _EVENT_CUES.items()
        }
        if not scores or max(scores.values(), default=0) == 0:
            return "generic"
        return max(scores, key=scores.get)

    def _reset_sentence_state(self, state: dict[str, Any]) -> None:
        """Close one sentence while retaining its text as discourse context for the next sentence."""
        state["discourse_history"].extend(state["received_tokens"])
        state["received_tokens"] = []
        state["tagged_tokens"] = []
        state["syntactic_structure"] = {"open_constituents": [], "dependencies": [], "parse_candidates": []}
        state["semantic_concepts"] = []
        state["thematic_roles"] = {}
        state["sentence_hypotheses"] = {}
        state["sentence_complete"] = False
        state["current_sentence_interpretation"] = None

    def process(self, request: ComprehensionRequest) -> ComprehensionResult:
        """Execute operations 1–38 for each token, updating sentence beliefs and biological reference dynamics."""
        if not isinstance(request, ComprehensionRequest):
            raise TypeError("request must be a ComprehensionRequest")
        tokens = request.to_tokens()
        if not tokens:
            raise ValueError("request produced no tokens")
        state: dict[str, Any] = {
            "request": request,
            "sequence_id": request.sequence_id,
            "source_text": request.text if request.text is not None else " ".join(tokens),
            "all_input_tokens": tokens,
            "received_tokens": [],
            "discourse_history": list(request.discourse_context),
            "tagged_tokens": [],
            "syntactic_structure": {"open_constituents": [], "dependencies": [], "parse_candidates": []},
            "semantic_concepts": [],
            "thematic_roles": {},
            "sentence_hypotheses": {},
            "sentence_interpretations": [],
            "operation_trace": [],
            "operation_outputs": {},
            "token_records": [],
            "learning_memory": self._learning_memory,
            "sentence_complete": False,
        }
        lif_config = LIFConfig(
            population_size=self.config.lif_population_size,
            steps_per_operation=self.config.lif_steps_per_operation,
            dt_ms=self.config.lif_dt_ms,
            membrane_tau_ms=self.config.lif_membrane_tau_ms,
            refractory_ms=self.config.lif_refractory_ms,
            rest_mv=self.config.lif_rest_mv,
            reset_mv=self.config.lif_reset_mv,
            threshold_mv=self.config.lif_threshold_mv,
            resistance_mohm=self.config.lif_resistance_mohm,
            current_gain_pa=self.config.lif_current_gain_pa,
            recurrent_gain_pa=self.config.lif_recurrent_gain_pa,
            inhibition_gain_pa=self.config.lif_inhibition_gain_pa,
        )
        simulator = ComprehensionLIFSimulator(lif_config) if self.config.simulate_biology else None
        for token_index, surface in enumerate(tokens):
            if state["sentence_complete"]:
                self._reset_sentence_state(state)
            state["current_surface"] = surface
            state["current_word"] = _normalize_word(surface)
            state["token_index"] = token_index
            state["is_last_input"] = token_index == len(tokens) - 1
            state["has_sentence_boundary"] = bool(_PUNCTUATION.search(surface))
            state["operation_outputs"] = {}
            for spec in OPERATION_REGISTRY:
                method = getattr(self, spec.method_name)
                output = method(state)
                state["operation_outputs"][spec.operation_id] = output
                biological_step = None
                if simulator is not None:
                    drives = population_drives(
                        spec.number,
                        _clamp(float(output.get("activation", 0.35))),
                        prediction_error=_clamp(float(state.get("semantic_prediction_error", 0.0))),
                        conflict=_clamp(float(state.get("representational_conflict", 0.0))),
                        working_memory_load=_clamp(len(state["received_tokens"]) / 12.0),
                    )
                    biological_step = simulator.step(spec.operation_id, drives)
                trace = {
                    "operation_id": spec.operation_id,
                    "operation_number": spec.number,
                    "stage_id": spec.stage_id,
                    "stage": spec.stage_name,
                    "operation": spec.name,
                    "method": spec.method_name,
                    "status": "executed",
                    "mathematical_operation": spec.mathematical_form,
                    "biological_basis": spec.biological_basis,
                    "functional_output": output,
                    "biological_step": biological_step,
                    "token_index": token_index,
                    "token": surface,
                }
                state["operation_trace"].append(trace)
            state["token_records"].append({
                "index": token_index,
                "token": surface,
                "normalized": state["current_word"],
                "lexical_category": state.get("actual_lexical_entry", {}).get("pos"),
                "predicted_word_probability": float(state.get("actual_word_probability", 0.0)),
                "surprisal_bits": float(state.get("word_surprisal_bits", 0.0)),
                "prediction_entropy_bits": float(state.get("prediction_entropy_bits", 0.0)),
                "semantic_prediction_error": float(state.get("semantic_prediction_error", 0.0)),
                "n400_estimate_uv": float(state.get("n400_estimate_uv", 0.0)),
                "representational_conflict": float(state.get("representational_conflict", 0.0)),
                "reanalysis_required": bool(state.get("reanalysis_required", False)),
                "p600_estimate_uv": float(state.get("p600_estimate_uv", 0.0)),
                "sentence_complete": bool(state.get("sentence_complete", False)),
            })
        if not state["sentence_interpretations"] and state.get("current_sentence_interpretation"):
            state["sentence_interpretations"].append(state["current_sentence_interpretation"])
        stage_trace = [
            {
                "stage_id": stage.stage_id,
                "stage": stage.name,
                "purpose": stage.purpose,
                "operation_ids": [
                    item.operation_id for item in OPERATION_REGISTRY if item.stage_id == stage.stage_id
                ],
                "operations_per_token": stage.last_operation - stage.first_operation + 1,
                "executed_operations": (stage.last_operation - stage.first_operation + 1) * len(tokens),
                "status": "executed",
                "biological_basis": stage.biological_basis,
            }
            for stage in STAGE_REGISTRY
        ]
        biological = simulator.snapshot() if simulator is not None else {
            "enabled": False,
            "model": "disabled",
            "scientific_scope": "The LIF reference simulation was disabled by configuration.",
        }
        predictions = state.get("next_word_distribution", {})
        next_predictions = [
            {"word": word, "probability": float(probability)}
            for word, probability in sorted(predictions.items(), key=lambda item: item[1], reverse=True)[:10]
        ]
        finite = _finite_tree({
            "operation_trace": state["operation_trace"],
            "token_records": state["token_records"],
            "sentence_interpretations": state["sentence_interpretations"],
            "biological": biological,
        })
        expected_operations = len(tokens) * len(OPERATION_REGISTRY)
        diagnostics = {
            "valid": bool(
                finite
                and len(OPERATION_REGISTRY) == 38
                and len(state["operation_trace"]) == expected_operations
                and len(stage_trace) == 7
                and len(state["token_records"]) == len(tokens)
            ),
            "registered_operation_count": len(OPERATION_REGISTRY),
            "operations_executed": len(state["operation_trace"]),
            "expected_operations": expected_operations,
            "token_count": len(tokens),
            "sentence_count": len(state["sentence_interpretations"]),
            "finite_outputs": finite,
            "mean_word_surprisal_bits": float(np.mean([
                item["surprisal_bits"] for item in state["token_records"]
            ])),
            "mean_n400_estimate_uv": float(np.mean([
                item["n400_estimate_uv"] for item in state["token_records"]
            ])),
            "mean_p600_estimate_uv": float(np.mean([
                item["p600_estimate_uv"] for item in state["token_records"]
            ])),
            "learning_updates": int(self._learning_memory["learning_updates"]),
            "biological_valid": biological.get("numerically_valid") if simulator is not None else None,
        }
        metadata = {
            "algorithm": "TDPIA - Top-Down Prediction and Incremental Comprehension Algorithm",
            "version": "0.1.0",
            "operation_registry": "TDPI-001..TDPI-038",
            "stage_registry": "A..G",
            "language": request.language,
            "functional_model": "deterministic lexical-feature, probabilistic, and rule-based reference implementation",
            "biological_model": "coupled current-based LIF populations" if simulator is not None else None,
            "annotation_storage": "ore_logger.catalog.sync_repository indexes class/function docstrings into ore_logger.sqlite3",
            "scientific_limitations": [
                "The built-in lexicon and event schemas are small transparent priors, not a comprehensive language model.",
                "Callers should provide lexical categories, semantic features, and world knowledge for their target language or domain.",
                "N400 and P600 values are illustrative model estimates, not EEG measurements or clinical quantities.",
                "The LIF populations are coarse, uncalibrated analogies and are not anatomically mapped or validated.",
                "Passing the software tests validates code behavior, not human comprehension or biological accuracy.",
            ],
            "configuration": self.config.to_dict(),
        }
        return ComprehensionResult(
            sequence_id=request.sequence_id,
            source_text=state["source_text"],
            tokens=tokens,
            token_records=state["token_records"],
            sentence_interpretations=state["sentence_interpretations"],
            operation_trace=state["operation_trace"],
            stage_trace=stage_trace,
            next_word_predictions=next_predictions,
            biological=biological,
            diagnostics=diagnostics,
            metadata=metadata,
        )

    def op_01_reconstruct_context(self, state: dict[str, Any]) -> dict[str, Any]:
        """OPERATION 01 — Reconstruct context by retrieving recent words and weighted discourse cues.

        Mathematics: use recency weights a_i = rho^distance, then pool feature/context evidence.
        Biological analogy: recurrent activity and attention maintain currently relevant lexical and
        conceptual information. This is a functional approximation, not a neuronal memory equation.
        """
        history = list(state["discourse_history"]) + list(state["received_tokens"])
        recent = history[-12:]
        weights = [self.config.context_decay ** distance for distance in range(len(recent) - 1, -1, -1)]
        concepts = Counter(recent)
        activation = _clamp(sum(weights) / max(1.0, len(weights)))
        state["reconstructed_context"] = {
            "recent_tokens": recent,
            "recency_weights": weights,
            "active_concepts": dict(concepts),
            "context_strength": activation,
        }
        return {"activation": activation, "context_size": len(recent), "active_concepts": len(concepts)}

    def op_02_infer_situation(self, state: dict[str, Any]) -> dict[str, Any]:
        """OPERATION 02 — Infer a sentence-level event hypothesis from the active context.

        Mathematics: score event hypotheses from contextual cues and normalize with softmax.
        Biological analogy: distributed event and situation representations become active together
        when relevant language and conceptual cues are retrieved.
        """
        words = set(state["discourse_history"]) | set(state["received_tokens"])
        cue_counts = {
            event: len(words.intersection(cues))
            for event, cues in _EVENT_CUES.items()
        }
        if max(cue_counts.values(), default=0) == 0:
            # Do not turn absence of evidence into an arbitrary preference for the first event.
            scores = {event: 0.0 for event in cue_counts}
            scores["generic"] = 1.0
        else:
            scores = {
                event: float(count) + (0.05 if count else 0.0)
                for event, count in cue_counts.items()
            }
            scores["generic"] = 0.0
        probabilities = _softmax(scores, temperature=0.8)
        best = max(probabilities, key=probabilities.get)
        state["sentence_hypotheses"] = {
            event: {"probability": probability, "score": scores[event]}
            for event, probability in probabilities.items()
        }
        state["situation_hypothesis"] = best
        state["situation_uncertainty"] = _entropy(probabilities)
        return {"activation": probabilities[best], "event": best, "hypotheses": state["sentence_hypotheses"]}

    def op_03_predict_continuations(self, state: dict[str, Any]) -> dict[str, Any]:
        """OPERATION 03 — Generate phrase-level continuations rather than a single next-word guess.

        Mathematics: combine event/schema compatibility and learned bigram support, then softmax the
        candidate scores and calculate continuation entropy. Biological analogy: top-down cortical
        signals preactivate graded structural and conceptual continuations.
        """
        words = self._candidate_vocabulary(state, ["NOUN", "VERB", "ADJ", "ADV"])
        event = state.get("situation_hypothesis", "generic")
        last = state["received_tokens"][-1] if state["received_tokens"] else "<START>"
        bigrams = state["learning_memory"]["bigrams"]
        schema = _EVENT_SCHEMAS.get(event, _EVENT_SCHEMAS["generic"])
        scores: dict[str, float] = {}
        for word in words:
            entry = self._entry_for(word, state)
            score = 0.15
            if event == "cooking" and (entry["features"].get("edible", 0.0) or entry["features"].get("tool", 0.0)):
                score += 0.7
            if event in {"eating", "medical"} and entry["pos"] == "NOUN":
                score += 0.15
            if entry["pos"] in self._expected_pos(state):
                score += 0.25
            score += 0.12 * math.log1p(float(bigrams.get(f"{last}|{word}", 0)))
            if word in " ".join(schema).split():
                score += 0.1
            scores[word] = score
        ranked = sorted(_softmax(scores, self.config.prediction_temperature).items(),
                        key=lambda item: item[1], reverse=True)[:self.config.max_candidates]
        distribution = _softmax(dict(ranked), self.config.prediction_temperature)
        state["continuation_candidates"] = [
            {"word": word, "probability": probability, "pos": self._entry_for(word, state)["pos"]}
            for word, probability in distribution.items()
        ]
        state["continuation_distribution"] = distribution
        state["continuation_entropy"] = _entropy(distribution)
        return {"activation": _clamp(1.0 - state["continuation_entropy"] / 5.0),
                "candidate_count": len(distribution), "entropy_bits": state["continuation_entropy"]}

    def op_04_examine_syntax(self, state: dict[str, Any]) -> dict[str, Any]:
        """OPERATION 04 — Inspect the partial grammatical structure and identify open constituents.

        Mathematics: parse the sequence of received lexical categories into completed and unresolved
        slots. Biological analogy: recurrent temporal-frontal representations maintain partial parses
        while additional words arrive.
        """
        tags = [item["pos"] for item in state["tagged_tokens"]]
        words = state["received_tokens"]
        open_constituents: list[str] = []
        if tags and tags[-1] == "DET":
            open_constituents.append("determiner_phrase_requires_nominal_head")
        if tags and tags[-1] == "ADP":
            open_constituents.append("prepositional_phrase_requires_complement")
        if tags and tags[-1] == "AUX":
            open_constituents.append("auxiliary_requires_verb_or_predicate")
        if "who" in words[-4:] or "which" in words[-4:]:
            open_constituents.append("relative_clause_requires_predicate")
        if tags and not any(tag in {"VERB", "AUX"} for tag in tags):
            open_constituents.append("main_clause_predicate_unresolved")
        state["syntactic_structure"]["open_constituents"] = open_constituents
        state["syntactic_structure"]["received_categories"] = tags
        return {"activation": _clamp(len(tags) / 8.0 + 0.2),
                "received_categories": tags, "open_constituents": open_constituents}

    def op_05_predict_syntactic_category(self, state: dict[str, Any]) -> dict[str, Any]:
        """OPERATION 05 — Predict the next grammatical category using the partial parse.

        Mathematics: assign contextual scores to part-of-speech categories and normalize using softmax.
        Biological analogy: grammatical-category populations receive graded activation from earlier
        words and unresolved phrase structures.
        """
        expected = self._expected_pos(state)
        scores = {category: (1.0 if category in expected else 0.1) for category in
                  ("NOUN", "VERB", "ADJ", "ADV", "DET", "PRON", "ADP", "AUX")}
        distribution = _softmax(scores, temperature=0.55)
        state["syntactic_category_prediction"] = distribution
        state["expected_pos"] = expected
        return {"activation": max(distribution.values()), "expected_categories": expected,
                "category_probabilities": distribution}

    def op_06_predict_dependencies(self, state: dict[str, Any]) -> dict[str, Any]:
        """OPERATION 06 — Predict unresolved grammatical dependencies across intervening words.

        Mathematics: maintain conditional expectations for missing subject, object, and relative-clause
        links. Biological analogy: distributed activity can preserve filler and argument expectations
        while intervening lexical material is processed.
        """
        words = state["received_tokens"]
        tags = [item["pos"] for item in state["tagged_tokens"]]
        dependencies: list[dict[str, Any]] = []
        if words and words[-1] in {"who", "which", "that"}:
            dependencies.append({"relation": "relative_clause_predicate", "status": "open", "cue": words[-1]})
        if "VERB" in tags:
            verb_index = max(index for index, tag in enumerate(tags) if tag == "VERB")
            if not any(tag == "NOUN" for tag in tags[verb_index + 1:]):
                dependencies.append({"relation": "verb_object", "status": "possibly_open", "head_index": verb_index})
        if tags and not any(tag in {"VERB", "AUX"} for tag in tags):
            dependencies.append({"relation": "main_clause_predicate", "status": "expected"})
        state["syntactic_structure"]["dependencies"] = dependencies
        return {"activation": _clamp(len(dependencies) / 3.0),
                "expected_dependencies": dependencies}

    def op_07_predict_agreement(self, state: dict[str, Any]) -> dict[str, Any]:
        """OPERATION 07 — Carry agreement features and structural constraints into future predictions.

        Mathematics: derive a weighted compatibility expectation for number and person features.
        Biological analogy: recurrent feature representations let earlier subjects constrain later
        verb forms; the constraint is graded rather than an absolute rule.
        """
        subject = None
        for item in state["tagged_tokens"]:
            if item["pos"] in {"NOUN", "PRON"}:
                subject = item
                break
            if item["pos"] in {"VERB", "AUX"}:
                break
        expected_number = subject.get("number", "singular") if subject else None
        state["agreement_expectation"] = {
            "subject": subject.get("word") if subject else None,
            "expected_number": expected_number,
            "violation": 0.0,
        }
        return {"activation": 0.55 if subject else 0.15,
                "agreement_expectation": state["agreement_expectation"]}

    def op_08_maintain_competing_parses(self, state: dict[str, Any]) -> dict[str, Any]:
        """OPERATION 08 — Maintain several syntactic alternatives under ambiguity.

        Mathematics: normalize parse scores with softmax and retain a bounded beam of alternatives.
        Biological analogy: competing population activity can preserve partially active parses before
        later evidence resolves attachment or category ambiguity.
        """
        open_slots = state["syntactic_structure"].get("open_constituents", [])
        alternatives = [
            {"parse": "incremental_preferred", "score": 0.8 - 0.1 * len(open_slots)},
            {"parse": "alternative_attachment", "score": 0.45 + 0.04 * len(open_slots)},
            {"parse": "reanalysis_candidate", "score": 0.2},
        ]
        probabilities = _softmax({item["parse"]: item["score"] for item in alternatives}, 0.8)
        parses = [
            {"parse": item["parse"], "probability": probabilities[item["parse"]]}
            for item in alternatives
        ]
        parses.sort(key=lambda item: item["probability"], reverse=True)
        state["syntactic_structure"]["parse_candidates"] = parses[:self.config.max_hypotheses]
        state["parse_uncertainty"] = _entropy(probabilities)
        return {"activation": _clamp(1.0 - state["parse_uncertainty"] / 2.0),
                "parse_candidates": state["syntactic_structure"]["parse_candidates"]}

    def op_09_retrieve_semantic_concepts(self, state: dict[str, Any]) -> dict[str, Any]:
        """OPERATION 09 — Retrieve concepts and semantic features associated with processed words.

        Mathematics: aggregate feature activations through weighted concept associations and compare
        concept vectors in a shared feature space. Biological analogy: distributed semantic-memory
        populations activate related concepts through learned connections.
        """
        concepts = []
        for word in state["received_tokens"]:
            entry = self._entry_for(word, state)
            concepts.append({"word": word, "category": entry["pos"], "features": entry["features"]})
        active = sorted({item["word"] for item in concepts})
        state["activated_semantic_concepts"] = concepts
        state["active_concept_inventory"] = active
        return {"activation": _clamp(len(active) / 8.0), "active_concepts": active[-12:]}

    def op_10_predict_semantic_categories(self, state: dict[str, Any]) -> dict[str, Any]:
        """OPERATION 10 — Predict semantic classes that fit the current event and grammatical slot.

        Mathematics: produce a probability distribution over concepts such as food, tool, person,
        and location. Biological analogy: top-down context preactivates likely semantic-category
        populations before a word is recognized.
        """
        words = state["received_tokens"]
        expected_role = "instrument" if "with" in words[-3:] else (
            "patient" if any(item["pos"] == "VERB" for item in state["tagged_tokens"]) else "agent"
        )
        event = state.get("situation_hypothesis", "generic")
        if expected_role == "instrument" or (event == "cooking" and "cut" in words):
            categories = ["tool", "artifact", "physical"]
        elif expected_role == "patient" and event in {"cooking", "eating"}:
            categories = ["edible", "physical", "concrete"]
        elif expected_role == "agent":
            categories = ["animate", "human", "physical"]
        else:
            categories = ["physical", "abstract", "location"]
        distribution = _softmax({category: 1.0 if category in categories[:2] else 0.3 for category in categories}, 0.7)
        state["predicted_semantic_categories"] = distribution
        state["expected_thematic_role"] = expected_role
        return {"activation": max(distribution.values()), "expected_role": expected_role,
                "semantic_categories": distribution}

    def op_11_predict_thematic_roles(self, state: dict[str, Any]) -> dict[str, Any]:
        """OPERATION 11 — Predict agent, patient, instrument, and location roles for the developing event.

        Mathematics: combine grammatical position priors with feature-role compatibility and normalize
        role scores. Biological analogy: predicate and participant populations interact to bind event
        roles while leaving unfilled roles active.
        """
        tags = [item["pos"] for item in state["tagged_tokens"]]
        words = state["received_tokens"]
        if "with" in words[-3:]:
            expected = "instrument"
        elif "VERB" in tags:
            expected = "patient"
        elif tags and tags[-1] in {"NOUN", "PRON"}:
            expected = "predicate"
        else:
            expected = "agent"
        roles = {role: (0.82 if role == expected else 0.045) for role in _ROLE_NAMES}
        distribution = _softmax(roles, 0.7)
        existing: dict[str, str] = {}
        nouns = [item["word"] for item in state["tagged_tokens"] if item["pos"] in {"NOUN", "PRON"}]
        verbs = [item["word"] for item in state["tagged_tokens"] if item["pos"] == "VERB"]
        if nouns:
            existing["agent"] = nouns[0]
        if verbs:
            existing["predicate"] = verbs[-1]
        if len(nouns) > 1:
            existing["patient"] = nouns[-1]
        state["predicted_role_distribution"] = distribution
        state["predicted_role"] = expected
        state["thematic_roles"] = existing
        return {"activation": max(distribution.values()), "expected_role": expected,
                "role_probabilities": distribution, "existing_roles": existing}

    def op_12_predict_semantic_features(self, state: dict[str, Any]) -> dict[str, Any]:
        """OPERATION 12 — Construct an expected semantic feature vector from likely candidate concepts.

        Mathematics: calculate V_hat = sum_i p_i V_i using candidate probabilities as weights.
        Biological analogy: contextual top-down signals preactivate shared semantic features in
        distributed conceptual populations.
        """
        candidates = state.get("continuation_candidates", [])
        if not candidates:
            candidates = [
                {"word": word, "probability": 1.0}
                for word in self._candidate_vocabulary(state, ["NOUN"])[:5]
            ]
        vectors = [self._entry_for(item["word"], state)["features"] for item in candidates]
        weights = [float(item.get("probability", 0.0)) for item in candidates]
        expected = _mean_feature_vectors(vectors, weights)
        state["predicted_semantic_features"] = expected
        return {"activation": _clamp(sum(expected.values()) / max(1, len(expected))),
                "predicted_feature_count": len(expected), "predicted_features": expected}

    def op_13_retrieve_world_knowledge(self, state: dict[str, Any]) -> dict[str, Any]:
        """OPERATION 13 — Retrieve entity, action, causal, and functional world knowledge.

        Mathematics: activate knowledge entries as weighted associations from active concepts and
        retrieve caller-supplied facts when present. Biological analogy: semantic and episodic memory
        systems contribute learned facts and object-action relationships.
        """
        request: ComprehensionRequest = state["request"]
        retrieved: dict[str, dict[str, float]] = {}
        overrides = {_normalize_word(key): value for key, value in request.world_knowledge.items()}
        for word in state["received_tokens"][-10:]:
            if word in overrides:
                retrieved[word] = {key: float(value) for key, value in overrides[word].items()}
            else:
                entry = self._entry_for(word, state)
                retrieved[word] = {
                    key: float(value) for key, value in entry["features"].items() if value > 0.0
                }
        state["retrieved_world_knowledge"] = retrieved
        return {"activation": _clamp(len(retrieved) / 8.0),
                "retrieved_entities": sorted(retrieved)}

    def op_14_retrieve_event_schemas(self, state: dict[str, Any]) -> dict[str, Any]:
        """OPERATION 14 — Retrieve a familiar event script and estimate its current sequence position.

        Mathematics: use P(E_t+1|E_t,K) as a transition prior over event-schema steps.
        Biological analogy: distributed event knowledge and memory retrieval support predictions
        about typical sequences without requiring a fixed script to be followed.
        """
        event = state.get("situation_hypothesis", "generic")
        schema = _EVENT_SCHEMAS.get(event, _EVENT_SCHEMAS["generic"])
        completed = min(len(schema) - 1, len(state["received_tokens"]) // 3)
        state["event_schema"] = {"event": event, "steps": list(schema), "current_step": schema[completed]}
        return {"activation": _clamp(0.3 + len(schema) / 10.0),
                "event": event, "schema_steps": list(schema), "current_step": schema[completed]}

    def op_15_evaluate_plausibility(self, state: dict[str, Any]) -> dict[str, Any]:
        """OPERATION 15 — Estimate whether observed event participants fit ordinary world knowledge.

        Mathematics: score role-feature compatibility and combine it with an event prior P(E|K).
        Biological analogy: conceptual and event knowledge constrain interpretation, but linguistic
        context can support unusual or metaphorical events as well.
        """
        roles = state.get("thematic_roles", {})
        scores: list[float] = []
        event = state.get("situation_hypothesis", "generic")
        for role, word in roles.items():
            features = self._entry_for(word, state)["features"]
            if role == "agent":
                scores.append(max(features.get("animate", 0.0), features.get("human", 0.0), 0.25))
            elif role == "patient" and event in {"cooking", "eating"}:
                scores.append(max(features.get("edible", 0.0), 0.2))
            elif role == "instrument":
                scores.append(max(features.get("tool", 0.0), 0.15))
        compatibility = float(np.mean(scores)) if scores else 0.55
        request_facts = state["request"].world_knowledge
        supplied_support = 0.0
        if request_facts:
            supplied_support = _clamp(len(request_facts) / 5.0)
        plausibility = _clamp(0.85 * compatibility + 0.15 * supplied_support)
        state["world_plausibility"] = plausibility
        return {"activation": plausibility, "plausibility": plausibility,
                "role_compatibility_scores": scores}

    def op_16_estimate_world_reliability(self, state: dict[str, Any]) -> dict[str, Any]:
        """OPERATION 16 — Weight world-knowledge priors according to context reliability.

        Mathematics: adjust the world prior exponent beta when fictional or hypothetical cues appear.
        Biological analogy: precision modulation changes the influence of prior knowledge relative
        to the current linguistic evidence.
        """
        context = " ".join(
            list(state["discourse_history"]) + list(state["received_tokens"])
        ).casefold()
        fiction_cues = {"fiction", "fictional", "imagine", "pretend", "dream", "dragon", "story", "hypothetical"}
        fictional = any(cue in context for cue in fiction_cues)
        reliability = 0.35 if fictional else 0.85
        state["world_knowledge_reliability"] = reliability
        state["world_knowledge_is_fictional_context"] = fictional
        return {"activation": reliability, "reliability": reliability, "fictional_context": fictional}

    def op_17_recognize_word(self, state: dict[str, Any]) -> dict[str, Any]:
        """OPERATION 17 — Represent the incoming word and retrieve its lexical candidate.

        Mathematics: treat the supplied orthographic token as recognized evidence and attach a lexical
        confidence score. Biological analogy: visual or auditory input pathways activate lexical
        representations; this software receives already tokenized text.
        """
        word = state["current_word"]
        entry = self._entry_for(word, state)
        entry["recognition_confidence"] = 1.0 if entry["known_lexeme"] else 0.65
        entry["input_modality"] = "text"
        state["actual_lexical_entry"] = entry
        state["recognized_word"] = word
        return {"activation": entry["recognition_confidence"], "word": word,
                "pos": entry["pos"], "known_lexeme": entry["known_lexeme"]}

    def op_18_calculate_word_probability(self, state: dict[str, Any]) -> dict[str, Any]:
        """OPERATION 18 — Combine grammatical, semantic, world, and learned evidence for the actual word.

        Mathematics: add weighted evidence to candidate log scores and normalize with softmax.
        Biological analogy: bottom-up lexical activity interacts with top-down syntactic and conceptual
        expectations; the model keeps its candidate evidence inspectable.
        """
        actual = state["recognized_word"]
        expected_pos = state.get("expected_pos", [])
        candidates = self._candidate_vocabulary(state, expected_pos or None)
        if actual not in candidates:
            candidates.append(actual)
        expected_features = state.get("predicted_semantic_features", {})
        predicted_categories = state.get("predicted_semantic_categories", {})
        event = state.get("situation_hypothesis", "generic")
        prior_distribution = state.get("continuation_distribution", {})
        last = state["received_tokens"][-1] if state["received_tokens"] else "<START>"
        bigrams = state["learning_memory"]["bigrams"]
        scores: dict[str, float] = {}
        for candidate in candidates:
            entry = self._entry_for(candidate, state)
            syntax_score = 0.7 if not expected_pos or entry["pos"] in expected_pos else -0.8
            feature_similarity = _cosine_similarity(expected_features, entry["features"])
            category_support = max(
                (float(prob) for feature, prob in predicted_categories.items()
                 if entry["features"].get(feature, 0.0) > 0.0),
                default=0.0,
            )
            world_score = 0.35 if (
                event == "cooking" and (entry["features"].get("edible", 0.0) or entry["features"].get("tool", 0.0))
            ) else 0.0
            learned = math.log1p(float(bigrams.get(f"{last}|{candidate}", 0)))
            continuation = math.log(max(float(prior_distribution.get(candidate, 0.01)), _EPSILON))
            score = (
                self.config.syntax_weight * syntax_score
                + self.config.semantic_weight * feature_similarity
                + self.config.world_weight * (world_score + category_support * 0.2)
                + self.config.lexical_weight * (learned + continuation)
            )
            scores[candidate] = score
        distribution = _softmax(scores, self.config.prediction_temperature)
        probability = max(_EPSILON, float(distribution.get(actual, 0.0)))
        state["actual_word_distribution"] = distribution
        state["actual_word_probability"] = probability
        state["candidate_word_scores"] = scores
        return {"activation": probability, "actual_word": actual, "probability": probability,
                "candidate_count": len(distribution)}

    def op_19_calculate_surprisal(self, state: dict[str, Any]) -> dict[str, Any]:
        """OPERATION 19 — Calculate the information carried by the word under its pre-input probability.

        Mathematics: I(w_t) = -log2(P(w_t|context)); a probability floor prevents undefined values.
        Biological analogy: neural systems can represent graded mismatch and expectation strength, but
        an individual neuron is not assumed to calculate a logarithm.
        """
        probability = max(_EPSILON, float(state.get("actual_word_probability", _EPSILON)))
        surprisal = -math.log2(probability)
        state["word_surprisal_bits"] = surprisal
        return {"activation": _clamp(surprisal / 8.0), "surprisal_bits": surprisal,
                "actual_word_probability": probability}

    def op_20_calculate_uncertainty(self, state: dict[str, Any]) -> dict[str, Any]:
        """OPERATION 20 — Quantify pre-input uncertainty across all candidate words.

        Mathematics: H(W|C) = -sum_w p(w|C) log2 p(w|C), distinct from the surprisal of the actual word.
        Biological analogy: the spread of activity across competing lexical representations is a proxy
        for uncertainty before a specific word is received.
        """
        probabilities = state.get("actual_word_distribution", {})
        uncertainty = _entropy(probabilities)
        state["prediction_entropy_bits"] = uncertainty
        return {"activation": _clamp(uncertainty / 6.0), "entropy_bits": uncertainty,
                "candidate_count": len(probabilities)}

    def op_21_retrieve_predicted_features(self, state: dict[str, Any]) -> dict[str, Any]:
        """OPERATION 21 — Retrieve the semantic feature vector that was active before the word arrived.

        Mathematics: copy the expected vector V_hat formed as a probability-weighted sum of concepts.
        Biological analogy: contextual top-down signals preactivate distributed semantic features.
        """
        expected = dict(state.get("predicted_semantic_features", {}))
        state["expected_features_for_comparison"] = expected
        return {"activation": _clamp(sum(expected.values()) / max(1, len(expected))),
                "expected_feature_count": len(expected)}

    def op_22_extract_actual_features(self, state: dict[str, Any]) -> dict[str, Any]:
        """OPERATION 22 — Retrieve semantic features associated with the recognized lexical item.

        Mathematics: represent the word as a feature vector V_t = [f_1,...,f_n] using caller evidence
        where available. Biological analogy: lexical activation spreads to associated conceptual
        features distributed across semantic-memory networks.
        """
        actual = dict(state["actual_lexical_entry"]["features"])
        state["actual_semantic_features"] = actual
        return {"activation": _clamp(sum(actual.values()) / max(1, len(actual))),
                "actual_feature_count": len(actual), "actual_features": actual}

    def op_23_calculate_semantic_error(self, state: dict[str, Any]) -> dict[str, Any]:
        """OPERATION 23 — Calculate residual and precision-weighted lexical-semantic prediction error.

        Mathematics: r = V_actual - V_expected and E = r^T Pi r / n, with unit feature precision
        unless the caller supplies richer reliability estimates. Biological analogy: predictive-coding
        accounts describe residual activity when actual input differs from top-down expectations.
        """
        expected = state.get("expected_features_for_comparison", {})
        actual = state.get("actual_semantic_features", {})
        dimensions = sorted(set(expected) | set(actual))
        residual = {key: float(actual.get(key, 0.0) - expected.get(key, 0.0)) for key in dimensions}
        error = float(np.mean([value * value for value in residual.values()])) if residual else 0.0
        state["semantic_residual"] = residual
        state["semantic_prediction_error"] = error
        return {"activation": _clamp(math.sqrt(error)), "weighted_error": error,
                "residual_norm": float(math.sqrt(sum(value * value for value in residual.values())))}

    def op_24_estimate_n400(self, state: dict[str, Any]) -> dict[str, Any]:
        """OPERATION 24 — Estimate a negative-going N400-like response from semantic mismatch.

        Mathematics: N400_hat = -gain * mismatch_score, where mismatch combines feature error and
        normalized surprisal. Biological analogy: N400 is an EEG component related to lexical-semantic
        processing; this scalar is a model estimate, not an EEG waveform or direct neural readout.
        """
        mismatch = _clamp(
            0.72 * math.sqrt(max(0.0, state.get("semantic_prediction_error", 0.0)))
            + 0.28 * min(1.0, state.get("word_surprisal_bits", 0.0) / 8.0)
        )
        amplitude = -self.config.n400_gain_uv * mismatch
        state["n400_mismatch_score"] = mismatch
        state["n400_estimate_uv"] = amplitude
        return {"activation": mismatch, "mismatch_score": mismatch,
                "n400_estimate_uv": amplitude}

    def op_25_compare_interpretations(self, state: dict[str, Any]) -> dict[str, Any]:
        """OPERATION 25 — Compare syntactic role expectations with semantic evidence for the actual word.

        Mathematics: estimate Jensen-Shannon divergence between role distributions and add agreement
        mismatch. Biological analogy: competing language representations interact with monitoring
        networks; disagreement increases demand for further evaluation.
        """
        syntax = dict(state.get("predicted_role_distribution", {}))
        features = state.get("actual_semantic_features", {})
        semantic_scores = {
            "agent": max(features.get("animate", 0.0), features.get("human", 0.0), 0.02),
            "patient": max(features.get("edible", 0.0), features.get("physical", 0.0) * 0.25, 0.02),
            "instrument": max(features.get("tool", 0.0), features.get("artifact", 0.0) * 0.35, 0.02),
            "location": max(features.get("location", 0.0), 0.02),
            "predicate": max(features.get("action", 0.0), 0.02),
            "other": 0.05,
        }
        semantic = _softmax(semantic_scores, 0.8)
        divergence = _js_divergence(syntax, semantic)
        agreement = state.get("agreement_expectation", {})
        actual_entry = state.get("actual_lexical_entry", {})
        agreement_violation = 0.0
        expected_number = agreement.get("expected_number")
        if actual_entry.get("pos") in {"VERB", "AUX"} and expected_number:
            actual_number = actual_entry.get("number")
            if actual_number and actual_number != expected_number:
                agreement_violation = 1.0
        state["semantic_role_distribution"] = semantic
        state["syntactic_semantic_divergence"] = divergence
        state["agreement_violation"] = agreement_violation
        conflict = _clamp(0.75 * divergence + 0.25 * agreement_violation)
        state["representational_conflict"] = conflict
        return {"activation": conflict, "js_divergence_bits": divergence,
                "agreement_violation": agreement_violation, "conflict": conflict}

    def op_26_detect_reanalysis(self, state: dict[str, Any]) -> dict[str, Any]:
        """OPERATION 26 — Decide whether structural conflict warrants an alternative analysis.

        Mathematics: combine syntax-semantics divergence, agreement violation, and parse uncertainty,
        then compare with a configurable threshold. Biological analogy: conflict-monitoring and
        attentional-control systems may allocate more processing to a difficult sentence region.
        """
        conflict = float(state.get("representational_conflict", 0.0))
        agreement = float(state.get("agreement_violation", 0.0))
        ambiguity = _clamp(float(state.get("parse_uncertainty", 0.0)) / 2.0)
        score = _clamp(0.72 * conflict + 0.18 * agreement + 0.10 * ambiguity)
        required = bool(score >= self.config.conflict_threshold)
        state["reanalysis_score"] = score
        state["reanalysis_required"] = required
        return {"activation": score, "reanalysis_score": score, "reanalysis_required": required}

    def op_27_reconstruct_alternatives(self, state: dict[str, Any]) -> dict[str, Any]:
        """OPERATION 27 — Regenerate and rescore alternative parses when current evidence conflicts.

        Mathematics: update parse scores with incoming-word compatibility and normalize via softmax.
        Biological analogy: recurrent language-network interactions can revise earlier attachments
        and keep competing structural hypotheses available.
        """
        existing = state["syntactic_structure"].get("parse_candidates", [])
        actual_pos = state.get("actual_lexical_entry", {}).get("pos", "UNKNOWN")
        adjusted: dict[str, float] = {}
        for item in existing or [{"parse": "incremental_preferred", "probability": 1.0}]:
            score = math.log(max(float(item.get("probability", 0.01)), _EPSILON))
            if actual_pos in {"NOUN", "VERB"}:
                score += 0.18
            if item["parse"] == "reanalysis_candidate" and state.get("reanalysis_required"):
                score += 0.75
            adjusted[item["parse"]] = score
        adjusted.setdefault("late_attachment", -0.5)
        probabilities = _softmax(adjusted, 0.75)
        alternatives = [
            {"parse": parse, "probability": probability}
            for parse, probability in sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
        ][:self.config.max_hypotheses]
        if state.get("reanalysis_required"):
            state["syntactic_structure"]["parse_candidates"] = alternatives
        state["alternative_parse_candidates"] = alternatives
        return {"activation": _clamp(state.get("reanalysis_score", 0.0)),
                "reconstructed_alternatives": alternatives}

    def op_28_estimate_p600(self, state: dict[str, Any]) -> dict[str, Any]:
        """OPERATION 28 — Estimate positive-going P600-like activity from conflict and revision demand.

        Mathematics: P600_hat = gain * (conflict + revision_cost), bounded to the configured scale.
        Biological analogy: P600 is an EEG component often associated with reanalysis and monitoring,
        though its functional interpretation is debated; this is not a direct physiological measure.
        """
        conflict = _clamp(float(state.get("representational_conflict", 0.0)))
        revision = 0.35 if state.get("reanalysis_required") else 0.0
        amplitude = self.config.p600_gain_uv * min(1.0, conflict + revision)
        state["p600_estimate_uv"] = amplitude
        return {"activation": _clamp((conflict + revision) / 1.35),
                "revision_cost": revision, "p600_estimate_uv": amplitude}

    def op_29_integrate_syntax(self, state: dict[str, Any]) -> dict[str, Any]:
        """OPERATION 29 — Attach the actual word to the evolving grammatical representation.

        Mathematics: update the parse posterior P(G_t|W_1:t) and preserve unresolved dependencies.
        Biological analogy: distributed neural activity updates structural representations incrementally
        as each recognized word is incorporated.
        """
        word = state["recognized_word"]
        entry = state["actual_lexical_entry"]
        if word and entry["pos"] not in {"PUNCT"}:
            state["tagged_tokens"].append({
                "word": word, "surface": state["current_surface"], "pos": entry["pos"],
                "number": entry.get("number", "singular"),
            })
        tags = [item["pos"] for item in state["tagged_tokens"]]
        state["syntactic_structure"]["current_tags"] = tags
        state["syntactic_structure"]["posterior_confidence"] = _clamp(
            max(state.get("syntactic_category_prediction", {"NOUN": 0.5}).values())
        )
        return {"activation": _clamp(len(tags) / 10.0),
                "integrated_category": entry["pos"], "syntactic_token_count": len(tags)}

    def op_30_integrate_semantics(self, state: dict[str, Any]) -> dict[str, Any]:
        """OPERATION 30 — Integrate the actual word's meaning into the event and thematic-role state.

        Mathematics: update M_t = F(M_t-1,V_t,G_t) by adding actual features and assigning its
        context-predicted role. Biological analogy: semantic and combinatorial networks bind lexical
        meaning to an evolving event representation.
        """
        word = state["recognized_word"]
        features = dict(state.get("actual_semantic_features", {}))
        role = state.get("expected_thematic_role", "other")
        if any(float(value) > 0.0 for value in features.values()):
            state["semantic_concepts"].append({"word": word, "features": features, "role": role})
            if role in {"agent", "patient", "instrument", "location", "predicate"}:
                state["thematic_roles"][role] = word
        state["integrated_semantic_features"] = _mean_feature_vectors(
            [item["features"] for item in state["semantic_concepts"]],
            [1.0] * len(state["semantic_concepts"]),
        )
        return {"activation": _clamp(len(state["semantic_concepts"]) / 10.0),
                "integrated_concept_count": len(state["semantic_concepts"]),
                "assigned_role": role}

    def op_31_update_hypotheses(self, state: dict[str, Any]) -> dict[str, Any]:
        """OPERATION 31 — Apply Bayesian-style updating to competing sentence-level event hypotheses.

        Mathematics: posterior(H_i) is proportional to likelihood(word|H_i) times prior(H_i).
        Biological analogy: recurrent competition changes the relative activation of alternative
        sentence interpretations as new evidence arrives.
        """
        priors = state.get("sentence_hypotheses", {})
        word = state["recognized_word"]
        scores: dict[str, float] = {}
        for event, item in priors.items():
            prior = max(float(item.get("probability", 0.0)), _EPSILON)
            cue = _EVENT_CUES.get(event, set())
            likelihood = 1.6 if word in cue else 0.55
            scores[event] = math.log(prior) + math.log(likelihood)
        posterior = _softmax(scores, 1.0)
        state["sentence_hypotheses"] = {
            event: {"probability": probability, "score": scores[event]}
            for event, probability in posterior.items()
        }
        state["situation_hypothesis"] = max(posterior, key=posterior.get) if posterior else "generic"
        return {"activation": max(posterior.values()) if posterior else 0.0,
                "posterior_hypotheses": state["sentence_hypotheses"]}

    def op_32_reconcile_predictions(self, state: dict[str, Any]) -> dict[str, Any]:
        """OPERATION 32 — Reconcile expected and observed semantic features using precision-weighted error.

        Mathematics: epsilon = V_actual - V_expected; V_updated = V_expected + eta*Pi*epsilon.
        Biological analogy: bidirectional neural signaling allows sensory evidence to modify current
        representations while top-down predictions continue to shape subsequent processing.
        """
        expected = state.get("expected_features_for_comparison", {})
        actual = state.get("actual_semantic_features", {})
        residual = state.get("semantic_residual", {})
        precision = 1.0
        rate = 0.35
        updated = {
            key: float(expected.get(key, 0.0)) + rate * precision * float(residual.get(key, 0.0))
            for key in sorted(set(expected) | set(actual))
        }
        state["reconciled_semantic_representation"] = updated
        return {"activation": _clamp(sum(updated.values()) / max(1, len(updated))),
                "updated_feature_count": len(updated), "precision": precision, "integration_rate": rate}

    def op_33_maintain_or_revise(self, state: dict[str, Any]) -> dict[str, Any]:
        """OPERATION 33 — Maintain a coherent interpretation or mark it for revision.

        Mathematics: combine syntax, semantic fit, world plausibility, and conflict into a hypothesis
        score; preserve uncertainty instead of forcing a winner. Biological analogy: monitoring and
        cognitive-control systems regulate further analysis when competing interpretations persist.
        """
        parses = state["syntactic_structure"].get("parse_candidates", [])
        parse_confidence = float(parses[0]["probability"]) if parses else 0.5
        world = float(state.get("world_plausibility", 0.5))
        conflict = float(state.get("representational_conflict", 0.0))
        score = _clamp(0.4 * parse_confidence + 0.35 * world + 0.25 * (1.0 - conflict))
        revised = bool(state.get("reanalysis_required") and score < 0.7)
        state["interpretation_confidence"] = score
        state["interpretation_status"] = "revise" if revised else (
            "provisional" if score < 0.55 else "maintain"
        )
        return {"activation": score, "confidence": score, "decision": state["interpretation_status"]}

    def op_34_update_working_context(self, state: dict[str, Any]) -> dict[str, Any]:
        """OPERATION 34 — Append the recognized word and update recency-weighted working context.

        Mathematics: decay prior activations exponentially and add the current item as fresh input.
        Biological analogy: recurrent activity and selective attention maintain task-relevant words
        and unresolved dependencies in an evolving working-memory representation.
        """
        word = state["recognized_word"]
        if word and word not in {".", "?", "!", ",", ";", ":"}:
            state["received_tokens"].append(word)
        context = state["received_tokens"][-12:]
        activation = {
            item: self.config.context_decay ** distance
            for distance, item in enumerate(reversed(context))
        }
        state["working_context_activation"] = activation
        return {"activation": _clamp(len(context) / 12.0),
                "working_context_size": len(context), "active_items": len(activation)}

    def op_35_predict_next_word(self, state: dict[str, Any]) -> dict[str, Any]:
        """OPERATION 35 — Generate the next-word distribution from the newly reconciled context.

        Mathematics: calculate P(W_t+1|W_1:t) by combining category, feature, and learned association
        scores then applying softmax. Biological analogy: updated high-level representations feed back
        to lower-level lexical populations as fresh expectations.
        """
        expected_pos = self._expected_pos(state)
        candidates = self._candidate_vocabulary(state, expected_pos or None)
        last = state["received_tokens"][-1] if state["received_tokens"] else "<START>"
        bigrams = state["learning_memory"]["bigrams"]
        expected_features = state.get("reconciled_semantic_representation", {})
        scores = {}
        for word in candidates:
            entry = self._entry_for(word, state)
            score = 0.7 if entry["pos"] in expected_pos else 0.0
            score += 0.8 * _cosine_similarity(expected_features, entry["features"])
            score += 0.25 * math.log1p(float(bigrams.get(f"{last}|{word}", 0)))
            scores[word] = score
        distribution = _softmax(scores, self.config.prediction_temperature)
        state["next_word_distribution"] = distribution
        return {"activation": _clamp(max(distribution.values()) if distribution else 0.0),
                "predicted_next_words": sorted(distribution.items(), key=lambda item: item[1], reverse=True)[:8]}

    def op_36_detect_completion(self, state: dict[str, Any]) -> dict[str, Any]:
        """OPERATION 36 — Detect a sentence boundary using punctuation, input boundary, and structure.

        Mathematics: estimate P(complete|S_t) from boundary cues and unresolved dependencies. Biological
        analogy: prosodic, syntactic, semantic, and discourse signals jointly contribute to boundary
        recognition; punctuation or end-of-input is treated as an explicit software cue.
        """
        explicit_boundary = bool(state.get("has_sentence_boundary"))
        input_end = bool(state.get("is_last_input"))
        unresolved = state["syntactic_structure"].get("dependencies", [])
        structural_score = 0.85 if not unresolved else 0.58
        probability = 0.98 if explicit_boundary else (
            structural_score if input_end else 0.12
        )
        complete = bool((explicit_boundary or input_end) and probability >= self.config.completion_threshold)
        state["sentence_completion_probability"] = probability
        state["sentence_complete"] = complete
        return {"activation": probability, "completion_probability": probability,
                "sentence_complete": complete, "explicit_boundary": explicit_boundary}

    def op_37_consolidate_interpretation(self, state: dict[str, Any]) -> dict[str, Any]:
        """OPERATION 37 — Consolidate completed syntax, meaning, roles, and world knowledge into one result.

        Mathematics: construct R_sentence = F(G_final,M_final,K,D) and report confidence and plausibility.
        Biological analogy: distributed language and conceptual networks support an integrated
        sentence-level representation that can be used by subsequent discourse processing.
        """
        interpretation = {
            "text": " ".join(state["received_tokens"]),
            "tokens": list(state["received_tokens"]),
            "syntactic_categories": list(state["syntactic_structure"].get("current_tags", [])),
            "thematic_roles": dict(state.get("thematic_roles", {})),
            "event_hypothesis": state.get("situation_hypothesis", "generic"),
            "event_hypotheses": state.get("sentence_hypotheses", {}),
            "world_plausibility": float(state.get("world_plausibility", 0.5)),
            "confidence": float(state.get("interpretation_confidence", 0.5)),
            "status": state.get("interpretation_status", "provisional"),
            "sentence_complete": bool(state.get("sentence_complete", False)),
        }
        state["current_sentence_interpretation"] = interpretation
        if state["sentence_complete"]:
            state["sentence_interpretations"].append(interpretation)
        return {"activation": interpretation["confidence"],
                "sentence_complete": interpretation["sentence_complete"],
                "event_hypothesis": interpretation["event_hypothesis"]}

    def op_38_update_long_term_knowledge(self, state: dict[str, Any]) -> dict[str, Any]:
        """OPERATION 38 — Update long-term predictive associations from the observed word sequence.

        Mathematics: strengthen a word-pair association by Delta W = eta*A_i*A_j and update event
        counts using an incremental learning rate. Biological analogy: experience-dependent synaptic
        plasticity changes future expectations; the software update is an abstract statistical analogue.
        """
        memory = state["learning_memory"]
        word = state["recognized_word"]
        previous = state["received_tokens"][-2] if len(state["received_tokens"]) >= 2 else "<START>"
        key = f"{previous}|{word}"
        memory["bigrams"][key] += 1
        event = state.get("situation_hypothesis", "generic")
        memory["event_counts"][event] += 1
        old_weight = float(memory["association_weights"][key])
        memory["association_weights"][key] = old_weight + self.config.learning_rate * (
            1.0 - old_weight
        )
        memory["learning_updates"] += 1
        memory["processed_sequences"] = max(1, int(memory["processed_sequences"]))
        state["learning_memory"] = memory
        return {"activation": _clamp(memory["association_weights"][key]),
                "association": key, "association_weight": memory["association_weights"][key],
                "learning_update_count": int(memory["learning_updates"])}
