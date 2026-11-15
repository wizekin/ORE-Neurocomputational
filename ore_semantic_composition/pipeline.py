"""Executable nine-stage semantic composition from word concepts to sentence meaning.

The functional pipeline preserves evidence, competing readings, provenance and
abstentions. Its optional LIF track is a separate computational reference, not
a measured or validated reproduction of human semantic processing.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, is_dataclass
from difflib import SequenceMatcher
from itertools import permutations
from typing import Any, Iterable, Mapping, Sequence
import hashlib
import json
import math
import re
import unicodedata

import numpy as np

from .biology import SemanticCompositionLIFSimulator
from .contracts import (
    PredicateFrame,
    SemanticCompositionConfig,
    SemanticCompositionRequest,
    SemanticCompositionResult,
)
from .registry import OPERATION_REGISTRY, STAGE_REGISTRY

_EPS = 1e-9

_NEGATION = {"not", "n't", "never", "no", "neither", "nor", "nowhere", "nothing"}
_MODAL_FORCE = {
    "may": ("possibility", 0.45), "might": ("possibility", 0.35),
    "can": ("ability_or_possibility", 0.55), "could": ("ability_or_possibility", 0.45),
    "must": ("necessity_or_obligation", 0.92), "should": ("expectation_or_advice", 0.72),
    "shall": ("obligation_or_prediction", 0.78), "will": ("prediction_or_volition", 0.68),
    "would": ("conditional_or_volitional", 0.58), "ought": ("obligation", 0.82),
}
_QUANTIFIERS = {
    "all": ("universal", "∀"), "every": ("universal", "∀"),
    "each": ("universal", "∀"), "both": ("universal_pair", "∀"),
    "some": ("existential", "∃"), "any": ("existential_or_free_choice", "∃"),
    "a": ("indefinite_existential", "∃"), "an": ("indefinite_existential", "∃"),
    "no": ("negative_existential", "¬∃"), "none": ("negative_existential", "¬∃"),
    "most": ("proportional", "MOST"), "many": ("proportional", "MANY"),
    "few": ("proportional", "FEW"), "several": ("proportional", "SEVERAL"),
    "one": ("exact_count", "=1"), "two": ("exact_count", "=2"),
    "three": ("exact_count", "=3"), "four": ("exact_count", "=4"),
    "five": ("exact_count", "=5"),
}
_CAUSAL_CUES = {
    "because": ("because", 0.92), "therefore": ("therefore", 0.92),
    "thus": ("therefore", 0.88), "hence": ("therefore", 0.88),
    "so": ("so", 0.72), "since": ("since", 0.62),
    "consequently": ("therefore", 0.92), "thereby": ("thereby", 0.78),
}
_ROLE_BY_DEPENDENCY = {
    "nsubj": "Agent", "nsubj:pass": "Patient", "nsubjpass": "Patient",
    "csubj": "Agent", "obj": "Patient", "dobj": "Patient",
    "iobj": "Recipient", "obl:agent": "Agent", "agent": "Agent",
    "obl:to": "Goal", "obl:from": "Source", "obl:with": "Instrument",
    "attr": "Attribute", "acomp": "Attribute", "xcomp": "Theme",
    "ccomp": "Proposition",
}
_PREDICATE_CATEGORIES = {"verb", "predicate", "adjective", "auxiliary", "preposition"}
_NOUN_CATEGORIES = {"noun", "proper_noun", "pronoun", "nominal", "name"}


def _get(value: Any, key: str, default: Any = None) -> Any:
    """Read a field from either a mapping or a typed object without mutating it."""
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _normalize(value: str) -> str:
    """Normalize a token for lexical lookup while retaining the original surface elsewhere."""
    normalized = unicodedata.normalize("NFKC", value).casefold().strip()
    return re.sub(r"^\W+|\W+$", "", normalized, flags=re.UNICODE)


def _tokenize(text: str) -> list[str]:
    """Split text into word-like tokens and punctuation while preserving token order."""
    return re.findall(r"[\w]+(?:['’][\w]+)*|[^\w\s]", text, flags=re.UNICODE)


def _cosine(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    """Compute cosine similarity over the union of sparse semantic feature dimensions."""
    keys = sorted(set(left) | set(right))
    if not keys:
        return 0.0
    a = np.asarray([float(left.get(key, 0.0)) for key in keys], dtype=np.float64)
    b = np.asarray([float(right.get(key, 0.0)) for key in keys], dtype=np.float64)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= _EPS:
        return 0.0
    return float(np.clip(np.dot(a, b) / denominator, -1.0, 1.0))


def _softmax(values: Sequence[float], temperature: float) -> list[float]:
    """Convert finite candidate scores to a numerically stable probability distribution."""
    if not values:
        return []
    vector = np.asarray(values, dtype=np.float64) / max(float(temperature), _EPS)
    vector = vector - np.max(vector)
    weights = np.exp(np.clip(vector, -60.0, 60.0))
    total = float(np.sum(weights))
    if total <= _EPS or not math.isfinite(total):
        return [1.0 / len(values)] * len(values)
    return [float(value / total) for value in weights]


def _entropy(probabilities: Sequence[float]) -> float:
    """Calculate Shannon entropy in nats for a normalized or unnormalized non-negative vector."""
    if not probabilities:
        return 0.0
    values = np.asarray(probabilities, dtype=np.float64)
    total = float(values.sum())
    if total <= _EPS:
        return 0.0
    values = values / total
    return float(-np.sum(values * np.log(np.maximum(values, _EPS))))


def _jsonable(value: Any) -> Any:
    """Convert nested values into JSON-safe Python primitives."""
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    return repr(value)


def _feature_map(concept: Any) -> dict[str, float]:
    """Extract and bound a concept's sparse semantic feature map."""
    source = _get(concept, "semantic_features", {}) or {}
    result: dict[str, float] = {}
    for key, raw in source.items():
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            result[str(key)] = float(np.clip(value, 0.0, 1.0))
    return result


class SemanticCompositionProcessor:
    """Compose word concepts into predicates, roles, propositions, operators, and a sentence graph.

    The processor is read-only: it never learns from a recognition request or
    mutates the concept inventory. Supply reviewed concept entries and, where
    possible, explicit predicate frames. If syntax is absent, bounded local
    fallbacks are marked as such instead of being represented as parsed facts.
    """

    def __init__(
        self,
        concepts: Iterable[Any] = (),
        predicate_frames: Iterable[PredicateFrame] = (),
        config: SemanticCompositionConfig | None = None,
    ) -> None:
        """Build immutable lookup indexes for concept aliases and predicate valency frames."""
        self.config = config or SemanticCompositionConfig()
        self.concepts = tuple(concepts)
        self.frames = tuple(predicate_frames)
        self._concept_by_id: dict[str, Any] = {}
        self._alias_index: dict[str, list[Any]] = defaultdict(list)
        for concept in self.concepts:
            concept_id = str(_get(concept, "concept_id", "") or "")
            label = str(_get(concept, "label", "") or "")
            if not concept_id or not label:
                raise ValueError("each concept must provide non-empty concept_id and label")
            if concept_id in self._concept_by_id:
                raise ValueError(f"duplicate concept_id: {concept_id}")
            self._concept_by_id[concept_id] = concept
            aliases = (label, *tuple(_get(concept, "aliases", ()) or ()))
            for alias in aliases:
                normalized = _normalize(str(alias))
                if normalized:
                    self._alias_index[normalized].append(concept)
        self._frames_by_lemma: dict[str, PredicateFrame] = {}
        self._frames_by_concept: dict[str, PredicateFrame] = {}
        for frame in self.frames:
            if not isinstance(frame, PredicateFrame):
                raise TypeError("predicate_frames must contain PredicateFrame records")
            self._frames_by_lemma[_normalize(frame.predicate)] = frame
            if frame.predicate_concept_id:
                self._frames_by_concept[frame.predicate_concept_id] = frame
        self.model_fingerprint = self._fingerprint_inventory()

    def _fingerprint_inventory(self) -> str:
        """Create a stable lightweight inventory fingerprint for result provenance."""
        rows = []
        for concept_id in sorted(self._concept_by_id):
            concept = self._concept_by_id[concept_id]
            rows.append((
                concept_id,
                str(_get(concept, "label", "")),
                tuple(_get(concept, "aliases", ()) or ()),
                tuple(sorted(_feature_map(concept).items())),
            ))
        return hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()

    def _syntax_payload(self, syntax_result: Any) -> dict[str, Any]:
        """Normalize an optional SyntaxResult object or mapping to a read-only dictionary."""
        if syntax_result is None:
            return {}
        if isinstance(syntax_result, Mapping):
            return dict(syntax_result)
        if hasattr(syntax_result, "to_dict"):
            payload = syntax_result.to_dict()
            return dict(payload) if isinstance(payload, Mapping) else {}
        return {
            key: _get(syntax_result, key)
            for key in (
                "tokens", "dependency_arcs", "selected_pos", "predicate_arguments",
                "thematic_roles", "constituency_tree", "diagnostics",
            )
            if _get(syntax_result, key) is not None
        }

    def _record(
        self,
        operation_index: int,
        output: Mapping[str, Any],
        confidence: float = 1.0,
        evidence: Mapping[str, Any] | None = None,
    ) -> None:
        """Append one registry-linked operation row and advance its biological-reference population."""
        spec = OPERATION_REGISTRY[operation_index - 1]
        biological = None
        if self._simulator is not None:
            biological = self._simulator.advance(spec.stage_id, confidence)
            self._biology_operation_states.append({
                "operation_id": spec.operation_id,
                "population": spec.stage_id,
                **biological,
            })
        self._trace.append({
            "operation_id": spec.operation_id,
            "stage_id": spec.stage_id,
            "stage_name": spec.stage_name,
            "operation": spec.name,
            "description": spec.description,
            "mathematical_form": spec.mathematical_form,
            "biological_reference": spec.biological_reference,
            "evidence": _jsonable(evidence or {}),
            "output": _jsonable(output),
            "confidence": float(np.clip(confidence, 0.0, 1.0)),
            "biological_state": biological,
            "status": "completed",
        })

    def _stage_one_retrieve_concepts(
        self, request: SemanticCompositionRequest, tokens: list[str], syntax: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        """Perform the six concept-retrieval operations for every word token."""
        normalized = [_normalize(token) for token in tokens]
        self._record(1, {"token_count": len(tokens), "normalized_forms": normalized[:30]})
        hypotheses = request.concept_hypotheses
        lexical_rows: list[list[dict[str, Any]]] = []
        max_frequency = max(
            [float(_get(item, "frequency", 1.0) or 1.0) for item in self.concepts] or [1.0]
        )
        for index, form in enumerate(normalized):
            candidates: dict[str, dict[str, Any]] = {}
            for concept in self._alias_index.get(form, ()):
                concept_id = str(_get(concept, "concept_id"))
                candidates[concept_id] = {
                    "concept": concept, "concept_id": concept_id,
                    "label": str(_get(concept, "label", form)),
                    "lexical_similarity": 1.0, "exact_match": True,
                }
            if not candidates and form:
                for concept in self.concepts:
                    concept_id = str(_get(concept, "concept_id"))
                    forms = (
                        str(_get(concept, "label", "")),
                        *(str(alias) for alias in (_get(concept, "aliases", ()) or ())),
                    )
                    similarity = max(
                        (SequenceMatcher(None, form, _normalize(alias)).ratio() for alias in forms),
                        default=0.0,
                    )
                    if similarity >= self.config.lexical_similarity_floor:
                        candidates[concept_id] = {
                            "concept": concept, "concept_id": concept_id,
                            "label": str(_get(concept, "label", form)),
                            "lexical_similarity": float(similarity), "exact_match": False,
                        }
            lexical_rows.append(list(candidates.values()))
        self._record(2, {
            "candidate_counts": [len(items) for items in lexical_rows],
            "exact_matches": sum(
                1 for items in lexical_rows for item in items if item["exact_match"]
            ),
        })
        constrained: list[list[dict[str, Any]]] = []
        constraint_warnings: list[str] = []
        for index, items in enumerate(lexical_rows):
            allowed = set(hypotheses.get(index, ()))
            if allowed:
                filtered = [item for item in items if item["concept_id"] in allowed]
                if filtered:
                    items = filtered
                else:
                    constraint_warnings.append(
                        f"Token {index}: supplied concept hypotheses did not match inventory candidates."
                    )
            constrained.append(items[: self.config.max_concepts_per_token])
        self._record(3, {
            "constrained_counts": [len(items) for items in constrained],
            "constraint_warnings": constraint_warnings,
        })
        candidates_with_features: list[list[dict[str, Any]]] = []
        for items in constrained:
            enriched = []
            for item in items:
                concept = item["concept"]
                enriched.append({
                    **item,
                    "features": _feature_map(concept),
                    "lexical_category": str(_get(concept, "lexical_category", "unknown")),
                    "frequency": float(_get(concept, "frequency", 1.0) or 1.0),
                    "source_reference": str(_get(concept, "source_reference", "unspecified")),
                })
            candidates_with_features.append(enriched)
        self._record(4, {
            "feature_dimensions": [
                sorted({key for item in items for key in item["features"]})[:30]
                for items in candidates_with_features
            ],
        })
        context_fit: list[list[float]] = []
        for items in candidates_with_features:
            context_fit.append([
                _cosine(item["features"], request.context_features) if request.context_features else 0.0
                for item in items
            ])
        self._record(5, {
            "context_evidence_present": bool(request.context_features),
            "context_fit": [[round(value, 4) for value in row] for row in context_fit],
        })
        word_records: list[dict[str, Any]] = []
        for index, items in enumerate(candidates_with_features):
            scores = []
            for candidate_index, item in enumerate(items):
                lexical = item["lexical_similarity"]
                context = max(0.0, context_fit[index][candidate_index])
                frequency = math.log1p(max(0.0, item["frequency"])) / math.log1p(max_frequency)
                score = (
                    self.config.lexical_weight * lexical
                    + self.config.context_weight * context
                    + self.config.frequency_weight * frequency
                )
                scores.append(score)
            probabilities = _softmax(scores, self.config.softmax_temperature)
            ranked = []
            for item, score, probability in sorted(
                zip(items, scores, probabilities),
                key=lambda row: row[2],
                reverse=True,
            ):
                ranked.append({
                    "concept_id": item["concept_id"],
                    "label": item["label"],
                    "lexical_similarity": float(item["lexical_similarity"]),
                    "context_fit": float(context_fit[index][items.index(item)]),
                    "frequency": float(item["frequency"]),
                    "score": float(score),
                    "probability": float(probability),
                    "features": item["features"],
                    "lexical_category": item["lexical_category"],
                    "source_reference": item["source_reference"],
                    "exact_match": item["exact_match"],
                })
            top = ranked[0] if ranked else None
            second = ranked[1]["probability"] if len(ranked) > 1 else 0.0
            posterior_confidence = float(top["probability"]) if top else 0.0
            confidence = (
                posterior_confidence * float(top["lexical_similarity"])
                if top else 0.0
            )
            margin = posterior_confidence - second
            selected = (
                top if top and confidence >= self.config.confidence_threshold
                and margin >= self.config.ambiguity_margin else None
            )
            word_records.append({
                "token_index": index,
                "token": tokens[index],
                "normalized_form": normalized[index],
                "candidates": ranked,
                "selected_concept_id": selected["concept_id"] if selected else None,
                "selected_label": selected["label"] if selected else None,
                "selected_features": selected["features"] if selected else {},
                "selected_lexical_category": selected["lexical_category"] if selected else "unknown",
                "confidence": confidence,
                "margin": margin,
                "ambiguous": bool(top and selected is None),
                "unresolved": not bool(ranked),
            })
        self._record(6, {
            "tokens_with_candidates": sum(bool(row["candidates"]) for row in word_records),
            "tokens_selected": sum(bool(row["selected_concept_id"]) for row in word_records),
            "tokens_abstained": sum(not bool(row["selected_concept_id"]) for row in word_records),
        }, confidence=float(np.mean([row["confidence"] for row in word_records]) if word_records else 0.0))
        self._diagnostics["concept_constraint_warnings"].extend(constraint_warnings)
        return word_records

    def _stage_two_combine_meanings(
        self, request: SemanticCompositionRequest, tokens: list[str],
        word_concepts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Bind individual word meanings into an ordered, role-preserving composition workspace."""
        workspace = [{"token_index": i, "token": token, "node_id": f"token:{i}"} for i, token in enumerate(tokens)]
        self._record(7, {"workspace_nodes": len(workspace)})
        token_nodes: list[dict[str, Any]] = []
        for record in word_concepts:
            features: dict[str, float] = defaultdict(float)
            for candidate in record["candidates"]:
                for name, value in candidate["features"].items():
                    features[name] += float(value) * candidate["probability"]
            token_nodes.append({
                **record,
                "posterior_features": {key: float(np.clip(value, 0.0, 1.0)) for key, value in features.items()},
            })
        self._record(8, {
            "token_nodes_with_features": sum(bool(row["posterior_features"]) for row in token_nodes),
        })
        contextual_nodes = []
        for node in token_nodes:
            adjusted = dict(node["posterior_features"])
            for feature, value in request.context_features.items():
                adjusted[feature] = float(np.clip(
                    adjusted.get(feature, 0.0) + 0.25 * float(value), 0.0, 1.0
                ))
            contextual_nodes.append({**node, "contextual_features": adjusted})
        self._record(9, {
            "context_feature_count": len(request.context_features),
            "modulated_nodes": len(contextual_nodes),
        })
        pooled: dict[str, float] = defaultdict(float)
        denominator = 0.0
        for node in contextual_nodes:
            if node["contextual_features"]:
                weight = 1.0 / math.sqrt(max(1, len(contextual_nodes)))
                denominator += weight
                for feature, value in node["contextual_features"].items():
                    pooled[feature] += float(value) * weight
        pooled_features = {
            key: float(np.clip(value / max(denominator, _EPS), 0.0, 1.0))
            for key, value in pooled.items()
        }
        self._record(10, {"pooled_feature_count": len(pooled_features)})
        association = []
        for left in contextual_nodes:
            association.append([
                _cosine(left["contextual_features"], right["contextual_features"])
                for right in contextual_nodes
            ])
        self._record(11, {"association_matrix_size": len(association)})
        result = {
            "workspace": workspace,
            "token_nodes": contextual_nodes,
            "pooled_features": pooled_features,
            "association_matrix": association,
            "composition_policy": (
                "Posterior-weighted feature pooling; token-level nodes and pairwise relations "
                "are retained so word order and later predicate-role bindings are not erased."
            ),
        }
        self._record(12, {
            "preserved_token_nodes": len(contextual_nodes),
            "preserved_pairwise_relations": sum(len(row) for row in association),
            "pooled_feature_dimensions": len(pooled_features),
        })
        return result

    def _pos_at(self, syntax: Mapping[str, Any], token_index: int) -> str | None:
        """Return the selected part-of-speech label for one token when syntax evidence exists."""
        for item in syntax.get("selected_pos", ()) or ():
            index = _get(item, "token_index", _get(item, "index"))
            if index == token_index:
                return str(_get(item, "tag", _get(item, "pos", _get(item, "label", "")))).upper() or None
        return None

    def _dependency_arcs(self, syntax: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Normalize common dependency-arc field names into head/dependent index records."""
        arcs = []
        for arc in syntax.get("dependency_arcs", ()) or ():
            head = _get(arc, "head_index", _get(arc, "head"))
            dependent = _get(arc, "dependent_index", _get(arc, "dependent"))
            relation = str(_get(arc, "relation", _get(arc, "dependency_relation", "dep"))).lower()
            try:
                dependent_index = int(dependent)
                head_index = None if head in (None, "ROOT", "root", -1) else int(head)
            except (TypeError, ValueError):
                continue
            arcs.append({
                "head_index": head_index,
                "dependent_index": dependent_index,
                "relation": relation,
                "source": "syntax_dependency",
            })
        return arcs

    def _stage_three_predicate_arguments(
        self, tokens: list[str], word_concepts: list[dict[str, Any]],
        syntax: Mapping[str, Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Detect predicates and compose each with argument candidates from syntax and valency frames."""
        predicates = []
        for index, token in enumerate(tokens):
            record = word_concepts[index]
            category = record["selected_lexical_category"].casefold()
            pos = (self._pos_at(syntax, index) or "").upper()
            if category in _PREDICATE_CATEGORIES or pos in {"VERB", "AUX", "ADJ"} or pos.startswith(("VB", "JJ")):
                label = record["selected_label"] or token
                frame = self._frames_by_concept.get(record["selected_concept_id"]) or self._frames_by_lemma.get(_normalize(label))
                predicates.append({
                    "predicate_index": index, "predicate_token": token,
                    "predicate_label": label, "predicate_concept_id": record["selected_concept_id"],
                    "confidence": record["confidence"] or 0.25, "_frame": frame,
                })
        syntax_predicates = syntax.get("predicate_arguments", ()) or ()
        for item in syntax_predicates:
            index = _get(item, "predicate_index", _get(item, "predicate_token_index"))
            try:
                index = int(index)
            except (TypeError, ValueError):
                continue
            if 0 <= index < len(tokens) and not any(row["predicate_index"] == index for row in predicates):
                record = word_concepts[index]
                label = record["selected_label"] or tokens[index]
                frame = self._frames_by_concept.get(record["selected_concept_id"]) or self._frames_by_lemma.get(_normalize(label))
                predicates.append({
                    "predicate_index": index, "predicate_token": tokens[index],
                    "predicate_label": label, "predicate_concept_id": record["selected_concept_id"],
                    "confidence": max(record["confidence"], 0.35), "_frame": frame,
                })
        predicates.sort(key=lambda item: item["predicate_index"])
        self._record(13, {
            "predicate_indexes": [item["predicate_index"] for item in predicates],
            "predicate_count": len(predicates),
        })
        frames = []
        for predicate in predicates:
            frame = predicate["_frame"]
            if frame is None:
                frame = PredicateFrame(
                    predicate=predicate["predicate_label"],
                    roles=("Agent", "Patient", "Recipient", "Theme", "Experiencer", "Stimulus", "Instrument", "Location", "Goal", "Source"),
                    source_reference="fallback:caller-frame-not-supplied",
                )
                predicate["_frame"] = frame
            frames.append({
                "predicate_index": predicate["predicate_index"],
                "predicate": predicate["predicate_label"],
                "roles": list(frame.roles),
                "required_roles": list(frame.required_roles),
                "source_reference": frame.source_reference,
                "frame_supplied": frame.source_reference != "fallback:caller-frame-not-supplied",
            })
        self._record(14, {
            "frames_retrieved": len(frames),
            "caller_supplied_frames": sum(bool(item["frame_supplied"]) for item in frames),
        })
        arcs = self._dependency_arcs(syntax)
        self._record(15, {
            "dependency_arc_count": len(arcs),
            "dependency_relations": sorted({item["relation"] for item in arcs}),
        })
        argument_frames = []
        for predicate in predicates:
            predicate_index = predicate["predicate_index"]
            frame = predicate["_frame"]
            arguments = []
            for arc in arcs:
                if arc["head_index"] != predicate_index:
                    continue
                dependent = arc["dependent_index"]
                if not 0 <= dependent < len(tokens):
                    continue
                relation = arc["relation"]
                role = _ROLE_BY_DEPENDENCY.get(relation, "Oblique" if relation.startswith("obl") else "Theme")
                arguments.append({
                    "token_index": dependent, "token": tokens[dependent],
                    "proposed_role": role, "dependency_relation": relation,
                    "syntactic_confidence": 0.90, "source": arc["source"],
                })
            # Some syntax adapters expose predicate-centered argument frames directly.
            for syntax_item in syntax_predicates:
                try:
                    item_index = int(_get(syntax_item, "predicate_index", _get(syntax_item, "predicate_token_index")))
                except (TypeError, ValueError):
                    continue
                if item_index != predicate_index:
                    continue
                for arg in (_get(syntax_item, "arguments", ()) or ()):
                    try:
                        arg_index = int(_get(arg, "token_index", _get(arg, "argument_index")))
                    except (TypeError, ValueError):
                        continue
                    if 0 <= arg_index < len(tokens) and not any(row["token_index"] == arg_index for row in arguments):
                        arguments.append({
                            "token_index": arg_index, "token": tokens[arg_index],
                            "proposed_role": str(_get(arg, "role", "Theme")),
                            "dependency_relation": str(_get(arg, "relation", "adapter_argument")),
                            "syntactic_confidence": float(_get(arg, "confidence", 0.85)),
                            "source": "syntax_predicate_argument_adapter",
                        })
            arguments.sort(key=lambda item: item["token_index"])
            argument_frames.append({
                "predicate_index": predicate_index,
                "predicate": predicate["predicate_label"],
                "arguments": arguments,
                "frame_roles": list(frame.roles),
            })
        self._record(16, {
            "argument_slot_counts": [len(item["arguments"]) for item in argument_frames],
        })
        token_features = {
            row["token_index"]: row["selected_features"]
            for row in word_concepts
        }
        for frame_row, predicate in zip(argument_frames, predicates):
            frame = predicate["_frame"]
            scored = []
            for arg in frame_row["arguments"]:
                preference = frame.selectional_preferences.get(arg["proposed_role"], {})
                semantic_fit = _cosine(token_features.get(arg["token_index"], {}), preference) if preference else 0.5
                score = (
                    self.config.syntax_weight * arg["syntactic_confidence"]
                    + self.config.selectional_weight * semantic_fit
                )
                scored.append({
                    "token_index": arg["token_index"],
                    "role": arg["proposed_role"],
                    "syntax_fit": arg["syntactic_confidence"],
                    "semantic_fit": semantic_fit,
                    "compatibility_score": score,
                })
            frame_row["compatibility"] = scored
        self._record(17, {
            "scored_argument_count": sum(len(item.get("compatibility", ())) for item in argument_frames),
        })
        for frame_row, predicate in zip(argument_frames, predicates):
            frame = predicate["_frame"]
            filled = {item["role"] for item in frame_row.get("compatibility", ())}
            frame_row["role_status"] = {
                role: ("filled" if role in filled else "missing_required" if role in frame.required_roles else "optional_or_unobserved")
                for role in frame.roles
            }
            frame_row["completeness"] = (
                sum(role in filled for role in frame.required_roles) / len(frame.required_roles)
                if frame.required_roles else 1.0
            )
        self._record(18, {
            "frames_with_all_required_roles": sum(
                all(value != "missing_required" for value in item["role_status"].values())
                for item in argument_frames
            ),
        })
        return predicates, argument_frames

    def _stage_four_thematic_roles(
        self, predicates: list[dict[str, Any]], argument_frames: list[dict[str, Any]],
        word_concepts: list[dict[str, Any]], syntax: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        """Assign Agent, Patient, and other thematic roles using transparent syntax and feature evidence."""
        all_roles: list[dict[str, Any]] = []
        token_features = {row["token_index"]: row["selected_features"] for row in word_concepts}
        self._record(19, {
            "candidate_role_sets": [
                {"predicate_index": row["predicate_index"], "roles": row["frame_roles"]}
                for row in argument_frames
            ],
        })
        for frame_row, predicate in zip(argument_frames, predicates):
            frame = predicate["_frame"]
            for argument in frame_row["arguments"]:
                relation = argument["dependency_relation"].lower()
                proposed = argument["proposed_role"]
                voice = "passive" if "pass" in relation else "active_or_unmarked"
                argument["voice_hint"] = voice
                argument["syntactic_role_evidence"] = {
                    role: (1.0 if role == proposed else 0.25)
                    for role in frame.roles
                }
        self._record(20, {
            "arguments_with_syntactic_role_evidence": sum(len(row["arguments"]) for row in argument_frames),
        })
        for frame_row, predicate in zip(argument_frames, predicates):
            frame = predicate["_frame"]
            for argument in frame_row["arguments"]:
                preference = frame.selectional_preferences.get(argument["proposed_role"], {})
                features = token_features.get(argument["token_index"], {})
                argument["semantic_role_evidence"] = {
                    role: (
                        _cosine(features, frame.selectional_preferences[role])
                        if frame.selectional_preferences.get(role) else 0.5
                    )
                    for role in frame.roles
                }
        self._record(21, {
            "arguments_with_semantic_evidence": sum(len(row["arguments"]) for row in argument_frames),
        })
        for frame_row, predicate in zip(argument_frames, predicates):
            frame = predicate["_frame"]
            for argument in frame_row["arguments"]:
                argument["role_scores"] = {
                    role: (
                        self.config.syntax_weight * argument["syntactic_role_evidence"][role]
                        + self.config.selectional_weight * argument["semantic_role_evidence"][role]
                    )
                    for role in frame.roles
                }
        self._record(22, {
            "role_score_vectors": sum(len(row["arguments"]) for row in argument_frames),
        })
        for frame_row, predicate in zip(argument_frames, predicates):
            frame = predicate["_frame"]
            for argument in frame_row["arguments"]:
                roles = list(frame.roles)
                probabilities = _softmax(
                    [argument["role_scores"][role] for role in roles],
                    self.config.softmax_temperature,
                )
                argument["role_posteriors"] = dict(zip(roles, probabilities))
        self._record(23, {
            "role_posteriors_computed": sum(len(row["arguments"]) for row in argument_frames),
        })
        for frame_row in argument_frames:
            for argument in frame_row["arguments"]:
                ranked = sorted(
                    argument["role_posteriors"].items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
                top_probability = ranked[0][1] if ranked else 0.0
                margin = top_probability - (ranked[1][1] if len(ranked) > 1 else 0.0)
                selected = (
                    ranked[0][0]
                    if top_probability >= self.config.confidence_threshold
                    and margin >= self.config.ambiguity_margin else None
                )
                role_record = {
                    "predicate_index": frame_row["predicate_index"],
                    "predicate": frame_row["predicate"],
                    "argument_index": argument["token_index"],
                    "argument": argument["token"],
                    "selected_role": selected,
                    "role_candidates": [
                        {"role": role, "probability": probability}
                        for role, probability in ranked
                    ],
                    "confidence": float(top_probability),
                    "margin": float(margin),
                    "abstained": selected is None,
                    "evidence_source": argument["source"],
                    "voice_hint": argument["voice_hint"],
                }
                all_roles.append(role_record)
        self._record(24, {
            "role_assignments": sum(row["selected_role"] is not None for row in all_roles),
            "role_abstentions": sum(row["abstained"] for row in all_roles),
        }, confidence=float(np.mean([row["confidence"] for row in all_roles]) if all_roles else 0.0))
        return all_roles

    def _stage_five_propositions(
        self, tokens: list[str], predicates: list[dict[str, Any]],
        argument_frames: list[dict[str, Any]], roles: list[dict[str, Any]],
        syntax: Mapping[str, Any], word_concepts: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Construct event propositions with entity references and validated participant bindings."""
        propositions = []
        for predicate in predicates:
            index = predicate["predicate_index"]
            propositions.append({
                "proposition_id": f"prop:{index}",
                "predicate_index": index,
                "predicate": predicate["predicate_label"],
                "predicate_concept_id": predicate["predicate_concept_id"],
                "arguments": [],
                "polarity": "positive",
                "modal_operators": [],
                "tense": None,
                "aspect": None,
                "voice": None,
                "confidence": 0.0,
            })
        self._record(25, {"proposition_shell_count": len(propositions)})
        entities = {}
        for row in word_concepts:
            index = row["token_index"]
            entities[index] = {
                "entity_id": f"entity:{index}",
                "token_index": index,
                "surface": row["token"],
                "concept_id": row["selected_concept_id"],
                "concept_label": row["selected_label"],
                "features": row["selected_features"],
                "confidence": row["confidence"],
            }
        self._record(26, {"entity_reference_count": len(entities)})
        for proposition in propositions:
            frame = next(
                (item for item in argument_frames if item["predicate_index"] == proposition["predicate_index"]),
                {"arguments": []},
            )
            for argument in frame["arguments"]:
                role = next(
                    (item for item in roles
                     if item["predicate_index"] == proposition["predicate_index"]
                     and item["argument_index"] == argument["token_index"]),
                    None,
                )
                entity = entities.get(argument["token_index"])
                proposition["arguments"].append({
                    "entity_id": entity["entity_id"] if entity else None,
                    "token_index": argument["token_index"],
                    "surface": argument["token"],
                    "role": role["selected_role"] if role else None,
                    "role_confidence": role["confidence"] if role else 0.0,
                    "role_alternatives": role["role_candidates"] if role else [],
                })
        self._record(27, {
            "proposition_argument_bindings": sum(len(item["arguments"]) for item in propositions),
        })
        auxiliaries = {token.casefold() for token in tokens}
        past_hint = any(
            token.casefold() in {"did", "was", "were", "had"}
            or (len(token) > 3 and token.casefold().endswith("ed"))
            for token in tokens
        )
        tense = (
            "future" if auxiliaries.intersection({"will", "shall"})
            else "past" if past_hint else "unspecified"
        )
        voice = "passive" if any(
            "pass" in str(_get(arc, "relation", "")).lower()
            for arc in (syntax.get("dependency_arcs", ()) or ())
        ) else "unspecified"
        for proposition in propositions:
            proposition["tense"] = tense
            proposition["aspect"] = "progressive" if any(token.casefold() in {"is", "are", "was", "were"} for token in tokens) and any(token.casefold().endswith("ing") for token in tokens) else "unspecified"
            proposition["voice"] = voice
        self._record(28, {"tense": tense, "aspect_marked": any(item["aspect"] != "unspecified" for item in propositions), "voice": voice})
        for proposition in propositions:
            argument_ids = [item["entity_id"] for item in proposition["arguments"]]
            proposition["well_formed"] = bool(proposition["predicate"]) and all(argument_ids)
            frame = next(
                (item for item in argument_frames if item["predicate_index"] == proposition["predicate_index"]),
                None,
            )
            proposition["missing_required_roles"] = (
                [
                    role for role, status in frame["role_status"].items()
                    if status == "missing_required"
                ] if frame else []
            )
        self._record(29, {
            "well_formed_propositions": sum(item["well_formed"] for item in propositions),
            "incomplete_required_roles": sum(len(item["missing_required_roles"]) for item in propositions),
        })
        for proposition in propositions:
            word_confidence = word_concepts[proposition["predicate_index"]]["confidence"]
            role_confidences = [item["role_confidence"] for item in proposition["arguments"] if item["role"]]
            proposition["confidence"] = float(np.mean([word_confidence, *role_confidences])) if role_confidences else float(word_confidence)
            proposition["confidence_interpretation"] = "support in supplied lexical/syntactic evidence; not truth probability"
        self._record(30, {
            "proposition_confidences": [
                round(item["confidence"], 4) for item in propositions
            ],
        }, confidence=float(np.mean([item["confidence"] for item in propositions]) if propositions else 0.0))
        return propositions

    def _stage_six_operators(
        self, tokens: list[str], propositions: list[dict[str, Any]], syntax: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Detect negation and modality and attach explicit operators to proposition scopes."""
        lowered = [_normalize(token) for token in tokens]
        negation_cues = [
            {"token_index": index, "token": tokens[index], "kind": "negation"}
            for index, token in enumerate(lowered)
            if token in _NEGATION or token.endswith("n't")
        ]
        negation_cues.extend(
            {"token_index": int(_get(arc, "dependent_index", -1)), "token": "syntax_negation", "kind": "negation"}
            for arc in (syntax.get("dependency_arcs", ()) or ())
            if str(_get(arc, "relation", "")).lower() == "neg"
            and isinstance(_get(arc, "dependent_index", -1), int)
        )
        self._record(31, {"negation_cue_count": len(negation_cues), "cue_indexes": [item["token_index"] for item in negation_cues]})
        scopes = []
        for cue in negation_cues:
            valid = [p for p in propositions if p["predicate_index"] >= 0]
            target = min(valid, key=lambda p: abs(p["predicate_index"] - cue["token_index"])) if valid else None
            scopes.append({
                **cue,
                "target_proposition_id": target["proposition_id"] if target else None,
                "scope_method": "nearest_predicate_within_sentence",
                "scope_confidence": 0.60 if target else 0.0,
            })
        self._record(32, {"negation_scopes": len(scopes), "scoped_cues": sum(item["target_proposition_id"] is not None for item in scopes)})
        for proposition in propositions:
            count = sum(item["target_proposition_id"] == proposition["proposition_id"] for item in scopes)
            proposition["polarity"] = "negative" if count % 2 else "positive"
            proposition["negation_count"] = count
        self._record(33, {
            "negative_proposition_count": sum(item["polarity"] == "negative" for item in propositions),
            "double_negation_count": sum(item["negation_count"] >= 2 for item in propositions),
        })
        modal_cues = [
            {
                "token_index": index,
                "token": token,
                "modal_type": _MODAL_FORCE[token][0],
                "force": _MODAL_FORCE[token][1],
            }
            for index, token in enumerate(lowered) if token in _MODAL_FORCE
        ]
        self._record(34, {"modal_cue_count": len(modal_cues), "modal_indexes": [item["token_index"] for item in modal_cues]})
        modal_scopes = []
        for cue in modal_cues:
            target = min(
                propositions,
                key=lambda p: abs(p["predicate_index"] - cue["token_index"]),
                default=None,
            )
            modal_scopes.append({
                **cue,
                "target_proposition_id": target["proposition_id"] if target else None,
                "scope_confidence": 0.65 if target else 0.0,
            })
        self._record(35, {
            "modal_scopes": len(modal_scopes),
            "modal_force_values": [item["force"] for item in modal_cues],
        })
        for proposition in propositions:
            proposition["modal_operators"] = [
                {
                    "token": item["token"], "modal_type": item["modal_type"],
                    "force": item["force"], "scope_confidence": item["scope_confidence"],
                }
                for item in modal_scopes
                if item["target_proposition_id"] == proposition["proposition_id"]
            ]
            proposition["operator_structure"] = {
                "polarity": proposition["polarity"],
                "modality": proposition["modal_operators"],
                "scope_order": "modal-over-negation-unless-syntax-provides-explicit-scope",
            }
        self._record(36, {
            "propositions_with_operators": sum(
                item["polarity"] == "negative" or bool(item["modal_operators"])
                for item in propositions
            ),
        })
        return {
            "negation_cues": negation_cues,
            "negation_scopes": scopes,
            "modal_cues": modal_cues,
            "modal_scopes": modal_scopes,
        }

    def _stage_seven_quantifiers(
        self, tokens: list[str], word_concepts: list[dict[str, Any]],
        propositions: list[dict[str, Any]], syntax: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        """Represent quantifier meanings, variable domains, and bounded scope alternatives."""
        lowered = [_normalize(token) for token in tokens]
        cues = [
            {
                "token_index": index, "token": tokens[index],
                "quantifier_type": _QUANTIFIERS[token][0],
                "symbol": _QUANTIFIERS[token][1],
            }
            for index, token in enumerate(lowered) if token in _QUANTIFIERS
        ]
        self._record(37, {"quantifier_cue_count": len(cues), "quantifier_indexes": [item["token_index"] for item in cues]})
        arcs = self._dependency_arcs(syntax)
        domains = []
        for cue in cues:
            det_dependents = [
                arc["head_index"] for arc in arcs
                if arc["dependent_index"] == cue["token_index"]
                and arc["relation"] in {"det", "det:predet", "nummod"}
                and arc["head_index"] is not None
            ]
            fallback_head = next(
                (
                    index for index in range(cue["token_index"] + 1, len(tokens))
                    if (
                        word_concepts[index]["selected_lexical_category"].casefold() in _NOUN_CATEGORIES
                        or (self._pos_at(syntax, index) or "").upper().startswith(("NN", "PRP"))
                        or (self._pos_at(syntax, index) or "").upper() in {"NOUN", "PROPN", "PRON"}
                    )
                ),
                None,
            )
            head = det_dependents[0] if det_dependents else fallback_head
            domains.append({
                **cue,
                "domain_head_index": head if 0 <= head < len(tokens) else None,
                "domain_head": tokens[head] if 0 <= head < len(tokens) else None,
                "domain_concept_id": word_concepts[head]["selected_concept_id"] if 0 <= head < len(tokens) else None,
                "domain_features": word_concepts[head]["selected_features"] if 0 <= head < len(tokens) else {},
                "domain_source": "dependency_determiner_arc" if det_dependents else "nearest_following_nominal_fallback",
            })
        self._record(38, {"quantified_domain_count": len(domains)})
        variables = []
        for number, domain in enumerate(domains, start=1):
            variables.append({
                **domain,
                "variable": f"x{number}",
                "restrictor": f"{domain['domain_head'] or 'entity'}(x{number})",
                "bound_entity_type": domain["domain_concept_id"] or "unresolved_nominal",
            })
        self._record(39, {"quantifier_variable_count": len(variables)})
        orders = []
        if variables:
            for order in list(permutations(range(len(variables))))[: self.config.max_scope_readings]:
                orders.append({
                    "order": list(order),
                    "quantifier_indexes": [variables[index]["token_index"] for index in order],
                    "scope_rank": len(orders) + 1,
                })
        else:
            orders = [{"order": [], "quantifier_indexes": [], "scope_rank": 1}]
        self._record(40, {"scope_candidate_count": len(orders)})
        formulas = []
        for variable in variables:
            quantifier_type = variable["quantifier_type"]
            operator = variable["symbol"]
            restrictor = variable["restrictor"]
            if quantifier_type in {"universal", "universal_pair"}:
                formula = f"{operator}{variable['variable']} ({restrictor} → P({variable['variable']}))"
            elif quantifier_type == "negative_existential":
                formula = f"¬∃{variable['variable']} ({restrictor} ∧ P({variable['variable']}))"
            elif quantifier_type == "exact_count":
                formula = f"{operator}{variable['variable']} ({restrictor} ∧ P({variable['variable']}))"
            else:
                formula = f"{operator}{variable['variable']} ({restrictor} ∧ P({variable['variable']}))"
            formulas.append({**variable, "formula": formula})
        self._record(41, {
            "formula_count": len(formulas),
            "formula_examples": [item["formula"] for item in formulas[:5]],
        })
        readings = []
        for order in orders:
            reading_formulas = [
                formulas[index]["formula"] for index in order["order"]
            ]
            readings.append({
                **order,
                "formulas": reading_formulas,
                "combined_scope_formula": " > ".join(reading_formulas) if reading_formulas else None,
                "confidence": 1.0 / max(1, len(orders)),
                "ambiguity_preserved": len(orders) > 1,
            })
        result = []
        for variable in formulas:
            alternatives = [
                {
                    "scope_rank": reading["scope_rank"],
                    "quantifier_order": reading["quantifier_indexes"],
                    "formula": reading["combined_scope_formula"],
                    "confidence": reading["confidence"],
                }
                for reading in readings
            ]
            result.append({
                **variable,
                "scope_alternatives": alternatives,
                "selected_scope_rank": alternatives[0]["scope_rank"] if alternatives else None,
                "scope_ambiguous": len(alternatives) > 1,
                "interpretation_note": (
                    "Symbolic scope sketch; predicate P is a placeholder where syntax does "
                    "not supply a complete quantified formula."
                ),
            })
        self._record(42, {
            "quantifiers_with_scope_alternatives": len(result),
            "ambiguous_quantifiers": sum(item["scope_ambiguous"] for item in result),
        })
        return result

    def _stage_eight_causality(
        self, tokens: list[str], propositions: list[dict[str, Any]],
        argument_frames: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Detect explicit causal constructions and create conservative directed event relations."""
        lowered = [_normalize(token) for token in tokens]
        cues = [
            {
                "token_index": index, "token": token,
                "cue_type": _CAUSAL_CUES[token][0],
                "cue_strength": _CAUSAL_CUES[token][1],
            }
            for index, token in enumerate(lowered) if token in _CAUSAL_CUES
        ]
        multiword = ("as a result", 0.90)
        joined = " ".join(lowered)
        if multiword[0] in joined:
            index = joined.split().index("as")
            cues.append({
                "token_index": index, "token": "as a result",
                "cue_type": "therefore", "cue_strength": multiword[1],
            })
        causative_verbs = {"cause", "causes", "caused", "causing", "lead", "leads", "led", "result", "results", "resulted"}
        cues.extend(
            {
                "token_index": index, "token": token,
                "cue_type": "causative_predicate", "cue_strength": 0.86,
            }
            for index, token in enumerate(lowered) if token in causative_verbs
        )
        self._record(43, {"causal_cue_count": len(cues), "cues": cues})
        ordered = sorted(propositions, key=lambda item: item["predicate_index"])
        candidates = []
        for cue in cues:
            before = [item for item in ordered if item["predicate_index"] < cue["token_index"]]
            after = [item for item in ordered if item["predicate_index"] > cue["token_index"]]
            if cue["cue_type"] == "because":
                cause = after[0] if after else None
                effect = before[-1] if before else (after[1] if len(after) > 1 else None)
            elif cue["cue_type"] == "since":
                cause = after[0] if after else None
                effect = before[-1] if before else (after[1] if len(after) > 1 else None)
            else:
                cause = before[-1] if before else None
                effect = after[0] if after else None
            if cause and effect and cause["proposition_id"] != effect["proposition_id"]:
                candidates.append({
                    "cue": cue["token"],
                    "cue_type": cue["cue_type"],
                    "cause_proposition_id": cause["proposition_id"],
                    "effect_proposition_id": effect["proposition_id"],
                    "cause_index": cause["predicate_index"],
                    "effect_index": effect["predicate_index"],
                    "cue_strength": cue["cue_strength"],
                })
            elif cue["cue_type"] == "causative_predicate":
                frame = next(
                    (item for item in argument_frames if item["predicate_index"] == cue["token_index"]),
                    None,
                )
                if frame and len(frame["arguments"]) >= 2:
                    arguments = frame["arguments"]
                    subject = next((a for a in arguments if a["proposed_role"] == "Agent"), arguments[0])
                    obj = next((a for a in arguments if a["proposed_role"] == "Patient"), arguments[-1])
                    candidates.append({
                        "cue": cue["token"], "cue_type": "causative_predicate",
                        "cause_argument_index": subject["token_index"],
                        "effect_argument_index": obj["token_index"],
                        "cause_proposition_id": None, "effect_proposition_id": None,
                        "cause_index": subject["token_index"], "effect_index": obj["token_index"],
                        "cue_strength": cue["cue_strength"],
                    })
        self._record(44, {"cause_effect_candidate_count": len(candidates)})
        directed = []
        for candidate in candidates:
            if candidate["cause_index"] > candidate["effect_index"] and candidate["cue_type"] not in {"because", "since"}:
                candidate["direction_note"] = "causal cue direction conflicts with token order; retained as cue-specific interpretation"
            else:
                candidate["direction_note"] = "direction assigned by explicit connective or causative predicate frame"
            directed.append({
                **candidate,
                "relation": "CAUSES",
                "direction": "cause_to_effect",
            })
        self._record(45, {
            "directed_relation_count": len(directed),
            "directions": [item["direction"] for item in directed],
        })
        for relation in directed:
            cause_index = relation["cause_index"]
            effect_index = relation["effect_index"]
            relation["temporal_order_evidence"] = (
                "cause_precedes_effect" if cause_index <= effect_index else "effect_precedes_cause"
            )
            relation["temporal_compatibility"] = 0.65 if cause_index <= effect_index else 0.45
            relation["temporal_is_not_causal_proof"] = True
        self._record(46, {
            "temporal_evidence_count": len(directed),
            "temporal_rule": "order contributes evidence only; it never creates a causal link by itself",
        })
        for relation in directed:
            structural = 1.0 if relation["cause_proposition_id"] and relation["effect_proposition_id"] else 0.75
            confidence = (
                relation["cue_strength"] * 0.60
                + structural * 0.25
                + relation["temporal_compatibility"] * 0.15
            )
            relation["confidence"] = float(np.clip(confidence, 0.0, 1.0))
            relation["confidence_basis"] = "explicit marker + structural alignment + temporal compatibility"
        self._record(47, {
            "causal_confidences": [round(item["confidence"], 4) for item in directed],
        }, confidence=float(np.mean([item["confidence"] for item in directed]) if directed else 0.0))
        graph = [
            {
                "relation_id": f"cause:{index}",
                "relation": item["relation"],
                "cause_proposition_id": item["cause_proposition_id"],
                "effect_proposition_id": item["effect_proposition_id"],
                "cause_argument_index": item.get("cause_argument_index"),
                "effect_argument_index": item.get("effect_argument_index"),
                "cue": item["cue"],
                "confidence": item["confidence"],
                "source": "explicit_causal_lexical_cue",
                "direction": item["direction"],
            }
            for index, item in enumerate(directed)
        ]
        self._record(48, {"causal_graph_edges": len(graph)})
        return graph

    def _stage_nine_sentence_representation(
        self,
        tokens: list[str],
        word_concepts: list[dict[str, Any]],
        combined: dict[str, Any],
        argument_frames: list[dict[str, Any]],
        roles: list[dict[str, Any]],
        propositions: list[dict[str, Any]],
        operators: dict[str, Any],
        quantifiers: list[dict[str, Any]],
        causal: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Integrate all semantic components into a typed directed sentence graph and validate it."""
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        for row in word_concepts:
            nodes.append({
                "node_id": f"token:{row['token_index']}",
                "node_type": "word_concept",
                "token_index": row["token_index"],
                "surface": row["token"],
                "concept_id": row["selected_concept_id"],
                "label": row["selected_label"],
                "confidence": row["confidence"],
                "alternatives": row["candidates"],
            })
        for row in propositions:
            nodes.append({
                "node_id": row["proposition_id"], "node_type": "proposition",
                "predicate": row["predicate"], "predicate_index": row["predicate_index"],
                "polarity": row["polarity"], "modal_operators": row["modal_operators"],
                "confidence": row["confidence"],
            })
        for item in quantifiers:
            node_id = f"quantifier:{item['token_index']}"
            nodes.append({
                "node_id": node_id, "node_type": "quantifier",
                "symbol": item["symbol"], "formula": item["formula"],
                "scope_ambiguous": item["scope_ambiguous"],
            })
            edges.append({
                "edge_type": "quantifies", "source": node_id,
                "target": f"token:{item['domain_head_index']}",
            })
        for role in roles:
            proposition_id = f"prop:{role['predicate_index']}"
            edges.append({
                "edge_type": "thematic_role",
                "source": proposition_id,
                "target": f"token:{role['argument_index']}",
                "role": role["selected_role"],
                "alternatives": role["role_candidates"],
                "confidence": role["confidence"],
            })
        for proposition in propositions:
            for argument in proposition["arguments"]:
                edges.append({
                    "edge_type": "argument_binding",
                    "source": proposition["proposition_id"],
                    "target": f"token:{argument['token_index']}",
                    "role": argument["role"],
                })
        for scope in operators["negation_scopes"]:
            if scope["target_proposition_id"]:
                edges.append({
                    "edge_type": "negates", "source": f"token:{scope['token_index']}",
                    "target": scope["target_proposition_id"],
                })
        for scope in operators["modal_scopes"]:
            if scope["target_proposition_id"]:
                edges.append({
                    "edge_type": "modalizes", "source": f"token:{scope['token_index']}",
                    "target": scope["target_proposition_id"],
                    "modal_type": scope["modal_type"], "force": scope["force"],
                })
        for relation in causal:
            if relation["cause_proposition_id"] and relation["effect_proposition_id"]:
                edges.append({
                    "edge_type": "causes",
                    "source": relation["cause_proposition_id"],
                    "target": relation["effect_proposition_id"],
                    "confidence": relation["confidence"],
                    "cue": relation["cue"],
                })
        self._record(49, {"typed_node_count": len(nodes)})
        self._record(50, {"typed_edge_count": len(edges)})
        graph = {
            "nodes": nodes,
            "edges": edges,
            "token_order": [f"token:{index}" for index in range(len(tokens))],
            "pooled_features": combined["pooled_features"],
            "association_matrix": combined["association_matrix"],
        }
        self._record(51, {
            "integrated_graph_nodes": len(nodes),
            "integrated_graph_edges": len(edges),
            "proposition_count": len(propositions),
        })
        concept_entropies = [
            _entropy([item["probability"] for item in row["candidates"]])
            for row in word_concepts if row["candidates"]
        ]
        role_entropies = [
            _entropy([item["probability"] for item in row["role_candidates"]])
            for row in roles if row["role_candidates"]
        ]
        scope_ambiguity = sum(item["scope_ambiguous"] for item in quantifiers)
        unresolved = sum(row["unresolved"] or row["ambiguous"] for row in word_concepts)
        uncertainty = {
            "mean_concept_entropy_nats": float(np.mean(concept_entropies)) if concept_entropies else 0.0,
            "mean_role_entropy_nats": float(np.mean(role_entropies)) if role_entropies else 0.0,
            "ambiguous_quantifier_count": int(scope_ambiguity),
            "unresolved_or_ambiguous_word_count": int(unresolved),
            "unassigned_role_count": sum(row["selected_role"] is None for row in roles),
            "causal_link_count": len(causal),
        }
        self._record(52, uncertainty)
        node_ids = {node["node_id"] for node in nodes}
        graph_errors = []
        for edge in edges:
            if edge["source"] not in node_ids or edge["target"] not in node_ids:
                graph_errors.append({
                    "edge_type": edge["edge_type"], "source": edge["source"],
                    "target": edge["target"],
                })
        valid_references = not graph_errors
        representation = {
            "representation_type": "typed_sentence_semantic_graph",
            "sequence_id": self._sequence_id,
            "language": self._language,
            "source_text": " ".join(tokens),
            "nodes": nodes,
            "edges": edges,
            "propositions": propositions,
            "quantifier_scopes": quantifiers,
            "negation_and_modality": operators,
            "causal_relations": causal,
            "pooled_semantic_features": combined["pooled_features"],
            "uncertainty": uncertainty,
            "validation": {
                "valid_references": valid_references,
                "graph_errors": graph_errors,
                "token_count": len(tokens),
            },
        }
        self._record(53, {
            "valid_references": valid_references,
            "graph_error_count": len(graph_errors),
            "uncertainty_dimensions": len(uncertainty),
        }, confidence=1.0 if valid_references else 0.0)
        self._record(54, {
            "schema": "ore.semantic_composition.9stage.v1",
            "tokens": len(tokens),
            "propositions": len(propositions),
            "operations": 54,
        }, confidence=1.0 if valid_references else 0.0)
        diagnostics = {
            **self._diagnostics,
            "graph_errors": graph_errors,
            "valid": valid_references,
            "operation_count": len(self._trace),
            "stage_count": len(STAGE_REGISTRY),
            "scientific_limitations": [
                "The functional equations are engineering approximations, not measured neural computations.",
                "The LIF track is a configurable computational reference, not an anatomically fitted brain model.",
                "Without reviewed concept inventories and predicate frames, word senses and role assignments may abstain or use marked fallbacks.",
                "Causal links are emitted only when an explicit causal marker or causative construction is detected.",
                "Quantifier formulas are schematic when complete predicate scope is unavailable.",
            ],
        }
        return representation, diagnostics

    def process(self, request: SemanticCompositionRequest) -> SemanticCompositionResult:
        """Execute all nine stages and return a complete 54-operation semantic result."""
        if not isinstance(request, SemanticCompositionRequest):
            raise TypeError("request must be a SemanticCompositionRequest")
        if request.tokens is not None:
            tokens = [str(token) for token in request.tokens]
            original_text = " ".join(tokens)
        else:
            assert request.text is not None
            tokens = _tokenize(request.text)
            original_text = request.text
        if not tokens:
            raise ValueError("semantic composition requires at least one token")
        if len(tokens) > self.config.max_tokens:
            raise ValueError(f"sentence contains {len(tokens)} tokens; max_tokens={self.config.max_tokens}")
        syntax = self._syntax_payload(request.syntax_result)
        self._diagnostics = {}
        if syntax.get("tokens") and list(syntax["tokens"]) != tokens:
            self._diagnostics["syntax_alignment_warning"] = "syntax token sequence differs from request tokens"
        self._sequence_id = request.sequence_id
        self._language = request.language
        self._trace: list[dict[str, Any]] = []
        self._biology_operation_states: list[dict[str, Any]] = []
        self._diagnostics.setdefault("concept_constraint_warnings", [])
        self._simulator = (
            SemanticCompositionLIFSimulator(self.config)
            if self.config.simulate_biology else None
        )
        self._diagnostics["syntax_supplied"] = request.syntax_result is not None
        self._diagnostics["concept_inventory_size"] = len(self.concepts)
        self._diagnostics["predicate_frame_inventory_size"] = len(self.frames)

        word_concepts = self._stage_one_retrieve_concepts(request, tokens, syntax)
        combined = self._stage_two_combine_meanings(request, tokens, word_concepts)
        predicates, argument_frames = self._stage_three_predicate_arguments(tokens, word_concepts, syntax)
        roles = self._stage_four_thematic_roles(predicates, argument_frames, word_concepts, syntax)
        propositions = self._stage_five_propositions(
            tokens, predicates, argument_frames, roles, syntax, word_concepts,
        )
        operators = self._stage_six_operators(tokens, propositions, syntax)
        quantifiers = self._stage_seven_quantifiers(tokens, word_concepts, propositions, syntax)
        causal = self._stage_eight_causality(tokens, propositions, argument_frames)
        representation, diagnostics = self._stage_nine_sentence_representation(
            tokens, word_concepts, combined, argument_frames, roles,
            propositions, operators, quantifiers, causal,
        )
        diagnostics["token_alignment_warning"] = bool(self._diagnostics.get("syntax_alignment_warning"))
        diagnostics["token_alignment_message"] = self._diagnostics.get("syntax_alignment_warning")
        biological = (
            self._simulator.summary()
            if self._simulator is not None
            else {
                "model": "disabled",
                "layers": {},
                "operation_states": [],
                "scientific_scope": "Biological-reference simulation disabled by configuration.",
            }
        )
        biological["operation_states"] = self._biology_operation_states
        return SemanticCompositionResult(
            sequence_id=request.sequence_id,
            original_text=original_text,
            tokens=tokens,
            word_concepts=word_concepts,
            combined_meaning=combined,
            predicate_arguments=argument_frames,
            thematic_roles=roles,
            propositions=propositions,
            negation_modality=operators,
            quantifier_scopes=quantifiers,
            causal_relations=causal,
            sentence_representation=representation,
            biological=biological,
            operation_trace=self._trace,
            diagnostics=diagnostics,
            metadata={
                "registry_id": "ORE-SEMANTIC-COMPOSITION-54",
                "registry_version": "1.0.0",
                "concept_inventory_fingerprint": self.model_fingerprint,
                "request_source_reference": request.source_reference,
                "configuration": self.config.to_dict(),
                "functional_track": "deterministic and inspectable",
                "biological_track": "optional current-based LIF computational reference",
            },
        )
