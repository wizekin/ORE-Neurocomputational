"""Adapters connecting ORE lexical, syntactic, semantic-composition and discourse
processors to the N9 controller.

These adapters pass explicit evidence between stages. They do not manufacture
lexical entries, semantic features, predicate frames, or world facts.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

import numpy as np

from ore_discourse import DiscourseN7Adapter, DiscourseProcessor
from ore_lexical_access import LexicalAccessRequest
from ore_semantic_composition.contracts import SemanticCompositionRequest
from ore_syntax import SyntaxRequest

from .contracts import N9Adapters


_SPECIAL_GRAPHEMES = {"<UNK>", "<PAD>", "<BLANK>", "<BOS>", "<EOS>"}
_POS_ALIASES = {
    "NOUN": "NOUN", "PROPN": "PROPN", "PROPER_NOUN": "PROPN",
    "VERB": "VERB", "AUX": "AUX", "ADJ": "ADJ", "ADJECTIVE": "ADJ",
    "ADV": "ADV", "ADVERB": "ADV", "PRON": "PRON", "PRONOUN": "PRON",
    "DET": "DET", "DETERMINER": "DET", "ADP": "ADP", "PREPOSITION": "ADP",
    "ADPOSITION": "ADP", "NUM": "NUM", "NUMBER": "NUM", "CONJ": "CONJ",
    "CCONJ": "CCONJ", "SCONJ": "SCONJ", "PART": "PART", "PUNCT": "PUNCT",
    "INTJ": "INTJ", "X": "X",
}


def _finite_feature_map(value: Any) -> dict[str, float]:
    """Keep only finite, bounded semantic features accepted by ORE contracts."""
    if not isinstance(value, Mapping):
        return {}
    output: dict[str, float] = {}
    for key, raw in value.items():
        try:
            number = float(raw)
        except (TypeError, ValueError):
            continue
        if isinstance(key, str) and key.strip() and math.isfinite(number) and 0.0 <= number <= 1.0:
            output[key] = number
    return output


def _context_features(context: Any) -> dict[str, float]:
    if not isinstance(context, Mapping):
        return {}
    return _finite_feature_map(context.get("semantic_features", context.get("features", {})))


def _normalise_spans(value: Any, token_count: int) -> list[tuple[int, int]]:
    """Validate [start, end) spans without guessing missing word boundaries."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    spans: list[tuple[int, int]] = []
    previous_end = 0
    for raw in value:
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != 2:
            continue
        try:
            start, end = int(raw[0]), int(raw[1])
        except (TypeError, ValueError):
            continue
        if start < previous_end or start < 0 or end <= start or end > token_count:
            continue
        spans.append((start, end))
        previous_end = end
    return spans


def _decode_grapheme_spans(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Decode N9 posterior rows only inside caller-supplied word spans."""
    class_names = [str(item) for item in payload.get("class_names", ())]
    probabilities = payload.get("grapheme_probabilities")
    if probabilities is None or not class_names:
        return []
    matrix = np.asarray(probabilities, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(class_names):
        raise ValueError("N6 adapter expected grapheme_probabilities shaped [patch, class]")
    spans = _normalise_spans(payload.get("word_spans"), matrix.shape[0])
    decoded: list[dict[str, Any]] = []
    for start, end in spans:
        symbols: list[str] = []
        confidence: list[float] = []
        for row in matrix[start:end]:
            if not np.isfinite(row).all() or float(row.sum()) <= 0.0:
                continue
            index = int(np.argmax(row))
            symbol = class_names[index]
            if symbol in _SPECIAL_GRAPHEMES:
                continue
            symbols.append(symbol)
            confidence.append(float(row[index]))
        decoded.append({
            "form": "".join(symbols).strip(),
            "start": start,
            "end": end,
            "visual_score": float(np.mean(confidence)) if confidence else 0.0,
        })
    return decoded


def _word_candidates(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Prefer explicit word hypotheses; otherwise decode N9's supplied spans."""
    supplied = payload.get("lexical_hypotheses")
    if isinstance(supplied, Sequence) and not isinstance(supplied, (str, bytes)):
        rows = [dict(item) for item in supplied if isinstance(item, Mapping)]
        if rows:
            return sorted(rows, key=lambda row: (int(row.get("start", 0)), int(row.get("end", 0))))
    return _decode_grapheme_spans(payload)


def _canonical_pos(category: str) -> str | None:
    return _POS_ALIASES.get(str(category).strip().upper())


def _usable_word_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [
        row for row in rows
        if str(row.get("form", row.get("token", ""))).strip()
    ]


class LexicalAccessN6Adapter:
    """Implement N9's N6 contract using the existing word-level lexical processor."""

    def __init__(self, processor: Any, *, language: str = "en") -> None:
        if not callable(getattr(processor, "process", None)):
            raise TypeError("processor must expose process(LexicalAccessRequest)")
        self.processor = processor
        self.language = language

    def resolve(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, Mapping):
            raise TypeError("N6 payload must be a mapping")
        rows = _word_candidates(payload)
        context_features = _context_features(payload.get("context", {}))
        sequence_id = str(payload.get("sequence_id", "n9-lexical"))
        language = str(payload.get("language", self.language))
        lexical_hypotheses: list[dict[str, Any]] = []
        pos_features: list[dict[str, Any]] = []
        morphology: list[dict[str, Any]] = []
        phonological_forms: list[dict[str, Any]] = []
        operation_count = 0

        for index, row in enumerate(rows):
            form = str(row.get("form", row.get("token", ""))).strip()
            start = int(row.get("start", index))
            end = int(row.get("end", start + 1))
            if not form:
                lexical_hypotheses.append({
                    **row, "form": "", "score": 0.0,
                    "source": "ore_lexical_access_abstention",
                    "semantics_verified": False,
                })
                pos_features.append({"token_index": index, "pos_candidates": {}})
                continue

            result = self.processor.process(LexicalAccessRequest(
                sequence_id=f"{sequence_id}:word:{index}",
                word=form,
                language=language,
                context_features=context_features,
                source_reference="ore_n9.N6.LexicalAccessN6Adapter",
            ))
            operation_count += len(result.operation_trace)
            identities = list(result.lexical_identity_candidates)
            meanings = list(result.meaning_candidates)
            categories = list(result.grammatical_categories)
            top_identity = result.selected_lexical_identity or (identities[0] if identities else None)
            identity_score = float(top_identity.get("posterior", 0.0)) if top_identity else 0.0
            semantic_candidates = [{
                "concept_id": str(item["meaning_id"]),
                "label": str(item.get("gloss") or item["meaning_id"]),
                "probability": float(item["posterior"]),
                "features": dict(item.get("features", {})),
                "source_reference": str(item.get("source_reference", "unspecified")),
            } for item in meanings]
            semantic_features: dict[str, float] = {}
            for candidate in semantic_candidates:
                for feature, value in candidate["features"].items():
                    semantic_features[feature] = semantic_features.get(feature, 0.0) + (
                        float(candidate["probability"]) * float(value)
                    )
            semantic_features = {
                key: min(1.0, max(0.0, value))
                for key, value in semantic_features.items()
            }
            pos_candidates: dict[str, float] = {}
            for item in categories:
                canonical = _canonical_pos(str(item.get("category", "")))
                if canonical:
                    pos_candidates[canonical] = float(item["posterior"])
            best_category = max(
                categories, key=lambda item: float(item.get("posterior", 0.0)),
                default=None,
            )
            selected_pos = (
                _canonical_pos(str(best_category.get("category", "")))
                if best_category else None
            )

            lexical_hypotheses.append({
                "form": form,
                "start": start,
                "end": end,
                "score": identity_score,
                "source": "ore_lexical_access",
                "semantics_verified": bool(semantic_candidates),
                "entry_id": top_identity.get("entry_id") if top_identity else None,
                "lexical_identity_candidates": identities,
                "selected_lexical_identity": top_identity,
                "meaning_candidates": meanings,
                "selected_meaning": result.selected_meaning,
                "semantic_candidates": semantic_candidates,
                "semantic_features": semantic_features,
                "pos": selected_pos,
                "pos_candidates": pos_candidates,
                "morphology": result.preferred_morphology,
                "diagnostics": result.diagnostics,
                "visual_score": float(row.get("visual_score", row.get("score", 0.0))),
            })
            pos_features.append({
                "token_index": index,
                "token": form,
                "pos": selected_pos,
                "pos_candidates": pos_candidates,
                "semantic_features": semantic_features,
            })
            if result.preferred_morphology:
                morphology.append({"token_index": index, **result.preferred_morphology})
            if top_identity:
                phonemes = next(
                    (getattr(entry, "phonemes", ()) for entry in getattr(self.processor, "entries", ())
                     if str(getattr(entry, "entry_id", "")) == str(top_identity.get("entry_id"))),
                    (),
                )
                if phonemes:
                    phonological_forms.append({
                        "token_index": index, "form": form, "phonemes": list(phonemes),
                    })

        return {
            "lexical_hypotheses": lexical_hypotheses,
            "pos_features": pos_features,
            "morphology": morphology,
            "phonological_forms": phonological_forms,
            "operation_count": operation_count,
            "status": "completed" if lexical_hypotheses else "unavailable_no_word_spans",
            "source_reference": "ore_lexical_access",
        }


class SyntaxN9Adapter:
    """Implement N9's syntax-parser slot using ORE's 58-operation syntax processor."""

    def __init__(self, processor: Any, *, language: str = "en") -> None:
        if not callable(getattr(processor, "process", None)):
            raise TypeError("processor must expose process(SyntaxRequest)")
        self.processor = processor
        self.language = language

    def parse(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        lexical_rows = _usable_word_rows(_word_candidates(payload))
        tokens = [str(item.get("form", item.get("token", ""))).strip() for item in lexical_rows]
        if not tokens:
            return {
                "status": "unavailable_no_lexical_tokens",
                "tokens": [], "parses": [], "dependency_arcs": [],
                "predicate_arguments": [], "thematic_roles": [],
                "phrases": [], "scope": {}, "discourse_relations": [], "coreference": [],
            }

        lexical_evidence: dict[int, dict[str, float]] = {}
        semantic_features: dict[int, dict[str, float]] = {}
        supplied_pos = payload.get("pos_features", ())
        for index, row in enumerate(lexical_rows):
            candidate_row = row.get("pos_candidates", {})
            if not candidate_row and isinstance(supplied_pos, Sequence) and index < len(supplied_pos):
                supplied = supplied_pos[index]
                if isinstance(supplied, Mapping):
                    candidate_row = supplied.get("pos_candidates", supplied.get("pos", {}))
            canonical: dict[str, float] = {}
            if isinstance(candidate_row, Mapping):
                for category, raw in candidate_row.items():
                    tag = _canonical_pos(str(category))
                    try:
                        value = float(raw)
                    except (TypeError, ValueError):
                        continue
                    if tag and math.isfinite(value) and 0.0 <= value <= 1.0:
                        canonical[tag] = value
            if canonical:
                lexical_evidence[index] = {"pos_candidates": canonical}
            features = _finite_feature_map(row.get("semantic_features", {}))
            if features:
                semantic_features[index] = features

        result = self.processor.process(SyntaxRequest(
            sequence_id=str(payload.get("sequence_id", "n9-syntax")),
            tokens=tuple(tokens),
            language=str(payload.get("language", self.language)),
            lexical_evidence=lexical_evidence,
            semantic_features=semantic_features,
            source_reference="ore_n9.SyntaxN9Adapter",
        ))
        output = result.to_dict()
        return {
            **output,
            "parses": output.get("constituency_parses", []),
            "scope": {},
            "discourse_relations": [],
            "coreference": [],
            "status": "completed",
        }


class SemanticCompositionN8Adapter:
    """Implement N9's N8 slot with sentence-level semantic composition.

    This is intentionally distinct from SemanticN8Adapter, which exposes
    individual concepts only. This adapter returns predicate-role propositions
    and a typed sentence graph from SemanticCompositionProcessor.
    """

    def __init__(self, processor: Any, *, language: str = "en") -> None:
        if not callable(getattr(processor, "process", None)):
            raise TypeError("processor must expose process(SemanticCompositionRequest)")
        self.processor = processor
        self.language = language
        self._known_concept_ids = {
            str(getattr(item, "concept_id"))
            for item in getattr(processor, "concepts", ())
            if getattr(item, "concept_id", None)
        }

    def compose(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, Mapping):
            raise TypeError("N8 payload must be a mapping")
        lexical_rows = _usable_word_rows(_word_candidates(payload))
        tokens = [str(item.get("form", item.get("token", ""))).strip() for item in lexical_rows]
        if not tokens:
            return {
                "semantic_graph": {"nodes": [], "edges": []},
                "propositions": [], "completeness": 0.0, "ambiguity": 1.0,
                "status": "unavailable_no_lexical_tokens",
            }

        concept_hypotheses: dict[int, list[str]] = {}
        for index, row in enumerate(lexical_rows):
            candidates = row.get("semantic_candidates", ())
            allowed = [
                str(item.get("concept_id"))
                for item in candidates
                if isinstance(item, Mapping)
                and item.get("concept_id")
                and (not self._known_concept_ids or str(item["concept_id"]) in self._known_concept_ids)
            ]
            if allowed:
                concept_hypotheses[index] = list(dict.fromkeys(allowed))

        syntax = payload.get("syntax", payload.get("syntax_result"))
        if hasattr(syntax, "to_dict") and callable(syntax.to_dict):
            syntax = syntax.to_dict()
        if not isinstance(syntax, Mapping):
            syntax = {}
        request = SemanticCompositionRequest(
            sequence_id=str(payload.get("sequence_id", "n9-semantic")),
            tokens=tuple(tokens),
            language=str(payload.get("language", self.language)),
            concept_hypotheses=concept_hypotheses,
            syntax_result=dict(syntax),
            context_features=_context_features(payload.get("context", {})),
            source_reference="ore_n9.SemanticCompositionN8Adapter",
        )
        result = self.processor.process(request)
        result_payload = result.to_dict()
        representation = result_payload["sentence_representation"]

        nodes: list[dict[str, Any]] = []
        semantic_word_rows = {
            int(row["token_index"]): row
            for row in result_payload.get("word_concepts", [])
        }
        for node in representation.get("nodes", []):
            node_id = str(node.get("node_id", "")).strip()
            node_type = str(node.get("node_type", "semantic"))
            if not node_id:
                continue
            node_payload = dict(node)
            if node_type == "word_concept":
                try:
                    token_index = int(node.get("token_index", -1))
                except (TypeError, ValueError):
                    token_index = -1
                word_row = semantic_word_rows.get(token_index, {})
                features = dict(word_row.get("selected_features", {}))
                node_payload["semantic_features"] = features
                if max(features.get("HUMAN", 0.0), features.get("PERSON", 0.0)) >= 0.5:
                    node_type = "person"
                else:
                    node_type = "concept"
            nodes.append({
                **node_payload,
                "id": node_id,
                "type": node_type,
                "label": str(node.get("label") or node.get("predicate") or node.get("surface") or node_id),
            })
        edges: list[dict[str, Any]] = []
        for edge in representation.get("edges", []):
            source, target = str(edge.get("source", "")), str(edge.get("target", ""))
            relation = str(edge.get("edge_type", edge.get("relation", ""))).strip()
            if source and target and relation:
                edges.append({**dict(edge), "source": source, "target": target, "relation": relation})

        compatibility_by_argument: dict[tuple[int, int], dict[str, Any]] = {}
        for frame in result_payload.get("predicate_arguments", []):
            predicate_index = int(frame.get("predicate_index", -1))
            for item in frame.get("compatibility", []):
                token_index = int(item.get("token_index", -1))
                compatibility_by_argument[(predicate_index, token_index)] = dict(item)

        discourse_propositions: list[dict[str, Any]] = []
        for proposition in result_payload.get("propositions", []):
            predicate_index = int(proposition.get("predicate_index", -1))
            argument_map: dict[str, str] = {}
            argument_details: list[dict[str, Any]] = []
            for position, argument in enumerate(proposition.get("arguments", [])):
                role = str(argument.get("role") or f"participant_{position + 1}")
                base_role, suffix = role, 2
                while role in argument_map:
                    role = f"{base_role}_{suffix}"
                    suffix += 1
                entity_id = argument.get("entity_id")
                token_index = int(argument.get("token_index", -1))
                # Sentence-graph token nodes are the shared entity references
                # consumed by N9 graph validation and the N7 discourse adapter.
                graph_entity_id = f"token:{token_index}" if token_index >= 0 else entity_id
                argument_map[role] = str(graph_entity_id or argument.get("surface", ""))
                compatibility = compatibility_by_argument.get((
                    predicate_index, int(argument.get("token_index", -1))
                ), {})
                argument_details.append({
                    **dict(argument),
                    "selectional_compatibility": compatibility.get("semantic_fit"),
                    "compatibility_score": compatibility.get("compatibility_score"),
                })
            modal_operators = list(proposition.get("modal_operators", []))
            discourse_propositions.append({
                **dict(proposition),
                "sentence_index": 0,
                "text": result_payload.get("original_text", " ".join(tokens)),
                "arguments": argument_map,
                "argument_details": argument_details,
                "negated": proposition.get("polarity") == "negative",
                "modality": modal_operators[0].get("token", "asserted") if modal_operators else "asserted",
                "source_reference": "ore_semantic_composition",
            })

        word_rows = result_payload.get("word_concepts", [])
        token_count = max(1, len(tokens))
        selected_count = sum(bool(row.get("selected_concept_id")) for row in word_rows)
        token_coverage = selected_count / token_count
        propositions = result_payload.get("propositions", [])
        proposition_coverage = (
            sum(bool(row.get("well_formed")) for row in propositions) / len(propositions)
            if propositions else 0.0
        )
        role_rows = result_payload.get("thematic_roles", [])
        role_coverage = (
            sum(row.get("selected_role") is not None for row in role_rows) / len(role_rows)
            if role_rows else (1.0 if not propositions else 0.0)
        )
        completeness = float(np.clip(
            0.45 * token_coverage + 0.35 * proposition_coverage + 0.20 * role_coverage,
            0.0, 1.0,
        ))
        uncertainty = result_payload.get("sentence_representation", {}).get("uncertainty", {})
        ambiguity_parts = [
            int(uncertainty.get("unresolved_or_ambiguous_word_count", 0)) / token_count,
            int(uncertainty.get("unassigned_role_count", 0)) / max(1, len(role_rows)),
        ]
        quantifier_count = len(result_payload.get("quantifier_scopes", []))
        if quantifier_count:
            ambiguity_parts.append(
                int(uncertainty.get("ambiguous_quantifier_count", 0)) / quantifier_count
            )
        ambiguity = float(np.clip(np.mean(ambiguity_parts), 0.0, 1.0))
        return {
            "semantic_graph": {"nodes": nodes, "edges": edges},
            "propositions": discourse_propositions,
            "completeness": completeness,
            "ambiguity": ambiguity,
            "sentence_representation": representation,
            "semantic_result": result_payload,
            "operation_trace": result_payload.get("operation_trace", []),
            "diagnostics": result_payload.get("diagnostics", {}),
            "status": "completed" if result_payload.get("diagnostics", {}).get("valid") else "completed_with_warnings",
        }


def build_n9_language_adapters(
    *,
    lexical_access: Any,
    syntax: Any,
    semantic_composition: Any,
    discourse: DiscourseProcessor | None = None,
    base_adapters: N9Adapters | None = None,
    language: str = "en",
) -> N9Adapters:
    """Build N6/N7/N8/syntax adapters while preserving existing N1-N5 adapters.

    Supply processors constructed with the inventories and evidence appropriate
    to the target language/domain. A caller-supplied N7 adapter is retained.
    """
    base = base_adapters or N9Adapters()
    return N9Adapters(
        n1=base.n1,
        n2=base.n2,
        n3=base.n3,
        n4=base.n4,
        n5=base.n5,
        n6=LexicalAccessN6Adapter(lexical_access, language=language),
        n7=base.n7 or DiscourseN7Adapter(discourse),
        n8=SemanticCompositionN8Adapter(semantic_composition, language=language),
        syntax=SyntaxN9Adapter(syntax, language=language),
    )


__all__ = [
    "LexicalAccessN6Adapter",
    "SemanticCompositionN8Adapter",
    "SyntaxN9Adapter",
    "build_n9_language_adapters",
]
