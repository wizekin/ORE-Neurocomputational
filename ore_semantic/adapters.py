"""Adapters for the existing ORE reading and N9 integration contracts."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from .contracts import SemanticRequest
from .pipeline import SemanticCognitionProcessor


def _token_from_record(record: Any) -> str | None:
    if isinstance(record, str) and record.strip():
        return record
    if isinstance(record, Mapping):
        for key in ("orthography", "form", "surface", "word", "text", "token"):
            value = record.get(key)
            if isinstance(value, str) and value.strip():
                return value
        graphemes = record.get("graphemes")
        if isinstance(graphemes, (list, tuple)) and graphemes:
            value = "".join(str(item) for item in graphemes)
            return value if value.strip() else None
    return None


def semantic_request_from_reading_result(
    result: Any,
    *,
    language: str = "und",
    task: str = "comprehension",
    source_reference: str | None = None,
) -> SemanticRequest:
    """Adapt an ore_reading result/dict to semantic cognition without mutation."""
    if isinstance(result, Mapping):
        payload = result
    elif hasattr(result, "to_dict") and callable(result.to_dict):
        payload = result.to_dict()
    else:
        payload = {
            "sequence_id": getattr(result, "sequence_id", None),
            "tokens": getattr(result, "tokens", None),
            "normalized_text": getattr(result, "normalized_text", None),
        }

    sequence_id = payload.get("sequence_id")
    if not isinstance(sequence_id, str) or not sequence_id.strip():
        raise ValueError("reading result must contain a non-empty sequence_id")
    token_records = payload.get("tokens")
    tokens: list[str] = []
    if isinstance(token_records, (list, tuple)):
        for record in token_records:
            token = _token_from_record(record)
            if token:
                tokens.append(token)
    if not tokens:
        normalized = payload.get("normalized_text")
        if isinstance(normalized, str):
            tokens = [part for part in normalized.split() if part]
    if not tokens:
        raise ValueError("reading result must contain at least one usable token")
    return SemanticRequest(
        sequence_id=sequence_id,
        tokens=tokens,
        language=language,
        task=task,
        source_reference=source_reference or "ore_reading_result",
    )


class SemanticN8Adapter:
    """Expose semantic cognition through the existing ore_n9 N8 adapter contract.

    N9 expects a mapping with semantic_graph, propositions, completeness and
    ambiguity. This adapter returns concept nodes only: it deliberately does not
    fabricate predicates, thematic roles, syntax, or relation edges.
    """

    def __init__(self, processor: SemanticCognitionProcessor) -> None:
        self.processor = processor

    def compose(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        sequence_id = payload.get("sequence_id")
        if not isinstance(sequence_id, str) or not sequence_id.strip():
            sequence_id = "ore-n9-semantic-adapter"

        raw_hypotheses = payload.get("lexical_hypotheses", ())
        tokens: list[str] = []
        if isinstance(raw_hypotheses, (list, tuple)):
            for item in raw_hypotheses:
                token = _token_from_record(item)
                if token:
                    tokens.append(token)
        if not tokens:
            explicit_tokens = payload.get("tokens")
            if isinstance(explicit_tokens, (list, tuple)):
                tokens = [token for record in explicit_tokens
                          if (token := _token_from_record(record))]
        if not tokens:
            text = payload.get("text")
            if isinstance(text, str) and text.strip():
                tokens = [text]
        if not tokens:
            # The adapter must not invent a word when upstream lexical access failed.
            return {
                "semantic_graph": {"nodes": [], "edges": []},
                "propositions": [],
                "completeness": 0.0,
                "ambiguity": 1.0,
                "semantic_result": None,
                "status": "unavailable_no_lexical_hypotheses",
            }

        context_snapshot = payload.get("context", {})
        semantic_context: Mapping[str, float] = {}
        if isinstance(context_snapshot, Mapping):
            candidate_context = context_snapshot.get("semantic_features", context_snapshot.get("features", {}))
            if isinstance(candidate_context, Mapping):
                semantic_context = candidate_context

        request = SemanticRequest(
            sequence_id=sequence_id,
            tokens=tokens,
            language=str(payload.get("language", "und")),
            task=str(payload.get("task", "comprehension")),
            context=semantic_context,
            source_reference="ore_n9.N8.SemanticN8Adapter",
        )
        result = self.processor.process(request)
        result_payload = result.to_dict()

        nodes_by_id: dict[str, dict[str, Any]] = {}
        for token in result.tokens:
            for candidate in token.get("candidates", []):
                concept_id = candidate.get("concept_id")
                if not isinstance(concept_id, str) or not concept_id:
                    continue
                nodes_by_id.setdefault(concept_id, {
                    "id": concept_id,
                    "type": "CONCEPT",
                    "label": candidate.get("label", concept_id),
                    "lexical_category": candidate.get("lexical_category", "unknown"),
                    "source_reference": candidate.get("source_reference", "unspecified"),
                })

        token_count = max(len(result.tokens), 1)
        completeness = float(result.diagnostics["resolved_token_count"]) / token_count
        ambiguity_values = [
            float(item.get("ambiguity", 1.0)) / max(
                1.0, __import__("math").log(max(len(item.get("candidates", [])), 2))
            )
            for item in result.tokens
        ]
        ambiguity = sum(ambiguity_values) / max(len(ambiguity_values), 1)
        return {
            "semantic_graph": {
                "nodes": list(nodes_by_id.values()),
                "edges": [],
            },
            "propositions": [],
            "completeness": float(max(0.0, min(1.0, completeness))),
            "ambiguity": float(max(0.0, min(1.0, ambiguity))),
            "semantic_result": result_payload,
            "status": "concept_access_only",
        }
