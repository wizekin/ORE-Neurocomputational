"""Adapters for passing existing ORE syntax and semantic outputs into composition."""
from __future__ import annotations

from typing import Any, Mapping

from .contracts import SemanticCompositionRequest


def _payload(value: Any) -> dict[str, Any]:
    """Convert an ORE result object or mapping into an ordinary dictionary."""
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "to_dict"):
        result = value.to_dict()
        if isinstance(result, Mapping):
            return dict(result)
    raise TypeError("upstream result must be a mapping or expose to_dict()")


def composition_request_from_results(
    sequence_id: str,
    syntax_result: Any,
    semantic_result: Any | None = None,
    context_features: Mapping[str, float] | None = None,
    language: str = "en",
) -> SemanticCompositionRequest:
    """Build a composition request from optional ore_syntax and ore_semantic results.

    The syntax result supplies the ordered token sequence and dependency structure.
    When a semantic result is supplied, its selected concept IDs and candidate IDs
    are copied into a token-indexed hypothesis constraint; the concept inventory
    itself remains the caller's responsibility and is not inferred from a result.
    """
    syntax = _payload(syntax_result)
    tokens = syntax.get("tokens")
    if not tokens:
        raise ValueError("syntax_result must contain a non-empty tokens sequence")
    hypotheses: dict[int, list[str]] = {}
    if semantic_result is not None:
        semantic = _payload(semantic_result)
        for index, row in enumerate(semantic.get("tokens", ()) or ()):
            ids: list[str] = []
            selected = row.get("selected_concept_id") or row.get("selected_concept")
            if isinstance(selected, Mapping):
                selected = selected.get("concept_id")
            if isinstance(selected, str) and selected:
                ids.append(selected)
            for candidate in row.get("candidates", ()) or ():
                concept_id = candidate.get("concept_id") if isinstance(candidate, Mapping) else None
                if concept_id and concept_id not in ids:
                    ids.append(str(concept_id))
            if ids:
                hypotheses[index] = ids
    return SemanticCompositionRequest(
        sequence_id=sequence_id,
        tokens=tuple(str(token) for token in tokens),
        language=language,
        concept_hypotheses=hypotheses,
        syntax_result=syntax,
        context_features=context_features or {},
        source_reference="adapter:ore_syntax+ore_semantic",
    )
