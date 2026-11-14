"""Adapters from existing ORE results into the reading request contract."""
from __future__ import annotations

from typing import Any, Mapping

from .contracts import ReadingRequest


def reading_request_from_n6_result(
    result: Any,
    *,
    language: str = "und",
    reading_direction: str = "ltr",
    source_reference: str | None = None,
) -> ReadingRequest:
    """Create a reading request from an ore_n6.N6Result-like object or dict.

    N6 remains responsible for grapheme inference and orthographic segmentation;
    this adapter passes its token orthographies downstream without modifying it.
    """
    payload: Mapping[str, Any]
    if isinstance(result, Mapping):
        payload = result
    elif hasattr(result, "to_dict"):
        payload = result.to_dict()
    else:
        payload = {
            "sequence_id": getattr(result, "sequence_id", None),
            "tokens": getattr(result, "tokens", None),
            "metadata": getattr(result, "metadata", {}),
        }
    sequence_id = payload.get("sequence_id")
    token_records = payload.get("tokens")
    if not isinstance(sequence_id, str) or not sequence_id.strip():
        raise ValueError("N6 result must contain a non-empty sequence_id")
    if not isinstance(token_records, (list, tuple)) or not token_records:
        raise ValueError("N6 result must contain at least one lexical token")
    tokens: list[str] = []
    for index, token in enumerate(token_records):
        if not isinstance(token, Mapping):
            raise TypeError(f"N6 token {index} must be a mapping")
        orthography = token.get("orthography")
        if not isinstance(orthography, str) or not orthography.strip():
            graphemes = token.get("graphemes")
            if isinstance(graphemes, (list, tuple)) and graphemes:
                orthography = "".join(str(item) for item in graphemes)
        if not isinstance(orthography, str) or not orthography.strip():
            raise ValueError(f"N6 token {index} has no usable orthography")
        tokens.append(orthography)
    metadata = payload.get("metadata") or {}
    inferred_source = source_reference or metadata.get("schema") or "ore_n6_result"
    return ReadingRequest(
        sequence_id=sequence_id,
        tokens=tokens,
        language=language,
        reading_direction=reading_direction,
        source_reference=str(inferred_source),
    )
