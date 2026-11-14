"""Adapters preserve evidence when connecting existing ORE stages."""
from __future__ import annotations
from typing import Any, Mapping
from .contracts import LexicalContextRequest

def _token_from_record(record: Any) -> str | None:
    if isinstance(record,str) and record.strip(): return record
    if isinstance(record,Mapping):
        for key in ("orthography","form","surface","word","text","token"):
            value=record.get(key)
            if isinstance(value,str) and value.strip(): return value
    return None

def lexical_context_request_from_reading_result(result: Any, *, language: str="und",
        context_features: Mapping[str,float] | None=None,
        source_reference: str="ore_reading_result") -> LexicalContextRequest:
    """Adapt an ore_reading result/dict without changing or inventing tokens."""
    if isinstance(result,Mapping): payload=result
    elif hasattr(result,"to_dict") and callable(result.to_dict): payload=result.to_dict()
    else:
        payload={"sequence_id":getattr(result,"sequence_id",None),
                 "tokens":getattr(result,"tokens",None),
                 "normalized_text":getattr(result,"normalized_text",None)}
    sequence_id=payload.get("sequence_id")
    if not isinstance(sequence_id,str) or not sequence_id.strip():
        raise ValueError("reading result must contain a non-empty sequence_id")
    tokens=[t for record in (payload.get("tokens") or ()) if (t:=_token_from_record(record)) is not None]
    if not tokens:
        text=payload.get("normalized_text")
        if isinstance(text,str): tokens=[part for part in text.split() if part]
    if not tokens: raise ValueError("reading result must contain at least one usable token")
    return LexicalContextRequest(sequence_id=sequence_id,tokens=tokens,language=language,
        context_features=dict(context_features or {}),source_reference=source_reference)
