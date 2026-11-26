"""Contract tests: canonical artifacts, digests, serialization (spec 2.2, Section 9)."""
from __future__ import annotations

import json

import pytest

from ore_hic import (
    CompilationSession,
    InputModality,
    InputRecord,
    QualityFlag,
    canonical_json,
    digest_payload,
    digest_text,
)
from ore_hic.contracts import HIC_SCHEMA_VERSION


def test_canonical_json_is_deterministic():
    payload = {"b": 1, "a": [3, 2], "nested": {"z": True, "y": None}}
    assert canonical_json(payload) == canonical_json(json.loads(json.dumps(payload)))


def test_digest_is_stable_and_content_sensitive():
    assert digest_payload({"a": 1}) == digest_payload({"a": 1})
    assert digest_payload({"a": 1}) != digest_payload({"a": 2})


def test_input_record_preserves_content_byte_for_byte():
    session = CompilationSession(user="u")
    record = session.create_input_record(
        content="Assign each task to exactly one module.",
        modality=InputModality.TEXT,
        source_identity="test",
    )
    assert record.content == "Assign each task to exactly one module."
    assert record.integrity_digest == digest_text(record.content)


def test_input_record_serialization_round_trip():
    session = CompilationSession(user="u")
    record = session.create_input_record(
        content="idea text",
        modality=InputModality.SPEECH_TRANSCRIPT,
        source_identity="mic-1",
        transcription_confidence=0.7,
        quality_flags=[QualityFlag.LOW_TRANSCRIPTION_CONFIDENCE],
    )
    restored = InputRecord.from_dict(json.loads(record.to_json()))
    assert restored.content == record.content
    assert restored.modality is InputModality.SPEECH_TRANSCRIPT
    assert restored.transcription_confidence == 0.7
    assert QualityFlag.LOW_TRANSCRIPTION_CONFIDENCE in restored.quality_flags
    assert restored.integrity_digest == record.integrity_digest


def test_speech_transcript_requires_confidence():
    from ore_hic import HICError, SpeechTranscriptAdapter
    from ore_hic.session import CompilationSession

    session = CompilationSession(user="u")
    adapter = SpeechTranscriptAdapter()
    with pytest.raises(HICError):
        adapter.capture(session, "hello", source_identity="mic")


def test_low_confidence_transcripts_keep_their_uncertainty():
    from ore_hic import SpeechTranscriptAdapter

    session = CompilationSession(user="u")
    adapter = SpeechTranscriptAdapter()
    record = adapter.capture(session, "hello", source_identity="mic", transcription_confidence=0.5)
    # Uncertain extraction is represented, never silently discarded.
    assert QualityFlag.LOW_TRANSCRIPTION_CONFIDENCE in record.quality_flags


def test_schema_version_is_declared():
    session = CompilationSession(user="u")
    record = session.create_input_record(content="x", modality=InputModality.TEXT, source_identity="t")
    assert record.to_dict()["schema_version"] == HIC_SCHEMA_VERSION


def test_span_validation_rejects_bad_ranges():
    session = CompilationSession(user="u")
    record = session.create_input_record(content="abcdef", modality=InputModality.TEXT, source_identity="t")
    with pytest.raises(ValueError):
        record.span(3, 2)
    with pytest.raises(ValueError):
        record.span(0, 99)


def test_empty_input_record_is_rejected():
    session = CompilationSession(user="u")
    with pytest.raises(ValueError):
        session.create_input_record(content="", modality=InputModality.TEXT, source_identity="t")


def test_semantic_digest_ignores_housekeeping_ids():
    """Two artifacts with equal semantic content share a semantic digest."""
    from ore_hic.contracts import SourceSpan
    from ore_hic.semantic import ExtractedConcept, SemanticGraph

    span = SourceSpan("in-1", 0, 5, "tasks")
    concept = ExtractedConcept("USG-TASK", "task", None, span, 0.9)
    graph_a = SemanticGraph(
        artifact_id="sg-a", record_id="in-a", concepts=[concept], relations=[],
        attributes=[], interpretations=[], dangling_references=(), usg_backend="x",
    )
    graph_b = SemanticGraph(
        artifact_id="sg-b", record_id="in-b", concepts=[concept], relations=[],
        attributes=[], interpretations=[], dangling_references=(), usg_backend="x",
    )
    assert graph_a.semantic_digest == graph_b.semantic_digest
    assert graph_a.content_digest != graph_b.content_digest
