"""Phase A tests: capture, modality, normalization, segmentation (Steps 1-7)."""
from __future__ import annotations

import pytest

from ore_hic import (
    CompilationSession,
    DeliverableKind,
    HICError,
    IdeaSegmenter,
    InputModality,
    Normalizer,
    TextInputAdapter,
    detect_modality,
)
from ore_hic.input import BCIStreamAdapter, FileSourceAdapter


def test_step1_session_has_stable_identity():
    session = CompilationSession(user="alice", project="p", workspace="w")
    assert session.session_id.startswith("sess-")
    same_again = CompilationSession(user="alice", project="p", workspace="w")
    assert same_again.session_id == session.session_id


def test_step2_raw_input_is_immutable_and_digested():
    session = CompilationSession(user="u")
    adapter = TextInputAdapter()
    raw = "Assign each task\r\nto exactly one module.   "
    record = adapter.capture(session, raw, source_identity="user-1")
    assert record.content == raw  # original preserved, including CRLF
    assert record.integrity_digest
    assert record.version == 1


def test_step3_modality_detection():
    assert detect_modality("Assign tasks to modules") is InputModality.TEXT
    assert detect_modality("def f():\n    import os\n    return 1") is InputModality.CODE
    assert detect_modality("a | b\n--- | ---\n1 | 2") is InputModality.TABLE
    assert detect_modality("hello", filename="diagram.png") is InputModality.DIAGRAM


def test_step4_normalization_is_a_derived_view_with_source_mapping():
    session = CompilationSession(user="u")
    adapter = TextInputAdapter()
    record = adapter.capture(session, "Assign  each task\r\nto one module.", source_identity="u")
    normalizer = Normalizer()
    view = normalizer.normalize(record)
    # The original is untouched; the view is derived.
    assert "\r" not in view.text
    assert "  " not in view.text
    assert record.content == "Assign  each task\r\nto one module."
    assert view.corrections
    assert view.mapping  # source mapping exists


def test_step5_segments_preserve_source_spans():
    session = CompilationSession(user="u")
    adapter = TextInputAdapter()
    text = "Assign each task to exactly one module. Modules must have capacity limits."
    record = adapter.capture(session, text, source_identity="u")
    view = Normalizer().normalize(record)
    segmentation = IdeaSegmenter().segment(record, view)
    assert segmentation
    for segment in segmentation:
        # Every fragment retains its original source span.
        assert segment.span.artifact_id == record.artifact_id
        assert 0 <= segment.span.start <= segment.span.end <= len(record.content)
        snippet = segment.text.split()[0].casefold()
        assert snippet in record.content.casefold()


def test_step6_deliverable_identification():
    session = CompilationSession(user="u")
    adapter = TextInputAdapter()
    record = adapter.capture(
        session,
        "I want an algorithm that assigns tasks. Please implement the module.",
        source_identity="u",
    )
    view = Normalizer().normalize(record)
    segmenter = IdeaSegmenter()
    segments = segmenter.segment(record, view)
    deliverable, rule = segmenter.identify_deliverable(segments)
    assert deliverable in {DeliverableKind.COMBINATION, DeliverableKind.IMPLEMENTATION, DeliverableKind.ALGORITHM}
    assert rule


def test_step6_conceptual_only_requests_do_not_generate_code_intent():
    segmenter = IdeaSegmenter()
    from ore_hic.contracts import SourceSpan

    span = SourceSpan("x", 0, 1)
    segments = [
        type("S", (), {"text": "explain how sorting works", "span": span})(),
    ]
    deliverable, _ = segmenter.identify_deliverable(segments)
    assert deliverable is DeliverableKind.EXPLANATION


def test_step7_absent_context_stays_unknown():
    session = CompilationSession(user="u")
    segmenter = IdeaSegmenter(known_context_keys=("repo.layout", "usg.schema.version"))
    facts = segmenter.resolve_context(session)
    by_key = {f.key: f for f in facts}
    assert by_key["repo.layout"].status == "UNKNOWN"
    assert by_key["repo.layout"].value == ""
    # Supplied facts are resolved with source and version.
    facts = segmenter.resolve_context(session, (("repo.layout", "src/", "project", "3"),))
    by_key = {f.key: f for f in facts}
    assert by_key["repo.layout"].status == "RESOLVED"
    assert by_key["repo.layout"].source == "project"


def test_bci_adapter_is_disabled_without_measured_signals():
    from ore_hic import GateViolation

    adapter = BCIStreamAdapter(decoder=None)
    assert adapter.enabled is False
    with pytest.raises(GateViolation):
        adapter.validate("command stream")


def test_bci_adapter_requires_callable_decoder():
    with pytest.raises(HICError):
        BCIStreamAdapter(decoder="not-callable")


def test_file_adapter_classifies_code_context():
    session = CompilationSession(user="u")
    adapter = FileSourceAdapter()
    record = adapter.capture(session, "def solve():\n    return 42\n", source_identity="repo")
    assert record.modality is InputModality.CODE
