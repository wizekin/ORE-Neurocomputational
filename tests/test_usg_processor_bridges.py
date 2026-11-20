"""Tests for direct bridges from USG resources to existing ORE processors."""
from __future__ import annotations

from dataclasses import replace

import pytest

from ore_usg import (
    USGKnowledgeAdapter,
    USGPredicateFrameConversionError,
    build_lexical_access_processor,
    build_semantic_composition_processor,
    to_ore_predicate_frame,
)
from test_usg_structure_adapter import _structure_bundle


def test_lexical_access_processor_uses_imported_ore_lexical_entries():
    resource = USGKnowledgeAdapter().convert(_structure_bundle())

    processor = build_lexical_access_processor(resource)

    assert processor.entries == resource.lexical.entries
    assert processor.by_form["bank"]
    assert processor.entries[0].source_reference.startswith("usg-knowledge://")


def test_semantic_composition_processor_receives_imported_concepts_and_frames():
    resource = USGKnowledgeAdapter().convert(_structure_bundle())

    processor = build_semantic_composition_processor(resource)

    assert processor.concepts == resource.semantic.concepts
    assert len(processor.frames) == 1
    assert processor.frames[0].predicate == "transfer"
    assert processor.frames[0].roles == ("AGENT", "THEME", "RECIPIENT")
    assert processor.frames[0].source_reference.startswith("usg-knowledge://")


def test_predicate_frame_conversion_rejects_uninterpretable_role_shapes():
    resource = USGKnowledgeAdapter().convert(_structure_bundle())
    frame = resource.structures.predicate_frames[0]
    malformed = replace(frame, roles_json='{"AGENT": true}')

    with pytest.raises(USGPredicateFrameConversionError, match="roles array"):
        to_ore_predicate_frame(malformed)
