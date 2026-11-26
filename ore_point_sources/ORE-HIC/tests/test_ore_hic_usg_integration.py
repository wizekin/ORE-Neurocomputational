"""USG adapter and ORE integration tests."""
from __future__ import annotations

import pytest

from ore_hic import HICUSGAdapter, StarterUSGRegistry


def test_adapter_resolves_built_in_concepts_with_rules():
    usg = HICUSGAdapter()
    reference = usg.resolve_concept("task")
    assert reference is not None
    assert reference.concept_id == "USG-TASK"
    assert reference.rule_id == "HIC-USG-R01"
    assert reference.mapping_type == "EQUIVALENT"


def test_adapter_resolves_plural_forms():
    usg = HICUSGAdapter()
    assert usg.resolve_concept("tasks") is not None
    assert usg.resolve_concept("modules") is not None
    assert usg.resolve_concept("capacities") is not None


def test_unresolved_words_stay_unresolved():
    usg = HICUSGAdapter()
    assert usg.resolve_concept("flibberwock") is None


def test_predicate_frames_are_available():
    usg = HICUSGAdapter()
    frame = usg.resolve_frame("assigns-to")
    assert frame is not None
    assert "agent" in frame.roles and "target" in frame.roles
    assert usg.resolve_frame("made-up-predicate") is None


def test_rule_registry_is_inspectable():
    usg = HICUSGAdapter()
    from ore_hic.usg_adapter import USGLookupError

    rule = usg.rule("HIC-USG-R02")
    assert "never new concepts" in rule.statement
    with pytest.raises(USGLookupError):
        usg.rule("HIC-USG-NOPE")


def test_external_index_is_validated():
    from ore_hic.usg_adapter import USGLookupError

    with pytest.raises(USGLookupError):
        HICUSGAdapter(external_index=object())  # no lookup method


def test_ore_usg_bridge_uses_knowledge_index_when_available():
    """When ore_usg is importable, lookups delegate to USGKnowledgeIndex."""
    pytest.importorskip("ore_usg")

    class FakeItem:
        record_id = "ore:concept:123"
        lifecycle_state = "CANDIDATE"

    class FakeResult:
        items = (FakeItem(),)

    class FakeIndex:
        def lookup(self, query, kind=None, language=None):
            assert kind == "SEMANTIC_CONCEPT"
            return FakeResult()

    usg = HICUSGAdapter(external_index=FakeIndex())
    assert usg.backend.startswith("ore_usg+")
    # Built-in registry still resolves directly...
    assert usg.resolve_concept("task").concept_id == "USG-TASK"
    # ...and unknown words become hypothesis mappings via the external index.
    reference = usg.resolve_concept("flibberwock")
    assert reference is not None
    assert reference.mapping_type == "RELATED_TO"  # hypothesis, not authoritative
    assert reference.rule_id == "HIC-USG-R07"
    assert reference.lifecycle_state == "CANDIDATE"


def test_starter_registry_classes_are_acyclic():
    registry = StarterUSGRegistry()
    for concept_id in registry.concepts:
        path = registry.class_path(concept_id)
        assert len(path) == len(set(path)), f"cycle in class path of {concept_id}"
