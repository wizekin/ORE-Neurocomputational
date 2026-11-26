"""Projection acceptance tests 1-8: typing, direction, qualifiers, provenance, integrity.

Covers ORE Point 8 acceptance tests:
1. Aliases resolve to one node only when USG identity confirms.
2. Reversing a directed edge changes the projection unless explicitly symmetric.
3. Relation types remain distinct.
4. Negation, modality, attribution, conditionals and time are preserved.
5. N-ary relations preserve all roles.
6. Every element has source ID and snapshot version.
7. Identical snapshot/config yields a reproducible graph hash.
8. Dangling edges are quarantined/reported.
"""
from __future__ import annotations

from copy import deepcopy

import pytest

from ore_graph_intelligence.projection import (
    IntegrityValidator,
    USGGraphProjector,
)
from ore_graph_intelligence.schemas.graph_schema import RelationSignature

from .fixtures import bundle_with_dangling_only, graph_bundle


@pytest.fixture()
def snapshot():
    return USGGraphProjector().project(graph_bundle())


def test_aliases_resolve_per_usg_identity(snapshot):
    """Test 1: two records sharing a surface label stay distinct nodes."""
    aliases = snapshot.node_id_by_alias("Money")
    assert len(aliases) == 1, "distinct concepts must not merge by surface similarity"
    labels = {node.label for node in snapshot.nodes}
    assert "Money" in labels and "River" in labels


def test_reversing_directed_edge_changes_projection(snapshot):
    """Test 2: reversing a directed edge yields a distinct triple set."""
    edge = next(edge for edge in snapshot.edges if edge.relation_type == "INSTANCE_OF")
    reversed_triple = (edge.target_id, edge.relation_type, edge.source_id)
    existing = snapshot.existing_triples()
    assert (edge.source_id, edge.relation_type, edge.target_id) in existing
    assert reversed_triple not in existing, "reversal must not project as the same fact"
    assert edge.reversed_signature() != (edge.source_id, edge.relation_type, edge.target_id)


def test_relation_types_remain_distinct(snapshot):
    """Test 3: DENOTES and INSTANCE_OF keep separate signatures."""
    denotes = snapshot.schema.signature("DENOTES")
    instance_of = snapshot.schema.signature("INSTANCE_OF")
    assert denotes is not None and instance_of is not None
    assert denotes.relation_type != instance_of.relation_type
    assert denotes.range_kinds != instance_of.range_kinds or denotes.domain_kinds != instance_of.domain_kinds


def test_logical_scope_preserved(snapshot):
    """Test 4: negation, modality and temporal scope survive projection."""
    negated = [e for e in snapshot.edges if e.qualifiers.polarity == "negative"]
    assert len(negated) == 1
    qualifiers = negated[0].qualifiers
    assert qualifiers.modality == "factual"
    assert qualifiers.temporal_scope == "2026"
    assert "polarity=negative" in qualifiers.scope_summary()


def test_nary_relations_preserve_all_roles(snapshot):
    """Test 5: the transfer relation reifies with AGENT/THEME/RECIPIENT role edges."""
    frames = [node for node in snapshot.nodes if node.node_kind == "PREDICATE_FRAME"]
    assert len(frames) == 1
    relation_events = [node for node in snapshot.nodes if node.node_kind == "RELATION_EVENT"]
    assert relation_events, "n-ary relations must reify as explicit relation/event nodes"
    event_roles = set()
    for edge in snapshot.edges:
        if edge.relation_type == "transfer":
            for role in edge.roles:
                event_roles.add(role.role)
    assert {"AGENT", "THEME", "RECIPIENT"} <= event_roles, (
        "every declared role of the n-ary relation must be preserved"
    )


def test_every_element_carries_provenance(snapshot):
    """Test 6: nodes carry source ids; the snapshot carries a digest and version."""
    for node in snapshot.nodes:
        assert node.provenance is not None
        assert node.provenance.source_ids, f"node {node.node_id} lacks source ids"
    assert snapshot.provenance.snapshot_hash
    assert snapshot.schema.schema_version
    for edge in snapshot.edges:
        assert edge.provenance is not None
        assert edge.evidence_ids is not None


def test_identical_snapshot_reproduces_graph_hash():
    """Test 7: identical bundle and scope produce the same graph hash."""
    projector = USGGraphProjector()
    first = projector.project(graph_bundle())
    second = projector.project(deepcopy(graph_bundle()))
    assert first.graph_hash == second.graph_hash


def test_dangling_edges_quarantined_and_reported():
    """Test 8: relations to non-admitted endpoints are quarantined, not dropped."""
    snap = USGGraphProjector().project(bundle_with_dangling_only())
    quarantined = {
        (record.record_id, record.reason) for record in snap.quarantine
    }
    assert any(
        record_id == "relation:money-part-of-restricted"
        for record_id, _reason in quarantined
    ), "dangling relation must be reported"
    report = IntegrityValidator().validate(snap)
    assert report.passed, "quarantined records must not break integrity"


def test_projection_never_mutates_source_bundle():
    """Read-only guarantee: the source records stay untouched by projection."""
    bundle = graph_bundle()
    original = deepcopy(bundle)
    USGGraphProjector().project(bundle)
    assert bundle == original


def test_relation_signature_rejects_unknown_cardinality():
    """Schema guards: malformed signatures cannot enter the graph schema."""
    with pytest.raises(ValueError):
        RelationSignature(
            relation_type="is_a",
            domain_kinds=frozenset({"CONCEPT"}),
            range_kinds=frozenset({"SEMANTIC_CLASS"}),
            cardinality="sometimes",
        )
