"""Acceptance tests for incremental invalidation and dependency-aware recompute."""
from __future__ import annotations

import random

import pytest

from ore_runtime import (
    DependencyGraph, DependencyRef, InvalidationEngine, SourceRecord,
)


def build_graph() -> tuple[DependencyGraph, InvalidationEngine]:
    """Build a two-source, three-artifact dependency graph for invalidation tests."""
    graph = DependencyGraph()
    graph.register_source("usg.word.walker", {"sense": "person"})
    graph.register_source("usg.rule.plural", {"form": "s"})
    engine = InvalidationEngine(graph)
    return graph, engine


def register_artifacts(graph: DependencyGraph) -> None:
    """Register leaf and derived artifacts over the two sources."""
    word = graph.source("usg.word.walker").as_dependency()
    rule = graph.source("usg.rule.plural").as_dependency()
    graph.register_artifact("lemma", (word,), rules=("lemmatize",))
    graph.register_source("lemma", {"lemma": "walk"})
    graph.register_artifact("inflection", (DependencyRef("lemma", 1), rule),
                            rules=("inflect",))
    graph.register_source("inflection", {"form": "walks"})
    graph.register_artifact("summary", (DependencyRef("inflection", 1),),
                            rules=("summarize",))


def test_changing_a_source_marks_only_relevant_downstream_stale():
    """Invalidation: a source change marks all and only relevant dependents stale."""
    graph, engine = build_graph()
    register_artifacts(graph)
    # declare the source change through its validated write contract
    graph.register_source("usg.rule.plural", {"form": "es"})
    report = engine.invalidate([DependencyRef("usg.rule.plural")])
    # the inflection artifact depends on the rule; summary depends on inflection.
    assert "inflection" in report.marked_stale
    assert "summary" in report.marked_stale
    # lemma does not depend on the changed rule and stays untouched.
    assert "lemma" not in report.marked_stale
    assert "lemma" not in report.candidate_artifacts
    assert graph.artifact("lemma").state.value == "current"


def test_stale_artifact_is_never_reported_current():
    """Stale conclusions are marked and never appear current until recomputed."""
    graph, engine = build_graph()
    register_artifacts(graph)
    graph.register_source("usg.rule.plural", {"form": "es"})
    engine.invalidate([DependencyRef("usg.rule.plural")])
    assert engine.is_current("inflection") is False
    assert graph.artifact("inflection").previous_hash
    assert graph.artifact("inflection").stale_reason


def test_unrelated_source_change_leaves_artifacts_fresh():
    """Changing an unrelated record does not touch unrelated artifacts."""
    graph, engine = build_graph()
    register_artifacts(graph)
    graph.register_source("usg.word.unrelated", {"sense": "x"})
    report = engine.invalidate([DependencyRef("usg.word.unrelated")])
    assert report.marked_stale == ()
    assert graph.artifact("summary").state.value == "current"


def test_recompute_plan_is_topologically_ordered():
    """Recompute plans order artifacts so dependencies recompute first."""
    graph, engine = build_graph()
    register_artifacts(graph)
    graph.register_source("usg.rule.plural", {"form": "es"})
    engine.invalidate([DependencyRef("usg.rule.plural")])
    levels = engine.recompute_plan(engine.stale_artifacts())
    flat = [artifact for level in levels for artifact in level]
    assert flat.index("inflection") < flat.index("summary")
    assert all(len(level) >= 1 for level in levels)


def test_incremental_recompute_matches_full_recompute():
    """Incremental correctness: incremental equals full recomputation."""
    rng = random.Random(1234)
    graph = DependencyGraph()
    engine = InvalidationEngine(graph)
    sources = [f"src.{i}" for i in range(6)]
    for source_id in sources:
        graph.register_source(source_id, {"value": rng.random()})

    def evaluate(dep_refs):
        """Simulate a deterministic derived computation over dependencies."""
        return {"sum": sum(
            graph.source(ref.record_id).payload["value"]
            for ref in dep_refs
        )}

    artifacts = {}
    for index in range(8):
        chosen = rng.sample(sources, k=3)
        deps = tuple(graph.source(item).as_dependency() for item in chosen)
        artifact_id = f"art.{index}"
        graph.register_artifact(artifact_id, deps, content=evaluate(deps))
        artifacts[artifact_id] = list(deps)

    full_before = {
        artifact_id: graph.artifact(artifact_id).content
        for artifact_id in artifacts
    }
    changed = sources[0]
    graph.register_source(changed, {"value": 999.0})
    report = engine.invalidate([changed])
    engine.recompute(
        report.marked_stale,
        lambda record: evaluate(
            [graph.source(dep.record_id).as_dependency() for dep in record.deps]
        ),
    )
    recomputed = {
        artifact_id: graph.artifact(artifact_id).content
        for artifact_id in report.marked_stale
    }
    # full recomputation over the same sources equals the incremental results
    for artifact_id in report.marked_stale:
        deps = [graph.source(dep.record_id) for dep in artifacts[artifact_id]]
        expected = {"sum": sum(item.payload["value"] for item in deps)}
        assert recomputed[artifact_id] == expected
    # unaffected artifacts keep their original content
    for artifact_id in report.unaffected:
        assert graph.artifact(artifact_id).content == full_before[artifact_id]


def test_artifact_hash_covers_inputs_versions_rules_and_config():
    """Artifact hashes include inputs, versions, rules, config, algorithm version."""
    from ore_runtime import compute_artifact_hash
    base = compute_artifact_hash(
        (DependencyRef("r", 1, "h"),), ("rule",), {"a": 1}, "1.0.0",
    )
    changed_input = compute_artifact_hash(
        (DependencyRef("r", 2, "h"),), ("rule",), {"a": 1}, "1.0.0",
    )
    changed_rule = compute_artifact_hash(
        (DependencyRef("r", 1, "h"),), ("other",), {"a": 1}, "1.0.0",
    )
    changed_config = compute_artifact_hash(
        (DependencyRef("r", 1, "h"),), ("rule",), {"a": 2}, "1.0.0",
    )
    changed_version = compute_artifact_hash(
        (DependencyRef("r", 1, "h"),), ("rule",), {"a": 1}, "2.0.0",
    )
    assert len({base, changed_input, changed_rule, changed_config, changed_version}) == 5


def test_source_validation_contract_rejects_invalid_payloads():
    """USG-style write contracts validate payloads before versioning them."""
    graph = DependencyGraph()
    graph.register_source(
        "validated", {"value": 1},
        validator=lambda payload: payload["value"] > 0,
    )
    with pytest.raises(ValueError):
        graph.register_source(
            "validated", {"value": -1},
            validator=lambda payload: payload["value"] > 0,
        )
    record = graph.source("validated")
    assert isinstance(record, SourceRecord)
    assert record.version == 1


def test_recompute_reuses_fresh_cache_when_hashes_match():
    """Fresh caches with matching dependency hashes are reused, not recomputed."""
    graph, engine = build_graph()
    register_artifacts(graph)
    recomputed: list[str] = []

    def recompute_fn(record):
        """Record each recomputation call."""
        recomputed.append(record.artifact_id)
        return {"recomputed": True}

    report = engine.recompute(["lemma"], recompute_fn)
    assert report.recomputed == ()
    assert report.reused == ("lemma",)
    assert recomputed == []
