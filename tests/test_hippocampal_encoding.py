"""Functional, biological-surrogate, registry, and SQLite annotation tests for Point 01."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from ore_hippocampal_encoding import (
    EncodingConfig,
    EventInput,
    HippocampalEncodingProcessor,
    OPERATION_REGISTRY,
    register_annotations,
)
from ore_hippocampal_encoding.biology import calcium_threshold_plasticity
from ore_hippocampal_encoding.functional import (
    apply_attention_gain,
    cosine_similarity,
    normalize_vector,
    softmax,
    stdp_weight_delta,
)


def _event(
    event_id: str,
    timestamp_ms: float,
    visual: tuple[float, ...] = (1.0, 0.2, 0.1),
    scene: tuple[float, ...] = (0.1, 0.8, 0.2),
) -> EventInput:
    """Create a small deterministic multi-channel episode with explicit entity relations."""
    return EventInput(
        event_id=event_id,
        sensory_signals={"visual_object": visual, "scene_context": scene,
                         "auditory_speech": (0.2, 0.1, 0.9)},
        context=(0.2, 0.3, 0.7),
        timestamp_ms=timestamp_ms,
        components=(
            {"id": "person-1", "kind": "person", "features": (0.8, 0.1, 0.2)},
            {"id": "object-1", "kind": "object", "features": (0.1, 0.9, 0.3)},
        ),
        relations=({"source": "person-1", "type": "looked_at", "target": "object-1"},),
        goals={"visual_object": 0.8},
        salience={"auditory_speech": 0.6},
        uncertainty={"scene_context": 0.2},
        modulatory_state={"acetylcholine": 0.6, "noradrenaline": 0.7, "dopamine": 0.5},
    )


def test_registry_contains_the_source_ordered_fifteen_operations() -> None:
    """Protect the exact Point 01 sequence and expose evidence levels and equations."""
    assert len(OPERATION_REGISTRY) == 15
    assert [item.operation_id for item in OPERATION_REGISTRY] == [f"HE{i:02d}" for i in range(1, 16)]
    assert all(item.mathematical_form and item.biological_reference and item.evidence_level
               for item in OPERATION_REGISTRY)


def test_vector_and_attention_math_is_finite_and_normalized() -> None:
    """Verify L2 normalization, cosine similarity, and softmax attention invariants."""
    unit = normalize_vector([3.0, 4.0])
    assert np.isclose(np.linalg.norm(unit), 1.0)
    assert np.allclose(normalize_vector([0.0, 0.0]), [0.0, 0.0])
    assert np.isclose(cosine_similarity([1.0, 0.0], [2.0, 0.0]), 1.0)
    weights = softmax([1.0, 2.0, 3.0], temperature=0.7)
    assert np.isclose(float(np.sum(weights)), 1.0)
    _, attended = apply_attention_gain(
        {"visual": np.array([1.0, 0.0]), "auditory": np.array([0.0, 1.0])},
        {"visual": 1.0}, {"auditory": 0.4}, {"visual": 0.1}, 0.8,
    )
    assert all(np.all(np.isfinite(value)) for value in attended.values())


def test_calcium_threshold_plasticity_selects_ltp_ltd_or_no_change() -> None:
    """Exercise all three branches of the simplified local calcium plasticity rule."""
    weights = np.zeros((2, 2), dtype=np.float64)
    activity = [0.5, 1.0]
    potentiated, ltp = calcium_threshold_plasticity(weights, activity, activity, 0.8, 0.1, 0.2, 0.6)
    depressed, ltd = calcium_threshold_plasticity(np.ones((2, 2)), activity, activity, 0.4, 0.1, 0.2, 0.6)
    unchanged, low = calcium_threshold_plasticity(weights, activity, activity, 0.1, 0.1, 0.2, 0.6)
    assert ltp["direction"] == "potentiation" and np.all(potentiated >= 0.0)
    assert ltd["direction"] == "depression" and float(np.mean(depressed)) < 1.0
    assert low["direction"] == "no_hebbian_change" and np.allclose(unchanged, weights)


def test_stdp_teaching_rule_has_timing_dependent_sign() -> None:
    """Potentiate for pre-before-post pairs and depress for post-before-pre pairs."""
    assert stdp_weight_delta([0.0], [5.0]) > 0.0
    assert stdp_weight_delta([5.0], [0.0]) < 0.0


def test_integrated_pipeline_emits_all_stages_and_biological_population_states() -> None:
    """Run the complete encoding pipeline and inspect intermediate functional and DG/CA3/CA1 outputs."""
    processor = HippocampalEncodingProcessor(EncodingConfig(random_seed=11))
    result = processor.process(_event("episode-001", 100.0))
    assert len(result.operation_trace) == 15
    assert [row["operation_id"] for row in result.operation_trace] == [
        item.operation_id for item in OPERATION_REGISTRY
    ]
    assert len(result.item_representation) == 32
    assert len(result.context_representation) == 32
    assert len(result.temporal_representation) == 32
    assert len(result.entorhinal_representation) == 96
    assert len(result.dentate_activity) == 96
    assert len(result.ca3_activity) == 64
    assert len(result.ca1_activity) == 64
    assert np.count_nonzero(result.dentate_activity) <= 12
    assert result.novelty["score"] == 1.0
    assert result.event_index["event_id"] == "episode-001"
    assert any(edge["type"] == "looked_at" for edge in result.event_index["edges"])
    assert result.trace_quality["persistence_claim"] == "not_inferred"
    assert result.metadata["operation_count"] == 15
    assert result.biological["sleep_replay_and_slow_systems_consolidation"] == "out_of_scope_for_point_01"
    assert np.all(np.isfinite(result.ca1_activity))


def test_repeated_event_is_less_novel_and_state_is_recurrently_updated() -> None:
    """Compare later input to the saved event code without treating initial encoding as consolidation."""
    processor = HippocampalEncodingProcessor()
    first = processor.process(_event("episode-a", 100.0))
    second = processor.process(_event("episode-b", 100.0))
    assert second.novelty["score"] < first.novelty["score"]
    assert second.novelty["nearest_episode_id"] == "episode-a"
    assert second.event_boundary["reason"] in {
        "change_score_exceeded_threshold", "event_continuity_candidate"
    }
    assert len(second.cortical_feedback) == 64


def test_event_contract_rejects_nonfinite_inputs() -> None:
    """Reject invalid numeric observations before any internal neural state is touched."""
    try:
        EventInput(event_id="bad", sensory_signals={"vision": [1.0, float("nan")]})
    except ValueError as error:
        assert "finite" in str(error)
    else:
        raise AssertionError("non-finite sensory data must be rejected")


def test_every_package_class_and_function_has_a_docstring() -> None:
    """Require every class and function, including private helpers and methods, to have an annotation."""
    package_root = Path(__file__).resolve().parents[1] / "ore_hippocampal_encoding"
    for source_path in package_root.glob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                assert ast.get_docstring(node), f"missing annotation docstring: {source_path.name}:{node.name}"


def test_all_package_annotations_are_persisted_in_logger_sqlite(tmp_path: Path) -> None:
    """Index the package and verify every AST class/function resolves to a SQLite annotation row."""
    root = Path(__file__).resolve().parents[1]
    database_path = tmp_path / "point01-annotations.sqlite3"
    report = register_annotations(database_path, root)
    assert report["parse_errors"] == 0
    from ore_logger.database import LoggerDatabase

    database = LoggerDatabase(database_path)
    for source_path in (root / "ore_hippocampal_encoding").glob("*.py"):
        module_path = source_path.relative_to(root).with_suffix("")
        if module_path.name == "__init__":
            module_path = module_path.parent
        module = module_path.as_posix().replace("/", ".")
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))

        def check_body(body: list[ast.stmt], parent: str = "") -> None:
            """Check nested source definitions against the logger's exact qualified-name convention."""
            for node in body:
                if isinstance(node, ast.ClassDef):
                    qualname = f"{parent}.{node.name}" if parent else node.name
                    row = database.get_annotation(module, qualname, "class", source_path.relative_to(root).as_posix())
                    assert row is not None and row["annotation"].strip()
                    assert row["annotation_source"] == "docstring"
                    check_body(node.body, qualname)
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    qualname = f"{parent}.{node.name}" if parent else node.name
                    kind = "async_function" if isinstance(node, ast.AsyncFunctionDef) else "function"
                    row = database.get_annotation(module, qualname, kind, source_path.relative_to(root).as_posix())
                    assert row is not None and row["annotation"].strip()
                    assert row["annotation_source"] == "docstring"
                    check_body(node.body, qualname + ".<locals>")
                else:
                    for child in ast.iter_child_nodes(node):
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                            check_body([child], parent)

        check_body(tree.body)
    assert database.summary()["symbols"] >= 30


def test_pipeline_records_function_annotations_and_fifteen_operations_in_sqlite(tmp_path: Path) -> None:
    """Verify the integrated run recorder stores class/function annotations and operation events."""
    root = Path(__file__).resolve().parents[1]
    database_path = tmp_path / "point01-run.sqlite3"
    processor = HippocampalEncodingProcessor(database_path=database_path, repository_root=root)
    result = processor.process(_event("sqlite-episode", 250.0))
    assert result.metadata["sqlite_logger"]["enabled"] is True

    from ore_logger.database import LoggerDatabase

    database = LoggerDatabase(database_path)
    runs = database.list_runs()
    assert len(runs) == 1 and runs[0]["status"] == "completed"
    annotations = database.search_annotations("HippocampalEncodingProcessor", limit=20)
    assert any(row["symbol_kind"] == "class" for row in annotations)
    assert any(row["symbol_kind"] == "function" and row["qualified_name"] == "HippocampalEncodingProcessor.process"
               for row in annotations)
    events = database.list_events(runs[0]["run_id"])
    operations = [row for row in events if row["event_kind"] == "operation"]
    assert len(operations) == 15
    assert all(row.get("annotation") for row in operations)
