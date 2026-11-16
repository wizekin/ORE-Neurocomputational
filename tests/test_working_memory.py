"""Software-contract tests for ORE's working-memory and cognitive-control subsystem."""

from __future__ import annotations

import ast
from pathlib import Path

from ore_logger.catalog import sync_repository
from ore_logger.database import LoggerDatabase
from ore_working_memory import (
    OPERATION_REGISTRY, STAGE_REGISTRY, WorkingMemoryConfig,
    WorkingMemoryProcessor, WorkingMemoryRequest,
)


def _no_biology(**overrides):
    """Build a deterministic processor without the optional neural-reference simulation."""
    values = {"simulate_biology": False}
    values.update(overrides)
    return WorkingMemoryProcessor(WorkingMemoryConfig(**values))


def test_registry_and_trace_cover_the_canonical_forty_operations() -> None:
    """Check stable operation ordering, stage distribution, and biological-step correspondence."""
    request = WorkingMemoryRequest(
        sequence_id="trace-001", text="The student read the sentence carefully.",
    )
    result = WorkingMemoryProcessor().process(request)
    assert len(STAGE_REGISTRY) == 5
    assert [stage.operation_count for stage in STAGE_REGISTRY] == [8, 6, 7, 9, 10]
    assert len(OPERATION_REGISTRY) == 40
    assert len(result.operation_trace) == 40
    assert [row["operation_id"] for row in result.operation_trace] == [
        f"WMCC-{index:03d}" for index in range(1, 41)
    ]
    assert [row["operation_name"] for row in result.operation_trace] == [
        item.operation_name for item in OPERATION_REGISTRY
    ]
    assert result.biological["operation_count"] == 40
    assert result.biological["units"]["voltage"] == "mV"
    assert result.diagnostics["valid"] is True


def test_linguistic_buffers_are_separate_bounded_and_rehearsed() -> None:
    """Verify modality-specific buffers, capacity limits, activation values, and rehearsal traces."""
    request = WorkingMemoryRequest(
        sequence_id="buffers-001",
        text="The small researcher carefully examined the complicated sentence structure.",
    )
    result = _no_biology(buffer_capacity=3).process(request)
    assert set(result.buffers) == {
        "phonological", "lexical_semantic", "syntactic", "event_discourse", "executive_goal",
    }
    assert all(len(items) <= 3 for items in result.buffers.values())
    assert any(item["rehearsal_count"] > 0 for items in result.buffers.values() for item in items)
    assert all(
        0.0 <= item["activation"] <= 1.0
        for items in result.buffers.values() for item in items
    )


def test_incomplete_syntactic_slots_are_maintained_and_only_explicit_evidence_binds() -> None:
    """Preserve an unresolved relative-clause slot unless a matching explicit binding is supplied."""
    open_request = WorkingMemoryRequest(
        sequence_id="syntax-open-001",
        text="The book that the student borrowed",
        partial_structures=[{
            "structure_id": "relative-1", "label": "relative clause",
            "expected_slots": ["object"],
        }],
    )
    open_result = _no_biology().process(open_request)
    assert open_result.incomplete_structures["unresolved_slot_count"] == 1
    assert open_result.incomplete_structures["complete"] is False

    bound_request = WorkingMemoryRequest(
        sequence_id="syntax-bound-001",
        text="The book that the student borrowed",
        partial_structures=[{
            "structure_id": "relative-1", "label": "relative clause",
            "expected_slots": ["object"],
            "candidate_bindings": [{
                "slot": "object", "value": "the book", "confidence": 0.95,
            }],
        }],
    )
    bound_result = _no_biology().process(bound_request)
    assert bound_result.incomplete_structures["unresolved_slot_count"] == 0
    assert bound_result.incomplete_structures["complete"] is True


def test_unresolved_long_distance_dependency_is_preserved() -> None:
    """Retain an upstream unresolved filler-gap dependency without inventing its gap site."""
    request = WorkingMemoryRequest(
        sequence_id="dependency-001",
        text="Which article did the editor say that the author revised",
        syntax_result={
            "long_distance_dependencies": [{
                "dependency_id": "wh-1", "filler": "Which article",
                "expected_role": "object", "unresolved": True,
            }]
        },
    )
    result = _no_biology().process(request)
    assert result.incomplete_structures["pending_dependencies"][0]["dependency_id"] == "wh-1"
    assert result.incomplete_structures["complete"] is False


def test_semantic_competition_preserves_alternatives_and_abstains_when_ambiguous() -> None:
    """Keep two close readings visible and abstain instead of forcing an unsupported decision."""
    request = WorkingMemoryRequest(
        sequence_id="ambiguity-001", text="I saw her duck.",
        interpretations=[
            {
                "hypothesis_id": "duck-bird", "interpretation": "I saw the duck belonging to her",
                "lexical_score": 0.78, "semantic_score": 0.77, "syntactic_score": 0.76,
                "discourse_score": 0.55, "world_score": 0.65, "prior": 0.5,
            },
            {
                "hypothesis_id": "duck-action", "interpretation": "I saw her lower her head",
                "lexical_score": 0.77, "semantic_score": 0.78, "syntactic_score": 0.76,
                "discourse_score": 0.55, "world_score": 0.65, "prior": 0.5,
            },
        ],
    )
    result = _no_biology(
        semantic_temperature=0.9, confidence_threshold=0.8, ambiguity_margin=0.2,
    ).process(request)
    assert result.semantic_resolution["abstained"] is True
    assert result.semantic_resolution["selected"] is None
    assert len(result.semantic_resolution["alternatives"]) == 2
    assert abs(sum(item["posterior"] for item in result.semantic_resolution["alternatives"]) - 1.0) < 1e-8


def test_executive_control_retains_goal_bias_separately_from_evidence() -> None:
    """Verify explicit goals influence focus without overwriting source evidence scores."""
    request = WorkingMemoryRequest(
        sequence_id="control-001", text="The scientist examined the sample.",
        goals={"priority": 0.9, "features": {"careful": 1.0, "analysis": 0.8}},
        interpretations=[
            {
                "hypothesis_id": "h1", "interpretation": "careful analysis",
                "features": {"careful": 1.0, "analysis": 0.9}, "semantic_score": 0.6,
            },
            {
                "hypothesis_id": "h2", "interpretation": "rapid completion",
                "features": {"speed": 1.0}, "semantic_score": 0.8,
            },
        ],
    )
    result = _no_biology().process(request)
    assert result.executive_control["control_demand"] > 0.0
    assert result.executive_control["focus_hypothesis_id"] == "h1"
    assert result.executive_control["regulated_alternatives_preserved"] == 2
    provisional = result.executive_control["provisional_interpretation_decision"]
    assert provisional["is_final_semantic_decision"] is False
    assert provisional["status"] in {"provisionally_accepted", "continue_evaluation"}
    assert result.operation_trace[18]["operation_id"] == "WMCC-019"
    assert result.operation_trace[19]["operation_id"] == "WMCC-020"
    assert "goal_bias" in result.semantic_resolution["evidence_channels"]



def test_39_source_operations_map_to_canonical_trace_and_missing_semantic_steps_execute() -> None:
    """Trace source operation IDs while explicitly evaluating prediction error and consolidation."""
    from ore_working_memory.source_mapping import SOURCE_OPERATION_MAP, WM_EXTENSION_REGISTRY

    request = WorkingMemoryRequest(
        sequence_id="source-coverage-001",
        text="The scientist examined the sample.",
        interpretations=[
            {
                "hypothesis_id": "h1", "interpretation": "careful scientific examination",
                "lexical_score": 0.95, "semantic_score": 0.96, "syntactic_score": 0.94,
                "discourse_score": 0.92, "world_score": 0.95, "prior": 0.95,
                "pragmatic_score": 0.95,
            },
            {
                "hypothesis_id": "h2", "interpretation": "casual observation",
                "lexical_score": 0.15, "semantic_score": 0.15, "syntactic_score": 0.12,
                "discourse_score": 0.10, "world_score": 0.12, "prior": 0.10,
                "pragmatic_score": 0.10,
            },
        ],
        context={"predicted_interpretation_distribution": {"h1": 0.75, "h2": 0.25}},
    )
    result = _no_biology().process(request)

    assert len(SOURCE_OPERATION_MAP) == 39
    assert len(WM_EXTENSION_REGISTRY) == 3
    assert result.diagnostics["source_operation_mapping_complete"] is True
    assert result.diagnostics["source_extension_operation_count"] == 3
    assert set(result.diagnostics["source_extension_ids"]) == {"WMX-001", "WMX-002", "WMX-003"}
    assert all("source_operation_ids" in row for row in result.operation_trace)
    assert result.operation_trace[1]["source_operation_ids"] == ["WM-SRC-001"]

    prediction_error = next(
        row for row in result.metadata["source_extension_trace"] if row["operation_id"] == "WMX-001"
    )
    assert prediction_error["status"] == "evaluated"
    assert prediction_error["error_metric"] == "total_variation_distance"
    assert 0.0 <= prediction_error["prediction_error"] <= 1.0
    consolidation = next(
        row for row in result.metadata["source_extension_trace"] if row["operation_id"] == "WMX-002"
    )
    if result.semantic_resolution["selected"] is None:
        assert consolidation["status"] == "held_unresolved"
    else:
        assert consolidation["status"] == "consolidated"
        assert consolidation["consolidated_record"]["durable_episodic_storage"] is False


def test_working_memory_update_gate_honors_explicit_task_switch_permission() -> None:
    """Keep the prior task-control profile when an explicit update-gate veto is supplied."""
    request = WorkingMemoryRequest(
        sequence_id="update-gate-001",
        text="The sentence remains available.",
        task_sequence=["reading", "parsing"],
        context={"previous_task": "reading", "allow_working_memory_update": False},
    )
    result = _no_biology().process(request)
    gate = result.task_switching["working_memory_update_gate"]
    extension = next(
        row for row in result.metadata["source_extension_trace"] if row["operation_id"] == "WMX-003"
    )

    assert gate["switch_detected"] is True
    assert gate["update_permission"] is False
    assert gate["open"] is False
    assert gate["reason"] == "explicitly_blocked"
    assert extension["status"] == "closed"
    assert result.task_switching["reconfigured_after_switch"] is False


def test_semantic_prediction_error_is_not_fabricated_without_a_prediction_signal() -> None:
    """Return an explicit not-observed status when no prior prediction is supplied."""
    result = _no_biology().process(WorkingMemoryRequest(
        sequence_id="prediction-unavailable-001",
        text="The student read the book.",
    ))
    prediction = next(
        row for row in result.metadata["source_extension_trace"] if row["operation_id"] == "WMX-001"
    )
    assert prediction["status"] == "not_observed"
    assert prediction["prediction_error"] is None
    assert prediction["does_not_fabricate_missing_prediction"] is True


def test_task_switching_records_switch_cost_and_resumption() -> None:
    """Detect task changes and report explicit cost, interference, and resumed task state."""
    request = WorkingMemoryRequest(
        sequence_id="switch-001", text="The sentence remains available.",
        task_sequence=["reading", "parsing", "reading", "memory_retrieval"],
        context={"previous_task": "reading", "task_cue_strength": 0.5},
    )
    result = _no_biology().process(request)
    task = result.task_switching
    assert task["switch_count"] == 3
    assert task["estimated_total_switch_cost_ms"] > 0.0
    assert task["interference"] >= 0.0
    assert "reading" in task["resumed_tasks"]
    assert task["suspended_checkpoints"]
    assert task["resumption_events"][0]["checkpoint_restored"] is True
    assert task["active_control_state"]["task"] == "memory_retrieval"


def test_serialization_and_disabled_biology_are_json_compatible() -> None:
    """Serialize the complete result while making neural-reference disablement explicit."""
    request = WorkingMemoryRequest(sequence_id="json-001", tokens=["Hello", "world", "."])
    result = _no_biology().process(request)
    payload = result.to_dict()
    assert payload["schema"] == "ore.working_memory_cognitive_control.1.0"
    assert len(payload["operation_trace"]) == 40
    assert payload["biological"]["model"] == "disabled"
    assert payload["tokens"] == ["Hello", "world", "."]


def test_request_and_configuration_validation_reject_invalid_inputs() -> None:
    """Reject missing text/tokens, invalid candidate probabilities, and non-positive limits."""
    try:
        WorkingMemoryRequest(sequence_id="invalid-001")
    except ValueError as error:
        assert "exactly one" in str(error)
    else:
        raise AssertionError("missing sentence input should fail")

    try:
        WorkingMemoryRequest(
            sequence_id="invalid-002", text="A sentence",
            interpretations=[{"semantic_score": 1.5}],
        )
    except ValueError as error:
        assert "semantic_score" in str(error)
    else:
        raise AssertionError("out-of-range evidence should fail")

    try:
        WorkingMemoryConfig(buffer_capacity=0)
    except ValueError as error:
        assert "limits must be positive" in str(error)
    else:
        raise AssertionError("zero buffer capacity should fail")


def test_every_working_memory_class_and_function_has_a_docstring() -> None:
    """Check every source class and function has a plain-English annotation for the SQLite catalog."""
    package_root = Path(__file__).resolve().parents[1] / "ore_working_memory"
    for source_path in package_root.glob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                assert ast.get_docstring(node), (
                    f"missing annotation docstring: {source_path.name}:{node.name}"
                )


def test_working_memory_annotations_are_stored_in_logger_sqlite_database(tmp_path) -> None:
    """Index the repository and verify class/method annotations are persisted in the annotations table."""
    root = Path(__file__).resolve().parents[1]
    database = LoggerDatabase(tmp_path / "working-memory-annotations.sqlite3")
    report = sync_repository(database, root)
    assert report["parse_errors"] == 0
    expected = (
        ("ore_working_memory.contracts", "WorkingMemoryRequest", "class"),
        ("ore_working_memory.contracts", "WorkingMemoryRequest.__post_init__", "function"),
        ("ore_working_memory.contracts", "WorkingMemoryConfig", "class"),
        ("ore_working_memory.biology", "WorkingMemoryLIFSimulator", "class"),
        ("ore_working_memory.biology", "WorkingMemoryLIFSimulator.step", "function"),
        ("ore_working_memory.registry", "OperationSpec", "class"),
        ("ore_working_memory.registry", "registry_as_dicts", "function"),
        ("ore_working_memory.pipeline", "WorkingMemoryProcessor", "class"),
        ("ore_working_memory.pipeline", "WorkingMemoryProcessor.process", "function"),
        ("ore_working_memory.pipeline", "WorkingMemoryProcessor._op_emit_task_state", "function"),
        ("ore_working_memory.pipeline", "WorkingMemoryProcessor._op_chunk_stored_information", "function"),
        ("ore_working_memory.pipeline", "WorkingMemoryProcessor._op_revise_interpretations", "function"),
        ("ore_working_memory.pipeline", "WorkingMemoryProcessor._op_evaluate_processing_completion", "function"),
        ("ore_working_memory.pipeline", "WorkingMemoryProcessor._op_preserve_suspended_task_states", "function"),
    )
    for module, qualified_name, kind in expected:
        row = database.get_annotation(module, qualified_name, kind)
        assert row is not None
        assert row["annotation"].strip()
        assert row["annotation_source"] == "docstring"
    with database.connection() as connection:
        stored = connection.execute(
            "SELECT COUNT(*) FROM annotations WHERE module_name LIKE 'ore_working_memory.%'"
        ).fetchone()[0]
    assert stored >= 45



def test_operation_registry_names_match_the_uploaded_wmcc_source_sequence() -> None:
    """Protect the exact 40 canonical operation names and their original stage allocation."""
    expected = [
        "Initialize cognitive processing environment",
        "Receive linguistic information",
        "Encode incoming linguistic representations",
        "Distribute representations across working-memory buffers",
        "Organize and chunk stored information",
        "Maintain activation",
        "Regulate memory capacity and attention",
        "Perform content-addressable retrieval",
        "Detect incomplete syntactic structures",
        "Generate grammatical expectations",
        "Maintain unresolved dependencies",
        "Maintain nested syntactic structures",
        "Update incomplete structures",
        "Maintain dependencies through working-memory updates",
        "Maintain cognitive goals",
        "Monitor sentence-processing progress",
        "Detect competing processing demands",
        "Allocate attention",
        "Regulate competing representations",
        "Determine interpretation acceptance",
        "Coordinate processing subsystems",
        "Identify semantic ambiguity",
        "Generate alternative interpretations",
        "Retrieve supporting semantic knowledge",
        "Evaluate grammatical compatibility",
        "Evaluate contextual compatibility",
        "Calculate the overall evidence supporting each interpretation",
        "Allow competing interpretations to interact",
        "Select an interpretation or preserve multiple possibilities",
        "Revise an interpretation when new information arrives",
        "Identify the currently active cognitive task",
        "Detect when a task switch is necessary",
        "Preserve the current task state before switching",
        "Activate the new cognitive task",
        "Reduce interference from the previous task",
        "Monitor the new task and determine when to return",
        "Resume suspended tasks",
        "Learn from task-switching history",
        "Integrate the cognitive state",
        "Evaluate whether processing is complete",
    ]
    assert [item.operation_name for item in OPERATION_REGISTRY] == expected
    assert [item.operation_id for item in OPERATION_REGISTRY] == [
        f"WMCC-{index:03d}" for index in range(1, 41)
    ]


def test_semantic_knowledge_compatibility_and_new_evidence_revision_are_applied() -> None:
    """Use supplied concept support and explicit later evidence to revise a retained interpretation."""
    request = WorkingMemoryRequest(
        sequence_id="revision-001",
        text="The researcher examined the sample.",
        interpretations=[
            {
                "hypothesis_id": "careful", "interpretation": "careful examination",
                "lexical_score": 0.85, "semantic_score": 0.84, "syntactic_score": 0.82,
                "discourse_score": 0.8, "world_score": 0.8, "prior": 0.6,
                "features": {"careful": 1.0, "analysis": 0.8},
            },
            {
                "hypothesis_id": "rapid", "interpretation": "rapid examination",
                "lexical_score": 0.35, "semantic_score": 0.3, "syntactic_score": 0.35,
                "discourse_score": 0.35, "world_score": 0.35, "prior": 0.4,
                "features": {"speed": 1.0},
            },
        ],
        context={
            "semantic_knowledge": {"careful": 0.8, "rapid": 0.3},
            "revision_evidence": {
                "rapid": {
                    "lexical_score": 1.0, "semantic_score": 1.0,
                    "syntactic_score": 1.0, "discourse_score": 1.0,
                    "world_score": 1.0, "pragmatic_score": 1.0,
                }
            },
        },
    )
    result = _no_biology(semantic_temperature=0.35).process(request)
    assert result.semantic_resolution["revision_applied"] is True
    assert result.semantic_resolution["selected_hypothesis_id"] == "rapid"
    support = {
        item["hypothesis_id"]: item
        for item in result.operation_trace[23]["functional_output"]["support"]
    }
    assert support["careful"]["knowledge_score"] == 0.8
    assert support["rapid"]["knowledge_score"] == 0.3


def test_task_transition_learning_uses_observed_history_not_estimated_costs() -> None:
    """Update task-transition estimates only from caller-supplied observed transition costs."""
    request = WorkingMemoryRequest(
        sequence_id="task-learning-001",
        text="A sentence for task switching.",
        task_sequence=["reading", "parsing"],
        context={
            "previous_task": "reading",
            "prior_transition_costs": {"reading->parsing": 100.0},
            "observed_transition_costs": {"reading->parsing": 200.0},
            "task_cost_learning_rate": 0.2,
        },
    )
    task = _no_biology().process(request).task_switching
    assert task["learned_transition_costs"]["reading->parsing"] == 120.0
    assert task["learning_source"] == "explicit_observed_history"


def test_processing_completion_is_blocked_by_open_syntax_and_unresolved_references() -> None:
    """Do not mark comprehension complete while required syntax or references remain open."""
    request = WorkingMemoryRequest(
        sequence_id="completion-open-001",
        text="The scientist who discovered the compound",
        partial_structures=[{
            "structure_id": "relative-clause", "label": "relative clause",
            "expected_slots": ["main_clause_completion"],
        }],
        context={"unresolved_references": ["it"]},
    )
    result = _no_biology().process(request)
    assert result.diagnostics["processing_complete"] is False
    assert result.operation_trace[-1]["operation_id"] == "WMCC-040"
    assert result.operation_trace[-1]["functional_output"]["status"] == "partial_or_unresolved"


