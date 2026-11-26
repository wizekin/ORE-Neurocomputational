"""Integration tests: module registry, workspace publication, logging, USG (Point 15).

Covers the integration map of the specification: registration with the ORE
ModuleRegistry (ore_contracts), publication into the Shared Cognitive
Workspace (ore_workspace), the ore_logger annotation bridge, canonical USG
semantic memory, and the demonstration entry point.
"""
from __future__ import annotations

import sqlite3
import subprocess
import sys

from ore_contracts import (
    ComputeBudget,
    ContractCatalog,
    ModuleRegistry,
    ModuleStatus,
    OperationMode,
    RunContext,
)

from ore_agentic import (
    AgenticAnnotationRecorder,
    AgenticDatabase,
    GoalRequest,
    ORE_AGENTIC_CAPABILITIES,
    ORE_AGENTIC_MODULE_ID,
    OperationSpec,
    OPERATION_REGISTRY,
    WorkflowStore,
    build_default_output_contract,
    build_ore_agentic_manifest,
    record_result_to_sqlite,
    register_annotations,
    register_ore_agentic_contracts,
    register_ore_agentic_module,
    registry_as_dicts,
)

from agentic_point15_helpers import (
    REPO_ROOT,
    FakeUSGIndex,
    make_engine,
    make_request,
)


# ------------------------------------------------------------- registration

def test_module_registers_with_the_central_registry():
    """The agentic layer registers with the ORE ModuleRegistry (Section 4)."""
    catalog = ContractCatalog()
    registry = ModuleRegistry(catalog)
    backend = register_ore_agentic_module(registry)
    manifest = registry.get_manifest(ORE_AGENTIC_MODULE_ID)
    assert manifest is not None
    assert manifest.identity.module_id == ORE_AGENTIC_MODULE_ID
    assert manifest.identity.package_path == "ore_agentic"
    for capability in ORE_AGENTIC_CAPABILITIES:
        records = registry.find_by_capability(capability)
        assert any(record.module_id == ORE_AGENTIC_MODULE_ID
                   for record in records)
    context = RunContext("agentic-test", OperationMode.INFER, 0, ComputeBudget())
    result = backend.execute({"goal_request": make_request()}, context)
    assert result.status is ModuleStatus.SUCCESS
    assert result.processing_metadata["operation_steps"] == 50
    # A typed-input violation abstains instead of crashing.
    abstained = backend.execute({"goal_request": {"not": "a request"}}, context)
    assert abstained.status is ModuleStatus.ABSTAINED
    # Double binding is refused without replace.
    try:
        register_ore_agentic_module(registry)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_contract_registration_is_idempotent():
    """Registering the agentic contracts twice returns the existing ones."""
    catalog = ContractCatalog()
    first = register_ore_agentic_contracts(catalog)
    second = register_ore_agentic_contracts(catalog)
    assert [c.contract_id for c in first] == [c.contract_id for c in second]
    assert len(catalog.list_contracts()) >= 3


def test_manifest_declares_responsibilities_and_limitations():
    """The manifest declares responsibilities, capabilities and limitations."""
    manifest = build_ore_agentic_manifest()
    assert len(manifest.responsibilities) >= 5
    assert manifest.limitations  # honesty about the reference implementation
    texts = (*manifest.limitations, *manifest.responsibilities)
    assert any("USG" in text for text in texts)


# ------------------------------------------------------------------ USG

def test_engine_integrates_the_canonical_usg_semantic_memory():
    """The engine resolves entities and retrieves through USG (Steps 2, 37, 41)."""
    index = FakeUSGIndex(labels={"evidence": "evidence (epistemic item)"})
    engine = make_engine(semantic_index=index)
    request = GoalRequest(
        objective="Analyze the evidence about attention.",
        requester="usg-tester",
        subgoals=make_request().subgoals)
    result = engine.run(request)
    assert result.status.value == "succeeded"
    assert result.interpretation is not None
    assert "evidence" in result.interpretation.entities
    # The semantic memory wraps the canonical index read-only; every proposal
    # produced by the run is a non-authoritative candidate.
    assert engine.semantic.usg_index is index
    assert engine.semantic.proposals()
    assert all(proposal.lifecycle.value == "CANDIDATE"
               for proposal in engine.semantic.proposals())
    assert index.write_calls == []


# ------------------------------------------------------------- workspace

def test_engine_publishes_results_to_the_shared_cognitive_workspace():
    """The engine publishes close-out artifacts into the workspace (Step 50)."""
    from ore_workspace import PermissionPolicy, WorkspaceManager

    manager = WorkspaceManager(logger_database_path=None, enable_logging=False)
    permissions = PermissionPolicy()
    permissions.grant("ore_agentic", ("artifact_references", "module_status",
                                      "unresolved_questions"))
    workspace = manager.create_workspace(
        "analyze the evidence", "understand_sentence",
        permissions=permissions)
    from ore_agentic.persistence.workspace_bridge import WorkspaceBridge

    bridge = WorkspaceBridge(workspace)
    engine = make_engine(workspace_bridge=bridge)
    result = engine.run(make_request())
    assert result.workspace_revision is not None
    assert result.workspace_revision >= 1
    assert bridge.published_findings
    state = workspace.state
    assert state.artifact_references  # the authoritative workspace holds the artifact
    assert any(item.get("workflow_id") == result.workflow_id
               for item in state.artifact_references)
    manager.close()


def test_workspace_bridge_reports_permission_denials():
    """A permission-denied publication is reported, never bypassed."""
    from ore_workspace import PermissionPolicy, WorkspaceManager
    from ore_workspace.models import UpdateDecision

    manager = WorkspaceManager(logger_database_path=None, enable_logging=False)
    workspace = manager.create_workspace(
        "input", "understand_sentence",
        permissions=PermissionPolicy({"ore_agentic": ()}))  # no grants
    from ore_agentic.persistence.workspace_bridge import WorkspaceBridge

    bridge = WorkspaceBridge(workspace)
    validation = bridge.publish_workflow_result({"workflow_id": "wf-x",
                                                 "status": "succeeded"})
    assert validation.decision is UpdateDecision.REJECTED  # least privilege
    assert bridge.denied_publications == ["wf-x"]
    assert bridge.published_findings == []
    manager.close()


# -------------------------------------------------------------- logging

def test_register_annotations_indexes_the_package(tmp_path):
    """The ore_logger catalogue indexes the ore_agentic scope (Step 43)."""
    database_path = tmp_path / "annotations.sqlite3"
    report = register_annotations(database_path=database_path,
                                  repository_root=REPO_ROOT)
    assert report["scope"] == "ore_agentic"
    assert report["symbols_written"] >= 40
    connection = sqlite3.connect(database_path)
    try:
        rows = connection.execute(
            "SELECT COUNT(*) FROM annotations WHERE module_name LIKE 'ore_agentic%'"
        ).fetchone()
        assert rows[0] >= 40
    finally:
        connection.close()


def test_record_result_to_sqlite_writes_one_event_per_step(tmp_path):
    """One ore_logger event is written per pipeline step (Step 43)."""
    result = make_engine().run(make_request())
    report = record_result_to_sqlite(result,
                                     database_path=tmp_path / "logger.sqlite3",
                                     repository_root=REPO_ROOT)
    assert report["events_written"] == 50
    assert report["class_annotation_linked"]
    connection = sqlite3.connect(tmp_path / "logger.sqlite3")
    try:
        rows = connection.execute(
            "SELECT COUNT(*) FROM events WHERE event_kind = 'operation'"
        ).fetchone()
        assert rows[0] >= 50
    finally:
        connection.close()


def test_annotation_recorder_redacts_and_never_breaks_work(tmp_path):
    """The engine-time recorder strips secrets and swallows logging errors."""
    recorder = AgenticAnnotationRecorder(database_path=tmp_path / "rec.sqlite3")
    with recorder:
        event = recorder.record("agentic_run", task_id="wf-1",
                                metadata={"api_key": "sk-abcdef123456",
                                          "status": "ok"})
    assert "api_key" not in event["metadata"]  # secret keys are dropped
    assert event["metadata"]["status"] == "ok"
    # A failing database never breaks the engine (best-effort logging).
    broken = AgenticAnnotationRecorder(database_path="/nonexistent/dir/x.sqlite3")
    with broken:
        broken.record("agentic_run")  # no exception raised
    assert broken.events


# ------------------------------------------------------ demo and operations

def test_demo_entry_point_runs_end_to_end():
    """``python -m ore_agentic`` executes the full demonstration."""
    completed = subprocess.run(
        [sys.executable, "-m", "ore_agentic"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=120,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(REPO_ROOT)})
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert "Point 15 Agentic Workflow and Memory Layer" in completed.stdout
    assert "trace steps recorded: 50" in completed.stdout
    assert "operation registry: 50 source steps defined" in completed.stdout


def test_operation_registry_covers_all_50_steps_and_phases():
    """The operation registry defines exactly the 50 source steps in 8 phases."""
    assert len(OPERATION_REGISTRY) == 50
    assert [spec.source_step for spec in OPERATION_REGISTRY] == list(range(1, 51))
    assert [spec.operation_id for spec in OPERATION_REGISTRY][
        -1] == "P15-50"
    phases = {spec.phase for spec in OPERATION_REGISTRY}
    assert phases == {"I", "II", "III", "IV", "V", "VI", "VII", "VIII"}
    for spec in OPERATION_REGISTRY:
        assert isinstance(spec, OperationSpec)
        assert spec.purpose and spec.safeguard and spec.mathematical_form
    assert len(registry_as_dicts()) == 50


def test_default_output_contract_is_deterministic():
    """The default deliverable contract declares its criteria (Step 24)."""
    contract = build_default_output_contract()
    assert contract.contract_id == "agentic.workflow.deliverable"
    kinds = {criterion.kind.value for criterion in contract.criteria}
    assert {"completeness", "schema", "evidence"} <= kinds


def test_engine_database_persists_the_full_workflow(tmp_path):
    """The engine persists workflows, tasks, results and episodes (Section 4)."""
    database_path = tmp_path / "engine.sqlite3"
    engine = make_engine(database_path=str(database_path))
    result = engine.run(make_request())
    store = WorkflowStore(AgenticDatabase(database_path))
    row = store.get(result.workflow_id)
    assert row is not None
    assert row["status"] == "succeeded"
    assert row["idempotency_key"] is None
    tasks = store.tasks(result.workflow_id)
    assert {task["task_id"] for task in tasks} >= {
        "task-alpha", "task-beta", "task-reduce"}
    assert store.results(result.workflow_id)
    assert store.list_checkpoints(result.workflow_id)  # checkpoints persisted
    memory_rows = engine.memory_store.episodes_for(
        workflow_id=result.workflow_id)
    assert len(memory_rows) >= 4  # task episodes + snapshot + close episode
    proposals = engine.memory_store.semantic_proposals()
    assert proposals  # candidate proposals persisted
