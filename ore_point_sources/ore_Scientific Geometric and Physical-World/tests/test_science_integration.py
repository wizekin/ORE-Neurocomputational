"""Acceptance tests for ore_science integration with the ORE framework.

Covers module registration through ``ore_contracts``, workspace publication through
``ore_workspace``, SQLite logging through ``ore_logger``, backend failure isolation,
and the operation registry (acceptance criteria in Section 8 of the specification).
"""
from __future__ import annotations

import json
from pathlib import Path

from ore_contracts import ContractCatalog, ModuleRegistry, ModuleStatus
from ore_science import ScienceConfig, ScienceRequest, ScientificReasoner
from ore_science.annotations import index_science_annotations, record_science_result
from ore_science.orchestration import create_science_workspace
from ore_science.registration import (
    MODULE_ID,
    ORE_SCIENCE_CONTRACT,
    register_science_module,
)
from ore_science.registry import ALL_OPERATIONS, OPERATION_REGISTRY, STATION_REGISTRY
from ore_workspace.workspace import WorkspaceManager


def _config() -> ScienceConfig:
    """Return a fast test configuration."""
    return ScienceConfig(
        collocation_points=41, time_steps=160, pin_basis_size=6, pin_max_iterations=40
    )


def _heat_request() -> ScienceRequest:
    """Return the canonical heat-diffusion request."""
    return ScienceRequest(
        question=(
            "Predict the temperature field of a 1 m rod after 60 s with thermal "
            "diffusivity alpha = 1e-4 m2/s, initial temperature 100 degC, ends held "
            "at 0 degC"
        ),
        request_id="test_integration",
    )


# ---------------------------------------------------------------------------
# Module registration (Section 6 ORE integration mapping)
# ---------------------------------------------------------------------------
def test_module_registration_binds_backend_and_manifest() -> None:
    """The scientific layer registers its contract, manifest, and backend with ORE."""
    registry = ModuleRegistry(ContractCatalog())
    backend = register_science_module(registry)
    manifest = registry.get_manifest(MODULE_ID)
    assert manifest is not None
    assert manifest.identity.module_id == MODULE_ID
    assert "science.independent_validation" in manifest.capabilities
    assert "science.model_routing" in manifest.capabilities
    described = backend.describe()
    assert described.identity == manifest.identity
    assert registry.find_by_capability("science.task_interpretation")


def test_registered_backend_executes_request_and_returns_typed_artifact() -> None:
    """The bound backend executes a request and returns a contract artifact."""
    registry = ModuleRegistry(ContractCatalog())
    backend = register_science_module(registry)

    class _Context:
        run_id = "run_science_test"

    result = backend.execute(
        {"science_request": {"question": _heat_request().question}}, _Context()
    )
    assert result.status is ModuleStatus.SUCCESS
    assert len(result.artifacts) == 1
    artifact = result.artifacts[0]
    assert artifact.envelope.contract_id == ORE_SCIENCE_CONTRACT
    assert artifact.metadata["outcome"] == "COMPLETED"
    payload = artifact.payload
    for field in ("outcome", "result_status", "validation_reports", "operation_trace"):
        assert field in payload


def test_failed_backend_call_does_not_corrupt_coordinator() -> None:
    """A failed model service call returns a typed failure without corrupting the registry."""
    registry = ModuleRegistry(ContractCatalog())
    backend = register_science_module(registry)

    class _Context:
        run_id = "run_failure_test"

    failed = backend.execute(
        {"science_request": {"question": ""}}, _Context()
    )
    assert failed.status is ModuleStatus.FAILED
    abstained = backend.execute({"unexpected_port": {}}, _Context())
    assert abstained.status is ModuleStatus.ABSTAINED
    # The coordinator-side registry remains healthy after the failures.
    assert registry.get_manifest(MODULE_ID) is not None
    assert registry.bind_backend(MODULE_ID, backend)


# ---------------------------------------------------------------------------
# Workspace publication (Step 46)
# ---------------------------------------------------------------------------
def test_workspace_publication_preserves_artifacts_and_validation() -> None:
    """Workspace publication preserves immutable raw output and validation references."""
    with WorkspaceManager(enable_logging=False) as manager:
        workspace = create_science_workspace(manager, "science input")
        reasoner = ScientificReasoner(_config(), workspace=workspace)
        result = reasoner.process(_heat_request())
        assert result.diagnostics["valid"]
        fields = {record["field"] for record in result.workspace_publications}
        assert "artifact_references" in fields
        assert "semantic_hypotheses" in fields
        snapshot = workspace.snapshot()
        artifacts = snapshot.state["artifact_references"]
        hypotheses = snapshot.state["semantic_hypotheses"]
        assert len(artifacts) == 1
        assert len(hypotheses) == 1
        artifact = artifacts[0]
        assert artifact["kind"] == "scientific_raw_output"
        assert artifact["provenance_ref"]["digest"]
        assert artifact["producer_module"] == "ore_science"
        hypothesis = hypotheses[0]
        assert hypothesis["validation_passed"] is True
        assert hypothesis["status"] == "COMPUTATIONALLY_VALIDATED"


def test_workspace_publication_without_permissions_is_rejected_not_raised() -> None:
    """Publication without granted permissions records rejection instead of raising."""
    with WorkspaceManager(enable_logging=False) as manager:
        workspace = manager.create_workspace("science input", current_goal="reason")
        reasoner = ScientificReasoner(_config(), workspace=workspace)
        result = reasoner.process(_heat_request())
        decisions = {
            record["field"]: record["decision"] for record in result.workspace_publications
        }
        assert decisions and all(
            decision == "rejected" for decision in decisions.values()
        )
        assert result.diagnostics["valid"], "the scientific result itself remains valid"


# ---------------------------------------------------------------------------
# SQLite logger (Section 6: ore_logger integration)
# ---------------------------------------------------------------------------
def test_annotation_indexing_covers_the_package(tmp_path: Path) -> None:
    """Annotation indexing registers the package's symbols in the shared logger."""
    report = index_science_annotations(
        Path(__file__).resolve().parent.parent, database_path=tmp_path / "logger.sqlite3"
    )
    counts = report["science_annotations"]
    assert counts.get("class", 0) >= 15
    assert counts.get("function", 0) >= 100
    assert report["science_symbol_count"] >= 150


def test_result_recording_persists_all_operations(tmp_path: Path) -> None:
    """Result recording persists every operation and the run completes in SQLite."""
    reasoner = ScientificReasoner(_config())
    result = reasoner.process(_heat_request())
    metadata = record_science_result(
        result, Path(__file__).resolve().parent.parent, database_path=tmp_path / "logger.sqlite3"
    )
    assert metadata["operation_count"] == 52
    assert metadata["stage_count"] == 5
    assert metadata["diagnostics_valid"] is True
    assert metadata["database_path"]
    from ore_logger import LoggerDatabase

    database = LoggerDatabase(tmp_path / "logger.sqlite3")
    with database.connect() as connection:
        runs = connection.execute("SELECT status, event_count FROM runs").fetchall()
        events = connection.execute("SELECT COUNT(*) AS n FROM events").fetchall()
    assert runs[0]["status"] == "completed"
    assert int(events[0]["n"]) >= 52


# ---------------------------------------------------------------------------
# Operation registry (52-step trace)
# ---------------------------------------------------------------------------
def test_operation_registry_declares_fifty_two_steps() -> None:
    """The operation registry declares the full 52-step loop with stable identifiers."""
    assert len(OPERATION_REGISTRY) == 52
    assert len(ALL_OPERATIONS) == 52
    steps = [spec.step for spec in OPERATION_REGISTRY]
    assert steps == list(range(1, 53))
    identifiers = [spec.operation_id for spec in OPERATION_REGISTRY]
    assert identifiers[0] == "SCI-01" and identifiers[-1] == "SCI-52"
    assert len(set(identifiers)) == 52


def test_station_registry_covers_architectural_components() -> None:
    """The station registry covers the architectural components A through H."""
    stations = {station.station_id for station in STATION_REGISTRY}
    assert {"A", "B", "C", "D", "E", "F", "G", "H"} <= stations
    covered = {step for station in STATION_REGISTRY for step in station.steps}
    assert covered == set(range(1, 53))


def test_operation_trace_records_every_step_with_status() -> None:
    """Every operation trace row records phase, station, status, and explanation."""
    result = ScientificReasoner(_config()).process(_heat_request())
    for row in result.operation_trace:
        assert row["operation_id"].startswith("SCI-")
        assert row["phase"].startswith("PHASE-")
        assert row["station"]
        assert row["functional_status"] in (
            "executed", "not_applicable", "blocked", "failed", "not_configured"
        )
        assert row["explanation"]


# ---------------------------------------------------------------------------
# Provenance and reproducibility (Step 50)
# ---------------------------------------------------------------------------
def test_repeated_runs_produce_reproducible_outputs() -> None:
    """Repeated runs with fixed deterministic settings produce identical output hashes."""
    first = ScientificReasoner(_config()).process(_heat_request())
    second = ScientificReasoner(_config()).process(_heat_request())
    assert first.provenance_ref["digest"] == second.provenance_ref["digest"]
    third = ScientificReasoner(
        ScienceConfig(collocation_points=41, time_steps=160, pin_basis_size=6,
                      pin_max_iterations=40, random_seed=999)
    ).process(_heat_request())
    assert third.provenance_ref["digest"] != first.provenance_ref["digest"]


def test_result_json_round_trips_through_the_logger_payload() -> None:
    """The full result JSON round-trips through serialization for logging."""
    result = ScientificReasoner(_config()).process(_heat_request())
    encoded = json.dumps(result.to_dict())
    decoded = json.loads(encoded)
    assert decoded["schema"] == "ore.science.52_step.v1"
    assert len(decoded["operation_trace"]) == 52
    assert decoded["provenance_ref"]["digest"]
