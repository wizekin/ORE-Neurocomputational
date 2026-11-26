"""Phase 6 compatibility, observability and performance regression coverage."""
from __future__ import annotations

import pytest

from ore_logger.database import LoggerDatabase
from ore_contracts import ComputeBudget, ContractCatalog, OperationMode, RunContext
from ore_contracts.module_registry import ModuleRegistry
from ore_usg.compatibility import (
    USGCompatibilityError,
    USGMigration,
    USGMigrationRegistry,
    inspect_ore_contract_catalog,
    inspect_usg_compatibility,
    register_usg_contracts,
    require_usg_compatibility,
)
from ore_usg.observability import USGSQLiteObservability, _safe_metadata
from ore_usg import make_usg_bundle_artifact, register_usg_knowledge_module
from ore_usg.annotation_sync import synchronize_annotations
from test_usg_lexical_adapter import _bundle


def _bundle():
    return {
        "contract": "USG-KNOWLEDGE/0.1.0",
        "schema_version": "0.1.0",
        "namespace": "test",
        "bundle_id": "phase6-test",
        "records": [{"id": "sense:1", "kind": "LEXICAL_SENSE"}],
        "relations": [{"id": "relation:1"}],
        "mappings": [],
        "evidence": [{"id": "evidence:1"}],
        "contexts": [{"id": "context:1"}],
        "sources": [{"id": "source:1"}],
    }


def test_compatibility_rejects_unknown_contract_and_schema_versions():
    assert inspect_usg_compatibility(_bundle()) == ()
    invalid = {**_bundle(), "contract": "USG-KNOWLEDGE/9.0.0", "schema_version": "9.0.0"}
    assert len(inspect_usg_compatibility(invalid)) == 2
    with pytest.raises(USGCompatibilityError):
        require_usg_compatibility(invalid)


def test_explicit_migration_chain_preserves_semantics_or_declares_change():
    registry = USGMigrationRegistry()
    registry.register(USGMigration(
        migration_id="metadata-only",
        source_contract="USG-KNOWLEDGE/0.0.9",
        target_contract="USG-KNOWLEDGE/0.1.0",
        transform=lambda bundle: {**bundle, "contract": "USG-KNOWLEDGE/0.1.0"},
        semantic_effect="PRESERVED",
        description="Update the envelope contract label only.",
    ))
    old = {**_bundle(), "contract": "USG-KNOWLEDGE/0.0.9"}
    result = registry.apply(old, migration_ids=("metadata-only",))
    assert result.bundle["contract"] == "USG-KNOWLEDGE/0.1.0"
    assert result.source_digest != result.target_digest
    assert not result.semantic_change_declared

    registry.register(USGMigration(
        migration_id="bad-preservation-claim",
        source_contract="USG-KNOWLEDGE/0.1.0",
        target_contract="USG-KNOWLEDGE/0.2.0",
        transform=lambda bundle: {**bundle, "records": []},
        semantic_effect="PRESERVED",
        description="This must fail because it removes source assertions.",
    ))
    with pytest.raises(USGCompatibilityError):
        registry.apply(result.bundle, migration_ids=("bad-preservation-claim",))


def test_ore_catalog_version_check_reports_missing_required_contracts():
    catalog = ContractCatalog()
    register_usg_contracts(catalog)
    issues = inspect_ore_contract_catalog(catalog)
    assert any("ORE.BLACKBOARD.USG_MODULE_KNOWLEDGE@1.0.0" in item for item in issues)


def test_declared_semantic_change_migration_is_explicit_and_traceable(tmp_path):
    registry = USGMigrationRegistry()
    migration = USGMigration(
        migration_id="semantic-change-v1-to-v2",
        source_contract="USG-KNOWLEDGE/0.1.0",
        target_contract="USG-KNOWLEDGE/0.2.0",
        transform=lambda bundle: {
            **bundle,
            "contract": "USG-KNOWLEDGE/0.2.0",
            "records": [*bundle["records"], {"id": "new:assertion"}],
        },
        semantic_effect="DECLARED_CHANGE",
        description="Add an explicitly reviewed assertion under the new source profile.",
    )
    registry.register(migration)
    result = registry.apply(_bundle(), migration_ids=(migration.migration_id,))
    assert result.semantic_change_declared is True
    observer = USGSQLiteObservability(tmp_path / "ore.sqlite3")
    observer.record_migration(result, migration)
    with observer.database.connection() as connection:
        row = connection.execute("SELECT * FROM usg_migration_events").fetchone()
    assert row["migration_id"] == migration.migration_id
    assert row["semantic_change_declared"] == 1


def test_migration_requires_explicit_known_contiguous_chain():
    registry = USGMigrationRegistry()
    with pytest.raises(ValueError):
        registry.apply(_bundle(), migration_ids=())
    with pytest.raises(KeyError):
        registry.apply(_bundle(), migration_ids=("missing",))


def test_sqlite_events_and_source_lineage_are_traceable_and_idempotent(tmp_path):
    database = LoggerDatabase(tmp_path / "ore.sqlite3")
    observer = USGSQLiteObservability(database)
    observer.record_source_bundle(
        artifact_id="source-artifact",
        payload_hash="sha-source",
        contract_id="ORE.USG.KNOWLEDGE_BUNDLE",
        schema_version="0.1.0",
        producer_module="source",
        producer_version="1",
        bundle=_bundle(),
        run_id="run-1",
    )
    assert not observer.record_artifact_lineage(
        artifact_id="source-artifact",
        payload_hash="sha-source",
        contract_id="ORE.USG.KNOWLEDGE_BUNDLE",
        schema_version="0.1.0",
        producer_module="source",
        producer_version="1",
        run_id="run-1",
        source_namespace="test",
        source_bundle_id="phase6-test",
        source_bundle_digest="sha-source",
        source_record_ids=("context:1", "evidence:1", "sense:1", "relation:1", "source:1"),
    )
    observer.record_artifact_lineage(
        artifact_id="derived-artifact",
        payload_hash="sha-derived",
        contract_id="ORE.USG.KNOWLEDGE",
        schema_version="1.0.0",
        producer_module="ore.usg_knowledge_import",
        producer_version="0.1.0",
        parent_artifact_ids=("source-artifact",),
        run_id="run-1",
        source_namespace="test",
        source_bundle_id="phase6-test",
        source_bundle_digest="digest",
        source_record_ids=("record:1",),
    )
    trace = observer.trace("derived-artifact")
    assert trace["parents"][0]["artifact"]["source_bundle_id"] == "phase6-test"
    assert "sense:1" in trace["parents"][0]["artifact"]["source_record_ids"]

    observer.record_module_execution(
        module_id="ore.usg_knowledge_import",
        module_version="0.1.0",
        operation="convert",
        status="SUCCESS",
        duration_ms=1.25,
        run_id="run-1",
        input_artifact_ids=("source-artifact",),
        output_artifact_ids=("derived-artifact",),
        bundle_id="phase6-test",
        bundle_digest="digest",
        metadata={"candidate_count": 2, "definition": "private source definition"},
    )
    event = observer.list_module_execution_events(run_id="run-1")[0]
    assert event["status"] == "SUCCESS"
    assert event["metadata"] == {"candidate_count": 2}
    assert event["input_artifact_ids"] == ["source-artifact"]


def test_conflicting_lineage_cannot_overwrite_existing_artifact(tmp_path):
    observer = USGSQLiteObservability(tmp_path / "ore.sqlite3")
    values = dict(
        artifact_id="immutable",
        payload_hash="hash-a",
        contract_id="ORE.USG.KNOWLEDGE",
        schema_version="1.0.0",
        producer_module="importer",
        producer_version="0.1.0",
    )
    observer.record_artifact_lineage(**values)
    with pytest.raises(ValueError):
        observer.record_artifact_lineage(**{**values, "payload_hash": "hash-b"})


def test_metadata_sanitizer_bounds_nested_values_and_omits_raw_text():
    value = _safe_metadata({
        "safe": "x" * 400,
        "definition": "do not persist",
        "nested": {"evidence_text": "private", "count": 3},
        "many": list(range(40)),
    })
    assert len(value["safe"]) == 240
    assert "definition" not in value
    assert "evidence_text" not in value["nested"]
    assert len(value["many"]) == 21


def test_lineage_trace_detects_cycles_without_infinite_recursion(tmp_path):
    observer = USGSQLiteObservability(tmp_path / "ore.sqlite3")
    common = dict(
        contract_id="ORE.USG.KNOWLEDGE",
        schema_version="1.0.0",
        producer_module="test",
        producer_version="1",
    )
    observer.record_artifact_lineage(artifact_id="a", payload_hash="a", parent_artifact_ids=("b",), **common)
    observer.record_artifact_lineage(artifact_id="b", payload_hash="b", parent_artifact_ids=("a",), **common)
    trace = observer.trace("a")
    assert trace["parents"][0]["parents"][0]["cycle"] is True



def test_registered_import_module_records_execution_and_source_to_output_lineage(tmp_path):
    observer = USGSQLiteObservability(tmp_path / "ore.sqlite3")
    catalog = ContractCatalog()
    registry = ModuleRegistry(catalog)
    register_usg_knowledge_module(registry, observability=observer)
    context = RunContext(
        run_id="phase6-observed-import",
        mode=OperationMode.FUNCTIONAL,
        deterministic_seed=41,
        compute_budget=ComputeBudget(),
    )
    source = make_usg_bundle_artifact(
        _bundle(),
        run_id=context.run_id,
        producer_id="test.usg_source",
    )
    backend = registry.get_backend("ore.usg_knowledge_import")
    result = backend.execute({"knowledge_bundle": source}, context)

    assert result.status.value == "success"
    events = observer.list_module_execution_events(run_id=context.run_id)
    assert len(events) == 1
    assert events[0]["module_id"] == "ore.usg_knowledge_import"
    assert events[0]["input_artifact_ids"] == [source.payload_hash]
    assert len(events[0]["output_artifact_ids"]) == 1

    source_lineage = observer.get_artifact_lineage(source.payload_hash)
    output_hash = result.artifacts[0].payload_hash
    output_lineage = observer.get_artifact_lineage(output_hash)
    assert source_lineage is not None
    assert output_lineage is not None
    assert output_lineage["parent_artifact_ids"] == [source.payload_hash]
    assert "sense:en:bank:financial" in source_lineage["source_record_ids"]
    trace = observer.trace(output_hash)
    assert trace["parents"][0]["artifact"]["artifact_id"] == source.payload_hash



def test_annotation_sync_covers_module_class_function_and_method_definitions(tmp_path):
    package = tmp_path / "ore_usg"
    package.mkdir()
    (package / "__init__.py").write_text('"""Package."""\n', encoding="utf-8")
    (package / "sample.py").write_text(
        '"""Sample."""\n'
        'def top_level(value):\n'
        '    return value\n'
        'class Example:\n'
        '    def method(self):\n'
        '        return True\n',
        encoding="utf-8",
    )
    report = synchronize_annotations(
        tmp_path,
        database_path=tmp_path / "ore_logs" / "annotations.sqlite3",
        include_paths={"ore_usg"},
    )
    assert report["coverage_complete"] is True
    assert report["expected_symbols"] == 5
    assert report["annotated_symbols"] == 5
