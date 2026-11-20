"""Acceptance tests for the read-only USG knowledge-to-ORE lexical adapter."""
from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path

import pytest

from ore_contracts import (
    ComputeBudget,
    ContractCatalog,
    ModuleStatus,
    OperationMode,
    RunContext,
)
from ore_contracts.module_registry import ModuleRegistry
from ore_contracts.module_logging import register_module_annotations
from ore_logger.database import LoggerDatabase
from ore_usg import (
    USGKnowledgeLexicalAdapter,
    USGKnowledgeValidationError,
    make_usg_bundle_artifact,
    register_usg_lexical_module,
    validate_knowledge_bundle,
)


def _bundle() -> dict:
    """Create a deterministic fixture with polysemy and source-rights boundaries."""
    timestamp = "2026-11-12T00:00:00Z"
    return {
        "contract": "USG-KNOWLEDGE/0.1.0",
        "namespace": "urn:usg:adapter-test",
        "bundle_id": "ore-usg-lexical-test",
        "created_at": timestamp,
        "records": [
            {
                "id": "lemma:en:bank", "kind": "LEMMA", "version": 1,
                "state": "ACCEPTED", "created_at": timestamp,
                "source_ids": ["source:permitted"], "evidence_ids": ["evidence:lemma"],
                "context_ids": ["context:english"],
                "payload": {"lemma": "bank", "language": "en", "part_of_speech": "NOUN"},
            },
            {
                "id": "sense:en:bank:financial", "kind": "LEXICAL_SENSE", "version": 1,
                "state": "ACCEPTED", "created_at": timestamp,
                "source_ids": ["source:permitted"], "evidence_ids": ["evidence:financial"],
                "context_ids": ["context:finance"],
                "payload": {
                    "lemma_record_id": "lemma:en:bank", "language": "en",
                    "definition": "A financial institution.", "sense_source_id": "source:permitted",
                },
            },
            {
                "id": "sense:en:bank:river", "kind": "LEXICAL_SENSE", "version": 1,
                "state": "VALIDATED_FOR_PROFILE", "created_at": timestamp,
                "source_ids": ["source:internal"], "evidence_ids": ["evidence:river"],
                "context_ids": ["context:geography"],
                "payload": {
                    "lemma_record_id": "lemma:en:bank", "language": "en",
                    "definition": "The land alongside a river.", "sense_source_id": "source:internal",
                },
            },
            {
                "id": "sense:en:bank:candidate", "kind": "LEXICAL_SENSE", "version": 1,
                "state": "CANDIDATE", "created_at": timestamp,
                "source_ids": ["source:permitted"], "evidence_ids": ["evidence:candidate"],
                "context_ids": ["context:finance"],
                "payload": {
                    "lemma_record_id": "lemma:en:bank", "language": "en",
                    "definition": "Candidate meaning, not accepted.", "sense_source_id": "source:permitted",
                },
            },
            {
                "id": "form:en:banks", "kind": "LEXICAL_FORM", "version": 1,
                "state": "ACCEPTED", "created_at": timestamp,
                "source_ids": ["source:permitted"], "evidence_ids": ["evidence:form"],
                "context_ids": ["context:english"],
                "payload": {"form": "banks", "language": "en", "script": "Latn"},
            },
            {
                "id": "lemma:en:restricted", "kind": "LEMMA", "version": 1,
                "state": "ACCEPTED", "created_at": timestamp,
                "source_ids": ["source:restricted"], "evidence_ids": ["evidence:restricted"],
                "context_ids": ["context:english"],
                "payload": {"lemma": "restricted", "language": "en", "part_of_speech": "ADJ"},
            },
            {
                "id": "lemma:ar:عين", "kind": "LEMMA", "version": 1,
                "state": "CANDIDATE", "created_at": timestamp,
                "source_ids": ["source:permitted"], "evidence_ids": ["evidence:arabic"],
                "context_ids": ["context:arabic"],
                "payload": {"lemma": "عين", "language": "ar", "part_of_speech": "NOUN"},
            },
        ],
        "sources": [
            {
                "id": "source:permitted", "provider": "Adapter fixture", "source_type": "OTHER",
                "source_locator": None, "retrieved_at": timestamp, "license_id": "MIT",
                "rights_status": "KNOWN_PERMITTED", "redistribution": "ALLOWED",
                "attribution": "Adapter fixture source",
            },
            {
                "id": "source:internal", "provider": "Internal fixture", "source_type": "USG_DOCUMENT",
                "source_locator": None, "retrieved_at": timestamp, "license_id": None,
                "rights_status": "INTERNAL", "redistribution": "NOT_APPLICABLE",
                "attribution": "Internal fixture source",
            },
            {
                "id": "source:restricted", "provider": "Restricted fixture", "source_type": "DICTIONARY",
                "source_locator": None, "retrieved_at": timestamp, "license_id": None,
                "rights_status": "UNKNOWN", "redistribution": "UNKNOWN",
                "attribution": "Restricted fixture source",
            },
        ],
        "evidence": [
            {
                "id": evidence_id, "source_id": source_id, "evidence_type": "DEFINITION",
                "support_status": "SUPPORTS", "captured_at": timestamp, "transformation_ids": [],
            }
            for evidence_id, source_id in (
                ("evidence:lemma", "source:permitted"),
                ("evidence:financial", "source:permitted"),
                ("evidence:river", "source:internal"),
                ("evidence:candidate", "source:permitted"),
                ("evidence:form", "source:permitted"),
                ("evidence:restricted", "source:restricted"),
                ("evidence:arabic", "source:permitted"),
            )
        ],
        "contexts": [
            {"id": "context:english", "context_type": "LANGUAGE", "value": "en"},
            {"id": "context:arabic", "context_type": "LANGUAGE", "value": "ar"},
            {"id": "context:finance", "context_type": "DOMAIN", "value": "finance"},
            {"id": "context:geography", "context_type": "DOMAIN", "value": "geography"},
        ],
        "mappings": [],
        "relations": [],
    }


def test_accepted_lemma_preserves_multiple_senses_and_separate_glosses():
    """Keep polysemy as multiple ORE meanings rather than choosing or merging senses."""
    bundle = _bundle()
    next(item for item in bundle["evidence"] if item["id"] == "evidence:river")[
        "support_status"
    ] = "CONTRADICTS"
    original = deepcopy(bundle)
    resource = USGKnowledgeLexicalAdapter().convert(bundle)

    bank = next(item for item in resource.entries if item.orthography == "bank")
    assert bank.entry_id.startswith("usg:")
    assert bank.entry_id != "lemma:en:bank"
    assert bank.lexical_category == "noun"
    assert bank.language == "en"
    assert len({item.meaning_id for item in bank.meanings}) == 2
    assert all(item.meaning_id.startswith("usg:") for item in bank.meanings)
    assert any("sense%3Aen%3Abank%3Afinancial" in item.source_reference for item in bank.meanings)
    assert any("sense%3Aen%3Abank%3Ariver" in item.source_reference for item in bank.meanings)
    assert {item.gloss for item in bank.meanings} == {
        "A financial institution.", "The land alongside a river.",
    }
    assert all(item.features == {} for item in bank.meanings)
    assert "lemma%3Aen%3Abank" in bank.source_reference
    assert "source%3Apermitted" in bank.source_reference
    assert "context%3Aenglish" in bank.source_reference
    assert bundle == original, "the adapter must not mutate its caller-owned bundle"

    surface_form = next(item for item in resource.entries if item.orthography == "banks")
    assert surface_form.meanings == ()
    assert surface_form.lexical_category == "unknown"
    assert surface_form.language == "en"
    assert "surface_form_only" in surface_form.verified_by
    assert resource.report.meanings_created == 2
    assert "sense:en:bank:candidate" not in resource.report.imported_record_ids
    assert any("neutral default of 1.0" in item for item in resource.report.warnings)
    contradiction = next(
        item for item in resource.report.evidence_summaries
        if item.evidence_id == "evidence:river"
    )
    assert contradiction.support_status == "CONTRADICTS"
    finance_context = next(
        item for item in resource.report.context_summaries
        if item.context_id == "context:finance"
    )
    assert finance_context.context_type == "DOMAIN"
    assert finance_context.value == "finance"


def test_candidate_records_require_explicit_opt_in_and_keep_their_state():
    """Candidate lexical material stays out of the default active lexicon."""
    default_resource = USGKnowledgeLexicalAdapter().convert(_bundle())
    candidate_resource = USGKnowledgeLexicalAdapter(
        include_candidate_states=True
    ).convert(_bundle())

    assert not any(item.orthography == "عين" for item in default_resource.entries)
    arabic = next(item for item in candidate_resource.entries if item.orthography == "عين")
    assert arabic.verified_by.startswith("USG_KNOWLEDGE_STATE:CANDIDATE")
    bank = next(item for item in candidate_resource.entries if item.orthography == "bank")
    assert "sense:en:bank:candidate" in candidate_resource.report.imported_record_ids


def test_record_ids_are_scoped_to_the_usg_bundle_namespace():
    """Do not collide when independent USG bundles reuse local record IDs."""
    first = USGKnowledgeLexicalAdapter().convert(_bundle())
    second_bundle = _bundle()
    second_bundle["namespace"] = "urn:usg:another-namespace"
    second_bundle["bundle_id"] = "another-bundle"
    second = USGKnowledgeLexicalAdapter().convert(second_bundle)

    first_bank = next(item for item in first.entries if item.orthography == "bank")
    second_bank = next(item for item in second.entries if item.orthography == "bank")
    assert first_bank.entry_id != second_bank.entry_id
    assert {
        item.meaning_id for item in first_bank.meanings
    }.isdisjoint({
        item.meaning_id for item in second_bank.meanings
    })
    assert "lemma:en:bank" in first.report.imported_record_ids
    assert "lemma:en:bank" in second.report.imported_record_ids


def test_unknown_rights_source_is_excluded_and_reported():
    """Do not expose lexical entries whose contributing sources have unknown rights."""
    resource = USGKnowledgeLexicalAdapter().convert(_bundle())
    assert not any(item.orthography == "restricted" for item in resource.entries)
    exclusion = next(
        item for item in resource.report.excluded_records
        if item.record_id == "lemma:en:restricted"
    )
    assert exclusion.reason == "source_rights_not_permitted_for_local_import"
    summary = next(
        item for item in resource.report.source_summaries
        if item.source_id == "source:restricted"
    )
    assert summary.usable_for_local_import is False
    assert summary.source_locator is None
    assert summary.license_id is None


@pytest.mark.parametrize(
    "unicode_id",
    [
        "lemma:ar:عين",
        "lemma:fr:café",
        "lemma:fr:cafe\u0301",
        "lemma:ja:猫",
    ],
)
def test_unicode_letters_and_combining_marks_are_valid_in_record_ids(unicode_id: str):
    """Accept Unicode scripts and both composed and decomposed accented IDs."""
    bundle = _bundle()
    original_id = bundle["records"][0]["id"]
    bundle["records"][0]["id"] = unicode_id
    for record in bundle["records"]:
        payload = record.get("payload", {})
        if payload.get("lemma_record_id") == original_id:
            payload["lemma_record_id"] = unicode_id

    assert validate_knowledge_bundle(bundle) == ()


@pytest.mark.parametrize("invalid_id", ["lemma:en:bad id", "lemma:en:bad\\nname", "lemma:en:bad#name", ":lemma:en:bad"])
def test_unicode_id_validation_still_rejects_whitespace_controls_and_bad_punctuation(invalid_id: str):
    """Unicode support must not admit whitespace, controls, or unapproved punctuation."""
    bundle = _bundle()
    bundle["records"][0]["id"] = invalid_id

    assert any("records[0].id is not a valid" in error for error in validate_knowledge_bundle(bundle))


def test_invalid_cross_reference_is_rejected_before_conversion():
    """Reject malformed knowledge snapshots rather than silently dropping bad references."""
    bundle = _bundle()
    bundle["records"][1]["payload"]["lemma_record_id"] = "lemma:missing"
    errors = validate_knowledge_bundle(bundle)
    assert any("must reference a LEMMA record" in item for item in errors)
    with pytest.raises(USGKnowledgeValidationError):
        USGKnowledgeLexicalAdapter().convert(bundle)


def test_duplicate_ids_and_unsupported_contract_version_are_rejected():
    """Detect identity collisions and refuse an unrecognized knowledge contract."""
    bundle = _bundle()
    bundle["records"].append(deepcopy(bundle["records"][0]))
    assert any("duplicate ID" in item for item in validate_knowledge_bundle(bundle))
    bundle = _bundle()
    bundle["contract"] = "USG-KNOWLEDGE/0.2.0"
    assert any("contract must equal" in item for item in validate_knowledge_bundle(bundle))


def test_ore_module_registration_emits_a_validated_typed_lexicon_artifact():
    """Execute the registered module through ORE's normal contracts and backend registry."""
    catalog = ContractCatalog()
    registry = ModuleRegistry(catalog)
    register_usg_lexical_module(registry)
    run_context = RunContext(
        run_id="usg-lexical-adapter-test",
        mode=OperationMode.FUNCTIONAL,
        deterministic_seed=17,
        compute_budget=ComputeBudget(),
    )
    input_bundle = _bundle()
    input_artifact = make_usg_bundle_artifact(
        input_bundle,
        run_id=run_context.run_id,
        producer_id="test.usg_bundle",
    )
    assert input_artifact.envelope.schema_version == "0.1.0"
    assert input_artifact.payload == input_bundle
    assert input_artifact.payload is not input_bundle
    backend = registry.get_backend("ore.usg_lexical_import")
    assert backend is not None
    result = backend.execute({"knowledge_bundle": input_artifact}, run_context)

    assert result.status is ModuleStatus.SUCCESS
    assert len(result.artifacts) == 1
    output = result.artifacts[0]
    assert catalog.validate_artifact(output).valid
    assert output.envelope.contract_id == "ORE.USG.LEXICON"
    assert output.payload.bundle_id == "ore-usg-lexical-test"
    assert len(output.payload.entries) == 2
    assert output.metadata["runtime_mutation_performed"] is False
    assert output.envelope.parent_artifact_ids == (input_artifact.payload_hash,)


def test_every_usg_package_class_and_callable_is_indexed_in_ore_sqlite_annotations(tmp_path):
    """Verify all new package classes, functions, and methods have SQLite annotations."""
    root = Path(__file__).resolve().parents[1]
    database_path = tmp_path / "ore_logger.sqlite3"
    report = register_module_annotations(database_path, root)
    assert report["parse_errors"] == 0
    assert report["scope"] == "ore_contracts"
    assert set(report["indexed_scopes"]) == {"ore_contracts", "ore_usg"}

    database = LoggerDatabase(database_path)
    package_root = root / "ore_usg"
    python_files = sorted(package_root.rglob("*.py"))
    assert python_files

    expected: list[tuple[str, str, str, str]] = []

    def collect_nested(node: ast.AST, module_name: str, relative: str, parent: str = "") -> None:
        """Mirror the logger's traversal through control-flow and local definition scopes."""
        if isinstance(node, ast.ClassDef):
            qualified_name = f"{parent}.{node.name}" if parent else node.name
            expected.append((module_name, qualified_name, "class", relative))
            for child in node.body:
                collect_nested(child, module_name, relative, qualified_name)
            return
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            qualified_name = f"{parent}.{node.name}" if parent else node.name
            symbol_kind = "async_function" if isinstance(node, ast.AsyncFunctionDef) else "function"
            expected.append((module_name, qualified_name, symbol_kind, relative))
            local_parent = qualified_name + ".<locals>"
            for child in node.body:
                collect_nested(child, module_name, relative, local_parent)
            return
        for child in ast.iter_child_nodes(node):
            collect_nested(child, module_name, relative, parent)

    for source_path in python_files:
        relative = source_path.relative_to(root).as_posix()
        module_parts = list(Path(relative).with_suffix("").parts)
        if module_parts and module_parts[-1] == "__init__":
            module_parts.pop()
        module_name = ".".join(module_parts) or "__root__"
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=relative)
        for node in tree.body:
            collect_nested(node, module_name, relative)


    assert expected, "the adapter package should declare classes and callables"
    missing = [
        (module_name, qualified_name, kind, relative)
        for module_name, qualified_name, kind, relative in expected
        if database.get_annotation(
            module_name, qualified_name, kind, relative,
        ) is None
    ]
    assert not missing, f"unindexed ORE USG symbols: {missing}"
