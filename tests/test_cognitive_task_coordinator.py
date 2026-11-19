"""Focused acceptance tests for dependency-aware main brain coordination."""
import ast
from pathlib import Path

import pytest

from ore_contracts import (
    ComputeBudget,
    ContractArtifact,
    ContractCatalog,
    InputPort,
    ModuleDependency,
    ModuleIdentity,
    ModuleManifest,
    ModuleRegistry,
    ModuleResult,
    ModuleStatus,
    OperationMode,
    OutputPort,
    RunContext,
    make_envelope,
)
from ore_contracts.coordinator import (
    CognitiveTask,
    CognitiveTaskCoordinator,
    calculate_module_priority,
    normalized_entropy,
    weighted_disagreement,
)
from ore_logger.database import LoggerDatabase
from ore_logger.catalog import sync_repository


class _Backend:
    """Provide deterministic contract-producing behavior for coordinator acceptance tests."""

    def __init__(self, manifest, output_contract, source_port, input_contract=None):
        """Store the manifest and contract names used by this test backend."""
        self.manifest = manifest
        self.output_contract = output_contract
        self.source_port = source_port
        self.input_contract = input_contract

    def describe(self):
        """Return the exact manifest registered for this backend."""
        return self.manifest

    def execute(self, inputs, context):
        """Create one typed output whose lineage points to the exact input artifact."""
        source = inputs[self.source_port]
        output = ContractArtifact(
            make_envelope(
                self.output_contract,
                context.run_id,
                self.manifest.identity.module_id,
                parent_artifact_ids=(source.envelope.fingerprint(),),
            ),
            {"derived_from": source.payload, "producer": self.manifest.identity.module_id},
        )
        return ModuleResult(
            self.manifest.identity.module_id,
            self.manifest.identity.module_version,
            ModuleStatus.SUCCESS,
            artifacts=(output,),
            provenance={"test_backend": True},
        )


def _pipeline():
    """Create a two-stage typed pipeline with one external input and exact schema versions."""
    catalog = ContractCatalog()
    catalog.register_model("ORE.TEST.INPUT", "1.0.0", str)
    catalog.register_model("ORE.TEST.ENCODED", "1.0.0", dict)
    catalog.register_model("ORE.TEST.MEANING", "1.0.0", dict)

    encoder = ModuleManifest(
        ModuleIdentity("ore.test_encoder", "1.0.0", "Test encoder"),
        ("encode the external input",),
        (InputPort("source", "ORE.TEST.INPUT", ("1.0.0",), expected_data_type="builtins.str"),),
        (OutputPort(
            "encoded", "ORE.TEST.ENCODED", "1.0.0", "builtins.dict",
            "encoded test representation",
        ),),
        ("cognition.test_encoding",),
    )
    semantic = ModuleManifest(
        ModuleIdentity("ore.test_semantic", "1.0.0", "Test semantic module"),
        ("interpret the encoded representation",),
        (InputPort("encoded", "ORE.TEST.ENCODED", ("1.0.0",), expected_data_type="builtins.dict"),),
        (OutputPort(
            "meaning", "ORE.TEST.MEANING", "1.0.0", "builtins.dict",
            "semantic test output",
        ),),
        ("cognition.test_semantics",),
    )
    registry = ModuleRegistry(catalog)
    for manifest in (encoder, semantic):
        registry.register(manifest)
    registry.bind_backend(
        encoder.identity.module_id,
        _Backend(encoder, "ORE.TEST.ENCODED", "source"),
    )
    registry.bind_backend(
        semantic.identity.module_id,
        _Backend(semantic, "ORE.TEST.MEANING", "encoded"),
    )
    context = RunContext(
        run_id="coordinator-acceptance",
        mode=OperationMode.FUNCTIONAL,
        deterministic_seed=11,
        compute_budget=ComputeBudget(),
    )
    source = ContractArtifact(
        make_envelope("ORE.TEST.INPUT", context.run_id, "ore.test_input"),
        "The brain coordinates specialized modules.",
    )
    return catalog, registry, context, source


def test_coordinator_discovers_upstream_producer_and_publishes_typed_lineage():
    """Resolve a missing input through a compatible producer and execute two DAG layers."""
    catalog, registry, context, source = _pipeline()
    task = CognitiveTask(
        goal="interpret the input",
        required_capabilities=("cognition.test_semantics",),
        input_artifacts={"source": source},
        expected_outputs=("ORE.TEST.MEANING",),
    )
    coordinator = CognitiveTaskCoordinator(
        registry, catalog, enable_logging=False, max_workers=2,
    )

    result = coordinator.coordinate(task, context)

    assert result.status == "RESOLVED"
    assert result.plan.layers == (
        ("ore.test_encoder",),
        ("ore.test_semantic",),
    )
    assert result.plan.edges == (("ore.test_encoder", "ore.test_semantic"),)
    assert result.plan.capability_coverage == 1.0
    assert result.module_results["ore.test_encoder"].status is ModuleStatus.SUCCESS
    assert result.module_results["ore.test_semantic"].status is ModuleStatus.SUCCESS
    assert len(result.artifacts) == 3
    assert len(result.workspace_state["artifact_references"]) == 2
    assert result.workspace_state["module_status"]["ore.test_semantic"] == "SUCCESS"
    semantic_output = next(
        item for item in result.artifacts if item.contract_id == "ORE.TEST.MEANING"
    )
    assert semantic_output.parent_artifact_ids
    assert semantic_output.payload_hash


def test_missing_required_capability_blocks_without_inventing_a_provider():
    """An unregistered capability produces a visible blocked task and unresolved question."""
    catalog = ContractCatalog()
    registry = ModuleRegistry(catalog)
    task = CognitiveTask(
        goal="perform an unavailable capability",
        required_capabilities=("cognition.not_registered",),
    )
    result = CognitiveTaskCoordinator(
        registry, catalog, enable_logging=False,
    ).coordinate(task)

    assert result.status == "BLOCKED"
    assert result.plan.capability_coverage == 0.0
    assert result.plan.missing_capabilities == ("cognition.not_registered",)
    assert not result.module_results
    assert result.workspace_state["unresolved_questions"]


def test_dependency_cycles_are_detected_before_any_module_is_invoked():
    """Mutually dependent modules remain visible in the rejected graph and are never executed."""
    catalog = ContractCatalog()
    registry = ModuleRegistry(catalog)
    first = ModuleManifest(
        ModuleIdentity("ore.cycle_a", "1.0.0", "Cycle A"),
        ("cycle test",), (), (), ("cognition.cycle_a",),
        dependencies=(ModuleDependency("ore.cycle_b"),),
    )
    second = ModuleManifest(
        ModuleIdentity("ore.cycle_b", "1.0.0", "Cycle B"),
        ("cycle test",), (), (), ("cognition.cycle_b",),
        dependencies=(ModuleDependency("ore.cycle_a"),),
    )
    registry.register(first)
    registry.register(second)
    registry.bind_backend("ore.cycle_a", _Backend(first, "unused", "unused"))
    registry.bind_backend("ore.cycle_b", _Backend(second, "unused", "unused"))
    task = CognitiveTask(goal="detect a cycle", required_capabilities=("cognition.cycle_a",))

    result = CognitiveTaskCoordinator(
        registry, catalog, enable_logging=False,
    ).coordinate(task)

    assert result.status == "FAILED"
    assert result.plan.cycle_nodes == ("ore.cycle_a", "ore.cycle_b")
    assert not result.trace or all(item.operation != "execute_module" for item in result.trace)


def test_priority_entropy_and_weighted_disagreement_are_mathematically_bounded():
    """Verify task priority, normalized entropy and reliability-weighted population variance."""
    assert calculate_module_priority(
        relevance=1.0, downstream_importance=1.0, urgency=1.0,
    ) == pytest.approx(1.0)
    assert normalized_entropy((0.5, 0.5)) == pytest.approx(1.0)
    assert normalized_entropy((1.0, 0.0)) == pytest.approx(0.0, abs=1e-8)
    assert weighted_disagreement(((1.0, 1.0), (3.0, 1.0))) == pytest.approx(1.0)
    assert weighted_disagreement(()) is None
    with pytest.raises(ValueError):
        normalized_entropy((0.7, 0.7))
    with pytest.raises(ValueError):
        calculate_module_priority(
            relevance=1.0, downstream_importance=0.5, urgency=0.5,
            delta=0.2,
        )


def test_every_coordinator_class_and_function_is_indexed_in_logger_sqlite(tmp_path):
    """Synchronize the ORE AST catalog and verify every coordinator definition has an annotation."""
    root = Path(__file__).resolve().parents[1]
    source_path = root / "ore_contracts" / "coordinator.py"
    database = LoggerDatabase(tmp_path / "coordinator-annotations.sqlite3")
    report = sync_repository(database, root, include_paths={"ore_contracts"})
    assert report["parse_errors"] == 0

    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    expected = []

    def collect(body, parent=""):
        """Collect qualified class and callable names using the logger's AST naming convention."""
        for node in body:
            if isinstance(node, ast.ClassDef):
                qualified = f"{parent}.{node.name}" if parent else node.name
                expected.append((qualified, "class"))
                collect(node.body, qualified)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualified = f"{parent}.{node.name}" if parent else node.name
                kind = "async_function" if isinstance(node, ast.AsyncFunctionDef) else "function"
                expected.append((qualified, kind))
                collect(node.body, qualified + ".<locals>")
            else:
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, ast.Module):
                        collect(child.body, parent)

    collect(tree.body)
    assert expected
    for qualified_name, kind in expected:
        annotation = database.get_annotation(
            "ore_contracts.coordinator", qualified_name, kind,
            "ore_contracts/coordinator.py",
        )
        assert annotation is not None, f"missing SQLite annotation: {qualified_name} ({kind})"
        assert annotation["annotation"].strip()
