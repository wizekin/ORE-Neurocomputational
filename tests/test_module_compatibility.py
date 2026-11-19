"""Exact contract/version compatibility, explicit adapters, and graph coverage tests."""
from dataclasses import replace

from ore_contracts.adapters import make_envelope
from ore_contracts.contracts_registry import ContractCatalog
from ore_contracts.module_adapters import CallableModuleAdapter
from ore_contracts.module_artifacts import ContractArtifact
from ore_contracts.module_compatibility import CompatibilityStatus, CompatibilityValidator
from ore_contracts.module_manifest import InputPort, ModuleIdentity, ModuleManifest, OutputPort
from ore_contracts.module_registry import ModuleRegistry


def _manifest(module_id, contract_version, *, input_versions=(), output=False):
    """Construct one module endpoint with an optionally versioned contract input/output."""
    inputs = (
        (InputPort("in", "ORE.STRING", tuple(input_versions), expected_data_type="builtins.str"),)
        if input_versions else ()
    )
    outputs = (
        (OutputPort("out", "ORE.STRING", contract_version, "builtins.str", "string output"),)
        if output else ()
    )
    return ModuleManifest(
        ModuleIdentity(module_id, "1.0.0", module_id),
        ("exchange string evidence",), inputs, outputs, ("language.string_exchange",),
    )


def _backend(manifest):
    """Return a structurally valid no-op backend for registry availability checks."""
    return CallableModuleAdapter(manifest, lambda inputs, context: ())


def test_contract_mismatch_and_missing_backend_are_not_accepted():
    """Do not infer compatibility from similar payloads or a manifest without a runtime backend."""
    catalog = ContractCatalog()
    catalog.register_model("ORE.STRING", "1.0.0", str)
    catalog.register_model("ORE.OTHER", "1.0.0", str)
    registry = ModuleRegistry(catalog)
    producer = _manifest("ore.source", "1.0.0", output=True)
    consumer = ModuleManifest(
        ModuleIdentity("ore.target", "1.0.0", "Target"), (),
        (InputPort("in", "ORE.OTHER", ("1.0.0",), expected_data_type="builtins.str"),),
        (), ("language.other",),
    )
    registry.register(producer)
    registry.register(consumer)
    validator = CompatibilityValidator(registry)
    assert validator.validate_connection("ore.source", "ore.target", "out", "in").status is CompatibilityStatus.MODULE_NOT_AVAILABLE
    registry.bind_backend("ore.source", _backend(producer))
    registry.bind_backend("ore.target", _backend(consumer))
    report = validator.validate_connection("ore.source", "ore.target", "out", "in")
    assert report.status is CompatibilityStatus.INCOMPATIBLE_CONTRACT


def test_exact_version_is_direct_and_registered_converter_is_explicit():
    """A non-accepted producer version is rejected unless a converter emits the exact target schema."""
    catalog = ContractCatalog()
    catalog.register_model("ORE.STRING", "1.0.0", str)
    catalog.register_model("ORE.STRING", "2.0.0", str)
    registry = ModuleRegistry(catalog)
    producer = _manifest("ore.source", "1.0.0", output=True)
    consumer = _manifest("ore.target", "2.0.0", input_versions=("2.0.0",))
    registry.register(producer)
    registry.register(consumer)
    registry.bind_backend("ore.source", _backend(producer))
    registry.bind_backend("ore.target", _backend(consumer))
    validator = CompatibilityValidator(registry)
    report = validator.validate_connection("ore.source", "ore.target", "out", "in")
    assert report.status is CompatibilityStatus.INCOMPATIBLE_VERSION

    def convert(artifact):
        """Convert the payload and stamp the exact target schema version."""
        envelope = replace(
            make_envelope("ORE.STRING", "converted-run", "ore.converter"),
            schema_version="2.0.0",
        )
        return ContractArtifact(envelope, artifact.payload.upper())

    validator.register_converter("ORE.STRING", "1.0.0", "2.0.0", convert)
    source_artifact = ContractArtifact(
        make_envelope("ORE.STRING", "run-1", "ore.source"), "hello"
    )
    result = validator.validate_connection(
        "ore.source", "ore.target", "out", "in", artifact=source_artifact
    )
    assert result.status is CompatibilityStatus.COMPATIBLE_WITH_ADAPTER
    assert result.adapter_required is True
    assert result.structural_validation is True


def test_compatibility_graph_exports_only_validated_edges_and_coverage():
    """Create JSON graph data with one validated connection and exact coverage counts."""
    catalog = ContractCatalog()
    catalog.register_model("ORE.STRING", "1.0.0", str)
    registry = ModuleRegistry(catalog)
    producer = _manifest("ore.source", "1.0.0", output=True)
    consumer = _manifest("ore.target", "1.0.0", input_versions=("1.0.0",))
    registry.register(producer)
    registry.register(consumer)
    registry.bind_backend("ore.source", _backend(producer))
    registry.bind_backend("ore.target", _backend(consumer))
    validator = CompatibilityValidator(registry)
    graph = validator.build_compatibility_map()
    assert len(graph["edges"]) == 1
    assert graph["interface_validation_coverage"] == 1.0
    assert '"DIRECTLY_COMPATIBLE"' in validator.export_json()
