"""Validation tests for the universal ORE module identity and manifest contracts."""
import pytest

from ore_contracts.module_manifest import (
    InputPort, ModuleDependency, ModuleIdentity, ModuleManifest, OutputPort,
)


def _identity(module_id="ore.test"):
    """Return a minimal stable module identity."""
    return ModuleIdentity(module_id, "1.0.0", "Test module", package_path="ore_test")


def test_identity_requires_stable_id_and_full_semantic_version():
    """Reject unstable names and incomplete versions."""
    with pytest.raises(ValueError):
        ModuleIdentity("Test Module", "1.0", "Bad")
    with pytest.raises(ValueError):
        ModuleIdentity("ore.test", "1.0", "Bad")


def test_ports_validate_contract_versions_and_output_availability():
    """Require explicit input versions and conditions for conditional outputs."""
    with pytest.raises(ValueError):
        InputPort("in", "ORE.TEXT", ())
    with pytest.raises(ValueError):
        OutputPort("out", "ORE.TEXT", "1.0.0", "builtins.str", "conditional output",
                   availability="conditional")
    input_port = InputPort("in", "ORE.TEXT", ("1.0.0",), multiple=True)
    output_port = OutputPort(
        "out", "ORE.TEXT", "1.0.0", "builtins.str", "text output",
        availability="conditional", availability_condition="text was detected",
    )
    assert input_port.multiple is True
    assert output_port.availability == "conditional"


def test_manifest_requires_namespaced_unique_capabilities_and_port_ids():
    """Keep the manifest deterministic and reject duplicate or unnamespaced declarations."""
    valid = InputPort("source", "ORE.TEXT", ("1.0.0",))
    with pytest.raises(ValueError):
        ModuleManifest(
            _identity(), ("process text",), (valid, valid), (),
            ("language.tokenize",),
        )
    with pytest.raises(ValueError):
        ModuleManifest(_identity(), (), (), (), ("tokenize",))
    manifest = ModuleManifest(
        _identity(), ("process text",), (valid,),
        (OutputPort("tokens", "ORE.TEXT", "1.0.0", "builtins.str", "normalized text"),),
        ("language.tokenize",),
        dependencies=(ModuleDependency("ore.normalizer", required=False),),
    )
    restored = ModuleManifest.from_dict(manifest.to_dict())
    assert restored == manifest
    assert restored.dependencies[0].required is False
