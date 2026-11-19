"""Module manifest registration, backend binding, discovery, and inventory tests."""
import pytest

from ore_contracts.contracts_registry import ContractCatalog
from ore_contracts.module_manifest import (
    InputPort, ModuleDependency, ModuleIdentity, ModuleManifest, OutputPort,
)
from ore_contracts.module_registry import ModuleRegistry


def _catalog():
    """Create two simple explicit payload contracts."""
    catalog = ContractCatalog()
    catalog.register_model("ORE.TEXT", "1.0.0", str)
    catalog.register_model("ORE.RESULT", "1.0.0", dict)
    return catalog


def _manifest(module_id="ore.reader", *, dependencies=()):
    """Construct a reader manifest using registered catalog types."""
    return ModuleManifest(
        identity=ModuleIdentity(module_id, "1.0.0", module_id, package_path=module_id.replace(".", "_")),
        responsibilities=("read input text",),
        input_ports=(InputPort("text", "ORE.TEXT", ("1.0.0",), expected_data_type="builtins.str"),),
        output_ports=(OutputPort("result", "ORE.RESULT", "1.0.0", "builtins.dict", "structured result"),),
        capabilities=("language.text_input",),
        dependencies=tuple(dependencies),
    )


class _Backend:
    """A minimal backend exposing the declared structural protocol."""

    def __init__(self, manifest):
        """Store the manifest advertised by this test backend."""
        self.manifest = manifest

    def describe(self):
        """Return the registered manifest."""
        return self.manifest

    def execute(self, inputs, context):
        """Provide a callable implementation for registry binding validation."""
        return None


def test_register_valid_module_and_reject_duplicate_or_missing_contract():
    """A manifest can register without a backend, while duplicates and unknown contracts fail."""
    registry = ModuleRegistry(_catalog())
    manifest = _manifest()
    registry.register(manifest)
    diagnostics = registry.diagnostics("ore.reader")
    assert diagnostics["manifest_status"] == "REGISTERED"
    assert diagnostics["backend_status"] == "NOT_BOUND"
    assert diagnostics["runnable"] is False
    with pytest.raises(ValueError):
        registry.register(manifest)
    bad = ModuleManifest(
        ModuleIdentity("ore.bad", "1.0.0", "Bad"),
        (), (InputPort("input", "ORE.MISSING", ("1.0.0",)),), (),
        ("language.bad",),
    )
    with pytest.raises(ValueError, match="unknown contract"):
        registry.register(bad)


def test_backend_binding_and_capability_discovery_keep_states_separate():
    """Capability search reports declared, bound, validated, disabled, and runnable separately."""
    registry = ModuleRegistry(_catalog())
    manifest = _manifest()
    registry.register(manifest)
    binding = registry.bind_backend("ore.reader", _Backend(manifest))
    assert binding.validated
    record = registry.find_by_capability("language.text_input")[0]
    assert record.declared and record.registered and record.backend_bound and record.validated
    assert record.runnable
    registry.set_enabled("ore.reader", False)
    disabled = registry.find_by_capability("language.text_input")[0]
    assert disabled.disabled and not disabled.runnable
    assert registry.unbind_backend("ore.reader")
    assert registry.get_backend("ore.reader") is None


def test_incompatible_backend_manifest_is_rejected_without_binding():
    """Backend port definitions must exactly agree with the registered module declaration."""
    registry = ModuleRegistry(_catalog())
    manifest = _manifest()
    registry.register(manifest)
    different = _manifest()
    different = ModuleManifest(
        identity=different.identity, responsibilities=different.responsibilities,
        input_ports=(), output_ports=different.output_ports,
        capabilities=different.capabilities,
    )
    with pytest.raises(ValueError, match="ports"):
        registry.bind_backend("ore.reader", _Backend(different))


def test_dependency_completeness_and_repository_inventory_are_explicit(tmp_path):
    """Compute required-dependency completeness and inspect actual Python definitions."""
    catalog = _catalog()
    registry = ModuleRegistry(catalog)
    dependency = ModuleDependency("ore.source", required=True)
    target = _manifest("ore.consumer", dependencies=(dependency,))
    source = _manifest("ore.source")
    registry.register(source)
    registry.register(target)
    registry.bind_backend("ore.source", _Backend(source))
    assert registry.diagnostics("ore.consumer")["dependency_completeness"] == 1.0
    package = tmp_path / "ore_fake"
    package.mkdir()
    (package / "__init__.py").write_text("from .pipeline import FakeProcessor\n", encoding="utf-8")
    (package / "pipeline.py").write_text(
        "class FakeRequest:\n    pass\n"
        "class FakeResult:\n    pass\n"
        "class FakeProcessor:\n    def process(self, request):\n        return FakeResult()\n",
        encoding="utf-8",
    )
    inventory = registry.inventory_repository(tmp_path)
    fake = next(item for item in inventory if item["package_path"] == "ore_fake")
    assert "FakeProcessor" in fake["processor_classes"]
    assert "FakeRequest" in fake["request_types"]
    assert "FakeResult" in fake["result_types"]
    assert fake["capability_status"] == "NOT_DECLARED"
    assert registry.registry_coverage(4) == 0.5


def test_real_repository_inventory_finds_existing_ore_packages():
    """Scan the repository's actual 60-plus package directories without inventing capabilities."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    inventory = ModuleRegistry(_catalog()).inventory_repository(root)
    package_names = {item["package_path"] for item in inventory}
    assert len(inventory) >= 50
    assert {"ore_contracts", "ore_lexical_access", "ore_syntax", "ore_semantic_composition"}.issubset(package_names)
    assert all(item["capability_status"] == "NOT_DECLARED" for item in inventory)
