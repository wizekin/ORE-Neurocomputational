"""Point 24 integration tests: the three-tier adapter framework and the 50-step orchestrator.

These tests exercise the real integration surface: the 23 point packages from
"23 Extra ORE Algorithms" are registered and bound through the honest
three-tier adapter framework, an end-to-end cognitive task runs through the
canonical ore_contracts coordinator, and the UnifiedOREIntegrator executes
all 50 steps of the Point 24 algorithm producing a reproducible audit.

When the 23 point packages are absent from the repository, the tier tests
degrade to honest absence assertions (no fabrication), matching the spec's
evidence-state rules.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ore_contracts import register_existing_contracts  # noqa: E402
from ore_contracts.adapters import make_envelope  # noqa: E402
from ore_contracts.contracts_registry import ContractCatalog  # noqa: E402
from ore_contracts.coordinator import CognitiveTask, CognitiveTaskCoordinator  # noqa: E402
from ore_contracts.module_artifacts import ContractArtifact  # noqa: E402
from ore_contracts.module_registry import ModuleRegistry  # noqa: E402

from ore_point24 import (  # noqa: E402
    NATIVE_TIER_PACKAGES,
    PointIntegrationManager,
    UnifiedOREIntegrator,
)
from ore_point24 import adapters as p24_adapters  # noqa: E402
from ore_point24.coverage import assert_full_point_coverage  # noqa: E402
from ore_point24.spec import POINT_FAMILIES  # noqa: E402


def _point_packages_present() -> bool:
    for family in POINT_FAMILIES:
        package = family.package
        if not package or package == "ore_point24":
            continue
        try:
            importlib.import_module(package)
        except ImportError:
            return False
    return True


POINT_PACKAGES_PRESENT = _point_packages_present()
pytestmark_point_packages = pytest.mark.skipif(
    not POINT_PACKAGES_PRESENT,
    reason="the 23 point packages are not installed in this repository",
)


def _fresh_registry() -> tuple[ContractCatalog, ModuleRegistry]:
    catalog = ContractCatalog()
    register_existing_contracts(catalog)
    return catalog, ModuleRegistry(catalog)


def test_point24_module_registers_and_executes():
    """The Point 24 self-module is bound and emits a typed coverage artifact."""
    from ore_point24.compatibility import register_point24_contracts
    from ore_point24.execution import build_run_context
    catalog, registry = _fresh_registry()
    register_point24_contracts(catalog)
    module = p24_adapters.Point24IntegrationModule(registry)
    registry.register(module.describe())
    registry.bind_backend(module.MODULE_ID, module)
    backend = registry.get_backend(module.MODULE_ID)
    result = backend.execute({}, build_run_context(run_id="test-p24"))
    assert result.status.value == "success"
    assert result.artifacts[0].envelope.contract_id == "P24-COVERAGE"
    assert result.artifacts[0].payload["registered_modules"] >= 1


def test_status_adapter_probe_detects_missing_symbols():
    """The status adapter honestly reports missing expected symbols (partial state)."""
    adapter = p24_adapters.PointModuleAdapter(
        point=99, family="demo", package="ore_point24",
        expected_symbols=("UnifiedOREIntegrator", "DoesNotExistAnywhere"),
    )
    from ore_point24.execution import build_run_context
    result = adapter.execute({}, build_run_context(run_id="test-probe"))
    assert result.status.value == "partial"
    assert "DoesNotExistAnywhere" in result.warnings[0]
    validation = adapter.validate_binding()
    assert validation["valid"] is False
    assert validation["missing_symbols"] == ["DoesNotExistAnywhere"]


@pytest.mark.parametrize("package", sorted(p24_adapters.STATUS_TIER_SYMBOLS))
def test_status_tier_packages_importable(package):
    """Every status-tier package imports and exposes its expected symbols."""
    module = importlib.import_module(package)
    for symbol in p24_adapters.STATUS_TIER_SYMBOLS[package]:
        assert hasattr(module, symbol), f"{package}.{symbol} missing"


@pytest.mark.parametrize("package", NATIVE_TIER_PACKAGES)
def test_native_tier_packages_have_registration_surface(package):
    """Every native-tier package exposes its own ORE registration function."""
    if package == "ore_rl":
        from ore_rl.modules import register_rl_example_modules  # noqa: F401
    elif package == "ore_science":
        from ore_science.registration import register_science_module  # noqa: F401
    elif package == "ore_temporal":
        from ore_temporal.registration import register_ore_temporal_module  # noqa: F401
    elif package == "ore_agentic":
        from ore_agentic.registration import register_ore_agentic_module  # noqa: F401
    elif package == "ore_learning":
        from ore_learning.registration import register_ore_learning_module  # noqa: F401
    elif package == "ore_runtime":
        from ore_runtime.registration import register_runtime_module  # noqa: F401
    elif package == "ore_blackboard":
        from ore_blackboard import register_specialist_modules  # noqa: F401
    elif package == "ore_formal_reasoning":
        from ore_formal_reasoning.bridge import FormalReasoningSpecialist  # noqa: F401


@ pytestmark_point_packages
def test_full_three_tier_integration():
    """All 23 point packages integrate through the three honest tiers."""
    catalog, registry = _fresh_registry()
    manager = PointIntegrationManager(registry)
    report = manager.integrate_all()
    assert report.failures == {}
    # native tier: 8 packages with their own backends
    assert len(report.native_modules) == 8
    assert report.native_modules["ore_rl"], "ore_rl example modules missing"
    assert report.native_modules["ore_blackboard"], "blackboard specialists missing"
    # manifest-only tier: ore_security registered without a backend
    assert report.manifest_only_modules == ["ore.security.trust"]
    assert registry.get_backend("ore.security.trust") is None
    # status tier: the remaining 14 points
    assert len(report.status_modules) == 14
    # point 24 self-module
    assert registry.get_backend("point24.integration") is not None


@ pytestmark_point_packages
def test_end_to_end_task_through_canonical_coordinator():
    """An end-to-end task runs the rl chain plus the point 24 module to RESOLVED."""
    catalog, registry = _fresh_registry()
    manager = PointIntegrationManager(registry)
    manager.integrate_all()
    from ore_rl.modules import CONTRACT_INPUT_TEXT
    artifact = ContractArtifact(
        make_envelope(CONTRACT_INPUT_TEXT, "p24-e2e-run", "ore_point24"), 
        "the gradient governs the response",
    )
    task = CognitiveTask(
        goal="Point 24 end-to-end integration demonstration",
        required_capabilities=(
            "cognition.rl.input_analysis",
            "cognition.rl.knowledge_retrieval",
            "cognition.rl.hypothesis_generation",
            "cognition.rl.verification",
            "point24.unified_integration",
        ),
        input_artifacts={"source_text": artifact},
        expected_outputs=("ORE.RL.VERIFIED_CONCLUSION", "P24-COVERAGE"),
    )
    coordinator = CognitiveTaskCoordinator(registry, catalog, enable_logging=False, max_workers=2)
    result = coordinator.coordinate(task)
    assert result.status == "RESOLVED"
    assert result.plan.executable is True
    assert "ore.rl.verifier" in result.module_results
    assert result.module_results["ore.rl.verifier"].status.value == "success"
    assert "point24.integration" in result.module_results
    assert len(result.artifacts) >= 5
    # the ledger records honest evidence states with execution evidence
    ledger = manager.build_capability_ledger(executed_module_ids=result.module_results.keys())
    assert_full_point_coverage(ledger)
    counts = ledger.state_counts()
    assert counts["implemented_and_tested"] >= 2  # points 10 and 24
    assert counts["registered_backend_unavailable"] >= 1  # ore_security, honestly


@ pytestmark_point_packages
def test_orchestrator_runs_all_fifty_steps():
    """The UnifiedOREIntegrator executes all 50 steps and produces a full audit."""
    integrator = UnifiedOREIntegrator(repository_root=str(REPO_ROOT))
    audit = integrator.run()
    assert audit["executed_step_count"] == 50
    assert audit["total_step_count"] == 50
    steps = {step["step"] for step in audit["steps"]}
    assert steps == set(range(1, 51))
    # the audit distinguishes final statuses instead of claiming completion
    final_status = audit["records"]["audit"]["final_status"]
    assert final_status["assessment_complete"] == "yes"
    assert final_status["externally_validated"] == "not claimed"
    traceability = audit["records"]["audit"]["traceability"]
    assert traceability["denominator"] == 24
    assert traceability["numerator"] == 24
    # coverage matrix over evidence states, never a binary flag
    coverage = audit["records"]["coverage_matrix"]["coverage_by_evidence_state"]
    assert sum(coverage.values()) == 24


def test_orchestrator_absent_packages_stay_honest(tmp_path, monkeypatch):
    """Without the point packages the audit still runs with honest absence states."""
    # Simulate the absence of every point package: the ledger must record
    # PROPOSED (or registered-unavailable) states, never fabricated presence.
    from ore_point24.spec import POINT_FAMILIES
    point_packages = {
        family.package for family in POINT_FAMILIES
        if family.package and family.package != "ore_point24"
    }
    monkeypatch.setattr(
        p24_adapters, "_package_importable",
        lambda package: package not in point_packages,
    )
    integrator = UnifiedOREIntegrator(repository_root=str(tmp_path))
    audit = integrator.run()
    # Steps that require the point packages record their failure explicitly
    # instead of fabricating evidence.
    ledger_counts = audit["records"]["coverage_matrix"]["coverage_by_evidence_state"]
    # Only this package (point 24 itself) may reach implemented_and_tested;
    # the 23 external points must honestly degrade without fabricated presence.
    assert ledger_counts["implemented_and_tested"] <= 1
    point_states = audit["records"]["coverage_matrix"]["point_states"]
    external_tested = [
        key for key, record in point_states.items()
        if record["evidence_state"] == "implemented_and_tested" and record["point"] != 24
    ]
    assert external_tested == []
    # the run completes with every step recorded as executed or skipped
    assert audit["total_step_count"] == 50
    assert audit["executed_step_count"] + len(
        [s for s in audit["steps"] if s["status"] == "skipped"]
    ) == 50
