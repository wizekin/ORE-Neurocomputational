"""Point-package integration adapters for the 23 point implementations.

Each of the 23 point packages is integrated through one of three honest
tiers, exactly as the Point 24 specification requires (Section 1.1, Section
5 and Step 48):

- **Native tier** - packages that ship their own ``register_*_module``
  functions and ORE backends (ore_rl, ore_science, ore_temporal, ore_agentic,
  ore_learning, ore_runtime, ore_blackboard, ore_formal_reasoning).  Their
  manifests and backends are registered by their own code.
- **Manifest-only tier** - packages that register a manifest and contracts
  but ship no backend (ore_security); the manifest is registered and the
  module honestly remains *registered, backend unavailable*.
- **Status tier** - the remaining packages are bound through
  :class:`PointModuleAdapter`, a narrow integration-status adapter that
  imports the package, verifies its expected symbols and emits a typed
  ``P24-POINT-STATUS`` artifact.  It provides integration status reporting,
  never a claim that the package's cognitive capability is module-bound.

A module present in the repository is not thereby registered, backend-bound,
validated or integrated; each of those facts is evidenced separately.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from ore_contracts import ContractEnvelope, EvidenceKind, RunContext, sha256_json
from ore_contracts.module_manifest import (
    InputPort,
    ModuleIdentity,
    ModuleManifest,
    OutputPort,
)
from ore_contracts.module_protocol import ModuleResult, ModuleStatus
from ore_contracts.module_registry import ModuleRegistry
from ore_contracts.module_artifacts import ContractArtifact

from .artifacts import make_envelope
from .compatibility import register_point24_contracts
from .core import new_id, utc_now_iso
from .coverage import CapabilityLedger
from .inventory import module_state_vector
from .spec import POINT_FAMILIES, POINTS_BY_NUMBER

ADAPTERS_SCHEMA = "ore.point24.adapters.v1"
STATUS_CONTRACT = "P24-POINT-STATUS"
STATUS_CONTRACT_VERSION = "1.0.0"
BASELINE_CONTRACT = "P24-BASELINE"


# ---------------------------------------------------------------------------
# Expected symbols per status-tier package (verified by the probe, not docs)
# ---------------------------------------------------------------------------

STATUS_TIER_SYMBOLS: dict[str, tuple[str, ...]] = {
    "ore_control": ("OperatorRegistry", "PRECONDITION_LIBRARY"),
    "ore_neurosymbolic": ("AmbiguityResolver", "NSRLPolicy"),
    "ore_cwm": ("BetaBinomialEstimator", "PredictionRecord"),
    "ore_planning": ("OperatorDiscovery", "CapabilityRegistryAdapter"),
    "ore_bayesian_causal": ("NeuralPopulation", "PredictiveCodingCircuit"),
    "ore_graph_intelligence": ("CandidateGenerator", "CandidateVerifier", "PromotionGate"),
    "ore_evolution_intelligence": ("ParetoArchive", "PolicyIRGenome"),
    "ore_macl": ("AntColonyOptimizer", "PeerCoordinator"),
    "ore_hic": ("AlgoIRBuilder", "AlgoIRValidator"),
    "ore_distributed": ("ORE_DISTRIBUTED_CONTRACT", "ORE_DISTRIBUTED_VERSION"),
    "ore_data": ("ORE_DATA_CONTRACT", "PipelineConfig"),
    "ore_hacol": ("HardwareAwareOptimizer", "HardwareProfile"),
    "ore_runtime_governance": ("ComponentManifest", "CapabilityAuthority"),
    "ore_cros": ("ResourceAction", "RuntimeControlState"),
}

# Packages integrated through their own registration functions (native tier).
NATIVE_TIER_PACKAGES: tuple[str, ...] = (
    "ore_rl", "ore_science", "ore_temporal", "ore_agentic",
    "ore_learning", "ore_runtime", "ore_blackboard", "ore_formal_reasoning",
)

# Packages that register a manifest but ship no backend (manifest-only tier).
MANIFEST_ONLY_TIER_PACKAGES: tuple[str, ...] = ("ore_security",)


@dataclass(frozen=True, slots=True)
class PointModuleAdapter:
    """A narrow status adapter for one point package (spec Section 5.1).

    The adapter exposes the required module interface - stable identity and
    version, declared capabilities, exact contracts, validation state,
    backend availability and typed result status - while executing only an
    import-and-symbol probe.  It never advertises a cognitive capability
    merely because the capability is mentioned in the package documentation.
    """

    point: int
    family: str
    package: str
    module_version: str = "1.0.0"
    expected_symbols: tuple[str, ...] = ()

    @property
    def module_id(self) -> str:
        return f"point{self.point:02d}.status"

    @property
    def capability(self) -> str:
        return f"point{self.point:02d}.integration_status"

    def describe(self) -> ModuleManifest:
        """The exact manifest registered for this adapter."""
        return ModuleManifest(
            identity=ModuleIdentity(
                module_id=self.module_id,
                module_version=self.module_version,
                module_name=f"Point {self.point} integration status adapter ({self.package})",
                package_path=self.package,
            ),
            responsibilities=(
                f"Import {self.package} and verify its integration surface; "
                "report typed integration status only",
            ),
            input_ports=(
                InputPort(
                    port_id="baseline",
                    contract_id=BASELINE_CONTRACT,
                    accepted_versions=("1.0.0",),
                    required=False,
                    expected_data_type="object",
                ),
            ),
            output_ports=(
                OutputPort(
                    port_id="integration_status",
                    contract_id=STATUS_CONTRACT,
                    contract_version=STATUS_CONTRACT_VERSION,
                    output_data_type="object",
                    description=f"Point {self.point} package integration status record",
                ),
            ),
            capabilities=(self.capability,),
            limitations=(
                "status adapter only; it does not expose the package's cognitive capabilities",
            ),
        )

    def validate_binding(self) -> dict[str, Any]:
        """Import probe executed before the backend is accepted."""
        try:
            module = importlib.import_module(self.package)
        except Exception as exc:  # noqa: BLE001
            return {
                "valid": False,
                "method": "import probe",
                "error": f"{type(exc).__name__}: {exc}",
            }
        missing = [
            symbol for symbol in self.expected_symbols
            if not hasattr(module, symbol)
        ]
        return {
            "valid": not missing,
            "method": "import probe with expected-symbol verification",
            "missing_symbols": missing,
            "package_imported": True,
        }

    def _probe(self) -> dict[str, Any]:
        try:
            module = importlib.import_module(self.package)
        except Exception as exc:  # noqa: BLE001
            return {
                "point": self.point,
                "package": self.package,
                "state": "unavailable",
                "import_ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "symbols_present": [],
                "symbols_missing": list(self.expected_symbols),
                "probed_at": utc_now_iso(),
            }
        missing = [s for s in self.expected_symbols if not hasattr(module, s)]
        present = [s for s in self.expected_symbols if hasattr(module, s)]
        return {
            "point": self.point,
            "family": self.family,
            "package": self.package,
            "state": "importable_symbols_verified" if not missing else "importable_symbols_missing",
            "import_ok": True,
            "package_version": str(getattr(module, "__version__", "unreported")),
            "symbols_present": present,
            "symbols_missing": missing,
            "probed_at": utc_now_iso(),
        }

    def execute(
        self,
        inputs: Mapping[str, Any],
        context: RunContext,
    ) -> ModuleResult:
        """Run the bounded status probe and emit one typed artifact."""
        payload = self._probe()
        status = ModuleStatus.SUCCESS if payload["import_ok"] and not payload.get("symbols_missing") else (
            ModuleStatus.PARTIAL if payload["import_ok"] else ModuleStatus.UNAVAILABLE
        )
        envelope = make_envelope(
            contract_id=STATUS_CONTRACT,
            schema_version=STATUS_CONTRACT_VERSION,
            run_id=context.run_id,
            producer_id=self.module_id,
            config={"package": self.package, "point": self.point},
            evidence_kind=EvidenceKind.OBSERVED,
        )
        artifact = ContractArtifact(envelope=envelope, payload=payload)
        return ModuleResult(
            module_id=self.module_id,
            module_version=self.module_version,
            status=status,
            artifacts=(artifact,),
            diagnostics={"probe": "import_and_symbols"},
            warnings=(
                (f"missing expected symbols: {', '.join(payload.get('symbols_missing', ()))}",)
                if payload.get("symbols_missing") else ()
            ),
            provenance={"package": self.package, "probe": "import_and_symbols"},
        )


# ---------------------------------------------------------------------------
# Integration manager
# ---------------------------------------------------------------------------

class Point24IntegrationModule:
    """The Point 24 self-module: unified integration reporting.

    Point 24's own capability is the integration layer itself.  Its execute()
    emits a typed ``P24-COVERAGE`` artifact summarizing the registry state -
    registered, bound, validated and runnable module counts - so the point 24
    capability is demonstrably module-bound and testable like any other.
    """

    MODULE_ID = "point24.integration"
    MODULE_VERSION = "1.0.0"
    CAPABILITY = "point24.unified_integration"

    def __init__(self, registry: ModuleRegistry) -> None:
        self.registry = registry

    def describe(self) -> ModuleManifest:
        return ModuleManifest(
            identity=ModuleIdentity(
                module_id=self.MODULE_ID,
                module_version=self.MODULE_VERSION,
                module_name="Point 24 unified integration layer",
                package_path="ore_point24",
            ),
            responsibilities=(
                "Report the evidence-based integration state of all 24 point families",
                "Emit the typed coverage ledger artifact for audit and traceability",
            ),
            input_ports=(
                InputPort(
                    port_id="baseline",
                    contract_id=BASELINE_CONTRACT,
                    accepted_versions=("1.0.0",),
                    required=False,
                    expected_data_type="object",
                ),
            ),
            output_ports=(
                OutputPort(
                    port_id="coverage_summary",
                    contract_id="P24-COVERAGE",
                    contract_version="1.0.0",
                    output_data_type="object",
                    description="Registry-wide integration coverage summary",
                ),
            ),
            capabilities=(self.CAPABILITY,),
        )

    def validate_binding(self) -> dict[str, Any]:
        return {"valid": True, "method": "point 24 self-module binding check"}

    def execute(
        self,
        inputs: Mapping[str, Any],
        context: RunContext,
    ) -> ModuleResult:
        manifests = self.registry.list_manifests()
        entries = []
        bound = 0
        validated = 0
        runnable = 0
        for manifest in manifests:
            diagnostics = self.registry.diagnostics(manifest.identity.module_id)
            backend_status = str(diagnostics.get("backend_status", "NOT_BOUND"))
            if backend_status == "BOUND":
                bound += 1
            if diagnostics.get("backend_validated"):
                validated += 1
            if diagnostics.get("runnable"):
                runnable += 1
            entries.append({
                "module_id": manifest.identity.module_id,
                "backend_status": backend_status,
                "validated": bool(diagnostics.get("backend_validated")),
                "runnable": bool(diagnostics.get("runnable")),
            })
        payload = {
            "schema": "ore.point24.coverage_summary.v1",
            "entries": entries,
            "registered_modules": len(manifests),
            "bound_backends": bound,
            "validated_backends": validated,
            "runnable_modules": runnable,
            "reported_at": utc_now_iso(),
        }
        envelope = make_envelope(
            contract_id="P24-COVERAGE",
            schema_version="1.0.0",
            run_id=context.run_id,
            producer_id=self.MODULE_ID,
            config={"module": self.MODULE_ID},
            evidence_kind=EvidenceKind.DERIVED,
        )
        return ModuleResult(
            module_id=self.MODULE_ID,
            module_version=self.MODULE_VERSION,
            status=ModuleStatus.SUCCESS,
            artifacts=(ContractArtifact(envelope=envelope, payload=payload),),
            diagnostics={"summary": "registry-wide coverage summary"},
            provenance={"package": "ore_point24"},
        )


def _package_importable(package: str) -> bool:
    """Return whether a package is actually importable in this environment.

    Source presence is an evidenced fact, never an assumption; the helper is
    module-level so absence simulations can patch it cleanly.
    """
    if not package:
        return False
    try:
        importlib.import_module(package)
    except Exception:  # noqa: BLE001 - any import failure means absent
        return False
    return True


@dataclass(slots=True)
class PointIntegrationReport:
    """The result of integrating all point packages (Steps 2, 8, 46)."""

    native_modules: dict[str, list[str]] = field(default_factory=dict)
    manifest_only_modules: list[str] = field(default_factory=list)
    status_modules: dict[str, str] = field(default_factory=dict)  # point -> module id
    failures: dict[str, str] = field(default_factory=dict)
    contracts_registered: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": ADAPTERS_SCHEMA,
            "native_modules": {k: sorted(v) for k, v in self.native_modules.items()},
            "manifest_only_modules": sorted(self.manifest_only_modules),
            "status_modules": dict(self.status_modules),
            "failures": dict(self.failures),
            "contracts_registered": sorted(self.contracts_registered),
        }


class PointIntegrationManager:
    """Register and bind every point package through the three honest tiers.

    The manager is the Step 48 consolidation in code: it reuses
    ``ore_contracts`` (registry + catalog) as the single integration surface,
    calls each native package's own registration function, registers the
    manifest-only tier without inventing backends, and binds the status tier
    through :class:`PointModuleAdapter`.
    """

    def __init__(self, registry: ModuleRegistry) -> None:
        if not isinstance(registry, ModuleRegistry):
            raise TypeError("registry must be an ore_contracts ModuleRegistry")
        self.registry = registry
        self.catalog = registry.contract_catalog
        self.report = PointIntegrationReport()

    # -- public API --------------------------------------------------------

    def integrate_all(self, *, include_point24_contracts: bool = True) -> PointIntegrationReport:
        """Integrate every point package present in the repository."""
        if include_point24_contracts:
            definitions = register_point24_contracts(self.catalog)
            self.report.contracts_registered.extend(
                f"{d.contract_id}@{d.version}" for d in definitions
            )
        self._register_point24_self_module()
        self._integrate_native_tier()
        self._integrate_manifest_only_tier()
        self._integrate_status_tier()
        return self.report

    def _register_point24_self_module(self) -> None:
        """Register Point 24's own integration-reporting module."""
        module = Point24IntegrationModule(self.registry)
        try:
            if self.registry.get_manifest(module.MODULE_ID) is None:
                self.registry.register(module.describe())
            if self.registry.get_backend(module.MODULE_ID) is None:
                self.registry.bind_backend(module.MODULE_ID, module)
        except Exception as exc:  # noqa: BLE001
            self.report.failures["ore_point24"] = (
                f"point 24 self-module registration failed: {type(exc).__name__}: {exc}"
            )

    # -- native tier ---------------------------------------------------------

    def _integrate_native_tier(self) -> None:
        for package in NATIVE_TIER_PACKAGES:
            try:
                modules = self._register_native(package)
                self.report.native_modules[package] = modules
            except Exception as exc:  # noqa: BLE001
                self.report.failures[package] = (
                    f"native registration failed: {type(exc).__name__}: {exc}"
                )

    def _register_native(self, package: str) -> list[str]:
        if package == "ore_rl":
            from ore_rl.modules import register_rl_example_modules
            backends = register_rl_example_modules(self.catalog, self.registry)
            return sorted(backends.keys())
        if package == "ore_science":
            from ore_science.registration import register_science_module
            register_science_module(self.registry)
            return ["ore_science"]
        if package == "ore_temporal":
            from ore_temporal.registration import register_ore_temporal_module
            register_ore_temporal_module(self.registry, replace=True)
            return ["ore.temporal_intelligence"]
        if package == "ore_agentic":
            from ore_agentic.registration import register_ore_agentic_module
            register_ore_agentic_module(self.registry, replace=True)
            return ["ore.agentic_workflow"]
        if package == "ore_learning":
            from ore_learning.registration import register_ore_learning_module
            register_ore_learning_module(self.registry, replace=True)
            return ["ore.replaceable_learned_models"]
        if package == "ore_runtime":
            from ore_runtime.registration import register_runtime_module
            register_runtime_module(self.registry, self.catalog)
            return ["ore.runtime.engine"]
        if package == "ore_blackboard":
            from ore_usg import register_usg_contracts
            from ore_blackboard import (
                LexicalSpecialist,
                MemorySpecialist,
                SemanticSpecialist,
                SyntaxSpecialist,
                VerificationSpecialist,
                register_blackboard_contracts,
                register_specialist_modules,
            )
            register_usg_contracts(self.catalog)
            register_blackboard_contracts(self.catalog)
            specialists = (
                LexicalSpecialist(), SyntaxSpecialist(), SemanticSpecialist(),
                MemorySpecialist(), VerificationSpecialist(),
            )
            registration = register_specialist_modules(self.registry, self.catalog, specialists)
            return sorted(registration.registered_modules)
        if package == "ore_formal_reasoning":
            from ore_formal_reasoning.bridge import (
                FormalReasoningSpecialist,
                register_formal_reasoning_contracts,
            )
            register_formal_reasoning_contracts(self.catalog)
            specialist = FormalReasoningSpecialist()
            manifest = specialist.describe()
            if self.registry.get_manifest(manifest.identity.module_id) is None:
                self.registry.register(manifest)
            if self.registry.get_backend(manifest.identity.module_id) is None:
                self.registry.bind_backend(manifest.identity.module_id, specialist)
            return [manifest.identity.module_id]
        raise KeyError(package)  # pragma: no cover - static list is exhaustive

    # -- manifest-only tier ----------------------------------------------------

    def _integrate_manifest_only_tier(self) -> None:
        for package in MANIFEST_ONLY_TIER_PACKAGES:
            try:
                if package == "ore_security":
                    from ore_security.integration import register_with_ore
                    result = register_with_ore(catalog=self.catalog, module_registry=self.registry)
                    module_id = str(result.get("module_id", "ore.security.trust"))
                    # Honest state: manifest registered, backend unavailable.
                    self.report.manifest_only_modules.append(module_id)
            except Exception as exc:  # noqa: BLE001
                self.report.failures[package] = (
                    f"manifest registration failed: {type(exc).__name__}: {exc}"
                )

    # -- status tier ------------------------------------------------------------

    def _integrate_status_tier(self) -> None:
        for family in POINT_FAMILIES:
            package = family.package
            if not package or package in NATIVE_TIER_PACKAGES or package in MANIFEST_ONLY_TIER_PACKAGES:
                continue
            if package == "ore_point24":
                continue
            symbols = STATUS_TIER_SYMBOLS.get(package, ())
            adapter = PointModuleAdapter(
                point=family.point,
                family=family.family,
                package=package,
                expected_symbols=symbols,
            )
            try:
                if self.registry.get_manifest(adapter.module_id) is None:
                    self.registry.register(adapter.describe())
                if self.registry.get_backend(adapter.module_id) is None:
                    self.registry.bind_backend(adapter.module_id, adapter)
                self.report.status_modules[str(family.point)] = adapter.module_id
            except Exception as exc:  # noqa: BLE001
                self.report.failures[package] = (
                    f"status adapter registration failed: {type(exc).__name__}: {exc}"
                )

    # -- ledger construction ------------------------------------------------------

    def build_capability_ledger(
        self,
        *,
        executed_module_ids: Iterable[str] = (),
        validated_module_ids: Iterable[str] | None = None,
    ) -> CapabilityLedger:
        """Build the evidence-based capability ledger over the 24 points.

        ``executed_module_ids`` are modules that actually executed in an
        end-to-end coordinated task (integration-test evidence);
        ``validated_module_ids`` defaults to every bound and validated
        backend whose status probe reports a verified import.
        """
        executed = set(executed_module_ids)
        ledger = CapabilityLedger()
        for family in POINT_FAMILIES:
            package = family.package or ""
            module_ids: list[str] = []
            source_present = _package_importable(package)
            if package == "ore_point24":
                module_ids = [Point24IntegrationModule.MODULE_ID]
            elif package in self.report.native_modules:
                module_ids = list(self.report.native_modules[package])
            elif str(family.point) in self.report.status_modules:
                module_ids = [self.report.status_modules[str(family.point)]]
            elif package == "ore_security":
                module_ids = ["ore.security.trust"]
            manifest_registered = any(
                self.registry.get_manifest(module_id) is not None for module_id in module_ids
            )
            backend_bound = any(
                self.registry.get_backend(module_id) is not None for module_id in module_ids
            )
            if validated_module_ids is None:
                validated = backend_bound and all(
                    self.registry.diagnostics(module_id).get("backend_validated", False)
                    for module_id in module_ids
                )
            else:
                validated = bool(module_ids) and all(
                    module_id in set(validated_module_ids) for module_id in module_ids
                )
            integration_tested = bool(module_ids) and any(
                module_id in executed for module_id in module_ids
            )
            notes: list[str] = []
            if package == "ore_point24":
                notes.append("point 24 self-module: unified integration reporting (this package)")
            elif package in self.report.native_modules:
                notes.append("native module registration (package-provided backend)")
            elif str(family.point) in self.report.status_modules:
                notes.append(
                    "status adapter only: package importable, cognitive capability not module-bound"
                )
            elif package == "ore_security":
                notes.append("manifest registered; package ships no module backend")
            if package in self.report.failures:
                notes.append(self.report.failures[package])
            ledger.add(
                f"point{family.point:02d}.integration",
                point=family.point,
                layer=family.primary_layer,
                owner_module=", ".join(module_ids) or "<none>",
                vector=module_state_vector(
                    source_present=source_present,
                    manifest_registered=manifest_registered,
                    backend_bound=backend_bound,
                    validated=validated,
                    integration_tested=integration_tested,
                ),
                notes=notes,
            )
        return ledger


__all__ = [
    "ADAPTERS_SCHEMA",
    "MANIFEST_ONLY_TIER_PACKAGES",
    "NATIVE_TIER_PACKAGES",
    "STATUS_CONTRACT",
    "STATUS_CONTRACT_VERSION",
    "STATUS_TIER_SYMBOLS",
    "PointIntegrationManager",
    "PointIntegrationReport",
    "PointModuleAdapter",
]
