"""Phase II, Step 8 - separate registration from executable capability.

Module readiness is the conjunction ``R_m = M_m and B_m and V_m and D_m and E_m``
where ``M`` is a valid manifest, ``B`` a compatible bound backend, ``V`` a
passed validation, ``D`` ready dependencies and ``E`` explicit enablement.  A
manifest without a bound backend is *unavailable* for execution, and capability
discovery exposes the full state, never only a module name.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from ore_contracts.module_registry import ModuleRegistry

from .core import new_id, sha256_digest, utc_now_iso

READINESS_SCHEMA = "ore.point24.readiness.v1"


@dataclass(frozen=True, slots=True)
class ModuleReadiness:
    """The Step 8 conjunction with every component separately evidenced."""

    module_id: str
    manifest_valid: bool
    backend_bound: bool
    validation_passed: bool
    dependencies_ready: bool
    explicitly_enabled: bool

    @property
    def runnable(self) -> bool:
        """R_m: true only when every required condition is independently met."""
        return (
            self.manifest_valid
            and self.backend_bound
            and self.validation_passed
            and self.dependencies_ready
            and self.explicitly_enabled
        )

    def failed_conditions(self) -> tuple[str, ...]:
        reasons = []
        if not self.manifest_valid:
            reasons.append("manifest missing or invalid")
        if not self.backend_bound:
            reasons.append("no compatible backend bound")
        if not self.validation_passed:
            reasons.append("validation not passed")
        if not self.dependencies_ready:
            reasons.append("required dependencies not ready")
        if not self.explicitly_enabled:
            reasons.append("module not enabled")
        return tuple(reasons)

    def to_dict(self) -> dict[str, Any]:
        return {
            "module_id": self.module_id,
            "manifest_valid": self.manifest_valid,
            "backend_bound": self.backend_bound,
            "validation_passed": self.validation_passed,
            "dependencies_ready": self.dependencies_ready,
            "explicitly_enabled": self.explicitly_enabled,
            "runnable": self.runnable,
            "failed_conditions": list(self.failed_conditions()),
        }


class ReadinessTracker:
    """Compute and cache Step 8 readiness states from a live ModuleRegistry.

    The tracker is a read-only facade: it never mutates registry state and it
    treats a manifest, a backend binding, validation, dependencies and
    enablement as five distinct facts.
    """

    def __init__(self, registry: ModuleRegistry) -> None:
        if not isinstance(registry, ModuleRegistry):
            raise TypeError("registry must be an ore_contracts ModuleRegistry")
        self.registry = registry
        self._cache: dict[str, ModuleReadiness] = {}

    def inspect(self, module_id: str) -> ModuleReadiness:
        """Evaluate R_m for one module from registry evidence only."""
        manifest = self.registry.get_manifest(module_id)
        diagnostics = self.registry.diagnostics(module_id)
        manifest_valid = manifest is not None
        backend_bound = diagnostics.get("backend_status") == "BOUND"
        validation_passed = bool(diagnostics.get("backend_validated"))
        dependencies_ready = bool(diagnostics.get("dependency_complete", True)) and manifest is not None
        explicitly_enabled = not bool(diagnostics.get("disabled"))
        readiness = ModuleReadiness(
            module_id=module_id,
            manifest_valid=manifest_valid,
            backend_bound=backend_bound,
            validation_passed=validation_passed,
            dependencies_ready=dependencies_ready,
            explicitly_enabled=explicitly_enabled,
        )
        self._cache[module_id] = readiness
        return readiness

    def inspect_all(self) -> tuple[ModuleReadiness, ...]:
        return tuple(
            self.inspect(manifest.identity.module_id)
            for manifest in self.registry.list_manifests()
        )

    def runnable_modules(self) -> tuple[str, ...]:
        return tuple(
            readiness.module_id for readiness in self.inspect_all() if readiness.runnable
        )

    def unavailable_modules(self) -> tuple[str, ...]:
        """Modules that are registered but have no executable backend."""
        return tuple(
            readiness.module_id
            for readiness in self.inspect_all()
            if not readiness.runnable
        )

    def report(self) -> dict[str, Any]:
        entries = [entry.to_dict() for entry in self.inspect_all()]
        payload = {
            "schema": READINESS_SCHEMA,
            "report_id": new_id("point24_readiness"),
            "generated_at": utc_now_iso(),
            "modules": entries,
            "runnable_count": sum(1 for entry in entries if entry["runnable"]),
            "total": len(entries),
        }
        payload["digest"] = sha256_digest(payload)
        return payload


def capability_discovery(
    registry: ModuleRegistry,
    capability: str,
) -> dict[str, Any]:
    """Expose capability discovery with the full state, not only a name."""
    records = registry.find_by_capability(capability)
    return {
        "capability": capability,
        "declarations": [
            {
                "module_id": record.module_id,
                "declared": record.declared,
                "registered": record.registered,
                "backend_bound": record.backend_bound,
                "validated": record.validated,
                "disabled": record.disabled,
                "runnable": record.runnable,
            }
            for record in records
        ],
        "any_runnable": any(record.runnable for record in records),
    }


__all__ = [
    "READINESS_SCHEMA",
    "ModuleReadiness",
    "ReadinessTracker",
    "capability_discovery",
]
