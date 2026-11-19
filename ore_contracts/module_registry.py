"""Manifest registry, runtime backend bindings, capability discovery, and repository inventory."""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
import threading
from typing import Any, Mapping

from .contracts_registry import ContractCatalog
from .module_manifest import ModuleManifest


@dataclass(frozen=True, slots=True)
class BackendBinding:
    """Record an executable backend independently from its registered manifest."""

    module_id: str
    backend: Any = field(compare=False, repr=False)
    validated: bool = False
    validation_method: str = "protocol checks"
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CapabilityRecord:
    """Expose declared capability and runtime availability as separate, inspectable states."""

    capability: str
    module_id: str
    declared: bool
    registered: bool
    backend_bound: bool
    validated: bool
    disabled: bool
    runnable: bool


class ModuleRegistry:
    """Maintain versioned module manifests and runtime bindings without rewriting processors."""

    def __init__(self, contract_catalog: ContractCatalog, event_logger: Any | None = None) -> None:
        """Create an empty registry connected to the shared contract catalog and optional SQLite logger."""
        if not isinstance(contract_catalog, ContractCatalog):
            raise TypeError("contract_catalog must be a ContractCatalog")
        self.contract_catalog = contract_catalog
        self.event_logger = event_logger
        self._manifests: dict[str, ModuleManifest] = {}
        self._bindings: dict[str, BackendBinding] = {}
        self._disabled: set[str] = set()
        self._lock = threading.RLock()

    def _record(self, method_name: str, metadata: Mapping[str, Any]) -> None:
        """Write a method-linked audit event when an event logger was explicitly configured."""
        if self.event_logger is not None:
            self.event_logger.record(
                method_name, dict(metadata), class_name="ModuleRegistry",
                module_name="ore_contracts.module_registry",
            )

    @staticmethod
    def _matches_data_type(definition: Any, declared_type: str) -> bool:
        """Ensure a port's declared payload type agrees with its registered contract model."""
        if declared_type in {"object", "Any", "typing.Any"} or definition.model_type is None:
            return True
        model_types = definition.model_type if isinstance(definition.model_type, tuple) else (definition.model_type,)
        return declared_type in {
            f"{model.__module__}.{model.__qualname__}" for model in model_types
        }

    def register(self, manifest: ModuleManifest, *, replace: bool = False) -> ModuleManifest:
        """Validate every manifest reference and register one identity without binding a backend."""
        if not isinstance(manifest, ModuleManifest):
            raise TypeError("manifest must be a ModuleManifest")
        module_id = manifest.identity.module_id
        for port in manifest.input_ports:
            for version in port.accepted_versions:
                definition = self.contract_catalog.get_contract(port.contract_id, version)
                if definition is None:
                    raise ValueError(f"input port {port.port_id} references unknown contract {port.contract_id}@{version}")
                if not self._matches_data_type(definition, port.expected_data_type):
                    raise ValueError(
                        f"input port {port.port_id} data type {port.expected_data_type} "
                        f"does not match contract model {definition.model_type}"
                    )
        for port in manifest.output_ports:
            definition = self.contract_catalog.get_contract(port.contract_id, port.contract_version)
            if definition is None:
                raise ValueError(f"output port {port.port_id} references unknown contract {port.contract_id}@{port.contract_version}")
            if not self._matches_data_type(definition, port.output_data_type):
                raise ValueError(
                    f"output port {port.port_id} data type {port.output_data_type} "
                    f"does not match contract model {definition.model_type}"
                )
        with self._lock:
            if module_id in self._manifests and not replace:
                raise ValueError(f"module identity is already registered: {module_id}")
            if module_id in self._bindings and replace:
                raise ValueError("cannot replace a manifest while a backend is bound; unbind it first")
            self._manifests[module_id] = manifest
            self._disabled.discard(module_id)
        self._record("register", {
            "module_id": module_id, "module_version": manifest.identity.module_version,
            "capability_count": len(manifest.capabilities), "backend_bound": False,
        })
        return manifest

    def get_manifest(self, module_id: str) -> ModuleManifest | None:
        """Return the registered manifest, or None when the identity is unknown."""
        with self._lock:
            return self._manifests.get(module_id)

    def bind_backend(self, module_id: str, backend: Any) -> BackendBinding:
        """Bind a protocol-compatible backend only after its declared manifest matches."""
        with self._lock:
            manifest = self._manifests.get(module_id)
            if manifest is None:
                raise KeyError(f"module manifest is not registered: {module_id}")
        describe = getattr(backend, "describe", None)
        execute = getattr(backend, "execute", None)
        if not callable(describe) or not callable(execute):
            raise TypeError("backend must expose callable describe() and execute(inputs, context)")
        described = describe()
        if not isinstance(described, ModuleManifest):
            raise TypeError("backend.describe() must return a ModuleManifest")
        if described.identity != manifest.identity:
            raise ValueError("backend identity/version does not match the registered manifest")
        if described.input_ports != manifest.input_ports or described.output_ports != manifest.output_ports:
            raise ValueError("backend ports do not match the registered manifest")
        if described.capabilities != manifest.capabilities:
            raise ValueError("backend capabilities do not match the registered manifest")
        validation_method = "protocol and manifest structural checks"
        validated = True
        validate_binding = getattr(backend, "validate_binding", None)
        if callable(validate_binding):
            result = validate_binding()
            if isinstance(result, Mapping):
                validated = bool(result.get("valid", False))
                validation_method = str(result.get("method", "backend validation hook"))
                diagnostics = dict(result)
            else:
                validated = bool(result)
                validation_method = "backend validation hook"
                diagnostics = {"valid": validated}
            if not validated:
                raise ValueError("backend validation hook rejected the binding")
        else:
            diagnostics = {"valid": True, "method": validation_method}
        binding = BackendBinding(
            module_id=module_id, backend=backend, validated=validated,
            validation_method=validation_method, diagnostics=diagnostics,
        )
        with self._lock:
            self._bindings[module_id] = binding
        self._record("bind_backend", {
            "module_id": module_id, "backend_type": f"{type(backend).__module__}.{type(backend).__qualname__}",
            "validated": validated, "validation_method": validation_method,
        })
        return binding

    def unbind_backend(self, module_id: str) -> bool:
        """Remove only the runtime binding while preserving the module manifest."""
        with self._lock:
            removed = self._bindings.pop(module_id, None) is not None
        self._record("unbind_backend", {"module_id": module_id, "removed": removed})
        return removed

    def get_backend(self, module_id: str) -> Any | None:
        """Return a bound backend only when present; a manifest by itself is never executable."""
        with self._lock:
            binding = self._bindings.get(module_id)
            return binding.backend if binding is not None else None

    def set_enabled(self, module_id: str, enabled: bool) -> None:
        """Enable or disable a registered module without deleting its declared capabilities."""
        with self._lock:
            if module_id not in self._manifests:
                raise KeyError(f"unknown module: {module_id}")
            if enabled:
                self._disabled.discard(module_id)
            else:
                self._disabled.add(module_id)
        self._record("set_enabled", {"module_id": module_id, "enabled": bool(enabled)})

    def find_by_capability(self, capability: str) -> tuple[CapabilityRecord, ...]:
        """Return every declaring module and distinguish manifest, backend, validation, and enabled states."""
        if not capability.strip():
            raise ValueError("capability must be non-empty")
        with self._lock:
            records = []
            for module_id, manifest in self._manifests.items():
                if capability not in manifest.capabilities:
                    continue
                binding = self._bindings.get(module_id)
                disabled = module_id in self._disabled
                dependencies_ready = all(
                    dependency.module_id in self._manifests
                    and dependency.module_id in self._bindings
                    and dependency.module_id not in self._disabled
                    and self._bindings[dependency.module_id].validated
                    for dependency in manifest.dependencies if dependency.required
                )
                records.append(CapabilityRecord(
                    capability=capability, module_id=module_id, declared=True, registered=True,
                    backend_bound=binding is not None, validated=bool(binding and binding.validated),
                    disabled=disabled,
                    runnable=bool(binding and binding.validated and not disabled and dependencies_ready),
                ))
        return tuple(sorted(records, key=lambda item: item.module_id))

    def list_manifests(self) -> tuple[ModuleManifest, ...]:
        """Return manifests sorted by stable module ID."""
        with self._lock:
            return tuple(self._manifests[key] for key in sorted(self._manifests))

    def diagnostics(self, module_id: str) -> dict[str, Any]:
        """Report declared ports/capabilities, backend state, and dependency readiness."""
        with self._lock:
            manifest = self._manifests.get(module_id)
            binding = self._bindings.get(module_id)
            disabled = module_id in self._disabled
            known_ids = set(self._manifests)
        if manifest is None:
            raise KeyError(f"unknown module: {module_id}")
        required = [item for item in manifest.dependencies if item.required]
        available = [
            item for item in required
            if item.module_id in known_ids and item.module_id in self._bindings
            and item.module_id not in self._disabled and self._bindings[item.module_id].validated
        ]
        completeness = len(available) / len(required) if required else 1.0
        dependencies_ready = len(available) == len(required)
        return {
            "module_id": module_id,
            "module_version": manifest.identity.module_version,
            "manifest_status": "REGISTERED",
            "backend_status": "NOT_BOUND" if binding is None else "BOUND",
            "backend_validated": bool(binding and binding.validated),
            "runnable": bool(binding and binding.validated and not disabled and dependencies_ready),
            "disabled": disabled,
            "capabilities": list(manifest.capabilities),
            "input_ports": [item.port_id for item in manifest.input_ports],
            "output_ports": [item.port_id for item in manifest.output_ports],
            "dependencies": [
                {
                    "module_id": item.module_id, "required": item.required,
                    "available": item in available if item.required else (
                        item.module_id in known_ids and item.module_id in self._bindings
                        and item.module_id not in self._disabled
                        and self._bindings[item.module_id].validated
                    ),
                }
                for item in manifest.dependencies
            ],
            "dependency_completeness": completeness,
            "dependency_complete": len(available) == len(required),
            "limitations": list(manifest.limitations),
        }

    def inventory_repository(self, repository_root: str | Path) -> tuple[dict[str, Any], ...]:
        """Inspect actual Python source in each ore_* package without promoting docs to capabilities."""
        root = Path(repository_root).expanduser().resolve()
        if not root.is_dir():
            raise NotADirectoryError(str(root))
        inventory: list[dict[str, Any]] = []
        for package_dir in sorted(
            path for path in root.glob("ore_*")
            if path.is_dir() and (path / "__init__.py").is_file()
        ):
            source_files = sorted(package_dir.rglob("*.py"))
            classes: set[str] = set()
            processors: set[str] = set()
            request_types: set[str] = set()
            result_types: set[str] = set()
            imported_ore_packages: set[str] = set()
            entry_points: set[str] = set()
            mode_evidence: set[str] = set()
            parse_errors: list[str] = []
            for source_file in source_files:
                relative = source_file.relative_to(root).as_posix()
                try:
                    text = source_file.read_text(encoding="utf-8")
                    tree = ast.parse(text, filename=relative)
                except (OSError, UnicodeDecodeError, SyntaxError) as exc:
                    parse_errors.append(f"{relative}: {type(exc).__name__}: {exc}")
                    continue
                if source_file.name == "__init__.py":
                    for node in tree.body:
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and not node.name.startswith("_"):
                            entry_points.add(node.name)
                        elif isinstance(node, ast.ImportFrom):
                            for alias in node.names:
                                if not alias.name.startswith("_"):
                                    entry_points.add(alias.asname or alias.name)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        classes.add(node.name)
                        methods = {item.name for item in node.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))}
                        if methods & {"process", "execute", "run"}:
                            processors.add(node.name)
                        if node.name.endswith("Request"):
                            request_types.add(node.name)
                        if node.name.endswith(("Result", "Response")):
                            result_types.add(node.name)
                    elif isinstance(node, ast.Import):
                        imported_ore_packages.update(alias.name.split(".")[0] for alias in node.names if alias.name.startswith("ore_"))
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        if node.module.startswith("ore_"):
                            imported_ore_packages.add(node.module.split(".")[0])
                    elif isinstance(node, ast.Name):
                        lowered = node.id.casefold()
                        if "biolog" in lowered or "lif" in lowered or "neural" in lowered:
                            mode_evidence.add("BIOLOGICAL")
                        if "functional" in lowered or "process" in lowered or "execute" in lowered:
                            mode_evidence.add("FUNCTIONAL")
                if "simulate_biology" in text or "Biological" in text or "LIF" in text:
                    mode_evidence.add("BIOLOGICAL")
                if "def process(" in text or "def execute(" in text:
                    mode_evidence.add("FUNCTIONAL")
            modes = (
                ("HYBRID",) if {"BIOLOGICAL", "FUNCTIONAL"}.issubset(mode_evidence)
                else tuple(sorted(mode_evidence & {"BIOLOGICAL", "FUNCTIONAL"}))
            )
            inventory.append({
                "module_id": package_dir.name.replace("_", ".", 1),
                "package_path": package_dir.name,
                "python_files": len(source_files),
                "public_entry_points": sorted(entry_points),
                "classes": sorted(classes),
                "processor_classes": sorted(processors),
                "request_types": sorted(request_types),
                "result_types": sorted(result_types),
                "internal_ore_dependencies": sorted(imported_ore_packages - {package_dir.name}),
                "execution_modes_evidence": list(modes),
                "parse_errors": parse_errors,
                "implementation_observed_in_source": bool(source_files) and not parse_errors,
                "capabilities": [],
                "capability_status": "NOT_DECLARED",
            })
        return tuple(inventory)

    def registry_coverage(self, total_identified_modules: int) -> float:
        """Calculate validated manifest coverage over source-inventoried modules."""
        if total_identified_modules < 0:
            raise ValueError("total_identified_modules cannot be negative")
        with self._lock:
            validated_manifests = len(self._manifests)
        return validated_manifests / total_identified_modules if total_identified_modules else 1.0

    def registered_count(self) -> int:
        """Return the number of registered manifests, independently of bound backends."""
        with self._lock:
            return len(self._manifests)
