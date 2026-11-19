"""Typed identity, ports, dependencies, and manifests for universal ORE communication."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Mapping


_MODULE_ID = re.compile(r"^[a-z][a-z0-9_.-]*$")
_CAPABILITY_ID = re.compile(r"^[a-z][a-z0-9_-]*(?:\.[a-z0-9_-]+)+$")
_VERSION = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$")


def validate_semantic_version(version: str, field_name: str = "version") -> None:
    """Require a complete, explicit semantic version instead of guessing compatibility."""
    if not isinstance(version, str) or not _VERSION.fullmatch(version):
        raise ValueError(f"{field_name} must be a full semantic version such as 1.0.0")


@dataclass(frozen=True, slots=True)
class ModuleIdentity:
    """Give a cognitive module a stable identity independent of its Python location."""

    module_id: str
    module_version: str
    module_name: str
    parent_module: str | None = None
    package_path: str | None = None
    biological_reference: str | None = None

    def __post_init__(self) -> None:
        """Validate stable IDs, semantic versions, and optional descriptive fields."""
        if not _MODULE_ID.fullmatch(self.module_id):
            raise ValueError("module_id must be a stable lowercase namespaced identifier")
        validate_semantic_version(self.module_version, "module_version")
        if not isinstance(self.module_name, str) or not self.module_name.strip():
            raise ValueError("module_name must be non-empty")
        if self.parent_module is not None and not _MODULE_ID.fullmatch(self.parent_module):
            raise ValueError("parent_module must be a valid module identifier")
        if self.package_path is not None and not self.package_path.strip():
            raise ValueError("package_path cannot be empty when supplied")


@dataclass(frozen=True, slots=True)
class InputPort:
    """Declare one contract accepted by a module, including cardinality and validation rules."""

    port_id: str
    contract_id: str
    accepted_versions: tuple[str, ...]
    required: bool = True
    expected_data_type: str = "object"
    multiple: bool = False
    validation_requirements: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Reject incomplete ports, duplicate versions, and ambiguous input types."""
        if not self.port_id.strip() or not self.contract_id.strip():
            raise ValueError("input port_id and contract_id must be non-empty")
        if not self.accepted_versions or len(set(self.accepted_versions)) != len(self.accepted_versions):
            raise ValueError("accepted_versions must be a non-empty tuple of unique versions")
        for version in self.accepted_versions:
            validate_semantic_version(version, "accepted contract version")
        if not self.expected_data_type.strip():
            raise ValueError("expected_data_type must be non-empty")
        if any(not item.strip() for item in self.validation_requirements):
            raise ValueError("validation requirements must be non-empty strings")


@dataclass(frozen=True, slots=True)
class OutputPort:
    """Declare a produced contract and whether it is unconditional or conditional."""

    port_id: str
    contract_id: str
    contract_version: str
    output_data_type: str
    description: str
    availability: str = "always"
    availability_condition: str | None = None

    def __post_init__(self) -> None:
        """Validate the output contract and make conditional availability explicit."""
        if not all(value.strip() for value in (
            self.port_id, self.contract_id, self.output_data_type, self.description
        )):
            raise ValueError("output port fields must be non-empty")
        validate_semantic_version(self.contract_version, "contract_version")
        if self.availability not in {"always", "conditional"}:
            raise ValueError("availability must be 'always' or 'conditional'")
        if self.availability == "conditional" and not (
            isinstance(self.availability_condition, str) and self.availability_condition.strip()
        ):
            raise ValueError("conditional outputs require an availability_condition")
        if self.availability == "always" and self.availability_condition is not None:
            raise ValueError("always-available outputs must not declare a condition")


@dataclass(frozen=True, slots=True)
class ModuleDependency:
    """Declare a module dependency without implying that it is scheduled or executed."""

    module_id: str
    required: bool = True
    purpose: str = ""

    def __post_init__(self) -> None:
        """Validate the referenced module identifier and optional purpose."""
        if not _MODULE_ID.fullmatch(self.module_id):
            raise ValueError("dependency module_id must be a valid module identifier")


@dataclass(frozen=True, slots=True)
class ModuleManifest:
    """Describe a module's responsibilities, communication ports, capabilities, and dependencies."""

    identity: ModuleIdentity
    responsibilities: tuple[str, ...]
    input_ports: tuple[InputPort, ...]
    output_ports: tuple[OutputPort, ...]
    capabilities: tuple[str, ...]
    dependencies: tuple[ModuleDependency, ...] = ()
    execution_modes: tuple[str, ...] = ("FUNCTIONAL",)
    biological_reference: str | None = None
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Enforce unique ports, namespaced capabilities, and explicit execution modes."""
        if not isinstance(self.identity, ModuleIdentity):
            raise TypeError("identity must be a ModuleIdentity")
        if any(not item.strip() for item in self.responsibilities):
            raise ValueError("responsibilities must contain non-empty descriptions")
        for label, ports in (("input", self.input_ports), ("output", self.output_ports)):
            identifiers = [port.port_id for port in ports]
            if len(identifiers) != len(set(identifiers)):
                raise ValueError(f"{label} port identifiers must be unique")
        if any(not _CAPABILITY_ID.fullmatch(item) for item in self.capabilities):
            raise ValueError("capabilities must use namespaced identifiers such as language.syntax_parsing")
        if len(set(self.capabilities)) != len(self.capabilities):
            raise ValueError("capabilities must be unique within a manifest")
        allowed_modes = {"FUNCTIONAL", "BIOLOGICAL", "HYBRID", "TRAIN", "INFER", "VALIDATE"}
        if not self.execution_modes or not set(self.execution_modes).issubset(allowed_modes):
            raise ValueError(f"execution_modes must be a non-empty subset of {sorted(allowed_modes)}")
        dependency_ids = [item.module_id for item in self.dependencies]
        if len(dependency_ids) != len(set(dependency_ids)):
            raise ValueError("dependencies must not repeat a module identifier")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly manifest with enum-free, explicit port records."""
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ModuleManifest":
        """Reconstruct a manifest from a mapping while restoring typed nested records."""
        data = dict(value)
        identity = data["identity"]
        data["identity"] = identity if isinstance(identity, ModuleIdentity) else ModuleIdentity(**identity)
        data["input_ports"] = tuple(
            item if isinstance(item, InputPort) else InputPort(**{**item, "accepted_versions": tuple(item["accepted_versions"])})
            for item in data.get("input_ports", ())
        )
        data["output_ports"] = tuple(
            item if isinstance(item, OutputPort) else OutputPort(**item)
            for item in data.get("output_ports", ())
        )
        data["dependencies"] = tuple(
            item if isinstance(item, ModuleDependency) else ModuleDependency(**item)
            for item in data.get("dependencies", ())
        )
        for key in ("responsibilities", "capabilities", "execution_modes", "limitations"):
            data[key] = tuple(data.get(key, ()))
        return cls(**data)
