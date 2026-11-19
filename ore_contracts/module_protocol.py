"""Structural Python protocol and standardized execution results for ORE modules."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence, TYPE_CHECKING, runtime_checkable

if TYPE_CHECKING:
    from .module_artifacts import ContractArtifact
    from .module_manifest import ModuleManifest
    from . import RunContext


class ModuleStatus(str, Enum):
    """Represent successful, partial, abstained, unavailable, and failed execution distinctly."""

    SUCCESS = "success"
    PARTIAL = "partial"
    ABSTAINED = "abstained"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ModuleResult:
    """Return a module identity, explicit status, artifacts, diagnostics, and provenance."""

    module_id: str
    module_version: str
    status: ModuleStatus
    artifacts: tuple["ContractArtifact", ...] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)
    processing_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate identity and normalize string statuses without hiding execution failures."""
        if not self.module_id.strip() or not self.module_version.strip():
            raise ValueError("ModuleResult requires a non-empty module identity and version")
        if not isinstance(self.status, ModuleStatus):
            object.__setattr__(self, "status", ModuleStatus(self.status))
        if any(not isinstance(item, str) or not item.strip() for item in self.warnings):
            raise ValueError("warnings must contain non-empty strings")


@runtime_checkable
class OREModuleProtocol(Protocol):
    """Define the universal structural interface without forcing legacy inheritance."""

    def describe(self) -> "ModuleManifest":
        """Return the validated manifest implemented by this module adapter."""
        ...

    def execute(
        self,
        inputs: Mapping[str, "ContractArtifact | Sequence[ContractArtifact]"],
        context: "RunContext",
    ) -> ModuleResult:
        """Execute compatible inputs under the caller-provided run context."""
        ...
