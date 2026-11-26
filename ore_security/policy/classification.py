"""Data classification and resource descriptors with handling rules (steps 4-5).

Classification follows the asset-inventory phase: every protected resource carries a
sensitivity class, an owner, allowed purposes, retention, recipients and permitted
transformations. The classification is carried through caches, logs, exports and
derived aggregates by referencing the descriptor rather than copying policy data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Mapping


class DataClassification(IntEnum):
    """Ordered sensitivity classes used by the default-deny decision predicate."""

    PUBLIC = 0
    INTERNAL = 1
    CONFIDENTIAL = 2
    HIGHLY_SENSITIVE = 3

    @classmethod
    def parse(cls, value: str | "DataClassification") -> "DataClassification":
        """Parse a classification name case-insensitively; unknown values are rejected."""
        if isinstance(value, cls):
            return value
        if not isinstance(value, str) or not value.strip():
            raise ValueError("classification must be a non-empty string")
        normalized = value.strip().upper().replace("-", "_").replace(" ", "_")
        try:
            return cls[normalized]
        except KeyError as exc:
            raise ValueError(f"unknown data classification: {value!r}") from exc


@dataclass(frozen=True, slots=True)
class ResourceDescriptor:
    """One protected resource with classification and handling rules (step 4)."""

    uri: str
    classification: DataClassification
    owner: str
    allowed_purposes: frozenset[str] = field(default_factory=frozenset)
    allowed_recipients: frozenset[str] = field(default_factory=frozenset)
    retention_days: int = 3650
    permitted_transformations: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if not self.uri.strip():
            raise ValueError("uri must be non-empty")
        if not self.owner.strip():
            raise ValueError("owner must be non-empty")
        if isinstance(self.classification, str):
            raise TypeError("classification must be a DataClassification, not a string")
        if self.retention_days < 0:
            raise ValueError("retention_days must be non-negative")


class ResourceRegistry:
    """Registry of resource descriptors consulted during authorization."""

    def __init__(self) -> None:
        self._resources: dict[str, ResourceDescriptor] = {}

    def register(self, descriptor: ResourceDescriptor, *, replace: bool = False) -> None:
        """Register a resource descriptor; replacement must be explicit."""
        if not isinstance(descriptor, ResourceDescriptor):
            raise TypeError("descriptor must be a ResourceDescriptor")
        if descriptor.uri in self._resources and not replace:
            raise ValueError(f"resource already registered: {descriptor.uri}")
        self._resources[descriptor.uri] = descriptor

    def resolve(self, uri: str) -> ResourceDescriptor | None:
        """Exact-match resolution; policy treats an unresolved resource as deny."""
        return self._resources.get(uri)

    def resolve_or_default(self, uri: str) -> ResourceDescriptor | None:
        """Resolve by exact match, then by longest registered prefix."""
        exact = self._resources.get(uri)
        if exact is not None:
            return exact
        best: ResourceDescriptor | None = None
        best_length = -1
        for candidate_uri, candidate in self._resources.items():
            if uri.startswith(candidate_uri + "/") and len(candidate_uri) > best_length:
                best, best_length = candidate, len(candidate_uri)
        return best

    def inventory_report(self) -> list[dict[str, Any]]:
        """Return the classified asset inventory for audits (step 1)."""
        return [
            {
                "uri": descriptor.uri,
                "classification": descriptor.classification.name,
                "owner": descriptor.owner,
                "allowed_purposes": sorted(descriptor.allowed_purposes),
                "retention_days": descriptor.retention_days,
            }
            for descriptor in sorted(self._resources.values(), key=lambda item: item.uri)
        ]
