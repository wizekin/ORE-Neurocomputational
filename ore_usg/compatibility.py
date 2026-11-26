"""Explicit USG/ORE contract compatibility checks and reviewed migrations."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Callable, Mapping

from .contracts import (
    ORE_USG_BUNDLE_CONTRACT,
    ORE_USG_KNOWLEDGE_CONTRACT,
    ORE_USG_LEXICON_CONTRACT,
    ORE_USG_QUERY_CONTRACT,
    ORE_USG_RETRIEVAL_CONTRACT,
    USG_KNOWLEDGE_CONTRACT,
    USG_KNOWLEDGE_SCHEMA_VERSION,
)

SUPPORTED_ORE_CONTRACT_VERSIONS = {
    ORE_USG_BUNDLE_CONTRACT: "0.1.0",
    ORE_USG_LEXICON_CONTRACT: "1.0.0",
    ORE_USG_KNOWLEDGE_CONTRACT: "1.0.0",
    ORE_USG_QUERY_CONTRACT: "1.0.0",
    ORE_USG_RETRIEVAL_CONTRACT: "1.0.0",
    "ORE.BLACKBOARD.USG_MODULE_KNOWLEDGE": "1.0.0",
}


class USGCompatibilityError(ValueError):
    """Raised when an imported resource cannot be interpreted under known contracts."""

    def __init__(self, issues: tuple[str, ...]) -> None:
        self.issues = issues
        super().__init__("; ".join(issues))


def inspect_usg_compatibility(
    bundle: Mapping[str, Any],
    *,
    expected_contract: str = USG_KNOWLEDGE_CONTRACT,
    expected_schema_version: str = USG_KNOWLEDGE_SCHEMA_VERSION,
) -> tuple[str, ...]:
    """Return incompatibilities without attempting implicit conversion or migration."""
    issues: list[str] = []
    contract = bundle.get("contract", bundle.get("contract_id", expected_contract))
    version = bundle.get(
        "schema_version",
        bundle.get(
            "version",
            contract.rsplit("/", 1)[-1] if isinstance(contract, str) and "/" in contract else None,
        ),
    )
    if contract != expected_contract:
        issues.append(f"unsupported USG contract {contract!r}; expected {expected_contract!r}")
    if version != expected_schema_version:
        issues.append(
            f"unsupported USG schema version {version!r}; expected {expected_schema_version!r}"
        )
    return tuple(issues)


def require_usg_compatibility(bundle: Mapping[str, Any]) -> None:
    """Raise if a bundle declares a USG contract/schema outside the supported profile."""
    issues = inspect_usg_compatibility(bundle)
    if issues:
        raise USGCompatibilityError(issues)


def inspect_ore_contract_catalog(catalog: Any) -> tuple[str, ...]:
    """Report required ORE contracts that are absent or registered at another version."""
    getter = getattr(catalog, "get_contract", None)
    if not callable(getter):
        raise TypeError("catalog must expose get_contract(contract_id, version)")
    issues = []
    for contract_id, version in SUPPORTED_ORE_CONTRACT_VERSIONS.items():
        if getter(contract_id, version) is None:
            issues.append(f"ORE contract {contract_id}@{version} is not registered")
    return tuple(issues)


def require_ore_contract_catalog(catalog: Any) -> None:
    """Raise when the live ORE contract catalogue cannot satisfy the integration."""
    issues = inspect_ore_contract_catalog(catalog)
    if issues:
        raise USGCompatibilityError(issues)


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _semantic_projection(bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Project source-backed knowledge fields; ignore only export timestamps/format metadata."""
    excluded = {
        "created_at", "exported_at", "generated_at", "migration_history",
        "contract", "contract_id", "schema_version",
    }
    return {
        key: value for key, value in bundle.items()
        if key not in excluded
    }


@dataclass(frozen=True, slots=True)
class USGMigration:
    """One explicitly reviewed transformation between two named USG contract versions."""

    migration_id: str
    source_contract: str
    target_contract: str
    transform: Callable[[Mapping[str, Any]], Mapping[str, Any]]
    semantic_effect: str
    description: str

    def __post_init__(self) -> None:
        if not self.migration_id.strip() or not self.source_contract.strip() or not self.target_contract.strip():
            raise ValueError("migration ID and source/target contracts must be non-empty")
        if not callable(self.transform):
            raise TypeError("migration transform must be callable")
        if self.semantic_effect not in {"PRESERVED", "DECLARED_CHANGE"}:
            raise ValueError("semantic_effect must be PRESERVED or DECLARED_CHANGE")
        if not self.description.strip():
            raise ValueError("migration description is required")
        if self.semantic_effect == "DECLARED_CHANGE" and len(self.description.strip()) < 20:
            raise ValueError("semantic-changing migrations require a meaningful explanation")


@dataclass(frozen=True, slots=True)
class USGMigrationResult:
    """Detached migrated bundle and auditable digest/history information."""

    bundle: Mapping[str, Any]
    source_digest: str
    target_digest: str
    migration_ids: tuple[str, ...]
    semantic_change_declared: bool


class USGMigrationRegistry:
    """Apply only an explicitly supplied, contiguous migration chain."""

    def __init__(self) -> None:
        self._migrations: dict[str, USGMigration] = {}

    def register(self, migration: USGMigration) -> None:
        if not isinstance(migration, USGMigration):
            raise TypeError("migration must be a USGMigration")
        if migration.migration_id in self._migrations:
            raise ValueError(f"duplicate migration ID: {migration.migration_id}")
        self._migrations[migration.migration_id] = migration

    def apply(
        self,
        bundle: Mapping[str, Any],
        *,
        migration_ids: tuple[str, ...] | list[str],
    ) -> USGMigrationResult:
        if not isinstance(bundle, Mapping):
            raise TypeError("bundle must be a mapping")
        if not migration_ids:
            raise ValueError("an explicit non-empty migration chain is required")
        current: Mapping[str, Any] = json.loads(json.dumps(bundle, ensure_ascii=False, default=str))
        source_digest = _canonical_digest(current)
        for migration_id in migration_ids:
            migration = self._migrations.get(migration_id)
            if migration is None:
                raise KeyError(f"unknown migration: {migration_id}")
            actual_contract = current.get("contract", current.get("contract_id", migration.source_contract))
            if actual_contract != migration.source_contract:
                raise USGCompatibilityError((
                    f"migration {migration_id} expects {migration.source_contract}, got {actual_contract}",
                ))
            before = _semantic_projection(current)
            transformed = migration.transform(current)
            if not isinstance(transformed, Mapping):
                raise TypeError(f"migration {migration_id} must return a mapping")
            detached = json.loads(json.dumps(transformed, ensure_ascii=False, default=str))
            after = _semantic_projection(detached)
            if migration.semantic_effect == "PRESERVED" and before != after:
                raise USGCompatibilityError((
                    f"migration {migration_id} declared PRESERVED but changed source-backed semantics",
                ))
            actual_target = detached.get("contract", detached.get("contract_id"))
            if actual_target != migration.target_contract:
                raise USGCompatibilityError((
                    f"migration {migration_id} declared target {migration.target_contract}, "
                    f"but produced {actual_target!r}",
                ))
            current = detached
        target_digest = _canonical_digest(current)
        return USGMigrationResult(
            bundle=current,
            source_digest=source_digest,
            target_digest=target_digest,
            migration_ids=tuple(migration_ids),
            semantic_change_declared=any(
                self._migrations[item].semantic_effect == "DECLARED_CHANGE"
                for item in migration_ids
            ),
        )


__all__ = [
    "SUPPORTED_ORE_CONTRACT_VERSIONS",
    "USGCompatibilityError",
    "USGMigration",
    "USGMigrationRegistry",
    "USGMigrationResult",
    "inspect_ore_contract_catalog",
    "inspect_usg_compatibility",
    "require_ore_contract_catalog",
    "require_usg_compatibility",
]
