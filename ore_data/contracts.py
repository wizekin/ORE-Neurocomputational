"""Data contracts and authority boundaries for the Point 18 infrastructure (Phase A).

Phase A establishes what each ORE data family is (Step 2), how records are
canonically serialized and identified with domain separation (Step 3), which
exactness class every operation requires (Step 4), how fresh derived structures
must stay (Step 5), and the resource and error budgets approximate structures
must respect (Step 6). Step 1's store inventory is the audit record every
later decision must cite: no proposed structure is adopted before the current
workload is measured.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

#: Domain-separation prefix for every content identifier in this package.
ORE_DATA_IDENTITY_DOMAIN = "ORE.DATA.IDENTITY"


class DataClass(str, Enum):
    """Correctness and mutability contract of one data family (Step 2)."""

    EXACT_TRANSACTIONAL = "exact_transactional"
    APPEND_ONLY_EVENT = "append_only_event"
    IMMUTABLE_ARTIFACT = "immutable_artifact"
    APPROXIMATE_TELEMETRY = "approximate_telemetry"
    DERIVED_INDEX = "derived_index"
    CACHE = "cache"


class ExactnessClass(str, Enum):
    """The correctness contract an operation requires (Step 4)."""

    EXACT_MEMBERSHIP = "exact_membership"
    BOUNDED_ERROR_ESTIMATE = "bounded_error_estimate"
    CANDIDATE_SIMILARITY = "candidate_similarity"
    CRYPTOGRAPHIC_INTEGRITY = "cryptographic_integrity"


class UpdateMode(str, Enum):
    """How a derived structure tracks its authoritative source (Step 5)."""

    SYNCHRONOUS = "synchronous"
    TRANSACTIONAL = "transactional"
    QUEUED = "queued"
    PERIODIC_RECONCILE = "periodic_reconcile"


class MutabilityContract:
    """Which lifecycle operations a data family permits (Step 2)."""

    __slots__ = ("deletable", "correctable", "versioned", "replicable")

    def __init__(
        self,
        *,
        deletable: bool,
        correctable: bool,
        versioned: bool,
        replicable: bool,
    ) -> None:
        """Record the four lifecycle permissions of one family."""
        object.__setattr__(
            self,
            "deletable",
            bool(deletable),
        )
        object.__setattr__(self, "correctable", bool(correctable))
        object.__setattr__(self, "versioned", bool(versioned))
        object.__setattr__(self, "replicable", bool(replicable))

    def __setattr__(self, name: str, value: Any) -> None:  # pragma: no cover - guard
        raise AttributeError("mutability contracts are immutable")

    def to_dict(self) -> dict[str, bool]:
        """Return the permission record as a plain dictionary."""
        return {
            "deletable": self.deletable,
            "correctable": self.correctable,
            "versioned": self.versioned,
            "replicable": self.replicable,
        }


@dataclass(frozen=True, slots=True)
class StoreDescriptor:
    """One measured data store or index in the Step 1 inventory."""

    store_id: str
    store_kind: str
    authoritative_owner: str
    schema_summary: str
    approximate_write_rate_per_s: float
    approximate_read_rate_per_s: float
    data_volume_bytes: int
    retention_rule: str
    backup_behavior: str
    measured: bool = False

    def __post_init__(self) -> None:
        """Validate the inventory record (Step 1)."""
        if not self.store_id.strip():
            raise ValueError("store_id must be non-empty")
        if not self.authoritative_owner.strip():
            raise ValueError("authoritative_owner must be non-empty")
        if self.approximate_write_rate_per_s < 0 or self.approximate_read_rate_per_s < 0:
            raise ValueError("measured rates must be non-negative")
        if self.data_volume_bytes < 0:
            raise ValueError("data_volume_bytes must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        """Return the store record as a plain dictionary."""
        return {
            "store_id": self.store_id,
            "store_kind": self.store_kind,
            "authoritative_owner": self.authoritative_owner,
            "schema_summary": self.schema_summary,
            "approximate_write_rate_per_s": self.approximate_write_rate_per_s,
            "approximate_read_rate_per_s": self.approximate_read_rate_per_s,
            "data_volume_bytes": self.data_volume_bytes,
            "retention_rule": self.retention_rule,
            "backup_behavior": self.backup_behavior,
            "measured": self.measured,
        }


@dataclass(frozen=True, slots=True)
class DataStoreInventory:
    """The Step 1 audit record of every current ORE data store and index."""

    stores: tuple[StoreDescriptor, ...] = ()

    def add(self, store: StoreDescriptor) -> None:
        """One cannot mutate a frozen inventory; use :meth:`with_store`."""
        raise TypeError("inventories are immutable; use with_store()")

    def with_store(self, store: StoreDescriptor) -> "DataStoreInventory":
        """Return a new inventory including one more store record."""
        return DataStoreInventory(stores=self.stores + (store,))

    def as_dicts(self) -> list[dict[str, Any]]:
        """Return every store record as a plain dictionary."""
        return [store.to_dict() for store in self.stores]

    def owners(self) -> tuple[str, ...]:
        """Return the distinct authoritative owners in this inventory."""
        return tuple(dict.fromkeys(store.authoritative_owner for store in self.stores))


@dataclass(frozen=True, slots=True)
class FreshnessContract:
    """Maximum acceptable lag between authoritative writes and a derived structure (Step 5)."""

    derived_structure: str
    update_mode: UpdateMode
    max_lag_events: int

    def __post_init__(self) -> None:
        """Validate the freshness contract."""
        if not self.derived_structure.strip():
            raise ValueError("derived_structure must be non-empty")
        if isinstance(self.update_mode, str) and not isinstance(self.update_mode, UpdateMode):
            raise TypeError("update_mode must be an UpdateMode")
        if self.max_lag_events < 0:
            raise ValueError("max_lag_events must be non-negative")

    def within_lag(self, observed_lag: int) -> bool:
        """Return whether one observed lag satisfies this contract."""
        return observed_lag <= self.max_lag_events

    def to_dict(self) -> dict[str, Any]:
        """Return the contract as a plain dictionary."""
        return {
            "derived_structure": self.derived_structure,
            "update_mode": self.update_mode.value,
            "max_lag_events": self.max_lag_events,
        }


@dataclass(frozen=True, slots=True)
class ErrorBudget:
    """Resource and error budget of one approximate structure (Step 6)."""

    structure: str
    memory_budget_bytes: int
    expected_cardinality: int
    target_error_rate: float
    update_rate_per_s: float
    query_rate_per_s: float
    rebuild_cost_estimate_s: float
    config_version: str = "0.1.0"

    def __post_init__(self) -> None:
        """Validate the budget (Step 6: versioned configuration)."""
        if not self.structure.strip():
            raise ValueError("structure must be non-empty")
        if self.memory_budget_bytes <= 0:
            raise ValueError("memory_budget_bytes must be positive")
        if self.expected_cardinality <= 0:
            raise ValueError("expected_cardinality must be positive")
        if not 0.0 < self.target_error_rate < 1.0:
            raise ValueError("target_error_rate must be within (0, 1)")
        for name in ("update_rate_per_s", "query_rate_per_s", "rebuild_cost_estimate_s"):
            value = float(getattr(self, name))
            if value < 0 or not math.isfinite(value):
                raise ValueError(f"{name} must be finite and non-negative")

    def to_dict(self) -> dict[str, Any]:
        """Return the budget as a plain dictionary."""
        return {
            "structure": self.structure,
            "memory_budget_bytes": self.memory_budget_bytes,
            "expected_cardinality": self.expected_cardinality,
            "target_error_rate": self.target_error_rate,
            "update_rate_per_s": self.update_rate_per_s,
            "query_rate_per_s": self.query_rate_per_s,
            "rebuild_cost_estimate_s": self.rebuild_cost_estimate_s,
            "config_version": self.config_version,
        }


@dataclass(frozen=True, slots=True)
class DataFamilyContract:
    """One data family's contract across Steps 2, 4, 5 and 6."""

    family_id: str
    data_class: DataClass
    exactness: ExactnessClass
    authoritative_owner: str
    mutability: MutabilityContract
    freshness: FreshnessContract | None = None
    error_budget: ErrorBudget | None = None

    def __post_init__(self) -> None:
        """Validate the family contract."""
        if not self.family_id.strip():
            raise ValueError("family_id must be non-empty")
        if not isinstance(self.data_class, DataClass):
            raise TypeError("data_class must be a DataClass")
        if not isinstance(self.exactness, ExactnessClass):
            raise TypeError("exactness must be an ExactnessClass")
        if self.data_class is DataClass.DERIVED_INDEX and self.freshness is None:
            raise ValueError(
                "derived-index families must declare a freshness contract (Step 5)"
            )
        if self.data_class is DataClass.APPROXIMATE_TELEMETRY and self.error_budget is None:
            raise ValueError(
                "approximate-telemetry families must declare an error budget (Step 6)"
            )

    def approximate_may_replace_authority(self) -> bool:
        """Always False: an approximate structure never replaces the authority (Step 2)."""
        return False

    def to_dict(self) -> dict[str, Any]:
        """Return the family contract as a plain dictionary."""
        return {
            "family_id": self.family_id,
            "data_class": self.data_class.value,
            "exactness": self.exactness.value,
            "authoritative_owner": self.authoritative_owner,
            "mutability": self.mutability.to_dict(),
            "freshness": self.freshness.to_dict() if self.freshness else None,
            "error_budget": self.error_budget.to_dict() if self.error_budget else None,
        }


class DataContractRegistry:
    """Registry of every data-family contract (Steps 2, 4, 5, 6)."""

    def __init__(self) -> None:
        """Create an empty contract registry."""
        self._families: dict[str, DataFamilyContract] = {}

    def register(self, contract: DataFamilyContract) -> None:
        """Register one family contract; identifiers stay unique."""
        if contract.family_id in self._families:
            raise ValueError(f"duplicate family contract {contract.family_id!r}")
        self._families[contract.family_id] = contract

    def require(self, family_id: str) -> DataFamilyContract:
        """Return one family contract or raise for unclassified families (Step 2)."""
        if family_id not in self._families:
            raise KeyError(
                f"family {family_id!r} has no data contract; classify it before use (Step 2)"
            )
        return self._families[family_id]

    def families(self) -> tuple[DataFamilyContract, ...]:
        """Return every registered family contract."""
        return tuple(self._families.values())

    def as_dicts(self) -> list[dict[str, Any]]:
        """Return every family contract as a plain dictionary."""
        return [contract.to_dict() for contract in self._families.values()]


# ---------------------------------------------------------------------------


def _check_json_compatible(value: Any) -> None:
    """Raise for values JSON canonicalization cannot serialize deterministically."""
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            raise ValueError("canonical records cannot contain NaN or infinite floats")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"canonical record keys must be strings, got {type(key).__name__}")
            _check_json_compatible(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _check_json_compatible(item)
        return
    raise ValueError(
        f"canonical records must be JSON-compatible; got {type(value).__name__}"
    )


def canonicalize(record: Mapping[str, Any], schema_version: str = "0.1.0") -> bytes:
    """Canonicalize one record into deterministic bytes (Step 3).

    Canonicalization defines field order (sorted keys), separators, UTF-8
    encoding, null handling (kept as explicit nulls) and the schema version
    that frames the record: two byte-identical artifacts produce the same
    content identifier under the chosen hash algorithm.
    """
    if not isinstance(record, Mapping):
        raise TypeError("canonicalize expects a mapping record")
    _check_json_compatible(dict(record))
    payload = json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return payload.encode("utf-8")


def domain_tag(domain: str, schema_version: str = "0.1.0") -> bytes:
    """Return the length-prefixed domain tag framing one hash namespace (Step 3).

    ``id(b) = H(domain-tag || b)`` with domain separation so different object
    types can never accidentally share one hash namespace. The length prefix
    keeps the framing unambiguous for any byte sequence.
    """
    if not domain.strip():
        raise ValueError("domain must be non-empty")
    tag = f"{len(domain)}|{domain}|{schema_version}|"
    return tag.encode("utf-8")


def content_id(
    payload: bytes,
    domain: str = ORE_DATA_IDENTITY_DOMAIN,
    schema_version: str = "0.1.0",
) -> str:
    """Return the domain-separated SHA-256 content identifier of one payload (Step 3)."""
    if not isinstance(payload, (bytes, bytearray)):
        raise TypeError("content_id hashes bytes")
    digest = hashlib.sha256(domain_tag(domain, schema_version) + bytes(payload)).hexdigest()
    return f"sha256:{digest}"


def canonical_key(record_type: str, entity_key: str) -> str:
    """Return the stable composite key of one record (Step 3)."""
    if not record_type.strip() or not entity_key.strip():
        raise ValueError("record_type and entity_key must be non-empty")
    return f"{record_type}:{entity_key}"


def verify_content_id(payload: bytes, expected: str, domain: str = ORE_DATA_IDENTITY_DOMAIN) -> bool:
    """Recompute one payload's content identifier and compare it with the expected value."""
    return content_id(payload, domain) == expected


# ---------------------------------------------------------------------------


def default_data_contracts() -> DataContractRegistry:
    """Build the canonical family contracts of the Point 18 integration set.

    The families mirror source section 1.3: USG authoritative semantics, the
    shared workspace, the ore_logger execution record, retrieval telemetry,
    the derived approximate indexes, immutable artifacts and caches. Every
    approximate family declares its error budget (Step 6) and every derived
    family its freshness contract (Step 5).
    """
    registry = DataContractRegistry()
    rows = (
        DataFamilyContract(
            family_id="ore.usg.semantic_records",
            data_class=DataClass.EXACT_TRANSACTIONAL,
            exactness=ExactnessClass.EXACT_MEMBERSHIP,
            authoritative_owner="ore_usg",
            mutability=MutabilityContract(deletable=True, correctable=True, versioned=True, replicable=True),
        ),
        DataFamilyContract(
            family_id="ore.workspace.findings",
            data_class=DataClass.EXACT_TRANSACTIONAL,
            exactness=ExactnessClass.EXACT_MEMBERSHIP,
            authoritative_owner="ore_workspace",
            mutability=MutabilityContract(deletable=False, correctable=True, versioned=True, replicable=True),
        ),
        DataFamilyContract(
            family_id="ore.logger.execution_events",
            data_class=DataClass.APPEND_ONLY_EVENT,
            exactness=ExactnessClass.EXACT_MEMBERSHIP,
            authoritative_owner="ore_logger",
            mutability=MutabilityContract(deletable=False, correctable=False, versioned=True, replicable=True),
        ),
        DataFamilyContract(
            family_id="ore.data.retrieval_telemetry",
            data_class=DataClass.APPROXIMATE_TELEMETRY,
            exactness=ExactnessClass.BOUNDED_ERROR_ESTIMATE,
            authoritative_owner="ore_data",
            mutability=MutabilityContract(deletable=False, correctable=False, versioned=False, replicable=False),
            freshness=FreshnessContract("retrieval_telemetry", UpdateMode.PERIODIC_RECONCILE, 10_000),
            error_budget=ErrorBudget("retrieval_telemetry", 1 << 20, 100_000, 0.01, 100.0, 50.0, 5.0),
        ),
        DataFamilyContract(
            family_id="ore.data.bloom_membership",
            data_class=DataClass.DERIVED_INDEX,
            exactness=ExactnessClass.BOUNDED_ERROR_ESTIMATE,
            authoritative_owner="ore_data",
            mutability=MutabilityContract(deletable=False, correctable=False, versioned=True, replicable=False),
            freshness=FreshnessContract("bloom_membership", UpdateMode.QUEUED, 1_000),
            error_budget=ErrorBudget("bloom_membership", 1 << 21, 50_000, 0.01, 200.0, 1_000.0, 3.0),
        ),
        DataFamilyContract(
            family_id="ore.data.similarity_candidates",
            data_class=DataClass.DERIVED_INDEX,
            exactness=ExactnessClass.CANDIDATE_SIMILARITY,
            authoritative_owner="ore_data",
            mutability=MutabilityContract(deletable=False, correctable=False, versioned=True, replicable=False),
            freshness=FreshnessContract("similarity_candidates", UpdateMode.QUEUED, 5_000),
            error_budget=ErrorBudget("similarity_candidates", 1 << 22, 200_000, 0.05, 50.0, 100.0, 30.0),
        ),
        DataFamilyContract(
            family_id="ore.data.immutable_artifacts",
            data_class=DataClass.IMMUTABLE_ARTIFACT,
            exactness=ExactnessClass.CRYPTOGRAPHIC_INTEGRITY,
            authoritative_owner="ore_data.content_store",
            mutability=MutabilityContract(deletable=False, correctable=False, versioned=True, replicable=True),
        ),
        DataFamilyContract(
            family_id="ore.data.hot_cache",
            data_class=DataClass.CACHE,
            exactness=ExactnessClass.BOUNDED_ERROR_ESTIMATE,
            authoritative_owner="ore_data",
            mutability=MutabilityContract(deletable=True, correctable=False, versioned=False, replicable=False),
            freshness=FreshnessContract("hot_cache", UpdateMode.PERIODIC_RECONCILE, 0),
            error_budget=ErrorBudget("hot_cache", 1 << 20, 10_000, 0.02, 500.0, 500.0, 1.0),
        ),
    )
    for contract in rows:
        registry.register(contract)
    return registry


def default_store_inventory(measured: bool = False) -> DataStoreInventory:
    """Build the Step 1 inventory of the repository's current data stores.

    The descriptors record the stores the Point 18 pipeline must extend
    rather than replace; the numeric rates are placeholders until measured on
    a real workload (``measured=False`` until a workload audit supplies real
    numbers, exactly as Step 1 requires).
    """
    inventory = DataStoreInventory()
    for store in (
        StoreDescriptor(
            store_id="ore.usg.knowledge",
            store_kind="in-memory USG knowledge records",
            authoritative_owner="ore_usg",
            schema_summary="USG-KNOWLEDGE/0.1.0 records, sources, evidence, contexts, relations",
            approximate_write_rate_per_s=0.0,
            approximate_read_rate_per_s=0.0,
            data_volume_bytes=0,
            retention_rule="versioned bundles, superseded versions deprecated",
            backup_behavior="bundle snapshots with digests",
            measured=measured,
        ),
        StoreDescriptor(
            store_id="ore.logger.sqlite",
            store_kind="SQLite execution event database",
            authoritative_owner="ore_logger",
            schema_summary="execution events, annotations, provenance rows",
            approximate_write_rate_per_s=0.0,
            approximate_read_rate_per_s=0.0,
            data_volume_bytes=0,
            retention_rule="append-only execution history",
            backup_behavior="SQLite file copy",
            measured=measured,
        ),
        StoreDescriptor(
            store_id="ore.workspace.persistence",
            store_kind="workspace state persistence",
            authoritative_owner="ore_workspace",
            schema_summary="versioned findings, hypotheses, task state, revisions",
            approximate_write_rate_per_s=0.0,
            approximate_read_rate_per_s=0.0,
            data_volume_bytes=0,
            retention_rule="revision history with merge policies",
            backup_behavior="workspace snapshots",
            measured=measured,
        ),
        StoreDescriptor(
            store_id="ore.artifacts.directory",
            store_kind="artifact directory",
            authoritative_owner="experiment_workbench",
            schema_summary="execution bundles, model artifacts, reports",
            approximate_write_rate_per_s=0.0,
            approximate_read_rate_per_s=0.0,
            data_volume_bytes=0,
            retention_rule="validated artifacts kept by policy",
            backup_behavior="manual archive",
            measured=measured,
        ),
    ):
        inventory = inventory.with_store(store)
    return inventory


def required_exactness_for(operation: str) -> ExactnessClass:
    """Classify one operation's exactness requirement (Step 4).

    USG concept identity resolution requires exact authoritative lookup;
    approximate similarity may only nominate candidates; frequency and
    cardinality are bounded-error estimates; artifact and snapshot handling
    demands cryptographic integrity.
    """
    table = {
        "concept_identity_resolution": ExactnessClass.EXACT_MEMBERSHIP,
        "exact_record_lookup": ExactnessClass.EXACT_MEMBERSHIP,
        "audit_event_count": ExactnessClass.EXACT_MEMBERSHIP,
        "usage_frequency_estimate": ExactnessClass.BOUNDED_ERROR_ESTIMATE,
        "distinct_concept_estimate": ExactnessClass.BOUNDED_ERROR_ESTIMATE,
        "latency_percentile": ExactnessClass.BOUNDED_ERROR_ESTIMATE,
        "heavy_hitter_ranking": ExactnessClass.BOUNDED_ERROR_ESTIMATE,
        "duplicate_candidate_search": ExactnessClass.CANDIDATE_SIMILARITY,
        "similar_record_search": ExactnessClass.CANDIDATE_SIMILARITY,
        "artifact_integrity": ExactnessClass.CRYPTOGRAPHIC_INTEGRITY,
        "snapshot_consistency": ExactnessClass.CRYPTOGRAPHIC_INTEGRITY,
    }
    if operation not in table:
        raise ValueError(
            f"operation {operation!r} has no declared exactness class (Step 4); "
            "declare it before routing"
        )
    return table[operation]


__all__ = [
    "ORE_DATA_IDENTITY_DOMAIN",
    "DataClass",
    "ExactnessClass",
    "UpdateMode",
    "MutabilityContract",
    "StoreDescriptor",
    "DataStoreInventory",
    "FreshnessContract",
    "ErrorBudget",
    "DataFamilyContract",
    "DataContractRegistry",
    "canonicalize",
    "domain_tag",
    "content_id",
    "canonical_key",
    "verify_content_id",
    "default_data_contracts",
    "default_store_inventory",
    "required_exactness_for",
]
