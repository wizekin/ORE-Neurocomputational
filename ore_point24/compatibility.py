"""Phase II, Step 7 - version every module boundary.

Register exact contract identifiers and versions, make compatibility an
explicit predicate ``K(p, q)`` between a producer output port ``p`` and a
consumer input port ``q``, and reject execution edges inferred from similar
names or approximate semantic resemblance.  Legacy modules go through
adapters; incompatible outputs are never silently coerced.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from ore_contracts.contracts_registry import ContractCatalog, ContractDefinition

from .core import sha256_digest

COMPATIBILITY_SCHEMA = "ore.point24.compatibility.v1"

# Point 24 shared contracts (exact versions; migration requires registration).
POINT24_CONTRACTS: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    ("P24-CANONICAL-ARTIFACT", "1.0.0",
     "Canonical artifact envelope payload (payload-specific schemas apply).", ()),
    ("P24-BASELINE", "1.0.0", "Immutable integration baseline manifest.",
     ("schema", "captured_at", "repository_root")),
    ("P24-COVERAGE", "1.0.0", "Point-to-layer map and capability ledger.",
     ("schema", "entries")),
    ("P24-POINT-STATUS", "1.0.0", "Point package integration status record.",
     ("point", "package", "state")),
    ("P24-EVIDENCE-CLAIM", "1.0.0", "Provenance-aware semantic claim.",
     ("claim_id", "statement", "evidence")),
    ("P24-VERIFICATION-REPORT", "1.0.0", "Independent verification report.",
     ("artifact_id", "passed", "checks")),
    ("P24-TASK-REQUEST", "1.0.0", "Typed cognitive task request T=(G,C,I,O,B,P).",
     ("goal", "constraints", "inputs", "expected_outputs")),
    ("P24-AUDIT", "1.0.0", "Final integration audit and traceability record.",
     ("schema", "baseline", "coverage")),
    ("P24-POLICY-DECISION", "1.0.0", "Deterministic admission or MAPE-K policy decision.",
     ("decision", "reason")),
    ("P24-PLAN", "1.0.0", "Candidate plan with preconditions and costs.",
     ("objective", "steps")),
)


def register_point24_contracts(catalog: ContractCatalog) -> tuple[ContractDefinition, ...]:
    """Register the shared Point 24 contracts idempotently (exact versions)."""
    registered: list[ContractDefinition] = []
    for contract_id, version, description, required in POINT24_CONTRACTS:
        existing = catalog.get_contract(contract_id, version)
        if existing is not None:
            registered.append(existing)
            continue
        registered.append(catalog.register_model(
            contract_id, version, dict,
            required_fields=required, description=description,
        ))
    return tuple(registered)


@dataclass(frozen=True, slots=True)
class PortCompatibility:
    """The explicit K(p, q) decision between one producer and one consumer."""

    producer_module: str
    producer_port: str
    producer_contract: str
    producer_version: str
    consumer_module: str
    consumer_port: str
    consumer_contract: str
    consumer_accepted_versions: tuple[str, ...]
    compatible: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "producer": f"{self.producer_module}:{self.producer_port}",
            "producer_contract": f"{self.producer_contract}@{self.producer_version}",
            "consumer": f"{self.consumer_module}:{self.consumer_port}",
            "consumer_contract": self.consumer_contract,
            "consumer_accepted_versions": list(self.consumer_accepted_versions),
            "compatible": self.compatible,
            "reason": self.reason,
        }


def check_port_compatibility(
    producer_contract_id: str,
    producer_version: str,
    consumer_contract_id: str,
    consumer_accepted_versions: Iterable[str],
    *,
    producer_module: str = "producer",
    producer_port: str = "out",
    consumer_module: str = "consumer",
    consumer_port: str = "in",
) -> PortCompatibility:
    """Evaluate K(p, q): exact contract identity and exact version membership.

    An edge is legal only for the exact declared versions and validation
    constraints.  Similar names never match: the comparison is on full
    identifiers, never on substrings or fuzzy resemblance.
    """
    accepted = tuple(consumer_accepted_versions)
    if producer_contract_id != consumer_contract_id:
        return PortCompatibility(
            producer_module, producer_port, producer_contract_id, producer_version,
            consumer_module, consumer_port, consumer_contract_id, accepted, False,
            f"contract identity mismatch: {producer_contract_id!r} != {consumer_contract_id!r}",
        )
    if producer_version not in accepted:
        return PortCompatibility(
            producer_module, producer_port, producer_contract_id, producer_version,
            consumer_module, consumer_port, consumer_contract_id, accepted, False,
            f"version {producer_version} not in accepted versions {list(accepted)}; "
            "an explicit adapter is required",
        )
    return PortCompatibility(
        producer_module, producer_port, producer_contract_id, producer_version,
        consumer_module, consumer_port, consumer_contract_id, accepted, True,
        "exact contract and version match",
    )


class CompatibilityCatalogue:
    """A catalogue of evaluated K(p, q) edges with stable digests.

    No execution edge may be inferred from similar names; every edge in the
    catalogue was produced by an exact contract identity and version check.
    """

    def __init__(self) -> None:
        self._edges: dict[str, PortCompatibility] = {}

    def evaluate(self, edge: PortCompatibility) -> PortCompatibility:
        key = f"{edge.producer_module}:{edge.producer_port}->{edge.consumer_module}:{edge.consumer_port}"
        self._edges[key] = edge
        return edge

    def legal_edges(self) -> tuple[PortCompatibility, ...]:
        return tuple(edge for edge in self._edges.values() if edge.compatible)

    def illegal_edges(self) -> tuple[PortCompatibility, ...]:
        return tuple(edge for edge in self._edges.values() if not edge.compatible)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": COMPATIBILITY_SCHEMA,
            "edges": [edge.to_dict() for edge in self._edges.values()],
        }
        payload["digest"] = sha256_digest(payload)
        return payload


__all__ = [
    "COMPATIBILITY_SCHEMA",
    "POINT24_CONTRACTS",
    "CompatibilityCatalogue",
    "PortCompatibility",
    "check_port_compatibility",
    "register_point24_contracts",
]
