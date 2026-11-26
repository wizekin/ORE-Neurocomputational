"""PQC migration policy, inventory and interoperability testing (steps 52-53, 56-57).

``CryptoPolicy`` decides which algorithms may be used for which purpose, enforcing
minimum security categories and hybrid requirements for long-lived confidentiality
(harvest-now/decrypt-later exposure). ``CryptoInventory`` aggregates the algorithms
actually used by the KMS, signing service and transport. ``MigrationManager``
prioritizes migration by data lifetime and tracks hybrid interoperability tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..exceptions import AlgorithmRejectedError
from .identifiers import (
    AlgorithmIdentifier,
    parse_algorithm_identifier,
    registered_algorithm_ids,
)


@dataclass(frozen=True, slots=True)
class CryptoPolicy:
    """Allowed algorithms and agility rules; unknown identifiers are rejected."""

    allowed_algorithm_ids: frozenset[str] = frozenset(registered_algorithm_ids())
    min_security_category: int = 1
    require_hybrid_for_long_lived: bool = True
    long_lived_threshold_years: float = 10.0

    def check(self, raw_identifier: str, *, purpose: str,
              data_lifetime_years: float = 0.0) -> AlgorithmIdentifier:
        """Validate one algorithm for a purpose; reject downgrade and weak choices."""
        identifier = parse_algorithm_identifier(raw_identifier)
        if identifier.algorithm_id not in self.allowed_algorithm_ids:
            raise AlgorithmRejectedError(
                f"algorithm {identifier.algorithm_id} is not in the allowed set"
            )
        if identifier.security_category < self.min_security_category:
            raise AlgorithmRejectedError(
                f"algorithm {identifier.algorithm_id} is security category "
                f"{identifier.security_category}, below the required "
                f"{self.min_security_category}"
            )
        if (self.require_hybrid_for_long_lived
                and identifier.kind == "key-establishment"
                and not identifier.is_pqc
                and data_lifetime_years >= self.long_lived_threshold_years):
            raise AlgorithmRejectedError(
                f"classical {identifier.algorithm_id} alone cannot protect "
                f"confidentiality over {data_lifetime_years} years "
                "(harvest-now/decrypt-later); require a hybrid establishment"
            )
        if purpose not in {"key-establishment", "signature", "encryption", "hash", "any"}:
            raise AlgorithmRejectedError(f"unknown purpose {purpose!r}")
        if purpose != "any" and identifier.kind != purpose:
            raise AlgorithmRejectedError(
                f"algorithm {identifier.algorithm_id} is a {identifier.kind} "
                f"primitive and cannot serve purpose {purpose!r}"
            )
        return identifier


class CryptoInventory:
    """Inventory of cryptographic dependencies feeding the migration manager."""

    def __init__(self) -> None:
        self._usages: list[dict[str, Any]] = []

    def record_usage(self, component: str, raw_identifier: str, *,
                     purpose: str, data_lifetime_years: float = 0.0) -> dict[str, Any]:
        """Record one algorithm usage observed in the system (step 52)."""
        identifier = parse_algorithm_identifier(raw_identifier)
        entry = {
            "component": component,
            "algorithm_id": identifier.algorithm_id,
            "kind": identifier.kind,
            "is_pqc": identifier.is_pqc,
            "security_category": identifier.security_category,
            "purpose": purpose,
            "data_lifetime_years": data_lifetime_years,
            "harvest_now_decrypt_later_risk": (
                identifier.kind == "key-establishment"
                and not identifier.is_pqc
                and data_lifetime_years >= 10.0
            ),
        }
        self._usages.append(entry)
        return entry

    def usages(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._usages)

    def report(self) -> dict[str, Any]:
        """Summarize families in use and long-lived confidentiality risks."""
        families = sorted({entry["algorithm_id"] for entry in self._usages})
        risks = [entry for entry in self._usages if entry["harvest_now_decrypt_later_risk"]]
        return {
            "usage_count": len(self._usages),
            "algorithms_in_use": families,
            "pq_algorithms_in_use": sorted(
                {entry["algorithm_id"] for entry in self._usages if entry["is_pqc"]}
            ),
            "long_lived_classical_establishment": [
                {"component": entry["component"], "algorithm": entry["algorithm_id"],
                 "lifetime_years": entry["data_lifetime_years"]}
                for entry in risks
            ],
        }


MIGRATION_STATUSES = ("DEPLOYED", "HYBRID", "MIGRATING", "PLANNED", "RETIRED", "NOT_STARTED")


class MigrationManager:
    """Prioritize and track the post-quantum migration (step 53, 57)."""

    def __init__(self, inventory: CryptoInventory) -> None:
        self.inventory = inventory
        self._plans: dict[str, dict[str, Any]] = {}

    def prioritize(self) -> list[dict[str, Any]]:
        """Rank migrations by exposure: long-lived confidentiality first."""
        report = self.inventory.report()
        priorities: list[dict[str, Any]] = []
        for risk in report["long_lived_classical_establishment"]:
            priorities.append({
                "component": risk["component"],
                "current_algorithm": risk["algorithm"],
                "priority": "HIGH",
                "driver": "harvest-now/decrypt-later over "
                          f"{risk['lifetime_years']} years",
            })
        for entry in self.inventory.usages():
            if entry["kind"] == "signature" and not entry["is_pqc"]:
                priorities.append({
                    "component": entry["component"],
                    "current_algorithm": entry["algorithm_id"],
                    "priority": "MEDIUM",
                    "driver": "long-lived signature verification and slow upgrade cycles",
                })
        priorities.sort(key=lambda item: {"HIGH": 0, "MEDIUM": 1}.get(item["priority"], 2))
        return priorities

    def set_status(self, algorithm_id: str, status: str, *, note: str = "") -> None:
        """Record the migration status for one algorithm."""
        if status not in MIGRATION_STATUSES:
            raise ValueError(f"status must be one of {MIGRATION_STATUSES}")
        parse_algorithm_identifier(algorithm_id)
        self._plans[algorithm_id] = {"status": status, "note": note}

    def status_report(self) -> dict[str, Any]:
        return {
            "priorities": self.prioritize(),
            "plans": dict(self._plans),
            "standardized_families": ["ML-KEM", "ML-DSA", "SLH-DSA"],
            "policy_note": "use vetted standardized libraries; never implement PQC primitives independently",
        }
