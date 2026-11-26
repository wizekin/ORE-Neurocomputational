"""Post-quantum algorithm identifiers with unambiguous parsing (steps 53-55).

Tracks the NIST-standardized families: ML-KEM (key encapsulation), ML-DSA (digital
signatures) and SLH-DSA (hash-based signatures), alongside the classical algorithms
used by this layer. Parsing is strict and unambiguous: normalization is limited to
case, whitespace and separator adjustment, and anything outside the registry is
rejected rather than guessed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..exceptions import AlgorithmRejectedError

SECURITY_CATEGORIES = {1: "128-bit classical / 64-bit quantum", 3: "192-bit / 96-bit quantum",
                       5: "256-bit / 128-bit quantum"}

PQC_FAMILIES: dict[str, dict[str, int]] = {
    "ML-KEM": {"512": 1, "768": 3, "1024": 5},
    "ML-DSA": {"44": 2, "65": 3, "87": 5},
    "SLH-DSA": {"128s": 1, "128f": 1, "192s": 3, "192f": 3, "256s": 5, "256f": 5},
}

CLASSICAL_ALGORITHMS: dict[str, int] = {
    "Ed25519": 2,
    "X25519": 2,
    "AES-256-GCM": 5,
    "SHA-256": 2,
    "SHA-384": 3,
    "SHA-512": 5,
}

PQC_KINDS = {"ML-KEM": "key-establishment", "ML-DSA": "signature", "SLH-DSA": "signature"}
CLASSICAL_KINDS = {
    "Ed25519": "signature", "X25519": "key-establishment", "AES-256-GCM": "encryption",
    "SHA-256": "hash", "SHA-384": "hash", "SHA-512": "hash",
}


@dataclass(frozen=True, slots=True)
class AlgorithmIdentifier:
    """One unambiguous algorithm identifier with family, parameter set and category."""

    algorithm_id: str
    family: str
    parameter_set: str
    kind: str
    security_category: int
    is_pqc: bool

    def __post_init__(self) -> None:
        if not self.algorithm_id.strip():
            raise ValueError("algorithm_id must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm_id": self.algorithm_id,
            "family": self.family,
            "parameter_set": self.parameter_set,
            "kind": self.kind,
            "security_category": self.security_category,
            "security_note": SECURITY_CATEGORIES.get(self.security_category, ""),
            "is_pqc": self.is_pqc,
        }

    def serialize(self) -> str:
        """Round-trip-safe serialization: parse(serialize(x)) == x."""
        return self.algorithm_id


def normalize_identifier(raw: str) -> str:
    """Normalize case, whitespace and separators without guessing semantics."""
    if not isinstance(raw, str):
        raise AlgorithmRejectedError("algorithm identifier must be a string")
    import re
    compact = raw.strip().upper().replace("_", "-").replace(" ", "-")
    return re.sub(r"-+", "-", compact).strip("-")


def parse_algorithm_identifier(raw: str) -> AlgorithmIdentifier:
    """Parse strictly; unknown families or parameter sets are rejected."""
    normalized = normalize_identifier(raw)
    for family, parameter_sets in PQC_FAMILIES.items():
        if normalized == family:
            raise AlgorithmRejectedError(
                f"{raw!r} names a family without a parameter set; use e.g. {family}-768"
            )
        if normalized.startswith(family + "-"):
            parameter = normalized[len(family) + 1:]
            # match parameter sets case-insensitively but canonicalize to the
            # registered spelling (SLH-DSA uses lowercase s/f suffixes)
            canonical = next((registered for registered in parameter_sets
                              if registered.upper() == parameter.upper()), None)
            if canonical is None:
                raise AlgorithmRejectedError(
                    f"unknown {family} parameter set {parameter!r}; "
                    f"choose from {sorted(parameter_sets)}"
                )
            return AlgorithmIdentifier(
                algorithm_id=f"{family}-{canonical}", family=family,
                parameter_set=canonical, kind=PQC_KINDS[family],
                security_category=parameter_sets[canonical], is_pqc=True,
            )
    for classical, category in CLASSICAL_ALGORITHMS.items():
        if normalized == classical.upper().replace(" ", ""):
            return AlgorithmIdentifier(
                algorithm_id=classical, family=classical, parameter_set="",
                kind=CLASSICAL_KINDS[classical], security_category=category,
                is_pqc=False,
            )
    raise AlgorithmRejectedError(
        f"unsupported or ambiguous algorithm identifier: {raw!r}"
    )


def registered_algorithm_ids() -> list[str]:
    """All identifiers accepted by the parser, for inventory and policy checks."""
    identifiers = [
        f"{family}-{parameter}"
        for family, parameters in PQC_FAMILIES.items()
        for parameter in parameters
    ]
    identifiers.extend(CLASSICAL_ALGORITHMS)
    return sorted(identifiers)
