"""Canonical serialization used by every digest, signature, and proof binding.

All signed or hashed structures are serialized to deterministic UTF-8 JSON with sorted
keys, compact separators, and no whitespace, then digested with SHA-256. Storing the
canonicalization identifier alongside each digest (spec Point 20, step 22) makes the
digest computation reproducible and auditable across versions.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

CANONICALIZATION_ID = "json-utf8-sorted-compact-v1"
DEFAULT_HASH_ALGORITHM = "SHA-256"
SUPPORTED_HASH_ALGORITHMS = frozenset({"SHA-256", "SHA-384", "SHA-512"})
FORBIDDEN_HASH_ALGORITHMS = frozenset({"MD5", "SHA-1"})


def canonical_json(value: Any) -> str:
    """Serialize any JSON-compatible structure deterministically."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=_fallback)


def canonical_bytes(value: Any) -> bytes:
    """Return the canonical UTF-8 encoding used for every digest and signature."""
    return canonical_json(value).encode("utf-8")


def _fallback(value: Any) -> Any:
    """Coerce uncommon types deterministically instead of raising mid-serialization."""
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, (set, frozenset)):
        return sorted(str(item) for item in value)
    if isinstance(value, (bytes, bytearray)):
        return value.hex()
    return str(value)


def digest_bytes(data: bytes, algorithm: str = DEFAULT_HASH_ALGORITHM) -> str:
    """Hex digest of raw bytes using an approved algorithm identifier."""
    if algorithm in FORBIDDEN_HASH_ALGORITHMS:
        raise ValueError(f"{algorithm} is not collision-resistant and is forbidden for integrity use")
    if algorithm not in SUPPORTED_HASH_ALGORITHMS:
        raise ValueError(f"unsupported hash algorithm: {algorithm}")
    return hashlib.new(algorithm.replace("-", "").lower(), data).hexdigest()


def digest_structure(value: Any, algorithm: str = DEFAULT_HASH_ALGORITHM) -> str:
    """Hex digest of the canonical serialization of a structure."""
    return digest_bytes(canonical_bytes(value), algorithm)


def content_digest(value: Any) -> dict[str, str]:
    """Content hash record carrying algorithm and canonicalization identifiers."""
    return {
        "algorithm": DEFAULT_HASH_ALGORITHM,
        "canonicalization": CANONICALIZATION_ID,
        "digest": digest_structure(value),
    }
