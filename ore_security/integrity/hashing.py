"""Content hashing with algorithm and canonicalization identifiers (step 22).

A hash detects change only when compared with a trusted digest, so every content
hash record stores the algorithm and the canonicalization identifier next to the
digest. Collision-broken algorithms (MD5, SHA-1) are rejected outright.
"""
from __future__ import annotations

from typing import Any

from ..canonical import (
    CANONICALIZATION_ID,
    DEFAULT_HASH_ALGORITHM,
    digest_bytes,
    digest_structure,
)


class ContentHasher:
    """Compute and verify content hashes carrying algorithm metadata."""

    def __init__(self, algorithm: str = DEFAULT_HASH_ALGORITHM) -> None:
        # digest_bytes enforces the approved-algorithm policy
        digest_bytes(b"policy-check", algorithm)
        self.algorithm = algorithm

    def hash_bytes(self, data: bytes) -> dict[str, str]:
        """Hash raw content and return the full hash record."""
        return {
            "algorithm": self.algorithm,
            "canonicalization": "raw-bytes-v1",
            "digest": digest_bytes(data, self.algorithm),
        }

    def hash_structure(self, value: Any) -> dict[str, str]:
        """Hash the canonical serialization of a structure."""
        return {
            "algorithm": self.algorithm,
            "canonicalization": CANONICALIZATION_ID,
            "digest": digest_structure(value, self.algorithm),
        }

    def verify_bytes(self, data: bytes, expected_digest: str) -> bool:
        """Compare a recomputed digest with a trusted digest."""
        return digest_bytes(data, self.algorithm) == expected_digest

    def verify_structure(self, value: Any, expected_digest: str) -> bool:
        return digest_structure(value, self.algorithm) == expected_digest
