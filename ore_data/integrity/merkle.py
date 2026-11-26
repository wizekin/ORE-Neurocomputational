"""Merkle tree commitments and proofs with domain-separated tags (Phase J).

Steps 54-55: canonical leaf records are hashed with a leaf tag and pairs of
child hashes with a node tag, so leaves and internal nodes can never be
confused (source 4.7); the odd-node rule promotes the last unpaired node
unchanged, identically in construction and verification. Merkle proofs
recompute the root from sibling hashes and path information; a proof
succeeds only when the recomputed root equals the trusted committed root,
and success establishes consistency with that committed root under
cryptographic assumptions, never the factual correctness of the record
(source 1.2).
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence, Tuple


#: Domain-separated tags (source 4.7): leaf and internal-node hashes never share a namespace.
LEAF_TAG = b"ORE.DATA.MERKLE.LEAF/0.1.0\x00"
NODE_TAG = b"ORE.DATA.MERKLE.NODE/0.1.0\x00"


class MerkleError(ValueError):
    """Raised on malformed Merkle commitments or proofs."""


def leaf_hash(record: bytes) -> bytes:
    """Return H(leaf-tag || record) for one canonical record (Step 54)."""
    if not isinstance(record, (bytes, bytearray)):
        raise TypeError("records hash as bytes")
    return hashlib.sha256(LEAF_TAG + bytes(record)).digest()


def node_hash(left: bytes, right: bytes) -> bytes:
    """Return H(node-tag || left || right) for one internal node (Step 54)."""
    if len(left) != 32 or len(right) != 32:
        raise MerkleError("internal nodes hash 32-byte child digests")
    return hashlib.sha256(NODE_TAG + left + right).digest()


@dataclass(frozen=True, slots=True)
class MerkleCommitment:
    """One committed Merkle root with its snapshot metadata (Step 54)."""

    snapshot_id: str
    root: str
    leaf_count: int
    schema_version: str
    created_at: str
    anchoring: Tuple[Tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        """Validate the commitment record."""
        if not self.snapshot_id.strip():
            raise ValueError("snapshot_id must be non-empty")
        if len(self.root) != 64:
            raise ValueError("root must be a 64-character hex digest")
        if self.leaf_count <= 0:
            raise ValueError("commitments cover at least one record")
        object.__setattr__(self, "anchoring", tuple(self.anchoring))

    def to_dict(self) -> dict[str, object]:
        """Return the commitment as a plain dictionary."""
        return {
            "snapshot_id": self.snapshot_id,
            "root": self.root,
            "leaf_count": self.leaf_count,
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "anchoring": [list(pair) for pair in self.anchoring],
        }


@dataclass(frozen=True, slots=True)
class MerkleProof:
    """Sibling hashes and path information for one leaf (Step 55)."""

    leaf_index: int
    leaf_count: int
    siblings: Tuple[Tuple[bytes, bool], ...]  # (sibling digest, sibling_is_left)

    def __post_init__(self) -> None:
        """Validate the proof structure."""
        if self.leaf_index < 0:
            raise ValueError("leaf_index must be non-negative")
        if self.leaf_count <= 0:
            raise ValueError("leaf_count must be positive")
        if self.leaf_index >= self.leaf_count:
            raise ValueError("leaf_index out of range")
        object.__setattr__(self, "siblings", tuple(self.siblings))

    def to_dict(self) -> dict[str, object]:
        """Return the proof as a plain dictionary."""
        return {
            "leaf_index": self.leaf_index,
            "leaf_count": self.leaf_count,
            "siblings": [
                [digest.hex(), is_left] for digest, is_left in self.siblings
            ],
        }


@dataclass(frozen=True, slots=True)
class IntegrityResult:
    """Result of one Merkle proof verification (Step 55)."""

    consistent: bool
    recomputed_root: str
    trusted_root: str
    note: str

    def to_dict(self) -> dict[str, object]:
        """Return the result as a plain dictionary."""
        return {
            "consistent": self.consistent,
            "recomputed_root": self.recomputed_root,
            "trusted_root": self.trusted_root,
            "note": self.note,
        }


def _now() -> str:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class MerkleTree:
    """One built Merkle tree over canonical records (Steps 54-55)."""

    def __init__(self, records: Sequence[bytes]) -> None:
        """Build the tree from canonical record bytes (Step 54)."""
        if not records:
            raise MerkleError("snapshots cover at least one record")
        self.leaf_count = len(records)
        self._leaf_hashes = [leaf_hash(record) for record in records]
        self._levels: list[list[bytes]] = [list(self._leaf_hashes)]
        while len(self._levels[-1]) > 1:
            self._levels.append(self._build_next_level(self._levels[-1]))

    @staticmethod
    def _build_next_level(level: list[bytes]) -> list[bytes]:
        """Pair children into nodes; the odd node is promoted unchanged (Step 54)."""
        next_level: list[bytes] = []
        for index in range(0, len(level) - 1, 2):
            next_level.append(node_hash(level[index], level[index + 1]))
        if len(level) % 2 == 1:
            next_level.append(level[-1])  # odd-node rule: promote unchanged
        return next_level

    def root(self) -> bytes:
        """Return the committed root digest."""
        return self._levels[-1][0]

    def commitment(self, snapshot_id: str, schema_version: str = "0.1.0") -> MerkleCommitment:
        """Return the snapshot's commitment record (Step 54)."""
        return MerkleCommitment(
            snapshot_id=snapshot_id,
            root=self.root().hex(),
            leaf_count=self.leaf_count,
            schema_version=schema_version,
            created_at=_now(),
        )

    def proof(self, leaf_index: int) -> MerkleProof:
        """Generate the audit proof for one leaf (Step 55)."""
        if not 0 <= leaf_index < self.leaf_count:
            raise ValueError(f"leaf index {leaf_index} out of range")
        siblings: list[tuple[bytes, bool]] = []
        index = leaf_index
        for level in self._levels[:-1]:
            level_size = len(level)
            if index % 2 == 0:
                sibling_index = index + 1
                sibling_is_left = False
            else:
                sibling_index = index - 1
                sibling_is_left = True
            if sibling_index < level_size:
                siblings.append((level[sibling_index], sibling_is_left))
            else:
                siblings.append((b"", False))  # promoted odd node: no sibling
            index //= 2
        return MerkleProof(
            leaf_index=leaf_index,
            leaf_count=self.leaf_count,
            siblings=tuple(siblings),
        )


def verify_merkle_proof(
    record: bytes,
    proof: MerkleProof,
    trusted_root: str,
) -> IntegrityResult:
    """Verify one record against a trusted committed root (Step 55).

    The proof's sibling hashes and path information recompute the root;
    acceptance requires equality with the trusted root. Consistency with
    the committed root is not factual correctness of the record.
    """
    current = leaf_hash(record)
    index = proof.leaf_index
    for sibling, sibling_is_left in proof.siblings:
        if not sibling:
            # Odd-node promotion: the current hash moves up unchanged.
            pass
        elif sibling_is_left:
            current = node_hash(sibling, current)
        else:
            current = node_hash(current, sibling)
        index //= 2
    recomputed = current.hex()
    consistent = recomputed == trusted_root
    return IntegrityResult(
        consistent=consistent,
        recomputed_root=recomputed,
        trusted_root=trusted_root,
        note=(
            "consistent with the committed root under cryptographic assumptions; "
            "this does not establish that the original record was factually "
            "correct (source 1.2)"
        )
        if consistent
        else "proof failed: the recomputed root differs from the trusted root",
    )


def build_merkle_snapshot(
    records: Sequence[bytes],
    snapshot_id: str,
    schema_version: str = "0.1.0",
) -> tuple[MerkleTree, MerkleCommitment]:
    """Section 5.1 interface: build one snapshot and its commitment (Step 54)."""
    tree = MerkleTree(records)
    return tree, tree.commitment(snapshot_id, schema_version)


class IntegrityVerifier:
    """Snapshot verifier with proof latency measurement (Step 55 + section 8.5)."""

    def __init__(self) -> None:
        """Create one verifier with no trusted roots yet."""
        self._roots: dict[str, str] = {}
        self.proof_verifications = 0
        self.proof_verification_ms: list[float] = []
        self.failed_verifications = 0

    def publish_root(self, commitment: MerkleCommitment) -> None:
        """Publish one trusted root (the trust anchor stays explicit, source 9)."""
        self._roots[commitment.snapshot_id] = commitment.root

    def trusted_root(self, snapshot_id: str) -> str:
        """Return the trusted root of one snapshot."""
        if snapshot_id not in self._roots:
            raise KeyError(f"no trusted root published for {snapshot_id!r}")
        return self._roots[snapshot_id]

    def verify(
        self,
        snapshot_id: str,
        record: bytes,
        proof: MerkleProof,
    ) -> IntegrityResult:
        """Verify one record against the snapshot's trusted root (Step 55)."""
        root = self.trusted_root(snapshot_id)
        started = time.perf_counter()
        result = verify_merkle_proof(record, proof, root)
        self.proof_verification_ms.append((time.perf_counter() - started) * 1000.0)
        self.proof_verifications += 1
        if not result.consistent:
            self.failed_verifications += 1
        return result

    def stats(self) -> dict[str, object]:
        """Return verifier statistics (metric: proof verification latency)."""
        latencies = sorted(self.proof_verification_ms)
        p50 = latencies[len(latencies) // 2] if latencies else 0.0
        p95 = latencies[int(0.95 * len(latencies))] if latencies else 0.0
        return {
            "proof_verifications": self.proof_verifications,
            "failed_verifications": self.failed_verifications,
            "proof_latency_p50_ms": round(p50, 4),
            "proof_latency_p95_ms": round(p95, 4),
        }


__all__ = [
    "LEAF_TAG",
    "NODE_TAG",
    "MerkleError",
    "leaf_hash",
    "node_hash",
    "MerkleCommitment",
    "MerkleProof",
    "IntegrityResult",
    "MerkleTree",
    "verify_merkle_proof",
    "build_merkle_snapshot",
    "IntegrityVerifier",
]
