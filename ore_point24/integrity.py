"""Phase VI, Step 29 - enforce integrity and durable lineage.

Hash canonical artifacts, store content-addressed immutable references, link
parent and child artifacts, and use Merkle trees for efficient integrity
proofs when a collection is large enough to justify them.  Hashing proves
integrity *relative to a trusted digest*; it never proves the truth of the
artifact contents.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .core import canonical_json, sha256_digest

INTEGRITY_SCHEMA = "ore.point24.integrity.v1"


@dataclass(frozen=True, slots=True)
class MerkleNode:
    """One Merkle tree node: leaf hash or deterministic parent hash."""

    node_id: str
    digest: str
    children: tuple[str, ...] = ()
    leaf_key: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "digest": self.digest,
            "children": list(self.children),
            "leaf_key": self.leaf_key,
        }


class MerkleTree:
    """A binary Merkle tree with efficient inclusion proofs.

    A parent hash is computed from child hashes in deterministic order, so a
    changed leaf must change the root digest (under standard cryptographic
    hash assumptions).
    """

    def __init__(self, leaves: Mapping[str, str] | None = None) -> None:
        self._nodes: dict[str, MerkleNode] = {}
        self._leaf_digests: dict[str, str] = {}
        self._root: str | None = None
        for key, value in (leaves or {}).items():
            self.add_leaf(key, value)
        if self._leaf_digests:
            self._rebuild()

    def add_leaf(self, key: str, value: str) -> str:
        """Add (or replace) one leaf; the tree rebuilds deterministically."""
        digest = sha256_digest({"key": key, "value": value})
        self._leaf_digests[key] = digest
        self._rebuild()
        return digest

    def _rebuild(self) -> None:
        self._nodes = {}
        level: list[tuple[str, MerkleNode]] = []
        for key in sorted(self._leaf_digests):
            node_id = f"leaf:{key}"
            self._nodes[node_id] = MerkleNode(
                node_id=node_id, digest=self._leaf_digests[key], leaf_key=key
            )
            level.append((node_id, self._nodes[node_id]))
        level_index = 0
        while len(level) > 1:
            next_level: list[tuple[str, MerkleNode]] = []
            for index in range(0, len(level), 2):
                if index + 1 < len(level):
                    left_id, left = level[index]
                    right_id, right = level[index + 1]
                    digest = sha256_digest(
                        {"left": left.digest, "right": right.digest}
                    )
                    node_id = f"node:{level_index}:{index // 2}"
                    self._nodes[node_id] = MerkleNode(
                        node_id=node_id, digest=digest,
                        children=(left_id, right_id),
                    )
                else:
                    node_id = level[index][0]
                next_level.append((node_id, self._nodes[node_id]))
            level = next_level
            level_index += 1
        self._root = level[0][0] if level else None

    def root_digest(self) -> str | None:
        if self._root is None:
            return None
        return self._nodes[self._root].digest

    def inclusion_proof(self, key: str) -> list[dict[str, Any]] | None:
        """Return the sibling path from a leaf to the root.

        Each step records the sibling digest and whether the current node is
        the left child, so verification can replay the exact deterministic
        construction (left digest first, then right digest).
        """
        if key not in self._leaf_digests or self._root is None:
            return None
        proof: list[dict[str, Any]] = []
        node_id = f"leaf:{key}"
        while node_id != self._root:
            parent = next(
                (node for node in self._nodes.values() if node_id in node.children),
                None,
            )
            if parent is None:
                return None
            if len(parent.children) == 1:
                # The node was promoted unchanged (odd level width).
                proof.append({"parent_id": parent.node_id, "sibling": None,
                              "current_is_left": True})
            else:
                left_id, right_id = parent.children
                sibling_id = right_id if node_id == left_id else left_id
                proof.append({
                    "parent_id": parent.node_id,
                    "sibling": self._nodes[sibling_id].digest,
                    "current_is_left": node_id == left_id,
                })
            node_id = parent.node_id
        return proof

    def verify(self, key: str, value: str) -> bool:
        """Verify that (key, value) is an authenticated leaf of the tree."""
        digest = sha256_digest({"key": key, "value": value})
        proof = self.inclusion_proof(key)
        if proof is None:
            return False
        current = digest
        for step in proof:
            sibling = step["sibling"]
            if sibling is None:
                continue  # promoted unchanged
            if step["current_is_left"]:
                current = sha256_digest({"left": current, "right": sibling})
            else:
                current = sha256_digest({"left": sibling, "right": current})
        return current == self.root_digest()

    def summary(self) -> dict[str, Any]:
        return {
            "schema": INTEGRITY_SCHEMA,
            "leaf_count": len(self._leaf_digests),
            "root_digest": self.root_digest(),
            "node_count": len(self._nodes),
        }


class ContentAddressedStore:
    """An append-only, content-addressed artifact reference store.

    References are immutable: writing different content under an existing
    digest is impossible by construction, and lineage links between parent
    and child references are preserved.
    """

    def __init__(self) -> None:
        self._blobs: dict[str, str] = {}
        self._lineage: dict[str, tuple[str, ...]] = {}

    def put(self, content: str, *, parents: Iterable[str] = ()) -> str:
        digest = sha256_digest(content)
        if digest in self._blobs and self._blobs[digest] != content:
            raise ValueError("digest collision: refusing to overwrite content")
        self._blobs.setdefault(digest, content)
        self._lineage.setdefault(digest, tuple(parents))
        return digest

    def get(self, digest: str) -> str | None:
        return self._blobs.get(digest)

    def parents(self, digest: str) -> tuple[str, ...]:
        return self._lineage.get(digest, ())

    def merkle_tree(self) -> MerkleTree:
        """Build a Merkle tree over the full store for efficient proofs."""
        return MerkleTree({digest: content for digest, content in sorted(self._blobs.items())})

    def summary(self) -> dict[str, Any]:
        return {
            "schema": INTEGRITY_SCHEMA,
            "blob_count": len(self._blobs),
            "lineage_entries": len(self._lineage),
            "note": "hashing proves integrity relative to a trusted digest, not truth of contents",
        }


__all__ = [
    "INTEGRITY_SCHEMA",
    "ContentAddressedStore",
    "MerkleNode",
    "MerkleTree",
]
