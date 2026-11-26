"""Merkle trees over batch records with inclusion proofs (step 24).

Leaves are the canonical digests of records in insertion order (leaf ordering is
part of the tree contract). Odd final nodes are duplicated so sibling pairing is
always defined; the tree version identifier records this convention. The root can
be signed or externally anchored; inclusion proofs let a verifier confirm a record
belongs to a committed batch without downloading every record.
"""
from __future__ import annotations

from typing import Any

from ..canonical import canonical_bytes, digest_bytes
from ..exceptions import MerkleError

TREE_VERSION = "merkle-sha256-dup-odd-v1"


def _leaf_digest(record: Any) -> bytes:
    return bytes.fromhex(digest_bytes(b"leaf|" + canonical_bytes(record)))


def _node_digest(left: bytes, right: bytes) -> bytes:
    return bytes.fromhex(digest_bytes(b"node|" + left + right))


class MerkleTree:
    """Binary Merkle tree over canonically digested records."""

    def __init__(self, records: list[Any]) -> None:
        if not records:
            raise MerkleError("a Merkle tree requires at least one record")
        self.records = list(records)
        self._levels: list[list[bytes]] = [self._build_leaves()]
        while len(self._levels[-1]) > 1:
            self._levels.append(self._build_level(self._levels[-1]))

    def _build_leaves(self) -> list[bytes]:
        return [_leaf_digest(record) for record in self.records]

    @staticmethod
    def _build_level(level: list[bytes]) -> list[bytes]:
        if len(level) % 2 == 1:
            level = level + [level[-1]]  # duplicate the odd final node (documented convention)
        return [_node_digest(level[i], level[i + 1]) for i in range(0, len(level), 2)]

    @property
    def root(self) -> str:
        """Merkle root as a hex digest."""
        return self._levels[-1][0].hex()

    @property
    def leaf_count(self) -> int:
        return len(self.records)

    def inclusion_proof(self, index: int) -> list[dict[str, Any]]:
        """Sibling path from leaf ``index`` to the root."""
        if not 0 <= index < len(self.records):
            raise MerkleError(f"leaf index {index} out of range for {len(self.records)} leaves")
        proof: list[dict[str, Any]] = []
        current_index = index
        for level in self._levels[:-1]:
            sibling_index = current_index ^ 1
            sibling = level[sibling_index] if sibling_index < len(level) else level[current_index]
            proof.append({"digest": sibling.hex(), "position": "right" if sibling_index > current_index else "left"})
            current_index //= 2
        return proof

    @staticmethod
    def verify_inclusion(record: Any, index: int, proof: list[dict[str, Any]],
                         root: str) -> bool:
        """Verify that one record belongs to the batch committed by ``root``.

        The claimed index is validated against the sibling positions of the proof
        path: at each level the parity of the current position determines which
        side the sibling must occupy, so a proof for one leaf cannot be presented
        for another index.
        """
        if not isinstance(proof, list) or not proof:
            return False
        if not isinstance(index, int) or index < 0:
            return False
        current = _leaf_digest(record)
        position = index
        for step in proof:
            sibling_hex = step.get("digest", "")
            side = step.get("position")
            try:
                sibling = bytes.fromhex(sibling_hex)
            except ValueError:
                return False
            expected_side = "right" if position % 2 == 0 else "left"
            if side != expected_side:
                return False
            if side == "left":
                current = _node_digest(sibling, current)
            else:
                current = _node_digest(current, sibling)
            position //= 2
        return current.hex() == root
