"""Tamper-evident hash chain for audit events (step 23).

Each appended event is folded into the running chain:

    h_i = H(h_{i-1} || canonical(e_i))

Verification recomputes the chain and detects modification and reordering. Deleting
the tail of a chain cannot be detected without a trusted checkpoint, which is why
``verify`` accepts checkpoint heads: a chain without an anchored checkpoint cannot
detect deletion of the entire chain.
"""
from __future__ import annotations

from typing import Any

from ..canonical import canonical_bytes, digest_bytes
from ..exceptions import CheckpointError, IntegrityError

GENESIS_DIGEST = "0" * 64


class HashChain:
    """Append-only event chain with sequence numbers and head digests."""

    def __init__(self) -> None:
        self._entries: list[dict[str, Any]] = []
        self._head = GENESIS_DIGEST

    def append(self, event: dict[str, Any]) -> dict[str, Any]:
        """Fold one event into the chain and return its chained record."""
        if not isinstance(event, dict):
            raise IntegrityError("chain events must be mappings")
        sequence = len(self._entries) + 1
        record = {
            "sequence": sequence,
            "event": dict(event),
            "previous_digest": self._head,
            "digest": digest_bytes(
                bytes.fromhex(self._head) + canonical_bytes(event)
            ),
        }
        self._entries.append(record)
        self._head = record["digest"]
        return record

    @property
    def head(self) -> str:
        """Current chain head digest."""
        return self._head

    @property
    def length(self) -> int:
        return len(self._entries)

    def entries(self) -> tuple[dict[str, Any], ...]:
        """Return the chained records in order."""
        return tuple(self._entries)

    def verify(self, *, expected_head: str | None = None,
               from_sequence: int = 1) -> dict[str, Any]:
        """Recompute the chain; report tampering, reordering or truncation."""
        previous = GENESIS_DIGEST if from_sequence <= 1 else None
        start_index = max(0, from_sequence - 1)
        if from_sequence > 1:
            if start_index > len(self._entries):
                raise IntegrityError("verification start sequence is beyond the chain")
            previous = self._entries[start_index - 1]["digest"]
        expected_sequence = start_index + 1
        for record in self._entries[start_index:]:
            if record["sequence"] != expected_sequence:
                raise IntegrityError(
                    f"chain sequence broken at position {expected_sequence}: "
                    f"found {record['sequence']}; events were reordered or deleted"
                )
            recomputed = digest_bytes(
                bytes.fromhex(record["previous_digest"]) + canonical_bytes(record["event"])
            )
            if recomputed != record["digest"]:
                raise IntegrityError(
                    f"chain digest mismatch at sequence {record['sequence']}: event was modified"
                )
            if record["previous_digest"] != previous:
                raise IntegrityError(
                    f"chain link broken at sequence {record['sequence']}: history was spliced"
                )
            previous = record["digest"]
            expected_sequence += 1
        if expected_head is not None and self._head != expected_head:
            raise CheckpointError(
                "chain head does not match the trusted checkpoint: committed history was truncated"
            )
        return {
            "valid": True,
            "verified_entries": len(self._entries) - start_index,
            "head": self._head,
        }

    @classmethod
    def from_entries(cls, records: list[dict[str, Any]]) -> "HashChain":
        """Rebuild a chain from serialized records (backup restoration)."""
        chain = cls()
        chain._entries = [dict(record) for record in records]
        chain._head = records[-1]["digest"] if records else GENESIS_DIGEST
        return chain
