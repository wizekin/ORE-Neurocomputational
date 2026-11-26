"""Signed integrity checkpoints anchored in a separate trust domain (steps 23-24).

A checkpoint covers an event range: the hash-chain head and the Merkle root of the
batch's records. Checkpoints are signed with a dedicated key and anchored externally
(by recording the checkpoint digest in a second, independently controlled ledger).
Verification of restored backups replays the chain against the checkpoint heads.
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Callable

from ..canonical import canonical_bytes, digest_structure
from ..contracts import IntegrityCheckpoint
from ..exceptions import CheckpointError
from ..keys.kms import LocalKMS
from .chain import HashChain
from .merkle import TREE_VERSION, MerkleTree

Clock = Callable[[], float]


class ExternalAnchorVault:
    """Second trust domain that anchors checkpoint digests (log-gap protection).

    The anchor is intentionally a separate ledger with its own signing key so that
    compromise of the primary audit service cannot rewrite anchored history.
    """

    def __init__(self, *, clock: Clock = time.time) -> None:
        self.clock = clock
        self._anchors: dict[str, dict[str, Any]] = {}

    def anchor(self, checkpoint: IntegrityCheckpoint) -> str:
        """Record a checkpoint digest and return the anchor reference."""
        anchor_id = "anchor-" + uuid.uuid4().hex[:12]
        self._anchors[anchor_id] = {
            "checkpoint_id": checkpoint.checkpoint_id,
            "digest": digest_structure(checkpoint.signed_payload()),
            "range": [checkpoint.range_start, checkpoint.range_end],
            "anchored_at": self.clock(),
        }
        return anchor_id

    def verify_anchor(self, anchor_id: str, checkpoint: IntegrityCheckpoint) -> bool:
        """Confirm an anchored checkpoint still matches its recorded digest."""
        anchor = self._anchors.get(anchor_id)
        if anchor is None:
            raise CheckpointError(f"unknown anchor reference: {anchor_id}")
        return (
            anchor["checkpoint_id"] == checkpoint.checkpoint_id
            and anchor["digest"] == digest_structure(checkpoint.signed_payload())
        )

    def anchor_count(self) -> int:
        return len(self._anchors)


class CheckpointService:
    """Create and verify signed checkpoints over audit history."""

    def __init__(self, kms: LocalKMS, signing_key_id: str, *,
                 clock: Clock = time.time) -> None:
        self.kms = kms
        self.signing_key_id = signing_key_id
        self.clock = clock

    def create_checkpoint(self, chain: HashChain, batch_records: list[dict[str, Any]],
                          *, anchor_vault: ExternalAnchorVault | None = None
                          ) -> IntegrityCheckpoint:
        """Sign a checkpoint over the chain head and the batch Merkle root."""
        if chain.length == 0:
            raise CheckpointError("cannot checkpoint an empty chain")
        if not batch_records:
            raise CheckpointError("cannot checkpoint an empty batch")
        start = batch_records[0].get("sequence", 1)
        end = batch_records[-1].get("sequence", chain.length)
        merkle_root = MerkleTree(batch_records).root
        checkpoint = IntegrityCheckpoint(
            checkpoint_id="ckpt-" + uuid.uuid4().hex[:12],
            range_start=int(start),
            range_end=int(end),
            chain_head=chain.head,
            merkle_root=merkle_root,
            tree_version=TREE_VERSION,
            signature="",
            signer_key_id=self.signing_key_id,
            timestamp=self.clock(),
        )
        signature, _ = self.kms.sign(self.signing_key_id, canonical_bytes(checkpoint.signed_payload()))
        signed = IntegrityCheckpoint(
            checkpoint_id=checkpoint.checkpoint_id,
            range_start=checkpoint.range_start,
            range_end=checkpoint.range_end,
            chain_head=checkpoint.chain_head,
            merkle_root=checkpoint.merkle_root,
            tree_version=checkpoint.tree_version,
            signature=signature.hex(),
            signer_key_id=self.signing_key_id,
            timestamp=checkpoint.timestamp,
        )
        if anchor_vault is not None:
            anchor_vault.anchor(signed)
        return signed

    def verify_checkpoint(self, checkpoint: IntegrityCheckpoint,
                          *, anchor_vault: ExternalAnchorVault | None = None,
                          anchor_id: str | None = None) -> dict[str, Any]:
        """Verify a checkpoint signature and optional external anchor."""
        signature_ok = self.kms.verify(
            checkpoint.signer_key_id,
            canonical_bytes(checkpoint.signed_payload()),
            bytes.fromhex(checkpoint.signature),
            signature_time=checkpoint.timestamp,
        )
        if not signature_ok:
            raise CheckpointError(f"checkpoint {checkpoint.checkpoint_id} signature invalid")
        anchor_ok: bool | None = None
        if anchor_vault is not None:
            if anchor_id is None:
                raise CheckpointError("anchor verification requires an anchor_id")
            anchor_ok = anchor_vault.verify_anchor(anchor_id, checkpoint)
            if not anchor_ok:
                raise CheckpointError(
                    f"checkpoint {checkpoint.checkpoint_id} does not match its external anchor"
                )
        return {
            "checkpoint_id": checkpoint.checkpoint_id,
            "signature_valid": True,
            "anchor_valid": anchor_ok,
            "range": [checkpoint.range_start, checkpoint.range_end],
            "chain_head": checkpoint.chain_head,
            "merkle_root": checkpoint.merkle_root,
        }
