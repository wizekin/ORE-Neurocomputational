"""Digest-based gossip dissemination with bounded payloads (Step 32).

Peers periodically exchange digests or version summaries, then repair only
the missing entries, instead of repeatedly transmitting entire state. Rate
limits and payload caps bound gossip amplification.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any, Iterable, Mapping, Sequence


def digest_of_entries(entries: Iterable[str]) -> str:
    """Return a deterministic digest of an entry identifier set."""
    ordered = sorted(set(entries))
    return hashlib.sha256("\n".join(ordered).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class GossipDigest:
    """One peer's version summary for digest comparison."""

    sender: str
    version: int
    entry_ids: tuple[str, ...]
    digest: str

    def to_dict(self) -> dict[str, Any]:
        """Return the digest message as a plain dictionary."""
        return {
            "sender": self.sender,
            "version": self.version,
            "entry_count": len(self.entry_ids),
            "digest": self.digest,
        }


class GossipDissemination:
    """Run bounded gossip rounds with digest exchange and selective repair."""

    def __init__(
        self,
        node_id: str,
        *,
        max_batch: int = 16,
        max_round_messages: int = 64,
        max_payload_bytes: int = 64 * 1024,
    ) -> None:
        """Create one node's gossip state with explicit caps."""
        if not node_id.strip():
            raise ValueError("node_id must be non-empty")
        if max_batch < 1 or max_round_messages < 1 or max_payload_bytes < 1:
            raise ValueError("caps must be positive")
        self.node_id = node_id
        self.max_batch = int(max_batch)
        self.max_round_messages = int(max_round_messages)
        self.max_payload_bytes = int(max_payload_bytes)
        self.version = 0
        self._entries: dict[str, Any] = {}
        self.messages_sent = 0
        self.messages_rejected = 0
        self.repairs_sent = 0
        self.repairs_received = 0

    def digest(self) -> GossipDigest:
        """Build the current digest of locally held entries."""
        return GossipDigest(
            sender=self.node_id,
            version=self.version,
            entry_ids=tuple(sorted(self._entries)),
            digest=digest_of_entries(self._entries.keys()),
        )

    def add_local(self, entry_id: str, payload: Any) -> bool:
        """Record one local entry; repeated identifiers are ignored."""
        if not entry_id.strip():
            raise ValueError("entry_id must be non-empty")
        if entry_id in self._entries:
            return False
        self._entries[entry_id] = payload
        self.version += 1
        return True

    def apply_remote(self, entries: Mapping[str, Any]) -> int:
        """Apply a repair batch; returns the count of new entries absorbed."""
        if len(str(entries)) > self.max_payload_bytes:
            self.messages_rejected += 1
            raise ValueError("repair batch exceeds the bounded payload size")
        if len(entries) > self.max_batch * 4:
            self.messages_rejected += 1
            raise ValueError("repair batch exceeds the bounded entry count")
        applied = 0
        for entry_id, payload in entries.items():
            if entry_id not in self._entries:
                self._entries[entry_id] = payload
                self.version += 1
                applied += 1
        self.repairs_received += applied
        return applied

    def _check_budget(self) -> bool:
        """Apply the per-round message admission control."""
        return self.messages_sent < self.max_round_messages

    def exchange(self, peer_digest: GossipDigest, peer_entries: Mapping[str, Any]) -> dict[str, Any]:
        """Run one digest exchange and selective repair with a peer (Step 32).

        Returns a report describing what each side learned; the caller routes
        the actual payloads through the transport.
        """
        if not self._check_budget():
            self.messages_rejected += 1
            raise RuntimeError("gossip message budget for this round is exhausted")
        self.messages_sent += 1
        missing_locally = {
            entry_id: None for entry_id in peer_digest.entry_ids
            if entry_id not in self._entries
        }
        peer_needs = {
            entry_id: self._entries[entry_id]
            for entry_id in sorted(self._entries)
            if entry_id not in set(peer_digest.entry_ids)
        }
        # Cap the repair batch before sending.
        peer_needs = dict(list(peer_needs.items())[: self.max_batch])
        missing_locally = dict(list(missing_locally.items())[: self.max_batch])
        applied = self.apply_remote(
            {entry_id: peer_entries[entry_id] for entry_id in missing_locally if entry_id in peer_entries}
        )
        self.repairs_sent += len(peer_needs)
        return {
            "peer": peer_digest.sender,
            "received_new": applied,
            "sent_repair": len(peer_needs),
            "local_version": self.version,
            "converged": self.digest().digest == peer_digest.digest
            or (not peer_needs and not missing_locally),
        }

    def entries(self) -> dict[str, Any]:
        """Return the locally held entries."""
        return dict(self._entries)

    def entry_ids(self) -> tuple[str, ...]:
        """Return locally held entry identifiers sorted."""
        return tuple(sorted(self._entries))

    def convergence_check(self, other: "GossipDissemination") -> bool:
        """Return whether two gossip states hold identical entry sets."""
        return set(self._entries) == set(other._entries)


def run_gossip_rounds(
    members: Sequence[GossipDissemination],
    pairs: Sequence[tuple[int, int]],
    entries_provider: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[bool, int]:
    """Drive explicit pairwise gossip rounds; returns (converged, rounds_run).

    Each pair exchanges digests and repairs until the whole membership holds
    identical entry sets, bounded by the per-node message budgets.
    """
    if len(members) < 1:
        raise ValueError("membership must not be empty")
    for round_index, (left_index, right_index) in enumerate(pairs, start=1):
        left, right = members[left_index], members[right_index]
        right_entries = entries_provider.get(right.node_id, {}) if entries_provider else right.entries()
        left_entries = entries_provider.get(left.node_id, {}) if entries_provider else left.entries()
        left.exchange(right.digest(), right_entries)
        right.exchange(left.digest(), left_entries)
        if all(member.convergence_check(members[0]) for member in members):
            return True, round_index
    converged = all(member.convergence_check(members[0]) for member in members)
    return converged, len(pairs)


__all__ = [
    "GossipDigest",
    "GossipDissemination",
    "digest_of_entries",
    "run_gossip_rounds",
]
