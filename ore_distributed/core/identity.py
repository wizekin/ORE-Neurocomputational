"""Durable node identities, credentials, and message authentication (Step 6).

Node identity is provisioned once and bound to authorization policy and audit
records; it is never inferred solely from an IP address, hostname, or transient
container identifier. Transport authentication is deliberately separate from
semantic trust in a node's findings.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import json
import secrets
from pathlib import Path
from typing import Any, Mapping


def _non_empty(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def canonical_bytes(value: Any) -> bytes:
    """Serialize a JSON-compatible value deterministically for signing."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    ).encode("utf-8")


def digest_hex(value: Any) -> str:
    """Return the SHA-256 hex digest of the canonical serialization of a value."""
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


class ClusterCredentialAuthority:
    """Provision per-node credentials derived from a shared cluster secret.

    The authority issues HMAC keys bound to a durable node identity; admission to
    the voting configuration additionally requires the authorized membership
    change procedure in ``ore_distributed.membership.admission``.
    """

    def __init__(self, cluster_id: str, master_secret: str | None = None) -> None:
        """Create an authority for one cluster with an optional fixed secret."""
        self.cluster_id = _non_empty("cluster_id", cluster_id)
        self._master_secret = (
            master_secret.encode("utf-8")
            if master_secret is not None
            else secrets.token_bytes(32)
        )
        self._issued: dict[str, bytes] = {}

    def issue(self, node_id: str) -> "NodeCredential":
        """Issue a credential for a known durable node identity."""
        node_id = _non_empty("node_id", node_id)
        key = hmac.new(self._master_secret, node_id.encode("utf-8"), hashlib.sha256).digest()
        self._issued[node_id] = key
        return NodeCredential(node_id=node_id, cluster_id=self.cluster_id, key=key)

    def verify_credential(self, credential: "NodeCredential") -> bool:
        """Return whether a credential was issued by this authority."""
        expected = self._issued.get(credential.node_id)
        return expected is not None and hmac.compare_digest(expected, credential.key)

    def key_for(self, node_id: str) -> bytes | None:
        """Return the issued key for a node, if provisioned."""
        return self._issued.get(node_id)


@dataclass(slots=True)
class NodeCredential:
    """Hold the durable identity and signing key of one node."""

    node_id: str
    cluster_id: str
    key: bytes

    def __post_init__(self) -> None:
        """Validate identity text and key material length."""
        self.node_id = _non_empty("node_id", self.node_id)
        self.cluster_id = _non_empty("cluster_id", self.cluster_id)
        if not isinstance(self.key, bytes) or len(self.key) < 16:
            raise ValueError("key must be at least 16 bytes of key material")

    def sign(self, value: Any) -> str:
        """Return an HMAC-SHA256 signature over the canonical serialization."""
        return hmac.new(self.key, canonical_bytes(value), hashlib.sha256).hexdigest()

    def verify(self, value: Any, signature: str) -> bool:
        """Verify a signature over the canonical serialization of a value."""
        if not isinstance(signature, str):
            return False
        expected = self.sign(value)
        return hmac.compare_digest(expected, signature)


@dataclass(frozen=True, slots=True)
class NodeIdentity:
    """A durable node identity bound to its cluster and authorization record."""

    node_id: str
    cluster_id: str
    created_at: str
    roles: tuple[str, ...] = ("worker",)
    credential: NodeCredential | None = None

    def __post_init__(self) -> None:
        """Validate identity fields and normalize roles."""
        object.__setattr__(self, "node_id", _non_empty("node_id", self.node_id))
        object.__setattr__(self, "cluster_id", _non_empty("cluster_id", self.cluster_id))
        object.__setattr__(self, "created_at", _non_empty("created_at", self.created_at))
        roles = tuple(_non_empty("role", item) for item in self.roles) or ("worker",)
        object.__setattr__(self, "roles", roles)

    def to_public_dict(self) -> dict[str, Any]:
        """Return identity metadata without key material."""
        return {
            "node_id": self.node_id,
            "cluster_id": self.cluster_id,
            "created_at": self.created_at,
            "roles": list(self.roles),
        }


def create_node_identity(
    node_id: str,
    cluster_id: str,
    authority: ClusterCredentialAuthority,
    created_at: str = "1970-01-01T00:00:00Z",
    roles: tuple[str, ...] = ("worker",),
) -> NodeIdentity:
    """Provision a durable identity with credentials from the cluster authority."""
    credential = authority.issue(node_id)
    return NodeIdentity(
        node_id=node_id,
        cluster_id=cluster_id,
        created_at=created_at,
        roles=roles,
        credential=credential,
    )


def persist_identity(identity: NodeIdentity, path: str | Path) -> None:
    """Persist identity metadata and key material for restart durability (Step 6)."""
    if identity.credential is None:
        raise ValueError("identity must carry a credential to persist")
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "node_id": identity.node_id,
        "cluster_id": identity.cluster_id,
        "created_at": identity.created_at,
        "roles": list(identity.roles),
        "key_hex": identity.credential.key.hex(),
    }
    file_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_identity(path: str | Path, cluster_id: str | None = None) -> NodeIdentity:
    """Reload a persisted identity; node IDs stay stable across restarts."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"identity file not found: {file_path}")
    payload: Mapping[str, Any] = json.loads(file_path.read_text(encoding="utf-8"))
    stored_cluster = str(payload["cluster_id"])
    if cluster_id is not None and stored_cluster != cluster_id:
        raise ValueError(
            f"identity belongs to cluster {stored_cluster!r}, not {cluster_id!r}"
        )
    return NodeIdentity(
        node_id=str(payload["node_id"]),
        cluster_id=stored_cluster,
        created_at=str(payload["created_at"]),
        roles=tuple(payload.get("roles", ("worker",))),
        credential=NodeCredential(
            node_id=str(payload["node_id"]),
            cluster_id=stored_cluster,
            key=bytes.fromhex(str(payload["key_hex"])),
        ),
    )


__all__ = [
    "ClusterCredentialAuthority",
    "NodeCredential",
    "NodeIdentity",
    "canonical_bytes",
    "digest_hex",
    "create_node_identity",
    "persist_identity",
    "load_identity",
]
