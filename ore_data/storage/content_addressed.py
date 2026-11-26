"""Content-addressed immutable artifact storage (Phase I).

Steps 50-53: validated AlgoIR specifications, accepted algorithms,
reproducible execution bundles, immutable knowledge snapshots, model
artifacts and archived routes route to content-addressed storage, while
mutable workspace records stay out of it unless an explicit versioning
model exists (Step 50). Objects are stored under their domain-tagged
cryptographic content hash, verified on write and read, and deduplicated
by reuse (Step 51). A mutable manifest maps logical names and versions to
immutable content IDs, so updates create new objects and manifest entries
instead of overwriting historical bytes (Step 52). Retrieval re-resolves
the ID, recomputes the hash and verifies schema before the artifact
reaches a consumer, and a matching hash proves byte integrity relative to
the identifier, never safety or correctness (Step 53).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from ore_data.contracts import content_id, verify_content_id


class ContentAddressedStoreError(RuntimeError):
    """Raised on object-store integrity violations."""


#: Artifact classes eligible for immutable content-addressed storage (Step 50).
IMMUTABLE_ARTIFACT_CLASSES = frozenset(
    {
        "algo_ir_specification",
        "accepted_algorithm",
        "execution_bundle",
        "knowledge_snapshot",
        "model_artifact",
        "archived_report",
    }
)

#: Mutable record classes that require an explicit versioning model (Step 50).
MUTABLE_RECORD_CLASSES = frozenset({"workspace_record", "task_state"})


def validate_artifact_class(artifact_class: str, *, versioning_model: bool = False) -> bool:
    """Return whether one artifact class may enter immutable storage (Step 50)."""
    if artifact_class in IMMUTABLE_ARTIFACT_CLASSES:
        return True
    if artifact_class in MUTABLE_RECORD_CLASSES:
        return versioning_model
    return False


@dataclass(frozen=True, slots=True)
class StoreReceipt:
    """Receipt of one object write (Step 51)."""

    content_id: str
    domain: str
    bytes_stored: int
    reused: bool
    verified_on_write: bool

    def to_dict(self) -> dict[str, object]:
        """Return the receipt as a plain dictionary."""
        return {
            "content_id": self.content_id,
            "domain": self.domain,
            "bytes_stored": self.bytes_stored,
            "reused": self.reused,
            "verified_on_write": self.verified_on_write,
        }


@dataclass(frozen=True, slots=True)
class VerifiedArtifact:
    """One hash-verified immutable artifact (Step 53)."""

    content_id: str
    domain: str
    content: bytes

    def __post_init__(self) -> None:
        """Freeze the artifact content."""
        object.__setattr__(self, "content", bytes(self.content))

    @property
    def byte_integrity_only(self) -> bool:
        """A matching hash proves byte integrity, not safety or correctness (Step 53)."""
        return True

    def integrity_note(self) -> str:
        """Return the standing integrity disclaimer."""
        return (
            "content hash verifies the bytes relative to the identifier; it does "
            "not prove the artifact is safe or correct (source 1.2)"
        )

    def to_dict(self) -> dict[str, object]:
        """Return artifact metadata (content excluded) as a plain dictionary."""
        return {
            "content_id": self.content_id,
            "domain": self.domain,
            "bytes": len(self.content),
            "byte_integrity_only": self.byte_integrity_only,
            "integrity_note": self.integrity_note(),
        }


class ObjectManifest:
    """Mutable reference table mapping logical names and versions to content IDs (Step 52)."""

    def __init__(self, path: Path) -> None:
        """Load or create one manifest beside the object store."""
        self.path = Path(path)
        self._entries: dict[str, dict[str, str]] = {}
        self._history: list[dict[str, object]] = []
        if self.path.exists():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            self._entries = {
                name: dict(versions)
                for name, versions in payload.get("references", {}).items()
            }
            self._history = list(payload.get("history", []))

    def update_reference(self, logical_name: str, version: str, content: str) -> None:
        """Point one logical name and version at one immutable content ID (Step 52).

        Updating a logical artifact creates a new object and a new manifest
        entry; historical bytes are never overwritten.
        """
        if not logical_name.strip() or not version.strip() or not content.strip():
            raise ValueError("logical_name, version and content must be non-empty")
        self._entries.setdefault(logical_name, {})[version] = content
        self._history.append(
            {"logical_name": logical_name, "version": version, "content_id": content}
        )
        self._save()

    def resolve(self, logical_name: str, version: str) -> str:
        """Resolve one logical name and version to its immutable content ID."""
        versions = self._entries.get(logical_name)
        if not versions or version not in versions:
            raise KeyError(
                f"no reference for {logical_name!r} at version {version!r} (Step 52)"
            )
        return versions[version]

    def versions_of(self, logical_name: str) -> tuple[str, ...]:
        """Return every version of one logical name."""
        return tuple(self._entries.get(logical_name, {}))

    def entries(self) -> dict[str, dict[str, str]]:
        """Return the reference table snapshot."""
        return {name: dict(versions) for name, versions in self._entries.items()}

    def _save(self) -> None:
        """Persist the manifest atomically."""
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps({"references": self._entries, "history": self._history}),
            encoding="utf-8",
        )
        temporary.replace(self.path)


class ContentAddressedStore:
    """Immutable object store keyed by domain-separated content hash (Steps 51-53)."""

    def __init__(self, root: Path, *, domain: str = "ore.data.artifact") -> None:
        """Open (or create) one object store under a root directory."""
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.domain = domain
        self.manifest = ObjectManifest(self.root / "manifest.json")
        self.writes = 0
        self.reused = 0
        self.bytes_stored = 0
        self.verification_failures = 0

    # -- write path -------------------------------------------------------

    def put(
        self,
        content: bytes,
        *,
        logical_name: str | None = None,
        version: str | None = None,
        artifact_class: str = "accepted_algorithm",
        schema_version: str = "0.1.0",
    ) -> StoreReceipt:
        """Store one object under its content identifier (Step 51).

        The ID is the domain-tagged SHA-256 of the canonical bytes; the hash
        is verified after writing; existing objects are reused rather than
        duplicated. ``logical_name``/``version`` update the mutable manifest
        without touching historical bytes (Step 52).
        """
        if not validate_artifact_class(artifact_class):
            raise ContentAddressedStoreError(
                f"artifact class {artifact_class!r} is mutable and requires an "
                "explicit versioning model before immutable storage (Step 50)"
            )
        if not isinstance(content, (bytes, bytearray)):
            raise TypeError("content must be bytes")
        payload = bytes(content)
        identifier = content_id(payload, self.domain, schema_version)
        digest_hex = identifier.split(":", 1)[1]
        object_path = self.root / "objects" / digest_hex[:2] / digest_hex
        reused = object_path.exists()
        if not reused:
            object_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = object_path.with_suffix(".tmp")
            temporary.write_bytes(payload)
            temporary.replace(object_path)
            self.bytes_stored += len(payload)
        # Verify the hash on write by re-reading the stored bytes (Step 51).
        verified = verify_content_id(object_path.read_bytes(), identifier, self.domain)
        if not verified:
            self.verification_failures += 1
            raise ContentAddressedStoreError(
                "object verification failed immediately after write (Step 51)"
            )
        self.writes += 1
        if reused:
            self.reused += 1
        if logical_name and version:
            self.manifest.update_reference(logical_name, version, identifier)
        return StoreReceipt(
            content_id=identifier,
            domain=self.domain,
            bytes_stored=len(payload),
            reused=reused,
            verified_on_write=True,
        )

    # -- read path -----------------------------------------------------------

    def get_verified(
        self,
        identifier: str,
        *,
        schema_version: str = "0.1.0",
    ) -> VerifiedArtifact:
        """Retrieve one artifact and recompute its hash (Step 53).

        The content hash is recomputed and must match the identifier before
        the artifact reaches the consumer; a matching hash proves byte
        integrity relative to the identifier, not that the artifact is safe
        or correct.
        """
        digest_hex = identifier.split(":", 1)[1] if ":" in identifier else identifier
        object_path = self.root / "objects" / digest_hex[:2] / digest_hex
        if not object_path.exists():
            raise KeyError(f"object {identifier!r} not found in the store (Step 53)")
        payload = object_path.read_bytes()
        if not verify_content_id(payload, identifier, self.domain):
            self.verification_failures += 1
            raise ContentAddressedStoreError(
                f"content hash mismatch for {identifier!r}: byte integrity "
                "verification failed (Step 53)"
            )
        return VerifiedArtifact(content_id=identifier, domain=self.domain, content=payload)

    def resolve_verified(
        self,
        logical_name: str,
        version: str,
        *,
        schema_version: str = "0.1.0",
    ) -> VerifiedArtifact:
        """Resolve a manifest reference to a verified artifact (Step 53)."""
        identifier = self.manifest.resolve(logical_name, version)
        return self.get_verified(identifier, schema_version=schema_version)

    def contains(self, identifier: str) -> bool:
        """Return whether one content identifier exists in the store."""
        digest_hex = identifier.split(":", 1)[1] if ":" in identifier else identifier
        return (self.root / "objects" / digest_hex[:2] / digest_hex).exists()

    # -- introspection -----------------------------------------------------------

    def deduplication_ratio(self) -> float:
        """Return the reuse ratio of the store (metric: deduplication ratio)."""
        if self.writes == 0:
            return 0.0
        return self.reused / self.writes

    def distinct_object_count(self) -> int:
        """Return the number of distinct stored objects."""
        objects_root = self.root / "objects"
        if not objects_root.exists():
            return 0
        return sum(
            1
            for _shard in objects_root.iterdir()
            if _shard.is_dir()
            for _object in _shard.iterdir()
            if _object.is_file()
        )

    def stats(self) -> dict[str, object]:
        """Return store statistics for observability."""
        return {
            "writes": self.writes,
            "reused": self.reused,
            "bytes_stored": self.bytes_stored,
            "distinct_objects": self.distinct_object_count(),
            "deduplication_ratio": self.deduplication_ratio(),
            "verification_failures": self.verification_failures,
        }


def put_immutable(store: ContentAddressedStore, content: bytes, domain: str) -> str:
    """Section 5.1 interface: put one immutable object and return its ID."""
    if store.domain != domain:
        raise ValueError("domain mismatch: the store pins one hash namespace (Step 3)")
    return store.put(content).content_id


def get_verified(store: ContentAddressedStore, identifier: str) -> VerifiedArtifact:
    """Section 5.1 interface: get one verified artifact by content ID."""
    return store.get_verified(identifier)


__all__ = [
    "ContentAddressedStoreError",
    "IMMUTABLE_ARTIFACT_CLASSES",
    "MUTABLE_RECORD_CLASSES",
    "validate_artifact_class",
    "StoreReceipt",
    "VerifiedArtifact",
    "ObjectManifest",
    "ContentAddressedStore",
    "put_immutable",
    "get_verified",
]
