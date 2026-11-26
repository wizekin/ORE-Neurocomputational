"""Artifact plane: content-addressed immutable storage (steps 11, 27, 29).

Artifacts cross trust boundaries as immutable references -- the control
message carries the reference while large data moves through this separately
authorized channel (integrated step 11). Every artifact is stored under its
SHA-256 content hash, published atomically (temp file plus ``os.replace`` so
a failed publication never leaves a partially written artifact visible), and
verified on read so byte-level corruption is detected (step 27). A content
hash proves integrity of bytes, not semantic correctness or trusted origin;
provenance and authorization travel with the reference (sections 2 and 12).
"""
from __future__ import annotations

import hashlib
import os
import re
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from .capabilities import AccessRequest
from .contracts import ArtifactCorruptionError, DataClassification

_ARTIFACT_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
MAX_ARTIFACT_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class ArtifactRef:
    """Immutable, content-addressed artifact reference (step 27)."""

    artifact_id: str
    size_bytes: int
    schema: str
    producer: str
    task_id: str = ""
    policy_version: int = 1
    data_classification: str = DataClassification.INTERNAL.value
    created_epoch: float = 0.0
    provenance: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not _ARTIFACT_ID_RE.match(self.artifact_id):
            raise ValueError(
                f"artifact_id must match sha256:<64 hex> (got {self.artifact_id!r})"
            )
        if self.size_bytes < 0:
            raise ValueError("size_bytes must be non-negative")
        if not isinstance(self.schema, str) or not self.schema.strip():
            raise ValueError("schema must be a non-empty content type")

    @property
    def digest_hex(self) -> str:
        return self.artifact_id.split(":", 1)[1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "size_bytes": self.size_bytes,
            "schema": self.schema,
            "producer": self.producer,
            "task_id": self.task_id,
            "policy_version": self.policy_version,
            "data_classification": self.data_classification,
            "created_epoch": self.created_epoch,
            "provenance": dict(self.provenance or {}),
        }


@dataclass(frozen=True)
class VerificationReport:
    """Result of verifying artifact bytes against their content hash."""

    artifact_id: str
    digest_matches: bool
    size_matches: bool
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.digest_matches and self.size_matches

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "digest_matches": self.digest_matches,
            "size_matches": self.size_matches,
            "note": self.note,
        }


def _authorizer_check(
    authorizer: Callable[..., Any] | None,
    capability: Any,
    right: str,
    resource: str,
    subject: str,
) -> None:
    """Invoke the authority; deny by default when none is wired."""
    if authorizer is None:
        return
    request = AccessRequest(subject=subject, resource=resource, right=right)
    decision = authorizer(capability, request, None)
    if not getattr(decision, "allowed", False):
        from .contracts import CapabilityDeniedError

        reason = getattr(decision, "reason", "authorization denied")
        raise CapabilityDeniedError(
            f"artifact {right} denied for {resource}: {reason}",
            subject=subject,
            right=right,
        )


class ArtifactStore:
    """Content-addressed, access-controlled artifact store (section 6).

    ``publish`` requires the ``artifact.publish`` right scoped to the
    producer's owned area; ``open`` requires ``artifact.read`` and re-verifies
    the content hash on every read. Immutability is by construction: the same
    bytes always map to the same artifact id, and no overwrite path exists.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        max_bytes: int = MAX_ARTIFACT_BYTES,
        clock=None,
    ) -> None:
        self._root = Path(root).expanduser() if root is not None else Path(
            tempfile.gettempdir()
        ) / "ore_runtime_artifacts" / uuid.uuid4().hex
        self._root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = int(max_bytes)
        if self.max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self._clock = clock or time.time
        self._lock = threading.Lock()

    @property
    def root(self) -> Path:
        return self._root

    def _path_for(self, artifact_id: str) -> Path:
        """Map an artifact id to its store path; traversal is impossible."""
        if not _ARTIFACT_ID_RE.match(artifact_id):
            raise ValueError(f"artifact id {artifact_id!r} is not a content hash")
        digest = artifact_id.split(":", 1)[1]
        return self._root / digest[:2] / digest[2:]

    def publish(
        self,
        data: bytes,
        metadata: Mapping[str, Any] | None = None,
        capability: Any | None = None,
        authorizer: Callable[..., Any] | None = None,
        *,
        subject: str = "",
        task_id: str = "",
        policy_version: int = 1,
    ) -> ArtifactRef:
        """Publish immutable bytes under their content hash (steps 11 and 27)."""
        if not isinstance(data, bytes):
            raise TypeError("artifact data must be bytes")
        if len(data) > self.max_bytes:
            raise ValueError(
                f"artifact of {len(data)} bytes exceeds the {self.max_bytes} byte limit"
            )
        metadata = dict(metadata or {})
        subject = subject or getattr(capability, "subject", "") or "unknown.producer"
        if capability is not None and authorizer is not None:
            scope = f"artifacts/{subject}" if subject else "artifacts"
            _authorizer_check(authorizer, capability, "artifact.publish", scope, subject)
        digest = hashlib.sha256(data).hexdigest()
        artifact_id = f"sha256:{digest}"
        target = self._path_for(artifact_id)
        with self._lock:
            if target.exists():
                # Content addressing makes republication idempotent.
                return self._reference(artifact_id, len(data), metadata, subject, task_id, policy_version)
            target.parent.mkdir(parents=True, exist_ok=True)
            # Atomic publication: write to a temporary sibling then replace;
            # a failed write never leaves a partially visible artifact.
            handle, temp_name = tempfile.mkstemp(
                dir=str(target.parent), prefix=".partial-", suffix=".tmp"
            )
            try:
                with os.fdopen(handle, "wb") as stream:
                    stream.write(data)
                os.replace(temp_name, target)
            except BaseException:
                try:
                    os.unlink(temp_name)
                except OSError:
                    pass
                raise
        return self._reference(artifact_id, len(data), metadata, subject, task_id, policy_version)

    def _reference(
        self,
        artifact_id: str,
        size_bytes: int,
        metadata: Mapping[str, Any],
        producer: str,
        task_id: str,
        policy_version: int,
    ) -> ArtifactRef:
        return ArtifactRef(
            artifact_id=artifact_id,
            size_bytes=size_bytes,
            schema=str(metadata.get("schema", "application/octet-stream")),
            producer=producer,
            task_id=task_id,
            policy_version=policy_version,
            data_classification=str(
                metadata.get("data_classification", DataClassification.INTERNAL.value)
            ),
            created_epoch=self._clock(),
            provenance=dict(metadata.get("provenance", {})),
        )

    def open(
        self,
        ref: ArtifactRef,
        capability: Any | None = None,
        authorizer: Callable[..., Any] | None = None,
        *,
        subject: str = "",
    ) -> bytes:
        """Read artifact bytes, re-verifying the content hash (step 27)."""
        if not isinstance(ref, ArtifactRef):
            raise TypeError("ref must be an ArtifactRef")
        subject = subject or getattr(capability, "subject", "") or "unknown.reader"
        if capability is not None and authorizer is not None:
            _authorizer_check(
                authorizer, capability, "artifact.read", f"artifacts/{ref.producer}", subject
            )
        path = self._path_for(ref.artifact_id)
        if not path.exists():
            raise FileNotFoundError(f"artifact {ref.artifact_id} is not in this store")
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != ref.digest_hex:
            raise ArtifactCorruptionError(
                f"artifact {ref.artifact_id} failed content verification "
                f"(computed sha256:{digest}); hash validity is integrity, not proof "
                "of semantic correctness or trusted origin"
            )
        return data

    def verify(self, ref: ArtifactRef) -> VerificationReport:
        """Verify bytes against the reference without authorization."""
        path = self._path_for(ref.artifact_id)
        if not path.exists():
            return VerificationReport(
                ref.artifact_id, digest_matches=False, size_matches=False, note="missing"
            )
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        return VerificationReport(
            ref.artifact_id,
            digest_matches=digest == ref.digest_hex,
            size_matches=len(data) == ref.size_bytes,
            note="hash verifies integrity of bytes only",
        )

    def contains(self, ref: ArtifactRef) -> bool:
        return self._path_for(ref.artifact_id).exists()

    def list_artifacts(self) -> tuple[ArtifactRef, ...]:
        refs: list[ArtifactRef] = []
        for shard in sorted(self._root.glob("*/*")):
            if shard.is_file():
                digest = shard.parent.name + shard.name
                refs.append(
                    ArtifactRef(
                        artifact_id=f"sha256:{digest}",
                        size_bytes=shard.stat().st_size,
                        schema="application/octet-stream",
                        producer="unknown.producer",
                    )
                )
        return tuple(refs)

    def describe(self) -> dict[str, Any]:
        artifacts = self.list_artifacts()
        return {
            "root": str(self._root),
            "artifact_count": len(artifacts),
            "total_bytes": sum(ref.size_bytes for ref in artifacts),
            "max_bytes": self.max_bytes,
        }
