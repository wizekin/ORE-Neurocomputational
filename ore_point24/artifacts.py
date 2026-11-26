"""Phase II, Step 6 - canonical artifact envelopes.

One canonical envelope wraps every module output: artifact ID and type,
contract ID and exact version, producer module and execution run, payload or
payload reference, payload digest, creation timestamp, parent artifact IDs,
source and evidence references, validation status, permission and retention
metadata.  Identity is the deterministic fingerprint ``F(a) = SHA256(canonical(a))``
built on ``ore_contracts.module_artifacts.content_hash``; identical IDs with
different content are rejected as identity corruption.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ore_contracts import ContractEnvelope, EvidenceKind, sha256_json
from ore_contracts.module_artifacts import ContractArtifact, content_hash

from .core import new_id, sha256_digest, utc_now_iso

CANONICAL_ENVELOPE_CONTRACT = "P24-CANONICAL-ARTIFACT"
CANONICAL_ENVELOPE_VERSION = "1.0.0"


def make_envelope(
    *,
    contract_id: str,
    schema_version: str,
    run_id: str,
    producer_id: str,
    config: Mapping[str, Any],
    evidence_kind: EvidenceKind | str = EvidenceKind.DERIVED,
    parent_artifact_ids: tuple[str, ...] = (),
    created_at: str | None = None,
) -> ContractEnvelope:
    """Build a provenance-complete ore_contracts envelope (config is hashed)."""
    return ContractEnvelope(
        contract_id=contract_id,
        schema_version=schema_version,
        run_id=run_id,
        producer_id=producer_id,
        config_hash=sha256_json(dict(config)),
        created_at=created_at or utc_now_iso(),
        evidence_kind=evidence_kind,
        parent_artifact_ids=tuple(parent_artifact_ids),
    )


class CanonicalArtifactLedger:
    """An append-only ledger enforcing Step 6 identity and lineage rules.

    - Identical canonical artifacts produce identical fingerprints.
    - Reusing an artifact ID for different content is rejected as identity
      corruption.
    - Missing parent artifact references are reported by ``validate_lineage``.
    """

    def __init__(self) -> None:
        self._artifacts: dict[str, ContractArtifact] = {}
        self._fingerprints: dict[str, str] = {}
        self._ids_by_fingerprint: dict[str, set[str]] = {}
        self._metadata: dict[str, dict[str, Any]] = {}

    def register(
        self,
        artifact: ContractArtifact,
        *,
        artifact_id: str | None = None,
        artifact_type: str = "artifact",
        source_references: tuple[str, ...] = (),
        permissions: Mapping[str, Any] | None = None,
        retention: str = "task_scoped",
    ) -> str:
        """Register one artifact under a stable identity; reject collisions."""
        if not isinstance(artifact, ContractArtifact):
            raise TypeError("artifact must be an ore_contracts ContractArtifact")
        fingerprint = artifact.payload_hash
        stable_id = artifact_id or f"artifact_{fingerprint[:16]}"
        if stable_id in self._artifacts:
            existing = self._artifacts[stable_id]
            if existing.payload_hash != fingerprint:
                raise ValueError(
                    f"identity corruption: artifact ID {stable_id!r} already exists with "
                    "different content"
                )
            if existing.envelope != artifact.envelope:
                raise ValueError(
                    f"identity corruption: artifact ID {stable_id!r} already exists with "
                    "a different provenance envelope"
                )
            return stable_id
        for other_id in self._ids_by_fingerprint.get(fingerprint, set()):
            # Same content under a different ID is legal; it is deduplicated by
            # fingerprint and reported through `duplicates`.
            continue
        self._artifacts[stable_id] = artifact
        self._fingerprints[stable_id] = fingerprint
        self._ids_by_fingerprint.setdefault(fingerprint, set()).add(stable_id)
        self._metadata[stable_id] = {
            "artifact_id": stable_id,
            "artifact_type": artifact_type,
            "fingerprint": fingerprint,
            "contract": f"{artifact.envelope.contract_id}@{artifact.envelope.schema_version}",
            "producer": artifact.envelope.producer_id,
            "run": artifact.envelope.run_id,
            "parents": list(artifact.envelope.parent_artifact_ids),
            "source_references": list(source_references),
            "validation_status": "registered",
            "permissions": dict(permissions or {}),
            "retention": retention,
            "registered_at": utc_now_iso(),
        }
        return stable_id

    def get(self, artifact_id: str) -> ContractArtifact | None:
        return self._artifacts.get(artifact_id)

    def metadata(self, artifact_id: str) -> dict[str, Any] | None:
        record = self._metadata.get(artifact_id)
        return dict(record) if record else None

    def fingerprint(self, artifact_id: str) -> str | None:
        return self._fingerprints.get(artifact_id)

    def duplicates(self) -> dict[str, list[str]]:
        """Content-addressed duplicate groups: same fingerprint, several IDs."""
        return {
            fingerprint: sorted(ids)
            for fingerprint, ids in self._ids_by_fingerprint.items()
            if len(ids) > 1
        }

    def validate_lineage(self, artifact_id: str) -> tuple[str, ...]:
        """Return the list of missing parent references for one artifact."""
        artifact = self._artifacts.get(artifact_id)
        if artifact is None:
            raise KeyError(artifact_id)
        missing = [
            parent for parent in artifact.envelope.parent_artifact_ids
            if parent not in self._artifacts
        ]
        return tuple(missing)

    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._artifacts))

    def size(self) -> int:
        return len(self._artifacts)

    def summary(self) -> dict[str, Any]:
        """Return a detached summary of the ledger for audit records."""
        return {
            "artifact_count": len(self._artifacts),
            "unique_fingerprints": len(self._ids_by_fingerprint),
            "duplicates": self.duplicates(),
            "artifacts": {
                artifact_id: {
                    "fingerprint": self._fingerprints[artifact_id],
                    "missing_parents": list(self.validate_lineage(artifact_id)),
                    **(self._metadata[artifact_id] if artifact_id in self._metadata else {}),
                }
                for artifact_id in sorted(self._artifacts)
            },
        }


def canonical_artifact(
    payload: Mapping[str, Any],
    *,
    contract_id: str = CANONICAL_ENVELOPE_CONTRACT,
    schema_version: str = CANONICAL_ENVELOPE_VERSION,
    run_id: str,
    producer_id: str,
    config: Mapping[str, Any] | None = None,
    evidence_kind: EvidenceKind | str = EvidenceKind.DERIVED,
    parent_artifact_ids: tuple[str, ...] = (),
) -> ContractArtifact:
    """Create a ContractArtifact whose payload hash is its stable identity."""
    envelope = make_envelope(
        contract_id=contract_id,
        schema_version=schema_version,
        run_id=run_id,
        producer_id=producer_id,
        config=config or {"producer": producer_id},
        evidence_kind=evidence_kind,
        parent_artifact_ids=parent_artifact_ids,
    )
    return ContractArtifact(envelope=envelope, payload=dict(payload))


__all__ = [
    "CANONICAL_ENVELOPE_CONTRACT",
    "CANONICAL_ENVELOPE_VERSION",
    "CanonicalArtifactLedger",
    "canonical_artifact",
    "make_envelope",
]
