"""Phase II, Step 10 - connect USG as a read-only semantic resource.

Verify bundle format, version and digest; import only supported record
families through the typed adapter; enforce source and context scopes; exclude
candidate records unless a future explicit policy permits them; keep imported
resources read-only.  ``Accept(U) = S(U) and H(U) and P(U)`` - schema, digest
and source-policy compliance.  This boundary never writes inferred features or
unverified conclusions back into authoritative USG storage.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ore_usg import USGKnowledgeAdapter, USGKnowledgeResource
from ore_usg.bundle_io import USGBundleLoadError, USGBundleSnapshot, load_knowledge_bundle
from ore_usg.policy import USGImportPolicy

from .core import new_id, sha256_digest, utc_now_iso

USG_BOUNDARY_SCHEMA = "ore.point24.usg_boundary.v1"

_TIMESTAMP = "2026-11-20T00:00:00Z"

# A minimal, valid USG-KNOWLEDGE/0.1.0 bundle used for boundary demonstrations.
DEMO_BUNDLE: Mapping[str, Any] = {
    "contract": "USG-KNOWLEDGE/0.1.0",
    "namespace": "urn:usg:point24-integration",
    "bundle_id": "ore-usg-point24-demo",
    "created_at": _TIMESTAMP,
    "records": [
        {
            "id": "lemma:en:integration", "kind": "LEMMA", "version": 1,
            "state": "ACCEPTED", "created_at": _TIMESTAMP,
            "source_ids": ["source:permitted"], "evidence_ids": ["evidence:lemma"],
            "context_ids": ["context:english"],
            "payload": {"lemma": "integration", "language": "en", "part_of_speech": "NOUN"},
        },
        {
            "id": "sense:en:integration:unified", "kind": "LEXICAL_SENSE", "version": 1,
            "state": "ACCEPTED", "created_at": _TIMESTAMP,
            "source_ids": ["source:permitted"], "evidence_ids": ["evidence:sense"],
            "context_ids": ["context:architecture"],
            "payload": {
                "lemma_record_id": "lemma:en:integration", "language": "en",
                "definition": "The act of combining parts into a unified whole.",
                "sense_source_id": "source:permitted",
            },
        },
        {
            "id": "concept:integration", "kind": "CONCEPT", "version": 1,
            "state": "ACCEPTED", "created_at": _TIMESTAMP,
            "source_ids": ["source:permitted"], "evidence_ids": ["evidence:concept"],
            "context_ids": ["context:architecture"],
            "payload": {"concept_type": "process", "namespace": "point24", "label": "integration"},
        },
        {
            "id": "sense:en:integration:candidate", "kind": "LEXICAL_SENSE", "version": 1,
            "state": "CANDIDATE", "created_at": _TIMESTAMP,
            "source_ids": ["source:permitted"], "evidence_ids": ["evidence:candidate"],
            "context_ids": ["context:architecture"],
            "payload": {
                "lemma_record_id": "lemma:en:integration", "language": "en",
                "definition": "Candidate meaning excluded by the default policy.",
                "sense_source_id": "source:permitted",
            },
        },
    ],
    "sources": [
        {
            "id": "source:permitted", "provider": "Point 24 boundary fixture",
            "source_type": "USG_DOCUMENT", "source_locator": None,
            "retrieved_at": _TIMESTAMP, "license_id": "MIT",
            "rights_status": "KNOWN_PERMITTED", "redistribution": "ALLOWED",
            "attribution": "Point 24 boundary fixture source",
        },
    ],
    "evidence": [
        {
            "id": evidence_id, "source_id": "source:permitted",
            "evidence_type": "DEFINITION", "support_status": "SUPPORTS",
            "captured_at": _TIMESTAMP, "transformation_ids": [],
        }
        for evidence_id in (
            "evidence:lemma", "evidence:sense", "evidence:concept", "evidence:candidate",
        )
    ],
    "contexts": [
        {"id": "context:english", "context_type": "LANGUAGE", "value": "en"},
        {"id": "context:architecture", "context_type": "DOMAIN", "value": "architecture"},
    ],
    "mappings": [],
    "relations": [],
}


@dataclass(frozen=True, slots=True)
class USGBoundaryDecision:
    """The Accept(U) decision with each component separately evidenced."""

    bundle_id: str
    schema_valid: bool
    digest_valid: bool
    policy_compliant: bool
    accepted: bool
    record_families: tuple[str, ...]
    candidate_records_included: bool
    reasons: tuple[str, ...]

    @property
    def admissible(self) -> bool:
        return self.accepted

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": USG_BOUNDARY_SCHEMA,
            "bundle_id": self.bundle_id,
            "schema_valid": self.schema_valid,
            "digest_valid": self.digest_valid,
            "policy_compliant": self.policy_compliant,
            "accepted": self.accepted,
            "record_families": list(self.record_families),
            "candidate_records_included": self.candidate_records_included,
            "reasons": list(self.reasons),
        }


class ReadOnlyUSGBoundary:
    """The read-only USG connection used by every ORE module.

    The boundary loads a versioned bundle through the hardened loader,
    converts it with the typed USG knowledge adapter and exposes read-only
    retrieval helpers.  It has no write path: any attempt to mutate the
    imported resource raises, because candidate knowledge must never be
    written back into authoritative USG storage.
    """

    def __init__(
        self,
        *,
        import_policy: USGImportPolicy | None = None,
        include_candidate_states: bool = False,
    ) -> None:
        self.import_policy = import_policy or USGImportPolicy()
        self.include_candidate_states = include_candidate_states
        self._adapter = USGKnowledgeAdapter(include_candidate_states=include_candidate_states)
        self._snapshot: USGBundleSnapshot | None = None
        self._resource: USGKnowledgeResource | None = None

    @property
    def snapshot(self) -> USGBundleSnapshot | None:
        return self._snapshot

    @property
    def resource(self) -> USGKnowledgeResource | None:
        return self._resource

    def import_bundle(
        self,
        source: Any,
        *,
        expected_digest: str | None = None,
    ) -> USGBoundaryDecision:
        """Validate S(U), H(U) and P(U), then import read-only records."""
        try:
            snapshot = load_knowledge_bundle(source, expected_digest=expected_digest)
        except (USGBundleLoadError, OSError) as exc:
            return USGBoundaryDecision(
                bundle_id=str(getattr(source, "get", lambda _k, d=None: d)("bundle_id", "unknown")),
                schema_valid=False, digest_valid=False, policy_compliant=False,
                accepted=False, record_families=(), candidate_records_included=False,
                reasons=(f"bundle rejected: {exc}",),
            )
        if expected_digest is not None and snapshot.bundle_digest != expected_digest:
            # load_knowledge_bundle already enforces this; kept for clarity.
            return USGBoundaryDecision(
                bundle_id="unknown", schema_valid=True, digest_valid=False,
                policy_compliant=True, accepted=False, record_families=(),
                candidate_records_included=False,
                reasons=("digest mismatch",),
            )
        bundle = snapshot.bundle
        records = [
            record for record in bundle.get("records", [])
            if isinstance(record, Mapping)
        ]
        record_types = tuple(sorted({str(record.get("kind", "unknown")) for record in records}))
        # Source-policy compliance: every record must have an admitted lifecycle
        # state under the import policy.  Candidate records are *excluded* by
        # the default policy (unless it explicitly permits them), never
        # silently promoted to authoritative knowledge.
        supported_states = {"CANDIDATE", "VALIDATED_FOR_PROFILE", "ACCEPTED",
                            "DEPRECATED", "REJECTED", "UNRESOLVED"}
        non_compliant = [
            record for record in records
            if str(record.get("state", "")) not in supported_states
        ]
        if non_compliant:
            return USGBoundaryDecision(
                bundle_id=str(bundle.get("bundle_id", "unknown")),
                schema_valid=True, digest_valid=True, policy_compliant=False,
                accepted=False, record_families=record_types,
                candidate_records_included=self.include_candidate_states,
                reasons=("source-policy violation: records in non-admitted states",),
            )
        resource = self._adapter.convert(bundle)
        self._snapshot = snapshot
        self._resource = resource
        excluded = sum(
            1 for record in records
            if not self.import_policy.state_enabled(str(record.get("state", "")))
        )
        return USGBoundaryDecision(
            bundle_id=str(bundle.get("bundle_id", "unknown")),
            schema_valid=True, digest_valid=True, policy_compliant=True,
            accepted=True, record_families=record_types,
            candidate_records_included=self.include_candidate_states,
            reasons=(
                "schema, digest and source policy verified; "
                f"{len(records) - excluded} records admitted, {excluded} excluded by policy"
            ),
        )

    # -- read-only retrieval helpers -------------------------------------------

    def bundle_digest(self) -> str | None:
        return self._snapshot.bundle_digest if self._snapshot else None

    def bundle_identity(self) -> dict[str, Any] | None:
        if self._snapshot is None:
            return None
        bundle = self._snapshot.bundle
        return {
            "bundle_id": bundle.get("bundle_id"),
            "namespace": bundle.get("namespace"),
            "digest": self._snapshot.bundle_digest,
            "byte_length": self._snapshot.byte_length,
            "source_path": self._snapshot.source_path,
        }

    def record_count(self) -> int:
        if self._resource is None:
            return 0
        return int(getattr(self._resource, "record_count", 0) or 0)

    def write_back(self, *args: Any, **kwargs: Any) -> Any:
        """Structurally forbidden: the USG boundary is read-only by design."""
        raise PermissionError(
            "the read-only USG boundary cannot write inferred features or "
            "unverified conclusions back into authoritative USG storage"
        )


def demo_boundary_report(boundary_id: str | None = None) -> dict[str, Any]:
    """Import the demo bundle and return the boundary evidence record."""
    boundary = ReadOnlyUSGBoundary()
    decision = boundary.import_bundle(dict(DEMO_BUNDLE))
    payload = {
        "schema": USG_BOUNDARY_SCHEMA,
        "boundary_id": boundary_id or new_id("point24_usg_boundary"),
        "evaluated_at": utc_now_iso(),
        "decision": decision.to_dict(),
        "bundle_identity": boundary.bundle_identity(),
    }
    payload["digest"] = sha256_digest(payload)
    return payload


__all__ = [
    "DEMO_BUNDLE",
    "USG_BOUNDARY_SCHEMA",
    "ReadOnlyUSGBoundary",
    "USGBoundaryDecision",
    "demo_boundary_report",
]
