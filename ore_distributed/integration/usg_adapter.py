"""USG semantic validation gate for distributed findings (Step 47).

The adapter validates semantic structures, concept identifiers, relations,
evidence, and contradiction handling with the repository's own ore_usg
contracts before any promotion; it rejects invalid semantic schemas and
preserves evidence lineage. CROS-style adaptation and consensus ordering
never replace this epistemic validation.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Mapping

from ore_usg.validation import (
    USGKnowledgeValidationError,
    validate_knowledge_bundle,
)

from ore_distributed.replication.promotion import (
    CandidateAssertion,
    USGPromotionGate,
)


def _canonical_json(value: Any) -> str:
    """Serialize a bundle deterministically for digests."""
    import json

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _timestamp() -> str:
    """Return a timezone-aware USG timestamp."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass(frozen=True, slots=True)
class USGValidationOutcome:
    """One validation decision with lineage preserved for audit (Step 47)."""

    assertion_id: str
    valid: bool
    errors: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    relation_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the outcome as a plain dictionary."""
        return {
            "assertion_id": self.assertion_id,
            "valid": self.valid,
            "errors": list(self.errors),
            "evidence_ids": list(self.evidence_ids),
            "relation_ids": list(self.relation_ids),
        }


class USGValidationGate:
    """Validate candidate assertions as USG knowledge bundles (Step 47).

    A distributed observation becomes a candidate USG assertion only after
    schema validation through the repository's own USG knowledge contract;
    evidence lineage stays attached to the assertion throughout.
    """

    def __init__(self, namespace: str = "ore.distributed", gate: USGPromotionGate | None = None) -> None:
        """Create a validation gate bound to one USG namespace."""
        if not namespace.strip():
            raise ValueError("namespace must be non-empty")
        self.namespace = namespace.strip()
        self.gate = gate or USGPromotionGate()
        self.outcomes: list[USGValidationOutcome] = []
        self.rejected_schemas = 0

    def build_bundle(
        self,
        candidate: CandidateAssertion,
        *,
        evidence_payloads: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """Assemble a USG-KNOWLEDGE/0.1.0 bundle from one candidate assertion.

        The candidate's subject and object become two CONCEPT records joined
        by a directed relation; evidence payloads attach to the records so the
        bundle validation exercises the repository's exact ore_usg contracts.
        """
        evidence_payloads = evidence_payloads or {}
        subject_id = f"concept_{sha256(candidate.subject.encode('utf-8')).hexdigest()[:10]}"
        object_id = f"concept_{sha256(candidate.predicate.encode('utf-8')).hexdigest()[:10]}"
        source_id = f"src_{sha256(candidate.source_reference.encode('utf-8')).hexdigest()[:12]}"
        evidence_id = f"ev_{sha256(candidate.observed_event_id.encode('utf-8')).hexdigest()[:12]}"
        context_id = (
            f"ctx_{sha256(str(sorted(candidate.context)).encode('utf-8')).hexdigest()[:12]}"
        )
        now = _timestamp()
        bundle = {
            "contract": "USG-KNOWLEDGE/0.1.0",
            "schema_version": "0.1.0",
            "namespace": self.namespace,
            "bundle_id": f"bundle_{candidate.assertion_id}",
            "created_at": now,
            "records": [
                {
                    "id": subject_id,
                    "kind": "CONCEPT",
                    "version": 1,
                    "state": "CANDIDATE",
                    "created_at": now,
                    "source_ids": [source_id],
                    "evidence_ids": [evidence_id],
                    "context_ids": [context_id],
                    "payload": {
                        "concept_type": "ENTITY",
                        "namespace": self.namespace,
                        "label": candidate.subject,
                    },
                },
                {
                    "id": object_id,
                    "kind": "CONCEPT",
                    "version": 1,
                    "state": "CANDIDATE",
                    "created_at": now,
                    "source_ids": [source_id],
                    "evidence_ids": [evidence_id],
                    "context_ids": [context_id],
                    "payload": {
                        "concept_type": "PREDICATION",
                        "namespace": self.namespace,
                        "label": candidate.predicate,
                    },
                },
            ],
            "sources": [
                {
                    "id": source_id,
                    "provider": "ore.distributed.node",
                    "source_type": "MODEL_PROPOSAL",
                    "source_locator": candidate.source_reference,
                    "retrieved_at": now,
                    "rights_status": "INTERNAL",
                    "redistribution": "NOT_APPLICABLE",
                    "attribution": f"ore-distributed:{candidate.observed_event_id}",
                }
            ],
            "evidence": [
                {
                    "id": evidence_id,
                    "source_id": source_id,
                    "evidence_type": "RELATION_OBSERVATION",
                    "support_status": "SUPPORTS",
                    "locator": candidate.source_reference,
                    "text_digest": candidate.provenance_digest,
                    "captured_at": now,
                    "transformation_ids": [],
                    "payload": dict(evidence_payloads),
                }
            ],
            "contexts": [
                {
                    "id": context_id,
                    "context_type": "TASK",
                    "value": str(candidate.context.get("task", "distributed_observation")),
                    "language_profile": None,
                    "scope": self.namespace,
                    "payload": dict(candidate.context),
                }
            ],
            "mappings": [],
            "relations": [
                {
                    "id": f"rel_{candidate.assertion_id}",
                    "subject_record_id": subject_id,
                    "predicate": candidate.predicate,
                    "object_record_id": object_id,
                    "direction": "DIRECTED",
                    "state": "CANDIDATE",
                    "evidence_ids": [evidence_id],
                    "context_ids": [context_id],
                    "payload": dict(candidate.payload),
                }
            ],
        }
        return bundle

    def validate_candidate(
        self,
        candidate: CandidateAssertion,
        *,
        evidence_payloads: Mapping[str, str] | None = None,
    ) -> USGValidationOutcome:
        """Validate one candidate's semantic schema with ore_usg contracts."""
        bundle = self.build_bundle(candidate, evidence_payloads=evidence_payloads)
        errors = validate_knowledge_bundle(bundle)
        if errors:
            self.rejected_schemas += 1
        outcome = USGValidationOutcome(
            assertion_id=candidate.assertion_id,
            valid=not errors,
            errors=tuple(errors),
            evidence_ids=tuple(
                evidence["id"] for evidence in bundle["evidence"]
            ),
            relation_ids=tuple(relation["id"] for relation in bundle["relations"]),
        )
        self.outcomes.append(outcome)
        return outcome

    def validate_and_promote(
        self,
        candidate: CandidateAssertion,
        *,
        evidence_payloads: Mapping[str, str] | None = None,
    ) -> USGValidationOutcome:
        """Require ore_usg schema validity before protocol-level promotion.

        Promotion to a candidate USG assertion needs both the USG schema
        validation and the protocol-level promotion gate; neither alone is
        sufficient (Step 30 and Step 47).
        """
        outcome = self.validate_candidate(candidate, evidence_payloads=evidence_payloads)
        if not outcome.valid:
            return outcome
        self.gate.evaluate(candidate)
        return outcome

    def reject_invalid_semantic_schema(self, malformed: Mapping[str, Any]) -> tuple[str, ...]:
        """Reject an invalid semantic schema outright and report the errors."""
        errors = validate_knowledge_bundle(dict(malformed))
        if errors:
            self.rejected_schemas += 1
        return tuple(errors)

    def evidence_lineage(self, assertion_id: str) -> tuple[str, ...]:
        """Return the preserved evidence identifiers of one assertion (Step 47)."""
        for outcome in self.outcomes:
            if outcome.assertion_id == assertion_id:
                return outcome.evidence_ids
        return ()


def usg_rejects_invalid_schema_check(gate: USGValidationGate) -> tuple[bool, str]:
    """Run the acceptance check: USG rejects invalid semantic schemas."""
    errors = gate.reject_invalid_semantic_schema({"contract": "wrong"})
    return bool(errors), "USG validation must reject malformed bundles"


__all__ = [
    "USGValidationOutcome",
    "USGValidationGate",
    "usg_rejects_invalid_schema_check",
    "USGKnowledgeValidationError",
]
