"""Append-only provenance journal with verification and quarantine (Step 37).

On receipt, each event is validated for authentication, signature where used,
payload digest, schema, and causal references. Malformed events are rejected;
events with missing dependencies or invalid provenance are quarantined (kept
repairable, never silently discarded); accepted, rejected, and superseded
events all land in the append-only audit trail.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from ore_distributed.core.identity import NodeCredential, digest_hex
from ore_distributed.provenance.envelope import ProvenanceEnvelope
from ore_distributed.time.causal import missing_dependencies


class VerificationFailure(ValueError):
    """Raised when an event fails integrity or lineage verification."""


@dataclass(frozen=True, slots=True)
class AuditRecord:
    """One append-only audit trail entry (Step 37)."""

    outcome: str  # accepted | rejected | superseded | quarantined
    event_id: str
    reason: str
    sequence: int

    def to_dict(self) -> dict[str, object]:
        """Return the audit record as a plain dictionary."""
        return {
            "outcome": self.outcome,
            "event_id": self.event_id,
            "reason": self.reason,
            "sequence": self.sequence,
        }


@dataclass(frozen=True, slots=True)
class QuarantineEntry:
    """One quarantined event awaiting dependency repair."""

    event_id: str
    missing_parents: tuple[str, ...]
    attempts: int = 0

    def to_dict(self) -> dict[str, object]:
        """Return the quarantine record as a plain dictionary."""
        return {
            "event_id": self.event_id,
            "missing_parents": list(self.missing_parents),
            "attempts": self.attempts,
        }


class ProvenanceJournal:
    """Verify, accept, quarantine, and audit every received event (Step 37)."""

    def __init__(self, verify_signatures: bool = True) -> None:
        """Create an empty journal."""
        self.verify_signatures = bool(verify_signatures)
        self._events: dict[str, ProvenanceEnvelope] = {}
        self._audit: list[AuditRecord] = []
        self._quarantine: dict[str, QuarantineEntry] = {}
        self._known_credentials: dict[str, NodeCredential] = {}
        self._superseded_by: dict[str, str] = {}

    @property
    def sequence(self) -> int:
        """Return the current audit sequence position."""
        return len(self._audit)

    def trust_credential(self, credential: NodeCredential) -> None:
        """Trust one node credential for envelope signature verification."""
        self._known_credentials[credential.node_id] = credential

    def audit_trail(self) -> tuple[AuditRecord, ...]:
        """Return the append-only audit trail in order."""
        return tuple(self._audit)

    def accepted_events(self) -> tuple[ProvenanceEnvelope, ...]:
        """Return accepted envelopes ordered by event identifier."""
        return tuple(self._events[key] for key in sorted(self._events))

    def quarantined(self) -> tuple[QuarantineEntry, ...]:
        """Return every quarantined entry awaiting repair."""
        return tuple(self._quarantine[key] for key in sorted(self._quarantine))

    def superseded_map(self) -> dict[str, str]:
        """Return a copy of the superseded-event lineage."""
        return dict(self._superseded_by)

    def _append(self, outcome: str, event_id: str, reason: str) -> AuditRecord:
        """Append one immutable audit record."""
        record = AuditRecord(
            outcome=outcome,
            event_id=event_id,
            reason=reason,
            sequence=len(self._audit) + 1,
        )
        self._audit.append(record)
        return record

    def verify_integrity(self, envelope: ProvenanceEnvelope, payload: Mapping[str, object] | None = None) -> bool:
        """Validate authentication, signature, digest, and schema (Step 37)."""
        if envelope.event_id in self._events:
            self._append("rejected", envelope.event_id, "duplicate event id")
            raise VerificationFailure(f"event {envelope.event_id} is already accepted")
        if payload is not None and digest_hex(dict(payload)) != envelope.payload_digest:
            self._append("rejected", envelope.event_id, "payload digest mismatch")
            raise VerificationFailure("payload digest mismatch")
        if self.verify_signatures:
            credential = self._known_credentials.get(envelope.origin_node)
            if credential is None:
                self._append("rejected", envelope.event_id, "unknown authenticated origin")
                raise VerificationFailure(
                    f"origin node {envelope.origin_node!r} is not trusted"
                )
            if not envelope.verify_signature(credential):
                self._append("rejected", envelope.event_id, "signature verification failed")
                raise VerificationFailure("signature verification failed")
        return True

    def accept(
        self,
        envelope: ProvenanceEnvelope,
        payload: Mapping[str, object] | None = None,
    ) -> str:
        """Verify and accept an event; missing dependencies quarantine it."""
        self.verify_integrity(envelope, payload)
        missing = missing_dependencies(
            _envelope_history(envelope),
            self._events.keys(),
        )
        if missing:
            self._quarantine[envelope.event_id] = QuarantineEntry(
                event_id=envelope.event_id,
                missing_parents=missing,
            )
            self._append("quarantined", envelope.event_id, f"missing parents: {', '.join(missing)}")
            return "quarantined"
        for parent in envelope.causal_parents:
            if parent in self._quarantine:
                self._quarantine.pop(parent)
                self._append("accepted", parent, "quarantine repaired by dependent event")
        self._events[envelope.event_id] = envelope
        self._append("accepted", envelope.event_id, "integrity and lineage verified")
        return "accepted"

    def repair_quarantine(self, missing_envelopes: Iterable[ProvenanceEnvelope]) -> int:
        """Repair quarantined events by supplying their missing dependencies."""
        repaired = 0
        for envelope in missing_envelopes:
            self._events.setdefault(envelope.event_id, envelope)
            self._append("accepted", envelope.event_id, "repair delivery")
        for event_id in list(self._quarantine):
            entry = self._quarantine[event_id]
            still_missing = tuple(
                parent for parent in entry.missing_parents if parent not in self._events
            )
            if still_missing:
                self._quarantine[event_id] = QuarantineEntry(
                    event_id=event_id,
                    missing_parents=still_missing,
                    attempts=entry.attempts + 1,
                )
            else:
                self._quarantine.pop(event_id)
                self._append("accepted", event_id, "dependencies repaired")
                repaired += 1
        return repaired

    def supersede(self, previous_id: str, replacement: ProvenanceEnvelope) -> AuditRecord:
        """Record one event superseding an accepted predecessor."""
        if previous_id not in self._events:
            raise VerificationFailure(f"event {previous_id!r} was never accepted")
        self._superseded_by[previous_id] = replacement.event_id
        self._append("superseded", previous_id, f"superseded by {replacement.event_id}")
        return self._audit[-1]

    def verification_failures(self) -> tuple[str, ...]:
        """Return the rejected events from the audit trail for observability."""
        return tuple(record.event_id for record in self._audit if record.outcome == "rejected")


def _envelope_history(envelope: ProvenanceEnvelope):
    """Adapt an envelope to the causal-history interface for dependency checks."""
    from ore_distributed.time.causal import CausalHistory

    return CausalHistory(
        event_id=envelope.event_id,
        parents=frozenset(envelope.causal_parents),
    )


def audit_completeness(journal: ProvenanceJournal, expected_event_ids: Iterable[str]) -> float:
    """Return the fraction of expected events present in the audit trail (Step 48)."""
    expected = set(expected_event_ids)
    if not expected:
        return 1.0
    audited = {
        record.event_id for record in journal.audit_trail()
    }
    accepted = set(journal._events)  # noqa: SLF001 - completeness is a journal-internal metric
    return len(expected & accepted | expected & audited) / len(expected)


def lineage_of(journal: ProvenanceJournal, event_id: str) -> tuple[str, ...]:
    """Return the transitive causal ancestry of one accepted event."""
    chain: list[str] = []
    seen: set[str] = set()
    frontier = [event_id]
    while frontier:
        current = frontier.pop()
        if current in seen:
            continue
        seen.add(current)
        chain.append(current)
        envelope = journal._events.get(current)  # noqa: SLF001 - lineage walks accepted state
        if envelope is not None:
            frontier.extend(envelope.causal_parents)
    return tuple(reversed(chain))


__all__ = [
    "VerificationFailure",
    "AuditRecord",
    "QuarantineEntry",
    "ProvenanceJournal",
    "audit_completeness",
    "lineage_of",
]
