"""Candidate lifecycle audit reconstructed from recorded events (step 40, test 25).

The audit rebuilds, per candidate, the ordered lifecycle stages (proposed -> scored ->
ranked -> published -> verified -> dispositioned) from the coordinator's event stream -
either the in-memory recorder events or the SQLite rows written by ore_logger - and
reports gaps so a candidate that silently vanished is visible in the audit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .events import GraphEventRecorder, GraphLifecycleEvent

#: Ordered lifecycle stages a candidate can pass through.
LIFECYCLE_STAGES = ("proposed", "scored", "ranked", "published", "verified", "dispositioned")


@dataclass
class CandidateAudit:
    """The reconstructed lifecycle of one candidate."""

    candidate_id: str
    correlation_id: str
    stages: tuple[str, ...]
    final_disposition: str | None
    gaps: tuple[str, ...]
    evidence_ids: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id, "correlation_id": self.correlation_id,
            "stages": list(self.stages), "final_disposition": self.final_disposition,
            "gaps": list(self.gaps), "evidence_ids": list(self.evidence_ids),
        }


@dataclass
class AuditTrail:
    """Full audit result for one correlation ID."""

    correlation_id: str
    event_count: int
    candidate_audits: list[CandidateAudit]
    lifecycle_complete: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "correlation_id": self.correlation_id, "event_count": self.event_count,
            "candidate_audits": [audit.as_dict() for audit in self.candidate_audits],
            "lifecycle_complete": self.lifecycle_complete,
        }


class CandidateLifecycleAudit:
    """Reconstruct candidate lifecycles from events (in-memory or SQLite rows)."""

    def audit_events(self, events: Sequence[GraphLifecycleEvent]) -> AuditTrail:
        """Audit from an ordered event sequence for one correlation ID."""
        if not events:
            return AuditTrail("", 0, [], True)
        correlation_id = events[0].correlation_id
        stages_by_candidate: dict[str, list[str]] = {}
        dispositions: dict[str, str] = {}
        evidence: dict[str, tuple[str, ...]] = {}
        for event in events:
            payload = event.payload
            for candidate_id, stage in _candidate_stage_pairs(event.kind, payload):
                timeline = stages_by_candidate.setdefault(candidate_id, [])
                if stage not in timeline:
                    timeline.append(stage)
            for candidate_id, disposition in _candidate_dispositions(payload):
                dispositions[candidate_id] = disposition
                timeline = stages_by_candidate.setdefault(candidate_id, [])
                if "dispositioned" not in timeline:
                    timeline.append("dispositioned")
            for candidate_id, evidence_ids in _candidate_evidence(payload):
                evidence[candidate_id] = tuple(evidence_ids)
        audits: list[CandidateAudit] = []
        complete = True
        for candidate_id, stages in sorted(stages_by_candidate.items()):
            gaps = _lifecycle_gaps(stages)
            if gaps:
                complete = False
            audits.append(CandidateAudit(
                candidate_id=candidate_id, correlation_id=correlation_id,
                stages=tuple(stages), final_disposition=dispositions.get(candidate_id),
                gaps=tuple(gaps), evidence_ids=evidence.get(candidate_id, ()),
            ))
        return AuditTrail(correlation_id, len(events), audits, complete)

    def audit_correlation(
        self, recorder: GraphEventRecorder, correlation_id: str,
    ) -> AuditTrail:
        """Audit from an in-memory recorder for one correlation ID."""
        return self.audit_events(recorder.by_correlation_id(correlation_id))

    def audit_sqlite_rows(self, rows: Sequence[Mapping[str, Any]]) -> AuditTrail:
        """Audit from ore_logger SQLite event rows (run events with metadata_json)."""
        events: list[GraphLifecycleEvent] = []
        for row in rows:
            metadata = _metadata_of(row)
            events.append(GraphLifecycleEvent(
                kind=str(row.get("event_kind") or metadata.get("kind", "error")),
                correlation_id=str(
                    metadata.get("correlation_id") or row.get("run_id") or "",
                ),
                sequence_no=int(row.get("sequence_no") or len(events) + 1),
                occurred_at_utc=str(row.get("timestamp_utc") or ""),
                payload=metadata,
                duration_ms=row.get("duration_ms"),
            ))
        return self.audit_events(events)


def _candidate_stage_pairs(kind: str, payload: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Map event kinds and payloads to (candidate_id, lifecycle stage) pairs."""
    pairs: list[tuple[str, str]] = []
    if kind == "candidates_generated":
        for item in _as_items(payload.get("kept")):
            pairs.append((str(item.get("candidate_id", "")), "proposed"))
    elif kind == "candidates_scored":
        for item in _as_items(payload.get("scored")):
            pairs.append((str(item.get("candidate_id", "")), "scored"))
    elif kind == "candidates_ranked":
        for item in _as_items(payload.get("ranked")):
            pairs.append((str(item.get("candidate_id", "")), "ranked"))
    elif kind == "hypotheses_published":
        for item in _as_items(payload.get("published")):
            pairs.append((str(item.get("candidate_id", "")), "published"))
    elif kind == "verification_completed":
        for item in _as_items(payload.get("verified")):
            pairs.append((str(item.get("candidate_id", "")), "verified"))
    return [pair for pair in pairs if pair[0]]


def _as_items(value: Any) -> list[dict[str, Any]]:
    """Return only list-of-dict payload values; counts and scalars are skipped."""
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _candidate_dispositions(payload: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Extract (candidate_id, disposition) from verification payloads."""
    pairs: list[tuple[str, str]] = []
    for item in _as_items(payload.get("verified")):
        candidate_id = str(item.get("candidate_id", ""))
        disposition = item.get("disposition")
        if candidate_id and disposition:
            pairs.append((candidate_id, str(disposition)))
    return pairs


def _candidate_evidence(payload: Mapping[str, Any]) -> list[tuple[str, tuple[str, ...]]]:
    """Extract (candidate_id, evidence_ids) from verification payloads."""
    pairs: list[tuple[str, tuple[str, ...]]] = []
    for item in _as_items(payload.get("verified")):
        candidate_id = str(item.get("candidate_id", ""))
        if candidate_id and item.get("evidence_ids"):
            pairs.append((candidate_id, tuple(map(str, item["evidence_ids"]))))
    return pairs


def _lifecycle_gaps(stages: Sequence[str]) -> list[str]:
    """Missing expected stages; hard-rejected candidates stop where they stopped."""
    known = set(stages)
    if "proposed" not in known:
        return []
    gaps: list[str] = []
    for stage in ("scored", "ranked"):
        if stage not in known:
            gaps.append(stage)
    return gaps


def _metadata_of(row: Mapping[str, Any]) -> dict[str, Any]:
    """Parse the metadata_json field of ore_logger event rows."""
    metadata = row.get("metadata") or row.get("metadata_json") or {}
    if isinstance(metadata, str):
        import json

        try:
            metadata = json.loads(metadata)
        except ValueError:
            metadata = {}
    return dict(metadata) if isinstance(metadata, dict) else {}


__all__ = ["AuditTrail", "CandidateAudit", "CandidateLifecycleAudit", "LIFECYCLE_STAGES"]
