"""Compact event summaries (specification Step 31).

Event summaries preserve event type, timestamp range, magnitude, confidence, source, and
evidence references while retaining a reference to the original window for audit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..contracts import EventSeverity, TypedEventRecord

__all__ = ("EventSummary", "summarize_events")


@dataclass(frozen=True, slots=True)
class EventSummary:
    """A compact, publishable summary of one or more typed event records."""

    summary_id: str
    event_type: str
    started_at: float
    ended_at: float
    n_records: int
    peak_magnitude: float | None
    mean_confidence: float | None
    severity: EventSeverity
    source_id: str
    evidence_refs: tuple[str, ...]
    window_reference: str
    cause_hypothesis: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize the event summary."""
        return {
            "summary_id": self.summary_id,
            "event_type": self.event_type,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "n_records": self.n_records,
            "peak_magnitude": self.peak_magnitude,
            "mean_confidence": self.mean_confidence,
            "severity": self.severity.value,
            "source_id": self.source_id,
            "evidence_refs": list(self.evidence_refs),
            "window_reference": self.window_reference,
            "cause_hypothesis": self.cause_hypothesis,
        }


def summarize_events(events: tuple[TypedEventRecord, ...], window_reference: str,
                     summary_id: str) -> EventSummary | None:
    """Fold typed event records of one type into a single compact summary."""
    if not events:
        return None
    magnitudes = [item.magnitude for item in events if item.magnitude is not None]
    confidences = [item.confidence for item in events if item.confidence is not None]
    severity = max((item.severity for item in events),
                   key=lambda item: (item is EventSeverity.CRITICAL,
                                     item is EventSeverity.WARNING))
    return EventSummary(
        summary_id=summary_id,
        event_type=events[0].event_type,
        started_at=min(item.interval[0] for item in events),
        ended_at=max(item.interval[1] for item in events),
        n_records=len(events),
        peak_magnitude=max(abs(item) for item in magnitudes) if magnitudes else None,
        mean_confidence=(sum(confidences) / len(confidences) if confidences else None),
        severity=severity,
        source_id=events[0].source_id,
        evidence_refs=tuple(sorted({ref for item in events for ref in item.evidence_refs})),
        window_reference=window_reference,
        cause_hypothesis=events[0].cause_hypothesis,
    )
