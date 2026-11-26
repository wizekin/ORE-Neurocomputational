"""Idea segmentation into claims and intentions, deliverable and context resolution.

Implements spec Steps 5-7: split the normalized idea into goals, descriptions,
examples, constraints, preferences, questions, facts and proposed actions while
preserving each fragment's source span; identify the requested deliverable; and
resolve project context, treating absent information as unknown rather than as
permission to invent modules or APIs.
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from .contracts import (
    ArtifactKind,
    CanonicalArtifact,
    DeliverableKind,
    FragmentKind,
    SourceSpan,
)
from .input import NormalizedView
from .session import CompilationSession, InputRecord

_FRAGMENT_RULES: tuple[tuple[FragmentKind, str, str], ...] = (
    (FragmentKind.QUESTION, r"^(what|why|how|when|where|which|who|is|are|can|could|should)\b.*\?$", "HIC-SEG-Q1"),
    (FragmentKind.GOAL, r"\b(i want|i need|we want|the goal is|build|create|design|implement|compute|optimize|maximi[sz]e|minimi[sz]e)\b", "HIC-SEG-G1"),
    (FragmentKind.CONSTRAINT, r"\b(must|has to|required|cannot|can't|only|exactly one|at most|at least|no more than|not exceed|never|always)\b", "HIC-SEG-C1"),
    (FragmentKind.PREFERENCE, r"\b(prefer|preferably|should|usually|better to|ideally|favor)\b", "HIC-SEG-P1"),
    (FragmentKind.EXAMPLE, r"\b(for example|for instance|e\.g\.|such as|say)\b", "HIC-SEG-E1"),
    (FragmentKind.FACT, r"\b(is|are|consists of|has|contains|exists|there are)\b", "HIC-SEG-F1"),
    (FragmentKind.ACTION, r"\b(assign|compute|check|validate|send|process|store|update|execute|transform)\b", "HIC-SEG-A1"),
)

_DELIVERABLE_RULES: tuple[tuple[DeliverableKind, tuple[str, ...], str], ...] = (
    (DeliverableKind.IMPLEMENTATION, ("implement", "code", "function", "program", "module", "software", "library"), "HIC-DEL-1"),
    (DeliverableKind.ALGORITHM, ("algorithm", "procedure", "method", "solver"), "HIC-DEL-2"),
    (DeliverableKind.PSEUDOCODE, ("pseudocode", "pseudo-code", "sketch"), "HIC-DEL-3"),
    (DeliverableKind.TEST_SUITE, ("test suite", "tests", "test cases"), "HIC-DEL-4"),
    (DeliverableKind.MATH_MODEL, ("mathematical model", "formali[sz]e", "equations", "formulation"), "HIC-DEL-5"),
    (DeliverableKind.EXPLANATION, ("explain", "explanation", "describe", "clarify"), "HIC-DEL-6"),
    (DeliverableKind.WORKFLOW, ("workflow", "pipeline", "process steps"), "HIC-DEL-7"),
)


@dataclass(frozen=True, slots=True)
class Segment:
    """One claim or intention fragment with preserved provenance (spec Step 5)."""

    segment_id: str
    kind: FragmentKind
    text: str
    span: SourceSpan
    rule_id: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "kind": self.kind.value,
            "text": self.text,
            "span": self.span.to_dict(),
            "rule_id": self.rule_id,
            "confidence": self.confidence,
        }


@dataclass(frozen=True, slots=True)
class ProjectContextFact:
    """One resolved contextual fact with explicit source and version (Step 7)."""

    key: str
    value: str
    source: str
    version: str
    status: str  # RESOLVED | UNKNOWN

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "source": self.source,
            "version": self.version,
            "status": self.status,
        }


class SegmentationResult(CanonicalArtifact):
    """Stage product for Phase A segmentation outputs (Steps 5-7).

    Not one of the twelve canonical artifacts of spec 2.2: it is an intermediate
    view that the ledger still tracks with full lineage and digests.
    """

    kind = "SEGMENTATION_VIEW"

    def __init__(
        self,
        *,
        artifact_id: str,
        record_id: str,
        segments: Iterable[Segment],
        deliverable: DeliverableKind,
        deliverable_rule: str,
        context_facts: Iterable[ProjectContextFact],
        version: int = 1,
    ) -> None:
        self.artifact_id = artifact_id
        self.record_id = record_id
        self.segments = tuple(segments)
        self.deliverable = deliverable
        self.deliverable_rule = deliverable_rule
        self.context_facts = tuple(context_facts)
        self.version = version

    def segments_of(self, kind: FragmentKind) -> tuple[Segment, ...]:
        return tuple(s for s in self.segments if s.kind is kind)

    def content_payload(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "record_id": self.record_id,
            "segments": [s.to_dict() for s in self.segments],
            "deliverable": self.deliverable.value,
            "deliverable_rule": self.deliverable_rule,
            "context_facts": [f.to_dict() for f in self.context_facts],
            "version": self.version,
        }


class IdeaSegmenter:
    """Segment an idea into claims and intentions (spec Steps 5-7)."""

    def __init__(self, known_context_keys: Iterable[str] = ()) -> None:
        self.known_context_keys = frozenset(known_context_keys)

    def segment(self, record: InputRecord, view: NormalizedView) -> tuple[Segment, ...]:
        """Step 5: split the normalized text and preserve source spans."""
        units = self._split_units(view.text)
        segments: list[Segment] = []
        for index, (text, norm_start, norm_end) in enumerate(units):
            for kind, pattern, rule_id in _FRAGMENT_RULES:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    span = self._to_source_span(record, view, norm_start, norm_end, text)
                    segments.append(
                        Segment(
                            segment_id=f"seg-{index + 1:03d}",
                            kind=kind,
                            text=text,
                            span=span,
                            rule_id=rule_id,
                            confidence=0.9 if kind in (FragmentKind.QUESTION, FragmentKind.GOAL, FragmentKind.CONSTRAINT) else 0.7,
                        )
                    )
                    break
            else:
                span = self._to_source_span(record, view, norm_start, norm_end, text)
                segments.append(
                    Segment(
                        segment_id=f"seg-{index + 1:03d}",
                        kind=FragmentKind.DESCRIPTION,
                        text=text,
                        span=span,
                        rule_id="HIC-SEG-D1",
                        confidence=0.6,
                    )
                )
        return tuple(segments)

    @staticmethod
    def _split_units(text: str) -> list[tuple[str, int, int]]:
        """Split into sentence-like units with normalized offsets."""
        units: list[tuple[str, int, int]] = []
        pos = 0
        for raw in re.split(r"(?<=[.;!?\n])\s+", text):
            if not raw.strip():
                pos += len(raw)
                continue
            start = text.index(raw, pos)
            units.append((raw.strip(" \t"), start, start + len(raw)))
            pos = start + len(raw)
        return units

    @staticmethod
    def _to_source_span(record: InputRecord, view: NormalizedView, start: int, end: int, text: str) -> SourceSpan:
        # The normalized view is derived; the authoritative span points into the
        # original InputRecord content so traceability never depends on a view.
        original = record.content
        probe = text.strip()
        found = original.find(probe[: min(len(probe), 32)])
        if found >= 0:
            return record.span(found, found + len(probe))
        return record.span(0, len(original))

    def identify_deliverable(self, segments: Sequence[Segment]) -> tuple[DeliverableKind, str]:
        """Step 6: determine what the user actually wants to receive."""
        scores: dict[DeliverableKind, int] = {}
        rule_by_kind: dict[DeliverableKind, str] = {}
        text_blob = " ".join(s.text for s in segments).lower()
        for kind, keywords, rule_id in _DELIVERABLE_RULES:
            score = sum(1 for kw in keywords if kw in text_blob)
            if score:
                scores[kind] = scores.get(kind, 0) + score
                rule_by_kind[kind] = rule_id
        if not scores:
            return DeliverableKind.UNDETERMINED, "HIC-DEL-0"
        requested = max(scores.items(), key=lambda kv: (kv[1], kv[0].value))
        kinds = [k for k, v in scores.items() if v == requested[1]]
        if len(kinds) > 1:
            return DeliverableKind.COMBINATION, "HIC-DEL-MULTI"
        return requested[0], rule_by_kind[requested[0]]

    def resolve_context(
        self,
        session: CompilationSession,
        context_facts: Iterable[tuple[str, str, str, str]] = (),
    ) -> tuple[ProjectContextFact, ...]:
        """Step 7: resolve project context; absent information stays UNKNOWN."""
        resolved: list[ProjectContextFact] = []
        supplied = {key: (value, source, version) for key, value, source, version in context_facts}
        for key, value in session.settings.context_facts.items():
            resolved.append(
                ProjectContextFact(key=key, value=value, source="session-settings", version="1", status="RESOLVED")
            )
        for key, (value, source, version) in supplied.items():
            resolved.append(
                ProjectContextFact(key=key, value=value, source=source, version=version, status="RESOLVED")
            )
        for key in self.known_context_keys:
            if key not in {f.key for f in resolved}:
                resolved.append(
                    ProjectContextFact(key=key, value="", source="none", version="unknown", status="UNKNOWN")
                )
        return tuple(resolved)

    def run(
        self,
        session: CompilationSession,
        record: InputRecord,
        view: NormalizedView,
        context_facts: Iterable[tuple[str, str, str]] = (),
    ) -> SegmentationResult:
        """Execute Steps 5-7 and register the segmentation artifact."""
        segments = self.segment(record, view)
        deliverable, rule = self.identify_deliverable(segments)
        if deliverable is DeliverableKind.UNDETERMINED:
            deliverable = DeliverableKind.ALGORITHM
            rule = "HIC-DEL-DEFAULT-ALGORITHM"
        context = self.resolve_context(session, context_facts)
        result = SegmentationResult(
            artifact_id=session.next_artifact_id(ArtifactKind.INPUT_RECORD),
            record_id=record.artifact_id,
            segments=segments,
            deliverable=deliverable,
            deliverable_rule=rule,
            context_facts=context,
        )
        session.ledger.record(
            stage="A",
            step_id="HIC05-07",
            description="Segment idea, identify deliverable, resolve project context",
            inputs=[record],
            outputs=[result],
            rule_references={s.rule_id for s in segments} | {rule},
        )
        return result
