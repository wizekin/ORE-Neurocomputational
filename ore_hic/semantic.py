"""Semantic interpretation and USG grounding (spec Phase B, Steps 8-13).

Deterministic, rule-based extraction of concepts, relations, attributes and
competing interpretations. Every extracted item cites the exact USG pattern or
rule that supports it, or is labeled as a hypothesis; competing interpretations
are preserved instead of collapsed (spec Step 13); semantic assertions use a
ranking model P(h | X, C) proportional to P(X | h, C) P(h | C), which ranks
interpretations but never proves them true (spec 3.1).
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .contracts import (
    ArtifactKind,
    CanonicalArtifact,
    Modality,
    SourceSpan,
    UnitDeclaration,
    USGReference,
)
from .segmentation import Segment, SegmentationResult
from .session import CompilationSession, InputRecord
from .usg_adapter import HICUSGAdapter

_WORD = re.compile(r"[A-Za-z][A-Za-z_-]*")
_NUMBER = re.compile(r"\b\d+(?:\.\d+)?\b")

_UNIT_LEXICON: dict[str, UnitDeclaration] = {
    "ms": UnitDeclaration("ms", "time"),
    "millisecond": UnitDeclaration("ms", "time"),
    "milliseconds": UnitDeclaration("ms", "time"),
    "s": UnitDeclaration("s", "time"),
    "second": UnitDeclaration("s", "time"),
    "seconds": UnitDeclaration("s", "time"),
    "mb": UnitDeclaration("MB", "memory"),
    "megabyte": UnitDeclaration("MB", "memory"),
    "gb": UnitDeclaration("GB", "memory"),
    "%": UnitDeclaration("percent", "ratio"),
    "count": UnitDeclaration("count", "cardinality"),
    "unit": UnitDeclaration("unit", "cardinality"),
    "units": UnitDeclaration("unit", "cardinality"),
}

_MODALITY_MARKERS: tuple[tuple[Modality, tuple[str, ...]], ...] = (
    (Modality.MUST, ("must", "has to", "required to", "cannot", "can't", "never", "always", "exactly one", "no more than", "not exceed", "at most", "at least")),
    (Modality.MAY, ("may", "can", "allowed to", "optionally")),
    (Modality.SHOULD, ("should", "preferably", "prefer", "ideally")),
    (Modality.USUALLY, ("usually", "typically", "in general", "normally")),
)


@dataclass(frozen=True, slots=True)
class ExtractedConcept:
    """A candidate concept linked to USG or explicitly unresolved (Step 8)."""

    concept_key: str
    surface: str
    usg_reference: USGReference | None
    span: SourceSpan
    confidence: float
    unresolved: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "concept_key": self.concept_key,
            "surface": self.surface,
            "usg_reference": self.usg_reference.to_dict() if self.usg_reference else None,
            "span": self.span.to_dict(),
            "confidence": self.confidence,
            "unresolved": self.unresolved,
        }


@dataclass(frozen=True, slots=True)
class ExtractedRelation:
    """A typed relation with argument roles and frame provenance (Step 9)."""

    relation_key: str
    predicate: str
    roles: Mapping[str, str]
    span: SourceSpan
    frame_id: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "relation_key": self.relation_key,
            "predicate": self.predicate,
            "roles": dict(sorted(self.roles.items())),
            "span": self.span.to_dict(),
            "frame_id": self.frame_id,
            "confidence": self.confidence,
        }


@dataclass(frozen=True, slots=True)
class ExtractedAttribute:
    """A property with declared type, unit, optionality and status (Step 10)."""

    attribute_key: str
    owner_key: str
    name: str
    value: str
    declared_type: str
    unit: UnitDeclaration | None
    optionality: str  # REQUIRED | OPTIONAL
    modality: Modality
    measurement_status: str  # MEASURED | ASSUMED | ESTIMATED | USER_DEFINED
    span: SourceSpan
    rule_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "attribute_key": self.attribute_key,
            "owner_key": self.owner_key,
            "name": self.name,
            "value": self.value,
            "declared_type": self.declared_type,
            "unit": self.unit.to_dict() if self.unit else None,
            "optionality": self.optionality,
            "modality": self.modality.value,
            "measurement_status": self.measurement_status,
            "span": self.span.to_dict(),
            "rule_id": self.rule_id,
        }


@dataclass(frozen=True, slots=True)
class InterpretationHypothesis:
    """One candidate reading of an ambiguous phrase (spec 3.1, Step 13)."""

    hypothesis_id: str
    reading: str
    usg_reference: USGReference | None
    likelihood: float  # P(X | h, C)
    prior: float  # P(h | C)
    supporting: tuple[str, ...]
    contradicting: tuple[str, ...]

    @property
    def posterior_weight(self) -> float:
        """Unnormalized P(h | X, C) proportional to likelihood times prior."""
        return self.likelihood * self.prior

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "reading": self.reading,
            "usg_reference": self.usg_reference.to_dict() if self.usg_reference else None,
            "likelihood": self.likelihood,
            "prior": self.prior,
            "posterior_weight": self.posterior_weight,
            "supporting": list(self.supporting),
            "contradicting": list(self.contradicting),
        }


@dataclass(frozen=True, slots=True)
class CompetingInterpretation:
    """A phrase with retained alternatives and their evidence (Step 13)."""

    phrase: str
    span: SourceSpan
    hypotheses: tuple[InterpretationHypothesis, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "phrase": self.phrase,
            "span": self.span.to_dict(),
            "hypotheses": [h.to_dict() for h in self.hypotheses],
        }


class SemanticGraph(CanonicalArtifact):
    """Canonical artifact: concepts, relations, attributes, interpretations."""

    kind = ArtifactKind.SEMANTIC_GRAPH

    def __init__(
        self,
        *,
        artifact_id: str,
        record_id: str,
        concepts: Iterable[ExtractedConcept],
        relations: Iterable[ExtractedRelation],
        attributes: Iterable[ExtractedAttribute],
        interpretations: Iterable[CompetingInterpretation],
        dangling_references: Iterable[str],
        usg_backend: str,
        version: int = 1,
    ) -> None:
        self.artifact_id = artifact_id
        self.record_id = record_id
        self.concepts = tuple(concepts)
        self.relations = tuple(relations)
        self.attributes = tuple(attributes)
        self.interpretations = tuple(interpretations)
        self.dangling_references = tuple(sorted(set(dangling_references)))
        self.usg_backend = usg_backend
        self.version = version

    def concept_by_key(self, key: str) -> ExtractedConcept | None:
        for concept in self.concepts:
            if concept.concept_key == key:
                return concept
        return None

    def resolved_concepts(self) -> tuple[ExtractedConcept, ...]:
        return tuple(c for c in self.concepts if not c.unresolved)

    def content_payload(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "record_id": self.record_id,
            "concepts": [c.to_dict() for c in self.concepts],
            "relations": [r.to_dict() for r in self.relations],
            "attributes": [a.to_dict() for a in self.attributes],
            "interpretations": [i.to_dict() for i in self.interpretations],
            "dangling_references": list(self.dangling_references),
            "usg_backend": self.usg_backend,
            "version": self.version,
        }


class SemanticInterpreter:
    """Extract and ground semantic structure from segmented input (Steps 8-13)."""

    def __init__(self, usg: HICUSGAdapter | None = None) -> None:
        self.usg = usg or HICUSGAdapter()

    # -- Step 8: concepts ----------------------------------------------------

    def extract_concepts(self, segments: Sequence[Segment]) -> tuple[ExtractedConcept, ...]:
        concepts: list[ExtractedConcept] = []
        seen: dict[str, int] = {}
        for segment in segments:
            for match in _WORD.finditer(segment.text):
                surface = match.group(0)
                candidate_forms = (surface, _singular(surface))
                for form in candidate_forms:
                    reference = self.usg.resolve_concept(form)
                    if reference is not None:
                        key = reference.concept_id
                        concepts.append(
                            ExtractedConcept(
                                concept_key=key,
                                surface=surface,
                                usg_reference=reference,
                                span=segment.span,
                                confidence=0.95 if form == surface else 0.85,
                            )
                        )
                        seen[key] = seen.get(key, 0) + 1
                        break
                else:
                    if surface[0].isupper() or surface in _NOMINAL_HINTS:
                        concepts.append(
                            ExtractedConcept(
                                concept_key=f"UNRESOLVED:{surface.casefold()}",
                                surface=surface,
                                usg_reference=None,
                                span=segment.span,
                                confidence=0.5,
                                unresolved=True,
                            )
                        )
        # Deduplicate by key keeping the first occurrence order.
        unique: dict[str, ExtractedConcept] = {}
        for concept in concepts:
            unique.setdefault(concept.concept_key, concept)
        return tuple(unique.values())

    # -- Step 9: relations ---------------------------------------------------

    def extract_relations(self, segments: Sequence[Segment]) -> tuple[ExtractedRelation, ...]:
        relations: list[ExtractedRelation] = []
        for index, segment in enumerate(segments, start=1):
            for frame in self.usg.registry.relation_frames.values():
                for pattern in frame.surface_patterns:
                    match = re.search(pattern, segment.text, re.IGNORECASE)
                    if match:
                        roles = {
                            role: match.groupdict().get(role, "").strip().casefold()
                            for role in frame.roles
                            if match.groupdict().get(role)
                        }
                        if not roles:
                            continue
                        relations.append(
                            ExtractedRelation(
                                relation_key=f"rel-{index:03d}-{frame.predicate}",
                                predicate=frame.predicate,
                                roles=roles,
                                span=segment.span,
                                frame_id=frame.frame_id,
                                confidence=0.85,
                            )
                        )
                        break
        return tuple(relations)

    # -- Step 10: attributes -------------------------------------------------

    def extract_attributes(self, segments: Sequence[Segment]) -> tuple[ExtractedAttribute, ...]:
        attributes: list[ExtractedAttribute] = []
        for index, segment in enumerate(segments, start=1):
            modality = self._modality(segment.text)
            for match in re.finditer(
                r"\b(capacity|load|cost|demand|weight|limit|threshold|score)\b(?:\s+(?:of|=|:)\s*([\w.]+))?",
                segment.text,
                re.IGNORECASE,
            ):
                name = match.group(1).casefold()
                raw_value = match.group(2)
                value, unit, status = self._value_unit(segment.text, name, raw_value)
                attributes.append(
                    ExtractedAttribute(
                        attribute_key=f"attr-{index:03d}-{name}",
                        owner_key="",
                        name=name,
                        value=value,
                        declared_type="float" if unit else "string",
                        unit=unit,
                        optionality="REQUIRED" if modality is Modality.MUST else "OPTIONAL",
                        modality=modality,
                        measurement_status=status,
                        span=segment.span,
                        rule_id="HIC-USG-R05",
                    )
                )
        return tuple(attributes)

    @staticmethod
    def _modality(text: str) -> Modality:
        lowered = text.casefold()
        for modality, markers in _MODALITY_MARKERS:
            if any(marker in lowered for marker in markers):
                return modality
        return Modality.DESCRIPTIVE

    @staticmethod
    def _value_unit(text: str, name: str, raw_value: str | None) -> tuple[str, UnitDeclaration | None, str]:
        if raw_value and _NUMBER.fullmatch(raw_value):
            return raw_value, UnitDeclaration("count", "cardinality"), "USER_DEFINED"
        window = text.casefold()
        for unit_word, unit in _UNIT_LEXICON.items():
            if re.search(rf"\b{re.escape(name)}\b[^.]*\b{re.escape(unit_word)}\b", window):
                return "unspecified", unit, "USER_DEFINED"
        if raw_value:
            return raw_value, None, "USER_DEFINED"
        return "unspecified", None, "ASSUMED"

    # -- Step 13: competing interpretations -----------------------------------

    def extract_competing_interpretations(
        self, segments: Sequence[Segment]
    ) -> tuple[CompetingInterpretation, ...]:
        """Retain alternative readings with evidence; never collapse them.

        One entry per ambiguous phrase (first occurrence wins); repetitions of
        the same phrase do not create duplicate ambiguities.
        """
        competing: list[CompetingInterpretation] = []
        seen_phrases: set[str] = set()
        for segment in segments:
            lowered = segment.text.casefold()
            if re.search(r"\bcapacity\b", lowered) and "capacity" not in seen_phrases:
                seen_phrases.add("capacity")
                competing.append(
                    CompetingInterpretation(
                        phrase="capacity",
                        span=segment.span,
                        hypotheses=(
                            InterpretationHypothesis(
                                hypothesis_id="cap-h1",
                                reading="capacity measured in task count",
                                usg_reference=self.usg.resolve_concept("capacity"),
                                likelihood=0.6,
                                prior=0.6,
                                supporting=("tasks are the counted work items",),
                                contradicting=("resource cost wording may imply units",),
                            ),
                            InterpretationHypothesis(
                                hypothesis_id="cap-h2",
                                reading="capacity measured in resource units",
                                usg_reference=self.usg.resolve_concept("capacity"),
                                likelihood=0.55,
                                prior=0.55,
                                supporting=("modules consume resources",),
                                contradicting=("no explicit resource unit stated",),
                            ),
                        ),
                    )
                )
        return tuple(competing)

    # -- full graph -----------------------------------------------------------

    def interpret(
        self,
        session: CompilationSession,
        record: InputRecord,
        segmentation: SegmentationResult,
    ) -> SemanticGraph:
        """Run Steps 8-13 and register the SemanticGraph artifact."""
        concepts = self.extract_concepts(segmentation.segments)
        relations = self.extract_relations(segmentation.segments)
        attributes = self.extract_attributes(segmentation.segments)
        interpretations = self.extract_competing_interpretations(segmentation.segments)
        dangling = tuple(
            c.surface for c in concepts if c.unresolved and c.concept_key not in {"UNRESOLVED:the", "UNRESOLVED:a"}
        )
        graph = SemanticGraph(
            artifact_id=session.next_artifact_id(ArtifactKind.SEMANTIC_GRAPH),
            record_id=record.artifact_id,
            concepts=concepts,
            relations=relations,
            attributes=attributes,
            interpretations=interpretations,
            dangling_references=dangling,
            usg_backend=self.usg.backend,
        )
        session.register(graph)
        session.ledger.record(
            stage="B",
            step_id="HIC08-13",
            description="Semantic interpretation with USG grounding and competing interpretations",
            inputs=[record, segmentation],
            outputs=[graph],
            rule_references={r.rule_id for r in self.usg.registry.rules.values()},
            usg_references=[c.usg_reference for c in concepts if c.usg_reference],
        )
        return graph


def _singular(word: str) -> str:
    lowered = word.casefold()
    if lowered.endswith("ies"):
        return lowered[:-3] + "y"
    if lowered.endswith("ses"):
        return lowered[:-2]
    if lowered.endswith("s") and not lowered.endswith("ss"):
        return lowered[:-1]
    return lowered


_NOMINAL_HINTS = frozenset({
    "algorithm", "system", "process", "workflow", "service", "engine", "scheduler",
    "optimizer", "solver", "manager", "coordinator", "adapter", "registry", "policy",
})
