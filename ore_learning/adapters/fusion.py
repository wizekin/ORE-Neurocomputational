"""Semantic mapping and multimodal fusion (Point 14, Steps 21-22).

Modality outputs are mapped into USG-aligned concepts with uncertainty and modality
provenance attached; ambiguous evidence is never promoted to a certain fact.  Fusion
uses the documented reliability/relevance weighting s(c) = sum_j a_j r_j s_j(c) /
sum_j a_j r_j; when modalities disagree the competing observations and evidence are
retained, never averaged away.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..contracts import Modality, canonical_payload_digest
from ..functional import fusion_score

__all__ = ("FusedConcept", "FusionResult", "ModalityObservation", "SemanticMapper",
           "fuse_observations", "map_to_semantics", "UNKNOWN_USG_CONCEPT")


UNKNOWN_USG_CONCEPT = "usg:learning:unknown@1"


@dataclass(slots=True)
class ModalityObservation:
    """One typed observation produced by a perception specialist (Step 20)."""

    modality: Modality
    source_ref: str
    timestamp: float
    region: str = ""
    time_interval: tuple[float, float] | None = None
    concept_hint: str = ""
    support: float = 0.5
    confidence: float = 0.5
    payload_digest: str = ""
    adapter: str = ""

    def __post_init__(self) -> None:
        """Digest the observation content when not supplied."""
        if not self.payload_digest:
            self.payload_digest = canonical_payload_digest({
                "modality": self.modality.value, "source_ref": self.source_ref,
                "region": self.region, "concept_hint": self.concept_hint,
                "support": self.support})

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {"modality": self.modality.value, "source_ref": self.source_ref,
                "timestamp": self.timestamp, "region": self.region,
                "time_interval": list(self.time_interval) if self.time_interval else None,
                "concept_hint": self.concept_hint, "support": round(self.support, 6),
                "confidence": round(self.confidence, 6),
                "payload_digest": self.payload_digest, "adapter": self.adapter}


class SemanticMapper:
    """Step 21 -- map modality observations into USG-aligned concepts.

    The mapping is explicit: known hints resolve to declared USG ids, everything
    else maps to the explicit unknown concept instead of being silently coerced.
    """

    def __init__(self, mapping: Mapping[str, str] | None = None) -> None:
        """Configure the explicit hint -> USG id mapping."""
        self.mapping = dict(mapping or {
            "edge": "usg:perception:visual_edge@1",
            "surface": "usg:perception:visual_surface@1",
            "pattern": "usg:perception:visual_pattern@1",
            "speech": "usg:perception:auditory_speech@1",
            "tone": "usg:perception:auditory_tone@1",
            "positive": "usg:affect:sentiment_positive@1",
            "negative": "usg:affect:sentiment_negative@1",
            "neutral": "usg:affect:sentiment_neutral@1",
        })

    def map(self, hint: str) -> tuple[str, bool]:
        """Return (usg_id, known); unknown hints stay explicitly unknown."""
        usg_id = self.mapping.get(hint)
        if usg_id:
            return usg_id, True
        return UNKNOWN_USG_CONCEPT, False

    def map_observation(self, observation: ModalityObservation) -> str:
        """Map one observation's concept hint to its USG id."""
        usg_id, _ = self.map(observation.concept_hint)
        return usg_id


@dataclass(slots=True)
class FusedConcept:
    """One candidate concept with its fused score and supporting observations."""

    usg_id: str
    score: float
    supports: dict[str, float] = field(default_factory=dict)
    reliabilities: dict[str, float] = field(default_factory=dict)
    relevances: dict[str, float] = field(default_factory=dict)
    observations: list[ModalityObservation] = field(default_factory=list)
    known_concept: bool = True

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {"usg_id": self.usg_id, "score": round(self.score, 6),
                "supports": {key: round(value, 6)
                             for key, value in self.supports.items()},
                "reliabilities": self.reliabilities, "relevances": self.relevances,
                "observation_digests": [obs.payload_digest
                                        for obs in self.observations],
                "known_concept": self.known_concept}


@dataclass(slots=True)
class FusionResult:
    """The outcome of one fusion run (Step 22)."""

    ranked: list[FusedConcept]
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    note: str = ("scores are evidence strength under the documented fusion rule, "
                 "never proof; conflicting observations are retained")

    def top(self) -> FusedConcept | None:
        """The highest-scoring concept."""
        return self.ranked[0] if self.ranked else None

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {"ranked": [concept.to_dict() for concept in self.ranked],
                "conflicts": list(self.conflicts), "note": self.note}


def map_to_semantics(observations: Sequence[ModalityObservation],
                     mapper: SemanticMapper | None = None) -> list[ModalityObservation]:
    """Attach USG-aligned concept ids to observations (Step 21).

    Ambiguity is preserved: unknown hints map to the explicit unknown concept and the
    observation keeps its uncertainty rather than becoming a certain fact.
    """
    mapper = mapper or SemanticMapper()
    mapped = []
    for observation in observations:
        usg_id, known = mapper.map(observation.concept_hint)
        updated = ModalityObservation(
            modality=observation.modality, source_ref=observation.source_ref,
            timestamp=observation.timestamp, region=observation.region,
            time_interval=observation.time_interval,
            concept_hint=usg_id if known else UNKNOWN_USG_CONCEPT,
            support=observation.support, confidence=observation.confidence,
            payload_digest=observation.payload_digest, adapter=observation.adapter)
        mapped.append(updated)
    return mapped


def fuse_observations(observations: Sequence[ModalityObservation], *,
                      reliabilities: Mapping[str, float] | None = None,
                      relevances: Mapping[str, float] | None = None,
                      conflict_margin: float = 0.10) -> FusionResult:
    """Step 22 -- reliability-weighted fusion with explicit conflict retention.

    Each modality contributes its support for each candidate concept; the fused
    score applies the documented weighting.  When the top two concepts fall within
    ``conflict_margin`` of each other, or different modalities disagree about the
    top candidate, the conflict is recorded and both sets of competing observations
    remain visible in the result.
    """
    if not observations:
        return FusionResult(ranked=[], conflicts=[], note="no observations to fuse")
    reliability_map = dict(reliabilities or {
        Modality.TEXT.value: 0.7, Modality.IMAGE.value: 0.8, Modality.AUDIO.value: 0.6,
        Modality.VIDEO.value: 0.65, Modality.SPATIAL.value: 0.75,
        Modality.STRUCTURED.value: 0.7})
    relevance_map = dict(relevances or {})

    # group observations by candidate concept
    groups: dict[str, list[ModalityObservation]] = {}
    for observation in observations:
        groups.setdefault(observation.concept_hint, []).append(observation)

    fused: list[FusedConcept] = []
    for concept, members in groups.items():
        supports: dict[str, float] = {}
        rel: dict[str, float] = {}
        rlv: dict[str, float] = {}
        for observation in members:
            modality = observation.modality.value
            # a modality's support for a concept is the max over its observations
            supports[modality] = max(supports.get(modality, 0.0), observation.support)
            rel[modality] = min(1.0, max(0.0, reliability_map.get(modality, 0.5)
                                         * observation.confidence))
            rlv[modality] = relevance_map.get(modality, 1.0)
        score = fusion_score(list(supports.values()), list(rel.values()),
                             list(rlv.values()))
        known = not concept.startswith("usg:learning:unknown")
        fused.append(FusedConcept(usg_id=concept, score=score, supports=supports,
                                  reliabilities=rel, relevances=rlv,
                                  observations=list(members), known_concept=known))
    fused.sort(key=lambda concept: (-concept.score, concept.usg_id))

    conflicts: list[dict[str, Any]] = []
    if len(fused) >= 2 and fused[0].score - fused[1].score <= conflict_margin:
        conflicts.append({
            "kind": "score_margin",
            "candidates": [fused[0].usg_id, fused[1].usg_id],
            "scores": [round(fused[0].score, 6), round(fused[1].score, 6)],
            "retained_observations":
                [obs.payload_digest for obs in fused[0].observations + fused[1].observations],
            "rule": "retain competing observations; apply the documented fusion rule "
                    "or request more evidence"})
    # cross-modality disagreement on the top candidate
    top_concept = fused[0]
    for modality, support in top_concept.supports.items():
        for other in fused[1:]:
            other_support = other.supports.get(modality)
            if other_support is not None and other_support > support + conflict_margin:
                conflicts.append({
                    "kind": "modality_disagreement",
                    "modality": modality,
                    "preferred": other.usg_id,
                    "preferred_support": round(other_support, 6),
                    "fused_top": top_concept.usg_id,
                    "retained_observations": [obs.payload_digest
                                              for obs in other.observations],
                    "rule": "the disagreeing observation remains visible"})
    return FusionResult(ranked=fused, conflicts=conflicts)
