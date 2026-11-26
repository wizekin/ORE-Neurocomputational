"""USG adapter: concept, relation, and rule resolution for HIC (spec Phase B).

The adapter grounds HIC extractions in the Universal Semantic Grammar. It ships
a small, explicit starter registry of concepts, classes, predicate frames and
rules so the pipeline works standalone; when an ``ore_usg`` knowledge resource
is supplied, lookups are delegated to ``USGKnowledgeIndex`` and HIC-internal
matches are labeled as hypothesis mappings rather than authoritative facts.
Nothing here may silently promote an unresolved candidate into a USG concept.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .contracts import USGReference


class USGLookupError(Exception):
    """Raised when the adapter is misconfigured (never for unresolved words)."""


@dataclass(frozen=True, slots=True)
class USGConceptRecord:
    """One concept in the HIC starter registry with an explicit class path."""

    concept_id: str
    label: str
    parent_class: str
    definition: str
    aliases: tuple[str, ...] = ()
    lifecycle_state: str = "VALIDATED_FOR_PROFILE"

    def matches(self, surface: str) -> bool:
        normalized = surface.strip().casefold()
        return normalized in {self.label.casefold(), *(a.casefold() for a in self.aliases)}


@dataclass(frozen=True, slots=True)
class USGRelationFrame:
    """One predicate frame with declared roles (spec Step 9)."""

    frame_id: str
    predicate: str
    roles: tuple[str, ...]
    definition: str
    surface_patterns: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class USGRuleRecord:
    """One named, inspectable interpretation rule cited in provenance."""

    rule_id: str
    statement: str
    stage: str


def _singular(word: str) -> str:
    """Conservative English singularization used only as a lookup fallback."""
    lowered = word.casefold()
    if lowered.endswith("ies"):
        return lowered[:-3] + "y"
    if lowered.endswith("ses"):
        return lowered[:-2]
    if lowered.endswith("s") and not lowered.endswith("ss"):
        return lowered[:-1]
    return lowered


class StarterUSGRegistry:
    """The explicit, versioned HIC-internal USG starter registry."""

    REGISTRY_VERSION = "hic-usg-starter/0.1.0"

    def __init__(self) -> None:
        self.concepts: dict[str, USGConceptRecord] = {}
        self.relation_frames: dict[str, USGRelationFrame] = {}
        self.rules: dict[str, USGRuleRecord] = {}
        for record in _STARTER_CONCEPTS:
            self.concepts[record.concept_id] = record
        for frame in _STARTER_FRAMES:
            self.relation_frames[frame.predicate] = frame
        for rule in _STARTER_RULES:
            self.rules[rule.rule_id] = rule

    def resolve_concept(self, surface: str) -> USGConceptRecord | None:
        """Exact label/alias match, then a conservative singular-form fallback.

        Unresolved words remain unresolved (HIC-USG-R02); morphology is only a
        lookup aid, never evidence of semantic identity.
        """
        for form in (surface.strip(), _singular(surface.strip())):
            normalized = form.casefold()
            for record in self.concepts.values():
                if normalized in {record.label.casefold(), *(a.casefold() for a in record.aliases)}:
                    return record
        return None

    def resolve_frame(self, predicate: str) -> USGRelationFrame | None:
        return self.relation_frames.get(predicate)

    def class_path(self, concept_id: str) -> tuple[str, ...]:
        """Return the explicit inheritance chain from the concept to its roots."""
        chain: list[str] = []
        current: str | None = concept_id
        seen: set[str] = set()
        while current is not None and current not in seen:
            seen.add(current)
            record = self.concepts.get(current)
            if record is None:
                break
            chain.append(record.parent_class)
            current = record.parent_class if record.parent_class in self.concepts else None
        return tuple(chain)

    def compatible(self, child_id: str, parent_id: str) -> bool:
        """Check an explicit is-a link following USG class rules (spec Step 11)."""
        return parent_id in self.class_path(child_id)


_STARTER_CONCEPTS: tuple[USGConceptRecord, ...] = (
    USGConceptRecord("USG-TASK", "task", "USG-WORK-ITEM", "A unit of work to be performed or allocated."),
    USGConceptRecord("USG-WORK-ITEM", "work item", "USG-THING", "Any item of work; superclass of tasks and jobs."),
    USGConceptRecord("USG-MODULE", "module", "USG-RESOURCE", "A processing or computational unit that can receive work."),
    USGConceptRecord("USG-RESOURCE", "resource", "USG-THING", "An entity with capabilities and limited availability."),
    USGConceptRecord("USG-THING", "thing", "", "Root class of concrete entities."),
    USGConceptRecord("USG-CAPACITY", "capacity", "USG-QUANTITY", "The maximum amount a module can hold or process."),
    USGConceptRecord("USG-LOAD", "load", "USG-QUANTITY", "The currently committed amount on a module."),
    USGConceptRecord("USG-COST", "cost", "USG-QUANTITY", "Resource expenditure caused by an assignment."),
    USGConceptRecord("USG-DEMAND", "demand", "USG-QUANTITY", "The amount a task requires from a module."),
    USGConceptRecord("USG-QUANTITY", "quantity", "USG-PROPERTY", "A measurable, unit-bearing value."),
    USGConceptRecord("USG-PROPERTY", "property", "USG-ABSTRACT", "An attributed characteristic of an entity."),
    USGConceptRecord("USG-ABSTRACT", "abstract", "", "Root class of abstract concepts."),
    USGConceptRecord(
        "USG-ELIGIBILITY", "eligibility", "USG-PROPERTY",
        "A binary permission property stating whether a task may run on a module.",
        aliases=("eligible", "eligibility", "permitted"),
    ),
    USGConceptRecord(
        "USG-CAPABILITY", "capability", "USG-PROPERTY",
        "The suitability of a module for a task, often scored.",
        aliases=("suitability", "capability score"),
    ),
    USGConceptRecord("USG-ASSIGNMENT", "assignment", "USG-EVENT", "The act or result of assigning a task to a module."),
    USGConceptRecord("USG-EVENT", "event", "USG-ABSTRACT", "Something that happens or is performed."),
    USGConceptRecord("USG-OBJECTIVE", "objective", "USG-ABSTRACT", "The goal an algorithm should achieve."),
    USGConceptRecord("USG-PREFERENCE", "preference", "USG-OBJECTIVE", "A soft, non-mandatory wish about outcomes."),
    USGConceptRecord("USG-CONSTRAINT", "constraint", "USG-ABSTRACT", "A mandatory restriction on allowed solutions."),
    USGConceptRecord("USG-FAILURE", "failure", "USG-EVENT", "An outcome in which objectives are not met."),
    USGConceptRecord("USG-REPORT", "report", "USG-ARTIFACT", "A structured output describing results."),
    USGConceptRecord("USG-ARTIFACT", "artifact", "USG-THING", "A produced, inspectable object."),
    USGConceptRecord("USG-ITERATION", "iteration", "USG-EVENT", "A repeated computational step."),
    USGConceptRecord("USG-INPUT", "input", "USG-ARTIFACT", "Data entering a computation."),
    USGConceptRecord("USG-OUTPUT", "output", "USG-ARTIFACT", "Data leaving a computation."),
)

_STARTER_FRAMES: tuple[USGRelationFrame, ...] = (
    USGRelationFrame(
        "USG-FRAME-ASSIGNS", "assigns-to", ("agent", "patient", "target", "condition"),
        "Someone/something assigns a work item to a resource.",
        surface_patterns=(r"assign(?:s|ed)?\s+(?P<patient>[\w\s-]+?)\s+to\s+(?P<target>[\w\s-]+)",),
    ),
    USGRelationFrame(
        "USG-FRAME-DEPENDS", "depends-on", ("patient", "target"),
        "A work item or value depends on another.",
        surface_patterns=(r"(?P<patient>[\w\s-]+?)\s+depends\s+on\s+(?P<target>[\w\s-]+)",),
    ),
    USGRelationFrame(
        "USG-FRAME-MEASURES", "measures", ("agent", "patient"),
        "An entity measures a quantity.",
        surface_patterns=(r"(?P<agent>[\w\s-]+?)\s+measur(?:es|ed)\s+(?P<patient>[\w\s-]+)",),
    ),
    USGRelationFrame(
        "USG-FRAME-CONTAINS", "contains", ("agent", "patient"),
        "A container holds a contained item.",
        surface_patterns=(r"(?P<agent>[\w\s-]+?)\s+contains?\s+(?P<patient>[\w\s-]+)",),
    ),
    USGRelationFrame(
        "USG-FRAME-TRANSFORMS", "transforms", ("agent", "patient", "target"),
        "An agent transforms a patient into a target.",
        surface_patterns=(r"transforms?\s+(?P<patient>[\w\s-]+?)\s+into\s+(?P<target>[\w\s-]+)",),
    ),
    USGRelationFrame(
        "USG-FRAME-TRIGGERS", "triggers", ("agent", "patient"),
        "An event triggers another.",
        surface_patterns=(r"(?P<agent>[\w\s-]+?)\s+triggers?\s+(?P<patient>[\w\s-]+)",),
    ),
    USGRelationFrame(
        "USG-FRAME-VALIDATES", "validates", ("agent", "patient"),
        "An agent checks validity of a patient.",
        surface_patterns=(r"(?P<agent>[\w\s-]+?)\s+validat(?:es|ed)\s+(?P<patient>[\w\s-]+)",),
    ),
    USGRelationFrame(
        "USG-FRAME-CAUSES", "causes", ("agent", "patient"),
        "A causes B.",
        surface_patterns=(r"(?P<agent>[\w\s-]+?)\s+caus(?:es|ed)\s+(?P<patient>[\w\s-]+)",),
    ),
)

_STARTER_RULES: tuple[USGRuleRecord, ...] = (
    USGRuleRecord("HIC-USG-R01", "A surface form resolves to a USG concept only by exact label or alias match.", "B"),
    USGRuleRecord("HIC-USG-R02", "Unresolved surface forms become unresolved candidates, never new concepts.", "B"),
    USGRuleRecord("HIC-USG-R03", "Relation extraction must cite the USG predicate frame and its roles.", "B"),
    USGRuleRecord("HIC-USG-R04", "Competing interpretations are retained with supporting and contradicting evidence.", "B"),
    USGRuleRecord("HIC-USG-R05", "Attributes carry declared type, unit, optionality, and measurement status.", "B"),
    USGRuleRecord("HIC-USG-R06", "Class compatibility follows the explicit USG class hierarchy.", "B"),
    USGRuleRecord("HIC-USG-R07", "Matches against an external ore_usg resource are hypothesis mappings until reviewed.", "B"),
)


class HICUSGAdapter:
    """Resolve and validate semantic concepts, relations, rules, and evidence."""

    def __init__(self, registry: StarterUSGRegistry | None = None, external_index: Any | None = None) -> None:
        self.registry = registry or StarterUSGRegistry()
        self.external_index = external_index
        if external_index is not None and not hasattr(external_index, "lookup"):
            raise USGLookupError("external_index must expose lookup(query, ...) like USGKnowledgeIndex")

    @property
    def backend(self) -> str:
        if self.external_index is not None:
            return f"ore_usg+{StarterUSGRegistry.REGISTRY_VERSION}"
        return StarterUSGRegistry.REGISTRY_VERSION

    def resolve_concept(self, surface: str) -> USGReference | None:
        """Resolve a surface form; unresolved forms return ``None`` (Step 8)."""
        record = self.registry.resolve_concept(surface)
        if record is not None:
            return USGReference(
                concept_id=record.concept_id,
                rule_id="HIC-USG-R01",
                mapping_type="EQUIVALENT",
                lifecycle_state=record.lifecycle_state,
            )
        if self.external_index is not None:
            try:
                result = self.external_index.lookup(surface, kind="SEMANTIC_CONCEPT")
            except Exception:  # noqa: BLE001 - external index failures degrade to unresolved
                result = None
            if result is not None and getattr(result, "items", ()):
                top = result.items[0]
                return USGReference(
                    concept_id=str(top.record_id),
                    rule_id="HIC-USG-R07",
                    mapping_type="RELATED_TO",
                    lifecycle_state=str(top.lifecycle_state or "CANDIDATE"),
                )
        return None

    def resolve_frame(self, predicate: str) -> USGRelationFrame | None:
        return self.registry.resolve_frame(predicate)

    def compatible(self, child_concept_id: str, parent_concept_id: str) -> bool:
        return self.registry.compatible(child_concept_id, parent_concept_id)

    def class_path(self, concept_id: str) -> tuple[str, ...]:
        return self.registry.class_path(concept_id)

    def rule(self, rule_id: str) -> USGRuleRecord:
        if rule_id not in self.registry.rules:
            raise USGLookupError(f"unknown USG rule {rule_id}")
        return self.registry.rules[rule_id]

    def rules_payload(self) -> list[dict[str, str]]:
        return [
            {"rule_id": r.rule_id, "statement": r.statement, "stage": r.stage}
            for r in sorted(self.registry.rules.values(), key=lambda r: r.rule_id)
        ]
