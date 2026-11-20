"""Read-only import of USG semantic classes, predicate frames, and relations."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Mapping

from .contracts import USGImportExclusion
from .lexical import _canonical_json, _ore_record_id
from .policy import USGImportPolicy
from .validation import require_valid_knowledge_bundle


@dataclass(frozen=True, slots=True)
class USGSemanticClass:
    """A semantic class with its original payload preserved as canonical JSON."""

    class_id: str
    usg_record_id: str
    class_key: str
    definition: str
    lifecycle_state: str
    source_reference: str
    payload_json: str


@dataclass(frozen=True, slots=True)
class USGPredicateFrame:
    """A predicate frame with role declarations preserved without reinterpretation."""

    frame_id: str
    usg_record_id: str
    frame_key: str
    frame_version: str
    lifecycle_state: str
    source_reference: str
    roles_json: str
    payload_json: str


@dataclass(frozen=True, slots=True)
class USGKnowledgeRelation:
    """An explicit directed USG relation between namespaced ORE record IDs."""

    relation_id: str
    subject_id: str
    predicate: str
    object_id: str
    lifecycle_state: str
    evidence_ids: tuple[str, ...]
    context_ids: tuple[str, ...]
    source_reference: str


@dataclass(frozen=True, slots=True)
class USGStructureImportReport:
    """Counts and exclusions for semantic-structure imports."""

    bundle_id: str
    classes_created: int
    frames_created: int
    relations_created: int
    imported_record_ids: tuple[str, ...]
    imported_relation_ids: tuple[str, ...]
    excluded_records: tuple[USGImportExclusion, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class USGStructureResource:
    """Immutable semantic structure snapshot derived from one USG bundle."""

    classes: tuple[USGSemanticClass, ...]
    predicate_frames: tuple[USGPredicateFrame, ...]
    relations: tuple[USGKnowledgeRelation, ...]
    namespace: str
    bundle_id: str
    bundle_digest: str
    report: USGStructureImportReport


def _source_reference(
    namespace: str,
    bundle_id: str,
    record_id: str,
    state: str,
    source_ids: tuple[str, ...],
    evidence_ids: tuple[str, ...],
    context_ids: tuple[str, ...],
    collection: str,
) -> str:
    """Build a stable source locator for one structure or relation."""
    from urllib.parse import quote

    return (
        f"usg-knowledge://{quote(namespace, safe='')}/{quote(bundle_id, safe='')}/"
        f"{collection}/{quote(record_id, safe='')}?state={quote(state, safe='')}"
        f"&sources={','.join(quote(item, safe='') for item in source_ids)}"
        f"&evidence={','.join(quote(item, safe='') for item in evidence_ids)}"
        f"&contexts={','.join(quote(item, safe='') for item in context_ids)}"
    )


class USGSemanticStructureAdapter:
    """Import semantic classes, predicate frames, and directed relations."""

    def __init__(self, *, include_candidate_states: bool = False) -> None:
        """Create an adapter with candidate import disabled unless explicitly enabled."""
        self.policy = USGImportPolicy(include_candidate_states=include_candidate_states)

    def convert(self, bundle: Mapping[str, Any]) -> USGStructureResource:
        """Convert supported structural records while preserving unknown payload fields."""
        require_valid_knowledge_bundle(bundle)
        namespace = str(bundle["namespace"])
        bundle_id = str(bundle["bundle_id"])
        digest = sha256(_canonical_json(bundle)).hexdigest()
        sources = {item["id"]: item for item in bundle["sources"]}
        evidence = {item["id"]: item for item in bundle["evidence"]}
        records = {item["id"]: item for item in bundle["records"]}
        eligible_ids: dict[str, str] = {}
        imported: set[str] = set()
        excluded: list[USGImportExclusion] = []
        classes: list[USGSemanticClass] = []
        frames: list[USGPredicateFrame] = []
        supported_kinds = {
            "LEXICAL_FORM", "LEMMA", "LEXICAL_SENSE", "CONCEPT",
            "SEMANTIC_CLASS", "PREDICATE_FRAME",
        }

        for record in sorted(bundle["records"], key=lambda row: str(row["id"])):
            kind = str(record["kind"])
            if kind not in {"SEMANTIC_CLASS", "PREDICATE_FRAME"}:
                continue
            record_id = str(record["id"])
            state = str(record["state"])
            if not self.policy.state_enabled(state):
                excluded.append(USGImportExclusion(
                    record_id, kind, f"lifecycle_state_not_enabled:{state}",
                ))
                continue
            usable, reason, source_ids = self.policy.source_access(record, sources, evidence)
            if not usable:
                excluded.append(USGImportExclusion(record_id, kind, reason))
                continue
            payload = record["payload"]
            ore_id = _ore_record_id(namespace, bundle_id, record)
            eligible_ids[record_id] = ore_id
            common = _source_reference(
                namespace, bundle_id, record_id, state, source_ids,
                tuple(str(item) for item in record.get("evidence_ids", ())),
                tuple(str(item) for item in record.get("context_ids", ())),
                "records",
            )
            payload_json = _canonical_json(payload).decode("utf-8")
            if kind == "SEMANTIC_CLASS":
                classes.append(USGSemanticClass(
                    class_id=ore_id,
                    usg_record_id=record_id,
                    class_key=str(payload["class_key"]),
                    definition=str(payload["definition"]),
                    lifecycle_state=state,
                    source_reference=common,
                    payload_json=payload_json,
                ))
            else:
                frames.append(USGPredicateFrame(
                    frame_id=ore_id,
                    usg_record_id=record_id,
                    frame_key=str(payload["frame_key"]),
                    frame_version=str(payload["version"]),
                    lifecycle_state=state,
                    source_reference=common,
                    roles_json=_canonical_json(payload["roles"]).decode("utf-8"),
                    payload_json=payload_json,
                ))
            imported.add(record_id)

        # Build the shared ID index for every active, rights-permitted record
        # kind supported by the lexical, concept, and structure adapters. This
        # lets explicit USG relations link lexical senses/forms to concepts and
        # classes without inventing a relation type or rewriting source records.
        for record_id, record in records.items():
            if record.get("kind") in supported_kinds and record_id not in eligible_ids:
                state = str(record["state"])
                if self.policy.state_enabled(state):
                    usable, _, _ = self.policy.source_access(record, sources, evidence)
                    if usable:
                        eligible_ids[record_id] = _ore_record_id(namespace, bundle_id, record)

        relations: list[USGKnowledgeRelation] = []
        imported_relations: set[str] = set()
        for relation in sorted(bundle["relations"], key=lambda row: str(row["id"])):
            relation_id = str(relation["id"])
            subject = str(relation["subject_record_id"])
            object_ = str(relation["object_record_id"])
            if subject not in eligible_ids or object_ not in eligible_ids:
                excluded.append(USGImportExclusion(
                    relation_id, "RELATION",
                    "both_relation_endpoints_must_be_active_rights_permitted_supported_records",
                ))
                continue
            state = str(relation["state"])
            if not self.policy.state_enabled(state):
                excluded.append(USGImportExclusion(
                    relation_id, "RELATION", f"lifecycle_state_not_enabled:{state}",
                ))
                continue
            usable, reason, source_ids = self.policy.source_access(
                relation, sources, evidence,
            )
            if not usable:
                excluded.append(USGImportExclusion(relation_id, "RELATION", reason))
                continue
            scope = sha256(_canonical_json({
                "namespace": namespace, "bundle_id": bundle_id,
            })).hexdigest()
            from urllib.parse import quote

            stable_relation_id = f"usg-rel:{scope}:{quote(relation_id, safe='')}"
            evidence_ids = tuple(str(item) for item in relation["evidence_ids"])
            context_ids = tuple(str(item) for item in relation["context_ids"])
            relations.append(USGKnowledgeRelation(
                relation_id=stable_relation_id,
                subject_id=eligible_ids[subject],
                predicate=str(relation["predicate"]),
                object_id=eligible_ids[object_],
                lifecycle_state=state,
                evidence_ids=evidence_ids,
                context_ids=context_ids,
                source_reference=_source_reference(
                    namespace, bundle_id, relation_id, state, source_ids,
                    evidence_ids, context_ids, "relations",
                ),
            ))
            imported_relations.add(relation_id)

        classes.sort(key=lambda item: item.class_id)
        frames.sort(key=lambda item: item.frame_id)
        relations.sort(key=lambda item: item.relation_id)
        report = USGStructureImportReport(
            bundle_id=bundle_id,
            classes_created=len(classes),
            frames_created=len(frames),
            relations_created=len(relations),
            imported_record_ids=tuple(sorted(imported)),
            imported_relation_ids=tuple(sorted(imported_relations)),
            excluded_records=tuple(sorted(
                excluded, key=lambda item: (item.record_id, item.reason),
            )),
            warnings=(
                "Predicate-frame roles and class payloads are preserved as canonical JSON, not inferred into ORE execution semantics.",
                "Relations retain their declared predicate and direction; no inverse or transitive relations are generated.",
                "Relations are imported only when both endpoints are eligible concepts or semantic structures.",
                "USG lifecycle states and evidence references do not establish semantic truth.",
            ),
        )
        return USGStructureResource(
            classes=tuple(classes),
            predicate_frames=tuple(frames),
            relations=tuple(relations),
            namespace=namespace,
            bundle_id=bundle_id,
            bundle_digest=digest,
            report=report,
        )


__all__ = [
    "USGSemanticClass",
    "USGPredicateFrame",
    "USGKnowledgeRelation",
    "USGStructureImportReport",
    "USGStructureResource",
    "USGSemanticStructureAdapter",
]
