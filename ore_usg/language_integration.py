"""Phase 3 read-only integration of lexical, semantic, contextual and evidence data."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Mapping
from urllib.parse import quote

from ore_semantic.language_knowledge import (
    ContextAssignment,
    ContextualRecord,
    EvidenceAttachment,
    EvidenceRecord,
    LexicalConceptLink,
    LexicalConceptLinkGroup,
    MorphologicalFormLink,
    SourceProvenance,
)

from .contracts import USGImportExclusion
from .lexical import _canonical_json, _ore_record_id
from .policy import USGImportPolicy
from .semantic_knowledge import USGSemanticKnowledgeResource
from .contracts import USGLexiconResource
from .validation import require_valid_knowledge_bundle


_FORM_TO_LEMMA = frozenset({
    "FORM_OF", "INFLECTED_FORM_OF", "SURFACE_FORM_OF",
    "MORPHOLOGICAL_FORM_OF", "DERIVED_FORM_OF",
})
_LEMMA_TO_FORM = frozenset({
    "HAS_FORM", "HAS_INFLECTED_FORM", "HAS_SURFACE_FORM",
    "HAS_MORPHOLOGICAL_FORM",
})
_LEXICAL_KINDS = frozenset({"LEMMA", "LEXICAL_FORM", "LEXICAL_SENSE"})
_SEMANTIC_KINDS = frozenset({"CONCEPT", "LEXICAL_SENSE"})


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _aux_id(prefix: str, namespace: str, bundle_id: str, source_id: str) -> str:
    scope = sha256(_canonical_json({"namespace": namespace, "bundle_id": bundle_id})).hexdigest()
    return f"{prefix}:{scope}:{quote(source_id, safe='')}"


def _source_reference(namespace: str, bundle_id: str, collection: str, source_id: str,
                      state: str, source_ids: tuple[str, ...], evidence_ids: tuple[str, ...],
                      context_ids: tuple[str, ...]) -> str:
    return (
        f"usg-knowledge://{quote(namespace, safe='')}/{quote(bundle_id, safe='')}/"
        f"{collection}/{quote(source_id, safe='')}?state={quote(state, safe='')}"
        f"&sources={','.join(quote(x, safe='') for x in source_ids)}"
        f"&evidence={','.join(quote(x, safe='') for x in evidence_ids)}"
        f"&contexts={','.join(quote(x, safe='') for x in context_ids)}"
    )


def _assertion_status(
    lifecycle_state: str,
    evidence_ids: tuple[str, ...],
    evidence_by_id: Mapping[str, Mapping[str, Any]],
    *,
    candidate_enabled: bool,
) -> tuple[str, tuple[str, ...]]:
    statuses = tuple(sorted({
        str(evidence_by_id[item].get("support_status", "UNASSESSED"))
        for item in evidence_ids if item in evidence_by_id
    }))
    if lifecycle_state == "UNRESOLVED":
        return "UNRESOLVED", statuses
    if lifecycle_state == "CANDIDATE":
        return ("CANDIDATE" if candidate_enabled else "UNRESOLVED"), statuses
    if "SUPPORTS" in statuses and "CONTRADICTS" in statuses:
        return "CONFLICTED", statuses
    if "CONTRADICTS" in statuses:
        return "CONTRADICTED", statuses
    if "NEUTRAL" in statuses and "SUPPORTS" not in statuses:
        return "NEUTRAL", statuses
    if "UNASSESSED" in statuses or not statuses:
        return "UNASSESSED", statuses
    return "LINKED", statuses


@dataclass(frozen=True, slots=True)
class USGLanguageIntegrationReport:
    """Import summary and explanations for missing or uncertain cross-resource links."""

    bundle_id: str
    lexical_concept_links_created: int
    lexical_concept_link_groups_created: int
    missing_lexical_concept_links: int
    ambiguous_lexical_concept_groups: int
    conflicting_lexical_concept_links: int
    morphology_links_created: int
    missing_morphology_links: int
    ambiguous_morphology_links: int
    contexts_created: int
    context_assignments_created: int
    evidence_records_created: int
    evidence_attachments_created: int
    source_records_created: int
    imported_record_ids: tuple[str, ...]
    imported_relation_ids: tuple[str, ...]
    exclusions: tuple[USGImportExclusion, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class USGLanguageIntegrationResource:
    """Typed Phase 3 cross-resource links and source-backed context/evidence snapshot."""

    lexical_concept_links: tuple[LexicalConceptLink, ...]
    lexical_concept_groups: tuple[LexicalConceptLinkGroup, ...]
    morphology_links: tuple[MorphologicalFormLink, ...]
    contexts: tuple[ContextualRecord, ...]
    context_assignments: tuple[ContextAssignment, ...]
    evidence: tuple[EvidenceRecord, ...]
    evidence_attachments: tuple[EvidenceAttachment, ...]
    sources: tuple[SourceProvenance, ...]
    namespace: str
    bundle_id: str
    bundle_digest: str
    report: USGLanguageIntegrationReport

    def __post_init__(self) -> None:
        if not self.namespace.strip() or not self.bundle_id.strip():
            raise ValueError("namespace and bundle_id must be non-empty")
        if len(self.bundle_digest) != 64 or any(x not in "0123456789abcdef" for x in self.bundle_digest):
            raise ValueError("bundle_digest must be a lowercase SHA-256 digest")
        context_ids = {item.context_id for item in self.contexts}
        evidence_ids = {item.evidence_id for item in self.evidence}
        source_ids = {item.source_id for item in self.sources}
        if any(item.context_id not in context_ids for item in self.context_assignments):
            raise ValueError("context assignment references an absent context")
        if any(item.evidence_id not in evidence_ids for item in self.evidence_attachments):
            raise ValueError("evidence attachment references absent evidence")
        if any(item.source_id not in source_ids for item in self.evidence):
            raise ValueError("evidence record references an absent source")

    def contexts_for(self, record_id: str) -> tuple[ContextualRecord, ...]:
        """Return explicit context records attached to an exact ORE or USG record ID."""
        ids = {
            item.context_id for item in self.context_assignments
            if item.target_id == record_id or item.source_record_id == record_id
        }
        return tuple(item for item in self.contexts if item.context_id in ids)

    def evidence_for(self, record_id: str) -> tuple[EvidenceRecord, ...]:
        """Return attached evidence summaries without promoting them to verified facts."""
        ids = {
            item.evidence_id for item in self.evidence_attachments
            if item.target_id == record_id or item.source_record_id == record_id
        }
        return tuple(item for item in self.evidence if item.evidence_id in ids)

    def source_by_id(self, source_id: str) -> SourceProvenance | None:
        return next((item for item in self.sources if item.source_id == source_id), None)


class USGLanguageIntegrationAdapter:
    """Link the Phase 1/2 lexical and semantic resources using only explicit USG evidence."""

    def __init__(self, *, include_candidate_states: bool = False) -> None:
        self.policy = USGImportPolicy(include_candidate_states=include_candidate_states)

    def convert(
        self,
        bundle: Mapping[str, Any],
        lexical: USGLexiconResource,
        semantic: USGSemanticKnowledgeResource,
    ) -> USGLanguageIntegrationResource:
        require_valid_knowledge_bundle(bundle)
        namespace, bundle_id = str(bundle["namespace"]), str(bundle["bundle_id"])
        digest = sha256(_canonical_json(bundle)).hexdigest()
        identity = (namespace, bundle_id, digest)
        if identity != (lexical.namespace, lexical.bundle_id, lexical.bundle_digest):
            raise ValueError("lexical resource does not match the Phase 3 input bundle")
        if identity != (semantic.namespace, semantic.bundle_id, semantic.bundle_digest):
            raise ValueError("semantic resource does not match the Phase 3 input bundle")

        records = {str(item["id"]): item for item in bundle["records"]}
        sources_raw = {str(item["id"]): item for item in bundle["sources"]}
        evidence_raw = {str(item["id"]): item for item in bundle["evidence"]}
        context_raw = {str(item["id"]): item for item in bundle["contexts"]}
        exclusions: list[USGImportExclusion] = []
        imported_records: set[str] = set()
        imported_relations: set[str] = set()

        sources = tuple(
            SourceProvenance(
                source_id=str(item["id"]),
                provider=str(item["provider"]),
                source_type=str(item["source_type"]),
                rights_status=str(item["rights_status"]),
                redistribution=str(item["redistribution"]),
                attribution=str(item["attribution"]),
                usable_for_local_import=str(item["rights_status"]) in self.policy.allowed_local_rights,
                license_id=item.get("license_id"),
                source_locator=item.get("source_locator"),
                retrieved_at=str(item["retrieved_at"]),
                content_digest=item.get("content_digest"),
            )
            for item in sorted(bundle["sources"], key=lambda row: str(row["id"]))
        )
        source_models = {item.source_id: item for item in sources}

        evidence_records = tuple(
            EvidenceRecord(
                evidence_id=str(item["id"]),
                source_id=str(item["source_id"]),
                evidence_type=str(item["evidence_type"]),
                support_status=str(item["support_status"]),
                locator=item.get("locator"),
                text_digest=item.get("text_digest"),
                captured_at=str(item["captured_at"]),
                transformation_ids=tuple(str(x) for x in item["transformation_ids"]),
                source_rights_status=source_models[str(item["source_id"])].rights_status,
                source_redistribution=source_models[str(item["source_id"])].redistribution,
                source_license_id=source_models[str(item["source_id"])].license_id,
                usable_for_local_reasoning=source_models[str(item["source_id"])].usable_for_local_import,
                verified_fact=False,
                interpretation="EVIDENCE_SUMMARY_NOT_VERIFIED_FACT",
            )
            for item in sorted(bundle["evidence"], key=lambda row: str(row["id"]))
        )
        evidence_models = {item.evidence_id: item for item in evidence_records}

        # Record IDs are indexed only after lifecycle and source-rights checks.
        ore_ids: dict[str, str] = {}
        source_ids_by_record: dict[str, tuple[str, ...]] = {}
        for raw_id, row in sorted(records.items()):
            if str(row["kind"]) not in {
                "LEXICAL_FORM", "LEMMA", "LEXICAL_SENSE", "CONCEPT",
                "SEMANTIC_CLASS", "PREDICATE_FRAME",
            }:
                continue
            state = str(row["state"])
            if not (self.policy.state_enabled(state) or state == "UNRESOLVED"):
                continue
            usable, reason, resolved_source_ids = self.policy.source_access(row, sources_raw, evidence_raw)
            if not usable:
                exclusions.append(USGImportExclusion(raw_id, str(row["kind"]), reason))
                continue
            ore_ids[raw_id] = _ore_record_id(namespace, bundle_id, row)
            source_ids_by_record[raw_id] = resolved_source_ids

        # Source records and context_ids establish context assignments; contexts
        # remain categorical metadata and are never converted to fake activation scores.
        context_members: dict[str, set[str]] = {context_id: set() for context_id in context_raw}
        assignments: list[ContextAssignment] = []
        for raw_id, row in sorted(records.items()):
            if raw_id not in ore_ids:
                continue
            record_contexts = tuple(str(x) for x in row.get("context_ids", ()))
            for context_id in record_contexts:
                if context_id not in context_raw:
                    continue
                context_members.setdefault(context_id, set()).add(raw_id)
                assignments.append(ContextAssignment(
                    context_id=context_id,
                    target_id=ore_ids[raw_id],
                    target_kind=str(row["kind"]),
                    source_record_id=raw_id,
                ))
            imported_records.add(raw_id)

        contexts = tuple(
            ContextualRecord(
                context_id=context_id,
                context_type=str(row["context_type"]),
                value=str(row["value"]),
                language_profile=row.get("language_profile"),
                scope=row.get("scope"),
                source_record_ids=tuple(sorted(context_members.get(context_id, set()))),
            )
            for context_id, row in sorted(context_raw.items())
        )

        evidence_attachments: list[EvidenceAttachment] = []
        def attach_evidence(target_id: str, target_kind: str, raw_source_id: str,
                            evidence_ids: tuple[str, ...]) -> None:
            for evidence_id in evidence_ids:
                if evidence_id in evidence_models:
                    evidence_attachments.append(EvidenceAttachment(
                        evidence_id=evidence_id,
                        target_id=target_id,
                        target_kind=target_kind,
                        source_record_id=raw_source_id,
                    ))

        for raw_id, row in sorted(records.items()):
            if raw_id not in ore_ids:
                continue
            attach_evidence(
                ore_ids[raw_id], str(row["kind"]), raw_id,
                tuple(str(x) for x in row.get("evidence_ids", ())),
            )

        # Build ORE lookup indexes for imported lemmas/forms and exact sense meanings.
        entry_by_source_id: dict[str, Any] = {}
        meaning_by_source_id: dict[str, tuple[str, Any]] = {}
        for raw_id, row in records.items():
            if raw_id not in ore_ids:
                continue
            if row["kind"] in {"LEMMA", "LEXICAL_FORM"}:
                entry = next((item for item in lexical.entries if item.entry_id == ore_ids[raw_id]), None)
                if entry is not None:
                    entry_by_source_id[raw_id] = entry
            if row["kind"] == "LEXICAL_SENSE":
                meaning_id = ore_ids[raw_id]
                for entry in lexical.entries:
                    meaning = next((item for item in entry.meanings if item.meaning_id == meaning_id), None)
                    if meaning is not None:
                        meaning_by_source_id[raw_id] = (entry.entry_id, meaning)
                        break
        concept_by_source_id = {
            item.usg_record_id: item for item in semantic.concept_records
        }

        link_rows: list[LexicalConceptLink] = []
        links_by_entry: dict[str, list[LexicalConceptLink]] = {}
        links_by_meaning: dict[str, list[LexicalConceptLink]] = {}

        def add_link(
            *, lexical_source_id: str, semantic_source_id: str,
            predicate: str, relationship_kind: str, source_kind: str,
            lifecycle_state: str, evidence_ids: tuple[str, ...],
            context_ids: tuple[str, ...], source_reference: str,
            link_id: str,
            source_subject_record_id: str | None = None,
            source_object_record_id: str | None = None,
        ) -> None:
            lexical_row = records[lexical_source_id]
            lexical_kind = str(lexical_row["kind"])
            if lexical_kind == "LEXICAL_SENSE":
                owner = meaning_by_source_id.get(lexical_source_id)
                if owner is None:
                    exclusions.append(USGImportExclusion(
                        link_id, "LEXICAL_CONCEPT_LINK", "lexical_sense_meaning_not_imported",
                    ))
                    return
                entry_id, meaning = owner
                meaning_id = meaning.meaning_id
            else:
                entry = entry_by_source_id.get(lexical_source_id)
                if entry is None:
                    exclusions.append(USGImportExclusion(
                        link_id, "LEXICAL_CONCEPT_LINK", "lexical_entry_not_imported",
                    ))
                    return
                entry_id, meaning_id = entry.entry_id, None
            concept_record = concept_by_source_id.get(semantic_source_id)
            if concept_record is None:
                exclusions.append(USGImportExclusion(
                    link_id, "LEXICAL_CONCEPT_LINK", "semantic_concept_not_imported",
                ))
                return
            status, evidence_statuses = _assertion_status(
                lifecycle_state, evidence_ids, evidence_raw,
                candidate_enabled=self.policy.include_candidate_states,
            )
            link = LexicalConceptLink(
                link_id=link_id,
                lexical_entry_id=entry_id,
                lexical_meaning_id=meaning_id,
                semantic_concept_id=concept_record.concept.concept_id,
                source_lexical_record_id=lexical_source_id,
                source_concept_record_id=semantic_source_id,
                predicate=predicate,
                relationship_kind=relationship_kind,
                direction="DIRECTED",
                status=status,
                lifecycle_state=lifecycle_state,
                evidence_ids=evidence_ids,
                evidence_statuses=evidence_statuses,
                context_ids=context_ids,
                source_reference=source_reference,
                source_kind=source_kind,
                source_subject_record_id=source_subject_record_id or lexical_source_id,
                source_object_record_id=source_object_record_id or semantic_source_id,
            )
            link_rows.append(link)
            links_by_entry.setdefault(entry_id, []).append(link)
            if meaning_id:
                links_by_meaning.setdefault(meaning_id, []).append(link)

        # Each imported lexical sense and the ORE meaning generated from that same
        # source record share the same version-scoped identity. This is a lossless
        # representation link, not an inferred lexical or synonym relationship.
        for raw_id, (entry_id, meaning) in sorted(meaning_by_source_id.items()):
            concept_record = concept_by_source_id.get(raw_id)
            if concept_record is None:
                continue
            row = records[raw_id]
            evidence_ids = tuple(str(x) for x in row.get("evidence_ids", ()))
            context_ids = tuple(str(x) for x in row.get("context_ids", ()))
            add_link(
                lexical_source_id=raw_id,
                semantic_source_id=raw_id,
                predicate="SENSE_IDENTITY",
                relationship_kind="LEXICAL_MEANING_IDENTITY",
                source_kind="SOURCE_RECORD_IDENTITY",
                lifecycle_state=str(row["state"]),
                evidence_ids=evidence_ids,
                context_ids=context_ids,
                source_reference=_source_reference(
                    namespace, bundle_id, "records", raw_id, str(row["state"]),
                    source_ids_by_record[raw_id], evidence_ids, context_ids,
                ),
                link_id=f"usg-link:{quote(raw_id, safe='')}:sense-identity",
                source_subject_record_id=raw_id,
                source_object_record_id=raw_id,
            )

        # Explicit relation/mapping rows create further lexical-to-concept links
        # only when one endpoint is a lexical record and the other a semantic concept.
        relation_specs: list[tuple[Any, ...]] = []
        for row in sorted(bundle["relations"], key=lambda x: str(x["id"])):
            raw_id = str(row["id"])
            subject = str(row["subject_record_id"])
            object_ = str(row["object_record_id"])
            if subject not in ore_ids or object_ not in ore_ids:
                continue
            if not self.policy.state_enabled(str(row["state"])) and str(row["state"]) != "UNRESOLVED":
                continue
            subject_kind, object_kind = str(records[subject]["kind"]), str(records[object_]["kind"])
            if subject_kind in _LEXICAL_KINDS and object_kind in _SEMANTIC_KINDS:
                lexical_id, semantic_id = subject, object_
            elif object_kind in _LEXICAL_KINDS and subject_kind in _SEMANTIC_KINDS:
                lexical_id, semantic_id = object_, subject
            else:
                continue
            usable, reason, relation_source_ids = self.policy.source_access(row, sources_raw, evidence_raw)
            if not usable:
                exclusions.append(USGImportExclusion(raw_id, "LEXICAL_CONCEPT_LINK", reason))
                continue
            evidence_ids = tuple(str(x) for x in row["evidence_ids"])
            context_ids = tuple(str(x) for x in row["context_ids"])
            relation_specs.append((
                raw_id, lexical_id, semantic_id, subject, object_, str(row["predicate"]), "EXPLICIT_RELATION",
                str(row["state"]), evidence_ids, context_ids,
                _source_reference(namespace, bundle_id, "relations", raw_id, str(row["state"]),
                                  relation_source_ids, evidence_ids, context_ids),
            ))
            imported_relations.add(raw_id)

        for row in sorted(bundle["mappings"], key=lambda x: str(x["id"])):
            raw_id = str(row["id"])
            subject = str(row["source_record_id"])
            object_ = str(row["target_record_id"])
            if subject not in ore_ids or object_ not in ore_ids:
                continue
            if not self.policy.state_enabled(str(row["state"])) and str(row["state"]) != "UNRESOLVED":
                continue
            subject_kind, object_kind = str(records[subject]["kind"]), str(records[object_]["kind"])
            if subject_kind in _LEXICAL_KINDS and object_kind in _SEMANTIC_KINDS:
                lexical_id, semantic_id = subject, object_
            elif object_kind in _LEXICAL_KINDS and subject_kind in _SEMANTIC_KINDS:
                lexical_id, semantic_id = object_, subject
            else:
                continue
            if not (subject_kind in {"LEXICAL_SENSE"} or object_kind in {"LEXICAL_SENSE"}):
                if subject_kind != "LEMMA" and object_kind != "LEMMA":
                    continue
            usable, reason, relation_source_ids = self.policy.source_access(row, sources_raw, evidence_raw)
            if not usable:
                exclusions.append(USGImportExclusion(raw_id, "LEXICAL_CONCEPT_LINK", reason))
                continue
            evidence_ids = tuple(str(x) for x in row["evidence_ids"])
            context_ids = tuple(str(x) for x in row["context_ids"])
            relation_specs.append((
                raw_id, lexical_id, semantic_id, subject, object_, str(row["mapping_type"]), "EXPLICIT_MAPPING",
                str(row["state"]), evidence_ids, context_ids,
                _source_reference(namespace, bundle_id, "mappings", raw_id, str(row["state"]),
                                  relation_source_ids, evidence_ids, context_ids),
            ))
            imported_relations.add(raw_id)

        for raw_id, lexical_id, semantic_id, source_subject_id, source_object_id, predicate, source_kind, state, evidence_ids, context_ids, source_ref in relation_specs:
            add_link(
                lexical_source_id=lexical_id,
                semantic_source_id=semantic_id,
                predicate=predicate,
                relationship_kind=predicate,
                source_kind=source_kind,
                lifecycle_state=state,
                evidence_ids=evidence_ids,
                context_ids=context_ids,
                source_reference=source_ref,
                link_id=f"usg-link:{quote(raw_id, safe='')}:{quote(lexical_id, safe='')}",
                source_subject_record_id=source_subject_id,
                source_object_record_id=source_object_id,
            )

        # Missing links are explicit abstentions, not guesses. Candidate concepts
        # are never collapsed when entry/meaning-level alternatives are grouped.
        for raw_id, entry in entry_by_source_id.items():
            if not links_by_entry.get(entry.entry_id):
                row = records[raw_id]
                link = LexicalConceptLink(
                    link_id=f"usg-link:{quote(raw_id, safe='')}:missing",
                    lexical_entry_id=entry.entry_id,
                    lexical_meaning_id=None,
                    semantic_concept_id=None,
                    source_lexical_record_id=raw_id,
                    source_concept_record_id=None,
                    predicate="NO_EXPLICIT_CONCEPT_LINK",
                    relationship_kind="MISSING",
                    direction="DIRECTED",
                    status="MISSING",
                    lifecycle_state=str(row["state"]),
                    evidence_ids=tuple(str(x) for x in row.get("evidence_ids", ())),
                    evidence_statuses=tuple(sorted({
                        str(evidence_raw[x].get("support_status", "UNASSESSED"))
                        for x in row.get("evidence_ids", ()) if x in evidence_raw
                    })),
                    context_ids=tuple(str(x) for x in row.get("context_ids", ())),
                    source_reference=_source_reference(
                        namespace, bundle_id, "records", raw_id, str(row["state"]),
                        source_ids_by_record[raw_id], tuple(str(x) for x in row.get("evidence_ids", ())),
                        tuple(str(x) for x in row.get("context_ids", ())),
                    ),
                    source_kind="IMPORT_ABSTENTION",
                    source_subject_record_id=raw_id,
                    source_object_record_id=None,
                )
                link_rows.append(link)
                links_by_entry.setdefault(entry.entry_id, []).append(link)

        link_groups: list[LexicalConceptLinkGroup] = []
        for entry_id, links in sorted(links_by_entry.items()):
            candidates = tuple(sorted({
                item.semantic_concept_id for item in links if item.semantic_concept_id is not None
            }))
            missing = not candidates
            status = "MISSING" if missing else (
                "CONFLICTED" if any(item.status in {"CONFLICTED", "CONTRADICTED"} for item in links)
                else "AMBIGUOUS" if len(candidates) > 1
                else "UNRESOLVED" if any(item.status == "UNRESOLVED" for item in links)
                else "LINKED"
            )
            link_groups.append(LexicalConceptLinkGroup(
                group_id=f"usg-link-group:{quote(entry_id, safe='')}:entry",
                lexical_entry_id=entry_id,
                lexical_meaning_id=None,
                scope="ENTRY",
                candidate_concept_ids=candidates,
                link_ids=tuple(item.link_id for item in links),
                status=status,
            ))
        for meaning_id, links in sorted(links_by_meaning.items()):
            candidates = tuple(sorted({
                item.semantic_concept_id for item in links if item.semantic_concept_id is not None
            }))
            status = (
                "CONFLICTED" if any(item.status in {"CONFLICTED", "CONTRADICTED"} for item in links)
                else "AMBIGUOUS" if len(candidates) > 1
                else "UNRESOLVED" if any(item.status == "UNRESOLVED" for item in links)
                else "MISSING" if not candidates
                else "LINKED"
            )
            entry_id = links[0].lexical_entry_id
            link_groups.append(LexicalConceptLinkGroup(
                group_id=f"usg-link-group:{quote(meaning_id, safe='')}:meaning",
                lexical_entry_id=entry_id,
                lexical_meaning_id=meaning_id,
                scope="MEANING",
                candidate_concept_ids=candidates,
                link_ids=tuple(item.link_id for item in links),
                status=status,
            ))

        # Explicit morphology: only recognized relation labels and correct
        # source record kinds establish a form-to-lemma connection.
        morphology_by_form: dict[str, list[MorphologicalFormLink]] = {}
        form_source_ids = {
            raw_id: entry.entry_id for raw_id, entry in entry_by_source_id.items()
            if records[raw_id]["kind"] == "LEXICAL_FORM"
        }
        lemma_source_ids = {
            raw_id: entry.entry_id for raw_id, entry in entry_by_source_id.items()
            if records[raw_id]["kind"] == "LEMMA"
        }
        morphology_rows: list[MorphologicalFormLink] = []
        for row in sorted(bundle["relations"], key=lambda x: str(x["id"])):
            raw_id = str(row["id"])
            subject, object_ = str(row["subject_record_id"]), str(row["object_record_id"])
            if subject not in ore_ids or object_ not in ore_ids:
                continue
            subject_kind, object_kind = str(records[subject]["kind"]), str(records[object_]["kind"])
            predicate = str(row["predicate"]).strip().upper()
            if subject_kind == "LEXICAL_FORM" and object_kind == "LEMMA":
                form_id, lemma_id = subject, object_
                direction = "FORM_TO_LEMMA" if predicate in _FORM_TO_LEMMA else "UNRESOLVED"
            elif subject_kind == "LEMMA" and object_kind == "LEXICAL_FORM":
                form_id, lemma_id = object_, subject
                direction = "LEMMA_TO_FORM" if predicate in _LEMMA_TO_FORM else "UNRESOLVED"
            else:
                continue
            if form_id not in form_source_ids or lemma_id not in lemma_source_ids:
                exclusions.append(USGImportExclusion(raw_id, "MORPHOLOGY_LINK", "form_or_lemma_lexical_entry_not_imported"))
                continue
            state = str(row["state"])
            if not self.policy.state_enabled(state) and state != "UNRESOLVED":
                continue
            usable, reason, relation_sources = self.policy.source_access(row, sources_raw, evidence_raw)
            if not usable:
                exclusions.append(USGImportExclusion(raw_id, "MORPHOLOGY_LINK", reason))
                continue
            form_payload = records[form_id]["payload"]
            evidence_ids = tuple(str(x) for x in row["evidence_ids"])
            context_ids = tuple(str(x) for x in row["context_ids"])
            status, evidence_statuses = _assertion_status(
                state, evidence_ids, evidence_raw,
                candidate_enabled=self.policy.include_candidate_states,
            )
            if direction == "UNRESOLVED":
                status = "UNRESOLVED"
            link = MorphologicalFormLink(
                link_id=f"usg-morph:{quote(raw_id, safe='')}",
                form_entry_id=form_source_ids[form_id],
                lemma_entry_id=lemma_source_ids[lemma_id] if direction != "UNRESOLVED" else None,
                form_record_id=form_id,
                lemma_record_id=lemma_id,
                predicate=str(row["predicate"]),
                direction=direction,
                status=status,
                lifecycle_state=state,
                language=str(form_payload["language"]),
                script=str(form_payload["script"]) if form_payload.get("script") is not None else None,
                grammatical_information_json=_json({
                    key: value for key, value in form_payload.items()
                    if key not in {"form", "language", "script"}
                }),
                evidence_ids=evidence_ids,
                evidence_statuses=evidence_statuses,
                context_ids=context_ids,
                source_reference=_source_reference(
                    namespace, bundle_id, "relations", raw_id, state,
                    relation_sources, evidence_ids, context_ids,
                ),
                source_kind="EXPLICIT_RELATION",
                source_subject_record_id=subject,
                source_object_record_id=object_,
            )
            morphology_rows.append(link)
            morphology_by_form.setdefault(form_source_ids[form_id], []).append(link)
            imported_relations.add(raw_id)

        for form_raw_id, form_entry_id in sorted(form_source_ids.items()):
            if not morphology_by_form.get(form_entry_id):
                row = records[form_raw_id]
                morphology_rows.append(MorphologicalFormLink(
                    link_id=f"usg-morph:{quote(form_raw_id, safe='')}:missing",
                    form_entry_id=form_entry_id,
                    lemma_entry_id=None,
                    form_record_id=form_raw_id,
                    lemma_record_id=None,
                    predicate=None,
                    direction="UNRESOLVED",
                    status="MISSING",
                    lifecycle_state=str(row["state"]),
                    language=str(row["payload"]["language"]),
                    script=str(row["payload"].get("script")) if row["payload"].get("script") is not None else None,
                    grammatical_information_json=_json({
                        key: value for key, value in row["payload"].items()
                        if key not in {"form", "language", "script"}
                    }),
                    evidence_ids=tuple(str(x) for x in row.get("evidence_ids", ())),
                    evidence_statuses=tuple(sorted({
                        str(evidence_raw[x].get("support_status", "UNASSESSED"))
                        for x in row.get("evidence_ids", ()) if x in evidence_raw
                    })),
                    context_ids=tuple(str(x) for x in row.get("context_ids", ())),
                    source_reference=_source_reference(
                        namespace, bundle_id, "records", form_raw_id, str(row["state"]),
                        source_ids_by_record[form_raw_id], tuple(str(x) for x in row.get("evidence_ids", ())),
                        tuple(str(x) for x in row.get("context_ids", ())),
                    ),
                    source_kind="IMPORT_ABSTENTION",
                ))
            elif len({item.lemma_record_id for item in morphology_by_form[form_entry_id] if item.lemma_record_id}) > 1:
                morphology_rows = [
                    MorphologicalFormLink(
                        link_id=item.link_id, form_entry_id=item.form_entry_id,
                        lemma_entry_id=item.lemma_entry_id, form_record_id=item.form_record_id,
                        lemma_record_id=item.lemma_record_id, predicate=item.predicate,
                        direction=item.direction, status="AMBIGUOUS",
                        lifecycle_state=item.lifecycle_state, language=item.language, script=item.script,
                        grammatical_information_json=item.grammatical_information_json,
                        evidence_ids=item.evidence_ids, evidence_statuses=item.evidence_statuses,
                        context_ids=item.context_ids, source_reference=item.source_reference,
                        source_kind=item.source_kind,
                        source_subject_record_id=item.source_subject_record_id,
                        source_object_record_id=item.source_object_record_id,
                    ) if item.form_entry_id == form_entry_id else item
                    for item in morphology_rows
                ]

        # Evidence and contexts from relation/mapping assertions remain attached
        # to their exact source assertion IDs, including contradiction metadata.
        for row in sorted(bundle["relations"], key=lambda x: str(x["id"])):
            raw_id = str(row["id"])
            relation_target_id = _aux_id("usg-rel", namespace, bundle_id, raw_id)
            attach_evidence(
                relation_target_id,
                "RELATION", raw_id, tuple(str(x) for x in row["evidence_ids"]),
            )
            for context_id in row.get("context_ids", ()):
                if context_id in context_raw:
                    assignments.append(ContextAssignment(
                        context_id=str(context_id), target_id=relation_target_id,
                        target_kind="RELATION", source_record_id=raw_id,
                    ))
        for row in sorted(bundle["mappings"], key=lambda x: str(x["id"])):
            raw_id = str(row["id"])
            mapping_target_id = _aux_id("usg-map", namespace, bundle_id, raw_id)
            attach_evidence(
                mapping_target_id,
                "MAPPING", raw_id, tuple(str(x) for x in row["evidence_ids"]),
            )
            for context_id in row.get("context_ids", ()):
                if context_id in context_raw:
                    assignments.append(ContextAssignment(
                        context_id=str(context_id), target_id=mapping_target_id,
                        target_kind="MAPPING", source_record_id=raw_id,
                    ))

        assignments = sorted(assignments, key=lambda x: (x.context_id, x.target_id, x.target_kind))
        evidence_attachments = sorted(
            set(evidence_attachments),
            key=lambda x: (x.target_kind, x.target_id, x.evidence_id),
        )
        link_rows.sort(key=lambda x: (x.lexical_entry_id, x.lexical_meaning_id or "", x.link_id))
        link_groups.sort(key=lambda x: (x.scope, x.lexical_entry_id, x.lexical_meaning_id or ""))
        morphology_rows.sort(key=lambda x: (x.form_entry_id, x.link_id))
        report = USGLanguageIntegrationReport(
            bundle_id=bundle_id,
            lexical_concept_links_created=len(link_rows),
            lexical_concept_link_groups_created=len(link_groups),
            missing_lexical_concept_links=sum(item.status == "MISSING" for item in link_rows),
            ambiguous_lexical_concept_groups=sum(item.status == "AMBIGUOUS" for item in link_groups),
            conflicting_lexical_concept_links=sum(item.status in {"CONFLICTED", "CONTRADICTED"} for item in link_rows),
            morphology_links_created=len(morphology_rows),
            missing_morphology_links=sum(item.status == "MISSING" for item in morphology_rows),
            ambiguous_morphology_links=sum(item.status == "AMBIGUOUS" for item in morphology_rows),
            contexts_created=len(contexts),
            context_assignments_created=len(assignments),
            evidence_records_created=len(evidence_records),
            evidence_attachments_created=len(evidence_attachments),
            source_records_created=len(sources),
            imported_record_ids=tuple(sorted(imported_records)),
            imported_relation_ids=tuple(sorted(imported_relations)),
            exclusions=tuple(sorted(exclusions, key=lambda x: (x.record_id, x.reason))),
            warnings=(
                "Lexical meaning-to-concept identity links exist only where the same imported USG LEXICAL_SENSE record explicitly generated both representations.",
                "Additional lexical-concept links are imported only from explicit USG relation or mapping rows with lexical and semantic endpoints.",
                "One-to-many and many-to-one links are preserved as independent directed link records; ambiguous groups do not choose a preferred concept.",
                "Missing links are explicit abstentions. Contradicting, supporting, neutral and unassessed evidence statuses are retained per assertion.",
                "Surface-form-to-lemma morphology is created only for explicit form/lemma relation pairs with recognized direction predicates; unknown labels remain unresolved and are never guessed.",
                "Context records retain type, value, language profile and scope; no synthetic numeric context feature values are created.",
                "Evidence summaries are not verified facts. Local reasoning use is disabled whenever the linked source rights do not permit local import; redistribution permissions are retained separately.",
            ),
        )
        return USGLanguageIntegrationResource(
            lexical_concept_links=tuple(link_rows),
            lexical_concept_groups=tuple(link_groups),
            morphology_links=tuple(morphology_rows),
            contexts=contexts,
            context_assignments=tuple(assignments),
            evidence=tuple(evidence_records),
            evidence_attachments=tuple(evidence_attachments),
            sources=sources,
            namespace=namespace,
            bundle_id=bundle_id,
            bundle_digest=digest,
            report=report,
        )


__all__ = [
    "USGLanguageIntegrationAdapter", "USGLanguageIntegrationResource",
    "USGLanguageIntegrationReport",
]
