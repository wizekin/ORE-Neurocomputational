"""Read-only conversion of complete USG semantic knowledge into ORE."""
from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json
import math
from typing import Any, Mapping
from urllib.parse import quote

from ore_semantic.contracts import SemanticConcept
from ore_semantic.knowledge import SemanticCategory, SemanticCategoryMembership, SemanticRelation
from ore_semantic_composition.contracts import PredicateFrame

from .contracts import USGImportExclusion
from .lexical import _canonical_json, _lexical_category, _ore_record_id, _positive_number, _source_reference
from .policy import USGImportPolicy
from .validation import require_valid_knowledge_bundle


_ACTIVE = frozenset({"ACCEPTED", "VALIDATED_FOR_PROFILE"})
_RETAIN_INACTIVE = frozenset({"UNRESOLVED"})
_ENDPOINT_KINDS = frozenset({
    "LEXICAL_FORM", "LEMMA", "LEXICAL_SENSE", "CONCEPT", "SEMANTIC_CLASS", "PREDICATE_FRAME",
})
_MEMBERSHIP_PREDICATES = frozenset({"INSTANCE_OF", "CLASSIFIED_AS", "MEMBER_OF", "HAS_CLASS"})
_HIERARCHY_PREDICATES = frozenset({
    "SUBCLASS_OF", "IS_A", "KIND_OF", "NARROWER_THAN", "BROADER_THAN",
    "PARENT_OF", "CHILD_OF", "SUBTYPE_OF", "SUPERTYPE_OF",
    "HYPONYM_OF", "HYPERNYM_OF",
})


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _aux_id(prefix: str, namespace: str, bundle_id: str, source_id: str) -> str:
    scope = sha256(_canonical_json({"namespace": namespace, "bundle_id": bundle_id})).hexdigest()
    return f"{prefix}:{scope}:{quote(source_id, safe='')}"


def _relation_type(predicate: str) -> str:
    value = predicate.strip().upper()
    if value == "UNRESOLVED":
        return "UNRESOLVED"
    if value in {"EQUIVALENT", "EQUIVALENCE", "SAME_AS"}:
        return "EQUIVALENCE"
    if value in _HIERARCHY_PREDICATES:
        return "HIERARCHY"
    if value in {"RELATED_TO", "ASSOCIATED_WITH", "OVERLAPPING", "OVERLAPS", "CO_OCCURS_WITH"}:
        return "ASSOCIATION"
    return "OTHER"


def _assertion_status(state: str, evidence_ids: tuple[str, ...], evidence: Mapping[str, Mapping[str, Any]], candidate_enabled: bool, endpoint_states: tuple[str, ...] = ()) -> str:
    if state == "UNRESOLVED" or "UNRESOLVED" in endpoint_states:
        return "UNRESOLVED"
    if candidate_enabled and (state == "CANDIDATE" or "CANDIDATE" in endpoint_states):
        return "CANDIDATE"
    statuses = {
        str(evidence[item].get("support_status", "UNASSESSED"))
        for item in evidence_ids if item in evidence
    }
    if "SUPPORTS" in statuses and "CONTRADICTS" in statuses:
        return "CONFLICTED"
    if "CONTRADICTS" in statuses:
        return "CONTRADICTED"
    if "UNASSESSED" in statuses or not statuses:
        return "UNASSESSED"
    return "ACTIVE"


def _aux_reference(namespace: str, bundle_id: str, collection: str, source_id: str, state: str,
                   source_ids: tuple[str, ...], evidence_ids: tuple[str, ...],
                   context_ids: tuple[str, ...]) -> str:
    return (
        f"usg-knowledge://{quote(namespace, safe='')}/{quote(bundle_id, safe='')}/"
        f"{collection}/{quote(source_id, safe='')}?state={quote(state, safe='')}"
        f"&sources={','.join(quote(x, safe='') for x in source_ids)}"
        f"&evidence={','.join(quote(x, safe='') for x in evidence_ids)}"
        f"&contexts={','.join(quote(x, safe='') for x in context_ids)}"
    )


@dataclass(frozen=True, slots=True)
class USGSemanticConceptRecord:
    concept: SemanticConcept
    usg_record_id: str
    record_kind: str
    lifecycle_state: str
    active: bool
    definition: str | None
    lemma_record_id: str | None
    lemma_source_reference: str | None
    sense_source_id: str | None
    source_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    context_ids: tuple[str, ...]
    payload_json: str


@dataclass(frozen=True, slots=True)
class USGPredicateRole:
    name: str | None
    description: str | None
    required: bool | None
    constraints_json: str
    semantic_relationships_json: str
    raw_json: str


@dataclass(frozen=True, slots=True)
class USGPredicateStructure:
    frame_id: str
    usg_record_id: str
    frame_key: str
    frame_version: str
    lifecycle_state: str
    active: bool
    status: str
    source_reference: str
    roles: tuple[USGPredicateRole, ...]
    ore_frame: PredicateFrame | None
    unresolved_fields: tuple[str, ...]
    constraints_json: str
    semantic_relationships_json: str
    payload_json: str


@dataclass(frozen=True, slots=True)
class USGSemanticKnowledgeImportReport:
    bundle_id: str
    semantic_concepts_created: int
    lexical_senses_converted: int
    concept_records_converted: int
    semantic_categories_created: int
    category_memberships_created: int
    relations_created: int
    predicate_structures_created: int
    executable_predicate_frames_created: int
    unresolved_or_conflicted_memberships: int
    imported_record_ids: tuple[str, ...]
    imported_relation_ids: tuple[str, ...]
    excluded_records: tuple[USGImportExclusion, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class USGSemanticKnowledgeResource:
    concepts: tuple[SemanticConcept, ...]
    concept_records: tuple[USGSemanticConceptRecord, ...]
    categories: tuple[SemanticCategory, ...]
    memberships: tuple[SemanticCategoryMembership, ...]
    relations: tuple[SemanticRelation, ...]
    predicate_frames: tuple[USGPredicateStructure, ...]
    namespace: str
    bundle_id: str
    bundle_digest: str
    report: USGSemanticKnowledgeImportReport

    @property
    def active_concepts(self) -> tuple[SemanticConcept, ...]:
        active_ids = {item.concept.concept_id for item in self.concept_records if item.active}
        return tuple(item for item in self.concepts if item.concept_id in active_ids)

    @property
    def executable_frames(self) -> tuple[PredicateFrame, ...]:
        return tuple(
            item.ore_frame for item in self.predicate_frames
            if item.active and item.status == "EXECUTABLE" and item.ore_frame is not None
        )


class USGSemanticKnowledgeAdapter:
    """Import senses, concepts, categories, explicit links, and predicate structures."""

    def __init__(self, *, include_candidate_states: bool = False) -> None:
        self.policy = USGImportPolicy(include_candidate_states=include_candidate_states)

    def _eligible(self, state: str) -> tuple[bool, bool]:
        active = self.policy.state_enabled(state)
        return active or state in _RETAIN_INACTIVE, active

    def convert(self, bundle: Mapping[str, Any]) -> USGSemanticKnowledgeResource:
        require_valid_knowledge_bundle(bundle)
        namespace, bundle_id = str(bundle["namespace"]), str(bundle["bundle_id"])
        digest = sha256(_canonical_json(bundle)).hexdigest()
        records = {str(row["id"]): row for row in bundle["records"]}
        sources = {str(row["id"]): row for row in bundle["sources"]}
        evidence = {str(row["id"]): row for row in bundle["evidence"]}
        exclusions: list[USGImportExclusion] = []
        imported_records: set[str] = set()
        endpoint_ids: dict[str, str] = {}
        source_ids_by_record: dict[str, tuple[str, ...]] = {}

        # Index all permitted endpoints once, including unresolved records for
        # explicit, inactive knowledge inspection.
        for raw_id, row in sorted(records.items()):
            if row.get("kind") not in _ENDPOINT_KINDS:
                continue
            state = str(row["state"])
            eligible, _ = self._eligible(state)
            if not eligible:
                exclusions.append(USGImportExclusion(
                    raw_id, str(row["kind"]), f"lifecycle_state_not_enabled:{state}",
                ))
                continue
            usable, reason, source_ids = self.policy.source_access(row, sources, evidence)
            if not usable:
                exclusions.append(USGImportExclusion(raw_id, str(row["kind"]), reason))
                continue
            endpoint_ids[raw_id] = _ore_record_id(namespace, bundle_id, row)
            source_ids_by_record[raw_id] = source_ids

        # Convert each lexical sense and concept to a separate SemanticConcept.
        concept_records: list[USGSemanticConceptRecord] = []
        concepts: list[SemanticConcept] = []
        for raw_id, row in sorted(records.items()):
            kind = str(row["kind"])
            if kind not in {"LEXICAL_SENSE", "CONCEPT"}:
                continue
            if raw_id not in endpoint_ids:
                eligible, _ = self._eligible(str(row["state"]))
                if eligible and not any(item.record_id == raw_id for item in exclusions):
                    exclusions.append(USGImportExclusion(
                        raw_id, kind, f"source_not_permitted_or_state_not_enabled:{row['state']}",
                    ))
                continue
            payload = row["payload"]
            state = str(row["state"])
            definition = payload.get("definition")
            definition = definition if isinstance(definition, str) else None
            label = payload.get("label")
            if not isinstance(label, str) or not label.strip():
                label = definition
            if not isinstance(label, str) or not label.strip():
                exclusions.append(USGImportExclusion(raw_id, kind, "no_explicit_concept_label_or_definition"))
                continue
            if kind == "LEXICAL_SENSE":
                frequency = _positive_number(payload.get("prior"), 1.0)
                lemma_id = str(payload["lemma_record_id"])
                lemma_record = records.get(lemma_id)
                lemma_source_reference = None
                lemma_alias = None
                if lemma_record is not None and lemma_id in endpoint_ids:
                    lemma_payload = lemma_record["payload"]
                    lemma_value = lemma_payload.get("lemma")
                    if isinstance(lemma_value, str) and lemma_value.strip():
                        lemma_alias = lemma_value.strip()
                    lexical_category = _lexical_category(lemma_payload.get("part_of_speech"))
                    lemma_source_reference = _source_reference(
                        namespace, bundle_id, lemma_record, source_ids_by_record[lemma_id],
                    )
                else:
                    lexical_category = "unknown"
                sense_source = str(payload["sense_source_id"])
            else:
                lexical_value = payload.get("lexical_category", "unknown")
                lexical_category = lexical_value.strip() if isinstance(lexical_value, str) and lexical_value.strip() else "unknown"
                frequency = _positive_number(payload.get("frequency"), 1.0)
                lemma_id = None
                lemma_source_reference = None
                lemma_alias = None
                sense_source = None
            language_value = payload.get("language", "und")
            language = language_value.strip() if isinstance(language_value, str) and language_value.strip() else "und"
            aliases_value = payload.get("aliases", ())
            alias_values = [x.strip() for x in aliases_value if isinstance(x, str) and x.strip()] if isinstance(aliases_value, (list, tuple)) else []
            if lemma_alias and lemma_alias.casefold() not in {item.casefold() for item in alias_values}:
                alias_values.append(lemma_alias)
            aliases = tuple(alias_values)
            source_ids = source_ids_by_record[raw_id]
            evidence_ids = tuple(str(x) for x in row.get("evidence_ids", ()))
            context_ids = tuple(str(x) for x in row.get("context_ids", ()))
            concept = SemanticConcept(
                concept_id=endpoint_ids[raw_id],
                label=label,
                aliases=aliases,
                semantic_features={},
                modality_features={},
                lexical_category=lexical_category,
                frequency=frequency,
                language=language,
                verified_by=f"USG_KNOWLEDGE_STATE:{state};not_an_ORE_truth_verification",
                source_reference=_source_reference(namespace, bundle_id, row, source_ids),
                definition=definition,
            )
            concepts.append(concept)
            concept_records.append(USGSemanticConceptRecord(
                concept=concept,
                usg_record_id=raw_id,
                record_kind=kind,
                lifecycle_state=state,
                active=self.policy.state_enabled(state),
                definition=definition,
                lemma_record_id=lemma_id,
                lemma_source_reference=lemma_source_reference,
                sense_source_id=sense_source,
                source_ids=source_ids,
                evidence_ids=evidence_ids,
                context_ids=context_ids,
                payload_json=_json(payload),
            ))
            imported_records.add(raw_id)

        # Convert semantic classes into ORE categories and resolve explicit
        # parent_class_keys only when the key identifies exactly one category.
        raw_categories: list[SemanticCategory] = []
        for raw_id, row in sorted(records.items()):
            if row.get("kind") != "SEMANTIC_CLASS" or raw_id not in endpoint_ids:
                continue
            payload = row["payload"]
            parent_value = payload.get("parent_class_keys", ())
            parent_keys = tuple(
                x for x in parent_value if isinstance(x, str) and x.strip()
            ) if isinstance(parent_value, (list, tuple)) else ()
            raw_categories.append(SemanticCategory(
                category_id=endpoint_ids[raw_id],
                class_key=str(payload["class_key"]),
                definition=str(payload["definition"]),
                unresolved_parent_keys=parent_keys,
                lifecycle_state=str(row["state"]),
                source_reference=_source_reference(namespace, bundle_id, row, source_ids_by_record[raw_id]),
                payload_json=_json(payload),
            ))
            imported_records.add(raw_id)
        by_class_key: dict[str, list[str]] = {}
        for category in raw_categories:
            by_class_key.setdefault(category.class_key, []).append(category.category_id)
        categories: list[SemanticCategory] = []
        for category in raw_categories:
            payload = json.loads(category.payload_json)
            keys_value = payload.get("parent_class_keys", ())
            keys = tuple(x for x in keys_value if isinstance(x, str) and x.strip()) if isinstance(keys_value, (list, tuple)) else ()
            parents: list[str] = []
            unresolved: list[str] = []
            for key in keys:
                matches = by_class_key.get(key, [])
                if len(matches) == 1 and matches[0] != category.category_id:
                    parents.append(matches[0])
                else:
                    unresolved.append(key)
            categories.append(replace(
                category,
                parent_category_ids=tuple(sorted(set(parents))),
                unresolved_parent_keys=tuple(sorted(set(unresolved))),
            ))

        # Preserve every predicate role and all constraints/relationships. A
        # native ORE PredicateFrame is exposed only when its supported semantics
        # are complete enough to compile without dropping declared constraints.
        predicate_structures: list[USGPredicateStructure] = []
        for raw_id, row in sorted(records.items()):
            if row.get("kind") != "PREDICATE_FRAME" or raw_id not in endpoint_ids:
                continue
            payload, state = row["payload"], str(row["state"])
            active = self.policy.state_enabled(state)
            roles_value = payload.get("roles")
            roles_list = roles_value if isinstance(roles_value, list) else []
            roles: list[USGPredicateRole] = []
            role_names: list[str] = []
            unresolved: list[str] = []
            can_compile = isinstance(roles_value, list) and bool(roles_value)
            for index, item in enumerate(roles_list):
                if isinstance(item, str) and item.strip():
                    name, description, required, constraints, relationships = item, None, None, None, None
                elif isinstance(item, Mapping):
                    raw_name = item.get("role", item.get("name"))
                    name = raw_name.strip() if isinstance(raw_name, str) and raw_name.strip() else None
                    raw_required = item.get("required")
                    required = raw_required if isinstance(raw_required, bool) else None
                    if "required" in item and required is None:
                        unresolved.append(f"roles[{index}].required")
                    raw_description = item.get("description", item.get("definition"))
                    description = raw_description if isinstance(raw_description, str) else None
                    role_constraint_keys = ("constraints", "argument_constraints", "semantic_constraints", "selectional_preferences")
                    constraints = {key: item[key] for key in role_constraint_keys if key in item} or None
                    role_relation_keys = ("semantic_relationships", "relations", "relationships")
                    relationships = {key: item[key] for key in role_relation_keys if key in item} or None
                else:
                    name, description, required, constraints, relationships = None, None, None, item, None
                if name is None:
                    can_compile = False
                    unresolved.append(f"roles[{index}].name")
                elif name in role_names:
                    can_compile = False
                    unresolved.append(f"roles[{index}].duplicate_name")
                else:
                    role_names.append(name)
                roles.append(USGPredicateRole(
                    name=name,
                    description=description,
                    required=required,
                    constraints_json=_json(constraints),
                    semantic_relationships_json=_json(relationships),
                    raw_json=_json(item),
                ))
                if constraints not in (None, {}, [], ()):
                    unresolved.append(f"roles[{index}].constraints_not_compiled")
                if relationships not in (None, {}, [], ()):
                    unresolved.append(f"roles[{index}].semantic_relationships_not_compiled")
            if not isinstance(roles_value, list) or not roles_value:
                unresolved.append("roles")
            required_roles = [role.name for role in roles if role.name and role.required is True]
            declared_required = payload.get("required_roles", ())
            if declared_required not in (None, (), []):
                if not isinstance(declared_required, list) or any(not isinstance(x, str) or not x.strip() for x in declared_required):
                    can_compile = False
                    unresolved.append("required_roles")
                else:
                    for name in declared_required:
                        if name not in role_names:
                            can_compile = False
                            unresolved.append(f"required_roles.{name}")
                        elif name not in required_roles:
                            required_roles.append(name)
            preferences = payload.get("selectional_preferences", {})
            normalized_preferences: dict[str, dict[str, float]] = {}
            if not isinstance(preferences, Mapping):
                can_compile = False
                unresolved.append("selectional_preferences")
            else:
                for role_name, feature_values in preferences.items():
                    if role_name not in role_names or not isinstance(feature_values, Mapping):
                        can_compile = False
                        unresolved.append(f"selectional_preferences.{role_name}")
                        continue
                    normalized_preferences[str(role_name)] = {}
                    for feature, raw_value in feature_values.items():
                        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)) or not math.isfinite(float(raw_value)) or not 0.0 <= float(raw_value) <= 1.0:
                            can_compile = False
                            unresolved.append(f"selectional_preferences.{role_name}.{feature}")
                        else:
                            normalized_preferences[str(role_name)][str(feature)] = float(raw_value)
            frame_constraint_keys = ("constraints", "argument_constraints", "role_constraints", "semantic_constraints")
            constraints_value = {key: payload[key] for key in frame_constraint_keys if key in payload} or None
            frame_relationship_keys = ("semantic_relationships", "role_relationships", "relations", "relationships")
            relationships_value = {key: payload[key] for key in frame_relationship_keys if key in payload} or None
            if constraints_value not in (None, {}, [], ()):
                unresolved.append("constraints_not_compiled")
            if relationships_value not in (None, {}, [], ()):
                unresolved.append("semantic_relationships_not_compiled")
            if payload.get("complete") is False:
                unresolved.append("source_marks_frame_incomplete")
            if payload.get("resolved") is False or state == "UNRESOLVED":
                unresolved.append("source_marks_frame_unresolved")
            if payload.get("missing_roles") not in (None, (), []):
                unresolved.append("source_declares_missing_roles")
            if payload.get("unresolved_roles") not in (None, (), []):
                unresolved.append("source_declares_unresolved_roles")
            blocked = bool(unresolved)
            ore_frame: PredicateFrame | None = None
            if active and can_compile and role_names and not blocked and payload.get("complete") is not False and payload.get("resolved") is not False:
                try:
                    ore_frame = PredicateFrame(
                        predicate=str(payload["frame_key"]),
                        roles=tuple(role_names),
                        required_roles=tuple(required_roles),
                        selectional_preferences=normalized_preferences,
                        source_reference=_source_reference(namespace, bundle_id, row, source_ids_by_record[raw_id]) + f"&frame_version={quote(str(payload['version']), safe='')}",
                    )
                except (TypeError, ValueError):
                    unresolved.append("ore_predicate_frame_contract")
            status = (
                "UNRESOLVED" if state == "UNRESOLVED" or payload.get("resolved") is False
                else "INCOMPLETE" if unresolved or payload.get("complete") is False
                else "EXECUTABLE" if ore_frame is not None
                else "PRESERVED_ONLY"
            )
            predicate_structures.append(USGPredicateStructure(
                frame_id=endpoint_ids[raw_id],
                usg_record_id=raw_id,
                frame_key=str(payload["frame_key"]),
                frame_version=str(payload["version"]),
                lifecycle_state=state,
                active=active,
                status=status,
                source_reference=_source_reference(namespace, bundle_id, row, source_ids_by_record[raw_id]),
                roles=tuple(roles),
                ore_frame=ore_frame,
                unresolved_fields=tuple(sorted(set(unresolved))),
                constraints_json=_json(constraints_value),
                semantic_relationships_json=_json(relationships_value),
                payload_json=_json(payload),
            ))
            imported_records.add(raw_id)

        # Relations and mappings are source-backed directed edges. The raw
        # predicate/mapping type is never replaced by an inferred relation.
        relations: list[SemanticRelation] = []
        memberships: list[SemanticCategoryMembership] = []
        imported_relation_ids: set[str] = set()
        category_ids = {item.category_id for item in categories}
        raw_edges: list[tuple[Any, ...]] = []
        for row in sorted(bundle["relations"], key=lambda x: str(x["id"])):
            raw_id = str(row["id"])
            subject, object_ = str(row["subject_record_id"]), str(row["object_record_id"])
            if subject not in endpoint_ids or object_ not in endpoint_ids:
                exclusions.append(USGImportExclusion(raw_id, "RELATION", "both_relation_endpoints_must_be_preserved_rights_permitted_supported_records"))
                continue
            state = str(row["state"])
            eligible, _ = self._eligible(state)
            if not eligible:
                exclusions.append(USGImportExclusion(raw_id, "RELATION", f"lifecycle_state_not_enabled:{state}"))
                continue
            usable, reason, source_ids = self.policy.source_access(row, sources, evidence)
            if not usable:
                exclusions.append(USGImportExclusion(raw_id, "RELATION", reason))
                continue
            evidence_ids = tuple(str(x) for x in row["evidence_ids"])
            context_ids = tuple(str(x) for x in row["context_ids"])
            relation_id = _aux_id("usg-rel", namespace, bundle_id, raw_id)
            source_ref = _aux_reference(namespace, bundle_id, "relations", raw_id, state, source_ids, evidence_ids, context_ids)
            raw_edges.append((relation_id, endpoint_ids[subject], str(row["predicate"]), endpoint_ids[object_], state, evidence_ids, context_ids, source_ref, "RELATION"))
            imported_relation_ids.add(raw_id)

        for row in sorted(bundle["mappings"], key=lambda x: str(x["id"])):
            raw_id = str(row["id"])
            subject, object_ = str(row["source_record_id"]), str(row["target_record_id"])
            if subject not in endpoint_ids or object_ not in endpoint_ids:
                exclusions.append(USGImportExclusion(raw_id, "MAPPING", "both_mapping_endpoints_must_be_preserved_semantic_concepts"))
                continue
            if records[subject]["kind"] not in {"CONCEPT", "LEXICAL_SENSE"} or records[object_]["kind"] not in {"CONCEPT", "LEXICAL_SENSE"}:
                exclusions.append(USGImportExclusion(raw_id, "MAPPING", "mapping_endpoints_must_be_CONCEPT_or_LEXICAL_SENSE"))
                continue
            state = str(row["state"])
            eligible, _ = self._eligible(state)
            if not eligible:
                exclusions.append(USGImportExclusion(raw_id, "MAPPING", f"lifecycle_state_not_enabled:{state}"))
                continue
            usable, reason, source_ids = self.policy.source_access(row, sources, evidence)
            if not usable:
                exclusions.append(USGImportExclusion(raw_id, "MAPPING", reason))
                continue
            evidence_ids = tuple(str(x) for x in row["evidence_ids"])
            context_ids = tuple(str(x) for x in row["context_ids"])
            relation_id = _aux_id("usg-map", namespace, bundle_id, raw_id)
            source_ref = _aux_reference(namespace, bundle_id, "mappings", raw_id, state, source_ids, evidence_ids, context_ids)
            raw_edges.append((relation_id, endpoint_ids[subject], str(row["mapping_type"]), endpoint_ids[object_], state, evidence_ids, context_ids, source_ref, "MAPPING"))
            imported_relation_ids.add(raw_id)

        endpoint_states = {
            endpoint_ids[raw_id]: str(row["state"])
            for raw_id, row in records.items() if raw_id in endpoint_ids
        }
        for relation_id, subject_id, predicate, object_id, state, evidence_ids, context_ids, source_ref, source_kind in raw_edges:
            relation_status = _assertion_status(
                state, evidence_ids, evidence, self.policy.include_candidate_states,
                (endpoint_states.get(subject_id, "UNRESOLVED"), endpoint_states.get(object_id, "UNRESOLVED")),
            )
            if predicate.strip().upper() == "UNRESOLVED":
                relation_status = "UNRESOLVED"
            relations.append(SemanticRelation(
                relation_id=relation_id,
                subject_id=subject_id,
                object_id=object_id,
                predicate=predicate,
                relation_type=_relation_type(predicate),
                direction="DIRECTED",
                lifecycle_state=state,
                evidence_ids=evidence_ids,
                context_ids=context_ids,
                source_reference=source_ref,
                source_kind=source_kind,
                status=relation_status,
            ))
            if predicate.strip().upper() in _MEMBERSHIP_PREDICATES:
                subject_is_category, object_is_category = subject_id in category_ids, object_id in category_ids
                if subject_is_category != object_is_category:
                    evidence_statuses = tuple(sorted({
                        str(evidence[x].get("support_status", "UNASSESSED"))
                        for x in evidence_ids if x in evidence
                    }))
                    memberships.append(SemanticCategoryMembership(
                        membership_id=relation_id,
                        member_id=object_id if subject_is_category else subject_id,
                        category_id=subject_id if subject_is_category else object_id,
                        predicate=predicate,
                        direction="DIRECTED",
                        status=_assertion_status(
                            state, evidence_ids, evidence, self.policy.include_candidate_states,
                            (endpoint_states.get(subject_id, "UNRESOLVED"), endpoint_states.get(object_id, "UNRESOLVED")),
                        ),
                        lifecycle_state=state,
                        evidence_ids=evidence_ids,
                        evidence_statuses=evidence_statuses,
                        context_ids=context_ids,
                        source_reference=source_ref,
                        source_subject_id=subject_id,
                        source_object_id=object_id,
                    ))

        # Hierarchy parent IDs are added only for explicit parent keys or
        # explicit hierarchy relation records. Original directed relations remain.
        parent_map = {item.category_id: set(item.parent_category_ids) for item in categories}
        for relation in relations:
            if relation.relation_type != "HIERARCHY" or relation.subject_id not in category_ids or relation.object_id not in category_ids:
                continue
            if relation.predicate.strip().upper() == "BROADER_THAN":
                parent_map[relation.object_id].add(relation.subject_id)
            else:
                parent_map[relation.subject_id].add(relation.object_id)
        categories = [
            replace(item, parent_category_ids=tuple(sorted(parent_map[item.category_id])))
            for item in categories
        ]

        concepts.sort(key=lambda x: x.concept_id)
        concept_records.sort(key=lambda x: x.concept.concept_id)
        categories.sort(key=lambda x: x.category_id)
        memberships.sort(key=lambda x: x.membership_id)
        relations.sort(key=lambda x: x.relation_id)
        predicate_structures.sort(key=lambda x: x.frame_id)
        unresolved_count = sum(
            item.status in {"UNRESOLVED", "CONFLICTED", "CONTRADICTED", "UNASSESSED"}
            for item in memberships
        )
        report = USGSemanticKnowledgeImportReport(
            bundle_id=bundle_id,
            semantic_concepts_created=len(concepts),
            lexical_senses_converted=sum(x.record_kind == "LEXICAL_SENSE" for x in concept_records),
            concept_records_converted=sum(x.record_kind == "CONCEPT" for x in concept_records),
            semantic_categories_created=len(categories),
            category_memberships_created=len(memberships),
            relations_created=len(relations),
            predicate_structures_created=len(predicate_structures),
            executable_predicate_frames_created=sum(x.ore_frame is not None for x in predicate_structures),
            unresolved_or_conflicted_memberships=unresolved_count,
            imported_record_ids=tuple(sorted(imported_records)),
            imported_relation_ids=tuple(sorted(imported_relation_ids)),
            excluded_records=tuple(sorted(exclusions, key=lambda x: (x.record_id, x.reason))),
            warnings=(
                "Every LEXICAL_SENSE and CONCEPT keeps a separate ORE concept ID; no senses are merged.",
                "Definitions, original record IDs, source/evidence/context IDs, and complete raw predicate/class payloads are retained.",
                "Equivalence and association mappings preserve source-to-target orientation and never collapse concept identities.",
                "Category hierarchy and membership are derived only from explicit parent_class_keys or explicit USG relation rows.",
                "Multiple category memberships are not inherently conflicts; evidence is marked conflicted only when both SUPPORTS and CONTRADICTS occur for that assertion.",
                "UNRESOLVED source records are retained as inactive records and are excluded from active reasoning inventories.",
                "Uncompiled predicate constraints and semantic relationships remain preserved verbatim; those frames are not passed to executable ORE processors.",
                "No inverse, transitive, synonym, category-membership, or other relationship is inferred.",
            ),
        )
        return USGSemanticKnowledgeResource(
            concepts=tuple(concepts),
            concept_records=tuple(concept_records),
            categories=tuple(categories),
            memberships=tuple(memberships),
            relations=tuple(relations),
            predicate_frames=tuple(predicate_structures),
            namespace=namespace,
            bundle_id=bundle_id,
            bundle_digest=digest,
            report=report,
        )


class USGSenseToConceptAdapter:
    """Focused adapter for USG lexical senses and concepts."""
    def __init__(self, *, include_candidate_states: bool = False) -> None:
        self._adapter = USGSemanticKnowledgeAdapter(include_candidate_states=include_candidate_states)
    def convert(self, bundle: Mapping[str, Any]) -> tuple[USGSemanticConceptRecord, ...]:
        return self._adapter.convert(bundle).concept_records


class USGSemanticCategoryAdapter:
    """Focused adapter for semantic categories and explicit memberships."""
    def __init__(self, *, include_candidate_states: bool = False) -> None:
        self._adapter = USGSemanticKnowledgeAdapter(include_candidate_states=include_candidate_states)
    def convert(self, bundle: Mapping[str, Any]) -> tuple[tuple[SemanticCategory, ...], tuple[SemanticCategoryMembership, ...]]:
        resource = self._adapter.convert(bundle)
        return resource.categories, resource.memberships


class USGConceptRelationAdapter:
    """Focused adapter for explicit relations and concept mappings."""
    def __init__(self, *, include_candidate_states: bool = False) -> None:
        self._adapter = USGSemanticKnowledgeAdapter(include_candidate_states=include_candidate_states)
    def convert(self, bundle: Mapping[str, Any]) -> tuple[SemanticRelation, ...]:
        return self._adapter.convert(bundle).relations


class USGPredicateFrameAdapter:
    """Focused adapter for complete, incomplete, and unresolved predicate frames."""
    def __init__(self, *, include_candidate_states: bool = False) -> None:
        self._adapter = USGSemanticKnowledgeAdapter(include_candidate_states=include_candidate_states)
    def convert(self, bundle: Mapping[str, Any]) -> tuple[USGPredicateStructure, ...]:
        return self._adapter.convert(bundle).predicate_frames


__all__ = [
    "USGSemanticConceptRecord", "USGPredicateRole", "USGPredicateStructure",
    "USGSemanticKnowledgeImportReport", "USGSemanticKnowledgeResource",
    "USGSemanticKnowledgeAdapter", "USGSenseToConceptAdapter",
    "USGSemanticCategoryAdapter", "USGConceptRelationAdapter", "USGPredicateFrameAdapter",
]
