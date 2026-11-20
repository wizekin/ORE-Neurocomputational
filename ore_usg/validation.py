"""Dependency-light validation for the USG-KNOWLEDGE/0.1.0 interchange profile."""
from __future__ import annotations

from datetime import datetime
import unicodedata
from typing import Any, Mapping, Sequence

from .contracts import USG_KNOWLEDGE_CONTRACT


_STATES = {
    "CANDIDATE", "VALIDATED_FOR_PROFILE", "ACCEPTED",
    "DEPRECATED", "REJECTED", "UNRESOLVED",
}
_RECORD_KINDS = {
    "LEXICAL_FORM", "LEMMA", "LEXICAL_SENSE",
    "CONCEPT", "SEMANTIC_CLASS", "PREDICATE_FRAME",
}
_REQUIRED_PAYLOAD = {
    "LEXICAL_FORM": ("form", "language", "script"),
    "LEMMA": ("lemma", "language", "part_of_speech"),
    "LEXICAL_SENSE": ("lemma_record_id", "language", "definition", "sense_source_id"),
    "CONCEPT": ("concept_type", "namespace", "label"),
    "SEMANTIC_CLASS": ("class_key", "definition"),
    "PREDICATE_FRAME": ("frame_key", "version", "roles"),
}
_RIGHTS = {"KNOWN_PERMITTED", "RESTRICTED", "UNKNOWN", "PROHIBITED", "INTERNAL"}
_REDISTRIBUTION = {"ALLOWED", "RESTRICTED", "UNKNOWN", "PROHIBITED", "NOT_APPLICABLE"}
_SOURCE_TYPES = {"DICTIONARY", "LEXICAL_DATABASE", "CORPUS", "USG_DOCUMENT", "MODEL_PROPOSAL", "OTHER"}
_EVIDENCE_TYPES = {"DEFINITION", "EXAMPLE", "ATTESTATION", "RELATION_OBSERVATION", "CANONICAL_DOCUMENT", "MODEL_SUGGESTION", "OTHER"}
_SUPPORT_STATES = {"SUPPORTS", "CONTRADICTS", "NEUTRAL", "UNASSESSED"}
_CONTEXT_TYPES = {"LANGUAGE", "DOMAIN", "DISCOURSE", "TIME", "TASK", "REGISTER", "REGION", "SOURCE"}
_MAPPING_TYPES = {"EQUIVALENT", "OVERLAPPING", "NARROWER_THAN", "BROADER_THAN", "RELATED_TO", "UNRESOLVED"}


class USGKnowledgeValidationError(ValueError):
    """Raised when an imported bundle violates the supported USG knowledge contract."""

    def __init__(self, errors: Sequence[str]) -> None:
        self.errors = tuple(errors)
        super().__init__("USG knowledge bundle validation failed: " + "; ".join(self.errors))


def _is_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_id(value: Any) -> bool:
    """Return whether value is a stable Unicode identifier under the USG profile.

    Identifiers start with a Unicode letter or number. Subsequent characters may
    be letters, numbers, combining marks attached to a preceding letter/number,
    or the ASCII separators already supported by the original profile.
    """
    if not _is_text(value):
        return False
    first_category = unicodedata.category(value[0])
    if first_category[0] not in {"L", "N"}:
        return False

    has_base_character = True
    for character in value[1:]:
        category = unicodedata.category(character)
        if category[0] in {"L", "N"}:
            has_base_character = True
        elif category[0] == "M" and has_base_character:
            # Combining marks belong to the immediately preceding base sequence.
            has_base_character = True
        elif character in "._:/@+-":
            has_base_character = False
        else:
            return False
    return True


def _is_timestamp(value: Any) -> bool:
    if not _is_text(value):
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


def _rows(value: Any, label: str, errors: list[str]) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        errors.append(f"{label} must be an array")
        return []
    output: list[Mapping[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            errors.append(f"{label}[{index}] must be an object")
        else:
            output.append(item)
    return output


def _require_fields(
    item: Mapping[str, Any], fields: Sequence[str], label: str, errors: list[str]
) -> None:
    for field in fields:
        if field not in item:
            errors.append(f"{label} is missing required field {field}")


def _validate_id_list(
    value: Any, label: str, known_ids: set[str] | None, errors: list[str],
    *, minimum: int = 0,
) -> None:
    if not isinstance(value, list):
        errors.append(f"{label} must be an array")
        return
    if len(value) < minimum:
        errors.append(f"{label} must contain at least {minimum} item(s)")
    valid_ids = [item for item in value if _is_id(item)]
    if len(valid_ids) != len(set(valid_ids)):
        errors.append(f"{label} must not contain duplicate IDs")
    for item in value:
        if not _is_id(item):
            errors.append(f"{label} contains an invalid ID: {item!r}")
        elif known_ids is not None and item not in known_ids:
            errors.append(f"{label} references unknown ID {item}")


def _unique_ids(rows: Sequence[Mapping[str, Any]], label: str, errors: list[str]) -> set[str]:
    seen: set[str] = set()
    for index, item in enumerate(rows):
        item_id = item.get("id")
        if not _is_id(item_id):
            errors.append(f"{label}[{index}].id is not a valid non-empty ID")
        elif item_id in seen:
            errors.append(f"{label} contains duplicate ID {item_id}")
        else:
            seen.add(item_id)
    return seen


def validate_knowledge_bundle(bundle: Mapping[str, Any]) -> tuple[str, ...]:
    """Validate required fields, IDs, and cross-references used by the lexical adapter.

    This validator covers the structural and referential rules needed by ORE.
    It deliberately does not replace USG-Workbench's full Draft 2020-12
    JSON-Schema validator and does not evaluate whether a lexical claim is true.
    """
    errors: list[str] = []
    if not isinstance(bundle, Mapping):
        return ("bundle must be an object",)

    required_top = (
        "contract", "namespace", "bundle_id", "created_at", "records",
        "sources", "evidence", "contexts", "mappings", "relations",
    )
    _require_fields(bundle, required_top, "bundle", errors)
    if bundle.get("contract") != USG_KNOWLEDGE_CONTRACT:
        errors.append(f"contract must equal {USG_KNOWLEDGE_CONTRACT}")
    for field in ("namespace", "bundle_id"):
        if not _is_text(bundle.get(field)):
            errors.append(f"{field} must be non-empty text")
    if not _is_timestamp(bundle.get("created_at")):
        errors.append("created_at must be a timezone-aware date-time")

    records = _rows(bundle.get("records"), "records", errors)
    sources = _rows(bundle.get("sources"), "sources", errors)
    evidence = _rows(bundle.get("evidence"), "evidence", errors)
    contexts = _rows(bundle.get("contexts"), "contexts", errors)
    mappings = _rows(bundle.get("mappings"), "mappings", errors)
    relations = _rows(bundle.get("relations"), "relations", errors)

    record_ids = _unique_ids(records, "records", errors)
    source_ids = _unique_ids(sources, "sources", errors)
    evidence_ids = _unique_ids(evidence, "evidence", errors)
    context_ids = _unique_ids(contexts, "contexts", errors)
    _unique_ids(mappings, "mappings", errors)
    _unique_ids(relations, "relations", errors)

    for index, source in enumerate(sources):
        label = f"sources[{index}]"
        _require_fields(
            source,
            ("id", "provider", "source_type", "retrieved_at", "rights_status", "redistribution", "attribution"),
            label, errors,
        )
        if not _is_text(source.get("provider")) or not _is_text(source.get("attribution")):
            errors.append(f"{label} provider and attribution must be non-empty")
        source_type = source.get("source_type")
        rights_status = source.get("rights_status")
        redistribution = source.get("redistribution")
        if not isinstance(source_type, str) or source_type not in _SOURCE_TYPES:
            errors.append(f"{label} has unsupported source_type")
        if not isinstance(rights_status, str) or rights_status not in _RIGHTS:
            errors.append(f"{label} has unsupported rights_status")
        if not isinstance(redistribution, str) or redistribution not in _REDISTRIBUTION:
            errors.append(f"{label} has unsupported redistribution")
        if not _is_timestamp(source.get("retrieved_at")):
            errors.append(f"{label} retrieved_at must be a timezone-aware date-time")
        if (
            isinstance(rights_status, str)
            and rights_status in {"RESTRICTED", "UNKNOWN", "PROHIBITED"}
            and redistribution == "ALLOWED"
        ):
            errors.append(f"{label} cannot allow redistribution when source rights are {rights_status}")

    for index, item in enumerate(evidence):
        label = f"evidence[{index}]"
        _require_fields(
            item, ("id", "source_id", "evidence_type", "support_status", "captured_at", "transformation_ids"),
            label, errors,
        )
        source_id = item.get("source_id")
        if not _is_id(source_id) or source_id not in source_ids:
            errors.append(f"{label} references unknown source_id {source_id}")
        evidence_type = item.get("evidence_type")
        support_status = item.get("support_status")
        if not isinstance(evidence_type, str) or evidence_type not in _EVIDENCE_TYPES:
            errors.append(f"{label} has unsupported evidence_type")
        if not isinstance(support_status, str) or support_status not in _SUPPORT_STATES:
            errors.append(f"{label} has unsupported support_status")
        if not _is_timestamp(item.get("captured_at")):
            errors.append(f"{label} captured_at must be a timezone-aware date-time")
        _validate_id_list(item.get("transformation_ids"), f"{label}.transformation_ids", None, errors)

    for index, context in enumerate(contexts):
        label = f"contexts[{index}]"
        _require_fields(context, ("id", "context_type", "value"), label, errors)
        context_type = context.get("context_type")
        if (
            not isinstance(context_type, str)
            or context_type not in _CONTEXT_TYPES
            or not _is_text(context.get("value"))
        ):
            errors.append(f"{label} has invalid context_type or empty value")

    record_by_id = {item.get("id"): item for item in records if _is_id(item.get("id"))}
    for index, record in enumerate(records):
        label = f"records[{index}]"
        _require_fields(
            record, ("id", "kind", "version", "state", "created_at", "source_ids", "evidence_ids", "payload"),
            label, errors,
        )
        kind = record.get("kind")
        if not isinstance(kind, str) or kind not in _RECORD_KINDS:
            errors.append(f"{label} has unsupported kind {kind!r}")
        state = record.get("state")
        if not isinstance(state, str) or state not in _STATES:
            errors.append(f"{label} has unsupported lifecycle state")
        version = record.get("version")
        if not isinstance(version, int) or isinstance(version, bool) or version < 1:
            errors.append(f"{label}.version must be a positive integer")
        if not _is_timestamp(record.get("created_at")):
            errors.append(f"{label}.created_at must be a timezone-aware date-time")
        _validate_id_list(record.get("source_ids"), f"{label}.source_ids", source_ids, errors)
        _validate_id_list(record.get("evidence_ids"), f"{label}.evidence_ids", evidence_ids, errors)
        if "context_ids" in record:
            _validate_id_list(record.get("context_ids"), f"{label}.context_ids", context_ids, errors)
        payload = record.get("payload")
        if not isinstance(payload, Mapping) or not payload:
            errors.append(f"{label}.payload must be a non-empty object")
        elif isinstance(kind, str) and kind in _REQUIRED_PAYLOAD:
            for field in _REQUIRED_PAYLOAD[kind]:
                if field not in payload:
                    errors.append(f"{label}.payload is missing required field {field}")

        if isinstance(payload, Mapping) and isinstance(kind, str) and kind in {
            "LEXICAL_FORM", "LEMMA", "LEXICAL_SENSE",
        }:
            text_fields = {
                "LEXICAL_FORM": ("form", "language", "script"),
                "LEMMA": ("lemma", "language", "part_of_speech"),
                "LEXICAL_SENSE": ("lemma_record_id", "language", "definition", "sense_source_id"),
            }[kind]
            for field in text_fields:
                if not _is_text(payload.get(field)):
                    errors.append(f"{label}.payload.{field} must be non-empty text")

        if kind == "LEXICAL_SENSE" and isinstance(payload, Mapping):
            lemma_id = payload.get("lemma_record_id")
            lemma = record_by_id.get(lemma_id) if _is_id(lemma_id) else None
            if lemma is None or lemma.get("kind") != "LEMMA":
                errors.append(f"{label}.payload.lemma_record_id must reference a LEMMA record")
            sense_source_id = payload.get("sense_source_id")
            if not _is_id(sense_source_id) or sense_source_id not in source_ids:
                errors.append(f"{label}.payload.sense_source_id must reference a source")

    for index, mapping in enumerate(mappings):
        label = f"mappings[{index}]"
        _require_fields(
            mapping,
            ("id", "source_record_id", "target_record_id", "mapping_type", "state", "evidence_ids", "context_ids"),
            label, errors,
        )
        source_record_id = mapping.get("source_record_id")
        target_record_id = mapping.get("target_record_id")
        if (
            not _is_id(source_record_id) or source_record_id not in record_ids
            or not _is_id(target_record_id) or target_record_id not in record_ids
        ):
            errors.append(f"{label} references an unknown record")
        if _is_id(source_record_id) and source_record_id == target_record_id:
            errors.append(f"{label} cannot map a record to itself")
        mapping_type = mapping.get("mapping_type")
        state = mapping.get("state")
        if (
            not isinstance(mapping_type, str) or mapping_type not in _MAPPING_TYPES
            or not isinstance(state, str) or state not in _STATES
        ):
            errors.append(f"{label} has unsupported mapping_type or state")
        _validate_id_list(mapping.get("evidence_ids"), f"{label}.evidence_ids", evidence_ids, errors, minimum=1)
        _validate_id_list(mapping.get("context_ids"), f"{label}.context_ids", context_ids, errors)

    for index, relation in enumerate(relations):
        label = f"relations[{index}]"
        _require_fields(
            relation,
            ("id", "subject_record_id", "predicate", "object_record_id", "direction", "state", "evidence_ids", "context_ids"),
            label, errors,
        )
        subject_id = relation.get("subject_record_id")
        object_id = relation.get("object_record_id")
        if (
            not _is_id(subject_id) or subject_id not in record_ids
            or not _is_id(object_id) or object_id not in record_ids
        ):
            errors.append(f"{label} references an unknown record")
        if _is_id(subject_id) and subject_id == object_id:
            errors.append(f"{label} cannot be a self-relation")
        if not _is_text(relation.get("predicate")) or relation.get("direction") != "DIRECTED":
            errors.append(f"{label} has invalid predicate or direction")
        state = relation.get("state")
        if not isinstance(state, str) or state not in _STATES:
            errors.append(f"{label} has unsupported state")
        _validate_id_list(relation.get("evidence_ids"), f"{label}.evidence_ids", evidence_ids, errors, minimum=1)
        _validate_id_list(relation.get("context_ids"), f"{label}.context_ids", context_ids, errors)

    return tuple(sorted(set(errors)))


def require_valid_knowledge_bundle(bundle: Mapping[str, Any]) -> None:
    """Raise one stable exception when the bundle fails supported structural checks."""
    errors = validate_knowledge_bundle(bundle)
    if errors:
        raise USGKnowledgeValidationError(errors)


__all__ = [
    "USGKnowledgeValidationError",
    "validate_knowledge_bundle",
    "require_valid_knowledge_bundle",
]
