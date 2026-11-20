"""Read-only conversion of USG concepts and explicit concept mappings."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Mapping
from urllib.parse import quote

from ore_semantic.contracts import SemanticConcept

from .contracts import USGImportExclusion
from .lexical import (
    _canonical_json,
    _ore_record_id,
    _positive_number,
    _source_reference,
    _source_ids_for_record,
)
from .validation import require_valid_knowledge_bundle
from .policy import USGImportPolicy


@dataclass(frozen=True, slots=True)
class USGConceptMapping:
    """An explicit, typed mapping between two imported ORE concepts."""

    mapping_id: str
    source_concept_id: str
    target_concept_id: str
    mapping_type: str
    lifecycle_state: str
    evidence_ids: tuple[str, ...]
    context_ids: tuple[str, ...]
    source_reference: str


@dataclass(frozen=True, slots=True)
class USGSemanticImportReport:
    """Explain concept/mapping import and exclusion decisions."""

    bundle_id: str
    concepts_created: int
    mappings_created: int
    imported_record_ids: tuple[str, ...]
    excluded_records: tuple[USGImportExclusion, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class USGSemanticResource:
    """A namespaced semantic snapshot; mappings never collapse concepts."""

    concepts: tuple[SemanticConcept, ...]
    mappings: tuple[USGConceptMapping, ...]
    bundle_id: str
    namespace: str
    bundle_digest: str
    report: USGSemanticImportReport


class USGConceptAdapter:
    """Import explicit USG concepts and mappings without inferring semantic features."""

    def __init__(self, *, include_candidate_states: bool = False) -> None:
        if not isinstance(include_candidate_states, bool):
            raise TypeError("include_candidate_states must be a bool")
        self.include_candidate_states = include_candidate_states
        self.policy = USGImportPolicy(include_candidate_states=include_candidate_states)

    def _state_enabled(self, state: str) -> bool:
        return self.policy.state_enabled(state)

    def convert(self, bundle: Mapping[str, Any]) -> USGSemanticResource:
        """Convert eligible CONCEPT records and explicit mappings between them."""
        require_valid_knowledge_bundle(bundle)
        namespace = str(bundle["namespace"])
        bundle_id = str(bundle["bundle_id"])
        digest = sha256(_canonical_json(bundle)).hexdigest()
        source_by_id = {item["id"]: item for item in bundle["sources"]}
        evidence_by_id = {item["id"]: item for item in bundle["evidence"]}

        concepts: list[SemanticConcept] = []
        excluded: list[USGImportExclusion] = []
        imported: set[str] = set()
        concept_ids: dict[str, str] = {}
        concept_records = [
            row for row in bundle["records"] if row.get("kind") == "CONCEPT"
        ]
        for record in sorted(concept_records, key=lambda row: str(row["id"])):
            state = str(record["state"])
            if not self._state_enabled(state):
                excluded.append(USGImportExclusion(
                    str(record["id"]), "CONCEPT", f"lifecycle_state_not_enabled:{state}",
                ))
                continue
            usable, reason, source_ids = self.policy.source_access(record, source_by_id, evidence_by_id)
            if not usable:
                excluded.append(USGImportExclusion(str(record["id"]), "CONCEPT", reason))
                continue
            payload = record["payload"]
            aliases_value = payload.get("aliases", ())
            aliases = tuple(
                item for item in aliases_value
                if isinstance(item, str) and item.strip()
            ) if isinstance(aliases_value, (list, tuple)) else ()
            language = payload.get("language", "und")
            category = payload.get("lexical_category", "unknown")
            if not isinstance(language, str) or not language.strip():
                language = "und"
            if not isinstance(category, str) or not category.strip():
                category = "unknown"
            ore_id = _ore_record_id(namespace, bundle_id, record)
            concept_ids[str(record["id"])] = ore_id
            concepts.append(SemanticConcept(
                concept_id=ore_id,
                label=str(payload["label"]),
                aliases=aliases,
                semantic_features={},
                modality_features={},
                lexical_category=category,
                frequency=_positive_number(payload.get("frequency"), 1.0),
                prototype_id=None,
                language=language,
                verified_by=(
                    f"USG_KNOWLEDGE_STATE:{state};"
                    "not_an_ORE_truth_verification"
                ),
                source_reference=_source_reference(
                    namespace, bundle_id, record, source_ids,
                ),
            ))
            imported.add(str(record["id"]))

        mappings: list[USGConceptMapping] = []
        for mapping in sorted(bundle["mappings"], key=lambda row: str(row["id"])):
            source_record_id = str(mapping["source_record_id"])
            target_record_id = str(mapping["target_record_id"])
            if (
                source_record_id not in concept_ids
                or target_record_id not in concept_ids
            ):
                excluded.append(USGImportExclusion(
                    str(mapping["id"]), "MAPPING",
                    "both_mapping_endpoints_must_be_importable_concepts",
                ))
                continue
            state = str(mapping["state"])
            if not self._state_enabled(state):
                excluded.append(USGImportExclusion(
                    str(mapping["id"]), "MAPPING",
                    f"lifecycle_state_not_enabled:{state}",
                ))
                continue
            usable, reason, source_ids = self.policy.source_access(mapping, source_by_id, evidence_by_id)
            if not usable:
                excluded.append(USGImportExclusion(
                    str(mapping["id"]), "MAPPING", reason,
                ))
                continue
            mapping_scope = sha256(_canonical_json({
                "namespace": namespace, "bundle_id": bundle_id,
            })).hexdigest()
            mapping_id = (
                f"usg-map:{mapping_scope}:{quote(str(mapping['id']), safe='')}"
            )
            source_reference = (
                f"usg-knowledge://{quote(namespace, safe='')}/"
                f"{quote(bundle_id, safe='')}/mappings/"
                f"{quote(str(mapping['id']), safe='')}"
                f"?state={quote(state, safe='')}"
                f"&evidence={','.join(quote(str(v), safe='') for v in mapping['evidence_ids'])}"
                f"&contexts={','.join(quote(str(v), safe='') for v in mapping['context_ids'])}"
                f"&sources={','.join(quote(v, safe='') for v in source_ids)}"
            )
            mappings.append(USGConceptMapping(
                mapping_id=mapping_id,
                source_concept_id=concept_ids[source_record_id],
                target_concept_id=concept_ids[target_record_id],
                mapping_type=str(mapping["mapping_type"]),
                lifecycle_state=state,
                evidence_ids=tuple(str(v) for v in mapping["evidence_ids"]),
                context_ids=tuple(str(v) for v in mapping["context_ids"]),
                source_reference=source_reference,
            ))
            imported.add(str(mapping["id"]))

        warnings = (
            "No numeric semantic or modality features are inferred from concept labels or definitions.",
            "Mappings preserve their declared type; EQUIVALENT and RELATED_TO never merge concept records.",
            "USG lifecycle states do not establish semantic truth or ORE verification.",
            "Only mappings whose endpoints are both imported concepts are exposed.",
        )
        concepts.sort(key=lambda item: item.concept_id)
        mappings.sort(key=lambda item: item.mapping_id)
        report = USGSemanticImportReport(
            bundle_id=bundle_id,
            concepts_created=len(concepts),
            mappings_created=len(mappings),
            imported_record_ids=tuple(sorted(imported)),
            excluded_records=tuple(sorted(
                excluded, key=lambda item: (item.record_id, item.reason),
            )),
            warnings=warnings,
        )
        return USGSemanticResource(
            concepts=tuple(concepts),
            mappings=tuple(mappings),
            bundle_id=bundle_id,
            namespace=namespace,
            bundle_digest=digest,
            report=report,
        )


__all__ = [
    "USGConceptAdapter",
    "USGConceptMapping",
    "USGSemanticImportReport",
    "USGSemanticResource",
]
