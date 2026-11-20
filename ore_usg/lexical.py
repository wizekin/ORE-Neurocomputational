"""Read-only conversion from USG-KNOWLEDGE bundles to ORE lexical records."""
from __future__ import annotations

from hashlib import sha256
import json
import math
from typing import Any, Mapping
from urllib.parse import quote

from ore_lexical_context.contracts import LexicalEntry, LexicalMeaning

from .contracts import (
    USG_LEXICAL_ADAPTER_VERSION,
    USGLexiconImportReport,
    USGLexiconResource,
    USGContextSummary,
    USGEvidenceSummary,
    USGImportExclusion,
    USGSourceSummary,
)
from .validation import require_valid_knowledge_bundle
from .policy import LOCAL_IMPORT_RIGHTS, USGImportPolicy


_LOCAL_USE_RIGHTS = LOCAL_IMPORT_RIGHTS
_POS_MAP = {
    "NOUN": "noun", "PROPN": "proper_noun", "VERB": "verb", "AUX": "auxiliary",
    "ADJ": "adjective", "ADV": "adverb", "PRON": "pronoun", "DET": "determiner",
    "ADP": "adposition", "NUM": "numeral", "PART": "particle",
    "CCONJ": "coordinating_conjunction",
    "SCONJ": "subordinating_conjunction",
    "INTJ": "interjection",
    "PUNCT": "punctuation",
}


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")


def _positive_number(value: Any, default: float = 1.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) and number > 0.0 else default


def _lexical_category(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        return "unknown"
    normalized = value.strip().upper()
    return _POS_MAP.get(normalized, normalized.casefold())


def _source_ids_for_record(
    record: Mapping[str, Any],
    evidence_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[str, ...]:
    source_ids = set(record.get("source_ids", ()))
    for evidence_id in record.get("evidence_ids", ()):
        evidence = evidence_by_id.get(evidence_id)
        if evidence is not None:
            source_id = evidence.get("source_id")
            if isinstance(source_id, str) and source_id:
                source_ids.add(source_id)
    payload = record.get("payload", {})
    if record.get("kind") == "LEXICAL_SENSE" and isinstance(payload, Mapping):
        sense_source_id = payload.get("sense_source_id")
        if isinstance(sense_source_id, str) and sense_source_id:
            source_ids.add(sense_source_id)
    return tuple(sorted(source_ids))


def _ore_record_id(namespace: str, bundle_id: str, record: Mapping[str, Any]) -> str:
    """Scope an ORE lexical ID to its USG namespace, bundle, record, and version."""
    scope = sha256(_canonical_json({"namespace": namespace, "bundle_id": bundle_id})).hexdigest()
    record_id = quote(str(record["id"]), safe="")
    return f"usg:{scope}:{record_id}@{int(record['version'])}"


def _source_reference(
    namespace: str,
    bundle_id: str,
    record: Mapping[str, Any],
    source_ids: tuple[str, ...],
) -> str:
    """Build a stable locator containing record, lifecycle, and source provenance."""
    record_id = quote(str(record["id"]), safe="")
    source_list = ",".join(quote(item, safe="") for item in source_ids)
    evidence_list = ",".join(quote(str(item), safe="") for item in record.get("evidence_ids", ()))
    context_list = ",".join(quote(str(item), safe="") for item in record.get("context_ids", ()))
    return (
        f"usg-knowledge://{quote(namespace, safe='')}/{quote(bundle_id, safe='')}/records/{record_id}"
        f"?version={int(record['version'])}&state={quote(str(record['state']), safe='')}"
        f"&sources={source_list}&evidence={evidence_list}&contexts={context_list}"
    )


class USGKnowledgeLexicalAdapter:
    """Convert a USG knowledge snapshot into an ORE lexicon without mutating either side.

    ACCEPTED and VALIDATED_FOR_PROFILE records are enabled by default. CANDIDATE
    records can be included only through explicit opt-in. UNRESOLVED, DEPRECATED,
    and REJECTED records are never imported as active lexical entries.
    """

    def __init__(self, *, include_candidate_states: bool = False) -> None:
        if not isinstance(include_candidate_states, bool):
            raise TypeError("include_candidate_states must be a bool")
        self.include_candidate_states = include_candidate_states
        self.policy = USGImportPolicy(include_candidate_states=include_candidate_states)

    @property
    def version(self) -> str:
        """Return the adapter's stable semantic version."""
        return USG_LEXICAL_ADAPTER_VERSION

    def _state_enabled(self, state: str) -> bool:
        return self.policy.state_enabled(state)

    def _source_access(
        self,
        record: Mapping[str, Any],
        source_by_id: Mapping[str, Mapping[str, Any]],
        evidence_by_id: Mapping[str, Mapping[str, Any]],
    ) -> tuple[bool, str, tuple[str, ...]]:
        """Apply the shared lifecycle-independent source-rights policy."""
        return self.policy.source_access(record, source_by_id, evidence_by_id)

    def convert(self, bundle: Mapping[str, Any]) -> USGLexiconResource:
        """Validate and convert lexical forms, lemmas, and senses into ORE records."""
        require_valid_knowledge_bundle(bundle)
        bundle_digest = sha256(_canonical_json(bundle)).hexdigest()
        namespace = str(bundle["namespace"])
        bundle_id = str(bundle["bundle_id"])

        records = tuple(bundle["records"])
        source_by_id = {item["id"]: item for item in bundle["sources"]}
        evidence_by_id = {item["id"]: item for item in bundle["evidence"]}

        excluded: list[USGImportExclusion] = []
        imported_record_ids: set[str] = set()
        warnings: list[str] = []
        source_summaries = tuple(
            USGSourceSummary(
                source_id=str(item["id"]),
                provider=str(item["provider"]),
                rights_status=str(item["rights_status"]),
                redistribution=str(item["redistribution"]),
                attribution=str(item["attribution"]),
                usable_for_local_import=item["rights_status"] in _LOCAL_USE_RIGHTS,
                license_id=item.get("license_id"),
                source_locator=item.get("source_locator"),
                retrieved_at=str(item["retrieved_at"]),
                content_digest=item.get("content_digest"),
            )
            for item in sorted(bundle["sources"], key=lambda row: row["id"])
        )

        evidence_summaries = tuple(
            USGEvidenceSummary(
                evidence_id=str(item["id"]),
                source_id=str(item["source_id"]),
                evidence_type=str(item["evidence_type"]),
                support_status=str(item["support_status"]),
                locator=item.get("locator"),
                text_digest=item.get("text_digest"),
                captured_at=str(item["captured_at"]),
                transformation_ids=tuple(str(value) for value in item["transformation_ids"]),
            )
            for item in sorted(bundle["evidence"], key=lambda row: row["id"])
        )
        context_summaries = tuple(
            USGContextSummary(
                context_id=str(item["id"]),
                context_type=str(item["context_type"]),
                value=str(item["value"]),
                language_profile=item.get("language_profile"),
                scope=item.get("scope"),
            )
            for item in sorted(bundle["contexts"], key=lambda row: row["id"])
        )

        eligible_lemmas: dict[str, Mapping[str, Any]] = {}
        for record in records:
            if record["kind"] != "LEMMA":
                continue
            state = str(record["state"])
            if not self._state_enabled(state):
                excluded.append(USGImportExclusion(
                    str(record["id"]), "LEMMA", f"lifecycle_state_not_enabled:{state}",
                ))
                continue
            usable, reason, _ = self._source_access(record, source_by_id, evidence_by_id)
            if not usable:
                excluded.append(USGImportExclusion(str(record["id"]), "LEMMA", reason))
                continue
            eligible_lemmas[str(record["id"])] = record

        senses_by_lemma: dict[str, list[tuple[Mapping[str, Any], tuple[str, ...]]]] = {}
        for record in records:
            if record["kind"] != "LEXICAL_SENSE":
                continue
            state = str(record["state"])
            if not self._state_enabled(state):
                excluded.append(USGImportExclusion(
                    str(record["id"]), "LEXICAL_SENSE", f"lifecycle_state_not_enabled:{state}",
                ))
                continue
            lemma_id = str(record["payload"]["lemma_record_id"])
            if lemma_id not in eligible_lemmas:
                excluded.append(USGImportExclusion(
                    str(record["id"]), "LEXICAL_SENSE", "referenced_lemma_not_importable",
                ))
                continue
            lemma_language = str(eligible_lemmas[lemma_id]["payload"]["language"])
            if str(record["payload"]["language"]) != lemma_language:
                excluded.append(USGImportExclusion(
                    str(record["id"]), "LEXICAL_SENSE", "sense_language_differs_from_lemma_language",
                ))
                continue
            usable, reason, source_ids = self._source_access(record, source_by_id, evidence_by_id)
            if not usable:
                excluded.append(USGImportExclusion(str(record["id"]), "LEXICAL_SENSE", reason))
                continue
            senses_by_lemma.setdefault(lemma_id, []).append((record, source_ids))

        entries: list[LexicalEntry] = []
        meanings_created = 0
        for lemma_id in sorted(eligible_lemmas):
            record = eligible_lemmas[lemma_id]
            payload = record["payload"]
            lemma_senses = sorted(
                senses_by_lemma.get(lemma_id, ()), key=lambda pair: str(pair[0]["id"])
            )
            meanings = tuple(
                LexicalMeaning(
                    meaning_id=_ore_record_id(namespace, bundle_id, sense),
                    features={},
                    prior=_positive_number(sense["payload"].get("prior"), 1.0),
                    gloss=str(sense["payload"]["definition"]),
                    verified_by=(
                        f"USG_KNOWLEDGE_STATE:{sense['state']};"
                        "not_an_ORE_truth_verification"
                    ),
                    source_reference=_source_reference(
                        namespace, bundle_id, sense, sense_sources,
                    ),
                )
                for sense, sense_sources in lemma_senses
            )
            source_ids = _source_ids_for_record(record, evidence_by_id)
            entries.append(LexicalEntry(
                entry_id=_ore_record_id(namespace, bundle_id, record),
                orthography=str(payload["lemma"]),
                frequency=_positive_number(payload.get("frequency"), 1.0),
                meanings=meanings,
                lexical_category=_lexical_category(payload["part_of_speech"]),
                language=str(payload["language"]),
                verified_by=(
                    f"USG_KNOWLEDGE_STATE:{record['state']};"
                    "not_an_ORE_truth_verification"
                ),
                source_reference=_source_reference(namespace, bundle_id, record, source_ids),
            ))
            imported_record_ids.add(lemma_id)
            imported_record_ids.update(str(sense["id"]) for sense, _ in lemma_senses)
            meanings_created += len(meanings)

        lemma_forms = {
            (entry.language.casefold(), entry.orthography.casefold())
            for entry in entries
        }
        for record in records:
            if record["kind"] != "LEXICAL_FORM":
                continue
            state = str(record["state"])
            if not self._state_enabled(state):
                excluded.append(USGImportExclusion(
                    str(record["id"]), "LEXICAL_FORM", f"lifecycle_state_not_enabled:{state}",
                ))
                continue
            usable, reason, source_ids = self._source_access(record, source_by_id, evidence_by_id)
            if not usable:
                excluded.append(USGImportExclusion(str(record["id"]), "LEXICAL_FORM", reason))
                continue
            payload = record["payload"]
            form = str(payload["form"])
            language = str(payload["language"])
            duplicate_key = (language.casefold(), form.casefold())
            if duplicate_key in lemma_forms:
                excluded.append(USGImportExclusion(
                    str(record["id"]), "LEXICAL_FORM", "surface_form_already_represented_by_lemma",
                ))
                continue
            entries.append(LexicalEntry(
                entry_id=_ore_record_id(namespace, bundle_id, record),
                orthography=form,
                meanings=(),
                lexical_category="unknown",
                language=language,
                verified_by=(
                    f"USG_KNOWLEDGE_STATE:{state};"
                    "surface_form_only_no_lemma_or_sense_inferred"
                ),
                source_reference=_source_reference(namespace, bundle_id, record, source_ids),
            ))
            imported_record_ids.add(str(record["id"]))
            lemma_forms.add(duplicate_key)

        if any(
            _positive_number(item["payload"].get("frequency"), 0.0) == 0.0
            for item in eligible_lemmas.values()
        ):
            warnings.append(
                "Missing or invalid lemma frequency uses ORE's neutral default of 1.0; it is not an observed frequency."
            )
        warnings.append("USG definitions are preserved as lexical glosses; no numeric semantic features are inferred.")
        warnings.append("Lexical forms without an explicit lemma link remain surface-form-only entries.")
        warnings.append("Importing a USG lifecycle state does not establish semantic truth or ORE verification.")

        entries.sort(key=lambda item: (
            item.language.casefold(), item.orthography.casefold(), item.entry_id,
        ))
        report = USGLexiconImportReport(
            bundle_id=bundle_id,
            records_seen=len(records),
            lexical_records_seen=sum(
                item["kind"] in {"LEXICAL_FORM", "LEMMA", "LEXICAL_SENSE"} for item in records
            ),
            entries_created=len(entries),
            meanings_created=meanings_created,
            imported_record_ids=tuple(sorted(imported_record_ids)),
            excluded_records=tuple(sorted(excluded, key=lambda item: (item.record_id, item.reason))),
            source_summaries=source_summaries,
            evidence_summaries=evidence_summaries,
            context_summaries=context_summaries,
            warnings=tuple(warnings),
        )
        return USGLexiconResource(
            entries=tuple(entries),
            bundle_id=bundle_id,
            namespace=namespace,
            bundle_digest=bundle_digest,
            adapter_version=self.version,
            report=report,
        )

    def to_lexicon(self, bundle: Mapping[str, Any]) -> tuple[LexicalEntry, ...]:
        """Return only the ORE LexicalEntry tuple for direct processor construction."""
        return self.convert(bundle).entries


__all__ = ["USGKnowledgeLexicalAdapter"]
