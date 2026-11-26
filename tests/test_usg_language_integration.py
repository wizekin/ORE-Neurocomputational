"""Local regression tests for ORE-USG Phase 3 language and meaning integration."""
from __future__ import annotations

from copy import deepcopy

from ore_lexical_access.contracts import LexicalAccessRequest
from ore_semantic_composition import SemanticCompositionConfig, SemanticCompositionRequest
from ore_usg import USGKnowledgeAdapter, USGKnowledgeIndex, build_semantic_composition_processor
from test_usg_lexical_adapter import _bundle


def _phase3_bundle():
    bundle = deepcopy(_bundle())
    timestamp = "2026-11-12T00:00:00Z"
    banks_form = next(row for row in bundle["records"] if row["id"] == "form:en:banks")
    banks_form["payload"]["grammatical_features"] = {
        "number": "plural", "case": "unspecified",
    }

    # Context descriptors retain language profile and scope; these are not
    # converted into synthetic numerical context features.
    for context in bundle["contexts"]:
        if context["id"] == "context:finance":
            context.update(language_profile="en", scope="domain:finance")
        elif context["id"] == "context:geography":
            context.update(language_profile="en", scope="domain:geography")

    bundle["records"].extend([
        {
            "id": "lemma:en:banker", "kind": "LEMMA", "version": 1,
            "state": "ACCEPTED", "created_at": timestamp,
            "source_ids": ["source:permitted"], "evidence_ids": ["evidence:lemma"],
            "context_ids": ["context:finance"],
            "payload": {"lemma": "banker", "language": "en", "part_of_speech": "NOUN"},
        },
        {
            "id": "form:en:banked", "kind": "LEXICAL_FORM", "version": 1,
            "state": "ACCEPTED", "created_at": timestamp,
            "source_ids": ["source:permitted"], "evidence_ids": ["evidence:form"],
            "context_ids": ["context:english"],
            "payload": {
                "form": "banked", "language": "en", "script": "Latn",
                "grammatical_features": {"tense": "past", "form": "participle"},
            },
        },
        {
            "id": "concept:money", "kind": "CONCEPT", "version": 1,
            "state": "ACCEPTED", "created_at": timestamp,
            "source_ids": ["source:permitted"], "evidence_ids": ["evidence:financial"],
            "context_ids": ["context:finance"],
            "payload": {
                "concept_type": "GENERAL", "namespace": "urn:usg:test",
                "label": "Money", "language": "en",
                "definition": "A medium of exchange and store of value.",
            },
        },
        {
            "id": "concept:finance", "kind": "CONCEPT", "version": 1,
            "state": "ACCEPTED", "created_at": timestamp,
            "source_ids": ["source:permitted"], "evidence_ids": ["evidence:financial"],
            "context_ids": ["context:finance"],
            "payload": {
                "concept_type": "GENERAL", "namespace": "urn:usg:test",
                "label": "Finance", "language": "en",
                "definition": "The management of money and assets.",
            },
        },
    ])
    bundle["evidence"].extend([
        {
            "id": "evidence:contradiction", "source_id": "source:permitted",
            "evidence_type": "RELATION_OBSERVATION", "support_status": "CONTRADICTS",
            "captured_at": timestamp, "transformation_ids": [],
        },
        {
            "id": "evidence:neutral", "source_id": "source:permitted",
            "evidence_type": "RELATION_OBSERVATION", "support_status": "NEUTRAL",
            "captured_at": timestamp, "transformation_ids": [],
        },
        {
            "id": "evidence:unassessed", "source_id": "source:permitted",
            "evidence_type": "MODEL_SUGGESTION", "support_status": "UNASSESSED",
            "captured_at": timestamp, "transformation_ids": [],
        },
    ])
    bundle["relations"].extend([
        {
            "id": "relation:financial-sense-denotes-money",
            "subject_record_id": "sense:en:bank:financial",
            "predicate": "DENOTES", "object_record_id": "concept:money",
            "direction": "DIRECTED", "state": "ACCEPTED",
            "evidence_ids": ["evidence:financial", "evidence:contradiction"],
            "context_ids": ["context:finance"],
        },
        {
            "id": "relation:river-sense-denotes-money",
            "subject_record_id": "sense:en:bank:river",
            "predicate": "DENOTES", "object_record_id": "concept:money",
            "direction": "DIRECTED", "state": "ACCEPTED",
            "evidence_ids": ["evidence:neutral"],
            "context_ids": ["context:geography"],
        },
        {
            "id": "relation:lemma-bank-denotes-finance",
            "subject_record_id": "lemma:en:bank",
            "predicate": "LEXICALIZES", "object_record_id": "concept:finance",
            "direction": "DIRECTED", "state": "ACCEPTED",
            "evidence_ids": ["evidence:unassessed"],
            "context_ids": ["context:finance"],
        },
        {
            "id": "relation:banks-inflected-form-of-bank",
            "subject_record_id": "form:en:banks",
            "predicate": "INFLECTED_FORM_OF", "object_record_id": "lemma:en:bank",
            "direction": "DIRECTED", "state": "ACCEPTED",
            "evidence_ids": ["evidence:form"],
            "context_ids": ["context:english"],
        },
        {
            "id": "relation:banks-inflected-form-of-banker",
            "subject_record_id": "form:en:banks",
            "predicate": "INFLECTED_FORM_OF", "object_record_id": "lemma:en:banker",
            "direction": "DIRECTED", "state": "ACCEPTED",
            "evidence_ids": ["evidence:neutral"],
            "context_ids": ["context:english"],
        },
        {
            "id": "relation:restricted-banked-lemma",
            "subject_record_id": "form:en:banked",
            "predicate": "INFLECTED_FORM_OF", "object_record_id": "lemma:en:banker",
            "direction": "DIRECTED", "state": "ACCEPTED",
            "evidence_ids": ["evidence:restricted"],
            "context_ids": ["context:english"],
        },
    ])
    bundle["mappings"].append({
        "id": "mapping:river-sense-related-finance",
        "source_record_id": "sense:en:bank:river",
        "target_record_id": "concept:finance",
        "mapping_type": "RELATED_TO", "state": "ACCEPTED",
        "evidence_ids": ["evidence:unassessed"],
        "context_ids": ["context:geography"],
    })
    return bundle


def test_lexical_senses_link_to_distinct_concepts_and_keep_ambiguity_and_conflicts():
    bundle = _phase3_bundle()
    original = deepcopy(bundle)
    resource = USGKnowledgeAdapter().convert(bundle)

    bank = next(item for item in resource.lexical.entries if item.orthography == "bank")
    meanings = {item.gloss: item for item in bank.meanings}
    financial = meanings["A financial institution."]
    river = meanings["The land alongside a river."]

    # The source sense itself is a direct identity link; other explicit USG
    # relations/mappings add one-to-many concept targets without merging them.
    assert len(financial.semantic_concept_ids) == 2
    assert len(river.semantic_concept_ids) == 3
    assert len(set(financial.semantic_concept_ids) | set(river.semantic_concept_ids)) == 4
    assert financial.semantic_link_status == "CONFLICTED"
    assert river.semantic_link_status == "AMBIGUOUS"
    assert len(bank.semantic_concept_ids) >= 4
    assert bank.semantic_link_status in {"AMBIGUOUS", "CONFLICTED"}

    links = resource.lexical_concept_links
    assert any(
        item.source_subject_record_id == "sense:en:bank:financial"
        and item.source_object_record_id == "concept:money"
        and item.status == "CONFLICTED"
        for item in links
    )
    assert any(
        item.source_subject_record_id == "sense:en:bank:river"
        and item.source_object_record_id == "concept:money"
        for item in links
    )
    assert any(
        item.source_kind == "SOURCE_RECORD_IDENTITY"
        and item.source_subject_record_id == item.source_object_record_id
        for item in links
    )

    index = USGKnowledgeIndex(resource)
    candidates = index.concepts_for_lexical_entry(bank.entry_id)
    assert len({item.concept_id for item in candidates}) >= 4
    assert bank.entry_id in {item.lexical_entry_id for item in index.lexical_concept_links_for(bank.entry_id)}
    assert bundle == original


def test_missing_and_ambiguous_lexical_links_are_explicit_abstentions():
    resource = USGKnowledgeAdapter().convert(_phase3_bundle())
    banked = next(item for item in resource.lexical.entries if item.orthography == "banked")
    banks = next(item for item in resource.lexical.entries if item.orthography == "banks")

    assert banked.semantic_link_status == "MISSING"
    assert banks.semantic_link_status in {"AMBIGUOUS", "CONFLICTED"}
    groups = resource.lexical_concept_groups
    assert any(
        item.lexical_entry_id == banked.entry_id and item.status == "MISSING"
        for item in groups
    )


def test_morphology_uses_only_explicit_relations_and_preserves_form_grammar():
    resource = USGKnowledgeAdapter().convert(_phase3_bundle())
    banks = next(item for item in resource.lexical.entries if item.orthography == "banks")
    banked = next(item for item in resource.lexical.entries if item.orthography == "banked")

    assert banks.morphology_link_status == "AMBIGUOUS"
    assert banks.lemma_entry_id is None  # two explicit USG targets; no arbitrary winner
    assert banks.script == "Latn"
    assert banks.grammatical_information_json is not None
    assert '"number":"plural"' in banks.grammatical_information_json
    assert banked.morphology_link_status == "MISSING"
    assert banked.lemma_entry_id is None

    morphology = USGKnowledgeIndex(resource).morphology_for_form(banks.entry_id)
    assert {item.lemma_record_id for item in morphology} == {
        "lemma:en:bank", "lemma:en:banker",
    }
    assert all(item.direction == "FORM_TO_LEMMA" for item in morphology)
    assert all(item.source_subject_record_id == item.form_record_id for item in morphology)


def test_context_and_evidence_are_attached_with_source_rights_not_fact_claims():
    resource = USGKnowledgeAdapter().convert(_phase3_bundle())
    bank = next(item for item in resource.lexical.entries if item.orthography == "bank")
    financial = next(item for item in bank.meanings if item.gloss == "A financial institution.")

    contexts = resource.contexts_for(financial.meaning_id)
    assert {item.context_type for item in contexts} == {"DOMAIN"}
    assert any(
        item.value == "finance" and item.language_profile == "en"
        and item.scope == "domain:finance"
        for item in contexts
    )
    evidence = resource.evidence_for(financial.meaning_id)
    assert {item.support_status for item in evidence} >= {"SUPPORTS"}
    assert all(item.verified_fact is False for item in evidence)
    assert all(item.interpretation == "EVIDENCE_SUMMARY_NOT_VERIFIED_FACT" for item in evidence)

    all_evidence = {item.evidence_id: item for item in resource.evidence_records}
    assert all_evidence["evidence:contradiction"].support_status == "CONTRADICTS"
    assert all_evidence["evidence:neutral"].support_status == "NEUTRAL"
    assert all_evidence["evidence:unassessed"].support_status == "UNASSESSED"

    restricted = resource.source_provenance
    restricted_source = next(item for item in restricted if item.source_id == "source:restricted")
    assert restricted_source.usable_for_local_import is False
    assert restricted_source.redistributable is False
    assert all_evidence["evidence:restricted"].usable_for_local_reasoning is False


def test_context_and_evidence_are_available_to_ore_language_requests_and_results():
    resource = USGKnowledgeAdapter().convert(_phase3_bundle())
    bank = next(item for item in resource.lexical.entries if item.orthography == "bank")
    financial = next(item for item in bank.meanings if item.gloss == "A financial institution.")
    contexts = resource.contexts_for(financial.meaning_id)
    evidence = resource.evidence_for(financial.meaning_id)

    lexical_request = LexicalAccessRequest(
        sequence_id="phase3-lexical-request",
        word="bank",
        language="en",
        contextual_records=contexts,
        evidence_records=evidence,
        source_provenance=resource.source_provenance,
    )
    assert lexical_request.contextual_records == contexts
    assert lexical_request.evidence_records == evidence

    processor = build_semantic_composition_processor(resource, config=SemanticCompositionConfig(
        simulate_biology=False,
    ))
    semantic_request = SemanticCompositionRequest(
        sequence_id="phase3-semantic-request",
        tokens=("bank",),
        language="en",
        contextual_records=contexts,
        evidence_records=evidence,
        source_provenance=resource.source_provenance,
    )
    result = processor.process(semantic_request)
    assert result.metadata["contextual_records"]
    assert result.metadata["evidence_records"]
    assert result.metadata["source_provenance"]
    assert result.metadata["evidence_is_verified_fact"] is False
