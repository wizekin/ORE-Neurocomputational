"""Reference specialist modules for blackboard coordination (demonstration workload).

These specialists are deliberately simple, deterministic, rule-based processors.
They exist to exercise and validate the blackboard coordination algorithm over a
realistic sentence-understanding workload, including USG-sourced lexical
polysemy, competing hypotheses, conflicts, deferral, and abstention. They are
reference demonstrators, not validated language models: their lexical knowledge
comes from a USG bundle when supplied, plus a small built-in fallback lexicon
that is always disclosed as built-in rather than imported knowledge.
"""
from __future__ import annotations

import re
from typing import Any, Mapping

from ore_contracts import ContractEnvelope, EvidenceKind
from ore_contracts.module_artifacts import ContractArtifact
from ore_contracts.module_manifest import (
    InputPort,
    ModuleDependency,
    ModuleIdentity,
    ModuleManifest,
    OutputPort,
)
from ore_contracts.module_protocol import ModuleResult, ModuleStatus
from ore_usg.contracts import ORE_USG_LEXICON_CONTRACT

from .bridge import (
    BLACKBOARD_CONTRACT_VERSION,
    BLACKBOARD_FINDING_BATCH_CONTRACT,
    BLACKBOARD_INPUT_CONTRACT,
    BLACKBOARD_MEMORY_STORE_CONTRACT,
    BLACKBOARD_SNAPSHOT_CONTRACT,
    BLACKBOARD_USG_MODULE_KNOWLEDGE_CONTRACT,
    BLACKBOARD_VERIFICATION_RULES_CONTRACT,
)
from .contracts import sha256_json

_BLACKBOARD_PACKAGE_VERSION = "1.0.0"

FALLBACK_LEXICON: Mapping[str, tuple[tuple[str, str], ...]] = {
    "the": (("determiner", "definite article"),),
    "a": (("determiner", "indefinite article"),),
    "boy": (("noun", "a male child"),),
    "door": (("noun", "a movable barrier"),),
    "opened": (("verb", "to make something open"),),
    "captain": (("noun", "a person who commands"),),
    "saw": (("verb", "to perceive with the eyes"),),
    "river": (("noun", "a large flowing watercourse"),),
    "by": (("adposition", "near or beside"),),
    "near": (("adposition", "close to"),),
    "fisherman": (("noun", "a person who catches fish"),),
    "boat": (("noun", "a small watercraft"),),
}


class _ReferenceSpecialist:
    """Shared helpers and declaration attributes for the reference specialists."""

    package_path = "ore_blackboard.specialists"
    subscription_fields: tuple[str, ...] = ()
    sensitivity_weights: Mapping[str, float] = {}
    activation_threshold: float = 0.5

    def describe(self) -> ModuleManifest:  # pragma: no cover - abstract
        """Return the specialist manifest; concrete specialists override this."""
        raise NotImplementedError

    def execute(self, inputs, context):  # pragma: no cover - abstract
        """Execute the specialist; concrete specialists override this."""
        raise NotImplementedError

    @staticmethod
    def _batch(
        findings: list[Mapping[str, Any]],
        run_id: str,
        producer_id: str,
        diagnostics: Mapping[str, Any],
    ) -> ContractArtifact:
        """Wrap finding envelopes into the typed finding-batch artifact (Step 17)."""
        from ore_workspace.models import utc_now
        return ContractArtifact(
            envelope=ContractEnvelope(
                contract_id=BLACKBOARD_FINDING_BATCH_CONTRACT,
                schema_version=BLACKBOARD_CONTRACT_VERSION,
                run_id=run_id,
                producer_id=producer_id,
                config_hash=sha256_json({"module": producer_id}),
                created_at=utc_now(),
                evidence_kind=EvidenceKind.DERIVED,
            ),
            payload={"findings": findings, "diagnostics": dict(diagnostics)},
        )

    @staticmethod
    def _input_root_finding_id(snapshot_payload: Mapping[str, Any]) -> str | None:
        """Locate the accepted input-root observation finding in the snapshot state."""
        for record in snapshot_payload.get("state", {}).get("artifact_references", []):
            if record.get("finding_type") == "input_root_observation":
                return record.get("finding_id")
        return None


class LexicalSpecialist(_ReferenceSpecialist):
    """Produce per-token lexical hypotheses with USG-sourced polysemy (Step 16 demo)."""

    module_id = "ore.blackboard.lexical_specialist"
    write_fields = {"lexical_out": "lexical_hypotheses"}
    subscription_fields = ("current_input", "artifact_references", "unresolved_questions")
    sensitivity_weights = {
        "current_input": 1.0, "artifact_references": 1.0, "unresolved_questions": 0.8,
    }
    activation_threshold = 0.5
    usg_knowledge_domains = ("LEXICAL", "MORPHOLOGY", "CONTEXT", "EVIDENCE")
    usg_knowledge_priority = 1.0

    def describe(self) -> ModuleManifest:
        """Return this specialist's validated manifest (Step 6)."""
        return ModuleManifest(
            identity=ModuleIdentity(
                module_id=self.module_id,
                module_version=_BLACKBOARD_PACKAGE_VERSION,
                module_name="Reference lexical specialist",
                package_path=self.package_path,
            ),
            responsibilities=(
                "Produce per-token lexical candidates with source spans and provenance.",
            ),
            input_ports=(
                InputPort("input_root", BLACKBOARD_INPUT_CONTRACT, ("1.0.0",),
                          required=True, expected_data_type="builtins.dict"),
                InputPort("workspace_snapshot", BLACKBOARD_SNAPSHOT_CONTRACT, ("1.0.0",),
                          required=True, expected_data_type="builtins.dict"),
                InputPort("usg_lexicon", ORE_USG_LEXICON_CONTRACT, ("1.0.0",),
                          required=False, expected_data_type="ore_usg.contracts.USGLexiconResource"),
                InputPort("usg_knowledge", BLACKBOARD_USG_MODULE_KNOWLEDGE_CONTRACT, ("1.0.0",),
                          required=False, expected_data_type="ore_blackboard.usg_integration.USGModuleKnowledge"),
            ),
            output_ports=(
                OutputPort("lexical_out", BLACKBOARD_FINDING_BATCH_CONTRACT, "1.0.0",
                           "builtins.dict", "Per-token lexical candidate findings."),
            ),
            capabilities=("language.lexical_access",),
            limitations=(
                "Reference rule-based toy processor; not a validated lexical model.",
                "Built-in fallback lexicon is disclosed as built-in, not imported knowledge.",
            ),
        )

    def execute(self, inputs: Mapping[str, Any], context: Any) -> ModuleResult:
        """Emit one lexical candidate per token sense with span and USG provenance."""
        root_payload = inputs["input_root"].payload
        usg_lexicon = None
        usg_meta = None
        routed_knowledge = None
        if "usg_lexicon" in inputs:
            usg_lexicon = inputs["usg_lexicon"].payload
            usg_meta = inputs["usg_lexicon"].metadata
        if "usg_knowledge" in inputs:
            routed_knowledge = inputs["usg_knowledge"].payload
            usg_meta = {
                "usg_bundle_id": routed_knowledge.bundle_id,
                "usg_bundle_digest": routed_knowledge.bundle_digest,
            }
        received_feedback = (
            tuple(item.to_dict() for item in routed_knowledge.feedback)
            if routed_knowledge is not None else ()
        )
        snapshot_payload = inputs["workspace_snapshot"].payload
        input_root_finding_id = self._input_root_finding_id(snapshot_payload)
        tokens = root_payload["tokens"]
        findings: list[dict[str, Any]] = []
        used_usg_records: list[str] = []
        unknown_tokens: list[str] = []
        for index, token_record in enumerate(tokens):
            token = token_record["token"]
            candidates = (
                self._routed_candidates(token, routed_knowledge)
                if routed_knowledge is not None else ()
            )
            if not candidates:
                candidates = tuple(
                    (*item, {}) for item in self._candidates(token, usg_lexicon)
                )
            if not candidates:
                unknown_tokens.append(token)
                findings.append({
                    "field": "lexical_hypotheses",
                    "key": token.casefold(),
                    "finding_type": "lexical_unknown",
                    "payload": {
                        "word": token,
                        "candidate": "unknown",
                        "part_of_speech": "unknown",
                        "unknown": True,
                        "token_index": index,
                        "usg_feedback_received": [
                            item for item in received_feedback
                            if not item.get("query") or str(item.get("query")).casefold() == token.casefold()
                        ],
                    },
                    "source": "current_input",
                    "source_span": [token_record["start"], token_record["end"]],
                    "rules_applied": ("token_lookup_miss", "preserve_unknown_token"),
                    "confidence": 0.1,
                    "dependencies": (
                        [input_root_finding_id] if input_root_finding_id else []
                    ),
                    "notes": "Downstream feedback received; the unknown token remains unknown until explicit evidence is available.",
                })
                continue
            for source_label, gloss, part_of_speech, prior, usg_reference, extra in candidates:
                payload = {
                    "word": token,
                    "candidate": gloss,
                    "gloss": gloss,
                    "part_of_speech": part_of_speech,
                    "source_label": source_label,
                    "token_index": index,
                    "usg_feedback_received": [
                        item for item in received_feedback
                        if not item.get("query") or str(item.get("query")).casefold() == token.casefold()
                    ],
                    **dict(extra),
                }
                envelope: dict[str, Any] = {
                    "field": "lexical_hypotheses",
                    "key": token.casefold(),
                    "finding_type": "lexical_hypothesis",
                    "payload": payload,
                    "source": "current_input",
                    "source_span": [token_record["start"], token_record["end"]],
                    "rules_applied": (
                        "exact_surface_lookup", "preserve_competing_senses",
                        "preserve_semantic_link_status",
                    ),
                    "confidence": prior,
                    "dependencies": (
                        [input_root_finding_id] if input_root_finding_id else []
                    ),
                }
                if usg_reference is not None:
                    envelope["usg_references"] = [usg_reference]
                    envelope["source"] = f"usg_knowledge_bundle:{usg_reference['bundle_id']}"
                    envelope["evidence"] = [{
                        "bundle_id": usg_reference["bundle_id"],
                        "bundle_digest": usg_reference["bundle_digest"],
                        "record_id": usg_reference["record_id"],
                        "source_reference": usg_reference["source_reference"],
                        "evidence_statuses": usg_reference.get("evidence_statuses", []),
                    }]
                    used_usg_records.append(usg_reference["record_id"])
                    if extra.get("semantic_link_status") in {
                        "MISSING", "AMBIGUOUS", "CONFLICTED", "CONTRADICTED",
                        "UNRESOLVED", "CANDIDATE", "NEUTRAL", "UNASSESSED",
                    }:
                        envelope["payload"]["usg_feedback_requests"] = [{
                            "target_module": "ore.blackboard.semantic_specialist",
                            "feedback_kind": "LEXICAL_SEMANTIC_LINK_" + extra["semantic_link_status"],
                            "query": token,
                            "record_id": usg_reference["record_id"],
                            "reason": (
                                "lexical meaning has a non-final semantic-link state; "
                                "keep competing meanings separate"
                            ),
                        }]
                findings.append(envelope)
        route_stats = {
            "cache_hits": routed_knowledge.cache_hits if routed_knowledge is not None else 0,
            "cache_misses": routed_knowledge.cache_misses if routed_knowledge is not None else 0,
            "domains": list(routed_knowledge.domains) if routed_knowledge is not None else [],
        }
        diagnostics = {
            "token_count": len(tokens),
            "unknown_tokens": unknown_tokens,
            "usg_sourced": bool(used_usg_records),
            "usg_bundle_id": (usg_meta or {}).get("usg_bundle_id"),
            "usg_routing": route_stats,
            "usg_feedback_received": [dict(item) for item in received_feedback],
            "usg_feedback_consumed_count": len(received_feedback),
            "confidence_semantics": "rule-based lexical prior; not a calibrated probability",
        }
        return ModuleResult(
            module_id=self.module_id,
            module_version=_BLACKBOARD_PACKAGE_VERSION,
            status=ModuleStatus.SUCCESS,
            artifacts=(self._batch(findings, context.run_id, self.module_id, diagnostics),),
            diagnostics=diagnostics,
            provenance={"lexicon": "usg" if routed_knowledge is not None or usg_lexicon else "builtin_fallback"},
        )

    def _routed_candidates(self, token: str, knowledge: Any):
        """Project exact routed USG meanings without scanning the complete lexicon."""
        results: list[tuple[str, str, str, float, Any, Mapping[str, Any]]] = []
        for entry in knowledge.lexical_entries:
            if entry.orthography.casefold() != token.casefold():
                continue
            for meaning in entry.meanings:
                evidence_ids = tuple(knowledge.evidence_by_record.get(meaning.meaning_id, ()))
                evidence_by_id = {item.evidence_id: item for item in knowledge.evidence}
                evidence_rows = [
                    {
                        "evidence_id": evidence_id,
                        "source_id": evidence_by_id[evidence_id].source_id,
                        "support_status": evidence_by_id[evidence_id].support_status,
                        "source_rights_status": evidence_by_id[evidence_id].source_rights_status,
                        "source_redistribution": evidence_by_id[evidence_id].source_redistribution,
                        "verified_fact": False,
                    }
                    for evidence_id in evidence_ids if evidence_id in evidence_by_id
                ]
                context_ids = tuple(knowledge.context_by_record.get(meaning.meaning_id, ()))
                context_by_id = {item.context_id: item for item in knowledge.contexts}
                context_rows = [
                    {
                        "context_id": context_id,
                        "context_type": context_by_id[context_id].context_type,
                        "value": context_by_id[context_id].value,
                        "scope": context_by_id[context_id].scope,
                    }
                    for context_id in context_ids if context_id in context_by_id
                ]
                links = tuple(
                    item for item in knowledge.lexical_concept_links
                    if item.lexical_entry_id == entry.entry_id
                    and item.lexical_meaning_id == meaning.meaning_id
                )
                concept_ids = tuple(sorted(set(meaning.semantic_concept_ids) | {
                    item.semantic_concept_id for item in links
                    if item.semantic_concept_id is not None
                }))
                statuses = {item.status for item in links}
                link_status = meaning.semantic_link_status
                if "CONFLICTED" in statuses:
                    link_status = "CONFLICTED"
                elif "CONTRADICTED" in statuses:
                    link_status = "CONTRADICTED"
                elif len(concept_ids) > 1:
                    link_status = "AMBIGUOUS"
                elif statuses and link_status == "UNLINKED":
                    link_status = next(iter(statuses))
                reference = {
                    "bundle_id": knowledge.bundle_id,
                    "bundle_digest": knowledge.bundle_digest,
                    "record_id": meaning.meaning_id,
                    "lexical_entry_id": entry.entry_id,
                    "source_reference": meaning.source_reference,
                    "semantic_concept_ids": list(concept_ids),
                    "semantic_link_status": link_status,
                    "evidence_ids": list(evidence_ids),
                    "evidence_statuses": [item["support_status"] for item in evidence_rows],
                    "context_ids": list(context_ids),
                }
                extra = {
                    "meaning_id": meaning.meaning_id,
                    "semantic_concept_ids": list(concept_ids),
                    "semantic_link_status": link_status,
                    "evidence": evidence_rows,
                    "context_ids": list(context_ids),
                    "contexts": context_rows,
                    "lemma_entry_id": entry.lemma_entry_id,
                    "morphology_link_status": entry.morphology_link_status,
                    "script": entry.script,
                    "grammatical_information_json": entry.grammatical_information_json,
                }
                results.append((
                    "usg", meaning.gloss, entry.lexical_category, meaning.prior,
                    reference, extra,
                ))
        return tuple(results)

    def _candidates(self, token, usg_lexicon):
        """Yield ``(source_label, gloss, pos, prior, usg_reference)`` per token sense."""
        results: list[tuple[str, str, str, float, Any]] = []
        if usg_lexicon is not None:
            for entry in usg_lexicon.entries:
                if entry.orthography.casefold() != token.casefold():
                    continue
                for meaning in entry.meanings:
                    results.append((
                        "usg", meaning.gloss, entry.lexical_category, meaning.prior,
                        {
                            "bundle_id": usg_lexicon.bundle_id,
                            "bundle_digest": usg_lexicon.bundle_digest,
                            "record_id": meaning.meaning_id,
                            "source_reference": meaning.source_reference,
                        },
                    ))
        for part_of_speech, gloss in FALLBACK_LEXICON.get(token.casefold(), ()):
            if not any(existing[1] == gloss for existing in results):
                results.append(("builtin_fallback", gloss, part_of_speech, 0.5, None))
        return tuple(results)


class SyntaxSpecialist(_ReferenceSpecialist):
    """Build a deterministic structural hypothesis from lexical candidates."""

    module_id = "ore.blackboard.syntax_specialist"
    write_fields = {"syntax_out": "syntactic_hypotheses"}
    subscription_fields = ("lexical_hypotheses",)
    sensitivity_weights = {"lexical_hypotheses": 1.0}
    activation_threshold = 0.5

    def describe(self) -> ModuleManifest:
        """Return this specialist's validated manifest (Step 6)."""
        return ModuleManifest(
            identity=ModuleIdentity(
                module_id=self.module_id,
                module_version=_BLACKBOARD_PACKAGE_VERSION,
                module_name="Reference syntax specialist",
                package_path=self.package_path,
            ),
            responsibilities=(
                "Compose one deterministic subject-verb-object structural hypothesis.",
            ),
            input_ports=(
                InputPort("input_root", BLACKBOARD_INPUT_CONTRACT, ("1.0.0",),
                          required=True, expected_data_type="builtins.dict"),
                InputPort("workspace_snapshot", BLACKBOARD_SNAPSHOT_CONTRACT, ("1.0.0",),
                          required=True, expected_data_type="builtins.dict"),
            ),
            output_ports=(
                OutputPort("syntax_out", BLACKBOARD_FINDING_BATCH_CONTRACT, "1.0.0",
                           "builtins.dict", "One structural hypothesis finding."),
            ),
            capabilities=("language.syntax_parsing",),
            dependencies=(ModuleDependency("ore.blackboard.lexical_specialist", required=True),),
            limitations=("Reference toy parser; no treebank training and no learned grammar.",),
        )

    def execute(self, inputs: Mapping[str, Any], context: Any) -> ModuleResult:
        """Compose a structural hypothesis or abstain when no verb is present."""
        state = inputs["workspace_snapshot"].payload["state"]
        lexical = state.get("lexical_hypotheses", {})
        if not lexical:
            return ModuleResult(
                module_id=self.module_id,
                module_version=_BLACKBOARD_PACKAGE_VERSION,
                status=ModuleStatus.ABSTAINED,
                diagnostics={"reason": "no lexical hypotheses are available yet"},
            )
        tokens: list[dict[str, Any]] = []
        for word, records in lexical.items():
            chosen = records[0] if records else None
            if chosen is None:
                continue
            tokens.append({
                "word": word,
                "part_of_speech": chosen.get("part_of_speech", "unknown"),
                "finding_id": chosen.get("finding_id"),
                "unknown": bool(chosen.get("unknown")),
                "token_index": int(chosen.get("token_index", 0)),
            })
        tokens.sort(key=lambda item: (item["token_index"], item["word"]))
        verbs = [item for item in tokens if item["part_of_speech"] == "verb"]
        nouns = [item for item in tokens if item["part_of_speech"] == "noun"]
        if not verbs:
            return ModuleResult(
                module_id=self.module_id,
                module_version=_BLACKBOARD_PACKAGE_VERSION,
                status=ModuleStatus.ABSTAINED,
                diagnostics={"reason": "no verb candidate; structural parse deferred"},
            )
        verb = verbs[0]
        subject = nouns[0]["word"] if nouns else "unknown"
        object_word = nouns[-1]["word"] if len(nouns) > 1 else None
        parse = "subject-verb-object" if object_word else "subject-verb"
        dependencies = tuple(
            item["finding_id"] for item in tokens
            if item["finding_id"] and item is not verb
        )
        finding = {
            "field": "syntactic_hypotheses",
            "finding_type": "structural_hypothesis",
            "payload": {
                "parse": parse,
                "candidate": parse,
                "subject": subject,
                "verb": verb["word"],
                "object": object_word,
                "token_count": len(tokens),
            },
            "source": "module_inference",
            "rules_applied": ("first_noun_subject", "first_verb_predicate", "last_noun_object"),
            "dependencies": list(dependencies),
            "confidence": 0.6,
        }
        return ModuleResult(
            module_id=self.module_id,
            module_version=_BLACKBOARD_PACKAGE_VERSION,
            status=ModuleStatus.SUCCESS,
            artifacts=(self._batch([finding], context.run_id, self.module_id, {"parse": parse}),),
            diagnostics={"parse": parse, "verb_found": verb["word"]},
            provenance={"upstream": "lexical_hypotheses"},
        )


class SemanticSpecialist(_ReferenceSpecialist):
    """Compose propositions while comparing USG meanings and preserving alternatives."""

    module_id = "ore.blackboard.semantic_specialist"
    write_fields = {"semantic_out": "semantic_hypotheses"}
    subscription_fields = (
        "lexical_hypotheses", "syntactic_hypotheses", "unresolved_questions",
    )
    sensitivity_weights = {
        "lexical_hypotheses": 0.5, "syntactic_hypotheses": 0.8,
        "unresolved_questions": 0.7,
    }
    activation_threshold = 0.5
    usg_knowledge_domains = (
        "SEMANTIC", "RELATIONAL", "PREDICATE", "CONTEXT", "EVIDENCE",
        "SOURCE_PROVENANCE",
    )
    usg_knowledge_priority = 0.95
    MAX_PROPOSITIONS = 64

    def describe(self) -> ModuleManifest:
        """Return the typed specialist declaration, including its optional routed knowledge."""
        return ModuleManifest(
            identity=ModuleIdentity(
                module_id=self.module_id,
                module_version=_BLACKBOARD_PACKAGE_VERSION,
                module_name="Reference semantic composition specialist",
                package_path=self.package_path,
            ),
            responsibilities=(
                "Compose propositions from structure and lexical candidates.",
                "Compare explicitly linked USG concepts and supporting/contradicting evidence.",
                "Preserve ambiguous, unresolved and contradictory readings as alternatives.",
            ),
            input_ports=(
                InputPort("workspace_snapshot", BLACKBOARD_SNAPSHOT_CONTRACT, ("1.0.0",),
                          required=True, expected_data_type="builtins.dict"),
                InputPort("usg_knowledge", BLACKBOARD_USG_MODULE_KNOWLEDGE_CONTRACT, ("1.0.0",),
                          required=False, expected_data_type="ore_blackboard.usg_integration.USGModuleKnowledge"),
            ),
            output_ports=(
                OutputPort("semantic_out", BLACKBOARD_FINDING_BATCH_CONTRACT, "1.0.0",
                           "builtins.dict", "Proposition-level semantic findings with evidence."),
            ),
            capabilities=("language.semantic_composition",),
            dependencies=(ModuleDependency("ore.blackboard.syntax_specialist", required=True),),
            limitations=(
                "Reference toy compositor; semantic evidence scores are uncalibrated and are not probabilities.",
                "USG evidence status is preserved as source evidence, not treated as independently verified fact.",
            ),
        )

    def execute(self, inputs: Mapping[str, Any], context: Any) -> ModuleResult:
        """Compare each bounded meaning combination against explicit USG evidence and links."""
        state = inputs["workspace_snapshot"].payload["state"]
        structures = state.get("syntactic_hypotheses", [])
        if not structures:
            return ModuleResult(
                module_id=self.module_id,
                module_version=_BLACKBOARD_PACKAGE_VERSION,
                status=ModuleStatus.ABSTAINED,
                diagnostics={"reason": "no structural hypothesis is available yet"},
            )
        structure = structures[0]
        lexical = state.get("lexical_hypotheses", {})
        subject_word = structure.get("subject")
        object_word = structure.get("object")
        verb_word = structure.get("verb")
        routed = inputs["usg_knowledge"].payload if "usg_knowledge" in inputs else None
        received_feedback = (
            tuple(item.to_dict() for item in routed.feedback)
            if routed is not None else ()
        )

        def senses(word: Any) -> list[dict[str, Any]]:
            records = lexical.get(str(word).casefold(), []) if word else []
            output = []
            for record in records:
                output.append({
                    "gloss": record.get("gloss", record.get("candidate", "unknown")),
                    "finding_id": record.get("finding_id"),
                    "unknown": bool(record.get("unknown")),
                    "meaning_id": record.get("meaning_id"),
                    "semantic_concept_ids": tuple(record.get("semantic_concept_ids", ())),
                    "semantic_link_status": str(record.get("semantic_link_status", "UNLINKED")),
                    "evidence": tuple(record.get("evidence", ())),
                    "context_ids": tuple(record.get("context_ids", ())),
                    "usg_references": tuple(record.get("usg_references", ())),
                })
            return output or [{
                "gloss": "unknown", "finding_id": None, "unknown": True,
                "meaning_id": None, "semantic_concept_ids": (),
                "semantic_link_status": "MISSING", "evidence": (),
                "context_ids": (), "usg_references": (),
            }]

        subject_senses = senses(subject_word)
        object_senses = senses(object_word) if object_word else [{
            "gloss": None, "finding_id": None, "unknown": False,
            "meaning_id": None, "semantic_concept_ids": (),
            "semantic_link_status": "UNLINKED", "evidence": (),
            "context_ids": (), "usg_references": (),
        }]
        verb_senses = senses(verb_word)

        evidence_by_id = {
            item.evidence_id: item for item in (routed.evidence if routed is not None else ())
        }
        relations = tuple(routed.relations) if routed is not None else ()
        memberships = tuple(routed.memberships) if routed is not None else ()
        frames = tuple(routed.predicate_frames) if routed is not None else ()
        frame_matches = tuple(
            item for item in frames
            if str(getattr(item, "frame_key", "")).casefold() == str(verb_word or "").casefold()
        )
        findings: list[dict[str, Any]] = []
        all_feedback_requests: list[dict[str, Any]] = []
        if structure.get("finding_id"):
            structural_dependency = str(structure["finding_id"])
        else:
            structural_dependency = None

        combinations = (
            (subject_sense, verb_sense, object_sense)
            for subject_sense in subject_senses
            for verb_sense in verb_senses
            for object_sense in object_senses
        )
        for combination_index, (subject_sense, verb_sense, object_sense) in enumerate(combinations):
            if combination_index >= self.MAX_PROPOSITIONS:
                break
            proposition = (
                f"{subject_word}({subject_sense['gloss']}) "
                f"{verb_word}({verb_sense['gloss']}) "
                f"{object_word}({object_sense['gloss']})".strip()
            )
            role_senses = (
                ("agent", subject_word, subject_sense),
                ("predicate", verb_word, verb_sense),
                ("patient", object_word, object_sense),
            )
            concept_ids = {
                concept_id
                for _role, _word, sense in role_senses
                for concept_id in sense.get("semantic_concept_ids", ())
            }
            evidence_rows: dict[str, dict[str, Any]] = {}
            contexts: set[str] = set()
            link_statuses: list[str] = []
            dependencies: set[str] = set()
            if structural_dependency:
                dependencies.add(structural_dependency)
            for role, word, sense in role_senses:
                if sense.get("finding_id"):
                    dependencies.add(str(sense["finding_id"]))
                contexts.update(str(item) for item in sense.get("context_ids", ()))
                status = str(sense.get("semantic_link_status", "UNLINKED"))
                if status not in {"UNLINKED", "LINKED"}:
                    link_statuses.append(status)
                for raw_evidence in sense.get("evidence", ()):
                    if not isinstance(raw_evidence, Mapping):
                        continue
                    evidence_id = str(raw_evidence.get("evidence_id", ""))
                    if not evidence_id:
                        continue
                    evidence_rows[evidence_id] = {
                        **dict(raw_evidence),
                        "role": role,
                        "word": str(word or ""),
                        "verified_fact": False,
                    }

            # Include evidence only from explicit relations/memberships incident to
            # the candidate concepts. The source assertion and its direction survive.
            for relation in relations:
                if relation.subject_id not in concept_ids and relation.object_id not in concept_ids:
                    continue
                for evidence_id in relation.evidence_ids:
                    item = evidence_by_id.get(evidence_id)
                    if item is not None:
                        evidence_rows[evidence_id] = {
                            "evidence_id": item.evidence_id,
                            "source_id": item.source_id,
                            "support_status": item.support_status,
                            "evidence_type": item.evidence_type,
                            "verified_fact": False,
                            "relationship_id": relation.relation_id,
                            "relationship_predicate": relation.predicate,
                            "relationship_direction": relation.direction,
                        }
            for membership in memberships:
                if membership.member_id not in concept_ids and membership.category_id not in concept_ids:
                    continue
                for evidence_id in membership.evidence_ids:
                    item = evidence_by_id.get(evidence_id)
                    if item is not None:
                        evidence_rows[evidence_id] = {
                            "evidence_id": item.evidence_id,
                            "source_id": item.source_id,
                            "support_status": item.support_status,
                            "evidence_type": item.evidence_type,
                            "verified_fact": False,
                            "relationship_id": membership.membership_id,
                            "relationship_predicate": membership.predicate,
                            "relationship_direction": membership.direction,
                        }

            # Frame-level evidence and context are attached independently of
            # whether a frame is executable. An incomplete frame is still useful
            # evidence about the current interpretation, but it is not executable.
            for frame in frame_matches:
                frame_id = str(getattr(frame, "frame_id", ""))
                frame_status = str(getattr(frame, "status", "PRESERVED_ONLY"))
                contexts.update(str(item) for item in routed.context_by_record.get(frame_id, ()))
                for evidence_id in routed.evidence_by_record.get(frame_id, ()):
                    item = evidence_by_id.get(evidence_id)
                    if item is not None:
                        evidence_rows[evidence_id] = {
                            "evidence_id": item.evidence_id,
                            "source_id": item.source_id,
                            "support_status": item.support_status,
                            "evidence_type": item.evidence_type,
                            "verified_fact": False,
                            "predicate_frame_id": frame_id,
                            "predicate_frame_status": frame_status,
                        }

            evidence_values = tuple(evidence_rows[key] for key in sorted(evidence_rows))
            evidence_statuses = {str(item.get("support_status", "UNASSESSED")) for item in evidence_values}
            supporting = sum(item == "SUPPORTS" for item in evidence_statuses)
            contradicting = sum(item == "CONTRADICTS" for item in evidence_statuses)
            evidence_conflict = supporting > 0 and contradicting > 0
            evidence_score = sum(
                1.0 if item.get("support_status") == "SUPPORTS"
                else -1.0 if item.get("support_status") == "CONTRADICTS"
                else 0.0
                for item in evidence_values
            )
            if evidence_conflict:
                assertion_status = "CONFLICTED"
            elif contradicting:
                assertion_status = "CONTRADICTED"
            elif link_statuses:
                assertion_status = sorted(link_statuses)[0]
            elif evidence_statuses and evidence_statuses.issubset({"SUPPORTS"}):
                assertion_status = "SUPPORTED"
            elif evidence_statuses:
                assertion_status = "UNASSESSED"
            else:
                assertion_status = "UNASSESSED"

            semantic_roles = {
                role: {
                    "word": str(word) if word is not None else None,
                    "gloss": sense.get("gloss"),
                    "meaning_id": sense.get("meaning_id"),
                    "semantic_concept_ids": list(sense.get("semantic_concept_ids", ())),
                    "link_status": sense.get("semantic_link_status", "UNLINKED"),
                }
                for role, word, sense in role_senses
            }
            feedback_requests: list[dict[str, Any]] = []
            for role, word, sense in role_senses:
                status = str(sense.get("semantic_link_status", "UNLINKED"))
                if word and status in {
                    "MISSING", "AMBIGUOUS", "CONFLICTED", "CONTRADICTED",
                    "UNRESOLVED", "CANDIDATE", "NEUTRAL", "UNASSESSED",
                }:
                    feedback_requests.append({
                        "target_module": self.module_id,
                        "feedback_kind": "SEMANTIC_LINK_" + status,
                        "query": str(word),
                        "record_id": sense.get("meaning_id"),
                        "reason": (
                            "semantic composition retained a non-final lexical-to-concept "
                            "link status; retrieve alternatives without choosing a winner"
                        ),
                    })
            if evidence_conflict:
                feedback_requests.append({
                    "target_module": self.module_id,
                    "feedback_kind": "CONTRADICTORY_SEMANTIC_EVIDENCE",
                    "query": str(verb_word or ""),
                    "record_id": next(iter(sorted(concept_ids)), None),
                    "reason": "both supporting and contradicting source evidence are attached to this interpretation",
                })
            if len(subject_senses) > 1 and not any(item.get("context_ids") for item in subject_senses):
                feedback_requests.append({
                    "target_module": "ore.blackboard.lexical_specialist",
                    "feedback_kind": "MISSING_DISAMBIGUATING_CONTEXT",
                    "query": str(subject_word or ""),
                    "reason": "multiple subject meanings remain without an explicit disambiguating context",
                })
            if len(object_senses) > 1 and not any(item.get("context_ids") for item in object_senses):
                feedback_requests.append({
                    "target_module": "ore.blackboard.lexical_specialist",
                    "feedback_kind": "MISSING_DISAMBIGUATING_CONTEXT",
                    "query": str(object_word or ""),
                    "reason": "multiple object meanings remain without an explicit disambiguating context",
                })
            if routed is not None and verb_word and not frame_matches:
                feedback_requests.append({
                    "target_module": self.module_id,
                    "feedback_kind": "MISSING_PREDICATE_FRAME",
                    "query": str(verb_word),
                    "reason": "no exact USG predicate frame was available for the predicate",
                })
            elif any(
                str(getattr(frame, "status", "PRESERVED_ONLY")) in {"UNRESOLVED", "INCOMPLETE"}
                for frame in frame_matches
            ):
                feedback_requests.append({
                    "target_module": self.module_id,
                    "feedback_kind": "UNRESOLVED_PREDICATE_FRAME",
                    "query": str(verb_word),
                    "record_id": next(
                        str(frame.frame_id) for frame in frame_matches
                        if str(getattr(frame, "status", "PRESERVED_ONLY")) in {"UNRESOLVED", "INCOMPLETE"}
                    ),
                    "reason": (
                        "an exact predicate frame exists but has unresolved or incomplete fields; "
                        "preserve it as a non-executable alternative"
                    ),
                })
            all_feedback_requests.extend(feedback_requests)

            usg_references = [
                reference for _role, _word, sense in role_senses
                for reference in sense.get("usg_references", ())
            ]
            frame_ids = tuple(str(item.frame_id) for item in frame_matches)
            findings.append({
                "field": "semantic_hypotheses",
                "finding_type": "semantic_proposition",
                "payload": {
                    "proposition": proposition,
                    "candidate": proposition,
                    "predicate": verb_word,
                    "agent": subject_word,
                    "agent_sense": subject_sense["gloss"],
                    "patient": object_word,
                    "patient_sense": object_sense["gloss"],
                    "semantic_roles": semantic_roles,
                    "semantic_concept_ids": sorted(concept_ids),
                    "usg_evidence": list(evidence_values),
                    "usg_evidence_score": evidence_score,
                    "usg_evidence_score_calibrated_probability": False,
                    "evidence_conflict": evidence_conflict,
                    "evidence_assertion_status": assertion_status,
                    "context_ids": sorted(contexts),
                    "predicate_frame_ids": list(frame_ids),
                    "predicate_frame_statuses": [
                        getattr(item, "status", "PRESERVED") for item in frame_matches
                    ],
                    "usg_feedback_received": [
                        item for item in received_feedback
                        if not item.get("query")
                        or str(item.get("query")).casefold() in {
                            str(subject_word or "").casefold(),
                            str(object_word or "").casefold(),
                            str(verb_word or "").casefold(),
                        }
                    ],
                    "usg_feedback_requests": feedback_requests,
                },
                "source": "module_inference",
                "rules_applied": (
                    "predicate_argument_composition",
                    "compare_supporting_and_contradicting_evidence",
                    "preserve_ambiguous_and_unresolved_alternatives",
                    "do_not_promote_evidence_to_verified_fact",
                ),
                "dependencies": sorted(dependencies),
                "usg_references": usg_references,
                "evidence": list(evidence_values),
            })

        competing = len(findings) > 1
        diagnostics = {
            "competing_interpretations": competing,
            "proposition_count": len(findings),
            "proposition_limit": self.MAX_PROPOSITIONS,
            "evidence_is_calibrated_probability": False,
            "usg_evidence_conflicts": sum(
                bool(item["payload"]["evidence_conflict"]) for item in findings
            ),
            "feedback_requests": len(all_feedback_requests),
            "feedback_received": len(received_feedback),
            "usg_feedback_received": [dict(item) for item in received_feedback],
            "usg_routing": (
                {
                    "cache_hits": routed.cache_hits,
                    "cache_misses": routed.cache_misses,
                    "domains": list(routed.domains),
                    "record_count": routed.record_count,
                } if routed is not None else {"available": False}
            ),
        }
        return ModuleResult(
            module_id=self.module_id,
            module_version=_BLACKBOARD_PACKAGE_VERSION,
            status=ModuleStatus.SUCCESS,
            artifacts=(self._batch(
                findings, context.run_id, self.module_id, diagnostics,
            ),),
            diagnostics=diagnostics,
            provenance={
                "upstream": "syntactic_hypotheses",
                "usg_source_mutation_performed": False,
                "evidence_interpretation": "source support status only; not fact verification",
            },
        )


class MemorySpecialist(_ReferenceSpecialist):
    """Retrieve caller-supplied episodic memories relevant to the current content."""

    module_id = "ore.blackboard.memory_specialist"
    write_fields = {"memory_out": "retrieved_memories"}
    subscription_fields = ("semantic_hypotheses",)
    sensitivity_weights = {"semantic_hypotheses": 1.0}
    activation_threshold = 0.5

    def describe(self) -> ModuleManifest:
        """Return this specialist's validated manifest (Step 6)."""
        return ModuleManifest(
            identity=ModuleIdentity(
                module_id=self.module_id,
                module_version=_BLACKBOARD_PACKAGE_VERSION,
                module_name="Reference memory retrieval specialist",
                package_path=self.package_path,
            ),
            responsibilities=("Retrieve caller-supplied episodic memories for current content.",),
            input_ports=(
                InputPort("workspace_snapshot", BLACKBOARD_SNAPSHOT_CONTRACT, ("1.0.0",),
                          required=True, expected_data_type="builtins.dict"),
                InputPort("memory_store", BLACKBOARD_MEMORY_STORE_CONTRACT, ("1.0.0",),
                          required=False, expected_data_type="builtins.dict"),
            ),
            output_ports=(
                OutputPort("memory_out", BLACKBOARD_FINDING_BATCH_CONTRACT, "1.0.0",
                           "builtins.dict", "Retrieved episodic memory findings."),
            ),
            capabilities=("memory.episodic_retrieval",),
            limitations=(
                "Read-only retrieval of caller-supplied episodes; no durable memory writes.",
            ),
        )

    def execute(self, inputs: Mapping[str, Any], context: Any) -> ModuleResult:
        """Retrieve matching episodes or abstain when no store was supplied."""
        if "memory_store" not in inputs:
            return ModuleResult(
                module_id=self.module_id,
                module_version=_BLACKBOARD_PACKAGE_VERSION,
                status=ModuleStatus.ABSTAINED,
                diagnostics={"reason": "memory_store_not_supplied"},
            )
        episodes = inputs["memory_store"].payload["episodes"]
        state = inputs["workspace_snapshot"].payload["state"]
        semantics = state.get("semantic_hypotheses", [])
        content_words: set[str] = set()
        for structure in semantics:
            for key in ("predicate", "agent"):
                value = structure.get(key)
                if value:
                    content_words.add(str(value).casefold())
            for key in ("agent_sense", "patient_sense"):
                value = structure.get(key)
                if value:
                    content_words.update(
                        token.casefold() for token in re.findall(r"\w+", str(value))
                    )
        findings: list[dict[str, Any]] = []
        dependencies: set[str] = set()
        for structure in semantics:
            if structure.get("finding_id"):
                dependencies.add(structure["finding_id"])
        for episode in episodes:
            keywords = {str(item).casefold() for item in episode.get("keywords", ())}
            matched = sorted(keywords & content_words)
            relevance = len(matched) / max(1, len(keywords)) if keywords else 0.0
            if relevance <= 0:
                continue
            findings.append({
                "field": "retrieved_memories",
                "finding_type": "episodic_memory",
                "payload": {
                    "memory_id": episode.get("memory_id"),
                    "summary": episode.get("text"),
                    "matched_keywords": matched,
                    "relevance_score": round(relevance, 3),
                },
                "source": "caller_supplied_memory_store",
                "rules_applied": ("keyword_overlap_retrieval",),
                "dependencies": sorted(dependencies),
                "confidence": min(1.0, relevance),
            })
        if not findings:
            return ModuleResult(
                module_id=self.module_id,
                module_version=_BLACKBOARD_PACKAGE_VERSION,
                status=ModuleStatus.ABSTAINED,
                diagnostics={"reason": "no episode matched current content"},
            )
        return ModuleResult(
            module_id=self.module_id,
            module_version=_BLACKBOARD_PACKAGE_VERSION,
            status=ModuleStatus.SUCCESS,
            artifacts=(self._batch(
                findings, context.run_id, self.module_id, {"retrieved": len(findings)},
            ),),
            diagnostics={"retrieved": len(findings)},
            provenance={"source": "caller_supplied_memory_store"},
        )


class VerificationSpecialist(_ReferenceSpecialist):
    """Register verification outcomes and unresolved questions (Step 21 demo)."""

    module_id = "ore.blackboard.verification_specialist"
    write_fields = {"verification_out": "unresolved_questions"}
    subscription_fields = ("semantic_hypotheses", "retrieved_memories")
    sensitivity_weights = {
        "semantic_hypotheses": 1.0, "retrieved_memories": 0.5,
    }
    activation_threshold = 0.5

    def describe(self) -> ModuleManifest:
        """Return this specialist's validated manifest (Step 6)."""
        return ModuleManifest(
            identity=ModuleIdentity(
                module_id=self.module_id,
                module_version=_BLACKBOARD_PACKAGE_VERSION,
                module_name="Reference verification specialist",
                package_path=self.package_path,
            ),
            responsibilities=(
                "Check semantic hypotheses against verification rules.",
                "Register unresolved questions for competing readings.",
            ),
            input_ports=(
                InputPort("workspace_snapshot", BLACKBOARD_SNAPSHOT_CONTRACT, ("1.0.0",),
                          required=True, expected_data_type="builtins.dict"),
                InputPort("verification_rules", BLACKBOARD_VERIFICATION_RULES_CONTRACT,
                          ("1.0.0",), required=False, expected_data_type="builtins.dict"),
            ),
            output_ports=(
                OutputPort("verification_out", BLACKBOARD_FINDING_BATCH_CONTRACT, "1.0.0",
                           "builtins.dict", "Verification question findings."),
            ),
            capabilities=("cognition.verification",),
            limitations=(
                "Reference rule checker; verification is procedural, not empirical truth.",
            ),
        )

    def execute(self, inputs: Mapping[str, Any], context: Any) -> ModuleResult:
        """Publish one unresolved question per open ambiguity or one verified record."""
        state = inputs["workspace_snapshot"].payload["state"]
        semantics = state.get("semantic_hypotheses", [])
        if not semantics:
            return ModuleResult(
                module_id=self.module_id,
                module_version=_BLACKBOARD_PACKAGE_VERSION,
                status=ModuleStatus.ABSTAINED,
                diagnostics={"reason": "no semantic hypothesis is available yet"},
            )
        rules = ()
        if "verification_rules" in inputs:
            rules = tuple(
                dict(item) for item in inputs["verification_rules"].payload["rules"]
            )
        dependencies = {
            structure["finding_id"] for structure in semantics if structure.get("finding_id")
        }
        findings: list[dict[str, Any]] = []
        competing = len(semantics) > 1
        if competing:
            labels = sorted(str(structure.get("candidate", "")) for structure in semantics)
            feedback_requests: list[dict[str, Any]] = []
            seen_requests: set[tuple[str, str, str | None]] = set()
            for structure in semantics:
                for role_details in structure.get("semantic_roles", {}).values():
                    if not isinstance(role_details, Mapping):
                        continue
                    word = role_details.get("word")
                    status = str(role_details.get("link_status", "UNLINKED"))
                    meaning_id = role_details.get("meaning_id")
                    if word and status in {
                        "AMBIGUOUS", "MISSING", "CONFLICTED", "CONTRADICTED",
                        "UNRESOLVED", "CANDIDATE", "NEUTRAL", "UNASSESSED",
                    }:
                        key = ("ore.blackboard.lexical_specialist", str(word).casefold(), str(meaning_id) if meaning_id else None)
                        if key not in seen_requests:
                            seen_requests.add(key)
                            feedback_requests.append({
                                "target_module": "ore.blackboard.lexical_specialist",
                                "feedback_kind": "VERIFICATION_" + status + "_LEXICAL_MEANING",
                                "query": str(word),
                                "record_id": str(meaning_id) if meaning_id else None,
                                "reason": (
                                    "verification retained competing readings and found a non-final "
                                    "lexical-semantic link; retrieve context and alternatives"
                                ),
                            })
                for request in structure.get("usg_feedback_requests", ()):
                    if isinstance(request, Mapping):
                        key = (
                            str(request.get("target_module", "ore.blackboard.semantic_specialist")),
                            str(request.get("query", "")).casefold(),
                            str(request.get("record_id")) if request.get("record_id") else None,
                        )
                        if key not in seen_requests:
                            seen_requests.add(key)
                            feedback_requests.append(dict(request))
            findings.append({
                "field": "unresolved_questions",
                "key": "question:sense_ambiguity",
                "finding_type": "unresolved_question",
                "payload": {
                    "question_id": "question:sense_ambiguity",
                    "question": "Competing semantic readings remain; request context or source evidence before selecting a reading.",
                    "status": "unresolved",
                    "alternatives": labels,
                    "resolved": False,
                    "usg_feedback_requests": feedback_requests,
                },
                "source": "module_inference",
                "rules_applied": (
                    "competing_readings_check",
                    "request_missing_or_ambiguous_knowledge",
                    "preserve_alternatives_without_selection",
                ),
                "dependencies": sorted(dependencies),
            })
        else:
            structure = semantics[0]
            allowed = True
            for rule in rules:
                predicate = rule.get("requires_verb")
                if predicate and predicate != structure.get("predicate"):
                    continue
                if rule.get("forbid_unknown_arguments") and (
                    structure.get("patient_sense") == "unknown"
                    or structure.get("agent_sense") == "unknown"
                ):
                    allowed = False
            findings.append({
                "field": "unresolved_questions",
                "key": "verify:composition",
                "finding_type": "verification_record",
                "payload": {
                    "question_id": "verify:composition",
                    "question": "Single reading checked against verification rules.",
                    "status": "verified" if allowed else "rejected",
                    "resolved": allowed,
                    "proposition": structure.get("proposition"),
                },
                "source": "module_inference",
                "rules_applied": ("rule_set_check",) if rules else ("structural_check",),
                "dependencies": sorted(dependencies),
            })
        return ModuleResult(
            module_id=self.module_id,
            module_version=_BLACKBOARD_PACKAGE_VERSION,
            status=ModuleStatus.SUCCESS,
            artifacts=(self._batch(
                findings, context.run_id, self.module_id, {"competing": competing},
            ),),
            diagnostics={"competing_readings": competing, "rules_supplied": bool(rules)},
            provenance={"upstream": "semantic_hypotheses"},
        )


def default_specialists() -> tuple[Any, ...]:
    """Return the reference specialist set for the sentence-understanding workload."""
    return (
        LexicalSpecialist(),
        SyntaxSpecialist(),
        SemanticSpecialist(),
        MemorySpecialist(),
        VerificationSpecialist(),
    )


__all__ = [
    "FALLBACK_LEXICON",
    "LexicalSpecialist",
    "MemorySpecialist",
    "SemanticSpecialist",
    "SyntaxSpecialist",
    "VerificationSpecialist",
    "default_specialists",
]
