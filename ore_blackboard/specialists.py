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
    subscription_fields = ("current_input", "artifact_references")
    sensitivity_weights = {"current_input": 1.0, "artifact_references": 1.0}
    activation_threshold = 0.5

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
        if "usg_lexicon" in inputs:
            usg_lexicon = inputs["usg_lexicon"].payload
            usg_meta = inputs["usg_lexicon"].metadata
        snapshot_payload = inputs["workspace_snapshot"].payload
        input_root_finding_id = self._input_root_finding_id(snapshot_payload)
        tokens = root_payload["tokens"]
        findings: list[dict[str, Any]] = []
        used_usg_records: list[str] = []
        unknown_tokens: list[str] = []
        for index, token_record in enumerate(tokens):
            token = token_record["token"]
            candidates = self._candidates(token, usg_lexicon)
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
                    },
                    "source": "current_input",
                    "source_span": [token_record["start"], token_record["end"]],
                    "rules_applied": ("token_lookup_miss",),
                    "confidence": 0.1,
                    "dependencies": (
                        [input_root_finding_id] if input_root_finding_id else []
                    ),
                })
                continue
            for source_label, gloss, part_of_speech, prior, usg_reference in candidates:
                envelope: dict[str, Any] = {
                    "field": "lexical_hypotheses",
                    "key": token.casefold(),
                    "finding_type": "lexical_hypothesis",
                    "payload": {
                        "word": token,
                        "candidate": gloss,
                        "gloss": gloss,
                        "part_of_speech": part_of_speech,
                        "source_label": source_label,
                        "token_index": index,
                    },
                    "source": "current_input",
                    "source_span": [token_record["start"], token_record["end"]],
                    "rules_applied": ("exact_surface_lookup", "preserve_competing_senses"),
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
                    }]
                    used_usg_records.append(usg_reference["record_id"])
                findings.append(envelope)
        diagnostics = {
            "token_count": len(tokens),
            "unknown_tokens": unknown_tokens,
            "usg_sourced": bool(used_usg_records),
            "usg_bundle_id": (usg_meta or {}).get("usg_bundle_id"),
            "confidence_semantics": "rule-based lookup strength; not a calibrated probability",
        }
        return ModuleResult(
            module_id=self.module_id,
            module_version=_BLACKBOARD_PACKAGE_VERSION,
            status=ModuleStatus.SUCCESS,
            artifacts=(self._batch(findings, context.run_id, self.module_id, diagnostics),),
            diagnostics=diagnostics,
            provenance={"lexicon": "usg" if usg_lexicon else "builtin_fallback"},
        )

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
    """Compose propositions, preserving competing readings (Step 21 demo)."""

    module_id = "ore.blackboard.semantic_specialist"
    write_fields = {"semantic_out": "semantic_hypotheses"}
    subscription_fields = ("lexical_hypotheses", "syntactic_hypotheses")
    sensitivity_weights = {"lexical_hypotheses": 0.2, "syntactic_hypotheses": 0.8}
    activation_threshold = 0.5

    def describe(self) -> ModuleManifest:
        """Return this specialist's validated manifest (Step 6)."""
        return ModuleManifest(
            identity=ModuleIdentity(
                module_id=self.module_id,
                module_version=_BLACKBOARD_PACKAGE_VERSION,
                module_name="Reference semantic composition specialist",
                package_path=self.package_path,
            ),
            responsibilities=(
                "Compose propositions from structure and lexical candidates.",
                "Preserve competing interpretations when lexical senses remain open.",
            ),
            input_ports=(
                InputPort("workspace_snapshot", BLACKBOARD_SNAPSHOT_CONTRACT, ("1.0.0",),
                          required=True, expected_data_type="builtins.dict"),
            ),
            output_ports=(
                OutputPort("semantic_out", BLACKBOARD_FINDING_BATCH_CONTRACT, "1.0.0",
                           "builtins.dict", "Proposition-level semantic findings."),
            ),
            capabilities=("language.semantic_composition",),
            dependencies=(ModuleDependency("ore.blackboard.syntax_specialist", required=True),),
            limitations=("Reference toy compositor; no world knowledge or learned semantics.",),
        )

    def execute(self, inputs: Mapping[str, Any], context: Any) -> ModuleResult:
        """Compose one proposition per open sense combination of the arguments."""
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

        def senses(word):
            records = lexical.get(str(word).casefold(), []) if word else []
            return [
                {
                    "gloss": record.get("gloss", record.get("candidate", "unknown")),
                    "finding_id": record.get("finding_id"),
                    "unknown": bool(record.get("unknown")),
                }
                for record in records
            ] or [{"gloss": "unknown", "finding_id": None, "unknown": True}]

        subject_senses = senses(subject_word)
        object_senses = senses(object_word) if object_word else [{"gloss": None}]
        verb_senses = senses(verb_word)
        findings: list[dict[str, Any]] = []
        dependencies: set[str] = set()
        if structure.get("finding_id"):
            dependencies.add(structure["finding_id"])
        for subject_sense in subject_senses:
            for object_sense in object_senses:
                proposition = (
                    f"{subject_word}({subject_sense['gloss']}) "
                    f"{verb_word}({verb_senses[0]['gloss']}) "
                    f"{object_word}({object_sense['gloss']})".strip()
                )
                for sense in (subject_sense, object_sense, verb_senses[0]):
                    if sense.get("finding_id"):
                        dependencies.add(sense["finding_id"])
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
                    },
                    "source": "module_inference",
                    "rules_applied": ("predicate_argument_composition", "preserve_open_senses"),
                    "dependencies": sorted(dependencies),
                    "confidence": 0.55,
                })
        competing = len(findings) > 1
        return ModuleResult(
            module_id=self.module_id,
            module_version=_BLACKBOARD_PACKAGE_VERSION,
            status=ModuleStatus.SUCCESS,
            artifacts=(self._batch(
                findings, context.run_id, self.module_id,
                {"competing_interpretations": competing, "proposition_count": len(findings)},
            ),),
            diagnostics={
                "competing_interpretations": competing,
                "proposition_count": len(findings),
            },
            provenance={"upstream": "syntactic_hypotheses"},
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
            findings.append({
                "field": "unresolved_questions",
                "key": "question:sense_ambiguity",
                "finding_type": "unresolved_question",
                "payload": {
                    "question_id": "question:sense_ambiguity",
                    "question": "Competing semantic readings remain; select one reading.",
                    "status": "unresolved",
                    "alternatives": labels,
                    "resolved": False,
                },
                "source": "module_inference",
                "rules_applied": ("competing_readings_check",),
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
