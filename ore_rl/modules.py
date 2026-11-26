"""Deterministic example cognitive modules registered through real ORE contracts.

These modules give the Gymnasium-compatible environment (and every test and
demo) a runnable cognitive pipeline without depending on heavyweight external
lexicons. They are registered exactly like any other ORE module:

    ContractCatalog.register_model -> ModuleRegistry.register(ModuleManifest)
    -> ModuleRegistry.bind_backend(backend with describe()/execute())

Execution happens through ``CognitiveTaskCoordinator.coordinate`` in the
environment, so the RL layer never bypasses the coordinator's validation,
timeouts, or workspace publication (Step 19).

Deterministic behavior contract: for identical inputs, seeds, and depth the
modules produce identical outputs, which the environment relies on for
reproducible evaluation.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Mapping

from ore_contracts import (
    ContractArtifact,
    ContractCatalog,
    InputPort,
    ModuleIdentity,
    ModuleManifest,
    ModuleRegistry,
    ModuleResult,
    ModuleStatus,
    OutputPort,
    RunContext,
)


MODULE_PREFIX = "ore.rl"

CONTRACT_INPUT_TEXT = "ORE.RL.INPUT_TEXT"
CONTRACT_INPUT_ANALYSIS = "ORE.RL.INPUT_ANALYSIS"
CONTRACT_KNOWLEDGE_RECORDS = "ORE.RL.KNOWLEDGE_RECORDS"
CONTRACT_HYPOTHESIS_SET = "ORE.RL.HYPOTHESIS_SET"
CONTRACT_CAUSAL_CHECK = "ORE.RL.CAUSAL_CHECK"
CONTRACT_VERIFIED_CONCLUSION = "ORE.RL.VERIFIED_CONCLUSION"
CONTRACT_CONCURRENCY_SUPPORT = "ORE.RL.CONCURRENCY_SUPPORT"

CAP_INPUT_ANALYSIS = "cognition.rl.input_analysis"
CAP_KNOWLEDGE_RETRIEVAL = "cognition.rl.knowledge_retrieval"
CAP_HYPOTHESIS_GENERATION = "cognition.rl.hypothesis_generation"
CAP_CAUSAL_ANALYSIS = "cognition.rl.causal_analysis"
CAP_VERIFICATION = "cognition.rl.verification"

ALL_CAPABILITIES = (
    CAP_INPUT_ANALYSIS,
    CAP_KNOWLEDGE_RETRIEVAL,
    CAP_HYPOTHESIS_GENERATION,
    CAP_CAUSAL_ANALYSIS,
    CAP_VERIFICATION,
)

MODULE_IDS = (
    f"{MODULE_PREFIX}.input_analyzer",
    f"{MODULE_PREFIX}.knowledge_retriever",
    f"{MODULE_PREFIX}.hypothesis_generator",
    f"{MODULE_PREFIX}.causal_analyzer",
    f"{MODULE_PREFIX}.verifier",
)


@dataclass(frozen=True, slots=True)
class SimulatedCostProfile:
    """Deterministic cost model per module and depth level (Steps 12, 23)."""

    base_latency_ms: float
    latency_per_depth: float
    base_cpu_cost: float
    cpu_per_depth: float
    memory_mb: float

    def at_depth(self, depth: int) -> tuple[float, float, float]:
        depth = max(1, min(5, int(depth)))
        return (
            self.base_latency_ms + self.latency_per_depth * depth,
            self.base_cpu_cost + self.cpu_per_depth * depth,
            self.memory_mb,
        )


DEFAULT_COST_PROFILES: Mapping[str, SimulatedCostProfile] = {
    f"{MODULE_PREFIX}.input_analyzer": SimulatedCostProfile(8.0, 1.5, 0.8, 0.2, 2.0),
    f"{MODULE_PREFIX}.knowledge_retriever": SimulatedCostProfile(20.0, 6.0, 2.0, 1.0, 8.0),
    f"{MODULE_PREFIX}.hypothesis_generator": SimulatedCostProfile(24.0, 5.0, 2.5, 0.9, 6.0),
    f"{MODULE_PREFIX}.causal_analyzer": SimulatedCostProfile(18.0, 4.0, 1.8, 0.7, 5.0),
    f"{MODULE_PREFIX}.verifier": SimulatedCostProfile(30.0, 7.0, 3.0, 1.2, 9.0),
}


_LEXICON_TAGS: Mapping[str, str] = {
    "the": "DET",
    "a": "DET",
    "an": "DET",
    "claim": "FRAME",
    "define": "FRAME",
    "term": "FRAME",
    "verify": "FRAME",
    "reported": "MOD",
    "primary": "MOD",
    "observed": "MOD",
    "canonical": "MOD",
    "secondary": "MOD",
    "technical": "MOD",
    "governs": "VERB",
    "governed": "VERB",
    "recalled": "VERB",
    "used": "VERB",
    "influences": "VERB",
    "after": "PREP",
    "in": "PREP",
    "by": "PREP",
    "was": "AUX",
    "response": "GENERIC",
    "event": "GENERIC",
    "research": "GENERIC",
}


def _tag_tokens(text: str) -> list[tuple[str, str, int]]:
    """Tokenize, lexicon-tag, and keep first-occurrence order with positions."""
    raw = [token.strip(".,!?;:").lower() for token in text.split()]
    raw = [token for token in raw if token]
    return [
        (token, _LEXICON_TAGS.get(token, "NOUN"), position)
        for position, token in enumerate(raw)
    ]


def register_rl_contracts(catalog: ContractCatalog) -> None:
    """Register the typed payload models used by the example modules."""
    catalog.register_model(CONTRACT_INPUT_TEXT, "1.0.0", str)
    catalog.register_model(CONTRACT_INPUT_ANALYSIS, "1.0.0", dict)
    catalog.register_model(CONTRACT_KNOWLEDGE_RECORDS, "1.0.0", dict)
    catalog.register_model(CONTRACT_HYPOTHESIS_SET, "1.0.0", dict)
    catalog.register_model(CONTRACT_CAUSAL_CHECK, "1.0.0", dict)
    catalog.register_model(CONTRACT_VERIFIED_CONCLUSION, "1.0.0", dict)
    catalog.register_model(CONTRACT_CONCURRENCY_SUPPORT, "1.0.0", dict)


def build_module_manifests() -> tuple[ModuleManifest, ...]:
    """Manifests for the five example cognitive modules.

    Module dependencies are intentionally NOT declared as ``ModuleDependency``
    records: the planner would then re-recruit upstream modules on every
    single-module RL step. Instead, prerequisite ordering is enforced by the
    RL action catalogue (``prerequisite_capabilities``) and satisfied inputs
    are supplied as validated external artifacts, so a step executes exactly
    the requested module through the real coordinator.
    """
    return (
        ModuleManifest(
            ModuleIdentity(f"{MODULE_PREFIX}.input_analyzer", "1.0.0", "RL input analyzer"),
            ("analyze the task input into key terms and coverage signals",),
            (InputPort("source_text", CONTRACT_INPUT_TEXT, ("1.0.0",)),),
            (
                OutputPort(
                    "input_analysis",
                    CONTRACT_INPUT_ANALYSIS,
                    "1.0.0",
                    "builtins.dict",
                    "token and key-term analysis",
                ),
            ),
            (CAP_INPUT_ANALYSIS,),
        ),
        ModuleManifest(
            ModuleIdentity(f"{MODULE_PREFIX}.knowledge_retriever", "1.0.0", "RL knowledge retriever"),
            ("retrieve supporting knowledge records for the analyzed input",),
            (InputPort("input_analysis", CONTRACT_INPUT_ANALYSIS, ("1.0.0",)),),
            (
                OutputPort(
                    "knowledge_records",
                    CONTRACT_KNOWLEDGE_RECORDS,
                    "1.0.0",
                    "builtins.dict",
                    "retrieved knowledge records",
                ),
            ),
            (CAP_KNOWLEDGE_RETRIEVAL,),
        ),
        ModuleManifest(
            ModuleIdentity(
                f"{MODULE_PREFIX}.hypothesis_generator", "1.0.0", "RL hypothesis generator"
            ),
            ("compose a bounded hypothesis set from analysis and knowledge",),
            (
                InputPort("input_analysis", CONTRACT_INPUT_ANALYSIS, ("1.0.0",)),
                InputPort("knowledge_records", CONTRACT_KNOWLEDGE_RECORDS, ("1.0.0",)),
            ),
            (
                OutputPort(
                    "hypothesis_set",
                    CONTRACT_HYPOTHESIS_SET,
                    "1.0.0",
                    "builtins.dict",
                    "bounded candidate hypotheses",
                ),
            ),
            (CAP_HYPOTHESIS_GENERATION,),
        ),
        ModuleManifest(
            ModuleIdentity(f"{MODULE_PREFIX}.causal_analyzer", "1.0.0", "RL causal analyzer"),
            ("check causal consistency among the generated hypotheses",),
            (InputPort("hypothesis_set", CONTRACT_HYPOTHESIS_SET, ("1.0.0",)),),
            (
                OutputPort(
                    "causal_check",
                    CONTRACT_CAUSAL_CHECK,
                    "1.0.0",
                    "builtins.dict",
                    "causal consistency report",
                ),
            ),
            (CAP_CAUSAL_ANALYSIS,),
        ),
        ModuleManifest(
            ModuleIdentity(f"{MODULE_PREFIX}.verifier", "1.0.0", "RL verifier"),
            ("verify the hypothesis set against retrieved knowledge",),
            (
                InputPort("hypothesis_set", CONTRACT_HYPOTHESIS_SET, ("1.0.0",)),
                InputPort("knowledge_records", CONTRACT_KNOWLEDGE_RECORDS, ("1.0.0",)),
            ),
            (
                OutputPort(
                    "verified_conclusion",
                    CONTRACT_VERIFIED_CONCLUSION,
                    "1.0.0",
                    "builtins.dict",
                    "independently verified conclusion",
                ),
            ),
            (CAP_VERIFICATION,),
        ),
    )


def _artifact(
    contract_id: str, run_id: str, producer: str, payload: Any
) -> ContractArtifact:
    from ore_contracts.adapters import make_envelope

    return ContractArtifact(
        make_envelope(contract_id, run_id, producer, evidence_kind="DERIVED"),
        payload,
    )


class _ExampleBackend:
    """Shared backend behavior: describe(), execute(), and deterministic outputs."""

    def __init__(
        self,
        manifest: ModuleManifest,
        *,
        cost_profile: SimulatedCostProfile,
        reliability: float = 0.95,
        failure_injection: Mapping[str, float] | None = None,
    ) -> None:
        self.manifest = manifest
        self.cost_profile = cost_profile
        self.reliability = reliability
        self.failure_injection = dict(failure_injection or {})
        self.execution_count = 0

    def describe(self) -> ModuleManifest:
        return self.manifest

    @property
    def module_id(self) -> str:
        return self.manifest.identity.module_id

    def execute(self, inputs: Mapping[str, Any], context: RunContext) -> ModuleResult:
        try:
            settings = json.loads(context.settings_json or "{}")
            depth = int(settings.get("rl_depth", 2))
        except (TypeError, ValueError):
            depth = 2
        depth = max(1, min(5, depth))
        latency, cpu, memory = self.cost_profile.at_depth(depth)
        self.execution_count += 1
        time.sleep(min(0.002, latency / 20000.0))
        from .contracts import stable_hash as _stable_hash

        draw = int(_stable_hash({"run": context.run_id, "module": self.module_id, "depth": depth})[:8], 16)
        rng_state = draw / 0xFFFFFFFF
        if rng_state < self.failure_injection.get(self.module_id, 0.0):
            return ModuleResult(
                self.module_id,
                self.manifest.identity.module_version,
                ModuleStatus.FAILED,
                diagnostics={"failure": "injected_failure", "depth": depth},
            )
        payload = self._produce(inputs, context, depth)
        output_contract = self.manifest.output_ports[0].contract_id
        artifact = _artifact(output_contract, context.run_id, self.module_id, payload)
        return ModuleResult(
            self.module_id,
            self.manifest.identity.module_version,
            ModuleStatus.SUCCESS,
            artifacts=(artifact,),
            diagnostics={
                "capability_support": {self.manifest.capabilities[0]: self.reliability},
                "rl_depth": depth,
                "measured_latency_ms": latency,
                "measured_cpu_cost": cpu,
                "measured_memory_mb": memory,
            },
            processing_metadata={
                "rl_depth": depth,
                "latency_ms": latency,
                "cpu_cost": cpu,
                "memory_mb": memory,
            },
        )

    def _produce(
        self, inputs: Mapping[str, Any], context: RunContext, depth: int
    ) -> Mapping[str, Any]:
        raise NotImplementedError


class InputAnalyzerBackend(_ExampleBackend):
    """N1-style deterministic token and key-term analysis."""

    def _produce(
        self, inputs: Mapping[str, Any], context: RunContext, depth: int
    ) -> Mapping[str, Any]:
        text = ""
        for value in inputs.values():
            if isinstance(value, ContractArtifact):
                text = str(value.payload)
                break
        tagged = _tag_tokens(text)
        nouns = [
            {"term": token, "token_position": position}
            for token, tag, position in tagged
            if tag == "NOUN"
        ]
        key_terms = [str(item["term"]) for item in nouns]
        return {
            "tokens": [token for token, _, _ in tagged],
            "token_count": len(tagged),
            "key_terms": key_terms,
            "noun_positions": [int(item["token_position"]) for item in nouns],
            "subject_candidate": key_terms[0] if key_terms else None,
            "coverage_terms": key_terms,
            "depth": depth,
        }


class KnowledgeRetrieverBackend(_ExampleBackend):
    """Deterministic simulated knowledge retrieval with depth-dependent recall.

    If a USG knowledge resource is attached at environment construction, the
    retriever consults it read-only through ``USGKnowledgeIndex.lookup``.
    """

    def __init__(self, manifest: ModuleManifest, **kwargs: Any) -> None:
        super().__init__(manifest, **kwargs)
        self.usg_index: Any | None = None

    def _produce(
        self, inputs: Mapping[str, Any], context: RunContext, depth: int
    ) -> Mapping[str, Any]:
        analysis = self._payload_of(inputs, CONTRACT_INPUT_ANALYSIS)
        key_terms = list(analysis.get("key_terms", ())) if analysis else []
        positions = list(analysis.get("noun_positions", ())) if analysis else []
        records: list[dict[str, Any]] = []
        for index, term in enumerate(key_terms):
            token_position = int(positions[index]) if index < len(positions) else index
            usg_match = None
            if self.usg_index is not None:
                retrieval = self.usg_index.lookup(term)
                if retrieval.exact_match and retrieval.items:
                    usg_match = retrieval.items[0].record_id
            reliability = 0.5 + 0.1 * min(depth, 4)
            records.append(
                {
                    "term": term,
                    "token_position": token_position,
                    "source": "usg" if usg_match else "internal_corpus",
                    "record_id": usg_match or f"rec_{index}",
                    "supports": token_position < depth,
                    "source_reliability": round(min(reliability, 0.95), 3),
                    "verified": bool(usg_match),
                }
            )
        return {
            "records": records,
            "record_count": len(records),
            "depth": depth,
            "consulted_usg": self.usg_index is not None,
        }

    @staticmethod
    def _payload_of(inputs: Mapping[str, Any], contract_id: str) -> Mapping[str, Any] | None:
        for value in inputs.values():
            if isinstance(value, ContractArtifact) and value.envelope.contract_id == contract_id:
                if isinstance(value.payload, Mapping):
                    return value.payload
        return None


class HypothesisGeneratorBackend(_ExampleBackend):
    """Bounded hypothesis composition from analysis and knowledge (Step 15)."""

    def _produce(
        self, inputs: Mapping[str, Any], context: RunContext, depth: int
    ) -> Mapping[str, Any]:
        analysis = KnowledgeRetrieverBackend._payload_of(inputs, CONTRACT_INPUT_ANALYSIS) or {}
        knowledge = KnowledgeRetrieverBackend._payload_of(
            inputs, CONTRACT_KNOWLEDGE_RECORDS
        ) or {}
        terms = list(analysis.get("key_terms", ()))
        records = list(knowledge.get("records", ()))
        support_by_term = {
            str(record.get("term")): bool(record.get("supports")) for record in records
        }
        reliability_by_term = {
            str(record.get("term")): float(record.get("source_reliability", 0.5))
            for record in records
        }
        candidates: list[dict[str, Any]] = []
        limit = max(1, depth)
        for term in terms[:limit]:
            supported = support_by_term.get(term, False)
            reliability = reliability_by_term.get(term, 0.5)
            candidates.append(
                {
                    "hypothesis": f"hypothesis::{term}",
                    "term": term,
                    "supported": supported,
                    "confidence": round(
                        (0.4 + 0.1 * depth) * reliability if supported else 0.2 * reliability, 3
                    ),
                }
            )
        return {
            "hypotheses": candidates,
            "hypothesis_count": len(candidates),
            "depth": depth,
        }


class CausalAnalyzerBackend(_ExampleBackend):
    """Deterministic causal consistency check over the hypothesis set."""

    def _produce(
        self, inputs: Mapping[str, Any], context: RunContext, depth: int
    ) -> Mapping[str, Any]:
        hypotheses = KnowledgeRetrieverBackend._payload_of(inputs, CONTRACT_HYPOTHESIS_SET) or {}
        candidates = list(hypotheses.get("hypotheses", ()))
        supported = [item for item in candidates if item.get("supported")]
        consistency = len(supported) / max(1, len(candidates))
        return {
            "consistent_fraction": round(consistency, 3),
            "conflicts": [
                item.get("term")
                for item in candidates
                if not item.get("supported")
            ][: max(0, 5 - depth)],
            "depth": depth,
        }


class VerifierBackend(_ExampleBackend):
    """Independent verification of hypotheses against retrieved knowledge."""

    def _produce(
        self, inputs: Mapping[str, Any], context: RunContext, depth: int
    ) -> Mapping[str, Any]:
        hypotheses = KnowledgeRetrieverBackend._payload_of(inputs, CONTRACT_HYPOTHESIS_SET) or {}
        knowledge = KnowledgeRetrieverBackend._payload_of(
            inputs, CONTRACT_KNOWLEDGE_RECORDS
        ) or {}
        candidates = list(hypotheses.get("hypotheses", ()))
        records = list(knowledge.get("records", ()))
        supported_terms = {str(record.get("term")) for record in records if record.get("supports")}
        best = None
        for item in candidates:
            eligible = item.get("supported") or item.get("term") in supported_terms
            if not eligible:
                continue
            if best is None or float(item.get("confidence", 0.0)) > float(
                best.get("confidence", 0.0)
            ):
                best = item
        verified = best is not None
        return {
            "verified": verified,
            "verified_hypothesis": best,
            "verified_term": best.get("term") if best else None,
            "supporting_record_count": len(supported_terms),
            "depth": depth,
        }


def build_example_backends(
    *,
    failure_injection: Mapping[str, float] | None = None,
) -> dict[str, _ExampleBackend]:
    """Instantiate all five example module backends."""
    failure_injection = failure_injection or {}
    backends: dict[str, _ExampleBackend] = {}
    for manifest in build_module_manifests():
        module_id = manifest.identity.module_id
        profile = DEFAULT_COST_PROFILES[module_id]
        cls = {
            f"{MODULE_PREFIX}.input_analyzer": InputAnalyzerBackend,
            f"{MODULE_PREFIX}.knowledge_retriever": KnowledgeRetrieverBackend,
            f"{MODULE_PREFIX}.hypothesis_generator": HypothesisGeneratorBackend,
            f"{MODULE_PREFIX}.causal_analyzer": CausalAnalyzerBackend,
            f"{MODULE_PREFIX}.verifier": VerifierBackend,
        }[module_id]
        backends[module_id] = cls(
            manifest,
            cost_profile=profile,
            failure_injection=failure_injection,
        )
    return backends


def register_rl_example_modules(
    catalog: ContractCatalog,
    registry: ModuleRegistry,
    *,
    failure_injection: Mapping[str, float] | None = None,
) -> dict[str, _ExampleBackend]:
    """Register contracts, manifests, and backends in one call (ore_usg pattern)."""
    register_rl_contracts(catalog)
    backends = build_example_backends(failure_injection=failure_injection)
    for manifest in build_module_manifests():
        registry.register(manifest)
        registry.bind_backend(manifest.identity.module_id, backends[manifest.identity.module_id])
    return backends
