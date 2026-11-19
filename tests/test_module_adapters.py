"""Legacy adapter tests, including a real lexical-to-syntax-to-semantic ORE pipeline."""
import re

from ore_contracts import ComputeBudget, OperationMode, RunContext, make_envelope
from ore_contracts.contracts_registry import ContractCatalog
from ore_contracts.module_adapters import LegacyProcessorAdapter
from ore_contracts.module_artifacts import ContractArtifact
from ore_contracts.module_compatibility import CompatibilityValidator
from ore_contracts.module_manifest import (
    InputPort, ModuleDependency, ModuleIdentity, ModuleManifest, OutputPort,
)
from ore_contracts.module_protocol import ModuleStatus
from ore_contracts.module_registry import ModuleRegistry
from ore_lexical_access import LexicalAccessConfig, LexicalAccessProcessor, LexicalAccessRequest
from ore_lexical_context import LexicalEntry
from ore_semantic import SemanticConcept
from ore_semantic_composition import (
    PredicateFrame, SemanticCompositionConfig, SemanticCompositionProcessor,
    composition_request_from_results,
)
from ore_syntax import SyntaxConfig, SyntaxProcessor, SyntaxRequest


class _BatchLexicalProcessor:
    """Run the legacy one-word lexical processor over a sentence without changing its API."""

    def __init__(self, processor):
        """Store the existing lexical-access processor used for each word token."""
        self.processor = processor

    def process(self, text):
        """Return token-indexed POS evidence and the full per-word lexical result records."""
        tokens = re.findall(r"[^\W_]+", text, flags=re.UNICODE)
        evidence = {}
        results = []
        category_names = {
            "noun": "NOUN", "proper_noun": "PROPN", "verb": "VERB",
            "adjective": "ADJ", "adverb": "ADV", "determiner": "DET",
        }
        for index, token in enumerate(tokens):
            result = self.processor.process(LexicalAccessRequest(
                sequence_id=f"lex-{index}-{token}", word=token, language="en",
            ))
            categories = {
                category_names[item["category"]]: float(item["posterior"])
                for item in result.grammatical_categories
                if item["category"] in category_names
            }
            if categories:
                evidence[index] = categories
            results.append(result.to_dict())
        return {"category_evidence": evidence, "word_results": results}


def test_existing_lexical_syntax_and_semantic_processors_exchange_typed_artifacts():
    """Connect three existing processors through validated ports and ContractArtifact envelopes."""
    from ore_semantic_composition.contracts import SemanticCompositionResult
    from ore_syntax.contracts import SyntaxResult

    catalog = ContractCatalog()
    catalog.register_model("ORE.TEXT.INPUT", "1.0.0", str)
    catalog.register_model("ORE.LEXICAL.BATCH", "1.0.0", dict)
    catalog.register_model("ORE.SYNTAX.RESULT", "1.0.0", SyntaxResult, required_fields=("tokens",))
    catalog.register_model("ORE.SEMANTIC.RESULT", "1.0.0", SemanticCompositionResult)

    lexical_manifest = ModuleManifest(
        ModuleIdentity("ore.lexical_access", "1.0.0", "ORE Lexical Access", package_path="ore_lexical_access"),
        ("retrieve word identity, meaning, morphology, and category evidence",),
        (InputPort("text", "ORE.TEXT.INPUT", ("1.0.0",), expected_data_type="builtins.str"),),
        (OutputPort(
            "lexical_batch", "ORE.LEXICAL.BATCH", "1.0.0", "builtins.dict",
            "per-token lexical results and grammatical-category evidence",
        ),),
        ("language.lexical_access",),
        execution_modes=("FUNCTIONAL", "BIOLOGICAL"),
    )
    syntax_manifest = ModuleManifest(
        ModuleIdentity("ore.syntax", "1.0.0", "ORE Syntax", package_path="ore_syntax"),
        ("parse sentence tokens and dependencies using optional lexical evidence",),
        (
            InputPort("text", "ORE.TEXT.INPUT", ("1.0.0",), expected_data_type="builtins.str"),
            InputPort(
                "lexical", "ORE.LEXICAL.BATCH", ("1.0.0",),
                required=False, expected_data_type="builtins.dict",
            ),
        ),
        (OutputPort(
            "syntax", "ORE.SYNTAX.RESULT", "1.0.0",
            "ore_syntax.contracts.SyntaxResult", "typed syntax parse result",
        ),),
        ("language.syntax_parsing",),
        dependencies=(ModuleDependency("ore.lexical_access", required=True),),
        execution_modes=("FUNCTIONAL", "BIOLOGICAL"),
    )
    semantic_manifest = ModuleManifest(
        ModuleIdentity(
            "ore.semantic_composition", "1.0.0", "Semantic Composition",
            package_path="ore_semantic_composition",
        ),
        ("compose token concepts and predicate roles",),
        (InputPort(
            "syntax", "ORE.SYNTAX.RESULT", ("1.0.0",),
            expected_data_type="ore_syntax.contracts.SyntaxResult",
        ),),
        (OutputPort(
            "semantic", "ORE.SEMANTIC.RESULT", "1.0.0",
            "ore_semantic_composition.contracts.SemanticCompositionResult",
            "sentence-level semantic composition",
        ),),
        ("language.semantic_composition",),
        dependencies=(ModuleDependency("ore.syntax", required=True),),
        execution_modes=("FUNCTIONAL", "BIOLOGICAL"),
    )

    lexicon = (
        LexicalEntry("scientist", "scientist", frequency=15, lexical_category="noun", language="en"),
        LexicalEntry("studies", "studies", frequency=12, lexical_category="verb", language="en"),
        LexicalEntry("neurons", "neurons", frequency=18, lexical_category="noun", language="en"),
    )
    lexical_processor = LexicalAccessProcessor(
        lexicon, config=LexicalAccessConfig(simulate_biology=False)
    )
    lexical_backend = LegacyProcessorAdapter(
        lexical_manifest,
        _BatchLexicalProcessor(lexical_processor),
        lambda inputs, context: inputs["text"].payload,
        lambda result, context: ContractArtifact(
            make_envelope("ORE.LEXICAL.BATCH", context.run_id, "ore.lexical_access"),
            result,
        ),
        contract_catalog=catalog,
    )

    syntax_processor = SyntaxProcessor(config=SyntaxConfig(simulate_biology=False))
    syntax_backend = LegacyProcessorAdapter(
        syntax_manifest,
        syntax_processor,
        lambda inputs, context: SyntaxRequest(
            sequence_id=context.run_id,
            text=inputs["text"].payload,
            lexical_evidence=(
                inputs["lexical"].payload["category_evidence"]
                if inputs.get("lexical") is not None else {}
            ),
        ),
        lambda result, context: ContractArtifact(
            make_envelope("ORE.SYNTAX.RESULT", context.run_id, "ore.syntax"),
            result,
        ),
        contract_catalog=catalog,
    )

    concepts = (
        SemanticConcept(
            "scientist", "scientist", semantic_features={"HUMAN": 1.0, "AGENTIVE": 0.9},
            lexical_category="noun", verified_by="test", source_reference="test://scientist",
        ),
        SemanticConcept(
            "study", "studies", aliases=("study",),
            semantic_features={"EVENT": 0.8, "COGNITION": 1.0},
            lexical_category="verb", verified_by="test", source_reference="test://studies",
        ),
        SemanticConcept(
            "neuron", "neurons", aliases=("neuron",),
            semantic_features={"BIOLOGICAL_ENTITY": 1.0, "CELL": 0.9},
            lexical_category="noun", verified_by="test", source_reference="test://neurons",
        ),
    )
    frame = PredicateFrame(
        predicate="studies", predicate_concept_id="study",
        roles=("Agent", "Patient"), required_roles=("Agent", "Patient"),
        selectional_preferences={
            "Agent": {"HUMAN": 1.0, "AGENTIVE": 0.8},
            "Patient": {"BIOLOGICAL_ENTITY": 1.0},
        },
        source_reference="test://frames/studies",
    )
    semantic_processor = SemanticCompositionProcessor(
        concepts, (frame,), SemanticCompositionConfig(simulate_biology=False)
    )
    semantic_backend = LegacyProcessorAdapter(
        semantic_manifest,
        semantic_processor,
        lambda inputs, context: composition_request_from_results(
            sequence_id=context.run_id, syntax_result=inputs["syntax"].payload,
        ),
        lambda result, context: ContractArtifact(
            make_envelope("ORE.SEMANTIC.RESULT", context.run_id, "ore.semantic_composition"),
            result,
        ),
        contract_catalog=catalog,
    )

    registry = ModuleRegistry(catalog)
    for manifest in (lexical_manifest, syntax_manifest, semantic_manifest):
        registry.register(manifest)
    registry.bind_backend("ore.lexical_access", lexical_backend)
    registry.bind_backend("ore.syntax", syntax_backend)
    registry.bind_backend("ore.semantic_composition", semantic_backend)
    compatibility = CompatibilityValidator(registry)
    assert compatibility.validate_connection(
        "ore.lexical_access", "ore.syntax", "lexical_batch", "lexical"
    ).compatible
    assert compatibility.validate_connection(
        "ore.syntax", "ore.semantic_composition", "syntax", "syntax"
    ).compatible
    assert registry.diagnostics("ore.semantic_composition")["dependency_complete"]

    context = RunContext(
        run_id="universal-communication-integration",
        mode=OperationMode.FUNCTIONAL,
        deterministic_seed=7,
        compute_budget=ComputeBudget(),
    )
    text_artifact = ContractArtifact(
        make_envelope("ORE.TEXT.INPUT", context.run_id, "ore.input"),
        "The scientist studies neurons.",
    )
    lexical_result = lexical_backend.execute({"text": text_artifact}, context)
    assert lexical_result.status is ModuleStatus.SUCCESS
    assert len(lexical_result.artifacts[0].payload["word_results"]) == 4
    syntax_result = syntax_backend.execute(
        {"text": text_artifact, "lexical": lexical_result.artifacts[0]}, context
    )
    assert syntax_result.status is ModuleStatus.SUCCESS
    assert catalog.validate_artifact(syntax_result.artifacts[0]).valid
    assert compatibility.validate_connection(
        "ore.syntax", "ore.semantic_composition", "syntax", "syntax",
        artifact=syntax_result.artifacts[0],
    ).compatible

    semantic_result = semantic_backend.execute({"syntax": syntax_result.artifacts[0]}, context)
    assert semantic_result.status is ModuleStatus.SUCCESS
    assert catalog.validate_artifact(semantic_result.artifacts[0]).valid
    assert semantic_result.artifacts[0].payload.diagnostics["syntax_supplied"] is True


def test_existing_canonical_controller_and_backend_registry_can_be_adapted():
    """Wrap the existing B-N canonical controller without replacing its backend registry."""
    from ore_contracts import CanonicalBackendRegistry, CanonicalOREController
    from ore_contracts.module_adapters import CanonicalControllerAdapter

    catalog = ContractCatalog()
    catalog.register_model("ORE.TEXT.INPUT", "1.0.0", str)
    catalog.register_model("ORE.CANONICAL.SUMMARY", "1.0.0", dict)
    manifest = ModuleManifest(
        ModuleIdentity("ore.canonical_controller", "1.0.0", "Canonical controller"),
        ("run the existing canonical ORE controller",),
        (InputPort("input", "ORE.TEXT.INPUT", ("1.0.0",), expected_data_type="builtins.str"),),
        (OutputPort(
            "run", "ORE.CANONICAL.SUMMARY", "1.0.0", "builtins.dict",
            "canonical controller run summary",
        ),),
        ("system.canonical_orchestration",),
        execution_modes=("FUNCTIONAL", "BIOLOGICAL", "HYBRID", "VALIDATE"),
    )
    controller = CanonicalOREController(CanonicalBackendRegistry())
    adapter = CanonicalControllerAdapter(
        manifest,
        controller,
        "input",
        lambda result, context: ContractArtifact(
            make_envelope("ORE.CANONICAL.SUMMARY", context.run_id, "ore.canonical_controller"),
            {
                "run_id": result.run_id,
                "mode": result.mode.value,
                "channel_status": dict(result.channel_status),
                "diagnostics": dict(result.diagnostics),
            },
        ),
        contract_catalog=catalog,
    )
    registry = ModuleRegistry(catalog)
    registry.register(manifest)
    registry.bind_backend("ore.canonical_controller", adapter)
    context = RunContext(
        run_id="canonical-controller-adapter",
        mode=OperationMode.FUNCTIONAL,
        deterministic_seed=3,
        compute_budget=ComputeBudget(),
    )
    source = ContractArtifact(
        make_envelope("ORE.TEXT.INPUT", context.run_id, "ore.input"),
        "test payload",
    )
    result = adapter.execute({"input": source}, context)
    assert result.status is ModuleStatus.PARTIAL
    assert result.artifacts[0].payload["channel_status"]["FUNCTIONAL"] == "CHANNEL_UNAVAILABLE"
    assert catalog.validate_artifact(result.artifacts[0]).valid
