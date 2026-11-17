"""Source-ordered Point 4 semantic composition over ORE's established composition engine.

This adapter turns Point 2 lexical-semantic candidates and Point 3 grammatical relations into
a provenance-preserving proposition graph. It maps the existing nine-stage/54-operation engine
to the source's 14 steps, adds modifier links and semantic-compatibility reporting, and recomputes
partial proposition states over each growing word prefix.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import islice, product
import math
from typing import Any, Mapping, Sequence

from ore_semantic_composition import (
    PredicateFrame,
    SemanticCompositionConfig,
    SemanticCompositionProcessor,
    SemanticCompositionRequest,
)
from ore_sensory_perception.contracts import to_jsonable


POINT4_STEP_SPECS: tuple[dict[str, Any], ...] = (
    {
        "operation_id": "LC-P04-01",
        "name": "Receive lexical-semantic candidates and the current syntactic structure",
        "objective": "Align word-sense alternatives, semantic records, dependency/constituency structure, morphology, confidence and provenance by token.",
        "mathematical_operations": ("C_i={(s_k,p_k)}", "p_k=a_k/(sum_j a_j+epsilon)", "state merge M_t=M_(t-1) union input_t"),
        "biological_reference": "Distributed temporal semantic and fronto-temporo-parietal language activity changes as lexical and syntactic evidence arrives; a software record merge is only an interface model.",
    },
    {
        "operation_id": "LC-P04-02",
        "name": "Activate semantic features relevant to the current composition",
        "objective": "Retrieve sense-specific concept features, predicate frames, entity properties and modifier contributions while keeping alternative senses distinct.",
        "mathematical_operations": ("x_rel=g elementwise-multiplied by x", "a_j=phi(sum_i w_ij*x_i-theta_j)", "cosine feature/role compatibility"),
        "biological_reference": "Distributed semantic populations and recurrent selection among feature patterns are candidate analogies; a feature vector is not a literal neural data format.",
    },
    {
        "operation_id": "LC-P04-03",
        "name": "Identify the semantic head of each local relation",
        "objective": "Use syntactic heads and lexical categories to identify predicates, state heads, entity heads, property heads and relational nouns.",
        "mathematical_operations": ("P(type|features,position)", "F_p=(r_1,...,r_m)", "predicate-frame arity and optional-slot lookup"),
        "biological_reference": "Temporal semantic representations and frontal/temporal language interactions contribute predicate and property information; no single semantic-composition centre is assumed.",
    },
    {
        "operation_id": "LC-P04-04",
        "name": "Create or update variables for entities and events",
        "objective": "Allocate sentence-local entity/event references, preserve concept features, and leave uncertain identity or incomplete constituents unresolved.",
        "mathematical_operations": ("x_i=entity variable; e_j=event variable", "identity constraint x_i=x_j only when supplied", "feature assignment f(x_i)"),
        "biological_reference": "Participant-specific distributed activity and recurrent maintenance are plausible analogies; literal symbolic variable allocation is not claimed.",
    },
    {
        "operation_id": "LC-P04-05",
        "name": "Map grammatical functions onto semantic argument roles",
        "objective": "Combine dependency relations, case, voice, agreement, construction and predicate-frame preferences without equating subject with agent or object with patient.",
        "mathematical_operations": ("S(r,x)=w_s*S_syntax+w_m*S_semantic", "argmax over compatible assignments", "soft role posteriors and feature unification"),
        "biological_reference": "Fronto-temporal networks integrate grammatical structure with lexical event knowledge; a literal combinatorial optimizer is not implied.",
    },
    {
        "operation_id": "LC-P04-06",
        "name": "Bind licensed arguments into predicate propositions",
        "objective": "Create predicate-centered event/state propositions and bind available entity variables to role slots while preserving open and competing slots.",
        "mathematical_operations": ("P(e)=PREDICATE(e)", "ROLE(e,x)", "G_t=G_(t-1) union {(e,r,x)}", "role/filler tensor product u_r tensor f_x as one candidate encoding"),
        "biological_reference": "Distributed event and participant representations must preserve relational structure; the exact neural binding mechanism remains unresolved.",
    },
    {
        "operation_id": "LC-P04-07",
        "name": "Compose modifiers and properties with their syntactic targets",
        "objective": "Attach adjectives, adverbs, possessives, compounds and other licensed modifiers to the correct entity, event or relation without confusing composition with association.",
        "mathematical_operations": ("h=phi(W_head*x_head+W_mod*x_mod+b)", "F_combined=F_head union gated(F_modifier)", "constraint narrowing and modifier-edge creation"),
        "biological_reference": "ATL, frontal, temporal and parietal contributions are reported across phrase-composition paradigms; their division of labour depends on task and construction.",
    },
    {
        "operation_id": "LC-P04-08",
        "name": "Compose grammatical operators with lexical content",
        "objective": "Represent negation, quantifiers, tense, aspect, modality, determiners and relevant particles with their scope and provenance.",
        "mathematical_operations": ("NEG(P)=not P", "FOR_ALL(x,R(x),P(x)); EXISTS(x,R(x),P(x))", "operator nesting and event/reference/speech time indices"),
        "biological_reference": "Distributed semantic and grammatical populations encode function-word and morphology contributions; literal symbolic theorem proving is not claimed.",
    },
    {
        "operation_id": "LC-P04-09",
        "name": "Evaluate semantic compatibility without performing pragmatic inference",
        "objective": "Compare participant and modifier features with predicate selectional preferences; report tensions without declaring unusual sentences impossible.",
        "mathematical_operations": ("C(type(x),role) in [0,1]", "E_sem=sum_i lambda_i*(1-C_i)", "anomaly=-log P(combination)", "entropy H=-sum_k p_k*log(p_k)"),
        "biological_reference": "Semantic anomalies can modulate temporal/frontal network activity and control demands; compatibility scores are model abstractions, not established neural probabilities.",
    },
    {
        "operation_id": "LC-P04-10",
        "name": "Maintain competing compositional readings",
        "objective": "Keep separate candidate propositions for unresolved word senses, role mappings, modifier attachments and operator scopes.",
        "mathematical_operations": ("Score(M_k)=alpha*L_k+beta*S_k+gamma*C_k", "P(M_k|E) proportional to P(E|M_k)P(M_k)", "bounded top-K and entropy"),
        "biological_reference": "Parallel activation and competition are plausible; a software candidate list or beam is not evidence for a literal list stored in cortex.",
    },
    {
        "operation_id": "LC-P04-11",
        "name": "Update the partial proposition as each new constituent arrives",
        "objective": "Recompose the growing sentence prefix, update affected entities/events/roles/operators, and preserve local revisions and unresolved slots.",
        "mathematical_operations": ("M_t=U(M_(t-1),L_t,S_t)", "h_t=alpha*h_(t-1)+beta*x_t", "delta_t=x_t-x_hat_t", "incremental graph add/update/remove"),
        "biological_reference": "Semantic composition is incremental and recurrent in comprehension research; recomputing word-prefix snapshots is an engineering approximation rather than a literal neural algorithm.",
    },
    {
        "operation_id": "LC-P04-12",
        "name": "Check whether the basic proposition is sufficiently complete",
        "objective": "Check required predicate roles and operator/modifier status while distinguishing filled, pending, unexpressed, optional and unresolved information.",
        "mathematical_operations": ("R=n_filled/n_required", "predicate and role constraint satisfaction", "confidence-gated completion with explicit missingness labels"),
        "biological_reference": "Comprehension supports ellipsis and incomplete utterances; a rigid checklist is not a model of the brain's full understanding.",
    },
    {
        "operation_id": "LC-P04-13",
        "name": "Consolidate the proposition graph with evidence and uncertainty",
        "objective": "Serialize entities, events, roles, modifiers, operators, alternatives, unresolved slots, confidence and source provenance in a validated graph.",
        "mathematical_operations": ("G=(V,E,Phi)", "endpoint and role consistency checks", "confidence propagation and duplicate-edge consolidation"),
        "biological_reference": "Sentence meaning is supported by interacting temporal, frontal and parietal networks; the graph is an interpretable computational model, not a settled neural code.",
    },
    {
        "operation_id": "LC-P04-14",
        "name": "Pass the literal proposition to contextual and pragmatic integration",
        "objective": "Package basic sentence meaning, alternatives and uncertainty while explicitly deferring discourse, world-knowledge and speaker-intention reasoning.",
        "mathematical_operations": ("interface={graph,readings,uncertainty,provenance}", "information-preserving serialization", "completed_stage=P04; next_stage=contextual_pragmatic"),
        "biological_reference": "A literal compositional representation remains available for wider discourse and social-cognitive interaction; this stage does not infer hidden speaker intent.",
    },
)


_POINT4_TO_SMC: dict[int, tuple[int, ...]] = {
    1: tuple(range(1, 7)),
    2: tuple(range(7, 13)),
    3: (13, 14),
    4: (25, 26),
    5: tuple(range(15, 25)),
    6: tuple(range(16, 19)) + tuple(range(25, 31)),
    7: (8, 9, 12, 49, 50),
    8: tuple(range(31, 43)),
    9: (17, 18, 21, 22, 23, 24, 52),
    10: (6, 23, 24, 42),
    11: tuple(range(7, 31)) + (49, 50, 51),
    12: (18, 29),
    13: tuple(range(43, 55)),
    14: (53, 54),
}

_MODIFIER_RELATIONS = {
    "amod": ("HAS_PROPERTY", "entity_property"),
    "advmod": ("MODIFIES", "event_or_property_modifier"),
    "compound": ("CLASSIFIED_BY", "nominal_classifier"),
    "nmod": ("RELATED_TO", "nominal_relation"),
    "poss": ("POSSESSED_BY", "possessive_relation"),
    "nummod": ("HAS_QUANTITY", "quantity_modifier"),
    "det": ("DETERMINED_BY", "determiner"),
}


def _payload(value: Any) -> dict[str, Any]:
    """Convert an upstream result object or mapping into an ordinary dictionary."""
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "to_dict"):
        result = value.to_dict()
        if isinstance(result, Mapping):
            return dict(result)
    raise TypeError("upstream result must be a mapping or expose to_dict()")


def _bounded_features(value: Any) -> dict[str, float]:
    """Read finite semantic feature strengths and bound them to the shared inventory contract."""
    output: dict[str, float] = {}
    if isinstance(value, Mapping):
        for key, raw in value.items():
            try:
                number = float(raw)
            except (TypeError, ValueError):
                continue
            if str(key).strip() and math.isfinite(number):
                output[str(key)] = max(0.0, min(1.0, number))
    return output


def _derive_concepts(
    lexical_items: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[int, list[str]]]:
    """Build provenance-tracked concept records from Point 2 sense candidates without inventing features."""
    concepts_by_id: dict[str, dict[str, Any]] = {}
    hypotheses: dict[int, list[str]] = {}
    for token_index, item in enumerate(lexical_items):
        token_form = str(item.get("form", item.get("token", ""))).strip()
        candidate_ids: list[str] = []
        lexical_candidates = item.get("lexical_candidates", ()) or ()
        for candidate_index, candidate in enumerate(lexical_candidates):
            if not isinstance(candidate, Mapping):
                continue
            entry_id = str(candidate.get("entry_id", candidate.get("concept_id", f"token-{token_index}-entry-{candidate_index}")))
            label = str(candidate.get("canonical_word_form", candidate.get("orthography", token_form))).strip()
            meanings = candidate.get("meaning_candidates", ()) or ()
            if meanings:
                sense_rows = [
                    row for row in meanings if isinstance(row, Mapping)
                ]
            elif candidate.get("semantic_features"):
                sense_rows = [{
                    "meaning_id": candidate.get("sense_id", "supplied-semantic-features"),
                    "features": candidate.get("semantic_features", {}),
                    "gloss": candidate.get("gloss", ""),
                    "source_reference": candidate.get("source_reference", "unspecified"),
                    "posterior_within_entry": 1.0,
                }]
            else:
                sense_rows = []
            for sense_index, sense in enumerate(sense_rows):
                sense_id = str(sense.get("meaning_id", f"sense-{sense_index}"))
                concept_id = f"{entry_id}::{sense_id}"
                features = _bounded_features(sense.get("features", sense.get("semantic_features", {})))
                if not features:
                    continue
                aliases = list(dict.fromkeys(
                    value for value in (label, token_form, candidate.get("orthography"))
                    if isinstance(value, str) and value.strip()
                ))
                source_reference = str(
                    sense.get("source_reference", candidate.get("source_reference", "unspecified"))
                )
                concepts_by_id.setdefault(concept_id, {
                    "concept_id": concept_id,
                    "label": label or token_form,
                    "aliases": tuple(aliases),
                    "semantic_features": features,
                    "lexical_category": str(candidate.get("lexical_category", "unknown")),
                    "frequency": max(0.01, float(candidate.get("frequency", 1.0) or 1.0)),
                    "language": str(candidate.get("language", "und")),
                    "verified_by": str(candidate.get("verified_by", "upstream-point2-inventory")),
                    "source_reference": source_reference,
                    "upstream_entry_id": entry_id,
                    "upstream_sense_id": sense_id,
                    "upstream_lexical_probability": max(0.0, min(1.0, float(candidate.get("posterior", 0.0)))),
                    "upstream_sense_probability": max(0.0, min(1.0, float(sense.get("posterior_within_entry", 1.0)))),
                    "gloss": str(sense.get("gloss", "")),
                })
                candidate_ids.append(concept_id)
        if not candidate_ids and item.get("semantic_features"):
            features = _bounded_features(item.get("semantic_features"))
            if features:
                concept_id = str(item.get("concept_id", f"token-{token_index}::supplied"))
                concepts_by_id.setdefault(concept_id, {
                    "concept_id": concept_id,
                    "label": token_form,
                    "aliases": (token_form,),
                    "semantic_features": features,
                    "lexical_category": str(item.get("lexical_category", "unknown")),
                    "frequency": max(0.01, float(item.get("frequency", 1.0) or 1.0)),
                    "language": str(item.get("language", "und")),
                    "verified_by": str(item.get("verified_by", "caller-supplied")),
                    "source_reference": str(item.get("source_reference", "unspecified")),
                })
                candidate_ids.append(concept_id)
        if candidate_ids:
            hypotheses[token_index] = list(dict.fromkeys(candidate_ids))
    return list(concepts_by_id.values()), hypotheses


def _derive_predicate_frames(
    lexical_items: Sequence[Mapping[str, Any]],
) -> list[PredicateFrame]:
    """Extract only explicit role/selectional frames already supplied with Point 2 lexical properties."""
    frames: dict[str, PredicateFrame] = {}
    for item in lexical_items:
        for candidate in item.get("lexical_candidates", ()) or ():
            if not isinstance(candidate, Mapping):
                continue
            label = str(candidate.get("canonical_word_form", candidate.get("orthography", ""))).strip()
            properties = candidate.get("lexical_properties", {})
            if not isinstance(properties, Mapping):
                continue
            raw = properties.get("predicate_frame", properties.get("argument_structure"))
            if not isinstance(raw, Mapping):
                continue
            raw_roles = raw.get("roles", raw.get("argument_roles", ()))
            if isinstance(raw_roles, Mapping):
                roles = tuple(str(key) for key in raw_roles)
            elif isinstance(raw_roles, Sequence) and not isinstance(raw_roles, (str, bytes)):
                roles = tuple(str(role) for role in raw_roles)
            else:
                roles = ()
            if not label or not roles:
                continue
            required = raw.get("required_roles", ())
            if not isinstance(required, Sequence) or isinstance(required, (str, bytes)):
                required = ()
            preferences = raw.get("selectional_preferences", {})
            if not isinstance(preferences, Mapping):
                preferences = {}
            meaning_rows = candidate.get("meaning_candidates", ()) or ()
            first_meaning = meaning_rows[0] if meaning_rows and isinstance(meaning_rows[0], Mapping) else {}
            frame = PredicateFrame(
                predicate=label,
                roles=roles,
                required_roles=tuple(str(role) for role in required),
                selectional_preferences={
                    str(role): _bounded_features(features)
                    for role, features in preferences.items()
                    if isinstance(features, Mapping) and str(role) in roles
                },
                predicate_concept_id=(
                    f"{candidate.get('entry_id')}::{first_meaning.get('meaning_id')}"
                    if candidate.get("entry_id") and first_meaning.get("meaning_id") else None
                ),
                source_reference=str(raw.get("source_reference", candidate.get("source_reference", "upstream-point2-lexical-frame"))),
            )
            frames[label.casefold()] = frame
    return list(frames.values())


def _normalize_frames(values: Sequence[Any]) -> tuple[PredicateFrame, ...]:
    """Validate supplied PredicateFrame objects or convert explicitly structured frame mappings."""
    output = []
    for value in values:
        if isinstance(value, PredicateFrame):
            output.append(value)
            continue
        if not isinstance(value, Mapping):
            raise TypeError("predicate_frames must contain PredicateFrame objects or mappings")
        roles = value.get("roles", ())
        required = value.get("required_roles", ())
        output.append(PredicateFrame(
            predicate=str(value.get("predicate", "")),
            roles=tuple(str(role) for role in roles),
            required_roles=tuple(str(role) for role in required),
            selectional_preferences={
                str(role): _bounded_features(features)
                for role, features in value.get("selectional_preferences", {}).items()
                if isinstance(features, Mapping)
            },
            predicate_concept_id=value.get("predicate_concept_id"),
            source_reference=str(value.get("source_reference", "caller-supplied")),
        ))
    return tuple(output)


def _softmax(scores: Sequence[float]) -> list[float]:
    """Convert finite scores to a numerically stable relative distribution."""
    if not scores:
        return []
    valid = [float(value) if math.isfinite(float(value)) else -1e9 for value in scores]
    peak = max(valid)
    weights = [math.exp(max(-100.0, item - peak)) for item in valid]
    total = sum(weights)
    return [value / total for value in weights] if total > 0.0 else [1.0 / len(scores)] * len(scores)


@dataclass(frozen=True, slots=True)
class Point4Request:
    """Bundle Point 2 lexical-semantic evidence with a Point 3 syntactic result.

    The concept inventory may be provided directly. If it is omitted, the request factory
    converts only explicit Point 2 meaning-feature records into concept candidates and retains
    their entry/sense identifiers and confidence as provenance.
    """

    sequence_id: str
    tokens: Sequence[str]
    syntax_result: Any
    lexical_items: Sequence[Mapping[str, Any]] = ()
    concept_inventory: Sequence[Any] = ()
    concept_hypotheses: Mapping[int, Sequence[str]] = field(default_factory=dict)
    predicate_frames: Sequence[Any] = ()
    context_features: Mapping[str, float] = field(default_factory=dict)
    language: str = "en"
    source_reference: str | None = None
    utterance_complete: bool | None = None

    def __post_init__(self) -> None:
        """Validate aligned inputs and preserve only supplied lexical, semantic and syntactic evidence."""
        if not self.sequence_id.strip():
            raise ValueError("sequence_id must be non-empty")
        tokens = tuple(str(token) for token in self.tokens)
        if not tokens or any(not token.strip() for token in tokens):
            raise ValueError("tokens must be a non-empty sequence of non-empty strings")
        if not self.language.strip():
            raise ValueError("language must be non-empty")
        syntax = _payload(self.syntax_result)
        syntax_tokens = syntax.get("tokens")
        if syntax_tokens and list(syntax_tokens) != list(tokens):
            raise ValueError("syntax_result token sequence must exactly match Point 4 tokens")
        lexical_items = tuple(dict(item) for item in self.lexical_items)
        if lexical_items and len(lexical_items) != len(tokens):
            raise ValueError("lexical_items must contain one item per token")
        concepts = tuple(self.concept_inventory)
        hypotheses = {int(index): tuple(str(item) for item in values) for index, values in self.concept_hypotheses.items()}
        if not concepts and lexical_items:
            derived, derived_hypotheses = _derive_concepts(lexical_items)
            concepts = tuple(derived)
            if not hypotheses:
                hypotheses = {index: tuple(values) for index, values in derived_hypotheses.items()}
        if any(index < 0 or index >= len(tokens) for index in hypotheses):
            raise ValueError("concept_hypotheses indexes must reference existing tokens")
        if any(not str(key).strip() for key in self.context_features):
            raise ValueError("context feature labels must be non-empty")
        for key, raw in self.context_features.items():
            value = float(raw)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("context feature values must lie in [0, 1]")
        frames = _normalize_frames(self.predicate_frames) if self.predicate_frames else (
            tuple(_derive_predicate_frames(lexical_items)) if lexical_items else ()
        )
        if self.utterance_complete is None:
            utterance_complete = tokens[-1] in {".", "!", "?", "。", "؟"}
        else:
            utterance_complete = bool(self.utterance_complete)
        object.__setattr__(self, "tokens", tokens)
        object.__setattr__(self, "syntax_result", syntax)
        object.__setattr__(self, "lexical_items", lexical_items)
        object.__setattr__(self, "concept_inventory", concepts)
        object.__setattr__(self, "concept_hypotheses", hypotheses)
        object.__setattr__(self, "predicate_frames", frames)
        object.__setattr__(self, "utterance_complete", utterance_complete)

    @classmethod
    def from_point3_result(
        cls,
        sequence_id: str,
        point3_result: Any,
        *,
        point2_results: Sequence[Any] | None = None,
        concept_inventory: Sequence[Any] = (),
        predicate_frames: Sequence[Any] = (),
        context_features: Mapping[str, float] | None = None,
        language: str | None = None,
        source_reference: str | None = None,
        utterance_complete: bool | None = None,
    ) -> "Point4Request":
        """Create a Point 4 request from a Point3Result and its embedded or separately supplied Point 2 records."""
        payload = _payload(point3_result)
        tokens = tuple(str(token) for token in payload.get("tokens", ()))
        if not tokens:
            raise ValueError("point3_result must contain a non-empty token sequence")
        lexical_items = list(payload.get("lexical_item_records", ()) or ())
        if point2_results is not None:
            if len(point2_results) != len(tokens):
                raise ValueError("point2_results must contain exactly one result for each Point 3 token")
            lexical_items = []
            for index, raw_result in enumerate(point2_results):
                result = _payload(raw_result)
                candidates = list(result.get("lexical_candidates", ()) or ())
                lexical_items.append({
                    "form": tokens[index],
                    "lexical_candidates": candidates,
                    "confidence": float(result.get("confidence", 0.0)),
                })
        syntax = {
            "tokens": list(tokens),
            "selected_pos": list(payload.get("selected_pos", ()) or ()),
            "dependency_arcs": list(payload.get("dependency_relations", payload.get("dependency_arcs", ())) or ()),
            "constituent_groups": list(payload.get("constituent_groups", payload.get("phrases", ())) or ()),
            "constituency_tree": payload.get("constituency_tree", {}),
            "syntax_hierarchy": payload.get("syntax_hierarchy", payload.get("hierarchy", {})),
            "incremental_updates": list(payload.get("incremental_updates", ()) or ()),
            "source_schema": payload.get("schema", "ore.syntax.external"),
        }
        return cls(
            sequence_id=sequence_id,
            tokens=tokens,
            syntax_result=syntax,
            lexical_items=lexical_items,
            concept_inventory=concept_inventory,
            concept_hypotheses={},
            predicate_frames=predicate_frames,
            context_features=context_features or {},
            language=language or str(payload.get("language", "en")),
            source_reference=source_reference or f"adapter:point3:{payload.get('schema', 'unknown')}",
            utterance_complete=utterance_complete,
        )


@dataclass(slots=True)
class Point4Result:
    """Expose compositional meanings, proposition graphs, modifiers, alternatives and separate biology."""

    sequence_id: str
    tokens: Sequence[str]
    word_concepts: Sequence[Mapping[str, Any]]
    semantic_workspace: Mapping[str, Any]
    semantic_heads: Sequence[Mapping[str, Any]]
    entity_variables: Sequence[Mapping[str, Any]]
    event_variables: Sequence[Mapping[str, Any]]
    predicate_arguments: Sequence[Mapping[str, Any]]
    thematic_roles: Sequence[Mapping[str, Any]]
    propositions: Sequence[Mapping[str, Any]]
    modifier_relations: Sequence[Mapping[str, Any]]
    operators: Mapping[str, Any]
    quantifier_scopes: Sequence[Mapping[str, Any]]
    causal_relations: Sequence[Mapping[str, Any]]
    semantic_compatibility: Sequence[Mapping[str, Any]]
    competing_readings: Sequence[Mapping[str, Any]]
    incremental_updates: Sequence[Mapping[str, Any]]
    proposition_completeness: Sequence[Mapping[str, Any]]
    proposition_graph: Mapping[str, Any]
    confidence: float
    functional_trace: Sequence[Mapping[str, Any]]
    biological_trace: Sequence[Mapping[str, Any]]
    biological_summary: Mapping[str, Any]
    handoff: Mapping[str, Any]
    diagnostics: Mapping[str, Any]
    annotation_report: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the proposition graph while keeping literal composition separate from pragmatics."""
        return to_jsonable({
            "schema": "ore.language_comprehension.point04.v1",
            "sequence_id": self.sequence_id,
            "tokens": self.tokens,
            "word_concepts": self.word_concepts,
            "semantic_workspace": self.semantic_workspace,
            "semantic_heads": self.semantic_heads,
            "entity_variables": self.entity_variables,
            "event_variables": self.event_variables,
            "predicate_arguments": self.predicate_arguments,
            "thematic_roles": self.thematic_roles,
            "propositions": self.propositions,
            "modifier_relations": self.modifier_relations,
            "operators": self.operators,
            "quantifier_scopes": self.quantifier_scopes,
            "causal_relations": self.causal_relations,
            "semantic_compatibility": self.semantic_compatibility,
            "competing_readings": self.competing_readings,
            "incremental_updates": self.incremental_updates,
            "proposition_completeness": self.proposition_completeness,
            "proposition_graph": self.proposition_graph,
            "confidence": self.confidence,
            "functional_trace": self.functional_trace,
            "biological_trace": self.biological_trace,
            "biological_summary": self.biological_summary,
            "handoff": self.handoff,
            "diagnostics": self.diagnostics,
            "annotation_report": self.annotation_report,
            "scope": {
                "sentence_level_proposition_built": True,
                "predicate_argument_relations_built": True,
                "thematic_roles_composed": True,
                "literal_semantic_compatibility_evaluated": True,
                "discourse_contextual_inference_performed": False,
                "speaker_intention_inferred": False,
                "sarcasm_or_irony_interpreted": False,
                "unstated_causal_story_invented": False,
                "biological_trace_is_measured_human_data": False,
            },
        })


class Point4SemanticComposer:
    """Run the 14-step source composition algorithm through ORE's nine-stage/54-operation engine.

    Point 2 candidate meanings are converted into concept-inventory records when available.
    Point 3 dependency relations constrain argument binding. The existing composition engine
    performs predicate/role/proposition/operator processing; this adapter adds modifier links,
    source-level incremental snapshots, semantic conflict summaries and the 14-step interface.
    """

    def __init__(
        self,
        config: SemanticCompositionConfig | None = None,
        *,
        max_candidate_readings: int = 6,
        semantic_conflict_threshold: float = 0.18,
        register_annotations: bool = True,
        repository_root: str | None = None,
        database_path: str | None = None,
    ) -> None:
        """Configure the semantic-composition engine, uncertainty bounds and shared SQLite catalogue."""
        if max_candidate_readings < 1:
            raise ValueError("max_candidate_readings must be positive")
        if not 0.0 <= semantic_conflict_threshold <= 1.0:
            raise ValueError("semantic_conflict_threshold must lie in [0, 1]")
        self.config = config or SemanticCompositionConfig()
        self.max_candidate_readings = int(max_candidate_readings)
        self.semantic_conflict_threshold = float(semantic_conflict_threshold)
        if register_annotations:
            from .annotations import register_point04_annotations

            self.annotation_report = register_point04_annotations(repository_root, database_path)
        else:
            self.annotation_report = {"status": "annotation_registration_disabled"}

    def _syntax_for_prefix(
        self,
        request: Point4Request,
        prefix_end: int,
    ) -> dict[str, Any]:
        """Reuse the matching Point 3 prefix snapshot or conservatively trim the final syntax evidence."""
        syntax = dict(request.syntax_result)
        snapshots = syntax.get("incremental_updates", ()) or ()
        snapshot = next(
            (
                item for item in snapshots
                if int(item.get("prefix_end_exclusive", -1)) == prefix_end
            ),
            None,
        )
        if snapshot is not None:
            return {
                "tokens": list(request.tokens[:prefix_end]),
                "selected_pos": list(snapshot.get("selected_pos", ())),
                "dependency_arcs": list(snapshot.get("dependency_relations", ())),
                "constituent_groups": list(snapshot.get("constituent_groups", ())),
                "constituency_alternatives": list(snapshot.get("constituency_alternatives", ())),
                "incremental_updates": list(snapshots[:prefix_end]),
            }
        arcs = [
            dict(item) for item in syntax.get("dependency_arcs", ())
            if int(item.get("dependent_index", -1)) < prefix_end
            and (
                item.get("head_index") is None
                or int(item.get("head_index", -1)) < prefix_end
            )
        ]
        return {
            "tokens": list(request.tokens[:prefix_end]),
            "selected_pos": [
                dict(item) for item in syntax.get("selected_pos", ())
                if int(item.get("token_index", -1)) < prefix_end
            ],
            "dependency_arcs": arcs,
            "constituent_groups": [
                dict(item) for item in syntax.get("constituent_groups", ())
                if int(item.get("end", 0)) <= prefix_end
            ],
            "constituency_tree": syntax.get("constituency_tree", {}),
        }

    def _run_engine(
        self,
        request: Point4Request,
        prefix_end: int,
        *,
        final_prefix: bool,
    ) -> Any:
        """Run the established 54-operation semantic engine on one sentence prefix."""
        config = self.config if final_prefix else replace(self.config, simulate_biology=False)
        processor = SemanticCompositionProcessor(
            request.concept_inventory,
            request.predicate_frames,
            config,
        )
        return processor.process(SemanticCompositionRequest(
            sequence_id=f"{request.sequence_id}-prefix-{prefix_end}",
            tokens=tuple(request.tokens[:prefix_end]),
            language=request.language,
            concept_hypotheses={
                index: values
                for index, values in request.concept_hypotheses.items()
                if index < prefix_end
            },
            syntax_result=self._syntax_for_prefix(request, prefix_end),
            context_features=request.context_features,
            source_reference=request.source_reference,
        ))

    @staticmethod
    def _semantic_heads(
        word_concepts: Sequence[Mapping[str, Any]],
        selected_pos: Sequence[Mapping[str, Any]],
        propositions: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """Identify event/state predicates and retain the syntactic and lexical evidence for each head."""
        pos_by_index = {
            int(item.get("token_index", -1)): str(item.get("tag", "X"))
            for item in selected_pos
        }
        output = []
        predicate_indexes = {int(item.get("predicate_index", -1)) for item in propositions}
        for index, word in enumerate(word_concepts):
            category = str(word.get("selected_lexical_category", "unknown")).casefold()
            pos = pos_by_index.get(index, "X")
            if index not in predicate_indexes and category not in {"verb", "adjective", "adverb"} and pos not in {"VERB", "AUX", "ADJ"}:
                continue
            kind = (
                "event_or_state_predicate" if pos in {"VERB", "AUX"} or category == "verb"
                else "property_head" if pos == "ADJ" or category == "adjective"
                else "relational_or_modifier_head"
            )
            output.append({
                "token_index": index,
                "token": word.get("token"),
                "concept_id": word.get("selected_concept_id"),
                "semantic_label": word.get("selected_label"),
                "lexical_category": category,
                "syntactic_category": pos,
                "semantic_head_kind": kind,
                "predicate_proposition_present": index in predicate_indexes,
                "confidence": float(word.get("confidence", 0.0)),
            })
        return output

    @staticmethod
    def _entity_variables(
        word_concepts: Sequence[Mapping[str, Any]],
        selected_pos: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """Create distinct sentence-local entity placeholders from nominal and pronominal tokens."""
        nominal = {"NOUN", "PROPN", "PRON", "NUM"}
        pos_by_index = {
            int(item.get("token_index", -1)): str(item.get("tag", "X"))
            for item in selected_pos
        }
        entities = []
        for index, word in enumerate(word_concepts):
            pos = pos_by_index.get(index, "X")
            category = str(word.get("selected_lexical_category", "")).casefold()
            if pos not in nominal and category not in {"noun", "proper_noun", "pronoun"}:
                continue
            entities.append({
                "variable": f"x{index + 1}",
                "entity_id": f"entity:{index}",
                "token_index": index,
                "surface": word.get("token"),
                "concept_id": word.get("selected_concept_id"),
                "concept_label": word.get("selected_label"),
                "features": dict(word.get("selected_features", {})),
                "candidate_senses": [
                    {
                        "concept_id": item.get("concept_id"),
                        "label": item.get("label"),
                        "probability": item.get("probability", 0.0),
                    }
                    for item in word.get("candidates", ())
                ],
                "confidence": float(word.get("confidence", 0.0)),
                "identity_status": "sentence_local_placeholder; cross-mention identity is not inferred here",
            })
        return entities

    @staticmethod
    def _event_variables(
        propositions: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """Represent predicate propositions as event/state variables with their current completeness."""
        return [
            {
                "variable": f"e{index + 1}",
                "event_id": item.get("proposition_id"),
                "predicate_index": item.get("predicate_index"),
                "predicate": item.get("predicate"),
                "predicate_concept_id": item.get("predicate_concept_id"),
                "polarity": item.get("polarity", "positive"),
                "tense": item.get("tense"),
                "aspect": item.get("aspect"),
                "voice": item.get("voice"),
                "confidence": float(item.get("confidence", 0.0)),
            }
            for index, item in enumerate(propositions)
        ]

    @staticmethod
    def _modifier_relations(
        tokens: Sequence[str],
        syntax: Mapping[str, Any],
        word_concepts: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """Compose syntactically licensed modifiers as typed relations without flattening concept features."""
        selected_pos = {
            int(item.get("token_index", -1)): str(item.get("tag", "X"))
            for item in syntax.get("selected_pos", ())
        }
        output = []
        for arc in syntax.get("dependency_arcs", ()):
            relation = str(arc.get("relation", "")).lower()
            if relation not in _MODIFIER_RELATIONS:
                continue
            dependent = int(arc.get("dependent_index", -1))
            head = arc.get("head_index")
            if head is None:
                continue
            head = int(head)
            if not (0 <= dependent < len(tokens) and 0 <= head < len(tokens)):
                continue
            relation_label, kind = _MODIFIER_RELATIONS[relation]
            modifier = word_concepts[dependent]
            target = word_concepts[head]
            modifier_features = dict(modifier.get("selected_features", {}))
            target_features = dict(target.get("selected_features", {}))
            union = {
                key: max(float(target_features.get(key, 0.0)), float(modifier_features.get(key, 0.0)))
                for key in set(target_features) | set(modifier_features)
            }
            shared = {
                key: min(float(target_features[key]), float(modifier_features[key]))
                for key in set(target_features) & set(modifier_features)
            }
            output.append({
                "modifier_index": dependent,
                "modifier": tokens[dependent],
                "modifier_concept_id": modifier.get("selected_concept_id"),
                "modifier_candidate_concepts": [item.get("concept_id") for item in modifier.get("candidates", ())],
                "target_index": head,
                "target": tokens[head],
                "target_concept_id": target.get("selected_concept_id"),
                "dependency_relation": relation,
                "semantic_relation": relation_label,
                "modifier_kind": kind,
                "target_syntactic_category": selected_pos.get(head, "X"),
                "modifier_features": modifier_features,
                "target_features": target_features,
                "combined_feature_union": union,
                "shared_feature_intersection": shared,
                "confidence": min(
                    float(modifier.get("confidence", 0.0)),
                    float(target.get("confidence", 0.0)),
                    float(arc.get("confidence", 1.0)),
                ),
                "provenance": "Point 3 dependency relation + Point 2 lexical-semantic features",
            })
        return output

    def _semantic_compatibility(
        self,
        argument_frames: Sequence[Mapping[str, Any]],
        predicate_frames: Sequence[PredicateFrame],
    ) -> list[dict[str, Any]]:
        """Report predicate-role selectional compatibility separately from broader pragmatic plausibility."""
        frame_by_predicate = {frame.predicate.casefold(): frame for frame in predicate_frames}
        rows = []
        for frame in argument_frames:
            predicate = str(frame.get("predicate", ""))
            explicit = predicate.casefold() in frame_by_predicate
            explicit_frame = frame_by_predicate.get(predicate.casefold())
            for compatibility in frame.get("compatibility", ()):
                semantic_fit = float(compatibility.get("semantic_fit", 0.5))
                rows.append({
                    "predicate_index": frame.get("predicate_index"),
                    "predicate": predicate,
                    "argument_index": compatibility.get("token_index"),
                    "role_candidate": compatibility.get("role"),
                    "syntactic_fit": float(compatibility.get("syntax_fit", 0.0)),
                    "semantic_feature_fit": semantic_fit,
                    "predicate_frame_supplied": explicit,
                    "predicate_frame_source": explicit_frame.source_reference if explicit_frame else "fallback-or-unspecified",
                    "status": (
                        "possible_selectional_tension"
                        if explicit and semantic_fit < 0.18
                        else "compatible_or_unresolved"
                    ),
                    "scope_note": "literal compositional compatibility only; not a pragmatic impossibility judgment",
                })
        return rows

    def _candidate_readings(
        self,
        propositions: Sequence[Mapping[str, Any]],
        thematic_roles: Sequence[Mapping[str, Any]],
        word_concepts: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """Build bounded separate proposition readings from top lexical-sense and role alternatives."""
        output: list[dict[str, Any]] = []
        words_by_index = {int(item["token_index"]): item for item in word_concepts}
        roles_by_key = {
            (int(item.get("predicate_index", -1)), int(item.get("argument_index", -1))): item
            for item in thematic_roles
        }
        for proposition in propositions:
            predicate_index = int(proposition.get("predicate_index", -1))
            predicate_word = words_by_index.get(predicate_index, {})
            predicate_senses = list(predicate_word.get("candidates", ()))[:2] or [{
                "concept_id": proposition.get("predicate_concept_id"),
                "label": proposition.get("predicate"),
                "probability": 1.0,
            }]
            argument_options: list[list[dict[str, Any]]] = []
            for argument in proposition.get("arguments", ()):
                argument_index = int(argument.get("token_index", -1))
                word = words_by_index.get(argument_index, {})
                sense_options = list(word.get("candidates", ()))[:2] or [{
                    "concept_id": None, "label": argument.get("surface"), "probability": 1.0,
                }]
                role_record = roles_by_key.get((predicate_index, argument_index), {})
                role_options = list(role_record.get("role_candidates", ()))[:2] or [{
                    "role": argument.get("role"), "probability": 1.0,
                }]
                argument_options.append([
                    {
                        "sense": sense,
                        "role": role,
                        "score": max(0.0, float(sense.get("probability", 1.0)))
                        * max(0.0, float(role.get("probability", 1.0))),
                    }
                    for sense, role in product(sense_options, role_options)
                ])
            combinations = (
                list(islice(product(*argument_options), self.max_candidate_readings))
                if argument_options else [()]
            )
            for predicate_sense in predicate_senses:
                for combination in combinations:
                    arguments = []
                    score = max(1e-12, float(predicate_sense.get("probability", 1.0)))
                    for original, choice in zip(proposition.get("arguments", ()), combination):
                        sense = choice["sense"]
                        role = choice["role"]
                        score *= max(1e-12, float(choice["score"]))
                        arguments.append({
                            **dict(original),
                            "concept_id": sense.get("concept_id"),
                            "concept_label": sense.get("label"),
                            "role": role.get("role"),
                            "role_probability": float(role.get("probability", 0.0)),
                        })
                    output.append({
                        "reading_id": f"{proposition.get('proposition_id')}:reading-{len(output) + 1}",
                        "proposition_id": proposition.get("proposition_id"),
                        "predicate": proposition.get("predicate"),
                        "predicate_concept_id": predicate_sense.get("concept_id"),
                        "predicate_sense_label": predicate_sense.get("label"),
                        "arguments": arguments,
                        "polarity": proposition.get("polarity", "positive"),
                        "modal_operators": list(proposition.get("modal_operators", ())),
                        "unnormalized_evidence_score": score,
                    })
            if len(output) >= self.max_candidate_readings:
                break
        output = output[:self.max_candidate_readings]
        probabilities = _softmax([
            math.log(max(1e-12, float(item["unnormalized_evidence_score"])))
            for item in output
        ])
        for item, probability in zip(output, probabilities):
            item["posterior_like_support"] = probability
            item["unnormalized_evidence_score"] = float(item["unnormalized_evidence_score"])
        return output

    def _proposition_completeness(
        self,
        argument_frames: Sequence[Mapping[str, Any]],
        propositions: Sequence[Mapping[str, Any]],
        thematic_roles: Sequence[Mapping[str, Any]],
        request: Point4Request,
    ) -> list[dict[str, Any]]:
        """Distinguish filled required roles from pending or unexpressed roles using explicit frames."""
        frame_by_predicate = {frame.predicate.casefold(): frame for frame in request.predicate_frames}
        proposition_by_index = {
            int(item.get("predicate_index", -1)): item for item in propositions
        }
        output = []
        for frame in argument_frames:
            predicate = str(frame.get("predicate", ""))
            explicit = frame_by_predicate.get(predicate.casefold())
            required = tuple(explicit.required_roles) if explicit else ()
            role_status = dict(frame.get("role_status", {}))
            selected_roles = {
                str(item.get("selected_role"))
                for item in thematic_roles
                if int(item.get("predicate_index", -1)) == int(frame.get("predicate_index", -2))
                and item.get("selected_role") is not None
            }
            filled = selected_roles
            filled_required = [role for role in required if role in filled]
            missing = [role for role in required if role not in filled]
            if not explicit:
                status = "not_assessable_no_explicit_required_roles"
                ratio = None
                missing_status = "not_inferred"
            else:
                ratio = len(filled_required) / max(1, len(required))
                if not missing:
                    status = "complete"
                    missing_status = "none"
                elif request.utterance_complete:
                    status = "incomplete_or_unexpressed_required_roles"
                    missing_status = "not_expressed_or_unspecified"
                else:
                    status = "partial_sentence_waiting_for_more_input"
                    missing_status = "unresolved_pending_input"
            proposition = proposition_by_index.get(int(frame.get("predicate_index", -1)), {})
            output.append({
                "predicate_index": frame.get("predicate_index"),
                "predicate": predicate,
                "predicate_frame_supplied": explicit is not None,
                "required_roles": list(required),
                "role_status": role_status,
                "filled_required_roles": filled_required,
                "missing_required_roles": missing,
                "missing_role_interpretation": missing_status,
                "completion_ratio": ratio,
                "status": status,
                "proposition_confidence": float(proposition.get("confidence", 0.0)),
                "confidence_is_not_truth_probability": True,
            })
        return output

    @staticmethod
    def _prefix_state(
        prefix_end: int,
        tokens: Sequence[str],
        semantic_result: Any,
        modifier_relations: Sequence[Mapping[str, Any]],
        previous: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """Summarize one recomputed prefix and compare proposition/argument state with its predecessor."""
        words = list(semantic_result.word_concepts)
        propositions = list(semantic_result.propositions)
        current = {
            "entity_indexes_with_candidates": [
                int(row["token_index"]) for row in words if row.get("candidates")
            ],
            "predicate_ids": [str(row.get("proposition_id")) for row in propositions],
            "open_role_slots": [
                {
                    "predicate_index": row.get("predicate_index"),
                    "missing_required_roles": list(row.get("missing_required_roles", ())),
                    "role_status": dict(next(
                        (
                            item.get("role_status", {})
                            for item in semantic_result.predicate_arguments
                            if int(item.get("predicate_index", -1)) == int(row.get("predicate_index", -1))
                        ),
                        {},
                    )),
                }
                for row in propositions
                if row.get("missing_required_roles")
            ],
            "modifier_relations": [dict(item) for item in modifier_relations],
            "propositions": [
                {
                    "proposition_id": row.get("proposition_id"),
                    "predicate": row.get("predicate"),
                    "arguments": [dict(arg) for arg in row.get("arguments", ())],
                    "polarity": row.get("polarity"),
                    "confidence": float(row.get("confidence", 0.0)),
                }
                for row in propositions
            ],
        }
        previous_predicates = set(previous.get("predicate_ids", ())) if previous else set()
        previous_props = {
            item["proposition_id"]: item
            for item in (previous.get("propositions", ()) if previous else ())
        }
        added = [
            row for row in current["propositions"]
            if row["proposition_id"] not in previous_predicates
        ]
        revised = [
            row for row in current["propositions"]
            if row["proposition_id"] in previous_props
            and row != previous_props[row["proposition_id"]]
        ]
        return {
            "prefix_end_exclusive": prefix_end,
            "new_token_index": prefix_end - 1,
            "new_token": tokens[prefix_end - 1],
            "entity_candidate_count": len(current["entity_indexes_with_candidates"]),
            "predicate_count": len(current["predicate_ids"]),
            "new_propositions": added,
            "revised_propositions": revised,
            "open_role_slots": current["open_role_slots"],
            "modifier_relation_count": len(current["modifier_relations"]),
            "partial_semantic_state": current,
            "is_final_sentence_state": prefix_end == len(tokens),
            "revision_policy": "recompute affected prefix through the established engine and retain explicit before/after records",
        }

    def _biological_trace(
        self,
        operation_trace: Sequence[Mapping[str, Any]],
        biological_summary: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        """Aggregate the 54 underlying semantic-composition operation states into 14 source-level steps."""
        states = {
            str(item.get("operation_id")): item
            for item in biological_summary.get("operation_states", ())
        }
        output = []
        for index, specification in enumerate(POINT4_STEP_SPECS, start=1):
            selected = [
                states[f"SMC-{number:03d}"]
                for number in _POINT4_TO_SMC[index]
                if f"SMC-{number:03d}" in states
            ]
            rates = [
                float(item.get("mean_firing_rate_hz", 0.0))
                for item in selected if isinstance(item, Mapping)
            ]
            output.append({
                "operation_id": specification["operation_id"],
                "status": (
                    "simulated_reference_not_measured_human_data"
                    if selected and biological_summary.get("model") != "disabled"
                    else "biological_simulation_disabled"
                ),
                "underlying_SMC_operation_ids": [f"SMC-{number:03d}" for number in _POINT4_TO_SMC[index]],
                "operation_count": len(selected),
                "population_names": sorted({
                    str(item.get("population")) for item in selected if item.get("population")
                }),
                "mean_reference_firing_rate_hz": (
                    sum(rates) / len(rates) if rates else 0.0
                ),
                "simulated_time_ms": float(biological_summary.get("simulated_time_ms", 0.0)),
                "biological_scope": "existing nine-population semantic-composition LIF reference; not anatomically fitted or measured",
            })
        return output

    def process(self, request: Point4Request) -> Point4Result:
        """Execute the 14 source steps and return an uncertainty-aware sentence proposition graph."""
        if not isinstance(request, Point4Request):
            raise TypeError("request must be a Point4Request")
        prefix_results = []
        previous_state = None
        semantic_result = None
        for prefix_end in range(1, len(request.tokens) + 1):
            semantic_result = self._run_engine(
                request,
                prefix_end,
                final_prefix=prefix_end == len(request.tokens),
            )
            prefix_syntax = self._syntax_for_prefix(request, prefix_end)
            prefix_modifiers = self._modifier_relations(
                request.tokens[:prefix_end],
                prefix_syntax,
                semantic_result.word_concepts,
            )
            state = self._prefix_state(
                prefix_end,
                request.tokens,
                semantic_result,
                prefix_modifiers,
                previous_state,
            )
            prefix_results.append(state)
            previous_state = state["partial_semantic_state"]
        assert semantic_result is not None

        syntax = dict(request.syntax_result)
        word_concepts = list(semantic_result.word_concepts)
        propositions = [dict(item) for item in semantic_result.propositions]
        predicate_arguments = [dict(item) for item in semantic_result.predicate_arguments]
        thematic_roles = [dict(item) for item in semantic_result.thematic_roles]
        modifier_relations = self._modifier_relations(
            request.tokens, syntax, word_concepts
        )
        semantic_heads = self._semantic_heads(
            word_concepts, syntax.get("selected_pos", ()), propositions
        )
        entity_variables = self._entity_variables(
            word_concepts, syntax.get("selected_pos", ())
        )
        event_variables = self._event_variables(propositions)
        compatibility = self._semantic_compatibility(
            predicate_arguments, request.predicate_frames
        )
        readings = self._candidate_readings(
            propositions, thematic_roles, word_concepts
        )
        completeness = self._proposition_completeness(
            predicate_arguments, propositions, thematic_roles, request
        )

        # Add wrapper-level modifier/property edges to the existing directed graph.
        graph = {
            key: value
            for key, value in dict(semantic_result.sentence_representation).items()
        }
        nodes = [dict(item) for item in graph.get("nodes", ())]
        edges = [dict(item) for item in graph.get("edges", ())]
        existing_nodes = {str(item.get("node_id")) for item in nodes}
        for relation in modifier_relations:
            modifier_node = f"token:{relation['modifier_index']}"
            target_node = f"token:{relation['target_index']}"
            if modifier_node in existing_nodes and target_node in existing_nodes:
                edges.append({
                    "edge_type": "modifier_composition",
                    "source": target_node,
                    "target": modifier_node,
                    "semantic_relation": relation["semantic_relation"],
                    "dependency_relation": relation["dependency_relation"],
                    "confidence": relation["confidence"],
                })
        for row in completeness:
            for role in row["missing_required_roles"]:
                edges.append({
                    "edge_type": "unresolved_argument_slot",
                    "source": f"prop:{row['predicate_index']}",
                    "target": f"slot:{row['predicate_index']}:{role}",
                    "role": role,
                    "status": row["missing_role_interpretation"],
                })
                nodes.append({
                    "node_id": f"slot:{row['predicate_index']}:{role}",
                    "node_type": "unresolved_semantic_role",
                    "predicate_index": row["predicate_index"],
                    "role": role,
                    "status": row["missing_role_interpretation"],
                })
        graph["nodes"] = nodes
        graph["edges"] = edges
        graph["modifier_relations"] = modifier_relations
        graph["competing_readings"] = readings
        graph["proposition_completeness"] = completeness
        node_ids = {str(item.get("node_id")) for item in nodes}
        graph_errors = [
            {
                "edge_type": item.get("edge_type"),
                "source": item.get("source"),
                "target": item.get("target"),
            }
            for item in edges
            if str(item.get("source")) not in node_ids or str(item.get("target")) not in node_ids
        ]
        graph["validation"] = {
            **dict(graph.get("validation", {})),
            "valid_references": not graph_errors,
            "graph_errors": graph_errors,
            "node_count": len(nodes),
            "edge_count": len(edges),
        }
        biological = dict(semantic_result.biological)
        biological_trace = self._biological_trace(
            semantic_result.operation_trace,
            biological,
        )
        operators = {
            "negation_and_modality": semantic_result.negation_modality,
            "quantifier_scopes": semantic_result.quantifier_scopes,
            "causal_relations": semantic_result.causal_relations,
        }
        unresolved_role_count = sum(
            len(row["missing_required_roles"]) for row in completeness
        )
        concept_confidences = [
            float(row.get("confidence", 0.0)) for row in word_concepts
            if row.get("candidates")
        ]
        proposition_confidences = [
            float(row.get("confidence", 0.0)) for row in propositions
        ]
        confidence_values = concept_confidences + proposition_confidences
        confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0.0

        functional_outputs = (
            {
                "tokens": list(request.tokens),
                "lexical_items_received": len(request.lexical_items),
                "concept_inventory_size": len(request.concept_inventory),
                "concept_hypotheses": {
                    str(index): list(values) for index, values in request.concept_hypotheses.items()
                },
                "syntactic_dependency_count": len(syntax.get("dependency_arcs", ())),
                "token_alignment_preserved": True,
            },
            {
                "word_concepts": word_concepts,
                "semantic_workspace": semantic_result.combined_meaning,
                "feature_dimensions": sorted({
                    feature for row in word_concepts
                    for feature in row.get("selected_features", {})
                }),
            },
            {"semantic_heads": semantic_heads},
            {
                "entity_variables": entity_variables,
                "event_variables": event_variables,
                "identity_resolution_scope": "sentence-local placeholders; discourse coreference is deferred",
            },
            {
                "predicate_arguments": predicate_arguments,
                "thematic_roles": thematic_roles,
                "mapping_caveat": "grammatical subject/object are not universally Agent/Patient; explicit frames improve role mapping",
            },
            {"propositions": propositions},
            {
                "modifier_relations": modifier_relations,
                "modifier_composition_edges": [
                    item for item in edges if item.get("edge_type") == "modifier_composition"
                ],
            },
            operators,
            {
                "semantic_compatibility": compatibility,
                "conflicts_are_not_pragmatic_impossibility_judgments": True,
            },
            {
                "competing_readings": readings,
                "alternatives_are_separate_not_blended": True,
            },
            {
                "incremental_updates": prefix_results,
                "prefix_recomputation_count": len(prefix_results),
                "revisions_are_explicit": True,
            },
            {
                "proposition_completeness": completeness,
                "unresolved_required_role_count": unresolved_role_count,
                "missing_roles_distinguish_pending_from_unexpressed": True,
            },
            {
                "proposition_graph": graph,
                "graph_validation": graph["validation"],
            },
            {
                "handoff": {
                    "target": "Point 5 contextual and pragmatic integration",
                    "literal_proposition_graph": graph,
                    "competing_readings": readings,
                    "uncertainty": graph.get("uncertainty", {}),
                    "unresolved_slots": completeness,
                    "deferred": [
                        "discourse-history integration",
                        "physical-situation and world-knowledge inference",
                        "speaker-intention inference",
                        "sarcasm, irony and figurative interpretation",
                        "unstated causal-story completion",
                    ],
                },
            },
        )
        functional_trace = []
        for specification, output in zip(POINT4_STEP_SPECS, functional_outputs, strict=True):
            functional_trace.append({
                **specification,
                "status": "completed",
                "functional_result": to_jsonable(output),
                "underlying_SMC_operation_ids": [
                    f"SMC-{number:03d}" for number in _POINT4_TO_SMC[len(functional_trace) + 1]
                ],
            })
        diagnostics = {
            "valid": (
                len(functional_trace) == 14
                and len(semantic_result.operation_trace) == 54
                and not graph_errors
            ),
            "source_step_count": len(functional_trace),
            "underlying_SMC_operation_count": len(semantic_result.operation_trace),
            "underlying_operation_order_valid": [
                item.get("operation_id") for item in semantic_result.operation_trace
            ] == [f"SMC-{index:03d}" for index in range(1, 55)],
            "graph_valid": not graph_errors,
            "graph_error_count": len(graph_errors),
            "concept_inventory_size": len(request.concept_inventory),
            "predicate_frame_inventory_size": len(request.predicate_frames),
            "tokens_with_semantic_candidates": sum(bool(row.get("candidates")) for row in word_concepts),
            "predicate_count": len(propositions),
            "modifier_relation_count": len(modifier_relations),
            "competing_reading_count": len(readings),
            "incremental_prefix_count": len(prefix_results),
            "incremental_prefix_count_matches_tokens": len(prefix_results) == len(request.tokens),
            "unresolved_required_role_count": unresolved_role_count,
            "semantic_compatibility_conflict_count": sum(
                row["status"] == "possible_selectional_tension" for row in compatibility
            ),
            "biological_simulation_enabled": biological.get("model") != "disabled",
            "biological_reference_is_measured_human_data": False,
            "contextual_pragmatic_inference_performed": False,
            "warnings": list(semantic_result.diagnostics.get("scientific_limitations", ())) + [
                "A caller-supplied concept inventory and predicate frames are needed for well-grounded lexical senses and role constraints.",
                "When Point 2 supplies multiple senses, candidates are retained separately; the underlying SMC engine re-estimates concept support from its inventory and configuration.",
                "Fallback predicate frames and dependency-to-role mappings are transparent engineering defaults, not language universals.",
                "Modifier composition edges are based on Point 3 dependency labels and supplied Point 2 features.",
                "Prefix snapshots recompute each available word prefix through the existing batch engine; this is an inspectable approximation of incremental neural composition.",
                "Tense/aspect, negation and quantifier scope are only as accurate as the underlying compact English-oriented semantic-composition rules.",
                "No discourse history, physical situation, speaker intention, sarcasm or irony is inferred in Point 4.",
            ],
        }
        handoff = functional_outputs[-1]["handoff"]
        return Point4Result(
            sequence_id=request.sequence_id,
            tokens=request.tokens,
            word_concepts=word_concepts,
            semantic_workspace=semantic_result.combined_meaning,
            semantic_heads=semantic_heads,
            entity_variables=entity_variables,
            event_variables=event_variables,
            predicate_arguments=predicate_arguments,
            thematic_roles=thematic_roles,
            propositions=propositions,
            modifier_relations=modifier_relations,
            operators=operators,
            quantifier_scopes=semantic_result.quantifier_scopes,
            causal_relations=semantic_result.causal_relations,
            semantic_compatibility=compatibility,
            competing_readings=readings,
            incremental_updates=prefix_results,
            proposition_completeness=completeness,
            proposition_graph=graph,
            confidence=max(0.0, min(1.0, confidence)),
            functional_trace=functional_trace,
            biological_trace=biological_trace,
            biological_summary=biological,
            handoff=handoff,
            diagnostics=diagnostics,
            annotation_report=self.annotation_report,
        )
