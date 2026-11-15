"""The inspectable 54-operation registry for sentence-level semantic composition.

Every operation has a stable identifier, a plain-English functional statement,
a mathematical form, and a deliberately qualified biological-reference analogy.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class OperationSpec:
    """Describe one executable semantic-composition operation and its evidence boundary."""

    operation_id: str
    stage_id: str
    stage_name: str
    name: str
    description: str
    mathematical_form: str
    biological_reference: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StageSpec:
    """Describe one of the nine ordered sentence-semantic processing stages."""

    stage_id: str
    name: str
    purpose: str
    operation_ids: tuple[str, ...]
    biological_reference: str


_RAW = (
    ("S1", "Retrieve individual word concepts", "concept_retrieval",
     "Recover concept candidates for each token from a caller-supplied, provenance-tracked inventory.",
     "Distributed lexical-semantic activation and cue-driven retrieval; not a literal neuronal lookup.",
     (
       ("Normalize token cue", "Normalize Unicode and case while preserving the original token.", "n = normalize(w)", "Orthographic/lexical populations compare a stable word-form cue.", ("surface_token",), ("normalized_form",)),
       ("Retrieve lexical matches", "Look up exact labels and aliases, then retain bounded near-match candidates.", "L(c|w) = max_a similarity(w,a)", "Lexical populations activate learned word-form/concept associations.", ("normalized_form", "concept_inventory"), ("candidate_ids",)),
       ("Apply explicit concept constraints", "Restrict or prioritize candidates named by upstream lexical hypotheses.", "C' = C ∩ H when H is supplied", "Top-down lexical predictions bias candidate activation.", ("candidate_ids", "upstream_hypotheses"), ("constrained_candidates",)),
       ("Retrieve concept features", "Attach each candidate's semantic feature map, lexical category, and provenance.", "f_c = inventory[c].features", "Distributed semantic feature populations reinstate learned feature patterns.", ("constrained_candidates", "concept_inventory"), ("candidate_feature_maps",)),
       ("Integrate contextual evidence", "Measure each candidate's compatibility with caller-supplied sentence context.", "X(c) = cosine(f_c, x_context)", "Context-sensitive feedback changes the gain of lexical-semantic populations.", ("candidate_feature_maps", "context_features"), ("context_fit",)),
       ("Estimate word-concept posterior", "Combine lexical, context, and frequency evidence and preserve close alternatives.", "P(c|w,x) = softmax(S(c)/T)", "Competitive activation and lateral inhibition settle candidate activity.", ("lexical_scores", "context_fit", "frequency"), ("word_concept_posteriors",)),
     )),
    ("S2", "Combine word meanings", "semantic_binding",
     "Bind token concepts into a sentence workspace while preserving token-local meanings.",
     "Temporal co-activation, distributed feature convergence, and recurrent integration.",
     (
       ("Initialize composition workspace", "Create ordered token nodes and an empty sentence feature state.", "H_0 = 0", "Working-memory populations maintain ordered lexical-semantic activity.", ("word_concept_posteriors",), ("workspace",)),
       ("Project concepts into token nodes", "Create token-indexed semantic nodes with weighted candidate features.", "h_i = Σ_c P(c|w_i) f_c", "Distributed semantic features co-activate in response to each word.", ("word_concept_posteriors",), ("token_semantic_nodes",)),
       ("Modulate by sentence context", "Apply context-feature gain without overwriting the retrieved concept evidence.", "h_i' = clip(h_i + g_x x, 0, 1)", "Context feedback modulates semantic feature population gain.", ("token_semantic_nodes", "context_features"), ("contextual_token_nodes",)),
       ("Pool shared semantic features", "Aggregate compatible features across tokens using bounded weighted pooling.", "H_f = Σ_i α_i h_i / (Σ_i α_i + ε)", "Temporal semantic hubs integrate distributed feature activity.", ("contextual_token_nodes",), ("pooled_sentence_features",)),
       ("Compute associative co-activation", "Measure pairwise feature overlap and semantic relatedness between token nodes.", "A_ij = cosine(h_i', h_j')", "Associative connections support co-activation of related concepts.", ("contextual_token_nodes",), ("association_matrix",)),
       ("Preserve compositional distinctions", "Retain token-specific and role-sensitive features instead of flattening the sentence to a bag of words.", "H = {h_i', A_ij, H_f}", "Recurrent binding maintains local feature patterns alongside global context.", ("contextual_token_nodes", "association_matrix", "pooled_sentence_features"), ("compositional_workspace",)),
     )),
    ("S3", "Predicate-argument composition", "predicate_argument",
     "Identify predicates, retrieve their valency frames, and bind syntactic dependents to argument slots.",
     "Verb-centered temporal/frontal integration with maintained participant representations.",
     (
       ("Identify predicate candidates", "Use upstream POS/dependency evidence and concept categories to find event or relational predicates.", "V = {i : POS_i ∈ predicate_categories}", "Predicate-sensitive populations respond to lexical category and construction cues.", ("tokens", "word_concepts", "syntax_result"), ("predicate_candidates",)),
       ("Retrieve predicate frames", "Retrieve caller-supplied valency frames by predicate lemma or concept ID.", "F(v) = frame_index[lemma(v)]", "Lexical event knowledge reinstates learned argument-structure expectations.", ("predicate_candidates", "predicate_frame_inventory"), ("predicate_frames",)),
       ("Extract syntactic dependents", "Read predicate-linked subjects, objects, obliques, and complements from syntax output.", "D(v) = {d : head(d)=v}", "Syntactic dependency activity links a predicate with candidate participants.", ("predicate_candidates", "dependency_arcs"), ("predicate_dependents",)),
       ("Create ordered argument slots", "Instantiate role slots from frame valency and attach candidate token indexes.", "A_v = {(role_k, token_j)}", "Working-memory and relational populations maintain participant slots.", ("predicate_frames", "predicate_dependents"), ("argument_slots",)),
       ("Score predicate-argument compatibility", "Combine dependency fit, role expectation, and semantic feature compatibility.", "s(v,a,r)=w_s S_syn + w_f cosine(f_a,p_r)", "Semantic constraints feed back to predicate-argument selection.", ("argument_slots", "selectional_preferences", "token_features"), ("argument_compatibility",)),
       ("Resolve argument completeness", "Mark roles as filled, optional, missing, or unresolved without inventing participants.", "status(r) ∈ {filled, optional, missing, unresolved}", "Control and working-memory populations track unsatisfied role expectations.", ("argument_compatibility", "required_roles"), ("predicate_argument_frames",)),
     )),
    ("S4", "Assign thematic roles", "thematic_role",
     "Map syntactic participants to semantic roles such as Agent, Patient, Experiencer, and Theme.",
     "Role-sensitive event representations with syntax-to-semantics convergence.",
     (
       ("Generate role candidates", "Enumerate roles licensed by each predicate frame and construction.", "R_v = frame(v).roles", "Event populations activate candidate participant-role bindings.", ("predicate_argument_frames",), ("role_candidates",)),
       ("Map syntax to role evidence", "Translate dependency relations and grammatical voice into candidate role evidence.", "E_syn(a,r) = relation_voice_match(a,r)", "Syntactic and frontal activity biases likely event-role assignments.", ("predicate_argument_frames", "dependency_arcs", "voice_cues"), ("syntactic_role_evidence",)),
       ("Add semantic feature evidence", "Compare each participant's concept features with the frame's role preferences.", "E_sem(a,r) = cosine(f_a, preference_r)", "Distributed semantic features constrain role population activity.", ("role_candidates", "token_features", "selectional_preferences"), ("semantic_role_evidence",)),
       ("Combine role evidence", "Weight syntactic and semantic evidence while retaining evidence provenance.", "S(a,r)=w_s E_syn + w_m E_sem", "Convergent inputs bias role-selective populations.", ("syntactic_role_evidence", "semantic_role_evidence"), ("role_scores",)),
       ("Normalize role competition", "Normalize competing role scores for each participant with a stable softmax.", "P(r|a,v)=softmax(S(a,r)/T)", "Lateral inhibition models competition among candidate role bindings.", ("role_scores",), ("role_posteriors",)),
       ("Select or abstain on roles", "Select roles only above confidence and margin gates; otherwise return ranked alternatives.", "select iff p_1≥τ and p_1-p_2≥δ", "Control populations resolve competition or preserve unresolved activity.", ("role_posteriors",), ("thematic_role_assignments",)),
     )),
    ("S5", "Construct propositions", "proposition",
     "Build typed predicate-centered propositions and bind their participants and event metadata.",
     "Event-semantic integration and maintenance of predicate-participant bindings.",
     (
       ("Create proposition shells", "Create one proposition record for each supported predicate event.", "p_v = {predicate:v, arguments:∅}", "Predicate-centered activity initiates an event representation.", ("predicate_candidates",), ("proposition_shells",)),
       ("Bind entity references", "Link argument token nodes to stable within-sentence entity references.", "e_i = canonical_ref(token_i)", "Temporal binding maintains participant identity across an event.", ("proposition_shells", "argument_slots"), ("entity_bindings",)),
       ("Attach thematic arguments", "Attach selected and unresolved role bindings to their proposition.", "p.arguments = {(role,e)}", "Role and event populations converge on participant structure.", ("entity_bindings", "thematic_role_assignments"), ("argument_bound_propositions",)),
       ("Encode event attributes", "Attach tense, aspect, voice, and event-kind cues when syntax supplies evidence.", "event = (predicate, roles, tense, aspect, voice)", "Distributed temporal and event-feature populations contribute event attributes.", ("argument_bound_propositions", "syntax_result"), ("event_propositions",)),
       ("Check proposition completeness", "Validate predicate identity, argument references, and missing required roles.", "valid(p)=predicate_present ∧ refs_resolve", "Prediction-error populations signal incomplete event bindings.", ("event_propositions", "predicate_frames"), ("validated_propositions",)),
       ("Estimate proposition confidence", "Combine predicate, argument, role, and upstream syntax confidence without treating it as truth.", "C(p)=weighted_mean(C_v,C_a,C_r,C_syn)", "Population-level activation summarizes representational support, not truth.", ("validated_propositions", "evidence_scores"), ("proposition_confidence",)),
     )),
    ("S6", "Represent negation and modality", "negation_modality",
     "Represent negative polarity and modal force as explicit operators with scoped targets.",
     "Frontal control and prediction-error processing over proposition representations.",
     (
       ("Detect negation operators", "Find explicit negative words, contractions, and syntax negation dependencies.", "N = {i : token_i ∈ negation_lexicon}", "Control populations detect polarity-changing linguistic cues.", ("tokens", "dependency_arcs"), ("negation_cues",)),
       ("Determine negation scope", "Attach each negation cue to the closest supported predicate or clause scope.", "scope(n)=argmin_p distance(n,p) subject to clause_fit", "Working memory maintains the target of a polarity operator.", ("negation_cues", "proposition_spans"), ("negation_scopes",)),
       ("Apply proposition polarity", "Set proposition polarity while preserving positive lexical content and operator provenance.", "polarity(p)=(-1)^count(neg_scope(p))", "Prediction-error and control populations distinguish positive from negative propositions.", ("negation_scopes", "validated_propositions"), ("polarized_propositions",)),
       ("Detect modal operators", "Detect possibility, necessity, ability, obligation, and prediction markers.", "M = {i : token_i ∈ modal_lexicon}", "Frontal control detects modal and task-force cues.", ("tokens", "syntax_result"), ("modal_cues",)),
       ("Estimate modal force", "Assign a transparent qualitative force score and label from the modal lexicon.", "μ(m) ∈ [0,1] by explicit cue table", "Gain modulation represents stronger or weaker commitment in a reference network.", ("modal_cues",), ("modal_force",)),
       ("Scope and compose operators", "Attach negation and modality to propositions and preserve operator order when uncertain.", "p' = MOD_m(NEG_n(p))", "Frontal-temporal integration composes operator and event activity.", ("polarized_propositions", "modal_force", "negation_scopes"), ("operator_composed_propositions",)),
     )),
    ("S7", "Represent quantifiers and their scope", "quantifier_scope",
     "Represent quantified noun phrases, variable domains, and alternative scope readings.",
     "Working-memory maintenance of referents and hierarchical operator-scope integration.",
     (
       ("Detect quantifier expressions", "Identify universal, existential, negative, proportional, and numeral quantifiers.", "Q = {i : token_i ∈ quantifier_lexicon}", "Lexical-semantic populations recognize quantity and distributive cues.", ("tokens",), ("quantifier_cues",)),
       ("Locate quantified domains", "Link a quantifier to its noun phrase head using syntax or a local fallback.", "domain(q)=NP_head(q)", "Syntactic working memory binds a quantifier to its nominal domain.", ("quantifier_cues", "dependency_arcs", "tokens"), ("quantified_domains",)),
       ("Create variable bindings", "Create typed variables and restrictor predicates for quantified entities.", "q x : Restrictor(x)", "Relational binding maintains an entity variable and its property set.", ("quantified_domains", "word_concepts"), ("quantifier_variables",)),
       ("Generate scope candidates", "Enumerate bounded proposition-scope orders when multiple operators interact.", "S = permutations(Q) bounded by K", "Working memory maintains competing hierarchical scope hypotheses.", ("quantifier_variables", "propositions"), ("scope_candidates",)),
       ("Interpret quantifier operators", "Map quantifier labels to symbolic ∀, ∃, ¬∃, proportional, or exact-count operators.", "⟦every⟧=∀; ⟦some⟧=∃; ⟦no⟧=¬∃", "Abstract semantic populations represent operator type and restrictor relation.", ("quantifier_variables", "scope_candidates"), ("quantified_formulas",)),
       ("Preserve scope ambiguity", "Rank scope readings using syntax and linear-order evidence without collapsing unresolved readings.", "P(S_k) ∝ exp(score(S_k)/T)", "Control and working-memory populations maintain unresolved scope alternatives.", ("quantified_formulas", "scope_candidates"), ("quantifier_scope_readings",)),
     )),
    ("S8", "Represent cause and effect", "causal_inference",
     "Identify explicit causal constructions and represent directed cause-to-effect relations.",
     "Event integration with temporal order, causal cue detection, and prediction-error comparison.",
     (
       ("Detect causal markers", "Find explicit connectives and causative predicate constructions.", "C = {i : token_i ∈ causal_lexicon}", "Lexical and event populations respond to causal connective cues.", ("tokens", "propositions"), ("causal_cues",)),
       ("Align cause and effect clauses", "Map causal cues or causative predicate arguments to event propositions.", "(cause,effect)=align(cue, clause_spans)", "Working memory binds two event representations in a relation frame.", ("causal_cues", "propositions", "syntax_result"), ("cause_effect_candidates",)),
       ("Infer relation direction", "Use connective-specific direction rules and argument roles to orient each candidate edge.", "cause→effect according to cue_frame", "Temporal/event populations represent directional event relations.", ("cause_effect_candidates",), ("directed_causal_candidates",)),
       ("Integrate temporal evidence", "Add explicit tense, aspect, and temporal connective evidence without assuming temporal order proves causation.", "S_t = cue_time_fit(event_cause,event_effect)", "Temporal populations contribute order evidence separately from causal evidence.", ("directed_causal_candidates", "event_propositions"), ("temporal_causal_evidence",)),
       ("Score causal support", "Combine explicit-cue strength, structural fit, and temporal compatibility with a conservative gate.", "C_cause=w_c cue+w_s syntax+w_t temporal", "Prediction-error circuits compare expected and observed event relations.", ("temporal_causal_evidence",), ("causal_confidence",)),
       ("Build causal relation graph", "Emit directed, provenance-labelled edges and retain uncertain candidate links.", "G_c=(V_events,E_cause)", "Distributed event representations integrate candidate causal links.", ("causal_confidence", "directed_causal_candidates"), ("causal_graph",)),
     )),
    ("S9", "Generate sentence-level semantic representations", "sentence_integration",
     "Integrate entities, propositions, operators, quantifiers, and causal edges into one inspectable output graph.",
     "Recurrent integration across temporal semantic, frontal control, and working-memory references.",
     (
       ("Assemble semantic graph nodes", "Create typed nodes for concepts, entities, predicates, propositions, and operators.", "V = V_concept ∪ V_entity ∪ V_event ∪ V_operator", "Distributed semantic populations contribute the component node representations.", ("word_concepts", "propositions", "operator_composed_propositions"), ("semantic_nodes",)),
       ("Assemble semantic graph edges", "Connect nodes using argument, role, scope, quantifier, and causal relations.", "E = E_arg ∪ E_role ∪ E_scope ∪ E_cause", "Relational populations bind semantic nodes into a structured network.", ("semantic_nodes", "thematic_role_assignments", "quantifier_scope_readings", "causal_graph"), ("semantic_edges",)),
       ("Integrate sentence meaning", "Combine proposition-level content with operator and relation structure without flattening scope.", "S = (V,E,Φ,H)", "Recurrent temporal-frontal activity integrates local content and global relations.", ("semantic_nodes", "semantic_edges", "pooled_sentence_features"), ("integrated_semantic_graph",)),
       ("Estimate global uncertainty", "Summarize posterior entropy, unresolved roles, missing concepts, and competing readings.", "U = H(P_concepts)+H(P_roles)+H(P_scope)", "Control and prediction-error populations track unresolved competition.", ("integrated_semantic_graph", "word_concept_posteriors", "role_posteriors", "scope_candidates"), ("uncertainty_report",)),
       ("Validate representation integrity", "Check references, graph endpoints, scope targets, and JSON-safe values.", "valid = all(references_resolve ∧ endpoints_exist)", "Monitoring populations provide an error-checking analogue for integrated output.", ("integrated_semantic_graph", "uncertainty_report"), ("validated_sentence_representation",)),
       ("Serialize complete semantic result", "Return the sentence representation, evidence provenance, all stage traces, and scientific limitations.", "R = serialize(S, trace, diagnostics)", "A final population readout is a software reporting analogy, not a neural readout claim.", ("validated_sentence_representation", "operation_trace", "biological_summary"), ("semantic_composition_result",)),
     )),
)


def _build_registry() -> tuple[tuple[OperationSpec, ...], tuple[StageSpec, ...]]:
    """Expand the compact source table into stable operation and stage records."""
    operations: list[OperationSpec] = []
    stages: list[StageSpec] = []
    counter = 1
    for stage_code, stage_name, population, purpose, biological, rows in _RAW:
        operation_ids: list[str] = []
        stage_id = f"STAGE-{stage_code}"
        for name, description, formula, bio_note, inputs, outputs in rows:
            operation_id = f"SMC-{counter:03d}"
            operation_ids.append(operation_id)
            operations.append(OperationSpec(
                operation_id, stage_id, stage_name, name, description, formula,
                bio_note, tuple(inputs), tuple(outputs),
            ))
            counter += 1
        stages.append(StageSpec(
            stage_id, stage_name, purpose, tuple(operation_ids), biological,
        ))
    return tuple(operations), tuple(stages)


OPERATION_REGISTRY, STAGE_REGISTRY = _build_registry()


def registry_as_dicts() -> list[dict[str, Any]]:
    """Return the stable 54-operation registry as ordinary dictionaries for docs and APIs."""
    return [asdict(item) for item in OPERATION_REGISTRY]


def stages_as_dicts() -> list[dict[str, Any]]:
    """Return the nine stage descriptions as ordinary dictionaries for docs and APIs."""
    return [asdict(item) for item in STAGE_REGISTRY]
