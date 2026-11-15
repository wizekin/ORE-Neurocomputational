"""Stable 64-operation registry for the eight-stage ORE discourse algorithm.

Each operation records a functional responsibility, an inspectable mathematical
sketch, and a biologically motivated computational analogy.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class StageSpec:
    """Describe one discourse stage and the representation it produces."""

    stage_id: str
    stage_name: str
    operation_count: int
    primary_output: str
    biological_systems: str

    def to_dict(self) -> dict[str, object]:
        """Return the stage definition as a plain dictionary."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class OperationSpec:
    """Describe one executable operation and its functional/neural interpretation."""

    operation_id: str
    stage_id: str
    stage_name: str
    operation_name: str
    functional_responsibility: str
    mathematical_form: str
    biological_reference: str

    def to_dict(self) -> dict[str, str]:
        """Return the operation definition as a plain dictionary."""
        return asdict(self)


_STAGE_ROWS = (
    ("S1", "Maintain discourse context", "Entities, active memory, common ground and information status", "Hippocampal/temporal memory cues, temporal-parietal binding and frontal working-memory control"),
    ("S2", "Integrate consecutive sentences", "Linked propositions, coherence relations and bridging inferences", "Posterior temporal language networks with inferior-parietal and frontal integration"),
    ("S3", "Identify speaker intention", "Belief, goal, communicative-act and mental-state hypotheses", "Distributed social-cognition and mentalizing networks interacting with language/control systems"),
    ("S4", "Understand indirect requests", "Separated literal interpretation and intended listener action", "Pragmatic inference and frontal control interacting with temporal semantic representations"),
    ("S5", "Interpret irony", "Contextual, semantic and prosodic incongruity evidence", "Right-lateralized prosody-sensitive and bilateral semantic/pragmatic integration (coarse analogy)"),
    ("S6", "Construct situation models", "People, events, states, motivations and causal event graph", "Temporal-parietal event construction, hippocampal relational binding and frontal goal maintenance"),
    ("S7", "Track temporal discourse relations", "Ordered event intervals, overlap, duration and temporal shifts", "Temporal sequencing and working-memory networks supporting event order and interval integration"),
    ("S8", "Maintain topic and narrative coherence", "Global topic, narrative progression, consistency and summary", "Distributed semantic networks with frontal monitoring and hippocampal event integration"),
)

# Each row: stage index, operation label, functional description, mathematical sketch, neural analogy.
_ROWS = (
    (0, "Initialize discourse state", "Create a bounded state for the current text, prior turns, entities and evidence channels.", "D_0 = {M_0, E_0, G_0, I_0}", "Working-memory gating establishes an active context state."),
    (0, "Retrieve prior-turn memory", "Read caller-supplied prior turns and episodic records without modifying their source.", "R_t = retrieve(q_t, M_{t-1}); confidence = sim(q_t, m)", "Cue-dependent reinstatement is a hippocampal/temporal memory analogy."),
    (0, "Extract entity mentions", "Collect explicit entity records and conservative surface-form mentions with sentence locations.", "m_i = (entity_id, span_i, sentence_i, features_i)", "Temporal lexical populations activate candidate referents from word-form cues."),
    (0, "Resolve entity references", "Link pronouns and repeated mentions to candidates using recency, salience and feature compatibility.", "P(e|m) = softmax(w_r R + w_s S + w_f F)", "Recurrent temporal-parietal binding supports a candidate referent representation."),
    (0, "Update entity salience", "Combine recency, mention frequency and topical relevance into bounded entity activation.", "S_e(t) = aR_e + bF_e + cT_e", "Attention and memory-control populations maintain task-relevant referents."),
    (0, "Integrate common ground", "Merge caller-declared shared facts while preserving their provenance and source labels.", "G_t = G_{t-1} union G^{caller}_t", "Shared-context activation is a distributed memory hypothesis, not a dedicated common-ground store."),
    (0, "Track given-new information", "Label each proposition/entity as given, bridging, or newly introduced relative to active context.", "status(x_t) = 1[x in M_{t-1}] else NEW", "Old/new sensitivity and working-memory updating provide a coarse functional analogy."),
    (0, "Apply working-memory decay", "Reduce inactive salience while retaining durable entities and explicitly pinned common ground.", "S'_e = S_e exp(-lambda Delta t)", "Time-dependent activity decay and gated maintenance approximate limited working memory."),
    (1, "Extract sentence propositions", "Represent each sentence as a predicate-centered proposition with arguments, polarity and modality.", "p_i = (predicate_i, args_i, neg_i, modality_i)", "Temporal language populations activate clause/event candidates."),
    (1, "Link adjacent propositions", "Generate links between neighboring sentence propositions when entity or event evidence overlaps.", "L(i,j) = alpha J(E_i,E_j) + beta sim(F_i,F_j)", "Recurrent temporal/parietal integration binds adjacent clause representations."),
    (1, "Detect discourse connectives", "Detect explicit contrast, cause, explanation, condition, elaboration and sequence markers.", "C_i = indicator(connective_i in lexicon)", "Connective-sensitive language populations bias relation-specific integration."),
    (1, "Score semantic continuity", "Combine shared entities, predicate compatibility and topic-feature overlap.", "cont(i,j) = w_e O_e + w_p sim(P_i,P_j) + w_t cos(T_i,T_j)", "Semantic integration can be approximated by overlapping distributed activation."),
    (1, "Infer coherence relations", "Rank elaboration, contrast, explanation, cause, result and temporal-sequence hypotheses.", "P(r|i,j) = softmax(score_r(i,j)/T)", "Frontal-temporal competition selects a relation while retaining alternatives."),
    (1, "Generate bridging inferences", "Add only supportable links that connect an introduced entity or event to explicit neighboring evidence.", "B_i = argmax_b score(b|p_{i-1}, p_i, G)", "Associative completion resembles memory-supported integration; it is not unconstrained fact generation."),
    (1, "Build proposition graph", "Store proposition nodes and typed discourse edges with validated endpoint identifiers.", "G_P = (V_P, E_relation)", "Distributed relational binding provides a graph-like computational analogy."),
    (1, "Estimate inferential compatibility", "Measure whether adjacent propositions can jointly hold given polarity and explicit relations.", "K(i,j) = 1 - contradiction(i,j) + support(i,j)", "Prediction-error monitoring flags incompatible activated sentence representations."),
    (2, "Initialize speaker belief state", "Load explicit beliefs and distinguish them from beliefs inferred from utterance evidence.", "B_s^0 = B_s^{caller}", "Mental-state representations are modeled as distributed hypotheses, not direct access to another mind."),
    (2, "Infer communicative act", "Estimate assertion, question, request, command, promise, apology and greeting probabilities.", "P(a|u,c) = softmax(W_a phi(u,c)/T)", "Language-act cues recruit temporal language and frontal action/control representations."),
    (2, "Estimate speaker goals", "Retrieve declared goals and infer bounded conversational goals from the utterance form.", "P(g|u,c) proportional to P(u|g,c)P(g)", "Goal maintenance and social inference are represented by interacting population activity."),
    (2, "Infer mental-state cues", "Identify explicit belief, desire, knowledge, uncertainty and intention expressions.", "MS_s = {belief, desire, know, intend, uncertainty}", "Mentalizing-network activity is a coarse functional reference, not a neural localization claim."),
    (2, "Compare speaker model with common ground", "Estimate whether the speaker treats a proposition as shared, new, uncertain or contested.", "d_s = divergence(B_s, G)", "Joint social-context and memory integration is approximated by cross-population interaction."),
    (2, "Estimate epistemic commitment", "Estimate commitment from assertion strength, hedges, evidentials and modality.", "c(p) = clip(assertion - hedge - uncertainty, 0, 1)", "Graded commitment resembles graded population evidence and frontal monitoring."),
    (2, "Rank speaker intentions", "Combine act, goal, belief and context evidence into a ranked intention distribution.", "P(I|u,c) = softmax(s_act+s_goal+s_belief)", "Competitive selection among pragmatic hypotheses is a network-level analogy."),
    (2, "Preserve intent alternatives", "Retain near-tied intentions and abstain from a forced mental-state conclusion.", "margin = P(I_1)-P(I_2); abstain if margin < theta", "Parallel candidate activity and controlled selection preserve uncertainty."),
    (3, "Derive literal utterance meaning", "Retain the sentence's question, proposition or imperative interpretation before pragmatic enrichment.", "L(u) = parse_literal(u)", "Lexical-semantic representations provide the literal baseline for pragmatic inference."),
    (3, "Detect indirect-request constructions", "Recognize conventional ability, willingness, permission and hint constructions.", "C_indirect = 1[pattern(u) in request_templates]", "Frontal/temporal construction and action representations are jointly recruited."),
    (3, "Map utterance to intended action", "Generate a listener-action hypothesis without overwriting the literal question or statement.", "P(a|u,c) proportional to P(u|a,c)P(a|c)", "Pragmatic inference maps linguistic cues onto action/goal representations."),
    (3, "Integrate listener and scene context", "Use explicit addressee abilities, scene state and shared goals to rank feasible actions.", "score(a) = log P(a|u) + gamma feasibility(a,c)", "Context-sensitive frontal control biases candidate actions."),
    (3, "Estimate politeness and indirectness", "Estimate surface directness separately from request strength and politeness cues.", "indirectness = 1 - directness(u); politeness = cue_sum(u)", "Social-pragmatic evaluation is a distributed computational analogy."),
    (3, "Discriminate question from request", "Compare information-seeking and action-seeking interpretations for interrogative forms.", "P(request|u,c) = sigmoid(s_action - s_information)", "Action and language populations compete under contextual control."),
    (3, "Score intended action confidence", "Combine pattern, action plausibility and listener-context fit with an abstention threshold.", "conf(a) = sigmoid(w^T x_a)", "Graded action-intention activity supports uncertainty-aware output."),
    (3, "Preserve literal/intended distinction", "Return literal content, inferred request and evidence as separate fields.", "U = {L(u), I(a), evidence, uncertainty}", "Multiple representational channels remain active rather than being collapsed."),
    (4, "Detect lexical polarity", "Identify explicit positive, negative and scalar evaluative cues in the sentence.", "v(u) = sum_i polarity(token_i)", "Distributed affective-semantic features contribute valence signals."),
    (4, "Retrieve contextual outcome", "Retrieve the local event and outcome context relevant to the evaluative statement.", "C(u) = retrieve(event_{<t}, entities)", "Context reinstatement links current utterance to prior event representations."),
    (4, "Measure semantic valence mismatch", "Compare lexical evaluation with the polarity of the described or known outcome.", "m_sem = max(0, -v(u) v(event))", "Mismatch monitoring is approximated by competing semantic activity."),
    (4, "Integrate prosodic evidence", "Read caller-supplied pitch, stress, duration, laughter and intonation mismatch cues.", "m_pro = sigmoid(w^T prosody)", "Prosody-sensitive auditory populations modulate a multimodal interpretation."),
    (4, "Evaluate pragmatic incongruity", "Combine semantic, contextual and prosodic mismatch without treating any one cue as decisive.", "I = w_s m_sem + w_c m_ctx + w_p m_pro", "Distributed conflict monitoring raises incongruity-related activity."),
    (4, "Estimate irony versus literal reading", "Convert incongruity and explicit irony markers into bounded competing hypotheses.", "P(irony) = sigmoid(k(I-theta))", "Right/left semantic-pragmatic integration is represented as a coarse network hypothesis."),
    (4, "Distinguish irony and sarcasm cues", "Use target, negative evaluation and interpersonal markers to label only supported subtypes.", "P(subtype|cues) = softmax(s_subtype)", "Social-affective interpretation interacts with language and control populations."),
    (4, "Preserve irony evidence and uncertainty", "Return the literal reading, ironic alternative, cue contributions and confidence.", "U = {literal, ironic, m_sem, m_ctx, m_pro}", "Multiple interpretations remain active until evidence is sufficient."),
    (5, "Instantiate discourse participants", "Create people/agent records from explicit entities and supported participant mentions.", "People = unique(agent_mentions)", "Temporal-parietal person/event representations are linked through relational binding."),
    (5, "Extract event predicates", "Build event records from explicit propositions and conservative predicate cues.", "e_i = (predicate, participants, time, polarity)", "Event-selective distributed populations integrate clause and participant signals."),
    (5, "Assign event participant roles", "Map explicit subject, object, recipient and experiencer fields to event roles.", "role(a,e) = caller_role(a) or supported_syntax(a,e)", "Thematic-role and action representations interact with temporal language networks."),
    (5, "Track entity states", "Record explicit states and state changes without filling missing properties from stereotypes.", "state_t(e) = update(state_{t-1}(e), explicit_change)", "Relational memory binds entities to changing event/state features."),
    (5, "Infer goal and motivation hypotheses", "Connect explicit desires, plans and purpose clauses to participant motivations.", "P(m|e) proportional to P(e|m)P(m)", "Goal representations interact with event construction and working memory."),
    (5, "Add causal event structure", "Link events using explicit cause/result connectives or caller-provided causal evidence.", "cause(e_i,e_j) = evidence(connective, explicit_cause)", "Hippocampal relational binding and temporal-parietal event integration provide an analogy."),
    (5, "Build situation event graph", "Assemble people, events, states, goals and typed causal relations.", "G_S = (V_people union V_event union V_state, E_role union E_cause)", "Distributed relational binding forms a situation-model-like network state."),
    (5, "Validate situation consistency", "Check IDs, participant references, causal endpoints and explicit state contradictions.", "valid(G_S) = 1[all endpoints valid and no hard conflict]", "Prediction-error and control systems monitor model inconsistency."),
    (6, "Extract temporal expressions", "Read explicit dates, clock expressions, durations and temporal adverbs.", "T_i = parse_time(surface_i, metadata_i)", "Temporal-language and memory systems activate temporal anchors."),
    (6, "Infer event ordering", "Convert before, after, then, first, finally and chronological metadata into directed order links.", "i before j implies t_i < t_j", "Sequential event representations support ordered temporal binding."),
    (6, "Estimate event duration and intervals", "Represent explicit starts, ends and duration ranges as bounded intervals.", "I_i = [start_i, end_i]; duration_i = end_i-start_i", "Interval maintenance is a working-memory/computational analogy."),
    (6, "Detect overlap and containment", "Compare interval bounds to identify overlaps, containment and non-overlap.", "overlap(i,j) = max(0, min(end_i,end_j)-max(start_i,start_j))", "Temporal integration compares concurrent event representations."),
    (6, "Resolve temporal connectives", "Assign sentence-level temporal relation labels while retaining ambiguous alternatives.", "P(r|i,j) = softmax(score_r(time_i,time_j,connective))", "Connective-sensitive temporal populations bias order/overlap hypotheses."),
    (6, "Detect time shifts and anchors", "Track tense/adverbial changes and explicit narrative time jumps.", "shift_i = t_anchor(i)-t_anchor(i-1)", "Temporal reference maintenance and updating are modeled as recurrent state changes."),
    (6, "Normalize discourse timeline", "Build a stable event order and interval table from supported temporal constraints.", "order = topological_sort(E_before); intervals = normalize(I)", "Sequential binding produces an ordered event representation."),
    (6, "Validate temporal constraints", "Detect cycles, negative durations, impossible interval bounds and unresolved anchors.", "valid(T) = acyclic(E_before) and duration_i >= 0", "Prediction-error monitoring flags incompatible temporal constraints."),
    (7, "Aggregate global topic features", "Build a frequency-normalized content-word feature distribution across the discourse.", "T_d(w) = tf(w)/sum_v tf(v)", "Distributed semantic populations contribute topic-level activation."),
    (7, "Estimate topic continuity", "Compare adjacent sentence feature vectors using cosine similarity.", "sim(T_i,T_j) = dot(T_i,T_j)/(norm_i norm_j + eps)", "Recurrent semantic activation maintains a topic representation."),
    (7, "Track narrative progression", "Measure the introduction, recurrence and resolution of participants and event predicates.", "progress = recurrence + new_event_rate + resolved_goals", "Hippocampal event sequences and frontal goal tracking provide a functional analogy."),
    (7, "Integrate temporal and causal structure", "Measure how much of the event graph is supported by explicit temporal and causal links.", "integration = |E_time union E_cause|/(|V_event|+eps)", "Relational binding integrates event order and causal dependencies."),
    (7, "Detect explicit contradictions", "Find same-entity/same-predicate propositions with incompatible polarity or state values.", "conflict(i,j) = 1[args_i=args_j and polarity_i != polarity_j]", "Conflict/error-monitoring populations flag incompatible activated propositions."),
    (7, "Compute global narrative coherence", "Combine entity continuity, topic continuity, relation support and consistency.", "C = w_e E + w_t T + w_r R + w_k K", "Distributed semantic and memory signals are monitored by frontal control."),
    (7, "Revise a bounded global discourse state", "Reweight local hypotheses by global evidence while preserving original observations.", "q' = (1-rho)q + rho softmax(log(q)+g global)", "Top-down semantic feedback modulates but does not replace bottom-up sentence evidence."),
    (7, "Generate coherent state summary", "Serialize the current entities, topic, events, time line, unresolved hypotheses and confidence.", "summary = project(G_context, G_prop, G_situation, G_time)", "A distributed global representation is read out for downstream language/action systems."),
)

STAGE_REGISTRY = tuple(
    StageSpec(
        stage_id=row[0], stage_name=row[1], operation_count=8,
        primary_output=row[2], biological_systems=row[3],
    )
    for row in _STAGE_ROWS
)

OPERATION_REGISTRY = tuple(
    OperationSpec(
        operation_id=f"DC-{index + 1:03d}",
        stage_id=f"S{stage_index + 1}",
        stage_name=_STAGE_ROWS[stage_index][1],
        operation_name=row[1],
        functional_responsibility=row[2],
        mathematical_form=row[3],
        biological_reference=row[4],
    )
    for index, row in enumerate(_ROWS)
    for stage_index in (row[0],)
)

if len(STAGE_REGISTRY) != 8 or len(OPERATION_REGISTRY) != 64:
    raise RuntimeError("Discourse registry must contain eight stages and 64 unique operations")
if len({item.operation_id for item in OPERATION_REGISTRY}) != len(OPERATION_REGISTRY):
    raise RuntimeError("Discourse operation identifiers must be unique")


def registry_as_dicts() -> dict[str, list[dict[str, object]]]:
    """Return stable stage and operation registries for inspection and serialization."""
    return {
        "stages": [stage.to_dict() for stage in STAGE_REGISTRY],
        "operations": [operation.to_dict() for operation in OPERATION_REGISTRY],
    }
