"""Stable 39-operation registry for the eight-stage ORE discourse algorithm.

Each registered operation combines one or more lower-level stage calculations
into a named, traceable unit. The registry links these units to equations,
stage outputs, and qualified biological-reference descriptions.
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
    """Describe one registered operation and its functional/neural interpretation."""

    operation_id: str
    stage_id: str
    stage_name: str
    operation_name: str
    functional_responsibility: str
    mathematical_form: str
    biological_reference: str
    source_output_indices: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        """Return the operation definition, including its stage-output mapping."""
        return asdict(self)


_STAGE_ROWS = (
    ("S1", "Maintain discourse context",
     "Entities, active memory, common ground and information status",
     "Hippocampal/temporal memory cues, temporal-parietal binding and frontal working-memory control"),
    ("S2", "Integrate consecutive sentences",
     "Linked propositions, coherence relations and bridging inferences",
     "Posterior temporal language networks with inferior-parietal and frontal integration"),
    ("S3", "Identify speaker intention",
     "Belief, goal, communicative-act and mental-state hypotheses",
     "Distributed social-cognition and mentalizing networks interacting with language/control systems"),
    ("S4", "Understand indirect requests",
     "Separated literal interpretation and intended listener action",
     "Pragmatic inference and frontal control interacting with temporal semantic representations"),
    ("S5", "Interpret irony",
     "Contextual, semantic and prosodic incongruity evidence",
     "Right-lateralized prosody-sensitive and bilateral semantic/pragmatic integration (coarse analogy)"),
    ("S6", "Construct situation models",
     "People, events, states, motivations and causal event graph",
     "Temporal-parietal event construction, hippocampal relational binding and frontal goal maintenance"),
    ("S7", "Track temporal discourse relations",
     "Ordered event intervals, overlap, duration and temporal shifts",
     "Temporal sequencing and working-memory networks supporting event order and interval integration"),
    ("S8", "Maintain topic and narrative coherence",
     "Global topic, narrative progression, consistency and summary",
     "Distributed semantic networks with frontal monitoring and hippocampal event integration"),
)

# Each row: stage index, name, functional responsibility, equation, neural analogy,
# and the one-based source outputs from that stage's detailed computational bundle.
_ROWS = (
    (0, "Initialize context and retrieve discourse memory",
     "Initialize the active discourse state and retrieve only caller-supplied prior-turn or episodic records.",
     "D_t = initialize(M_t, E_t, G_t, I_t); R_t = retrieve(cue_t, M_t)",
     "Working-memory gating and cue-dependent hippocampal/temporal reinstatement provide a functional analogy.",
     (1, 2)),
    (0, "Extract entity mentions and resolve references",
     "Collect explicit and conservative surface mentions, then rank candidate antecedents using recency, salience and entity compatibility.",
     "P(e|m) = softmax((w_r R + w_s S + w_f F)/T)",
     "Temporal lexical and temporoparietal populations support candidate referent activation and recurrent binding.",
     (3, 4)),
    (0, "Update entity salience",
     "Combine recency, mention frequency and topical relevance into bounded activation for each discourse entity.",
     "S_e(t) = aR_e(t) + b(1-exp(-n_e)) + cT_e",
     "Attention and memory-control populations maintain currently relevant referents.",
     (5,)),
    (0, "Integrate common ground and given/new information",
     "Merge caller-declared shared facts while distinguishing given, bridging and newly introduced information.",
     "G_t = G_(t-1) union G_caller; status(x_t) = GIVEN or NEW",
     "Distributed memory and old/new information updating provide a coarse functional reference.",
     (6, 7)),
    (0, "Apply working-memory decay",
     "Reduce inactive entity activation while preserving caller-pinned common ground and durable context records.",
     "S'_e = S_e * exp(-lambda * Delta_t)",
     "Time-dependent activity decay and gated maintenance approximate limited working-memory persistence.",
     (8,)),

    (1, "Extract and link sentence propositions",
     "Construct predicate-centered propositions and link adjacent sentences using shared participants and semantic evidence.",
     "p_i=(v_i,args_i,neg_i,mod_i); L(i,j)=alpha*J(E_i,E_j)+beta*cos(T_i,T_j)",
     "Posterior temporal and inferior-parietal populations bind successive clause representations.",
     (1, 2)),
    (1, "Detect discourse connectives",
     "Identify explicit contrast, cause, explanation, condition, elaboration and sequence markers.",
     "C_i = 1[connective_i belongs to the discourse cue inventory]",
     "Connective-sensitive language populations bias relation-specific integration.",
     (3,)),
    (1, "Score semantic continuity",
     "Measure continuity from shared entities, predicate compatibility and topic-feature overlap.",
     "cont(i,j)=w_e*J(E_i,E_j)+w_p*sim(P_i,P_j)+w_t*cos(T_i,T_j)",
     "Overlapping distributed semantic activation supports sentence-to-sentence integration.",
     (4,)),
    (1, "Infer coherence relations and bridging links",
     "Rank relation hypotheses and add bridging links only when a reference is supported by explicit or resolved evidence.",
     "P(r|i,j)=softmax(score_r/T); b*=argmax_b score(b|p_i,p_j,G)",
     "Frontal-temporal competition and memory-supported associative completion provide the analogy.",
     (5, 6)),
    (1, "Build proposition graph and assess compatibility",
     "Create a typed proposition graph and separately report inferential support and polarity conflicts.",
     "G_P=(V_P,E_r); K(i,j)=support(i,j)-contradiction(i,j)",
     "Relational binding and prediction-error monitoring provide a graph-like computational reference.",
     (7, 8)),

    (2, "Initialize speaker beliefs and infer communicative act",
     "Load declared speaker beliefs and estimate assertion, question, request, command, promise, apology or greeting.",
     "P(a|u,c)=softmax(W_a*phi(u,c)/T)",
     "Temporal language representations interact with frontal action and control populations.",
     (1, 2)),
    (2, "Estimate speaker goals",
     "Retain declared goals and infer bounded conversational goals from the utterance and discourse context.",
     "P(g|u,c) proportional_to P(u|g,c)*P(g|c)",
     "Goal maintenance and social inference are modeled as interacting population activity.",
     (3,)),
    (2, "Infer explicit mental-state cues",
     "Identify language indicating belief, desire, knowledge, intention, uncertainty or commitment.",
     "MS_s={belief, desire, know, intend, uncertainty}",
     "Mentalizing-network activity is a coarse functional reference, not direct access to another mind.",
     (4,)),
    (2, "Compare speaker model with common ground",
     "Estimate shared versus new or contested content and quantify epistemic commitment from assertion, hedges and modality.",
     "d_s=divergence(B_s,G); c(p)=clip(assertion-hedge-uncertainty,0,1)",
     "Joint social-context, memory and frontal monitoring provide a distributed computational analogy.",
     (5, 6)),
    (2, "Rank speaker intentions and preserve alternatives",
     "Combine act, goal, belief and context evidence; preserve close alternatives and abstain when the margin is weak.",
     "P(I|u,c)=softmax(s_act+s_goal+s_belief); abstain if p1-p2<theta",
     "Competitive pragmatic hypotheses and controlled selection preserve uncertainty.",
     (7, 8)),

    (3, "Preserve literal meaning and detect indirect-request forms",
     "Build the literal sentence interpretation and independently detect conventional ability, willingness and hint constructions.",
     "L(u)=parse_literal(u); C_indirect=1[pattern(u) in request_templates]",
     "Lexical-semantic and frontal/temporal construction representations provide a literal-to-pragmatic bridge.",
     (1, 2)),
    (3, "Map utterance to intended listener action",
     "Generate candidate actions that could satisfy the utterance without overwriting its literal question or statement.",
     "P(a|u,c) proportional_to P(u|a,c)*P(a|c)",
     "Pragmatic inference maps linguistic cues onto action and goal representations.",
     (3,)),
    (3, "Integrate listener and scene context",
     "Rank candidate actions using only supplied addressee abilities, scene state and shared goals.",
     "score(a)=log(P(a|u)+eps)+gamma*feasibility(a,c)",
     "Context-sensitive frontal control biases candidate action populations.",
     (4,)),
    (3, "Estimate politeness and distinguish request from question",
     "Estimate directness and politeness separately, then compare action-seeking and information-seeking readings.",
     "P(request|u,c)=sigmoid(s_action-s_information); indirectness=1-directness",
     "Social-pragmatic evaluation interacts with action and language representations.",
     (5, 6)),
    (3, "Score intended action and preserve literal/intended separation",
     "Apply a confidence and abstention threshold while returning literal content, intended action and supporting cues as distinct fields.",
     "conf(a)=sigmoid(w^T*x_a); U={literal,intended,evidence,uncertainty}",
     "Graded action-intention activity supports uncertainty-aware selection.",
     (7, 8)),

    (4, "Detect lexical polarity and retrieve contextual outcome",
     "Estimate literal valence and retrieve the preceding event or outcome relevant to the evaluative statement.",
     "v(u)=sum_i polarity(token_i); C(u)=retrieve(event_<t,entities)",
     "Distributed affective-semantic features combine with context reinstatement.",
     (1, 2)),
    (4, "Measure semantic valence mismatch",
     "Compare literal evaluative polarity with the polarity of the described or explicitly supplied outcome.",
     "m_sem=max(0,-v(u)*v(event))",
     "Competing semantic representations provide a coarse mismatch-monitoring analogy.",
     (3,)),
    (4, "Integrate prosody and evaluate pragmatic incongruity",
     "Combine caller-supplied intonation, stress, duration and laughter cues with semantic and contextual mismatch.",
     "I=w_s*m_sem+w_c*m_context+w_p*m_prosody",
     "Prosody-sensitive auditory and bilateral semantic/pragmatic populations are represented as a coarse network hypothesis.",
     (4, 5)),
    (4, "Estimate irony/sarcasm alternatives and preserve uncertainty",
     "Rank literal, ironic and sarcasm-candidate readings while retaining cue contributions and an abstention state.",
     "P(irony)=sigmoid(k*(I-theta)); P(subtype|cues)=softmax(s_subtype)",
     "Distributed social-affective interpretation and controlled selection retain competing readings.",
     (6, 7, 8)),

    (5, "Instantiate participants and extract event predicates",
     "Create supported people/participant records and event candidates from explicit propositions and conservative predicate cues.",
     "People=unique(agent_mentions); e_i=(predicate_i,participants_i,time_i,polarity_i)",
     "Temporal-parietal person/event representations are linked through relational binding.",
     (1, 2)),
    (5, "Assign event participant roles",
     "Map explicit subject, object, recipient and experiencer information to predicate-centered event roles.",
     "role(a,e)=caller_role(a) or supported_syntax(a,e)",
     "Thematic-role and action representations interact with temporal language networks.",
     (3,)),
    (5, "Track entity states",
     "Record explicit states and changes while leaving unobserved properties unknown.",
     "state_t(e)=update(state_(t-1)(e),explicit_change)",
     "Relational memory binds entities to changing event and state features.",
     (4,)),
    (5, "Infer goals and causal event links",
     "Connect declared or text-supported goals to participants and link events only when a connective or explicit causal field supports the relation.",
     "P(m|e) proportional_to P(e|m)*P(m); cause(e_i,e_j)=evidence_i,j",
     "Hippocampal relational binding and frontal goal maintenance provide a functional analogy.",
     (5, 6)),
    (5, "Build and validate the situation event graph",
     "Assemble people, events, states, goals and causal edges, then validate references and explicit contradictions.",
     "G_S=(V_people union V_event union V_state,E_role union E_cause); valid(G_S)",
     "Distributed event construction and prediction-error monitoring approximate situation-model integration.",
     (7, 8)),

    (6, "Extract temporal expressions and infer event ordering",
     "Read explicit time cues and convert before/after/then and caller metadata into supported event-order constraints.",
     "T_i=parse_time(surface_i,metadata_i); i before j implies t_i<t_j",
     "Temporal language and sequential event representations support ordered binding.",
     (1, 2)),
    (6, "Estimate event duration and intervals",
     "Represent explicit starts, ends and duration ranges without fabricating missing absolute anchors.",
     "I_i=[start_i,end_i]; duration_i=end_i-start_i",
     "Interval maintenance is a computational analogy for temporal working memory.",
     (3,)),
    (6, "Detect overlap and containment",
     "Compare available interval bounds to identify overlap, containment and non-overlap.",
     "O(i,j)=max(0,min(e_i,e_j)-max(s_i,s_j))",
     "Temporal integration compares concurrent event representations.",
     (4,)),
    (6, "Resolve temporal connectives and detect time shifts",
     "Assign temporal relation labels and track tense/adverbial changes while preserving unresolved anchors.",
     "P(r|i,j)=softmax(score_r(time_i,time_j,connective)); shift_i=t_i-t_(i-1)",
     "Temporal reference maintenance and recurrent updating provide the functional analogy.",
     (5, 6)),
    (6, "Normalize timeline and validate temporal constraints",
     "Build an ordered event timeline and detect cycles, negative durations and incompatible interval bounds.",
     "order=topological_sort(E_before); valid(T)=acyclic(E_before) and duration_i>=0",
     "Sequential binding and prediction-error monitoring provide a computational reference.",
     (7, 8)),

    (7, "Aggregate global topic features",
     "Build a frequency-normalized topic distribution from discourse content words.",
     "T_d(w)=tf(w)/(sum_v tf(v)+eps)",
     "Distributed semantic populations contribute topic-level activation.",
     (1,)),
    (7, "Estimate topic continuity",
     "Compare neighboring sentence topic vectors and retain evidence for likely topic changes.",
     "sim(T_i,T_j)=dot(T_i,T_j)/(norm(T_i)*norm(T_j)+eps)",
     "Recurrent semantic activation maintains a topic representation.",
     (2,)),
    (7, "Track narrative progression",
     "Measure participant and predicate recurrence, event introduction, and declared or resolved goal threads.",
     "progress=recurrence+new_event_rate+resolved_goal_rate",
     "Hippocampal event sequences and frontal goal tracking provide a functional analogy.",
     (3,)),
    (7, "Integrate temporal/causal structure and detect contradictions",
     "Combine supported event links and identify incompatible propositions sharing an explicit participant and predicate.",
     "integration=|E_time union E_cause|/(|V_event|+eps); conflict=1[args_i=args_j and polarity_i!=polarity_j]",
     "Relational binding integrates event order and cause while conflict-monitoring populations flag mismatch.",
     (4, 5)),
    (7, "Revise global state and generate narrative coherence summary",
     "Blend bounded global evidence with local links, calculate coherence and serialize topic, events, conflicts and unresolved hypotheses.",
     "q'=(1-rho)q+rho*softmax(log(q)+g*global); C=w_eE+w_tT+w_rR+w_kK",
     "Top-down semantic feedback and frontal monitoring modulate the distributed global representation.",
     (6, 7, 8)),
)

STAGE_REGISTRY = tuple(
    StageSpec(
        stage_id=row[0],
        stage_name=row[1],
        operation_count=sum(1 for operation in _ROWS if operation[0] == stage_index),
        primary_output=row[2],
        biological_systems=row[3],
    )
    for stage_index, row in enumerate(_STAGE_ROWS)
)

OPERATION_REGISTRY = tuple(
    OperationSpec(
        operation_id=f"DC-{index:03d}",
        stage_id=f"S{row[0] + 1}",
        stage_name=_STAGE_ROWS[row[0]][1],
        operation_name=row[1],
        functional_responsibility=row[2],
        mathematical_form=row[3],
        biological_reference=row[4],
        source_output_indices=row[5],
    )
    for index, row in enumerate(_ROWS, start=1)
)

if len(STAGE_REGISTRY) != 8 or len(OPERATION_REGISTRY) != 39:
    raise RuntimeError("Discourse registry must contain eight stages and 39 operations")
if len({item.operation_id for item in OPERATION_REGISTRY}) != len(OPERATION_REGISTRY):
    raise RuntimeError("Discourse operation identifiers must be unique")
if sum(item.operation_count for item in STAGE_REGISTRY) != len(OPERATION_REGISTRY):
    raise RuntimeError("Stage operation counts must sum to the registry size")
for operation in OPERATION_REGISTRY:
    if not operation.source_output_indices or any(
        index < 1 or index > 8 for index in operation.source_output_indices
    ):
        raise RuntimeError(f"Invalid source-output mapping for {operation.operation_id}")


def registry_as_dicts() -> dict[str, list[dict[str, object]]]:
    """Return stable stage and operation registries for inspection and serialization."""
    return {
        "stages": [stage.to_dict() for stage in STAGE_REGISTRY],
        "operations": [operation.to_dict() for operation in OPERATION_REGISTRY],
    }
