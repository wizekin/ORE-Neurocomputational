"""Canonical 40-operation registry for working memory and cognitive control.

The operation IDs and order follow Brain Language Algorithm.md (WMCC-001..WMCC-040).
Mathematical forms are implementable abstractions; biological references are hypotheses,
not claims that neurons literally execute these software operations.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class StageSpec:
    """Describe one processing stage, its output, and candidate biological systems."""

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
    """Describe one canonical operation, including its function, mathematics, and biological analogy."""

    operation_id: str
    stage_id: str
    stage_name: str
    operation_name: str
    functional_responsibility: str
    mathematical_form: str
    biological_reference: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return the operation definition as a plain dictionary."""
        return asdict(self)


_STAGE_ROWS = (
    ("S1", "Linguistic working-memory buffers",
     "Updated phonological, lexical-semantic, syntactic, event/discourse, and executive/goal buffers; chunks and retrieval candidates",
     "Distributed language and working-memory networks, including temporal language representations, inferior-parietal maintenance, phonological-loop processes, and prefrontal attention; these are functional correspondences, not isolated buffer locations"),
    ("S2", "Maintaining incomplete syntactic structures",
     "Completed and open constituents, grammatical expectations, nested structures, unresolved dependencies, and dependency-linked memory items",
     "Posterior temporal and inferior frontal language regions interacting with parietal retrieval and verbal working-memory systems; exact division is debated"),
    ("S3", "Executive control over sentence interpretations",
     "Active goals, processing-quality monitor, conflict and demand signals, attention allocation, controlled candidate activity, and coordination messages",
     "Lateral prefrontal and frontoparietal control networks interacting with anterior cingulate/conflict-monitoring and language-selective inferior frontal systems"),
    ("S4", "Resolving competing semantic interpretations",
     "Ranked semantic hypotheses with lexical, grammatical, contextual, knowledge and pragmatic evidence, posterior probabilities, revisions, or abstention",
     "Distributed temporal semantic representations interacting with inferior frontal selection, recurrent competition, and discourse/context systems"),
    ("S5", "Cognitive task switching",
     "Active task, switch decisions, preserved checkpoints, reconfigured priorities, interference, task completion, resumption, learned transition costs, and integrated completion state",
     "Frontoparietal task control, cingulo-opercular/presupplementary motor task updating, and basal-ganglia gating as coarse functional analogies"),
)

# Rows: stage, exact canonical name, functional responsibility, mathematical form,
# qualified biological-reference note, inputs, outputs.
_OPERATION_ROWS = (
    ("S1", "Initialize cognitive processing environment",
     "Create the shared cognitive state, empty buffers, syntax/dependency stores, candidate store, control state, task manager, and cross-system references.",
     "Omega_0 = {W_0, U_0, P_0, C_0, T_0}; initialize only declared neutral values.",
     "Coordinated baseline activity across distributed working-memory, language, control, and task-set networks; the software state is not a literal neural snapshot.",
     ("request", "configuration"), ("cognitive_state", "subsystem_stores")),
    ("S1", "Receive linguistic information",
     "Accept the supplied text or token sequence together with upstream lexical, semantic, syntactic, and discourse evidence without silently inventing missing channels.",
     "I_t = {surface, supplied_features, source, time}; preserve the source and representation type.",
     "Language input arrives through auditory or visual pathways and is rapidly coupled to temporal language representations; modality is preserved in the input contract.",
     ("request.text", "request.tokens", "upstream_evidence"), ("received_input", "provenance")),
    ("S1", "Encode incoming linguistic representations",
     "Create ordered token records and feature references while preserving caller-supplied token boundaries and the original sequence.",
     "X_t = [x_1,...,x_n]; token_index_i = i; encode only observed or supplied features.",
     "Orthographic or phonological form activates distributed word-form and phonological representations; token objects are engineering representations.",
     ("received_input", "language"), ("tokens", "token_records")),
    ("S1", "Distribute representations across working-memory buffers",
     "Route references to phonological, lexical-semantic, syntactic, event/discourse, and executive/goal buffers while retaining cross-references.",
     "W_t = {B_phon, B_lexsem, B_syn, B_event, B_exec}; B_i <- B_i + f_i(X_t).",
     "Phonological maintenance, temporal semantic stores, inferior-parietal and prefrontal processes interact; the five buffers are functional software channels, not five anatomical containers.",
     ("token_records", "upstream_evidence", "goals"), ("buffers", "cross_references")),
    ("S1", "Organize and chunk stored information",
     "Group explicitly related or contiguous surface units into inspectable chunks while preserving token references and recording whether grouping came from supplied structure or a surface heuristic.",
     "sim(x_i,x_j) = (x_i dot x_j)/(||x_i|| ||x_j||); chunk = Combine(x_i,...,x_j) only when a declared grouping rule passes.",
     "Chunking may reduce the number of separately maintained units and can involve temporal, parietal, and frontal interactions; the specific chunk boundaries are model assumptions.",
     ("buffers", "explicit_chunks", "token_records"), ("chunks", "chunk_links")),
    ("S1", "Maintain activation",
     "Update item accessibility using baseline activation, retrieval-cue support, rehearsal/refresh, time decay, and similarity-based interference.",
     "A_i(t)=B_i+sum_j W_j S_ji+sum_r rho_r R_ir-lambda_i Delta_t-mu sum_(k!=i) sim(i,k)A_k.",
     "Persistent/recurrent population activity, synaptic mechanisms, and attentional refreshing are candidate mechanisms; the scalar activation is a computational proxy.",
     ("buffers", "chunks", "focus_cues", "elapsed_steps"), ("item_activations", "interference_estimates")),
    ("S1", "Regulate memory capacity and attention",
     "Prioritize goal-relevant, active items under a configured resource budget and retain explicit records of deprioritized or evicted items.",
     "alpha_i = softmax(A_i/tau); sum_i r_i <= K; retain top-K under the declared policy.",
     "Attentional selection and limited-access maintenance are supported functional analogies; K is an engineering constraint, not a universal human word capacity.",
     ("item_activations", "goals", "buffer_capacity"), ("bounded_buffers", "attention_weights", "evicted_items")),
    ("S1", "Perform content-addressable retrieval",
     "Rank active-memory and supplied contextual candidates by overlap with the retrieval request; return ranked candidates and refresh items that are reused.",
     "M* = argmax_i Match(Q,M_i); P(M_i|Q)=softmax(beta Match(Q,M_i)).",
     "Cue-dependent reinstatement and associative retrieval are functional analogies; retrieval uses only caller-supplied cues and retained representations.",
     ("bounded_buffers", "retrieval_query", "context"), ("retrieval_candidates", "refreshed_items")),
    ("S2", "Detect incomplete syntactic structures",
     "Identify supplied open constituents, missing obligatory elements, unresolved slots, and incomplete structures from upstream syntax.",
     "G_s=(V,E); U_t={v in V : completion(v)=0}.",
     "Frontal-temporal language interactions maintain expectations and unfinished constituents during incremental comprehension.",
     ("syntax_result", "partial_structures"), ("open_structures", "unresolved_slots")),
    ("S2", "Generate grammatical expectations",
     "Generate explicit, source-labelled expectations for possible grammatical continuations and score supplied candidate fillers against open slots.",
     "P(X_(t+1)|X_(1:t)); expectation_priority = normalize(score_i).",
     "Predictive syntactic processing and retrieval cues can pre-activate likely continuations; generated expectations are grammar/resource-conditioned hypotheses.",
     ("open_structures", "grammar_rules", "syntactic_candidates"), ("grammatical_expectations", "slot_candidates")),
    ("S2", "Maintain unresolved dependencies",
     "Preserve explicitly unresolved filler-gap, argument, and other long-distance dependency records until evidence resolves them.",
     "D_i=(a_i,b_i,r_i); A(D_i)=A_0 exp(-lambda d_i); D(t+1)=D(t)+Delta D.",
     "Long-distance dependencies create retrieval demands involving posterior temporal, inferior frontal, and working-memory systems.",
     ("syntax_result", "context", "dependency_records"), ("pending_dependencies", "dependency_links")),
    ("S2", "Maintain nested syntactic structures",
     "Represent open constituents and embedded clauses in a depth-aware stack/agenda while retaining parent-child relations.",
     "S_(t+1)=Push(S_t,N) on open; Pop(S_t) on completion; alternatively maintain a ranked open-node graph.",
     "Hierarchical sentence representations recruit fronto-temporal language circuitry; stack and agenda structures are computational alternatives.",
     ("open_structures", "pending_dependencies"), ("syntactic_stack", "parent_child_links")),
    ("S2", "Update incomplete structures",
     "Match incoming elements to open slots, bind only adequately supported candidates, and retain unmatched slots for later evidence.",
     "delta_t=X_t-Xhat_t; bind(s,c) only if P(c|s,evidence) >= theta; otherwise remain open.",
     "Incremental parsing updates constituent and dependency representations as each word arrives; thresholded binding is an explicit software policy.",
     ("syntactic_stack", "slot_candidates", "incoming_tokens"), ("updated_structures", "resolved_slots")),
    ("S2", "Maintain dependencies through working-memory updates",
     "Protect or reactivate buffer items referenced by unresolved structures and keep explicit links between dependency records and memory entries.",
     "A_D(t)=w_a A_a(t)+w_b A_b(t)+w_r R; priority(D_i)=softmax(A_Di).",
     "Cue-based reactivation and sustained accessibility help support dependency completion; protection is represented as priority metadata, not permanent storage.",
     ("updated_structures", "pending_dependencies", "buffers"), ("dependency_memory_links", "incomplete_structure_report")),
    ("S3", "Maintain cognitive goals",
     "Retain the current comprehension objective, task constraints, and goal features; compare candidate operations with the active goal.",
     "G_t=[g_1,...,g_n]; relevance(O_i,G_t)=cosine(O_i,G_t).",
     "Prefrontal representations can maintain task goals and bias ongoing language processing; the goal vector is a software abstraction.",
     ("request.goals", "context.goal", "candidate_features"), ("active_goal", "goal_fits")),
    ("S3", "Monitor sentence-processing progress",
     "Inspect memory availability, syntactic completeness, semantic consistency, reference/dependency status, and interpretation confidence to estimate processing quality.",
     "Z_t=[S_c,M_c,D_c,I_c,R_c]; Q_t=sum_k w_k Z_tk.",
     "Monitoring likely involves interactions among control, language, and conflict-sensitive networks; Q is an inspectable composite estimate.",
     ("buffers", "incomplete_structure_report", "candidates", "context"), ("processing_status", "processing_quality")),
    ("S3", "Detect competing processing demands",
     "Detect candidate competition and incompatible active requirements; estimate whether the conflict warrants additional control.",
     "Delta A=|A_1-A_2|; H(P)=-sum_i P_i log P_i; T_c=w_1 H+w_2 U+w_3 D.",
     "Anterior cingulate and frontoparietal monitoring are candidate correlates of conflict-sensitive control; thresholds are not biological constants.",
     ("processing_status", "candidate_activations", "unresolved_structures"), ("conflict", "control_trigger")),
    ("S3", "Allocate attention",
     "Allocate bounded processing gain to information that is relevant and diagnostic for the active goal and current uncertainty.",
     "A_i'=A_i(1+gamma_i); gamma_i=f(goal,relevance,uncertainty,control).",
     "Frontoparietal attention and gain modulation provide a broad functional analogy; software weights are not measured neural gain.",
     ("active_goal", "conflict", "buffers", "evidence_reliability"), ("attention_allocation", "attended_items")),
    ("S3", "Regulate competing representations",
     "Increase supported task-relevant representations and reduce the relative influence of incompatible or weakly supported competitors without deleting them.",
     "A_i(t+1)=A_i(t)+E_i+G_i-sum_(j!=i) beta_ij A_j(t).",
     "Biased competition and inhibitory interactions are plausible network-level analogies; explicit scalar inhibition is not a literal circuit claim.",
     ("attention_allocation", "candidate_evidence", "active_goal"), ("regulated_activations", "top_down_bias")),
    ("S3", "Determine interpretation acceptance",
     "Decide whether the current interpretation is sufficiently supported to proceed provisionally or whether uncertainty requires continued evaluation.",
     "accept iff P_max>theta_accept and P_max-P_second>theta_margin and no blocking conflict.",
     "Control can permit provisional selection or continued processing; the software thresholds are configurable decision policies.",
     ("regulated_activations", "conflict", "posterior"), ("control_decision", "provisional_focus")),
    ("S3", "Coordinate processing subsystems",
     "Exchange structured requests and state updates among memory, syntax, executive control, semantic resolution, and task management without allowing one module to overwrite another's state.",
     "W_(t+1)=F(W_t,I_t,C_t,M); M is a directed subsystem influence/communication matrix.",
     "Language and domain-general control networks interact recurrently; the adjacency matrix is a software dependency/communication abstraction.",
     ("buffers", "incomplete_structures", "control", "resolution", "task"), ("coordination_messages", "integrated_stage_state")),
    ("S4", "Identify semantic ambiguity",
     "Identify lexical, attachment, thematic-role, literal/pragmatic, and proposition-level ambiguity from supplied alternatives or ambiguity annotations.",
     "A(x)={m_1,...,m_n}; H(x)=-sum_i P(m_i|x) log P(m_i|x).",
     "Distributed semantic representations can co-activate for ambiguous expressions; ambiguity is identified from evidence rather than assumed for every sentence.",
     ("tokens", "interpretations", "syntax_result", "context"), ("ambiguity_sources", "ambiguity_entropy")),
    ("S4", "Generate alternative interpretations",
     "Construct distinct candidate readings from supplied hypotheses and available lexical/syntactic evidence; preserve source attribution and unknowns.",
     "P_candidates = Generate(L,S,C); N_candidates <= N_lexical x N_structural before pruning.",
     "Multiple semantic/structural populations may remain concurrently active; candidate enumeration is a software hypothesis generator.",
     ("ambiguity_sources", "interpretations", "tokens"), ("candidate_interpretations", "candidates")),
    ("S4", "Retrieve supporting semantic knowledge",
     "Retrieve explicit caller-supplied concept features and relation evidence relevant to each candidate; do not manufacture world facts.",
     "K(P_i)=sum_r w_r Match(P_i,r); knowledge source must be tagged.",
     "Temporal semantic memory and distributed conceptual associations provide a broad analogy; this prototype uses supplied knowledge only.",
     ("candidates", "context.semantic_knowledge", "knowledge_graph"), ("knowledge_support", "knowledge_scores")),
    ("S4", "Evaluate grammatical compatibility",
     "Evaluate each interpretation against supplied syntactic structure, grammatical roles, and explicitly available grammar evidence; do not reject unusual but grammatical events.",
     "G_i=GrammarMatch(P_i,S), normalized to [0,1] when calibrated evidence exists.",
     "Frontotemporal language mechanisms support syntactic unification and role assignment; compatibility scores remain software estimates.",
     ("candidates", "syntax_result", "grammatical_expectations"), ("grammar_compatibility", "syntactic_support")),
    ("S4", "Evaluate contextual compatibility",
     "Compare candidates with caller-provided discourse context, topic, speaker/situation information, and contextual features while preserving absent evidence as unavailable.",
     "C_i=cosine(P_i,D); alternatively P(P_i|D) from a calibrated context model.",
     "Temporal, parietal, and prefrontal context systems may contribute; cosine similarity is only a convenient computational proxy.",
     ("candidates", "context", "discourse_state"), ("contextual_compatibility", "context_support")),
    ("S4", "Calculate the overall evidence supporting each interpretation",
     "Integrate separately labelled lexical, semantic, grammatical, contextual, world-knowledge, pragmatic, and prior evidence into a comparable candidate score.",
     "S_i=w_g G_i+w_l L_i+w_c C_i+w_k K_i+w_p P_i; posterior may use Bayes' rule.",
     "Distributed semantic evidence is combined with structural constraints and contextual/pragmatic support; fixed weights are engineering parameters.",
     ("knowledge_support", "grammar_compatibility", "contextual_compatibility", "candidates"), ("integrated_scores", "evidence_ledger")),
    ("S4", "Allow competing interpretations to interact",
     "Update candidate activations through bounded recurrent competition while preserving all candidates and normalizing their posterior probabilities.",
     "A_i(t+1)=A_i(t)+S_i-beta sum_(j!=i)A_j(t); p=softmax(S/tau).",
     "Recurrent excitation/inhibition is a generic settling analogy, not a fitted semantic neural circuit.",
     ("integrated_scores", "candidate_activations"), ("posterior", "competition_diagnostics")),
    ("S4", "Select an interpretation or preserve multiple possibilities",
     "Select only when confidence and margin criteria pass; otherwise abstain and retain ranked plausible alternatives.",
     "P*=argmax_i P_i; select iff P(P*)>theta and P(P*)-P(P_2)>delta.",
     "Selection can be withheld while multiple candidate representations remain active; thresholds are application-level policies.",
     ("posterior", "ambiguity_entropy", "conflict"), ("selection", "ranked_alternatives", "abstention")),
    ("S4", "Revise an interpretation when new information arrives",
     "Update previously maintained candidates with newly supplied evidence, recompute their posterior, and notify control when the preferred reading changes.",
     "P(P_i|E_old,E_new) proportional to P(E_new|P_i,E_old) P(P_i|E_old).",
     "Incremental reanalysis and prediction-error-driven revision are broad functional analogies; revision evidence must be explicit.",
     ("previous_candidates", "revision_evidence", "posterior"), ("revised_candidates", "revision_report")),
    ("S5", "Identify the currently active cognitive task",
     "Maintain the current task label, objective, rules, required data, and attentional configuration.",
     "T_t=[goal_t,rules_t,data_t,attention_t]; T_t belongs to the declared task-state set.",
     "Prefrontal task-set representations maintain currently relevant rules and goals; task labels are software abstractions.",
     ("task_sequence", "context.current_task"), ("active_task", "task_set")),
    ("S5", "Detect when a task switch is necessary",
     "Compare incoming demands with the active task and trigger a switch only when urgency, conflict, and goal relevance justify it.",
     "D_switch=w_1 N+w_2 C+w_3 U+w_4 G; switch iff D_switch>theta_s.",
     "Task switching recruits frontoparietal and cingulo-opercular updating; switch thresholds are model parameters.",
     ("active_task", "incoming_demand", "goals", "conflict"), ("switch_events", "switch_costs")),
    ("S5", "Preserve the current task state before switching",
     "Checkpoint the information required to resume a suspended task, including token position, working-memory references, open syntax, candidates, and goal.",
     "K_t={W_t,S_t,P_t,G_t,R_t}; K_(t+1)=f(K_t,Delta_t).",
     "Task-state maintenance and gating are candidate analogies; the software checkpoint is not a digital copy of a brain state.",
     ("active_task", "buffers", "incomplete_structures", "candidates"), ("suspended_checkpoints", "preserved_tasks")),
    ("S5", "Activate the new cognitive task",
     "Select the next task and reconfigure attention and processing-channel weights for its declared task family.",
     "A(T_i)=w_1 R_i+w_2 U_i+w_3 C_i; T*=argmax_i A(T_i).",
     "Prefrontal and frontoparietal control can reconfigure task priorities; the software profile is an explicit control policy.",
     ("switch_events", "task_sets", "attention_allocation"), ("current_task", "task_control_weights")),
    ("S5", "Reduce interference from the previous task",
     "Estimate incompatible carry-over and reduce its activation/priority while preserving all information needed for later retrieval.",
     "A_i'=A_i-eta I_i; suppression is reversible and is not deletion.",
     "Inhibitory control and task-set shielding are functional analogies; no physiological inhibition is measured here.",
     ("suspended_checkpoints", "current_task", "control_demand"), ("interference_report", "suppressed_task_items")),
    ("S5", "Monitor the new task and determine when to return",
     "Monitor explicit task completion and pending requirements; determine whether the active task is complete or whether the original task remains blocked.",
     "C(T_i)=1 when declared completion conditions pass; else C(T_i)=0/unknown.",
     "Control-monitoring systems may track task progress; software completion is based on supplied conditions and subsystem diagnostics.",
     ("current_task", "task_outputs", "pending_requirements"), ("task_completion_status", "return_decision")),
    ("S5", "Resume suspended tasks",
     "Restore the highest-priority suspended checkpoint when the auxiliary task has completed or when the previous task again becomes relevant.",
     "T_next=argmax_i [R_i+omega P_i]; restore only a matching preserved checkpoint.",
     "Reactivation/gating of a prior task set is a coarse functional analogy; the simulator does not reproduce hippocampal episodic reinstatement.",
     ("suspended_checkpoints", "task_completion_status", "task_priorities"), ("resumed_tasks", "restored_task_state")),
    ("S5", "Learn from task-switching history",
     "Record switch triggers and outcomes and update a transparent moving estimate of transition costs using only explicitly available observations.",
     "C_ij(t+1)=(1-alpha)C_ij(t)+alpha c_ij(t); V(T_j)=U(T_j)-C_ij.",
     "Experience-dependent control adaptation is a broad learning analogy; this run-local estimator is not synaptic plasticity.",
     ("switch_events", "observed_transition_costs", "task_history"), ("task_transition_history", "learned_transition_costs")),
    ("S5", "Integrate the cognitive state",
     "Synchronize memory, syntax, control, semantic, and task reports into one inspectable state while retaining subsystem ownership and evidence provenance.",
     "Omega_t=[W_t,U_t,P_t,C_t,T_t]; validate referential integrity across the components.",
     "Distributed systems need coordinated integration; a shared state object is a software architecture, not a single neural workspace claim.",
     ("buffers", "incomplete_structure_report", "executive_control", "semantic_resolution", "task_state"), ("integrated_cognitive_state", "final_reports")),
    ("S5", "Evaluate whether processing is complete",
     "Evaluate sentence-end, required syntactic completion, sufficient semantic interpretation, reference/dependency status, and unresolved conflicts; return complete, partial, or pending.",
     "C_overall = S_c AND M_c AND R_c AND no_blocking_conflict; otherwise retain pending state.",
     "Comprehension can remain partial or revisable; this completion gate is an application-level criterion, not a neural end marker.",
     ("integrated_cognitive_state", "incomplete_structure_report", "semantic_resolution", "context"), ("processing_completion", "final_status")),
)


STAGE_REGISTRY = tuple(
    StageSpec(
        stage_id=stage[0],
        stage_name=stage[1],
        operation_count=sum(1 for operation in _OPERATION_ROWS if operation[0] == stage[0]),
        primary_output=stage[2],
        biological_systems=stage[3],
    )
    for stage in _STAGE_ROWS
)

OPERATION_REGISTRY = tuple(
    OperationSpec(
        operation_id=f"WMCC-{index:03d}",
        stage_id=row[0],
        stage_name=next(stage[1] for stage in _STAGE_ROWS if stage[0] == row[0]),
        operation_name=row[1],
        functional_responsibility=row[2],
        mathematical_form=row[3],
        biological_reference=row[4],
        inputs=tuple(row[5]),
        outputs=tuple(row[6]),
    )
    for index, row in enumerate(_OPERATION_ROWS, start=1)
)

if len(STAGE_REGISTRY) != 5 or len(OPERATION_REGISTRY) != 40:
    raise RuntimeError("Working-memory registry must contain five stages and exactly 40 operations")
if tuple(stage.operation_count for stage in STAGE_REGISTRY) != (8, 6, 7, 9, 10):
    raise RuntimeError("Working-memory stage counts must be 8, 6, 7, 9, and 10")
if sum(stage.operation_count for stage in STAGE_REGISTRY) != len(OPERATION_REGISTRY):
    raise RuntimeError("Working-memory stage counts must sum to 40")
if tuple(item.operation_id for item in OPERATION_REGISTRY) != tuple(
    f"WMCC-{index:03d}" for index in range(1, 41)
):
    raise RuntimeError("Working-memory operation identifiers must be ordered WMCC-001..WMCC-040")
if len({item.operation_id for item in OPERATION_REGISTRY}) != len(OPERATION_REGISTRY):
    raise RuntimeError("Working-memory operation identifiers must be unique")


def registry_as_dicts() -> dict[str, list[dict[str, object]]]:
    """Return stable stage and operation records for APIs, tests, and documentation."""
    return {
        "stages": [stage.to_dict() for stage in STAGE_REGISTRY],
        "operations": [operation.to_dict() for operation in OPERATION_REGISTRY],
    }
