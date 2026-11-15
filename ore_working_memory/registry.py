"""Stable 30-operation registry for working memory and cognitive control.

Each operation links a functional responsibility to a mathematical sketch, stage output, and
qualified biological-reference hypothesis.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class StageSpec:
    """Describe one processing stage and its primary working-memory representation."""

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
    """Describe one numbered operation, its equation, and its biological-reference interpretation."""

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
        """Return the operation with its input and output names as a dictionary."""
        return asdict(self)


_STAGE_ROWS = (
    ("S1", "Linguistic working-memory buffers",
     "Bounded phonological, lexical, syntactic, semantic, and discourse-specific active traces",
     "Phonological-loop and language-related temporal representations interacting with inferior-parietal storage and prefrontal maintenance"),
    ("S2", "Maintain incomplete syntactic structures",
     "Nested open phrase/clause frames, unresolved argument slots, and filler-gap dependencies",
     "Posterior temporal syntactic representations interacting with inferior frontal and parietal working-memory/retrieval systems"),
    ("S3", "Executive control over sentence interpretations",
     "Conflict estimate, control demand, goal compatibility, top-down bias, and selected focus",
     "Lateral prefrontal control interacting with anterior cingulate conflict monitoring and frontoparietal attention"),
    ("S4", "Resolve competing semantic interpretations",
     "Posterior distribution over interpretations, entropy, margin, selected reading, or abstention",
     "Distributed temporal semantic representations with inferior frontal selection, recurrent competition, and monitoring"),
    ("S5", "Cognitive task switching",
     "Task-set sequence, switch detection, latency estimate, reconfiguration, interference, and resumption state",
     "Frontoparietal control, pre-supplementary motor/cingulo-opercular task-set updating, and basal-ganglia gating as a coarse functional analogy"),
)

_OPERATION_ROWS = (
    ("S1", "Tokenize and encode the linguistic stream", "Create ordered, index-preserving token records from the supplied surface text or token sequence.", "x = (w_1, ..., w_n); n <= N_max", "Auditory/phonological and orthographic inputs activate distributed language representations.", ("text_or_tokens",), ("tokens",)),
    ("S1", "Allocate modality-specific working buffers", "Copy token references into separate phonological, lexical, syntactic, semantic, and discourse channels without conflating their roles.", "B_m = select_m(x); m in {phon, lex, syn, sem, disc}", "Phonological-loop and language representations interact with parietal and frontal maintenance; software channels are not literal brain buffers.", ("tokens",), ("buffers",)),
    ("S1", "Calculate recency and salience activation", "Estimate each buffer item's activation from age and supplied focus cues.", "a_i(t) = s_i * exp(-lambda * age_i)", "Persistent/recurrent population activity is a candidate mechanism for maintaining currently relevant information.", ("buffers", "focus_cues"), ("item_activations",)),
    ("S1", "Apply decay and capacity constraints", "Reduce older activations and retain the strongest items up to the configured buffer capacity.", "B'_m = topK({a_i : a_i >= theta}, K)", "Limited attentional access and interference motivate a bounded-resource approximation; the numeric capacity is an engineering setting.", ("item_activations",), ("bounded_buffers", "evicted_items")),
    ("S1", "Rehearse and refresh active items", "Refresh a small focus set and count refresh events while preserving the original token order.", "a_i' = min(1, a_i + eta * r_i)", "Articulatory rehearsal and recurrent maintenance are functional analogies, not simulated speech motor activity.", ("bounded_buffers",), ("rehearsed_buffers",)),
    ("S1", "Retrieve and bind context cues", "Rank active items by activation and overlap with explicit retrieval cues or supplied prior tokens.", "R_i = alpha*a_i + (1-alpha)*cue_match_i", "Cue-dependent reinstatement is a memory retrieval analogy; only caller-supplied context is used.", ("rehearsed_buffers", "context"), ("retrieval_candidates",)),
    ("S2", "Identify incomplete phrase and clause frames", "Read explicitly supplied partial structures and unresolved syntax outputs, keeping expected and filled slots separate.", "U_f = expected_f \\ filled_f", "Frontal-temporal language interactions maintain expectations across phrases and clauses.", ("partial_structures", "syntax_result"), ("open_frames",)),
    ("S2", "Encode hierarchical stack and dependency slots", "Order open frames by nesting depth and preserve parent-child links and outstanding argument positions.", "Stack = sort(frames, depth); depth(child) > depth(parent)", "Nested linguistic dependencies can require sustained representations across intervening material.", ("open_frames",), ("syntactic_stack",)),
    ("S2", "Maintain unresolved filler-gap dependencies", "Retain only dependencies explicitly marked unresolved or open by upstream syntax evidence.", "D_t = D_(t-1) union D_unresolved", "A maintained filler representation is a functional analogy for long-distance retrieval demands.", ("syntax_result", "context"), ("pending_dependencies",)),
    ("S2", "Score retrieval cues for open slots", "Compare candidate bindings with the slot label, confidence, and available lexical/context overlap.", "q(c,s) = 0.6*confidence + 0.4*slot_match", "Retrieval cues and structural constraints guide access to a previously pending dependency.", ("syntactic_stack", "pending_dependencies", "candidate_bindings"), ("slot_candidates",)),
    ("S2", "Bind supported incoming material", "Fill an open slot only when explicit candidate evidence clears the retrieval threshold.", "bind(s,c) iff q(c,s) >= theta_retrieval", "A supported constituent can satisfy a maintained syntactic expectation; weak evidence remains unresolved.", ("slot_candidates",), ("updated_frames", "resolved_slots")),
    ("S2", "Emit the incomplete-structure state", "Report filled and unfilled slots, unresolved dependencies, and stack depth without guessing missing constituents.", "W_syn = (frames, filled, unresolved, depth)", "Frontal/parietal monitoring is represented as an inspectable state report.", ("updated_frames", "pending_dependencies"), ("incomplete_structure_report",)),
    ("S3", "Generate interpretation hypotheses", "Normalize caller-provided readings or create one explicitly neutral surface-reading fallback.", "H = {h_1, ..., h_k}, k <= K_max", "Multiple semantic/structural populations may remain concurrently active during ambiguity.", ("interpretations", "tokens"), ("hypotheses",)),
    ("S3", "Measure competition and interpretation conflict", "Estimate conflict from normalized hypothesis entropy and the margin between the two strongest preliminary scores.", "C = 0.5*H(p)/log(k) + 0.5*(1 - (p_1-p_2))", "Anterior cingulate and frontoparietal monitoring are commonly used as functional analogies for conflict-sensitive control.", ("hypotheses",), ("conflict",)),
    ("S3", "Allocate executive control demand", "Combine conflict, unresolved syntax load, and explicit goal priority into a bounded control-demand estimate.", "D = clip(0.45*C + 0.35*U + 0.20*G, 0, 1)", "Lateral prefrontal and parietal control resources are represented as a bounded control allocation.", ("conflict", "incomplete_structure_report", "goals"), ("control_demand",)),
    ("S3", "Evaluate goal compatibility", "Compare supplied goal features with each hypothesis or use an explicit per-hypothesis goal score.", "g(h) = cosine(features_h, goal_features)", "Goal representations in prefrontal systems can bias selection among currently active alternatives.", ("hypotheses", "goals"), ("goal_fit",)),
    ("S3", "Apply top-down interpretation bias", "Modulate each hypothesis with its prior and goal compatibility while retaining the unmodified evidence channels.", "b(h) = gain * D * (0.65*g(h) + 0.35*prior_h)", "Top-down control changes selection pressure; it does not establish that a hypothesis is true.", ("control_demand", "goal_fit", "hypotheses"), ("top_down_bias",)),
    ("S3", "Select an active focus without deleting alternatives", "Choose the highest control-biased hypothesis as the current focus while preserving all other candidates.", "focus = argmax_h(base_score_h + b(h))", "Attentional selection can prioritize one representation while competing representations remain available.", ("top_down_bias", "hypotheses"), ("control_focus", "preserved_alternatives")),
    ("S4", "Integrate semantic evidence channels", "Combine lexical, semantic, syntactic, discourse, world, and top-down evidence using fixed normalized weights.", "z_h = .20L + .28S + .20Y + .16D + .16W + b(h)", "Distributed semantic evidence interacts with structural constraints and frontal selection.", ("hypotheses", "top_down_bias"), ("candidate_scores",)),
    ("S4", "Normalize candidate activations", "Convert candidate scores to a stable temperature-scaled posterior distribution.", "p(h) = softmax(z_h / T)", "Population activation competition is approximated by a normalized candidate distribution.", ("candidate_scores",), ("posterior",)),
    ("S4", "Run recurrent lateral competition", "Iteratively mix posterior beliefs with a competition-adjusted softmax while damping large updates.", "p_(t+1) = (1-rho)*p_t + rho*softmax(z/T - gamma*(1-p_t))", "Recurrent excitation/inhibition is used as a generic settling analogy, not a fitted semantic neural circuit.", ("posterior", "candidate_scores"), ("settled_posterior",)),
    ("S4", "Measure uncertainty and posterior margin", "Calculate normalized entropy and the difference between the strongest and second-strongest readings.", "H_n = -sum(p_i log p_i)/log(k); margin = p_1-p_2", "Distributed activity and monitoring signals can reflect competition, but the software entropy is not a direct neural measurement.", ("settled_posterior",), ("entropy", "posterior_margin")),
    ("S4", "Apply confidence and abstention thresholds", "Commit only when both the best posterior and the top-two margin meet configured thresholds.", "select iff p_1 >= tau_c and (p_1-p_2) >= tau_m", "Controlled selection can be withheld when evidence remains weak or competing readings remain close.", ("settled_posterior", "entropy", "posterior_margin"), ("selection_decision",)),
    ("S4", "Construct the interpretation report", "Return the chosen reading or abstention, ranked alternatives, evidence, and convergence diagnostics.", "R = (h*, p, H_n, margin, evidence, abstain)", "Monitoring and language systems jointly inform a final report; no literal neural readout is claimed.", ("selection_decision", "settled_posterior"), ("semantic_resolution",)),
    ("S5", "Encode the current task set", "Read an explicit task sequence or construct a one-task default from the supplied context.", "T = (t_1, ..., t_n); t_1 = context.current_task if given", "Prefrontal task-set representations maintain the currently relevant rule or goal.", ("task_sequence", "context"), ("task_sets",)),
    ("S5", "Detect task-set transitions", "Compare adjacent task labels and any supplied previous task to identify switch boundaries.", "switch_i = 1[t_i != t_(i-1)]", "Task-set updating recruits frontoparietal and cingulo-opercular control systems.", ("task_sets", "context.previous_task"), ("switch_events",)),
    ("S5", "Calculate switch latency and control cost", "Estimate switch cost from a baseline, task-family similarity, task inertia, and cue strength.", "cost = C_0*(1 + inertia*similarity)*(1-cue_strength)", "Switch costs are behavioral constructs in software; simulated milliseconds are not measured reaction times.", ("switch_events", "context.task_cue_strength"), ("switch_costs",)),
    ("S5", "Reconfigure task-specific control weights", "Assign transparent attentional weights to the current task's relevant buffer and reasoning channels.", "w_task = normalize(task_profile)", "Frontoparietal control reconfigures processing priorities according to task demands.", ("task_sets", "switch_events"), ("task_control_weights",)),
    ("S5", "Manage interference and suspended task state", "Estimate cross-task interference and record suspend/resume transitions without destroying prior task records.", "I = task_similarity * inertia * control_demand", "Task-set shielding and context restoration are functional analogies for interference control.", ("task_control_weights", "switch_events", "context"), ("interference_report", "suspended_tasks")),
    ("S5", "Emit task history and active control state", "Summarize the task sequence, switches, latency estimates, interference, and current task profile.", "W_task = (T, switches, sum(cost), I, active_profile)", "A distributed control state is exposed as a software-level inspection record.", ("switch_events", "switch_costs", "task_control_weights", "interference_report"), ("task_switching_report",)),
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
        operation_id=f"WMC-{index:03d}",
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

if len(STAGE_REGISTRY) != 5 or len(OPERATION_REGISTRY) != 30:
    raise RuntimeError("Working-memory registry must contain five stages and 30 operations")
if sum(stage.operation_count for stage in STAGE_REGISTRY) != len(OPERATION_REGISTRY):
    raise RuntimeError("Working-memory stage counts must sum to 30")
if len({item.operation_id for item in OPERATION_REGISTRY}) != len(OPERATION_REGISTRY):
    raise RuntimeError("Working-memory operation identifiers must be unique")


def registry_as_dicts() -> dict[str, list[dict[str, object]]]:
    """Return stable stage and operation records for APIs, tests, and documentation."""
    return {
        "stages": [stage.to_dict() for stage in STAGE_REGISTRY],
        "operations": [operation.to_dict() for operation in OPERATION_REGISTRY],
    }
