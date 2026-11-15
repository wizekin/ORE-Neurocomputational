"""Canonical seven-stage, 38-operation registry for TDPIA sentence comprehension."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class StageSpec:
    """Describe one ordered comprehension stage and the operations assigned to it."""
    stage_id: str
    name: str
    first_operation: int
    last_operation: int
    purpose: str
    biological_basis: str

    def to_dict(self) -> dict[str, object]:
        """Convert the stage contract into a serialization-friendly dictionary."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class OperationSpec:
    """Specify one operation's functional purpose, equation, biology, and Python entry point."""
    operation_id: str
    number: int
    stage_id: str
    stage_name: str
    name: str
    method_name: str
    purpose: str
    mathematical_form: str
    biological_basis: str

    def to_dict(self) -> dict[str, object]:
        """Convert the operation contract into a serialization-friendly dictionary."""
        return asdict(self)


STAGE_REGISTRY = (
    StageSpec("A", "Sentence-level prediction", 1, 3,
              "Reconstruct the active context and anticipate sentence continuations.",
              "Recurrent language activity and top-down activation of situation representations."),
    StageSpec("B", "Syntactic prediction", 4, 8,
              "Predict grammatical categories, dependencies, agreement, and alternative parses.",
              "Distributed temporal-frontal syntactic representations and recurrent competition."),
    StageSpec("C", "Semantic prediction", 9, 12,
              "Activate concepts and predict semantic categories, roles, and feature patterns.",
              "Distributed semantic-memory populations and feature-based concept activation."),
    StageSpec("D", "World-knowledge prediction", 13, 16,
              "Retrieve event knowledge and assess plausibility while weighting prior reliability.",
              "Interactions among semantic memory, episodic retrieval, and contextual control."),
    StageSpec("E", "Word-level surprisal", 17, 20,
              "Recognize the input and quantify word probability, surprisal, and prior uncertainty.",
              "Bottom-up lexical activation interacting with top-down graded expectations."),
    StageSpec("F", "N400 and P600 modeling", 21, 28,
              "Compare expected and actual meaning, estimate N400-like mismatch, and model reanalysis/P600-like activity.",
              "Distributed semantic access, prediction-error signaling, conflict monitoring, and revision."),
    StageSpec("G", "Incremental sentence comprehension", 29, 38,
              "Integrate each word, update hypotheses and working context, complete the sentence, and adapt future predictions.",
              "Recurrent language networks, working-memory maintenance, control, and experience-dependent plasticity."),
)

_OPERATION_ROWS = (
    ("A", "Reconstruct current linguistic context", "op_01_reconstruct_context",
     "Retrieve recent words and discourse cues, then construct a recency-weighted active context.",
     "c_i = decay^distance; C_t = sum_i c_i X_i / sum_i c_i",
     "Recurrent activity and selective attention maintain task-relevant lexical and conceptual information."),
    ("A", "Infer the sentence-level situation", "op_02_infer_situation",
     "Infer likely event or situation hypotheses from the context while retaining alternatives.",
     "P(z|C) = softmax(score(z,C)/T)",
     "Distributed event and situation representations are activated by contextual cues."),
    ("A", "Predict possible sentence continuations", "op_03_predict_continuations",
     "Generate candidate phrase continuations and estimate their relative plausibility.",
     "P(F_i|C) = softmax(s_i); H(F|C) = -sum_i p_i log2(p_i)",
     "Top-down cortical activation preactivates graded structural and conceptual continuations."),
    ("B", "Examine the current syntactic structure", "op_04_examine_syntax",
     "Inspect processed categories, incomplete constituents, and the current grammatical position.",
     "G_t = parse(tags_1:t); open(G_t) = unresolved grammatical slots",
     "Recurrent temporal-frontal language activity maintains partial syntactic structures."),
    ("B", "Predict the next syntactic category", "op_05_predict_syntactic_category",
     "Estimate probabilities over the next part-of-speech category given the current parse.",
     "P(c_next|G_t,C_t) = softmax(s_c/T)",
     "Graded activation of grammatical categories reflects structural expectations."),
    ("B", "Predict grammatical dependencies", "op_06_predict_dependencies",
     "Maintain expected subjects, objects, relative-clause links, and other unresolved dependencies.",
     "P(d_i|G_t,C_t); L_syntax = -log P(G_t|C_t)",
     "Distributed neural populations maintain nonlocal grammatical expectations across intervening words."),
    ("B", "Predict agreement and structural constraints", "op_07_predict_agreement",
     "Carry number, person, tense, and agreement requirements forward to later input.",
     "S_agree(x) = sum_i omega_i Match(x,f_i)",
     "Recurrent feature representations allow earlier grammatical information to constrain later forms."),
    ("B", "Maintain competing syntactic predictions", "op_08_maintain_competing_parses",
     "Preserve several possible parses and normalize their competing support rather than committing too early.",
     "P(G_i|C) = softmax(s_i); retain top-k alternatives",
     "Competitive population activity can maintain several partially active structural representations."),
    ("C", "Retrieve activated semantic concepts", "op_09_retrieve_semantic_concepts",
     "Retrieve contextually relevant concepts and their semantic features.",
     "A_j = sum_i W_ij A_i; Sim(u,v) = cosine(u,v)",
     "Distributed semantic-memory networks activate learned concept associations."),
    ("C", "Predict upcoming semantic categories", "op_10_predict_semantic_categories",
     "Estimate likely conceptual categories such as food, instrument, person, or location.",
     "P(k_next|M_t,C_t) = softmax(s_k/T)",
     "Contextual preactivation biases semantic-category populations before the next word arrives."),
    ("C", "Predict thematic roles", "op_11_predict_thematic_roles",
     "Estimate agent, patient/theme, instrument, location, and other event-participant roles.",
     "P(role_i=x|event,C_t); role scores combine syntax and feature compatibility",
     "Predicate and participant representations interact to establish event-role bindings."),
    ("C", "Predict semantic features", "op_12_predict_semantic_features",
     "Construct an expected semantic feature vector from weighted candidate concepts.",
     "V_hat = sum_i P(c_i|C_t) V_i",
     "Top-down connections preactivate graded conceptual features in distributed representations."),
    ("D", "Retrieve relevant world knowledge", "op_13_retrieve_world_knowledge",
     "Retrieve relevant entity, action, causal, and functional knowledge for active concepts.",
     "A_K = sum_i W_iK A_i; P(K_j|C_t)",
     "Semantic and episodic memory systems supply learned facts and associations."),
    ("D", "Retrieve event schemas and scripts", "op_14_retrieve_event_schemas",
     "Activate familiar event sequences and estimate likely subsequent event steps.",
     "P(E_t+1|E_t,K)",
     "Distributed event knowledge and memory retrieval provide sequential situation expectations."),
    ("D", "Evaluate real-world plausibility", "op_15_evaluate_plausibility",
     "Score whether the event and role assignments fit retrieved world knowledge.",
     "P(E|C,K) proportional to P(C|E) P(E|K)",
     "Conceptual and event knowledge constrain interpretation without dictating it."),
    ("D", "Estimate world-knowledge reliability", "op_16_estimate_world_reliability",
     "Reduce the influence of literal world priors in fictional, hypothetical, or weak-evidence contexts.",
     "P(E|C,K) proportional to P(C|E)^alpha P(E|K)^beta",
     "Precision-weighted control adjusts the influence of priors relative to linguistic evidence."),
    ("E", "Receive and recognize the incoming word", "op_17_recognize_word",
     "Represent the recognized lexical item and its orthographic/phonological confidence.",
     "P(w_t|x_t)",
     "Visual or auditory input pathways activate lexical representations and candidate words."),
    ("E", "Calculate the probability of the actual word", "op_18_calculate_word_probability",
     "Combine syntactic, semantic, world-knowledge, and learned association evidence for the actual word.",
     "P(w_t|C) = exp(s_w) / sum_j exp(s_j)",
     "Bottom-up lexical activation is integrated with top-down contextual expectations."),
    ("E", "Calculate word-level surprisal", "op_19_calculate_surprisal",
     "Quantify the information carried by the word under its pre-input contextual probability.",
     "I(w_t) = -log2 P(w_t|C_t-1)",
     "Neural systems represent graded mismatch; logarithmic surprisal is a computational descriptor."),
    ("E", "Calculate prediction uncertainty", "op_20_calculate_uncertainty",
     "Measure uncertainty across all candidate words before the actual word arrives.",
     "H(W|C) = -sum_w P(w|C) log2 P(w|C)",
     "Distributed competition and population dispersion provide proxies for uncertainty."),
    ("F", "Retrieve predicted semantic features", "op_21_retrieve_predicted_features",
     "Retrieve the expected feature vector generated by semantic prediction.",
     "V_hat_t = sum_i P(c_i|C_t) V_i",
     "Top-down signals preactivate distributed semantic features."),
    ("F", "Extract actual word semantic features", "op_22_extract_actual_features",
     "Retrieve the lexical-semantic feature representation associated with the recognized word.",
     "V_t = [f_1, f_2, ..., f_n]",
     "Lexical activation spreads to associated conceptual features across semantic networks."),
    ("F", "Calculate lexical-semantic prediction error", "op_23_calculate_semantic_error",
     "Compare actual and expected feature vectors using residual and precision-weighted energy.",
     "r_t = V_t - V_hat_t; E = r_t^T Pi_t r_t",
     "Predictive-coding accounts model residual activity when actual semantic input differs from expectation."),
    ("F", "Estimate N400 amplitude", "op_24_estimate_n400",
     "Map semantic mismatch and contextual access difficulty to an explicitly approximate negative ERP estimate.",
     "N400_t = -k_N * mismatch_score (microvolt-like scale)",
     "N400 is an EEG component associated with lexical-semantic processing; the model is not a direct neural measurement."),
    ("F", "Compare syntactic and semantic interpretations", "op_25_compare_interpretations",
     "Quantify disagreement among syntactic role expectations, semantic role evidence, and world knowledge.",
     "D_JS(P,Q) = 0.5 KL(P||M) + 0.5 KL(Q||M), M=(P+Q)/2",
     "Distributed language representations interact with monitoring and control networks."),
    ("F", "Detect the need for structural reanalysis", "op_26_detect_reanalysis",
     "Use structural conflict, agreement violations, and parse uncertainty to decide whether to reanalyse.",
     "C_t = a D_syntax + b D_semantics + c D_thematic; reanalyse if C_t > theta",
     "Conflict-monitoring and attentional control can increase processing of problematic sentence regions."),
    ("F", "Reconstruct alternative syntactic interpretations", "op_27_reconstruct_alternatives",
     "Regenerate and rescore alternative parses when the current interpretation is insufficient.",
     "P(G_i|W_t,C_t-1) proportional to P(W_t|G_i,C_t-1)P(G_i|C_t-1)",
     "Recurrent interactions among language networks support structural revision and recovery."),
    ("F", "Estimate P600-like activity", "op_28_estimate_p600",
     "Map conflict and revision demand to a positive-going P600-like ERP estimate.",
     "P600_t = k_P * (conflict + revision_cost)",
     "P600 is a debated ERP component associated with reanalysis/monitoring in many paradigms; this is only a proxy."),
    ("G", "Integrate the incoming word into syntax", "op_29_integrate_syntax",
     "Attach the word to the current grammatical representation and update unresolved dependencies.",
     "P(G_t|W_1:t)",
     "Distributed neural activity updates the evolving structural representation word by word."),
    ("G", "Integrate the incoming word into semantics", "op_30_integrate_semantics",
     "Add lexical meaning to the event representation and update participant roles and propositions.",
     "M_t = F(M_t-1,V_t,G_t)",
     "Semantic and combinatorial networks integrate lexical meaning with current sentence structure."),
    ("G", "Update competing sentence interpretations", "op_31_update_hypotheses",
     "Update the posterior probability of each sentence interpretation using the new word.",
     "P(H_i|W_1:t) proportional to P(W_t|H_i,W_1:t-1)P(H_i|W_1:t-1)",
     "Competitive activation and recurrent interaction support continuous hypothesis updating."),
    ("G", "Reconcile predictions with actual input", "op_32_reconcile_predictions",
     "Apply precision-weighted residuals to correct the active interpretation and future expectations.",
     "epsilon_t = X_t - X_hat_t; M_t = M_t-1 + eta Pi_t epsilon_t",
     "Bidirectional neural signaling lets sensory evidence modify top-down expectations."),
    ("G", "Decide whether to maintain or revise interpretation", "op_33_maintain_or_revise",
     "Retain a coherent interpretation or initiate revision when conflict and uncertainty remain high.",
     "S(H_i) = w1 S_syntax + w2 S_semantics + w3 S_world - w4 C_conflict",
     "Monitoring and cognitive-control systems regulate further interpretation work."),
    ("G", "Update linguistic working context", "op_34_update_working_context",
     "Append the recognized word and decay or preserve context activations according to relevance.",
     "A_i(t) = A_i(t-1) exp(-lambda delta_t) + I_i(t)",
     "Recurrent activity, selective attention, and working-memory mechanisms maintain active context."),
    ("G", "Generate predictions for the next word", "op_35_predict_next_word",
     "Generate a fresh next-word distribution from the reconciled sentence state.",
     "P(W_t+1|W_1:t)",
     "Updated high-level representations feed back to lower-level lexical prediction."),
    ("G", "Detect sentence completion", "op_36_detect_completion",
     "Combine punctuation, input-boundary, syntactic, and dependency evidence to detect completion.",
     "P(complete|S_t); complete when boundary evidence and structural confidence support it",
     "Prosodic, syntactic, semantic, and discourse cues jointly support boundary recognition."),
    ("G", "Consolidate the final sentence interpretation", "op_37_consolidate_interpretation",
     "Construct a final proposition from syntax, semantics, thematic roles, and world knowledge.",
     "R_sentence = F(G_final,M_final,K,D)",
     "Distributed language and conceptual networks support an integrated sentence-level representation."),
    ("G", "Update long-term predictive knowledge", "op_38_update_long_term_knowledge",
     "Update local word associations and event expectations so later runs can benefit from experience.",
     "theta_t+1 = theta_t + eta grad log P(W_t|C_t-1); Delta W_ij = eta A_i A_j",
     "Experience-dependent synaptic plasticity changes future predictions; the implemented update is an abstract learning analogue."),
)

_OPERATION_STAGE_NAMES = {stage.stage_id: stage.name for stage in STAGE_REGISTRY}
OPERATION_REGISTRY = tuple(
    OperationSpec(
        operation_id=f"TDPI-{number:03d}",
        number=number,
        stage_id=row[0],
        stage_name=_OPERATION_STAGE_NAMES[row[0]],
        name=row[1],
        method_name=row[2],
        purpose=row[3],
        mathematical_form=row[4],
        biological_basis=row[5],
    )
    for number, row in enumerate(_OPERATION_ROWS, start=1)
)


def registry_as_dicts() -> dict[str, list[dict[str, object]]]:
    """Return the complete stage and operation registries for inspection and logging."""
    return {
        "stages": [stage.to_dict() for stage in STAGE_REGISTRY],
        "operations": [operation.to_dict() for operation in OPERATION_REGISTRY],
    }
