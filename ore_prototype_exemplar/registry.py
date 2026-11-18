"""Canonical 16-operation registry for Point 07 prototype/exemplar concept representation."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class OperationSpec:
    """Describe a source-ordered operation, candidate mathematical form, and biological scope."""

    operation_id: str
    name: str
    purpose: str
    mathematical_form: str
    biological_reference: str
    evidence_level: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return an ordinary dictionary suitable for traces and SQLite event metadata."""
        return asdict(self)


_ROWS = (
    ("PE01", "Detect object or event and open a learning episode",
     "Accumulate sensory evidence until a coherent candidate item can be represented with temporal and contextual identity.",
     "A_k(t)=integral_0^t(e_k(tau)-b_k)d tau; accept when A_k(t)>=threshold.",
     "Sensory pathways, perceptual grouping, and attentional gating; this evidence integral is a computational surrogate.",
     "perceptual grouping and evidence accumulation supported; explicit integral is a model",
     ("sensory_input", "task_goal"), ("candidate_episode", "coherence", "episode_identity")),
    ("PE02", "Extract modality-specific and association features",
     "Encode observed continuous, categorical, binary, and relational attributes while keeping uncertainty and unobserved features distinct.",
     "x=f_theta(s); store (estimate_i, uncertainty_i, modality_i, reliability_i).",
     "Visual, auditory, somatosensory, motor, and association cortical populations; no single feature table is implied.",
     "multimodal feature extraction supported; canonical feature objects are computational abstractions",
     ("candidate_episode", "sensory_modalities"), ("feature_map", "observation_mask", "uncertainty")),
    ("PE03", "Select feature relevance and attentional weights",
     "Allocate gain using current attention, goals, and prior knowledge; normalize competition across represented dimensions.",
     "a_i=.50A_i+.30G_i+.20P_i; w_i=exp(a_i/tau)/sum_j exp(a_j/tau).",
     "Prefrontal and parietal control networks can bias sensory gain and task-relevant feature processing.",
     "attention/goal modulation supported; softmax weighting is computational",
     ("feature_map", "attention", "goals", "prior_knowledge"), ("feature_gain", "task_metric")),
    ("PE04", "Bind the feature combination into an episode trace",
     "Bind co-occurring features, relations, context, and time into a distinct memory cue without immediately averaging away the instance.",
     "h_i=phi(W_x x_i+W_c c_i+W_r r_i); preserve episode_id and context.",
     "Hippocampal/medial-temporal relational binding and pattern separation; this is a functional analogy.",
     "episodic and relational binding supported; vector binding is an abstraction",
     ("feature_map", "context", "relations"), ("episodic_trace", "retrieval_cues")),
    ("PE05", "Retrieve candidate categories and retain uncertainty",
     "Compare the new item with known category summaries and exemplar sets, incorporate priors, and keep an unknown option.",
     "P(C|x) proportional to P(x|C)P(C); do not force a label below the acceptance criterion.",
     "Medial-temporal retrieval interacts with temporal association cortex and prefrontal decision control.",
     "prior-guided categorization supported; Bayesian posterior is a normative model",
     ("feature_map", "category_memory", "category_priors"), ("candidate_categories", "provisional_posterior")),
    ("PE06", "Update the exemplar pathway with individual instances",
     "Store the exact feature/context/relation conjunction, its accessibility, confidence, and identity as a distinct episode.",
     "M_j(t)=M_j0 exp(-lambda t)+sum_r q_r exp(-lambda(t-t_r)); retain E_C={e_1,...,e_n}.",
     "Hippocampal and medial-temporal systems contribute to episodic binding and retrieval; cortical systems represent component content.",
     "episodic memory supported; exponential accessibility is a tractable approximation",
     ("episodic_trace", "category_assignment"), ("exemplar_store", "memory_strength")),
    ("PE07", "Update the prototype pathway with category central tendency",
     "Update reliability-weighted numeric means/variances and categorical distributions while preserving the original episodes separately.",
     "mu_i=sum_n q_n x_ni/sum_n q_n; categorical P(v|C)=sum_n q_n I(x_ni=v)/sum_n q_n.",
     "Repeated co-activation and distributed cortical plasticity can support gradual extraction of regularities.",
     "regularity learning supported; means and frequency tables are computational summaries",
     ("labelled_episode", "reliability"), ("prototype_statistics", "within_category_variability")),
    ("PE08", "Separate typicality from diagnosticity and exceptions",
     "Estimate feature prevalence, between-category contrast, and within-category variability; do not treat every common feature as necessary.",
     "D_i=logit(P(f_i|C))-logit(P(f_i|not C)); retain low-frequency high-information items.",
     "Distributed cortical comparison and prefrontal task control can emphasize category-relevant information.",
     "contrastive learning supported; information scores are model-based",
     ("prototype_statistics", "competitor_statistics"), ("typicality", "diagnosticity", "exception_candidates")),
    ("PE09", "Compute similarity to category prototypes",
     "Align candidate and prototype features, weight mismatches, and transform distance into a similarity-like score.",
     "d_P=(sum_i w_i |x_i-p_i|^r)^(1/r); S_P=exp(-c d_P).",
     "Perceptual/temporal association representations and distributed cortical category patterns support similarity judgments.",
     "similarity-based categorization supported; Minkowski distance and exponential kernel are computational",
     ("candidate_features", "prototype", "feature_weights"), ("prototype_distance", "prototype_similarity")),
    ("PE10", "Retrieve and compare individual exemplars",
     "Retrieve partial-cue matches and aggregate similarity weighted by accessibility, diagnosticity, and contextual relevance.",
     "S_E=sum_j a_j exp(-c d(x,e_j)); also retain max_j match and supporting episode IDs.",
     "Hippocampal pattern completion, medial-temporal episodic retrieval, and cortical reinstatement can contribute.",
     "episodic retrieval supported; generalized-context aggregation is a computational model",
     ("candidate_features", "exemplar_store", "context"), ("retrieval_ranking", "exemplar_similarity")),
    ("PE11", "Combine similarity evidence with category expectations",
     "Normalize category-level prototype and exemplar evidence with category priors and preserve graded confidence.",
     "P(C|x)=beta_C S_C(x)/sum_k beta_k S_k(x), implemented as stable log-softmax.",
     "Prefrontal decision networks integrate evidence, expectations, task costs, and response criteria.",
     "evidence/prior integration supported; Luce probability normalization is computational",
     ("prototype_scores", "exemplar_scores", "category_priors"), ("pathway_probabilities", "confidence")),
    ("PE12", "Select or blend prototype and exemplar pathways",
     "Set an item- and context-sensitive blend according to prototype stability, exemplar retrieval, typicality, and task demand.",
     "S_H=alpha S_P+(1-alpha)S_E; alpha=sigmoid(theta_0+theta_1 R_P-theta_2 R_E+theta_3 T+theta_4 U).",
     "Distributed frontal, parietal, cortical, and medial-temporal interactions may control strategy; no fixed one-region mapping.",
     "both representational levels behaviorally supported; controller parameters require empirical fitting",
     ("prototype_reliability", "exemplar_reliability", "typicality", "task_context"), ("blend_weight", "hybrid_score")),
    ("PE13", "Detect and protect informative exceptions",
     "Detect covariance-aware atypicality and retain unusual category members when labels or specific exemplars support membership.",
     "D_M=sqrt((x-mu)^T Sigma^(-1)(x-mu)); preserve if outlier and exemplar/label evidence are strong.",
     "Hippocampal episode binding can retain unique conjunctions while distributed cortex captures shared regularities.",
     "exception memory and episodic specificity supported; Mahalanobis threshold is a model parameter",
     ("prototype_distance", "exemplar_matches", "category_feedback"), ("outlier_score", "exception_flag")),
    ("PE14", "Learn from feedback and prediction error",
     "Compare a prediction with explicit outcome information and update attention, labels, exemplar strengths, and prototype statistics cautiously.",
     "delta=y-yhat; mu_C <- mu_C+eta q(x-mu_C); w <- w-eta_w grad L.",
     "Neuromodulatory and corticostriatal teaching signals can gate plasticity; feedback signals are task-dependent.",
     "feedback-dependent learning supported; scalar prediction error/gradient are computational analogies",
     ("prediction", "explicit_feedback", "reliability"), ("prediction_error", "plasticity_gate", "revised_memory")),
    ("PE15", "Consolidate repeated structure without deleting episodic detail",
     "Strengthen recurring category patterns and replay selected traces while retaining distinctive exemplars and contextual provenance.",
     "prototype stability increases with repeated evidence; exemplar accessibility <- decay + replay increments.",
     "Hippocampal-cortical interaction and repeated reactivation can support gradual systems-level learning.",
     "systems consolidation and replay supported; exact scheduling/strengthening rules are approximations",
     ("episode_store", "prototype_statistics", "replay_selection"), ("consolidated_prototype", "retained_exemplars")),
    ("PE16", "Test generalization and produce a graded category decision",
     "Evaluate held-out members, compare prototype-only/exemplar-only/hybrid predictions, and report uncertainty rather than forced certainty.",
     "L=-sum_t log P(y_t|x_t,Theta); compare held-out accuracy, NLL, and Brier score.",
     "Distributed cortical and medial-temporal representations interact with prefrontal decision systems during generalization.",
     "generalization and model comparison supported; software metrics are validation criteria",
     ("hybrid_posterior", "acceptance_threshold", "held_out_labels"), ("category_decision", "uncertainty", "validation_metrics")),
)

OPERATION_REGISTRY = tuple(
    OperationSpec(operation_id, name, purpose, mathematics, biology, evidence, inputs, outputs)
    for operation_id, name, purpose, mathematics, biology, evidence, inputs, outputs in _ROWS
)


def registry_as_dicts() -> list[dict[str, object]]:
    """Export all sixteen operation specifications in their source-defined order."""
    return [row.to_dict() for row in OPERATION_REGISTRY]
