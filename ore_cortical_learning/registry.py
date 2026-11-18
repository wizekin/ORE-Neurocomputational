"""Canonical 14-operation registry for gradual neocortical learning from repeated experience.

Operation order follows Point_02_Gradual_Cortical_Learning_From_Repeated_Experience.md. Each
mathematical form is an implementable abstraction, not a claim of literal neuronal arithmetic.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class OperationSpec:
    """Describe one source-ordered cortical-learning operation and its biological caveats."""

    operation_id: str
    name: str
    purpose: str
    mathematical_form: str
    biological_reference: str
    evidence_level: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return the operation specification as a plain dictionary."""
        return asdict(self)


_OPERATION_ROWS = (
    ("CL01", "Receive a new encounter and preserve its distinct features",
     "Accept sensory, action, context, goal, and optional hippocampal episode-trace signals without erasing encounter-specific differences.",
     "x_t = concat(sensory_t, action_t, context_t, goal_t, episode_trace_t).",
     "Distributed sensory and association cortices process parallel content streams; context remains relevant to which synapses and populations are engaged.",
     "distributed_cortical_processing_supported; concatenated vector is a model representation",
     ("sensory_activity", "action_activity", "context", "goal_activity", "episode_trace"), ("encounter_vector", "context_label")),
    ("CL02", "Construct distributed cortical representations",
     "Transform the encounter into recurrent population activity with nonlinear activation, feedback, and inhibitory competition.",
     "z_t = sigmoid(W_in x_t + gamma W_rec z_(t-1) + b - I_t).",
     "Cortical representations are distributed across specialized and association networks with recurrent excitation and inhibition; one homogeneous matrix is a simplification.",
     "distributed_population_coding_supported; rate network is an engineering surrogate",
     ("encounter_vector", "cortical_weights", "previous_activity"), ("cortical_activity", "population_state")),
    ("CL03", "Compare the current pattern with previous cortical activity",
     "Measure overlap with retained encounters while keeping contextual match and surface differences visible.",
     "s(t,k) = z_t dot z_k / (||z_t|| ||z_k|| + epsilon); d(t,k)=||z_t-z_k||_2.",
     "Recurrent networks and memory-guided responses can reinstate overlapping population patterns; cosine similarity and Euclidean distance are model metrics.",
     "pattern_overlap_is_useful_framework; explicit vector metric is computational",
     ("current_activity", "encounter_history", "context_id"), ("similarity_scores", "nearest_prior_encounter")),
    ("CL04", "Detect features and relationships that recur across encounters",
     "Accumulate feature means and pairwise co-activity/covariance statistics instead of treating one exposure as a stable rule.",
     "mu_t=(1-alpha)mu_(t-1)+alpha z_t; M_t=(1-alpha)M_(t-1)+alpha z_t z_t^T.",
     "Repeated co-activation can contribute to Hebbian and covariance-related synaptic plasticity; running averages are statistical surrogates for multiple timescales of traces.",
     "repeated_activity_and_plasticity_supported; EMA matrices are abstractions",
     ("cortical_activity", "running_means", "coactivity_statistics"), ("feature_statistics", "relationship_statistics")),
    ("CL05", "Estimate whether a detected relationship is reliable",
     "Estimate conditional co-activation and evidence strength, including variability across distinct contexts and inconsistent observations.",
     "P(z_j active | z_i active)=(a_ij+success_ij)/(a_ij+b_ij+observations_ij).",
     "Repeated observations, outcomes, and context can constrain plasticity; Bayesian counts are a model-based uncertainty representation, not neuronal probability storage.",
     "reliability_estimation_is_computational_hypothesis; contextual evidence is essential",
     ("relationship_counts", "context_history", "encounter_diversity"), ("reliability_matrix", "uncertainty_estimate")),
    ("CL06", "Activate existing cortical predictions",
     "Generate downstream activity from learned weights and calculate a prediction error only when a target is actually observed.",
     "yhat_t=f(W_pred z_t); e_t=y_observed-yhat_t.",
     "Predictive and feedback interactions occur in distributed cortical circuits; a single explicit subtraction vector is not assumed.",
     "predictive_processing_supported; error vector is an abstraction",
     ("cortical_activity", "prediction_weights", "observed_activity"), ("predicted_activity", "prediction_error")),
    ("CL07", "Determine whether plasticity should be induced",
     "Combine local pre/post activity timing, decaying eligibility traces, behavioral modulation, reliability, and local state to gate learning.",
     "E_t=lambda E_(t-1)+phi(pre,post,delta_t); Q_t=E_t times G_t times R_t.",
     "Plasticity depends on timing, dendritic/postsynaptic state, inhibition, and neuromodulation; one scalar multiplicative gate is only an abstraction.",
     "eligibility_and_modulation_mechanisms_supported; scalar gate is simplified",
     ("pre_post_activity", "timing", "local_state", "modulators", "reliability"), ("eligibility_trace", "plasticity_gate")),
    ("CL08", "Apply small cumulative synaptic modifications",
     "Update only eligible connections with a small Hebbian/covariance increment and prediction-error correction where observed targets exist.",
     "deltaW=eta G R Cov(pre,post); deltaW_out=eta G (error times sigmoid_derivative) z^T.",
     "Long-term potentiation, depression, spike-timing-dependent plasticity, and other local learning rules vary by synapse and circuit; the implementation selects one teaching rule.",
     "activity_dependent_plasticity_strongly_supported; selected equations are model alternatives",
     ("eligibility_trace", "coactivity", "prediction_error", "learning_rate"), ("local_weight_delta", "updated_weights")),
    ("CL09", "Preserve distributed knowledge while integrating new evidence",
     "Constrain update magnitude and penalize movement of important connections away from a slowly maintained reference state.",
     "J=L_new+lambda sum_ij F_ij (W_ij-W_ref_ij)^2; ||deltaW||_infinity <= delta_max.",
     "Distributed coding, inhibition, synaptic constraints, and multiple memory systems reduce interference; explicit importance-matrix regularization is a machine-learning analogy.",
     "interference_control_objective_supported; EWC_like_penalty_is_computational",
     ("candidate_update", "prior_weights", "importance_estimates"), ("bounded_update", "interference_report")),
    ("CL10", "Integrate examples across time rather than relying on one exposure",
     "Update running feature prototypes and relation statistics across varied encounters, retaining context-dependent branches.",
     "mu_N=mu_(N-1)+(z_N-mu_(N-1))/N; or mu_t=(1-alpha)mu_(t-1)+alpha z_t.",
     "Cortical learning generally accumulates evidence across experiences and multiple timescales; EMA statistics are software proxies, not a single biological clock.",
     "gradual_integration_cls_framework; exact statistic is an abstraction",
     ("current_activity", "previous_statistics", "context_id"), ("updated_prototypes", "encounter_count")),
    ("CL11", "Regulate activity and synaptic strength",
     "Adjust population excitability toward a target, apply an inhibitory competition term, and keep synaptic weights bounded.",
     "gain_j <- gain_j+kappa(rho-mean_activity_j); W <- clip(W+deltaW,w_min,w_max).",
     "Inhibitory circuits and homeostatic plasticity stabilize activity and synaptic scaling over distinct timescales; gain control is a coarse rate-model surrogate.",
     "homeostatic_and_inhibitory_mechanisms_supported; scalar target is not universal",
     ("population_activity", "target_activity", "synaptic_weights"), ("regulated_activity", "bounded_weights")),
    ("CL12", "Test whether the updated network responds to recurring structure",
     "Evaluate prediction error on caller-marked held-out encounters with varied surface details and context; do not call training fit generalization proof.",
     "E_test=(1/N) sum_k ||y_k-yhat_k||_2^2.",
     "Sensory feedback and behavioral outcomes continuously constrain predictions; an explicit held-out test set exists only in the computational experiment.",
     "model_validation_metric; held_out_status_is_caller_declared",
     ("held_out_input", "prediction", "observed_activity", "context"), ("generalization_error", "evaluation_status")),
    ("CL13", "Retain uncertainty when evidence is inconsistent",
     "Maintain context-specific relationship counts and Beta-style uncertainty; preserve separate branches when outcomes differ across contexts.",
     "p_ij|c ~ Beta(a_ijc,b_ijc); E[p_ij|c]=a_ijc/(a_ijc+b_ijc).",
     "Context-sensitive representations and plasticity are biologically plausible; a Beta distribution is a statistical representation, not a neuronal state variable.",
     "context_sensitivity_supported; Bayesian posterior is a software abstraction",
     ("positive_evidence", "negative_evidence", "context_id"), ("context_branches", "posterior_uncertainty")),
    ("CL14", "Continue gradual learning across subsequent encounters",
     "Persist the updated learner state and return it to the next encounter without transferring or replacing episodic memories.",
     "(W,M,R,H)_(t+1)=F((W,M,R,H)_t,e_t); each encounter reuses prior state with bounded increments.",
     "The complementary-learning-systems framework describes interacting fast episodic and slower cortical learning dynamics; Point 02 excludes sleep replay and systems consolidation.",
     "systems_level_cls_framework; unified state transition is an engineering model",
     ("updated_weights", "running_statistics", "reliability", "homeostasis"), ("persistent_cortical_state", "next_encounter_ready")),
)

OPERATION_REGISTRY: tuple[OperationSpec, ...] = tuple(
    OperationSpec(
        operation_id=row[0], name=row[1], purpose=row[2], mathematical_form=row[3],
        biological_reference=row[4], evidence_level=row[5], inputs=row[6], outputs=row[7],
    ) for row in _OPERATION_ROWS
)


def registry_as_dicts() -> list[dict[str, object]]:
    """Return all 14 canonical operations in their original source order."""
    return [item.to_dict() for item in OPERATION_REGISTRY]
