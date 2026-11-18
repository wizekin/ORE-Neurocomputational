"""Canonical 18-operation registry for Point 03 sleep-dependent systems consolidation.

Operation order follows the Point 03 source. Equations are model abstractions or analytical
diagnostics, not literal neuronal arithmetic.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class OperationSpec:
    """Describe one source-ordered replay operation, its mathematical abstraction, and caveats."""

    operation_id: str
    name: str
    purpose: str
    mathematical_form: str
    biological_reference: str
    evidence_level: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return the operation definition as a plain dictionary for logs and documentation."""
        return asdict(self)


_ROWS = (
    ("SC01", "Enter an offline consolidation opportunity",
     "Gate replay using supplied sleep-state classification and confidence.",
     "G=1[state in {NREM, allowed_awake_rest} and p(state)>=theta].",
     "State-dependent replay occurs in NREM and awake rest; state classification is a measurement/model layer.",
     "offline_replay_supported; state gate is an engineering decision",
     ("sleep_state", "state_confidence"), ("offline_gate",)),
    ("SC02", "Estimate cortical slow-oscillation phase",
     "Estimate Up/Down timing from a supplied phase or filtered physiological signal.",
     "z(t)=x_SO(t)+iH{x_SO(t)}; phi_SO=arg(z).",
     "Neocortical slow oscillations alternate Down and Up states; Hilbert phase is an analysis tool.",
     "slow_oscillation_supported; phase extraction is analytical",
     ("slow_oscillation_signal", "sample_rate_hz"), ("slow_phase", "phase_confidence")),
    ("SC03", "Detect thalamocortical spindle events",
     "Identify sigma-burst candidates and their overlap with cortical Up-state timing.",
     "SP=1[envelope>=theta_SP and duration>=d_min].",
     "Thalamic reticular, thalamocortical, and cortical circuits generate spindles.",
     "spindles_and_nesting_supported; envelope threshold is model-specific",
     ("spindle_envelope", "spindle_duration", "slow_phase"), ("spindle_event", "spindle_strength")),
    ("SC04", "Detect hippocampal sharp-wave ripple events",
     "Score ripple-band and sharp-wave evidence, keeping event detection separate from memory identity.",
     "SWR=1[0.6 z_ripple+0.4 z_sharp>=theta and d_min<=duration<=d_max].",
     "CA3-associated sharp waves and CA1 ripples mark coordinated hippocampal population events.",
     "SWRs_supported; score coefficients and thresholds are model choices",
     ("ripple_band_z", "sharp_wave_z", "ripple_duration"), ("SWR_event", "SWR_score")),
    ("SC05", "Select a hippocampal trace for reactivation",
     "Rank eligible traces using supplied recency, salience, value, goal relevance, interference, and cue match.",
     "Q_i=w_rR_i+w_sS_i+w_vV_i+w_gG_i-w_dD_i; P_i=softmax(Q_i/tau).",
     "Replay content can be selective; the universal neural prioritization rule is unknown.",
     "selective_replay_supported; priority equation is hypothetical",
     ("trace_set", "partial_cue", "priority_attributes"), ("selected_trace", "selection_probabilities")),
    ("SC06", "Reconstruct the trace through hippocampal pattern completion",
     "Reactivate a stored population pattern from a complete or partial cue using recurrent association.",
     "h_(k+1)=sigmoid(W_H h_k+B u-inhibition).",
     "CA3 recurrent circuitry is implicated in associative retrieval; CA1 and entorhinal inputs contribute.",
     "pattern_completion_supported; rate network is a surrogate",
     ("selected_trace", "partial_cue", "W_H"), ("completed_hippocampal_pattern",)),
    ("SC07", "Replay the temporal order of associated elements",
     "Reconstruct ordered items and permit compressed forward or reverse replay.",
     "logP(S)=sum_t log(T[e_t,e_(t+1)]+epsilon); t'_k=gamma(t_k-t_0).",
     "Sequence replay and temporal compression are observed in some tasks; not all memories replay linearly.",
     "sequence_replay_supported; transition and compression equations are abstractions",
     ("trace_sequence", "transition_structure", "direction"), ("replay_sequence", "replay_times")),
    ("SC08", "Coordinate replay with cortical Up states and spindles",
     "Estimate graded SO-spindle-ripple phase alignment rather than treating timing as a binary guarantee.",
     "PLV=|mean(exp(i phi_SO(t_SWR)))|; G_align=g(phi_SO,SP,SWR).",
     "Nested SO, spindle, and ripple events can coordinate hippocampal-cortical communication.",
     "rhythm_coupling_supported; scalar combined gate is simplified",
     ("slow_phase", "spindle_event", "SWR_event", "spindle_phase"), ("alignment_score", "phase_locking")),
    ("SC09", "Propagate reactivated information toward cortical networks",
     "Transform hippocampal activity through effective hippocampal-cortical connectivity.",
     "I_C=W_HC h; c_(t+1)=sigmoid(W_CC c_t+I_C+I_other).",
     "Subiculum, entorhinal, parahippocampal, association, and prefrontal pathways distribute activity.",
     "anatomical_pathways_supported; matrices aggregate multiple pathways",
     ("hippocampal_pattern", "W_HC", "cortical_state"), ("cortical_drive",)),
    ("SC10", "Reactivate cortical elements associated with the episode",
     "Measure partial distributed reinstatement across feature groups.",
     "R_i=cos(c(t),c_i); R_total=sum_m omega_m R_(i,m).",
     "Hippocampal reactivation can coincide with reinstatement in distributed feature-selective cortices.",
     "cortical_reinstatement_supported; similarity is a model metric",
     ("cortical_drive", "stored_cortical_pattern"), ("cortical_activity", "reinstatement_score")),
    ("SC11", "Convert coordinated co-activity into synaptic eligibility",
     "Maintain a decaying trace of pre/post cortical activity and its timing.",
     "E_j(t+1)=rho E_j(t)+F(pre_j,post_j,delta_t).",
     "Eligibility traces and timing-dependent plasticity are supported at synapses; one matrix is simplified.",
     "eligibility_mechanisms_supported; matrix trace is an abstraction",
     ("pre_activity", "post_activity", "timing", "eligibility"), ("eligibility_matrix",)),
    ("SC12", "Apply timing-sensitive synaptic plasticity",
     "Apply a bounded, state- and alignment-modulated update to eligible cortical connections.",
     "deltaW=eta*G_align*m*E*STDP(delta_t); W=clip(W+deltaW).",
     "LTP/LTD and STDP-like changes depend on synapse, cell type, receptors, and modulators.",
     "activity_dependent_plasticity_supported; chosen rule is one model",
     ("eligibility_matrix", "alignment", "modulators", "W_CC"), ("weight_delta", "updated_W_CC")),
    ("SC13", "Integrate new associations with existing cortical knowledge",
     "Strengthen supported overlap while preserving distinctive information and trace-specific exemplars.",
     "a_i=softmax(beta*cos(c_new,c_i)); L_sep=sum_i!=j max(0,mu-cos(c_i,c_j)).",
     "Distributed cortical representations overlap; similarity-weighted merging is a computational hypothesis.",
     "distributed_integration_supported; separation penalty is computational",
     ("new_cortical_pattern", "prior_cortical_patterns"), ("updated_prototype", "separation_report")),
    ("SC14", "Repeat replay across multiple events and offline periods",
     "Accumulate small memory-specific updates over repeated replay opportunities.",
     "W_(k+1)=W_k+deltaW_k-lambda_H(W_k-W_0).",
     "Replay recurs across SWRs and NREM periods; cumulative weight rule is an abstraction.",
     "repeated_replay_supported; recurrence schedule is not universal",
     ("replay_history", "weight_state", "new_opportunity"), ("cumulative_updates", "replay_count")),
    ("SC15", "Stabilize cortical representations and maintain network balance",
     "Constrain activity and weight magnitudes with compensatory homeostatic gain.",
     "e_i=r_target-rbar_i; W_ij'=s_i W_ij.",
     "Homeostatic regulation and synaptic renormalization interact with selective consolidation.",
     "homeostatic_processes_supported; global scaling surrogate is simplified",
     ("cortical_activity", "target_activity", "W_CC"), ("homeostatic_gain", "stable_weights")),
    ("SC16", "Measure cortical support for later retrieval",
     "Evaluate cue-driven reinstatement and estimate support with and without hippocampal input.",
     "M_i=alpha*Sim(cue,c_i)+beta*Acc_i; D_HPC=Perf_intact-Perf_reduced.",
     "Some consolidated memories gain cortical support; dependence varies by task and memory type.",
     "variable_cortical_support_supported; independence estimate requires controlled tests",
     ("partial_cue", "cortical_patterns", "retrieval_probe"), ("retrieval_support", "dependence_estimate")),
    ("SC17", "Continue consolidation across time",
     "Maintain evolving cortical and hippocampal state across offline periods and later reactivation.",
     "C_i(t+1)=C_i+eta_i(1-C_i)-delta_i C_i.",
     "Consolidation unfolds over time and later experience can update, integrate, or interfere.",
     "time_evolving_consolidation_supported; scalar index is phenomenological",
     ("prior_consolidation", "alignment", "interference", "later_replay"), ("updated_consolidation_state",)),
    ("SC18", "Validate whether replay improved memory",
     "Compare external pre/post performance and report association without inferring causality from correlation.",
     "DeltaP=P_post-P_pre; r=Pearson(coupling,DeltaP).",
     "Selective replay perturbation and later behavior provide stronger tests than event detection alone.",
     "causal_validation_requires_experiment; correlation is not causation",
     ("pre_performance", "post_performance", "coupling_history"), ("retention_change", "validation_report")),
)


OPERATION_REGISTRY = tuple(OperationSpec(*row) for row in _ROWS)


def registry_as_dicts() -> list[dict[str, object]]:
    """Return all 18 operation definitions in canonical source order."""
    return [row.to_dict() for row in OPERATION_REGISTRY]
