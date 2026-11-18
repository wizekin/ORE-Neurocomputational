"""Canonical 16-operation registry for Point 05 novelty-modulated consolidation.

The order and biological themes follow Point_05_Novelty_Modulated_Hippocampal_Consolidation.md.
Equations are candidate model abstractions, never literal claims about neuronal arithmetic.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class OperationSpec:
    """Describe one source-ordered operation, its purpose, equation, and biological caveat."""

    operation_id: str
    name: str
    purpose: str
    mathematical_form: str
    biological_reference: str
    evidence_level: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return the operation specification as a plain dictionary for logs and tests."""
        return asdict(self)


_ROWS = (
    ("NC01", "Retrieve the recently encoded hippocampal trace",
     "Select the episode-specific population code, encoding time, and synapses eligible for post-learning modulation.",
     "H_E=[h_1,...,h_n]; S_E={j:a_j>theta_a}.",
     "Hippocampal relational binding and event-specific ensemble recruitment support rapid episodic encoding.",
     "hippocampal_trace_supported; explicit vector and synapse set are model representations",
     ("memory_trace", "encoding_time", "active_synapses"),
     ("trace_target", "encoding_reference", "eligible_synapse_set")),
    ("NC02", "Establish transient synaptic eligibility tags",
     "Create local, temporary tags only at learning-active synapses and retain their individual decay times.",
     "T_j(t0)=f(activity,calcium,local_signalling); T_j(t)=T_j(t0)e^(-dt/tau_T).",
     "Synaptic tagging and capture is an influential supported framework; the tag is a biochemical state, not a literal label.",
     "STC_framework_supported; exponential tag decay is a simplified kinetic model",
     ("active_synapses", "encoding_activity", "encoding_time"),
     ("synaptic_tags", "tag_timestamps")),
    ("NC03", "Monitor the post-learning interval",
     "Track later events, time since encoding, sleep onset, reward, stress, and the remaining temporal eligibility.",
     "W_time(dt)=exp(-dt/tau_W).",
     "Post-encoding experience and internal state can modulate consolidation during task-dependent windows.",
     "post_learning_modulation_supported; time constant is model-specific",
     ("post_learning_events", "encoding_time", "internal_state"),
     ("time_weighted_events", "temporal_eligibility")),
    ("NC04", "Compare post-learning events with predictions and familiarity",
     "Estimate prediction error, prior-pattern similarity, surprise, and the dimensions that differ from familiar knowledge.",
     "e=x-x_hat; mismatch=1-exp(-RMS(e)); surprise=-log P(x|F).",
     "Hippocampal and cortical networks detect novelty and contextual mismatch; explicit distance functions are analytical proxies.",
     "novelty_and_mismatch_supported; distance and information measures are computational",
     ("event_features", "prior_knowledge", "predicted_features"),
     ("prediction_error", "familiarity", "novelty_evidence")),
    ("NC05", "Estimate novelty type, relevance, and salience",
     "Distinguish familiar, common, distinct, and uncertain novelty while incorporating goals, arousal, value, and stress.",
     "S_N=a*N+b*goal+c*arousal_efficiency+d*value-e*stress.",
     "Common-versus-distinct novelty is a proposed theoretical distinction; novelty is not identical to intensity.",
     "novelty_heterogeneity_supported; common/distinct mapping remains a hypothesis",
     ("novelty_evidence", "goal_relevance", "arousal", "motivation"),
     ("novelty_type", "novelty_salience")),
    ("NC06", "Recruit VTA and locus-coeruleus pathways",
     "Estimate separate, interacting VTA and LC population responses without assigning all hippocampal dopamine to VTA.",
     "m=[m_VTA,m_LC]=g(novelty,salience,context,goal).",
     "VTA dopaminergic modulation and LC noradrenergic/dopaminergic-capable projections can influence hippocampal memory persistence.",
     "VTA_LC_involvement_supported; recruitment function is a rate-model surrogate",
     ("novelty_type", "salience", "motivational_state"),
     ("vta_activity", "lc_activity", "pathway_uncertainty")),
    ("NC07", "Modulate hippocampal dopamine and noradrenaline",
     "Estimate local DA and NE separately, including release-source contributions and effective clearance over time.",
     "dD/dt=R_D-k_D*D; dNE/dt=R_NE-k_NE*NE.",
     "DA and NE have distinct receptor systems; LC can contribute both NE and hippocampal DA.",
     "neuromodulator_role_supported; concentration dynamics are coarse approximations",
     ("vta_activity", "lc_activity", "elapsed_time"),
     ("hippocampal_dopamine", "hippocampal_noradrenaline")),
    ("NC08", "Engage D1/D5 and adrenergic plasticity gates",
     "Estimate nonlinear receptor occupancy and combine it with local synaptic activity to maintain learning specificity.",
     "O_DA=D^n/(K_D^n+D^n); G_DA=f(O_DA,local_activity).",
     "Hippocampal D1/D5 and adrenergic signalling can modulate persistent plasticity in a context-dependent manner.",
     "receptor_modulation_supported; Hill occupancy is a simplified model",
     ("dopamine", "noradrenaline", "synaptic_activity"),
     ("d1_d5_gate", "adrenergic_gate")),
    ("NC09", "Produce or mobilize plasticity-related resources",
     "Estimate a transient PRP-like resource pool through interacting second-messenger and kinase pathway gates.",
     "dP/dt=q(G_DA,G_NE,G_Ca)-k_P*P.",
     "cAMP/PKA, ERK/MAPK, CREB, calcium and protein-synthesis pathways can contribute; no universal molecular chain is assumed.",
     "plasticity_resource_mechanisms_supported; resource pool is a coarse latent variable",
     ("d1_d5_gate", "adrenergic_gate", "calcium_activity", "elapsed_time"),
     ("plasticity_resource_pool", "resource_uncertainty")),
    ("NC10", "Capture resources at eligible learning synapses",
     "Match local tags with available resources and compatibility; untagged or inactive synapses should not gain the same support.",
     "C_j=T_j*P_j*M_j; accumulated capture is integral(C_j dt).",
     "Synaptic tagging and capture provides local specificity for later plasticity-related resources.",
     "STC_mechanism_supported; multiplicative coincidence is a model rule",
     ("synaptic_tags", "local_resources", "compatibility"),
     ("synapse_capture_scores", "capture_probability")),
    ("NC11", "Stabilize labile synaptic modifications",
     "Increase persistence when capture is sufficient while preserving potentiation/depression sign and allowing unsupported changes to decay.",
     "L_j(t+1)=L_j(t)+rho*P(stabilize_j)-mu*L_j(t).",
     "Early and late LTP and other forms of synaptic plasticity have different persistence; not every change is potentiation.",
     "persistent_synaptic_plasticity_supported; update equation is abstract",
     ("capture_probability", "signed_early_plasticity", "synaptic_persistence"),
     ("stabilized_synapses", "updated_synaptic_lifetimes")),
    ("NC12", "Stabilize the distributed hippocampal ensemble",
     "Aggregate synapse-level persistence into a feature-aligned ensemble estimate while preserving non-uniform stabilization.",
     "H_persist=sum_j omega_j*z_j; Q_H=cos(h_encoding,h_current).",
     "Hippocampal memories depend on distributed ensembles and relations rather than one uniformly strengthened synapse.",
     "ensemble_memory_supported; scalar persistence and coherence are model summaries",
     ("synaptic_persistence", "feature_indices", "hippocampal_pattern"),
     ("ensemble_persistence", "pattern_coherence")),
    ("NC13", "Estimate later retrieval expression and trace dominance",
     "Combine latent trace persistence with cue match and context fit, then compare the target against competing traces.",
     "R_i=a*H_i+b*Match_i+c*Context_i-d*Interference_i; P_i=softmax(R_i/tau).",
     "Retrieval expression depends on multiple cues and competitors; a stronger latent trace is not guaranteed to win every retrieval.",
     "retrieval_competition_supported; weighted score and softmax are decision-model abstractions",
     ("trace_persistence", "retrieval_cue", "context", "candidate_traces"),
     ("retrieval_score", "retrieval_probability")),
    ("NC14", "Track interference vulnerability separately from trace persistence",
     "Estimate overlap and conflict with later learning, distinguishing retrieval masking from actual trace weakening.",
     "O_ik=cos(h_i,h_k); I_i=sum(k!=i) O_ik*S_k*C_conflict(i,k).",
     "Novelty-associated memory dominance can coexist with increased susceptibility to interference in specific paradigms.",
     "interference_and_competition_supported; burden is a candidate model measure",
     ("target_trace", "competing_traces", "cue_conflict"),
     ("interference_risk", "competitor_contributions")),
    ("NC15", "Run parallel sleep-associated cortical consolidation",
     "Use supplied sleep, slow-oscillation, spindle, ripple, replay, and coordination evidence to update cortical integration independently.",
     "C_cortex'=C_cortex+eta*f(sleep,replay,coordination,prior_knowledge).",
     "Slow oscillations, spindles, and hippocampal ripples can support coordinated hippocampal-cortical replay and integration.",
     "sleep_and_replay_supported; supplied evidence and integration rule are model abstractions",
     ("sleep_state", "rhythm_evidence", "memory_specific_replay", "prior_knowledge"),
     ("cortical_integration", "hippocampal_cortical_coordination")),
    ("NC16", "Update multidimensional memory state and provenance",
     "Combine synaptic stabilization, episodic accessibility, cortical integration, semantic structure, retrieval, and uncertainty.",
     "M_E(t+1)=F(M_E(t),Novelty,DA,NE,STC,Sleep,Replay,Interference).",
     "Memory outcomes are distributed and multi-dimensional; provenance distinguishes supplied evidence from inferred mechanisms.",
     "multidimensional_memory_supported; aggregate state remains a computational representation",
     ("hippocampal_state", "cortical_state", "retrieval", "uncertainty"),
     ("consolidated_memory_state", "provenance_record")),
)


OPERATION_REGISTRY = tuple(OperationSpec(*row) for row in _ROWS)


def registry_as_dicts() -> list[dict[str, object]]:
    """Return all 16 Point 05 operations in canonical source order."""
    return [operation.to_dict() for operation in OPERATION_REGISTRY]
