"""Source-grounded 21-stage and sub-operation registry.

Mathematical expressions are executable model approximations or measurement
readouts. Biological descriptions identify plausible mechanisms and populations;
they are not claims that neurons explicitly calculate matrices or probabilities.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class OperationSpec:
    """Describe one executable sub-operation and its mathematical/biological mapping."""

    name: str
    equation: str
    functional_responsibility: str
    biological_reference: str


@dataclass(frozen=True)
class StageSpec:
    """Describe a source algorithm stage and its ordered sub-operations."""

    stage_id: str
    name: str
    phase: str
    objective: str
    biological_interpretation: str
    operations: tuple[OperationSpec, ...]


def _op(name: str, equation: str, function: str, biology: str) -> OperationSpec:
    """Create one immutable operation specification."""
    return OperationSpec(name, equation, function, biology)


def _stage(number: int, name: str, phase: str, objective: str, biology: str,
           operations: tuple[OperationSpec, ...]) -> StageSpec:
    """Create a sequentially numbered distributed-only stage."""
    return StageSpec(f"DS-{number:02d}", name, phase, objective, biology, operations)


_STAGE_DATA = (
    _stage(1, "Receive incoming conceptual information", "I — input encoding",
        "Receive linguistic, sensory, action, or recalled information and convert available signals into model-ready evidence.",
        "Auditory, visual, somatosensory, and language pathways transform external signals into activity that can reach property-sensitive cortical populations.",
        (
            _op("signal_sampling", "u[n] = s(n Δt)", "Sample an optional time-series at a defined interval.", "Sensory receptors and early sensory pathways transduce temporally varying input."),
            _op("signal_decomposition", "U(f) = rFFT(u[n])", "Estimate frequency-domain components when a sampled signal is supplied.", "Cochlear and cortical populations are frequency-sensitive; rFFT is an engineering readout, not a literal neuronal Fourier transform."),
            _op("feature_extraction", "f_input = [f1, ..., fn]", "Read supplied property evidence and token-level linguistic cues.", "Sensory association populations extract form, sound, movement, texture, and other properties."),
            _op("signal_normalization", "f_hat = (f − μ)/(σ + ε)", "Bound and normalize input strengths before activation.", "Adaptation and gain control regulate response ranges in sensory populations."),
            _op("temporal_filtering", "y[t] = α y[t−1] + (1−α)u[t]", "Apply a causal low-pass response to sampled signal envelopes.", "Synaptic and membrane time constants filter temporal input."),
        )),
    _stage(2, "Extract property-specific information", "I — input encoding",
        "Separate the incoming evidence into property-specific feature channels without collapsing them into one semantic vector.",
        "Distributed perceptual association populations respond selectively to properties such as shape, color, movement, sound, texture, and function.",
        (
            _op("property_feature_extraction", "f_r = φ_r(W_r u)", "Map each property cue into its relevant population channel.", "Property-sensitive populations in visual, auditory, tactile, and action-related cortex respond to distinct feature families."),
            _op("dimensionality_reduction", "z_r = P_r f_r", "Compress each population's feature vector while keeping separate population channels.", "Hierarchical cortical processing pools local activity into population-level patterns."),
            _op("feature_selectivity", "m_i = 1[f_i > θ_i]", "Mark evidence dimensions that exceed a feature-specific threshold.", "Neuronal response thresholds and tuning curves create selective responses."),
            _op("graded_activation", "a_i = σ(f_i − θ_i)", "Convert evidence into graded activation rather than an all-or-none semantic decision.", "Population firing rates vary with stimulus strength and tuning."),
        )),
    _stage(3, "Activate the initial concept-related neural populations", "I — input encoding",
        "Initiate activity in populations directly supported by the incoming stimulus.",
        "Neurons responsive to the current word, sound, image, touch, or action recruit connected cortical populations through synaptic input.",
        (
            _op("weighted_input", "z = W_in u", "Weight input features by their channel-specific projection.", "Synaptic efficacy weights signals arriving at a neural population."),
            _op("nonlinear_activation", "a = φ(z)", "Apply a bounded nonlinear activation to each population feature.", "Membrane integration and firing-rate input-output functions are nonlinear."),
            _op("threshold_detection", "m_i = 1[a_i > θ_i]", "Identify which dimensions are initially active.", "Spike thresholds and population response criteria define activity."),
            _op("initial_population_activity", "A_r = mean_i(a_{r,i})", "Measure initial activity independently for every population.", "Population firing-rate summaries approximate aggregate neural response."),
        )),
    _stage(4, "Retrieve learned distributed concept associations", "I — input encoding",
        "Spread activation through learned inter-population and experience-based associations.",
        "Previously strengthened synaptic pathways allow an activated population to reactivate connected property populations; no single semantic hub is required.",
        (
            _op("associative_activation_propagation", "a[t+1] = φ(W a[t] + b)", "Propagate evidence over the distributed connection graph.", "Recurrent excitatory synapses transmit activation between connected cortical populations."),
            _op("target_population_accumulation", "a_j = φ(Σ_i w_ji a_i + b_j)", "Accumulate incoming support separately at each target population.", "Postsynaptic integration sums synaptic influences before neuronal response."),
        )),
    _stage(5, "Reactivate perceptual-property networks", "II — distributed reactivation",
        "Reinstate visual, auditory, tactile, and spatial properties even when the object is not physically present.",
        "Associative cortical networks reinstate previously acquired property information in modality- and property-sensitive populations.",
        (
            _op("property_network_reactivation", "x_r[t+1] = φ(W_rs x_s[t])", "Reactivate perceptual channels through learned cross-population links.", "Top-down and associative synaptic inputs can reinstate perceptual-property populations."),
            _op("multiple_property_activation", "X_perception = {x_form, x_color, x_motion, x_sound}", "Keep simultaneously active property channels distinct.", "Different property-sensitive populations contribute concurrently to a concept."),
            _op("property_activation_strength", "A_property = Σ_i a_i", "Measure activity strength within each property channel.", "Population firing and local ensemble activity provide strength estimates."),
            _op("remembered_pattern_similarity", "S(x,p) = (x·p)/(||x||||p|| + ε)", "Compare the current population pattern with a stored property pattern.", "Associative cortical circuits can reinstate patterns similar to previously experienced states."),
        )),
    _stage(6, "Reactivate action and motor-related representations", "II — distributed reactivation",
        "Retrieve object-associated manipulations, movement patterns, and action sequences without requiring overt movement.",
        "Premotor, parietal, and posterior temporal action-related systems can contribute knowledge about object manipulation and use.",
        (
            _op("motor_pattern_retrieval", "x_action = φ(W_action,semantic x_semantic)", "Reinstate action-property activity linked to current evidence.", "Premotor and parietal populations represent action plans and learned object interactions."),
            _op("action_sequence_representation", "M = [m_1, ..., m_T]", "Represent a sequence of action-related states.", "Motor and premotor networks encode temporally structured action patterns."),
            _op("action_prediction", "m_hat[t+1] = F(m[t], c)", "Estimate a plausible next action state from current activity and context.", "Forward models and recurrent motor circuits contribute action prediction."),
            _op("motor_pattern_similarity", "S_motor = cosine(x_action, p_action)", "Compare retrieved action activity with learned action patterns.", "Population-pattern similarity is a computational readout of action-related representations."),
        )),
    _stage(7, "Reactivate emotional and affective representations", "II — distributed reactivation",
        "Retrieve affective value and emotional associations linked to the current concept and context.",
        "Affective and valuation-related systems, including amygdala-connected and interoceptive networks, contribute emotional significance to distributed representations.",
        (
            _op("affective_feature_representation", "f_affect = [valence, arousal, familiarity]", "Represent caller-supplied affective dimensions as their own feature population.", "Affective and interoceptive networks encode value, arousal, and bodily significance."),
            _op("emotional_activation", "a_affect = φ(W_affect x)", "Reactivate affective patterns from concept and episodic associations.", "Associative inputs can reactivate amygdala-connected and cortical valuation populations."),
            _op("valuation", "V = Σ_i w_i a_i", "Calculate a task-readable value estimate from affective dimensions.", "Valuation-related neural activity integrates positive and negative significance."),
            _op("context_dependent_valuation", "V(c|k) = f(V(c), k)", "Modulate retrieved affect by explicit current context.", "Affective meaning varies with context, goals, and prior experience."),
        )),
    _stage(8, "Retrieve associated episodic experiences", "II — distributed reactivation",
        "Retrieve experiences that match the current distributed cue and reinstate their associated details.",
        "Medial-temporal and hippocampal systems can reinstate contextual episodes; episodic retrieval supports rather than replaces distributed semantic representation.",
        (
            _op("episodic_cue_similarity", "S_i = cosine(cue, episode_i)", "Compare available population cues with each stored episode.", "Hippocampal pattern completion is associated with cue-based episodic retrieval."),
            _op("episodic_retrieval_probability", "P_i = softmax(β(S_i + familiarity_i))", "Normalize episode evidence into competing retrieval weights.", "Relative retrieval strength is a computational approximation of competition among memory traces."),
            _op("episodic_memory_strength", "M_i = P_i × durability_i", "Weight episode reinstatement by retrieval probability and supplied durability.", "Memory traces differ in strength and accessibility across experiences."),
        )),
    _stage(9, "Establish recurrent connections between the active populations", "II — distributed interaction",
        "Allow active populations to continuously influence one another until activity stabilizes or a bounded iteration limit is reached.",
        "Recurrent synaptic connectivity supports reciprocal interactions among distributed cortical populations without requiring all signals to pass through an ATL hub.",
        (
            _op("recurrent_network_update", "x[t+1] = φ(W x[t] + b)", "Update each population from its local activity and incoming population activity.", "Recurrent excitatory and inhibitory synapses drive ongoing population dynamics."),
            _op("bidirectional_population_communication", "u_r = Σ_s W_rs x_s; u_s = Σ_r W_sr x_r", "Permit reciprocal information flow between population pairs.", "Reciprocal anatomical and functional pathways support bidirectional influence."),
            _op("recurrent_refinement", "x[t+1] = (1−λ)x[t] + λ φ(Wx[t])", "Damp repeated updates to prevent unstable activation.", "Membrane time constants and recurrent inhibition regulate network dynamics."),
            _op("convergence_measurement", "Δ = ||x[t+1] − x[t]||", "Measure whether the distributed activation pattern is stabilizing.", "Stable population activity is a model-level analogue of settling network dynamics."),
        )),
    _stage(10, "Apply local excitation and inhibition", "II — distributed interaction",
        "Regulate local activity, limit runaway excitation, and resolve competition among populations.",
        "Local excitatory and inhibitory synaptic interactions regulate firing and permit selection without a central semantic store.",
        (
            _op("local_excitatory_drive", "E_r = W_exc x_r", "Calculate locally supported excitation.", "Excitatory synapses increase postsynaptic depolarization."),
            _op("local_inhibitory_drive", "I_r = W_inh mean(x_group)", "Estimate inhibitory influence from competing activity.", "Interneurons and inhibitory synapses regulate local population responses."),
            _op("net_population_activity", "z_r = E_r − I_r", "Combine excitatory and inhibitory influences.", "Postsynaptic membrane potential reflects interacting excitatory and inhibitory currents."),
            _op("nonlinear_firing", "x_r = σ(z_r)", "Convert net drive into bounded graded activity.", "Neuronal input-output curves are nonlinear and saturating."),
            _op("divisive_normalization", "x'_r = x_r/(ε + pooled_activity)", "Scale responses by local pooled activity.", "Divisive normalization is a candidate description of population gain control."),
        )),
    _stage(11, "Dynamically adjust the contribution of different property networks", "II — distributed interaction",
        "Change the relative influence of populations according to evidence reliability and the current task while preserving separate states.",
        "Attention, signal reliability, and behavioral demands influence the strength of different property-specific cortical contributions.",
        (
            _op("population_reliability_estimation", "r_r = reliability(x_r, input_r)", "Estimate each population's reliability from supplied evidence and signal quality.", "Sensory precision and attentional gain vary across channels."),
            _op("contribution_normalization", "g_r = exp(r_r/T)/Σ_s exp(r_s/T)", "Normalize contributions across active populations.", "Competitive gain control is a computational approximation of relative attentional weighting."),
            _op("weighted_population_contribution", "x_weighted = {g_r x_r}_r", "Create weighted readouts without merging the population states.", "Attention modulates the influence of specialized populations while their activity remains distributed."),
        )),
    _stage(12, "Coordinate the timing of distributed activity", "II — distributed interaction",
        "Measure whether population activity is temporally coordinated during the same processing event.",
        "Neural oscillations, communication delays, and temporal coordination may contribute to interactions; phase coherence is a candidate model readout, not a demonstrated semantic binding mechanism.",
        (
            _op("oscillatory_phase_representation", "φ_r(t) = 2π f_r t + phase_r", "Assign an explicit model phase to each population from its activity and frequency parameter.", "Neural populations exhibit oscillatory activity at multiple timescales."),
            _op("interpopulation_phase_difference", "Δφ_rs = φ_r − φ_s", "Calculate phase offsets between population pairs.", "Relative phase describes the timing relation between oscillatory population signals."),
            _op("phase_coherence", "R = |mean(exp(i Δφ_rs))|", "Estimate pairwise phase consistency.", "Phase-locking value is a signal-analysis measure used to describe temporal coordination."),
            _op("temporal_cross_correlation", "C_rs(τ) = Σ_t x_r(t)x_s(t+τ)", "Estimate lagged correlation from the recorded activation history.", "Temporal correlation can reveal delayed interactions among population signals."),
        )),
    _stage(13, "Construct a distributed semantic cell assembly", "II — distributed interaction",
        "Represent the concept as a coordinated pattern of co-active property populations, not as a single stored vector.",
        "Coordinated neuronal populations form functional assemblies through connectivity and co-activation across distributed cortical regions.",
        (
            _op("population_coactivation", "C_rs = a_r a_s", "Measure simultaneous activation of population pairs.", "Co-active neurons and populations can form coordinated functional assemblies."),
            _op("population_correlation", "ρ_rs = Cov(x_r,x_s)/(σ_r σ_s + ε)", "Estimate correlation between population activity patterns.", "Correlated activity is a common systems-neuroscience measure of population coordination."),
            _op("distributed_assembly_activation", "A_c = {x_visual, x_auditory, ..., x_spatial}", "Retain the assembly as a collection of distinct population states.", "A concept-related cell assembly is distributed across participating populations."),
            _op("assembly_membership", "m_r = 1[mean(x_r) > θ_r]", "Identify populations contributing above a configurable activity threshold.", "Assembly membership is approximated from population recruitment."),
        )),
    _stage(14, "Complete an incomplete concept representation through associative activation", "II — distributed interaction",
        "Infer missing properties from partial input while preserving the distinction between observed and inferred features.",
        "Learned associative pathways can reinstate missing property activity from incomplete cues.",
        (
            _op("initialize_partial_state", "x[0] = x_observed", "Start pattern completion from the original observed population features.", "Partial sensory cues can initiate an incomplete distributed activity pattern."),
            _op("recurrent_pattern_completion", "x[t+1] = φ(Wx[t] + b)", "Iteratively reactivate missing feature dimensions through recurrence and memory templates.", "Associative recurrent connections can restore activity linked to incomplete cues."),
            _op("clamp_observed_features", "x_i[t+1] = x_i[0] for observed i", "Keep observed feature dimensions anchored during completion.", "This is a computational safeguard that distinguishes stimulus-driven evidence from inferred activation."),
            _op("completion_convergence", "||x[t+1] − x[t]|| < ε", "Stop completion when changes are small or the iteration budget is exhausted.", "Settling dynamics are an approximation of recurrent pattern stabilization."),
        )),
    _stage(15, "Activate category-specific distributed circuits", "III — interpretation and output",
        "Activate category patterns that match the collective property activity while allowing overlapping categories.",
        "Object and action categories recruit partially distinct but overlapping property-sensitive neural circuits.",
        (
            _op("category_feature_prototypes", "p_category,r = mean(p_concept,r)", "Build one prototype per category and population from supplied concept patterns.", "Category structure can reflect recurring property patterns across exemplars."),
            _op("category_circuit_activation", "A_c = Σ_r α_r S_r(c)", "Score category activation from separate population comparisons.", "Category-specific circuits are distributed and partially overlapping."),
            _op("distributed_category_similarity", "S(c) = Σ_r α_r cosine(x_r,p_c,r)", "Compare each population with the corresponding category pattern.", "Distributed population similarity is a model readout, not a central category code."),
        )),
    _stage(16, "Apply contextual and task-dependent semantic modulation", "III — interpretation and output",
        "Bias relevant properties according to explicit context and task demands without modifying stored concept knowledge.",
        "Control networks, including prefrontal and posterior temporal/parietal systems, can bias distributed semantic retrieval according to current goals.",
        (
            _op("task_vector_construction", "q = [q_visual, ..., q_spatial]", "Represent task relevance separately for each population.", "Goal representations in control networks bias downstream processing."),
            _op("feature_task_relevance", "r_i = similarity(feature_i, q)", "Estimate relevance of features to the supplied task.", "Attention selects information useful to the current behavioral objective."),
            _op("task_dependent_modulation", "a'_i = φ(a_i + λr_i − μd_i)", "Boost relevant features and suppress distractors.", "Top-down control modulates gain and competitive inhibition."),
            _op("context_conditioned_integration", "A(c|k,q) = F(c,k,q)", "Combine current context and task relevance in candidate scoring.", "Context affects semantic interpretation through distributed recurrent interactions."),
        )),
    _stage(17, "Distinguish competing semantic interpretations", "III — interpretation and output",
        "Compare candidate concepts using evidence pooled analytically across their respective population patterns, then quantify ambiguity.",
        "Competing distributed assemblies can co-activate; recurrent excitation, inhibition, and context influence their relative strength.",
        (
            _op("candidate_evidence_scoring", "S(c) = Σ_r α_r S_r(c)", "Aggregate population-specific evidence for each candidate.", "Distributed evidence contributes to competing candidate assemblies."),
            _op("posterior_like_normalization", "P(c|x) = softmax(β S(c))", "Normalize relative candidate scores for a readable confidence distribution.", "The probability distribution is a computational readout, not proof of Bayesian arithmetic in neurons."),
            _op("interpretation_margin", "Δ = S(c_1) − S(c_2)", "Measure the gap between the strongest and next candidate.", "Competition may resolve toward a stronger distributed activation pattern."),
            _op("interpretation_entropy", "H(C) = −Σ_c P(c) log P(c)", "Quantify uncertainty over candidate interpretations.", "Entropy is an analysis measure for uncertainty in competing hypotheses."),
        )),
    _stage(18, "Generate a semantic response or behavioral action", "III — interpretation and output",
        "Read out the distributed property pattern into a naming, description, category, decision, or action response.",
        "Distributed conceptual activity influences language production, decision, and action-planning systems outside the property populations.",
        (
            _op("distributed_feature_readout", "y = W_out {x_r}_r", "Select task-relevant features from the distributed population collection.", "Downstream language and action systems read information from distributed populations."),
            _op("concept_classification", "c* = argmax_c S(c)", "Select the highest-supported interpretation when confidence permits.", "Behavioral recognition reflects the relative strength of competing representations."),
            _op("lexical_output_distribution", "P(w_i) = softmax(z_i)", "Create word-label alternatives from available concept names.", "Language production networks map conceptual evidence to lexical forms."),
            _op("behavioral_action_selection", "m* = argmax_m Q(m|S,q)", "Select an explicitly stored typical action when the task requests action output.", "Premotor and motor planning systems convert action-related representations into behavior."),
        )),
    _stage(19, "Update distributed connections through experience", "IV — learning and resilience",
        "Use explicitly authorized experience to update co-active population connections while controlling drift.",
        "Experience-dependent synaptic plasticity changes effective coupling between co-active neural populations; simple Hebbian and STDP equations are approximations.",
        (
            _op("hebbian_coactivation_update", "Δw_ij = η a_i a_j", "Strengthen links between co-active population signals when learning is authorized.", "Hebbian plasticity captures the association between coincident pre- and postsynaptic activity."),
            _op("bounded_weight_update", "w[t+1] = clip(w[t] + Δw)", "Apply bounded connection updates.", "Synaptic strengths are biologically regulated and do not grow without limit."),
            _op("synaptic_decay_normalization", "Δw = η a_i a_j − λw", "Apply decay and normalization to reduce unbounded weight growth.", "Homeostatic plasticity and synaptic scaling help regulate network activity."),
            _op("spike_timing_dependent_update", "Δw = A+e^(−Δt/τ+) or −A−e^(Δt/τ−)", "Apply a timing-sensitive update from supplied pre/post timing differences.", "STDP describes timing-dependent synaptic plasticity under particular experimental conditions."),
        )),
    _stage(20, "Maintain redundancy and distributed representations", "IV — learning and resilience",
        "Measure overlapping feature support and connectivity so no single property population is treated as the complete concept store.",
        "Overlapping representations and recurrent connections provide alternative retrieval paths, while some properties remain relatively specialized.",
        (
            _op("overlapping_feature_coverage", "R_c = union_r F_c,r", "Count concept properties supported across participating populations.", "Distributed cortical representations overlap without requiring every feature to be present everywhere."),
            _op("damage_resilience_coverage", "Coverage = accessible_features / represented_features", "Estimate remaining feature availability after a simulated population loss.", "Redundant pathways can provide partial compensation after focal disruption."),
            _op("population_network_degree", "d_i = Σ_j 1[W_ij ≠ 0]", "Measure each population's network degree and connectivity contribution.", "Highly connected populations can influence network coordination without storing the full concept."),
        )),
    _stage(21, "Simulate localized damage and progressive semantic degradation", "IV — learning and resilience",
        "Apply counterfactual population or connection lesions and measure selective versus widespread loss of distributed semantic performance.",
        "Localized lesions can cause selective property loss; progressive damage to interconnected populations can produce broader semantic impairment. This is a network abstraction, not a disease simulator.",
        (
            _op("population_lesion_mask", "x' = D x", "Attenuate selected population activity using a lesion mask.", "A focal lesion can reduce activity in selected property-sensitive populations."),
            _op("connection_lesion_mask", "W' = D_r W D_s", "Attenuate connections while keeping unlesioned population state available.", "White-matter and cortico-cortical pathways connect distributed semantic populations."),
            _op("progressive_network_degradation", "w[t+1] = w[t] − δ[t]", "Increase lesion severity by a supplied progressive degradation parameter.", "Progressive neurodegeneration can affect increasingly broad connected networks."),
            _op("post_lesion_activity_reconstruction", "x'[t+1] = φ(W'x'[t] + b)", "Recompute residual population activity and allow surviving pathways to contribute.", "Remaining pathways can partially compensate, depending on connectivity and damage extent."),
            _op("semantic_performance_delta", "ΔQ = Q_damaged − Q_healthy", "Compare pre- and post-lesion candidate evidence and feature coverage.", "Behavioral semantic loss can be modality-selective or cross-modal as network integrity declines."),
        )),
)

STAGE_REGISTRY: tuple[StageSpec, ...] = _STAGE_DATA
ALL_OPERATIONS: tuple[OperationSpec, ...] = tuple(
    operation for stage in STAGE_REGISTRY for operation in stage.operations
)

if len(STAGE_REGISTRY) != 21:
    raise RuntimeError("Distributed-only registry must contain exactly 21 stages.")
if tuple(stage.stage_id for stage in STAGE_REGISTRY) != tuple(
    f"DS-{number:02d}" for number in range(1, 22)
):
    raise RuntimeError("Distributed-only stages must be contiguous and ordered.")


def registry_as_dicts() -> list[dict[str, Any]]:
    """Return a JSON-compatible copy of every stage and ordered sub-operation."""
    return [
        {
            **asdict(stage),
            "operations": [asdict(operation) for operation in stage.operations],
        }
        for stage in STAGE_REGISTRY
    ]


def operation_count() -> int:
    """Return the number of registered executable sub-operations."""
    return len(ALL_OPERATIONS)
