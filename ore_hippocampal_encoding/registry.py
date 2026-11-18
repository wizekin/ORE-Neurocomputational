"""Canonical operation registry for Point 01 rapid hippocampal episodic encoding.

The ordered operations follow Point_01_Rapid_Hippocampal_Encoding.md. Equations are executable
model abstractions or diagnostics, not assertions that neurons literally manipulate symbols,
matrices, cosine distances, graphs, or scalar novelty values.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class OperationSpec:
    """Describe one encoding operation, including inputs, equations, biology, and limitations."""

    operation_id: str
    name: str
    purpose: str
    mathematical_form: str
    biological_reference: str
    evidence_level: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return the operation definition as a plain dictionary for tracing and documentation."""
        return asdict(self)


_OPERATION_ROWS = (
    ("HE01", "Detect and segment the incoming event",
     "Separate a candidate episode from ongoing background while preserving preceding context.",
     "D_t = 0.65 d(x_t,x_{t-1}) + 0.25 d(c_t,c_{t-1}) + 0.10 g_t; boundary = D_t > theta.",
     "Distributed sensory and association cortex, attention/control systems, and hippocampal sensitivity to contextual transitions; no single boundary detector is assumed.",
     "distributed_mechanism; scalar change score is a model choice",
     ("sensory_stream", "prior_event", "context", "goals"), ("event_boundary", "candidate_event")),
    ("HE02", "Extract relevant features in parallel sensory systems",
     "Transform modality-specific inputs into feature-bearing population vectors while preserving channel identity.",
     "x_m = f_m(s_m); x = concatenate(normalize(x_m) for each observed modality m).",
     "Visual, auditory, somatosensory, motor, language, and association networks process features in parallel.",
     "strong_general_support; exact feature transforms are task-specific",
     ("sensory_signals",), ("modality_feature_vectors", "integrated_sensory_vector")),
    ("HE03", "Select and weight information by attention and relevance",
     "Allocate relative gain to attended, salient, goal-relevant channels while retaining uncertainty.",
     "a_m = softmax((goal_m + salience_m + 1 - uncertainty_m)/tau); x'_m = n a_m x_m.",
     "Distributed frontoparietal, thalamic, sensory, and neuromodulatory gain control; softmax is an engineering surrogate.",
     "gain_modulation_supported; exact normalized rule is a model choice",
     ("feature_vectors", "goals", "salience", "uncertainty"), ("attention_weights", "attended_features")),
    ("HE04", "Build item and object representations",
     "Combine high-level feature conjunctions into item-specific codes and preserve supplied item identities.",
     "i_k = phi(W_v x_v + W_a x_a + W_l x_l + b); pooled item code = weighted feature aggregation.",
     "Perirhinal cortex and its interactions with lateral entorhinal cortex contribute to complex item/object representations.",
     "anatomical_and_functional_support; explicit vector is a model representation",
     ("attended_features", "component_features"), ("item_codes", "item_representation")),
    ("HE05", "Build scene, spatial, and contextual representations",
     "Represent setting and relative scene structure so event items are not encoded as disconnected elements.",
     "c_s = f_PHC(scene, spatial, context); r_ij = p_j - p_i when explicit position codes exist.",
     "Parahippocampal and posterior-medial systems contribute scene/contextual information to medial entorhinal pathways.",
     "anatomical_and_functional_support; context dimensions are caller/model defined",
     ("context_vector", "scene_features", "spatial_relations"), ("context_representation")),
    ("HE06", "Represent temporal order and event sequence",
     "Tag the event with time and preserve the order of supplied components and transitions.",
     "q(t) = [sin(omega_k t), cos(omega_k t)]_{k=1..K}; T_ij = P(s_(t+1)=j | s_t=i).",
     "Hippocampal-entorhinal population dynamics, oscillatory timing, and cortical interactions support temporal/sequential coding; no single code is assumed.",
     "broad_support; sinusoidal basis is a computational analogy",
     ("timestamp_ms", "ordered_components"), ("temporal_representation", "transition_sequence")),
    ("HE07", "Converge inputs through entorhinal cortex",
     "Combine item, contextual, and temporal streams while retaining separable channel segments.",
     "z = concatenate(w_i normalize(i), w_c normalize(c), w_t normalize(q)).",
     "Lateral and medial entorhinal streams provide interacting, partially differentiated inputs to the hippocampal formation.",
     "anatomical_support; exact concatenation is a model choice",
     ("item_representation", "context_representation", "temporal_representation"), ("entorhinal_code")),
    ("HE08", "Estimate novelty and distinctiveness",
     "Compare the present entorhinal code with stored episode codes without treating novelty as a binary save signal.",
     "N = 1 - max_j cosine(z, E_j); N = 1 when no prior comparison record is available.",
     "Novelty/mismatch depends on hippocampal, cortical, dopaminergic, noradrenergic, and attentional interactions; cosine distance is a model metric.",
     "family_of_processes_supported; scalar novelty metric is a model choice",
     ("entorhinal_code", "prior_episode_codes"), ("novelty_score", "nearest_episode")),
    ("HE09", "Perform dentate-gyrus sparse pattern separation",
     "Use competitive activation and inhibition to make overlapping experiences more distinguishable.",
     "u = ReLU(W_DG z); h_DG = topK(u,k); k = ceil(rho n_DG).",
     "Dentate granule-cell sparsity and local excitation/inhibition support discrimination of similar inputs; top-k implements a circuit-level surrogate.",
     "substantial_circuit_support; top-k is a computational approximation",
     ("entorhinal_code", "novelty_score"), ("sparse_dentate_code", "active_granule_units")),
    ("HE10", "Link co-active features in CA3",
     "Form an autoassociative event representation using DG drive, direct entorhinal drive, and recurrent activity.",
     "h_CA3 = sigmoid(W_DG h_DG + W_EC z + gamma W_rec h_prev); W_rec <- W_rec + eta h h^T.",
     "CA3 receives DG and entorhinal inputs and contains recurrent collaterals implicated in associative encoding and later pattern completion.",
     "strong_anatomical_support; autoassociative interpretation is influential, not exclusive",
     ("dentate_code", "entorhinal_code", "prior_ca3_activity"), ("ca3_association_code", "recurrent_association")),
    ("HE11", "Integrate CA3 and direct entorhinal input in CA1",
     "Combine associative CA3 activity with direct cortical/entorhinal input and quantify model discrepancy.",
     "h_CA1 = tanh(W_3 h_CA3 + W_EC z); e = ||direct_EC - mapped_CA3||_2 / sqrt(n).",
     "CA1 receives CA3-mediated Schaffer collateral input and direct entorhinal temporoammonic input; simple comparator equations are abstractions.",
     "strong_anatomical_support; comparator framing is task/model dependent",
     ("ca3_code", "entorhinal_code"), ("ca1_integrated_code", "prediction_error_estimate")),
    ("HE12", "Induce rapid synaptic plasticity at active connections",
     "Update selected CA3 recurrent weights according to co-activity, calcium proxy, and modulatory state.",
     "C_ij = pre_i post_j; deltaW = +eta C if Ca>theta_LTP, -eta C if theta_LTD<Ca<=theta_LTP, else 0.",
     "Activity-dependent LTP/LTD, NMDA-receptor/calcium signalling, CaMKII-related processes, and neuromodulators influence hippocampal synapses.",
     "strong_synaptic_evidence; thresholds and calcium proxy are model assumptions",
     ("pre_activity", "post_activity", "calcium_proxy", "modulators"), ("updated_weights", "plasticity_report")),
    ("HE13", "Construct a relational event index",
     "Bind item, context, time, and caller-provided relations to a distinctive hippocampal event code.",
     "G=(V,E); index = {h_E <-> item_i, context_j, time_k, relation_l}.",
     "Hippocampal relational binding/indexing is a major theoretical framework; explicit symbolic graph storage is only an engineering representation.",
     "strong_theoretical_framework; symbolic graph is not claimed to be neuronal storage format",
     ("ca1_code", "item_nodes", "context_node", "time_node", "relations"), ("event_graph", "hippocampal_index")),
    ("HE14", "Route hippocampal output into the wider network",
     "Relay the integrated episode representation toward subicular, entorhinal, and cortical networks through a recurrent model update.",
     "x_cortex(t+1) = normalize((1-lambda)x_cortex(t) + lambda W_HC h_CA1).",
     "Subicular and entorhinal output routes interact with distributed cortical and subcortical systems; this is not slow systems consolidation.",
     "network_level_support; uniform feedback operator is a model choice",
     ("ca1_code", "previous_cortical_state"), ("cortical_feedback_state")),
    ("HE15", "Assess availability of the initial event-specific trace",
     "Estimate distinctiveness, relational binding, plasticity, and population stability without asserting permanent storage.",
     "Q_E = w_N N + w_B B + w_P P + w_S S; persistence is not inferred from Q_E.",
     "Initial encoding and durable consolidation are distinct; later repetition-dependent cortical learning and sleep replay are excluded.",
     "conceptual_boundary_strong; quality score is an engineering diagnostic",
     ("novelty", "event_graph", "plasticity_report", "ca1_activity"), ("initial_trace", "trace_quality", "consolidation_status_not_evaluated")),
)

OPERATION_REGISTRY: tuple[OperationSpec, ...] = tuple(
    OperationSpec(
        operation_id=row[0], name=row[1], purpose=row[2], mathematical_form=row[3],
        biological_reference=row[4], evidence_level=row[5], inputs=row[6], outputs=row[7],
    )
    for row in _OPERATION_ROWS
)


def registry_as_dicts() -> list[dict[str, object]]:
    """Return all 15 canonical source-ordered operations for tests, documentation, and logging."""
    return [item.to_dict() for item in OPERATION_REGISTRY]
