"""Canonical 16-operation registry for Point 20 reconstructive memory."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ReconstructionOperationSpec:
    """Describe one source-aligned operation and its mathematical and biological interpretation."""

    operation_id: str
    name: str
    purpose: str
    mathematical_form: str
    biological_reference: str
    evidence_level: str = "distributed_biological_function_with_computational_abstraction"

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-data operation definition for traces and documentation."""
        return asdict(self)


_ROWS = (
    ("define_retrieval_target", "Detect retrieval demand and define the question",
     "r_i = sim(q, f_i) * a_i; retain target type and precision",
     "Goal-directed retrieval involves lateral prefrontal control, temporal semantic systems, and medial-temporal memory interactions."),
    ("reactivate_episodic_fragments", "Reactivate the strongest available episodic fragments",
     "A_j = sum_i(w_ij * c_i); P(M_j|c) = softmax(beta * A_j)",
     "Hippocampal pattern completion interacts with distributed cortical sensory, spatial, temporal, and relational reinstatement."),
    ("separate_retrieved_from_missing", "Separate recovered evidence from missing information",
     "e_i in {supported, contradicted, unknown, not_queried}; missing != absent",
     "Retrieval monitoring depends on hippocampal and prefrontal interactions; absent retrieval is not proof of absent encoding."),
    ("activate_candidate_schemas", "Activate relevant schemas, scripts, and cultural knowledge",
     "Z_k = alpha*sim(S_k,E) + gamma*fit(S_k,C) + eta*G_k; w_k=softmax(Z_k/tau)",
     "Medial prefrontal and temporal semantic systems contribute to schema activation and context-sensitive knowledge use."),
    ("generate_inferred_completions", "Generate candidate completions and label them as inferences",
     "P(h|S,C) proportional to P(h|S)*P(h|C); retain top-K alternatives",
     "Schema-consistent expectations can bias distributed semantic and event representations; generated completions are not episodic evidence."),
    ("combine_evidence_and_priors", "Combine episodic evidence and prior expectations cautiously",
     "P(h|E,S,C) proportional to P(E|h,S,C)*P(h|S,C); logL=lambda_E*logP(E|h)+lambda_S*logP(h|S,C)-lambda_X*X",
     "Bottom-up episodic reinstatement interacts recurrently with top-down schema expectations and controlled evaluation."),
    ("check_event_constraints", "Test candidates against temporal, spatial, and causal constraints",
     "C(h)=sum_m omega_m*phi_m(h,G), with explicit penalties for violated constraints",
     "Hippocampal relational representations interact with prefrontal temporal, spatial, and causal reasoning."),
    ("monitor_candidate_sources", "Perform source monitoring for each candidate detail",
     "P(s|f) proportional to P(f|s)*P(s); assign a source only above threshold",
     "Lateral prefrontal evaluation interacts with hippocampal contextual reinstatement and sensory feature information."),
    ("separate_familiarity_and_recollection", "Detect familiarity without treating it as proof",
     "D(h)=theta_F*F + theta_R*R - theta_U*U; autobiographical precision increases theta_R",
     "Familiarity and recollection have partly separable medial-temporal and cortical contributions; neither is an infallible truth signal."),
    ("search_competing_episodes", "Search for contradictory evidence and competing episodes",
     "Q_k=sum_i(a_i*e_ik)-sum_j(b_j*x_jk); missing cue is not contradiction unless encoded",
     "Hippocampal pattern separation/retrieval interacts with prefrontal monitoring to reduce cross-episode blending."),
    ("calibrate_detail_confidence", "Estimate confidence independently of coherence",
     "Conf(h)=sigmoid(beta_0+beta_1*E+beta_2*R-beta_3*I-beta_4*X-beta_5*U)",
     "Confidence monitoring draws on distributed memory, monitoring, and control systems; the logistic function is a calibration model."),
    ("select_reporting_policy", "Choose whether to report, qualify, clarify, or omit",
     "a*=argmin_a sum_h P(h|E,S,C)*Loss(a,h)",
     "Prefrontal control and goal-dependent decision systems regulate response caution and clarification seeking."),
    ("monitor_retelling_provenance", "Monitor retelling and post-event information",
     "G_p=(V_source union V_detail, E_parent, E_transform); preserve timestamp and source lineage",
     "Repeated retrieval and post-event information can alter later remembering; reconsolidation is context- and task-dependent."),
    ("update_from_reliable_feedback", "Update the reconstruction from reliable corrective evidence",
     "PosteriorOdds(h)=PriorOdds(h)*P(F|h)/P(F|not h); independent evidence groups only",
     "Memory updating involves hippocampal-cortical interaction and prefrontal evaluation of feedback reliability."),
    ("update_general_schema_separately", "Update general schema knowledge separately from a particular episode",
     "p_hat_(t+1)=p_hat_t+eta*(y_t-p_hat_t); beta-binomial counts only verified evidence",
     "Repeated reliable experiences can update cortical semantic/schema knowledge without rewriting an individual hippocampal episode."),
    ("finalize_provenance_aware_report", "Finalize a provenance-aware reconstruction and retain uncertainty",
     "R=(supported,inferred,source_uncertain,contradicted,alternatives,confidence,provenance)",
     "Distributed episodic, semantic, sensory, and control systems contribute to the final report; the record keeps their evidence classes distinct."),
)

OPERATION_REGISTRY = tuple(
    ReconstructionOperationSpec(
        operation_id=f"P20-{index:02d}",
        name=name,
        purpose=purpose,
        mathematical_form=formula,
        biological_reference=biology,
    )
    for index, (name, purpose, formula, biology) in enumerate(_ROWS, start=1)
)


def registry_as_dicts() -> list[dict[str, Any]]:
    """Return all 16 operations in source order for inspection and validation."""
    return [item.to_dict() for item in OPERATION_REGISTRY]
