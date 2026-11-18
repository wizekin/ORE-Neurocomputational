"""Source-ordered registry for the twenty Point 24 corrective-learning operations."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Point24OperationSpec:
    operation_id: str
    name: str
    purpose: str
    mathematical_form: str
    biological_reference: str
    evidence_level: str = "distributed_learning_memory_and_control_network_abstraction"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_ROWS = (
    ("preserve_knowledge_state", "Record the current interpretation, evidence, source, context, and confidence",
     "M_t=(H_t,E_t,C_t,S_t,q_t)",
     "Distributed cortical knowledge, hippocampal episode binding, and prefrontal goal maintenance."),
    ("generate_predictions", "Generate testable consequences and predictions that distinguish alternatives",
     "P(y|H_i,C); y_hat=sum_i P(H_i)*E[y|H_i]",
     "Experience-dependent cortical predictions interact with memory and attention systems."),
    ("detect_mismatch", "Compare predicted and observed information without treating surprise as proof of error",
     "delta=y-y_hat; surprise=-log P(y_obs); D_KL(P_obs||P_pred)",
     "Medial frontal monitoring and salience/attention systems can respond to unexpected input."),
    ("classify_error", "Keep competing perceptual, semantic, relational, factual, inferential, source, confidence, and procedure causes",
     "P(z|x,H,C)",
     "Control and memory networks support diagnosis; no single error-classification centre is assumed."),
    ("select_information_action", "Choose internal inference, clarification, independent checking, or a discriminating test",
     "EIG(a)=H(H|D)-E_r[H(H|D,r,a)]",
     "Prefrontal control maintains goals and selects an information-seeking action."),
    ("acquire_clarification", "Identify the highest-value question or check and preserve uncertainty when no answer is available",
     "Utility(a)=EIG(a)-lambda*Cost(a)",
     "Attention and language systems support targeted question formation and cue selection."),
    ("evaluate_feedback", "Weight evidence by reliability, relevance, diagnosticity, directness, and independence",
     "w_e=r_e*s_e*d_e*m_e; group duplicate sources before updating",
     "Memory/source monitoring and control processes evaluate provenance and credibility."),
    ("reactivate_prior_interpretation", "Retrieve the earlier belief while retaining its provenance and false-or-unverified status",
     "M_old=(H_old,source_old,t_old)",
     "Hippocampal retrieval interacts with prefrontal monitoring; retrieval can strengthen correct or incorrect content."),
    ("generate_corrected_candidates", "Construct evidence-supported alternatives without inventing facts or unsupported hypotheses",
     "H_candidates={H_i supplied or explicitly derived from evidence}",
     "Frontotemporal language and semantic association systems contribute when the error is linguistic or conceptual."),
    ("compare_hypotheses", "Update competing interpretations using evidence-weighted Bayesian likelihoods",
     "P(H_i|E,C) proportional to P(E|H_i,C)P(H_i|C)",
     "Distributed memory and learning systems change relative activation with new evidence."),
    ("update_knowledge_components", "Edit only affected relations or features and preserve unaffected knowledge",
     "theta_(t+1)=theta_t+eta*r*g*Delta",
     "Synaptic plasticity and hippocampal-cortical interactions can support selective updating; exact mechanisms vary."),
    ("build_relational_explanation", "Explain the corrected relation, scope, conditions, examples, non-examples, and boundaries",
     "G=(V,E,R)",
     "Semantic association and frontotemporal integration support relational conceptual structure."),
    ("retrieve_corrected_understanding", "Test unaided retrieval rather than relying on rereading or recognition",
     "A(t)=A0*exp(-lambda*t); A_(k+1)=A_k+alpha*R_k*(1-A_k)",
     "Hippocampal-cortical retrieval and prefrontal monitoring support recall and error detection."),
    ("give_specific_retrieval_feedback", "Identify residual errors and keep the old belief labelled as superseded",
     "e_k=1[response_k != target]; feedback_k=target-response_k",
     "Error monitoring, attention, and memory updating interact after a retrieval attempt."),
    ("test_transfer", "Test structurally related new cases, counterexamples, and changed contexts",
     "y_hat=f_theta(z(x)); Transfer=correct_probes/total_probes",
     "Hippocampal relational memory and cortical semantic knowledge contribute to generalization."),
    ("calibrate_confidence", "Compare reported confidence with verified accuracy over repeated outcomes",
     "Brier=(1/N)*sum_i(p_i-o_i)^2",
     "Metacognitive monitoring interacts with control and memory; confidence is not a direct accuracy signal."),
    ("consolidate_correction", "Strengthen the corrected representation while preserving useful episode and source context",
     "S_(t+1)=S_t+beta*I_t*R_t*C_t",
     "Hippocampal-cortical interaction, reactivation, and sleep-related processes can support consolidation."),
    ("monitor_belief_regression", "Check delayed accessibility of the correction against renewed activation of the old belief",
     "B_(t+1)=B_t-gamma*M_c(t)+rho*F_old(t)",
     "Memory accessibility, familiarity, and retrieval interact; regression is not inevitable."),
    ("evaluate_closed_loop", "Return to the earliest failed stage or stop only when evidence, retrieval, and transfer meet task thresholds",
     "Stop=1 iff U<tau_U, Q>tau_Q, T>tau_T",
     "Prefrontal control and metacognitive evaluation support task-dependent continuation decisions."),
    ("store_and_validate_outcome", "Persist provenance, uncertainty, diagnostics, and the complete functional/biological trace",
     "L=lambda1*L_recall+lambda2*L_transfer+lambda3*L_calibration",
     "Distributed memory and learning systems retain outcomes; SQLite logging is an engineering instrumentation layer."),
)

OPERATION_REGISTRY = tuple(
    Point24OperationSpec(f"P24-{index:02d}", *row)
    for index, row in enumerate(_ROWS, start=1)
)


def registry_as_dicts() -> list[dict[str, Any]]:
    """Return all twenty operations in source order."""
    return [item.to_dict() for item in OPERATION_REGISTRY]
