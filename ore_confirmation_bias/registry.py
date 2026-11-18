"""Source-ordered Point 21 confirmation-bias operation registry."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Point21OperationSpec:
    """Document one computational operation and its biological-reference interpretation."""

    operation_id: str
    name: str
    purpose: str
    mathematical_form: str
    biological_reference: str
    evidence_level: str = "distributed_biological_function_with_engineering_abstraction"

    def to_dict(self) -> dict[str, Any]:
        """Return the operation metadata as plain data."""
        return asdict(self)


_ROWS = (
    ("initialize_belief", "Define the current belief state", "P(H_k)=prior_k; sum_k P(H_k)=1; L_k=log(P(H_k))", "Distributed memory retrieval and medial-prefrontal valuation/control contribute prior knowledge and confidence."),
    ("build_expectations", "Activate predictions for the focal and alternative hypotheses", "T_(k,f)=P(f|H_k); DeltaT_f=T_(focus,f)-mean_alt(T_(k,f))", "Recurrent prefrontal-to-sensory/association modulation can prioritize expected diagnostic features."),
    ("select_sources", "Select information sources without allowing agreement alone to determine utility", "U(s)=D(s)R(s)+aC(s)+bA(s)+cT(s)-dK(s); P(s)=softmax(U/tau)", "Valuation, memory accessibility, social evaluation, and executive selection interact."),
    ("allocate_attention", "Allocate processing to evidence features using relevance and expected uncertainty reduction", "G_i=G_0+alpha R_i+gamma S_i+delta U_i+eta C_i; a_i=softmax(G_i/tau)", "Frontoparietal control modulates sensory and association representations; attention and encoding remain distinct."),
    ("encode_provenance", "Encode evidence content with source, time, context, and evidential direction", "M_i=(x_i,s_i,c_i,d_i,q_i,t_i); strength_i=f(attention,novelty,measurement)", "Hippocampal relational binding interacts with distributed cortical content and source/context representations."),
    ("interpret_evidence", "Generate and rank competing interpretations of ambiguous observations", "P(I_k|E) proportional to P(E|I_k)P(I_k)^rho", "Semantic, episodic, contextual, and control systems jointly constrain interpretation."),
    ("assess_quality", "Estimate reliability, diagnosticity, measurement quality, and dependence", "w_i=r_i*d_i*m_i*a_i/sqrt(n_group)", "Metacognitive/source monitoring and control systems evaluate provenance and uncertainty."),
    ("compare_symmetrically", "Evaluate evidence for every hypothesis using the same likelihood scale", "ell_(i,k)=log P(E_i|H_k)-mean_j log P(E_i|H_j)", "Distributed evidence representations are compared under common decision criteria."),
    ("readout_evidence", "Convert encoded evidence into judgment while measuring congruence-dependent readout", "L_k=log P(H_k)+sum_i g_i*w_i*ell_(i,k)", "Parietal decision evidence and prefrontal readout/control can assign unequal weights to represented evidence."),
    ("update_confidence", "Update belief and confidence according to weighted evidence and a bounded learning rate", "P*_k=softmax(L_k); B_(t+1)=(1-lambda)B_t+lambda P*", "Distributed valuation, metacognitive confidence, and control networks update decision state."),
    ("retrieve_memory", "Retrieve evidence while preserving original content and source provenance", "R_i=M_i*cue_i*(1+mu*(2C_i-1))", "Hippocampal pattern completion and cortical reinstatement interact with current semantic expectations."),
    ("detect_feedback", "Identify recurrent loops among source choice, attention, interpretation, readout, and memory", "B_(t+1)=F(B_t,S_t,A_t,I_t,R_t); bias_gap=congruent_weight-incongruent_weight", "Recurrent interactions among valuation, attention, memory, and control can stabilize a belief."),
    ("apply_corrective_control", "Pre-register predictions, generate alternatives, and select diagnostic disconfirmation tests", "IG(s)=H(H|E)-E_e[H(H|E,e)]; s*=argmax_s IG(s)*reliability(s)/cost(s)", "Controlled reasoning, working memory, and metacognitive monitoring support deliberate evidence search."),
    ("validate_stages", "Measure selection, attention, interpretation, readout, confidence, and memory bias separately", "bias_stage=mean(metric|congruent)-mean(metric|incongruent); BS=N^-1 sum(p-y)^2", "Behavioral and neural readouts must be task-specific; validation is an empirical research procedure, not a neural calculation."),
)

OPERATION_REGISTRY = tuple(
    Point21OperationSpec(f"P21-{index:02d}", *row)
    for index, row in enumerate(_ROWS, start=1)
)


def registry_as_dicts() -> list[dict[str, Any]]:
    """Return all fourteen operations in source order."""
    return [item.to_dict() for item in OPERATION_REGISTRY]
