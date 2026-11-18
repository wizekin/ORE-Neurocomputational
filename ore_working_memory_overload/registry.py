"""Source-ordered Point 19 operation registry."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Point19OperationSpec:
    """Describe one algorithmic operation, its formula, and its biological reference."""

    operation_id: str
    name: str
    purpose: str
    mathematical_form: str
    biological_reference: str
    evidence_level: str = "distributed_biological_function_with_engineering_abstraction"

    def to_dict(self) -> dict[str, Any]:
        """Return the operation definition as a plain dictionary."""
        return asdict(self)


_ROWS = (
    ("establish_goal", "Establish the current comprehension goal", "w_f = softmax(alpha_g*G_f + alpha_c*C_f + alpha_r*R_f)", "Distributed prefrontal and parietal goal maintenance and attentional selection."),
    ("segment_stream", "Segment incoming language into candidate units", "P(z|x,C) proportional to P(x|z)P(z|C)", "Auditory or visual language systems with lexical and prosodic boundary sensitivity."),
    ("encode_features", "Encode lexical, grammatical, semantic, and source features", "x_i = f_lex(u_i,C,G)", "Distributed temporal lexical-semantic and phonological representations."),
    ("build_partial_structure", "Build an incremental structure and open dependencies", "G_(t+1) = G_t union {v_i,e_i,*}", "Frontotemporal language interactions maintaining incomplete constituents and relations."),
    ("maintain_dependencies", "Maintain only still-needed representations and relations", "A_i(t)=B_i+R_i(t)-I_i(t)-D_i(t)", "Active maintenance, attentional refreshing, and cue-dependent reactivation."),
    ("estimate_load", "Estimate task demand without a universal item-count limit", "L = sum_j beta_j*load_j - chunk_relief - context_support", "Distributed control and working-memory networks under task-dependent demand."),
    ("chunk_information", "Group explicitly related elements while preserving internal links", "N_eff=N_unbound+sum_G gamma_G", "Knowledge-supported chunking through interacting temporal, parietal, and frontal systems."),
    ("allocate_attention", "Balance new processing and maintenance priorities", "a_i = softmax(v_i/tau); maximize sum(v_i*a_i)-lambda*Cost(a)", "Frontoparietal attention and executive-control interactions."),
    ("estimate_interference", "Estimate similarity-based competition among active items", "I_i=sum_(j!=i) sim(x_i,x_j)*A_j*g_j", "Cue overlap and competitive retrieval across distributed representations."),
    ("retrieve_by_cue", "Retrieve earlier material using diagnostic cues", "P(i|q)=softmax(tau*M(i,q))", "Associative retrieval and medial-temporal pattern completion as functional analogies."),
    ("integrate_retrieved", "Integrate supported retrievals into the evolving event model", "E_(t+1)=E_t+eta_t*DeltaE_t", "Language-network unification with semantic/event integration."),
    ("detect_overload", "Detect overload from converging behavioral and demand signals", "O_t=sigmoid(theta_0+theta_1*L+theta_2*T+theta_3*R+theta_4*E-theta_5*C)", "Performance monitoring and control networks; no single signal is diagnostic."),
    ("identify_at_risk", "Rank items and relations at risk of becoming inaccessible", "R_i=w_1*(1-A_i)+w_2*I_i+w_3*D_i+w_4*U_i", "Distributed maintenance and retrieval monitoring."),
    ("select_recovery", "Select recovery actions by expected benefit minus added cost", "a*=argmax_a[E(accuracy_gain|a)-lambda*load(a)-mu*time(a)]", "Prefrontal control over reactivation, rehearsal, reanalysis, and clarification."),
    ("diagnose_failure", "Distinguish encoding, maintenance, retrieval, binding, and integration failures", "z*=argmax_z P(z|observed_evidence)", "Memory monitoring, source monitoring, and controlled retrieval."),
    ("reconstruct_supported", "Reconstruct only from surviving evidence and label inference", "P(h|e,C) proportional to P(e|h,C)P(h|C)", "Schema-guided retrieval and hippocampal-cortical reconstruction, kept defeasible."),
    ("check_coherence", "Check global grammatical, semantic, and discourse consistency", "Q=alpha*G+beta*S+gamma*D-delta*X", "Distributed language, semantic, and control systems."),
    ("commit_partial", "Commit a revisable interpretation with confidence and provenance", "I={supported,inferred,unknown,confidence,source}", "Temporary maintenance and event-model integration; not automatic durable episodic storage."),
    ("adapt_difficulty", "Adapt future processing priorities from observed difficulty", "beta_(t+1)=clip(beta_t+rho*error_signal*driver,0,1)", "Experience-dependent control and learning across multiple biological timescales."),
)

OPERATION_REGISTRY = tuple(
    Point19OperationSpec(f"P19-{index:02d}", name, purpose, formula, biology)
    for index, (name, purpose, formula, biology) in enumerate(_ROWS, start=1)
)


def registry_as_dicts() -> list[dict[str, Any]]:
    """Return all 19 operations in source order for documentation and diagnostics."""
    return [item.to_dict() for item in OPERATION_REGISTRY]
