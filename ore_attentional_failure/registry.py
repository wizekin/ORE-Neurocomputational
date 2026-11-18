"""Source-ordered formal operation registry for Point 18."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class AttentionOperationSpec:
    """Describe one source-ordered Point 18 operation and its mathematical abstraction."""

    operation_id: str
    name: str
    purpose: str
    mathematical_form: str
    evidence_level: str = "distributed_biological_principles_with_computational_abstraction"


_NAMES = [
    ("establish_task_goal", "Represent the current task and weighted information requirements",
     "R_i = sim(g, x_i); normalize requirement importance"),
    ("estimate_state", "Estimate alertness, fatigue, distraction, and current availability",
     "G_t = clip(A_t * (1-alpha*F_t) * (1-beta*D_t), 0, 1)"),
    ("detect_competitors", "Detect competing internal and external inputs without discarding safety signals",
     "D_t = clip(sum_j salience_j * irrelevance_j, 0, 1)"),
    ("compute_priority", "Rank candidates by relevance, salience, novelty, history, urgency, and cost",
     "P_i = w_R R_i + w_U U_i + w_S S_i + w_N N_i + w_H H_i - w_C C_i; softmax(P/tau)"),
    ("bias_sensory_processing", "Bias sensory feature gain while preserving raw low-priority traces",
     "x_tilde[i,k] = g[i,k] * x[i,k]"),
    ("gate_working_memory", "Admit selected information under a dynamic working-memory budget",
     "maximize sum(U_i) subject to sum(C_i) <= B_t"),
    ("sustain_attention", "Track temporal engagement and stability across successive windows",
     "A[t+1] = lambda*A[t] + (1-lambda)*I[t] - gamma*F[t] - eta*D[t]"),
    ("detect_lapse", "Detect probable attentional lapses using convergent indicators and hysteresis",
     "L_t = w_1*z_RT + w_2*z_omission + w_3*z_state + w_4*z_distraction"),
    ("reconstruct_missing", "Identify missing or weak items and retrieve only source-supported content",
     "M_i = 1 - c_i"),
    ("audit_completeness", "Compare encoded information with required elements and dependencies",
     "C_combined = exp(sum_i normalized(r_i)*log(max(c_i, epsilon)))"),
    ("select_recovery", "Choose a feasible recovery action by expected benefit minus cost",
     "a* = argmax_a [E(delta C | a) - kappa*Cost(a)]"),
    ("reorient_goal", "Restore the task goal and last verified position",
     "A_goal = cosine(s_t, g)"),
    ("reencode_information", "Re-encode recovered items with source and confidence provenance",
     "h[t+1] = h[t] + rho_t*phi(e_t)"),
    ("verify_repair", "Recheck coverage, dependencies, temporal order, and residual gaps",
     "G_residual = sum_i r_i*(1-c_i)"),
    ("adapt_future_policy", "Adapt future attention allocation only from repeated contextual feedback",
     "theta[t+1] = clip(theta[t] + eta*gradient(J_t))"),
    ("preserve_uncertainty", "Retain unknowns and alternatives when recovery is impossible",
     "H(Z|E) = -sum_z P(z|E)*log(P(z|E))"),
    ("gate_downstream_release", "Release only when confidence, coverage, and critical requirements pass",
     "Release = 1 iff C_combined>=theta_C and Coverage>=theta_R and critical_gaps=empty"),
]

OPERATION_REGISTRY = tuple(
    AttentionOperationSpec(
        operation_id=f"P18-{index:02d}",
        name=name,
        purpose=purpose,
        mathematical_form=formula,
    )
    for index, (name, purpose, formula) in enumerate(_NAMES, start=1)
)


def registry_as_dicts() -> list[dict[str, Any]]:
    """Return the formal registry in source order for documentation and inspection."""
    return [asdict(item) for item in OPERATION_REGISTRY]
