"""Source-ordered registry for the seventeen Point 22 operations."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Point22OperationSpec:
    """Describe one source-sequenced operation and its biological-reference interpretation."""

    operation_id: str
    name: str
    purpose: str
    mathematical_form: str
    biological_reference: str
    evidence_level: str = "distributed_systems_account_with_computational_abstraction"

    def to_dict(self) -> dict[str, Any]:
        """Return plain serializable metadata."""
        return asdict(self)


_ROWS = (
    ("receive_problem", "Identify the target quantity, output type, uncertainty, stakes, and available budget",
     "q=[q_1,...,q_m]; T in {class,frequency,probability,magnitude,cause,choice}; tau_T=f(cost)",
     "Sensory/language systems, working memory, and frontoparietal control maintain the task goal."),
    ("activate_knowledge", "Retrieve semantic knowledge, episodic examples, prototypes, and contextual expectations",
     "A_i=alpha*recency_i+beta*salience_i+gamma*fluency_i+delta*learning_i",
     "Hippocampal and medial-temporal retrieval interacts with distributed cortical semantic and valuation systems."),
    ("select_strategy", "Choose a fast strategy only when its cue assumptions fit the task and error cost",
     "H*=argmin_H(E[L|H,E]+lambda*Cost(H))",
     "Learned associations, valuation, and prefrontal control jointly support rapid strategy selection."),
    ("availability", "Estimate event likelihood from the accessibility of retrieved examples",
     "P_avail(E)=sigmoid(b+w1*N+w2*ease+w3*vividness)",
     "Distributed memory retrieval, cortical associations, attention, and salience shape example accessibility."),
    ("representativeness", "Compare the case with category prototypes without treating similarity as probability",
     "Sim(x,p)=x.p/(||x||*||p||); D_w=sum_j w_j*(x_j-p_j)^2",
     "Distributed feature representations and memory-based pattern completion support resemblance judgments."),
    ("base_rate_check", "Combine population frequencies with diagnostic likelihoods",
     "P(C|x)=P(x|C)P(C)/sum_k P(x|C_k)P(C_k)",
     "Learned statistical regularities, memory, working memory, and control contribute to prior-likelihood integration."),
    ("anchor_adjustment", "Adjust an initial numerical value toward an independently supported estimate",
     "x_hat=a+rho*(x_star-a); rho in [0,1]",
     "Parietal magnitude processing, memory, and prefrontal control contribute to numerical estimation."),
    ("fluency_monitor", "Separate ease and familiarity from independent evidence and evidential support",
     "Conf_subjective=Conf_evidence+eta*Fluency",
     "Perceptual/conceptual fluency and familiarity involve distributed cortical and medial-temporal processes plus metacognition."),
    ("sample_uncertainty", "Quantify sampling variability before generalizing from short or small samples",
     "SE(x_bar)=s/sqrt(n); SE(p)=sqrt(p*(1-p)/n)",
     "Pattern detection and working memory interact with learned statistical expectations; calibration is task-dependent."),
    ("causal_check", "Distinguish observed association from temporal precedence and identified causal effects",
     "r_xy=Cov(X,Y)/(sigma_X*sigma_Y); association alone does not identify do(X)",
     "Memory for event structure and distributed prefrontal/association networks support causal model comparison."),
    ("affect_urgency", "Model how salience, utility, and time pressure change cue weighting and checking priority",
     "W_i=exp(beta*u_i+gamma*s_i)/sum_j exp(beta*u_j+gamma*s_j)",
     "Amygdala, insula, vmPFC, striatum, and neuromodulatory pathways can modulate valuation and attention."),
    ("intuitive_judgment", "Combine accessible cues into a quick initial answer and a separate confidence estimate",
     "S_k=sum_j theta_j*c_jk; P(k)=softmax(S_k/tau)",
     "Distributed valuation, retrieval, and readout systems contribute to rapid choice and metacognitive confidence."),
    ("detect_conflict", "Detect cue disagreement, overconfidence, statistical omissions, and costly shortcuts",
     "Conflict=||P_intuitive-P_verified||_1/2; trigger=conflict>threshold or risk>threshold",
     "Anterior/medial frontal monitoring and lateral prefrontal control are implicated in conflict and control demands."),
    ("verify_evidence", "Recheck denominators, conditional probabilities, independent evidence, uncertainty, and causal alternatives",
     "Posterior odds=prior odds*likelihood ratio; interval=estimate +/- z*SE",
     "Working memory, statistical reasoning, and control networks support deliberate comparison; no single region computes Bayes."),
    ("ecological_fit", "Evaluate cue validity and out-of-sample risk before retaining or rejecting a heuristic",
     "Risk(H)=N^-1*sum_i L(y_hat_i,y_i)+lambda*Cost(H)",
     "Learning and valuation systems adapt strategy use to environmental regularities and error costs."),
    ("feedback_learning", "Update cue reliability gradually using prediction error and context-tagged outcomes",
     "delta_t=y_t-y_hat_t; w_(t+1)=w_t+alpha*delta_t*x_t",
     "Corticostriatal, dopaminergic, hippocampal, and prefrontal systems contribute to outcome learning."),
    ("store_rule_decide", "Store context and assumptions, choose by verified probability/expected loss, or defer when uncertain",
     "d*=argmin_d sum_y P(y|E)*L(d,y); calibration=Brier(y,p)",
     "Memory systems retain the episode and its context; prefrontal/valuation systems support final choice and deferral."),
)

OPERATION_REGISTRY = tuple(
    Point22OperationSpec(f"P22-{index:02d}", *row)
    for index, row in enumerate(_ROWS, start=1)
)


def registry_as_dicts() -> list[dict[str, Any]]:
    """Return all seventeen operations in exact source order."""
    return [item.to_dict() for item in OPERATION_REGISTRY]
