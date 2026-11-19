"""Source-ordered operation definitions for the Point 06 cognitive-load algorithm."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Point06OperationSpec:
    """Describe one source step, its computational equation, and cautious biological reference."""

    operation_id: str
    source_step: int
    name: str
    purpose: str
    mathematical_form: str
    biological_reference: str
    evidence_level: str = "functional_computational_model_with_distributed_biological_correspondence"

    def to_dict(self) -> dict[str, Any]:
        """Return the operation definition as a plain dictionary."""
        return asdict(self)


_ROWS = (
    ("define_goal", "Define a measurable comprehension goal and the evidence required for success.",
     "R_i = sim(unit_i, goal) * importance_i; defer when R_i < threshold",
     "Distributed prefrontal and frontoparietal goal maintenance biases processing toward task-relevant information."),
    ("inspect_complexity", "Estimate element demand and interactions while accounting for relevant prior knowledge.",
     "C_task = sum(a_i) + lambda*sum(b_ij); C_eff = C_task*(1 - eta*K)",
     "Distributed cortical representations and memory systems make familiar structures easier to retrieve and integrate."),
    ("retrieve_prerequisites", "Retrieve the smallest source-supported set of prerequisites needed by the current goal.",
     "P(v) = prerequisite_closure(v); score(m)=w_r*Rel+w_p*Prereq-w_d*Distraction",
     "Cue-dependent retrieval may involve hippocampal relational completion and distributed cortical semantic representations."),
    ("segment_material", "Segment the input at meaningful semantic and structural boundaries.",
     "J(P)=sum(Coherence(B_j))-alpha*Overload(B_j)-beta*LostRelations(P)",
     "Language, sensory, attentional, and association systems contribute to detecting meaningful boundaries."),
    ("form_meaningful_chunks", "Group coherent units while preserving internal relations and reconstruction requirements.",
     "Q_chunk = reconstructed_required_relations / total_required_relations",
     "Expertise-dependent chunking uses distributed long-term knowledge interacting with active task representations."),
    ("order_by_prerequisites", "Schedule prerequisite-ready concepts before dependent concepts and preserve cycles explicitly.",
     "(v_i -> v_j) in E implies i < j for an acyclic topological order",
     "Retrieval, attention, and control support sequential access to prerequisite knowledge; the graph is an engineering model."),
    ("select_active_subgoal", "Select one principal subgoal using relevance, readiness, and stated priority.",
     "P(g_i|s)=exp(beta*U_i)/sum_j exp(beta*U_j)",
     "Frontoparietal control contributes to goal maintenance and regulation of competing task sets."),
    ("reduce_presentation_demands", "Estimate and target avoidable search, split-attention, ambiguity, and redundancy costs.",
     "L_avoidable = alpha*S + beta*N + gamma*A + delta*R",
     "Perceptual attention and executive coordination are affected by how explanatory information is arranged."),
    ("externalize_structure", "Represent concepts and explicit relations in an outline, graph, or task-appropriate map.",
     "G=(V,E,W); external_state_(t+1)=external_state_t union DeltaE_t",
     "External notes change the task environment; the brain still perceives and interprets the representation."),
    ("select_working_set", "Maintain only material required by the active subgoal and its dependencies.",
     "A_i=alpha*Relevance+beta*Dependency+gamma*Uncertainty-delta*Interference",
     "Working-memory maintenance and selection are distributed across task-relevant cortical and control systems."),
    ("monitor_overload", "Combine performance and effort indicators while distinguishing productive effort from breakdown.",
     "O_t=w_1*Error+w_2*Rereading+w_3*Switching+w_4*Fatigue+w_5*Effort",
     "Performance monitoring and cognitive control involve distributed cingulo-opercular and frontoparietal interactions."),
    ("adapt_learning_step", "Adjust pace, active-set size, and support in response to estimated utilization and comprehension.",
     "rho=L/(Capacity+epsilon); choose advance, clarify, or reduce/scaffold by calibrated thresholds",
     "Adaptive control depends on monitoring, memory access, attention reallocation, and behavioral adjustment."),
    ("provide_guided_scaffold", "Use supplied examples and fade guidance gradually as the learner demonstrates competence.",
     "Support_(t+1)=clip(Support_t-mu*Accuracy,0,1) after repeated criterion-level success",
     "Structured examples provide input for schema learning; the biological account is distributed and timescale-dependent."),
    ("integrate_chunk_globally", "Connect the current chunk to required causal, temporal, hierarchical, or contrastive relations.",
     "Q_structure=sum(w_ij*correct_ij)/sum(w_ij*required_ij)",
     "Language, semantic, memory, and control systems interact during local-to-global integration."),
    ("retrieve_and_fade_support", "Test reconstruction and reduce external support only after repeated reliable retrieval.",
     "A=correctly_reconstructed_units_and_relations/required_units_and_relations",
     "Retrieval and repeated reactivation can support memory accessibility; no single equation captures consolidation."),
    ("verify_comprehension", "Evaluate local explanation, relation accuracy, transfer, and evidence-based explanation.",
     "Q=alpha*A_local+beta*A_relations+gamma*A_transfer+delta*A_explanation",
     "Distributed self-monitoring, memory retrieval, and control support checking; confidence alone is insufficient."),
)

OPERATION_REGISTRY = tuple(
    Point06OperationSpec(f"P06-{index:02d}", index, name, purpose, formula, biology)
    for index, (name, purpose, formula, biology) in enumerate(_ROWS, start=1)
)


def registry_as_dicts() -> list[dict[str, Any]]:
    """Return the complete 16-operation registry in source order."""
    return [item.to_dict() for item in OPERATION_REGISTRY]