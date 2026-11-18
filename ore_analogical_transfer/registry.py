"""Canonical 18-stage operation registry for Point 11 relational analogical transfer."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class OperationSpec:
    """Describe one source-aligned transfer stage and its mathematical/biological interpretation."""

    operation_id: str
    name: str
    purpose: str
    mathematical_form: str
    biological_reference: str
    evidence_level: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return the operation specification as a plain dictionary."""
        return asdict(self)


_ROWS = (
    ("P11-01", "Detect need for relational transfer",
     "Inspect the target goal and uncertainty; activate analogy only as a candidate strategy, never as a forced explanation.",
     "R(s|q)=sum_i w_i*q_i*s_i; P(s|q)=softmax(beta*R)",
     "Prefrontal goal maintenance interacts with semantic and episodic retrieval; partial-cue completion may reactivate relevant memories.",
     "goal_and_cue_driven_retrieval_supported; activation equation is a model",
     ("target_graph", "goal", "context"), ("transfer_relevance", "candidate_strategy")),
    ("P11-02", "Retrieve candidate source cases",
     "Rank multiple source episodes/schemas by relational cues, goal, structure, and prior transfer relevance; retain failures and exceptions.",
     "S_retrieve=alpha*S_cue+gamma*S_goal+delta*S_structure+eta*S_success-lambda*S_irrelevant",
     "Hippocampal retrieval interacts with distributed cortical semantic knowledge; recall is reconstructive.",
     "episodic_and_semantic_retrieval_supported; weighted rank is computational",
     ("target_cues", "source_memory", "goal"), ("source_candidates", "retrieval_confidence")),
    ("P11-03", "Encode the target situation",
     "Construct target entities, quantities, observed relations, context, unknowns, and directed/causal distinctions.",
     "x_i in R^d; z_ij=(x_ij-mu_j)/(sigma_j+epsilon); mask missing features",
     "Domain-sensitive posterior cortical systems and working memory represent target content and task-relevant elements.",
     "domain_representation_supported; graph record is computational",
     ("target_observations", "domain_knowledge"), ("target_graph", "target_provenance")),
    ("P11-04", "Reconstruct source relational structure",
     "Recover source entities, role bindings, core relations, boundary conditions, and source evidence status.",
     "A_ij^(k)=typed adjacency; E_core={r:w_r>=tau}",
     "Hippocampal-cortical retrieval can reassemble distributed episodic features; semantic knowledge helps interpret relations.",
     "relational_memory_supported; graph extraction is computational",
     ("retrieved_source", "episode_lineage"), ("source_graph", "core_relations")),
    ("P11-05", "Separate relational from surface similarity",
     "Score what entities look like separately from what roles they play; penalize distracting surface-only matches.",
     "Sim_total=rho*Sim_surface+(1-rho)*Sim_relational; S_contrast=S_rel-kappa*S_surface_distraction",
     "Prefrontal control supports task-relevant selection and interference reduction while perceptual systems supply attributes.",
     "surface_vs_structure_distinction_supported; score is computational",
     ("source_graph", "target_graph", "goal"), ("surface_scores", "relational_scores")),
    ("P11-06", "Abstract relations into role-independent forms",
     "Canonicalize domain-specific predicates while preserving arity, argument order, direction, causal status, and quantitative form.",
     "Sim(r_s,r_t)=cos(v_s,v_t); strict type/direction gates precede semantic similarity",
     "Relations and relational integration are represented by distributed systems; a universal symbolic ontology is an engineering device.",
     "relational_abstraction_supported_at_functional_level",
     ("source_relations", "target_relations", "relation_ontology"), ("canonical_relation_signatures")),
    ("P11-07", "Generate candidate entity correspondences",
     "Compare functional and causal roles, not just labels; allow partial mappings and leave unmatched source elements unmapped.",
     "C_ij=a*Sim_role+b*Sim_relation+c*Sim_attribute+d*Sim_goal; partial assignment search",
     "Working memory maintains alternatives; prefrontal networks integrate correspondences and hippocampus supports relational retrieval.",
     "relational_mapping_supported; assignment solver is computational",
     ("candidate_entity_pairs",), ("partial_entity_mappings", "local_confidence")),
    ("P11-08", "Test relational consistency",
     "Check argument order, direction, causality, sequence, and quantitative constraints; penalize contradictions but separate unmatched links from refutation.",
     "Score(M)=sum C_ij+mu*sum W(r_s,r_t)-nu*Contradictions(M); SC=preserved_core/eligible_core",
     "Rostrolateral prefrontal regions contribute to relational integration; parietal and lateral prefrontal systems support relation representation and maintenance.",
     "higher_order_relational_integration_supported; mapping score is computational",
     ("partial_mappings", "source_relations", "target_relations"), ("consistent_mappings", "structural_consistency")),
    ("P11-09", "Integrate higher-order relational patterns",
     "Group connected first-order relations and represent dependencies such as enables, constrains, regulates, precedes, and counteracts.",
     "Coherence=sum w_r*Preserve(r)+xi*Connectedness-zeta*Conflict",
     "Rostrolateral prefrontal cortex is repeatedly implicated in integrating relations into higher-order structures.",
     "higher_order_integration_supported; exact neural code remains unknown",
     ("consistent_mappings", "higher_order_relations"), ("coherent_relational_alignment")),
    ("P11-10", "Retrieve and bind relational fragments",
     "Retrieve episodes that share structure despite different surface content; combine only compatible fragments and preserve provenance.",
     "x_hat=f_retrieve(q,W_memory); Conf=direct+alpha*support-beta*completion_uncertainty",
     "Hippocampus contributes to binding and retrieval of relational episodes; it is not a dedicated analogy engine.",
     "hippocampal_relational_memory_supported; retrieval confidence model is computational",
     ("partial_target_cues", "episodic_memory", "semantic_memory"), ("retrieved_relation_fragments", "source_lineage")),
    ("P11-11", "Control working-memory load and interference",
     "Prioritize relations by goal relevance, causal centrality, evidence, and distraction; maintain only a capacity-bounded active set.",
     "P(r)=w1*Goal+w2*Causal+w3*Evidence-w4*Interference; maximize sum P(r) subject to sum cost(r)<=B",
     "Dorsolateral prefrontal systems support relation maintenance; ventrolateral prefrontal systems contribute interference control; parietal systems represent relations.",
     "working_memory_and_control_supported; budget is an engineering approximation",
     ("active_relations", "goal", "capacity_budget"), ("maintained_relation_set", "suppressed_distractors")),
    ("P11-12", "Generate candidate target inferences",
     "Transfer source relations not already in the target only as hypotheses, with an explicit correspondence chain and scope caveats.",
     "P(h_T|M,G_S,G_T); Conf(h)=Conf(M)*Rel(rule_S)*Fit(target_conditions)",
     "Relational integration, working-memory manipulation, and source knowledge contribute to inference generation.",
     "analogical_inference_supported; confidence product is heuristic",
     ("source_relations", "selected_mapping", "target_graph"), ("transfer_hypotheses", "support_chains")),
    ("P11-13", "Check constraints and disanalogies",
     "Test target-domain material, mechanism, scale, boundary conditions, and causal dynamics; reject or qualify unsupported extensions.",
     "P(h|e,M) proportional to P(e|h,M)*P(h|M); score=support+evidence-lambda*violations-omega*assumptions",
     "Executive control and domain-specific semantic knowledge contribute to evaluating analogy limits.",
     "domain_constraint_evaluation_supported; policy score is computational",
     ("transfer_hypotheses", "target_constraints", "domain_knowledge"), ("qualified_hypotheses", "disanalogies")),
    ("P11-14", "Adapt transferred knowledge to the target",
     "Substitute target entities and translate quantities using target-specific equations, units, and parameter bridges.",
     "theta_T=T(theta_S,phi_T); dimensional analysis and unit-consistency checks",
     "Domain-specific semantic knowledge combines with abstract relations and current target evidence.",
     "domain_adaptation_is_required; parameter bridge is explicit software policy",
     ("qualified_hypotheses", "parameter_bridges", "target_domain_laws"), ("target_explanation", "adapted_parameters")),
    ("P11-15", "Evaluate predictions against evidence",
     "Compare observable consequences with later target evidence; distinguish accurate mapping from lucky answer and untested from false.",
     "delta=y-y_hat; L=-log P(y_observed|model); Brier=(p-y)^2",
     "Prediction/outcome comparison can recruit learning and control systems; multiple learning signals operate at different levels.",
     "prediction_error_learning_supported; scalar loss is computational",
     ("target_predictions", "later_evidence"), ("validation_metrics", "prediction_error")),
    ("P11-16", "Revise mapping or reject transfer",
     "Localize the mismatch, revise correspondences or scope, search alternative sources, and stop when constraints hold or transfer is unsupported.",
     "M[t+1]=argmax_M[Score(M)-lambda*Conflict(M|evidence_t)]; stop if abs(delta_score)<epsilon",
     "Recurrent comparison and control are consistent with interactive distributed processing; stopping rule is model-defined.",
     "iterative_revision_is_plausible; exact cycle is computational",
     ("mapping", "validation_feedback", "alternative_sources"), ("revised_mapping", "no_transfer_decision")),
    ("P11-17", "Abstract a reusable analogy schema",
     "Compare validated transfers across domains, remove domain-specific labels, and retain conditions, examples, and exceptions.",
     "Frequency(r)=k/n; p_hat=(k+alpha)/(n+alpha+beta)",
     "Repeated relational learning can contribute to generalized knowledge in interaction with episodic memory.",
     "schema_abstraction_supported_at_functional_level; induction rule is computational",
     ("successful_transfers", "counterexamples"), ("conditional_analogy_schema", "reliability")),
    ("P11-18", "Consolidate transfer knowledge",
     "Store the full episode and boundaries; strengthen reusable schema knowledge only as justified while retaining episodic context and exceptions.",
     "theta[t+1]=theta[t]+eta*c*(y-y_hat); W_r(t)=sum q_i*exp(-lambda*(t-t_i))",
     "Hippocampal and cortical memory systems interact during reactivation and longer-term integration; sleep and later experience can influence consolidation.",
     "episodic_semantic_complementarity_supported; accumulator is a memory-model approximation",
     ("target_episode", "mapping", "validation", "schema_version"), ("retrievable_episode", "updated_conditional_schema")),
)


OPERATION_REGISTRY = tuple(
    OperationSpec(
        operation_id=row[0], name=row[1], purpose=row[2], mathematical_form=row[3],
        biological_reference=row[4], evidence_level=row[5], inputs=row[6], outputs=row[7],
    )
    for row in _ROWS
)


def registry_as_dicts() -> list[dict[str, object]]:
    """Return the source-aligned 18-stage registry in canonical order."""
    return [item.to_dict() for item in OPERATION_REGISTRY]
