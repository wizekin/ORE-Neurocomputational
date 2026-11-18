"""Canonical source-order registry for Point 09 schema formation.

The first eighteen entries cover the learning and use algorithm. The remaining entries expose
the source's end-to-end pseudocode, mathematical inventory, biological mapping, validation,
failure safeguards, and integrated result summary as inspectable operations.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class OperationSpec:
    """Describe one source-aligned operation and its mathematical and biological scope."""

    operation_id: str
    name: str
    purpose: str
    mathematical_form: str
    biological_reference: str
    evidence_level: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a plain dictionary suitable for registries, annotations, and logs."""
        return asdict(self)


_ROWS = (
    ("P09-01", "Define the schema-learning problem",
     "Define reusable situation structure as context, roles, slots, relations, constraints, and goals.",
     "S=(C,R,T,K,G)",
     "Distributed hippocampal, medial-temporal, prefrontal, and cortical systems contribute to schema knowledge.",
     "cognitive_abstraction_supported; explicit frame is a computational model",
     ("experience_set", "learning_objective"), ("schema_target",)),
    ("P09-02", "Segment experiences into ordered events",
     "Detect event boundaries and retain temporal order, local context, and links among sub-events.",
     "B_t = weighted change across perception, goals, actions, context, and temporal gap",
     "Event segmentation recruits distributed perceptual and memory systems, including hippocampal context processing.",
     "event_segmentation_supported; weighted scalar score is computational",
     ("multimodal_episode",), ("ordered_event_fragments", "boundary_evidence")),
    ("P09-03", "Encode episode-specific entities and relations",
     "Bind entities, attributes, actions, event context, and directed relations while preserving episode identity.",
     "G_i=(V_i,E_i,A_i); softmax(M) is one optional role-assignment surrogate",
     "Hippocampal and medial-temporal systems support relational and episodic binding.",
     "episodic_relational_binding_supported; graph and matrix are abstractions",
     ("event_fragments", "entities", "relations"), ("episode_graphs", "episode_identity")),
    ("P09-04", "Group recurring situation classes",
     "Compare context, goals, event sequences, role structure, and relation patterns across episodes.",
     "Sim(E_i,E_j)=weighted feature, goal, relation, event, and context similarity",
     "Memory reactivation and medial-temporal/prefrontal interactions may support integration across experiences.",
     "cross_episode_integration_supported; threshold clustering is computational",
     ("episode_graphs",), ("candidate_episode_groups", "similarity_evidence")),
    ("P09-05", "Identify recurring functional roles",
     "Infer what each participant contributes to event and goal structure rather than relying only on labels.",
     "Score(e,r)=weighted action, relation, goal, and context compatibility",
     "Distributed conceptual, episodic, and control networks can encode who-did-what relations.",
     "role_like_structure_is_computationally_plausible; no universal role register established",
     ("episode_entities", "events", "relations", "goals"), ("role_assignments", "ambiguity")),
    ("P09-06", "Abstract roles into variable slots",
     "Replace episode-specific entities with reusable placeholders and estimate allowable filler types.",
     "F(s_k)={f_1,...,f_m}; P(f|s_k,C) from reliability-weighted evidence",
     "Stable relational positions may arise from distributed population codes and learned associations.",
     "slot_abstraction_is_cognitive_model; literal neural slots are not established",
     ("role_assignments", "entity_types"), ("schema_slots", "filler_distributions")),
    ("P09-07", "Extract typed relations among slots",
     "Align episode relations to slots, retain direction, and estimate support using only eligible episodes.",
     "Conf(rho)=(N_support+a)/(N_eligible+a+b)",
     "Hippocampal relational binding and prefrontal integration contribute to relational memory.",
     "relational_learning_supported; typed graph and Beta smoothing are computational",
     ("episode_relations", "role_to_slot_maps"), ("typed_schema_relations", "relation_confidence")),
    ("P09-08", "Learn temporal transitions and alternative paths",
     "Estimate context-sensitive event transitions while distinguishing frequent order from mandatory precedence.",
     "P(e_j|e_i,C)=(N(i->j,C)+lambda)/(N(i,C)+lambda*|J|)",
     "Hippocampal sequence representations and prefrontal expectations can contribute to event models.",
     "sequence_and_prediction_systems_supported; Markov estimate is a model",
     ("ordered_events", "context"), ("transition_probabilities", "branching_paths")),
    ("P09-09", "Separate association from causal evidence",
     "Keep co-occurrence, functional dependence, and causal hypotheses as different evidential categories.",
     "Delta=P(Y|X)-P(Y|not X); Delta alone does not identify causal effect",
     "Hippocampal-prefrontal relational inference is relevant; causal processing is distributed.",
     "causal_status_separation_is_methodological; conditional contrast is non-causal",
     ("typed_relations", "event_order", "outcomes"), ("evidence_status", "causal_uncertainty")),
    ("P09-10", "Separate invariant from incidental structure",
     "Estimate regularity, predictive value, goal relevance, and context dependence without erasing rare diagnostic cases.",
     "U(f)=eta1*stability+eta2*prediction+eta3*goal_relevance-eta4*context_penalty",
     "Hierarchical learning and dimensionality reduction are theoretical accounts of abstraction.",
     "abstraction_supported_at_cognitive_level; utility score is computational",
     ("slot_support", "relation_support", "outcomes"), ("core_typical_optional_properties")),
    ("P09-11", "Construct the integrated relational frame",
     "Combine slots, directed relations, transitions, constraints, goals, confidence, and source episodes.",
     "S=(V_s,E_r,P_t,C_s,G_s,Theta)",
     "Schema knowledge is distributed; no single brain region stores a complete explicit frame.",
     "integrated_schema_is_computational_abstraction",
     ("slots", "relations", "transitions", "goals"), ("schema_frame", "evidence_links")),
    ("P09-12", "Compose hierarchical and reusable schemas",
     "Relate broader and narrower schemas and identify reusable role/relation components.",
     "S_child -> S_parent + DeltaS_child; score=log P(E|S)-kappa*complexity(S)",
     "Hippocampal-prefrontal and cortical systems may support hierarchical structured knowledge.",
     "hierarchical_schema_learning_is_active_research; set inclusion is computational",
     ("schema_frames",), ("parent_child_links", "reusable_components")),
    ("P09-13", "Bind goals, roles, and expected outcomes",
     "Connect participant goals and action alternatives to supported outcomes.",
     "Q(s,a,g)=E[R_g+gamma*V(s')|s,a,g]",
     "Orbitofrontal and medial prefrontal systems are candidates for goal and outcome structure.",
     "goal_and_outcome_processing_supported; Q-value is reinforcement-learning analogy",
     ("roles", "events", "goals", "outcomes"), ("goal_conditioned_frame",)),
    ("P09-14", "Generate context-conditioned predictions",
     "Rank likely roles, fillers, relations, and next events using schema evidence and context.",
     "P(S|O,C) proportional to P(O|S,C)*P(S|C)",
     "Schemas can guide retrieval, attention, expectation, and interpretation.",
     "schema_guided_prediction_supported; Bayesian expression is computational",
     ("schema_frame", "partial_observation", "context"), ("ranked_predictions",)),
    ("P09-15", "Fill gaps while preserving provenance",
     "Generate plausible completions but retain inferred status, uncertainty, and source distinction.",
     "f_hat=argmax_f P(f|slot,S,C,O); uncertainty from posterior spread",
     "Prior knowledge can shape recall and interpretation, with risks of source-memory errors.",
     "schema_completion_supported; explicit provenance tags are software safeguards",
     ("partial_observation", "slot_distributions"), ("inferred_items", "confidence", "provenance")),
    ("P09-16", "Compute prediction error for new episodes",
     "Align new observations to a schema and quantify event, role, and relation mismatch.",
     "delta=o-o_hat; categorical surprise=-log P(o|S,C)",
     "Prediction-error and novelty signals can influence learning and episodic encoding.",
     "prediction_error_relevant; scalar surprise is computational",
     ("new_episode", "schema_predictions"), ("mismatch", "surprise", "counterevidence")),
    ("P09-17", "Integrate, specialize, revise, or split",
     "Update consistent evidence, specialize recurring variants, and preserve highly incongruent episodes separately.",
     "theta_new=(1-eta)*theta_old+eta*x; Bayesian count update is an alternative",
     "Hippocampal-prefrontal interaction may support schema-consistent integration and incongruent encoding.",
     "schema_updating_supported; threshold policy is computational",
     ("schema", "new_episode", "mismatch"), ("updated_schema", "specialized_or_separate_frame")),
    ("P09-18", "Preserve flexibility and exceptions",
     "Track eligible counts, counterexamples, context shift, and confidence to prevent rigid overgeneralization.",
     "P(rho|C)=(N+alpha)/(N_eligible+alpha+beta); KL(P_recent||P_stored)",
     "Schemas aid prediction but can bias memory; updating is context- and task-dependent.",
     "flexibility_and_false_memory_risk_supported; divergence alarm is computational",
     ("schema_evidence", "counterexamples", "recent_context"), ("calibrated_schema", "review_signal")),
    ("P09-19", "Execute the end-to-end schema-learning sequence",
     "Coordinate segmentation, episodic binding, role/slot induction, frame construction, prediction, and revision.",
     "episodes -> groups -> roles -> slots -> relations -> transitions -> frames -> updates",
     "The end-to-end cascade is a software organization, not a literal neural serial program.",
     "integrated_pipeline_is_engineering_architecture",
     ("episode_collection",), ("schema_collection", "operation_trace")),
    ("P09-20", "Expose the mathematical operation inventory",
     "Expose the estimators and measures used for binding, similarity, role assignment, inference, and update.",
     "similarity; softmax; Beta smoothing; transition probability; mutual information; KL divergence",
     "These are candidate mathematical tools, not claims of explicit neuronal arithmetic.",
     "mathematical_inventory_is_model_documentation",
     ("operation_trace",), ("equation_inventory", "parameter_diagnostics")),
    ("P09-21", "Simulate interacting biological network roles",
     "Model hippocampal episodic binding, cortical co-activation, prefrontal control, and orbitofrontal goals.",
     "w_new=clip(w+eta*(coactivation-w)); context and goal gain modulate schema activation",
     "Network roles are distributed and task-dependent; rate/associative models are simplifications.",
     "network_contributions_supported_at_broad_level; implementation_is_surrogate",
     ("episodes", "schema_frames", "goals"), ("biological_state", "network_diagnostics")),
    ("P09-22", "Expose validation measures",
     "Provide measurable outputs for role generalization, relation preservation, sequence prediction, calibration, and source memory.",
     "slot accuracy; relation F1; transition log-loss; calibration; false-inference rate",
     "Behavioral and neural validation require separate empirical experiments.",
     "validation_metrics_are_testable_proposals",
     ("predictions", "held_out_episodes"), ("validation_metrics")),
    ("P09-23", "Check known failure modes and safeguards",
     "Detect surface matching, role ambiguity, causal overclaim, overgeneralization, and source-confusion risks.",
     "risk flags from ambiguity, novelty, missing evidence, and confidence thresholds",
     "Known cognitive risks motivate safeguards; risk flags are engineering diagnostics.",
     "failure_modes_are_model_tests_not_neural_localizations",
     ("schema", "prediction", "mismatch"), ("risk_flags", "safeguards")),
    ("P09-24", "Summarize schema state and evidence",
     "Return a revisable relational model with source episode links, uncertainty, predictions, and lineage.",
     "S=(roles,slots,relations,transitions,goals,context,confidence,exceptions)",
     "The frame is a research-level computational representation of distributed knowledge.",
     "integrated_output_is_computational_abstraction",
     ("schema_frames", "diagnostics"), ("schema_result", "provenance", "summary")),
)


OPERATION_REGISTRY = tuple(
    OperationSpec(
        operation_id=row[0], name=row[1], purpose=row[2], mathematical_form=row[3],
        biological_reference=row[4], evidence_level=row[5], inputs=row[6], outputs=row[7],
    )
    for row in _ROWS
)


def registry_as_dicts() -> list[dict[str, object]]:
    """Return the canonical 24-stage source-aligned Point 09 registry."""
    return [item.to_dict() for item in OPERATION_REGISTRY]
