"""Canonical source-ordered registry for the 14 operations of Point 02."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class OperationSpec:
    """Describe a Point 02 operation and its mathematical and biological interpretation."""

    operation_id: str
    source_step: int
    name: str
    purpose: str
    mathematical_form: str
    biological_reference: str
    evidence_level: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return the operation record as a plain dictionary."""
        return asdict(self)


_ROWS = (
    (1, "Prepare incoming information and establish context",
     "Identify modality, task, goal, active working-memory context, and capacity constraints before retrieval.",
     "P(C_j|X,G) ∝ P(X|C_j,G)P(C_j|G); g=σ(αA+βW−γL)",
     "Sensory/language systems provide input; prefrontal goal maintenance and distributed attention/working-memory networks maintain task context.",
     "contextual_control_supported; probability and capacity equations are computational proxies",
     ("new_input", "current_context", "learner_state"), ("context_frame", "processing_gate")),
    (2, "Extract concepts, features, events, and typed relations",
     "Create a structured representation while preserving qualifiers, negation, source text, and confidence.",
     "x_i=w_i f_i; P(f)=1−∏_k(1−p_k); r=(subject,predicate,object,qualifiers)",
     "Distributed semantic/language representations contribute conceptual features and relations; explicit tuples are software structures.",
     "semantic_feature_and_relation_processing_supported; extraction heuristics are approximate",
     ("context_frame", "new_input"), ("new_representation", "entities", "features", "relations")),
    (3, "Generate multiple retrieval cues",
     "Construct semantic, functional, relational, contextual, and contrastive cues and weight their distinctiveness.",
     "U(q_i)=salience_i×relevance_i×distinctiveness_i; normalized cue weights",
     "Partial feature activation and recurrent association can provide multiple routes to stored representations.",
     "cue_dependent_retrieval_supported; utility score is an engineering heuristic",
     ("new_representation", "context_frame"), ("retrieval_cues",)),
    (4, "Activate and retrieve candidate prior memories",
     "Spread activation through bounded graph associations and retrieve semantic, episodic, and schematic candidates.",
     "a_(t+1)=φ(λWa_t+u); A_j=Σ_i q_iw_ij; P(M_j|Q)=softmax(A_j/T)",
     "Hippocampal pattern completion and distributed cortical activation interact with prefrontal retrieval control.",
     "pattern_completion_and_associative_retrieval_supported; matrix and softmax are model abstractions",
     ("retrieval_cues", "knowledge_network"), ("candidate_memories", "activation_sources")),
    (5, "Evaluate and rank candidate knowledge",
     "Rank by relevance, reliability, contextual compatibility, specificity, and evidential support rather than familiarity alone.",
     "R_j=αActivation+βRelevance+γReliability+δContextFit−λSuperficialOverlapPenalty",
     "Prefrontal control and distributed semantic systems may bias retrieval toward task-relevant representations.",
     "controlled_retrieval_supported; weighted ranking is computational",
     ("candidate_memories", "new_representation", "context_frame"), ("ranked_candidates", "candidate_uncertainty")),
    (6, "Compare new information at surface and relational levels",
     "Identify shared features, functional complementarity, relational overlap, mismatches, and explicit contradictions.",
     "Sim=weighted(Jaccard_features, relation_match, function_match); X=Σ_i χ_i·conflict_i",
     "Distributed semantic coding, hippocampal relational retrieval, and frontoparietal comparison contribute to comparison.",
     "similarity_and_relational_comparison_supported; similarity is not proof of truth or causation",
     ("new_representation", "ranked_candidates", "knowledge_relations"), ("comparison_profile", "contradictions")),
    (7, "Align relational structures and generate bounded analogies",
     "Map roles between domains, reward preserved relations, record unmatched elements, and prevent analogy limits from being ignored.",
     "S_map(f)=Σ_(u,r,v) ω_r Match((f(u),r,f(v)),E); Q_analogy=S_map−mapping_cost",
     "Analogical reasoning recruits distributed prefrontal/parietal systems and retrieval from temporal/medial-temporal memory systems.",
     "relational_analogy_supported; graph mapping score is a computational model",
     ("comparison_profile", "ranked_candidates"), ("analogy_maps", "transferred_hypotheses", "mapping_limits")),
    (8, "Estimate schema fit, novelty, and prediction error",
     "Distinguish schema-consistent, complementary, ambiguous, conflicting, and highly novel information.",
     "I(X)=−log P(X|S,C); δ=x−x_hat; D_KL(Q||P); F=Σw_i Compat_i−Σv_j Conflict_j",
     "Medial prefrontal and hippocampal systems contribute to schema use, novelty processing, and distinctive episodic encoding.",
     "schema_congruence_and_novelty_supported; scalar fit and KL are model abstractions",
     ("new_representation", "schemas", "comparison_profile"), ("fit_state", "novelty", "prediction_error")),
    (9, "Choose an integration strategy without erasing conflict",
     "Select integration, schema completion, candidate revision, separate episodic storage, or deferred decision.",
     "EU(a)=Σ_o P(o|a,X)U(o)−Cost(a); η=η_0 g(evidence,novelty,conflict)",
     "Multiple memory systems and prefrontal monitoring can support differentiated treatment of congruent and incongruent input.",
     "integration_strategy_is_computational_policy; conflict preservation is a scientific safeguard",
     ("fit_state", "source_reliability", "comparison_profile"), ("integration_plan",)),
    (10, "Bind information to concepts, relations, and contextual cues",
     "Create source-linked associations while retaining whether each relation was observed, reported, inferred, or hypothesized.",
     "Δw_ij=ηa_ia_j; W_new=W_old+ηaaᵀ; M=(X,C,T,G,R)",
     "Hippocampal relational binding interacts with distributed cortical semantic representations and synaptic plasticity.",
     "episodic_binding_and_associative_plasticity_supported; outer-product rule is simplified",
     ("new_representation", "integration_plan", "retrieval_cues"), ("bound_memory", "bidirectional_links")),
    (11, "Update schemas cautiously and preserve exceptions",
     "Modify only affected schema components when evidence quality and repeated support justify it; never overwrite automatically.",
     "θ_new=(1−η)θ_old+ηx; θ_new=(τ_oldθ_old+τ_newx)/(τ_old+τ_new)",
     "mPFC-cortical interactions and hippocampal memory integration may support schema refinement over time.",
     "schema_updating_supported; precision-weighted update is a normative approximation",
     ("bound_memory", "schema_fit", "evidence_quality"), ("updated_schema_candidate", "schema_lineage")),
    (12, "Encode unresolved novelty as a distinct episode",
     "Preserve surprising or schema-inconsistent information with source, time, context, violated expectation, and uncertainty.",
     "D(X)=αSurprisal+βUniqueness+γContextNovelty; StoreEpisode=I(X>θ_X or I>θ_I)",
     "Hippocampal/medial-temporal systems contribute to distinctive episodic encoding; novelty-related neuromodulation may modulate learning.",
     "episodic_encoding_and_novelty_relevant; distinctiveness equation is computational",
     ("new_representation", "fit_state", "context_frame"), ("new_episode", "expectation_error")),
    (13, "Consolidate and reorganize connections over time",
     "Revisit useful links, strengthen supported associations, weaken misleading ones, and abstract common structure without erasing exceptions.",
     "Δw_ij=ηr_ijc_ij; g=Σ_iω_ix_i/Σ_iω_i; w(t+1)=ρw(t)+Δw",
     "Hippocampal-cortical reactivation, replay, sleep-associated processes, and distributed cortical integration are candidate contributors.",
     "systems_consolidation_and_replay_supported; slow-weight model is an abstraction",
     ("bound_memory", "new_episode", "association_state"), ("consolidated_network", "consolidation_record")),
    (14, "Test retrieval routes and refine the knowledge network",
     "Probe links from multiple cues, score recovered relations, correct false retrievals, and refine fragile routes.",
     "S_ret=correct_elements/(required_elements+ε); E_cal=|confidence−accuracy|; H(Q)=−Σp(q)log p(q)",
     "Retrieval reactivates hippocampal/cortical representations and can itself update memory; feedback supports correction.",
     "retrieval_practice_and_memory_modification_supported; software route test is not human recall",
     ("retrieval_routes", "bound_memory", "evidence"), ("retrieval_diagnostics", "refined_associations")),
)

OPERATION_REGISTRY = tuple(
    OperationSpec(f"P02-{index:02d}", *row) for index, row in enumerate(_ROWS, start=1)
)


def registry_as_dicts() -> list[dict[str, object]]:
    """Return the canonical 14-stage Point 02 registry as dictionaries."""
    return [item.to_dict() for item in OPERATION_REGISTRY]
