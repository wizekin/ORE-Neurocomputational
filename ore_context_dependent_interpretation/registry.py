"""Canonical 16-stage operation registry for Point 12 context-dependent interpretation."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class OperationSpec:
    """Describe one source-aligned operation and its mathematical and biological interpretation."""

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
    ("P12-01", "Detect and encode the incoming event",
     "Encode linguistic or environmental input while preserving modality, explicit facts, source reliability, and event identity.",
     "x_t=Encode(input_t); F_t={lexical, perceptual, temporal, speaker, explicit_claims}",
     "Distributed perceptual and language-responsive systems provide the initial event representation; the feature record is an engineering abstraction.",
     "input_encoding_supported; feature schema is computational",
     ("incoming_event",), ("encoded_event", "input_provenance")),
    ("P12-02", "Retrieve the immediately preceding discourse state",
     "Retrieve recent turns, referents, topic state, and unresolved discourse commitments; down-weight stale or topic-incompatible records.",
     "R_D=TopK(sim(q,d_i)*reliability_i*exp(-lambda*age_i)*topic_fit_i)",
     "Language and temporoparietal systems integrate prior discourse; hippocampal and working-memory systems contribute to maintaining relevant event history.",
     "discourse_context_influence_supported; retrieval score is computational",
     ("encoded_event", "discourse_history"), ("active_discourse", "topic_context")),
    ("P12-03", "Retrieve relevant long-term knowledge",
     "Retrieve semantic and episodic knowledge cued by the event, while retaining source identity, confidence, and known exceptions.",
     "R_K=TopK(sim(q,k_i)*confidence_i*reliability_i*goal_fit_i)",
     "Temporal semantic systems and medial temporal memory interact during knowledge access; ranked retrieval is a model.",
     "knowledge_guided_comprehension_supported; ranking is computational",
     ("encoded_event", "long_term_knowledge"), ("retrieved_knowledge", "knowledge_provenance")),
    ("P12-04", "Estimate the current task goal",
     "Use explicit task instructions and maintained task state to estimate what information is relevant; do not invent a speaker or user intention.",
     "G_t=Select(G_explicit,G_maintained,G_event_features); source_weight_s=f(G_t,s)",
     "Medial prefrontal and lateral prefrontal control systems contribute to goal/context maintenance and selection.",
     "goal_sensitive_processing_supported; goal selector is computational",
     ("task_instruction", "maintained_goal", "event_features"), ("current_goal", "goal_source_preferences")),
    ("P12-05", "Extract physical and perceptual scene context",
     "Extract relevant scene cues and observed entities; distinguish visual or environmental evidence from inferred scene content.",
     "E_t=EncodeScene(scene); relevance(e)=sim(e,q)*reliability(e)",
     "Visual, auditory, and action-related systems provide grounded scene evidence that can interact with conceptual interpretation.",
     "perceptual_context_influence_supported; cue representation is computational",
     ("scene_input", "current_event"), ("scene_evidence", "scene_provenance")),
    ("P12-06", "Retrieve cultural, conventional, and shared knowledge",
     "Retrieve conventions and speaker-linked information only to the extent that they plausibly apply; do not treat a group convention as universal.",
     "C_t=Retrieve(conventions | cue, goal, speaker, applicability)",
     "Temporal semantic and social-cognitive networks can contribute conventional and speaker-context knowledge.",
     "conventional_context_influence_supported; applicability score is computational",
     ("event_cues", "convention_memory", "speaker_model"), ("applicable_conventions", "speaker_context")),
    ("P12-07", "Construct a provisional situation model",
     "Combine supported entities, relations, events, goals, and open questions while retaining the provenance of each element.",
     "M_t=Integrate(D_t,K_t,E_t,C_t,G_t); facts carry provenance and confidence",
     "Posterior midline, angular-gyrus, temporal, and medial temporal systems contribute to situation and event-model construction.",
     "situation_model_construction_supported; explicit object graph is computational",
     ("active_discourse", "retrieved_knowledge", "scene_evidence", "goal"), ("situation_model", "open_questions")),
    ("P12-08", "Generate candidate interpretations",
     "Generate multiple plausible meanings from the input and retrieved context, preserving an open-set alternative instead of forcing a closed choice.",
     "H={h_1,...,h_n, h_other}; H=Generate(input, context, goal)",
     "Semantic activation and controlled selection can jointly make alternatives available; explicit candidate enumeration is a software model.",
     "context_sensitive_alternatives_supported; generator is computational",
     ("encoded_event", "situation_model", "context_evidence"), ("candidate_hypotheses")),
    ("P12-09", "Estimate prior plausibility from context",
     "Estimate a context-conditioned prior from relevant discourse, knowledge, scene, convention, and speaker evidence.",
     "P(h|C,G)=softmax(ContextFit(h,C,G)/T)",
     "Distributed predictive and memory systems support context-sensitive expectations; softmax is a model, not neural arithmetic.",
     "contextual_expectation_supported; probability model is computational",
     ("candidate_hypotheses", "context_evidence", "goal"), ("context_priors", "prior_provenance")),
    ("P12-10", "Evaluate incoming evidence against each candidate",
     "Score direct linguistic or perceptual evidence separately from contextual expectation so context cannot silently replace the input.",
     "L(h)=P(x|h,C); retain independent input_fit(h) and context_fit(h)",
     "Language and perceptual systems analyze incoming evidence while control networks can maintain competing interpretations.",
     "input_context_interaction_supported; likelihood is computational",
     ("candidate_hypotheses", "encoded_event"), ("candidate_likelihoods", "input_evidence_scores")),
    ("P12-11", "Resolve competition while preserving uncertainty",
     "Combine prior and likelihood, rank alternatives, and abstain when posterior, margin, or uncertainty criteria do not support commitment.",
     "P(h|x,C,G) proportional to P(x|h,C)*P(h|C,G); H=-sum p(h)log p(h)",
     "Inferior frontal and broader control systems may contribute to selection and conflict handling; Bayesian updating and entropy are abstractions.",
     "contextual integration_and_uncertainty_supported; exact calculation is computational",
     ("context_priors", "candidate_likelihoods"), ("posterior_candidates", "selection_or_ambiguity")),
    ("P12-12", "Integrate the selected interpretation into the situation model",
     "Integrate high-confidence interpretation and explicit evidence; record inferred content as inferred and preserve open alternatives.",
     "M_(t+1)=Update(M_t,x_t,h_t); provenance(h)=candidate_inference",
     "Posterior midline and temporoparietal systems contribute to event-model integration; explicit provenance tracking is an engineering safeguard.",
     "incremental_situation_update_supported; data structure is computational",
     ("selected_candidate", "explicit_input", "current_situation"), ("updated_situation", "unresolved_items")),
    ("P12-13", "Check coherence and detect context conflicts",
     "Compare new claims with maintained facts and context; localize contradictions without automatically rejecting surprising but possible events.",
     "Conflict(k)=1[f_old(k) != f_new(k)]; keep source and time of both claims",
     "Semantic control, episodic comparison, and conflict monitoring can contribute to detecting inconsistencies.",
     "coherence_monitoring_supported; mismatch rule is computational",
     ("updated_situation", "incoming_facts", "evidence_ledger"), ("conflicts", "coherence_diagnostics")),
    ("P12-14", "Revise the interpretation when new evidence requires it",
     "Reweight alternatives against explicit disambiguating evidence, revise provisional commitments, and retain the revision trail.",
     "P_(t+1)(h) proportional to P_t(h)*exp(-lambda*conflict(h)); renormalize",
     "Recurrent processing and memory updating support reinterpretation; this explicit revision rule is computational.",
     "reinterpretation_and_revision_supported; penalty rule is computational",
     ("posterior_candidates", "new_evidence", "conflicts"), ("revised_candidates", "revision_status")),
    ("P12-15", "Adapt contextual weighting for the next input",
     "Update source weights cautiously from observed relevance while preventing one mistaken interpretation from permanently dominating later context.",
     "w_s(t+1)=clip(w_s(t)+eta*(support_s- baseline),0,1)",
     "Attention and control systems flexibly change the influence of information sources; scalar update is a model.",
     "flexible_contextual_selection_supported; update equation is computational",
     ("validated_interpretation", "source_support"), ("adapted_context_weights")),
    ("P12-16", "Produce a task-appropriate interpretation or action",
     "Return a concise selected meaning, alternatives, or clarification request according to the current goal and uncertainty.",
     "output=SelectTaskOutput(G, posterior, uncertainty, conflicts)",
     "Language production, decision, and action systems use task-relevant comprehension states; output policy is computational.",
     "goal_conditioned_comprehension_supported; output policy is computational",
     ("posterior_candidates", "goal", "uncertainty", "situation_model"), ("interpretation_or_action", "confidence", "provenance")),
)


OPERATION_REGISTRY = tuple(
    OperationSpec(
        operation_id=row[0], name=row[1], purpose=row[2],
        mathematical_form=row[3], biological_reference=row[4],
        evidence_level=row[5], inputs=row[6], outputs=row[7],
    )
    for row in _ROWS
)


def registry_as_dicts() -> list[dict[str, object]]:
    """Return the source-ordered 16-stage Point 12 operation registry."""
    return [item.to_dict() for item in OPERATION_REGISTRY]
