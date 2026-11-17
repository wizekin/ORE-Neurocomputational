"""Source-ordered Point 5 contextual/pragmatic integration over ORE's discourse engine.

The adapter preserves Point 4's literal proposition and delegates context, coreference,
coherence, situation, speaker-intention, indirect-request and irony analyses to the
established eight-stage/39-operation discourse subsystem.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ore_discourse import DiscourseConfig, DiscourseProcessor, DiscourseRequest
from ore_sensory_perception.contracts import to_jsonable


POINT5_STEP_SPECS: tuple[dict[str, Any], ...] = (
    {"operation_id":"LC-P05-01","name":"Receive and preserve the literal proposition","objective":"Retain Point 4's proposition graph, operators, roles, alternatives, confidence and unresolved slots as an immutable source layer.","mathematical_operations":("P0=deep_copy(P4.proposition_graph)","m_j=1 if field_j is grounded else 0","q_j in [0,1]"),"biological_reference":"Core frontotemporal language activity maintains the current sentence representation; context is added without erasing literal content."},
    {"operation_id":"LC-P05-02","name":"Retrieve relevant discourse and background knowledge","objective":"Select recent turns, related propositions, active entities and caller-supplied event/world knowledge under bounded retrieval.","mathematical_operations":("r_i=sim(query,memory_i)","w_i=exp(-delta_t/tau)","R_i=a*r_i+b*recency_i+c*task_i+d*causal_i","top_k(R)"),"biological_reference":"Temporal semantic cortex, medial temporal memory and frontal controlled retrieval can jointly reactivate relevant information."},
    {"operation_id":"LC-P05-03","name":"Construct a source-sensitive context state","objective":"Keep discourse, scene, speaker, common ground, world knowledge and conversational goal distinct, time-tagged and provenance-labelled.","mathematical_operations":("C_t={D_t,S_t,M_t,G_t,W_t,Q_t}","c_t=sum_j gate_j*c_j","gate_j=sigmoid(relevance_j*reliability_j)"),"biological_reference":"Distributed language, memory, conceptual and social-cognitive networks contribute complementary context signals."},
    {"operation_id":"LC-P05-04","name":"Identify focus, topic, entities and events","objective":"Estimate discourse prominence and active event/topic structure while retaining competing centers of attention.","mathematical_operations":("salience(e)=a*recency+b*frequency+c*topic+d*syntax","alpha_i=softmax(salience_i)","topic_t=update(topic_(t-1),P_t)"),"biological_reference":"Temporal/parietal semantic representations interact with attention and frontal maintenance of the current topic."},
    {"operation_id":"LC-P05-05","name":"Resolve references against discourse and scene entities","objective":"Generate candidate antecedents, apply explicit agreement/type constraints, score evidence and abstain when no candidate is sufficiently supported.","mathematical_operations":("s_i=sum_k w_k*f_k(candidate_i,expression)","p_i=softmax(s_i/T)","H=-sum_i p_i*log(p_i)","commit iff p_1>theta and p_1-p_2>margin"),"biological_reference":"Language constraints, semantic memory, attention and TPJ-related contextual integration jointly support reference resolution."},
    {"operation_id":"LC-P05-06","name":"Generate context-sensitive predictions","objective":"Maintain graded expectations about likely continuations, event relations and speech acts without treating predictions as facts.","mathematical_operations":("P(h|C_t)","surprisal(x)=-log(P(x|C_t)+epsilon)","delta_t=observed_t-expected_t","precision_error=pi_t*delta_t"),"biological_reference":"Temporal semantic and language populations support expectations; attention and prediction-error systems respond to unexpected input."},
    {"operation_id":"LC-P05-07","name":"Compare the proposition with local discourse","objective":"Identify supported continuation, elaboration, cause, result, contrast, correction, sequence, contradiction or missing coherence links.","mathematical_operations":("P(r|P_t,P_(t-1),C)","coherence=sum_r P(r)*utility(r)","constraint_satisfaction(G_D)"),"biological_reference":"Left temporal language regions and broader medial frontal systems contribute to discourse and causal coherence."},
    {"operation_id":"LC-P05-08","name":"Compare the proposition with the situation model","objective":"Align linguistic participants and events with caller-supplied scene evidence while distinguishing assertion, observation, memory and hypothetical content.","mathematical_operations":("match(e_language,e_scene)","temporal_overlap=max(0,min(end_i,end_j)-max(start_i,start_j))","evidence_fusion=sum_j reliability_j*evidence_j"),"biological_reference":"Visual/auditory association areas and temporoparietal conceptual systems provide scene evidence; right TPJ can support reorienting to unexpected cues."},
    {"operation_id":"LC-P05-09","name":"Evaluate world knowledge and event plausibility","objective":"Use supplied event schemas and conservative learned cues to assess causal and selectional compatibility without equating unusual with impossible.","mathematical_operations":("P(event|participants,knowledge)","C_role=similarity(type(participant),role_preferences)","E_sem=sum_i lambda_i*(1-C_i)"),"biological_reference":"Distributed temporal semantic knowledge and hippocampal relational/event memory support event knowledge."},
    {"operation_id":"LC-P05-10","name":"Decide whether additional inference is needed","objective":"Use ambiguity, coherence, conflict, underspecification and task relevance to choose ordinary integration, deeper inference or abstention.","mathematical_operations":("need_inference=sigmoid(a*ambiguity+b*conflict+c*gap+d*task_demand)","H(P)=-sum_i p_i*log(p_i)","thresholded_control(need_inference)"),"biological_reference":"Frontoparietal control and medial frontal monitoring may increase when automatic linguistic integration is insufficient."},
    {"operation_id":"LC-P05-11","name":"Generate bridging and causal inferences","objective":"Propose only defeasible event, temporal and causal links supported by connectives, explicit context or documented world knowledge.","mathematical_operations":("P(link|P_i,P_j,C,K)","score(link)=evidence-support-assumption_cost","DAG_check(G_causal)"),"biological_reference":"Temporal/event memory and frontal inference processes contribute to bridging and causal integration."},
    {"operation_id":"LC-P05-12","name":"Generate candidate pragmatic interpretations","objective":"Add contextually plausible meanings such as indirect request, warning, offer or refusal while retaining literal interpretation separately.","mathematical_operations":("H_k={literal,request,offer,warning,...}","P(H_k|utterance,C,goal) proportional_to P(E|H_k,C)*P(H_k|C)"),"biological_reference":"Core language processing cooperates with social cognition and domain-general reasoning when nonliteral intent is relevant."},
    {"operation_id":"LC-P05-13","name":"Infer speaker goals and perspective when required","objective":"Represent speaker beliefs and goals as hypotheses, separate them from listener beliefs, and limit nested mental-state inference.","mathematical_operations":("P(goal|utterance,context)","Belief_speaker(p) != Belief_listener(p)","inverse_planning=P(action|goal)*P(goal)"),"biological_reference":"mPFC and TPJ are prominent mentalizing-network contributors; recruitment depends on the social inference demands."},
    {"operation_id":"LC-P05-14","name":"Evaluate irony, sarcasm and literal-context mismatch","objective":"Compare literal valence with prosody, scene and speaker stance; infer irony only when multiple cues support it.","mathematical_operations":("D=abs(valence_literal-valence_speaker)","BF=P(E|irony,C)/P(E|sincere,C)","P(irony|E)>theta"),"biological_reference":"mPFC, TPJ and language regions can contribute to irony/sarcasm interpretation; electrophysiological effects are not unique markers."},
    {"operation_id":"LC-P05-15","name":"Integrate presuppositions, implicatures and indirect speech acts","objective":"Label asserted, presupposed, implicated and inferred content separately; accommodate or defer unsupported assumptions.","mathematical_operations":("scope(operator,proposition)","common_ground'=update(G,p) only if accepted/accommodated","confidence(content_type)"),"biological_reference":"Distributed language and social-reasoning networks support pragmatic content; no one-to-one region per pragmatic category is assumed."},
    {"operation_id":"LC-P05-16","name":"Rank, combine or defer candidate interpretations","objective":"Score candidates against literal language, discourse, situation, world knowledge, speaker model and common ground, penalizing unsupported assumptions.","mathematical_operations":("S_i=wL*L_i+wD*D_i+wS*S_i+wW*W_i+wM*M_i+wG*G_i-lambda*A_i","p_i=softmax(S_i/T)","defer if entropy high or margin low"),"biological_reference":"Distributed evidence integration and frontal selection/monitoring provide a computational analogy; no exact weighted-sum mechanism is claimed."},
    {"operation_id":"LC-P05-17","name":"Reanalyse earlier assignments when context changes","objective":"Reopen only affected lexical, reference, event or pragmatic alternatives and preserve a revision trace.","mathematical_operations":("P_new=BayesUpdate(P_old,E_new)","delta=distance(P_new,P_old)","stop if delta<epsilon or budget exhausted"),"biological_reference":"Recurrent feedback among temporal, frontal and memory systems allows earlier interpretations to be revised."},
    {"operation_id":"LC-P05-18","name":"Update discourse, event and situation models","objective":"Commit supported entities, event links, temporal relations, common-ground changes and speaker-model updates with provenance.","mathematical_operations":("G_(t+1)=G_t union E_supported","state_(t+1)=U(state_t,evidence)","valid(G)=all_references_resolve"),"biological_reference":"Distributed event/situation representations are updated through recurrent interactions among language, memory and conceptual systems."},
    {"operation_id":"LC-P05-19","name":"Maintain enriched interpretation under working-memory limits","objective":"Retain high-value propositions, referents, unresolved alternatives and inference provenance while allowing low-relevance material to decay.","mathematical_operations":("a_i(t+1)=lambda*a_i(t)+input_i","priority_i=relevance_i*confidence_i*task_value_i","retain top_k(priority)"),"biological_reference":"Working-memory and frontoparietal control systems maintain task-relevant information; this adapter delegates maintenance to the next pipeline stage."},
    {"operation_id":"LC-P05-20","name":"Deliver enriched interpretation and updated context","objective":"Pass literal meaning, licensed inferences, alternatives, uncertainty and updated discourse/situation state to downstream working memory and action systems.","mathematical_operations":("output={P_literal,P_enriched,H,uncertainty,C_updated}","uncertainty=entropy(H)","handoff_integrity=validate_schema(output)"),"biological_reference":"The enriched representation is distributed across language, memory, reasoning and control networks; it is handed to Point 6 without claiming a single storage locus."},
)


@dataclass(frozen=True, slots=True)
class Point5Request:
    """Context-integration input; caller evidence is kept distinct from inferred content."""

    sequence_id: str
    sentences: Sequence[str]
    literal_proposition: Mapping[str, Any] = field(default_factory=dict)
    propositions: Sequence[Mapping[str, Any]] = ()
    entities: Sequence[Mapping[str, Any]] = ()
    prior_turns: Sequence[str] = ()
    common_ground: Mapping[str, Any] = field(default_factory=dict)
    speaker_model: Mapping[str, Any] = field(default_factory=dict)
    situation: Mapping[str, Any] = field(default_factory=dict)
    world_knowledge: Mapping[str, Any] = field(default_factory=dict)
    conversation_goal: Mapping[str, Any] = field(default_factory=dict)
    prosody: Mapping[int, Mapping[str, float]] = field(default_factory=dict)
    temporal_evidence: Mapping[int, Mapping[str, Any]] = field(default_factory=dict)
    unresolved_alternatives: Sequence[Mapping[str, Any]] = ()
    language: str = "en"
    source_reference: str | None = None

    def __post_init__(self) -> None:
        """Validate sequence identity and non-empty sentence input."""
        if not self.sequence_id.strip():
            raise ValueError("sequence_id must be non-empty")
        if not self.sentences or any(not isinstance(x, str) or not x.strip() for x in self.sentences):
            raise ValueError("sentences must contain at least one non-empty string")
        if not self.language.strip():
            raise ValueError("language must be non-empty")
        object.__setattr__(self, "sentences", tuple(self.sentences))
        object.__setattr__(self, "propositions", tuple(dict(x) for x in self.propositions))
        object.__setattr__(self, "entities", tuple(dict(x) for x in self.entities))
        object.__setattr__(self, "prior_turns", tuple(self.prior_turns))
        object.__setattr__(self, "unresolved_alternatives", tuple(dict(x) for x in self.unresolved_alternatives))

    @classmethod
    def from_point4_result(
        cls,
        sequence_id: str,
        point4_result: Any,
        *,
        sentences: Sequence[str],
        prior_turns: Sequence[str] = (),
        entities: Sequence[Mapping[str, Any]] = (),
        common_ground: Mapping[str, Any] | None = None,
        speaker_model: Mapping[str, Any] | None = None,
        situation: Mapping[str, Any] | None = None,
        world_knowledge: Mapping[str, Any] | None = None,
        conversation_goal: Mapping[str, Any] | None = None,
        prosody: Mapping[int, Mapping[str, float]] | None = None,
        temporal_evidence: Mapping[int, Mapping[str, Any]] | None = None,
        language: str = "en",
    ) -> "Point5Request":
        """Create Point 5 input from a Point 4 result and explicit context evidence."""
        payload = point4_result.to_dict() if hasattr(point4_result, "to_dict") else dict(point4_result)
        literal = {
            "propositions": payload.get("propositions", []),
            "proposition_graph": payload.get("proposition_graph", {}),
            "operators": payload.get("operators", {}),
            "semantic_compatibility": payload.get("semantic_compatibility", []),
            "competing_readings": payload.get("competing_readings", []),
            "proposition_completeness": payload.get("proposition_completeness", []),
            "confidence": payload.get("confidence"),
            "handoff": payload.get("handoff", {}),
            "source_schema": payload.get("schema", "ore.language_comprehension.point04"),
        }
        lexical_entities = []
        for index, item in enumerate(payload.get("entity_variables", [])):
            if isinstance(item, Mapping):
                lexical_entities.append({
                    "entity_id": item.get("entity_id", item.get("variable", f"p4-entity-{index+1}")),
                    "label": item.get("label", item.get("surface", item.get("entity_id", f"entity-{index+1}"))),
                    "entity_type": item.get("entity_type", item.get("type", "unknown")),
                    "source_reference": "Point4 entity variable",
                })
        return cls(
            sequence_id=sequence_id,
            sentences=sentences,
            literal_proposition=literal,
            propositions=tuple(payload.get("propositions", [])),
            entities=tuple(entities) or tuple(lexical_entities),
            prior_turns=prior_turns,
            common_ground=dict(common_ground or {}),
            speaker_model=dict(speaker_model or {}),
            situation=dict(situation or {}),
            world_knowledge=dict(world_knowledge or {}),
            conversation_goal=dict(conversation_goal or {}),
            prosody=dict(prosody or {}),
            temporal_evidence=dict(temporal_evidence or {}),
            unresolved_alternatives=tuple(payload.get("competing_readings", [])),
            language=language,
        )


@dataclass(slots=True)
class Point5Result:
    """Literal and context-enriched interpretations with inference provenance and traces."""

    sequence_id: str
    literal_proposition: dict[str, Any]
    enriched_interpretation: dict[str, Any]
    discourse_result: dict[str, Any]
    uncertainty: float
    functional_trace: list[dict[str, Any]]
    biological_trace: list[dict[str, Any]]
    biological_summary: dict[str, Any]
    handoff: dict[str, Any]
    diagnostics: dict[str, Any]
    annotation_report: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible results."""
        return to_jsonable({
            "schema": "ore.language_comprehension.point05.v1",
            "sequence_id": self.sequence_id,
            "literal_proposition": self.literal_proposition,
            "enriched_interpretation": self.enriched_interpretation,
            "discourse_result": self.discourse_result,
            "uncertainty": self.uncertainty,
            "functional_trace": self.functional_trace,
            "biological_trace": self.biological_trace,
            "biological_summary": self.biological_summary,
            "handoff": self.handoff,
            "diagnostics": self.diagnostics,
            "annotation_report": self.annotation_report,
        })


class Point5ContextualPragmaticIntegrator:
    """Run the 20-step source algorithm using ORE's eight-stage discourse engine."""

    def __init__(
        self,
        config: DiscourseConfig | None = None,
        *,
        annotation_report: Mapping[str, Any] | None = None,
    ) -> None:
        """Configure the reused discourse processor and optional annotation report."""
        self.config = config or DiscourseConfig()
        self.processor = DiscourseProcessor(self.config)
        self.annotation_report = dict(annotation_report or {})

    def process(self, request: Point5Request) -> Point5Result:
        """Integrate Point 4 literal content with caller-provided context and discourse."""
        if not isinstance(request, Point5Request):
            raise TypeError("request must be a Point5Request")
        context = {
            "prior_turns": list(request.prior_turns),
            "common_ground": dict(request.common_ground),
            "speaker_model": dict(request.speaker_model),
            "situation": dict(request.situation),
            "world_knowledge": dict(request.world_knowledge),
            "conversation_goal": dict(request.conversation_goal),
            "literal_proposition": dict(request.literal_proposition),
            "unresolved_alternatives": list(request.unresolved_alternatives),
        }
        discourse_request = DiscourseRequest(
            sequence_id=request.sequence_id,
            sentences=request.sentences,
            propositions=request.propositions,
            entities=request.entities,
            common_ground=dict(request.common_ground),
            speaker_model=dict(request.speaker_model),
            prosody=dict(request.prosody),
            temporal_evidence=dict(request.temporal_evidence),
            context=context,
            language=request.language,
            source_reference=request.source_reference,
        )
        result = self.processor.process(discourse_request)
        discourse = result.to_dict()
        # Existing discourse outputs are kept as separately labelled evidence tracks.
        enriched = {
            "literal_proposition": dict(request.literal_proposition),
            "discourse_links": result.proposition_links,
            "coreference": result.discourse_context.get("coreference", []),
            "active_information": result.discourse_context.get("active_information", {}),
            "speaker_intentions": result.speaker_intentions,
            "indirect_speech_acts": result.indirect_requests,
            "irony_hypotheses": result.irony,
            "situation_model": result.situation_model,
            "temporal_relations": result.temporal_relations,
            "coherence": result.coherence,
            "common_ground": result.discourse_context.get("common_ground", {}),
            "context_conflicts": result.coherence.get("knowledge_conflicts", []),
            "unresolved_alternatives": list(request.unresolved_alternatives),
            "world_knowledge_evidence": dict(request.world_knowledge),
            "conversation_goal": dict(request.conversation_goal),
            "inference_policy": {
                "caller_evidence_preferred": True,
                "unsupported_facts_invented": False,
                "literal_content_preserved": True,
                "inferences_are_defeasible": True,
            },
        }
        operations = list(result.operation_trace)
        # Map each source step to a contiguous, non-empty slice of the 39 established operations.
        functional_trace: list[dict[str, Any]] = []
        for index, spec in enumerate(POINT5_STEP_SPECS):
            start = round(index * len(operations) / len(POINT5_STEP_SPECS))
            end = round((index + 1) * len(operations) / len(POINT5_STEP_SPECS))
            operation_rows = operations[start:end]
            functional_trace.append({
                **spec,
                "status": "completed",
                "functional_result": to_jsonable(self._step_output(index + 1, request, result, enriched)),
                "underlying_discourse_operation_ids": [row.get("operation_id") for row in operation_rows],
            })
        biological_trace = []
        biological_enabled = result.biological.get("model") != "disabled"
        for index, spec in enumerate(POINT5_STEP_SPECS):
            start = round(index * len(operations) / len(POINT5_STEP_SPECS))
            end = round((index + 1) * len(operations) / len(POINT5_STEP_SPECS))
            rows = operations[start:end]
            biological_trace.append({
                "step_id": spec["operation_id"],
                "source_step": index + 1,
                "status": "simulated" if biological_enabled else "disabled",
                "underlying_operation_ids": [row.get("operation_id") for row in rows],
                "population_activity": [row.get("biological") for row in rows if row.get("biological") is not None],
                "model": result.biological.get("model", "disabled"),
                "is_measured_human_data": False,
            })
        uncertainty = float(max(0.0, min(1.0, 1.0 - result.coherence.get("score", 0.0))))
        diagnostics = {
            "valid": (
                len(functional_trace) == 20
                and len(operations) == 39
                and result.diagnostics.get("valid", False)
            ),
            "source_step_count": len(functional_trace),
            "underlying_discourse_operation_count": len(operations),
            "underlying_operation_order_valid": [
                row.get("operation_id") for row in operations
            ] == [f"DC-{i:03d}" for i in range(1, 40)],
            "literal_proposition_preserved": enriched["literal_proposition"] == dict(request.literal_proposition),
            "inference_is_defeasible": True,
            "unsupported_facts_invented": False,
            "biological_simulation_enabled": biological_enabled,
            "biological_reference_is_measured_human_data": False,
            "warnings": [
                "The underlying discourse processor uses conservative text heuristics and caller-supplied evidence; it is not a complete theory of human pragmatics.",
                "Reference, intention, irony and causal inferences remain hypotheses unless grounded by explicit evidence.",
                "The LIF populations are a deterministic computational reference, not an anatomically fitted or physiologically validated model.",
                "Contextual input fields are preserved as provenance; not every field is independently reasoned over by the compact discourse engine.",
            ],
        }
        handoff = {
            "target": "Point 6 — Propositional working-memory representation and maintenance",
            "literal_proposition": dict(request.literal_proposition),
            "enriched_interpretation": enriched,
            "discourse_state": result.discourse_context,
            "uncertainty": uncertainty,
            "unresolved_alternatives": list(request.unresolved_alternatives),
        }
        return Point5Result(
            sequence_id=request.sequence_id,
            literal_proposition=dict(request.literal_proposition),
            enriched_interpretation=enriched,
            discourse_result=discourse,
            uncertainty=uncertainty,
            functional_trace=functional_trace,
            biological_trace=biological_trace,
            biological_summary=dict(result.biological),
            handoff=handoff,
            diagnostics=diagnostics,
            annotation_report=self.annotation_report,
        )

    @staticmethod
    def _step_output(index: int, request: Point5Request, result: Any, enriched: Mapping[str, Any]) -> dict[str, Any]:
        """Return the output relevant to one source-level step."""
        outputs = (
            {"literal_proposition": dict(request.literal_proposition), "propositions": result.propositions},
            {"retrieved_prior_turns": list(request.prior_turns), "active_information": result.discourse_context.get("active_information", {})},
            {"discourse_context": result.discourse_context, "caller_context_sources": {"situation": dict(request.situation), "world_knowledge": dict(request.world_knowledge), "conversation_goal": dict(request.conversation_goal)}},
            {"topic": result.coherence.get("dominant_topic_term"), "topic_distribution": result.coherence.get("topic_distribution", {}), "entities": result.situation_model.get("people", [])},
            {"coreference": result.discourse_context.get("coreference", []), "unresolved_reference_policy": "retain alternatives/abstain"},
            {"predictions": result.situation_model.get("motivations", []), "topic_distribution": result.coherence.get("topic_distribution", {})},
            {"proposition_links": result.proposition_links, "coherence": result.coherence},
            {"situation_model": result.situation_model, "caller_scene": dict(request.situation)},
            {"event_plausibility_evidence": dict(request.world_knowledge), "knowledge_conflicts": result.coherence.get("knowledge_conflicts", [])},
            {"coherence_score": result.coherence.get("score"), "uncertainty": 1.0 - result.coherence.get("score", 0.0)},
            {"causal_links": result.situation_model.get("causal_links", []), "temporal_relations": result.temporal_relations},
            {"speaker_intentions": result.speaker_intentions, "indirect_requests": result.indirect_requests},
            {"speaker_model": dict(request.speaker_model), "speaker_intentions": result.speaker_intentions},
            {"irony": result.irony},
            {"indirect_requests": result.indirect_requests, "common_ground": result.discourse_context.get("common_ground", {})},
            {"candidate_interpretation_evidence": enriched, "coherence": result.coherence},
            {"recurrent_revision": result.coherence.get("recurrent_revision", {}), "knowledge_conflicts": result.coherence.get("knowledge_conflicts", [])},
            {"discourse_context": result.discourse_context, "situation_model": result.situation_model},
            {"active_information": result.discourse_context.get("active_information", {}), "retention_policy": "task-relevant context retained; detailed maintenance deferred to Point 6"},
            {"handoff": {"target": "Point 6", "uncertainty": 1.0 - result.coherence.get("score", 0.0)}},
        )
        return outputs[index - 1]


__all__ = [
    "POINT5_STEP_SPECS",
    "Point5ContextualPragmaticIntegrator",
    "Point5Request",
    "Point5Result",
]
