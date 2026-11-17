"""Source-ordered Point 6 propositional working-memory adapter.

This layer reuses ORE's established working-memory/cognitive-control subsystem. It keeps
literal propositions, contextual enrichment, working-memory state, and neural-reference
outputs separate and traceable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Mapping, Sequence

from ore_sensory_perception.contracts import to_jsonable
from ore_working_memory import (
    WorkingMemoryConfig,
    WorkingMemoryProcessor,
    WorkingMemoryRequest,
)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    """Return a mapping from an existing mapping or a result object's to_dict method."""
    if isinstance(value, Mapping):
        return value
    converter = getattr(value, "to_dict", None)
    if callable(converter):
        converted = converter()
        if isinstance(converted, Mapping):
            return converted
    raise TypeError("point5_result must be a mapping or expose to_dict()")


def _score(value: Any, default: float = 0.5) -> float:
    """Convert an optional confidence-like value to a finite normalized score."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not 0.0 <= number <= 1.0:
        return default
    return number


POINT6_STEP_SPECS: tuple[dict[str, Any], ...] = (
    {
        "operation_id": "LC-P06-01",
        "name": "Receive the developing meaning representation",
        "objective": "Receive Point 5's proposition, context, referents, relations, confidence and unresolved information without restarting earlier language stages.",
        "mathematical_operations": ("x_t=F(x_(t-1),u_t,c_t)", "preserve provenance(source,representation,time)", "validate required input channels"),
        "biological_reference": "Distributed temporal/frontal language populations interact with parietal and association networks; the software request is not a literal neural packet.",
    },
    {
        "operation_id": "LC-P06-02",
        "name": "Segment the interpretation into proposition candidates",
        "objective": "Maintain proposition-level units and clause relationships, including embedded, coordinated and incomplete clauses.",
        "mathematical_operations": ("B_t=sigmoid(w^T f_t+b)", "hierarchical grouping of clause nodes", "G_P=(V_proposition,E_embedding,E_coordination)"),
        "biological_reference": "Left inferior frontal and temporal language regions contribute to structural-semantic organization; clause boundaries are not assigned to one isolated region.",
    },
    {
        "operation_id": "LC-P06-03",
        "name": "Establish entity and event identities",
        "objective": "Retain coreference identities and event references while preserving competing antecedents when context does not support a unique choice.",
        "mathematical_operations": ("s_j=w_a*A_j+w_g*G_j+w_r*R_j+w_d*D_j+w_c*C_j", "p_j=softmax(s_j/tau)", "H=-sum_j p_j log(p_j)"),
        "biological_reference": "Temporal/parietal language representations interact with semantic memory and attentional retrieval during reference integration.",
    },
    {
        "operation_id": "LC-P06-04",
        "name": "Bind predicates to their arguments and semantic roles",
        "objective": "Keep role–filler relations explicit, including open roles and noncanonical mappings between grammatical functions and semantic roles.",
        "mathematical_operations": ("b_r=phi(r,e)", "B=sum_r r_tensor_product e_r", "role assignment as constrained classification"),
        "biological_reference": "Left inferior frontal and middle temporal language systems contribute to predicate/role integration in a distributed network.",
    },
    {
        "operation_id": "LC-P06-05",
        "name": "Attach operators, modifiers and scope",
        "objective": "Preserve negation, tense, aspect, modality, quantification, degree and modifiers at their correct target and scope.",
        "mathematical_operations": ("operator_edge=(operator,target,scope)", "NOT(P)=1-P only in a simple truth-valued model", "scope consistency constraints"),
        "biological_reference": "Distributed language processing integrates operators; complex scope may increase frontoparietal memory and control demands.",
    },
    {
        "operation_id": "LC-P06-06",
        "name": "Preserve stated, inferred, presupposed and unresolved content",
        "objective": "Keep provenance labels and confidence separate from proposition content so defeasible enrichment never becomes an asserted fact.",
        "mathematical_operations": ("source_label in {explicit,entailed,inferred,presupposed,assumed,unresolved}", "q_(t+1)=q_t+eta*(e_t-q_t)", "confidence and provenance stored as independent fields"),
        "biological_reference": "Language and domain-general reasoning/social systems may contribute different evidence; named provenance labels are a software requirement, not literal neural tags.",
    },
    {
        "operation_id": "LC-P06-07",
        "name": "Link propositions into event and discourse structure",
        "objective": "Preserve temporal, causal, conditional, contrastive, elaborative and referential relations while avoiding unsupported causal links.",
        "mathematical_operations": ("G_D=(V,E,lambda_V,lambda_E)", "interval constraint propagation", "acyclicity and relation-consistency checks"),
        "biological_reference": "Angular gyrus and posterior cingulate may contribute to broader proposition binding within distributed context and memory networks.",
    },
    {
        "operation_id": "LC-P06-08",
        "name": "Construct or update the situation model",
        "objective": "Maintain the supported entities, events, states, temporal/spatial information and goal relations described by the current discourse.",
        "mathematical_operations": ("G_t=G_(t-1) union DeltaG_t", "S_(t+1)=T(S_t,E_t)", "consistency-check(new_state,prior_state)"),
        "biological_reference": "Language regions interact with memory and conceptual association systems; a situation model need not be a vivid visual image.",
    },
    {
        "operation_id": "LC-P06-09",
        "name": "Integrate linguistic-semantic and conceptual-event information",
        "objective": "Connect words and propositions to event knowledge without collapsing language, semantic features and situation evidence into one undifferentiated vector.",
        "mathematical_operations": ("z=W_l*l+W_s*s+W_e*e", "h=f(z)", "compatibility=similarity(event_representation,knowledge)"),
        "biological_reference": "Temporal/inferior-frontal language networks interact with parietal and broader association systems supporting conceptual-event integration.",
    },
    {
        "operation_id": "LC-P06-10",
        "name": "Allocate limited working-memory resources",
        "objective": "Prioritize task-relevant propositions, open dependencies, referents and events without assuming a universal fixed number of proposition slots.",
        "mathematical_operations": ("pi_i=a*R_i+b*U_i+c*T_i+d*D_i", "alpha_i=softmax(pi_i)", "sum_i m_i<=C"),
        "biological_reference": "Left inferior frontal and bilateral parietal systems support temporary maintenance; multiple-demand control may increase with memory load.",
    },
    {
        "operation_id": "LC-P06-11",
        "name": "Maintain active information during comprehension",
        "objective": "Keep relevant items accessible through activation, refresh and rehearsal while allowing low-priority content to decay without silently deleting its provenance.",
        "mathematical_operations": ("x_(t+1)=lambda*x_t+u_t", "dx/dt=-x/tau+u(t)", "tau*dx/dt=-x+W_rec*f(x)+B*u+noise"),
        "biological_reference": "Distributed language and working-memory networks may use persistent activity, reactivation and activity-silent mechanisms; persistent firing is not the only account.",
    },
    {
        "operation_id": "LC-P06-12",
        "name": "Update the representation as new words and clauses arrive",
        "objective": "Add new predicates, arguments and relations, resolve open roles when supported, revise affected interpretations and recompute item priorities.",
        "mathematical_operations": ("M_(t+1)=U(M_t,u_(t+1))", "delta_t=u_t-u_hat_t", "local graph patch rather than full rebuild"),
        "biological_reference": "Recurrent interactions between language populations and working-memory support systems permit incremental revision; this adapter processes one supplied sequence per request.",
    },
    {
        "operation_id": "LC-P06-13",
        "name": "Protect active representations from interference",
        "objective": "Distinguish similar propositions and referents using identity, role, temporal and task cues rather than selecting the most recent item automatically.",
        "mathematical_operations": ("sim(x_i,x_j)=dot(x_i,x_j)/(norm(x_i)*norm(x_j))", "r_i=b1*sim+b2*role+b3*context", "competitive suppression and cue-based selection"),
        "biological_reference": "Selective attention and competitive neural dynamics may limit interference; no single frontal gate is claimed.",
    },
    {
        "operation_id": "LC-P06-14",
        "name": "Retrieve maintained information when later input depends on it",
        "objective": "Reactivate the appropriate entity, event or proposition from grammatical, semantic, recency and discourse-focus cues; retain candidates when retrieval remains uncertain.",
        "mathematical_operations": ("R_i=gamma1*C_i+gamma2*G_i+gamma3*S_i+gamma4*F_i", "p_i=softmax(R_i/T)", "abstain when posterior margin is low"),
        "biological_reference": "Retrieval is distributed across language and memory-related systems; there is no single dedicated centre for every referential dependency.",
    },
    {
        "operation_id": "LC-P06-15",
        "name": "Monitor proposition completeness for the current task",
        "objective": "Track obligatory roles, unresolved references, pending operators, conflicts and task-relevant gaps without requiring the entire discourse to be closed.",
        "mathematical_operations": ("K(P)=sum_j(w_j*c_j)/sum_j(w_j)", "open_role_count=sum_i missing_roles_i", "task-relative completion threshold"),
        "biological_reference": "Language and control networks monitor unresolved structure; completion is task-relative rather than an all-or-nothing neural state.",
    },
    {
        "operation_id": "LC-P06-16",
        "name": "Compress and organize information without losing essential relations",
        "objective": "Group related events and propositions into chunks while preserving roles, scope, order, source and uncertainty; retain pointers for later reactivation.",
        "mathematical_operations": ("min_Mtilde D(M,Mtilde)+lambda*Cost(Mtilde)", "task-weighted feature selection", "graph-preserving subgraph grouping"),
        "biological_reference": "Chunking and relational organization reduce processing demands; the abstract compression objective is an engineering analogy.",
    },
    {
        "operation_id": "LC-P06-17",
        "name": "Recruit executive control only when maintenance demands require it",
        "objective": "Estimate unresolved dependencies, conflict, ambiguity and task demand, and scale selection/control support only when needed.",
        "mathematical_operations": ("D_t=a*N_open+b*I+c*A+d*T", "control_gain=sigmoid(k*(D_t-theta))", "adaptive attention allocation"),
        "biological_reference": "Dorsolateral prefrontal and parietal multiple-demand systems may contribute under high control or memory load; not every sentence requires strong recruitment.",
    },
    {
        "operation_id": "LC-P06-18",
        "name": "Maintain a coherent active state and hand it to downstream systems",
        "objective": "Package active propositions, situation relations, provenance, confidence, unresolved alternatives and reactivation pointers for ongoing comprehension or later memory/decision processing.",
        "mathematical_operations": ("O_t={P_active,G_discourse,S_situation,U_unresolved,Q_confidence}", "handoff_integrity=validate_schema(O_t)", "temporary_state != durable_memory"),
        "biological_reference": "Temporary maintenance involves distributed language and parietal working-memory networks; durable learning and decision selection are downstream processes.",
    },
)


@dataclass(frozen=True, slots=True)
class Point6Request:
    """Validated working-memory input with optional Point 5 and upstream evidence."""

    sequence_id: str
    text: str
    point5_interpretation: Mapping[str, Any] = field(default_factory=dict)
    syntax_result: Any | None = None
    partial_structures: Sequence[Mapping[str, Any]] = ()
    interpretations: Sequence[Mapping[str, Any]] = ()
    task_sequence: Sequence[str] = ("comprehension",)
    goals: Mapping[str, Any] = field(default_factory=dict)
    context: Mapping[str, Any] = field(default_factory=dict)
    language: str = "en"
    source_reference: str | None = None

    def __post_init__(self) -> None:
        """Reject invalid text and normalize caller records before buffer allocation."""
        if not isinstance(self.sequence_id, str) or not self.sequence_id.strip():
            raise ValueError("sequence_id must be non-empty")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("text must be a non-empty surface representation")
        if not isinstance(self.point5_interpretation, Mapping):
            raise TypeError("point5_interpretation must be a mapping")
        if not isinstance(self.goals, Mapping) or not isinstance(self.context, Mapping):
            raise TypeError("goals and context must be mappings")
        if not isinstance(self.language, str) or not self.language.strip():
            raise ValueError("language must be non-empty")
        object.__setattr__(self, "partial_structures", tuple(dict(x) for x in self.partial_structures))
        object.__setattr__(self, "interpretations", tuple(dict(x) for x in self.interpretations))
        object.__setattr__(self, "task_sequence", tuple(self.task_sequence))

    @classmethod
    def from_point5_result(
        cls,
        sequence_id: str,
        point5_result: Any,
        *,
        text: str | None = None,
        syntax_result: Any | None = None,
        partial_structures: Sequence[Mapping[str, Any]] = (),
        interpretations: Sequence[Mapping[str, Any]] = (),
        task_sequence: Sequence[str] = ("comprehension",),
        goals: Mapping[str, Any] | None = None,
        context: Mapping[str, Any] | None = None,
        language: str = "en",
    ) -> "Point6Request":
        """Create a working-memory request while preserving Point 5's literal and enriched tracks."""
        payload = _as_mapping(point5_result)
        discourse = payload.get("discourse_result", {})
        enriched = payload.get("enriched_interpretation", {})
        literal = payload.get("literal_proposition", {})
        discourse = discourse if isinstance(discourse, Mapping) else {}
        enriched = enriched if isinstance(enriched, Mapping) else {}
        literal = literal if isinstance(literal, Mapping) else {}

        surface_sentences = discourse.get("sentences", [])
        if text is None:
            if isinstance(surface_sentences, Sequence) and not isinstance(surface_sentences, str):
                text = "\n".join(str(item) for item in surface_sentences if str(item).strip())
            if not text:
                raise ValueError("Point 5 result has no surface sentences; provide text explicitly")

        p4_completeness = literal.get("proposition_completeness", [])
        generated_structures = []
        if isinstance(p4_completeness, Sequence) and not isinstance(p4_completeness, str):
            for index, item in enumerate(p4_completeness):
                if not isinstance(item, Mapping):
                    continue
                required = [str(x) for x in item.get("required_roles", ())]
                filled = [str(x) for x in item.get("filled_required_roles", ())]
                missing = [str(x) for x in item.get("missing_required_roles", ())]
                if missing:
                    generated_structures.append({
                        "structure_id": f"point4-predicate-{index + 1}",
                        "label": f"predicate argument frame: {item.get('predicate', 'predicate')}",
                        "expected_slots": required or missing,
                        "filled_slots": filled,
                        "source": "Point4 proposition_completeness",
                        "candidate_bindings": [],
                    })

        supplied_alternatives = list(interpretations)
        if not supplied_alternatives:
            propositions = literal.get("propositions", discourse.get("propositions", []))
            if propositions:
                literal_summary = json.dumps(
                    to_jsonable(propositions), ensure_ascii=False, sort_keys=True
                )
                coherence_score = _score(
                    enriched.get("coherence", {}).get("score", 0.5)
                )
                literal_confidence = _score(literal.get("confidence", 0.5))
                supplied_alternatives.append({
                    "hypothesis_id": "point5-literal-reading",
                    "interpretation": literal_summary,
                    "prior": literal_confidence,
                    "lexical_score": 0.5,
                    "semantic_score": literal_confidence,
                    "syntactic_score": 0.5,
                    "discourse_score": coherence_score,
                    "world_score": 0.5,
                    "pragmatic_score": 0.5,
                    "evidence_source": "Point5 literal proposition; neutral channels remain unmeasured",
                })
            alternatives = enriched.get("unresolved_alternatives", ())
            if isinstance(alternatives, Sequence) and not isinstance(alternatives, str):
                for index, item in enumerate(alternatives[:7]):
                    if not isinstance(item, Mapping):
                        continue
                    description = item.get(
                        "interpretation", item.get("label", item.get("reading", item))
                    )
                    supplied_alternatives.append({
                        "hypothesis_id": str(item.get(
                            "reading_id", item.get("hypothesis_id", f"point5-alt-{index + 1}")
                        )),
                        "interpretation": json.dumps(
                            to_jsonable(description), ensure_ascii=False, sort_keys=True
                        ),
                        "prior": _score(item.get(
                            "prior", item.get("confidence", item.get("score", 0.5))
                        )),
                        "lexical_score": _score(item.get("lexical_score", 0.5)),
                        "semantic_score": _score(
                            item.get("semantic_score", item.get("confidence", 0.5))
                        ),
                        "syntactic_score": _score(item.get("syntactic_score", 0.5)),
                        "discourse_score": _score(item.get("discourse_score", 0.5)),
                        "world_score": _score(item.get("world_score", 0.5)),
                        "pragmatic_score": _score(item.get("pragmatic_score", 0.5)),
                        "evidence_source": "Point5 unresolved alternative; uncalibrated confidence retained",
                    })
        if not supplied_alternatives:
            supplied_alternatives = [{
                "hypothesis_id": "point5-surface-reading",
                "interpretation": str(text),
                "prior": 1.0,
                "lexical_score": 0.5,
                "semantic_score": 0.5,
                "syntactic_score": 0.5,
                "discourse_score": 0.5,
                "world_score": 0.5,
                "pragmatic_score": 0.5,
                "evidence_source": "neutral surface fallback because no structured proposition was supplied",
            }]

        merged_context = {
            "point5_result": dict(payload),
            "point5_literal_proposition": dict(literal),
            "point5_enriched_interpretation": dict(enriched),
            "point5_discourse_state": dict(enriched.get("active_information", {})),
            "situation_model": dict(enriched.get("situation_model", {})),
            "common_ground": dict(enriched.get("common_ground", {})),
            "speaker_intentions": list(enriched.get("speaker_intentions", ())),
            "unresolved_alternatives": list(enriched.get("unresolved_alternatives", ())),
            "ambiguity_sources": (
                [{"kind": "Point5_unresolved_alternative", "source": "Point5", "item": item}
                 for item in enriched.get("unresolved_alternatives", ())]
            ),
            "context_features": dict(enriched.get("coherence", {}).get("topic_distribution", {})),
            "focus_cues": [
                str(enriched.get("coherence", {}).get("dominant_topic_term"))
            ] if enriched.get("coherence", {}).get("dominant_topic_term") else [],
            "semantic_knowledge": dict(enriched.get("world_knowledge_evidence", {})),
            "current_task": "comprehension",
        }
        merged_context.update(dict(context or {}))
        return cls(
            sequence_id=sequence_id,
            text=str(text),
            point5_interpretation=dict(payload),
            syntax_result=syntax_result,
            partial_structures=tuple(generated_structures) + tuple(partial_structures),
            interpretations=tuple(supplied_alternatives),
            task_sequence=task_sequence,
            goals=dict(goals or {}),
            context=merged_context,
            language=language,
            source_reference="Point 5 contextual/pragmatic integration handoff",
        )


@dataclass(slots=True)
class Point6Result:
    """Active propositional working-memory representation and inspectable operation traces."""

    sequence_id: str
    point5_interpretation: dict[str, Any]
    working_memory_state: dict[str, Any]
    biological_summary: dict[str, Any]
    functional_trace: list[dict[str, Any]]
    biological_trace: list[dict[str, Any]]
    handoff: dict[str, Any]
    diagnostics: dict[str, Any]
    annotation_report: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize Point 6 without converting its temporary state into durable memory."""
        return to_jsonable({
            "schema": "ore.language_comprehension.point06.v1",
            "sequence_id": self.sequence_id,
            "point5_interpretation": self.point5_interpretation,
            "working_memory_state": self.working_memory_state,
            "biological_summary": self.biological_summary,
            "functional_trace": self.functional_trace,
            "biological_trace": self.biological_trace,
            "handoff": self.handoff,
            "diagnostics": self.diagnostics,
            "annotation_report": self.annotation_report,
        })


class Point6PropositionalWorkingMemory:
    """Execute the 18-step Point 6 adapter over ORE's five-stage/40-operation WM engine."""

    def __init__(
        self,
        config: WorkingMemoryConfig | None = None,
        *,
        annotation_report: Mapping[str, Any] | None = None,
    ) -> None:
        """Configure bounded working-memory buffers, inference, control and optional LIF reference."""
        self.config = config or WorkingMemoryConfig()
        self.processor = WorkingMemoryProcessor(self.config)
        self.annotation_report = dict(annotation_report or {})

    def process(self, request: Point6Request) -> Point6Result:
        """Maintain propositions, context and uncertainty in a temporary task-limited state."""
        if not isinstance(request, Point6Request):
            raise TypeError("request must be a Point6Request")
        wm_request = WorkingMemoryRequest(
            sequence_id=request.sequence_id,
            text=request.text,
            syntax_result=request.syntax_result,
            partial_structures=request.partial_structures,
            interpretations=request.interpretations,
            task_sequence=request.task_sequence,
            goals=request.goals,
            context=request.context,
            language=request.language,
            source_reference=request.source_reference,
        )
        output = self.processor.process(wm_request)
        wm = output.to_dict()
        operation_rows = list(output.operation_trace)
        extension_rows = list(output.metadata.get("source_extension_trace", ()))
        functional_trace = []
        biological_trace = []
        boundaries = [
            round(index * len(operation_rows) / len(POINT6_STEP_SPECS))
            for index in range(len(POINT6_STEP_SPECS) + 1)
        ]
        for index, spec in enumerate(POINT6_STEP_SPECS):
            rows = operation_rows[boundaries[index]:boundaries[index + 1]]
            extensions = []
            if index == 16:
                extensions.extend(row for row in extension_rows if row.get("operation_id") == "WMX-001")
            if index == 17:
                extensions.extend(row for row in extension_rows if row.get("operation_id") == "WMX-002")
            functional_trace.append({
                **spec,
                "status": "completed",
                "functional_result": to_jsonable(self._step_output(index + 1, request, output)),
                "underlying_working_memory_operation_ids": [
                    row.get("operation_id") for row in rows
                ] + [row.get("operation_id") for row in extensions],
                "underlying_source_operation_ids": [
                    source_id for row in rows for source_id in row.get("source_operation_ids", ())
                ] + [row.get("source_operation_id") for row in extensions if row.get("source_operation_id")],
                "source_extensions": to_jsonable(extensions),
            })
            biological_trace.append({
                "step_id": spec["operation_id"],
                "source_step": index + 1,
                "status": "simulated" if output.biological.get("model") != "disabled" else "disabled",
                "underlying_operation_ids": [row.get("operation_id") for row in rows],
                "population_updates": [row.get("biological_update", {}) for row in rows],
                "model": output.biological.get("model", "disabled"),
                "is_measured_human_data": False,
            })

        state = {
            "original_text": output.original_text,
            "tokens": output.tokens,
            "buffers": output.buffers,
            "incomplete_structures": output.incomplete_structures,
            "executive_control": output.executive_control,
            "semantic_resolution": output.semantic_resolution,
            "task_switching": output.task_switching,
            "diagnostics": output.diagnostics,
            "operation_trace": output.operation_trace,
            "metadata": output.metadata,
            "temporary_state_only": True,
            "durable_memory_written": False,
        }
        handoff = {
            "target": "Downstream comprehension, long-term-memory and decision systems",
            "active_propositions": request.point5_interpretation.get("literal_proposition", {}).get(
                "propositions", []
            ),
            "situation_model": request.context.get("situation_model", {}),
            "working_memory_state": {
                "buffers": output.buffers,
                "incomplete_structures": output.incomplete_structures,
                "semantic_resolution": output.semantic_resolution,
                "executive_control": output.executive_control,
            },
            "unresolved_alternatives": request.context.get("unresolved_alternatives", []),
            "task_state": output.task_switching,
            "uncertainty": output.semantic_resolution.get("normalized_entropy", 0.0),
            "reactivation_pointers": output.incomplete_structures.get("pending_dependencies", []),
            "durable_storage_performed": False,
        }
        diagnostics = {
            "valid": (
                len(functional_trace) == 18
                and len(operation_rows) == 40
                and len(extension_rows) == 2
                and all(rows for rows in [
                    operation_rows[boundaries[i]:boundaries[i + 1]]
                    for i in range(len(POINT6_STEP_SPECS))
                ])
                and output.diagnostics.get("valid", False)
                and len({row.get("operation_id") for row in operation_rows}) == 40
            ),
            "source_step_count": len(functional_trace),
            "underlying_working_memory_operation_count": len(operation_rows),
            "underlying_operation_order_valid": [
                row.get("operation_id") for row in operation_rows
            ] == [f"WMCC-{i:03d}" for i in range(1, 41)],
            "source_extension_count": len(extension_rows),
            "source_extension_ids": [row.get("operation_id") for row in extension_rows],
            "source_operation_mapping_complete": output.diagnostics.get("source_operation_mapping_complete", False),
            "temporary_state_only": True,
            "durable_memory_written": False,
            "biological_simulation_enabled": output.biological.get("model") != "disabled",
            "biological_reference_is_measured_human_data": False,
            "warnings": [
                "The working-memory engine is a bounded, evidence-driven software approximation, not a validated theory of human working memory.",
                "The adapter processes the supplied sequence in one request; it does not preserve a live neural state across separate calls.",
                "The canonical engine contains transparent heuristics and neutral defaults when upstream evidence is absent.",
                "The biological-reference simulator is a computational LIF analogy, not measured human neural data.",
                "Temporary working-memory consolidation is limited to the current software episode; no durable episodic memory write is performed.",
            ] + list(output.diagnostics.get("warnings", ())),
        }
        return Point6Result(
            sequence_id=request.sequence_id,
            point5_interpretation=dict(request.point5_interpretation),
            working_memory_state=state,
            biological_summary=dict(output.biological),
            functional_trace=functional_trace,
            biological_trace=biological_trace,
            handoff=handoff,
            diagnostics=diagnostics,
            annotation_report=self.annotation_report,
        )

    @staticmethod
    def _step_output(index: int, request: Point6Request, result: Any) -> dict[str, Any]:
        """Expose a stable, step-specific view of the shared working-memory processor output."""
        literal = request.point5_interpretation.get("literal_proposition", {})
        enriched = request.point5_interpretation.get("enriched_interpretation", {})
        outputs = (
            {"received_text": request.text, "point5_literal_and_enriched_tracks": request.point5_interpretation},
            {"tokens": result.tokens, "token_count": result.diagnostics.get("token_count")},
            {"coreference": enriched.get("coreference", []), "situation_entities": request.context.get("situation_model", {}).get("people", []), "situation_events": request.context.get("situation_model", {}).get("events", [])},
            {"literal_propositions": literal.get("propositions", []), "predicate_graph": literal.get("proposition_graph", {})},
            {"operators": literal.get("operators", {}), "predicate_graph_edges": literal.get("proposition_graph", {}).get("edges", []), "modifiers_and_roles": [edge for edge in literal.get("proposition_graph", {}).get("edges", []) if "role" in str(edge.get("edge_type", "")).casefold() or "modifier" in str(edge.get("edge_type", "")).casefold()]},
            {"literal_proposition": literal, "pragmatic_enrichment": enriched, "provenance_policy": enriched.get("inference_policy", {})},
            {"discourse_links": enriched.get("discourse_links", []), "temporal_relations": enriched.get("temporal_relations", [])},
            {"situation_model": request.context.get("situation_model", {})},
            {"semantic_resolution": result.semantic_resolution, "evidence_ledger": result.semantic_resolution.get("evidence_ledger", [])},
            {"buffer_sizes": result.diagnostics.get("buffer_sizes", {}), "incomplete_structures": result.incomplete_structures},
            {"buffers": result.buffers, "activation_and_rehearsal": {name: [{"token": item.get("token"), "activation": item.get("activation"), "rehearsal_count": item.get("rehearsal_count")} for item in rows] for name, rows in result.buffers.items()}},
            {"incremental_request_text": request.text, "token_positions": list(range(len(result.tokens))), "incremental_processing_scope": "per-request sequence; not a streaming cross-call state"},
            {"interference_estimates": {name: [{"token": item.get("token"), "similarity_interference": item.get("similarity_interference", 0.0)} for item in rows] for name, rows in result.buffers.items()}},
            {"retrieval_candidates": result.incomplete_structures.get("pending_dependencies", []), "buffer_references": result.incomplete_structures.get("frames", [])},
            {"completion": result.incomplete_structures, "unresolved_slot_count": result.diagnostics.get("unresolved_slot_count", 0)},
            {"chunk_count": result.diagnostics.get("chunk_count", 0), "buffers": result.buffers, "compression_scope": "canonical engine's chunking and bounded buffer records"},
            {"executive_control": result.executive_control, "task_switching": result.task_switching},
            {"active_buffers": result.buffers, "situation_model": request.context.get("situation_model", {}), "unresolved": result.semantic_resolution.get("alternatives", []), "temporary_only": True},
        )
        return to_jsonable(outputs[index - 1])


__all__ = [
    "POINT6_STEP_SPECS",
    "Point6PropositionalWorkingMemory",
    "Point6Request",
    "Point6Result",
]
