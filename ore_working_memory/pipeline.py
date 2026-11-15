"""Canonical 40-operation working-memory and cognitive-control processing pipeline.

The functional track uses bounded data structures, cue-based retrieval, explicit ambiguity, and
conservative slot binding. A separate optional LIF reference advances once per registered operation.
"""
from __future__ import annotations

import math
import re
from typing import Any, Mapping, Sequence

from .biology import WorkingMemoryLIFSimulator
from .contracts import WorkingMemoryConfig, WorkingMemoryRequest, WorkingMemoryResult
from .registry import OPERATION_REGISTRY, STAGE_REGISTRY, OperationSpec


_TOKEN_PATTERN = re.compile(r"\w+(?:['’]\w+)?|[^\w\s]", flags=re.UNICODE)
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "am",
    "of", "to", "in", "on", "at", "for", "by", "with", "from", "as", "and",
    "or", "but", "if", "then", "that", "this", "these", "those", "it", "he",
    "she", "they", "we", "you", "i", "me", "him", "her", "them", "my", "your",
}
_SYNTACTIC_MARKERS = {
    "a", "an", "the", "to", "of", "in", "on", "at", "for", "by", "with",
    "because", "although", "if", "that", "which", "who", "whom", "whose",
    "what", "when", "where", "while", "and", "or", "but", "not", "never",
}
_TASK_FAMILIES = {
    "comprehension": "language", "reading": "language", "parsing": "language",
    "semantic_resolution": "language", "interpretation": "language",
    "memory_retrieval": "memory", "working_memory": "memory", "recall": "memory",
    "planning": "executive", "task_switching": "executive", "attention": "executive",
    "inhibition": "executive", "visual": "perception", "auditory": "perception",
    "reasoning": "reasoning", "inference": "reasoning", "calculation": "reasoning",
}
_OPERATION_BY_ID = {item.operation_id: item for item in OPERATION_REGISTRY}


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Bound a scalar to the requested interval before it enters a normalized activation channel."""
    return max(low, min(high, float(value)))


def _softmax(values: Sequence[float], temperature: float = 1.0) -> list[float]:
    """Convert finite candidate scores into a numerically stable temperature-scaled distribution."""
    if not values:
        return []
    scale = max(float(temperature), 1e-12)
    scaled = [float(value) / scale for value in values]
    pivot = max(scaled)
    exponentials = [math.exp(max(-700.0, value - pivot)) for value in scaled]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def _cosine_similarity(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    """Measure cosine overlap between two sparse, non-negative feature mappings."""
    keys = set(left) | set(right)
    if not keys:
        return 0.5
    a = [float(left.get(key, 0.0)) for key in keys]
    b = [float(right.get(key, 0.0)) for key in keys]
    norm_a = math.sqrt(sum(value * value for value in a))
    norm_b = math.sqrt(sum(value * value for value in b))
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return 0.0
    return _clip(sum(x * y for x, y in zip(a, b)) / (norm_a * norm_b))


def _surface_similarity(left: str, right: str) -> float:
    """Estimate lexical-form overlap with character-set Jaccard similarity, not numeric cosine."""
    a = str(left).casefold()
    b = str(right).casefold()
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    set_a, set_b = set(a), set(b)
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else 0.0


def _as_mapping(value: Any) -> Mapping[str, Any]:
    """Read an upstream result as a mapping without changing the original object."""
    if isinstance(value, Mapping):
        return value
    converter = getattr(value, "to_dict", None)
    if callable(converter):
        converted = converter()
        if isinstance(converted, Mapping):
            return converted
    return {}


def _task_family(task: str) -> str:
    """Map a task label to a broad family used only for a simple interference estimate."""
    normalized = task.strip().lower().replace(" ", "_")
    return _TASK_FAMILIES.get(normalized, normalized.split("_", 1)[0])


class WorkingMemoryProcessor:
    """Coordinate the five WMCC stages and emit the canonical 40-operation execution trace.

    The processor is deterministic. It does not use a trained parser or world-knowledge model;
    upstream evidence is accepted explicitly and weak evidence is retained as unresolved.
    Executive acceptance is provisional; semantic resolution can revise it when new evidence arrives.
    """

    def __init__(self, config: WorkingMemoryConfig | None = None) -> None:
        """Store validated configuration for buffer limits, inference, task costs, and biology."""
        self.config = config or WorkingMemoryConfig()
        self._simulator: WorkingMemoryLIFSimulator | None = None

    def process(self, request: WorkingMemoryRequest) -> WorkingMemoryResult:
        """Run the canonical WMCC-001..WMCC-040 sequence and return all five subsystem states."""
        if not isinstance(request, WorkingMemoryRequest):
            raise TypeError("request must be a WorkingMemoryRequest")
        self._simulator = (
            WorkingMemoryLIFSimulator(self.config) if self.config.simulate_biology else None
        )
        state: dict[str, Any] = {"trace": []}
        operations = (
            self._op_initialize_cognitive_environment,
            self._op_receive_linguistic_information,
            self._op_tokenize_and_encode,
            self._op_allocate_linguistic_buffers,
            self._op_chunk_stored_information,
            self._op_calculate_buffer_activation,
            self._op_decay_and_capacity,
            self._op_retrieve_context_cues,
            self._op_extract_incomplete_structures,
            self._op_score_slot_retrieval_cues,
            self._op_track_unresolved_dependencies,
            self._op_build_syntactic_stack,
            self._op_bind_supported_slots,
            self._op_emit_incomplete_state,
            self._op_maintain_cognitive_goals,
            self._op_monitor_sentence_progress,
            self._op_measure_conflict,
            self._op_allocate_control,
            self._op_regulate_competing_representations,
            self._op_determine_interpretation_acceptance,
            self._op_coordinate_subsystems,
            self._op_identify_semantic_ambiguity,
            self._op_prepare_interpretations,
            self._op_retrieve_supporting_semantic_knowledge,
            self._op_evaluate_grammatical_compatibility,
            self._op_evaluate_contextual_compatibility,
            self._op_integrate_semantic_evidence,
            self._op_run_recurrent_competition,
            self._op_apply_abstention,
            self._op_revise_interpretations,
            self._op_encode_task_sets,
            self._op_detect_task_switches,
            self._op_preserve_suspended_task_states,
            self._op_reconfigure_task_control,
            self._op_manage_task_interference,
            self._op_monitor_new_task,
            self._op_resume_suspended_tasks,
            self._op_learn_task_switching_history,
            self._op_integrate_cognitive_state,
            self._op_evaluate_processing_completion,
        )
        for operation in operations:
            operation(state, request)

        biological = self._simulator.snapshot() if self._simulator else {
            "model": "disabled", "operation_count": 0, "layers": {},
            "scope": "Neural-reference simulation was disabled by configuration.",
        }
        unresolved_slots = sum(
            len(frame.get("unresolved_slots", ())) for frame in state["structures"]
        )
        probabilities = state.get("posterior", [])
        probability_sum = sum(probabilities)
        buffer_sizes = {name: len(items) for name, items in state["buffers"].items()}
        valid = (
            len(state["trace"]) == 40
            and [row["operation_id"] for row in state["trace"]]
            == [f"WMCC-{index:03d}" for index in range(1, 41)]
            and all(size <= self.config.buffer_capacity for size in buffer_sizes.values())
            and (not probabilities or math.isclose(probability_sum, 1.0, rel_tol=1e-7, abs_tol=1e-7))
            and all(math.isfinite(value) for value in probabilities)
            and (not self._simulator or biological.get("operation_count") == 40)
        )
        warnings = list(state.get("warnings", []))
        if state["structures"]:
            warnings.append("Unfilled syntactic slots are retained; this module does not generate missing words.")
        if state["resolution"].get("selected") is None and len(state["candidates"]) > 1:
            warnings.append("Semantic interpretation was not committed because confidence or margin was insufficient.")
        if not request.interpretations:
            warnings.append("No explicit interpretation set was supplied; one neutral surface-reading hypothesis was used.")
        diagnostics = {
            "valid": bool(valid), "stage_count": len(STAGE_REGISTRY),
            "operation_count": len(state["trace"]),
            "expected_operation_count": len(OPERATION_REGISTRY),
            "operation_ids_complete": valid,
            "stage_operation_counts": {stage.stage_id: stage.operation_count for stage in STAGE_REGISTRY},
            "token_count": len(state["tokens"]), "buffer_sizes": buffer_sizes,
            "chunk_count": len(state.get("chunks", [])),
            "incomplete_structure_count": len(state["structures"]),
            "unresolved_slot_count": unresolved_slots,
            "interpretation_count": len(state["candidates"]),
            "normalized_posterior_sum": probability_sum,
            "semantic_entropy": state["resolution"].get("normalized_entropy", 0.0),
            "task_switch_count": len(state["task"].get("switch_events", [])),
            "processing_complete": state.get("processing_completion", {}).get("complete", False),
            "warnings": warnings,
        }
        metadata = {
            "language": request.language, "source_reference": request.source_reference,
            "configuration": self.config.to_dict(),
            "biological_scope": biological.get("scope", ""),
            "operation_registry": "WMCC-001..WMCC-040 from Brain Language Algorithm.md",
            "inference_scope": (
                "Rule-based and evidence-driven software prototype. It is not a validated theory "
                "of human working memory or a general-purpose language parser."
            ),
        }
        return WorkingMemoryResult(
            sequence_id=request.sequence_id,
            original_text=state.get("original_text", ""),
            tokens=list(state["tokens"]), buffers=state["buffers"],
            incomplete_structures=state.get("incomplete_report", {}),
            executive_control=state["control"], semantic_resolution=state["resolution"],
            task_switching=state["task"], biological=biological,
            operation_trace=state["trace"], diagnostics=diagnostics, metadata=metadata,
        )

    def _record(self, state: dict[str, Any], operation_id: str,
                output: Mapping[str, Any], signal: float = 0.0) -> None:
        """Append one canonical operation trace row and advance its linked neural-reference population."""
        if state.get("_suppress_records"):
            return
        spec: OperationSpec = _OPERATION_BY_ID[operation_id]
        neural = (
            self._simulator.step(operation_id, spec.stage_id, _clip(signal))
            if self._simulator else {"model": "disabled", "operation_id": operation_id}
        )
        state["trace"].append({
            "operation_id": spec.operation_id, "stage_id": spec.stage_id,
            "operation_name": spec.operation_name, "functional_output": dict(output),
            "mathematical_form": spec.mathematical_form,
            "biological_reference": spec.biological_reference, "biological_update": neural,
        })

    def _run_silently(self, state: dict[str, Any], operation: Any,
                      request: WorkingMemoryRequest) -> None:
        """Run a supporting calculation without creating a second canonical operation trace row."""
        previous = bool(state.get("_suppress_records", False))
        state["_suppress_records"] = True
        try:
            operation(state, request)
        finally:
            state["_suppress_records"] = previous

    def _op_initialize_cognitive_environment(
        self, state: dict[str, Any], request: WorkingMemoryRequest,
    ) -> None:
        """Initialize the shared working-memory, syntax, control, semantic, and task state containers."""
        state.update({
            "original_text": "", "received_input": {}, "tokens": [], "token_records": [],
            "buffers": {}, "buffer_cross_references": [], "chunks": [], "evicted_items": {},
            "retrieval_candidates": [], "structures": [], "syntactic_stack": [],
            "pending_dependencies": [], "slot_candidates": [], "resolved_slots": [],
            "grammatical_expectations": [], "incomplete_report": {},
            "control": {}, "candidates": [], "posterior": [], "candidate_scores": [],
            "resolution": {}, "task": {}, "coordination_messages": [], "warnings": [],
        })
        state["cognitive_state"] = {
            "initialized": True, "subsystems": [
                "working_memory", "syntactic_maintenance", "executive_control",
                "semantic_resolution", "task_management",
            ],
        }
        self._record(state, "WMCC-001", {
            "initialized": True, "subsystems": state["cognitive_state"]["subsystems"],
        }, 0.1)

    def _op_receive_linguistic_information(
        self, state: dict[str, Any], request: WorkingMemoryRequest,
    ) -> None:
        """Receive the supplied linguistic sample and retain its source and representation type."""
        raw_text = request.text if request.text is not None else " ".join(request.tokens or ())
        state["original_text"] = raw_text
        state["received_input"] = {
            "text": raw_text, "tokens": list(request.tokens) if request.tokens is not None else None,
            "representation_type": "token_sequence" if request.tokens is not None else "surface_text",
            "language": request.language, "source_reference": request.source_reference,
            "upstream_syntax_supplied": request.syntax_result is not None,
            "partial_structure_count": len(request.partial_structures),
        }
        self._record(state, "WMCC-002", {
            "representation_type": state["received_input"]["representation_type"],
            "character_count": len(raw_text),
            "source_reference": request.source_reference,
            "upstream_syntax_supplied": request.syntax_result is not None,
        }, min(1.0, len(raw_text) / 200.0))

    def _op_chunk_stored_information(
        self, state: dict[str, Any], request: WorkingMemoryRequest,
    ) -> None:
        """Create surface- or caller-defined chunks while preserving references to their source tokens."""
        supplied = request.context.get("chunks", ())
        if isinstance(supplied, Mapping):
            supplied = [supplied]
        chunks: list[dict[str, Any]] = []
        if supplied:
            for index, raw in enumerate(supplied):
                item = dict(raw) if isinstance(raw, Mapping) else {"token_positions": list(raw)}
                raw_positions = item.get("token_positions", ())
                if isinstance(raw_positions, (str, int)):
                    raw_positions = [raw_positions]
                positions = [
                    int(value) for value in raw_positions
                    if str(value).lstrip("-").isdigit() and 0 <= int(value) < len(state["tokens"])
                ]
                if not positions and item.get("text"):
                    wanted = str(item["text"]).casefold()
                    positions = [record["index"] for record in state["token_records"]
                                 if record["surface"].casefold() in wanted.split()]
                chunks.append({
                    "chunk_id": str(item.get("chunk_id", f"chunk-{index + 1}")),
                    "token_positions": positions, "text": str(item.get("text", " ".join(
                        state["tokens"][position] for position in positions
                    ))),
                    "chunk_kind": str(item.get("chunk_kind", "caller_defined")),
                    "evidence_source": "caller_supplied",
                })
        else:
            current: list[int] = []
            for record in state["token_records"]:
                if not record["is_word"]:
                    if current:
                        chunks.append({
                            "chunk_id": f"surface-{len(chunks) + 1}",
                            "token_positions": current,
                            "text": " ".join(state["tokens"][position] for position in current),
                            "chunk_kind": "surface_sequence",
                            "evidence_source": "surface_contiguity_only",
                        })
                        current = []
                    continue
                current.append(record["index"])
            if current:
                chunks.append({
                    "chunk_id": f"surface-{len(chunks) + 1}", "token_positions": current,
                    "text": " ".join(state["tokens"][position] for position in current),
                    "chunk_kind": "surface_sequence", "evidence_source": "surface_contiguity_only",
                })
        for chunk in chunks:
            chunk["token_references"] = [
                {"position": position, "token": state["tokens"][position]}
                for position in chunk["token_positions"]
            ]
        state["chunks"] = chunks
        for items in state["buffers"].values():
            for item in items:
                item["chunk_ids"] = [
                    chunk["chunk_id"] for chunk in chunks
                    if item.get("position") in chunk["token_positions"]
                ]
        self._record(state, "WMCC-005", {
            "chunk_count": len(chunks), "chunks": chunks[:20],
            "semantic_claims_inferred": False if not supplied else None,
        }, min(1.0, len(chunks) / max(1, len(state["token_records"]))))

    def _op_maintain_cognitive_goals(
        self, state: dict[str, Any], request: WorkingMemoryRequest,
    ) -> None:
        """Keep the active comprehension objective and its declared constraints available to control."""
        goal_features = dict(request.goals.get("features", {}))
        goal_name = str(request.goals.get("name", request.context.get(
            "current_goal", "understand_sentence",
        )))
        state["control"].update({
            "current_goal": goal_name, "goal_features": goal_features,
            "goal_priority": _clip(float(request.goals.get(
                "priority", request.context.get("goal_priority", 0.5),
            ))),
            "goal_constraints": list(request.goals.get("constraints", ())),
        })
        self._record(state, "WMCC-015", {
            "goal": goal_name, "goal_priority": state["control"]["goal_priority"],
            "goal_feature_count": len(goal_features),
            "constraint_count": len(state["control"]["goal_constraints"]),
        }, state["control"]["goal_priority"])

    def _op_monitor_sentence_progress(
        self, state: dict[str, Any], request: WorkingMemoryRequest,
    ) -> None:
        """Estimate sentence-processing quality from memory, syntax, reference, and available semantic status."""
        syntax_complete = not any(frame.get("unresolved_slots") for frame in state["structures"])
        dependency_complete = not state["pending_dependencies"]
        memory_load = sum(len(items) for items in state["buffers"].values())
        memory_capacity = max(1, self.config.buffer_capacity * max(1, len(state["buffers"])))
        reference_clear = not bool(request.context.get("unresolved_references", ()))
        status = {
            "syntactic_completeness": float(syntax_complete),
            "dependency_completion": float(dependency_complete),
            "memory_availability": _clip(1.0 - memory_load / memory_capacity),
            "reference_resolution": float(reference_clear),
            "semantic_consistency": 0.5,
            "interpretation_confidence": 0.0,
            "contextual_relevance": _clip(float(request.context.get("context_relevance", 0.5))),
        }
        weights = {
            "syntactic_completeness": 0.22, "dependency_completion": 0.18,
            "memory_availability": 0.12, "reference_resolution": 0.14,
            "semantic_consistency": 0.14, "interpretation_confidence": 0.10,
            "contextual_relevance": 0.10,
        }
        quality = sum(status[key] * weight for key, weight in weights.items())
        state["processing_status"] = status
        state["processing_quality"] = _clip(quality)
        self._record(state, "WMCC-016", {
            "status": status, "processing_quality": state["processing_quality"],
            "memory_item_count": memory_load,
        }, state["processing_quality"])

    def _op_coordinate_subsystems(
        self, state: dict[str, Any], request: WorkingMemoryRequest,
    ) -> None:
        """Coordinate cross-subsystem requests without allowing one subsystem to overwrite another."""
        messages = []
        if state["incomplete_report"].get("unresolved_slot_count", 0):
            messages.append({"from": "syntax", "to": "working_memory", "request": "protect_dependency_items"})
            messages.append({"from": "syntax", "to": "semantic_resolution", "request": "retain_open_structure_alternatives"})
        if state["control"].get("control_demand", 0.0) > 0.0:
            messages.append({"from": "executive_control", "to": "semantic_resolution", "request": "preserve_ranked_candidates"})
        if request.context.get("unresolved_references"):
            messages.append({"from": "discourse_context", "to": "working_memory", "request": "retrieve_reference_candidates"})
        messages.append({"from": "controller", "to": "task_management", "request": "maintain_active_task"})
        state["coordination_messages"] = messages
        state["cognitive_state"]["stage_snapshot"] = {
            "buffer_count": len(state["buffers"]),
            "open_structure_count": len(state["structures"]),
            "control_demand": state["control"].get("control_demand", 0.0),
        }
        self._record(state, "WMCC-021", {
            "message_count": len(messages), "messages": messages,
        }, min(1.0, len(messages) / 5.0))

    def _op_identify_semantic_ambiguity(
        self, state: dict[str, Any], request: WorkingMemoryRequest,
    ) -> None:
        """Identify ambiguity sources from supplied alternatives and annotations without inventing ambiguity."""
        sources = list(request.context.get("ambiguity_sources", ()) or ())
        terms = request.context.get("ambiguous_terms", ())
        if isinstance(terms, str):
            terms = [terms]
        for term in terms:
            sources.append({"kind": "annotated_term", "term": str(term), "source": "caller_context"})
        if len(request.interpretations) > 1:
            sources.append({
                "kind": "multiple_interpretation_hypotheses",
                "hypothesis_ids": [
                    str(item.get("hypothesis_id", item.get("id", f"H-{index + 1:03d}")))
                    for index, item in enumerate(request.interpretations)
                ],
                "source": "caller_supplied",
            })
        priors = [_clip(float(item.get("prior", 0.5))) for item in request.interpretations]
        if priors and sum(priors) > 0:
            total = sum(priors)
            probs = [value / total for value in priors]
            entropy = -sum(value * math.log(max(value, 1e-12)) for value in probs)
            normalized_entropy = entropy / math.log(len(probs)) if len(probs) > 1 else 0.0
        else:
            normalized_entropy = 0.0
        state["ambiguity_sources"] = sources
        state["ambiguity_entropy"] = _clip(normalized_entropy)
        self._record(state, "WMCC-022", {
            "ambiguity_detected": bool(sources or len(request.interpretations) > 1),
            "source_count": len(sources), "sources": sources,
            "prior_entropy": state["ambiguity_entropy"],
        }, state["ambiguity_entropy"])

    def _op_retrieve_supporting_semantic_knowledge(
        self, state: dict[str, Any], request: WorkingMemoryRequest,
    ) -> None:
        """Retrieve only declared concept features or relation support for each interpretation candidate."""
        knowledge = request.context.get("semantic_knowledge", {})
        by_id = knowledge if isinstance(knowledge, Mapping) else {}
        support_rows = []
        for candidate in state["candidates"]:
            explicit = candidate.get("knowledge_score")
            evidence_source = "candidate_explicit" if explicit is not None else None
            keyed = by_id.get(candidate["hypothesis_id"]) if isinstance(by_id, Mapping) else None
            if explicit is None and keyed is not None:
                if isinstance(keyed, Mapping):
                    explicit = _cosine_similarity(candidate.get("features", {}), keyed)
                else:
                    explicit = _clip(float(keyed))
                evidence_source = "context.semantic_knowledge"
            if explicit is None and isinstance(knowledge, Mapping) and knowledge:
                feature_map = knowledge.get("features", knowledge)
                if isinstance(feature_map, Mapping) and all(
                    isinstance(value, (int, float)) for value in feature_map.values()
                ):
                    explicit = _cosine_similarity(candidate.get("features", {}), feature_map)
                    evidence_source = "context.semantic_knowledge_features"
            if explicit is not None:
                candidate["world_score"] = _clip(float(explicit))
                candidate["knowledge_score"] = candidate["world_score"]
            else:
                candidate["knowledge_score"] = candidate.get("world_score", 0.5)
                evidence_source = candidate.get("evidence_source", "neutral_or_caller_score")
            candidate["knowledge_evidence_source"] = evidence_source
            support_rows.append({
                "hypothesis_id": candidate["hypothesis_id"],
                "knowledge_score": candidate["knowledge_score"],
                "source": evidence_source,
                "evidence_available": evidence_source not in {"neutral_or_caller_score", None},
            })
        state["knowledge_support"] = support_rows
        mean_score = sum(item["knowledge_score"] for item in support_rows) / max(1, len(support_rows))
        self._record(state, "WMCC-024", {
            "support": support_rows, "mean_knowledge_score": mean_score,
            "external_knowledge_invented": False,
        }, mean_score)

    def _op_evaluate_grammatical_compatibility(
        self, state: dict[str, Any], request: WorkingMemoryRequest,
    ) -> None:
        """Score grammatical support using candidate annotations and supplied syntax evidence only."""
        syntax = _as_mapping(request.syntax_result)
        support_map = syntax.get("candidate_support", request.context.get("grammar_support", {}))
        support_map = support_map if isinstance(support_map, Mapping) else {}
        rows = []
        for candidate in state["candidates"]:
            value = candidate.get("grammar_compatibility")
            source = "candidate_explicit" if value is not None else None
            if value is None and candidate["hypothesis_id"] in support_map:
                value = support_map[candidate["hypothesis_id"]]
                source = "upstream_syntax_or_context"
            if value is None:
                value = candidate.get("syntactic_score", 0.5)
                source = "candidate_syntactic_score" if "syntactic_score" in candidate else "neutral_default"
            candidate["grammar_compatibility"] = _clip(float(value))
            candidate["syntactic_score"] = candidate["grammar_compatibility"]
            candidate["grammar_evidence_source"] = source
            rows.append({
                "hypothesis_id": candidate["hypothesis_id"],
                "grammar_compatibility": candidate["grammar_compatibility"], "source": source,
            })
        state["grammar_compatibility"] = rows
        self._record(state, "WMCC-025", {
            "candidate_scores": rows,
            "supplied_syntax_available": request.syntax_result is not None,
        }, sum(item["grammar_compatibility"] for item in rows) / max(1, len(rows)))

    def _op_evaluate_contextual_compatibility(
        self, state: dict[str, Any], request: WorkingMemoryRequest,
    ) -> None:
        """Compare candidate features with explicitly supplied discourse features and preserve missing evidence."""
        context_features = request.context.get(
            "context_features", request.context.get("discourse_features", {}),
        )
        context_features = context_features if isinstance(context_features, Mapping) else {}
        rows = []
        for candidate in state["candidates"]:
            explicit = candidate.get("contextual_compatibility")
            if explicit is not None:
                score, source = _clip(float(explicit)), "candidate_explicit"
            elif context_features and candidate.get("features"):
                score, source = _cosine_similarity(candidate["features"], context_features), "caller_context_features"
            else:
                score = _clip(float(candidate.get("discourse_score", 0.5)))
                source = "candidate_discourse_score" if "discourse_score" in candidate else "neutral_default"
            candidate["contextual_compatibility"] = score
            candidate["discourse_score"] = score
            candidate["context_evidence_source"] = source
            rows.append({
                "hypothesis_id": candidate["hypothesis_id"],
                "contextual_compatibility": score, "source": source,
            })
        state["contextual_compatibility"] = rows
        self._record(state, "WMCC-026", {
            "candidate_scores": rows, "context_features_supplied": bool(context_features),
        }, sum(item["contextual_compatibility"] for item in rows) / max(1, len(rows)))

    def _op_revise_interpretations(
        self, state: dict[str, Any], request: WorkingMemoryRequest,
    ) -> None:
        """Apply explicit new evidence to retained hypotheses and rerun bounded semantic selection when needed."""
        revision_evidence = request.context.get("revision_evidence", {})
        applied = 0
        if isinstance(revision_evidence, Mapping):
            allowed = {
                "prior", "lexical_score", "semantic_score", "syntactic_score",
                "discourse_score", "world_score", "pragmatic_score",
                "grammar_compatibility", "contextual_compatibility",
            }
            for candidate in state["candidates"]:
                update = revision_evidence.get(candidate["hypothesis_id"])
                if not isinstance(update, Mapping):
                    continue
                for field_name, raw in update.items():
                    if field_name in allowed:
                        candidate[field_name] = _clip(float(raw))
                        applied += 1
        if applied:
            self._run_silently(state, self._op_integrate_semantic_evidence, request)
            self._run_silently(state, self._op_run_recurrent_competition, request)
            self._run_silently(state, self._op_apply_abstention, request)
        state["resolution"]["revision_count"] = state["resolution"].get("revision_count", 0) + (1 if applied else 0)
        state["resolution"]["revision_applied"] = applied > 0
        state["resolution"]["revision_evidence_count"] = applied
        self._record(state, "WMCC-030", {
            "revision_applied": applied > 0, "updated_score_count": applied,
            "revision_count": state["resolution"]["revision_count"],
            "selected_after_revision": state["resolution"].get("selected_hypothesis_id"),
        }, min(1.0, applied / max(1, len(state["candidates"]))))

    def _op_preserve_suspended_task_states(
        self, state: dict[str, Any], request: WorkingMemoryRequest,
    ) -> None:
        """Checkpoint the task data needed to resume processing after an explicit task switch."""
        checkpoints = []
        for event in state["task"].get("switch_events", []):
            task_name = event["from_task"]
            checkpoints.append({
                "task": task_name,
                "suspended_at_sequence_index": event["sequence_index"],
                "token_position": len(state.get("tokens", [])),
                "working_memory_references": {
                    name: [item.get("position") for item in items]
                    for name, items in state.get("buffers", {}).items()
                },
                "open_structure_ids": [
                    frame["structure_id"] for frame in state.get("structures", [])
                    if frame.get("unresolved_slots")
                ],
                "candidate_ids": [
                    item["hypothesis_id"] for item in state.get("candidates", [])
                ],
                "goal": state.get("control", {}).get("current_goal"),
                "state_preserved": True,
            })
        state["task"]["suspended_checkpoints"] = checkpoints
        state["task"]["suspended_tasks"] = [
            {"task": item["task"], "suspended_at_sequence_index": item["suspended_at_sequence_index"],
             "state_preserved": item["state_preserved"]}
            for item in checkpoints
        ]
        self._record(state, "WMCC-033", {
            "checkpoint_count": len(checkpoints), "checkpoints": checkpoints,
        }, min(1.0, len(checkpoints) / max(1, len(state["task"].get("tasks", [])))))

    def _op_monitor_new_task(
        self, state: dict[str, Any], request: WorkingMemoryRequest,
    ) -> None:
        """Monitor explicit completion information for the active task without inferring success from its label."""
        supplied = request.context.get("task_completion")
        current = state["task"].get("current_task")
        status: str
        if isinstance(supplied, Mapping):
            value = supplied.get(current)
            status = "complete" if value is True or value == "complete" else (
                "incomplete" if value is False or value == "incomplete" else "unknown"
            )
        elif isinstance(supplied, bool):
            status = "complete" if supplied else "incomplete"
        else:
            status = "unknown"
        pending = list(request.context.get("pending_task_requirements", ()) or ())
        if pending and status == "complete":
            status = "incomplete"
        state["task"]["completion_status"] = status
        state["task"]["pending_requirements"] = pending
        state["task"]["return_decision"] = bool(
            status == "complete" or request.context.get("allow_task_resumption", False)
        )
        self._record(state, "WMCC-036", {
            "active_task": current, "completion_status": status,
            "pending_requirement_count": len(pending),
            "return_decision": state["task"]["return_decision"],
        }, 1.0 if status == "complete" else 0.0)

    def _op_resume_suspended_tasks(
        self, state: dict[str, Any], request: WorkingMemoryRequest,
    ) -> None:
        """Restore task checkpoints when the suspended task reappears and resumption is justified."""
        tasks = state["task"].get("tasks", [])
        resumed = []
        restored = None
        for checkpoint in state["task"].get("suspended_checkpoints", []):
            start = checkpoint["suspended_at_sequence_index"] + 1
            future_positions = [
                index for index, task in enumerate(tasks[start:], start=start)
                if task == checkpoint["task"]
            ]
            allowed = bool(future_positions) and (
                state["task"].get("completion_status") == "complete"
                or request.context.get("allow_task_resumption", False)
                or checkpoint["task"] in tasks[start:]
            )
            if allowed:
                resumed.append({
                    "task": checkpoint["task"], "restored_from_sequence_index": start - 1,
                    "resumed_at_sequence_index": future_positions[0],
                    "checkpoint_restored": True,
                })
                restored = checkpoint
        state["task"]["resumed_tasks"] = sorted({item["task"] for item in resumed})
        state["task"]["resumption_events"] = resumed
        state["task"]["restored_task_state"] = restored
        self._record(state, "WMCC-037", {
            "resumed_tasks": resumed, "restored_checkpoint": restored,
        }, min(1.0, len(resumed) / max(1, len(state["task"].get("suspended_checkpoints", [])))))

    def _op_learn_task_switching_history(
        self, state: dict[str, Any], request: WorkingMemoryRequest,
    ) -> None:
        """Update transition-cost estimates only from explicit observed history, not simulated cost estimates."""
        alpha = _clip(float(request.context.get("task_cost_learning_rate", 0.2)))
        learned = dict(request.context.get("prior_transition_costs", {}) or {})
        history = list(request.context.get("task_history", ()) or ())
        observations = request.context.get("observed_transition_costs", {})
        for row in history:
            if isinstance(row, Mapping) and row.get("observed_cost_ms") is not None:
                key = f"{row.get('from_task')}->{row.get('to_task')}"
                previous = float(learned.get(key, row["observed_cost_ms"]))
                learned[key] = (1.0 - alpha) * previous + alpha * float(row["observed_cost_ms"])
        if isinstance(observations, Mapping):
            for key, raw in observations.items():
                observed = float(raw)
                previous = float(learned.get(key, observed))
                learned[str(key)] = (1.0 - alpha) * previous + alpha * observed
        transition_history = [
            {
                "from_task": event["from_task"], "to_task": event["to_task"],
                "estimated_cost_ms": next((
                    item["estimated_cost_ms"] for item in state["task"].get("switch_costs", [])
                    if item["sequence_index"] == event["sequence_index"]
                ), None),
                "observation_supplied": f"{event['from_task']}->{event['to_task']}" in observations
                if isinstance(observations, Mapping) else False,
            }
            for event in state["task"].get("switch_events", [])
        ]
        state["task"]["task_transition_history"] = transition_history
        state["task"]["learned_transition_costs"] = learned
        state["task"]["learning_source"] = "explicit_observed_history" if history or observations else "no_new_observations"
        self._record(state, "WMCC-038", {
            "history_count": len(transition_history), "learned_transition_costs": learned,
            "learning_source": state["task"]["learning_source"],
        }, min(1.0, len(learned) / 10.0))

    def _op_integrate_cognitive_state(
        self, state: dict[str, Any], request: WorkingMemoryRequest,
    ) -> None:
        """Integrate all subsystem reports while preserving each subsystem's source and state ownership."""
        self._run_silently(state, self._op_build_resolution_report, request)
        self._run_silently(state, self._op_emit_task_state, request)
        state["integrated_cognitive_state"] = {
            "working_memory": {
                "buffer_sizes": {name: len(items) for name, items in state["buffers"].items()},
                "chunk_count": len(state.get("chunks", [])),
            },
            "syntax": {
                "complete": state["incomplete_report"].get("complete", True),
                "open_structure_count": len(state["structures"]),
                "pending_dependency_count": len(state["pending_dependencies"]),
            },
            "executive_control": {
                "goal": state["control"].get("current_goal"),
                "control_demand": state["control"].get("control_demand", 0.0),
                "conflict": state["control"].get("conflict", 0.0),
            },
            "semantics": {
                "selected": state["resolution"].get("selected"),
                "abstained": state["resolution"].get("abstained", True),
                "candidate_count": len(state["candidates"]),
            },
            "task": {
                "current_task": state["task"].get("current_task"),
                "switch_count": len(state["task"].get("switch_events", [])),
                "resumed_tasks": state["task"].get("resumed_tasks", []),
            },
            "coordination_messages": state.get("coordination_messages", []),
        }
        self._record(state, "WMCC-039", {
            "integrated_subsystems": [
                "working_memory", "syntax", "executive_control", "semantics", "task",
            ],
            "candidate_count": len(state["candidates"]),
            "switch_count": len(state["task"].get("switch_events", [])),
        }, state["control"].get("control_demand", 0.0))

    def _op_evaluate_processing_completion(
        self, state: dict[str, Any], request: WorkingMemoryRequest,
    ) -> None:
        """Evaluate input-end, syntax, semantic, reference, and conflict conditions for completion."""
        syntax_complete = bool(state["incomplete_report"].get("complete", True))
        semantic_ready = bool(
            state["resolution"].get("selected") is not None
            or (len(state["candidates"]) <= 1 and not state["resolution"].get("abstained", False))
        )
        reference_complete = not bool(request.context.get("unresolved_references", ()))
        conflict_clear = not bool(state["resolution"].get("abstained", False))
        end_of_input = bool(request.context.get("end_of_input", True))
        complete = all((
            end_of_input, syntax_complete, semantic_ready, reference_complete, conflict_clear,
        ))
        status = "complete" if complete else (
            "pending_input" if not end_of_input else "partial_or_unresolved"
        )
        state["processing_completion"] = {
            "complete": complete, "status": status, "end_of_input": end_of_input,
            "syntactic_complete": syntax_complete, "semantic_ready": semantic_ready,
            "references_resolved": reference_complete, "blocking_conflict_clear": conflict_clear,
            "unresolved_reference_count": len(request.context.get("unresolved_references", ()) or ()),
        }
        self._record(state, "WMCC-040", state["processing_completion"], 1.0 if complete else 0.0)

    def _op_tokenize_and_encode(self, state: dict[str, Any], request: WorkingMemoryRequest) -> None:
        """Create ordered token records while preserving the caller's supplied token boundaries."""
        tokens = list(request.tokens) if request.tokens is not None else _TOKEN_PATTERN.findall(request.text or "")
        tokens = tokens[:self.config.max_tokens]
        state["tokens"] = tokens
        state["token_records"] = [
            {"index": index, "surface": token, "normalized": token.casefold(),
             "is_word": any(character.isalnum() for character in token)}
            for index, token in enumerate(tokens)
        ]
        self._record(state, "WMCC-003", {
            "token_count": len(tokens), "token_records": state["token_records"],
        }, min(1.0, len(tokens) / max(1, self.config.max_tokens)))

    def _op_allocate_linguistic_buffers(
        self, state: dict[str, Any], request: WorkingMemoryRequest,
    ) -> None:
        """Create separate modality-labelled token-reference buffers without merging their functions."""
        buffers: dict[str, list[dict[str, Any]]] = {
            "phonological": [], "lexical_semantic": [], "syntactic": [],
            "event_discourse": [], "executive_goal": [],
        }
        for record in state["token_records"]:
            token = record["surface"]
            normalized = record["normalized"]
            base = {"position": record["index"], "token": token, "normalized": normalized,
                    "activation": 0.0, "rehearsal_count": 0}
            if record["is_word"]:
                buffers["phonological"].append(dict(base))
                buffers["lexical_semantic"].append(dict(base))
                if normalized in _SYNTACTIC_MARKERS:
                    buffers["syntactic"].append(dict(base))
                if normalized not in _STOPWORDS:
                    buffers["event_discourse"].append(dict(base))
            else:
                buffers["phonological"].append(dict(base))
                buffers["syntactic"].append(dict(base))
        for feature, strength in request.goals.get("features", {}).items():
            buffers["executive_goal"].append({
                "position": -1, "token": str(feature), "normalized": str(feature).casefold(),
                "activation": _clip(float(strength)), "rehearsal_count": 0,
                "representation_type": "goal_feature",
            })
        state["buffers"] = buffers
        state["buffer_cross_references"] = [
            {"token_position": record["index"], "token": record["surface"],
             "buffer_memberships": [
                 name for name, items in buffers.items()
                 if any(item.get("position") == record["index"] for item in items)
             ]}
            for record in state["token_records"]
        ]
        self._record(state, "WMCC-004", {
            "buffers": list(buffers),
            "item_counts": {name: len(items) for name, items in buffers.items()},
            "cross_reference_count": len(state["buffer_cross_references"]),
        }, min(1.0, sum(len(items) for items in buffers.values()) / 60.0))

    def _op_calculate_buffer_activation(
        self, state: dict[str, Any], request: WorkingMemoryRequest,
    ) -> None:
        """Estimate each item activation from token age and explicitly supplied focus cues."""
        focus = request.context.get("focus_cues", ())
        if isinstance(focus, str):
            focus = [focus]
        focus_terms = {str(item).casefold() for item in focus}
        last_position = max(0, len(state["tokens"]) - 1)
        for items in state["buffers"].values():
            for item in items:
                age = last_position - item["position"]
                salience = 1.15 if item["normalized"] in focus_terms else 1.0
                item["activation"] = _clip(
                    salience * math.exp(-self.config.activation_decay_per_token * age)
                )
        self._run_silently(state, self._op_rehearse_active_items, request)
        values = [
            item["activation"] for items in state["buffers"].values() for item in items
        ]
        average = sum(values) / max(1, len(values))
        state["mean_buffer_activation"] = average
        interference = {}
        lexical_items = state["buffers"].get("lexical_semantic", [])
        for item in lexical_items:
            item["similarity_interference"] = sum(
                _surface_similarity(item["normalized"], other["normalized"])
                * other["activation"]
                for other in lexical_items if other["position"] != item["position"]
            )
            interference[str(item["position"])] = item["similarity_interference"]
        state["activation_interference"] = interference
        self._record(state, "WMCC-006", {
            "mean_activation": average, "decay_per_token": self.config.activation_decay_per_token,
            "focus_cue_count": len(focus_terms), "interference_estimates": interference,
        }, average)

    def _op_decay_and_capacity(self, state: dict[str, Any], request: WorkingMemoryRequest) -> None:
        """Apply an activation-ranked capacity limit and retain the evicted items for inspection."""
        evicted: dict[str, list[dict[str, Any]]] = {}
        for name, items in state["buffers"].items():
            ranked = sorted(items, key=lambda item: (item["activation"], item["position"]), reverse=True)
            retained = ranked[:self.config.buffer_capacity]
            retained_keys = {
                (item.get("position"), item.get("normalized"), item.get("representation_type"))
                for item in retained
            }
            evicted[name] = [
                dict(item) for item in items
                if (item.get("position"), item.get("normalized"), item.get("representation_type"))
                not in retained_keys
            ]
            state["buffers"][name] = sorted(retained, key=lambda item: item["position"])
        state["evicted_items"] = evicted
        total_evicted = sum(len(items) for items in evicted.values())
        self._record(state, "WMCC-007", {
            "capacity_per_buffer": self.config.buffer_capacity,
            "retained_counts": {name: len(items) for name, items in state["buffers"].items()},
            "evicted_counts": {name: len(items) for name, items in evicted.items()},
        }, min(1.0, total_evicted / max(1, sum(len(v) for v in evicted.values()) + 20)))

    def _op_rehearse_active_items(self, state: dict[str, Any], request: WorkingMemoryRequest) -> None:
        """Refresh the most active items in each buffer and record rehearsal counts."""
        for items in state["buffers"].values():
            focus_items = sorted(items, key=lambda item: item["activation"], reverse=True)[:3]
            focus_positions = {item["position"] for item in focus_items}
            for item in items:
                if item["position"] in focus_positions:
                    item["activation"] = _clip(item["activation"] + self.config.rehearsal_gain)
                    item["rehearsal_count"] += 1
        total = sum(len(items) for items in state["buffers"].values())
        average = sum(
            item["activation"] for items in state["buffers"].values() for item in items
        ) / max(1, total)
        self._record(state, "WMCC-006", {
            "rehearsal_gain": self.config.rehearsal_gain,
            "refreshed_item_count": sum(
                item["rehearsal_count"] for items in state["buffers"].values() for item in items
            ),
            "mean_activation_after_rehearsal": average,
        }, average)

    def _op_retrieve_context_cues(
        self, state: dict[str, Any], request: WorkingMemoryRequest,
    ) -> None:
        """Rank maintained lexical items by activation and explicit retrieval-cue overlap."""
        cues = request.context.get("retrieval_cues", ())
        prior = request.context.get("prior_tokens", ())
        if isinstance(cues, str):
            cues = [cues]
        if isinstance(prior, str):
            prior = [prior]
        cue_terms = {str(value).casefold() for value in list(cues) + list(prior)}
        candidates = []
        for item in state["buffers"].get("lexical", []):
            match = 1.0 if item["normalized"] in cue_terms else 0.0
            score = 0.7 * item["activation"] + 0.3 * match
            candidates.append({
                "position": item["position"], "token": item["token"],
                "activation": item["activation"], "cue_match": match,
                "retrieval_score": _clip(score),
            })
        candidates.sort(key=lambda item: (item["retrieval_score"], item["position"]), reverse=True)
        state["retrieval_candidates"] = candidates[:self.config.buffer_capacity]
        self._record(state, "WMCC-008", {
            "cue_terms": sorted(cue_terms), "retrieval_candidates": state["retrieval_candidates"][:10],
        }, candidates[0]["retrieval_score"] if candidates else 0.0)

    def _op_extract_incomplete_structures(
        self, state: dict[str, Any], request: WorkingMemoryRequest,
    ) -> None:
        """Extract caller-declared incomplete phrase frames and unresolved syntax dependencies."""
        syntax = _as_mapping(request.syntax_result)
        raw_structures = list(request.partial_structures)
        raw_structures.extend(syntax.get("incomplete_structures", ()) or ())
        raw_structures.extend(request.context.get("incomplete_structures", ()) or ())
        structures: list[dict[str, Any]] = []
        seen: set[str] = set()
        for index, raw in enumerate(raw_structures[:self.config.max_syntactic_frames]):
            item = dict(raw)
            frame_id = str(item.get("structure_id", item.get("id", f"frame-{index + 1}")))
            if frame_id in seen:
                continue
            seen.add(frame_id)
            expected = item.get("expected_slots", item.get("open_slots", ())) or ()
            filled = item.get("filled_slots", item.get("filled", ())) or ()
            if isinstance(expected, str):
                expected = [expected]
            if isinstance(filled, str):
                filled = [filled]
            filled_names = {str(value) for value in filled}
            pending = [str(slot) for slot in expected if str(slot) not in filled_names]
            if not pending and item.get("status") in {"open", "incomplete", "unresolved"}:
                pending = [str(item.get("expected", "unspecified"))]
            structures.append({
                "structure_id": frame_id, "label": str(item.get("label", item.get("type", "syntactic_frame"))),
                "depth": max(0, int(item.get("depth", 0) or 0)),
                "parent_id": item.get("parent_id"),
                "expected_slots": [str(slot) for slot in expected],
                "filled_slots": [str(slot) for slot in filled],
                "unresolved_slots": pending,
                "candidate_bindings": [dict(value) for value in item.get("candidate_bindings", ())],
                "source": item.get("source", "caller_or_upstream_syntax"),
            })
        state["structures"] = structures
        self._record(state, "WMCC-009", {
            "frame_count": len(structures),
            "frames": [
                {"structure_id": frame["structure_id"], "label": frame["label"],
                 "unresolved_slots": frame["unresolved_slots"]}
                for frame in structures
            ],
        }, min(1.0, len(structures) / max(1, self.config.max_syntactic_frames)))

    def _op_build_syntactic_stack(
        self, state: dict[str, Any], request: WorkingMemoryRequest,
    ) -> None:
        """Order incomplete structures by nesting depth and preserve parent-child relationships."""
        stack = sorted(
            state["structures"], key=lambda frame: (frame["depth"], frame["structure_id"]),
        )
        state["syntactic_stack"] = [
            {"stack_index": index, "structure_id": frame["structure_id"],
             "label": frame["label"], "depth": frame["depth"],
             "parent_id": frame["parent_id"], "open_slot_count": len(frame["unresolved_slots"])}
            for index, frame in enumerate(stack)
        ]
        self._record(state, "WMCC-012", {
            "stack_depth": len(stack), "stack": state["syntactic_stack"],
        }, min(1.0, len(stack) / max(1, self.config.max_syntactic_frames)))

    def _op_track_unresolved_dependencies(
        self, state: dict[str, Any], request: WorkingMemoryRequest,
    ) -> None:
        """Retain only dependencies explicitly marked unresolved by syntax or discourse context."""
        syntax = _as_mapping(request.syntax_result)
        dependencies = list(syntax.get("unresolved_dependencies", ()) or ())
        dependencies.extend(request.context.get("unresolved_dependencies", ()) or ())
        for item in syntax.get("long_distance_dependencies", ()) or ():
            if item.get("unresolved") is True or item.get("resolved") is False or item.get("status") == "open":
                dependencies.append(item)
        unique: dict[str, dict[str, Any]] = {}
        for index, raw in enumerate(dependencies):
            item = dict(raw)
            dependency_id = str(item.get("dependency_id", item.get("id", f"dependency-{index + 1}")))
            unique.setdefault(dependency_id, {
                "dependency_id": dependency_id, "filler": item.get("filler"),
                "expected_role": item.get("expected_role", item.get("role")),
                "target_clause": item.get("target_clause"), "status": "unresolved",
                "source": item.get("source", "upstream_syntax"),
            })
        state["pending_dependencies"] = list(unique.values())
        self._record(state, "WMCC-011", {
            "pending_dependency_count": len(unique),
            "pending_dependencies": state["pending_dependencies"],
        }, min(1.0, len(unique) / 5.0))

    def _op_score_slot_retrieval_cues(
        self, state: dict[str, Any], request: WorkingMemoryRequest,
    ) -> None:
        """Score explicit candidate bindings against each open syntactic slot and its confidence."""
        contextual = list(request.context.get("syntactic_candidates", ()) or ())
        scored: list[dict[str, Any]] = []
        for frame in state["structures"]:
            for slot in frame["unresolved_slots"]:
                candidates = list(frame["candidate_bindings"]) + contextual
                for candidate in candidates:
                    candidate_slot = str(candidate.get("slot", candidate.get("expected_role", "")))
                    slot_match = 1.0 if candidate_slot.casefold() == slot.casefold() else (
                        0.5 if candidate_slot in {"*", "any"} else 0.0
                    )
                    confidence = _clip(float(candidate.get("confidence", candidate.get("score", 0.0))))
                    score = 0.6 * confidence + 0.4 * slot_match
                    if slot_match > 0.0:
                        scored.append({
                            "structure_id": frame["structure_id"], "slot": slot,
                            "candidate": candidate.get("value", candidate.get("label")),
                            "confidence": confidence, "slot_match": slot_match,
                            "retrieval_score": _clip(score),
                        })
        scored.sort(key=lambda item: item["retrieval_score"], reverse=True)
        state["slot_candidates"] = scored
        declared = request.context.get("grammatical_expectations", ())
        expectations = [dict(item) if isinstance(item, Mapping) else {"expected": str(item)}
                        for item in declared]
        for frame in state["structures"]:
            for slot in frame["unresolved_slots"]:
                normalized = slot.casefold()
                expected_category = (
                    "noun_phrase" if any(term in normalized for term in ("object", "subject", "agent", "patient"))
                    else "complement_or_constituent"
                )
                expectations.append({
                    "structure_id": frame["structure_id"], "slot": slot,
                    "expected_category": expected_category,
                    "source": "slot_label_heuristic",
                    "confidence": 0.5,
                })
        state["grammatical_expectations"] = expectations
        self._record(state, "WMCC-010", {
            "scored_candidate_count": len(scored), "top_candidates": scored[:10],
            "grammatical_expectations": expectations,
        }, max(
            scored[0]["retrieval_score"] if scored else 0.0,
            min(1.0, len(expectations) / 5.0),
        ))

    def _op_bind_supported_slots(self, state: dict[str, Any], request: WorkingMemoryRequest) -> None:
        """Fill an open slot only when an explicitly supplied candidate clears the retrieval threshold."""
        bound: list[dict[str, Any]] = []
        for candidate in state["slot_candidates"]:
            if candidate["retrieval_score"] < self.config.retrieval_threshold:
                continue
            frame = next(
                (item for item in state["structures"]
                 if item["structure_id"] == candidate["structure_id"]),
                None,
            )
            if frame is None or candidate["slot"] not in frame["unresolved_slots"]:
                continue
            frame["unresolved_slots"].remove(candidate["slot"])
            frame["filled_slots"].append(candidate["slot"])
            bound.append(dict(candidate))
        state["resolved_slots"] = bound
        self._record(state, "WMCC-013", {
            "resolved_slot_count": len(bound), "resolved_slots": bound,
            "remaining_open_slots": sum(len(frame["unresolved_slots"]) for frame in state["structures"]),
        }, min(1.0, len(bound) / max(1, len(state["slot_candidates"]))))

    def _op_emit_incomplete_state(self, state: dict[str, Any], request: WorkingMemoryRequest) -> None:
        """Serialize open frames and dependencies without inserting unsupported words or constituents."""
        protected_positions = {
            int(item["token_position"]) for item in state["pending_dependencies"]
            if item.get("token_position") is not None
        }
        links = []
        for dependency in state["pending_dependencies"]:
            token_position = dependency.get("token_position", dependency.get("filler_position"))
            if token_position is not None:
                links.append({
                    "dependency_id": dependency["dependency_id"],
                    "token_position": int(token_position), "protection": "dependency_retrieval_cue",
                })
        for frame in state["structures"]:
            for position in frame.get("referenced_token_positions", ()):
                protected_positions.add(int(position))
        for items in state["buffers"].values():
            for item in items:
                if item.get("position") in protected_positions:
                    item["dependency_protected"] = True
                    item["activation"] = _clip(item.get("activation", 0.0) + 0.1)
        state["dependency_memory_links"] = links
        state["incomplete_report"] = {
            "stack": state["syntactic_stack"], "frames": state["structures"],
            "pending_dependencies": state["pending_dependencies"],
            "resolved_slots": state["resolved_slots"],
            "dependency_memory_links": links,
            "unresolved_slot_count": sum(
                len(frame["unresolved_slots"]) for frame in state["structures"]
            ),
            "complete": not any(frame["unresolved_slots"] for frame in state["structures"])
            and not state["pending_dependencies"],
        }
        self._record(state, "WMCC-014", {
            "complete": state["incomplete_report"]["complete"],
            "unresolved_slot_count": state["incomplete_report"]["unresolved_slot_count"],
            "pending_dependency_count": len(state["pending_dependencies"]),
            "dependency_memory_links": links,
            "protected_buffer_items": len(protected_positions),
        }, 1.0 - min(1.0, state["incomplete_report"]["unresolved_slot_count"] / 5.0))

    def _op_prepare_interpretations(
        self, state: dict[str, Any], request: WorkingMemoryRequest,
    ) -> None:
        """Normalize supplied interpretation hypotheses and preserve all distinct candidate readings."""
        raw_candidates = list(request.interpretations)
        if not raw_candidates:
            surface = request.text if request.text is not None else " ".join(request.tokens or ())
            raw_candidates = [{
                "hypothesis_id": "H-001", "interpretation": f"surface reading: {surface}",
                "prior": 1.0, "lexical_score": 0.5, "semantic_score": 0.5,
                "syntactic_score": 0.5, "discourse_score": 0.5, "world_score": 0.5,
                "features": {}, "evidence_source": "neutral_fallback",
            }]
        candidates = []
        for index, raw in enumerate(raw_candidates[:self.config.max_interpretations]):
            item = dict(raw)
            candidates.append({
                "hypothesis_id": str(item.get("hypothesis_id", item.get("id", f"H-{index + 1:03d}"))),
                "interpretation": str(item.get("interpretation", item.get("label", item.get("text", f"hypothesis {index + 1}")))),
                "prior": _clip(float(item.get("prior", 0.5))),
                "lexical_score": _clip(float(item.get("lexical_score", 0.5))),
                "semantic_score": _clip(float(item.get("semantic_score", 0.5))),
                "syntactic_score": _clip(float(item.get("syntactic_score", 0.5))),
                "discourse_score": _clip(float(item.get("discourse_score", 0.5))),
                "world_score": _clip(float(item.get("world_score", item.get("knowledge_score", 0.5)))),
                "knowledge_score": item.get("knowledge_score"),
                "grammar_compatibility": item.get("grammar_compatibility"),
                "contextual_compatibility": item.get("contextual_compatibility"),
                "pragmatic_score": _clip(float(item.get("pragmatic_score", 0.5))),
                "goal_compatibility": item.get("goal_compatibility"),
                "features": dict(item.get("features", {})),
                "evidence_source": item.get("evidence_source", "caller_supplied"),
            })
        state["candidates"] = candidates
        self._record(state, "WMCC-023", {
            "candidate_count": len(candidates),
            "candidate_ids": [item["hypothesis_id"] for item in candidates],
        }, min(1.0, len(candidates) / max(1, self.config.max_interpretations)))

    def _op_measure_conflict(self, state: dict[str, Any], request: WorkingMemoryRequest) -> None:
        """Estimate interpretation conflict from preliminary score entropy and the top-two margin."""
        candidates = state["candidates"]
        conflict_inputs = candidates or list(request.interpretations)
        preliminary = [
            0.20 * float(item.get("lexical_score", 0.5))
            + 0.28 * float(item.get("semantic_score", 0.5))
            + 0.20 * float(item.get("syntactic_score", 0.5))
            + 0.16 * float(item.get("discourse_score", 0.5))
            + 0.16 * float(item.get("world_score", item.get("knowledge_score", 0.5)))
            + 0.05 * float(item.get("prior", 0.5))
            for item in conflict_inputs[:self.config.max_interpretations]
        ]
        if not preliminary:
            preliminary = [0.5]
        posterior = _softmax(preliminary, self.config.semantic_temperature)
        ordered = sorted(posterior, reverse=True)
        if len(posterior) <= 1:
            entropy, margin = 0.0, 1.0
        else:
            entropy_raw = -sum(value * math.log(max(value, 1e-12)) for value in posterior)
            entropy = entropy_raw / math.log(len(posterior))
            margin = ordered[0] - ordered[1]
        conflict = _clip(0.5 * entropy + 0.5 * (1.0 - margin))
        state["control"].update({
            "conflict": conflict, "preliminary_entropy": entropy,
            "preliminary_margin": margin, "preliminary_posterior": posterior,
        })
        self._record(state, "WMCC-017", {
            "conflict": conflict, "normalized_entropy": entropy, "top_two_margin": margin,
        }, conflict)

    def _op_allocate_control(self, state: dict[str, Any], request: WorkingMemoryRequest) -> None:
        """Allocate a bounded executive-control demand from conflict, open syntax, and goal priority."""
        open_count = sum(len(frame["unresolved_slots"]) for frame in state["structures"])
        open_load = _clip(open_count / max(1, self.config.max_syntactic_frames))
        goal_priority = _clip(float(request.goals.get("priority", request.context.get("goal_priority", 0.5))))
        conflict = state["control"]["conflict"]
        demand = _clip(0.45 * conflict + 0.35 * open_load + 0.20 * goal_priority)
        active_items = [
            {"buffer": name, "position": item.get("position"), "token": item.get("token"),
             "activation": item.get("activation", 0.0)}
            for name, items in state["buffers"].items() for item in items
        ]
        attention = _softmax(
            [float(item["activation"]) + (0.15 if item["buffer"] == "syntactic" else 0.0)
             for item in active_items], 0.5,
        )
        for item, weight in zip(active_items, attention):
            item["attention_weight"] = weight
        state["control"].update({
            "syntactic_open_load": open_load, "goal_priority": goal_priority,
            "control_demand": demand, "attention_allocation": active_items,
            "control_resources": {
                "monitoring": _clip(0.5 + 0.5 * conflict),
                "maintenance": _clip(0.5 + 0.5 * open_load),
                "goal_bias": _clip(goal_priority),
                "attention": _clip(0.5 + 0.5 * conflict),
            },
        })
        self._record(state, "WMCC-018", {
            "control_demand": demand, "open_syntax_load": open_load,
            "goal_priority": goal_priority, "attention_item_count": len(active_items),
            "attention_weight_sum": sum(attention),
        }, demand)

    def _op_evaluate_goal_fit(
        self, state: dict[str, Any], request: WorkingMemoryRequest,
    ) -> None:
        """Compare candidate features with declared goal features or retain an explicit goal score."""
        goal_features = request.goals.get("features", {})
        fits = []
        for candidate in state["candidates"]:
            explicit = candidate["goal_compatibility"]
            fit = (
                _clip(float(explicit)) if explicit is not None
                else _cosine_similarity(candidate["features"], goal_features)
            )
            candidate["goal_fit"] = fit
            fits.append({"hypothesis_id": candidate["hypothesis_id"], "goal_fit": fit})
        state["control"]["goal_fits"] = fits
        self._record(state, "WMCC-019", {
            "goal_fits": fits,
        }, sum(item["goal_fit"] for item in fits) / max(1, len(fits)))

    def _op_regulate_competing_representations(
        self, state: dict[str, Any], request: WorkingMemoryRequest,
    ) -> None:
        """Apply goal-sensitive excitation and lateral inhibition while retaining every supplied interpretation."""
        raw_candidates = list(request.interpretations[:self.config.max_interpretations])
        activations = []
        if raw_candidates:
            for index, candidate in enumerate(raw_candidates):
                prior = _clip(float(candidate.get("prior", 0.5)))
                base = (
                    0.20 * _clip(float(candidate.get("lexical_score", 0.5)))
                    + 0.28 * _clip(float(candidate.get("semantic_score", 0.5)))
                    + 0.20 * _clip(float(candidate.get("syntactic_score", 0.5)))
                    + 0.16 * _clip(float(candidate.get("discourse_score", 0.5)))
                    + 0.16 * _clip(float(candidate.get("world_score", candidate.get("knowledge_score", 0.5))))
                    + 0.05 * prior
                )
                explicit_fit = candidate.get("goal_compatibility")
                goal_fit = (
                    _clip(float(explicit_fit))
                    if explicit_fit is not None
                    else _cosine_similarity(candidate.get("features", {}), request.goals.get("features", {}))
                )
                goal_bias = self.config.control_gain * state["control"].get("control_demand", 0.0) * (
                    0.65 * goal_fit + 0.35 * prior
                )
                activations.append({
                    "hypothesis_id": str(candidate.get("hypothesis_id", candidate.get("id", f"H-{index + 1:03d}"))),
                    "interpretation": str(candidate.get("interpretation", candidate.get("label", candidate.get("text", f"hypothesis {index + 1}")))),
                    "base_activation": base,
                    "goal_fit": goal_fit,
                    "goal_bias": goal_bias,
                    "pre_competition_activation": max(0.0, base + goal_bias),
                })
        regulated = []
        for index, candidate in enumerate(activations):
            competitors = sum(
                other["pre_competition_activation"]
                for other_index, other in enumerate(activations) if other_index != index
            )
            normalized_inhibition = self.config.lateral_inhibition * (
                competitors / max(1, len(activations) - 1)
            )
            regulated.append({
                **candidate,
                "lateral_inhibition": normalized_inhibition,
                "regulated_activation": max(
                    0.0, candidate["pre_competition_activation"] - normalized_inhibition,
                ),
            })
        probabilities = _softmax(
            [item["regulated_activation"] for item in regulated],
            self.config.semantic_temperature,
        )
        for item, probability in zip(regulated, probabilities):
            item["regulated_probability"] = probability
        state["control"]["regulated_hypotheses"] = regulated
        state["control"]["regulated_posterior"] = probabilities
        state["control"]["regulated_alternatives_preserved"] = len(regulated)
        self._record(state, "WMCC-019", {
            "candidate_count": len(regulated),
            "regulated_hypotheses": regulated,
            "posterior_sum": sum(probabilities),
            "all_alternatives_preserved": True,
        }, max(probabilities) if probabilities else 0.0)

    def _op_determine_interpretation_acceptance(
        self, state: dict[str, Any], request: WorkingMemoryRequest,
    ) -> None:
        """Make a provisional control decision from regulated candidates without replacing final semantic resolution."""
        regulated = state["control"].get("regulated_hypotheses", [])
        probabilities = state["control"].get("regulated_posterior", [])
        ranked = sorted(
            zip(regulated, probabilities),
            key=lambda pair: pair[1], reverse=True,
        )
        if ranked:
            best, confidence = ranked[0]
            second_probability = ranked[1][1] if len(ranked) > 1 else 0.0
            margin = confidence - second_probability
            blocking_conflict = bool(request.context.get("blocking_interpretation_conflict", False))
            accepted = (
                confidence >= self.config.confidence_threshold
                and margin >= self.config.ambiguity_margin
                and not blocking_conflict
            )
            decision = {
                "status": "provisionally_accepted" if accepted else "continue_evaluation",
                "accepted": accepted,
                "hypothesis_id": best["hypothesis_id"] if accepted else None,
                "focus_hypothesis_id": best["hypothesis_id"],
                "focus_interpretation": best["interpretation"],
                "confidence": confidence,
                "top_two_margin": margin,
                "confidence_threshold": self.config.confidence_threshold,
                "margin_threshold": self.config.ambiguity_margin,
                "blocking_conflict": blocking_conflict,
                "preserved_alternative_count": len(ranked) - (1 if accepted else 0),
                "is_final_semantic_decision": False,
            }
        else:
            decision = {
                "status": "awaiting_candidate_generation", "accepted": False,
                "hypothesis_id": None, "focus_hypothesis_id": None,
                "focus_interpretation": None, "confidence": 0.0,
                "top_two_margin": 0.0, "preserved_alternative_count": 0,
                "is_final_semantic_decision": False,
            }
        state["control"]["provisional_interpretation_decision"] = decision
        self._record(state, "WMCC-020", decision, decision.get("confidence", 0.0))

    def _op_apply_top_down_bias(
        self, state: dict[str, Any], request: WorkingMemoryRequest,
    ) -> None:
        """Add a bounded goal/prior bias to each candidate while preserving its source evidence scores."""
        demand = state["control"]["control_demand"]
        self._run_silently(state, self._op_evaluate_goal_fit, request)
        for candidate in state["candidates"]:
            goal_fit = candidate.get("goal_fit", 0.5)
            candidate["top_down_bias"] = self.config.control_gain * demand * (
                0.65 * goal_fit + 0.35 * candidate["prior"]
            )
        biases = [item["top_down_bias"] for item in state["candidates"]]
        mean_bias = sum(biases) / max(1, len(biases))
        state["control"]["mean_top_down_bias"] = mean_bias
        self._record(state, "WMCC-019", {
            "candidate_biases": [
                {"hypothesis_id": item["hypothesis_id"], "bias": item["top_down_bias"]}
                for item in state["candidates"]
            ],
        }, _clip(mean_bias))

    def _op_select_control_focus(self, state: dict[str, Any], request: WorkingMemoryRequest) -> None:
        """Select one temporary focus candidate without deleting competing interpretation records."""
        def base_score(item: Mapping[str, Any]) -> float:
            """Combine evidence channels for temporary executive focus selection."""
            return (
                0.20 * item["lexical_score"] + 0.28 * item["semantic_score"]
                + 0.20 * item["syntactic_score"] + 0.16 * item["discourse_score"]
                + 0.16 * item["world_score"] + 0.05 * item["prior"]
                + item.get("top_down_bias", 0.0)
            )
        if not state["candidates"]:
            state["control"].update({
                "focus_hypothesis_id": None, "focus_interpretation": None,
                "alternatives_preserved": 0, "focus_pending_candidate_generation": True,
            })
            self._record(state, "WMCC-020", {
                "focus_hypothesis_id": None, "focus_pending_candidate_generation": True,
            }, 0.0)
            return
        focus = max(state["candidates"], key=base_score)
        state["control"]["focus_hypothesis_id"] = focus["hypothesis_id"]
        state["control"]["focus_interpretation"] = focus["interpretation"]
        state["control"]["alternatives_preserved"] = max(0, len(state["candidates"]) - 1)
        state["control"]["focus_pending_candidate_generation"] = False
        self._record(state, "WMCC-020", {
            "focus_hypothesis_id": focus["hypothesis_id"],
            "focus_interpretation": focus["interpretation"],
            "preserved_alternative_count": state["control"]["alternatives_preserved"],
        }, _clip(base_score(focus)))

    def _op_integrate_semantic_evidence(
        self, state: dict[str, Any], request: WorkingMemoryRequest,
    ) -> None:
        """Combine lexical, semantic, syntactic, discourse, world, and top-down candidate evidence."""
        self._run_silently(state, self._op_apply_top_down_bias, request)
        scores = []
        for candidate in state["candidates"]:
            score = (
                0.20 * candidate["lexical_score"] + 0.25 * candidate["semantic_score"]
                + 0.20 * candidate["syntactic_score"] + 0.16 * candidate["discourse_score"]
                + 0.14 * candidate["world_score"] + 0.05 * candidate["prior"]
                + 0.05 * candidate.get("pragmatic_score", 0.5)
                + candidate.get("top_down_bias", 0.0)
            )
            candidate["integrated_score"] = score
            scores.append(score)
        state["candidate_scores"] = scores
        state["evidence_ledger"] = [
            {
                "hypothesis_id": item["hypothesis_id"],
                "channels": {
                    "lexical": item["lexical_score"], "semantic": item["semantic_score"],
                    "grammatical": item["syntactic_score"], "contextual": item["discourse_score"],
                    "knowledge": item["world_score"], "pragmatic": item.get("pragmatic_score", 0.5),
                    "prior": item["prior"], "top_down_bias": item.get("top_down_bias", 0.0),
                },
                "integrated_score": item["integrated_score"],
            }
            for item in state["candidates"]
        ]
        self._record(state, "WMCC-027", {
            "integrated_scores": [
                {"hypothesis_id": item["hypothesis_id"], "score": item["integrated_score"]}
                for item in state["candidates"]
            ],
            "evidence_ledger": state["evidence_ledger"],
        }, _clip(max(scores) if scores else 0.0))

    def _op_normalize_posteriors(self, state: dict[str, Any], request: WorkingMemoryRequest) -> None:
        """Normalize integrated candidate activations with a numerically stable softmax."""
        posterior = _softmax(state["candidate_scores"], self.config.semantic_temperature)
        state["posterior"] = posterior
        for candidate, probability in zip(state["candidates"], posterior):
            candidate["posterior"] = probability
        self._record(state, "WMCC-028", {
            "posterior": [
                {"hypothesis_id": item["hypothesis_id"], "probability": item["posterior"]}
                for item in state["candidates"]
            ],
            "posterior_sum": sum(posterior),
        }, max(posterior) if posterior else 0.0)

    def _op_run_recurrent_competition(
        self, state: dict[str, Any], request: WorkingMemoryRequest,
    ) -> None:
        """Settle candidate posteriors through damped recurrent competition without deleting hypotheses."""
        self._run_silently(state, self._op_normalize_posteriors, request)
        posterior = list(state["posterior"])
        scores = state["candidate_scores"]
        iterations = 0
        change = 0.0
        for iteration in range(self.config.inference_iterations):
            adjusted = [
                score / self.config.semantic_temperature
                - self.config.lateral_inhibition * (1.0 - probability)
                for score, probability in zip(scores, posterior)
            ]
            driven = _softmax(adjusted, 1.0)
            updated = [
                (1.0 - self.config.recurrent_damping) * old
                + self.config.recurrent_damping * new
                for old, new in zip(posterior, driven)
            ]
            total = sum(updated) or 1.0
            updated = [value / total for value in updated]
            change = sum(abs(new - old) for new, old in zip(updated, posterior))
            posterior = updated
            iterations = iteration + 1
            if change < 1e-7:
                break
        state["posterior"] = posterior
        for candidate, probability in zip(state["candidates"], posterior):
            candidate["posterior"] = probability
        state["control"]["competition_iterations"] = iterations
        state["control"]["competition_converged"] = change < 1e-5
        self._record(state, "WMCC-028", {
            "posterior": posterior, "iterations": iterations,
            "final_l1_change": change, "converged": change < 1e-5,
        }, max(posterior) if posterior else 0.0)

    def _op_measure_uncertainty(
        self, state: dict[str, Any], request: WorkingMemoryRequest,
    ) -> None:
        """Calculate normalized posterior entropy and the separation between the top two readings."""
        posterior = state["posterior"]
        ordered = sorted(posterior, reverse=True)
        entropy_raw = -sum(value * math.log(max(value, 1e-12)) for value in posterior)
        entropy = entropy_raw / math.log(len(posterior)) if len(posterior) > 1 else 0.0
        margin = ordered[0] - ordered[1] if len(ordered) > 1 else 1.0
        confidence = ordered[0] if ordered else 0.0
        state["resolution"].update({
            "normalized_entropy": _clip(entropy), "top_two_margin": _clip(margin),
            "confidence": _clip(confidence),
        })
        self._record(state, "WMCC-029", {
            "normalized_entropy": entropy, "top_two_margin": margin,
            "maximum_posterior": confidence,
        }, entropy)

    def _op_apply_abstention(self, state: dict[str, Any], request: WorkingMemoryRequest) -> None:
        """Require both posterior confidence and top-two separation before committing to one reading."""
        self._run_silently(state, self._op_measure_uncertainty, request)
        ranked = sorted(
            zip(state["candidates"], state["posterior"]),
            key=lambda pair: pair[1], reverse=True,
        )
        best_candidate, best_probability = ranked[0]
        margin = state["resolution"]["top_two_margin"]
        accepted = (
            best_probability >= self.config.confidence_threshold
            and margin >= self.config.ambiguity_margin
        )
        reasons = [] if accepted else [
            reason for condition, reason in (
                (best_probability < self.config.confidence_threshold, "posterior_below_threshold"),
                (margin < self.config.ambiguity_margin, "top_two_margin_below_threshold"),
            ) if condition
        ]
        state["resolution"].update({
            "selected_hypothesis_id": best_candidate["hypothesis_id"] if accepted else None,
            "selected": best_candidate["interpretation"] if accepted else None,
            "abstained": not accepted, "abstention_reasons": reasons,
        })
        self._run_silently(state, self._op_select_control_focus, request)
        self._run_silently(state, self._op_build_resolution_report, request)
        self._record(state, "WMCC-029", {
            "accepted": accepted, "best_posterior": best_probability, "margin": margin,
            "confidence_threshold": self.config.confidence_threshold,
            "margin_threshold": self.config.ambiguity_margin,
            "preserved_alternative_count": max(0, len(ranked) - (1 if accepted else 0)),
        }, best_probability if accepted else 1.0 - best_probability)

    def _op_build_resolution_report(
        self, state: dict[str, Any], request: WorkingMemoryRequest,
    ) -> None:
        """Return ranked interpretations, evidence provenance, convergence, and the abstention decision."""
        alternatives = []
        for candidate in sorted(
            state["candidates"], key=lambda item: item["posterior"], reverse=True
        ):
            alternatives.append({
                key: candidate[key] for key in (
                    "hypothesis_id", "interpretation", "prior", "lexical_score",
                    "semantic_score", "syntactic_score", "discourse_score", "world_score", "pragmatic_score",
                    "knowledge_score", "grammar_compatibility", "contextual_compatibility",
                    "goal_fit", "top_down_bias", "integrated_score", "posterior", "evidence_source",
                ) if key in candidate
            })
        state["resolution"].update({
            "alternatives": alternatives, "candidate_count": len(alternatives),
            "evidence_channels": [
                "lexical", "semantic", "grammatical", "contextual",
                "world_knowledge", "pragmatic", "prior", "goal_bias",
            ],
            "competition_iterations": state["control"].get("competition_iterations", 0),
            "competition_converged": state["control"].get("competition_converged", True),
        })
        self._record(state, "WMCC-039", {
            "selected": state["resolution"]["selected"],
            "abstained": state["resolution"]["abstained"],
            "alternative_count": len(alternatives),
        }, state["resolution"]["confidence"])

    def _op_encode_task_sets(self, state: dict[str, Any], request: WorkingMemoryRequest) -> None:
        """Read the explicit task sequence or initialize one current task from supplied context."""
        if request.task_sequence:
            tasks, source = list(request.task_sequence), "request.task_sequence"
        else:
            context_tasks = request.context.get("task_sequence", ())
            if isinstance(context_tasks, str):
                context_tasks = [context_tasks]
            tasks = list(context_tasks)
            if not tasks:
                tasks = [str(request.context.get("current_task", "comprehension"))]
            source = "context_or_default"
        state["task"].update({
            "tasks": tasks, "source": source,
            "previous_task": request.context.get("previous_task"),
            "current_task": tasks[-1] if tasks else None,
        })
        self._record(state, "WMCC-031", {
            "tasks": tasks, "source": source,
            "previous_task": state["task"]["previous_task"],
        }, min(1.0, len(tasks) / 10.0))

    def _op_detect_task_switches(
        self, state: dict[str, Any], request: WorkingMemoryRequest,
    ) -> None:
        """Mark task transitions relative to the supplied previous task and within the current sequence."""
        tasks = state["task"]["tasks"]
        prior = state["task"].get("previous_task")
        events = []
        for index, task in enumerate(tasks):
            if prior is not None and task != prior:
                events.append({
                    "sequence_index": index, "from_task": prior, "to_task": task,
                    "from_family": _task_family(prior), "to_family": _task_family(task),
                    "is_switch": True,
                })
            prior = task
        state["task"]["switch_events"] = events
        self._run_silently(state, self._op_calculate_switch_costs, request)
        self._record(state, "WMCC-032", {
            "switch_count": len(events), "switch_events": events,
            "estimated_total_switch_cost_ms": state["task"].get("estimated_total_switch_cost_ms", 0.0),
        }, min(1.0, len(events) / max(1, len(tasks))))

    def _op_calculate_switch_costs(
        self, state: dict[str, Any], request: WorkingMemoryRequest,
    ) -> None:
        """Estimate task-switch cost from task-family similarity, inertia, and explicit cue strength."""
        cue_strength = _clip(float(request.context.get("task_cue_strength", 0.65)))
        costs = []
        for event in state["task"]["switch_events"]:
            similarity = 1.0 if event["from_family"] == event["to_family"] else 0.15
            cost = self.config.task_switch_cost_ms * (
                1.0 + self.config.task_inertia * similarity
            ) * (1.0 - cue_strength)
            costs.append({
                "sequence_index": event["sequence_index"],
                "from_task": event["from_task"], "to_task": event["to_task"],
                "family_similarity": similarity, "cue_strength": cue_strength,
                "estimated_cost_ms": cost,
            })
        state["task"]["switch_costs"] = costs
        state["task"]["estimated_total_switch_cost_ms"] = sum(
            item["estimated_cost_ms"] for item in costs
        )
        self._record(state, "WMCC-032", {
            "switch_costs": costs,
            "estimated_total_switch_cost_ms": state["task"]["estimated_total_switch_cost_ms"],
        }, min(1.0, state["task"]["estimated_total_switch_cost_ms"] / 500.0))

    def _op_reconfigure_task_control(
        self, state: dict[str, Any], request: WorkingMemoryRequest,
    ) -> None:
        """Assign transparent attention weights to channels relevant to the current task family."""
        task = str(state["task"].get("current_task") or "comprehension").casefold()
        family = _task_family(task)
        profiles = {
            "language": {"phonological": 0.15, "lexical_semantic": 0.20, "syntactic": 0.25, "event_discourse": 0.25, "executive_goal": 0.15},
            "memory": {"phonological": 0.20, "lexical_semantic": 0.40, "syntactic": 0.10, "event_discourse": 0.20, "executive_goal": 0.10},
            "executive": {"phonological": 0.10, "lexical_semantic": 0.10, "syntactic": 0.15, "event_discourse": 0.15, "executive_goal": 0.50},
            "perception": {"phonological": 0.35, "lexical_semantic": 0.20, "syntactic": 0.10, "event_discourse": 0.20, "executive_goal": 0.15},
            "reasoning": {"phonological": 0.10, "lexical_semantic": 0.10, "syntactic": 0.20, "event_discourse": 0.40, "executive_goal": 0.20},
        }
        profile = profiles.get(family, profiles["language"])
        state["task"]["task_family"] = family
        state["task"]["control_weights"] = profile
        state["task"]["reconfigured_after_switch"] = bool(state["task"].get("switch_events"))
        self._record(state, "WMCC-034", {
            "current_task": task, "task_family": family, "control_weights": profile,
        }, max(profile.values()))

    def _op_manage_task_interference(
        self, state: dict[str, Any], request: WorkingMemoryRequest,
    ) -> None:
        """Estimate task interference and preserve suspended task-state references for later resumption."""
        events = state["task"]["switch_events"]
        interference = 0.0
        suspended = state["task"].get("suspended_tasks", [])
        for event in events:
            similarity = 1.0 if event["from_family"] == event["to_family"] else 0.15
            interference += similarity * self.config.task_inertia * state["control"]["control_demand"]
        state["task"]["interference"] = _clip(interference)
        state["task"]["suppressed_previous_task"] = [
            {"task": item["task"], "priority_reduced": True, "checkpoint_retained": True}
            for item in suspended
        ]
        self._record(state, "WMCC-035", {
            "interference": state["task"]["interference"],
            "suspended_tasks": suspended, "resumed_tasks": state["task"].get("resumed_tasks", []),
        }, state["task"]["interference"])

    def _op_emit_task_state(self, state: dict[str, Any], request: WorkingMemoryRequest) -> None:
        """Emit the complete task-set history, switch costs, active profile, and interference summary."""
        state["task"].update({
            "switch_count": len(state["task"].get("switch_events", [])),
            "mean_switch_cost_ms": (
                state["task"].get("estimated_total_switch_cost_ms", 0.0)
                / max(1, len(state["task"].get("switch_costs", [])))
            ),
            "active_control_state": {
                "task": state["task"].get("current_task"),
                "family": state["task"].get("task_family"),
                "weights": state["task"].get("control_weights", {}),
            },
        })
        self._record(state, "WMCC-039", {
            "tasks": state["task"].get("tasks", []),
            "switch_count": state["task"]["switch_count"],
            "estimated_total_switch_cost_ms": state["task"].get("estimated_total_switch_cost_ms", 0.0),
            "interference": state["task"].get("interference", 0.0),
            "active_control_state": state["task"]["active_control_state"],
        }, state["task"].get("interference", 0.0))
