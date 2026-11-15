"""Thirty-operation working-memory and cognitive-control processing pipeline.

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
    """Coordinate five working-memory and cognitive-control stages with inspectable operation traces.

    The processor is deterministic. It does not use a trained parser or world-knowledge model;
    upstream evidence is accepted explicitly and weak evidence is retained as unresolved.
    """

    def __init__(self, config: WorkingMemoryConfig | None = None) -> None:
        """Store validated configuration for buffer limits, inference, task costs, and biology."""
        self.config = config or WorkingMemoryConfig()
        self._simulator: WorkingMemoryLIFSimulator | None = None

    def process(self, request: WorkingMemoryRequest) -> WorkingMemoryResult:
        """Run all 30 operations in order and return functional, biological, and diagnostic tracks."""
        if not isinstance(request, WorkingMemoryRequest):
            raise TypeError("request must be a WorkingMemoryRequest")
        self._simulator = (
            WorkingMemoryLIFSimulator(self.config) if self.config.simulate_biology else None
        )
        original_text = request.text if request.text is not None else " ".join(request.tokens or ())
        state: dict[str, Any] = {
            "tokens": [], "buffers": {}, "evicted_items": {}, "retrieval_candidates": [],
            "structures": [], "syntactic_stack": [], "pending_dependencies": [],
            "slot_candidates": [], "resolved_slots": [], "control": {}, "candidates": [],
            "posterior": [], "resolution": {}, "task": {}, "trace": [],
        }
        operations = (
            self._op_tokenize_and_encode, self._op_allocate_linguistic_buffers,
            self._op_calculate_buffer_activation, self._op_decay_and_capacity,
            self._op_rehearse_active_items, self._op_retrieve_context_cues,
            self._op_extract_incomplete_structures, self._op_build_syntactic_stack,
            self._op_track_unresolved_dependencies, self._op_score_slot_retrieval_cues,
            self._op_bind_supported_slots, self._op_emit_incomplete_state,
            self._op_prepare_interpretations, self._op_measure_conflict,
            self._op_allocate_control, self._op_evaluate_goal_fit,
            self._op_apply_top_down_bias, self._op_select_control_focus,
            self._op_integrate_semantic_evidence, self._op_normalize_posteriors,
            self._op_run_recurrent_competition, self._op_measure_uncertainty,
            self._op_apply_abstention, self._op_build_resolution_report,
            self._op_encode_task_sets, self._op_detect_task_switches,
            self._op_calculate_switch_costs, self._op_reconfigure_task_control,
            self._op_manage_task_interference, self._op_emit_task_state,
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
            len(state["trace"]) == len(OPERATION_REGISTRY)
            and all(size <= self.config.buffer_capacity for size in buffer_sizes.values())
            and (not probabilities or math.isclose(probability_sum, 1.0, rel_tol=1e-7, abs_tol=1e-7))
            and all(math.isfinite(value) for value in probabilities)
        )
        warnings = []
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
            "token_count": len(state["tokens"]), "buffer_sizes": buffer_sizes,
            "incomplete_structure_count": len(state["structures"]),
            "unresolved_slot_count": unresolved_slots,
            "interpretation_count": len(state["candidates"]),
            "normalized_posterior_sum": probability_sum,
            "semantic_entropy": state["resolution"].get("normalized_entropy", 0.0),
            "task_switch_count": len(state["task"].get("switch_events", [])),
            "warnings": warnings,
        }
        metadata = {
            "language": request.language, "source_reference": request.source_reference,
            "configuration": self.config.to_dict(),
            "biological_scope": biological.get("scope", ""),
            "inference_scope": (
                "Rule-based and evidence-driven software prototype. It is not a validated theory "
                "of human working memory or a general-purpose language parser."
            ),
        }
        return WorkingMemoryResult(
            sequence_id=request.sequence_id, original_text=original_text,
            tokens=list(state["tokens"]), buffers=state["buffers"],
            incomplete_structures=state.get("incomplete_report", {}),
            executive_control=state["control"], semantic_resolution=state["resolution"],
            task_switching=state["task"], biological=biological,
            operation_trace=state["trace"], diagnostics=diagnostics, metadata=metadata,
        )

    def _record(self, state: dict[str, Any], operation_id: str,
                output: Mapping[str, Any], signal: float = 0.0) -> None:
        """Append registry metadata, functional output, and the matching biological update."""
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
        self._record(state, "WMC-001", {
            "token_count": len(tokens), "token_records": state["token_records"],
        }, min(1.0, len(tokens) / max(1, self.config.max_tokens)))

    def _op_allocate_linguistic_buffers(
        self, state: dict[str, Any], request: WorkingMemoryRequest,
    ) -> None:
        """Create separate modality-labelled token-reference buffers without merging their functions."""
        buffers: dict[str, list[dict[str, Any]]] = {
            "phonological": [], "lexical": [], "syntactic": [], "semantic": [], "discourse": [],
        }
        for record in state["token_records"]:
            token = record["surface"]
            normalized = record["normalized"]
            base = {"position": record["index"], "token": token, "normalized": normalized,
                    "activation": 0.0, "rehearsal_count": 0}
            if record["is_word"]:
                buffers["phonological"].append(dict(base))
                buffers["lexical"].append(dict(base))
                if normalized in _SYNTACTIC_MARKERS:
                    buffers["syntactic"].append(dict(base))
                if normalized not in _STOPWORDS:
                    buffers["semantic"].append(dict(base))
                    buffers["discourse"].append(dict(base))
            else:
                buffers["phonological"].append(dict(base))
                buffers["syntactic"].append(dict(base))
        state["buffers"] = buffers
        self._record(state, "WMC-002", {
            "modalities": list(buffers),
            "item_counts": {name: len(items) for name, items in buffers.items()},
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
        values = [
            item["activation"] for items in state["buffers"].values() for item in items
        ]
        average = sum(values) / max(1, len(values))
        state["mean_buffer_activation"] = average
        self._record(state, "WMC-003", {
            "mean_activation": average, "decay_per_token": self.config.activation_decay_per_token,
            "focus_cue_count": len(focus_terms),
        }, average)

    def _op_decay_and_capacity(self, state: dict[str, Any], request: WorkingMemoryRequest) -> None:
        """Apply an activation-ranked capacity limit and retain the evicted items for inspection."""
        evicted: dict[str, list[dict[str, Any]]] = {}
        for name, items in state["buffers"].items():
            ranked = sorted(items, key=lambda item: (item["activation"], item["position"]), reverse=True)
            retained = ranked[:self.config.buffer_capacity]
            kept_positions = {item["position"] for item in retained}
            evicted[name] = [dict(item) for item in items if item["position"] not in kept_positions]
            state["buffers"][name] = sorted(retained, key=lambda item: item["position"])
        state["evicted_items"] = evicted
        total_evicted = sum(len(items) for items in evicted.values())
        self._record(state, "WMC-004", {
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
        self._record(state, "WMC-005", {
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
        self._record(state, "WMC-006", {
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
        self._record(state, "WMC-007", {
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
        self._record(state, "WMC-008", {
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
        self._record(state, "WMC-009", {
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
        self._record(state, "WMC-010", {
            "scored_candidate_count": len(scored), "top_candidates": scored[:10],
        }, scored[0]["retrieval_score"] if scored else 0.0)

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
        self._record(state, "WMC-011", {
            "resolved_slot_count": len(bound), "resolved_slots": bound,
            "remaining_open_slots": sum(len(frame["unresolved_slots"]) for frame in state["structures"]),
        }, min(1.0, len(bound) / max(1, len(state["slot_candidates"]))))

    def _op_emit_incomplete_state(self, state: dict[str, Any], request: WorkingMemoryRequest) -> None:
        """Serialize open frames and dependencies without inserting unsupported words or constituents."""
        state["incomplete_report"] = {
            "stack": state["syntactic_stack"], "frames": state["structures"],
            "pending_dependencies": state["pending_dependencies"],
            "resolved_slots": state["resolved_slots"],
            "unresolved_slot_count": sum(
                len(frame["unresolved_slots"]) for frame in state["structures"]
            ),
            "complete": not any(frame["unresolved_slots"] for frame in state["structures"])
            and not state["pending_dependencies"],
        }
        self._record(state, "WMC-012", {
            "complete": state["incomplete_report"]["complete"],
            "unresolved_slot_count": state["incomplete_report"]["unresolved_slot_count"],
            "pending_dependency_count": len(state["pending_dependencies"]),
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
                "world_score": _clip(float(item.get("world_score", 0.5))),
                "goal_compatibility": item.get("goal_compatibility"),
                "features": dict(item.get("features", {})),
                "evidence_source": item.get("evidence_source", "caller_supplied"),
            })
        state["candidates"] = candidates
        self._record(state, "WMC-013", {
            "candidate_count": len(candidates),
            "candidate_ids": [item["hypothesis_id"] for item in candidates],
        }, min(1.0, len(candidates) / max(1, self.config.max_interpretations)))

    def _op_measure_conflict(self, state: dict[str, Any], request: WorkingMemoryRequest) -> None:
        """Estimate interpretation conflict from preliminary score entropy and the top-two margin."""
        candidates = state["candidates"]
        preliminary = [
            0.20 * item["lexical_score"] + 0.28 * item["semantic_score"]
            + 0.20 * item["syntactic_score"] + 0.16 * item["discourse_score"]
            + 0.16 * item["world_score"] + 0.05 * item["prior"]
            for item in candidates
        ]
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
        self._record(state, "WMC-014", {
            "conflict": conflict, "normalized_entropy": entropy, "top_two_margin": margin,
        }, conflict)

    def _op_allocate_control(self, state: dict[str, Any], request: WorkingMemoryRequest) -> None:
        """Allocate a bounded executive-control demand from conflict, open syntax, and goal priority."""
        open_count = sum(len(frame["unresolved_slots"]) for frame in state["structures"])
        open_load = _clip(open_count / max(1, self.config.max_syntactic_frames))
        goal_priority = _clip(float(request.goals.get("priority", request.context.get("goal_priority", 0.5))))
        conflict = state["control"]["conflict"]
        demand = _clip(0.45 * conflict + 0.35 * open_load + 0.20 * goal_priority)
        state["control"].update({
            "syntactic_open_load": open_load, "goal_priority": goal_priority,
            "control_demand": demand,
            "control_resources": {
                "monitoring": _clip(0.5 + 0.5 * conflict),
                "maintenance": _clip(0.5 + 0.5 * open_load),
                "goal_bias": _clip(goal_priority),
            },
        })
        self._record(state, "WMC-015", {
            "control_demand": demand, "open_syntax_load": open_load, "goal_priority": goal_priority,
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
        self._record(state, "WMC-016", {
            "goal_fits": fits,
        }, sum(item["goal_fit"] for item in fits) / max(1, len(fits)))

    def _op_apply_top_down_bias(
        self, state: dict[str, Any], request: WorkingMemoryRequest,
    ) -> None:
        """Add a bounded goal/prior bias to each candidate while preserving its source evidence scores."""
        demand = state["control"]["control_demand"]
        for candidate in state["candidates"]:
            goal_fit = candidate.get("goal_fit", 0.5)
            candidate["top_down_bias"] = self.config.control_gain * demand * (
                0.65 * goal_fit + 0.35 * candidate["prior"]
            )
        biases = [item["top_down_bias"] for item in state["candidates"]]
        mean_bias = sum(biases) / max(1, len(biases))
        state["control"]["mean_top_down_bias"] = mean_bias
        self._record(state, "WMC-017", {
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
        focus = max(state["candidates"], key=base_score)
        state["control"]["focus_hypothesis_id"] = focus["hypothesis_id"]
        state["control"]["focus_interpretation"] = focus["interpretation"]
        state["control"]["alternatives_preserved"] = max(0, len(state["candidates"]) - 1)
        self._record(state, "WMC-018", {
            "focus_hypothesis_id": focus["hypothesis_id"],
            "focus_interpretation": focus["interpretation"],
            "preserved_alternative_count": state["control"]["alternatives_preserved"],
        }, _clip(base_score(focus)))

    def _op_integrate_semantic_evidence(
        self, state: dict[str, Any], request: WorkingMemoryRequest,
    ) -> None:
        """Combine lexical, semantic, syntactic, discourse, world, and top-down candidate evidence."""
        scores = []
        for candidate in state["candidates"]:
            score = (
                0.20 * candidate["lexical_score"] + 0.28 * candidate["semantic_score"]
                + 0.20 * candidate["syntactic_score"] + 0.16 * candidate["discourse_score"]
                + 0.16 * candidate["world_score"] + 0.05 * candidate["prior"]
                + candidate.get("top_down_bias", 0.0)
            )
            candidate["integrated_score"] = score
            scores.append(score)
        state["candidate_scores"] = scores
        self._record(state, "WMC-019", {
            "integrated_scores": [
                {"hypothesis_id": item["hypothesis_id"], "score": item["integrated_score"]}
                for item in state["candidates"]
            ],
        }, _clip(max(scores) if scores else 0.0))

    def _op_normalize_posteriors(self, state: dict[str, Any], request: WorkingMemoryRequest) -> None:
        """Normalize integrated candidate activations with a numerically stable softmax."""
        posterior = _softmax(state["candidate_scores"], self.config.semantic_temperature)
        state["posterior"] = posterior
        for candidate, probability in zip(state["candidates"], posterior):
            candidate["posterior"] = probability
        self._record(state, "WMC-020", {
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
        self._record(state, "WMC-021", {
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
        self._record(state, "WMC-022", {
            "normalized_entropy": entropy, "top_two_margin": margin,
            "maximum_posterior": confidence,
        }, entropy)

    def _op_apply_abstention(self, state: dict[str, Any], request: WorkingMemoryRequest) -> None:
        """Require both posterior confidence and top-two separation before committing to one reading."""
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
        self._record(state, "WMC-023", {
            "accepted": accepted, "best_posterior": best_probability, "margin": margin,
            "confidence_threshold": self.config.confidence_threshold,
            "margin_threshold": self.config.ambiguity_margin,
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
                    "semantic_score", "syntactic_score", "discourse_score", "world_score",
                    "goal_fit", "top_down_bias", "integrated_score", "posterior", "evidence_source",
                ) if key in candidate
            })
        state["resolution"].update({
            "alternatives": alternatives, "candidate_count": len(alternatives),
            "evidence_channels": [
                "lexical", "semantic", "syntactic", "discourse", "world", "goal_bias",
            ],
            "competition_iterations": state["control"].get("competition_iterations", 0),
            "competition_converged": state["control"].get("competition_converged", True),
        })
        self._record(state, "WMC-024", {
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
        self._record(state, "WMC-025", {
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
        self._record(state, "WMC-026", {
            "switch_count": len(events), "switch_events": events,
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
        self._record(state, "WMC-027", {
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
            "language": {"phonological": 0.15, "lexical": 0.20, "syntactic": 0.25, "semantic": 0.25, "executive": 0.15},
            "memory": {"phonological": 0.20, "lexical": 0.40, "syntactic": 0.10, "semantic": 0.20, "executive": 0.10},
            "executive": {"phonological": 0.10, "lexical": 0.10, "syntactic": 0.15, "semantic": 0.15, "executive": 0.50},
            "perception": {"phonological": 0.35, "lexical": 0.20, "syntactic": 0.10, "semantic": 0.20, "executive": 0.15},
            "reasoning": {"phonological": 0.10, "lexical": 0.10, "syntactic": 0.20, "semantic": 0.40, "executive": 0.20},
        }
        profile = profiles.get(family, profiles["language"])
        state["task"]["task_family"] = family
        state["task"]["control_weights"] = profile
        state["task"]["reconfigured_after_switch"] = bool(state["task"].get("switch_events"))
        self._record(state, "WMC-028", {
            "current_task": task, "task_family": family, "control_weights": profile,
        }, max(profile.values()))

    def _op_manage_task_interference(
        self, state: dict[str, Any], request: WorkingMemoryRequest,
    ) -> None:
        """Estimate task interference and preserve suspended task-state references for later resumption."""
        events = state["task"]["switch_events"]
        interference = 0.0
        suspended = []
        seen_tasks: set[str] = set()
        for event in events:
            similarity = 1.0 if event["from_family"] == event["to_family"] else 0.15
            interference += similarity * self.config.task_inertia * state["control"]["control_demand"]
            if event["from_task"] not in seen_tasks:
                suspended.append({
                    "task": event["from_task"],
                    "suspended_at_sequence_index": event["sequence_index"],
                    "state_preserved": True,
                })
                seen_tasks.add(event["from_task"])
        resumed = []
        tasks = state["task"]["tasks"]
        for item in suspended:
            if item["task"] in tasks[item["suspended_at_sequence_index"] + 1:]:
                resumed.append(item["task"])
        state["task"]["interference"] = _clip(interference)
        state["task"]["suspended_tasks"] = suspended
        state["task"]["resumed_tasks"] = sorted(set(resumed))
        self._record(state, "WMC-029", {
            "interference": state["task"]["interference"],
            "suspended_tasks": suspended, "resumed_tasks": state["task"]["resumed_tasks"],
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
        self._record(state, "WMC-030", {
            "tasks": state["task"].get("tasks", []),
            "switch_count": state["task"]["switch_count"],
            "estimated_total_switch_cost_ms": state["task"].get("estimated_total_switch_cost_ms", 0.0),
            "interference": state["task"].get("interference", 0.0),
            "active_control_state": state["task"]["active_control_state"],
        }, state["task"].get("interference", 0.0))
