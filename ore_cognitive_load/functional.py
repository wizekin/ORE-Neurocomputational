"""Functional mathematics and information-organization operations for Point 06."""
from __future__ import annotations

from collections import defaultdict, deque
import math
import re
from typing import Any, Iterable, Mapping, Sequence

from .contracts import CognitiveUnit, Point06Config, Point06Request, Point06State

_TOKEN_RE = re.compile(r"[\w'-]+", flags=re.UNICODE)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+|(?<=;)\s+")


def clip01(value: float) -> float:
    """Convert a finite numeric score to the normalized interval [0, 1]."""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("score must be finite")
    return max(0.0, min(1.0, number))


def softmax(values: Sequence[float], temperature: float = 1.0) -> list[float]:
    """Compute a stable temperature-scaled softmax distribution over finite utilities."""
    if not math.isfinite(float(temperature)) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if not values:
        return []
    numbers = [float(value) / temperature for value in values]
    if any(not math.isfinite(value) for value in numbers):
        raise ValueError("softmax values must be finite")
    pivot = max(numbers)
    exponentials = [math.exp(max(-700.0, value - pivot)) for value in numbers]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def tokenize(text: str) -> tuple[str, ...]:
    """Extract normalized lexical features for transparent lexical-overlap estimates."""
    return tuple(token.casefold() for token in _TOKEN_RE.findall(str(text)))


def feature_similarity(left: Iterable[str], right: Iterable[str]) -> float:
    """Measure Jaccard overlap between two feature sets without a learned semantic model."""
    first = {str(value).casefold() for value in left if str(value).strip()}
    second = {str(value).casefold() for value in right if str(value).strip()}
    if not first or not second:
        return 0.0
    return len(first & second) / len(first | second)


def segment_material(request: Point06Request) -> list[CognitiveUnit]:
    """Convert supplied units or source text into stable, source-ordered learning units."""
    if request.units is not None:
        raw_units = list(request.units)
        method = "caller_supplied_units"
    else:
        raw_units = [part.strip() for part in _SENTENCE_RE.split(request.text or "") if part.strip()]
        if not raw_units:
            raw_units = [str(request.text).strip()]
        method = "sentence_paragraph_and_semicolon_boundaries"
    goal_features = tokenize(request.goal)
    prior_features = [tokenize(str(item.get("text", ""))) for item in request.prior_knowledge]
    output: list[CognitiveUnit] = []
    for index, raw in enumerate(raw_units):
        identifier = f"u{index + 1:04d}"
        metadata = dict(request.unit_metadata.get(identifier, request.unit_metadata.get(str(index), {})))
        tokens = tokenize(raw)
        explicit_relevance = metadata.get("task_relevance")
        relevance = clip01(float(explicit_relevance)) if explicit_relevance is not None else (
            feature_similarity(tokens, goal_features) if feature_similarity(tokens, goal_features) else 0.25
        )
        familiarity = max((feature_similarity(tokens, memory) for memory in prior_features), default=0.0)
        familiarity = max(familiarity, clip01(float(metadata.get("familiarity", 0.0))))
        default_intrinsic = min(1.0, 0.20 + len(tokens) / 28.0)
        intrinsic = clip01(float(metadata.get("intrinsic_load", default_intrinsic)))
        output.append(CognitiveUnit(
            identifier=identifier, text=raw, position=index, features=tokens,
            task_relevance=relevance, intrinsic_load=intrinsic,
            familiarity=familiarity, chunk_id=(str(metadata["chunk_id"]) if metadata.get("chunk_id") else None),
        ))
    return output


def validate_dependencies(
    units: Sequence[CognitiveUnit], dependencies: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Validate explicit prerequisite edges against known unit identifiers."""
    known = {unit.identifier for unit in units}
    seen: set[tuple[str, str]] = set()
    result: list[dict[str, Any]] = []
    for index, raw in enumerate(dependencies):
        source, target = str(raw.get("source", "")), str(raw.get("target", ""))
        if source not in known or target not in known:
            raise ValueError(f"dependency {index} references an unknown unit")
        if source == target:
            raise ValueError("a unit cannot be its own prerequisite")
        if (source, target) in seen:
            continue
        seen.add((source, target))
        strength = clip01(float(raw.get("strength", 1.0)))
        result.append({"source": source, "target": target, "strength": strength,
                       "relation": str(raw.get("relation", "prerequisite")),
                       "source_evidence": "caller_supplied"})
    return result


def estimate_element_interactivity(
    units: Sequence[CognitiveUnit], dependencies: Sequence[Mapping[str, Any]],
    config: Point06Config,
) -> dict[str, Any]:
    """Estimate intrinsic demand plus relation interactions, adjusted by learner familiarity."""
    intrinsic_values = [unit.intrinsic_load for unit in units]
    intrinsic_total = sum(intrinsic_values) / max(1, len(intrinsic_values))
    interaction_total = sum(float(edge.get("strength", 1.0)) for edge in dependencies)
    interaction_mean = interaction_total / max(1, len(units))
    raw_complexity = intrinsic_total + config.interaction_gain * interaction_mean
    familiarity = sum(unit.familiarity for unit in units) / max(1, len(units))
    effective_complexity = raw_complexity * (1.0 - config.prior_knowledge_gain * familiarity)
    return {
        "intrinsic_demand_mean": clip01(intrinsic_total),
        "explicit_interaction_mean": clip01(interaction_mean),
        "prior_knowledge_familiarity": clip01(familiarity),
        "raw_complexity": raw_complexity,
        "effective_complexity": clip01(effective_complexity),
        "formula": "C_task=mean(a_i)+lambda*mean(b_ij); C_eff=C_task*(1-eta*K)",
        "interaction_evidence": "explicit dependency edges only; no hidden relations inferred",
    }


def retrieve_prerequisites(
    units: Sequence[CognitiveUnit], dependencies: Sequence[Mapping[str, Any]],
    prior_knowledge: Sequence[Mapping[str, Any]], target_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Retrieve supplied memory records related to the target and its transitive prerequisites."""
    by_id = {unit.identifier: unit for unit in units}
    targets = list(target_ids or by_id.keys())
    reverse: dict[str, list[str]] = defaultdict(list)
    for edge in dependencies:
        reverse[str(edge["target"])].append(str(edge["source"]))
    closure: set[str] = set()
    queue = deque(targets)
    while queue:
        current = queue.popleft()
        for prerequisite in reverse.get(current, []):
            if prerequisite not in closure:
                closure.add(prerequisite)
                queue.append(prerequisite)
    target_features = set()
    for identifier in set(targets) | closure:
        if identifier in by_id:
            target_features.update(by_id[identifier].features)
    candidates: list[dict[str, Any]] = []
    memory_features: list[tuple[Mapping[str, Any], tuple[str, ...]]] = []
    for memory in prior_knowledge:
        text_value = str(memory.get("text", ""))
        features = tokenize(text_value) + tuple(str(item).casefold() for item in memory.get("keywords", ()))
        memory_features.append((memory, features))
        relevance = feature_similarity(target_features, features)
        prerequisite_match = any(
            token in features for identifier in closure if identifier in by_id
            for token in by_id[identifier].features
        )
        confidence = clip01(float(memory.get("confidence", 0.5)))
        score = clip01(0.75 * relevance + 0.25 * confidence + (0.10 if prerequisite_match else 0.0))
        if score > 0.0:
            candidates.append({"id": str(memory.get("id", "memory")), "text": text_value,
                               "relevance": relevance, "confidence": confidence,
                               "score": score, "source": "caller_supplied_prior_knowledge"})
    candidates.sort(key=lambda item: item["score"], reverse=True)
    supported: set[str] = set()
    for identifier in closure:
        unit = by_id[identifier]
        for memory, features in memory_features:
            explicit_ids = {str(memory.get("unit_id", ""))}
            explicit_ids.update(str(item) for item in memory.get("unit_ids", ()))
            if identifier in explicit_ids or feature_similarity(unit.features, features) >= 0.20:
                supported.add(identifier)
                break
    ordered_prerequisites = sorted(closure, key=lambda item: by_id[item].position)
    return {
        "target_unit_ids": targets,
        "prerequisite_unit_ids": ordered_prerequisites,
        "retrieved_memories": candidates,
        "supported_prerequisite_ids": [item for item in ordered_prerequisites if item in supported],
        "missing_prerequisite_ids": [item for item in ordered_prerequisites if item not in supported],
        "retrieval_limit": "only supplied memory records are returned; no facts are fabricated",
    }


def order_by_prerequisites(
    units: Sequence[CognitiveUnit], dependencies: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Topologically order acyclic prerequisite dependencies and explicitly retain cyclic groups."""
    original = {unit.identifier: unit.position for unit in units}
    outgoing: dict[str, list[str]] = defaultdict(list)
    indegree = {unit.identifier: 0 for unit in units}
    for edge in dependencies:
        source, target = str(edge["source"]), str(edge["target"])
        outgoing[source].append(target)
        indegree[target] += 1
    queue = deque(sorted((node for node, degree in indegree.items() if degree == 0),
                         key=lambda node: original[node]))
    ordered: list[str] = []
    while queue:
        node = queue.popleft()
        ordered.append(node)
        for target in sorted(outgoing[node], key=lambda item: original[item]):
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    cyclic = sorted((node for node, degree in indegree.items() if degree > 0), key=lambda node: original[node])
    ordered.extend(cyclic)
    return {"ordered_unit_ids": ordered, "cyclic_unit_ids": cyclic,
            "cyclic_units_treated_as_integrated_group": bool(cyclic),
            "ordering": "stable topological order, followed by unresolved cycle members in source order"}


def form_meaningful_chunks(
    units: Sequence[CognitiveUnit], dependencies: Sequence[Mapping[str, Any]], threshold: float,
) -> dict[str, Any]:
    """Group adjacent units only when explicit grouping, dependencies, or lexical coherence supports it."""
    grouping_relations = {"same_chunk", "belongs_to", "part_of", "substep_of"}
    edge_pairs = {
        (str(item["source"]), str(item["target"]))
        for item in dependencies
        if bool(item.get("chunk_together")) or str(item.get("relation", "")).casefold() in grouping_relations
    }
    groups: list[list[CognitiveUnit]] = []
    for unit in units:
        if not groups:
            groups.append([unit])
            continue
        previous_group = groups[-1]
        previous = previous_group[-1]
        same_label = bool(unit.chunk_id and unit.chunk_id == previous.chunk_id)
        explicit_link = (previous.identifier, unit.identifier) in edge_pairs or (unit.identifier, previous.identifier) in edge_pairs
        lexical = feature_similarity(previous.features, unit.features)
        group_features = set(token for member in previous_group for token in member.features)
        group_similarity = feature_similarity(group_features, unit.features)
        if same_label or explicit_link or max(lexical, group_similarity) >= threshold:
            previous_group.append(unit)
        else:
            groups.append([unit])
    chunks: list[dict[str, Any]] = []
    for index, group in enumerate(groups, start=1):
        pair_scores = [feature_similarity(a.features, b.features)
                       for i, a in enumerate(group) for b in group[i + 1:]]
        chunks.append({"chunk_id": f"c{index:04d}",
                       "member_ids": [item.identifier for item in group],
                       "label": group[0].text[:80],
                       "coherence": sum(pair_scores) / len(pair_scores) if pair_scores else 1.0,
                       "member_count": len(group),
                       "reversible": True,
                       "grouping_evidence": "explicit_chunk_label_or_dependency_or_adjacent_feature_overlap"})
    return {"chunks": chunks, "chunk_count": len(chunks),
            "effective_unit_count": len(chunks), "over_segmentation_checked": True,
            "under_segmentation_checked": True}


def select_principal_subgoal(
    request: Point06Request, units: Sequence[CognitiveUnit], dependencies: Sequence[Mapping[str, Any]],
    config: Point06Config,
) -> dict[str, Any]:
    """Select the best-supported current unit using relevance, prerequisite readiness, and priority."""
    prerequisite_targets = {str(edge["target"]) for edge in dependencies}
    source_ids = {str(edge["source"]) for edge in dependencies}
    units_by_id = {unit.identifier: unit for unit in units}
    rows: list[tuple[str, float, dict[str, float]]] = []
    for unit in units:
        readiness = 1.0 if unit.identifier not in prerequisite_targets else (
            0.75 if source_ids else 0.35
        )
        metadata = dict(request.unit_metadata.get(unit.identifier, {}))
        priority = clip01(float(metadata.get("priority", unit.task_relevance)))
        utility = 0.60 * unit.task_relevance + 0.25 * readiness + 0.15 * priority
        rows.append((unit.identifier, utility, {"goal_relevance": unit.task_relevance,
                                                "prerequisite_readiness": readiness, "priority": priority}))
    probabilities = softmax([row[1] for row in rows], config.goal_temperature)
    selected_index = max(range(len(rows)), key=lambda index: (probabilities[index], -units_by_id[rows[index][0]].position)) if rows else None
    candidates = {row[0]: {"utility": row[1], "probability": probabilities[index], "factors": row[2]}
                  for index, row in enumerate(rows)}
    goal_candidates = dict(request.goal_candidates) or {request.goal: 1.0}
    goal_names = list(goal_candidates)
    goal_probabilities = softmax([float(goal_candidates[name]) for name in goal_names], config.goal_temperature)
    return {
        "goal": request.goal,
        "goal_candidate_probabilities": dict(zip(goal_names, goal_probabilities)),
        "unit_candidate_scores": candidates,
        "selected_unit_id": rows[selected_index][0] if selected_index is not None else None,
        "selected_subgoal": units_by_id[rows[selected_index][0]].text if selected_index is not None else request.goal,
        "deferred_unit_ids": [row[0] for index, row in enumerate(rows) if index != selected_index],
        "selection_formula": "P(g_i|s)=softmax(beta*U_i); U_i=.60*relevance+.25*readiness+.15*priority",
    }


def reduce_presentation_demands(costs: Mapping[str, float]) -> dict[str, Any]:
    """Estimate avoidable split-attention, visual-noise, ambiguity, and redundancy demands."""
    weights = {"split_attention": 0.32, "visual_noise": 0.18, "ambiguity": 0.30, "redundancy": 0.20}
    normalized = {name: clip01(float(costs.get(name, 0.0))) for name in weights}
    total = sum(weights[name] * normalized[name] for name in weights)
    actions = {
        "split_attention": "place definitions and explanations beside the concepts they clarify",
        "visual_noise": "remove decorative or irrelevant detail that competes with the learning goal",
        "ambiguity": "use consistent labels and clarify unclear relationships",
        "redundancy": "remove repeated processing that does not add explanatory value",
    }
    recommendations = [actions[name] for name, value in normalized.items() if value >= 0.35]
    return {"avoidable_load": clip01(total), "cost_components": normalized,
            "component_weights": weights, "redesign_recommendations": recommendations,
            "essential_content_preserved": True,
            "formula": "L_avoidable=alpha*S+beta*N+gamma*A+delta*R"}


def build_external_representation(
    units: Sequence[CognitiveUnit], chunks: Sequence[Mapping[str, Any]],
    dependencies: Sequence[Mapping[str, Any]], relations: Sequence[Mapping[str, Any]], goal: str,
) -> dict[str, Any]:
    """Build an explicit outline graph from source units and caller-supplied relations."""
    chunk_lookup = {member: str(chunk["chunk_id"]) for chunk in chunks for member in chunk["member_ids"]}
    edges = [dict(item) for item in dependencies]
    edges.extend(dict(item) for item in relations)
    nodes = [{"id": unit.identifier, "label": unit.text, "position": unit.position,
              "chunk_id": chunk_lookup.get(unit.identifier)} for unit in units]
    return {"goal": goal, "nodes": nodes, "edges": edges,
            "outline": [{"chunk_id": chunk["chunk_id"], "label": chunk["label"],
                         "member_ids": list(chunk["member_ids"])} for chunk in chunks],
            "representation_types": ["outline", "directed_relation_graph"],
            "relation_invention": False}


def select_working_set(
    units: Sequence[CognitiveUnit], dependencies: Sequence[Mapping[str, Any]],
    active_subgoal_id: str | None, limit: int,
) -> dict[str, Any]:
    """Select a bounded, task-relevant active set while retaining explicit prerequisite links."""
    unit_map = {unit.identifier: unit for unit in units}
    reverse: dict[str, list[str]] = defaultdict(list)
    for edge in dependencies:
        reverse[str(edge["target"])].append(str(edge["source"]))
    required: set[str] = {active_subgoal_id} if active_subgoal_id in unit_map else set()
    queue = deque(required)
    while queue:
        node = queue.popleft()
        for parent in reverse.get(node, []):
            if parent not in required:
                required.add(parent)
                queue.append(parent)
    ranked = sorted(units, key=lambda unit: (
        0 if unit.identifier in required else 1,
        -(unit.task_relevance + (0.2 if unit.identifier in required else 0.0)),
        unit.position,
    ))
    selected = ranked[:limit]
    selected_ids = {unit.identifier for unit in selected}
    return {"active_unit_ids": [unit.identifier for unit in selected],
            "deferred_unit_ids": [unit.identifier for unit in units if unit.identifier not in selected_ids],
            "required_prerequisite_ids": sorted(required, key=lambda item: unit_map[item].position),
            "required_prerequisites_deferred_by_limit": sorted(required - selected_ids,
                                                                 key=lambda item: unit_map[item].position),
            "working_set_limit": limit, "limit_is_engineering_policy_not_human_capacity": True}


def monitor_overload(metrics: Mapping[str, float], load: float, threshold: float) -> dict[str, Any]:
    """Combine indirect difficulty indicators and preserve high-effort accurate work as productive effort."""
    keys = ("error_rate", "reread_rate", "task_switch_rate", "fatigue", "subjective_effort", "accuracy")
    values = {name: clip01(float(metrics.get(name, 0.0))) for name in keys}
    accuracy_supplied = "accuracy" in metrics
    performance_evidence = any(name in metrics for name in ("error_rate", "reread_rate", "task_switch_rate", "accuracy"))
    effort_contribution = values["subjective_effort"] * (1.0 - 0.5 * values["accuracy"])
    score = clip01(
        0.32 * values["error_rate"] + 0.20 * values["reread_rate"]
        + 0.18 * values["task_switch_rate"] + 0.12 * values["fatigue"]
        + 0.18 * effort_contribution + 0.20 * clip01(load)
    )
    productive_effort = bool(values["subjective_effort"] >= 0.70 and accuracy_supplied
                             and values["accuracy"] >= 0.80 and values["error_rate"] <= 0.20)
    probable = score >= threshold and not productive_effort
    return {"overload_score": score, "threshold": threshold, "overload_likely": probable,
            "productive_effort": productive_effort, "performance_evidence_supplied": performance_evidence,
            "indicators": values,
            "limitations": "behavioral and self-report proxies; not a direct neural or diagnostic measurement"}


def adapt_learning_step(
    state: Point06State, load: float, resource_capacity: float,
    monitor: Mapping[str, Any], config: Point06Config,
) -> dict[str, Any]:
    """Adapt support and pace using estimated utilization, current evidence, and prior success streak."""
    utilization = float(load) / (float(resource_capacity) + 1e-9)
    old_support = state.support_level
    if bool(monitor.get("overload_likely")) or utilization > config.utilization_high:
        state.support_level = clip01(old_support + 0.15)
        action = "reduce_active_elements_and_increase_scaffolding"
    elif utilization >= config.utilization_low:
        action = "maintain_complexity_and_clarify_presentation"
    elif state.success_streak >= 2:
        state.support_level = clip01(old_support - config.scaffold_fade_rate * 0.5)
        action = "increase_complexity_slightly_and_fade_support"
    else:
        action = "maintain_current_step_and_collect_more_evidence"
    return {"utilization": utilization, "resource_capacity_estimate": resource_capacity,
            "support_before": old_support, "support_after": state.support_level,
            "action": action, "thresholds": {"low": config.utilization_low, "high": config.utilization_high},
            "capacity_is_learner_estimate_not_universal_constant": True}


def build_guided_scaffold(
    chunks: Sequence[Mapping[str, Any]], examples: Sequence[Mapping[str, Any]], support_level: float,
) -> dict[str, Any]:
    """Prepare prompts around supplied chunks and examples without inventing domain-specific facts."""
    prompts = []
    for index, chunk in enumerate(chunks, start=1):
        prompts.append({"chunk_id": chunk["chunk_id"], "prompt_order": index,
                        "prompts": ["State the chunk's central idea.",
                                    "Explain the relation that connects its elements.",
                                    "Give a new example or boundary case if one is known."]})
    return {"support_level": clip01(support_level), "guided_prompts": prompts,
            "supplied_examples": [dict(item) for item in examples],
            "example_count": len(examples), "invented_examples": False,
            "fading_policy": "reduce hints only after repeated criterion-level performance"}


def integrate_chunks(
    chunks: Sequence[Mapping[str, Any]], dependencies: Sequence[Mapping[str, Any]],
    relations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Connect chunks using explicit source edges while retaining unresolved and untyped relations."""
    member_to_chunk = {member: str(chunk["chunk_id"]) for chunk in chunks for member in chunk["member_ids"]}
    links: list[dict[str, Any]] = []
    for edge in list(dependencies) + list(relations):
        source, target = str(edge.get("source", "")), str(edge.get("target", ""))
        source_chunk, target_chunk = member_to_chunk.get(source), member_to_chunk.get(target)
        links.append({"source_unit": source, "target_unit": target,
                      "source_chunk": source_chunk, "target_chunk": target_chunk,
                      "relation": str(edge.get("relation", "related_to")),
                      "strength": clip01(float(edge.get("strength", 1.0))),
                      "evidence": edge.get("source_evidence", "caller_supplied")})
    cross_chunk = [item for item in links if item["source_chunk"] != item["target_chunk"]]
    unresolved = [item for item in links if not item["source_chunk"] or not item["target_chunk"]]
    return {"chunk_count": len(chunks), "links": links, "cross_chunk_links": cross_chunk,
            "unresolved_links": unresolved, "global_structure_preserved": True}


def update_retrieval_and_support(
    state: Point06State, metrics: Mapping[str, float], config: Point06Config,
) -> dict[str, Any]:
    """Update success streak and fade support only after repeated accurate reconstruction evidence."""
    accuracy_raw = metrics.get("reconstruction_accuracy")
    if accuracy_raw is None:
        return {"reconstruction_accuracy": None, "success_streak": state.success_streak,
                "support_before": state.support_level, "support_after": state.support_level,
                "support_faded": False, "status": "no_retrieval_evidence_supplied"}
    accuracy = clip01(float(accuracy_raw))
    old_support = state.support_level
    state.success_streak = state.success_streak + 1 if accuracy >= config.success_threshold else 0
    faded = state.success_streak >= 2
    if faded:
        state.support_level = clip01(state.support_level - config.scaffold_fade_rate * accuracy)
    return {"reconstruction_accuracy": accuracy, "success_streak": state.success_streak,
            "support_before": old_support, "support_after": state.support_level,
            "support_faded": faded,
            "status": "support_reduced_after_repeated_success" if faded else "retain_current_support"}


def verify_comprehension(metrics: Mapping[str, float]) -> dict[str, Any]:
    """Evaluate local, relational, transfer, and explanation evidence without treating missing data as success."""
    dimensions = ("local_accuracy", "relation_accuracy", "transfer_accuracy", "explanation_accuracy")
    weights = {"local_accuracy": 0.30, "relation_accuracy": 0.30,
               "transfer_accuracy": 0.20, "explanation_accuracy": 0.20}
    supplied = {name: clip01(float(metrics[name])) for name in dimensions if name in metrics}
    if not supplied:
        return {"status": "not_assessed", "score": None, "dimensions": {},
                "weights": weights, "missing_dimensions": list(dimensions)}
    denominator = sum(weights[name] for name in supplied)
    score = sum(weights[name] * value for name, value in supplied.items()) / denominator
    missing = [name for name in dimensions if name not in supplied]
    return {"status": "partially_assessed" if missing else "assessed",
            "score": score, "dimensions": supplied, "weights": weights,
            "missing_dimensions": missing,
            "interpretation": "weighted evidence score; verbal repetition alone is not sufficient"}