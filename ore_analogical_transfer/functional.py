"""Graph matching, relational abstraction, analogy scoring, and guarded transfer operations."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import math
from typing import Any, Mapping, Sequence

from .contracts import (
    AnalogicalConfig, AnalogicalEntity, AnalogicalRelation, Disanalogy,
    EntityCorrespondence, HigherOrderRelation, MappingCandidate, ParameterBridge,
    SituationGraph, SourceCandidate, TransferConstraint, TransferHypothesis,
)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Bound a finite value to a closed interval."""
    number = float(value)
    if not math.isfinite(number) or low > high:
        raise ValueError("value must be finite and low must not exceed high")
    return min(high, max(low, number))


def _tokens(value: Any) -> set[str]:
    """Create a lightweight case-folded token set without a learned language model."""
    text = str(value).casefold()
    return {token for token in "".join(
        char if char.isalnum() else " " for char in text
    ).split() if token}


def _jaccard(left: set[str], right: set[str]) -> float:
    """Compute Jaccard overlap with a neutral score for two empty feature sets."""
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def canonical_relation_name(name: str, config: AnalogicalConfig) -> str:
    """Map a domain-specific predicate to a declared abstract relation label."""
    normalized = str(name).casefold().strip()
    return config.relation_aliases.get(normalized, normalized)


def relation_signature(relation: AnalogicalRelation, config: AnalogicalConfig) -> tuple[Any, ...]:
    """Return a strict signature including relation arity, direction, and causal status."""
    return (
        canonical_relation_name(relation.name, config),
        len(relation.arguments),
        relation.direction,
        relation.causal_status,
    )


def relation_similarity(
    source: AnalogicalRelation, target: AnalogicalRelation, config: AnalogicalConfig
) -> float:
    """Compare relation labels and signatures while refusing direction/arity mismatches."""
    if len(source.arguments) != len(target.arguments) or source.direction != target.direction:
        return 0.0
    source_name = canonical_relation_name(source.name, config)
    target_name = canonical_relation_name(target.name, config)
    if source_name == target_name:
        label_score = 1.0
    else:
        label_score = _jaccard(_tokens(source_name), _tokens(target_name))
    if label_score <= 0.0:
        return 0.0
    if source.causal_status == target.causal_status:
        causal_score = 1.0
    elif "unspecified" in {source.causal_status, target.causal_status}:
        causal_score = 0.7
    else:
        causal_score = 0.0
    source_conditions = {str(k) + "=" + str(v) for k, v in source.conditions.items()}
    target_conditions = {str(k) + "=" + str(v) for k, v in target.conditions.items()}
    condition_score = 1.0 if not source_conditions or not target_conditions else _jaccard(
        source_conditions, target_conditions
    )
    quantity_score = (
        1.0 if not source.quantitative_form or not target.quantitative_form
        else float(source.quantitative_form.casefold() == target.quantitative_form.casefold())
    )
    return clamp(0.60 * label_score + 0.20 * causal_score
                 + 0.10 * condition_score + 0.10 * quantity_score)


def _incident_role_signatures(
    entity_id: str, graph: SituationGraph, config: AnalogicalConfig
) -> list[tuple[str, int, int, str]]:
    """Describe the entity's argument roles across incident first-order relations."""
    signatures = []
    for relation in graph.relations:
        if entity_id not in relation.arguments:
            continue
        signatures.append((
            canonical_relation_name(relation.name, config),
            relation.arguments.index(entity_id),
            len(relation.arguments),
            relation.causal_status,
        ))
    return signatures


def _role_similarity(
    source: AnalogicalEntity, source_graph: SituationGraph,
    target: AnalogicalEntity, target_graph: SituationGraph,
    config: AnalogicalConfig,
) -> float:
    """Compare relational position and explicit role hints independently of surface properties."""
    source_roles = set(source.role_hints)
    target_roles = set(target.role_hints)
    hint_similarity = _jaccard(source_roles, target_roles)
    source_signature = {
        ":".join(map(str, item))
        for item in _incident_role_signatures(source.entity_id, source_graph, config)
    }
    target_signature = {
        ":".join(map(str, item))
        for item in _incident_role_signatures(target.entity_id, target_graph, config)
    }
    relation_position = _jaccard(source_signature, target_signature)
    if source_roles or target_roles:
        return 0.80 * hint_similarity + 0.20 * relation_position
    return relation_position


def _surface_similarity(source: AnalogicalEntity, target: AnalogicalEntity) -> float:
    """Compare entity types, descriptive labels, and attributes as a deliberately separate channel."""
    type_score = float(source.kind == target.kind)
    label_score = _jaccard(_tokens(source.label), _tokens(target.label))
    source_attrs = {
        str(key) + "=" + str(value).casefold()
        for key, value in source.attributes.items()
    }
    target_attrs = {
        str(key) + "=" + str(value).casefold()
        for key, value in target.attributes.items()
    }
    attribute_score = _jaccard(source_attrs, target_attrs)
    return 0.25 * type_score + 0.25 * label_score + 0.50 * attribute_score


def _goal_similarity(source: AnalogicalEntity, target: AnalogicalEntity,
                     source_graph: SituationGraph, target_graph: SituationGraph) -> float:
    """Estimate whether an entity's role hints support the current transfer goal."""
    goals = set(source_graph.goals) & set(target_graph.goals)
    if not goals:
        return 0.0
    role_overlap = len((set(source.role_hints) | set(target.role_hints)) & goals)
    return clamp(role_overlap / max(1, len(goals)))


def candidate_entity_correspondences(
    source: SituationGraph,
    target: SituationGraph,
    config: AnalogicalConfig | None = None,
) -> tuple[EntityCorrespondence, ...]:
    """Generate scored source-target role candidates without forcing name or appearance identity."""
    cfg = config or AnalogicalConfig()
    rows: list[EntityCorrespondence] = []
    weight_total = cfg.surface_weight + cfg.role_weight + cfg.relation_weight
    surface_weight = cfg.surface_weight / max(1e-9, weight_total)
    role_weight = cfg.role_weight / max(1e-9, weight_total)
    relation_weight = cfg.relation_weight / max(1e-9, weight_total)
    for source_entity in source.entities.values():
        for target_entity in target.entities.values():
            role = _role_similarity(source_entity, source, target_entity, target, cfg)
            surface = _surface_similarity(source_entity, target_entity)
            goal = _goal_similarity(source_entity, target_entity, source, target)
            relational = clamp(0.75 * role + 0.25 * goal)
            score = (
                role_weight * role + surface_weight * surface
                + relation_weight * relational
                - cfg.surface_distraction_penalty * surface * (1.0 - role) * surface_weight
            )
            if score >= cfg.mapping_threshold:
                rows.append(EntityCorrespondence(
                    source_entity_id=source_entity.entity_id,
                    target_entity_id=target_entity.entity_id,
                    score=score, role_similarity=role,
                    relational_similarity=relational, surface_similarity=surface,
                    goal_similarity=goal, confidence=clamp(score),
                ))
    return tuple(sorted(rows, key=lambda item: (
        -item.score, item.source_entity_id, item.target_entity_id
    )))


def generate_partial_mappings(
    source: SituationGraph,
    target: SituationGraph,
    correspondences: Sequence[EntityCorrespondence],
    config: AnalogicalConfig | None = None,
) -> tuple[dict[str, str], ...]:
    """Search high-scoring partial one-to-one entity mappings with an explicit unmatched choice."""
    cfg = config or AnalogicalConfig()
    by_source: dict[str, list[EntityCorrespondence]] = defaultdict(list)
    for pair in correspondences:
        by_source[pair.source_entity_id].append(pair)
    source_ids = sorted(
        source.entities,
        key=lambda key: (-source.entities[key].importance,
                         -max((p.score for p in by_source[key]), default=0.0), key),
    )[:cfg.max_source_entities]
    beam: list[tuple[float, dict[str, str], frozenset[str]]] = [(0.0, {}, frozenset())]
    for source_id in source_ids:
        next_beam: list[tuple[float, dict[str, str], frozenset[str]]] = []
        for score, mapping, used_targets in beam:
            next_beam.append((score, dict(mapping), used_targets))
            for pair in by_source.get(source_id, ()):
                if pair.target_entity_id in used_targets:
                    continue
                extended = dict(mapping)
                extended[source_id] = pair.target_entity_id
                next_beam.append((
                    score + pair.score * source.entities[source_id].importance,
                    extended, used_targets | {pair.target_entity_id},
                ))
        next_beam.sort(key=lambda item: (
            -item[0], tuple(sorted(item[1].items()))
        ))
        beam = next_beam[:max(cfg.mapping_limit * 8, 32)]
    unique: dict[tuple[tuple[str, str], ...], dict[str, str]] = {}
    for _, mapping, _ in beam:
        if mapping:
            unique[tuple(sorted(mapping.items()))] = mapping
    ranked = sorted(unique.values(), key=lambda mapping: (
        -sum(next((pair.score for pair in correspondences
                   if pair.source_entity_id == s and pair.target_entity_id == t), 0.0)
             for s, t in mapping.items()),
        tuple(sorted(mapping.items())),
    ))
    return tuple(ranked[:cfg.mapping_limit])


def _target_relation_match(
    source_relation: AnalogicalRelation,
    mapped_arguments: tuple[str, ...],
    target: SituationGraph,
    config: AnalogicalConfig,
) -> tuple[AnalogicalRelation | None, float, bool]:
    """Find the best target edge with mapped arguments and flag incompatible direction or causality."""
    best: AnalogicalRelation | None = None
    best_score = 0.0
    contradiction = False
    for target_relation in target.relations:
        if target_relation.arguments != mapped_arguments:
            if source_relation.direction == "undirected" and target_relation.direction == "undirected":
                if set(target_relation.arguments) != set(mapped_arguments):
                    continue
            else:
                continue
        similarity = relation_similarity(source_relation, target_relation, config)
        if similarity > best_score:
            best, best_score = target_relation, similarity
        if (
            canonical_relation_name(source_relation.name, config)
            == canonical_relation_name(target_relation.name, config)
            and (source_relation.direction != target_relation.direction
                 or (source_relation.causal_status != "unspecified"
                     and target_relation.causal_status != "unspecified"
                     and source_relation.causal_status != target_relation.causal_status)
                 or target_relation.evidential_status == "contradicted")
        ):
            contradiction = True
    return best, best_score, contradiction


def _largest_component_ratio(
    relations: Sequence[AnalogicalRelation], selected_ids: set[str]
) -> float:
    """Estimate whether preserved relations belong to one connected relational component."""
    chosen = [item for item in relations if item.relation_id in selected_ids]
    if not chosen:
        return 0.0
    neighbors: dict[str, set[str]] = defaultdict(set)
    for relation in chosen:
        for left in relation.arguments:
            for right in relation.arguments:
                if left != right:
                    neighbors[left].add(right)
    visited: set[str] = set()
    largest = 0
    for node in neighbors:
        if node in visited:
            continue
        stack = [node]
        component: set[str] = set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            visited.add(current)
            stack.extend(neighbors[current] - component)
        largest = max(largest, len(component))
    largest_relation_count = sum(
        1 for item in chosen if set(item.arguments).issubset(
            max(
                (component for component in _connected_components(chosen)),
                key=len, default=set()
            )
        )
    )
    return largest_relation_count / len(chosen) if chosen else 0.0


def _connected_components(relations: Sequence[AnalogicalRelation]) -> list[set[str]]:
    """Return entity components induced by a selected set of relation edges."""
    neighbors: dict[str, set[str]] = defaultdict(set)
    for relation in relations:
        for left in relation.arguments:
            neighbors[left].update(item for item in relation.arguments if item != left)
    components: list[set[str]] = []
    visited: set[str] = set()
    for node in neighbors:
        if node in visited:
            continue
        stack = [node]
        component: set[str] = set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            visited.add(current)
            stack.extend(neighbors[current] - component)
        components.append(component)
    return components


def score_mapping(
    source: SituationGraph,
    target: SituationGraph,
    entity_mapping: Mapping[str, str],
    correspondences: Sequence[EntityCorrespondence],
    config: AnalogicalConfig | None = None,
) -> MappingCandidate:
    """Score node-role matches, preserved edges, contradictions, and higher-order coherence."""
    cfg = config or AnalogicalConfig()
    reverse_pairs = {
        (item.source_entity_id, item.target_entity_id): item for item in correspondences
    }
    selected_pairs = [
        reverse_pairs[(source_id, target_id)]
        for source_id, target_id in entity_mapping.items()
        if (source_id, target_id) in reverse_pairs
    ]
    node_score = (
        sum(item.score for item in selected_pairs) / len(selected_pairs)
        if selected_pairs else 0.0
    )
    preserved: list[str] = []
    missing: list[str] = []
    contradicted: list[str] = []
    relation_pairs: list[tuple[str, str]] = []
    weighted_supported = 0.0
    weighted_eligible = 0.0
    target_relation_by_id = {item.relation_id: item for item in target.relations}
    source_relation_map = {item.relation_id: item for item in source.relations}
    mapping_by_source_relation: dict[str, str] = {}
    for relation in source.relations:
        if relation.evidential_status == "contradicted":
            continue
        weighted_eligible += relation.importance * relation.confidence
        if any(argument not in entity_mapping for argument in relation.arguments):
            missing.append(relation.relation_id)
            continue
        mapped_args = tuple(entity_mapping[item] for item in relation.arguments)
        match, similarity, conflict = _target_relation_match(relation, mapped_args, target, cfg)
        if conflict:
            contradicted.append(relation.relation_id)
        if match is not None and similarity >= 0.5 and not conflict:
            preserved.append(relation.relation_id)
            relation_pairs.append((relation.relation_id, match.relation_id))
            mapping_by_source_relation[relation.relation_id] = match.relation_id
            weighted_supported += relation.importance * relation.confidence * similarity
        else:
            missing.append(relation.relation_id)
    relation_score = weighted_supported / max(1e-9, weighted_eligible)
    structural_consistency = clamp(weighted_supported / max(1e-9, weighted_eligible))
    higher_preserved = 0.0
    higher_weight = 0.0
    target_higher = target.higher_order_relations
    for link in source.higher_order_relations:
        higher_weight += link.importance * link.confidence
        left = mapping_by_source_relation.get(link.subject_relation_id)
        right = mapping_by_source_relation.get(link.object_relation_id)
        if left and right and any(
            candidate.subject_relation_id == left
            and candidate.object_relation_id == right
            and candidate.predicate == link.predicate
            for candidate in target_higher
        ):
            higher_preserved += link.importance * link.confidence
        elif left and right and any(
            candidate.subject_relation_id == right
            and candidate.object_relation_id == left
            and candidate.predicate == link.predicate
            for candidate in target_higher
        ):
            contradicted.append(link.link_id)
    higher_score = higher_preserved / max(1e-9, higher_weight) if higher_weight else 0.0
    preserved_relations = [source_relation_map[key] for key in preserved]
    connectedness = _largest_component_ratio(preserved_relations, set(preserved))
    contradiction_mass = sum(
        source_relation_map[key].importance for key in contradicted if key in source_relation_map
    )
    contradiction_mass += sum(
        link.importance for link in source.higher_order_relations if link.link_id in contradicted
    )
    coherence = clamp(0.70 * relation_score + 0.20 * connectedness + 0.10 * higher_score)
    total = (
        0.30 * node_score + 0.55 * relation_score
        + 0.10 * connectedness + 0.05 * higher_score
        - cfg.contradiction_penalty * contradiction_mass / max(1.0, len(source.relations))
    )
    coverage = len(selected_pairs) / max(1, min(len(source.entities), cfg.max_source_entities))
    confidence = clamp(
        0.55 * structural_consistency + 0.25 * node_score + 0.20 * coherence
    ) * clamp(coverage)
    return MappingCandidate(
        mapping_id="map:" + hashlib.sha256(
            repr(tuple(sorted(entity_mapping.items()))).encode("utf-8")
        ).hexdigest()[:12],
        correspondences=tuple(selected_pairs),
        relation_pairs=tuple(relation_pairs),
        preserved_relation_ids=tuple(sorted(preserved)),
        missing_relation_ids=tuple(sorted(set(missing))),
        contradicted_relation_ids=tuple(sorted(set(contradicted))),
        node_score=node_score, relation_score=relation_score,
        structural_consistency=structural_consistency, connectedness=connectedness,
        coherence_score=coherence, total_score=total, confidence=confidence,
    )


def rank_mappings(
    source: SituationGraph,
    target: SituationGraph,
    config: AnalogicalConfig | None = None,
) -> tuple[MappingCandidate, ...]:
    """Generate and structurally rank bounded partial entity mappings."""
    cfg = config or AnalogicalConfig()
    pairs = candidate_entity_correspondences(source, target, cfg)
    mappings = generate_partial_mappings(source, target, pairs, cfg)
    results = [
        score_mapping(source, target, mapping, pairs, cfg)
        for mapping in mappings
    ]
    results.sort(key=lambda item: (
        -item.total_score, -item.structural_consistency,
        tuple(sorted(item.entity_map.items()))
    ))
    return tuple(results[:cfg.mapping_limit])


def _cue_similarity(source: SituationGraph, target: SituationGraph, config: AnalogicalConfig) -> float:
    """Compare relational cue signatures, emphasizing predicates and argument roles over labels."""
    source_relations = {
        canonical_relation_name(item.name, config) + "/" + str(len(item.arguments))
        for item in source.relations
    }
    target_relations = {
        canonical_relation_name(item.name, config) + "/" + str(len(item.arguments))
        for item in target.relations
    }
    return _jaccard(source_relations, target_relations)


def _goal_similarity_graph(source: SituationGraph, target: SituationGraph) -> float:
    """Compare declared task goals for source retrieval."""
    return _jaccard(set(source.goals), set(target.goals))


def _relational_relevance(source: SituationGraph, target: SituationGraph,
                          config: AnalogicalConfig) -> float:
    """Measure shared abstract relation vocabulary independently from surface labels."""
    left = {canonical_relation_name(item.name, config) for item in source.relations}
    right = {canonical_relation_name(item.name, config) for item in target.relations}
    return _jaccard(left, right)


def retrieve_sources(
    target: SituationGraph,
    memory: Sequence[SituationGraph],
    config: AnalogicalConfig | None = None,
) -> tuple[tuple[SourceCandidate, ...], tuple[SituationGraph, ...]]:
    """Rank stored source graphs by relational cue, goal, and higher-order structural relevance."""
    cfg = config or AnalogicalConfig()
    rows: list[tuple[SourceCandidate, SituationGraph]] = []
    for source in memory:
        if source.case_id == target.case_id:
            continue
        cue = _cue_similarity(source, target, cfg)
        goal = _goal_similarity_graph(source, target)
        relation = _relational_relevance(source, target, cfg)
        higher = _jaccard(
            {item.predicate for item in source.higher_order_relations},
            {item.predicate for item in target.higher_order_relations},
        )
        score = 0.45 * cue + 0.20 * goal + 0.25 * relation + 0.10 * higher
        candidate = SourceCandidate(
            case_id=source.case_id, domain=source.domain, retrieval_score=score,
            cue_similarity=cue, goal_similarity=goal,
            relational_relevance=relation, source_reference=source.source_reference,
        )
        rows.append((candidate, source))
    rows.sort(key=lambda row: (-row[0].retrieval_score, row[0].case_id))
    selected = rows[:cfg.retrieval_limit]
    return tuple(row[0] for row in selected), tuple(row[1] for row in selected)


def prioritize_working_memory(
    source: SituationGraph,
    target: SituationGraph,
    mapping: MappingCandidate,
    config: AnalogicalConfig | None = None,
) -> tuple[dict[str, Any], ...]:
    """Select goal-relevant relations under a capacity budget and suppress distracting matches."""
    cfg = config or AnalogicalConfig()
    active_source = set(mapping.preserved_relation_ids) | set(mapping.missing_relation_ids)
    rows: list[dict[str, Any]] = []
    for relation in source.relations:
        if relation.relation_id not in active_source:
            continue
        arguments = [source.entities[item] for item in relation.arguments]
        centrality = sum(entity.importance for entity in arguments) / len(arguments)
        goal_rel = float(any(
            goal in relation.name or any(goal in hint for entity in arguments for hint in entity.role_hints)
            for goal in target.goals
        ))
        evidence = relation.confidence * relation.importance
        interference = 0.0
        for entity in arguments:
            target_entity_id = mapping.entity_map.get(entity.entity_id)
            if target_entity_id:
                target_entity = target.entities[target_entity_id]
                surface = _surface_similarity(entity, target_entity)
                role = _role_similarity(entity, source, target_entity, target, cfg)
                interference += surface * (1.0 - role)
        interference /= max(1, len(arguments))
        priority = (
            0.30 * centrality + 0.25 * goal_rel + 0.25 * evidence
            + 0.20 * centrality - cfg.interference_penalty * interference
        )
        rows.append({
            "relation_id": relation.relation_id, "priority": priority,
            "cost": max(0.1, len(relation.arguments) * 0.5),
            "goal_relevance": goal_rel, "evidence_strength": evidence,
            "interference": interference,
        })
    rows.sort(key=lambda item: (-item["priority"], item["relation_id"]))
    chosen: list[dict[str, Any]] = []
    spent = 0.0
    for row in rows:
        if spent + row["cost"] <= cfg.working_memory_budget:
            chosen.append({**row, "active": True})
            spent += row["cost"]
        else:
            chosen.append({**row, "active": False})
    for item in chosen:
        item["budget_spent"] = spent
    return tuple(chosen)


def recurrent_relation_activation(
    active_relation_ids: Sequence[str],
    source: SituationGraph,
    mapping: MappingCandidate,
    iterations: int = 3,
) -> dict[str, float]:
    """Propagate support among relations that share mapped entities as a recurrent activation proxy."""
    active = set(active_relation_ids)
    relations = {item.relation_id: item for item in source.relations}
    activation = {
        key: (relations[key].confidence * relations[key].importance if key in relations else 0.0)
        for key in active
    }
    selected = [relations[key] for key in active if key in relations]
    for _ in range(max(1, iterations)):
        updated: dict[str, float] = {}
        for relation in selected:
            neighbors = [
                other for other in selected if other.relation_id != relation.relation_id
                and set(other.arguments) & set(relation.arguments)
            ]
            feedback = sum(activation[item.relation_id] for item in neighbors) / max(1, len(neighbors))
            updated[relation.relation_id] = clamp(
                0.70 * activation[relation.relation_id] + 0.30 * feedback
            )
        activation = updated
    return activation


def _conditions_fit(relation: AnalogicalRelation, target: SituationGraph) -> tuple[float, tuple[str, ...]]:
    """Compare source relation assumptions with target context and expose unmet conditions."""
    if not relation.conditions:
        return 1.0, ()
    observed = {**target.context}
    met = 0
    unmet: list[str] = []
    for key, expected in relation.conditions.items():
        if key not in observed:
            unmet.append("target context does not specify " + str(key))
        elif observed[key] == expected:
            met += 1
        else:
            unmet.append("target context differs for " + str(key))
    score = (met + 0.35 * len(unmet)) / max(1, len(relation.conditions))
    return clamp(score), tuple(unmet)


def _constraint_check(
    relation: AnalogicalRelation,
    target: SituationGraph,
    config: AnalogicalConfig,
) -> tuple[bool, tuple[Disanalogy, ...]]:
    """Apply explicit hard/soft target constraints to one transferred relation."""
    canonical = canonical_relation_name(relation.name, config)
    issues: list[Disanalogy] = []
    reject = False
    for constraint in target.constraints:
        constraint_predicate = canonical_relation_name(constraint.predicate, config) if constraint.predicate else ""
        applies = not constraint_predicate or constraint_predicate == canonical
        if not applies:
            continue
        if constraint.kind == "unsupported_relation":
            if constraint_predicate == canonical:
                reject = reject or constraint.severity == "hard"
                issues.append(Disanalogy(
                    "constraint:" + constraint.constraint_id, "domain_limit",
                    constraint.explanation or "Target domain declares this relation unsupported",
                    (relation.relation_id,),
                    "hard_stop" if constraint.severity == "hard" else "warning",
                ))
        elif constraint.kind in {"forbid_relation", "require_relation"}:
            endpoints_match = (
                (not constraint.subject or constraint.subject in relation.arguments)
                and (not constraint.object or constraint.object in relation.arguments)
            )
            if endpoints_match and constraint.kind == "forbid_relation":
                reject = reject or constraint.severity == "hard"
                issues.append(Disanalogy(
                    "constraint:" + constraint.constraint_id, "forbidden_target_relation",
                    constraint.explanation or "Target constraints forbid the transferred relation",
                    (relation.relation_id,),
                    "hard_stop" if constraint.severity == "hard" else "warning",
                ))
        elif constraint.kind in {"condition_equals", "condition_not_equals"}:
            if constraint.key not in target.context:
                issue = True
            elif constraint.kind == "condition_equals":
                issue = target.context[constraint.key] != constraint.expected_value
            else:
                issue = target.context[constraint.key] == constraint.expected_value
            if issue:
                reject = reject or constraint.severity == "hard"
                issues.append(Disanalogy(
                    "constraint:" + constraint.constraint_id, "boundary_condition",
                    constraint.explanation or "Transferred relation boundary condition is not met",
                    (relation.relation_id,),
                    "hard_stop" if constraint.severity == "hard" else "warning",
                ))
    return not reject, tuple(issues)


def build_transfer_hypotheses(
    source: SituationGraph,
    target: SituationGraph,
    mapping: MappingCandidate,
    config: AnalogicalConfig | None = None,
) -> tuple[tuple[TransferHypothesis, ...], tuple[TransferHypothesis, ...], tuple[Disanalogy, ...]]:
    """Project unmapped source edges into the target only when all argument roles are mapped."""
    cfg = config or AnalogicalConfig()
    hypotheses: list[TransferHypothesis] = []
    rejected: list[TransferHypothesis] = []
    disanalogies: list[Disanalogy] = []
    target_edges = target.relations
    for relation in source.relations:
        if relation.relation_id in mapping.preserved_relation_ids:
            continue
        entity_map = mapping.entity_map
        if any(argument not in entity_map for argument in relation.arguments):
            continue
        mapped_arguments = tuple(entity_map[item] for item in relation.arguments)
        exact_named_edges = [
            edge for edge in target_edges
            if edge.arguments == mapped_arguments
            and canonical_relation_name(edge.name, cfg) == canonical_relation_name(relation.name, cfg)
        ]
        conflicting_named_edges = [
            edge for edge in exact_named_edges
            if relation_similarity(relation, edge, cfg) < 0.5
            or edge.evidential_status == "contradicted"
        ]
        if exact_named_edges and not conflicting_named_edges:
            continue
        condition_fit, caveats = _conditions_fit(relation, target)
        confidence = clamp(
            mapping.confidence * relation.confidence * condition_fit
            * max(0.0, 1.0 - 0.20 * len(mapping.contradicted_relation_ids))
        )
        mapped_relation = AnalogicalRelation(
            relation_id="transfer:" + relation.relation_id,
            name=relation.name, arguments=mapped_arguments,
            direction=relation.direction, conditions=relation.conditions,
            confidence=confidence, importance=relation.importance,
            causal_status=relation.causal_status, quantitative_form=relation.quantitative_form,
            evidential_status="hypothesized", provenance="analogical_transfer",
        )
        allowed, issues = _constraint_check(mapped_relation, target, cfg)
        disanalogies.extend(issues)
        status = "candidate"
        reason = ""
        if conflicting_named_edges:
            disanalogies.append(Disanalogy(
                "relation-conflict:" + relation.relation_id,
                "relational_disanalogy",
                "Target contains a relation with the same mapped endpoints but incompatible direction, causal status, or evidence",
                (relation.relation_id,), "hard_stop", "observed_target_relation",
            ))
            allowed = False
        if not allowed or confidence < cfg.transfer_confidence_threshold:
            status = "rejected"
            reason = (
                "target_relation_conflict" if conflicting_named_edges
                else "hard_target_constraint" if not allowed
                else "low_transfer_confidence"
            )
        elif caveats:
            status = "qualified"
        hypothesis = TransferHypothesis(
            hypothesis_id="hyp:" + relation.relation_id,
            source_relation_id=relation.relation_id, relation=mapped_relation,
            confidence=confidence,
            support_chain=(
                "source relation " + relation.relation_id,
                "role-preserving entity mapping " + mapping.mapping_id,
                "structural consistency " + format(mapping.structural_consistency, ".3f"),
            ),
            caveats=caveats, target_status=status, rejected_reason=reason,
        )
        if status == "rejected":
            rejected.append(hypothesis)
        else:
            hypotheses.append(hypothesis)
    for source_entity in source.entities.values():
        target_id = mapping.entity_map.get(source_entity.entity_id)
        if target_id is None:
            continue
        target_entity = target.entities[target_id]
        if source_entity.kind != target_entity.kind:
            disanalogies.append(Disanalogy(
                "kind:" + source_entity.entity_id, "surface_disanalogy",
                "Mapped entities differ in their domain-specific kinds; only role-level structure may transfer",
                tuple(item.relation_id for item in source.relations
                      if source_entity.entity_id in item.arguments),
                "warning",
            ))
    return tuple(hypotheses), tuple(rejected), tuple(disanalogies)


def translate_parameters(
    source: SituationGraph,
    bridges: Sequence[ParameterBridge],
) -> tuple[dict[str, Any], ...]:
    """Translate only explicitly bridged numeric context parameters with compatible dimensions."""
    output: list[dict[str, Any]] = []
    for bridge in bridges:
        if bridge.source_parameter not in source.context:
            continue
        value = source.context[bridge.source_parameter]
        if not isinstance(value, (int, float)):
            raise TypeError("parameter bridge source values must be numeric")
        output.append({
            "source_parameter": bridge.source_parameter,
            "source_value": float(value), "source_unit": bridge.source_unit,
            "target_parameter": bridge.target_parameter,
            "target_value": bridge.translate(float(value)),
            "target_unit": bridge.target_unit,
            "dimension": bridge.target_dimension or bridge.source_dimension,
            "bridge": bridge.to_dict(),
        })
    return tuple(output)


def abstract_relation_templates(
    source: SituationGraph,
    mapping: MappingCandidate,
    config: AnalogicalConfig | None = None,
) -> tuple[dict[str, Any], ...]:
    """Remove domain-specific entity IDs while retaining argument-role and relation structure."""
    cfg = config or AnalogicalConfig()
    templates = []
    for relation in source.relations:
        if relation.evidential_status == "contradicted":
            continue
        if any(argument not in mapping.entity_map for argument in relation.arguments):
            continue
        templates.append({
            "canonical_relation": canonical_relation_name(relation.name, cfg),
            "argument_roles": [
                source.entities[argument].role_hints or (source.entities[argument].kind,)
                for argument in relation.arguments
            ],
            "arity": len(relation.arguments), "direction": relation.direction,
            "causal_status": relation.causal_status,
            "conditions": dict(relation.conditions),
            "quantitative_form": relation.quantitative_form,
            "importance": relation.importance,
        })
    return tuple(templates)


def validate_transfer(
    result_hypotheses: Sequence[TransferHypothesis],
    observed_relations: Sequence[AnalogicalRelation],
    config: AnalogicalConfig | None = None,
) -> tuple[dict[str, Any], ...]:
    """Compare candidate transferred edges to later evidence without treating missing data as failure."""
    cfg = config or AnalogicalConfig()
    outcomes: list[dict[str, Any]] = []
    for hypothesis in result_hypotheses:
        predicted = hypothesis.relation
        matching = [
            item for item in observed_relations
            if item.arguments == predicted.arguments
            and canonical_relation_name(item.name, cfg) == canonical_relation_name(predicted.name, cfg)
        ]
        if not matching:
            continue
        observation = matching[0]
        actual = 0.0 if observation.evidential_status == "contradicted" else 1.0
        probability = clamp(hypothesis.confidence)
        brier = (probability - actual) ** 2
        surprise = -math.log(max(1e-9, probability if actual else 1.0 - probability))
        outcomes.append({
            "hypothesis_id": hypothesis.hypothesis_id,
            "source_relation_id": hypothesis.source_relation_id,
            "predicted_relation": predicted.to_dict(),
            "observed_relation": observation.to_dict(),
            "validated": bool(actual), "brier_score": brier, "surprise": surprise,
        })
    return tuple(outcomes)
