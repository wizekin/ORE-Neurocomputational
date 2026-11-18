"""Core statistical operations for schema roles, relations, temporal structure, and uncertainty.

These functions estimate reusable relational structure from multiple distinct episodes. Their
equations are computational models; they are not literal arithmetic attributed to neurons.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence

from .contracts import (
    EpisodeInput, RoleAssignment, SchemaConfig, SchemaRelation, SchemaSlot,
)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Bound a finite scalar to the requested closed interval."""
    number = float(value)
    if not math.isfinite(number) or low > high:
        raise ValueError("value must be finite and low must not exceed high")
    return min(high, max(low, number))


def canonical_role(role: str, aliases: Mapping[str, str] | None = None) -> str:
    """Normalize a role label and optionally map a known synonym to its canonical role."""
    normalized = " ".join(str(role).casefold().strip().replace("-", " ").split())
    return (aliases or {}).get(normalized, normalized)


def context_signature(context: Mapping[str, Any]) -> str:
    """Create a stable context key for context-conditioned event-transition estimates."""
    if not context:
        return "unspecified"
    payload = json.dumps(dict(context), sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _set_similarity(left: Iterable[str], right: Iterable[str]) -> float:
    """Return Jaccard similarity with a neutral empty-empty convention."""
    a, b = set(left), set(right)
    if not a or not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def event_boundary_scores(episode: EpisodeInput) -> list[dict[str, Any]]:
    """Estimate event-boundary salience between successive supplied event segments."""
    output: list[dict[str, Any]] = []
    for index, event in enumerate(episode.events):
        if index == 0:
            score = 1.0
            changes = {"first_event": True}
        else:
            previous = episode.events[index - 1]
            event_change = float(previous.event_type != event.event_type)
            prior_roles = set(previous.participants)
            next_roles = set(event.participants)
            role_change = 1.0 - _set_similarity(prior_roles, next_roles)
            context_change = float(dict(previous.context) != dict(event.context))
            time_gap = 0.0
            if previous.timestamp is not None and event.timestamp is not None:
                time_gap = min(1.0, abs(event.timestamp - previous.timestamp) / 60.0)
            score = clamp(0.40 * event_change + 0.25 * role_change
                          + 0.20 * context_change + 0.15 * time_gap)
            changes = {
                "event_type_changed": bool(event_change),
                "participant_roles_changed": bool(role_change),
                "event_context_changed": bool(context_change),
                "normalized_time_gap": time_gap,
            }
        output.append({"event_index": index, "event_id": event.event_id or str(index),
                       "event_type": event.event_type, "boundary_score": score,
                       "change_evidence": changes})
    return output


def episode_similarity(left: EpisodeInput, right: EpisodeInput,
                        aliases: Mapping[str, str] | None = None) -> float:
    """Compare situation, goals, participant roles, relation predicates, and event sequences."""
    roles_left = {
        canonical_role(role, aliases)
        for event in left.events for role in event.participants
    }
    roles_right = {
        canonical_role(role, aliases)
        for event in right.events for role in event.participants
    }
    def relation_patterns(episode: EpisodeInput) -> set[str]:
        """Represent observed relation direction using the participant roles of its endpoints."""
        entity_roles: dict[str, set[str]] = defaultdict(set)
        for event in episode.events:
            for role, entity_id in event.participants.items():
                entity_roles[entity_id].add(canonical_role(role, aliases))
        patterns: set[str] = set()
        for relation in episode.relations:
            subject = sorted(entity_roles.get(relation.subject_id, set()))
            object_roles = sorted(entity_roles.get(relation.object_id, set()))
            if subject and object_roles:
                for subject_role in subject:
                    for object_role in object_roles:
                        patterns.add(subject_role + ">" + relation.predicate.casefold() + ">" + object_role)
        return patterns

    relation_left = relation_patterns(left)
    relation_right = relation_patterns(right)
    event_left = {item.event_type for item in left.events}
    event_right = {item.event_type for item in right.events}
    context_left = {str(k) + "=" + str(v) for k, v in left.context.items()}
    context_right = {str(k) + "=" + str(v) for k, v in right.context.items()}
    parts = (
        (0.25, _set_similarity(event_left, event_right)),
        (0.20, _set_similarity(left.goals, right.goals)),
        (0.20, _set_similarity(roles_left, roles_right)),
        (0.20, _set_similarity(relation_left, relation_right)),
        (0.15, _set_similarity(context_left, context_right)),
    )
    return clamp(sum(weight * score for weight, score in parts))


def group_related_episodes(
    episodes: Sequence[EpisodeInput],
    threshold: float,
    aliases: Mapping[str, str] | None = None,
) -> list[list[EpisodeInput]]:
    """Cluster related experiences by thresholded connected components of pairwise similarity.

    Connected components are a transparent baseline, not a claim that the brain runs a
    global clustering algorithm. Borderline comparisons remain available in the score matrix.
    """
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must lie in [0, 1]")
    count = len(episodes)
    parent = list(range(count))

    def find(index: int) -> int:
        """Return the current disjoint-set representative with path compression."""
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        """Join two episode components when their similarity reaches the threshold."""
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for left in range(count):
        for right in range(left + 1, count):
            if episode_similarity(episodes[left], episodes[right], aliases) >= threshold:
                union(left, right)
    grouped: dict[int, list[EpisodeInput]] = defaultdict(list)
    for index, episode in enumerate(episodes):
        grouped[find(index)].append(episode)
    return sorted(grouped.values(), key=lambda group: min(item.episode_id for item in group))


def _raw_role_occurrences(
    episodes: Sequence[EpisodeInput], aliases: Mapping[str, str] | None = None
) -> dict[tuple[str, str], set[str]]:
    """Collect role labels observed for each entity in each distinct episode."""
    output: dict[tuple[str, str], set[str]] = defaultdict(set)
    for episode in episodes:
        for event in episode.events:
            for role, entity_id in event.participants.items():
                output[(episode.episode_id, entity_id)].add(canonical_role(role, aliases))
    return output


def infer_role_assignments(
    episodes: Sequence[EpisodeInput], config: SchemaConfig
) -> list[RoleAssignment]:
    """Score candidate functional roles while preserving entities with genuinely ambiguous roles."""
    occurrences = _raw_role_occurrences(episodes, config.role_aliases)
    entity_role_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    role_type_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for episode in episodes:
        for event in episode.events:
            for role, entity_id in event.participants.items():
                canonical = canonical_role(role, config.role_aliases)
                entity_role_counts[(episode.episode_id, entity_id)][canonical] += 1
                role_type_counts[canonical][episode.entities[entity_id].entity_type] += 1
    output: list[RoleAssignment] = []
    for (episode_id, entity_id), roles in sorted(occurrences.items()):
        episode_roles = entity_role_counts[(episode_id, entity_id)]
        total = max(1, sum(episode_roles.values()))
        entity = next(ep.entities[entity_id] for ep in episodes if ep.episode_id == episode_id)
        scores: dict[str, float] = {}
        for role in sorted(roles):
            role_total = sum(role_type_counts[role].values()) or 1
            type_match = role_type_counts[role][entity.entity_type] / role_total
            role_frequency = episode_roles[role] / total
            scores[role] = clamp(0.75 * role_frequency + 0.25 * type_match)
        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        selected = ordered[0][0] if ordered else None
        ambiguous = len(ordered) > 1 and ordered[0][1] - ordered[1][1] < 0.12
        output.append(RoleAssignment(
            episode_id=episode_id, entity_id=entity_id, role_scores=scores,
            selected_role=None if ambiguous else selected, ambiguous=ambiguous,
        ))
    return output


def induce_slots(
    episodes: Sequence[EpisodeInput],
    assignments: Sequence[RoleAssignment],
    config: SchemaConfig,
) -> dict[str, SchemaSlot]:
    """Abstract episode-specific participants into variable slots with type and filler distributions."""
    role_examples: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    role_types: dict[str, Counter[str]] = defaultdict(Counter)
    role_episode_support: dict[str, set[str]] = defaultdict(set)
    for episode in episodes:
        for event in episode.events:
            for role, entity_id in event.participants.items():
                name = canonical_role(role, config.role_aliases)
                role_examples[name][episode.episode_id].add(entity_id)
                role_types[name][episode.entities[entity_id].entity_type] += 1
                role_episode_support[name].add(episode.episode_id)
    total_episodes = max(1, len(episodes))
    slots: dict[str, SchemaSlot] = {}
    for name in sorted(role_examples):
        support_ids = role_episode_support[name]
        support = len(support_ids)
        coverage = support / total_episodes
        if total_episodes < 2:
            status = "uncertain"
        elif coverage >= config.core_coverage_threshold:
            status = "core"
        elif coverage >= config.typical_coverage_threshold:
            status = "typical"
        else:
            status = "optional"
        counts = role_types[name]
        total_types = sum(counts.values()) or 1
        type_probabilities = {
            entity_type: count / total_types for entity_type, count in sorted(counts.items())
        }
        fillers = tuple(sorted({
            filler for episode_fillers in role_examples[name].values()
            for filler in episode_fillers
        }))
        confidence = clamp(coverage * (support / (support + 1.0)))
        slots[name] = SchemaSlot(
            slot_id="slot:" + name, name=name,
            allowed_type_probabilities=type_probabilities, observed_fillers=fillers,
            support_episodes=support, coverage=coverage, status=status,
            confidence=confidence,
        )
    return slots


def _episode_role_maps(
    episode: EpisodeInput, config: SchemaConfig, assignments: Sequence[RoleAssignment]
) -> dict[str, str]:
    """Map episode entity identities to their best supported schema role when unambiguous."""
    assignment_map = {
        item.entity_id: item for item in assignments if item.episode_id == episode.episode_id
    }
    observed: dict[str, set[str]] = defaultdict(set)
    for event in episode.events:
        for role, entity_id in event.participants.items():
            observed[entity_id].add(canonical_role(role, config.role_aliases))
    output: dict[str, str] = {}
    for entity_id, roles in observed.items():
        assignment = assignment_map.get(entity_id)
        if assignment and assignment.selected_role:
            output[entity_id] = assignment.selected_role
        elif len(roles) == 1:
            output[entity_id] = next(iter(roles))
    return output


def induce_relations(
    episodes: Sequence[EpisodeInput],
    slots: Mapping[str, SchemaSlot],
    assignments: Sequence[RoleAssignment],
    config: SchemaConfig,
) -> tuple[SchemaRelation, ...]:
    """Align directed episode relations to slots and estimate support over all eligible episodes."""
    support_by_key: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    role_maps: dict[str, dict[str, str]] = {}
    present_roles_by_episode: dict[str, set[str]] = {}
    for episode in episodes:
        role_map = _episode_role_maps(episode, config, assignments)
        role_maps[episode.episode_id] = role_map
        present_roles_by_episode[episode.episode_id] = set(role_map.values())
        for relation in episode.relations:
            subject_role = role_map.get(relation.subject_id)
            object_role = role_map.get(relation.object_id)
            if not subject_role or not object_role:
                continue
            key = (subject_role, relation.predicate.casefold(), object_role,
                   relation.evidential_status)
            support_by_key[key].add(episode.episode_id)

    eligible_by_key: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    for key in support_by_key:
        subject, _, object_role, _ = key
        for episode in episodes:
            present = present_roles_by_episode[episode.episode_id]
            if subject in present and object_role in present:
                eligible_by_key[key].add(episode.episode_id)

    relations: list[SchemaRelation] = []
    for key in sorted(support_by_key):
        subject, predicate, object_role, evidence_status = key
        eligible_ids = eligible_by_key[key]
        support_ids = support_by_key[key]
        eligible_count = len(eligible_ids)
        support_count = len(support_ids)
        alpha, beta = config.relation_alpha, config.relation_beta
        confidence = (support_count + alpha) / (eligible_count + alpha + beta)
        if eligible_count < 2:
            status = "uncertain"
        elif confidence >= 0.85:
            status = "core"
        elif confidence >= 0.50:
            status = "typical"
        else:
            status = "optional"
        relations.append(SchemaRelation(
            subject_slot=subject, predicate=predicate, object_slot=object_role,
            evidential_status=evidence_status, support_count=support_count,
            eligible_count=eligible_count, confidence=clamp(confidence), status=status,
            supporting_episode_ids=tuple(sorted(support_ids)),
        ))
    return tuple(relations)

def estimate_transitions(
    episodes: Sequence[EpisodeInput], config: SchemaConfig
) -> tuple[dict[str, Any], ...]:
    """Estimate smoothed event-to-event transition probabilities globally and by exact context."""
    event_types = sorted({event.event_type for episode in episodes for event in episode.events})
    if not event_types:
        return ()
    buckets: dict[str, list[EpisodeInput]] = {"*": list(episodes)}
    for episode in episodes:
        buckets.setdefault(context_signature(episode.context), []).append(episode)
    output: list[dict[str, Any]] = []
    for context_key, bucket in sorted(buckets.items()):
        counts: Counter[tuple[str, str]] = Counter()
        eligible: Counter[str] = Counter()
        for episode in bucket:
            for left, right in zip(episode.events, episode.events[1:]):
                counts[(left.event_type, right.event_type)] += 1
                eligible[left.event_type] += 1
        for source in event_types:
            denominator = eligible[source]
            if denominator == 0:
                continue
            for target in event_types:
                count = counts[(source, target)]
                probability = (count + config.transition_alpha) / (
                    denominator + config.transition_alpha * len(event_types)
                )
                output.append({
                    "from_event": source, "to_event": target,
                    "context_key": context_key, "count": count,
                    "eligible_count": denominator, "probability": clamp(probability),
                    "mandatory": bool(count >= 2 and probability >= 0.85),
                    "status": "common" if probability >= 0.50 else "alternative",
                })
    return tuple(output)


def estimate_categorical_distribution(
    episodes: Sequence[EpisodeInput], values: Iterable[str]
) -> dict[str, float]:
    """Estimate a reliability-weighted categorical distribution over goals or outcomes."""
    counts: dict[str, float] = defaultdict(float)
    total = 0.0
    for episode in episodes:
        unique_values = set(values_for_episode(episode, values))
        for value in unique_values:
            counts[value] += episode.reliability
            total += episode.reliability
    if total <= 0.0:
        return {}
    return {key: value / total for key, value in sorted(counts.items())}


def values_for_episode(episode: EpisodeInput, selector: Iterable[str]) -> tuple[str, ...]:
    """Resolve the selected episode field used by categorical evidence aggregation."""
    selected = tuple(selector)
    if selected and selected[0] == "__goals__":
        return tuple(episode.goals)
    if selected and selected[0] == "__outcomes__":
        return tuple(episode.outcomes)
    return selected


def estimate_goal_distribution(episodes: Sequence[EpisodeInput]) -> dict[str, float]:
    """Estimate goal frequencies while counting each goal at most once per episode."""
    counts: dict[str, float] = defaultdict(float)
    total = 0.0
    for episode in episodes:
        for goal in set(episode.goals):
            counts[goal] += episode.reliability
            total += episode.reliability
    return {key: value / total for key, value in sorted(counts.items())} if total else {}


def estimate_outcome_distribution(episodes: Sequence[EpisodeInput]) -> dict[str, float]:
    """Estimate reliability-weighted outcome frequencies across supporting episodes."""
    counts: dict[str, float] = defaultdict(float)
    total = 0.0
    for episode in episodes:
        for outcome in set(episode.outcomes):
            counts[outcome] += episode.reliability
            total += episode.reliability
    return {key: value / total for key, value in sorted(counts.items())} if total else {}


def estimate_context_profile(episodes: Sequence[EpisodeInput]) -> dict[str, float]:
    """Estimate a compact frequency profile for observed key-value context cues."""
    counts: dict[str, float] = defaultdict(float)
    total = 0.0
    for episode in episodes:
        cues = {str(key) + "=" + str(value) for key, value in episode.context.items()}
        for cue in cues:
            counts[cue] += episode.reliability
            total += episode.reliability
    return {key: value / total for key, value in sorted(counts.items())} if total else {}


def rank_schema_candidates(
    episode: EpisodeInput, schemas: Sequence[Any], config: SchemaConfig
) -> list[dict[str, Any]]:
    """Rank existing schema frames against current roles, events, goals, and context."""
    observed_roles = {
        canonical_role(role, config.role_aliases)
        for event in episode.events for role in event.participants
    }
    observed_events = {event.event_type for event in episode.events}
    observed_goals = set(episode.goals)
    observed_context = {
        str(key) + "=" + str(value) for key, value in episode.context.items()
    }
    ranked: list[dict[str, Any]] = []
    for schema in schemas:
        roles = set(schema.slots)
        role_score = _set_similarity(observed_roles, roles)
        event_score = _set_similarity(observed_events, schema.event_types)
        goal_score = _set_similarity(observed_goals, schema.goals)
        context_score = _set_similarity(observed_context, schema.context_profile)
        score = clamp(0.40 * role_score + 0.30 * event_score
                      + 0.20 * goal_score + 0.10 * context_score)
        ranked.append({"schema_id": schema.schema_id, "score": score,
                       "components": {"roles": role_score, "events": event_score,
                                      "goals": goal_score, "context": context_score}})
    return sorted(ranked, key=lambda item: (-item["score"], item["schema_id"]))


def compose_schema_hierarchy(
    schemas: Sequence[Any], similarity_threshold: float
) -> dict[str, dict[str, tuple[str, ...]]]:
    """Link narrower frames to broader slot-containing frames without introducing cycles."""
    relationships = {
        schema.schema_id: {"parents": set(), "children": set(), "components": set()}
        for schema in schemas
    }
    for index, left in enumerate(schemas):
        left_roles = set(left.slots)
        for right in schemas[index + 1:]:
            right_roles = set(right.slots)
            shared = left_roles & right_roles
            union = left_roles | right_roles
            similarity = len(shared) / len(union) if union else 1.0
            if not shared or similarity < similarity_threshold:
                continue
            relationships[left.schema_id]["components"].update(shared)
            relationships[right.schema_id]["components"].update(shared)
            if left_roles < right_roles:
                relationships[left.schema_id]["parents"].add(right.schema_id)
                relationships[right.schema_id]["children"].add(left.schema_id)
            elif right_roles < left_roles:
                relationships[right.schema_id]["parents"].add(left.schema_id)
                relationships[left.schema_id]["children"].add(right.schema_id)
    return {
        schema_id: {key: tuple(sorted(values)) for key, values in row.items()}
        for schema_id, row in relationships.items()
    }


def association_strength_delta(previous: float, coactivation: float, learning_rate: float) -> float:
    """Apply a bounded Hebbian-style error update to one cortical association strength."""
    old = clamp(previous)
    target = clamp(coactivation)
    rate = clamp(learning_rate)
    return clamp(old + rate * (target - old))


def causal_evidence_contrast(
    p_outcome_given_relation: float, p_outcome_without_relation: float
) -> float:
    """Compute a predictive association contrast without interpreting it as causal proof."""
    return clamp(p_outcome_given_relation) - clamp(p_outcome_without_relation)
