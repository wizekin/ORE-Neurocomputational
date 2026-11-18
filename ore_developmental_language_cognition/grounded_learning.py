"""Grounded word learning, conceptual hierarchy, and memory-stabilization stages."""
from __future__ import annotations

from typing import Any, Mapping

from .contracts import DevelopmentalConfig, DevelopmentalExperience, DevelopmentalLearnerState
from .functional import clamp, cosine_similarity, entropy, softmax, vectorize, mean_vector


def map_word_forms_to_meanings(
    event: DevelopmentalExperience,
    segmentation: Mapping[str, Any],
    multimodal: Mapping[str, Any],
    state: DevelopmentalLearnerState,
    config: DevelopmentalConfig,
) -> dict[str, Any]:
    """Map a candidate word form to uncertain scene referents and previously learned concepts."""
    word = event.target_word or (
        str(segmentation.get("candidate_word_forms", [""])[0])
        if segmentation.get("candidate_word_forms") else ""
    )
    candidates_by_meaning: dict[str, dict[str, Any]] = {}
    for referent in multimodal.get("referent_posterior", ()):
        meaning_id = str(referent["referent_id"])
        candidates_by_meaning[meaning_id] = {
            "meaning": meaning_id, "semantic_type": "entity",
            "source": "scene_grounding", "score": float(referent["probability"]),
        }
    known_entry = state.lexicon.get(word, {})
    known_meanings = list(known_entry.get("candidate_meanings", ()))
    syntactic_type = str(event.syntactic_hints.get("semantic_type", ""))
    for meaning in known_meanings:
        meaning_id = str(meaning)
        concept = state.concept_store.get(meaning_id, {})
        type_fit = (
            1.0 if not syntactic_type or syntactic_type == concept.get("semantic_type", syntactic_type)
            else 0.25
        )
        prior_score = 0.35 * type_fit + 0.65
        if meaning_id in candidates_by_meaning:
            row = candidates_by_meaning[meaning_id]
            row["score"] = min(2.0, float(row["score"]) + 0.30 * prior_score)
            row["source"] = "scene_grounding_plus_prior_lexicon"
        else:
            candidates_by_meaning[meaning_id] = {
                "meaning": meaning_id,
                "semantic_type": concept.get("semantic_type", "concept"),
                "source": "prior_lexical_concept_link",
                "score": prior_score,
            }
    candidates: list[dict[str, Any]] = list(candidates_by_meaning.values())
    if not candidates and word:
        candidates.append({
            "meaning": word, "semantic_type": syntactic_type or "unknown",
            "source": "provisional_form_only", "score": 0.5,
        })
    probabilities = softmax([item["score"] for item in candidates], config.multimodal_temperature)
    for item, probability in zip(candidates, probabilities):
        item["probability"] = probability
    candidates.sort(key=lambda item: item["probability"], reverse=True)
    return {
        "word_form": word, "candidate_meanings": candidates,
        "meaning_confidence": candidates[0]["probability"] if candidates else 0.0,
        "meaning_uncertainty_nats": entropy([x["probability"] for x in candidates]),
        "ambiguity_preserved": len(candidates) > 1 and candidates[0]["probability"] < 0.8,
    }


def accelerate_vocabulary_learning(
    event: DevelopmentalExperience,
    mapping: Mapping[str, Any],
    state: DevelopmentalLearnerState,
    config: DevelopmentalConfig,
) -> dict[str, Any]:
    """Use prior vocabulary, concepts, syntax, and social support to grow a provisional lexicon."""
    word = str(mapping.get("word_form", "")).lower()
    if not word:
        return {"updated": False, "reason": "no_candidate_word", "learning_efficiency": 0.0}
    entry = state.lexicon.setdefault(word, {
        "forms": [word], "exposures": 0, "contexts": [], "mean_confidence": 0.0,
        "status": "provisional", "semantic_type": event.syntactic_hints.get("semantic_type", "unknown"),
        "candidate_meanings": [],
    })
    old_exposures = int(entry.get("exposures", 0))
    confidence = float(mapping.get("meaning_confidence", 0.0))
    entry["exposures"] = old_exposures + 1
    entry["mean_confidence"] = (
        old_exposures * float(entry.get("mean_confidence", 0.0)) + confidence
    ) / (old_exposures + 1)
    context = " ".join(event.context_tokens[:12])
    if context and context not in entry["contexts"]:
        entry["contexts"] = (list(entry.get("contexts", ())) + [context])[-12:]
    entry["candidate_meanings"] = list(dict.fromkeys(
        list(entry.get("candidate_meanings", ()))
        + [str(x["meaning"]) for x in mapping.get("candidate_meanings", ())]
    ))[:20]
    lexical_coverage = clamp(len(state.lexicon) / (len(state.lexicon) + 12.0))
    concept_coverage = clamp(len(state.concept_store) / (len(state.concept_store) + 10.0))
    social_support = clamp(event.social_contingency)
    efficiency = clamp(
        0.35 + 0.30 * lexical_coverage + 0.25 * concept_coverage + 0.10 * social_support
    )
    entry["status"] = (
        "established" if entry["exposures"] >= config.lexical_establishment_exposures
        and entry["mean_confidence"] >= 0.55 else "provisional"
    )
    return {
        "updated": True, "word_form": word, "exposures": entry["exposures"],
        "status": entry["status"], "mean_confidence": entry["mean_confidence"],
        "learning_efficiency": efficiency, "lexical_coverage": lexical_coverage,
        "concept_coverage": concept_coverage, "social_support": social_support,
        "uncertainty_reduced_by_prior_knowledge": efficiency > 0.35,
    }


def build_grounded_concepts(
    event: DevelopmentalExperience,
    mapping: Mapping[str, Any],
    state: DevelopmentalLearnerState,
    config: DevelopmentalConfig,
) -> dict[str, Any]:
    """Form concepts from repeated sensory examples and retain hierarchy and feature stability."""
    examples = list(event.concept_examples)
    if not examples and event.scene_referents:
        examples = [item for item in event.scene_referents if item.get("label")]
    if not examples:
        return {
            "updated_concepts": [], "similarities": [],
            "concept_count": len(state.concept_store), "relations": list(state.concept_relations[-100:]),
        }
    updated: list[dict[str, Any]] = []
    query_vectors: list[list[float]] = []
    for example in examples:
        label = str(example.get("concept", example.get("label", ""))).strip().lower()
        if not label:
            continue
        features = vectorize(example.get("features", example.get("feature_vector", ())))
        if features:
            query_vectors.append(features)
        entry = state.concept_store.setdefault(label, {
            "prototype": features, "exposures": 0, "feature_counts": {},
            "parent": str(example.get("parent", "")),
            "semantic_type": str(example.get("semantic_type", "entity")),
            "relations": [], "stable_features": [], "incidental_features": [],
        })
        count = int(entry.get("exposures", 0))
        if features:
            prior = list(entry.get("prototype", features))
            width = min(len(prior), len(features))
            rate = config.concept_learning_rate / (1.0 + 0.05 * count)
            entry["prototype"] = [
                prior[index] + rate * (features[index] - prior[index]) for index in range(width)
            ]
        attributes = example.get("attributes", {})
        if isinstance(attributes, Mapping):
            for key, value in attributes.items():
                counts = entry["feature_counts"]
                counts[str(key)] = counts.get(str(key), 0) + int(bool(value))
        entry["exposures"] = count + 1
        relation = example.get("relation")
        if relation:
            row = {
                "source": label, "relation": str(relation),
                "target": str(example.get("related_to", "")),
            }
            if row not in entry["relations"]:
                entry["relations"].append(row)
            if row not in state.concept_relations:
                state.concept_relations.append(row)
        parent = str(example.get("parent", ""))
        if parent:
            entry["parent"] = parent
        total = max(1, int(entry["exposures"]))
        entry["stable_features"] = sorted(
            key for key, value in entry["feature_counts"].items() if value / total >= 0.7
        )
        entry["incidental_features"] = sorted(
            key for key, value in entry["feature_counts"].items() if value / total <= 0.3
        )
        updated.append({
            "concept": label, "exposures": entry["exposures"], "parent": entry["parent"],
            "semantic_type": entry["semantic_type"],
            "stable_features": list(entry["stable_features"]),
            "incidental_features": list(entry["incidental_features"]),
        })
    similarities: list[dict[str, Any]] = []
    query = mean_vector(query_vectors)
    if query:
        for label, entry in state.concept_store.items():
            score = cosine_similarity(query, entry.get("prototype", ()))
            similarities.append({
                "concept": label, "similarity": score, "parent": entry.get("parent", ""),
            })
        similarities.sort(key=lambda item: item["similarity"], reverse=True)
    return {
        "updated_concepts": updated, "similarities": similarities,
        "hierarchy": {label: entry.get("parent", "") for label, entry in state.concept_store.items()},
        "relations": list(state.concept_relations[-100:]),
        "concept_count": len(state.concept_store),
        "stable_features_are_experience_based": True,
    }


def stabilize_developmental_memory(
    event: DevelopmentalExperience,
    mapping: Mapping[str, Any],
    concepts: Mapping[str, Any],
    state: DevelopmentalLearnerState,
    config: DevelopmentalConfig,
) -> dict[str, Any]:
    """Bind a source-linked episode and consolidate recurring associations gradually."""
    word = str(mapping.get("word_form", "")).lower()
    meanings = [str(item["meaning"]) for item in mapping.get("candidate_meanings", ())[:3]]
    cue_tokens = set(word.split()) | {item.lower() for item in event.context_tokens}
    retrieved = []
    for episode in reversed(state.episodic_memory):
        episode_tokens = set(str(episode.get("word", "")).split())
        episode_tokens |= set(episode.get("context_tokens", ()))
        overlap = len(cue_tokens & episode_tokens) / max(1, len(cue_tokens | episode_tokens))
        if overlap > 0.0:
            retrieved.append({"episode_id": episode.get("experience_id"), "relevance": overlap})
        if len(retrieved) >= 5:
            break
    episode = {
        "experience_id": event.experience_id, "word": word, "meanings": meanings,
        "context_tokens": list(event.context_tokens), "speaker_intention": event.speaker_intention,
        "age_months": event.age_months,
        "grounded_concepts": list(concepts.get("updated_concepts", ())),
        "retrieval_count": 0, "source": "specific_experience",
    }
    state.episodic_memory.append(episode)
    state.episodic_memory = state.episodic_memory[-config.maximum_episodes:]
    association_updates: list[dict[str, Any]] = []
    for meaning in meanings:
        key = word + "->" + meaning
        previous = state.semantic_associations.get(key, 0.0)
        count = sum(
            1 for item in state.episodic_memory
            if item.get("word") == word and meaning in item.get("meanings", ())
        )
        target = clamp(count / 4.0)
        updated = clamp(previous + config.memory_learning_rate * (target - previous))
        state.semantic_associations[key] = updated
        association_updates.append({
            "association": key, "before": previous, "after": updated,
            "repeated_episode_count": count,
        })
    for item in retrieved:
        for episode_item in state.episodic_memory:
            if episode_item["experience_id"] == item["episode_id"]:
                episode_item["retrieval_count"] = int(episode_item.get("retrieval_count", 0)) + 1
                break
    return {
        "episode": episode, "retrieved_prior_episodes": retrieved,
        "association_updates": association_updates,
        "episodic_count": len(state.episodic_memory),
        "semantic_associations": dict(state.semantic_associations),
        "episodic_and_generalized_knowledge_separated": True,
    }
