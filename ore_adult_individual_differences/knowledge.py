"""Knowledge retrieval, expertise chunking, language/cultural cues, and evidence-safe learning."""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .contracts import AdultComprehensionExperience, AdultDifferencesConfig, AdultLearnerState
from .functional import clamp, jaccard_similarity, softmax, token_set


def retrieve_schema_candidates(
    event: AdultComprehensionExperience,
    units: Mapping[str, Any],
    goal: Mapping[str, Any],
    learner: AdultLearnerState,
    config: AdultDifferencesConfig,
) -> dict[str, Any]:
    """Rank domain-specific schemas and select only those above a schema-match threshold."""
    candidates = learner.prior_knowledge.get(event.domain, [])
    observed = [row["content"] for row in units.get("units", ())]
    active_features = token_set(observed)
    expertise = learner.expertise_by_domain.get(event.domain, 0.0)
    ranked: list[dict[str, Any]] = []
    for index, raw in enumerate(candidates):
        if not isinstance(raw, Mapping):
            continue
        schema_id = str(raw.get("schema_id", raw.get("id", raw.get("schema", f"schema-{index}"))))
        schema = raw.get("schema", raw.get("description", schema_id))
        features = token_set(raw.get("features", ()))
        overlap = len(active_features & features) / max(1, len(active_features | features))
        reliability = clamp(float(raw.get("reliability", 0.5)))
        goal_fit = clamp(float(raw.get("goal_fit", 0.5)))
        conflict = clamp(float(raw.get("conflict", 0.0)))
        recency = clamp(float(raw.get("recency", 0.5)))
        domain_match = 1.0 if str(raw.get("domain", event.domain)).lower() == event.domain else 0.0
        score = (
            0.34 * overlap + 0.22 * goal_fit + 0.22 * reliability
            + 0.10 * recency + 0.08 * expertise + 0.04 * domain_match
            - 0.28 * conflict
        )
        ranked.append({
            "schema_id": schema_id, "schema": schema, "score": score,
            "similarity": overlap, "goal_fit": goal_fit, "reliability": reliability,
            "recency": recency, "conflict": conflict, "domain_match": bool(domain_match),
            "expertise": expertise, "features": sorted(features),
            "chunks": list(raw.get("chunks", ())),
            "exceptions": list(raw.get("exceptions", ())),
            "relations": list(raw.get("relations", ())),
            "source_id": str(raw.get("source_id", "")),
        })
    ranked.sort(key=lambda row: row["score"], reverse=True)
    probabilities = softmax([row["score"] for row in ranked], temperature=0.35)
    for row, probability in zip(ranked, probabilities):
        row["probability"] = probability
    selected = (
        ranked[0] if ranked and float(ranked[0]["score"]) >= config.schema_match_threshold
        else None
    )
    return {
        "ranked_schemas": ranked[:8],
        "selected_schema": selected,
        "schema_match_threshold": config.schema_match_threshold,
        "schema_uncertainty_nats": -sum(
            item["probability"] * math.log(item["probability"])
            for item in ranked if item["probability"] > 0.0
        ),
        "domain_scope": event.domain,
        "expertise_is_domain_specific": True,
        "similarity_is_not_truth": True,
    }


def convert_familiar_details_to_chunks(
    units: Mapping[str, Any],
    retrieved: Mapping[str, Any],
    config: AdultDifferencesConfig,
) -> dict[str, Any]:
    """Compress familiar unit groups only when schema fit and cost advantage justify it."""
    unit_rows = list(units.get("units", ()))
    separate_cost = sum(max(0.1, float(row.get("cost", 1.0))) for row in unit_rows)
    selected_schema = retrieved.get("selected_schema")
    chunks: list[dict[str, Any]] = []
    consumed: set[str] = set()
    if selected_schema:
        schema_confidence = clamp(
            float(selected_schema.get("score", 0.0))
            * float(selected_schema.get("probability", 0.0))
        )
        for index, raw_chunk in enumerate(selected_schema.get("chunks", ())):
            if not isinstance(raw_chunk, Mapping):
                continue
            required = token_set(raw_chunk.get("features", raw_chunk.get("members", ())))
            members = [
                row for row in unit_rows
                if token_set(row.get("content", "")) & required
                and row["unit_id"] not in consumed
            ]
            matched_features = set().union(
                *(token_set(row.get("content", "")) for row in members)
            ) if members else set()
            fit = len(matched_features & required) / max(1, len(required))
            exceptions = list(raw_chunk.get("exceptions", ()))
            exception_cost = float(raw_chunk.get("exception_cost", 0.0)) + 0.5 * len(exceptions)
            schema_cost = max(0.2, float(raw_chunk.get("schema_cost", 1.0)))
            chunk_cost = schema_cost + exception_cost
            member_cost = sum(max(0.1, float(row.get("cost", 1.0))) for row in members)
            confidence = clamp(fit * schema_confidence)
            if (
                len(members) >= 2
                and confidence >= config.chunk_confidence_threshold
                and chunk_cost < member_cost
            ):
                ids = [row["unit_id"] for row in members]
                consumed.update(ids)
                chunks.append({
                    "chunk_id": str(raw_chunk.get("chunk_id", raw_chunk.get("name", f"chunk-{index}"))),
                    "label": str(raw_chunk.get("label", raw_chunk.get("name", "familiar-schema"))),
                    "member_unit_ids": ids,
                    "members": [row["content"] for row in members],
                    "confidence": confidence, "match": fit,
                    "cost_separate": member_cost, "cost_chunk": chunk_cost,
                    "exceptions_retained": exceptions,
                    "schema_id": selected_schema["schema_id"],
                })
    remaining = [row for row in unit_rows if row["unit_id"] not in consumed]
    chunk_cost = sum(float(row["cost_chunk"]) for row in chunks) + sum(
        max(0.1, float(row.get("cost", 1.0))) for row in remaining
    )
    return {
        "chunks": chunks,
        "uncompressed_units": remaining,
        "separate_representation_cost": separate_cost,
        "compressed_representation_cost": chunk_cost,
        "cost_reduction": max(0.0, separate_cost - chunk_cost),
        "effective_unit_count": len(chunks) + len(remaining),
        "original_unit_count": len(unit_rows),
        "exceptions_and_novel_details_preserved": True,
        "chunking_is_conditional_not_automatic": True,
    }


def weight_language_cues(
    event: AdultComprehensionExperience,
    learner: AdultLearnerState,
) -> dict[str, Any]:
    """Compute proficiency-sensitive candidate posteriors from learned language cue reliability."""
    language = event.language
    proficiency = learner.language_proficiency.get(language, 0.5)
    cue_reliability = learner.language_cue_reliability.get(language, {})
    hypotheses = list(event.interpretation_candidates)
    if not hypotheses:
        hypotheses = [{
            "hypothesis_id": "evidence-preserving-input",
            "label": "retain literal input pending interpretation",
            "prior": 1.0,
            "cue_likelihoods": {},
        }]
    log_scores: list[float] = []
    rows: list[dict[str, Any]] = []
    for index, hypothesis in enumerate(hypotheses):
        prior = max(1e-6, float(hypothesis.get("prior", 1.0 / len(hypotheses))))
        log_score = math.log(prior)
        effective_weights: dict[str, float] = {}
        cue_likelihoods = hypothesis.get("cue_likelihoods", {})
        for cue, observed in event.language_cues.items():
            base_reliability = clamp(float(cue_reliability.get(str(cue).lower(), 0.5)))
            if isinstance(observed, Mapping):
                observation_strength = clamp(float(observed.get("strength", observed.get("confidence", 0.5))))
            else:
                observation_strength = clamp(float(observed))
            # Neutral evidence (0.5) carries no cue-specific weight; strong observed
            # evidence is discounted by proficiency rather than treating a language
            # label as a binary cognitive advantage or disadvantage.
            evidence_strength = min(1.0, 2.0 * abs(observation_strength - 0.5))
            likelihood = clamp(float(cue_likelihoods.get(cue, 0.5)))
            adjusted = 0.5 + proficiency * (likelihood - 0.5)
            weight = base_reliability * (0.25 + 0.75 * proficiency) * evidence_strength
            log_score += weight * math.log(max(1e-6, adjusted))
            effective_weights[str(cue)] = weight
        rows.append({
            "hypothesis_id": str(hypothesis.get("hypothesis_id", f"hypothesis-{index}")),
            "label": hypothesis.get("label", hypothesis.get("hypothesis_id", f"hypothesis-{index}")),
            "prior": prior, "effective_cue_weights": effective_weights,
            "cue_likelihoods": dict(cue_likelihoods),
            "source": str(hypothesis.get("source", "candidate_model")),
        })
        log_scores.append(log_score)
    peak = max(log_scores)
    unnormalized = [math.exp(max(-700.0, score - peak)) for score in log_scores]
    total = sum(unnormalized)
    for row, score in zip(rows, unnormalized):
        row["probability"] = score / total
    rows.sort(key=lambda item: item["probability"], reverse=True)
    return {
        "language": language, "proficiency_estimate": proficiency,
        "cue_reliability": dict(cue_reliability),
        "ranked_interpretations": rows,
        "cue_weights_are_experience_specific": True,
        "no_universal_bilingual_advantage_or_disadvantage_assumed": True,
    }


def incorporate_cultural_and_lived_experience(
    event: AdultComprehensionExperience,
    learner: AdultLearnerState,
) -> dict[str, Any]:
    """Retrieve individual contextual conventions without demographic or group stereotypes."""
    cues = token_set(event.cultural_cues)
    cues |= {
        str(key).lower() for key, value in event.context_observations.items()
        if float(value) >= 0.5
    }
    cues |= token_set(event.metadata.get("demonstrated_context_cues", ()))
    conventions: dict[str, Mapping[str, Any]] = {}
    for context_id, learned in learner.cultural_experience.items():
        learned_conventions = learned.get("conventions", {})
        if isinstance(learned_conventions, Mapping):
            for convention_id, record in learned_conventions.items():
                if isinstance(record, Mapping):
                    conventions[str(convention_id)] = record
    for convention_id, record in event.cultural_conventions.items():
        if isinstance(record, Mapping):
            # Explicit current-task evidence overrides a stored default for this event only.
            conventions[str(convention_id)] = record
    rows: list[dict[str, Any]] = []
    for convention_id, record in conventions.items():
        required = token_set(record.get("cues", record.get("features", ())))
        overlap = len(cues & required) / max(1, len(cues | required))
        context_fit = clamp(float(record.get("context_fit", overlap)))
        reliability = clamp(float(record.get("reliability", 0.5)))
        applicability = clamp(float(record.get("applicability", 0.5)))
        score = 0.50 * overlap + 0.25 * context_fit + 0.25 * reliability * applicability
        rows.append({
            "context_id": convention_id,
            "interpretation": record.get("interpretation", record.get("meaning", "")),
            "matched_cues": sorted(cues & required),
            "score": score, "reliability": reliability,
            "source": str(record.get("source", "individual_experience")),
            "explicitly_demonstrated": bool(record.get("explicitly_demonstrated", False)),
        })
    probabilities = softmax([row["score"] for row in rows], temperature=0.4)
    for row, probability in zip(rows, probabilities):
        row["probability"] = probability
    rows.sort(key=lambda item: item["probability"], reverse=True)
    uncertainty = -sum(
        row["probability"] * math.log(row["probability"])
        for row in rows if row["probability"] > 0.0
    )
    return {
        "context_hypotheses": rows[:8],
        "context_uncertainty_nats": uncertainty,
        "context_cues_used": sorted(cues),
        "individual_context_not_group_label": True,
        "stereotype_based_inference_used": False,
        "unfamiliar_convention_requests_clarification": not bool(rows),
    }


def integrate_knowledge_with_provenance(
    units: Mapping[str, Any],
    retrieved: Mapping[str, Any],
    language: Mapping[str, Any],
    culture: Mapping[str, Any],
) -> dict[str, Any]:
    """Compose an active situation model while keeping observed and inferred content separate."""
    observed = [
        {
            "content": row["content"], "source_id": row["source_id"],
            "confidence": row["confidence"], "provenance": row["provenance"],
        }
        for row in units.get("units", ())
    ]
    retrieved_items = [
        {
            "schema_id": row["schema_id"], "schema": row["schema"],
            "confidence": row["probability"] * row["reliability"],
            "provenance": "retrieved_long_term_knowledge", "source_id": row["source_id"],
        }
        for row in retrieved.get("ranked_schemas", ())[:3]
    ]
    language_hypotheses = [
        {
            "hypothesis_id": row["hypothesis_id"], "interpretation": row["label"],
            "confidence": row["probability"], "provenance": "language_cue_inference",
        }
        for row in language.get("ranked_interpretations", ())[:5]
    ]
    cultural_hypotheses = [
        {
            "context_id": row["context_id"], "interpretation": row["interpretation"],
            "confidence": row["probability"], "provenance": "individual_context_inference",
        }
        for row in culture.get("context_hypotheses", ())[:5]
    ]
    inferred = language_hypotheses + cultural_hypotheses
    return {
        "observed": observed,
        "retrieved": retrieved_items,
        "inferred": inferred,
        "uncertain": [
            row for row in inferred if float(row["confidence"]) < 0.65
        ],
        "active_schema": retrieved_items[0] if retrieved_items else None,
        "situation_model": {
            "observed_contents": [row["content"] for row in observed],
            "active_schema": retrieved_items[0]["schema"] if retrieved_items else None,
            "candidate_interpretations": language_hypotheses,
            "contextual_interpretations": cultural_hypotheses,
        },
        "provenance_map": {
            "observed_content": "input_source",
            "retrieved_knowledge": "learner_prior_knowledge",
            "language_inference": "language_cue_model",
            "cultural_inference": "individual_lived_experience",
        },
        "source_and_inference_boundaries_preserved": True,
    }


def update_knowledge_without_false_assimilation(
    event: AdultComprehensionExperience,
    state: AdultLearnerState,
    config: AdultDifferencesConfig,
) -> dict[str, Any]:
    """Update knowledge cautiously and retain exceptions, contradictions, and episode sources."""
    domain = event.domain
    estimates = state.knowledge_estimates.setdefault(domain, {})
    updates: list[dict[str, Any]] = []
    exceptions: list[dict[str, Any]] = []
    contradictions: list[dict[str, Any]] = []
    for index, fact in enumerate(event.new_facts):
        fact_id = str(fact.get("fact_id", fact.get("key", f"fact-{index}")))
        value = fact.get("value")
        reliability = clamp(float(fact.get("reliability", event.evidence_quality)))
        repetitions = max(1, int(fact.get("repetitions", 1)))
        novelty = clamp(float(fact.get("novelty", 0.5)))
        source_id = str(fact.get("source_id", event.experience_id))
        previous = estimates.get(fact_id)
        if previous is None:
            previous = {
                "value": value, "confidence": reliability * min(1.0, repetitions / 2.0),
                "value_counts": {repr(value): repetitions},
                "value_examples": {repr(value): value},
                "sources": [source_id], "exceptions": [],
                "update_count": 0,
            }
            estimates[fact_id] = previous
            updates.append({
                "fact_id": fact_id, "classification": "new_observation",
                "value": value, "reliability": reliability,
                "stored_provisionally": True,
            })
        else:
            prior_value = previous.get("value")
            conflict = prior_value != value
            expected = fact.get("expected_value", prior_value)
            expected_conflict = expected != value
            if conflict:
                if reliability < 0.45:
                    classification = "possible_input_error"
                elif novelty >= 0.70 and repetitions >= 2:
                    classification = "novel_exception"
                else:
                    classification = "unresolved_contradiction"
                record = {
                    "fact_id": fact_id, "prior_value": prior_value, "new_value": value,
                    "reliability": reliability, "repetitions": repetitions,
                    "novelty": novelty, "source_id": source_id,
                    "classification": classification,
                }
                if classification == "novel_exception":
                    previous.setdefault("exceptions", []).append(record)
                    exceptions.append(record)
                elif classification == "unresolved_contradiction":
                    contradictions.append(record)
                updates.append({
                    "fact_id": fact_id, "classification": classification,
                    "prior_preserved": True, "new_evidence_recorded": True,
                })
            value_key = repr(value)
            counts = previous.setdefault("value_counts", {})
            counts[value_key] = int(counts.get(value_key, 0)) + repetitions
            previous.setdefault("value_examples", {})[value_key] = value
            if source_id not in previous.setdefault("sources", []):
                previous["sources"].append(source_id)
            best_key = max(counts, key=counts.get)
            dominant_support = counts[best_key] / max(1, sum(counts.values()))
            enough_evidence = repetitions >= 2 or reliability >= 0.90
            if enough_evidence and dominant_support >= 0.70:
                target_value = previous["value_examples"][best_key]
                if isinstance(prior_value, (int, float)) and isinstance(target_value, (int, float)):
                    alpha = (
                        config.semantic_update_rate * reliability
                        * min(1.0, repetitions / 3.0) * (0.5 + 0.5 * novelty)
                    )
                    previous["value"] = float(prior_value) + alpha * (
                        float(target_value) - float(prior_value)
                    )
                else:
                    previous["value"] = target_value
                previous["confidence"] = clamp(
                    float(previous.get("confidence", 0.5))
                    + config.semantic_update_rate * (reliability - float(previous.get("confidence", 0.5)))
                )
                previous["update_count"] = int(previous.get("update_count", 0)) + 1
            updates.append({
                "fact_id": fact_id, "classification": "repeated_evidence_update" if enough_evidence else "evidence_accumulated",
                "expected_value_conflict": bool(expected_conflict),
                "dominant_value_support": dominant_support,
                "current_value": previous.get("value"),
                "prior_value_retained_until_repeated_support": not enough_evidence,
            })
        state.episodic_records.append({
            "experience_id": event.experience_id, "domain": domain,
            "fact_id": fact_id, "value": value, "source_id": source_id,
            "reliability": reliability, "classification": updates[-1]["classification"],
        })
    state.episodic_records = state.episodic_records[-500:]
    return {
        "updates": updates, "exceptions": exceptions, "contradictions": contradictions,
        "knowledge_estimates": dict(estimates),
        "episodic_record_count": len(state.episodic_records),
        "source_provenance_preserved": True,
        "false_assimilation_prevented_by_conflict_retention": True,
    }
