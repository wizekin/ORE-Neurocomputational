"""Functional operations for Point 12 context-dependent interpretation."""
from __future__ import annotations

from datetime import datetime, timezone
from math import exp, log
import re
from typing import Any, Mapping, Sequence

from .contracts import (
    CandidateHypothesis, ContextEvidence, ContextInterpretationConfig, ContextState,
    InputEvent, ScoredCandidate, SituationModel,
)


def tokenize(text: str) -> tuple[str, ...]:
    """Normalize a text string into a stable tuple of lower-case lexical cues."""
    return tuple(re.findall(r"[\w'-]+", str(text).lower(), flags=re.UNICODE))


def similarity(left: Sequence[str], right: Sequence[str]) -> float:
    """Compute Jaccard similarity between two cue sets, returning zero for empty evidence."""
    a, b = set(left), set(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _evidence_tokens(item: ContextEvidence) -> tuple[str, ...]:
    """Collect lexical cues from context text, concepts, and fact keys and values."""
    values = [item.content, *item.concepts, *item.facts.keys()]
    values.extend(str(value) for value in item.facts.values())
    return tokenize(" ".join(values))


def _age_hours(timestamp: str) -> float:
    """Estimate non-negative age in hours, using UTC when a timestamp is naive."""
    if not timestamp:
        return 0.0
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)).total_seconds() / 3600.0)
    except (TypeError, ValueError):
        return 0.0


def retrieve_context(
    event: InputEvent,
    state: ContextState,
    config: ContextInterpretationConfig,
) -> dict[str, tuple[dict[str, Any], ...]]:
    """Retrieve goal-relevant discourse, knowledge, scene, convention, and speaker records."""
    query = tokenize(event.content + " " + " ".join(event.scene_cues)
                     + " " + " ".join(str(v) for v in event.features.values()))
    goal = (event.goal or state.current_goal).lower()
    sources = {
        "discourse": state.discourse_history,
        "knowledge": state.long_term_knowledge,
        "scene": state.scene_evidence,
        "convention": state.conventions,
        "speaker": state.speaker_models,
    }
    results: dict[str, tuple[dict[str, Any], ...]] = {}
    for source_type, records in sources.items():
        ranked: list[dict[str, Any]] = []
        for item in records:
            cue_score = similarity(query, _evidence_tokens(item))
            if not query:
                cue_score = 0.0
            goal_score = 1.0 if not item.goal_tags or not goal else (
                1.0 if any(tag in goal or goal in tag for tag in item.goal_tags) else 0.35
            )
            speaker_score = (
                1.0 if not event.speaker_id or not item.speaker_id
                or event.speaker_id == item.speaker_id else 0.45
            )
            topic_score = (
                1.0 if not state.current_topic_id or not item.topic_id
                or state.current_topic_id == item.topic_id else 0.25
            )
            recency = exp(-log(2.0) * _age_hours(item.timestamp)
                          / config.context_recency_half_life)
            if item.source_type == "discourse" and topic_score < 0.5:
                recency *= topic_score
            base = state.context_weights.get(source_type, 1.0)
            goal_weight = state.goal_source_preferences.get(source_type, 1.0)
            retrieval_score = (
                (0.15 + 0.85 * cue_score) * item.confidence * item.reliability
                * goal_score * speaker_score * recency * base * goal_weight
            )
            if retrieval_score >= config.context_match_threshold:
                ranked.append({
                    "evidence": item, "evidence_id": item.evidence_id,
                    "source_type": source_type, "similarity": cue_score,
                    "goal_fit": goal_score, "recency": recency,
                    "retrieval_score": retrieval_score,
                })
        ranked.sort(key=lambda row: (-row["retrieval_score"], row["evidence_id"]))
        results[source_type] = tuple(ranked[:config.maximum_context_items_per_source])
    return results


def select_goal(event: InputEvent, state: ContextState) -> tuple[str, str]:
    """Choose an explicit event or maintained task goal without inferring personal intent."""
    if event.goal.strip():
        return event.goal.strip(), "explicit_event_goal"
    if state.current_goal.strip():
        return state.current_goal.strip(), "maintained_context_goal"
    suggested = event.features.get("task_goal")
    if isinstance(suggested, str) and suggested.strip():
        return suggested.strip(), "event_feature_goal"
    return "general_interpretation", "default_goal"


def generate_candidates(
    event: InputEvent,
    retrieved: Mapping[str, Sequence[Mapping[str, Any]]],
    config: ContextInterpretationConfig,
) -> tuple[CandidateHypothesis, ...]:
    """Generate explicit candidates, context-suggested alternatives, and an open-set fallback."""
    candidates: list[CandidateHypothesis] = list(event.candidates)
    seen = {item.candidate_id for item in candidates}
    for rows in retrieved.values():
        for row in rows:
            item = row["evidence"]
            proposed = item.facts.get("candidate_meanings", ())
            if isinstance(proposed, str):
                proposed = (proposed,)
            single = item.facts.get("candidate_meaning")
            if single:
                proposed = tuple(proposed) + (str(single),)
            for index, meaning in enumerate(proposed or ()):
                text = str(meaning).strip()
                candidate_id = "context:" + re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
                if text and candidate_id not in seen:
                    candidates.append(CandidateHypothesis(
                        candidate_id=candidate_id, meaning=text,
                        cue_terms=tuple(tokenize(text)),
                        assumptions=("candidate generated from contextual evidence",),
                    ))
                    seen.add(candidate_id)
    if not candidates:
        return (CandidateHypothesis(
            candidate_id="general", meaning="general interpretation",
            cue_terms=tuple(tokenize(event.content)),
        ),)
    if config.unknown_candidate_id not in seen:
        candidates.append(CandidateHypothesis(
            candidate_id=config.unknown_candidate_id,
            meaning="other or not-yet-considered interpretation",
            assumptions=("open-set alternative retained to reduce forced interpretation",),
        ))
    return tuple(candidates)


def estimate_context_priors(
    candidates: Sequence[CandidateHypothesis],
    retrieved: Mapping[str, Sequence[Mapping[str, Any]]],
    goal: str,
    temperature: float = 1.0,
) -> tuple[dict[str, float], dict[str, float], dict[str, tuple[str, ...]]]:
    """Estimate normalized contextual priors while retaining which evidence contributed."""
    source_weights = {
        "discourse": 1.15, "knowledge": 1.0, "scene": 1.10,
        "convention": 0.75, "speaker": 0.80,
    }
    raw_scores: dict[str, float] = {}
    support: dict[str, tuple[str, ...]] = {}
    for candidate in candidates:
        cue_terms = set(candidate.cue_terms) | set(tokenize(candidate.meaning))
        weighted_match = 0.0
        total_weight = 0.0
        evidence_ids: list[str] = []
        for source_type, rows in retrieved.items():
            for row in rows:
                item = row["evidence"]
                evidence_weight = (
                    float(row["retrieval_score"]) * source_weights.get(source_type, 1.0)
                )
                if not item.goal_tags or not goal:
                    goal_multiplier = 1.0
                else:
                    goal_multiplier = 1.0 if any(
                        tag in goal.lower() or goal.lower() in tag for tag in item.goal_tags
                    ) else 0.35
                item_cues = set(_evidence_tokens(item))
                match = len(cue_terms & item_cues) / max(1, len(cue_terms))
                if candidate.meaning.lower() in item.content.lower():
                    match = max(match, 1.0)
                if match > 0:
                    evidence_ids.append(item.evidence_id)
                weighted_match += evidence_weight * goal_multiplier * match
                total_weight += evidence_weight
        contextual_fit = weighted_match / total_weight if total_weight else 0.0
        hint = candidate.prior_hint if candidate.prior_hint is not None else 0.5
        raw_scores[candidate.candidate_id] = (
            1.4 * contextual_fit + 0.25 * (hint - 0.5)
        )
        support[candidate.candidate_id] = tuple(sorted(set(evidence_ids)))
    priors = softmax(raw_scores, temperature)
    return priors, raw_scores, support


def score_input_evidence(
    event: InputEvent,
    candidates: Sequence[CandidateHypothesis],
) -> tuple[dict[str, float], dict[str, float]]:
    """Score direct event support separately from context and return likelihoods and cue scores."""
    event_cues = set(tokenize(event.content))
    event_cues.update(tokenize(" ".join(event.scene_cues)))
    event_cues.update(tokenize(" ".join(str(v) for v in event.features.values())))
    event_cues.update(tokenize(" ".join(str(v) for v in event.explicit_facts.values())))
    likelihoods: dict[str, float] = {}
    scores: dict[str, float] = {}
    for candidate in candidates:
        if candidate.candidate_id in event.candidate_evidence:
            score = event.candidate_evidence[candidate.candidate_id]
        else:
            cues = set(candidate.cue_terms)
            score = (len(cues & event_cues) / len(cues)) if cues else 0.5
            if not cues:
                score = 0.5
        score = min(1.0, max(0.0, score))
        # A neutral floor prevents absence of a cue from becoming proof of falsity.
        likelihoods[candidate.candidate_id] = 0.25 + 0.75 * score
        scores[candidate.candidate_id] = score
    return likelihoods, scores


def softmax(scores: Mapping[str, float], temperature: float = 1.0) -> dict[str, float]:
    """Normalize arbitrary finite candidate scores into a stable probability distribution."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not scores:
        return {}
    peak = max(float(value) for value in scores.values())
    values = {key: exp((float(value) - peak) / temperature)
              for key, value in scores.items()}
    total = sum(values.values())
    if not total or not total < float("inf"):
        raise ValueError("Candidate scores must produce a finite positive normalization")
    return {key: value / total for key, value in values.items()}


def bayesian_update(
    priors: Mapping[str, float],
    likelihoods: Mapping[str, float],
    minimum_probability: float = 1e-6,
) -> dict[str, float]:
    """Combine contextual priors and direct-input likelihoods by normalized Bayes weighting."""
    if set(priors) != set(likelihoods):
        raise ValueError("Priors and likelihoods must contain the same candidate identifiers")
    products = {
        key: max(minimum_probability, float(priors[key]))
        * max(minimum_probability, float(likelihoods[key]))
        for key in priors
    }
    total = sum(products.values())
    if total <= 0:
        raise ValueError("Bayesian update has no positive mass")
    return {key: value / total for key, value in products.items()}


def entropy(probabilities: Mapping[str, float]) -> float:
    """Calculate Shannon entropy in nats for a normalized non-negative distribution."""
    return -sum(float(p) * log(float(p)) for p in probabilities.values() if p > 0.0)


def resolve_competition(
    scored: Sequence[ScoredCandidate],
    config: ContextInterpretationConfig,
) -> tuple[str, ScoredCandidate | None, float, float]:
    """Select only when posterior, margin, and normalized-entropy criteria agree."""
    if not scored:
        return "unresolved", None, 0.0, 0.0
    ranked = sorted(scored, key=lambda row: (-row.posterior, row.candidate.candidate_id))
    top = ranked[0]
    second = ranked[1].posterior if len(ranked) > 1 else 0.0
    margin = max(0.0, top.posterior - second)
    uncertainty = entropy({item.candidate.candidate_id: item.posterior for item in ranked})
    normalized = uncertainty / log(len(ranked)) if len(ranked) > 1 else 0.0
    if (
        top.posterior >= config.selection_threshold
        and margin >= config.minimum_posterior_margin
        and normalized <= config.maximum_normalized_entropy
        and top.candidate.candidate_id != config.unknown_candidate_id
    ):
        return "selected", top, uncertainty, normalized
    return "ambiguous", None, uncertainty, normalized


def detect_conflicts(
    situation: SituationModel,
    event: InputEvent,
) -> tuple[dict[str, Any], ...]:
    """Find direct contradictions between new explicit facts and the maintained situation model."""
    conflicts: list[dict[str, Any]] = []
    for key, incoming in event.explicit_facts.items():
        if key not in situation.facts or situation.facts[key] == incoming:
            continue
        prior = situation.fact_provenance.get(key, {})
        conflicts.append({
            "key": str(key), "previous_value": situation.facts[key],
            "incoming_value": incoming, "previous_provenance": prior,
            "incoming_provenance": "explicit_input",
            "conflict_type": "value_mismatch",
        })
    return tuple(conflicts)


def revise_candidates(
    scored: Sequence[ScoredCandidate],
    conflicts: Sequence[Mapping[str, Any]],
    config: ContextInterpretationConfig,
) -> tuple[ScoredCandidate, ...]:
    """Reweight candidates against new explicit facts without deleting the rejected alternatives."""
    if not scored or not conflicts:
        return tuple(scored)
    penalties: dict[str, float] = {}
    for item in scored:
        penalty = 0.0
        for conflict in conflicts:
            key = str(conflict["key"])
            if key in item.candidate.facts and item.candidate.facts[key] != conflict["incoming_value"]:
                penalty += config.conflict_penalty
        penalties[item.candidate.candidate_id] = penalty
    raw = {
        item.candidate.candidate_id:
            log(max(1e-12, item.posterior)) - penalties[item.candidate.candidate_id]
        for item in scored
    }
    revised_probabilities = softmax(raw)
    return tuple(ScoredCandidate(
        candidate=item.candidate, context_prior=item.context_prior,
        input_likelihood=item.input_likelihood,
        posterior=revised_probabilities[item.candidate.candidate_id],
        context_score=item.context_score, input_score=item.input_score,
        evidence_ids=item.evidence_ids,
        conflict_penalty=penalties[item.candidate.candidate_id],
    ) for item in scored)


def update_situation_model(
    situation: SituationModel,
    event: InputEvent,
    status: str,
    selected: ScoredCandidate | None,
    uncertainty: float,
    conflicts: Sequence[Mapping[str, Any]],
    candidates: Sequence[ScoredCandidate],
) -> SituationModel:
    """Integrate explicit claims and supported interpretations while preserving unresolved alternatives."""
    for key, value in event.explicit_facts.items():
        situation.facts[str(key)] = value
        situation.fact_provenance[str(key)] = {
            "status": "stated_or_observed", "provenance": "explicit_input",
            "event_id": event.event_id, "confidence": event.reliability,
        }
    event_record = {
        "event_id": event.event_id, "content": event.content,
        "status": status, "selected_meaning": (
            selected.candidate.meaning if selected is not None else None
        ), "confidence": selected.posterior if selected is not None else None,
        "uncertainty": uncertainty,
    }
    situation.events.append(event_record)
    situation.conflicts.extend(dict(item) for item in conflicts)
    if selected is not None:
        for key, value in selected.candidate.facts.items():
            if key not in situation.facts:
                situation.facts[str(key)] = value
                situation.fact_provenance[str(key)] = {
                    "status": "inferred", "provenance": "candidate_hypothesis",
                    "event_id": event.event_id, "candidate_id": selected.candidate.candidate_id,
                    "confidence": selected.posterior,
                }
        situation.confidence = selected.posterior
    else:
        situation.unresolved_items.append({
            "event_id": event.event_id,
            "candidate_ids": [item.candidate.candidate_id for item in candidates],
            "candidate_posteriors": {
                item.candidate.candidate_id: item.posterior for item in candidates
            },
            "reason": "multiple or low-confidence interpretations remain",
        })
        situation.confidence = max((item.posterior for item in candidates), default=0.0)
    return situation


def adapt_context_weights(
    state: ContextState,
    selected: ScoredCandidate | None,
    retrieved: Mapping[str, Sequence[Mapping[str, Any]]],
    rate: float,
) -> dict[str, float]:
    """Make a small bounded adjustment to context-source weights after a confident interpretation."""
    if selected is None:
        return dict(state.context_weights)
    cue_terms = set(selected.candidate.cue_terms) | set(tokenize(selected.candidate.meaning))
    updated: dict[str, float] = {}
    for source_type, current in state.context_weights.items():
        rows = retrieved.get(source_type, ())
        support = 0.0
        if rows:
            supports = []
            for row in rows:
                evidence = row["evidence"]
                overlap = similarity(tuple(cue_terms), _evidence_tokens(evidence))
                supports.append(overlap * float(row["retrieval_score"]))
            support = max(supports, default=0.0)
        value = min(1.0, max(0.0, float(current) + rate * (support - 0.35)))
        updated[source_type] = value
    state.context_weights.update(updated)
    return updated


def build_evidence_ledger(
    event: InputEvent,
    retrieved: Mapping[str, Sequence[Mapping[str, Any]]],
    candidates: Sequence[ScoredCandidate],
) -> tuple[dict[str, Any], ...]:
    """Create a provenance ledger for the input, retrieved context, and candidate inferences."""
    rows: list[dict[str, Any]] = [{
        "evidence_id": "input:" + event.event_id, "source_type": "input",
        "content": event.content, "status": "explicit_input",
        "provenance": "current_event", "reliability": event.reliability,
    }]
    for source_type, items in retrieved.items():
        for row in items:
            item = row["evidence"]
            rows.append({
                "evidence_id": item.evidence_id, "source_type": source_type,
                "content": item.content, "status": "retrieved_context",
                "provenance": item.provenance,
                "confidence": item.confidence, "reliability": item.reliability,
                "retrieval_score": row["retrieval_score"],
            })
    for candidate in candidates:
        rows.append({
            "evidence_id": "candidate:" + candidate.candidate.candidate_id,
            "source_type": "candidate", "content": candidate.candidate.meaning,
            "status": "hypothesis", "provenance": "candidate_generation",
            "posterior": candidate.posterior,
            "assumptions": list(candidate.candidate.assumptions),
            "supporting_evidence_ids": list(candidate.evidence_ids),
        })
    return tuple(rows)
