"""Functional parsing, garden-path detection, reanalysis, and adaptation operations."""
from __future__ import annotations

from collections import defaultdict
from math import exp, log
import re
from typing import Any, Mapping, Sequence

from .contracts import (
    LexicalCandidate,
    ParseHypothesis,
    Point23Config,
    Point23State,
    SentenceRequest,
)


_EPS = 1e-12
_TOKEN_PATTERN = re.compile(r"\w+(?:['’]\w+)*|[^\w\s]", re.UNICODE)


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """Bound a finite scalar."""
    return max(lower, min(upper, float(value)))


def softmax(scores: Mapping[str, float], temperature: float = 1.0) -> dict[str, float]:
    """Convert log scores to a normalized distribution using stable exponentiation."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not scores:
        return {}
    scaled = {key: float(value) / temperature for key, value in scores.items()}
    peak = max(scaled.values())
    weights = {key: exp(max(-700.0, value - peak)) for key, value in scaled.items()}
    total = sum(weights.values())
    return {key: value / total for key, value in weights.items()}


def entropy(probabilities: Mapping[str, float]) -> float:
    """Calculate Shannon entropy in nats for a probability distribution."""
    return -sum(float(value) * log(max(_EPS, float(value)))
                for value in probabilities.values() if value > 0.0)


def normalize_prior(parses: Sequence[ParseHypothesis]) -> dict[str, float]:
    """Normalize parse priors, using a uniform fallback only when all supplied mass is zero."""
    if not parses:
        return {}
    total = sum(item.prior for item in parses)
    if total <= _EPS:
        return {item.parse_id: 1.0 / len(parses) for item in parses}
    return {item.parse_id: item.prior / total for item in parses}


def segment_input(request: SentenceRequest) -> dict[str, Any]:
    """Tokenize text into word/punctuation units while retaining offsets and boundary cues."""
    tokens = []
    for index, match in enumerate(_TOKEN_PATTERN.finditer(request.text)):
        raw = match.group(0)
        is_punctuation = bool(re.fullmatch(r"[^\w\s]", raw, re.UNICODE))
        tokens.append({
            "index": index,
            "text": raw,
            "normalized": raw.casefold(),
            "start": match.start(),
            "end": match.end(),
            "is_punctuation": is_punctuation,
            "boundary_after": raw in {".", "?", "!", ";", ":"},
        })
    return {
        "tokens": tokens,
        "token_count": len(tokens),
        "word_count": sum(not row["is_punctuation"] for row in tokens),
        "punctuation_positions": [row["index"] for row in tokens if row["is_punctuation"]],
        "language": request.language,
        "modality": request.modality,
        "segmentation_method": "unicode_word_and_punctuation_regex",
        "segmentation_is_provisional": True,
    }


def activate_lexical_entries(
    request: SentenceRequest,
    segmentation: Mapping[str, Any],
) -> dict[str, Any]:
    """Retrieve candidate lemmas, meanings, and grammatical categories per word."""
    output = []
    for token in segmentation["tokens"]:
        if token["is_punctuation"]:
            output.append({
                **token,
                "candidates": [{"category": "punctuation", "lemma": token["text"],
                               "sense": "surface boundary", "prior": 1.0, "features": {}}],
                "unknown_word": False,
            })
            continue
        entries = request.lexicon.get(token["normalized"],
                                      request.lexicon.get(token["text"], ()))
        if not entries:
            candidates = [{
                "category": "unknown",
                "lemma": token["normalized"],
                "sense": "",
                "prior": 1.0,
                "features": {},
            }]
            unknown = True
        else:
            total = sum(item.prior for item in entries)
            candidates = [{
                "category": item.category,
                "lemma": item.lemma or token["normalized"],
                "sense": item.sense,
                "prior": (item.prior / total if total > 0 else 1.0 / len(entries)),
                "features": dict(item.features),
            } for item in entries]
            unknown = False
        output.append({**token, "candidates": candidates, "unknown_word": unknown})
    return {
        "lexical_tokens": output,
        "unknown_word_count": sum(bool(row.get("unknown_word")) for row in output),
        "ambiguous_word_count": sum(len(row["candidates"]) > 1 for row in output),
        "candidate_entries_are_hypotheses": True,
    }


def generate_syntactic_continuations(
    request: SentenceRequest,
    lexical: Mapping[str, Any],
) -> dict[str, Any]:
    """Use supplied grammar-generated candidates or construct a transparent surface-only fallback."""
    parses = list(request.parse_hypotheses)
    fallback_used = False
    if not parses:
        fallback_used = True
        categories = {}
        for token in lexical["lexical_tokens"]:
            if token["is_punctuation"]:
                continue
            top = max(token["candidates"], key=lambda item: item["prior"])
            categories[token["index"]] = top["category"]
        parses = [ParseHypothesis(
            parse_id="surface-only",
            description="Surface-order fallback; no full grammar analysis was supplied.",
            prior=1.0,
            token_categories=categories,
            semantic_plausibility=0.5,
        )]
    token_count = len(lexical["lexical_tokens"])
    for parse in parses:
        for index in parse.token_categories:
            if int(index) >= token_count:
                raise ValueError(
                    f"parse {parse.parse_id!r} references token index {index} beyond sentence length"
                )
        for index in parse.thematic_roles.values():
            if int(index) >= token_count:
                raise ValueError(
                    f"parse {parse.parse_id!r} thematic role references token index {index} beyond sentence length"
                )
        for dependency in parse.dependencies:
            if dependency.head_index >= token_count or dependency.dependent_index >= token_count:
                raise ValueError(
                    f"parse {parse.parse_id!r} dependency references a token index beyond sentence length"
                )
        for point in parse.choice_points:
            if point.token_index >= token_count:
                raise ValueError(
                    f"parse {parse.parse_id!r} choice point references a token index beyond sentence length"
                )
    return {
        "parse_hypotheses": parses,
        "candidate_count": len(parses),
        "beam_width": len(parses),
        "fallback_used": fallback_used,
        "candidate_generation_contract": (
            "Candidates are supplied by a grammar/analysis front end; the package scores and revises them."
            if not fallback_used else
            "No grammar candidates were supplied; created a surface-only baseline, not a syntactic parse."
        ),
    }


def estimate_structural_expectations(
    parses: Sequence[ParseHypothesis],
    lexical: Mapping[str, Any],
    config: Point23Config,
) -> dict[str, Any]:
    """Score every parse incrementally and calculate token surprisal and posterior trajectories."""
    token_rows = lexical["lexical_tokens"]
    priors = normalize_prior(parses)
    cumulative = {item.parse_id: log(max(_EPS, priors[item.parse_id])) for item in parses}
    token_trace = []
    surprisal_values = []
    posterior_history = []
    for token in token_rows:
        if token["is_punctuation"]:
            token_trace.append({
                "token_index": token["index"], "token": token["text"],
                "surprisal": 0.0, "parse_probabilities": softmax(cumulative, config.parse_temperature),
                "punctuation": True,
            })
            posterior_history.append(softmax(cumulative, config.parse_temperature))
            continue
        lexical = {
            item["category"]: float(item["prior"]) for item in token["candidates"]
        }
        prior_distribution = softmax(cumulative, config.parse_temperature)
        predictive_mass = 0.0
        increments: dict[str, float] = {}
        for parse in parses:
            expected_category = parse.token_categories.get(token["index"])
            if expected_category is None or expected_category == "unknown":
                lexical_likelihood = 1.0
            else:
                lexical_likelihood = lexical.get(str(expected_category), 0.0)
                if lexical_likelihood <= 0.0 and any(
                    item["category"] == "unknown" for item in token["candidates"]
                ):
                    lexical_likelihood = 0.5
            structural = float(parse.structural_likelihoods.get(token["index"], 1.0))
            evidence = max(_EPS, lexical_likelihood) * max(_EPS, structural)
            increments[parse.parse_id] = log(evidence)
            predictive_mass += prior_distribution[parse.parse_id] * evidence
        surprisal = -log(max(_EPS, predictive_mass))
        for parse_id, increment in increments.items():
            cumulative[parse_id] += increment
        posterior = softmax(cumulative, config.parse_temperature)
        surprisal_values.append(surprisal)
        token_trace.append({
            "token_index": token["index"],
            "token": token["text"],
            "surprisal": surprisal,
            "parse_probabilities": posterior,
            "lexical_structural_log_evidence": increments,
        })
        posterior_history.append(posterior)
    return {
        "token_trace": token_trace,
        "parse_log_scores": cumulative,
        "parse_probabilities": softmax(cumulative, config.parse_temperature),
        "posterior_history": posterior_history,
        "mean_surprisal": sum(surprisal_values) / len(surprisal_values) if surprisal_values else 0.0,
        "maximum_surprisal": max(surprisal_values, default=0.0),
        "expectations_are_experience_dependent": True,
    }


def integrate_semantic_constraints(
    parses: Sequence[ParseHypothesis],
    structural: Mapping[str, Any],
    config: Point23Config,
) -> dict[str, Any]:
    """Add parse-specific semantic plausibility while preserving syntactic evidence separately."""
    syntax = structural["parse_log_scores"]
    semantic_log = {
        item.parse_id: log(max(_EPS, item.semantic_plausibility))
        for item in parses
    }
    combined = {
        item.parse_id: float(syntax[item.parse_id]) + semantic_log[item.parse_id]
        for item in parses
    }
    return {
        "syntax_log_scores": dict(syntax),
        "semantic_log_likelihoods": semantic_log,
        "combined_log_scores": combined,
        "parse_probabilities": softmax(combined, config.parse_temperature),
        "semantic_plausibility_used_as_constraint_not_veto": True,
    }


def select_provisional_parse(
    parses: Sequence[ParseHypothesis],
    integration: Mapping[str, Any],
    config: Point23Config,
) -> dict[str, Any]:
    """Select the highest-scoring candidate while retaining a bounded beam of alternatives."""
    probabilities = dict(integration["parse_probabilities"])
    ranked = sorted(parses, key=lambda item: probabilities[item.parse_id], reverse=True)
    beam = ranked[:config.beam_width]
    top = beam[0]
    return {
        "initial_parse_id": top.parse_id,
        "initial_parse_description": top.description,
        "beam_parse_ids": [item.parse_id for item in beam],
        "parse_probabilities": probabilities,
        "parse_entropy": entropy(probabilities),
        "normalized_parse_entropy": (
            entropy(probabilities) / log(len(probabilities)) if len(probabilities) > 1 else 0.0
        ),
        "premature_commitment_risk": len(beam) == 1 or probabilities[top.parse_id] > 0.90,
    }


def maintain_dependencies(
    parses: Sequence[ParseHypothesis],
    selected: Mapping[str, Any],
    lexical: Mapping[str, Any],
    request: SentenceRequest,
    config: Point23Config,
) -> dict[str, Any]:
    """Maintain candidate dependencies and roles with decay, interference, and capacity constraints."""
    parse_by_id = {item.parse_id: item for item in parses}
    best = parse_by_id[selected["initial_parse_id"]]
    token_rows = lexical["lexical_tokens"]
    n = len(token_rows)
    rows = []
    for dependency in best.dependencies:
        age = max(0, n - 1 - max(dependency.head_index, dependency.dependent_index))
        decay = exp(-config.memory_decay * age)
        competing = sum(
            1 for item in best.dependencies
            if item.relation == dependency.relation and item != dependency
        )
        interference = config.interference_weight * competing
        activation = clamp(decay - interference - request.working_memory_load * 0.35)
        rows.append({
            "head_index": dependency.head_index,
            "head_token": token_rows[dependency.head_index]["text"],
            "dependent_index": dependency.dependent_index,
            "dependent_token": token_rows[dependency.dependent_index]["text"],
            "relation": dependency.relation,
            "thematic_role": dependency.thematic_role,
            "activation": activation,
        })
    load = clamp(
        (len(rows) / max(1, config.beam_width)) * 0.35
        + request.working_memory_load * 0.55
    )
    available = config.working_memory_capacity * (1.0 - request.working_memory_load)
    return {
        "active_parse_id": best.parse_id,
        "active_dependencies": rows,
        "dependency_count": len(rows),
        "estimated_load": load,
        "available_capacity": available,
        "capacity_pressure": load > available,
        "unresolved_dependency_labels": list(best.unresolved_dependencies),
        "decay_and_interference_are_approximations": True,
    }


def detect_disambiguating_conflict(
    structural: Mapping[str, Any],
    selected: Mapping[str, Any],
    request: SentenceRequest,
) -> dict[str, Any]:
    """Detect surprise, entropy, posterior shifts, and changes to the leading parse."""
    history = structural["posterior_history"]
    if history:
        choice_index = min(max(0, int(selected.get("initial_choice_index", 0))), len(history) - 1)
        first_best = max(history[choice_index], key=history[choice_index].get)
        last_best = max(history[-1], key=history[-1].get)
        last = history[-1]
        shifts = [
            0.5 * sum(abs(row.get(key, 0.0) - history[index - 1].get(key, 0.0))
                      for key in row)
            for index, row in enumerate(history) if index > 0
        ]
        shift = max(shifts, default=0.0)
    else:
        first_best = selected["initial_parse_id"]
        last_best = first_best
        last = dict(selected["parse_probabilities"])
        shift = 0.0
    changed = first_best != last_best
    word_rows = [row for row in structural["token_trace"] if not row.get("punctuation")]
    last_surprisal = max((float(row["surprisal"]) for row in word_rows), default=0.0)
    normalized_entropy = (
        entropy(last) / log(len(last)) if len(last) > 1 else 0.0
    )
    cue_conflict = clamp(0.6 * min(1.0, last_surprisal / 4.0) + 0.4 * shift)
    return {
        "initial_leading_parse_id": first_best,
        "current_leading_parse_id": last_best,
        "leading_parse_changed": changed,
        "posterior_shift": shift,
        "last_token_surprisal": last_surprisal,
        "normalized_parse_entropy": normalized_entropy,
        "conflict_score": cue_conflict,
        "disambiguating_cue_detected": changed or cue_conflict >= 0.20,
        "task_goal": request.task_goal,
    }


def test_parse_viability(
    parses: Sequence[ParseHypothesis],
    integrated: Mapping[str, Any],
    config: Point23Config,
) -> dict[str, Any]:
    """Retain candidates above the viability floor and document low-ranked alternatives."""
    probabilities = dict(integrated["parse_probabilities"])
    ranked = sorted(parses, key=lambda item: probabilities[item.parse_id], reverse=True)
    viable = [item for item in ranked if probabilities[item.parse_id] >= config.minimum_viability]
    if not viable and ranked:
        viable = [ranked[0]]
    rejected = [item.parse_id for item in ranked if item not in viable]
    return {
        "viable_parse_ids": [item.parse_id for item in viable],
        "rejected_or_low_viability_parse_ids": rejected,
        "viability_scores": probabilities,
        "viability_floor": config.minimum_viability,
        "best_parse_id": viable[0].parse_id if viable else None,
        "all_candidates_preserved_for_audit": True,
    }


def trigger_reanalysis(
    selected: Mapping[str, Any],
    conflict: Mapping[str, Any],
    viability: Mapping[str, Any],
    config: Point23Config,
) -> dict[str, Any]:
    """Trigger reanalysis on leading-parse change, loss of viability, or sufficiently large conflict."""
    initial = selected["initial_parse_id"]
    best = viability["best_parse_id"]
    probs = viability["viability_scores"]
    best_probability = probs.get(best, 0.0) if best else 0.0
    initial_probability = probs.get(initial, 0.0)
    margin = best_probability - initial_probability
    initial_viable = initial in viability["viable_parse_ids"]
    trigger = (
        (best is not None and best != initial and margin >= config.revision_margin)
        or not initial_viable
        or conflict["conflict_score"] >= config.conflict_threshold
    )
    return {
        "revision_requested": trigger,
        "initial_parse_id": initial,
        "candidate_parse_id": best,
        "initial_parse_probability": initial_probability,
        "candidate_parse_probability": best_probability,
        "candidate_advantage": margin,
        "initial_parse_remains_viable": initial_viable,
        "trigger_reasons": [
            reason for condition, reason in (
                (best is not None and best != initial and margin >= config.revision_margin,
                 "alternative_parse_outweighs_initial_parse"),
                (not initial_viable, "initial_parse_below_viability_floor"),
                (conflict["conflict_score"] >= config.conflict_threshold, "conflict_threshold_exceeded"),
            ) if condition
        ],
    }


def locate_revision_points(
    parses: Sequence[ParseHypothesis],
    trigger: Mapping[str, Any],
) -> dict[str, Any]:
    """Find the earliest structural choice whose value differs between initial and candidate parses."""
    by_id = {item.parse_id: item for item in parses}
    old = by_id[trigger["initial_parse_id"]]
    new = by_id.get(trigger["candidate_parse_id"])
    if new is None:
        return {"revision_points": [], "earliest_revision_index": None}
    old_points = {item.decision_id: item for item in old.choice_points}
    new_points = {item.decision_id: item for item in new.choice_points}
    differences = []
    for decision_id in sorted(set(old_points) | set(new_points)):
        before = old_points.get(decision_id)
        after = new_points.get(decision_id)
        if before is None or after is None or before.choice != after.choice:
            positions = [item.token_index for item in (before, after) if item is not None]
            differences.append({
                "decision_id": decision_id,
                "token_index": min(positions) if positions else 0,
                "before": before.choice if before else None,
                "after": after.choice if after else None,
            })
    differences.sort(key=lambda item: item["token_index"])
    return {
        "revision_points": differences,
        "earliest_revision_index": differences[0]["token_index"] if differences else None,
        "earliest_revision_choice": differences[0] if differences else None,
    }


def reconstruct_alternative_structure(
    parses: Sequence[ParseHypothesis],
    viability: Mapping[str, Any],
    trigger: Mapping[str, Any],
    points: Mapping[str, Any],
) -> dict[str, Any]:
    """Select the best viable alternative without fabricating a parse outside the supplied grammar."""
    initial = trigger["initial_parse_id"]
    candidates = list(viability["viable_parse_ids"])
    alternatives = [item for item in candidates if item != initial]
    chosen = (
        trigger["candidate_parse_id"]
        if trigger["revision_requested"] and trigger["candidate_parse_id"] in candidates
        else initial
    )
    success = bool(trigger["revision_requested"] and chosen != initial and alternatives)
    return {
        "revision_attempted": bool(trigger["revision_requested"]),
        "revision_succeeded": success,
        "original_parse_id": initial,
        "revised_parse_id": chosen,
        "alternative_parse_ids": alternatives,
        "revision_points": list(points["revision_points"]),
        "earliest_revision_index": points["earliest_revision_index"],
        "candidate_parse_generation": "selects among supplied grammar candidates; does not invent unsupported structures",
    }


def reassign_thematic_roles(
    parses: Sequence[ParseHypothesis],
    parse_id: str,
    lexical: Mapping[str, Any],
    probabilities: Mapping[str, float],
) -> dict[str, Any]:
    """Recompute semantic role links from the selected structure and preserve uncertainty."""
    by_id = {item.parse_id: item for item in parses}
    parse = by_id[parse_id]
    token_rows = lexical["lexical_tokens"]
    assignments = {}
    for role, index in parse.thematic_roles.items():
        assignments[role] = {
            "token_index": int(index),
            "token": token_rows[int(index)]["text"],
            "confidence": float(probabilities.get(parse_id, 0.0)),
        }
    return {
        "parse_id": parse_id,
        "description": parse.description,
        "thematic_roles": assignments,
        "dependencies": [
            {
                "head": token_rows[item.head_index]["text"],
                "dependent": token_rows[item.dependent_index]["text"],
                "relation": item.relation,
                "thematic_role": item.thematic_role,
            } for item in parse.dependencies
        ],
        "roles_recomputed_from_selected_parse": True,
        "roles_not_supplied": not bool(assignments),
    }


def reconcile_interpretations(
    initial_parse_id: str,
    final_parse_id: str,
    probabilities: Mapping[str, float],
    parses: Sequence[ParseHypothesis],
) -> dict[str, Any]:
    """Keep the initial parse trace while separately calibrating syntax, semantics, and role confidence."""
    by_id = {item.parse_id: item for item in parses}
    initial_probability = float(probabilities.get(initial_parse_id, 0.0))
    final_probability = float(probabilities.get(final_parse_id, 0.0))
    semantic = float(by_id[final_parse_id].semantic_plausibility)
    return {
        "initial_parse_id": initial_parse_id,
        "final_parse_id": final_parse_id,
        "initial_parse_residual_activation": initial_probability,
        "final_structure_confidence": final_probability,
        "semantic_confidence": clamp(final_probability * semantic),
        "thematic_role_confidence": final_probability,
        "interpretation_changed": initial_parse_id != final_parse_id,
        "initial_analysis_erased": False,
        "reconciliation_complete": final_probability >= 0.5,
    }


def decide_reanalysis_depth(
    parses: Sequence[ParseHypothesis],
    parse_id: str,
    probabilities: Mapping[str, float],
    memory: Mapping[str, Any],
    request: SentenceRequest,
    config: Point23Config,
) -> dict[str, Any]:
    """Choose a complete parse, task-sufficient gist, or explicit unresolved state."""
    best_probability = float(probabilities.get(parse_id, 0.0))
    parse = next(item for item in parses if item.parse_id == parse_id)
    entropy_norm = (
        entropy(probabilities) / log(len(probabilities)) if len(probabilities) > 1 else 0.0
    )
    unresolved = list(parse.unresolved_dependencies)
    if memory["capacity_pressure"]:
        unresolved.append("working_memory_capacity_pressure")
    ambiguity = entropy_norm >= config.full_reanalysis_threshold
    if request.task_goal == "precise":
        status = "full" if best_probability >= config.confidence_threshold and not unresolved and not ambiguity else "unresolved"
    elif best_probability >= 0.50 and not unresolved:
        status = "good_enough"
    else:
        status = "unresolved"
    return {
        "interpretation_status": status,
        "selected_parse_id": parse_id,
        "parse_confidence": best_probability,
        "normalized_entropy": entropy_norm,
        "working_memory_pressure": memory["capacity_pressure"],
        "unresolved_ambiguities": list(dict.fromkeys(unresolved)),
        "processing_goal": request.task_goal,
        "stopping_reason": (
            "task-sufficient interpretation accepted" if status == "good_enough"
            else "precision threshold met" if status == "full"
            else "uncertainty or unresolved dependencies remain"
        ),
        "full_reanalysis_is_not_assumed_automatic": True,
    }


def integrate_surface_cues(
    request: SentenceRequest,
    parses: Sequence[ParseHypothesis],
    base_log_scores: Mapping[str, float],
    lexical: Mapping[str, Any],
    config: Point23Config,
) -> dict[str, Any]:
    """Fuse prosody, punctuation, and discourse likelihoods as reliability-weighted parse evidence."""
    cue_sources = {
        "context": request.context_likelihoods,
        "prosody": request.prosody_likelihoods,
        "punctuation": request.punctuation_likelihoods,
        "discourse": request.discourse_likelihoods,
    }
    contributions: dict[str, dict[str, float]] = {}
    final_scores: dict[str, float] = {}
    for parse in parses:
        cue_rows = {}
        for cue_name, mapping in cue_sources.items():
            value = float(mapping.get(parse.parse_id, 1.0))
            if value != 1.0:
                cue_rows[cue_name] = log(max(_EPS, value))
        for cue_name, value in parse.cue_likelihoods.items():
            cue_rows[f"parse:{cue_name}"] = log(max(_EPS, float(value)))
        contributions[parse.parse_id] = cue_rows
        final_scores[parse.parse_id] = float(base_log_scores[parse.parse_id]) + sum(cue_rows.values())
    probs = softmax(final_scores, config.parse_temperature)
    ranked = sorted(parses, key=lambda item: probs[item.parse_id], reverse=True)
    return {
        "cue_log_contributions": contributions,
        "final_log_scores": final_scores,
        "parse_probabilities": probs,
        "final_parse_id": ranked[0].parse_id,
        "final_parse_description": ranked[0].description,
        "cue_sources_used": [name for name, values in cue_sources.items() if values],
        "punctuation_tokens": [
            row["text"] for row in lexical["lexical_tokens"] if row["is_punctuation"]
        ],
        "prosodic_and_written_cues_remain_modality_specific": True,
    }


def update_expectations_from_experience(
    request: SentenceRequest,
    parses: Sequence[ParseHypothesis],
    final_parse_id: str,
    state: Point23State,
    config: Point23Config,
) -> dict[str, Any]:
    """Update construction exposure and only treat an explicit feedback parse as outcome evidence."""
    construction = request.construction_id
    counts = state.construction_counts.setdefault(construction, {})
    state.construction_exposures[construction] = state.construction_exposures.get(construction, 0) + 1
    feedback = request.feedback_parse_id
    updated = False
    if feedback is not None:
        old = float(counts.get(feedback, 0.0))
        counts[feedback] = old + config.learning_rate
        updated = True
    prior = {item.parse_id: item.prior for item in parses}
    total = sum(prior.values())
    prior_dist = ({key: value / total for key, value in prior.items()}
                  if total > _EPS else {key: 1.0 / len(prior) for key in prior})
    outcomes = {item.parse_id: float(counts.get(item.parse_id, 0.0)) for item in parses}
    denominator = sum(outcomes.values()) + config.prior_smoothing
    learned = {
        item.parse_id: (
            outcomes[item.parse_id] + config.prior_smoothing * prior_dist[item.parse_id]
        ) / denominator if denominator > 0 else prior_dist[item.parse_id]
        for item in parses
    }
    history = {
        "sentence_id": request.sentence_id,
        "construction_id": construction,
        "selected_parse_id": final_parse_id,
        "feedback_parse_id": feedback,
        "feedback_applied": updated,
        "exposure_count": state.construction_exposures[construction],
    }
    state.interpretation_history.append(history)
    state.interpretation_history = state.interpretation_history[-500:]
    return {
        **history,
        "learned_construction_expectations": learned,
        "learning_requires_explicit_feedback": True,
        "exposure_is_not_treated_as_correctness": True,
    }


def expected_behavioral_signatures(
    parses: Sequence[ParseHypothesis],
    token_trace: Sequence[Mapping[str, Any]],
    revisions: Sequence[Mapping[str, Any]],
    final_probabilities: Mapping[str, float],
    role_output: Mapping[str, Any],
) -> dict[str, Any]:
    """Produce observable predictions for behavioral validation without pretending to measure them."""
    disambiguation = max(
        (row for row in token_trace if not row.get("punctuation")),
        key=lambda row: row.get("surprisal", 0.0),
        default=None,
    )
    best = max(final_probabilities, key=final_probabilities.get) if final_probabilities else None
    return {
        "predicted_processing_cost_by_token": [
            {"token_index": row["token_index"], "token": row["token"],
             "surprisal": row["surprisal"], "relative_cost": row["surprisal"]}
            for row in token_trace if not row.get("punctuation")
        ],
        "highest_surprisal_region": (
            {"token_index": disambiguation["token_index"], "token": disambiguation["token"]}
            if disambiguation else None
        ),
        "revision_count": sum(bool(row.get("revision_succeeded")) for row in revisions),
        "final_parse_id": best,
        "thematic_role_probe_targets": list(role_output["thematic_roles"]),
        "rereading_is_not_equated_with_successful_reanalysis": True,
        "neural_measures_are_convergent_not_unique_step_labels": True,
    }


def validate_point23(
    trace: Sequence[Mapping[str, Any]],
    probabilities: Mapping[str, float],
    parses: Sequence[ParseHypothesis],
) -> dict[str, Any]:
    """Validate source order, normalized parse mass, and selected parse membership."""
    expected = [f"P23-{index:02d}" for index in range(1, 18)]
    observed = [str(row.get("operation_id")) for row in trace]
    failures = []
    if observed != expected:
        failures.append("operation_order_mismatch")
    if set(probabilities) != {item.parse_id for item in parses}:
        failures.append("parse_probability_membership_mismatch")
    if probabilities and abs(sum(float(value) for value in probabilities.values()) - 1.0) > 1e-8:
        failures.append("parse_probabilities_not_normalized")
    return {
        "operation_order_verified": observed == expected,
        "operation_count": len(trace),
        "parse_probabilities_normalized": bool(probabilities)
            and abs(sum(float(value) for value in probabilities.values()) - 1.0) <= 1e-8,
        "parse_ids_match_candidates": set(probabilities) == {item.parse_id for item in parses},
        "failure_signatures": failures,
        "valid": not failures,
    }
