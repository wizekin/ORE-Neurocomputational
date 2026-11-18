"""Functional computations for Point 15's cooperating but distinct cognitive systems."""
from __future__ import annotations

import math
import re
from typing import Any, Mapping, Sequence

from .contracts import LanguageEvent, LanguageLearnerState, LanguageProcessingConfig


_TOKEN = re.compile(r"[A-Za-z0-9_]+(?:['’-][A-Za-z0-9_]+)*|[^\w\s]", re.UNICODE)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Bound a scalar while keeping NaN and infinities from entering persistent state."""
    if not math.isfinite(float(value)):
        raise ValueError("value must be finite")
    return max(low, min(high, float(value)))


def tokenize(text: str) -> list[str]:
    """Tokenize a textual signal into ordered words and punctuation marks."""
    return [match.group(0) for match in _TOKEN.finditer(str(text))]


def _words(value: Any) -> set[str]:
    """Extract normalized lexical tokens from scalar or nested textual context values."""
    if value is None:
        return set()
    if isinstance(value, Mapping):
        result: set[str] = set()
        for key, item in value.items():
            result.update(_words(key))
            result.update(_words(item))
        return result
    if isinstance(value, (tuple, list, set)):
        result: set[str] = set()
        for item in value:
            result.update(_words(item))
        return result
    return {item.lower() for item in tokenize(str(value)) if item.isalnum() or "_" in item}


def sigmoid(value: float) -> float:
    """Compute a numerically stable logistic transform."""
    value = float(value)
    if value >= 0:
        z = math.exp(-min(value, 700.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(value, -700.0))
    return z / (1.0 + z)


def softmax(scores: Sequence[float], temperature: float = 1.0) -> list[float]:
    """Normalize competing scores into a stable probability distribution."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not scores:
        return []
    scaled = [float(item) / temperature for item in scores]
    peak = max(scaled)
    exps = [math.exp(max(-700.0, item - peak)) for item in scaled]
    total = sum(exps)
    return [item / total for item in exps]


def shannon_entropy(probabilities: Sequence[float]) -> float:
    """Calculate uncertainty in nats for a normalized or unnormalized distribution."""
    total = sum(max(0.0, float(item)) for item in probabilities)
    if total <= 0:
        return 0.0
    return -sum(
        (float(item) / total) * math.log(float(item) / total)
        for item in probabilities if item > 0
    )


def encode_sensory_signal(event: LanguageEvent) -> dict[str, Any]:
    """Preserve modality, order, segments, and signal quality before linguistic interpretation.

    For speech, sensory_stream is assumed to be a transcript or symbolic feature sequence;
    this function does not decode raw audio waveforms.
    """
    segments = (
        list(event.sensory_stream)
        if isinstance(event.sensory_stream, (tuple, list))
        else [str(event.sensory_stream)]
    )
    combined = " ".join(segments)
    tokens = tokenize(combined)
    quality = clamp(float(event.cognitive_state.get("signal_quality", 1.0)))
    missing = list(event.metadata.get("missing_segments", ()))
    timings = list(event.metadata.get("timing_cues", ())) if event.modality == "speech" else []
    orthography = {
        "character_count": len(combined),
        "capitalized_token_positions": [
            index for index, token in enumerate(tokens) if token[:1].isupper()
        ],
    } if event.modality == "text" else {}
    return {
        "modality": event.modality,
        "ordered_segments": segments,
        "ordered_tokens": tokens,
        "token_positions": list(range(len(tokens))),
        "signal_quality": quality,
        "missing_segments": missing,
        "timing_or_prosodic_cues": timings,
        "orthographic_cues": orthography,
        "transcript_assumption_for_speech": event.modality == "speech",
        "signal_uncertainty": clamp(1.0 - quality + min(0.25, 0.05 * len(missing))),
    }


def convert_to_language_units(
    event: LanguageEvent,
    features: Mapping[str, Any],
    config: LanguageProcessingConfig,
) -> dict[str, Any]:
    """Generate phonological/orthographic candidate units using signal and context fit."""
    tokens = [str(x) for x in features.get("ordered_tokens", ())]
    context = _words(event.linguistic_context)
    alternatives: dict[str, list[dict[str, Any]]] = {}
    candidate_entropy: list[float] = []
    for token in tokens:
        if not token.isalnum() and "_" not in token:
            continue
        key = token.lower()
        supplied = event.lexical_alternatives.get(key, event.lexical_alternatives.get(token, ()))
        choices = [str(item) for item in supplied] if supplied else [token]
        if token not in choices:
            choices.insert(0, token)
        unique = list(dict.fromkeys(choices))[:config.max_candidates_per_token]
        scores: list[float] = []
        rows: list[dict[str, Any]] = []
        for candidate in unique:
            normalized = candidate.lower()
            sensory_fit = 1.0 if normalized == key else 0.65
            context_fit = 1.0 if normalized in context else 0.5
            score = config.sensory_weight * sensory_fit + config.context_weight * context_fit
            scores.append(score)
            rows.append({"unit": candidate, "sensory_fit": sensory_fit,
                         "context_fit": context_fit, "score": score})
        probabilities = softmax(scores, config.lexical_temperature)
        for row, probability in zip(rows, probabilities):
            row["probability"] = probability
        alternatives[key] = rows
        candidate_entropy.append(shannon_entropy(probabilities))
    return {
        "candidate_units": alternatives,
        "unit_sequence": [token.lower() for token in tokens],
        "mean_candidate_entropy": sum(candidate_entropy) / max(1, len(candidate_entropy)),
        "context_used": bool(context),
        "sensory_context_weights": {
            "sensory": config.sensory_weight, "context": config.context_weight,
        },
    }


def activate_language_representations(
    features: Mapping[str, Any],
    units: Mapping[str, Any],
    prior_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a recurrent, language-selective state without adding real-world inferences."""
    tokens = list(units.get("unit_sequence", ()))
    prior = prior_state or {}
    prior_activation = float(prior.get("activation", 0.0))
    quality = float(features.get("signal_quality", 1.0))
    activation = clamp(0.65 * quality + 0.25 * min(1.0, len(tokens) / 12.0)
                       + 0.10 * prior_activation)
    return {
        "tokens": tokens,
        "modality": features.get("modality"),
        "activation": activation,
        "recurrent_prior_activation": prior_activation,
        "linguistic_form_confidence": clamp(activation * (1.0 - float(features.get("signal_uncertainty", 0.0)))),
        "real_world_interpretation_confidence": None,
        "language_knowledge_accessed": True,
        "network_scope": "distributed_language_selective_frontotemporal_function",
    }


def retrieve_lexical_knowledge(
    event: LanguageEvent,
    units: Mapping[str, Any],
    state: LanguageLearnerState,
    config: LanguageProcessingConfig,
) -> dict[str, Any]:
    """Rank lexical candidates using signal fit, context prior, frequency, and learned strength."""
    output: dict[str, Any] = {}
    entropies: list[float] = []
    for word, candidates in units.get("candidate_units", {}).items():
        scores: list[float] = []
        rows: list[dict[str, Any]] = []
        for candidate in candidates:
            surface = str(candidate["unit"])
            entry = event.lexicon.get(surface.lower(), event.lexicon.get(surface, {}))
            entry = entry if isinstance(entry, Mapping) else {"definition": str(entry)}
            frequency = clamp(float(entry.get("frequency", 0.5)))
            contextual_prior = clamp(float(entry.get("context_prior", candidate.get("context_fit", 0.5))))
            learned = state.lexical_strengths.get(surface.lower(), 0.5)
            score = (
                0.45 * float(candidate["sensory_fit"])
                + 0.25 * contextual_prior
                + 0.15 * frequency
                + 0.15 * learned
            )
            scores.append(score)
            rows.append({
                "word": surface,
                "grammar": entry.get("grammar", entry.get("part_of_speech")),
                "properties": dict(entry.get("properties", {})) if isinstance(entry.get("properties", {}), Mapping) else {},
                "definition": entry.get("definition"),
                "score": score,
            })
        probabilities = softmax(scores, config.lexical_temperature)
        for row, probability in zip(rows, probabilities):
            row["probability"] = probability
        rows.sort(key=lambda item: item["probability"], reverse=True)
        output[word] = rows
        entropies.append(shannon_entropy([row["probability"] for row in rows]))
    confidences = [rows[0]["probability"] for rows in output.values() if rows]
    return {
        "lexical_candidates": output,
        "lexical_confidence": sum(confidences) / max(1, len(confidences)),
        "lexical_uncertainty_nats": sum(entropies),
        "alternatives_preserved": any(len(rows) > 1 for rows in output.values()),
    }


def build_incremental_structure(
    event: LanguageEvent,
    lexical: Mapping[str, Any],
    config: LanguageProcessingConfig,
    state: LanguageLearnerState | None = None,
) -> dict[str, Any]:
    """Build a lightweight incremental structural hypothesis and construction matches.

    This is a transparent symbolic approximation for tests and integration, not a full parser.
    """
    tokens = [token.lower() for token in tokenize(
        " ".join(event.sensory_stream) if isinstance(event.sensory_stream, (tuple, list))
        else str(event.sensory_stream)
    ) if token.isalnum() or "_" in token]
    roles: dict[str, list[str]] = {"subject_candidates": [], "predicate_candidates": [],
                                  "object_candidates": [], "modifiers": [], "connectors": []}
    verbs = {"is", "are", "was", "were", "be", "been", "has", "have", "had", "do", "does",
             "did", "can", "could", "will", "would", "should", "must", "make", "makes",
             "made", "use", "uses", "used", "cause", "causes", "caused", "increase",
             "increases", "increased", "decrease", "decreases", "decreased", "expand",
             "expands", "expanded", "close", "closes", "closed"}
    conjunctions = {"and", "or", "but", "because", "although", "if", "then", "when", "while"}
    verb_indices = [i for i, word in enumerate(tokens) if word in verbs or word.endswith("ed")]
    for word in tokens:
        if word in conjunctions:
            roles["connectors"].append(word)
    if tokens:
        first_verb = verb_indices[0] if verb_indices else max(1, len(tokens) - 1)
        roles["subject_candidates"] = tokens[:first_verb][:4]
        roles["predicate_candidates"] = tokens[first_verb:first_verb + 3]
        roles["object_candidates"] = tokens[first_verb + 1:first_verb + 5]
        roles["modifiers"] = [word for word in tokens if word.endswith(("ly", "ous", "ive", "al"))]
    matched: list[dict[str, Any]] = []
    for pattern in event.construction_patterns:
        expected = [str(x).lower() for x in pattern.get("tokens", ())]
        overlap = len(set(expected) & set(tokens)) / max(1, len(set(expected)))
        name = str(pattern.get("name", "unnamed"))
        learned_strength = (
            state.construction_strengths.get(name.lower(), 0.5) if state is not None else 0.5
        )
        score = clamp(
            0.35 * float(pattern.get("prior", 0.5))
            + 0.40 * overlap
            + 0.25 * learned_strength
        )
        if score > 0:
            matched.append({"name": name, "score": score, "learned_strength": learned_strength,
                            "relations": list(pattern.get("relations", ()))})
    matched.sort(key=lambda item: item["score"], reverse=True)
    pattern_probabilities = softmax(
        [item["score"] for item in matched], config.construction_temperature
    )
    for item, probability in zip(matched, pattern_probabilities):
        item["probability"] = probability
    structural_alternatives = [
        {"pattern": item["name"], "probability": item["probability"]} for item in matched
    ]
    best_probability = matched[0]["probability"] if matched else 0.5
    return {
        "tokens": tokens,
        "roles": roles,
        "construction_matches": matched,
        "structural_alternatives": structural_alternatives,
        "structural_confidence": best_probability,
        "unresolved_dependencies": [
            word for word in tokens if word in {"that", "which", "who", "if", "because"}
        ],
        "incremental": True,
        "parser_scope": "rule_based_partial_structure_not_full_syntactic_parser",
    }


def compose_sentence_meaning(
    lexical: Mapping[str, Any],
    structure: Mapping[str, Any],
    event: LanguageEvent,
) -> dict[str, Any]:
    """Compose a sentence-level linguistic representation, retaining direct-content provenance."""
    tokens = list(structure.get("tokens", ()))
    selected_words = {
        word: rows[0]["word"] for word, rows in lexical.get("lexical_candidates", {}).items() if rows
    }
    roles = structure.get("roles", {})
    propositions: list[dict[str, Any]] = []
    subject = list(roles.get("subject_candidates", ()))
    predicate = list(roles.get("predicate_candidates", ()))
    objects = list(roles.get("object_candidates", ()))
    if predicate:
        propositions.append({
            "subject_candidates": subject,
            "predicate_candidates": predicate,
            "object_candidates": objects,
            "source": "linguistic_structure",
            "confidence": float(structure.get("structural_confidence", 0.5)),
        })
    explicit = {
        "utterance": " ".join(event.sensory_stream) if isinstance(event.sensory_stream, (tuple, list))
                     else str(event.sensory_stream),
        "tokens": tokens,
        "selected_lexical_forms": selected_words,
        "propositions": propositions,
        "construction_relations": [
            relation for row in structure.get("construction_matches", ())
            for relation in row.get("relations", ())
        ],
        "source": "linguistic_input_only",
    }
    return {
        "sentence_meaning": {
            "tokens": tokens,
            "propositions": propositions,
            "construction_relations": explicit["construction_relations"],
            "compositional_confidence": clamp(
                0.5 * float(lexical.get("lexical_confidence", 0.0))
                + 0.5 * float(structure.get("structural_confidence", 0.5))
            ),
        },
        "said_or_encoded": explicit,
        "ambiguity_preserved": bool(structure.get("unresolved_dependencies"))
                               or bool(lexical.get("alternatives_preserved")),
    }


def retrieve_conceptual_knowledge(
    event: LanguageEvent,
    linguistic: Mapping[str, Any],
    config: LanguageProcessingConfig,
) -> dict[str, Any]:
    """Retrieve relevant world knowledge while marking it as retrieved, not explicitly said."""
    cues = _words(linguistic.get("sentence_meaning", {}))
    context_words = _words(event.situational_context)
    candidates: list[dict[str, Any]] = []
    for key, value in event.conceptual_memory.items():
        entry_words = _words(value) | _words(key)
        overlap = len(cues & entry_words) / max(1, len(entry_words))
        context_fit = len(context_words & entry_words) / max(1, len(entry_words))
        # A small prefix/stem proxy lets inflected forms such as "expanded" cue "expand".
        morphological_match = any(
            len(left) >= 4 and len(right) >= 4
            and (left.startswith(right) or right.startswith(left))
            for left in cues for right in entry_words
        )
        reliability = clamp(float(value.get("reliability", 0.7))) if isinstance(value, Mapping) else 0.7
        relevance = clamp((0.60 * overlap + 0.20 * context_fit
                           + 0.20 * float(morphological_match)) * reliability)
        if relevance >= config.concept_relevance_threshold:
            candidates.append({
                "concept": str(key), "knowledge": value, "overlap": overlap,
                "context_fit": context_fit, "reliability": reliability,
                "relevance": relevance, "provenance": "conceptual_memory",
            })
    candidates.sort(key=lambda item: item["relevance"], reverse=True)
    return {
        "activated_concepts": candidates,
        "conceptual_confidence": (
            sum(row["relevance"] for row in candidates) / len(candidates) if candidates else 0.0
        ),
        "retrieval_is_not_explicit_utterance_content": True,
    }


def estimate_processing_demand(
    linguistic: Mapping[str, Any],
    concepts: Mapping[str, Any],
    event: LanguageEvent,
) -> dict[str, Any]:
    """Estimate ambiguity, maintenance, novelty, and deliberate-comparison demands separately."""
    structure = linguistic.get("linguistic_structure", {})
    lexical = linguistic.get("lexical", {})
    ambiguity = clamp(float(lexical.get("lexical_uncertainty_nats", 0.0)) / 2.0
                      + 0.25 * len(structure.get("unresolved_dependencies", ())))
    dependency_load = clamp(len(structure.get("tokens", ())) / 30.0
                            + 0.12 * len(structure.get("unresolved_dependencies", ())))
    known = _words(event.conceptual_memory)
    cue_words = _words(linguistic.get("sentence_meaning", {}))
    novelty = clamp(1.0 - len(known & cue_words) / max(1, len(cue_words)))
    goal = event.task_goal.lower()
    deliberate = 1.0 if any(word in goal for word in ("reason", "infer", "calculate", "prove", "compare", "derive")) else 0.0
    cognitive = event.cognitive_state
    weights = cognitive.get("demand_weights", {})
    a, b, c, d = (float(weights.get(key, default)) for key, default in
                  (("ambiguity", 0.35), ("working_memory", 0.30), ("novelty", 0.20), ("deliberate", 0.15)))
    raw = a * ambiguity + b * dependency_load + c * novelty + d * deliberate
    demand = clamp(raw)
    capacity = clamp(float(cognitive.get("control_capacity", 0.65)))
    fatigue = clamp(float(cognitive.get("fatigue", 0.0)))
    effective_capacity = clamp(capacity * (1.0 - 0.45 * fatigue))
    return {
        "ambiguity": ambiguity, "working_memory_demand": dependency_load,
        "novelty": novelty, "deliberate_comparison": deliberate,
        "demand_score": demand, "capacity_estimate": effective_capacity,
        "domain_general_demand_is_separate_from_language_content": True,
        "weight_parameters": {"ambiguity": a, "working_memory": b, "novelty": c, "deliberate": d},
    }


def recruit_domain_general_control(
    demand: Mapping[str, Any],
    linguistic: Mapping[str, Any],
    event: LanguageEvent,
    config: LanguageProcessingConfig,
) -> dict[str, Any]:
    """Allocate graded control and working-memory support only when demand approaches capacity."""
    difference = float(demand["demand_score"]) - float(demand["capacity_estimate"])
    allocation = sigmoid(config.control_gamma * difference)
    recruited = allocation >= config.control_recruitment_threshold
    tokens = list(linguistic.get("sentence_meaning", {}).get("tokens", ()))
    return {
        "control_recruited": recruited,
        "control_allocation": allocation if recruited else allocation * 0.25,
        "demand_capacity_gap": difference,
        "maintenance_plan": tokens[-min(8, len(tokens)):] if recruited else [],
        "selection_required": recruited and bool(linguistic.get("ambiguity_preserved")),
        "strategy": "maintain_compare_and_monitor" if recruited else "routine_language_processing",
        "core_language_computation_remains_separate": True,
    }


def apply_deliberate_reasoning(
    event: LanguageEvent,
    linguistic: Mapping[str, Any],
    concepts: Mapping[str, Any],
    control: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply supplied explicit inference rules only when the task calls for deliberate reasoning."""
    goal = event.task_goal.lower()
    requested = bool(event.cognitive_state.get("requires_reasoning", False)) or any(
        word in goal for word in ("reason", "infer", "calculate", "prove", "compare", "derive", "logic")
    )
    explicit = linguistic.get("said_or_encoded", {})
    known = _words(explicit) | _words(concepts.get("activated_concepts", ()))
    inferred: list[dict[str, Any]] = []
    if requested:
        for index, rule in enumerate(event.reasoning_rules):
            premises = [str(item).lower() for item in rule.get("if_all", ())]
            if premises and all(_words(item).issubset(known) for item in premises):
                inferred.append({
                    "rule_id": str(rule.get("rule_id", f"rule-{index + 1}")),
                    "conclusion": rule.get("then"),
                    "premises": premises,
                    "source": "deliberate_reasoning_rule",
                    "is_inferred": True,
                })
    return {
        "reasoning_requested": requested,
        "reasoning_invoked": requested and bool(event.reasoning_rules),
        "reasoning_result": inferred,
        "said_or_encoded": explicit,
        "inferred": inferred,
        "inference_count": len(inferred),
        "requires_control_support": bool(control.get("control_recruited")) if requested else False,
        "linguistic_content_not_rewritten_by_inference": True,
    }


def infer_social_pragmatics(
    event: LanguageEvent,
    linguistic: Mapping[str, Any],
    config: LanguageProcessingConfig,
) -> dict[str, Any]:
    """Infer communicative intention only when context or task indicates a pragmatic demand."""
    situation = event.situational_context
    speaker = event.speaker_model
    goal = event.task_goal.lower()
    explicit_need = bool(situation.get("requires_intention_inference", False)) or bool(
        speaker.get("requires_perspective_taking", False)
    )
    pragmatic_goal = any(word in goal for word in ("pragmatic", "intention", "speaker", "irony", "indirect"))
    cues = _words(situation)
    utterance = str(linguistic.get("said_or_encoded", {}).get("utterance", "")).lower()
    conventional_indirect = "it is cold" in utterance or "could you" in utterance or "would you" in utterance
    recruited = explicit_need or pragmatic_goal or conventional_indirect
    if not recruited:
        return {
            "recruited": False, "intention_hypotheses": [],
            "most_likely_intention": None, "confidence": 0.0,
            "reason": "no task or contextual requirement for intention inference",
        }
    options = list(event.speaker_model.get("candidate_intentions", ())) or [
        "inform", "request_action", "ask_question", "express_attitude"
    ]
    hint = str(event.speaker_model.get("intention_hint", "")).lower()
    rows: list[dict[str, Any]] = []
    for option in options:
        normalized = str(option).lower()
        score = 0.0
        if normalized and normalized in hint:
            score += 1.5
        keywords = {
            "request_action": {"window", "close", "open", "please", "could", "would"},
            "inform": {"is", "are", "was", "means", "because"},
            "ask_question": {"what", "why", "when", "where", "who", "how"},
            "express_attitude": {"unfortunately", "wonderful", "terrible", "cold"},
        }.get(normalized, set())
        score += 0.25 * len(keywords & _words(utterance))
        score += 0.15 * len(cues & _words(speaker.get("goals", ())))
        rows.append({"intention": normalized, "score": score})
    probabilities = softmax([row["score"] for row in rows], 0.6)
    for row, probability in zip(rows, probabilities):
        row["probability"] = probability
    rows.sort(key=lambda item: item["probability"], reverse=True)
    confidence = rows[0]["probability"] if rows else 0.0
    return {
        "recruited": True, "intention_hypotheses": rows,
        "most_likely_intention": rows[0]["intention"] if rows else None,
        "confidence": confidence,
        "uncertainty_nats": shannon_entropy([row["probability"] for row in rows]),
        "above_confidence_threshold": confidence >= config.pragmatic_threshold,
        "literal_meaning_preserved": True,
        "mental_state_estimate_is_uncertain": True,
    }


def integrate_interacting_systems(
    linguistic: Mapping[str, Any],
    concepts: Mapping[str, Any],
    demand: Mapping[str, Any],
    control: Mapping[str, Any],
    reasoning: Mapping[str, Any],
    pragmatic: Mapping[str, Any],
    event: LanguageEvent,
) -> dict[str, Any]:
    """Coordinate subsystem outputs while retaining separate provenance and inference labels."""
    explicit = dict(linguistic.get("said_or_encoded", {}))
    retrieved = list(concepts.get("activated_concepts", ()))
    inferred = list(reasoning.get("inferred", ()))
    pragmatic_rows = list(pragmatic.get("intention_hypotheses", ()))
    background_dependent = bool(retrieved or inferred or pragmatic_rows)
    return {
        "linguistic_representation": linguistic.get("sentence_meaning", {}),
        "explicit_content": explicit,
        "conceptual_knowledge": retrieved,
        "control_support": dict(control),
        "reasoning_conclusions": inferred,
        "pragmatic_hypotheses": pragmatic_rows,
        "task_goal": event.task_goal,
        "background_assumptions_used": background_dependent,
        "provenance_map": {
            "explicit_content": "sensory_and_linguistic_input",
            "conceptual_knowledge": "retrieved_long_term_knowledge",
            "reasoning_conclusions": "domain_general_inference",
            "pragmatic_hypotheses": "speaker_and_situational_inference",
            "control_support": "domain_general_capacity_allocation",
        },
        "system_separation_preserved": True,
        "control_demand_score": demand.get("demand_score", 0.0),
    }


def detect_mismatch_and_revise(
    event: LanguageEvent,
    integrated: Mapping[str, Any],
    config: LanguageProcessingConfig,
) -> dict[str, Any]:
    """Classify evidence mismatch and revise only the representation layer affected."""
    predicted = set(_words(event.predicted_response))
    observed = set(_words(event.observed_response))
    mismatch_types: list[str] = []
    residual = 0.0
    next_token_residual = 0.0
    if predicted and observed and predicted != observed:
        residual = 1.0 - len(predicted & observed) / max(1, len(predicted | observed))
        mismatch_types.append("response_or_pragmatic")
    predicted_next = set(_words(event.expected_next_tokens))
    observed_next = set(_words(event.new_evidence.get("observed_next_tokens", ())))
    if predicted_next and observed_next and predicted_next != observed_next:
        next_token_residual = 1.0 - (
            len(predicted_next & observed_next) / max(1, len(predicted_next | observed_next))
        )
        mismatch_types.append("next_input_prediction")
    if event.new_evidence.get("perceptual_conflict"):
        mismatch_types.append("perceptual")
    if event.new_evidence.get("lexical_correction"):
        mismatch_types.append("lexical")
    if event.new_evidence.get("structural_correction"):
        mismatch_types.append("structural")
    if event.new_evidence.get("conceptual_correction"):
        mismatch_types.append("conceptual")
    if event.new_evidence.get("pragmatic_correction"):
        mismatch_types.append("pragmatic")
    detected = residual >= config.revision_threshold or bool(mismatch_types)
    revisions: list[dict[str, Any]] = []
    if detected:
        for kind in dict.fromkeys(mismatch_types):
            if kind == "response_or_pragmatic":
                correction = event.new_evidence.get("response_correction")
            elif kind == "next_input_prediction":
                correction = list(event.new_evidence.get("observed_next_tokens", ()))
            else:
                correction = event.new_evidence.get(f"{kind}_correction")
            revisions.append({
                "layer": kind,
                "correction": correction,
                "action": "reopen_affected_candidates_and_reassess",
            })
    return {
        "mismatch_detected": detected,
        "mismatch_types": list(dict.fromkeys(mismatch_types)),
        "response_residual": residual,
        "next_token_prediction_residual": next_token_residual,
        "prediction_error_is_layer_specific": True,
        "revisions": revisions[:config.max_revisions],
        "original_explicit_content_preserved": True,
        "updated_uncertainty": clamp(
            float(integrated.get("control_demand_score", 0.0)) * 0.25
            + max(residual, next_token_residual) * 0.5 + (0.2 if detected else 0.0)
        ),
    }


def select_task_output(
    event: LanguageEvent,
    integrated: Mapping[str, Any],
    mismatch: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a comprehension, answer, response, or action plan without unsupported elaboration."""
    goal = event.task_goal.lower()
    explicit = integrated.get("explicit_content", {})
    sentence = explicit.get("utterance", "")
    if any(word in goal for word in ("act", "follow_instruction", "execute")):
        mode = "action_plan"
        result: Any = {
            "instruction_text": sentence,
            "action_candidates": list(event.metadata.get("action_candidates", ())),
            "requires_confirmation": bool(mismatch.get("mismatch_detected")),
        }
    elif any(word in goal for word in ("answer", "respond")):
        mode = "response"
        result = event.metadata.get("response_text", sentence)
    else:
        mode = "comprehension"
        result = {
            "linguistic_content": sentence,
            "explicit_tokens": list(explicit.get("tokens", ())),
            "unresolved_uncertainty": bool(mismatch.get("mismatch_detected")),
        }
    return {
        "mode": mode, "result": result,
        "response_must_not_exceed_evidence": True,
        "overt_response_required": mode != "comprehension",
    }


def update_learned_representations(
    event: LanguageEvent,
    state: LanguageLearnerState,
    linguistic: Mapping[str, Any],
    concepts: Mapping[str, Any],
    config: LanguageProcessingConfig,
) -> dict[str, Any]:
    """Update lexical/construction strengths gradually from reliable, repeated feedback only."""
    feedback = event.feedback
    kind = str(feedback.get("kind", ""))
    reliability = clamp(float(feedback.get("reliability", 0.0)))
    repetitions = max(0, int(feedback.get("repetitions", 1)))
    token = str(feedback.get("token", "")).lower()
    construction = str(feedback.get("construction", "")).lower()
    correct = bool(feedback.get("correct", False))
    eligible = (
        kind in {"correction", "confirmation", "repeated_exposure"}
        and reliability >= config.learning_reliability_threshold
        and repetitions >= config.repeated_evidence_threshold
    )
    updates: list[dict[str, Any]] = []
    if eligible and token:
        current = state.lexical_strengths.get(token, 0.5)
        target = 1.0 if correct else 0.0
        repetition_weight = min(1.0, repetitions / (repetitions + 2.0))
        updated = clamp(current + config.learning_rate * reliability
                        * repetition_weight * (target - current))
        state.lexical_strengths[token] = updated
        key = "lexical:" + token
        state.evidence_counts[key] = state.evidence_counts.get(key, 0) + repetitions
        updates.append({"kind": "lexical", "key": token, "before": current,
                        "after": updated, "repetitions": repetitions})
    if eligible and construction:
        current = state.construction_strengths.get(construction, 0.5)
        target = 1.0 if correct else 0.0
        updated = clamp(current + config.learning_rate * reliability
                        * min(1.0, repetitions / (repetitions + 2.0)) * (target - current))
        state.construction_strengths[construction] = updated
        key = "construction:" + construction
        state.evidence_counts[key] = state.evidence_counts.get(key, 0) + repetitions
        updates.append({"kind": "construction", "key": construction,
                        "before": current, "after": updated, "repetitions": repetitions})
    previous_activation = state.language_activation
    observed_activation = clamp(float(
        linguistic.get("language_state", {}).get("activation", previous_activation)
    ))
    state.language_activation = clamp(0.70 * previous_activation + 0.30 * observed_activation)
    state.process_count += 1
    state.episode_history.append({
        "event_id": event.event_id, "task_goal": event.task_goal,
        "feedback_kind": kind or "none", "learning_eligible": eligible,
        "updated_keys": [row["kind"] + ":" + row["key"] for row in updates],
    })
    return {
        "eligible": eligible,
        "reason": "reliable_repeated_evidence" if eligible else "insufficient_reliability_or_repetition",
        "updates": updates,
        "updated_lexicon": dict(state.lexical_strengths),
        "construction_strengths": dict(state.construction_strengths),
        "language_activation_before": previous_activation,
        "language_activation_after": state.language_activation,
        "one_anomalous_event_does_not_rewrite_a_general_rule": True,
    }
