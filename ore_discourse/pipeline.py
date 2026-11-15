"""Functional eight-stage discourse integration pipeline for ORE.

The parser is deliberately conservative and evidence-aware. Caller-provided
propositions, entities, temporal data and prosody take precedence over lexical
heuristics; unsupported mental states or world facts remain unresolved.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math
import re
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .biology import DiscourseLIFSimulator
from .contracts import DiscourseConfig, DiscourseRequest, DiscourseResult
from .registry import OPERATION_REGISTRY, STAGE_REGISTRY


_TOKEN_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)*|\d+(?:[.,]\d+)*|[^\w\s]", re.UNICODE)
_WORD_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)
_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "of", "to", "in",
    "on", "at", "by", "for", "from", "with", "as", "is", "am", "are", "was", "were",
    "be", "been", "being", "do", "does", "did", "have", "has", "had", "i", "you",
    "he", "she", "it", "we", "they", "me", "him", "her", "us", "them", "my", "your",
    "his", "its", "our", "their", "this", "that", "these", "those", "who", "what",
    "which", "when", "where", "why", "how", "not", "no", "so", "very", "just",
}
_POSITIVE = {
    "good", "great", "excellent", "wonderful", "brilliant", "perfect", "fantastic",
    "amazing", "love", "like", "happy", "success", "successful", "beautiful", "nice",
}
_NEGATIVE = {
    "bad", "terrible", "awful", "horrible", "broken", "failed", "failure", "disaster",
    "wrong", "hate", "sad", "angry", "painful", "useless", "stupid", "late", "lost",
}
_NEGATIONS = {"not", "no", "never", "neither", "n't", "cannot"}
_MODALS = {"may", "might", "could", "would", "should", "must", "can", "will"}
_TEMPORAL = {
    "before": "before", "after": "after", "afterward": "after", "subsequently": "after", "then": "after", "later": "after",
    "earlier": "before", "previously": "before", "while": "overlap",
    "during": "overlap", "meanwhile": "overlap", "simultaneously": "overlap",
    "first": "before", "finally": "after", "next": "after",
}
_CAUSAL = {
    "because": "cause", "since": "cause", "therefore": "result", "thus": "result",
    "consequently": "result", "hence": "result", "so": "result",
}
_CONNECTIVES = {
    "but": "contrast", "however": "contrast", "although": "contrast",
    "though": "contrast", "yet": "contrast", "because": "explanation",
    "since": "explanation", "therefore": "result", "thus": "result",
    "for example": "elaboration", "for instance": "elaboration", "also": "elaboration",
    "moreover": "elaboration", "furthermore": "elaboration", "then": "sequence",
    "next": "sequence", "after": "sequence", "before": "sequence",
    "if": "condition", "unless": "condition",
}
_REQUEST_RE = re.compile(
    r"\b(?:could|can|would|will)\s+you\s+(?:please\s+)?(.+?)[?.!]*$",
    re.IGNORECASE,
)
_MIND_RE = re.compile(r"\bwould\s+you\s+mind\s+(.+?)[?.!]*$", re.IGNORECASE)
_ASK_RE = re.compile(r"\b(?:could|can)\s+i\s+ask\s+you\s+to\s+(.+?)[?.!]*$", re.IGNORECASE)
_PRONOUNS = {"he", "she", "they", "him", "her", "them", "it", "this", "that", "these", "those"}
_TIME_WORDS = {
    "now", "today", "yesterday", "tomorrow", "tonight", "currently", "later",
    "earlier", "before", "after", "then", "meanwhile", "during", "while", "first",
    "finally", "next", "previously", "recently", "soon",
}
_PREDICATE_CUES = {
    "be", "is", "am", "are", "was", "were", "become", "became", "remain", "stays",
    "go", "went", "come", "came", "arrive", "arrived", "leave", "left", "open",
    "opened", "close", "closed", "break", "broke", "broken", "fail", "failed",
    "win", "won", "lose", "lost", "build", "built", "make", "made", "say", "said",
    "tell", "told", "ask", "asked", "think", "thought", "know", "knew", "believe",
    "want", "wanted", "hope", "hoped", "plan", "planned", "need", "needed",
    "give", "gave", "take", "took", "eat", "ate", "work", "worked", "run", "ran",
    "help", "helped", "rain", "rained", "snow", "snowed", "happen", "happened",
}
_PERSON_TYPES = {"person", "people", "human", "speaker", "listener", "agent"}


def _tokens(text: str) -> list[str]:
    """Return Unicode-aware token strings while preserving punctuation boundaries."""
    return _TOKEN_RE.findall(text)


def _content_features(text: str) -> Counter[str]:
    """Count normalized content words for interpretable topic and continuity vectors."""
    words = [word.casefold() for word in _WORD_RE.findall(text)]
    return Counter(word for word in words if word not in _STOPWORDS and len(word) > 1)


def _cosine(left: Mapping[str, float], right: Mapping[str, float], epsilon: float = 1e-8) -> float:
    """Calculate sparse-vector cosine similarity with an explicit zero-vector guard."""
    keys = set(left) | set(right)
    if not keys:
        return 0.0
    dot = sum(float(left.get(key, 0.0)) * float(right.get(key, 0.0)) for key in keys)
    norm_left = math.sqrt(sum(float(left.get(key, 0.0)) ** 2 for key in keys))
    norm_right = math.sqrt(sum(float(right.get(key, 0.0)) ** 2 for key in keys))
    return float(np.clip(dot / (norm_left * norm_right + epsilon), 0.0, 1.0))


def _softmax(scores: Mapping[str, float], temperature: float) -> dict[str, float]:
    """Normalize named hypothesis scores into a stable temperature-scaled distribution."""
    if not scores:
        return {}
    keys = list(scores)
    values = np.asarray([float(scores[key]) for key in keys], dtype=np.float64)
    values = (values - np.max(values)) / max(temperature, 1e-8)
    exponentials = np.exp(np.clip(values, -60.0, 60.0))
    probabilities = exponentials / max(float(exponentials.sum()), 1e-12)
    return {key: float(value) for key, value in zip(keys, probabilities)}


def _sigmoid(value: float) -> float:
    """Map a finite or bounded scalar score to a numerically stable unit interval."""
    clipped = float(np.clip(value, -40.0, 40.0))
    return 1.0 / (1.0 + math.exp(-clipped))


def _json_value(value: Any) -> Any:
    """Convert nested inference output into JSON-compatible scalar and container values."""
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _safe_number(value: Any, default: float = 0.0) -> float:
    """Convert an arbitrary caller value into a finite float or a safe default."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _sentence_texts(request: DiscourseRequest) -> list[str]:
    """Resolve the discourse surface strings from explicit sentences or proposition records."""
    if request.sentences is not None and len(request.sentences) > 0:
        return list(request.sentences)
    indexed: dict[int, str] = {}
    for fallback, item in enumerate(request.propositions):
        index = int(item.get("sentence_index", item.get("index", fallback)))
        text = str(item.get("text", item.get("sentence", ""))).strip()
        if not text:
            predicate = str(item.get("predicate", "event")).strip()
            args = item.get("arguments", {})
            if isinstance(args, Mapping):
                participants = " ".join(str(value) for value in args.values())
            elif isinstance(args, Sequence) and not isinstance(args, str):
                participants = " ".join(str(value) for value in args)
            else:
                participants = ""
            text = f"{predicate} {participants}".strip()
        indexed[index] = text or f"proposition {fallback + 1}"
    if not indexed:
        return []
    return [indexed[index] for index in range(max(indexed) + 1)]


def _entity_record(raw: Mapping[str, Any], fallback_index: int) -> dict[str, Any]:
    """Normalize a caller entity record while retaining original metadata and provenance."""
    label = str(raw.get("label", raw.get("name", raw.get("text", "")))).strip()
    entity_id = str(raw.get("entity_id", raw.get("id", label.casefold().replace(" ", "_")))).strip()
    aliases = raw.get("aliases", ())
    if isinstance(aliases, str):
        aliases = (aliases,)
    aliases = tuple(str(alias).strip() for alias in aliases if str(alias).strip())
    if label and label not in aliases:
        aliases = (label, *aliases)
    return {
        "entity_id": entity_id or f"entity-{fallback_index + 1}",
        "label": label or f"entity-{fallback_index + 1}",
        "entity_type": str(raw.get("entity_type", raw.get("type", "unknown"))).casefold(),
        "aliases": aliases,
        "semantic_features": dict(raw.get("semantic_features", {})),
        "source_reference": raw.get("source_reference"),
        "salience": float(np.clip(_safe_number(raw.get("salience", 0.5), 0.5), 0.0, 1.0)),
        "metadata": dict(raw),
    }


def _resolve_sentence_propositions(
    sentences: Sequence[str],
    explicit: Sequence[Mapping[str, Any]],
    entity_mentions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Create sentence-level propositions, preferring structured caller-supplied records."""
    explicit_by_index: dict[int, dict[str, Any]] = {}
    for fallback, raw in enumerate(explicit):
        index = int(raw.get("sentence_index", raw.get("index", fallback)))
        explicit_by_index[index] = dict(raw)
    mentions_by_sentence: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for mention in entity_mentions:
        mentions_by_sentence[int(mention["sentence_index"])].append(dict(mention))
    propositions: list[dict[str, Any]] = []
    for index, sentence in enumerate(sentences):
        raw = dict(explicit_by_index.get(index, {}))
        words = [word.casefold() for word in _WORD_RE.findall(sentence)]
        predicate = str(raw.get("predicate") or next(
            (word for word in words if word in _PREDICATE_CUES), words[0] if words else "utterance"
        ))
        negated = bool(raw.get("negated", any(word in _NEGATIONS for word in words)))
        arguments = raw.get("arguments", {})
        if not arguments:
            matched = mentions_by_sentence.get(index, [])
            arguments = {
                f"participant_{position + 1}": item["entity_id"]
                for position, item in enumerate(matched[:4])
            }
        modality = str(raw.get("modality", next(
            (word for word in words if word in _MODALS), "asserted"
        )))
        propositions.append({
            **raw,
            "proposition_id": str(raw.get("proposition_id", raw.get("id", f"prop-{index + 1}"))),
            "sentence_index": index,
            "text": str(raw.get("text", sentence)),
            "predicate": predicate,
            "arguments": dict(arguments) if isinstance(arguments, Mapping) else list(arguments),
            "negated": negated,
            "modality": modality,
            "polarity": str(raw.get("polarity", "negative" if negated else "positive")),
            "topic_features": dict(_content_features(sentence)),
        })
    return propositions


def _connective_relation(text: str) -> tuple[str | None, str | None]:
    """Return the first explicit discourse connective and its coarse relation label."""
    normalized = text.casefold()
    for connective in sorted(_CONNECTIVES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(connective)}\b", normalized):
            return connective, _CONNECTIVES[connective]
    return None, None


def _op_signal(record: Mapping[str, Any]) -> float:
    """Read the normalized scalar drive assigned to one functional operation."""
    return float(np.clip(_safe_number(record.get("drive", 0.0)), 0.0, 1.0))


class DiscourseProcessor:
    """Run the eight-stage, 64-operation discourse and pragmatic integration model.

    The deterministic functional track maintains explicit evidence channels. An
    optional coupled LIF reference is advanced in registry order and reported
    separately; neither track is claimed to reproduce a complete human brain.
    """

    def __init__(self, config: DiscourseConfig | None = None) -> None:
        """Create the processor with validated numerical and biological settings."""
        self.config = config or DiscourseConfig()

    def process(
        self,
        request: DiscourseRequest,
        on_operation: Callable[[dict[str, Any]], None] | None = None,
    ) -> DiscourseResult:
        """Execute all 64 operations and return context, intent, event, time and coherence tracks."""
        if not isinstance(request, DiscourseRequest):
            raise TypeError("request must be a DiscourseRequest")
        sentences = _sentence_texts(request)
        if not sentences:
            raise ValueError("the request must resolve to at least one sentence")
        if len(sentences) > self.config.max_sentences:
            raise ValueError(f"request exceeds max_sentences={self.config.max_sentences}")

        stage_outputs: list[list[dict[str, Any]]] = []
        warnings: list[str] = []
        context_state, entities, mentions, resolved_mentions, out = self._stage_context(
            request, sentences
        )
        stage_outputs.append(out)
        propositions, proposition_links, out = self._stage_propositions(
            request, sentences, mentions, resolved_mentions
        )
        stage_outputs.append(out)
        intentions, speaker_state, out = self._stage_intentions(
            request, sentences, propositions, context_state
        )
        stage_outputs.append(out)
        indirect_requests, out = self._stage_indirect_requests(
            request, sentences, context_state
        )
        stage_outputs.append(out)
        irony, out = self._stage_irony(request, sentences, propositions)
        stage_outputs.append(out)
        situation_model, out = self._stage_situation(
            request, entities, propositions, intentions, proposition_links
        )
        stage_outputs.append(out)
        temporal_relations, timeline, temporal_validation, out = self._stage_temporal(
            request, sentences, propositions
        )
        stage_outputs.append(out)
        coherence, out = self._stage_coherence(
            sentences, propositions, proposition_links, situation_model,
            temporal_relations, context_state, entities,
        )
        stage_outputs.append(out)

        simulator = DiscourseLIFSimulator(self.config) if self.config.simulate_biology else None
        trace: list[dict[str, Any]] = []
        for index, operation in enumerate(OPERATION_REGISTRY):
            """Record each completed operation and advance its matching neural-reference population."""
            stage_index = int(operation.stage_id[1:]) - 1
            op_index = index % 8
            functional_output = dict(stage_outputs[stage_index][op_index])
            signal = _op_signal(functional_output)
            neural = (
                simulator.step(operation.operation_id, operation.stage_id, signal)
                if simulator is not None else {
                    "status": "skipped",
                    "reason": "simulate_biology=False",
                    "operation_id": operation.operation_id,
                }
            )
            row = {
                **operation.to_dict(),
                "status": "executed",
                "functional_output": _json_value(functional_output),
                "biological": _json_value(neural),
            }
            trace.append(row)
            if on_operation is not None:
                on_operation(dict(row))

        biological = simulator.snapshot() if simulator is not None else {
            "model": "disabled",
            "operation_count": 0,
            "scope": "The biological-reference track was disabled by configuration.",
        }
        coherence["timeline"] = timeline
        diagnostics = {
            "valid": len(trace) == 64 and len({row["operation_id"] for row in trace}) == 64,
            "operation_count": len(trace),
            "expected_operation_count": 64,
            "stage_count": 8,
            "sentence_count": len(sentences),
            "entity_count": len(entities),
            "proposition_count": len(propositions),
            "temporal_constraints_valid": temporal_validation["valid"],
            "unresolved_references": [
                dict(item) for item in resolved_mentions if item.get("status") == "unresolved"
            ],
            "warnings": warnings,
            "scientific_scope": (
                "Software behavior is testable; behavioral and biological validity require "
                "independent datasets and experiments."
            ),
        }
        metadata = {
            "schema": "ore.discourse.eight_stage.v1",
            "language": request.language,
            "source_reference": request.source_reference,
            "operation_registry": "ORE-DISCOURSE-64",
            "configuration": self.config.to_dict(),
            "inference_is_read_only": True,
            "caller_facts_separated_from_heuristics": True,
        }
        return DiscourseResult(
            sequence_id=request.sequence_id,
            sentences=list(sentences),
            propositions=propositions,
            discourse_context=context_state,
            proposition_links=proposition_links,
            speaker_intentions=intentions,
            indirect_requests=indirect_requests,
            irony=irony,
            situation_model=situation_model,
            temporal_relations=temporal_relations,
            coherence=coherence,
            biological=biological,
            operation_trace=trace,
            diagnostics=diagnostics,
            metadata=metadata,
        )

    def _stage_context(
        self, request: DiscourseRequest, sentences: Sequence[str]
    ) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        """Maintain prior-turn memory, entities, coreference candidates and common ground."""
        prior_turns = list(request.context.get("prior_turns", []))
        episodic = dict(request.context.get("episodic_context", {}))
        entities = [_entity_record(raw, i) for i, raw in enumerate(request.entities)]
        by_id = {entity["entity_id"]: entity for entity in entities}
        mentions: list[dict[str, Any]] = []
        for sentence_index, sentence in enumerate(sentences):
            for entity in entities:
                for alias in entity["aliases"]:
                    if alias and re.search(rf"\b{re.escape(alias)}\b", sentence, re.IGNORECASE):
                        mentions.append({
                            "entity_id": entity["entity_id"], "surface": alias,
                            "sentence_index": sentence_index, "source": "caller_entity_alias",
                        })
            if not entities:
                for match in re.finditer(r"\b[A-Z][a-z]{2,}\b", sentence):
                    if match.start() == 0:
                        continue
                    label = match.group(0)
                    entity_id = label.casefold().replace(" ", "_")
                    if entity_id not in by_id:
                        entity = _entity_record({"entity_id": entity_id, "label": label}, len(entities))
                        entities.append(entity)
                        by_id[entity_id] = entity
                    mentions.append({
                        "entity_id": entity_id, "surface": label,
                        "sentence_index": sentence_index, "source": "capitalization_heuristic",
                    })
        for entity in entities:
            entity["mention_count"] = sum(1 for item in mentions if item["entity_id"] == entity["entity_id"])
            entity["last_mentioned_sentence"] = max(
                (item["sentence_index"] for item in mentions if item["entity_id"] == entity["entity_id"]),
                default=-1,
            )
            recency = (
                math.exp(-self.config.memory_decay * max(0, len(sentences) - 1 - entity["last_mentioned_sentence"]))
                if entity["last_mentioned_sentence"] >= 0 else 0.0
            )
            frequency = 1.0 - math.exp(-entity["mention_count"])
            topic = 1.0 if any(
                alias.casefold() in " ".join(sentences).casefold() for alias in entity["aliases"]
            ) else 0.0
            entity["salience"] = float(np.clip(
                self.config.salience_recency_gain * recency
                + self.config.salience_frequency_gain * frequency
                + self.config.salience_topic_gain * topic,
                0.0, 1.0,
            ))

        resolved_mentions: list[dict[str, Any]] = []
        for sentence_index, sentence in enumerate(sentences):
            words = [word.casefold() for word in _WORD_RE.findall(sentence)]
            for pronoun in sorted(_PRONOUNS.intersection(words)):
                candidates = [
                    entity for entity in entities
                    if entity["last_mentioned_sentence"] < sentence_index
                    and (
                        pronoun not in {"he", "she", "him", "her", "who"}
                        or entity["entity_type"] in _PERSON_TYPES
                        or entity["entity_type"] == "unknown"
                    )
                ]
                ranked = sorted(
                    candidates,
                    key=lambda item: (
                        0.65 * float(item["salience"])
                        + 0.35 * math.exp(-0.7 * (sentence_index - item["last_mentioned_sentence"]))
                    ),
                    reverse=True,
                )
                if ranked:
                    top = ranked[0]
                    second_score = (
                        0.65 * float(ranked[1]["salience"])
                        + 0.35 * math.exp(-0.7 * (sentence_index - ranked[1]["last_mentioned_sentence"]))
                        if len(ranked) > 1 else 0.0
                    )
                    top_score = (
                        0.65 * float(top["salience"])
                        + 0.35 * math.exp(-0.7 * (sentence_index - top["last_mentioned_sentence"]))
                    )
                    resolved_mentions.append({
                        "surface": pronoun, "sentence_index": sentence_index,
                        "entity_id": top["entity_id"], "confidence": float(np.clip(top_score - 0.25 * second_score, 0, 1)),
                        "status": "resolved" if top_score - second_score >= 0.05 else "ambiguous",
                        "alternatives": [item["entity_id"] for item in ranked[:3]],
                        "method": "recency_salience_compatibility",
                    })
                else:
                    resolved_mentions.append({
                        "surface": pronoun, "sentence_index": sentence_index,
                        "entity_id": None, "confidence": 0.0, "status": "unresolved",
                        "alternatives": [], "method": "no_supported_antecedent",
                    })

        common_ground = dict(request.common_ground)
        declared_facts = list(common_ground.get("facts", [])) if isinstance(common_ground.get("facts", []), list) else []
        context_state = {
            "prior_turns": prior_turns,
            "episodic_context": episodic,
            "semantic_memory": dict(request.context.get("semantic_memory", {})),
            "common_ground": common_ground,
            "declared_common_ground_facts": declared_facts,
            "entities": entities[: self.config.max_entities],
            "entity_mentions": mentions,
            "coreference": resolved_mentions,
            "active_information": {
                "given_entities": sorted({item["entity_id"] for item in mentions}),
                "new_entity_ids": sorted({
                    item["entity_id"] for item in mentions
                    if not any(
                        str(fact).casefold() in str(item["surface"]).casefold()
                        for fact in declared_facts
                    )
                }),
                "prior_turn_count": len(prior_turns),
            },
            "memory_policy": "read_only_input_plus_bounded_current_discourse_state",
        }
        signals = [
            {"value": len(context_state), "drive": 0.45},
            {"value": len(prior_turns), "drive": float(np.clip(len(prior_turns) / 5, 0, 1))},
            {"value": len(mentions), "drive": float(np.clip(len(mentions) / 5, 0, 1))},
            {"value": len(resolved_mentions), "drive": float(np.mean([x["confidence"] for x in resolved_mentions]) if resolved_mentions else 0.0)},
            {"value": max((x["salience"] for x in entities), default=0.0), "drive": max((x["salience"] for x in entities), default=0.0)},
            {"value": len(declared_facts), "drive": float(np.clip(len(declared_facts) / 5, 0, 1))},
            {"value": len(context_state["active_information"]["given_entities"]), "drive": float(np.clip(len(context_state["active_information"]["given_entities"]) / 5, 0, 1))},
            {"value": len(entities), "drive": float(np.clip(len(entities) / max(1, self.config.max_entities), 0, 1))},
        ]
        return context_state, entities, mentions, resolved_mentions, signals

    def _stage_propositions(
        self,
        request: DiscourseRequest,
        sentences: Sequence[str],
        mentions: Sequence[Mapping[str, Any]],
        resolved_mentions: Sequence[Mapping[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        """Integrate sentence propositions through entity overlap, connectives and inference."""
        propositions = _resolve_sentence_propositions(sentences, request.propositions, mentions)
        for resolved in resolved_mentions:
            if resolved.get("entity_id") and resolved.get("status") == "resolved":
                index = int(resolved["sentence_index"])
                if 0 <= index < len(propositions):
                    arguments = dict(propositions[index].get("arguments", {}))
                    arguments.setdefault("coreferent", resolved["entity_id"])
                    propositions[index]["arguments"] = arguments
        links: list[dict[str, Any]] = []
        for index in range(1, len(propositions)):
            previous, current = propositions[index - 1], propositions[index]
            left_entities = set(str(value) for value in previous.get("arguments", {}).values()) if isinstance(previous.get("arguments"), Mapping) else set(previous.get("arguments", []))
            right_entities = set(str(value) for value in current.get("arguments", {}).values()) if isinstance(current.get("arguments"), Mapping) else set(current.get("arguments", []))
            entity_overlap = len(left_entities & right_entities) / max(1, len(left_entities | right_entities))
            topic_similarity = _cosine(previous.get("topic_features", {}), current.get("topic_features", {}))
            connective, relation = _connective_relation(str(current["text"]))
            if relation is None:
                relation = "continuation" if entity_overlap > 0 or topic_similarity > 0.25 else "topic_shift_candidate"
            score = float(np.clip(0.55 * entity_overlap + 0.45 * topic_similarity, 0, 1))
            links.append({
                "source": previous["proposition_id"], "target": current["proposition_id"],
                "source_sentence": index - 1, "target_sentence": index,
                "relation": relation, "connective": connective,
                "entity_overlap": entity_overlap, "topic_similarity": topic_similarity,
                "continuity_score": score, "confidence": float(np.clip(score + (0.18 if connective else 0.0), 0, 1)),
                "evidence": "explicit_connective_plus_surface_feature_overlap" if connective else "surface_feature_overlap",
            })
        bridging = [
            item for item in resolved_mentions
            if item.get("status") == "resolved" and item.get("sentence_index", 0) > 0
        ]
        for item in bridging:
            links.append({
                "source_sentence": int(item["sentence_index"]) - 1,
                "target_sentence": int(item["sentence_index"]),
                "relation": "bridging_reference",
                "entity_id": item["entity_id"],
                "confidence": float(item["confidence"]),
                "evidence": "resolved_pronoun_reference",
            })
        mean_continuity = float(np.mean([item["continuity_score"] for item in links if "continuity_score" in item])) if links else 0.5
        signals = [
            {"value": len(propositions), "drive": float(np.clip(len(propositions) / 5, 0, 1))},
            {"value": len(links), "drive": float(np.clip(len(links) / 5, 0, 1))},
            {"value": sum(bool(item.get("connective")) for item in links), "drive": float(np.mean([0.9 if item.get("connective") else 0.0 for item in links]) if links else 0.0)},
            {"value": mean_continuity, "drive": mean_continuity},
            {"value": len([x for x in links if x["relation"] not in {"continuation", "topic_shift_candidate"}]), "drive": float(np.clip(len(links) / 4, 0, 1))},
            {"value": len(bridging), "drive": float(np.clip(len(bridging) / 3, 0, 1))},
            {"value": len(links), "drive": float(np.clip(len(links) / 5, 0, 1))},
            {"value": mean_continuity, "drive": mean_continuity},
        ]
        return propositions, links, signals

    def _stage_intentions(
        self,
        request: DiscourseRequest,
        sentences: Sequence[str],
        propositions: Sequence[Mapping[str, Any]],
        context_state: Mapping[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
        """Infer communicative acts, goals and speaker-state hypotheses without claiming certainty."""
        declared = dict(request.speaker_model)
        beliefs = dict(declared.get("beliefs", {}))
        declared_goals = list(declared.get("goals", []))
        output: list[dict[str, Any]] = []
        all_goals = list(declared_goals)
        act_counts: Counter[str] = Counter()
        for index, sentence in enumerate(sentences):
            lower = sentence.casefold().strip()
            words = [word.casefold() for word in _WORD_RE.findall(lower)]
            question = sentence.rstrip().endswith("?")
            imperative = bool(words and words[0] in {
                "please", "open", "close", "send", "tell", "give", "take", "help",
                "look", "listen", "wait", "stop", "bring", "call", "make", "read",
            })
            if re.search(r"\b(i promise|i will|i swear)\b", lower):
                act = "promise"
            elif re.search(r"\b(i am sorry|i apologize|sorry)\b", lower):
                act = "apology"
            elif re.search(r"\b(hello|hi|good morning|good evening)\b", lower):
                act = "greeting"
            elif question:
                act = "question"
            elif imperative:
                act = "command"
            elif re.search(r"\b(please|could you|would you|can you)\b", lower):
                act = "request"
            else:
                act = "assertion"
            act_scores = {
                "assertion": 0.35, "question": 0.30, "request": 0.28,
                "command": 0.12, "promise": 0.08, "apology": 0.08, "greeting": 0.08,
            }
            act_scores[act] += 0.95
            if "please" in words:
                act_scores["request"] += 0.22
            if imperative:
                act_scores["command"] += 0.25
            distribution = _softmax(act_scores, self.config.intention_temperature)
            goals: list[str] = []
            for cue, label in (
                ("want", "achieve_desired_state"), ("hope", "achieve_desired_state"),
                ("need", "obtain_needed_resource"), ("plan", "complete_planned_action"),
                ("intend", "perform_intended_action"), ("ask", "obtain_information_or_action"),
                ("please", "influence_listener_action"),
            ):
                if re.search(rf"\b{cue}\w*\b", lower):
                    goals.append(label)
            all_goals.extend(goals)
            mental_cues = [
                cue for cue in ("think", "believe", "know", "want", "hope", "fear", "plan", "intend", "remember")
                if re.search(rf"\b{cue}\w*\b", lower)
            ]
            hedges = sum(word in {"maybe", "perhaps", "probably", "possibly", "i think", "i guess"} for word in words)
            assertive = 1.0 if act == "assertion" and not hedges else 0.65
            commitment = float(np.clip(assertive - 0.12 * hedges - 0.12 * sum(word in _MODALS for word in words), 0, 1))
            ranked = sorted(distribution.items(), key=lambda pair: pair[1], reverse=True)
            margin = ranked[0][1] - ranked[1][1]
            output.append({
                "sentence_index": index, "communicative_act": act,
                "act_distribution": distribution, "goals": list(dict.fromkeys(goals)),
                "beliefs": beliefs, "mental_state_cues": mental_cues,
                "epistemic_commitment": commitment,
                "ranked_intentions": [{"label": key, "probability": value} for key, value in ranked],
                "margin": margin, "abstain": margin < 0.10,
                "evidence": "surface_form_and_explicit_speaker_model",
            })
            act_counts[act] += 1
        speaker_state = {
            "declared_beliefs": beliefs,
            "declared_goals": declared_goals,
            "inferred_goal_hypotheses": list(dict.fromkeys(all_goals)),
            "communicative_act_counts": dict(act_counts),
            "mental_state_inference_scope": "surface_evidence_only",
            "common_ground_available": bool(context_state.get("common_ground")),
        }
        mean_commitment = float(np.mean([item["epistemic_commitment"] for item in output])) if output else 0.0
        signals = [
            {"value": len(beliefs), "drive": float(np.clip(len(beliefs) / 5, 0, 1))},
            {"value": len(output), "drive": float(np.clip(len(output) / 5, 0, 1))},
            {"value": len(all_goals), "drive": float(np.clip(len(all_goals) / 5, 0, 1))},
            {"value": sum(len(item["mental_state_cues"]) for item in output), "drive": float(np.clip(sum(len(item["mental_state_cues"]) for item in output) / 5, 0, 1))},
            {"value": bool(context_state.get("common_ground")), "drive": 0.5 if context_state.get("common_ground") else 0.0},
            {"value": mean_commitment, "drive": mean_commitment},
            {"value": max((item["ranked_intentions"][0]["probability"] for item in output), default=0.0), "drive": max((item["ranked_intentions"][0]["probability"] for item in output), default=0.0)},
            {"value": sum(item["abstain"] for item in output), "drive": float(np.mean([item["abstain"] for item in output]) if output else 0.0)},
        ]
        return output, speaker_state, signals

    def _stage_indirect_requests(
        self, request: DiscourseRequest, sentences: Sequence[str], context_state: Mapping[str, Any]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Separate literal form from listener-directed action hypotheses for each utterance."""
        requests: list[dict[str, Any]] = []
        for index, sentence in enumerate(sentences):
            lower = sentence.casefold().strip()
            match = _REQUEST_RE.search(sentence)
            mind_match = _MIND_RE.search(sentence)
            ask_match = _ASK_RE.search(sentence)
            action_text = None
            construction = None
            strength = 0.0
            if ask_match:
                action_text, construction, strength = ask_match.group(1), "permission_preface", 0.88
            elif mind_match:
                action_text, construction, strength = mind_match.group(1), "willingness_question", 0.90
            elif match:
                action_text, construction, strength = match.group(1), "ability_or_willingness_question", 0.78
            elif re.match(r"^(please\s+)?(open|close|send|give|take|help|bring|call|tell|wait|stop|look|listen)\b", lower):
                action_text, construction, strength = lower, "imperative_request", 0.92
            elif re.search(r"\bit('?s| is) cold\b", lower) and isinstance(request.context.get("scene", {}), Mapping) and request.context.get("scene", {}).get("window_open"):
                action_text, construction, strength = "close the window", "contextual_environmental_hint", 0.58
            literal_type = "question" if sentence.rstrip().endswith("?") else "assertion"
            if construction == "imperative_request":
                literal_type = "imperative"
            if action_text:
                action_text = re.sub(r"^(please\s+)", "", action_text.strip(), flags=re.IGNORECASE)
                action_text = re.sub(r"[?.!]+$", "", action_text).strip()
            abilities = request.context.get("listener_abilities", {})
            feasible = True
            if action_text and isinstance(abilities, Mapping):
                required = action_text.split()[0].casefold()
                if required in abilities:
                    feasible = bool(abilities[required])
            context_gain = 0.12 if request.context.get("listener_goal") else 0.0
            confidence = float(np.clip(strength + context_gain - (0.25 if not feasible else 0.0), 0, 1))
            probability = _sigmoid(5.0 * (confidence - self.config.request_threshold))
            requests.append({
                "sentence_index": index,
                "literal_interpretation": {"type": literal_type, "content": sentence},
                "intended_action": action_text,
                "construction": construction,
                "is_indirect_request": bool(action_text and construction != "imperative_request"),
                "request_probability": probability if action_text else 0.0,
                "directness": 1.0 if construction == "imperative_request" else (0.35 if action_text else 0.0),
                "politeness_cues": ["please"] if "please" in lower else [],
                "listener_action_feasible": feasible if action_text else None,
                "confidence": confidence if action_text else 0.0,
                "preserved_literal_content": True,
                "evidence": "conventional_form_and_optional_listener_context" if action_text else "no_supported_action_hypothesis",
            })
        signals = [
            {"value": sum(bool(item["intended_action"]) for item in requests), "drive": float(np.mean([bool(item["intended_action"]) for item in requests]) if requests else 0.0)},
            {"value": sum(item["construction"] is not None for item in requests), "drive": float(np.mean([item["construction"] is not None for item in requests]) if requests else 0.0)},
            {"value": sum(bool(item["intended_action"]) for item in requests), "drive": float(np.mean([item["request_probability"] for item in requests]) if requests else 0.0)},
            {"value": sum(bool(item["listener_action_feasible"]) for item in requests if item["listener_action_feasible"] is not None), "drive": float(np.mean([bool(item["listener_action_feasible"]) for item in requests if item["listener_action_feasible"] is not None]) if any(item["listener_action_feasible"] is not None for item in requests) else 0.0)},
            {"value": float(np.mean([len(item["politeness_cues"]) for item in requests])) if requests else 0.0, "drive": float(np.mean([bool(item["politeness_cues"]) for item in requests]) if requests else 0.0)},
            {"value": sum(item["literal_interpretation"]["type"] == "question" for item in requests), "drive": float(np.mean([item["literal_interpretation"]["type"] == "question" for item in requests]) if requests else 0.0)},
            {"value": max((item["confidence"] for item in requests), default=0.0), "drive": max((item["confidence"] for item in requests), default=0.0)},
            {"value": len(requests), "drive": 1.0 if requests else 0.0},
        ]
        return requests, signals

    def _stage_irony(
        self, request: DiscourseRequest, sentences: Sequence[str],
        propositions: Sequence[Mapping[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Evaluate semantic, contextual and explicit prosodic incongruity without forcing irony."""
        irony: list[dict[str, Any]] = []
        for index, sentence in enumerate(sentences):
            words = {word.casefold() for word in _WORD_RE.findall(sentence)}
            positive = len(words & _POSITIVE)
            negative = len(words & _NEGATIVE)
            current_valence = positive - negative
            prior_text = " ".join(sentences[:index]).casefold()
            context_negative = any(word in prior_text.split() for word in _NEGATIVE)
            context_positive = any(word in prior_text.split() for word in _POSITIVE)
            semantic_mismatch = 1.0 if (positive and context_negative) or (negative and context_positive) else 0.0
            event_valence = -1.0 if context_negative else (1.0 if context_positive else 0.0)
            contextual_mismatch = float(bool(current_valence and event_valence and current_valence * event_valence < 0))
            prosody = dict(request.prosody.get(index, {}))
            prosodic_score = float(np.clip(
                0.35 * _safe_number(prosody.get("intonation_mismatch", 0.0))
                + 0.25 * _safe_number(prosody.get("exaggeration", 0.0))
                + 0.20 * _safe_number(prosody.get("laughter", 0.0))
                + 0.20 * _safe_number(prosody.get("stress_incongruity", 0.0)),
                0.0, 1.0,
            ))
            explicit_marker = bool(re.search(r"\b(sarcasm|sarcastically|ironic|ironically)\b", sentence, re.IGNORECASE))
            incongruity = float(np.clip(
                0.42 * semantic_mismatch + 0.38 * contextual_mismatch + 0.20 * prosodic_score
                + (0.25 if explicit_marker else 0.0), 0.0, 1.0
            ))
            probability = _sigmoid(5.0 * (incongruity - self.config.irony_threshold))
            if explicit_marker:
                probability = max(probability, 0.90)
            ironic = probability >= self.config.irony_threshold
            subtype = "sarcasm_candidate" if ironic and negative else ("irony_candidate" if ironic else "undetermined")
            irony.append({
                "sentence_index": index,
                "literal_valence": "positive" if current_valence > 0 else ("negative" if current_valence < 0 else "neutral"),
                "contextual_event_valence": "negative" if event_valence < 0 else ("positive" if event_valence > 0 else "unknown"),
                "semantic_mismatch": semantic_mismatch,
                "contextual_mismatch": contextual_mismatch,
                "prosodic_incongruity": prosodic_score,
                "explicit_irony_marker": explicit_marker,
                "incongruity_score": incongruity,
                "irony_probability": probability,
                "interpretation": subtype if ironic else "literal_or_underdetermined",
                "prosody_available": bool(prosody),
                "evidence": ["lexical_valence", "prior_context"] + (["caller_prosody"] if prosody else []),
            })
        signals = [
            {"value": sum(bool(item["literal_valence"] != "neutral") for item in irony), "drive": float(np.mean([item["literal_valence"] != "neutral" for item in irony]) if irony else 0.0)},
            {"value": sum(item["contextual_event_valence"] != "unknown" for item in irony), "drive": float(np.mean([item["contextual_event_valence"] != "unknown" for item in irony]) if irony else 0.0)},
            {"value": float(np.mean([item["semantic_mismatch"] for item in irony])) if irony else 0.0, "drive": float(np.mean([item["semantic_mismatch"] for item in irony]) if irony else 0.0)},
            {"value": float(np.mean([item["prosodic_incongruity"] for item in irony])) if irony else 0.0, "drive": float(np.mean([item["prosodic_incongruity"] for item in irony]) if irony else 0.0)},
            {"value": float(np.mean([item["incongruity_score"] for item in irony])) if irony else 0.0, "drive": float(np.mean([item["incongruity_score"] for item in irony]) if irony else 0.0)},
            {"value": float(np.mean([item["irony_probability"] for item in irony])) if irony else 0.0, "drive": float(np.mean([item["irony_probability"] for item in irony]) if irony else 0.0)},
            {"value": sum(item["interpretation"] == "sarcasm_candidate" for item in irony), "drive": float(np.mean([item["interpretation"] == "sarcasm_candidate" for item in irony]) if irony else 0.0)},
            {"value": sum(item["prosody_available"] for item in irony), "drive": float(np.mean([item["prosody_available"] for item in irony]) if irony else 0.0)},
        ]
        return irony, signals

    def _stage_situation(
        self, request: DiscourseRequest, entities: Sequence[Mapping[str, Any]],
        propositions: Sequence[Mapping[str, Any]], intentions: Sequence[Mapping[str, Any]],
        proposition_links: Sequence[Mapping[str, Any]],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Construct a typed event situation model from people, roles, states, goals and causes."""
        people = [
            {"entity_id": item["entity_id"], "label": item["label"], "type": item["entity_type"]}
            for item in entities if item["entity_type"] in _PERSON_TYPES
        ]
        events: list[dict[str, Any]] = []
        states: list[dict[str, Any]] = []
        for index, proposition in enumerate(propositions):
            arguments = proposition.get("arguments", {})
            if isinstance(arguments, Mapping):
                participants = [{"role": str(role), "entity_id": str(value)} for role, value in arguments.items()]
            elif isinstance(arguments, Sequence) and not isinstance(arguments, str):
                participants = [{"role": f"participant_{n + 1}", "entity_id": str(value)} for n, value in enumerate(arguments)]
            else:
                participants = []
            predicate = str(proposition.get("predicate", "event"))
            event = {
                "event_id": f"event-{index + 1}",
                "proposition_id": proposition["proposition_id"],
                "predicate": predicate,
                "sentence_index": index,
                "participants": participants,
                "polarity": proposition.get("polarity", "positive"),
                "negated": bool(proposition.get("negated", False)),
                "modality": proposition.get("modality", "asserted"),
                "time_evidence": dict(request.temporal_evidence.get(index, {})),
            }
            events.append(event)
            if predicate in {"be", "is", "am", "are", "was", "were", "become", "became", "remain", "stays"}:
                states.append({
                    "state_id": f"state-{index + 1}", "event_id": event["event_id"],
                    "participants": participants, "description": proposition.get("text", ""),
                    "source": "explicit_copular_or_state_predicate",
                })
        declared_goals = list(request.speaker_model.get("goals", []))
        inferred_motivations = [
            {"goal": goal, "source": "declared_or_surface_inferred_goal"}
            for goal in declared_goals
        ]
        for intent in intentions:
            for goal in intent.get("goals", []):
                inferred_motivations.append({
                    "sentence_index": intent["sentence_index"], "goal": goal,
                    "source": "surface_intention_hypothesis",
                })
        causal_links = [
            {
                "source_sentence": item["source_sentence"],
                "target_sentence": item["target_sentence"],
                "relation": item["relation"],
                "confidence": item["confidence"],
            }
            for item in proposition_links
            if item.get("relation") in {"explanation", "result", "cause"}
        ]
        for index, proposition in enumerate(propositions):
            if proposition.get("causes") is not None:
                causal_links.append({
                    "source_sentence": index,
                    "target": proposition.get("causes"),
                    "relation": "explicit_cause_field",
                    "confidence": 1.0,
                    "evidence": "caller_proposition",
                })
        event_ids = {event["event_id"] for event in events}
        valid = all(
            participant["entity_id"]
            for event in events for participant in event["participants"]
        )
        valid = valid and all(
            0 <= int(link["source_sentence"]) < len(propositions)
            and 0 <= int(link["target_sentence"]) < len(propositions)
            for link in causal_links
            if "target_sentence" in link
        )
        situation = {
            "people": people,
            "events": events,
            "states": states,
            "motivations": inferred_motivations,
            "causal_links": causal_links,
            "entity_state_policy": "only_explicit_state_predicates_are_instantiated",
            "validation": {"valid": bool(valid), "event_ids": sorted(event_ids)},
        }
        signals = [
            {"value": len(people), "drive": float(np.clip(len(people) / 5, 0, 1))},
            {"value": len(events), "drive": float(np.clip(len(events) / 5, 0, 1))},
            {"value": sum(len(event["participants"]) for event in events), "drive": float(np.clip(sum(len(event["participants"]) for event in events) / 8, 0, 1))},
            {"value": len(states), "drive": float(np.clip(len(states) / 4, 0, 1))},
            {"value": len(inferred_motivations), "drive": float(np.clip(len(inferred_motivations) / 5, 0, 1))},
            {"value": len(causal_links), "drive": float(np.clip(len(causal_links) / 4, 0, 1))},
            {"value": len(event_ids) + len(causal_links), "drive": float(np.clip((len(event_ids) + len(causal_links)) / 8, 0, 1))},
            {"value": valid, "drive": 1.0 if valid else 0.0},
        ]
        return situation, signals

    def _stage_temporal(
        self, request: DiscourseRequest, sentences: Sequence[str],
        propositions: Sequence[Mapping[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
        """Track order, interval duration, overlap, temporal connectives and narrative time shifts."""
        relations: list[dict[str, Any]] = []
        intervals: list[dict[str, Any]] = []
        shifts: list[dict[str, Any]] = []
        for index, sentence in enumerate(sentences):
            evidence = dict(request.temporal_evidence.get(index, {}))
            words = {word.casefold() for word in _WORD_RE.findall(sentence)}
            expressions = sorted(words & _TIME_WORDS)
            duration = evidence.get("duration")
            if duration is None:
                match = re.search(r"\bfor\s+(\d+(?:\.\d+)?)\s+(seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b", sentence, re.IGNORECASE)
                if match:
                    duration = {"value": float(match.group(1)), "unit": match.group(2).casefold()}
            start = evidence.get("start")
            end = evidence.get("end")
            interval = {
                "event_id": f"event-{index + 1}",
                "sentence_index": index,
                "start": start, "end": end, "duration": duration,
                "time_expression": evidence.get("time_expression") or expressions,
                "anchor": evidence.get("anchor"),
                "tense": evidence.get("tense"),
                "interval_source": "caller_metadata" if evidence else ("surface_temporal_cue" if expressions else "unknown"),
            }
            intervals.append(interval)
            connective, relation = _connective_relation(sentence)
            temporal_relation = next(
                (_TEMPORAL[word] for word in sorted(_TEMPORAL, key=len, reverse=True)
                 if re.search(rf"\b{re.escape(word)}\b", sentence, re.IGNORECASE)),
                None,
            )
            if temporal_relation and index > 0:
                source, target = (index, index - 1) if temporal_relation == "before" else (index - 1, index)
                relations.append({
                    "source_sentence": source, "target_sentence": target,
                    "relation": temporal_relation, "connective": connective or next(
                        (word for word in _TEMPORAL if re.search(rf"\b{word}\b", sentence, re.IGNORECASE)), None
                    ),
                    "confidence": 0.80, "evidence": "explicit_temporal_connective",
                })
            elif index > 0 and start is not None:
                relations.append({
                    "source_sentence": index - 1, "target_sentence": index,
                    "relation": "ordered_by_metadata", "confidence": 0.95,
                    "evidence": "caller_temporal_metadata",
                })
            if expressions and any(word in {"yesterday", "tomorrow", "later", "earlier", "recently", "soon"} for word in expressions):
                shifts.append({
                    "sentence_index": index, "expressions": expressions,
                    "anchor": evidence.get("anchor"), "source": "explicit_temporal_adverb",
                })
        for left_index, left in enumerate(intervals):
            for right in intervals[left_index + 1:]:
                if isinstance(left.get("start"), (int, float)) and isinstance(left.get("end"), (int, float)) and isinstance(right.get("start"), (int, float)) and isinstance(right.get("end"), (int, float)):
                    overlap = max(0.0, min(float(left["end"]), float(right["end"])) - max(float(left["start"]), float(right["start"])))
                    relations.append({
                        "source_sentence": left["sentence_index"],
                        "target_sentence": right["sentence_index"],
                        "relation": "overlap" if overlap > 0 else "non_overlap",
                        "overlap_duration": overlap,
                        "confidence": 1.0, "evidence": "numeric_interval_comparison",
                    })
        before_edges = [
            (int(item["source_sentence"]), int(item["target_sentence"]))
            for item in relations if item["relation"] == "before"
        ]
        order = self._topological_order(len(sentences), before_edges)
        cycle = order is None
        timeline = order if order is not None else list(range(len(sentences)))
        validation = {
            "valid": not cycle and all(
                not (
                    isinstance(item.get("duration"), Mapping)
                    and _safe_number(item["duration"].get("value"), 0.0) < 0
                )
                for item in intervals
            ),
            "cycle_detected": cycle,
            "sentence_order": timeline,
            "unresolved_intervals": sum(item["interval_source"] == "unknown" for item in intervals),
        }
        signals = [
            {"value": len([item for item in intervals if item["time_expression"]]), "drive": float(np.mean([bool(item["time_expression"]) for item in intervals]) if intervals else 0.0)},
            {"value": len(relations), "drive": float(np.clip(len(relations) / 5, 0, 1))},
            {"value": sum(bool(item["duration"]) for item in intervals), "drive": float(np.mean([bool(item["duration"]) for item in intervals]) if intervals else 0.0)},
            {"value": sum(item["relation"] == "overlap" for item in relations), "drive": float(np.mean([item["relation"] == "overlap" for item in relations]) if relations else 0.0)},
            {"value": len([item for item in relations if item.get("connective")]), "drive": float(np.mean([bool(item.get("connective")) for item in relations]) if relations else 0.0)},
            {"value": len(shifts), "drive": float(np.clip(len(shifts) / 4, 0, 1))},
            {"value": len(timeline), "drive": float(np.clip(len(relations) / max(1, len(sentences)), 0, 1))},
            {"value": validation["valid"], "drive": 1.0 if validation["valid"] else 0.0},
        ]
        return relations, intervals, validation, signals

    @staticmethod
    def _topological_order(node_count: int, edges: Sequence[tuple[int, int]]) -> list[int] | None:
        """Return a deterministic topological order or None when temporal constraints cycle."""
        adjacency: dict[int, set[int]] = {index: set() for index in range(node_count)}
        indegree = [0] * node_count
        for source, target in edges:
            if source == target or not (0 <= source < node_count and 0 <= target < node_count):
                return None
            if target not in adjacency[source]:
                adjacency[source].add(target)
                indegree[target] += 1
        ready = [index for index, degree in enumerate(indegree) if degree == 0]
        order: list[int] = []
        while ready:
            node = ready.pop(0)
            order.append(node)
            for target in sorted(adjacency[node]):
                indegree[target] -= 1
                if indegree[target] == 0:
                    ready.append(target)
                    ready.sort()
        return order if len(order) == node_count else None

    def _stage_coherence(
        self, sentences: Sequence[str], propositions: Sequence[Mapping[str, Any]],
        proposition_links: Sequence[Mapping[str, Any]], situation: Mapping[str, Any],
        temporal_relations: Sequence[Mapping[str, Any]], context_state: Mapping[str, Any],
        entities: Sequence[Mapping[str, Any]],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Maintain global topic, narrative progression and consistency across all eight stages."""
        sentence_features = [_content_features(sentence) for sentence in sentences]
        global_counts: Counter[str] = Counter()
        for features in sentence_features:
            global_counts.update(features)
        total = max(1, sum(global_counts.values()))
        topic_distribution = {word: count / total for word, count in global_counts.most_common(30)}
        similarities = [
            _cosine(sentence_features[index - 1], sentence_features[index])
            for index in range(1, len(sentence_features))
        ]
        topic_continuity = float(np.mean(similarities)) if similarities else 0.5
        predicate_counts = Counter(str(item.get("predicate", "")) for item in propositions)
        entity_counts = Counter(
            str(value)
            for proposition in propositions
            for value in (
                proposition.get("arguments", {}).values()
                if isinstance(proposition.get("arguments"), Mapping)
                else proposition.get("arguments", [])
            )
        )
        progression = {
            "recurrent_predicates": sorted(key for key, count in predicate_counts.items() if key and count > 1),
            "recurrent_entities": sorted(key for key, count in entity_counts.items() if key and count > 1),
            "new_event_count": len(situation.get("events", [])),
            "goal_hypotheses": len(situation.get("motivations", [])),
        }
        relation_support = float(np.clip(
            (len(proposition_links) + len(temporal_relations) + len(situation.get("causal_links", [])))
            / max(1, 2 * len(propositions)), 0.0, 1.0
        ))
        conflicts: list[dict[str, Any]] = []
        for left_index, left in enumerate(propositions):
            for right in propositions[left_index + 1:]:
                left_args = set(str(value) for value in (
                    left.get("arguments", {}).values()
                    if isinstance(left.get("arguments"), Mapping) else left.get("arguments", [])
                ))
                right_args = set(str(value) for value in (
                    right.get("arguments", {}).values()
                    if isinstance(right.get("arguments"), Mapping) else right.get("arguments", [])
                ))
                same_predicate = str(left.get("predicate", "")) == str(right.get("predicate", ""))
                shared_participant = bool(left_args & right_args)
                opposite_polarity = bool(left.get("negated")) != bool(right.get("negated"))
                if same_predicate and shared_participant and opposite_polarity:
                    conflicts.append({
                        "left": left["proposition_id"], "right": right["proposition_id"],
                        "type": "explicit_polarity_conflict",
                        "shared_participants": sorted(left_args & right_args),
                        "evidence": "same_predicate_and_participant_with_opposite_negation",
                    })
        entity_continuity = float(np.clip(
            sum(item["continuity_score"] for item in proposition_links if "continuity_score" in item)
            / max(1, sum("continuity_score" in item for item in proposition_links)),
            0.0, 1.0
        )) if proposition_links else 0.5
        consistency = 1.0 - min(1.0, len(conflicts) / max(1, len(propositions)))
        cfg = self.config
        raw_weights = np.asarray([
            cfg.coherence_entity_gain, cfg.coherence_topic_gain,
            cfg.coherence_relation_gain, cfg.coherence_consistency_gain,
        ], dtype=np.float64)
        weights = raw_weights / max(float(raw_weights.sum()), cfg.epsilon)
        components = np.asarray([entity_continuity, topic_continuity, relation_support, consistency])
        global_score = float(np.clip(np.dot(weights, components), 0.0, 1.0))
        local_beliefs = np.asarray([max(0.0, item.get("confidence", 0.0)) for item in proposition_links], dtype=np.float64)
        if local_beliefs.size:
            evidence = float(np.mean(local_beliefs))
            revised = (1.0 - cfg.recurrent_damping) * global_score + cfg.recurrent_damping * evidence
        else:
            revised = global_score
        topic_name = max(topic_distribution, key=topic_distribution.get) if topic_distribution else None
        coherence = {
            "score": float(np.clip(revised, 0.0, 1.0)),
            "components": {
                "entity_continuity": entity_continuity,
                "topic_continuity": topic_continuity,
                "relation_support": relation_support,
                "consistency": consistency,
            },
            "weights": {
                "entity": float(weights[0]), "topic": float(weights[1]),
                "relations": float(weights[2]), "consistency": float(weights[3]),
            },
            "topic_distribution": topic_distribution,
            "dominant_topic_term": topic_name,
            "narrative_progression": progression,
            "knowledge_conflicts": conflicts,
            "recurrent_revision": {
                "initial_score": global_score, "revised_score": float(revised),
                "damping": cfg.recurrent_damping,
                "original_sentences_preserved": True,
            },
            "summary": (
                f"Topic focus: {topic_name}; {len(propositions)} propositions, "
                f"{len(situation.get('events', []))} events, {len(conflicts)} explicit conflicts."
                if topic_name else
                f"{len(propositions)} propositions and {len(conflicts)} explicit conflicts; topic is undetermined."
            ),
        }
        signals = [
            {"value": len(topic_distribution), "drive": float(np.clip(len(topic_distribution) / 10, 0, 1))},
            {"value": topic_continuity, "drive": topic_continuity},
            {"value": len(progression["new_event_count"] if isinstance(progression["new_event_count"], list) else situation.get("events", [])), "drive": float(np.clip(len(situation.get("events", [])) / 5, 0, 1))},
            {"value": relation_support, "drive": relation_support},
            {"value": len(conflicts), "drive": float(np.clip(len(conflicts) / max(1, len(propositions)), 0, 1))},
            {"value": revised, "drive": revised},
            {"value": float(revised), "drive": float(revised)},
            {"value": len(coherence), "drive": float(np.clip(global_score, 0, 1))},
        ]
        return coherence, signals


class DiscourseN7Adapter:
    """Adapt ORE N9 discourse payloads to the stand-alone eight-stage discourse processor."""

    def __init__(self, processor: DiscourseProcessor | None = None) -> None:
        """Use an existing discourse processor or construct one with default settings."""
        self.processor = processor or DiscourseProcessor()

    def predict(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Implement the N9 N7 predict contract and return its expected context fields."""
        discourse_context = dict(payload.get("discourse_context", {}))
        propositions = list(payload.get("propositions", []))
        semantic_graph = dict(payload.get("semantic_graph", {}))
        entities = []
        for node in semantic_graph.get("nodes", []):
            if isinstance(node, Mapping):
                entities.append({
                    "entity_id": node.get("id"),
                    "label": node.get("label", node.get("name", node.get("id", ""))),
                    "entity_type": node.get("type", "concept"),
                    "source_reference": node.get("source_reference"),
                })
        lexical_hypotheses = list(payload.get("lexical_hypotheses", []))
        if not propositions and lexical_hypotheses:
            forms = [
                str(item.get("form", item.get("token", ""))).strip()
                for item in lexical_hypotheses if isinstance(item, Mapping)
            ]
            forms = [form for form in forms if form]
            if forms:
                propositions = [{"sentence_index": 0, "text": " ".join(forms)}]
        sentence_texts = []
        for item in propositions:
            if isinstance(item, Mapping):
                text = str(item.get("text", item.get("sentence", ""))).strip()
                if text:
                    sentence_texts.append(text)
        if not sentence_texts:
            current = str(discourse_context.get("current_utterance", "")).strip()
            if current:
                sentence_texts = [current]
        if not propositions and sentence_texts:
            propositions = [
                {"sentence_index": index, "text": sentence}
                for index, sentence in enumerate(sentence_texts)
            ]
        if not sentence_texts and not propositions:
            return {
                "intents": [], "predictions": {}, "situation_model": {},
                "coherence": 0.0, "uncertainty": 1.0, "temporal_links": [],
                "causal_links": [], "proposition_compatibility": [],
                "knowledge_conflicts": [], "indirect_requests": [], "irony": [],
                "diagnostics": {"status": "no_discourse_evidence"},
            }
        request = DiscourseRequest(
            sequence_id=str(discourse_context.get("sequence_id", "n9-discourse")),
            sentences=sentence_texts or None,
            propositions=propositions,
            entities=entities,
            common_ground=dict(discourse_context.get("common_ground", {})),
            speaker_model=dict(discourse_context.get("speaker_model", {})),
            context={
                **discourse_context,
                "prior_turns": discourse_context.get("prior_turns", []),
                "document_context": discourse_context.get("document_context", {}),
            },
            language=str(discourse_context.get("language", "en")),
        )
        result = self.processor.process(request)
        return {
            "intents": result.speaker_intentions,
            "predictions": {
                "dominant_topic_term": result.coherence.get("dominant_topic_term"),
                "topic_distribution": result.coherence.get("topic_distribution", {}),
                "next_event_hypotheses": result.situation_model.get("motivations", []),
            },
            "situation_model": result.situation_model,
            "coherence": float(result.coherence.get("score", 0.0)),
            "uncertainty": float(np.clip(
                1.0 - result.coherence.get("score", 0.0)
                + np.mean([item.get("abstain", False) for item in result.speaker_intentions])
                if result.speaker_intentions else 1.0,
                0.0, 1.0,
            )),
            "temporal_links": result.temporal_relations,
            "causal_links": result.situation_model.get("causal_links", []),
            "proposition_compatibility": result.proposition_links,
            "knowledge_conflicts": result.coherence.get("knowledge_conflicts", []),
            "indirect_requests": result.indirect_requests,
            "irony": result.irony,
            "discourse_context": result.discourse_context,
            "operation_trace": result.operation_trace,
            "diagnostics": result.diagnostics,
        }


__all__ = [
    "DiscourseN7Adapter", "DiscourseProcessor",
    "OPERATION_REGISTRY", "STAGE_REGISTRY",
]
