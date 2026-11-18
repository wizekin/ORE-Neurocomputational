"""Mathematical and functional operations for Point 20 reconstructive memory."""
from __future__ import annotations

import math
import re
from typing import Any, Mapping, Sequence

from .contracts import (
    DetailCandidate, EpisodicFragment, EpisodicRelation, FeedbackRecord, Point20Config, Point20Request,
    Point20State, SchemaProfile, SourceRecord,
)


_WORDS = re.compile(r"[a-z0-9]+", flags=re.IGNORECASE)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp a finite value to a closed interval."""
    number = float(value)
    if not math.isfinite(number) or low > high:
        raise ValueError("value must be finite and low must not exceed high")
    return min(high, max(low, number))


def sigmoid(value: float) -> float:
    """Compute a numerically stable logistic activation."""
    number = float(value)
    if number >= 0.0:
        z = math.exp(-min(number, 700.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(number, -700.0))
    return z / (1.0 + z)


def softmax(scores: Sequence[float], temperature: float = 1.0) -> list[float]:
    """Convert finite scores into a stable temperature-scaled probability distribution."""
    if not scores:
        return []
    scale = float(temperature)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("temperature must be finite and positive")
    values = [float(score) for score in scores]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("scores must be finite")
    maximum = max(values)
    weights = [math.exp(max(-700.0, min(0.0, (value - maximum) / scale))) for value in values]
    total = sum(weights)
    return [weight / total for weight in weights]


def value_key(value: Any) -> str:
    """Create a stable categorical key without conflating booleans, nulls, and text."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("feature values must be finite")
        return format(value, ".12g")
    if isinstance(value, Mapping):
        return "{" + ",".join(
            f"{str(key).casefold()}:{value_key(value[key])}" for key in sorted(value, key=str)
        ) + "}"
    if isinstance(value, (tuple, list)):
        return "[" + ",".join(value_key(item) for item in value) + "]"
    return str(value).strip().casefold()


def feature_similarity(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    """Measure exact categorical overlap or cosine similarity for numeric feature mappings."""
    shared = set(left) & set(right)
    if not shared:
        return 0.0
    categorical = any(
        not isinstance(left[key], (int, float, bool))
        or not isinstance(right[key], (int, float, bool))
        for key in shared
    )
    if categorical:
        return sum(value_key(left[key]) == value_key(right[key]) for key in shared) / len(shared)
    a = [float(left[key]) for key in shared]
    b = [float(right[key]) for key in shared]
    if any(not math.isfinite(x) for x in a + b):
        raise ValueError("numeric feature values must be finite")
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return 0.0
    return clamp(sum(x * y for x, y in zip(a, b)) / (norm_a * norm_b), -1.0, 1.0)


def _token_overlap(left: str, right: str) -> float:
    """Estimate text-cue overlap with token-set Jaccard similarity."""
    a = set(_WORDS.findall(left.casefold()))
    b = set(_WORDS.findall(right.casefold()))
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def define_retrieval_target(request: Point20Request) -> dict[str, Any]:
    """Represent the retrieval question, target features, precision, and task context."""
    cue_features = dict(request.cue) if isinstance(request.cue, Mapping) else {}
    targets = list(request.target_features)
    if not targets:
        targets = sorted({item.feature_type for item in request.episodic_fragments})
    precision_weights = {
        "gist": {"gist": 0.75, "detail": 0.25, "source": 0.35},
        "balanced": {"gist": 0.45, "detail": 0.55, "source": 0.70},
        "exact": {"gist": 0.15, "detail": 0.85, "source": 1.0},
    }[request.precision]
    return {
        "event_id": request.event_id,
        "cue": dict(request.cue) if isinstance(request.cue, Mapping) else request.cue,
        "cue_features": cue_features,
        "target_features": targets,
        "task_type": request.task_type,
        "precision": request.precision,
        "precision_weights": precision_weights,
        "target_episode_id": request.target_episode_id,
        "context": dict(request.context),
        "what_happened_vs_what_is_typical": "reconstruct_this_event",
    }


def reactivate_episodic_fragments(
    request: Point20Request,
    target: Mapping[str, Any],
    config: Point20Config,
) -> dict[str, Any]:
    """Rank event-specific fragments from cue overlap while preserving episode identity."""
    cue = target["cue"]
    cue_features = target["cue_features"]
    context_cues = request.context.get("retrieval_context_features", {})
    if not isinstance(context_cues, Mapping):
        context_cues = {}
    ranked: list[dict[str, Any]] = []
    for fragment in request.episodic_fragments:
        if cue_features:
            cue_fit = feature_similarity(cue_features, fragment.cue_features)
        elif isinstance(cue, str):
            cue_fit = _token_overlap(cue, str(fragment.content))
        else:
            cue_fit = 0.0
        context_fit = feature_similarity(context_cues, fragment.source_features) if context_cues else 0.0
        episode_match = (
            not request.target_episode_id
            or fragment.episode_id == request.target_episode_id
        )
        episode_unidentified = bool(request.target_episode_id) and not fragment.episode_id
        activation = (
            0.55 * fragment.retrieval_strength
            + 0.30 * fragment.episodic_context
            + 0.10 * cue_fit
            + 0.05 * context_fit
        )
        if cue_features or (isinstance(cue, str) and cue.strip()):
            activation *= 0.35 + 0.65 * max(cue_fit, context_fit)
        if not episode_match:
            activation *= 0.70
        ranked.append({
            "detail_id": fragment.detail_id,
            "feature_type": fragment.feature_type,
            "content": fragment.content,
            "episode_id": fragment.episode_id,
            "episode_match": episode_match,
            "episode_unidentified": episode_unidentified,
            "source_type": fragment.source_type,
            "source_id": fragment.source_id,
            "cue_fit": cue_fit,
            "context_fit": context_fit,
            "activation": clamp(activation),
            "status": fragment.status,
            "retrieval_strength": fragment.retrieval_strength,
            "episodic_context": fragment.episodic_context,
            "attention_encoding": fragment.attention_encoding,
            "centrality": fragment.centrality,
            "confidence": clamp(0.55 * fragment.retrieval_strength + 0.45 * fragment.episodic_context),
        })
    probabilities = softmax([row["activation"] for row in ranked], config.retrieval_temperature)
    for row, probability in zip(ranked, probabilities):
        row["normalized_activation"] = probability
        row["reactivated"] = row["activation"] >= 0.18
    ranked.sort(key=lambda item: (-item["activation"], item["detail_id"]))
    relation_rows: list[dict[str, Any]] = []
    for relation in request.episodic_relations:
        triple = relation.triple()
        if cue_features:
            cue_fit = feature_similarity(cue_features, triple)
        elif isinstance(cue, str):
            cue_fit = _token_overlap(cue, " ".join(triple.values()))
        else:
            cue_fit = 0.0
        activation = (
            0.52 * relation.retrieval_strength
            + 0.38 * relation.episodic_context
            + 0.10 * cue_fit
        )
        episode_match = (
            not request.target_episode_id
            or relation.episode_id == request.target_episode_id
        )
        episode_unidentified = bool(request.target_episode_id) and not relation.episode_id
        activation = activation * (1.0 if episode_match else 0.70)
        relation_rows.append({
            "relation_id": relation.relation_id,
            **triple,
            "episode_id": relation.episode_id,
            "episode_match": episode_match,
            "episode_unidentified": episode_unidentified,
            "source_type": relation.source_type,
            "source_id": relation.source_id,
            "status": relation.status,
            "cue_fit": cue_fit,
            "activation": clamp(activation),
            "reactivated": activation >= 0.18,
            "retrieval_strength": relation.retrieval_strength,
            "episodic_context": relation.episodic_context,
        })
    relation_probabilities = softmax(
        [item["activation"] for item in relation_rows], config.retrieval_temperature,
    )
    for row, probability in zip(relation_rows, relation_probabilities):
        row["normalized_activation"] = probability
    relation_rows.sort(key=lambda item: (-item["activation"], item["relation_id"]))
    return {
        "fragments": ranked,
        "relations": relation_rows,
        "reactivated_relation_ids": [
            item["relation_id"] for item in relation_rows if item["reactivated"]
        ],
        "reactivated_ids": [row["detail_id"] for row in ranked if row["reactivated"]],
        "episode_ids": sorted({row["episode_id"] for row in ranked if row["reactivated"] and row["episode_id"]}),
        "normalization_sum": sum(probabilities),
        "relation_normalization_sum": sum(relation_probabilities),
        "not_retrieved_is_not_absent": True,
    }


def build_feature_ledger(
    request: Point20Request,
    retrieval: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a per-feature evidence ledger that distinguishes unknown, contradicted, and unqueried."""
    target_features = list(request.target_features)
    if not target_features:
        target_features = sorted({item.feature_type for item in request.episodic_fragments})
    reactivated_ids = set(retrieval.get("reactivated_ids", ()))
    rows: dict[str, dict[str, Any]] = {}
    for feature in target_features:
        fragments = [item for item in request.episodic_fragments if item.feature_type == feature]
        active = [item for item in fragments if item.detail_id in reactivated_ids]
        target_active = [
            item for item in active
            if not request.target_episode_id or item.episode_id == request.target_episode_id
        ]
        unidentified_active = [
            item for item in active
            if request.target_episode_id and not item.episode_id
        ]
        competing = [
            item for item in active
            if request.target_episode_id and item.episode_id
            and item.episode_id != request.target_episode_id
        ]
        direct = [item for item in target_active if item.status == "direct_evidence"]
        retrieved = [item for item in target_active if item.status == "episodically_retrieved"]
        weak = [
            item for item in target_active
            if item.status in {"weakly_retrieved", "source_uncertain", "unknown"}
        ] + unidentified_active
        contradicted = [item for item in target_active if item.status == "contradicted"]
        not_retrieved = [item for item in target_active if item.status == "not_retrieved"]
        supporting = direct + retrieved + weak
        if contradicted and not supporting:
            status = "contradicted"
        elif direct:
            status = "direct_evidence"
        elif retrieved:
            status = "episodically_retrieved"
        elif weak:
            status = "weakly_retrieved"
        elif not_retrieved:
            status = "not_retrieved"
        else:
            status = "not_retrieved"
        rows[feature] = {
            "feature_type": feature,
            "status": status,
            "values": [
                {
                    "detail_id": item.detail_id,
                    "content": item.content,
                    "status": "source_uncertain" if item in unidentified_active or item.status == "unknown" else item.status,
                    "retrieval_strength": item.retrieval_strength,
                    "episodic_context": item.episodic_context,
                    "source_type": item.source_type,
                    "source_id": item.source_id,
                    "episode_id": item.episode_id,
                }
                for item in (*target_active, *unidentified_active)
            ],
            "evidence_strength": max(
                (item.retrieval_strength * item.attention_encoding for item in supporting),
                default=0.0,
            ),
            "likely_encoded": max((item.attention_encoding for item in target_active), default=0.0),
            "centrality": max((item.centrality for item in target_active), default=0.0),
            "competing_episode_values": [
                {"content": item.content, "episode_id": item.episode_id,
                 "detail_id": item.detail_id, "source_type": item.source_type}
                for item in competing
            ],
            "unavailable_not_absent": status in {"not_retrieved", "weakly_retrieved"},
        }
    active_relation_ids = set(retrieval.get("reactivated_relation_ids", ()))
    relation_rows = [
        {
            **item.triple(),
            "relation_id": item.relation_id,
            "status": item.status,
            "source_type": item.source_type,
            "source_id": item.source_id,
            "episode_id": item.episode_id,
            "retrieval_strength": item.retrieval_strength,
            "episodic_context": item.episodic_context,
            "reactivated": item.relation_id in active_relation_ids,
            "episode_match": (
                not request.target_episode_id or item.episode_id == request.target_episode_id
            ),
            "episode_unidentified": bool(request.target_episode_id) and not item.episode_id,
        }
        for item in request.episodic_relations
    ]
    retrieved_relations = [
        item for item in relation_rows
        if item["reactivated"] and item["episode_match"]
        and item["status"] in {"direct_evidence", "episodically_retrieved"}
    ]
    relation_event_nodes = sorted({
        value for relation in retrieved_relations
        for value in (relation["subject"], relation["object"])
    })
    event_graph = {
        "nodes": relation_event_nodes,
        "edges": [
            {"relation_id": item["relation_id"], "subject": item["subject"],
             "predicate": item["predicate"], "object": item["object"],
             "source_id": item["source_id"], "episode_id": item["episode_id"]}
            for item in retrieved_relations
        ],
        "graph_is_episode_scoped": True,
    }
    if request.episodic_relations or any(schema.relation_templates for schema in request.schemas):
        weak_relations = [
            item for item in relation_rows
            if item["status"] in {"weakly_retrieved", "source_uncertain", "suggested_after_event"}
            or (item["reactivated"] and item["episode_unidentified"])
        ]
        contradicted_relations = [
            item for item in relation_rows
            if item["status"] == "contradicted" and item["episode_match"]
        ]
        relation_status = (
            "episodically_retrieved" if retrieved_relations else
            "contradicted" if contradicted_relations and not weak_relations else
            "weakly_retrieved" if weak_relations else
            "not_retrieved"
        )
        rows["relation"] = {
            "feature_type": "relation",
            "status": relation_status,
            "values": [
                {**dict(item), "status": "source_uncertain" if item["episode_unidentified"] or item["status"] == "unknown" else item["status"]}
                for item in relation_rows if item["reactivated"] and (item["episode_match"] or item["episode_unidentified"])
            ],
            "competing_episode_relations": [
                dict(item) for item in relation_rows if item["reactivated"] and not item["episode_match"]
            ],
            "evidence_strength": max(
                (item["retrieval_strength"] * item["episodic_context"] for item in relation_rows
                 if item["reactivated"] and item["episode_match"] and item["status"] in {"direct_evidence", "episodically_retrieved"}),
                default=0.0,
            ),
            "likely_encoded": max((item["retrieval_strength"] for item in relation_rows), default=0.0),
            "centrality": 0.8,
            "unavailable_not_absent": not bool(retrieved_relations),
        }
    return {
        "features": rows,
        "relations": relation_rows,
        "competing_relations": [item for item in relation_rows if item["reactivated"] and not item["episode_match"]],
        "event_graph": event_graph,
        "retrieved_feature_count": sum(row["status"] in {"direct_evidence", "episodically_retrieved"} for row in rows.values()),
        "missing_feature_count": sum(row["status"] in {"not_retrieved", "weakly_retrieved"} for row in rows.values()),
        "contradicted_feature_count": sum(row["status"] == "contradicted" for row in rows.values()),
        "status_definitions": {
            "not_retrieved": "No sufficiently strong event-specific fragment was reactivated; this does not establish absence.",
            "contradicted": "A supplied, source-labelled observation conflicts with a candidate value.",
            "not_queried": "A feature outside the current target set was not assessed.",
        },
    }


def _schema_feature_distribution(
    schema: SchemaProfile,
    feature: str,
    schema_beta_counts: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    """Combine base schema feature frequencies with verified beta-count updates from prior runs."""
    base = dict(schema.feature_probabilities.get(feature, {}))
    if not base:
        return {}
    adjusted: dict[str, float] = {}
    for value, probability in base.items():
        key = "|".join((schema.schema_id, feature, value_key(value)))
        counts = schema_beta_counts.get(key)
        if counts:
            alpha = max(1e-9, float(counts.get("alpha", 1.0)))
            beta = max(1e-9, float(counts.get("beta", 1.0)))
            adjusted[value] = alpha / (alpha + beta)
        else:
            adjusted[value] = probability
    total = sum(adjusted.values())
    return {value: score / total for value, score in adjusted.items()} if total > 0.0 else base


def _schema_evidence_fit(
    schema: SchemaProfile,
    ledger: Mapping[str, Any],
    schema_beta_counts: Mapping[str, Mapping[str, float]] | None = None,
) -> float:
    """Score how well a schema predicts available evidence without treating unknowns as mismatches."""
    scores: list[tuple[float, float]] = []
    for feature, row in ledger["features"].items():
        if row["status"] not in {"direct_evidence", "episodically_retrieved", "weakly_retrieved"}:
            continue
        distribution = _schema_feature_distribution(schema, feature, schema_beta_counts or {})
        if not distribution:
            continue
        for item in row["values"]:
            if item["status"] not in {"direct_evidence", "episodically_retrieved", "weakly_retrieved"}:
                continue
            value_probability = distribution.get(value_key(item["content"]), 0.0)
            reliability = clamp(item["retrieval_strength"])
            score = value_probability if value_probability > 0.0 else 0.01
            scores.append((score, reliability))
    for relation in ledger.get("relations", ()):
        if not relation.get("reactivated") or not relation.get("episode_match") or relation.get("status") not in {
            "direct_evidence", "episodically_retrieved", "weakly_retrieved", "contradicted"
        }:
            continue
        triple = {key: relation[key] for key in ("subject", "predicate", "object")}
        relation_templates = [
            item for item in schema.relation_templates
            if all(item[key] == triple[key] for key in ("subject", "predicate", "object"))
        ]
        if relation.get("status") == "contradicted":
            if not relation_templates:
                continue
            relation_probability = max(
                0.01, 1.0 - max(float(item["probability"]) for item in relation_templates)
            )
        else:
            if not relation_templates:
                continue
            relation_probability = max(float(item["probability"]) for item in relation_templates)
        scores.append((
            relation_probability,
            clamp(float(relation.get("retrieval_strength", 0.0))),
        ))
    if not scores:
        return 0.5
    log_fit = sum(weight * math.log(max(1e-9, score)) for score, weight in scores)
    weight_total = sum(weight for _, weight in scores)
    return clamp(math.exp(log_fit / max(1e-9, weight_total)))


def activate_candidate_schemas(
    request: Point20Request,
    ledger: Mapping[str, Any],
    config: Point20Config,
    schema_beta_counts: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, Any]:
    """Activate multiple candidate schemas from evidence fit, context fit, and goal relevance."""
    context = request.context
    goal_values = context.get("schema_goal_relevance", {})
    if not isinstance(goal_values, Mapping):
        goal_values = {}
    ranked: list[dict[str, Any]] = []
    for schema in request.schemas:
        evidence_fit = _schema_evidence_fit(schema, ledger, schema_beta_counts)
        context_fit = feature_similarity(schema.context_profile, {
            str(key): value for key, value in context.items()
            if isinstance(value, (int, float, bool))
        }) if schema.context_profile else 0.0
        goal_fit = clamp(float(goal_values.get(schema.schema_id, schema.goal_relevance)))
        log_prior = math.log(max(config.epsilon, schema.prior_probability))
        score = 0.58 * evidence_fit + 0.27 * context_fit + 0.15 * goal_fit + 0.10 * log_prior
        ranked.append({
            "schema_id": schema.schema_id,
            "name": schema.name,
            "evidence_fit": evidence_fit,
            "context_fit": context_fit,
            "goal_relevance": goal_fit,
            "prior_probability": schema.prior_probability,
            "score": score,
            "cultural_scope": schema.cultural_scope,
        })
    probabilities = softmax([item["score"] for item in ranked], config.schema_temperature)
    for row, probability in zip(ranked, probabilities):
        row["activation_probability"] = probability
    ranked.sort(key=lambda item: (-item["activation_probability"], item["schema_id"]))
    return {
        "ranked_schemas": ranked,
        "active_schema_ids": [item["schema_id"] for item in ranked[:config.max_active_schemas]],
        "normalization_sum": sum(probabilities),
        "schema_is_hypothesis_not_event_record": True,
    }



def _schema_relation_probability(
    schema: SchemaProfile,
    template: Mapping[str, Any],
    schema_beta_counts: Mapping[str, Mapping[str, float]] | None = None,
) -> float:
    """Resolve a relation-template probability with verified beta-count learning applied."""
    triple = {key: template[key] for key in ("subject", "predicate", "object")}
    key = "|".join((schema.schema_id, "relation", value_key(triple)))
    counts = (schema_beta_counts or {}).get(key)
    if counts:
        alpha = max(1e-9, float(counts.get("alpha", 1.0)))
        beta = max(1e-9, float(counts.get("beta", 1.0)))
        return alpha / (alpha + beta)
    return clamp(float(template["probability"]))


def generate_inferred_completions(
    request: Point20Request,
    ledger: Mapping[str, Any],
    schema_activation: Mapping[str, Any],
    config: Point20Config,
    schema_beta_counts: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, Any]:
    """Generate ranked alternatives only for missing or weak features and mark each as inferred."""
    schema_by_id = {item.schema_id: item for item in request.schemas}
    activation_by_id = {row["schema_id"]: row["activation_probability"]
                        for row in schema_activation["ranked_schemas"]}
    candidates: list[DetailCandidate] = []
    omitted: list[dict[str, Any]] = []
    for feature, feature_row in ledger["features"].items():
        if feature_row["status"] not in {"not_retrieved", "weakly_retrieved"}:
            continue
        proposals: list[tuple[float, SchemaProfile, str, float]] = []
        for schema_id in schema_activation["active_schema_ids"]:
            schema = schema_by_id[schema_id]
            distribution = _schema_feature_distribution(schema, feature, schema_beta_counts or {})
            for value, prior in distribution.items():
                score = activation_by_id.get(schema_id, 0.0) * prior
                proposals.append((score, schema, value, prior))
        proposals.sort(key=lambda row: (-row[0], row[1].schema_id, row[2]))
        for score, schema, value, prior in proposals[:config.max_completions_per_feature]:
            candidate_id = f"{schema.schema_id}:{feature}:{value}"
            candidates.append(DetailCandidate(
                candidate_id=candidate_id,
                feature_type=feature,
                content=value,
                status="inferred",
                schema_id=schema.schema_id,
                schema_support=activation_by_id.get(schema.schema_id, 0.0),
                schema_prior=clamp(prior * max(activation_by_id.get(schema.schema_id, 0.0), config.epsilon)),
                inference_proportion=1.0,
                provenance=[f"schema:{schema.schema_id}", "generated_as_inference"],
            ))
        if not proposals:
            omitted.append({"feature_type": feature, "reason": "no compatible schema proposal"})
    # Preserve participant/action/temporal/causal bindings as directed relation triples.
    existing_relations = [
        item for item in request.episodic_relations
        if item.status in {"direct_evidence", "episodically_retrieved"}
        and (not request.target_episode_id or item.episode_id == request.target_episode_id)
    ]
    relation_proposals: list[tuple[float, SchemaProfile, Mapping[str, Any]]] = []
    for schema_id in schema_activation["active_schema_ids"]:
        schema = schema_by_id[schema_id]
        for template in schema.relation_templates:
            triple = {
                "subject": template["subject"],
                "predicate": template["predicate"],
                "object": template["object"],
            }
            already_supported = any(
                item.triple() == triple
                and (not request.target_episode_id or item.episode_id == request.target_episode_id)
                for item in existing_relations
            )
            if already_supported:
                continue
            template_probability = _schema_relation_probability(schema, template, schema_beta_counts)
            relation_proposals.append((activation_by_id.get(schema_id, 0.0) * template_probability,
                                       schema, {**template, "effective_probability": template_probability}))
    relation_proposals.sort(key=lambda row: (-row[0], row[1].schema_id, str(row[2])))
    for score, schema, template in relation_proposals[:config.max_completions_per_feature * config.max_active_schemas]:
        triple = {
            "subject": template["subject"],
            "predicate": template["predicate"],
            "object": template["object"],
        }
        candidates.append(DetailCandidate(
            candidate_id=(
                f"{schema.schema_id}:relation:{triple['subject']}:"
                f"{triple['predicate']}:{triple['object']}"
            ),
            feature_type="relation",
            content=triple,
            status="inferred",
            schema_id=schema.schema_id,
            schema_support=activation_by_id.get(schema.schema_id, 0.0),
            schema_prior=clamp(
                float(template.get("effective_probability", template["probability"]))
                * max(activation_by_id.get(schema.schema_id, 0.0), config.epsilon)
            ),
            inference_proportion=1.0,
            provenance=[f"schema:{schema.schema_id}", "generated_relation_as_inference"],
        ))
    return {
        "candidates": candidates,
        "candidate_count": len(candidates),
        "omitted_features": omitted,
        "all_candidates_labelled_inferred": all(item.status == "inferred" for item in candidates),
    }


def evaluate_candidates_bayesian(
    candidates: Sequence[DetailCandidate],
    fragments: Sequence[EpisodicFragment],
    config: Point20Config,
    relations: Sequence[EpisodicRelation] = (),
    target_episode_id: str = "",
) -> dict[str, Any]:
    """Combine target-episode likelihoods with schema priors without borrowing similar-episode evidence."""
    evaluated: list[DetailCandidate] = []
    raw_scores: dict[str, float] = {}
    for candidate in candidates:
        evidence_rows: list[dict[str, Any]] = [
            {
                "evidence_id": item.detail_id,
                "content": item.content,
                "status": item.status,
                "reliability": clamp(item.retrieval_strength * item.attention_encoding),
                "episodic_context": item.episodic_context,
            }
            for item in fragments
            if item.feature_type == candidate.feature_type
            and (not target_episode_id or item.episode_id == target_episode_id)
            and item.status in {"direct_evidence", "episodically_retrieved", "weakly_retrieved", "contradicted"}
        ]
        if candidate.feature_type == "relation":
            evidence_rows.extend(
                {
                    "evidence_id": item.relation_id,
                    "content": item.triple(),
                    "status": item.status,
                    "reliability": clamp(item.retrieval_strength),
                    "episodic_context": item.episodic_context,
                }
                for item in relations
                if (not target_episode_id or item.episode_id == target_episode_id)
                and item.status in {"direct_evidence", "episodically_retrieved", "weakly_retrieved", "contradicted"}
            )
        log_likelihood = 0.0
        support_total = 0.0
        support_weight = 0.0
        contradiction_total = 0.0
        contradiction_weight = 0.0
        context_total = 0.0
        for evidence in evidence_rows:
            reliability = evidence["reliability"]
            match = value_key(evidence["content"]) == value_key(candidate.content)
            if evidence["status"] == "contradicted":
                # A contradicted observation penalizes that exact value. It does not prove
                # that every alternative value is true unless exclusivity is separately known.
                if match:
                    contradiction_total += reliability
                    contradiction_weight += reliability
                    log_likelihood += math.log(max(config.epsilon, 1.0 - reliability))
                continue
            if match:
                support_total += reliability * evidence["episodic_context"]
                support_weight += reliability
                log_likelihood += math.log(max(config.epsilon, reliability))
                context_total += evidence["episodic_context"] * reliability
                candidate.evidence_ids.append(evidence["evidence_id"])
            elif evidence["status"] != "weakly_retrieved" and candidate.feature_type != "relation":
                contradiction_total += reliability
                contradiction_weight += reliability
                log_likelihood += math.log(max(config.epsilon, 1.0 - reliability))
        candidate.evidence_strength = clamp(support_total / max(config.epsilon, support_weight))
        candidate.episodic_context = clamp(context_total / max(config.epsilon, support_weight))
        candidate.contradiction = clamp(contradiction_total / max(config.epsilon, contradiction_weight))
        candidate.evidence_likelihood = clamp(math.exp(max(-60.0, log_likelihood)))
        candidate.source_specificity = candidate.episodic_context
        log_prior = math.log(max(config.epsilon, candidate.schema_prior))
        raw_score = log_prior + log_likelihood - config.contradiction_penalty * candidate.contradiction
        raw_scores[candidate.candidate_id] = raw_score
        evaluated.append(candidate)
    grouped: dict[str, list[DetailCandidate]] = {}
    for candidate in evaluated:
        grouped.setdefault(candidate.feature_type, []).append(candidate)
    for feature, group in grouped.items():
        probabilities = softmax([raw_scores[item.candidate_id] for item in group], config.posterior_temperature)
        for candidate, probability in zip(group, probabilities):
            candidate.posterior = probability
    return {
        "candidates": evaluated,
        "candidate_count": len(evaluated),
        "feature_groups": {
            feature: {
                "candidate_ids": [item.candidate_id for item in group],
                "posterior_sum": sum(item.posterior for item in group),
            }
            for feature, group in grouped.items()
        },
        "bayesian_caution": "Posterior ranking is conditional on supplied priors and likelihood assumptions; it is not direct evidence of event occurrence.",
    }


def check_spatiotemporal_causal_constraints(
    candidates: Sequence[DetailCandidate],
    request: Point20Request,
) -> dict[str, Any]:
    """Score candidate values against explicit temporal, spatial, participant, and causal constraints."""
    constraints = request.context.get("constraints", ())
    if isinstance(constraints, Mapping):
        constraints = [constraints]
    if not isinstance(constraints, Sequence) or isinstance(constraints, (str, bytes)):
        raise TypeError("context.constraints must be a sequence of constraint mappings")
    updated: list[DetailCandidate] = []
    for candidate in candidates:
        results: list[dict[str, Any]] = []
        penalty = 0.0
        total_weight = 0.0
        for index, raw in enumerate(constraints):
            if not isinstance(raw, Mapping):
                continue
            feature = str(raw.get("feature_type", raw.get("feature", ""))).casefold()
            if feature and feature != candidate.feature_type:
                continue
            weight = clamp(float(raw.get("weight", 1.0)))
            allowed = raw.get("allowed_values")
            forbidden = raw.get("forbidden_values", ())
            expected_relation = raw.get("expected_relation")
            if expected_relation is None and all(key in raw for key in ("subject", "predicate", "object")):
                expected_relation = {key: raw[key] for key in ("subject", "predicate", "object")}
            minimum = raw.get("minimum")
            maximum = raw.get("maximum")
            satisfied = True
            reason = "no applicable constraint"
            if allowed is not None:
                allowed_values = {value_key(value) for value in allowed}
                satisfied = value_key(candidate.content) in allowed_values
                reason = "allowed_value_match" if satisfied else "not_in_allowed_values"
            if value_key(candidate.content) in {value_key(value) for value in forbidden}:
                satisfied = False
                reason = "explicitly_forbidden_value"
            if minimum is not None or maximum is not None:
                try:
                    number = float(candidate.content)
                    satisfied = satisfied and (minimum is None or number >= float(minimum))
                    satisfied = satisfied and (maximum is None or number <= float(maximum))
                    reason = "numeric_range_satisfied" if satisfied else "numeric_range_violated"
                except (TypeError, ValueError):
                    satisfied = False
                    reason = "non_numeric_value_for_numeric_constraint"
            if expected_relation is not None:
                expected = dict(expected_relation) if isinstance(expected_relation, Mapping) else None
                if not isinstance(candidate.content, Mapping) or expected is None:
                    satisfied = False
                    reason = "relation_candidate_or_constraint_not_structured"
                else:
                    relation_match = all(
                        value_key(candidate.content.get(key)) == value_key(expected.get(key))
                        for key in ("subject", "predicate", "object")
                    )
                    satisfied = satisfied and relation_match
                    reason = "event_relation_match" if satisfied else "event_relation_mismatch"
            if allowed is None and not forbidden and minimum is None and maximum is None and expected_relation is None:
                continue
            total_weight += weight
            if not satisfied:
                penalty += weight
            results.append({
                "constraint_id": str(raw.get("constraint_id", f"constraint-{index}")),
                "constraint_type": str(raw.get("constraint_type", "feature_value")),
                "satisfied": satisfied,
                "weight": weight,
                "hard": bool(raw.get("hard", False)),
                "reason": reason,
            })
        candidate.constraint_results = results
        candidate.contextual_fit = clamp(1.0 - penalty / max(config_epsilon(), total_weight))
        candidate.contradiction = clamp(
            candidate.contradiction + sum(
                item["weight"] for item in results if not item["satisfied"] and item["hard"]
            )
        )
        updated.append(candidate)
    observed_event_graph = {
        "nodes": sorted({
            value for relation in request.episodic_relations
            if relation.status in {"direct_evidence", "episodically_retrieved"}
            and (not request.target_episode_id or relation.episode_id == request.target_episode_id)
            for value in (relation.subject, relation.object)
        }),
        "edges": [
            {
                "relation_id": relation.relation_id,
                **relation.triple(),
                "source_id": relation.source_id,
                "status": relation.status,
                "episode_id": relation.episode_id,
            }
            for relation in request.episodic_relations
            if relation.status in {"direct_evidence", "episodically_retrieved"}
            and (not request.target_episode_id or relation.episode_id == request.target_episode_id)
        ],
        "source_relations_only": True,
    }
    proposed_event_graph = [
        {"candidate_id": candidate.candidate_id, **dict(candidate.content)}
        for candidate in updated
        if candidate.feature_type == "relation" and isinstance(candidate.content, Mapping)
    ]
    return {
        "candidates": updated,
        "observed_event_graph": observed_event_graph,
        "inferred_relation_candidates": proposed_event_graph,
        "constraint_count": len(constraints),
        "violations": [
            {"candidate_id": candidate.candidate_id, **item}
            for candidate in updated for item in candidate.constraint_results if not item["satisfied"]
        ],
        "coherence_is_not_truth": True,
    }


def config_epsilon() -> float:
    """Return a small positive denominator for normalized constraint aggregation."""
    return 1e-9


def monitor_candidate_sources(
    candidates: Sequence[DetailCandidate],
    fragments: Sequence[EpisodicFragment],
    sources: Sequence[SourceRecord],
    config: Point20Config,
    relations: Sequence[EpisodicRelation] = (),
    target_episode_id: str = "",
) -> dict[str, Any]:
    """Estimate source probabilities while keeping competing-episode details from supporting this event."""
    results: list[dict[str, Any]] = []
    for candidate in candidates:
        scores = {name: 0.01 for name in (
            "original_perception", "episodic_retrieval", "schema_inference",
            "imagination", "post_event_information", "later_testimony", "unknown",
        )}
        scores["schema_inference"] += 0.75
        evidence_ids = set(candidate.evidence_ids)
        matching_fragments = [
            item for item in fragments
            if item.feature_type == candidate.feature_type
            and value_key(item.content) == value_key(candidate.content)
            and (not target_episode_id or item.episode_id == target_episode_id)
        ]
        for fragment in matching_fragments:
            source_type = fragment.source_type
            scores[source_type] = scores.get(source_type, 0.0) + (
                fragment.reliability if hasattr(fragment, "reliability") else
                fragment.retrieval_strength * fragment.episodic_context
            )
            if fragment.detail_id in evidence_ids:
                scores[source_type] += 0.25
        if candidate.feature_type == "relation" and isinstance(candidate.content, Mapping):
            for relation in relations:
                if (
                    relation.triple() == dict(candidate.content)
                    and (not target_episode_id or relation.episode_id == target_episode_id)
                ):
                    scores[relation.source_type] = scores.get(relation.source_type, 0.0) + (
                        relation.retrieval_strength * relation.episodic_context
                    )
                    if relation.relation_id in evidence_ids:
                        scores[relation.source_type] += 0.25
        independent_sources: dict[tuple[str, str, str], SourceRecord] = {}
        for source in sources:
            group = source.independence_group or source.source_id
            key = (group, source.source_type, value_key(source.content))
            previous = independent_sources.get(key)
            if previous is None or source.reliability > previous.reliability:
                independent_sources[key] = source
        for source in independent_sources.values():
            detail_match = source.detail_id in {candidate.candidate_id, candidate.feature_type, ""}
            content_match = value_key(source.content) == value_key(candidate.content) if source.content is not None else False
            if detail_match and content_match:
                scores[source.source_type] = scores.get(source.source_type, 0.0) + source.reliability
        probabilities = softmax(
            [math.log(max(config.epsilon, scores[name])) for name in scores],
            config.source_temperature,
        )
        ranked = sorted(zip(scores, probabilities), key=lambda pair: pair[1], reverse=True)
        source_probabilities = {name: probability for name, probability in zip(scores, probabilities)}
        assigned, top_probability = ranked[0]
        candidate.source_probabilities = source_probabilities
        candidate.assigned_source = assigned if top_probability >= config.source_assignment_threshold else "unknown"
        candidate.source_uncertainty = clamp(1.0 - top_probability)
        candidate.source_specificity = max(
            candidate.source_specificity,
            source_probabilities.get("original_perception", 0.0)
            + source_probabilities.get("episodic_retrieval", 0.0),
        )
        if candidate.assigned_source == "unknown":
            candidate.status = "source_uncertain"
        elif candidate.assigned_source in {"original_perception", "episodic_retrieval"}:
            if candidate.evidence_strength >= config.retrieval_promotion_threshold and candidate.episodic_context >= config.retrieval_promotion_threshold and candidate.contradiction < 0.25:
                candidate.status = "episodically_retrieved"
                candidate.inference_proportion = 0.0
            else:
                candidate.status = "source_uncertain"
        elif candidate.assigned_source in {"post_event_information", "later_testimony"}:
            candidate.status = "suggested_after_event"
        results.append({
            "candidate_id": candidate.candidate_id,
            "source_probabilities": source_probabilities,
            "assigned_source": candidate.assigned_source,
            "source_uncertainty": candidate.source_uncertainty,
            "source_specificity": candidate.source_specificity,
        })
    return {"candidates": list(candidates), "source_assignments": results}


def assess_familiarity_and_recollection(
    candidates: Sequence[DetailCandidate],
    fragments: Sequence[EpisodicFragment],
    sources: Sequence[SourceRecord],
    precision: str,
    config: Point20Config,
    relations: Sequence[EpisodicRelation] = (),
    target_episode_id: str = "",
) -> dict[str, Any]:
    """Separate general familiarity from source-specific recollection of the requested episode."""
    output: list[dict[str, Any]] = []
    for candidate in candidates:
        matches = [
            item for item in fragments
            if item.feature_type == candidate.feature_type
            and value_key(item.content) == value_key(candidate.content)
        ]
        familiar_sources = [
            item for item in sources
            if item.content is not None and value_key(item.content) == value_key(candidate.content)
        ]
        repetition = min(1.0, len(familiar_sources) / 4.0)
        relation_matches = [
            item for item in relations
            if candidate.feature_type == "relation" and isinstance(candidate.content, Mapping)
            and item.triple() == dict(candidate.content)
        ]
        fragment_familiarity = max((item.familiarity for item in matches), default=0.0)
        relation_familiarity = min(1.0, len(relation_matches) / 3.0)
        familiarity = clamp(0.50 * fragment_familiarity + 0.20 * relation_familiarity + 0.30 * repetition)
        target_matches = [
            item for item in matches
            if not target_episode_id or item.episode_id == target_episode_id
        ]
        target_relation_matches = [
            item for item in relation_matches
            if not target_episode_id or item.episode_id == target_episode_id
        ]
        recollection = clamp(
            (
                sum(item.recollection * item.episodic_context * item.retrieval_strength for item in target_matches)
                + sum(item.episodic_context * item.retrieval_strength for item in target_relation_matches)
            ) / max(1, len(target_matches) + len(target_relation_matches))
        )
        uncertainty = candidate.source_uncertainty
        familiarity_weight = config.familiarity_weight * (0.70 if precision == "exact" else 1.0)
        recollection_weight = config.recollection_decision_weight * (1.15 if precision == "exact" else 0.85)
        decision_signal = (
            familiarity_weight * familiarity
            + recollection_weight * recollection
            - 0.35 * uncertainty
        )
        candidate.familiarity = familiarity
        candidate.recollection = recollection
        output.append({
            "candidate_id": candidate.candidate_id,
            "familiarity": familiarity,
            "source_specific_recollection": recollection,
            "source_uncertainty": uncertainty,
            "decision_signal": decision_signal,
            "familiarity_not_proof": True,
        })
    return {"candidates": list(candidates), "dimensions": output}


def search_competing_episodes(
    candidates: Sequence[DetailCandidate],
    request: Point20Request,
    fragments: Sequence[EpisodicFragment],
    relations: Sequence[EpisodicRelation] = (),
) -> dict[str, Any]:
    """Compare candidate details with similar episodes to identify source blending and overlap."""
    comparisons: list[dict[str, Any]] = []
    for candidate in candidates:
        matches: list[str] = []
        target_support = any(
            item.feature_type == candidate.feature_type
            and value_key(item.content) == value_key(candidate.content)
            and (not request.target_episode_id or item.episode_id == request.target_episode_id)
            for item in fragments
        )
        if candidate.feature_type == "relation" and isinstance(candidate.content, Mapping):
            target_support = target_support or any(
                item.triple() == dict(candidate.content)
                and (not request.target_episode_id or item.episode_id == request.target_episode_id)
                for item in relations
            )
        for fragment in fragments:
            if (
                request.target_episode_id
                and fragment.episode_id
                and fragment.episode_id != request.target_episode_id
                and fragment.feature_type == candidate.feature_type
                and value_key(fragment.content) == value_key(candidate.content)
            ):
                matches.append(fragment.episode_id)
        if candidate.feature_type == "relation" and isinstance(candidate.content, Mapping):
            for relation in relations:
                if (
                    request.target_episode_id
                    and relation.episode_id
                    and relation.episode_id != request.target_episode_id
                    and relation.triple() == dict(candidate.content)
                ):
                    matches.append(relation.episode_id)
        for episode in request.competing_episodes:
            episode_id = str(episode.get("episode_id", ""))
            raw_fragments = episode.get("fragments", ())
            if isinstance(raw_fragments, Mapping):
                raw_fragments = [raw_fragments]
            for raw in raw_fragments:
                if not isinstance(raw, Mapping):
                    continue
                feature = str(raw.get("feature_type", raw.get("feature", ""))).casefold()
                value = raw.get("content", raw.get("value"))
                if feature == candidate.feature_type and value_key(value) == value_key(candidate.content):
                    matches.append(episode_id or "unspecified_competing_episode")
                    break
            if candidate.feature_type == "relation":
                for raw in episode.get("relations", ()):
                    if not isinstance(raw, Mapping):
                        continue
                    triple = {key: raw.get(key) for key in ("subject", "predicate", "object")}
                    if triple == dict(candidate.content):
                        matches.append(episode_id or "unspecified_competing_episode")
                        break
        candidate.competing_episode_ids = sorted(set(matches))
        only_competing_source = bool(matches) and not target_support
        if only_competing_source:
            candidate.contradiction = clamp(candidate.contradiction + 0.15)
            candidate.source_uncertainty = clamp(candidate.source_uncertainty + 0.10)
        comparisons.append({
            "candidate_id": candidate.candidate_id,
            "target_episode_support": target_support,
            "matching_competing_episode_ids": candidate.competing_episode_ids,
            "possible_episode_blend": only_competing_source,
            "absence_of_target_match_is_not_a_contradiction": True,
        })
    return {"candidates": list(candidates), "comparisons": comparisons}


def calibrate_detail_confidence(
    candidates: Sequence[DetailCandidate],
    config: Point20Config,
) -> dict[str, Any]:
    """Estimate each detail's confidence independently of how coherent the total story sounds."""
    results: list[dict[str, Any]] = []
    for candidate in candidates:
        inferred = 1.0 if candidate.status == "inferred" or candidate.inference_proportion >= 0.99 else candidate.inference_proportion
        logit = (
            config.calibration_bias
            + config.evidence_weight * candidate.evidence_strength
            + config.recollection_weight * candidate.recollection
            - config.inference_penalty * inferred
            - config.contradiction_penalty * candidate.contradiction
            - config.source_uncertainty_penalty * candidate.source_uncertainty
        )
        confidence = sigmoid(logit)
        candidate.confidence = confidence
        results.append({
            "candidate_id": candidate.candidate_id,
            "confidence": confidence,
            "evidence_strength": candidate.evidence_strength,
            "source_specific_recollection": candidate.recollection,
            "inferred_proportion": inferred,
            "contradiction": candidate.contradiction,
            "source_uncertainty": candidate.source_uncertainty,
            "posterior": candidate.posterior,
            "coherence_used_as_confidence": False,
        })
    return {
        "candidates": list(candidates),
        "per_detail": results,
        "calibration_note": "Illustrative logistic coefficients require empirical calibration; subjective confidence is not accuracy.",
    }


def choose_response_policy(
    request: Point20Request,
    ledger: Mapping[str, Any],
    candidates: Sequence[DetailCandidate],
    config: Point20Config,
) -> dict[str, Any]:
    """Select report, qualify, clarify, or omit using task-specific expected-loss costs."""
    high_stakes = str(request.context.get("stakes", "normal")).casefold() in {"high", "critical"}
    precision = request.precision
    false_claim_cost = 5.0 if high_stakes or precision == "exact" else 2.0
    if precision == "gist":
        false_claim_cost *= 1.0 - 0.5 * config.gist_detail_prior_weight
    omission_cost = 0.8 if precision == "gist" else 1.4
    clarification_cost = 0.25 if high_stakes or precision == "exact" else 0.45
    grouped: dict[str, list[DetailCandidate]] = {}
    for item in candidates:
        grouped.setdefault(item.feature_type, []).append(item)
    decisions: list[dict[str, Any]] = []
    feature_importance = request.context.get("feature_importance", {})
    if not isinstance(feature_importance, Mapping):
        feature_importance = {}
    for feature, feature_row in ledger["features"].items():
        importance = clamp(float(feature_importance.get(feature, feature_row["centrality"] or 0.5)))
        options: list[dict[str, Any]] = []
        for candidate in grouped.get(feature, []):
            probability = clamp(max(candidate.confidence, candidate.feedback_probability))
            report_loss = (1.0 - probability) * false_claim_cost
            qualify_loss = 0.28 * (1.0 - probability) * false_claim_cost + 0.22
            clarify_loss = clarification_cost + 0.10 * (1.0 - probability)
            omit_loss = omission_cost * importance
            losses = {
                "report": report_loss,
                "qualify": qualify_loss,
                "clarify": clarify_loss,
                "omit": omit_loss,
            }
            action = min(losses, key=losses.get)
            if action == "report" and probability < config.report_confidence_threshold:
                action = "qualify" if probability >= config.clarify_confidence_threshold else (
                    "clarify" if high_stakes or precision == "exact" or importance >= 0.7 else "omit"
                )
            if candidate.status in {"inferred", "source_uncertain", "suggested_after_event"} and action == "report":
                action = "qualify"
            options.append({
                "candidate_id": candidate.candidate_id,
                "content": candidate.content,
                "action": action,
                "expected_losses": losses,
                "confidence": probability,
                "status": candidate.status,
            })
        if options:
            selected = min(
                options,
                key=lambda item: (
                    item["expected_losses"][item["action"]],
                    item["candidate_id"],
                ),
            )
            decisions.append({
                "feature_type": feature,
                "selected": selected,
                "alternatives": options,
                "reason": "minimum expected response loss subject to provenance/status guardrails",
            })
        elif feature_row["status"] in {"not_retrieved", "weakly_retrieved"}:
            action = "clarify" if high_stakes or precision == "exact" or importance >= 0.7 else "omit"
            decisions.append({
                "feature_type": feature,
                "selected": {"candidate_id": None, "content": None, "action": action,
                             "expected_losses": {"clarify": clarification_cost, "omit": omission_cost * importance},
                             "confidence": 0.0, "status": "unknown"},
                "alternatives": [],
                "reason": "no supported or schema-generated candidate; preserve unknown status",
            })
    return {
        "decisions": decisions,
        "reported_features": [item["feature_type"] for item in decisions if item["selected"]["action"] == "report"],
        "qualified_features": [item["feature_type"] for item in decisions if item["selected"]["action"] == "qualify"],
        "clarification_features": [item["feature_type"] for item in decisions if item["selected"]["action"] == "clarify"],
        "omitted_features": [item["feature_type"] for item in decisions if item["selected"]["action"] == "omit"],
        "expected_loss_policy": {
            "false_claim_cost": false_claim_cost,
            "omission_cost": omission_cost,
            "clarification_cost": clarification_cost,
            "high_stakes": high_stakes,
            "precision": precision,
        },
    }


def _has_cycle(nodes: Mapping[str, Any], edges: Sequence[tuple[str, str]]) -> bool:
    """Detect cycles in a provenance graph with a depth-first traversal."""
    adjacency: dict[str, list[str]] = {key: [] for key in nodes}
    for left, right in edges:
        adjacency.setdefault(left, []).append(right)
        adjacency.setdefault(right, [])
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> bool:
        """Return true when a recursive edge reaches a node already on the active path."""
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        for child in adjacency.get(node, ()):
            if visit(child):
                return True
        visiting.remove(node)
        visited.add(node)
        return False

    return any(visit(node) for node in adjacency if node not in visited)


def build_provenance_graph(
    request: Point20Request,
    candidates: Sequence[DetailCandidate],
    prior_sources: Sequence[SourceRecord] = (),
) -> dict[str, Any]:
    """Build a source/detail/schema provenance graph and preserve the timeline of later information."""
    nodes: dict[str, dict[str, Any]] = {
        f"event:{request.event_id}": {"node_type": "original_event", "event_id": request.event_id}
    }
    for episode_id in {
        item.episode_id for item in (*request.episodic_fragments, *request.episodic_relations)
        if item.episode_id
    }:
        nodes.setdefault(
            f"event:{episode_id}",
            {"node_type": "episodic_event", "event_id": episode_id},
        )
    edges: list[tuple[str, str]] = []
    source_records = [*request.source_history, *prior_sources]
    source_by_id = {item.source_id: item for item in source_records}
    for relation in request.episodic_relations:
        source_id = relation.source_id or f"relation-source:{relation.relation_id}"
        if source_id not in source_by_id:
            source_by_id[source_id] = SourceRecord(
                source_id=source_id,
                source_type=relation.source_type,
                detail_id=relation.relation_id,
                content=relation.triple(),
                reliability=relation.retrieval_strength,
                timestamp=relation.timestamp,
                parent_source_ids=(f"event:{relation.episode_id}",) if relation.episode_id else (),
                operation="episodic_relation",
            )
    for fragment in request.episodic_fragments:
        source_id = fragment.source_id or f"fragment-source:{fragment.detail_id}"
        if source_id not in source_by_id:
            source_by_id[source_id] = SourceRecord(
                source_id=source_id,
                source_type=fragment.source_type,
                detail_id=fragment.detail_id,
                content=fragment.content,
                reliability=fragment.retrieval_strength,
                timestamp=fragment.timestamp,
                parent_source_ids=(fragment.provenance_ids or ((f"event:{fragment.episode_id}",) if fragment.episode_id else ())),
                source_features=fragment.source_features,
                operation="episodic_fragment",
            )
    source_records = list(source_by_id.values())
    for source in source_records:
        nodes[f"source:{source.source_id}"] = {
            "node_type": "source",
            **source.to_dict(),
        }
        if source.parent_source_ids:
            parents = source.parent_source_ids
        elif source.source_type in {"original_perception", "episodic_retrieval"}:
            parents = (f"event:{request.event_id}",)
        else:
            parents = (f"external:{source.source_id}",)
        for parent in parents:
            parent_key = parent if parent.startswith(("source:", "event:", "detail:", "schema:", "external:")) else f"source:{parent}"
            if parent_key not in nodes:
                if parent_key.startswith("schema:"):
                    nodes[parent_key] = {"node_type": "schema", "schema_id": parent_key.split(":", 1)[1]}
                elif parent_key.startswith("external:"):
                    nodes[parent_key] = {"node_type": "external_source_origin", "source_id": parent_key.split(":", 1)[1]}
                else:
                    nodes[parent_key] = {"node_type": "unresolved_parent_reference", "source_id": parent}
            edges.append((parent_key, f"source:{source.source_id}"))
        if source.detail_id:
            detail_key = f"detail:{source.detail_id}"
            nodes.setdefault(detail_key, {"node_type": "detail", "detail_id": source.detail_id})
            edges.append((f"source:{source.source_id}", detail_key))
    for candidate in candidates:
        candidate_key = f"candidate:{candidate.candidate_id}"
        nodes[candidate_key] = {
            "node_type": "candidate_detail",
            "candidate_id": candidate.candidate_id,
            "feature_type": candidate.feature_type,
            "content": candidate.content,
            "status": candidate.status,
            "assigned_source": candidate.assigned_source,
            "provenance": list(candidate.provenance),
        }
        if candidate.schema_id:
            schema_key = f"schema:{candidate.schema_id}"
            nodes.setdefault(schema_key, {"node_type": "schema", "schema_id": candidate.schema_id})
            edges.append((schema_key, candidate_key))
        for source_id in candidate.evidence_ids:
            source_key = f"detail:{source_id}"
            nodes.setdefault(source_key, {"node_type": "detail", "detail_id": source_id})
            edges.append((source_key, candidate_key))
    feedback_timeline: list[dict[str, Any]] = []
    for feedback in request.feedback:
        source_key = f"source:{feedback.source_id}"
        nodes.setdefault(source_key, {
            "node_type": "feedback_source",
            "source_id": feedback.source_id,
            "source_type": feedback.source_type,
            "reliability": feedback.reliability,
            "timestamp": feedback.timestamp,
            "independence_group": feedback.independence_group,
        })
        feedback_key = f"feedback:{feedback.feedback_id}"
        nodes[feedback_key] = {
            "node_type": "feedback",
            **feedback.to_dict(),
        }
        edges.append((source_key, feedback_key))
        feedback_timeline.append({
            "feedback_id": feedback.feedback_id,
            "feature_type": feedback.feature_type,
            "source_id": feedback.source_id,
            "timestamp": feedback.timestamp,
            "verified": feedback.verified,
        })
    cycle = _has_cycle(nodes, edges)
    if cycle:
        raise ValueError("source provenance contains a cycle; lineage must be a directed acyclic graph")
    return {
        "nodes": nodes,
        "edges": [{"from": left, "to": right} for left, right in edges],
        "source_timeline": [
            {"source_id": item.source_id, "source_type": item.source_type,
             "detail_id": item.detail_id, "content": item.content, "reliability": item.reliability,
             "timestamp": item.timestamp, "independence_group": item.independence_group,
             "parent_source_ids": list(item.parent_source_ids), "operation": item.operation}
            for item in sorted(source_records, key=lambda row: (row.timestamp, row.source_id))
        ],
        "feedback_timeline": sorted(feedback_timeline, key=lambda row: (row["timestamp"], row["feedback_id"])),
        "cycle_detected": cycle,
        "earliest_source_preserved": True,
        "repeated_retelling_is_not_original_perception": True,
    }


def monitor_retelling_and_post_event_information(
    request: Point20Request,
    state: Point20State,
    candidates: Sequence[DetailCandidate],
    prior_sources: Sequence[SourceRecord] = (),
) -> dict[str, Any]:
    """Track repeated reports and later suggestions without overwriting the earliest source labels."""
    graph = build_provenance_graph(request, candidates, prior_sources)
    sources = graph["source_timeline"]
    after_event = [
        item for item in sources
        if item["source_type"] in {"post_event_information", "later_testimony", "imagination", "schema_inference"}
    ]
    repeated: dict[str, int] = {}
    for item in sources:
        if item["detail_id"]:
            repeated[item["detail_id"]] = repeated.get(item["detail_id"], 0) + 1
    drift = [
        {"detail_id": detail_id, "source_count": count,
         "retelling_or_post_event_exposure": count > 1}
        for detail_id, count in repeated.items()
    ]
    relation_drift: dict[str, int] = {}
    for source in sources:
        if isinstance(source.get("content"), Mapping) and all(
            key in source["content"] for key in ("subject", "predicate", "object")
        ):
            relation_key = "|".join(str(source["content"][key]) for key in ("subject", "predicate", "object"))
            relation_drift[relation_key] = relation_drift.get(relation_key, 0) + 1
    state.source_versions.extend(
        {
            "source_id": item["source_id"], "source_type": item["source_type"],
            "detail_id": item["detail_id"], "content": item.get("content"),
            "reliability": item.get("reliability", 0.5), "timestamp": item["timestamp"],
            "independence_group": item.get("independence_group", ""),
            "event_id": request.event_id,
            "parent_source_ids": list(item.get("parent_source_ids", ())),
            "operation": item.get("operation", "reported"),
        }
        for item in sources if item.get("operation") != "prior_reconstruction"
    )
    state.source_versions.extend(
        {
            "source_id": f"reconstruction:{request.event_id}:{state.run_count}:{candidate.candidate_id}",
            "source_type": "schema_inference",
            "detail_id": candidate.candidate_id,
            "content": candidate.content,
            "reliability": candidate.confidence,
            "timestamp": float(state.run_count),
            "independence_group": candidate.candidate_id,
            "parent_source_ids": [f"schema:{candidate.schema_id}"] if candidate.schema_id else [],
            "operation": "reconstructive_candidate",
        }
        for candidate in candidates
    )
    state.source_versions = state.source_versions[-500:]
    return {
        "provenance_graph": graph,
        "post_event_or_reconstruction_sources": after_event,
        "repeated_detail_exposure": drift,
        "repeated_relation_exposure": [
            {"triple": key, "source_count": count, "retelling_or_post_event_exposure": count > 1}
            for key, count in relation_drift.items()
        ],
        "source_relabeling_performed": False,
        "reconsolidation_assumption": "No claim that every retrieval triggers reconsolidation.",
    }


def update_reconstruction_from_feedback(
    candidates: Sequence[DetailCandidate],
    feedback: Sequence[FeedbackRecord],
    state: Point20State,
    config: Point20Config,
    event_id: str = "unspecified",
) -> dict[str, Any]:
    """Update candidate odds with reliable independent feedback while preserving prior versions."""
    updates: list[dict[str, Any]] = []
    ignored: list[dict[str, Any]] = []
    by_id = {item.candidate_id: item for item in candidates}
    for record in sorted(feedback, key=lambda item: (item.timestamp, item.feedback_id)):
        group = f"{event_id}|{record.independence_group or record.source_id}"
        if group in state.feedback_groups_seen:
            ignored.append({"feedback_id": record.feedback_id, "reason": "duplicate_independence_group"})
            continue
        if record.reliability < config.min_feedback_reliability:
            ignored.append({"feedback_id": record.feedback_id, "reason": "source_reliability_below_threshold"})
            continue
        matches = []
        if record.candidate_id and record.candidate_id in by_id:
            matches = [by_id[record.candidate_id]]
        else:
            matches = [
                item for item in candidates
                if item.feature_type == record.feature_type
                and value_key(item.content) == value_key(record.value)
            ]
        if not matches:
            ignored.append({"feedback_id": record.feedback_id, "reason": "no_matching_candidate"})
            continue
        state.feedback_groups_seen.append(group)
        for candidate in matches:
            prior = clamp(candidate.feedback_probability if candidate.feedback_probability > 0.0 else (
                candidate.schema_prior * candidate.evidence_likelihood
                if candidate.schema_prior > 0.0 else candidate.confidence
            ))
            prior_odds = prior / max(config.epsilon, 1.0 - prior)
            likelihood_if_true = record.reliability if record.supports else 1.0 - record.reliability
            likelihood_if_false = 1.0 - record.reliability if record.supports else record.reliability
            likelihood_ratio = likelihood_if_true / max(config.epsilon, likelihood_if_false)
            posterior_odds = prior_odds * likelihood_ratio
            updated = clamp(posterior_odds / (1.0 + posterior_odds))
            before = candidate.feedback_probability
            candidate.feedback_probability = updated
            candidate.feedback_history.append({
                "feedback_id": record.feedback_id,
                "source_id": record.source_id,
                "reliability": record.reliability,
                "supports": record.supports,
                "prior_probability": prior,
                "likelihood_ratio": likelihood_ratio,
                "feedback_probability": updated,
            })
            state.reconstruction_versions.append({
                "event_id": event_id,
                "candidate_id": candidate.candidate_id,
                "feedback_id": record.feedback_id,
                "feedback_probability_before": before,
                "feedback_probability_after": updated,
                "source_id": record.source_id,
            })
            updates.append({
                "candidate_id": candidate.candidate_id,
                "feedback_id": record.feedback_id,
                "prior_odds": prior_odds,
                "likelihood_ratio": likelihood_ratio,
                "posterior_odds": posterior_odds,
                "feedback_probability": updated,
            })
    return {
        "updates": updates,
        "ignored_feedback": ignored,
        "independent_feedback_groups_seen": list(state.feedback_groups_seen),
        "previous_versions_preserved": True,
        "schema_updates_performed": False,
    }


def update_general_schema_knowledge(
    request: Point20Request,
    state: Point20State,
    config: Point20Config,
) -> dict[str, Any]:
    """Update general schema feature counts only from verified, reliable, independent feedback."""
    updates: list[dict[str, Any]] = []
    ignored: list[dict[str, Any]] = []
    used_groups: set[str] = set()
    schema_by_id = {schema.schema_id: schema for schema in request.schemas}
    for record in sorted(request.feedback, key=lambda item: (item.timestamp, item.feedback_id)):
        group = "|".join((record.schema_id, record.feature_type, value_key(record.value), record.independence_group or record.source_id))
        if not record.use_for_schema_update:
            continue
        if group in state.schema_feedback_groups_seen or group in used_groups:
            ignored.append({"feedback_id": record.feedback_id, "reason": "duplicate_or_dependent_feedback"})
            continue
        if not record.verified or record.reliability < config.min_feedback_reliability:
            ignored.append({"feedback_id": record.feedback_id, "reason": "not_verified_or_unreliable"})
            continue
        if not record.schema_id:
            ignored.append({"feedback_id": record.feedback_id, "reason": "schema_id_missing"})
            continue
        key = "|".join((record.schema_id, record.feature_type, value_key(record.value)))
        schema = schema_by_id.get(record.schema_id)
        if schema is not None and record.feature_type == "relation" and isinstance(record.value, Mapping):
            triple_key = value_key(record.value)
            base_probability = max(
                (
                    float(template["probability"])
                    for template in schema.relation_templates
                    if value_key({key: template[key] for key in ("subject", "predicate", "object")}) == triple_key
                ),
                default=0.5,
            )
        else:
            base_probability = (
                schema.feature_probabilities.get(record.feature_type, {}).get(value_key(record.value), 0.5)
                if schema is not None else 0.5
            )
        prior_strength = config.schema_prior_strength
        counts = state.schema_beta_counts.setdefault(
            key,
            {
                "alpha": max(config.epsilon, base_probability * prior_strength),
                "beta": max(config.epsilon, (1.0 - base_probability) * prior_strength),
            },
        )
        before = dict(counts)
        increment = config.schema_learning_rate * record.reliability
        if record.supports:
            counts["alpha"] += increment
        else:
            counts["beta"] += increment
        used_groups.add(group)
        state.schema_feedback_groups_seen.append(group)
        updates.append({
            "schema_id": record.schema_id,
            "feature_type": record.feature_type,
            "value": record.value,
            "counts_before": before,
            "counts_after": dict(counts),
            "posterior_mean": counts["alpha"] / (counts["alpha"] + counts["beta"]),
            "feedback_id": record.feedback_id,
            "independence_group": group,
        })
    return {
        "updates": updates,
        "ignored_feedback": ignored,
        "schema_parameter_count": len(state.schema_beta_counts),
        "event_record_overwritten": False,
        "learning_source": "verified_feedback_only",
    }


def finalize_reconstruction(
    request: Point20Request,
    ledger: Mapping[str, Any],
    candidates: Sequence[DetailCandidate],
    policy: Mapping[str, Any],
    provenance_graph: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble evidence classes, inferred alternatives, uncertainty, confidence, and source lineage."""
    supported: list[dict[str, Any]] = []
    existing_supported_ids: set[str] = set()
    for fragment in request.episodic_fragments:
        if (
            fragment.status in {"direct_evidence", "episodically_retrieved"}
            and (not request.target_episode_id or fragment.episode_id == request.target_episode_id)
        ):
            supported.append({
                "detail_id": fragment.detail_id,
                "feature_type": fragment.feature_type,
                "content": fragment.content,
                "status": fragment.status,
                "confidence": clamp(fragment.retrieval_strength * fragment.episodic_context),
                "source_type": fragment.source_type,
                "source_id": fragment.source_id,
                "episode_id": fragment.episode_id,
            })
            existing_supported_ids.add(fragment.detail_id)
    episode_uncertain_details = [
        dict(value)
        for row in ledger["features"].values()
        for value in row.get("values", ())
        if value.get("status") == "source_uncertain"
    ]
    competing_episode_details = [
        {"feature_type": feature, **dict(value)}
        for feature, row in ledger["features"].items()
        for value in row.get("competing_episode_values", ())
    ]
    promoted = [
        item for item in candidates
        if item.status == "episodically_retrieved"
        and item.feature_type != "relation"
        and item.evidence_strength >= 0.75
        and item.episodic_context >= 0.75
        and item.contradiction < 0.25
    ]
    promoted_relations = [
        item for item in candidates
        if item.status == "episodically_retrieved"
        and item.feature_type == "relation"
        and isinstance(item.content, Mapping)
        and item.evidence_strength >= 0.75
        and item.episodic_context >= 0.75
        and item.contradiction < 0.25
    ]
    for item in promoted:
        if item.candidate_id not in existing_supported_ids:
            supported.append({
                "detail_id": item.candidate_id,
                "feature_type": item.feature_type,
                "content": item.content,
                "status": "episodically_retrieved",
                "confidence": item.confidence,
                "source_type": item.assigned_source,
                "source_id": "source-monitoring-inference-promotion",
                "episode_id": request.target_episode_id or request.event_id,
                "supporting_evidence_ids": list(item.evidence_ids),
            })
    supported_relations = [
        {
            "relation_id": item.relation_id,
            **item.triple(),
            "status": item.status,
            "confidence": clamp(item.retrieval_strength * item.episodic_context),
            "source_type": item.source_type,
            "source_id": item.source_id,
            "episode_id": item.episode_id,
        }
        for item in request.episodic_relations
        if any(
            row["relation_id"] == item.relation_id and row["reactivated"] and row["episode_match"]
            for row in ledger.get("relations", ())
        )
        and (not request.target_episode_id or item.episode_id == request.target_episode_id)
        and item.status in {"direct_evidence", "episodically_retrieved"}
    ]

    for item in promoted_relations:
        triple = dict(item.content)
        if not any(all(existing.get(key) == triple[key] for key in ("subject", "predicate", "object"))
                   for existing in supported_relations):
            supported_relations.append({
                "relation_id": item.candidate_id,
                **triple,
                "status": "episodically_retrieved",
                "confidence": item.confidence,
                "source_type": item.assigned_source,
                "source_id": "source-monitoring-inference-promotion",
                "episode_id": request.target_episode_id or request.event_id,
                "supporting_evidence_ids": list(item.evidence_ids),
            })
    inferred = [item.to_dict() for item in candidates
                if item.status == "inferred" and item.feature_type != "relation"
                and item.contradiction < 0.75]
    post_event_suggested = [
        item.to_dict() for item in candidates
        if item.status == "suggested_after_event" and item.feature_type != "relation"
    ]
    source_uncertain = [
        item.to_dict() for item in candidates
        if (item.status == "source_uncertain" or item.assigned_source == "unknown")
        and item.feature_type != "relation"
    ]
    contradicted = [
        item.to_dict() for item in candidates
        if (item.status == "contradicted" or item.contradiction >= 0.75)
        and item.feature_type != "relation"
    ]
    inferred_relations = [
        item.to_dict() for item in candidates
        if item.feature_type == "relation" and item.status == "inferred"
        and item.contradiction < 0.75
    ]
    post_event_suggested_relations = [
        item.to_dict() for item in candidates
        if item.feature_type == "relation" and item.status == "suggested_after_event"
    ] + [
        dict(item) for item in ledger.get("relations", ())
        if item.get("reactivated") and item.get("episode_match")
        and item.get("status") == "suggested_after_event"
    ] + [
        dict(item) for item in ledger.get("relations", ())
        if item.get("reactivated") and item.get("episode_match")
        and item.get("status") == "suggested_after_event"
    ]
    source_uncertain_relations = [
        item.to_dict() for item in candidates
        if item.feature_type == "relation" and (
            item.status == "source_uncertain" or item.assigned_source == "unknown"
        )
    ]
    contradicted_relations = [
        item.to_dict() for item in candidates
        if item.feature_type == "relation" and (
            item.status == "contradicted" or item.contradiction >= 0.75
        )
    ] + [
        dict(item) for item in ledger.get("relations", ())
        if item.get("reactivated") and item.get("episode_match")
        and item.get("status") == "contradicted"
    ]
    episode_uncertain_relations = [
        dict(item) for item in ledger.get("relations", ())
        if item.get("reactivated") and (item.get("episode_unidentified") or item.get("status") in {"source_uncertain", "unknown"})
    ]
    competing_relations = [
        dict(item) for item in ledger.get("relations", ())
        if item.get("reactivated") and not item.get("episode_match")
        and not item.get("episode_unidentified")
    ]
    unresolved_relations = [
        dict(item) for item in ledger.get("relations", ())
        if (item.get("episode_match") or item.get("episode_unidentified"))
        and item.get("status") in {"not_retrieved", "weakly_retrieved", "contradicted", "unknown"}
    ]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in candidates:
        grouped.setdefault(item.feature_type, []).append({
            "candidate_id": item.candidate_id,
            "content": item.content,
            "posterior": item.posterior,
            "confidence": item.confidence,
            "status": item.status,
            "assigned_source": item.assigned_source,
            "contradiction": item.contradiction,
        })
    unresolved = [
        {"feature_type": feature, "status": row["status"], "likely_encoded": row["likely_encoded"],
         "centrality": row["centrality"], "unavailable_not_absent": row["unavailable_not_absent"]}
        for feature, row in ledger["features"].items()
        if row["status"] in {"not_retrieved", "weakly_retrieved", "contradicted"}
    ]
    selected_decisions = [dict(item) for item in policy["decisions"]]
    gist = request.context.get("gist")
    if gist is None:
        gist = " ".join(str(item["content"]) for item in supported)
    return {
        "event_id": request.event_id,
        "gist": gist,
        "supported_details": supported,
        "supported_relations": supported_relations,
        "event_graph": dict(ledger.get("event_graph", {})),
        "inferred_details": inferred,
        "inferred_relations": inferred_relations,
        "post_event_suggested_relations": post_event_suggested_relations,
        "source_uncertain_relations": source_uncertain_relations,
        "contradicted_relations": contradicted_relations,
        "unresolved_relations": unresolved_relations,
        "episode_uncertain_relations": episode_uncertain_relations,
        "competing_relations": competing_relations,
        "post_event_suggested_details": post_event_suggested,
        "source_uncertain_details": source_uncertain,
        "episode_uncertain_details": episode_uncertain_details,
        "competing_episode_details": competing_episode_details,
        "contradicted_details": contradicted,
        "unresolved_features": unresolved,
        "unresolved_alternatives": grouped,
        "response_decisions": selected_decisions,
        "confidence_by_detail": {
            item.candidate_id: {"confidence": item.confidence, "posterior": item.posterior,
                                "feedback_probability": item.feedback_probability,
                                "status": item.status, "source": item.assigned_source}
            for item in candidates
        },
        "provenance_graph": provenance_graph,
        "coherent_does_not_mean_historically_accurate": True,
        "not_retrieved_does_not_mean_absent": True,
    }


def brier_score(probabilities: Sequence[float], outcomes: Sequence[int | bool]) -> float:
    """Compute mean squared probability error for externally labelled calibration examples."""
    if len(probabilities) != len(outcomes):
        raise ValueError("probabilities and outcomes must have equal lengths")
    if not probabilities:
        return 0.0
    errors = []
    for probability, outcome in zip(probabilities, outcomes):
        p = clamp(probability)
        if outcome not in (0, 1, False, True):
            raise ValueError("outcomes must be binary")
        errors.append((p - int(outcome)) ** 2)
    return sum(errors) / len(errors)
