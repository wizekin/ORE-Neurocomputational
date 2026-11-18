"""End-to-end biologically inspired 16-stage prototype/exemplar learning processor."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import time
import uuid
from typing import Any, Mapping, Sequence

from .biology import BiologicalPrototypeExemplarSystem
from .contracts import (
    CategoryState, EpisodeInput, ExemplarTrace, LearningRequest, PrototypeExemplarConfig,
    PrototypeExemplarResult,
)
from .functional import (
    category_posterior, canonical_features, canonical_relations, clamp,
    estimate_feature_diagnosticity, exemplar_similarity, mahalanobis_outlier_score,
    model_validation_metrics, prototype_features, prototype_similarity,
    update_feature_attention,
)
from .registry import OPERATION_REGISTRY


class PrototypeExemplarProcessor:
    """Learn and classify categories using independent prototype and exemplar pathways.

    The processor preserves distinct episode traces and separately updates statistical category
    summaries. Its modular network is biologically inspired rather than an anatomical simulation;
    its equations are executable hypotheses whose parameters should be fitted to behavioral data.
    """

    def __init__(
        self,
        config: PrototypeExemplarConfig | None = None,
        database_path: str | Path | None = None,
        repository_root: str | Path | None = None,
    ) -> None:
        """Initialize category memory, distributed biological surrogates, and optional SQLite logging."""
        self.config = config or PrototypeExemplarConfig()
        self.biological_system = BiologicalPrototypeExemplarSystem(self.config)
        self.categories: dict[str, CategoryState] = {}
        self.feature_attention: dict[str, float] = {}
        self.process_count = 0
        self.database_path = Path(database_path).expanduser().resolve() if database_path else None
        self.repository_root = (
            Path(repository_root).expanduser().resolve()
            if repository_root else Path(__file__).resolve().parents[1]
        )

    def _state(self, category_id: str) -> CategoryState:
        """Return a persistent category state, creating an empty prototype/exemplar pair if needed."""
        return self.categories.setdefault(category_id, CategoryState(category_id=category_id))

    def _feature_weights(self, episode: EpisodeInput, attention: Mapping[str, float],
                         goals: Mapping[str, float], prior: Mapping[str, float]) -> dict[str, float]:
        """Combine task attention, goal relevance, learned diagnosticity, and observation reliability."""
        features = canonical_features(episode, self.config.feature_aliases)
        names = [name for name, value in features.items() if value is not None]
        gains = self.biological_system.control.select_feature_gains(names, attention, goals, prior)
        if not self.categories:
            return {name: max(0.05, gains.get(name, 1.0)) for name in names}
        diagnosticity: dict[str, float] = {}
        for category in self.categories:
            rows = estimate_feature_diagnosticity(self.categories, category)
            for name, value in rows.items():
                if value is not None:
                    diagnosticity[name] = max(diagnosticity.get(name, 0.0), float(value))
        weights: dict[str, float] = {}
        for name in names:
            raw_reliability = episode.feature_reliability.get(name, episode.reliability)
            discriminability = diagnosticity.get(name, 0.5)
            weights[name] = max(0.03, gains.get(name, 1.0)
                                * (0.55 + 0.45 * discriminability)
                                * max(0.05, raw_reliability))
        return weights

    def _update_prototype(self, category_id: str, episode: EpisodeInput,
                          feature_weights: Mapping[str, float]) -> dict[str, Any]:
        """Increment weighted numeric moments, categorical frequencies, and typed relation support."""
        state = self._state(category_id)
        prototype = state.prototype
        features = canonical_features(episode, self.config.feature_aliases)
        numeric_row: dict[str, float] = {}
        updated_numeric = updated_categorical = 0
        for feature, value in features.items():
            if value is None:
                continue
            observation_weight = (episode.reliability
                                  * episode.feature_reliability.get(feature, 1.0))
            if observation_weight <= 0.0:
                continue
            prototype.feature_observations[feature] = (
                prototype.feature_observations.get(feature, 0) + 1
            )
            prototype.feature_weights[feature] = float(feature_weights.get(feature, 0.5))
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric_value = float(value)
                old_weight = prototype.numeric_weight.get(feature, 0.0)
                new_weight = old_weight + observation_weight
                old_mean = prototype.numeric_mean.get(feature, numeric_value)
                delta = numeric_value - old_mean
                new_mean = old_mean + observation_weight / max(new_weight, 1e-12) * delta
                prototype.numeric_m2[feature] = (
                    prototype.numeric_m2.get(feature, 0.0)
                    + observation_weight * delta * (numeric_value - new_mean)
                )
                prototype.numeric_mean[feature] = new_mean
                prototype.numeric_weight[feature] = new_weight
                prototype.numeric_weight_sq[feature] = (
                    prototype.numeric_weight_sq.get(feature, 0.0) + observation_weight ** 2
                )
                numeric_row[feature] = numeric_value
                updated_numeric += 1
            else:
                token = f"{type(value).__name__}:{value}"
                counts = prototype.categorical_counts.setdefault(feature, {})
                counts[token] = counts.get(token, 0.0) + observation_weight
                updated_categorical += 1
        for relation in canonical_relations(episode, self.config.feature_aliases):
            relation_key = " :: ".join(relation)
            prototype.relation_counts[relation_key] = (
                prototype.relation_counts.get(relation_key, 0.0) + episode.reliability
            )
        if numeric_row:
            state.numeric_observations.append(numeric_row)
            if len(state.numeric_observations) > self.config.max_exemplars_per_category * 2:
                state.numeric_observations.pop(0)
        prototype.total_weight += episode.reliability
        prototype.update_count += 1
        prototype.supporting_episode_ids.append(episode.episode_id)
        state.prior_count += episode.reliability
        state.update_count += 1
        return {"numeric_features_updated": updated_numeric,
                "categorical_features_updated": updated_categorical,
                "prototype": prototype.to_dict()}

    def _store_exemplar(self, episode: EpisodeInput, category_id: str,
                        diagnosticity: float = 0.0, is_exception: bool = False) -> tuple[ExemplarTrace, bool]:
        """Create and store an independent episode trace while enforcing unique exemplar identity."""
        state = self._state(category_id)
        existing = state.exemplars.get(episode.episode_id)
        if existing is not None:
            same_content = (
                existing.features == canonical_features(episode, self.config.feature_aliases)
                and existing.context == dict(episode.context)
                and existing.relations == canonical_relations(episode, self.config.feature_aliases)
                and existing.reliability == episode.reliability
                and existing.category_id == category_id
            )
            if not same_content:
                raise ValueError(f"episode_id {episode.episode_id!r} already exists with different content")
            return existing, False
        trace = self.biological_system.hippocampus.bind(
            episode, category_id, self.config.feature_aliases,
            diagnosticity=diagnosticity, is_exception=is_exception,
        )
        state.exemplars[episode.episode_id] = trace
        self.biological_system.hippocampus.store(trace)
        self._prune_redundant_exemplars(category_id)
        return trace, True

    def _prune_redundant_exemplars(self, category_id: str) -> int:
        """Remove only low-value redundant traces when over capacity, protecting exceptions."""
        state = self._state(category_id)
        excess = len(state.exemplars) - self.config.max_exemplars_per_category
        if excess <= 0:
            return 0
        candidates = []
        now = time.time()
        for trace in state.exemplars.values():
            if trace.is_exception:
                continue
            uniqueness = 1.0 / (1.0 + max(0, len(trace.features)))
            value = (0.45 * trace.base_strength + 0.35 * trace.diagnosticity
                     + 0.20 * uniqueness)
            age_penalty = min(0.25, max(0.0, now - trace.created_at) / 100000.0)
            candidates.append((value - age_penalty, trace.episode_id))
        removed = 0
        for _, episode_id in sorted(candidates):
            if removed >= excess:
                break
            state.exemplars.pop(episode_id, None)
            self.biological_system.hippocampus.memories.get(category_id, {}).pop(episode_id, None)
            removed += 1
        return removed

    def _operation(self, trace: list[dict[str, Any]], index: int,
                   functional_output: Mapping[str, Any], biological_output: Mapping[str, Any],
                   status: str = "completed") -> None:
        """Append a structured, source-ordered algorithm operation with its math and biological rationale."""
        spec = OPERATION_REGISTRY[index]
        trace.append({
            "step": index + 1,
            "operation_id": spec.operation_id,
            "operation": spec.name,
            "status": status,
            "explanation": spec.purpose,
            "functional_output": dict(functional_output),
            "biological_output": dict(biological_output),
            "evidence_level": spec.evidence_level,
            "mathematical_form": spec.mathematical_form,
        })

    def _score_categories(
        self, episode: EpisodeInput, weights: Mapping[str, float], task_context: str,
        known_category: str | None, now: float,
    ) -> dict[str, Any]:
        """Calculate independent prototype/exemplar evidence, exception signals, and hybrid scores."""
        outputs: dict[str, Any] = {}
        for category_id, state in self.categories.items():
            prototype = prototype_similarity(episode, state, weights, self.config)
            retrieved = self.biological_system.hippocampus.retrieve(episode, state, weights, now)
            exemplars = exemplar_similarity(retrieved)
            outlier = mahalanobis_outlier_score(
                episode, state, self.config.feature_aliases
            )
            exemplar_support = (exemplars["best_similarity"]
                                >= self.config.exception_similarity_threshold)
            labelled_support = known_category == category_id
            exception_score = 0.0
            is_exception = False
            if outlier >= self.config.outlier_threshold:
                exception_score = clamp((outlier - self.config.outlier_threshold)
                                        / self.config.outlier_threshold)
                if exemplar_support or labelled_support:
                    is_exception = True
                else:
                    exception_score *= 0.25
            typicality = float(prototype["similarity"])
            alpha = self._blend_weight(
                float(prototype["reliability"]), float(exemplars["reliability"]),
                typicality, exception_score, task_context,
            )
            hybrid = alpha * typicality + (1.0 - alpha) * float(exemplars["similarity"])
            outputs[category_id] = {
                "prototype": prototype,
                "exemplars": exemplars,
                "retrieval": retrieved,
                "outlier_score": outlier,
                "exception_score": exception_score,
                "is_exception": is_exception,
                "prototype_weight": alpha,
                "hybrid_score": clamp(hybrid),
            }
        return outputs

    def _blend_weight(self, prototype_confidence: float, exemplar_confidence: float,
                      typicality: float, exception_score: float, task_context: str) -> float:
        """Compute a dynamic prototype/exemplar mixing coefficient from the current evidence state."""
        from .functional import blend_weight
        return blend_weight(prototype_confidence, exemplar_confidence, typicality,
                            exception_score, task_context)

    def _run_episode(
        self, episode: EpisodeInput, attention: Mapping[str, float], goals: Mapping[str, float],
        prior: Mapping[str, float], task_context: str, known_category: str | None = None,
        feedback_category: str | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Execute all sixteen source operations for one learning or held-out categorization event."""
        operations: list[dict[str, Any]] = []
        now = episode.timestamp if episode.timestamp is not None else time.time()
        # 01: detection and episode coherence.
        observed_count = sum(value is not None for value in episode.features.values())
        coherence = clamp((observed_count / max(1, len(episode.features))) * episode.reliability)
        self._operation(operations, 0, {
            "episode_id": episode.episode_id, "coherence": coherence,
            "candidate_detected": coherence > 0.0, "timestamp": now,
            "context_keys": sorted(episode.context),
        }, {"sensory_registration": "candidate event registered",
            "attention_gate": "coherence and task relevance gate further processing"})

        # 02: modality-specific and association feature encoding.
        encoded = self.biological_system.sensory.encode(episode, self.config.feature_aliases)
        features = encoded["features"]
        self._operation(operations, 1, encoded, {
            "sensory_association_network": "perceptual and modality-specific feature evidence",
            "uncertainty_preserved": True,
        })

        # 03: task-dependent feature gain.
        weights = self._feature_weights(episode, attention, goals, prior)
        self.feature_attention.update(weights)
        self._operation(operations, 2, {
            "feature_weights": weights, "sum_of_normalized_attention": sum(
                self.biological_system.control.last_feature_gains.values()),
        }, {"prefrontal_parietal_control": self.biological_system.control.snapshot()})

        # 04: create an episode-specific binding candidate without assuming a category label.
        pending_category = known_category or self.config.unknown_category
        pending_trace = self.biological_system.hippocampus.bind(
            episode, pending_category, self.config.feature_aliases
        )
        self._operation(operations, 3, {
            "episode_id": pending_trace.episode_id,
            "bound_feature_count": len(pending_trace.features),
            "bound_relation_count": len(pending_trace.relations),
            "context_binding": bool(pending_trace.context),
        }, {"hippocampal_episode_network": "conjunctive episode binding and pattern separation"})

        # 05: retrieve candidate categories while retaining an unresolved option.
        candidates = list(self.categories)
        if known_category and known_category not in candidates:
            self._state(known_category)
            candidates.append(known_category)
        candidate_scores = {}
        for category_id in candidates:
            state = self.categories[category_id]
            proto = prototype_similarity(episode, state, weights, self.config)
            candidate_scores[category_id] = proto["similarity"] * max(
                1e-6, state.prior_count
            )
        provisional = (max(candidate_scores, key=candidate_scores.get)
                       if candidate_scores else self.config.unknown_category)
        self._operation(operations, 4, {
            "candidate_categories": candidates,
            "provisional_candidate": provisional,
            "provisional_similarity": candidate_scores.get(provisional, 0.0),
            "unknown_option_retained": True,
        }, {"medial_temporal_retrieval": "candidate memory cues",
            "prefrontal_control": "uncertain candidates remain unresolved"})

        # Preserve pre-update atypicality for supervised learning examples.
        pre_update_outlier = (mahalanobis_outlier_score(
            episode, self.categories[known_category], self.config.feature_aliases
        ) if known_category in self.categories else 0.0)

        # 06: episodic exemplar update for labelled training examples.
        trace = None
        exemplar_added = False
        if known_category is not None:
            diagnostic_values = estimate_feature_diagnosticity(
                self.categories, known_category
            ).values()
            diagnostic = max((value for value in diagnostic_values if value is not None),
                             default=0.0)
            trace, exemplar_added = self._store_exemplar(
                episode, known_category, diagnosticity=float(diagnostic),
                is_exception=pre_update_outlier >= self.config.outlier_threshold,
            )
        self._operation(operations, 5, {
            "category_id": known_category, "exemplar_added": exemplar_added,
            "exemplar_count": (len(self.categories[known_category].exemplars)
                               if known_category else 0),
            "episode_identity_preserved": True,
        }, {"hippocampal_episode_network": self.biological_system.hippocampus.snapshot()})

        # 07: distributed prototype statistics update.
        prototype_update = None
        if known_category is not None and exemplar_added:
            prototype_update = self._update_prototype(known_category, episode, weights)
        self._operation(operations, 6, {
            "category_id": known_category, "prototype_updated": prototype_update is not None,
            "update": prototype_update,
        }, {"distributed_cortical_network": "gradual regularity accumulation from repeated exposures"})

        # 08: estimate diagnosticity after incorporating the labelled training event.
        diagnosticity = {
            category: estimate_feature_diagnosticity(self.categories, category)
            for category in self.categories
        }
        self._operation(operations, 7, {
            "category_diagnosticity": diagnosticity,
            "typicality_is_not_necessity": True,
            "exceptions_are_retained": True,
        }, {"distributed_association_cortex": "contrastive category regularities",
            "prefrontal_control": "task-dependent feature relevance"})

        # 09-12: compare both pathways, integrate priors, and select a flexible blend.
        scores = self._score_categories(episode, weights, task_context, known_category, now)
        proto_scores = {cat: row["prototype"]["similarity"] for cat, row in scores.items()}
        exemplar_scores = {cat: row["exemplars"]["similarity"] for cat, row in scores.items()}
        hybrid_scores = {cat: row["hybrid_score"] for cat, row in scores.items()}
        priors = {cat: state.prior_count for cat, state in self.categories.items()}
        proto_probabilities = category_posterior(
            proto_scores, priors, self.config.choice_temperature, self.config.beta_prior
        ) if proto_scores else {}
        exemplar_probabilities = category_posterior(
            exemplar_scores, priors, self.config.choice_temperature, self.config.beta_prior
        ) if exemplar_scores else {}
        self._operation(operations, 8, {
            "prototype_scores": proto_scores,
            "category_distances": {cat: row["prototype"]["distance"] for cat, row in scores.items()},
        }, {"ventral_temporal_and_distributed_cortical": "prototype-level similarity comparison"})
        self._operation(operations, 9, {
            "exemplar_scores": exemplar_scores,
            "retrieved_episode_ids": {cat: row["exemplars"]["retrieved_ids"]
                                       for cat, row in scores.items()},
            "strongest_matches": {cat: row["exemplars"].get("best_episode_id")
                                  for cat, row in scores.items()},
        }, {"hippocampal_and_medial_temporal": "partial-cue retrieval of separate memories"})
        self._operation(operations, 10, {
            "prototype_only_probabilities": proto_probabilities,
            "exemplar_only_probabilities": exemplar_probabilities,
            "category_priors": priors,
        }, {"prefrontal_decision_network": "combine evidence and category expectation"})
        hybrid_probabilities = category_posterior(
            hybrid_scores, priors, self.config.choice_temperature, self.config.beta_prior
        ) if hybrid_scores else {}
        self.biological_system.control.register_strategy(
            sum(row["prototype_weight"] for row in scores.values()) / max(1, len(scores))
        )
        self._operation(operations, 11, {
            "prototype_weights": {cat: row["prototype_weight"] for cat, row in scores.items()},
            "hybrid_scores": hybrid_scores,
            "hybrid_probabilities": hybrid_probabilities,
        }, {"distributed_strategy_control": "prototype/exemplar influence changes with reliability, typicality and task"})

        # 13: mark exceptions without interpreting outlier status as non-membership.
        exceptions = {
            cat: {"outlier_score": row["outlier_score"], "exception_score": row["exception_score"],
                  "is_exception": row["is_exception"]}
            for cat, row in scores.items()
        }
        if trace is not None and known_category in scores and scores[known_category]["is_exception"]:
            trace.is_exception = True
        self._operation(operations, 12, {
            "category_exception_diagnostics": exceptions,
            "training_item_pre_update_outlier": pre_update_outlier,
            "unusual_member_not_automatically_rejected": True,
        }, {"hippocampal_episode_network": "retain distinctive conjunctions as explicit traces"})

        # 14: use only explicitly supplied feedback after the prediction has already been calculated.
        best_predicted = (max(hybrid_probabilities, key=hybrid_probabilities.get)
                          if hybrid_probabilities else self.config.unknown_category)
        confidence = float(hybrid_probabilities.get(best_predicted, 0.0))
        prediction_error: float | None = None
        feedback_gate = 0.0
        feedback_update: dict[str, Any] | None = None
        if feedback_category is not None:
            prediction_error = 0.0 if best_predicted == feedback_category else 1.0
            feedback_gate = self.biological_system.feedback.compute_gate(
                prediction_error, episode.reliability, feedback_available=True
            )
            self._state(feedback_category)
            revised_weights = update_feature_attention(
                self.feature_attention, episode, feedback_category, self.categories,
                self.config.feedback_learning_rate * feedback_gate,
                self.config.feature_aliases,
            )
            self.feature_attention.update(revised_weights)
            feedback_trace, added = self._store_exemplar(
                episode, feedback_category, diagnosticity=0.0,
                is_exception=(feedback_category == best_predicted and bool(
                    scores.get(feedback_category, {}).get("is_exception", False)
                )),
            )
            feedback_update = ({"prototype_updated": False} if not added else
                               self._update_prototype(feedback_category, episode, revised_weights))
            feedback_update["exemplar_added"] = added
            feedback_update["trace_id"] = feedback_trace.episode_id
        else:
            self.biological_system.feedback.compute_gate(0.0, episode.reliability,
                                                         feedback_available=False)
        self._operation(operations, 13, {
            "feedback_available": feedback_category is not None,
            "feedback_category": feedback_category,
            "predicted_category_before_feedback": best_predicted,
            "prediction_error": prediction_error,
            "learning_gate": feedback_gate,
            "feedback_update": feedback_update,
            "unlabelled_surprise_is_not_a_teaching_signal": True,
        }, {"neuromodulatory_feedback_gate": self.biological_system.feedback.snapshot()})

        # 15: replay a subset of retrieved traces and update distributed cortical population activity.
        replayed = 0
        cortical_update = None
        consolidation_category = feedback_category or known_category
        should_consolidate = (
            (known_category is not None and exemplar_added)
            or (feedback_category is not None and bool(feedback_update and feedback_update.get("exemplar_added")))
        )
        if consolidation_category is not None:
            selected_ids = scores.get(consolidation_category, {}).get("exemplars", {}).get(
                "retrieved_ids", []
            )
            replayed = self.biological_system.hippocampus.rehearse(
                consolidation_category, selected_ids, now
            )
            if should_consolidate:
                cortical_update = self.biological_system.cortex.learn(
                    consolidation_category, features, episode.reliability
                )
        self._operation(operations, 14, {
            "consolidation_category": consolidation_category,
            "replayed_exemplars": replayed,
            "prototype_update_count": (self.categories[consolidation_category].prototype.update_count
                                        if consolidation_category in self.categories else 0),
            "cortical_update": cortical_update,
            "retained_exemplar_count": (len(self.categories[consolidation_category].exemplars)
                                         if consolidation_category in self.categories else 0),
        }, {"hippocampal_cortical_interaction": "replay supports regularity learning while traces remain available"})

        # 16: final decision, uncertainty, and pathway comparison outputs.
        if hybrid_probabilities:
            predicted = max(hybrid_probabilities, key=hybrid_probabilities.get)
            confidence = float(hybrid_probabilities[predicted])
            best_evidence = float(hybrid_scores.get(predicted, 0.0))
            if confidence < self.config.acceptance_threshold or best_evidence < 0.10:
                predicted = self.config.unknown_category
        else:
            predicted = self.config.unknown_category
            confidence = 0.0
        prediction = {
            "episode_id": episode.episode_id,
            "predicted_category": predicted,
            "candidate_category": best_predicted,
            "confidence": confidence,
            "prototype_only_probabilities": proto_probabilities,
            "exemplar_only_probabilities": exemplar_probabilities,
            "hybrid_probabilities": hybrid_probabilities,
            "category_evidence": scores,
            "decision_status": "accepted" if predicted != self.config.unknown_category else "unresolved",
            "task_context": task_context,
        }
        self._operation(operations, 15, {
            "predicted_category": predicted, "confidence": confidence,
            "decision_status": prediction["decision_status"],
            "prototype_only_metrics_available_after_feedback": False,
            "held_out_evaluation_is_separate_from_training": True,
        }, {"distributed_generalization": "category-level summary and episodic evidence jointly support decisions"})
        for operation_row in operations:
            operation_row["episode_id"] = episode.episode_id
        return operations, prediction

    def process(self, request: LearningRequest) -> PrototypeExemplarResult:
        """Train from labelled category examples, classify held-out cases, and record all sixteen stages."""
        started = time.perf_counter()
        all_operations: list[dict[str, Any]] = []
        predictions: list[dict[str, Any]] = []
        for category_id in sorted(request.training_examples):
            for episode in request.training_examples[category_id]:
                rows, _ = self._run_episode(
                    episode, request.attention_weights, request.goal_weights,
                    request.prior_weights, request.task_context, known_category=category_id,
                )
                all_operations.extend(rows)
        for episode in request.prediction_episodes:
            feedback_category = request.feedback_labels.get(episode.episode_id)
            rows, prediction = self._run_episode(
                episode, request.attention_weights, request.goal_weights,
                request.prior_weights, request.task_context,
                feedback_category=feedback_category,
            )
            all_operations.extend(rows)
            predictions.append(prediction)
        labelled_ids = set(request.feedback_labels)
        validation_predictions = [row for row in predictions if row["episode_id"] in labelled_ids]
        validation_labels = [request.feedback_labels[row["episode_id"]]
                             for row in validation_predictions]
        validation = {
            "evaluation_uses_post_prediction_labels_only": True,
            "hybrid": model_validation_metrics(validation_labels, validation_predictions, "hybrid"),
            "prototype_only": model_validation_metrics(validation_labels, validation_predictions,
                                                        "prototype_only"),
            "exemplar_only": model_validation_metrics(validation_labels, validation_predictions,
                                                        "exemplar_only"),
            "model_comparison_note": (
                "Compare held-out metrics; a hybrid model is not assumed to outperform simpler models."
            ),
        }
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.process_count += 1
        logger = self._log_sqlite(all_operations, elapsed_ms)
        return PrototypeExemplarResult(
            operation_trace=all_operations,
            categories={name: state.to_dict() for name, state in sorted(self.categories.items())},
            predictions=predictions,
            validation=validation,
            biological=self.biological_system.snapshot(),
            logger=logger,
        )

    def fit(self, training_examples: Mapping[str, Sequence[EpisodeInput]],
            attention_weights: Mapping[str, float] | None = None,
            goal_weights: Mapping[str, float] | None = None,
            prior_weights: Mapping[str, float] | None = None) -> dict[str, Any]:
        """Incrementally learn category prototypes and preserve all distinct training exemplars."""
        request = LearningRequest(
            training_examples=training_examples,
            attention_weights=attention_weights or {},
            goal_weights=goal_weights or {},
            prior_weights=prior_weights or {},
        )
        result = self.process(request)
        return {"categories": result.categories, "operation_trace": result.operation_trace,
                "biological": result.biological, "logger": result.logger}

    def classify(self, episode: EpisodeInput, task_context: str = "general_category_learning",
                 attention_weights: Mapping[str, float] | None = None,
                 goal_weights: Mapping[str, float] | None = None,
                 prior_weights: Mapping[str, float] | None = None) -> dict[str, Any]:
        """Classify one new case without incorporating its identity or inferred label into memory."""
        if not self.categories:
            raise RuntimeError("fit at least one category before classification")
        started = time.perf_counter()
        rows, prediction = self._run_episode(
            episode, attention_weights or {}, goal_weights or {}, prior_weights or {},
            task_context,
        )
        self.process_count += 1
        logger = self._log_sqlite(rows, (time.perf_counter() - started) * 1000.0)
        prediction["operation_trace"] = rows
        prediction["logger"] = logger
        return prediction

    def _log_sqlite(self, rows: list[dict[str, Any]], elapsed_ms: float) -> dict[str, Any]:
        """Persist each source-ordered operation and its docstring annotation in the ORE SQLite logger."""
        if self.database_path is None:
            return {"enabled": False, "reason": "database_path_not_configured"}
        from ore_logger.recorder import SQLiteRunRecorder
        run_id = uuid.uuid4().hex
        started_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        recorder = SQLiteRunRecorder(
            database_path=self.database_path,
            repository_root=self.repository_root,
            run_id=run_id,
            component="ore_prototype_exemplar",
            callable_name="PrototypeExemplarProcessor.process",
            started_at_utc=started_at,
        )
        for row in rows:
            recorder.log_operation({
                "step": row["step"], "operation": row["operation"],
                "status": row["status"], "explanation": row["explanation"],
                "functional_output": row["functional_output"],
                "biological_output": row["biological_output"],
                "evidence_level": row["evidence_level"],
                "mathematical_form": row["mathematical_form"],
            })
        return {"enabled": True, **recorder.finish("completed", elapsed_ms,
                                                  output_type="PrototypeExemplarResult")}

    def get_category(self, category_id: str) -> dict[str, Any] | None:
        """Return one category's summary and separate exemplar identifiers, if it has been learned."""
        state = self.categories.get(category_id)
        return state.to_dict() if state else None

    def snapshot(self) -> dict[str, Any]:
        """Export the learned category index, current attention profile, and biological subsystem state."""
        return {"categories": {name: state.to_dict() for name, state in self.categories.items()},
                "feature_attention": dict(self.feature_attention),
                "biological": self.biological_system.snapshot(),
                "process_count": self.process_count}
