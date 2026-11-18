"""End-to-end 16-stage processor for commonality detection and revisable concept formation."""
from __future__ import annotations

from datetime import datetime, timezone
import math
from pathlib import Path
import time
import uuid
from typing import Any, Mapping, Sequence

from .biology import BiologicalCommonalityNetwork
from .contracts import (
    ConceptCandidate,
    ConceptLearningConfig,
    ConceptLearningRequest,
    ConceptLearningResult,
    EpisodeInput,
)
from .functional import (
    build_associative_index,
    build_concept_candidate,
    canonical_features,
    canonical_relations,
    clamp,
    compress_shared_structure,
    estimate_contrastive_diagnosticity,
    estimate_feature_statistics,
    estimate_prediction_mismatch,
    estimate_relation_strengths,
    predict_episode,
)
from .registry import OPERATION_REGISTRY


class CommonalityDetectionConceptFormationProcessor:
    """Learn a revisable concept by comparing distinct experiences across 16 source-ordered stages.

    The processor stores episode-specific records separately from the provisional concept and
    retains all supporting examples for future revision. It models shared features and relations,
    not prototype/exemplar theories, semantic hubs, schemas, or analogical transfer; those belong
    to later source points. Equations and rate populations are computational surrogates rather
    than claims about literal neuronal arithmetic.
    """

    def __init__(
        self,
        config: ConceptLearningConfig | None = None,
        database_path: str | Path | None = None,
        repository_root: str | Path | None = None,
    ) -> None:
        """Initialize episode stores, concept state, associative indices, and optional SQLite logging."""
        self.config = config or ConceptLearningConfig()
        self.network = BiologicalCommonalityNetwork(self.config)
        self.target_examples: dict[str, dict[str, EpisodeInput]] = {}
        self.comparison_examples: dict[str, dict[str, EpisodeInput]] = {}
        self.concepts: dict[str, ConceptCandidate] = {}
        self.associative_indices: dict[str, dict[str, Any]] = {}
        self.database_path = Path(database_path).expanduser().resolve() if database_path else None
        self.repository_root = (
            Path(repository_root).expanduser().resolve()
            if repository_root else Path(__file__).resolve().parents[1]
        )
        self.process_count = 0

    def _store_for(self, concept_id: str, category: bool) -> dict[str, EpisodeInput]:
        """Return the persistent positive or comparison episode store for one concept identifier."""
        stores = self.target_examples if category else self.comparison_examples
        return stores.setdefault(concept_id, {})

    def _known_training_episode(
        self, concept_id: str, episode_id: str
    ) -> tuple[EpisodeInput | None, bool | None]:
        """Find an episode in either training group and return its established category role."""
        target = self.target_examples.get(concept_id, {})
        comparison = self.comparison_examples.get(concept_id, {})
        if episode_id in target:
            return target[episode_id], True
        if episode_id in comparison:
            return comparison[episode_id], False
        return None, None

    def _add_training_episodes(
        self,
        concept_id: str,
        episodes: Sequence[EpisodeInput],
        category: bool,
    ) -> list[EpisodeInput]:
        """Add new labelled evidence once and reject conflicting reuse of an episode identifier."""
        destination = self._store_for(concept_id, category)
        added: list[EpisodeInput] = []
        for episode in episodes:
            existing, existing_category = self._known_training_episode(
                concept_id, episode.episode_id
            )
            if existing is not None:
                if existing.to_dict() != episode.to_dict():
                    raise ValueError(
                        f"episode_id {episode.episode_id!r} already exists with different content"
                    )
                if existing_category is not category:
                    raise ValueError(
                        f"episode_id {episode.episode_id!r} cannot be both target and comparison evidence"
                    )
                continue
            if len(self.target_examples.get(concept_id, {})) + len(
                self.comparison_examples.get(concept_id, {})
            ) >= self.config.max_episodes_per_concept:
                raise ValueError("configured episode capacity has been reached")
            destination[episode.episode_id] = episode
            added.append(episode)
        return added

    def _build_candidate(
        self,
        request: ConceptLearningRequest,
        target: Sequence[EpisodeInput],
        comparison: Sequence[EpisodeInput],
        previous: ConceptCandidate | None,
    ) -> ConceptCandidate:
        """Build the concept from persistent examples using biologically motivated attention gains."""
        feature_names = sorted({
            feature
            for episode in tuple(target) + tuple(comparison)
            for feature in canonical_features(episode, self.config.feature_aliases)
        })
        control_gains = self.network.attention.select_feature_gains(
            feature_names,
            request.attention_weights,
            request.goal_weights,
            request.prior_weights,
        )
        return build_concept_candidate(
            concept_id=request.concept_id,
            target_episodes=target,
            comparison_episodes=comparison,
            aliases=self.config.feature_aliases,
            attention=control_gains,
            goals=control_gains,
            prior=control_gains,
            beta_prior=self.config.beta_prior,
            variance_floor=self.config.variance_floor,
            minimum_shared_weight=self.config.minimum_shared_weight,
            previous=previous,
        )

    def _information_gain(
        self,
        target_statistics: Mapping[str, Mapping[str, float | int]],
        comparison_statistics: Mapping[str, Mapping[str, float | int]],
        target_episodes: Sequence[EpisodeInput],
        comparison_episodes: Sequence[EpisodeInput],
    ) -> dict[str, float | None]:
        """Estimate feature information gain from target-versus-background uncertainty reduction."""
        target_weight = sum(item.reliability for item in target_episodes)
        comparison_weight = sum(item.reliability for item in comparison_episodes)
        total = target_weight + comparison_weight
        if total <= 0.0 or not comparison_episodes:
            return {feature: None for feature in target_statistics}
        prior = clamp(target_weight / total, 1e-9, 1.0 - 1e-9)

        def entropy(probability: float) -> float:
            """Compute binary entropy in nats with safe probability boundaries."""
            p = clamp(probability, 1e-9, 1.0 - 1e-9)
            return -p * math.log(p) - (1.0 - p) * math.log(1.0 - p)

        base_entropy = entropy(prior)
        output: dict[str, float | None] = {}
        for feature in sorted(set(target_statistics) | set(comparison_statistics)):
            target_row = target_statistics.get(feature)
            comparison_row = comparison_statistics.get(feature)
            if target_row is None or comparison_row is None:
                output[feature] = None
                continue
            p_target = clamp(float(target_row["prevalence"]), 1e-9, 1.0 - 1e-9)
            p_other = clamp(float(comparison_row["prevalence"]), 1e-9, 1.0 - 1e-9)
            p_present = prior * p_target + (1.0 - prior) * p_other
            posterior_present = prior * p_target / max(p_present, 1e-12)
            p_absent = 1.0 - p_present
            posterior_absent = (
                prior * (1.0 - p_target) / max(p_absent, 1e-12)
            )
            conditional = (
                p_present * entropy(posterior_present)
                + p_absent * entropy(posterior_absent)
            )
            output[feature] = max(0.0, base_entropy - conditional)
        return output

    def _make_operation(
        self,
        index: int,
        functional: dict[str, Any],
        biological: dict[str, Any] | None = None,
        status: str = "completed",
    ) -> dict[str, Any]:
        """Create an inspectable operation result using the canonical source-order registry."""
        spec = OPERATION_REGISTRY[index]
        return {
            "step": index + 1,
            "operation_id": spec.operation_id,
            "operation": spec.name,
            "status": status,
            "explanation": spec.purpose,
            "evidence_level": spec.evidence_level,
            "mathematical_form": spec.mathematical_form,
            "functional_output": functional,
            "biological_output": biological or {},
        }

    def _learn_cortex(
        self,
        target: Sequence[EpisodeInput],
        candidate: ConceptCandidate,
        modulatory_gate: float = 1.0,
    ) -> dict[str, Any]:
        """Apply reliability-weighted cortical learning with optional feedback modulation."""
        return self.network.learn_cortical_commonality(
            target,
            self.config.feature_aliases,
            candidate.feature_prevalence,
            modulatory_gate=modulatory_gate,
        )

    def _log_sqlite(
        self, rows: list[dict[str, Any]], elapsed_ms: float
    ) -> dict[str, Any]:
        """Persist all 16 operation annotations and outcomes in the reusable ORE SQLite logger."""
        if self.database_path is None:
            return {"enabled": False, "reason": "database_path_not_configured"}
        from ore_logger.recorder import SQLiteRunRecorder

        run_id = uuid.uuid4().hex
        started_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        recorder = SQLiteRunRecorder(
            database_path=self.database_path,
            repository_root=self.repository_root,
            run_id=run_id,
            component="ore_commonality_concepts",
            callable_name="CommonalityDetectionConceptFormationProcessor.process",
            started_at_utc=started_at,
        )
        for row in rows:
            recorder.log_operation({
                "step": row["step"],
                "operation": row["operation"],
                "status": row["status"],
                "explanation": row["explanation"],
                "functional_output": row["functional_output"],
                "biological_output": row["biological_output"],
                "evidence_level": row["evidence_level"],
                "mathematical_form": row["mathematical_form"],
            })
        return {
            "enabled": True,
            **recorder.finish("completed", elapsed_ms, output_type="ConceptLearningResult"),
        }

    def process(self, request: ConceptLearningRequest) -> ConceptLearningResult:
        """Process episodes through all 16 Point 06 operations and return a revisable concept."""
        started = time.perf_counter()
        concept_id = request.concept_id
        previous = self.concepts.get(concept_id)

        # Add initial training evidence exactly once. Prediction examples remain held out here.
        new_targets = self._add_training_episodes(
            concept_id, request.target_episodes, category=True
        )
        new_comparisons = self._add_training_episodes(
            concept_id, request.comparison_episodes, category=False
        )
        target = tuple(self.target_examples.get(concept_id, {}).values())
        comparison = tuple(self.comparison_examples.get(concept_id, {}).values())
        if len(target) < 2:
            raise ValueError("commonality detection requires at least two target episodes")
        self.network.register_episodes(target + comparison)
        self.process_count += 1
        rows: list[dict[str, Any]] = []

        # CC01: keep each encounter as a distinct, retrievable episode.
        rows.append(self._make_operation(0, {
            "concept_id": concept_id,
            "new_target_episode_ids": [item.episode_id for item in new_targets],
            "new_comparison_episode_ids": [item.episode_id for item in new_comparisons],
            "target_episode_count": len(target),
            "comparison_episode_count": len(comparison),
            "episode_identity_preserved": True,
        }, {
            "candidate_systems": ["hippocampal_formation", "medial_temporal_structures"],
            "episode_specific_records_are_not_collapsed": True,
        }))

        # CC02: canonicalize modality-specific features and preserve the missing-data mask.
        aliases = self.config.feature_aliases
        aligned_target = {
            episode.episode_id: canonical_features(episode, aliases)
            for episode in target
        }
        feature_names = sorted({
            feature
            for episode in target + comparison
            for feature in canonical_features(episode, aliases)
        })
        modality_counts: dict[str, int] = {}
        unobserved_counts: dict[str, int] = {}
        for episode in target:
            for raw_feature, modality in episode.feature_modalities.items():
                canonical = aliases.get(raw_feature, raw_feature)
                modality_counts[modality] = modality_counts.get(modality, 0) + 1
                if canonical_features(episode, aliases).get(canonical) is None:
                    unobserved_counts[canonical] = unobserved_counts.get(canonical, 0) + 1
        rows.append(self._make_operation(1, {
            "canonical_feature_names": feature_names,
            "episode_feature_maps": aligned_target,
            "feature_modality_observation_counts": modality_counts,
            "unknown_feature_counts": unobserved_counts,
            "zero_is_observed_absence": True,
        }, {
            "systems": ["visual", "auditory", "somatosensory", "motor", "association_cortices"],
            "modality_specific_encoders_are_external_inputs": True,
        }))

        # CC03: select feature gains from task attention, goals, and prior feature salience.
        control_gains = self.network.attention.select_feature_gains(
            feature_names,
            request.attention_weights,
            request.goal_weights,
            request.prior_weights,
        )
        rows.append(self._make_operation(2, {
            "feature_gains": control_gains,
            "competition_temperature": self.network.attention.temperature,
        }, {
            "control_surrogate": "prefrontal_parietal_gain_modulation",
            "gain_values_are_model_parameters": True,
        }))

        # CC04: retrieve related prior episodes from the hippocampal episode index.
        retrieval_sources = new_targets or list(target[-1:])
        retrieval_rows = [
            {
                "query_episode_id": episode.episode_id,
                "related_episodes": self.network.hippocampus.retrieve(episode, aliases),
            }
            for episode in retrieval_sources
        ]
        rows.append(self._make_operation(3, {
            "retrieval_results": retrieval_rows,
            "retrieval_source_count": len(retrieval_rows),
        }, {
            "network": "hippocampal_associative_episode_index",
            "pattern_completion_is_approximated_by_partial_feature_cue_matching": True,
        }, "completed" if retrieval_rows else "completed_no_new_query_episode"))

        # CC05: align aliases and relation endpoints without deleting meaningful differences.
        aligned_relations = {
            episode.episode_id: list(canonical_relations(episode, aliases))
            for episode in target
        }
        rows.append(self._make_operation(4, {
            "canonical_features": aligned_target,
            "canonical_relations": aligned_relations,
            "alignment_policy": "alias normalization; graded feature values retained",
        }, {
            "context_and_viewpoint_variation_are_not_automatically_erased": True,
            "feature_aliases_are_user_supplied": True,
        }))

        # CC06: estimate prevalence using only observed values, weighted by reliability.
        target_total_weight = sum(item.reliability for item in target)
        target_statistics = estimate_feature_statistics(
            target, aliases, total_reference_weight=target_total_weight
        )
        rows.append(self._make_operation(5, {
            "feature_statistics": target_statistics,
            "prevalence_definition": "reliability-weighted average over observed episodes only",
            "unknown_values_excluded_from_denominator": True,
        }, {
            "mechanism": "repeated_cross_episode_coactivation_and_distributed_plasticity",
            "frequency_is_not_equated_with_category_meaning": True,
        }))

        # CC07: estimate feature conjunctions and typed relations.
        from .functional import estimate_pairwise_associations
        associations = estimate_pairwise_associations(
            target, aliases, beta_prior=self.config.beta_prior
        )
        relation_rows = estimate_relation_strengths(target, aliases)
        rows.append(self._make_operation(6, {
            "pairwise_cooccurrence_pmi": associations,
            "typed_relation_support": relation_rows,
            "relation_edge_count": len(relation_rows),
        }, {
            "systems": ["hippocampal_relational_binding", "distributed_association_cortex"],
            "pmi_and_graph_are_analysis_tools": True,
        }))

        # CC08: compare target prevalence to alternative-category/background prevalence.
        contrastive = estimate_contrastive_diagnosticity(
            target, comparison, aliases, self.config.beta_prior
        )
        comparison_statistics = estimate_feature_statistics(
            comparison, aliases,
            total_reference_weight=sum(item.reliability for item in comparison),
        )
        information_gain = self._information_gain(
            target_statistics, comparison_statistics, target, comparison
        )
        rows.append(self._make_operation(7, {
            "contrastive_diagnosticity": contrastive,
            "information_gain_estimate": information_gain,
            "comparison_episode_count": len(comparison),
            "diagnosticity_unknown_when_no_comparison_evidence": not bool(comparison),
        }, {
            "systems": ["prefrontal_evaluation", "contrastive_learning_networks"],
            "no_background_examples_means_no_confirmed_diagnosticity": True,
        }, "completed_with_contrastive_evidence" if comparison else "completed_without_contrastive_examples"))

        # CC09: combine within-category stability, coverage, and effective evidence size.
        provisional = self._build_candidate(request, target, comparison, previous)
        rows.append(self._make_operation(8, {
            "feature_variance": provisional.feature_variance,
            "feature_reliability": provisional.feature_reliability,
            "feature_uncertainty": provisional.uncertainty,
            "variable_feature_exceptions": provisional.exceptions,
            "effective_sample_size": provisional.feature_effective_sample_size,
        }, {
            "within_category_variation_preserved": True,
            "outlier_and_exception_handling_is_uncertainty_aware": True,
        }))

        # CC10: integrate recurring, relevant and diagnostic evidence into a provisional concept.
        rows.append(self._make_operation(9, {
            "concept_id": provisional.concept_id,
            "feature_prevalence": provisional.feature_prevalence,
            "diagnosticity": provisional.diagnosticity,
            "feature_weights": provisional.feature_weights,
            "shared_features": provisional.shared_features,
            "cooccurrence_pmi": provisional.cooccurrence_pmi,
            "relation_edges": provisional.relation_edges,
            "supporting_episode_ids": provisional.supporting_episode_ids,
        }, {
            "interacting_systems": ["hippocampus", "medial_prefrontal_networks", "cortical_association"],
            "representation_is_provisional_and_revisable": True,
        }))

        # CC11: compress weak incidental detail while retaining the backing episodes and exceptions.
        compression = compress_shared_structure(
            provisional, self.config.minimum_shared_weight
        )
        rows.append(self._make_operation(10, compression, {
            "selective_abstraction_not_loss_of_episode_records": True,
            "exception_and_variance_records_retained": True,
        }))

        # CC12: apply gradual recurrent cortical population plasticity.
        cortical = self._learn_cortex(target, provisional)
        provisional.cortical_pattern = list(cortical["cortical_pattern"])
        rows.append(self._make_operation(11, cortical, {
            "population": "distributed_sensory_and_association_cortex_rate_surrogate",
            "learning_rule": cortical["learning_rule"],
            "not_a_single_concept_centre": True,
        }, "completed_with_new_cortical_learning" if cortical["learning_update_performed"] else "completed_no_new_training_episode"))

        # CC13: predict properties and category membership for held-out examples.
        known_ids = set(self.target_examples.get(concept_id, {})) | set(
            self.comparison_examples.get(concept_id, {})
        )
        prediction_examples = []
        for episode in request.prediction_episodes:
            existing, _ = self._known_training_episode(concept_id, episode.episode_id)
            if existing is not None:
                if existing.to_dict() != episode.to_dict():
                    raise ValueError(
                        f"episode_id {episode.episode_id!r} was reused with different prediction content"
                    )
                continue
            prediction_examples.append(episode)
        predictions = [
            predict_episode(
                episode, provisional, aliases,
                self.config.prediction_prior, self.config.prediction_threshold,
            )
            for episode in prediction_examples
        ]
        rows.append(self._make_operation(12, {
            "predictions": predictions,
            "prediction_episode_count": len(predictions),
            "posterior_available": any(
                item["posterior_category_probability"] is not None for item in predictions
            ),
            "held_out_examples_preserved": True,
        }, {
            "candidate_systems": ["concept_guided_expectation", "sensory_prediction", "memory_retrieval"],
            "prediction_is_probabilistic": True,
        }, "completed" if predictions else "completed_no_heldout_prediction_examples"))

        # CC14: compare predictions with observations and revise only from explicitly labelled feedback.
        mismatch_rows = [
            estimate_prediction_mismatch(item, self.config.mismatch_threshold)
            for item in predictions
        ]
        feedback_target = [
            episode for episode in prediction_examples if episode.category_label is True
        ]
        feedback_comparison = [
            episode for episode in prediction_examples if episode.category_label is False
        ]
        feedback_gate = self.network.estimate_feedback_gates(
            mismatch_rows, predictions, prediction_examples
        )
        new_feedback_target = self._add_training_episodes(
            concept_id, feedback_target, category=True
        )
        new_feedback_comparison = self._add_training_episodes(
            concept_id, feedback_comparison, category=False
        )
        feedback_added = new_feedback_target + new_feedback_comparison
        if feedback_added:
            all_training = tuple(self.target_examples[concept_id].values()) + tuple(
                self.comparison_examples[concept_id].values()
            )
            self.network.register_episodes(all_training)
            target = tuple(self.target_examples[concept_id].values())
            comparison = tuple(self.comparison_examples[concept_id].values())
            revised = self._build_candidate(
                request, target, comparison, previous=provisional
            )
            compression = compress_shared_structure(
                revised, self.config.minimum_shared_weight
            )
            cortical = self._learn_cortex(
                target, revised, modulatory_gate=float(feedback_gate["mean_learning_gate"])
            )
            revised.cortical_pattern = list(cortical["cortical_pattern"])
            provisional = revised
        rows.append(self._make_operation(13, {
            "prediction_mismatch": mismatch_rows,
            "mismatch_threshold": self.config.mismatch_threshold,
            "new_positive_feedback_ids": [item.episode_id for item in new_feedback_target],
            "new_negative_feedback_ids": [item.episode_id for item in new_feedback_comparison],
            "feedback_examples_added": len(feedback_added),
            "revised_feature_weights": provisional.feature_weights,
            "revision_only_uses_labelled_feedback": True,
            "corticostriatal_feedback_gate": feedback_gate,
        }, {
            "candidate_mechanisms": [
                "hippocampal_retrieval", "prefrontal_evaluation",
                "corticostriatal_dopaminergic_feedback", "error_modulated_learning"
            ],
            "one_surprising_example_does_not_automatically_split_a_category": True,
            "unlabelled_mismatch_does_not_open_feedback_gate": True,
        }, "completed_with_feedback_revision" if feedback_added else "completed_without_labelled_revision"))

        # CC15: connect multiple feature and relation cues to the current concept and evidence.
        index = build_associative_index(provisional)
        self.associative_indices[concept_id] = index
        rows.append(self._make_operation(14, {
            "associative_index": index,
            "feature_cue_count": len(index["feature_to_concept"]),
            "relation_cue_count": len(index["relation_to_concept"]),
            "concept_count": len(self.concepts) + (0 if concept_id in self.concepts else 1),
        }, {
            "system": "distributed_cortical_associative_access",
            "episodic_identity_and_support_links_preserved": True,
        }))

        # CC16: store the final revisable representation with uncertainty and provenance.
        self.concepts[concept_id] = provisional
        final_concept = provisional.to_dict()
        final_concept["compressed_shared_structure"] = compression
        final_concept["representation_scope"] = (
            "shared features and relations only; prototype/exemplar, semantic hub, schema, "
            "inference, and analogical transfer are reserved for later points"
        )
        final_concept["diagnosticity_available"] = bool(comparison)
        final_concept["revision_count"] = provisional.update_count - 1
        rows.append(self._make_operation(15, {
            "concept_id": concept_id,
            "shared_feature_count": len(provisional.shared_features),
            "relation_count": len(provisional.relation_edges),
            "supporting_episode_count": len(provisional.supporting_episode_ids),
            "comparison_episode_count": len(provisional.comparison_episode_ids),
            "exceptions": provisional.exceptions,
            "uncertainty": provisional.uncertainty,
            "provenance_count": len(provisional.provenance),
            "revisable": True,
        }, {
            "representation": "distributed_shared_feature_and_relation_structure",
            "not_a_prototype_or_exemplar_model": True,
            "new_evidence_can_refine_or_differentiate_the_concept": True,
        }))

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        logger = self._log_sqlite(rows, elapsed_ms)
        return ConceptLearningResult(
            concept_id=concept_id,
            operation_trace=rows,
            concept=final_concept,
            related_episode_retrieval={
                "queries": retrieval_rows,
                "retrieved_episode_count": sum(
                    len(row["related_episodes"]) for row in retrieval_rows
                ),
            },
            predictions=predictions,
            mismatch_and_revision={
                "mismatch": mismatch_rows,
                "feedback_added": len(feedback_added),
                "positive_feedback_ids": [item.episode_id for item in new_feedback_target],
                "negative_feedback_ids": [item.episode_id for item in new_feedback_comparison],
            },
            associative_index=index,
            biological=self.network.snapshot(),
            logger=logger,
        )

    def get_concept(self, concept_id: str) -> dict[str, Any] | None:
        """Return a stored concept and its support metadata, or None when it has not been learned."""
        candidate = self.concepts.get(concept_id)
        return candidate.to_dict() if candidate is not None else None

    def snapshot(self) -> dict[str, Any]:
        """Return concept and episode-store sizes plus biological population metadata."""
        return {
            "concept_count": len(self.concepts),
            "concept_ids": sorted(self.concepts),
            "target_episode_counts": {
                key: len(value) for key, value in self.target_examples.items()
            },
            "comparison_episode_counts": {
                key: len(value) for key, value in self.comparison_examples.items()
            },
            "process_count": self.process_count,
            "biological_network": self.network.snapshot(),
        }
