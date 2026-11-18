"""End-to-end 17-stage Point 10 schema-guided inference processor.

This package consumes previously learned Point 09 schema frames. It retrieves and compares
candidate schemas, predicts missing properties and relations, estimates event transitions,
selects diagnostic queries/actions, tracks prediction errors, and updates schema likelihoods only
after explicit authorization and repeated validated feedback.
"""
from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from .biology import BiologicalSchemaInferenceNetwork
from .contracts import (
    ActionDecision, CaseEvidence, EvidenceItem, FeedbackResult, InferenceConfig,
    InferenceResult, InferredProperty, RelationEvidence, SchemaProfile,
)
from .functional import (
    merge_case_evidence, predict_missing_properties, predict_missing_relations,
    predict_next_events, rank_schema_profiles, select_action,
    select_hierarchical_category, select_information_query,
)
from .registry import OPERATION_REGISTRY


def _operation(
    trace: list[dict[str, Any]], index: int,
    functional: Mapping[str, Any], biological: Mapping[str, Any] | None = None,
    status: str = "completed",
) -> None:
    """Append a source-aligned inference stage with its equations and biological interpretation."""
    spec = OPERATION_REGISTRY[index]
    trace.append({
        "step": index + 1, "operation_id": spec.operation_id,
        "operation": spec.name, "status": status, "explanation": spec.purpose,
        "evidence_level": spec.evidence_level, "mathematical_form": spec.mathematical_form,
        "functional_output": dict(functional), "biological_output": dict(biological or {}),
    })


def _profile_summary(profile: SchemaProfile) -> dict[str, Any]:
    """Summarize a schema graph without copying individual episode details."""
    return {
        "schema_id": profile.schema_id, "name": profile.frame.name,
        "slot_names": sorted(profile.frame.slots),
        "relations": [
            {"subject": item.subject_slot, "predicate": item.predicate,
             "object": item.object_slot, "confidence": item.confidence,
             "evidential_status": item.evidential_status}
            for item in profile.frame.relations
        ],
        "event_types": list(profile.frame.event_types),
        "transition_count": len(profile.frame.transitions),
        "goal_names": sorted(profile.frame.goals),
        "parent_schema_ids": list(profile.frame.parent_schema_ids),
        "child_schema_ids": list(profile.frame.child_schema_ids),
    }


def _prediction_objects(rows: Sequence[Mapping[str, Any]]) -> tuple[InferredProperty, ...]:
    """Convert generic prediction rows into immutable, provenance-aware output records."""
    output = []
    for row in rows:
        output.append(InferredProperty(
            kind=str(row["kind"]), name=str(row["name"]), value=row["value"],
            confidence=float(row["confidence"]), source_schema_id=str(row["source_schema_id"]),
            provenance=str(row.get("provenance", "schema_prediction")),
            status=str(row.get("status", "inferred")),
            supporting_evidence_ids=tuple(row.get("supporting_evidence_ids", ())),
        ))
    return tuple(output)


def _evidence_ledger(
    case: CaseEvidence, retrieved: Sequence[Mapping[str, Any]],
    predictions: Sequence[InferredProperty],
) -> tuple[dict[str, Any], ...]:
    """Build a ledger that separates current evidence, superseded observations, memory, and inference."""
    rows: list[dict[str, Any]] = []
    for feature, item in sorted(case.features.items()):
        rows.append({
            "content_type": "feature", "feature": feature, "value": item.value,
            "source": item.source, "status": item.status, "reliability": item.reliability,
            "evidence_id": item.evidence_id, "provenance": "current_case",
        })
    for item in case.evidence_history:
        rows.append({
            "content_type": "feature", "feature": item.feature, "value": item.value,
            "source": item.source, "status": "superseded_observation",
            "reliability": item.reliability, "evidence_id": item.evidence_id,
            "provenance": "prior_case_snapshot",
        })
    for relation in case.relations:
        rows.append({
            "content_type": "relation",
            "value": {"subject": relation.subject, "predicate": relation.predicate,
                      "object": relation.object},
            "source": relation.source, "status": relation.status,
            "reliability": relation.reliability, "evidence_id": relation.evidence_id,
            "provenance": "current_case",
        })
    for recalled in retrieved:
        rows.append({
            "content_type": "episodic_memory", "case_id": recalled["case_id"],
            "source": recalled["source_reference"], "similarity": recalled["similarity"],
            "status": "retrieved", "provenance": "specific_episode",
        })
    for prediction in predictions:
        rows.append({
            "content_type": prediction.kind, "feature": prediction.name,
            "value": prediction.value, "source": prediction.source_schema_id,
            "status": "inferred", "confidence": prediction.confidence,
            "provenance": prediction.provenance, "observed": False,
        })
    return tuple(rows)


class SchemaGuidedInferenceProcessor:
    """Perform uncertainty-aware schema matching, completion, generalization, and feedback.

    The input schema library contains Point 09 frames or profiles around those frames. The
    processor does not derive a new schema from a single case and never treats predictions as
    observed evidence. Updating reusable schema likelihoods requires explicit permission.
    """

    def __init__(
        self,
        schemas: Sequence[SchemaProfile] = (),
        config: InferenceConfig | None = None,
        database_path: str | Path | None = None,
        repository_root: str | Path | None = None,
    ) -> None:
        """Initialize schema profiles, biological surrogates, episodic memory, and logger settings."""
        self.config = config or InferenceConfig()
        self.profiles: dict[str, SchemaProfile] = {}
        self.biological_network = BiologicalSchemaInferenceNetwork(self.config)
        self.database_path = Path(database_path).expanduser().resolve() if database_path else None
        self.repository_root = (
            Path(repository_root).expanduser().resolve()
            if repository_root else Path(__file__).resolve().parents[1]
        )
        self.profile_update_history: list[dict[str, Any]] = []
        for profile in schemas:
            self.add_schema(profile)

    def add_schema(self, profile: SchemaProfile) -> None:
        """Register one previously learned Point 09 schema profile without re-learning its frame."""
        if not isinstance(profile, SchemaProfile):
            raise TypeError("profile must be a SchemaProfile around a Point 09 SchemaFrame")
        existing = self.profiles.get(profile.schema_id)
        if existing is not None and existing.frame.to_dict() != profile.frame.to_dict():
            raise ValueError("schema_id already exists with a different Point 09 frame")
        self.profiles[profile.schema_id] = profile
        self.biological_network.semantic.index_profiles((profile,))

    def add_schema_frame(
        self,
        frame: Any,
        feature_probabilities: Mapping[str, Mapping[str, float]] | None = None,
        feature_diagnosticity: Mapping[str, float] | None = None,
        prior_probability: float | None = None,
    ) -> SchemaProfile:
        """Wrap a Point 09 frame with optional cue likelihoods and add it to the inference library."""
        profile = SchemaProfile(
            frame=frame, feature_probabilities=feature_probabilities or {},
            feature_diagnosticity=feature_diagnosticity or {},
            prior_probability=prior_probability,
        )
        self.add_schema(profile)
        return profile

    def _retrieve_candidates(self, case: CaseEvidence) -> tuple[list[SchemaProfile], dict[str, Any]]:
        """Use distributed cue activation and context/goal gain to preselect candidate schemas."""
        profiles = list(self.profiles.values())
        if not profiles:
            raise RuntimeError("register at least one Point 09 schema before inference")
        semantic = self.biological_network.semantic.activate(case, limit=len(profiles))
        prefrontal = self.biological_network.prefrontal.modulate(case, profiles)
        combined = {
            profile.schema_id: 0.60 * semantic.get(profile.schema_id, 0.0)
            + 0.40 * prefrontal.get(profile.schema_id, 0.0)
            for profile in profiles
        }
        has_signal = any(value > 0.0 for value in combined.values())
        ranked_ids = sorted(combined, key=lambda key: (-combined[key], key))
        if not has_signal:
            selected = profiles
            retrieval_mode = "broad_fallback_no_discriminating_activation"
        else:
            keep = set(ranked_ids[:self.config.max_candidates])
            selected = [item for item in profiles if item.schema_id in keep]
            retrieval_mode = "cue_and_context_guided_top_k"
        diagnostics = {
            "semantic_activation": semantic, "prefrontal_gain": prefrontal,
            "combined_retrieval_score": combined, "retrieval_mode": retrieval_mode,
            "candidate_schema_ids": [item.schema_id for item in selected],
            "unknown_candidate_retained": True,
        }
        return selected, diagnostics

    def infer(self, case: CaseEvidence) -> InferenceResult:
        """Run all 17 Point 10 stages and return a provenance-aware interpretation."""
        started = time.perf_counter()
        if not isinstance(case, CaseEvidence):
            raise TypeError("case must be a CaseEvidence record")
        profiles, retrieval = self._retrieve_candidates(case)
        trace: list[dict[str, Any]] = []

        evidence_summary = {
            "case_id": case.case_id, "feature_count": len(case.features),
            "usable_observation_count": len(case.usable_evidence),
            "explicitly_contradicted_count": sum(
                item.status == "contradicted" for item in case.usable_evidence
            ),
            "unknown_or_unobserved_count": sum(
                item.status in {"unknown", "not_observed"} for item in case.features.values()
            ),
            "relation_count": len(case.relations),
            "event_count": len(case.event_history),
            "predictions_inserted_into_evidence": False,
        }
        _operation(trace, 0, evidence_summary)

        _operation(trace, 1, retrieval, {
            "temporal_semantic": self.biological_network.semantic.snapshot(),
            "medial_prefrontal": self.biological_network.prefrontal.snapshot(),
        })
        schema_graphs = [_profile_summary(profile) for profile in profiles]
        _operation(trace, 2, {
            "candidate_schema_graphs": schema_graphs,
            "retrieved_episode_store_size": len(self.biological_network.hippocampus.cases),
        })
        alignment = {
            profile.schema_id: {
                "matched_feature_models": sorted(set(case.features) & set(profile.feature_probabilities)),
                "role_slots_with_evidence": sorted(
                    key.split(":", 1)[1] for key in case.features
                    if key.startswith("role:") and key.split(":", 1)[1] in profile.frame.slots
                ),
                "observed_relations": [
                    {"subject": item.subject, "predicate": item.predicate, "object": item.object,
                     "status": item.status}
                    for item in case.relations
                ],
            } for profile in profiles
        }
        _operation(trace, 3, {"evidence_to_slot_alignment": alignment})

        diagnosticity = {
            item.feature: {
                "reliability": item.reliability,
                "profile_weights": {
                    profile.schema_id: profile.feature_diagnosticity.get(item.feature, 1.0)
                    for profile in profiles
                },
                "status": item.status,
            } for item in case.usable_evidence
        }
        _operation(trace, 4, {
            "feature_diagnosticity": diagnosticity,
            "redundancy_warning": (
                "Feature dependencies are not automatically independent; correlated cues require "
                "calibration or a joint feature model."
            ),
        })

        ranked = rank_schema_profiles(case, profiles, self.config)
        _operation(trace, 5, {
            "candidate_posteriors": [item.to_dict() for item in ranked],
            "unknown_posterior": next(
                (item.posterior for item in ranked if item.schema_id == "unknown"), 0.0
            ),
        })

        selected_id, selected_category, confidence, selection_reason = select_hierarchical_category(
            ranked, profiles, self.config
        )
        selected_profile = self.profiles.get(selected_id) if selected_id else None
        inferred_property_rows: tuple[dict[str, Any], ...] = ()
        inferred_relation_rows: tuple[dict[str, Any], ...] = ()
        next_event_rows: tuple[dict[str, Any], ...] = ()
        if selected_profile is not None:
            selected_posterior = next(
                (item.posterior for item in ranked if item.schema_id == selected_profile.schema_id),
                0.0,
            )
            inferred_property_rows = predict_missing_properties(
                case, selected_profile, selected_posterior, self.config
            )
            inferred_relation_rows = predict_missing_relations(
                case, selected_profile, selected_posterior, self.config
            )
        _operation(trace, 6, {
            "predictions_by_selected_schema": list(inferred_property_rows),
            "prediction_count": len(inferred_property_rows),
            "all_predictions_marked_inferred": all(
                item.get("status") == "inferred" and item.get("observed") is False
                for item in inferred_property_rows
            ),
        })
        _operation(trace, 7, {
            "inferred_relations": list(inferred_relation_rows),
            "relation_prediction_count": len(inferred_relation_rows),
            "observed_and_inferred_relations_separate": True,
        })
        _operation(trace, 8, {
            "selected_schema_id": selected_id, "selected_category": selected_category,
            "confidence": confidence, "selection_reason": selection_reason,
            "hierarchical_fallback": selection_reason == "broader_parent_selected",
        })

        if selected_profile is not None:
            selected_posterior = next(
                (item.posterior for item in ranked if item.schema_id == selected_profile.schema_id),
                0.0,
            )
            next_event_rows = predict_next_events(
                case, selected_profile, selected_posterior, self.config
            )
        _operation(trace, 9, {
            "next_event_predictions": list(next_event_rows),
            "temporal_and_causal_claims_separated": True,
        })

        # A new inference may use predictions to guide attention, but it never mutates the
        # current evidence record or relabels an unobserved value as an observation.
        _operation(trace, 10, {
            "current_evidence_snapshot_unchanged": True,
            "revised_evidence_required_for_confirmation": True,
            "predicted_detail_promoted_to_observed": False,
        })

        query = select_information_query(case, ranked, profiles, self.config)
        focus_feature = None
        if query is None and selection_reason == "abstained_low_confidence_or_margin":
            focus_feature = self.biological_network.attention.choose_diagnostic_feature(
                profiles, set(case.features)
            )
        _operation(trace, 11, {
            "query_recommendation": query.to_dict() if query else None,
            "diagnostic_feature_focus": focus_feature,
            "query_executed_automatically": False,
        })

        # Memory retrieval is separate from current sensory input. The trace stores references
        # and similarity, not a fabricated sensory confirmation.
        biological = self.biological_network.process(case, ranked, profiles)
        retrieved = biological.get("episodic_recall", [])
        predictions = (
            tuple(_prediction_objects(inferred_property_rows))
            + tuple(_prediction_objects(inferred_relation_rows))
            + tuple(_prediction_objects(next_event_rows))
        )
        ledger = _evidence_ledger(case, retrieved, predictions)
        _operation(trace, 12, {
            "evidence_ledger": list(ledger),
            "source_categories": [
                "current_case", "prior_case_snapshot", "specific_episode", "schema_prediction"
            ],
        }, {
            "hippocampal_recall": self.biological_network.hippocampus.snapshot(),
        })

        action = select_action(case, ranked, self.config)
        _operation(trace, 13, {
            "action_decision": action.to_dict() if action else None,
            "action_requires_caller_supplied_options": True,
            "action_risk_included": action is not None,
        }, {"orbitofrontal_action_selection": biological.get("action_selection")})

        _operation(trace, 14, {
            "outcome_observed": False, "prediction_error_status": "pending_feedback",
            "untested_predictions_are_not_counted_as_errors": True,
        })
        _operation(trace, 15, {
            "schema_update_authorized": False,
            "schema_update_requires": [
                "explicit_learning_authorization", "validated_feedback",
                "minimum_distinct_case_count", "source_and_context_preservation",
            ],
            "profile_revision_count": len(self.profile_update_history),
        })
        self.biological_network.observe_case(case)
        biological = self.biological_network.snapshot()
        diagnostics = {
            "candidate_count": len(ranked), "registered_schema_count": len(self.profiles),
            "usable_evidence_count": len(case.usable_evidence),
            "inferred_property_count": len(inferred_property_rows),
            "inferred_relation_count": len(inferred_relation_rows),
            "next_event_prediction_count": len(next_event_rows),
            "episodic_recall_count": len(retrieved),
            "retrieved_case_ids": [item["case_id"] for item in retrieved],
            "selected_schema_id": selected_id, "selection_reason": selection_reason,
            "duration_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "scientific_scope": (
                "Bayesian, graph, and utility equations are candidate computational models; "
                "biological components are functional surrogates, not literal neural arithmetic."
            ),
        }
        _operation(trace, 16, {
            "case_retained_in_episodic_store": case.case_id in self.biological_network.hippocampus.cases,
            "schema_updated_during_inference": False,
            "schema_ids_available_for_future_inference": sorted(self.profiles),
            "diagnostics": diagnostics,
        }, biological)
        result = InferenceResult(
            case_id=case.case_id, candidate_scores=ranked,
            selected_schema_id=selected_id, selected_category=selected_category,
            confidence=confidence, inferred_properties=_prediction_objects(inferred_property_rows),
            inferred_relations=_prediction_objects(inferred_relation_rows),
            next_event_predictions=_prediction_objects(next_event_rows),
            query_recommendation=query, action_decision=action,
            evidence_ledger=ledger, operation_trace=tuple(trace),
            biological=biological, diagnostics=diagnostics,
        )
        if self.database_path:
            record_result_to_sqlite(result, self.database_path, self.repository_root)
        return result

    def refine(
        self,
        previous: CaseEvidence,
        additional_evidence: Sequence[EvidenceItem],
        relations: Sequence[Any] = (),
        event_history: Sequence[str] = (),
    ) -> CaseEvidence:
        """Create a revised case snapshot while retaining superseded observations in history."""
        return merge_case_evidence(previous, additional_evidence, relations, event_history)

    def evaluate_outcome(
        self,
        result: InferenceResult,
        observed_evidence: Sequence[EvidenceItem],
        observed_next_events: Sequence[str] = (),
        observed_relations: Sequence[RelationEvidence] = (),
        *,
        learning_authorized: bool = False,
        validated_case_ids: Sequence[str] = (),
    ) -> FeedbackResult:
        """Score tested predictions and update feature likelihoods only after explicit validated authorization."""
        if result.case_id not in self.biological_network.hippocampus.cases:
            raise KeyError("the inference case is not present in episodic memory")
        feedback_by_feature = {item.feature: item for item in observed_evidence}
        tested: list[dict[str, Any]] = []
        brier_values: list[float] = []
        surprise_values: list[float] = []
        for prediction in result.inferred_properties:
            observation = feedback_by_feature.get(prediction.name)
            if observation is None or observation.status not in {"observed", "contradicted"}:
                continue
            actual = float(
                str(observation.value).casefold() == str(prediction.value).casefold()
            )
            probability = clamp(prediction.confidence)
            brier = (probability - actual) ** 2
            surprise = -math.log(max(self.config.probability_floor,
                                     probability if actual else 1.0 - probability))
            tested.append({
                "kind": prediction.kind, "name": prediction.name, "predicted_value": prediction.value,
                "observed_value": observation.value, "correct": bool(actual),
                "brier_score": brier, "surprise": surprise,
                "prediction_source": prediction.source_schema_id,
                "observation_source": observation.source,
            })
            brier_values.append(brier)
            surprise_values.append(surprise)
        relation_feedback = {
            (item.subject, item.predicate, item.object): item
            for item in observed_relations
        }
        for prediction in result.inferred_relations:
            predicted = prediction.value
            key = (
                str(predicted.get("subject", "")).casefold(),
                str(predicted.get("predicate", "")).casefold(),
                str(predicted.get("object", "")).casefold(),
            )
            observation = relation_feedback.get(key)
            if observation is None or observation.status not in {"observed", "contradicted"}:
                continue
            actual = float(observation.status == "observed")
            probability = clamp(prediction.confidence)
            brier = (probability - actual) ** 2
            surprise = -math.log(max(
                self.config.probability_floor, probability if actual else 1.0 - probability
            ))
            tested.append({
                "kind": "relation", "name": prediction.name, "predicted_value": predicted,
                "observed_status": observation.status, "correct": bool(actual),
                "brier_score": brier, "surprise": surprise,
                "prediction_source": prediction.source_schema_id,
                "observation_source": observation.source,
            })
            brier_values.append(brier)
            surprise_values.append(surprise)
        for prediction in result.next_event_predictions:
            if not observed_next_events:
                continue
            actual = float(prediction.name in {str(item).casefold() for item in observed_next_events})
            probability = clamp(prediction.confidence)
            brier = (probability - actual) ** 2
            surprise = -math.log(max(self.config.probability_floor,
                                     probability if actual else 1.0 - probability))
            tested.append({
                "kind": "next_event", "name": prediction.name, "correct": bool(actual),
                "brier_score": brier, "surprise": surprise,
                "observation_source": "later_event_observation",
            })
            brier_values.append(brier)
            surprise_values.append(surprise)
        untested = (
            len(result.inferred_properties) + len(result.inferred_relations)
            + len(result.next_event_predictions) - len(tested)
        )
        updated_id: str | None = None
        update_status = "not_authorized"
        if learning_authorized:
            unique_cases = sorted(set(str(item) for item in validated_case_ids))
            if len(unique_cases) < self.config.minimum_update_cases:
                update_status = "insufficient_distinct_validated_cases"
            elif result.selected_schema_id is None:
                update_status = "no_selected_schema"
            elif not observed_evidence:
                update_status = "no_validated_feature_feedback"
            else:
                profile = self.profiles[result.selected_schema_id]
                counts = {
                    feature: dict(values) for feature, values in profile.feature_counts.items()
                }
                probabilities = {
                    feature: dict(values) for feature, values in profile.feature_probabilities.items()
                }
                updated_features: list[str] = []
                for observation in observed_evidence:
                    if observation.status not in {"observed", "contradicted"}:
                        continue
                    feature = observation.feature
                    feature_counts = counts.setdefault(feature, {})
                    if not feature_counts:
                        initial = probabilities.get(feature, {})
                        scale = max(1, len(profile.frame.supporting_episode_ids))
                        feature_counts.update({key: value * scale for key, value in initial.items()})
                    key = str(observation.value).casefold()
                    feature_counts[key] = feature_counts.get(key, 0.0) + (
                        self.config.update_learning_rate * observation.reliability
                    )
                    total = sum(feature_counts.values())
                    if total > 0.0:
                        probabilities[feature] = {
                            name: value / total for name, value in feature_counts.items()
                        }
                        updated_features.append(feature)
                if updated_features:
                    revised = SchemaProfile(
                        frame=profile.frame, feature_probabilities=probabilities,
                        feature_diagnosticity=profile.feature_diagnosticity,
                        prior_probability=profile.prior_probability, feature_counts=counts,
                    )
                    self.profiles[profile.schema_id] = revised
                    self.biological_network.semantic.index_profiles((revised,))
                    updated_id = profile.schema_id
                    update_status = "feature_likelihoods_updated_after_authorized_validation"
                    self.profile_update_history.append({
                        "schema_id": profile.schema_id, "validated_case_ids": unique_cases,
                        "updated_features": sorted(set(updated_features)),
                    })
                else:
                    update_status = "no_usable_feature_feedback"
        return FeedbackResult(
            case_id=result.case_id, tested_predictions=tuple(tested),
            untested_prediction_count=max(0, untested),
            mean_brier_score=(sum(brier_values) / len(brier_values) if brier_values else None),
            mean_surprise=(sum(surprise_values) / len(surprise_values) if surprise_values else None),
            schema_update_authorized=learning_authorized, updated_schema_id=updated_id,
            update_status=update_status,
        )


def register_annotations(
    database_path: str | Path, repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Index all Point 10 class and callable docstrings in the ORE SQLite annotation catalogue."""
    from ore_logger.catalog import sync_repository
    from ore_logger.database import LoggerDatabase

    root = (Path(repository_root).expanduser().resolve() if repository_root
            else Path(__file__).resolve().parents[1])
    return sync_repository(LoggerDatabase(database_path), root,
                           include_paths={"ore_schema_guided_inference"})


def record_result_to_sqlite(
    result: InferenceResult,
    database_path: str | Path,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Write the 17-stage inference trace to the ORE SQLite event table and index annotations."""
    from ore_logger.catalog import sync_repository
    from ore_logger.database import LoggerDatabase

    root = (Path(repository_root).expanduser().resolve() if repository_root
            else Path(__file__).resolve().parents[1])
    database = LoggerDatabase(database_path)
    catalog = sync_repository(database, root, include_paths={"ore_schema_guided_inference"})
    run_id = database.create_run(
        script_path="ore_schema_guided_inference/point10",
        module_name="ore_schema_guided_inference.pipeline",
        entry_kind="schema_guided_inference",
        working_directory=str(root),
    )
    started = time.perf_counter()
    try:
        for sequence, item in enumerate(result.operation_trace, start=1):
            database.record_event(
                run_id=run_id, sequence_no=sequence, event_kind="operation",
                source_path="ore_schema_guided_inference/pipeline.py", elapsed_ms=0.0,
                metadata={
                    "operation_id": item.get("operation_id"), "operation": item.get("operation"),
                    "status": item.get("status"), "evidence_level": item.get("evidence_level"),
                    "mathematical_form": item.get("mathematical_form"),
                    "functional_output": item.get("functional_output"),
                    "biological_output": item.get("biological_output"),
                },
            )
        database.finish_run(
            run_id, "completed", duration_ms=round((time.perf_counter() - started) * 1000.0, 3)
        )
    except Exception as exc:
        database.finish_run(
            run_id, "failed", duration_ms=round((time.perf_counter() - started) * 1000.0, 3),
            error_summary=type(exc).__name__ + ": " + str(exc),
        )
        raise
    return {
        "run_id": run_id, "annotation_catalog": catalog,
        "database_summary": database.summary(),
    }
