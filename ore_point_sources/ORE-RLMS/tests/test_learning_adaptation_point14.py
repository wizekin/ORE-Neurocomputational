"""Acceptance tests: transfer learning, TTT, multimodal gateway, and fusion (Steps 12-22)."""
from __future__ import annotations

import numpy as np
import pytest

from ore_learning.adaptation import (
    AdaptationComparison,
    AdaptationDataset,
    AdaptationMethod,
    AdaptationPolicyError,
    IsolatedTrainingRun,
    TTTExperiment,
    TTTPolicyError,
    evaluate_promotion,
    frozen_feature_quality,
    select_adaptation_method,
    split_dataset,
)
from ore_learning.adapters import (
    ModalityObservation,
    ModalityPayload,
    ModalityValidationError,
    ObservationGateway,
    fuse_observations,
    map_to_semantics,
)
from ore_learning.config import LearningConfig
from ore_learning.contracts import ApprovalRecord, Modality
from ore_learning.nn import TinyLinearModel, TinyMLP


def make_dataset(n: int = 120, seed: int = 5) -> tuple[np.ndarray, np.ndarray,
                                                       AdaptationDataset]:
    """A linear regression dataset with a deterministic split."""
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n, 4))
    targets = features @ np.array([[0.8], [-1.2], [0.3], [0.5]]) \
        + 0.03 * rng.normal(size=(n, 1))
    splits = split_dataset(features, targets, seed=seed)
    dataset = AdaptationDataset(
        "ds-adapt", "cc-by-4.0", "2024-Q1", features, targets, splits,
        filtering_rules=("drop-nulls", "unit-check"))
    return features, targets, dataset


def test_transfer_evaluation_uses_held_out_data_without_leakage():
    """Acceptance: transfer evaluation uses held-out data without leakage."""
    _, _, dataset = make_dataset()
    audit = dataset.leakage_audit()
    assert audit["status"] == "leak-free"
    assert audit["train_test_overlap"] == []
    # the training run touches only the train split
    model = TinyLinearModel(4, 1, seed=2)
    run = IsolatedTrainingRun(model, dataset, AdaptationMethod.LINEAR_PROBE,
                              seed=7, epochs=10)
    record = run.run()
    assert record["train_split_size"] == len(dataset.splits["train"])
    assert record["train_split_size"] < len(dataset.features)
    held_out = set(dataset.splits["test"])
    trained_on = set(dataset.splits["train"])
    assert trained_on.isdisjoint(held_out)
    # quality is measured on the held-out split only
    quality = frozen_feature_quality(run.model, dataset)
    assert np.isfinite(quality)


def test_leakage_audit_flags_near_duplicates_and_overlap():
    """Near-duplicate rows across splits are flagged; overlap is refused."""
    rng = np.random.default_rng(3)
    features = rng.normal(size=(6, 2))
    targets = rng.normal(size=(6, 1))
    leaky = AdaptationDataset("ds-leak", "mit", "2024", features, targets,
                              {"train": [0, 1, 2], "validation": [],
                               "test": [3, 4, 5]})
    # inject a duplicate of a train row into the test split
    leaky.features[3] = leaky.features[0]
    audit = leaky.leakage_audit()
    assert audit["status"] == "leaky"
    assert len(audit["cross_split_duplicates"]) >= 1
    with pytest.raises(AdaptationPolicyError, match="leakage"):
        AdaptationDataset("ds-overlap", "mit", "2024", features, targets,
                          {"train": [0, 1, 2], "validation": [],
                           "test": [2, 3]})


def test_adapted_model_promotion_requires_predefined_improvement():
    """Acceptance: adapted model promotion requires a predefined improvement."""
    _, _, dataset = make_dataset()
    base = TinyLinearModel(4, 1, seed=2)
    base_quality = frozen_feature_quality(base, dataset)
    run = IsolatedTrainingRun(base.copy(), dataset, AdaptationMethod.LINEAR_PROBE,
                              seed=7, epochs=12)
    run.run()
    adapted_quality = frozen_feature_quality(run.model, dataset)
    comparison = AdaptationComparison(
        base_quality=base_quality, adapted_quality=adapted_quality,
        base_latency_ms=1.0, adapted_latency_ms=1.0,
        base_memory_bytes=base.memory_bytes(),
        adapted_memory_bytes=run.model.memory_bytes(),
        prior_capability_regression=0.0, dataset_id=dataset.dataset_id)
    assert comparison.adaptation_gain > 0
    # a gain below the configured minimum is not promotable
    strict = evaluate_promotion(comparison,
                                LearningConfig(min_adaptation_gain=1e9))
    assert not strict.approved
    # the measured gain meets the default minimum -> promotion passes
    default = evaluate_promotion(comparison, LearningConfig())
    assert default.approved and not default.approved_tradeoff
    # a prior-capability regression blocks promotion even with a gain
    regressed = AdaptationComparison(
        base_quality=base_quality, adapted_quality=adapted_quality + 5.0,
        base_latency_ms=1.0, adapted_latency_ms=1.0,
        base_memory_bytes=base.memory_bytes(),
        adapted_memory_bytes=run.model.memory_bytes(),
        prior_capability_regression=0.2, dataset_id=dataset.dataset_id)
    assert not evaluate_promotion(regressed, LearningConfig()).approved


def test_approved_tradeoff_requires_recorded_approval():
    """Step 16: quality-for-latency trade-offs need a human approval record."""
    comparison = AdaptationComparison(
        base_quality=0.90, adapted_quality=0.87,  # drop within the bound
        base_latency_ms=100.0, adapted_latency_ms=10.0,
        base_memory_bytes=100, adapted_memory_bytes=100,
        prior_capability_regression=0.0, dataset_id="ds")
    without_approval = evaluate_promotion(comparison, LearningConfig())
    assert not without_approval.approved
    approval = ApprovalRecord("appr-3", "reviewer@ore", "quality-latency trade-off")
    with_approval = evaluate_promotion(comparison, LearningConfig(),
                                       approval=approval)
    assert with_approval.approved and with_approval.approved_tradeoff
    # a quality drop beyond the bound is never promotable
    too_far = AdaptationComparison(
        base_quality=0.90, adapted_quality=0.60,
        base_latency_ms=100.0, adapted_latency_ms=1.0,
        base_memory_bytes=100, adapted_memory_bytes=100,
        prior_capability_regression=0.0, dataset_id="ds")
    assert not evaluate_promotion(too_far, LearningConfig(),
                                  approval=approval).approved


def test_frozen_features_first_and_smallest_method_ladder():
    """Steps 12-13: frozen reuse first; the smallest suitable method next."""
    _, _, dataset = make_dataset()
    model = TinyLinearModel(4, 1, seed=2)
    # frozen representations already suffice at a modest target
    frozen = frozen_feature_quality(model, dataset)
    assert select_adaptation_method(frozen, target=frozen) is \
        AdaptationMethod.FROZEN_REUSE
    assert select_adaptation_method(frozen, target=frozen - 1e-9) is \
        AdaptationMethod.FROZEN_REUSE
    # insufficient reuse -> the smallest ladder step that meets the target
    measured = {AdaptationMethod.LINEAR_PROBE.value: 0.5,
                AdaptationMethod.PEFT_ADAPTER.value: 0.8,
                AdaptationMethod.FULL_FINETUNE.value: 0.9}
    assert select_adaptation_method(-2.0, target=0.45,
                                    adapted_qualities=measured) is \
        AdaptationMethod.LINEAR_PROBE
    assert select_adaptation_method(-2.0, target=0.75,
                                    adapted_qualities=measured) is \
        AdaptationMethod.PEFT_ADAPTER
    # nothing measured meets the target -> cheapest experiment requested
    assert select_adaptation_method(-2.0, target=0.99,
                                    adapted_qualities=measured) is \
        AdaptationMethod.LINEAR_PROBE


def test_isolated_training_run_records_provenance_and_touches_no_production_state():
    """Step 15: seeds, optimizer, digests recorded; USG untouched; inactive by default."""
    _, _, dataset = make_dataset()
    model = TinyLinearModel(4, 1, seed=11)
    run = IsolatedTrainingRun(model, dataset, AdaptationMethod.LINEAR_PROBE,
                              seed=42, epochs=5, learning_rate=0.03, batch_size=4)
    record = run.run()
    assert record["seed"] == 42 and record["optimizer"] == "sgd"
    assert record["learning_rate"] == 0.03 and record["batch_size"] == 4
    assert record["candidate_digest"] == model.digest()
    assert record["base_digest"] != record["candidate_digest"] or record[
        "final_loss"] is not None
    assert record["usg_modified"] is False
    assert record["active_by_default"] is False


def test_ttt_cannot_persist_into_the_production_artifact():
    """Acceptance: TTT cannot persist into the production artifact."""
    model = TinyLinearModel(3, 1, seed=5)
    canonical_digest = model.digest()
    experiment = TTTExperiment(LearningConfig(ttt_max_steps=4), enabled=True)
    rng = np.random.default_rng(1)
    result = experiment.run(model, rng.normal(size=(8, 3)))
    assert result.steps_used >= 1
    assert result.persisted is False
    # the canonical artifact is untouched
    assert model.digest() == canonical_digest
    # persistence is structurally refused
    with pytest.raises(TTTPolicyError, match="cannot persist"):
        experiment.persist_to_canonical(result)
    # still untouched after the refused attempt
    assert model.digest() == canonical_digest


def test_ttt_is_opt_in():
    """Step 17: TTT refuses to run unless explicitly enabled."""
    model = TinyLinearModel(3, 1, seed=5)
    disabled = TTTExperiment(LearningConfig(), enabled=False)
    with pytest.raises(TTTPolicyError, match="opt-in"):
        disabled.run(model, np.zeros((4, 3)))


def test_multimodal_gateway_preserves_references_and_timestamps():
    """Acceptance: multimodal gateway preserves source references and timestamps."""
    gateway = ObservationGateway()
    rng = np.random.default_rng(7)
    payloads = [
        ModalityPayload(Modality.IMAGE, "cam/frame-42", 1700001000.5,
                        rng.normal(size=(16, 16)), encoding="float64"),
        ModalityPayload(Modality.TEXT, "notes/page-3", 1700001001.0,
                        "the pattern is clear", encoding="utf-8",
                        coordinate_frame="document"),
        ModalityPayload(Modality.SPATIAL, "lidar/scan-9", 1700001002.25,
                        rng.normal(size=(10, 3)), coordinate_frame="robot-base"),
    ]
    normalized = gateway.validate_many(payloads)
    assert len(normalized) == 3 and not gateway.rejected
    by_ref = {item.source_ref: item for item in normalized}
    assert set(by_ref) == {"cam/frame-42", "notes/page-3", "lidar/scan-9"}
    assert by_ref["cam/frame-42"].timestamp == 1700001000.5
    assert by_ref["lidar/scan-9"].coordinate_frame == "robot-base"
    assert by_ref["cam/frame-42"].payload.modality is Modality.IMAGE
    assert all(item.digest for item in normalized)


def test_invalid_modality_dimensions_or_encodings_are_rejected():
    """Acceptance: invalid modality dimensions or encodings are rejected."""
    gateway = ObservationGateway(LearningConfig(max_image_side_pixels=64,
                                                max_audio_seconds=1.0))
    rng = np.random.default_rng(2)
    invalid = [
        ModalityPayload(Modality.IMAGE, "bad/img-dims", 1.0,
                        rng.normal(size=(4, 4, 7))),          # too many channels
        ModalityPayload(Modality.IMAGE, "bad/img-size", 1.0,
                        rng.normal(size=(128, 128))),          # exceeds side limit
        ModalityPayload(Modality.IMAGE, "bad/img-nan", 1.0,
                        np.array([[np.nan] * 4] * 4)),         # non-finite
        ModalityPayload(Modality.AUDIO, "bad/audio-dim", 1.0,
                        rng.normal(size=96000), sample_rate=48000.0),  # 2 s > 1 s
        ModalityPayload(Modality.AUDIO, "bad/audio-rate", 1.0,
                        rng.normal(size=(100,)), sample_rate=0.0),
        ModalityPayload(Modality.TEXT, "bad/text-empty", 1.0, "   "),
        ModalityPayload(Modality.SPATIAL, "bad/spatial", 1.0,
                        rng.normal(size=(5, 4))),               # not (N,2/3)
        ModalityPayload(Modality.VIDEO, "bad/video", 1.0,
                        [np.zeros((4, 4)), np.zeros((5, 5))]),  # mixed frame shapes
    ]
    accepted = gateway.validate_many(invalid)
    assert accepted == []
    reasons = {item["source_ref"]: item["reason"] for item in gateway.rejected}
    assert len(reasons) == 8
    assert "channel" in reasons["bad/img-dims"]
    assert "exceeds the declared limit" in reasons["bad/img-size"]
    assert "non-finite" in reasons["bad/img-nan"]
    assert "duration" in reasons["bad/audio-dim"]
    assert "sample_rate" in reasons["bad/audio-rate"]
    assert "non-empty string" in reasons["bad/text-empty"]
    assert "spatial" in reasons["bad/spatial"]
    assert "one shape" in reasons["bad/video"]
    with pytest.raises(ModalityValidationError):
        gateway.validate(ModalityPayload(Modality.IMAGE, "bad/direct", 1.0,
                                         np.zeros((3, 3, 5))))


def test_conflicting_modality_observations_remain_visible():
    """Acceptance: conflicting modality observations remain visible."""
    # scenario 1 -- a near-tie between modalities: retained as a score-margin conflict
    near_tie = [
        ModalityObservation(Modality.IMAGE, "img-1", 1.0, region="center",
                            concept_hint="edge", support=0.90, confidence=0.8),
        ModalityObservation(Modality.TEXT, "txt-1", 1.1,
                            concept_hint="surface", support=0.90, confidence=0.9),
    ]
    fused = fuse_observations(map_to_semantics(near_tie))
    assert len(fused.ranked) == 2
    margin = fused.ranked[0].score - fused.ranked[1].score
    assert margin <= 0.10
    assert any(conflict["kind"] == "score_margin" for conflict in fused.conflicts)
    margin_conflict = next(c for c in fused.conflicts if c["kind"] == "score_margin")
    # the competing observations from BOTH concepts are retained, not averaged away
    assert len(margin_conflict["retained_observations"]) == 2
    assert fused.top().score < 1.0  # a fusion score is evidence strength, not proof

    # scenario 2 -- one modality contradicts the fused top: retained as disagreement
    contradiction = [
        ModalityObservation(Modality.IMAGE, "img-2", 2.0,
                            concept_hint="edge", support=0.98, confidence=0.9),
        ModalityObservation(Modality.IMAGE, "img-3", 2.1,
                            concept_hint="surface", support=0.30, confidence=0.9),
        ModalityObservation(Modality.AUDIO, "aud-2", 2.2,
                            concept_hint="edge", support=0.10, confidence=0.5),
        ModalityObservation(Modality.AUDIO, "aud-3", 2.3,
                            concept_hint="surface", support=0.95, confidence=0.5),
    ]
    fused_two = fuse_observations(map_to_semantics(contradiction))
    assert "visual_edge" in fused_two.top().usg_id
    disagreements = [c for c in fused_two.conflicts
                     if c["kind"] == "modality_disagreement"]
    assert disagreements
    assert disagreements[0]["modality"] == "audio"
    assert "visual_surface" in disagreements[0]["preferred"]
    # the disagreeing audio observation remains visible in the result
    assert disagreements[0]["retained_observations"]
    # every original observation stays attached to its concept
    retained_digests = {obs.payload_digest for concept in fused_two.ranked
                        for obs in concept.observations}
    assert len(retained_digests) == 4


def test_semantic_mapping_preserves_ambiguity_as_explicit_unknown():
    """Step 21: unknown concepts map to the explicit unknown id, never coerced."""
    from ore_learning.adapters import SemanticMapper, UNKNOWN_USG_CONCEPT

    mapper = SemanticMapper()
    known_id, known = mapper.map("edge")
    unknown_id, unknown = mapper.map("unheard-of-hint")
    assert known and known_id.startswith("usg:")
    assert not unknown and unknown_id == UNKNOWN_USG_CONCEPT
    observations = [ModalityObservation(Modality.TEXT, "t", 1.0,
                                        concept_hint="unheard-of-hint")]
    mapped = map_to_semantics(observations, mapper)
    assert mapped[0].concept_hint == UNKNOWN_USG_CONCEPT
    fused = fuse_observations(mapped)
    assert fused.ranked[0].known_concept is False
