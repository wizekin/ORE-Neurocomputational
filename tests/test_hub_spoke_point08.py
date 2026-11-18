"""Acceptance tests for the 15-step Point 08 distributed semantic implementation."""
from __future__ import annotations

from pathlib import Path

import pytest

from ore_logger import LoggerDatabase
from ore_hub_spoke import (
    HubAndSpokeConfig,
    HubAndSpokeRequest,
    LearningExperience,
    POINT08_STAGE_REGISTRY,
    Point08Config,
    Point08DistributedSemanticProcessor,
    TemporalAssociationMemory,
    index_point08_annotations,
)
from ore_semantic.contracts import SemanticConcept


def _concepts() -> list[SemanticConcept]:
    """Create grounded overlapping bell/phone concepts for multimodal retrieval tests."""
    return [
        SemanticConcept(
            concept_id="c-bell",
            label="bell",
            aliases=("handbell",),
            semantic_features={"ARTIFACT": 1.0, "SIGNAL_DEVICE": 0.95, "OBJECT": 0.6},
            modality_features={
                "visual": {"DOME_SHAPE": 0.95, "METAL_SURFACE": 0.8},
                "auditory": {"RINGING": 1.0, "RESONANT_TONE": 0.9},
                "tactile": {"COOL_METAL": 0.75, "SOLID_WEIGHT": 0.6},
                "spatial_motor": {"STRIKE_WITH_HAND": 0.9, "HOLD_HANDLE": 0.7},
            },
            lexical_category="noun",
            frequency=8.0,
            verified_by="test",
            source_reference="test://bell",
        ),
        SemanticConcept(
            concept_id="c-phone",
            label="phone",
            aliases=("telephone",),
            semantic_features={"ARTIFACT": 1.0, "SIGNAL_DEVICE": 0.85, "OBJECT": 0.6},
            modality_features={
                "visual": {"RECTANGULAR_SCREEN": 0.95, "METAL_SURFACE": 0.35},
                "auditory": {"RINGTONE": 0.85, "VOICE_SOUND": 0.8},
                "tactile": {"SMOOTH_SURFACE": 0.85},
                "spatial_motor": {"TAP_SCREEN": 0.95, "HOLD_IN_HAND": 0.8},
            },
            lexical_category="noun",
            frequency=10.0,
            verified_by="test",
            source_reference="test://phone",
        ),
    ]


def _experience(
    episode_id: str = "bell-episode-1",
    timestamps: dict[str, float] | None = None,
) -> LearningExperience:
    """Build a sourced visual/auditory/action event with explicit millisecond timing."""
    return LearningExperience(
        concept_id="c-bell",
        modality_features={
            "visual": {"DOME_SHAPE": 0.95, "METAL_SURFACE": 0.8},
            "auditory": {"RINGING": 1.0, "RESONANT_TONE": 0.9},
            "spatial_motor": {"STRIKE_WITH_HAND": 0.9},
        },
        semantic_features={"ARTIFACT": 1.0, "SIGNAL_DEVICE": 0.95},
        confidence=0.95,
        verified_by="research_dataset",
        source_reference="dataset://bell/multimodal-episode-1",
        episode_id=episode_id,
        modality_timestamps_ms=timestamps or {
            "visual": 1000.0, "auditory": 1075.0, "spatial_motor": 1100.0,
        },
    )


def _processor(tmp_path: Path | None = None, biology: bool = True):
    """Construct a deterministic Point 08 processor with small test populations."""
    database_path = tmp_path / "point08.sqlite3" if tmp_path else None
    return Point08DistributedSemanticProcessor(
        _concepts(),
        hub_config=HubAndSpokeConfig(
            hidden_dim=12, recurrent_steps=4, steps_per_stage=1,
            population_size=4, simulate_biology=biology, random_seed=808,
        ),
        point08_config=Point08Config(
            temporal_window_ms=500.0, eligibility_tau_ms=200.0,
            association_learning_rate=0.2, recall_gain=0.8,
            minimum_recall_activation=0.01,
        ),
        database_path=database_path,
        repository_root=Path(__file__).resolve().parents[1],
    )


def test_registry_has_exactly_fifteen_source_ordered_steps():
    """Expose all fifteen Library algorithm stages in the documented order."""
    assert len(POINT08_STAGE_REGISTRY) == 15
    assert [item.stage_id for item in POINT08_STAGE_REGISTRY] == [
        f"P08-{index:02d}" for index in range(1, 16)
    ]
    assert all(item.mathematical_form and item.biological_reference for item in POINT08_STAGE_REGISTRY)


def test_temporal_eligibility_uses_timestamps_and_feature_strengths():
    """Close cross-modal events form stronger eligible links than distant events."""
    memory = TemporalAssociationMemory(Point08Config(
        temporal_window_ms=500.0, eligibility_tau_ms=100.0,
    ))
    close = memory.preview([_experience(timestamps={
        "visual": 10.0, "auditory": 30.0, "spatial_motor": 50.0,
    })])
    distant = memory.preview([_experience("distant", timestamps={
        "visual": 10.0, "auditory": 5000.0, "spatial_motor": 9000.0,
    })])
    assert close["candidate_pair_count"] > 0
    assert close["eligible_event_count"] > 0
    assert distant["candidate_pair_count"] == 0
    assert close["candidate_pairs"][0]["eligibility_increment"] > 0.0
    assert memory.snapshot()["pair_count"] == 0


def test_untimed_experiences_do_not_fabricate_temporal_cooccurrence():
    """Missing modality timestamps remain unknown rather than being treated as synchronous."""
    experience = LearningExperience(
        concept_id="c-bell",
        modality_features={"visual": {"DOME_SHAPE": 0.9},
                           "auditory": {"RINGING": 0.9}},
        verified_by="research_dataset",
        source_reference="dataset://bell/untimed",
        episode_id="untimed-bell",
    )
    memory = TemporalAssociationMemory()
    preview = memory.preview([experience])
    assert preview["candidate_pair_count"] == 0
    assert preview["untimed_episode_ids"] == ["untimed-bell"]
    assert memory.learn([experience])["status"] == "no_temporally_supported_pairs"
    assert memory.snapshot()["pair_count"] == 0


def test_learning_is_preview_only_without_explicit_authorization():
    """Preview both mapping and temporal plasticity without mutating persistent memories."""
    processor = _processor()
    experience = _experience()
    result = processor.process(
        HubAndSpokeRequest(
            sequence_id="point08-preview", text="bell",
            modality_inputs={"visual": {"DOME_SHAPE": 0.9}},
        ),
        learning_experiences=[experience],
        learning_authorized=False,
    )
    assert result.diagnostics["valid"] is True
    assert result.temporal_associations["pair_count"] == 0
    assert result.diagnostics["temporal_learning_mutated"] is False
    assert result.stage_trace[4]["functional_output"]["temporal_eligibility_preview"][
        "candidate_pair_count"
    ] > 0
    assert result.stage_trace[5]["status"] == "preview_only_authorization_required"
    assert processor.core.model_generation == 0


def test_authorized_learning_commits_temporal_and_biological_associations():
    """Commit timestamped cross-modal links and their separate biological plasticity reference."""
    processor = _processor()
    result = processor.process(
        HubAndSpokeRequest(
            sequence_id="point08-learn", text="bell",
            modality_inputs={"visual": {"DOME_SHAPE": 0.95}},
        ),
        learning_experiences=[_experience()],
        learning_authorized=True,
    )
    assert result.temporal_associations["pair_count"] > 0
    assert result.biological_plasticity["update_count"] == 1
    assert result.diagnostics["temporal_learning_mutated"] is True
    assert result.hub_spoke_result["learning"]["stage_18"]["status"] == "committed"
    assert result.stage_trace[5]["functional_output"]["cross_modal_temporal_links"][
        "updated_pair_count"
    ] > 0


def test_visual_cue_reactivates_learned_auditory_and_action_features():
    """Use prior timestamped co-occurrence to reinstate auditory/action spokes from visual input."""
    processor = _processor(biology=False)
    processor.process(
        HubAndSpokeRequest(
            sequence_id="point08-learn", text="bell",
            modality_inputs={"visual": {"DOME_SHAPE": 0.95}},
        ),
        learning_experiences=[_experience()],
        learning_authorized=True,
    )
    result = processor.process(HubAndSpokeRequest(
        sequence_id="point08-visual-query",
        modality_inputs={"visual": {"DOME_SHAPE": 0.95}},
        task="describe",
    ))
    recalled = result.cross_modal_recall["automatically_recalled_features"]
    assert "auditory" in recalled
    assert "RINGING" in recalled["auditory"]
    assert "spatial_motor" in recalled
    source_step = result.stage_trace[0]["functional_output"]
    assert source_step["direct_modalities"] == ["visual"]
    assert set(source_step["recalled_modalities"]) >= {"auditory", "spatial_motor"}
    core_memory = result.hub_spoke_result["stage_outputs"]["HS-04"]["combined_spoke_magnitudes"]
    assert core_memory["auditory"] > 0.0
    assert "auditory" not in result.hub_spoke_result["stage_outputs"]["HS-06"]["direct_input_modalities"]


def test_direct_and_recalled_modality_evidence_remain_distinguishable_in_core_request():
    """The underlying 21-stage processor tracks recalled cues separately from direct input."""
    processor = _processor(biology=False)
    result = processor.process(HubAndSpokeRequest(
        sequence_id="separate-provenance",
        modality_inputs={"visual": {"DOME_SHAPE": 0.9}},
        recalled_modality_inputs={"auditory": {"RINGING": 0.6}},
    ))
    first = result.hub_spoke_result["stage_outputs"]["HS-01"]
    assert first["direct_modalities"] == ["visual"]
    assert first["recalled_modalities"] == ["auditory"]
    assert first["recalled_features_not_treated_as_direct_observations"] is True
    assert result.cross_modal_recall["direct_evidence_unchanged"] == {
        "visual": {"DOME_SHAPE": 0.9}
    }


def test_learning_contract_rejects_invalid_or_unmatched_modality_timestamps():
    """Require finite non-negative times that refer to modalities with observed features."""
    with pytest.raises(ValueError, match="timestamps must be finite"):
        LearningExperience(
            concept_id="c-bell", modality_features={"visual": {"DOME_SHAPE": 0.9}},
            modality_timestamps_ms={"visual": float("nan")},
        )
    with pytest.raises(ValueError, match="must refer to a modality"):
        LearningExperience(
            concept_id="c-bell", modality_features={"visual": {"DOME_SHAPE": 0.9}},
            modality_timestamps_ms={"auditory": 20.0},
        )


def test_fifteen_source_steps_keep_underlying_functional_and_lif_traces():
    """Retain the exact fifteen-step source trace and all 21 detailed core stages."""
    result = _processor().process(HubAndSpokeRequest(
        sequence_id="point08-full-run", text="bell",
        modality_inputs={
            "visual": {"DOME_SHAPE": 0.9, "METAL_SURFACE": 0.8},
            "auditory": {"RINGING": 0.9},
        },
        context_features={"SIGNAL_DEVICE": 0.9},
        task="describe",
    ))
    assert result.diagnostics["valid"] is True
    assert len(result.stage_trace) == 15
    assert len(result.hub_spoke_result["stage_trace"]) == 21
    assert result.hub_spoke_result["biological"]["enabled"] is True
    assert result.hub_spoke_result["biological"]["stage_count"] == 21
    assert result.stage_trace[6]["biological_output"]["lif_enabled"] is True
    assert result.stage_trace[13]["functional_output"]["type"] in {
        "description", "concept_identification", "abstention"
    }


def test_semantic_control_and_context_are_explicit_in_source_trace():
    """Carry the task, context, and relevance constraints into selection and interpretation stages."""
    result = _processor(biology=False).process(HubAndSpokeRequest(
        sequence_id="point08-context",
        text="bell",
        modality_inputs={"visual": {"DOME_SHAPE": 0.9}},
        context_features={"SIGNAL_DEVICE": 0.95},
        task="describe",
        task_relevance={"RINGING": 0.9, "SIGNAL_DEVICE": 0.8},
        irrelevant_features={"OBJECT": 0.6},
    ))
    assert result.stage_trace[11]["functional_output"]["task"] == "describe"
    assert result.stage_trace[12]["functional_output"]["context_features"]["SIGNAL_DEVICE"] == 0.95
    assert "task_relevance" in result.stage_trace[12]["functional_output"]


def test_sqlite_indexes_annotations_and_records_each_point08_operation(tmp_path: Path):
    """Persist Point 08 classes/functions and one annotated SQLite operation per source step."""
    processor = _processor(tmp_path=tmp_path, biology=False)
    result = processor.process(
        HubAndSpokeRequest(
            sequence_id="point08-sqlite",
            text="bell",
            modality_inputs={"visual": {"DOME_SHAPE": 0.9}},
        ),
        learning_experiences=[_experience()],
        learning_authorized=True,
    )
    assert result.logger["database_path"]
    database = LoggerDatabase(result.logger["database_path"])
    assert database.get_annotation(
        "ore_hub_spoke.point08", "Point08DistributedSemanticProcessor", "class"
    )
    assert database.get_annotation(
        "ore_hub_spoke.point08", "Point08DistributedSemanticProcessor.process", "function"
    )
    assert database.get_annotation(
        "ore_hub_spoke.point08", "TemporalAssociationMemory.learn", "function"
    )
    runs = database.list_runs()
    assert len(runs) == 1 and runs[0]["status"] == "completed"
    events = database.list_events(runs[0]["run_id"])
    operations = [event for event in events if event["event_kind"] == "operation"]
    assert len(operations) == 15
    assert all(event.get("annotation") for event in operations)


def test_point08_annotation_indexer_registers_the_extended_hub_spoke_package(tmp_path: Path):
    """Index source docstrings for Point 08 additions in the established ORE SQLite catalogue."""
    root = Path(__file__).resolve().parents[1]
    database_path = tmp_path / "point08-annotations.sqlite3"
    report = index_point08_annotations(root, database_path)
    assert report["parse_errors"] == 0
    assert report["point08_symbol_count"] > 50
    database = LoggerDatabase(database_path)
    assert database.get_annotation(
        "ore_hub_spoke.point08", "BiologicalEligibilityPlasticity", "class"
    )
