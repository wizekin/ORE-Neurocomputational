"""Tests for audio framing, continuous-speech boundaries, reading attention and output."""
from __future__ import annotations

import numpy as np
import pytest

from ore_reading.modalities import (
    AuditoryConfig,
    AuditoryInputAdapter,
    ReadingAttentionController,
    SpeechOutputAdapter,
)


def test_audio_frames_and_continuous_speech_hypotheses_preserve_time() -> None:
    """Extract frame-level acoustic features and identify speech-like intervals."""
    sample_rate = 16_000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    waveform = np.zeros(sample_rate, dtype=np.float32)
    first = (time >= 0.20) & (time <= 0.48)
    second = (time >= 0.72) & (time <= 0.96)
    waveform[first] = 0.35 * np.sin(2.0 * np.pi * 180.0 * time[first])
    waveform[second] = 0.28 * np.sin(2.0 * np.pi * 240.0 * time[second])

    adapter = AuditoryInputAdapter(AuditoryConfig(minimum_speech_frames=3))
    result = adapter.frame_waveform(waveform, sample_rate_hz=sample_rate, sequence_id="audio-1")
    segmented = adapter.segment_continuous_speech(result)

    assert result.diagnostics["asr_performed"] is False
    assert len(result.frames) > 50
    assert all(row["spectral_centroid_hz"] >= 0 for row in result.frames)
    assert [row["source_operation_id"] for row in segmented.operation_trace] == [
        "IRA-OP-039", "IRA-OP-046"
    ]
    assert len(segmented.segments) == 2
    assert segmented.segments[0]["start_seconds"] < segmented.segments[1]["start_seconds"]
    assert segmented.segments[0]["hypothesis_status"] == "time_segment_only_not_word_recognition"


def test_audio_adapter_rejects_invalid_waveform_and_sampling_rate() -> None:
    """Reject malformed acoustic input before extracting numerical features."""
    adapter = AuditoryInputAdapter()
    with pytest.raises(ValueError, match="one-dimensional"):
        adapter.frame_waveform(np.zeros((4, 4)), sample_rate_hz=16_000)
    with pytest.raises(ValueError, match="positive integer"):
        adapter.frame_waveform(np.zeros(100), sample_rate_hz=0)


def test_next_unit_attention_uses_uncertainty_and_keeps_sequence_order_for_ties() -> None:
    """Prioritize an uncertain upcoming word and return a bounded dwell recommendation."""
    result = {
        "tokens": [
            {"surface": "the", "posterior_entropy": 0.05, "pronunciation_confidence": 0.98},
            {"surface": "ambiguous", "posterior_entropy": 0.8, "pronunciation_confidence": 0.45,
             "lexical_abstain": True, "posterior_margin": 0.02},
            {"surface": "word", "posterior_entropy": 0.2, "pronunciation_confidence": 0.9},
        ]
    }
    plan = ReadingAttentionController().select_next_unit(result, current_index=0, lookahead=2)
    assert plan.source_operation_id == "IRA-OP-061"
    assert plan.next_index == 1
    assert plan.token == "ambiguous"
    assert plan.suggested_dwell_ms > 180
    assert plan.control_scope == "discrete_reading_unit_selection_not_eye_tracking"


def test_speech_output_requires_a_backend_and_supports_silent_output() -> None:
    """Prepare supplied pronunciations and dispatch only through an explicit backend."""
    result = {
        "decoded_text": "cat",
        "tokens": [{"orthography": "cat", "pronunciation": ["K", "AE", "T"]}],
    }
    adapter = SpeechOutputAdapter()
    plan = adapter.prepare(result, language="en", mode="overt")
    assert plan.status == "prepared_backend_required"
    assert plan.phonemes == ["K", "AE", "T"]
    assert adapter.dispatch(plan).status == "prepared_backend_required"

    class FakeBackend:
        """Small test backend that records phoneme requests."""

        def synthesize(self, phonemes, *, language):
            """Return a deterministic response proving that dispatch occurred."""
            return {"phones": list(phonemes), "language": language}

    dispatched = adapter.dispatch(plan, backend=FakeBackend())
    assert dispatched.status == "dispatched_to_backend"
    assert dispatched.backend_result["phones"] == ["K", "AE", "T"]
    silent = adapter.prepare(result, mode="silent")
    assert silent.status == "silent_output_ready"
    assert silent.text == "cat"


def test_extension_registry_covers_four_explicit_source_boundaries() -> None:
    """Keep missing source input/output operations discoverable by their original IDs."""
    from ore_reading.modalities import SOURCE_OPERATION_IMPLEMENTATIONS

    assert {row["source_operation_id"] for row in SOURCE_OPERATION_IMPLEMENTATIONS} == {
        "IRA-OP-039", "IRA-OP-046", "IRA-OP-060", "IRA-OP-061"
    }
