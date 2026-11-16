"""Focused acceptance tests for ORE's integrated six-stage sensory-perception implementation."""
from __future__ import annotations

import ast
from pathlib import Path
import shutil

import numpy as np
import pytest

try:
    from ore_logger import LoggerDatabase
except ImportError:
    LoggerDatabase = None

from ore_sensory_perception import (
    OPERATION_REGISTRY, STAGE_STEP_COUNTS, PhonemeSpec, SensoryConfig,
    SensoryPerceptionProcessor, SensoryRequest, register_sensory_annotations,
)


def _page_fixture() -> np.ndarray:
    """Construct a deterministic high-contrast pseudo-page with repeated glyph-like strokes."""
    page = np.full((88, 288), 0.94, dtype=np.float64)
    for row, start, stop in ((16, 10, 258), (40, 18, 250), (64, 12, 270)):
        page[row:row+5, start:stop:9] = 0.06
        page[row-4:row+8:4, start+3:stop:18] = 0.12
    return page


def _audio_fixture(duration_seconds: float = 0.32, sample_rate_hz: float = 16000.0) -> np.ndarray:
    """Build a reproducible harmonic waveform with a time-varying amplitude envelope."""
    time = np.arange(int(duration_seconds*sample_rate_hz), dtype=np.float64)/sample_rate_hz
    envelope = np.clip(time/0.03, 0, 1) * np.clip((duration_seconds-time)/0.03, 0, 1)
    return envelope * (0.42*np.sin(2*np.pi*180*time) + 0.13*np.sin(2*np.pi*540*time))


def _processor(simulate_biology: bool = False) -> SensoryPerceptionProcessor:
    """Create a deterministic processor without changing the live repository annotation database."""
    return SensoryPerceptionProcessor(config=SensoryConfig(simulate_biology=simulate_biology),
                                      register_annotations=False)


def test_registry_has_exact_39_ordered_operations_and_stage_counts() -> None:
    """Check the specified six stage lengths and stable operation identifiers."""
    assert len(OPERATION_REGISTRY) == 39
    assert STAGE_STEP_COUNTS == {1: 6, 2: 6, 3: 6, 4: 6, 5: 7, 6: 8}
    assert [x.operation_id for x in OPERATION_REGISTRY] == [f"ISP-{i:03d}" for i in range(1, 40)]
    for stage, count in STAGE_STEP_COUNTS.items():
        assert [x.step for x in OPERATION_REGISTRY if x.stage == stage] == list(range(1, count+1))


def test_combined_path_runs_39_functional_and_biological_updates() -> None:
    """Exercise both modalities and confirm distinct functional and simulated neural traces."""
    result = _processor(True).process(SensoryRequest(
        sample_id="combined-001", visual_image=_page_fixture(),
        auditory_waveform=_audio_fixture(), sample_rate_hz=16000.0,
        phoneme_inventory=(PhonemeSpec("p","stop","bilabial",False),
                           PhonemeSpec("b","stop","bilabial",True),
                           PhonemeSpec("m","nasal","bilabial",True)),
    ))
    assert len(result.operation_trace) == len(result.biological_trace) == 39
    assert result.operation_trace[0]["operation_id"] == "ISP-001"
    assert result.operation_trace[-1]["operation_id"] == "ISP-039"
    assert result.integration["visual_available"] and result.integration["auditory_available"]
    assert result.biological_trace[-1]["biological_status"] == "simulated_reference_not_measured_brain_data"
    posterior = np.asarray(result.phoneme_output["candidate_probabilities"])
    assert posterior.shape[1] == 3 and np.all(np.isfinite(posterior))
    assert np.allclose(np.sum(posterior, axis=1), 1.0, atol=1e-6)


def test_visual_only_run_skips_audio_without_fabricating_a_waveform() -> None:
    """Confirm independent visual processing and explicit skipped auditory operations."""
    result = _processor().process(SensoryRequest(sample_id="visual-only", visual_image=_page_fixture()))
    assert result.integration["visual_available"] is True
    assert result.integration["auditory_available"] is False
    assert result.integration["text_region_count"] >= 1
    assert result.integration["character_hypothesis_count"] >= 1
    assert all(x["status"] == "skipped_no_auditory_input" for x in result.operation_trace[12:])
    assert result.phoneme_output == {}


def test_audio_only_run_keeps_acoustic_phonetic_and_candidate_frames_aligned() -> None:
    """Validate time alignment between acoustic vectors, population codes and phoneme posteriors."""
    result = _processor().process(SensoryRequest(
        sample_id="audio-only", auditory_waveform=_audio_fixture(0.48), sample_rate_hz=16000.0,
        phonetic_context={"b": 1.2, "p": 0.8},
    ))
    assert result.integration["auditory_available"] and not result.integration["visual_available"]
    assert all(x["status"] == "skipped_no_visual_input" for x in result.operation_trace[:12])
    acoustic = np.asarray(result.auditory_output["acoustic_feature_matrix"])
    phonetic = np.asarray(result.phonetic_output["phonetic_population_code"])
    posterior = np.asarray(result.phoneme_output["candidate_probabilities"])
    assert acoustic.ndim == 2 and acoustic.shape[0] > 1
    assert phonetic.shape[0] == posterior.shape[0] == acoustic.shape[0]
    assert np.all(np.isfinite(acoustic))
    assert np.allclose(np.sum(posterior, axis=1), 1.0, atol=1e-6)


@pytest.mark.parametrize("sample_count", [32, 160, 512])
def test_short_audio_windows_preserve_time_alignment(sample_count: int) -> None:
    """Ensure STFT, periodicity, formants and candidate decoders handle the shortest valid input."""
    rate = 16000.0
    time = np.arange(sample_count)/rate
    result = _processor().process(SensoryRequest(
        sample_id=f"short-{sample_count}", auditory_waveform=np.sin(2*np.pi*180*time),
        sample_rate_hz=rate,
    ))
    assert len(result.operation_trace) == 39
    matrix = np.asarray(result.auditory_output["acoustic_feature_matrix"])
    posterior = np.asarray(result.phoneme_output["candidate_probabilities"])
    assert matrix.shape[0] == posterior.shape[0]


def test_invalid_requests_are_rejected() -> None:
    """Reject missing modalities, sample-rate omissions and non-finite visual input."""
    with pytest.raises(ValueError):
        SensoryRequest(sample_id="empty")
    with pytest.raises(ValueError):
        SensoryRequest(sample_id="bad-audio", auditory_waveform=np.ones(64))
    with pytest.raises(ValueError):
        SensoryRequest(sample_id="bad-image", visual_image=np.asarray([[np.nan]]))


def test_every_class_and_function_has_a_logger_ready_docstring() -> None:
    """AST-check all package class/function definitions for the logger's docstring annotation source."""
    root = Path(__file__).resolve().parents[1]/"ore_sensory_perception"
    missing = []
    for source in sorted(root.glob("*.py")):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if ast.get_docstring(node) is None:
                    missing.append(f"{source.name}:{node.lineno}:{node.name}")
    assert missing == []


def test_annotations_reach_the_existing_ore_sqlite_catalogue(tmp_path: Path) -> None:
    """Index a temporary copy with the real logger and verify function/class descriptions in SQLite."""
    if LoggerDatabase is None:
        pytest.skip("the complete ORE logger package is unavailable in this standalone test workspace")
    source_root = Path(__file__).resolve().parents[1]
    isolated = tmp_path/"ore-project"
    isolated.mkdir()
    shutil.copytree(source_root/"ore_logger", isolated/"ore_logger")
    shutil.copytree(source_root/"ore_sensory_perception", isolated/"ore_sensory_perception")
    (isolated/"pyproject.toml").write_text("[project]\nname='annotation-test'\n", encoding="utf-8")
    path = tmp_path/"annotations.sqlite3"
    report = register_sensory_annotations(isolated, path)
    rows = LoggerDatabase(path).search_annotations("ore_sensory_perception", limit=1000)
    rows = [row for row in rows if row["module_name"].startswith("ore_sensory_perception")]
    assert report["symbols_written"] > 0
    assert rows and all(row["source_docstring"] for row in rows)
    assert any(row["symbol_kind"] == "class" for row in rows)
    assert any(row["symbol_kind"] == "function" for row in rows)
