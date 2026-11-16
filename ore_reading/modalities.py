"""Auditory input, reading-unit attention, and speech-output adapters.

The auditory path extracts acoustic features and proposes timing-based speech
segments. It does not perform ASR, infer phonemes, or fabricate a transcript.
Speech output is prepared for a pluggable synthesis backend; no voice is invented
when a backend has not been supplied.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable
import math

import numpy as np


SOURCE_OPERATION_IMPLEMENTATIONS = (
    {
        "source_operation_id": "IRA-OP-039",
        "name": "Extract Temporal Acoustic Information",
        "implementation": "AuditoryInputAdapter.frame_waveform",
        "status": "implemented_feature_extraction",
    },
    {
        "source_operation_id": "IRA-OP-046",
        "name": "Generate Continuous-Speech Segmentation Hypotheses",
        "implementation": "AuditoryInputAdapter.segment_continuous_speech",
        "status": "implemented_heuristic_segmentation",
    },
    {
        "source_operation_id": "IRA-OP-060",
        "name": "Produce Overt Speech or Silent Reading Output",
        "implementation": "SpeechOutputAdapter.dispatch",
        "status": "backend_interface_and_silent_output",
    },
    {
        "source_operation_id": "IRA-OP-061",
        "name": "Direct Attention to the Next Reading Unit",
        "implementation": "ReadingAttentionController.select_next_unit",
        "status": "implemented_discrete_attention_control",
    },
)


@dataclass(frozen=True, slots=True)
class AuditoryConfig:
    """Control acoustic framing and conservative energy-based speech segmentation."""

    frame_duration_ms: float = 25.0
    hop_duration_ms: float = 10.0
    minimum_energy: float = 1e-4
    relative_energy_threshold: float = 0.18
    minimum_speech_frames: int = 3
    maximum_silence_frames: int = 8

    def __post_init__(self) -> None:
        """Reject invalid temporal and energy thresholds."""
        if not math.isfinite(self.frame_duration_ms) or self.frame_duration_ms <= 0:
            raise ValueError("frame_duration_ms must be finite and positive")
        if not math.isfinite(self.hop_duration_ms) or self.hop_duration_ms <= 0:
            raise ValueError("hop_duration_ms must be finite and positive")
        if self.hop_duration_ms > self.frame_duration_ms:
            raise ValueError("hop_duration_ms cannot exceed frame_duration_ms")
        if not 0.0 <= self.relative_energy_threshold <= 1.0:
            raise ValueError("relative_energy_threshold must lie in [0, 1]")
        if self.minimum_energy < 0 or self.minimum_speech_frames < 1 or self.maximum_silence_frames < 0:
            raise ValueError("energy and frame-count thresholds are invalid")


@dataclass(slots=True)
class AuditoryInputResult:
    """Frame features and uncommitted continuous-speech segmentation hypotheses."""

    sequence_id: str
    sample_rate_hz: int
    frames: list[dict[str, Any]]
    segments: list[dict[str, Any]]
    operation_trace: list[dict[str, Any]]
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize auditory features, proposed segment intervals and trace rows."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AttentionPlan:
    """A discrete next-unit selection, not an eye-tracking or saccade measurement."""

    next_index: int | None
    token: str | None
    priority_score: float
    suggested_dwell_ms: int
    candidate_scores: tuple[Mapping[str, Any], ...]
    source_operation_id: str = "IRA-OP-061"
    control_scope: str = "discrete_reading_unit_selection_not_eye_tracking"


@runtime_checkable
class SpeechSynthesisBackend(Protocol):
    """Contract for an external text-to-speech or phoneme-to-speech implementation."""

    def synthesize(self, phonemes: Sequence[str], *, language: str) -> Any:
        """Synthesize supplied phonological units using an explicitly configured backend."""
        ...


@dataclass(slots=True)
class SpeechOutputPlan:
    """Prepared speech or silent-reading output with an explicit backend status."""

    mode: str
    text: str
    phonemes: list[str]
    language: str
    status: str
    backend_result: Any = None
    source_operation_id: str = "IRA-OP-060"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly speech-output plan."""
        return asdict(self)


class AuditoryInputAdapter:
    """Extract overlapping acoustic frames and conservative segmentation hypotheses."""

    def __init__(self, config: AuditoryConfig | None = None) -> None:
        """Initialize frame dimensions and segmentation thresholds."""
        self.config = config or AuditoryConfig()

    def frame_waveform(
        self,
        waveform: Sequence[float] | np.ndarray,
        *,
        sample_rate_hz: int,
        sequence_id: str = "auditory-sequence",
    ) -> AuditoryInputResult:
        """Compute RMS, zero-crossing rate and spectral centroid over overlapping frames."""
        signal = np.asarray(waveform, dtype=np.float32)
        if signal.ndim != 1 or signal.size == 0:
            raise ValueError("waveform must be a non-empty one-dimensional signal")
        if not np.isfinite(signal).all():
            raise ValueError("waveform must contain only finite samples")
        if not isinstance(sample_rate_hz, int) or sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be a positive integer")
        if not sequence_id.strip():
            raise ValueError("sequence_id must be non-empty")

        frame_size = max(1, int(round(sample_rate_hz * self.config.frame_duration_ms / 1000.0)))
        hop_size = max(1, int(round(sample_rate_hz * self.config.hop_duration_ms / 1000.0)))
        starts = list(range(0, max(1, signal.size - frame_size + 1), hop_size))
        if not starts or starts[-1] + frame_size < signal.size:
            starts.append(max(0, signal.size - frame_size))
        if signal.size < frame_size:
            starts = [0]
        window = np.hanning(frame_size).astype(np.float32)
        frequency_bins = np.fft.rfftfreq(frame_size, d=1.0 / sample_rate_hz)
        frames: list[dict[str, Any]] = []
        for frame_index, start in enumerate(starts):
            frame = signal[start:start + frame_size]
            if frame.size < frame_size:
                frame = np.pad(frame, (0, frame_size - frame.size))
            rms = float(np.sqrt(np.mean(np.square(frame), dtype=np.float64)))
            signs = np.signbit(frame)
            zero_crossing_rate = float(np.mean(signs[1:] != signs[:-1])) if frame_size > 1 else 0.0
            spectrum = np.abs(np.fft.rfft(frame * window)).astype(np.float64)
            spectral_mass = float(spectrum.sum())
            centroid = (
                float(np.dot(frequency_bins, spectrum) / spectral_mass)
                if spectral_mass > 0.0 else 0.0
            )
            frames.append({
                "frame_index": frame_index,
                "start_sample": int(start),
                "end_sample": int(min(start + frame_size, signal.size)),
                "start_seconds": float(start / sample_rate_hz),
                "end_seconds": float(min(start + frame_size, signal.size) / sample_rate_hz),
                "rms_energy": rms,
                "zero_crossing_rate": zero_crossing_rate,
                "spectral_centroid_hz": centroid,
            })

        operation_trace = [{
            "source_operation_id": "IRA-OP-039",
            "status": "executed",
            "inputs_observed": {
                "sample_count": int(signal.size), "sample_rate_hz": sample_rate_hz,
                "frame_size_samples": frame_size, "hop_size_samples": hop_size,
            },
            "outputs_produced": {
                "frame_count": len(frames),
                "feature_names": ["rms_energy", "zero_crossing_rate", "spectral_centroid_hz"],
                "first_frame": dict(frames[0]),
                "last_frame": dict(frames[-1]),
            },
            "implementation": "overlapping waveform frames; Hann-window magnitude spectrum",
        }]
        return AuditoryInputResult(
            sequence_id=sequence_id,
            sample_rate_hz=sample_rate_hz,
            frames=frames,
            segments=[],
            operation_trace=operation_trace,
            diagnostics={
                "valid": bool(frames and all(math.isfinite(row["spectral_centroid_hz"]) for row in frames)),
                "frame_count": len(frames),
                "frame_duration_ms": self.config.frame_duration_ms,
                "hop_duration_ms": self.config.hop_duration_ms,
                "asr_performed": False,
                "phonemes_inferred": False,
                "scope": "acoustic feature extraction only",
            },
        )

    def segment_continuous_speech(self, result: AuditoryInputResult) -> AuditoryInputResult:
        """Create time intervals using adaptive energy threshold and a bounded silence gap."""
        if not isinstance(result, AuditoryInputResult):
            raise TypeError("result must be an AuditoryInputResult")
        if not result.frames:
            raise ValueError("auditory result has no frames to segment")
        energies = np.asarray([row["rms_energy"] for row in result.frames], dtype=np.float64)
        noise_floor = float(np.quantile(energies, 0.20))
        upper_energy = float(np.quantile(energies, 0.90))
        threshold = max(
            self.config.minimum_energy,
            noise_floor + self.config.relative_energy_threshold * max(0.0, upper_energy - noise_floor),
        )
        active = energies >= threshold
        spans: list[tuple[int, int]] = []
        start: int | None = None
        last_active: int | None = None
        silence_run = 0
        for index, is_active in enumerate(active.tolist()):
            if is_active:
                if start is None:
                    start = index
                last_active = index
                silence_run = 0
            elif start is not None:
                silence_run += 1
                if silence_run > self.config.maximum_silence_frames:
                    assert last_active is not None
                    spans.append((start, last_active + 1))
                    start, last_active, silence_run = None, None, 0
        if start is not None and last_active is not None:
            spans.append((start, last_active + 1))

        segments: list[dict[str, Any]] = []
        for segment_index, (start_frame, end_frame) in enumerate(spans):
            if end_frame - start_frame < self.config.minimum_speech_frames:
                continue
            selected = result.frames[start_frame:end_frame]
            rms_values = [float(row["rms_energy"]) for row in selected]
            segments.append({
                "segment_index": segment_index,
                "start_frame": start_frame,
                "end_frame_exclusive": end_frame,
                "start_seconds": float(selected[0]["start_seconds"]),
                "end_seconds": float(selected[-1]["end_seconds"]),
                "duration_seconds": float(selected[-1]["end_seconds"] - selected[0]["start_seconds"]),
                "mean_rms_energy": float(np.mean(rms_values)),
                "peak_rms_energy": float(np.max(rms_values)),
                "mean_zero_crossing_rate": float(np.mean([
                    row["zero_crossing_rate"] for row in selected
                ])),
                "mean_spectral_centroid_hz": float(np.mean([
                    row["spectral_centroid_hz"] for row in selected
                ])),
                "frame_count": len(selected),
                "energy_threshold": threshold,
                "confidence_proxy": float(np.clip(np.mean(rms_values) / max(upper_energy, 1e-12), 0.0, 1.0)),
                "hypothesis_status": "time_segment_only_not_word_recognition",
            })
        result.segments = segments
        result.operation_trace.append({
            "source_operation_id": "IRA-OP-046",
            "status": "executed",
            "inputs_observed": {
                "frame_count": len(result.frames),
                "adaptive_energy_threshold": threshold,
                "maximum_silence_frames": self.config.maximum_silence_frames,
            },
            "outputs_produced": {
                "segment_count": len(segments),
                "segments": segments,
            },
            "implementation": "adaptive RMS threshold with silence-gap hysteresis",
        })
        result.diagnostics.update({
            "speech_segments_proposed": len(segments),
            "adaptive_energy_threshold": threshold,
            "continuous_speech_segmentation": "heuristic",
            "asr_performed": False,
            "phoneme_or_word_boundaries_verified": False,
        })
        return result


class ReadingAttentionController:
    """Select the next reading unit using lexical uncertainty and explicit priorities."""

    def select_next_unit(
        self,
        reading_result: Any,
        *,
        current_index: int = -1,
        priorities: Mapping[int, float] | None = None,
        lookahead: int = 4,
        base_dwell_ms: int = 180,
    ) -> AttentionPlan:
        """Choose the highest-priority upcoming token without claiming eye-movement control."""
        tokens = getattr(reading_result, "tokens", None)
        if tokens is None and isinstance(reading_result, Mapping):
            tokens = reading_result.get("tokens")
        if not isinstance(tokens, Sequence):
            raise TypeError("reading_result must expose an ordered tokens sequence")
        if lookahead < 1 or base_dwell_ms < 1:
            raise ValueError("lookahead and base_dwell_ms must be positive")
        priority_map = dict(priorities or {})
        candidates: list[dict[str, Any]] = []
        start = max(0, current_index + 1)
        for index in range(start, min(len(tokens), start + lookahead)):
            token = tokens[index]
            surface = str(token.get("orthography", token.get("surface", "")))
            entropy = float(token.get("posterior_entropy", token.get("ambiguity", 0.0)) or 0.0)
            confidence = float(token.get("pronunciation_confidence", 0.5) or 0.0)
            margin = float(token.get("posterior_margin", 0.0) or 0.0)
            abstained = bool(token.get("lexical_abstain", token.get("status") == "ABSTAIN"))
            explicit_priority = float(priority_map.get(index, 0.0))
            values = (entropy, confidence, margin, explicit_priority)
            if any(not math.isfinite(value) for value in values):
                raise ValueError("attention evidence must be finite")
            if not 0.0 <= explicit_priority <= 1.0:
                raise ValueError("explicit priorities must lie in [0, 1]")
            score = float(np.clip(
                0.40 * min(1.0, max(0.0, entropy))
                + 0.25 * (1.0 - min(1.0, max(0.0, confidence)))
                + 0.20 * float(abstained)
                + 0.15 * explicit_priority
                + 0.05 * (1.0 - min(1.0, max(0.0, margin))),
                0.0, 1.0,
            ))
            candidates.append({"index": index, "token": surface, "priority_score": score})
        if not candidates:
            return AttentionPlan(None, None, 0.0, 0, ())
        # Sequential order wins ties; the controller may prioritize a later uncertain unit.
        selected = max(candidates, key=lambda row: (row["priority_score"], -row["index"]))
        dwell = int(round(base_dwell_ms * (1.0 + selected["priority_score"])))
        return AttentionPlan(
            next_index=int(selected["index"]),
            token=str(selected["token"]),
            priority_score=float(selected["priority_score"]),
            suggested_dwell_ms=dwell,
            candidate_scores=tuple(candidates),
        )


class SpeechOutputAdapter:
    """Prepare pronunciation output and optionally dispatch it to a supplied synthesis backend."""

    def prepare(
        self,
        reading_result: Any,
        *,
        language: str = "en",
        mode: str = "overt",
    ) -> SpeechOutputPlan:
        """Collect existing pronunciation hypotheses without generating missing phonemes."""
        if mode not in {"overt", "silent"}:
            raise ValueError("mode must be 'overt' or 'silent'")
        tokens = getattr(reading_result, "tokens", None)
        if tokens is None and isinstance(reading_result, Mapping):
            tokens = reading_result.get("tokens", [])
        if not isinstance(tokens, Sequence):
            raise TypeError("reading_result must expose token records")
        decoded = getattr(reading_result, "decoded_text", None)
        if decoded is None and isinstance(reading_result, Mapping):
            decoded = reading_result.get("decoded_text", reading_result.get("normalized_text", ""))
        phonemes: list[str] = []
        for token in tokens:
            phones = token.get("pronunciation", ())
            phonemes.extend(str(phone) for phone in phones)
        if mode == "silent":
            return SpeechOutputPlan(
                mode="silent", text=str(decoded or ""), phonemes=phonemes, language=language,
                status="silent_output_ready",
            )
        status = "prepared_backend_required" if phonemes else "unavailable_no_pronunciation"
        return SpeechOutputPlan(
            mode="overt", text=str(decoded or ""), phonemes=phonemes,
            language=language, status=status,
        )

    def dispatch(
        self,
        plan: SpeechOutputPlan,
        *,
        backend: SpeechSynthesisBackend | None = None,
    ) -> SpeechOutputPlan:
        """Dispatch prepared phonemes only when a compatible external speech backend exists."""
        if not isinstance(plan, SpeechOutputPlan):
            raise TypeError("plan must be a SpeechOutputPlan")
        if plan.mode == "silent":
            return plan
        if backend is None:
            plan.status = "prepared_backend_required" if plan.phonemes else "unavailable_no_pronunciation"
            return plan
        if not callable(getattr(backend, "synthesize", None)):
            raise TypeError("backend must expose synthesize(phonemes, language=...)")
        if not plan.phonemes:
            plan.status = "unavailable_no_pronunciation"
            return plan
        plan.backend_result = backend.synthesize(plan.phonemes, language=plan.language)
        plan.status = "dispatched_to_backend"
        return plan
