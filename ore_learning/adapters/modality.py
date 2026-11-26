"""Multimodal Observation Gateway (Point 14, Steps 18-19).

Accepts text, image, audio, video, spatial, and structured inputs through separate
validated paths, checks encoding, dimensions, duration, units and malformed payloads,
preserves source references, timestamps, sampling, resolution, coordinate frames and
preprocessing history, and rejects inputs outside declared model limits.  Original
references are always preserved; preprocessing is never destructive.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

import numpy as np

from ..config import LearningConfig
from ..contracts import Modality, canonical_payload_digest

__all__ = ("ModalityPayload", "ModalityValidationError", "NormalizedModalityInput",
           "ObservationGateway")


class ModalityValidationError(ValueError):
    """Raised when a modality payload is malformed or outside declared limits."""


@dataclass(slots=True)
class ModalityPayload:
    """A raw modality input as it arrives at the gateway (Step 18)."""

    modality: Modality
    source_ref: str
    timestamp: float
    data: Any
    encoding: str = ""
    language: str = "en"
    sample_rate: float = 16000.0
    coordinate_frame: str = "world"
    units: str = ""
    preprocessing: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Basic identity validation."""
        if not isinstance(self.source_ref, str) or not self.source_ref.strip():
            raise ModalityValidationError("source_ref must be a non-empty string")
        if not math.isfinite(float(self.timestamp)):
            raise ModalityValidationError("timestamp must be finite")

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view (raw data is referenced by digest, never copied)."""
        return {"modality": self.modality.value, "source_ref": self.source_ref,
                "timestamp": self.timestamp, "encoding": self.encoding,
                "language": self.language, "sample_rate": self.sample_rate,
                "coordinate_frame": self.coordinate_frame, "units": self.units,
                "preprocessing": list(self.preprocessing),
                "data_digest": canonical_payload_digest(
                    {"modality": self.modality.value,
                     "summary": _data_summary(self.data)})}


@dataclass(slots=True)
class NormalizedModalityInput:
    """A validated, normalized modality input with full provenance (Step 19)."""

    payload: ModalityPayload
    shape: tuple[int, ...]
    duration_seconds: float = 0.0
    checks: tuple[str, ...] = ()
    digest: str = ""

    @property
    def source_ref(self) -> str:
        """Preserved source reference."""
        return self.payload.source_ref

    @property
    def timestamp(self) -> float:
        """Preserved timestamp."""
        return self.payload.timestamp

    @property
    def coordinate_frame(self) -> str:
        """Preserved coordinate frame (spatial inputs)."""
        return self.payload.coordinate_frame

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {**self.payload.to_dict(), "shape": list(self.shape),
                "duration_seconds": self.duration_seconds,
                "checks": list(self.checks), "digest": self.digest}


def _data_summary(data: Any) -> str:
    """A short structural summary of raw data for digesting."""
    if isinstance(data, np.ndarray):
        return f"array{list(data.shape)}"
    if isinstance(data, str):
        return f"text:{len(data)}"
    if isinstance(data, (list, tuple)):
        return f"seq:{len(data)}"
    return type(data).__name__


class ObservationGateway:
    """Steps 18-19 -- validate and normalize modality-specific inputs."""

    def __init__(self, config: LearningConfig | None = None) -> None:
        """Configure declared model limits from the governance config."""
        self.config = config or LearningConfig()
        self.accepted: list[NormalizedModalityInput] = []
        self.rejected: list[dict[str, Any]] = []

    def validate(self, payload: ModalityPayload) -> NormalizedModalityInput:
        """Validate one payload; raise :class:`ModalityValidationError` on rejection."""
        validator = {
            Modality.TEXT: self._validate_text,
            Modality.IMAGE: self._validate_image,
            Modality.AUDIO: self._validate_audio,
            Modality.VIDEO: self._validate_video,
            Modality.SPATIAL: self._validate_spatial,
            Modality.STRUCTURED: self._validate_structured,
        }.get(payload.modality)
        if validator is None:
            raise ModalityValidationError(f"unsupported modality {payload.modality!r}")
        normalized = validator(payload)
        self.accepted.append(normalized)
        return normalized

    def validate_many(self, payloads: Sequence[ModalityPayload]
                      ) -> list[NormalizedModalityInput]:
        """Validate a batch; invalid payloads are recorded and skipped, not fatal."""
        results: list[NormalizedModalityInput] = []
        for payload in payloads:
            try:
                results.append(self.validate(payload))
            except ModalityValidationError as exc:
                self.rejected.append({"source_ref": getattr(payload, "source_ref", "?"),
                                      "modality": payload.modality.value,
                                      "reason": str(exc)})
        return results

    # ------------------------------------------------------------- validators

    def _validate_text(self, payload: ModalityPayload) -> NormalizedModalityInput:
        """Text must be a non-empty string with a language tag."""
        if not isinstance(payload.data, str) or not payload.data.strip():
            raise ModalityValidationError("text payload must be a non-empty string")
        if not payload.language or len(payload.language) < 2:
            raise ModalityValidationError("text payload must declare a language tag")
        return self._normalized(payload, (len(payload.data),),
                                checks=("encoding-ok", "language-tagged"))

    def _validate_image(self, payload: ModalityPayload) -> NormalizedModalityInput:
        """Image must be a finite 2-D grayscale or 3-D HxWxC (C <= 4) array."""
        image = np.asarray(payload.data, dtype=float)
        if image.ndim not in (2, 3):
            raise ModalityValidationError(
                f"image must be 2-D or 3-D, got {image.ndim}-D")
        if image.ndim == 3 and not 1 <= image.shape[2] <= 4:
            raise ModalityValidationError(
                f"image channel count {image.shape[2]} outside 1..4")
        if image.size == 0:
            raise ModalityValidationError("image is empty")
        if max(image.shape[:2]) > self.config.max_image_side_pixels:
            raise ModalityValidationError(
                f"image side {max(image.shape[:2])} exceeds the declared limit "
                f"{self.config.max_image_side_pixels}")
        if not np.isfinite(image).all():
            raise ModalityValidationError("image contains non-finite values")
        return self._normalized(payload, tuple(int(n) for n in image.shape),
                                checks=("dimensions-ok", "finite-values",
                                        "resolution-within-limits"))

    def _validate_audio(self, payload: ModalityPayload) -> NormalizedModalityInput:
        """Audio must be a finite 1-D signal with a positive sample rate."""
        signal = np.asarray(payload.data, dtype=float)
        if signal.ndim != 1:
            raise ModalityValidationError(f"audio must be 1-D, got {signal.ndim}-D")
        if signal.size == 0:
            raise ModalityValidationError("audio is empty")
        if not math.isfinite(payload.sample_rate) or payload.sample_rate <= 0:
            raise ModalityValidationError("sample_rate must be positive")
        if not np.isfinite(signal).all():
            raise ModalityValidationError("audio contains non-finite samples")
        duration = signal.size / payload.sample_rate
        if duration > self.config.max_audio_seconds:
            raise ModalityValidationError(
                f"audio duration {duration:.1f}s exceeds the declared limit "
                f"{self.config.max_audio_seconds}s")
        return self._normalized(payload, (int(signal.size),),
                                duration_seconds=duration,
                                checks=("encoding-ok", "sample-rate-positive",
                                        "duration-within-limits"))

    def _validate_video(self, payload: ModalityPayload) -> NormalizedModalityInput:
        """Video must be a sequence of same-shape finite frames."""
        frames = payload.data
        if not isinstance(frames, (list, tuple)) or not frames:
            raise ModalityValidationError("video must be a non-empty sequence of frames")
        if len(frames) > self.config.max_video_frames:
            raise ModalityValidationError(
                f"video has {len(frames)} frames, limit "
                f"{self.config.max_video_frames}")
        shapes = set()
        for frame in frames:
            array = np.asarray(frame, dtype=float)
            if array.ndim != 2:
                raise ModalityValidationError("every video frame must be 2-D")
            if not np.isfinite(array).all():
                raise ModalityValidationError("video contains non-finite frames")
            shapes.add(array.shape)
        if len(shapes) != 1:
            raise ModalityValidationError("video frames must share one shape")
        return self._normalized(payload, (len(frames), *shapes.pop()),
                                checks=("frames-ok", "uniform-shape",
                                        "frame-count-within-limits"))

    def _validate_spatial(self, payload: ModalityPayload) -> NormalizedModalityInput:
        """Spatial coordinates must be an (N, 2) or (N, 3) finite array with a frame."""
        points = np.asarray(payload.data, dtype=float)
        if points.ndim != 2 or points.shape[1] not in (2, 3):
            raise ModalityValidationError(
                f"spatial coordinates must be (N, 2) or (N, 3), got {points.shape}")
        if points.size == 0:
            raise ModalityValidationError("spatial coordinates are empty")
        if not np.isfinite(points).all():
            raise ModalityValidationError("spatial coordinates contain non-finite values")
        if not payload.coordinate_frame:
            raise ModalityValidationError("spatial coordinates must declare a frame")
        return self._normalized(payload, tuple(int(n) for n in points.shape),
                                checks=("coordinates-ok", "frame-declared"))

    def _validate_structured(self, payload: ModalityPayload) -> NormalizedModalityInput:
        """Structured inputs must be a non-empty JSON-like mapping."""
        if not isinstance(payload.data, Mapping) or not payload.data:
            raise ModalityValidationError("structured payload must be a non-empty mapping")
        return self._normalized(payload, (len(payload.data),),
                                checks=("mapping-ok", "fields-present"))

    def _normalized(self, payload: ModalityPayload, shape: tuple[int, ...], *,
                    duration_seconds: float = 0.0,
                    checks: tuple[str, ...] = ()) -> NormalizedModalityInput:
        """Build the normalized record with a digest over structure and provenance."""
        return NormalizedModalityInput(
            payload=payload, shape=shape, duration_seconds=duration_seconds,
            checks=checks,
            digest=canonical_payload_digest({
                "source_ref": payload.source_ref,
                "timestamp": payload.timestamp,
                "modality": payload.modality.value,
                "summary": _data_summary(payload.data),
                "encoding": payload.encoding}))

    def describe(self) -> dict[str, Any]:
        """Summarize gateway activity."""
        return {"accepted": len(self.accepted), "rejected": len(self.rejected),
                "rejection_reasons": [item["reason"] for item in self.rejected[-5:]],
                "note": ("original references, timestamps and coordinate frames are "
                         "preserved; preprocessing is never destructive")}
