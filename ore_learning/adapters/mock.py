"""Deterministic mock and reference specialists (Point 14, Stage 1 mock adapter).

These adapters implement the SpecialistAdapter protocol on top of the tiny numpy
models in :mod:`ore_learning.nn`.  They are deterministic under fixed seeds and let
the whole RLMS lifecycle be exercised end-to-end without external frameworks.
"""
from __future__ import annotations

import hashlib
import time
from typing import Any, Mapping

import numpy as np

from ..contracts import AdapterKind, SchemaDescriptor
from ..nn import TinyLinearModel
from .base import ExecutionContext, validate_payload_against_schema

__all__ = ("BrokenAdapter", "CountingAdapter", "MockCaptionAdapter",
           "MockRegressionAdapter", "MockSpeechAdapter", "MockSpatialAdapter",
           "MockTextSentimentAdapter", "MockVisionAdapter", "build_mock_registry")


def _stable_hash(text: str) -> int:
    """A stable (non-python-hash) integer digest of a string."""
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


class MockTextSentimentAdapter:
    """Text -> (label, score) sentiment specialist over deterministic lexical features."""

    adapter_kind = AdapterKind.TEXT
    input_schema = SchemaDescriptor("text", "tokens", ("text", "language"))
    output_schema = SchemaDescriptor("text", "sentiment", ("label", "score"))

    POSITIVE = ("good", "great", "excellent", "clear", "improve", "gain", "strong")
    NEGATIVE = ("bad", "poor", "fail", "broken", "error", "loss", "weak")

    def invoke(self, payload: Mapping[str, Any],
               execution: ExecutionContext) -> dict[str, Any]:
        """Score sentiment from lexical overlap; checks a checkpoint per call."""
        execution.checkpoint()
        text = str(payload.get("text", "")).lower()
        words = set(text.split())
        positive = len(words & set(self.POSITIVE))
        negative = len(words & set(self.NEGATIVE))
        score = 0.5 + 0.25 * (positive - negative)
        score = float(min(1.0, max(0.0, score)))
        label = "positive" if score > 0.55 else "negative" if score < 0.45 else "neutral"
        return {"label": label, "score": round(score, 6)}


class MockVisionAdapter:
    """Image -> pooled feature descriptor (deterministic, region-anchored)."""

    adapter_kind = AdapterKind.VISION
    input_schema = SchemaDescriptor("image", "features", ("region", "descriptor"))
    output_schema = SchemaDescriptor("image", "features", ("region", "descriptor"))

    def invoke(self, payload: Mapping[str, Any],
               execution: ExecutionContext) -> dict[str, Any]:
        """Pool row and column means of the supplied image region."""
        execution.checkpoint()
        image = np.asarray(payload["image"], dtype=float)
        if image.ndim == 3:
            image = image.mean(axis=2)
        region = str(payload.get("region", "full-frame"))
        descriptor = [round(float(image.mean(axis=0).mean()), 6),
                      round(float(image.mean(axis=1).mean()), 6),
                      round(float(image.std()), 6)]
        return {"region": region, "descriptor": descriptor}


class MockCaptionAdapter:
    """Image features -> a deterministic caption proposal with confidence."""

    adapter_kind = AdapterKind.VISION
    input_schema = SchemaDescriptor("image", "features", ("region", "descriptor"))
    output_schema = SchemaDescriptor("image", "caption",
                                     ("region", "caption", "confidence"))

    VOCABULARY = ("edge", "surface", "pattern", "region", "texture")

    def invoke(self, payload: Mapping[str, Any],
               execution: ExecutionContext) -> dict[str, Any]:
        """Derive a caption template from the descriptor with bounded confidence."""
        execution.checkpoint()
        region = str(payload.get("region", "full-frame"))
        descriptor = [float(value) for value in np.atleast_1d(payload["descriptor"])]
        token = self.VOCABULARY[_stable_hash(region + repr(descriptor)) % len(
            self.VOCABULARY)]
        confidence = 0.4 + 0.2 * (float(np.mean(np.abs(descriptor))) % 1.0)
        return {"region": region, "caption": f"{token} observed in {region}",
                "confidence": round(float(min(0.95, confidence)), 6)}


class MockSpeechAdapter:
    """Audio -> deterministic pseudo-transcript over energy peaks."""

    adapter_kind = AdapterKind.AUDIO
    input_schema = SchemaDescriptor("audio", "features", ("interval", "descriptor"))
    output_schema = SchemaDescriptor("audio", "transcript",
                                     ("interval", "transcript", "confidence"))

    TOKENS = ("silence", "speech", "noise", "tone", "burst")

    def invoke(self, payload: Mapping[str, Any],
               execution: ExecutionContext) -> dict[str, Any]:
        """Transcribe energy-peak structure into a token sequence."""
        execution.checkpoint()
        signal = np.asarray(payload["audio"], dtype=float)
        sample_rate = float(payload.get("sample_rate", 16000.0))
        interval = str(payload.get("interval", "0.0-1.0"))
        if signal.size == 0:
            return {"interval": interval, "transcript": "silence", "confidence": 0.5}
        peaks = np.argmax(signal.reshape(-1, max(1, signal.size // 8)), axis=1)
        tokens = [self.TOKENS[int(peak + index) % len(self.TOKENS)]
                  for index, peak in enumerate(peaks[:8])]
        confidence = float(min(0.95, 0.5 + np.std(signal) / (2.0 * (abs(np.mean(signal))
                                                                    + 1e-9))))
        return {"interval": interval, "transcript": " ".join(tokens) or "silence",
                "confidence": round(confidence, 6)}

    @property
    def sample_rate_hint(self) -> float:
        """The assumed default sample rate."""
        return 16000.0


class MockSpatialAdapter:
    """Spatial coordinates -> normalized points in the declared coordinate frame."""

    adapter_kind = AdapterKind.SPATIAL
    input_schema = SchemaDescriptor("spatial", "coordinates", ("frame", "points"))
    output_schema = SchemaDescriptor("spatial", "coordinates", ("frame", "points"))

    def invoke(self, payload: Mapping[str, Any],
               execution: ExecutionContext) -> dict[str, Any]:
        """Normalize coordinates into [0, 1]^d of the frame's bounding box."""
        execution.checkpoint()
        points = np.asarray(payload["points"], dtype=float)
        frame = str(payload.get("frame", "world"))
        if points.size == 0:
            return {"frame": frame, "points": []}
        low, high = points.min(axis=0), points.max(axis=0)
        span = np.where(high - low > 1e-12, high - low, 1.0)
        normalized = ((points - low) / span).round(6).tolist()
        return {"frame": frame, "points": normalized}


class MockRegressionAdapter:
    """Structured features -> regression output from a seeded tiny linear model."""

    adapter_kind = AdapterKind.MOCK
    input_schema = SchemaDescriptor("structured", "regression", ("values", "intervals"))
    output_schema = SchemaDescriptor("structured", "regression", ("values", "intervals"))

    def __init__(self, model: TinyLinearModel | None = None, *,
                 n_inputs: int = 4, seed: int = 41) -> None:
        """Build or accept the underlying tiny model."""
        self.model = model or TinyLinearModel(n_inputs, 1, seed=seed)

    def invoke(self, payload: Mapping[str, Any],
               execution: ExecutionContext) -> dict[str, Any]:
        """Predict one value with a calibrated-looking confidence."""
        execution.checkpoint()
        values = np.atleast_2d(np.asarray(payload["values"], dtype=float))
        prediction = float(self.model.predict(values)[0, 0])
        intervals = [[prediction - 0.1, prediction + 0.1]]
        confidence = 0.5 + 0.25 * abs(prediction) % 0.4
        return {"values": [round(prediction, 6)], "intervals": intervals,
                "confidence": round(min(0.95, confidence), 6)}


class CountingAdapter:
    """A checkpoint-cooperating adapter that performs N tiny work units."""

    adapter_kind = AdapterKind.MOCK
    input_schema = SchemaDescriptor("structured", "regression", ("values", "intervals"))
    output_schema = SchemaDescriptor("structured", "regression", ("values", "intervals"))

    def __init__(self, steps: int = 8, sleep_seconds: float = 0.0) -> None:
        """Configure the number of work units and per-unit sleep."""
        self.steps = int(steps)
        self.sleep_seconds = float(sleep_seconds)
        self.calls = 0

    def invoke(self, payload: Mapping[str, Any],
               execution: ExecutionContext) -> dict[str, Any]:
        """Run the configured number of checkpointed work units."""
        self.calls += 1
        for _ in range(self.steps):
            execution.checkpoint()
            if self.sleep_seconds:
                time.sleep(self.sleep_seconds)
        return {"values": [float(len(payload.get("values", [])))],
                "intervals": [[0.0, 1.0]], "confidence": 0.5}


class BrokenAdapter:
    """An adapter whose output violates its declared schema (negative testing)."""

    adapter_kind = AdapterKind.MOCK
    input_schema = SchemaDescriptor("structured", "regression", ("values", "intervals"))
    output_schema = SchemaDescriptor("structured", "regression", ("values", "intervals"))

    def invoke(self, payload: Mapping[str, Any],
               execution: ExecutionContext) -> dict[str, Any]:
        """Emit a malformed payload missing a required field."""
        execution.checkpoint()
        return {"values": [1.0]}  # missing "intervals"


def build_mock_registry() -> "AdapterRegistry":
    """Assemble the default registry of deterministic reference adapters."""
    from .base import AdapterRegistry

    registry = AdapterRegistry()
    for adapter in (MockTextSentimentAdapter(), MockVisionAdapter(),
                    MockCaptionAdapter(), MockSpeechAdapter(), MockSpatialAdapter(),
                    MockRegressionAdapter()):
        registry.register(adapter)
    return registry
