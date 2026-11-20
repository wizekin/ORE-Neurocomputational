"""Typed contracts for raw text-image localization and glyph segmentation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True, slots=True)
class TextImageRequest:
    """Represent one original image and its explicit script/layout metadata."""

    sequence_id: str
    image: np.ndarray
    script_profile: Mapping[str, Any] = field(default_factory=dict)
    source_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject malformed images and ambiguous reading directions before processing."""
        if not isinstance(self.sequence_id, str) or not self.sequence_id.strip():
            raise ValueError("sequence_id must be a non-empty string")
        array = np.asarray(self.image)
        if array.ndim not in (2, 3) or min(array.shape[:2]) < 4:
            raise ValueError("image must be a 2-D grayscale or 3-D colour image at least 4x4")
        if array.ndim == 3 and array.shape[2] not in (3, 4):
            raise ValueError("colour images must have three or four channels")
        if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
            raise ValueError("image must contain only finite numeric pixel values")
        if np.issubdtype(array.dtype, np.floating) and np.any(array < 0.0):
            raise ValueError("floating image values cannot be negative")
        if not isinstance(self.script_profile, Mapping) or not isinstance(self.metadata, Mapping):
            raise TypeError("script_profile and metadata must be mappings")
        direction = str(self.script_profile.get("reading_direction", "ltr")).lower()
        if direction not in {"ltr", "rtl", "ttb", "btt"}:
            raise ValueError("reading_direction must be ltr, rtl, ttb, or btt")
        layout = str(self.script_profile.get("layout", "horizontal")).lower()
        if layout not in {"horizontal", "vertical"}:
            raise ValueError("layout must be horizontal or vertical")


@dataclass(frozen=True, slots=True)
class LocalizedGlyph:
    """Store one normalized glyph crop with its original-image coordinates."""

    patch_id: str
    image: np.ndarray
    bounding_box: tuple[int, int, int, int]
    line_index: int
    word_index: int
    reading_order: int
    segmentation_confidence: float
    alternative_boxes: tuple[tuple[int, int, int, int], ...] = ()

    def __post_init__(self) -> None:
        """Validate crop dimensions, coordinate order and bounded segmentation confidence."""
        if not self.patch_id.strip():
            raise ValueError("patch_id must be non-empty")
        image = np.asarray(self.image)
        if image.ndim != 2 or min(image.shape) < 4 or not np.isfinite(image).all():
            raise ValueError("localized glyph image must be a finite grayscale array at least 4x4")
        x0, y0, x1, y1 = self.bounding_box
        if min(x0, y0) < 0 or x1 <= x0 or y1 <= y0:
            raise ValueError("bounding_box must be a valid non-negative xyxy rectangle")
        if not 0.0 <= float(self.segmentation_confidence) <= 1.0:
            raise ValueError("segmentation_confidence must lie in [0, 1]")


@dataclass(frozen=True, slots=True)
class SegmentationHypothesis:
    """Describe a bounded alternative segmentation without discarding the primary result."""

    hypothesis_id: str
    bounding_boxes: tuple[tuple[int, int, int, int], ...]
    probability: float
    reason: str

    def __post_init__(self) -> None:
        """Validate a normalized confidence-like weight and all alternative boxes."""
        if not self.hypothesis_id.strip() or not self.reason.strip():
            raise ValueError("hypothesis_id and reason must be non-empty")
        if not 0.0 <= float(self.probability) <= 1.0:
            raise ValueError("probability must lie in [0, 1]")
        for x0, y0, x1, y1 in self.bounding_boxes:
            if min(x0, y0) < 0 or x1 <= x0 or y1 <= y0:
                raise ValueError("alternative bounding boxes must be valid xyxy rectangles")


@dataclass(frozen=True, slots=True)
class TextLocalizationResult:
    """Preserve image evidence, ordered glyph patches, word spans and segmentation alternatives."""

    sequence_id: str
    grayscale_image: np.ndarray
    ink_mask: np.ndarray
    localized_glyphs: tuple[LocalizedGlyph, ...]
    word_spans: tuple[tuple[int, int], ...]
    line_information: tuple[Mapping[str, Any], ...]
    segmentation_hypotheses: tuple[SegmentationHypothesis, ...]
    script_profile: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a compact JSON-compatible summary without serializing raw pixel arrays."""
        return {
            "schema": "ore.text_localization.v1",
            "sequence_id": self.sequence_id,
            "image_shape": list(np.asarray(self.grayscale_image).shape),
            "glyphs": [
                {
                    "patch_id": glyph.patch_id,
                    "bounding_box": list(glyph.bounding_box),
                    "line_index": glyph.line_index,
                    "word_index": glyph.word_index,
                    "reading_order": glyph.reading_order,
                    "segmentation_confidence": float(glyph.segmentation_confidence),
                    "alternative_boxes": [list(box) for box in glyph.alternative_boxes],
                }
                for glyph in self.localized_glyphs
            ],
            "word_spans": [list(span) for span in self.word_spans],
            "line_information": [dict(item) for item in self.line_information],
            "segmentation_hypotheses": [
                {
                    "hypothesis_id": item.hypothesis_id,
                    "bounding_boxes": [list(box) for box in item.bounding_boxes],
                    "probability": float(item.probability),
                    "reason": item.reason,
                }
                for item in self.segmentation_hypotheses
            ],
            "script_profile": dict(self.script_profile),
            "diagnostics": dict(self.diagnostics),
        }

    def to_n9_request(
        self,
        *,
        task: str = "read",
        context: Mapping[str, Any] | None = None,
        source_id: str | None = None,
    ) -> Any:
        """Convert normalized glyph images and word intervals to the established N9 contract."""
        from ore_n9 import N9Patch, N9Request

        patches = tuple(
            N9Patch(
                patch_id=glyph.patch_id,
                image=np.asarray(glyph.image, dtype=np.float32),
                segmentation_confidence=float(glyph.segmentation_confidence),
                x=(glyph.bounding_box[0] + glyph.bounding_box[2]) / 2.0,
                y=(glyph.bounding_box[1] + glyph.bounding_box[3]) / 2.0,
                source_metadata={
                    "bounding_box": list(glyph.bounding_box),
                    "line_index": glyph.line_index,
                    "word_index": glyph.word_index,
                    "reading_order": glyph.reading_order,
                    "localization_source": "ore_text_localization",
                },
            )
            for glyph in self.localized_glyphs
        )
        return N9Request(
            sequence_id=self.sequence_id,
            patches=patches,
            task=task,
            script_profile=dict(self.script_profile),
            context=dict(context or {}),
            word_spans=self.word_spans,
            source_id=source_id,
        )


@dataclass(frozen=True, slots=True)
class TrainingGlyph:
    """Provide a labelled printed-glyph image for supervised N3 model fitting."""

    grapheme: str
    image: np.ndarray

    def __post_init__(self) -> None:
        """Ensure training labels and grayscale examples are suitable for the N1/N2 feature path."""
        if not isinstance(self.grapheme, str) or not self.grapheme.strip():
            raise ValueError("grapheme must be a non-empty label")
        image = np.asarray(self.image)
        if image.ndim != 2 or min(image.shape) < 4 or not np.isfinite(image).all():
            raise ValueError("training image must be a finite grayscale array at least 4x4")
