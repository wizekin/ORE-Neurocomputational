"""Validated engineering controls for printed-text localization."""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class TextLocalizationConfig:
    """Control thresholding, projection grouping, crop normalization and bounded alternatives."""

    canvas_size: int = 32
    canvas_margin: int = 3
    crop_padding_pixels: int = 2
    line_gap_pixels: int = 1
    word_gap_ratio: float = 0.55
    minimum_word_gap_pixels: int = 2
    minimum_glyph_width_ratio: float = 0.22
    maximum_glyph_aspect_ratio: float = 1.35
    valley_depth_threshold: float = 0.38
    maximum_valley_cuts_per_region: int = 3
    maximum_segmentation_hypotheses: int = 8
    minimum_ink_pixels: int = 2
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        """Validate bounded image sizes and dimensionless segmentation thresholds."""
        if self.canvas_size < 8 or not 0 <= self.canvas_margin < self.canvas_size // 2:
            raise ValueError("canvas_size must be >= 8 and canvas_margin must be less than half")
        if self.crop_padding_pixels < 0 or self.line_gap_pixels < 0 or self.minimum_word_gap_pixels < 0:
            raise ValueError("pixel gap and padding settings must be non-negative")
        for name in (
            "word_gap_ratio", "minimum_glyph_width_ratio",
            "maximum_glyph_aspect_ratio", "valley_depth_threshold",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.valley_depth_threshold >= 1.0:
            raise ValueError("valley_depth_threshold must be less than one")
        if self.maximum_glyph_aspect_ratio < 1.0:
            raise ValueError("maximum_glyph_aspect_ratio must be at least one")
        if self.maximum_valley_cuts_per_region < 0 or self.maximum_segmentation_hypotheses < 0:
            raise ValueError("alternative limits must be non-negative")
        if self.minimum_ink_pixels < 1 or not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("minimum_ink_pixels and epsilon must be positive")
