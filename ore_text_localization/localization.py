"""Raw-image normalization, text-line localization and glyph-crop construction."""
from __future__ import annotations

from typing import Any

import numpy as np

from .config import TextLocalizationConfig
from .contracts import (
    LocalizedGlyph,
    SegmentationHypothesis,
    TextImageRequest,
    TextLocalizationResult,
)
from .segmentation import SegmentationResult, segment_ink_mask


def normalize_grayscale_image(image: np.ndarray) -> np.ndarray:
    """Convert grayscale/RGB/RGBA pixels to finite float32 luminance in [0, 1]."""
    array = np.asarray(image)
    if array.ndim == 3:
        rgb = array[..., :3].astype(np.float64)
        if np.issubdtype(array.dtype, np.integer):
            rgb = rgb / max(float(np.iinfo(array.dtype).max), 1.0)
        elif float(np.max(rgb)) > 1.0:
            rgb = rgb / float(np.max(rgb))
        rgb = np.clip(rgb, 0.0, 1.0)
        gray = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    elif array.ndim == 2:
        gray = array.astype(np.float64)
        if np.issubdtype(array.dtype, np.integer):
            gray = gray / max(float(np.iinfo(array.dtype).max), 1.0)
        elif float(np.max(gray)) > 1.0:
            gray = gray / float(np.max(gray))
        gray = np.clip(gray, 0.0, 1.0)
    else:
        raise ValueError("image must be a 2-D grayscale or 3-D RGB/RGBA array")
    if not np.isfinite(gray).all():
        raise ValueError("image contains non-finite luminance values")
    return np.asarray(gray, dtype=np.float32)


def otsu_threshold(gray: np.ndarray, bins: int = 256) -> float:
    """Estimate a global ink threshold by maximizing between-class histogram variance."""
    values = np.clip(np.asarray(gray, dtype=np.float64), 0.0, 1.0)
    histogram, edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    probabilities = histogram.astype(np.float64)
    probabilities /= max(float(probabilities.sum()), 1.0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    cumulative = np.cumsum(probabilities)
    means = np.cumsum(probabilities * centers)
    total_mean = float(means[-1])
    denominator = cumulative * (1.0 - cumulative)
    numerator = (total_mean * cumulative - means) ** 2
    variance = np.divide(
        numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-12,
    )
    return float(centers[int(np.argmax(variance))])


def _resize_to_canvas(crop: np.ndarray, config: TextLocalizationConfig) -> np.ndarray:
    """Aspect-preservingly resample a crop into the fixed N9 grayscale input canvas."""
    image = np.asarray(crop, dtype=np.float64)
    height, width = image.shape
    available = config.canvas_size - 2 * config.canvas_margin
    scale = min(available / max(height, 1), available / max(width, 1))
    target_h = max(1, min(available, int(round(height * scale))))
    target_w = max(1, min(available, int(round(width * scale))))
    source_x = np.arange(width, dtype=np.float64)
    target_x = np.linspace(0.0, max(width - 1, 0), target_w)
    horizontal = np.stack([np.interp(target_x, source_x, row) for row in image], axis=0)
    source_y = np.arange(height, dtype=np.float64)
    target_y = np.linspace(0.0, max(height - 1, 0), target_h)
    resized = np.stack(
        [np.interp(target_y, source_y, horizontal[:, column]) for column in range(target_w)],
        axis=1,
    )
    canvas = np.ones((config.canvas_size, config.canvas_size), dtype=np.float32)
    top = (config.canvas_size - target_h) // 2
    left = (config.canvas_size - target_w) // 2
    canvas[top:top + target_h, left:left + target_w] = np.clip(resized, 0.0, 1.0)
    return canvas


def _make_ink_mask(gray: np.ndarray, epsilon: float) -> tuple[np.ndarray, float, str]:
    """Threshold the darker/lighter foreground according to the image-border background."""
    if float(np.std(gray)) <= epsilon:
        return np.zeros(gray.shape, dtype=bool), float(np.mean(gray)), "uniform_image"
    threshold = otsu_threshold(gray)
    border = np.concatenate((gray[0], gray[-1], gray[1:-1, 0], gray[1:-1, -1]))
    background_level = float(np.median(border))
    polarity = "dark_on_light" if background_level >= threshold else "light_on_dark"
    ink = gray <= threshold if polarity == "dark_on_light" else gray >= threshold
    return np.asarray(ink, dtype=bool), threshold, polarity


class TextLocalizer:
    """Localize constrained printed text into N9-ready grayscale glyph patches.

    This module proposes boxes and preserves uncertainty; it does not recognize grapheme
    identities. Joined scripts, cursive handwriting, severe skew and complex page layouts
    require stronger specialized segmentation models.
    """

    def __init__(self, config: TextLocalizationConfig | None = None) -> None:
        """Store validated localization controls without fitting or mutating a recognition model."""
        self.config = config or TextLocalizationConfig()

    def localize(self, request: TextImageRequest) -> TextLocalizationResult:
        """Normalize the page, segment reading-order glyphs and retain bounded alternatives."""
        if not isinstance(request, TextImageRequest):
            raise TypeError("request must be a TextImageRequest")
        cfg = self.config
        gray = normalize_grayscale_image(request.image)
        ink_mask, threshold, polarity = _make_ink_mask(gray, cfg.epsilon)
        if polarity == "light_on_dark":
            gray = 1.0 - gray
        profile = dict(request.script_profile)
        layout = str(profile.get("layout", "horizontal")).lower()
        direction = str(profile.get(
            "reading_direction", "ttb" if layout == "vertical" else "ltr"
        )).lower()
        segmentation: SegmentationResult = segment_ink_mask(
            ink_mask, cfg, reading_direction=direction, vertical_layout=(layout == "vertical"),
        )

        glyphs: list[LocalizedGlyph] = []
        pad = cfg.crop_padding_pixels
        height, width = gray.shape
        for index, item in enumerate(segmentation.glyphs):
            x0, y0, x1, y1 = item.box
            crop = gray[
                max(0, y0 - pad):min(height, y1 + pad),
                max(0, x0 - pad):min(width, x1 + pad),
            ]
            if crop.size == 0:
                continue
            normalized = _resize_to_canvas(crop, cfg)
            glyphs.append(LocalizedGlyph(
                patch_id=f"{request.sequence_id}:glyph:{index:04d}",
                image=normalized,
                bounding_box=(x0, y0, x1, y1),
                line_index=item.line_index,
                word_index=item.word_index,
                reading_order=len(glyphs),
                segmentation_confidence=float(item.confidence),
                alternative_boxes=tuple(item.alternative_boxes),
            ))

        retained_spans: list[tuple[int, int]] = []
        if len(glyphs) == len(segmentation.glyphs):
            retained_spans = list(segmentation.word_spans)
        elif glyphs:
            current_word = glyphs[0].word_index
            start = 0
            for index, glyph in enumerate(glyphs[1:], start=1):
                if glyph.word_index != current_word:
                    retained_spans.append((start, index))
                    start, current_word = index, glyph.word_index
            retained_spans.append((start, len(glyphs)))

        primary_boxes = tuple(glyph.bounding_box for glyph in glyphs)
        candidate_rows: list[tuple[tuple[int, int, int, int], ...]] = []
        group_ids = sorted({
            item.segmentation_group_id for item in segmentation.glyphs
            if item.segmentation_group_id is not None
        })
        for group_id in group_ids:
            group_indices = [
                index for index, item in enumerate(segmentation.glyphs)
                if item.segmentation_group_id == group_id
            ]
            if not group_indices:
                continue
            original_box = segmentation.glyphs[group_indices[0]].alternative_boxes[0]
            first_index = group_indices[0]
            grouped = set(group_indices)
            boxes: list[tuple[int, int, int, int]] = []
            for index, item in enumerate(segmentation.glyphs):
                if index == first_index:
                    boxes.append(original_box)
                elif index not in grouped:
                    boxes.append(item.box)
            candidate_rows.append(tuple(boxes))
            if len(candidate_rows) >= cfg.maximum_segmentation_hypotheses:
                break

        hypotheses: list[SegmentationHypothesis] = []
        primary_weight = 1.0 if not candidate_rows else 0.80
        if glyphs:
            hypotheses.append(SegmentationHypothesis(
                hypothesis_id=f"{request.sequence_id}:segmentation:primary",
                bounding_boxes=primary_boxes,
                probability=primary_weight,
                reason="projection segmentation with conservative valley splits",
            ))
        alternative_weight = (1.0 - primary_weight) / max(len(candidate_rows), 1)
        for index, boxes in enumerate(candidate_rows):
            hypotheses.append(SegmentationHypothesis(
                hypothesis_id=f"{request.sequence_id}:segmentation:alternative:{index + 1}",
                bounding_boxes=boxes,
                probability=alternative_weight,
                reason="retain a connected region as an unsplit boundary alternative",
            ))

        diagnostics: dict[str, Any] = {
            "status": "ok" if glyphs else "no_text_like_ink_detected",
            "threshold": float(threshold),
            "foreground_polarity": polarity,
            "layout": layout,
            "reading_direction": direction,
            "glyph_count": len(glyphs),
            "word_count": len(retained_spans),
            "line_count": len(segmentation.lines),
            "segmentation_alternatives_retained": len(candidate_rows),
            "recognition_performed": False,
            "mathematical_methods": [
                "Otsu between-class variance threshold",
                "horizontal/vertical ink projection and run-length grouping",
                "low-ink valley depth for candidate touching-glyph cuts",
                "aspect-preserving bilinear interpolation to a fixed grayscale canvas",
            ],
            "limitations": [
                "projection-based printed-text segmentation; not general OCR",
                "joined/cursive scripts and ligatures may remain unsplit or be segmented incorrectly",
                "grapheme identity is delegated to a fitted N3 model and downstream ORE modules",
            ],
        }
        return TextLocalizationResult(
            sequence_id=request.sequence_id,
            grayscale_image=gray,
            ink_mask=ink_mask,
            localized_glyphs=tuple(glyphs),
            word_spans=tuple(retained_spans),
            line_information=tuple(segmentation.lines),
            segmentation_hypotheses=tuple(hypotheses),
            script_profile=profile,
            diagnostics=diagnostics,
        )

    def process(self, request: TextImageRequest) -> TextLocalizationResult:
        """Alias localize() to match the process-style APIs used across ORE packages."""
        return self.localize(request)


__all__ = ["TextLocalizer", "normalize_grayscale_image", "otsu_threshold"]
