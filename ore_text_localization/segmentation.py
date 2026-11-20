"""Projection-based printed-text segmentation with explicit split alternatives.

The algorithm is intentionally constrained: it proposes line, word and glyph boxes from
binarized ink projections. It is not a general OCR model and cannot reliably separate
handwriting, joined scripts or every touching/overlapping glyph.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import TextLocalizationConfig


@dataclass(frozen=True, slots=True)
class GlyphBox:
    """Represent one primary glyph candidate in working-image coordinates."""

    box: tuple[int, int, int, int]
    line_index: int
    word_index: int
    confidence: float
    alternative_boxes: tuple[tuple[int, int, int, int], ...] = ()
    segmentation_group_id: int | None = None


@dataclass(frozen=True, slots=True)
class SegmentationResult:
    """Return ordered glyph boxes, word intervals, line boxes and alternative box sequences."""

    glyphs: tuple[GlyphBox, ...]
    word_spans: tuple[tuple[int, int], ...]
    lines: tuple[dict[str, Any], ...]
    alternatives: tuple[tuple[tuple[int, int, int, int], ...], ...]


def _runs(active: np.ndarray, max_gap: int = 0) -> list[tuple[int, int]]:
    """Return half-open runs of active indices, optionally bridging small inactive gaps."""
    indices = np.flatnonzero(np.asarray(active, dtype=bool))
    if indices.size == 0:
        return []
    groups: list[tuple[int, int]] = []
    start = previous = int(indices[0])
    for raw in indices[1:]:
        current = int(raw)
        if current - previous > max_gap + 1:
            groups.append((start, previous + 1))
            start = current
        previous = current
    groups.append((start, previous + 1))
    return groups


def _smooth_projection(values: np.ndarray) -> np.ndarray:
    """Apply a short symmetric moving average to suppress single-column segmentation noise."""
    if values.size < 3:
        return values.astype(np.float64)
    padded = np.pad(np.asarray(values, dtype=np.float64), (1, 1), mode="edge")
    return 0.25 * padded[:-2] + 0.50 * padded[1:-1] + 0.25 * padded[2:]


def _valley_cuts(
    projection: np.ndarray,
    start: int,
    end: int,
    line_height: int,
    config: TextLocalizationConfig,
) -> list[int]:
    """Find bounded low-ink valleys only in components wide enough to contain multiple glyphs."""
    width = end - start
    if (
        config.maximum_valley_cuts_per_region == 0
        or width <= config.maximum_glyph_aspect_ratio * line_height
    ):
        return []
    local = _smooth_projection(projection[start:end])
    if local.size < 5:
        return []
    minimum_width = max(1, int(round(line_height * config.minimum_glyph_width_ratio)))
    minima = [
        index for index in range(1, local.size - 1)
        if local[index] <= local[index - 1] and local[index] < local[index + 1]
    ]
    candidates: list[tuple[float, int]] = []
    for index in minima:
        cut = start + index
        if cut - start < minimum_width or end - cut < minimum_width:
            continue
        left_peak = float(np.max(local[:index + 1]))
        right_peak = float(np.max(local[index:]))
        reference = min(left_peak, right_peak)
        if reference <= 0.0:
            continue
        depth = 1.0 - float(local[index]) / reference
        relative = index / max(local.size, 1)
        if 0.12 <= relative <= 0.88 and depth >= config.valley_depth_threshold:
            candidates.append((depth, cut))
    candidates.sort(reverse=True)
    selected: list[int] = []
    for _, cut in candidates:
        if all(abs(cut - other) >= minimum_width for other in selected):
            selected.append(cut)
        if len(selected) >= config.maximum_valley_cuts_per_region:
            break
    return sorted(selected)


def segment_ink_mask(
    ink_mask: np.ndarray,
    config: TextLocalizationConfig | None = None,
    *,
    reading_direction: str = "ltr",
    vertical_layout: bool = False,
) -> SegmentationResult:
    """Segment a binary ink mask into line/word/glyph boxes and retain unsplit alternatives."""
    cfg = config or TextLocalizationConfig()
    mask = np.asarray(ink_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("ink_mask must be two-dimensional")
    working = mask.T if vertical_layout else mask
    line_projection = np.sum(working, axis=1)
    line_runs = _runs(line_projection >= cfg.minimum_ink_pixels, cfg.line_gap_pixels)
    all_glyphs: list[GlyphBox] = []
    word_spans: list[tuple[int, int]] = []
    lines: list[dict[str, Any]] = []
    alternatives: list[tuple[tuple[int, int, int, int], ...]] = []

    reverse_lines = (
        (reading_direction == "btt") if not vertical_layout
        else (reading_direction in {"ttb", "rtl"})
    )
    reverse_glyphs = (
        (reading_direction == "rtl") if not vertical_layout
        else (reading_direction == "btt")
    )
    if reverse_lines:
        line_runs.reverse()

    for line_index, (line_start, line_end) in enumerate(line_runs):
        line_mask = working[line_start:line_end, :]
        column_projection = np.sum(line_mask, axis=0)
        raw_runs = _runs(column_projection >= cfg.minimum_ink_pixels)
        if reverse_glyphs:
            raw_runs.reverse()
        line_height = max(1, line_end - line_start)
        word_gap_threshold = max(
            cfg.minimum_word_gap_pixels,
            int(round(line_height * cfg.word_gap_ratio)),
        )
        line_start_glyph = len(all_glyphs)
        current_word_start = len(all_glyphs)
        previous_end: int | None = None
        word_index = 0
        line_word_spans: list[tuple[int, int]] = []

        for run_index, (run_start, run_end) in enumerate(raw_runs):
            if previous_end is not None and abs(run_start - previous_end) > word_gap_threshold:
                if len(all_glyphs) > current_word_start:
                    word_spans.append((current_word_start, len(all_glyphs)))
                    line_word_spans.append((current_word_start, len(all_glyphs)))
                    word_index += 1
                current_word_start = len(all_glyphs)

            cuts = _valley_cuts(column_projection, run_start, run_end, line_height, cfg)
            boundaries = [run_start, *cuts, run_end]
            local_boxes: list[tuple[int, int, int, int]] = []
            for left, right in zip(boundaries, boundaries[1:]):
                local_ink = line_mask[:, left:right]
                occupied_rows = np.flatnonzero(np.any(local_ink, axis=1))
                if occupied_rows.size == 0:
                    continue
                top = line_start + int(occupied_rows[0])
                bottom = line_start + int(occupied_rows[-1]) + 1
                box = (top, left, bottom, right) if vertical_layout else (left, top, right, bottom)
                local_boxes.append(box)

            if reverse_glyphs:
                local_boxes.reverse()
            original_box = (
                (line_start, run_start, line_end, run_end)
                if vertical_layout else
                (run_start, line_start, run_end, line_end)
            )
            group_id = line_index * 100000 + run_index if cuts else None
            if cuts:
                alternatives.append((original_box,))
            for box in local_boxes:
                all_glyphs.append(GlyphBox(
                    box=box,
                    line_index=line_index,
                    word_index=word_index,
                    confidence=0.62 if cuts else 0.88,
                    alternative_boxes=(original_box,) if cuts else (),
                    segmentation_group_id=group_id,
                ))
            previous_end = run_end

        if len(all_glyphs) > current_word_start:
            word_spans.append((current_word_start, len(all_glyphs)))
            line_word_spans.append((current_word_start, len(all_glyphs)))
        if len(all_glyphs) > line_start_glyph:
            line_boxes = [item.box for item in all_glyphs[line_start_glyph:]]
            region = (
                min(box[0] for box in line_boxes),
                min(box[1] for box in line_boxes),
                max(box[2] for box in line_boxes),
                max(box[3] for box in line_boxes),
            )
            lines.append({
                "line_index": line_index,
                "bounding_box": list(region),
                "glyph_count": len(all_glyphs) - line_start_glyph,
                "word_spans": [list(span) for span in line_word_spans],
            })

    return SegmentationResult(
        glyphs=tuple(all_glyphs),
        word_spans=tuple(sorted(word_spans, key=lambda span: span[0])),
        lines=tuple(lines),
        alternatives=tuple(alternatives[:cfg.maximum_segmentation_hypotheses]),
    )
