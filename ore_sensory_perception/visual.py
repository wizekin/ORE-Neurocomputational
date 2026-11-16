"""Visual stages 1–2: text-region selection and distributed character segmentation."""
from __future__ import annotations

from typing import Any
import numpy as np
from scipy import ndimage as ndi

from .contracts import SensoryConfig, SensoryRequest
from .math_ops import grayscale_image, otsu_threshold, safe_summary, sigmoid, softmax


def _gabor_features(image: np.ndarray, config: SensoryConfig) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Apply oriented multi-scale Gabor filters; quadrature energy approximates V1 feature populations."""
    radius = config.gabor_size // 2
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    maps, summary = [], []
    for theta in config.visual_orientations:
        for frequency in config.visual_frequencies:
            xr = xx * np.cos(theta) + yy * np.sin(theta)
            yr = -xx * np.sin(theta) + yy * np.cos(theta)
            envelope = np.exp(-(xr**2 + 0.49 * yr**2) / (2.0 * 2.8**2))
            even = envelope * np.cos(2.0 * np.pi * frequency * xr)
            odd = envelope * np.cos(2.0 * np.pi * frequency * xr + np.pi / 2)
            for kernel in (even, odd):
                kernel -= np.mean(kernel)
                kernel /= max(float(np.linalg.norm(kernel)), 1e-12)
            re, ro = (ndi.convolve(image, k, mode="reflect") for k in (even, odd))
            energy = np.sqrt(re**2 + ro**2)
            maps.append(energy)
            summary.append({"orientation_rad": float(theta), "spatial_frequency": float(frequency),
                            "mean_energy": float(np.mean(energy))})
    stack = np.stack(maps)
    gain = ndi.gaussian_filter(np.sum(stack, axis=0), sigma=2.0) + config.epsilon
    return stack / gain[None, :, :], summary


def _tile_feature_records(stack: np.ndarray, contrast: np.ndarray, tile_size: int) -> list[dict[str, Any]]:
    """Pool edge density, orientation entropy, frequency preference and contrast into local tile vectors."""
    records = []
    for y0 in range(0, stack.shape[1], tile_size):
        for x0 in range(0, stack.shape[2], tile_size):
            y1, x1 = min(y0 + tile_size, stack.shape[1]), min(x0 + tile_size, stack.shape[2])
            tile = stack[:, y0:y1, x0:x1]
            energy = np.mean(tile, axis=(1, 2))
            distribution = energy / max(float(np.sum(energy)), 1e-12)
            entropy = -float(np.sum(distribution * np.log(distribution + 1e-12)))
            entropy /= max(float(np.log(max(distribution.size, 2))), 1e-12)
            pooled = np.sum(tile, axis=0)
            records.append({
                "tile_id": len(records), "bbox": (y0, x0, y1, x1),
                "edge_density": float(np.mean(pooled > np.quantile(pooled, 0.65))),
                "orientation_entropy": entropy,
                "spatial_frequency_index": float(np.dot(energy, np.arange(energy.size)) /
                                                  max(float(np.sum(energy)), 1e-12)),
                "element_density": float(np.mean(pooled > np.mean(pooled))),
                "contrast_sd": float(np.std(contrast[y0:y1, x0:x1])),
                "center": ((x0 + x1) / 2.0, (y0 + y1) / 2.0),
            })
    return records


def _descriptor(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Create a normalized 16-value shape descriptor for optional caller-supplied grapheme prototypes."""
    y0, x0, y1, x1 = bbox
    crop = np.asarray(mask[y0:y1, x0:x1], dtype=np.float64)
    if crop.size == 0:
        return np.zeros(16)
    height, width = crop.shape
    resized = ndi.zoom(crop, (16 / max(height, 1), 16 / max(width, 1)), order=1)[:16, :16]
    padded = np.zeros((16, 16))
    padded[:resized.shape[0], :resized.shape[1]] = resized
    rows, cols = np.mean(padded, axis=1), np.mean(padded, axis=0)
    return np.asarray([
        width / max(height, 1), np.mean(crop), np.std(crop),
        *[np.mean(rows[i:i + 4]) for i in (0, 4, 8, 12)],
        *[np.mean(cols[i:i + 4]) for i in (0, 4, 8, 12)],
        np.mean(np.abs(np.diff(rows))), np.mean(np.abs(np.diff(cols))),
        np.mean(padded[:8, :8]), np.mean(padded[:8, 8:]), np.mean(padded[8:, :]),
    ], dtype=np.float64)


class TextRegionDetector:
    """Implement Stage 1: text-like visual grouping, region confidence and attention, not semantic OCR."""

    def __init__(self, config: SensoryConfig) -> None:
        """Store visual filter, tile and threshold parameters; they are engineering settings, not neural constants."""
        self.config = config

    def step_01(self, state: dict[str, Any]) -> dict[str, Any]:
        """Receive pixels and estimate luminance, local variance and center-surround contrast (local normalization)."""
        request: SensoryRequest = state["request"]
        if request.visual_image is None:
            state["visual_available"] = False
            return {"status": "skipped_no_visual_input"}
        gray = grayscale_image(request.visual_image, self.config.epsilon)
        sigma = max(1.0, self.config.visual_tile_size / 8.0)
        mean = ndi.gaussian_filter(gray, sigma=sigma)
        variance = np.maximum(ndi.gaussian_filter(gray**2, sigma=sigma) - mean**2, 0.0)
        contrast = (gray - mean) / (np.sqrt(variance) + self.config.epsilon)
        state.update({"visual_available": True, "gray": gray, "local_luminance": mean,
                      "local_variance": variance, "local_contrast": contrast})
        return {"image_shape": list(gray.shape), "luminance": safe_summary(mean),
                "local_contrast": safe_summary(contrast)}

    def step_02(self, state: dict[str, Any]) -> dict[str, Any]:
        """Extract oriented edges and spatial-frequency channels using Gabor convolution and quadrature energy."""
        if not state.get("visual_available"):
            return {"status": "skipped_no_visual_input"}
        stack, summary = _gabor_features(state["local_contrast"], self.config)
        state.update({"visual_feature_stack": stack, "visual_filter_summary": summary})
        return {"filter_channels": int(stack.shape[0]), "feature_maps": safe_summary(stack),
                "filter_summary": summary}

    def step_03(self, state: dict[str, Any]) -> dict[str, Any]:
        """Detect repeated text-like spatial organization through tile features and logistic population scores."""
        if not state.get("visual_available"):
            return {"status": "skipped_no_visual_input"}
        records = _tile_feature_records(state["visual_feature_stack"], state["local_contrast"],
                                        self.config.visual_tile_size)
        matrix = np.asarray([[r["edge_density"], r["orientation_entropy"], r["spatial_frequency_index"],
                              r["element_density"], r["contrast_sd"]] for r in records])
        z = np.clip((matrix - np.mean(matrix, axis=0)) /
                    (np.std(matrix, axis=0) + self.config.epsilon), -4.0, 4.0)
        probabilities = sigmoid(z @ np.asarray([0.65, 0.65, 0.35, 0.75, 0.65]) - 0.15)
        for record, probability, features in zip(records, probabilities, z, strict=True):
            record["text_probability"], record["z_features"] = float(probability), features
        state["tile_records"] = records
        return {"tile_count": len(records), "tile_text_probabilities": probabilities,
                "feature_dimensions": int(matrix.shape[1]) if matrix.ndim == 2 else 0}

    def step_04(self, state: dict[str, Any]) -> dict[str, Any]:
        """Group candidate tiles by spatial proximity, baseline alignment and common region membership."""
        if not state.get("visual_available"):
            return {"status": "skipped_no_visual_input"}
        candidates = [r for r in state.get("tile_records", [])
                      if r["text_probability"] >= max(0.40, self.config.visual_text_threshold - 0.18)]
        remaining = {r["tile_id"]: r for r in candidates}
        groups = []
        while remaining:
            _, first = remaining.popitem()
            group, changed = [first], True
            while changed:
                changed = False
                for key, item in list(remaining.items()):
                    x, y = item["center"]
                    linked = False
                    for other in group:
                        ox, oy = other["center"]
                        distance = np.exp(-((x - ox)**2 + (y - oy)**2) /
                                          (2.0 * (1.8 * self.config.visual_tile_size)**2))
                        alignment = np.exp(-abs(y - oy) / max(self.config.visual_tile_size, 1))
                        if 0.45 * distance + 0.30 * alignment + 0.25 >= 0.26:
                            linked = True
                            break
                    if linked:
                        group.append(item)
                        remaining.pop(key)
                        changed = True
            groups.append(group)
        state["tile_groups"] = groups
        return {"candidate_tile_count": len(candidates), "group_count": len(groups),
                "grouping_rule": "distance + alignment + approximate height similarity"}

    def step_05(self, state: dict[str, Any]) -> dict[str, Any]:
        """Estimate text-region boundaries by group bounding-box union and pooled tile confidence."""
        if not state.get("visual_available"):
            return {"status": "skipped_no_visual_input"}
        height, width = state["gray"].shape
        regions = []
        for group in state.get("tile_groups", []):
            boxes = [item["bbox"] for item in group]
            y0, x0 = max(0, min(b[0] for b in boxes)), max(0, min(b[1] for b in boxes))
            y1, x1 = min(height, max(b[2] for b in boxes)), min(width, max(b[3] for b in boxes))
            if y1 > y0 and x1 > x0:
                regions.append({"region_id": len(regions), "bbox": (y0, x0, y1, x1),
                                "confidence": float(np.mean([x["text_probability"] for x in group])),
                                "tile_count": len(group)})
        regions.sort(key=lambda r: r["confidence"], reverse=True)
        state["text_regions"] = regions[:self.config.max_text_regions]
        return {"region_count": len(state["text_regions"]), "regions": state["text_regions"],
                "status": "ok" if regions else "no_text_like_regions_detected"}

    def step_06(self, state: dict[str, Any]) -> dict[str, Any]:
        """Allocate normalized competitive attention using regional confidence and optional caller-supplied gain."""
        if not state.get("visual_available"):
            return {"status": "skipped_no_visual_input"}
        regions = state.get("text_regions", [])
        if not regions:
            state.update({"attended_regions": [], "visual_attention_weights": np.zeros(0)})
            return {"attention_weights": [], "status": "no_regions_to_attend"}
        external = state["request"].visual_attention or {}
        logits = [np.log(max(r["confidence"], self.config.epsilon) /
                         max(1 - r["confidence"], self.config.epsilon)) +
                  float(external.get(str(r["region_id"]), 0.0)) for r in regions]
        weights = softmax(np.asarray(logits))
        attended = [dict(r, attention_weight=float(w)) for r, w in zip(regions, weights, strict=True)]
        state.update({"attended_regions": attended, "visual_attention_weights": weights})
        return {"attention_weights": weights, "priority_order": [r["region_id"] for r in attended]}


class CharacterSegmentationProcessor:
    """Implement Stage 2 as overlapping character-feature grouping rather than conventional one-box-per-letter OCR."""

    def __init__(self, config: SensoryConfig) -> None:
        """Store candidate limits and scale settings while preserving alternative segmentations."""
        self.config = config

    def step_07(self, state: dict[str, Any]) -> dict[str, Any]:
        """Detect local strokes and connected components using adaptive intensity and contrast thresholds."""
        if not state.get("visual_available"):
            return {"status": "skipped_no_visual_input"}
        gray = state["gray"]
        threshold = otsu_threshold(gray)
        contrast = np.abs(gray - ndi.gaussian_filter(gray, 2.0))
        ink = (gray < threshold) | (contrast > max(float(np.quantile(contrast, 0.78)), 0.05))
        parts = []
        for region in state.get("attended_regions", []):
            y0, x0, y1, x1 = region["bbox"]
            labels, _ = ndi.label(ink[y0:y1, x0:x1], structure=np.ones((3, 3), dtype=np.int8))
            for label_id, slices in enumerate(ndi.find_objects(labels), start=1):
                if slices is None:
                    continue
                ys, xs = slices
                py0, px0, py1, px1 = y0 + ys.start, x0 + xs.start, y0 + ys.stop, x0 + xs.stop
                component = labels[ys, xs] == label_id
                area = int(np.sum(component))
                if area < 2:
                    continue
                parts.append({"part_id": len(parts), "region_id": region["region_id"],
                              "bbox": (py0, px0, py1, px1), "area": area,
                              "height": py1 - py0, "width": px1 - px0,
                              "center": ((px0 + px1) / 2, (py0 + py1) / 2),
                              "density": float(area / max((py1-py0)*(px1-px0), 1)),
                              "component_mask": component})
        state["character_parts"] = parts[:self.config.max_character_candidates * 3]
        return {"component_count": len(state["character_parts"]), "ink_threshold": float(threshold),
                "local_shape_features": [{k: p[k] for k in ("part_id", "bbox", "area", "height", "width", "density")}
                                         for p in state["character_parts"][:128]]}

    def step_08(self, state: dict[str, Any]) -> dict[str, Any]:
        """Combine strokes into glyph candidates and compare 16-dimensional descriptors with supplied grapheme prototypes."""
        if not state.get("visual_available"):
            return {"status": "skipped_no_visual_input"}
        request: SensoryRequest = state["request"]
        groups = []
        for part in sorted(state.get("character_parts", []), key=lambda p: (p["region_id"], p["center"][0])):
            match = None
            for group in groups:
                anchor = group[-1]
                if anchor["region_id"] != part["region_id"]:
                    continue
                dx = part["center"][0] - anchor["center"][0]
                overlap = max(0.0, min(part["bbox"][2], anchor["bbox"][2]) -
                              max(part["bbox"][0], anchor["bbox"][0]))
                ratio = overlap / max(min(part["height"], anchor["height"]), 1)
                if dx <= 0.75 * max(part["height"], anchor["height"]) and ratio >= 0.25:
                    match = group
                    break
            if match is None:
                groups.append([part])
            else:
                match.append(part)
        glyphs = []
        for group in groups[:self.config.max_character_candidates]:
            y0, x0 = min(p["bbox"][0] for p in group), min(p["bbox"][1] for p in group)
            y1, x1 = max(p["bbox"][2] for p in group), max(p["bbox"][3] for p in group)
            mask = np.zeros(state["gray"].shape, dtype=bool)
            for part in group:
                py0, px0, py1, px1 = part["bbox"]
                mask[py0:py1, px0:px1] |= part["component_mask"]
            features = _descriptor(mask, (y0, x0, y1, x1))
            hypotheses = []
            for label, prototype in (request.grapheme_prototypes or {}).items():
                vector = np.asarray(prototype, dtype=np.float64).ravel()
                if vector.size == features.size and np.all(np.isfinite(vector)):
                    distance = float(np.mean((features - vector)**2))
                    hypotheses.append({"label": str(label), "score": float(np.exp(-distance / 0.08))})
            if hypotheses:
                probs = softmax(np.log(np.asarray([h["score"] for h in hypotheses]) + 1e-12))
                for item, probability in zip(hypotheses, probs, strict=True):
                    item["probability"] = float(probability)
            glyphs.append({"glyph_id": len(glyphs), "region_id": group[0]["region_id"],
                           "bbox": (y0, x0, y1, x1), "part_ids": [p["part_id"] for p in group],
                           "features": features, "hypotheses": hypotheses,
                           "segmentation_uncertainty": float(len(group) > 1)})
        state["glyph_hypotheses"] = glyphs
        return {"glyph_candidate_count": len(glyphs), "prototype_labels_supplied": len(request.grapheme_prototypes or {}),
                "known_label_count": sum(bool(g["hypotheses"]) for g in glyphs)}

    def step_09(self, state: dict[str, Any]) -> dict[str, Any]:
        """Retain both whole-glyph and candidate split interpretations using vertical-projection valleys."""
        if not state.get("visual_available"):
            return {"status": "skipped_no_visual_input"}
        alternatives = []
        for glyph in state.get("glyph_hypotheses", []):
            y0, x0, y1, x1 = glyph["bbox"]
            crop = state["gray"][y0:y1, x0:x1]
            projection = np.sum(crop < otsu_threshold(crop), axis=0).astype(float)
            cuts = []
            if projection.size >= 5:
                smooth = ndi.gaussian_filter1d(projection, sigma=1.0)
                minima = np.where((smooth[1:-1] <= smooth[:-2]) & (smooth[1:-1] < smooth[2:]))[0] + 1
                for cut in minima:
                    relative = cut / max(projection.size, 1)
                    depth = 1.0 - float(smooth[cut] / max(float(np.max(smooth)), 1.0))
                    if 0.18 <= relative <= 0.82 and depth >= 0.35:
                        cuts.append({"x": int(x0 + cut), "valley_depth": depth,
                                     "left_bbox": (y0, x0, y1, int(x0 + cut)),
                                     "right_bbox": (y0, int(x0 + cut), y1, x1)})
            glyph["segmentation_alternatives"] = cuts
            alternatives.append({"glyph_id": glyph["glyph_id"], "unsplit_bbox": glyph["bbox"],
                                 "alternative_cuts": cuts, "retained_unsplit_hypothesis": True})
        state["segmentation_alternatives"] = alternatives
        return {"candidate_groups": len(alternatives),
                "alternative_cut_count": sum(len(x["alternative_cuts"]) for x in alternatives),
                "policy": "retain unsplit and split alternatives"}

    def step_10(self, state: dict[str, Any]) -> dict[str, Any]:
        """Encode normalized spatial position and ordinal reading order while retaining ambiguous neighboring features."""
        if not state.get("visual_available"):
            return {"status": "skipped_no_visual_input"}
        ordered = sorted(state.get("glyph_hypotheses", []),
                         key=lambda g: (g["bbox"][0], g["bbox"][1]))
        height, width = state["gray"].shape
        positions = []
        for ordinal, glyph in enumerate(ordered):
            y0, x0, y1, x1 = glyph["bbox"]
            code = {"glyph_id": glyph["glyph_id"], "ordinal_position": ordinal,
                    "x_relative": float((x0 + x1) / (2 * max(width, 1))),
                    "y_relative": float((y0 + y1) / (2 * max(height, 1))),
                    "overlapping_alternatives": len(glyph.get("segmentation_alternatives", []))}
            glyph["position_code"] = code
            positions.append(code)
        state["ordered_glyphs"] = ordered
        return {"ordered_glyph_count": len(ordered), "position_codes": positions}

    def step_11(self, state: dict[str, Any]) -> dict[str, Any]:
        """Increase font and scale tolerance through shape normalization without mirror or letter-order invariance."""
        if not state.get("visual_available"):
            return {"status": "skipped_no_visual_input"}
        for glyph in state.get("ordered_glyphs", []):
            vector = np.asarray(glyph["features"], dtype=np.float64)
            glyph["invariant_features"] = vector / max(float(np.linalg.norm(vector)), self.config.epsilon)
            y0, x0, y1, x1 = glyph["bbox"]
            glyph["view_consistency"] = float(1 / (1 + (x1-x0) / max(y1-y0, 1)))
        return {"invariant_glyph_count": len(state.get("ordered_glyphs", [])),
                "invariance_scope": "scale/translation normalization; order and mirror distinctions retained"}

    def step_12(self, state: dict[str, Any]) -> dict[str, Any]:
        """Resolve glyph candidates with explicit orthographic priors using Bayesian log-evidence and entropy."""
        if not state.get("visual_available"):
            return {"status": "skipped_no_visual_input"}
        prior = state["request"].orthographic_prior or {}
        posteriors = []
        for glyph in state.get("ordered_glyphs", []):
            hypotheses = glyph.get("hypotheses", [])
            if not hypotheses:
                posteriors.append({"glyph_id": glyph["glyph_id"], "candidates": [],
                                   "status": "unlabelled_visual_shape"})
                continue
            scores = np.asarray([np.log(max(h.get("probability", 1e-8), 1e-8)) +
                                 np.log(max(float(prior.get(h["label"], 1.0)), 1e-8)) for h in hypotheses])
            probabilities = softmax(scores)
            candidates = [{"label": h["label"], "probability": float(p)}
                          for h, p in zip(hypotheses, probabilities, strict=True)]
            posteriors.append({"glyph_id": glyph["glyph_id"], "candidates": candidates,
                               "entropy": float(-np.sum(probabilities * np.log(probabilities + 1e-12))),
                               "status": "posterior_with_visual_and_optional_context"})
        state["letter_posteriors"] = posteriors
        return {"posterior_count": len(posteriors),
                "labelled_count": sum(bool(item["candidates"]) for item in posteriors),
                "context_used": bool(prior)}
