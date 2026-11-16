"""Functional implementations of VWFA steps 1–24, grouped into six documented subsystems."""
from __future__ import annotations

import hashlib
import math
from typing import Any, Sequence

import numpy as np
from scipy import ndimage as ndi

from .contracts import VWFAConfig, VWFARequest
from .math_ops import (
    connected_regions, contrast_normalize, cosine_similarity, divisive_normalize,
    extract_ordered_ngrams, gabor_energy_bank, morphology_segmentations,
    normalize_orthography, ordered_edit_similarity, pool_region_descriptor,
    safe_array_summary, softmax,
)


def hashed_orthographic_vector(value: str, dimensions: int = 64) -> np.ndarray:
    """Create a deterministic signed character n-gram vector for explicit orthographic input.

    Feature hashing is a reproducible engineering representation, not a literal neural code. It is
    used only for caller-supplied orthographic sequences or explicitly labelled learning examples.
    """
    normalized = normalize_orthography(value)
    vector = np.zeros(dimensions, dtype=np.float64)
    padded = f"^{normalized}$"
    for width in (1, 2, 3, 4):
        for start in range(max(0, len(padded) - width + 1)):
            token = padded[start:start + width].encode("utf-8")
            digest = hashlib.blake2b(token, digest_size=8).digest()
            integer = int.from_bytes(digest, "little")
            vector[integer % dimensions] += (1.0 if (integer >> 9) & 1 else -1.0) / math.sqrt(width)
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def _vector_mean(vectors: Sequence[np.ndarray], dimensions: int = 0) -> np.ndarray:
    """Average equal-dimensional feature vectors and return a defined zero vector for empty evidence."""
    if not vectors:
        return np.zeros(dimensions, dtype=np.float64)
    arrays = [np.asarray(item, dtype=np.float64).ravel() for item in vectors]
    if len({item.size for item in arrays}) != 1:
        raise ValueError("all vectors must have the same feature dimension")
    return np.mean(np.stack(arrays), axis=0)


class StageAProcessor:
    """Implement Stage A: luminance, contrast, oriented visual features, conjunctions, and spatial layout."""

    def __init__(self, config: VWFAConfig) -> None:
        """Store the numerical filter and local-contrast settings for early visual processing."""
        self.config = config

    def step_01(self, state: dict[str, Any]) -> dict[str, Any]:
        """Sample visual input and estimate local luminance, variance, contrast, and normalized contrast."""
        image = state["request"].visual_image
        if image is None:
            state.update({"visual_available": False, "normalized_image": None})
            return {"status": "upstream_orthography_only"}
        result = contrast_normalize(image, self.config.local_contrast_sigma, self.config.epsilon)
        state.update(result)
        state["visual_available"] = True
        return {"input_shape": list(result["grayscale"].shape),
                "normalized_image": safe_array_summary(result["normalized_image"]),
                "local_luminance": safe_array_summary(result["local_luminance"]),
                "local_variance": safe_array_summary(result["local_variance"])}

    def step_02(self, state: dict[str, Any]) -> dict[str, Any]:
        """Extract oriented multi-scale visual responses and quadrature energy with divisive gain control."""
        image = state.get("normalized_image")
        if image is None:
            state["gabor_stack"] = np.zeros((0, 0, 0))
            return {"status": "skipped_no_pixel_image", "feature_channels": 0}
        energy, responses = gabor_energy_bank(
            image, self.config.orientations, self.config.spatial_frequencies,
            self.config.gabor_size, self.config.gabor_sigma, self.config.gabor_gamma,
        )
        state["gabor_stack"] = divisive_normalize(energy, self.config.epsilon)
        state["gabor_responses"] = responses
        return {"orientation_frequency_channels": int(energy.shape[0]),
                "even_responses": safe_array_summary(responses["even_responses"]),
                "odd_responses": safe_array_summary(responses["odd_responses"]),
                "quadrature_energy": safe_array_summary(energy),
                "normalized_energy": safe_array_summary(state["gabor_stack"])}

    def step_03(self, state: dict[str, Any]) -> dict[str, Any]:
        """Combine simple visual channels into nonlinear edge, contour, corner, and shape conjunction evidence."""
        stack = state.get("gabor_stack")
        if stack is None or stack.size == 0:
            conjunction = np.zeros((0, 0), dtype=np.float64)
        else:
            total = np.sum(np.maximum(stack, 0.0), axis=0)
            maximum = np.max(stack, axis=0)
            coherence = ndi.gaussian_filter(maximum, 1.0, mode="reflect")
            conjunction = np.tanh(0.35 * total + 0.55 * maximum + 0.10 * coherence)
        state["conjunction_map"] = conjunction
        return {"conjunction_map": safe_array_summary(conjunction), "nonlinearity": "tanh"}

    def step_04(self, state: dict[str, Any]) -> dict[str, Any]:
        """Find candidate visual regions and preserve normalized positions without asserting OCR character identities."""
        stack = state.get("gabor_stack")
        regions = connected_regions(state["conjunction_map"]) if stack is not None and stack.size else []
        descriptors, positions = [], []
        height, width = (stack.shape[1], stack.shape[2]) if stack is not None and stack.size else (1, 1)
        for index, region in enumerate(regions):
            descriptors.append(pool_region_descriptor(stack, region, grid_size=4))
            y0, x0, y1, x1 = region["bbox"]
            positions.append({"index": index, "x_relative": float((x0 + x1) / (2 * max(width - 1, 1))),
                              "y_relative": float((y0 + y1) / (2 * max(height - 1, 1)))})
        state.update({"regions": regions, "visual_descriptors": descriptors, "position_codes": positions})
        return {"region_count": len(regions), "region_bounds": [list(r["bbox"]) for r in regions],
                "position_codes": positions, "descriptor_dimensions": int(descriptors[0].size) if descriptors else 0,
                "sequence_source": "not_inferred_by_stage_A",
                "upstream_symbols_available": bool(state["request"].orthographic_sequence)}


class StageBProcessor:
    """Implement Stage B: simple/hierarchical visual populations, pooling, position tolerance, and grapheme candidates."""

    def __init__(self, config: VWFAConfig) -> None:
        """Store settings for hierarchical receptive-field and population transformations."""
        self.config = config

    def step_05(self, state: dict[str, Any]) -> dict[str, Any]:
        """Construct distributed simple-feature population vectors from image descriptors or supplied symbols."""
        descriptors = state.get("visual_descriptors", [])
        source = "image-derived descriptors"
        if not descriptors and state["request"].orthographic_sequence:
            descriptors = [hashed_orthographic_vector(ch) for ch in state["request"].orthographic_sequence]
            source = "hashed upstream orthography"
        vectors = [np.asarray(v, dtype=np.float64) for v in descriptors]
        activity = np.maximum(np.stack(vectors), 0.0) if vectors else np.zeros((0, 0))
        if vectors:
            activity /= np.maximum(np.linalg.norm(activity, axis=1, keepdims=True), 1e-12)
        state.update({"simple_population_vectors": vectors, "simple_population_activity": activity,
                      "simple_population_source": source if vectors else "no_candidate_evidence"})
        return {"source": state["simple_population_source"], "population_matrix": safe_array_summary(activity),
                "population_count": int(activity.shape[0]), "feature_dimension": int(activity.shape[1]) if activity.ndim == 2 else 0}

    def step_06(self, state: dict[str, Any]) -> dict[str, Any]:
        """Compose simple population features through deterministic nonlinear hierarchy levels."""
        vectors = state.get("simple_population_vectors", [])
        if not vectors:
            state["hierarchical_vectors"] = []
            return {"status": "no_visual_or_upstream_population"}
        dimension = vectors[0].size
        rng = np.random.default_rng(self.config.random_seed + dimension)
        weight = rng.normal(0.0, 1.0 / math.sqrt(max(dimension, 1)), (dimension, dimension))
        hierarchy = [0.5 * np.tanh(weight @ v) + 0.5 * np.tanh(weight @ np.tanh(weight @ v)) for v in vectors]
        state["hierarchical_vectors"] = hierarchy
        return {"layer_count": 2, "weight_matrix_shape": list(weight.shape),
                "hierarchical_population": safe_array_summary(np.stack(hierarchy)), "activation": "tanh"}

    def step_07(self, state: dict[str, Any]) -> dict[str, Any]:
        """Pool local identity features for small visual changes while retaining an independent ordinal-position channel."""
        vectors = state.get("hierarchical_vectors", [])
        positions = state.get("position_codes", [])
        identity = (np.stack([0.5 * v + 0.5 * ndi.gaussian_filter1d(v, 1.0) for v in vectors])
                    if vectors else np.zeros((0, 0)))
        sequence = state["request"].orthographic_sequence
        combined = []
        for i, vector in enumerate(identity):
            if i < len(positions):
                xy = [positions[i]["x_relative"], positions[i]["y_relative"]]
            else:
                xy = [i / max(len(identity) - 1, 1), 0.5]
            combined.append(np.concatenate((vector, np.asarray(xy, dtype=np.float64))))
        state.update({"pooled_identity_features": identity, "position_preserving_features": combined})
        return {"identity_pool": safe_array_summary(identity), "position_channel_count": 2,
                "position_preserving_dimension": int(combined[0].size) if combined else 0,
                "order_channel_retained": True, "upstream_sequence_length": len(sequence)}

    def step_08(self, state: dict[str, Any]) -> dict[str, Any]:
        """Match image descriptors to supplied grapheme prototypes or preserve caller-provided orthography unchanged."""
        request = state["request"]
        sequence = list(request.orthographic_sequence)
        posteriors = []
        if sequence:
            posteriors = [{"position": i, "candidates": [{"grapheme": ch, "probability": 1.0}],
                           "unknown_mass": 0.0, "source": "caller_supplied_orthographic_sequence"}
                          for i, ch in enumerate(sequence)]
            status = "upstream_orthography_preserved"
        elif request.grapheme_prototypes:
            features = state.get("pooled_identity_features", np.zeros((0, 0)))
            for i, vector in enumerate(features):
                distances = [float(np.sum((vector - np.asarray(p.mean))**2 / np.asarray(p.variance)))
                             if len(p.mean) == vector.size else math.inf for p in request.grapheme_prototypes]
                finite = np.isfinite(distances)
                if np.any(finite):
                    labels = [p.label for p, ok in zip(request.grapheme_prototypes, finite) if ok]
                    probs = softmax(-0.5 * np.asarray(distances)[finite], self.config.population_temperature)
                    candidates = [{"grapheme": label, "probability": float(prob)} for label, prob in zip(labels, probs)]
                    posteriors.append({"position": i, "candidates": candidates, "unknown_mass": 0.0,
                                       "source": "caller_supplied_prototype_match"})
                else:
                    posteriors.append({"position": i, "candidates": [], "unknown_mass": 1.0,
                                       "source": "prototype_dimension_mismatch"})
            status = "prototype_match" if any(item["candidates"] for item in posteriors) else "unknown"
        else:
            posteriors = [{"position": i, "candidates": [], "unknown_mass": 1.0, "source": "no_labelled_prototypes"}
                          for i, _ in enumerate(state.get("position_preserving_features", []))]
            status = "awaiting_labelled_grapheme_prototypes"
        inferred = sequence or [max(x["candidates"], key=lambda c: c["probability"])["grapheme"]
                                for x in posteriors if x["candidates"]]
        state.update({"grapheme_posteriors": posteriors, "orthographic_sequence": inferred,
                      "grapheme_recognition_status": status})
        return {"status": status, "posterior_positions": posteriors,
                "mean_unknown_mass": float(np.mean([x["unknown_mass"] for x in posteriors])) if posteriors else 1.0}


class StageCProcessor:
    """Implement Stage C: posterior, intermediate, and anterior VWFA representations and visual-form invariance."""

    def __init__(self, config: VWFAConfig) -> None:
        """Store parameters for posterior-to-anterior visual-word-form transformations."""
        self.config = config

    def step_09(self, state: dict[str, Any]) -> dict[str, Any]:
        """Activate posterior VWFA populations using detailed, position-sensitive grapheme and visual evidence."""
        vectors = state.get("hierarchical_vectors", [])
        posterior = np.stack(vectors) if vectors else np.zeros((0, 0))
        evidence = float(np.mean([max((c["probability"] for c in p["candidates"]), default=0.0)
                                  for p in state.get("grapheme_posteriors", [])])) if state.get("grapheme_posteriors") else 0.0
        state["posterior_vwfa_activity"] = posterior
        return {"posterior_activity": safe_array_summary(posterior),
                "symbol_count": len(state.get("orthographic_sequence", [])),
                "mean_grapheme_evidence": evidence, "visual_detail_retained": True}

    def step_10(self, state: dict[str, Any]) -> dict[str, Any]:
        """Build overlapping ordered n-grams and distance-weighted directed non-adjacent letter-pair activity."""
        sequence = state.get("orthographic_sequence", [])
        ngrams, pairs = extract_ordered_ngrams(sequence), []
        posteriors = state.get("grapheme_posteriors", [])
        for i, left in enumerate(sequence):
            for j in range(i + 1, len(sequence)):
                lp = max((c["probability"] for c in posteriors[i]["candidates"]), default=1.0) if i < len(posteriors) else 1.0
                rp = max((c["probability"] for c in posteriors[j]["candidates"]), default=1.0) if j < len(posteriors) else 1.0
                gap = j - i
                pairs.append({"left_position": i, "right_position": j, "pair": f"{left}{sequence[j]}",
                              "gap": gap, "kernel": math.exp(-gap / 2.0), "activation": lp * rp * math.exp(-gap / 2.0)})
        state.update({"sublexical_ngrams": ngrams, "directed_pair_activity": pairs})
        return {"ngram_count": len(ngrams), "ngrams": ngrams, "directed_pair_count": len(pairs), "pair_activity": pairs}

    def step_11(self, state: dict[str, Any]) -> dict[str, Any]:
        """Transform posterior and sublexical evidence into a more abstract anterior VWFA representation."""
        posterior = state.get("posterior_vwfa_activity", np.zeros((0, 0)))
        base = np.mean(posterior, axis=0) if posterior.size else hashed_orthographic_vector(
            "".join(state.get("orthographic_sequence", [])))
        pairs = state.get("directed_pair_activity", [])
        ngram_density = len(state.get("sublexical_ngrams", [])) / max(len(state.get("orthographic_sequence", [])), 1)
        pair_strength = float(np.mean([p["activation"] for p in pairs])) if pairs else 0.0
        anterior = np.tanh(base + 0.05 * ngram_density + 0.05 * pair_strength)
        state.update({"anterior_vwfa_activity": anterior, "abstract_orthographic_code": anterior})
        return {"anterior_activity": safe_array_summary(anterior), "input_dimension": int(base.size),
                "ngram_density": float(ngram_density), "directed_pair_mean": pair_strength}

    def step_12(self, state: dict[str, Any]) -> dict[str, Any]:
        """Estimate visual-view consistency while preserving orthographic identity, orientation, and letter order."""
        request = state["request"]
        view_codes = []
        for view in request.visual_views:
            local = contrast_normalize(view, self.config.local_contrast_sigma, self.config.epsilon)
            energy, _ = gabor_energy_bank(local["normalized_image"], self.config.orientations,
                                           self.config.spatial_frequencies, self.config.gabor_size,
                                           self.config.gabor_sigma, self.config.gabor_gamma)
            view_codes.append(np.mean(energy, axis=(1, 2)))
        scores = [cosine_similarity(view_codes[0], code) for code in view_codes[1:]] if len(view_codes) >= 2 else []
        invariance = float(np.mean(scores)) if scores else 0.0
        state["invariance_score"] = invariance
        normalized = normalize_orthography("".join(state.get("orthographic_sequence", [])))
        state["orthographic_identity_code"] = normalized
        return {"status": "estimated_from_supplied_visual_views" if scores else "not_measured_without_two_visual_views",
                "view_similarities": scores, "invariance_score": invariance,
                "normalized_orthography": normalized, "ordinal_sequence_preserved": True, "mirror_invariance": False}


class StageDProcessor:
    """Implement Stage D: whole-word, morphemic, semantic, and recurrent lexical representations."""

    def __init__(self, config: VWFAConfig) -> None:
        """Store lexical competition, morphology-search, and recurrent-settling parameters."""
        self.config = config

    def step_13(self, state: dict[str, Any]) -> dict[str, Any]:
        """Score supplied whole-word forms by ordered similarity and a frequency prior without inventing lexicon entries."""
        observed, candidates = state.get("orthographic_identity_code", ""), []
        for entry in state["request"].lexicon:
            similarity = ordered_edit_similarity(observed, entry.word)
            score = similarity + 0.05 * math.log1p(entry.frequency)
            candidates.append({"word": entry.word, "similarity": similarity, "frequency": float(entry.frequency),
                               "score": score, "source_reference": entry.source_reference,
                               "semantic_features": dict(entry.semantic_features)})
        probabilities = softmax(np.asarray([x["score"] for x in candidates]), self.config.population_temperature) if candidates else np.zeros(0)
        for item, probability in zip(candidates, probabilities):
            item["probability"] = float(probability)
        candidates.sort(key=lambda x: x["probability"], reverse=True)
        state["whole_word_candidates"] = candidates
        state["whole_word_posterior"] = {x["word"]: x["probability"] for x in candidates}
        return {"candidate_count": len(candidates), "candidates": candidates,
                "posterior_entropy": float(-np.sum(probabilities * np.log(probabilities + 1e-12))) if probabilities.size else 0.0,
                "status": "ranked_caller_supplied_lexicon" if candidates else "no_lexicon_supplied"}

    def step_14(self, state: dict[str, Any]) -> dict[str, Any]:
        """Find candidate morpheme boundaries using only the caller-supplied concatenative inventory."""
        observed = state.get("orthographic_identity_code", "")
        if not observed and state.get("whole_word_candidates"):
            observed = state["whole_word_candidates"][0]["word"]
        analyses = morphology_segmentations(observed, state["request"].morphology_inventory,
                                             self.config.max_morphology_depth)
        for item in analyses:
            item["form_score"] = 1.0
            item["log_posterior_score"] = float(item["frequency_score"])
        state["morphology_candidates"] = analyses
        return {"candidate_count": len(analyses), "analyses": analyses,
                "status": "analyses_from_supplied_inventory" if analyses else "no_supported_decomposition"}

    def step_15(self, state: dict[str, Any]) -> dict[str, Any]:
        """Maintain concurrent whole-word, stem, prefix, suffix, and inflection activation channels."""
        whole = float(max(state.get("whole_word_posterior", {}).values(), default=0.0))
        roles = {"stem": 0.0, "prefix": 0.0, "suffix": 0.0, "inflection": 0.0, "unspecified": 0.0}
        analyses = state.get("morphology_candidates", [])
        probabilities = softmax(np.asarray([x["frequency_score"] for x in analyses]), self.config.population_temperature) if analyses else np.zeros(0)
        for analysis, probability in zip(analyses, probabilities):
            for role in analysis["roles"]:
                roles[role] = max(roles.get(role, 0.0), float(probability))
        combined = {"whole_word": whole, **roles}
        total = sum(combined.values())
        if total > 0.0:
            combined = {key: value / total for key, value in combined.items()}
        state["parallel_morphology_activity"] = combined
        return {"parallel_populations": combined, "whole_word_and_morphemes_coactive": bool(analyses and whole > 0.0)}

    def step_16(self, state: dict[str, Any]) -> dict[str, Any]:
        """Compose only caller-supplied lexical and morpheme semantic/grammatical attributes in the wider language network."""
        request, sums, weights, grammar, sources = state["request"], {}, {}, {}, []
        for item in state.get("whole_word_candidates", [])[:1]:
            for key, value in item["semantic_features"].items():
                sums[key] = sums.get(key, 0.0) + item["probability"] * value
                weights[key] = weights.get(key, 0.0) + item["probability"]
            sources.append("supplied_lexical_entry")
        inventory = {item.form: item for item in request.morphology_inventory}
        for analysis in state.get("morphology_candidates", [])[:3]:
            for form, sem, gram in zip(analysis["morphemes"], analysis["semantic_features"], analysis["grammatical_features"]):
                if form not in inventory:
                    continue
                for key, value in sem.items():
                    sums[key], weights[key] = sums.get(key, 0.0) + value, weights.get(key, 0.0) + 1.0
                for key, value in gram.items():
                    grammar[key] = max(grammar.get(key, 0.0), value)
                if sem or gram:
                    sources.append(f"supplied_morpheme:{form}")
        composed = {key: sums[key] / max(weights[key], 1e-12) for key in sums}
        state.update({"composed_language_features": composed, "grammatical_features": grammar})
        return {"semantic_features": composed, "grammatical_features": grammar,
                "evidence_sources": sorted(set(sources)), "invented_meanings": False}

    def step_17(self, state: dict[str, Any]) -> dict[str, Any]:
        """Stabilize competing word candidates with recurrent evidence, context, and convergence checks."""
        candidates = state.get("whole_word_candidates", [])
        if not candidates:
            state.update({"stabilized_candidates": [], "selected_candidate": None})
            return {"status": "abstained_no_lexical_candidates", "converged": True, "iterations": 0}
        base = np.log(np.maximum([x["probability"] for x in candidates], 1e-12))
        context = state["request"].context_features
        scores = np.asarray([sum(context.get(k, 0.0) * v for k, v in x["semantic_features"].items())
                             for x in candidates], dtype=np.float64)
        activity, converged, iterations = softmax(base + 0.2 * scores, self.config.population_temperature), False, 0
        for iterations in range(1, self.config.recurrent_steps + 1):
            prediction = softmax(base + 0.2 * scores + 0.15 * np.log(activity + 1e-12), self.config.population_temperature)
            updated = (1.0 - self.config.recurrent_damping) * activity + self.config.recurrent_damping * prediction
            if np.linalg.norm(updated - activity) < self.config.convergence_tolerance:
                activity, converged = updated, True
                break
            activity = updated
        ranked = [{"word": x["word"], "probability": float(p), "context_score": float(s)}
                  for x, p, s in zip(candidates, activity, scores)]
        ranked.sort(key=lambda x: x["probability"], reverse=True)
        confidence = ranked[0]["probability"]
        margin = confidence - ranked[1]["probability"] if len(ranked) > 1 else confidence
        selected = ranked[0]["word"] if confidence >= 0.55 and margin >= 0.05 else None
        state.update({"stabilized_candidates": ranked, "selected_candidate": selected})
        return {"status": "settled" if converged else "iteration_limit_reached", "converged": converged,
                "iterations": iterations, "ranked_candidates": ranked, "selected_candidate": selected,
                "top_confidence": confidence, "top_two_margin": margin}


class StageEProcessor:
    """Implement Stage E: feedforward visual connectivity, cortical language graph, feedback, and attention."""

    def __init__(self, config: VWFAConfig) -> None:
        """Store recurrent graph and attention settings for distributed reading-network calculations."""
        self.config = config
        self.nodes = ("retina_lgn", "v1", "v2", "v4", "vwfa_posterior", "vwfa_anterior",
                      "ifg", "mtg", "atl", "stg", "angular_gyrus", "parietal")
        self.edges = {
            "retina_lgn": {"v1": 0.65}, "v1": {"v2": 0.55}, "v2": {"v4": 0.50},
            "v4": {"vwfa_posterior": 0.60, "parietal": 0.15},
            "vwfa_posterior": {"vwfa_anterior": 0.55, "mtg": 0.20},
            "vwfa_anterior": {"mtg": 0.30, "ifg": 0.18, "atl": 0.16},
            "mtg": {"atl": 0.22, "ifg": 0.15, "stg": 0.12},
            "ifg": {"vwfa_anterior": 0.12, "parietal": 0.10},
            "atl": {"mtg": 0.10, "angular_gyrus": 0.12}, "stg": {"mtg": 0.10},
            "angular_gyrus": {"atl": 0.08, "parietal": 0.10}, "parietal": {"ifg": 0.08, "v4": 0.05},
        }

    def _adjacency(self) -> np.ndarray:
        """Build a row-normalized directed adjacency matrix from the named regional pathway graph."""
        matrix = np.zeros((len(self.nodes), len(self.nodes)))
        index = {name: i for i, name in enumerate(self.nodes)}
        for source, targets in self.edges.items():
            for target, weight in targets.items():
                matrix[index[target], index[source]] = weight
        return matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1.0)

    def step_18(self, state: dict[str, Any]) -> dict[str, Any]:
        """Propagate early visual evidence toward posterior VWFA over a damped feedforward graph."""
        adjacency = self._adjacency()
        feature = np.asarray(state.get("anterior_vwfa_activity", np.zeros(1)))
        strength = float(np.mean(np.abs(feature))) if feature.size else 0.0
        drive = np.zeros(len(self.nodes))
        drive[0] = strength if state.get("visual_available") else float(bool(state.get("orthographic_sequence"))) * 0.2
        activity = np.zeros_like(drive)
        for _ in range(4):
            activity = np.tanh(adjacency @ activity + drive)
        state.update({"bottom_up_network_state": dict(zip(self.nodes, activity.tolist())),
                      "network_adjacency": adjacency})
        return {"regions": state["bottom_up_network_state"], "feedforward_trace": activity.tolist(),
                "adjacency_shape": list(adjacency.shape)}

    def step_19(self, state: dict[str, Any]) -> dict[str, Any]:
        """Exchange VWFA evidence with IFG, MTG, ATL, STG, angular-gyrus, and parietal model populations."""
        adjacency = state.get("network_adjacency", self._adjacency())
        previous = np.asarray([state.get("bottom_up_network_state", {}).get(name, 0.0) for name in self.nodes])
        language = np.zeros(len(self.nodes))
        lexical = max(state.get("whole_word_posterior", {}).values(), default=0.0)
        semantic = float(np.mean(list(state.get("composed_language_features", {}).values()))) if state.get("composed_language_features") else 0.0
        for i, name in enumerate(self.nodes):
            if name in {"ifg", "mtg", "atl", "stg", "angular_gyrus"}:
                language[i] = 0.5 * lexical + 0.5 * semantic
        activity, history = previous.copy(), [previous.copy()]
        for _ in range(5):
            activity = np.tanh(0.65 * previous + 0.35 * (adjacency @ activity + language))
            history.append(activity.copy())
        matrix = np.stack(history)
        connectivity = np.corrcoef(matrix, rowvar=False) if np.all(np.std(matrix, axis=0) > 1e-12) else np.zeros((len(self.nodes), len(self.nodes)))
        connectivity = np.nan_to_num(connectivity, nan=0.0, posinf=0.0, neginf=0.0)
        state.update({"regional_activity": dict(zip(self.nodes, activity.tolist())),
                      "functional_connectivity": connectivity})
        return {"regional_activity": state["regional_activity"], "connectivity_matrix": safe_array_summary(connectivity),
                "connectivity_measure": "correlation_across_recurrent_update_iterations",
                "network_node_order": list(self.nodes)}

    def step_20(self, state: dict[str, Any]) -> dict[str, Any]:
        """Compute top-down orthographic predictions, residual error, precision, and bounded bottom-up feedback."""
        visual = np.asarray(state.get("anterior_vwfa_activity", np.zeros(1)))
        context = state["request"].context_features
        context_value = float(np.mean(list(context.values()))) if context else 0.0
        language = state.get("regional_activity", {})
        prediction = np.full(visual.shape, np.tanh(context_value + np.mean([language.get(n, 0.0) for n in ("ifg", "mtg", "atl")])))
        error = visual - prediction
        precision = 1.0 / (float(np.var(error)) + self.config.epsilon)
        refined = np.tanh(visual + 0.1 * (precision / (1.0 + precision)) * prediction)
        state.update({"prediction_error": error, "precision_weighted_state": refined,
                      "refined_orthographic_state": refined})
        return {"prediction": safe_array_summary(prediction), "prediction_error": safe_array_summary(error),
                "prediction_error_l2": float(np.linalg.norm(error)), "precision": precision,
                "refined_state": safe_array_summary(refined), "symbol_count": len(state.get("orthographic_sequence", []))}

    def step_21(self, state: dict[str, Any]) -> dict[str, Any]:
        """Allocate softmax attention over candidate positions and pool attended features with bounded task gain."""
        sequence, posteriors = state.get("orthographic_sequence", []), state.get("grapheme_posteriors", [])
        salience = np.asarray([
            max((c["probability"] for c in posteriors[i]["candidates"]), default=0.25)
            if i < len(posteriors) else 0.25 for i in range(len(sequence))
        ]) if sequence else np.asarray([float(np.mean(np.abs(state.get("refined_orthographic_state", [0.0]))))])
        attention = softmax(salience, temperature=0.5)
        features = state.get("position_preserving_features", [])
        attended = attention @ np.stack(features) if features and len(features) == len(attention) else np.asarray(state.get("refined_orthographic_state", [0.0]))
        context = state["request"].context_features
        gain = float(np.clip(1.0 + (np.mean(np.abs(list(context.values()))) if context else 0.0), 0.5, 2.0))
        state.update({"attention_weights": attention, "attended_representation": attended * gain,
                      "attended_position": int(np.argmax(attention)) if sequence else None})
        return {"attention_weights": attention.tolist(), "attended_position": state["attended_position"],
                "attended_representation": safe_array_summary(state["attended_representation"]), "task_gain": gain}


class StageFProcessor:
    """Implement Stage F: authorized literacy learning, word selectivity, and fast/slow consolidation."""

    def __init__(self, config: VWFAConfig) -> None:
        """Store learning rates without enabling weight updates unless each request explicitly authorizes them."""
        self.config = config

    def step_22(self, state: dict[str, Any], learning_state: dict[str, Any]) -> dict[str, Any]:
        """Learn visual-to-phonological associations using a delta rule and caller-supplied labelled targets."""
        request, examples = state["request"], state["request"].learning_examples
        if not examples:
            return {"status": "awaiting_explicit_learning_examples", "weights_updated": False}
        dimensions = {len(item.phonological_features) for item in examples}
        if len(dimensions) != 1:
            raise ValueError("all phonological learning targets must have the same dimension")
        target_size = dimensions.pop()
        weights = learning_state.get("visual_phonological_weights")
        if weights is None:
            weights = np.zeros((64, target_size))
        if weights.shape != (64, target_size):
            raise ValueError("phonological target dimension changed; reset learning state explicitly")
        delta, losses = np.zeros_like(weights), []
        for example in examples:
            visual = hashed_orthographic_vector(example.orthographic_form)
            target = np.asarray(example.phonological_features)
            error = target - visual @ weights
            delta += np.outer(visual, error)
            losses.append(float(np.mean(error**2)))
        if request.learning_authorized:
            weights = weights + self.config.learning_rate * delta / len(examples)
            learning_state["visual_phonological_weights"] = weights
        state["learning_weights_preview"] = weights
        return {"status": "updated" if request.learning_authorized else "preview_only_learning_not_authorized",
                "weights_updated": bool(request.learning_authorized), "example_count": len(examples),
                "mean_squared_prediction_error": float(np.mean(losses)), "weight_shape": list(weights.shape),
                "delta_norm": float(np.linalg.norm(delta / len(examples)))}

    def step_23(self, state: dict[str, Any], learning_state: dict[str, Any]) -> dict[str, Any]:
        """Estimate word-versus-control population selectivity and contrastive loss from labelled examples."""
        examples = state["request"].learning_examples
        words = [hashed_orthographic_vector(x.orthographic_form) for x in examples if x.category == "word"]
        controls = [hashed_orthographic_vector(x.orthographic_form) for x in examples if x.category == "control"]
        if not words or not controls:
            return {"status": "needs_word_and_control_labels", "word_selectivity": None,
                    "specialization_loss": None, "weights_updated": False}
        word_mean, control_mean = _vector_mean(words), _vector_mean(controls)
        selectivity = word_mean - control_mean
        probabilities = softmax(np.asarray([word_mean @ word_mean, word_mean @ control_mean]),
                                 self.config.population_temperature)
        loss = float(-math.log(max(float(probabilities[0]), 1e-12)))
        if state["request"].learning_authorized:
            previous = learning_state.get("word_selectivity_vector", np.zeros(64))
            learning_state["word_selectivity_vector"] = previous + self.config.learning_rate * selectivity
        state["word_selectivity"] = selectivity
        return {"status": "estimated_from_labelled_word_control_examples",
                "word_selectivity": safe_array_summary(selectivity), "selectivity_norm": float(np.linalg.norm(selectivity)),
                "specialization_loss": loss, "word_response_proxy": float(word_mean @ selectivity),
                "control_response_proxy": float(control_mean @ selectivity),
                "weights_updated": bool(state["request"].learning_authorized)}

    def step_24(self, state: dict[str, Any], learning_state: dict[str, Any]) -> dict[str, Any]:
        """Consolidate authorized orthographic experiences into fast and slow prototype memories using exponential averages."""
        examples = [x for x in state["request"].learning_examples if x.category == "word"]
        if not examples:
            return {"status": "awaiting_word_learning_examples", "memory_updated": False,
                    "fast_memory_count": len(learning_state.get("fast_prototypes", {})),
                    "slow_memory_count": len(learning_state.get("slow_prototypes", {}))}
        if not state["request"].learning_authorized:
            return {"status": "preview_only_learning_not_authorized", "memory_updated": False,
                    "example_count": len(examples), "fast_memory_count": len(learning_state.get("fast_prototypes", {})),
                    "slow_memory_count": len(learning_state.get("slow_prototypes", {}))}
        fast, slow = learning_state.setdefault("fast_prototypes", {}), learning_state.setdefault("slow_prototypes", {})
        for example in examples:
            vector, key = hashed_orthographic_vector(example.orthographic_form), normalize_orthography(example.orthographic_form)
            fast[key] = (1.0 - self.config.learning_rate) * fast.get(key, vector) + self.config.learning_rate * vector
            slow[key] = (1.0 - self.config.consolidation_rate) * slow.get(key, fast[key]) + self.config.consolidation_rate * fast[key]
        return {"status": "consolidated", "memory_updated": True, "example_count": len(examples),
                "fast_memory_count": len(fast), "slow_memory_count": len(slow),
                "fast_prototype_norms": {k: float(np.linalg.norm(v)) for k, v in fast.items()},
                "slow_prototype_norms": {k: float(np.linalg.norm(v)) for k, v in slow.items()}}
