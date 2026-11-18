"""Shared mathematical helpers and early speech-perception stages for Point 16."""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .contracts import DevelopmentalConfig, DevelopmentalExperience, DevelopmentalLearnerState


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp a finite scalar to a closed interval."""
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("value must be finite")
    return max(low, min(high, value))


def sigmoid(value: float) -> float:
    """Compute a numerically stable logistic response."""
    value = float(value)
    if value >= 0.0:
        z = math.exp(-min(value, 700.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(value, -700.0))
    return z / (1.0 + z)


def softmax(scores: Sequence[float], temperature: float = 1.0) -> list[float]:
    """Normalize candidate scores into a stable probability distribution."""
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if not scores:
        return []
    scaled = [float(value) / temperature for value in scores]
    peak = max(scaled)
    values = [math.exp(max(-700.0, item - peak)) for item in scaled]
    total = sum(values)
    return [value / total for value in values]


def entropy(probabilities: Sequence[float]) -> float:
    """Compute Shannon uncertainty in nats for normalized or unnormalized probabilities."""
    total = sum(max(0.0, float(value)) for value in probabilities)
    if total <= 0.0:
        return 0.0
    return -sum(
        (float(value) / total) * math.log(float(value) / total)
        for value in probabilities if value > 0.0
    )


def _is_number(value: Any) -> bool:
    """Check whether a value is numeric and finite."""
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def vectorize(value: Any) -> list[float]:
    """Convert a mapping or sequence into a finite numeric feature vector."""
    if isinstance(value, Mapping):
        return [float(value[key]) for key in sorted(value) if _is_number(value[key])]
    if isinstance(value, (tuple, list)):
        return [float(item) for item in value if _is_number(item)]
    return []


def euclidean(left: Sequence[float], right: Sequence[float]) -> float:
    """Compute root-mean-square Euclidean distance for compatible vectors."""
    size = min(len(left), len(right))
    if size == 0:
        return 0.0
    return math.sqrt(sum((float(left[i]) - float(right[i])) ** 2 for i in range(size)) / size)


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Compute cosine similarity mapped to [0, 1], safely handling zero vectors."""
    size = min(len(left), len(right))
    if size == 0:
        return 0.0
    dot = sum(float(left[i]) * float(right[i]) for i in range(size))
    left_norm = math.sqrt(sum(float(left[i]) ** 2 for i in range(size)))
    right_norm = math.sqrt(sum(float(right[i]) ** 2 for i in range(size)))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return 0.0
    return clamp(0.5 + 0.5 * dot / (left_norm * right_norm))


def mean_vector(vectors: Sequence[Sequence[float]]) -> list[float]:
    """Return a component-wise mean for non-empty, compatible feature vectors."""
    usable = [list(map(float, vector)) for vector in vectors if vector]
    if not usable:
        return []
    width = min(len(item) for item in usable)
    return [sum(item[index] for item in usable) / len(usable) for index in range(width)]


def feature_overlap(left: Any, right: Any) -> float:
    """Measure overlap between normalized tokens or labels in nested evidence."""
    def flatten(value: Any) -> set[str]:
        if isinstance(value, Mapping):
            result = {str(key).lower() for key in value}
            for item in value.values():
                result |= flatten(item)
            return result
        if isinstance(value, (list, tuple, set)):
            result: set[str] = set()
            for item in value:
                result |= flatten(item)
            return result
        return {str(value).lower()} if value not in (None, "") else set()
    a, b = flatten(left), flatten(right)
    return len(a & b) / max(1, len(a | b))


def extract_acoustic_features(event: DevelopmentalExperience) -> dict[str, Any]:
    """Create overlapping auditory feature frames from supplied features or a raw waveform.

    Raw waveform support computes RMS, zero-crossing rate, and coarse DFT-band energies. It is a
    transparent feature front end, not a speech recognizer or auditory-neuron simulation.
    """
    if event.acoustic_frames:
        frames: list[list[float]] = []
        names: list[list[str]] = []
        for raw in event.acoustic_frames:
            if isinstance(raw, Mapping):
                keys = sorted(key for key, value in raw.items() if _is_number(value))
                names.append(keys)
                frames.append([float(raw[key]) for key in keys])
            else:
                vector = vectorize(raw)
                frames.append(vector)
                names.append([f"f{i}" for i in range(len(vector))])
        if frames and frames[0]:
            width = min(len(frame) for frame in frames)
            frames = [frame[:width] for frame in frames]
        return _summarize_frames(frames, "supplied_feature_frames", names)

    samples = list(event.audio_samples)
    if not samples:
        return _summarize_frames([], "no_acoustic_input", [])
    largest_power = 2 ** int(math.floor(math.log2(max(32, min(256, len(samples))))))
    frame_length = min(256, max(32, largest_power))
    hop = max(16, frame_length // 2)
    frames = []
    for start in range(0, max(1, len(samples) - frame_length + 1), hop):
        segment = samples[start:start + frame_length]
        if len(segment) < frame_length:
            segment += [0.0] * (frame_length - len(segment))
        rms = math.sqrt(sum(value * value for value in segment) / frame_length)
        crossings = sum(
            1 for i in range(1, len(segment))
            if (segment[i - 1] < 0.0 <= segment[i]) or (segment[i - 1] >= 0.0 > segment[i])
        ) / max(1, frame_length - 1)
        bands = [0.0, 0.0, 0.0, 0.0]
        for bin_index in range(1, 17):
            real = 0.0
            imag = 0.0
            for sample_index, value in enumerate(segment):
                angle = 2.0 * math.pi * bin_index * sample_index / frame_length
                real += value * math.cos(angle)
                imag -= value * math.sin(angle)
            bands[min(3, (bin_index - 1) // 4)] += real * real + imag * imag
        total = sum(bands) + 1e-12
        frames.append([rms, crossings] + [value / total for value in bands])
    names = [["rms", "zero_crossing_rate", "band_1", "band_2", "band_3", "band_4"] for _ in frames]
    return _summarize_frames(frames, "raw_waveform_frame_features", names)


def _summarize_frames(
    frames: Sequence[Sequence[float]],
    source: str,
    feature_names: Sequence[Sequence[str]],
) -> dict[str, Any]:
    """Compute temporal summaries from an ordered sequence of acoustic feature vectors."""
    normalized = [[float(v) for v in frame] for frame in frames if frame]
    if not normalized:
        return {
            "source": source, "frames": [], "frame_count": 0, "feature_dimension": 0,
            "temporal_change": 0.0, "uncertainty": 1.0, "feature_names": [],
        }
    width = min(len(frame) for frame in normalized)
    normalized = [frame[:width] for frame in normalized]
    changes = [euclidean(normalized[i], normalized[i - 1]) for i in range(1, len(normalized))]
    temporal_change = sum(changes) / max(1, len(changes))
    return {
        "source": source, "frames": normalized, "frame_count": len(normalized),
        "feature_dimension": width, "temporal_change": temporal_change,
        "uncertainty": clamp(1.0 / (1.0 + temporal_change)),
        "feature_names": list(feature_names[0][:width]) if feature_names else [],
        "mean_vector": mean_vector(normalized),
    }


def establish_broad_sensitivity(
    features: Mapping[str, Any],
    state: DevelopmentalLearnerState,
) -> dict[str, Any]:
    """Preserve broad acoustic discrimination before substantial environment-specific tuning."""
    frames = features.get("frames", ())
    distances = [euclidean(frames[index], frames[index - 1]) for index in range(1, len(frames))]
    observed_range = clamp(sum(distances) / max(1, len(distances)))
    sensitivity = clamp(0.75 * state.broad_sensitivity + 0.25 * max(0.45, observed_range))
    state.broad_sensitivity = sensitivity
    return {
        "broad_sensitivity": sensitivity, "observed_contrast_range": observed_range,
        "contrast_dimensions": int(features.get("feature_dimension", 0)),
        "feature_space_kept_multidimensional": True, "adult_phoneme_inventory_assumed": False,
    }


def build_provisional_phonetic_categories(
    features: Mapping[str, Any],
    event: DevelopmentalExperience,
    state: DevelopmentalLearnerState,
    config: DevelopmentalConfig,
) -> dict[str, Any]:
    """Update provisional sound prototypes using repeated examples and optional labels."""
    frames = [list(map(float, frame)) for frame in features.get("frames", ()) if frame]
    assignments: list[dict[str, Any]] = []
    created: list[str] = []
    for index, frame in enumerate(frames):
        label = event.phonetic_label or ""
        if label and label in state.phonetic_prototypes:
            key = label
        elif state.phonetic_prototypes:
            distances = {
                name: euclidean(frame, prototype)
                for name, prototype in state.phonetic_prototypes.items()
                if len(prototype) == len(frame)
            }
            key = min(distances, key=distances.get) if distances else ""
            if not key or distances[key] > config.category_creation_distance:
                key = label or f"provisional_{len(state.phonetic_prototypes) + 1:02d}"
                created.append(key)
        else:
            key = label or "provisional_01"
            created.append(key)
        previous = state.phonetic_prototypes.get(key, frame)
        count = state.phonetic_counts.get(key, 0)
        rate = config.prototype_rate / (1.0 + 0.04 * count)
        width = min(len(previous), len(frame))
        updated = [previous[i] + rate * (frame[i] - previous[i]) for i in range(width)]
        state.phonetic_prototypes[key] = updated
        state.phonetic_counts[key] = count + 1
        assignments.append({
            "frame_index": index, "category": key,
            "distance_before_update": euclidean(frame, previous), "exposure_count": count + 1,
        })
    confidence = (
        sum(1.0 / (1.0 + row["distance_before_update"]) for row in assignments) / len(assignments)
        if assignments else 0.0
    )
    return {
        "assignments": assignments, "created_categories": sorted(set(created)),
        "prototypes": dict(state.phonetic_prototypes), "category_confidence": clamp(confidence),
        "adult_category_structure_assumed": False,
    }


def accumulate_statistical_regularities(
    event: DevelopmentalExperience,
    state: DevelopmentalLearnerState,
    config: DevelopmentalConfig,
) -> dict[str, Any]:
    """Accumulate syllable frequencies and conditional transitions with smoothing."""
    syllables = list(event.syllables)
    for syllable in syllables:
        state.syllable_counts[syllable] = state.syllable_counts.get(syllable, 0) + 1
    for left, right in zip(syllables, syllables[1:]):
        key = left + "\\u241f" + right
        state.transition_counts[key] = state.transition_counts.get(key, 0) + 1
    total = sum(state.syllable_counts.values())
    vocab_count = max(1, len(state.syllable_counts))
    smoothing = config.statistical_smoothing
    frequencies = {
        token: (count + smoothing) / (total + smoothing * vocab_count)
        for token, count in state.syllable_counts.items()
    }
    transitions: dict[str, float] = {}
    for key, count in state.transition_counts.items():
        left = key.split("\\u241f", 1)[0]
        outgoing = sum(
            value for transition, value in state.transition_counts.items()
            if transition.startswith(left + "\\u241f")
        )
        transitions[key] = (count + smoothing) / (
            outgoing + smoothing * max(1, len(state.syllable_counts))
        )
    uncertainty = -sum(value * math.log(value) for value in frequencies.values() if value > 0.0)
    information_proxy = 0.0
    for key, probability in transitions.items():
        left, right = key.split("\\u241f", 1)
        p_left = frequencies.get(left, 1e-9)
        p_right = frequencies.get(right, 1e-9)
        information_proxy += probability * math.log(
            max(1e-9, probability) / max(1e-9, p_left * p_right)
        )
    return {
        "syllable_frequencies": frequencies, "transition_probabilities": transitions,
        "syllable_entropy": uncertainty, "transition_information_proxy": information_proxy,
        "observations": total, "smoothing": smoothing,
    }


def infer_word_boundaries(
    event: DevelopmentalExperience,
    statistics: Mapping[str, Any],
    state: DevelopmentalLearnerState,
    config: DevelopmentalConfig,
) -> dict[str, Any]:
    """Combine transitional probabilities with prosodic, stress, phonotactic, and context cues."""
    syllables = list(event.syllables)
    transitions = statistics.get("transition_probabilities", {})
    boundary_scores: list[dict[str, Any]] = []
    boundaries: list[int] = []
    for boundary in range(1, len(syllables)):
        key = syllables[boundary - 1] + "\\u241f" + syllables[boundary]
        probability = float(transitions.get(key, 1.0 / max(2, len(syllables))))
        transitional = clamp(-math.log(max(1e-9, probability)) / 5.0)
        cue = event.boundary_cues[boundary - 1] if boundary - 1 < len(event.boundary_cues) else {}
        pause = clamp(float(cue.get("pause", 0.0)))
        stress = clamp(float(cue.get("stress", 0.0)))
        phonotactic = clamp(float(cue.get("phonotactic_boundary", 0.0)))
        context = clamp(float(cue.get("context_boundary", 0.0)))
        score = clamp(0.45 * transitional + 0.20 * pause + 0.15 * stress
                      + 0.12 * phonotactic + 0.08 * context)
        state.word_boundary_counts[key] = state.word_boundary_counts.get(key, 0) + int(
            score >= config.segmentation_threshold
        )
        boundary_scores.append({
            "boundary_index": boundary, "left_syllable": syllables[boundary - 1],
            "right_syllable": syllables[boundary], "transition_probability": probability,
            "transitional_evidence": transitional, "pause": pause, "stress": stress,
            "phonotactic": phonotactic, "context": context, "score": score,
        })
        if score >= config.segmentation_threshold:
            boundaries.append(boundary)
    segments: list[list[str]] = []
    start = 0
    for boundary in boundaries:
        segments.append(syllables[start:boundary])
        start = boundary
    if syllables:
        segments.append(syllables[start:])
    words = ["".join(segment) for segment in segments]
    if not syllables and event.target_word:
        words = [event.target_word]
    return {
        "boundary_scores": boundary_scores, "boundary_indices": boundaries,
        "candidate_word_forms": words,
        "boundary_confidence": (
            sum(row["score"] for row in boundary_scores) / len(boundary_scores)
            if boundary_scores else 0.0
        ),
        "segmentation_is_probabilistic": True, "single_transition_rule_not_assumed": True,
    }


def tune_environment_specific_perception(
    event: DevelopmentalExperience,
    state: DevelopmentalLearnerState,
    config: DevelopmentalConfig,
) -> dict[str, Any]:
    """Tune native-language contrasts while retaining unfamiliar-category representations."""
    language = event.native_language or "environment"
    state.language_exposure_counts[language] = state.language_exposure_counts.get(language, 0) + 1
    feedback = event.phonetic_feedback
    native_updated: dict[str, float] = {}
    unfamiliar_updated: dict[str, float] = {}
    for observation in feedback.get("native_contrasts", ()):
        key = str(observation.get("contrast", "unspecified"))
        quality = clamp(float(observation.get("discrimination_success", 0.5)))
        previous = state.native_contrast_strengths.get(key, 0.5)
        updated = clamp(previous + config.prototype_rate * (quality - previous))
        state.native_contrast_strengths[key] = updated
        native_updated[key] = updated
    for observation in feedback.get("unfamiliar_contrasts", ()):
        key = str(observation.get("contrast", "unspecified"))
        quality = clamp(float(observation.get("discrimination_success", 0.5)))
        previous = state.unfamiliar_contrast_strengths.get(key, 0.5)
        updated = clamp(previous + 0.35 * config.prototype_rate * (quality - previous))
        state.unfamiliar_contrast_strengths[key] = updated
        unfamiliar_updated[key] = updated
    return {
        "environment": language, "exposure_count": state.language_exposure_counts[language],
        "native_contrast_strengths": native_updated,
        "unfamiliar_contrast_strengths": unfamiliar_updated,
        "unfamiliar_contrasts_are_not_deleted": True,
    }


def integrate_multimodal_referents(
    event: DevelopmentalExperience,
    config: DevelopmentalConfig,
) -> dict[str, Any]:
    """Combine speech, scene, gaze, gesture, and social contingency as uncertain referent evidence."""
    referents = list(event.scene_referents)
    if not referents:
        return {
            "referent_posterior": [], "selected_referent": None, "confidence": 0.0,
            "cue_confidence": {}, "ambiguity_preserved": True,
        }
    rows: list[dict[str, Any]] = []
    for referent in referents:
        identifier = str(referent.get("referent_id", referent.get("id", referent.get("label", ""))))
        label = str(referent.get("label", identifier)).lower()
        word_fit = 1.0 if event.target_word and event.target_word in label else 0.5
        gaze_fit = 1.0 if event.gaze_target and event.gaze_target == identifier else 0.0
        gesture_fit = 1.0 if event.gesture_target and event.gesture_target == identifier else 0.0
        intention_fit = 1.0 if event.speaker_intention and event.speaker_intention in str(
            referent.get("communicative_role", "")
        ).lower() else 0.0
        salience = clamp(float(referent.get("salience", 0.5)))
        social = event.social_contingency
        score = (0.20 * word_fit + 0.25 * gaze_fit + 0.20 * gesture_fit
                 + 0.10 * intention_fit + 0.15 * salience + 0.10 * social)
        rows.append({
            "referent_id": identifier, "label": label, "score": score,
            "cue_fits": {"speech": word_fit, "gaze": gaze_fit, "gesture": gesture_fit,
                         "intention": intention_fit, "scene_salience": salience,
                         "social_contingency": social},
        })
    probabilities = softmax([row["score"] for row in rows], config.multimodal_temperature)
    for row, probability in zip(rows, probabilities):
        row["probability"] = probability
    rows.sort(key=lambda item: item["probability"], reverse=True)
    confidence = rows[0]["probability"] if rows else 0.0
    return {
        "referent_posterior": rows,
        "selected_referent": rows[0]["referent_id"] if rows else None,
        "confidence": confidence,
        "cue_confidence": {
            "gaze": float(bool(event.gaze_target)), "gesture": float(bool(event.gesture_target)),
            "social_contingency": event.social_contingency, "speech": float(bool(event.target_word)),
        },
        "ambiguity_preserved": confidence < 0.75 or len(rows) > 1,
        "cue_independence_is_only_an_approximation": True,
    }
