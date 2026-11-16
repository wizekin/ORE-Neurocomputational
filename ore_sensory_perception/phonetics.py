"""Phonetic feature Stage 5: infer speech-relevant articulatory properties from acoustic evidence."""
from __future__ import annotations

from typing import Any
import numpy as np

from .contracts import SensoryConfig, SensoryRequest
from .math_ops import safe_summary, sigmoid, softmax


_MANNERS = ("stop", "fricative", "nasal", "approximant", "affricate", "vowel")
_PLACES = ("bilabial", "labiodental", "alveolar", "postalveolar", "palatal", "velar", "glottal", "unknown")


def _robust_normalize(values: np.ndarray, center: float | None = None,
                      scale: float | None = None, epsilon: float = 1e-8) -> tuple[np.ndarray, tuple[float, float]]:
    """Use median/MAD (or a supplied speaker baseline) to reduce scale differences across speakers."""
    x = np.asarray(values, dtype=np.float64)
    finite = x[np.isfinite(x)]
    mu = float(center) if center is not None else (float(np.median(finite)) if finite.size else 0.0)
    mad = float(scale) if scale is not None else (
        1.4826 * float(np.median(np.abs(finite - mu))) if finite.size else 1.0
    )
    if mad <= epsilon:
        mad = float(np.std(finite)) if finite.size else 1.0
    used = max(abs(mad), epsilon)
    return np.clip((x - mu) / used, -8.0, 8.0), (mu, used)


def _feature_posterior(features: np.ndarray, centers: np.ndarray, scale: float) -> np.ndarray:
    """Decode distributed feature vectors with Gaussian radial-basis population tuning and softmax competition."""
    distance = np.mean((features[:, None, :] - centers[None, :, :])**2, axis=2)
    return softmax(-distance / max(scale, 1e-6), axis=1)


class PhoneticFeatureDetector:
    """Implement Stage 5 as uncertain voicing, manner, place, vowel and coarticulatory population evidence."""

    def __init__(self, config: SensoryConfig) -> None:
        """Store feature temperature and normalization settings for the phonetic population decoder."""
        self.config = config

    def step_25(self, state: dict[str, Any]) -> dict[str, Any]:
        """Detect voicing from periodicity, low-frequency energy and release-to-voicing timing."""
        if not state.get("auditory_available"):
            return {"status": "skipped_no_auditory_input"}
        acoustic = state["acoustic_feature_matrix"]
        rms, periodic = acoustic[:, 4], np.clip(acoustic[:, 6], 0.0, 1.0)
        energy = np.asarray(state.get("frame_energy", rms**2))
        low = np.asarray(state.get("low_frequency_energy", np.zeros_like(rms)))
        low_ratio = np.resize(low / np.maximum(energy, self.config.epsilon), rms.shape)
        rate = state["sample_rate_hz"]
        vot = state.get("vot_seconds", float("nan"))
        temporal_voicing = 0.5 if not np.isfinite(vot) else float(sigmoid((0.025 - vot) / 0.012))
        logits = 2.0 * periodic + 1.1 * np.clip(low_ratio, 0, 1) + temporal_voicing - 1.2
        probabilities = sigmoid(logits)
        state.update({"voicing_probability": probabilities, "low_band_ratio": low_ratio,
                      "voicing_cues": np.column_stack((periodic, low_ratio, np.full(rms.shape, temporal_voicing)))})
        return {"voicing_probability": safe_summary(probabilities),
                "mean_voiced_evidence": float(np.mean(probabilities)),
                "cues": ["periodicity", "low-frequency energy", "voice-onset timing"],
                "interpretation": "probabilistic voicing evidence, not direct observation of vocal-fold vibration"}

    def step_26(self, state: dict[str, Any]) -> dict[str, Any]:
        """Detect manner by combining burst, frication, nasal, periodic and closure cues using population competition."""
        if not state.get("auditory_available"):
            return {"status": "skipped_no_auditory_input"}
        acoustic = state["acoustic_feature_matrix"]
        rms, centroid, periodic = acoustic[:, 4], acoustic[:, 5], np.clip(acoustic[:, 6], 0, 1)
        high = np.resize(state["high_frequency_energy"] / np.maximum(state["frame_energy"], self.config.epsilon), rms.shape)
        low = np.resize(state["low_frequency_energy"] / np.maximum(state["frame_energy"], self.config.epsilon), rms.shape)
        burst = np.maximum(np.gradient(rms), 0.0) if rms.size > 1 else np.zeros_like(rms)
        frication = np.clip(centroid / max(float(np.max(centroid)), 1.0), 0, 1) * (1-periodic) * high
        nasal, closure = low * periodic, np.clip(1-rms/max(float(np.max(rms)), self.config.epsilon), 0, 1)
        raw = np.column_stack((burst, frication, nasal, periodic, closure))
        normalized = np.column_stack([_robust_normalize(raw[:, i])[0] for i in range(raw.shape[1])])
        centers = np.asarray([
            [0.8, -0.3, -0.2, 0.2, 0.9], [-0.2, 1.0, -0.5, -0.8, -0.2],
            [-0.4, -0.4, 1.0, 0.9, -0.1], [-0.2, -0.5, -0.1, 0.8, -0.4],
            [0.8, 0.7, -0.3, 0.1, 0.5], [-0.3, -0.6, -0.2, 1.0, -0.5],
        ])
        posterior = _feature_posterior(normalized, centers, 2.5 * self.config.phoneme_temperature)
        state.update({"manner_features": normalized, "manner_posteriors": posterior, "manner_labels": _MANNERS})
        return {"manner_labels": list(_MANNERS), "manner_posteriors": safe_summary(posterior),
                "feature_schema": ["burst", "frication", "nasal_band", "periodicity", "closure"]}

    def step_27(self, state: dict[str, Any]) -> dict[str, Any]:
        """Infer uncertain place-of-articulation classes from spectral bursts and F1/F2 transitions."""
        if not state.get("auditory_available"):
            return {"status": "skipped_no_auditory_input"}
        acoustic = state["acoustic_feature_matrix"]
        formants = np.nan_to_num(np.asarray(state.get("formants_hz", np.zeros((acoustic.shape[0], 3)))), nan=0.0)
        if formants.shape[0] != acoustic.shape[0]:
            formants = np.resize(formants, (acoustic.shape[0], 3))
        df = np.gradient(formants, axis=0) if formants.shape[0] > 1 else np.zeros_like(formants)
        dc = np.gradient(acoustic[:, 5]) if acoustic.shape[0] > 1 else np.zeros(acoustic.shape[0])
        dp = np.gradient(acoustic[:, 6]) if acoustic.shape[0] > 1 else np.zeros(acoustic.shape[0])
        raw = np.column_stack((df[:, 0], df[:, 1], np.maximum(dc, 0), acoustic[:, 5], dp))
        features = np.column_stack([_robust_normalize(raw[:, i])[0] for i in range(raw.shape[1])])
        centers = np.asarray([
            [-0.7,-0.8,0.4,-0.5,0.0],[-0.4,0.1,0.2,0.0,0.1],[-0.2,0.4,0.1,0.1,0.2],
            [0.0,0.7,0.2,0.3,0.1],[0.3,0.8,0.6,0.8,0.3],[0.5,0.4,0.1,0.2,-0.1],
            [0.8,-0.5,0.4,0.5,-0.2],[-0.1,-0.1,0.0,-0.2,0.0],
        ])
        posterior = _feature_posterior(features, centers, 3.0 * self.config.phoneme_temperature)
        state.update({"place_features": features, "place_posteriors": posterior, "place_labels": _PLACES})
        return {"place_labels": list(_PLACES), "place_posteriors": safe_summary(posterior),
                "evidence_source": "formant transition and burst spectral shape"}

    def step_28(self, state: dict[str, Any]) -> dict[str, Any]:
        """Estimate vowel height/backness and a rounding tendency from F1/F2 with speaker variability retained."""
        if not state.get("auditory_available"):
            return {"status": "skipped_no_auditory_input"}
        formants = np.nan_to_num(np.asarray(state.get("formants_hz", np.zeros((0, 3)))), nan=0.0)
        f1, f2 = formants[:, 0], formants[:, 1]
        estimates = np.column_stack((np.clip(1-(f1-250)/650, 0, 1),
                                     np.clip(1-(f2-700)/1800, 0, 1),
                                     np.clip((1800-f2)/1200, 0, 1)))
        state["vowel_feature_estimates"] = estimates
        return {"vowel_feature_summary": safe_summary(estimates),
                "feature_order": ["height", "backness", "rounding_proxy"],
                "interpretation": "coarse formant-derived estimates, not universal articulatory labels"}

    def step_29(self, state: dict[str, Any]) -> dict[str, Any]:
        """Normalize acoustic-phonetic measurements with caller baselines or robust median/MAD statistics."""
        if not state.get("auditory_available"):
            return {"status": "skipped_no_auditory_input"}
        acoustic = np.asarray(state["acoustic_feature_matrix"], dtype=np.float64)
        baseline = state["request"].speaker_baseline or {}
        normalized, statistics = np.zeros_like(acoustic), {}
        for column in range(acoustic.shape[1]):
            name, value = f"feature_{column}", baseline.get(f"feature_{column}")
            if isinstance(value, (tuple, list, np.ndarray)) and len(value) >= 2:
                center, scale = float(value[0]), float(value[1])
                z, used = _robust_normalize(acoustic[:, column], center, scale)
            elif isinstance(value, (int, float)):
                z, used = _robust_normalize(acoustic[:, column], float(value))
            else:
                z, used = _robust_normalize(acoustic[:, column])
            normalized[:, column] = z
            statistics[name] = {"mean": used[0], "scale": used[1], "baseline_supplied": value is not None}
        state.update({"normalized_phonetic_features": np.nan_to_num(normalized),
                      "speaker_statistics": statistics})
        return {"normalized_features": safe_summary(normalized), "speaker_statistics": statistics}

    def step_30(self, state: dict[str, Any]) -> dict[str, Any]:
        """Represent coarticulation with preceding/current/following acoustic states and temporal differences."""
        if not state.get("auditory_available"):
            return {"status": "skipped_no_auditory_input"}
        features = state["normalized_phonetic_features"]
        previous = np.vstack((features[:1], features[:-1]))
        following = np.vstack((features[1:], features[-1:]))
        delta = following - previous
        trajectory = np.concatenate((features, previous, delta, following), axis=1)
        state.update({"coarticulatory_trajectory": trajectory, "coarticulatory_delta": delta})
        return {"trajectory": safe_summary(trajectory), "base_dimensions": int(features.shape[1]),
                "trajectory_dimensions": int(trajectory.shape[1]),
                "temporal_model": "current + preceding + following + temporal difference"}

    def step_31(self, state: dict[str, Any]) -> dict[str, Any]:
        """Combine phonetic cue groups into one normalized distributed auditory/STG population-code analogue."""
        if not state.get("auditory_available"):
            return {"status": "skipped_no_auditory_input"}
        population = np.concatenate((state["voicing_probability"][:, None], state["manner_posteriors"],
                                     state["place_posteriors"], state["vowel_feature_estimates"],
                                     state["coarticulatory_trajectory"]), axis=1)
        population /= np.maximum(np.linalg.norm(population, axis=1, keepdims=True), self.config.epsilon)
        state["phonetic_population_code"] = population
        return {"population_code": safe_summary(population), "population_dimensions": int(population.shape[1]),
                "population_groups": {"voicing": 1, "manner": len(_MANNERS), "place": len(_PLACES),
                                      "vowel": 3, "coarticulation": int(state["coarticulatory_trajectory"].shape[1])}}
