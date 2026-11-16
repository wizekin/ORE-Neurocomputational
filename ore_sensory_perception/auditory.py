"""Auditory stages 3–4: cochlear-inspired transduction and acoustic feature analysis."""
from __future__ import annotations

from typing import Any
import numpy as np
from scipy import linalg, ndimage as ndi, signal

from .contracts import SensoryConfig, SensoryRequest
from .math_ops import normalized_autocorrelation, safe_summary, softmax


def _hz_to_erb(frequency: np.ndarray) -> np.ndarray:
    """Map hertz onto an ERB-like frequency scale to space model channels cochlearly."""
    return 21.4 * np.log10(1.0 + 0.00437 * np.asarray(frequency, dtype=np.float64))


def _erb_to_hz(value: np.ndarray) -> np.ndarray:
    """Convert ERB-like coordinates into hertz for filterbank center-frequency construction."""
    return (10.0 ** (np.asarray(value, dtype=np.float64) / 21.4) - 1.0) / 0.00437


def _lpc_formants(frame: np.ndarray, sample_rate_hz: float, order: int) -> list[float]:
    """Estimate F1–F3 from LPC polynomial roots; return NaNs when the signal has no stable formants."""
    x = np.asarray(frame, dtype=np.float64).ravel()
    if x.size < order + 4 or np.std(x) < 1e-7:
        return [float("nan")] * 3
    emphasized = signal.lfilter([1.0, -0.97], [1.0], x) * np.hamming(x.size)
    ac = signal.correlate(emphasized, emphasized, mode="full", method="fft")[x.size - 1:]
    if ac.size <= order or ac[0] <= 1e-12:
        return [float("nan")] * 3
    try:
        coefficients = np.linalg.solve(linalg.toeplitz(ac[:order]) + np.eye(order) * 1e-8,
                                       -ac[1:order + 1])
    except np.linalg.LinAlgError:
        return [float("nan")] * 3
    roots = np.roots(np.r_[1.0, coefficients])
    candidates = []
    for root in roots:
        if np.imag(root) <= 0 or abs(root) <= 0:
            continue
        frequency = float(np.angle(root) * sample_rate_hz / (2 * np.pi))
        bandwidth = float(-sample_rate_hz / np.pi * np.log(max(abs(root), 1e-12)))
        if 90 <= frequency <= min(5000.0, sample_rate_hz * 0.45) and 0 < bandwidth < 450:
            candidates.append(frequency)
    selected = sorted(candidates)[:3]
    return selected + [float("nan")] * (3 - len(selected))


class AuditorySignalProcessor:
    """Implement Stage 3: waveform reception, cochlear tonotopy, transduction, timing and auditory grouping."""

    def __init__(self, config: SensoryConfig) -> None:
        """Store channel and temporal parameters; they are computational settings, not cochlear measurements."""
        self.config = config

    def step_13(self, state: dict[str, Any]) -> dict[str, Any]:
        """Receive sampled sound, remove DC offset and preserve amplitude/time metadata (pressure-wave input)."""
        request: SensoryRequest = state["request"]
        if request.auditory_waveform is None:
            state["auditory_available"] = False
            return {"status": "skipped_no_auditory_input"}
        waveform = np.asarray(request.auditory_waveform, dtype=np.float64).ravel()
        waveform = waveform - float(np.mean(waveform))
        rate = float(request.sample_rate_hz)
        state.update({"auditory_available": True, "waveform": waveform,
                      "sample_rate_hz": rate, "sample_time_seconds": np.arange(waveform.size) / rate})
        return {"sample_count": int(waveform.size), "sample_rate_hz": rate,
                "duration_seconds": float(waveform.size / rate), "waveform_summary": safe_summary(waveform)}

    def step_14(self, state: dict[str, Any]) -> dict[str, Any]:
        """Separate sound into ERB-spaced bandpass channels as a proxy for basilar-membrane tonotopy."""
        if not state.get("auditory_available"):
            return {"status": "skipped_no_auditory_input"}
        waveform, rate = state["waveform"], state["sample_rate_hz"]
        nyquist = rate / 2
        low, high = min(self.config.cochlear_min_hz, nyquist * 0.2), min(self.config.cochlear_max_hz, nyquist * 0.94)
        if high <= low * 1.2:
            low, high = max(10.0, nyquist * 0.08), nyquist * 0.94
        points = np.linspace(_hz_to_erb(np.array([low]))[0], _hz_to_erb(np.array([high]))[0],
                             self.config.cochlear_channels + 2)
        edges = _erb_to_hz(points)
        channels, centers = [], []
        for i in range(self.config.cochlear_channels):
            lower, upper = max(5.0, float(edges[i])), min(nyquist * 0.98, float(edges[i + 2]))
            if upper <= lower * 1.03:
                continue
            sos = signal.butter(3, [lower, upper], btype="bandpass", fs=rate, output="sos")
            channels.append(signal.sosfilt(sos, waveform))
            centers.append(float(edges[i + 1]))
        matrix = np.stack(channels) if channels else np.zeros((0, waveform.size))
        state.update({"cochlear_channels": matrix, "cochlear_centers_hz": np.asarray(centers)})
        return {"channel_count": int(matrix.shape[0]), "center_frequencies_hz": centers,
                "channel_activity": safe_summary(matrix)}

    def step_15(self, state: dict[str, Any]) -> dict[str, Any]:
        """Rectify and compress channel envelopes into auditory-nerve firing-rate proxies."""
        if not state.get("auditory_available"):
            return {"status": "skipped_no_auditory_input"}
        channels = state.get("cochlear_channels", np.zeros((0, 0)))
        if not channels.size:
            state["auditory_nerve_rates"] = channels
            return {"status": "no_valid_cochlear_channels"}
        envelopes = np.abs(signal.hilbert(channels, axis=1))
        compressed = np.log1p(6.0 * envelopes)
        threshold = np.quantile(compressed, 0.12, axis=1, keepdims=True)
        rates = np.clip(compressed - threshold, 0.0, 1.0)
        rates /= np.maximum(np.max(rates, axis=1, keepdims=True), self.config.epsilon)
        state.update({"cochlear_envelopes": envelopes, "auditory_nerve_rates": rates})
        return {"rate_population_shape": list(rates.shape), "envelope_summary": safe_summary(envelopes),
                "rate_summary": safe_summary(rates),
                "transduction": "envelope rectification + log compression + threshold + saturation"}

    def step_16(self, state: dict[str, Any]) -> dict[str, Any]:
        """Preserve temporal envelopes, derivatives, onset events and synchronization across frequency channels."""
        if not state.get("auditory_available"):
            return {"status": "skipped_no_auditory_input"}
        rates = state.get("auditory_nerve_rates", np.zeros((0, 0)))
        if not rates.size:
            state.update({"temporal_envelopes": rates, "onset_times_seconds": np.zeros(0)})
            return {"status": "no_auditory_rate_activity"}
        envelopes = ndi.gaussian_filter1d(rates, sigma=max(1.0, 0.004 * state["sample_rate_hz"]), axis=1)
        derivative = np.diff(envelopes, axis=1, prepend=envelopes[:, :1])
        pooled = np.mean(envelopes, axis=0)
        change = np.diff(pooled, prepend=pooled[:1])
        peaks, _ = signal.find_peaks(change, height=max(float(np.quantile(change, 0.92)), 0.01),
                                     distance=max(1, int(0.015 * state["sample_rate_hz"])))
        onsets = peaks / state["sample_rate_hz"]
        state.update({"temporal_envelopes": envelopes, "temporal_derivative": derivative,
                      "onset_times_seconds": onsets, "pooled_auditory_envelope": pooled})
        return {"envelope_summary": safe_summary(envelopes), "onset_count": int(peaks.size),
                "onset_times_seconds": onsets, "temporal_change_summary": safe_summary(derivative)}

    def step_17(self, state: dict[str, Any]) -> dict[str, Any]:
        """Group coherent channels by envelope correlation and harmonicity, modeling auditory scene analysis."""
        if not state.get("auditory_available"):
            return {"status": "skipped_no_auditory_input"}
        envelopes, centers = state.get("temporal_envelopes", np.zeros((0, 0))), state.get("cochlear_centers_hz", np.zeros(0))
        if not envelopes.size:
            state["auditory_stream_candidates"] = []
            return {"status": "no_channels_to_group"}
        adjacency = np.nan_to_num(np.corrcoef(envelopes), nan=0.0) > 0.18 if envelopes.shape[0] > 1 else np.ones((1, 1), dtype=bool)
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                ratio = max(centers[i], centers[j]) / max(min(centers[i], centers[j]), 1e-6)
                harmonic = round(ratio) in (2, 3, 4) and abs(ratio - round(ratio)) < 0.12
                if harmonic:
                    adjacency[i, j] = adjacency[j, i] = True
        visited, streams = set(), []
        for index in range(envelopes.shape[0]):
            if index in visited:
                continue
            pending, members = [index], []
            visited.add(index)
            while pending:
                current = pending.pop()
                members.append(current)
                for neighbor in np.where(adjacency[current])[0]:
                    if int(neighbor) not in visited:
                        visited.add(int(neighbor))
                        pending.append(int(neighbor))
            streams.append({"stream_id": len(streams), "channel_indices": members,
                            "center_frequency_range_hz": [float(np.min(centers[members])), float(np.max(centers[members]))],
                            "salience": float(np.mean(envelopes[members])),
                            "envelope": np.mean(envelopes[members], axis=0),
                            "grouping_evidence": "envelope_coherence_and_harmonic_relation"})
        state["auditory_stream_candidates"] = streams
        return {"stream_count": len(streams), "streams": [{k: v for k, v in item.items() if k != "envelope"}
                                                           for item in streams],
                "limitation": "channel grouping does not perform independent multi-speaker source separation"}

    def step_18(self, state: dict[str, Any]) -> dict[str, Any]:
        """Prioritize auditory streams with softmax attention gain and optional caller-supplied selection bias."""
        if not state.get("auditory_available"):
            return {"status": "skipped_no_auditory_input"}
        streams = state.get("auditory_stream_candidates", [])
        if not streams:
            state["attended_auditory_streams"] = []
            return {"attention_weights": [], "status": "no_streams_to_attend"}
        external = state["request"].auditory_attention or {}
        logits = np.asarray([np.log(max(s["salience"], self.config.epsilon)) +
                             float(external.get(str(s["stream_id"]), 0.0)) for s in streams])
        weights = softmax(logits)
        attended = [dict(s, attention_weight=float(w)) for s, w in zip(streams, weights, strict=True)]
        state.update({"attended_auditory_streams": attended, "auditory_attention_weights": weights})
        return {"attention_weights": weights, "selected_stream_id": int(np.argmax(weights))}


class AcousticFeatureExtractor:
    """Implement Stage 4: time-frequency, energy, pitch, formant and transition feature extraction."""

    def __init__(self, config: SensoryConfig) -> None:
        """Store analysis window and LPC parameters, keeping acoustic values distinct from phoneme labels."""
        self.config = config

    def step_19(self, state: dict[str, Any]) -> dict[str, Any]:
        """Construct an overlapping STFT map that describes frequency content as a function of time."""
        if not state.get("auditory_available"):
            return {"status": "skipped_no_auditory_input"}
        waveform, rate = state["waveform"], state["sample_rate_hz"]
        frame = min(max(32, int(round(self.config.auditory_frame_ms * rate / 1000))), max(waveform.size, 32))
        hop = max(1, int(round(self.config.auditory_hop_ms * rate / 1000)))
        frequencies, times, transformed = signal.stft(
            waveform, fs=rate, nperseg=frame, noverlap=max(0, frame-hop), boundary="zeros", padded=True)
        state.update({"stft_frequencies_hz": frequencies, "stft_times_seconds": times,
                      "stft_complex": transformed, "frame_length": frame, "hop_length": hop})
        return {"spectrogram_shape": list(transformed.shape), "frame_length_samples": frame,
                "hop_length_samples": hop, "time_bins": int(times.size), "frequency_bins": int(frequencies.size)}

    def step_20(self, state: dict[str, Any]) -> dict[str, Any]:
        """Extract spectral power, frame energy, centroid and broad low/high frequency energy ratios."""
        if not state.get("auditory_available"):
            return {"status": "skipped_no_auditory_input"}
        power, frequencies = np.abs(state["stft_complex"])**2, state["stft_frequencies_hz"]
        energy = np.sum(power, axis=0)
        centroid = np.sum(frequencies[:, None] * power, axis=0) / np.maximum(energy, self.config.epsilon)
        low = frequencies < min(500.0, state["sample_rate_hz"] * 0.2)
        high = frequencies > min(2500.0, state["sample_rate_hz"] * 0.3)
        low_energy = np.sum(power[low], axis=0) if np.any(low) else np.zeros_like(energy)
        high_energy = np.sum(power[high], axis=0) if np.any(high) else np.zeros_like(energy)
        rms = np.sqrt(np.maximum(energy, 0.0))
        state.update({"spectral_power": power, "frame_energy": energy, "spectral_centroid_hz": centroid,
                      "frame_rms": rms, "low_frequency_energy": low_energy, "high_frequency_energy": high_energy})
        return {"spectral_power": safe_summary(power), "frame_energy": safe_summary(energy),
                "spectral_centroid_hz": safe_summary(centroid),
                "low_high_energy": {"low_mean": float(np.mean(low_energy)), "high_mean": float(np.mean(high_energy))}}

    def step_21(self, state: dict[str, Any]) -> dict[str, Any]:
        """Estimate F0 and periodicity using autocorrelation on windows centered to match the STFT time grid."""
        if not state.get("auditory_available"):
            return {"status": "skipped_no_auditory_input"}
        fs, source = state["sample_rate_hz"], state["waveform"]
        centers = np.rint(state["stft_times_seconds"] * fs).astype(int)
        length = state["frame_length"]
        offsets = np.arange(length) - length // 2
        padded = np.pad(source, (length, length))
        frames = padded[centers[:, None] + offsets[None, :] + length]
        f0, periodicity = [], []
        for frame in frames:
            corr, lags = normalized_autocorrelation(frame, max(1, int(fs/500)), max(2, int(fs/65)))
            if not corr.size:
                f0.append(float("nan")); periodicity.append(0.0); continue
            index, peak = int(np.argmax(corr)), float(np.max(corr))
            lag = int(lags[index])
            f0.append(float(fs/lag) if peak > 0.12 and lag else float("nan"))
            periodicity.append(float(np.clip(peak, 0, 1)))
        state.update({"frame_signal": frames, "f0_hz": np.asarray(f0), "periodicity": np.asarray(periodicity)})
        return {"f0_hz": safe_summary(np.nan_to_num(f0)), "periodicity": safe_summary(np.asarray(periodicity)),
                "voiced_candidate_fraction": float(np.mean(np.isfinite(f0)))}

    def step_22(self, state: dict[str, Any]) -> dict[str, Any]:
        """Estimate vocal-tract formants from LPC resonance frequencies and bandwidth-filtered polynomial roots."""
        if not state.get("auditory_available"):
            return {"status": "skipped_no_auditory_input"}
        rate = state["sample_rate_hz"]
        order = max(8, int(self.config.lpc_order_per_khz * rate / 1000) + 2)
        formants = np.asarray([_lpc_formants(frame, rate, order) for frame in state["frame_signal"]])
        state["formants_hz"] = formants
        return {"formants_hz": safe_summary(np.nan_to_num(formants)), "lpc_order": order,
                "estimation_scope": "LPC proxy; uncertain roots remain NaN internally"}

    def step_23(self, state: dict[str, Any]) -> dict[str, Any]:
        """Extract formant/spectral derivatives, acoustic onset and a cautious candidate voice-onset-time interval."""
        if not state.get("auditory_available"):
            return {"status": "skipped_no_auditory_input"}
        centroid = state["spectral_centroid_hz"]
        formants = np.nan_to_num(state["formants_hz"], nan=0.0)
        delta_centroid = np.gradient(centroid) if centroid.size > 1 else np.zeros_like(centroid)
        delta_formants = np.gradient(formants, axis=0) if formants.shape[0] > 1 else np.zeros_like(formants)
        pooled = state.get("pooled_auditory_envelope", np.asarray([]))
        if pooled.size:
            change = np.diff(pooled, prepend=pooled[:1])
            peaks, _ = signal.find_peaks(change, height=max(float(np.quantile(change, 0.92)), 0.01))
            vot = float((peaks[0] / state["sample_rate_hz"])) if peaks.size else float("nan")
        else:
            vot = float("nan")
        state.update({"formant_transitions": delta_formants, "spectral_centroid_delta": delta_centroid,
                      "vot_seconds": vot})
        return {"formant_transition": safe_summary(delta_formants), "centroid_change": safe_summary(delta_centroid),
                "vot_seconds": vot, "transition_model": "first temporal difference / onset interval proxy"}

    def step_24(self, state: dict[str, Any]) -> dict[str, Any]:
        """Build a time-indexed acoustic vector without assigning phonetic labels or language-specific categories."""
        if not state.get("auditory_available"):
            return {"status": "skipped_no_auditory_input"}
        formants = np.nan_to_num(state["formants_hz"], nan=0.0)
        matrix = np.column_stack((state["f0_hz"], formants, state["frame_rms"],
                                  state["spectral_centroid_hz"], state["periodicity"]))
        state["acoustic_feature_matrix"] = np.nan_to_num(matrix)
        return {"acoustic_feature_matrix": safe_summary(state["acoustic_feature_matrix"]),
                "feature_order": ["F0", "F1", "F2", "F3", "RMS", "spectral_centroid", "periodicity"],
                "frame_count": int(matrix.shape[0]),
                "stage_boundary": "acoustic features do not themselves constitute phoneme categories"}
