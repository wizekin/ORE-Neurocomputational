"""Operation 7: cochlear-inspired auditory features and dorsal/ventral language streams."""
from __future__ import annotations

import numpy as np
from scipy import signal


def _hz_to_erb(frequency: np.ndarray) -> np.ndarray:
    """Convert frequencies in hertz to an equivalent-rectangular-bandwidth scale."""
    return 21.4 * np.log10(1.0 + 0.00437 * frequency)


def _erb_to_hz(value: np.ndarray) -> np.ndarray:
    """Convert equivalent-rectangular-bandwidth positions back to hertz."""
    return (10.0 ** (value / 21.4) - 1.0) / 0.00437


def cochlear_filterbank(spectrum: np.ndarray, frequencies: np.ndarray, bands: int = 16) -> np.ndarray:
    """Pool spectral energy through triangular filters spaced on an ERB-like frequency scale."""
    if spectrum.ndim != 2:
        raise ValueError("spectrum must have [frequency, time] dimensions")
    limits = np.asarray([max(float(frequencies[0]), 20.0), float(frequencies[-1])])
    erb_points = np.linspace(_hz_to_erb(limits[:1])[0], _hz_to_erb(limits[1:])[0], bands + 2)
    centers = _erb_to_hz(erb_points)
    energies = []
    for index in range(1, bands + 1):
        lower, center, upper = centers[index - 1:index + 2]
        rising = np.clip((frequencies - lower) / max(center - lower, 1e-9), 0.0, 1.0)
        falling = np.clip((upper - frequencies) / max(upper - center, 1e-9), 0.0, 1.0)
        weights = np.minimum(rising, falling)
        norm = float(np.sum(weights))
        if norm > 0.0:
            weights = weights / norm
        energies.append(weights @ spectrum)
    return np.asarray(energies)


class AuditoryStreamProcessor:
    """Represent acoustic input for ventral comprehension and dorsal auditory-motor projections."""

    def process(self, waveform: np.ndarray, sample_rate_hz: float) -> dict[str, object]:
        """Compute STFT power, cochlear-inspired band energies and modality-specific stream vectors."""
        audio = np.asarray(waveform, dtype=np.float64)
        if audio.ndim != 1 or audio.size < 16 or not np.all(np.isfinite(audio)):
            raise ValueError("waveform must be a finite one-dimensional array with at least 16 samples")
        if sample_rate_hz <= 40.0:
            raise ValueError("sample_rate_hz must exceed 40 Hz for the configured auditory filterbank")
        nperseg = min(256, audio.size)
        frequencies, times, transformed = signal.stft(
            audio, fs=sample_rate_hz, nperseg=nperseg, noverlap=nperseg // 2,
            boundary=None, padded=False,
        )
        power = np.abs(transformed) ** 2
        bands = cochlear_filterbank(power, frequencies, bands=16)
        envelope = np.abs(signal.hilbert(audio))
        temporal_envelope = np.asarray([
            float(np.mean(envelope)), float(np.std(envelope)),
            float(np.max(envelope)), float(np.mean(np.diff(envelope) ** 2)) if envelope.size > 1 else 0.0,
        ])
        band_summary = np.mean(bands, axis=1)
        phonological = np.concatenate((band_summary, temporal_envelope))
        phonological /= max(float(np.linalg.norm(phonological)), 1e-12)
        ventral = np.maximum(phonological, 0.0)
        dorsal = np.gradient(phonological)
        zero_crossing_rate = float(np.mean(np.signbit(audio[1:]) != np.signbit(audio[:-1])))
        return {
            "stft_power": power,
            "stft_frequencies_hz": frequencies,
            "stft_times_seconds": times,
            "cochlear_band_energies": bands,
            "phonological_feature_vector": phonological,
            "ventral_comprehension_vector": ventral,
            "dorsal_auditory_motor_vector": dorsal,
            "temporal_envelope": temporal_envelope,
            "zero_crossing_rate": zero_crossing_rate,
            "biological_analogue": (
                "Cochlear frequency-selective encoding and auditory cortical ventral/dorsal pathways; "
                "STFT and ERB-like filters are engineering approximations, not literal neural transforms."
            ),
        }
