"""Biologically motivated, explicitly approximate networks for Point 11 reasoning.

These classes are computational surrogates inspired by predictive-processing,
frontoparietal hypothesis competition, hippocampal evidence binding, cortical
causal maps and neuromodulatory uncertainty signalling. They are not literal
neural simulations and make no physiological claims.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Bound a finite scalar and reject invalid ranges."""
    number = float(value)
    if not math.isfinite(number) or low > high:
        raise ValueError("value must be finite and low must not exceed high")
    return min(high, max(low, number))


def tokenize(text: str) -> tuple[str, ...]:
    """Tokenize Unicode word forms without claiming full linguistic analysis."""
    return tuple(token.casefold()
                 for token in re.findall(r"[^\W_]+(?:['’-][^\W_]+)?", text, flags=re.UNICODE))


def vectorize_tokens(tokens: Iterable[str], dimensions: int = 64) -> np.ndarray:
    """Create deterministic signed-hash vectors for transparent feature proxies."""
    if dimensions < 8:
        raise ValueError("dimensions must be at least 8")
    vector = np.zeros(dimensions, dtype=float)
    for token in tokens:
        digest = hashlib.blake2b(str(token).casefold().encode("utf-8"), digest_size=8).digest()
        integer = int.from_bytes(digest, "big")
        index = integer % dimensions
        sign = 1.0 if ((integer >> 8) & 1) else -1.0
        vector[index] += sign
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


@dataclass
class NeuralPopulation:
    """Maintain bounded recurrent population activity with excitatory and inhibitory drive."""

    name: str
    dimensions: int = 64
    recurrent_scale: float = 0.12
    seed: int = 11
    activity: np.ndarray = field(init=False, repr=False)
    weights: np.ndarray = field(init=False, repr=False)
    update_count: int = 0

    def __post_init__(self) -> None:
        """Initialize deterministic recurrent weights and neutral activity."""
        if self.dimensions < 8 or not math.isfinite(self.recurrent_scale) or self.recurrent_scale < 0:
            raise ValueError("invalid neural population dimensions or recurrent scale")
        rng = np.random.default_rng(self.seed)
        self.weights = rng.normal(0.0, 1.0 / math.sqrt(self.dimensions),
                                  (self.dimensions, self.dimensions)) * self.recurrent_scale
        self.activity = np.zeros(self.dimensions, dtype=float)

    def update(self, incoming: Sequence[float], gain: float = 1.0,
               inhibition: float = 0.0) -> np.ndarray:
        """Advance one recurrent activation step using a clipped tanh transfer function."""
        vector = np.asarray(incoming, dtype=float)
        if vector.shape != (self.dimensions,) or not np.all(np.isfinite(vector)):
            raise ValueError("incoming must be a finite vector matching population dimensions")
        if not math.isfinite(gain) or gain < 0 or not math.isfinite(inhibition):
            raise ValueError("gain and inhibition must be finite and gain non-negative")
        drive = gain * vector + self.weights @ self.activity - inhibition
        self.activity = np.tanh(np.clip(drive, -20.0, 20.0))
        self.update_count += 1
        return self.activity.copy()

    def snapshot(self) -> dict[str, Any]:
        """Expose compact population diagnostics without exporting internal weight matrices."""
        return {"name": self.name, "dimensions": self.dimensions,
                "update_count": self.update_count,
                "mean_activity": float(np.mean(self.activity)),
                "activity_energy": float(np.mean(self.activity ** 2)),
                "model": "recurrent_population_surrogate"}


class PredictiveCodingCircuit:
    """Maintain hierarchical predictions and report precision-weighted prediction errors.

    Inspired by predictive-processing accounts of perception: each level holds a
    prediction; incoming signals generate bottom-up errors weighted by an estimated
    precision. This is an engineering surrogate for Bayesian-brain-style inference,
    not a measurement of cortical activity.
    """

    def __init__(self, dimensions: int = 64, precision: float = 1.0) -> None:
        """Initialize neutral predictions at every level with a bounded precision."""
        if dimensions < 8:
            raise ValueError("dimensions must be at least 8")
        if not math.isfinite(precision) or not 0.0 <= precision <= 4.0:
            raise ValueError("precision must be finite and in [0, 4]")
        self.dimensions = dimensions
        self.precision = float(precision)
        self.prediction = np.zeros(dimensions, dtype=float)
        self.last_error = np.zeros(dimensions, dtype=float)
        self.update_count = 0

    def compute_error(self, signal: Sequence[float]) -> dict[str, Any]:
        """Return the precision-weighted prediction error for one incoming signal."""
        vector = np.asarray(signal, dtype=float)
        if vector.shape != (self.dimensions,) or not np.all(np.isfinite(vector)):
            raise ValueError("signal must be a finite vector matching circuit dimensions")
        error = vector - self.prediction
        self.last_error = error * self.precision
        self.update_count += 1
        return {"absolute_error": float(np.mean(np.abs(error))),
                "precision_weighted_error": float(np.mean(np.abs(self.last_error))),
                "surprise_proxy": float(np.mean(error ** 2) * self.precision)}

    def update_prediction(self, signal: Sequence[float], rate: float = 0.25) -> np.ndarray:
        """Move the internal prediction toward the signal with a bounded learning rate."""
        vector = np.asarray(signal, dtype=float)
        if vector.shape != (self.dimensions,) or not np.all(np.isfinite(vector)):
            raise ValueError("signal must be a finite vector matching circuit dimensions")
        if not math.isfinite(rate) or not 0.0 <= rate <= 1.0:
            raise ValueError("rate must be finite and in [0, 1]")
        self.prediction = self.prediction + rate * (vector - self.prediction)
        return self.prediction.copy()

    def snapshot(self) -> dict[str, Any]:
        """Report prediction stability and accumulated error statistics."""
        return {"update_count": self.update_count, "precision": self.precision,
                "mean_prediction": float(np.mean(self.prediction)),
                "mean_absolute_error": float(np.mean(np.abs(self.last_error))),
                "model": "predictive_coding_surrogate"}


class HypothesisCompetitionNetwork:
    """Model prefrontal-style competition between candidate hypotheses."""

    def __init__(self) -> None:
        """Initialize the competition accumulator."""
        self.competitions = 0
        self.last_winner: str | None = None

    def compete(self, hypotheses: Mapping[str, float]) -> dict[str, Any]:
        """Run normalized competition over hypothesis strengths (softmax-like)."""
        if not hypotheses:
            raise ValueError("competition requires at least one hypothesis")
        strengths = {str(name): clamp(value) for name, value in hypotheses.items()}
        total = sum(strengths.values())
        shares = ({name: value / total for name, value in strengths.items()}
                  if total > 0 else {name: 1.0 / len(strengths) for name in strengths})
        winner = max(shares, key=lambda name: (shares[name], name))
        entropy = -sum(share * math.log2(share) for share in shares.values() if share > 0)
        self.competitions += 1
        self.last_winner = winner
        return {"winner": winner, "shares": shares,
                "competition_entropy_bits": entropy,
                "note": "competition shares are not posterior probabilities"}

    def snapshot(self) -> dict[str, Any]:
        """Return competition diagnostics."""
        return {"competitions": self.competitions, "last_winner": self.last_winner,
                "model": "prefrontal_hypothesis_competition_surrogate"}


class HippocampalEvidenceNetwork:
    """Bind evidence episodes with context metadata for later reinstatement."""

    def __init__(self, dimensions: int = 64) -> None:
        """Initialize an episodic vector store with fixed feature dimensionality."""
        self.dimensions = dimensions
        self.episodes: dict[str, np.ndarray] = {}
        self.metadata: dict[str, dict[str, Any]] = {}
        self.bind_count = 0

    def bind(self, evidence_id: str, tokens: Sequence[str],
             metadata: Mapping[str, Any]) -> dict[str, Any]:
        """Bind one evidence episode and its provenance without overwriting identity."""
        if evidence_id in self.episodes:
            raise ValueError("evidence_id already exists")
        vector = vectorize_tokens(tokens or ("missing",), self.dimensions)
        self.episodes[evidence_id] = vector
        self.metadata[evidence_id] = dict(metadata)
        self.bind_count += 1
        return {"evidence_id": evidence_id, "bound": True,
                "context_keys": sorted(str(key) for key in metadata),
                "binding_model": "hippocampal_evidence_binding_surrogate"}

    def reinstate(self, tokens: Sequence[str], limit: int = 5) -> list[dict[str, Any]]:
        """Rank stored evidence episodes by cosine match to a retrieval cue."""
        cue = vectorize_tokens(tokens or ("missing",), self.dimensions)
        rows = []
        for evidence_id, vector in self.episodes.items():
            product = float(np.dot(cue, vector))
            rows.append({"evidence_id": evidence_id, "similarity": product,
                         "metadata": dict(self.metadata[evidence_id])})
        return sorted(rows, key=lambda row: (-row["similarity"], row["evidence_id"]))[:max(0, limit)]

    def snapshot(self) -> dict[str, Any]:
        """Return episode and binding counts as hippocampal-model diagnostics."""
        return {"episode_count": len(self.episodes), "binding_count": self.bind_count,
                "model": "hippocampal_evidence_binding_surrogate"}


class CorticalCausalMapNetwork:
    """Maintain bounded co-activation weights as a surrogate for causal maps.

    Weights encode temporal/spatial co-occurrence strength. The module explicitly
    separates these associative weights from causal edges: co-activation is not
    causation, and the pipeline keeps causal confidence classes separate.
    """

    def __init__(self, learning_rate: float = 0.1) -> None:
        """Initialize associative strengths and a bounded learning rate."""
        if not math.isfinite(learning_rate) or not 0.0 <= learning_rate <= 1.0:
            raise ValueError("learning_rate must lie in [0, 1]")
        self.learning_rate = float(learning_rate)
        self.associations: dict[tuple[str, str], float] = {}
        self.update_count = 0

    def observe(self, cues: Sequence[str], outcomes: Sequence[str], gate: float = 1.0) -> int:
        """Apply a bounded co-activation update to cue-outcome associations."""
        gate = clamp(gate)
        updates = 0
        for cue in set(cues):
            for outcome in set(outcomes):
                if cue == outcome:
                    continue
                key = (str(cue), str(outcome))
                old = self.associations.get(key, 0.0)
                self.associations[key] = clamp(old + self.learning_rate * gate * (1.0 - old))
                updates += 1
        self.update_count += updates
        return updates

    def snapshot(self) -> dict[str, Any]:
        """Return association statistics without exporting the full weight map."""
        values = list(self.associations.values())
        return {"association_count": len(self.associations),
                "update_count": self.update_count,
                "mean_association": float(np.mean(values)) if values else 0.0,
                "association_is_not_causation": True,
                "model": "cortical_causal_map_surrogate"}


class NeuromodulatoryUncertaintyGate:
    """Combine surprise, expected information gain and reliability into a learning gate."""

    def __init__(self) -> None:
        """Initialize the gate accumulator."""
        self.update_count = 0
        self.last_gate = 0.0

    def compute(self, surprise: float, information_gain: float, reliability: float) -> dict[str, float]:
        """Compute an illustrative uncertainty-salience gate for model updates."""
        for name, value in (("surprise", surprise), ("information_gain", information_gain),
                            ("reliability", reliability)):
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        gate = clamp((0.40 * clamp(surprise) + 0.30 * clamp(information_gain)
                      + 0.30 * clamp(reliability)) * (0.5 + 0.5 * clamp(reliability)))
        self.last_gate = gate
        self.update_count += 1
        return {"learning_gate": gate, "surprise_signal": clamp(surprise),
                "information_gain_signal": clamp(information_gain),
                "source_reliability": clamp(reliability)}

    def snapshot(self) -> dict[str, Any]:
        """Expose gate activity without claims about a specific neuromodulator level."""
        return {"update_count": self.update_count, "last_gate": self.last_gate,
                "model": "neuromodulatory_uncertainty_gate_surrogate"}


class BiologicalBayesianCausalNetwork:
    """Coordinate predictive coding, hypothesis competition, evidence binding and causal maps."""

    def __init__(self, dimensions: int = 64, learning_rate: float = 0.1) -> None:
        """Create interacting surrogates with consistent representation dimensions."""
        self.dimensions = dimensions
        self.population = NeuralPopulation("uncertainty_population", dimensions, seed=111)
        self.predictive_coding = PredictiveCodingCircuit(dimensions)
        self.hypothesis_competition = HypothesisCompetitionNetwork()
        self.hippocampus = HippocampalEvidenceNetwork(dimensions)
        self.causal_map = CorticalCausalMapNetwork(learning_rate)
        self.uncertainty_gate = NeuromodulatoryUncertaintyGate()

    def encode(self, tokens: Sequence[str]) -> np.ndarray:
        """Encode a token set and update the recurrent uncertainty population."""
        vector = vectorize_tokens(tokens, self.dimensions)
        return self.population.update(vector)

    def process_step(self, step: int, tokens: Sequence[str], surprise: float,
                     reliability: float, prediction_error: float = 0.0) -> dict[str, Any]:
        """Advance surrogate activity for one functional operation."""
        vector = vectorize_tokens(tokens or ("neutral",), self.dimensions)
        activity = self.population.update(vector, gain=1.0, inhibition=0.02 * step)
        error_report = self.predictive_coding.compute_error(vector)
        gate = self.uncertainty_gate.compute(surprise, prediction_error, reliability)
        return {"uncertainty_population": self.population.snapshot(),
                "predictive_coding": error_report,
                "activity_mean": float(np.mean(activity)),
                "uncertainty_gate": gate,
                "interpretation": "functional surrogate, not literal neural simulation"}

    def bind_evidence(self, evidence_id: str, tokens: Sequence[str],
                      metadata: Mapping[str, Any]) -> dict[str, Any]:
        """Bind one evidence episode into the hippocampal surrogate."""
        return self.hippocampus.bind(evidence_id, tokens, metadata)

    def compete_hypotheses(self, hypotheses: Mapping[str, float]) -> dict[str, Any]:
        """Run one hypothesis competition cycle."""
        return self.hypothesis_competition.compete(hypotheses)

    def observe_association(self, cues: Sequence[str], outcomes: Sequence[str],
                            gate: float = 1.0) -> int:
        """Update the associative causal-map surrogate."""
        return self.causal_map.observe(cues, outcomes, gate)

    def snapshot(self) -> dict[str, Any]:
        """Return diagnostics for the complete surrogate network."""
        return {"uncertainty_population": self.population.snapshot(),
                "predictive_coding": self.predictive_coding.snapshot(),
                "hypothesis_competition": self.hypothesis_competition.snapshot(),
                "hippocampus": self.hippocampus.snapshot(),
                "causal_map": self.causal_map.snapshot(),
                "uncertainty_gate": self.uncertainty_gate.snapshot(),
                "literal_neuron_simulation": False,
                "biological_fidelity": "computational surrogate"}


__all__ = [
    "BiologicalBayesianCausalNetwork",
    "CorticalCausalMapNetwork",
    "HippocampalEvidenceNetwork",
    "HypothesisCompetitionNetwork",
    "NeuralPopulation",
    "NeuromodulatoryUncertaintyGate",
    "PredictiveCodingCircuit",
    "clamp",
    "tokenize",
    "vectorize_tokens",
]
