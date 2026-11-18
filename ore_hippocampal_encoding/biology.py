"""Biologically motivated DG, CA3, CA1, and synaptic-plasticity reference models.

These deterministic population simulations implement candidate circuit mechanisms. They are not
anatomically fitted networks, cell-level digital twins, or proof that neurons execute these equations.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .contracts import EncodingConfig
from .functional import sigmoid


def calcium_threshold_plasticity(
    weights: np.ndarray,
    pre_activity: Sequence[float],
    post_activity: Sequence[float],
    calcium_proxy: float,
    learning_rate: float,
    ltd_threshold: float,
    ltp_threshold: float,
    weight_decay: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply a thresholded calcium/Hebbian teaching rule to a synaptic weight matrix.

    Calcium is a normalized model proxy. Above the upper threshold co-activity potentiates;
    between thresholds it depresses; below the lower threshold the Hebbian term is zero.
    This rule is deliberately schematic and varies across real hippocampal synapses.
    """
    matrix = np.asarray(weights, dtype=np.float64)
    pre = np.asarray(pre_activity, dtype=np.float64).reshape(-1)
    post = np.asarray(post_activity, dtype=np.float64).reshape(-1)
    if matrix.shape != (pre.size, post.size):
        raise ValueError("weight matrix shape must match pre- and post-activity dimensions")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(pre)) or not np.all(np.isfinite(post)):
        raise ValueError("plasticity inputs must be finite")
    if not 0.0 <= calcium_proxy <= 1.0:
        raise ValueError("calcium_proxy must lie in [0, 1]")
    if learning_rate < 0.0 or weight_decay < 0.0 or not 0.0 <= ltd_threshold < ltp_threshold <= 1.0:
        raise ValueError("invalid plasticity rate or calcium thresholds")

    coincidence = np.outer(pre, post)
    if calcium_proxy > ltp_threshold:
        direction = "potentiation"
        delta = learning_rate * coincidence
    elif calcium_proxy > ltd_threshold:
        direction = "depression"
        delta = -learning_rate * coincidence
    else:
        direction = "no_hebbian_change"
        delta = np.zeros_like(matrix)
    updated = np.clip((1.0 - weight_decay) * matrix + delta, -1.0, 1.0)
    report = {
        "rule": "calcium_thresholded_hebbian_surrogate",
        "direction": direction,
        "calcium_proxy": float(calcium_proxy),
        "ltd_threshold": float(ltd_threshold),
        "ltp_threshold": float(ltp_threshold),
        "mean_absolute_delta": float(np.mean(np.abs(updated - matrix))),
        "active_pair_count": int(np.count_nonzero(coincidence)),
        "weight_matrix_shape": list(matrix.shape),
        "biological_status": "simplified plasticity abstraction; not a universal synaptic rule",
    }
    return updated, report


class DentateGyrus:
    """Sparse competitive population model for dentate-gyrus pattern separation.

    A fixed random projection creates heterogeneous granule-cell drive. Winner selection models
    sparse recruitment shaped by excitation and inhibition; it does not reproduce the full DG
    microcircuit or claim a universal biological sparsity percentage.
    """

    def __init__(self, config: EncodingConfig, input_size: int) -> None:
        """Initialize deterministic entorhinal-to-granule projection weights."""
        self.config = config
        self.input_size = int(input_size)
        if self.input_size < 1:
            raise ValueError("input_size must be positive")
        rng = np.random.default_rng(config.random_seed)
        self.weights = rng.normal(0.0, 1.0 / np.sqrt(self.input_size),
                                  size=(config.dentate_units, self.input_size))
        self.last_drive = np.zeros(config.dentate_units, dtype=np.float64)
        self.last_activity = np.zeros(config.dentate_units, dtype=np.float64)

    def encode(self, entorhinal_code: Sequence[float], novelty: float = 0.0) -> tuple[np.ndarray, dict[str, Any]]:
        """Project entorhinal input and retain a novelty-adjusted sparse top-k granule code."""
        vector = np.asarray(entorhinal_code, dtype=np.float64).reshape(-1)
        if vector.size != self.input_size:
            raise ValueError(f"expected entorhinal vector of size {self.input_size}, got {vector.size}")
        if not np.all(np.isfinite(vector)) or not 0.0 <= novelty <= 1.0:
            raise ValueError("input must be finite and novelty must lie in [0, 1]")
        drive = np.maximum(self.weights @ vector, 0.0)
        # Novelty modestly increases the number of eligible winners; this is a model hypothesis.
        fraction = min(1.0, self.config.dentate_active_fraction * (1.0 + 0.5 * novelty))
        winner_count = max(1, min(self.config.dentate_units, int(np.ceil(fraction * self.config.dentate_units))))
        winner_indices = np.argpartition(drive, -winner_count)[-winner_count:]
        activity = np.zeros_like(drive)
        maximum = float(np.max(drive[winner_indices])) if winner_indices.size else 0.0
        if maximum > 1e-12:
            activity[winner_indices] = drive[winner_indices] / maximum
        elif winner_indices.size:
            # Keep a defined sparse code even for a zero/negative projected event.
            activity[winner_indices] = 1.0
        self.last_drive = drive
        self.last_activity = activity
        report = {
            "active_units": int(np.count_nonzero(activity)),
            "population_size": self.config.dentate_units,
            "sparsity": float(np.count_nonzero(activity) / self.config.dentate_units),
            "requested_active_fraction": float(fraction),
            "winner_indices": sorted(int(index) for index in winner_indices),
            "mean_drive": float(np.mean(drive)),
            "interpretation": "competitive sparse coding surrogate for DG granule-cell recruitment",
        }
        return activity, report


class CA3AutoassociativeNetwork:
    """DG/entorhinal driven recurrent network for associative event coding and plasticity."""

    def __init__(self, config: EncodingConfig, entorhinal_size: int, dentate_size: int) -> None:
        """Initialize deterministic afferent projections and a zero recurrent association matrix."""
        self.config = config
        self.entorhinal_size = int(entorhinal_size)
        self.dentate_size = int(dentate_size)
        if min(self.entorhinal_size, self.dentate_size) < 1:
            raise ValueError("input dimensions must be positive")
        rng = np.random.default_rng(config.random_seed + 1)
        self.dg_weights = rng.normal(0.0, 0.5 / np.sqrt(self.dentate_size),
                                     size=(config.ca3_units, self.dentate_size))
        self.ec_weights = rng.normal(0.0, 0.5 / np.sqrt(self.entorhinal_size),
                                     size=(config.ca3_units, self.entorhinal_size))
        self.recurrent_weights = np.zeros((config.ca3_units, config.ca3_units), dtype=np.float64)
        self.previous_activity = np.zeros(config.ca3_units, dtype=np.float64)

    def encode(
        self, dentate_activity: Sequence[float], entorhinal_code: Sequence[float]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Integrate DG drive, direct EC drive, and recurrent CA3 activity through a sigmoid."""
        dg = np.asarray(dentate_activity, dtype=np.float64).reshape(-1)
        ec = np.asarray(entorhinal_code, dtype=np.float64).reshape(-1)
        if dg.size != self.dentate_size or ec.size != self.entorhinal_size:
            raise ValueError("CA3 inputs do not match initialized pathway dimensions")
        if not np.all(np.isfinite(dg)) or not np.all(np.isfinite(ec)):
            raise ValueError("CA3 inputs must be finite")
        dg_drive = self.dg_weights @ dg
        ec_drive = self.ec_weights @ ec
        recurrent_drive = self.config.recurrent_gain * (self.recurrent_weights @ self.previous_activity)
        activity = sigmoid(dg_drive + ec_drive + recurrent_drive)
        report = {
            "mean_dentate_drive": float(np.mean(np.abs(dg_drive))),
            "mean_entorhinal_drive": float(np.mean(np.abs(ec_drive))),
            "mean_recurrent_drive": float(np.mean(np.abs(recurrent_drive))),
            "active_units": int(np.count_nonzero(activity > 0.5)),
            "interpretation": "CA3 autoassociative/recurrent coding surrogate",
        }
        self.previous_activity = activity.copy()
        return activity, report

    def update_associations(self, activity: Sequence[float], calcium_proxy: float) -> dict[str, Any]:
        """Update recurrent co-activity weights using the configured calcium-threshold rule."""
        vector = np.asarray(activity, dtype=np.float64).reshape(-1)
        if vector.size != self.config.ca3_units:
            raise ValueError("CA3 activity has the wrong population size")
        self.recurrent_weights, report = calcium_threshold_plasticity(
            self.recurrent_weights,
            vector,
            vector,
            calcium_proxy,
            self.config.plasticity_learning_rate,
            self.config.plasticity_ltd_threshold,
            self.config.plasticity_ltp_threshold,
            self.config.plasticity_weight_decay,
        )
        report["target"] = "CA3 recurrent collateral association matrix"
        return report


class CA1Integrator:
    """Integrate CA3-mediated associative activity and direct entorhinal input."""

    def __init__(self, config: EncodingConfig, entorhinal_size: int) -> None:
        """Initialize direct-EC and CA3 projections into CA1 model neurons."""
        self.config = config
        self.entorhinal_size = int(entorhinal_size)
        rng = np.random.default_rng(config.random_seed + 2)
        self.ec_weights = rng.normal(0.0, 0.5 / np.sqrt(self.entorhinal_size),
                                     size=(config.ca1_units, self.entorhinal_size))
        self.ca3_weights = rng.normal(0.0, 0.5 / np.sqrt(config.ca3_units),
                                      size=(config.ca1_units, config.ca3_units))

    def integrate(
        self, ca3_activity: Sequence[float], entorhinal_code: Sequence[float]
    ) -> tuple[np.ndarray, float, dict[str, Any]]:
        """Combine CA3 and direct EC projections and calculate a normalized mismatch estimate."""
        ca3 = np.asarray(ca3_activity, dtype=np.float64).reshape(-1)
        ec = np.asarray(entorhinal_code, dtype=np.float64).reshape(-1)
        if ca3.size != self.config.ca3_units or ec.size != self.entorhinal_size:
            raise ValueError("CA1 inputs do not match initialized pathway dimensions")
        ca3_drive = self.ca3_weights @ ca3
        ec_drive = self.ec_weights @ ec
        activity = np.tanh(ca3_drive + ec_drive)
        width = max(ca3_drive.size, ec_drive.size)
        left = np.pad(ca3_drive, (0, width - ca3_drive.size))
        right = np.pad(ec_drive, (0, width - ec_drive.size))
        mismatch = float(np.linalg.norm(left - right) / np.sqrt(max(width, 1)))
        report = {
            "ca3_input_magnitude": float(np.linalg.norm(ca3_drive)),
            "direct_entorhinal_input_magnitude": float(np.linalg.norm(ec_drive)),
            "integrated_activity_magnitude": float(np.linalg.norm(activity)),
            "mismatch_interpretation": "model discrepancy metric; CA1 is not a single-purpose comparator",
        }
        return activity, mismatch, report
