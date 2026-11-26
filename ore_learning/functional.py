"""Pure numeric and decision functions shared by the RLMS pipeline (Point 14).

Every function here is deterministic, side-effect free, and covered by the acceptance
tests.  The formulas implement the specification's mathematical reference section:
eligibility conjunction (Step 7), utility ranking (Step 8), MoE gating (Step 9),
modality fusion (Step 22), GAN objectives (Step 23), SHAP weights (Step 28), LIME
surrogate loss (Step 29), RLVR reward (Step 35), FedAvg weighting (Step 41), and
QUBO objectives with penalties (Step 45), plus the calibration and drift statistics
used by Phase J.
"""
from __future__ import annotations

from itertools import combinations
import math
from typing import Callable, Sequence

import numpy as np

__all__ = (
    "brier_score",
    "clip_update_norm",
    "counterfactual_greedy_search",
    "expected_calibration_error",
    "exhaustive_qubo",
    "fedavg_weights",
    "fusion_score",
    "gan_discriminator_objective",
    "gan_generator_objective",
    "kl_divergence",
    "moe_combined_output",
    "moe_routing_concentration",
    "mse",
    "normalize_to_unit_interval",
    "one_hot",
    "population_stability_index",
    "quantization_error",
    "quantize_weights",
    "reliability_bins",
    "sigmoid",
    "simulated_annealing_qubo",
    "shapley_subset_weight",
    "softmax",
    "utility_score",
    "weighted_ridge_fit",
    "weighted_r_squared",
)


# --------------------------------------------------------------------- helpers

def sigmoid(x) -> np.ndarray:
    """Numerically stable logistic function."""
    z = np.asarray(x, dtype=float)
    out = np.empty_like(z)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exponents = np.exp(z[~positive])
    out[~positive] = exponents / (1.0 + exponents)
    return out


def softmax(x, axis: int = -1) -> np.ndarray:
    """Row-wise softmax with max subtraction for stability."""
    z = np.asarray(x, dtype=float)
    shifted = z - np.max(z, axis=axis, keepdims=True)
    exponents = np.exp(shifted)
    return exponents / np.sum(exponents, axis=axis, keepdims=True)


def one_hot(index: int, size: int) -> np.ndarray:
    """Return a float one-hot vector."""
    vector = np.zeros(size, dtype=float)
    vector[index] = 1.0
    return vector


def mse(predictions, targets) -> float:
    """Mean squared error between two equally shaped sequences."""
    predicted = np.asarray(predictions, dtype=float)
    actual = np.asarray(targets, dtype=float)
    if predicted.shape != actual.shape:
        raise ValueError("predictions and targets must have the same shape")
    if predicted.size == 0:
        raise ValueError("mse requires at least one sample")
    return float(np.mean((predicted - actual) ** 2))


def normalize_to_unit_interval(values) -> np.ndarray:
    """Min-max normalize a finite array into [0, 1]; constant arrays map to 0.5."""
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError("cannot normalize an empty array")
    low, high = float(np.min(array)), float(np.max(array))
    if not (np.isfinite(array)).all():
        raise ValueError("values must be finite")
    if high - low <= 1e-12:
        return np.full(array.shape, 0.5)
    return (array - low) / (high - low)


# ------------------------------------------------------- routing math (Steps 7-10)

def utility_score(quality: float, calibration: float, robustness: float,
                  latency_penalty: float, resource_penalty: float,
                  uncertainty_penalty: float, *, weight_quality: float,
                  weight_calibration: float, weight_robustness: float,
                  weight_latency: float, weight_resource: float,
                  weight_uncertainty: float) -> float:
    """Step 8 -- U(m,t) = w_q Q + w_c C + w_r R - w_l L - w_k K - w_u U.

    Inputs are expected in [0, 1]; penalties are positive numbers whose contribution
    is subtracted.  The score ranks eligible candidates only -- it can never override
    hard eligibility constraints.
    """
    terms = (quality, calibration, robustness, latency_penalty, resource_penalty,
             uncertainty_penalty)
    for name, value in zip(("quality", "calibration", "robustness", "latency_penalty",
                            "resource_penalty", "uncertainty_penalty"), terms):
        if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be a finite fraction in [0, 1]")
    weights = (weight_quality, weight_calibration, weight_robustness, weight_latency,
               weight_resource, weight_uncertainty)
    for name, value in zip(("weight_quality", "weight_calibration", "weight_robustness",
                            "weight_latency", "weight_resource", "weight_uncertainty"),
                           weights):
        if float(value) < 0.0:
            raise ValueError(f"{name} must be non-negative")
    score = (weight_quality * quality + weight_calibration * calibration
             + weight_robustness * robustness - weight_latency * latency_penalty
             - weight_resource * resource_penalty - weight_uncertainty * uncertainty_penalty)
    return float(score)


def moe_combined_output(gates, expert_outputs) -> np.ndarray:
    """Step 9 -- y = sum_i g_i(x) E_i(x) for gated expert outputs.

    ``gates`` are per-expert scalar weights (already normalized or normalized here);
    ``expert_outputs`` are equally shaped arrays.  The combination is a routing
    reference: an ORE coordinator still decides which external specialist runs.
    """
    gate_array = np.asarray(gates, dtype=float)
    outputs = np.asarray(expert_outputs, dtype=float)
    if gate_array.ndim != 1 or gate_array.size == 0:
        raise ValueError("gates must be a non-empty 1-D sequence")
    if outputs.shape[0] != gate_array.size:
        raise ValueError("one gate is required per expert output")
    total = float(np.sum(gate_array))
    if total <= 0.0:
        raise ValueError("gate weights must be positive")
    normalized = gate_array / total
    if not np.isclose(float(np.sum(normalized)), 1.0, atol=1e-9):
        raise ValueError("normalized gates must sum to one")
    return normalized @ outputs


def moe_routing_concentration(gates) -> float:
    """Step 9 -- share captured by the strongest expert (router-concentration monitor)."""
    gate_array = np.asarray(gates, dtype=float)
    if gate_array.size == 0:
        raise ValueError("gates must not be empty")
    total = float(np.sum(gate_array))
    if total <= 0.0:
        raise ValueError("gate weights must be positive")
    return float(np.max(gate_array) / total)


def fusion_score(supports: Sequence[float], reliabilities: Sequence[float],
                 relevances: Sequence[float]) -> float:
    """Step 22 -- s(c) = sum_j alpha_j r_j s_j(c) / sum_j alpha_j r_j.

    Reliability- and relevance-weighted support for one candidate concept.  The
    result is evidence strength, never proof; competing observations are retained by
    the caller.
    """
    support = np.asarray(supports, dtype=float)
    reliability = np.asarray(reliabilities, dtype=float)
    relevance = np.asarray(relevances, dtype=float)
    if not (support.shape == reliability.shape == relevance.shape) or support.size == 0:
        raise ValueError("supports, reliabilities and relevances must align and be non-empty")
    if np.any(reliability < 0.0) or np.any(relevance < 0.0):
        raise ValueError("reliabilities and relevances must be non-negative")
    if np.any(reliability > 1.0):
        raise ValueError("reliabilities must be calibrated fractions in [0, 1]")
    weights = relevance * reliability
    total_weight = float(np.sum(weights))
    if total_weight <= 0.0:
        return 0.0
    return float(np.sum(weights * support) / total_weight)


# ------------------------------------------------- calibration math (Steps 48-49)

def brier_score(probabilities, outcomes) -> float:
    """Brier score N^-1 sum_i (p_i - y_i)^2 for binary outcomes."""
    probability = np.asarray(probabilities, dtype=float)
    outcome = np.asarray(outcomes, dtype=float)
    if probability.shape != outcome.shape or probability.size == 0:
        raise ValueError("probabilities and outcomes must align and be non-empty")
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("probabilities must lie in [0, 1]")
    if np.any((outcome < 0.0) | (outcome > 1.0)):
        raise ValueError("outcomes must be 0/1 indicators")
    return float(np.mean((probability - outcome) ** 2))


def reliability_bins(probabilities, outcomes, n_bins: int = 10):
    """Return (bin_proportions, bin_accuracies, bin_counts) for a reliability diagram."""
    probability = np.asarray(probabilities, dtype=float)
    outcome = np.asarray(outcomes, dtype=float)
    if probability.shape != outcome.shape or probability.size == 0:
        raise ValueError("probabilities and outcomes must align and be non-empty")
    if not 1 <= n_bins <= 20:
        raise ValueError("n_bins must be in [1, 20]")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    indices = np.clip(np.digitize(probability, edges[1:-1], right=True), 0, n_bins - 1)
    proportions = np.zeros(n_bins)
    accuracies = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=int)
    for bin_index in range(n_bins):
        mask = indices == bin_index
        counts[bin_index] = int(np.sum(mask))
        if counts[bin_index]:
            proportions[bin_index] = float(np.mean(probability[mask]))
            accuracies[bin_index] = float(np.mean(outcome[mask]))
    return proportions, accuracies, counts


def expected_calibration_error(probabilities, outcomes, n_bins: int = 10) -> float:
    """ECE = sum_b (n_b / N) |acc_b - conf_b| over reliability bins."""
    _, accuracies, counts = reliability_bins(probabilities, outcomes, n_bins=n_bins)
    proportions, _, _ = reliability_bins(probabilities, outcomes, n_bins=n_bins)
    total = int(np.sum(counts))
    if total == 0:
        raise ValueError("at least one sample is required")
    return float(np.sum((counts / total) * np.abs(accuracies - proportions)))


def population_stability_index(expected, observed, n_bins: int = 10) -> float:
    """PSI = sum_i (q_i - p_i) ln(q_i / p_i) over shared quantile bins.

    ``expected`` is the reference distribution, ``observed`` the recent one.  Small
    epsilon floors keep the logarithm finite for empty bins.
    """
    reference = np.asarray(expected, dtype=float)
    recent = np.asarray(observed, dtype=float)
    if reference.size == 0 or recent.size == 0:
        raise ValueError("both distributions must be non-empty")
    combined = np.concatenate([reference, recent])
    quantiles = np.quantile(combined, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(quantiles)
    if edges.size < 2:
        return 0.0
    indices_reference = np.clip(np.searchsorted(edges[1:-1], reference, side="right"),
                                0, edges.size - 2)
    indices_recent = np.clip(np.searchsorted(edges[1:-1], recent, side="right"),
                             0, edges.size - 2)
    n_bins_eff = edges.size - 1
    epsilon = 1e-6
    psi_total = 0.0
    for bin_index in range(n_bins_eff):
        p = float(np.sum(indices_reference == bin_index)) / reference.size
        q = float(np.sum(indices_recent == bin_index)) / recent.size
        p = max(p, epsilon)
        q = max(q, epsilon)
        psi_total += (q - p) * math.log(q / p)
    return float(psi_total)


def kl_divergence(expected, observed, n_bins: int = 10) -> float:
    """Binned KL divergence D(observed || expected) over shared quantile bins."""
    reference = np.asarray(expected, dtype=float)
    recent = np.asarray(observed, dtype=float)
    if reference.size == 0 or recent.size == 0:
        raise ValueError("both distributions must be non-empty")
    combined = np.concatenate([reference, recent])
    edges = np.unique(np.quantile(combined, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size < 2:
        return 0.0
    indices_reference = np.clip(np.searchsorted(edges[1:-1], reference, side="right"),
                                0, edges.size - 2)
    indices_recent = np.clip(np.searchsorted(edges[1:-1], recent, side="right"),
                             0, edges.size - 2)
    epsilon = 1e-6
    divergence = 0.0
    for bin_index in range(edges.size - 1):
        p = max(float(np.sum(indices_reference == bin_index)) / reference.size, epsilon)
        q = max(float(np.sum(indices_recent == bin_index)) / recent.size, epsilon)
        divergence += q * math.log(q / p)
    return float(divergence)


# ---------------------------------------------- explainability math (Steps 28-30)

def shapley_subset_weight(subset_size: int, total_features: int) -> float:
    """Shapley cardinal weight |S|!(|F|-|S|-1)!/|F|! for S ⊆ F\\{i}."""
    if not 0 <= subset_size <= total_features - 1:
        raise ValueError("subset must be a proper subset of the feature set")
    numerator = math.factorial(subset_size) * math.factorial(total_features - subset_size - 1)
    return numerator / math.factorial(total_features)


def exact_shapley_values(evaluate: Callable[[np.ndarray], float], instance: np.ndarray,
                         background: np.ndarray) -> np.ndarray:
    """Exact Shapley feature attributions for small feature sets.

    ``evaluate`` maps a feature vector to a scalar; masked features are taken from
    background rows and averaged.  Efficiency holds: sum(phi) = f(x) - E[f(background)].
    """
    x = np.asarray(instance, dtype=float)
    bg = np.asarray(background, dtype=float)
    if x.ndim != 1 or bg.ndim != 2 or bg.shape[1] != x.size:
        raise ValueError("instance must be 1-D and background 2-D with matching width")
    if bg.shape[0] == 0:
        raise ValueError("background must contain at least one row")
    n_features = x.size
    if n_features > 16:
        raise ValueError("exact Shapley is limited to 16 features; use the sampling variant")

    def masked_mean(feature_mask: int) -> float:
        """Average f() over background rows with the masked features from x."""
        columns = [index for index in range(n_features) if feature_mask >> index & 1]
        rows = bg.copy()
        if columns:
            rows[:, columns] = x[columns]
        return float(np.mean([evaluate(row) for row in rows]))

    baseline_values = {}
    values = np.zeros(n_features)
    for feature in range(n_features):
        others = [j for j in range(n_features) if j != feature]
        total = 0.0
        for size in range(n_features):
            weight = shapley_subset_weight(size, n_features)
            for subset in combinations(others, size):
                mask_with = 0
                for index in subset:
                    mask_with |= 1 << index
                mask_without = mask_with
                mask_with |= 1 << feature
                if mask_without not in baseline_values:
                    baseline_values[mask_without] = masked_mean(mask_without)
                if mask_with not in baseline_values:
                    baseline_values[mask_with] = masked_mean(mask_with)
                total += weight * (baseline_values[mask_with] - baseline_values[mask_without])
        values[feature] = total
    return values


def weighted_ridge_fit(perturbations: np.ndarray, responses: np.ndarray,
                       sample_weights: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    """Solve min_w sum_i pi_i (y_i - z_i w)^2 + ridge ||w||^2 (LIME surrogate step)."""
    z = np.asarray(perturbations, dtype=float)
    y = np.asarray(responses, dtype=float)
    w = np.asarray(sample_weights, dtype=float)
    if z.ndim != 2 or y.ndim != 1 or w.ndim != 1:
        raise ValueError("perturbations must be 2-D; responses and weights 1-D")
    if not (z.shape[0] == y.size == w.size) or z.shape[0] == 0:
        raise ValueError("perturbation rows, responses and weights must align and be non-empty")
    weighted_z = z * w[:, None]
    gram = z.T @ weighted_z + ridge * np.eye(z.shape[1])
    rhs = weighted_z.T @ y
    return np.linalg.solve(gram, rhs)


def weighted_r_squared(responses: np.ndarray, predictions: np.ndarray,
                       sample_weights: np.ndarray) -> float:
    """Weighted coefficient of determination for surrogate local fidelity."""
    y = np.asarray(responses, dtype=float)
    y_hat = np.asarray(predictions, dtype=float)
    w = np.asarray(sample_weights, dtype=float)
    if not (y.size == y_hat.size == w.size) or y.size == 0:
        raise ValueError("responses, predictions and weights must align and be non-empty")
    total_weight = float(np.sum(w))
    if total_weight <= 0.0:
        return 0.0
    mean = float(np.sum(w * y) / total_weight)
    residual = float(np.sum(w * (y - y_hat) ** 2))
    total = float(np.sum(w * (y - mean) ** 2))
    if total <= 1e-12:
        return 1.0 if residual <= 1e-12 else 0.0
    return float(1.0 - residual / total)


def counterfactual_greedy_search(evaluate: Callable[[np.ndarray], float],
                                 instance: np.ndarray, direction: int,
                                 mutable: Sequence[bool],
                                 max_changes: int, step: float = 0.5,
                                 max_iterations: int = 64) -> np.ndarray | None:
    """Step 30 -- greedy minimal-perturbation search that flips a scalar score.

    ``direction`` is +1 to increase or -1 to decrease ``evaluate``.  Only features
    flagged mutable may change, at most ``max_changes`` of them; the search flips the
    single most useful feature per iteration.  Returns None when no feasible
    counterfactual exists within the budget (feasibility is reported by the caller).
    """
    x = np.asarray(instance, dtype=float).copy()
    if x.ndim != 1:
        raise ValueError("instance must be a 1-D feature vector")
    mutable_flags = np.asarray(mutable, dtype=bool)
    if mutable_flags.shape != x.shape:
        raise ValueError("mutability flags must match the feature vector")
    if max_changes < 0:
        raise ValueError("max_changes must be non-negative")
    current = evaluate(x)
    changed = np.zeros_like(mutable_flags)
    n_changes = 0
    for _ in range(max_iterations):
        if n_changes >= max_changes:
            return None
        best_index, best_value, best_score = -1, None, current
        for index in range(x.size):
            if not mutable_flags[index] or changed[index]:
                continue
            for candidate in (x[index] + step, x[index] - step):
                trial = x.copy()
                trial[index] = candidate
                score = evaluate(trial)
                if direction * (score - current) > direction * (best_score - current) + 1e-12:
                    best_index, best_value, best_score = index, candidate, score
        if best_index < 0:
            return None
        x[best_index] = best_value
        changed[best_index] = True
        n_changes += 1
        if direction * (best_score - current) > 0.0 and _crossed(current, best_score, direction):
            return x
        current = best_score
    return None


def _crossed(before: float, after: float, direction: int) -> bool:
    """True when the greedy move crossed the decision boundary (sign change of margin)."""
    return (direction > 0 and after > 0.0 and before <= 0.0) or \
           (direction < 0 and after < 0.0 and before >= 0.0)


# ---------------------------------------------------- GAN objectives (Step 23)

def gan_discriminator_objective(real_probabilities, fake_probabilities) -> float:
    """max_D E log D(x) + E_z log(1 - D(G(z))), reported as a loss to minimize."""
    p_real = np.clip(np.asarray(real_probabilities, dtype=float), 1e-7, 1.0 - 1e-7)
    p_fake = np.clip(np.asarray(fake_probabilities, dtype=float), 1e-7, 1.0 - 1e-7)
    if p_real.size == 0 or p_fake.size == 0:
        raise ValueError("both real and fake batches are required")
    return float(-np.mean(np.log(p_real)) - np.mean(np.log(1.0 - p_fake)))


def gan_generator_objective(fake_probabilities) -> float:
    """Non-saturating generator objective -E_z log D(G(z))."""
    p_fake = np.clip(np.asarray(fake_probabilities, dtype=float), 1e-7, 1.0 - 1e-7)
    if p_fake.size == 0:
        raise ValueError("the fake batch must not be empty")
    return float(-np.mean(np.log(p_fake)))


# ------------------------------------------------------ QUBO math (Steps 44-47)

def qubo_energy(bits: np.ndarray, qubo_matrix: np.ndarray) -> float:
    """Evaluate x^T Q x for a binary vector x."""
    x = np.asarray(bits, dtype=float)
    q = np.asarray(qubo_matrix, dtype=float)
    if x.ndim != 1 or q.ndim != 2 or q.shape[0] != q.shape[1] or q.shape[0] != x.size:
        raise ValueError("QUBO matrix must be square and match the bit-vector length")
    if np.any((x != 0.0) & (x != 1.0)):
        raise ValueError("QUBO variables must be binary")
    return float(x @ q @ x)


def exhaustive_qubo(qubo_matrix: np.ndarray):
    """Exhaustively enumerate all bit configurations; return (best_bits, best_energy)."""
    q = np.asarray(qubo_matrix, dtype=float)
    n = q.shape[0]
    if n > 20:
        raise ValueError("exhaustive search is limited to 20 variables")
    best_bits, best_energy = None, math.inf
    for mask in range(1 << n):
        bits = np.array([(mask >> i) & 1 for i in range(n)], dtype=float)
        energy = qubo_energy(bits, q)
        if energy < best_energy:
            best_bits, best_energy = bits, energy
    return best_bits, float(best_energy)


def simulated_annealing_qubo(qubo_matrix: np.ndarray, *, steps: int = 400,
                             initial_temperature: float = 2.0, seed: int = 7):
    """Seeded Ising-style reference solver (single-bit-flip annealing)."""
    q = np.asarray(qubo_matrix, dtype=float)
    n = q.shape[0]
    if n == 0:
        raise ValueError("the QUBO matrix must not be empty")
    if steps <= 0 or initial_temperature <= 0.0:
        raise ValueError("steps and temperature must be positive")
    rng = np.random.default_rng(seed)
    bits = (rng.random(n) < 0.5).astype(float)
    energy = qubo_energy(bits, q)
    best_bits, best_energy = bits.copy(), energy
    for step_index in range(steps):
        temperature = initial_temperature * (1.0 - step_index / steps) + 1e-6
        index = int(rng.integers(0, n))
        proposal = bits.copy()
        proposal[index] = 1.0 - proposal[index]
        proposal_energy = qubo_energy(proposal, q)
        delta = proposal_energy - energy
        if delta <= 0.0 or rng.random() < math.exp(-delta / temperature):
            bits, energy = proposal, proposal_energy
            if energy < best_energy:
                best_bits, best_energy = bits.copy(), energy
    return best_bits, float(best_energy)


# ------------------------------------------------- federated math (Steps 41-42)

def fedavg_weights(sample_counts: Sequence[int]) -> np.ndarray:
    """FedAvg aggregation weights n_k / sum_j n_j for declared participant counts."""
    counts = np.asarray(sample_counts, dtype=float)
    if counts.size == 0 or np.any(counts < 0) or not np.all(np.isfinite(counts)):
        raise ValueError("sample counts must be a non-empty array of non-negative numbers")
    total = float(np.sum(counts))
    if total <= 0.0:
        raise ValueError("at least one participant must declare a positive sample count")
    return counts / total


def clip_update_norm(update: np.ndarray, max_norm: float) -> tuple[np.ndarray, float]:
    """L2-clip a parameter update to ``max_norm``; returns (clipped, applied_norm)."""
    vector = np.asarray(update, dtype=float)
    if max_norm <= 0.0:
        raise ValueError("max_norm must be positive")
    norm = float(np.linalg.norm(vector))
    if norm <= max_norm or norm == 0.0:
        return vector, norm
    return vector * (max_norm / norm), max_norm


def dp_gaussian_noise(shape, scale: float, seed: int) -> np.ndarray:
    """Seeded Gaussian noise for differentially private updates."""
    if scale < 0.0:
        raise ValueError("noise scale must be non-negative")
    return np.random.default_rng(seed).normal(0.0, scale, size=shape)


# ------------------------------------------------- edge compression (Step 39)

def quantize_weights(weights: dict[str, np.ndarray], bits: int = 8):
    """Symmetric per-tensor quantization; returns (quantized, scales, dequantized).

    Values are mapped to signed integers in [-2^(b-1), 2^(b-1)-1] with one scale per
    tensor; dequantization multiplies back.  The uncompressed original is preserved
    by the caller for rollback (compression never destroys the reference artifact).
    """
    if bits < 2 or bits > 16:
        raise ValueError("bits must be in [2, 16]")
    if not weights:
        raise ValueError("weights must not be empty")
    quantized: dict[str, np.ndarray] = {}
    scales: dict[str, float] = {}
    dequantized: dict[str, np.ndarray] = {}
    limit = 2.0 ** (bits - 1) - 1.0
    for name, array in weights.items():
        values = np.asarray(array, dtype=float)
        max_abs = float(np.max(np.abs(values))) if values.size else 0.0
        scale = max_abs / limit if max_abs > 0.0 else 1.0
        codes = np.round(values / scale) if scale > 0.0 else np.zeros_like(values)
        codes = np.clip(codes, -limit - 1.0, limit)
        quantized[name] = codes
        scales[name] = scale
        dequantized[name] = codes * scale
    return quantized, scales, dequantized


def quantization_error(original: dict[str, np.ndarray],
                       dequantized: dict[str, np.ndarray]) -> float:
    """Root-mean-square deviation between original and dequantized weights."""
    if set(original) != set(dequantized):
        raise ValueError("weight dictionaries must cover the same tensors")
    squared, count = 0.0, 0
    for name in original:
        difference = np.asarray(original[name], dtype=float) - np.asarray(
            dequantized[name], dtype=float)
        squared += float(np.sum(difference ** 2))
        count += difference.size
    if count == 0:
        raise ValueError("weights must not be empty")
    return float(math.sqrt(squared / count))
