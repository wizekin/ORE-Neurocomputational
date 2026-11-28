"""Numeric primitives for Point 11: distributions, Bayes' rule, HMMs, evidence theory.

Every function is deterministic and pure so inference results can be reproduced from
recorded configuration alone. Uncertainty from sampling is reported explicitly instead
of being hidden inside point estimates.
"""
from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

TOLERANCE = 1e-9


def validate_distribution(name: str, values: Mapping[str, float],
                          states: Sequence[str]) -> dict[str, float]:
    """Validate a finite categorical distribution over the exact states of one variable."""
    if set(values) != set(states):
        raise ValueError(f"{name} must define exactly the states {list(states)}")
    result: dict[str, float] = {}
    total = 0.0
    for state in states:
        number = float(values[state])
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ValueError(f"{name}[{state}] must be finite and in [0, 1]")
        result[state] = number
        total += number
    if abs(total - 1.0) > TOLERANCE:
        raise ValueError(f"{name} must sum to 1 (got {total})")
    return result


def normalize_weights(name: str, weights: Mapping[str, float]) -> dict[str, float]:
    """Normalize non-negative finite weights into a probability distribution."""
    if not weights:
        raise ValueError(f"{name} must not be empty")
    cleaned: dict[str, float] = {}
    total = 0.0
    for key, value in weights.items():
        number = float(value)
        if not math.isfinite(number) or number < 0.0:
            raise ValueError(f"{name}[{key}] must be finite and non-negative")
        cleaned[str(key)] = number
        total += number
    if total <= 0.0:
        raise ValueError(f"{name} cannot normalize because the total mass is zero")
    return {key: value / total for key, value in cleaned.items()}


def shannon_entropy(distribution: Mapping[str, float]) -> float:
    """Return H(p) = -sum p log2 p in bits, ignoring zero-probability states."""
    total = sum(float(v) for v in distribution.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError("entropy requires a normalized distribution")
    return -sum(p * math.log2(p) for p in (float(v) for v in distribution.values()) if p > 0.0)


def kl_divergence(p: Mapping[str, float], q: Mapping[str, float], base: float = 2.0) -> float:
    """Return D_KL(p || q) with the natural or base-2 logarithm, checking support."""
    if set(p) != set(q):
        raise ValueError("KL divergence requires identical supports")
    total = 0.0
    for key in p:
        p_i, q_i = float(p[key]), float(q[key])
        if p_i > 0.0 and q_i <= 0.0:
            raise ValueError("KL divergence is infinite when q assigns zero to a state p supports")
        if p_i > 0.0:
            total += p_i * math.log(p_i / q_i, base)
    return total


def jensen_shannon_divergence(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    """Return the Jensen-Shannon divergence (base 2, bounded in [0, 1]) between distributions."""
    if set(p) != set(q):
        raise ValueError("Jensen-Shannon divergence requires identical supports")
    keys = sorted(p)
    p_vec = [float(p[k]) for k in keys]
    q_vec = [float(q[k]) for k in keys]
    m_vec = [(a + b) / 2.0 for a, b in zip(p_vec, q_vec)]

    def _entropy(vec: Sequence[float]) -> float:
        return -sum(x * math.log2(x) for x in vec if x > 0.0)

    return max(0.0, _entropy(m_vec) - 0.5 * _entropy(p_vec) - 0.5 * _entropy(q_vec))


def bayes_rule(prior: Mapping[str, float], likelihood: Mapping[str, float]) -> dict[str, float]:
    """Apply Bayes' rule P(H|E) = P(E|H)P(H) / P(E) with explicit normalization.

    The prior must be a normalized distribution and the likelihood must assign a finite
    non-negative value to every hypothesis. The evidence normalization P(E) is computed
    explicitly, so a zero denominator is surfaced instead of being silently replaced.
    """
    if set(prior) != set(likelihood):
        raise ValueError("Bayes' rule requires the prior and likelihood to share supports")
    prior_checked = validate_distribution("prior", prior, list(prior))
    posterior_weights: dict[str, float] = {}
    evidence_probability = 0.0
    for hypothesis, prior_value in prior_checked.items():
        like = float(likelihood[hypothesis])
        if not math.isfinite(like) or like < 0.0:
            raise ValueError(f"likelihood[{hypothesis}] must be finite and non-negative")
        weight = prior_value * like
        posterior_weights[hypothesis] = weight
        evidence_probability += weight
    if evidence_probability <= 0.0:
        raise ValueError("evidence probability P(E) is zero; the likelihood excludes every hypothesis")
    return {hypothesis: weight / evidence_probability
            for hypothesis, weight in posterior_weights.items()}


def bayes_update_sequential(prior: Mapping[str, float],
                            likelihoods: Sequence[Mapping[str, float]],
                            conditional_independence_justified: bool = False,
                            ) -> tuple[dict[str, float], list[dict[str, float]], bool]:
    """Update a prior with several observations, requiring an independence justification.

    Returns the final posterior, the per-observation posteriors, and whether the
    conditional-independence assumption was documented before repeated updating.
    """
    if not conditional_independence_justified and len(likelihoods) > 1:
        raise ValueError("multiple observations require a justified conditional-independence claim")
    current = validate_distribution("prior", prior, list(prior))
    history: list[dict[str, float]] = []
    for index, likelihood in enumerate(likelihoods, start=1):
        current = bayes_rule(current, likelihood)
        history.append(dict(current))
    return dict(current), history, conditional_independence_justified


def validate_stochastic_matrix(name: str, matrix: Mapping[str, Mapping[str, float]],
                               row_states: Sequence[str], column_states: Sequence[str]) -> dict[str, dict[str, float]]:
    """Validate that a transition or emission matrix has exact shape and normalized rows."""
    if set(matrix) != set(row_states):
        raise ValueError(f"{name} rows must be exactly {list(row_states)}")
    checked: dict[str, dict[str, float]] = {}
    for row in row_states:
        checked[row] = validate_distribution(f"{name}[{row}]", matrix[row], column_states)
    return checked


def validate_hmm_spec(hidden_states: Sequence[str], observation_states: Sequence[str],
                      initial_distribution: Mapping[str, float],
                      transition_matrix: Mapping[str, Mapping[str, float]],
                      emission_matrix: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    """Validate an HMM's stochastic matrices and state/observation mapping."""
    if len(set(hidden_states)) != len(hidden_states) or not hidden_states:
        raise ValueError("hidden states must be unique and non-empty")
    if len(set(observation_states)) != len(observation_states) or not observation_states:
        raise ValueError("observation states must be unique and non-empty")
    initial = validate_distribution("initial_distribution", initial_distribution, hidden_states)
    transitions = validate_stochastic_matrix("transition_matrix", transition_matrix,
                                             hidden_states, hidden_states)
    emissions = validate_stochastic_matrix("emission_matrix", emission_matrix,
                                           hidden_states, observation_states)
    return {"hidden_states": list(hidden_states), "observation_states": list(observation_states),
            "initial_distribution": initial, "transition_matrix": transitions,
            "emission_matrix": emissions}


def forward_filter(hidden_states: Sequence[str], transitions: Mapping[str, Mapping[str, float]],
                   emissions: Mapping[str, Mapping[str, float]],
                   initial_distribution: Mapping[str, float],
                   observations: Sequence[str | None],
                   ) -> dict[str, Any]:
    """Run HMM filtering P(z_t | x_1:t), treating missing observations explicitly.

    A missing observation (None) contributes no emission update: the prediction step is
    carried forward and the step is flagged as missing so stale state estimates are never
    silently presented as filtered evidence.
    """
    current = validate_distribution("initial_distribution", initial_distribution, hidden_states)
    filtered: list[dict[str, float]] = []
    missing_steps: list[int] = []
    log_likelihood = 0.0
    for step, observation in enumerate(observations, start=1):
        predicted = {
            state: sum(transitions[previous][state] * current[previous] for previous in hidden_states)
            for state in hidden_states
        }
        if observation is None:
            missing_steps.append(step)
            current = normalize_weights("predicted_state", predicted)
            filtered.append(dict(current))
            continue
        if observation not in next(iter(emissions.values())):
            raise ValueError(f"observation {observation} is not in the emission alphabet")
        unnormalized = {
            state: emissions[state][observation] * predicted[state] for state in hidden_states
        }
        total = sum(unnormalized.values())
        if total <= 0.0:
            raise ValueError(f"emission {observation} has zero probability under every hidden state")
        log_likelihood += math.log(total)
        current = {state: value / total for state, value in unnormalized.items()}
        filtered.append(dict(current))
    return {"filtered": filtered, "missing_steps": missing_steps,
            "log_likelihood": log_likelihood, "final": dict(current)}


def backward_smoothing(hidden_states: Sequence[str], transitions: Mapping[str, Mapping[str, float]],
                       emissions: Mapping[str, Mapping[str, float]],
                       observations: Sequence[str | None],
                       initial_distribution: Mapping[str, float] | None = None) -> dict[str, Any]:
    """Run the forward-backward smoother P(z_t | x_1:T) over a complete observation sequence.

    Smoothing is only invoked when the future observations are actually available; each
    missing observation is skipped in both passes and the step keeps its explicit flag.
    """
    length = len(observations)
    if length == 0:
        raise ValueError("smoothing requires at least one observation")
    forward = forward_filter(hidden_states, transitions, emissions,
                             initial_distribution or _uniform_initial(hidden_states),
                             observations)
    alphas = forward["filtered"]
    betas: list[dict[str, float]] = [dict.fromkeys(hidden_states, 1.0) for _ in range(length)]
    for step in range(length - 2, -1, -1):
        observation = observations[step + 1]
        for state in hidden_states:
            total = 0.0
            for successor in hidden_states:
                emission = 1.0 if observation is None else emissions[successor][observation]
                total += transitions[state][successor] * emission * betas[step + 1][successor]
            betas[step][state] = total
    smoothed: list[dict[str, float]] = []
    for step in range(length):
        weights = {state: alphas[step][state] * betas[step][state] for state in hidden_states}
        smoothed.append(normalize_weights(f"smoothed[{step}]", weights))
    return {"smoothed": smoothed, "missing_steps": forward["missing_steps"],
            "log_likelihood": forward["log_likelihood"]}


def _uniform_initial(hidden_states: Sequence[str]) -> dict[str, float]:
    """Return the uniform initial distribution used when smoothing recomputes the forward pass."""
    return {state: 1.0 / len(hidden_states) for state in hidden_states}


def viterbi_decode(hidden_states: Sequence[str], transitions: Mapping[str, Mapping[str, float]],
                   emissions: Mapping[str, Mapping[str, float]],
                   initial_distribution: Mapping[str, float],
                   observations: Sequence[str | None],
                   ) -> dict[str, Any]:
    """Infer the most likely hidden-state sequence with the Viterbi dynamic program.

    Missing observations contribute no emission factor (all emission scores equal one),
    and every missing step is reported so the decode cannot claim observed support.
    """
    if not observations:
        raise ValueError("Viterbi decoding requires at least one observation")
    scores: list[dict[str, float]] = []
    backpointers: list[dict[str, str]] = []
    first_observation = observations[0]
    initial = {
        state: float(initial_distribution[state]) *
        (1.0 if first_observation is None else emissions[state][first_observation])
        for state in hidden_states
    }
    scores.append(normalize_weights("initial_viterbi", initial))
    backpointers.append(dict.fromkeys(hidden_states, hidden_states[0]))
    for step in range(1, len(observations)):
        observation = observations[step]
        row: dict[str, float] = {}
        pointers: dict[str, str] = {}
        for state in hidden_states:
            best_previous, best_score = None, -1.0
            for previous in hidden_states:
                candidate = scores[step - 1][previous] * transitions[previous][state]
                if candidate > best_score:
                    best_previous, best_score = previous, candidate
            emission = 1.0 if observation is None else emissions[state][observation]
            row[state] = best_score * emission
            pointers[state] = best_previous or hidden_states[0]
        scores.append(normalize_weights(f"viterbi[{step}]", row))
        backpointers.append(pointers)
    final_state = max(scores[-1], key=lambda state: scores[-1][state])
    path = [final_state]
    for step in range(len(observations) - 1, 0, -1):
        path.append(backpointers[step][path[-1]])
    path.reverse()
    missing_steps = [index for index, value in enumerate(observations, start=1) if value is None]
    return {"path": path, "final_scores": dict(scores[-1]),
            "missing_steps": missing_steps, "path_probability_note":
            "scores are renormalized per step; the argmax path ordering is invariant"}


def detect_distribution_shift(filtered: Sequence[Mapping[str, float]],
                              reference: Mapping[str, float] | None = None,
                              threshold: float = 0.25) -> dict[str, Any]:
    """Flag time steps where the filtered state diverges from a reference distribution.

    The reference is the mean filtered distribution when not supplied. Divergence is the
    Jensen-Shannon divergence; flagged steps expose uncertainty instead of silently
    carrying stale state estimates.
    """
    if not filtered:
        raise ValueError("shift detection requires at least one filtered distribution")
    states = sorted(filtered[0])
    if reference is None:
        reference = {
            state: sum(step[state] for step in filtered) / len(filtered) for state in states
        }
    flagged: list[dict[str, Any]] = []
    divergences: list[float] = []
    for index, step in enumerate(filtered, start=1):
        divergence = jensen_shannon_divergence(step, reference)
        divergences.append(divergence)
        if divergence > threshold:
            flagged.append({"step": index, "js_divergence": divergence,
                            "interpretation": "filtered_state_diverged_from_reference"})
    return {"reference": dict(reference), "divergences": divergences,
            "flagged_steps": flagged, "threshold": threshold,
            "max_divergence": max(divergences) if divergences else 0.0}


@dataclass(frozen=True)
class DempsterShaferResult:
    """Hold a combined mass function with its conflict mass and belief/plausibility bounds."""

    combined: Mapping[frozenset[str], float]
    conflict_mass: float
    belief: Mapping[frozenset[str], float]
    plausibility: Mapping[frozenset[str], float]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly view with sorted set keys."""
        def _key(value: frozenset[str]) -> str:
            return "{" + ",".join(sorted(value)) + "}" if value else "{}"

        return {
            "combined": {_key(key): float(value) for key, value in self.combined.items()},
            "conflict_mass": self.conflict_mass,
            "belief": {_key(key): float(value) for key, value in self.belief.items()},
            "plausibility": {_key(key): float(value) for key, value in self.plausibility.items()},
        }


def combine_mass_functions(first: Mapping[frozenset[str], float],
                           second: Mapping[frozenset[str], float],
                           ) -> DempsterShaferResult:
    """Combine two mass functions with Dempster's rule, preserving the conflict mass.

    Conflict K is the total mass of contradictory pairs. When K is large the combination
    is reported as strongly conflicted rather than being normalized into false agreement.
    """
    if not (0.0 <= sum(first.values()) <= 1.0 + 1e-9) or not (0.0 <= sum(second.values()) <= 1.0 + 1e-9):
        raise ValueError("mass functions must have total mass in [0, 1]")
    combined: dict[frozenset[str], float] = {}
    conflict = 0.0
    for subset_a, mass_a in first.items():
        for subset_b, mass_b in second.items():
            intersection = subset_a & subset_b
            if not intersection:
                conflict += mass_a * mass_b
            else:
                combined[intersection] = combined.get(intersection, 0.0) + mass_a * mass_b
    if conflict >= 1.0:
        return DempsterShaferResult({}, 1.0, {}, {})
    normalized = {key: value / (1.0 - conflict) for key, value in combined.items()}
    belief: dict[frozenset[str], float] = {}
    plausibility: dict[frozenset[str], float] = {}
    all_keys = set(normalized) | {frozenset()}
    for subset in all_keys:
        belief[subset] = sum(value for key, value in normalized.items() if key <= subset)
        plausibility[subset] = sum(value for key, value in normalized.items() if key & subset)
    return DempsterShaferResult(normalized, conflict, belief, plausibility)


def evidence_to_mass(states: Sequence[str], supported_state: str,
                     reliability: float) -> dict[frozenset[str], float]:
    """Convert one piece of evidence into a mass function over one variable's states.

    The supported state receives mass proportional to source reliability; the remaining
    mass stays with the whole frame (ignorance) instead of being forced onto negation.
    """
    if supported_state not in states:
        raise ValueError(f"supported state {supported_state} is not part of the frame")
    reliability = min(max(float(reliability), 0.0), 1.0)
    return {frozenset({supported_state}): reliability, frozenset(states): 1.0 - reliability}


def brier_score(predictions: Sequence[float], labels: Sequence[int]) -> float:
    """Return the mean Brier score over held-out labelled predictions."""
    if len(predictions) != len(labels) or not predictions:
        raise ValueError("Brier score needs equally sized, non-empty prediction/label sequences")
    total = 0.0
    for prediction, label in zip(predictions, labels):
        if not 0.0 <= float(prediction) <= 1.0 or label not in (0, 1):
            raise ValueError("predictions must lie in [0, 1] and labels must be 0 or 1")
        total += (float(prediction) - float(label)) ** 2
    return total / len(predictions)


def log_loss(predictions: Sequence[float], labels: Sequence[int],
             clip: float = 1e-12) -> float:
    """Return the mean log loss with clipping to keep logarithms finite."""
    if len(predictions) != len(labels) or not predictions:
        raise ValueError("log loss needs equally sized, non-empty prediction/label sequences")
    total = 0.0
    for prediction, label in zip(predictions, labels):
        probability = min(max(float(prediction), clip), 1.0 - clip)
        if label not in (0, 1):
            raise ValueError("labels must be 0 or 1")
        total += -math.log(probability) if label == 1 else -math.log(1.0 - probability)
    return total / len(predictions)


def reliability_diagram(predictions: Sequence[float], labels: Sequence[int],
                        bins: int = 5) -> list[dict[str, float]]:
    """Bin predictions and report per-bin confidence, accuracy and count."""
    if bins < 2:
        raise ValueError("reliability diagrams need at least two bins")
    records = [(float(p), int(y)) for p, y in zip(predictions, labels)]
    edges = [index / bins for index in range(bins + 1)]
    diagram: list[dict[str, float]] = []
    for index in range(bins):
        low, high = edges[index], edges[index + 1]
        in_bin = [record for record in records
                  if (low <= record[0] < high or (index == bins - 1 and record[0] == high))]
        count = len(in_bin)
        diagram.append({
            "bin_low": low, "bin_high": high, "count": count,
            "confidence": (sum(p for p, _ in in_bin) / count) if count else 0.0,
            "accuracy": (sum(y for _, y in in_bin) / count) if count else 0.0,
        })
    return diagram


def expected_calibration_error(predictions: Sequence[float], labels: Sequence[int],
                               bins: int = 5) -> float:
    """Return the expected calibration error over reliability-diagram bins."""
    diagram = reliability_diagram(predictions, labels, bins=bins)
    total = sum(row["count"] for row in diagram)
    if total == 0:
        raise ValueError("calibration error needs at least one labelled prediction")
    return sum(row["count"] / total * abs(row["confidence"] - row["accuracy"])
               for row in diagram)


def effective_sample_size(weights: Sequence[float]) -> float:
    """Return Kish effective sample size (sum w)^2 / sum w^2 for weighted samples."""
    values = [float(weight) for weight in weights]
    if any(value < 0.0 or not math.isfinite(value) for value in values):
        raise ValueError("weights must be finite and non-negative")
    total = sum(values)
    if total <= 0.0:
        return 0.0
    return total * total / sum(value * value for value in values)


def seeded_random(seed: int) -> random.Random:
    """Return a deterministic random generator for reproducible sampling."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    return random.Random(int(seed))


def sample_from_distribution(distribution: Mapping[str, float],
                             rng: random.Random) -> str:
    """Draw one state from a categorical distribution using a supplied generator."""
    total = sum(float(value) for value in distribution.values())
    if total <= 0.0:
        raise ValueError("cannot sample from a zero-mass distribution")
    threshold = rng.random() * total
    cumulative = 0.0
    ordered = sorted(distribution)
    for index, state in enumerate(ordered):
        cumulative += float(distribution[state])
        if cumulative >= threshold or index == len(ordered) - 1:
            return state
    raise AssertionError("unreachable sampling fallback")


def weighted_std_error(values: Iterable[float], weights: Iterable[float]) -> float | None:
    """Return weighted standard error, or None when effective sample size is insufficient."""
    value_list = [float(value) for value in values]
    weight_list = [float(weight) for weight in weights]
    if len(value_list) != len(weight_list) or not value_list:
        raise ValueError("standard error needs equally sized non-empty sequences")
    if any(not math.isfinite(value) for value in value_list):
        raise ValueError("standard-error values must be finite")
    ess = effective_sample_size(weight_list)
    if ess <= 1.0:
        return None
    mean = sum(v * w for v, w in zip(value_list, weight_list)) / sum(weight_list)
    variance = sum(w * (v - mean) ** 2 for v, w in zip(value_list, weight_list)) / sum(weight_list)
    return math.sqrt(max(variance, 0.0) / ess)
