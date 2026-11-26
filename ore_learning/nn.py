"""Deterministic numpy reference models for the RLMS laboratory (Point 14).

These small models are the substrate for Steps 12-17 (transfer learning, TTT),
Steps 23-25 (GAN and diffusion experiments), Steps 39-43 (compression, distillation,
federated aggregation) and the executor's mock specialists.  They are deliberately
tiny, fully deterministic under a fixed seed, and self-contained: the RLMS treats
every model -- including these -- as a replaceable artifact behind typed adapters,
never as an authority.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .functional import mse, sigmoid, softmax

__all__ = (
    "BigramLM",
    "MaskDenoiserLM",
    "TinyDiscriminator",
    "TinyGenerator",
    "TinyLinearModel",
    "TinyMLP",
    "parameter_digest",
)


def parameter_digest(weights: Mapping[str, np.ndarray]) -> str:
    """Stable sha256 digest over the canonical serialization of weight tensors."""
    payload = {name: np.asarray(array, dtype=float).round(12).tolist()
               for name, array in sorted(weights.items())}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


class _ParameterizedModel:
    """Shared plumbing for the tiny models: parameters, digests, copies, sizes."""

    def parameter_names(self) -> tuple[str, ...]:
        """Names of the trainable tensors, in canonical order."""
        return tuple(sorted(self.parameters()))

    def parameters(self) -> dict[str, np.ndarray]:
        """Return the trainable tensors (subclasses override)."""
        raise NotImplementedError

    def set_parameters(self, weights: Mapping[str, np.ndarray]) -> None:
        """Copy the supplied tensors into the model."""
        current = self.parameters()
        if set(weights) != set(current):
            raise ValueError("parameter sets must match exactly")
        for name, array in weights.items():
            current[name][...] = np.asarray(array, dtype=float)

    def parameter_vector(self) -> np.ndarray:
        """Flatten all tensors into one vector (FedAvg operates on these)."""
        return np.concatenate([np.asarray(array, dtype=float).ravel()
                               for name in self.parameter_names()
                               for array in [self.parameters()[name]]])

    def set_parameter_vector(self, vector: Sequence[float]) -> None:
        """Restore parameters from a flat vector produced by ``parameter_vector``."""
        flat = np.asarray(vector, dtype=float)
        expected = self.parameter_vector().size
        if flat.size != expected:
            raise ValueError(f"parameter vector has {flat.size} entries, expected {expected}")
        cursor = 0
        for name in self.parameter_names():
            array = self.parameters()[name]
            size = array.size
            array[...] = flat[cursor:cursor + size].reshape(array.shape)
            cursor += size

    def num_parameters(self) -> int:
        """Total number of scalar parameters."""
        return int(sum(np.asarray(array).size for array in self.parameters().values()))

    def memory_bytes(self, dtype_bytes: int = 4) -> int:
        """Memory footprint of the parameters at the given precision."""
        return self.num_parameters() * dtype_bytes

    def digest(self) -> str:
        """Artifact digest of the current parameters."""
        return parameter_digest(self.parameters())

    def copy(self) -> "_ParameterizedModel":
        """Deep copy with independent tensors."""
        clone = type(self).from_config(self.config_dict())
        clone.set_parameters({name: np.asarray(array).copy()
                              for name, array in self.parameters().items()})
        return clone

    def config_dict(self) -> dict[str, Any]:
        """Constructor arguments as a JSON-safe dictionary."""
        raise NotImplementedError

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "_ParameterizedModel":
        """Rebuild an instance from ``config_dict`` output."""
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        """Serialize the model with its parameters for artifact storage."""
        return {"class": type(self).__name__, "config": self.config_dict(),
                "weights": {name: np.asarray(array).round(12).tolist()
                            for name, array in self.parameters().items()}}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "_ParameterizedModel":
        """Restore a serialized model artifact."""
        registry = {model.__name__: model for model in
                    (TinyLinearModel, TinyMLP, TinyGenerator, TinyDiscriminator,
                     BigramLM, MaskDenoiserLM)}
        model_class = registry.get(str(payload.get("class")))
        if model_class is None:
            raise ValueError(f"unknown model class {payload.get('class')!r}")
        model = model_class.from_config(payload["config"])
        model.set_parameters({name: np.asarray(array, dtype=float)
                              for name, array in payload["weights"].items()})
        return model


class TinyLinearModel(_ParameterizedModel):
    """A linear scorer/regressor: y = x @ W + b.

    Used for transfer-learning experiments (frozen-feature reuse vs adaptation),
    SHAP/LIME diagnostics with known ground-truth coefficients, and quantization
    demonstrations where the exact answer is computable in closed form.
    """

    def __init__(self, n_inputs: int, n_outputs: int = 1, seed: int = 11) -> None:
        if n_inputs <= 0 or n_outputs <= 0:
            raise ValueError("dimensions must be positive")
        rng = np.random.default_rng(seed)
        self.n_inputs = int(n_inputs)
        self.n_outputs = int(n_outputs)
        self.weights: dict[str, np.ndarray] = {
            "W": rng.normal(0.0, 0.1, size=(n_inputs, n_outputs)),
            "b": np.zeros(n_outputs),
        }

    def parameters(self) -> dict[str, np.ndarray]:
        """Return the weight and bias tensors."""
        return self.weights

    def config_dict(self) -> dict[str, Any]:
        """Constructor configuration."""
        return {"n_inputs": self.n_inputs, "n_outputs": self.n_outputs,
                "seed": 11}

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "TinyLinearModel":
        """Rebuild from a config dictionary."""
        return TinyLinearModel(int(config["n_inputs"]), int(config["n_outputs"]),
                               int(config.get("seed", 11)))

    def forward(self, x) -> np.ndarray:
        """Score a batch of row vectors."""
        matrix = np.atleast_2d(np.asarray(x, dtype=float))
        if matrix.shape[1] != self.n_inputs:
            raise ValueError(f"expected {self.n_inputs} inputs, received {matrix.shape[1]}")
        return matrix @ self.weights["W"] + self.weights["b"]

    def predict(self, x) -> np.ndarray:
        """Regression output."""
        return self.forward(x)

    def predict_proba(self, x) -> np.ndarray:
        """Class-1 probability for a single-output model."""
        scores = self.forward(x)
        if scores.shape[1] != 1:
            raise ValueError("predict_proba requires a single output head")
        return sigmoid(scores)[:, 0]

    def fit(self, x, y, *, epochs: int = 60, learning_rate: float = 0.05,
            batch_size: int = 8, seed: int = 0, l2: float = 0.0) -> list[float]:
        """Deterministic mini-batch gradient descent on squared error."""
        matrix = np.atleast_2d(np.asarray(x, dtype=float))
        target = np.atleast_2d(np.asarray(y, dtype=float))
        if matrix.shape[0] != target.shape[0]:
            raise ValueError("features and targets must align")
        if target.shape[1] != self.n_outputs:
            raise ValueError("target width must match the output head")
        losses: list[float] = []
        order = np.arange(matrix.shape[0])
        rng = np.random.default_rng(seed)
        for _ in range(epochs):
            rng.shuffle(order)
            for start in range(0, matrix.shape[0], batch_size):
                batch = order[start:start + batch_size]
                prediction = matrix[batch] @ self.weights["W"] + self.weights["b"]
                error = prediction - target[batch]
                gradient_w = matrix[batch].T @ error / batch.size + l2 * self.weights["W"]
                gradient_b = np.mean(error, axis=0)
                self.weights["W"] -= learning_rate * gradient_w
                self.weights["b"] -= learning_rate * gradient_b
            losses.append(mse(self.predict(matrix), target))
        return losses

    def train_output_layer(self, x, gradient_output, learning_rate) -> None:
        """Linear-probe step: update only the output weights from dL/dOutput."""
        matrix = np.atleast_2d(np.asarray(x, dtype=float))
        gradient = np.atleast_2d(np.asarray(gradient_output, dtype=float))
        self.weights["W"] -= learning_rate * (matrix.T @ gradient)
        self.weights["b"] -= learning_rate * np.mean(gradient, axis=0)

    def _full_finetune_step(self, x, target, learning_rate) -> float:
        """One full fine-tuning gradient step; returns the batch loss."""
        matrix = np.atleast_2d(np.asarray(x, dtype=float))
        desired = np.atleast_2d(np.asarray(target, dtype=float))
        prediction = matrix @ self.weights["W"] + self.weights["b"]
        error = prediction - desired
        loss = float(np.mean(error ** 2))
        gradient = 2.0 * error / matrix.shape[0]
        self.weights["W"] -= learning_rate * (matrix.T @ gradient)
        self.weights["b"] -= learning_rate * np.mean(gradient, axis=0)
        return loss


class TinyMLP(_ParameterizedModel):
    """A one-hidden-layer tanh network with an optional sigmoid output head.

    The generator and discriminator of the GAN laboratory and the students of the
    distillation experiments are instances of this class.
    """

    def __init__(self, n_inputs: int, n_hidden: int, n_outputs: int, *,
                 output_activation: str = "identity", seed: int = 13) -> None:
        if min(n_inputs, n_hidden, n_outputs) <= 0:
            raise ValueError("dimensions must be positive")
        if output_activation not in ("identity", "sigmoid"):
            raise ValueError("output_activation must be 'identity' or 'sigmoid'")
        rng = np.random.default_rng(seed)
        self.n_inputs = int(n_inputs)
        self.n_hidden = int(n_hidden)
        self.n_outputs = int(n_outputs)
        self.output_activation = output_activation
        self.weights: dict[str, np.ndarray] = {
            "W1": rng.normal(0.0, 0.3, size=(n_inputs, n_hidden)) / math.sqrt(n_inputs),
            "b1": np.zeros(n_hidden),
            "W2": rng.normal(0.0, 0.3, size=(n_hidden, n_outputs)) / math.sqrt(n_hidden),
            "b2": np.zeros(n_outputs),
        }

    def parameters(self) -> dict[str, np.ndarray]:
        """Return the four trainable tensors."""
        return self.weights

    def config_dict(self) -> dict[str, Any]:
        """Constructor configuration."""
        return {"n_inputs": self.n_inputs, "n_hidden": self.n_hidden,
                "n_outputs": self.n_outputs, "output_activation": self.output_activation,
                "seed": 13}

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "TinyMLP":
        """Rebuild from a config dictionary."""
        return TinyMLP(int(config["n_inputs"]), int(config["n_hidden"]),
                       int(config["n_outputs"]),
                       output_activation=str(config.get("output_activation", "identity")),
                       seed=int(config.get("seed", 13)))

    def forward(self, x):
        """Forward pass returning (output, cache) where cache supports backprop."""
        matrix = np.atleast_2d(np.asarray(x, dtype=float))
        if matrix.shape[1] != self.n_inputs:
            raise ValueError(f"expected {self.n_inputs} inputs, received {matrix.shape[1]}")
        hidden = np.tanh(matrix @ self.weights["W1"] + self.weights["b1"])
        output = hidden @ self.weights["W2"] + self.weights["b2"]
        if self.output_activation == "sigmoid":
            output = sigmoid(output)
        if np.asarray(x).ndim == 1 and matrix.shape[0] == 1:
            return (output[0], (matrix, hidden))
        return output, (matrix, hidden)

    def predict(self, x) -> np.ndarray:
        """Forward pass without the backprop cache."""
        output, _ = self.forward(x)
        return np.atleast_2d(output)

    def backward(self, cache, gradient_output: np.ndarray) -> dict[str, np.ndarray]:
        """Backpropagate dL/dOutput through the tanh hidden layer."""
        matrix, hidden = cache
        gradient_output = np.atleast_2d(np.asarray(gradient_output, dtype=float))
        if self.output_activation == "sigmoid":
            # d sigmoid / dz = s(1-s); caller supplies dL/ds
            output = hidden @ self.weights["W2"] + self.weights["b2"]
            sigmoid_out = sigmoid(output)
            gradient_output = gradient_output * sigmoid_out * (1.0 - sigmoid_out)
        gradient_w2 = hidden.T @ gradient_output
        gradient_b2 = np.sum(gradient_output, axis=0)
        gradient_hidden = gradient_output @ self.weights["W2"].T
        gradient_pre = gradient_hidden * (1.0 - hidden ** 2)
        gradient_w1 = matrix.T @ gradient_pre
        gradient_b1 = np.sum(gradient_pre, axis=0)
        return {"W1": gradient_w1, "b1": gradient_b1,
                "W2": gradient_w2, "b2": gradient_b2}

    def input_gradient(self, cache, gradient_output: np.ndarray) -> np.ndarray:
        """Backpropagate dL/dOutput to dL/dInput (used by the GAN generator step)."""
        matrix, hidden = cache
        gradient_output = np.atleast_2d(np.asarray(gradient_output, dtype=float))
        if self.output_activation == "sigmoid":
            output = hidden @ self.weights["W2"] + self.weights["b2"]
            sigmoid_out = sigmoid(output)
            gradient_output = gradient_output * sigmoid_out * (1.0 - sigmoid_out)
        gradient_hidden = gradient_output @ self.weights["W2"].T
        gradient_pre = gradient_hidden * (1.0 - hidden ** 2)
        return gradient_pre @ self.weights["W1"].T

    def sgd_step(self, gradients: Mapping[str, np.ndarray],
                 learning_rate: float) -> None:
        """Apply one gradient step."""
        for name, gradient in gradients.items():
            self.weights[name] -= learning_rate * np.asarray(gradient, dtype=float)

    def fit(self, x, y, *, epochs: int = 100, learning_rate: float = 0.05,
            seed: int = 0) -> list[float]:
        """Full-batch squared-error training (deterministic)."""
        matrix = np.atleast_2d(np.asarray(x, dtype=float))
        target = np.atleast_2d(np.asarray(y, dtype=float))
        losses: list[float] = []
        for _ in range(epochs):
            output, cache = self.forward(matrix)
            losses.append(float(np.mean((output - target) ** 2)))
            gradient = 2.0 * (output - target) / output.shape[0]
            self.sgd_step(self.backward(cache, gradient), learning_rate)
        return losses

    def train_output_layer(self, x, gradient_output, learning_rate) -> None:
        """Linear-probe step: update only the output head from dL/dOutput."""
        matrix = np.atleast_2d(np.asarray(x, dtype=float))
        _, cache = self.forward(matrix)
        gradient = np.atleast_2d(np.asarray(gradient_output, dtype=float))
        if self.output_activation == "sigmoid":
            output = self.predict(matrix)
            gradient = gradient * output * (1.0 - output)
        gradients = self.backward(cache, gradient)
        self.weights["W2"] -= learning_rate * gradients["W2"]
        self.weights["b2"] -= learning_rate * gradients["b2"]

    def _full_finetune_step(self, x, target, learning_rate) -> float:
        """One full fine-tuning gradient step; returns the batch loss."""
        matrix = np.atleast_2d(np.asarray(x, dtype=float))
        desired = np.atleast_2d(np.asarray(target, dtype=float))
        output, cache = self.forward(matrix)
        loss = float(np.mean((output - desired) ** 2))
        gradient = 2.0 * (output - desired) / matrix.shape[0]
        self.sgd_step(self.backward(cache, gradient), learning_rate)
        return loss


class TinyGenerator:
    """Noise-to-data generator for the GAN laboratory (a TinyMLP wrapper)."""

    def __init__(self, noise_dim: int = 4, data_dim: int = 2, hidden: int = 8,
                 seed: int = 21) -> None:
        """Build the underlying network."""
        self.noise_dim = int(noise_dim)
        self.data_dim = int(data_dim)
        self.hidden = int(hidden)
        self.seed = int(seed)
        self.network = TinyMLP(self.noise_dim, self.hidden, self.data_dim,
                               output_activation="identity", seed=self.seed)

    def generate(self, noise) -> np.ndarray:
        """Map a noise batch to synthetic samples."""
        return self.network.predict(noise)

    def sample(self, n: int, seed: int = 5) -> np.ndarray:
        """Draw ``n`` synthetic samples from seeded Gaussian noise."""
        noise = np.random.default_rng(seed).normal(size=(n, self.noise_dim))
        return self.generate(noise)


class TinyDiscriminator:
    """Sample-to-probability discriminator for the GAN laboratory."""

    def __init__(self, data_dim: int = 2, hidden: int = 8, seed: int = 22) -> None:
        """Build the underlying network."""
        self.data_dim = int(data_dim)
        self.hidden = int(hidden)
        self.seed = int(seed)
        self.network = TinyMLP(self.data_dim, self.hidden, 1,
                               output_activation="sigmoid", seed=self.seed)

    def score(self, samples) -> np.ndarray:
        """Return D(x) in [0, 1] for a batch of samples."""
        return self.network.predict(samples)[:, 0]


class BigramLM(_ParameterizedModel):
    """Autoregressive baseline language model over a tiny integer vocabulary.

    Predicts the next token from the current token with additive-smoothed
    conditional counts; generation is strictly left-to-right (Step 25 baseline).
    """

    def __init__(self, vocab_size: int = 8) -> None:
        if vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")
        self.vocab_size = int(vocab_size)
        self.counts: np.ndarray = np.zeros((vocab_size, vocab_size))

    def parameters(self) -> dict[str, np.ndarray]:
        """The transition count matrix is the single parameter tensor."""
        return {"counts": self.counts}

    def set_parameters(self, weights: Mapping[str, np.ndarray]) -> None:
        """Copy the transition matrix."""
        array = np.asarray(weights["counts"], dtype=float)
        if array.shape != self.counts.shape:
            raise ValueError("transition matrix shape mismatch")
        self.counts[...] = array

    def config_dict(self) -> dict[str, Any]:
        """Constructor configuration."""
        return {"vocab_size": self.vocab_size}

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "BigramLM":
        """Rebuild from a config dictionary."""
        return BigramLM(int(config["vocab_size"]))

    def fit(self, sequences: Sequence[Sequence[int]]) -> None:
        """Accumulate transition counts from integer sequences."""
        for sequence in sequences:
            for current, nxt in zip(sequence, sequence[1:]):
                self.counts[current, nxt] += 1.0

    def next_distribution(self, token: int) -> np.ndarray:
        """Additive-smoothed conditional distribution over the next token."""
        row = self.counts[token] + 0.25
        return row / np.sum(row)

    def generate(self, prompt: Sequence[int], length: int) -> list[int]:
        """Greedy left-to-right continuation, one token per step."""
        tokens = [int(t) for t in prompt]
        for _ in range(length):
            distribution = self.next_distribution(tokens[-1])
            tokens.append(int(np.argmax(distribution)))
        return tokens


class MaskDenoiserLM(_ParameterizedModel):
    """Discrete-diffusion reference: iteratively mask and denoise token sequences.

    Trained (here: counted) to restore a masked token from its unmasked context;
    generation proceeds by parallel refinement steps rather than left-to-right
    decoding, which is the property Step 25 asks to benchmark against the
    autoregressive baseline.
    """

    def __init__(self, vocab_size: int = 8, sequence_length: int = 6) -> None:
        if vocab_size < 2 or sequence_length < 2:
            raise ValueError("vocab_size and sequence_length must be at least 2")
        self.vocab_size = int(vocab_size)
        self.sequence_length = int(sequence_length)
        self.mask_token = vocab_size
        # context-counts[position, observed_token, candidate_token]
        self.context_counts: np.ndarray = np.zeros(
            (sequence_length, vocab_size + 1, vocab_size))

    def parameters(self) -> dict[str, np.ndarray]:
        """The context-count tensor is the single parameter tensor."""
        return {"context_counts": self.context_counts}

    def set_parameters(self, weights: Mapping[str, np.ndarray]) -> None:
        """Copy the context-count tensor."""
        array = np.asarray(weights["context_counts"], dtype=float)
        if array.shape != self.context_counts.shape:
            raise ValueError("context-count tensor shape mismatch")
        self.context_counts[...] = array

    def config_dict(self) -> dict[str, Any]:
        """Constructor configuration."""
        return {"vocab_size": self.vocab_size, "sequence_length": self.sequence_length}

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "MaskDenoiserLM":
        """Rebuild from a config dictionary."""
        return MaskDenoiserLM(int(config["vocab_size"]),
                              int(config["sequence_length"]))

    def fit(self, sequences: Sequence[Sequence[int]]) -> None:
        """Count context/target co-occurrences for every masking position."""
        for sequence in sequences:
            for position in range(self.sequence_length):
                context = sequence[position]
                for target in range(self.sequence_length):
                    if target == position:
                        continue
                    self.context_counts[position, sequence[target], sequence[context]] += 1.0

    def denoise(self, corrupted: Sequence[int]) -> list[int]:
        """Restore every masked position in parallel from unmasked context."""
        restored = [int(t) for t in corrupted]
        for position, token in enumerate(restored):
            if token != self.mask_token:
                continue
            votes = np.zeros(self.vocab_size)
            for other_position, other_token in enumerate(restored):
                if other_position == position:
                    continue
                votes += self.context_counts[position, other_token]
            votes += 0.25
            restored[position] = int(np.argmax(votes))
        return restored

    def generate(self, length: int, steps: int, seed: int = 3) -> list[int]:
        """Iterative refinement: start fully masked, denoise once per step."""
        sequence = [self.mask_token] * length
        rng = np.random.default_rng(seed)
        for _ in range(steps):
            candidate = self.denoise(sequence)
            # re-mask a random subset so refinement remains iterative
            keep = rng.random(length) < 0.5
            sequence = [c if keep[i] else self.mask_token
                        for i, c in enumerate(candidate)]
        return self.denoise(sequence)
