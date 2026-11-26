"""Algorithmic GNN: encode-process-decode with stepwise and final supervision (steps 43-44).

The model imitates the *state transitions* of trusted reference algorithms on narrow
tasks. It is isolated from production USG writes by construction: nothing in this
module touches the USG, the workspace, or the promotion gate - it only predicts
advisory outputs that deterministic algorithms must check (steps 46 and 48).

Architecture (CLRS-style): node/edge encoders, a shared-weight message-passing
processor run for ``rounds`` rounds, and a task-appropriate decoder (per-node for
stepwise states and sorting ranks, per-edge for MST membership). Backpropagation is
exact and implemented on numpy; training reuses the shared AdamOptimizer.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Mapping, Sequence

import numpy as np

from ..embeddings.trainer import AdamOptimizer
from ..schemas.model_manifest import build_weights_hash
from .task_generators import NAR_REFERENCE_VERSION, NarrowTask, reference_solve
from .trace_dataset import AlgorithmTrace, NODE_FEATURE_DIM, build_trace

_EPS = 1e-12
_ROLLOUT_EPS = 1e-4


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _relu(values: np.ndarray) -> np.ndarray:
    return np.maximum(values, 0.0)


@dataclass(frozen=True, slots=True)
class NARModelConfig:
    """Reproducible configuration for one algorithmic GNN (manifest fields, step 24)."""

    task_type: str
    latent_dim: int = 16
    rounds: int = 3
    supervision: str = "stepwise"
    target_kind: str = "node"
    seed: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "task_type": self.task_type, "latent_dim": self.latent_dim,
            "rounds": self.rounds, "supervision": self.supervision,
            "target_kind": self.target_kind, "seed": self.seed,
        }


@dataclass
class NARTrainingReport:
    """Training provenance: config, dataset size, loss curve, timing, weights hash."""

    config: NARModelConfig
    epochs: int
    final_loss: float
    loss_curve: tuple[float, ...]
    n_traces: int
    training_seconds: float
    weights_hash: str
    implementation_version: str = NAR_REFERENCE_VERSION

    def as_dict(self) -> dict[str, object]:
        return {
            "config": self.config.as_dict(), "epochs": self.epochs,
            "final_loss": self.final_loss, "loss_curve": list(self.loss_curve),
            "n_traces": self.n_traces, "training_seconds": self.training_seconds,
            "weights_hash": self.weights_hash,
            "implementation_version": self.implementation_version,
        }


class _Network:
    """Parameter container plus exact forward/backward for the algorithmic GNN."""

    def __init__(self, config: NARModelConfig) -> None:
        self.config = config
        rng = np.random.default_rng(config.seed)
        latent = config.latent_dim
        node_in = NODE_FEATURE_DIM + 1  # static features plus the current state
        limit = np.sqrt(6.0 / (node_in + latent))
        self.params: dict[str, np.ndarray] = {
            "node_encoder_w": rng.uniform(-limit, limit, (node_in, latent)),
            "node_encoder_b": np.zeros(latent),
            "edge_encoder_w": rng.normal(0.0, 0.1, (3, latent)),
            "edge_encoder_b": np.zeros(latent),
            "message_w": rng.normal(0.0, 0.1, (2 * latent, latent)),
            "message_b": np.zeros(latent),
            "update_w": rng.normal(0.0, 0.1, (latent, latent)),
            "update_b": np.zeros(latent),
        }
        out_limit = np.sqrt(6.0 / (2 * latent + 1))
        if config.target_kind == "node":
            self.params["node_decoder_w"] = rng.uniform(-out_limit, out_limit, (latent, 1))
            self.params["node_decoder_b"] = np.zeros(1)
        else:
            self.params["edge_decoder_w"] = rng.uniform(-out_limit, out_limit, (2 * latent, 1))
            self.params["edge_decoder_b"] = np.zeros(1)

    # ------------------------------------------------------------------ forward

    def forward(
        self,
        node_features: np.ndarray,
        edge_features: np.ndarray,
        state: np.ndarray,
        sources: np.ndarray,
        targets: np.ndarray,
        *,
        n_nodes: int,
    ) -> tuple[np.ndarray, list[dict[str, np.ndarray]]]:
        """Run encode-process-decode; returns raw predictions and backprop caches."""
        x_in = np.concatenate([node_features, state[:, None]], axis=1)
        z_pre = x_in @ self.params["node_encoder_w"] + self.params["node_encoder_b"]
        h = _relu(z_pre)
        e = _relu(
            edge_features @ self.params["edge_encoder_w"] + self.params["edge_encoder_b"]
        )
        caches: list[dict[str, np.ndarray]] = []
        for _ in range(self.config.rounds):
            message_cat = np.concatenate([h[sources], e], axis=1)
            m_pre = message_cat @ self.params["message_w"] + self.params["message_b"]
            m = _relu(m_pre)
            agg = np.zeros((n_nodes, h.shape[1]), dtype=np.float64)
            np.add.at(agg, targets, m)
            h_pre = h @ self.params["update_w"] + self.params["update_b"] + agg
            h_out = _relu(h_pre)
            caches.append({
                "h_in": h, "e": e, "message_cat": message_cat, "m_pre": m_pre,
                "agg": agg, "h_pre": h_pre, "h_out": h_out, "z_pre": z_pre,
            })
            h = h_out
        if self.config.target_kind == "node":
            y = (h @ self.params["node_decoder_w"] + self.params["node_decoder_b"])[:, 0]
        else:
            pairs = np.concatenate([h[sources], h[targets]], axis=1)
            y = (pairs @ self.params["edge_decoder_w"] + self.params["edge_decoder_b"])[:, 0]
            caches[-1]["h_final"] = h
        return y, caches

    # ------------------------------------------------------------------ backward

    def backward(
        self,
        caches: list[dict[str, np.ndarray]],
        grad_y: np.ndarray,
        x_in: np.ndarray,
        edge_features: np.ndarray,
        sources: np.ndarray,
        targets: np.ndarray,
        *,
        n_nodes: int,
    ) -> dict[str, np.ndarray]:
        """Exact gradients of the scalar loss w.r.t. every parameter."""
        grads = {name: np.zeros_like(value) for name, value in self.params.items()}
        latent = self.config.latent_dim
        if self.config.target_kind == "node":
            grad_col = grad_y[:, None]
            grads["node_decoder_w"] += caches[-1]["h_out"].T @ grad_col
            grads["node_decoder_b"] += grad_col.sum(axis=0)
            dh = grad_col @ self.params["node_decoder_w"].T
        else:
            h_final = caches[-1]["h_final"]
            pairs = np.concatenate([h_final[sources], h_final[targets]], axis=1)
            grads["edge_decoder_w"] += pairs.T @ grad_y[:, None]
            grads["edge_decoder_b"] += grad_y.sum()
            pair_grad = grad_y[:, None] @ self.params["edge_decoder_w"].T
            dh = np.zeros((n_nodes, latent), dtype=np.float64)
            np.add.at(dh, sources, pair_grad[:, :latent])
            np.add.at(dh, targets, pair_grad[:, latent:])
        for position, cache in enumerate(reversed(caches)):
            d_pre = dh * (cache["h_pre"] > 0)
            grads["update_w"] += cache["h_in"].T @ d_pre
            grads["update_b"] += d_pre.sum(axis=0)
            dh_in = d_pre @ self.params["update_w"].T
            d_m = d_pre[targets]
            d_m_pre = d_m * (cache["m_pre"] > 0)
            grads["message_w"] += cache["message_cat"].T @ d_m_pre
            grads["message_b"] += d_m_pre.sum(axis=0)
            d_cat = d_m_pre @ self.params["message_w"].T
            d_e = d_cat[:, latent:]
            np.add.at(dh_in, sources, d_cat[:, :latent])
            grads["edge_encoder_w"] += edge_features.T @ d_e
            grads["edge_encoder_b"] += d_e.sum(axis=0)
            if position == len(caches) - 1:
                # Only round 0 consumes the encoder output as its input.
                d_z = dh_in * (cache["z_pre"] > 0)
                grads["node_encoder_w"] += x_in.T @ d_z
                grads["node_encoder_b"] += d_z.sum(axis=0)
            dh = dh_in
        return grads


def _pair_loss(
    y: np.ndarray,
    target: np.ndarray,
    *,
    binary: bool,
) -> tuple[float, np.ndarray]:
    """Return the scalar loss and dL/dy for one prediction/target pair."""
    if binary:
        p = _sigmoid(y)
        loss = -float(np.mean(
            target * np.log(p + _EPS) + (1.0 - target) * np.log(1.0 - p + _EPS),
        ))
        grad_y = (p - target) / y.size
        return loss, grad_y
    residual = y - target
    loss = float(np.mean(residual ** 2))
    grad_y = 2.0 * residual / y.size
    return loss, grad_y


def _accumulate(dst: dict[str, np.ndarray], src: Mapping[str, np.ndarray]) -> None:
    for name, value in src.items():
        dst[name] = dst[name] + value


class NARTrainer:
    """Train an algorithmic GNN on reference traces for one narrow task type (step 44)."""

    def __init__(
        self,
        *,
        latent_dim: int = 16,
        rounds: int = 3,
        epochs: int = 80,
        learning_rate: float = 0.05,
        seed: int = 0,
    ) -> None:
        if latent_dim < 4 or rounds < 1 or epochs < 1:
            raise ValueError("latent_dim >= 4, rounds >= 1 and epochs >= 1 are required")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        self.latent_dim = int(latent_dim)
        self.rounds = int(rounds)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.seed = int(seed)

    def train(self, traces: Sequence[AlgorithmTrace]) -> TrainedNARModel:
        """Fit one model to traces of a single task type; seeds and config are recorded."""
        if not traces:
            raise ValueError("training requires at least one trace")
        task_type = traces[0].task.task_type
        for trace in traces:
            if trace.task.task_type != task_type:
                raise ValueError("all traces must share one task type")
        supervision = traces[0].supervision
        target_kind = traces[0].target_kind
        binary = task_type in {"graph_traversal_bfs", "minimum_spanning_tree"}
        config = NARModelConfig(
            task_type=task_type, latent_dim=self.latent_dim, rounds=self.rounds,
            supervision=supervision, target_kind=target_kind, seed=self.seed,
        )
        network = _Network(config)
        optimizer = AdamOptimizer(self.learning_rate)
        rng = np.random.default_rng(self.seed)
        order = list(range(len(traces)))
        loss_curve: list[float] = []
        started = perf_counter()
        for _epoch in range(self.epochs):
            rng.shuffle(order)
            epoch_loss = 0.0
            for index in order:
                trace = traces[index]
                loss, grads = self._trace_loss_and_grads(network, trace, binary=binary)
                optimizer.step(network.params, grads)
                epoch_loss += loss
            loss_curve.append(epoch_loss / len(traces))
        report = NARTrainingReport(
            config=config, epochs=self.epochs,
            final_loss=loss_curve[-1] if loss_curve else float("nan"),
            loss_curve=tuple(loss_curve), n_traces=len(traces),
            training_seconds=perf_counter() - started,
            weights_hash=build_weights_hash(network.params),
        )
        return TrainedNARModel(
            network=network, config=config, report=report,
            trained_sizes=sorted({t.task.n_nodes for t in traces}),
            trained_topologies=sorted({t.task.topology for t in traces}),
            reference_version=traces[0].implementation_version,
        )

    @staticmethod
    def _trace_loss_and_grads(
        network: _Network,
        trace: AlgorithmTrace,
        *,
        binary: bool,
    ) -> tuple[float, dict[str, np.ndarray]]:
        n_nodes = trace.task.n_nodes
        sources, targets = trace.edge_sources, trace.edge_targets
        x_in = np.concatenate([trace.node_features, trace.initial_state[:, None]], axis=1)
        total = 0.0
        grads: dict[str, np.ndarray] | None = None
        if trace.supervision == "stepwise":
            states = trace.step_states
            pairs = len(states) - 1
            for index in range(pairs):
                y, caches = network.forward(
                    trace.node_features, trace.edge_features, states[index],
                    sources, targets, n_nodes=n_nodes,
                )
                loss, grad_y = _pair_loss(y, states[index + 1], binary=binary)
                total += loss
                step_grads = network.backward(
                    caches, grad_y, x_in, trace.edge_features, sources, targets,
                    n_nodes=n_nodes,
                )
                if grads is None:
                    grads = step_grads
                else:
                    _accumulate(grads, step_grads)
            return total / max(1, pairs), grads or {}
        y, caches = network.forward(
            trace.node_features, trace.edge_features, trace.initial_state,
            sources, targets, n_nodes=n_nodes,
        )
        target = (
            trace.final_node_target if network.config.target_kind == "node"
            else trace.final_edge_target
        )
        loss, grad_y = _pair_loss(y, target, binary=binary)
        grads = network.backward(
            caches, grad_y, x_in, trace.edge_features, sources, targets,
            n_nodes=n_nodes,
        )
        return loss, grads


@dataclass
class TrainedNARModel:
    """A trained algorithmic GNN plus its provenance and rollout machinery."""

    network: _Network
    config: NARModelConfig
    report: NARTrainingReport
    trained_sizes: list[int]
    trained_topologies: list[str]
    reference_version: str

    # ------------------------------------------------------------------ prediction

    def predict_step(self, trace: AlgorithmTrace, state: np.ndarray) -> np.ndarray:
        """Predict the next intermediate state from the current state (stepwise tasks)."""
        y, _ = self.network.forward(
            trace.node_features, trace.edge_features, state,
            trace.edge_sources, trace.edge_targets, n_nodes=trace.task.n_nodes,
        )
        if self.config.task_type == "graph_traversal_bfs":
            return (_sigmoid(y) > 0.5).astype(np.float64)
        return y

    def predict_final(self, trace: AlgorithmTrace) -> np.ndarray:
        """Predict final per-node or per-edge outputs from the initial state (final tasks)."""
        y, _ = self.network.forward(
            trace.node_features, trace.edge_features, trace.initial_state,
            trace.edge_sources, trace.edge_targets, n_nodes=trace.task.n_nodes,
        )
        if self.config.task_type == "minimum_spanning_tree":
            return (_sigmoid(y) > 0.5).astype(np.float64)
        return y

    def rollout(self, trace: AlgorithmTrace) -> NARPrediction:
        """Iterate the learned transition until fixpoint or the step bound."""
        if trace.supervision == "final":
            return NARPrediction(
                task_id=trace.task.task_id,
                predicted_states=(),
                final_outputs=self.predict_final(trace),
                termination_step=0,
                steps_run=1,
                binary_outputs=self.config.task_type == "minimum_spanning_tree",
            )
        state = trace.initial_state.copy()
        states = [state.copy()]
        termination = 0
        for step in range(trace.task.n_nodes):
            next_state = self.predict_step(trace, state)
            states.append(next_state)
            termination = step + 1
            if float(np.max(np.abs(next_state - state))) < _ROLLOUT_EPS:
                break
            state = next_state
        return NARPrediction(
            task_id=trace.task.task_id,
            predicted_states=tuple(states),
            final_outputs=states[-1],
            termination_step=termination,
            steps_run=termination,
            binary_outputs=self.config.task_type == "graph_traversal_bfs",
        )

    def weights_hash(self) -> str:
        return self.report.weights_hash

    def manifest(self) -> dict[str, object]:
        """Model artifact manifest: version, weights hash, config, limitations (step 24)."""
        return {
            "task_type": self.config.task_type,
            "config": self.config.as_dict(),
            "weights_hash": self.report.weights_hash,
            "trained_sizes": list(self.trained_sizes),
            "trained_topologies": list(self.trained_topologies),
            "reference_version": self.reference_version,
            "limitations": [
                "advisory only; deterministic algorithms validate correctness-critical outputs",
                "OOD sizes and unseen topologies require abstention or fallback",
            ],
        }


@dataclass(frozen=True, slots=True)
class NARPrediction:
    """Raw model rollout outputs; never treated as truth."""

    task_id: str
    predicted_states: tuple[np.ndarray, ...]
    final_outputs: np.ndarray
    termination_step: int
    steps_run: int
    binary_outputs: bool


@dataclass(frozen=True, slots=True)
class AlgorithmicProposal:
    """The Point 8 NAR contract result: advisory output with a deterministic check."""

    task_id: str
    task_type: str
    advisory: bool
    disposition: str  # advisory | fallback_to_reference | abstained
    ood_flags: tuple[str, ...]
    predicted_outputs: np.ndarray
    reference_outputs: np.ndarray
    exact_agreement: float
    model_version: str
    reference_version: str
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id, "task_type": self.task_type,
            "advisory": self.advisory, "disposition": self.disposition,
            "ood_flags": list(self.ood_flags), "exact_agreement": self.exact_agreement,
            "model_version": self.model_version,
            "reference_version": self.reference_version, "notes": list(self.notes),
        }


class NeuralAlgorithmicService:
    """Point 8 contract: ``predict(task) -> AlgorithmicProposal`` - advisory only.

    Every proposal runs the trusted deterministic reference on the same task and
    compares the learned rollout against it (step 46). Learned results stay advisory;
    exact algorithms remain the correctness authority, and OOD inputs fall back or
    abstain according to the gate policy (step 48).
    """

    def __init__(
        self,
        models: Mapping[str, TrainedNARModel],
        *,
        agreement_threshold: float = 0.9,
    ) -> None:
        if agreement_threshold <= 0 or agreement_threshold > 1:
            raise ValueError("agreement_threshold must be in (0, 1]")
        self.models = dict(models)
        self.agreement_threshold = float(agreement_threshold)

    def supports(self, task_type: str) -> bool:
        return task_type in self.models

    def predict(self, task: NarrowTask) -> AlgorithmicProposal:
        """Predict for one task; deterministic check and OOD policy included."""
        model = self.models.get(task.task_type)
        if model is None:
            return AlgorithmicProposal(
                task_id=task.task_id, task_type=task.task_type,
                advisory=False, disposition="abstained",
                ood_flags=("no_trained_model",),
                predicted_outputs=np.zeros(0, dtype=np.float64),
                reference_outputs=reference_solve(task).final,
                exact_agreement=0.0, model_version="none",
                reference_version=task.implementation_version,
                notes=("no trained model for this task type; reference only",),
            )
        trace = build_trace(task)
        ood_flags: list[str] = []
        if task.n_nodes > max(model.trained_sizes):
            ood_flags.append("size_extrapolation")
        if task.topology not in model.trained_topologies:
            ood_flags.append("unseen_topology")
        prediction = model.rollout(trace)
        reference_final = _reference_target(trace)
        agreement = _agreement(
            prediction.final_outputs, reference_final,
            binary=prediction.binary_outputs or trace.target_kind == "edge",
        )
        notes: list[str] = []
        if ood_flags:
            disposition = "fallback_to_reference"
            advisory = False
            notes.append("OOD input: deterministic reference output is authoritative")
        elif agreement >= self.agreement_threshold:
            disposition = "advisory"
            advisory = True
            notes.append("learned rollout agrees with the deterministic oracle")
        else:
            disposition = "fallback_to_reference"
            advisory = False
            notes.append("learned rollout disagrees with the deterministic oracle; reference wins")
        return AlgorithmicProposal(
            task_id=task.task_id, task_type=task.task_type,
            advisory=advisory, disposition=disposition,
            ood_flags=tuple(ood_flags),
            predicted_outputs=prediction.final_outputs,
            reference_outputs=reference_final,
            exact_agreement=agreement,
            model_version=model.report.weights_hash[:16],
            reference_version=task.implementation_version,
            notes=tuple(notes),
        )


def _reference_target(trace: AlgorithmTrace) -> np.ndarray:
    """Reference output in exactly the tensor space the model predicts in."""
    if trace.supervision == "stepwise":
        return trace.step_states[-1]
    if trace.target_kind == "edge":
        return trace.final_edge_target
    return trace.final_node_target


def _agreement(predicted: np.ndarray, reference: np.ndarray, *, binary: bool) -> float:
    """Fraction agreement between learned and deterministic outputs."""
    if predicted.shape != reference.shape:
        return 0.0
    if binary:
        return float(np.mean((predicted > 0.5) == (reference > 0.5)))
    return float(np.mean(np.abs(predicted - reference) < 0.05))


__all__ = [
    "AlgorithmicProposal", "NARModelConfig", "NARPrediction", "NARTrainer",
    "NARTrainingReport", "NeuralAlgorithmicService", "TrainedNARModel",
]
