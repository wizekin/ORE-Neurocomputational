"""Export model families to portable graphs (specification Step 41)."""
from __future__ import annotations

from typing import Any

import numpy as np

from ..models.baseline import MovingAverageBaseline, SimpleKalmanBaseline
from ..models.cfc import ClosedFormContinuousNetwork
from ..models.lnn import LiquidNeuralNetwork
from ..models.snn import SpikingNeuralNetwork
from ..models.ssm import SelectiveStateSpaceModel
from ..representations.portable_graph import GraphNode, PortableGraph

__all__ = ("export_model_graph", "export_baseline_graph")


def _augmented(state_weights: np.ndarray, input_weights: np.ndarray,
               bias: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build the augmented [x; u] linear map used by graph exports."""
    augmented = np.concatenate([state_weights, input_weights], axis=1)
    return (augmented, np.asarray(bias, dtype=float))


def export_model_graph(model: Any, graph_id: str | None = None) -> PortableGraph:
    """Export a temporal model to its hardware-independent portable graph."""
    if isinstance(model, ClosedFormContinuousNetwork):
        return _export_cfc(model, graph_id)
    if isinstance(model, LiquidNeuralNetwork):
        return _export_lnn(model, graph_id)
    if isinstance(model, SelectiveStateSpaceModel):
        return _export_ssm(model, graph_id)
    if isinstance(model, SpikingNeuralNetwork):
        return _export_snn(model, graph_id)
    if isinstance(model, (MovingAverageBaseline, SimpleKalmanBaseline)):
        return export_baseline_graph(model, graph_id)
    raise TypeError(f"no portable-graph export is defined for {type(model).__name__}")


def _export_cfc(model: ClosedFormContinuousNetwork, graph_id: str | None) -> PortableGraph:
    """Export the CfC network: augmented linear -> tanh target -> exp_relax -> readout."""
    augmented, bias = _augmented(model.weights_state, model.weights_input, model.bias)
    nodes = (
        GraphNode("cfc_target_linear", "linear", ("__state__", "__control__"),
                  {"weights": augmented, "bias": bias}),
        GraphNode("cfc_target", "tanh", ("cfc_target_linear",), {}),
        GraphNode("cfc_relax", "exp_relax", ("__state__", "cfc_target"),
                  {"tau": model.tau}),
        GraphNode("cfc_readout", "linear", ("cfc_relax",),
                  {"weights": model.readout, "bias": model.readout_bias}),
        GraphNode("prediction", "output", ("cfc_readout",), {}),
    )
    return PortableGraph(
        graph_id=graph_id or f"cfc-{model.identity().config_hash}",
        graph_version="1.0.0",
        nodes=nodes,
        state_nodes=("cfc_relax",),
        event_channels=(),
        input_names=("__state__", "__control__"),
        output_names=("prediction",),
        metadata={"family": "cfc", "tau": model.tau},
    )


def _export_lnn(model: LiquidNeuralNetwork, graph_id: str | None) -> PortableGraph:
    """Export the LNN as a self-contained bounded-Euler liquid cell plus readout."""
    augmented, bias = _augmented(model.weights_state, model.weights_input, model.bias)
    nodes = (
        GraphNode("lnn_integrate", "euler_step", ("__state__", "__control__"),
                  {"tau": model.tau, "max_step": model.max_step,
                   "weights": augmented, "bias": bias}),
        GraphNode("lnn_readout", "linear", ("lnn_integrate",),
                  {"weights": model.readout, "bias": model.readout_bias}),
        GraphNode("prediction", "output", ("lnn_readout",), {}),
    )
    return PortableGraph(
        graph_id=graph_id or f"lnn-{model.identity().config_hash}",
        graph_version="1.0.0",
        nodes=nodes,
        state_nodes=("lnn_integrate",),
        event_channels=(),
        input_names=("__state__", "__control__"),
        output_names=("prediction",),
        metadata={"family": "lnn", "tau": model.tau, "max_step": model.max_step},
    )


def _export_ssm(model: SelectiveStateSpaceModel, graph_id: str | None) -> PortableGraph:
    """Export the selective SSM as a single ssm_cell with declared parameters."""
    nodes = (
        GraphNode("ssm_cell", "ssm_cell", ("__state__", "__control__"),
                  {"input_weights": model.input_weights,
                   "gate_weights": model.gate_weights,
                   "select_weights": model.select_weights,
                   "readout": model.readout,
                   "skip": model.skip,
                   "base_decay": model.base_decay,
                   "state_norm_bound": model.state_norm_bound}),
        GraphNode("prediction", "output", ("ssm_cell",), {}),
    )
    return PortableGraph(
        graph_id=graph_id or f"ssm-{model.identity().config_hash}",
        graph_version="1.0.0",
        nodes=nodes,
        state_nodes=("ssm_cell",),
        event_channels=(),
        input_names=("__state__", "__control__"),
        output_names=("prediction",),
        metadata={"family": "ssm", "base_decay": model.base_decay},
    )


def _export_snn(model: SpikingNeuralNetwork, graph_id: str | None) -> PortableGraph:
    """Export the SNN as a lif_cell with declared membrane parameters."""
    nodes = (
        GraphNode("lif_cell", "lif_cell", ("__state__", "__control__"),
                  {"n_neurons": model.n_neurons,
                   "tau_m": model.tau_m, "v_rest": model.v_rest,
                   "v_threshold": model.v_threshold, "v_reset": model.v_reset,
                   "resistance": model.resistance,
                   "refractory_period": model.refractory_period,
                   "encoding_offsets": model.encoding_offsets}),
        GraphNode("membrane", "output", ("lif_cell",), {}),
    )
    return PortableGraph(
        graph_id=graph_id or f"snn-{model.identity().config_hash}",
        graph_version="1.0.0",
        nodes=nodes,
        state_nodes=("lif_cell",),
        event_channels=("spikes",),
        input_names=("__state__", "__control__"),
        output_names=("membrane",),
        metadata={"family": "snn", "encoding": model.encoding,
                  "n_neurons": model.n_neurons},
    )


def export_baseline_graph(model: Any, graph_id: str | None = None) -> PortableGraph:
    """Export a deterministic baseline as an identity persistence graph."""
    n_channels = int(getattr(model, "n_channels", 1))
    nodes = (
        GraphNode("persistence", "linear", ("__control__",),
                  {"weights": np.eye(n_channels), "bias": np.zeros(n_channels)}),
        GraphNode("prediction", "output", ("persistence",), {}),
    )
    return PortableGraph(
        graph_id=graph_id or f"baseline-{type(model).__name__.lower()}",
        graph_version="1.0.0",
        nodes=nodes,
        state_nodes=(),
        event_channels=(),
        input_names=("__control__",),
        output_names=("prediction",),
        metadata={"family": "baseline", "kind": type(model).__name__},
    )
