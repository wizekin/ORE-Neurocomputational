"""Deterministic CPU reference backend (specification Step 42).

Implements the portable-operation contract with plain numpy kernels.  Behaviour is
deterministic for correctness tests; accelerated backends are added only after this
reference is stable, and simulation remains the default development path.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np

from ..representations.portable_graph import (
    GraphNode,
    PortableGraph,
)

__all__ = ("CPUReferenceBackend", "GraphExecutionResult")


@dataclass(frozen=True, slots=True)
class GraphExecutionResult:
    """The deterministic result of one portable-graph execution."""

    outputs: Mapping[str, Any]
    states: Mapping[str, Any]
    events: Mapping[str, Any]
    executed_nodes: tuple[str, ...]
    backend: str = "cpu_reference"

    def to_dict(self) -> dict[str, Any]:
        """Serialize the execution result."""
        return {
            "outputs": {key: _plain(value) for key, value in self.outputs.items()},
            "states": {key: _plain(value) for key, value in self.states.items()},
            "events": {key: _plain(value) for key, value in self.events.items()},
            "executed_nodes": list(self.executed_nodes),
            "backend": self.backend,
        }


def _plain(value: Any) -> Any:
    """Convert numpy values into plain Python containers."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


class CPUReferenceBackend:
    """Execute portable graphs deterministically on the CPU reference kernels."""

    def __init__(self) -> None:
        """Initialize the kernel table."""
        self._kernels = {
            "input": self._kernel_input,
            "output": self._kernel_output,
            "linear": self._kernel_linear,
            "tanh": self._kernel_tanh,
            "sigmoid": self._kernel_sigmoid,
            "exp_relax": self._kernel_exp_relax,
            "ssm_cell": self._kernel_ssm_cell,
            "lif_cell": self._kernel_lif_cell,
            "euler_step": self._kernel_euler_step,
            "rk4_step": self._kernel_rk4_step,
            "threshold": self._kernel_threshold,
            "gate": self._kernel_gate,
        }

    @property
    def name(self) -> str:
        """Return the backend name."""
        return "cpu_reference"

    def capabilities(self) -> dict[str, Any]:
        """Return the reference backend capability profile."""
        from .capabilities import BackendCapabilities
        return BackendCapabilities(
            backend_name=self.name,
            supported_operations=tuple(sorted(self._kernels)),
            precision_bits=64,
            memory_mb=4096.0,
            event_routing=True,
            runtime_version=f"numpy-{np.__version__}",
            available=True,
            notes="deterministic CPU reference simulator; default development path",
        ).to_dict()

    def execute(self, graph: PortableGraph, inputs: Mapping[str, Any],
                states: Mapping[str, Any] | None = None) -> GraphExecutionResult:
        """Execute the graph in node order, seeding declared inputs and state nodes."""
        graph.validate_supported_operations()
        values: dict[str, Any] = {}
        for name in graph.input_names:
            if name not in inputs:
                raise KeyError(f"missing graph input {name!r}")
            values[name] = np.asarray(inputs[name], dtype=float)
        if "__dt__" in inputs:
            values["__dt__"] = float(inputs["__dt__"])
        for name in graph.state_nodes:
            if states and name in states:
                values[name] = np.asarray(states[name], dtype=float).copy()
        outputs: dict[str, Any] = {}
        events: dict[str, Any] = {}
        new_states: dict[str, Any] = {}
        executed: list[str] = []
        for node in graph.nodes:
            if node.operation == "input":
                executed.append(node.node_id)
                continue
            result = self._kernels[node.operation](node, values)
            values[node.node_id] = result["value"]
            if "state" in result:
                new_states[node.node_id] = result["state"]
            if "event" in result:
                events[node.node_id] = result["event"]
            if node.operation == "output":
                outputs[node.node_id] = result["value"]
            executed.append(node.node_id)
        for name in graph.output_names:
            if name in values:
                outputs[name] = values[name]
        return GraphExecutionResult(outputs=outputs, states=new_states, events=events,
                                    executed_nodes=tuple(executed))

    # ------------------------------------------------------------------ kernels

    def _kernel_input(self, node: GraphNode, values: dict[str, Any]) -> dict[str, Any]:
        """Declare a graph input."""
        return {"value": values[node.inputs[0]]}

    def _kernel_output(self, node: GraphNode, values: dict[str, Any]) -> dict[str, Any]:
        """Mark an output value."""
        return {"value": values[node.inputs[0]]}

    def _kernel_linear(self, node: GraphNode, values: dict[str, Any]) -> dict[str, Any]:
        """Compute y = W [x1; x2; ...] + b with declared parameters."""
        weights = np.asarray(node.parameters["weights"], dtype=float)
        pieces = [np.asarray(values[name], dtype=float).reshape(-1)
                  for name in node.inputs]
        x = pieces[0] if len(pieces) == 1 else np.concatenate(pieces)
        bias = np.asarray(node.parameters.get("bias", np.zeros(weights.shape[0])),
                          dtype=float)
        return {"value": weights @ x + bias}

    def _kernel_tanh(self, node: GraphNode, values: dict[str, Any]) -> dict[str, Any]:
        """Apply the elementwise tanh nonlinearity."""
        return {"value": np.tanh(np.asarray(values[node.inputs[0]], dtype=float))}

    def _kernel_sigmoid(self, node: GraphNode, values: dict[str, Any]) -> dict[str, Any]:
        """Apply the elementwise logistic function."""
        x = np.asarray(values[node.inputs[0]], dtype=float)
        return {"value": 1.0 / (1.0 + np.exp(-x))}

    def _kernel_exp_relax(self, node: GraphNode, values: dict[str, Any]) -> dict[str, Any]:
        """Closed-form relaxation x' = target + (x - target) exp(-dt / tau)."""
        state = np.asarray(values[node.inputs[0]], dtype=float)
        target = np.asarray(values[node.inputs[1]], dtype=float)
        dt = float(values.get("__dt__", 0.0))
        tau = float(node.parameters["tau"])
        if tau <= 0.0:
            raise ValueError("exp_relax requires a positive tau parameter")
        return {"value": target + (state - target) * math.exp(-dt / tau),
                "state": target + (state - target) * math.exp(-dt / tau)}

    def _kernel_euler_step(self, node: GraphNode, values: dict[str, Any]) -> dict[str, Any]:
        """Bounded-step explicit Euler integration of liquid dynamics (Step 18).

        Parameters: tau, max_step, weights (augmented [state; control] map), and bias.
        The tanh drive is recomputed at every substep from the evolving state, matching
        the reference model's vector field dx/dt = (tanh(W [x; u] + b) - x) / tau.
        """
        state = np.asarray(values[node.inputs[0]], dtype=float).copy()
        control = np.asarray(values[node.inputs[1]], dtype=float).reshape(-1)
        dt = float(values.get("__dt__", 0.0))
        tau = float(node.parameters["tau"])
        max_step = float(node.parameters["max_step"])
        weights = np.asarray(node.parameters["weights"], dtype=float)
        bias = np.asarray(node.parameters["bias"], dtype=float)
        if tau <= 0.0 or max_step <= 0.0:
            raise ValueError("euler_step requires positive tau and max_step")
        current = 0.0
        while current < dt:
            step = min(max_step, dt - current)
            drive = np.tanh(weights @ np.concatenate([state, control]) + bias)
            state = state + step * (drive - state) / tau
            current += step
        return {"value": state, "state": state}

    def _kernel_rk4_step(self, node: GraphNode, values: dict[str, Any]) -> dict[str, Any]:
        """One RK4 step of the same relaxation field used by euler_step."""
        state = np.asarray(values[node.inputs[0]], dtype=float)
        target = np.asarray(values[node.inputs[1]], dtype=float)
        dt = float(values.get("__dt__", 0.0))
        tau = float(node.parameters["tau"])
        if tau <= 0.0:
            raise ValueError("rk4_step requires a positive tau parameter")

        def field(x: np.ndarray) -> np.ndarray:
            """Relaxation field toward the target."""
            return (target - x) / tau

        k1 = field(state)
        k2 = field(state + 0.5 * dt * k1)
        k3 = field(state + 0.5 * dt * k2)
        k4 = field(state + dt * k3)
        new_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return {"value": new_state, "state": new_state}

    def _kernel_ssm_cell(self, node: GraphNode, values: dict[str, Any]) -> dict[str, Any]:
        """One selective state-space update with declared parameters."""
        state = np.asarray(values[node.inputs[0]], dtype=float)
        control = np.asarray(values[node.inputs[1]], dtype=float)
        dt = float(values.get("__dt__", 0.0))
        input_weights = np.asarray(node.parameters["input_weights"], dtype=float)
        gate_weights = np.asarray(node.parameters["gate_weights"], dtype=float)
        select_weights = np.asarray(node.parameters["select_weights"], dtype=float)
        readout = np.asarray(node.parameters["readout"], dtype=float)
        skip = np.asarray(node.parameters["skip"], dtype=float)
        base_decay = float(node.parameters["base_decay"])
        state_norm_bound = float(node.parameters.get("state_norm_bound", 1e3))
        decay = np.power(base_decay, 1.0 / (1.0 + np.exp(-(select_weights @ control))))
        gate = 1.0 / (1.0 + np.exp(-(gate_weights @ control)))
        scaled = np.power(decay, min(dt, 1.0)) if dt > 0.0 else np.ones_like(decay)
        new_state = scaled * state + gate * (input_weights @ control)
        norm = float(np.linalg.norm(new_state))
        if norm > state_norm_bound:
            new_state = new_state * (state_norm_bound / norm)
        prediction = readout @ new_state + skip @ control
        return {"value": prediction, "state": new_state}

    def _kernel_lif_cell(self, node: GraphNode, values: dict[str, Any]) -> dict[str, Any]:
        """One LIF advance with declared membrane parameters."""
        from ..functional import lif_advance
        n_neurons = int(node.parameters["n_neurons"])
        state = np.asarray(values[node.inputs[0]], dtype=float).reshape(-1)
        if state.size < 2 * n_neurons:
            state = np.concatenate([state, np.zeros(2 * n_neurons - state.size)])
        voltage = state[:n_neurons].copy()
        refractory = state[n_neurons: 2 * n_neurons].copy()
        control = np.asarray(values[node.inputs[1]], dtype=float).reshape(-1)
        n_channels = max(int(control.size), 1)
        neurons_per_channel = max(n_neurons // n_channels, 1)
        dt = float(values.get("__dt__", 0.0))
        tau_m = float(node.parameters["tau_m"])
        v_rest = float(node.parameters["v_rest"])
        v_threshold = float(node.parameters["v_threshold"])
        v_reset = float(node.parameters["v_reset"])
        resistance = float(node.parameters["resistance"])
        refractory_period = float(node.parameters["refractory_period"])
        offsets = np.asarray(node.parameters.get("encoding_offsets",
                                                 np.zeros(n_neurons)), dtype=float)
        repeated = np.repeat(control, neurons_per_channel)[:n_neurons]
        current = np.abs(repeated) + offsets * 0.5
        new_voltage, spikes, new_refractory = lif_advance(
            voltage, current, dt, tau_m, v_rest, v_threshold, v_reset, resistance,
            refractory_period, refractory)
        combined_state = np.concatenate([new_voltage, new_refractory])
        return {"value": new_voltage, "state": combined_state,
                "event": {"spikes": int(np.count_nonzero(spikes)),
                          "spike_mask": spikes.tolist()}}

    def _kernel_threshold(self, node: GraphNode, values: dict[str, Any]) -> dict[str, Any]:
        """Deterministic threshold comparison."""
        x = np.asarray(values[node.inputs[0]], dtype=float)
        threshold = float(node.parameters["threshold"])
        return {"value": (np.abs(x) > threshold).astype(float)}

    def _kernel_gate(self, node: GraphNode, values: dict[str, Any]) -> dict[str, Any]:
        """Elementwise gating of one signal by another (both in [0, 1])."""
        signal = np.asarray(values[node.inputs[0]], dtype=float)
        gate = np.asarray(values[node.inputs[1]], dtype=float)
        return {"value": signal * np.clip(gate, 0.0, 1.0)}
