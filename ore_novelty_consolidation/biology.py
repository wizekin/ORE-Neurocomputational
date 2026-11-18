"""Rate-population surrogates for neuromodulatory, hippocampal, and cortical processes.

The simulated populations make circuit-level dependencies executable. They are deliberately
coarse rate models, not neuron-by-neuron digital twins or direct biochemical measurements.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import ConsolidationConfig, InternalState, SleepEvidence
from .functional import clamp, hill_occupancy, sigmoid, temporal_eligibility


class NeuromodulatoryPopulationModel:
    """Represent VTA and locus-coeruleus populations as interacting rate-coded ensembles.

    The model separates VTA-related dopaminergic drive from LC-related noradrenergic drive,
    while allowing LC activity to contribute to hippocampal dopamine as supported by animal
    research. The pathway coefficients are modelling choices.
    """

    def __init__(self, config: ConsolidationConfig) -> None:
        """Initialize deterministic VTA and LC population biases and pathway gains."""
        self.config = config
        self.vta_bias = np.linspace(-0.18, 0.18, config.vta_units)
        self.lc_bias = np.linspace(0.12, -0.12, config.lc_units)
        self.last_vta_activity = np.zeros(config.vta_units, dtype=float)
        self.last_lc_activity = np.zeros(config.lc_units, dtype=float)

    def recruit_pathways(
        self, novelty: Mapping[str, Any], salience: float, state: InternalState,
        reward_value: float,
    ) -> dict[str, Any]:
        """Recruit distinct VTA/LC population responses from novelty and current motivational state."""
        novelty_score = clamp(float(novelty.get("novelty_score", 0.5)))
        novelty_type = str(novelty.get("novelty_type", "undetermined"))
        distinctness = 1.0 if novelty_type == "distinct" else 0.45 if novelty_type == "common" else 0.25
        arousal_efficiency = 1.0 - abs(state.arousal - 0.55) / 0.55
        inputs = np.asarray([
            novelty_score, clamp(salience), distinctness, state.goal_relevance,
            clamp(0.5 * state.motivational_value + 0.5 * reward_value),
            clamp(arousal_efficiency), state.stress,
        ], dtype=float)
        vta_drive = (
            0.78 * inputs[0] + 0.48 * inputs[1] + 0.30 * inputs[2]
            + 0.30 * inputs[3] + 0.38 * inputs[4] + 0.16 * inputs[5]
            - 0.24 * inputs[6]
        )
        lc_drive = (
            0.56 * inputs[0] + 0.72 * inputs[1] + 0.30 * inputs[2]
            + 0.20 * inputs[3] + 0.22 * inputs[5] + 0.18 * inputs[6]
        )
        self.last_vta_activity = 1.0 / (1.0 + np.exp(-np.clip(vta_drive + self.vta_bias, -40, 40)))
        self.last_lc_activity = 1.0 / (1.0 + np.exp(-np.clip(lc_drive + self.lc_bias, -40, 40)))
        return {
            "vta_mean_activity": float(np.mean(self.last_vta_activity)),
            "lc_mean_activity": float(np.mean(self.last_lc_activity)),
            "vta_population_activity": self.last_vta_activity.tolist(),
            "lc_population_activity": self.last_lc_activity.tolist(),
            "novelty_type_hypothesis": novelty_type,
            "pathway_uncertainty": (
                "VTA/LC recruitment is a model estimate; pathway-specific neural activity was not measured."
            ),
        }


class HippocampalPlasticityModel:
    """Model receptor gates, resource availability, and synapse-specific plasticity support."""

    def __init__(self, config: ConsolidationConfig) -> None:
        """Store the kinetic and nonlinear parameters used by hippocampal plasticity estimates."""
        self.config = config
        self.last_resource_pool = 0.0
        self.last_d1_d5_gate = 0.0
        self.last_adrenergic_gate = 0.0

    def estimate_neuromodulators(
        self, pathway: Mapping[str, Any], elapsed_ms: float
    ) -> dict[str, float]:
        """Estimate dopamine and noradrenaline after release and local clearance dynamics."""
        vta = clamp(float(pathway["vta_mean_activity"]))
        lc = clamp(float(pathway["lc_mean_activity"]))
        dopamine_decay = temporal_eligibility(elapsed_ms, self.config.dopamine_tau_ms)
        noradrenaline_decay = temporal_eligibility(elapsed_ms, self.config.noradrenaline_tau_ms)
        dopamine = clamp((0.72 * vta + 0.28 * lc) * dopamine_decay)
        noradrenaline = clamp(0.88 * lc * noradrenaline_decay)
        return {
            "hippocampal_dopamine_estimate": dopamine,
            "hippocampal_noradrenaline_estimate": noradrenaline,
            "dopamine_decay_factor": dopamine_decay,
            "noradrenaline_decay_factor": noradrenaline_decay,
            "lc_dopamine_contribution_estimate": 0.28 * lc * dopamine_decay,
        }

    def calculate_receptor_gates(
        self, dopamine: float, noradrenaline: float, synaptic_activity: Sequence[float]
    ) -> dict[str, float]:
        """Estimate D1/D5 and adrenergic occupancy gated by local synaptic activity."""
        activity = float(np.mean(np.asarray(synaptic_activity, dtype=float)))
        d1_occupancy = hill_occupancy(
            dopamine, self.config.d1_d5_kd, self.config.d1_d5_hill
        )
        adrenergic_occupancy = hill_occupancy(
            noradrenaline, self.config.adrenergic_kd, 1.0
        )
        local_coincidence = 0.30 + 0.70 * clamp(activity)
        d1_gate = clamp(d1_occupancy * local_coincidence)
        adrenergic_gate = clamp(adrenergic_occupancy * local_coincidence)
        self.last_d1_d5_gate = d1_gate
        self.last_adrenergic_gate = adrenergic_gate
        return {
            "d1_d5_occupancy_estimate": d1_occupancy,
            "adrenergic_occupancy_estimate": adrenergic_occupancy,
            "local_activity_coincidence": local_coincidence,
            "d1_d5_plasticity_gate": d1_gate,
            "adrenergic_plasticity_gate": adrenergic_gate,
        }

    def produce_plasticity_resources(
        self, gates: Mapping[str, float], elapsed_ms: float, time_eligibility: float
    ) -> dict[str, float]:
        """Generate a transient PRP-like resource pool using graded intracellular pathway gates."""
        da_gate = clamp(gates["d1_d5_plasticity_gate"])
        adr_gate = clamp(gates["adrenergic_plasticity_gate"])
        combined_gate = 0.65 * da_gate + 0.35 * adr_gate
        production = sigmoid(8.0 * (combined_gate - self.config.resource_threshold))
        persistence = temporal_eligibility(elapsed_ms, self.config.resource_tau_ms)
        amount = clamp(
            self.config.resource_gain * production * persistence * time_eligibility
        )
        self.last_resource_pool = amount
        return {
            "combined_intracellular_gate": combined_gate,
            "production_fraction": production,
            "resource_decay_factor": persistence,
            "plasticity_resource_pool": amount,
        }

    def sleep_cortical_integration(
        self,
        previous_integration: float,
        evidence: SleepEvidence,
        replay_estimate: float,
    ) -> dict[str, Any]:
        """Integrate supplied NREM/awake-rest replay evidence into a separate cortical state."""
        state_gate = 0.0
        if evidence.sleep_state == "nrem":
            state_gate = evidence.state_confidence
        elif evidence.sleep_state == "awake_rest" and self.config.allow_awake_rest_replay:
            state_gate = 0.35 * evidence.state_confidence
        rhythm_coherence = (
            evidence.slow_oscillation_strength
            * evidence.spindle_strength
            * evidence.ripple_strength
        ) ** (1.0 / 3.0)
        coordination = clamp(
            0.55 * evidence.coordination_strength + 0.45 * rhythm_coherence
        )
        effective_replay = clamp(replay_estimate) * state_gate
        sleep_signal = state_gate * (0.55 * coordination + 0.45 * effective_replay)
        drive = clamp(
            sleep_signal * (0.65 + 0.35 * evidence.prior_knowledge_support)
        )
        delta = self.config.cortical_integration_rate * drive * (1.0 - clamp(previous_integration))
        updated = clamp(previous_integration + delta)
        return {
            "sleep_state_gate": state_gate,
            "rhythm_coherence": rhythm_coherence,
            "coordination_estimate": coordination,
            "memory_specific_replay": clamp(replay_estimate),
            "effective_replay": effective_replay,
            "cortical_integration_drive": drive,
            "cortical_integration_before": clamp(previous_integration),
            "cortical_integration_delta": delta,
            "cortical_integration_after": updated,
            "hippocampal_persistence_updated_independently": True,
        }


class BiologicalConsolidationNetwork:
    """Coordinate rate-coded neuromodulatory and hippocampal plasticity model components."""

    def __init__(self, config: ConsolidationConfig) -> None:
        """Construct VTA/LC populations and hippocampal receptor/resource models."""
        self.config = config
        self.neuromodulatory = NeuromodulatoryPopulationModel(config)
        self.hippocampal = HippocampalPlasticityModel(config)
        self.operation_count = 0

    def recruit_and_modulate(
        self, novelty: Mapping[str, Any], salience: float, state: InternalState,
        reward_value: float, elapsed_ms: float,
    ) -> dict[str, Any]:
        """Run the novelty-to-pathway-to-neuromodulator chain for one post-learning event."""
        self.operation_count += 1
        pathway = self.neuromodulatory.recruit_pathways(
            novelty, salience, state, reward_value
        )
        neuromodulators = self.hippocampal.estimate_neuromodulators(pathway, elapsed_ms)
        return {"pathways": pathway, "neuromodulators": neuromodulators}

    def receptor_and_resource_state(
        self, dopamine: float, noradrenaline: float,
        synaptic_activity: Sequence[float], elapsed_ms: float, eligibility: float,
    ) -> dict[str, Any]:
        """Compute receptor-dependent plasticity gates and time-limited resource availability."""
        gates = self.hippocampal.calculate_receptor_gates(
            dopamine, noradrenaline, synaptic_activity
        )
        resources = self.hippocampal.produce_plasticity_resources(
            gates, elapsed_ms, eligibility
        )
        return {"gates": gates, "resources": resources}

    def integrate_sleep(
        self, previous_integration: float, evidence: SleepEvidence, replay_estimate: float
    ) -> dict[str, Any]:
        """Apply parallel sleep-associated cortical integration without forcing hippocampal loss."""
        return self.hippocampal.sleep_cortical_integration(
            previous_integration, evidence, replay_estimate
        )

    def snapshot(self) -> dict[str, Any]:
        """Return current rate-population activity and plasticity model state."""
        return {
            "model_type": "coarse_rate_population_surrogate",
            "vta_units": self.config.vta_units,
            "lc_units": self.config.lc_units,
            "cortical_units": self.config.cortical_units,
            "last_vta_mean_activity": float(np.mean(self.neuromodulatory.last_vta_activity)),
            "last_lc_mean_activity": float(np.mean(self.neuromodulatory.last_lc_activity)),
            "last_d1_d5_gate": self.hippocampal.last_d1_d5_gate,
            "last_adrenergic_gate": self.hippocampal.last_adrenergic_gate,
            "last_resource_pool": self.hippocampal.last_resource_pool,
            "operation_count": self.operation_count,
            "interpretation": (
                "Population rates and pathway values are illustrative estimates, not direct neural measurements."
            ),
        }
