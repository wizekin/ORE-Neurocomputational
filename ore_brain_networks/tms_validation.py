"""Operation 11: virtual TMS perturbation contrasts for computational causal-model validation."""
from __future__ import annotations

from typing import Mapping

import numpy as np


def gaussian_pulse(
    time_ms: np.ndarray, center_ms: float, width_ms: float, amplitude: float = 1.0
) -> np.ndarray:
    """Construct a unitless Gaussian model perturbation envelope in the requested simulation time axis."""
    if width_ms <= 0.0:
        raise ValueError("virtual pulse width_ms must be positive")
    times = np.asarray(time_ms, dtype=np.float64)
    return float(amplitude) * np.exp(-0.5 * ((times - center_ms) / width_ms) ** 2)


def simulate_virtual_tms(
    source_activity: np.ndarray,
    source_names: tuple[str, ...],
    target: str,
    control_target: str | None = None,
    center_ms: float | None = None,
    width_ms: float = 8.0,
    strength: float = 0.25,
    mode: str = "attenuate",
    sample_interval_ms: float = 1.0,
) -> dict[str, object]:
    """Apply a target-specific virtual pulse and construct active, sham and control model conditions.

    strength is a dimensionless simulation control. It is not a physical TMS dose, coil setting,
    clinical recommendation or parameter for use with a human participant.
    """
    activity = np.asarray(source_activity, dtype=np.float64)
    if activity.ndim != 2 or activity.shape[1] != len(source_names):
        raise ValueError("source_activity columns must match source_names")
    if target not in source_names:
        raise KeyError(f"target {target!r} is not a named neural population")
    if control_target is not None and control_target not in source_names:
        raise KeyError(f"control target {control_target!r} is not a named neural population")
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must lie in [0, 1]")
    if mode not in {"attenuate", "excite"}:
        raise ValueError("mode must be 'attenuate' or 'excite'")
    if sample_interval_ms <= 0.0:
        raise ValueError("sample_interval_ms must be positive")
    times = np.arange(activity.shape[0], dtype=np.float64) * float(sample_interval_ms)
    center = float(center_ms if center_ms is not None else (times[-1] / 2.0 if times.size else 0.0))
    pulse = gaussian_pulse(times, center, width_ms, 1.0)
    target_index = source_names.index(target)
    active = activity.copy()
    if mode == "attenuate":
        active[:, target_index] *= 1.0 - strength * pulse
    else:
        active[:, target_index] += strength * pulse
    control = activity.copy()
    if control_target is not None:
        control_index = source_names.index(control_target)
        if mode == "attenuate":
            control[:, control_index] *= 1.0 - strength * pulse
        else:
            control[:, control_index] += strength * pulse
    sham = activity.copy()
    baseline_level = float(np.mean(np.abs(activity))) if activity.size else 0.0
    active_level = float(np.mean(np.abs(active))) if active.size else 0.0
    sham_level = float(np.mean(np.abs(sham))) if sham.size else 0.0
    control_level = float(np.mean(np.abs(control))) if control.size else 0.0
    return {
        "baseline": activity,
        "active_target": active,
        "sham": sham,
        "active_control": control,
        "pulse_envelope": pulse,
        "target": target,
        "control_target": control_target,
        "mode": mode,
        "strength_unitless": strength,
        "active_minus_sham_activity_effect": active_level - sham_level,
        "active_minus_control_activity_effect": active_level - control_level,
        "baseline_activity_proxy": baseline_level,
    }


class TMSValidationModel:
    """Run virtual target/sham/control conditions and compare supplied observed causal effects."""

    def process(
        self,
        source_activity: np.ndarray,
        source_names: tuple[str, ...],
        experiment_data: Mapping[str, object],
    ) -> dict[str, object]:
        """Simulate explicitly requested model perturbations without specifying human stimulation settings."""
        conditions = experiment_data.get("tms_conditions")
        if not conditions:
            return {"status": "awaiting_virtual_TMS_conditions_or_experimental_data"}
        outputs = []
        for condition in conditions:
            outputs.append(simulate_virtual_tms(
                source_activity=source_activity,
                source_names=source_names,
                target=str(condition["target"]),
                control_target=str(condition["control_target"]) if condition.get("control_target") else None,
                center_ms=float(condition["center_ms"]) if condition.get("center_ms") is not None else None,
                width_ms=float(condition.get("width_ms", 8.0)),
                strength=float(condition.get("strength", 0.25)),
                mode=str(condition.get("mode", "attenuate")),
                sample_interval_ms=float(condition.get("sample_interval_ms", 1.0)),
            ))
        observed = experiment_data.get("tms_observed_effects")
        model_effects = np.asarray([item["active_minus_sham_activity_effect"] for item in outputs])
        discrepancy = None
        if observed is not None:
            observed_array = np.asarray(observed, dtype=np.float64).reshape(-1)
            if observed_array.shape == model_effects.shape:
                discrepancy = float(np.mean((observed_array - model_effects) ** 2))
        return {
            "status": "virtual_perturbations_simulated",
            "conditions": outputs,
            "model_effects": model_effects,
            "observed_effect_mse": discrepancy,
            "safety_scope": (
                "Simulation only. Does not prescribe stimulation targets, pulse timing, intensity, "
                "frequency or clinical/research protocols for humans."
            ),
        }
