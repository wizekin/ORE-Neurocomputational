"""Operation 4: hierarchical visual populations with attention and recurrent contextual feedback."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .contracts import BNBEAConfig


def _relu(values: np.ndarray) -> np.ndarray:
    """Apply a rectified-linear population response to feature arrays."""
    return np.maximum(np.asarray(values, dtype=np.float64), 0.0)


def _normalize_channels(values: np.ndarray) -> np.ndarray:
    """Normalize each feature channel by its spatial energy to stabilize later hierarchy levels."""
    denominator = np.sqrt(np.mean(values**2, axis=(1, 2), keepdims=True) + 1e-8)
    return values / denominator


class VisualHierarchyProcessor:
    """Build Retina/LGN-to-ventral/dorsal visual representations with simple feedback dynamics."""

    def __init__(self, config: BNBEAConfig | None = None) -> None:
        """Store recurrent depth and damping for the hierarchical visual reference."""
        self.config = config or BNBEAConfig()

    def process(self, feature_stack: np.ndarray, attention_map: np.ndarray | None = None) -> dict[str, object]:
        """Transform Gabor maps into increasingly pooled feature combinations and spatial attention."""
        stack = np.asarray(feature_stack, dtype=np.float64)
        if stack.ndim != 3:
            raise ValueError("feature_stack must have shape [channels, height, width]")
        retina = np.mean(stack, axis=0, keepdims=True)
        lgn = _normalize_channels(_relu(stack - np.mean(stack, axis=(1, 2), keepdims=True)))
        v1 = _normalize_channels(_relu(lgn))
        v2 = _normalize_channels(_relu(ndi.gaussian_filter(v1, sigma=(0, 1.0, 1.0))))
        pairwise = np.concatenate((v2, np.roll(v2, 1, axis=0), np.roll(v2, -1, axis=0)), axis=0)
        v4 = _normalize_channels(_relu(pairwise))
        ventral = np.mean(v4, axis=0, keepdims=True)
        dorsal = np.stack((np.mean(v1, axis=0), np.gradient(np.mean(v1, axis=0), axis=1),
                           np.gradient(np.mean(v1, axis=0), axis=0)), axis=0)
        salience = np.mean(v4**2, axis=0)
        if attention_map is None:
            attention_logits = salience - float(np.max(salience))
        else:
            attention_logits = np.asarray(attention_map, dtype=np.float64)
            if attention_logits.shape != salience.shape:
                raise ValueError("attention_map shape must match visual spatial dimensions")
        weights = np.exp(np.clip(attention_logits, -50.0, 0.0))
        weights /= max(float(np.sum(weights)), 1e-12)
        feedback = float(self.config.recurrent_damping)
        recurrent_v1 = v1.copy()
        for _ in range(self.config.recurrent_steps):
            top_down = ndi.gaussian_filter(np.mean(v4, axis=0), sigma=2.0)[None, ...]
            recurrent_v1 = _relu((1.0 - feedback) * recurrent_v1 + feedback * (v1 + 0.1 * top_down))
        pooled_attention = np.sum(recurrent_v1 * weights[None, ...], axis=(1, 2))
        return {
            "layers": {
                "retina_lgn": retina,
                "lgn": lgn,
                "v1": v1,
                "v2": v2,
                "v4": v4,
                "ventral_temporal": ventral,
                "dorsal_spatial": dorsal,
                "v1_recurrent": recurrent_v1,
            },
            "attention_weights": weights,
            "attention_weighted_features": pooled_attention,
            "retinotopic_shape": tuple(int(x) for x in v1.shape[1:]),
            "position_information_preserved": True,
            "biological_analogue": (
                "Hierarchical receptive-field complexity, spatial attention and feedback across "
                "visual pathways; layer transforms are abstractions, not an anatomical wiring diagram."
            ),
        }
