"""Operation 6: recurrent distributed IFG/MTG/ATL language-network integration."""
from __future__ import annotations

import hashlib
from typing import Mapping

import numpy as np

from .contracts import BNBEAConfig


REGIONS = ("VWFA", "MTG", "IFG", "ATL", "STG", "AG")

_CONNECTION_WEIGHTS = {
    ("VWFA", "MTG"): 0.42, ("MTG", "IFG"): 0.32, ("MTG", "ATL"): 0.36,
    ("ATL", "MTG"): 0.18, ("IFG", "MTG"): 0.20, ("STG", "MTG"): 0.31,
    ("STG", "IFG"): 0.22, ("AG", "ATL"): 0.24, ("ATL", "AG"): 0.16,
    ("VWFA", "IFG"): 0.10, ("IFG", "VWFA"): 0.06, ("MTG", "VWFA"): 0.08,
}


def feature_hash_vector(features: Mapping[str, float], dimensions: int = 16) -> np.ndarray:
    """Map named semantic/context features into a deterministic signed feature-hash vector."""
    vector = np.zeros(dimensions, dtype=np.float64)
    for key, raw_value in features.items():
        digest = hashlib.sha256(str(key).encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign * float(raw_value)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0.0 else vector


def _relu(values: np.ndarray) -> np.ndarray:
    """Apply a non-negative neuronal-population activation function."""
    return np.maximum(values, 0.0)


class LanguageNetworkProcessor:
    """Propagate orthographic, auditory, semantic and contextual evidence through regional populations."""

    def __init__(self, config: BNBEAConfig | None = None, dimensions: int = 16) -> None:
        """Initialize recurrent-network convergence and a fixed feature-vector dimensionality."""
        self.config = config or BNBEAConfig()
        self.dimensions = int(dimensions)
        if self.dimensions < 4:
            raise ValueError("language network dimensions must be >= 4")

    def process(
        self,
        lexical_candidate: str | None,
        semantic_features: Mapping[str, float],
        context_features: Mapping[str, float],
        auditory_vector: np.ndarray | None = None,
    ) -> dict[str, object]:
        """Run damped recurrent propagation and report each region separately from concept integration."""
        semantic = feature_hash_vector(semantic_features, self.dimensions)
        context = feature_hash_vector(context_features, self.dimensions)
        lexical = feature_hash_vector({lexical_candidate or "unresolved_orthography": 1.0}, self.dimensions)
        auditory = np.zeros(self.dimensions) if auditory_vector is None else np.resize(
            np.asarray(auditory_vector, dtype=np.float64), self.dimensions
        )
        activity = {
            "VWFA": lexical,
            "MTG": 0.35 * lexical + 0.25 * semantic,
            "IFG": 0.25 * context,
            "ATL": semantic.copy(),
            "STG": auditory,
            "AG": 0.5 * semantic + 0.25 * context,
        }
        matrix = np.zeros((len(REGIONS), len(REGIONS)), dtype=np.float64)
        for source_index, source in enumerate(REGIONS):
            for target_index, target in enumerate(REGIONS):
                matrix[target_index, source_index] = _CONNECTION_WEIGHTS.get((source, target), 0.0)
        history = []
        damping = float(self.config.recurrent_damping)
        for _ in range(self.config.recurrent_steps):
            previous = np.stack([activity[name] for name in REGIONS])
            mixed = matrix @ previous
            updated: dict[str, np.ndarray] = {}
            for index, region in enumerate(REGIONS):
                drive = mixed[index]
                if region == "ATL":
                    drive = drive + 0.18 * semantic
                if region == "IFG":
                    drive = drive + 0.18 * context
                updated[region] = _relu((1.0 - damping) * activity[region] + damping * drive)
            activity = updated
            history.append({name: float(np.mean(vector)) for name, vector in activity.items()})
        stacked = np.stack([activity[name] for name in REGIONS])
        centered = stacked - np.mean(stacked, axis=1, keepdims=True)
        norms = np.linalg.norm(centered, axis=1)
        denominator = norms[:, None] * norms[None, :]
        covariance = centered @ centered.T
        connectivity = np.divide(
            covariance, denominator, out=np.zeros_like(covariance), where=denominator > 1e-12
        )
        np.fill_diagonal(connectivity, (norms > 1e-12).astype(np.float64))
        return {
            "regional_activity": {name: vector for name, vector in activity.items()},
            "regional_activity_summary": {name: float(np.linalg.norm(vector)) for name, vector in activity.items()},
            "integrated_concept_vector": activity["ATL"],
            "control_vector": activity["IFG"],
            "functional_connectivity": connectivity,
            "recurrent_history": history,
            "region_order": list(REGIONS),
            "connection_weights": {f"{source}->{target}": weight for (source, target), weight in _CONNECTION_WEIGHTS.items()},
            "biological_analogue": (
                "Distributed recurrent IFG/MTG/ATL language processing with VWFA/STG/AG inputs; "
                "this network is a testable computational abstraction, not a literal connectome."
            ),
        }
