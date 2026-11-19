"""Provenance envelope and deterministic content hashing for exchanged ORE artifacts."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from numbers import Real
from typing import Any, Mapping

import numpy as np

from . import ContractEnvelope


def _canonical_payload(value: Any) -> Any:
    """Convert supported payload values into a deterministic, type-tagged JSON structure."""
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            return {"__ndarray_object__": _canonical_payload(value.tolist()), "shape": list(value.shape)}
        if np.issubdtype(value.dtype, np.number) and not bool(np.isfinite(value).all()):
            raise ValueError("artifact arrays cannot contain NaN or infinity")
        contiguous = np.ascontiguousarray(value)
        return {
            "__ndarray__": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
            "dtype": contiguous.dtype.str,
            "shape": list(contiguous.shape),
        }
    if isinstance(value, np.generic):
        return _canonical_payload(value.item())
    if isinstance(value, Real) and not isinstance(value, bool):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("artifact payload cannot contain NaN or infinity")
        return number
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _canonical_payload(value.to_dict())
    if hasattr(value, "__dataclass_fields__"):
        from dataclasses import asdict
        return _canonical_payload(asdict(value))
    if isinstance(value, Mapping):
        if any(not isinstance(key, (str, int, float, bool)) for key in value):
            raise TypeError("artifact mappings must use scalar JSON-compatible keys")
        return {str(key): _canonical_payload(value[key]) for key in sorted(value, key=lambda item: str(item))}
    if isinstance(value, (tuple, list)):
        return [_canonical_payload(item) for item in value]
    raise TypeError(f"unsupported artifact payload type: {type(value).__module__}.{type(value).__qualname__}")


def content_hash(value: Any) -> str:
    """Compute a stable SHA-256 digest, preserving NumPy array dtype, shape, and bytes."""
    encoded = json.dumps(
        _canonical_payload(value), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class ContractArtifact:
    """Pair a typed payload with the existing ORE ContractEnvelope and immutable metadata."""

    envelope: ContractEnvelope
    payload: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Require an existing contract envelope and immediately validate deterministic hashability."""
        if not isinstance(self.envelope, ContractEnvelope):
            raise TypeError("envelope must be an existing ore_contracts.ContractEnvelope")
        content_hash(self.payload)

    @property
    def payload_hash(self) -> str:
        """Return the SHA-256 content fingerprint of the payload."""
        return content_hash(self.payload)

    def to_dict(self, include_payload: bool = True) -> dict[str, Any]:
        """Serialize envelope metadata and payload while keeping array fingerprints explicit."""
        data: dict[str, Any] = {
            "envelope": self.envelope.to_dict(),
            "payload_hash": self.payload_hash,
            "metadata": dict(self.metadata),
        }
        if include_payload:
            data["payload"] = _canonical_payload(self.payload)
        return data
