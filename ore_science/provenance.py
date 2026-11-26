"""Scientific provenance manifests and reproducibility records (Step 50).

Every run records dataset identity and version, preprocessing, model weights and
versions, code revision, solver settings, random seeds, hardware and runtime,
equation versions, input and output hashes, and validation outcomes. Secrets and
sensitive raw inputs are redacted. Repeated execution with fixed deterministic
settings must reproduce the same output hashes; the recorder exposes a
``reproducibility_check`` that compares manifests exactly.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = ["ProvenanceRecorder", "ProvenanceRef", "RunManifest", "redact_secrets"]


_REDACT_KEYS = ("api_key", "token", "secret", "password", "credential", "license_key")


def redact_secrets(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of the payload with secret-looking keys redacted."""
    cleaned: dict[str, Any] = {}
    for key, value in payload.items():
        lowered = str(key).lower()
        if any(marker in lowered for marker in _REDACT_KEYS):
            cleaned[str(key)] = "[REDACTED]"
        elif isinstance(value, Mapping):
            cleaned[str(key)] = redact_secrets(value)
        else:
            cleaned[str(key)] = value
    return cleaned


def _code_revision() -> str:
    """Best-effort git revision of the repository, or 'unavailable'."""
    try:
        output = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return output.stdout.strip()
    except (subprocess.SubprocessError, OSError, FileNotFoundError):
        return "unavailable"


def _stable_hash(value: Any) -> str:
    """Deterministic SHA-256 over a JSON-serialized value."""
    serialized = json.dumps(value, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ProvenanceRef:
    """Reference to a recorded manifest: identity, digest, and timestamp."""

    provenance_id: str
    digest: str
    recorded_at: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize the provenance reference."""
        return {
            "provenance_id": self.provenance_id,
            "digest": self.digest,
            "recorded_at": self.recorded_at,
        }


@dataclass(frozen=True, slots=True)
class RunManifest:
    """Complete scientific provenance for one execution (Step 50)."""

    task_id: str
    question: str
    datasets: tuple[Mapping[str, Any], ...] = ()
    preprocessing: tuple[str, ...] = ()
    models: tuple[Mapping[str, Any], ...] = ()
    solver_settings: Mapping[str, Any] = field(default_factory=dict)
    random_seed: int = 0
    equation_versions: tuple[str, ...] = ()
    input_hash: str = ""
    output_hash: str = ""
    validation_summary: Mapping[str, Any] = field(default_factory=dict)
    runtime: Mapping[str, Any] = field(default_factory=dict)
    code_revision: str = ""
    result_status: str = ""
    sensitive_notes: tuple[str, ...] = ()

    def digest(self) -> str:
        """Content digest of the manifest excluding non-reproducible runtime fields."""
        payload = {
            "task_id": self.task_id,
            "question": self.question,
            "datasets": list(self.datasets),
            "preprocessing": list(self.preprocessing),
            "models": list(self.models),
            "solver_settings": dict(self.solver_settings),
            "random_seed": self.random_seed,
            "equation_versions": list(self.equation_versions),
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "validation_summary": dict(self.validation_summary),
            "code_revision": self.code_revision,
        }
        return _stable_hash(payload)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the manifest with secrets redacted."""
        return redact_secrets(
            {
                "task_id": self.task_id,
                "question": self.question,
                "datasets": list(self.datasets),
                "preprocessing": list(self.preprocessing),
                "models": list(self.models),
                "solver_settings": dict(self.solver_settings),
                "random_seed": self.random_seed,
                "equation_versions": list(self.equation_versions),
                "input_hash": self.input_hash,
                "output_hash": self.output_hash,
                "validation_summary": dict(self.validation_summary),
                "runtime": dict(self.runtime),
                "code_revision": self.code_revision,
                "result_status": self.result_status,
                "digest": self.digest(),
            }
        )


class ProvenanceRecorder:
    """Build, hash, and retain provenance manifests (Section 3.H)."""

    def __init__(self) -> None:
        """Initialize the in-memory manifest store."""
        self._manifests: dict[str, RunManifest] = {}

    def record(self, manifest: RunManifest) -> ProvenanceRef:
        """Record one manifest and return its reference."""
        provenance_id = f"prov_{manifest.task_id}_{len(self._manifests) + 1:04d}"
        self._manifests[provenance_id] = manifest
        return ProvenanceRef(
            provenance_id=provenance_id,
            digest=manifest.digest(),
            recorded_at=_utc_now(),
        )

    def get(self, provenance_id: str) -> RunManifest:
        """Return a recorded manifest by reference."""
        if provenance_id not in self._manifests:
            raise KeyError(f"unknown provenance reference {provenance_id!r}")
        return self._manifests[provenance_id]

    def runtime_environment(self) -> dict[str, Any]:
        """Capture the hardware and runtime environment for the manifest."""
        return {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or "unreported",
            "numpy_version": __import__("numpy").__version__,
            "gpu_units_requested": 0,
            "note": "CPU-only reference execution in this repository",
        }

    def code_revision(self) -> str:
        """Return the current repository revision."""
        return _code_revision()

    @staticmethod
    def reproducibility_check(
        first: RunManifest, second: RunManifest
    ) -> dict[str, Any]:
        """Compare two manifests for reproducibility under fixed settings."""
        reproducible = (
            first.digest() == second.digest() and first.output_hash == second.output_hash
        )
        return {
            "reproducible": bool(reproducible),
            "digest_match": first.digest() == second.digest(),
            "output_hash_match": first.output_hash == second.output_hash,
            "seed_match": first.random_seed == second.random_seed,
            "settings_match": first.solver_settings == second.solver_settings,
            "note": (
                "fixed seeds and deterministic settings must produce identical output hashes"
            ),
        }

    def build_manifest(
        self,
        *,
        task_id: str,
        question: str,
        models: tuple[Mapping[str, Any], ...],
        solver_settings: Mapping[str, Any],
        random_seed: int,
        equation_versions: tuple[str, ...],
        input_payload: Any,
        output_payload: Any,
        validation_summary: Mapping[str, Any],
        result_status: str,
        datasets: tuple[Mapping[str, Any], ...] = (),
        preprocessing: tuple[str, ...] = (),
    ) -> RunManifest:
        """Assemble a complete manifest from the run's components."""
        return RunManifest(
            task_id=task_id,
            question=question,
            datasets=tuple(dict(item) for item in datasets),
            preprocessing=tuple(preprocessing),
            models=tuple(dict(item) for item in models),
            solver_settings=dict(solver_settings),
            random_seed=int(random_seed),
            equation_versions=tuple(equation_versions),
            input_hash=_stable_hash(input_payload),
            output_hash=_stable_hash(output_payload),
            validation_summary=dict(validation_summary),
            runtime=self.runtime_environment(),
            code_revision=self.code_revision(),
            result_status=result_status,
        )


def _utc_now() -> str:
    """Current UTC timestamp in ISO-8601."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")
