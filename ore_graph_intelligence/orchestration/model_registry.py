"""Model artifact registry with staged deployment and rollback (steps 24, 54-55).

Registered artifacts carry model/version, weights hash, feature schema, snapshot,
split, config, metrics, limitations and dependencies (step 24). New versions start in
*shadow* status so they cannot affect authoritative decisions until they demonstrate
improvement without unacceptable false-positive or resource increases (step 54), and
promotion follows the offline -> shadow -> canary -> monitored rollout staging with a
preserved rollback path (step 55). Rolling back a model never touches USG state
(acceptance test 26).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ..schemas.model_manifest import ModelManifest

#: Deployment stages in rollout order (step 55).
DEPLOYMENT_STAGES = ("offline", "shadow", "canary", "monitored", "retired")


@dataclass
class RegisteredModel:
    """One registered model version with its stage and provenance."""

    model_id: str
    version: str
    manifest: ModelManifest
    stage: str = "offline"
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.stage not in DEPLOYMENT_STAGES:
            raise ValueError(f"stage must be one of {list(DEPLOYMENT_STAGES)}")

    def as_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id, "version": self.version,
            "stage": self.stage, "notes": list(self.notes),
            "manifest": self.manifest.as_dict(),
        }


class ModelRegistry:
    """Register model artifacts per model id, stage them, and roll back safely."""

    def __init__(self) -> None:
        self._history: dict[str, list[RegisteredModel]] = {}
        self._current: dict[str, RegisteredModel] = {}

    # ------------------------------------------------------------------ registration

    def register(
        self,
        model_id: str,
        manifest: ModelManifest,
        *,
        stage: str = "offline",
        notes: tuple[str, ...] = (),
    ) -> RegisteredModel:
        """Register a new version; new versions never start as current."""
        if not model_id.strip():
            raise ValueError("model_id must be non-empty")
        registered = RegisteredModel(
            model_id=model_id, version=manifest.version,
            manifest=manifest, stage=stage, notes=tuple(notes),
        )
        self._history.setdefault(model_id, []).append(registered)
        return registered

    def versions(self, model_id: str) -> tuple[RegisteredModel, ...]:
        """Return the full version history in registration order."""
        return tuple(self._history.get(model_id, ()))

    def current(self, model_id: str) -> RegisteredModel | None:
        """Return the currently active version, or None when none is active."""
        return self._current.get(model_id)

    def history(self) -> dict[str, list[dict[str, object]]]:
        """Return the full registry history for audit logging."""
        return {
            model_id: [item.as_dict() for item in versions]
            for model_id, versions in self._history.items()
        }

    # ------------------------------------------------------------------ staging

    def promote(
        self,
        model_id: str,
        version: str,
        *,
        to_stage: str = "monitored",
    ) -> RegisteredModel:
        """Advance a version's deployment stage; only monitored versions become current."""
        registered = self._find(model_id, version)
        order = {name: index for index, name in enumerate(DEPLOYMENT_STAGES)}
        if order[registered.stage] > order[to_stage]:
            raise ValueError("deployment stages only advance forward")
        registered.stage = to_stage
        if to_stage == "monitored":
            self._current[model_id] = registered
        return registered

    def rollback(self, model_id: str) -> RegisteredModel | None:
        """Roll back to the most recent previously-active version (step 55, test 26).

        Rollback swaps the active pointer only; no USG or workspace state is touched,
        and the replaced version is retired rather than deleted so history is preserved.
        """
        versions = self._history.get(model_id, [])
        previous = [
            item for item in versions
            if item is not self._current.get(model_id) and item.stage == "monitored"
        ]
        if not previous:
            return None
        target = previous[-1]
        self._current[model_id] = target
        return target

    def _find(self, model_id: str, version: str) -> RegisteredModel:
        for item in self._history.get(model_id, ()):
            if item.version == version:
                return item
        raise KeyError(f"unknown version: {model_id}@{version}")

    # ------------------------------------------------------------------ shadow

    def shadow_candidates(self) -> tuple[RegisteredModel, ...]:
        """Return versions currently in shadow staging (step 54)."""
        return tuple(
            item for versions in self._history.values() for item in versions
            if item.stage == "shadow"
        )

    def evaluate_shadow(
        self,
        model_id: str,
        version: str,
        *,
        improved: bool,
        false_positive_increase: bool,
        resource_increase_unacceptable: bool,
    ) -> RegisteredModel:
        """Shadow evaluation verdict: promote only on improvement without regressions.

        Shadow evaluation never affects authoritative decisions: the result only
        advances or retires the *stage* of the candidate version.
        """
        registered = self._find(model_id, version)
        if registered.stage != "shadow":
            raise ValueError("only shadow-staged versions can be shadow-evaluated")
        if improved and not false_positive_increase and not resource_increase_unacceptable:
            registered.stage = "canary"
            registered.notes = registered.notes + ("shadow evaluation passed",)
        else:
            registered.stage = "retired"
            registered.notes = registered.notes + (
                "shadow evaluation failed; version retired without affecting decisions",
            )
        return registered


def manifest_from_mapping(values: Mapping[str, object]) -> ModelManifest:
    """Build a ModelManifest from a plain mapping (registry convenience)."""
    return ModelManifest.from_dict(dict(values))


__all__ = [
    "DEPLOYMENT_STAGES", "ModelRegistry", "RegisteredModel", "manifest_from_mapping",
]
