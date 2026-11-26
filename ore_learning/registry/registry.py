"""The Replaceable Learned-Model Registry (Point 14, Steps 6, 51-56).

Immutable model identity, artifact digests, schema compatibility, lifecycle states,
and lineage live here.  The registry refuses missing digests, incompatible schemas,
and provenance-incomplete entries at admission; queries exclude revoked, quarantined,
retired, and expired entries; lifecycle transitions follow an explicit state machine
with approval gates for high-risk promotions.  Corrections are new linked events.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable, Iterable, Sequence

from ..contracts import (
    ApprovalRecord,
    Capability,
    CapabilityRequest,
    LifecycleEventType,
    LifecycleState,
    ModelLifecycleEvent,
    ModelManifest,
    ValidationReport,
    default_schema_catalog,
    validate_schema_against_catalog,
)
from ..contracts.events import HIGH_RISK_TRANSITIONS
from .lineage import LineageGraph

__all__ = ("RegistryRejection", "LifecycleTransitionError", "ModelRegistry",
           "RegistryQueryResult", "ALLOWED_TRANSITIONS", "STATE_EVENT_TYPE")


class RegistryRejection(ValueError):
    """Raised when a manifest fails registry admission (acceptance tests 1-4)."""


class LifecycleTransitionError(ValueError):
    """Raised when a lifecycle transition violates the state machine or approval gate."""


#: The explicit lifecycle state machine (Section 2.2); corrections become new events.
ALLOWED_TRANSITIONS: dict[LifecycleState, tuple[LifecycleState, ...]] = {
    LifecycleState.PROPOSED: (LifecycleState.VALIDATED, LifecycleState.REVOKED,
                              LifecycleState.RETIRED, LifecycleState.QUARANTINED),
    LifecycleState.VALIDATED: (LifecycleState.CANARY, LifecycleState.ACTIVE,
                               LifecycleState.DEPRECATED, LifecycleState.REVOKED,
                               LifecycleState.RETIRED, LifecycleState.QUARANTINED),
    LifecycleState.CANARY: (LifecycleState.ACTIVE, LifecycleState.VALIDATED,
                            LifecycleState.QUARANTINED, LifecycleState.DEPRECATED,
                            LifecycleState.RETIRED, LifecycleState.REVOKED),
    LifecycleState.ACTIVE: (LifecycleState.CANARY, LifecycleState.DEPRECATED,
                            LifecycleState.QUARANTINED, LifecycleState.RETIRED,
                            LifecycleState.REVOKED),
    LifecycleState.QUARANTINED: (LifecycleState.VALIDATED,
                                  LifecycleState.DEPRECATED, LifecycleState.REVOKED,
                                  LifecycleState.RETIRED),
    LifecycleState.DEPRECATED: (LifecycleState.RETIRED, LifecycleState.REVOKED,
                                LifecycleState.ACTIVE),
    LifecycleState.RETIRED: (),
    LifecycleState.REVOKED: (),
}

#: Mapping from target state to the lifecycle event type it emits.
STATE_EVENT_TYPE: dict[LifecycleState, LifecycleEventType] = {
    LifecycleState.PROPOSED: LifecycleEventType.REGISTERED,
    LifecycleState.QUARANTINED: LifecycleEventType.QUARANTINED,
    LifecycleState.VALIDATED: LifecycleEventType.VALIDATED,
    LifecycleState.CANARY: LifecycleEventType.CANARY_STARTED,
    LifecycleState.ACTIVE: LifecycleEventType.ACTIVATED,
    LifecycleState.DEPRECATED: LifecycleEventType.ROLLED_BACK,
    LifecycleState.REVOKED: LifecycleEventType.REVOKED,
    LifecycleState.RETIRED: LifecycleEventType.RETIRED,
}

#: States that can never serve new assignments (Steps 6 and 51).
_NON_SELECTABLE_STATES = (LifecycleState.REVOKED, LifecycleState.QUARANTINED,
                          LifecycleState.RETIRED, LifecycleState.DEPRECATED,
                          LifecycleState.PROPOSED)


@dataclass(slots=True)
class RegistryQueryResult:
    """Outcome of one registry query, logged for audit (Step 6)."""

    task_id: str
    matched: list[ModelManifest]
    rejected: dict[str, str]
    queried_at: float

    @property
    def eligible(self) -> list[ModelManifest]:
        """The matched manifests (compatibility filtering happens in routing)."""
        return self.matched


class ModelRegistry:
    """Versioned registry of replaceable learned models (Section 2.1, Step 6)."""

    def __init__(self, event_listener: Callable[[ModelLifecycleEvent], None] | None = None,
                 schema_catalog: dict[tuple[str, str], tuple[str, ...]] | None = None) -> None:
        """Create an empty registry with an optional lifecycle-event listener."""
        self._entries: dict[tuple[str, str], ModelManifest] = {}
        self._events: list[ModelLifecycleEvent] = []
        self._sequence = 0
        self._listener = event_listener
        self._catalog = schema_catalog if schema_catalog is not None else default_schema_catalog()
        self.lineage = LineageGraph()
        self.query_log: list[RegistryQueryResult] = []

    # ------------------------------------------------------------- admission

    def register(self, manifest: ModelManifest, *, base_artifact: str = "",
                 replace: bool = False) -> ModelManifest:
        """Admit a manifest after full admission checks (acceptance 1-4).

        Rejects missing artifact digests, unknown or field-incompatible schemas, and
        provenance-incomplete entries.  Identity (model_id, version) is immutable:
        re-registering the same identity with a different digest is refused unless
        ``replace`` is explicitly requested for a byte-identical manifest.
        """
        if not isinstance(manifest, ModelManifest):
            raise RegistryRejection("only ModelManifest objects can be registered")
        if not manifest.artifact_digest or len(manifest.artifact_digest) < 8:
            raise RegistryRejection("artifact digest is missing or too short")
        ok, reason = validate_schema_against_catalog(manifest.input_schema, self._catalog)
        if not ok:
            raise RegistryRejection(f"input schema rejected: {reason}")
        ok, reason = validate_schema_against_catalog(manifest.output_schema, self._catalog)
        if not ok:
            raise RegistryRejection(f"output schema rejected: {reason}")
        if manifest.input_schema.schema_version != manifest.schema_version and \
                manifest.output_schema.schema_version != manifest.schema_version:
            raise RegistryRejection(
                "schema_version must match the input or output schema descriptor version")
        if manifest.training_data_provenance is None or \
                not manifest.training_data_provenance.dataset_id or \
                not manifest.training_data_provenance.license:
            raise RegistryRejection("provenance-incomplete entry: dataset id and license "
                                    "are mandatory")

        key = (manifest.model_id, manifest.version)
        existing = self._entries.get(key)
        if existing is not None:
            if existing.artifact_digest != manifest.artifact_digest and not replace:
                raise RegistryRejection(
                    f"{manifest.identity} is already registered with a different artifact "
                    "digest; artifact digests prevent silent substitution")
            if existing == manifest:
                return existing
        self._entries[key] = manifest
        self.lineage.register(manifest, base_artifact=base_artifact)
        self._emit(LifecycleEventType.REGISTERED, manifest, None, manifest.lifecycle_state,
                   reason="manifest admitted to the registry",
                   evidence_refs=(manifest.artifact_digest,))
        return manifest

    # ------------------------------------------------------------- queries

    def get(self, model_id: str, version: str) -> ModelManifest | None:
        """Return one manifest or None."""
        return self._entries.get((model_id, version))

    def require(self, model_id: str, version: str) -> ModelManifest:
        """Return one manifest or raise KeyError."""
        manifest = self.get(model_id, version)
        if manifest is None:
            raise KeyError(f"{model_id}@{version} is not registered")
        return manifest

    def list_models(self, states: Sequence[LifecycleState] | None = None) -> list[ModelManifest]:
        """List manifests, optionally filtered by lifecycle state."""
        entries = list(self._entries.values())
        if states:
            wanted = set(states)
            entries = [entry for entry in entries if entry.lifecycle_state in wanted]
        return sorted(entries, key=lambda entry: (entry.model_id, entry.version))

    def query(self, request: CapabilityRequest) -> RegistryQueryResult:
        """Step 6 -- retrieve capability- and modality-compatible, selectable models.

        Revoked, quarantined, retired, deprecated, and still-proposed entries are
        rejected with explicit reasons; the query itself is logged.
        """
        rejected: dict[str, str] = {}
        matched: list[ModelManifest] = []
        required = set(item.value for item in request.capabilities)
        for manifest in self.list_models():
            identity = manifest.identity
            declared = {item.value if hasattr(item, "value") else str(item)
                        for item in manifest.supported_tasks}
            if manifest.lifecycle_state in _NON_SELECTABLE_STATES:
                rejected[identity] = f"lifecycle state {manifest.lifecycle_state.value} " \
                                     "is not selectable"
                continue
            if not required.intersection(declared):
                rejected[identity] = "declared tasks do not cover the requested capabilities"
                continue
            if request.modality not in manifest.modalities:
                rejected[identity] = (f"modality {request.modality.value} unsupported "
                                      f"(has {[m.value for m in manifest.modalities]})")
                continue
            if manifest.output_schema.schema_type != request.output_contract.schema_type:
                rejected[identity] = (
                    f"output schema {manifest.output_schema.schema_type} does not satisfy "
                    f"the task contract {request.output_contract.schema_type}")
                continue
            if manifest.output_schema.schema_version != request.output_contract.schema_version:
                rejected[identity] = "output schema version does not satisfy the task contract"
                continue
            if not set(request.languages).intersection(manifest.languages):
                rejected[identity] = "no shared supported language"
                continue
            matched.append(manifest)
        result = RegistryQueryResult(task_id=request.task_id, matched=matched,
                                     rejected=rejected, queried_at=time.time())
        self.query_log.append(result)
        return result

    def verify_artifact(self, model_id: str, version: str, digest: str) -> bool:
        """Check that a runtime artifact still matches its registered digest."""
        manifest = self.get(model_id, version)
        if manifest is None:
            return False
        return bool(digest) and digest == manifest.artifact_digest

    # ------------------------------------------------------------- lifecycle

    def transition(self, model_id: str, version: str, to_state: LifecycleState, *,
                   reason: str = "", task_id: str = "",
                   actor: str = "ore_learning.registry",
                   approval: ApprovalRecord | None = None,
                   validation: ValidationReport | None = None) -> ModelLifecycleEvent:
        """Apply one lifecycle transition under the state machine (Steps 51-56).

        Activations and promotions of high-risk models require a recorded approval;
        entering VALIDATED from QUARANTINED (re-entry) requires re-evaluation and
        approval; retiring preserves lineage and evaluation history by construction.
        """
        manifest = self.require(model_id, version)
        from_state = manifest.lifecycle_state
        if to_state not in ALLOWED_TRANSITIONS[from_state]:
            raise LifecycleTransitionError(
                f"transition {from_state.value} -> {to_state.value} is not allowed")
        if to_state is LifecycleState.VALIDATED:
            if validation is None or not validation.verified:
                raise LifecycleTransitionError(
                    "entering validated requires an independently verified report")
            if from_state is LifecycleState.QUARANTINED and approval is None:
                raise LifecycleTransitionError(
                    "quarantine re-entry requires re-evaluation and approval")
        updated = manifest.with_state(to_state)
        self._entries[(model_id, version)] = updated
        event = self._emit(STATE_EVENT_TYPE[to_state], updated, from_state,
                           to_state, reason=reason, task_id=task_id, actor=actor,
                           approval=approval)
        return event

    def promote(self, model_id: str, version: str, *, risk_class: str,
                reason: str = "", task_id: str = "",
                approval: ApprovalRecord | None = None,
                validation: ValidationReport | None = None,
                to_state: LifecycleState = LifecycleState.ACTIVE) -> ModelLifecycleEvent:
        """Steps 52-55 -- promotion with approval gating for high-risk transitions.

        ``risk_class`` comes from the task policy.  High-risk activations or
        promotions without a recorded approval are refused.
        """
        if to_state not in (LifecycleState.ACTIVE, LifecycleState.CANARY):
            raise LifecycleTransitionError("promotion targets active or canary states")
        if risk_class in ("high", "critical") and approval is None:
            raise LifecycleTransitionError(
                "high-risk promotion requires recorded human approval (Step 55)")
        if to_state is LifecycleState.CANARY and validation is not None and \
                not validation.verified:
            raise LifecycleTransitionError("cannot enter canary without verification")
        return self.transition(model_id, version, to_state, reason=reason,
                               task_id=task_id,
                               actor="ore_learning.promotion", approval=approval,
                               validation=validation)

    def quarantine(self, model_id: str, version: str, reason: str,
                   task_id: str = "") -> ModelLifecycleEvent:
        """Step 51 -- quarantine a degraded model deterministically."""
        return self.transition(model_id, version, LifecycleState.QUARANTINED,
                               reason=reason, task_id=task_id,
                               actor="ore_learning.monitoring")

    def rollback(self, model_id: str, version: str, fallback_model_id: str,
                 fallback_version: str, reason: str,
                 task_id: str = "") -> tuple[ModelLifecycleEvent, ModelLifecycleEvent]:
        """Step 51 -- roll back to a known-good version: current model is deprecated,
        the validated fallback is (re)activated, and both events are linked."""
        deprecation = self.transition(model_id, version, LifecycleState.DEPRECATED,
                                      reason=f"rolled back: {reason}", task_id=task_id,
                                      actor="ore_learning.monitoring")
        fallback = self.require(fallback_model_id, fallback_version)
        if fallback.lifecycle_state is not LifecycleState.VALIDATED:
            raise LifecycleTransitionError(
                "rollback target must be a validated known-good version")
        activation = self.transition(fallback_model_id, fallback_version,
                                     LifecycleState.ACTIVE,
                                     reason=f"known-good fallback after {reason}",
                                     task_id=task_id, actor="ore_learning.monitoring")
        return deprecation, activation

    def activate_fallback(self, model_id: str, version: str, reason: str,
                          task_id: str = "") -> ModelLifecycleEvent:
        """Step 51 -- activate a validated known-good fallback after a failure."""
        return self.transition(model_id, version, LifecycleState.ACTIVE,
                               reason=reason, task_id=task_id,
                               actor="ore_learning.monitoring")

    def retire(self, model_id: str, version: str, reason: str) -> ModelLifecycleEvent:
        """Step 56 -- retire a specialist; lineage and evaluations are preserved."""
        return self.transition(model_id, version, LifecycleState.RETIRED, reason=reason,
                               actor="ore_learning.retirement")

    def events(self, model_id: str | None = None,
               version: str | None = None) -> list[ModelLifecycleEvent]:
        """Return lifecycle events, optionally for one model identity."""
        events = list(self._events)
        if model_id is not None:
            events = [event for event in events if event.model_id == model_id]
            if version is not None:
                events = [event for event in events if event.model_version == version]
        return events

    def _emit(self, event_type: LifecycleEventType, manifest: ModelManifest,
              from_state: LifecycleState | None, to_state: LifecycleState, *,
              reason: str, task_id: str = "", actor: str = "ore_learning.registry",
              approval: ApprovalRecord | None = None,
              evidence_refs: tuple[str, ...] = ()) -> ModelLifecycleEvent:
        """Append one lifecycle event and notify the listener."""
        self._sequence += 1
        event = ModelLifecycleEvent(
            event_id=f"evt-{manifest.model_id}-{self._sequence:06d}",
            event_type=event_type, model_id=manifest.model_id,
            model_version=manifest.version, from_state=from_state, to_state=to_state,
            task_id=task_id, actor=actor, reason=reason,
            evidence_refs=evidence_refs or (manifest.artifact_digest,),
            sequence=self._sequence, timestamp=time.time(), approval=approval)
        self._events.append(event)
        if self._listener is not None:
            self._listener(event)
        return event

    def describe(self) -> dict[str, Any]:
        """Summarize registry contents for diagnostics."""
        states: dict[str, int] = {}
        for manifest in self._entries.values():
            states[manifest.lifecycle_state.value] = states.get(
                manifest.lifecycle_state.value, 0) + 1
        return {"models": len(self._entries), "states": states,
                "events": len(self._events), "queries": len(self.query_log),
                "lineage": self.lineage.describe()}
