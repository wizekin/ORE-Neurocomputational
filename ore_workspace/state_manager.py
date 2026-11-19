"""Validate, commit, version and publish all mutations to the shared cognitive workspace."""
from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

from .models import (
    APPEND_ONLY_FIELDS, Finding, UpdateDecision, ValidationResult,
    WorkspaceEvent, WorkspaceRevision, canonical_json, utc_now,
)
from .merge_policies import MergePolicyRegistry


class WorkspaceStateManager:
    """Enforce publication contracts and atomically integrate findings into a workspace."""

    def __init__(self, workspace, observer=None, merge_policies: MergePolicyRegistry | None = None) -> None:
        """Bind a workspace, logger observer and field-specific merge-policy registry."""
        self.workspace = workspace
        self.observer = observer
        self.merge_policies = merge_policies or MergePolicyRegistry()

    def _record(self, method_name: str, metadata: dict[str, Any]) -> None:
        """Send a compact operation record to the observer without coupling it to shared state."""
        if self.observer:
            self.observer.record(
                method_name, metadata, class_name="WorkspaceStateManager",
                module_name="ore_workspace.state_manager",
            )

    def validate(self, finding: Finding) -> ValidationResult:
        """Validate schema, ownership, revisions, provenance, dependencies and duplicate identity."""
        workspace = self.workspace
        with workspace.lock:
            return self._validate_locked(finding)

    def _validate_locked(self, finding: Finding) -> ValidationResult:
        """Run all publication checks while the workspace revision and registries are stable."""
        state = self.workspace._state
        revision = state.workspace_revision
        if finding.workspace_id != state.workspace_id:
            return ValidationResult(UpdateDecision.REJECTED, finding.finding_id,
                                   "finding references a different workspace", revision)
        if finding.target.field not in self.merge_policies._policies:
            return ValidationResult(UpdateDecision.REJECTED, finding.finding_id,
                                   f"unknown or non-writable field: {finding.target.field}", revision)
        if not self.workspace.permissions.can_write(finding.producer_module, finding.target.field):
            return ValidationResult(UpdateDecision.REJECTED, finding.finding_id,
                                   "producer is not authorized to modify the target field", revision)
        try:
            json.dumps(dict(finding.payload), ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError):
            return ValidationResult(UpdateDecision.REJECTED, finding.finding_id,
                                   "payload must contain finite JSON-compatible values", revision)
        if finding.base_workspace_revision > revision:
            return ValidationResult(UpdateDecision.REJECTED, finding.finding_id,
                                   "finding was generated against a future workspace revision", revision)
        if (finding.provenance.source_span is not None
            and finding.provenance.source.casefold() in {"current_input", "input", "current_input_span"}):
            start, end = finding.provenance.source_span
            if end > len(state.current_input):
                return ValidationResult(UpdateDecision.REJECTED, finding.finding_id,
                                       "provenance source span exceeds the current input", revision)
        if finding.finding_id in self.workspace.findings or finding.finding_id in self.workspace.pending_findings:
            return ValidationResult(UpdateDecision.DUPLICATE, finding.finding_id,
                                   "finding ID has already been submitted", revision)
        fingerprint = finding.fingerprint()
        if fingerprint in self.workspace.finding_fingerprints:
            return ValidationResult(UpdateDecision.DUPLICATE, finding.finding_id,
                                   "equivalent finding content has already been submitted", revision)
        missing = [dependency for dependency in finding.dependencies
                   if dependency not in state.accepted_updates]
        if missing:
            return ValidationResult(UpdateDecision.DEFERRED, finding.finding_id,
                                   f"missing dependencies: {', '.join(missing)}", revision)
        rebased_from = None
        if finding.base_workspace_revision < revision:
            if finding.target.field not in APPEND_ONLY_FIELDS:
                return ValidationResult(UpdateDecision.DEFERRED, finding.finding_id,
                                       "stale mutable-field update must be regenerated from a fresh snapshot", revision)
            rebased_from = finding.base_workspace_revision
        return ValidationResult(UpdateDecision.ACCEPTED, finding.finding_id,
                               "finding passed publication validation", revision,
                               rebased_from_revision=rebased_from)

    def submit(self, finding: Finding) -> ValidationResult:
        """Validate and atomically accept, merge, defer, reject or register a submitted finding."""
        workspace = self.workspace
        event = None
        conflict_record = None
        with workspace.lock:
            result = self._validate_locked(finding)
            if result.decision is UpdateDecision.DUPLICATE:
                self._record("submit", {"finding_id": finding.finding_id, "decision": "duplicate"})
                return result
            if result.decision is UpdateDecision.DEFERRED:
                workspace.pending_findings[finding.finding_id] = finding
                if finding.finding_id not in workspace._state.pending_updates:
                    workspace._state.pending_updates.append(finding.finding_id)
                self._advance_metadata_revision(
                    change_origin="state_manager.defer", validation_result="deferred",
                    changed_fields=("pending_updates",), finding_ids=(finding.finding_id,),
                    dependencies=finding.dependencies,
                )
                deferred = replace(result, workspace_revision=workspace._state.workspace_revision)
                self._record("submit", {"finding_id": finding.finding_id, "decision": "deferred"})
                return deferred
            if result.decision is UpdateDecision.REJECTED:
                workspace.rejected_findings[finding.finding_id] = finding
                workspace._state.rejected_updates.append(finding.finding_id)
                self._advance_metadata_revision(
                    change_origin="state_manager.reject", validation_result="rejected",
                    changed_fields=("rejected_updates",), finding_ids=(finding.finding_id,),
                    dependencies=finding.dependencies,
                )
                self._record("submit", {"finding_id": finding.finding_id, "decision": "rejected", "reason": result.message})
                return replace(result, workspace_revision=workspace._state.workspace_revision)
            candidate_state = workspace._state.clone()
            try:
                changed = self.merge_policies.apply(candidate_state, finding)
            except (TypeError, ValueError, KeyError) as exc:
                workspace.rejected_findings[finding.finding_id] = finding
                workspace._state.rejected_updates.append(finding.finding_id)
                self._advance_metadata_revision(
                    change_origin="state_manager.reject_merge", validation_result="rejected",
                    changed_fields=("rejected_updates",), finding_ids=(finding.finding_id,),
                    dependencies=finding.dependencies,
                )
                rejected = ValidationResult(
                    UpdateDecision.REJECTED, finding.finding_id,
                    f"field merge validation failed: {type(exc).__name__}: {exc}",
                    workspace._state.workspace_revision,
                )
                self._record("submit", {"finding_id": finding.finding_id, "decision": "rejected", "reason": rejected.message})
                return rejected
            if not changed:
                workspace.findings[finding.finding_id] = finding
                workspace.finding_fingerprints[finding.fingerprint()] = finding.finding_id
                candidate_state.accepted_updates.append(finding.finding_id)
                candidate_state.pending_updates = [item for item in candidate_state.pending_updates if item != finding.finding_id]
                workspace._state = candidate_state
                self._advance_metadata_revision(
                    change_origin=finding.producer_module, validation_result="merged_no_state_delta",
                    changed_fields=(), finding_ids=(finding.finding_id,), dependencies=finding.dependencies,
                )
                result = ValidationResult(
                    UpdateDecision.MERGED, finding.finding_id,
                    "finding was integrated as an idempotent field-policy merge", workspace._state.workspace_revision,
                )
                self._record("submit", {"finding_id": finding.finding_id, "decision": "merged_no_state_delta"})
                return result
            conflicting = self._find_conflicting_finding(finding)
            if conflicting is not None:
                conflict_record = workspace.conflicts.register(
                    workspace.workspace_id, conflicting, finding,
                    conflict_type=f"{finding.target.field}_hypothesis_conflict",
                )
            next_revision = workspace._state.workspace_revision + 1
            candidate_state.workspace_revision = next_revision
            candidate_state.updated_at = utc_now()
            candidate_state.accepted_updates.append(finding.finding_id)
            candidate_state.pending_updates = [item for item in candidate_state.pending_updates if item != finding.finding_id]
            workspace._state = candidate_state
            workspace.findings[finding.finding_id] = finding
            workspace.finding_fingerprints[finding.fingerprint()] = finding.finding_id
            workspace.revisions.add(WorkspaceRevision(
                revision=next_revision,
                snapshot=candidate_state.to_dict(),
                changed_fields=(finding.target.field,),
                accepted_finding_ids=(finding.finding_id,),
                change_origin=finding.producer_module,
                dependencies=finding.dependencies,
                validation_result="conflict_preserved" if conflict_record else "accepted",
            ))
            event = WorkspaceEvent(
                workspace_id=workspace.workspace_id,
                workspace_revision=next_revision,
                event_type="workspace_field_updated",
                changed_fields=(finding.target.field,),
                producer_module=finding.producer_module,
                finding_ids=(finding.finding_id,),
            )
            workspace.events.append(event)
            decision = UpdateDecision.CONFLICT if conflict_record else UpdateDecision.ACCEPTED
            result = ValidationResult(
                decision, finding.finding_id,
                "finding committed and competing evidence retained" if conflict_record else "finding committed",
                next_revision, changed_fields=(finding.target.field,),
                conflict_id=conflict_record.conflict_id if conflict_record else None,
                rebased_from_revision=result.rebased_from_revision,
            )
            self._record("submit", {
                "finding_id": finding.finding_id, "decision": decision.value,
                "workspace_revision": next_revision, "changed_field": finding.target.field,
                "conflict_id": conflict_record.conflict_id if conflict_record else None,
            })
        if event is not None:
            self.notify_subscribers(event)
            self.reconsider_pending()
        return result

    def notify_subscribers(self, event: WorkspaceEvent):
        """Publish a committed event and retain every subscriber's independent delivery result."""
        deliveries = self.workspace.event_bus.publish(event)
        with self.workspace.lock:
            self.workspace.delivery_results.extend(deliveries)
        self._record("notify_subscribers", {
            "event_id": event.event_id,
            "delivered": sum(item.delivered for item in deliveries),
            "failed": sum(not item.delivered for item in deliveries),
        })
        return deliveries

    def _find_conflicting_finding(self, finding: Finding) -> Finding | None:
        """Find an earlier accepted hypothesis for the same target with a distinct candidate."""
        incoming_candidate = self._candidate_value(finding)
        if incoming_candidate is None:
            return None
        for existing in self.workspace.findings.values():
            if existing.target != finding.target:
                continue
            existing_candidate = self._candidate_value(existing)
            if existing_candidate is not None and existing_candidate != incoming_candidate:
                return existing
        return None

    def _candidate_value(self, finding: Finding) -> Any:
        """Extract an explicit interpretation label without treating unrelated payload keys as conflict."""
        for key in ("candidate", "interpretation", "label", "parse", "meaning"):
            if key in finding.payload:
                return canonical_json(finding.payload[key])
        return None

    def _advance_metadata_revision(
        self,
        *,
        change_origin: str,
        validation_result: str,
        changed_fields: tuple[str, ...],
        finding_ids: tuple[str, ...],
        dependencies: tuple[str, ...],
    ) -> None:
        """Version metadata-only queue changes so every exposed state snapshot remains reproducible."""
        state = self.workspace._state
        revision = state.workspace_revision + 1
        state.workspace_revision = revision
        state.updated_at = utc_now()
        self.workspace.revisions.add(WorkspaceRevision(
            revision=revision, snapshot=state.to_dict(), changed_fields=changed_fields,
            accepted_finding_ids=finding_ids, change_origin=change_origin,
            dependencies=dependencies, validation_result=validation_result,
        ))

    def reconsider_pending(self) -> tuple[ValidationResult, ...]:
        """Retry deferred findings whose dependencies may now be available after a new commit."""
        with self.workspace.lock:
            candidates = tuple(self.workspace.pending_findings.values())
            accepted = set(self.workspace._state.accepted_updates)
        results = []
        for finding in candidates:
            if set(finding.dependencies).issubset(accepted):
                with self.workspace.lock:
                    self.workspace.pending_findings.pop(finding.finding_id, None)
                    self.workspace._state.pending_updates = [
                        item for item in self.workspace._state.pending_updates if item != finding.finding_id
                    ]
                results.append(self.submit(finding))
        return tuple(results)

    def resolve_conflict(
        self,
        conflict_id: str,
        *,
        preferred_finding_id: str | None = None,
        resolution_candidates: tuple[str, ...] = (),
    ):
        """Record an explicit control decision while retaining the findings involved in the conflict."""
        record = self.workspace.conflicts.resolve(
            conflict_id, preferred_finding_id=preferred_finding_id,
            resolution_candidates=resolution_candidates,
        )
        self._record("resolve_conflict", {
            "conflict_id": conflict_id, "preferred_finding_id": preferred_finding_id,
        })
        return record