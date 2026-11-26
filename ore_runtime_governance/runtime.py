"""The trusted core: ``ComponentRuntime`` wires all six planes (step 7).

The trusted core stays small -- identity, manifest validation, capability
checks, resource reservations, lifecycle transitions, IPC routing, and audit
emission (integrated step 7); domain reasoning lives in ordinary services.
``register``/``start``/``stop`` follow the section-6 interface contracts and
return typed outcomes. Security decisions are made before dispatch and
rechecked at sensitive transitions (section 3F). ``quarantine`` revokes
grants, freezes dispatch, isolates artifacts, and preserves evidence
(step 22). ``explain`` answers the governance question of section 11: which
component ran, under whose authority, with what budget, using which data,
under which policy, and how it recovered.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Mapping

from .artifacts import ArtifactStore
from .audit import AuditEvent, AuditLogger, InMemoryEventSink, SQLiteEventSink
from .capabilities import AccessRequest, CapabilityAuthority
from .contracts import (
    AdmissionDeniedError,
    BackendUnavailableError,
    ComponentManifest,
    ExecutionTier,
    IllegalTransitionError,
    MessageEnvelope,
    OutcomeKind,
    ResourceVector,
)
from .ipc import MessageBus
from .isolation import (
    ContainerSandboxBackend,
    ExecutionHandle,
    ExecutionState,
    InProcessBackend,
    MicroVMBackend,
    SubprocessExecutionBackend,
)
from .lifecycle import LifecycleMachine, LifecycleState, RestartPolicy, Supervisor
from .policy import OperatorControl, PolicyEngine, default_policy
from .registry import (
    ComponentRegistry,
    DependencyGraph,
    RegistrationResult,
    ServiceLease,
)
from .resources import ResourceManager
from .workspace_guards import BlackboardGuard, USGGuard

DEFAULT_CAPACITY = ResourceVector(cpu_shares=4.0, memory_bytes=512 * 1024 * 1024)


class RuntimeHandle:
    """Facade handle returned by ``ComponentRuntime.start``."""

    def __init__(self, runtime: "ComponentRuntime", execution: ExecutionHandle, reservation_id: str) -> None:
        self.runtime = runtime
        self.execution = execution
        self.reservation_id = reservation_id
        self.component_id = execution.component_id

    @property
    def state(self) -> str:
        return self.execution.state

    def wait(self, timeout: float | None = None) -> ExecutionHandle:
        return self.execution

    def stop(self, reason: str = "operator stop", deadline_seconds: float = 5.0) -> Any:
        return self.runtime.stop(self, reason=reason, deadline_seconds=deadline_seconds)

    def describe(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "execution": self.execution.describe(),
            "reservation_id": self.reservation_id,
        }


class StopResult:
    """Typed outcome of ``ComponentRuntime.stop`` (section 6)."""

    def __init__(
        self,
        component_id: str,
        outcome: OutcomeKind,
        reason: str,
        *,
        reservations_released: int = 0,
        elapsed_seconds: float = 0.0,
        quarantine: bool = False,
    ) -> None:
        self.component_id = component_id
        self.outcome = outcome
        self.reason = reason
        self.reservations_released = reservations_released
        self.elapsed_seconds = elapsed_seconds
        self.quarantine = quarantine

    def to_dict(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "outcome": self.outcome.value,
            "reason": self.reason,
            "reservations_released": self.reservations_released,
            "elapsed_seconds": self.elapsed_seconds,
            "quarantine": self.quarantine,
        }


class ComponentRuntime:
    """Microkernel-inspired user-space governance layer (section 1)."""

    def __init__(
        self,
        *,
        capacity: ResourceVector | None = None,
        bounds: ResourceVector | None = None,
        database_path: str | None = None,
        artifact_root: str | None = None,
        repository_root: str | None = None,
        policy: Any | None = None,
        audit_sink: Any | None = None,
        secret: str | bytes | None = None,
        workspace: Any | None = None,
        persist_audit: bool = True,
    ) -> None:
        self.capacity = capacity or DEFAULT_CAPACITY
        self.bounds = bounds
        self.audit = AuditLogger(
            audit_sink
            if audit_sink is not None
            else (
                SQLiteEventSink(database_path)
                if database_path and persist_audit
                else InMemoryEventSink()
            )
        )
        self.registry = ComponentRegistry(bounds=self.bounds)
        self.authority = CapabilityAuthority(secret)
        self.resources = ResourceManager(self.capacity, label="cros_root")
        self.lifecycle = LifecycleMachine()
        self.bus = MessageBus(
            capability_resolver=self._resolve_capability,
            authorizer=self.authority.authorize,
            authenticator=self.registry.authenticate,
            on_event=self._audit_ipc_event,
        )
        self.artifacts = ArtifactStore(artifact_root)
        self.policy = PolicyEngine(policy or default_policy())
        self.operator_control = OperatorControl(on_event=self._audit_policy_event)
        self.supervisor = Supervisor(
            RestartPolicy(), on_quarantine=self._on_supervisor_quarantine
        )
        self.blackboard_guard = BlackboardGuard(
            workspace, authorizer=self.authority.authorize, on_event=self._audit_ipc_event
        )
        self.usg_guard = USGGuard()
        self.backends: dict[str, Any] = {
            ExecutionTier.IN_PROCESS.value: InProcessBackend(),
            ExecutionTier.LOCAL_PROCESS.value: SubprocessExecutionBackend(
                repository_root=repository_root
            ),
            ExecutionTier.CONTAINER_SANDBOX.value: ContainerSandboxBackend(),
            ExecutionTier.MICROVM.value: MicroVMBackend(),
        }
        self._grants: dict[str, dict[str, Any]] = {}
        self._handles: dict[str, RuntimeHandle] = {}
        self._graph: DependencyGraph | None = None
        self._lock = threading.RLock()

    # -- plane 1: registration and authorization (steps 3, 16) -----------------

    def register(
        self,
        manifest: ComponentManifest,
        *,
        replace: bool = False,
    ) -> RegistrationResult:
        """Validate, register, and authorize a component (steps 3, 8, 16)."""
        with self._lock:
            self.lifecycle.apply(
                manifest.component_id,
                LifecycleState.VALIDATED,
                actor="registry",
                event="registration",
                reason="manifest validated against contracts",
                policy_version=self.policy.version,
            )
            result = self.registry.register(manifest, replace=replace)
            if not result.registered:
                self._record(
                    "registration_rejected",
                    target=manifest.component_id,
                    outcome="denied",
                    reason=result.reason,
                )
                self.lifecycle.apply(
                    manifest.component_id,
                    LifecycleState.FAILED,
                    actor="registry",
                    reason=result.reason,
                    policy_version=self.policy.version,
                )
                return result
            policy_decision = self.policy.evaluate(
                manifest.component_id,
                "register",
                {
                    "trust_class": manifest.trust_class.value,
                    "consequence": manifest.consequence.value,
                    "tier": manifest.isolation.tier.value,
                    "deadline_seconds": manifest.resources.deadline_seconds,
                    "audit_attached": True,
                },
            )
            if not policy_decision.allowed:
                self._record(
                    "registration_rejected",
                    target=manifest.component_id,
                    outcome="denied",
                    reason=policy_decision.reason,
                    policy_version=self.policy.version,
                )
                self.lifecycle.apply(
                    manifest.component_id,
                    LifecycleState.FAILED,
                    actor="policy",
                    reason=policy_decision.reason,
                    policy_version=self.policy.version,
                )
                return RegistrationResult(
                    component_id=manifest.component_id,
                    registered=False,
                    reason=policy_decision.reason,
                )
            self.lifecycle.apply(
                manifest.component_id,
                LifecycleState.AUTHORIZED,
                actor="policy",
                reason="registration policy allowed",
                policy_version=self.policy.version,
            )
            grants = self.authority.issue_component_grants(
                manifest, ttl_seconds=max(60.0, manifest.health.lease_ttl_seconds * 4)
            )
            self._grants[manifest.component_id] = grants
            for right, capability in grants.items():
                self._record(
                    "capability_issue",
                    target=manifest.component_id,
                    outcome="ok",
                    right=right,
                    capability_id=capability.capability_id,
                    expires_at=capability.expires_at_epoch,
                )
            self.lifecycle.apply(
                manifest.component_id,
                LifecycleState.READY,
                actor="registry",
                reason="registered, authorized, and ready",
                policy_version=self.policy.version,
            )
            self._graph = None  # rebuild lazily from current registry
            self._record(
                "registration",
                target=manifest.component_id,
                outcome="ok",
                digest=result.manifest_digest,
                generation=result.generation,
            )
            return result

    def issue(
        self,
        subject: str,
        right: str,
        *,
        constraints: Mapping[str, Any] | None = None,
        ttl_seconds: float = 3600.0,
        resource: str | None = None,
        task_id: str | None = None,
    ) -> Any:
        """Issue an additional scoped grant (operator/service path)."""
        if resource is None:
            from .capabilities import DEFAULT_RIGHT_SCOPES

            resource = DEFAULT_RIGHT_SCOPES.get(right, "runtime")
        capability = self.authority.issue(
            subject,
            resource,
            right,
            constraints=constraints,
            ttl_seconds=ttl_seconds,
            task_id=task_id,
        )
        self._record(
            "capability_issue",
            target=subject,
            outcome="ok",
            right=right,
            capability_id=capability.capability_id,
        )
        return capability

    def authorize(self, capability: Any, request: AccessRequest, now: float | None = None) -> Any:
        decision = self.authority.authorize(capability, request, now=now)
        if not decision.allowed:
            self._record(
                "authorization_denied",
                actor=request.subject,
                target=request.resource,
                outcome="denied",
                right=request.right,
                reason=decision.reason,
                policy_version=decision.policy_version,
            )
        return decision

    # -- plane 2: execution (steps 5, 9, 38) -----------------------------------

    def start(
        self,
        component_id: str,
        profile: Mapping[str, Any] | None = None,
    ) -> RuntimeHandle:
        """Reserve resources and dispatch through the chosen tier (step 38)."""
        profile = dict(profile or {})
        with self._lock:
            manifest = self.registry.manifest(component_id)
            state = self.lifecycle.state(component_id)
            if state not in (LifecycleState.READY, LifecycleState.RUNNING):
                raise IllegalTransitionError(
                    f"component {component_id} is {state!r}, not READY; dispatch refused",
                    from_state=state,
                    to_state=LifecycleState.RUNNING.value,
                )
            task_id = str(profile.get("task_id", f"task-{component_id}"))
            request_vector = manifest.resources.requested
            policy_decision = self.policy.evaluate(
                component_id,
                "dispatch",
                {
                    "trust_class": manifest.trust_class.value,
                    "consequence": manifest.consequence.value,
                    "tier": manifest.isolation.tier.value,
                    "deadline_seconds": manifest.resources.deadline_seconds,
                    "audit_attached": True,
                },
            )
            dependencies_ready = self._dependencies_ready(component_id)
            reservation = self.resources.reserve(
                task_id,
                request_vector,
                owner=component_id,
                limits=manifest.resources.limits,
                deadline_monotonic=time.monotonic() + manifest.resources.deadline_seconds,
                policy_ok=policy_decision.allowed,
                dependencies_ready=dependencies_ready,
            )
            if not reservation.granted:
                self._record(
                    "authorization_denied",
                    actor=component_id,
                    target="resource_plane",
                    task_id=task_id,
                    outcome="denied",
                    reason=reservation.reason,
                    policy_version=self.policy.version,
                )
                raise AdmissionDeniedError(reservation.reason)
            self._record(
                "reservation",
                target=component_id,
                task_id=task_id,
                outcome="ok",
                reservation_id=reservation.reservation.reservation_id
                if reservation.reservation
                else "",
                amounts=request_vector.to_dict(),
                owner=component_id,
            )
            backend = self.backends.get(manifest.isolation.tier.value)
            if backend is None or not backend.available():
                self.resources.release(
                    reservation.reservation.reservation_id, "backend unavailable"
                )
                raise BackendUnavailableError(
                    f"execution tier {manifest.isolation.tier.value} is unavailable "
                    "on this host; isolation never silently degrades"
                )
            deadline = time.monotonic() + manifest.resources.deadline_seconds
            execution = backend.run(
                manifest,
                profile.get("payload", {}),
                budget_seconds=manifest.resources.deadline_seconds,
                deadline_monotonic=deadline,
                capability=self._grants.get(component_id, {}).get("task.execute"),
            )
            handle = RuntimeHandle(self, execution, reservation.reservation.reservation_id)
            self._handles[execution.handle_id] = handle
            if state == LifecycleState.READY:
                self.lifecycle.apply(
                    component_id,
                    LifecycleState.RUNNING,
                    actor="dispatch",
                    reason=f"dispatched to tier {manifest.isolation.tier.value}",
                    policy_version=self.policy.version,
                )
            if execution.state == ExecutionState.FINISHED:
                self.supervisor.report_success(component_id)
            elif execution.state in (ExecutionState.FAILED, ExecutionState.TIMED_OUT):
                self._on_execution_failure(component_id, execution)
            self._record(
                "dispatch",
                target=component_id,
                task_id=task_id,
                outcome="ok",
                handle_id=execution.handle_id,
                tier=manifest.isolation.tier.value,
                trace_id=str(profile.get("trace_id", "")),
            )
            return handle

    def stop(
        self,
        handle: RuntimeHandle,
        *,
        reason: str = "operator stop",
        deadline_seconds: float = 5.0,
    ) -> StopResult:
        """Quiesce, stop, and release every owned reservation (step 13)."""
        with self._lock:
            component_id = handle.component_id
            state = self.lifecycle.state(component_id)
            if state == LifecycleState.RUNNING:
                self.lifecycle.apply(
                    component_id,
                    LifecycleState.QUIESCING,
                    actor="stop",
                    reason=reason,
                    policy_version=self.policy.version,
                )
            started = time.monotonic()
            if handle.execution.state == ExecutionState.RUNNING:
                handle.execution.cancel(reason)
            outcome = OutcomeKind.OK
            if handle.execution.state == ExecutionState.TIMED_OUT:
                outcome = OutcomeKind.TIMED_OUT
            elif handle.execution.state == ExecutionState.FAILED:
                outcome = OutcomeKind.INTERNAL
            elif handle.execution.state == ExecutionState.CANCELLED:
                outcome = OutcomeKind.CANCELLED
            released = self.resources.release(handle.reservation_id, reason=f"stop: {reason}")
            released_count = 1 if released.released else 0
            if released.released:
                self._record(
                    "reservation_release",
                    target=component_id,
                    outcome="ok",
                    reservation_id=handle.reservation_id,
                )
            if state in (LifecycleState.RUNNING, LifecycleState.QUIESCING):
                self.lifecycle.apply(
                    component_id,
                    LifecycleState.STOPPED,
                    actor="stop",
                    reason=reason,
                    policy_version=self.policy.version,
                )
            if handle.execution.state == ExecutionState.FINISHED:
                self.supervisor.report_success(component_id)
            elapsed = time.monotonic() - started
            result = StopResult(
                component_id,
                outcome,
                reason,
                reservations_released=released_count,
                elapsed_seconds=elapsed,
            )
            self._record(
                "cancellation" if outcome == OutcomeKind.CANCELLED else "restart",
                target=component_id,
                outcome=outcome.value,
                reason=reason,
            )
            return result

    def quarantine(self, component_id: str, reason: str, *, preserve_artifacts: bool = True) -> dict[str, Any]:
        """Revoke grants, freeze dispatch, isolate, preserve evidence (step 22)."""
        with self._lock:
            generation = self.authority.revoke_subject(component_id)
            self.registry.expire_leases()
            self._record(
                "capability_revoke",
                target=component_id,
                outcome="ok",
                generation=generation,
                reason=reason,
            )
            state = self.lifecycle.state(component_id)
            if state is not None and state != LifecycleState.QUARANTINED:
                self.lifecycle.apply(
                    component_id,
                    LifecycleState.QUARANTINED,
                    actor="quarantine",
                    reason=reason,
                    policy_version=self.policy.version,
                )
            # Release ghost reservations but never logs or diagnostics.
            active_tasks = {
                handle.execution.component_id for handle in self._handles.values()
            }
            recovered = self.resources.reconcile(
                active_tasks, reason=f"quarantine of {component_id}"
            )
            self._record(
                "quarantine",
                target=component_id,
                outcome="ok",
                reason=reason,
                recovered_reservations=len(recovered),
                evidence_preserved=True,
            )
            return {
                "component_id": component_id,
                "generation": generation,
                "recovered_reservations": len(recovered),
                "diagnostics_preserved": True,
                "note": "logs and failure cause are never deleted by quarantine",
            }

    def _on_supervisor_quarantine(self, component_id: str, reason: str) -> None:
        self.quarantine(component_id, reason)

    def _on_execution_failure(self, component_id: str, execution: ExecutionHandle) -> None:
        self._record(
            "crash" if execution.state == ExecutionState.FAILED else "timeout",
            target=component_id,
            outcome=execution.state,
            reason=str(execution.exit_info)[:300],
        )
        report = self.supervisor.report_failure(
            component_id,
            reason=f"execution {execution.state}",
        )
        if report.decision == "quarantine":
            self.quarantine(component_id, report.reason)
        else:
            self._record(
                "restart",
                target=component_id,
                outcome="retry",
                attempt=report.attempt,
                delay_seconds=report.delay_seconds,
                reason=report.reason,
            )

    # -- plane 3: communication ------------------------------------------------

    def register_handler(self, recipient: str, handler: Any) -> None:
        """Route IPC messages for a component to its handler (step 10)."""
        self.bus.register_handler(recipient, handler)

    def request(self, envelope: MessageEnvelope, *, cancellation: Any | None = None) -> Any:
        """Send one typed request through the bounded bus (steps 10-13)."""
        return self.bus.request(envelope, cancellation=cancellation)

    # -- plane 4/5: artifacts, workspace, USG ----------------------------------

    def publish_artifact(
        self,
        data: bytes,
        metadata: Mapping[str, Any] | None = None,
        capability: Any | None = None,
        *,
        subject: str = "",
        task_id: str = "",
    ) -> Any:
        ref = self.artifacts.publish(
            data,
            metadata,
            capability,
            self.authority.authorize,
            subject=subject or getattr(capability, "subject", ""),
            task_id=task_id,
        )
        self._record(
            "artifact_published",
            target=ref.artifact_id,
            task_id=task_id,
            outcome="ok",
            producer=ref.producer,
            size_bytes=ref.size_bytes,
        )
        return ref

    def read_artifact(self, ref: Any, capability: Any | None = None, *, subject: str = "") -> bytes:
        data = self.artifacts.open(
            ref, capability, self.authority.authorize, subject=subject or getattr(capability, "subject", "")
        )
        self._record(
            "artifact_read",
            target=ref.artifact_id,
            outcome="ok",
            subject=subject or getattr(capability, "subject", ""),
        )
        return data

    def issue_lease(self, component_id: str, ttl_seconds: float | None = None) -> ServiceLease:
        manifest = self.registry.manifest(component_id)
        ttl = ttl_seconds or manifest.health.lease_ttl_seconds
        lease = self.registry.issue_lease(component_id, ttl)
        self._record(
            "health_lease",
            target=component_id,
            outcome="ok",
            lease_id=lease.lease_id,
            expires_at_epoch=lease.expires_at_epoch,
        )
        return lease

    # -- plane 6: observability and governance ----------------------------------

    def health(self, component_id: str) -> dict[str, Any]:
        """Distinguish process liveness, responsiveness, capability (step 34)."""
        registry_health = self.registry.health(component_id)
        execution_states = [
            handle.execution.state
            for handle in self._handles.values()
            if handle.component_id == component_id
        ]
        process_alive = bool(execution_states) and any(
            state in (ExecutionState.FINISHED, ExecutionState.RUNNING)
            for state in execution_states
        ) or not execution_states
        service_responsive = registry_health["lease_active"]
        task_capable = (
            self.lifecycle.state(component_id) in (LifecycleState.READY, LifecycleState.RUNNING)
            and self.resources.available().cpu_shares > 0
        )
        return {
            **registry_health,
            "process_alive": process_alive,
            "service_responsive": service_responsive,
            "task_capable": task_capable,
        }

    def dependency_graph(self) -> DependencyGraph:
        if self._graph is None:
            self._graph = DependencyGraph(
                self.registry.manifest(component_id)
                for component_id in self.registry.list_components()
            )
        return self._graph

    def explain(self, component_id: str | None = None, task_id: str = "") -> dict[str, Any]:
        """Governance explanation (step 52, section 11)."""
        target = component_id or "runtime"
        lifecycle_history = [
            record.to_dict()
            for record in self.lifecycle.history(component_id)
        ] if component_id else []
        grants = (
            {
                right: capability.capability_id
                for right, capability in self._grants.get(component_id, {}).items()
            }
            if component_id
            else {}
        )
        return {
            "component": target,
            "task_id": task_id,
            "lifecycle": self.lifecycle.state(component_id) if component_id else "n/a",
            "lifecycle_history": lifecycle_history,
            "authority": {
                "grants": grants,
                "generation": self.authority.generation_of(component_id) if component_id else 0,
            },
            "budget": self.resources.describe(),
            "policy": {
                "version": self.policy.version,
                "checksum": self.policy.checksum(),
            },
            "ipc": self.bus.statistics_snapshot(),
            "supervisor": self.supervisor.describe(),
            "audit_events": self.audit.count(),
        }

    def shutdown(self) -> dict[str, Any]:
        """Release everything the runtime owns (graceful shutdown)."""
        with self._lock:
            for handle in list(self._handles.values()):
                if handle.execution.state == ExecutionState.RUNNING:
                    handle.execution.cancel("runtime shutdown")
            for reservation in self.resources.active_reservations():
                self.resources.release(
                    reservation.reservation_id, reason="runtime shutdown"
                )
            audit_events = self.audit.count()
            self.audit.close()
            return {
                "reservations_released": True,
                "audit_events": audit_events,
            }

    def __enter__(self) -> "ComponentRuntime":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        self.shutdown()
        return False

    # -- internal wiring ---------------------------------------------------------

    def _resolve_capability(self, capability_id: str) -> Any:
        for grants in self._grants.values():
            for capability in grants.values():
                if capability.capability_id == capability_id:
                    return capability
        return _UnknownCapability(capability_id)

    def _dependencies_ready(self, component_id: str) -> bool:
        try:
            graph = self.dependency_graph()
        except Exception:
            return True
        ready = {
            cid
            for cid in self.registry.list_components()
            if self.lifecycle.state(cid)
            in (LifecycleState.READY, LifecycleState.RUNNING, LifecycleState.STOPPED)
        }
        return graph.dependencies_ready(component_id, ready)

    def _audit_ipc_event(self, kind: str, metadata: Mapping[str, Any]) -> None:
        mapped = {
            "queue_backpressure": "queue_backpressure",
            "authorization_denied": "authorization_denied",
            "ipc_pause": "ipc_pause",
        }.get(kind, "authorization_denied")
        self._record(mapped, outcome="denied", **dict(metadata))

    def _audit_policy_event(self, kind: str, metadata: Mapping[str, Any]) -> None:
        self._record(kind, outcome=str(metadata.get("outcome", "ok")), **dict(metadata))

    def _record(self, kind: str, **metadata: Any) -> None:
        actor = str(metadata.pop("actor", "system"))
        target = str(metadata.pop("target", ""))
        task_id = str(metadata.pop("task_id", ""))
        self.audit.record(
            AuditEvent(
                kind=kind,
                actor=actor,
                target=target,
                task_id=task_id,
                policy_version=self.policy.version,
                outcome=str(metadata.pop("outcome", "ok")),
                metadata=metadata,
            )
        )


class _UnknownCapability:
    """Stand-in for unresolvable capability ids: always denied."""

    def __init__(self, capability_id: str) -> None:
        self.capability_id = capability_id
        self.subject = ""
        self.rights = frozenset()
        self.expires_at_epoch = 0.0
        self.generation = -1
