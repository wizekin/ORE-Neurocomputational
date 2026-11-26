"""Verification plane: fault injection, model checks, probes, benchmarks (steps 47-50).

``FaultInjector`` kills workers, pauses the IPC plane, fills queues,
corrupts messages, exhausts quotas, and simulates database interruption
(integrated step 49). ``ModelBasedLifecycleChecker`` exhaustively explores
the lifecycle transition table and randomized reservation/attenuation
sequences -- formal-methods-lite over the abstract models with stated
assumptions (step 48). ``SecurityProbeSuite`` exercises success and denial
paths at every trust boundary (step 47). ``IsolationBenchmark`` measures
worker startup, IPC round-trips, and verifies that the in-process and
isolated backends produce equivalent outputs for deterministic reference
tasks (step 50). Simulated workers provide the fault payloads.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

from .capabilities import Capability, CapabilityAuthority, AccessRequest
from .contracts import (
    CapabilityDeniedError,
    EnvelopeValidationError,
    MessageEnvelope,
    QueueBackpressureError,
    ResourceVector,
)
from .isolation import ExecutionState, InProcessBackend, SubprocessExecutionBackend
from .lifecycle import ALLOWED_TRANSITIONS, LifecycleMachine, LifecycleState
from .resources import ResourceManager


# -- reference workers ---------------------------------------------------------


def noop_worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Deterministic reference worker used as the default entry point."""
    return {"echo": dict(payload), "worker": "ore_runtime.testing.noop_worker"}


def echo_worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Echo worker for subprocess equivalence checks (step 50)."""
    return {"echo": dict(payload), "worker": "ore_runtime.testing.echo_worker"}


def failing_worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    raise RuntimeError("simulated worker failure")


def sleep_worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Blocking worker used for deadline/timeout fault injection."""
    import time as _time

    _time.sleep(float(payload.get("seconds", 30.0)))
    return {"slept": float(payload.get("seconds", 30.0))}


# -- fault injection (integrated step 49) ---------------------------------------


class FaultInjector:
    """Deterministic fault injection against runtime components (step 49)."""

    def __init__(self, *, rng: random.Random | None = None) -> None:
        self._rng = rng or random.Random(2026)
        self.injected: list[str] = []

    def kill_worker(self, handle) -> str:
        """Terminate a running worker handle (crash simulation)."""
        handle.cancel("fault injection: worker killed")
        self.injected.append(f"kill_worker:{handle.handle_id}")
        return handle.state

    def pause_ipc(self, bus) -> str:
        bus.pause()
        self.injected.append("pause_ipc")
        return "paused"

    def resume_ipc(self, bus) -> str:
        bus.resume()
        return "resumed"

    def fill_queue(self, bus, count: int, *, sender: str = "fault.comp", recipient: str = "sink.svc") -> int:
        """Burst envelopes to trigger back-pressure (queue overload, step 49)."""
        bus.register_handler(recipient, lambda envelope: {"ok": True})
        rejected = 0
        for index in range(count):
            envelope = MessageEnvelope(
                sender=sender,
                recipient=recipient,
                payload={"burst": index},
                priority=0,
            )
            try:
                bus.request(envelope)
            except QueueBackpressureError:
                rejected += 1
            except (CapabilityDeniedError, EnvelopeValidationError):
                # Bus without wiring still accepts fill requests; these
                # exceptions mean a wired bus denied the burst early.
                rejected += 1
        self.injected.append(f"fill_queue:{count}:{rejected} rejected")
        return rejected

    def corrupt_envelope(self, envelope: MessageEnvelope, *, mode: str = "schema") -> MessageEnvelope:
        """Return a deliberately corrupted copy (corrupt messages, step 49).

        The schema mode bypasses constructor validation so the *bus's* schema
        check is the component that rejects the message.
        """
        if mode == "schema":
            corrupted = MessageEnvelope(
                sender=envelope.sender,
                recipient=envelope.recipient,
                payload=dict(envelope.payload),
            )
            object.__setattr__(corrupted, "schema_version", envelope.schema_version + 1)
            return corrupted
        if mode == "payload":
            from dataclasses import replace

            return replace(envelope, payload={"corrupted": {"not": "a", "mapping": True}})
        if mode == "oversized":
            return MessageEnvelope(
                sender=envelope.sender,
                recipient=envelope.recipient,
                payload={"blob": "x" * (1 << 20)},
            )
        raise ValueError(f"unknown corruption mode {mode!r}")

    def exhaust_quota(self, manager: ResourceManager, task_prefix: str = "fault", count: int = 8) -> int:
        """Reserve until the capacity denies further admission (step 49)."""
        denied = 0
        for index in range(count):
            result = manager.reserve(
                f"{task_prefix}-{index}", ResourceVector(cpu_shares=1.0)
            )
            if not result.granted:
                denied += 1
        self.injected.append(f"exhaust_quota:{denied} denied")
        return denied

    def revoke_mid_task(self, authority: CapabilityAuthority, capability: Capability) -> str:
        authority.revoke_subject(capability.subject)
        self.injected.append(f"revoke_mid_task:{capability.subject}")
        return "revoked"


# -- model-based checks (integrated step 48) -------------------------------------


class ModelBasedLifecycleChecker:
    """Exhaustive model checking of the lifecycle machine (step 48)."""

    def __init__(self) -> None:
        self.violations: list[str] = []

    def exhaustive_transition_check(self) -> dict[str, Any]:
        """Verify machine behaviour against the abstract transition table."""
        violations: list[str] = []
        states = LifecycleState.ALL
        for source in states:
            allowed = ALLOWED_TRANSITIONS[source]
            for target in states:
                machine = LifecycleMachine()
                machine.force_state("model.comp", source, reason="model check")
                legal = target in allowed
                try:
                    if source in LifecycleState.TERMINAL:
                        # terminal revival requires a new authorized generation
                        machine.apply(
                            "model.comp", target, actor="model", new_generation=True
                        )
                        revived = True
                        if not legal:
                            violations.append(
                                f"terminal transition {source}->{target} applied "
                                "although not in the declared table"
                            )
                    else:
                        machine.apply("model.comp", target, actor="model")
                        if not legal:
                            violations.append(f"transition {source}->{target} applied")
                except Exception:
                    if legal:
                        violations.append(f"legal transition {source}->{target} rejected")
        # terminal states never return to RUNNING without a new generation
        for terminal in LifecycleState.TERMINAL:
            machine = LifecycleMachine()
            machine.force_state("revive.comp", terminal, reason="model check")
            try:
                machine.apply("revive.comp", LifecycleState.RUNNING, actor="model")
                violations.append(
                    f"terminal {terminal} silently returned to running without "
                    "a new authorized generation"
                )
            except Exception:
                pass
        self.violations.extend(violations)
        return {
            "states_checked": len(states),
            "pairs_checked": len(states) ** 2,
            "violations": list(violations),
            "model_assumptions": (
                "abstract machine only; a proof over this model does not "
                "prove the Python implementation correct (section 7, step 48)"
            ),
        }

    def reservation_invariant_check(
        self, *, capacity: ResourceVector | None = None, steps: int = 200, seed: int = 7
    ) -> dict[str, Any]:
        """Randomized reserve/release sequences never over-allocate (step 48)."""
        rng = random.Random(seed)
        manager = ResourceManager(
            capacity or ResourceVector(cpu_shares=5, memory_bytes=100, worker_slots=3)
        )
        live: dict[str, Any] = {}
        violations: list[str] = []
        for step in range(steps):
            action = rng.choice(("reserve", "release"))
            if action == "reserve" or not live:
                task_id = f"task-{step}"
                request = ResourceVector(
                    cpu_shares=rng.choice((0.0, 0.5, 1.0)),
                    memory_bytes=rng.choice((0, 10, 30)),
                    worker_slots=rng.choice((0, 1)),
                )
                result = manager.reserve(task_id, request)
                if result.granted:
                    live[result.reservation.reservation_id] = result.reservation
            elif live:
                reservation_id = rng.choice(sorted(live))
                manager.release(reservation_id, reason="model check")
                live.pop(reservation_id, None)
            reserved = manager.reserved_total()
            capacity_now = manager.capacity()
            for kind in ResourceVector.kinds():
                if reserved.amounts()[kind] > capacity_now.amounts()[kind] + 1e-9:
                    violations.append(
                        f"step {step}: over-allocation of {kind} "
                        f"({reserved.amounts()[kind]} > {capacity_now.amounts()[kind]})"
                    )
        self.violations.extend(violations)
        return {
            "steps": steps,
            "violations": violations,
            "final_reserved": manager.reserved_total().to_dict(),
        }

    def capability_attenuation_check(self, *, trials: int = 40, seed: int = 11) -> dict[str, Any]:
        """Attenuation never grants more than the parent (section 5.3)."""
        rng = random.Random(seed)
        authority = CapabilityAuthority()
        catalog = sorted(
            {
                "ipc.send",
                "task.execute",
                "workspace.read_snapshot",
                "artifact.read",
                "artifact.publish",
            }
        )
        violations: list[str] = []
        for trial in range(trials):
            parent_rights = frozenset(rng.sample(catalog, rng.randint(1, len(catalog))))
            parent = authority.issue(
                "model.comp", "workspace", parent_rights, ttl_seconds=60
            )
            child = authority.attenuate(
                parent,
                rights=rng.sample(sorted(parent_rights), rng.randint(1, len(parent_rights))),
                resource="workspace" if rng.random() < 0.5 else "workspace/scope",
                ttl_seconds=rng.choice((1.0, 30.0, None)),
            )
            if not child.rights.issubset(parent.rights):
                violations.append(f"trial {trial}: child rights exceed parent")
            if not child.expires_at_epoch <= parent.expires_at_epoch:
                violations.append(f"trial {trial}: child expiry extends parent")
            if not (child.resource == parent.resource or child.resource.startswith(parent.resource)):
                violations.append(f"trial {trial}: child scope escapes parent")
        self.violations.extend(violations)
        return {"trials": trials, "violations": violations}


# -- security probes (integrated step 47) ------------------------------------------


@dataclass
class ProbeReport:
    """Outcome of the boundary security probes (step 47)."""

    results: dict[str, bool] = field(default_factory=dict)
    details: dict[str, str] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(self.results.values())

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "results": dict(self.results), "details": dict(self.details)}


class SecurityProbeSuite:
    """Run success and denial paths at each trust boundary (step 47)."""

    def __init__(self, runtime) -> None:
        self.runtime = runtime

    def run(self) -> ProbeReport:
        report = ProbeReport()
        for probe_name, probe in (
            ("unauthorized_workspace_write", self._probe_unauthorized_write),
            ("expired_capability", self._probe_expired_capability),
            ("revoked_generation", self._probe_revoked_generation),
            ("impersonation", self._probe_impersonation),
            ("oversized_payload", self._probe_oversized_payload),
            ("unsupported_version", self._probe_unsupported_version),
            ("child_authority_escalation", self._probe_attenuation),
        ):
            try:
                ok, detail = probe()
            except Exception as exc:
                ok, detail = False, f"probe raised {type(exc).__name__}: {exc}"
            report.results[probe_name] = ok
            report.details[probe_name] = detail
        return report

    # -- individual probes ---------------------------------------------------

    def _probe_unauthorized_write(self) -> tuple[bool, str]:
        authority = self.runtime.authority
        guard = self.runtime.blackboard_guard
        stranger = authority.issue(
            "intruder.comp", "workspace", ["workspace.append_observation"], ttl_seconds=60,
            constraints={"fields": ["lexical_hypotheses"]},
        )
        try:
            result = guard.append_observation(
                stranger,
                self.runtime.authority.authorize,
                field="lexical_hypotheses",
                payload={"k": "v"},
            )
            if result.allowed and self.runtime.blackboard_guard.backed_by_ore_workspace:
                # ore_workspace permission policy should refuse the intruder;
                # if it accepted we record a failure detail for review.
                return (
                    False,
                    "ore_workspace accepted an unauthorized producer write",
                )
            if result.allowed:
                return False, "guard accepted an unauthorized writer"
            return True, f"denied: {result.reason[:120]}"
        except CapabilityDeniedError as exc:
            return True, f"denied by authority: {exc.reason[:120]}"

    def _probe_expired_capability(self) -> tuple[bool, str]:
        authority = self.runtime.authority
        ghost = authority.issue(
            "ghost.comp", "workspace", ["workspace.read_snapshot"], ttl_seconds=0.01,
        )
        time.sleep(0.02)
        decision = authority.authorize(
            ghost,
            AccessRequest("ghost.comp", "workspace", "workspace.read_snapshot"),
        )
        return (not decision.allowed), f"expired decision: {decision.reason[:100]}"

    def _probe_revoked_generation(self) -> tuple[bool, str]:
        authority = self.runtime.authority
        cap = authority.issue("stale.comp", "workspace", ["workspace.read_snapshot"], ttl_seconds=60)
        authority.revoke_subject("stale.comp")
        decision = authority.authorize(
            cap, AccessRequest("stale.comp", "workspace", "workspace.read_snapshot")
        )
        return (not decision.allowed), f"generation decision: {decision.reason[:100]}"

    def _probe_impersonation(self) -> tuple[bool, str]:
        authority = self.runtime.authority
        real = authority.issue("real.comp", "ipc", ["ipc.send"], ttl_seconds=60)
        fake = authority.issue("fake.comp", "ipc", ["ipc.send"], ttl_seconds=60)
        decision = authority.authorize(
            fake, AccessRequest("real.comp", "ipc/echo.svc", "ipc.send")
        )
        return (not decision.allowed), f"impersonation: {decision.reason[:100]}"

    def _probe_oversized_payload(self) -> tuple[bool, str]:
        bus = self.runtime.bus
        cap = self.runtime.authority.issue(
            "probe.comp", "ipc", ["ipc.send"], ttl_seconds=60
        )
        try:
            bus.request(
                MessageEnvelope(
                    sender="probe.comp",
                    recipient="echo.svc",
                    payload={"blob": "z" * (bus.max_payload_bytes + 1)},
                    auth_capability_id=cap.capability_id,
                )
            )
            return False, "oversized payload accepted"
        except EnvelopeValidationError as exc:
            return True, f"rejected: {str(exc)[:100]}"

    def _probe_unsupported_version(self) -> tuple[bool, str]:
        try:
            MessageEnvelope(sender="probe.comp", recipient="echo.svc", schema_version=99)
            return False, "unsupported schema version accepted"
        except EnvelopeValidationError as exc:
            return True, f"rejected: {str(exc)[:100]}"

    def _probe_attenuation(self) -> tuple[bool, str]:
        authority = self.runtime.authority
        parent = authority.issue(
            "parent.comp", "workspace", ["workspace.read_snapshot", "workspace.propose_fact"],
            ttl_seconds=60,
        )
        try:
            authority.attenuate(parent, rights=["ipc.send"])
            return False, "child obtained rights beyond the parent"
        except CapabilityDeniedError as exc:
            return True, f"escalation denied: {exc.reason[:100]}"


# -- benchmarks (integrated step 50) ------------------------------------------------


@dataclass
class BenchmarkReport:
    """Isolation overhead measurements (step 50)."""

    repeats: int = 0
    in_process_ms: list[float] = field(default_factory=list)
    subprocess_ms: list[float] = field(default_factory=list)
    outputs_equivalent: bool = False
    note: str = ""

    def percentile(self, series: list[float], fraction: float) -> float:
        if not series:
            return float("nan")
        ordered = sorted(series)
        index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
        return ordered[index]

    def to_dict(self) -> dict[str, Any]:
        return {
            "repeats": self.repeats,
            "in_process_p50_ms": self.percentile(self.in_process_ms, 0.50),
            "in_process_p95_ms": self.percentile(self.in_process_ms, 0.95),
            "subprocess_p50_ms": self.percentile(self.subprocess_ms, 0.50),
            "subprocess_p95_ms": self.percentile(self.subprocess_ms, 0.95),
            "startup_overhead_ms": (
                self.percentile(self.subprocess_ms, 0.50)
                - self.percentile(self.in_process_ms, 0.50)
                if self.subprocess_ms and self.in_process_ms
                else None
            ),
            "outputs_equivalent": self.outputs_equivalent,
            "note": self.note,
        }


def benchmark_overhead(
    payload: Mapping[str, Any] | None = None,
    *,
    repeats: int = 3,
    repository_root: str | None = None,
) -> BenchmarkReport:
    """Compare in-process and subprocess execution on the reference task."""
    from .contracts import ComponentManifest

    payload = dict(payload or {"left": 21, "right": 2})
    manifest = ComponentManifest(
        component_id="bench.comp",
        entry_point="ore_runtime.testing:echo_worker",
        trust_class="trusted_service",
    )
    in_backend = InProcessBackend()
    sub_backend = SubprocessExecutionBackend(repository_root=repository_root)
    report = BenchmarkReport(repeats=repeats)
    if not sub_backend.available():
        report.note = "subprocess backend unavailable on this host; in-process only"
        return report
    equivalent = True
    for _ in range(repeats):
        started = time.perf_counter()
        handle = in_backend.run(manifest, payload, budget_seconds=30.0)
        report.in_process_ms.append((time.perf_counter() - started) * 1000.0)
        in_result = dict(handle.result)
        started = time.perf_counter()
        handle = sub_backend.run(manifest, payload, budget_seconds=30.0)
        report.subprocess_ms.append((time.perf_counter() - started) * 1000.0)
        sub_result = dict(handle.result)
        if handle.state != ExecutionState.FINISHED:
            equivalent = False
        if in_result.get("echo") != sub_result.get("echo"):
            equivalent = False
    report.outputs_equivalent = equivalent
    report.note = (
        "measured on the deterministic reference task with representative "
        "serialization; run against real ORE workloads before setting targets"
    )
    return report
