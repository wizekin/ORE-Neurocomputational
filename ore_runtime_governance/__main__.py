"""Command-line demo and self-test for the ORE runtime (Point 22)."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

from . import benchmark_overhead
from .contracts import (
    ComponentManifest,
    ExecutionTier,
    IsolationProfile,
    ResourceProfile,
    ResourceVector,
    TrustClass,
)
from .registry import inventory_repository
from .runtime import ComponentRuntime
from .testing import (
    ModelBasedLifecycleChecker,
    SecurityProbeSuite,
    echo_worker,
)


def _demo_components() -> list[ComponentManifest]:
    return [
        ComponentManifest(
            component_id="echo.service",
            version="1.0.0",
            owner="ore_core",
            trust_class=TrustClass.TRUSTED_SERVICE,
            purpose="Deterministic echo service for demonstration",
            entry_point="ore_runtime.testing:echo_worker",
            capabilities=("ipc.send", "task.execute", "artifact.publish", "artifact.read"),
            resources=ResourceProfile(
                requested=ResourceVector(cpu_shares=0.5, memory_bytes=64 * 1024 * 1024),
                limits=ResourceVector(cpu_shares=1.0, memory_bytes=128 * 1024 * 1024),
                deadline_seconds=10.0,
            ),
            isolation=IsolationProfile(tier=ExecutionTier.IN_PROCESS),
        ),
        ComponentManifest(
            component_id="isolated.worker",
            version="1.0.0",
            owner="ore_core",
            trust_class=TrustClass.TRUSTED_SERVICE,
            purpose="Subprocess worker demonstrating OS-level isolation",
            entry_point="ore_runtime.testing:echo_worker",
            capabilities=("task.execute",),
            resources=ResourceProfile(
                requested=ResourceVector(cpu_shares=0.5, memory_bytes=64 * 1024 * 1024),
                limits=ResourceVector(cpu_shares=1.0, memory_bytes=256 * 1024 * 1024),
                deadline_seconds=10.0,
            ),
            isolation=IsolationProfile(tier=ExecutionTier.LOCAL_PROCESS),
        ),
    ]


def run_demo(repository_root: str | None = None) -> dict:
    workdir = Path(tempfile.mkdtemp(prefix="ore-runtime-demo-"))
    runtime = ComponentRuntime(
        database_path=str(workdir / "audit.sqlite3"),
        artifact_root=str(workdir / "artifacts"),
        repository_root=repository_root,
        capacity=ResourceVector(cpu_shares=4.0, memory_bytes=1024 * 1024 * 1024),
    )
    with runtime:
        registrations = [runtime.register(manifest) for manifest in _demo_components()]
        runtime.register_handler("echo.service", lambda envelope: {"echo": dict(envelope.payload)})
        runtime.issue_lease("echo.service")

        # in-process execution
        handle = runtime.start("echo.service", profile={"payload": {"demo": True}})
        # isolated subprocess execution
        isolated = runtime.start("isolated.worker", profile={"payload": {"demo": True}})
        stop = runtime.stop(isolated)

        # capability-scoped artifact round trip
        capability = runtime.authority.issue(
            "echo.service", "artifacts/echo.service", ["artifact.publish", "artifact.read"],
            ttl_seconds=60,
        )
        ref = runtime.publish_artifact(
            b"ore runtime demo artifact", {"schema": "text/demo"}, capability,
            subject="echo.service",
        )
        payload = runtime.read_artifact(ref, capability)

        # governance explanation
        explanation = runtime.explain("echo.service", task_id="demo-task")
        return {
            "registrations": [
                {"component_id": r.component_id, "registered": r.registered}
                for r in registrations
            ],
            "in_process_state": handle.execution.state,
            "in_process_result": dict(handle.execution.result),
            "isolated_state": isolated.execution.state,
            "isolated_result": dict(isolated.execution.result),
            "stop_outcome": stop.outcome.value,
            "artifact": {
                "artifact_id": ref.artifact_id,
                "bytes": len(payload),
                "content_matches": payload == b"ore runtime demo artifact",
            },
            "audit_events": runtime.audit.count(),
            "lifecycle": runtime.lifecycle.describe(),
            "explanation_keys": sorted(explanation),
        }


def run_self_test() -> dict:
    checker = ModelBasedLifecycleChecker()
    results = {
        "lifecycle_model": checker.exhaustive_transition_check(),
        "reservations_model": checker.reservation_invariant_check(steps=120),
        "attenuation_model": checker.capability_attenuation_check(trials=24),
    }
    violations = [
        violation
        for section in results.values()
        for violation in section.get("violations", [])
    ]
    runtime = ComponentRuntime()
    with runtime:
        for manifest in _demo_components():
            runtime.register(manifest)
        runtime.register_handler("echo.service", lambda envelope: {"echo": dict(envelope.payload)})
        probes = SecurityProbeSuite(runtime).run()
        results["security_probes"] = probes.to_dict()
    results["model_violations"] = violations
    results["passed"] = not violations and probes.passed
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ORE runtime (Point 22) demo tool")
    parser.add_argument("--demo", action="store_true", help="run the scripted governance demo")
    parser.add_argument("--self-test", action="store_true", help="run model checks and probes")
    parser.add_argument("--benchmark", action="store_true", help="measure isolation overhead")
    parser.add_argument("--inventory", action="store_true", help="inventory repository components")
    parser.add_argument("--json", action="store_true", help="emit JSON only")
    parser.add_argument(
        "--repository-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="repository root for subprocess PYTHONPATH and inventory scans",
    )
    args = parser.parse_args(argv)
    if not (args.demo or args.self_test or args.benchmark or args.inventory):
        parser.print_help()
        return 0
    if args.demo:
        report = run_demo(args.repository_root)
    elif args.self_test:
        report = run_self_test()
    elif args.benchmark:
        report = benchmark_overhead(repository_root=args.repository_root).to_dict()
    else:
        report = inventory_repository(args.repository_root)
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
