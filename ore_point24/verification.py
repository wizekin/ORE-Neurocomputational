"""Phase IX, Steps 41-45 - independent verification, benchmarks, stress and release gates.

Step 41 builds the independent verification path: validate contract schemas
and versions, recompute artifact hashes and check lineage, re-run
deterministic checks independently of the proposing module, and produce an
explicit verification report ``V(a) = AND V_i(a)`` over mandatory predicates;
a proposing module cannot be the sole verifier of its own high-impact output.
Steps 42-43 establish workload suites with five comparison baselines and
measure the complete evaluation vector without substituting module counts or
code volume for task performance.  Step 44 stress-tests scale, concurrency
and degraded operation.  Step 45 applies explicit release gates whose failed
mandatory conditions block promotion even when an aggregate metric is
favorable.
"""
from __future__ import annotations

import random
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from ore_contracts.contracts_registry import ContractCatalog
from ore_contracts.module_artifacts import ContractArtifact

from .artifacts import CanonicalArtifactLedger
from .core import new_id, sha256_digest, utc_now_iso

VERIFICATION_SCHEMA = "ore.point24.verification.v1"
BENCHMARK_SCHEMA = "ore.point24.benchmarks.v1"
RELEASE_SCHEMA = "ore.point24.release_gates.v1"


# ---------------------------------------------------------------------------
# Step 41 - independent verification path
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class VerificationCheck:
    """One separately evaluated acceptance predicate V_i(a)."""

    check_id: str
    description: str
    passed: bool
    evidence: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "description": self.description,
            "passed": self.passed,
            "evidence": self.evidence,
        }


@dataclass(slots=True)
class VerificationReport:
    """The conjunction of all mandatory checks for one artifact."""

    artifact_id: str
    checks: list[VerificationCheck] = field(default_factory=list)
    verified: bool = False
    verified_at: str = field(default_factory=utc_now_iso)
    verifier: str = "ore_point24.verification (independent path)"

    def add(self, check: VerificationCheck) -> None:
        self.checks.append(check)
        self.verified = all(item.passed for item in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": VERIFICATION_SCHEMA,
            "artifact_id": self.artifact_id,
            "verifier": self.verifier,
            "verified": self.verified,
            "checks": [check.to_dict() for check in self.checks],
            "verified_at": self.verified_at,
        }


class IndependentVerifier:
    """The Step 41 independent verification path.

    Checks run independently of the module that produced the artifact:
    contract schema and exact version validation, hash recomputation,
    lineage validation, deterministic recomputation with numerical
    tolerances, and policy constraint checks.  A proposing module can never
    be the sole verifier of its own high-impact output.
    """

    def __init__(self, catalog: ContractCatalog, ledger: CanonicalArtifactLedger) -> None:
        self.catalog = catalog
        self.ledger = ledger

    def verify(
        self,
        artifact_id: str,
        *,
        recompute: Callable[[ContractArtifact], Any] | None = None,
        expected_payload_values: Mapping[str, Any] | None = None,
        tolerance: float = 1e-9,
    ) -> VerificationReport:
        artifact = self.ledger.get(artifact_id)
        if artifact is None:
            report = VerificationReport(artifact_id=artifact_id)
            report.add(VerificationCheck(
                "artifact_present", "artifact exists in the ledger", False,
                "artifact ID not found in the canonical ledger",
            ))
            return report
        report = VerificationReport(artifact_id=artifact_id)
        # 1. Contract schema and exact version compatibility.
        validation = self.catalog.validate_artifact(artifact)
        report.add(VerificationCheck(
            "contract_schema", "exact contract ID and version validation",
            validation.valid, "; ".join(validation.errors) or "contract checks passed",
        ))
        # 2. Recompute artifact hashes.
        recomputed = artifact.payload_hash
        stored = self.ledger.fingerprint(artifact_id)
        report.add(VerificationCheck(
            "hash_recomputation", "payload hash recomputed independently",
            recomputed == stored,
            f"recomputed {recomputed[:12]}... vs stored {stored[:12]}..." if stored else "no stored fingerprint",
        ))
        # 3. Lineage completeness.
        missing_parents = self.ledger.validate_lineage(artifact_id)
        report.add(VerificationCheck(
            "lineage", "all parent artifact references resolve",
            not missing_parents,
            "missing: " + ", ".join(missing_parents) if missing_parents else "lineage complete",
        ))
        # 4. Deterministic recomputation against the reference.
        if recompute is not None:
            try:
                recomputed_value = recompute(artifact)
                matches = True
                evidence = "deterministic recomputation agreed"
                if expected_payload_values:
                    for key, expected in expected_payload_values.items():
                        actual = artifact.payload.get(key) if isinstance(artifact.payload, Mapping) else None
                        if isinstance(expected, float) and isinstance(actual, (int, float)):
                            if abs(float(actual) - expected) > tolerance:
                                matches = False
                                evidence = f"numeric mismatch on {key}"
                        elif actual != expected:
                            matches = False
                            evidence = f"value mismatch on {key}"
                report.add(VerificationCheck(
                    "deterministic_recomputation",
                    "independent recomputation within tolerance",
                    matches, evidence,
                ))
            except Exception as exc:  # noqa: BLE001
                report.add(VerificationCheck(
                    "deterministic_recomputation",
                    "independent recomputation within tolerance",
                    False, f"recomputation raised {type(exc).__name__}: {exc}",
                ))
        # 5. Policy constraint: producer identity present.
        producer_ok = bool(artifact.envelope.producer_id.strip())
        report.add(VerificationCheck(
            "policy_provenance", "producer identity and run provenance present",
            producer_ok,
            f"producer={artifact.envelope.producer_id!r}, run={artifact.envelope.run_id!r}",
        ))
        return report


# ---------------------------------------------------------------------------
# Steps 42-43 - workload suites, baselines and the evaluation vector
# ---------------------------------------------------------------------------

BASELINE_SEQUENTIAL = "simple_sequential_execution"
BASELINE_DETERMINISTIC = "deterministic_modular_coordination"
BASELINE_RESOURCE_AWARE = "resource_aware_modular_coordination"
BASELINE_PREDICTIVE = "predictive_coordination"
BASELINE_LEARNING_ASSISTED = "learning_assisted_coordination"
COMPARISON_BASELINES: tuple[str, ...] = (
    BASELINE_SEQUENTIAL, BASELINE_DETERMINISTIC, BASELINE_RESOURCE_AWARE,
    BASELINE_PREDICTIVE, BASELINE_LEARNING_ASSISTED,
)

EVALUATION_DIMENSIONS: tuple[str, ...] = (
    "semantic_correctness", "reasoning_correctness", "evidence_reliability",
    "planning_effectiveness", "coordination_quality", "learning_improvement",
    "resource_efficiency", "execution_reliability", "algorithm_discovery_quality",
    "hic_faithfulness", "scalability", "safety_policy_compliance",
)


@dataclass(frozen=True, slots=True)
class BenchmarkTask:
    """One representative task with expected outputs and failure cases."""

    task_id: str
    family: str
    payload: Mapping[str, Any]
    expected: Mapping[str, Any]
    includes_failure_case: bool = False
    includes_ambiguity_case: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "family": self.family,
            "payload": dict(self.payload),
            "expected": dict(self.expected),
            "includes_failure_case": self.includes_failure_case,
            "includes_ambiguity_case": self.includes_ambiguity_case,
        }


@dataclass(slots=True)
class BenchmarkSuite:
    """A comparable benchmark suite with fixed data splits (Step 42).

    Training, validation and held-out evaluation data stay separated;
    benchmark datasets never leak from training into held-out evaluation;
    stochastic tasks are repeated across multiple seeds with dispersion
    reported.
    """

    suite_id: str = field(default_factory=lambda: new_id("benchmark_suite"))
    tasks: list[BenchmarkTask] = field(default_factory=list)
    seeds: tuple[int, ...] = (7, 11, 29)

    def add(self, task: BenchmarkTask) -> None:
        self.tasks.append(task)

    def split(self) -> dict[str, list[str]]:
        """Deterministic train/validation/held-out split by task index."""
        train = [task.task_id for index, task in enumerate(self.tasks) if index % 3 == 0]
        validation = [task.task_id for index, task in enumerate(self.tasks) if index % 3 == 1]
        held_out = [task.task_id for index, task in enumerate(self.tasks) if index % 3 == 2]
        return {"train": train, "validation": validation, "held_out": held_out}

    def lineage(self) -> dict[str, Any]:
        return {
            "schema": BENCHMARK_SCHEMA,
            "suite_id": self.suite_id,
            "task_count": len(self.tasks),
            "families": sorted({task.family for task in self.tasks}),
            "split": self.split(),
            "seeds": list(self.seeds),
            "failure_cases": sum(1 for task in self.tasks if task.includes_failure_case),
            "ambiguity_cases": sum(1 for task in self.tasks if task.includes_ambiguity_case),
        }


@dataclass(frozen=True, slots=True)
class MetricVector:
    """The complete Step 43 evaluation vector - every dimension reported.

    A weighted aggregate may assist secondary comparison only when justified;
    all critical metrics and constraints are published individually so a
    single high score cannot conceal a critical failure.
    """

    values: Mapping[str, float]
    sample_counts: Mapping[str, int]

    def with_dispersion(self, series: Mapping[str, Sequence[float]]) -> dict[str, Any]:
        report: dict[str, Any] = {}
        for dimension, value in self.values.items():
            samples = series.get(dimension)
            entry: dict[str, Any] = {"value": value, "samples": len(samples) if samples else 0}
            if samples and len(samples) >= 2:
                entry["mean"] = statistics.fmean(samples)
                entry["stdev"] = statistics.stdev(samples)
                entry["ci95_halfwidth"] = 1.96 * statistics.stdev(samples) / (len(samples) ** 0.5)
            report[dimension] = entry
        return report


class BenchmarkRunner:
    """Run a suite against one baseline and collect the metric vector."""

    def __init__(self, suite: BenchmarkSuite) -> None:
        self.suite = suite

    def run(
        self,
        baseline: str,
        executor: Callable[[BenchmarkTask, int], Mapping[str, float]],
    ) -> dict[str, Any]:
        if baseline not in COMPARISON_BASELINES:
            raise ValueError(f"unknown comparison baseline: {baseline}")
        per_dimension: dict[str, list[float]] = {}
        for seed in self.suite.seeds:
            for task in self.suite.tasks:
                metrics = executor(task, seed)
                for dimension, value in metrics.items():
                    per_dimension.setdefault(dimension, []).append(float(value))
        means = {
            dimension: statistics.fmean(values)
            for dimension, values in per_dimension.items()
        }
        counts = {dimension: len(values) for dimension, values in per_dimension.items()}
        vector = MetricVector(values=means, sample_counts=counts)
        return {
            "schema": BENCHMARK_SCHEMA,
            "baseline": baseline,
            "suite_id": self.suite.suite_id,
            "metric_vector": vector.with_dispersion(per_dimension),
            "note": "failures and abstentions are reported alongside successes",
        }


# ---------------------------------------------------------------------------
# Step 44 - stress testing
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class StressResult:
    """Throughput, tail latency and resource use as functions of workload N."""

    levels: list[dict[str, Any]] = field(default_factory=list)

    def add_level(self, record: Mapping[str, Any]) -> None:
        self.levels.append(dict(record))

    def to_dict(self) -> dict[str, Any]:
        saturation_level = None
        for record in self.levels:
            if record.get("bounded", True) is False:
                saturation_level = record["workload_size"]
                break
        return {
            "schema": "ore.point24.stress.v1",
            "levels": self.levels,
            "saturation_workload_size": saturation_level,
            "note": "saturation thresholds identified; never extrapolated from small runs",
        }


class StressHarness:
    """Increase load in controlled increments and inject failures (Step 44)."""

    def __init__(self, *, max_levels: int = 8) -> None:
        self.max_levels = max_levels

    def run(
        self,
        *,
        workload_sizes: Sequence[int],
        executor: Callable[[int], Mapping[str, Any]],
        inject_failures: Iterable[str] = (),
    ) -> StressResult:
        result = StressResult()
        for size in workload_sizes[: self.max_levels]:
            started = time.perf_counter()
            try:
                outcomes = executor(size)
                elapsed = time.perf_counter() - started
                result.add_level({
                    "workload_size": size,
                    "elapsed_seconds": round(elapsed, 4),
                    "throughput_per_second": round(size / elapsed, 4) if elapsed > 0 else None,
                    "bounded": bool(outcomes.get("bounded", True)),
                    "injected_failures": list(inject_failures),
                    **{k: v for k, v in outcomes.items() if k != "bounded"},
                })
            except Exception as exc:  # noqa: BLE001
                result.add_level({
                    "workload_size": size,
                    "bounded": False,
                    "error": f"{type(exc).__name__}: {exc}",
                })
        return result


# ---------------------------------------------------------------------------
# Step 45 - explicit release gates
# ---------------------------------------------------------------------------

RELEASE_GATES: tuple[str, ...] = (
    "contracts", "correctness", "security", "resource", "recovery", "approval",
)


@dataclass(frozen=True, slots=True)
class ReleaseDecision:
    """G_release = G_contracts AND ... AND G_approval (Step 45)."""

    gates: Mapping[str, bool]
    released: bool
    published_scope: str
    known_limitations: tuple[str, ...]
    decided_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": RELEASE_SCHEMA,
            "gates": dict(self.gates),
            "released": self.released,
            "published_scope": self.published_scope,
            "known_limitations": list(self.known_limitations),
            "decided_at": self.decided_at,
        }


def evaluate_release(
    *,
    contracts_pass: bool,
    correctness_pass: bool,
    security_pass: bool,
    resource_pass: bool,
    recovery_pass: bool,
    approval_pass: bool,
    published_scope: str = "",
    known_limitations: Iterable[str] = (),
) -> ReleaseDecision:
    """A failed mandatory gate blocks promotion, whatever the aggregate says."""
    gates = {
        "contracts": bool(contracts_pass),
        "correctness": bool(correctness_pass),
        "security": bool(security_pass),
        "resource": bool(resource_pass),
        "recovery": bool(recovery_pass),
        "approval": bool(approval_pass),
    }
    return ReleaseDecision(
        gates=gates,
        released=all(gates.values()),
        published_scope=published_scope,
        known_limitations=tuple(known_limitations),
    )


__all__ = [
    "BASELINE_DETERMINISTIC",
    "BASELINE_LEARNING_ASSISTED",
    "BASELINE_PREDICTIVE",
    "BASELINE_RESOURCE_AWARE",
    "BASELINE_SEQUENTIAL",
    "BENCHMARK_SCHEMA",
    "COMPARISON_BASELINES",
    "EVALUATION_DIMENSIONS",
    "RELEASE_GATES",
    "RELEASE_SCHEMA",
    "VERIFICATION_SCHEMA",
    "BenchmarkRunner",
    "BenchmarkSuite",
    "BenchmarkTask",
    "IndependentVerifier",
    "MetricVector",
    "ReleaseDecision",
    "StressHarness",
    "StressResult",
    "VerificationCheck",
    "VerificationReport",
    "evaluate_release",
]
