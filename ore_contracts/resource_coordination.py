"""Adaptive cognitive resource selection layered over ORE's validated module executor.

Information gain in this first version is explicitly heuristic. The implementation
records its source and uses observed execution history for reliability and cost updates;
it does not claim calibrated predictive information gain when no such model exists.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import math
from pathlib import Path
import sqlite3
import time
from typing import Any, Mapping, Sequence

from . import ComputeBudget, OperationMode, RunContext
from .coordinator import CognitiveTask, CognitiveTaskCoordinator
from .module_logging import SQLiteModuleEventLogger, register_module_annotations
from .module_protocol import ModuleResult, ModuleStatus
from .module_registry import ModuleRegistry
from .resource_profiles import (
    CapabilityDemand,
    CoordinationResult,
    ModuleResourceEstimate,
    ResourceDecision,
    ResourceExecutionPlan,
    TaskProfile,
)


def normalized_entropy(probabilities: Sequence[float], epsilon: float = 1e-12) -> float | None:
    """Compute normalized Shannon entropy for a valid finite probability distribution."""
    values = tuple(float(item) for item in probabilities)
    if len(values) < 2:
        return None
    if any(not math.isfinite(item) or item < 0.0 or item > 1.0 for item in values):
        raise ValueError("probabilities must be finite values in [0, 1]")
    if not math.isclose(sum(values), 1.0, rel_tol=1e-7, abs_tol=1e-9):
        raise ValueError("probabilities must sum to one")
    if epsilon <= 0.0 or not math.isfinite(epsilon):
        raise ValueError("epsilon must be finite and positive")
    entropy = -sum(item * math.log(item + epsilon) for item in values)
    return min(1.0, max(0.0, entropy / math.log(len(values))))


def _utc_now() -> datetime:
    """Return the current timezone-aware UTC timestamp for persistent resource history."""
    return datetime.now(timezone.utc)


def _validate_support(support: Mapping[str, float] | None) -> dict[str, float]:
    """Normalize capability confidence values and reject invalid evidence scores."""
    normalized: dict[str, float] = {}
    for capability, raw_value in (support or {}).items():
        if not isinstance(capability, str) or not capability.strip():
            raise ValueError("support keys must be non-empty capability IDs")
        value = float(raw_value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"support for {capability!r} must be finite and in [0, 1]")
        normalized[capability] = value
    return normalized


class SQLiteResourceHistory:
    """Persist resource reliability and measured costs in the existing ORE logger database."""

    def __init__(self, database_path: str | Path) -> None:
        """Create the resource-history table inside the supplied ore_logger SQLite file."""
        self.path = Path(database_path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS resource_module_history (
                    task_family TEXT NOT NULL,
                    module_id TEXT NOT NULL,
                    successes INTEGER NOT NULL DEFAULT 0,
                    failures INTEGER NOT NULL DEFAULT 0,
                    consecutive_failures INTEGER NOT NULL DEFAULT 0,
                    average_latency_ms REAL,
                    average_memory_mb REAL,
                    average_cpu_cost REAL,
                    circuit_open_until_utc TEXT,
                    updated_at_utc TEXT NOT NULL,
                    PRIMARY KEY(task_family, module_id)
                )"""
            )

    def _connect(self) -> sqlite3.Connection:
        """Open a SQLite connection with a bounded busy timeout."""
        connection = sqlite3.connect(str(self.path), timeout=20.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=20000")
        return connection

    @contextmanager
    def _connection(self):
        """Commit or roll back one operation, then always close its connection.

        sqlite3.Connection's built-in context manager manages transactions but
        does not close the connection. Explicit closure is essential on Windows,
        where an open SQLite handle prevents deletion of the temporary database.
        """
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def get(self, task_family: str, module_id: str) -> dict[str, Any]:
        """Return stored observations or neutral defaults when a module has no history."""
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM resource_module_history WHERE task_family=? AND module_id=?",
                (task_family, module_id),
            ).fetchone()
        if row is None:
            return {
                "successes": 0, "failures": 0, "consecutive_failures": 0,
                "average_latency_ms": None, "average_memory_mb": None,
                "average_cpu_cost": None, "circuit_open_until_utc": None,
            }
        return dict(row)

    def record_outcome(
        self,
        task_family: str,
        module_id: str,
        success: bool,
        *,
        circuit_breaker_threshold: int,
        circuit_breaker_cooldown_seconds: float,
    ) -> dict[str, Any]:
        """Update Beta-Bernoulli counts and open a timed circuit after repeated failures."""
        current = self.get(task_family, module_id)
        successes = int(current["successes"]) + int(success)
        failures = int(current["failures"]) + int(not success)
        consecutive = 0 if success else int(current["consecutive_failures"]) + 1
        opened_until = current["circuit_open_until_utc"]
        if not success and consecutive >= circuit_breaker_threshold:
            opened_until = (
                _utc_now() + timedelta(seconds=circuit_breaker_cooldown_seconds)
            ).isoformat()
        elif success:
            opened_until = None
        with self._connection() as connection:
            connection.execute(
                """INSERT INTO resource_module_history
                   (task_family,module_id,successes,failures,consecutive_failures,
                    circuit_open_until_utc,updated_at_utc)
                   VALUES (?,?,?,?,?,?,?)
                   ON CONFLICT(task_family,module_id) DO UPDATE SET
                    successes=excluded.successes, failures=excluded.failures,
                    consecutive_failures=excluded.consecutive_failures,
                    circuit_open_until_utc=excluded.circuit_open_until_utc,
                    updated_at_utc=excluded.updated_at_utc""",
                (
                    task_family, module_id, successes, failures, consecutive,
                    opened_until, _utc_now().isoformat(),
                ),
            )
        return self.get(task_family, module_id)

    def record_measurements(
        self,
        task_family: str,
        module_id: str,
        *,
        latency_ms: float | None = None,
        memory_mb: float | None = None,
        cpu_cost: float | None = None,
        alpha: float = 0.2,
    ) -> dict[str, Any]:
        """Update measured costs by exponential moving average without inventing missing metrics."""
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        measurements = (latency_ms, memory_mb, cpu_cost)
        if any(
            value is not None and (not math.isfinite(float(value)) or value < 0.0)
            for value in measurements
        ):
            raise ValueError("measured costs must be finite and non-negative")
        current = self.get(task_family, module_id)
        fields = ("average_latency_ms", "average_memory_mb", "average_cpu_cost")
        updated = []
        for previous, measured in zip(fields, measurements):
            old = current[previous]
            if measured is None:
                updated.append(old)
            elif old is None:
                updated.append(float(measured))
            else:
                updated.append(alpha * float(measured) + (1.0 - alpha) * float(old))
        with self._connection() as connection:
            connection.execute(
                """INSERT INTO resource_module_history
                   (task_family,module_id,average_latency_ms,average_memory_mb,
                    average_cpu_cost,updated_at_utc)
                   VALUES (?,?,?,?,?,?)
                   ON CONFLICT(task_family,module_id) DO UPDATE SET
                    average_latency_ms=COALESCE(
                        excluded.average_latency_ms,resource_module_history.average_latency_ms
                    ),
                    average_memory_mb=COALESCE(
                        excluded.average_memory_mb,resource_module_history.average_memory_mb
                    ),
                    average_cpu_cost=COALESCE(
                        excluded.average_cpu_cost,resource_module_history.average_cpu_cost
                    ),
                    updated_at_utc=excluded.updated_at_utc""",
                (task_family, module_id, *updated, _utc_now().isoformat()),
            )
        return self.get(task_family, module_id)


class ModuleReliabilityTracker:
    """Estimate module reliability with a Beta(2,2) prior and protect against repeated failures."""

    def __init__(
        self,
        history: SQLiteResourceHistory,
        *,
        prior_alpha: float = 2.0,
        prior_beta: float = 2.0,
        circuit_breaker_threshold: int = 3,
        circuit_breaker_cooldown_seconds: float = 60.0,
    ) -> None:
        """Connect reliability estimates to persistent history and configure failure protection."""
        if prior_alpha <= 0 or prior_beta <= 0:
            raise ValueError("Beta prior parameters must be positive")
        if circuit_breaker_threshold < 1 or circuit_breaker_cooldown_seconds <= 0:
            raise ValueError("circuit-breaker threshold and cooldown must be positive")
        self.history = history
        self.prior_alpha = float(prior_alpha)
        self.prior_beta = float(prior_beta)
        self.circuit_breaker_threshold = int(circuit_breaker_threshold)
        self.circuit_breaker_cooldown_seconds = float(circuit_breaker_cooldown_seconds)

    def probability(self, task_family: str, module_id: str) -> float:
        """Return the Beta posterior mean for successful module execution."""
        row = self.history.get(task_family, module_id)
        alpha = self.prior_alpha + int(row["successes"])
        beta = self.prior_beta + int(row["failures"])
        return alpha / (alpha + beta)

    def failure_penalty(self, task_family: str, module_id: str) -> float:
        """Return the posterior expected failure rate for the selection score."""
        return 1.0 - self.probability(task_family, module_id)

    def is_circuit_open(self, task_family: str, module_id: str) -> bool:
        """Check whether the module's persisted circuit-breaker cooldown is still active."""
        row = self.history.get(task_family, module_id)
        value = row.get("circuit_open_until_utc")
        if not value:
            return False
        until = datetime.fromisoformat(value)
        if until.tzinfo is None:
            until = until.replace(tzinfo=timezone.utc)
        return until > _utc_now()

    def record_outcome(self, task_family: str, module_id: str, success: bool) -> None:
        """Persist one observed success or failure and update the circuit-breaker state."""
        self.history.record_outcome(
            task_family, module_id, success,
            circuit_breaker_threshold=self.circuit_breaker_threshold,
            circuit_breaker_cooldown_seconds=self.circuit_breaker_cooldown_seconds,
        )


class ModuleResourceEstimator:
    """Combine declared assumptions and measured moving averages into transparent cost estimates."""

    def __init__(
        self,
        history: SQLiteResourceHistory,
        reliability: ModuleReliabilityTracker,
        declared_estimates: Mapping[str, ModuleResourceEstimate] | None = None,
        *,
        moving_average_alpha: float = 0.2,
    ) -> None:
        """Configure cost history, reliability estimates, and the EMA learning rate."""
        if not 0.0 < moving_average_alpha <= 1.0:
            raise ValueError("moving_average_alpha must be in (0, 1]")
        self.history = history
        self.reliability = reliability
        self.declared_estimates = dict(declared_estimates or {})
        self.moving_average_alpha = moving_average_alpha

    def estimate(
        self,
        task_family: str,
        module_id: str,
        expected_information_gain: float,
    ) -> ModuleResourceEstimate:
        """Estimate costs from measurements, declarations, or labelled engineering assumptions."""
        prior = self.declared_estimates.get(module_id)
        row = self.history.get(task_family, module_id)
        latency = row["average_latency_ms"]
        memory = row["average_memory_mb"]
        cpu = row["average_cpu_cost"]
        sources = []
        if latency is None:
            latency = prior.expected_latency_ms if prior else 1000.0
            sources.append("declared" if prior else "engineering_assumption")
        else:
            sources.append("measured_history")
        if memory is None:
            memory = prior.expected_memory_mb if prior else 128.0
            sources.append("declared" if prior else "engineering_assumption")
        else:
            sources.append("measured_history")
        if cpu is None:
            cpu = prior.expected_cpu_cost if prior else 1.0
            sources.append("declared" if prior else "engineering_assumption")
        else:
            sources.append("measured_history")
        probability = self.reliability.probability(task_family, module_id)
        if int(row["successes"]) + int(row["failures"]) == 0 and prior is not None:
            probability = prior.estimated_success_probability
        if expected_information_gain < 0.0 or expected_information_gain > 1.0:
            raise ValueError("expected_information_gain must be in [0, 1]")
        return ModuleResourceEstimate(
            module_id=module_id,
            expected_latency_ms=float(latency),
            expected_memory_mb=float(memory),
            expected_cpu_cost=float(cpu),
            expected_information_gain=float(expected_information_gain),
            estimated_success_probability=float(probability),
            estimate_source="+".join(sorted(set(sources))),
        )

    def record_measurement(
        self,
        task_family: str,
        module_id: str,
        *,
        latency_ms: float | None = None,
        memory_mb: float | None = None,
        cpu_cost: float | None = None,
    ) -> None:
        """Persist only metrics actually returned by the execution trace or backend metadata."""
        self.history.record_measurements(
            task_family, module_id, latency_ms=latency_ms, memory_mb=memory_mb,
            cpu_cost=cpu_cost, alpha=self.moving_average_alpha,
        )


class InformationGainEstimator:
    """Estimate first-version information gain from capability relevance and explicit belief entropy."""

    def uncertainty_profile(
        self,
        demands: Sequence[CapabilityDemand],
        support: Mapping[str, float],
        belief_distributions: Mapping[str, Sequence[float]] | None = None,
    ) -> dict[str, float]:
        """Calculate per-capability deficits, optionally raising them using belief entropy."""
        distributions = belief_distributions or {}
        output: dict[str, float] = {}
        for demand in demands:
            current = support.get(demand.capability_id, 0.0)
            deficit = max(0.0, demand.target_confidence - current)
            uncertainty = (
                deficit / max(demand.target_confidence, 1e-12)
                if demand.target_confidence else 0.0
            )
            probabilities = distributions.get(demand.capability_id)
            if probabilities is not None:
                entropy = normalized_entropy(probabilities)
                if entropy is not None:
                    uncertainty = max(uncertainty, entropy)
            output[demand.capability_id] = min(1.0, max(0.0, uncertainty))
        return output

    def estimate(
        self,
        module_capabilities: Sequence[str],
        demands: Sequence[CapabilityDemand],
        uncertainty: Mapping[str, float],
    ) -> tuple[float, float]:
        """Return heuristic information gain and capability coverage, not calibrated EIG."""
        relevant = [item for item in demands if item.capability_id in module_capabilities]
        total_weight = sum(item.priority for item in demands)
        if not relevant or total_weight <= 0.0:
            return 0.0, 0.0
        weighted_uncertainty = sum(
            item.priority * uncertainty.get(item.capability_id, 0.0) for item in relevant
        )
        relevance = sum(item.priority for item in relevant) / max(total_weight, 1e-12)
        average_uncertainty = weighted_uncertainty / max(
            sum(item.priority for item in relevant), 1e-12,
        )
        gain = min(1.0, max(0.0, relevance * average_uncertainty))
        return gain, min(1.0, max(0.0, relevance))


class ResourceScorer:
    """Apply expected utility with normalized resource cost and failure penalties."""

    def __init__(
        self,
        *,
        information_gain_weight: float = 0.5,
        failure_penalty_weight: float = 0.5,
        epsilon: float = 1e-6,
        latency_budget_ms: float = 5000.0,
        memory_budget_mb: float = 512.0,
        cpu_budget_units: float = 10.0,
    ) -> None:
        """Configure score weights, numeric stability, and default normalization budgets."""
        values = (
            information_gain_weight, failure_penalty_weight, epsilon,
            latency_budget_ms, memory_budget_mb, cpu_budget_units,
        )
        if any(not math.isfinite(float(value)) or value < 0.0 for value in values):
            raise ValueError("scoring weights and budgets must be finite and non-negative")
        if epsilon <= 0 or latency_budget_ms <= 0 or memory_budget_mb <= 0 or cpu_budget_units <= 0:
            raise ValueError("epsilon and normalization budgets must be positive")
        self.information_gain_weight = information_gain_weight
        self.failure_penalty_weight = failure_penalty_weight
        self.epsilon = epsilon
        self.latency_budget_ms = latency_budget_ms
        self.memory_budget_mb = memory_budget_mb
        self.cpu_budget_units = cpu_budget_units

    def normalized_cost(self, estimate: ModuleResourceEstimate, budget: ComputeBudget | None) -> float:
        """Compute a dimensionless weighted cost ratio from latency, memory, and CPU estimates."""
        latency_budget = (
            budget.wall_clock_seconds * 1000.0
            if budget and budget.wall_clock_seconds is not None else self.latency_budget_ms
        )
        memory_budget = (
            budget.max_memory_mb if budget and budget.max_memory_mb is not None else self.memory_budget_mb
        )
        latency_ratio = estimate.expected_latency_ms / latency_budget
        memory_ratio = estimate.expected_memory_mb / memory_budget
        cpu_ratio = estimate.expected_cpu_cost / self.cpu_budget_units
        return 0.5 * latency_ratio + 0.3 * memory_ratio + 0.2 * cpu_ratio

    def score(
        self,
        capability_gain: float,
        estimate: ModuleResourceEstimate,
        failure_penalty: float,
        budget: ComputeBudget | None,
    ) -> tuple[float, float]:
        """Calculate Score = P(G + lambda_I I)/(epsilon + C) - lambda_F F."""
        values = (capability_gain, failure_penalty)
        if any(not math.isfinite(float(value)) or not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("capability gain and failure penalty must be in [0, 1]")
        cost = self.normalized_cost(estimate, budget)
        utility = estimate.estimated_success_probability * (
            capability_gain + self.information_gain_weight * estimate.expected_information_gain
        )
        score = utility / (self.epsilon + cost) - self.failure_penalty_weight * failure_penalty
        if not math.isfinite(score):
            raise ArithmeticError("resource selection score is not finite")
        return score, cost


class ExecutionPlanBuilder:
    """Greedily select runnable modules, close dependencies, and topologically order the result."""

    def __init__(
        self,
        registry: ModuleRegistry,
        resource_estimator: ModuleResourceEstimator,
        information_gain_estimator: InformationGainEstimator | None = None,
        scorer: ResourceScorer | None = None,
    ) -> None:
        """Bind planning to the registry and configured numerical estimators."""
        self.registry = registry
        self.resource_estimator = resource_estimator
        self.information_gain_estimator = information_gain_estimator or InformationGainEstimator()
        self.scorer = scorer or ResourceScorer()

    def _dependency_closure(
        self,
        module_id: str,
        already_executed: set[str],
    ) -> tuple[set[str], str | None]:
        """Resolve runnable required dependencies recursively and reject cycles."""
        selected: set[str] = set()
        visiting: set[str] = set()

        def visit(current_id: str) -> str | None:
            """Walk one dependency branch and collect each runnable module once."""
            if current_id in already_executed or current_id in selected:
                return None
            if current_id in visiting:
                return f"dependency cycle includes {current_id}"
            manifest = self.registry.get_manifest(current_id)
            if manifest is None:
                return f"required dependency is not registered: {current_id}"
            availability = self.registry.diagnostics(current_id)
            if (
                not availability.get("backend_validated")
                or availability.get("disabled")
                or not availability.get("runnable")
            ):
                return f"required dependency is not runnable: {current_id}"
            visiting.add(current_id)
            for dependency in manifest.dependencies:
                if dependency.required:
                    error = visit(dependency.module_id)
                    if error:
                        return error
            visiting.remove(current_id)
            selected.add(current_id)
            return None

        error = visit(module_id)
        return selected, error

    @staticmethod
    def _topological_order(selected: set[str], registry: ModuleRegistry) -> tuple[str, ...]:
        """Order selected modules with required dependencies first using deterministic Kahn sorting."""
        remaining = set(selected)
        ordered: list[str] = []
        while remaining:
            ready = sorted(
                module_id for module_id in remaining
                if not any(
                    dependency.required and dependency.module_id in remaining
                    for dependency in registry.get_manifest(module_id).dependencies
                )
            )
            if not ready:
                raise ValueError("selected module dependencies contain a cycle")
            ordered.extend(ready)
            remaining.difference_update(ready)
        return tuple(ordered)

    def build(
        self,
        profile: TaskProfile,
        current_support: Mapping[str, float] | None = None,
        *,
        belief_distributions: Mapping[str, Sequence[float]] | None = None,
        already_executed: Sequence[str] = (),
        remaining_module_calls: int | None = None,
        budget: ComputeBudget | None = None,
    ) -> ResourceExecutionPlan:
        """Discover, score, greedily select, and dependency-order the best remaining module set."""
        if not isinstance(profile, TaskProfile):
            raise TypeError("profile must be a TaskProfile")
        support = _validate_support(current_support)
        executed = set(already_executed)
        call_limit = (
            profile.max_module_calls if remaining_module_calls is None
            else max(0, remaining_module_calls)
        )
        uncertainty = self.information_gain_estimator.uncertainty_profile(
            profile.demands, support, belief_distributions,
        )
        mandatory = [item for item in profile.demands if item.mandatory]
        mandatory_uncertainty = max(
            (uncertainty[item.capability_id] for item in mandatory), default=0.0,
        )
        recruit_optional = bool(profile.metadata.get("recruit_optional", False)) or (
            mandatory_uncertainty > 1.0 - profile.stop_confidence
        )
        active_demands = [
            item for item in profile.demands
            if item.target_confidence > support.get(item.capability_id, 0.0)
            and (item.mandatory or recruit_optional)
        ]
        total_weighted_deficit = sum(
            item.priority * max(0.0, item.target_confidence - support.get(item.capability_id, 0.0))
            for item in active_demands
        )
        manifests = {item.identity.module_id: item for item in self.registry.list_manifests()}
        providers: dict[str, set[str]] = {item.capability_id: set() for item in active_demands}
        blocked: set[str] = set()
        decisions: dict[str, ResourceDecision] = {}
        for demand in active_demands:
            runnable_records = {
                record.module_id: record
                for record in self.registry.find_by_capability(demand.capability_id)
            }
            for module_id, manifest in manifests.items():
                if demand.capability_id not in manifest.capabilities or module_id in executed:
                    continue
                record = runnable_records.get(module_id)
                if record is None or not record.runnable:
                    blocked.add(module_id)
                    diagnostics = self.registry.diagnostics(module_id)
                    status = (
                        "module has no runnable validated backend"
                        if diagnostics.get("backend_validated") is not True
                        else "module is disabled or has an unavailable required dependency"
                    )
                    decisions.setdefault(module_id, ResourceDecision(
                        module_id, False, 0.0, 0.0, 0.0, status,
                        estimate_source="availability_check",
                    ))
                    continue
                if self.resource_estimator.reliability.is_circuit_open(profile.task_family, module_id):
                    blocked.add(module_id)
                    decisions.setdefault(module_id, ResourceDecision(
                        module_id, False, 0.0, 0.0, 0.0,
                        "circuit breaker is open after repeated failures",
                        estimate_source="reliability_history",
                    ))
                    continue
                providers[demand.capability_id].add(module_id)
        candidate_capabilities: dict[str, tuple[str, ...]] = {}
        for module_id in sorted(set().union(*providers.values()) if providers else set()):
            candidate_capabilities[module_id] = tuple(sorted(
                capability for capability, values in providers.items() if module_id in values
            ))
        selected: set[str] = set()
        preferred: dict[str, str] = {}
        covered: set[str] = set()
        estimated_costs: dict[str, float] = {}
        expected_reduction = 0.0
        available_budget_calls = call_limit
        total_estimated_latency = 0.0
        total_estimated_memory = 0.0
        chosen_candidate_ids: set[str] = set()

        while available_budget_calls > 0:
            remaining = [item for item in active_demands if item.capability_id not in covered]
            if not remaining:
                break
            scored: list[tuple[float, str, ModuleResourceEstimate, float, float, set[str]]] = []
            for module_id, capability_ids in candidate_capabilities.items():
                if module_id in selected or module_id in chosen_candidate_ids:
                    continue
                newly_covered = [item for item in remaining if item.capability_id in capability_ids]
                if not newly_covered:
                    continue
                gain_numerator = sum(
                    item.priority * max(0.0, item.target_confidence - support.get(item.capability_id, 0.0))
                    for item in newly_covered
                )
                gain = gain_numerator / max(total_weighted_deficit, 1e-12)
                info_gain, _ = self.information_gain_estimator.estimate(
                    capability_ids, active_demands, uncertainty,
                )
                estimate = self.resource_estimator.estimate(profile.task_family, module_id, info_gain)
                failure_penalty = self.resource_estimator.reliability.failure_penalty(
                    profile.task_family, module_id,
                )
                score, cost = self.scorer.score(gain, estimate, failure_penalty, budget)
                closure, error = self._dependency_closure(module_id, executed | selected)
                if error:
                    decisions[module_id] = ResourceDecision(
                        module_id, False, score, cost, info_gain, error, gain,
                        estimate.estimated_success_probability, failure_penalty,
                        estimate.estimate_source,
                    )
                    chosen_candidate_ids.add(module_id)
                    continue
                new_dependencies = closure - selected - executed
                scored.append((score, module_id, estimate, cost, gain, new_dependencies))
            if not scored:
                break
            scored.sort(key=lambda item: (-item[0], item[1]))
            accepted = False
            for score, module_id, estimate, cost, gain, closure in scored:
                new_count = len(closure - selected - executed)
                if new_count > available_budget_calls:
                    decisions[module_id] = ResourceDecision(
                        module_id, False, score, cost, estimate.expected_information_gain,
                        "module plus required dependencies exceed the remaining module-call limit",
                        gain, estimate.estimated_success_probability,
                        self.resource_estimator.reliability.failure_penalty(profile.task_family, module_id),
                        estimate.estimate_source,
                    )
                    chosen_candidate_ids.add(module_id)
                    continue
                proposed_ids = closure - selected - executed
                proposed_estimates = {
                    item_id: self.resource_estimator.estimate(profile.task_family, item_id, 0.0)
                    for item_id in proposed_ids
                }
                proposed_latency = total_estimated_latency + sum(
                    item.expected_latency_ms for item in proposed_estimates.values()
                )
                proposed_memory = total_estimated_memory + sum(
                    item.expected_memory_mb for item in proposed_estimates.values()
                )
                if budget and budget.wall_clock_seconds is not None and proposed_latency > budget.wall_clock_seconds * 1000.0:
                    decisions[module_id] = ResourceDecision(
                        module_id, False, score, cost, estimate.expected_information_gain,
                        "estimated latency exceeds the remaining wall-clock budget",
                        gain, estimate.estimated_success_probability,
                        self.resource_estimator.reliability.failure_penalty(profile.task_family, module_id),
                        estimate.estimate_source,
                    )
                    chosen_candidate_ids.add(module_id)
                    continue
                if budget and budget.max_memory_mb is not None and proposed_memory > budget.max_memory_mb:
                    decisions[module_id] = ResourceDecision(
                        module_id, False, score, cost, estimate.expected_information_gain,
                        "conservative aggregate memory estimate exceeds the configured memory budget",
                        gain, estimate.estimated_success_probability,
                        self.resource_estimator.reliability.failure_penalty(profile.task_family, module_id),
                        estimate.estimate_source,
                    )
                    chosen_candidate_ids.add(module_id)
                    continue
                selected.update(proposed_ids)
                available_budget_calls -= new_count
                total_estimated_latency = proposed_latency
                total_estimated_memory = proposed_memory
                for item_id, item_estimate in proposed_estimates.items():
                    estimated_costs[item_id] = self.scorer.normalized_cost(item_estimate, budget)
                    if item_id != module_id and item_id not in decisions:
                        decisions[item_id] = ResourceDecision(
                            item_id, True, 0.0, estimated_costs[item_id], 0.0,
                            f"required dependency of {module_id}",
                            estimate_source=item_estimate.estimate_source,
                        )
                chosen_candidate_ids.add(module_id)
                decisions[module_id] = ResourceDecision(
                    module_id, True, score, cost, estimate.expected_information_gain,
                    "highest positive marginal utility among runnable candidates",
                    gain, estimate.estimated_success_probability,
                    self.resource_estimator.reliability.failure_penalty(profile.task_family, module_id),
                    estimate.estimate_source,
                )
                for demand in remaining:
                    if demand.capability_id in candidate_capabilities[module_id]:
                        covered.add(demand.capability_id)
                        preferred[demand.capability_id] = module_id
                        expected_reduction += (
                            demand.priority * uncertainty.get(demand.capability_id, 0.0)
                            * estimate.expected_information_gain
                        )
                accepted = True
                break
            if not accepted:
                break
        for module_id in sorted(candidate_capabilities):
            if module_id in decisions:
                continue
            capability_ids = candidate_capabilities[module_id]
            remaining = [item for item in active_demands if item.capability_id not in covered]
            if not any(item.capability_id in capability_ids for item in remaining):
                reason = "redundant: another selected module already covers its unmet capability"
            else:
                reason = "not selected: lower marginal utility or resource constraints"
            info_gain, _ = self.information_gain_estimator.estimate(
                capability_ids, active_demands, uncertainty,
            )
            estimate = self.resource_estimator.estimate(profile.task_family, module_id, info_gain)
            failure_penalty = self.resource_estimator.reliability.failure_penalty(
                profile.task_family, module_id,
            )
            score, cost = self.scorer.score(0.0, estimate, failure_penalty, budget)
            decisions[module_id] = ResourceDecision(
                module_id, False, score, cost, info_gain, reason, 0.0,
                estimate.estimated_success_probability, failure_penalty, estimate.estimate_source,
            )
        unmet = tuple(sorted(
            item.capability_id for item in active_demands if item.capability_id not in covered
        ))
        order = self._topological_order(selected, self.registry) if selected else ()
        estimated_total_cost = sum(estimated_costs.values())
        return ResourceExecutionPlan(
            task_id=profile.task_id,
            selected_modules=tuple(sorted(selected)),
            execution_order=order,
            estimated_total_cost=estimated_total_cost,
            expected_uncertainty_reduction=min(
                1.0, max(0.0, expected_reduction / max(total_weighted_deficit, 1e-12)),
            ),
            decisions=tuple(decisions[key] for key in sorted(decisions)),
            unmet_capabilities=unmet,
            blocked_modules=tuple(sorted(blocked)),
            preferred_modules_by_capability=preferred,
            uncertainty_by_capability=uncertainty,
        )


class AdaptiveResourceCoordinator:
    """Coordinate task-specific module selection, execution, feedback, recruitment, and SQLite learning."""

    def __init__(
        self,
        registry: ModuleRegistry,
        contract_catalog: Any,
        *,
        repository_root: str | Path | None = None,
        logger_database_path: str | Path | None = None,
        workspace_manager: Any | None = None,
        enable_logging: bool = True,
        max_workers: int = 1,
        default_timeout_seconds: float | None = None,
        declared_estimates: Mapping[str, ModuleResourceEstimate] | None = None,
        scorer: ResourceScorer | None = None,
        circuit_breaker_threshold: int = 3,
        circuit_breaker_cooldown_seconds: float = 60.0,
    ) -> None:
        """Connect adaptive planning to ORE's registry, executor, workspace, and shared logger database."""
        self.repository_root = (
            Path(repository_root).expanduser().resolve()
            if repository_root else Path(__file__).resolve().parents[1]
        )
        self.logger_database_path = (
            Path(logger_database_path).expanduser().resolve()
            if logger_database_path else self.repository_root / "ore_logs" / "ore_logger.sqlite3"
        )
        self.registry = registry
        self.enable_logging = enable_logging
        self.history = SQLiteResourceHistory(self.logger_database_path)
        self.reliability = ModuleReliabilityTracker(
            self.history,
            circuit_breaker_threshold=circuit_breaker_threshold,
            circuit_breaker_cooldown_seconds=circuit_breaker_cooldown_seconds,
        )
        self.resource_estimator = ModuleResourceEstimator(
            self.history, self.reliability, declared_estimates,
        )
        self.information_gain_estimator = InformationGainEstimator()
        self.scorer = scorer or ResourceScorer()
        self.plan_builder = ExecutionPlanBuilder(
            registry, self.resource_estimator, self.information_gain_estimator, self.scorer,
        )
        self.executor = CognitiveTaskCoordinator(
            registry, contract_catalog,
            workspace_manager=workspace_manager,
            repository_root=self.repository_root,
            logger_database_path=self.logger_database_path,
            enable_logging=enable_logging,
            max_workers=max_workers,
            default_timeout_seconds=default_timeout_seconds,
        )
        self._annotations = register_module_annotations(
            self.logger_database_path, self.repository_root,
        )

    @staticmethod
    def _task_with_resource_plan(
        task: CognitiveTask,
        profile: TaskProfile,
        plan: ResourceExecutionPlan,
        *,
        previous_results: Mapping[str, ModuleResult],
        excluded_modules: Sequence[str],
        carried_artifacts: Mapping[str, Any],
    ) -> CognitiveTask:
        """Build a normal ORE task with explicit provider preferences and completed results."""
        selected_capabilities = set(plan.preferred_modules_by_capability)
        required = tuple(
            demand.capability_id for demand in profile.demands
            if demand.mandatory and (
                demand.capability_id in selected_capabilities
                or demand.capability_id in plan.unmet_capabilities
            )
        )
        optional = tuple(
            demand.capability_id for demand in profile.demands
            if not demand.mandatory and demand.capability_id in selected_capabilities
        )
        metadata = dict(task.metadata)
        metadata.update({
            "resource_module_preferences": dict(plan.preferred_modules_by_capability),
            "resource_excluded_modules": list(excluded_modules),
            "_completed_module_results": dict(previous_results),
            "resource_execution_plan": plan.to_dict(),
        })
        inputs = dict(task.input_artifacts)
        for key, value in carried_artifacts.items():
            existing = inputs.get(key)
            existing_values = (
                tuple(existing) if isinstance(existing, (tuple, list))
                else (() if existing is None else (existing,))
            )
            new_values = tuple(value) if isinstance(value, (tuple, list)) else (value,)
            merged = {item.envelope.fingerprint(): item for item in (*existing_values, *new_values)}
            inputs[key] = tuple(merged[key] for key in sorted(merged))
        return CognitiveTask(
            goal=task.goal,
            required_capabilities=required,
            optional_capabilities=optional,
            input_artifacts=inputs,
            expected_outputs=task.expected_outputs,
            execution_mode=profile.execution_mode,
            priority=task.priority,
            max_replans=task.max_replans,
            metadata=metadata,
            task_id=task.task_id,
        )

    def _carry_forward_artifacts(
        self,
        artifacts: Sequence[Any],
        plan: ResourceExecutionPlan,
    ) -> dict[str, tuple[Any, ...]]:
        """Expose prior validated artifacts under compatible next-round input port IDs."""
        carried: dict[str, list[Any]] = {}
        for module_id in plan.selected_modules:
            manifest = self.registry.get_manifest(module_id)
            if manifest is None:
                continue
            for port in manifest.input_ports:
                matches_by_id = {
                    artifact.envelope.fingerprint(): artifact
                    for artifact in artifacts
                    if artifact.envelope.contract_id == port.contract_id
                    and artifact.envelope.schema_version in port.accepted_versions
                }
                matches = [matches_by_id[key] for key in sorted(matches_by_id)]
                if matches:
                    carried[port.port_id] = matches if port.multiple else [matches[-1]]
        return {key: tuple(value) for key, value in carried.items()}

    def _inspect_task_evidence(
        self,
        task: CognitiveTask,
        profile: TaskProfile,
        initial_support: Mapping[str, float],
    ) -> dict[str, float]:
        """Merge support metadata from validated input artifacts into current task state."""
        support = dict(initial_support)
        requested = {item.capability_id for item in profile.demands}
        pending = []
        for value in task.input_artifacts.values():
            pending.extend(value.values()) if isinstance(value, Mapping) else pending.append(value)
        while pending:
            value = pending.pop()
            if isinstance(value, (tuple, list)):
                pending.extend(value)
                continue
            if not hasattr(value, "envelope") or not hasattr(value, "metadata"):
                continue
            if not self.executor.catalog.validate_artifact(value).valid:
                continue
            metadata = value.metadata
            if not isinstance(metadata, Mapping):
                continue
            capability = metadata.get("capability_id")
            confidence = metadata.get("capability_support", metadata.get("confidence"))
            if capability not in requested or confidence is None:
                continue
            score = float(confidence)
            if math.isfinite(score) and 0.0 <= score <= 1.0:
                support[capability] = max(support.get(capability, 0.0), score)
        return support

    @staticmethod
    def _reported_support(
        results: Mapping[str, ModuleResult],
        registry: ModuleRegistry,
    ) -> dict[str, float]:
        """Extract only explicit, bounded capability-support evidence reported by module backends."""
        support: dict[str, float] = {}
        for module_id, result in results.items():
            manifest = registry.get_manifest(module_id)
            if manifest is None:
                continue
            evidence = result.diagnostics.get("capability_support")
            if evidence is None:
                evidence = result.processing_metadata.get("capability_support")
            if not isinstance(evidence, Mapping):
                continue
            for capability, raw_value in evidence.items():
                if capability not in manifest.capabilities:
                    continue
                value = float(raw_value)
                if math.isfinite(value) and 0.0 <= value <= 1.0:
                    support[capability] = max(support.get(capability, 0.0), value)
        return support

    @staticmethod
    def _task_satisfied(profile: TaskProfile, support: Mapping[str, float]) -> bool:
        """Require every mandatory target and the priority-weighted task stop confidence."""
        mandatory = [item for item in profile.demands if item.mandatory]
        if not mandatory:
            if not bool(profile.metadata.get("recruit_optional", False)):
                return True
            mandatory = list(profile.demands)
        if any(support.get(item.capability_id, 0.0) < item.target_confidence for item in mandatory):
            return False
        total_priority = sum(item.priority for item in mandatory)
        if total_priority <= 0.0:
            return True
        overall = sum(
            item.priority * support.get(item.capability_id, 0.0) for item in mandatory
        ) / total_priority
        return overall >= profile.stop_confidence

    def _record(
        self,
        logger: SQLiteModuleEventLogger | None,
        class_name: str,
        method_name: str,
        metadata: Mapping[str, Any],
    ) -> None:
        """Record adaptive decisions linked to AST-generated class and method annotations."""
        if logger is not None:
            logger.record(
                method_name, metadata, class_name=class_name,
                module_name="ore_contracts.resource_coordination",
            )

    def _learn_from_results(
        self,
        task_family: str,
        new_results: Mapping[str, ModuleResult],
        trace: Sequence[Any],
    ) -> None:
        """Update reliability and measured execution costs after real backend outcomes."""
        measured: dict[str, float] = {}
        for event in trace:
            if getattr(event, "operation", getattr(event, "kind", "")) == "execute_module":
                duration = getattr(event, "simulation_time_ms", None)
                if duration is not None:
                    measured[event.module_id] = float(duration)
        for module_id, result in new_results.items():
            if result.status is ModuleStatus.SUCCESS:
                self.reliability.record_outcome(task_family, module_id, True)
            elif result.status is ModuleStatus.FAILED:
                self.reliability.record_outcome(task_family, module_id, False)
            metadata = result.processing_metadata
            duration = measured.get(module_id, metadata.get("duration_ms"))
            memory = metadata.get("memory_mb")
            cpu = metadata.get("cpu_cost")
            if duration is not None or memory is not None or cpu is not None:
                self.resource_estimator.record_measurement(
                    task_family, module_id,
                    latency_ms=float(duration) if duration is not None else None,
                    memory_mb=float(memory) if memory is not None else None,
                    cpu_cost=float(cpu) if cpu is not None else None,
                )

    def coordinate(
        self,
        profile: TaskProfile,
        task: CognitiveTask,
        context: RunContext | None = None,
        *,
        current_support: Mapping[str, float] | None = None,
        belief_distributions: Mapping[str, Sequence[float]] | None = None,
    ) -> CoordinationResult:
        """Adaptively execute bounded resource batches and recruit alternatives until objectives are met."""
        if not isinstance(profile, TaskProfile) or not isinstance(task, CognitiveTask):
            raise TypeError("profile and task must be TaskProfile and CognitiveTask instances")
        if profile.execution_mode != task.execution_mode:
            raise ValueError("task execution_mode and task profile execution_mode must match")
        if task.task_id != profile.task_id:
            raise ValueError("task.task_id must match profile.task_id for an auditable run")
        support = _validate_support(current_support or task.metadata.get("capability_support", {}))
        support = self._inspect_task_evidence(task, profile, support)
        base_context = context or RunContext(
            task.task_id, OperationMode(profile.execution_mode), 0, ComputeBudget(),
        )
        if base_context.mode.value != profile.execution_mode:
            raise ValueError("RunContext.mode must match the task profile execution_mode")
        started = time.monotonic()
        all_results: dict[str, ModuleResult] = {}
        all_artifacts: list[Any] = []
        plans: list[ResourceExecutionPlan] = []
        total_calls = 0
        stop_reason = "no_recruitment_needed"
        logger_context = (
            SQLiteModuleEventLogger(
                self.logger_database_path, self.repository_root,
                display_label="ORE adaptive resource coordinator",
            ) if self.enable_logging else nullcontext(None)
        )
        with logger_context as logger:
            self._record(logger, "AdaptiveResourceCoordinator", "coordinate", {
                "task_id": profile.task_id, "task_family": profile.task_family,
                "initial_support": support, "max_module_calls": profile.max_module_calls,
                "max_replans": profile.max_replans,
            })
            if self._task_satisfied(profile, support):
                return CoordinationResult(
                    profile.task_id, (), {}, support, 0, True,
                    "mandatory_capabilities_already_satisfied",
                )
            for round_index in range(profile.max_replans + 1):
                elapsed = time.monotonic() - started
                budget = base_context.compute_budget
                remaining_seconds = (
                    None if budget.wall_clock_seconds is None
                    else max(0.0, budget.wall_clock_seconds - elapsed)
                )
                round_budget = (
                    replace(budget, wall_clock_seconds=remaining_seconds)
                    if remaining_seconds is not None and remaining_seconds > 0 else budget
                )
                if remaining_seconds is not None and remaining_seconds <= 0:
                    stop_reason = "wall_clock_budget_exhausted"
                    break
                remaining_calls = profile.max_module_calls - total_calls
                if remaining_calls <= 0:
                    stop_reason = "module_call_limit_reached"
                    break
                plan = self.plan_builder.build(
                    profile, support,
                    belief_distributions=belief_distributions,
                    already_executed=tuple(all_results),
                    remaining_module_calls=remaining_calls,
                    budget=round_budget,
                )
                plans.append(plan)
                self._record(logger, "ExecutionPlanBuilder", "build", {
                    "task_id": profile.task_id, "round": round_index, "plan": plan.to_dict(),
                })
                if not plan.selected_modules:
                    stop_reason = (
                        "mandatory_capabilities_satisfied" if self._task_satisfied(profile, support)
                        else "no_runnable_or_budget_feasible_candidate"
                    )
                    break
                candidate_ids = {
                    record.module_id
                    for demand in profile.demands
                    for record in self.registry.find_by_capability(demand.capability_id)
                }
                non_selected_providers = sorted(
                    candidate_ids - set(plan.selected_modules) - set(all_results)
                )
                carried = self._carry_forward_artifacts(all_artifacts, plan)
                round_task = self._task_with_resource_plan(
                    task, profile, plan, previous_results=all_results,
                    excluded_modules=non_selected_providers, carried_artifacts=carried,
                )
                round_context = replace(
                    base_context,
                    run_id=(
                        base_context.run_id if round_index == 0
                        else f"{base_context.run_id}:resource:{round_index}"
                    ),
                    compute_budget=round_budget,
                )
                previous_ids = set(all_results)
                result = self.executor.coordinate(round_task, round_context)
                new_results = {
                    module_id: module_result
                    for module_id, module_result in result.module_results.items()
                    if module_id not in previous_ids
                }
                all_results.update(new_results)
                all_artifacts.extend(
                    item.artifact for item in result.artifacts if item.artifact is not None
                )
                called_modules = {
                    event.module_id for event in result.trace
                    if getattr(event, "operation", "") in {"execute_module", "module_execution_failed"}
                }
                total_calls += len(called_modules)
                self._learn_from_results(profile.task_family, new_results, result.trace)
                reported = self._reported_support(new_results, self.registry)
                for capability, value in reported.items():
                    support[capability] = max(support.get(capability, 0.0), value)
                self._record(logger, "AdaptiveResourceCoordinator", "coordinate", {
                    "task_id": profile.task_id, "round": round_index,
                    "new_module_results": sorted(new_results),
                    "module_statuses": {key: value.status.value for key, value in new_results.items()},
                    "reported_support": reported, "updated_support": support,
                    "total_module_calls": total_calls,
                    "remaining_uncertainty": plan.unmet_capabilities,
                    "workspace_revision": result.workspace_revision,
                })
                if self._task_satisfied(profile, support):
                    stop_reason = "mandatory_capability_targets_and_stop_confidence_satisfied"
                    break
                if round_index >= profile.max_replans:
                    stop_reason = "maximum_replanning_rounds_reached"
                    break
                if not new_results:
                    stop_reason = "no_new_module_results"
                    break
            completed = self._task_satisfied(profile, support)
            if completed and stop_reason in {"no_recruitment_needed", "mandatory_capabilities_satisfied"}:
                stop_reason = "mandatory_capabilities_satisfied"
            self._record(logger, "AdaptiveResourceCoordinator", "coordinate", {
                "task_id": profile.task_id, "completed": completed,
                "stop_reason": stop_reason, "total_module_calls": total_calls,
                "final_support": support,
            })
            return CoordinationResult(
                task_id=profile.task_id,
                execution_plans=tuple(plans),
                module_results=all_results,
                final_support=support,
                total_module_calls=total_calls,
                completed=completed,
                stop_reason=stop_reason,
            )


__all__ = [
    "AdaptiveResourceCoordinator", "ExecutionPlanBuilder", "InformationGainEstimator",
    "ModuleReliabilityTracker", "ModuleResourceEstimator", "ResourceScorer",
    "SQLiteResourceHistory", "normalized_entropy",
]
