"""Acceptance tests for adaptive ORE cognitive resource allocation."""
import ast
from pathlib import Path

import pytest

from ore_contracts import (
    ComputeBudget, ContractCatalog, ModuleDependency, ModuleIdentity, ModuleManifest,
    ModuleRegistry, ModuleResult, ModuleStatus, OperationMode, RunContext,
)
from ore_contracts.coordinator import CognitiveTask
from ore_contracts.module_logging import register_module_annotations
from ore_contracts.resource_profiles import (
    CapabilityDemand, ModuleResourceEstimate, TaskProfile, TaskProfileRegistry,
)
from ore_contracts.resource_coordination import (
    AdaptiveResourceCoordinator, ExecutionPlanBuilder, InformationGainEstimator,
    ModuleResourceEstimator, ModuleReliabilityTracker, ResourceScorer,
    SQLiteResourceHistory, normalized_entropy,
)
from ore_logger.database import LoggerDatabase


class _Backend:
    """Provide a deterministic backend that reports evidence-backed capability support."""

    def __init__(self, manifest, support):
        """Store the registered manifest and the support score returned by execution."""
        self.manifest = manifest
        self.support = support

    def describe(self):
        """Return the exact registered manifest for structural validation."""
        return self.manifest

    def execute(self, inputs, context):
        """Return a successful module result with explicit capability-support evidence."""
        capability = self.manifest.capabilities[0]
        return ModuleResult(
            self.manifest.identity.module_id,
            self.manifest.identity.module_version,
            ModuleStatus.SUCCESS,
            diagnostics={"capability_support": {capability: self.support}},
        )


def _manifest(module_id, capabilities, dependencies=()):
    """Build a port-free test manifest with explicit capability and dependency declarations."""
    return ModuleManifest(
        ModuleIdentity(module_id, "1.0.0", module_id),
        ("provide a deterministic test capability",),
        (), (), tuple(capabilities), tuple(dependencies),
    )


def _registry(manifests, support=None, unbound=()):
    """Register manifests and bind only the backends requested by the test."""
    catalog = ContractCatalog()
    registry = ModuleRegistry(catalog)
    for manifest in manifests:
        registry.register(manifest)
    scores = support or {}
    for manifest in manifests:
        if manifest.identity.module_id not in set(unbound):
            registry.bind_backend(
                manifest.identity.module_id,
                _Backend(manifest, scores.get(manifest.identity.module_id, 0.95)),
            )
    return catalog, registry


def _components(tmp_path, registry, declared=None, threshold=3):
    """Create resource estimators that persist to one temporary logger database."""
    history = SQLiteResourceHistory(tmp_path / "ore_logger.sqlite3")
    reliability = ModuleReliabilityTracker(history, circuit_breaker_threshold=threshold)
    estimator = ModuleResourceEstimator(history, reliability, declared)
    return history, reliability, estimator


def test_entropy_is_normalized_and_rejects_invalid_distributions():
    """Verify the entropy range, certainty edge case, and probability constraints."""
    assert normalized_entropy((0.5, 0.5)) == pytest.approx(1.0)
    assert normalized_entropy((1.0, 0.0)) == pytest.approx(0.0, abs=1e-8)
    assert normalized_entropy((1.0,)) is None
    with pytest.raises(ValueError):
        normalized_entropy((0.7, 0.7))
    with pytest.raises(ValueError):
        normalized_entropy((0.5, float("nan")))


def test_task_profiles_are_deterministic_and_validate_requirements():
    """Register stable task templates and reject duplicate or invalid capability demands."""
    registry = TaskProfileRegistry()
    profile = TaskProfile(
        "task-a", "word_recognition",
        (CapabilityDemand("vision.feature_extraction", 1.0, True, 0.9),),
    )
    registry.register(profile)
    assert registry.resolve("word_recognition") is profile
    assert registry.resolve("word_recognition", task_id="task-b") == TaskProfile(
        "task-b", "word_recognition",
        (CapabilityDemand("vision.feature_extraction", 1.0, True, 0.9),),
    )
    enum_profile = TaskProfile(
        "task-enum", "enum_mode",
        (CapabilityDemand("language.lexical_access"),),
        execution_mode=OperationMode.FUNCTIONAL,
    )
    assert enum_profile.execution_mode == "FUNCTIONAL"
    assert registry.list_profiles() == (profile,)
    with pytest.raises(ValueError):
        registry.register(profile)
    with pytest.raises(KeyError):
        registry.resolve("missing")
    with pytest.raises(ValueError):
        CapabilityDemand("", 1.0, True, 0.9)
    with pytest.raises(ValueError):
        CapabilityDemand("vision.feature_extraction", 1.1, True, 0.9)


def test_beta_reliability_circuit_breaker_and_moving_average_persist(tmp_path):
    """Check Beta posterior updates, timed failure protection, and EMA resource estimates."""
    history, reliability, estimator = _components(tmp_path, None, threshold=2)
    assert reliability.probability("metaphor", "ore.analogy") == pytest.approx(0.5)
    reliability.record_outcome("metaphor", "ore.analogy", False)
    reliability.record_outcome("metaphor", "ore.analogy", False)
    assert reliability.probability("metaphor", "ore.analogy") == pytest.approx(2 / 6)
    assert reliability.is_circuit_open("metaphor", "ore.analogy")
    estimator.record_measurement(
        "metaphor", "ore.analogy", latency_ms=100, memory_mb=50, cpu_cost=2,
    )
    estimator.record_measurement(
        "metaphor", "ore.analogy", latency_ms=200, memory_mb=70, cpu_cost=4,
    )
    row = history.get("metaphor", "ore.analogy")
    assert row["average_latency_ms"] == pytest.approx(120)
    assert row["average_memory_mb"] == pytest.approx(54)
    assert row["average_cpu_cost"] == pytest.approx(2.4)


def test_score_is_finite_with_zero_cost_and_uses_failure_penalty():
    """Verify numeric stability at zero resource cost and lower utility after failures."""
    scorer = ResourceScorer()
    estimate = ModuleResourceEstimate("ore.fast", 0, 0, 0, 0.5, 0.8, "measured_history")
    score, cost = scorer.score(0.8, estimate, 0.2, None)
    assert score > 0
    assert cost == 0
    assert scorer.score(0.8, estimate, 0.9, None)[0] < score


def test_greedy_planner_selects_multi_capability_provider_and_orders_dependencies(tmp_path):
    """Choose the efficient provider for multiple deficits and schedule its prerequisite first."""
    base = _manifest("ore.base", ("reasoning.base",))
    specialist = _manifest(
        "ore.specialist",
        ("reasoning.semantic", "reasoning.context"),
        (ModuleDependency("ore.base"),),
    )
    duplicate = _manifest("ore.duplicate", ("reasoning.semantic",))
    catalog, registry = _registry((base, specialist, duplicate))
    _, _, estimator = _components(tmp_path, registry, {
        "ore.base": ModuleResourceEstimate("ore.base", 100, 20, 0.5, 0.2, 0.9, "declared"),
        "ore.specialist": ModuleResourceEstimate("ore.specialist", 120, 40, 0.8, 0.9, 0.95, "declared"),
        "ore.duplicate": ModuleResourceEstimate("ore.duplicate", 1000, 300, 5, 0.2, 0.7, "declared"),
    })
    plan = ExecutionPlanBuilder(
        registry, estimator, InformationGainEstimator(), ResourceScorer(),
    ).build(
        TaskProfile("task-b", "metaphor", (
            CapabilityDemand("reasoning.semantic", 1.0, True, 0.9),
            CapabilityDemand("reasoning.context", 0.9, True, 0.85),
        ), max_module_calls=3),
        {},
        budget=ComputeBudget(),
    )
    assert plan.preferred_modules_by_capability == {
        "reasoning.semantic": "ore.specialist",
        "reasoning.context": "ore.specialist",
    }
    assert plan.execution_order == ("ore.base", "ore.specialist")
    assert plan.selected_modules == ("ore.base", "ore.specialist")
    assert "ore.duplicate" not in plan.selected_modules
    assert not plan.unmet_capabilities


def test_planner_rejects_unavailable_modules_and_respects_call_limit(tmp_path):
    """Do not select modules without validated backends or exceed the module-call cap."""
    available = _manifest("ore.available", ("language.lexical_access",))
    unavailable = _manifest("ore.unavailable", ("language.lexical_access",))
    _, registry = _registry((available, unavailable), unbound=("ore.unavailable",))
    _, _, estimator = _components(tmp_path, registry)
    plan = ExecutionPlanBuilder(registry, estimator).build(
        TaskProfile(
            "task-c", "word", (CapabilityDemand("language.lexical_access"),),
            max_module_calls=1,
        ),
        {},
        remaining_module_calls=0,
    )
    assert plan.selected_modules == ()
    assert plan.unmet_capabilities == ("language.lexical_access",)
    assert "ore.unavailable" in plan.blocked_modules


def test_adaptive_coordinator_recruits_an_alternative_after_insufficient_support(tmp_path):
    """Recruit a second provider only after the first provider misses its confidence target."""
    first = _manifest("ore.first", ("semantic.interpretation",))
    second = _manifest("ore.second", ("semantic.interpretation",))
    catalog, registry = _registry(
        (first, second), {"ore.first": 0.6, "ore.second": 0.95},
    )
    declared = {
        "ore.first": ModuleResourceEstimate("ore.first", 100, 20, 0.5, 0.8, 0.95, "declared"),
        "ore.second": ModuleResourceEstimate("ore.second", 200, 30, 0.6, 0.8, 0.95, "declared"),
    }
    coordinator = AdaptiveResourceCoordinator(
        registry, catalog,
        repository_root=Path(__file__).resolve().parents[1],
        logger_database_path=tmp_path / "ore_logger.sqlite3",
        enable_logging=False,
        declared_estimates=declared,
    )
    profile = TaskProfile(
        "adaptive-task", "metaphor",
        (CapabilityDemand("semantic.interpretation", 1.0, True, 0.9),),
        stop_confidence=0.9, max_module_calls=2, max_replans=1,
    )
    task = CognitiveTask(
        goal="interpret metaphor",
        required_capabilities=("semantic.interpretation",),
        execution_mode="FUNCTIONAL",
        task_id="adaptive-task",
    )
    result = coordinator.coordinate(
        profile, task,
        RunContext("adaptive-run", OperationMode.FUNCTIONAL, 0, ComputeBudget()),
    )
    assert result.completed
    assert result.final_support["semantic.interpretation"] == pytest.approx(0.95)
    assert result.total_module_calls == 2
    assert len(result.execution_plans) == 2
    assert set(result.module_results) == {"ore.first", "ore.second"}
    assert result.stop_reason == "mandatory_capability_targets_and_stop_confidence_satisfied"


def test_every_resource_coordination_definition_has_a_sqlite_annotation(tmp_path):
    """Index every new resource-coordination class and callable in ore_logger.annotations."""
    root = Path(__file__).resolve().parents[1]
    database_path = tmp_path / "ore_logger.sqlite3"
    report = register_module_annotations(database_path, root)
    assert report["parse_errors"] == 0
    database = LoggerDatabase(database_path)
    for relative_path in (
        "ore_contracts/resource_profiles.py",
        "ore_contracts/resource_coordination.py",
    ):
        module_name = relative_path[:-3].replace("/", ".")
        tree = ast.parse((root / relative_path).read_text(encoding="utf-8"))
        expected = []

        def collect(body, parent=""):
            """Collect qualified class and callable names for the logger annotation check."""
            for node in body:
                if isinstance(node, ast.ClassDef):
                    qualified = f"{parent}.{node.name}" if parent else node.name
                    expected.append((qualified, "class"))
                    collect(node.body, qualified)
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    qualified = f"{parent}.{node.name}" if parent else node.name
                    kind = "async_function" if isinstance(node, ast.AsyncFunctionDef) else "function"
                    expected.append((qualified, kind))
                    collect(node.body, qualified + ".<locals>")
                else:
                    for child in ast.iter_child_nodes(node):
                        if isinstance(child, ast.Module):
                            collect(child.body, parent)

        collect(tree.body)
        assert expected
        for qualified_name, kind in expected:
            annotation = database.get_annotation(
                module_name, qualified_name, kind, relative_path,
            )
            assert annotation is not None, f"missing annotation: {relative_path}:{qualified_name}"
            assert annotation["annotation"].strip()
