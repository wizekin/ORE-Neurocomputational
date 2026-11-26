"""Acceptance tests for ORE Point 12 search algorithms and hierarchical planning."""
import time

from ore_contracts import (
    ComputeBudget,
    ContractArtifact,
    ContractCatalog,
    InputPort,
    ModuleIdentity,
    ModuleManifest,
    ModuleRegistry,
    ModuleResult,
    ModuleStatus,
    OperationMode,
    OutputPort,
    RunContext,
    make_envelope,
)
from ore_contracts.coordinator import CognitiveTaskCoordinator
from ore_planning import (
    CostModel,
    CostVector,
    GoalConstraintAdapter,
    GoalSpec,
    HierarchicalPlanner,
    HTNDecomposer,
    HTNMethod,
    OperatorSpec,
    OutcomeEvaluator,
    PartialOrderBuilder,
    PlannerBudget,
    PlanState,
    Predicate,
    ResourceSpec,
    ScaledHeuristic,
    SearchEngine,
    SuccessCondition,
    TaskSpec,
    UnmetConditionHeuristic,
    ZeroHeuristic,
    build_default_methods,
)
from ore_planning.capability_adapter import CapabilityRegistryAdapter
from ore_planning.contracts import PlanNodeStatus
from ore_planning.replanner import OutcomeEvaluation


class _Backend:
    """Provide deterministic contract-producing behavior for planner tests."""

    def __init__(self, manifest, output_contract, source_port, *, fail=False):
        self.manifest = manifest
        self.output_contract = output_contract
        self.source_port = source_port
        self.fail = fail
        self.calls = 0

    def describe(self):
        return self.manifest

    def execute(self, inputs, context):
        self.calls += 1
        if self.fail:
            return ModuleResult(
                self.manifest.identity.module_id,
                self.manifest.identity.module_version,
                ModuleStatus.FAILED,
                diagnostics={"reason": "injected test failure"},
            )
        source = inputs[self.source_port]
        output = ContractArtifact(
            make_envelope(
                self.output_contract, context.run_id,
                self.manifest.identity.module_id,
                parent_artifact_ids=(source.envelope.fingerprint(),),
            ),
            {"derived": True, "producer": self.manifest.identity.module_id},
        )
        return ModuleResult(
            self.manifest.identity.module_id,
            self.manifest.identity.module_version,
            ModuleStatus.SUCCESS,
            artifacts=(output,),
        )


def _manifest(module_id, capability, input_contract, output_contract):
    """Build one two-port manifest bound to a single capability."""
    data_type = "builtins.str" if input_contract == "ORE.TEST.INPUT" else "builtins.dict"
    return ModuleManifest(
        ModuleIdentity(module_id, "1.0.0", f"Test {module_id}"),
        (f"provide {capability}",),
        (
            InputPort(
                "source", input_contract, ("1.0.0",),
                expected_data_type=data_type,
            ) if input_contract else
            InputPort(
                "source", output_contract, ("1.0.0",), required=False,
                expected_data_type=data_type,
            ),
        ),
        (OutputPort(
            "output", output_contract, "1.0.0", "builtins.dict", "test output",
        ),),
        (capability,),
    )


def _pipeline(fail_semantic=False, extra_encoder=None):
    """Create the standard two-stage lexical/semantic pipeline registry."""
    catalog = ContractCatalog()
    catalog.register_model("ORE.TEST.INPUT", "1.0.0", str)
    catalog.register_model("ORE.TEST.ENCODED", "1.0.0", dict)
    catalog.register_model("ORE.TEST.MEANING", "1.0.0", dict)
    encoder = _manifest(
        "ore.test_encoder", "language.lexical_access",
        "ORE.TEST.INPUT", "ORE.TEST.ENCODED",
    )
    semantic = _manifest(
        "ore.test_semantic", "language.semantic_composition",
        "ORE.TEST.ENCODED", "ORE.TEST.MEANING",
    )
    registry = ModuleRegistry(catalog)
    registry.register(encoder)
    registry.register(semantic)
    encoder_backend = _Backend(encoder, "ORE.TEST.ENCODED", "source")
    semantic_backend = _Backend(
        semantic, "ORE.TEST.MEANING", "source", fail=fail_semantic,
    )
    registry.bind_backend("ore.test_encoder", encoder_backend)
    registry.bind_backend("ore.test_semantic", semantic_backend)
    backends = {"ore.test_encoder": encoder_backend, "ore.test_semantic": semantic_backend}
    if extra_encoder is not None:
        module_id = extra_encoder
        alt = _manifest(
            module_id, "language.lexical_access", "ORE.TEST.INPUT", "ORE.TEST.ENCODED",
        )
        registry.register(alt)
        alt_backend = _Backend(alt, "ORE.TEST.ENCODED", "source")
        registry.bind_backend(module_id, alt_backend)
        backends[module_id] = alt_backend
    return catalog, registry, backends


def _context(run_id="planner-tests"):
    return RunContext(
        run_id=run_id, mode=OperationMode.FUNCTIONAL,
        deterministic_seed=3, compute_budget=ComputeBudget(),
    )


def _source_artifact(run_id="planner-tests"):
    return ContractArtifact(
        make_envelope("ORE.TEST.INPUT", run_id, "ore.test_input"),
        "The planner coordinates specialized modules.",
    )


def _planner(registry, catalog, **kwargs):
    coordinator = CognitiveTaskCoordinator(
        registry, catalog, enable_logging=False, max_workers=2,
    )
    defaults = {
        "budget": PlannerBudget(max_search_nodes=2000, max_time_seconds=20, max_depth=8),
        "heuristic": ZeroHeuristic(),
        "max_replans": 1,
    }
    defaults.update(kwargs)
    return HierarchicalPlanner(registry, catalog, coordinator=coordinator, **defaults)


GOAL_TEXT = "produce the semantic meaning of the input"
EXTERNAL = ("ORE.TEST.INPUT",)


# ---------------------------------------------------------------------------
# Acceptance 1: goal parsing creates typed predicates and preserves ambiguity
# ---------------------------------------------------------------------------
def test_goal_parsing_creates_typed_predicates_and_preserves_ambiguity():
    adapter = GoalConstraintAdapter()
    formalization = adapter.formalize(
        "extract lexical, syntactic and semantic representations",
    )
    assert len(formalization.interpretations) == 2
    assert formalization.ambiguous is True
    assert formalization.clarification_request is not None
    primary = formalization.primary
    assert primary.predicates and all(
        item.name == "goal.requires" for item in primary.predicates
    )
    labels = {item.interpretation_label for item in formalization.interpretations}
    assert labels == {"primary", "aggregate_capability"}


def test_goal_parsing_single_interpretation_and_critical_markers():
    adapter = GoalConstraintAdapter()
    formalization = adapter.formalize("parse the syntax of the sentence")
    assert formalization.ambiguous is False
    assert formalization.clarification_request is None
    assert len(formalization.primary.success_conditions) == 1
    critical = adapter.formalize("must verify the syntactic parse")
    assert all(
        condition.critical for condition in critical.primary.success_conditions
    )
    assert critical.primary.critical_constraints


def test_goal_parsing_extracts_forbidden_operations():
    adapter = GoalConstraintAdapter()
    formalization = adapter.formalize("summarize the discourse without translation")
    assert "translation" in formalization.primary.forbidden_operations


# ---------------------------------------------------------------------------
# Acceptance 2: operator eligibility checks schema, capability, permission,
# and dependency
# ---------------------------------------------------------------------------
def test_operator_eligibility_rejects_unavailable_modules():
    _catalog, registry, _ = _pipeline()
    unbound = _manifest(
        "ore.test_unbound", "language.discourse_processing",
        "ORE.TEST.ENCODED", "ORE.TEST.MEANING",
    )
    registry.register(unbound)
    adapter = CapabilityRegistryAdapter(registry)
    goal = GoalSpec(
        goal_id="g1", description="discourse",
        predicates=(Predicate("goal.requires", ("language.discourse_processing",)),),
        success_conditions=(
            SuccessCondition(Predicate("goal.satisfied", ("language.discourse_processing",))),
        ),
    )
    discovery = adapter.discover(goal)
    assert discovery.unsatisfied_capabilities == ("language.discourse_processing",)
    reasons = {item.reason for item in discovery.rejections}
    assert "module backend is not bound" in reasons


def test_operator_eligibility_rejects_forbidden_and_missing_dependencies():
    _catalog, registry, _ = _pipeline()
    adapter = CapabilityRegistryAdapter(registry)
    goal = GoalSpec(
        goal_id="g2", description="semantic",
        predicates=(Predicate("goal.requires", ("language.semantic_composition",)),),
        success_conditions=(
            SuccessCondition(
                Predicate("goal.satisfied", ("language.semantic_composition",)),
            ),
        ),
        forbidden_operations=("ore.test_semantic",),
    )
    discovery = adapter.discover(goal)
    assert discovery.unsatisfied_capabilities == ("language.semantic_composition",)
    assert any("forbidden" in item.reason for item in discovery.rejections)

    goal_no_external = GoalSpec(
        goal_id="g3", description="semantic",
        predicates=(Predicate("goal.requires", ("language.semantic_composition",)),),
        success_conditions=(
            SuccessCondition(
                Predicate("goal.satisfied", ("language.semantic_composition",)),
            ),
        ),
    )
    discovery = adapter.discover(goal_no_external)
    missing = [item for item in discovery.rejections if "missing dependency" in item.reason]
    assert missing, "encoder should be rejected without the external input contract"
    discovery = adapter.discover(goal_no_external, external_contracts=EXTERNAL)
    assert not any("missing dependency" in item.reason for item in discovery.rejections)


def test_operator_discovery_includes_transitive_producers():
    _catalog, registry, _ = _pipeline()
    adapter = CapabilityRegistryAdapter(registry)
    goal = GoalSpec(
        goal_id="g4", description="semantic",
        predicates=(Predicate("goal.requires", ("language.semantic_composition",)),),
        success_conditions=(
            SuccessCondition(
                Predicate("goal.satisfied", ("language.semantic_composition",)),
            ),
        ),
    )
    discovery = adapter.discover(goal, external_contracts=EXTERNAL)
    capabilities = {item.module_capability for item in discovery.operators}
    assert capabilities == {"language.semantic_composition", "language.lexical_access"}
    assert "language.lexical_access" in discovery.capability_map


# ---------------------------------------------------------------------------
# Acceptance 3-6: HTN decomposition
# ---------------------------------------------------------------------------
def _discovery_for(registry, capability="language.semantic_composition"):
    adapter = CapabilityRegistryAdapter(registry)
    goal = GoalSpec(
        goal_id=f"goal_{capability}", description="test",
        predicates=(Predicate("goal.requires", (capability,)),),
        success_conditions=(SuccessCondition(Predicate("goal.satisfied", (capability,))),),
    )
    return adapter.discover(goal, external_contracts=EXTERNAL)


def test_htn_decomposition_is_deterministic():
    _catalog, registry, _ = _pipeline()
    discovery = _discovery_for(registry)
    goal = GoalSpec(
        goal_id="goal_det", description="det",
        predicates=(Predicate("goal.requires", ("language.semantic_composition",)),),
        success_conditions=(
            SuccessCondition(Predicate("goal.satisfied", ("language.semantic_composition",))),
        ),
    )
    methods = build_default_methods(goal, discovery)
    decomposer = HTNDecomposer(methods)
    root = TaskSpec(
        task_id="root", task_type="goal.achieve", compound=True,
        completion=None,
    )
    first = decomposer.decompose(root, frozenset(), discovery)
    second = decomposer.decompose(root, frozenset(), discovery)
    assert first.to_dict() == second.to_dict()
    assert len(first.alternatives) == 2
    assert first.alternatives[0].provenance.method_id == "goal.achieve.parallel"


def test_htn_cycle_and_depth_limits_terminate_safely():
    _catalog, registry, _ = _pipeline()
    discovery = _discovery_for(registry)
    cyclic_method = HTNMethod(
        method_id="cycle.method", task_type="goal.achieve",
        applicability_conditions=(),
        subtasks=(
            TaskSpec(
                task_id="cycle_child", task_type="goal.achieve", compound=True,
                completion=None,
            ),
        ),
        version="1.0.0",
    )
    decomposer = HTNDecomposer((cyclic_method,))
    root = TaskSpec(
        task_id="root", task_type="goal.achieve", compound=True, completion=None,
    )
    started = time.monotonic()
    result = decomposer.decompose(root, frozenset(), discovery)
    assert time.monotonic() - started < 5.0
    assert result.cycle_count >= 1
    assert result.alternatives[0].status is PlanNodeStatus.UNSATISFIED

    deep_budget = PlannerBudget(max_depth=1)
    nested_method = HTNMethod(
        method_id="nested.stage", task_type="compound.stage",
        applicability_conditions=(),
        subtasks=(
            TaskSpec(
                task_id="nested_leaf", task_type="language.semantic_composition",
                completion=Predicate(
                    "goal.satisfied", ("language.semantic_composition",)
                ),
                compound=False,
            ),
        ),
        version="1.0.0",
    )
    root_method = HTNMethod(
        method_id="nested.root", task_type="goal.achieve",
        applicability_conditions=(),
        subtasks=(
            TaskSpec(
                task_id="nested_stage", task_type="compound.stage",
                completion=None, compound=True,
            ),
        ),
        version="1.0.0",
    )
    shallow = HTNDecomposer((nested_method, root_method), budget=deep_budget)
    result = shallow.decompose(root, frozenset(), discovery)
    assert result.truncated is True


def test_htn_preserves_alternatives_with_ranking_and_rationale():
    _catalog, registry, _ = _pipeline()
    discovery = _discovery_for(registry)
    goal = GoalSpec(
        goal_id="goal_alt", description="alt",
        predicates=(Predicate("goal.requires", ("language.semantic_composition",)),),
        success_conditions=(
            SuccessCondition(Predicate("goal.satisfied", ("language.semantic_composition",))),
        ),
    )
    methods = build_default_methods(goal, discovery)
    decomposer = HTNDecomposer(methods)
    root = TaskSpec(
        task_id="root", task_type="goal.achieve", compound=True, completion=None,
    )
    result = decomposer.decompose(root, frozenset(), discovery)
    assert len(result.alternatives) == 2
    priorities = [
        method.priority for method in decomposer.methods_for("goal.achieve")
    ]
    assert priorities == sorted(priorities, reverse=True)
    assert all(
        tree.provenance is not None for tree in result.alternatives
    )


def test_htn_marks_leaves_without_eligible_operators_unsatisfied():
    _catalog, registry, _ = _pipeline()
    discovery = _discovery_for(registry)
    method = HTNMethod(
        method_id="gap.method", task_type="goal.achieve",
        applicability_conditions=(),
        subtasks=(
            TaskSpec(
                task_id="gap_task", task_type="language.discourse_processing",
                completion=Predicate("goal.satisfied", ("language.discourse_processing",)),
                compound=False,
            ),
        ),
        version="1.0.0",
    )
    decomposer = HTNDecomposer((method,))
    root = TaskSpec(
        task_id="root", task_type="goal.achieve", compound=True, completion=None,
    )
    result = decomposer.decompose(root, frozenset(), discovery)
    assert result.unsatisfied_leaves == ("gap_task",)
    leaves = result.alternatives[0].leaves()
    assert leaves[0].status is PlanNodeStatus.UNSATISFIED
    assert leaves[0].task.task_type == "language.discourse_processing"


# ---------------------------------------------------------------------------
# Acceptance 7-8: partial-order graph
# ---------------------------------------------------------------------------
def _toy_operators():
    """Create a two-capability toy operator pool with known costs."""
    cheap = OperatorSpec(
        operator_id="op.a.cheap",
        preconditions=(),
        effects=(Predicate("artifact.available", ("X",)),
                 Predicate("goal.satisfied", ("test.a",))),
        input_schema={}, output_schema={"out": ("1.0.0",)},
        module_capability="test.a",
        estimated_cost=CostVector(time_ms=100.0),
    )
    expensive = OperatorSpec(
        operator_id="op.a.expensive",
        preconditions=(),
        effects=(Predicate("artifact.available", ("X",)),
                 Predicate("goal.satisfied", ("test.a",))),
        input_schema={}, output_schema={"out": ("1.0.0",)},
        module_capability="test.a",
        estimated_cost=CostVector(time_ms=300.0),
    )
    consumer = OperatorSpec(
        operator_id="op.b",
        preconditions=(Predicate("artifact.available", ("X",)),),
        effects=(Predicate("goal.satisfied", ("test.b",)),),
        input_schema={"in": ("1.0.0",)}, output_schema={"out": ("1.0.0",)},
        module_capability="test.b",
        estimated_cost=CostVector(time_ms=100.0),
    )
    return {"op.a.cheap": cheap, "op.a.expensive": expensive, "op.b": consumer}


def _toy_tree():
    leaves = (
        TaskSpec(
            task_id="t_a", task_type="test.a", compound=False,
            completion=Predicate("goal.satisfied", ("test.a",)),
        ),
        TaskSpec(
            task_id="t_b", task_type="test.b", compound=False,
            completion=Predicate("goal.satisfied", ("test.b",)),
        ),
    )
    from ore_planning import TaskTree
    return TaskTree(
        task=TaskSpec(task_id="root", task_type="goal.achieve", compound=True),
        children=tuple(TaskTree(task=leaf) for leaf in leaves),
    )


def _toy_goal():
    return GoalSpec(
        goal_id="toy", description="toy goal",
        predicates=(Predicate("goal.requires", ("test.b",)),),
        success_conditions=(
            SuccessCondition(Predicate("goal.satisfied", ("test.b",)), critical=True),
        ),
    )


def test_partial_order_graph_has_dependencies_and_concurrency():
    operators = _toy_operators()
    builder = PartialOrderBuilder(operators)
    plan = builder.build("p1", "toy", _toy_tree(), frozenset())
    assert plan.valid
    assert ("p1_n01_test_a", "p1_n02_test_b") in {
        (edge.predecessor, edge.successor) for edge in plan.edges
    }
    assert len(plan.concurrency_levels) == 2
    assert plan.ready_frontier == ("p1_n01_test_a",)


def test_partial_order_allows_independent_concurrency():
    operators = {
        "op.x": OperatorSpec(
            operator_id="op.x", preconditions=(), effects=(Predicate("done", ("x",)),),
            input_schema={}, output_schema={"o": ("1.0.0",)},
            module_capability="test.x", estimated_cost=CostVector(time_ms=10.0),
        ),
        "op.y": OperatorSpec(
            operator_id="op.y", preconditions=(), effects=(Predicate("done", ("y",)),),
            input_schema={}, output_schema={"o": ("1.0.0",)},
            module_capability="test.y", estimated_cost=CostVector(time_ms=10.0),
        ),
    }
    from ore_planning import TaskTree
    tree = TaskTree(
        task=TaskSpec(task_id="root", task_type="goal.achieve", compound=True),
        children=(
            TaskTree(task=TaskSpec(
                task_id="tx", task_type="test.x", compound=False,
                completion=Predicate("done", ("x",)),
            )),
            TaskTree(task=TaskSpec(
                task_id="ty", task_type="test.y", compound=False,
                completion=Predicate("done", ("y",)),
            )),
        ),
    )
    builder = PartialOrderBuilder(operators)
    plan = builder.build("p2", "toy", tree, frozenset())
    assert plan.valid
    assert not plan.edges, "independent tasks must stay unordered"
    assert plan.concurrency_levels == (("p2_n01_test_x", "p2_n02_test_y"),)


def test_mutex_resource_constraints_serialize_conflicting_operations():
    operators = {
        "op.x": OperatorSpec(
            operator_id="op.x", preconditions=(), effects=(Predicate("done", ("x",)),),
            input_schema={}, output_schema={"o": ("1.0.0",)},
            module_capability="test.x", estimated_cost=CostVector(time_ms=10.0),
            resource_requirements=ResourceSpec(
                required_units={"gpu": 1.0}, exclusive_resources=("gpu:0",),
            ),
        ),
        "op.y": OperatorSpec(
            operator_id="op.y", preconditions=(), effects=(Predicate("done", ("y",)),),
            input_schema={}, output_schema={"o": ("1.0.0",)},
            module_capability="test.y", estimated_cost=CostVector(time_ms=10.0),
            resource_requirements=ResourceSpec(
                required_units={"gpu": 1.0}, exclusive_resources=("gpu:0",),
            ),
        ),
    }
    from ore_planning import TaskTree
    tree = TaskTree(
        task=TaskSpec(task_id="root", task_type="goal.achieve", compound=True),
        children=(
            TaskTree(task=TaskSpec(
                task_id="tx", task_type="test.x", compound=False,
                completion=Predicate("done", ("x",)),
            )),
            TaskTree(task=TaskSpec(
                task_id="ty", task_type="test.y", compound=False,
                completion=Predicate("done", ("y",)),
            )),
        ),
    )
    builder = PartialOrderBuilder(operators)
    plan = builder.build("p3", "toy", tree, frozenset())
    assert plan.serialized_conflicts == (("p3_n01_test_x", "p3_n02_test_y", "gpu:0"),)
    assert plan.concurrency_levels == (("p3_n01_test_x",), ("p3_n02_test_y",))
    resource_edges = [
        edge for edge in plan.edges if edge.dependency_type.value == "resource"
    ]
    assert len(resource_edges) == 1


def test_partial_order_reports_cycles_as_invalid():
    operators = {
        "op.a": OperatorSpec(
            operator_id="op.a",
            preconditions=(Predicate("artifact.available", ("Y",)),),
            effects=(Predicate("artifact.available", ("X",)),),
            input_schema={}, output_schema={"o": ("1.0.0",)},
            module_capability="test.a",
        ),
        "op.b": OperatorSpec(
            operator_id="op.b",
            preconditions=(Predicate("artifact.available", ("X",)),),
            effects=(Predicate("artifact.available", ("Y",)),),
            input_schema={}, output_schema={"o": ("1.0.0",)},
            module_capability="test.b",
        ),
    }
    from ore_planning import TaskTree
    tree = TaskTree(
        task=TaskSpec(task_id="root", task_type="goal.achieve", compound=True),
        children=(
            TaskTree(task=TaskSpec(
                task_id="ta", task_type="test.a", compound=False,
                completion=Predicate("goal.satisfied", ("test.a",)),
            )),
            TaskTree(task=TaskSpec(
                task_id="tb", task_type="test.b", compound=False,
                completion=Predicate("goal.satisfied", ("test.b",)),
            )),
        ),
    )
    builder = PartialOrderBuilder(operators)
    plan = builder.build("p4", "toy", tree, frozenset())
    assert plan.valid is False
    assert set(plan.cycle_nodes) == {"p4_n01_test_a", "p4_n02_test_b"}
    assert plan.ready_frontier == ()


# ---------------------------------------------------------------------------
# Acceptance 9-13: search engine
# ---------------------------------------------------------------------------
def _run_toy_search(heuristic, mode="astar", budget=None, operators=None):
    operators = operators or _toy_operators()
    builder = PartialOrderBuilder(operators)
    plan = builder.build("p9", "toy", _toy_tree(), frozenset())
    engine = SearchEngine(
        CostModel(), heuristic, budget=budget or PlannerBudget(
            max_search_nodes=500, max_time_seconds=10, max_depth=8,
        ), mode=mode,
    )
    state = PlanState(
        workspace_revision=0, achieved_facts=frozenset(),
        pending_tasks=tuple(node.node_id for node in plan.nodes),
    )
    return engine.search(plan, _toy_goal(), operators, state)


def test_astar_returns_known_optimum_on_toy_graph():
    operators = _toy_operators()
    outcome = _run_toy_search(ZeroHeuristic(), operators=operators)
    assert outcome.goal_achieved
    cost_model = CostModel()
    expected = (
        cost_model.vector_cost(operators["op.a.cheap"].estimated_cost)
        + cost_model.vector_cost(operators["op.b"].estimated_cost)
    )
    assert outcome.total_cost == expected
    assert outcome.optimality_claim is True


def test_inadmissible_heuristic_makes_no_optimality_claim():
    outcome = _run_toy_search(ScaledHeuristic(ZeroHeuristic(), 2.0))
    assert outcome.goal_achieved
    assert outcome.optimality_claim is False


def test_admissible_inconsistent_heuristic_keeps_claim_via_exhaustion():
    outcome = _run_toy_search(UnmetConditionHeuristic())
    assert outcome.goal_achieved
    assert outcome.optimality_claim is True


def test_beam_search_respects_k_and_records_pruned_branches():
    operators = {
        "op.x": OperatorSpec(
            operator_id="op.x", preconditions=(),
            effects=(Predicate("goal.satisfied", ("test.x",)),),
            input_schema={}, output_schema={"o": ("1.0.0",)},
            module_capability="test.x",
            estimated_cost=CostVector(time_ms=10.0),
        ),
        "op.y": OperatorSpec(
            operator_id="op.y", preconditions=(),
            effects=(Predicate("goal.satisfied", ("test.y",)),),
            input_schema={}, output_schema={"o": ("1.0.0",)},
            module_capability="test.y",
            estimated_cost=CostVector(time_ms=10.0),
        ),
    }
    from ore_planning import TaskTree
    tree = TaskTree(
        task=TaskSpec(task_id="root", task_type="goal.achieve", compound=True),
        children=(
            TaskTree(task=TaskSpec(
                task_id="tx", task_type="test.x", compound=False,
                completion=Predicate("goal.satisfied", ("test.x",)),
            )),
            TaskTree(task=TaskSpec(
                task_id="ty", task_type="test.y", compound=False,
                completion=Predicate("goal.satisfied", ("test.y",)),
            )),
        ),
    )
    goal = GoalSpec(
        goal_id="beam", description="beam goal",
        predicates=(Predicate("goal.requires", ("test.x",)),),
        success_conditions=(
            SuccessCondition(Predicate("goal.satisfied", ("test.x",)), critical=True),
            SuccessCondition(Predicate("goal.satisfied", ("test.y",)), critical=True),
        ),
    )
    builder = PartialOrderBuilder(operators)
    plan = builder.build("p_beam", "beam", tree, frozenset())
    engine = SearchEngine(
        CostModel(), ZeroHeuristic(),
        budget=PlannerBudget(
            max_search_nodes=500, max_time_seconds=10, max_depth=8, beam_width=1,
        ),
        mode="beam",
    )
    state = PlanState(
        workspace_revision=0, achieved_facts=frozenset(), pending_tasks=(),
    )
    outcome = engine.search(plan, goal, operators, state)
    assert outcome.goal_achieved
    assert outcome.stats.beam_pruned >= 1
    assert outcome.optimality_claim is False


def test_duplicate_states_are_detected_and_pruned():
    operators = _toy_operators()
    outcome = _run_toy_search(ZeroHeuristic(), operators=operators)
    assert outcome.stats.duplicates_pruned + outcome.stats.dominance_pruned >= 1


def test_search_budgets_are_enforced():
    outcome = _run_toy_search(
        ZeroHeuristic(),
        budget=PlannerBudget(
            max_search_nodes=1, max_time_seconds=10, max_depth=8, beam_width=2,
        ),
    )
    assert outcome.goal_achieved is False
    assert outcome.stats.budget_exhausted is True
    assert outcome.stop_reason == "search_budget_exhausted"


# ---------------------------------------------------------------------------
# Acceptance 14-20: execution, feedback, replanning, provenance
# ---------------------------------------------------------------------------
def test_existing_coordinator_executes_plan_without_duplicate_scheduling():
    catalog, registry, backends = _pipeline()
    planner = _planner(registry, catalog)
    planning = planner.plan(
        GOAL_TEXT, workspace_revision=0, external_contracts=EXTERNAL,
    )
    assert planning.status == "planned"
    assert planning.selected is not None
    run = planner.execute(
        planning, input_artifacts={"source": _source_artifact()},
        run_context=_context(),
    )
    assert run.status.value == "COMPLETE"
    assert run.goal_progress_score == 1.0
    assert run.termination_reason == "verified_success_conditions"
    assert backends["ore.test_encoder"].calls == 1
    assert backends["ore.test_semantic"].calls == 1
    assert len(run.feedback) == 2
    assert all(item.success for item in run.feedback)


def test_search_selects_cheapest_provider_among_alternatives():
    catalog, registry, backends = _pipeline(
        extra_encoder="ore.test_zencoder",
    )
    planner = _planner(
        registry, catalog,
        measured_costs={
            "ore.test_encoder": CostVector(time_ms=100.0),
            "ore.test_zencoder": CostVector(time_ms=500.0),
        },
    )
    planning = planner.plan(
        GOAL_TEXT, workspace_revision=0, external_contracts=EXTERNAL,
    )
    selected = planning.selected
    assert selected is not None
    chosen = dict(selected.outcome.operator_sequence)
    lexical_node = next(
        node.node_id for node in selected.plan.nodes
        if node.task.task_type == "language.lexical_access"
    )
    assert chosen[lexical_node] == "ore.test_encoder"
    assert selected.decision.optimality_claim is True
    assert backends["ore.test_zencoder"].calls == 0


def test_partial_failure_reports_incomplete_and_replans_safely():
    _catalog, registry, _backends = _pipeline(fail_semantic=True)
    planner = _planner(registry, _catalog, max_replans=1)
    planning = planner.plan(
        "must verify the semantic meaning of the input",
        workspace_revision=0, external_contracts=EXTERNAL,
    )
    run = planner.execute(
        planning, input_artifacts={"source": _source_artifact()},
        run_context=_context("failing-run"),
    )
    assert run.status.value == "INCOMPLETE"
    assert run.unresolved_subgoals
    assert run.critical_satisfied is False
    assert run.termination_reason in {
        "maximum_replans_reached", "no_better_plan_available",
    }
    assert run.replans >= 1
    assert any(not item.success for item in run.feedback)
    replan_steps = [
        item for item in planning.provenance.steps
        if item.phase == "E" and item.step == 36
    ]
    assert replan_steps, "the replanning step must be traced on failure"


def test_plan_predictions_are_compared_with_actual_outcomes():
    catalog, registry, _ = _pipeline()
    planner = _planner(registry, catalog)
    planning = planner.plan(
        GOAL_TEXT, workspace_revision=0, external_contracts=EXTERNAL,
    )
    run = planner.execute(
        planning, input_artifacts={"source": _source_artifact()},
        run_context=_context(),
    )
    assert run.feedback
    for item in run.feedback:
        assert item.predicted_cost is not None
        assert item.actual_cost is not None
    evaluator = OutcomeEvaluator(CostModel())
    evaluation = evaluator.evaluate_outcomes(run.feedback)
    assert isinstance(evaluation, OutcomeEvaluation)
    assert evaluation.mean_absolute_error >= 0.0
    assert set(evaluation.succeeded_operators) == {
        "ore.test_encoder", "ore.test_semantic",
    }


def test_workspace_revision_changes_invalidate_plan_conditions():
    catalog, registry, _ = _pipeline()
    planner = _planner(registry, catalog)
    planning = planner.plan(
        GOAL_TEXT, workspace_revision=3, external_contracts=EXTERNAL,
    )
    plan = planning.selected.plan
    evaluator = OutcomeEvaluator(CostModel())
    assert evaluator.revision_invalidated(plan, 3) is False
    assert evaluator.revision_invalidated(plan, 4) is True
    operators = {
        item.operator_id: item for item in planning.discovery.operators
    }
    state = PlanState(workspace_revision=4, achieved_facts=frozenset(), pending_tasks=())
    invalidated = evaluator.validate_preconditions(plan, operators, state)
    assert invalidated, "nodes requiring unavailable inputs must be recomputed"
    state_ok = PlanState(
        workspace_revision=4,
        achieved_facts=frozenset({
            "artifact.available(ORE.TEST.INPUT)",
            "artifact.available(ORE.TEST.ENCODED)",
        }),
        pending_tasks=(),
    )
    assert evaluator.validate_preconditions(plan, operators, state_ok) == ()


def test_replanning_threshold_requires_expected_benefit():
    weak = OutcomeEvaluator.should_replan(
        expected_quality_gain=0.2, expected_saved_cost=0.0,
    )
    assert weak.should_replan is False
    strong = OutcomeEvaluator.should_replan(
        expected_quality_gain=1.5, expected_saved_cost=1.0,
    )
    assert strong.should_replan is True
    failed = OutcomeEvaluator.should_replan(
        expected_quality_gain=0.0, expected_saved_cost=0.0,
        failed_operators=("ore.test_semantic",),
    )
    assert failed.should_replan is True
    assert "failed" in failed.rationale


def test_checkpoint_resume_does_not_duplicate_completed_operations():
    catalog, registry, backends = _pipeline(fail_semantic=True)
    planner = _planner(registry, catalog, max_replans=0)
    planning = planner.plan(
        GOAL_TEXT, workspace_revision=0, external_contracts=EXTERNAL,
    )
    selected = planning.selected
    report = planner.executor.submit(
        selected.plan, selected.goal, selected.outcome,
        input_artifacts={"source": _source_artifact()},
        run_context=_context("checkpoint-run"),
    )
    assert backends["ore.test_encoder"].calls == 1
    assert backends["ore.test_semantic"].calls == 1
    checkpoint = planner.executor.checkpoint(
        selected.plan, selected.goal, report,
    )
    assert set(checkpoint.completed_module_results) == {"ore.test_encoder"}

    backends["ore.test_semantic"].fail = False
    resumed = planner.executor.submit(
        selected.plan, selected.goal, selected.outcome,
        input_artifacts={"source": _source_artifact()},
        run_context=_context("resume-run"),
        checkpoint=checkpoint,
    )
    assert backends["ore.test_encoder"].calls == 1, "completed module must not rerun"
    assert backends["ore.test_semantic"].calls == 2
    assert all(item.success for item in resumed.feedback)


def test_logging_correlates_goal_plan_modules_and_revision():
    catalog, registry, _ = _pipeline()
    planner = _planner(registry, catalog)
    planning = planner.plan(
        GOAL_TEXT, workspace_revision=7, external_contracts=EXTERNAL,
    )
    run = planner.execute(
        planning, input_artifacts={"source": _source_artifact()},
        run_context=_context("correlate-run"),
    )
    provenance = planning.provenance
    goal_id = planning.selected.goal.goal_id
    plan_id = planning.selected.plan.plan_id
    goal_records = provenance.for_goal(goal_id)
    plan_records = provenance.for_plan(plan_id)
    assert goal_records and plan_records
    assert any(record.module_ids for record in goal_records)
    assert all(
        record.workspace_revision == 7
        for record in goal_records if record.event == "operator_discovery"
    )
    step_keys = {(item.phase, item.step) for item in provenance.steps}
    executed_keys = {
        (item.phase, item.step) for item in provenance.steps
        if item.phase == "E"
    }
    unconditional_e = {("E", step) for step in range(29, 41)} - {("E", 36)}
    assert executed_keys == unconditional_e, (
        "phase E steps 29-35 and 37-40 must be recorded on a clean run; "
        "step 36 (replan) is conditional by design"
    )
    assert ("A", 1) in step_keys and ("D", 28) in step_keys
    assert run.workspace_revision >= 0


def test_planner_step_registry_matches_specification():
    from ore_planning import PLANNER_STEPS
    assert len(PLANNER_STEPS) == 40
    phases = [phase for phase, _, _ in PLANNER_STEPS]
    assert phases.count("A") == 8
    assert phases.count("B") == 8
    assert phases.count("C") == 5
    assert phases.count("D") == 7
    assert phases.count("E") == 12
    for phase, step, description in PLANNER_STEPS:
        assert description.strip(), f"step {phase}{step} needs a description"


def test_mcts_is_explicitly_not_implemented():
    import ore_planning.planner as planner_module
    doc = planner_module.__doc__
    assert "MCTS" in doc
    assert "not implemented" in doc
    import ore_planning.search as search_module
    assert "MCTS" not in [
        name for name in dir(search_module) if not name.startswith("_")
    ]


def test_cost_model_goal_progress_hard_gates_critical_predicates():
    model = CostModel()
    goal = GoalSpec(
        goal_id="gates", description="gates",
        predicates=(Predicate("goal.requires", ("x",)),),
        success_conditions=(
            SuccessCondition(Predicate("goal.satisfied", ("x",)), weight=3.0),
            SuccessCondition(
                Predicate("goal.satisfied", ("y",)), weight=1.0, critical=True,
            ),
        ),
    )
    progress = model.goal_progress(goal, frozenset({"goal.satisfied(x)"}))
    assert progress.score == 0.75
    assert progress.critical_satisfied is False
    assert progress.critical_unsatisfied == ("goal.satisfied(y)",)
    assert progress.achieved is False


def test_calibrated_cost_model_updates_from_execution_history():
    from ore_planning import ExecutionFeedback
    model = CostModel()
    feedback = (
        ExecutionFeedback(
            task_id="t1", node_id=None, operator_id="op.x",
            actual_cost=CostVector(time_ms=4000.0, memory_mb=200.0),
            predicted_cost=CostVector(time_ms=100.0, memory_mb=1.0),
            outputs=(), success=True,
        ),
    )
    calibrated = model.calibrate(feedback)
    assert calibrated.time_scale_ms >= 4000.0
    assert calibrated.weights.version != model.weights.version
