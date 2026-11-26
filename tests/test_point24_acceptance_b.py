"""Point 24 acceptance tests 26-50: planning, execution, learning, resource, security.

These implement the second half of the 50-test acceptance plan in Section 6
of the Point 24 specification.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ore_contracts import register_existing_contracts  # noqa: E402
from ore_contracts.adapters import make_envelope  # noqa: E402
from ore_contracts.contracts_registry import ContractCatalog  # noqa: E402
from ore_contracts.coordinator import (  # noqa: E402
    CognitiveTask,
    CognitiveTaskCoordinator,
)
from ore_contracts.module_artifacts import ContractArtifact  # noqa: E402
from ore_contracts.module_protocol import ModuleResult, ModuleStatus  # noqa: E402
from ore_contracts.module_registry import ModuleRegistry  # noqa: E402

from ore_point24 import artifacts as p24_artifacts  # noqa: E402
from ore_point24 import evidence as p24_evidence  # noqa: E402
from ore_point24 import execution as p24_exec  # noqa: E402
from ore_point24 import learning_gates as p24_learn  # noqa: E402
from ore_point24 import memory as p24_memory  # noqa: E402
from ore_point24 import mapek as p24_mapek  # noqa: E402
from ore_point24 import planning as p24_planning  # noqa: E402
from ore_point24 import probabilistic as p24_prob  # noqa: E402
from ore_point24 import reasoning as p24_reason  # noqa: E402
from ore_point24 import recovery as p24_recovery  # noqa: E402
from ore_point24 import reliability as p24_reliability  # noqa: E402
from ore_point24 import search_algorithms as p24_search  # noqa: E402
from ore_point24 import uncertainty as p24_uncertainty  # noqa: E402
from ore_point24 import verification as p24_verify  # noqa: E402
from ore_point24.admission import (  # noqa: E402
    DeterministicAdmissionController,
    ResourceBudget,
)
from ore_point24.capabilities import CapabilityAuthority, SandboxEnforcer  # noqa: E402
from ore_point24.core import new_id  # noqa: E402
from ore_point24.decomposition import TaskDecomposer  # noqa: E402
from ore_point24.orchestrator import UnifiedOREIntegrator  # noqa: E402
from ore_point24.task_requests import IntentInterpreter  # noqa: E402


# --- Tests 26-35: task planning and execution ---------------------------------

def test_26_dag_correctness():
    """26. A valid dependency graph is topologically ordered."""
    nodes = ["a", "b", "c", "d"]
    edges = {"a": (), "b": ("a",), "c": ("a",), "d": ("b", "c")}
    layers, cycle = p24_planning.kahn_layers(nodes, edges)
    assert cycle is False
    order = [node for layer in layers for node in layer.node_ids]
    assert order.index("a") < order.index("b")
    assert order.index("a") < order.index("c")
    assert order.index("b") < order.index("d")
    assert order.index("c") < order.index("d")
    # parallelizable layers: b and c share a layer
    assert any({"b", "c"} <= set(layer.node_ids) for layer in layers)


def test_27_cycle_detection():
    """27. Cyclic module dependencies produce a planning failure with diagnostics."""
    layers, cycle = p24_planning.kahn_layers(
        ["a", "b", "c"], {"a": ("b",), "b": ("c",), "c": ("a",)},
    )
    assert cycle is True
    assert layers == []  # no layer can execute: residual graph without zero in-degree
    # self-loop is also a cycle
    _, self_cycle = p24_planning.kahn_layers(["x"], {"x": ("x",)})
    assert self_cycle is True


def test_28_unavailable_provider():
    """28. Missing backend is reported as a gap, not replaced by a fabricated capability."""
    catalog = ContractCatalog()
    register_existing_contracts(catalog)
    registry = ModuleRegistry(catalog)
    from ore_point24.compatibility import register_point24_contracts
    register_point24_contracts(catalog)
    interpreter = IntentInterpreter()
    request = interpreter.interpret(
        "plan something", required_capabilities=("planning.graph_search",),
        expected_outputs=("P24-PLAN",),
    )
    decomposer = TaskDecomposer()
    graph = decomposer.decompose(request)
    validator = p24_planning.DependencyPlanValidator(registry)
    plan = validator.validate(graph)
    assert plan.executable is False
    assert any("no runnable provider" in gap.reason for gap in plan.gaps)
    assert all("fabricated" not in gap.reason for gap in plan.gaps)


def test_29_bounded_execution():
    """29. Maximum concurrent module executions never exceed the configured limit."""
    def slow():
        time.sleep(0.02)
        return ModuleResult("m", "1.0.0", ModuleStatus.SUCCESS)

    calls = {f"m{index}": slow for index in range(8)}
    report = p24_exec.execute_bounded(
        calls, config=p24_exec.BoundedExecutionConfig(max_workers=3),
    )
    assert report.max_concurrency_observed <= 3
    assert report.bounded_within_limits is True
    assert report.outcome_counts()["success"] == 8


def test_30_timeout_handling():
    """30. A module exceeding its time limit returns an explicit timeout/failure state."""
    def too_slow():
        time.sleep(0.05)
        return ModuleResult("slow", "1.0.0", ModuleStatus.SUCCESS)

    report = p24_exec.execute_bounded(
        {"slow": too_slow},
        config=p24_exec.BoundedExecutionConfig(max_workers=1, default_timeout_seconds=0.001),
    )
    outcome = report.outcomes[0]
    assert outcome.outcome == "failed"
    assert outcome.error_type == "TimeoutError"


def test_31_result_identity():
    """31. A module result with the wrong producer or run identity is rejected."""
    result = ModuleResult(
        "real.module", "1.0.0", ModuleStatus.SUCCESS,
        artifacts=(
            p24_artifacts.canonical_artifact(
                {"schema": "demo"}, run_id="other-run", producer_id="real.module",
            ),
        ),
    )
    ok, reason = p24_exec.validate_result_identity(
        result, expected_module_id="real.module", run_id="this-run",
    )
    assert ok is False
    assert "run identity" in reason
    ok2, _ = p24_exec.validate_result_identity(
        result, expected_module_id="wrong.module", run_id="other-run",
    )
    assert ok2 is False


def test_32_output_schema():
    """32. Malformed result artifacts are not published."""
    catalog = ContractCatalog()
    register_existing_contracts(catalog)
    from ore_point24.compatibility import register_point24_contracts
    register_point24_contracts(catalog)
    malformed = p24_artifacts.canonical_artifact(
        {"missing": "required fields"}, contract_id="P24-POINT-STATUS",
        run_id="r", producer_id="p",  # payload lacks point/package/state
    )
    result = catalog.validate_artifact(malformed)
    assert result.valid is False
    assert any("missing required payload field" in error for error in result.errors)


def test_33_partial_result():
    """33. Partial completion is retained as partial, not success."""
    result = ModuleResult("partial.module", "1.0.0", ModuleStatus.PARTIAL)
    assert p24_exec.normalize_outcome(result.status) == "partial"
    report = p24_exec.execute_bounded({"partial.module": lambda: result})
    assert report.outcome_counts()["partial"] == 1
    assert report.outcome_counts()["success"] == 0


def test_34_abstention():
    """34. A module may abstain without being mislabelled as a crash."""
    abstained = ModuleResult("careful.module", "1.0.0", ModuleStatus.ABSTAINED)
    report = p24_exec.execute_bounded({"careful.module": lambda: abstained})
    assert report.outcomes[0].outcome == "abstained"
    assert report.outcomes[0].error_type is None


def test_35_canonical_controller_compatibility():
    """35. Task delegation does not silently change the established run path."""
    # The end-to-end demo task runs through the canonical
    # ore_contracts.CognitiveTaskCoordinator (B-N9 compatible), never a
    # replacement executor.
    integrator = UnifiedOREIntegrator(repository_root=str(REPO_ROOT))
    assert type(integrator.coordinator) is CognitiveTaskCoordinator
    assert integrator.coordinator.__class__.__module__ == "ore_contracts.coordinator"


# --- Tests 36-43: reasoning, learning and memory -------------------------------

def test_36_independent_verification():
    """36. A proposed conclusion cannot pass a mandatory proof gate without a verifier result."""
    verifier = p24_reason.DeterministicPropositionalVerifier()
    obligation = p24_reason.ProofObligation(
        claim="~p", assumptions=("p",), formal_system="propositional",
    )
    outcome = verifier.check(obligation)
    assert outcome.status == "invalid"  # the verifier rejects the contradiction
    channel = p24_reason.ProposalChannel(verifier=verifier)
    proposal = p24_reason.LearnedProposal(
        proposal_id=new_id("prop"), producer_model="learner", model_version="1",
        input_digest="0" * 64, candidate="~p", score=0.99, calibration_evidence=False,
    )
    channel.submit(proposal)
    verification = channel.verify(
        channel.to_formal_obligation(proposal.proposal_id, assumptions=("p",))
    )
    assert verification.status == "invalid"
    # an unverified proposal never counts as a verified conclusion
    channel2 = p24_reason.ProposalChannel()
    channel2.submit(proposal)
    assert channel2.unverified_proposals() == (proposal.proposal_id,)


def test_37_evidence_dependence():
    """37. Correlated evidence is not counted as independent corroboration."""
    claim = p24_evidence.SemanticClaim("the bank sentence is ambiguous")
    claim.add_evidence(p24_evidence.EvidenceItem("source.a", "supports", 0.6, "observed"))
    claim.add_evidence(p24_evidence.EvidenceItem("source.a_wire", "supports", 0.6, "syndicated"),
                       dependence_group="wire_service")
    claim.add_evidence(p24_evidence.EvidenceItem("source.a_wire2", "supports", 0.6, "syndicated"),
                       dependence_group="wire_service")
    claim.add_evidence(p24_evidence.EvidenceItem("source.b", "supports", 0.6, "observed"))
    # four evidence items, but only three independent (the wire pair counts once)
    assert claim.independent_corroboration_count() == 3
    grouped_score = claim.support_score()
    ungrouped = p24_evidence.SemanticClaim("the bank sentence is ambiguous")
    for source in ("source.a", "source.a_wire", "source.a_wire2", "source.b"):
        ungrouped.add_evidence(p24_evidence.EvidenceItem(source, "supports", 0.6, "observed"))
    # counting the correlated pair separately would inflate the score
    assert grouped_score < ungrouped.support_score()


def test_38_uncertainty_visibility():
    """38. Uncalibrated model scores are not exposed as calibrated probabilities."""
    proposal = p24_reason.LearnedProposal(
        proposal_id=new_id("prop"), producer_model="learner", model_version="1",
        input_digest="0" * 64, candidate="p", score=0.85, calibration_evidence=False,
    )
    assert proposal.to_dict()["score_is_calibrated_probability"] is False
    calibrated = p24_reason.LearnedProposal(
        proposal_id=new_id("prop"), producer_model="learner", model_version="1",
        input_digest="0" * 64, candidate="p", score=0.85, calibration_evidence=True,
    )
    assert calibrated.to_dict()["score_is_calibrated_probability"] is True


def test_39_candidate_isolation():
    """39. Learned rules and model-generated semantic claims remain candidates until authorized validation."""
    claim = p24_evidence.SemanticClaim("a model proposed this relation")
    claim.add_evidence(p24_evidence.EvidenceItem("model.output", "supports", 0.7, "model_proposal"))
    assert claim.status == "hypothesis"  # never auto-promoted
    claim.promote("verified")  # only an explicit, recorded transition
    assert claim.status == "verified"
    # candidates from discovery stay in the isolated registry until evaluated
    registry = p24_learn.CandidateStrategyRegistry()
    candidate = p24_learn.StrategyCandidate(
        candidate_id="cand-1", task_family="planning", origin="evolution",
        representation="tiny-program",
    )
    registry.register(candidate)
    assert registry.evaluations() == {}
    report = registry.evaluate(
        "cand-1",
        evaluator=lambda c: {"accuracy": 0.9},
        baseline={"accuracy": 0.8},
    )
    assert report["repeatable_gain"] is True


def test_40_learning_transaction_safety():
    """40. The parent model remains unchanged during candidate learning and evaluation."""
    split = p24_learn.DataSplit.create(
        train_ids=["t1"], validation_ids=["v1"], held_out_ids=["h1"], source_lineage="test",
    )
    manager = p24_learn.LearningTransactionManager()
    manager.register_family("family", "parent_fp")
    failing_gate = p24_learn.RegressionGate("safety", "violations", 0.0, 1.0, 0.0, "at_most")
    rejected = manager.evaluate(
        family="family", candidate_id="c1", candidate_fingerprint="cand_fp",
        split=split, gates=[failing_gate],
    )
    assert rejected.decision == "rejected"
    assert manager.active_model("family") == "parent_fp"  # parent untouched


def test_41_regression_gate():
    """41. A candidate failing any mandatory regression gate is not promoted."""
    split = p24_learn.DataSplit.create(
        train_ids=["t1"], validation_ids=["v1"], held_out_ids=["h1"], source_lineage="test",
    )
    manager = p24_learn.LearningTransactionManager()
    manager.register_family("family", "parent_fp")
    gates = (
        p24_learn.RegressionGate("accuracy", "acc", 0.80, 0.90, 0.0),  # improvement
        p24_learn.RegressionGate("safety", "violations", 0.0, 3.0, 0.0, "at_most"),  # violation
    )
    record = manager.evaluate(
        family="family", candidate_id="c2", candidate_fingerprint="cand_fp2",
        split=split, gates=gates,
    )
    assert record.decision == "rejected"
    assert "safety" in record.reason
    assert manager.active_model("family") == "parent_fp"


def test_42_memory_provenance():
    """42. Semantic, episodic and procedural memory entries preserve source and version references."""
    store = p24_memory.MemoryStore()
    record = p24_memory.MemoryRecord(
        "semantic", "integration preserves provenance", "artifact_source_id",
        model_version="model-1.2.0",
    )
    store.add(record)
    data = record.to_dict()
    assert data["source_artifact"] == "artifact_source_id"
    assert data["model_version"] == "model-1.2.0"
    assert data["version"] >= 1
    assert data["created_at"]
    assert store.counts()["semantic"] == 1
    # a vector match is only a retrieval hint, never a verification result
    retriever = p24_memory.MemoryRetriever(store)
    for item in retriever.retrieve("integration provenance"):
        assert item.authoritative == (store.get(item.record_id).status == "verified")


def test_43_approximate_index_safety():
    """43. Bloom or LSH candidate hits are checked against authoritative records before critical decisions."""
    bloom = p24_prob.BloomFilter(expected_items=32, false_positive_rate=0.05)
    bloom.add("record:known")
    assert bloom.might_contain("record:known") is True
    # A positive answer is a candidate: exact lookup must confirm.
    authoritative = {"record:known": "verified"}
    def exact_lookup(candidate_id):
        return authoritative.get(candidate_id)
    def critical_decision(candidate_id):
        assert bloom.might_contain(candidate_id)  # pre-filter
        exact = exact_lookup(candidate_id)
        if exact is None:
            return "rejected_not_authoritative"
        return "accepted"
    assert critical_decision("record:known") == "accepted"
    # false positives are expected and must fail the exact re-check
    for probe in range(1000):
        candidate = f"record:absent-{probe}"
        if bloom.might_contain(candidate):
            assert critical_decision(candidate) == "rejected_not_authoritative"
            break


# --- Tests 44-50: resource, security and reliability ----------------------------

def test_44_hard_resource_limits():
    """44. No learned policy can exceed the task's configured CPU, memory or time budget."""
    budget = ResourceBudget(
        max_workers=4, reserved_workers=1, wall_clock_seconds=10.0, max_memory_mb=100.0,
    )
    controller = DeterministicAdmissionController(budget)
    # A "learned" high score cannot even be submitted: admission is a pure
    # function of requirement vs budget (no score input exists by design).
    decision = controller.request("greedy_task", workers=8, wall_clock_seconds=2.0,
                                   memory_mb=10.0, optional=True)
    assert decision.decision == "deferred"
    assert "worker budget exceeded" in decision.reasons[0]
    mandatory = controller.request("greedy_mandatory", workers=8, wall_clock_seconds=2.0,
                                   memory_mb=10.0, optional=False)
    assert mandatory.decision == "rejected"
    # a learned MAPE-K policy also cannot bypass the hard ceilings
    gate = p24_mapek.PredictivePolicyGate()
    promotion = gate.evaluate_promotion("learned-policy")
    assert promotion.deterministic_override_available is True


def test_45_measured_cost_integrity():
    """45. Absent telemetry is stored as unavailable, not invented."""
    from ore_point24 import telemetry as p24_telemetry
    telemetry = p24_telemetry.ResourceTelemetry()
    measurement = telemetry.measure("m", "w", lambda: None)
    assert measurement.wall_clock_seconds != "unavailable"
    assert measurement.accelerator_utilization == "unavailable"
    assert measurement.network_io_bytes == "unavailable"
    assert measurement.average_memory_mb == "unavailable"


def test_46_reliability_evidence():
    """46. Module reliability changes only after an actual recorded outcome."""
    facade = p24_reliability.MeasuredReliabilityFacade(min_samples=3)
    try:
        before = facade.estimate("family", "module")
        assert before.point_estimate == 0.5  # Beta(2,2) prior
        assert before.evidence_state == "insufficient_observations"
        facade.record_outcome("family", "module", success=True)
        after_one = facade.estimate("family", "module")
        assert after_one.point_estimate > 0.5
        assert after_one.sample_count == 1
        for _ in range(3):
            facade.record_outcome("family", "module", success=True)
        measured = facade.estimate("family", "module")
        assert measured.sample_count == 4
        assert measured.evidence_state == "measured"
    finally:
        facade.close()


def test_47_retry_idempotence():
    """47. Retrying a task does not duplicate an externally visible side effect."""
    store = p24_recovery.IdempotencyKeyStore()
    side_effects: list[int] = []
    key = store.key_for("publish_result", {"artifact": "a1"})
    status_one, _ = store.execute_once(key, lambda: side_effects.append(1))
    status_two, _ = store.execute_once(key, lambda: side_effects.append(1))
    assert status_one == "applied"
    assert status_two == "duplicate"
    assert side_effects == [1]


def test_48_circuit_breaker():
    """48. A repeatedly failing module is excluded while its breaker is open."""
    facade = p24_reliability.MeasuredReliabilityFacade()
    try:
        for _ in range(3):
            facade.record_outcome("family", "flaky.module", success=False)
        assert facade.circuit_open("family", "flaky.module") is True
        assert facade.circuit_open("family", "healthy.module") is False
    finally:
        facade.close()


def test_49_sandbox_isolation():
    """49. Restricted execution cannot access undeclared filesystem or network resources."""
    authority = CapabilityAuthority(
        policy_permissions=("filesystem.read", "filesystem.write"),
    )
    enforcer = SandboxEnforcer(authority)
    spec = enforcer.build_spec(
        requested=("filesystem.read", "filesystem.write", "network.access"),
        granted=("filesystem.read", "filesystem.write", "network.access"),
        read_only_inputs=("/data/inputs",),
        writable_outputs=("/tmp/out",),
        wall_time_limit_seconds=5.0,
    )
    # network was granted by the operator but denied by policy
    assert enforcer.check_access(spec, permission="network.access") is False
    # undeclared filesystem paths are denied
    assert enforcer.check_access(spec, permission="filesystem.read", path="/etc/passwd") is False
    assert enforcer.check_access(spec, permission="filesystem.read", path="/data/inputs/file") is True
    assert enforcer.check_access(spec, permission="filesystem.write", path="/tmp/out/file") is True
    assert enforcer.check_access(spec, permission="filesystem.write", path="/data/inputs/file") is False
    # a failure to enforce a requested limit is an admission failure
    unbounded = enforcer.build_spec(
        requested=("filesystem.read",), granted=("filesystem.read",),
    )
    ok, _ = enforcer.limits_enforceable(unbounded)
    assert ok is False


def test_50_rollback_and_audit():
    """50. A failed release gate restores the previous known-good policy/model and records the decision."""
    controller = p24_recovery.PromotionController()
    controller.register("policy", "known_good_v1")
    promoted = controller.promote(
        "policy", candidate_fingerprint="candidate_v1",
        gates={"regression": True, "safety": True}, approved_by="release_manager",
    )
    assert promoted.decision == "promoted"
    assert controller.active_version("policy") == "candidate_v1"
    rollback = controller.rollback("policy", approved_by="release_manager")
    assert rollback is not None
    assert rollback.decision == "rolled_back"
    assert controller.active_version("policy") == "known_good_v1"
    history = controller.history("policy")
    assert [record.decision for record in history] == ["promoted", "rolled_back"]
    # a failed gate never overwrites the known-good version
    rejected = controller.promote(
        "policy", candidate_fingerprint="bad_candidate",
        gates={"regression": True, "safety": False}, approved_by="release_manager",
    )
    assert rejected.decision == "rejected"
    assert controller.active_version("policy") == "known_good_v1"


# --- Bayesian / causal / search acceptance (Steps 23-24 coverage) ----------------

def test_bayesian_update_records_assumptions():
    reasoner = p24_uncertainty.BayesianReasoner()
    model = p24_uncertainty.BayesianModel(2, 2, assumptions=("exchangeable trials",))
    update = reasoner.update("h", model, successes=8, failures=2, evidence_summary="10 trials")
    assert update.assumptions == ("exchangeable trials",)
    assert abs(update.posterior_mean - 10 / 14) < 1e-9
    with pytest.raises(ValueError, match="correlated evidence"):
        reasoner.update("h2", model, successes=3, failures=0,
                        evidence_summary="syndicated", evidence_dependent=True)


def test_causal_confounding_blocks_intervention_claim():
    structure = p24_uncertainty.CausalStructure(
        variables=("X", "Y", "Z"), edges=(("X", "Y"), ("Z", "X"), ("Z", "Y")),
    )
    result = p24_uncertainty.CausalAnalyst().query(
        structure, treatment="X", outcome="Y", observed_correlation=0.7,
    )
    assert result.estimable is False
    assert "not an intervention effect" in result.notes[0]


def test_search_methods_honour_their_assumptions():
    graph = {"a": {"b": 1.0, "c": 4.0}, "b": {"c": 2.0}, "c": {}}
    path, cost = p24_search.dijkstra(graph, "a", "c")
    assert path == ["a", "b", "c"] and cost == 3.0
    with pytest.raises(ValueError, match="non-negative"):
        p24_search.dijkstra({"a": {"b": -1.0}}, "a", "b")
    adversarial = p24_search.minimax(
        actions=lambda s: [0, 1],
        transition=lambda s, a: (1 - s[0], s[1] + [a]),
        utility=lambda s: 1.0 if (len(s[1]) >= 2 and s[1][-1] == 1) else 0.0,
        terminal=lambda s: len(s[1]) >= 2,
        depth=2, state=(0, []),
    )
    assert adversarial.best_action is not None
    _, trajectory = p24_search.dynamic_programming(
        stages=2, state_list=["L", "H"],
        value_terminal=lambda s: 0.0 if s == "L" else 2.0,
        transition_cost=lambda i, f, t: 1.0 if (f, t) == ("H", "L") else 0.0,
    )
    assert len(trajectory) == 3


def test_merkle_and_content_addressing():
    from ore_point24.integrity import ContentAddressedStore, MerkleTree
    tree = MerkleTree()
    for key in ("a", "b", "c", "d"):
        tree.add_leaf(key, f"value-{key}")
    root = tree.root_digest()
    assert tree.verify("b", "value-b") is True
    assert tree.verify("b", "tampered") is False
    tree.add_leaf("a", "value-changed")
    assert tree.root_digest() != root
    store = ContentAddressedStore()
    first = store.put("content-one")
    second = store.put("content-two", parents=(first,))
    assert store.parents(second) == (first,)
    assert store.get(first) == "content-one"
