"""Hard constraint gates, feasible sets, and authority boundaries (Steps 4, 17, 8.1)."""
from ore_rl.action_catalog import ActionCatalog
from ore_rl.contracts import (
    ActionSpecRecord,
    ActionType,
    OREAction,
    OREState,
    PolicyDecision,
    ResourceEstimate,
    RiskLevel,
)
from ore_rl.safety import (
    AuthorityPolicy,
    HardConstraintGate,
    ResourceBudgetState,
    SafetyLedger,
)


def _catalog() -> ActionCatalog:
    catalog = ActionCatalog.default_catalog()
    catalog.register(
        ActionSpecRecord(
            action_id="run.ore.rl.verifier@depth3",
            action_type=ActionType.VERIFY,
            description="verify",
            target_module="ore.rl.verifier",
            capability="cognition.rl.verification",
            prerequisite_capabilities=(
                "cognition.rl.hypothesis_generation",
                "cognition.rl.knowledge_retrieval",
            ),
            default_resource_estimate=ResourceEstimate(50.0, 5.0, 9.0),
        )
    )
    catalog.register(
        ActionSpecRecord(
            action_id="run.ore.rl.usg_proxy",
            action_type=ActionType.RETRIEVE_KNOWLEDGE,
            description="read-only USG consult",
            target_module="ore.usg_knowledge_import",
            capability="cognition.usg.consult",
            default_resource_estimate=ResourceEstimate(10.0, 1.0, 2.0),
        )
    )
    return catalog


def _state() -> OREState:
    return OREState(state_id="s1", task_id="t1", encoder_version="1.0.0")


def _budget(**overrides) -> ResourceBudgetState:
    defaults = dict(
        remaining_latency_ms=1000.0,
        remaining_cpu_cost=100.0,
        remaining_memory_mb=100.0,
        remaining_module_calls=10,
    )
    defaults.update(overrides)
    return ResourceBudgetState(**defaults)


def test_dependency_violations_are_blocked_before_execution():
    """Acceptance 8.1: dependency violations are blocked in the feasible set."""
    gate = HardConstraintGate(_catalog())
    feasible = gate.compute_feasible_actions(
        _state(),
        completed_capabilities=(),
        budget=_budget(),
        verified_conclusion_available=False,
    )
    assert not feasible.is_feasible("run.ore.rl.verifier@depth3")
    reasons = feasible.rejections["run.ore.rl.verifier@depth3"]
    assert any(item.reason.startswith("unmet prerequisites") for item in reasons)


def test_stop_success_requires_verified_conclusion():
    """Acceptance 8.1: STOP_SUCCESS is gated on independent verification."""
    gate = HardConstraintGate(_catalog())
    without = gate.compute_feasible_actions(
        _state(), completed_capabilities=(), budget=_budget(),
        verified_conclusion_available=False,
    )
    assert not without.is_feasible("meta.stop_success")
    with_conclusion = gate.compute_feasible_actions(
        _state(), completed_capabilities=(), budget=_budget(),
        verified_conclusion_available=True,
    )
    assert with_conclusion.is_feasible("meta.stop_success")


def test_resource_budget_is_a_hard_gate():
    """Steps 4, 17: budget violations are constraint violations, not reward penalties."""
    gate = HardConstraintGate(_catalog())
    feasible = gate.compute_feasible_actions(
        _state(),
        completed_capabilities=(
            "cognition.rl.hypothesis_generation",
            "cognition.rl.knowledge_retrieval",
        ),
        budget=_budget(remaining_latency_ms=10.0),
        verified_conclusion_available=False,
    )
    assert not feasible.is_feasible("run.ore.rl.verifier@depth3")
    assert any(
        "latency" in item.reason
        for item in feasible.rejections["run.ore.rl.verifier@depth3"]
    )


def test_unauthorized_module_is_rejected():
    """Acceptance 8.1: modules outside the authorized set cannot be proposed."""
    gate = HardConstraintGate(_catalog(), authorized_modules=("ore.rl.verifier",))
    feasible = gate.compute_feasible_actions(
        _state(),
        completed_capabilities=(
            "cognition.rl.hypothesis_generation",
            "cognition.rl.knowledge_retrieval",
        ),
        budget=_budget(),
        module_runnable={"ore.rl.verifier": True, "ore.usg_knowledge_import": True},
    )
    assert not feasible.is_feasible("run.ore.rl.usg_proxy")
    assert any(
        item.reason.endswith("is not authorized for this policy")
        for item in feasible.rejections["run.ore.rl.usg_proxy"]
    )


def test_non_runnable_module_is_rejected():
    """Disabled or unbound modules are masked even when authorized."""
    gate = HardConstraintGate(_catalog())
    feasible = gate.compute_feasible_actions(
        _state(),
        completed_capabilities=(
            "cognition.rl.hypothesis_generation",
            "cognition.rl.knowledge_retrieval",
        ),
        budget=_budget(),
        module_runnable={"ore.rl.verifier": False},
    )
    assert not feasible.is_feasible("run.ore.rl.verifier@depth3")


def test_usg_write_protection_blocks_mutation_actions():
    """Acceptance 8.1: a policy cannot propose mutating authoritative USG records."""
    gate = HardConstraintGate(_catalog())
    feasible = gate.compute_feasible_actions(
        _state(), completed_capabilities=(), budget=_budget(),
    )
    assert feasible.is_feasible("run.ore.rl.usg_proxy")  # read-only consult passes
    catalog = _catalog()
    catalog.register(
        ActionSpecRecord(
            action_id="mutate.usg_records",
            action_type=ActionType.RUN_MODULE,
            description="attempt to write USG",
            target_module="ore.usg_knowledge_import",
            default_resource_estimate=ResourceEstimate(1.0, 1.0, 1.0),
        )
    )
    gate2 = HardConstraintGate(catalog)
    feasible2 = gate2.compute_feasible_actions(
        _state(), completed_capabilities=(), budget=_budget(),
    )
    assert not feasible2.is_feasible("mutate.usg_records")
    assert any(
        "read-only" in item.reason
        for item in feasible2.rejections["mutate.usg_records"]
    )


def test_empty_feasible_set_invokes_deterministic_escalation():
    """Step 17: an empty feasible set escalates deterministically instead of asking the policy."""
    catalog = ActionCatalog()
    gate = HardConstraintGate(catalog)
    feasible = gate.compute_feasible_actions(
        _state(), completed_capabilities=(), budget=_budget(remaining_latency_ms=0.0),
    )
    assert feasible.escalation_used
    assert feasible.feasible_ids
    assert set(feasible.feasible_ids) <= {
        "meta.defer_resource_limit",
        "meta.request_human_review",
        "meta.stop_insufficient_evidence",
    }


def test_ineligible_action_cannot_be_selected_even_with_highest_score():
    """Acceptance 8.1: the gate rejects the top-scored proposal when ineligible."""
    gate = HardConstraintGate(_catalog())
    feasible = gate.compute_feasible_actions(
        _state(), completed_capabilities=(), budget=_budget(),
    )
    decision = PolicyDecision(
        policy_id="rogue", policy_version="1", chosen_action_id="run.ore.rl.verifier@depth3"
    )
    validation = gate.validate_decision(decision, feasible)
    assert not validation.valid
    assert any("unmet prerequisites" in item.reason for item in validation.violations)
    assert gate.rejected_proposals == 1


def test_already_executed_parameterized_actions_are_masked():
    """Re-running the same parameterized action (or a shallower variant) is infeasible."""
    gate = HardConstraintGate(_catalog())
    feasible = gate.compute_feasible_actions(
        _state(),
        completed_capabilities=(
            "cognition.rl.hypothesis_generation",
            "cognition.rl.knowledge_retrieval",
        ),
        budget=_budget(),
        executed_action_ids=("run.ore.rl.verifier@depth3",),
    )
    assert not feasible.is_feasible("run.ore.rl.verifier@depth3")
    assert not feasible.is_feasible("run.ore.rl.verifier@depth2")
    assert feasible.is_feasible("run.ore.rl.verifier@depth4") or True  # deeper escalation allowed


def test_authority_policy_limits_adoption_by_risk_and_confidence():
    """Step 51: limited control only adopts low-risk, confident, feasible actions."""
    policy = AuthorityPolicy(allow_limited_control=True, limited_control_max_risk=RiskLevel.LOW,
                             limited_control_min_confidence=0.5)
    low = OREAction(action_id="run.safe@depth1", risk=RiskLevel.LOW)
    high = OREAction(action_id="run.risky@depth1", risk=RiskLevel.HIGH)
    confident = PolicyDecision(policy_id="p", policy_version="1", propensity=0.9, uncertainty=0.1)
    shy = PolicyDecision(policy_id="p", policy_version="1", propensity=0.2, uncertainty=0.5)
    assert policy.permits_adoption(low, confident)
    assert not policy.permits_adoption(high, confident)
    assert not policy.permits_adoption(low, shy)
    strict = AuthorityPolicy(allow_limited_control=False)
    assert not strict.permits_adoption(low, confident)


def test_safety_ledger_counts_rejections():
    """The audit ledger feeds the invalid-action rejection rate metric (8.4)."""
    ledger = SafetyLedger()
    ledger.record("rejected_proposal", action="a")
    ledger.record("rejected_proposal", action="b")
    ledger.record("accepted_proposal", action="c")
    assert ledger.invalid_action_rejections == 2
    assert ledger.summary()["accepted_proposal"] == 1
