"""Typed ore_rl contract schemas: version rejection, validation, hashing (Step 3)."""
import pytest

from ore_rl.contracts import (
    RL_SCHEMA_VERSION,
    ActionSpecRecord,
    ActionType,
    OREAction,
    OREState,
    OREReward,
    ORETransition,
    OutcomeStatus,
    ParameterSpec,
    PolicyDecision,
    PolicyAuthority,
    ResourceEstimate,
    RiskLevel,
    UnknownSchemaVersionError,
    canonical_json,
    require_schema_version,
    stable_hash,
)


def test_unknown_state_schema_version_is_rejected():
    """Acceptance 8.1: unknown schema versions are rejected, not interpreted permissively."""
    with pytest.raises(UnknownSchemaVersionError):
        OREState(schema_version="9.9.9", state_id="s", encoder_version="1")
    with pytest.raises(UnknownSchemaVersionError):
        require_schema_version("0.0.1", context="test")


def test_every_contract_type_rejects_unknown_versions():
    """All five contract families enforce the schema version gate."""
    with pytest.raises(UnknownSchemaVersionError):
        OREAction(schema_version="2.0.0", action_id="a")
    with pytest.raises(UnknownSchemaVersionError):
        OREReward(schema_version="2.0.0")
    with pytest.raises(UnknownSchemaVersionError):
        ORETransition(schema_version="2.0.0")
    with pytest.raises(UnknownSchemaVersionError):
        PolicyDecision(schema_version="2.0.0", policy_id="p", policy_version="1")


def test_round_trip_serialization_is_lossless():
    """State, action, and reward dictionaries round-trip through the schemas."""
    state = OREState(
        state_id="s1",
        task_id="t1",
        task_family="fact_verification",
        encoder_version="1.0.0",
        feature_names=("a", "b"),
        features={"a": 1.0, "b": 2.0},
        missing_features=("b",),
    )
    restored = OREState.from_dict(state.to_dict())
    assert restored == state
    action = OREAction(
        action_id="run.module@depth3",
        action_type=ActionType.RUN_MODULE,
        parameters={"depth": 3},
        target_module="module",
        resource_estimate=ResourceEstimate(1.0, 2.0, 3.0),
        risk=RiskLevel.MEDIUM,
    )
    assert OREAction.from_dict(action.to_dict()) == action
    reward = OREReward(quality_delta=0.5, total=0.4, weights_version="w", verified=True)
    assert OREReward.from_dict(reward.to_dict()) == reward


def test_hashing_is_deterministic_and_order_independent():
    """Canonical JSON hashing is stable and key-order independent."""
    left = stable_hash({"b": 2, "a": 1})
    right = stable_hash({"a": 1, "b": 2})
    assert left == right
    assert canonical_json({"b": 2, "a": 1}) == canonical_json({"a": 1, "b": 2})
    assert stable_hash({"a": 1}) != stable_hash({"a": 2})


def test_parameter_validation_rejects_bad_values():
    """Step 16: missing, out-of-range, and type errors are reported per parameter."""
    spec = ParameterSpec("depth", "int", required=True, minimum=1, maximum=5)
    assert spec.validate(3) is None
    assert spec.validate(None) is not None
    assert spec.validate(0) is not None
    assert spec.validate(6) is not None
    assert spec.validate("3") is not None
    choice = ParameterSpec("mode", "str", required=True, choices=("fast", "slow"))
    assert choice.validate("fast") is None
    assert choice.validate("medium") is not None


def test_action_stable_id_depends_on_parameters():
    """Two parameterizations of one action receive distinct stable identities."""
    base = dict(action_id="run.mod", action_type=ActionType.RUN_MODULE, target_module="mod")
    first = OREAction(parameters={"depth": 1}, **base)
    second = OREAction(parameters={"depth": 2}, **base)
    assert first.stable_id != second.stable_id
    assert OREAction(parameters={"depth": 1}, **base).stable_id == first.stable_id


def test_outcome_statuses_are_distinct():
    """Step 20 / 8.1: success, timeout, exception, and exhaustion stay distinguishable."""
    statuses = {item.value for item in OutcomeStatus}
    assert {"SUCCESS", "PARTIAL", "INVALID_OUTPUT", "TIMEOUT", "EXCEPTION", "CANCELLED",
            "RESOURCE_EXHAUSTED", "TERMINAL"} <= statuses


def test_policy_decision_requires_identity():
    """Decisions always carry policy id and version for the audit trail."""
    with pytest.raises(ValueError):
        PolicyDecision(policy_id="", policy_version="1")
    decision = PolicyDecision(policy_id="p", policy_version="1", authority=PolicyAuthority.SHADOW)
    assert decision.authority is PolicyAuthority.SHADOW
    assert decision.schema_version == RL_SCHEMA_VERSION


def test_action_spec_record_serializes():
    """Catalog specs serialize with their parameter declarations."""
    spec = ActionSpecRecord(
        action_id="run.x",
        action_type=ActionType.RUN_MODULE,
        description="x",
        parameters=(ParameterSpec("depth", "int", required=False, minimum=1, maximum=5),),
        target_module="x",
        capability="cognition.test",
        prerequisite_capabilities=("cognition.pre",),
        default_resource_estimate=ResourceEstimate(1.0, 1.0, 1.0),
        risk=RiskLevel.LOW,
    )
    data = spec.to_dict()
    assert data["parameters"][0]["name"] == "depth"
    assert data["prerequisite_capabilities"] == ["cognition.pre"]
