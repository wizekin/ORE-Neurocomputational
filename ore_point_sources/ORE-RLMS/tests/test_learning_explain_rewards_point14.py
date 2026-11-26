"""Acceptance tests: SHAP/LIME/counterfactuals and PRM/RLVR validation (Steps 27-37)."""
from __future__ import annotations

import numpy as np
import pytest

from ore_learning.config import LearningConfig
from ore_learning.explainability import (
    ExplanationChannelError,
    ExplanationRegistry,
    LimeExplainer,
    ShapExplainer,
    build_explanation_record,
    counterfactual,
    feature_importance_stability,
)
from ore_learning.rewards import (
    PRMModel,
    ReasoningStep,
    ReasoningTrace,
    StepKind,
    apply_validators,
    compute_rlvr_reward,
    replay_reward,
    split_visible_hidden,
    IndependentTestSuite,
)

WEIGHTS = np.array([1.5, -2.0, 0.5, 1.0])
INSTANCE = np.array([0.5, -0.3, 1.2, -0.8])


def linear_model(x) -> float:
    """A scalar model with known ground-truth attributions."""
    return float(np.asarray(x, dtype=float) @ WEIGHTS)


@pytest.fixture()
def background() -> np.ndarray:
    """A seeded background distribution for SHAP."""
    return np.random.default_rng(2).normal(size=(16, 4))


def test_shap_report_includes_explainer_configuration_and_background(background):
    """Acceptance: SHAP report includes explainer configuration and background reference."""
    explainer = ShapExplainer()
    explanation = explainer.explain(linear_model, INSTANCE, background,
                                    feature_names=["a", "b", "c", "d"],
                                    background_ref="calibration:v3")
    record = explanation.to_dict()
    assert record["explainer_variant"] == "exact-enumeration"
    assert record["background_ref"] == "calibration:v3"
    assert record["background_size"] == 16
    assert record["approximation"] == "none (exact Shapley weights)"
    assert record["channel"] == "model_diagnostics"
    assert record["feature_names"] == ["a", "b", "c", "d"]
    assert "not causal proof" in record["note"]
    # large feature sets switch to the sampling variant with recorded settings
    wide = np.random.default_rng(5).normal(size=(8, 12))
    wide_model = lambda x: float(np.asarray(x) @ np.arange(1, 13))  # noqa: E731
    sampled = ShapExplainer().explain(wide_model, wide[0], wide)
    assert sampled.explainer_variant == "sampling-permutation"
    assert "sampling" in sampled.approximation


def test_shap_efficiency_property_holds_exactly(background):
    """Shapley efficiency: sum(phi) = f(x) - E[f(background)]."""
    explainer = ShapExplainer()
    explanation = explainer.explain(linear_model, INSTANCE, background)
    prediction = linear_model(INSTANCE)
    base = float(np.mean([linear_model(row) for row in background]))
    assert abs(float(np.sum(explanation.values)) - (prediction - base)) < 1e-9
    assert abs(explanation.efficiency_residual) < 1e-9
    # for a linear model, phi_i = w_i * (x_i - E[background_i])
    expected = WEIGHTS * (INSTANCE - background.mean(axis=0))
    np.testing.assert_allclose(explanation.values, expected, atol=1e-9)


def test_lime_report_includes_perturbation_method_and_local_fidelity():
    """Acceptance: LIME report includes perturbation method and local-fidelity measure."""
    explainer = LimeExplainer()
    explanation = explainer.explain(linear_model, INSTANCE, noise_scale=0.3, seed=13)
    record = explanation.to_dict()
    assert "gaussian" in record["perturbation_method"]
    assert "scale=0.3" in record["perturbation_method"]
    assert "seed=13" in record["perturbation_method"]
    assert record["local_fidelity"] > 0.99
    assert record["channel"] == "model_diagnostics"
    assert record["n_perturbations"] == LearningConfig().lime_num_samples
    assert "manifold" in record["note"]


def test_lime_recovers_linear_coefficients():
    """The weighted surrogate recovers the true linear coefficients."""
    explanation = LimeExplainer().explain(linear_model, INSTANCE, noise_scale=0.4,
                                          seed=5)
    np.testing.assert_allclose(explanation.surrogate_coefficients, WEIGHTS, atol=0.15)
    assert explanation.local_fidelity > 0.99


def test_explanation_diagnostics_are_never_recorded_as_formal_proof(background):
    """Acceptance: explanation diagnostics are never recorded as formal proof."""
    registry = ExplanationRegistry()
    shap = ShapExplainer().explain(linear_model, INSTANCE, background)
    lime = LimeExplainer().explain(linear_model, INSTANCE)
    cf = counterfactual(linear_model, INSTANCE, direction=-1,
                        mutable=[True, True, False, False])
    for record in (build_explanation_record(shap=shap),
                   build_explanation_record(lime=lime),
                   build_explanation_record(counterfactual_report=cf)):
        registry.register(record)
        assert record.is_formal_proof is False
        with pytest.raises(ExplanationChannelError, match="formal proof"):
            registry.register_as_proof(record)
    native = build_explanation_record(native_trace_ref="trace:task-42")
    registry.register(native)
    assert native.channel == "native_trace"
    summary = registry.describe()
    assert summary["channels"] == {"model_diagnostics": 3, "native_trace": 1}
    assert summary["formal_proof_records"] == 0


def test_counterfactuals_respect_domain_constraints():
    """Step 30: minimal counterfactuals under mutability constraints and feasibility."""
    report = counterfactual(linear_model, INSTANCE, direction=-1,
                            mutable=[True, True, False, False], step=1.0)
    assert report.found
    assert report.constraints_respected
    assert report.n_features_changed <= 2
    # immutable features never move
    original = INSTANCE.copy()
    changed = np.where(np.abs(report.counterfactual - original) > 1e-9)[0]
    assert set(changed.tolist()) <= {0, 1}
    # a fully immutable instance yields no counterfactual
    frozen = counterfactual(linear_model, INSTANCE, direction=-1,
                            mutable=[False] * 4)
    assert not frozen.found and frozen.counterfactual is None
    assert "causality" in frozen.note


def test_feature_importance_stability_across_samples():
    """Step 30: global importance is tested for stability across samples."""
    rng = np.random.default_rng(4)
    samples = rng.normal(size=(8, 4))

    def importance_fn(batch):
        return np.abs(WEIGHTS)  # stable importance

    stable = feature_importance_stability(importance_fn, samples)
    assert stable["stable"] and stable["stability"] > 0.9

    flip = {0: True}
    unstable = feature_importance_stability(
        lambda batch: (np.abs(WEIGHTS)[::-1] if batch[0][0] > 0
                       else np.abs(WEIGHTS)), samples)
    assert unstable["stability"] < stable["stability"]


def build_trace() -> tuple[ReasoningTrace, dict]:
    """A trace with two verified steps, one failing, one model-assessed."""
    trace = ReasoningTrace(task_id="task-v")
    trace.add(ReasoningStep("s1", StepKind.CALCULATION, "sum identity",
                            ("in:a",), ("identity",),
                            validator_name="arithmetic-identity"))
    trace.add(ReasoningStep("s2", StepKind.RULE_APPLICATION, "bounds",
                            ("in:b",), ("bound",),
                            validator_name="constraint-check"))
    trace.add(ReasoningStep("s3", StepKind.CALCULATION, "broken sum",
                            ("in:c",), ("identity",),
                            validator_name="arithmetic-identity"))
    trace.add(ReasoningStep("s4", StepKind.CONCLUSION, "open judgement",
                            ("in:d",), ("judgement",)))
    payloads = {
        "s1": {"a": 5.0, "b": 3.0, "c": 2.0},
        "s2": {"constraints": {"x": 4.0}, "values": {"x": 3.5}},
        "s3": {"a": 5.0, "b": 3.0, "c": 1.0},  # 5 != 3 + 1 -> fail
    }
    return trace, payloads


def test_prm_scores_stay_separate_from_validator_results():
    """Acceptance: PRM scores remain separate from deterministic validator results."""
    trace, payloads = build_trace()
    report = apply_validators(trace, payloads)
    prm = PRMModel()
    screening = prm.screen(trace)
    # PRM scores live in their own field and diagnostics, never in validator results
    for step in trace.steps:
        assert isinstance(step.prm_score, float) or step.prm_score is None
        assert step.validator_result in ("pass", "fail", "unavailable")
        assert step.prm_score is None or step.prm_score != step.validator_result
    diagnostics = trace.prm_diagnostics()
    assert len(diagnostics) == len(trace.steps)
    assert all(diagnostic.prm_model_id == "prm.reference@1.0.0"
               for diagnostic in diagnostics)
    # the validation report contains no PRM artefacts
    assert all("prm" not in check.validator.lower() for check in report.checks)
    assert screening["calibration_metadata"]["calibrated"] is False
    # a high PRM score cannot repair a failed validator
    failed_step = trace.steps[2]
    assert failed_step.prm_score >= 0.5 and failed_step.status.value == "failed"


def test_failed_intermediate_validation_blocks_verified_status():
    """Acceptance: failed intermediate validation blocks verified status."""
    trace, payloads = build_trace()
    report = apply_validators(trace, payloads)
    assert not trace.verified
    assert not report.verified
    assert [check.check_id for check in report.failed_checks] == ["s3"]
    assert trace.steps[2].status.value == "failed"
    assert trace.steps[3].status.value == "model_assessed"
    # a fully passing trace verifies
    clean = ReasoningTrace(task_id="task-clean")
    clean.add(ReasoningStep("s1", StepKind.CALCULATION, "sum",
                            ("in:a",), ("identity",),
                            validator_name="arithmetic-identity"))
    clean_report = apply_validators(clean, {"s1": {"a": 5.0, "b": 3.0, "c": 2.0}})
    assert clean.verified and clean_report.verified


def test_rlvr_reward_is_reproducible_from_validator_outputs():
    """Acceptance: RLVR reward is reproducible from validator outputs."""
    trace, payloads = build_trace()
    record = compute_rlvr_reward(trace, payloads, config=LearningConfig())
    # reward provenance preserves validator identity, versions, environment
    assert record.validator_identity == ("arithmetic-identity",
                                          "constraint-check",
                                          "arithmetic-identity", "none")
    assert record.test_versions == ("pass", "pass", "fail", "unavailable")
    assert record.environment.startswith("ore_learning.rewards")
    assert record.violation_penalty == 1.0
    assert record.reward < 0  # the violation outweighs the passing steps
    # recomputing from the same validator outputs reproduces digest and value
    assert replay_reward(record, trace, payloads, config=LearningConfig())
    # a tampered replay (different payloads) does not
    tampered = dict(payloads)
    tampered["s3"] = {"a": 5.0, "b": 3.0, "c": 2.0}  # now passes
    assert not replay_reward(record, trace, tampered, config=LearningConfig())


def test_reward_is_a_training_signal_not_a_correctness_record():
    """Step 36: rewards never assert correctness; verified status stays separate."""
    trace, payloads = build_trace()
    record = compute_rlvr_reward(trace, payloads)
    assert "not an authoritative fact" in record.note
    # a positive reward does not make the trace verified
    clean = ReasoningTrace(task_id="task-clean")
    clean.add(ReasoningStep("s1", StepKind.CALCULATION, "sum",
                            ("in:a",), ("identity",),
                            validator_name="arithmetic-identity"))
    clean_record = compute_rlvr_reward(clean, {"s1": {"a": 5.0, "b": 3.0, "c": 2.0}})
    assert clean_record.reward > 0 and clean.verified
    # but an unverified trace can still produce a computed reward (training signal)
    assert record.reward < 0 and not trace.verified


def test_independent_checks_detect_simple_reward_hacking():
    """Acceptance: independent checks detect simple reward-hacking cases."""
    cases = [(index, index * 2) for index in range(24)]
    suite = IndependentTestSuite(cases, lambda candidate, case:
                                 candidate(case[0]) == case[1])
    visible, hidden = split_visible_hidden(cases)
    assert len(visible) + len(hidden) == len(cases)
    assert len(hidden) == int(round(len(cases) * LearningConfig().hidden_test_fraction))
    # an honest candidate passes both partitions and is certified
    honest = suite.evaluate(lambda x: x * 2, candidate_id="honest")
    assert honest.certified and not honest.hacking_detected
    # a candidate that games the visible partition is caught by hidden tests
    visible_inputs = {case[0] for case in visible}
    gamer = lambda x: x * 2 if x in visible_inputs else -1  # noqa: E731
    hacked = suite.evaluate(gamer, candidate_id="gamer")
    assert hacked.hacking_detected or hacked.overfit_suspected
    assert not hacked.certified
    assert hacked.hidden_pass_rate < honest.hidden_pass_rate
    # a merely overfit candidate (partial hidden degradation) is not certified either
    sloppy = lambda x: x * 2 if x < 15 else 0  # noqa: E731
    sloppy_verdict = suite.evaluate(sloppy, candidate_id="sloppy")
    assert not sloppy_verdict.certified
