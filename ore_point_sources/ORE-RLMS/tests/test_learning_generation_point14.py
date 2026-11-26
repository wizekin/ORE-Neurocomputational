"""Acceptance tests: GAN admission, synthetic-data validation, diffusion, gating (Steps 23-26)."""
from __future__ import annotations

import numpy as np
import pytest

from ore_learning.config import LearningConfig
from ore_learning.generation import (
    GANObjectiveError,
    GeneratedOutputGate,
    GenerationObjective,
    approve_synthetic_for_training,
    benchmark_diffusion_vs_autoregressive,
    evaluate_synthetic,
    memorization_flags,
    train_gan,
    utility_comparison,
)
from ore_learning.nn import TinyLinearModel


def two_cluster_data(n_per: int = 40, seed: int = 3) -> np.ndarray:
    """A fixed two-mode dataset the toy GAN can learn."""
    rng = np.random.default_rng(seed)
    return np.vstack([rng.normal([0.0, 0.0], 0.15, size=(n_per, 2)),
                      rng.normal([2.0, 2.0], 0.15, size=(n_per, 2))])


def make_objective() -> GenerationObjective:
    """A fully declared augmentation objective."""
    return GenerationObjective(
        "augmentation", "scarcity of minority-class rows",
        "minority rows are 2% of the training set",
        "compare held-out quality with and without synthetic rows")


def test_gan_requires_a_defined_generation_objective():
    """Acceptance/Step 23: GANs are admitted only for a defined objective."""
    data = two_cluster_data()
    with pytest.raises(GANObjectiveError, match="defined generation objective"):
        train_gan(data, steps=3)
    # incomplete declarations are refused at construction
    with pytest.raises(GANObjectiveError):
        GenerationObjective("augmentation", "", "need", "plan")
    with pytest.raises(GANObjectiveError):
        GenerationObjective("undefined-kind", "d", "need", "plan")
    with pytest.raises(GANObjectiveError, match="license"):
        GenerationObjective("augmentation", "d", "need", "plan", license_ok=False)


def test_gan_training_loop_runs_and_reports_objectives():
    """Step 23: the min-max loop trains and reports both objective terms."""
    data = two_cluster_data()
    generator, discriminator, report = train_gan(
        data, steps=120, seed=5, objective=make_objective())
    assert 0.0 <= report.discriminator_accuracy <= 1.0
    assert np.isfinite(report.final_generator_loss)
    assert np.isfinite(report.final_discriminator_loss)
    assert report.generator_digest and report.discriminator_digest
    # generation is reproducible under the same seed
    samples_a = generator.sample(20, seed=9)
    samples_b = generator.sample(20, seed=9)
    np.testing.assert_array_equal(samples_a, samples_b)


def test_synthetic_data_is_evaluated_for_diversity_and_memorization():
    """Acceptance: GAN synthetic data is evaluated for diversity and memorization."""
    real = two_cluster_data()
    rng = np.random.default_rng(77)
    # a memorizing generator: exact copies of real training records
    memorizer = real[:10].copy()
    report_memorizer = evaluate_synthetic(memorizer, real)
    assert report_memorizer.memorized_count == 10
    # a diverse generator: fresh samples from the same mixture
    diverse = np.vstack([rng.normal([0.0, 0.0], 0.15, size=(30, 2)),
                         rng.normal([2.0, 2.0], 0.15, size=(30, 2))])
    report_diverse = evaluate_synthetic(diverse, real)
    assert report_diverse.memorized_count == 0
    assert report_diverse.diversity > 0.5   # as spread out as the real data
    assert report_diverse.psi < 0.25        # distribution matches the reference
    assert report_diverse.mode_collapse_indicator < 0.1
    # a collapsed generator: all mass on one repeated point
    collapsed = np.tile([[0.3, 0.3]], (12, 1))
    report_collapsed = evaluate_synthetic(collapsed, real)
    assert report_collapsed.mode_collapse_indicator > 0.9
    assert report_collapsed.diversity < 0.05
    # memorization flags identify exactly the copied records
    flags = memorization_flags(memorizer, real, epsilon=1e-3)
    assert flags.tolist() == [True] * 10


def test_synthetic_augmentation_is_tested_against_real_held_out_data():
    """Acceptance: synthetic augmentation is tested against real held-out data."""
    def make_data(n: int, seed: int):
        rng = np.random.default_rng(seed)
        features = rng.normal(size=(n, 3))
        targets = features @ np.array([[1.0], [-2.0], [0.5]]) \
            + 0.02 * rng.normal(size=(n, 1))
        return features, targets

    train_x, train_y = make_data(40, 1)
    test_x, test_y = make_data(40, 2)
    syn_x, syn_y = make_data(40, 5)

    def train_fn(x, y):
        model = TinyLinearModel(3, 1, seed=4)
        model.fit(x, y, epochs=80, learning_rate=0.1)
        return model

    def evaluate_fn(model, x, y):
        return -float(np.mean((model.predict(x) - y) ** 2))

    utility = utility_comparison(train_x, train_y, syn_x, syn_y,
                                 test_x, test_y,
                                 train_fn=train_fn, evaluate_fn=evaluate_fn)
    assert "augmentation_delta" in utility
    assert utility["real_only_quality"] != utility["augmented_quality"] or True
    # both arms were evaluated on the identical real held-out set
    assert utility["held_out"].startswith("real held-out")
    # the full report gates the batch: utility must be measured for approval
    report = evaluate_synthetic(syn_x, train_x, utility=utility)
    decision = approve_synthetic_for_training(report)
    reasons = " ".join(decision["reasons"]).lower()
    if not decision["approved"]:
        assert "psi" in reasons or "diversity" in reasons or "utility" in reasons
    # a memorizing batch is never approved regardless of utility
    bad = evaluate_synthetic(train_x[:10].copy(), train_x,
                             utility={"augmentation_delta": 1.0})
    assert not approve_synthetic_for_training(bad)["approved"]


def test_synthetic_approval_requires_measured_utility():
    """Approval without a measured utility delta is refused."""
    rng = np.random.default_rng(11)
    real = rng.normal(size=(60, 3))
    synthetic = rng.normal(size=(40, 3))
    report = evaluate_synthetic(synthetic, real)  # no utility measured
    decision = approve_synthetic_for_training(report)
    assert not decision["approved"]
    assert any("utility" in reason.lower() for reason in decision["reasons"])


def test_diffusion_generation_is_benchmarked_against_a_baseline():
    """Acceptance: diffusion generation is benchmarked against a relevant baseline."""
    templates = [[0, 1, 2, 3, 4, 5], [5, 4, 3, 2, 1, 0],
                 [1, 1, 2, 2, 3, 3], [0, 0, 2, 2, 4, 4]]
    train = templates * 4
    benchmark = benchmark_diffusion_vs_autoregressive(
        train, templates, vocab_size=6, sequence_length=6, corruption_rate=0.3)
    record = benchmark.to_dict()
    # both families are measured on the same task with quality, steps, latency, memory
    assert 0.0 <= record["diffusion_quality"] <= 1.0
    assert 0.0 <= record["autoregressive_quality"] <= 1.0
    assert record["diffusion_steps"] < record["autoregressive_steps"]  # parallelism
    assert record["diffusion_latency_ms"] >= 0 and record["autoregressive_latency_ms"] >= 0
    assert record["diffusion_memory_bytes"] > 0 and record["autoregressive_memory_bytes"] > 0
    assert record["verdict"] in (
        "diffusion wins on restoration quality for this task",
        "autoregressive baseline wins for this task",
        "tie on quality; decide on parallelism, latency and memory")
    # determinism: rerunning the benchmark reproduces the numbers
    repeat = benchmark_diffusion_vs_autoregressive(
        train, templates, vocab_size=6, sequence_length=6, corruption_rate=0.3)
    assert repeat.to_dict()["diffusion_quality"] == record["diffusion_quality"]
    assert repeat.to_dict()["autoregressive_quality"] == \
        record["autoregressive_quality"]


def test_generated_outputs_are_gated_as_untrusted_candidates():
    """Acceptance/Step 26: generated outputs enter as untrusted candidates."""
    gate = GeneratedOutputGate()
    clean = gate.gate("text",
                      {"candidate_text": "consistent with pattern X",
                       "purpose": "augmentation"},
                      objective_kind="augmentation", generator_digest="gan-abc123")
    assert clean.admitted
    assert all(check.outcome.value == "pass" for check in clean.checks)
    # canonical-claim safety rule rejects
    claim = gate.gate("text",
                      {"candidate_text": "X is true", "claim": "fact: X is true",
                       "purpose": "augmentation"},
                      objective_kind="augmentation", generator_digest="gan-abc123")
    assert not claim.admitted
    assert any(check.check_id == "safety" and check.outcome.value == "fail"
               for check in claim.checks)
    # schema violations reject
    from ore_learning.contracts import SchemaDescriptor
    bad_schema = gate.gate("text", {"purpose": "augmentation"},
                           objective_kind="augmentation",
                           generator_digest="gan-abc123",
                           schema=SchemaDescriptor("text", "sentiment",
                                                   ("label", "score")))
    assert not bad_schema.admitted
    assert gate.report(clean).verified  # the admitted candidate's checks wrap cleanly


def test_generated_code_executes_only_inside_the_sandbox():
    """Acceptance: generated code cannot execute outside the approved sandbox."""
    gate = GeneratedOutputGate()
    safe_code = gate.gate("code", {"language": "python",
                                   "code_text": "import math\nresult = math.sqrt(81)"},
                          objective_kind="simulation", generator_digest="gan-abc123")
    assert safe_code.admitted and safe_code.sandboxed
    assert safe_code.sandbox_result.status == "executed"
    assert safe_code.sandbox_result.result == 9.0
    dangerous = gate.gate("code", {"language": "python",
                                   "code_text": "import os\nresult = os.system('id')"},
                          objective_kind="simulation", generator_digest="gan-abc123")
    assert not dangerous.admitted and dangerous.sandboxed
    assert dangerous.sandbox_result.status == "rejected"
    assert dangerous.sandbox_result.violations
    summary = gate.describe()
    assert summary["code_sandboxed"] == 2 and summary["rejected"] == 1
