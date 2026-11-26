"""Acceptance tests: homomorphic encryption and differential privacy (FHE + Phase G)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from ore_security.contracts import PolicyDecision, PrivacyReleaseRequest
from ore_security.exceptions import (
    AuthorizationError,
    BudgetExceededError,
    HomomorphicEncryptionError,
    PrivacyError,
)
from ore_security.fhe import (
    PaillierContext,
    homomorphic_add,
    homomorphic_multiply,
    homomorphic_scalar_mul,
)
from ore_security.privacy import (
    GaussianMechanism,
    PrivacyAccountant,
    dp_mean,
    evaluate_utility,
    gaussian_sigma,
    subgroup_utility,
)


@pytest.fixture(scope="module")
def paillier():
    return PaillierContext("paillier-1024-test")


def test_fhe_adapter_rejects_unsupported_operations(paillier):
    """Acceptance: FHE adapter rejects unsupported operations and parameters."""
    c1 = paillier.encrypt(10)
    c2 = paillier.encrypt(20)
    with pytest.raises(HomomorphicEncryptionError):
        homomorphic_multiply(c1, c2)  # ciphertext x ciphertext is outside the algebra
    with pytest.raises(HomomorphicEncryptionError):
        PaillierContext("unsupported-parameter-set")
    with pytest.raises(HomomorphicEncryptionError):
        paillier.encrypt(-1)  # negative plaintext outside the message space


def test_fhe_output_remains_encrypted_until_authorized_decryption(paillier):
    """Acceptance: FHE output remains encrypted until authorized decryption."""
    c1 = paillier.encrypt(120)
    c2 = paillier.encrypt(30)
    total = homomorphic_scalar_mul(homomorphic_add(c1, c2), 3)
    denied = PolicyDecision(False, "p-1", "CLASSIFICATION_EXCEEDS_RULE")
    with pytest.raises(AuthorizationError):
        paillier.decrypt(total, denied)
    with pytest.raises(AuthorizationError):
        paillier.decrypt(total)  # no decision means no release
    allowed = PolicyDecision(True, "p-1", "ALLOWED_BY_RULE")
    assert paillier.decrypt(total, allowed) == 450


def test_fhe_evaluates_on_ciphertexts(paillier):
    """Eval(f, E(pk, x)) = E(pk, f(x)) holds for the supported algebra."""
    values = [3, 14, 15]
    ciphertexts = [paillier.encrypt(value) for value in values]
    total = ciphertexts[0]
    for ciphertext in ciphertexts[1:]:
        total = homomorphic_add(total, ciphertext)
    allowed = PolicyDecision(True, "p", "ALLOWED_BY_RULE")
    assert paillier.decrypt(total, allowed) == sum(values)
    assert paillier.decrypt(homomorphic_scalar_mul(total, 10), allowed) == sum(values) * 10


def test_dp_release_defines_privacy_unit_and_adjacency():
    """Acceptance: DP release defines protected privacy unit and adjacency relation."""
    with pytest.raises(ValueError):
        PrivacyReleaseRequest(
            "rel-1", "", "add/remove one person", "mean", 1.0, "Gaussian",
            1.0, 1e-6, "telemetry", "v1", calibration_reviewed=True,
        )
    with pytest.raises(ValueError):
        PrivacyReleaseRequest(
            "rel-1", "person", "", "mean", 1.0, "Gaussian",
            1.0, 1e-6, "telemetry", "v1", calibration_reviewed=True,
        )
    with pytest.raises(ValueError):
        PrivacyReleaseRequest(  # calibration must be explicitly reviewed
            "rel-1", "person", "add/remove one person", "mean", 1.0, "Gaussian",
            1.0, 1e-6, "telemetry", "v1", calibration_reviewed=False,
        )
    request = PrivacyReleaseRequest(
        "rel-1", "person", "add/remove one person", "mean latency", 2.0, "Gaussian",
        1.0, 1e-6, "telemetry", "v1", calibration_reviewed=True,
    )
    assert request.privacy_unit == "person"
    assert request.adjacency_relation == "add/remove one person"


def test_dp_mechanism_uses_reviewed_sensitivity_and_noise_calibration():
    """Acceptance: DP mechanism uses reviewed sensitivity and noise calibration."""
    sigma = gaussian_sigma(sensitivity=2.0, epsilon=1.0, delta=1e-6)
    assert sigma == pytest.approx(2.0 * math.sqrt(2 * math.log(1.25 / 1e-6)) / 1.0)
    mechanism = GaussianMechanism(2.0, 1.0, 1e-6, rng=np.random.default_rng(11))
    assert mechanism.sigma == pytest.approx(sigma)
    parameters = mechanism.parameters()
    assert parameters["calibration"] == "sigma = Delta_f * sqrt(2 ln(1.25/delta)) / epsilon"
    releases = [mechanism.release(100.0) for _ in range(4000)]
    utility = evaluate_utility(100.0, releases)
    assert abs(utility["bias"]) < sigma  # noise centered on the truth
    assert utility["rmse"] < 3 * sigma
    with pytest.raises(PrivacyError):
        gaussian_sigma(0.0, 1.0, 1e-6)   # sensitivity must be positive
    with pytest.raises(PrivacyError):
        gaussian_sigma(1.0, 1.0, 1.0)    # delta must be in (0, 1)


def test_privacy_accountant_rejects_release_exceeding_budget():
    """Acceptance: privacy accountant rejects a release exceeding budget."""
    accountant = PrivacyAccountant(epsilon_budget=2.0, delta_budget=1e-5)
    request = PrivacyReleaseRequest(
        "rel-big", "person", "add/remove one person", "mean cpu", 1.0, "Gaussian",
        2.5, 1e-6, "telemetry", "v1", calibration_reviewed=True,
    )
    with pytest.raises(BudgetExceededError):
        accountant.request_release(request)


def test_sequential_dp_releases_compose_under_accountant():
    """Acceptance: sequential DP releases compose under the configured accountant."""
    accountant = PrivacyAccountant(epsilon_budget=2.0, delta_budget=1e-5)
    for index, epsilon in enumerate((1.0, 0.5)):
        request = PrivacyReleaseRequest(
            f"rel-{index}", "person", "add/remove one person", f"stat-{index}",
            1.0, "Gaussian", epsilon, 1e-6, "telemetry", "v1", calibration_reviewed=True,
        )
        record = accountant.request_release(request)
        assert record.approved
    assert accountant.composed_epsilon == pytest.approx(1.5)
    assert accountant.composed_delta == pytest.approx(2e-6)
    final = PrivacyReleaseRequest(
        "rel-final", "person", "add/remove one person", "stat-final", 1.0, "Gaussian",
        0.5, 1e-6, "telemetry", "v1", calibration_reviewed=True,
    )
    accountant.request_release(final)
    assert accountant.composed_epsilon == pytest.approx(2.0)
    exhausted = PrivacyReleaseRequest(
        "rel-over", "person", "add/remove one person", "stat-over", 1.0, "Gaussian",
        0.1, 1e-6, "telemetry", "v1", calibration_reviewed=True,
    )
    with pytest.raises(BudgetExceededError):
        accountant.request_release(exhausted)
    ledger = accountant.ledger_report()
    assert ledger["approved_count"] == 3
    assert ledger["remaining"]["epsilon_remaining"] == pytest.approx(0.0)


def test_unreviewed_mechanisms_are_refused():
    """Mechanisms outside the reviewed set cannot draw from the budget."""
    accountant = PrivacyAccountant(epsilon_budget=5.0, delta_budget=1e-4)
    with pytest.raises(PrivacyError):
        accountant.request_release(PrivacyReleaseRequest(
            "rel-x", "person", "add/remove one person", "stat", 1.0, "Cauchy",
            1.0, 1e-6, "telemetry", "v1", calibration_reviewed=True,
        ))


def test_dp_telemetry_utility_and_subgroup_behavior():
    """Acceptance: DP telemetry is tested for utility and subgroup behavior."""
    rng = np.random.default_rng(5)
    values = list(np.clip(rng.normal(50.0, 5.0, size=500), 0.0, 100.0))
    result = dp_mean(values, clip_lower=0.0, clip_upper=100.0,
                     epsilon=1.0, delta=1e-6, rng=rng)
    true_mean = sum(values) / len(values)
    assert result["cohort_size"] == 500
    assert abs(result["released_value"] - true_mean) < 2.0
    # subgroup utility: releases for each subgroup stay close to subgroup truth
    true_by_group = {"group-a": 48.0, "group-b": 52.0}
    released_by_group = {
        group: dp_mean([value for value in values[:250]] if group == "group-a" else values[250:],
                       clip_lower=0.0, clip_upper=100.0, epsilon=1.0, delta=1e-6,
                       rng=rng)["released_value"]
        for group in true_by_group
    }
    report = subgroup_utility(true_by_group, released_by_group)
    for group in report:
        assert abs(report[group]["error"]) < 5.0
