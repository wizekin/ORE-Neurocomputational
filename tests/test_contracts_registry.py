"""Versioned contract catalog and deterministic artifact hashing tests."""
from dataclasses import dataclass

import numpy as np
import pytest

from ore_contracts.adapters import make_envelope
from ore_contracts.contracts_registry import ContractCatalog, register_existing_contracts
from ore_contracts.module_artifacts import ContractArtifact, content_hash


@dataclass(frozen=True)
class Payload:
    """Small typed payload used to test model validation."""

    value: str


def test_catalog_rejects_duplicates_and_validates_exact_contract_payloads():
    """Require a registered exact version and its declared Python model."""
    catalog = ContractCatalog()
    catalog.register_model("ORE.TEST", "1.0.0", Payload, required_fields=("value",))
    artifact = ContractArtifact(make_envelope("ORE.TEST", "run-1", "ore.producer"), Payload("valid"))
    assert catalog.validate_artifact(artifact).valid
    bad = ContractArtifact(make_envelope("ORE.TEST", "run-2", "ore.producer"), {"value": "not a Payload"})
    report = catalog.validate_artifact(bad)
    assert not report.valid
    assert "payload type mismatch" in report.errors[0]
    with pytest.raises(ValueError):
        catalog.register_model("ORE.TEST", "1.0.0", Payload)


def test_array_hash_is_deterministic_and_tracks_shape_and_dtype():
    """Hash NumPy arrays by dtype, shape, and contiguous numeric contents."""
    left = np.asarray([[1, 2], [3, 4]], dtype=np.int16)
    same = np.asarray([[1, 2], [3, 4]], dtype=np.int16)
    reshaped = np.asarray([1, 2, 3, 4], dtype=np.int16)
    other_dtype = np.asarray([[1, 2], [3, 4]], dtype=np.int64)
    assert content_hash(left) == content_hash(same)
    assert content_hash(left) != content_hash(reshaped)
    assert content_hash(left) != content_hash(other_dtype)


def test_nonfinite_payload_values_are_rejected_before_exchange():
    """Do not serialize NaN or infinity into deterministic artifacts."""
    with pytest.raises(ValueError):
        ContractArtifact(
            make_envelope("ORE.NUMERIC", "run-1", "ore.producer"),
            {"scores": [1.0, float("nan")]},
        )


def test_existing_if_models_are_registered_by_reference_not_redeclared():
    """The universal catalog reuses the current canonical ORE interface model classes."""
    catalog = ContractCatalog()
    definitions = register_existing_contracts(catalog)
    assert {item.contract_id for item in definitions} == {
        "IF-05", "IF-06", "IF-07", "IF-08", "IF-10", "IF-12", "IF-13",
    }
    assert catalog.get_contract("IF-05", "1.0.0").model_type.__name__ == "VisualFeatureBundle"
