"""Regression tests for randomized data contracts and unavailable estimates."""
from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

WORKBENCH_DIR = Path(__file__).resolve().parents[1] / "experiment_workbench"
if str(WORKBENCH_DIR) not in sys.path:
    sys.path.insert(0, str(WORKBENCH_DIR))

from auto_test import _invariant_checks, _random_dataclass  # noqa: E402
from ore_data.controller_adapter import StorageTelemetry  # noqa: E402
from ore_data.streaming.quantiles import QuantileEstimate  # noqa: E402


def _numeric_finiteness_check(output: object, path: str) -> dict[str, object]:
    checks = _invariant_checks(output)
    return next(
        check for check in checks
        if check["kind"] == "numeric_finiteness" and check["path"] == path
    )


def test_storage_telemetry_random_generation_respects_declared_ranges() -> None:
    for seed in range(50):
        telemetry = _random_dataclass(
            random.Random(seed),
            np.random.default_rng(seed),
            StorageTelemetry,
            "ore_data",
        )
        assert 0.0 <= telemetry.memory_pressure <= 1.0
        assert 0.0 <= telemetry.storage_pressure <= 1.0
        assert telemetry.queue_depth >= 0
        assert 0.0 <= telemetry.compaction_load <= 1.0


def test_empty_quantile_nan_is_allowed_only_when_unavailable() -> None:
    estimate = QuantileEstimate(
        percentile=0.5,
        value=float("nan"),
        unit="ms",
        source="ore.data.telemetry",
        nominal_error_bound=0.1,
        available=False,
    )

    check = _numeric_finiteness_check(estimate, "output.value")

    assert check["status"] == "allowed_unavailable"
    assert not any(item["status"] == "violation" for item in _invariant_checks(estimate))


def test_quantile_nan_is_still_a_violation_when_available() -> None:
    estimate = QuantileEstimate(
        percentile=0.5,
        value=float("nan"),
        unit="ms",
        source="ore.data.telemetry",
        nominal_error_bound=0.1,
        available=True,
    )

    assert _numeric_finiteness_check(estimate, "output.value")["status"] == "violation"


def test_unavailable_quantile_does_not_permit_infinity() -> None:
    estimate = QuantileEstimate(
        percentile=0.5,
        value=float("inf"),
        unit="ms",
        source="ore.data.telemetry",
        nominal_error_bound=0.1,
        available=False,
    )

    assert _numeric_finiteness_check(estimate, "output.value")["status"] == "violation"


@dataclass
class UnrelatedMeasurement:
    value: float
    available: bool = False


def test_unrelated_dataclass_nan_remains_a_violation() -> None:
    measurement = UnrelatedMeasurement(value=float("nan"))

    assert (
        _numeric_finiteness_check(measurement, "output.value")["status"]
        == "violation"
    )
