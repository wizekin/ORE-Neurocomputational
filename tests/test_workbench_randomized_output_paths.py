from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORKBENCH_DIR = ROOT / "experiment_workbench"
if str(WORKBENCH_DIR) not in sys.path:
    sys.path.insert(0, str(WORKBENCH_DIR))

import auto_test  # noqa: E402


def test_randomized_path_arguments_are_redirected_into_run_artifacts(tmp_path):
    token = auto_test._RANDOM_TEST_ARTIFACT_DIR.set(tmp_path / "side_effects" / "ore_agentic")
    try:
        generated = auto_test._random_value(
            random.Random(17),
            np.random.default_rng(17),
            "path",
            str | Path,
            package_name="ore_agentic",
        )
    finally:
        auto_test._RANDOM_TEST_ARTIFACT_DIR.reset(token)

    generated_path = Path(generated)
    assert generated_path.is_absolute()
    assert generated_path.parent == tmp_path / "side_effects" / "ore_agentic"
    assert generated_path.suffix == ".sqlite3"
    assert not (ROOT / generated_path.name).exists()


def test_randomized_path_arguments_keep_string_type_when_declared_as_string(tmp_path):
    token = auto_test._RANDOM_TEST_ARTIFACT_DIR.set(tmp_path / "side_effects" / "sample")
    try:
        generated = auto_test._random_value(
            random.Random(23),
            np.random.default_rng(23),
            "database_path",
            str,
            package_name="sample_package",
        )
    finally:
        auto_test._RANDOM_TEST_ARTIFACT_DIR.reset(token)

    assert isinstance(generated, str)
    generated_path = Path(generated)
    assert generated_path.parent == tmp_path / "side_effects" / "sample"
    assert generated_path.suffix == ".sqlite3"
