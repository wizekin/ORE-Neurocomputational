"""Shared pytest fixtures for the ore_hic test suite.

The fixture lives in conftest.py so every test module can request it without
importing it explicitly (imports used only as fixture parameters look unused
to linters and get stripped by autofixes).
"""
from __future__ import annotations

import pytest
from test_ore_hic_helpers import build_compiler, run_to_completion


@pytest.fixture(scope="module")
def completed_result():
    """One full 52-step compilation shared by pipeline tests."""
    compiler = build_compiler()
    return run_to_completion(compiler)
