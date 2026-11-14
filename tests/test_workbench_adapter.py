from pathlib import Path
import sys

import numpy as np

WORKBENCH_DIR = Path(__file__).resolve().parents[1] / "experiment_workbench"
if str(WORKBENCH_DIR) not in sys.path:
    sys.path.insert(0, str(WORKBENCH_DIR))

from adapter import accepts_parameter, discover_callables, execute


def _select(package_name, class_name, method_name):
    callables, errors = discover_callables(package_name)
    assert not errors, "\n".join(errors)
    matches = [
        item for item in callables
        if item.class_name == class_name and item.name == method_name
    ]
    assert matches, f"{class_name}.{method_name} was not discovered in {package_name}"
    return matches[0]


def test_discovery_finds_text_processor_entry_point():
    item = _select("ore_reading", "IntegratedReadingProcessor", "process")
    assert "ReadingRequest" in item.signature


def test_n1_image_execution_and_n2_upstream_handoff():
    n1_callable = _select("ore_n1", "N1Processor", "process")
    assert accepts_parameter(n1_callable, "image")

    image = np.zeros((48, 48), dtype=np.uint8)
    image[10:38, 18:30] = 220
    n1 = execute(
        "ore_n1",
        n1_callable,
        "Inspect this visual input.",
        {},
        runtime_kwargs={"image": image},
    )
    assert n1["status"] == "success", n1["traceback"]
    assert n1["observation"]["array_count"] > 0
    assert n1["profile"]["observed_call_count"] > 0

    n2_callable = _select("ore_n2", "N2Processor", "process")
    n2 = execute(
        "ore_n2",
        n2_callable,
        "Continue with the N1 spatial representation.",
        {},
        previous_output=n1["output"],
    )
    assert n2["status"] == "success", n2["traceback"]
    assert n2["observation"]["trace_count"] > 0
    assert n2["observation"]["array_count"] > 0


def test_streamlit_entrypoint_has_valid_python_syntax():
    import ast

    app_path = WORKBENCH_DIR / "app.py"
    ast.parse(app_path.read_text(encoding="utf-8"), filename=str(app_path))
