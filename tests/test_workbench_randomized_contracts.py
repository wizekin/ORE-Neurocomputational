from pathlib import Path
import importlib
import random
import sys

import numpy as np

WORKBENCH_DIR = Path(__file__).resolve().parents[1] / "experiment_workbench"
if str(WORKBENCH_DIR) not in sys.path:
    sys.path.insert(0, str(WORKBENCH_DIR))

adapter = importlib.import_module("adapter")
auto_test = importlib.import_module("auto_test")
CallableInfo = adapter.CallableInfo
_convert_constructor_kwargs = adapter._convert_constructor_kwargs
execute = adapter.execute
_align_active_engagement_identity = auto_test._align_active_engagement_identity
_call_arguments = auto_test._call_arguments
_constructor_arguments = auto_test._constructor_arguments


def _active_engagement_callable():
    return CallableInfo(
        label="ore_active_engagement.pipeline.ActiveEngagementProcessor.process",
        module="ore_active_engagement.pipeline",
        name="process",
        signature="(request, state=None)",
        doc="Run one active-engagement episode.",
        class_name="ActiveEngagementProcessor",
    )


def test_optional_dataclass_conversion_does_not_hide_contract_errors():
    from ore_active_engagement.pipeline import ActiveEngagementProcessor

    try:
        _convert_constructor_kwargs(
            ActiveEngagementProcessor,
            "ore_active_engagement",
            {"state": {"learner_id": "learner-1", "associative_weights": {"x": 1000.0}}},
        )
    except (TypeError, ValueError):
        pass
    else:
        raise AssertionError("Invalid nested dataclass data must be rejected, not left as a dict")


def test_randomized_active_engagement_inputs_respect_cross_object_contracts():
    from ore_active_engagement.contracts import ActiveEngagementRequest, ActiveEngagementState

    rng = random.Random(20261127)
    rng_np = np.random.default_rng(20261127)
    info = _active_engagement_callable()
    constructor_arguments = _constructor_arguments(
        "ore_active_engagement", info, rng, rng_np
    )
    call_arguments = _call_arguments(
        "ore_active_engagement", info, rng, rng_np
    )
    _align_active_engagement_identity(
        "ore_active_engagement", constructor_arguments, call_arguments
    )

    learner_id = constructor_arguments["learner_id"]
    assert call_arguments["request"]["learner_id"] == learner_id
    request = ActiveEngagementRequest(**call_arguments["request"])
    assert request.learner_id == learner_id

    state_payload = constructor_arguments.get("state")
    if state_payload is not None:
        assert state_payload["learner_id"] == learner_id
        assert all(0.0 <= value <= 1.0
                   for value in state_payload["associative_weights"].values())
        converted = _convert_constructor_kwargs(
            __import__("ore_active_engagement.pipeline", fromlist=["ActiveEngagementProcessor"])
            .ActiveEngagementProcessor,
            "ore_active_engagement",
            {"state": state_payload},
        )
        assert isinstance(converted["state"], ActiveEngagementState)
        assert converted["state"].learner_id == learner_id


def test_randomized_active_engagement_trial_executes_all_seventeen_operations():
    rng = random.Random(31)
    rng_np = np.random.default_rng(31)
    info = _active_engagement_callable()
    constructor_arguments = _constructor_arguments(
        "ore_active_engagement", info, rng, rng_np
    )
    call_arguments = _call_arguments(
        "ore_active_engagement", info, rng, rng_np
    )
    _align_active_engagement_identity(
        "ore_active_engagement", constructor_arguments, call_arguments
    )
    result = execute(
        "ore_active_engagement",
        info,
        "Understand the generated learning material.",
        {
            "__init__": constructor_arguments,
            **call_arguments,
        },
    )
    assert result["status"] == "success", result["traceback"]
    assert len(result["output"].operation_trace) == 17
