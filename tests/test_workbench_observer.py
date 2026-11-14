from dataclasses import dataclass
import numpy as np

from experiment_workbench.observer import PythonCallObserver, inspect_output


@dataclass
class _StageRecord:
    name: str
    finite: bool
    mean: float


@dataclass
class _SyntheticResult:
    stage_outputs: dict
    operation_trace: list
    diagnostics: dict


def _profile_target(value):
    return {"doubled": value * 2}


def test_inspect_output_collects_stage_arrays_trace_and_checks():
    output = _SyntheticResult(
        stage_outputs={"beliefs": np.asarray([[0.25, 0.75]], dtype=np.float32)},
        operation_trace=[{"operation_id": "OP-01", "operation_name": "normalize"}],
        diagnostics={
            "valid": True,
            "stages": [_StageRecord("beliefs", True, 0.5)],
            "warnings": ["synthetic warning"],
        },
    )

    observed = inspect_output(output)

    assert observed["output_type"] == "_SyntheticResult"
    assert observed["trace_count"] >= 2
    assert any(row.get("operation_id") == "OP-01" for row in observed["trace_rows"])
    assert any(row.get("name") == "beliefs" for row in observed["trace_rows"])
    assert observed["array_count"] == 1
    assert observed["numerical_summaries"][0]["shape"] == [1, 2]
    assert observed["numerical_summaries"][0]["finite"] is True
    assert {"check": "output.diagnostics.valid", "value": True} in observed["checks"]
    assert observed["warnings"] == ["synthetic warning"]


def test_python_call_observer_records_component_function_calls():
    with PythonCallObserver(__name__) as observer:
        result = _profile_target(21)

    report = observer.report()
    assert result["doubled"] == 42
    assert report["observed_call_count"] >= 1
    assert any(event["function"].endswith("._profile_target") for event in report["events"])


def test_live_observer_streams_call_return_and_source_line_events():
    streamed = []
    with PythonCallObserver(__name__, on_event=streamed.append, trace_lines=True) as observer:
        result = _profile_target(9)

    kinds = {event["kind"] for event in streamed}
    assert result["doubled"] == 18
    assert "call" in kinds
    assert "return" in kinds
    assert "line" in kinds
    assert all("explanation" in event for event in streamed)
    assert any(event.get("active_stack") for event in streamed if event["kind"] == "call")
    assert observer.report()["timeline_event_count"] == len(streamed)


def _raise_for_observer():
    raise RuntimeError("observer test exception")


def test_live_observer_reports_exceptions_without_swallowing_them():
    streamed = []
    try:
        with PythonCallObserver(__name__, on_event=streamed.append):
            _raise_for_observer()
    except RuntimeError as exc:
        assert str(exc) == "observer test exception"
    else:
        raise AssertionError("The original component exception should propagate.")

    assert any(event["kind"] == "exception" for event in streamed)
