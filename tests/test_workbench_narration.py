from __future__ import annotations

import json
from pathlib import Path
import sys

WORKBENCH_DIR = Path(__file__).resolve().parents[1] / "experiment_workbench"
if str(WORKBENCH_DIR) not in sys.path:
    sys.path.insert(0, str(WORKBENCH_DIR))


def test_n9_narration_catalog_covers_every_registered_operation():
    from narration import N9_OPERATION_EXPLANATIONS, explain_operation
    from ore_n9.pipeline import N9_ALGORITHM_STEPS

    assert len(N9_ALGORITHM_STEPS) == 120
    assert set(N9_OPERATION_EXPLANATIONS) == set(N9_ALGORITHM_STEPS)
    for index, operation in enumerate(N9_ALGORITHM_STEPS, start=1):
        translated = explain_operation(operation, index, "ok")
        assert translated["explanation"]
        assert translated["phase"]
        assert translated["phase_explanation"]


def test_execution_logger_writes_markdown_and_jsonl(tmp_path):
    from narration import OREExecutionLogger

    streamed = []
    logger = OREExecutionLogger(
        tmp_path, "unit-test-run", "ore_n9", "N9Processor.process",
        "2026-11-15T00:00:00+00:00", on_event=streamed.append,
    )
    logger.log_operation({
        "step": 1,
        "operation": "validate_incoming_observation_contract",
        "status": "ok",
        "patch_count": 2,
    })
    logger.log_python_event({
        "kind": "call",
        "function": "ore_n9.pipeline.N9Processor.process",
        "short_name": "process",
        "file": "pipeline.py",
        "line": 300,
    })
    metadata = logger.finish("success", 12.5, "N9Result")

    markdown = Path(metadata["markdown_path"]).read_text(encoding="utf-8")
    jsonl = [
        json.loads(line)
        for line in Path(metadata["jsonl_path"]).read_text(encoding="utf-8").splitlines()
    ]
    assert "Step 001" in markdown
    assert "Checks that every patch" in markdown
    assert any(item["kind"] == "operation" for item in jsonl)
    assert any(item["kind"] == "call" for item in jsonl)
    assert jsonl[-1]["kind"] == "run_finished"
    assert streamed[-1]["kind"] == "run_finished"
    assert metadata["event_count"] == len(jsonl)


def test_workbench_persists_all_120_n9_operation_events(tmp_path):
    from adapter import discover_callables, execute

    callables, errors = discover_callables("ore_n9")
    assert not errors, "\n".join(errors)
    n9_process = next(
        item for item in callables
        if item.class_name == "N9Processor" and item.name == "process"
    )

    result = execute(
        "ore_n9",
        n9_process,
        prompt="",
        supplied={
            "request_data": {
                "sequence_id": "narrator-integration-test",
                "patches": [{
                    "patch_id": "p0",
                    "grapheme_probabilities": {"a": 0.95, "b": 0.05},
                }],
            }
        },
        log_directory=tmp_path,
    )

    assert result["status"] == "success", result.get("traceback")
    assert len(result["output"].operation_trace) == 120
    log_path = Path(result["logs"]["jsonl_path"])
    events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    operation_events = [event for event in events if event.get("kind") == "operation"]
    assert len(operation_events) == 120
    assert [event["step"] for event in operation_events] == list(range(1, 121))
    assert operation_events[0]["explanation"]
    assert operation_events[-1]["operation"] == "finalize_integrated_n9_result"
