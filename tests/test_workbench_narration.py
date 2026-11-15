from __future__ import annotations

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


def test_ast_catalog_indexes_modules_classes_functions_and_nested_functions(tmp_path):
    from ore_logger.catalog import sync_repository
    from ore_logger.database import LoggerDatabase

    source = tmp_path / "sample.py"
    source.write_text(
        '''
"""A small sample module."""
class Demo:
    """Coordinates a small example workflow."""
    def work(self, values: list[int]) -> int:
        """Adds the supplied values and returns their total."""
        return sum(values)

    def outer(self):
        def inner(value):
            return value + 1
        return inner(2)

async def async_task(item: str) -> str:
    return item.strip()
''',
        encoding="utf-8",
    )
    database = LoggerDatabase(tmp_path / "logs" / "logger.sqlite3")
    report = sync_repository(database, tmp_path)

    assert report["files_seen"] == 1
    assert report["files_indexed"] == 1
    assert database.get_annotation("sample", "__module__", "module")
    assert database.get_annotation("sample", "Demo", "class")
    work = database.get_annotation("sample", "Demo.work", "function")
    assert work and "Adds the supplied values" in work["annotation"]
    inner = database.get_annotation("sample", "Demo.outer.<locals>.inner", "function")
    assert inner and "Computes" not in inner["annotation"]
    async_annotation = database.get_annotation("sample", "async_task", "async_function")
    assert async_annotation and "async_task" in async_annotation["qualified_name"]


def test_manual_annotation_survives_catalog_rescan_and_marks_stale(tmp_path):
    from ore_logger.catalog import sync_repository
    from ore_logger.database import LoggerDatabase

    source = tmp_path / "module.py"
    source.write_text("def calculate_total(values):\n    return sum(values)\n", encoding="utf-8")
    database = LoggerDatabase(tmp_path / "logger.sqlite3")
    sync_repository(database, tmp_path)
    symbol = database.get_annotation("module", "calculate_total", "function")
    assert symbol is not None
    database.save_manual_annotation(symbol["symbol_id"], "Adds all values and checks the total.")
    source.write_text("def calculate_total(values):\n    return sum(values) + 1\n", encoding="utf-8")
    sync_repository(database, tmp_path)
    refreshed = database.get_annotation("module", "calculate_total", "function")
    assert refreshed["annotation"] == "Adds all values and checks the total."
    assert refreshed["annotation_source"] == "manual"
    assert refreshed["stale"] == 1


def test_sqlite_recorder_links_python_events_to_function_and_class_annotations(tmp_path):
    from ore_logger.database import LoggerDatabase
    from ore_logger.recorder import SQLiteRunRecorder

    root = Path(__file__).resolve().parents[1]
    database_path = tmp_path / "ore_logger.sqlite3"
    recorder = SQLiteRunRecorder(
        database_path, root, "sqlite-recorder-test", "ore_n9",
        "N9Processor.process", "2026-11-15T00:00:00+00:00",
    )
    recorder.log_python_event({
        "kind": "call",
        "function": "ore_n9.pipeline.N9Processor.process",
        "short_name": "process",
        "qualname": "N9Processor.process",
        "module": "ore_n9.pipeline",
        "file": "ore_n9/pipeline.py",
        "line": 300,
    })
    metadata = recorder.finish("success", 14.2, "N9Result")

    database = LoggerDatabase(database_path)
    run = database.get_run("sqlite-recorder-test")
    events = database.list_events("sqlite-recorder-test")
    call = next(item for item in events if item["event_kind"] == "call")
    assert run["started_at_local"]
    assert run["display_label"].endswith("sqlite-r")
    assert call["qualified_name"] == "N9Processor.process"
    assert call["annotation"]
    assert call["class_annotation"]
    assert call["symbol_id"]
    assert "annotation" not in call["metadata"]
    assert metadata["database_path"] == str(database.path)


def test_workbench_persists_all_120_n9_operation_events_in_sqlite(tmp_path):
    from adapter import discover_callables, execute
    from ore_logger.database import LoggerDatabase

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
    database = LoggerDatabase(result["logs"]["database_path"])
    events = database.list_events(result["run_id"])
    operation_events = [event for event in events if event["event_kind"] == "operation"]
    assert len(operation_events) == 120
    assert [event["metadata"]["step"] for event in operation_events] == list(range(1, 121))
    assert operation_events[0]["annotation"]
    assert operation_events[-1]["qualified_name"] == "finalize_integrated_n9_result"
    assert result["logs"]["event_count"] >= 120
