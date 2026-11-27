import importlib
import json
import random
import sys
import types
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, NamedTuple, Optional, Sequence, TypedDict

try:
    from typing import NotRequired, Required
except ImportError:
    try:
        from typing_extensions import NotRequired, Required
    except ImportError:
        NotRequired = Required = None

import numpy as np
import pytest

WORKBENCH_DIR = Path(__file__).resolve().parents[1] / "experiment_workbench"
if str(WORKBENCH_DIR) not in sys.path:
    sys.path.insert(0, str(WORKBENCH_DIR))

adapter = importlib.import_module("adapter")
auto_test = importlib.import_module("auto_test")
CallableInfo = adapter.CallableInfo
_build_arguments = adapter.build_arguments
_bind_invocation = adapter._bind_invocation
_convert_constructor_kwargs = adapter._convert_constructor_kwargs
_convert_value = adapter._convert_value
execute = adapter.execute
_random_value = auto_test._random_value
_discover_callables = auto_test._discover_callables
_random_dataclass = auto_test._random_dataclass
_random_lexicon = auto_test._random_lexicon
_align_active_engagement_identity = auto_test._align_active_engagement_identity
_call_arguments = auto_test._call_arguments
_constructor_arguments = auto_test._constructor_arguments
_classify_failure = auto_test._classify_failure
_save_manual_run_report = auto_test.save_manual_run_report
_numeric_arrays = auto_test._numeric_arrays
_invariant_checks = auto_test._invariant_checks
_bounded_value = auto_test._bounded_value
_RANDOM_TEST_ARTIFACT_DIR = auto_test._RANDOM_TEST_ARTIFACT_DIR
_RANDOM_TEST_REPOSITORY_ROOT = auto_test._RANDOM_TEST_REPOSITORY_ROOT
inspect_output = importlib.import_module("observer").inspect_output


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

    with pytest.raises(ValueError, match="associative_weight"):
        _convert_constructor_kwargs(
            ActiveEngagementProcessor,
            "ore_active_engagement",
            {
                "state": {
                    "learner_id": "learner-1",
                    "associative_weights": {"x": 1000.0},
                }
            },
        )


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
    request = call_arguments["request"]
    assert isinstance(request, ActiveEngagementRequest)
    assert request.learner_id == learner_id

    state = constructor_arguments.get("state")
    if state is not None:
        assert isinstance(state, ActiveEngagementState)
        assert state.learner_id == learner_id
        assert all(0.0 <= value <= 1.0 for value in state.associative_weights.values())


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


@dataclass(frozen=True)
class _SampleRequest:
    sequence_id: str
    text: str


class _SampleMode(Enum):
    FAST = "fast"
    SAFE = "safe"


def test_request_data_is_converted_only_for_a_dataclass_request_parameter():
    def process(request: _SampleRequest):
        return request

    arguments = _build_arguments(
        process,
        "generated prompt",
        {"request_data": {"sequence_id": "seq-1", "text": "hello"}},
    )
    assert isinstance(arguments["request"], _SampleRequest)
    assert arguments["request"].sequence_id == "seq-1"


def test_request_data_is_preserved_when_it_is_a_real_callable_parameter():
    def process(request_data: Mapping[str, str]):
        return request_data

    arguments = _build_arguments(
        process, "ignored prompt", {"request_data": {"source": "manual"}}
    )
    assert arguments == {"request_data": {"source": "manual"}}


def test_request_data_rejects_ambiguous_or_conflicting_request_receivers():
    def multiple_requests(first: _SampleRequest, second: _SampleRequest):
        return first, second

    with pytest.raises(ValueError, match="multiple dataclass request parameters"):
        _build_arguments(
            multiple_requests,
            "prompt",
            {"request_data": {"sequence_id": "s", "text": "t"}},
        )

    def single_request(request: _SampleRequest):
        return request

    with pytest.raises(ValueError, match="conflicts"):
        _build_arguments(
            single_request,
            "prompt",
            {
                "request": {"sequence_id": "explicit", "text": "explicit"},
                "request_data": {"sequence_id": "reserved", "text": "reserved"},
            },
        )


def test_runtime_arguments_never_silently_overwrite_explicit_arguments():
    def process(image: np.ndarray):
        return image

    with pytest.raises(ValueError, match="conflict"):
        _build_arguments(
            process,
            "prompt",
            {"image": np.zeros((2, 2))},
            runtime_kwargs={"image": np.ones((2, 2))},
        )


def test_dataclass_conversion_rejects_unknown_fields_and_invalid_container_types():
    with pytest.raises(TypeError, match="Unexpected field"):
        _convert_value(
            _SampleRequest,
            {"sequence_id": "seq-1", "text": "hello", "typo": True},
        )

    with pytest.raises(TypeError, match="sequence input"):
        _convert_value(Sequence[str], "not-an-array")

    with pytest.raises(TypeError, match="mapping input"):
        _convert_value(Mapping[str, float], [("x", 0.5)])


def test_union_literal_enum_and_typed_mapping_conversion_are_strict():
    converted = _convert_value(
        Optional[_SampleRequest],
        {"sequence_id": "seq-2", "text": "world"},
    )
    assert isinstance(converted, _SampleRequest)

    assert _convert_value(_SampleMode, "fast") is _SampleMode.FAST
    assert _convert_value(_SampleMode, "SAFE") is _SampleMode.SAFE

    mapping = _convert_value(Mapping[str, float], {"salience": 0.75})
    assert mapping == {"salience": 0.75}

    with pytest.raises(ValueError, match="permitted literal"):
        _convert_value(__import__("typing").Literal["ltr", "rtl"], "diagonal")


def test_positional_only_callable_arguments_are_bound_before_execution():
    def process(text, /, uppercase=False):
        return text.upper() if uppercase else text

    kwargs = _build_arguments(process, "hello", {"text": "hello", "uppercase": True})
    args, call_kwargs = _bind_invocation(process, kwargs)
    assert process(*args, **call_kwargs) == "HELLO"


def test_random_value_uses_declared_types_before_parameter_names():
    rng = random.Random(70)
    rng_np = np.random.default_rng(70)

    sequence = _random_value(rng, rng_np, "sequence", Sequence[str])
    assert isinstance(sequence, (list, tuple))
    assert all(isinstance(item, str) for item in sequence)

    weights = _random_value(rng, rng_np, "weights", Mapping[str, float])
    assert isinstance(weights, dict)
    assert all(isinstance(key, str) for key in weights)
    assert all(0.0 <= value <= 1.0 for value in weights.values())

    dimensions = _random_value(rng, rng_np, "embedding_dimensions", int)
    assert dimensions >= 8

    reliability = _random_value(rng, rng_np, "reliability", float)
    assert 0.0 <= reliability <= 1.0


def test_contract_generator_respects_mutually_exclusive_reading_inputs():
    from ore_reading.contracts import ReadingRequest
    from ore_semantic.contracts import SemanticRequest

    rng = random.Random(71)
    rng_np = np.random.default_rng(71)
    for contract in (ReadingRequest, SemanticRequest):
        request = _random_dataclass(rng, rng_np, contract, "ore_reading")
        assert isinstance(request, contract)
        assert (request.text is None) != (request.tokens is None)


def test_package_specific_lexicon_records_match_their_actual_contracts():
    from ore_lexical_context.contracts import G2PRule, MorphemeEntry
    from ore_reading.contracts import G2PRule as ReadingG2PRule
    from ore_reading.contracts import ReadingLexiconEntry
    from ore_semantic.contracts import SemanticConcept

    rng = random.Random(72)

    morphemes = _random_lexicon(rng, "morphemes", "ore_lexical_context")
    assert all(isinstance(_convert_value(MorphemeEntry, item), MorphemeEntry) for item in morphemes)

    lexical_rules = _random_lexicon(rng, "g2p_rules", "ore_lexical_context")
    assert all(isinstance(_convert_value(G2PRule, item), G2PRule) for item in lexical_rules)

    reading_rules = _random_lexicon(rng, "g2p_rules", "ore_reading")
    assert all(
        isinstance(_convert_value(ReadingG2PRule, item), ReadingG2PRule)
        for item in reading_rules
    )

    reading_lexicon = _random_lexicon(rng, "lexicon", "ore_reading")
    assert all(
        isinstance(_convert_value(ReadingLexiconEntry, item), ReadingLexiconEntry)
        for item in reading_lexicon
    )

    concepts = _random_lexicon(rng, "concepts", "ore_semantic")
    assert all(isinstance(_convert_value(SemanticConcept, item), SemanticConcept) for item in concepts)


def test_failure_classifier_does_not_hide_component_type_errors_as_rejections():
    assert _classify_failure(
        "TypeError: internal function mismatch",
        "callable_execution",
        "File \"pipeline.py\", line 10, in process",
    ) == "unexpected_runtime_exception"

    assert _classify_failure(
        "TypeError: got an unexpected keyword argument 'sequence_id'",
        "call_argument_binding",
        "",
    ) == "workbench_dispatch_error"

    assert _classify_failure(
        "ValueError: listener_level must be novice, intermediate, or expert",
        "argument_building",
        "File \"contracts.py\", line 10, in __post_init__",
    ) == "input_contract_rejection_candidate"

    assert _classify_failure(
        "PermissionError: file access denied",
        "callable_execution",
        "",
    ) == "unexpected_runtime_exception"


def test_numeric_invariant_scanner_includes_scalars_and_native_sequences():
    observations = list(_numeric_arrays({"score": 0.8, "weights": [0.2, 0.8]}))
    assert len(observations) == 2
    checks = _invariant_checks({"score": 0.8, "weights": [0.2, 0.8]})
    assert any(item["kind"] == "numeric_finiteness" for item in checks)


def test_complex_arrays_and_generic_mappings_remain_observable():
    output = MappingProxyType({
        "confidence": np.asarray([1.0 + 2.0j, 2.0 - 1.0j], dtype=np.complex128),
        "weights": {0.2, 0.8},
    })
    observation = inspect_output(output)
    assert observation["array_count"] == 1
    assert observation["numerical_summaries"][0]["complex_values"] is True

    bounded = _bounded_value(output)
    sample = bounded["confidence"]["sample"]
    assert all("real" in item and "imaginary" in item for item in sample)

    checks = _invariant_checks(output)
    assert any(item["kind"] == "numeric_finiteness" for item in checks)
    assert not any(
        item["kind"] == "bounded_probability_or_confidence_candidate"
        for item in checks
    )


def test_path_generator_isolates_outputs_but_preserves_repository_roots(tmp_path):
    token = _RANDOM_TEST_ARTIFACT_DIR.set(tmp_path)
    try:
        rng = random.Random(73)
        rng_np = np.random.default_rng(73)
        database_path = _random_value(
            rng, rng_np, "database_path", str, package_name="ore_logger"
        )
        assert Path(database_path).parent == tmp_path
        assert Path(database_path).suffix == ".sqlite3"

        typed_root = _random_value(
            rng, rng_np, "repository_root", Path, package_name="ore_logger"
        )
        string_root = _random_value(
            rng, rng_np, "repository_root", str, package_name="ore_logger"
        )
        assert typed_root == Path.cwd()
        assert string_root == str(Path.cwd())
    finally:
        _RANDOM_TEST_ARTIFACT_DIR.reset(token)


def test_runtime_request_argument_conflicts_with_request_data_helper():
    def process(request: _SampleRequest):
        return request

    with pytest.raises(ValueError, match="conflicts"):
        _build_arguments(
            process,
            "prompt",
            {"request_data": {"sequence_id": "seq", "text": "payload"}},
            runtime_kwargs={
                "request": _SampleRequest(sequence_id="explicit", text="explicit")
            },
        )


def test_unresolved_forward_reference_keeps_matching_objects_but_rejects_unrelated_values():
    class UnresolvedOREContract:
        pass

    instance = UnresolvedOREContract()
    assert _convert_value("UnresolvedOREContract", instance) is instance
    with pytest.raises(TypeError, match="could not be resolved"):
        _convert_value("UnresolvedOREContract", object())
    with pytest.raises(TypeError, match="could not be resolved"):
        _convert_value("UnresolvedOREContract", {"field": "value"})


def test_trace_dataclass_preview_does_not_use_deepcopy():
    from dataclasses import dataclass
    from adapter import json_safe

    @dataclass
    class TraceRecord:
        values: np.ndarray

    record = TraceRecord(values=np.arange(2_000_000, dtype=np.float32))
    preview = json_safe(record)
    assert preview["values"]["shape"] == [2_000_000]
    assert preview["values"]["truncated"] is True


def test_prompt_mapping_respects_sequence_field_types():
    def process(sequence: Sequence[str], characters: Sequence[str]):
        return sequence, characters

    arguments = _build_arguments(
        process, "hello world", {}
    )
    # The name "sequence" maps to words; "characters" maps to characters.
    assert arguments["sequence"] == ["hello", "world"]
    assert arguments["characters"] == list("hello world")


def test_prompt_is_not_forced_into_non_text_sequence_contracts():
    def process(sequence: Sequence[float]):
        return sequence

    with pytest.raises(ValueError, match="Missing required argument"):
        _build_arguments(process, "not a numeric sequence", {})


class _SampleTypedDict(TypedDict):
    label: str
    weight: float


class _SampleNamedTuple(NamedTuple):
    label: str
    weight: float


def test_typed_dict_and_named_tuple_arguments_are_converted_strictly():
    typed = _convert_value(
        _SampleTypedDict, {"label": "cue", "weight": 0.7}
    )
    assert typed == {"label": "cue", "weight": 0.7}

    named = _convert_value(
        _SampleNamedTuple, {"label": "cue", "weight": 0.7}
    )
    assert named == _SampleNamedTuple("cue", 0.7)

    with pytest.raises(TypeError, match="Unexpected field"):
        _convert_value(
            _SampleTypedDict,
            {"label": "cue", "weight": 0.7, "typo": True},
        )

    with pytest.raises(ValueError, match="Missing required"):
        _convert_value(_SampleTypedDict, {"label": "cue"})


def test_constructor_validation_and_adapter_conversion_failures_are_distinguished():
    assert _classify_failure(
        "TypeError: unknown value type cannot be converted",
        "constructor_argument_conversion",
        "File \"adapter.py\", line 20, in _convert_value",
    ) == "workbench_input_generation_error"

    assert _classify_failure(
        "ValueError: processor learner_id must match ActiveEngagementState",
        "constructor_execution",
        "File \"pipeline.py\", line 20, in __init__",
    ) == "input_contract_rejection_candidate"

    assert _classify_failure(
        "TypeError: internal processor bug",
        "constructor_execution",
        "File \"pipeline.py\", line 21, in __init__",
    ) == "unexpected_runtime_exception"


def test_callable_discovery_skips_one_uninspectable_signature_without_hiding_others():
    module = types.ModuleType("synthetic_workbench_module")

    def usable(value: str):
        return value

    def uninspectable(value: str):
        return value

    usable.__module__ = module.__name__
    uninspectable.__module__ = module.__name__
    uninspectable.__signature__ = "not-a-signature"
    module.usable = usable
    module.uninspectable = uninspectable

    errors = []
    discovered = _discover_callables("ore_synthetic", [module], errors)
    assert [item.name for item in discovered] == ["usable"]
    assert any("uninspectable" in error and "signature unavailable" in error for error in errors)


def test_n5_model_state_schema_is_restored_independently_of_field_name():
    from ore_n5.contracts import N5ModelState

    rng = random.Random(74)
    rng_np = np.random.default_rng(74)
    model = auto_test._make_random_n5_model(rng, rng_np)
    assert isinstance(model, N5ModelState)

    restored = _convert_value(N5ModelState, model.to_dict())
    assert isinstance(restored, N5ModelState)
    assert restored.fingerprint == model.fingerprint
    assert restored.feature_dimension == model.feature_dimension

    generated = _random_dataclass(rng, rng_np, N5ModelState, "ore_n5")
    assert isinstance(generated, N5ModelState)


def test_manual_reports_bound_constructor_and_converted_array_inputs(tmp_path):
    payload = np.arange(2_000_000, dtype=np.float32)
    report_path = _save_manual_run_report(
        tmp_path,
        {
            "run_id": "manual-run-size-test",
            "component": "ore_synthetic",
            "callable": "Processor.process",
            "status": "success",
            "constructor_arguments": {"features": payload},
            "converted_constructor_arguments": {"features": payload},
            "arguments": {"text": "sample"},
            "output": {"ok": True},
        },
    )
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    for field_name in ("constructor_arguments", "converted_constructor_arguments"):
        serialized = report[field_name]["features"]
        assert serialized["shape"] == [2_000_000]
        assert serialized["truncated"] is True
        assert len(serialized["sample"]) == 24
    assert "constructor_arguments" not in report["run"]
    assert "converted_constructor_arguments" not in report["run"]


def test_observation_failures_do_not_turn_successful_calls_into_errors(monkeypatch):
    def fail_observation(_value):
        raise RuntimeError("diagnostic scanner unavailable")

    monkeypatch.setattr(adapter, "inspect_output", fail_observation)
    info = CallableInfo(
        label="builtins.len",
        module="builtins",
        name="len",
        signature="(obj, /)",
        doc="Return the length of an object.",
    )
    result = execute("builtins", info, "", {"obj": [1, 2, 3]})
    assert result["status"] == "success"
    assert result["output"] == 3
    assert "input_observation_error" in result
    assert "observation_error" in result


def test_execute_returns_structured_error_for_non_mapping_arguments():
    info = CallableInfo(
        label="builtins.len",
        module="builtins",
        name="len",
        signature="(obj, /)",
        doc="Return the length of an object.",
    )
    result = execute("builtins", info, "", 123)
    assert result["status"] == "error"
    assert result["failure_phase"] == "argument_building"
    assert "JSON object" in result["error"]


def test_required_typeddict_qualifiers_are_unwrapped_by_origin_name():
    from typing import get_origin
    try:
        from typing import NotRequired, Required
    except ImportError:
        try:
            from typing_extensions import NotRequired, Required
        except ImportError:
            pytest.skip("Required/NotRequired qualifiers are unavailable on this Python install")

    assert get_origin(Required[str]).__name__ == "Required"
    assert get_origin(NotRequired[float]).__name__ == "NotRequired"

    class _RequiredPayload(TypedDict, total=False):
        mandatory: Required[str]
        optional: NotRequired[float]

    converted = _convert_value(
        _RequiredPayload, {"mandatory": "present", "optional": 0.5}
    )
    assert converted == {"mandatory": "present", "optional": 0.5}


def test_request_data_is_preserved_as_an_extra_keyword_for_variadic_callables():
    def process(**kwargs):
        return kwargs

    arguments = _build_arguments(
        process, "prompt", {"request_data": {"source": "explicit-extra"}}
    )
    assert arguments == {"request_data": {"source": "explicit-extra"}}


def test_abstract_mapping_and_set_annotations_are_converted_recursively():
    from collections.abc import MutableMapping, Set as AbstractSet

    mapping = _convert_value(MutableMapping[str, float], {"salience": 0.8})
    values = _convert_value(AbstractSet[int], [1, 2, 2, 3])
    assert mapping == {"salience": 0.8}
    assert values == {1, 2, 3}


def test_repository_root_generator_uses_explicit_run_root(tmp_path):
    token = _RANDOM_TEST_REPOSITORY_ROOT.set(tmp_path)
    try:
        rng = random.Random(75)
        rng_np = np.random.default_rng(75)
        assert _random_value(
            rng, rng_np, "repository_root", Path, package_name="ore_logger"
        ) == tmp_path
        assert _random_value(
            rng, rng_np, "repository_root", str, package_name="ore_logger"
        ) == str(tmp_path)
        generated_root = _random_value(
            rng, rng_np, "root", Path, package_name="ore_logger"
        )
        assert generated_root.parent == tmp_path
    finally:
        _RANDOM_TEST_REPOSITORY_ROOT.reset(token)


def test_typeddict_required_qualifiers_override_total_false_defaults():
    if Required is None or NotRequired is None:
        pytest.skip("Required/NotRequired qualifiers are unavailable on this Python install")

    class _RequiredOptionalPayload(TypedDict, total=False):
        required_value: Required[str]
        optional_value: NotRequired[float]

    rng = random.Random(76)
    rng_np = np.random.default_rng(76)
    generated = _random_value(
        rng, rng_np, "payload", _RequiredOptionalPayload
    )
    assert "required_value" in generated
    assert isinstance(generated["required_value"], str)
    assert "optional_value" not in generated or isinstance(
        generated["optional_value"], float
    )


def test_observer_caps_array_summaries_without_claiming_unobserved_arrays_finite():
    from observer import inspect_output

    output = {
        "arrays": [
            np.asarray([1.0], dtype=np.float32)
            for _ in range(5)
        ]
    }
    observation = inspect_output(output, max_arrays=2)
    assert observation["array_count"] == 2
    assert observation["total_array_count"] == 5
    assert observation["omitted_array_summaries"] == 3
    finite_check = next(
        item for item in observation["checks"]
        if item["check"] == "numeric_arrays_finite"
    )
    assert finite_check["value"] is None


def test_numeric_sequence_scanner_processes_large_lists_in_bounded_chunks():
    values = [float(index % 10) for index in range(9_000)]
    chunks = list(_numeric_arrays(values, path="output.values"))
    assert len(chunks) == 3
    assert sum(array.size for _, array in chunks) == len(values)


def test_json_mapping_keys_are_coerced_to_declared_scalar_key_types():
    from typing import Literal

    integer_mapping = _convert_value(Mapping[int, float], {"0": 0.2, "12": 0.8})
    boolean_mapping = _convert_value(Mapping[bool, str], {"true": "yes", "false": "no"})
    literal_mapping = _convert_value(
        Mapping[Literal[1, 2], str], {"1": "one", "2": "two"}
    )
    assert integer_mapping == {0: 0.2, 12: 0.8}
    assert boolean_mapping == {True: "yes", False: "no"}
    assert literal_mapping == {1: "one", 2: "two"}

    with pytest.raises(TypeError, match="not an integer"):
        _convert_value(Mapping[int, str], {"not-index": "value"})


def test_callable_fields_reject_json_strings_instead_of_failing_later():
    from typing import Callable

    with pytest.raises(TypeError, match="Python integration"):
        _convert_value(Callable[..., Any], "not-a-function")


def test_numeric_name_hints_prioritize_integer_counts_and_shapes():
    rng = random.Random(77)
    assert 1 <= auto_test._random_number(rng, "feature_count", integer=True) <= 12
    assert 1 <= auto_test._random_number(rng, "context_size", integer=True) <= 12
    assert 1 <= auto_test._random_number(rng, "input_shape", integer=True) <= 128
    assert 0 <= auto_test._random_number(rng, "token_categories_key", integer=True) <= 8


def test_randomized_path_field_handles_path_forward_reference(tmp_path):
    token = _RANDOM_TEST_ARTIFACT_DIR.set(tmp_path)
    try:
        rng = random.Random(78)
        rng_np = np.random.default_rng(78)
        generated = _random_value(
            rng, rng_np, "output_path", "Path", package_name="ore_logger"
        )
        assert isinstance(generated, Path)
        assert generated.parent == tmp_path
    finally:
        _RANDOM_TEST_ARTIFACT_DIR.reset(token)


def test_observer_inspects_outputs_with_no_numpy_arrays():
    observation = inspect_output({"status": "ok", "result": "done"})
    assert observation["array_count"] == 0
    assert observation["total_array_count"] == 0
    assert observation["omitted_array_summaries"] == 0
    finite_check = next(
        item for item in observation["checks"]
        if item["check"] == "numeric_arrays_finite"
    )
    assert finite_check["value"] is True


def test_observer_setup_failures_are_reported_as_workbench_errors(monkeypatch):
    class BrokenObserver:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            raise RuntimeError("synthetic observer setup failure")

        def __exit__(self, *args):
            return False

        def report(self):
            return {}

    monkeypatch.setattr(adapter, "PythonCallObserver", BrokenObserver)
    info = CallableInfo(
        label="builtins.len",
        module="builtins",
        name="len",
        signature="(obj, /)",
        doc="Return the length of an object.",
    )
    result = execute("builtins", info, "", {"obj": [1]})
    assert result["status"] == "error"
    assert result["failure_phase"] == "observer_setup"
    assert "synthetic observer setup failure" in result["error"]
    assert _classify_failure(
        result["error"], result["failure_phase"], result["traceback"]
    ) == "test_harness_error"
