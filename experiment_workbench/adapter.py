"""Reusable discovery, dispatch, and execution adapter for Python packages."""
from __future__ import annotations

import dataclasses
from collections.abc import Iterable as ABCIterable, Mapping as ABCMapping, Sequence as ABCSequence
import importlib
import inspect
import pkgutil
import time
import traceback
import types
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, get_args, get_origin, get_type_hints

from observer import PythonCallObserver, inspect_output

@dataclass
class CallableInfo:
    label: str
    module: str
    name: str
    signature: str
    doc: str
    class_name: str | None = None

def _signature(obj):
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return "(signature unavailable)"

def discover_callables(package_name: str):
    """Find public functions and conventional methods on processor/model classes."""
    package = importlib.import_module(package_name)
    modules, errors = [package], []
    if hasattr(package, "__path__"):
        for item in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
            try:
                modules.append(importlib.import_module(item.name))
            except Exception as exc:
                errors.append(f"{item.name}: {type(exc).__name__}: {exc}")
    found = {}
    method_names = {"process", "run", "execute", "infer", "predict", "train",
                    "fit", "transform", "simulate", "validate", "forward",
                    "recognize", "initialize", "reset", "learn"}
    for module in modules:
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            if name.startswith("_") or obj.__module__ != module.__name__:
                continue
            sig = _signature(obj)
            found[(module.__name__, name, None)] = CallableInfo(
                f"{module.__name__}.{name}{sig}", module.__name__, name, sig,
                inspect.getdoc(obj) or "No docstring provided.")
        for class_name, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ != module.__name__ or class_name.startswith("_"):
                continue
            if not class_name.endswith(("Processor", "Simulator", "Model")):
                continue
            for method_name, method in inspect.getmembers(cls, inspect.isfunction):
                if method_name.startswith("_") or method_name not in method_names:
                    continue
                sig = _signature(method)
                label = f"{module.__name__}.{class_name}.{method_name}{sig}"
                found[(module.__name__, method_name, class_name)] = CallableInfo(
                    label, module.__name__, method_name, sig,
                    inspect.getdoc(method) or inspect.getdoc(cls) or "No docstring provided.",
                    class_name)
    return sorted(found.values(), key=lambda item: item.label), errors

PROMPT_NAMES = {"prompt", "text", "sentence", "query", "message", "input_text",
                "stimulus", "word", "token", "text_input", "content", "sequence",
                "characters", "orthography"}
PREVIOUS_PARAMETER_NAMES = {
    "n1_result", "source_n1_result", "n2_result", "n3_result", "n4_result",
    "n5_model", "n3_processor", "parent_model", "previous_result", "upstream_result", "prior_result",
}
PREVIOUS_FIELD_NAMES = PREVIOUS_PARAMETER_NAMES | {"n1_input", "n2_input", "n3_input", "n4_input"}

# Constructor parameters typed as Any/Sequence[Any] in older ORE packages still
# have well-defined inventory record classes. Keep those mappings in one adapter.
ORE_INVENTORY_TYPES = {
    ("ore_lexical_access", "lexicon"): ("ore_lexical_context.contracts", "LexicalEntry"),
    ("ore_lexical_access", "morphemes"): ("ore_lexical_context.contracts", "MorphemeEntry"),
    ("ore_lexical_context", "lexicon"): ("ore_lexical_context.contracts", "LexicalEntry"),
    ("ore_lexical_context", "morphemes"): ("ore_lexical_context.contracts", "MorphemeEntry"),
    ("ore_lexical_context", "g2p_rules"): ("ore_lexical_context.contracts", "G2PRule"),
    ("ore_reading", "lexicon"): ("ore_reading.contracts", "ReadingLexiconEntry"),
    ("ore_reading", "g2p_rules"): ("ore_reading.contracts", "G2PRule"),
    ("ore_semantic", "concepts"): ("ore_semantic.contracts", "SemanticConcept"),
}

def _is_dataclass_type(annotation):
    return inspect.isclass(annotation) and dataclasses.is_dataclass(annotation)

def _convert_value(annotation, value):
    """Convert JSON-shaped nested values into annotated ORE contracts."""
    if annotation is inspect.Parameter.empty or annotation is Any or value is None:
        return value
    if isinstance(annotation, str):
        return value
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in (types.UnionType, getattr(__import__("typing"), "Union")):
        if value is None:
            return None
        for candidate in args:
            if candidate is type(None):
                continue
            try:
                return _convert_value(candidate, value)
            except (TypeError, ValueError):
                continue
        return value
    if _is_dataclass_type(annotation):
        if isinstance(value, annotation):
            return value
        if not isinstance(value, dict):
            raise TypeError(f"{annotation.__name__} input must be a JSON object")
        return _build_dataclass(annotation, value, prompt="", previous_output=None)
    if origin in (list, tuple, set):
        item_type = args[0] if args else Any
        converted = [_convert_value(item_type, item) for item in value]
        return origin(converted) if origin is not tuple else tuple(converted)
    if origin in (ABCIterable, ABCSequence):
        item_type = args[0] if args else Any
        return [_convert_value(item_type, item) for item in value]
    if origin in (ABCMapping, dict) and isinstance(value, dict):
        key_type, value_type = args if len(args) == 2 else (Any, Any)
        return {_convert_value(key_type, key): _convert_value(value_type, item)
                for key, item in value.items()}
    try:
        import numpy as np
        if annotation is np.ndarray and not isinstance(value, np.ndarray):
            return np.asarray(value)
    except ImportError:
        pass
    return value

def _build_dataclass(cls, payload, prompt, previous_output):
    values = dict(payload or {})
    try:
        hints = get_type_hints(cls)
    except Exception:
        hints = {}
    fields = dataclasses.fields(cls)
    field_names = {field.name for field in fields}
    if not any(name in values for name in PROMPT_NAMES):
        for field in fields:
            if field.name.lower() in PROMPT_NAMES:
                values[field.name] = prompt
                break
    if "sequence_id" in field_names and "sequence_id" not in values:
        values["sequence_id"] = f"workbench-{uuid.uuid4().hex[:12]}"
    if previous_output is not None:
        for field in fields:
            if field.name in PREVIOUS_FIELD_NAMES and field.name not in values:
                values[field.name] = previous_output
    kwargs = {}
    for field in fields:
        if not field.init or field.name not in values:
            continue
        raw_value = values[field.name]
        if field.name == "n5_model" and isinstance(raw_value, dict) and raw_value.get("schema") == "ore.n5.plasticity_model.v1":
            from ore_n5.contracts import N5ModelState
            kwargs[field.name] = N5ModelState.from_dict(raw_value)
        else:
            kwargs[field.name] = _convert_value(
                hints.get(field.name, field.type), raw_value)
    required = [field.name for field in fields if field.init
                and field.default is dataclasses.MISSING
                and field.default_factory is dataclasses.MISSING
                and field.name not in kwargs]
    if required:
        raise ValueError(f"Cannot construct {cls.__name__}; provide fields: {', '.join(required)}")
    return cls(**kwargs)

def _type_hints(function):
    try:
        return get_type_hints(function)
    except Exception:
        return {}

def _convert_constructor_kwargs(cls, package_name, raw_kwargs, previous_output=None):
    signature = inspect.signature(cls)
    hints = _type_hints(cls.__init__)
    converted = {}
    if previous_output is not None:
        for name in PREVIOUS_PARAMETER_NAMES:
            if name in signature.parameters and name not in raw_kwargs:
                raw_kwargs[name] = previous_output
    for name, value in raw_kwargs.items():
        parameter = signature.parameters.get(name)
        annotation = hints.get(name, parameter.annotation if parameter else inspect.Parameter.empty)
        if (package_name, name) in ORE_INVENTORY_TYPES and isinstance(value, list):
            module_name, class_name = ORE_INVENTORY_TYPES[(package_name, name)]
            record_type = getattr(importlib.import_module(module_name), class_name)
            converted[name] = [_convert_value(record_type, item) for item in value]
        else:
            converted[name] = _convert_value(annotation, value)
    return converted

def build_arguments(function: Any, prompt: str, supplied: dict[str, Any],
                    previous_output: Any = None, runtime_kwargs: dict[str, Any] | None = None):
    """Build a callable's arguments from prompt, JSON, and optional upstream artifacts."""
    kwargs = dict(supplied)
    kwargs.update(runtime_kwargs or {})
    signature = inspect.signature(function)
    hints = _type_hints(function)
    request_payload = kwargs.pop("request_data", None)
    for name, parameter in signature.parameters.items():
        if name in kwargs:
            annotation = hints.get(name, parameter.annotation)
            if _is_dataclass_type(_annotation_dataclass(annotation)) and isinstance(kwargs[name], dict):
                kwargs[name] = _build_dataclass(_annotation_dataclass(annotation), kwargs[name], prompt, previous_output)
            continue
        annotation = hints.get(name, parameter.annotation)
        request_cls = _annotation_dataclass(annotation)
        if request_cls is not None:
            payload = request_payload if request_payload is not None else {}
            kwargs[name] = _build_dataclass(request_cls, payload, prompt, previous_output)
        elif name in PREVIOUS_PARAMETER_NAMES and previous_output is not None:
            kwargs[name] = previous_output
        elif name.lower() in PROMPT_NAMES:
            kwargs[name] = prompt
    # Convert typed JSON arrays/records after prompt and upstream mapping.
    for name, value in list(kwargs.items()):
        if name in signature.parameters:
            kwargs[name] = _convert_value(
                hints.get(name, signature.parameters[name].annotation), value)
    required = [name for name, parameter in signature.parameters.items()
                if parameter.default is inspect.Parameter.empty
                and parameter.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
                and name not in kwargs]
    if required:
        raise ValueError("Missing required argument(s): " + ", ".join(required)
                         + ". Add them in the Arguments (JSON) panel.")
    return kwargs

def _annotation_dataclass(annotation):
    if annotation is inspect.Parameter.empty or isinstance(annotation, str):
        return None
    origin = get_origin(annotation)
    candidates = get_args(annotation) if origin is not None else (annotation,)
    for candidate in candidates:
        if _is_dataclass_type(candidate):
            return candidate
    return None

def accepts_parameter(info: CallableInfo, parameter_name: str) -> bool:
    """Return whether a discovered callable explicitly accepts a named parameter."""
    module = importlib.import_module(info.module)
    target = getattr(module, info.name) if info.class_name is None else getattr(getattr(module, info.class_name), info.name)
    try:
        return parameter_name in inspect.signature(target).parameters
    except (TypeError, ValueError):
        return False

def execute(package_name: str, info: CallableInfo, prompt: str, supplied: dict[str, Any],
            previous_output: Any = None, runtime_kwargs: dict[str, Any] | None = None):
    """Invoke one component and return raw output plus structured execution observations."""
    run_id = uuid.uuid4().hex
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    observer = PythonCallObserver(package_name)
    result = {
        "run_id": run_id, "started_at": started_at, "component": package_name,
        "callable": info.label, "prompt": prompt, "status": "error", "elapsed_ms": None,
        "arguments": {}, "constructor_arguments": {}, "input_observation": {},
        "output": None, "error": None, "traceback": None,
        "profile": {}, "observation": {},
    }
    try:
        with observer:
            module = importlib.import_module(info.module)
            if info.class_name:
                cls = getattr(module, info.class_name)
                init_kwargs = dict(supplied.pop("__init__", {}) if "__init__" in supplied else {})
                if previous_output is not None and isinstance(previous_output, cls) and callable(getattr(previous_output, info.name, None)):
                    if init_kwargs:
                        raise ValueError("Cannot provide __init__ options when reusing an existing processor instance.")
                    instance = previous_output
                    result["reused_instance"] = True
                else:
                    instance = cls(**_convert_constructor_kwargs(cls, package_name, init_kwargs, previous_output))
                    result["reused_instance"] = False
                result["constructor_arguments"] = init_kwargs
                function = getattr(instance, info.name)
            else:
                function = getattr(module, info.name)
            kwargs = build_arguments(function, prompt, supplied, previous_output, runtime_kwargs)
            result["arguments"] = kwargs
            result["input_observation"] = inspect_output(kwargs)
            result["output"] = function(**kwargs)
            result["status"] = "success"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    finally:
        result["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
        result["profile"] = observer.report()
        result["observation"] = inspect_output(result["output"]) if result["status"] == "success" else {
            "output_type": None, "output_fields": [], "trace_rows": [],
            "trace_count": 0, "numerical_summaries": [], "array_count": 0,
            "warnings": [], "checks": [], "summary": {},
        }
    return result

def json_safe(value: Any, depth: int = 0):
    """Convert NumPy arrays, dataclasses and ordinary Python structures to JSON-safe values."""
    if depth > 10:
        return repr(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
            return str(value)
        return value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        try:
            return {"__type__": type(value).__name__,
                    **json_safe(dataclasses.asdict(value), depth + 1)}
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(key): json_safe(item, depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item, depth + 1) for item in value]
    if hasattr(value, "tolist"):
        try:
            return json_safe(value.tolist(), depth + 1)
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return {"__type__": type(value).__name__,
                    **json_safe(vars(value), depth + 1)}
        except Exception:
            pass
    return repr(value)
