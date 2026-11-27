"""Reusable discovery, dispatch, and execution adapter for Python packages."""
from __future__ import annotations

import asyncio
import dataclasses
import enum
import importlib
import inspect
import math
import pkgutil
import time
import traceback
import types
import uuid
from collections.abc import (
    Callable as ABCCallable,
    Collection as ABCCollection,
    Iterable as ABCIterable,
    Mapping as ABCMapping,
    MutableMapping as ABCMutableMapping,
    MutableSet as ABCMutableSet,
    Sequence as ABCSequence,
    Set as ABCSet,
)
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal, Union, get_args, get_origin, get_type_hints, is_typeddict
from uuid import UUID

try:
    from typing import NotRequired, Required
except ImportError:
    try:
        from typing_extensions import NotRequired, Required
    except ImportError:
        class _UnavailableTypingQualifier:
            pass

        NotRequired = Required = _UnavailableTypingQualifier

from observer import PythonCallObserver, _array_summary, inspect_output

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


def _type_hints(target):
    """Resolve annotations while retaining unresolved entries for explicit diagnostics."""
    try:
        return get_type_hints(target, include_extras=True)
    except Exception:
        return dict(getattr(target, "__annotations__", {}) or {})


def _is_union_origin(origin):
    return origin in (types.UnionType, Union)


def _annotation_dataclass(annotation):
    """Return the sole dataclass alternative in an annotation, if it is unambiguous."""
    if annotation is inspect.Parameter.empty or isinstance(annotation, str):
        return None
    origin = get_origin(annotation)
    candidates = get_args(annotation) if _is_union_origin(origin) else (annotation,)
    matches = [candidate for candidate in candidates if _is_dataclass_type(candidate)]
    return matches[0] if len(matches) == 1 else None


def _prompt_value(annotation, prompt: str, parameter_name: str):
    """Map the chat prompt only when it matches the target field's declared type."""
    if annotation is inspect.Parameter.empty or annotation is Any:
        return prompt

    if isinstance(annotation, str):
        if annotation in {"str", "Any"}:
            return prompt
        return _UNSET

    origin = get_origin(annotation)
    args = get_args(annotation)
    if _is_union_origin(origin):
        alternatives = [item for item in args if item is not type(None)]
        if len(alternatives) == 1:
            return _prompt_value(alternatives[0], prompt, parameter_name)
        return _UNSET

    if annotation is str:
        return prompt

    if origin in (list, tuple, set, frozenset, ABCSequence, ABCIterable):
        item_type = args[0] if args else Any
        if item_type not in (str, Any):
            return _UNSET
        if parameter_name.lower() in {"characters", "letters", "graphemes"}:
            tokens = list(prompt)
        else:
            tokens = prompt.split()
        if origin is tuple:
            return tuple(tokens)
        if origin is set:
            return set(tokens)
        if origin is frozenset:
            return frozenset(tokens)
        return tokens

    return _UNSET


def _literal_matches(value, candidate):
    return type(value) is type(candidate) and value == candidate


def _convert_mapping_key(annotation, value):
    """Convert JSON object keys to declared scalar key types without loosening values."""
    if isinstance(value, str):
        if annotation is int:
            text = value.strip()
            if text and (text.isdigit() or (text.startswith("-") and text[1:].isdigit())):
                return int(text)
            raise TypeError(f"mapping key {value!r} is not an integer")
        if annotation is float:
            try:
                parsed = float(value)
            except ValueError as exc:
                raise TypeError(f"mapping key {value!r} is not numeric") from exc
            if not math.isfinite(parsed):
                raise ValueError("mapping keys must be finite")
            return parsed
        if annotation is bool:
            if value.casefold() in {"true", "1"}:
                return True
            if value.casefold() in {"false", "0"}:
                return False
            raise TypeError(f"mapping key {value!r} is not a boolean")
        if get_origin(annotation) is Literal:
            for candidate in get_args(annotation):
                try:
                    converted = _convert_mapping_key(type(candidate), value)
                except (TypeError, ValueError):
                    continue
                if _literal_matches(converted, candidate):
                    return candidate
            raise ValueError(f"mapping key {value!r} is not a permitted literal")
    return _convert_value(annotation, value)


def _convert_value(annotation, value):
    """Convert JSON-shaped values according to their declared Python type.

    The converter is intentionally strict: it does not turn strings into arbitrary
    containers, silently retain dictionaries where dataclasses are expected, or
    discard contract fields that the caller mistyped.
    """
    if annotation is inspect.Parameter.empty or annotation is Any:
        return value

    if isinstance(annotation, str):
        simple = {
            "str": str, "int": int, "float": float, "bool": bool,
            "bytes": bytes, "Any": Any, "None": type(None),
            "Path": Path, "pathlib.Path": Path, "UUID": UUID,
            "datetime": datetime, "date": date,
        }
        if annotation in simple:
            return _convert_value(simple[annotation], value)
        if value is None and "None" in annotation:
            return None
        if not isinstance(value, ABCMapping):
            # Preserve an upstream object only when its runtime class name matches
            # an unresolved forward reference. Never pass an unrelated primitive
            # through merely because get_type_hints could not resolve the name.
            expected_names = set(
                annotation.replace("|", " ").replace("[", " ").replace("]", " ")
                .replace("'", " ").replace('"', " ").split()
            )
            actual_names = {base.__name__ for base in type(value).__mro__}
            if expected_names.intersection(actual_names):
                return value
        raise TypeError(
            f"Cannot safely convert this value because its type annotation "
            f"{annotation!r} could not be resolved"
        )

    origin = get_origin(annotation)
    args = get_args(annotation)

    if getattr(origin, "__name__", "") in {"Required", "NotRequired"}:
        return _convert_value(args[0], value)

    if origin is Annotated:
        base_type = args[0]
        converted = _convert_value(base_type, value)
        for constraint in args[1:]:
            metadata = constraint if isinstance(constraint, dict) else vars(constraint) if hasattr(constraint, "__dict__") else {}
            number = converted if isinstance(converted, (int, float)) and not isinstance(converted, bool) else None
            if number is not None:
                lower = metadata.get("ge", metadata.get("minimum", metadata.get("min")))
                upper = metadata.get("le", metadata.get("maximum", metadata.get("max")))
                exclusive_lower = metadata.get("gt")
                exclusive_upper = metadata.get("lt")
                if lower is not None and number < lower:
                    raise ValueError(f"value must be greater than or equal to {lower}")
                if upper is not None and number > upper:
                    raise ValueError(f"value must be less than or equal to {upper}")
                if exclusive_lower is not None and number <= exclusive_lower:
                    raise ValueError(f"value must be greater than {exclusive_lower}")
                if exclusive_upper is not None and number >= exclusive_upper:
                    raise ValueError(f"value must be less than {exclusive_upper}")
        return converted

    if origin is Literal:
        if any(_literal_matches(value, candidate) for candidate in args):
            return value
        raise ValueError(f"value {value!r} is not one of the permitted literal values {args!r}")

    if _is_union_origin(origin):
        if value is None:
            if type(None) in args:
                return None
            raise TypeError("None is not permitted by this input type")
        candidates = [candidate for candidate in args if candidate is not type(None)]
        if isinstance(value, dict):
            dataclass_candidates = [
                candidate for candidate in candidates if _is_dataclass_type(candidate)
            ]
            mapping_candidates = [
                candidate for candidate in candidates
                if get_origin(candidate) in (dict, ABCMapping)
                or candidate in (dict, ABCMapping)
            ]
            if len(dataclass_candidates) == 1 and not mapping_candidates:
                return _convert_value(dataclass_candidates[0], value)
            if len(dataclass_candidates) == 1 and mapping_candidates:
                candidate_cls = dataclass_candidates[0]
                names = {field.name for field in dataclasses.fields(candidate_cls) if field.init}
                if set(value).issubset(names):
                    return _convert_value(candidate_cls, value)

        direct = []
        for candidate in candidates:
            if _is_dataclass_type(candidate) and isinstance(value, candidate):
                direct.append(candidate)
            elif inspect.isclass(candidate) and candidate not in (dict, list, tuple, set):
                try:
                    if isinstance(value, candidate):
                        direct.append(candidate)
                except TypeError:
                    pass
        if len(direct) == 1:
            return value

        errors = []
        successes = []
        for candidate in candidates:
            try:
                successes.append(_convert_value(candidate, value))
            except (TypeError, ValueError) as exc:
                errors.append(exc)
        if len(successes) == 1:
            return successes[0]
        if len(successes) > 1:
            # Prefer an exact built-in container match over coercing it into another
            # union alternative; ambiguous dataclass alternatives remain an error.
            exact = [
                candidate for candidate in candidates
                if (candidate is dict and isinstance(value, dict))
                or (candidate is list and isinstance(value, list))
                or (candidate is tuple and isinstance(value, tuple))
                or (candidate is set and isinstance(value, set))
            ]
            if len(exact) == 1:
                return _convert_value(exact[0], value)
            raise TypeError("input matches more than one alternative of a union type")
        if errors:
            raise errors[0]
        return value

    if value is None:
        if annotation is type(None):
            return None
        raise TypeError(f"None is not valid for {annotation!r}")

    if inspect.isclass(annotation) and issubclass(annotation, enum.Enum):
        if isinstance(value, annotation):
            return value
        try:
            return annotation(value)
        except (TypeError, ValueError):
            if isinstance(value, str):
                try:
                    return annotation[value]
                except KeyError:
                    pass
            permitted = [member.value for member in annotation]
            raise ValueError(f"{value!r} is not a valid {annotation.__name__}; expected one of {permitted!r}")

    if _is_dataclass_type(annotation):
        if isinstance(value, annotation):
            return value
        if not isinstance(value, Mapping):
            raise TypeError(f"{annotation.__name__} input must be a JSON object")
        if (
            annotation.__name__ == "N5ModelState"
            and value.get("schema") == "ore.n5.plasticity_model.v1"
        ):
            from ore_n5.contracts import N5ModelState
            return N5ModelState.from_dict(dict(value))
        return _build_dataclass(annotation, value, prompt="", previous_output=None)

    if is_typeddict(annotation):
        if not isinstance(value, ABCMapping):
            raise TypeError(f"{annotation.__name__} input must be a JSON object")
        try:
            hints = get_type_hints(annotation, include_extras=True)
        except Exception:
            hints = dict(getattr(annotation, "__annotations__", {}) or {})
        allowed = set(hints)
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise TypeError(
                f"Unexpected field(s) for {annotation.__name__}: {', '.join(unknown)}"
            )
        required = set(getattr(annotation, "__required_keys__", allowed))
        for field_name, field_annotation in hints.items():
            qualifier = get_origin(field_annotation)
            if qualifier is Required:
                required.add(field_name)
            elif qualifier is NotRequired:
                required.discard(field_name)
        missing = sorted(required - set(value))
        if missing:
            raise ValueError(
                f"Missing required field(s) for {annotation.__name__}: {', '.join(missing)}"
            )
        return {
            key: _convert_value(hints.get(key, Any), item)
            for key, item in value.items()
        }

    if inspect.isclass(annotation) and issubclass(annotation, tuple) and hasattr(annotation, "_fields"):
        field_names = tuple(annotation._fields)
        field_types = _type_hints(annotation)
        if isinstance(value, ABCMapping):
            unknown = sorted(set(value) - set(field_names))
            defaults = getattr(annotation, "_field_defaults", {})
            missing = [
                name for name in field_names
                if name not in value and name not in defaults
            ]
            if unknown or missing:
                details = []
                if unknown:
                    details.append("unexpected: " + ", ".join(unknown))
                if missing:
                    details.append("missing: " + ", ".join(missing))
                raise TypeError(
                    f"Invalid {annotation.__name__} fields ({'; '.join(details)})"
                )
            values = [
                value[name] if name in value else defaults[name]
                for name in field_names
            ]
        elif isinstance(value, (list, tuple)) and not isinstance(value, (str, bytes)):
            if len(value) != len(field_names):
                raise ValueError(
                    f"{annotation.__name__} requires {len(field_names)} values"
                )
            values = list(value)
        else:
            raise TypeError(
                f"{annotation.__name__} input must be an object or positional sequence"
            )
        converted = [
            _convert_value(field_types.get(name, Any), item)
            for name, item in zip(field_names, values)
        ]
        return annotation(*converted)

    if annotation is Path:
        if isinstance(value, Path):
            return value
        if isinstance(value, str):
            return Path(value)
        raise TypeError("Path input must be a string or pathlib.Path")

    if annotation is UUID:
        if isinstance(value, UUID):
            return value
        if isinstance(value, str):
            return UUID(value)
        raise TypeError("UUID input must be a UUID string")

    if annotation is datetime:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        raise TypeError("datetime input must be an ISO-8601 string or datetime")

    if annotation is date:
        if isinstance(value, date) and not isinstance(value, datetime):
            return value
        if isinstance(value, str):
            return date.fromisoformat(value)
        raise TypeError("date input must be an ISO date string or date")

    if origin is ABCCallable or annotation is ABCCallable:
        if callable(value):
            return value
        raise TypeError(
            "callable inputs must be supplied by a Python integration, not JSON"
        )

    if annotation is bool:
        if isinstance(value, bool):
            return value
        raise TypeError("boolean input must be true or false")

    if annotation is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("integer input must be a JSON integer")
        return value

    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("numeric input must be a JSON number")
        return float(value)

    if annotation is str:
        if isinstance(value, str):
            return value
        raise TypeError("text input must be a string")

    if annotation is bytes:
        if isinstance(value, bytes):
            return value
        raise TypeError("bytes input must be bytes; JSON text is not decoded implicitly")

    if origin in (dict, ABCMapping, ABCMutableMapping) or annotation in (
        dict, ABCMapping, ABCMutableMapping
    ):
        if not isinstance(value, ABCMapping):
            raise TypeError("mapping input must be a JSON object")
        key_type, value_type = args if len(args) == 2 else (Any, Any)
        return {
            _convert_mapping_key(key_type, key): _convert_value(value_type, item)
            for key, item in value.items()
        }

    if origin in (
        list, tuple, set, frozenset, ABCIterable, ABCSequence,
        ABCCollection, ABCSet, ABCMutableSet,
    ) or annotation in (ABCIterable, ABCSequence, ABCCollection, ABCSet, ABCMutableSet):
        if isinstance(value, (str, bytes, bytearray, ABCMapping)) or not isinstance(value, ABCIterable):
            raise TypeError("sequence input must be an array-like value, not text or a mapping")
        raw_items = list(value)
        if origin is tuple and args and args[-1] is not Ellipsis:
            if len(raw_items) != len(args):
                raise ValueError(f"tuple input must contain exactly {len(args)} items")
            converted = [
                _convert_value(item_type, item)
                for item_type, item in zip(args, raw_items)
            ]
        else:
            item_type = args[0] if args else Any
            converted = [_convert_value(item_type, item) for item in raw_items]
        if origin is tuple:
            return tuple(converted)
        if origin in (set, ABCSet, ABCMutableSet) or annotation in (ABCSet, ABCMutableSet):
            return set(converted)
        if origin is frozenset:
            return frozenset(converted)
        if origin in (ABCIterable, ABCSequence):
            return converted
        return list(converted)

    try:
        import numpy as np
        if annotation is np.ndarray and not isinstance(value, np.ndarray):
            return np.asarray(value)
        if inspect.isclass(annotation) and issubclass(annotation, np.generic):
            return annotation(value)
    except (ImportError, TypeError, ValueError):
        pass

    if inspect.isclass(annotation):
        try:
            if isinstance(value, annotation):
                return value
        except TypeError:
            # Non-runtime-checkable Protocols and some abstract typing classes
            # cannot be checked with isinstance. Preserve an upstream object rather
            # than rejecting it solely because the annotation is non-runtime-checkable.
            return value
        raise TypeError(f"input must be an instance of {annotation.__name__}")
    return value


def _build_dataclass(cls, payload, prompt, previous_output):
    """Build a dataclass from a strict JSON mapping and apply Workbench defaults."""
    if not isinstance(payload, ABCMapping):
        raise TypeError(f"{cls.__name__} input must be a JSON object")
    values = dict(payload)
    try:
        hints = get_type_hints(cls, include_extras=True)
    except Exception:
        hints = _type_hints(cls)
    fields = dataclasses.fields(cls)
    init_fields = {field.name: field for field in fields if field.init}
    unknown = sorted(set(values) - set(init_fields))
    if unknown:
        raise TypeError(
            f"Unexpected field(s) for {cls.__name__}: {', '.join(unknown)}"
        )

    if not any(name in values for name in PROMPT_NAMES):
        for field in fields:
            if not field.init or field.name.lower() not in PROMPT_NAMES:
                continue
            prompt_value = _prompt_value(
                hints.get(field.name, field.type), prompt, field.name
            )
            if prompt_value is not _UNSET:
                values[field.name] = prompt_value
                break
    if "sequence_id" in init_fields and "sequence_id" not in values:
        values["sequence_id"] = f"workbench-{uuid.uuid4().hex[:12]}"
    if previous_output is not None:
        for field in fields:
            if field.init and field.name in PREVIOUS_FIELD_NAMES and field.name not in values:
                values[field.name] = previous_output

    kwargs = {}
    for field in fields:
        if not field.init or field.name not in values:
            continue
        raw_value = values[field.name]
        kwargs[field.name] = _convert_value(
            hints.get(field.name, field.type), raw_value
        )

    required = [
        field.name for field in fields
        if field.init and field.default is dataclasses.MISSING
        and field.default_factory is dataclasses.MISSING
        and field.name not in kwargs
    ]
    if required:
        raise ValueError(
            f"Cannot construct {cls.__name__}; provide fields: {', '.join(required)}"
        )
    return cls(**kwargs)


def _convert_constructor_kwargs(cls, package_name, raw_kwargs, previous_output=None):
    """Validate and convert constructor keyword arguments without mutating caller data."""
    if not isinstance(raw_kwargs, ABCMapping):
        raise TypeError("processor constructor arguments must be a JSON object")
    raw_values = dict(raw_kwargs)
    signature = inspect.signature(cls)
    hints = _type_hints(cls.__init__)
    if previous_output is not None:
        for name in PREVIOUS_PARAMETER_NAMES:
            if name in signature.parameters and name not in raw_values:
                raw_values[name] = previous_output

    has_var_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    allowed = {
        name for name, parameter in signature.parameters.items()
        if parameter.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    }
    unexpected = sorted(set(raw_values) - allowed) if not has_var_kwargs else []
    if unexpected:
        raise TypeError(
            f"Unexpected constructor argument(s) for {cls.__name__}: {', '.join(unexpected)}"
        )

    converted = {}
    for name, value in raw_values.items():
        parameter = signature.parameters.get(name)
        if parameter is None:
            converted[name] = value
            continue
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            raise TypeError(
                f"Constructor parameter {name!r} is positional-only and cannot be supplied in the JSON arguments panel"
            )
        annotation = hints.get(name, parameter.annotation)
        if (package_name, name) in ORE_INVENTORY_TYPES and isinstance(value, list):
            module_name, class_name = ORE_INVENTORY_TYPES[(package_name, name)]
            record_type = getattr(importlib.import_module(module_name), class_name)
            converted[name] = [_convert_value(record_type, item) for item in value]
        else:
            converted[name] = _convert_value(annotation, value)

    try:
        signature.bind(**converted)
    except TypeError as exc:
        raise TypeError(f"Invalid constructor arguments for {cls.__name__}: {exc}") from exc
    return converted


def build_arguments(function: Any, prompt: str, supplied: dict[str, Any],
                    previous_output: Any = None, runtime_kwargs: dict[str, Any] | None = None):
    """Build and type-check a callable's arguments from prompt, JSON, and upstream data."""
    if not isinstance(supplied, ABCMapping):
        raise TypeError("callable arguments must be a JSON object")
    if runtime_kwargs is not None and not isinstance(runtime_kwargs, ABCMapping):
        raise TypeError("runtime arguments must be a mapping")

    signature = inspect.signature(function)
    hints = _type_hints(function)
    kwargs = dict(supplied)
    runtime = dict(runtime_kwargs or {})
    collisions = sorted(set(kwargs).intersection(runtime))
    if collisions:
        raise ValueError(
            "Runtime argument(s) conflict with explicitly supplied argument(s): "
            + ", ".join(collisions)
            + ". Remove the duplicate from the JSON arguments or remove the runtime input."
        )
    kwargs.update(runtime)

    has_var_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    declared_request_parameters = [
        name for name, parameter in signature.parameters.items()
        if _annotation_dataclass(hints.get(name, parameter.annotation)) is not None
    ]
    request_payload = _UNSET
    if (
        "request_data" in kwargs
        and "request_data" not in signature.parameters
        and (declared_request_parameters or not has_var_kwargs)
    ):
        request_payload = kwargs.pop("request_data")

    for name, parameter in signature.parameters.items():
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if name in kwargs:
            annotation = hints.get(name, parameter.annotation)
            request_cls = _annotation_dataclass(annotation)
            if request_cls is not None and isinstance(kwargs[name], dict):
                kwargs[name] = _build_dataclass(
                    request_cls, kwargs[name], prompt, previous_output
                )
            continue

        annotation = hints.get(name, parameter.annotation)
        request_cls = _annotation_dataclass(annotation)
        if request_cls is not None:
            payload = {} if request_payload is _UNSET else request_payload
            if not isinstance(payload, ABCMapping):
                raise TypeError("request_data must be a JSON object for a dataclass request")
            kwargs[name] = _build_dataclass(
                request_cls, payload, prompt, previous_output
            )
        elif name in PREVIOUS_PARAMETER_NAMES and previous_output is not None:
            kwargs[name] = _convert_value(annotation, previous_output)
        elif name.lower() in PROMPT_NAMES:
            prompt_value = _prompt_value(annotation, prompt, name)
            if prompt_value is not _UNSET:
                kwargs[name] = _convert_value(annotation, prompt_value)

    if request_payload is not _UNSET:
        request_parameters = declared_request_parameters
        if not request_parameters:
            raise ValueError(
                "request_data was supplied, but this callable has no dataclass request parameter "
                "to receive it"
            )
        if len(request_parameters) != 1:
            raise ValueError(
                "request_data is ambiguous because this callable has multiple dataclass request "
                "parameters; supply each named request argument explicitly"
            )
        request_name = request_parameters[0]
        if request_name in supplied or request_name in runtime:
            raise ValueError(
                f"request_data conflicts with the explicitly supplied {request_name!r} argument"
            )

    accepted = {
        name for name, parameter in signature.parameters.items()
        if parameter.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    }
    unexpected = sorted(set(kwargs) - accepted) if not has_var_kwargs else []
    if unexpected:
        raise TypeError(
            f"Unexpected callable argument(s) for {_signature(function)}: {', '.join(unexpected)}"
        )

    # Conversion is applied to every explicitly supplied value, not just prompt-built fields.
    for name, value in list(kwargs.items()):
        parameter = signature.parameters.get(name)
        if parameter is None or parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD
        ):
            continue
        kwargs[name] = _convert_value(hints.get(name, parameter.annotation), value)
    return kwargs


def _bind_invocation(function, kwargs):
    """Convert named JSON values for positional-only parameters and bind the full call."""
    signature = inspect.signature(function)
    parameters = list(signature.parameters.values())
    positional_only = [
        index for index, parameter in enumerate(parameters)
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY and parameter.name in kwargs
    ]
    last_positional_only = max(positional_only, default=-1)
    args = []
    call_kwargs = dict(kwargs)
    for parameter in parameters[:last_positional_only + 1]:
        if parameter.kind is not inspect.Parameter.POSITIONAL_ONLY:
            continue
        if parameter.name in call_kwargs:
            args.append(call_kwargs.pop(parameter.name))
        elif parameter.default is not inspect.Parameter.empty:
            args.append(parameter.default)
        else:
            raise TypeError(
                f"Missing positional-only argument {parameter.name!r} for {_signature(function)}"
            )
    signature.bind(*args, **call_kwargs)
    return args, call_kwargs


def accepts_parameter(info: CallableInfo, parameter_name: str) -> bool:
    """Return whether a discovered callable explicitly or variadically accepts a parameter."""
    module = importlib.import_module(info.module)
    target = (
        getattr(module, info.name)
        if info.class_name is None
        else getattr(getattr(module, info.class_name), info.name)
    )
    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):
        return False
    return parameter_name in signature.parameters


async def _await_result(value):
    return await value


def _run_awaitable(value):
    """Resolve an awaitable when no event loop is already running in this thread."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_await_result(value))
    raise RuntimeError(
        "The Workbench cannot synchronously await this callable inside an already-running "
        "event loop; call it from a synchronous entry point instead."
    )


def execute(package_name: str, info: CallableInfo, prompt: str, supplied: dict[str, Any],
            previous_output: Any = None, runtime_kwargs: dict[str, Any] | None = None,
            on_event=None, trace_lines: bool = False,
            log_directory: str | Path | None = None):
    """Invoke a component and return output, observations, and optional narrative logs."""
    run_id = uuid.uuid4().hex
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    logger = None
    logging_error = None
    if log_directory is not None:
        try:
            from narration import OREExecutionLogger
            logger = OREExecutionLogger(
                log_directory, run_id, package_name, info.label, started_at,
                on_event=on_event,
            )
        except Exception as exc:
            logging_error = f"{type(exc).__name__}: {exc}"
    observer_callback = logger.log_python_event if logger is not None else on_event
    observer = None
    provided_argument_keys = (
        sorted(str(key) for key in supplied)
        if isinstance(supplied, ABCMapping) else []
    )
    result = {
        "run_id": run_id, "started_at": started_at, "component": package_name,
        "callable": info.label, "prompt": prompt, "status": "error", "elapsed_ms": None,
        "arguments": {}, "provided_argument_keys": provided_argument_keys,
        "constructor_arguments": {}, "input_observation": {},
        "output": None, "error": None, "traceback": None,
        "failure_phase": None, "profile": {}, "observation": {},
        "trace_lines_enabled": trace_lines,
    }
    if logging_error:
        result["logging_error"] = logging_error

    phase = "argument_building"
    supplied_values = dict(supplied) if isinstance(supplied, ABCMapping) else supplied
    try:
        if not isinstance(supplied_values, ABCMapping):
            raise TypeError("callable arguments must be a JSON object")
        if runtime_kwargs is not None and not isinstance(runtime_kwargs, ABCMapping):
            raise TypeError("runtime arguments must be a mapping")
        supplied_values = dict(supplied_values)
        phase = "observer_setup"
        observer = PythonCallObserver(
            package_name, on_event=observer_callback, trace_lines=trace_lines
        )
        with observer:
            phase = "module_import"
            module = importlib.import_module(info.module)
            if info.class_name:
                phase = "constructor_argument_conversion"
                cls = getattr(module, info.class_name)
                init_kwargs = dict(supplied_values.pop("__init__", {}) or {})
                if (
                    previous_output is not None
                    and isinstance(previous_output, cls)
                    and callable(getattr(previous_output, info.name, None))
                ):
                    if init_kwargs:
                        raise ValueError(
                            "Cannot provide __init__ options when reusing an existing processor instance."
                        )
                    instance = previous_output
                    result["reused_instance"] = True
                else:
                    converted_init = _convert_constructor_kwargs(
                        cls, package_name, init_kwargs, previous_output
                    )
                    result["converted_constructor_arguments"] = converted_init
                    phase = "constructor_execution"
                    instance = cls(**converted_init)
                    result["reused_instance"] = False
                result["constructor_arguments"] = init_kwargs
                function = getattr(instance, info.name)
            else:
                function = getattr(module, info.name)

            effective_runtime_kwargs = dict(runtime_kwargs or {})
            function_parameters = inspect.signature(function).parameters
            if (
                logger is not None
                and "on_operation" in function_parameters
                and "on_operation" not in supplied_values
            ):
                effective_runtime_kwargs.setdefault("on_operation", logger.log_operation)

            phase = "argument_building"
            kwargs = build_arguments(
                function, prompt, supplied_values, previous_output, effective_runtime_kwargs
            )
            result["arguments"] = kwargs

            phase = "input_observation"
            try:
                result["input_observation"] = inspect_output(kwargs)
            except Exception as exc:
                result["input_observation"] = {}
                result["input_observation_error"] = f"{type(exc).__name__}: {exc}"

            phase = "call_argument_binding"
            positional_args, call_kwargs = _bind_invocation(function, kwargs)

            phase = "callable_execution"
            output = function(*positional_args, **call_kwargs)
            if inspect.isawaitable(output):
                output = _run_awaitable(output)
            result["output"] = output
            result["status"] = "success"
            result["failure_phase"] = None
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        result["failure_phase"] = phase
    finally:
        result["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
        if observer is None:
            result["profile"] = {}
        else:
            try:
                result["profile"] = observer.report()
            except Exception as exc:
                result["profile"] = {}
                result["profiling_error"] = f"{type(exc).__name__}: {exc}"

        empty_observation = {
            "output_type": None, "output_fields": [], "trace_rows": [],
            "trace_count": 0, "numerical_summaries": [], "array_count": 0,
            "warnings": [], "checks": [], "summary": {},
        }
        if result["status"] == "success":
            try:
                result["observation"] = inspect_output(result["output"])
            except Exception as exc:
                result["observation"] = dict(empty_observation)
                result["observation_error"] = f"{type(exc).__name__}: {exc}"
        else:
            result["observation"] = empty_observation

        if logger is not None:
            try:
                logger.log_component_traces(result["observation"].get("trace_rows", []))
                result["logs"] = logger.finish(
                    result["status"], result["elapsed_ms"],
                    result["observation"].get("output_type"),
                    result.get("error"),
                )
            except Exception as exc:
                result["logging_error"] = f"{type(exc).__name__}: {exc}"
    return result

def json_safe(value: Any, depth: int = 0):
    """Create a bounded JSON-safe preview without expanding large numerical payloads."""
    if depth > 10:
        return repr(value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return str(value)
        return value
    if isinstance(value, enum.Enum):
        return json_safe(value.value, depth + 1)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, bytes):
        limit = 128
        return {
            "__bytes_hex__": value[:limit].hex(),
            "size": len(value),
            "truncated": len(value) > limit,
        }

    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            array = np.asarray(value)
            summary = _array_summary(array, "array") or {}
            numeric = bool(
                np.issubdtype(array.dtype, np.number) or array.dtype == np.bool_
            )
            sample_values = list(array.flat[:24]) if array.size else []
            if np.iscomplexobj(array):
                sample = [
                    {"real": float(item.real), "imaginary": float(item.imag)}
                    for item in sample_values
                ]
                minimum = summary.get("magnitude_min")
                maximum = summary.get("magnitude_max")
                mean = summary.get("magnitude_mean")
                summary_basis = "magnitude"
            else:
                sample = [json_safe(item, depth + 1) for item in sample_values]
                minimum = summary.get("min")
                maximum = summary.get("max")
                mean = summary.get("mean")
                summary_basis = "value"
            return {
                "__array__": True,
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "size": int(array.size),
                "numeric": numeric,
                "finite": summary.get("finite"),
                "minimum": minimum,
                "maximum": maximum,
                "mean": mean,
                "summary_basis": summary_basis,
                "sample": sample,
                "truncated": int(array.size) > len(sample_values),
            }
    except ImportError:
        pass

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        try:
            return {
                "__type__": type(value).__name__,
                **{
                    field.name: json_safe(getattr(value, field.name), depth + 1)
                    for field in dataclasses.fields(value)
                },
            }
        except Exception:
            pass
    if isinstance(value, ABCMapping):
        items = list(value.items())
        result = {
            str(key): json_safe(item, depth + 1)
            for key, item in items[:100]
        }
        if len(items) > 100:
            result["__omitted_items__"] = len(items) - 100
        return result
    if isinstance(value, (list, tuple)):
        items = [json_safe(item, depth + 1) for item in value[:100]]
        if len(value) > 100:
            items.append(f"... {len(value) - 100} more value(s)")
        return items
    if isinstance(value, (set, frozenset)):
        ordered = sorted(value, key=repr)
        items = [json_safe(item, depth + 1) for item in ordered[:100]]
        if len(ordered) > 100:
            items.append(f"... {len(ordered) - 100} more value(s)")
        return items
    if hasattr(value, "item") and callable(value.item):
        try:
            return json_safe(value.item(), depth + 1)
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return {
                "__type__": type(value).__name__,
                **json_safe(vars(value), depth + 1),
            }
        except Exception:
            pass
    return repr(value)
