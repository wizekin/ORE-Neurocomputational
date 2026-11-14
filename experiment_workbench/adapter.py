"""Reusable adapter for experimenting with Python component packages."""
from __future__ import annotations
import dataclasses, importlib, inspect, pkgutil, time, traceback
from dataclasses import dataclass
from typing import Any, get_args, get_origin, get_type_hints

@dataclass
class CallableInfo:
    label: str
    module: str
    name: str
    signature: str
    doc: str
    class_name: str | None = None

def _signature(obj):
    try: return str(inspect.signature(obj))
    except (TypeError, ValueError): return "(signature unavailable)"

def discover_callables(package_name: str):
    package = importlib.import_module(package_name)
    modules, errors = [package], []
    if hasattr(package, "__path__"):
        for item in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
            try: modules.append(importlib.import_module(item.name))
            except Exception as exc: errors.append(f"{item.name}: {type(exc).__name__}: {exc}")
    found = {}
    method_names = {"process", "run", "execute", "infer", "predict", "train",
                    "fit", "transform", "simulate", "validate", "forward"}
    for module in modules:
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            if name.startswith("_") or obj.__module__ != module.__name__: continue
            sig = _signature(obj)
            found[(module.__name__, name, None)] = CallableInfo(
                f"{module.__name__}.{name}{sig}", module.__name__, name, sig,
                inspect.getdoc(obj) or "No docstring provided.")
        for class_name, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ != module.__name__ or class_name.startswith("_"): continue
            if not (class_name.endswith(("Processor", "Simulator", "Model"))): continue
            for method_name, method in inspect.getmembers(cls, inspect.isfunction):
                if method_name.startswith("_") or method_name not in method_names: continue
                sig = _signature(method)
                label = f"{module.__name__}.{class_name}.{method_name}{sig}"
                found[(module.__name__, method_name, class_name)] = CallableInfo(
                    label, module.__name__, method_name, sig,
                    inspect.getdoc(method) or inspect.getdoc(cls) or "No docstring provided.",
                    class_name)
    return sorted(found.values(), key=lambda x: x.label), errors

PROMPT_NAMES = {"prompt", "text", "sentence", "query", "message", "input_text",
                "stimulus", "word", "token", "text_input", "content", "sequence",
                "characters", "orthography"}

def _dataclass_from_prompt(cls, prompt, supplied):
    values = dict(supplied or {})
    fields = dataclasses.fields(cls)
    names = {f.name for f in fields}
    if not any(k in values for k in PROMPT_NAMES):
        for f in fields:
            if f.name.lower() in PROMPT_NAMES:
                values[f.name] = prompt
                break
    required = [f.name for f in fields if f.init and f.default is dataclasses.MISSING
                and f.default_factory is dataclasses.MISSING and f.name not in values]
    if required:
        raise ValueError(f"Cannot construct {cls.__name__}; provide fields: {', '.join(required)}")
    return cls(**{k:v for k,v in values.items() if k in names})

def _resolve_dataclass(annotation):
    if annotation is inspect.Parameter.empty: return None
    if isinstance(annotation, str): return None
    origin = get_origin(annotation)
    candidates = get_args(annotation) if origin is not None else (annotation,)
    for candidate in candidates:
        if inspect.isclass(candidate) and dataclasses.is_dataclass(candidate):
            return candidate
    return None

def build_arguments(function: Any, prompt: str, supplied: dict[str, Any]):
    kwargs, signature = dict(supplied), inspect.signature(function)
    try: hints = get_type_hints(function)
    except Exception: hints = {}
    for name, parameter in signature.parameters.items():
        if name in kwargs: continue
        annotation = hints.get(name, parameter.annotation)
        request_cls = _resolve_dataclass(annotation)
        if request_cls:
            nested = kwargs.pop("request_data", kwargs.pop(name, {}))
            if not isinstance(nested, dict): raise ValueError(f"{name} must be a JSON object")
            kwargs[name] = _dataclass_from_prompt(request_cls, prompt, nested)
        elif name.lower() in PROMPT_NAMES:
            kwargs[name] = prompt
    required = [n for n,p in signature.parameters.items()
                if p.default is inspect.Parameter.empty
                and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
                and n not in kwargs]
    if required:
        raise ValueError("Missing required argument(s): " + ", ".join(required)
                         + ". Add them in the Arguments (JSON) panel.")
    return kwargs

def execute(package_name: str, info: CallableInfo, prompt: str, supplied: dict[str, Any]):
    started = time.perf_counter()
    result = {"component": package_name, "callable": info.label, "status": "error",
              "elapsed_ms": None, "arguments": {}, "output": None, "error": None,
              "traceback": None}
    try:
        module = importlib.import_module(info.module)
        if info.class_name:
            cls = getattr(module, info.class_name)
            init_kwargs = supplied.pop("__init__", {}) if "__init__" in supplied else {}
            instance = cls(**init_kwargs)
            function = getattr(instance, info.name)
        else:
            function = getattr(module, info.name)
        kwargs = build_arguments(function, prompt, supplied)
        result["arguments"] = kwargs
        result["output"] = function(**kwargs)
        result["status"] = "success"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    finally:
        result["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
    return result

def json_safe(value: Any, depth: int = 0):
    if depth > 8: return repr(value)
    if value is None or isinstance(value, (str, int, float, bool)): return value
    if isinstance(value, dict): return {str(k):json_safe(v,depth+1) for k,v in value.items()}
    if isinstance(value,(list,tuple,set)): return [json_safe(v,depth+1) for v in value]
    if hasattr(value,"tolist"):
        try: return json_safe(value.tolist(),depth+1)
        except Exception: pass
    if dataclasses.is_dataclass(value):
        try: return {"__type__":type(value).__name__,**json_safe(dataclasses.asdict(value),depth+1)}
        except Exception: pass
    if hasattr(value,"__dict__"):
        try: return {"__type__":type(value).__name__,**json_safe(vars(value),depth+1)}
        except Exception: pass
    return repr(value)
