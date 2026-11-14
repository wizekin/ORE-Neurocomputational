"""Reusable adapter for experimenting with Python component packages."""
from __future__ import annotations
import importlib, inspect, pkgutil, time, traceback
from dataclasses import dataclass
from typing import Any

@dataclass
class CallableInfo:
    label: str
    module: str
    name: str
    signature: str
    doc: str

def discover_callables(package_name: str):
    package = importlib.import_module(package_name)
    modules, errors = [package], []
    if hasattr(package, "__path__"):
        for item in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
            try: modules.append(importlib.import_module(item.name))
            except Exception as exc: errors.append(f"{item.name}: {type(exc).__name__}: {exc}")
    found = {}
    for module in modules:
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            if name.startswith("_") or obj.__module__ != module.__name__: continue
            try: sig = str(inspect.signature(obj))
            except (TypeError, ValueError): sig = "(signature unavailable)"
            found[(module.__name__, name)] = CallableInfo(
                f"{module.__name__}.{name}{sig}", module.__name__, name, sig,
                inspect.getdoc(obj) or "No docstring provided.")
    return sorted(found.values(), key=lambda x: x.label), errors

PROMPT_NAMES = {"prompt", "text", "sentence", "query", "message", "input_text",
                "stimulus", "word", "token", "text_input", "content"}

def build_arguments(function: Any, prompt: str, supplied: dict[str, Any]):
    kwargs, signature = dict(supplied), inspect.signature(function)
    for name, parameter in signature.parameters.items():
        if name not in kwargs and name.lower() in PROMPT_NAMES:
            kwargs[name] = prompt
            break
    required = [n for n, p in signature.parameters.items()
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
        function = getattr(importlib.import_module(info.module), info.name)
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
    if isinstance(value, dict): return {str(k): json_safe(v, depth+1) for k,v in value.items()}
    if isinstance(value, (list, tuple, set)): return [json_safe(v, depth+1) for v in value]
    if hasattr(value, "tolist"):
        try: return json_safe(value.tolist(), depth+1)
        except Exception: pass
    if hasattr(value, "__dict__"):
        try: return {"__type__": type(value).__name__, **json_safe(vars(value), depth+1)}
        except Exception: pass
    return repr(value)
