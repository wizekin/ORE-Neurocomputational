"""Randomized theory-oriented evaluation for the ORE Workbench.

The runner does not contain a prompt bank, fixed reference answers, or expected
output examples. It generates random inputs from callable signatures and typed
contracts, then records execution evidence and oracle-free invariants.
"""
from __future__ import annotations

import ast
import dataclasses
import enum
import importlib
import inspect
import json
import math
import pkgutil
import random
import secrets
import traceback
import types
import uuid
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, get_args, get_origin, get_type_hints

import numpy as np

from adapter import CallableInfo, execute, json_safe

_UNSET = object()
SAFE_METHODS = {
    "process", "run", "execute", "infer", "predict", "recognize", "transform",
    "simulate", "forward", "parse", "compose", "reason", "plan", "encode",
    "decode", "retrieve", "recall", "query", "classify", "validate",
    "evaluate", "convert", "adapt", "build", "create", "search", "step",
    "resolve", "integrate", "route", "dispatch", "observe", "score",
    "fit", "train", "learn", "learn_from", "update", "revise",
    "consolidate", "select", "attend", "allocate", "reconstruct",
}
UNSAFE_METHODS = {
    "save", "load", "export", "delete", "remove", "clear", "write",
    "persist", "connect", "download", "upload", "shutdown", "close",
    "reset", "reset_state", "initialize", "init",
}
TEXT_NAMES = {
    "prompt", "text", "sentence", "query", "message", "input_text",
    "stimulus", "word", "token", "text_input", "content", "sequence",
    "characters", "orthography", "utterance", "transcript", "description",
}
IMAGE_NAMES = {
    "image", "pixels", "frame", "grayscale_image", "visual_input",
    "image_array", "input_image",
}
PREVIOUS_NAMES = {
    "n1_result", "source_n1_result", "n2_result", "n3_result", "n4_result",
    "n5_model", "n3_processor", "parent_model", "previous_result",
    "upstream_result", "prior_result",
}
OPTIONAL_PARAMETER_PROBABILITY = 0.35
REGISTRY_TOKENS = ("OPERATION", "STAGE", "ALGORITHM_STEP", "ALGORITHM_STEPS", "STEP_REGISTRY")
BOUNDED_TOKENS = ("confidence", "probability", "probabilities", "posterior")
DISTRIBUTION_TOKENS = ("probability", "probabilities", "posterior", "distribution")
SPECIFIC_DOC_HINTS = {
    "ore_n1": ("N1", "VISUAL"), "ore_n2": ("N2", "VISUAL"),
    "ore_n3": ("N3", "ORTHOGRAPH"), "ore_n4": ("N4",),
    "ore_n5": ("N5",), "ore_n6": ("N6",), "ore_n9": ("N9",),
}


def discover_ore_packages(repository_root: str | Path) -> list[str]:
    root = Path(repository_root)
    if not root.exists():
        return []
    return sorted(
        path.name for path in root.iterdir()
        if path.is_dir() and path.name.startswith("ore_")
        and (path / "__init__.py").is_file()
    )


def _type_hints(target: Any) -> dict[str, Any]:
    try:
        return get_type_hints(target)
    except Exception:
        return {}


def _is_dataclass_type(annotation: Any) -> bool:
    return inspect.isclass(annotation) and dataclasses.is_dataclass(annotation)


def _random_string(rng: random.Random, minimum: int = 3, maximum: int = 22) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    return "".join(rng.choice(alphabet) for _ in range(rng.randint(minimum, maximum)))


def _random_prompt(rng: random.Random) -> str:
    tokens = []
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    for _ in range(rng.randint(1, 18)):
        token = "".join(rng.choice(alphabet) for _ in range(rng.randint(2, 13)))
        if rng.random() < 0.24:
            token += str(rng.randint(0, 999))
        tokens.append(token)
    separators = (" ", " / ", " : ", " , ")
    return "".join(token + (rng.choice(separators) if i < len(tokens) - 1 else "")
                   for i, token in enumerate(tokens))


def _random_number(rng: random.Random, name: str, integer: bool = False) -> int | float:
    mode = rng.randrange(10)
    values = (0.0, 1.0, -1.0, 1e-6, 1e3,
              rng.uniform(-1.0, 1.0), rng.uniform(-10.0, 10.0),
              rng.uniform(-100.0, 100.0))
    value = values[min(mode, len(values) - 1)]
    if any(token in name.lower() for token in ("frequency", "capacity", "limit", "iteration", "size", "index", "position", "count")):
        value = rng.randint(0, 100) if integer else rng.uniform(0.1, 1000.0)
    if any(token in name.lower() for token in ("confidence", "probability", "prior", "threshold")):
        value = rng.random()
        if mode == 0:
            value = 0.0
        elif mode == 1:
            value = 1.0
    return int(round(value)) if integer else float(value)


def _random_array(rng_np: np.random.Generator, rng: random.Random,
                  field_name: str) -> np.ndarray:
    name = field_name.lower()
    if any(token in name for token in IMAGE_NAMES) or "image" in name or "pixel" in name:
        height = rng.choice((16, 24, 32, 48, 64, 80))
        width = rng.choice((16, 24, 32, 48, 64, 80))
        mode = rng.randrange(4)
        if mode == 0:
            return rng_np.integers(0, 256, size=(height, width), dtype=np.uint8)
        if mode == 1:
            return rng_np.random((height, width), dtype=np.float32)
        if mode == 2:
            return rng_np.normal(0.0, 1.0, size=(height, width)).astype(np.float32)
        return rng_np.uniform(-1.0, 1.0, size=(height, width)).astype(np.float32)
    length = rng.choice((4, 8, 16, 32, 64, 96, 128))
    mode = rng.randrange(4)
    if mode == 0:
        return rng_np.random(length, dtype=np.float32)
    if mode == 1:
        return rng_np.normal(0.0, 1.0, size=length).astype(np.float32)
    if mode == 2:
        return rng_np.uniform(-3.0, 3.0, size=length).astype(np.float32)
    return rng_np.integers(-3, 4, size=length, dtype=np.int32)


def _random_lexicon(rng: random.Random, field_name: str) -> list[dict[str, Any]]:
    rows = []
    for _ in range(rng.randint(2, 12)):
        orthography = _random_string(rng, 2, 11)
        if field_name == "morphemes":
            rows.append({
                "morpheme_id": f"random-{rng.getrandbits(48):012x}",
                "form": orthography[:rng.randint(1, len(orthography))],
                "position": rng.choice(("prefix", "suffix", "infix")),
                "semantic_features": {
                    _random_string(rng, 3, 10).upper(): rng.random()
                    for _ in range(rng.randint(1, 4))
                },
                "frequency": rng.randint(1, 10000),
                "productivity": rng.random(),
            })
        else:
            rows.append({
                "entry_id": f"random-{rng.getrandbits(48):012x}",
                "orthography": orthography,
                "frequency": rng.randint(1, 100000),
                "lexical_category": rng.choice(("noun", "verb", "adjective", "other")),
                "language": rng.choice(("en", "ar", "fr", "de", "und")),
                "meanings": [{
                    "meaning_id": f"random-{rng.getrandbits(48):012x}",
                    "semantic_features": {
                        _random_string(rng, 3, 10).upper(): rng.random()
                        for _ in range(rng.randint(1, 5))
                    },
                } for _ in range(rng.randint(1, 3))],
            })
    return rows


def _valid_random_image(rng_np: np.random.Generator) -> np.ndarray:
    """Random normalized grayscale image suitable for the N1 input contract."""
    height = int(rng_np.choice((32, 40, 48, 64)))
    width = int(rng_np.choice((32, 40, 48, 64)))
    return rng_np.random((height, width), dtype=np.float32)


def _make_random_n3_processor(rng: random.Random, rng_np: np.random.Generator):
    """Fit an ephemeral N3 model using fresh random feature samples."""
    from ore_n3 import LabeledN2Patch, N3Processor
    labels = (_random_string(rng, 2, 6), _random_string(rng, 2, 6))
    references = []
    for label in labels:
        for _ in range(2):
            features = rng_np.normal(0.0, 1.0, size=4).astype(np.float32)
            references.append(LabeledN2Patch(label, features=features))
    engine = N3Processor()
    engine.fit(references, grapheme_sequences=[
        (labels[0], labels[1]), (labels[1], labels[0]),
    ])
    return engine


def _make_random_n3_result(rng: random.Random, rng_np: np.random.Generator):
    """Generate a real N3 result from a randomly fitted, temporary N3 model."""
    from ore_n3 import N3Patch, N3Sequence
    engine = _make_random_n3_processor(rng, rng_np)
    patches = [
        N3Patch(
            f"rand-patch-{index}-{rng.getrandbits(24):06x}",
            features=rng_np.normal(0.0, 1.0, size=4).astype(np.float32),
            segmentation_confidence=float(rng.random()),
        )
        for index in range(rng.randint(2, 6))
    ]
    return engine, engine.process(N3Sequence(
        f"rand-seq-{rng.getrandbits(32):08x}", patches
    ))


def _make_random_n4_result(rng: random.Random, rng_np: np.random.Generator):
    from ore_n4 import N4Input, N4Processor
    _, n3_result = _make_random_n3_result(rng, rng_np)
    return N4Processor().process(N4Input(n3_result))


def _make_random_n5_model(rng: random.Random, rng_np: np.random.Generator):
    from ore_n5 import N5Processor
    engine = _make_random_n3_processor(rng, rng_np)
    return N5Processor.initialize(engine)


def _random_visual_upstream(rng_np: np.random.Generator):
    from ore_n1 import N1Processor
    return N1Processor().process(_valid_random_image(rng_np))


def _random_value(rng: random.Random, rng_np: np.random.Generator,
                  name: str, annotation: Any = Any, depth: int = 0,
                  package_name: str = "ore_unknown") -> Any:
    """Generate fresh random values while following annotated Python contracts."""
    if depth > 6:
        return _random_string(rng)
    lower = name.lower()
    origin, args = get_origin(annotation), get_args(annotation)
    if package_name in {"ore_n3", "ore_n5", "ore_n6"} and lower in {"features", "feature_vector", "embedding", "vector"}:
        return rng_np.normal(0.0, 1.0, size=4).astype(np.float32)
    if lower == "n1_result" and package_name == "ore_n2":
        return _random_visual_upstream(rng_np)
    if lower == "n3_result" and package_name in {"ore_n4", "ore_n5"}:
        return _make_random_n3_result(rng, rng_np)[1]
    if lower == "n4_result" and package_name in {"ore_n5", "ore_n6"}:
        return _make_random_n4_result(rng, rng_np)
    if lower in {"n5_model", "parent_model"} and package_name in {"ore_n5", "ore_n6"}:
        return _make_random_n5_model(rng, rng_np)
    if lower == "feature_sha256":
        return _UNSET
    if lower in {"verified_at", "created_at", "captured_at", "timestamp"}:
        return datetime.now(timezone.utc).isoformat()
    if lower in PREVIOUS_NAMES:
        return _UNSET
    if lower in {"config", "settings", "options"}:
        return {}
    if lower in IMAGE_NAMES or "image" in lower or "pixel" in lower:
        return _random_array(rng_np, rng, lower)
    if lower in {"lexicon", "concepts", "morphemes", "g2p_rules"}:
        return _random_lexicon(rng, lower)
    if lower in {"features", "feature_vector", "embedding", "vector", "weights", "values"}:
        return _random_array(rng_np, rng, lower)
    if lower in {"context", "context_features", "semantic_features", "features_by_name"}:
        return {_random_string(rng, 3, 10).upper(): rng.random()
                for _ in range(rng.randint(0, 8))}
    if lower in {"payload", "data", "record", "bundle"}:
        return {
            "id": f"random-{rng.getrandbits(48):012x}",
            "payload": _random_string(rng),
            "attributes": {_random_string(rng, 3, 10): _random_number(rng, lower)
                           for _ in range(rng.randint(0, 5))},
        }
    if lower in TEXT_NAMES or lower in {"input", "value", "label", "name"}:
        return _random_prompt(rng)
    if lower.endswith("_id") or lower in {"id", "identifier", "sequence_id"}:
        return f"random-{rng.getrandbits(48):012x}"
    if lower in {"language", "language_code"}:
        return rng.choice(("en", "ar", "fr", "de", "und", _random_string(rng, 2, 5)))
    if lower == "request_data":
        return {
            "sequence_id": f"random-{rng.getrandbits(48):012x}",
            "text": _random_prompt(rng),
        }

    if annotation is inspect.Parameter.empty or annotation is Any or isinstance(annotation, str):
        if any(token in lower for token in ("count", "capacity", "limit", "iteration", "size", "index")):
            return rng.randint(0, 100)
        if any(token in lower for token in ("rate", "score", "weight", "threshold", "confidence", "probability")):
            return _random_number(rng, lower)
        return _random_string(rng)
    if origin is Literal:
        return rng.choice(args) if args else None
    if origin in (types.UnionType, getattr(__import__("typing"), "Union")):
        options = [item for item in args if item is not type(None)]
        if type(None) in args and rng.random() < 0.15:
            return None
        return _random_value(rng, rng_np, name, rng.choice(options), depth + 1, package_name) if options else None
    if inspect.isclass(annotation) and issubclass(annotation, enum.Enum):
        members = list(annotation)
        return rng.choice(members) if members else _UNSET
    if _is_dataclass_type(annotation):
        return _random_dataclass(rng, rng_np, annotation, package_name, depth + 1)
    if annotation is np.ndarray:
        return _random_array(rng_np, rng, lower)
    if annotation is str:
        return _random_prompt(rng) if lower in TEXT_NAMES else _random_string(rng)
    if annotation is bool:
        return bool(rng.getrandbits(1))
    if annotation is int:
        return _random_number(rng, lower, integer=True)
    if annotation is float:
        return _random_number(rng, lower)
    if annotation is bytes:
        return rng.randbytes(rng.randint(0, 32))
    if annotation is Path:
        return Path(".")
    if origin in (dict, Mapping):
        value_type = args[1] if len(args) > 1 else Any
        return {
            _random_string(rng, 3, 10): _random_value(
                rng, rng_np, name, value_type, depth + 1, package_name
            )
            for _ in range(rng.randint(0, 7))
        }
    if origin in (list, tuple, set, Sequence, Iterable):
        item_type = args[0] if args else Any
        if package_name == "ore_n3" and getattr(item_type, "__name__", "") == "LabeledN2Patch":
            labels = (_random_string(rng, 2, 5), _random_string(rng, 2, 5))
            return [
                {"grapheme": labels[index % 2],
                 "features": rng_np.normal(0.0, 1.0, size=4).astype(np.float32)}
                for index in range(4)
            ]
        minimum_count = 1 if lower in {"patches", "experiences", "lexicon_entries", "reference_patches"} else 0
        values = [
            _random_value(rng, rng_np, name.rstrip("s") or "item",
                          item_type, depth + 1, package_name)
            for _ in range(rng.randint(minimum_count, 8))
        ]
        return tuple(values) if origin is tuple else values
    if annotation is dict:
        return {_random_string(rng): _random_number(rng, lower)
                for _ in range(rng.randint(0, 6))}
    if annotation in (list, tuple, set):
        return []
    return _UNSET


def _random_dataclass(rng: random.Random, rng_np: np.random.Generator,
                      cls: type, package_name: str, depth: int = 0) -> dict[str, Any]:
    if cls.__name__ == "LabeledN2Patch":
        return {
            "grapheme": _random_string(rng, 2, 5),
            "features": rng_np.normal(0.0, 1.0, size=4).astype(np.float32),
        }
    if cls.__name__ == "N4Input":
        from ore_n4 import N4Processor
        _, n3_result = _make_random_n3_result(rng, rng_np)
        patch_count = int(np.asarray(n3_result.posterior_beliefs).shape[0])
        payload = {"n3_result": n3_result}
        if rng.random() < 0.35:
            payload["bottom_up_salience"] = rng_np.random(patch_count, dtype=np.float32)
        if rng.random() < 0.35:
            payload["task_relevance"] = rng_np.random(patch_count, dtype=np.float32)
        if rng.random() < 0.25:
            payload["goal_class_weights"] = rng_np.random(
                len(n3_result.class_names), dtype=np.float32
            )
        if rng.random() < 0.25:
            payload["segmentation_confidence"] = rng_np.random(patch_count, dtype=np.float32)
        return payload
    if cls.__name__ == "N5LearningRequest":
        from ore_n4 import N4Input, N4Processor
        from ore_n5 import N5Experience, N5Processor
        engine, n3_result = _make_random_n3_result(rng, rng_np)
        parent_model = N5Processor.initialize(engine)
        experiences = []
        for index in range(rng.randint(1, 8)):
            experiences.append(N5Experience(
                experience_id=f"rand-exp-{rng.getrandbits(32):08x}",
                patch_id=f"rand-patch-{rng.getrandbits(24):06x}",
                sequence_id=f"rand-seq-{rng.getrandbits(32):08x}",
                grapheme=rng.choice(parent_model.class_names),
                features=rng_np.normal(0.0, 1.0, size=parent_model.feature_dimension).astype(np.float32),
                verified_by=f"synthetic-reviewer-{rng.getrandbits(16):04x}",
                source_reference=f"synthetic-source://{rng.getrandbits(32):08x}",
                verified_at=datetime.now(timezone.utc).isoformat(),
                confidence=float(rng.random()),
                sequence_position=index,
            ))
        payload = {
            "parent_model": parent_model,
            "experiences": experiences,
            "independently_verified": bool(rng.getrandbits(1)),
            "user_authorized": bool(rng.getrandbits(1)),
        }
        if rng.random() < 0.5:
            payload["n4_result"] = N4Processor().process(N4Input(n3_result))
        return payload
    if cls.__name__ == "N6Input":
        from ore_n4 import N4Input, N4Processor
        from ore_n5 import N5Processor
        from ore_n6 import N6LexiconEntry
        engine, n3_result = _make_random_n3_result(rng, rng_np)
        n5_model = N5Processor.initialize(engine)
        lexicon_entries = [
            N6LexiconEntry(
                entry_id=f"rand-word-{rng.getrandbits(32):08x}",
                graphemes=tuple(rng.choice(n5_model.class_names)
                                for _ in range(rng.randint(1, 8))),
                frequency=float(rng.uniform(0.1, 1000.0)),
                verified_by=f"synthetic-reviewer-{rng.getrandbits(16):04x}",
                source_reference=f"synthetic-lexicon://{rng.getrandbits(32):08x}",
                language=rng.choice(("en", "ar", "fr", "und")),
            )
            for _ in range(rng.randint(1, 8))
        ]
        return {
            "n4_result": N4Processor().process(N4Input(n3_result)),
            "n5_model": n5_model,
            "lexicon_entries": lexicon_entries,
        }
    if cls.__name__ == "N3Sequence":
        return {
            "sequence_id": f"random-{rng.getrandbits(48):012x}",
            "patches": [
                {
                    "patch_id": f"rand-patch-{rng.getrandbits(24):06x}",
                    "features": rng_np.normal(0.0, 1.0, size=4).astype(np.float32),
                    "segmentation_confidence": float(rng.random()),
                }
                for _ in range(rng.randint(1, 6))
            ],
        }
    if cls.__name__ == "N3Patch":
        return {
            "patch_id": f"rand-patch-{rng.getrandbits(24):06x}",
            "features": rng_np.normal(0.0, 1.0, size=4).astype(np.float32),
            "segmentation_confidence": float(rng.random()),
        }
    if cls.__name__ == "N9Patch":
        source_kind = rng.choice(("image", "features", "grapheme_probabilities"))
        payload = {"patch_id": f"rand-patch-{rng.getrandbits(32):08x}"}
        if source_kind == "image":
            payload["image"] = _valid_random_image(rng_np)
        elif source_kind == "features":
            payload["features"] = rng_np.normal(0.0, 1.0, size=rng.choice((4, 8, 16, 32))).astype(np.float32)
        else:
            payload["grapheme_probabilities"] = {
                _random_string(rng, 1, 5): float(rng.random())
                for _ in range(rng.randint(2, 7))
            }
        payload["segmentation_confidence"] = float(rng.random())
        return payload
    hints = _type_hints(cls)
    payload = {}
    for field in dataclasses.fields(cls):
        if not field.init:
            continue
        required = field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING
        if field.name in {"verified_sequences", "feature_sha256"}:
            continue
        if cls.__name__ == "N3Patch" and field.name == "features":
            include_optional = True
        else:
            include_optional = required or rng.random() < OPTIONAL_PARAMETER_PROBABILITY
        if not include_optional:
            continue
        value = _random_value(rng, rng_np, field.name,
                              hints.get(field.name, field.type), depth + 1, package_name)
        if value is not _UNSET:
            payload[field.name] = value
    return payload


def _prepare_upstream(package_name: str, info: CallableInfo,
                      rng: random.Random, rng_np: np.random.Generator) -> Any:
    """Create random, ephemeral upstream objects for known dependent pipelines."""
    if package_name == "ore_n2" and info.name == "process":
        return _random_visual_upstream(rng_np)
    if package_name == "ore_n3" and info.class_name == "N3Processor" and info.name in {"process", "recognize"}:
        return _make_random_n3_processor(rng, rng_np)
    return None


def _discover_modules(package_name: str) -> tuple[list[Any], list[str]]:
    package = importlib.import_module(package_name)
    modules, errors = [package], []
    if hasattr(package, "__path__"):
        for item in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
            try:
                modules.append(importlib.import_module(item.name))
            except Exception as exc:
                errors.append(f"{item.name}: {type(exc).__name__}: {exc}")
    return modules, errors


def _discover_callables(package_name: str, modules: list[Any]) -> list[CallableInfo]:
    found = {}
    for module in modules:
        for name, function in inspect.getmembers(module, inspect.isfunction):
            if (function.__module__ != module.__name__ or name.startswith("_")
                    or name.lower() not in SAFE_METHODS or name.lower() in UNSAFE_METHODS):
                continue
            signature = str(inspect.signature(function))
            found[(module.__name__, name, None)] = CallableInfo(
                f"{module.__name__}.{name}{signature}", module.__name__, name,
                signature, inspect.getdoc(function) or "No docstring provided.",
            )
        for class_name, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ != module.__name__ or class_name.startswith("_"):
                continue
            for name, method in inspect.getmembers(cls, inspect.isfunction):
                if name.startswith("_") or name.lower() not in SAFE_METHODS or name.lower() in UNSAFE_METHODS:
                    continue
                signature = str(inspect.signature(method))
                found[(module.__name__, name, class_name)] = CallableInfo(
                    f"{module.__name__}.{class_name}.{name}{signature}",
                    module.__name__, name, signature,
                    inspect.getdoc(method) or inspect.getdoc(cls) or "No docstring provided.",
                    class_name,
                )
    priority = {
        "process": 0, "infer": 1, "predict": 2, "forward": 3,
        "transform": 4, "recognize": 5, "parse": 6, "compose": 7,
        "reason": 8, "plan": 9, "retrieve": 10, "query": 11,
        "fit": 12, "train": 13, "learn": 14,
    }
    return sorted(found.values(), key=lambda item: (
        priority.get(item.name, 20),
        0 if item.class_name and any(word in item.class_name.lower()
                                     for word in ("processor", "engine", "model", "pipeline", "adapter")) else 1,
        item.label,
    ))


def _constructor_arguments(package_name: str, info: CallableInfo,
                           rng: random.Random, rng_np: np.random.Generator) -> dict[str, Any]:
    if not info.class_name:
        return {}
    cls = getattr(importlib.import_module(info.module), info.class_name)
    try:
        signature = inspect.signature(cls)
    except (TypeError, ValueError):
        return {}
    hints = _type_hints(cls.__init__)
    supplied = {}
    for name, parameter in signature.parameters.items():
        if name in {"self", "cls"} or parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD
        ):
            continue
        if parameter.default is not inspect.Parameter.empty and rng.random() >= OPTIONAL_PARAMETER_PROBABILITY:
            continue
        value = _random_value(rng, rng_np, name,
                              hints.get(name, parameter.annotation), 0, package_name)
        if value is not _UNSET:
            supplied[name] = value
    return supplied


def _call_arguments(package_name: str, info: CallableInfo,
                    rng: random.Random, rng_np: np.random.Generator) -> dict[str, Any]:
    module = importlib.import_module(info.module)
    target = getattr(getattr(module, info.class_name), info.name) if info.class_name else getattr(module, info.name)
    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):
        return {}
    hints = _type_hints(target)
    supplied = {}
    for name, parameter in signature.parameters.items():
        if name in {"self", "cls", "on_operation", "callback", "progress_callback"}:
            continue
        if name in PREVIOUS_NAMES:
            continue
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        required = parameter.default is inspect.Parameter.empty
        if not required and rng.random() >= OPTIONAL_PARAMETER_PROBABILITY:
            continue
        value = _random_value(rng, rng_np, name,
                              hints.get(name, parameter.annotation), 0, package_name)
        if value is not _UNSET:
            supplied[name] = value
    return supplied


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: getattr(value, field.name) for field in dataclasses.fields(value)}
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return dict(vars(value))
    return {}


def _registry_entries(value: Any) -> list[Any]:
    if isinstance(value, Mapping):
        return list(value.values())
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return []


def _declared_operations(modules: list[Any]) -> list[dict[str, Any]]:
    records = {}
    for module in modules:
        for attr_name, value in vars(module).items():
            if not any(token in attr_name.upper() for token in REGISTRY_TOKENS):
                continue
            for index, raw in enumerate(_registry_entries(value)):
                if isinstance(raw, str):
                    row = {"step": index + 1, "operation_name": raw}
                else:
                    row = _as_mapping(raw)
                if not row:
                    continue
                identifier = next((row.get(key) for key in (
                    "operation_id", "step_id", "stage_id", "id", "number", "step",
                ) if row.get(key) is not None), None)
                name = next((row.get(key) for key in (
                    "operation_name", "step_name", "stage_name", "name", "operation", "title",
                ) if row.get(key) is not None), None)
                mathematical = next((row.get(key) for key in (
                    "mathematical_form", "equation", "formula", "mathematics",
                ) if row.get(key)), None)
                bio = next((row.get(key) for key in (
                    "biological_reference", "biological_basis", "neural_reference",
                ) if row.get(key)), None)
                key = f"{module.__name__}.{attr_name}.{identifier or name or index}"
                records[key] = {
                    "registry": attr_name, "module": module.__name__,
                    "identifier": str(identifier) if identifier is not None else None,
                    "name": str(name) if name is not None else None,
                    "mathematical_form": str(mathematical)[:1000] if mathematical else None,
                    "biological_reference": str(bio)[:1000] if bio else None,
                }
    return list(records.values())


def _theory_documents(root: Path, package_name: str) -> list[dict[str, Any]]:
    docs_root = root / "docs"
    package_tokens = {
        token.upper() for token in package_name.replace("ore_", "").split("_")
        if token and token not in {"n1", "n2", "n3", "n4", "n5", "n6", "n9"}
    }
    hints = SPECIFIC_DOC_HINTS.get(package_name, ())
    matches = []
    if docs_root.exists():
        for path in docs_root.glob("*.md"):
            stem = path.stem.upper()
            score = sum(2 for token in package_tokens if token in stem)
            score += sum(3 for hint in hints if hint in stem)
            if score:
                matches.append((score, path))
    matches.sort(key=lambda item: (-item[0], item[1].name))
    selected = [path for _, path in matches[:5]]
    local_readme = root / package_name / "README.md"
    if local_readme.is_file():
        selected.insert(0, local_readme)
    crosswalk = docs_root / "ORE_SOURCE_TO_CODE_IMPLEMENTATION_CROSSWALK.md"
    if crosswalk.is_file() and crosswalk not in selected:
        selected.append(crosswalk)
    output = []
    for path in selected[:6]:
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            headings = [line.strip("# ").strip() for line in content.splitlines()
                        if line.startswith("#") and len(line.strip("# ").strip()) < 180][:40]
            output.append({
                "path": str(path.relative_to(root)).replace("\\", "/"),
                "headings": headings,
                "excerpt": content[:4500],
            })
        except OSError:
            pass
    return output


def _source_inventory(root: Path, package_name: str) -> dict[str, Any]:
    folder = root / package_name
    files = sorted(folder.rglob("*.py")) if folder.exists() else []
    functions, classes, errors = [], [], []
    for path in files:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            relative = str(path.relative_to(root)).replace("\\", "/")
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
                    functions.append({"file": relative, "name": node.name, "line": node.lineno})
                elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                    classes.append({"file": relative, "name": node.name, "line": node.lineno})
        except Exception as exc:
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
    return {
        "python_file_count": len(files), "public_function_count": len(functions),
        "public_class_count": len(classes), "public_functions": functions[:500],
        "public_classes": classes[:250], "parse_errors": errors,
    }


def _numeric_arrays(value: Any, path: str = "output", depth: int = 0):
    if depth > 8:
        return
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.number):
            yield path, value
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            yield from _numeric_arrays(getattr(value, field.name), f"{path}.{field.name}", depth + 1)
    elif isinstance(value, Mapping):
        for key, child in value.items():
            yield from _numeric_arrays(child, f"{path}.{key}", depth + 1)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value[:100]):
            yield from _numeric_arrays(child, f"{path}[{index}]", depth + 1)


def _invariant_checks(output: Any) -> list[dict[str, Any]]:
    """Check output-independent mathematical constraints; never use answer keys."""
    checks = []
    for path, array in _numeric_arrays(output):
        if array.size == 0:
            checks.append({"kind": "array_non_empty", "path": path, "status": "warning"})
            continue
        finite = bool(np.isfinite(array).all())
        checks.append({
            "kind": "numeric_finiteness", "path": path,
            "status": "pass" if finite else "violation",
            "shape": list(array.shape), "dtype": str(array.dtype),
        })
        if not finite:
            continue
        lowered = path.lower()
        if any(token in lowered for token in BOUNDED_TOKENS):
            minimum, maximum = float(array.min()), float(array.max())
            checks.append({
                "kind": "bounded_probability_or_confidence_candidate",
                "path": path,
                "status": "pass" if minimum >= -1e-6 and maximum <= 1.0 + 1e-6 else "candidate_violation",
                "minimum": minimum, "maximum": maximum,
                "note": "Name-based hypothesis only; verify that this field is theoretically bounded to [0, 1].",
            })
        if any(token in lowered for token in DISTRIBUTION_TOKENS) and array.ndim >= 1:
            if np.issubdtype(array.dtype, np.floating) and array.shape[-1] > 1:
                if np.all(array >= -1e-6) and np.all(array <= 1.0 + 1e-6):
                    deviation = float(np.max(np.abs(array.sum(axis=-1) - 1.0)))
                    checks.append({
                        "kind": "probability_distribution_normalization_candidate",
                        "path": path,
                        "status": "pass" if deviation <= 1e-3 else "candidate_violation",
                        "max_sum_deviation": deviation,
                        "note": "Applied only to non-negative probability-like arrays; verify theoretical semantics.",
                    })
    return checks


def _trace_ids(observation: Mapping[str, Any]) -> set[str]:
    result = set()
    for row in observation.get("trace_rows", []):
        for key in ("operation_id", "step_id", "stage_id", "id", "step"):
            if row.get(key) is not None:
                result.add(str(row[key]))
    return result


def _registry_coverage(registry: list[dict[str, Any]],
                       observed_ids: set[str]) -> dict[str, Any]:
    declared = {row["identifier"]: row for row in registry if row.get("identifier") is not None}
    matched = set(declared).intersection(observed_ids)
    return {
        "declared_operation_count": len(registry),
        "declared_with_identifiers": len(declared),
        "observed_registered_operation_count": len(matched),
        "coverage_ratio": len(matched) / len(declared) if declared else None,
        "coverage_status": "measured_by_exact_identifier" if declared
                           else "registry_missing_or_without_stable_ids",
        "matched": [declared[key] for key in sorted(matched)],
        "not_observed": [row for key, row in declared.items() if key not in observed_ids],
        "trace_ids_without_registry_match": sorted(observed_ids - set(declared)),
    }


def _classify_failure(error: str | None) -> str | None:
    if not error:
        return None
    exception_name = error.split(":", 1)[0].strip()
    if exception_name == "PermissionError":
        return "authorization_gate_rejection"
    if exception_name in {"ValueError", "TypeError", "KeyError", "IndexError"}:
        return "input_contract_rejection_candidate"
    return "unexpected_runtime_exception"


def _bounded_value(value: Any, depth: int = 0) -> Any:
    if depth > 8:
        return "..."
    if isinstance(value, np.ndarray):
        arr = np.asarray(value)
        numeric = np.issubdtype(arr.dtype, np.number)
        finite = np.isfinite(arr) if numeric and arr.size else None
        return {
            "__array__": True, "shape": list(arr.shape), "dtype": str(arr.dtype),
            "size": int(arr.size),
            "finite": bool(finite.all()) if finite is not None else None,
            "minimum": float(arr.min()) if arr.size and numeric and finite.all() else None,
            "maximum": float(arr.max()) if arr.size and numeric and finite.all() else None,
            "mean": float(arr.mean()) if arr.size and numeric and finite.all() else None,
            "sample": arr.reshape(-1)[:24].tolist() if arr.size else [],
        }
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: _bounded_value(getattr(value, field.name), depth + 1)
                for field in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _bounded_value(item, depth + 1)
                for key, item in list(value.items())[:100]}
    if isinstance(value, (list, tuple, set)):
        items = [_bounded_value(item, depth + 1) for item in list(value)[:100]]
        if len(value) > 100:
            items.append(f"... {len(value) - 100} omitted")
        return items
    safe = json_safe(value)
    if isinstance(safe, (str, int, float, bool)) or safe is None:
        return safe
    return repr(safe)[:700]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    temp.replace(path)


def _package_markdown(report: Mapping[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        f"# Randomized theory-oriented evaluation: {report['package']}", "",
        f"- Random seed: {report['random_seed']}",
        f"- Package import: {report['import_status']}",
        f"- Discovered callables: {len(report['discovered_callables'])}",
        f"- Callables exercised: {summary['callables_exercised']}",
        f"- Random trials: {summary['trials_executed']}",
        f"- Unexpected execution errors: {summary['execution_errors']}",
        f"- Input/authorization rejections: {summary.get('input_rejections', 0)}",
        f"- Candidate invariant flags: {summary['candidate_invariant_violations']}", "",
        "## Evaluation boundary", "",
        "No fixed prompts, answer keys, reference labels, or pre-authored expected outputs are used. Inputs "
        "are generated randomly from callable signatures and typed data contracts. The automated checks "
        "are mathematical/structural invariants and registry-to-trace coverage. They can expose code defects, "
        "but they cannot prove biological validity or complete theoretical equivalence.", "",
        "Learning/update entry points, when discovered, operate on newly created in-memory models and random synthetic records only. Their randomized authorization flags exercise the code's permission gate; they do not represent genuine human verification or consent, and no persistent model write is performed.", "",
        "## Theory specification sources", "",
    ]
    if report["theory_documents"]:
        for doc in report["theory_documents"]:
            lines.append(f"- {doc['path']}: {', '.join(doc['headings'][:12])}")
    else:
        lines.append("- No package-matched Markdown specification was found.")
    lines.extend(["", "## Declared operation registry", ""])
    if report["declared_operations"]:
        lines.append(f"Declared entries: {len(report['declared_operations'])}")
        for row in report["declared_operations"][:80]:
            lines.append(f"- {row.get('identifier') or 'no-id'} {row.get('name') or ''} [{row['registry']}]")
            if row.get("mathematical_form"):
                lines.append(f"  - Mathematical form: {row['mathematical_form']}")
            if row.get("biological_reference"):
                lines.append(f"  - Biological reference: {row['biological_reference']}")
    else:
        lines.append("No machine-readable operation registry was discovered.")
    package_coverage = report.get("package_registry_coverage", {})
    lines.extend(["", "## Package-level operation coverage", ""])
    lines.append(
        f"Coverage status: {package_coverage.get('coverage_status', 'not measured')}; "
        f"{package_coverage.get('observed_registered_operation_count', 0)} of "
        f"{package_coverage.get('declared_with_identifiers', 0)} identified operations were observed."
    )
    for missing in package_coverage.get("not_observed", [])[:80]:
        lines.append(f"- Not observed: {missing.get('identifier')} {missing.get('name') or ''}")
    lines.extend(["", "## Callable-level randomized results", "",
                  "| Callable | Trials | Successes | Runtime errors | Input/guard rejections | Candidate invariant flags |",
                  "|---|---:|---:|---:|---:|---:|"])
    for item in report["callable_results"]:
        part = item["summary"]
        lines.append("| " + " | ".join(str(value) for value in (
            item["callable"], part["trials"], part["successes"],
            part["errors"], part["rejections"], part["candidate_invariant_violations"],
        )) + " |")
    lines.extend(["", "## Registry-to-trace coverage", ""])
    for item in report["callable_results"]:
        coverage = item["registry_coverage"]
        lines.append(
            f"- {item['callable']}: {coverage['coverage_status']}; "
            f"{coverage['observed_registered_operation_count']} / "
            f"{coverage['declared_with_identifiers']} operation identifiers observed."
        )
        for missing in coverage["not_observed"][:30]:
            lines.append(f"  - Not observed in this callable: {missing.get('identifier')} {missing.get('name') or ''}")
    lines.extend(["", "## Trial details", ""])
    for item in report["callable_results"]:
        lines.extend([f"### {item['callable']}", "",
                      f"Signature: {item['signature']}", "",
                      f"Description: {item['doc'][:500]}", ""])
        for trial in item["trials"][:25]:
            lines.extend([
                f"#### Trial {trial['trial_number']} — {trial['status']}",
                f"- Random input: {json.dumps(trial['generated_input'], ensure_ascii=False, default=str)[:1600]}",
                f"- Output type: {trial.get('output_type')}",
                f"- Duration: {trial.get('elapsed_ms')} ms",
                f"- Failure classification: {trial.get('failure_classification') or 'none'}",
            ])
            if trial.get("errors"):
                lines.append(f"- Error: {trial['errors']}")
            flags = [check for check in trial["invariant_checks"]
                     if check["status"] in {"violation", "candidate_violation"}]
            for flag in flags[:12]:
                lines.append(f"- Candidate invariant observation: {flag}")
            lines.append("")
        if len(item["trials"]) > 25:
            lines.append(f"{len(item['trials']) - 25} more trial records are included in the JSON report.")
    lines.extend([
        "", "## Questions for theory-gap review", "",
        "1. Do the discovered theory documents describe every declared operation and mathematical variable?",
        "2. Do real execution traces enter the algorithmic stages declared by the package?",
        "3. Which registered operations are absent from observed traces and need separate invocation or integration tests?",
        "4. Are candidate invariant flags genuine violations once the variable's theoretical meaning is checked?",
        "5. Which theoretical claims cannot be tested without domain-specific measurements or empirical data?",
        "",
    ])
    return "\n".join(lines)


def _write_package_report(report: dict[str, Any], report_dir: Path) -> dict[str, str]:
    folder = report_dir / "packages"
    json_path = folder / f"{report['package']}.json"
    markdown_path = folder / f"{report['package']}.md"
    _write_json(json_path, report)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(_package_markdown(report), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(markdown_path)}


def _prepare_package(root: Path, package_name: str) -> dict[str, Any]:
    item = {
        "package": package_name, "import_status": "unknown",
        "modules": [], "callables": [], "discovery_errors": [],
        "declared_operations": [], "theory_documents": [], "source_inventory": {},
    }
    try:
        modules, errors = _discover_modules(package_name)
        item["import_status"] = "success"
        item["modules"] = modules
        item["discovery_errors"] = errors
        item["callables"] = _discover_callables(package_name, modules)
        item["declared_operations"] = _declared_operations(modules)
        item["theory_documents"] = _theory_documents(root, package_name)
        item["source_inventory"] = _source_inventory(root, package_name)
    except Exception as exc:
        item["import_status"] = "error"
        item["discovery_errors"].append(f"{type(exc).__name__}: {exc}")
        item["traceback"] = traceback.format_exc()
    return item


def _notify(progress_callback, completed: int, total: int, **details: Any) -> None:
    if progress_callback is None:
        return
    payload = {
        "completed": completed, "total": total,
        "fraction": completed / total if total else (1.0 if str(details.get("phase", "")).startswith("Reports and ZIP") else 0.0),
        **details,
    }
    try:
        progress_callback(payload)
    except Exception:
        pass


def run_auto_test_suite(
    repository_root: str | Path,
    progress_callback=None,
    trials_per_callable: int = 25,
    max_callables_per_package: int = 0,
    random_seed: int | None = None,
) -> dict[str, Any]:
    """Execute randomized trials for all eligible callables in every ore_* package.

    A zero callable cap means all discovered callables. Each trial gets fresh
    random input, fresh constructor state, and a new processor instance.
    """
    root = Path(repository_root).resolve()
    packages = discover_ore_packages(root)
    run_id = uuid.uuid4().hex
    seed = int(random_seed if random_seed is not None else secrets.randbits(64))
    rng_py = random.Random(seed)
    rng_np = np.random.default_rng(seed)
    started_at = datetime.now(timezone.utc).isoformat()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_dir = root / "ore_reports" / "randomized_theory_tests" / f"{stamp}_{run_id[:8]}"
    report_dir.mkdir(parents=True, exist_ok=True)

    inventories = []
    for index, package_name in enumerate(packages, start=1):
        _notify(progress_callback, 0, 0, package=package_name,
                package_index=index, package_count=len(packages),
                phase="Importing package and discovering algorithms")
        inventory = _prepare_package(root, package_name)
        candidates = inventory["callables"]
        if max_callables_per_package > 0:
            candidates = candidates[:max_callables_per_package]
        inventory["selected_callables"] = candidates
        inventories.append(inventory)

    total_trials = sum(len(item["selected_callables"]) * trials_per_callable
                       for item in inventories)
    completed = 0
    package_reports = []

    for package_index, inventory in enumerate(inventories, start=1):
        package_name = inventory["package"]
        candidates = inventory["selected_callables"]
        report = {
            "schema": "ore.workbench.randomized_theory_evaluation.v1",
            "run_id": run_id, "random_seed": seed,
            "started_at_utc": started_at, "package": package_name,
            "import_status": inventory["import_status"],
            "discovered_callables": [item.label for item in inventory.get("callables", [])],
            "selected_callables": [item.label for item in candidates],
            "discovery_errors": inventory["discovery_errors"],
            "theory_documents": inventory["theory_documents"],
            "declared_operations": inventory["declared_operations"],
            "source_inventory": inventory["source_inventory"],
            "callable_results": [],
        }
        if inventory["import_status"] != "success":
            report["summary"] = {
                "callables_exercised": 0, "trials_executed": 0,
                "successes": 0, "execution_errors": 0, "input_rejections": 0,
                "candidate_invariant_violations": 0, "status": "package_import_failed",
            }
            report["report_files"] = _write_package_report(report, report_dir)
            package_reports.append(report)
            continue

        for callable_index, info in enumerate(candidates, start=1):
            trials = []
            successes = errors = rejections = invariant_flags = 0
            observed_trace_ids: set[str] = set()
            for trial_index in range(1, trials_per_callable + 1):
                _notify(
                    progress_callback, completed, total_trials,
                    package=package_name, package_index=package_index,
                    package_count=len(packages), callable=info.label,
                    callable_index=callable_index, callable_count=len(candidates),
                    trial_index=trial_index, trial_count=trials_per_callable,
                    phase="Generating fresh random input and executing",
                )
                trial_id = f"{run_id[:8]}-{package_name}-{callable_index}-{trial_index}"
                prompt = _random_prompt(rng_py)
                supplied: dict[str, Any] = {}
                init_args: dict[str, Any] = {}
                try:
                    supplied = _call_arguments(package_name, info, rng_py, rng_np)
                    init_args = _constructor_arguments(package_name, info, rng_py, rng_np)
                    if init_args:
                        supplied["__init__"] = init_args
                    supplied.setdefault("sequence_id", trial_id)
                    previous_output = _prepare_upstream(package_name, info, rng_py, rng_np)
                    result = execute(
                        package_name, info, prompt, supplied,
                        previous_output=previous_output, runtime_kwargs=None, on_event=None,
                        trace_lines=False, log_directory=None,
                    )
                    obs = result.get("observation", {})
                    checks = _invariant_checks(result.get("output"))
                    observed_trace_ids.update(_trace_ids(obs))
                    failure_classification = _classify_failure(result.get("error"))
                    if result.get("status") == "success":
                        successes += 1
                    elif failure_classification in {
                        "authorization_gate_rejection", "input_contract_rejection_candidate"
                    }:
                        rejections += 1
                    else:
                        errors += 1
                    flags = sum(check["status"] in {"violation", "candidate_violation"}
                                for check in checks)
                    invariant_flags += flags
                    trials.append({
                        "trial_id": trial_id, "trial_number": trial_index,
                        "status": result.get("status"),
                        "random_prompt": prompt,
                        "generated_input": _bounded_value({
                            "constructor_arguments": init_args,
                            "arguments": result.get("arguments", supplied),
                        }),
                        "elapsed_ms": result.get("elapsed_ms"),
                        "output_type": obs.get("output_type"),
                        "output": _bounded_value(result.get("output")),
                        "observation": _bounded_value(obs),
                        "python_profile": {
                            "observed_call_count": result.get("profile", {}).get("observed_call_count", 0),
                            "function_summary": result.get("profile", {}).get("function_summary", [])[:35],
                        },
                        "invariant_checks": checks,
                        "failure_classification": failure_classification,
                        "errors": result.get("error"),
                        "traceback": result.get("traceback"),
                    })
                except Exception as exc:
                    errors += 1
                    trials.append({
                        "trial_id": trial_id, "trial_number": trial_index,
                        "status": "input_generation_or_dispatch_error",
                        "failure_classification": "test_harness_error",
                        "random_prompt": prompt,
                        "generated_input": _bounded_value({
                            "constructor_arguments": init_args, "arguments": supplied,
                        }),
                        "elapsed_ms": None, "output_type": None,
                        "output": None, "observation": {}, "python_profile": {},
                        "invariant_checks": [], "errors": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    })
                completed += 1
                _notify(
                    progress_callback, completed, total_trials,
                    package=package_name, package_index=package_index,
                    package_count=len(packages), callable=info.label,
                    callable_index=callable_index, callable_count=len(candidates),
                    trial_index=trial_index, trial_count=trials_per_callable,
                    phase="Trial recorded",
                )
            coverage = _registry_coverage(inventory["declared_operations"], observed_trace_ids)
            report["callable_results"].append({
                "callable": info.label, "module": info.module, "name": info.name,
                "class_name": info.class_name, "signature": info.signature,
                "doc": info.doc, "registry_coverage": coverage,
                "summary": {
                    "trials": len(trials), "successes": successes,
                    "errors": errors, "rejections": rejections,
                    "candidate_invariant_violations": invariant_flags,
                },
                "trials": trials,
            })

        tested = report["callable_results"]
        observed_package_trace_ids: set[str] = set()
        for callable_result in tested:
            item_coverage = callable_result["registry_coverage"]
            observed_package_trace_ids.update(
                str(row["identifier"]) for row in item_coverage.get("matched", [])
                if row.get("identifier") is not None
            )
            observed_package_trace_ids.update(
                item_coverage.get("trace_ids_without_registry_match", [])
            )
        report["package_registry_coverage"] = _registry_coverage(
            inventory["declared_operations"], observed_package_trace_ids
        )
        report["summary"] = {
            "callables_exercised": len(tested),
            "trials_executed": sum(item["summary"]["trials"] for item in tested),
            "successes": sum(item["summary"]["successes"] for item in tested),
            "execution_errors": sum(item["summary"]["errors"] for item in tested),
            "input_rejections": sum(item["summary"]["rejections"] for item in tested),
            "candidate_invariant_violations": sum(
                item["summary"]["candidate_invariant_violations"] for item in tested
            ),
            "status": (
                "completed_with_runtime_or_invariant_observations"
                if any(item["summary"]["errors"] or item["summary"]["candidate_invariant_violations"] for item in tested)
                else "completed_with_input_or_guard_rejections"
                if any(item["summary"]["rejections"] for item in tested)
                else "completed_without_generic_invariant_flags"
            ),
        }
        report["report_files"] = _write_package_report(report, report_dir)
        package_reports.append(report)

    summary = {
        "schema": "ore.workbench.randomized_theory_summary.v1",
        "run_id": run_id, "random_seed": seed,
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": "ORE-Neurocomputational",
        "package_count": len(packages),
        "packages_imported": sum(item["import_status"] == "success" for item in inventories),
        "packages_with_callable": sum(bool(item["selected_callables"]) for item in inventories),
        "packages_without_callable": sum(not item["selected_callables"] for item in inventories),
        "callables_exercised": sum(len(item["callable_results"]) for item in package_reports),
        "planned_trials": total_trials, "completed_trials": completed,
        "successes": sum(item["summary"].get("successes", 0) for item in package_reports),
        "execution_errors": sum(item["summary"].get("execution_errors", 0) for item in package_reports),
        "input_rejections": sum(item["summary"].get("input_rejections", 0) for item in package_reports),
        "candidate_invariant_violations": sum(
            item["summary"].get("candidate_invariant_violations", 0) for item in package_reports
        ),
        "trials_per_callable": trials_per_callable,
        "max_callables_per_package": max_callables_per_package,
        "all_discovered_callables_included": max_callables_per_package == 0,
        "oracle_free": True, "predefined_prompts_or_expected_outputs": False,
        "learning_test_scope": (
            "Any learning/update trial uses a fresh in-memory model and randomly generated synthetic "
            "records. Authorization flags in these trials test the code gate only; they do not represent "
            "real-world independent verification or consent. No persistent model write is performed."
        ),
    }
    summary_json = report_dir / "summary.json"
    summary_md = report_dir / "summary.md"
    package_index = [{
        "package": item["package"], "import_status": item["import_status"],
        "summary": item["summary"],
        "callables_exercised": [row["callable"] for row in item["callable_results"]],
        "report_files": item.get("report_files", {}),
    } for item in package_reports]
    _write_json(summary_json, {**summary, "package_results": package_index})
    lines = [
        "# ORE randomized theory-oriented evaluation", "",
        f"- Run ID: {run_id}", f"- Random seed: {seed}",
        f"- Packages: {summary['package_count']}",
        f"- Imported packages: {summary['packages_imported']}",
        f"- Callables exercised: {summary['callables_exercised']}",
        f"- Random trials planned/completed: {total_trials}/{completed}",
        f"- Successful executions: {summary['successes']}",
        f"- Unexpected execution errors: {summary['execution_errors']}",
        f"- Input/authorization rejections: {summary.get('input_rejections', 0)}",
        f"- Candidate invariant flags: {summary['candidate_invariant_violations']}", "",
        "Fresh random inputs were generated from callable signatures. There are no fixed prompts or expected answers.",
        "This is execution and mathematical-invariant evidence, not proof of cognitive or biological validity.", "",
        "| Package | Import | Callables | Trials | Runtime errors | Rejections | Candidate invariant flags |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in package_reports:
        part = item["summary"]
        lines.append("| " + " | ".join(str(value) for value in (
            item["package"], item["import_status"], part.get("callables_exercised", 0),
            part.get("trials_executed", 0), part.get("execution_errors", 0),
            part.get("input_rejections", 0), part.get("candidate_invariant_violations", 0),
        )) + " |")
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    archive_path = report_dir / f"ore-randomized-theory-reports-{run_id[:8]}.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(summary_json, arcname="summary.json")
        archive.write(summary_md, arcname="summary.md")
        for item in package_reports:
            for ext in ("json", "markdown"):
                file_path = item.get("report_files", {}).get(ext)
                if file_path and Path(file_path).is_file():
                    suffix = "json" if ext == "json" else "md"
                    archive.write(file_path, arcname=f"packages/{item['package']}.{suffix}")
    _notify(progress_callback, completed, total_trials, package="All packages",
            package_index=len(packages), package_count=len(packages),
            callable="", callable_index=0, callable_count=0,
            trial_index=0, trial_count=0, phase="Reports and ZIP archive saved")
    return {
        "run_id": run_id, "random_seed": seed,
        "started_at_utc": started_at, "report_directory": str(report_dir),
        "summary": summary, "summary_json": str(summary_json),
        "summary_markdown": str(summary_md), "archive_path": str(archive_path),
        "package_reports": [{
            "package": item["package"], "import_status": item["import_status"],
            "summary": item["summary"],
            "json_path": item.get("report_files", {}).get("json"),
            "markdown_path": item.get("report_files", {}).get("markdown"),
        } for item in package_reports],
    }


def save_manual_run_report(repository_root: str | Path, result: Mapping[str, Any]) -> str:
    """Persist a JSON and Markdown report after each manual Workbench execution."""
    root = Path(repository_root).resolve()
    run_id = str(result.get("run_id") or uuid.uuid4().hex)
    folder = root / "ore_reports" / "manual_runs" / datetime.now(timezone.utc).strftime("%Y%m%d")
    json_path, markdown_path = folder / f"{run_id}.json", folder / f"{run_id}.md"
    payload = {
        "schema": "ore.workbench.manual_run.v1",
        "run": {key: value for key, value in result.items() if key not in {"output", "arguments"}},
        "arguments": _bounded_value(result.get("arguments", {})),
        "output": _bounded_value(result.get("output")),
        "input_observation": result.get("input_observation", {}),
        "observation": _bounded_value(result.get("observation", {})),
        "profile": _bounded_value(result.get("profile", {})),
        "live_events": _bounded_value(result.get("live_events", [])),
    }
    _write_json(json_path, payload)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(
        "# ORE Workbench manual experiment\n\n"
        f"- Run ID: {run_id}\n"
        f"- Package: {result.get('component')}\n"
        f"- Callable: {result.get('callable')}\n"
        f"- Status: {result.get('status')}\n"
        f"- Duration: {result.get('elapsed_ms')} ms\n\n"
        "This report contains software execution observations, not proof of theory validity.\n",
        encoding="utf-8",
    )
    return str(json_path)
