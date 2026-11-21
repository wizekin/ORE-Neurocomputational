"""Version-aware operator registry with deterministic enumeration."""
from __future__ import annotations

from threading import RLock
from .models import Operator


class OperatorRegistry:
    def __init__(self) -> None:
        self._operators: dict[tuple[str, str], Operator] = {}
        self._lock = RLock()

    def register(self, operator: Operator, *, replace: bool = False) -> None:
        operator_id = getattr(operator, "operator_id", "")
        version = getattr(operator, "version", "")
        if not operator_id or not version:
            raise ValueError("operators must declare non-empty operator_id and version")
        if getattr(operator, "estimated_cost", 0) < 0:
            raise ValueError("estimated_cost must be >= 0")
        key = (operator_id, version)
        with self._lock:
            if key in self._operators and not replace:
                raise ValueError(f"operator already registered: {operator_id}@{version}")
            self._operators[key] = operator

    def unregister(self, operator_id: str, version: str) -> bool:
        with self._lock:
            return self._operators.pop((operator_id, version), None) is not None

    def list_operators(self) -> tuple[Operator, ...]:
        with self._lock:
            return tuple(self._operators[key] for key in sorted(self._operators))
