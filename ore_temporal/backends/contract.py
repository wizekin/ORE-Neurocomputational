"""The hardware-independent backend contract (specification Step 40)."""
from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable

from ..representations.portable_graph import PortableGraph

__all__ = ("TemporalBackend",)


@runtime_checkable
class TemporalBackend(Protocol):
    """A backend executes portable graphs without leaking device scheduling details."""

    @property
    def name(self) -> str: ...

    def capabilities(self) -> "Any": ...

    def execute(self, graph: PortableGraph, inputs: Mapping[str, Any],
                states: Mapping[str, Any] | None = None) -> Mapping[str, Any]: ...
