"""Index ore_cwm symbols into the shared SQLite logger."""
from __future__ import annotations

from .observability import register_cwm_annotations

register_annotations = register_cwm_annotations

__all__ = ["register_annotations", "register_cwm_annotations"]
