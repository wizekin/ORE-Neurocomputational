"""Task-profile and cognitive-task helpers for adaptive USG resource recruitment."""
from __future__ import annotations

from typing import Any

from ore_contracts.resource_profiles import CapabilityDemand, TaskProfile
from ore_contracts.coordinator import CognitiveTask

from .contracts import ORE_USG_KNOWLEDGE_CONTRACT
from .registration import USG_KNOWLEDGE_CAPABILITIES, USG_KNOWLEDGE_MODULE_ID


def build_usg_knowledge_task_profile(
    task_id: str,
    *,
    max_module_calls: int = 1,
    max_replans: int = 0,
) -> TaskProfile:
    """Build a bounded profile requiring successful lexical, semantic, structure, and provenance import."""
    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError("task_id must be non-empty")
    return TaskProfile(
        task_id=task_id,
        task_family="usg_knowledge_import",
        demands=tuple(
            CapabilityDemand(
                capability_id=capability,
                priority=1.0,
                mandatory=True,
                target_confidence=0.9,
            )
            for capability in USG_KNOWLEDGE_CAPABILITIES
        ),
        stop_confidence=0.9,
        max_module_calls=max_module_calls,
        max_replans=max_replans,
        execution_mode="FUNCTIONAL",
        metadata={
            "support_semantics": (
                "operational import completion only; not semantic truth, coverage, or factual confidence"
            ),
            "expected_output_contract": ORE_USG_KNOWLEDGE_CONTRACT,
        },
    )


def build_usg_knowledge_cognitive_task(
    bundle_artifact: Any,
    task_id: str,
    *,
    goal: str = "Import and expose one validated USG knowledge bundle to ORE",
) -> CognitiveTask:
    """Create a normal CognitiveTask routed to the registered unified USG module."""
    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError("task_id must be non-empty")
    if not isinstance(goal, str) or not goal.strip():
        raise ValueError("goal must be non-empty")
    return CognitiveTask(
        goal=goal,
        required_capabilities=USG_KNOWLEDGE_CAPABILITIES,
        input_artifacts={
            USG_KNOWLEDGE_MODULE_ID: {"knowledge_bundle": bundle_artifact},
        },
        expected_outputs=(ORE_USG_KNOWLEDGE_CONTRACT,),
        execution_mode="FUNCTIONAL",
        priority=1.0,
        max_replans=0,
        metadata={
            "task_family": "usg_knowledge_import",
            "read_only": True,
        },
        task_id=task_id,
    )


def coordinate_usg_knowledge_import(
    adaptive_coordinator: Any,
    bundle_artifact: Any,
    task_id: str,
    *,
    context: Any | None = None,
) -> Any:
    """Run the USG import through ORE's adaptive coordinator and bounded task profile."""
    coordinate = getattr(adaptive_coordinator, "coordinate", None)
    if not callable(coordinate):
        raise TypeError("adaptive_coordinator must expose coordinate(profile, task, context)")
    profile = build_usg_knowledge_task_profile(task_id)
    task = build_usg_knowledge_cognitive_task(bundle_artifact, task_id)
    return coordinate(profile, task, context)


__all__ = [
    "build_usg_knowledge_task_profile",
    "build_usg_knowledge_cognitive_task",
    "coordinate_usg_knowledge_import",
]
