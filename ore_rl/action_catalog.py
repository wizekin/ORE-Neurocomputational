"""Action catalogue and parameter validation for ore_rl (Phase C, Steps 15-18).

Defines the primitive action catalogue (Step 15), validated parameters and
budgets (Step 16), feasible-set construction with dependency masks (Step 17,
delegated to safety.py), and explicit stop/defer actions that distinguish
successful completion from safe termination without sufficient evidence
(Step 18).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .contracts import (
    ActionSpecRecord,
    ActionType,
    OREAction,
    ParameterSpec,
    ResourceEstimate,
    RiskLevel,
)


@dataclass(frozen=True, slots=True)
class CatalogAction:
    """A catalog entry: the immutable spec plus its runtime binding state."""

    spec: ActionSpecRecord
    available: bool = True

    @property
    def action_id(self) -> str:
        return self.spec.action_id

    def instantiate(
        self,
        parameters: Mapping[str, Any] | None = None,
        *,
        resource_scale: float = 1.0,
    ) -> OREAction:
        """Validate parameters and build a concrete OREAction bound to this spec."""
        merged = dict(parameters or {})
        errors = self.validate_parameters(merged)
        if errors:
            raise ValueError(
                f"action {self.spec.action_id} rejected parameters: {'; '.join(errors)}"
            )
        estimate = ResourceEstimate(
            latency_ms=self.spec.default_resource_estimate.latency_ms * resource_scale,
            cpu_cost=self.spec.default_resource_estimate.cpu_cost * resource_scale,
            memory_mb=self.spec.default_resource_estimate.memory_mb * resource_scale,
        )
        return OREAction(
            action_id=self.spec.action_id,
            action_type=self.spec.action_type,
            parameters=merged,
            target_module=self.spec.target_module,
            capability=self.spec.capability,
            resource_estimate=estimate,
            risk=self.spec.risk,
        )

    def validate_parameters(self, parameters: Mapping[str, Any]) -> list[str]:
        """Reject missing, out-of-range, unknown, or contradictory parameters (Step 16)."""
        errors: list[str] = []
        known = {item.name for item in self.spec.parameters}
        for name in parameters:
            if name not in known:
                errors.append(f"unknown parameter {name!r} for action {self.spec.action_id}")
        for item in self.spec.parameters:
            error = item.validate(parameters.get(item.name))
            if error:
                errors.append(error)
        if self.spec.action_type is ActionType.ADJUST_DEPTH:
            target_depth = parameters.get("target_depth")
            delta = parameters.get("depth_delta")
            if target_depth is not None and delta is not None:
                errors.append(
                    "contradictory parameters: provide either target_depth or depth_delta, not both"
                )
        if self.spec.action_type is ActionType.CONCURRENT_BATCH:
            members = parameters.get("members", ())
            if not isinstance(members, Sequence) or isinstance(members, (str, bytes)) or not members:
                errors.append("concurrent batch requires a non-empty 'members' sequence")
            elif len(set(members)) != len(members):
                errors.append("concurrent batch members must be unique")
        return errors


class ActionCatalog:
    """Registry of named operations and predefined cognitive options (Step 15)."""

    def __init__(self) -> None:
        self._specs: dict[str, ActionSpecRecord] = {}

    # -- registration ------------------------------------------------------
    def register(self, spec: ActionSpecRecord, *, replace: bool = False) -> None:
        if spec.action_id in self._specs and not replace:
            raise ValueError(f"action already registered: {spec.action_id}")
        self._specs[spec.action_id] = spec

    def unregister(self, action_id: str) -> None:
        self._specs.pop(action_id, None)

    def get(self, action_id: str) -> CatalogAction | None:
        spec = self._specs.get(action_id)
        return CatalogAction(spec) if spec else None

    def require(self, action_id: str) -> CatalogAction:
        entry = self.get(action_id)
        if entry is None:
            raise KeyError(f"action not registered: {action_id}")
        return entry

    def list_actions(self) -> tuple[CatalogAction, ...]:
        return tuple(CatalogAction(spec) for spec in self._specs.values())

    @property
    def action_ids(self) -> tuple[str, ...]:
        return tuple(self._specs)

    def __len__(self) -> int:
        return len(self._specs)

    # -- default catalogue -------------------------------------------------
    @classmethod
    def default_catalog(cls) -> ActionCatalog:
        """The standard meta-action catalogue: stop, defer, review, depth control."""
        catalog = cls()
        catalog.register(
            ActionSpecRecord(
                action_id="meta.stop_success",
                action_type=ActionType.STOP_SUCCESS,
                description="Report task completion with a verified conclusion.",
                parameters=(
                    ParameterSpec(
                        "conclusion",
                        "str",
                        required=False,
                        description="identifier of the verified conclusion artifact",
                    ),
                ),
                risk=RiskLevel.HIGH,
            )
        )
        catalog.register(
            ActionSpecRecord(
                action_id="meta.stop_insufficient_evidence",
                action_type=ActionType.STOP_INSUFFICIENT_EVIDENCE,
                description="Terminate safely because available evidence is insufficient.",
                risk=RiskLevel.LOW,
            )
        )
        catalog.register(
            ActionSpecRecord(
                action_id="meta.defer_resource_limit",
                action_type=ActionType.DEFER_RESOURCE_LIMIT,
                description="Defer the task because the resource budget is exhausted.",
                risk=RiskLevel.LOW,
            )
        )
        catalog.register(
            ActionSpecRecord(
                action_id="meta.request_human_review",
                action_type=ActionType.REQUEST_HUMAN_REVIEW,
                description="Escalate the decision to a human reviewer.",
                risk=RiskLevel.LOW,
            )
        )
        catalog.register(
            ActionSpecRecord(
                action_id="meta.adjust_depth",
                action_type=ActionType.ADJUST_DEPTH,
                description=(
                    "Increase or decrease processing depth for subsequent operations."
                ),
                parameters=(
                    ParameterSpec(
                        "target_depth", "int", required=False, minimum=1, maximum=5
                    ),
                    ParameterSpec(
                        "depth_delta", "int", required=False, minimum=-2, maximum=2
                    ),
                ),
                risk=RiskLevel.LOW,
                depth_change=1,
            )
        )
        return catalog

    def add_run_module_action(
        self,
        *,
        module_id: str,
        capability: str,
        action_type: ActionType = ActionType.RUN_MODULE,
        prerequisite_capabilities: Sequence[str] = (),
        latency_ms: float,
        cpu_cost: float,
        memory_mb: float,
        depth_parameter: bool = False,
        risk: RiskLevel = RiskLevel.LOW,
        description: str = "",
    ) -> str:
        """Register the RUN_MODULE-style action for a coordinator-executable module."""
        parameters: tuple[ParameterSpec, ...] = (
            (ParameterSpec(
                "depth", "int", required=False, minimum=1, maximum=5,
                description="processing depth requested from the module",
            ),)
            if depth_parameter
            else ()
        )
        spec = ActionSpecRecord(
            action_id=f"run.{module_id}",
            action_type=action_type,
            description=description or f"Execute module {module_id} ({capability}).",
            parameters=parameters,
            target_module=module_id,
            capability=capability,
            prerequisite_capabilities=tuple(prerequisite_capabilities),
            default_resource_estimate=ResourceEstimate(latency_ms, cpu_cost, memory_mb),
            risk=risk,
        )
        self.register(spec)
        return spec.action_id

    def add_parameterized_module_actions(
        self,
        *,
        module_id: str,
        capability: str,
        action_type: ActionType,
        prerequisite_capabilities: Sequence[str],
        parameter_name: str,
        parameter_values: Sequence[int],
        cost_at_value,
        risk: RiskLevel = RiskLevel.LOW,
        description: str = "",
    ) -> tuple[str, ...]:
        """Register discrete parameterized variants of one module action.

        Baking parameters (e.g. processing depth) into distinct action ids
        turns them into first-class learnable choices for bandits and DQN/PPO,
        while each variant stays statically validated.
        """
        ids: list[str] = []
        for value in parameter_values:
            action_id = f"run.{module_id}@{parameter_name}{value}"
            latency_ms, cpu_cost, memory_mb = cost_at_value(value)
            spec = ActionSpecRecord(
                action_id=action_id,
                action_type=action_type,
                description=description or f"Execute module {module_id} ({capability}) at {parameter_name}={value}.",
                parameters=(
                    ParameterSpec(
                        parameter_name,
                        "int",
                        required=True,
                        minimum=value,
                        maximum=value,
                        description=f"fixed value {value} for this variant",
                    ),
                ),
                target_module=module_id,
                capability=capability,
                prerequisite_capabilities=tuple(prerequisite_capabilities),
                default_resource_estimate=ResourceEstimate(latency_ms, cpu_cost, memory_mb),
                risk=risk,
                depth_change=0,
            )
            self.register(spec)
            ids.append(action_id)
        return tuple(ids)

    def feasibility_view(
        self, completed_capabilities: Iterable[str]
    ) -> dict[str, tuple[str, ...]]:
        """Prerequisite capabilities not yet satisfied, per action (Step 11 inputs)."""
        completed = set(completed_capabilities)
        return {
            spec.action_id: tuple(
                cap for cap in spec.prerequisite_capabilities if cap not in completed
            )
            for spec in self._specs.values()
        }
