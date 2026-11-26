"""Gymnasium-compatible ORE training and simulation environment (Step 1.3, roadmap Stage 3).

API shape follows the Gymnasium ``reset(seed) -> (obs, info)`` /
``step(action) -> (obs, reward, terminated, truncated, info)`` convention
without importing gymnasium (the repository is numpy/scipy only).

Every cognitive action is executed through the real
``CognitiveTaskCoordinator`` with validated ``ContractArtifact`` inputs, so
module dispatch, timeout, and artifact validation follow the canonical ORE
path (Step 19). The RL-visible blackboard is a real ``SharedWorkspace``
persisted across the episode; the coordinator's own per-step workspaces are
harvested into it via ``FindingPublisher``.

Determinism: task sampling, module outputs, synthetic measured costs, and
failure injection are all deterministic functions of the seed and episode
identifiers. Wall-clock times are recorded for audit but do not influence
rewards, keeping training data reproducible (Step 8.2).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ore_contracts import (
    ComputeBudget,
    ContractArtifact,
    ContractCatalog,
    CognitiveTask,
    CognitiveTaskCoordinator,
    ModuleRegistry,
    ModuleResult,
    ModuleStatus,
    OperationMode,
    RunContext,
)
from ore_contracts.adapters import make_envelope
from ore_workspace import FindingPublisher, WorkspaceManager

from . import modules as example_modules
from .action_catalog import ActionCatalog
from .contracts import (
    ActionOutcome,
    ActionType,
    OREAction,
    OREState,
    OREReward,
    OutcomeStatus,
    ResourceEstimate,
    RiskLevel,
)
from .reward import (
    AttributionMethod,
    CostModel,
    EvidenceQualityEvaluator,
    EvaluatorIdentity,
    RewardWeights,
    TaskQualityEvaluator,
    TerminalOutcome,
    compute_step_reward,
    compute_terminal_reward,
)
from .safety import HardConstraintGate, ResourceBudgetState
from .state_encoder import (
    CapabilityStateSignal,
    DependencyStateSignal,
    GoalProgressSignal,
    RawStateSignals,
    ResourceStateSignal,
    StateEncoder,
    TaskContextSignal,
    UncertaintyHistorySignal,
    WorkspaceEvidenceSignal,
)

ENVIRONMENT_VERSION = "ore-rl-env-1.0.0"

TASK_FAMILIES = (
    "fact_verification",
    "story_comprehension",
    "term_definition",
)
DOMAINS = ("science", "general")

_VOCABULARY = (
    "gradient", "resonance", "circuit", "membrane", "particle", "cascade",
    "vector", "threshold", "envelope", "lattice", "photon", "synapse",
)

_TEMPLATES = {
    "fact_verification": "claim: the {filler} {subject} governs the {object} response",
    "story_comprehension": "the {filler} {subject} recalled the {object} after the event",
    "term_definition": "define the {filler} {subject} used in {object} research",
}

_FILLERS = ("reported", "primary", "observed", "canonical", "secondary")


@dataclass(frozen=True, slots=True)
class TaskInstance:
    """One deterministic cognitive task with reference ground truth (Step 21)."""

    task_id: str
    task_family: str
    domain: str
    input_text: str
    ground_truth_term: str
    difficulty: float
    urgency: float
    constraints: tuple[str, ...] = ()
    ground_truth_position: int = 0

    @property
    def requested_output(self) -> str:
        return {
            "fact_verification": "verified_claim",
            "story_comprehension": "answer_question",
            "term_definition": "definition",
        }[self.task_family]


class TaskSampler:
    """Deterministic task generator over the three example families."""

    def __init__(
        self,
        seed: int = 0,
        *,
        families: Sequence[str] = TASK_FAMILIES,
        n_tasks: int = 30,
    ) -> None:
        self.seed = int(seed)
        self.families = tuple(families)
        self.n_tasks = int(n_tasks)

    def sample(self) -> tuple[TaskInstance, ...]:
        import random

        random.Random(self.seed)  # seed the module generator deterministically
        tasks: list[TaskInstance] = []
        for index in range(self.n_tasks):
            family = self.families[index % len(self.families)]
            domain = DOMAINS[index % len(DOMAINS)]
            subject = _VOCABULARY[(index * 5 + 3) % len(_VOCABULARY)]
            obj = _VOCABULARY[(index * 7 + 1) % len(_VOCABULARY)]
            difficulty = 0.15 + 0.7 * ((index * 13 + 5) % 10) / 9.0
            position = min(2, int(difficulty * 3))
            filler = " ".join(_FILLERS[(index + offset) % len(_FILLERS)] for offset in range(position))
            text = _TEMPLATES[family].format(filler=filler, subject=subject, object=obj)
            tasks.append(
                TaskInstance(
                    task_id=f"task_{self.seed}_{index:03d}",
                    task_family=family,
                    domain=domain,
                    input_text=text,
                    ground_truth_term=subject,
                    difficulty=round(difficulty, 3),
                    urgency=round(0.2 + 0.6 * ((index * 3) % 5) / 4.0, 3),
                    constraints=("verify_before_answer",) if family == "fact_verification" else (),
                    ground_truth_position=position,
                )
            )
        return tuple(tasks)


@dataclass(frozen=True, slots=True)
class EnvironmentBudget:
    """Hard resource budget enforced by the environment gates (Step 12)."""

    latency_ms: float = 280.0
    cpu_cost: float = 60.0
    memory_mb: float = 64.0
    max_module_calls: int = 12
    max_steps: int = 20


@dataclass
class _EpisodeState:
    """Mutable per-episode bookkeeping (never exposed to policies directly)."""

    task: TaskInstance
    episode_id: str
    step_index: int = 0
    depth: int = 2
    completed_capabilities: set[str] = field(default_factory=set)
    executed_action_ids: set[str] = field(default_factory=set)
    artifacts: dict[str, ContractArtifact] = field(default_factory=dict)
    findings: list[dict[str, Any]] = field(default_factory=list)
    module_calls: int = 0
    latency_spent_ms: float = 0.0
    cpu_spent: float = 0.0
    memory_peak_mb: float = 0.0
    wasted_actions: int = 0
    hard_violations: int = 0
    action_history: list[str] = field(default_factory=list)
    action_outcomes: dict[str, list[float]] = field(default_factory=dict)
    verified_term: str | None = None
    hypothesis_terms: list[str] = field(default_factory=list)
    quality_total: float = 0.0
    evidence_total: float = 0.0
    terminated: bool = False
    stop_label: str = ""
    terminal_kind: str = ""


def _result_status_to_outcome(status: ModuleStatus) -> OutcomeStatus:
    return {
        ModuleStatus.SUCCESS: OutcomeStatus.SUCCESS,
        ModuleStatus.PARTIAL: OutcomeStatus.PARTIAL,
        ModuleStatus.FAILED: OutcomeStatus.EXCEPTION,
        ModuleStatus.ABSTAINED: OutcomeStatus.CANCELLED,
        ModuleStatus.UNAVAILABLE: OutcomeStatus.RESOURCE_EXHAUSTED,
    }[status]


class OREEnvironment:
    """Deterministic, Gymnasium-shaped ORE cognitive-decision environment."""

    def __init__(
        self,
        *,
        registry: ModuleRegistry,
        catalog: ContractCatalog,
        tasks: Sequence[TaskInstance] | None = None,
        sampler: TaskSampler | None = None,
        budget: EnvironmentBudget | None = None,
        weights: RewardWeights | None = None,
        cost_model: CostModel | None = None,
        failure_injection: Mapping[str, float] | None = None,
        usg_resource: Any | None = None,
        seed: int = 7,
        max_workers: int = 2,
    ) -> None:
        self.registry = registry
        self.catalog = catalog
        self.sampler = sampler or TaskSampler(seed=seed)
        self.tasks = tuple(tasks) if tasks is not None else self.sampler.sample()
        if not self.tasks:
            raise ValueError("environment requires at least one task")
        self.budget = budget or EnvironmentBudget()
        self.weights = weights or RewardWeights()
        self.cost_model = cost_model or CostModel(
            latency_budget_ms=self.budget.latency_ms,
            cpu_budget=self.budget.cpu_cost,
            memory_budget_mb=self.budget.memory_mb,
        )
        self.quality_evaluator = TaskQualityEvaluator()
        self.evidence_evaluator = EvidenceQualityEvaluator()
        self.evaluator_identity = EvaluatorIdentity(
            "ore_rl.environment", ENVIRONMENT_VERSION
        )
        self.seed = int(seed)
        self._episode_counter = 0
        self._task_pointer = 0
        self._rng_seed = seed
        self.coordinator = CognitiveTaskCoordinator(
            registry,
            catalog,
            enable_logging=False,
            max_workers=max_workers,
            default_timeout_seconds=10.0,
        )
        self.workspace_manager = WorkspaceManager(enable_logging=False)
        self.catalog_actions = self._build_catalog()
        self.encoder = StateEncoder(task_families=TASK_FAMILIES, domains=DOMAINS)
        self.gate = HardConstraintGate(self.catalog_actions)
        self._episode: _EpisodeState | None = None
        self._last_reward: OREReward | None = None
        self._workspace: Any | None = None
        self._publisher: FindingPublisher | None = None
        self._episode_history: list[dict[str, Any]] = []
        self._family_stats: dict[str, list[float]] = {}
        self.usg_resource = usg_resource
        if usg_resource is not None:
            self._attach_usg(usg_resource)
        self.last_info: dict[str, Any] = {}

    # -- construction helpers ------------------------------------------------
    def _attach_usg(self, resource: Any) -> None:
        from ore_usg.retrieval import USGKnowledgeIndex

        index = USGKnowledgeIndex(resource)
        backend = self.registry.get_backend(f"{example_modules.MODULE_PREFIX}.knowledge_retriever")
        if backend is not None and hasattr(backend, "usg_index"):
            backend.usg_index = index

    def _build_catalog(self) -> ActionCatalog:
        catalog = ActionCatalog.default_catalog()
        specs = {
            f"{example_modules.MODULE_PREFIX}.input_analyzer": (
                example_modules.CAP_INPUT_ANALYSIS,
                (),
            ),
            f"{example_modules.MODULE_PREFIX}.knowledge_retriever": (
                example_modules.CAP_KNOWLEDGE_RETRIEVAL,
                (example_modules.CAP_INPUT_ANALYSIS,),
            ),
            f"{example_modules.MODULE_PREFIX}.hypothesis_generator": (
                example_modules.CAP_HYPOTHESIS_GENERATION,
                (
                    example_modules.CAP_INPUT_ANALYSIS,
                    example_modules.CAP_KNOWLEDGE_RETRIEVAL,
                ),
            ),
            f"{example_modules.MODULE_PREFIX}.causal_analyzer": (
                example_modules.CAP_CAUSAL_ANALYSIS,
                (example_modules.CAP_HYPOTHESIS_GENERATION,),
            ),
            f"{example_modules.MODULE_PREFIX}.verifier": (
                example_modules.CAP_VERIFICATION,
                (
                    example_modules.CAP_HYPOTHESIS_GENERATION,
                    example_modules.CAP_KNOWLEDGE_RETRIEVAL,
                ),
            ),
        }
        for module_id, (capability, prerequisites) in specs.items():
            profile = example_modules.DEFAULT_COST_PROFILES[module_id]
            action_type = (
                ActionType.RETRIEVE_KNOWLEDGE
                if capability == example_modules.CAP_KNOWLEDGE_RETRIEVAL
                else (
                    ActionType.GENERATE_HYPOTHESES
                    if capability == example_modules.CAP_HYPOTHESIS_GENERATION
                    else (
                        ActionType.CAUSAL_ANALYSIS
                        if capability == example_modules.CAP_CAUSAL_ANALYSIS
                        else (
                            ActionType.VERIFY
                            if capability == example_modules.CAP_VERIFICATION
                            else ActionType.RUN_MODULE
                        )
                    )
                )
            )
            catalog.add_parameterized_module_actions(
                module_id=module_id,
                capability=capability,
                action_type=action_type,
                prerequisite_capabilities=prerequisites,
                parameter_name="depth",
                parameter_values=(1, 2, 3, 4, 5),
                cost_at_value=profile.at_depth,
                risk=RiskLevel.LOW,
            )
        catalog.register(_batch_spec())
        return catalog

    # -- gymnasium-shaped API -------------------------------------------------
    def reset(
        self, *, seed: int | None = None, options: Mapping[str, Any] | None = None
    ) -> tuple[OREState, dict[str, Any]]:
        """Start a new episode; deterministic for a given seed (Stage 3 exit test)."""
        if seed is not None:
            self.seed = int(seed)
            self._rng_seed = int(seed)
            self._task_pointer = 0
            self._episode_counter = 0
        if self._workspace is not None:
            self.workspace_manager.close_workspace(self._workspace.workspace_id)
            self._workspace = None
        task = self.tasks[self._task_pointer % len(self.tasks)]
        self._task_pointer += 1
        self._episode_counter += 1
        episode_id = f"ep_{self._rng_seed}_{self._episode_counter:04d}"
        self._episode = _EpisodeState(task=task, episode_id=episode_id)
        self._workspace = self.workspace_manager.create_workspace(
            task.input_text, f"complete_{task.task_family}"
        )
        self._publisher = FindingPublisher(self._workspace, "ore.rl_environment")
        state = self._observe()
        info = {
            "task_id": task.task_id,
            "task_family": task.task_family,
            "episode_id": episode_id,
            "feasible_action_ids": self._current_feasible_ids(),
            "workspace_id": self._workspace.workspace_id,
        }
        self.last_info = info
        return state, info

    def step(
        self, action: OREAction | str | tuple[str, Mapping[str, Any]]
    ) -> tuple[OREState, float, bool, bool, dict[str, Any]]:
        """Execute one validated cognitive action through the coordinator."""
        if self._episode is None or self._episode.terminated:
            raise RuntimeError("episode is not running; call reset() first")
        episode = self._episode
        concrete = self._concrete_action(action)
        state_before = self._observe()
        feasible = self._feasible_set(state_before)
        reward_total, outcome, terminated, truncated, info = self._execute(
            episode, concrete, feasible
        )
        episode.step_index += 1
        state_after = self._observe()
        info.update(
            {
                "episode_id": episode.episode_id,
                "step_index": episode.step_index,
                "outcome": outcome.to_dict(),
                "feasible_action_ids": self._current_feasible_ids(),
                "action_id": concrete.action_id,
                "reward_components": self._last_reward.to_dict(),
            }
        )
        self.last_info = info
        return state_after, reward_total, terminated, truncated, info

    # -- execution ---------------------------------------------------------------
    def _execute(
        self,
        episode: _EpisodeState,
        action: OREAction,
        feasible,
    ):
        quality_before = episode.quality_total
        evidence_before = episode.evidence_total
        if not feasible.is_feasible(action.action_id):
            episode.wasted_actions += 1
            outcome = ActionOutcome(
                status=OutcomeStatus.INVALID_OUTPUT,
                error="action rejected by the hard constraint gate",
            )
            reward = self._step_reward(episode, outcome, quality_before, evidence_before)
            return reward.total, outcome, False, False, {
                "rejected": True,
                "violations": [
                    {"constraint": item.constraint.value, "reason": item.reason}
                    for item in feasible.rejections.get(action.action_id, ())
                ],
            }
        action_type = action.action_type
        if action_type in (
            ActionType.STOP_SUCCESS,
            ActionType.STOP_INSUFFICIENT_EVIDENCE,
            ActionType.DEFER_RESOURCE_LIMIT,
            ActionType.REQUEST_HUMAN_REVIEW,
        ):
            return self._terminal_step(episode, action)
        if action_type is ActionType.ADJUST_DEPTH:
            outcome = self._adjust_depth(episode, action)
        elif action_type is ActionType.CONCURRENT_BATCH:
            outcome = self._concurrent_batch(episode, action)
        else:
            outcome = self._run_module(episode, action)
        self._refresh_quality(episode)
        self._refresh_evidence(episode)
        terminal = False
        truncated = self._budget_exhausted(episode)
        reward = self._step_reward(episode, outcome, quality_before, evidence_before)
        if outcome.status in (OutcomeStatus.EXCEPTION, OutcomeStatus.INVALID_OUTPUT):
            episode.wasted_actions += 1
        episode.action_history.append(action.action_id)
        outcomes = episode.action_outcomes.setdefault(action.action_id, [])
        outcomes.append(reward.total)
        info: dict[str, Any] = {
            "measured_latency_ms": outcome.measured_latency_ms,
            "measured_cpu_cost": outcome.measured_cpu_cost,
            "measured_memory_mb": outcome.measured_memory_mb,
            "artifacts_published": len(outcome.artifact_refs),
        }
        if truncated:
            terminal_outcome = ActionOutcome(status=OutcomeStatus.RESOURCE_EXHAUSTED)
            episode.terminated = True
            episode.terminal_kind = "TRUNCATED_BUDGET"
            self._episode_history.append(
                self._episode_record(episode, reward.total, terminal_outcome)
            )
            return reward.total, outcome, True, True, info
        return reward.total, outcome, terminal, truncated, info

    def _run_module(self, episode: _EpisodeState, action: OREAction) -> ActionOutcome:
        module_id = action.target_module
        assert module_id is not None
        depth = int(action.parameters.get("depth", episode.depth))
        profile = example_modules.DEFAULT_COST_PROFILES[module_id]
        latency, cpu, memory = profile.at_depth(depth)
        if action.step_budget_ms is not None and latency > action.step_budget_ms:
            return ActionOutcome(
                status=OutcomeStatus.TIMEOUT,
                measured_latency_ms=latency,
                error=f"synthetic latency {latency:.1f} ms exceeded step budget",
            )
        manifest = self.registry.get_manifest(module_id)
        if manifest is None:
            return ActionOutcome(status=OutcomeStatus.INVALID_OUTPUT, error="unknown module")
        inputs: dict[str, ContractArtifact] = {}
        for port in manifest.input_ports:
            if port.contract_id == example_modules.CONTRACT_INPUT_TEXT:
                inputs[port.port_id] = self._input_artifact(episode)
                continue
            artifact = episode.artifacts.get(port.contract_id)
            if artifact is not None:
                inputs[port.port_id] = artifact
        missing = [
            port.port_id
            for port in manifest.input_ports
            if port.required and port.port_id not in inputs
        ]
        if missing:
            return ActionOutcome(
                status=OutcomeStatus.INVALID_OUTPUT,
                error=f"missing required inputs: {missing}",
            )
        task = CognitiveTask(
            goal=f"execute {module_id} for {episode.task.task_id}",
            required_capabilities=(manifest.capabilities[0],),
            input_artifacts=inputs,
            expected_outputs=(manifest.output_ports[0].contract_id,),
            metadata={"input_text": episode.task.input_text},
        )
        run_id = f"rl-{episode.episode_id}-{episode.step_index:03d}"
        context = RunContext(
            run_id=run_id,
            mode=OperationMode.FUNCTIONAL,
            deterministic_seed=self._rng_seed,
            compute_budget=ComputeBudget(wall_clock_seconds=10.0),
            settings_json=f'{{"rl_depth": {depth}}}',
        )
        started = time.perf_counter()
        result = self.coordinator.coordinate(task, context)
        module_result: ModuleResult | None = result.module_results.get(module_id)
        episode.module_calls += 1
        episode.latency_spent_ms += latency
        episode.cpu_spent += cpu
        episode.memory_peak_mb = max(episode.memory_peak_mb, memory)
        if module_result is None or result.status in ("BLOCKED", "FAILED"):
            status = (
                OutcomeStatus.RESOURCE_EXHAUSTED
                if result.status == "BLOCKED"
                else OutcomeStatus.EXCEPTION
            )
            return ActionOutcome(
                status=status,
                measured_latency_ms=latency,
                measured_cpu_cost=cpu,
                measured_memory_mb=memory,
                error=f"coordinator status {result.status}",
            )
        outcome_status = _result_status_to_outcome(module_result.status)
        if outcome_status is not OutcomeStatus.SUCCESS:
            return ActionOutcome(
                status=outcome_status,
                measured_latency_ms=latency,
                measured_cpu_cost=cpu,
                measured_memory_mb=memory,
                error="module did not succeed",
            )
        capability = manifest.capabilities[0]
        if capability not in episode.completed_capabilities:
            episode.completed_capabilities.add(capability)
        for artifact in module_result.artifacts:
            episode.artifacts[artifact.envelope.contract_id] = artifact
            self._publish_finding(episode, module_id, artifact, depth)
        episode.executed_action_ids.add(action.action_id)
        self._absorb_payload(episode, module_result)
        return ActionOutcome(
            status=OutcomeStatus.SUCCESS,
            measured_latency_ms=latency,
            measured_cpu_cost=cpu,
            measured_memory_mb=memory,
            artifact_refs=tuple(
                artifact.envelope.fingerprint()
                for artifact in module_result.artifacts
            ),
            findings_published=len(module_result.artifacts),
        )

    def _concurrent_batch(self, episode: _EpisodeState, action: OREAction) -> ActionOutcome:
        members = tuple(action.parameters.get("members", ()))
        manifest_by_module = {
            entry.spec.target_module: entry
            for entry in self.catalog_actions.list_actions()
            if entry.spec.target_module
        }
        executed: list[str] = []
        latency_total = 0.0
        cpu_total = 0.0
        memory_peak = 0.0
        statuses: list[OutcomeStatus] = []
        for member in members:
            if not member.startswith("run."):
                statuses.append(OutcomeStatus.INVALID_OUTPUT)
                continue
            module_id = member[len("run."):].split("@", 1)[0]
            depth = episode.depth
            if "@depth" in member:
                try:
                    depth = int(member.split("@depth", 1)[1])
                except ValueError:
                    depth = episode.depth
            entry = manifest_by_module.get(module_id)
            if entry is None or not self._independent(episode, module_id):
                statuses.append(OutcomeStatus.INVALID_OUTPUT)
                continue
            member_action = OREAction(
                action_id=member,
                action_type=ActionType.RUN_MODULE,
                parameters={"depth": depth},
                target_module=module_id,
                capability=entry.spec.capability,
                resource_estimate=entry.spec.default_resource_estimate,
            )
            outcome = self._run_module(episode, member_action)
            statuses.append(outcome.status)
            executed.append(member)
            latency_total = max(latency_total, outcome.measured_latency_ms)
            cpu_total += outcome.measured_cpu_cost
            memory_peak = max(memory_peak, outcome.measured_memory_mb)
        if any(status is OutcomeStatus.SUCCESS for status in statuses):
            status = (
                OutcomeStatus.SUCCESS
                if all(status is OutcomeStatus.SUCCESS for status in statuses)
                else OutcomeStatus.PARTIAL
            )
        else:
            status = statuses[0] if statuses else OutcomeStatus.INVALID_OUTPUT
        return ActionOutcome(
            status=status,
            measured_latency_ms=latency_total,
            measured_cpu_cost=cpu_total,
            measured_memory_mb=memory_peak,
            error=None if status is OutcomeStatus.SUCCESS else "some batch members failed",
        )

    def _independent(self, episode: _EpisodeState, module_id: str) -> bool:
        spec = next(
            (
                item
                for item in self.catalog_actions.list_actions()
                if item.spec.target_module == module_id
            ),
            None,
        )
        if spec is None:
            return False
        return all(
            cap in episode.completed_capabilities
            for cap in spec.spec.prerequisite_capabilities
        )

    def _adjust_depth(self, episode: _EpisodeState, action: OREAction) -> ActionOutcome:
        target = action.parameters.get("target_depth")
        delta = action.parameters.get("depth_delta")
        if target is not None:
            new_depth = int(target)
        elif delta is not None:
            new_depth = episode.depth + int(delta)
        else:
            new_depth = episode.depth
        new_depth = max(1, min(5, new_depth))
        changed = new_depth != episode.depth
        episode.depth = new_depth
        return ActionOutcome(
            status=OutcomeStatus.SUCCESS if changed else OutcomeStatus.PARTIAL,
            measured_latency_ms=0.5,
            measured_cpu_cost=0.01,
            measured_memory_mb=0.0,
            error=None if changed else "depth unchanged",
        )

    def _terminal_step(
        self, episode: _EpisodeState, action: OREAction
    ) -> tuple[float, ActionOutcome, bool, bool, dict[str, Any]]:
        goals_met = (
            example_modules.CAP_VERIFICATION in episode.completed_capabilities
            and episode.verified_term == episode.task.ground_truth_term
        )
        evidence_sufficient = episode.evidence_total >= 0.35
        premature = action.action_type is ActionType.STOP_INSUFFICIENT_EVIDENCE and evidence_sufficient
        terminal = TerminalOutcome(
            goals_met=goals_met,
            constraints_preserved=episode.hard_violations == 0,
            final_answer_verified=episode.verified_term is not None,
            stopped_prematurely=premature,
        )
        reward = compute_terminal_reward(
            terminal,
            weights=self.weights,
            reward_spec_fingerprint="env-terminal",
            evaluator_identity=self.evaluator_identity,
            evidence_refs=(),
        )
        self._last_reward = reward
        outcome = ActionOutcome(status=OutcomeStatus.TERMINAL)
        episode.terminated = True
        episode.terminal_kind = action.action_type.value
        episode.stop_label = "correct" if goals_met else "premature"
        self._episode_history.append(self._episode_record(episode, reward.total, outcome))
        info = {
            "stop_label": episode.stop_label,
            "terminal_kind": episode.terminal_kind,
            "goals_met": goals_met,
        }
        return reward.total, outcome, True, False, info

    # -- reward and quality ---------------------------------------------------------
    def _step_reward(
        self,
        episode: _EpisodeState,
        outcome: ActionOutcome,
        quality_before: float,
        evidence_before: float,
    ):
        reward = compute_step_reward(
            quality_before=quality_before,
            quality_after=episode.quality_total,
            evidence_before=evidence_before,
            evidence_after=episode.evidence_total,
            outcome=outcome,
            weights=self.weights,
            cost_model=self.cost_model,
            evaluator_identity=self.evaluator_identity,
            reward_spec_fingerprint="env-step",
            evidence_refs=(),
            attribution_method=AttributionMethod.STEP.value,
            verified=True,
        )
        self._last_reward = reward
        return reward

    def _refresh_quality(self, episode: _EpisodeState) -> None:
        task = episode.task
        if episode.verified_term is not None:
            correctness = (
                1.0 if episode.verified_term == task.ground_truth_term else 0.5
            )
        elif task.ground_truth_term in episode.hypothesis_terms:
            correctness = 0.3
        else:
            correctness = 0.0
        flow = (
            example_modules.CAP_INPUT_ANALYSIS,
            example_modules.CAP_KNOWLEDGE_RETRIEVAL,
            example_modules.CAP_HYPOTHESIS_GENERATION,
            example_modules.CAP_VERIFICATION,
        )
        completeness = (
            sum(1 for cap in flow if cap in episode.completed_capabilities) / len(flow)
        )
        key_terms = episode.artifacts.get(
            example_modules.CONTRACT_INPUT_ANALYSIS
        )
        relevant_terms = (
            list(key_terms.payload.get("key_terms", ()))
            if key_terms is not None and isinstance(key_terms.payload, Mapping)
            else episode.hypothesis_terms
        )
        relevance = (
            1.0
            if (
                episode.verified_term is not None
                and episode.verified_term in relevant_terms
            )
            else (
                0.5
                if any(term in relevant_terms for term in episode.hypothesis_terms)
                else 0.0
            )
        )
        steps = max(1, episode.step_index + 1)
        compliance = 1.0 - min(1.0, episode.wasted_actions / steps)
        score = self.quality_evaluator.score(
            correctness=correctness,
            completeness=completeness,
            relevance=relevance,
            instruction_compliance=compliance,
            evidence_refs=[episode.task.task_id],
        )
        episode.quality_total = score.total

    def _refresh_evidence(self, episode: _EpisodeState) -> None:
        score = self.evidence_evaluator.score_findings(episode.findings)
        episode.evidence_total = score.total

    def _publish_finding(
        self,
        episode: _EpisodeState,
        module_id: str,
        artifact: ContractArtifact,
        depth: int,
    ) -> None:
        assert self._publisher is not None
        payload: dict[str, Any] = {
            "module": module_id,
            "contract_id": artifact.envelope.contract_id,
            "artifact_fingerprint": artifact.envelope.fingerprint(),
            "depth": depth,
        }
        if isinstance(artifact.payload, Mapping):
            payload["payload_keys"] = sorted(artifact.payload.keys())
            if "records" in artifact.payload:
                payload["record_count"] = len(artifact.payload["records"])
                payload["source_reliability"] = (
                    sum(
                        float(item.get("source_reliability", 0.5))
                        for item in artifact.payload["records"]
                    )
                    / max(1, len(artifact.payload["records"]))
                )
        result = self._publisher.publish(
            "semantic_hypotheses",
            payload,
            key=artifact.envelope.contract_id,
            finding_type="module_hypothesis",
            provenance=None,
            confidence=None,
        )
        relevant = False
        analysis = episode.artifacts.get(example_modules.CONTRACT_INPUT_ANALYSIS)
        if analysis is not None and isinstance(analysis.payload, Mapping):
            terms = analysis.payload.get("key_terms", ())
            relevant = episode.task.ground_truth_term in terms
        supported = artifact.envelope.contract_id in (
            example_modules.CONTRACT_VERIFIED_CONCLUSION,
            example_modules.CONTRACT_KNOWLEDGE_RECORDS,
        )
        episode.findings.append(
            {
                "finding_id": result.finding_id if result else "",
                "module": module_id,
                "has_provenance": True,
                "relevant": relevant,
                "supported": supported,
                "contradicts": False,
                "source_reliability": float(payload.get("source_reliability", 0.6)),
                "evidence_refs": [artifact.envelope.fingerprint()],
            }
        )

    def _absorb_payload(self, episode: _EpisodeState, result: ModuleResult) -> None:
        for artifact in result.artifacts:
            payload = artifact.payload
            if not isinstance(payload, Mapping):
                continue
            if artifact.envelope.contract_id == example_modules.CONTRACT_INPUT_ANALYSIS:
                episode.hypothesis_terms = list(payload.get("key_terms", ()))
            elif artifact.envelope.contract_id == example_modules.CONTRACT_HYPOTHESIS_SET:
                episode.hypothesis_terms = [
                    str(item.get("term"))
                    for item in payload.get("hypotheses", ())
                    if isinstance(item, Mapping)
                ]
            elif artifact.envelope.contract_id == example_modules.CONTRACT_VERIFIED_CONCLUSION:
                if payload.get("verified"):
                    verified = payload.get("verified_hypothesis")
                    episode.verified_term = (
                        str(verified.get("term")) if isinstance(verified, Mapping) else None
                    )
                else:
                    episode.verified_term = None

    def _input_artifact(self, episode: _EpisodeState) -> ContractArtifact:
        run_id = f"rl-{episode.episode_id}-input"
        return ContractArtifact(
            make_envelope(
                example_modules.CONTRACT_INPUT_TEXT,
                run_id,
                "ore.rl_environment",
                evidence_kind="OBSERVED",
            ),
            episode.task.input_text,
        )

    def _budget_exhausted(self, episode: _EpisodeState) -> bool:
        return (
            episode.latency_spent_ms >= self.budget.latency_ms
            or episode.cpu_spent >= self.budget.cpu_cost
            or episode.module_calls >= self.budget.max_module_calls
            or episode.step_index + 1 >= self.budget.max_steps
        )

    # -- observation -------------------------------------------------------------
    def _observe(self) -> OREState:
        episode = self._episode
        assert episode is not None and self._workspace is not None
        snapshot = self._workspace.snapshot()
        revision = int(getattr(snapshot, "revision", 0) or 0)
        findings = episode.findings
        verified_count = sum(1 for item in findings if item.get("supported"))
        module_ids = [item for item in example_modules.MODULE_IDS]
        readiness = []
        for module_id in module_ids:
            manifest = self.registry.get_manifest(module_id)
            records = (
                self.registry.find_by_capability(manifest.capabilities[0]) if manifest else ()
            )
            readiness.append(1.0 if records and records[0].runnable else 0.0)
        health = [0.95] * len(module_ids)
        queues = [0] * len(module_ids)
        estimated_cost = [
            example_modules.DEFAULT_COST_PROFILES[module_id].base_cpu_cost
            for module_id in module_ids
        ]
        specs = self.catalog_actions.feasibility_view(episode.completed_capabilities)
        dependency_flags = tuple(not specs[action_id] for action_id in sorted(specs))
        dependency_signal = DependencyStateSignal(
            action_ids=tuple(sorted(specs)),
            dependency_satisfied=dependency_flags,
        )
        family_history = self._family_stats.get(episode.task.task_family)
        signals = RawStateSignals(
            task=TaskContextSignal(
                task_family=episode.task.task_family,
                domain=episode.task.domain,
                requested_output=episode.task.requested_output,
                difficulty=episode.task.difficulty,
                urgency=episode.task.urgency,
                constraints=episode.task.constraints,
            ),
            goal=GoalProgressSignal(
                completed_subgoals=tuple(sorted(episode.completed_capabilities)),
                remaining_subgoals=tuple(
                    cap
                    for cap in example_modules.ALL_CAPABILITIES
                    if cap not in episode.completed_capabilities
                ),
                unresolved_requirements=() if not episode.findings else (
                    "evidence_gap",
                ),
                completion_confidence=episode.quality_total,
            ),
            workspace=WorkspaceEvidenceSignal(
                workspace_revision=revision,
                finding_count=len(findings),
                verified_finding_count=verified_count,
                unverified_finding_count=len(findings) - verified_count,
                contradiction_count=0,
                mean_source_reliability=(
                    sum(item.get("source_reliability", 0.6) for item in findings) / len(findings)
                    if findings
                    else 0.0
                ),
                evidence_quality_score=episode.evidence_total,
                evidence_quantity_score=min(1.0, len(findings) / 6.0),
            ),
            capabilities=CapabilityStateSignal(
                module_ids=tuple(module_ids),
                readiness=tuple(readiness),
                health=tuple(health),
                queue_lengths=tuple(queues),
                estimated_cost=tuple(estimated_cost),
            ),
            dependencies=dependency_signal,
            resources=ResourceStateSignal(
                remaining_wall_clock_fraction=max(
                    0.0, 1.0 - episode.latency_spent_ms / max(1e-9, self.budget.latency_ms)
                ),
                remaining_cpu_fraction=max(
                    0.0, 1.0 - episode.cpu_spent / max(1e-9, self.budget.cpu_cost)
                ),
                remaining_memory_fraction=max(
                    0.0,
                    1.0
                    - episode.memory_peak_mb / max(1e-9, self.budget.memory_mb),
                ),
                remaining_module_calls_fraction=max(
                    0.0,
                    1.0 - episode.module_calls / max(1, self.budget.max_module_calls),
                ),
                reserved_wall_clock_fraction=0.0,
            ),
            uncertainty=UncertaintyHistorySignal(
                evaluator_confidence=0.8,
                action_sample_counts={
                    action_id: len(values)
                    for action_id, values in episode.action_outcomes.items()
                },
                action_mean_outcomes={
                    action_id: sum(values) / len(values)
                    for action_id, values in episode.action_outcomes.items()
                    if values
                },
                similar_task_success_rate=(
                    sum(family_history) / len(family_history)
                    if family_history
                    else None
                ),
                similar_task_sample_count=len(family_history) if family_history else 0,
            ),
            task_id=episode.task.task_id,
            episode_id=episode.episode_id,
            step_index=episode.step_index,
            provenance={
                "workspace_id": self._workspace.workspace_id,
                "environment_version": ENVIRONMENT_VERSION,
            },
        )
        return self.encoder.encode(signals)

    # -- feasibility ------------------------------------------------------------
    def _feasible_set(self, state: OREState | None = None):
        episode = self._episode
        assert episode is not None
        state = state if state is not None else self._observe()
        budget = ResourceBudgetState(
            remaining_latency_ms=max(
                0.0, self.budget.latency_ms - episode.latency_spent_ms
            ),
            remaining_cpu_cost=max(0.0, self.budget.cpu_cost - episode.cpu_spent),
            remaining_memory_mb=max(
                0.0, self.budget.memory_mb - episode.memory_peak_mb
            ),
            remaining_module_calls=self.budget.max_module_calls - episode.module_calls,
        )
        runnable: dict[str, bool] = {}
        for module_id in example_modules.MODULE_IDS:
            manifest = self.registry.get_manifest(module_id)
            if manifest is None:
                runnable[module_id] = False
                continue
            records = self.registry.find_by_capability(manifest.capabilities[0])
            runnable[module_id] = bool(records) and records[0].runnable
        gate = HardConstraintGate(self.catalog_actions)
        return gate.compute_feasible_actions(
            state,
            completed_capabilities=episode.completed_capabilities,
            budget=budget,
            module_runnable=runnable,
            verified_conclusion_available=episode.verified_term is not None,
            current_depth=episode.depth,
            executed_action_ids=episode.executed_action_ids,
        )

    def _current_feasible_ids(self) -> tuple[str, ...]:
        return self._feasible_set().feasible_ids

    def feasible_actions(self) -> tuple[str, ...]:
        """Action ids currently feasible (the policy's action space)."""
        return self._current_feasible_ids()

    def build_action(self, action_id: str, **parameters: Any) -> OREAction:
        """Instantiate a catalog action with validated parameters.

        Parameterized module actions (``run.<module>@depth<k>``) carry their
        fixed depth in the spec; callers only override parameters explicitly.
        Concurrent batches auto-fill their members with every currently
        feasible, dependency-satisfied module action when the caller does not
        name an explicit member list (Step 15: "run independent eligible
        operations concurrently").
        """
        entry = self.catalog_actions.require(action_id)
        merged = dict(parameters)
        for item in entry.spec.parameters:
            if item.required and item.name not in merged:
                merged[item.name] = _fixed_value(item)
        if entry.spec.action_type is ActionType.CONCURRENT_BATCH and "members" not in merged:
            merged["members"] = self._auto_batch_members()
        return entry.instantiate(merged, resource_scale=1.0)

    def _auto_batch_members(self) -> tuple[str, ...]:
        episode = self._episode
        if episode is None:
            return ()
        members: list[str] = []
        for entry in self.catalog_actions.list_actions():
            spec = entry.spec
            if spec.target_module is None:
                continue
            if spec.action_id in episode.executed_action_ids:
                continue
            if not all(
                cap in episode.completed_capabilities
                for cap in spec.prerequisite_capabilities
            ):
                continue
            members.append(spec.action_id)
        return tuple(members)

    def _concrete_action(self, action: OREAction | str | tuple[str, Mapping[str, Any]]) -> OREAction:
        if isinstance(action, OREAction):
            return action
        if isinstance(action, str):
            return self.build_action(action)
        action_id, parameters = action
        return self.build_action(action_id, **dict(parameters))

    # -- episode records -----------------------------------------------------------
    def _episode_record(
        self, episode: _EpisodeState, terminal_reward: float, outcome: ActionOutcome
    ) -> dict[str, Any]:
        record = {
            "episode_id": episode.episode_id,
            "task_id": episode.task.task_id,
            "task_family": episode.task.task_family,
            "steps": episode.step_index,
            "module_calls": episode.module_calls,
            "latency_ms": round(episode.latency_spent_ms, 2),
            "cpu_cost": round(episode.cpu_spent, 2),
            "memory_mb": round(episode.memory_peak_mb, 2),
            "quality": round(episode.quality_total, 4),
            "evidence": round(episode.evidence_total, 4),
            "verified_term": episode.verified_term,
            "ground_truth": episode.task.ground_truth_term,
            "success": episode.verified_term == episode.task.ground_truth_term,
            "stop_label": episode.stop_label,
            "terminal_kind": episode.terminal_kind,
            "terminal_reward": round(terminal_reward, 4),
            "wasted_actions": episode.wasted_actions,
            "hard_violations": episode.hard_violations,
            "depth": episode.depth,
            "action_history": list(episode.action_history),
        }
        family = episode.task.task_family
        history = self._family_stats.setdefault(family, [])
        history.append(1.0 if record["success"] else 0.0)
        return record

    def episode_history(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._episode_history)

    def episode_summary(self) -> dict[str, Any]:
        if not self._episode_history:
            return {"episodes": 0}
        records = self._episode_history
        successes = [record for record in records if record["success"]]
        return {
            "episodes": len(records),
            "success_rate": len(successes) / len(records),
            "mean_quality": sum(record["quality"] for record in records) / len(records),
            "mean_evidence": sum(record["evidence"] for record in records) / len(records),
            "mean_latency_ms": sum(record["latency_ms"] for record in records) / len(records),
            "mean_steps": sum(record["steps"] for record in records) / len(records),
            "mean_module_calls": sum(record["module_calls"] for record in records) / len(records),
            "mean_cost_per_success": (
                sum(record["latency_ms"] for record in successes) / len(successes)
                if successes
                else None
            ),
        }

    def close(self) -> None:
        self.workspace_manager.close()

    def __enter__(self) -> OREEnvironment:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _fixed_value(item: Any) -> int:
    """Extract the single fixed value from a variant's parameter spec."""
    if item.minimum is not None and item.minimum == item.maximum:
        return int(item.minimum)
    return 1


def _batch_spec():
    from .contracts import ActionSpecRecord, ParameterSpec

    return ActionSpecRecord(
        action_id="meta.concurrent_batch",
        action_type=ActionType.CONCURRENT_BATCH,
        description="Run independent eligible operations concurrently in one step.",
        parameters=(
            ParameterSpec(
                "members",
                "sequence",
                required=False,
                description="action ids of the independent operations (auto-filled when omitted)",
            ),
        ),
        prerequisite_capabilities=(),
        default_resource_estimate=ResourceEstimate(24.0, 2.5, 6.0),
        risk=RiskLevel.MEDIUM,
    )
