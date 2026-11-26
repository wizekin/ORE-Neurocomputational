"""Hierarchical cognitive options: the Options Framework over ORE actions.

Point 10, Phase H (Steps 44-47):
    * An option is o = (I_o, pi_o, beta_o): initiation set, internal policy,
      termination condition (Step 44).
    * Every option declares a contract: prerequisites, permitted modules,
      required evidence, resource ceilings, expected outputs, failure paths,
      and termination rules; internal policies cannot bypass the global
      coordinator or hard constraints (Step 45).
    * A high-level policy learns option selection while internal procedures
      stay deterministic initially (Step 46).
    * Loop and nontermination detection: invocation ceilings, repeated state
      signatures, progress deltas, and wall-clock budgets (Step 47).

Initial option library: UNDERSTAND_INPUT, INVESTIGATE_HYPOTHESIS,
RETRIEVE_SUPPORTING_EVIDENCE, VERIFY_CONCLUSION, RESOLVE_CONTRADICTION,
PRODUCE_FINAL_RESULT.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..contracts import OREState, PolicyAuthority, PolicyDecision, RankedProposal, stable_hash
from .base import PolicyMetadata


@dataclass(frozen=True, slots=True)
class OptionContract:
    """Explicit contract and internal safeguards for one option (Step 45)."""

    prerequisites: tuple[str, ...]
    permitted_modules: tuple[str, ...]
    required_evidence: tuple[str, ...] = ()
    resource_ceiling_ms: float = 120.0
    expected_outputs: tuple[str, ...] = ()
    failure_paths: tuple[str, ...] = ("escalate",)
    max_internal_steps: int = 4
    max_invocations: int = 3


@dataclass(frozen=True, slots=True)
class CognitiveOption:
    """A temporally extended cognitive procedure (Step 44)."""

    option_id: str
    description: str
    initiation_capability: str | None
    termination_capability: str | None
    contract: OptionContract
    procedure: tuple[str, ...]

    def initiates(self, completed_capabilities: Sequence[str]) -> bool:
        """I_o: prerequisite capabilities must be complete."""
        completed = set(completed_capabilities)
        if self.initiation_capability and self.initiation_capability not in completed:
            return False
        return all(cap in completed for cap in self.contract.prerequisites)

    def terminates(
        self, completed_capabilities: Sequence[str], step: int
    ) -> bool:
        """beta_o: goal capability reached, internal step ceiling, or budget."""
        completed = set(completed_capabilities)
        if self.termination_capability and self.termination_capability in completed:
            return True
        return step >= self.contract.max_internal_steps

    def to_dict(self) -> dict[str, Any]:
        return {
            "option_id": self.option_id,
            "description": self.description,
            "initiation_capability": self.initiation_capability,
            "termination_capability": self.termination_capability,
            "procedure": list(self.procedure),
            "contract": {
                "prerequisites": list(self.contract.prerequisites),
                "permitted_modules": list(self.contract.permitted_modules),
                "max_internal_steps": self.contract.max_internal_steps,
                "max_invocations": self.contract.max_invocations,
                "resource_ceiling_ms": self.contract.resource_ceiling_ms,
            },
        }


class OptionLibrary:
    """The predefined option set (Step 44)."""

    def __init__(self) -> None:
        self._options: dict[str, CognitiveOption] = {}
        for option in self._default_options():
            self._options[option.option_id] = option

    @staticmethod
    def _default_options() -> tuple[CognitiveOption, ...]:
        analyzer = "run.ore.rl.input_analyzer"
        retriever = "run.ore.rl.knowledge_retriever"
        generator = "run.ore.rl.hypothesis_generator"
        verifier = "run.ore.rl.verifier"
        causal = "run.ore.rl.causal_analyzer"
        return (
            CognitiveOption(
                option_id="UNDERSTAND_INPUT",
                description="Analyze the task input into key terms and coverage signals.",
                initiation_capability=None,
                termination_capability="cognition.rl.input_analysis",
                contract=OptionContract(
                    prerequisites=(),
                    permitted_modules=(analyzer,),
                    expected_outputs=("ORE.RL.INPUT_ANALYSIS",),
                    max_internal_steps=2,
                    max_invocations=1,
                ),
                procedure=(analyzer,),
            ),
            CognitiveOption(
                option_id="RETRIEVE_SUPPORTING_EVIDENCE",
                description="Retrieve supporting knowledge for the analyzed input.",
                initiation_capability="cognition.rl.input_analysis",
                termination_capability="cognition.rl.knowledge_retrieval",
                contract=OptionContract(
                    prerequisites=("cognition.rl.input_analysis",),
                    permitted_modules=(retriever,),
                    expected_outputs=("ORE.RL.KNOWLEDGE_RECORDS",),
                    max_internal_steps=2,
                    max_invocations=2,
                ),
                procedure=(retriever,),
            ),
            CognitiveOption(
                option_id="INVESTIGATE_HYPOTHESIS",
                description="Generate a bounded hypothesis set from analysis and knowledge.",
                initiation_capability="cognition.rl.knowledge_retrieval",
                termination_capability="cognition.rl.hypothesis_generation",
                contract=OptionContract(
                    prerequisites=(
                        "cognition.rl.input_analysis",
                        "cognition.rl.knowledge_retrieval",
                    ),
                    permitted_modules=(generator,),
                    expected_outputs=("ORE.RL.HYPOTHESIS_SET",),
                    max_internal_steps=2,
                    max_invocations=2,
                ),
                procedure=(generator,),
            ),
            CognitiveOption(
                option_id="VERIFY_CONCLUSION",
                description="Verify the hypothesis set against retrieved knowledge.",
                initiation_capability="cognition.rl.hypothesis_generation",
                termination_capability="cognition.rl.verification",
                contract=OptionContract(
                    prerequisites=(
                        "cognition.rl.hypothesis_generation",
                        "cognition.rl.knowledge_retrieval",
                    ),
                    permitted_modules=(verifier,),
                    required_evidence=("ORE.RL.KNOWLEDGE_RECORDS",),
                    expected_outputs=("ORE.RL.VERIFIED_CONCLUSION",),
                    max_internal_steps=2,
                    max_invocations=2,
                ),
                procedure=(verifier,),
            ),
            CognitiveOption(
                option_id="RESOLVE_CONTRADICTION",
                description="Run causal consistency analysis over the hypotheses.",
                initiation_capability="cognition.rl.hypothesis_generation",
                termination_capability="cognition.rl.causal_analysis",
                contract=OptionContract(
                    prerequisites=("cognition.rl.hypothesis_generation",),
                    permitted_modules=(causal,),
                    expected_outputs=("ORE.RL.CAUSAL_CHECK",),
                    max_internal_steps=2,
                    max_invocations=1,
                ),
                procedure=(causal,),
            ),
            CognitiveOption(
                option_id="PRODUCE_FINAL_RESULT",
                description="Stop with the verified conclusion or defer safely.",
                initiation_capability="cognition.rl.verification",
                termination_capability=None,
                contract=OptionContract(
                    prerequisites=("cognition.rl.verification",),
                    permitted_modules=(),
                    failure_paths=("stop_insufficient_evidence", "defer_resource_limit"),
                    max_internal_steps=2,
                    max_invocations=2,
                ),
                procedure=("meta.stop_success", "meta.stop_insufficient_evidence"),
            ),
        )

    def list_options(self) -> tuple[CognitiveOption, ...]:
        return tuple(self._options.values())

    def get(self, option_id: str) -> CognitiveOption | None:
        return self._options.get(option_id)

    @property
    def option_ids(self) -> tuple[str, ...]:
        return tuple(self._options)


class OptionLoopDetector:
    """Detect option loops and nontermination (Step 47)."""

    def __init__(self, *, max_signature_repeats: int = 2, stall_window: int = 2) -> None:
        self.max_signature_repeats = int(max_signature_repeats)
        self.stall_window = int(stall_window)
        self.invocation_counts: dict[str, int] = {}
        self.state_signatures: dict[str, int] = {}
        self.progress_history: list[float] = []
        self.started = time.perf_counter()

    def state_signature(self, state: OREState) -> str:
        relevant = {
            name: round(float(value), 3)
            for name, value in state.features.items()
            if name.startswith(("goal.", "workspace.", "resource.", "dependency."))
        }
        return stable_hash(relevant)[:16]

    def record_invocation(self, option_id: str, state: OREState, progress: float) -> dict[str, Any]:
        """Record one option start; return escalation directive when looping."""
        self.invocation_counts[option_id] = self.invocation_counts.get(option_id, 0) + 1
        signature = self.state_signature(state)
        self.state_signatures[signature] = self.state_signatures.get(signature, 0) + 1
        self.progress_history.append(float(progress))
        directive: dict[str, Any] = {"escalate": False, "reason": ""}
        if (
            self.invocation_counts[option_id] > 1
            and self.state_signatures[signature] > self.max_signature_repeats
        ):
            directive = {
                "escalate": True,
                "reason": f"option {option_id} repeated state signature {signature}",
            }
        window = self.progress_history[-self.stall_window :]
        if len(window) == self.stall_window and all(
            abs(current - previous) < 1e-6
            for current, previous in zip(window, window[1:])
        ):
            directive = {"escalate": True, "reason": "progress stalled"}
        return directive

    def check_budget(self, option: CognitiveOption) -> bool:
        """Return True when the option exceeded its contract ceiling."""
        if self.invocation_counts.get(option.option_id, 0) > option.contract.max_invocations:
            return True
        return (time.perf_counter() - self.started) * 1000.0 > option.contract.resource_ceiling_ms * 10


class OptionRuntime:
    """Executes one option's internal procedure against an environment.

    The runtime steps through the option's deterministic internal policy
    (Step 46) but validates every primitive action against the environment's
    hard constraint gate; it can never bypass the coordinator (Step 45).
    """

    def __init__(self, environment: Any, *, max_total_steps: int = 16) -> None:
        self.environment = environment
        self.max_total_steps = int(max_total_steps)
        self.steps_used = 0
        self.completed_options: list[str] = []
        self.terminated = False
        self.terminal_kind: str = ""

    def run_option(self, option: CognitiveOption) -> dict[str, Any]:
        """Execute the option; return its execution report."""
        if self.terminated or self.steps_used >= self.max_total_steps:
            return {"option_id": option.option_id, "status": "SKIPPED", "steps": 0}
        report: dict[str, Any] = {
            "option_id": option.option_id,
            "steps": 0,
            "status": "RUNNING",
            "actions": [],
            "total_reward": 0.0,
        }
        internal_step = 0
        for primitive in option.procedure:
            if self.terminated or self.steps_used >= self.max_total_steps:
                break
            if option.terminates(
                self._completed_capabilities(), internal_step
            ):
                break
            action = self._resolve_action(primitive)
            state, reward, terminated, truncated, info = self.environment.step(action)
            self.steps_used += 1
            internal_step += 1
            report["steps"] += 1
            report["actions"].append(primitive)
            report["total_reward"] += float(reward)
            if info.get("rejected"):
                # Rejected primitives fall through to the procedure's next
                # (fallback) step; the failure path is recorded for audit.
                failure_path = self._failure_path(option, info)
                report["failure_path"] = failure_path
                continue
            if terminated or truncated:
                self.terminated = True
                self.terminal_kind = info.get("terminal_kind", "TRUNCATED")
                report["status"] = "TERMINATED"
                break
        if not self.terminated and not option.terminates(
            self._completed_capabilities(), internal_step
        ):
            # Deterministic fallback: complete the option's contract safely
            # (stop with the verified conclusion, otherwise stop insufficient).
            fallback = "meta.stop_success" if self._verified_available() else (
                "meta.stop_insufficient_evidence"
            )
            state, reward, terminated, truncated, info = self.environment.step(fallback)
            self.steps_used += 1
            report["steps"] += 1
            report["actions"].append(fallback)
            report["total_reward"] += float(reward)
            if terminated or truncated:
                self.terminated = True
                self.terminal_kind = info.get("terminal_kind", "STOP_INSUFFICIENT_EVIDENCE")
                report["status"] = "TERMINATED"
        if report["status"] == "RUNNING":
            report["status"] = "COMPLETED"
        self.completed_options.append(option.option_id)
        return report

    def _verified_available(self) -> bool:
        episode = getattr(self.environment, "_episode", None)
        if episode is None:
            return False
        return episode.verified_term is not None

    def _resolve_action(self, primitive: str) -> Any:
        """Resolve an option primitive into an environment action id.

        Bare module references (``run.<module>``) are bound to the depth
        variant closest to the coordinator's difficulty-scaled target depth;
        fully parameterized ids and meta actions pass through unchanged.
        """
        if primitive.startswith("run.") and "@depth" not in primitive:
            module = primitive[len("run."):]
            prefix = f"run.{module}@depth"
            variants = []
            for entry in self.environment.catalog_actions.list_actions():
                spec = entry.spec
                if spec.action_id.startswith(prefix):
                    try:
                        depth = int(spec.action_id.split("@depth", 1)[1])
                    except ValueError:
                        continue
                    variants.append((depth, spec.action_id))
            if variants:
                target = self._target_depth()
                best = min(variants, key=lambda item: abs(item[0] - target))
                return best[1]
        return primitive

    def _target_depth(self) -> int:
        episode = getattr(self.environment, "_episode", None)
        if episode is None:
            return 3
        state = self.environment._observe()
        difficulty = state.features.get("task.difficulty", 0.5)
        if difficulty is None or difficulty != difficulty:
            difficulty = 0.5
        return max(1, min(5, 3 + int(float(difficulty) >= 0.34) + int(float(difficulty) >= 0.67)))

    def _completed_capabilities(self) -> set[str]:
        episode = getattr(self.environment, "_episode", None)
        return set(episode.completed_capabilities) if episode is not None else set()

    @staticmethod
    def _failure_path(option: CognitiveOption, info: Mapping[str, Any]) -> str:
        violations = info.get("violations", ())
        reasons = " ".join(item.get("reason", "") for item in violations)
        if "verified conclusion" in reasons:
            return "stop_insufficient_evidence"
        if "budget" in reasons or "exceeds remaining" in reasons:
            return "defer_resource_limit"
        return option.contract.failure_paths[0] if option.contract.failure_paths else "escalate"


class HierarchicalOptionPolicy:
    """High-level policy that selects among validated options (Step 46).

    Option selection is learned with a bandit over option identities; the
    internal procedures remain deterministic. The policy only ever proposes
    options whose initiation sets are satisfied and whose invocation ceiling
    has not been exceeded.
    """

    def __init__(
        self,
        library: OptionLibrary | None = None,
        *,
        seed: int = 0,
        exploration: float = 0.35,
    ) -> None:
        self.library = library or OptionLibrary()
        self.rng: Any = None
        import numpy as np

        self._np = np
        self.rng = np.random.default_rng(seed)
        self.exploration = float(exploration)
        self.stats: dict[str, list[float]] = {}
        self.decisions = 0
        self.metadata = PolicyMetadata(
            policy_id="hierarchical_options",
            policy_version="1.0.0",
            algorithm_family="hierarchical_rl/options",
            uncertainty_behavior="option_level_ucb",
            execution_authority=PolicyAuthority.PROPOSAL,
            description="option-level bandit over deterministic procedures",
        )

    def eligible_options(
        self, completed_capabilities: Sequence[str], invocation_counts: Mapping[str, int]
    ) -> tuple[CognitiveOption, ...]:
        """Options whose initiation sets hold and ceilings are not exceeded.

        Options that would still add a capability (their termination
        capability is not yet completed) are preferred over re-runs; the
        fallback keeps options such as PRODUCE_FINAL_RESULT available.
        """
        completed = set(completed_capabilities)
        eligible: list[CognitiveOption] = []
        for option in self.library.list_options():
            if not option.initiates(completed):
                continue
            ceiling = invocation_counts.get(option.option_id, 0)
            if ceiling >= option.contract.max_invocations:
                continue
            eligible.append(option)
        progressing = [
            option
            for option in eligible
            if option.termination_capability and option.termination_capability not in completed
        ]
        if progressing:
            return tuple(progressing)
        # Everything achievable is done: offer the completion option.
        produce = [option for option in eligible if option.option_id == "PRODUCE_FINAL_RESULT"]
        return tuple(produce) if produce else tuple(eligible)

    def propose_actions(
        self,
        state: OREState,
        eligible_actions: Sequence[str],
        budget: Mapping[str, float] | None = None,
    ) -> PolicyDecision:
        """Rank options (identified by the option: prefix) within eligibility."""
        option_ids = [
            action_id[len("option:"):]
            for action_id in eligible_actions
            if action_id.startswith("option:")
        ]
        if not option_ids:
            return PolicyDecision(
                policy_id=self.metadata.policy_id,
                policy_version=self.metadata.policy_version,
                chosen_action_id="",
                fallback_used=True,
                rationale="no eligible options",
            )
        self.decisions += 1
        scores = {}
        for option_id in option_ids:
            history = self.stats.get(option_id, [])
            mean = sum(history) / len(history) if history else 0.0
            bonus = self.exploration * self._np.sqrt(
                self._np.log(1 + self.decisions) / (1 + len(history))
            )
            scores[option_id] = mean + bonus
        best = max(scores, key=lambda option_id: (scores[option_id], option_id))
        ranked = tuple(
            RankedProposal(
                action_id=f"option:{option_id}", score=scores[option_id], uncertainty=1.0
            )
            for option_id in sorted(scores, key=lambda o: -scores[o])
        )
        return PolicyDecision(
            policy_id=self.metadata.policy_id,
            policy_version=self.metadata.policy_version,
            ranked=ranked,
            chosen_action_id=f"option:{best}",
            propensity=1.0,
            uncertainty=0.5,
            authority=PolicyAuthority.PROPOSAL,
            rationale="option-level UCB over deterministic internal procedures",
        )

    def update_option_outcome(
        self, option_id: str, reward: float, *, verified: bool = True
    ) -> None:
        """Learn option selection from measured option-level outcomes (Step 46)."""
        if not verified:
            return
        self.stats.setdefault(option_id, []).append(float(reward))
