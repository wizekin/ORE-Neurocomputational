"""Bounded worker agents, the supervisor/worker protocol, and dispatch execution.

Specialist agents are bounded decision-making participants (spec Section 4C).
Each agent declares an explicit role, capability contract, permitted workspace
scopes, and a small set of declared actions; execution may choose among those
actions but can never enlarge authority, change the goal, or dispatch unapproved
work (Step 19). Completion means contract satisfaction, not merely a response
(Step 23). Deterministic specialists remain available as fallbacks (Step 14).
"""
from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from typing import Any

from .contracts import (
    AgentDescriptor,
    AgentResult,
    AgentTask,
    ArtifactRef,
    AutonomyLevel,
    ProvenanceTrail,
    SchemaSpec,
)
from .errors import (
    PermanentAgentError,
    TransientAgentError,
)
from .workspace_adapter import ScopedSnapshot, WorkspaceArtifactAdapter

# --------------------------------------------------------------------------- #
# Clocks and execution context
# --------------------------------------------------------------------------- #

class WallClock:
    """Provide monotonic wall-clock time for deadlines and latency accounting."""

    def now(self) -> float:
        """Return the current monotonic timestamp."""
        return time.perf_counter()


class LogicalClock:
    """Provide deterministic, manually advanced logical time for reproducible runs."""

    def __init__(self, start: float = 0.0, step: float = 1.0) -> None:
        """Initialize the logical clock with a start value and step size."""
        self._now = float(start)
        self._step = float(step)

    def now(self) -> float:
        """Advance by one step and return the new logical timestamp."""
        self._now += self._step
        return self._now


@dataclass
class ExecutionContext:
    """Carry the task, scoped snapshot, clock, and cooperative cancellation (Step 17)."""

    task: AgentTask
    snapshot: ScopedSnapshot
    clock: Any = field(default_factory=WallClock)
    progress: Callable[..., None] | None = None
    cancellation_requested: Callable[[str], bool] = field(default=lambda task_id: False)

    def report_progress(self, phase: str, state: str, *, blockers: str = "", estimate: float | None = None) -> None:
        """Emit progress with a phase and state; heartbeats are not progress (Step 20)."""
        if self.progress is not None:
            self.progress(phase=phase, state=state, blockers=blockers, estimate=estimate)

    def is_cancelled(self) -> bool:
        """Check the cooperative cancellation flag for this task."""
        return bool(self.cancellation_requested(self.task.task_id))


@dataclass(frozen=True, slots=True)
class AcceptDecision:
    """Record one agent's accept/reject decision with an explicit reason (Step 18)."""

    accepted: bool
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a detached decision record."""
        return {"accepted": self.accepted, "reason": self.reason}


# --------------------------------------------------------------------------- #
# Deterministic fallback specialists (Step 14)
# --------------------------------------------------------------------------- #

_BASIC_POS = {
    "the": "det", "a": "det", "an": "det", "of": "prep", "in": "prep", "on": "prep",
    "to": "prep", "with": "prep", "and": "conj", "or": "conj", "but": "conj",
    "is": "verb", "was": "verb", "are": "verb", "were": "verb", "chased": "verb",
    "saw": "verb", "ran": "verb", "bit": "verb", "ate": "verb", "startled": "verb",
    "argued": "verb", "claimed": "verb", "ignored": "verb",
}

_SUBJECT_ROLE_WORDS = frozenset({
    "dog", "cat", "scientist", "student", "reporter", "hunter", "fox", "teacher",
    "editor", "patient", "horse", "child", "man", "woman",
})


def deterministic_lexical_hypotheses(text: str) -> dict[str, Any]:
    """Produce baseline lexical hypotheses without any autonomous agent (Step 14)."""
    tokens = [token for token in text.split() if token]
    hypotheses = {
        token: [{"candidate": _BASIC_POS.get(token.lower().strip(".,!?;:"), "noun"),
                 "confidence": 0.6, "source": "deterministic_fallback"}]
        for token in tokens
    }
    coverage = len(hypotheses) / len(tokens) if tokens else 0.0
    return {"token_hypotheses": hypotheses, "coverage": coverage}


def deterministic_syntactic_hypothesis(text: str) -> dict[str, Any]:
    """Produce a baseline subject-verb-object parse without an autonomous agent."""
    tokens = [token.strip(".,!?;:").lower() for token in text.split() if token]
    verbs = [token for token in tokens if _BASIC_POS.get(token) == "verb"]
    parse = {"strategy": "deterministic_fallback", "tokens": tokens,
             "verb_count": len(verbs), "subject_guess": next(
                 (token for token in tokens if token in _SUBJECT_ROLE_WORDS), tokens[0] if tokens else "",
             )}
    return {"parse": parse, "interpretation": "deterministic_shallow_parse",
            "ambiguity": 0.5 if len(verbs) > 1 else 0.2}


def deterministic_semantic_hypothesis(text: str) -> dict[str, Any]:
    """Produce a baseline literal semantic interpretation without an autonomous agent."""
    tokens = {token.strip(".,!?;:").lower() for token in text.split() if token}
    if not tokens:
        return {"meaning": {}, "interpretation": "empty_input", "confidence": 0.0}
    subject = next((token for token in tokens if token in _SUBJECT_ROLE_WORDS), min(tokens))
    verb = next((token for token in tokens if _BASIC_POS.get(token) == "verb"), "relates_to")
    return {
        "meaning": {"subject": subject, "relation": verb, "literal": True},
        "interpretation": f"{subject} {verb.replace('_', ' ')} something",
        "confidence": 0.55,
    }


# --------------------------------------------------------------------------- #
# Worker protocol
# --------------------------------------------------------------------------- #

class BaseBoundedAgent:
    """Implement the supervisor/worker protocol for one bounded agent (Steps 17-23)."""

    declared_actions: tuple[str, ...] = ()

    def __init__(self, descriptor: AgentDescriptor, *, clock: Any = None) -> None:
        """Bind the agent to its contract and a clock for latency accounting."""
        if not isinstance(descriptor, AgentDescriptor):
            raise TypeError("descriptor must be an AgentDescriptor")
        self.descriptor = descriptor
        self.clock = clock or WallClock()
        self._recorded_results: dict[str, AgentResult] = {}
        self._cancelled_tasks: set[str] = set()
        self._rejected: dict[str, str] = {}
        self.execution_count = 0

    @property
    def agent_id(self) -> str:
        """Return the agent's stable identity."""
        return self.descriptor.agent_id

    def accept_task(self, task: AgentTask, snapshot: ScopedSnapshot) -> AcceptDecision:
        """Check capability, data, permission, and budget before accepting (Step 18)."""
        missing = set(task.required_capabilities) - set(self.descriptor.capability_ids)
        if missing:
            return AcceptDecision(False, f"missing capabilities: {sorted(missing)}")
        unsupported = set(task.permitted_workspace_fields) - set(self.descriptor.permitted_workspace_fields)
        if unsupported:
            return AcceptDecision(False, f"task requires out-of-scope fields: {sorted(unsupported)}")
        if task.cost_budget > self.descriptor.resources.cost_budget:
            return AcceptDecision(False, "task cost budget exceeds the agent envelope")
        if not str(snapshot.read("current_input") or "").strip():
            return AcceptDecision(False, "no current input available in the scoped snapshot")
        return AcceptDecision(True, "capability, data, permission, and budget checks passed")

    def execute(self, context: ExecutionContext) -> AgentResult:
        """Run one bounded execution, record its outcome, and honour cancellation (Steps 19-22)."""
        task = context.task
        if task.task_id in self._cancelled_tasks:
            self._rejected[task.task_id] = "cancelled before dispatch"
            raise PermanentAgentError(f"task {task.task_id} was cancelled before dispatch")
        started = self.clock.now()
        self.execution_count += 1
        try:
            result = self._perform(context)
        except PermanentAgentError:
            raise
        except TransientAgentError:
            raise
        except Exception as exc:
            raise TransientAgentError(f"agent {self.agent_id} failed on {task.task_id}: {exc}") from exc
        elapsed = max(0.0, self.clock.now() - started)
        recorded = AgentResult(
            agent_id=self.agent_id,
            agent_version=self.descriptor.agent_version,
            task_id=task.task_id,
            goal_id=task.goal_id,
            source_revision=context.snapshot.revision,
            outputs=dict(result["outputs"]),
            provenance=result.get("provenance") or ProvenanceTrail(
                agent_id=self.agent_id, task_id=task.task_id,
                operation=f"{type(self).__name__}.execute", workspace_revision=context.snapshot.revision,
            ),
            input_refs=tuple(result.get("input_refs", ())),
            evidence_refs=tuple(result.get("evidence_refs", ())),
            assumptions=tuple(result.get("assumptions", ())),
            uncertainty=float(result.get("uncertainty", 0.2)),
            resource_usage=float(result.get("resource_usage", 1.0)),
            elapsed_seconds=elapsed,
            self_evaluation=str(result.get("self_evaluation", "")),
            self_confidence=float(result.get("self_confidence", 0.5)),
            limitations=tuple(result.get("limitations", ())),
            verifier_requirements=tuple(
                task.verifier_requirements if task.requires_independent_verification else ()
            ),
        )
        self._recorded_results[task.task_id] = recorded
        return recorded

    def _perform(self, context: ExecutionContext) -> Mapping[str, Any]:
        """Produce one result payload; overridden by every specialist agent."""
        raise NotImplementedError("specialist agents must implement _perform")

    def cancel(self, task_id: str) -> bool:
        """Request cooperative cancellation for one task (Step 21)."""
        self._cancelled_tasks.add(task_id)
        return task_id in self._recorded_results or task_id in self._cancelled_tasks

    def reconcile(self, task_id: str) -> AgentResult | None:
        """Return the recorded outcome after a timeout, or None when unknown (Step 52)."""
        return self._recorded_results.get(task_id)

    def status(self, task_id: str) -> str:
        """Report the agent's local view of one task."""
        if task_id in self._recorded_results:
            return "completed"
        if task_id in self._cancelled_tasks:
            return "cancelled"
        return "unknown"

    def recorded_tasks(self) -> tuple[str, ...]:
        """Return every task this agent has recorded an outcome for."""
        return tuple(sorted(self._recorded_results))


def _input_evidence(adapter: WorkspaceArtifactAdapter, text: str) -> tuple[ArtifactRef, ...]:
    """Build the canonical input artifact reference for evidence citation."""
    return (adapter.artifact_reference(
        artifact_id="current_input", location="workspace:current_input", payload=text,
    ),)


# --------------------------------------------------------------------------- #
# Specialist agents
# --------------------------------------------------------------------------- #

class LexicalAnalysisAgent(BaseBoundedAgent):
    """Choose between surface and context-aware token analysis (bounded action selection)."""

    declared_actions = ("surface_tokenize", "context_aware_analysis")

    def _perform(self, context: ExecutionContext) -> Mapping[str, Any]:
        """Analyze every token and cite the input span as evidence."""
        text = str(context.snapshot.read("current_input") or "")
        existing = context.snapshot.read("lexical_hypotheses") or {}
        action = "context_aware_analysis" if existing else "surface_tokenize"
        context.report_progress("lexical_analysis", "running")
        tokens = [token for token in text.split() if token]
        hypotheses: dict[str, list[dict[str, Any]]] = {}
        for token in tokens:
            key = token.lower().strip(".,!?;:")
            candidates = [{
                "candidate": _BASIC_POS.get(key, "noun"),
                "confidence": 0.8 if action == "context_aware_analysis" else 0.65,
                "action": action,
            }]
            if key in existing and isinstance(existing[key], list) and existing[key]:
                prior = existing[key][0].get("candidate")
                if prior and prior != candidates[0]["candidate"]:
                    candidates.append({"candidate": prior, "confidence": 0.5, "action": "prior_context"})
            hypotheses[token] = candidates
        coverage = len(hypotheses) / len(tokens) if tokens else 0.0
        return {
            "outputs": {"token_hypotheses": hypotheses, "coverage": coverage},
            "evidence_refs": _input_evidence_for_context(context),
            "uncertainty": 0.15 if action == "context_aware_analysis" else 0.25,
            "self_confidence": 0.75,
            "self_evaluation": f"applied {action} to {len(tokens)} tokens",
            "assumptions": (f"chosen action: {action}",),
            "resource_usage": 1.0 + 0.05 * len(tokens),
        }


def _input_evidence_for_context(context: ExecutionContext) -> tuple[ArtifactRef, ...]:
    """Cite the canonical input artifact from inside an execution context."""
    text = str(context.snapshot.read("current_input") or "")
    return (ArtifactRef(
        artifact_id="current_input",
        payload_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        location="workspace:current_input",
    ),)


class SyntacticAnalysisAgent(BaseBoundedAgent):
    """Choose a parse strategy and request bounded clarification on high ambiguity."""

    declared_actions = ("shallow_parse", "structure_first_parse")

    def _perform(self, context: ExecutionContext) -> Mapping[str, Any]:
        """Parse with a chosen strategy and escalate ambiguity as a clarification request."""
        text = str(context.snapshot.read("current_input") or "")
        lexical = context.snapshot.read("lexical_hypotheses") or {}
        tokens = [token for token in text.split() if token]
        action = "structure_first_parse" if len(tokens) > 5 else "shallow_parse"
        context.report_progress("syntactic_analysis", "running")
        lowered = [token.strip(".,!?;:").lower() for token in tokens]
        verbs = [token for token in lowered if _BASIC_POS.get(token) == "verb"]
        subject = next((token for token in lowered if token in _SUBJECT_ROLE_WORDS), lowered[0] if lowered else "")
        ambiguity = min(1.0, 0.35 + 0.3 * max(0, len(verbs) - 1))
        if ambiguity >= 0.75:
            context.report_progress(
                "syntactic_analysis", "blocked",
                blockers="multiple verb candidates produce competing parses",
                estimate=None,
            )
        parse = {
            "strategy": action, "subject": subject,
            "verbs": verbs, "token_count": len(tokens),
            "lexical_coverage": len(lexical),
            "attachment": "subject_first",
        }
        return {
            "outputs": {
                "parse": parse,
                "interpretation": f"{subject or 'unknown'} performs {verbs[0] if verbs else 'an action'}",
                "ambiguity": ambiguity,
            },
            "evidence_refs": _input_evidence_for_context(context),
            "uncertainty": ambiguity,
            "self_confidence": 1.0 - ambiguity,
            "self_evaluation": f"applied {action}",
            "assumptions": (f"chosen action: {action}", f"verb candidates: {verbs}"),
            "resource_usage": 1.5 + 0.05 * len(tokens),
        }


class SemanticInterpretationAgent(BaseBoundedAgent):
    """Integrate lexical and syntactic evidence into a semantic proposal (Step 24-28)."""

    declared_actions = ("literal_integration", "contextual_integration")

    def _perform(self, context: ExecutionContext) -> Mapping[str, Any]:
        """Choose literal or contextual integration based on available workspace context."""
        lexical = context.snapshot.read("lexical_hypotheses") or {}
        syntactic = context.snapshot.read("syntactic_hypotheses") or []
        discourse = context.snapshot.read("discourse_context") or {}
        action = "contextual_integration" if discourse else "literal_integration"
        context.report_progress("semantic_interpretation", "running")
        latest_parse = syntactic[-1] if isinstance(syntactic, list) and syntactic else {}
        subject = (latest_parse.get("payload", {}).get("parse", {}) or {}).get("subject") or "unknown"
        verb = ((latest_parse.get("payload", {}).get("parse", {}) or {}).get("verbs") or ["relates"])[0]
        meaning = {
            "subject": subject, "relation": verb,
            "literal": action == "literal_integration",
            "lexical_support": len(lexical),
            "syntactic_support": len(syntactic),
        }
        confidence = min(0.9, 0.55 + 0.1 * (len(lexical) > 0) + 0.1 * (len(syntactic) > 0)
                         + (0.15 if discourse else 0.0))
        return {
            "outputs": {
                "meaning": meaning,
                "interpretation": f"{subject} {verb} a target (via {action})",
                "confidence": confidence,
            },
            "evidence_refs": _input_evidence_for_context(context),
            "uncertainty": 1.0 - confidence,
            "self_confidence": confidence,
            "self_evaluation": f"applied {action}",
            "assumptions": (f"chosen action: {action}",),
            "resource_usage": 2.0,
        }


class EvidenceVerificationAgent(BaseBoundedAgent):
    """Independently verify accepted findings and their evidence lineage (Step 42)."""

    declared_actions = ("deterministic_checks", "lineage_checks")

    def _perform(self, context: ExecutionContext) -> Mapping[str, Any]:
        """Run deterministic and lineage checks over the snapshot's accepted findings."""
        context.report_progress("evidence_verification", "running")
        text = str(context.snapshot.read("current_input") or "")
        checks: list[dict[str, Any]] = []
        verified_findings: list[str] = []
        failed: list[str] = []
        for hypothesis_field in ("lexical_hypotheses", "syntactic_hypotheses", "semantic_hypotheses"):
            value = context.snapshot.read(hypothesis_field)
            items: list[Any]
            if isinstance(value, Mapping):
                items = [record for records in value.values() if isinstance(records, list) for record in records]
            elif isinstance(value, list):
                items = value
            else:
                items = []
            for record in items:
                if not isinstance(record, Mapping):
                    continue
                finding_id = str(record.get("finding_id", ""))
                provenance = record.get("provenance") or {}
                passed = bool(finding_id) and bool(provenance.get("source"))
                checks.append({
                    "finding_id": finding_id, "field": field,
                    "check": "provenance_present", "passed": passed,
                })
                if passed:
                    verified_findings.append(finding_id)
                else:
                    failed.append(finding_id)
        input_ok = bool(text.strip())
        checks.append({"finding_id": "current_input", "field": "current_input",
                       "check": "non_empty_input", "passed": input_ok})
        return {
            "outputs": {
                "verified": input_ok and not failed,
                "checks": checks,
                "findings": verified_findings,
            },
            "evidence_refs": _input_evidence_for_context(context),
            "uncertainty": 0.05,
            "self_confidence": 0.9,
            "self_evaluation": f"verified {len(verified_findings)} findings with {len(failed)} failures",
            "resource_usage": 1.0,
        }


class CriticAgent(BaseBoundedAgent):
    """Produce structured critic feedback for debate rounds (Step 40)."""

    declared_actions = ("evidence_critique", "consistency_critique")

    def critique(
        self,
        *,
        claim: str,
        proposal: Mapping[str, Any],
        counterpart: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Identify the exact claim, violated rule, missing evidence, or counterexample."""
        missing_evidence: list[str] = []
        if "confidence" not in proposal:
            missing_evidence.append("proposal does not report confidence")
        evidence = proposal.get("evidence_refs") or proposal.get("evidence") or ()
        if not evidence:
            missing_evidence.append("proposal cites no evidence artifacts")
        counterexample = None
        if counterpart is not None:
            counterpart_interpretation = counterpart.get("interpretation")
            if counterpart_interpretation and counterpart_interpretation != proposal.get("interpretation"):
                counterexample = {
                    "competing_interpretation": counterpart_interpretation,
                    "note": "an alternative interpretation with its own evidence exists",
                }
        violated_rule = "evidence_required_for_authoritative_integration" if missing_evidence else "none"
        return {
            "claim": claim,
            "violated_rule": violated_rule,
            "missing_evidence": missing_evidence,
            "counterexample": counterexample,
            "failed_test": None,
        }

    def _perform(self, context: ExecutionContext) -> Mapping[str, Any]:
        """Critique the semantic proposals currently visible in the scoped snapshot."""
        context.report_progress("critique", "running")
        proposals = context.snapshot.read("semantic_hypotheses") or []
        critiques = []
        if isinstance(proposals, list):
            for record in proposals:
                if isinstance(record, Mapping):
                    payload = record.get("payload", record)
                    critiques.append(self.critique(
                        claim=str(payload.get("interpretation", "unnamed interpretation")),
                        proposal=payload,
                    ))
        return {
            "outputs": {"critiques": critiques, "critique_count": len(critiques)},
            "evidence_refs": _input_evidence_for_context(context),
            "uncertainty": 0.1,
            "self_confidence": 0.8,
            "self_evaluation": f"critiqued {len(critiques)} proposals",
            "resource_usage": 1.0,
        }


# --------------------------------------------------------------------------- #
# Bounded dispatch executor
# --------------------------------------------------------------------------- #

class TaskExecutor:
    """Dispatch agent work with optional wall-clock timeouts (Steps 21-22, 51-52)."""

    def __init__(self, *, max_workers: int = 4, use_threads: bool = True) -> None:
        """Configure a bounded worker pool for timeout-enforced execution."""
        if max_workers < 1:
            raise ValueError("max_workers must be at least one")
        self.max_workers = max_workers
        self.use_threads = use_threads
        self._pool = ThreadPoolExecutor(max_workers=max_workers) if use_threads else None

    def dispatch(
        self,
        agent: BaseBoundedAgent,
        context: ExecutionContext,
        *,
        timeout_seconds: float | None = None,
    ) -> tuple[AgentResult | None, str | None]:
        """Execute one task and return (result, failure_reason); timeout leaves the outcome unknown."""
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        effective_timeout = (
            timeout_seconds if timeout_seconds is not None
            else agent.descriptor.resources.timeout_seconds
        )
        if self._pool is None or effective_timeout is None:
            try:
                return agent.execute(context), None
            except (TransientAgentError, PermanentAgentError) as exc:
                return None, str(exc)
        future: Future = self._pool.submit(agent.execute, context)
        try:
            return future.result(timeout=effective_timeout), None
        except FutureTimeoutError:
            # A timeout does not prove the worker did nothing; reconciliation is required.
            return None, "timeout"
        except TransientAgentError as exc:
            return None, str(exc)
        except PermanentAgentError as exc:
            return None, str(exc)

    def close(self) -> None:
        """Release the bounded worker pool."""
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    def __enter__(self) -> TaskExecutor:
        """Return this executor for context-managed lifecycle handling."""
        return self

    def __exit__(self, exc_type: object, exc: BaseException | None, traceback: object) -> bool:
        """Shut down the worker pool without suppressing body exceptions."""
        self.close()
        return False


def default_specialist_descriptors() -> tuple[AgentDescriptor, ...]:
    """Return the default small role set where autonomy adds value (Section 11)."""
    return (
        AgentDescriptor(
            agent_id="agent.lexical.alpha", agent_version="1.0.0",
            role="lexical_analyst", autonomy=AutonomyLevel.BOUNDED_AUTONOMOUS_AGENT,
            capability_ids=("macl.capability.lexical_analysis",),
            accepted_input_schema=SchemaSpec({"current_input": "string"}),
            produced_output_schema=SchemaSpec({"token_hypotheses": "dict", "coverage": "number"}),
            permitted_workspace_fields=("current_input", "lexical_hypotheses"),
            writable_workspace_fields=("lexical_hypotheses",),
            autonomy_justification="lexical candidate selection benefits from adaptive action choice",
        ),
        AgentDescriptor(
            agent_id="agent.syntactic.alpha", agent_version="1.0.0",
            role="syntactic_analyst", autonomy=AutonomyLevel.BOUNDED_AUTONOMOUS_AGENT,
            capability_ids=("macl.capability.syntactic_analysis",),
            accepted_input_schema=SchemaSpec({"current_input": "string"}),
            produced_output_schema=SchemaSpec({"parse": "dict", "interpretation": "string", "ambiguity": "number"}),
            permitted_workspace_fields=("current_input", "lexical_hypotheses", "syntactic_hypotheses"),
            writable_workspace_fields=("syntactic_hypotheses",),
            autonomy_justification="parse strategy selection and clarification requests need bounded autonomy",
        ),
        AgentDescriptor(
            agent_id="agent.semantic.alpha", agent_version="1.0.0",
            role="semantic_interpreter", autonomy=AutonomyLevel.BOUNDED_AUTONOMOUS_AGENT,
            capability_ids=("macl.capability.semantic_interpretation",),
            accepted_input_schema=SchemaSpec({"current_input": "string"}),
            produced_output_schema=SchemaSpec({"meaning": "dict", "interpretation": "string", "confidence": "number"}),
            permitted_workspace_fields=(
                "current_input", "lexical_hypotheses", "syntactic_hypotheses",
                "semantic_hypotheses", "discourse_context",
            ),
            writable_workspace_fields=("semantic_hypotheses",),
            autonomy_justification="literal versus contextual integration is an adaptive evidence choice",
        ),
        AgentDescriptor(
            agent_id="agent.verifier.alpha", agent_version="1.0.0",
            role="evidence_verifier", autonomy=AutonomyLevel.TOOL_USING_WORKER,
            capability_ids=("macl.capability.evidence_verification",),
            accepted_input_schema=SchemaSpec({"current_input": "string"}),
            produced_output_schema=SchemaSpec({"verified": "boolean", "checks": "list", "findings": "list"}),
            permitted_workspace_fields=(
                "current_input", "lexical_hypotheses", "syntactic_hypotheses",
                "semantic_hypotheses", "artifact_references",
            ),
            writable_workspace_fields=(),
            tools=("deterministic_checks", "lineage_checks"),
        ),
        AgentDescriptor(
            agent_id="agent.critic.alpha", agent_version="1.0.0",
            role="critic", autonomy=AutonomyLevel.TOOL_USING_WORKER,
            capability_ids=("macl.capability.critic_review",),
            accepted_input_schema=SchemaSpec({"current_input": "string"}),
            produced_output_schema=SchemaSpec({"critiques": "list", "critique_count": "integer"}),
            permitted_workspace_fields=("current_input", "semantic_hypotheses"),
            writable_workspace_fields=(),
            tools=("evidence_critique", "consistency_critique"),
        ),
    )


def build_default_agents(
    *,
    clock: Any = None,
    adapter: WorkspaceArtifactAdapter | None = None,
) -> dict[str, BaseBoundedAgent]:
    """Instantiate the default specialist role set keyed by agent ID (Section 11)."""
    agents: dict[str, BaseBoundedAgent] = {}
    for descriptor in default_specialist_descriptors():
        if descriptor.role == "lexical_analyst":
            agents[descriptor.agent_id] = LexicalAnalysisAgent(descriptor, clock=clock)
        elif descriptor.role == "syntactic_analyst":
            agents[descriptor.agent_id] = SyntacticAnalysisAgent(descriptor, clock=clock)
        elif descriptor.role == "semantic_interpreter":
            agents[descriptor.agent_id] = SemanticInterpretationAgent(descriptor, clock=clock)
        elif descriptor.role == "evidence_verifier":
            agents[descriptor.agent_id] = EvidenceVerificationAgent(descriptor, clock=clock)
        elif descriptor.role == "critic":
            agents[descriptor.agent_id] = CriticAgent(descriptor, clock=clock)
    return agents
