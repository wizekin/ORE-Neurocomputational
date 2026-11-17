"""Point 7: long-term memory storage, retrieval, and decision use.

The functional track uses an explicit SQLite-backed store (in-memory by default).
The optional LIF track is a small biological-reference model, not a fitted human
memory circuit or a claim that neurons execute these software operations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
import math
import sqlite3
import threading
import uuid
from typing import Any, Mapping, Sequence

import numpy as np


class Provenance(str, Enum):
    """Source status kept independently from confidence."""

    EXPLICIT = "explicit"
    ENTAILED = "entailed"
    INFERRED = "inferred"
    REPORTED = "reported"
    HYPOTHETICAL = "hypothetical"
    PRESUPPOSED = "presupposed"
    UNRESOLVED = "unresolved"


POINT7_STEP_SPECS: tuple[dict[str, Any], ...] = (
    {"operation_id": "LC-P07-01", "name": "Receive the maintained proposition", "objective": "Accept Point 6 proposition and situation state without repeating earlier language processing.", "mathematical_operations": ("x=validate_schema(point6_handoff)", "preserve(source,confidence,scope)"), "biological_reference": "Distributed language and working-memory populations provide the active representation; the handoff is a software abstraction."},
    {"operation_id": "LC-P07-02", "name": "Validate content, provenance and unresolved fields", "objective": "Separate explicit, entailed, inferred, reported, hypothetical and unresolved components and detect task-critical gaps.", "mathematical_operations": ("c_i in [0,1]", "valid_i=content_i and provenance_i and scope_i", "abstain if critical unresolved fields remain"), "biological_reference": "Source monitoring and uncertainty depend on interacting memory and control systems; explicit labels are computational metadata."},
    {"operation_id": "LC-P07-03", "name": "Represent the current task goal", "objective": "Determine whether the proposition is needed for conversation, learning, later recall, question answering, judgment or action.", "mathematical_operations": ("g=encode(task_goal)", "r_i=similarity(component_i,g)", "horizon in {immediate,prospective,long_term}"), "biological_reference": "Prefrontal systems help maintain task goals and bias memory or response selection."},
    {"operation_id": "LC-P07-04", "name": "Estimate durable-retention priority", "objective": "Estimate novelty, relevance, distinctiveness, intentional learning and salience without treating emotion or attention as proof of accurate memory.", "mathematical_operations": ("E=w_n*N+w_r*R+w_d*D+w_g*G+w_s*S", "E in [0,1]"), "biological_reference": "Attention, novelty and motivational salience modulate medial-temporal encoding; weights are model parameters."},
    {"operation_id": "LC-P07-05", "name": "Select transient, episodic, semantic and immediate-use routes", "objective": "Allow one proposition to support more than one route while keeping episodic experience distinct from general knowledge.", "mathematical_operations": ("P(route|E,goal,content,knowledge)", "route_set subset {transient,episodic,semantic,decision}"), "biological_reference": "Episodic and semantic memory interact but are not identical; immediate use need not imply durable storage."},
    {"operation_id": "LC-P07-06", "name": "Extract the central event and its participants", "objective": "Identify event type, predicate, participants, semantic roles, object, outcome and unresolved event details.", "mathematical_operations": ("event_core=(predicate,{role:entity},outcome)", "role_binding=phi(role,participant)"), "biological_reference": "Distributed cortical event representations supply content to medial-temporal relational memory circuits."},
    {"operation_id": "LC-P07-07", "name": "Bind event content to its context", "objective": "Associate event content with available speaker, time, place, situation and discourse context without inventing missing details.", "mathematical_operations": ("m=Bind(event,context,relations)", "m≈event tensor_product context"), "biological_reference": "Hippocampal and entorhinal pathways support relational binding; the tensor expression is only a computational analogy."},
    {"operation_id": "LC-P07-08", "name": "Separate item and context information while preserving links", "objective": "Keep item-related and contextual features distinguishable but jointly retrievable.", "mathematical_operations": ("M_episode=f_bind(I,C,R)", "retrieve(I|C) and retrieve(C|I)"), "biological_reference": "Perirhinal, parahippocampal, entorhinal and hippocampal contributions are differentiated but interacting."},
    {"operation_id": "LC-P07-09", "name": "Differentiate similar episodes", "objective": "Reduce confusion among overlapping events using distinctive time, source, participant, location and outcome cues.", "mathematical_operations": ("s_ij=similarity(m_i,m_j)", "pattern_separation: d(f(x),f(y)) increases for overlapping inputs"), "biological_reference": "Hippocampal pattern-separation models are influential; this implementation uses transparent feature overlap, not a literal dentate-gyrus simulation."},
    {"operation_id": "LC-P07-10", "name": "Encode event and relational structure", "objective": "Preserve who did what to whom and temporal, causal and discourse relations rather than storing isolated labels.", "mathematical_operations": ("G=(V,E,lambda_V,lambda_E)", "edge=(source,relation,target)"), "biological_reference": "Hippocampal circuitry binds distributed relational content; no literal symbolic graph is asserted."},
    {"operation_id": "LC-P07-11", "name": "Initiate activity-dependent memory encoding", "objective": "Create an initially labile trace that can be reactivated and modified, with encoding quality dependent on attention, novelty and interference.", "mathematical_operations": ("delta_w_ij=eta*x_i*x_j", "optional STDP(t_pre-t_post)"), "biological_reference": "Synaptic plasticity and medial-temporal network interactions support learning; Hebbian learning is a simplified analogy."},
    {"operation_id": "LC-P07-12", "name": "Preserve source and confidence metadata", "objective": "Keep heard, read, witnessed, reported and inferred information distinct and avoid converting familiarity into truth.", "mathematical_operations": ("trace_i=(content_i,source_i,confidence_i)", "confidence != truth"), "biological_reference": "Source monitoring recruits distributed memory and control processes; the explicit record is an engineering safeguard."},
    {"operation_id": "LC-P07-13", "name": "Detect generalizable semantic knowledge candidates", "objective": "Identify potentially general facts while preserving negation, modality, hypothetical status, attribution and temporal scope.", "mathematical_operations": ("K_candidate=generalize(P)", "eligible=asserted and scope_preserved and provenance_supported"), "biological_reference": "Neocortical association and medial-prefrontal/hippocampal interactions contribute to knowledge integration; one example is not automatically a rule."},
    {"operation_id": "LC-P07-14", "name": "Compare new knowledge with existing knowledge", "objective": "Classify new content as confirming, extending, contradicting, context-dependent or unrelated.", "mathematical_operations": ("match=similarity(K_new,K_old)", "consistency=constraint_check(K_new,K_old)"), "biological_reference": "Memory-guided comparison involves distributed cortical and hippocampal systems; similarity alone does not establish compatibility."},
    {"operation_id": "LC-P07-15", "name": "Integrate knowledge without blindly overwriting", "objective": "Preserve exceptions, valid-time intervals, source reliability and competing claims; leave unverified claims pending.", "mathematical_operations": ("P(H|D)=P(D|H)P(H)/P(D)", "update only if evidence and scope criteria pass"), "biological_reference": "Schema-related and hippocampal-prefrontal interactions may support integration; Bayesian updating is normative, not a literal neural claim."},
    {"operation_id": "LC-P07-16", "name": "Model consolidation and reorganization over time", "objective": "Represent reactivation, rehearsal, interference and gradual integration without promising permanent or verbatim storage.", "mathematical_operations": ("S_(t+1)=S_t+alpha*R_t-beta*I_t", "consolidation_status in {labile,reactivated,stabilizing}"), "biological_reference": "Sleep, replay and systems consolidation influence memory in task-dependent ways; theories remain debated."},
    {"operation_id": "LC-P07-17", "name": "Translate a later question into retrieval cues", "objective": "Identify target entities, events, relations, time and whether episodic or semantic information is required.", "mathematical_operations": ("q=encode(question,goal)", "a_j=lambda1*sim(q,m_j)+lambda2*context_j+lambda3*goal_j"), "biological_reference": "Prefrontal retrieval goals bias access to hippocampal and cortical memory representations."},
    {"operation_id": "LC-P07-18", "name": "Reactivate candidate memory traces", "objective": "Activate a limited set of relevant episodic or semantic candidates from the cue and task context.", "mathematical_operations": ("activation_j=weighted_cue_match(q,m_j)", "rank(candidates,activation)"), "biological_reference": "Hippocampal-cortical interactions support retrieval; candidate ranking is a computational abstraction."},
    {"operation_id": "LC-P07-19", "name": "Complete partial memories while marking reconstruction", "objective": "Use linked event/context information to recover details but distinguish retrieved fields from inferred or guessed completions.", "mathematical_operations": ("m_star=F(q,W)", "field_status in {stored,reconstructed,inferred,unknown}"), "biological_reference": "Pattern completion is a model of associative reinstatement, not a guarantee of accurate recall."},
    {"operation_id": "LC-P07-20", "name": "Resolve competition among retrieved memories", "objective": "Compare identity, role, time, place and source, and abstain when two memories remain similarly plausible.", "mathematical_operations": ("P(m_j|q)=softmax(a_j/tau)", "margin=a_best-a_second"), "biological_reference": "Prefrontal control and medial-temporal retrieval interact; retrieval fluency alone is not accuracy."},
    {"operation_id": "LC-P07-21", "name": "Parse the immediate answer or judgment request", "objective": "Map who/what/when/where/why/how, truth, coherence and acceptability questions to their required evidence.", "mathematical_operations": ("schema=g(task_type,proposition)", "target_role=map(question_type,semantic_role)"), "biological_reference": "Language representations interact with goal maintenance and controlled retrieval during question answering."},
    {"operation_id": "LC-P07-22", "name": "Generate and evaluate candidate answers", "objective": "Compare explicit evidence, contextual support, retrieved knowledge, missing premises and unsupported assumptions.", "mathematical_operations": ("Score(a)=w_e*Evidence+w_c*Context+w_g*Goal-w_u*Assumptions", "P(a|evidence,context)"), "biological_reference": "Lateral prefrontal and parietal control may support comparison; scores are model quantities, not measured neural values."},
    {"operation_id": "LC-P07-23", "name": "Monitor conflict and calibrate the answer", "objective": "Use candidate separation and evidence quality to select, qualify, defer or request clarification.", "mathematical_operations": ("M=S_best-S_second", "answer only if M>=theta and evidence is sufficient", "C_cal=f(score,observed_accuracy)"), "biological_reference": "Anterior cingulate and lateral prefrontal systems contribute to conflict/control in some tasks; no universal threshold is claimed."},
    {"operation_id": "LC-P07-24", "name": "Generate feasible actions and predict outcomes", "objective": "Create constrained options, include wait/information-gathering choices, and estimate possible consequences.", "mathematical_operations": ("P(outcome|action,state,knowledge)", "A_feasible={a:constraints(a)=true}"), "biological_reference": "Prefrontal and orbitofrontal systems contribute to goal/outcome evaluation; basal-ganglia circuits participate in action gating."},
    {"operation_id": "LC-P07-25", "name": "Evaluate options and accumulate decision evidence", "objective": "Compare benefits, costs, risks, delay and reversibility while preserving multiple objectives where a scalar value is inadequate.", "mathematical_operations": ("EU(a)=sum_o P(o|a,s,K)*U(o)-Cost(a)", "X_(t+1)=X_t+delta_t+epsilon_t"), "biological_reference": "Evidence accumulation is useful for some decisions; complex social or long-horizon choices need not use one accumulator."},
    {"operation_id": "LC-P07-26", "name": "Apply a task-sensitive decision threshold", "objective": "Require stronger support for irreversible or high-consequence actions and account for the cost of delay.", "mathematical_operations": ("select if X_t>=theta(task_risk,delay_cost)", "theta increases with error cost"), "biological_reference": "Speed-accuracy settings vary with task and consequences; a software threshold is not a fixed neural constant."},
    {"operation_id": "LC-P07-27", "name": "Select and prepare a response or action plan", "objective": "Convert a sufficiently supported answer or option into a response plan without confusing selection with execution.", "mathematical_operations": ("a_star=argmax_a Score(a)", "response_plan=compile(a_star,output_constraints)"), "biological_reference": "Prefrontal response selection interacts with language production and basal-ganglia/motor systems when action is required."},
    {"operation_id": "LC-P07-28", "name": "Monitor outcomes and update memory/task state", "objective": "Compare actual outcomes with predictions, detect discrepancies and preserve the decision context for future learning.", "mathematical_operations": ("prediction_error=outcome-predicted_outcome", "state_(t+1)=update(state_t,error)"), "biological_reference": "Prediction-error learning is implemented differently across systems; a single scalar is not a universal brain signal."},
)


def _json(value: Any) -> str:
    """Serialize model records deterministically while preserving Unicode."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _tokens(value: Any) -> set[str]:
    """Return a compact case-folded token set for transparent cue matching."""
    if value is None:
        return set()
    if isinstance(value, Mapping):
        value = " ".join(str(k) + " " + str(v) for k, v in value.items())
    elif isinstance(value, (list, tuple, set)):
        value = " ".join(str(x) for x in value)
    return {part.casefold().strip(".,!?;:()[]{}\"'") for part in str(value).split() if part.strip(".,!?;:()[]{}\"'")}


def _similarity(left: Any, right: Any) -> float:
    """Jaccard similarity provides an inspectable baseline, not a semantic embedding."""
    a, b = _tokens(left), _tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _confidence(value: Any, default: float = 0.5) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) and 0.0 <= number <= 1.0 else default


@dataclass(frozen=True, slots=True)
class Point7Request:
    """Input to memory/decision processing; no previous stage is rerun."""

    sequence_id: str
    proposition: Mapping[str, Any] = field(default_factory=dict)
    components: Sequence[Mapping[str, Any]] = ()
    context: Mapping[str, Any] = field(default_factory=dict)
    task: Mapping[str, Any] = field(default_factory=dict)
    query: str | None = None
    candidate_answers: Sequence[Any] = ()
    feasible_options: Sequence[Mapping[str, Any]] = ()
    outcome_feedback: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.sequence_id, str) or not self.sequence_id.strip():
            raise ValueError("sequence_id must be non-empty")
        for name in ("proposition", "context", "task", "outcome_feedback"):
            if not isinstance(getattr(self, name), Mapping):
                raise TypeError(f"{name} must be a mapping")
        object.__setattr__(self, "components", tuple(dict(x) for x in self.components))
        object.__setattr__(self, "candidate_answers", tuple(self.candidate_answers))
        object.__setattr__(self, "feasible_options", tuple(dict(x) for x in self.feasible_options))

    @classmethod
    def from_point6_result(cls, sequence_id: str, point6_result: Any, **overrides: Any) -> "Point7Request":
        """Create a Point 7 request from a Point 6 result, preserving its handoff."""
        payload = point6_result if isinstance(point6_result, Mapping) else point6_result.to_dict()
        handoff = payload.get("handoff", {})
        if not isinstance(handoff, Mapping):
            handoff = {}
        proposition = {
            "active_propositions": handoff.get("active_propositions", []),
            "situation_model": handoff.get("situation_model", {}),
            "discourse_relations": handoff.get("discourse_relations", []),
            "unresolved_alternatives": handoff.get("unresolved_alternatives", []),
            "source_provenance": handoff.get("source_provenance", {}),
            "confidence": handoff.get("confidence", payload.get("diagnostics", {}).get("confidence", 0.5)),
        }
        proposition.update(overrides.pop("proposition", {}))
        context = dict(overrides.pop("context", {}))
        context.setdefault("point6_handoff", handoff)
        context.setdefault("situation_model", handoff.get("situation_model", {}))
        context.setdefault("working_memory_state", handoff.get("working_memory_state", {}))
        components = list(overrides.pop("components", ()))
        if not components:
            active = handoff.get("active_propositions", [])
            if isinstance(active, Sequence) and not isinstance(active, (str, bytes)):
                for index, item in enumerate(active):
                    if isinstance(item, Mapping):
                        content = item.get("text", item.get("predicate", item))
                        components.append({
                            "component_id": f"point6-literal-{index + 1}",
                            "content": content,
                            "provenance": "explicit",
                            "confidence": _confidence(item.get("confidence", 0.5)),
                        })
            alternatives = handoff.get("unresolved_alternatives", [])
            if isinstance(alternatives, Sequence) and not isinstance(alternatives, (str, bytes)):
                for index, item in enumerate(alternatives):
                    if isinstance(item, Mapping):
                        components.append({
                            "component_id": f"point6-unresolved-{index + 1}",
                            "content": item.get("interpretation", item.get("label", item)),
                            "provenance": "unresolved",
                            "confidence": _confidence(item.get("prior", item.get("confidence", 0.5))),
                        })
        return cls(sequence_id=sequence_id, proposition=proposition, components=components, context=context, **overrides)


class Point7MemoryStore:
    """Separate episodic and semantic SQLite stores; defaults to process-local memory."""

    def __init__(self, database_path: str = ":memory:") -> None:
        self.database_path = database_path
        self._lock = threading.RLock()
        self._db = sqlite3.connect(database_path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("""CREATE TABLE IF NOT EXISTS episodic_traces (
            created_order INTEGER PRIMARY KEY AUTOINCREMENT, trace_id TEXT UNIQUE NOT NULL,
            sequence_id TEXT NOT NULL, payload TEXT NOT NULL
        )""")
        self._db.execute("""CREATE TABLE IF NOT EXISTS semantic_knowledge (
            created_order INTEGER PRIMARY KEY AUTOINCREMENT, knowledge_id TEXT UNIQUE NOT NULL,
            claim_key TEXT NOT NULL, payload TEXT NOT NULL, status TEXT NOT NULL
        )""")
        self._db.execute("""CREATE TABLE IF NOT EXISTS outcome_feedback (
            created_order INTEGER PRIMARY KEY AUTOINCREMENT, feedback_id TEXT UNIQUE NOT NULL,
            sequence_id TEXT NOT NULL, payload TEXT NOT NULL
        )""")
        self._db.commit()

    def add_episode(self, sequence_id: str, payload: Mapping[str, Any]) -> str:
        trace_id = str(uuid.uuid4())
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO episodic_traces(trace_id,sequence_id,payload) VALUES(?,?,?)",
                (trace_id, sequence_id, _json(payload)),
            )
        return trace_id

    def add_semantic_candidate(self, payload: Mapping[str, Any], *, verified: bool = False) -> str:
        knowledge_id = str(uuid.uuid4())
        claim_key = str(payload.get("claim_key") or _json(payload.get("claim", payload)))
        status = "accepted" if verified else "pending_verification"
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO semantic_knowledge(knowledge_id,claim_key,payload,status) VALUES(?,?,?,?)",
                (knowledge_id, claim_key, _json(payload), status),
            )
        return knowledge_id

    def episodes(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM episodic_traces ORDER BY created_order").fetchall()
        return [{"trace_id": row["trace_id"], "sequence_id": row["sequence_id"], **json.loads(row["payload"])} for row in rows]

    def semantic_records(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM semantic_knowledge ORDER BY created_order").fetchall()
        return [{**json.loads(row["payload"]), "knowledge_id": row["knowledge_id"], "status": row["status"]} for row in rows]

    def update_episode(self, trace_id: str, updates: Mapping[str, Any]) -> None:
        """Update reactivation/consolidation metadata without rewriting event provenance."""
        with self._lock, self._db:
            row = self._db.execute(
                "SELECT payload FROM episodic_traces WHERE trace_id=?", (trace_id,)
            ).fetchone()
            if row is None:
                return
            payload = json.loads(row["payload"])
            payload.update(dict(updates))
            self._db.execute(
                "UPDATE episodic_traces SET payload=? WHERE trace_id=?",
                (_json(payload), trace_id),
            )

    def add_feedback(self, sequence_id: str, payload: Mapping[str, Any]) -> str:
        """Persist an outcome/prediction comparison separately from semantic facts."""
        feedback_id = str(uuid.uuid4())
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO outcome_feedback(feedback_id,sequence_id,payload) VALUES(?,?,?)",
                (feedback_id, sequence_id, _json(payload)),
            )
        return feedback_id

    def feedback_events(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM outcome_feedback ORDER BY created_order").fetchall()
        return [
            {"feedback_id": row["feedback_id"], "sequence_id": row["sequence_id"], **json.loads(row["payload"])}
            for row in rows
        ]

    def close(self) -> None:
        with self._lock:
            self._db.close()


class Point7LIFReference:
    """Ten-population current-based LIF reference for memory and decision functions."""

    POPULATIONS = (
        "hippocampal_binding", "entorhinal_interface", "perirhinal_item",
        "parahippocampal_context", "neocortical_semantic", "medial_prefrontal_schema",
        "dorsolateral_prefrontal_control", "anterior_cingulate_conflict",
        "orbitofrontal_value", "basal_ganglia_action",
    )

    def __init__(self, population_size: int = 12, dt_ms: float = 1.0, seed: int = 7) -> None:
        if population_size < 2 or dt_ms <= 0:
            raise ValueError("population_size must be >= 2 and dt_ms must be positive")
        self.population_size = population_size
        self.dt_ms = dt_ms
        self.seed = seed

    def run(self, operation_count: int = 28) -> dict[str, Any]:
        """Run deterministic current-based LIF populations as a functional analogy."""
        rng = np.random.default_rng(self.seed)
        n_pop = len(self.POPULATIONS)
        n = self.population_size * n_pop
        voltage = rng.normal(-65.0, 1.0, n)
        current = np.zeros(n, dtype=np.float64)
        refractory = np.zeros(n, dtype=np.int32)
        spikes = np.zeros(n, dtype=np.int32)
        tau_m, resistance, threshold, reset = 20.0, 1.0, -50.0, -65.0
        rates: dict[str, float] = {}
        for step in range(operation_count):
            phase = step / max(1, operation_count - 1)
            for pop_index, name in enumerate(self.POPULATIONS):
                start = pop_index * self.population_size
                end = start + self.population_size
                # Input profile changes across memory encoding, retrieval and choice.
                base = 14.0 + 10.0 * (0.5 + 0.5 * math.sin(phase * math.pi + pop_index * 0.37))
                current[start:end] = base + rng.normal(0.0, 0.25, self.population_size)
            active = refractory == 0
            voltage[active] += (self.dt_ms / tau_m) * (
                -(voltage[active] + 65.0) + resistance * current[active]
            )
            refractory[refractory > 0] -= 1
            fired = (voltage >= threshold) & (refractory == 0)
            spikes += fired.astype(np.int32)
            voltage[fired] = reset
            refractory[fired] = 2
        for i, name in enumerate(self.POPULATIONS):
            block = spikes[i * self.population_size:(i + 1) * self.population_size]
            rates[name] = float(block.sum() / (self.population_size * operation_count * self.dt_ms) * 1000.0)
        return {
            "population_names": list(self.POPULATIONS),
            "population_size": self.population_size,
            "operation_count": operation_count,
            "duration_ms": operation_count * self.dt_ms,
            "firing_rates_hz": rates,
            "spike_count": int(spikes.sum()),
            "biological_reference_is_measured_human_data": False,
            "model_scope": "current-based LIF functional analogy; not anatomically fitted",
        }


@dataclass
class Point7Result:
    """Outputs from memory encoding, retrieval, and immediate decision branches."""

    sequence_id: str
    functional_trace: list[dict[str, Any]]
    routes: list[str]
    retention_priority: float
    episodic_trace: dict[str, Any] | None
    semantic_candidate: dict[str, Any] | None
    retrieved_memories: list[dict[str, Any]]
    answer: dict[str, Any] | None
    action_plan: dict[str, Any] | None
    diagnostics: dict[str, Any]
    biological_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "functional_trace": self.functional_trace,
            "routes": self.routes,
            "retention_priority": self.retention_priority,
            "episodic_trace": self.episodic_trace,
            "semantic_candidate": self.semantic_candidate,
            "retrieved_memories": self.retrieved_memories,
            "answer": self.answer,
            "action_plan": self.action_plan,
            "diagnostics": self.diagnostics,
            "biological_summary": self.biological_summary,
        }


class Point7MemoryDecisionProcessor:
    """Execute Point 7's 28 source-level operations over a separate memory store."""

    def __init__(
        self,
        store: Point7MemoryStore | None = None,
        *,
        simulate_biology: bool = False,
        population_size: int = 12,
        decision_margin: float = 0.10,
        episodic_threshold: float = 0.60,
    ) -> None:
        if not 0.0 <= decision_margin <= 1.0:
            raise ValueError("decision_margin must be in [0, 1]")
        if not 0.0 <= episodic_threshold <= 1.0:
            raise ValueError("episodic_threshold must be in [0, 1]")
        self.store = store or Point7MemoryStore()
        self.simulate_biology = simulate_biology
        self.lif = Point7LIFReference(population_size=population_size) if simulate_biology else None
        self.decision_margin = decision_margin
        self.episodic_threshold = episodic_threshold

    @staticmethod
    def _components(request: Point7Request) -> list[dict[str, Any]]:
        if request.components:
            return [dict(item) for item in request.components]
        raw = request.proposition.get("components", [])
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            return [dict(item) for item in raw if isinstance(item, Mapping)]
        return []

    @staticmethod
    def _event_core(request: Point7Request, components: list[dict[str, Any]]) -> dict[str, Any]:
        proposition = request.proposition
        roles = proposition.get("semantic_roles", proposition.get("arguments", {}))
        if not isinstance(roles, Mapping):
            roles = {"arguments": roles}
        return {
            "predicate": proposition.get("predicate", proposition.get("event_type")),
            "participants_and_roles": dict(roles),
            "object_or_topic": proposition.get("object_or_topic"),
            "outcome": proposition.get("outcome"),
            "components": components,
        }

    def process(self, request: Point7Request) -> Point7Result:
        """Run all 28 operations; branch outputs remain separate and provenance-aware."""
        if not isinstance(request, Point7Request):
            raise TypeError("request must be a Point7Request")
        trace: list[dict[str, Any]] = []
        components = self._components(request)
        proposition = dict(request.proposition)
        task = dict(request.task)
        context = dict(request.context)

        def record(index: int, output: Any) -> None:
            spec = POINT7_STEP_SPECS[index]
            trace.append({
                "step": index + 1,
                "operation_id": spec["operation_id"],
                "name": spec["name"],
                "output": output,
            })

        # Shared preparation: validate source and unresolved information.
        valid_components = []
        invalid_components = []
        for item in components:
            confidence = _confidence(item.get("confidence", 0.5))
            provenance = str(item.get("provenance", "explicit")).casefold()
            normalized = {**item, "confidence": confidence, "provenance": provenance}
            (valid_components if provenance in {p.value for p in Provenance} else invalid_components).append(normalized)
        critical_unresolved = [
            str(x) for x in proposition.get("unresolved_variables", [])
            if str(x).strip()
        ]
        record(0, {"proposition_keys": sorted(proposition), "component_count": len(components)})
        record(1, {"valid_component_count": len(valid_components), "invalid_provenance_count": len(invalid_components), "unresolved_variables": critical_unresolved})

        goal = str(task.get("goal", task.get("type", "comprehension")))
        horizon = str(task.get("horizon", "immediate"))
        record(2, {"goal": goal, "horizon": horizon})

        features = task.get("retention_features", {})
        if not isinstance(features, Mapping):
            features = {}
        novelty = _confidence(features.get("novelty", task.get("novelty", 0.0)))
        relevance = _confidence(features.get("relevance", task.get("relevance", 0.0)))
        distinctiveness = _confidence(features.get("distinctiveness", 0.0))
        intentional = _confidence(features.get("intentional_learning", task.get("intentional_learning", 0.0)))
        salience = _confidence(features.get("salience", 0.0))
        retention = min(1.0, max(0.0, 0.20 * novelty + 0.30 * relevance + 0.15 * distinctiveness + 0.25 * intentional + 0.10 * salience))
        record(3, {"retention_priority": retention, "features": {"novelty": novelty, "relevance": relevance, "distinctiveness": distinctiveness, "intentional_learning": intentional, "salience": salience}})

        routes = ["transient"]
        if task.get("encode_episode", False) or retention >= self.episodic_threshold:
            routes.append("episodic")
        if task.get("generalizable_knowledge", False):
            routes.append("semantic")
        if task.get("requires_answer", False) or request.query:
            routes.append("immediate_answer")
        if task.get("requires_action_plan", False) or request.feasible_options:
            routes.append("decision")
        record(4, {"routes": routes})

        event_core = self._event_core(request, valid_components)
        record(5, event_core)
        episodic_payload = None
        if "episodic" in routes:
            episodic_payload = {
                "event_core": event_core,
                "context": context,
                "time": context.get("time"),
                "place": context.get("place"),
                "speaker_or_source": context.get("source", proposition.get("source")),
                "source_provenance": [x["provenance"] for x in valid_components],
                "confidence_by_component": [x["confidence"] for x in valid_components],
                "encoding_status": "initial_labile_trace_not_guaranteed_permanent",
                "reactivation_count": 0,
                "trace_strength": 0.25,
                "interference": 0.0,
                "consolidation_status": "labile",
                "item_features": sorted(_tokens(event_core)),
                "context_features": sorted(_tokens(context)),
                "relations": proposition.get("relations", proposition.get("discourse_relations", [])),
            }
            record(6, {"event_context_bound": True, "context_fields": sorted(context)})
            record(7, {"item_feature_count": len(episodic_payload["item_features"]), "context_feature_count": len(episodic_payload["context_features"])})
            existing = self.store.episodes()
            similarity = max((_similarity(episodic_payload["item_features"], x.get("event_core", {})) for x in existing), default=0.0)
            episodic_payload["nearest_episode_similarity"] = similarity
            episodic_payload["differentiation_status"] = "overlap_detected_preserve_context" if similarity >= 0.5 else "distinct_candidate"
            record(8, {"nearest_episode_similarity": similarity, "differentiation_status": episodic_payload["differentiation_status"]})
            record(9, {"relational_edge_count": len(episodic_payload["relations"]) if isinstance(episodic_payload["relations"], Sequence) else 0})
            episodic_payload["plasticity_model"] = {"rule": "delta_w=eta*x_i*x_j", "status": "functional_analogy_only"}
            record(10, episodic_payload["plasticity_model"])
            record(11, {"source_and_confidence_preserved": True, "component_count": len(valid_components)})
            episodic_payload["trace_id"] = self.store.add_episode(request.sequence_id, episodic_payload)
        else:
            for index in range(6, 12):
                record(index, {"status": "not_selected_by_retention_route"})

        # Semantic route: one example becomes a pending candidate unless explicitly verified.
        semantic_candidate = None
        if "semantic" in routes:
            asserted = bool(valid_components) and all(x["provenance"] in {"explicit", "entailed", "reported"} for x in valid_components)
            semantic_candidate = {
                "claim": proposition.get("claim", proposition.get("predicate", proposition)),
                "claim_key": str(proposition.get("claim_key") or proposition.get("predicate") or _json(proposition)),
                "source": context.get("source", proposition.get("source")),
                "valid_time": context.get("valid_time"),
                "provenance": [x["provenance"] for x in valid_components],
                "confidence": min((x["confidence"] for x in valid_components), default=_confidence(proposition.get("confidence"))),
                "status": "candidate_requires_prior_knowledge_and_reliability_check",
                "generalizable": bool(task.get("generalizable_knowledge")),
            }
            record(12, {"candidate_created": True, "asserted_or_reported_only": asserted})
            existing_semantic = self.store.semantic_records()
            matches = [x for x in existing_semantic if _similarity(semantic_candidate["claim"], x.get("claim")) >= 0.5]
            relation = "UNRELATED" if not matches else ("CONTEXT_DEPENDENT" if any(x.get("valid_time") != semantic_candidate["valid_time"] for x in matches) else "REQUIRES_CONSISTENCY_CHECK")
            semantic_candidate["existing_knowledge_relation"] = relation
            record(13, {"existing_matches": len(matches), "relation": relation})
            semantic_candidate["integration_rule"] = "pending_until_verified; preserve old claims and provenance"
            semantic_candidate["knowledge_id"] = self.store.add_semantic_candidate(
                semantic_candidate, verified=bool(task.get("semantic_verified", False) and asserted)
            )
            record(14, {"status": "accepted" if task.get("semantic_verified", False) and asserted else "pending_verification"})
            semantic_candidate["consolidation_model"] = {"reactivation": 0.0, "interference": 0.0, "status": "initial_trace"}
            record(15, semantic_candidate["consolidation_model"])
        else:
            for index in range(12, 16):
                record(index, {"status": "semantic_route_not_selected"})

        # Retrieval route: use both episodic and semantic traces, but never invent recovered details.
        retrieved: list[dict[str, Any]] = []
        if request.query:
            cue = _tokens(request.query)
            ranked = []
            for item in self.store.episodes():
                score = _similarity(cue, item.get("event_core", {})) * 0.55 + _similarity(cue, item.get("context", {})) * 0.30 + (0.15 if item.get("sequence_id") == request.sequence_id else 0.0)
                ranked.append((score, "episodic", item))
            for item in self.store.semantic_records():
                score = _similarity(cue, item.get("claim", {}))
                ranked.append((score, "semantic", item))
            ranked.sort(key=lambda x: x[0], reverse=True)
            retrieved = [{"score": score, "memory_type": kind, **item, "field_status": "stored_or_retrieved_not_reconstructed"} for score, kind, item in ranked[:int(task.get("retrieval_limit", 5))] if score > 0]
            record(16, {"query_cues": sorted(cue), "retrieval_target": task.get("retrieval_target", "episodic_or_semantic")})
            record(17, {"candidate_count": len(retrieved), "scores": [x["score"] for x in retrieved]})
            for item in retrieved:
                item["completion_status"] = "no_missing_fields_invented"
                if item.get("memory_type") == "episodic" and item.get("trace_id"):
                    reactivations = int(item.get("reactivation_count", 0)) + 1
                    strength = min(
                        1.0,
                        max(0.0, float(item.get("trace_strength", 0.25)) + 0.05
                            - 0.01 * float(item.get("interference", 0.0))),
                    )
                    consolidation_status = "reactivated" if reactivations >= 1 else "labile"
                    self.store.update_episode(item["trace_id"], {
                        "reactivation_count": reactivations,
                        "trace_strength": strength,
                        "consolidation_status": consolidation_status,
                    })
                    item.update({
                        "reactivation_count": reactivations,
                        "trace_strength": strength,
                        "consolidation_status": consolidation_status,
                    })
            record(18, {"pattern_completion": "stored links only; episodic retrieval increments reactivation metadata", "candidate_count": len(retrieved)})
            best = retrieved[0]["score"] if retrieved else 0.0
            second = retrieved[1]["score"] if len(retrieved) > 1 else 0.0
            retrieved_ambiguous = bool(len(retrieved) > 1 and best - second < self.decision_margin)
            record(19, {"best_score": best, "second_score": second, "margin": best - second, "abstained": retrieved_ambiguous})
        else:
            for index in range(16, 20):
                record(index, {"status": "no_retrieval_query"})

        # Answering route: extract semantic roles or compare caller-supplied candidates.
        answer = None
        if "immediate_answer" in routes:
            query = request.query or str(task.get("question", ""))
            q_tokens = _tokens(query)
            role_map = {"who": ("agent", "subject", "experiencer"), "whom": ("patient", "recipient", "object"), "what": ("patient", "theme", "object", "topic"), "where": ("location", "place"), "when": ("time", "temporal")}
            question_type = next((key for key in role_map if key in q_tokens), "lookup")
            roles = proposition.get("semantic_roles", proposition.get("arguments", {}))
            if not isinstance(roles, Mapping):
                roles = {}
            target_roles = role_map.get(question_type, ())
            role_hits = [(key, value) for key, value in roles.items() if str(key).casefold() in target_roles]
            if role_hits:
                candidates = [{"answer": value, "source": "explicit_proposition_role", "provenance": "explicit", "score": 1.0} for _, value in role_hits]
            else:
                candidates = [{"answer": x, "source": "caller_candidate", "provenance": "candidate", "score": _similarity(query, x)} for x in request.candidate_answers]
            record(20, {"question_type": question_type, "target_roles": list(target_roles), "candidate_count": len(candidates)})
            for candidate in candidates:
                candidate["evidence_score"] = candidate["score"]
                candidate["context_score"] = _similarity(candidate["answer"], context)
                candidate["unsupported_assumptions"] = 0.0 if candidate["source"] == "explicit_proposition_role" else 1.0 - candidate["score"]
                candidate["total_score"] = candidate["evidence_score"] + 0.2 * candidate["context_score"] - 0.5 * candidate["unsupported_assumptions"]
            candidates.sort(key=lambda x: x["total_score"], reverse=True)
            best = candidates[0] if candidates else None
            second_score = candidates[1]["total_score"] if len(candidates) > 1 else 0.0
            margin = (best["total_score"] - second_score) if best else 0.0
            enough = bool(best and best["evidence_score"] >= float(task.get("minimum_answer_evidence", 0.5)) and (len(candidates) == 1 or margin >= self.decision_margin))
            record(21, {"candidate_scores": [x["total_score"] for x in candidates], "margin": margin})
            record(22, {"conflict": bool(len(candidates) > 1 and margin < self.decision_margin), "calibration": "uncalibrated_model_score_not_empirical_probability"})
            answer = {
                "value": best["answer"] if enough else None,
                "status": "selected" if enough else "insufficient_or_competing_evidence",
                "confidence": min(1.0, max(0.0, best["total_score"] / 1.2)) if enough else 0.0,
                "provenance": best["provenance"] if enough else "unresolved",
                "candidates": candidates,
            }
        else:
            record(20, {"status": "answer_route_not_selected"})
            record(21, {"status": "answer_route_not_selected"})
            record(22, {"status": "answer_route_not_selected"})

        # Action planning route: explicit feasibility and transparent utility/cost scoring.
        action_plan = None
        if "decision" in routes:
            feasible = [
                dict(option) for option in request.feasible_options
                if option.get("feasible", True) and option.get("hard_constraints_satisfied", True)
            ]
            scored = []
            for option in feasible:
                utility = float(option.get("expected_utility", option.get("utility", 0.0)))
                cost = max(0.0, float(option.get("cost", 0.0)))
                risk = max(0.0, float(option.get("risk", 0.0)))
                score = utility - cost - float(task.get("risk_weight", 1.0)) * risk
                scored.append({**option, "decision_score": score})
            scored.sort(key=lambda x: x["decision_score"], reverse=True)
            record(23, {"feasible_option_count": len(feasible), "outcome_model": "caller-supplied expected outcomes; no unsupported forecast invented"})
            record(24, {"option_scores": [x["decision_score"] for x in scored], "evidence_accumulation": "one-pass supplied evidence"})
            if scored:
                margin = scored[0]["decision_score"] - (scored[1]["decision_score"] if len(scored) > 1 else 0.0)
                threshold = float(task.get("minimum_decision_margin", self.decision_margin))
                high_stakes = bool(task.get("high_consequence", False))
                selected = scored[0] if margin >= threshold and (not high_stakes or scored[0].get("evidence_sufficient", False)) else None
                action_plan = {
                    "selected_option": selected,
                    "status": "selected" if selected else "defer_or_gather_more_information",
                    "margin": margin,
                    "high_consequence_threshold_applied": high_stakes,
                    "candidates": scored,
                }
            else:
                action_plan = {"selected_option": None, "status": "no_feasible_option", "candidates": []}
            record(25, {"threshold": task.get("minimum_decision_margin", self.decision_margin), "high_consequence": bool(task.get("high_consequence", False))})
            record(26, action_plan)
        else:
            for index in range(23, 27):
                record(index, {"status": "decision_route_not_selected"})

        feedback = dict(request.outcome_feedback)
        prediction_error = None
        if feedback:
            predicted = feedback.get("predicted_outcome")
            observed = feedback.get("observed_outcome")
            if isinstance(predicted, (int, float)) and isinstance(observed, (int, float)):
                prediction_error = float(observed - predicted)
        feedback_id = self.store.add_feedback(request.sequence_id, {
            **feedback,
            "prediction_error": prediction_error,
            "learning_policy": "separate_feedback_record; no automatic semantic fact rewrite",
        }) if feedback else None
        record(27, {"feedback_received": bool(feedback), "feedback_id": feedback_id, "prediction_error": prediction_error, "memory_update": "feedback persisted separately; no automatic semantic fact rewrite"})

        biological = self.lif.run(len(POINT7_STEP_SPECS)) if self.lif else {}
        diagnostics = {
            "valid": len(trace) == len(POINT7_STEP_SPECS) and len(components) == len(valid_components) + len(invalid_components),
            "step_count": len(trace),
            "operation_count": len(POINT7_STEP_SPECS),
            "episodic_memory_count": len(self.store.episodes()),
            "semantic_memory_count": len(self.store.semantic_records()),
            "outcome_feedback_count": len(self.store.feedback_events()),
            "durable_storage_enabled": self.store.database_path != ":memory:",
            "durable_storage_written": bool((episodic_payload or semantic_candidate) and self.store.database_path != ":memory:"),
            "memory_store_write": bool(episodic_payload or semantic_candidate),
            "temporary_working_memory_is_not_long_term_memory": True,
            "biological_reference_is_measured_human_data": False,
            "warnings": [
                "Similarity uses transparent token overlap, not a trained semantic model.",
                "Semantic candidates remain pending unless explicitly verified.",
                "The LIF populations are functional analogies, not anatomically fitted human data.",
            ],
        }
        return Point7Result(
            sequence_id=request.sequence_id,
            functional_trace=trace,
            routes=routes,
            retention_priority=retention,
            episodic_trace=episodic_payload,
            semantic_candidate=semantic_candidate,
            retrieved_memories=retrieved,
            answer=answer,
            action_plan=action_plan,
            diagnostics=diagnostics,
            biological_summary=biological,
        )


# Concise aliases make the stage convenient to compose with the preceding pipeline.
Point7Processor = Point7MemoryDecisionProcessor
