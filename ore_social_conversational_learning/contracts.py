"""Typed state and event contracts for Point 14 social and conversational learning.

Partner beliefs and intentions are probabilistic estimates, not direct access to another person's
mind. Social approval, factual evidence, outcomes, and source reliability are kept as distinct data.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Mapping


def _finite(name: str, value: float) -> float:
    """Convert a numeric input to a finite float."""
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(name + " must be finite")
    return result


def _unit(name: str, value: float) -> float:
    """Validate a finite model probability or normalized strength in [0, 1]."""
    result = _finite(name, value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(name + " must be in [0, 1]")
    return result


def _text(name: str, value: str, allow_empty: bool = False) -> str:
    """Normalize text and reject empty identifiers unless explicitly permitted."""
    if not isinstance(value, str):
        raise ValueError(name + " must be a string")
    result = value.strip()
    if not allow_empty and not result:
        raise ValueError(name + " must be non-empty")
    return result


def _plain(value: Any) -> Any:
    """Convert dataclasses and nested containers into JSON-compatible structures."""
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _plain(value.to_dict())
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, set):
        return sorted((_plain(item) for item in value), key=repr)
    if hasattr(value, "item") and callable(value.item):
        return _plain(value.item())
    return value


@dataclass(frozen=True, slots=True)
class SocialSignal:
    """Represent one observed speech, gaze, gesture, prosodic, facial, or action cue."""

    signal_id: str
    modality: str
    content: str = ""
    reliability: float = 0.7
    salience: float = 0.5
    directed_to_learner: float = 0.5
    target_id: str = ""
    timestamp: float = 0.0
    attributes: Mapping[str, Any] = field(default_factory=dict)
    provenance: str = "observed_signal"

    def __post_init__(self) -> None:
        """Validate signal metadata while retaining modality and provenance."""
        _text("signal_id", self.signal_id)
        allowed = {"speech", "gaze", "gesture", "prosody", "face", "action", "timing", "other"}
        if self.modality not in allowed:
            raise ValueError("unsupported social-signal modality: " + self.modality)
        for name in ("reliability", "salience", "directed_to_learner"):
            object.__setattr__(self, name, _unit(name, getattr(self, name)))
        object.__setattr__(self, "timestamp", _finite("timestamp", self.timestamp))
        object.__setattr__(self, "content", _text("content", self.content, allow_empty=True))
        object.__setattr__(self, "target_id", _text("target_id", self.target_id, allow_empty=True))
        object.__setattr__(self, "attributes", dict(self.attributes))
        _text("provenance", self.provenance)

    def to_dict(self) -> dict[str, Any]:
        """Return the observed cue with modality-specific provenance intact."""
        return _plain(asdict(self))


@dataclass(frozen=True, slots=True)
class ReferentCandidate:
    """Describe a candidate object, event, property, action, or relational meaning."""

    referent_id: str
    description: str
    features: tuple[str, ...] = ()
    properties: Mapping[str, Any] = field(default_factory=dict)
    salience: float = 0.5
    context_fit: float = 0.5
    relation_type: str = "entity"

    def __post_init__(self) -> None:
        """Validate referent identity and normalize feature/context evidence."""
        _text("referent_id", self.referent_id)
        _text("description", self.description, allow_empty=True)
        object.__setattr__(self, "features", tuple(
            _text("feature", item).lower() for item in self.features
        ))
        for name in ("salience", "context_fit"):
            object.__setattr__(self, name, _unit(name, getattr(self, name)))
        object.__setattr__(self, "properties", dict(self.properties))
        _text("relation_type", self.relation_type)

    def to_dict(self) -> dict[str, Any]:
        """Return the referent description and its grounded properties."""
        return _plain(asdict(self))


@dataclass(frozen=True, slots=True)
class SocialFeedback:
    """Separate corrective information, task outcome, and social-affective reinforcement."""

    feedback_id: str
    kind: str
    content: str = ""
    target_key: str = ""
    factual_value: Any = None
    reliability: float = 0.7
    reward: float = 0.0
    confirms: bool | None = None
    source_id: str = ""
    provenance: str = "partner_feedback"

    def __post_init__(self) -> None:
        """Validate feedback channel and preserve its independent informational/reward fields."""
        _text("feedback_id", self.feedback_id)
        allowed = {
            "informational_correction", "recast", "confirmation",
            "outcome", "social_affective", "clarification", "other",
        }
        if self.kind not in allowed:
            raise ValueError("unsupported feedback kind: " + self.kind)
        object.__setattr__(self, "content", _text("content", self.content, allow_empty=True))
        object.__setattr__(self, "target_key", _text("target_key", self.target_key, allow_empty=True))
        object.__setattr__(self, "source_id", _text("source_id", self.source_id, allow_empty=True))
        object.__setattr__(self, "reliability", _unit("reliability", self.reliability))
        object.__setattr__(self, "reward", _finite("reward", self.reward))
        _text("provenance", self.provenance)

    def to_dict(self) -> dict[str, Any]:
        """Return feedback while keeping factual content and social reward separate."""
        return _plain(asdict(self))


@dataclass(frozen=True, slots=True)
class NarrativeEvent:
    """Represent a story event with its agent, goal, causal links, and information access."""

    event_id: str
    description: str
    agent_id: str = ""
    action: str = ""
    goal: str = ""
    timestamp: float = 0.0
    causal_parents: tuple[str, ...] = ()
    reveals_to: tuple[str, ...] = ()
    facts: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate narrative identity and preserve agent-specific event information."""
        _text("event_id", self.event_id)
        for name in ("description", "agent_id", "action", "goal"):
            object.__setattr__(self, name, _text(name, getattr(self, name), allow_empty=True))
        object.__setattr__(self, "timestamp", _finite("timestamp", self.timestamp))
        object.__setattr__(self, "causal_parents", tuple(str(x) for x in self.causal_parents))
        object.__setattr__(self, "reveals_to", tuple(str(x) for x in self.reveals_to))
        object.__setattr__(self, "facts", dict(self.facts))

    def to_dict(self) -> dict[str, Any]:
        """Return the story event as a typed graph-node record."""
        return _plain(asdict(self))


@dataclass(frozen=True, slots=True)
class SocialInteractionEvent:
    """Describe one multimodal conversational learning episode between learner and partner."""

    event_id: str
    learner_id: str
    partner_id: str
    utterance: str = ""
    signals: tuple[SocialSignal, ...] = ()
    referents: tuple[ReferentCandidate, ...] = ()
    learner_focus: str = ""
    partner_focus_hint: str = ""
    context: Mapping[str, Any] = field(default_factory=dict)
    learning_goal: str = ""
    candidate_intentions: tuple[str, ...] = ()
    intention_hint: str = ""
    target_word: str = ""
    candidate_mappings: Mapping[str, float] = field(default_factory=dict)
    predicted_partner_response: str = ""
    partner_response: str = ""
    learner_attempt: str = ""
    feedback: tuple[SocialFeedback, ...] = ()
    target_concept: str = ""
    partner_claims: Mapping[str, float] = field(default_factory=dict)
    partner_evidence: Mapping[str, float] = field(default_factory=dict)
    required_explanation_terms: tuple[str, ...] = ()
    required_relations: tuple[str, ...] = ()
    narrative_events: tuple[NarrativeEvent, ...] = ()
    outcome_reward: float = 0.0
    difficulty: float = 0.5
    competence_observation: float | None = None
    elapsed_hours: float = 0.5
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate interaction identifiers, cue collections, and model probabilities."""
        for name in ("event_id", "learner_id", "partner_id"):
            _text(name, getattr(self, name))
        for name in (
            "utterance", "learner_focus", "partner_focus_hint", "learning_goal",
            "intention_hint", "target_word", "predicted_partner_response",
            "partner_response", "learner_attempt", "target_concept",
        ):
            object.__setattr__(self, name, _text(name, getattr(self, name), allow_empty=True))
        object.__setattr__(self, "signals", tuple(self.signals))
        object.__setattr__(self, "referents", tuple(self.referents))
        object.__setattr__(self, "feedback", tuple(self.feedback))
        object.__setattr__(self, "narrative_events", tuple(self.narrative_events))
        if len({item.signal_id for item in self.signals}) != len(self.signals):
            raise ValueError("signal_id values must be unique within an interaction event")
        if len({item.referent_id for item in self.referents}) != len(self.referents):
            raise ValueError("referent_id values must be unique within an interaction event")
        for name in ("candidate_mappings", "partner_claims", "partner_evidence"):
            values = {str(k): _unit(name + "." + str(k), v)
                      for k, v in getattr(self, name).items()}
            object.__setattr__(self, name, values)
        object.__setattr__(self, "context", dict(self.context))
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "candidate_intentions", tuple(
            _text("candidate_intention", x).lower() for x in self.candidate_intentions
        ))
        object.__setattr__(self, "required_explanation_terms", tuple(
            _text("required_term", x).lower() for x in self.required_explanation_terms
        ))
        object.__setattr__(self, "required_relations", tuple(
            _text("required_relation", x).lower() for x in self.required_relations
        ))
        object.__setattr__(self, "outcome_reward", _finite("outcome_reward", self.outcome_reward))
        object.__setattr__(self, "difficulty", _unit("difficulty", self.difficulty))
        if self.competence_observation is not None:
            object.__setattr__(
                self, "competence_observation",
                _unit("competence_observation", self.competence_observation),
            )
        if _finite("elapsed_hours", self.elapsed_hours) < 0:
            raise ValueError("elapsed_hours must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        """Return the full interaction event with observed and inferred channels separable."""
        return _plain(asdict(self))


@dataclass
class PartnerModel:
    """Maintain uncertain estimates of one partner's knowledge, goals, reliability, and style."""

    partner_id: str
    knowledge_state: dict[str, float] = field(default_factory=dict)
    goal_beliefs: dict[str, float] = field(default_factory=dict)
    vocabulary: set[str] = field(default_factory=set)
    conventions: dict[str, str] = field(default_factory=dict)
    reliability: float = 0.5
    attention_focus: str = ""
    turn_style: dict[str, float] = field(default_factory=dict)
    interaction_count: int = 0
    last_seen: str = ""

    def __post_init__(self) -> None:
        """Validate and normalize partner-specific probability estimates."""
        _text("partner_id", self.partner_id)
        self.knowledge_state = {str(k): _unit("knowledge_state", v)
                                for k, v in self.knowledge_state.items()}
        self.goal_beliefs = {str(k): _unit("goal_belief", v)
                             for k, v in self.goal_beliefs.items()}
        self.vocabulary = {str(x).lower() for x in self.vocabulary}
        self.conventions = {str(k): str(v) for k, v in self.conventions.items()}
        self.reliability = _unit("reliability", self.reliability)
        self.attention_focus = str(self.attention_focus)
        self.turn_style = {str(k): _unit("turn_style", v) for k, v in self.turn_style.items()}
        if self.interaction_count < 0:
            raise ValueError("interaction_count must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        """Return the partner model with explicit uncertainty-bearing estimates."""
        return _plain(asdict(self))


@dataclass(frozen=True, slots=True)
class SocialEpisodeMemory:
    """Bind learned content to source person, context, interaction evidence, and confidence."""

    event_id: str
    partner_id: str
    learner_id: str
    content: str
    source_claims: Mapping[str, Any]
    context_tags: tuple[str, ...]
    modalities: tuple[str, ...]
    concept_keys: tuple[str, ...]
    feedback_ids: tuple[str, ...]
    confidence: float
    source_reliability: float
    episodic_strength: float
    consolidation_strength: float
    timestamp: str = ""
    provenance: str = "social_interaction_episode"

    def __post_init__(self) -> None:
        """Validate source-linked memory strength and detach metadata."""
        for name in ("event_id", "partner_id", "learner_id"):
            _text(name, getattr(self, name))
        for name in ("confidence", "source_reliability", "episodic_strength", "consolidation_strength"):
            object.__setattr__(self, name, _unit(name, getattr(self, name)))
        object.__setattr__(self, "source_claims", dict(self.source_claims))
        for name in ("context_tags", "modalities", "concept_keys", "feedback_ids"):
            object.__setattr__(self, name, tuple(str(x) for x in getattr(self, name)))
        object.__setattr__(self, "content", _text("content", self.content, allow_empty=True))
        object.__setattr__(self, "timestamp", str(self.timestamp))
        _text("provenance", self.provenance)

    def to_dict(self) -> dict[str, Any]:
        """Return episodic content and source metadata without collapsing provenance."""
        return _plain(asdict(self))


@dataclass
class LearnerSocialState:
    """Hold concept knowledge, partner models, discourse state, and source-linked social memories."""

    learner_id: str
    semantic_knowledge: dict[str, float] = field(default_factory=dict)
    knowledge_claims: dict[str, Any] = field(default_factory=dict)
    source_ledger: list[dict[str, Any]] = field(default_factory=list)
    word_referent_associations: dict[str, float] = field(default_factory=dict)
    action_values: dict[str, float] = field(default_factory=dict)
    retrieval_strengths: dict[str, float] = field(default_factory=dict)
    explanation_strength: dict[str, float] = field(default_factory=dict)
    skill_competence: dict[str, float] = field(default_factory=dict)
    partner_models: dict[str, PartnerModel] = field(default_factory=dict)
    episodic_memories: list[SocialEpisodeMemory] = field(default_factory=list)
    narrative_graph: dict[str, dict[str, Any]] = field(default_factory=dict)
    narrative_agent_states: dict[str, dict[str, Any]] = field(default_factory=dict)
    event_history: list[dict[str, Any]] = field(default_factory=list)
    cross_partner_patterns: dict[str, dict[str, Any]] = field(default_factory=dict)
    current_goal: str = ""
    current_shared_focus: str = ""
    last_turn_state: str = "unknown"
    unresolved_questions: list[str] = field(default_factory=list)
    interaction_count: int = 0

    def __post_init__(self) -> None:
        """Validate learner identity and normalize all persistent probability maps."""
        _text("learner_id", self.learner_id)
        for name in (
            "semantic_knowledge", "word_referent_associations", "action_values",
            "retrieval_strengths", "explanation_strength", "skill_competence",
        ):
            values = {str(k): _unit(name, v) for k, v in getattr(self, name).items()}
            setattr(self, name, values)
        self.knowledge_claims = dict(self.knowledge_claims)
        self.source_ledger = list(self.source_ledger)
        self.narrative_agent_states = {str(k): dict(v) for k, v in self.narrative_agent_states.items()}
        self.partner_models = {
            str(key): value if isinstance(value, PartnerModel) else PartnerModel(**value)
            for key, value in self.partner_models.items()
        }
        self.episodic_memories = [
            item if isinstance(item, SocialEpisodeMemory) else SocialEpisodeMemory(**item)
            for item in self.episodic_memories
        ]
        self.narrative_graph = {str(k): dict(v) for k, v in self.narrative_graph.items()}
        self.event_history = list(self.event_history)
        self.cross_partner_patterns = {str(k): dict(v) for k, v in self.cross_partner_patterns.items()}
        self.current_goal = str(self.current_goal)
        self.current_shared_focus = str(self.current_shared_focus)
        if self.interaction_count < 0:
            raise ValueError("interaction_count must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        """Return the complete learner state in a JSON-compatible form."""
        return _plain(asdict(self))


@dataclass(frozen=True, slots=True)
class SocialLearningConfig:
    """Configure cue integration, uncertainty, repair, scaffolding, and memory updates."""

    attention_temperature: float = 0.45
    intention_temperature: float = 0.60
    referent_temperature: float = 0.45
    cue_window_seconds: float = 2.0
    joint_attention_threshold: float = 0.58
    turn_yield_threshold: float = 0.62
    repair_error_threshold: float = 0.80
    repair_probability_threshold: float = 0.30
    knowledge_update_rate: float = 0.18
    association_learning_rate: float = 0.16
    reward_learning_rate: float = 0.15
    partner_model_learning_rate: float = 0.08
    scaffold_gain: float = 1.0
    scaffold_minimum: float = 0.10
    scaffold_maximum: float = 0.90
    explanation_retrieval_rate: float = 0.12
    consolidation_rate: float = 0.15
    memory_limit: int = 500
    narrative_surprise_threshold: float = 0.75

    def __post_init__(self) -> None:
        """Validate model temperatures, bounded learning rates, and memory limits."""
        for name in (
            "attention_temperature", "intention_temperature", "referent_temperature",
            "cue_window_seconds",
        ):
            if _finite(name, getattr(self, name)) <= 0:
                raise ValueError(name + " must be positive")
        for name in (
            "joint_attention_threshold", "turn_yield_threshold",
            "repair_probability_threshold", "knowledge_update_rate",
            "association_learning_rate", "reward_learning_rate",
            "partner_model_learning_rate", "scaffold_gain", "scaffold_minimum",
            "scaffold_maximum", "explanation_retrieval_rate", "consolidation_rate",
            "narrative_surprise_threshold",
        ):
            object.__setattr__(self, name, _unit(name, getattr(self, name)))
        if not 0.0 < self.repair_error_threshold:
            raise ValueError("repair_error_threshold must be positive")
        if self.scaffold_minimum > self.scaffold_maximum:
            raise ValueError("scaffold_minimum must not exceed scaffold_maximum")
        if self.memory_limit < 1:
            raise ValueError("memory_limit must be at least 1")


@dataclass(frozen=True, slots=True)
class SocialLearningResult:
    """Capture one completed 16-stage social interaction and conversational learning cycle."""

    event_id: str
    partner_id: str
    joint_attention: Mapping[str, Any]
    intention: Mapping[str, Any]
    multimodal_representation: Mapping[str, Any]
    semantic_grounding: Mapping[str, Any]
    turn_decision: Mapping[str, Any]
    audience_design: Mapping[str, Any]
    repair: Mapping[str, Any]
    feedback_learning: Mapping[str, Any]
    scaffolding: Mapping[str, Any]
    explanation: Mapping[str, Any]
    perspective_update: Mapping[str, Any]
    narrative_model: Mapping[str, Any]
    social_memory: Mapping[str, Any]
    transfer: Mapping[str, Any]
    operation_trace: tuple[Mapping[str, Any], ...]
    biological: Mapping[str, Any]
    final_state: Mapping[str, Any]
    output: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Validate identity and require exactly one trace entry for every Point 14 operation."""
        _text("event_id", self.event_id)
        _text("partner_id", self.partner_id)
        if len(self.operation_trace) != 16:
            raise ValueError("Point 14 results must contain exactly 16 operation trace records")

    def to_dict(self) -> dict[str, Any]:
        """Return the complete social-learning result in a serializable form."""
        return _plain(asdict(self))
