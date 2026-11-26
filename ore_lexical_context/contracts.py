"""Typed contracts for lexical, morphological, phonological and predictive processing."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence
import math

def _features(name: str, values: Mapping[str, float]) -> None:
    for key, raw in values.items():
        value = float(raw)
        if not isinstance(key, str) or not key.strip() or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must map feature names to finite values in [0, 1]")

@dataclass(frozen=True, slots=True)
class LexicalMeaning:
    meaning_id: str
    features: Mapping[str, float] = field(default_factory=dict)
    prior: float = 1.0
    gloss: str = ""
    verified_by: str = "unspecified"
    source_reference: str = "unspecified"
    semantic_concept_ids: tuple[str, ...] = ()
    semantic_link_status: str = "UNLINKED"
    def __post_init__(self):
        if not self.meaning_id.strip() or not math.isfinite(self.prior) or self.prior <= 0:
            raise ValueError("meaning_id and positive finite prior are required")
        _features("meaning features", self.features)
        if len(self.semantic_concept_ids) != len(set(self.semantic_concept_ids)):
            raise ValueError("semantic concept IDs on a lexical meaning must be unique")
        if self.semantic_link_status not in {"UNLINKED", "LINKED", "AMBIGUOUS", "MISSING", "CONFLICTED", "CONTRADICTED", "UNRESOLVED", "CANDIDATE", "NEUTRAL", "UNASSESSED"}:
            raise ValueError("unsupported lexical meaning semantic-link status")

@dataclass(frozen=True, slots=True)
class LexicalEntry:
    entry_id: str
    orthography: str
    frequency: float = 1.0
    exposure_count: float = 0.0
    phonemes: tuple[str, ...] = ()
    meanings: tuple[LexicalMeaning, ...] = ()
    lexical_category: str = "unknown"
    language: str = "und"
    verified_by: str = "unspecified"
    source_reference: str = "unspecified"
    semantic_concept_ids: tuple[str, ...] = ()
    semantic_link_status: str = "UNLINKED"
    lemma_entry_id: str | None = None
    morphology_link_status: str = "NOT_APPLICABLE"
    script: str | None = None
    grammatical_information_json: str | None = None
    morphology_source_reference: str | None = None
    def __post_init__(self):
        if not self.entry_id.strip() or not self.orthography.strip():
            raise ValueError("entry_id and orthography must be non-empty")
        if not math.isfinite(self.frequency) or self.frequency <= 0 or not math.isfinite(self.exposure_count) or self.exposure_count < 0:
            raise ValueError("frequency must be positive and exposure_count non-negative")
        if any(not isinstance(p, str) or not p.strip() for p in self.phonemes):
            raise ValueError("phonemes must be non-empty symbols")
        if any(not isinstance(m, LexicalMeaning) for m in self.meanings):
            raise TypeError("meanings must contain LexicalMeaning records")
        if len(self.semantic_concept_ids) != len(set(self.semantic_concept_ids)):
            raise ValueError("semantic concept IDs on a lexical entry must be unique")
        if self.semantic_link_status not in {"UNLINKED", "LINKED", "AMBIGUOUS", "MISSING", "CONFLICTED", "CONTRADICTED", "UNRESOLVED", "CANDIDATE", "NEUTRAL", "UNASSESSED"}:
            raise ValueError("unsupported lexical entry semantic-link status")
        if self.morphology_link_status not in {"NOT_APPLICABLE", "LINKED", "AMBIGUOUS", "MISSING", "CONFLICTED", "CONTRADICTED", "UNRESOLVED", "CANDIDATE", "NEUTRAL", "UNASSESSED"}:
            raise ValueError("unsupported lexical entry morphology-link status")

@dataclass(frozen=True, slots=True)
class MorphemeEntry:
    morpheme_id: str
    form: str
    morpheme_type: str
    semantic_features: Mapping[str, float] = field(default_factory=dict)
    phonemes: tuple[str, ...] = ()
    frequency: float = 1.0
    productivity: float = 1.0
    verified_by: str = "unspecified"
    source_reference: str = "unspecified"
    def __post_init__(self):
        if not self.morpheme_id.strip() or not self.form.strip() or self.morpheme_type not in {"prefix", "root", "suffix"}:
            raise ValueError("morpheme requires an id, form and prefix/root/suffix type")
        if not math.isfinite(self.frequency) or self.frequency <= 0 or not 0 <= self.productivity <= 1:
            raise ValueError("morpheme frequency/productivity is invalid")
        _features("morpheme features", self.semantic_features)

@dataclass(frozen=True, slots=True)
class G2PRule:
    graphemes: str
    phonemes: tuple[str, ...]
    reliability: float = 0.8
    language: str = "und"
    verified_by: str = "unspecified"
    source_reference: str = "unspecified"
    def __post_init__(self):
        if not self.graphemes or not self.phonemes or not 0 <= self.reliability <= 1:
            raise ValueError("G2P rule needs graphemes, phonemes and reliability in [0,1]")

@dataclass(frozen=True, slots=True)
class LexicalContextRequest:
    sequence_id: str
    text: str | None = None
    tokens: Sequence[str] | None = None
    language: str = "und"
    context_features: Mapping[str, float] = field(default_factory=dict)
    incoming_semantic_evidence: Mapping[int, Mapping[str, float]] = field(default_factory=dict)
    timestamp_ms: float | None = None
    source_reference: str | None = None
    contextual_records: Sequence[Any] = ()
    evidence_records: Sequence[Any] = ()
    source_provenance: Sequence[Any] = ()
    def __post_init__(self):
        if not self.sequence_id.strip() or ((self.text is None) == (self.tokens is None)):
            raise ValueError("sequence_id required; provide exactly one of text or tokens")
        if self.text is not None and not self.text.strip():
            raise ValueError("text cannot be empty")
        if self.tokens is not None and (not self.tokens or any(not isinstance(t, str) or not t.strip() for t in self.tokens)):
            raise ValueError("tokens must be non-empty strings")
        _features("context_features", self.context_features)
        for i, fs in self.incoming_semantic_evidence.items():
            if not isinstance(i, int) or i < 0: raise ValueError("semantic evidence keys must be token indices")
            _features(f"incoming_semantic_evidence[{i}]", fs)
        if self.timestamp_ms is not None and (not math.isfinite(self.timestamp_ms) or self.timestamp_ms < 0):
            raise ValueError("timestamp_ms must be finite and non-negative")
        if any(not isinstance(getattr(item, "context_id", None), str) for item in self.contextual_records):
            raise TypeError("contextual_records must contain typed context records")
        if any(not isinstance(getattr(item, "evidence_id", None), str) for item in self.evidence_records):
            raise TypeError("evidence_records must contain typed evidence records")
        if any(not isinstance(getattr(item, "source_id", None), str) for item in self.source_provenance):
            raise TypeError("source_provenance must contain typed source records")

@dataclass(frozen=True, slots=True)
class LexicalContextConfig:
    max_candidates: int = 20
    similarity_floor: float = 0.25
    max_edit_distance: int = 2
    base_resting_activation: float = 0.15
    frequency_gain: float = 0.35
    threshold_base: float = 0.55
    threshold_frequency_gain: float = 0.15
    threshold_adaptation_gain: float = 0.10
    stimulus_gain: float = 0.55
    repetition_gain: float = 0.22
    adaptation_gain: float = 0.18
    activation_temperature: float = 0.20
    history_tau_ms: float = 1200.0
    adaptation_tau_ms: float = 1800.0
    token_interval_ms: float = 250.0
    max_recent_repetitions: int = 8
    morphology_gain: float = 0.45
    whole_word_gain: float = 0.65
    g2p_gain: float = 0.20
    neighborhood_gain: float = 0.12
    shared_activation_gain: float = 0.08
    lateral_inhibition: float = 0.18
    recurrent_excitation: float = 0.12
    settle_steps: int = 6
    settle_damping: float = 0.55
    context_gain: float = 0.65
    confidence_threshold: float = 0.55
    ambiguity_margin: float = 0.08
    sensory_precision: float = 1.0
    prior_precision: float = 0.6
    prediction_error_gain: float = 0.65
    minimum_prediction_count: float = 1.0
    record_exposures: bool = True
    simulate_biology: bool = True
    dt_ms: float = 1.0
    steps_per_operation: int = 2
    membrane_tau_ms: float = 20.0
    refractory_ms: float = 2.0
    population_size: int = 12
    rest_mv: float = -70.0
    reset_mv: float = -68.0
    threshold_mv: float = -50.0
    membrane_resistance_mohm: float = 100.0
    current_gain_pa: float = 320.0
    recurrent_gain_pa: float = 18.0
    inhibition_gain_pa: float = 22.0
    adaptation_current_gain_pa: float = 4.0
    def __post_init__(self):
        if self.max_candidates < 1 or self.max_edit_distance < 0 or self.settle_steps < 1 or self.population_size < 2:
            raise ValueError("candidate/edit/settling/population limits are invalid")
        unit = ("similarity_floor","base_resting_activation","frequency_gain","threshold_frequency_gain","threshold_adaptation_gain",
                "stimulus_gain","repetition_gain","adaptation_gain","morphology_gain","whole_word_gain","g2p_gain",
                "neighborhood_gain","shared_activation_gain","lateral_inhibition","recurrent_excitation","settle_damping",
                "context_gain","confidence_threshold","ambiguity_margin","prediction_error_gain")
        for name in unit:
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0 <= value <= 1: raise ValueError(f"{name} must lie in [0,1]")
        positive = ("activation_temperature","history_tau_ms","adaptation_tau_ms","token_interval_ms","sensory_precision",
                    "prior_precision","minimum_prediction_count","dt_ms","membrane_tau_ms","membrane_resistance_mohm","current_gain_pa")
        for name in positive:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0: raise ValueError(f"{name} must be positive and finite")
        if self.steps_per_operation < 1 or self.refractory_ms < 0 or self.recurrent_gain_pa < 0 or self.inhibition_gain_pa < 0:
            raise ValueError("neural simulation parameters are invalid")
        if not self.rest_mv <= self.reset_mv < self.threshold_mv: raise ValueError("require rest <= reset < threshold")
    def to_dict(self): return asdict(self)

@dataclass(slots=True)
class LexicalContextResult:
    sequence_id: str
    original_text: str
    normalized_text: str
    tokens: list[dict[str, Any]]
    sentence_context: dict[str, float]
    biological: dict[str, Any]
    operation_trace: list[dict[str, Any]]
    diagnostics: dict[str, Any]
    metadata: dict[str, Any]
    def to_dict(self):
        def clean(v):
            if isinstance(v, Mapping): return {str(k): clean(x) for k,x in v.items()}
            if isinstance(v, (tuple,list)): return [clean(x) for x in v]
            if isinstance(v, float) and not math.isfinite(v): return None
            if isinstance(v, (str,int,float,bool)) or v is None: return v
            return repr(v)
        return clean({"schema":"ore.lexical_context.v1","sequence_id":self.sequence_id,"original_text":self.original_text,
                      "normalized_text":self.normalized_text,"tokens":self.tokens,"sentence_context":self.sentence_context,
                      "biological":self.biological,"operation_trace":self.operation_trace,"diagnostics":self.diagnostics,"metadata":self.metadata})
