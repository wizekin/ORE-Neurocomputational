"""Source-ordered registry for the seventeen Point 23 parsing operations."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Point23OperationSpec:
    """Describe one algorithmic operation and its distributed biological reference."""

    operation_id: str
    name: str
    purpose: str
    mathematical_form: str
    biological_reference: str
    evidence_level: str = "distributed_language_network_with_computational_abstraction"

    def to_dict(self) -> dict[str, Any]:
        """Return plain serializable operation metadata."""
        return asdict(self)


_ROWS = (
    ("segment_input", "Segment the spoken or written stream while preserving boundary cues",
     "P(b_t=1|a_1:T,c_t); X=(x_1,...,x_T)",
     "Auditory or visual input pathways and attention provide candidate word and boundary information."),
    ("activate_lexicon", "Retrieve lexical entries, senses, grammatical categories, and inflectional features",
     "P(c|x_t,h_t) proportional to P(x_t|c)P(c|h_t)",
     "Distributed temporal lexical/semantic representations interact with frontal language systems."),
    ("generate_parses", "Incrementally extend competing syntactic structures instead of assuming one parse",
     "P_t={P_1,...,P_k}; each P_i carries dependencies and decisions",
     "Temporal and inferior-frontal language networks support incremental structure building and unification."),
    ("estimate_expectations", "Estimate likely structural and lexical continuations from prior language experience",
     "P(x_(t+1),P_(t+1)|x_1:t,P_t)",
     "Experience-dependent expectations modulate activity across temporal and frontal language regions."),
    ("integrate_semantics", "Combine lexical meaning, event knowledge, plausibility, and thematic-role constraints",
     "S(P_i)=alpha*S_syntax+beta*S_lexical+gamma*S_semantic+delta*S_context",
     "Temporal semantic systems and frontal language regions contribute to combinatorial and event interpretation."),
    ("select_provisional_parse", "Select a provisional parse or retain a beam of competing hypotheses",
     "B_t=TopK(P_i,S(P_i)); p_i=softmax(S_i/tau)",
     "Incremental commitment and competition are distributed; a single universal parser strategy is not assumed."),
    ("maintain_dependencies", "Maintain open dependencies and thematic roles under decay and interference",
     "A_i(t)=A_i(t0)exp(-lambda*Delta t)-sum_j rho_ij*A_j(t)",
     "Working memory, attention, and frontotemporal language pathways maintain relations over intervening words."),
    ("detect_conflict", "Compare incoming cues with parse-specific predictions and quantify surprise/conflict",
     "Surprisal(x_t)=-log P(x_t|x_1:t-1); delta_t=observed-predicted",
     "Medial frontal conflict monitoring interacts with language and frontoparietal control networks."),
    ("test_parse_viability", "Test grammatical, lexical, and structural compatibility while retaining audit history",
     "V(P_i,t)=G_i(t)*L_i(t)*C_i(t)",
     "Temporal language representations and inferior-frontal selection/unification contribute to constraint checking."),
    ("trigger_revision", "Request reanalysis when a parse becomes incompatible or a competitor becomes more useful",
     "revise if V(P_best)<theta_V or S(P_alt)-S(P_best)>theta_Delta",
     "Left inferior frontal and broader control networks may contribute to increased analysis under conflict."),
    ("locate_choice_point", "Identify the earliest consequential decision that distinguishes competing structures",
     "(x_j,d_j,P_before,P_after,c_j)",
     "Reactivation of earlier lexical/structural information recruits memory and temporal-frontal interactions."),
    ("reconstruct_structure", "Construct alternative phrase structures, attachments, or clause boundaries",
     "P(P_i|X) proportional to P(X|P_i)P(P_i)",
     "Inferior-frontal and temporal language circuits cooperate during structural revision; localization is not one-to-one."),
    ("reassign_roles", "Recompute agent, patient, subject, object, and other semantic roles from the revised parse",
     "E=<predicate,agent,patient,recipient,instrument,location,time>",
     "Temporal semantic representations and frontotemporal integration support event-role composition."),
    ("reconcile_interpretations", "Preserve compatible information and separately update syntax, meaning, and role confidence",
     "C_new=C_old+eta*(C_evidence-C_old)",
     "Previously active interpretations can persist while revised representations are being integrated."),
    ("choose_reanalysis_depth", "Choose full reanalysis, task-sufficient gist, or explicit unresolved status",
     "EU_continue=P(correct|more processing)*Value_accuracy-Cost_processing",
     "Working memory, attention, and domain-general control contribute to task-sensitive effort allocation."),
    ("integrate_surface_cues", "Use prosody, punctuation, and discourse as reliability-weighted structural evidence",
     "S(P_i)=sum_m w_m*log P(c_m|P_i)",
     "Auditory prosody and visual boundary cues interact with temporal/frontal language and discourse systems."),
    ("adapt_expectations", "Store the outcome and gradually adapt construction expectations from exposure or feedback",
     "P_(t+1)(c)=(N_t(c)+alpha)/(N_t+alpha*K)",
     "Distributed memory and learning systems update language experience over repeated encounters."),
)

OPERATION_REGISTRY = tuple(
    Point23OperationSpec(f"P23-{index:02d}", *row)
    for index, row in enumerate(_ROWS, start=1)
)


def registry_as_dicts() -> list[dict[str, Any]]:
    """Return all seventeen operations in exact source order."""
    return [item.to_dict() for item in OPERATION_REGISTRY]
