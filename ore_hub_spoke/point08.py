"""Point 08 source-aligned distributed semantic learning on the ORE hub-and-spoke core.

This layer adds explicit temporal cross-modal association learning and a 15-step
trace matching Point_08_Distributed_Semantic_Representations_Hub_and_Spoke.md.
It reuses the existing 21-stage functional network and optional LIF reference,
rather than maintaining a second disconnected ATL/spoke implementation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import itertools
import math
from pathlib import Path
import time
import uuid
from typing import Any, Mapping, Sequence

from .contracts import (
    HubAndSpokeConfig,
    HubAndSpokeRequest,
    HubAndSpokeResult,
    LearningExperience,
    MODALITIES,
    canonical_modality,
)
from .pipeline import HubAndSpokeProcessor


@dataclass(frozen=True, slots=True)
class Point08StageSpec:
    """Describe one of the source document's ordered Point 08 operations."""

    stage_id: str
    name: str
    objective: str
    mathematical_form: str
    biological_reference: str
    evidence_level: str

    def to_dict(self) -> dict[str, str]:
        """Return a serializable copy of the stage specification."""
        return asdict(self)


POINT08_STAGE_REGISTRY: tuple[Point08StageSpec, ...] = (
    Point08StageSpec("P08-01", "Receive a meaningful input and identify the processing demand",
        "Accept sensory, linguistic, remembered, or action-related cues and retain the requested task without prematurely assigning a concept.",
        "X(t)={x_v,x_a,x_s,x_m,x_l,...}; keep a separate availability mask and task vector g(t).",
        "Distributed sensory and language systems register input in parallel; attention and task state bias subsequent processing.",
        "distributed input processing supported; the explicit typed signal set is a computational contract"),
    Point08StageSpec("P08-02", "Process each sensory stream in its specialized pathway",
        "Transform each available modality independently into a graded population-like feature pattern.",
        "r_m=phi_m(W_m x_m+b_m).",
        "Visual, auditory, somatosensory, chemosensory, motor, and association pathways have modality-sensitive response properties.",
        "hierarchical modality-specific processing supported; matrix transforms are computational abstractions"),
    Point08StageSpec("P08-03", "Preserve the modality-specific structure of each feature",
        "Keep source modality, feature identity, reliability, uncertainty, and provenance attached to each feature representation.",
        "F_i=(m_i,f_i,q_i,t_i); z_i=(f_i-mu_m)/sqrt(sigma_m^2+epsilon).",
        "Distributed association populations preserve modality-linked detail; there is no single feature table or universal neural z-score.",
        "modality-sensitive distributed representations supported; explicit records and normalization are model choices"),
    Point08StageSpec("P08-04", "Estimate the reliability and relevance of incoming features",
        "Weight each available stream according to signal quality, task relevance, and attention while preserving disagreement and uncertainty.",
        "q_m=sigmoid(alpha*S_m+beta*R_m+gamma*A_m); a_m=softmax(q_m/tau).",
        "Attention and gain modulation alter the relative influence of sensory and association pathways.",
        "attention and precision modulation supported; scalar confidence and softmax are computational"),
    Point08StageSpec("P08-05", "Detect temporal co-occurrence across modalities",
        "Estimate whether features from different spokes occurred close enough in time to support a cross-modal association.",
        "e_i(t)=integral r_i(s) exp(-(t-s)/tau_i) ds; C_ij <- rho*C_ij + x_i*x_j*exp(-|dt|/tau_e).",
        "Temporally related population activity can create eligibility traces for associative plasticity; timing and neuromodulation matter.",
        "activity-dependent temporal association supported; exponential eligibility is a testable approximation"),
    Point08StageSpec("P08-06", "Learn connections between modality-specific features and integrative representations",
        "Strengthen reliable spoke-to-spoke and spoke-to-hub associations while using bounded plasticity to limit indiscriminate coupling.",
        "Delta W_ij=eta*x_i*y_j; W_ij <- clip(rho*W_ij+eta*x_i*y_j*e_ij,0,1).",
        "Hebbian/LTP/LTD and eligibility-dependent plasticity can change connectivity among distributed cortical and transmodal populations.",
        "synaptic plasticity supported; the particular learning rule is a computational surrogate"),
    Point08StageSpec("P08-07", "Converge distributed feature activity on anterior temporal integration networks",
        "Project each retained spoke pattern into a shared representational space and preserve the identity of each source.",
        "z_input=sum_m a_m*A_m*r_m; z_ATL=phi(z_input+R*z_prior+b).",
        "Anterior temporal regions form a major transmodal semantic integration system through distributed convergent connections.",
        "ATL involvement across modalities supported; a single homogeneous hub node is an oversimplification"),
    Point08StageSpec("P08-08", "Integrate complementary features into a coherent conceptual pattern",
        "Bind compatible features and their typed relations, resolve conflicts using reliability and context, and retain distinctive detail.",
        "K_ij=f_i^T*M_ij*f_j; z_c=phi(W_f*f+W_r*r+b).",
        "Recurrent ATL and connected temporal/parietal association networks can support coherent feature and relation structure.",
        "semantic integration supported; explicit compatibility matrices are model hypotheses"),
    Point08StageSpec("P08-09", "Use recurrent feedback to reactivate relevant modality-specific features",
        "Send top-down predictions to relevant spokes and refine the integrated state without overwriting original observations.",
        "r_hat_m=phi_m(B_m*z_ATL); delta_m=r_m-r_hat_m; E=sum_m q_m*||delta_m||^2.",
        "Semantic processing is interactive: integrative activity can bias or reinstate modality-sensitive representations.",
        "recurrent interactions and feature reinstatement supported; a single error scalar is not universal"),
    Point08StageSpec("P08-10", "Retrieve a concept when only one modality or a partial cue is available",
        "Spread activation from the available cue through learned connections, retrieve complementary features, and retain multiple candidates with confidence.",
        "S(c|r_m)=cos(A_m*r_m,z_c); P(c|r_m)=softmax(beta*S_c).",
        "Cue-driven semantic retrieval and recurrent interactions allow access from words, pictures, sounds, actions, or partial features.",
        "cross-modal semantic access supported; cosine/softmax ranking are computational"),
    Point08StageSpec("P08-11", "Preserve category-relevant details and distinguish confusable concepts",
        "Compare nearby concepts and emphasize discriminative attributes rather than allowing broad commonalities to erase meaningful differences.",
        "D(f)=|P(f|c1)-P(f|c2)|; d(c1,c2)=sqrt((z1-z2)^T*M_g*(z1-z2)).",
        "Fine-grained semantic discrimination draws on ATL and connected ventral temporal/perirhinal and task-relevant networks.",
        "fine-grained semantic discrimination supported; metric learning is computational"),
    Point08StageSpec("P08-12", "Apply semantic control when automatic activation is insufficient",
        "Bias access toward weak but task-relevant properties and suppress competing irrelevant features without deleting stored knowledge.",
        "z_g=phi(W_g*z+U_g*g); S_g(c)=z_g^T*v_c.",
        "A semantic-control network including left inferior frontal and posterior temporal regions regulates goal-dependent retrieval.",
        "semantic-control network supported; task-conditioned state is a functional model"),
    Point08StageSpec("P08-13", "Combine semantic information with context and current goals",
        "Use discourse, environmental cues, speaker intention, and goals to select a context-appropriate interpretation.",
        "S(c|x,g)=lambda1*S_sensory+lambda2*S_semantic+lambda3*S_context.",
        "Control and broader reasoning networks bias semantic access according to current context and goals.",
        "context-sensitive semantic selection supported; the weighted sum is computational"),
    Point08StageSpec("P08-14", "Produce the required behaviour or response",
        "Generate a naming, explanation, comparison, inference, or action response while preserving uncertainty.",
        "y=argmax_a P(a|z_g,g); return structured attributes/relations for explanatory tasks.",
        "Semantic representations interact with lexical, motor, executive, and decision systems to guide output.",
        "task-dependent semantic output supported; output selection is an engineering abstraction"),
    Point08StageSpec("P08-15", "Update the distributed concept representation from new experience",
        "Use authorized, sourced experience and feedback to update modality-specific and cross-modal connections while protecting prior knowledge.",
        "W_(t+1)=W_t-eta*grad_W(L_t); use local bounded plasticity and replay as biological alternatives.",
        "Distributed cortical plasticity and recurrent semantic learning modify feature associations over experience.",
        "experience-dependent semantic learning supported; gradient descent is not literal neuronal arithmetic"),
)


@dataclass(frozen=True, slots=True)
class Point08Config:
    """Controls temporal association, eligibility, and cross-modal cue reinstatement."""

    temporal_window_ms: float = 1200.0
    eligibility_tau_ms: float = 180.0
    association_learning_rate: float = 0.12
    association_retention: float = 0.985
    recall_gain: float = 0.45
    minimum_recall_activation: float = 0.025
    max_recalled_features_per_modality: int = 64

    def __post_init__(self) -> None:
        """Validate finite temporal scales, bounded plasticity gains, and recall capacity."""
        for name in ("temporal_window_ms", "eligibility_tau_ms"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("association_learning_rate", "association_retention",
                     "recall_gain", "minimum_recall_activation"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and lie in [0, 1]")
        if self.max_recalled_features_per_modality < 1:
            raise ValueError("max_recalled_features_per_modality must be positive")


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp a finite numerical model value to a closed interval."""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("model values must be finite")
    return min(high, max(low, number))


def _node(modality: str, feature: str) -> str:
    """Create a stable, modality-qualified key for one semantic feature node."""
    return f"{canonical_modality(modality)}::{str(feature).strip()}"


class TemporalAssociationMemory:
    """Learn and retrieve cross-modal feature links from explicitly timed episodes."""

    def __init__(self, config: Point08Config | None = None) -> None:
        """Allocate cross-modal weights, supporting episode indices, and learning counters."""
        self.config = config or Point08Config()
        self.weights: dict[tuple[str, str], float] = {}
        self.supporting_episodes: dict[tuple[str, str], set[str]] = {}
        self.update_count = 0
        self.timestamped_episode_count = 0
        self.untimed_episode_ids: list[str] = []

    def _candidate_events(
        self, experiences: Sequence[LearningExperience]
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Calculate temporal eligibility for feature pairs across distinct modalities."""
        events: list[dict[str, Any]] = []
        untimed: list[str] = []
        for experience in experiences:
            modality_rows: list[tuple[str, float, Mapping[str, float]]] = []
            for raw_modality, features in experience.modality_features.items():
                modality = canonical_modality(raw_modality)
                timestamp = experience.modality_timestamps_ms.get(modality)
                if timestamp is not None:
                    modality_rows.append((modality, float(timestamp), features))
            if len(modality_rows) < 2:
                untimed.append(experience.episode_id or experience.concept_id)
                continue
            for left, right in itertools.combinations(sorted(modality_rows, key=lambda row: row[0]), 2):
                left_modality, left_time, left_features = left
                right_modality, right_time, right_features = right
                if left_modality == right_modality:
                    continue
                delay = abs(left_time - right_time)
                if delay > self.config.temporal_window_ms:
                    continue
                eligibility = math.exp(-delay / self.config.eligibility_tau_ms)
                if eligibility <= 1e-8:
                    continue
                for left_name, left_strength in left_features.items():
                    for right_name, right_strength in right_features.items():
                        source = _node(left_modality, left_name)
                        target = _node(right_modality, right_name)
                        pair = tuple(sorted((source, target)))
                        pre = _clip(left_strength)
                        post = _clip(right_strength)
                        coactivation = pre * post * experience.confidence
                        increment = coactivation * eligibility
                        if increment <= 0.0:
                            continue
                        events.append({
                            "pair": pair,
                            "source_node": source,
                            "target_node": target,
                            "episode_id": experience.episode_id or experience.concept_id,
                            "delay_ms": delay,
                            "eligibility": eligibility,
                            "coactivation": coactivation,
                            "increment": increment,
                            "confidence": experience.confidence,
                        })
        return events, untimed

    def preview(self, experiences: Sequence[LearningExperience]) -> dict[str, Any]:
        """Preview cross-modal eligibility and candidate weight changes without mutating memory."""
        events, untimed = self._candidate_events(experiences)
        aggregated: dict[tuple[str, str], float] = {}
        evidence: dict[tuple[str, str], set[str]] = {}
        for event in events:
            pair = event["pair"]
            aggregated[pair] = min(1.0, aggregated.get(pair, 0.0) + event["increment"])
            evidence.setdefault(pair, set()).add(event["episode_id"])
        rows = []
        for pair, increment in sorted(aggregated.items()):
            prior = self.weights.get(pair, 0.0)
            proposed = _clip(
                prior * self.config.association_retention
                + self.config.association_learning_rate * increment
            )
            rows.append({
                "source": pair[0], "target": pair[1],
                "prior_weight": prior, "eligibility_increment": increment,
                "proposed_weight": proposed,
                "supporting_episode_ids": sorted(evidence[pair]),
            })
        return {
            "status": "preview_only",
            "candidate_pair_count": len(rows),
            "timestamped_episode_count": len(experiences) - len(untimed),
            "untimed_episode_ids": untimed,
            "eligible_event_count": len(events),
            "candidate_pairs": rows,
            "temporal_window_ms": self.config.temporal_window_ms,
            "eligibility_tau_ms": self.config.eligibility_tau_ms,
            "weights_mutated": False,
        }

    def learn(self, experiences: Sequence[LearningExperience]) -> dict[str, Any]:
        """Commit bounded cross-modal association updates from explicitly authorized experience."""
        preview = self.preview(experiences)
        events, untimed = self._candidate_events(experiences)
        aggregated: dict[tuple[str, str], float] = {}
        evidence: dict[tuple[str, str], set[str]] = {}
        for event in events:
            pair = event["pair"]
            aggregated[pair] = min(1.0, aggregated.get(pair, 0.0) + event["increment"])
            evidence.setdefault(pair, set()).add(event["episode_id"])
        for pair in list(self.weights):
            self.weights[pair] *= self.config.association_retention
        for pair, increment in aggregated.items():
            self.weights[pair] = _clip(
                self.weights.get(pair, 0.0)
                + self.config.association_learning_rate * increment
            )
            self.supporting_episodes.setdefault(pair, set()).update(evidence[pair])
        self.update_count += int(bool(aggregated))
        self.timestamped_episode_count += len(experiences) - len(untimed)
        self.untimed_episode_ids.extend(untimed)
        return {
            **preview,
            "status": "committed" if aggregated else "no_temporally_supported_pairs",
            "updated_pair_count": len(aggregated),
            "memory_pair_count": len(self.weights),
            "update_count": self.update_count,
            "weights_mutated": bool(aggregated),
        }

    def spread_activation(
        self, modality_inputs: Mapping[str, Mapping[str, float]]
    ) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
        """Reactivate missing modality features through previously learned cross-modal links."""
        direct: dict[str, dict[str, float]] = {}
        for raw_modality, features in modality_inputs.items():
            modality = canonical_modality(raw_modality)
            target = direct.setdefault(modality, {})
            for feature, value in features.items():
                target[str(feature)] = max(target.get(str(feature), 0.0), _clip(value))
        direct_modalities = set(direct)
        recalled: dict[str, dict[str, float]] = {}
        details: list[dict[str, Any]] = []
        for pair, weight in self.weights.items():
            left, right = pair
            left_modality, left_feature = left.split("::", 1)
            right_modality, right_feature = right.split("::", 1)
            directional = ((left_modality, left_feature, right_modality, right_feature),
                           (right_modality, right_feature, left_modality, left_feature))
            for source_modality, source_feature, target_modality, target_feature in directional:
                if source_modality not in direct or target_modality in direct_modalities:
                    continue
                source_strength = direct[source_modality].get(source_feature, 0.0)
                activation = _clip(source_strength * weight * self.config.recall_gain)
                if activation < self.config.minimum_recall_activation:
                    continue
                target = recalled.setdefault(target_modality, {})
                if activation > target.get(target_feature, 0.0):
                    target[target_feature] = activation
                details.append({
                    "source": f"{source_modality}::{source_feature}",
                    "target": f"{target_modality}::{target_feature}",
                    "association_weight": weight,
                    "source_activation": source_strength,
                    "recalled_activation": activation,
                })
        for modality, values in list(recalled.items()):
            ranked = sorted(values.items(), key=lambda item: (-item[1], item[0]))
            recalled[modality] = dict(ranked[:self.config.max_recalled_features_per_modality])
        return recalled, {
            "direct_modalities": sorted(direct_modalities),
            "recalled_modalities": sorted(recalled),
            "recalled_feature_count": sum(len(values) for values in recalled.values()),
            "recalled_links": details,
            "association_memory_size": len(self.weights),
            "no_learned_associations": not bool(self.weights),
        }

    def snapshot(self) -> dict[str, Any]:
        """Export temporal link weights, provenance, and explicit absence of untimed associations."""
        return {
            "pair_count": len(self.weights),
            "weights": [
                {"source": pair[0], "target": pair[1], "weight": weight,
                 "supporting_episode_ids": sorted(self.supporting_episodes.get(pair, set()))}
                for pair, weight in sorted(self.weights.items())
            ],
            "update_count": self.update_count,
            "timestamped_episode_count": self.timestamped_episode_count,
            "untimed_episode_ids": list(self.untimed_episode_ids),
            "model_scope": "temporal associations are hypotheses from timestamped observations, not direct synaptic measurements",
        }


class BiologicalEligibilityPlasticity:
    """Maintain a separate STDP-like eligibility and bounded synaptic efficacy reference."""

    def __init__(self, config: Point08Config | None = None) -> None:
        """Initialize the synthetic temporal plasticity state independently of functional memory."""
        self.config = config or Point08Config()
        self.eligibility_traces: dict[tuple[str, str], float] = {}
        self.synaptic_efficacy: dict[tuple[str, str], float] = {}
        self.update_count = 0

    def preview(self, memory_preview: Mapping[str, Any]) -> dict[str, Any]:
        """Describe candidate eligibility traces and plasticity changes without changing neural state."""
        rows = []
        for event in memory_preview.get("candidate_pairs", []):
            pair = (str(event["source"]), str(event["target"]))
            eligibility = float(event["eligibility_increment"])
            prior = self.synaptic_efficacy.get(pair, 0.0)
            rows.append({
                "source": pair[0], "target": pair[1],
                "eligibility_trace": eligibility,
                "prior_efficacy": prior,
                "proposed_efficacy": _clip(
                    prior * self.config.association_retention
                    + self.config.association_learning_rate * eligibility
                ),
            })
        return {
            "model": "STDP_like_cross_modal_eligibility_reference",
            "status": "preview_only",
            "candidate_synapse_count": len(rows),
            "candidate_changes": rows,
            "weights_mutated": False,
            "model_scope": "computational plasticity reference, not a measurement of human synapses",
        }

    def learn(self, memory_preview: Mapping[str, Any]) -> dict[str, Any]:
        """Commit eligibility-gated bounded efficacy updates after explicit learning authorization."""
        preview = self.preview(memory_preview)
        for row in preview["candidate_changes"]:
            pair = (row["source"], row["target"])
            self.eligibility_traces[pair] = float(row["eligibility_trace"])
            self.synaptic_efficacy[pair] = float(row["proposed_efficacy"])
        if preview["candidate_changes"]:
            self.update_count += 1
        return {
            **preview,
            "status": "committed" if preview["candidate_changes"] else "no_eligible_synapses",
            "committed_synapse_count": len(preview["candidate_changes"]),
            "update_count": self.update_count,
            "weights_mutated": bool(preview["candidate_changes"]),
        }

    def snapshot(self) -> dict[str, Any]:
        """Return the separate synthetic eligibility and synaptic efficacy state."""
        return {
            "model": "STDP_like_cross_modal_eligibility_reference",
            "eligibility_traces": [
                {"source": pair[0], "target": pair[1], "value": value}
                for pair, value in sorted(self.eligibility_traces.items())
            ],
            "synaptic_efficacy": [
                {"source": pair[0], "target": pair[1], "value": value}
                for pair, value in sorted(self.synaptic_efficacy.items())
            ],
            "update_count": self.update_count,
            "model_scope": "not fitted to human neural recordings or direct synaptic measurements",
        }


@dataclass(slots=True)
class Point08Result:
    """Point 08's exact 15-step trace alongside the underlying validated 21-stage result."""

    sequence_id: str
    output: dict[str, Any]
    stage_trace: list[dict[str, Any]]
    hub_spoke_result: dict[str, Any]
    cross_modal_recall: dict[str, Any]
    temporal_associations: dict[str, Any]
    biological_plasticity: dict[str, Any]
    diagnostics: dict[str, Any]
    logger: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the source-aligned trace, hub/spoke details, associations, and diagnostics."""
        return {
            "schema": "ore.point08.distributed_semantics.v1",
            "sequence_id": self.sequence_id,
            "output": self.output,
            "stage_trace": self.stage_trace,
            "hub_spoke_result": self.hub_spoke_result,
            "cross_modal_recall": self.cross_modal_recall,
            "temporal_associations": self.temporal_associations,
            "biological_plasticity": self.biological_plasticity,
            "diagnostics": self.diagnostics,
            "logger": self.logger,
        }


class Point08DistributedSemanticProcessor:
    """Execute Point 08's 15 semantic operations over ORE's existing hub-and-spoke model."""

    def __init__(
        self,
        concepts: Sequence[Any],
        hub_config: HubAndSpokeConfig | None = None,
        point08_config: Point08Config | None = None,
        associations: Mapping[str, Mapping[str, float]] | None = None,
        database_path: str | Path | None = None,
        repository_root: str | Path | None = None,
    ) -> None:
        """Initialize the existing ATL/spoke core plus temporal and biological plasticity memories."""
        self.point08_config = point08_config or Point08Config()
        self.core = HubAndSpokeProcessor(concepts, config=hub_config, associations=associations)
        self.temporal_memory = TemporalAssociationMemory(self.point08_config)
        self.biological_plasticity = BiologicalEligibilityPlasticity(self.point08_config)
        self.database_path = Path(database_path).expanduser().resolve() if database_path else None
        self.repository_root = (
            Path(repository_root).expanduser().resolve()
            if repository_root else Path(__file__).resolve().parents[1]
        )
        self.process_count = 0

    def _merge_recalled_inputs(
        self, request: HubAndSpokeRequest, automatic: Mapping[str, Mapping[str, float]]
    ) -> dict[str, dict[str, float]]:
        """Combine caller-supplied and automatically recalled features without changing direct inputs."""
        merged: dict[str, dict[str, float]] = {}
        for source in (request.recalled_modality_inputs, automatic):
            for raw_modality, features in source.items():
                modality = canonical_modality(raw_modality)
                target = merged.setdefault(modality, {})
                for feature, value in features.items():
                    target[str(feature)] = max(target.get(str(feature), 0.0), _clip(value))
        return merged

    def _stage(
        self,
        index: int,
        functional_output: Mapping[str, Any],
        biological_output: Mapping[str, Any],
        status: str = "completed",
    ) -> dict[str, Any]:
        """Build a source-aligned operation trace with mathematical and biological annotations."""
        spec = POINT08_STAGE_REGISTRY[index]
        return {
            "step": index + 1,
            "stage_id": spec.stage_id,
            "name": spec.name,
            "status": status,
            "explanation": spec.objective,
            "mathematical_form": spec.mathematical_form,
            "biological_reference": spec.biological_reference,
            "evidence_level": spec.evidence_level,
            "functional_output": dict(functional_output),
            "biological_output": dict(biological_output),
        }

    def _stage_biology(
        self, base_result: HubAndSpokeResult, stage_ids: Sequence[str]
    ) -> dict[str, Any]:
        """Collect corresponding LIF events while retaining the source-to-core stage mapping."""
        selected = [
            row.get("biological_event")
            for row in base_result.stage_trace
            if row.get("stage_id") in set(stage_ids) and row.get("biological_event") is not None
        ]
        return {
            "underlying_hub_spoke_stages": list(stage_ids),
            "lif_events": selected,
            "lif_enabled": bool(base_result.biological.get("enabled")),
        }

    def _build_source_trace(
        self,
        request: HubAndSpokeRequest,
        base: HubAndSpokeResult,
        preview: Mapping[str, Any],
        temporal_update: Mapping[str, Any],
        plasticity: Mapping[str, Any],
        recall: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        """Map the source's fifteen steps to the underlying functional and biological operations."""
        stages = base.stage_outputs
        rows: list[dict[str, Any]] = []
        outputs: list[tuple[Mapping[str, Any], Mapping[str, Any], str]] = [
            (stages["HS-01"], self._stage_biology(base, ("HS-01",)), "completed"),
            (stages["HS-02"], self._stage_biology(base, ("HS-02",)), "completed"),
            (stages["HS-03"], self._stage_biology(base, ("HS-03",)), "completed"),
            (stages["HS-06"], self._stage_biology(base, ("HS-06",)), "completed"),
            (
                {"temporal_eligibility_preview": dict(preview),
                 "committed_update": dict(temporal_update)},
                {"eligibility_and_timing": dict(plasticity),
                 "networks": ["modality_spokes", "local_association_populations"]},
                "completed" if preview.get("eligible_event_count", 0) else "completed_without_timed_pairs",
            ),
            (
                {"spoke_to_hub_mapping": stages["HS-16"],
                 "cross_modal_alignment": stages["HS-17"],
                 "cross_modal_temporal_links": dict(temporal_update)},
                {"plasticity": dict(plasticity),
                 "biological_model_scope": "bounded Hebbian/STDP-like model, not a synaptic measurement"},
                str(stages["HS-18"].get("status", "preview_only")),
            ),
            (
                {"spoke_transmission": stages["HS-05"],
                 "anterior_temporal_integration": stages["HS-07"]},
                self._stage_biology(base, ("HS-05", "HS-07")),
                "completed",
            ),
            (stages["HS-08"], self._stage_biology(base, ("HS-08",)), "completed"),
            (stages["HS-09"], self._stage_biology(base, ("HS-09",)), "completed"),
            (
                {"memory_retrieval": stages["HS-04"],
                 "candidate_identification": stages["HS-11"],
                 "cross_modal_feature_recall": dict(recall)},
                self._stage_biology(base, ("HS-04", "HS-11")),
                "completed",
            ),
            (stages["HS-12"], self._stage_biology(base, ("HS-12",)), "completed"),
            (stages["HS-13"], self._stage_biology(base, ("HS-13",)), "completed"),
            (
                {"task": request.task,
                 "context_features": dict(request.context_features),
                 "task_relevance": dict(request.task_relevance),
                 "controlled_selection": stages["HS-13"],
                 "contextual_candidate_scores": stages["HS-11"].get("ranked_candidates", [])},
                self._stage_biology(base, ("HS-11", "HS-13")),
                "completed",
            ),
            (stages["HS-14"], self._stage_biology(base, ("HS-14",)), "completed"),
            (
                {"experience_acquisition": stages["HS-15"],
                 "mapping_learning": stages["HS-16"],
                 "modality_alignment": stages["HS-17"],
                 "knowledge_update": stages["HS-18"],
                 "temporal_cross_modal_update": dict(temporal_update)},
                {"lif_learning_stages": self._stage_biology(base, ("HS-15", "HS-16", "HS-17", "HS-18")),
                 "plasticity": dict(plasticity)},
                str(stages["HS-18"].get("status", "preview_only")),
            ),
        ]
        for index, (functional, biological, status) in enumerate(outputs):
            rows.append(self._stage(index, functional, biological, status))
        return rows

    def process(
        self,
        request: HubAndSpokeRequest,
        learning_experiences: Sequence[LearningExperience] = (),
        learning_authorized: bool = False,
    ) -> Point08Result:
        """Run fifteen source stages with separate direct/recalled inputs and explicit learning authorization."""
        started = time.perf_counter()
        experiences = tuple(learning_experiences)
        automatic_recall, recall_details = self.temporal_memory.spread_activation(
            request.modality_inputs
        )
        recalled_inputs = self._merge_recalled_inputs(request, automatic_recall)
        enriched_request = replace(request, recalled_modality_inputs=recalled_inputs)
        preview = self.temporal_memory.preview(experiences)
        base_result = self.core.process(
            enriched_request,
            learning_experiences=experiences,
            learning_authorized=learning_authorized,
        )
        if learning_authorized:
            temporal_update = self.temporal_memory.learn(experiences)
            biological_plasticity = self.biological_plasticity.learn(preview)
        else:
            temporal_update = {
                **preview,
                "status": "preview_only_authorization_required" if experiences else "skipped_no_experiences",
                "updated_pair_count": 0,
                "memory_pair_count": len(self.temporal_memory.weights),
                "weights_mutated": False,
            }
            biological_plasticity = self.biological_plasticity.preview(preview)
        trace = self._build_source_trace(
            request, base_result, preview, temporal_update, biological_plasticity,
            {**recall_details, "features": automatic_recall,
             "caller_supplied_recalled_modalities": sorted(request.recalled_modality_inputs)},
        )
        self.process_count += 1
        diagnostics = {
            "valid": bool(
                base_result.diagnostics.get("valid")
                and len(trace) == 15
                and [row["stage_id"] for row in trace]
                    == [f"P08-{index:02d}" for index in range(1, 16)]
                and all(row["mathematical_form"] and row["biological_reference"] for row in trace)
                and all(math.isfinite(float(value)) for value in self.temporal_memory.weights.values())
            ),
            "source_stage_count": len(trace),
            "expected_source_stage_count": 15,
            "underlying_hub_spoke_stage_count": len(base_result.stage_trace),
            "underlying_operation_count": len(base_result.operation_trace),
            "temporal_pair_count": len(self.temporal_memory.weights),
            "learning_authorized": bool(learning_authorized),
            "temporal_learning_mutated": bool(temporal_update.get("weights_mutated")),
            "base_learning_status": base_result.learning.get("stage_18", {}).get("status"),
            "scientific_scope": (
                "Functional and generic neural-dynamics models are computational hypotheses, "
                "not literal neural instructions or fitted human measurements."
            ),
        }
        result = Point08Result(
            sequence_id=request.sequence_id,
            output=base_result.output,
            stage_trace=trace,
            hub_spoke_result=base_result.to_dict(),
            cross_modal_recall={
                **recall_details,
                "automatically_recalled_features": automatic_recall,
                "passed_to_hub_spoke_core": recalled_inputs,
                "direct_evidence_unchanged": dict(request.modality_inputs),
            },
            temporal_associations=self.temporal_memory.snapshot(),
            biological_plasticity=self.biological_plasticity.snapshot(),
            diagnostics=diagnostics,
        )
        if self.database_path is not None:
            result.logger = record_point08_result(
                result, self.repository_root, self.database_path,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
            )
        return result

    def snapshot(self) -> dict[str, Any]:
        """Return the integrated hub-and-spoke generation and separate temporal/plasticity memories."""
        return {
            "process_count": self.process_count,
            "model_generation": self.core.model_generation,
            "temporal_associations": self.temporal_memory.snapshot(),
            "biological_plasticity": self.biological_plasticity.snapshot(),
        }


def index_point08_annotations(
    repository_root: str | Path,
    database_path: str | Path,
) -> dict[str, Any]:
    """Index Point 08 hub-and-spoke classes and functions in the shared ORE SQLite catalogue."""
    from ore_logger.catalog import sync_repository
    from ore_logger.database import LoggerDatabase

    root = Path(repository_root).expanduser().resolve()
    database = LoggerDatabase(database_path)
    report = sync_repository(database, root, include_paths={"ore_hub_spoke"})
    with database.connect() as connection:
        rows = connection.execute(
            "SELECT COUNT(*) FROM annotations WHERE source_path LIKE 'ore_hub_spoke/%'"
        ).fetchone()
    return {**report, "database_path": str(database.path), "point08_symbol_count": int(rows[0])}


def record_point08_result(
    result: Point08Result,
    repository_root: str | Path,
    database_path: str | Path,
    elapsed_ms: float | None = None,
) -> dict[str, Any]:
    """Persist all fifteen Point 08 operations and their annotations in the existing ORE SQLite logger."""
    from ore_logger.recorder import SQLiteRunRecorder

    root = Path(repository_root).expanduser().resolve()
    run_id = uuid.uuid4().hex
    started_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    recorder = SQLiteRunRecorder(
        database_path=database_path,
        repository_root=root,
        run_id=run_id,
        component="ore_hub_spoke.point08",
        callable_name="Point08DistributedSemanticProcessor.process",
        started_at_utc=started_at,
        entry_kind="point08_distributed_semantics",
    )
    for row in result.stage_trace:
        recorder.log_operation({
            "step": row["stage_id"],
            "operation": f"{row['stage_id']} {row['name']}",
            "phase": row["stage_id"],
            "status": row["status"],
            "explanation": (
                f"{row['explanation']} Mathematical form: {row['mathematical_form']} "
                f"Biological reference: {row['biological_reference']} "
                f"Evidence: {row['evidence_level']}"
            ),
            "functional_output": row["functional_output"],
            "biological_output": row["biological_output"],
        })
    duration = float(elapsed_ms) if elapsed_ms is not None else 0.0
    metadata = recorder.finish(
        "completed" if result.diagnostics.get("valid") else "completed_with_warnings",
        duration,
        output_type="Point08Result",
    )
    metadata.update({
        "source_stage_count": len(result.stage_trace),
        "underlying_hub_spoke_stage_count": result.diagnostics.get("underlying_hub_spoke_stage_count"),
        "temporal_pair_count": result.diagnostics.get("temporal_pair_count"),
        "diagnostics_valid": bool(result.diagnostics.get("valid")),
    })
    return metadata
